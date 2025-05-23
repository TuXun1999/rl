# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
from __future__ import annotations

import argparse
import functools
import os

import pytest
import torch

import torchrl.modules
from tensordict import LazyStackedTensorDict, pad, TensorDict, unravel_key_list
from tensordict.nn import InteractionType, TensorDictModule, TensorDictSequential
from tensordict.utils import assert_close
from torch import nn
from torchrl.data.tensor_specs import Bounded, Composite, Unbounded
from torchrl.envs import (
    CatFrames,
    Compose,
    EnvCreator,
    InitTracker,
    SerialEnv,
    TensorDictPrimer,
    TransformedEnv,
)
from torchrl.envs.utils import set_exploration_type, step_mdp
from torchrl.modules import (
    AdditiveGaussianModule,
    DecisionTransformerInferenceWrapper,
    DTActor,
    GRUModule,
    LSTMModule,
    MLP,
    MultiStepActorWrapper,
    NormalParamExtractor,
    OnlineDTActor,
    ProbabilisticActor,
    SafeModule,
    set_recurrent_mode,
    TanhDelta,
    TanhNormal,
    ValueOperator,
)
from torchrl.modules.models.decision_transformer import _has_transformers
from torchrl.modules.tensordict_module.common import (
    ensure_tensordict_compatible,
    is_tensordict_compatible,
    VmapModule,
)
from torchrl.modules.tensordict_module.probabilistic import (
    SafeProbabilisticModule,
    SafeProbabilisticTensorDictSequential,
)
from torchrl.modules.tensordict_module.sequence import SafeSequential
from torchrl.modules.utils import get_primers_from_module
from torchrl.objectives import DDPGLoss

if os.getenv("PYTORCH_TEST_FBCODE"):
    from pytorch.rl.test.mocking_classes import CountingEnv, DiscreteActionVecMockEnv
else:
    from mocking_classes import CountingEnv, DiscreteActionVecMockEnv

_has_functorch = False
try:
    try:
        from torch import vmap
    except ImportError:
        from functorch import vmap

    _has_functorch = True
except ImportError:
    pass


class TestTDModule:
    def test_multiple_output(self):
        class MultiHeadLinear(nn.Module):
            def __init__(self, in_1, out_1, out_2, out_3):
                super().__init__()
                self.linear_1 = nn.Linear(in_1, out_1)
                self.linear_2 = nn.Linear(in_1, out_2)
                self.linear_3 = nn.Linear(in_1, out_3)

            def forward(self, x):
                return self.linear_1(x), self.linear_2(x), self.linear_3(x)

        tensordict_module = SafeModule(
            MultiHeadLinear(5, 4, 3, 2),
            in_keys=["input"],
            out_keys=["out_1", "out_2", "out_3"],
        )
        td = TensorDict({"input": torch.randn(3, 5)}, batch_size=[3])
        td = tensordict_module(td)
        assert td.shape == torch.Size([3])
        assert "input" in td.keys()
        assert "out_1" in td.keys()
        assert "out_2" in td.keys()
        assert "out_3" in td.keys()
        assert td.get("out_3").shape == torch.Size([3, 2])

        # Using "_" key to ignore some output
        tensordict_module = SafeModule(
            MultiHeadLinear(5, 4, 3, 2),
            in_keys=["input"],
            out_keys=["_", "_", "out_3"],
        )
        td = TensorDict({"input": torch.randn(3, 5)}, batch_size=[3])
        td = tensordict_module(td)
        assert td.shape == torch.Size([3])
        assert "input" in td.keys()
        assert "out_3" in td.keys()
        assert "_" not in td.keys()
        assert td.get("out_3").shape == torch.Size([3, 2])

    def test_spec_key_warning(self):
        class MultiHeadLinear(nn.Module):
            def __init__(self, in_1, out_1, out_2):
                super().__init__()
                self.linear_1 = nn.Linear(in_1, out_1)
                self.linear_2 = nn.Linear(in_1, out_2)

            def forward(self, x):
                return self.linear_1(x), self.linear_2(x)

        spec_dict = {
            "_": Unbounded((4,)),
            "out_2": Unbounded((3,)),
        }

        # warning due to "_" in spec keys
        with pytest.warns(UserWarning, match='got a spec with key "_"'):
            tensordict_module = SafeModule(
                MultiHeadLinear(5, 4, 3),
                in_keys=["input"],
                out_keys=["_", "out_2"],
                spec=Composite(**spec_dict),
            )

    @pytest.mark.parametrize("safe", [True, False])
    @pytest.mark.parametrize("spec_type", [None, "bounded", "unbounded"])
    @pytest.mark.parametrize("lazy", [True, False])
    def test_stateful(self, safe, spec_type, lazy):
        torch.manual_seed(0)
        param_multiplier = 1
        if lazy:
            net = nn.LazyLinear(4 * param_multiplier)
        else:
            net = nn.Linear(3, 4 * param_multiplier)

        if spec_type is None:
            spec = None
        elif spec_type == "bounded":
            spec = Bounded(-0.1, 0.1, 4)
        elif spec_type == "unbounded":
            spec = Unbounded(4)

        if safe and spec is None:
            with pytest.raises(
                RuntimeError,
                match="is not a valid configuration as the tensor specs are not "
                "specified",
            ):
                tensordict_module = SafeModule(
                    module=net,
                    spec=spec,
                    in_keys=["in"],
                    out_keys=["out"],
                    safe=safe,
                )
            return
        else:
            tensordict_module = SafeModule(
                module=net,
                spec=spec,
                in_keys=["in"],
                out_keys=["out"],
                safe=safe,
            )

        td = TensorDict({"in": torch.randn(3, 3)}, [3])
        tensordict_module(td)
        assert td.shape == torch.Size([3])
        assert td.get("out").shape == torch.Size([3, 4])

        # test bounds
        if not safe and spec_type == "bounded":
            assert ((td.get("out") > 0.1) | (td.get("out") < -0.1)).any(), td.get("out")
        elif safe and spec_type == "bounded":
            assert ((td.get("out") < 0.1) | (td.get("out") > -0.1)).all()

    @pytest.mark.parametrize("safe", [True, False])
    @pytest.mark.parametrize("spec_type", [None, "bounded", "unbounded"])
    @pytest.mark.parametrize("out_keys", [["loc", "scale"], ["loc_1", "scale_1"]])
    @pytest.mark.parametrize("lazy", [True, False])
    @pytest.mark.parametrize(
        "exp_mode", [InteractionType.DETERMINISTIC, InteractionType.RANDOM, None]
    )
    def test_stateful_probabilistic(self, safe, spec_type, lazy, exp_mode, out_keys):
        torch.manual_seed(0)
        param_multiplier = 2
        if lazy:
            net = nn.LazyLinear(4 * param_multiplier)
        else:
            net = nn.Linear(3, 4 * param_multiplier)

        in_keys = ["in"]
        net = SafeModule(
            module=nn.Sequential(net, NormalParamExtractor()),
            spec=None,
            in_keys=in_keys,
            out_keys=out_keys,
        )

        if spec_type is None:
            spec = None
        elif spec_type == "bounded":
            spec = Bounded(-0.1, 0.1, 4)
        elif spec_type == "unbounded":
            spec = Unbounded(4)
        else:
            raise NotImplementedError

        kwargs = {"distribution_class": TanhNormal}
        if out_keys == ["loc", "scale"]:
            dist_in_keys = ["loc", "scale"]
        elif out_keys == ["loc_1", "scale_1"]:
            dist_in_keys = {"loc": "loc_1", "scale": "scale_1"}
        else:
            raise NotImplementedError

        if safe and spec is None:
            with pytest.raises(
                RuntimeError,
                match="is not a valid configuration as the tensor specs are not "
                "specified",
            ):
                prob_module = SafeProbabilisticModule(
                    in_keys=dist_in_keys,
                    out_keys=["out"],
                    spec=spec,
                    safe=safe,
                    **kwargs,
                )
            return
        else:
            prob_module = SafeProbabilisticModule(
                in_keys=dist_in_keys,
                out_keys=["out"],
                spec=spec,
                safe=safe,
                **kwargs,
            )

        tensordict_module = SafeProbabilisticTensorDictSequential(net, prob_module)
        td = TensorDict({"in": torch.randn(3, 3)}, [3])
        with set_exploration_type(exp_mode):
            tensordict_module(td)
        assert td.shape == torch.Size([3])
        assert td.get("out").shape == torch.Size([3, 4])

        # test bounds
        if not safe and spec_type == "bounded":
            assert ((td.get("out") > 0.1) | (td.get("out") < -0.1)).any()
        elif safe and spec_type == "bounded":
            assert ((td.get("out") < 0.1) | (td.get("out") > -0.1)).all()


class TestTDSequence:
    # Temporarily disabling this test until 473 is merged in tensordict
    # def test_in_key_warning(self):
    #     with pytest.warns(UserWarning, match='key "_" is for ignoring output'):
    #         tensordict_module = SafeModule(
    #             nn.Linear(3, 4), in_keys=["_"], out_keys=["out1"]
    #         )
    #     with pytest.warns(UserWarning, match='key "_" is for ignoring output'):
    #         tensordict_module = SafeModule(
    #             nn.Linear(3, 4), in_keys=["_", "key2"], out_keys=["out1"]
    #         )

    @pytest.mark.parametrize("safe", [True, False])
    @pytest.mark.parametrize("spec_type", [None, "bounded", "unbounded"])
    @pytest.mark.parametrize("lazy", [True, False])
    def test_stateful(self, safe, spec_type, lazy):
        torch.manual_seed(0)
        param_multiplier = 1
        if lazy:
            net1 = nn.LazyLinear(4)
            dummy_net = nn.LazyLinear(4)
            net2 = nn.LazyLinear(4 * param_multiplier)
        else:
            net1 = nn.Linear(3, 4)
            dummy_net = nn.Linear(4, 4)
            net2 = nn.Linear(4, 4 * param_multiplier)

        if spec_type is None:
            spec = None
        elif spec_type == "bounded":
            spec = Bounded(-0.1, 0.1, 4)
        elif spec_type == "unbounded":
            spec = Unbounded(4)

        kwargs = {}

        if safe and spec is None:
            pytest.skip("safe and spec is None is checked elsewhere")
        else:
            tdmodule1 = SafeModule(
                net1,
                spec=None,
                in_keys=["in"],
                out_keys=["hidden"],
                safe=False,
            )
            dummy_tdmodule = SafeModule(
                dummy_net,
                spec=None,
                in_keys=["hidden"],
                out_keys=["hidden"],
                safe=False,
            )
            tdmodule2 = SafeModule(
                spec=spec,
                module=net2,
                in_keys=["hidden"],
                out_keys=["out"],
                safe=False,
                **kwargs,
            )
            tdmodule = SafeSequential(tdmodule1, dummy_tdmodule, tdmodule2)

        assert hasattr(tdmodule, "__setitem__")
        assert len(tdmodule) == 3
        tdmodule[1] = tdmodule2
        assert len(tdmodule) == 3

        assert hasattr(tdmodule, "__delitem__")
        assert len(tdmodule) == 3
        del tdmodule[2]
        assert len(tdmodule) == 2

        assert hasattr(tdmodule, "__getitem__")
        assert tdmodule[0] is tdmodule1
        assert tdmodule[1] is tdmodule2

        td = TensorDict({"in": torch.randn(3, 3)}, [3])
        tdmodule(td)
        assert td.shape == torch.Size([3])
        assert td.get("out").shape == torch.Size([3, 4])

        # test bounds
        if not safe and spec_type == "bounded":
            assert ((td.get("out") > 0.1) | (td.get("out") < -0.1)).any()
        elif safe and spec_type == "bounded":
            assert ((td.get("out") < 0.1) | (td.get("out") > -0.1)).all()

    @pytest.mark.parametrize("safe", [True, False])
    @pytest.mark.parametrize("spec_type", [None, "bounded", "unbounded"])
    @pytest.mark.parametrize("lazy", [True, False])
    def test_stateful_probabilistic(self, safe, spec_type, lazy):
        torch.manual_seed(0)
        param_multiplier = 2
        if lazy:
            net1 = nn.LazyLinear(4)
            dummy_net = nn.LazyLinear(4)
            net2 = nn.LazyLinear(4 * param_multiplier)
        else:
            net1 = nn.Linear(3, 4)
            dummy_net = nn.Linear(4, 4)
            net2 = nn.Linear(4, 4 * param_multiplier)
        net2 = nn.Sequential(net2, NormalParamExtractor())

        if spec_type is None:
            spec = None
        elif spec_type == "bounded":
            spec = Bounded(-0.1, 0.1, 4)
        elif spec_type == "unbounded":
            spec = Unbounded(4)
        else:
            raise NotImplementedError

        kwargs = {"distribution_class": TanhNormal}

        if safe and spec is None:
            pytest.skip("safe and spec is None is checked elsewhere")
        else:
            tdmodule1 = SafeModule(
                net1,
                in_keys=["in"],
                out_keys=["hidden"],
                spec=None,
                safe=False,
            )
            dummy_tdmodule = SafeModule(
                dummy_net,
                in_keys=["hidden"],
                out_keys=["hidden"],
                spec=None,
                safe=False,
            )
            tdmodule2 = SafeModule(
                module=net2,
                in_keys=["hidden"],
                out_keys=["loc", "scale"],
                spec=None,
                safe=False,
            )

            prob_module = SafeProbabilisticModule(
                spec=spec,
                in_keys=["loc", "scale"],
                out_keys=["out"],
                safe=False,
                **kwargs,
            )
            tdmodule = SafeProbabilisticTensorDictSequential(
                tdmodule1, dummy_tdmodule, tdmodule2, prob_module
            )

        assert hasattr(tdmodule, "__setitem__")
        assert len(tdmodule) == 4
        tdmodule[1] = tdmodule2
        tdmodule[2] = prob_module
        assert len(tdmodule) == 4

        assert hasattr(tdmodule, "__delitem__")
        assert len(tdmodule) == 4
        del tdmodule[3]
        assert len(tdmodule) == 3

        assert hasattr(tdmodule, "__getitem__")
        assert tdmodule[0] is tdmodule1
        assert tdmodule[1] is tdmodule2
        assert tdmodule[2] is prob_module

        td = TensorDict({"in": torch.randn(3, 3)}, [3])
        tdmodule(td)
        assert td.shape == torch.Size([3])
        assert td.get("out").shape == torch.Size([3, 4])

        dist = tdmodule.get_dist(td)
        assert dist.rsample().shape[: td.ndimension()] == td.shape

        # test bounds
        if not safe and spec_type == "bounded":
            assert ((td.get("out") > 0.1) | (td.get("out") < -0.1)).any()
        elif safe and spec_type == "bounded":
            assert ((td.get("out") < 0.1) | (td.get("out") > -0.1)).all()

    def test_submodule_sequence(self):
        td_module_1 = SafeModule(
            nn.Linear(3, 2),
            in_keys=["in"],
            out_keys=["hidden"],
        )
        td_module_2 = SafeModule(
            nn.Linear(2, 4),
            in_keys=["hidden"],
            out_keys=["out"],
        )
        td_module = SafeSequential(td_module_1, td_module_2)

        td_1 = TensorDict({"in": torch.randn(5, 3)}, [5])
        sub_seq_1 = td_module.select_subsequence(out_keys=["hidden"])
        sub_seq_1(td_1)
        assert "hidden" in td_1.keys()
        assert "out" not in td_1.keys()
        td_2 = TensorDict({"hidden": torch.randn(5, 2)}, [5])
        sub_seq_2 = td_module.select_subsequence(in_keys=["hidden"])
        sub_seq_2(td_2)
        assert "out" in td_2.keys()
        assert td_2.get("out").shape == torch.Size([5, 4])

    @pytest.mark.parametrize("stack", [True, False])
    def test_sequential_partial(self, stack):
        torch.manual_seed(0)
        param_multiplier = 2

        net1 = nn.Linear(3, 4)

        net2 = nn.Linear(4, 4 * param_multiplier)
        net2 = nn.Sequential(net2, NormalParamExtractor())
        net2 = SafeModule(net2, in_keys=["b"], out_keys=["loc", "scale"])

        net3 = nn.Linear(4, 4 * param_multiplier)
        net3 = nn.Sequential(net3, NormalParamExtractor())
        net3 = SafeModule(net3, in_keys=["c"], out_keys=["loc", "scale"])

        spec = Bounded(-0.1, 0.1, 4)

        kwargs = {"distribution_class": TanhNormal}

        tdmodule1 = SafeModule(
            net1,
            in_keys=["a"],
            out_keys=["hidden"],
            spec=None,
            safe=False,
        )
        tdmodule2 = SafeProbabilisticTensorDictSequential(
            net2,
            SafeProbabilisticModule(
                in_keys=["loc", "scale"],
                out_keys=["out"],
                spec=spec,
                safe=True,
                **kwargs,
            ),
        )
        tdmodule3 = SafeProbabilisticTensorDictSequential(
            net3,
            SafeProbabilisticModule(
                in_keys=["loc", "scale"],
                out_keys=["out"],
                spec=spec,
                safe=True,
                **kwargs,
            ),
        )
        tdmodule = SafeSequential(
            tdmodule1, tdmodule2, tdmodule3, partial_tolerant=True
        )

        if stack:
            td = LazyStackedTensorDict.maybe_dense_stack(
                [
                    TensorDict({"a": torch.randn(3), "b": torch.randn(4)}, []),
                    TensorDict({"a": torch.randn(3), "c": torch.randn(4)}, []),
                ],
                0,
            )
            tdmodule(td)
            assert "loc" in td.keys()
            assert "scale" in td.keys()
            assert "out" in td.keys()
            assert td["out"].shape[0] == 2
            assert td["loc"].shape[0] == 2
            assert td["scale"].shape[0] == 2
            assert "b" not in td.keys()
            assert "b" in td[0].keys()
        else:
            td = TensorDict({"a": torch.randn(3), "b": torch.randn(4)}, [])
            tdmodule(td)
            assert "loc" in td.keys()
            assert "scale" in td.keys()
            assert "out" in td.keys()
            assert "b" in td.keys()


def test_is_tensordict_compatible():
    class MultiHeadLinear(nn.Module):
        def __init__(self, in_1, out_1, out_2, out_3):
            super().__init__()
            self.linear_1 = nn.Linear(in_1, out_1)
            self.linear_2 = nn.Linear(in_1, out_2)
            self.linear_3 = nn.Linear(in_1, out_3)

        def forward(self, x):
            return self.linear_1(x), self.linear_2(x), self.linear_3(x)

    td_module = SafeModule(
        MultiHeadLinear(5, 4, 3, 2),
        in_keys=["in_1", "in_2"],
        out_keys=["out_1", "out_2"],
    )
    assert is_tensordict_compatible(td_module)

    class MockCompatibleModule(nn.Module):
        def __init__(self, in_keys, out_keys):
            self.in_keys = in_keys
            self.out_keys = out_keys

        def forward(self, tensordict):
            pass

    compatible_nn_module = MockCompatibleModule(
        in_keys=["in_1", "in_2"],
        out_keys=["out_1", "out_2"],
    )
    assert is_tensordict_compatible(compatible_nn_module)

    class MockIncompatibleModuleNoKeys(nn.Module):
        def forward(self, input):
            pass

    incompatible_nn_module_no_keys = MockIncompatibleModuleNoKeys()
    assert not is_tensordict_compatible(incompatible_nn_module_no_keys)

    class MockIncompatibleModuleMultipleArgs(nn.Module):
        def __init__(self, in_keys, out_keys):
            self.in_keys = in_keys
            self.out_keys = out_keys

        def forward(self, input_1, input_2):
            pass

    incompatible_nn_module_multi_args = MockIncompatibleModuleMultipleArgs(
        in_keys=["in_1", "in_2"],
        out_keys=["out_1", "out_2"],
    )
    with pytest.raises(TypeError):
        is_tensordict_compatible(incompatible_nn_module_multi_args)


def test_ensure_tensordict_compatible():
    class MultiHeadLinear(nn.Module):
        def __init__(self, in_1, out_1, out_2, out_3):
            super().__init__()
            self.linear_1 = nn.Linear(in_1, out_1)
            self.linear_2 = nn.Linear(in_1, out_2)
            self.linear_3 = nn.Linear(in_1, out_3)

        def forward(self, x):
            return self.linear_1(x), self.linear_2(x), self.linear_3(x)

    td_module = SafeModule(
        MultiHeadLinear(5, 4, 3, 2),
        in_keys=["in_1", "in_2"],
        out_keys=["out_1", "out_2"],
    )
    ensured_module = ensure_tensordict_compatible(td_module)
    assert ensured_module is td_module
    with pytest.raises(TypeError):
        ensure_tensordict_compatible(td_module, in_keys=["input"])
    with pytest.raises(TypeError):
        ensure_tensordict_compatible(td_module, out_keys=["output"])

    class NonNNModule:
        def __init__(self):
            pass

        def forward(self, x):
            pass

    non_nn_module = NonNNModule()
    with pytest.raises(TypeError):
        ensure_tensordict_compatible(non_nn_module)

    class ErrorNNModule(nn.Module):
        def forward(self, in_1, in_2):
            pass

    error_nn_module = ErrorNNModule()
    with pytest.raises(TypeError):
        ensure_tensordict_compatible(error_nn_module, in_keys=["input"])

    nn_module = MultiHeadLinear(5, 4, 3, 2)
    ensured_module = ensure_tensordict_compatible(
        nn_module,
        in_keys=["x"],
        out_keys=["out_1", "out_2", "out_3"],
    )
    assert set(unravel_key_list(ensured_module.in_keys)) == {"x"}
    assert isinstance(ensured_module, TensorDictModule)


class TestLSTMModule:
    def test_errs(self):
        with pytest.raises(ValueError, match="batch_first"):
            lstm_module = LSTMModule(
                input_size=3,
                hidden_size=12,
                batch_first=False,
                in_keys=["observation", "hidden0", "hidden1"],
                out_keys=["intermediate", ("next", "hidden0"), ("next", "hidden1")],
            )
        with pytest.raises(ValueError, match="in_keys"):
            lstm_module = LSTMModule(
                input_size=3,
                hidden_size=12,
                batch_first=True,
                in_keys=[
                    "observation",
                    "hidden0",
                ],
                out_keys=["intermediate", ("next", "hidden0"), ("next", "hidden1")],
            )
        with pytest.raises(TypeError, match="incompatible function arguments"):
            lstm_module = LSTMModule(
                input_size=3,
                hidden_size=12,
                batch_first=True,
                in_keys="abc",
                out_keys=["intermediate", ("next", "hidden0"), ("next", "hidden1")],
            )
        with pytest.raises(ValueError, match="in_keys"):
            lstm_module = LSTMModule(
                input_size=3,
                hidden_size=12,
                batch_first=True,
                in_key="smth",
                in_keys=[
                    "observation",
                    "hidden0",
                ],
                out_keys=["intermediate", ("next", "hidden0"), ("next", "hidden1")],
            )
        with pytest.raises(ValueError, match="out_keys"):
            lstm_module = LSTMModule(
                input_size=3,
                hidden_size=12,
                batch_first=True,
                in_keys=["observation", "hidden0", "hidden1"],
                out_keys=["intermediate", ("next", "hidden0")],
            )
        with pytest.raises(TypeError, match="incompatible function arguments"):
            lstm_module = LSTMModule(
                input_size=3,
                hidden_size=12,
                batch_first=True,
                in_keys=["observation", "hidden0", "hidden1"],
                out_keys="abc",
            )
        with pytest.raises(ValueError, match="out_keys"):
            lstm_module = LSTMModule(
                input_size=3,
                hidden_size=12,
                batch_first=True,
                in_keys=["observation", "hidden0", "hidden1"],
                out_key="smth",
                out_keys=["intermediate", ("next", "hidden0")],
            )
        lstm_module = LSTMModule(
            input_size=3,
            hidden_size=12,
            batch_first=True,
            in_keys=["observation", "hidden0", "hidden1"],
            out_keys=["intermediate", ("next", "hidden0"), ("next", "hidden1")],
        )
        td = TensorDict({"observation": torch.randn(3)}, [])
        with pytest.raises(KeyError, match="is_init"):
            lstm_module(td)

    @pytest.mark.parametrize("default_val", [False, True, None])
    def test_set_recurrent_mode(self, default_val):
        lstm_module = LSTMModule(
            input_size=3,
            hidden_size=12,
            batch_first=True,
            in_keys=["observation", "hidden0", "hidden1"],
            out_keys=["intermediate", ("next", "hidden0"), ("next", "hidden1")],
            default_recurrent_mode=default_val,
        )
        assert lstm_module.recurrent_mode is bool(default_val)
        with set_recurrent_mode(True):
            assert lstm_module.recurrent_mode
            with set_recurrent_mode(False):
                assert not lstm_module.recurrent_mode
                with set_recurrent_mode("recurrent"):
                    assert lstm_module.recurrent_mode
                    with set_recurrent_mode("sequential"):
                        assert not lstm_module.recurrent_mode
                    assert lstm_module.recurrent_mode
                assert not lstm_module.recurrent_mode
            assert lstm_module.recurrent_mode
        assert lstm_module.recurrent_mode is bool(default_val)

    @pytest.mark.filterwarnings("ignore::DeprecationWarning")
    def test_set_temporal_mode(self):
        lstm_module = LSTMModule(
            input_size=3,
            hidden_size=12,
            batch_first=True,
            in_keys=["observation", "hidden0", "hidden1"],
            out_keys=["intermediate", ("next", "hidden0"), ("next", "hidden1")],
        )
        assert lstm_module.set_recurrent_mode(False) is lstm_module
        assert not lstm_module.set_recurrent_mode(False).recurrent_mode
        assert lstm_module.set_recurrent_mode(True) is not lstm_module
        assert lstm_module.set_recurrent_mode(True).recurrent_mode
        assert set(lstm_module.set_recurrent_mode(True).parameters()) == set(
            lstm_module.parameters()
        )

    def test_python_cudnn(self):
        lstm_module = LSTMModule(
            input_size=3,
            hidden_size=12,
            batch_first=True,
            dropout=0,
            num_layers=2,
            in_keys=["observation", "hidden0", "hidden1"],
            out_keys=["intermediate", ("next", "hidden0"), ("next", "hidden1")],
            default_recurrent_mode=True,
        )
        obs = torch.rand(10, 20, 3)

        hidden0 = torch.rand(10, 20, 2, 12)
        hidden1 = torch.rand(10, 20, 2, 12)

        is_init = torch.zeros(10, 20, dtype=torch.bool)
        assert isinstance(lstm_module.lstm, nn.LSTM)
        outs_ref = lstm_module(
            observation=obs, hidden0=hidden0, hidden1=hidden1, is_init=is_init
        )

        lstm_module.make_python_based()
        assert isinstance(lstm_module.lstm, torchrl.modules.LSTM)
        outs_rl = lstm_module(
            observation=obs, hidden0=hidden0, hidden1=hidden1, is_init=is_init
        )
        torch.testing.assert_close(outs_ref, outs_rl)

        lstm_module.make_cudnn_based()
        assert isinstance(lstm_module.lstm, nn.LSTM)
        outs_cudnn = lstm_module(
            observation=obs, hidden0=hidden0, hidden1=hidden1, is_init=is_init
        )
        torch.testing.assert_close(outs_ref, outs_cudnn)

    def test_noncontiguous(self):
        lstm_module = LSTMModule(
            input_size=3,
            hidden_size=12,
            batch_first=True,
            in_keys=["bork", "h0", "h1"],
            out_keys=["dork", ("next", "h0"), ("next", "h1")],
        )
        td = TensorDict(
            {
                "bork": torch.randn(3, 3),
                "is_init": torch.zeros(3, 1, dtype=torch.bool),
            },
            [3],
        )
        padded = pad(td, [0, 5])
        lstm_module(padded)

    @pytest.mark.parametrize("shape", [[], [2], [2, 3], [2, 3, 4]])
    @pytest.mark.parametrize("python_based", [True, False])
    def test_single_step(self, shape, python_based):
        td = TensorDict(
            {
                "observation": torch.zeros(*shape, 3),
                "is_init": torch.zeros(*shape, 1, dtype=torch.bool),
            },
            shape,
        )
        lstm_module = LSTMModule(
            input_size=3,
            hidden_size=12,
            batch_first=True,
            in_keys=["observation", "hidden0", "hidden1"],
            out_keys=["intermediate", ("next", "hidden0"), ("next", "hidden1")],
            python_based=python_based,
        )
        td = lstm_module(td)
        td_next = step_mdp(td, keep_other=True)
        td_next = lstm_module(td_next)

        assert not torch.isclose(
            td_next["next", "hidden0"], td["next", "hidden0"]
        ).any()

    @pytest.mark.parametrize("shape", [[], [2], [2, 3], [2, 3, 4]])
    @pytest.mark.parametrize("t", [1, 10])
    @pytest.mark.parametrize("python_based", [True, False])
    def test_single_step_vs_multi(self, shape, t, python_based):
        td = TensorDict(
            {
                "observation": torch.arange(t, dtype=torch.float32)
                .unsqueeze(-1)
                .expand(*shape, t, 3),
                "is_init": torch.zeros(*shape, t, 1, dtype=torch.bool),
            },
            [*shape, t],
        )
        lstm_module_ss = LSTMModule(
            input_size=3,
            hidden_size=12,
            batch_first=True,
            in_keys=["observation", "hidden0", "hidden1"],
            out_keys=["intermediate", ("next", "hidden0"), ("next", "hidden1")],
            python_based=python_based,
        )
        lstm_module_ms = lstm_module_ss.set_recurrent_mode()
        lstm_module_ms(td)
        td_ss = TensorDict(
            {
                "observation": torch.zeros(*shape, 3),
                "is_init": torch.zeros(*shape, 1, dtype=torch.bool),
            },
            shape,
        )
        for _t in range(t):
            lstm_module_ss(td_ss)
            td_ss = step_mdp(td_ss, keep_other=True)
            td_ss["observation"][:] = _t + 1
        torch.testing.assert_close(
            td_ss["hidden0"], td["next", "hidden0"][..., -1, :, :]
        )

    @pytest.mark.parametrize("shape", [[], [2], [2, 3], [2, 3, 4]])
    @pytest.mark.parametrize("python_based", [False, True])
    def test_multi_consecutive(self, shape, python_based):
        t = 20
        td = TensorDict(
            {
                "observation": torch.arange(t, dtype=torch.float32)
                .unsqueeze(-1)
                .expand(*shape, t, 3),
                "is_init": torch.zeros(*shape, t, 1, dtype=torch.bool),
            },
            [*shape, t],
        )
        if shape:
            td["is_init"][0, ..., 13, :] = True
        else:
            td["is_init"][13, :] = True

        lstm_module_ss = LSTMModule(
            input_size=3,
            hidden_size=12,
            num_layers=4,
            batch_first=True,
            in_keys=["observation", "hidden0", "hidden1"],
            out_keys=["intermediate", ("next", "hidden0"), ("next", "hidden1")],
            python_based=python_based,
        )
        lstm_module_ms = lstm_module_ss.set_recurrent_mode()
        lstm_module_ms(td)
        td_ss = TensorDict(
            {
                "observation": torch.zeros(*shape, 3),
                "is_init": torch.zeros(*shape, 1, dtype=torch.bool),
            },
            shape,
        )
        for _t in range(t):
            td_ss["is_init"][:] = td["is_init"][..., _t, :]
            lstm_module_ss(td_ss)
            td_ss = step_mdp(td_ss, keep_other=True)
            td_ss["observation"][:] = _t + 1
        # import ipdb; ipdb.set_trace()  # assert fails when python_based is True, why?
        torch.testing.assert_close(
            td_ss["intermediate"], td["intermediate"][..., -1, :]
        )

    @pytest.mark.parametrize("python_based", [True, False])
    @pytest.mark.parametrize("parallel", [True, False])
    @pytest.mark.parametrize("heterogeneous", [True, False])
    @pytest.mark.parametrize("within", [False, True])
    def test_lstm_parallel_env(self, python_based, parallel, heterogeneous, within):
        from torchrl.envs import InitTracker, ParallelEnv, TransformedEnv

        torch.manual_seed(0)
        num_envs = 3
        device = "cuda" if torch.cuda.device_count() else "cpu"
        # tests that hidden states are carried over with parallel envs
        lstm_module = LSTMModule(
            input_size=7,
            hidden_size=12,
            num_layers=2,
            in_key="observation",
            out_key="features",
            device=device,
            python_based=python_based,
        )
        if parallel:
            cls = ParallelEnv
        else:
            cls = SerialEnv

        if within:

            def create_transformed_env():
                primer = lstm_module.make_tensordict_primer()
                env = DiscreteActionVecMockEnv(
                    categorical_action_encoding=True, device=device
                )
                env = TransformedEnv(env)
                env.append_transform(InitTracker())
                env.append_transform(primer)
                return env

        else:
            create_transformed_env = functools.partial(
                DiscreteActionVecMockEnv,
                categorical_action_encoding=True,
                device=device,
            )

        if heterogeneous:
            create_transformed_env = [
                EnvCreator(create_transformed_env) for _ in range(num_envs)
            ]
        env = cls(
            create_env_fn=create_transformed_env,
            num_workers=num_envs,
        )
        if not within:
            env = env.append_transform(InitTracker())
            env.append_transform(lstm_module.make_tensordict_primer())

        mlp = TensorDictModule(
            MLP(
                in_features=12,
                out_features=7,
                num_cells=[],
                device=device,
            ),
            in_keys=["features"],
            out_keys=["logits"],
        )

        actor_model = TensorDictSequential(lstm_module, mlp)

        actor = ProbabilisticActor(
            module=actor_model,
            in_keys=["logits"],
            out_keys=["action"],
            distribution_class=torch.distributions.Categorical,
            return_log_prob=True,
        )
        for break_when_any_done in [False, True]:
            data = env.rollout(10, actor, break_when_any_done=break_when_any_done)
            assert (data.get(("next", "recurrent_state_c")) != 0.0).all()
            assert (data.get("recurrent_state_c") != 0.0).any()
        return data  # noqa

    @pytest.mark.parametrize("python_based", [True, False])
    @pytest.mark.parametrize("parallel", [True, False])
    @pytest.mark.parametrize("heterogeneous", [True, False])
    def test_lstm_parallel_within(self, python_based, parallel, heterogeneous):
        out_within = self.test_lstm_parallel_env(
            python_based, parallel, heterogeneous, within=True
        )
        out_not_within = self.test_lstm_parallel_env(
            python_based, parallel, heterogeneous, within=False
        )
        assert_close(out_within, out_not_within)

    @pytest.mark.skipif(
        not _has_functorch, reason="vmap can only be used with functorch"
    )
    def test_lstm_vmap_complex_model(self):
        # Tests that all ops in GRU are compatible with VMAP (when build using
        # the PT backend).
        # This used to fail when splitting the input based on the is_init mask.
        # This test is intended not only as a non-regression test but also
        # to make sure that any change provided to RNNs is compliant with vmap
        torch.manual_seed(0)
        input_size = 4
        hidden_size = 5
        num_layers = 1
        output_size = 3
        out_key = "out"

        embedding_module = TensorDictModule(
            in_keys=["observation"],
            out_keys=["embed"],
            module=torch.nn.Linear(input_size, input_size),
        )

        lstm_module = LSTMModule(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            in_key="embed",
            out_key="features",
            python_based=True,
        )
        mlp = TensorDictModule(
            MLP(
                in_features=hidden_size,
                out_features=output_size,
                num_cells=[],
            ),
            in_keys=["features"],
            out_keys=[out_key],
        )
        training_model = TensorDictSequential(
            embedding_module, lstm_module.set_recurrent_mode(), mlp
        )
        is_init = torch.zeros(50, 11, 1, dtype=torch.bool).bernoulli_(0.1)
        data = TensorDict(
            {"observation": torch.randn(50, 11, input_size), "is_init": is_init},
            [50, 11],
        )
        training_model(data)
        params = TensorDict.from_module(training_model)
        params = params.expand(2)

        def call(data, params):
            with params.to_module(training_model):
                return training_model(data)

        assert vmap(call, (None, 0))(data, params).shape == torch.Size((2, 50, 11))


class TestGRUModule:
    def test_errs(self):
        with pytest.raises(ValueError, match="batch_first"):
            gru_module = GRUModule(
                input_size=3,
                hidden_size=12,
                batch_first=False,
                in_keys=["observation", "hidden"],
                out_keys=["intermediate", ("next", "hidden")],
            )
        with pytest.raises(ValueError, match="in_keys"):
            gru_module = GRUModule(
                input_size=3,
                hidden_size=12,
                batch_first=True,
                in_keys=[
                    "observation",
                    "hidden0",
                    "hidden1",
                ],
                out_keys=["intermediate", ("next", "hidden")],
            )
        with pytest.raises(TypeError, match="incompatible function arguments"):
            gru_module = GRUModule(
                input_size=3,
                hidden_size=12,
                batch_first=True,
                in_keys="abc",
                out_keys=["intermediate", ("next", "hidden")],
            )
        with pytest.raises(ValueError, match="in_keys"):
            gru_module = GRUModule(
                input_size=3,
                hidden_size=12,
                batch_first=True,
                in_key="smth",
                in_keys=["observation", "hidden0", "hidden1"],
                out_keys=["intermediate", ("next", "hidden")],
            )
        with pytest.raises(ValueError, match="out_keys"):
            gru_module = GRUModule(
                input_size=3,
                hidden_size=12,
                batch_first=True,
                in_keys=["observation", "hidden"],
                out_keys=["intermediate", ("next", "hidden"), "other"],
            )
        with pytest.raises(TypeError, match="incompatible function arguments"):
            gru_module = GRUModule(
                input_size=3,
                hidden_size=12,
                batch_first=True,
                in_keys=["observation", "hidden"],
                out_keys="abc",
            )
        with pytest.raises(ValueError, match="out_keys"):
            gru_module = GRUModule(
                input_size=3,
                hidden_size=12,
                batch_first=True,
                in_keys=["observation", "hidden"],
                out_key="smth",
                out_keys=["intermediate", ("next", "hidden"), "other"],
            )
        gru_module = GRUModule(
            input_size=3,
            hidden_size=12,
            batch_first=True,
            in_keys=["observation", "hidden"],
            out_keys=["intermediate", ("next", "hidden")],
        )
        td = TensorDict({"observation": torch.randn(3)}, [])
        with pytest.raises(KeyError, match="is_init"):
            gru_module(td)

    @pytest.mark.parametrize("default_val", [False, True, None])
    def test_set_recurrent_mode(self, default_val):
        gru_module = GRUModule(
            input_size=3,
            hidden_size=12,
            batch_first=True,
            in_keys=["observation", "hidden"],
            out_keys=["intermediate", ("next", "hidden")],
            default_recurrent_mode=default_val,
        )
        assert gru_module.recurrent_mode is bool(default_val)
        with set_recurrent_mode(True):
            assert gru_module.recurrent_mode
            with set_recurrent_mode(False):
                assert not gru_module.recurrent_mode
                with set_recurrent_mode("recurrent"):
                    assert gru_module.recurrent_mode
                    with set_recurrent_mode("sequential"):
                        assert not gru_module.recurrent_mode
                    assert gru_module.recurrent_mode
                assert not gru_module.recurrent_mode
            assert gru_module.recurrent_mode
        assert gru_module.recurrent_mode is bool(default_val)

    @pytest.mark.filterwarnings("ignore::DeprecationWarning")
    def test_set_temporal_mode(self):
        gru_module = GRUModule(
            input_size=3,
            hidden_size=12,
            batch_first=True,
            in_keys=["observation", "hidden"],
            out_keys=["intermediate", ("next", "hidden")],
        )
        assert gru_module.set_recurrent_mode(False) is gru_module
        assert not gru_module.set_recurrent_mode(False).recurrent_mode
        assert gru_module.set_recurrent_mode(True) is not gru_module
        assert gru_module.set_recurrent_mode(True).recurrent_mode
        assert set(gru_module.set_recurrent_mode(True).parameters()) == set(
            gru_module.parameters()
        )

    def test_python_cudnn(self):
        gru_module = GRUModule(
            input_size=3,
            hidden_size=12,
            batch_first=True,
            dropout=0,
            num_layers=2,
            in_keys=["observation", "hidden0"],
            out_keys=["intermediate", ("next", "hidden0")],
        ).set_recurrent_mode(True)
        obs = torch.rand(10, 20, 3)

        hidden0 = torch.rand(10, 20, 2, 12)

        is_init = torch.zeros(10, 20, dtype=torch.bool)
        assert isinstance(gru_module.gru, nn.GRU)
        outs_ref = gru_module(observation=obs, hidden0=hidden0, is_init=is_init)

        gru_module.make_python_based()
        assert isinstance(gru_module.gru, torchrl.modules.GRU)
        outs_rl = gru_module(observation=obs, hidden0=hidden0, is_init=is_init)
        torch.testing.assert_close(outs_ref, outs_rl)

        gru_module.make_cudnn_based()
        assert isinstance(gru_module.gru, nn.GRU)
        outs_cudnn = gru_module(observation=obs, hidden0=hidden0, is_init=is_init)
        torch.testing.assert_close(outs_ref, outs_cudnn)

    def test_noncontiguous(self):
        gru_module = GRUModule(
            input_size=3,
            hidden_size=12,
            batch_first=True,
            in_keys=["bork", "h"],
            out_keys=["dork", ("next", "h")],
        )
        td = TensorDict(
            {
                "bork": torch.randn(3, 3),
                "is_init": torch.zeros(3, 1, dtype=torch.bool),
            },
            [3],
        )
        padded = pad(td, [0, 5])
        gru_module(padded)

    @pytest.mark.parametrize("shape", [[], [2], [2, 3], [2, 3, 4]])
    @pytest.mark.parametrize("python_based", [True, False])
    def test_single_step(self, shape, python_based):
        td = TensorDict(
            {
                "observation": torch.zeros(*shape, 3),
                "is_init": torch.zeros(*shape, 1, dtype=torch.bool),
            },
            shape,
        )
        gru_module = GRUModule(
            input_size=3,
            hidden_size=12,
            batch_first=True,
            in_keys=["observation", "hidden"],
            out_keys=["intermediate", ("next", "hidden")],
            python_based=python_based,
        )
        td = gru_module(td)
        td_next = step_mdp(td, keep_other=True)
        td_next = gru_module(td_next)

        assert not torch.isclose(td_next["next", "hidden"], td["next", "hidden"]).any()

    @pytest.mark.parametrize("shape", [[], [2], [2, 3], [2, 3, 4]])
    @pytest.mark.parametrize("t", [1, 10])
    @pytest.mark.parametrize("python_based", [True, False])
    def test_single_step_vs_multi(self, shape, t, python_based):
        td = TensorDict(
            {
                "observation": torch.arange(t, dtype=torch.float32)
                .unsqueeze(-1)
                .expand(*shape, t, 3),
                "is_init": torch.zeros(*shape, t, 1, dtype=torch.bool),
            },
            [*shape, t],
        )
        gru_module_ss = GRUModule(
            input_size=3,
            hidden_size=12,
            batch_first=True,
            in_keys=["observation", "hidden"],
            out_keys=["intermediate", ("next", "hidden")],
            python_based=python_based,
        )
        gru_module_ms = gru_module_ss.set_recurrent_mode()
        gru_module_ms(td)
        td_ss = TensorDict(
            {
                "observation": torch.zeros(*shape, 3),
                "is_init": torch.zeros(*shape, 1, dtype=torch.bool),
            },
            shape,
        )
        for _t in range(t):
            gru_module_ss(td_ss)
            td_ss = step_mdp(td_ss, keep_other=True)
            td_ss["observation"][:] = _t + 1
        torch.testing.assert_close(td_ss["hidden"], td["next", "hidden"][..., -1, :, :])

    @pytest.mark.parametrize("shape", [[], [2], [2, 3], [2, 3, 4]])
    @pytest.mark.parametrize("python_based", [True, False])
    def test_multi_consecutive(self, shape, python_based):
        t = 20
        td = TensorDict(
            {
                "observation": torch.arange(t, dtype=torch.float32)
                .unsqueeze(-1)
                .expand(*shape, t, 3),
                "is_init": torch.zeros(*shape, t, 1, dtype=torch.bool),
            },
            [*shape, t],
        )
        if shape:
            td["is_init"][0, ..., 13, :] = True
        else:
            td["is_init"][13, :] = True

        gru_module_ss = GRUModule(
            input_size=3,
            hidden_size=12,
            batch_first=True,
            in_keys=["observation", "hidden"],
            out_keys=["intermediate", ("next", "hidden")],
            python_based=python_based,
        )
        gru_module_ms = gru_module_ss.set_recurrent_mode()
        gru_module_ms(td)
        td_ss = TensorDict(
            {
                "observation": torch.zeros(*shape, 3),
                "is_init": torch.zeros(*shape, 1, dtype=torch.bool),
            },
            shape,
        )
        for _t in range(t):
            td_ss["is_init"][:] = td["is_init"][..., _t, :]
            gru_module_ss(td_ss)
            td_ss = step_mdp(td_ss, keep_other=True)
            td_ss["observation"][:] = _t + 1
        torch.testing.assert_close(
            td_ss["intermediate"], td["intermediate"][..., -1, :]
        )

    @pytest.mark.parametrize("python_based", [True, False])
    @pytest.mark.parametrize("parallel", [True, False])
    @pytest.mark.parametrize("heterogeneous", [True, False])
    @pytest.mark.parametrize("within", [False, True])
    def test_gru_parallel_env(self, python_based, parallel, heterogeneous, within):
        from torchrl.envs import InitTracker, ParallelEnv, TransformedEnv

        torch.manual_seed(0)
        num_workers = 3

        device = "cuda" if torch.cuda.device_count() else "cpu"
        # tests that hidden states are carried over with parallel envs
        gru_module = GRUModule(
            input_size=7,
            hidden_size=12,
            num_layers=2,
            in_key="observation",
            out_key="features",
            device=device,
            python_based=python_based,
        )

        if within:

            def create_transformed_env():
                primer = gru_module.make_tensordict_primer()
                env = DiscreteActionVecMockEnv(
                    categorical_action_encoding=True, device=device
                )
                env = TransformedEnv(env)
                env.append_transform(InitTracker())
                env.append_transform(primer)
                return env

        else:
            create_transformed_env = functools.partial(
                DiscreteActionVecMockEnv,
                categorical_action_encoding=True,
                device=device,
            )

        if parallel:
            cls = ParallelEnv
        else:
            cls = SerialEnv
        if heterogeneous:
            create_transformed_env = [
                EnvCreator(create_transformed_env) for _ in range(num_workers)
            ]

        env: ParallelEnv | SerialEnv = cls(
            create_env_fn=create_transformed_env,
            num_workers=num_workers,
        )
        if not within:
            primer = gru_module.make_tensordict_primer()
            env = env.append_transform(InitTracker())
            env.append_transform(primer)

        mlp = TensorDictModule(
            MLP(
                in_features=12,
                out_features=7,
                num_cells=[],
                device=device,
            ),
            in_keys=["features"],
            out_keys=["logits"],
        )

        actor_model = TensorDictSequential(gru_module, mlp)

        actor = ProbabilisticActor(
            module=actor_model,
            in_keys=["logits"],
            out_keys=["action"],
            distribution_class=torch.distributions.Categorical,
            return_log_prob=True,
        )
        for break_when_any_done in [False, True]:
            data = env.rollout(10, actor, break_when_any_done=break_when_any_done)
            assert (data.get("recurrent_state") != 0.0).any()
            assert (data.get(("next", "recurrent_state")) != 0.0).all()
        return data  # noqa

    @pytest.mark.parametrize("python_based", [True, False])
    @pytest.mark.parametrize("parallel", [True, False])
    @pytest.mark.parametrize("heterogeneous", [True, False])
    def test_gru_parallel_within(self, python_based, parallel, heterogeneous):
        out_within = self.test_gru_parallel_env(
            python_based, parallel, heterogeneous, within=True
        )
        out_not_within = self.test_gru_parallel_env(
            python_based, parallel, heterogeneous, within=False
        )
        assert_close(out_within, out_not_within)

    @pytest.mark.skipif(
        not _has_functorch, reason="vmap can only be used with functorch"
    )
    def test_gru_vmap_complex_model(self):
        # Tests that all ops in GRU are compatible with VMAP (when build using
        # the PT backend).
        # This used to fail when splitting the input based on the is_init mask.
        # This test is intended not only as a non-regression test but also
        # to make sure that any change provided to RNNs is compliant with vmap
        torch.manual_seed(0)
        input_size = 4
        hidden_size = 5
        num_layers = 1
        output_size = 3
        out_key = "out"

        embedding_module = TensorDictModule(
            in_keys=["observation"],
            out_keys=["embed"],
            module=torch.nn.Linear(input_size, input_size),
        )

        lstm_module = GRUModule(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            in_key="embed",
            out_key="features",
            python_based=True,
        )
        mlp = TensorDictModule(
            MLP(
                in_features=hidden_size,
                out_features=output_size,
                num_cells=[],
            ),
            in_keys=["features"],
            out_keys=[out_key],
        )
        training_model = TensorDictSequential(
            embedding_module, lstm_module.set_recurrent_mode(), mlp
        )
        is_init = torch.zeros(50, 11, 1, dtype=torch.bool).bernoulli_(0.1)
        data = TensorDict(
            {"observation": torch.randn(50, 11, input_size), "is_init": is_init},
            [50, 11],
        )
        training_model(data)
        params = TensorDict.from_module(training_model)
        params = params.expand(2)

        def call(data, params):
            with params.to_module(training_model):
                return training_model(data)

        assert vmap(call, (None, 0))(data, params).shape == torch.Size((2, 50, 11))


def test_safe_specs():

    out_key = ("a", "b")
    spec = Composite(Composite({out_key: Unbounded()}))
    original_spec = spec.clone()
    mod = SafeModule(
        module=nn.Linear(3, 1),
        spec=spec,
        out_keys=[out_key, ("other", "key")],
        in_keys=[],
    )
    assert original_spec == spec
    assert original_spec[out_key] == mod.spec[out_key]


def test_actor_critic_specs():
    action_key = ("agents", "action")
    spec = Composite(Composite({action_key: Unbounded(shape=(3,))}))
    policy_module = TensorDictModule(
        nn.Linear(3, 1),
        in_keys=[("agents", "observation")],
        out_keys=[action_key],
    )
    original_spec = spec.clone()
    module = TensorDictSequential(
        policy_module, AdditiveGaussianModule(spec=spec, action_key=action_key)
    )
    value_module = ValueOperator(
        module=module,
        in_keys=[("agents", "observation"), action_key],
        out_keys=[("agents", "state_action_value")],
    )
    assert original_spec == spec
    assert module[1].spec == spec
    DDPGLoss(actor_network=module, value_network=value_module)
    assert original_spec == spec
    assert module[1].spec == spec


def test_vmapmodule():
    lam = TensorDictModule(lambda x: x[0], in_keys=["x"], out_keys=["y"])
    sample_in = torch.ones((10, 3, 2))
    sample_in_td = TensorDict({"x": sample_in}, batch_size=[10])
    lam(sample_in)
    vm = VmapModule(lam, 0)
    vm(sample_in_td)
    assert (sample_in_td["x"][:, 0] == sample_in_td["y"]).all()


@pytest.mark.skipif(
    not _has_transformers, reason="transformers needed to test DT classes"
)
class TestDecisionTransformerInferenceWrapper:
    @pytest.mark.parametrize("online", [True, False])
    def test_dt_inference_wrapper(self, online):
        action_key = ("nested", ("action",))
        if online:
            dtactor = OnlineDTActor(
                state_dim=4, action_dim=2, transformer_config=DTActor.default_config()
            )
            in_keys = ["loc", "scale"]
            actor_module = TensorDictModule(
                dtactor,
                in_keys=["observation", action_key, "return_to_go"],
                out_keys=in_keys,
            )
            dist_class = TanhNormal
        else:
            dtactor = DTActor(
                state_dim=4, action_dim=2, transformer_config=DTActor.default_config()
            )
            in_keys = ["param"]
            actor_module = TensorDictModule(
                dtactor,
                in_keys=["observation", action_key, "return_to_go"],
                out_keys=in_keys,
            )
            dist_class = TanhDelta
        dist_kwargs = {
            "low": -1.0,
            "high": 1.0,
        }
        actor = ProbabilisticActor(
            in_keys=in_keys,
            out_keys=[action_key],
            module=actor_module,
            distribution_class=dist_class,
            distribution_kwargs=dist_kwargs,
        )
        inference_actor = DecisionTransformerInferenceWrapper(actor)
        sequence_length = 20
        td = TensorDict(
            {
                "observation": torch.randn(1, sequence_length, 4),
                action_key: torch.randn(1, sequence_length, 2),
                "return_to_go": torch.randn(1, sequence_length, 1),
            },
            [1],
        )
        with pytest.raises(
            ValueError,
            match="The value of out_action_key",
        ):
            result = inference_actor(td)
        inference_actor.set_tensor_keys(action=action_key, out_action=action_key)
        result = inference_actor(td)
        # checks that the seq length has disappeared
        assert result.get(action_key).shape == torch.Size([1, 2])
        assert inference_actor.out_keys == unravel_key_list(
            sorted([action_key, *in_keys, "observation", "return_to_go"], key=str)
        )
        assert set(result.keys(True, True)) - set(td.keys(True, True)) == set(
            inference_actor.out_keys
        ) - set(inference_actor.in_keys)


class TestBatchedActor:
    def test_batched_actor_exceptions(self):
        time_steps = 5
        actor_base = TensorDictModule(
            lambda x: torch.ones(
                x.shape[0], time_steps, 1, device=x.device, dtype=x.dtype
            ),
            in_keys=["observation_cat"],
            out_keys=["action"],
        )
        with pytest.raises(ValueError, match="Only a single init_key can be passed"):
            MultiStepActorWrapper(actor_base, n_steps=time_steps, init_key=["init_key"])

        batch = 2

        # The second env has frequent resets, the first none
        base_env = SerialEnv(
            batch,
            [lambda: CountingEnv(max_steps=5000), lambda: CountingEnv(max_steps=5)],
        )
        env = TransformedEnv(
            base_env,
            CatFrames(
                N=time_steps,
                in_keys=["observation"],
                out_keys=["observation_cat"],
                dim=-1,
            ),
        )
        actor = MultiStepActorWrapper(actor_base, n_steps=time_steps)
        with pytest.raises(KeyError, match="No init key was passed"):
            env.rollout(2, actor)

        env = TransformedEnv(
            base_env,
            Compose(
                InitTracker(),
                CatFrames(
                    N=time_steps,
                    in_keys=["observation"],
                    out_keys=["observation_cat"],
                    dim=-1,
                ),
            ),
        )
        td = env.rollout(10)[..., -1]["next"]
        actor = MultiStepActorWrapper(actor_base, n_steps=time_steps)
        with pytest.raises(RuntimeError, match="Cannot initialize the wrapper"):
            env.rollout(10, actor, tensordict=td, auto_reset=False)

        actor = MultiStepActorWrapper(actor_base, n_steps=time_steps - 1)
        with pytest.raises(RuntimeError, match="The action's time dimension"):
            env.rollout(10, actor)

    @pytest.mark.parametrize("time_steps", [3, 5])
    def test_batched_actor_simple(self, time_steps):

        batch = 2

        # The second env has frequent resets, the first none
        base_env = SerialEnv(
            batch,
            [lambda: CountingEnv(max_steps=5000), lambda: CountingEnv(max_steps=5)],
        )
        env = TransformedEnv(
            base_env,
            Compose(
                InitTracker(),
                CatFrames(
                    N=time_steps,
                    in_keys=["observation"],
                    out_keys=["observation_cat"],
                    dim=-1,
                ),
            ),
        )

        actor_base = TensorDictModule(
            lambda x: torch.ones(
                x.shape[0], time_steps, 1, device=x.device, dtype=x.dtype
            ),
            in_keys=["observation_cat"],
            out_keys=["action"],
        )
        actor = MultiStepActorWrapper(actor_base, n_steps=time_steps)
        # rollout = env.rollout(100, break_when_any_done=False)
        rollout = env.rollout(50, actor, break_when_any_done=False)
        unique = rollout[0]["observation"].unique()
        predicted = torch.arange(unique.numel())
        assert (unique == predicted).all()
        assert (
            rollout[1]["observation"]
            == (torch.arange(50) % 6).reshape_as(rollout[1]["observation"])
        ).all()


def test_get_primers_from_module():

    # No primers in the model
    module = MLP(in_features=10, out_features=10, num_cells=[])
    transform = get_primers_from_module(module)
    assert transform is None

    # 1 primer in the model
    gru_module = GRUModule(
        input_size=10,
        hidden_size=10,
        num_layers=1,
        in_keys=["input", "gru_recurrent_state", "is_init"],
        out_keys=["features", ("next", "gru_recurrent_state")],
    )
    transform = get_primers_from_module(gru_module)
    assert isinstance(transform, TensorDictPrimer)
    assert "gru_recurrent_state" in transform.primers

    # 2 primers in the model
    composed_model = TensorDictSequential(
        gru_module,
        LSTMModule(
            input_size=10,
            hidden_size=10,
            num_layers=1,
            in_keys=[
                "input",
                "lstm_recurrent_state_c",
                "lstm_recurrent_state_h",
                "is_init",
            ],
            out_keys=[
                "features",
                ("next", "lstm_recurrent_state_c"),
                ("next", "lstm_recurrent_state_h"),
            ],
        ),
    )
    transform = get_primers_from_module(composed_model)
    assert isinstance(transform, Compose)
    assert len(transform) == 2
    assert "gru_recurrent_state" in transform[0].primers
    assert "lstm_recurrent_state_c" in transform[1].primers
    assert "lstm_recurrent_state_h" in transform[1].primers


if __name__ == "__main__":
    args, unknown = argparse.ArgumentParser().parse_known_args()
    pytest.main([__file__, "--capture", "no", "--exitfirst"] + unknown)
