# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
from __future__ import annotations

import functools
import tempfile
from contextlib import nullcontext
from typing import List, Optional, Tuple

import torch
from tensordict.nn import TensorDictModule, TensorDictSequential, TensorDictModuleBase
from torchrl.envs.utils import step_mdp
from torchrl.objectives.utils import (
    _cache_values,
    _GAMMA_LMBDA_DEPREC_ERROR,
    _reduce,
    _vmap_func,
    default_value_kwargs,
    distance_loss,
    ValueEstimators,
)
from torch import nn, optim
from torchrl.collectors import SyncDataCollector
from torchrl.data import TensorDictPrioritizedReplayBuffer, TensorDictReplayBuffer
from torchrl.data.replay_buffers.storages import LazyMemmapStorage, LazyTensorStorage
from torchrl.envs import (
    CatTensors,
    Compose,
    DMControlEnv,
    DoubleToFloat,
    EnvCreator,
    InitTracker,
    ParallelEnv,
    RewardSum,
    StepCounter,
    TransformedEnv,
)
from torchrl.envs.libs.gym import GymEnv, set_gym_backend
from torchrl.envs.utils import ExplorationType, set_exploration_type
from torchrl.modules import AdditiveGaussianModule, MLP, TanhModule, ValueOperator
from torchrl.data.tensor_specs import Bounded, Composite, TensorSpec

from torchrl.objectives import SoftUpdate
from torchrl.objectives.td3 import TD3Loss
from torchrl.record import VideoRecorder

from rl_env import create_se2_env
import copy
# ====================================================================
# Environment utils
# -----------------


def env_maker(cfg, device="cpu", from_pixels=False):
    lib = cfg.env.library
    if lib in ("gym", "gymnasium"):
        with set_gym_backend(lib):
            return GymEnv(
                cfg.env.name,
                device=device,
                from_pixels=from_pixels,
                pixels_only=False,
            )
    elif lib == "dm_control":
        env = DMControlEnv(
            cfg.env.name, cfg.env.task, from_pixels=from_pixels, pixels_only=False
        )
        return TransformedEnv(
            env, CatTensors(in_keys=env.observation_spec.keys(), out_key="observation")
        )
    elif lib == "se2":
        env = create_se2_env(type = "se2", device=device)
        return env
    elif lib == "se2-fixed":
        env = create_se2_env(type = "se2-fixed", device=device)
        return env
    else:
        raise NotImplementedError(f"Unknown lib {lib}.")


def apply_env_transforms(env, max_episode_steps):
    transformed_env = TransformedEnv(
        env,
        Compose(
            StepCounter(max_steps=max_episode_steps),
            InitTracker(),
            DoubleToFloat(),
            RewardSum(),
        ),
    )
    return transformed_env


def make_environment(cfg, logger, device):
    """Make environments for training and evaluation."""
    partial = functools.partial(env_maker, cfg=cfg)
    parallel_env = ParallelEnv(
        cfg.collector.env_per_collector,
        EnvCreator(partial),
        serial_for_single=True,
        device=device,
    )
    parallel_env.set_seed(cfg.env.seed)

    train_env = apply_env_transforms(parallel_env, cfg.env.max_episode_steps)

    partial = functools.partial(env_maker, cfg=cfg, from_pixels=cfg.logger.video)
    trsf_clone = train_env.transform.clone()
    if cfg.logger.video:
        trsf_clone.insert(
            0, VideoRecorder(logger, tag="rendering/test", in_keys=["pixels"])
        )
    eval_env = TransformedEnv(
        ParallelEnv(
            1,
            EnvCreator(partial),
            serial_for_single=True,
            device=device,
        ),
        trsf_clone,
    )
    return train_env, eval_env


# ====================================================================
# Collector and replay buffer
# ---------------------------


def make_collector(cfg, train_env, actor_model_explore, compile_mode, device):
    """Make collector."""
    collector_device = cfg.collector.device
    if collector_device in ("", None):
        collector_device = device
    collector = SyncDataCollector(
        train_env,
        actor_model_explore,
        init_random_frames=cfg.collector.init_random_frames,
        frames_per_batch=cfg.collector.frames_per_batch,
        total_frames=cfg.collector.total_frames,
        reset_at_each_iter=cfg.collector.reset_at_each_iter,
        device=collector_device,
        compile_policy={"mode": compile_mode} if compile_mode else False,
        cudagraph_policy=cfg.compile.cudagraphs,
    )
    collector.set_seed(cfg.env.seed)
    return collector

def make_prior_collector(cfg, train_env, actor_model_explore, compile_mode, total_frames=1000):
    """Make collector."""
    device = cfg.collector.device
    if device in ("", None):
        if torch.cuda.is_available():
            device = torch.device("cuda:0")
        else:
            device = torch.device("cpu")
    collector = SyncDataCollector(
        train_env,
        actor_model_explore,
        init_random_frames=None,
        frames_per_batch=cfg.collector.frames_per_batch,
        total_frames=total_frames,
        device=device,
        compile_policy={"mode": compile_mode} if compile_mode else False,
        cudagraph_policy=cfg.compile.cudagraphs,
        exploration_type=ExplorationType.DETERMINISTIC,
    )

    return collector

def make_replay_buffer(
    batch_size: int,
    prb: bool = False,
    buffer_size: int = 1000000,
    scratch_dir: str | None = None,
    device: torch.device = "cpu",
    prefetch: int = 3,
    compile: bool = False,
):
    if compile:
        prefetch = 0
    if scratch_dir in ("", None):
        ctx = nullcontext(None)
    elif scratch_dir == "temp":
        ctx = tempfile.TemporaryDirectory()
    else:
        ctx = nullcontext(scratch_dir)
    with ctx as scratch_dir:
        storage_cls = (
            functools.partial(LazyTensorStorage, device=device, compilable=compile)
            if not scratch_dir
            else functools.partial(
                LazyMemmapStorage, device="cpu", scratch_dir=scratch_dir
            )
        )

        if prb:
            replay_buffer = TensorDictPrioritizedReplayBuffer(
                alpha=0.7,
                beta=0.5,
                pin_memory=False,
                prefetch=prefetch,
                storage=storage_cls(buffer_size),
                batch_size=batch_size,
                compilable=compile,
            )
        else:
            replay_buffer = TensorDictReplayBuffer(
                pin_memory=False,
                prefetch=prefetch,
                storage=storage_cls(buffer_size),
                batch_size=batch_size,
                compilable=compile,
            )
        if scratch_dir:
            replay_buffer.append_transform(lambda td: td.to(device))
        return replay_buffer


# ====================================================================
# Model
# -----


def make_td3_agent(cfg, train_env, eval_env, device):
    """Make TD3 agent."""
    # Define Actor Network
    in_keys = ["observation"]
    action_spec = train_env.action_spec_unbatched.to(device)
    actor_net = MLP(
        num_cells=cfg.network.hidden_sizes,
        out_features=action_spec.shape[-1],
        activation_class=get_activation(cfg),
        device=device,
    )

    in_keys_actor = in_keys
    actor_module = TensorDictModule(
        actor_net,
        in_keys=in_keys_actor,
        out_keys=["param"],
    )
    actor = TensorDictSequential(
        actor_module,
        TanhModule(
            in_keys=["param"],
            out_keys=["action"],
            spec=action_spec,
        ),
    )

    # Define Critic Network
    qvalue_net = MLP(
        num_cells=cfg.network.hidden_sizes,
        out_features=1,
        activation_class=get_activation(cfg),
        device=device,
    )

    qvalue = ValueOperator(
        in_keys=["action"] + in_keys,
        module=qvalue_net,
    )

    model = nn.ModuleList([actor, qvalue])

    # init nets
    with torch.no_grad(), set_exploration_type(ExplorationType.RANDOM):
        td = eval_env.fake_tensordict()
        td = td.to(device)
        for net in model:
            net(td)
    # Exploration wrappers:
    actor_model_explore = TensorDictSequential(
        actor,
        AdditiveGaussianModule(
            sigma_init=1,
            sigma_end=1,
            mean=0,
            std=0.1,
            spec=action_spec,
            device=device,
        ),
    )
    return model, actor_model_explore

class ChoosePolicy_ILRL(TensorDictModuleBase):
    def __init__(self, il_actor, rl_actor, qvalue_critic, device = "cpu"):
        super().__init__()
        self.il_policy = il_actor
        self.rl_actor = rl_actor
        self.qvalue_critic = qvalue_critic

        self.device = device

        
    def forward(self, tensordict: TensorDictModule):
        # Obtain the action from the IL policy
        il_action = self.il_policy(tensordict)
        il_qvalue = il_action.select(
            *self.qvalue_critic.in_keys, strict=False
        )

        # Obtain the action from the RL actor
        rl_action = self.rl_actor(tensordict)
        rl_qvalue = rl_action.select(
            *self.qvalue_critic.in_keys, strict=False
        )

        # Choose the action based on the Q-value of the two policies
        il_q_value = self.qvalue_critic(il_qvalue)["state_action_value"]
        rl_q_value = self.qvalue_critic(rl_qvalue)["state_action_value"]
        # Choose the action with the higher Q-value
        chosen_action = torch.where(il_q_value > rl_q_value, il_action["action"], rl_action["action"])
        rl_qvalue.set("action", chosen_action)
        return rl_qvalue

def make_ib_policy_agent(il_policy, rl_actor, qvalue_critic, device = "cpu"):


    return ChoosePolicy_ILRL(il_policy, rl_actor, qvalue_critic, device = device)

class ScaleLayer(nn.Module):
    def __init__(self, scale=-1.0, device=None):
        super().__init__()
        self.scale = scale
        self.device = device

    def forward(self, observation: torch.Tensor) -> torch.Tensor:
        # Extract the statistics in alphabet order
        output = observation.clone()
        output[..., 0:] = output[..., 0:] * self.scale
        
        # Small noise
        noise = torch.randn_like(output) * 0.01
        output = output + noise

        if len(output.shape) == 1:
            output = output.unsqueeze(0)
        
        output[..., [0, 1, 2]] = output[..., [1, 2, 0]]
        return output
def make_prior_agent(
        cfg, 
        device="cpu",
    ):  

    ## A prior determined actor (no exploration)

    actor_net = ScaleLayer(device=device)

    in_keys = ["observation_raw"]
    out_keys = ["action"]

    # Convert it into TensorDictModule
    actor = TensorDictModule(
        actor_net,
        in_keys=in_keys,
        out_keys=out_keys,
    )
    return actor
# ====================================================================
# TD3 Loss
# ---------

class TD3LossIBRL(TD3Loss):
    il_policy_network: TensorDictModule
    def __init__(
        self,
        actor_network: TensorDictModule,
        il_policy_network: TensorDictModule,
        qvalue_network: TensorDictModule | List[TensorDictModule],
        *,
        action_spec: TensorSpec = None,
        bounds: Optional[Tuple[float]] = None,
        num_qvalue_nets: int = 2,
        policy_noise: float = 0.2,
        noise_clip: float = 0.5,
        loss_function: str = "smooth_l1",
        delay_actor: bool = True,
        delay_qvalue: bool = True,
        gamma: float = None,
        priority_key: str = None,
        separate_losses: bool = False,
        reduction: str = None,
    ) -> None:
        super().__init__(
            actor_network=actor_network,
            qvalue_network=qvalue_network,
            action_spec=action_spec,
            bounds=bounds,
            num_qvalue_nets=num_qvalue_nets,
            policy_noise=policy_noise,
            noise_clip=noise_clip,
            loss_function=loss_function,
            delay_actor=delay_actor,
            delay_qvalue=delay_qvalue,
            gamma=gamma,
            priority_key=priority_key,
            separate_losses=separate_losses,
            reduction=reduction
        )
        self.convert_to_functional(
            il_policy_network,
            "il_policy_network"
        )
        self.choose_il = 0
    
    def value_loss(self, tensordict) -> Tuple[torch.Tensor, dict]:
        tensordict = tensordict.clone(False)

        act = tensordict.get(self.tensor_keys.action)

        # computing early for reprod
        noise = (torch.randn_like(act) * self.policy_noise).clamp(
            -self.noise_clip, self.noise_clip
        )

        with torch.no_grad():
            next_td_actor = step_mdp(tensordict).select(
                *self.actor_network.in_keys, strict=False
            )  # next_observation ->
            with self.target_actor_network_params.to_module(self.actor_network):
                next_td_actor = self.actor_network(next_td_actor)
            next_action = (next_td_actor.get(self.tensor_keys.action) + noise).clamp(
                self.min_action, self.max_action
            )
            next_td_actor.set(
                self.tensor_keys.action,
                next_action,
            )
            next_val_td = next_td_actor.select(
                *self.qvalue_network.in_keys, strict=False
            ).expand(
                self.num_qvalue_nets, *next_td_actor.batch_size
            )  # for next value estimation
            next_target_q1q2 = (
                self._vmap_qvalue_network00(
                    next_val_td,
                    self.target_qvalue_network_params,
                )
                .get(self.tensor_keys.state_action_value)
                .squeeze(-1)
            )

        # Find the new targe qvalues using the pretrained il policy
        with torch.no_grad():
            # Also, extract the "observation" from the tensordict
            keys = self.il_policy_network.in_keys + self.actor_network.in_keys
            next_td_actor = step_mdp(tensordict).select(
                *keys, strict=False
            )  # next_observation ->
            
            next_td_actor = self.il_policy_network(next_td_actor)
            next_action_il = next_td_actor.get(self.tensor_keys.action)
            next_td_actor.set(
                self.tensor_keys.action,
                next_action_il,
            )
            
            next_val_td = next_td_actor.select(
                *self.qvalue_network.in_keys, strict=False
            ).expand(
                self.num_qvalue_nets, *next_td_actor.batch_size
            )  # for next value estimation
            next_target_q1q2_il = (
                self._vmap_qvalue_network00(
                    next_val_td,
                    self.target_qvalue_network_params,
                )
                .get(self.tensor_keys.state_action_value)
                .squeeze(-1)
            )
        # min over the next target qvalues
        next_target_qvalue = next_target_q1q2.min(0)[0]
        next_target_qvalue_il = next_target_q1q2_il.min(0)[0]
        self.choose_il +=torch.count_nonzero(
            next_target_qvalue_il > next_target_qvalue).item()
        next_target_qvalue = torch.max(
            next_target_qvalue, next_target_qvalue_il
        )
       

        # set next target qvalues
        tensordict.set(
            ("next", self.tensor_keys.state_action_value),
            next_target_qvalue.unsqueeze(-1),
        )

        qval_td = tensordict.select(*self.qvalue_network.in_keys, strict=False).expand(
            self.num_qvalue_nets,
            *tensordict.batch_size,
        )
        # preditcted current qvalues
        current_qvalue = (
            self._vmap_qvalue_network00(
                qval_td,
                self.qvalue_network_params,
            )
            .get(self.tensor_keys.state_action_value)
            .squeeze(-1)
        )

        # compute target values for the qvalue loss (reward + gamma * next_target_qvalue * (1 - done))
        target_value = self.value_estimator.value_estimate(tensordict).squeeze(-1)

        td_error = (current_qvalue - target_value).pow(2)
        loss_qval = distance_loss(
            current_qvalue,
            target_value.expand_as(current_qvalue),
            loss_function=self.loss_function,
        ).sum(0)
        metadata = {
            "td_error": td_error,
            "next_state_value": next_target_qvalue.detach(),
            "pred_value": current_qvalue.detach(),
            "target_value": target_value.detach(),
        }
        loss_qval = _reduce(loss_qval, reduction=self.reduction)
        self._clear_weakrefs(
            tensordict,
            "actor_network_params",
            "qvalue_network_params",
            "target_actor_network_params",
            "target_qvalue_network_params",
        )
        return loss_qval, metadata


def make_loss_module(cfg, model, il_policy_network = None):
    """Make loss module and target network updater."""
    if il_policy_network is not None: 
        # Create TD3 loss with il policy bootstrapping
        loss_module = TD3LossIBRL(
            actor_network=model[0],
            il_policy_network=il_policy_network,
            qvalue_network=model[1],
            num_qvalue_nets=2,
            loss_function=cfg.optim.loss_function,
            delay_actor=True,
            delay_qvalue=True,
            action_spec=model[0][1].spec,
            policy_noise=cfg.optim.policy_noise,
            noise_clip=cfg.optim.noise_clip,
        )
    else:
        # Create TD3 loss
        loss_module = TD3Loss(
            actor_network=model[0],
            qvalue_network=model[1],
            num_qvalue_nets=2,
            loss_function=cfg.optim.loss_function,
            delay_actor=True,
            delay_qvalue=True,
            action_spec=model[0][1].spec,
            policy_noise=cfg.optim.policy_noise,
            noise_clip=cfg.optim.noise_clip,
        )
    loss_module.make_value_estimator(gamma=cfg.optim.gamma)

    # Define Target Network Updater
    target_net_updater = SoftUpdate(loss_module, eps=cfg.optim.target_update_polyak)
    return loss_module, target_net_updater


def make_optimizer(cfg, loss_module):
    critic_params = list(loss_module.qvalue_network_params.flatten_keys().values())
    actor_params = list(loss_module.actor_network_params.flatten_keys().values())

    optimizer_actor = optim.Adam(
        actor_params,
        lr=cfg.optim.lr,
        weight_decay=cfg.optim.weight_decay,
        eps=cfg.optim.adam_eps,
    )
    optimizer_critic = optim.Adam(
        critic_params,
        lr=cfg.optim.lr,
        weight_decay=cfg.optim.weight_decay,
        eps=cfg.optim.adam_eps,
    )
    return optimizer_actor, optimizer_critic


# ====================================================================
# General utils
# ---------


def log_metrics(logger, metrics, step):
    for metric_name, metric_value in metrics.items():
        logger.log_scalar(metric_name, metric_value, step)


def get_activation(cfg):
    if cfg.network.activation == "relu":
        return nn.ReLU
    elif cfg.network.activation == "tanh":
        return nn.Tanh
    elif cfg.network.activation == "leaky_relu":
        return nn.LeakyReLU
    else:
        raise NotImplementedError


def dump_video(module):
    if isinstance(module, VideoRecorder):
        module.dump()
