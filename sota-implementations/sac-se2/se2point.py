from collections import defaultdict
from typing import Optional

import numpy as np
import torch
import tqdm
from tensordict import TensorDict, TensorDictBase
from tensordict.nn import TensorDictModule
from torch import nn

from torchrl.data import Bounded, Composite, Unbounded
from torchrl.envs import (
    CatTensors,
    EnvBase,
    Transform,
    TransformedEnv,
    UnsqueezeTransform,
)
from torchrl.envs.transforms.transforms import _apply_to_composite
from torchrl.envs.utils import check_env_specs, step_mdp
from torchrl.envs.utils import check_env_specs, ExplorationType, set_exploration_type

## Section I: Helper functions to construct the env
from rl_env import SE2PointEnv
import bosdyn

import argparse
from spot import SPOT, SpotRLEnvSE2
from absl import app, flags

def main(argv):
    # Simple validality checks
    env = SE2PointEnv()
    check_env_specs(env)

    ## Apply transforms 
    # The first unsqueeze is to help stack the input to the network
    env = TransformedEnv(
        env,
        # ``Unsqueeze`` the observations that we will concatenate
        UnsqueezeTransform(
            dim=-1,
            in_keys=["x", "y", "theta"],
            in_keys_inv=["x", "y", "theta"],
        ),
    )
    class SinTransform(Transform):
        def _apply_transform(self, obs: torch.Tensor) -> None:
            return obs.sin()

        # The transform must also modify the data at reset time
        def _reset(
            self, tensordict: TensorDictBase, tensordict_reset: TensorDictBase
        ) -> TensorDictBase:
            return self._call(tensordict_reset)

        # _apply_to_composite will execute the observation spec transform across all
        # in_keys/out_keys pairs and write the result in the observation_spec which
        # is of type ``Composite``
        @_apply_to_composite
        def transform_observation_spec(self, observation_spec):
            return Bounded(
                low=-1,
                high=1,
                shape=observation_spec.shape,
                dtype=observation_spec.dtype,
                device=observation_spec.device,
            )


    class CosTransform(Transform):
        def _apply_transform(self, obs: torch.Tensor) -> None:
            return obs.cos()

        # The transform must also modify the data at reset time
        def _reset(
            self, tensordict: TensorDictBase, tensordict_reset: TensorDictBase
        ) -> TensorDictBase:
            return self._call(tensordict_reset)

        # _apply_to_composite will execute the observation spec transform across all
        # in_keys/out_keys pairs and write the result in the observation_spec which
        # is of type ``Composite``
        @_apply_to_composite
        def transform_observation_spec(self, observation_spec):
            return Bounded(
                low=-1,
                high=1,
                shape=observation_spec.shape,
                dtype=observation_spec.dtype,
                device=observation_spec.device,
            )


    t_sin = SinTransform(in_keys=["theta"], out_keys=["sin"])
    t_cos = CosTransform(in_keys=["theta"], out_keys=["cos"])
    env.append_transform(t_sin)
    env.append_transform(t_cos)

    cat_transform = CatTensors(
        in_keys=["sin", "cos", "x", "y"], dim=-1, out_key="observation", del_keys=False
    )
    env.append_transform(cat_transform)

    check_env_specs(env)


    # Policy network
    torch.manual_seed(0)
    env.set_seed(0)

    net = nn.Sequential(
        nn.LazyLinear(64),
        nn.Tanh(),
        nn.LazyLinear(64),
        nn.Tanh(),
        nn.LazyLinear(64),
        nn.Tanh(),
        nn.LazyLinear(3),
    )
    policy = TensorDictModule(
        net,
        in_keys=["observation"],
        out_keys=["action"],
    )

    # Optimizer
    optim = torch.optim.Adam(policy.parameters(), lr=2e-3)

    ## Formally, train the policy
    batch_size = 25
    n_iter = 5000 # set to 20_000 for a proper training
    pbar = tqdm.tqdm(range(n_iter // batch_size))
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optim, n_iter)
    logs = defaultdict(list)

    for _ in pbar:
        init_td = env.reset(env.gen_params(batch_size=[batch_size]))
        rollout = env.rollout(100, policy, tensordict=init_td, auto_reset=False)
        traj_return = rollout["next", "reward"].mean()
        (-traj_return).backward()
        gn = torch.nn.utils.clip_grad_norm_(net.parameters(), 1.0)
        optim.step()
        optim.zero_grad()
        pbar.set_description(
            f"reward: {traj_return: 4.4f}, "
            f"last reward: {rollout[..., -1]['next', 'reward'].mean(): 4.4f}, gradient norm: {gn: 4.4}"
        )
        logs["return"].append(traj_return.item())
        logs["last_reward"].append(rollout[..., -1]["next", "reward"].mean().item())
        scheduler.step()

    # Plot out the results
    def plot():
        import matplotlib
        from matplotlib import pyplot as plt

        is_ipython = "inline" in matplotlib.get_backend()
        if is_ipython:
            from IPython import display

        with plt.ion():
            plt.figure(figsize=(10, 5))
            plt.subplot(1, 2, 1)
            plt.plot(logs["return"])
            plt.title("returns")
            plt.xlabel("iteration")
            plt.subplot(1, 2, 2)
            plt.plot(logs["last_reward"])
            plt.title("last reward")
            plt.xlabel("iteration")
            plt.pause(2) # <---- add pause
            if is_ipython:
                display.display(plt.gcf())
                display.clear_output(wait=True)
            plt.show()

            # print('is this fig good? (y/n)')
            # x = input()
            # if x=="y":
            #     plt.savefig(r'C:\figures\papj.png')
            # else:
            #     print("big sad")


    plot()

    # Inference on SPOT
    # Create the robot interface
    # Parse args
    parser = argparse.ArgumentParser()
    bosdyn.client.util.add_base_arguments(parser)
    options = parser.parse_args(argv[1:])

    robot = SPOT(options)
    with robot.lease_alive():
        robot.power_on_stand()
        # Create the RL environment from the robot
        spot_env = SpotRLEnvSE2(robot)
        # The first unsqueeze is to help stack the input to the network
        spot_env = TransformedEnv(
            spot_env,
            # ``Unsqueeze`` the observations that we will concatenate
            UnsqueezeTransform(
                dim=-1,
                in_keys=["x", "y", "theta"],
                in_keys_inv=["x", "y", "theta"],
            ),
        )

        t_sin = SinTransform(in_keys=["theta"], out_keys=["sin"])
        t_cos = CosTransform(in_keys=["theta"], out_keys=["cos"])
        spot_env.append_transform(t_sin)
        spot_env.append_transform(t_cos)

        cat_transform = CatTensors(
            in_keys=["sin", "cos", "x", "y"], dim=-1, out_key="observation", del_keys=False
        )
        spot_env.append_transform(cat_transform)

        # Inference on the trained policy
        print("reset")
        init_td = spot_env.reset(spot_env.gen_params())
        print("rollout")
        rollout = spot_env.rollout(100, policy, tensordict=init_td, auto_reset=False)



if __name__ == "__main__":
    app.run(main)