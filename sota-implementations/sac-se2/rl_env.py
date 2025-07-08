from collections import defaultdict
from typing import Optional

import numpy as np
import torch

from tensordict import TensorDict, TensorDictBase


from torchrl.data import Bounded, Composite, Unbounded
from torchrl.envs import (
    EnvBase,
    Transform,
    TransformedEnv,
    UnsqueezeTransform,
    CatTensors,
    ObservationNorm,
    DoubleToFloat,
)

from torchrl.envs.transforms.transforms import _apply_to_composite
from torchrl.envs.utils import check_env_specs, step_mdp
from robot import SpotRLEnvSE2, SpotRLEnvBodyVelocitySE2
DEFAULT_X = 2.0
DEFAULT_ANGLE = np.pi

## Section I: Helper functions to construct the env

# 1. Step function
def _step(self, tensordict):
    x, y, theta = tensordict["x"], tensordict["y"], tensordict["theta"]

    dt = tensordict["params", "dt"]
    position_noise = tensordict["params", "position_noise"]
    angle_noise = tensordict["params", "angle_noise"]
    
    # Read the value
    u = tensordict["action"]

    if len(u.shape) == 1:
        vx = u[0].clamp(-tensordict["params", "max_velocity"], \
                        tensordict["params", "max_velocity"])
        vy = u[1].clamp(-tensordict["params", "max_velocity"], \
                        tensordict["params", "max_velocity"])
        vtheta = u[2].clamp(-tensordict["params", "max_angular_velocity"], \
                        tensordict["params", "max_angular_velocity"])
    else:
        vx = u[:, 0].clamp(-tensordict["params", "max_velocity"], \
                        tensordict["params", "max_velocity"])
        vy = u[:, 1].clamp(-tensordict["params", "max_velocity"], \
                        tensordict["params", "max_velocity"])
        vtheta = u[:, 2].clamp(-tensordict["params", "max_angular_velocity"], \
                        tensordict["params", "max_angular_velocity"])
    
    dist = (x ** 2 + y ** 2)**0.5
    # costs_location = -dist + torch.exp(-dist) + torch.exp(-10*dist)
    # costs_yaw = torch.exp(-abs(theta)) + torch.exp(-10*abs(theta))
    # costs = costs_location + costs_yaw
    costs = -dist - 0.4*abs(theta)
    assert (abs(theta) >= 0).all(), "Theta is negative, which is not allowed"
    assert (dist >= 0).all(), "Distance is negative, which is not allowed"
    # The ODE of motion of equation (with noise)
    noise_x = torch.rand(x.shape, generator=self.rng, device=self.device) * position_noise
    noise_y = torch.rand(y.shape, generator=self.rng, device=self.device) * position_noise
    noise_theta = torch.rand(theta.shape, generator=self.rng, device=self.device) * angle_noise
    nx = x + vx * dt + noise_x
    ny = y + vy * dt + noise_y
    ntheta = theta + vtheta * dt + noise_theta

    # The reward is depending on the current state
    reward = costs.view(*tensordict.shape, 1) # Expand the dim to be consistent with the shape?
    done = torch.zeros_like(reward, dtype=torch.bool)
    mask = reward > -0.25 #2.5
    done[mask] = True
    out = TensorDict(
        {
            "x": nx,
            "y": ny,
            "theta": ntheta,
            "params": tensordict["params"],
            "reward": reward,
            "done": done,
        },
        tensordict.shape,
    )
    return out

# 2. Reset function
def _reset(self, tensordict):
    if tensordict is None or tensordict.is_empty() or tensordict.get("_reset") is not None:
        # if no ``tensordict`` is passed, we generate a single set of hyperparameters
        # Otherwise, we assume that the input ``tensordict`` contains all the relevant
        # parameters to get started.
        tensordict = self.gen_params(batch_size=self.batch_size)


    high_x = torch.tensor(DEFAULT_X, device=self.device)
    high_angle = torch.tensor(DEFAULT_ANGLE, device=self.device)
    low_x = -high_x
    low_angle = -high_angle
    # for non batch-locked environments, the input ``tensordict`` shape dictates the number
    # of simulators run simultaneously. In other contexts, the initial
    # random state's shape will depend upon the environment batch-size instead.
    x = (
        torch.rand(tensordict.shape, generator=self.rng, device=self.device)
        * (high_x - low_x)
        + low_x
    )
    y = (
        torch.rand(tensordict.shape, generator=self.rng, device=self.device)
        * (high_x - low_x)
        + low_x
    )
    theta = (
        torch.rand(tensordict.shape, generator=self.rng, device=self.device)
        * (high_angle- low_angle)
        + low_angle
    )

    out = TensorDict(
        {
            "x": x,
            "y": y,
            "theta": theta,
            "params": tensordict["params"],
        },
        batch_size=tensordict.shape,
    )
    return out

# 3. Spec functions: basic input/output specifications for the environment
# Four to go: action_spec, observation_spec, reward_spec, done_spec
def _make_spec(self, td_params):
    # Under the hood, this will populate self.output_spec["observation"]
    self.observation_spec = Composite(
        x=Bounded(
            low=-torch.tensor(DEFAULT_X),
            high=torch.tensor(DEFAULT_X),
            shape=(),
            dtype=torch.float32,
        ),
        y=Bounded(
            low=-torch.tensor(DEFAULT_X),
            high=torch.tensor(DEFAULT_X),
            shape=(),
            dtype=torch.float32,
        ),
        theta=Bounded(
            low=-torch.tensor(DEFAULT_ANGLE),
            high=torch.tensor(DEFAULT_ANGLE),
            shape=(),
            dtype=torch.float32,
        ),
        # we need to add the ``params`` to the observation specs, as we want
        # to pass it at each step during a rollout
        params=make_composite_from_td(td_params["params"]),
        shape=(),
    )
    # since the environment is stateless, we expect the previous output as input.
    # For this, ``EnvBase`` expects some state_spec to be available
    self.state_spec = self.observation_spec.clone()
    # action-spec will be automatically wrapped in input_spec when
    # `self.action_spec = spec` will be called supported
    # NOTE: Might be a bug here
    self.action_spec = Bounded(
        low=np.array(\
            [-td_params["params", "max_velocity"], \
                -td_params["params", "max_velocity"],\
                    -td_params["params", "max_angular_velocity"]]),
        high=np.array(\
            [td_params["params", "max_velocity"], \
                td_params["params", "max_velocity"],\
                    td_params["params", "max_angular_velocity"]]),
        shape=(3,),        
        dtype=torch.float32,
    )
    

    self.reward_spec = Unbounded(shape=(*td_params.shape, 1))


def make_composite_from_td(td):
    # custom function to convert a ``tensordict`` in a similar spec structure
    # of unbounded values.
    composite = Composite(
        {
            key: make_composite_from_td(tensor)
            if isinstance(tensor, TensorDictBase)
            else Unbounded(dtype=tensor.dtype, device=tensor.device, shape=tensor.shape)
            for key, tensor in td.items()
        },
        shape=td.shape,
    )
    return composite


## 3. Wrap things together
# Random seeding
def _set_seed(self, seed: Optional[int]):
    rng = torch.Generator(device=self.device)
    rng = rng.manual_seed(seed)
    self.rng = rng
# Define the parameters
def gen_params(self, batch_size=None) -> TensorDictBase:
    """Returns a ``tensordict`` containing the physical parameters such 
    as gravitational force and torque or speed limits."""
    if batch_size is None:
        batch_size = []
    td = TensorDict(
        {
            "params": TensorDict(
                {
                    "max_velocity": 0.6,
                    "max_angular_velocity": 1.0,
                    "dt": 2.0,
                },
                [],
            )
        },
        [],
        device=self.device
    )
    if batch_size:
        td = td.expand(batch_size).contiguous()
    return td

# Finally, combine all things together
class SE2PointEnv(EnvBase):
    metadata = {
        "render_modes": ["human", "rgb_array"],
        "render_fps": 30,
    } # Seems to be related to rendering
    batch_locked = False # Usually set to false for "stateless" environment

    def __init__(self, td_params=None, seed=None, device="cpu"):
        if td_params is None:
            td_params = self.gen_params()

        super().__init__(device=device, batch_size=[])
        self._make_spec(td_params)
        if seed is None:
            seed = torch.empty((), dtype=torch.int64).random_().item()
        self.set_seed(seed)

    # Helpers: _make_step and gen_params
    gen_params = gen_params
    _make_spec = _make_spec

    # Mandatory methods: _step, _reset and _set_seed
    _reset = _reset
    _step = _step
    _set_seed = _set_seed

# Helper functions for SE2 env
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
def create_se2_env(env_name = "se2", robot = None, transform_state_dict = None, device = "cpu"):
    # Create the RL environment
    if env_name == "se2":
        env = SE2PointEnv(device=device)
    elif env_name == "spot":
        env = SpotRLEnvSE2(robot=robot, device=device)
    elif env_name == "spot-body-velocity":
        env = SpotRLEnvBodyVelocitySE2(robot=robot, device=device)
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

    t_sin = SinTransform(in_keys=["theta"], out_keys=["sin"])
    t_cos = CosTransform(in_keys=["theta"], out_keys=["cos"])
    env.append_transform(t_sin)
    env.append_transform(t_cos)

    out_key = "observation"
    cat_transform = CatTensors(
        in_keys=["sin", "cos", "x", "y"], dim=-1, out_key=out_key, del_keys=False
    )
    env.append_transform(cat_transform)

    out_key = "observation_raw"
    cat_transform = CatTensors(
        in_keys=["x", "y", "theta"], dim=-1, out_key=out_key, del_keys=False
    )
    env.append_transform(cat_transform)

    # obs_norm = ObservationNorm(in_keys=[out_key], standard_normal=True)
    # env.append_transform(obs_norm)
    # if transform_state_dict is not None:
    #     # Initialize the observation normalization transform with the
    #     # provided mean and std
    #     env.transform[-1].init_stats(3)
    #     env.transform[-1].loc.copy_(transform_state_dict["loc"])
    #     env.transform[-1].scale.copy_(transform_state_dict["scale"])
    
    return env
