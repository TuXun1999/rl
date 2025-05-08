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
from robot import SpotRLEnvChairMove

def create_chair_move_env(env_name = "spot-chair-move", robot = None, transform_state_dict = None, device = "cpu"):
    # Create the RL environment
    if env_name == "spot-chair-move":
        env = SpotRLEnvChairMove(device=device)
    # The first unsqueeze is to help stack the input to the network
    env = TransformedEnv(
        env,
    )

    out_key = "observation"
    cat_transform = CatTensors(
        in_keys=["object_current_pose",
                "robot_current_pose",
                "hand_current_pose]"], dim=-1, out_key=out_key, del_keys=False
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