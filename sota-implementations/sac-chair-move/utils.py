# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
from __future__ import annotations

import functools

import torch
from tensordict.nn import InteractionType, TensorDictModule
from tensordict.nn.distributions import NormalParamExtractor
from torch import nn, optim
from torchrl.collectors import SyncDataCollector
from torchrl.data import (
    LazyMemmapStorage,
    LazyTensorStorage,
    TensorDictPrioritizedReplayBuffer,
    TensorDictReplayBuffer,
    Bounded,
)
from torchrl.envs import (
    CatTensors,
    Compose,
    DMControlEnv,
    DoubleToFloat,
    EnvCreator,
    ParallelEnv,
    Transform,
    TransformedEnv,
    UnsqueezeTransform,
)
from torchrl.envs.libs.gym import GymEnv, set_gym_backend
from torchrl.envs import PendulumEnv
from torchrl.envs.transforms import InitTracker, RewardSum, StepCounter
from torchrl.envs.transforms.transforms import _apply_to_composite
from torchrl.envs.utils import ExplorationType, set_exploration_type
from torchrl.modules import MLP, ProbabilisticActor, ValueOperator
from torchrl.modules.distributions import TanhNormal
from torchrl.objectives import SoftUpdate
from torchrl.objectives.sac import SACLoss
from torchrl.record import VideoRecorder

from rl_env import create_chair_move_env
# ====================================================================
# Environment utils
# -----------------


def env_maker(cfg, device="cuda:0", transform_state_dict=None, from_pixels=False, robot=None):
    lib = cfg.env.library
    if lib in ("gym", "gymnasium"):
        with set_gym_backend(lib):
            return GymEnv(
                cfg.env.name, device=device, from_pixels=from_pixels, pixels_only=False
            )
    elif lib == "dm_control":
        env = DMControlEnv(
            cfg.env.name, cfg.env.task, from_pixels=from_pixels, pixels_only=False
        )
        return TransformedEnv(
            env, CatTensors(in_keys=env.observation_spec.keys(), out_key="observation")
        )
    elif lib == "spot-chair-move":
        env = create_chair_move_env("spot-chair-move", robot=robot, transform_state_dict=transform_state_dict, device=device)
        return env
    else:
        raise NotImplementedError(f"Unknown lib {lib}.")


def apply_env_transforms(env, max_episode_steps=1000):
    transformed_env = TransformedEnv(
        env,
        Compose(
            InitTracker(),
            StepCounter(max_episode_steps),
            DoubleToFloat(),
            RewardSum(),
        ),
    )
    return transformed_env

## Transforms for the environments
# Get the normalization stats through random exploration
def get_env_stats(cfg, obs_norm_idx=-1, robot=None):
    # Deprecated
    """Gets the stats of an environment."""
    lib = cfg.env.library
    if lib == "se2":
        proof_env = create_se2_env()
    else:
        proof_env = create_se2_env(env_name="spot", robot=robot)
    t = proof_env.transform[obs_norm_idx]
    print("*** Collect normalization stats ***")
    t.init_stats(cfg.collector.init_env_steps)
    print("*** Done ***")
    transform_state_dict = t.state_dict()
    proof_env.close()
    return transform_state_dict
def make_environment(cfg, logger, robot = None):
    # """Make environments for training and evaluation."""
    # Get the normalization stats from the random exploration in env
    transform_state_dict = None #get_env_stats(cfg, robot=robot)
    maker = functools.partial(env_maker, cfg, \
                transform_state_dict=transform_state_dict, from_pixels=False, robot=robot)
    parallel_env = ParallelEnv(
        cfg.collector.env_per_collector,
        EnvCreator(maker),
        serial_for_single=True,
    )
    parallel_env.set_seed(cfg.env.seed)

    train_env = apply_env_transforms(
        parallel_env, max_episode_steps=cfg.env.max_episode_steps
    )

    maker = functools.partial(env_maker, cfg, \
                    transform_state_dict=transform_state_dict, from_pixels=cfg.logger.video,\
                        robot=robot)
    eval_env = TransformedEnv(
        ParallelEnv(
            cfg.logger.num_eval_envs,
            EnvCreator(maker),
            serial_for_single=True,
        ),
        train_env.transform.clone(),
    )
    eval_env.set_seed(0)
    if cfg.logger.video:
        eval_env = eval_env.append_transform(
            VideoRecorder(logger, tag="rendered", in_keys=["pixels"])
        )
    return train_env, eval_env

# ====================================================================
# Collector and replay buffer
# ---------------------------


def make_collector(cfg, train_env, actor_model_explore, compile_mode, total_frames=1000):
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
        init_random_frames=cfg.collector.init_random_frames,
        frames_per_batch=cfg.collector.frames_per_batch,
        total_frames=total_frames,
        device=device,
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
    batch_size,
    prb=False,
    buffer_size=1000000,
    scratch_dir=None,
    device="cpu",
    prefetch=3,
):
    storage_cls = (
        functools.partial(LazyTensorStorage, device=device)
        if not scratch_dir
        else functools.partial(LazyMemmapStorage, device="cpu", scratch_dir=scratch_dir)
    )
    if prb:
        replay_buffer = TensorDictPrioritizedReplayBuffer(
            alpha=0.7,
            beta=0.5,
            pin_memory=False,
            prefetch=prefetch,
            storage=storage_cls(
                buffer_size,
            ),
            batch_size=batch_size,
        )
    else:
        replay_buffer = TensorDictReplayBuffer(
            pin_memory=False,
            prefetch=prefetch,
            storage=storage_cls(
                buffer_size,
            ),
            batch_size=batch_size,
        )
    if scratch_dir:
        replay_buffer.append_transform(lambda td: td.to(device))
    return replay_buffer


# ====================================================================
# Model
# -----
class ScaleLayer(nn.Module):
    def __init__(self, scale=-1.0):
        super().__init__()
        self.scale = scale

    def forward(self, observation: torch.Tensor) -> torch.Tensor:
        # Extract the statistics in alphabet order
        x = observation[..., 1]
        y = observation[..., 2]
        theta = observation[..., 0]
        output = torch.cat([self.scale * x, self.scale * y, self.scale * theta], dim=-1)
        if len(output.shape) == 1:
            output = output.unsqueeze(0)
        return output
def make_prior_agent(
        cfg, 
        device="cpu",
    ):  

    ## A prior determined actor (no exploration)

    actor_net = ScaleLayer()

    in_keys = ["observation_raw"]
    out_keys = ["action"]

    # Convert it into TensorDictModule
    actor = TensorDictModule(
        actor_net,
        in_keys=in_keys,
        out_keys=out_keys
    ).to(device=device)

    return actor



def make_sac_agent(cfg, train_env, eval_env, device):
    """Make SAC agent."""
    # Define Actor Network
    in_keys = ["observation"]
    action_spec = train_env.action_spec_unbatched.to(device)

    actor_net = MLP(
        num_cells=cfg.network.hidden_sizes,
        out_features=2 * action_spec.shape[-1],
        activation_class=get_activation(cfg),
        device=device,
    )

    dist_class = TanhNormal
    dist_kwargs = {
        "low": action_spec.space.low,
        "high": action_spec.space.high,
        "tanh_loc": False,
    }

    actor_extractor = NormalParamExtractor(
        scale_mapping=f"biased_softplus_{cfg.network.default_policy_scale}",
        scale_lb=cfg.network.scale_lb,
    ).to(device)
    actor_net = nn.Sequential(actor_net, actor_extractor)

    in_keys_actor = in_keys
    actor_module = TensorDictModule(
        actor_net,
        in_keys=in_keys_actor,
        out_keys=[
            "loc",
            "scale",
        ],
    )
    actor = ProbabilisticActor(
        spec=action_spec,
        in_keys=["loc", "scale"],
        module=actor_module,
        distribution_class=dist_class,
        distribution_kwargs=dist_kwargs,
        default_interaction_type=InteractionType.RANDOM,
        return_log_prob=False,
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
    return model, model[0]


# ====================================================================
# SAC Loss
# ---------


def make_loss_module(cfg, model):
    """Make loss module and target network updater."""
    # Create SAC loss
    loss_module = SACLoss(
        actor_network=model[0],
        qvalue_network=model[1],
        num_qvalue_nets=2,
        loss_function=cfg.optim.loss_function,
        delay_actor=False,
        delay_qvalue=True,
        alpha_init=cfg.optim.alpha_init,
    )
    loss_module.make_value_estimator(gamma=cfg.optim.gamma)

    # Define Target Network Updater
    target_net_updater = SoftUpdate(loss_module, eps=cfg.optim.target_update_polyak)
    return loss_module, target_net_updater


def split_critic_params(critic_params):
    critic1_params = []
    critic2_params = []

    for param in critic_params:
        data1, data2 = param.data.chunk(2, dim=0)
        critic1_params.append(nn.Parameter(data1))
        critic2_params.append(nn.Parameter(data2))
    return critic1_params, critic2_params


def make_sac_optimizer(cfg, loss_module):
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
    optimizer_alpha = optim.Adam(
        [loss_module.log_alpha],
        lr=3.0e-4,
    )
    return optimizer_actor, optimizer_critic, optimizer_alpha


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
