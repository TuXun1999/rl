# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
"""SAC Example.

This is a simple self-contained example of a SAC training script.

It supports state environments like MuJoCo.

The helper functions are coded in the utils.py associated with this script.
"""
from __future__ import annotations

import warnings

import hydra
import numpy as np
import torch
import torch.cuda
import tqdm
from tensordict import TensorDict
from tensordict.nn import CudaGraphModule
from torchrl._utils import compile_with_warmup, timeit
from torchrl.envs.utils import ExplorationType, set_exploration_type
from torchrl.objectives import group_optimizers
from torchrl.record.loggers import generate_exp_name, get_logger
from torchrl.modules import  ActorCriticOperator
from torchrl.modules.tensordict_module import SafeModule
from utils import (
    dump_video,
    log_metrics,
    make_collector,
    make_environment,
    make_loss_module,
    make_replay_buffer,
    make_sac_agent,
    make_sac_optimizer,
    make_prior_agent,
    make_prior_collector,
    get_env_stats
)

torch.set_float32_matmul_precision("high")
from robot import SPOT, SpotRLEnvSE2
from rl_env import create_se2_env
import bosdyn
import argparse
@hydra.main(version_base="1.1", config_path="", config_name="config")
def main(cfg: DictConfig):  # noqa: F821
    device = cfg.network.device
    if device in ("", None):
        if torch.cuda.is_available():
            device = torch.device("cuda:0")
        else:
            device = torch.device("cpu")
    device = torch.device(device)

    # Create logger
    exp_name = generate_exp_name("SAC_SE2", cfg.logger.exp_name)
    logger = None
    if cfg.logger.backend:
        logger = get_logger(
            logger_type=cfg.logger.backend,
            logger_name="sac_logging",
            experiment_name=exp_name,
            wandb_kwargs={
                "mode": cfg.logger.mode,
                "config": dict(cfg),
                "project": cfg.logger.project_name,
                "group": cfg.logger.group_name,
            },
        )

    torch.manual_seed(cfg.env.seed)
    np.random.seed(cfg.env.seed)
    
    robot = SPOT(cfg)
    robot.lease_alive()
    robot.power_on_stand()
    if cfg.new_graph:
        robot.create_graph()
    else:
        robot._upload_graph_and_snapshots()
    # Record the initial se2 pose of robot
    robot_start_pose = robot.get_base_pose_se2()
    # Create environments
    print("One simple test")
    print(robot._current_graph is not None)
    train_env, eval_env = make_environment(cfg, logger=logger, robot=robot)


    # Create agent
    model, exploration_policy = make_sac_agent(cfg, train_env, eval_env, device)
    prior_model = make_prior_agent(cfg, device)
    # Create SAC loss
    loss_module, target_net_updater = make_loss_module(cfg, model)

    compile_mode = None
    if cfg.compile.compile:
        compile_mode = cfg.compile.compile_mode
        if compile_mode in ("", None):
            if cfg.compile.cudagraphs:
                compile_mode = "default"
            else:
                compile_mode = "reduce-overhead"

    # Create off-policy collector
    collector = make_collector(
        cfg, train_env, exploration_policy, compile_mode=compile_mode, \
            total_frames=cfg.collector.total_frames
    )
    prior_collector = make_prior_collector(
        cfg, train_env, prior_model, compile_mode=compile_mode, \
            total_frames=cfg.collector.prior_frames
    )
    # t = input("Press y for a robot tempory shutdown test - ")
    # if t == "y":
    #     robot.lease_return()
    #     # Wait for the user to change the battery
    #     input("Battery level too low... please change the battery (Press any key to continue if done)")
    #     # Command the robot to stand up
    #     # robot.lease_alive()
    #     # robot.power_on_stand()
    #     # # Record the current se2 pose of robot
    #     # robot_start_pose = robot.get_base_pose_se2()

    #     # Re-initialize the robot object
    #     robot = SPOT(cfg)
    #     robot.lease_alive()
    #     robot.power_on_stand()
    #     # Update the agents
    #     train_env.update_robot(robot)
    #     eval_env.update_robot(robot)
    # Create replay buffer
    replay_buffer = make_replay_buffer(
        batch_size=cfg.optim.batch_size,
        prb=cfg.replay_buffer.prb,
        buffer_size=cfg.replay_buffer.size,
        scratch_dir=cfg.replay_buffer.scratch_dir,
        device=device,
    )
    prior_replay_buffer = make_replay_buffer(
        batch_size=cfg.optim.batch_size,
        prb=cfg.replay_buffer.prb,
        buffer_size=cfg.replay_buffer.size,
        scratch_dir=cfg.replay_buffer.scratch_dir,
        device=device,
    )

    # Create optimizers
    (
        optimizer_actor,
        optimizer_critic,
        optimizer_alpha,
    ) = make_sac_optimizer(cfg, loss_module)
    optimizer = group_optimizers(optimizer_actor, optimizer_critic, optimizer_alpha)
    del optimizer_actor, optimizer_critic, optimizer_alpha

    def update(sampled_tensordict):
        # Compute loss
        loss_td = loss_module(sampled_tensordict)

        actor_loss = loss_td["loss_actor"]
        q_loss = loss_td["loss_qvalue"]
        alpha_loss = loss_td["loss_alpha"]

        (actor_loss + q_loss + alpha_loss).sum().backward()
        optimizer.step()
        optimizer.zero_grad(set_to_none=True)

        # Update qnet_target params
        target_net_updater.step()
        return loss_td.detach()

    if cfg.compile.compile:
        update = compile_with_warmup(update, mode=compile_mode, warmup=1)

    if cfg.compile.cudagraphs:
        warnings.warn(
            "CudaGraphModule is experimental and may lead to silently wrong results. Use with caution.",
            category=UserWarning,
        )
        update = CudaGraphModule(update, in_keys=[], out_keys=[], warmup=5)

    # Main loop
    collected_frames = 0
    pbar = tqdm.tqdm(total=cfg.collector.total_frames)

    init_random_frames = cfg.collector.init_random_frames
    num_updates = int(cfg.collector.frames_per_batch * cfg.optim.utd_ratio)
    prb = cfg.replay_buffer.prb
    eval_iter = cfg.logger.eval_iter
    frames_per_batch = cfg.collector.frames_per_batch
    eval_rollout_steps = cfg.env.max_episode_steps

    collector_iter = iter(collector)
    total_iter = len(collector)

    prior_collector_iter = iter(prior_collector)
    total_prior_iter = len(prior_collector)
    '''
    Collect offline data
    '''
    # for i in range(total_prior_iter):
    #     timeit.printevery(num_prints=1000, total_count=total_iter, erase=True)
    #     print("***Collecting offline data***")
    #     with timeit("offline dataset"):
    #         tensordict = next(prior_collector_iter)
    #     with timeit("rb - extend"):
    #         tensordict = tensordict.reshape(-1)
    #         prior_replay_buffer.extend(tensordict)

    for i in range(total_iter):
        timeit.printevery(num_prints=1000, total_count=total_iter, erase=True)

        # Command the robot to go back to the original starting place 
        # (to ensure that the odom frame doesn't change too much in 
        # two different settings for safety concern)
        battery_state = robot.state_client.get_robot_state().battery_states[0]
        status = battery_state.Status.Name(battery_state.status)
        status = status[7:]  # get rid of STATUS_ prefix
        if battery_state.charge_percentage.value < 10: # Too low battery
            robot_current_pose = robot.get_base_pose_se2()
            # Go back to original starting pose
            robot.send_pose_command_se2(\
                robot_start_pose.position.x, \
                robot_start_pose.position.y,\
                robot_start_pose.angle)
            # Release the lease
            robot.lease_return()
            # Wait for the user to change the battery
            input("Battery level too low... please change the battery, self-right it \
                and release the control \n (Press any key to continue if done)")
            
            # Re-initialize the robot object
            robot = SPOT(cfg)
            robot.lease_alive()
            robot.power_on_stand()
            # Return the robot to the place where the training is paused
            robot.send_pose_command_se2(\
                robot_current_pose.position.x,
                robot_current_pose.position.y,
                robot_current_pose.angle
                )
            # Update the agents
            train_env.update_robot(robot)
            eval_env.update_robot(robot)


        with timeit("collect"):
            tensordict = next(collector_iter)

        # Update weights of the inference policy
        collector.update_policy_weights_()

        current_frames = tensordict.numel()
        pbar.update(current_frames)

        with timeit("rb - extend"):
            # Add to replay buffer
            tensordict = tensordict.reshape(-1)
            replay_buffer.extend(tensordict)

        collected_frames += current_frames

        # Optimization steps
        with timeit("train"):
            if collected_frames >= init_random_frames:
                losses = TensorDict(batch_size=[num_updates])
                for i in range(num_updates):
                    with timeit("rb - sample"):
                        sampled_tensordict = replay_buffer.sample()
                        # del sampled_tensordict["scale"]
                        # del sampled_tensordict["loc"]
                        # sampled_tensordict_prior = prior_replay_buffer.sample()
                        # sampled_tensordict = torch.cat(
                        #     (sampled_tensordict, sampled_tensordict_prior), dim=0
                        # )

                    with timeit("update"):
                        torch.compiler.cudagraph_mark_step_begin()
                        loss_td = update(sampled_tensordict).clone()
                    losses[i] = loss_td.select(
                        "loss_actor", "loss_qvalue", "loss_alpha"
                    )

                    # Update priority
                    if prb:
                        replay_buffer.update_priority(sampled_tensordict)
        episode_end = (
            tensordict["next", "done"]
            if tensordict["next", "done"].any()
            else tensordict["next", "truncated"]
        )
        episode_rewards = tensordict["next", "episode_reward"][episode_end]

        # Logging
        metrics_to_log = {}
        if len(episode_rewards) > 0:
            episode_length = tensordict["next", "step_count"][episode_end]
            metrics_to_log["train/reward"] = episode_rewards.sum() / len(episode_rewards)
            metrics_to_log["train/episode_length"] = episode_length.sum() / len(
                episode_length
            )
        if collected_frames >= init_random_frames:
            losses = losses.mean()
            metrics_to_log["train/q_loss"] = losses.get("loss_qvalue")
            metrics_to_log["train/actor_loss"] = losses.get("loss_actor")
            metrics_to_log["train/alpha_loss"] = losses.get("loss_alpha")
            metrics_to_log["train/alpha"] = loss_td["alpha"]
            metrics_to_log["train/entropy"] = loss_td["entropy"]

        
        if logger is not None:
            metrics_to_log.update(timeit.todict(prefix="time"))
            metrics_to_log["time/speed"] = pbar.format_dict["rate"]
            log_metrics(logger, metrics_to_log, collected_frames)

        
    print("Start Evaluation")
    # Evaluation
    
    with set_exploration_type(
        ExplorationType.DETERMINISTIC
    ), torch.no_grad(), timeit("eval"):
        eval_rollout = eval_env.rollout(
            eval_rollout_steps,
            model[0],
            auto_cast_to_device=True,
            break_when_any_done=True,
        )
        print(eval_rollout["next", "reward"].sum(-2).shape)
        eval_reward = eval_rollout["next", "reward"].sum(-2).mean().item()
        print(eval_reward)

    
    collector.shutdown()
    if not eval_env.is_closed:
        eval_env.close()
    if not train_env.is_closed:
        train_env.close()
    robot.lease_return()
    #     # Create the RL environment from the robot
    #     spot_env = create_se2_env("spot", robot, transform_state_dict=transform_state_dict, device=device)

    #     # Inference on the trained policy
    #     with set_exploration_type(
    #             ExplorationType.DETERMINISTIC
    #         ), torch.no_grad(), timeit("eval"):
                
    #             print("reset")
    #             print("rollout")
    #             rollout = spot_env.rollout(100, model[0], \
    #                 auto_cast_to_device=True,
    #                 break_when_any_done=True,)
    #             eval_reward = rollout["next", "reward"].sum(-2).mean().item()
    #             print(eval_reward)


if __name__ == "__main__":
    main()
