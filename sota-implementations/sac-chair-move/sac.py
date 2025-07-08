# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
"""TD3 Example.

This is a simple self-contained example of a TD3 training script.

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
from tensordict.nn import CudaGraphModule
from torchrl._utils import compile_with_warmup, timeit
from torchrl.envs.utils import ExplorationType, set_exploration_type
from torchrl.record.loggers import generate_exp_name, get_logger
from utils import (
    dump_video,
    log_metrics,
    make_collector,
    make_environment,
    make_loss_module,
    make_optimizer,
    make_replay_buffer,
<<<<<<<< HEAD:sota-implementations/td3-se2/td3.py
    make_td3_agent,
    make_prior_collector,
    make_prior_agent
)

torch.set_float32_matmul_precision("high")


========
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
>>>>>>>> 635a154392f9795ef04144fd9d914e194c3a85de:sota-implementations/sac-chair-move/sac.py
@hydra.main(version_base="1.1", config_path="", config_name="config")
def main(cfg: DictConfig):  # noqa: F821
    device = cfg.network.device
    if device in ("", None):
        if torch.cuda.is_available():
            device = torch.device("cuda:0")
        else:
            device = torch.device("cpu")
    else:
        device = torch.device(device)

    # Create logger
    exp_name = generate_exp_name("TD3", cfg.logger.exp_name)
    logger = None
    if cfg.logger.backend:
        logger = get_logger(
            logger_type=cfg.logger.backend,
            logger_name="td3_logging",
            experiment_name=exp_name,
            wandb_kwargs={
                "mode": cfg.logger.mode,
                "config": dict(cfg),
                "project": cfg.logger.project_name,
                "group": cfg.logger.group_name,
            },
        )

    # Set seeds
    torch.manual_seed(cfg.env.seed)
    np.random.seed(cfg.env.seed)

    # Create environments
    train_env, eval_env = make_environment(cfg, logger=logger)

    # Create agent
<<<<<<<< HEAD:sota-implementations/td3-se2/td3.py
    model, exploration_policy = make_td3_agent(cfg, train_env, eval_env, device)
    prior_model = make_prior_agent(cfg, device)

    # Create TD3 loss
========
    model, exploration_policy = make_sac_agent(cfg, train_env, eval_env, device)
    prior_model = make_prior_agent(cfg, device)
    # Create SAC loss
>>>>>>>> 635a154392f9795ef04144fd9d914e194c3a85de:sota-implementations/sac-chair-move/sac.py
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
<<<<<<<< HEAD:sota-implementations/td3-se2/td3.py
        cfg,
        train_env,
        exploration_policy,
        compile_mode=compile_mode,
        device=device,
========
        cfg, train_env, exploration_policy, compile_mode=compile_mode, \
            total_frames=cfg.collector.total_frames
>>>>>>>> 635a154392f9795ef04144fd9d914e194c3a85de:sota-implementations/sac-chair-move/sac.py
    )
    prior_collector = make_prior_collector(
        cfg, train_env, prior_model, compile_mode=compile_mode, \
            total_frames=cfg.collector.prior_frames
    )

    # Create replay buffer
    replay_buffer = make_replay_buffer(
        batch_size=cfg.optim.batch_size,
        prb=cfg.replay_buffer.prb,
        buffer_size=cfg.replay_buffer.size,
        scratch_dir=cfg.replay_buffer.scratch_dir,
        device=device,
        compile=bool(compile_mode),
    )
    prior_replay_buffer = make_replay_buffer(
        batch_size=cfg.optim.batch_size,
        prb=cfg.replay_buffer.prb,
        buffer_size=cfg.replay_buffer.size,
        scratch_dir=cfg.replay_buffer.scratch_dir,
        device=device,
    )
<<<<<<<< HEAD:sota-implementations/td3-se2/td3.py
========
    prior_replay_buffer = make_replay_buffer(
        batch_size=cfg.optim.batch_size,
        prb=cfg.replay_buffer.prb,
        buffer_size=cfg.replay_buffer.size,
        scratch_dir=cfg.replay_buffer.scratch_dir,
        device=device,
    )
>>>>>>>> 635a154392f9795ef04144fd9d914e194c3a85de:sota-implementations/sac-chair-move/sac.py
    # Create optimizers
    optimizer_actor, optimizer_critic = make_optimizer(cfg, loss_module)

    prb = cfg.replay_buffer.prb

    def update(sampled_tensordict, update_actor, prb=prb):

        # Compute loss
        q_loss, *_ = loss_module.value_loss(sampled_tensordict)

        # Update critic
        q_loss.backward()
        optimizer_critic.step()
        optimizer_critic.zero_grad(set_to_none=True)

        # Update actor
        if update_actor:
            actor_loss, *_ = loss_module.actor_loss(sampled_tensordict)

            actor_loss.backward()
            optimizer_actor.step()
            optimizer_actor.zero_grad(set_to_none=True)

            # Update target params
            target_net_updater.step()
        else:
            actor_loss = q_loss.new_zeros(())

        return q_loss.detach(), actor_loss.detach()

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
    delayed_updates = cfg.optim.policy_update_delay
    eval_rollout_steps = cfg.env.max_episode_steps
    eval_iter = cfg.logger.eval_iter
    frames_per_batch = cfg.collector.frames_per_batch
    update_counter = 0

    collector_iter = iter(collector)
    total_iter = len(collector)

    prior_collector_iter = iter(prior_collector)
    total_prior_iter = len(prior_collector)
<<<<<<<< HEAD:sota-implementations/td3-se2/td3.py

========
>>>>>>>> 635a154392f9795ef04144fd9d914e194c3a85de:sota-implementations/sac-chair-move/sac.py
    '''
    Collect offline data
    '''
    for i in range(total_prior_iter):
        timeit.printevery(num_prints=1000, total_count=total_iter, erase=True)
        print("***Collecting offline data***")
        with timeit("offline dataset"):
            tensordict = next(prior_collector_iter)
            # i = np.random.randint(tensordict.shape[0])
            # j = 2
            # print("===Test===")
            # print([i, j])
            # print(tensordict[i, j]["action"])
            # print(tensordict[i, j]["x"])
            # print(tensordict[i, j]["y"])
            # print(tensordict[i, j]["theta"])
            # print(tensordict[i, j]["next", "x"])
            # print(tensordict[i, j]["next", "y"])
            # print(tensordict[i, j]["next", "theta"])
            # print(tensordict[i, j]["next", "reward"])
            # assert False
        with timeit("rb - extend"):
            tensordict = tensordict.reshape(-1)
            prior_replay_buffer.extend(tensordict)
<<<<<<<< HEAD:sota-implementations/td3-se2/td3.py
    
    for _ in range(total_iter):
========

    for i in range(total_iter):
>>>>>>>> 635a154392f9795ef04144fd9d914e194c3a85de:sota-implementations/sac-chair-move/sac.py
        timeit.printevery(num_prints=1000, total_count=total_iter, erase=True)

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

        with timeit("train"):
            # Optimization steps
            if collected_frames >= init_random_frames:
<<<<<<<< HEAD:sota-implementations/td3-se2/td3.py
                (
                    actor_losses,
                    q_losses,
                ) = ([], [])
                for _ in range(num_updates):
                    # Update actor every delayed_updates
                    update_counter += 1
                    update_actor = update_counter % delayed_updates == 0

========
                losses = TensorDict(batch_size=[num_updates])
                for i in range(num_updates):
>>>>>>>> 635a154392f9795ef04144fd9d914e194c3a85de:sota-implementations/sac-chair-move/sac.py
                    with timeit("rb - sample"):
                        # Sample from replay buffer & the prior replay buffer
                        # (50/50 according to RLPD)
                        sampled_tensordict = replay_buffer.sample()
<<<<<<<< HEAD:sota-implementations/td3-se2/td3.py
                        del sampled_tensordict["param"]
                        # del sampled_tensordict["loc"]
                        sampled_tensordict_prior = prior_replay_buffer.sample()
                        sampled_tensordict = torch.cat(
                            (sampled_tensordict, sampled_tensordict_prior), dim=0
                        )
========
                        # del sampled_tensordict["scale"]
                        # del sampled_tensordict["loc"]
                        # sampled_tensordict_prior = prior_replay_buffer.sample()
                        # sampled_tensordict = torch.cat(
                        #     (sampled_tensordict, sampled_tensordict_prior), dim=0
                        # )
>>>>>>>> 635a154392f9795ef04144fd9d914e194c3a85de:sota-implementations/sac-chair-move/sac.py
                    with timeit("update"):
                        torch.compiler.cudagraph_mark_step_begin()
                        q_loss, actor_loss = update(sampled_tensordict, update_actor)

                    # Update priority
                    if prb:
                        with timeit("rb - priority"):
                            replay_buffer.update_priority(sampled_tensordict)

                    q_losses.append(q_loss.clone())
                    if update_actor:
                        actor_losses.append(actor_loss.clone())

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
            metrics_to_log["train/reward"] = episode_rewards.mean()
            metrics_to_log["train/episode_length"] = episode_length.sum() / len(
                episode_length
            )

        if collected_frames >= init_random_frames:
            metrics_to_log["train/q_loss"] = torch.stack(q_losses).mean()
            if update_actor:
                metrics_to_log["train/a_loss"] = torch.stack(actor_losses).mean()

        # Evaluation
        if abs(collected_frames % eval_iter) < frames_per_batch:
            with set_exploration_type(
                ExplorationType.DETERMINISTIC
            ), torch.no_grad(), timeit("eval"):
                eval_rollout = eval_env.rollout(
                    eval_rollout_steps,
                    exploration_policy,
                    auto_cast_to_device=True,
                    break_when_any_done=True,
                )
                eval_env.apply(dump_video)
                eval_reward = eval_rollout["next", "reward"].sum(-2).mean().item()
                metrics_to_log["eval/reward"] = eval_reward
        if logger is not None:
            metrics_to_log.update(timeit.todict(prefix="time"))
            metrics_to_log["time/speed"] = pbar.format_dict["rate"]
            log_metrics(logger, metrics_to_log, collected_frames)
    ## Final evaluation

    with set_exploration_type(
        ExplorationType.DETERMINISTIC
    ), torch.no_grad(), timeit("One final eval"):
        eval_rollout = eval_env.rollout(
            eval_rollout_steps,
            model[0],
            auto_cast_to_device=True,
            break_when_any_done=True,
        )
        print(eval_rollout["next", "reward"].sum(-2).shape)
        print(eval_rollout["next", "reward"].shape)
        eval_reward = eval_rollout["next", "reward"].sum(-2).mean().item()
        print(eval_reward)

    collector.shutdown()
    if not eval_env.is_closed:
        eval_env.close()
    if not train_env.is_closed:
        train_env.close()


<<<<<<<< HEAD:sota-implementations/td3-se2/td3.py
========
    ## Transfer the trained policy to real robot
    # A dummy safe module
    td_module_hidden = SafeModule(
        module = torch.nn.Identity(),
        in_keys=["observation"],
        out_keys=["observation"],
    )
     

    robot = SPOT(cfg)
    with robot.lease_alive():
        robot.power_on_stand()
        # Create the RL environment from the robot
        spot_env = create_se2_env("spot", robot, device=device)

        # Inference on the trained policy
        with set_exploration_type(
                ExplorationType.DETERMINISTIC
            ), torch.no_grad(), timeit("eval"):
                
                print("reset")
                print("rollout")
                rollout = spot_env.rollout(100, model[0], \
                    auto_cast_to_device=True,
                    break_when_any_done=True,)
                eval_reward = rollout["next", "reward"].sum(-2).mean().item()
                print(eval_reward)


>>>>>>>> 635a154392f9795ef04144fd9d914e194c3a85de:sota-implementations/sac-chair-move/sac.py
if __name__ == "__main__":
    main()
