import copy
import glob
import os
import time
from collections import deque

import gym
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from a2c_ppo_acktr import algo, utils
from a2c_ppo_acktr.algo import gail
from a2c_ppo_acktr.arguments import get_args
from a2c_ppo_acktr.envs import make_vec_envs
from a2c_ppo_acktr.model import Policy
from a2c_ppo_acktr.storage import RolloutStorage
from evaluation import evaluate


def main():
    args = get_args()

    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)

    if args.cuda and torch.cuda.is_available() and args.cuda_deterministic:
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True

    log_dir = os.path.expanduser(args.log_dir)
    eval_log_dir = log_dir + "_eval"
    utils.cleanup_log_dir(log_dir)
    utils.cleanup_log_dir(eval_log_dir)

    torch.set_num_threads(1)
    device = torch.device("cuda:0" if args.cuda else "cpu")
    
    # 构建环境
    envs = make_vec_envs(args.env_name, args.seed, args.num_processes,
                         args.gamma, args.log_dir, device, False)

    # 动作、价值预测网络
    # todo 既然动作、价值复用同一个base网络，看看他们是如何训练参数的，会不会互相干扰
    actor_critic = Policy(
        envs.observation_space.shape,
        envs.action_space,
        base_kwargs={'recurrent': args.recurrent_policy})
    actor_critic.to(device)

    # todo 这里构建的网络的作用是啥？
    # todo 后续将这里的代码全部注释补齐
    if args.algo == 'a2c':
        agent = algo.A2C_ACKTR(
            actor_critic,
            args.value_loss_coef,
            args.entropy_coef,
            lr=args.lr,
            eps=args.eps,
            alpha=args.alpha,
            max_grad_norm=args.max_grad_norm)
    elif args.algo == 'ppo':
        agent = algo.PPO(
            actor_critic,
            args.clip_param,
            args.ppo_epoch,
            args.num_mini_batch,
            args.value_loss_coef,
            args.entropy_coef,
            lr=args.lr,
            eps=args.eps,
            max_grad_norm=args.max_grad_norm)
    elif args.algo == 'acktr':
        agent = algo.A2C_ACKTR(
            actor_critic, args.value_loss_coef, args.entropy_coef, acktr=True)

    # 启动gail训练
    if args.gail:
        assert len(envs.observation_space.shape) == 1
        discr = gail.Discriminator(
            envs.observation_space.shape[0] + envs.action_space.shape[0], 100,
            device)
        file_name = os.path.join(
            args.gail_experts_dir, "trajs_{}.pt".format(
                args.env_name.split('-')[0].lower()))
        
        # todo 后续看它的代码
        # 这里的专家数据是直接来自采集的数据，而不是构建一个专家的网络来进行采集的
        expert_dataset = gail.ExpertDataset(
            file_name, num_trajectories=4, subsample_frequency=20)
        # 构建一个数据加载器，来进行gail训练时的mini-batch采样
        drop_last = len(expert_dataset) > args.gail_batch_size
        gail_train_loader = torch.utils.data.DataLoader(
            dataset=expert_dataset,
            batch_size=args.gail_batch_size,
            shuffle=True,
            drop_last=drop_last)
    
    # todo 这个是啥？看起来像是用来存储每一步的交互数据的，后续看看它的代码
    rollouts = RolloutStorage(args.num_steps, args.num_processes,
                              envs.observation_space.shape, envs.action_space,
                              actor_critic.recurrent_hidden_state_size)

    obs = envs.reset()
    rollouts.obs[0].copy_(obs)
    rollouts.to(device)

    episode_rewards = deque(maxlen=10) # 用来记录最近10个episode的奖励情况，后续在日志中输出这些奖励的统计信息

    start = time.time()
    # num_env_steps是总的交互步数，num_steps是每个环境交互的步数，num_processes是环境的数量，所以这里的num_updates就是总的更新次数
    num_updates = int(
        args.num_env_steps) // args.num_steps // args.num_processes
    for j in range(num_updates): # 总的训练更新次数

        if args.use_linear_lr_decay:
            # 这个应该就是学习率线性衰减的实现，随着训练的进行，学习率会逐渐降低，直到训练结束时降为0
            # todo 后续看看这个函数的实现，看看它是如何实现学习率线性衰减的
            # decrease learning rate linearly
            utils.update_linear_schedule(
                agent.optimizer, j, num_updates,
                agent.optimizer.lr if args.algo == "acktr" else args.lr)

        # 开始进行一次和环境的交互，交互的步数是args.num_steps
        for step in range(args.num_steps):
            # Sample actions
            with torch.no_grad():
                # 这里的recurrent_hidden_states是啥作用的，是用来针对序列的观察环境，可能对于前后关系大的环境来说，使用一个recurrent policy能够更好地捕捉前后关系的信息，从而做出更好的决策，所以这里的recurrent_hidden_states就是用来存储每个环境当前的hidden state的最新状态的，如果没有使用recurrent policy，那么这里的hidden state就是0
                # 输入状态预测动作和价值，同时更新rnn的hidden state
                value, action, action_log_prob, recurrent_hidden_states = actor_critic.act(
                    rollouts.obs[step], rollouts.recurrent_hidden_states[step],
                    rollouts.masks[step])

            # Obser reward and next obs
            obs, reward, done, infos = envs.step(action)

            for info in infos:
                # 看来环境中返回的info中包含了每个episode的奖励信息，这里是把这些奖励信息记录下来，后续在日志中输出这些奖励的统计信息
                if 'episode' in info.keys():
                    episode_rewards.append(info['episode']['r'])

            # If done then clean the history of observations.
            # 根据并行环境返回的done，构建masks，来标记当前的obs是否是一个新的episode的开始，如果是一个新的episode的开始，那么这个mask就是0，否则就是1，
            # 这样在后续进行训练的时候，就能够根据这个mask来正确地处理每个环境的hidden state了，如果是一个新的episode的开始，那么这个mask就是0，
            # 那么在进行训练的时候，hidden state就会被重置为0，如果不是一个新的episode的开始，那么这个mask就是1，那么在进行训练的时候，hidden state就会保持上一次的状态继续传播下去，这样就能够正确地处理每个环境的hidden state了
            masks = torch.FloatTensor(
                [[0.0] if done_ else [1.0] for done_ in done])
            # bad_masks 是用来标记当前的obs是否是一个新的episode的开始，如果是一个新的episode的开始，那么这个bad_mask就是0，否则就是1，这个和上面的masks是一样的，都是用来标记当前的obs是否是一个新的episode的开始的，只不过这个bad_masks是用来处理一些特殊情况的，比如说环境中可能会有一些特殊的状态，这些状态虽然不是一个新的episode的开始，但是在这些状态下，环境可能会返回一些特殊的信息，这些信息可能会对训练造成干扰，所以这里就使用了bad_masks来标记这些特殊状态，这样在进行训练的时候，就能够根据这个bad_masks来正确地处理这些特殊状态了，如果是一个特殊状态，那么这个bad_mask就是0，那么在进行训练的时候，hidden state就会被重置为0，如果不是一个特殊状态，那么这个bad_mask就是1，那么在进行训练的时候，hidden state就会保持上一次的状态继续传播下去，这样就能够正确地处理这些特殊状态了
            # todo 后续去二呢是怎么用的
            bad_masks = torch.FloatTensor(
                [[0.0] if 'bad_transition' in info.keys() else [1.0]
                 for info in infos])
            # 将采集的数据存储到rollouts中，后续会使用这些数据来进行训练
            rollouts.insert(obs, recurrent_hidden_states, action,
                            action_log_prob, value, reward, masks, bad_masks)

        with torch.no_grad():
            # 因为有最大步数的限制，如果因为最大步数限制而结束
            # 那么最后一步的时候游戏可能是没有结束，那么需要对这部分的游戏的obs计算一下价值预测，来进行训练时的回报计算
            # 不能直接等同为0，这样会让游戏学习撞墙
            next_value = actor_critic.get_value(
                rollouts.obs[-1], rollouts.recurrent_hidden_states[-1],
                rollouts.masks[-1]).detach()

        if args.gail:
            # 如果开启了gail
            if j >= 10: # 前10次对判断器进行预热训练，完成后则开始正式的gail训练，预热训练关闭
                envs.venv.eval()

            gail_epoch = args.gail_epoch
            if j < 10:
                gail_epoch = 100  # Warm up
            for _ in range(gail_epoch):
                discr.update(gail_train_loader, rollouts,
                             utils.get_vec_normalize(envs)._obfilt)

            for step in range(args.num_steps):
                rollouts.rewards[step] = discr.predict_reward(
                    rollouts.obs[step], rollouts.actions[step], args.gamma,
                    rollouts.masks[step])

        rollouts.compute_returns(next_value, args.use_gae, args.gamma,
                                 args.gae_lambda, args.use_proper_time_limits)

        value_loss, action_loss, dist_entropy = agent.update(rollouts)

        rollouts.after_update()

        # save for every interval-th episode or for the last epoch
        if (j % args.save_interval == 0
                or j == num_updates - 1) and args.save_dir != "":
            save_path = os.path.join(args.save_dir, args.algo)
            try:
                os.makedirs(save_path)
            except OSError:
                pass

            torch.save([
                actor_critic,
                getattr(utils.get_vec_normalize(envs), 'obs_rms', None)
            ], os.path.join(save_path, args.env_name + ".pt"))

        if j % args.log_interval == 0 and len(episode_rewards) > 1:
            total_num_steps = (j + 1) * args.num_processes * args.num_steps
            end = time.time()
            print(
                "Updates {}, num timesteps {}, FPS {} \n Last {} training episodes: mean/median reward {:.1f}/{:.1f}, min/max reward {:.1f}/{:.1f}\n"
                .format(j, total_num_steps,
                        int(total_num_steps / (end - start)),
                        len(episode_rewards), np.mean(episode_rewards),
                        np.median(episode_rewards), np.min(episode_rewards),
                        np.max(episode_rewards), dist_entropy, value_loss,
                        action_loss))

        if (args.eval_interval is not None and len(episode_rewards) > 1
                and j % args.eval_interval == 0):
            obs_rms = utils.get_vec_normalize(envs).obs_rms
            evaluate(actor_critic, obs_rms, args.env_name, args.seed,
                     args.num_processes, eval_log_dir, device)


if __name__ == "__main__":
    main()
