import torch
from torch.utils.data.sampler import BatchSampler, SubsetRandomSampler


def _flatten_helper(T, N, _tensor):
    return _tensor.view(T * N, *_tensor.size()[2:])


class RolloutStorage(object):
    def __init__(self, num_steps, num_processes, obs_shape, action_space,
                 recurrent_hidden_state_size):
        '''
        num_steps: 每个环境交互的步数
        num_processes: 环境的数量
        obs_shape: 环境的observation的shape
        action_space: 环境的action space
        recurrent_hidden_state_size: 如果使用了recurrent policy，那么这个就是rnn的hidden state的维度，否则就是0
        '''
        self.obs = torch.zeros(num_steps + 1, num_processes, *obs_shape) # 存储和环境交互相关的观察
        self.recurrent_hidden_states = torch.zeros(
            num_steps + 1, num_processes, recurrent_hidden_state_size)
        self.rewards = torch.zeros(num_steps, num_processes, 1)
        self.value_preds = torch.zeros(num_steps + 1, num_processes, 1)
        self.returns = torch.zeros(num_steps + 1, num_processes, 1)
        self.action_log_probs = torch.zeros(num_steps, num_processes, 1)
        if action_space.__class__.__name__ == 'Discrete':
            action_shape = 1
        else:
            action_shape = action_space.shape[0]
        self.actions = torch.zeros(num_steps, num_processes, action_shape)
        if action_space.__class__.__name__ == 'Discrete':
            self.actions = self.actions.long()
        self.masks = torch.ones(num_steps + 1, num_processes, 1)

        # Masks that indicate whether it's a true terminal state
        # or time limit end state
        # bad_masks = 1：表示真实终止状态
        # bad_masks = 0：表示时间限制终止状态
        self.bad_masks = torch.ones(num_steps + 1, num_processes, 1)

        self.num_steps = num_steps
        self.step = 0

    def to(self, device):
        '''
        将存储的所有数据都移动到指定的设备上
        '''
        self.obs = self.obs.to(device)
        self.recurrent_hidden_states = self.recurrent_hidden_states.to(device)
        self.rewards = self.rewards.to(device)
        self.value_preds = self.value_preds.to(device)
        self.returns = self.returns.to(device)
        self.action_log_probs = self.action_log_probs.to(device)
        self.actions = self.actions.to(device)
        self.masks = self.masks.to(device)
        self.bad_masks = self.bad_masks.to(device)

    def insert(self, obs, recurrent_hidden_states, actions, action_log_probs,
               value_preds, rewards, masks, bad_masks):
        self.obs[self.step + 1].copy_(obs)
        self.recurrent_hidden_states[self.step +
                                     1].copy_(recurrent_hidden_states)
        self.actions[self.step].copy_(actions)
        self.action_log_probs[self.step].copy_(action_log_probs)
        self.value_preds[self.step].copy_(value_preds)
        self.rewards[self.step].copy_(rewards)
        self.masks[self.step + 1].copy_(masks)
        self.bad_masks[self.step + 1].copy_(bad_masks)

        self.step = (self.step + 1) % self.num_steps

    def after_update(self):
        '''
        将存储的最后一步的数据复制到第一步的位置上，来为下一次的交互做准备
        '''
        self.obs[0].copy_(self.obs[-1])
        self.recurrent_hidden_states[0].copy_(self.recurrent_hidden_states[-1])
        self.masks[0].copy_(self.masks[-1])
        self.bad_masks[0].copy_(self.bad_masks[-1])

    def compute_returns(self,
                        next_value,
                        use_gae,
                        gamma,
                        gae_lambda,
                        use_proper_time_limits=True):
        '''
        todo 

        next_value: 最后一步的价值预测，用于计算最后一步的回报
        use_gae: 是否使用gae来计算回报，如果使用了gae，那么就会使用gae来计算回报，否则就使用普通的蒙特卡洛的方式来计算回报
        gamma: 折扣因子，用于计算回报的折扣
        gae_lambda: gae的lambda参数，用于计算gae的权重
        use_proper_time_limits: 是否使用proper time limits来计算回报，如果使用了proper time limits，那么就会根据bad_masks来正确地处理那些因为时间限制而结束的episode的回报计算，如果没有使用proper time limits，那么就会把所有因为时间限制而结束的episode都当做正常结束来计算回报
        '''
        # todo 这两个计算回报的方式有什么区别
        if use_proper_time_limits:
            if use_gae: # 如果使用了gae，那么就会使用gae来计算回报，否则就使用普通的蒙特卡洛的方式来计算回报
                # 看来每次采集数据都一定要采集num_steps步
                # 这里将最后一步的预测的价值存储到value_preds的最后一个位置上，来进行gae的计算
                self.value_preds[-1] = next_value
                gae = 0
                for step in reversed(range(self.rewards.size(0))): # 逆序遍历每一步
                    delta = self.rewards[step] + gamma * self.value_preds[
                        step + 1] * self.masks[step +
                                               1] - self.value_preds[step] # ppo中计算gae的核心公式，delta是当前的reward加上下一步的价值预测乘以折扣因子减去当前的价值预测
                    gae = delta + gamma * gae_lambda * self.masks[step + 1] * gae # gae的计算，gae是当前的delta加上下一步的gae乘以折扣因子乘以gae_lambda乘以mask，mask是用来标记当前的状态是否是一个新的episode的开始的，如果是一个新的episode的开始，那么这个mask就是0，否则就是1，这样在计算gae的时候，如果遇到了一个新的episode的开始，那么这个mask就是0，那么在计算gae的时候，gae就会被重置为0，这样就能够正确地处理每个环境的hidden state了，如果不是一个新的episode的开始，那么这个mask就是1，那么在计算gae的时候，gae就会保持上一次的状态继续传播下去，这样就能够正确地处理每个环境的hidden state了
                    gae = gae * self.bad_masks[step + 1] # 如果是超时终止，由于不知道未来的情况，所以最好直接用预测价值即可
                    self.returns[step] = gae + self.value_preds[step] # gae加上当前的价值预测就是当前的回报，这个是ppo中计算gae的核心公式
            else:
                # 这里将最后一步的预测的价值存储到value_preds的最后一个位置上
                self.returns[-1] = next_value
                for step in reversed(range(self.rewards.size(0))):
                    # 这里的回报计算方式是类似累积一个回合的总奖励的方式，当前的回报等于下一步的回报乘以折扣因子加上当前的奖励，这个是蒙特卡洛的方式来计算回报的核心公式
                    # 正常部分（当 bad_masks=1，真实终止）：returns[step] = returns[step+1] * gamma * masks[step+1] + rewards[step]
                    # 时间限制终止部分（当 bad_masks=0）：returns[step] = value_preds[step]
                    '''
                    当遇到时间限制终止时，不使用蒙特卡洛回报
                    而是使用价值函数的预测值 value_preds[step]
                    因为时间限制终止后，实际环境仍在继续，我们不知道后续的真实回报，所以用价值函数的估计来代替
                    '''
                    self.returns[step] = (self.returns[step + 1] * \
                        gamma * self.masks[step + 1] + self.rewards[step]) * self.bad_masks[step + 1] \
                        + (1 - self.bad_masks[step + 1]) * self.value_preds[step] 
        else:
            if use_gae:
                self.value_preds[-1] = next_value
                gae = 0
                for step in reversed(range(self.rewards.size(0))):
                    delta = self.rewards[step] + gamma * self.value_preds[
                        step + 1] * self.masks[step +
                                               1] - self.value_preds[step]
                    gae = delta + gamma * gae_lambda * self.masks[step +
                                                                  1] * gae
                    self.returns[step] = gae + self.value_preds[step]
            else:
                self.returns[-1] = next_value
                for step in reversed(range(self.rewards.size(0))):
                    self.returns[step] = self.returns[step + 1] * \
                        gamma * self.masks[step + 1] + self.rewards[step]

    def feed_forward_generator(self,
                               advantages,
                               num_mini_batch=None,
                               mini_batch_size=None):
        '''
        advantages: 优势函数的值，shape是(num_steps, num_processes)，这个是用来进行ppo训练的，如果ppo训练使用了gae，那么这个advantage就是gae计算出来的优势函数的值，如果ppo训练没有使用gae，那么这个advantage就是returns - value_preds计算出来的优势函数的值
        num_mini_batch: 进行ppo训练的时候，将数据分成多少个mini-batch进行训练，如果没有指定mini_batch_size的话，那么就根据num_mini_batch来计算mini_batch_size，如果两者都没有指定的话，那么就默认使用num_mini_batch=32来计算mini_batch_size
        mini_batch_size: 进行ppo训练的时候，每个mini-batch
        '''
        num_steps, num_processes = self.rewards.size()[0:2]
        batch_size = num_processes * num_steps

        if mini_batch_size is None: # 自动计算合适的mini_batch_size
            assert batch_size >= num_mini_batch, (
                "PPO requires the number of processes ({}) "
                "* number of steps ({}) = {} "
                "to be greater than or equal to the number of PPO mini batches ({})."
                "".format(num_processes, num_steps, num_processes * num_steps,
                          num_mini_batch))
            mini_batch_size = batch_size // num_mini_batch
        sampler = BatchSampler(
            SubsetRandomSampler(range(batch_size)), # 从 indices 中随机采样
            mini_batch_size, # 每批 32 个样本
            drop_last=True) # # 最后不足 32 的丢弃
        for indices in sampler:
            # 根据索引随机采样样本
            obs_batch = self.obs[:-1].view(-1, *self.obs.size()[2:])[indices]
            recurrent_hidden_states_batch = self.recurrent_hidden_states[:-1].view(
                -1, self.recurrent_hidden_states.size(-1))[indices]
            actions_batch = self.actions.view(-1,
                                              self.actions.size(-1))[indices]
            value_preds_batch = self.value_preds[:-1].view(-1, 1)[indices]
            return_batch = self.returns[:-1].view(-1, 1)[indices]
            masks_batch = self.masks[:-1].view(-1, 1)[indices]
            old_action_log_probs_batch = self.action_log_probs.view(-1,
                                                                    1)[indices]
            
            # 如果advantages是None，那么adv_targ也是None，否则就根据索引随机采样优势函数的值
            if advantages is None:
                adv_targ = None
            else:
                adv_targ = advantages.view(-1, 1)[indices]

            yield obs_batch, recurrent_hidden_states_batch, actions_batch, \
                value_preds_batch, return_batch, masks_batch, old_action_log_probs_batch, adv_targ

    def recurrent_generator(self, advantages, num_mini_batch):
        '''
        advantages: 优势函数的值，shape是(num_steps, num_processes)，这个是用来进行ppo训练的，如果ppo训练使用了gae，那么这个advantage就是gae计算出来的优势函数的值，如果ppo训练没有使用gae，那么这个advantage就是returns - value_preds计算出来的优势函数的值
        num_mini_batch: 进行ppo训练的时候，将数据分成多少个mini-batch进行训练，如果没有指定mini_batch_size的话，那么就根据num_mini_batch来计算mini_batch_size，如果两者都没有指定的话，那么就默认使用num_mini_batch=32来计算mini_batch_size
        '''

        num_processes = self.rewards.size(1) # num_processes是环境的数量
        assert num_processes >= num_mini_batch, (
            "PPO requires the number of processes ({}) "
            "to be greater than or equal to the number of "
            "PPO mini batches ({}).".format(num_processes, num_mini_batch))
        num_envs_per_batch = num_processes // num_mini_batch # 每个mini-batch中包含的环境数量
        perm = torch.randperm(num_processes) # 随机打乱环境的索引，来进行随机采样
        for start_ind in range(0, num_processes, num_envs_per_batch):
            # 根据随机打乱的环境索引来采样数据，来构建mini-batch
            # 这里一个索引的数据包含一个环境的所有交互数据
            obs_batch = []
            recurrent_hidden_states_batch = []
            actions_batch = []
            value_preds_batch = []
            return_batch = []
            masks_batch = []
            old_action_log_probs_batch = []
            adv_targ = []

            # 根据随机打乱的环境索引来采样数据，来构建mini-batch
            for offset in range(num_envs_per_batch):
                ind = perm[start_ind + offset]
                obs_batch.append(self.obs[:-1, ind])
                recurrent_hidden_states_batch.append(
                    self.recurrent_hidden_states[0:1, ind])
                actions_batch.append(self.actions[:, ind])
                value_preds_batch.append(self.value_preds[:-1, ind])
                return_batch.append(self.returns[:-1, ind])
                masks_batch.append(self.masks[:-1, ind])
                old_action_log_probs_batch.append(
                    self.action_log_probs[:, ind])
                adv_targ.append(advantages[:, ind])

            T, N = self.num_steps, num_envs_per_batch
            # These are all tensors of size (T, N, -1)
            obs_batch = torch.stack(obs_batch, 1) # shape 是(num_steps, num_envs_per_batch, obs_shape)
            actions_batch = torch.stack(actions_batch, 1) # shape 是(num_steps, num_envs_per_batch, action_shape)
            value_preds_batch = torch.stack(value_preds_batch, 1) # shape 是(num_steps, num_envs_per_batch, 1)
            return_batch = torch.stack(return_batch, 1) # shape 是(num_steps, num_envs_per_batch, 1)
            masks_batch = torch.stack(masks_batch, 1)   # shape 是(num_steps, num_envs_per_batch, 1)
            old_action_log_probs_batch = torch.stack(
                old_action_log_probs_batch, 1) # shape 是(num_steps, num_envs_per_batch, 1)
            adv_targ = torch.stack(adv_targ, 1) # shape 是(num_steps, num_envs_per_batch, 1)

            # States is just a (N, -1) tensor
            recurrent_hidden_states_batch = torch.stack(
                recurrent_hidden_states_batch, 1).view(N, -1)

            # Flatten the (T, N, ...) tensors to (T * N, ...)
            obs_batch = _flatten_helper(T, N, obs_batch) # shape 是(num_steps * num_envs_per_batch, obs_shape)
            actions_batch = _flatten_helper(T, N, actions_batch)
            value_preds_batch = _flatten_helper(T, N, value_preds_batch)
            return_batch = _flatten_helper(T, N, return_batch)
            masks_batch = _flatten_helper(T, N, masks_batch)
            old_action_log_probs_batch = _flatten_helper(T, N, \
                    old_action_log_probs_batch)
            adv_targ = _flatten_helper(T, N, adv_targ)

            yield obs_batch, recurrent_hidden_states_batch, actions_batch, \
                value_preds_batch, return_batch, masks_batch, old_action_log_probs_batch, adv_targ
