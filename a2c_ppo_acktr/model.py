import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from a2c_ppo_acktr.distributions import Bernoulli, Categorical, DiagGaussian
from a2c_ppo_acktr.utils import init


class Flatten(nn.Module):
    def forward(self, x):
        return x.view(x.size(0), -1)


class Policy(nn.Module):
    def __init__(self, obs_shape, action_space, base=None, base_kwargs=None):
        super(Policy, self).__init__()
        if base_kwargs is None:
            base_kwargs = {}

        # 这里的base，是提取输入的obs特征，同时输出预测的价值
        # 对于动作的预测，则是通过以下独立的分布来进行的
        if base is None:
            if len(obs_shape) == 3:
                base = CNNBase
            elif len(obs_shape) == 1:
                base = MLPBase
            else:
                raise NotImplementedError

        self.base = base(obs_shape[0], **base_kwargs)

        if action_space.__class__.__name__ == "Discrete":
            num_outputs = action_space.n
            self.dist = Categorical(self.base.output_size, num_outputs)
        elif action_space.__class__.__name__ == "Box":
            num_outputs = action_space.shape[0]
            self.dist = DiagGaussian(self.base.output_size, num_outputs)
        elif action_space.__class__.__name__ == "MultiBinary":
            num_outputs = action_space.shape[0]
            self.dist = Bernoulli(self.base.output_size, num_outputs)
        else:
            raise NotImplementedError

    @property
    def is_recurrent(self):
        return self.base.is_recurrent

    @property
    def recurrent_hidden_state_size(self):
        """Size of rnn_hx."""
        return self.base.recurrent_hidden_state_size

    def forward(self, inputs, rnn_hxs, masks):
        raise NotImplementedError

    def act(self, inputs, rnn_hxs, masks, deterministic=False):
        '''
        inputs: 当前的obs
        rnn_hxs: 如果使用了recurrent policy，那么这个就是rnn的hidden state，否则就是0
        masks: 用来标记当前的obs是否是一个新的episode的开始，如果是一个新的episode的开始，那么这个mask就是0，否则就是1
        deterministic: 是否使用确定性的动作，如果是True，那么就使用动作分布的mode作为动作，否则
        '''
        value, actor_features, rnn_hxs = self.base(inputs, rnn_hxs, masks)
        dist = self.dist(actor_features)

        if deterministic:
            action = dist.mode()
        else:
            action = dist.sample()

        action_log_probs = dist.log_probs(action)
        dist_entropy = dist.entropy().mean()

        return value, action, action_log_probs, rnn_hxs

    def get_value(self, inputs, rnn_hxs, masks):
        value, _, _ = self.base(inputs, rnn_hxs, masks)
        return value

    def evaluate_actions(self, inputs, rnn_hxs, masks, action):
        value, actor_features, rnn_hxs = self.base(inputs, rnn_hxs, masks)
        dist = self.dist(actor_features)

        action_log_probs = dist.log_probs(action)
        dist_entropy = dist.entropy().mean()

        return value, action_log_probs, dist_entropy, rnn_hxs


class NNBase(nn.Module):
    def __init__(self, recurrent, recurrent_input_size, hidden_size):
        '''
        recurrent: 是否使用recurrent policy
        recurrent_input_size: 如果使用了recurrent policy，那么这个就是rnn的输入维度，否则就是0
        hidden_size: rnn的hidden state的维度
        '''
        super(NNBase, self).__init__()

        self._hidden_size = hidden_size
        self._recurrent = recurrent

        if recurrent:
            # 构建GRU网络，输入维度是recurrent_input_size，输出维度是hidden_size
            # GRU的输入是当前的obs特征和上一个时间步的hidden state的拼接，输出是当前时间步的hidden state
            self.gru = nn.GRU(recurrent_input_size, hidden_size)
            # 初始化GRU的权重，偏置项初始化为0，权重矩阵使用orthogonal初始化
            for name, param in self.gru.named_parameters():
                if 'bias' in name:
                    nn.init.constant_(param, 0)
                elif 'weight' in name:
                    nn.init.orthogonal_(param)

    @property
    def is_recurrent(self):
        return self._recurrent

    @property
    def recurrent_hidden_state_size(self):
        if self._recurrent:
            return self._hidden_size
        return 1

    @property
    def output_size(self):
        return self._hidden_size

    def _forward_gru(self, x, hxs, masks):
        '''
        x: 当前的obs特征，经过了obs或者mlp提取后的特征表示
        hxs: 上一个时间步的hidden state，如果是实时交互，这里的hxs是实时更新的，如果是一次性训练，那么这里的hsx是从空状态开始
        masks: 用来标记当前的obs是否是一个新的episode的开始，如果是一个新的episode的开始，那么这个mask就是0，否则就是1
        '''
        if x.size(0) == hxs.size(0): # 这里应该是针对只有一个环境的情况，如果只有一个环境，那么就直接进行一次GRU的前向传播就可以了
            # 如果obs是新的episode的开始，那么就将hidden state重置为0，这里是通过masks来实现
            x, hxs = self.gru(x.unsqueeze(0), (hxs * masks).unsqueeze(0))
            x = x.squeeze(0)
            hxs = hxs.squeeze(0)
        else: # 这里是针对有多个环境的时候
            # 其中T代表和环境交互的步数，N代表环境的数量
            # x is a (T, N, -1) tensor that has been flatten to (T * N, -1)
            N = hxs.size(0)
            T = int(x.size(0) / N)

            # unflatten
            x = x.view(T, N, x.size(1)) # (T, N, features)  - 时间步 × 环境数 × 特征

            # Same deal with masks
            masks = masks.view(T, N) # (T, N)            - 时间步 × 环境数
            # 直观理解：把扁平化的数据还原成 (时间, 环境) 的表格形式，方便按时间处理。

            # Let's figure out which steps in the sequence have a zero for any agent
            # We will always assume t=0 has a zero in it as that makes the logic cleaner
            has_zeros = ((masks[1:] == 0.0) # 从 t=1 开始，找出 mask 为 0 的位置（episode 结束后的第一步）
                            .any(dim=-1)  # 只要 N 个环境中有任意一个环境在这一步 mask=0，跳过了 t=0，因为 t=0 往往是训练段的起始（由外部保证需要重置），所以一定为0
                            .nonzero() # 获取这些位置的索引，意味着"只要有任意一个环境在这一步需要重置，就把这一步标记为断点"
                            .squeeze()
                            .cpu())

            # +1 to correct the masks[1:]
            if has_zeros.dim() == 0: # 如果 has_zeros 是一个标量，说明只有一个断点，那么直接把它转换成列表，并且加1来修正索引
                # Deal with scalar
                has_zeros = [has_zeros.item() + 1]
            else:
                # todo has_zeros + 1 是什么意思？后续看看它的代码
                # 说是has_zeros获取的索引是以0为起点的，但是我们需要的是以1为起点的索引，所以需要加1来进行修正，有啥用
                # 大概是用于类似 【2:】这种索引的情况
                has_zeros = (has_zeros + 1).numpy().tolist()

            # add t=0 and t=T to the list
            # 将获取的索引位置加上0和T，0是为了处理起始位置，T是为了处理结束位置，这样就可以把整个序列分成若干段，每段之间的断点就是has_zeros中记录的位置
            # 看来真的是这样处理索引
            has_zeros = [0] + has_zeros + [T]

            hxs = hxs.unsqueeze(0)
            outputs = []
            # 为什么要分段，看md
            for i in range(len(has_zeros) - 1):
                # We can now process steps that don't have any zeros in masks together!
                # This is much faster
                # 获取第一段序列的起始和结束位置，第一段序列是从0到第一个断点的位置，这段序列中没有任何一个环境需要重置，所以可以一起进行GRU的前向传播
                start_idx = has_zeros[i]
                end_idx = has_zeros[i + 1]

                # 这里之所以要分段，是因为gru处理是，每次遇到一个点是游戏的重新开始时，那么传入的隐藏层状态就需要重置为0，如果不分段的话，那么就无法正确地处理这些断点位置的隐藏层状态了，所以只能分段来处理，每段之间的断点位置就是has_zeros中记录的位置
                # 由于hxs会实时更新，所以即使时没有结束的状态，也能够拿到上一次返回的hxs继续传播
                rnn_scores, hxs = self.gru(
                    x[start_idx:end_idx],
                    hxs * masks[start_idx].view(1, -1, 1))

                # 获取每一次分段后的输出结果，这里的结果应该是输出每一个时间步的预测结果
                outputs.append(rnn_scores)

            # assert len(outputs) == T
            # x is a (T, N, -1) tensor
            x = torch.cat(outputs, dim=0) # 将每一步的输出结果拼接起来，得到一个 (T, N, features) 的张量
            # flatten
            x = x.view(T * N, -1) # 将张量重新扁平化成 (T * N, features) 的形状，方便后续的处理
            hxs = hxs.squeeze(0) # 将隐藏层状态的第一维去掉，得到一个 (N, hidden_size) 的张量，表示每个环境当前的隐藏层状态的最新状态

        return x, hxs


class CNNBase(NNBase):
    def __init__(self, num_inputs, recurrent=False, hidden_size=512):
        super(CNNBase, self).__init__(recurrent, hidden_size, hidden_size)

        init_ = lambda m: init(m, nn.init.orthogonal_, lambda x: nn.init.
                               constant_(x, 0), nn.init.calculate_gain('relu'))

        self.main = nn.Sequential(
            init_(nn.Conv2d(num_inputs, 32, 8, stride=4)), nn.ReLU(),
            init_(nn.Conv2d(32, 64, 4, stride=2)), nn.ReLU(),
            init_(nn.Conv2d(64, 32, 3, stride=1)), nn.ReLU(), Flatten(),
            init_(nn.Linear(32 * 7 * 7, hidden_size)), nn.ReLU())

        init_ = lambda m: init(m, nn.init.orthogonal_, lambda x: nn.init.
                               constant_(x, 0))

        self.critic_linear = init_(nn.Linear(hidden_size, 1))

        self.train()

    def forward(self, inputs, rnn_hxs, masks):
        x = self.main(inputs / 255.0) # 如果是一个序列输入后，那么main的输出就是每一个时间步的特征表示，如果是一个单独的输入，那么main的输出就是这个输入的特征表示

        if self.is_recurrent:
            # 返回进一步提取后的特征，以及更新后的hidden state
            x, rnn_hxs = self._forward_gru(x, rnn_hxs, masks)

        # 如果是输入一个序列，那么critic_linear的输出就是每一个时间步的价值预测，如果是输入一个单独的输入，那么critic_linear的输出就是这个输入的价值预测
        return self.critic_linear(x), x, rnn_hxs


class MLPBase(NNBase):
    def __init__(self, num_inputs, recurrent=False, hidden_size=64):
        super(MLPBase, self).__init__(recurrent, num_inputs, hidden_size)

        if recurrent:
            num_inputs = hidden_size

        init_ = lambda m: init(m, nn.init.orthogonal_, lambda x: nn.init.
                               constant_(x, 0), np.sqrt(2))

        self.actor = nn.Sequential(
            init_(nn.Linear(num_inputs, hidden_size)), nn.Tanh(),
            init_(nn.Linear(hidden_size, hidden_size)), nn.Tanh())

        self.critic = nn.Sequential(
            init_(nn.Linear(num_inputs, hidden_size)), nn.Tanh(),
            init_(nn.Linear(hidden_size, hidden_size)), nn.Tanh())

        self.critic_linear = init_(nn.Linear(hidden_size, 1))

        self.train()

    def forward(self, inputs, rnn_hxs, masks):
        x = inputs

        if self.is_recurrent:
            x, rnn_hxs = self._forward_gru(x, rnn_hxs, masks)

        hidden_critic = self.critic(x)
        hidden_actor = self.actor(x)

        return self.critic_linear(hidden_critic), hidden_actor, rnn_hxs
