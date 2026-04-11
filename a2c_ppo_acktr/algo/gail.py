import h5py
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.data
from torch import autograd

from stable_baselines3.common.running_mean_std import RunningMeanStd

class Discriminator(nn.Module):
    def __init__(self, input_dim, hidden_dim, device):
        '''
        判别器，输入是state和action的拼接，输出是一个标量，表示当前的state-action对是来自专家还是来自策略

        input_dim: state_dim + action_dim
        hidden_dim: discriminator网络的隐藏层维度
        device: discriminator网络的计算设备
        '''
        super(Discriminator, self).__init__()

        self.device = device

        self.trunk = nn.Sequential(
            nn.Linear(input_dim, hidden_dim), nn.Tanh(),
            nn.Linear(hidden_dim, hidden_dim), nn.Tanh(),
            nn.Linear(hidden_dim, 1)).to(device)

        self.trunk.train()

        self.optimizer = torch.optim.Adam(self.trunk.parameters())

        self.returns = None
        self.ret_rms = RunningMeanStd(shape=())

    def compute_grad_pen(self,
                         expert_state,
                         expert_action,
                         policy_state,
                         policy_action,
                         lambda_=10):
        '''
        todo Lipschitz连续性
这个函数实现了 WGAN-GP（梯度惩罚） 中的梯度惩罚项，用于在GAIL（生成对抗模仿学习）中训练判别器时，强制判别器满足 1-Lipschitz 连续性约束。

1. 这是什么
这是判别器（Discriminator）的一个方法，用于计算梯度惩罚（gradient penalty）。它的输入是专家数据（state-action对）和策略数据（state-action对），输出是一个标量惩罚值，这个惩罚值会被加到判别器的损失函数中。

2. 为什么需要它
在原始的WGAN和GAIL中，为了确保训练的稳定性，需要判别器（或者叫critic）满足 Lipschitz连续性（即函数的输出变化不能太快）。WGAN-GP论文提出：与其通过权重裁剪（weight clipping）来强制Lipschitz约束（这样会带来各种问题），不如直接在损失函数中添加一个梯度惩罚项。

梯度惩罚的核心思想：在真实数据分布和生成数据分布之间的所有点（即两个分布连线上采样），判别器的梯度范数（gradient norm）应该接近1。

        expoert_state: 专家数据的状态，shape是(batch_size, state_dim)
        expert_action: 专家数据的动作，shape是(batch_size, action_dim)
        policy_state: 策略数据的状态，shape是(batch_size, state_dim)
        policy_action: 策略数据的动作，shape是(batch_size, action_dim)
        lambda_: 梯度惩罚的权重，默认值是10
        '''
        # 这是为了在专家数据分布和策略数据分布之间进行随机线性插值。WGAN-GP论文证明，在这种插值点处施加梯度惩罚，可以保证在整个数据空间中满足Lipschitz约束。
        alpha = torch.rand(expert_state.size(0), 1) # 随机值，shape是(batch_size, 1)，todo 作用 表示每个样本都有一个独立的随机系数
        # 将专家数据和策略数据各自进行拼接
        expert_data = torch.cat([expert_state, expert_action], dim=1)
        policy_data = torch.cat([policy_state, policy_action], dim=1)

        # 扩展alpha的维度，使其能够和expert_data进行逐元素乘法，得到一个随机线性插值的数据点
        alpha = alpha.expand_as(expert_data).to(expert_data.device)

        # todo 就这么随机的融合专家数据和策略数据来进行训练吗？后续看看它的代码
        # 这是关键的一步：在专家数据和策略数据之间进行随机线性插值
        '''
        当 alpha=0 时，得到纯策略数据
        当 alpha=1 时，得到纯专家数据
        当 alpha=0.5 时，得到两者中间点
        是的，就是这样随机的融合专家数据和策略数据来进行训练。这不是随便想的，而是WGAN-GP论文证明的有效方法。通过在专家数据和策略数据之间的随机点施加梯度惩罚，可以确保判别器在整个数据空间中满足Lipschitz连续性约束，从而提高训练的稳定性和效果。
        '''
        mixup_data = alpha * expert_data + (1 - alpha) * policy_data
        mixup_data.requires_grad = True

        # 进行判别预测
        disc = self.trunk(mixup_data)
        ones = torch.ones(disc.size()).to(disc.device) # 判别器预测预期的值
        grad = autograd.grad(
            outputs=disc,
            inputs=mixup_data,
            grad_outputs=ones, # 表示对每个输出分量求导的权重都是1 todo
            create_graph=True, # 留计算图，因为后续还要对梯度惩罚求导
            retain_graph=True, # 保留计算图，因为后续还要对梯度惩罚求导
            only_inputs=True)[0]

        '''
        grad.norm(2, dim=1)：计算每个样本的梯度L2范数
        - 1：我们希望梯度范数接近1（1-Lipschitz约束）WGAN-GP 的理论证明：只要约束专家数据和策略数据之间连线上的点满足梯度范数=1，就能近似保证整个空间的 1-Lipschitz 约束。
        .pow(2)：平方，惩罚偏离1的情况
        .mean()：对所有样本取平均
        lambda_ * ...：乘以权重系数（默认10），控制惩罚的强度

        感觉单纯就是将策略数据混进去，然后然后强制让判别器去认为这都是专家数据，轻微的噪音混淆，从而
        避免了判别器过于强大，导致训练不稳定的问题了
        '''
        grad_pen = lambda_ * (grad.norm(2, dim=1) - 1).pow(2).mean()
        return grad_pen

    def update(self, expert_loader, rollouts, obsfilt=None):
        '''
        更新判别器的参数

        expert_loader: 专家数据的dataloader，提供专家数据的mini-batch
        rollouts: 策略交互数据的存储对象，提供策略数据的mini-batch，这里的数据来自待训练的网络的交互数据，后续会使用这些数据来进行训练
        obsfilt: 用来对专家数据的状态进行过滤的函数，主要是用来对专家数据的状态进行归一化处理的，
        后续在训练的时候会使用这个函数来对专家数据的状态进行归一化处理，这样就能够让专家数据的状态和策略数据的状态在同一个尺度上进行训练了，
        如果没有这个函数的话，那么专家数据的状态和策略数据的状态可能在不同的尺度上进行训练，这样就会导致训练的效果不好了
        '''
        self.train()

        # 采集样本，这是待训练网络的交互数据，后续会使用这些数据来进行训练
        policy_data_generator = rollouts.feed_forward_generator(
            None, mini_batch_size=expert_loader.batch_size)

        loss = 0
        n = 0
        for expert_batch, policy_batch in zip(expert_loader,
                                              policy_data_generator):
            # 专家数据和待训练的交互数据
            policy_state, policy_action = policy_batch[0], policy_batch[2]
            # 根据交互数据的状态和动作，计算预测的判别结果
            policy_d = self.trunk(
                torch.cat([policy_state, policy_action], dim=1))

            # 状态数据的状态和动作
            expert_state, expert_action = expert_batch
            expert_state = obsfilt(expert_state.numpy(), update=False) # todo 对观察进行归一化处理，具体是如何归一化的
            expert_state = torch.FloatTensor(expert_state).to(self.device)
            expert_action = expert_action.to(self.device)
            # 判别器对专家数据进行判断
            expert_d = self.trunk(
                torch.cat([expert_state, expert_action], dim=1))

            # 计算判别器的损失，使用二分类交叉熵损失函数，专家数据的标签是1，待训练的交互数据的标签是0
            expert_loss = F.binary_cross_entropy_with_logits(
                expert_d,
                torch.ones(expert_d.size()).to(self.device))
            policy_loss = F.binary_cross_entropy_with_logits(
                policy_d,
                torch.zeros(policy_d.size()).to(self.device))

            gail_loss = expert_loss + policy_loss
            grad_pen = self.compute_grad_pen(expert_state, expert_action,
                                             policy_state, policy_action)

            # 将梯度惩罚加到损失函数中，进行反向传播和优化
            loss += (gail_loss + grad_pen).item()
            n += 1

            self.optimizer.zero_grad()
            (gail_loss + grad_pen).backward()
            self.optimizer.step()
        return loss / n

    def predict_reward(self, state, action, gamma, masks, update_rms=True):
        '''
        这个函数就是计算奖励的函数了，输入是当前的状态和动作，以及一些参数，输出是一个标量，表示当前的状态和动作的奖励值

        state: 当前的状态，shape是(batch_size, state_dim) 传入的是策略数据的状态
        action: 当前的动作，shape是(batch_size, action_dim)
        gamma: 折扣因子，用于计算回报的折扣
        masks: 用来标记当前的状态是否是一个新的episode的开始的mask，如果是一个新的episode的开始，那么这个mask就是0，否则就是1，这个和上面的masks是一样的
        update_rms: 是否更新返回值的均值和方差，用于归一化处理，默认值是True
        '''
        with torch.no_grad():
            self.eval()
            # 判断当前的state-action对是来自专家还是来自策略，输出一个标量，表示当前的state-action对是来自专家还是来自策略的概率
            d = self.trunk(torch.cat([state, action], dim=1)) 
            s = torch.sigmoid(d) # 如果是策略数据的状态那么这里的s应该是一个比较小的值，如果是专家数据的状态那么这里的s应该是一个比较大的值了
            # 奖励计算（核心转换）
            reward = s.log() - (1 - s).log() # 如果专家数据则是是一个比较大的值，如果是策略数据则是一个比较小的值，todo 后续看看如何使用这个值
            # 如果是专家数据，那么奖励是高的；如果是策略数据，那么奖励是低的；这样就能够引导策略去模仿专家了 todo
            if self.returns is None:
                self.returns = reward.clone()

            if update_rms:
                self.returns = self.returns * masks * gamma + reward # 计算回报，感觉这里是类似累积一个回合的总奖励的方式
                self.ret_rms.update(self.returns.cpu().numpy())

            return reward / np.sqrt(self.ret_rms.var[0] + 1e-8)


class ExpertDataset(torch.utils.data.Dataset):
    def __init__(self, file_name, num_trajectories=4, subsample_frequency=20):
        all_trajectories = torch.load(file_name)
        
        perm = torch.randperm(all_trajectories['states'].size(0))
        idx = perm[:num_trajectories]

        self.trajectories = {}
        
        # See https://github.com/pytorch/pytorch/issues/14886
        # .long() for fixing bug in torch v0.4.1
        start_idx = torch.randint(
            0, subsample_frequency, size=(num_trajectories, )).long()

        for k, v in all_trajectories.items():
            data = v[idx]

            if k != 'lengths':
                samples = []
                for i in range(num_trajectories):
                    samples.append(data[i, start_idx[i]::subsample_frequency])
                self.trajectories[k] = torch.stack(samples)
            else:
                self.trajectories[k] = data // subsample_frequency

        self.i2traj_idx = {}
        self.i2i = {}
        
        self.length = self.trajectories['lengths'].sum().item()

        traj_idx = 0
        i = 0

        self.get_idx = []
        
        for j in range(self.length):
            
            while self.trajectories['lengths'][traj_idx].item() <= i:
                i -= self.trajectories['lengths'][traj_idx].item()
                traj_idx += 1

            self.get_idx.append((traj_idx, i))

            i += 1
            
            
    def __len__(self):
        return self.length

    def __getitem__(self, i):
        traj_idx, i = self.get_idx[i]

        return self.trajectories['states'][traj_idx][i], self.trajectories[
            'actions'][traj_idx][i]
