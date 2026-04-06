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
        expoert_state: 专家数据的状态，shape是(batch_size, state_dim)
        expert_action: 专家数据的动作，shape是(batch_size, action_dim)
        policy_state: 策略数据的状态，shape是(batch_size, state_dim)
        policy_action: 策略数据的动作，shape是(batch_size, action_dim)
        lambda_: 梯度惩罚的权重，默认值是10
        '''
        alpha = torch.rand(expert_state.size(0), 1) # 随机值，shape是(batch_size, 1)，todo 作用
        expert_data = torch.cat([expert_state, expert_action], dim=1)
        policy_data = torch.cat([policy_state, policy_action], dim=1)

        alpha = alpha.expand_as(expert_data).to(expert_data.device)

        # todo 就这么随机的融合专家数据和策略数据来进行训练吗？后续看看它的代码
        mixup_data = alpha * expert_data + (1 - alpha) * policy_data
        mixup_data.requires_grad = True

        # 进行判别预测
        disc = self.trunk(mixup_data)
        ones = torch.ones(disc.size()).to(disc.device)
        grad = autograd.grad(
            outputs=disc,
            inputs=mixup_data,
            grad_outputs=ones,
            create_graph=True,
            retain_graph=True,
            only_inputs=True)[0]

        grad_pen = lambda_ * (grad.norm(2, dim=1) - 1).pow(2).mean()
        return grad_pen

    def update(self, expert_loader, rollouts, obsfilt=None):
        '''
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

            loss += (gail_loss + grad_pen).item()
            n += 1

            self.optimizer.zero_grad()
            (gail_loss + grad_pen).backward()
            self.optimizer.step()
        return loss / n

    def predict_reward(self, state, action, gamma, masks, update_rms=True):
        with torch.no_grad():
            self.eval()
            d = self.trunk(torch.cat([state, action], dim=1))
            s = torch.sigmoid(d)
            reward = s.log() - (1 - s).log()
            if self.returns is None:
                self.returns = reward.clone()

            if update_rms:
                self.returns = self.returns * masks * gamma + reward
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
