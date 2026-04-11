import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim


class PPO():
    def __init__(self,
                 actor_critic,
                 clip_param,
                 ppo_epoch,
                 num_mini_batch,
                 value_loss_coef,
                 entropy_coef,
                 lr=None,
                 eps=None,
                 max_grad_norm=None,
                 use_clipped_value_loss=True):
        '''
        actor_critic: 预测动作和价值的网络 
        clip_param: ppo的clip参数
        ppo_epoch: ppo的迭代次数
        num_mini_batch: 每次ppo迭代中，数据被分成多少个mini batch
        value_loss_coef: 价值损失的权重
        entropy_coef: 熵损失的权重
        lr: 学习率
        eps: adam优化器的eps参数
        max_grad_norm: 梯度裁剪的最大值
        use_clipped_value_loss: 是否使用clip的方式来计算价值损失
        '''

        self.actor_critic = actor_critic

        self.clip_param = clip_param
        self.ppo_epoch = ppo_epoch
        self.num_mini_batch = num_mini_batch

        self.value_loss_coef = value_loss_coef
        self.entropy_coef = entropy_coef

        self.max_grad_norm = max_grad_norm
        self.use_clipped_value_loss = use_clipped_value_loss

        self.optimizer = optim.Adam(actor_critic.parameters(), lr=lr, eps=eps)

    def update(self, rollouts):
        '''
        rollouts: 存储采集的数据的对象，包含了obs、actions、rewards、masks、value_preds等信息
        这个函数的作用是根据采集的数据来更新actor_critic网络的参数
        '''
        # 用计算得到的回报 - 价值预测来计算优势函数，并进行归一化处理
        advantages = rollouts.returns[:-1] - rollouts.value_preds[:-1]
        advantages = (advantages - advantages.mean()) / (
            advantages.std() + 1e-5)

        value_loss_epoch = 0
        action_loss_epoch = 0
        dist_entropy_epoch = 0

        for e in range(self.ppo_epoch):
            # 循环神经网络和非循环神经网络的处理方式不同
            # 生成数据生成器，根据ppo训练的设置来生成mini-batch的数据
            if self.actor_critic.is_recurrent:
                data_generator = rollouts.recurrent_generator(
                    advantages, self.num_mini_batch)
            else:
                data_generator = rollouts.feed_forward_generator(
                    advantages, self.num_mini_batch)

            for sample in data_generator:
                # 遍历每一个mini-batch的数据，来进行ppo训练
                obs_batch, recurrent_hidden_states_batch, actions_batch, \
                   value_preds_batch, return_batch, masks_batch, old_action_log_probs_batch, \
                        adv_targ = sample

                # Reshape to do in a single forward pass for all steps
                values, action_log_probs, dist_entropy, _ = self.actor_critic.evaluate_actions(
                    obs_batch, recurrent_hidden_states_batch, masks_batch,
                    actions_batch)

                # 计算ppo的损失函数，来进行反向传播和优化
                ratio = torch.exp(action_log_probs -
                                  old_action_log_probs_batch)
                surr1 = ratio * adv_targ
                surr2 = torch.clamp(ratio, 1.0 - self.clip_param,
                                    1.0 + self.clip_param) * adv_targ
                action_loss = -torch.min(surr1, surr2).mean()

                # 计算价值损失，使用clip的方式来计算价值损失，来避免价值函数过拟合的问题
                if self.use_clipped_value_loss:
                    # 避免新的价值预测和旧的价值预测之间的差距过大，来避免价值函数过拟合的问题
                    value_pred_clipped = value_preds_batch + \
                        (values - value_preds_batch).clamp(-self.clip_param, self.clip_param)
                    value_losses = (values - return_batch).pow(2) # 直接用新的价值预测来计算损失
                    value_losses_clipped = (
                        value_pred_clipped - return_batch).pow(2) # 用clip的方式来计算损失，来避免新的价值预测和旧的价值预测之间的差距过大，来避免价值函数过拟合的问题
                    # 这里选择更大的losses，相当于用更远的预测来计算损失，来避免价值函数过拟合的问题
                    # 这样参数更新的会更慢点，从更远的地方更新也避免了局部最优的问题，来避免价值函数过拟合的问题
                    value_loss = 0.5 * torch.max(value_losses,
                                                 value_losses_clipped).mean() # 取两者的最大值来计算损失，来避免新的价值预测和旧的价值预测之间的差距过大，来避免价值函数过拟合的问题
                else:
                    # 不使用裁剪的方式来计算价值损失，直接用新的价值预测来计算损失
                    value_loss = 0.5 * (return_batch - values).pow(2).mean()

                # 然后汇总所有的损失，来进行反向传播和优化
                self.optimizer.zero_grad()
                (value_loss * self.value_loss_coef + action_loss -
                 dist_entropy * self.entropy_coef).backward()
                nn.utils.clip_grad_norm_(self.actor_critic.parameters(),
                                         self.max_grad_norm)
                self.optimizer.step()

                value_loss_epoch += value_loss.item()
                action_loss_epoch += action_loss.item()
                dist_entropy_epoch += dist_entropy.item()
        
        
        num_updates = self.ppo_epoch * self.num_mini_batch

        value_loss_epoch /= num_updates
        action_loss_epoch /= num_updates
        dist_entropy_epoch /= num_updates

        return value_loss_epoch, action_loss_epoch, dist_entropy_epoch
