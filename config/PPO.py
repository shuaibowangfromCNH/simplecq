import os
os.environ['TORCHDYNAMO_DISABLE'] = '1'
os.environ['PYTORCH_JIT'] = '0'
import numpy as np
import torch
import wandb
import torch.nn as nn
from torch.distributions import MultivariateNormal
from torch.distributions import Categorical
# import gym
try:
    import gymnasium as gym
    print("✓ 使用 Gymnasium")
except ImportError:
    import gym
    print("⚠️ 使用旧版 Gym")

################################## 设备设定 ##################################
print("============================================================================================")
device = torch.device('cpu')
if(torch.cuda.is_available()):
    device = torch.device('cuda')
    torch.cuda.empty_cache() # 
    print("Device set to : " + str(torch.cuda.get_device_name(device)))
else:
    print("Device set to : cpu")
print("============================================================================================")


################################## PPO 算法 ##################################
class RolloutBuffer:
    def __init__(self):
        self.actions = []
        self.states = []
        self.logprobs = []
        self.rewards = []
        self.state_values = []
        self.is_terminals = []

    def clear(self):
        del self.actions[:]
        del self.states[:]
        del self.logprobs[:]
        del self.rewards[:]
        del self.state_values[:]
        del self.is_terminals[:]

class ActorCritic(nn.Module):
    def __init__(self, state_dim, action_dim, is_continuous_action_space, action_std_init):
        
        super(ActorCritic, self).__init__() # 继承父类的所有属性和方法

        self.is_continuous_action_space = is_continuous_action_space

        if is_continuous_action_space:
            self.action_dim = action_dim
            self.action_var = torch.full((action_dim,), action_std_init * action_std_init).to(device)

        # ActorNet
        if is_continuous_action_space:
            self.actor = nn.Sequential(
                nn.Linear(state_dim, 64),
                nn.Tanh(),
                nn.Linear(64, 64),
                nn.Tanh(),
                nn.Linear(64, action_dim),
                nn.Tanh()
            )
        else:
            self.actor = nn.Sequential(
                nn.Linear(state_dim, 64),
                nn.Tanh(),
                nn.Linear(64, 64),
                nn.Tanh(),
                nn.Linear(64, action_dim),
                nn.Softmax(dim=-1)
            )

        # CriticNet
        self.critic = nn.Sequential(
            nn.Linear(state_dim, 64),
            nn.Tanh(),
            nn.Linear(64, 64),
            nn.Tanh(),
            nn.Linear(64, 1)
        )

    def set_action_std(self, new_action_std):
        if self.is_continuous_action_space:
            self.action_var = torch.full((self.action_dim,), new_action_std * new_action_std).to(device)
        else:
            print("--------------------------------------------------------------------------------------------")
            print("WARNING : Calling ActorCritic::set_action_std() on discrete action space policy")
            print("--------------------------------------------------------------------------------------------")

    def forward(self):
        raise NotImplementedError       # 抽象方法定义，意味着子类在调用该父类时必须定义此函数，否则报错

    def take_action(self, state, deterministic=False):
        if self.is_continuous_action_space:
            action_mean = self.actor(state)
            cov_mat = torch.diag(self.action_var).unsqueeze(dim=0)      # 增加一个维度 [1, N, N]
            dist = MultivariateNormal(action_mean, cov_mat)     # 连续动作空间中，从分布中取一个动作值

            if deterministic:
                action = action_mean
            else:
                action = dist.sample()
        else:
            action_probs = self.actor(state)
            dist = Categorical(action_probs)        # 离散动作空间中，按照网络 softmax 得出的概率分布取一个动作

            if deterministic:
                action = torch.argmax(action_probs, dim=1).unsqueeze(dim=1)
            else:
                action = dist.sample()

        action_logprob = dist.log_prob(action)      # 计算所采样动作的 对数概率
        state_value = self.critic(state)

        return action.detach(), action_logprob.detach(), state_value.detach()       # .detach() 表示断开当前计算图，防止在采样时计算梯度
    
    def deterministic_action(self, state):
        if self.is_continuous_action_space:
            action_mean = self.actor(state)


    def evaluate(self, state, action):
        if self.is_continuous_action_space:
            action_mean = self.actor(state)
            action_var = self.action_var.expand_as(action_mean)
            cov_mat = torch.diag_embed(action_var).to(device)
            dist = MultivariateNormal(action_mean, cov_mat)

            if self.action_dim == 1:
                action = action.reshape(-1, self.action_dim)

        else:
            action_probs = self.actor(state)
            dist = Categorical(action_probs)

        action_logprobs = dist.log_prob(action)
        dist_entropy = dist.entropy()
        state_value = self.critic(state)

        return action_logprobs, state_value, dist_entropy


class PPO:
    def __init__(self, state_dim, action_dim, lr_actor, lr_critic, gamma, K_epochs, eps_clipping, is_continuous_action_space, action_std_init=0.6):
        self.is_continuous_action_space = is_continuous_action_space
        if is_continuous_action_space:
            self.action_std = action_std_init

        self.gamma = gamma
        self.eps_clipping = eps_clipping
        self.K_epochs = K_epochs

        self.buffer = RolloutBuffer()

        self.policy = ActorCritic(state_dim, action_dim, is_continuous_action_space, action_std_init).to(device)
        self.optimizer = torch.optim.Adam([
            {'params': self.policy.actor.parameters(), 'lr': lr_actor},
            {'params': self.policy.critic.parameters(), 'lr': lr_critic}
        ])

        self.policy_old = ActorCritic(state_dim, action_dim, is_continuous_action_space, action_std_init).to(device)
        self.policy_old.load_state_dict(self.policy.state_dict())

        self.MseLoss = nn.MSELoss()

    def set_action_std(self, new_action_std):
        if self.is_continuous_action_space:
            self.action_std = new_action_std
            self.policy.set_action_std(new_action_std)
            self.policy_old.set_action_std(new_action_std)      # 如果更新分布标准差，则给当前策略和旧策略同步更新，避免在计算重要性采比率比时由于分布形状不一致的问题导致的计算错误
        else:
            print("--------------------------------------------------------------------------------------------")
            print("WARNING : Calling PPO::set_action_std() on discrete action space policy")
            print("--------------------------------------------------------------------------------------------")

    def decay_action_std(self, action_std_decay_rate, min_action_std):      # 在强化学习的过程中逐渐减小动作空间中的标准差
        print("--------------------------------------------------------------------------------------------")
        if self.is_continuous_action_space:
            self.action_std = self.action_std - action_std_decay_rate
            self.action_std = round(self.action_std, 4)
            if (self.action_std <= min_action_std):
                self.action_std = min_action_std
                print("setting actor output action_std to min_action_std : ", self.action_std)
            else:
                print("setting actor output action_std to : ", self.action_std)
            self.set_action_std(self.action_std)

        else:
            print("WARNING : Calling PPO::decay_action_std() on discrete action space policy")
        print("--------------------------------------------------------------------------------------------")

    def select_action(self, state, deterministic=False):
        if self.is_continuous_action_space:
            with torch.no_grad():
                state = torch.FloatTensor(state).to(device)
                action, action_logprob, state_value = self.policy_old.take_action(state, deterministic)

            self.buffer.states.append(state)
            self.buffer.actions.append(action)
            self.buffer.logprobs.append(action_logprob)
            self.buffer.state_values.append(state_value)

            return action.detach().cpu().numpy().flatten()
        else:
            with torch.no_grad():
                state = torch.FloatTensor(state).to(device)
                action, action_logprob, state_value = self.policy_old.take_action(state)

            self.buffer.states.append(state)
            self.buffer.actions.append(action)
            self.buffer.logprobs.append(action_logprob)
            self.buffer.state_values.append(state_value)

            return action.item()

    def update(self):
    # 蒙特卡洛估计
        rewards = []
        discounted_reward = 0
        for reward, is_terminal in zip(reversed(self.buffer.rewards), reversed(self.buffer.is_terminals)):
            if is_terminal:
                discounted_reward = 0
            discounted_reward = reward + (self.gamma * discounted_reward)
            rewards.insert(0, discounted_reward)

        # 1. 处理states
        processed_states = []
        for state in self.buffer.states:
            if isinstance(state, np.ndarray):
                processed_states.append(torch.FloatTensor(state).to(device))
            elif isinstance(state, torch.Tensor):
                processed_states.append(state.to(device))
            else:
                # 如果既不是numpy也不是tensor，尝试转换
                processed_states.append(torch.FloatTensor([state]).to(device))
        
        # 2. 处理其他buffer数据
        processed_actions = []
        for action in self.buffer.actions:
            if not isinstance(action, torch.Tensor):
                processed_actions.append(torch.tensor([action], dtype=torch.float32).to(device))
            else:
                processed_actions.append(action.to(device))
        
        processed_logprobs = []
        for logprob in self.buffer.logprobs:
            if not isinstance(logprob, torch.Tensor):
                processed_logprobs.append(torch.tensor([logprob], dtype=torch.float32).to(device))
            else:
                processed_logprobs.append(logprob.to(device))
        
        processed_state_values = []
        for value in self.buffer.state_values:
            if not isinstance(value, torch.Tensor):
                processed_state_values.append(torch.tensor([value], dtype=torch.float32).to(device))
            else:
                processed_state_values.append(value.to(device))

        rewards = torch.tensor(rewards, dtype=torch.float32).to(device)
        old_states = torch.squeeze(torch.stack(processed_states, dim=0)).detach()
        old_actions = torch.squeeze(torch.stack(processed_actions, dim=0)).detach()
        old_logprobs = torch.squeeze(torch.stack(processed_logprobs, dim=0)).detach()
        old_state_values = torch.squeeze(torch.stack(processed_state_values, dim=0)).detach()

        # 计算优势（修复归一化问题）
        advantages = rewards.detach() - old_state_values.detach()
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-7)
        #打印loss
        policy_loss_epoch = 0.0
        value_loss_epoch = 0.0
        entropy_epoch = 0.0
        total_loss_epoch = 0.0

        for _ in range(self.K_epochs):
            logprobs, state_values, dist_entropy = self.policy.evaluate(old_states, old_actions)
            state_values = torch.squeeze(state_values)
            ratios = torch.exp(logprobs - old_logprobs.detach())

            surr1 = ratios * advantages
            surr2 = torch.clamp(ratios, 1-self.eps_clipping, 1+self.eps_clipping) * advantages

            # 显式计算各项损失的均值
            surrogate_loss = -torch.min(surr1, surr2).mean()
            critic_loss = 0.5 * self.MseLoss(state_values, rewards)
            entropy_loss = -0.01 * dist_entropy.mean()
            loss = surrogate_loss + critic_loss + entropy_loss

            # policy_loss_epoch += surrogate_loss.item()
            # value_loss_epoch += critic_loss.item()
            # entropy_epoch += dist_entropy.mean().item()   # 注意：log 原始 entropy
            # total_loss_epoch += loss.item()

            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()


        # policy_loss_epoch /= self.K_epochs
        # value_loss_epoch /= self.K_epochs
        # entropy_epoch /= self.K_epochs
        # total_loss_epoch /= self.K_epochs
        # wandb.log({
        #     "loss/total": total_loss_epoch,
        #     "loss/policy": policy_loss_epoch,
        #     "loss/value": value_loss_epoch,
        #     "policy/entropy": entropy_epoch,
        # })
        self.policy_old.load_state_dict(self.policy.state_dict())
        self.buffer.clear()


    def save(self, checkpoint_path):
        torch.save(self.policy_old.state_dict(), checkpoint_path)

    def load(self, checkpoint_path):
        self.policy_old.load_state_dict(torch.load(checkpoint_path, map_location=lambda storage, loc: storage))
        self.policy.load_state_dict(torch.load(checkpoint_path, map_location=lambda storage, loc: storage))

    # 用于测试,纯推理
    def act(self, state, deterministic=True):
        with torch.no_grad():
            state_t = torch.FloatTensor(state).to(device)
            action, _, _ = self.policy_old.take_action(state_t, deterministic=deterministic)
        return action.detach().cpu().numpy().flatten()



