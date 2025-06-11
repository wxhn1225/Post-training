"""
PPO算法实现
知乎链接：https://zhuanlan.zhihu.com/p/1906716770145932974
"""




# 导入PyTorch库，用于构建神经网络和进行张量计算
import torch
# 导入PyTorch的神经网络模块
import torch.nn as nn
# 导入PyTorch的优化器模块
import torch.optim as optim
# 导入PyTorch的函数模块，包含各种激活函数等
import torch.nn.functional as F
# 导入PyTorch的分布模块，用于创建概率分布
from torch.distributions import Categorical
# 导入NumPy库，用于数值计算
import numpy as np
# 导入OpenAI Gym库，用于强化学习环境
import gym


# 定义PPO算法的超参数类
class PPOConfig:
    def __init__(self):
        self.lr = 3e-4  # 学习率
        self.gamma = 0.99  # 折扣因子，用于计算未来奖励的现值
        self.lamda = 0.95  # GAE(广义优势估计)的参数
        self.eps_clip = 0.2  # PPO剪切参数，限制策略更新的幅度
        self.K_epochs = 4  # 每次数据收集后执行的训练轮数
        self.batch_size = 64  # 每次参数更新使用的批量大小
        self.buffer_size = 2048  # 经验回放缓冲区大小
        self.entropy_coef = 0.01  # 熵奖励系数，鼓励探索
        self.value_coef = 0.5  # 价值函数损失系数
        self.hidden_dim = 64  # 神经网络隐藏层维度
        self.max_episodes = 10000  # 最大训练回合数
        self.update_freq = self.buffer_size  # 更新频率，等于缓冲区大小


# 定义Actor-Critic网络结构
class ActorCritic(nn.Module):
    def __init__(self, state_dim, action_dim, hidden_dim):
        super(ActorCritic, self).__init__()
        # 定义共享的网络层
        self.fc_shared = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),  # 全连接层，输入状态维度，输出隐藏层维度
            nn.ReLU()  # ReLU激活函数
        )
        # 定义策略网络（Actor）
        self.fc_actor = nn.Linear(hidden_dim, action_dim)  # 输出动作维度
        # 定义价值网络（Critic）
        self.fc_critic = nn.Linear(hidden_dim, 1)  # 输出单个值（状态价值）

    def forward(self, x):
        # 前向传播，通过共享层
        x = self.fc_shared(x)
        return x

    def get_action(self, x):
        # 获取动作和对应的对数概率
        hidden = self.forward(x)  # 通过共享层
        logits = self.fc_actor(hidden)  # 通过策略头
        dist = Categorical(logits=logits)  # 创建分类分布
        action = dist.sample()  # 从分布中采样动作
        log_prob = dist.log_prob(action)  # 计算动作的对数概率
        return action.item(), log_prob  # 返回动作值和对数概率

    def get_value(self, x):
        # 获取状态价值
        hidden = self.forward(x)  # 通过共享层
        value = self.fc_critic(hidden)  # 通过价值头
        return value


# 定义PPO算法类
class PPO:
    def __init__(self, state_dim, action_dim, config):
        self.config = config  # 存储配置参数
        self.policy = ActorCritic(state_dim, action_dim, config.hidden_dim)  # 创建策略网络
        self.optimizer = optim.Adam(self.policy.parameters(), lr=config.lr)  # 使用Adam优化器
        # 创建旧策略网络，用于计算重要性采样比率
        self.old_policy = ActorCritic(state_dim, action_dim, config.hidden_dim)
        self.old_policy.load_state_dict(self.policy.state_dict())  # 初始与当前策略相同
        self.buffer = []  # 经验回放缓冲区
        self.mse_loss = nn.MSELoss()  # 均方误差损失，用于价值函数

    def update(self):
        # 从缓冲区提取数据并转换为张量
        states = torch.FloatTensor(np.array([t[0] for t in self.buffer]))  # 状态
        actions = torch.LongTensor(np.array([t[1] for t in self.buffer])).unsqueeze(1)  # 动作
        old_log_probs = torch.FloatTensor(np.array([t[2] for t in self.buffer])).unsqueeze(1)  # 旧策略的对数概率
        rewards = torch.FloatTensor(np.array([t[3] for t in self.buffer]))  # 奖励
        next_states = torch.FloatTensor(np.array([t[4] for t in self.buffer]))  # 下一个状态
        dones = torch.FloatTensor(np.array([t[5] for t in self.buffer]))  # 终止标志

        # 计算广义优势估计(GAE)和回报
        with torch.no_grad():  # 不计算梯度
            values = self.old_policy.get_value(states)  # 当前状态价值
            next_values = self.old_policy.get_value(next_states)  # 下一状态价值
            deltas = rewards + self.config.gamma * next_values * (1 - dones) - values  # 计算TD误差
            advantages = torch.zeros_like(rewards)  # 初始化优势函数
            advantage = 0  # 初始化优势值
            # 反向计算GAE
            for t in reversed(range(len(rewards))):
                advantage = deltas[t] + self.config.gamma * self.config.lamda * advantage * (1 - dones[t])
                advantages[t] = advantage
            returns = advantages + values.squeeze(1)  # 计算回报
            advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)  # 标准化优势

        # 多次更新策略
        for _ in range(self.config.K_epochs):
            indices = torch.randperm(len(self.buffer))  # 随机打乱索引
            # 小批量更新
            for start in range(0, len(self.buffer), self.config.batch_size):
                end = start + self.config.batch_size
                idx = indices[start:end]  # 获取当前批次的索引
                # 获取当前批次的数据
                batch_states = states[idx]
                batch_actions = actions[idx]
                batch_old_log_probs = old_log_probs[idx]
                batch_advantages = advantages[idx].unsqueeze(1)
                batch_returns = returns[idx].unsqueeze(1)

                # 计算新策略的概率和值
                hidden = self.policy(batch_states)  # 通过共享层
                logits = self.policy.fc_actor(hidden)  # 通过策略头
                dist = Categorical(logits=logits)  # 创建动作分布
                log_probs = dist.log_prob(batch_actions.squeeze(1)).unsqueeze(1)  # 新策略对数概率
                entropy = dist.entropy().mean()  # 计算熵（用于鼓励探索）
                values = self.policy.get_value(batch_states)  # 计算状态价值

                # 计算重要性采样比率
                ratios = torch.exp(log_probs - batch_old_log_probs)
                # 计算剪切目标函数
                surr1 = ratios * batch_advantages  # 未剪切的目标
                surr2 = torch.clamp(ratios, 1 - self.config.eps_clip,
                                    1 + self.config.eps_clip) * batch_advantages  # 剪切后的目标
                policy_loss = -torch.min(surr1, surr2).mean()  # 策略损失（取负是因为我们要最大化）
                value_loss = self.mse_loss(values, batch_returns)  # 价值函数损失
                # 总损失 = 策略损失 + 价值损失系数*价值损失 - 熵系数*熵（最大化熵）
                loss = policy_loss + self.config.value_coef * value_loss - self.config.entropy_coef * entropy

                # 执行梯度下降
                self.optimizer.zero_grad()  # 清空梯度
                loss.backward()  # 反向传播
                self.optimizer.step()  # 更新参数

        # 更新旧策略网络
        self.old_policy.load_state_dict(self.policy.state_dict())
        self.buffer = []  # 清空缓冲区

    def store_transition(self, transition):
        # 存储转移样本到缓冲区
        self.buffer.append(transition)
        # 当缓冲区满时执行更新
        if len(self.buffer) == self.config.update_freq:
            self.update()


# 训练函数
def train():
    env = gym.make('CartPole-v1')  # 创建CartPole环境
    config = PPOConfig()  # 创建配置
    state_dim = env.observation_space.shape[0]  # 获取状态维度
    action_dim = env.action_space.n  # 获取动作维度
    agent = PPO(state_dim, action_dim, config)  # 创建PPO智能体

    # 训练循环
    for episode in range(config.max_episodes):
        state = env.reset()  # 重置环境
        episode_reward = 0  # 初始化回合奖励

        while True:
            # 与环境交互
            action, log_prob = agent.policy.get_action(torch.FloatTensor(state))  # 选择动作
            next_state, reward, done, _ = env.step(action)  # 执行动作
            # 存储转移样本
            agent.store_transition((state, action, log_prob, reward, next_state, done))
            state = next_state  # 更新状态
            episode_reward += reward  # 累计奖励
            if done:  # 回合结束
                break

        print(f'Episode {episode}, Reward: {episode_reward}')  # 打印回合信息
        # 定期测试
        if episode % 10 == 0:
            test_reward = test(env, agent)  # 执行测试
            print(f'Test Reward: {test_reward}')  # 打印测试结果


# 测试函数
def test(env, agent, test_episodes=5):
    total_reward = 0  # 总奖励
    for _ in range(test_episodes):
        state = env.reset()  # 重置环境
        episode_reward = 0  # 回合奖励
        while True:
            action, _ = agent.policy.get_action(torch.FloatTensor(state))  # 选择动作
            next_state, reward, done, _ = env.step(action)  # 执行动作
            episode_reward += reward  # 累计奖励
            state = next_state  # 更新状态
            if done:  # 回合结束
                break
        total_reward += episode_reward  # 累计总奖励
    return total_reward / test_episodes  # 返回平均奖励


# 主程序入口
if __name__ == '__main__':
    train()  # 开始训练