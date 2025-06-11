import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Categorical
from torch.utils.data import DataLoader, TensorDataset
from collections import namedtuple
import gymnasium as gym

# 设备配置
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# 经验存储
Memory = namedtuple('Memory', ['state', 'action', 'reward', 'next_state', 'done', 'log_prob', 'value'])

class ActorNetwork(nn.Module):
    """简单的Actor网络"""
    def __init__(self, state_dim, action_dim, hidden_dim=64):
        super(ActorNetwork, self).__init__()
        self.fc1 = nn.Linear(state_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, action_dim)
        
    def forward(self, state):
        x = F.relu(self.fc1(state))
        x = F.relu(self.fc2(x))
        action_probs = F.softmax(self.fc3(x), dim=-1)
        return action_probs

class CriticNetwork(nn.Module):
    """简单的Critic网络"""
    def __init__(self, state_dim, hidden_dim=64):
        super(CriticNetwork, self).__init__()
        self.fc1 = nn.Linear(state_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, 1)
        
    def forward(self, state):
        x = F.relu(self.fc1(state))
        x = F.relu(self.fc2(x))
        value = self.fc3(x)
        return value

class PPO:
    def __init__(self, state_dim, action_dim, lr=3e-4, gamma=0.99, lam=0.95, 
                 eps_clip=0.2, k_epochs=4, batch_size=64):
        self.gamma = gamma
        self.lam = lam
        self.eps_clip = eps_clip
        self.k_epochs = k_epochs
        self.batch_size = batch_size
        
        # 网络初始化
        self.actor = ActorNetwork(state_dim, action_dim).to(device)
        self.critic = CriticNetwork(state_dim).to(device)
        
        # 优化器
        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=lr)
        self.critic_optimizer = torch.optim.Adam(self.critic.parameters(), lr=lr)
        
        # 经验存储
        self.memory = []
        
    def select_action(self, state):
        """选择动作"""
        state = torch.FloatTensor(state).to(device)
        
        with torch.no_grad():
            action_probs = self.actor(state)
            value = self.critic(state)
            
        dist = Categorical(action_probs)
        action = dist.sample()
        log_prob = dist.log_prob(action)
        
        return action.item(), log_prob.item(), value.item()
    
    def store_transition(self, state, action, reward, next_state, done, log_prob, value):
        """存储经验"""
        self.memory.append(Memory(state, action, reward, next_state, done, log_prob, value))
    
    def compute_gae(self, rewards, values, dones):
        """计算GAE (Generalized Advantage Estimation)"""
        advantages = []
        gae = 0
        
        for i in reversed(range(len(rewards))):
            if i == len(rewards) - 1:
                next_value = 0
            else:
                next_value = values[i + 1]
                
            delta = rewards[i] + self.gamma * next_value * (1 - dones[i]) - values[i]
            gae = delta + self.gamma * self.lam * (1 - dones[i]) * gae
            advantages.insert(0, gae)
            
        return advantages
    
    def update(self):
        """PPO更新"""
        if len(self.memory) == 0:
            return
            
        # 提取数据
        states = torch.FloatTensor([m.state for m in self.memory]).to(device)
        actions = torch.LongTensor([m.action for m in self.memory]).to(device)
        rewards = [m.reward for m in self.memory]
        dones = [m.done for m in self.memory]
        old_log_probs = torch.FloatTensor([m.log_prob for m in self.memory]).to(device)
        old_values = torch.FloatTensor([m.value for m in self.memory]).to(device)
        
        # 计算GAE和returns
        advantages = self.compute_gae(rewards, old_values.cpu().numpy(), dones)
        advantages = torch.FloatTensor(advantages).to(device)
        returns = advantages + old_values
        
        # 标准化advantages
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
        
        # 创建数据加载器
        dataset = TensorDataset(states, actions, old_log_probs, returns, advantages)
        dataloader = DataLoader(dataset, batch_size=self.batch_size, shuffle=True)
        
        # PPO更新循环
        for _ in range(self.k_epochs):
            for batch_states, batch_actions, batch_old_log_probs, batch_returns, batch_advantages in dataloader:
                
                # 计算当前策略的概率和价值
                action_probs = self.actor(batch_states)
                values = self.critic(batch_states).squeeze()
                
                dist = Categorical(action_probs)
                new_log_probs = dist.log_prob(batch_actions)
                entropy = dist.entropy()
                
                # 计算比率
                ratios = torch.exp(new_log_probs - batch_old_log_probs)
                
                # PPO损失
                surr1 = ratios * batch_advantages
                surr2 = torch.clamp(ratios, 1 - self.eps_clip, 1 + self.eps_clip) * batch_advantages
                actor_loss = -torch.min(surr1, surr2).mean() - 0.01 * entropy.mean()
                
                # Critic损失
                critic_loss = F.mse_loss(values, batch_returns)
                
                # 更新网络
                self.actor_optimizer.zero_grad()
                actor_loss.backward()
                torch.nn.utils.clip_grad_norm_(self.actor.parameters(), 0.5)
                self.actor_optimizer.step()
                
                self.critic_optimizer.zero_grad()
                critic_loss.backward()
                torch.nn.utils.clip_grad_norm_(self.critic.parameters(), 0.5)
                self.critic_optimizer.step()
        
        # 清空memory
        self.memory.clear()

def train_ppo(env_name='LunarLander-v3', max_episodes=2000, max_timesteps=500, 
              update_timesteps=2000, save_interval=500):
    """训练PPO智能体"""
    
    # 创建环境
    env = gym.make(env_name)
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.n
    
    # 创建PPO智能体
    ppo = PPO(state_dim, action_dim)
    
    # 训练统计
    episode_rewards = []
    timestep = 0
    
    print(f"开始训练 PPO 在环境 {env_name}")
    print(f"状态维度: {state_dim}, 动作维度: {action_dim}")
    
    for episode in range(max_episodes):
        state, _ = env.reset()
        episode_reward = 0
        
        for t in range(max_timesteps):
            timestep += 1
            
            # 选择动作
            action, log_prob, value = ppo.select_action(state)
            
            # 执行动作
            next_state, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            
            # 存储经验
            ppo.store_transition(state, action, reward, next_state, done, log_prob, value)
            
            state = next_state
            episode_reward += reward
            
            # 更新策略
            if timestep % update_timesteps == 0:
                ppo.update()
                print(f"Timestep {timestep}: 策略更新完成")
            
            if done:
                break
        
        episode_rewards.append(episode_reward)
        
        # 打印进度
        if episode % 100 == 0:
            avg_reward = np.mean(episode_rewards[-100:])
            print(f"Episode {episode}, 平均奖励: {avg_reward:.2f}")
        
        # 保存模型
        if episode % save_interval == 0 and episode > 0:
            torch.save({
                'actor_state_dict': ppo.actor.state_dict(),
                'critic_state_dict': ppo.critic.state_dict(),
                'episode': episode,
                'episode_rewards': episode_rewards
            }, f'ppo_checkpoint_episode_{episode}.pth')
            print(f"模型已保存到 ppo_checkpoint_episode_{episode}.pth")
    
    env.close()
    return episode_rewards

if __name__ == "__main__":
    # 训练PPO
    rewards = train_ppo()
    
    # 打印最终结果
    print("\n训练完成!")
    print(f"最后100个episode的平均奖励: {np.mean(rewards[-100:]):.2f}") 