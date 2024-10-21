import torch
import torch.nn as nn
import torch.optim as optim

import numpy as np

from collections import deque

class Actor(nn.Module):
  def __init__(self, state_dim, n_actions, units, dropout=0.25):
    super(Actor, self).__init__()
    self.fc1 = nn.Linear(state_dim, units)
    self.dropout1 = nn.Dropout(dropout)

    self.fc2 = nn.Linear(units, units)
    self.dropout2 = nn.Dropout(dropout)

    self.fc3 = nn.Linear(units, n_actions)
    self.softmax = nn.Softmax(dim=-1)

    self.loss_history = []

  def forward(self, x):
    x = torch.relu(self.fc1(x))
    x = self.dropout1(x)

    x = torch.relu(self.fc2(x))
    x = self.dropout2(x)

    x = self.fc3(x)
    x = self.softmax(x)
    return x
  
class Critic(nn.Module):
  def __init__(self, state_dim, units, dropout=0.25):
    super(Critic, self).__init__()
    self.fc1 = nn.Linear(state_dim, units)
    self.dropout1 = nn.Dropout(dropout)

    self.fc2 = nn.Linear(units, units)
    self.dropout2 = nn.Dropout(dropout)

    self.fc3 = nn.Linear(units, 1)

    self.loss_history = []

  def forward(self, x):
    x = torch.relu(self.fc1(x))
    x = self.dropout1(x)

    x = torch.relu(self.fc2(x))
    x = self.dropout2(x)

    x = self.fc3(x)
    return x

class ReplayBuffer():
  def __init__(self, max_size):
    self.memory = deque(maxlen=max_size)
    self.last_q_value = 0

  def add(self, state, action, reward, done):
    experience = (state, action, reward, 1 if done else 0)
    self.memory.append(experience)

  def get_memories(self):
    states, actions, rewards, dones = zip(*self.memory)

    states = np.array(states)
    actions = np.array(actions)
    rewards = np.array(rewards)
    dones = np.array(dones)

    return states, actions, rewards, dones

  def clear(self):
    self.memory.clear()
    self.last_q_value = 0

  def __len__(self):
    return len(self.memory)
  
class AgentA2C:
  def __init__(self, action_dim, observation_dim, gamma=0.99, actor_learning_rate=1e-3, critic_learning_rate=1e-3, units=32, dropout=0.25):
    self.action_dim = action_dim
    self.observation_dim = observation_dim

    self.gamma = gamma

    self.actor_learning_rate = actor_learning_rate
    self.critic_learning_rate = critic_learning_rate
    self.units = units
    self.dropout = dropout

    self.actor = Actor(self.observation_dim, self.action_dim, self.units, self.dropout)
    self.actor_opt = torch.optim.Adam(self.actor.parameters(), lr=self.actor_learning_rate)

    self.critic = Critic(self.observation_dim, self.units, self.dropout)
    self.critic_opt = torch.optim.Adam(self.critic.parameters(), lr=self.critic_learning_rate)

    self.memory = ReplayBuffer(20000)

    self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    self.actor.to(self.device)
    self.critic.to(self.device)
    print(f'Using device: {self.device}')

  def save(self, path):
    torch.save({
        'actor_state_dict': self.actor.state_dict(),
        'critic_state_dict': self.critic.state_dict(),
        'actor_learning_rate': self.actor_learning_rate,
        'critic_learning_rate': self.critic_learning_rate,
        'units': self.units,
        'dropout': self.dropout,
        'action_dim': self.action_dim,
        'observation_dim': self.observation_dim,
        'gamma': self.gamma,
        'actor_loss_history': self.actor.loss_history,
        'critic_loss_history': self.critic.loss_history
    }, path)

  def load(self, path):
    data = torch.load(path)
    self.actor.load_state_dict(data['actor_state_dict'])
    self.optimizer = optim.Adam(self.actor.parameters(), lr=data['actor_learning_rate'])
    self.critic.load_state_dict(data['critic_state_dict'])
    self.optimizer = optim.Adam(self.critic.parameters(), lr=data['critic_learning_rate'])
    
    self.units = data['units']
    self.dropout = data['dropout']

    self.action_dim = data['action_dim']
    self.observation_dim = data['observation_dim']

    self.gamma = data['gamma']

    self.actor.loss_history = data['actor_loss_history']
    self.critic.loss_history = data['critic_loss_history']

  def act(self, state):
    state_tensor = torch.FloatTensor(state).to(self.device)
    probabilities_tensor = self.actor(state_tensor)
    distribution_tensor = torch.distributions.Categorical(probs=probabilities_tensor)
    action_tensor = distribution_tensor.sample()

    return action_tensor.cpu().detach().item()

  def remember(self, state, action, reward, done):
    self.memory.add(state, action, reward, done)

  def learn(self):
    states, actions, rewards, dones = self.memory.get_memories()
    states_tensor = torch.FloatTensor(states).to(self.device)
    actions_tensor = torch.LongTensor(actions).to(self.device)
    rewards_tensor = torch.FloatTensor(rewards).to(self.device)
    dones_tensor = torch.FloatTensor(dones).to(self.device)

    values_tensor = self.critic(states_tensor).squeeze(-1)

    q_values = torch.zeros_like(rewards_tensor).to(self.device)
    last_state_tensor = torch.FloatTensor(self.memory.last_q_value).to(self.device)
    q_value = self.critic(last_state_tensor).detach()
    for i in reversed(range(len(rewards_tensor))):
      q_value = rewards_tensor[i] + self.gamma * q_value * (1 - dones_tensor[i])
      q_values[i] = q_value

    advantages = q_values - values_tensor

    probabilities_tensor = self.actor(states_tensor)
    distribution_tensor = torch.distributions.Categorical(probs=probabilities_tensor)
    log_probabilities = distribution_tensor.log_prob(actions_tensor)

    entropy_loss = distribution_tensor.entropy().mean()
    actor_loss = (-log_probabilities * advantages.detach()).mean()
    actor_loss = actor_loss - 1e-3 * entropy_loss

    self.actor_opt.zero_grad()
    actor_loss.backward()
    self.actor_opt.step()

    critic_loss = 1e-4 * advantages.pow(2).mean()

    self.critic_opt.zero_grad()
    critic_loss.backward()
    self.critic_opt.step()

    self.actor.loss_history.append(actor_loss.detach().item())
    self.critic.loss_history.append(critic_loss.detach().item())

    self.memory.clear()

  def get_last_state(self, last_state):
    self.memory.last_q_value = last_state