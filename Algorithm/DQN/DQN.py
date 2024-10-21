import torch
import torch.nn as nn
import torch.optim as optim

import numpy as np
import random

from collections import deque

class DQN(nn.Module):
  def __init__(self, action_dim, observation_dim, units=32):
    super(DQN, self).__init__()
    self.fc1 = nn.Linear(observation_dim, units)
    self.fc2 = nn.Linear(units, units)
    self.fc3 = nn.Linear(units, action_dim)

  def forward(self, x):
    x = torch.relu(self.fc1(x))
    x = torch.relu(self.fc2(x))
    x = self.fc3(x)
    return x

class ReplayBuffer():
  def __init__(self, max_size):
    self.memory = deque(maxlen=max_size)

  def add(self, state, action, reward, next_state, done):
    experience = (state, action, reward, next_state, done)
    self.memory.append(experience)

  def sample(self, batch_size):
    batch = random.sample(self.memory, batch_size)
    return batch

  def __len__(self):
    return len(self.memory)

class AgentDQN():
  def __init__(self, action_dim, observation_dim, epsilon=1, epsilon_decay=0.995, epsilon_min=0.01, discount_factor=0.9, update_frequency=1000, batch_size=32, learning_rate=0.001, units=32):
    self.action_dim = action_dim
    self.observation_dim = observation_dim

    self.epsilon = epsilon
    self.epsilon_decay = epsilon_decay
    self.epsilon_min = epsilon_min
    self.batch_size = batch_size
    self.discount_factor = discount_factor
    self.update_frequency = update_frequency
    self.updates_done = 0

    self.learning_rate = learning_rate
    self.units = units

    self.memory = ReplayBuffer(max_size=20000)

    self.model = DQN(action_dim, observation_dim, units)
    self.optimizer = optim.Adam(self.model.parameters(), lr=learning_rate)

    self.target_model = DQN(action_dim, observation_dim, units)
    self.target_model.load_state_dict(self.model.state_dict())

    self.criterion = nn.MSELoss()

    self.loss_history = []
    self.epsilon_history = []
    
    self.num_param_updates = 0

    self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    self.model.to(self.device)
    self.target_model.to(self.device)
    print(f'Using device: {self.device}')

  def save(self, path):
    torch.save({
        'model_state_dict': self.model.state_dict(),
        'target_model_state_dict': self.target_model.state_dict(),
        'learning_rate': self.learning_rate,
        'units': self.units,
        'action_dim': self.action_dim,
        'observation_dim': self.observation_dim,
        'epsilon': self.epsilon,
        'epsilon_decay': self.epsilon_decay,
        'epsilon_min': self.epsilon_min,
        'updates_frequency': self.update_frequency,
        'updates_done': self.updates_done,
        'batch_size': self.batch_size,
        'discount_factor': self.discount_factor,
        'memory': self.memory,
        'loss_history': self.loss_history,
        'epsilon_decay': self.epsilon_history,
    }, path)

  def load(self, path):
    data = torch.load(path)
    self.model.load_state_dict(data['model_state_dict'])
    self.optimizer = optim.Adam(self.model.parameters(), lr=data['learning_rate'])
    self.target_model.load_state_dict(data['target_model_state_dict'])

    self.units = data['units']

    self.action_dim = data['action_dim']
    self.observation_dim = data['observation_dim']

    self.epsilon = data['epsilon']
    self.epsilon_decay = data['epsilon_decay']
    self.epsilon_min = data['epsilon_min']
    self.batch_size = data['batch_size']
    self.discount_factor = data['discount_factor']
    self.update_frequency = data['updates_frequency']
    self.updates_done = data['updates_done']

    self.memory = data['memory']

    self.loss_history = data['loss_history']
    self.epsilon = data['epsilon_decay']

  def act_greedy(self, state):
    if random.uniform(0, 1) < self.epsilon:
      return random.randrange(self.action_dim)

    state_tensor = torch.from_numpy(state).to(self.device)

    with torch.no_grad():
      action_values = self.model(state_tensor)

    return torch.argmax(action_values).item()

  def act(self, state):
    state_tensor = torch.from_numpy(state).to(self.device)

    with torch.no_grad():
      action_values = self.model(state_tensor)

    return torch.argmax(action_values).item()

  def remember(self, state, action, reward, next_state, done):
    self.memory.add(state, action, reward, next_state, done)

  def learn(self, replays=10):
    for _ in range(replays):
      self.replay()

  def replay(self):
    if len(self.memory) < self.batch_size:
      return

    if self.epsilon > self.epsilon_min:
      self.epsilon *= self.epsilon_decay

    batch = self.memory.sample(self.batch_size)
    states, actions, rewards, next_states, dones = zip(*batch)

    states = np.array(states)
    actions = np.array(actions)
    rewards = np.array(rewards)
    next_states = np.array(next_states)
    dones = np.array(dones)

    states_tensor = torch.tensor(states, dtype=torch.float32).to(self.device)
    actions_tensor = torch.tensor(actions, dtype=torch.int64).to(self.device)
    rewards_tensor = torch.tensor(rewards, dtype=torch.float32).to(self.device)
    next_states_tensor = torch.tensor(next_states, dtype=torch.float32).to(self.device)
    dones_tensor = torch.tensor(dones, dtype=torch.float32).to(self.device)

    current_q_values = self.model(states_tensor).gather(1, actions_tensor.unsqueeze(1))
    next_q_values = self.target_model(next_states_tensor).detach().max(1)[0]
    
    target_q_values = rewards_tensor + (1.0 - dones_tensor) * self.discount_factor * next_q_values

    loss = self.criterion(current_q_values.squeeze(1), target_q_values)

    self.optimizer.zero_grad()
    loss.backward()
    self.optimizer.step()

    self.loss_history.append(loss.item())

    if  self.updates_done % self.update_frequency == 0:
      self.target_model.load_state_dict(self.model.state_dict())
      self.updates_done = 0