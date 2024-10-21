import torch
import torch.nn as nn
import torch.optim as optim

import numpy as np

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

class Actor(nn.Module):
  def __init__(self, state_dim, n_actions, units):
    super(Actor, self).__init__()
    self.fc1 = nn.Linear(state_dim, units)
    self.fc2 = nn.Linear(units, units)
    self.fc3 = nn.Linear(units, n_actions)
    self.softmax = nn.Softmax(dim=-1)

  def forward(self, x):
    x = torch.relu(self.fc1(x))
    x = torch.relu(self.fc2(x))
    x = self.fc3(x)
    x = self.softmax(x)
    return x

class Critic(nn.Module):
  def __init__(self, state_dim, units):
    super(Critic, self).__init__()
    self.fc1 = nn.Linear(state_dim, units)
    self.fc2 = nn.Linear(units, units)
    self.fc3 = nn.Linear(units, 1)

  def forward(self, x):
    x = torch.relu(self.fc1(x))
    x = torch.relu(self.fc2(x))
    x = self.fc3(x)
    return x
  
class ActorCritic(nn.Module):
  def __init__(self, state_dim, action_dim, units):
    super(ActorCritic, self).__init__()
    self.actor = Actor(state_dim, action_dim, units)
    self.critic = Critic(state_dim, units)

  def act(self, state):
    action_probs = self.actor(state)
    dist = torch.distributions.Categorical(action_probs)
    action = dist.sample()
    action_logprob = dist.log_prob(action)
    state_val = self.critic(state)
    return action.detach(), action_logprob.detach(), state_val.detach()

  def evaluate(self, state, action):
    action_probs = self.actor(state)
    dist = torch.distributions.Categorical(action_probs)
    action_logprobs = dist.log_prob(action)
    dist_entropy = dist.entropy()
    state_values = self.critic(state)
    return action_logprobs, state_values, dist_entropy
  
class AgentPPO:
  def __init__(self, action_dim, observation_dim, lr_actor=1e-4, lr_critic=1e-3, gamma=0.99, K_epochs=40, eps_clip=0.2, units=32):
    self.action_dim = action_dim
    self.observation_dim = observation_dim

    self.units = units

    self.actor_learning_rate = lr_actor
    self.critic_learning_rate = lr_critic
    
    self.gamma = gamma
    self.K_epochs = K_epochs
    self.eps_clip = eps_clip

    self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    self.buffer = RolloutBuffer()
    self.policy = ActorCritic(observation_dim, action_dim, units).to(self.device)
    self.optimizer = optim.Adam([
      {'params': self.policy.actor.parameters(), 'lr': lr_actor},
      {'params': self.policy.critic.parameters(), 'lr': lr_critic}
    ])
    self.policy_old = ActorCritic(observation_dim, action_dim, units).to(self.device)
    self.policy_old.load_state_dict(self.policy.state_dict())
    self.MseLoss = nn.MSELoss()

    self.loss_history = []

    print(f'Using device: {self.device}')

  def save(self, path):
    torch.save({
        'actor_state_dict': self.policy.actor.state_dict(),
        'critic_state_dict': self.policy.critic.state_dict(),
        'actor_learning_rate': self.actor_learning_rate,
        'critic_learning_rate': self.critic_learning_rate,
        'units': self.units,
        'action_dim': self.action_dim,
        'observation_dim': self.observation_dim,
        'gamma': self.gamma,
        'K_epochs': self.K_epochs,
        'eps_clip': self.eps_clip,
        'loss_history': self.loss_history,
    }, path)

  def load(self, path):
    data = torch.load(path)
    self.policy.actor.load_state_dict(data['actor_state_dict'])
    self.policy.critic.load_state_dict(data['critic_state_dict'])
    
    self.optimizer = optim.Adam([
      {'params': self.policy.actor.parameters(), 'lr': data['actor_learning_rate']},
      {'params': self.policy.critic.parameters(), 'lr': data['critic_learning_rate']}
    ])

    self.policy_old.load_state_dict(self.policy.state_dict())

    self.units = data['units']

    self.action_dim = data['action_dim']
    self.observation_dim = data['observation_dim']

    self.gamma = data['gamma']
    self.K_epochs = data['K_epochs']
    self.eps_clip = data['eps_clip']

    self.loss_history = data['loss_history']

  def select_action(self, state):
    with torch.no_grad():
      state = torch.FloatTensor(state).to(self.device)
      action, action_logprob, state_val = self.policy_old.act(state)
    self.buffer.states.append(state)
    self.buffer.actions.append(action)
    self.buffer.logprobs.append(action_logprob)
    self.buffer.state_values.append(state_val)
    return action.item()

  def update(self):
    rewards = []
    discounted_reward = 0
    for reward, is_terminal in zip(reversed(self.buffer.rewards), reversed(self.buffer.is_terminals)):
      if is_terminal:
        discounted_reward = 0
      discounted_reward = reward + (self.gamma * discounted_reward)
      rewards.insert(0, discounted_reward)

    rewards = torch.tensor(rewards, dtype=torch.float32).to(self.device)
    rewards = (rewards - rewards.mean()) / (rewards.std() + 1e-7)
    old_states = torch.squeeze(torch.stack(self.buffer.states, dim=0)).detach().to(self.device)
    old_actions = torch.squeeze(torch.stack(self.buffer.actions, dim=0)).detach().to(self.device)
    old_logprobs = torch.squeeze(torch.stack(self.buffer.logprobs, dim=0)).detach().to(self.device)
    old_state_values = torch.squeeze(torch.stack(self.buffer.state_values, dim=0)).detach().to(self.device)
    advantages = rewards - old_state_values

    update_loss = 0

    for _ in range(self.K_epochs):
      logprobs, state_values, dist_entropy = self.policy.evaluate(old_states, old_actions)
      state_values = torch.squeeze(state_values)
      ratios = torch.exp(logprobs - old_logprobs.detach())
      surr1 = ratios * advantages
      surr2 = torch.clamp(ratios, 1 - self.eps_clip, 1 + self.eps_clip) * advantages
      loss = -torch.min(surr1, surr2) + 0.5 * self.MseLoss(state_values, rewards) - 0.01 * dist_entropy

      self.optimizer.zero_grad()
      loss.mean().backward()
      self.optimizer.step()

      update_loss += loss.mean().item()

    self.loss_history.append(update_loss/self.K_epochs)

    self.policy_old.load_state_dict(self.policy.state_dict())
    self.buffer.clear()