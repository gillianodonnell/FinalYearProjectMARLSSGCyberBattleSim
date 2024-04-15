# agent_ppo.py

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from cyberbattle._env import cyberbattle_env
from .learner import Learner
import cyberbattle.agents.baseline.agent_wrapper as w
from torch.distributions import Categorical

# Define the PPO Actor-Critic Model


class ActorCritic(nn.Module):
    def __init__(self, state_space, action_space, hidden_size=256):
        super(ActorCritic, self).__init__()
        self.actor = nn.Sequential(
            nn.Linear(state_space, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, action_space),
            nn.Softmax(dim=-1)
        )
        self.critic = nn.Sequential(
            nn.Linear(state_space, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, 1)
        )

    def forward(self, state):
        action_probs = self.actor(state)
        state_values = self.critic(state)
        return action_probs, state_values

# PPO Agent


class PPOAgent(Learner):
    def __init__(self, observation_space, action_space, hidden_size=256, learning_rate=0.01, gamma=0.99):
        # Assuming observation space is a simple vector and action space is discrete
        state_space_dim = observation_space  # Replace with correct processing of observation_space
        action_space_dim = action_space  # Assuming discrete action space

        # Rest of your code...
        self.actor = nn.Sequential(
            nn.Linear(state_space_dim, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, action_space_dim),
            nn.Softmax(dim=-1)
        )
        self.critic = nn.Sequential(
            nn.Linear(self.state_space, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, 1)
        )

        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=learning_rate)
        self.critic_optimizer = optim.Adam(self.critic.parameters(), lr=learning_rate)

    def get_action(self, state):
        """Selects an action based on the policy network."""
        state = torch.from_numpy(state).float()
        action_probs = self.actor(state)
        action_distribution = Categorical(action_probs)
        action = action_distribution.sample()
        return action.item(), action_distribution.log_prob(action)

    def compute_returns(self, rewards, masks):
        """Compute the discounted returns."""
        returns = []
        R = 0
        for reward, mask in zip(reversed(rewards), reversed(masks)):
            R = reward + self.gamma * R * mask
            returns.insert(0, R)
        return returns

    def optimize_model(self, states, actions, log_probs, returns):
        """Optimizes the policy and value networks."""
        states = torch.tensor(states)
        actions = torch.tensor(actions)
        returns = torch.tensor(returns)
        log_probs = torch.stack(log_probs)

        # Compute advantage
        values = self.critic(states)
        advantage = returns - values.squeeze()

        # Update actor (policy network)
        self.actor_optimizer.zero_grad()
        actor_loss = -(log_probs * advantage.detach()).mean()
        actor_loss.backward()
        self.actor_optimizer.step()

        # Update critic (value network)
        self.critic_optimizer.zero_grad()
        critic_loss = advantage.pow(2).mean()
        critic_loss.backward()
        self.critic_optimizer.step()

    def explore(self, state):
        """Random exploration strategy."""
        # Example implementation - choose a random action
        return np.random.randint(self.action_space)

    def exploit(self, state):
        """Exploitation strategy using the policy network."""
        # Example implementation - choose the best action based on policy network
        action, _ = self.get_action(state)
        return action

    def on_step(self, state, action, reward, next_state, done):
        """Update method after each step in the environment."""
        # Example implementation - store transition and perform learning updates
        # This method needs to be filled with your logic for storing transitions
        # and updating the model.
        pass

# Training and Evaluation Functions


def train(agent, env, num_episodes):
    for episode in range(num_episodes):
        state = env.reset()
        for t in range(1000):  # or some maximum number of steps per episode
            action, _ = agent.select_action(state)
            next_state, reward, done, _ = env.step(action)
            # store the transition in memory
            # ...
            state = next_state
            if done:
                break
        agent.optimize_model()
        # additional logging and evaluation
