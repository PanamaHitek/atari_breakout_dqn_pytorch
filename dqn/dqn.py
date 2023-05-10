"""
Script Name: dqn.py

Neural network architecture for the Deep Q-Network (DQN) algorithm, used for training an agent to play a game from pixel input.
The neural network architecture consists of convolutional layers followed by fully connected layers. The agent learns to play the game
using reinforcement learning by interacting with the environment and updating the weights of the neural network
accordingly. The script also defines an epsilon-greedy action selection strategy, a replay memory buffer,
and a function for converting input image frames to PyTorch tensors. The replay memory buffer stores experience tuples
that the agent uses to update its Q-values, and the epsilon-greedy strategy balances exploration and exploitation during training.
"""

import random
import torch
import torch.nn as nn
import torch.nn.functional as F
import math

# Define the neural network architecture for the DQN algorithm
class DQN(nn.Module):
    def __init__(self, outputs, device):
        super(DQN, self).__init__()
        # Define the convolutional layers
        self.conv1 = nn.Conv2d(4, 32, kernel_size=8, stride=4, bias=False)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=4, stride=2, bias=False)
        self.conv3 = nn.Conv2d(64, 64, kernel_size=3, stride=1, bias=False)
        # Define the fully connected layers
        self.fc1 = nn.Linear(64 * 7 * 7, 512)
        self.fc2 = nn.Linear(512, outputs)
        # Store the device where the model will be trained
        self.device = device

    # Define the forward pass of the neural network
    def forward(self, x):
        x = x.to(self.device).float() / 255.
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        x = F.relu(self.fc1(x.view(x.size(0), -1)))
        return self.fc2(x)

    # Initialize the weights of the neural network
    def init_weights(self, m):
        if type(m) == nn.Linear:
            torch.nn.init.kaiming_normal_(m.weight, nonlinearity='relu')
            m.bias.data.fill_(0.0)
        if type(m) == nn.Conv2d:
            torch.nn.init.kaiming_normal_(m.weight, nonlinearity='relu')

# Define the epsilon-greedy action selection strategy
class ActionSelector(object):
    def __init__(self, initial_eps, final_eps, eps_decay,random_exp, policy_net, n_actions, dev):
        self.eps = initial_eps
        self.final_epsilon = final_eps
        self.initial_epsilon = initial_eps
        self.random_exploration_interval = random_exp
        self.policy_network = policy_net
        self.epsilon_decay = eps_decay
        self.possible_actions = n_actions
        self.device = dev

    '''
    Constructor used for evaluation
    '''
    def __init__(self, eps,  policy_net, n_actions, dev):
        self.eps = eps
        self.policy_network = policy_net
        self.possible_actions = n_actions
        self.device = dev

    def select_action(self, state, current_episode=1, training=False):
        sample = random.random()
        if training:
            # Update the value of epsilon during training based on the current episode
            self.eps = self.final_epsilon + (self.initial_epsilon - self.final_epsilon) * \
                       math.exp(-1. * (current_episode-self.random_exploration_interval) / self.epsilon_decay)
            self.eps = max(self.eps, self.final_epsilon)
        if sample > self.eps:
            with torch.no_grad():
                a = self.policy_network(state).max(1)[1].cpu().view(1, 1)
        else:
            a = torch.tensor([[random.randrange(self.possible_actions)]], device=self.device, dtype=torch.long)
        return a.cpu().numpy()[0, 0].item(), self.eps

# Define the replay memory data structure for storing experience tuples
class ReplayMemory(object):
    def __init__(self, capacity, state_shape, device):
        replay_buffer_capacity,height,width = state_shape
        # Store the capacity of the memory buffer, the device, and initialize the memory buffer
        self.capacity = capacity
        self.device = device
        self.actions = torch.zeros((capacity, 1), dtype=torch.long)
        self.states = torch.zeros((capacity, replay_buffer_capacity, height, width), dtype=torch.uint8)
        self.rewards = torch.zeros((capacity, 1), dtype=torch.int8)
        self.dones = torch.zeros((capacity, 1), dtype=torch.bool)
        # Initialize the position and size of the memory buffer
        self.position = 0
        self.size = 0

    # Add a new experience tuple to the memory buffer
    def push(self, state, action, reward, done):
        self.states[self.position] = state
        self.actions[self.position,0] = action
        self.rewards[self.position,0] = reward
        self.dones[self.position,0] = done
        # Update the position and size of the memory buffer
        self.position = (self.position + 1) % self.capacity
        self.size = max(self.size, self.position)

    # Sample a batch of experiences from the memory buffer
    def sample(self, batch_state):
        i = torch.randint(0, high=self.size, size=(batch_state,))
        # Get the current state and next state
        batch_next_state = self.states[i, 1:]
        batch_state = self.states[i, :4]
        # Get the action, reward, and done values for each experience tuple in the batch
        batch_reward = self.rewards[i].to(self.device).float()
        batch_done = self.dones[i].to(self.device).float()
        batch_actions = self.actions[i].to(self.device)
        # Return the batch of experiences
        return batch_state, batch_actions, batch_reward, batch_next_state, batch_done

    # Return the current size of the memory buffer
    def __len__(self):
        return self.size

# Define a function for converting an input image frame to a PyTorch tensor
def get_frame_tensor(pixels):
    pixels = torch.from_numpy(pixels)
    height = pixels.shape[-2]
    return pixels.view(1, height, height)
