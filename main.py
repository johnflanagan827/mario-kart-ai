#!/usr/bin/env python3

import socket
import random
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from collections import namedtuple
import numpy as np
import math
import os
import time as t

# =================================== DUELING DQN NETWORK =======================================

class DuelingDQNNetwork(nn.Module):
    """Class module for Dueling Deep Q-Network."""

    def __init__(self, input_dim, output_dim):
        super(DuelingDQNNetwork, self).__init__()
        self.fc1 = nn.Linear(input_dim, 24)
        self.fc2 = nn.Linear(24, 24)

        # Dueling streams
        self.value_stream = nn.Linear(24, 1)
        self.advantage_stream = nn.Linear(24, output_dim)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))

        value = self.value_stream(x)
        advantages = self.advantage_stream(x)

        # Combining value and advantage streams
        return value + (advantages - advantages.mean())


# ===================================== REPLAY MEMORY ==============================================

Transition = namedtuple('Transition', ('state', 'action', 'reward', 'next_state'))
EPSILON = 1e-10  # Small constant to avoid numerical instability


class PrioritizedReplayMemory:
    """Replay memory that uses priority values for sampling."""
    def __init__(self, capacity, alpha=0.6, beta_start=0.4, beta_frames=100000):
        self.capacity = capacity
        self.alpha = alpha
        self.beta_start = beta_start
        self.beta_frames = beta_frames
        self.frame = 1
        self.buffer = []
        self.priorities = np.zeros((capacity,), dtype=np.float32)

    def __len__(self):
        return len(self.buffer)

    def push(self, state, action, reward, next_state):
        max_priority = self.priorities.max() if self.buffer else 1.0

        if len(self.buffer) >= self.capacity:
            self.buffer.pop(0)
        self.buffer.append(Transition(state, action, reward, next_state))

        self.priorities[len(self.buffer) - 1] = max_priority

    def sample(self, batch_size):
        # Calculate sampling probabilities
        if len(self.buffer) == self.capacity:
            prios = self.priorities
        else:
            prios = self.priorities[:len(self.buffer)]

        prios = np.clip(prios, EPSILON, None)

        probs = prios ** self.alpha
        probs /= probs.sum() + EPSILON

        # Check for NaN values and replace them
        probs[np.isnan(probs)] = EPSILON

        # Sampling from buffer based on priority probabilities
        indices = np.random.choice(len(self.buffer), batch_size, p=probs)
        samples = [self.buffer[idx] for idx in indices]

        beta = min(1.0, self.beta_start + self.frame * (1.0 - self.beta_start) / self.beta_frames)
        weights = (len(self.buffer) * probs[indices]) ** (-beta)
        weights /= weights.max()
        self.frame += 1

        return samples, indices, np.array(weights, dtype=np.float32)

    def update_priorities(self, indices, priorities):
        for idx, priority in zip(indices, priorities):
            self.priorities[idx] = priority

# ======================================== HYPERPARAMETERS & INITIALIZATION =========================================


GAMMA = 0.99
EPSILON_START = 1.0
EPSILON_END = 0.1
EPSILON_DECAY = 200_000
MEMORY_SIZE = 100_000
BATCH_SIZE = 128
LR = 0.001
TARGET_UPDATE = 1000
NUM_EPISODES = 500


# Initialize
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = DuelingDQNNetwork(2, 5).to(device)
target_model = DuelingDQNNetwork(2, 5).to(device)

# Check if the model file exists and load it
model_path = 'final_model.pth'
if os.path.exists(model_path):
    model.load_state_dict(torch.load(model_path))

target_model.load_state_dict(model.state_dict())
optimizer = optim.Adam(model.parameters(), lr=LR)
memory = PrioritizedReplayMemory(MEMORY_SIZE)
epsilon = EPSILON_START
steps_done = 0
print_every = 100
total_reward = 0
loss = None
beta_start = 0.4
beta_frames = 100000
frame_idx = 0

# ======================================== TRAINING LOOP =========================================

with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
    s.connect(('127.0.0.1', 65431))

    for _ in range(NUM_EPISODES):
        s.sendall(b'reset\n')
        s.sendall(b'get-state\n')
        state = s.recv(1024).decode('utf-8')
        percent, time, velocity = [float(x) for x in state.strip('()').split(',')]
        while True:

            t.sleep(0.01)
            state_tensor = torch.FloatTensor([percent, time]).to(device)

            if random.random() < epsilon:
                action = random.randint(0, 4)
            else:
                with torch.no_grad():
                    action = model(state_tensor).argmax().item()

            # actions with corresponding commands
            if action == 0:
                s.sendall(b'move-sharp-left\n')
            elif action == 1:
                s.sendall(b'move-slight-left\n')
            elif action == 2:
                s.sendall(b'move-straight\n')
            elif action == 3:
                s.sendall(b'move-slight-right\n')
            elif action == 4:
                s.sendall(b'move-sharp-right\n')

            s.sendall(b'get-state\n')
            response = s.recv(1024).decode('utf-8')
            next_percent, next_time, next_velocity = [float(x) for x in response.strip('()').split(',')]
            reward = (next_percent / (next_time + 1)) * 100000

            memory.push(state_tensor, action, reward, torch.FloatTensor([next_percent, next_time]))

            # Learn using the enhanced techniques
            if len(memory) >= BATCH_SIZE:
                transitions, indices, weights = memory.sample(BATCH_SIZE)
                batch = Transition(*zip(*transitions))

                # Double DQN: Get Q-values from the main model and the target model
                state_batch = torch.stack(batch.state).to(device)
                action_batch = torch.tensor(batch.action, dtype=torch.long).to(device)
                reward_batch = torch.tensor(batch.reward).to(device)
                next_state_batch = torch.stack(batch.next_state).to(device)

                # Double DQN logic
                with torch.no_grad():
                    next_state_actions = model(next_state_batch).max(1)[1]
                    next_state_values = target_model(next_state_batch).gather(1, next_state_actions.unsqueeze(-1))
                    expected_q_values = reward_batch + GAMMA * next_state_values.squeeze()

                q_values = model(state_batch)
                state_action_values = q_values.gather(1, action_batch.unsqueeze(-1))

                # Calculate the TD error and then update the model
                td_errors = (state_action_values - expected_q_values.unsqueeze(1)).squeeze().detach().cpu().numpy()
                memory.update_priorities(indices, td_errors)

                loss = (torch.FloatTensor(weights).to(device) * F.mse_loss(state_action_values,
                                                                           expected_q_values.unsqueeze(1))).mean()
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

            epsilon = EPSILON_END + (EPSILON_START - EPSILON_END) * math.exp(-1. * frame_idx / EPSILON_DECAY)
            frame_idx += 1

            total_reward += reward

            # Periodically print out the average reward and current loss for monitoring.
            if frame_idx % print_every == 0:
                avg_reward = total_reward / print_every
                if loss is not None:
                    print(f"Loss: {loss.item():.4f}, ", end="")
                print(f"Average Reward: {avg_reward:.4f}, Epsilon: {epsilon:.4f}")
                total_reward = 0  # Reset accumulated reward after reporting.

            # Periodically update the target model with weights from the main model.
            if frame_idx % TARGET_UPDATE == 0:
                target_model.load_state_dict(model.state_dict())

            if next_velocity <= 0 and next_time >= 200:
                break

            else:
                # If not resetting, update the state to the next state for the next iteration.
                percent, time, velocity = next_percent, next_time, next_velocity

    # after episodes, save model and close socket connection
    torch.save(model.state_dict(), 'final_model.pth')
    s.close()
