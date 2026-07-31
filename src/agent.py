import random
import torch
import torch.nn as nn
import torch.optim as optim

from config.settings import (
    STATE_DIMENSION,
    ACTION_DIMENSION,
    LEARNING_RATE,
    GAMMA,
    BATCH_SIZE
)
from src.models import DuelingNoisyDQN
from src.buffer import PrioritizedReplayBuffer


class ProductionDDQNAgent:
    """Enterprise-grade Deep Q-Network Trading Agent utilizing Dueling Architectures and PER"""
    def __init__(self, state_dim=STATE_DIMENSION, action_dim=ACTION_DIMENSION):
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # Dual Network Topology Setup
        self.policy_net = DuelingNoisyDQN(self.state_dim, self.action_dim).to(self.device)
        self.target_net = DuelingNoisyDQN(self.state_dim, self.action_dim).to(self.device)
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.target_net.eval() # Target network parameters are updated strictly via synchronization
        
        # Optimizer and Prioritized Experience Replay Engine
        self.optimizer = optim.Adam(self.policy_net.parameters(), lr=LEARNING_RATE)
        self.memory = PrioritizedReplayBuffer()
        
        self.gamma = GAMMA
        self.batch_size = BATCH_SIZE

    def select_action(self, state):
        """
        Determines target trading execution action. 
        Explicit Epsilon-Greedy loops are removed; internal noise matrices handle exploration.
        """
        with torch.no_grad():
            # Ensure evaluation constraints if selecting actions for test deployment or live runs
            state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)
            q_values = self.policy_net(state_tensor)
            return q_values.argmax(dim=-1).item()

    def update_policy(self):
        """Performs a Double DQN policy gradient descent step utilizing PER Importance Sampling weights"""
        # Ensure the replay buffer contains sufficient experience frames
        if self.memory.tree.n_entries < self.batch_size:
            return 0.0
            
        # Sample prioritization batch arrays
        states, actions, rewards, next_states, dones, idxs, is_weights = self.memory.sample(self.batch_size)
        
        # Vectorized Type Conversion to GPU/CPU Tensor structures
        states = torch.FloatTensor(states).to(self.device)
        actions = torch.LongTensor(actions).unsqueeze(1).to(self.device)
        rewards = torch.FloatTensor(rewards).to(self.device)
        next_states = torch.FloatTensor(next_states).to(self.device)
        dones = torch.FloatTensor(dones).to(self.device)
        is_weights = torch.FloatTensor(is_weights).to(self.device)

        # 1. Fetch current step Q-value predictions for chosen actions
        current_q_values = self.policy_net(states).gather(1, actions).squeeze(1)
        
        # 2. Double DQN Action Target Value Extraction Loop
        with torch.no_grad():
            # A: Pick the absolute best action from the ONLINE network
            next_state_online_actions = self.policy_net(next_states).argmax(dim=-1).unsqueeze(1)
            # B: Evaluate the expected value of that choice using the TARGET network
            next_q_values = self.target_net(next_states).gather(1, next_state_online_actions).squeeze(1)
            # C: Compute expected temporal difference value targeting metrics
            target_q_values = rewards + (self.gamma * next_q_values * (1.0 - dones))
        
        # 3. Compute Temporal Difference Errors for priority weights re-calibration
        td_errors = (current_q_values - target_q_values).detach().cpu().numpy()
        self.memory.update_priorities(idxs, td_errors)
        
        # 4. Compute Weighted Mean Squared Error Loss using Importance Sampling scaling
        loss = (is_weights * (current_q_values - target_q_values).pow(2)).mean()
        
        # 5. Execute Optimization Optimization updates
        self.optimizer.zero_grad()
        loss.backward()
        
        # Structural Gradient Clipping to prevent exploding networks on unexpected market spikes
        nn.utils.clip_grad_norm_(self.policy_net.parameters(), max_norm=10.0)
        self.optimizer.step()
        
        # 6. Cycle state space Gaussian factor variations after weight adjustment optimization routines
        self.policy_net.reset_noise()
        self.target_net.reset_noise()
        
        return loss.item()

    def synchronize_target_network(self):
        """Hard synchronization copy connecting active target weight parameters"""
        self.target_net.load_state_dict(self.policy_net.state_dict())
