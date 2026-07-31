import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class NoisyLinear(nn.Module):
    """
    Noisy Linear Layer using Factorized Gaussian Noise.
    Replaces traditional Epsilon-Greedy exploration by injecting 
    learnable perturbations directly into network parameter weights.
    """
    def __init__(self, in_features, out_features, std_init=0.5):
        super(NoisyLinear, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.std_init = std_init

        # Learnable mean and standard deviation parameters
        self.weight_mu = nn.Parameter(torch.empty(out_features, in_features))
        self.weight_sigma = nn.Parameter(torch.empty(out_features, in_features))
        self.bias_mu = nn.Parameter(torch.empty(out_features))
        self.bias_sigma = nn.Parameter(torch.empty(out_features))

        # Static noise execution matrix buffers
        self.register_buffer('weight_epsilon', torch.empty(out_features, in_features))
        self.register_buffer('bias_epsilon', torch.empty(out_features))
        
        self.reset_parameters()
        self.reset_noise()

    def reset_parameters(self):
        """Initializes weight parameters using uniform distribution ranges"""
        mu_range = 1.0 / math.sqrt(self.in_features)
        self.weight_mu.data.uniform_(-mu_range, mu_range)
        self.weight_sigma.data.fill_(self.std_init / math.sqrt(self.in_features))
        
        self.bias_mu.data.uniform_(-mu_range, mu_range)
        self.bias_sigma.data.fill_(self.std_init / math.sqrt(self.out_features))

    def _scale_noise(self, size):
        """Generates factorized noise scaling factors"""
        x = torch.randn(size, device=self.weight_mu.device)
        return x.sign().mul_(x.abs().sqrt())

    def reset_noise(self):
        """Regenerates fresh random perturbations across active noise matrices"""
        epsilon_in = self._scale_noise(self.in_features)
        epsilon_out = self._scale_noise(self.out_features)
        
        # Outer product factorization reduces memory footprints from O(N^2) to O(N)
        self.weight_epsilon.copy_(epsilon_out.outer(epsilon_in))
        self.bias_epsilon.copy_(epsilon_out)

    def forward(self, x):
        """Applies stochastic parameters during training and deterministic weights during inference"""
        if self.training:
            return F.linear(
                x, 
                self.weight_mu + self.weight_sigma * self.weight_epsilon, 
                self.bias_mu + self.bias_sigma * self.bias_epsilon
            )
        else:
            return F.linear(x, self.weight_mu, self.bias_mu)


class DuelingNoisyDQN(nn.Module):
    """
    Dueling Deep Q-Network.
    Splits feature evaluation streams into state values and action advantages 
    to make more robust policy choices in stable or unvaried states.
    """
    def __init__(self, state_dim, action_dim):
        super(DuelingNoisyDQN, self).__init__()
        
        # Shared feature extraction backbone layer
        self.feature_layer = nn.Sequential(
            nn.Linear(state_dim, 128),
            nn.ReLU()
        )
        
        # Stream 1: State Value V(s) calculation module
        self.value_stream = nn.Sequential(
            NoisyLinear(128, 128),
            nn.ReLU(),
            NoisyLinear(128, 1)
        )
        
        # Stream 2: Action Advantage A(s, a) calculation module
        self.advantage_stream = nn.Sequential(
            NoisyLinear(128, 128),
            nn.ReLU(),
            NoisyLinear(128, action_dim)
        )

    def forward(self, state):
        """Evaluates incoming observation tensor states to output absolute action Q-values"""
        features = self.feature_layer(state)
        values = self.value_stream(features)
        advantages = self.advantage_stream(features)
        
        # Combine streams using mean-centering subtraction formula for numerical stability
        q_values = values + (advantages - advantages.mean(dim=-1, keepdim=True))
        return q_values

    def reset_noise(self):
        """Traverses the model modules to trigger noise resets across all embedded layers"""
        for module in self.modules():
            if isinstance(module, NoisyLinear):
                module.reset_noise()
