import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Normal, TransformedDistribution
from torch.distributions.transforms import TanhTransform

import numpy as np

# class RSNorm:
#     def __init__(self, input_dim, epsilon=1e-6):
#         # Initialize running mean and import warnings
#         self.mu = torch.zeros(input_dim)  # Running mean
#         self.variance = torch.ones(input_dim)  # Running variance
#         self.epsilon = epsilon  # Small constant for numerical stability
#         self.t = 1 # Time step, starting at 1
#         self.last_dim = 2

#     def type_check(self, o_t):
#         if type(o_t)==np.ndarray:
#             if o_t.ndim==1:
#                 return torch.from_numpy(o_t)
#             else:
#                 self.last_dim = o_t.ndim
#                 return torch.from_numpy(o_t.flatten())
#         elif type(o_t)==list:
#             return torch.tensor(o_t)
#         elif type(o_t)==torch.tensor:
#             return o_t

#     def update(self, o_t):
#         """
#         Update the running mean and variance with the new observation.
        
#         Args:
#         - o_t (numpy array): New observation vector of shape (input_dim,).
#         """
#         o_t = self.type_check(o_t)
#         delta = o_t - self.mu  # Difference from the current mean

#         self.mu += torch.div(delta,self.t)  # Update mean
#         self.variance = ((self.t - 1) / self.t)*(self.variance + ((delta**2) / self.t))  # Update variance
#         self.t += 1 # Increment timestep

#     def normalize(self, o_t):
#         """
#         Normalize the input observation using the current running mean and variance.
        
#         Args:
#         - o_t (numpy array): Observation vector to normalize.
        
#         Returns:
#         - o_norm (numpy array): Normalized observation vector.
#         """
#         o_t = self.type_check(o_t)
#         norm = (o_t - self.mu) / torch.sqrt(self.variance + self.epsilon)
#         if (np.asarray(norm).ndim!=self.last_dim):
#             return np.asarray(norm).reshape(-1, 1).T
#         return np.asarray(norm)

# # Example usage:
# input_dim = 4  # Dimension of input observations
# rsnorm = RSNorm(input_dim)

# # Simulate some input observations
# observations = [np.random.rand(input_dim) for _ in range(100)]

# # Update running statistics and normalize each observation
# normalized_observations = []
# for o_t in observations:
#     rsnorm.update(o_t)  # Update running mean and variance
#     o_norm = rsnorm.normalize(o_t)  # Normalize the observation
#     normalized_observations.append(o_norm)

# Orthogonal initialization function
def orthogonal_init(gain=1.0):
    def init_weights(m):
        if isinstance(m, nn.Linear):
            nn.init.orthogonal_(m.weight, gain=gain)
            if m.bias is not None:
                nn.init.zeros_(m.bias)
    return init_weights

class MLPBlock(nn.Module):
    def __init__(self, hidden_dim, dtype=torch.float32):
        super(MLPBlock, self).__init__()
        self.fc1 = nn.Linear(hidden_dim, hidden_dim * 4)
        self.fc2 = nn.Linear(hidden_dim * 4, hidden_dim)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        x = self.relu(x)
        return x

class ResidualBlock(nn.Module):
    def __init__(self, hidden_dim, dtype=torch.float32):
        super(ResidualBlock, self).__init__()
        self.norm = nn.LayerNorm(hidden_dim)
        self.fc1 = nn.Linear(hidden_dim, hidden_dim * 4)
        self.fc2 = nn.Linear(hidden_dim * 4, hidden_dim)
        self.relu = nn.ReLU()

        # Apply He Normal initialization to the dense layers
        nn.init.kaiming_normal_(self.fc1.weight, mode='fan_in', nonlinearity='relu')
        nn.init.kaiming_normal_(self.fc2.weight, mode='fan_in', nonlinearity='relu')

    def forward(self, x):
        res = x
        x = self.norm(x)
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        return res + x

class SACEncoder(nn.Module):
    def __init__(self, 
                 block_type,
                 input_dim, 
                 num_blocks, 
                 hidden_dim, 
                 dtype=torch.float32):
        super(SACEncoder, self).__init__()
        self.block_type = block_type
        self.num_blocks = num_blocks
        self.hidden_dim = hidden_dim
        self.dtype = dtype

        if self.block_type == "mlp":
            self.encoder = MLPBlock(hidden_dim, dtype=dtype)
        elif self.block_type == "residual":
            self.fc = nn.Linear(input_dim, hidden_dim)
            self.fc.apply(orthogonal_init(1.0))
            self.blocks = nn.ModuleList(
                [ResidualBlock(hidden_dim, dtype=dtype) for _ in range(self.num_blocks)]
            )
            self.norm = nn.LayerNorm(hidden_dim)

    def forward(self, x):
        if self.block_type == "mlp":
            x = self.encoder(x)
        elif self.block_type == "residual":
            x = self.fc(x)
            for block in self.blocks:
                x = block(x)
            x = self.norm(x)
        return x

# class SACActor(nn.Module):
#     def __init__(self, block_type, num_blocks, hidden_dim, action_dim, dtype=torch.float32):
#         super(SACActor, self).__init__()
#         self.encoder = SACEncoder(block_type, num_blocks, hidden_dim, dtype=dtype)
#         self.predictor = NormalTanhPolicy(action_dim, dtype=dtype)

#     def forward(self, observations, temperature=1.0):
#         observations = observations.to(dtype=self.encoder.dtype)
#         z = self.encoder(observations)
#         dist = self.predictor(z, temperature)
#         return dist

# class NormalTanhPolicy(nn.Module):
#     def __init__(self, action_dim, state_dependent_std=True, kernel_init_scale=1.0,
#                  log_std_min=-10.0, log_std_max=2.0, dtype=torch.float32):
#         super(NormalTanhPolicy, self).__init__()
#         self.action_dim = action_dim
#         self.state_dependent_std = state_dependent_std
#         self.log_std_min = log_std_min
#         self.log_std_max = log_std_max
#         self.dtype = dtype

#         # Mean layer
#         self.means_layer = nn.Linear(action_dim, action_dim)
#         self.means_layer.apply(orthogonal_init(kernel_init_scale))

#         # Standard deviation layer
#         if self.state_dependent_std:
#             self.log_stds_layer = nn.Linear(action_dim, action_dim)
#             self.log_stds_layer.apply(orthogonal_init(kernel_init_scale))
#         else:
#             # Fixed log standard deviation
#             self.register_parameter("log_stds", nn.Parameter(torch.zeros(action_dim), requires_grad=True))

#     def forward(self, inputs, temperature=1.0):
#         inputs = inputs.to(dtype=self.dtype)
#         means = self.means_layer(inputs)

#         if self.state_dependent_std:
#             log_stds = self.log_stds_layer(inputs)
#         else:
#             log_stds = self.log_stds

#         # Clamping log_stds to avoid instability
#         log_stds = self.log_std_min + (self.log_std_max - self.log_std_min) * 0.5 * (1 + torch.tanh(log_stds))

#         # Create the Normal distribution
#         stds = torch.exp(log_stds) * temperature
#         dist = Normal(loc=means, scale=stds)

#         # Apply the Tanh transformation
#         tanh_dist = TransformedDistribution(dist, [TanhTransform()])

#         return tanh_dist