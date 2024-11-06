from typing import Callable, Dict, List, Tuple, Optional

from gymnasium import spaces
import torch as th
from torch import nn

from stable_baselines3.common.policies import ActorCriticPolicy

class ResidualBlock(nn.Module):
    def __init__(self, in_channels, hidden_channels, act:Optional[nn.Module]=nn.LeakyReLU()):
        super(ResidualBlock, self).__init__()
        # First fully connected layer followed by activation
        self.fc1 = nn.Linear(in_channels, hidden_channels)
        self.act1 = act
        
        # Second fully connected layer followed by a Linear activation
        self.fc2 = nn.Linear(hidden_channels, in_channels)
        
        # Final activation after the skip connection
        self.act2 = act

    def forward(self, x):
        identity = x  # Save input for the residual connection

        # First FC layer + activation
        out = self.fc1(x)
        out = self.act1(out)

        # Second FC layer (Linear without activation)
        out = self.fc2(out)

        # Add the input (residual connection)
        out += identity
        
        # Final activation
        out = self.act2(out)

        return out

class residualNetwork(nn.Module):
    """
    Custom network for policy and value function.
    It receives as input the features extracted by the features extractor.

    :param feature_dim: dimension of the features extracted with the features_extractor (e.g. features from a CNN)
    :param last_layer_dim_pi: (int) number of units for the last layer of the policy network
    :param last_layer_dim_vf: (int) number of units for the last layer of the value network
    """

    def __init__(
        self,
        feature_dim: int,
        last_layer_dim_pi: int = 64,
        last_layer_dim_vf: int = 64,
    ):
        super().__init__()

        self.resblock1 = ResidualBlock(feature_dim, 256)
        self.fc_cat1 = nn.Sequential(nn.Linear(feature_dim*2, feature_dim),
                                     nn.ReLU())

        self.resblock2 = ResidualBlock(feature_dim, 32)
        self.fc_cat2 = nn.Sequential(nn.Linear(feature_dim*2, feature_dim),
                                     nn.ReLU())

        # IMPORTANT:
        # Save output dimensions, used to create the distributions
        self.latent_dim_pi = last_layer_dim_pi
        self.latent_dim_vf = last_layer_dim_vf

        # Policy network
        self.policy_net = nn.Sequential(
            nn.Linear(feature_dim, last_layer_dim_pi), nn.ReLU()
        )
        # Value network
        self.value_net = nn.Sequential(
            nn.Linear(feature_dim, last_layer_dim_vf), nn.ReLU()
        )

    def forward(self, features: th.Tensor) -> Tuple[th.Tensor, th.Tensor]:
        """
        :return: (th.Tensor, th.Tensor) latent_policy, latent_value of the specified network.
            If all layers are shared, then ``latent_policy == latent_value``
        """
        return self.forward_actor(features), self.forward_critic(features)

    def forward_actor(self, features: th.Tensor) -> th.Tensor:
        
        # First Residual Block
        res1 = self.resblock1(features)
        concat1 = th.cat([features, res1], dim=1)
        concat1 = self.fc_cat1(concat1)
    
        # Second Residual Block
        res2 = self.resblock2(concat1)
        concat2 = th.cat([concat1, res2], dim=1)
        concat2 = self.fc_cat2(concat2)

        return self.policy_net(concat2)

    def forward_critic(self, features: th.Tensor) -> th.Tensor:
        
        # First Residual Block
        res1 = self.resblock1(features)
        concat1 = th.cat([features, res1], dim=1)
        concat1 = self.fc_cat1(concat1)
    
        # Second Residual Block
        res2 = self.resblock2(concat1)
        concat2 = th.cat([concat1, res2], dim=1)
        concat2 = self.fc_cat2(concat2)
        
        return self.value_net(concat2)


class CustomActorCriticPolicy(ActorCriticPolicy):
    def __init__(
        self,
        observation_space: spaces.Space,
        action_space: spaces.Space,
        lr_schedule: Callable[[float], float],
        *args,
        **kwargs,
    ):
        # Disable orthogonal initialization
        kwargs["ortho_init"] = False
        super().__init__(
            observation_space,
            action_space,
            lr_schedule,
            # Pass remaining arguments to base class
            *args,
            **kwargs,
        )


    def _build_mlp_extractor(self) -> None:
        self.mlp_extractor = residualNetwork(self.features_dim)
