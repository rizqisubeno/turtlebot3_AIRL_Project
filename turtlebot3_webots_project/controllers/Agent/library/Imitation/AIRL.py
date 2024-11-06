import numpy as np
import torch as th
import torch.nn as nn
from torch.optim.adam import Adam
import torch.nn.functional as F

import os

from typing import Union, Type, Optional

from torch.utils.tensorboard.writer import SummaryWriter


def disable_gradient(network: nn.Module,
                     inplace:bool=False):
    """
    Disable the gradients of parameters in the network
    """
    for params in network.parameters():
        if(inplace):
            params.requires_grad_(False)
        else:
            params.requires_grad = False

class AIRLDiscriminator(nn.Module):
    """
    Discriminator used by AIRL, which takes s-a pair as input and output
    the probability that the s-a pair is sampled from demonstrations

    Parameters
    ----------
    state_shape: np.array
        shape of the state space
    gamma: float
        discount factor
    hidden_units_r: tuple
        hidden units of the discriminator r
    hidden_units_v: tuple
        hidden units of the discriminator v
    hidden_activation_r: nn.Module
        hidden activation of the discriminator r
    hidden_activation_v: nn.Module
        hidden activation of the discriminator v
    """
    def __init__(self,
                 state_shape,
                 action_shape :         Optional[int],
                 use_action :           bool                = False,
                 gamma :                float                 = 0.9,

                 hidden_units_r :       Union[tuple, int]   = (64, 64),
                 hidden_units_v :       Union[tuple, int]   = (64, 64),
 
                 hidden_activation_r :  nn.Module           = nn.ReLU(),
                 hidden_activation_v :  nn.Module           = nn.ReLU(),
                 ):

        #####################################################################################
        g_net = nn.ModuleList()
        g_net.append(nn.Linear(state_shape+action_shape if use_action else state_shape, 
                               hidden_units_r[0] if isinstance(hidden_units_r, tuple) else hidden_units_r))
        
        if(isinstance(hidden_units_r, tuple)):
            if(len(hidden_units_r)>1):
                g_net.append(hidden_activation_r)
            for i in range(0,len(hidden_units_r)):
                g_net.append(nn.Linear(hidden_units_r[i], 
                                       hidden_units_r[i+1] if i!=len(hidden_units_r)-1 else 1))
                if(i!=len(hidden_units_r)-1):
                    g_net.append(hidden_activation_r)
                
        self.g_net = nn.Sequential(*g_net)

        #####################################################################################
        h_net = nn.ModuleList()
        h_net.append(nn.Linear(state_shape, 
                               hidden_units_v[0] if isinstance(hidden_units_v, tuple) else hidden_units_v))
        
        if(isinstance(hidden_units_v, tuple)):
            if(len(hidden_units_v)>1):
                h_net.append(hidden_activation_v)
            for i in range(0,len(hidden_units_v)):
                h_net.append(nn.Linear(hidden_units_v[i], 
                                       hidden_units_v[i+1] if i!=len(hidden_units_v)-1 else 1))
                if(i!=len(hidden_units_v)-1):
                    h_net.append(hidden_activation_v)
                
        self.h_net = nn.Sequential(*h_net)

        self.gamma = gamma

    def f(self, 
          states: th.Tensor, 
          dones: th.Tensor, 
          next_states: th.Tensor) -> th.Tensor:
        """
        Calculate the f(s, s') function

        Parameters
        ----------
        states: torch.Tensor
            input states
        dones: torch.Tensor
            whether the state is the end of an episode
        next_states: torch.Tensor
            next state corresponding to the current state

        Returns
        -------
        f: value of the f(s, s') function
        """
        rs = self.g_net(states)
        vs = self.h_net(states)
        next_vs = self.h_net(next_states)
        return rs + self.gamma * (1 - dones) * next_vs - vs

    def forward(
            self,
            states: th.Tensor,
            dones: th.Tensor,
            log_pis: th.Tensor,
            next_states: th.Tensor
    ) -> th.Tensor:
        """
        Output the discriminator's result sigmoid(f - log_pi) without sigmoid

        Parameters
        ----------
        states: torch.Tensor
            input states
        dones: torch.Tensor
            whether the state is the end of an episode
        log_pis: torch.Tensor
            log(\pi(a|s))
        next_states: torch.Tensor
            next state corresponding to the current state

        Returns
        -------
        result: f - log_pi

        Description
        -----------
                                 e^(f)                      pi
        from loss = E_D (log ------------) - E_pi (log -----------)
                              e^(f) + pi                e^(f) + pi

        if we mixture expected feature from Demonstration and generator then

                        e^(f)       e^(f) + pi           
        loss = (log ------------ x -----------)
                      e^(f) + pi        pi     
        loss = log (e^(f)) - log pi  = f - log pi
        """
        return self.f(states, dones, next_states) - log_pis

    def calculate_reward(
            self,
            states: th.Tensor,
            dones: th.Tensor,
            log_pis: th.Tensor,
            next_states: th.Tensor,
            like_gail: Optional[bool]=False,
    ) -> th.Tensor:
        """
        Calculate reward using AIRL's learned reward signal f

        Parameters
        ----------
        states: torch.Tensor
            input states
        dones: torch.Tensor
            whether the state is the end of an episode
        log_pis: torch.Tensor
            log(\pi(a|s))
        next_states: torch.Tensor
            next state corresponding to the current state
        like_gail: boolean
            if True then calculate reward based on repo https://github.com/toshikwa/gail-airl-ppo.pytorch
            otherwise calculate reward only from f function based on repo https://github.com/HumanCompatibleAI/imitation

        Returns
        -------
        rewards: torch.Tensor
            reward signal
        """
        if (like_gail):
            with th.no_grad():
                logits = self.forward(states, dones, log_pis, next_states)
            return -F.logsigmoid(-logits)
        else:
            return self.f(states,
                          dones,
                          next_states)


class AIRL():
    """
    Implementation of AIRL, using PPO as the backbone RL algorithm

    Reference:
    ----------
    [1] Fu, J., Luo, K., and Levine, S.
    Learning robust rewards with adversarial inverse reinforcement learning.
    In International Conference on Learning Representations, 2018.

    Parameters
    ----------
    irl_config: Dict
        configuration of IRL 
    buffer_exp: SerializedBuffer
        buffer of demonstrations
    state_shape: np.array
        shape of the state space
    action_shape: np.array
        shape of the action space
    device: torch.device
        cpu or cuda
    seed: int
        random seed
    gamma: float
        discount factor
    rollout_length: int
        rollout length of the buffer
    mix_buffer: int
        times for rollout buffer to mix
    batch_size: int
        batch size for sampling from current policy and demonstrations
    lr_actor: float
        learning rate of the actor
    lr_critic: float
        learning rate of the critic
    lr_disc: float
        learning rate of the discriminator
    units_actor: tuple
        hidden units of the actor
    units_critic: tuple
        hidden units of the critic
    units_disc_r: tuple
        hidden units of the discriminator r
    units_disc_v: tuple
        hidden units of the discriminator v
    epoch_ppo: int
        at each update period, update ppo for these times
    epoch_disc: int
        at each update period, update the discriminator for these times
    clip_eps: float
        clip coefficient in PPO's objective
    lambd: float
        lambd factor
    coef_ent: float
        entropy coefficient
    max_grad_norm: float
        maximum gradient norm
    """
    def __init__(self, 
                irl_config,):
                


