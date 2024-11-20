import numpy as np
import torch as th
import torch.nn as nn
from torch.optim.adam import Adam
import torch.nn.functional as F

import os

from typing import Union, Type, Optional
from types import SimpleNamespace

import hydra
from hydra.utils import instantiate
from omegaconf import OmegaConf

from torch.utils.tensorboard.writer import SummaryWriter

from ..custom_rl_algo import Logger, PPO


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
                 gamma :                float                 = 0.99,

                 hidden_units_r :       Union[tuple, int]   = (64, 64),
                 hidden_units_v :       Union[tuple, int]   = (64, 64),
 
                 hidden_activation_r :  nn.Module           = nn.ReLU(),
                 hidden_activation_v :  nn.Module           = nn.ReLU(),
                 ):

        #####################################################################################
        super(AIRLDiscriminator, self).__init__()
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
        r"""
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
            reward_type: str = 'base',
    ) -> th.Tensor:
        r"""
        Calculate reward using AIRL's reward function

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
        reward_type: boolean
            if "gail" then calculate reward based on repo https://github.com/toshikwa/gail-airl-ppo.pytorch
            else if "airl_shaped" calculate reward only from f function based on repo https://github.com/HumanCompatibleAI/imitation
            else if "airl_base" calculate reward from self.g_net without self.h_net

        Returns
        -------
        rewards: torch.Tensor
            reward signal
        """
        if ("gail" in reward_type):
            with th.no_grad():
                logits = self.forward(states, dones, log_pis, next_states)
            return -F.logsigmoid(-logits)
        elif ("airl_shaped" in reward_type):
            return self.f(states, dones, next_states)
        elif ("airl_base" in reward_type):
            return self.g_net(states)
        else:
            return SystemError("reward_type is not defined")


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
                 env, 
                 config_path: str,
                 config_name: str):

        self.logger = Logger()
        self.env = env
        rl_params_cfg, irl_params_cfg = self.hydra_params_read(config_path, config_name)
        self.irl_params = self.read_param_cfg(params_cfg=irl_params_cfg)
        self.rl_params = self.read_param_cfg(params_cfg=rl_params_cfg)

        self.device = th.device("cuda") if self.irl_params.cuda_en else th.device("cpu")

        self.rl_algo = PPO(self.env,
                           config_path=None,
                           config_name=None,
                           bypass_class_cfg=True,
                           rl_params=self.rl_params)

        # checking the ppo buffer_size must be same with irl buffer_size 
        # for balancing when learning discriminator for expert buffer
        assert int(self.rl_algo.minibatch_size) == int(self.irl_params.batch_size)

        #checking mismatch each other irl and rl params
        assert abs(float(self.rl_params.gamma)-float(self.irl_params.gamma))<=1e-5

        # Expert's buffer.
        self.buffer_exp = instantiate(self.irl_params.buffer_exp,
                                      buffer_size=self.irl_params.batch_size,
                                      state_shape=self.env.single_observation_space.shape,
                                      action_shape=self.env.single_action_space.shape,
                                      device=self.device
                                    )
        print(self.buffer_exp)

        print(f"{tuple(self.irl_params.units_disc_r[0])=}")
        print(f"{tuple(self.irl_params.units_disc_v[0])=}")
        # Discriminator.
        self.disc = AIRLDiscriminator(
            state_shape=self.env.single_observation_space.shape[0],
            action_shape=self.env.single_action_space.shape[0],
            gamma=self.irl_params.gamma,
            hidden_units_r=tuple(self.irl_params.units_disc_r[0]),
            hidden_units_v=tuple(self.irl_params.units_disc_v[0]),
            hidden_activation_r=nn.ReLU(inplace=True),
            hidden_activation_v=nn.ReLU(inplace=True)
        ).to(self.device)

        self.learning_steps_disc = 0
        self.optim_disc = instantiate(self.irl_params.disc_optimizer, 
                                      params=self.disc.parameters())
        self.batch_size = self.irl_params.batch_size
        self.epoch_disc = self.irl_params.epoch_disc

    def load_buffer_data(self, path):
        self.buffer_exp.load(path)

    def hydra_params_read(self,
                          config_path: str,
                          config_name: str):
        config_path = "../../"+config_path if config_path[:2]=="./" else \
                      "../../"+config_path if config_path[:1]!="." else \
                      config_path
        print(config_path)
        # initialize hydra and load the configuration
        with hydra.initialize(config_path=config_path,
                              version_base="1.2"):
            cfg = hydra.compose(config_name=config_name)
        dict_cfg = OmegaConf.to_object(cfg)

        #re-formatted dict
        rl_new_cfg = {}
        irl_new_cfg = {}
        for item in dict_cfg.keys():
            if("irl" in item):
                for sub_item in dict_cfg[item].keys():
                    irl_new_cfg[sub_item] = dict_cfg[item][sub_item]
            elif("rl" in item):
                for sub_item in dict_cfg[item].keys():
                    rl_new_cfg[sub_item] = dict_cfg[item][sub_item]

        #adding manual 
        rl_new_cfg['exp_name']= "airl_rl_ppo"
        irl_new_cfg['exp_name']= dict_cfg['exp_name']
        return rl_new_cfg, irl_new_cfg
    
    def read_param_cfg(self, params_cfg, verbose:Optional[bool]=True):
        cfg_namespace = SimpleNamespace(**params_cfg)
        if(verbose):
            for key, val in params_cfg.items():
                self.logger.print("info", f"{key} :\t{val}")
        return cfg_namespace  

    def train(self, 
              num_steps: int = 1e6):
        
        for step in range(1, num_steps + 1):

            for _ in range(self.irl_params.epoch_disc):
                self.learning_steps_disc += 1

                # doing rollout trajectory from current policy
                self.rl_algo.do_rollout()

                # Samples from current policy's trajectories.
                states, _, _, dones, log_pis, next_states = \
                    self.rl_algo.rl_buffer.sample(self.batch_size)
                # Samples from expert's demonstrations.
                states_exp, actions_exp, _, dones_exp, next_states_exp = \
                    self.buffer_exp.sample(self.batch_size)
                # Calculate log probabilities of expert actions.
                with th.no_grad():
                    _, log_pis_exp, _, _ = self.rl_algo.agent.get_action_and_value(states_exp, actions_exp)
                    # log_pis_exp = self.actor.evaluate_log_pi(
                    #     states_exp, actions_exp)

                # Update discriminator.
                self.update_disc(
                    states, dones, log_pis, next_states, states_exp,
                    dones_exp, log_pis_exp, next_states_exp, writer
                )

            # We don't use reward signals here,
            states, actions, _, dones, log_pis, next_states = self.buffer.get()

            # Calculate rewards.
            rewards = self.disc.calculate_reward(
                states, dones, log_pis, next_states)

            # Update PPO using estimated rewards.
            self.update_ppo(
                states, actions, rewards, dones, log_pis, next_states, writer)

    # def update_disc(self, states, dones, log_pis, next_states,
    #                 states_exp, dones_exp, log_pis_exp,
    #                 next_states_exp, writer):
    #     # Output of discriminator is (-inf, inf), not [0, 1].
    #     logits_pi = self.disc(states, dones, log_pis, next_states)
    #     logits_exp = self.disc(
    #         states_exp, dones_exp, log_pis_exp, next_states_exp)

    #     # Discriminator is to maximize E_{\pi} [log(1 - D)] + E_{exp} [log(D)].
    #     loss_pi = -F.logsigmoid(-logits_pi).mean()
    #     loss_exp = -F.logsigmoid(logits_exp).mean()
    #     loss_disc = loss_pi + loss_exp

    #     self.optim_disc.zero_grad()
    #     loss_disc.backward()
    #     self.optim_disc.step()

    #     if self.learning_steps_disc % self.epoch_disc == 0:
    #         writer.add_scalar(
    #             'loss/disc', loss_disc.item(), self.learning_steps)

    #         # Discriminator's accuracies.
    #         with torch.no_grad():
    #             acc_pi = (logits_pi < 0).float().mean().item()
    #             acc_exp = (logits_exp > 0).float().mean().item()
    #         writer.add_scalar('stats/acc_pi', acc_pi, self.learning_steps)
    #         writer.add_scalar('stats/acc_exp', acc_exp, self.learning_steps)


