import os
import time
from types import SimpleNamespace
from typing import Optional, Type, Union

import gymnasium as gym

import hydra
import numpy as np
import torch as th
import torch.nn as nn
import torch.nn.functional as F
from hydra.utils import instantiate
from omegaconf import OmegaConf
from torch.optim.adam import Adam
from torch.utils.tensorboard.writer import SummaryWriter

from ..custom_rl_algo import PPO, Logger


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
                 env : gym.Env,
                 use_action :           bool                = False,
                 gamma :                float                 = 0.99,

                 hidden_units_r :       Union[tuple, int]   = (64, 64),
                 hidden_units_v :       Union[tuple, int]   = (64, 64),
                 hidden_units_g :       Union[tuple, int]   = (64, 64),
 
                 hidden_activation_r :  nn.Module           = nn.ReLU(),
                 hidden_activation_v :  nn.Module           = nn.ReLU(),
                 hidden_activation_g :  nn.Module           = nn.ReLU(),
                 ):

        #####################################################################################
        super(AIRLDiscriminator, self).__init__()
        self.env = env
        state_shape = np.prod(self.env.single_observation_space.shape)
        action_shape = np.prod(self.env.single_action_space.shape)
        # print(f"{state_shape=}")
        # print(f"{action_shape=}")
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

        #####################################################################################
        goal_net = nn.ModuleList()
        goal_net.append(nn.Linear(state_shape, 
                                  hidden_units_g[0] if isinstance(hidden_units_g, tuple) else hidden_units_g))
        
        if(isinstance(hidden_units_g, tuple)):
            if(len(hidden_units_g)>1):
                goal_net.append(hidden_activation_g)
            for i in range(0,len(hidden_units_g)):
                goal_net.append(nn.Linear(hidden_units_g[i], 
                                       hidden_units_g[i+1] if i!=len(hidden_units_g)-1 else 3))
                if(i!=len(hidden_units_g)-1):
                    goal_net.append(hidden_activation_g)
        
        goal_net.append(nn.Softmax(dim=1))
        self.goal_net = nn.Sequential(*goal_net)
        score_collision = -1
        score_goal = 1
        score_while_reaching = 0
        self.label_mapping = lambda x: th.where(x == 0, score_while_reaching, th.where(x == 1, score_collision, score_goal))

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

        goal_rew = self.label_mapping(self.goal_f(next_states,
                                                    no_grad=True).argmax(dim=1).type(th.int8)).unsqueeze(1)
        # print(f"{next_states=}")
        # print(f"{goal_rew=}")
        return rs + self.gamma * (1 - dones) * next_vs - vs + goal_rew

    def goal_f(self,
               states: th.Tensor,
               no_grad:bool=False) -> th.Tensor:
        if no_grad==True:
            with th.no_grad():
                return self.goal_net(states)
        return self.goal_net(states)

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
            reward_type: str = 'airl_base',
            no_grad: bool = False
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
            # return self.f(states, dones, next_states)
            # print(f"{states.shape=}")
            # print(f"{dones.shape=}")
            # print(f"{log_pis.shape=}")
            # print(f"{next_states.shape=}")
            if(no_grad==True):
                with th.no_grad():
                    return self.forward(states, dones, log_pis, next_states)
            return self.forward(states, dones, log_pis, next_states)
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
                 config_name: str,
                 expert_path: str):

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
        # self.buffer_exp = instantiate(self.irl_params.buffer_exp,
        #                               buffer_size=self.irl_params.batch_size,   #actually is larger than batch_size only for init
        #                               state_shape=self.env.single_observation_space.shape,
        #                               action_shape=self.env.single_action_space.shape,
        #                               device=self.device
        #                             )

        #try using modifiedBuffer
        self.buffer_exp = instantiate(self.irl_params.buffer_exp,
                                      device=self.device)

        print(self.buffer_exp)

        print(f"{tuple(self.irl_params.units_disc_r[0])=}")
        print(f"{tuple(self.irl_params.units_disc_v[0])=}")
        # Discriminator.
        self.disc = AIRLDiscriminator(
            env=self.env,
            gamma=self.irl_params.gamma,
            hidden_units_r=tuple(self.irl_params.units_disc_r[0]),
            hidden_units_v=tuple(self.irl_params.units_disc_v[0]),
            hidden_units_g=tuple(self.irl_params.units_disc_g[0]),
            hidden_activation_r=nn.ReLU(inplace=True),
            hidden_activation_v=nn.ReLU(inplace=True),
            hidden_activation_g=nn.ReLU(inplace=True),
        ).to(self.device)

        self.learning_steps_disc = 0
        self.optim_disc = instantiate(self.irl_params.disc_optimizer, 
                                      params=list(self.disc.g_net.parameters())+list(self.disc.h_net.parameters()))
        self.optim_goal_net = instantiate(self.irl_params.goal_net_optimizer, 
                                      params=self.disc.goal_net.parameters())
        self.batch_size = self.irl_params.batch_size
        self.epoch_disc = self.irl_params.epoch_disc

        self.writer = SummaryWriter(f"runs/{self.irl_params.exp_name}")
        self.writer.add_text(                
            "irl_hyperparameters",
            "|param|value|\n|-|-|\n%s" % (
                    "\n".join([f"|{key}|{value}|" for key, value in vars(self.irl_params).items()])),
            )

        self.scenario_idx_now = 0
        self.expert_path = expert_path
        self.load_buffer_data(self.expert_path+f"/trajectory_id_{self.scenario_idx_now}.pth")

    def load_buffer_data(self, path):
        self.buffer_exp.load(path)
        print(f"expert data: {path} loaded!")

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
        print(f"{dict_cfg=}")

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
        
        iter = 0
        self.n_steps = 0
        while self.n_steps<num_steps:
            #change scenario expert if rollout environment changed
            #using environment on rl algo == PPO because env rl_algo using for rollout
            if(self.env == self.env.unwrapped):
                if self.scenario_idx_now != self.rl_algo.env.envs[0].scenario_idx:
                    self.scenario_idx_now = self.rl_algo.env.envs[0].scenario_idx
                    self.load_buffer_data(self.expert_path+f"/trajectory_id_{self.scenario_idx_now}.pth")
            else:
                if self.scenario_idx_now != self.rl_algo.env.envs[0].unwrapped.scenario_idx:
                    self.scenario_idx_now = self.rl_algo.env.envs[0].unwrapped.scenario_idx
                    self.load_buffer_data(self.expert_path+f"/trajectory_id_{self.scenario_idx_now}.pth")

            # doing rollout trajectory from current policy
            if (self.n_steps == 0):
                # TRY NOT TO MODIFY: start the game
                self.rl_algo.global_step = 0
                self.rl_algo.start_time = time.time()
                next_ob, _ = self.env.reset(seed=self.irl_params.seed)
                next_ob = th.Tensor(next_ob).to(self.device)
                self.rl_algo.temp_next_ob = next_ob
                self.rl_algo.do_rollout(start_obs=next_ob)
            else:
                self.rl_algo.do_rollout()

            self.n_steps += self.rl_params.num_steps

            for _ in range(self.irl_params.epoch_disc):
                self.learning_steps_disc += 1

                # Samples from current policy's trajectories.
                obs, next_obs, _, logprobs, _, _, dones, _ = \
                    self.rl_algo.buffer.sample(int(self.irl_params.batch_size))
                obs = obs.reshape((-1,) + self.env.single_observation_space.shape)
                next_obs = next_obs.reshape((-1,) + self.env.single_observation_space.shape)

                # we modify in this part for robotic navigation reaching terminal state
                # because terminal state on absorbing state mode while training using rl
                # so terminal state when state[0:10] < thresh_collision and state[10]<thresh_goal
                # done is applied after 
                # print(f"{states.shape=}")
                # laser_min = th.min(obs[:,0:9], dim=1).values
                dist_to_goal = obs[:,10]

                if(self.env == self.env.unwrapped):
                    # dones = th.logical_or(laser_min<self.env.envs[0].agent_settings.collision_dist,
                    #                       dist_to_goal<self.env.envs[0].agent_settings.goal_dist)
                    dones = dist_to_goal<self.env.envs[0].agent_settings.goal_dist
                else:
                    # dones = th.logical_or(laser_min<self.env.envs[0].unwrapped.agent_settings.collision_dist,
                    #                       dist_to_goal<self.env.envs[0].unwrapped.agent_settings.goal_dist)
                    dones = dist_to_goal<self.env.envs[0].unwrapped.agent_settings.goal_dist
                dones = dones.unsqueeze(1).int()

                # Samples from expert's demonstrations.
                obs_exp, actions_exp, _, dones_exp, next_obs_exp = \
                    self.buffer_exp.sample(int(self.irl_params.batch_size))

                # dones_exp = dones_exp.unsqueeze(1).int()
                dones_exp = dones_exp.int()
                
                # Calculate log probabilities of expert actions.
                with th.no_grad():
                    _, logprobs_exp, _, _ = self.rl_algo.agent.get_action_and_value(obs_exp, actions_exp)

                logprobs_exp = logprobs_exp.unsqueeze(1)

                # Update discriminator.
                self.update_disc(
                    obs, dones, logprobs, next_obs, 
                    obs_exp, dones_exp, logprobs_exp, next_obs_exp
                )

            # We don't use reward signals here,
            obs, next_obs, acts, logprobs, values, terminations, dones, rewards  = self.rl_algo.buffer.get()

            new_obs = obs.reshape((-1,) + self.env.single_observation_space.shape)
            new_next_obs = next_obs.reshape((-1,) + self.env.single_observation_space.shape)
            # new_logprobs = logprobs.reshape(-1)
            new_logprobs = logprobs

            # we modify in this part for robotic navigation reaching terminal state
            # because terminal state on absorbing state mode while training using rl
            # so terminal state when state[0:10] < thresh_collision and state[10]<thresh_goal
            # done is applied after 
            # print(f"{states.shape=}")
            # laser_min = th.min(new_obs[:,0:9], dim=1).values
            dist_to_goal = new_obs[:,10]
            if(self.env == self.env.unwrapped):
                # new_dones = th.logical_or(laser_min<self.env.envs[0].agent_settings.collision_dist,
                #                       dist_to_goal<self.env.envs[0].agent_settings.goal_dist)
                new_dones = dist_to_goal<self.env.envs[0].agent_settings.goal_dist
            else:
                # new_dones = th.logical_or(laser_min<self.env.envs[0].unwrapped.agent_settings.collision_dist,
                #                       dist_to_goal<self.env.envs[0].unwrapped.agent_settings.goal_dist)
                new_dones = dist_to_goal<self.env.envs[0].unwrapped.agent_settings.goal_dist
            new_dones = new_dones.unsqueeze(1).int()

            # Calculate rewards.
            rewards = self.disc.calculate_reward(new_obs, 
                                                 new_dones, 
                                                 new_logprobs, 
                                                 new_next_obs,
                                                 reward_type="airl_shaped",
                                                 no_grad=True)
            print(f"estimate {rewards.shape=}")
            self.rl_algo.buffer.set(obs, 
                                    next_obs, 
                                    acts, 
                                    logprobs, 
                                    values, 
                                    terminations, 
                                    dones,
                                    rewards)

            # Update PPO using new estimated rewards.
            batch_data = self.rl_algo.calculate_gae()
            self.rl_algo.update_ppo(batch_data)

            #save discriminator

            if(self.n_steps % self.irl_params.save_every_nstep == 0):
                self.save_models(save_dir="./models/airl",
                                 iter=iter)
                iter += 1

    def update_disc(self, 
                    obs: th.Tensor,
                    dones: th.Tensor, 
                    logprobs: th.Tensor, 
                    next_obs: th.Tensor,
                    obs_exp: th.Tensor, 
                    dones_exp: th.Tensor, 
                    logprobs_exp: th.Tensor, 
                    next_obs_exp: th.Tensor):

        # # Output of discriminator is (-inf, inf), not [0, 1].
        # logits_pi = self.disc(obs, obs, logprobs, next_obs)
        # logits_exp = self.disc(obs_exp, dones_exp, logprobs_exp, next_obs_exp)

        # # Discriminator is to maximize E_{\pi} [log(1 - D)] + E_{exp} [log(D)].
        # loss_pi = -F.logsigmoid(-logits_pi).mean()
        # loss_exp = -F.logsigmoid(logits_exp).mean()
        # loss_disc = loss_pi + loss_exp

        # try to use imitation library loss using binary_cross_netropy_with_logits
        obs = th.concatenate([obs, obs_exp])

        dones = th.concatenate([dones, dones_exp])
        log_probs = th.concatenate([logprobs, logprobs_exp])
        next_obs = th.concatenate([next_obs, next_obs_exp])
        logits = th.concatenate([th.zeros(self.irl_params.batch_size, dtype=th.int),
                                 th.ones(self.irl_params.batch_size, dtype=th.int),
                                ]).to(self.device)

        disc_output = self.disc(obs,
                                dones, 
                                log_probs, 
                                next_obs)

        loss_disc = F.binary_cross_entropy_with_logits(disc_output.flatten(), logits.float())

        # # adding new penalty here
        # lambda_penalty = 0.5
        # # print(f"{next_obs.shape=}")
        min_laser_dist = th.min(next_obs[:,0:9], dim=1).values
        dist_to_goal = next_obs[:,10]

        if(self.env == self.env.unwrapped):
            collision_flags = min_laser_dist<th.Tensor([self.env.envs[0].agent_settings.collision_dist]).type(th.float).to(min_laser_dist.device)
            target = dist_to_goal<self.env.envs[0].agent_settings.goal_dist
        else:
            collision_flags = min_laser_dist<th.Tensor([self.env.envs[0].unwrapped.agent_settings.collision_dist]).type(th.float).to(min_laser_dist.device)
            target = dist_to_goal<self.env.envs[0].unwrapped.agent_settings.goal_dist

        goal_true_rew = []
        for i in range(len(collision_flags)):
            if(collision_flags[i].item==1):
                goal_true_rew.append(1) # output class for collision
            elif target[i].item==1:
                goal_true_rew.append(2) # output class for reaching goal target
            else:
                goal_true_rew.append(0) # output class for while reaching target
        goal_true_rew = th.Tensor(goal_true_rew).type(th.int8).to(min_laser_dist.device).unsqueeze(1)
        # with th.no_grad():
        #     print(f"{goal_true_rew.shape=}")
        #     print(f"{self.disc.goal_net(next_obs).argmax(dim=1).shape=}")
        goal_pred = self.disc.goal_f(next_obs,
                                     no_grad=False).argmax(dim=1).type(th.int8).unsqueeze(1)
            
        goal_pred.requires_grad=True
        assert goal_pred.requires_grad == True
        loss_goal_net = F.cross_entropy(goal_pred, goal_true_rew)
        # collision_penalty = collision_flags * th.clamp(self.disc.h_net(next_obs)-self.disc.h_net(obs), min=0)
        # regularization = lambda_penalty * collision_penalty.mean()

        # total_loss = loss_disc + regularization

        # update gradient discriminator
        self.optim_disc.zero_grad()
        loss_disc.backward()
        # total_loss.backward()
        self.optim_disc.step()

        # update gradient goal_net
        self.optim_goal_net.zero_grad()
        loss_goal_net.backward()
        self.optim_goal_net.step()


        if self.learning_steps_disc % self.irl_params.epoch_disc == 0:
            self.writer.add_scalar(
                'loss/disc', loss_disc.item(), self.n_steps)

            # # Discriminator's accuracies.
            # with th.no_grad():
            #     acc_pi = (logits_pi < 0).float().mean().item()
            #     acc_exp = (logits_exp > 0).float().mean().item()
            # self.writer.add_scalar('stats/acc_pi', acc_pi, self.n_steps)
            # self.writer.add_scalar('stats/acc_exp', acc_exp, self.n_steps)

    def save_models(self, 
                    save_dir: str,
                    iter: int):
        """
        Save the model

        Parameters
        ----------
        save_dir: str
            path to save
        """
        if not os.path.isdir(save_dir):
            os.mkdir(save_dir)
        # saving discriminator
        th.save(self.disc.state_dict(), f'{save_dir}/disc_{iter}.pkl')

        # saving actor policy
        self.rl_algo.agent.save_model(f'{save_dir}',
                                      "actor",
                                      iter)
