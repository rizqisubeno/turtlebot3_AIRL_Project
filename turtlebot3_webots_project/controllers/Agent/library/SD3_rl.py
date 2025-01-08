import os
import copy
import numpy as np
import random

import gymnasium as gym
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim.adam import Adam
from torch.utils.tensorboard.writer import SummaryWriter

import hydra
from hydra.utils import instantiate
from omegaconf import OmegaConf
from types import SimpleNamespace
from typing import Optional, Type, Union

from .PPO_rl import Logger
from .normalize import NormalizeObservation

class ReplayBuffer(object):
    def __init__(self, state_dim, action_dim, device, max_size=int(1e6)):
        self.max_size = max_size
        self.ptr = 0
        self.size = 0

        self.state = np.zeros((max_size, state_dim))
        self.action = np.zeros((max_size, action_dim))
        self.next_state = np.zeros((max_size, state_dim))
        self.reward = np.zeros((max_size, 1))
        self.done = np.zeros((max_size, 1))

        self.device = device

    def add(self, state, action, next_state, reward, done):
        self.state[self.ptr] = state
        self.action[self.ptr] = action
        self.next_state[self.ptr] = next_state
        self.reward[self.ptr] = reward
        self.done[self.ptr] = 1. - done

        self.ptr = (self.ptr + 1) % self.max_size
        self.size = min(self.size + 1, self.max_size)


    def sample(self, batch_size):
        ind = np.random.randint(0, self.size, size=batch_size)

        return (
            torch.FloatTensor(self.state[ind]).to(self.device),
            torch.FloatTensor(self.action[ind]).to(self.device),
            torch.FloatTensor(self.next_state[ind]).to(self.device),
            torch.FloatTensor(self.reward[ind]).to(self.device),
            torch.FloatTensor(self.done[ind]).to(self.device)
        )

class Actor(nn.Module):
    def __init__(self, 
                state_dim, 
                action_dim, 
                action_space, 
                hidden_sizes = [400, 300],
                activation_fn: nn.Module = nn.ReLU()):
        super(Actor, self).__init__()

        actor = nn.ModuleList()

        last_dim = state_dim
        for i in range(len(hidden_sizes)):
            actor.append(nn.Linear(last_dim, hidden_sizes[i]))
            actor.append(activation_fn)
            last_dim = hidden_sizes[i]
        actor.append(nn.Linear(last_dim, action_dim))

        self.actor_nn = nn.Sequential(*actor)
        # action rescaling
        self.register_buffer(
            "action_scale", torch.tensor((action_space.high - action_space.low) / 2.0, dtype=torch.float32)
        )
        self.register_buffer(
            "action_bias", torch.tensor((action_space.high + action_space.low) / 2.0, dtype=torch.float32)
        )

    def forward(self, state):
        act = self.actor_nn(state)
        return act * self.action_scale + self.action_bias

class Critic(nn.Module):
    def __init__(self, 
                 state_dim, 
                 action_dim, 
                 hidden_sizes=[400, 300],
                 activation_fn: nn.Module = nn.ReLU()):
        super(Critic, self).__init__()

        critic = nn.ModuleList()

        last_dim = state_dim + action_dim
        for i in range(len(hidden_sizes)):
            critic.append(nn.Linear(last_dim, hidden_sizes[i]))
            critic.append(activation_fn)
            last_dim = hidden_sizes[i]
        critic.append(nn.Linear(last_dim, 1))

        self.critic = nn.Sequential(*critic)

    def forward(self, state, action):
        if len(state.shape) == 3:
            sa = torch.cat([state, action], 2)  # when adding noise samples
        else:
            sa = torch.cat([state, action], 1)  # when without noise samples

        return self.critic(sa)


class SD3(object):
    def __init__(
        self,
        env,
        config_path: str | None,
        config_name: str | None,
        # state_dim,
        # action_dim,
        # max_action,
        # device,
        # discount=0.99,
        # tau=0.005,
        # policy_noise=0.2,
        # noise_clip=0.5,
        # actor_lr=1e-3,
        # critic_lr=1e-3,
        # hidden_sizes=[400, 300],
        # beta=0.001,
        # num_noise_samples=50,
        # with_importance_sampling=0,
        ):

        self.logger = Logger()
        
        assert isinstance(config_path, str)
        assert isinstance(config_name, str)
        params_cfg = self.hydra_params_read(config_path, config_name)
        self.params = self.read_param_cfg(params_cfg=params_cfg)

        if (self.params.use_rsnorm):
            self.logger.print("info", "Using Running Statistic Normalization")
            self.env = NormalizeObservation(env,
                                            epsilon=1e-8,
                                            is_training=True)
        else:
            self.env = env

        self.num_envs = 1

        # TRY NOT TO MODIFY: seeding
        random.seed(self.params.seed)
        np.random.seed(self.params.seed)
        torch.manual_seed(self.params.seed)
        torch.backends.cudnn.deterministic = self.params.torch_deterministic
        self.env.single_action_space.seed(self.params.seed)

        self.device = torch.device(
            "cuda" if torch.cuda.is_available() and self.params.cuda_en else "cpu")

        assert isinstance(env.single_action_space,
                          gym.spaces.Box), "only continuous action space is supported"

        self.actor1 = Actor(state_dim=np.array(self.env.single_observation_space.shape).prod(), 
                            action_dim=np.prod(self.env.single_action_space.shape), 
                            action_space=self.env.single_action_space, 
                            hidden_sizes = self.params.actor_hidden_size[0],
                            activation_fn=instantiate(self.params.actor_activation_fn)).to(self.device)
        self.actor1_target = copy.deepcopy(self.actor1)
        self.actor1_optimizer = instantiate(self.params.actor_optimizer, params=self.actor1.parameters())

        self.actor2 = Actor(state_dim=np.array(self.env.single_observation_space.shape).prod(), 
                            action_dim=np.prod(self.env.single_action_space.shape), 
                            action_space=self.env.single_action_space, 
                            hidden_sizes = self.params.actor_hidden_size[0],
                            activation_fn=instantiate(self.params.actor_activation_fn)).to(self.device)
        self.actor2_target = copy.deepcopy(self.actor2)
        self.actor2_optimizer = instantiate(self.params.actor_optimizer, params=self.actor2.parameters())

        self.critic1 = Critic(state_dim=np.array(self.env.single_observation_space.shape).prod(), 
                              action_dim=np.prod(self.env.single_action_space.shape), 
                              hidden_sizes = self.params.actor_hidden_size[0],
                              activation_fn=instantiate(self.params.critic_activation_fn)).to(self.device)
        self.critic1_target = copy.deepcopy(self.critic1)
        self.critic1_optimizer = instantiate(self.params.critic_optimizer, params=self.critic1.parameters())

        self.critic2 = Critic(state_dim=np.array(self.env.single_observation_space.shape).prod(), 
                              action_dim=np.prod(self.env.single_action_space.shape), 
                              hidden_sizes = self.params.actor_hidden_size[0],
                              activation_fn=instantiate(self.params.actor_activation_fn)).to(self.device)
        self.critic2_target = copy.deepcopy(self.critic2)
        self.critic2_optimizer = instantiate(self.params.critic_optimizer, params=self.critic2.parameters())

        self.replay_buffer = ReplayBuffer(state_dim=np.array(self.env.single_observation_space.shape).prod(),
                                          action_dim=np.prod(self.env.single_action_space.shape),
                                          device=self.device,
                                          max_size=int(self.params.replay_buffer_size))

        try:
            _ = self.env.unwrapped.writer
            self.logger.print("info", "SummaryWriter Found on Env")
        except AttributeError:
            self.logger.print("info", "Create SummaryWriter inside Env")
            self.env.envs[0].writer = SummaryWriter(
                f"runs/{self.params.exp_name}")

        self.env.envs[0].writer.add_text(
            "rl_hyperparameters",
            "|param|value|\n|-|-|\n%s" % (
                "\n".join([f"|{key}|{value}|" for key, value in vars(self.params).items()])),
        )
        self.save_config = "reset" if self.params.save_every_reset else "step"

        if ("reset" in self.save_config):
            self.reset_counter = 0
            self.save_iter = 0
        elif ("step" in self.save_config):
            self.num_timestep = 0
            self.save_iter = 0
    
    def hydra_params_read(self,
                          config_path: str,
                          config_name: str):
    
        config_path = "."+ config_path if config_path[:2] == "./" else \
                      "../"+config_path if config_path[:1] != "." else \
                      config_path
        # initialize hydra and load the configuration
        with hydra.initialize(config_path=config_path,
                              version_base="1.2"):
            cfg = hydra.compose(config_name=config_name)
        cfg = OmegaConf.to_object(cfg)

        # re-formatted dict
        new_cfg = {}
        for item in cfg.keys():

            assert isinstance(cfg, dict)

            if (isinstance(cfg[item], dict)):
                for sub_item in cfg[item].keys():
                    new_cfg[sub_item] = cfg[item][sub_item]
            else:
                new_cfg[item] = cfg[item]

        return new_cfg

    def read_param_cfg(self, params_cfg, verbose: Optional[bool] = True):
        cfg_namespace = SimpleNamespace(**params_cfg)
        if (verbose):
            for key, val in params_cfg.items():
                self.logger.print("info", f"{key} :\t{val}")
        return cfg_namespace

    def select_action(self, state):
        state = torch.FloatTensor(state.reshape(1, -1)).to(self.device)

        action1 = self.actor1(state)
        action2 = self.actor2(state)

        q1 = self.critic1(state, action1)
        q2 = self.critic2(state, action2)

        action = action1 if q1 >= q2 else action2

        return action.cpu().data.numpy().flatten()


    def train(self, 
              total_timesteps: int = int(1e6),):
        
        obs, done = self.env.reset(seed=self.params.seed)
        for global_step in range(total_timesteps):
            
            if global_step < self.params.learning_starts:
                actions = np.array([self.env.single_action_space.sample()
                                   for _ in range(self.num_envs)])
            else:
                actions = (self.select_action(np.array(obs)) + 
                           np.random.normal(loc=0, 
                                            scale=0.1, 
                                            size=np.prod(self.env.single_action_space.shape)
                                            )).clip(self.env.single_action_space.low, 
                                                    self.env.single_action_space.high)
                actions = actions.reshape(1,-1)
            # TRY NOT TO MODIFY: execute the game and log data.
            next_obs, rewards, terminations, truncations, infos = self.env.step(actions)

            # TRY NOT TO MODIFY: record rewards for plotting purposes
            if "final_info" in infos:
                for info in infos["final_info"]:
                    print(f"global_step={global_step}, episodic_return={info['episode']['r']}")
                    self.env.envs[0].writer.add_scalar(
                        "charts/episodic_return", info["episode"]["r"], global_step)
                    self.env.envs[0].writer.add_scalar(
                        "charts/episodic_length", info["episode"]["l"], global_step)
                    if ("reset" in self.save_config):
                        self.reset_counter += 1
                    break

            # saving model every step
            if ("reset" in self.save_config):
                if (self.reset_counter % self.params.save_step == 0 and self.reset_counter > 0):
                    self.reset_counter = 0
                    self.save_iter += 1
                    path = str(os.path.join(self.params.save_path, self.params.exp_name)) + \
                           str(f"_{int(self.save_iter)}")
                    print(f"{path=}")
                    self.save(path)
                    if (self.params.use_rsnorm):
                        path = self.params.save_path
                        path = path+"/" if path[-1] != "/" else path
                        last_path = str(os.path.join(
                            path, self.params.exp_name))+f"_norm_{int(self.save_iter)}"
                        np.savez(last_path,
                                 mean=self.env.obs_rms.mean,
                                 var=self.env.obs_rms.var,
                                 count=self.env.obs_rms.count)

            elif ("step" in self.save_config):
                self.num_timestep += 1
                if (self.num_timestep % self.params.save_step == 0 and self.num_timestep >= 0):
                    self.num_timestep = 0
                    self.save_iter += 1
                    path = str(os.path.join(self.params.save_path, self.params.exp_name)) + \
                           str(f"_{int(self.save_iter)}")
                    print(f"{path=}")
                    self.save(path)
                    if (self.params.use_rsnorm):
                        path = self.params.save_path
                        path = path+"/" if path[-1] != "/" else path
                        last_path = str(os.path.join(
                            path, self.params.exp_name))+f"_norm_{int(self.save_iter)}"
                        self.env.save
                        np.savez(last_path,
                                 mean=self.env.obs_rms.mean,
                                 var=self.env.obs_rms.var,
                                 count=self.env.obs_rms.count)
                        
            # TRY NOT TO MODIFY: save data to reply buffer; handle `final_observation`
            real_next_obs = next_obs.copy()
            assert isinstance(truncations, np.ndarray)
            assert isinstance(rewards, np.ndarray)
            assert isinstance(infos, dict)
            for idx, trunc in enumerate(truncations):
                if trunc:
                    real_next_obs[idx] = infos["final_observation"][idx]
            self.replay_buffer.add(obs, actions, real_next_obs,
                            rewards, np.array(terminations))

            # TRY NOT TO MODIFY: CRUCIAL step easy to overlook
            obs = next_obs

           # ALGO LOGIC: training.
            if global_step > self.params.learning_starts:
                if global_step % self.params.policy_frequency == 0:  # TD 3 Delayed update support
                    self.train_one_q_and_pi(self.replay_buffer, True)
                    self.train_one_q_and_pi(self.replay_buffer, False)


    def softmax_operator(self, q_vals, noise_pdf=None):
        max_q_vals = torch.max(q_vals, 1, keepdim=True).values
        norm_q_vals = q_vals - max_q_vals
        e_beta_normQ = torch.exp(self.params.beta * norm_q_vals)
        Q_mult_e = q_vals * e_beta_normQ

        numerators = Q_mult_e
        denominators = e_beta_normQ

        if self.params.with_importance_sampling:
            numerators /= noise_pdf
            denominators /= noise_pdf

        sum_numerators = torch.sum(numerators, 1)
        sum_denominators = torch.sum(denominators, 1)

        softmax_q_vals = sum_numerators / sum_denominators

        softmax_q_vals = torch.unsqueeze(softmax_q_vals, 1)
        return softmax_q_vals


    def calc_pdf(self, samples, mu=0):
        pdfs = 1/(self.policy_noise * np.sqrt(2 * np.pi)) * torch.exp( - (samples - mu)**2 / (2 * self.policy_noise**2) )
        pdf = torch.prod(pdfs, dim=2)
        return pdf


    def train_one_q_and_pi(self, replay_buffer, update_q1):
        state, action, next_state, reward, done = replay_buffer.sample(self.params.batch_size)

        with torch.no_grad():
            if update_q1:
                next_action = self.actor1_target(next_state)
            else:
                next_action = self.actor2_target(next_state)

            noise = torch.randn(
                (action.shape[0], self.params.num_noise_samples, action.shape[1]), 
                dtype=action.dtype, layout=action.layout, device=action.device
            )
            noise = noise * self.params.policy_noise
            
            noise_pdf = self.calc_pdf(noise) if self.params.with_importance_sampling else None
            
            noise = noise.clamp(-self.params.noise_clip, self.params.noise_clip)

            next_action = torch.unsqueeze(next_action, 1)

            action_min = torch.tensor(self.env.single_action_space.low, dtype=next_action.dtype, device=next_action.device).view(1, 1, -1)
            action_max = torch.tensor(self.env.single_action_space.high, dtype=next_action.dtype, device=next_action.device).view(1, 1, -1)
            next_action = (next_action + noise).clamp(action_min, action_max)

            next_state = torch.unsqueeze(next_state, 1)
            next_state = next_state.repeat((1, self.params.num_noise_samples, 1))

            next_Q1 = self.critic1_target(next_state, next_action)
            next_Q2 = self.critic2_target(next_state, next_action)

            next_Q = torch.min(next_Q1, next_Q2)
            next_Q = torch.squeeze(next_Q, 2)

            softmax_next_Q = self.softmax_operator(next_Q, noise_pdf)
            next_Q = softmax_next_Q

            target_Q = reward + (1-done) * self.params.gamma * next_Q

        if update_q1:
            current_Q = self.critic1(state, action)

            critic1_loss = F.mse_loss(current_Q, target_Q)

            self.critic1_optimizer.zero_grad()
            critic1_loss.backward()
            self.critic1_optimizer.step()

            actor1_loss = -self.critic1(state, self.actor1(state)).mean()
            
            self.actor1_optimizer.zero_grad()
            actor1_loss.backward()
            self.actor1_optimizer.step()

            for param, target_param in zip(self.critic1.parameters(), self.critic1_target.parameters()):
                target_param.data.copy_(self.params.tau * param.data + (1 - self.params.tau) * target_param.data)

            for param, target_param in zip(self.actor1.parameters(), self.actor1_target.parameters()):
                target_param.data.copy_(self.params.tau * param.data + (1 - self.params.tau) * target_param.data)

        else:
            current_Q = self.critic2(state, action)

            critic2_loss = F.mse_loss(current_Q, target_Q)

            self.critic2_optimizer.zero_grad()
            critic2_loss.backward()
            self.critic2_optimizer.step()

            actor2_loss = -self.critic2(state, self.actor2(state)).mean()
            
            self.actor2_optimizer.zero_grad()
            actor2_loss.backward()
            self.actor2_optimizer.step()

            for param, target_param in zip(self.critic2.parameters(), self.critic2_target.parameters()):
                target_param.data.copy_(self.params.tau * param.data + (1 - self.params.tau) * target_param.data)

            for param, target_param in zip(self.actor2.parameters(), self.actor2_target.parameters()):
                target_param.data.copy_(self.params.tau * param.data + (1 - self.params.tau) * target_param.data)


    def save(self, filename):
        torch.save(self.critic1.state_dict(), filename + "_critic1")
        torch.save(self.critic1_optimizer.state_dict(), filename + "_critic1_optimizer")
        torch.save(self.actor1.state_dict(), filename + "_actor1")
        torch.save(self.actor1_optimizer.state_dict(), filename + "_actor1_optimizer")

        torch.save(self.critic2.state_dict(), filename + "_critic2")
        torch.save(self.critic2_optimizer.state_dict(), filename + "_critic2_optimizer")
        torch.save(self.actor2.state_dict(), filename + "_actor2")
        torch.save(self.actor2_optimizer.state_dict(), filename + "_actor2_optimizer")

    def load(self, filename):
        self.critic1.load_state_dict(torch.load(filename + "_critic1"))
        self.critic1_optimizer.load_state_dict(torch.load(filename + "_critic1_optimizer"))
        self.actor1.load_state_dict(torch.load(filename + "_actor1"))
        self.actor1_optimizer.load_state_dict(torch.load(filename + "_actor1_optimizer"))

        self.critic2.load_state_dict(torch.load(filename + "_critic2"))
        self.critic2_optimizer.load_state_dict(torch.load(filename + "_critic2_optimizer"))
        self.actor2.load_state_dict(torch.load(filename + "_actor2"))
        self.actor2_optimizer.load_state_dict(torch.load(filename + "_actor2_optimizer"))