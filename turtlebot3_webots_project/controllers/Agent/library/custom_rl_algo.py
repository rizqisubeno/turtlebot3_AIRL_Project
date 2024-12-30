# ﷽
# try using configuration with hydra-core
import os
import random
import time
import warnings
from copy import deepcopy

import gymnasium as gym

from .ICM import ICM

# from library.tb3_agent import logger
import logging
from types import SimpleNamespace
from typing import Optional, Type, Union

import hydra
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from colorlog import ColoredFormatter
from hydra.utils import instantiate
from .normalize import NormalizeObservation
from omegaconf import OmegaConf
from stable_baselines3.common.buffers import ReplayBuffer

# import tyro
from torch.distributions.normal import Normal
from torch.optim.adam import Adam
from torch.utils.tensorboard.writer import SummaryWriter

from .clipped_gaussian import ClippedGaussian
from .Imitation.buffer import RolloutBuffer
from .Simba_Network import SACEncoder, PPOEncoder

from scenario_list import *

LOG_LEVEL = logging.DEBUG
LOGFORMAT = "  %(log_color)s%(levelname)-8s%(reset)s | %(log_color)s%(message)s%(reset)s"
logging.root.setLevel(LOG_LEVEL)
formatter = ColoredFormatter(LOGFORMAT)
stream = logging.StreamHandler()
stream.setLevel(LOG_LEVEL)
stream.setFormatter(formatter)

# Suppress UserWarnings specifically from gymnasium
warnings.filterwarnings("ignore", category=UserWarning, module="gymnasium")

distribution_classes = {
    'Normal': Normal,
    'ClippedGaussian': ClippedGaussian,
}

class Logger():
    def __init__(self):
        self.log = logging.getLogger('pythonConfig')
        self.log.setLevel(LOG_LEVEL)
        self.log.addHandler(stream)

    def print(self,
              type: str = "info",
              msg: str = ""):

        if "debug" in type:
            self.log.debug(msg)
        elif "info" in type:
            self.log.info(msg)
        elif "warn" in type:
            self.log.warning(msg)
        elif "err" in type:
            self.log.error(msg)
        elif "crit" in type:
            self.log.critical(msg)
        else:
            self.log.info(msg)


def layer_init(layer, std=np.sqrt(2), bias_const=0.0):
    torch.nn.init.orthogonal_(layer.weight, std)
    torch.nn.init.constant_(layer.bias, bias_const)
    return layer


class PPO_Agent_NN(nn.Module):
    def __init__(self,
                 envs,
                 logger: Logger,
                 distribution: str = "Normal",
                 rpo_alpha: Union[float, bool] = 0.1,
                 activation: nn.Module = nn.Tanh(),
                 use_tanh_output: bool = False,
                 device: Optional[torch.device] = None):
        super().__init__()
        self.device = device
        # if bool usually set as False
        # if not boolean, automatically set rpo_mode true with value on rpo_alpha
        if (isinstance(rpo_alpha, bool)):
            self.rpo_mode = rpo_alpha
        else:
            self.rpo_mode = True
            self.rpo_alpha = rpo_alpha
        self.logger = logger
        self.envs = envs
        if distribution not in distribution_classes:
            raise ValueError(f"Unknown distribution: {distribution}")
        else:
            self.distribution = distribution_classes[distribution]
            self.logger.print("info", f"use distribution: {self.distribution}")
        self.low = torch.from_numpy(self.envs.action_space.low).to(self.device)
        self.high = torch.from_numpy(
            self.envs.action_space.high).to(self.device)

        critic = nn.ModuleList()
        # critic.append(PPOEncoder(block_type="residual",
        #                          input_dim=envs.single_observation_space.shape[0],
        #                          num_blocks=2,
        #                          hidden_dim=128,))
        critic.append(nn.Linear(np.array(envs.single_observation_space.shape).prod(), 64))
        # critic.append(nn.LayerNorm(64)) # we adding layer norm
        critic.append(activation)
        critic.append(layer_init(nn.Linear(64, 64)))
        critic.append(activation)
        critic.append(layer_init(nn.Linear(64, 64)))
        critic.append(activation)
        critic.append(layer_init(nn.Linear(64, 1), std=1.0))
        self.critic = nn.Sequential(*critic)

        actor_layer = nn.ModuleList()
        # actor_layer.append(PPOEncoder(block_type="residual",
        #                               input_dim=envs.single_observation_space.shape[0],
        #                               num_blocks=2,
        #                               hidden_dim=128,))
        actor_layer.append(nn.Linear(np.array(envs.single_observation_space.shape).prod(), 64))
        # actor_layer.append(nn.LayerNorm(64)) # we adding layer norm
        actor_layer.append(activation)
        actor_layer.append(layer_init(nn.Linear(64, 64)))
        actor_layer.append(activation)
        actor_layer.append(layer_init(nn.Linear(64, 64)))
        actor_layer.append(activation)
        actor_layer.append(layer_init(nn.Linear(64, np.prod(envs.single_action_space.shape)), std=0.01))
        actor_layer.append(activation) if use_tanh_output else None

        self.actor_mean = nn.Sequential(*actor_layer)
        # print(f"{self.actor_mean}")
        self.actor_logstd = nn.Parameter(torch.zeros(
            1, np.prod(envs.single_action_space.shape)))

    def get_value(self, x):
        return self.critic(x)

    def get_action(self,
                   x,
                   deterministic: bool = False):
        with torch.no_grad():
            action_mean = self.actor_mean(x)
            action_logstd = self.actor_logstd.expand_as(action_mean)
            action_std = torch.exp(action_logstd)
            if (deterministic):
                action = action_mean
            else:
                if self.distribution == Normal:
                    action = Normal(loc=action_mean,
                                    scale=action_std).sample()
                elif self.distribution == ClippedGaussian:
                    action = ClippedGaussian(mean=action_mean,
                                             var=action_std**2,
                                             low=self.low,
                                             high=self.high).sample()
        return action

    # RPO_Mode is Robust Policy Optimization Mode
    def get_action_and_value(self, x, action=None):
        action_mean = self.actor_mean(x)
        action_logstd = self.actor_logstd.expand_as(action_mean)
        action_std = torch.exp(action_logstd)
        if self.distribution == Normal:
            probs = Normal(loc=action_mean,
                           scale=action_std)
        elif self.distribution == ClippedGaussian:
            probs = ClippedGaussian(mean=action_mean,
                                    var=action_std**2,
                                    low=self.low,
                                    high=self.high)
        if action is None:
            action = probs.sample()
        elif self.rpo_mode:  # new to RPO
            # sample again to add stochasticity to the policy
            z = torch.FloatTensor(
                action_mean.shape).uniform_(-self.rpo_alpha, self.rpo_alpha).to(self.device)
            action_mean = action_mean + z
            if self.distribution == Normal:
                probs = Normal(loc=action_mean,
                               scale=action_std)
            elif self.distribution == ClippedGaussian:
                probs = ClippedGaussian(mean=action_mean,
                                        var=action_std**2,
                                        low=self.low,
                                        high=self.high)

        # print(f"{probs.log_prob(action).shape=}")
        # print(f"{probs.log_prob(action)=}")
        # print(f"{probs.log_prob(action).sum(1)=}")
        return action, probs.log_prob(action).sum(1), probs.entropy().sum(1), self.critic(x)

    def save_model(self, path: str, exp_name: str, iter: int):
        """Save model to a specified path."""
        path = path+"/" if path[-1] != "/" else path
        last_path = str(os.path.join(path, exp_name))+f"_{iter}.pth"
        if os.path.exists(path):
            pass
        else:
            os.makedirs(path, exist_ok=True)
        torch.save(self.state_dict(), last_path)
        self.logger.print("info", f"Model saved to {path}")

    def load_model(self, path: str, exp_name: str, iter: int):
        """Load model from a specified path."""
        path = str(os.path.join(path, exp_name))+f"_{iter}.pth"
        self.load_state_dict(torch.load(path,
                                        map_location=self.device,
                                        weights_only=True))
        self.logger.print("info", f"Model loaded from {path}")

# specific for single environment only
class PPO():
    def __init__(self,
                 env,
                 config_path: str | None,
                 config_name: str | None,
                 # for airl purpose so not need parsing config again
                 bypass_class_cfg: Optional[bool] = False,
                 rl_params: Optional[dict] | SimpleNamespace = None):

        self.logger = Logger()

        self.bypass_class_cfg = bypass_class_cfg
        if (not self.bypass_class_cfg):

            assert isinstance(config_path, str)
            assert isinstance(config_name, str)
            params_cfg = self.hydra_params_read(config_path, config_name)
            self.params = self.read_param_cfg(params_cfg=params_cfg)
        else:
            self.params = rl_params

        if (self.params.use_rsnorm):
            self.logger.print("info", "Using Running Statistic Normalization")
            self.env = NormalizeObservation(env,
                                            epsilon=1e-8,
                                            is_training=True)
        else:
            self.env = env

        self.num_envs = 1

        self.batch_size = self.num_envs * self.params.num_steps
        self.minibatch_size = self.batch_size // self.params.num_minibatches

        # TRY NOT TO MODIFY: seeding
        random.seed(self.params.seed)
        np.random.seed(self.params.seed)
        torch.manual_seed(self.params.seed)
        torch.backends.cudnn.deterministic = self.params.torch_deterministic

        self.device = torch.device(
            "cuda" if torch.cuda.is_available() and self.params.cuda_en else "cpu")

        assert isinstance(env.single_action_space,
                          gym.spaces.Box), "only continuous action space is supported"

        # using buffer samples
        self.buffer = RolloutBuffer(buffer_size=self.params.num_steps,
                                    num_envs=self.num_envs,
                                    state_shape=self.env.single_observation_space.shape,
                                    action_shape=self.env.single_action_space.shape,
                                    device=self.device)

        # Initialize PPO_Agent_NN Function
        self.agent = PPO_Agent_NN(env,
                                  logger=self.logger,
                                  distribution=self.params.distributions,
                                  rpo_alpha=self.params.rpo_alpha,
                                  activation=instantiate(
                                      self.params.activation_fn),
                                  use_tanh_output=self.params.use_tanh_output,
                                  device=self.device).to(self.device)
        # needed for kl_rollback
        self.target_agent = PPO_Agent_NN(env,
                                         logger=self.logger,
                                         distribution=self.params.distributions,
                                         rpo_alpha=self.params.rpo_alpha,
                                         activation=instantiate(
                                             self.params.activation_fn),
                                         use_tanh_output=self.params.use_tanh_output,
                                         device=self.device).to(self.device)

        # self.optimizer = Adam(self.agent.parameters(), lr=self.params.learning_rate, eps=1e-8)
        self.optimizer = instantiate(
            self.params.optimizer, params=self.agent.parameters())
        if (self.params.use_icm):
            self.ICM = ICM(observation_shape=self.env.single_observation_space.shape[0],
                           action_shape=self.env.single_action_space.shape[0],
                           activation_func=nn.ReLU,
                           layer_size=2,
                           hidden_size=256,
                           alpha=0.1,
                           beta=0.2,
                           ).to(self.device)
            self.icm_optimizer = Adam(self.ICM.parameters(), lr=3e-4, eps=1e-8)

        try:
            _ = self.env.envs[0].writer
            self.logger.print("info", "SummaryWriter Found on Env")
            self.env.envs[0].writer.add_text(
                "rl_hyperparameters",
                "|param|value|\n|-|-|\n%s" % (
                    "\n".join([f"|{key}|{value}|" for key, value in vars(self.params).items()])),
            )
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
        config_path = "."+config_path if config_path[:2] == "./" else \
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

    def train(self,
              total_timesteps: Optional[int] = int(1e6)):
        self.num_iterations = int(total_timesteps // self.batch_size)

        if hasattr(self.env.envs[0], 'curriculum'):
            self.logger.print("info", "Curriculum Learning Used on Env")
            self.weights_all = np.empty((self.num_iterations, max_robot_scenario))
            self.arm_probs_all = np.empty((self.num_iterations, max_robot_scenario))

        for iteration in range(1, self.num_iterations + 1):
            # Annealing the rate if instructed to do so.
            if self.params.anneal_lr:
                frac = 1.0 - (iteration - 1.0) / self.num_iterations
                lrnow = frac * self.params.learning_rate
                self.optimizer.param_groups[0]["lr"] = lrnow

            if (iteration == 1):
                # if teacherexp3 used use this
                if hasattr(self.env.envs[0], 'curriculum'):
                    # pull an arm
                    task = self.env.envs[0].curriculum.get_task()
                    self.logger.print("info", f"1Task change to : {task}")

                # TRY NOT TO MODIFY: start the game
                self.global_step = 0
                self.start_time = time.time()
                next_ob, _ = self.env.reset(seed=self.params.seed)
                next_ob = torch.Tensor(next_ob).to(self.device)
                self.temp_next_ob = next_ob
                self.do_rollout(iteration,
                                start_obs=next_ob)
            else:
                self.do_rollout(iteration,)

            if (self.params.use_icm):
                self.update_icm()

            batch_data = self.calculate_gae()
            self.update_ppo(batch_data)



        self.env.envs[0].writer.close()
        self.env.close()

    def do_rollout(self,
                   iteration,
                   start_obs: Optional[torch.Tensor] = None):
        next_ob = self.temp_next_ob
        for step in range(0, self.params.num_steps):
            self.global_step += self.num_envs

            if (start_obs is not None and step == 0):
                ob = start_obs
            else:
                ob = next_ob
            # ALGO LOGIC: action logic
            with torch.no_grad():
                action, logprob, _, value = self.agent.get_action_and_value(ob)

            # TRY NOT TO MODIFY: execute the game and log data.
            # Modifying with clip like stable-baselines3
            # please aware that environment clipping only not applying to batch for clipping action
            # https://github.com/DLR-RM/stable-baselines3/blob/master/stable_baselines3/common/on_policy_algorithm.py#L216
            action_clipped = np.clip(action.cpu().numpy(),
                                     a_min=self.env.action_space.low,
                                     a_max=self.env.action_space.high)
            next_ob, reward, next_termination, next_truncation, info = self.env.step(action_clipped)

            if (self.params.use_reward_norm):
                # print(f"reward before : {reward}")
                clipped = np.clip(a=np.array(reward), 
                                  a_min=-10, 
                                  a_max=10)
                reward = 2*((clipped-(-10))/(10-(-10)))-1
                # print(f"reward after : {reward}")

            assert not np.isnan(action_clipped).any()

            # Correct next observation (for vec gym)
            real_next_ob = next_ob.copy()
            for idx, trunc in enumerate(next_truncation):
                if trunc:
                    real_next_ob[idx] = info["final_observation"][idx]
            next_ob = torch.Tensor(next_ob).to(self.device)

            # Collect trajectory (append to buffer)
            self.buffer.append(ob=torch.Tensor(ob).to(self.device),
                               next_ob=torch.Tensor(
                                   real_next_ob).to(self.device),
                               action=torch.Tensor(action).to(self.device),
                               logprob=torch.Tensor(logprob).to(self.device),
                               value=torch.Tensor(
                                   value.flatten()).to(self.device),
                               next_termination=torch.Tensor(
                                   next_termination).to(self.device),
                               next_done=torch.Tensor(np.logical_or(
                                   next_termination, next_truncation)).to(self.device),
                               reward=torch.tensor(reward).to(self.device).view(-1))

            # if teacherexp3 used
            if hasattr(self.env.envs[0], 'curriculum'):
                if(self.env.envs[0].next_step_is_done):
                    print(f"idx env : {self.env.envs[0].idx}")
                    #change scenario by providing the task
                    last_task = self.env.envs[0].scenario_idx
                    task = self.env.envs[0].curriculum.get_task()
                    self.env.envs[0].next_step_scenario_idx = task
                    self.logger.print("info", f"2Task change to : {task}")
                    
                    #resseting variable next step is done
                    self.env.envs[0].next_step_is_done = False


            # writing returns episode and length episode
            if "final_info" in info:
                for info in info["final_info"]:
                    if info and "episode" in info:
                        self.logger.print("info", f"global_step={self.global_step}, episodic_return={info['episode']['r'].item():.3f}")
                        self.env.envs[0].writer.add_scalar(
                            "charts/episodic_return", info["episode"]["r"], self.global_step)
                        self.env.envs[0].writer.add_scalar(
                            "charts/episodic_length", info["episode"]["l"], self.global_step)
                        # if teacherexp3 used update this function
                        if hasattr(self.env.envs[0], 'curriculum'):
                            self.env.envs[0].curriculum.update(last_task, info["episode"]["r"])
                            self.weights_all[iteration] = self.env.envs[0].curriculum._log_weights
                            self.arm_probs_all[iteration] = self.env.envs[0].curriculum.task_probabilities
                            self.logger.print("info", f"weights: {self.weights_all[iteration]}")
                            self.logger.print("info", f"arm_probs: {self.arm_probs_all[iteration]}")

                        # self.temp_next_ob = next_ob
                        # print(f"{self.temp_next_ob=}")
                        if ("reset" in self.save_config):
                            self.reset_counter += 1


            # saving model every step (inside rollout)
            if ("reset" in self.save_config):
                if (self.reset_counter % self.params.save_step == 0 and self.reset_counter > 0):
                    self.reset_counter = 0
                    self.save_iter += 1
                    self.agent.save_model(self.params.save_path,
                                          self.params.exp_name,
                                          int(self.save_iter))
            elif ("step" in self.save_config):
                self.num_timestep += 1
                if (self.num_timestep % self.params.save_step == 0 and self.num_timestep >= 0):
                    self.num_timestep = 0
                    self.save_iter += 1
                    self.agent.save_model(self.params.save_path,
                                          self.params.exp_name,
                                          int(self.save_iter))

        self.temp_next_ob = next_ob

    def update_icm(self):
        rollout_data = self.buffer.get()
        obs = rollout_data[0]
        next_obs = rollout_data[1]
        actions = rollout_data[2]
        # logprobs=rollout_data[3]
        # values=rollout_data[4]
        # next_terminations=rollout_data[5]
        # next_dones=rollout_data[6]
        rewards = rollout_data[7]

        global_step_temp = self.global_step-self.params.num_steps
        for i in range(obs.shape[0]):
            intrinsic_reward, inv_loss, fw_loss = self.ICM.calc_loss(obs[i],
                                                                     actions[i],
                                                                     next_obs[i])
            self.env.envs[0].writer.add_scalar("charts/ICM/env_reward",
                                               rewards[i],
                                               global_step_temp)
            self.env.envs[0].writer.add_scalar("charts/ICM/int_reward",
                                               intrinsic_reward.cpu().detach().numpy(),
                                               global_step_temp)

            rewards[i] += intrinsic_reward.cpu().detach().numpy()
            self.buffer.rewards[i] = rewards[i]

            total_loss = inv_loss + fw_loss
            self.icm_optimizer.zero_grad()
            total_loss.backward()
            self.icm_optimizer.step()

            global_step_temp += 1

        self.env.envs[0].writer.add_scalar("charts/ICM/total_reward",
                                           np.sum(np.asarray(
                                               [val*(self.env.agent_settings.gamma**j) for j, val in enumerate(rewards)])),
                                           global_step_temp)

    def calculate_gae(self):
        rollout_data = self.buffer.get()
        obs = rollout_data[0]
        next_obs = rollout_data[1]
        actions = rollout_data[2]
        logprobs = rollout_data[3]
        values = rollout_data[4]
        next_terminations = rollout_data[5]
        next_dones = rollout_data[6]
        rewards = rollout_data[7]

        # bootstrap value if not done
        with torch.no_grad():
            next_values = torch.zeros_like(values[0]).to(self.device)
            advantages = torch.zeros_like(rewards).to(self.device)
            lastgaelam = 0
            for t in reversed(range(self.params.num_steps)):
                if t == self.params.num_steps - 1:
                    next_values = self.agent.get_value(next_obs[t]).flatten()
                else:
                    value_mask = next_dones[t].bool()
                    next_values[value_mask] = self.agent.get_value(
                        next_obs[t][value_mask]).flatten()
                    next_values[~value_mask] = values[t + 1][~value_mask]
                delta = rewards[t] + self.params.gamma * \
                    next_values * (1 - next_terminations[t]) - values[t]
                advantages[t] = lastgaelam = delta + self.params.gamma * \
                    self.params.gae_lambda * (1 - next_dones[t]) * lastgaelam
            returns = advantages + values

        return (obs.reshape((-1,) + self.env.single_observation_space.shape),
                actions.reshape((-1,) + self.env.single_action_space.shape),
                logprobs.reshape(-1),
                advantages.reshape(-1),
                returns.reshape(-1),
                values.reshape(-1),
                )

    def update_ppo(self,
                   batch_data):
        b_obs = batch_data[0]
        b_actions = batch_data[1]
        b_logprobs = batch_data[2]
        b_advantages = batch_data[3]
        b_returns = batch_data[4]
        b_values = batch_data[5]

        b_inds = np.arange(self.batch_size)
        clipfracs = []
        for _ in range(self.params.num_epoch):
            np.random.shuffle(b_inds)
            self.target_agent.load_state_dict(self.agent.state_dict())
            approx_kl_list = []
            for start in range(0, self.batch_size, self.minibatch_size):
                end = start + self.minibatch_size
                mb_inds = b_inds[start:end]
                # print(f"{b_actions[mb_inds]=}")
                _, newlogprob, entropy, newvalue = self.agent.get_action_and_value(
                    b_obs[mb_inds], b_actions[mb_inds])
                logratio = newlogprob - b_logprobs[mb_inds]
                ratio = logratio.exp()

                with torch.no_grad():
                    # calculate approx_kl http://joschu.net/blog/kl-approx.html
                    old_approx_kl = (-logratio).mean()
                    approx_kl = ((ratio - 1) - logratio).mean()
                    clipfracs += [((ratio - 1.0).abs() >
                                   self.params.clip_coef).float().mean().item()]

                approx_kl_list.append(approx_kl.item())

                mb_advantages = b_advantages[mb_inds]
                if self.params.norm_adv:
                    mb_advantages = (
                        mb_advantages - mb_advantages.mean()) / (mb_advantages.std() + 1e-8)

                # Policy loss
                pg_loss1 = -mb_advantages * ratio
                pg_loss2 = -mb_advantages * \
                    torch.clamp(ratio, 1 - self.params.clip_coef,
                                1 + self.params.clip_coef)
                pg_loss = torch.max(pg_loss1, pg_loss2).mean()

                # Value loss
                newvalue = newvalue.view(-1)
                if self.params.clip_vloss:
                    v_loss_unclipped = (newvalue - b_returns[mb_inds]) ** 2
                    v_clipped = b_values[mb_inds] + torch.clamp(
                        newvalue - b_values[mb_inds],
                        -self.params.clip_coef,
                        self.params.clip_coef,
                    )
                    v_loss_clipped = (v_clipped - b_returns[mb_inds]) ** 2
                    v_loss_max = torch.max(v_loss_unclipped, v_loss_clipped)
                    v_loss = 0.5 * v_loss_max.mean()
                else:
                    v_loss = 0.5 * \
                        ((newvalue - b_returns[mb_inds]) ** 2).mean()

                entropy_loss = entropy.mean()
                loss = pg_loss - self.params.ent_coef * \
                    entropy_loss + v_loss * self.params.vf_coef

                self.optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(
                    self.agent.parameters(), self.params.max_grad_norm)
                self.optimizer.step()

            # self.logger.print("hidden", "RPO", f"approx kl max:{np.max(approx_kl_list):.3f}, min:{np.min(approx_kl_list):.7f}")
            if self.params.target_kl is not None:
                if approx_kl > self.params.target_kl and self.params.kle_stop:
                    self.logger.print(
                        "err", "approx kl reach target, stopping update")
                    break

                # adding kle_rollback if needed
                if self.params.kle_rollback:
                    if (b_logprobs[mb_inds] - self.agent.get_action_and_value(b_obs[mb_inds], b_actions[mb_inds])[1]).mean() > self.params.target_kl:
                        self.agent.load_state_dict(
                            self.target_agent.state_dict())
                        self.logger.print(
                            "err", "approx kl reach target, rollback the model")
                        break
        # print(f"{self.agent.actor_logstd=}")

        y_pred, y_true = b_values.cpu().numpy(), b_returns.cpu().numpy()
        var_y = np.var(y_true)
        explained_var = np.nan if var_y == 0 else 1 - \
            np.var(y_true - y_pred) / var_y

        # TRY NOT TO MODIFY: record rewards for plotting purposes
        self.env.envs[0].writer.add_scalar(
            "charts/learning_rate", self.optimizer.param_groups[0]["lr"], self.global_step)
        self.env.envs[0].writer.add_scalar(
            "losses/value_loss", v_loss.item(), self.global_step)
        self.env.envs[0].writer.add_scalar(
            "losses/policy_loss", pg_loss.item(), self.global_step)
        self.env.envs[0].writer.add_scalar(
            "losses/entropy", entropy_loss.item(), self.global_step)
        self.env.envs[0].writer.add_scalar(
            "losses/old_approx_kl", old_approx_kl.item(), self.global_step)
        self.env.envs[0].writer.add_scalar(
            "losses/approx_kl", approx_kl.item(), self.global_step)
        self.env.envs[0].writer.add_scalar(
            "losses/clipfrac", np.mean(clipfracs), self.global_step)
        self.env.envs[0].writer.add_scalar(
            "losses/explained_variance", explained_var, self.global_step)
        # print("SPS:", int(self.global_step / (time.time() - start_time)))
        self.env.envs[0].writer.add_scalar(
            "charts/SPS", int(self.global_step / (time.time() - self.start_time)), self.global_step)
        # adding logging for log std
        for i in range(self.agent.actor_logstd.shape[1]):
            self.env.envs[0].writer.add_scalar(
                f"charts/log_std_{i}", self.agent.actor_logstd[0][i].item(), self.global_step)

    def eval_once(self, iter):
        # load model based on last training
        self.agent.load_model(path=self.params.save_path,
                              exp_name=self.params.exp_name,
                              iter=iter)
        obs, _ = self.env.reset(seed=self.params.seed)

        isExit = False
        global_step = 0
        while (not isExit):
            predict_action = self.agent.get_action(
                torch.from_numpy(obs).to(self.device), deterministic=True)
            next_obs, rew, term, trunc, info = self.env.step(
                predict_action.cpu().numpy())

            obs = next_obs
            global_step += 1

            if "final_info" in info:
                for info in info["final_info"]:
                    if info and "episode" in info:
                        self.logger.print("info", f"global_step={global_step}, episodic_return={info['episode']['r'].item():.3f}")
                        self.env.envs[0].writer.add_scalar(
                            "charts/eval/episodic_return", info["episode"]["r"], global_step)
                        self.env.envs[0].writer.add_scalar(
                            "charts/eval/episodic_length", info["episode"]["l"], global_step)
                        isExit = True

        self.env.close()

# get from https://github.com/DLR-RM/stable-baselines3/blob/master/stable_baselines3/common/distributions.py
def sum_independent_dims(tensor: torch.Tensor) -> torch.Tensor:
    """
    Continuous actions are usually considered to be independent,
    so we can sum components of the ``log_prob`` or the entropy.

    :param tensor: shape: (n_batch, n_actions) or (n_batch,)
    :return: shape: (n_batch,) for (n_batch, n_actions) input, scalar for (n_batch,) input
    """
    if len(tensor.shape) > 1:
        tensor = tensor.sum(dim=1)
    else:
        tensor = tensor.sum()
    return tensor

# get from https://github.com/DLR-RM/stable-baselines3/blob/master/stable_baselines3/common/distributions.py
class TanhBijector:
    """
    Bijective transformation of a probability distribution
    using a squash function (tanh).

    :param epsilon: small value to avoid NaN due to numerical imprecision
    """

    def __init__(self,
                 epsilon: float = 1e-6):
        super().__init__()
        self.epsilon = epsilon

    @staticmethod
    def forward(x: torch.Tensor) -> torch.Tensor:
        return torch.tanh(x)

    @staticmethod
    def atanh(x: torch.Tensor) -> torch.Tensor:
        """
        Inverse of Tanh

        Taken from Pyro: https://github.com/pyro-ppl/pyro
        0.5 * torch.log((1 + x ) / (1 - x))
        """
        return 0.5 * (x.log1p() - (-x).log1p())

    @staticmethod
    def inverse(y: torch.Tensor) -> torch.Tensor:
        """
        Inverse tanh.

        :param y:
        :return:
        """
        eps = torch.finfo(y.dtype).eps
        # Clip the action to avoid NaN
        return TanhBijector.atanh(y.clamp(min=-1.0 + eps, max=1.0 - eps))

    def log_prob_correction(self, x: torch.Tensor) -> torch.Tensor:
        # Squash correction (from original SAC implementation)
        return torch.log(1.0 - torch.tanh(x) ** 2 + self.epsilon)

# Using Q(s,a) Predict Network for SAC


class SoftQNetwork(nn.Module):
    def __init__(self,
                 env: gym.Env,
                 logger: Logger,
                 device: Type[torch.device] = None,
                 num_blocks: int = 2,
                 hidden_size: int = 256,
                 activation: Type[nn.Module] = nn.ReLU):
        super().__init__()
        self.logger = logger
        self.device = device
        layer_list = nn.ModuleList()

        # layer_list.append(nn.Linear(np.array(env.single_observation_space.shape).prod() +
        #                             np.prod(env.single_action_space.shape), hidden_size[0]))
        # layer_list.append(activation())

        # for i in range(len(hidden_size)-1):
        #     layer_list.append(nn.Linear(hidden_size[i], hidden_size[i+1]))
        #     layer_list.append(activation())

        # layer_list.append(nn.Linear(hidden_size[-1], 1))

        layer_list.append(SACEncoder(block_type="residual",
                                     input_dim=env.single_observation_space.shape[0] +
                                     env.single_action_space.shape[0],
                                     num_blocks=num_blocks,
                                     hidden_dim=hidden_size,))
        layer_list.append(nn.Linear(hidden_size, 1))
        self.q_network = nn.Sequential(*layer_list)

    def forward(self, state, action):
        input = torch.cat([state, action], 1)
        return self.q_network(input)

    def save_model(self, path: str, exp_name: str, iter: int):
        """Save model to a specified path."""
        path = path+"/" if path[-1] != "/" else path
        last_path = str(os.path.join(path, exp_name))+f"_{iter}.pth"
        if os.path.exists(path):
            pass
        else:
            os.makedirs(path, exist_ok=True)
        torch.save(self.state_dict(), last_path)
        self.logger.print("info", f"Q-Net Model saved to {path}")

    def load_model(self, path: str, exp_name: str, iter: int):
        """Load model from a specified path."""
        path = str(os.path.join(path, exp_name))+f"_{iter}.pth"
        self.load_state_dict(torch.load(path,
                                        map_location=self.device,
                                        weights_only=True))
        self.logger.print("info", f"Q-Net Model loaded from {path}")


LOG_STD_MAX = 2
# default from cleanrl
# LOG_STD_MIN = -5
# default from stable-baselines3
LOG_STD_MIN = -20


class SAC_Actor(nn.Module):
    def __init__(self,
                 env: gym.Env,
                 logger: Logger,
                 device: Type[torch.device] = None,
                 #  hidden_size: tuple = (256),
                 num_blocks: int = 1,
                 hidden_size: int = 256,
                 #  activation: Type[nn.Module]=nn.ReLU
                 ):
        super().__init__()
        self.logger = logger
        self.device = device
        # layer_list = nn.ModuleList()
        # layer_list.append(nn.Linear(np.array(env.single_observation_space.shape).prod(), hidden_size[0]))
        # layer_list.append(activation())

        # for i in range(len(hidden_size)-1):
        #     layer_list.append(nn.Linear(hidden_size[i], hidden_size[i+1]))
        #     layer_list.append(activation())

        # self.base_nn = nn.Sequential(*layer_list)
        self.base_nn = SACEncoder(block_type="residual",
                                  input_dim=env.single_observation_space.shape[0],
                                  num_blocks=num_blocks,
                                  hidden_dim=hidden_size,)
        self.actor_mean = nn.Linear(
            hidden_size, np.prod(env.single_action_space.shape))
        self.actor_logstd = nn.Linear(
            hidden_size, np.prod(env.single_action_space.shape))
        self.gaussian_actions: Optional[torch.Tensor] = None

    def forward(self, state):
        input = self.base_nn(state)
        action_mean = self.actor_mean(input)
        action_logstd = self.actor_logstd(input)
        # Original Implementation to cap the standard deviation
        action_logstd = torch.clamp(action_logstd, LOG_STD_MIN, LOG_STD_MAX)

        return action_mean, action_logstd

    def log_prob(self,
                 actions: torch.Tensor,
                 gaussian_actions: Optional[torch.Tensor] = None,
                 epsilon: float = 1e-6) -> torch.Tensor:
        # Inverse tanh
        # Naive implementation (not stable): 0.5 * torch.log((1 + x) / (1 - x))
        # We use numpy to avoid numerical instability
        if gaussian_actions is None:
            # It will be clipped to avoid NaN when inversing tanh
            gaussian_actions = TanhBijector.inverse(actions)

        # Log likelihood for a Gaussian distribution
        log_prob = sum_independent_dims(
            self.distributions.log_prob(gaussian_actions))
        # Squash correction (from original SAC implementation)
        # this comes from the fact that tanh is bijective and differentiable
        log_prob -= torch.sum(torch.log(1 - actions**2 + epsilon), dim=1)
        return log_prob

    def get_action(self,
                   x,
                   deterministic: bool = False):
        mean, log_std = self(x)
        std = log_std.exp()
        self.distributions = torch.distributions.Normal(mean, std)
        if deterministic:
            self.gaussian_actions = self.distributions.mean
            actions = torch.tanh(self.gaussian_actions)
        else:
            self.gaussian_actions = self.distributions.rsample()
            actions = torch.tanh(self.gaussian_actions)
        log_prob = self.log_prob(actions, self.gaussian_actions).reshape(-1, 1)
        # print(f"{log_prob.shape=}")
        return actions, log_prob

    def save_model(self, path: str, exp_name: str, iter: int):
        """Save model to a specified path."""
        path = path+"/" if path[-1] != "/" else path
        last_path = str(os.path.join(path, exp_name))+f"_{iter}.pth"
        if os.path.exists(path):
            pass
        else:
            os.makedirs(path, exist_ok=True)
        torch.save(self.state_dict(), last_path)
        self.logger.print("info", f"Actor Model saved to {path}")

    def load_model(self, path: str, exp_name: str, iter: int):
        """Load model from a specified path."""
        path = str(os.path.join(path, exp_name))+f"_{iter}.pth"
        self.load_state_dict(torch.load(path,
                                        map_location=self.device,
                                        weights_only=True))
        self.logger.print("info", f"Actor Model loaded from {path}")


# specific for single environment only
class SAC():
    def __init__(self,
                 env,
                 config_path: str,
                 config_name: str):

        self.logger = Logger()

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

        self.device = torch.device(
            "cuda" if torch.cuda.is_available() and self.params.cuda_en else "cpu")

        assert isinstance(env.single_action_space,
                          gym.spaces.Box), "only continuous action space is supported"

        self.actor = SAC_Actor(env,
                               self.logger,
                               device=self.device,
                               num_blocks=self.params.policy_num_blocks,
                               hidden_size=self.params.policy_hidden_size).to(self.device)
        self.qf1 = SoftQNetwork(env,
                                self.logger,
                                device=self.device,
                                num_blocks=self.params.critic_num_blocks,
                                hidden_size=self.params.critic_hidden_size).to(self.device)
        self.qf2 = SoftQNetwork(env,
                                self.logger,
                                device=self.device,
                                num_blocks=self.params.critic_num_blocks,
                                hidden_size=self.params.critic_hidden_size).to(self.device)
        self.qf1_target = deepcopy(self.qf1)
        self.qf2_target = deepcopy(self.qf2)
        self.qf1_target.load_state_dict(self.qf1.state_dict())
        self.qf2_target.load_state_dict(self.qf2.state_dict())
        # self.q_optimizer = torch.optim.Adam(list(self.qf1.parameters()) + list(self.qf2.parameters()), lr=self.params.q_lr)
        self.q_optimizer = instantiate(self.params.q_optimizer, params=list(
            self.qf1.parameters()) + list(self.qf2.parameters()))
        # self.actor_optimizer = torch.optim.Adam(list(self.actor.parameters()), lr=self.params.policy_lr)
        self.actor_optimizer = instantiate(
            self.params.policy_optimizer, params=list(self.actor.parameters()))

        # Freeze target networks with respect to optimizers (only update via polyak averaging)
        for p in self.qf1_target.parameters():
            p.requires_grad = False
        for p in self.qf2_target.parameters():
            p.requires_grad = False

        # Automatic entropy tuning
        if isinstance(self.params.ent_coef, str):
            if ("auto" in self.params.ent_coef):
                self.target_entropy = - \
                    torch.prod(torch.Tensor(
                        env.single_action_space.shape).to(self.device)).item()
                self.log_alpha = torch.zeros(
                    1, requires_grad=True, device=self.device)
                self.alpha = self.log_alpha.exp().item()
                # self.a_optimizer = torch.optim.Adam([self.log_alpha], lr=self.params.q_lr)
                self.a_optimizer = instantiate(
                    self.params.alpha_optimizer, params=[self.log_alpha])
        else:
            self.alpha = self.params.ent_coef

        self.env.single_observation_space.dtype = np.float32
        self.buffer = ReplayBuffer(
            buffer_size=self.params.buffer_size,
            observation_space=self.env.single_observation_space,
            action_space=self.env.single_action_space,
            device=self.device,
            handle_timeout_termination=False,
        )

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
        config_path = "."+config_path if config_path[:2] == "./" else \
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
            if (isinstance(cfg[item], dict)):
                for sub_item in cfg[item].keys():
                    new_cfg[sub_item] = cfg[item][sub_item]
            else:
                new_cfg[item] = cfg[item]

        # validating variable type
        if (isinstance(new_cfg["buffer_size"], float)):
            new_cfg["buffer_size"] = int(new_cfg["buffer_size"])

        return new_cfg

    def read_param_cfg(self, params_cfg, verbose: Optional[bool] = True):
        cfg_namespace = SimpleNamespace(**params_cfg)
        if (verbose):
            for key, val in params_cfg.items():
                self.logger.print("info", f"{key} :\t{val}")
        return cfg_namespace

    def train(self,
              total_timesteps: int = int(1e6)):
        start_time = time.time()

        # TRY NOT TO MODIFY: start the game
        obs, _ = self.env.reset(seed=self.params.seed)
        for global_step in range(total_timesteps):
            # ALGO LOGIC: put action logic here
            if global_step < self.params.learning_starts:
                actions = np.array([self.env.single_action_space.sample()
                                   for _ in range(self.env.num_envs)])
            else:
                actions, _ = self.actor.get_action(
                    torch.Tensor(obs).to(self.device))
                actions = actions.detach().cpu().numpy()

            # TRY NOT TO MODIFY: execute the game and log data.
            next_obs, rewards, terminations, truncations, infos = self.env.step(
                actions)

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
                    self.actor.save_model(self.params.save_path,
                                          self.params.exp_name,
                                          int(self.save_iter))
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
                    self.actor.save_model(self.params.save_path,
                                          self.params.exp_name,
                                          int(self.save_iter))
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
            for idx, trunc in enumerate(truncations):
                if trunc:
                    real_next_obs[idx] = infos["final_observation"][idx]
            self.buffer.add(obs, real_next_obs, actions,
                            rewards, terminations, infos)

            # TRY NOT TO MODIFY: CRUCIAL step easy to overlook
            obs = next_obs

            # ALGO LOGIC: training.
            if global_step > self.params.learning_starts:
                data = self.buffer.sample(self.params.batch_size)
                with torch.no_grad():
                    next_state_actions, next_state_log_pi = self.actor.get_action(
                        data.next_observations)
                    qf1_next_target = self.qf1_target(
                        data.next_observations, next_state_actions)
                    qf2_next_target = self.qf2_target(
                        data.next_observations, next_state_actions)
                    min_qf_next_target = torch.min(
                        qf1_next_target, qf2_next_target) - self.alpha * next_state_log_pi
                    next_q_value = data.rewards.flatten() + (1 - data.dones.flatten()) * \
                        self.params.gamma * (min_qf_next_target).view(-1)

                qf1_a_values = self.qf1(
                    data.observations, data.actions).view(-1)
                qf2_a_values = self.qf2(
                    data.observations, data.actions).view(-1)
                qf1_loss = F.mse_loss(qf1_a_values, next_q_value)
                qf2_loss = F.mse_loss(qf2_a_values, next_q_value)
                qf_loss = qf1_loss + qf2_loss

                # optimize the model
                self.q_optimizer.zero_grad()
                qf_loss.backward()
                self.q_optimizer.step()

                if global_step % self.params.policy_frequency == 0:  # TD 3 Delayed update support
                    for _ in range(
                        self.params.policy_frequency
                    ):  # compensate for the delay by doing 'actor_update_interval' instead of 1
                        pi, log_pi = self.actor.get_action(data.observations)
                        qf1_pi = self.qf1(data.observations, pi)
                        qf2_pi = self.qf2(data.observations, pi)
                        min_qf_pi = torch.min(qf1_pi, qf2_pi)
                        actor_loss = ((self.alpha * log_pi) - min_qf_pi).mean()

                        self.actor_optimizer.zero_grad()
                        actor_loss.backward()
                        self.actor_optimizer.step()

                        if isinstance(self.params.ent_coef, str):
                            if ("auto" in self.params.ent_coef):
                                with torch.no_grad():
                                    _, log_pi = self.actor.get_action(
                                        data.observations)
                                alpha_loss = (-self.log_alpha.exp() *
                                              (log_pi + self.target_entropy)).mean()

                                self.a_optimizer.zero_grad()
                                alpha_loss.backward()
                                self.a_optimizer.step()
                                self.alpha = self.log_alpha.exp().item()

                # update the target networks
                if global_step % self.params.target_network_frequency == 0:
                    for param, target_param in zip(self.qf1.parameters(), self.qf1_target.parameters()):
                        target_param.data.copy_(
                            self.params.tau * param.data + (1 - self.params.tau) * target_param.data)
                    for param, target_param in zip(self.qf2.parameters(), self.qf2_target.parameters()):
                        target_param.data.copy_(
                            self.params.tau * param.data + (1 - self.params.tau) * target_param.data)

                if global_step % 100 == 0:
                    self.env.envs[0].writer.add_scalar(
                        "losses/qf1_values", qf1_a_values.mean().item(), global_step)
                    self.env.envs[0].writer.add_scalar(
                        "losses/qf2_values", qf2_a_values.mean().item(), global_step)
                    self.env.envs[0].writer.add_scalar(
                        "losses/qf1_loss", qf1_loss.item(), global_step)
                    self.env.envs[0].writer.add_scalar(
                        "losses/qf2_loss", qf2_loss.item(), global_step)
                    self.env.envs[0].writer.add_scalar(
                        "losses/qf_loss", qf_loss.item() / 2.0, global_step)
                    self.env.envs[0].writer.add_scalar(
                        "losses/actor_loss", actor_loss.item(), global_step)
                    self.env.envs[0].writer.add_scalar(
                        "losses/alpha", self.alpha, global_step)
                    # print("SPS:", int(global_step / (time.time() - start_time)))
                    self.env.envs[0].writer.add_scalar(
                        "charts/SPS", int(global_step / (time.time() - start_time)), global_step)
                    if isinstance(self.params.ent_coef, str):
                        if ("auto" in self.params.ent_coef):
                            self.env.envs[0].writer.add_scalar(
                                "losses/alpha_loss", alpha_loss.item(), global_step)

    def eval_once(self, iter):
        # load model based on last training
        self.actor.load_model(path=self.params.save_path,
                              exp_name=self.params.exp_name,
                              iter=iter)
        if (self.params.use_rsnorm):
            path = self.params.save_path
            path = path+"/" if path[-1] != "/" else path
            last_path = str(os.path.join(
                path, self.params.exp_name))+f"_norm_{int(iter)}.npz"
            data = np.load(last_path)

            self.env.obs_rms.mean = data["mean"]
            self.env.obs_rms.var = data["var"]
            self.env.obs_rms.count = data["count"]
            self.env.obs_rms.is_training = False
        obs, _ = self.env.reset(seed=self.params.seed)

        isExit = False
        global_step = 0
        while (not isExit):
            with torch.no_grad():
                predict_action, _ = self.actor.get_action(torch.from_numpy(
                    obs).to(self.device).float(), deterministic=True)
                next_obs, rew, term, trunc, info = self.env.step(
                    predict_action.cpu().numpy())

            obs = next_obs
            global_step += 1

            if "final_info" in info:
                for info in info["final_info"]:
                    if info and "episode" in info:
                        self.logger.print("info", f"global_step={global_step}, episodic_return={info['episode']['r'].item():.3f}")
                        self.env.envs[0].writer.add_scalar(
                            "charts/eval/episodic_return", info["episode"]["r"], global_step)
                        self.env.envs[0].writer.add_scalar(
                            "charts/eval/episodic_length", info["episode"]["l"], global_step)
                        isExit = True

        self.env.close()
