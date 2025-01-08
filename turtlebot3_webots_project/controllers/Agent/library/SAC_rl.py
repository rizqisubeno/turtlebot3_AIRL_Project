import os
import time
from copy import deepcopy
import random
import gymnasium as gym

from types import SimpleNamespace
from typing import Optional, Type, Union

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim.adam import Adam
from torch.utils.tensorboard.writer import SummaryWriter
from stable_baselines3.common.buffers import ReplayBuffer

import hydra
from hydra.utils import instantiate
from .normalize import NormalizeObservation
from omegaconf import OmegaConf
from .Simba_Network import SACEncoder

from .PPO_rl import Logger

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
                 num_blocks: int = 2,
                 hidden_size: int = 256,
                 activation: Type[nn.Module] = nn.ReLU):
        super().__init__()
        self.logger = logger
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
                 #  hidden_size: tuple = (256),
                 num_blocks: int = 1,
                 hidden_size: int = 256,
                 #  activation: Type[nn.Module]=nn.ReLU
                 ):
        super().__init__()
        self.logger = logger
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
            assert isinstance(self.gaussian_actions, torch.Tensor)
            actions = torch.tanh(self.gaussian_actions)
        else:
            self.gaussian_actions = self.distributions.rsample()
            assert isinstance(self.gaussian_actions, torch.Tensor)
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
                                        weights_only=True))
        self.logger.print("info", f"Actor Model loaded from {path}")


# specific for single environment only
class SAC():
    def __init__(self,
                 env,
                 config_path: str,
                 config_name: str):

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

        self.device = torch.device("cuda" if torch.cuda.is_available() and self.params.cuda_en 
                                   else "cpu")

        assert isinstance(env.single_action_space,
                          gym.spaces.Box), "only continuous action space is supported"

        self.actor = SAC_Actor(env,
                               self.logger,
                               num_blocks=self.params.policy_num_blocks,
                               hidden_size=self.params.policy_hidden_size).to(self.device)
        self.qf1 = SoftQNetwork(env,
                                self.logger,
                                num_blocks=self.params.critic_num_blocks,
                                hidden_size=self.params.critic_hidden_size).to(self.device)
        self.qf2 = SoftQNetwork(env,
                                self.logger,
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
            assert isinstance(truncations, np.ndarray)
            assert isinstance(rewards, np.ndarray)
            assert isinstance(infos, dict)
            for idx, trunc in enumerate(truncations):
                if trunc:
                    real_next_obs[idx] = infos["final_observation"][idx]
            self.buffer.add(obs, real_next_obs, actions,
                            rewards, np.array(terminations), [infos])

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
