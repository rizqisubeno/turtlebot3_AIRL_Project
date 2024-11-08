# docs and experiment results can be found at https://docs.cleanrl.dev/rl-algorithms/rpo/#rpo_continuous_actionpy
import os
import random
import time

from library.ICM import ICM

import gymnasium as gym
import numpy as np
import torch
import torch.nn as nn
from torch.optim.adam import Adam
# import tyro
from torch import distributions as td
from torch.distributions.normal import Normal
from library.clipped_gaussian import ClippedGaussian

from torch.utils.tensorboard.writer import SummaryWriter

from typing import Optional, Type, Union
from types import SimpleNamespace

# from library.tb3_agent import logger
import logging
from colorlog import ColoredFormatter

distribution_classes = {
    'Normal': Normal,
    'ClippedGaussian': ClippedGaussian,
}

class Logger():
    def __init__(self):
        LOG_LEVEL = logging.DEBUG
        LOGFORMAT = "  %(log_color)s%(levelname)-8s%(reset)s | %(log_color)s%(message)s%(reset)s"
        logging.root.setLevel(LOG_LEVEL)
        formatter = ColoredFormatter(LOGFORMAT)
        stream = logging.StreamHandler()
        stream.setLevel(LOG_LEVEL)
        stream.setFormatter(formatter)
        self.log = logging.getLogger('pythonConfig')
        self.log.setLevel(LOG_LEVEL)
        self.log.addHandler(stream)

    def print(self,
            type:str="info",
            msg:str=""):
        
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
                 logger, 
                 distribution:str="Normal",
                 rpo_alpha:Union[int, bool]=0.1,
                 activation:Type[nn.Module]=nn.Tanh,
                 use_tanh_output:bool=False,
                 device:Optional[torch.device]=None):
        super().__init__()
        self.device = device
        self.rpo_alpha = rpo_alpha
        self.logger = logger
        self.envs = envs
        if distribution not in distribution_classes:
            raise ValueError(f"Unknown distribution: {distribution}")
        else:
            self.distribution = distribution_classes[distribution]
            self.logger.print("info", f"use distribution: {self.distribution}")
        self.low = torch.from_numpy(self.envs.action_space.low).to(self.device)
        self.high = torch.from_numpy(self.envs.action_space.high).to(self.device)

        self.critic = nn.Sequential(
            layer_init(nn.Linear(np.array(envs.single_observation_space.shape).prod(), 64)),
            activation(),
            layer_init(nn.Linear(64, 64)),
            activation(),
            layer_init(nn.Linear(64, 1), std=1.0),
        )

        actor_layer = [nn.Linear(np.array(envs.single_observation_space.shape).prod(), 64), activation()]
        actor_layer.append(layer_init(nn.Linear(64, 64)))
        actor_layer.append(activation())
        actor_layer.append(layer_init(nn.Linear(64, np.prod(envs.single_action_space.shape)), std=0.01))
        actor_layer.append(activation()) if use_tanh_output else None
        
        self.actor_mean = nn.Sequential(*actor_layer)
        # print(f"{self.actor_mean}")
        self.actor_logstd = nn.Parameter(torch.zeros(1, np.prod(envs.single_action_space.shape)))

    def get_value(self, x):
        return self.critic(x)
    
    def get_action(self,
                   x,
                   deterministic:bool=False):
        with torch.no_grad():
            action_mean = self.actor_mean(x)
            action_logstd = self.actor_logstd.expand_as(action_mean)
            action_std = torch.exp(action_logstd)
            if(deterministic):
                action = action_mean
            else:
                if self.distribution==Normal:
                    action = Normal(loc=action_mean,
                                    scale=action_std).sample()
                elif self.distribution==ClippedGaussian:
                    action = ClippedGaussian(mean=action_mean,
                                            var=action_std**2, 
                                            low=self.low, 
                                            high=self.high).sample()
        return action        

    # RPO_Mode is Robust Policy Optimization Mode
    def get_action_and_value(self, x, action=None, RPO_Mode:Optional[bool]=False):
        action_mean = self.actor_mean(x)
        action_logstd = self.actor_logstd.expand_as(action_mean)
        action_std = torch.exp(action_logstd)
        if self.distribution==Normal:
            probs = Normal(loc=action_mean,
                            scale=action_std)
        elif self.distribution==ClippedGaussian:
            probs = ClippedGaussian(mean=action_mean,
                                    var=action_std**2, 
                                    low=self.low, 
                                    high=self.high)
        if action is None:
            action = probs.sample()
        elif RPO_Mode:  # new to RPO
            # sample again to add stochasticity to the policy
            z = torch.FloatTensor(action_mean.shape).uniform_(-self.rpo_alpha, self.rpo_alpha).to(self.device)
            action_mean = action_mean + z
            if self.distribution==Normal:
                probs = Normal(loc=action_mean,
                                scale=action_std)
            elif self.distribution==ClippedGaussian:
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
        path = path+"/" if path[-1]!="/" else path
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
                 params_cfg):
        
        self.env = env
        self.num_envs = 1

        self.params = self.read_param_cfg(params_cfg=params_cfg)

        self.batch_size = self.num_envs * self.params.num_steps 
        self.minibatch_size = self.batch_size // self.params.num_minibatches
        
        # TRY NOT TO MODIFY: seeding
        random.seed(self.params.seed)
        np.random.seed(self.params.seed)
        torch.manual_seed(self.params.seed)
        torch.backends.cudnn.deterministic = self.params.torch_deterministic
        
        self.device = torch.device("cuda" if torch.cuda.is_available() and self.params.cuda_en else "cpu")
        
        assert isinstance(env.single_action_space, gym.spaces.Box), "only continuous action space is supported"

        self.logger = Logger()

        # Initialize PPO_Agent_NN Function
        self.agent = PPO_Agent_NN(env, 
                                  logger=self.logger,
                                  distribution=self.params.distributions,
                                  rpo_alpha=self.params.rpo_alpha, 
                                  activation=self.params.activation_fn,
                                  use_tanh_output=self.params.use_tanh_output,
                                  device=self.device).to(self.device)
        self.optimizer = Adam(self.agent.parameters(), lr=self.params.learning_rate, eps=1e-8)
        if (self.params.use_icm):
            self.ICM = ICM(observation_shape=self.env.single_observation_space.shape[0],
                           action_shape=self.env.single_action_space.shape[0],
                           activation_func=nn.ReLU,
                           layer_size=2,
                           hidden_size=256,
                           alpha=0.1,
                           beta=0.2,
                           ).to(self.device)
            self.icm_optimizer = Adam(self.ICM.parameters(), lr=self.params.learning_rate, eps=1e-8)

        try:
            _ = self.env.unwrapped.writer
            self.logger.print("info", "SummaryWriter Found on Env")
            self.env.envs[0].writer.add_text(
                "rl_hyperparameters",
                "|param|value|\n|-|-|\n%s" % ("\n".join([f"|{key}|{value}|" for key, value in vars(self.params).items()])),
            )
        except AttributeError:
            self.logger.print("info", "Create SummaryWriter inside Env")
            self.env.envs[0].writer = SummaryWriter(f"runs/{self.params.exp_name}")

        self.save_config = "reset" if self.params.save_every_reset else "step"

        if("reset" in self.save_config):
            self.reset_counter = 0
            self.save_iter = 0
        elif("step" in self.save_config):
            self.num_timestep = 0
            self.save_iter = 0

    def read_param_cfg(self, params_cfg, verbose:Optional[bool]=False):
        cfg_namespace = SimpleNamespace(**params_cfg)
        if(verbose):
            for key, val in cfg_namespace.items():
                self.logger.print("info", f"{key} :\t{val}")
        return cfg_namespace 
    
    def train(self, total_timesteps:Optional[int]=int(1e6)):
        self.num_iterations = int(total_timesteps // self.batch_size)
        # ALGO Logic: Storage setup
        obs = torch.zeros((self.params.num_steps, self.num_envs) + self.env.single_observation_space.shape).to(self.device)
        next_obs = torch.zeros((self.params.num_steps, self.num_envs) + self.env.single_observation_space.shape).to(self.device)
        actions = torch.zeros((self.params.num_steps, self.num_envs) + self.env.single_action_space.shape).to(self.device)
        logprobs = torch.zeros((self.params.num_steps, self.num_envs)).to(self.device)
        rewards = torch.zeros((self.params.num_steps, self.num_envs)).to(self.device)
        next_dones = torch.zeros((self.params.num_steps, self.num_envs)).to(self.device)
        next_terminations = torch.zeros((self.params.num_steps, self.num_envs)).to(self.device)
        values = torch.zeros((self.params.num_steps, self.num_envs)).to(self.device)

        # TRY NOT TO MODIFY: start the game
        global_step = 0
        start_time = time.time()
        next_ob, _ = self.env.reset(seed=self.params.seed)
        next_ob = torch.Tensor(next_ob).to(self.device)
        next_done = torch.zeros(self.num_envs).to(self.device)
        next_termination = torch.zeros(self.num_envs).to(self.device)

        for iteration in range(1, self.num_iterations + 1):
            # Annealing the rate if instructed to do so.
            if self.params.anneal_lr:
                frac = 1.0 - (iteration - 1.0) / self.num_iterations
                lrnow = frac * self.params.learning_rate
                self.optimizer.param_groups[0]["lr"] = lrnow

            for step in range(0, self.params.num_steps):
                global_step += self.num_envs
                
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
                if(self.params.use_icm):
                    next_observation = torch.Tensor(next_ob).to(self.device)
                    intrinsic_reward, inv_loss, fw_loss = self.ICM.calc_loss(ob, action, next_observation)
                    # print(f"{reward=}")
                    # print(f"{intrinsic_reward=}")
                    self.env.envs[0].writer.add_scalar("charts/ICM/env_reward", 
                                                       reward, 
                                                       global_step)
                    self.env.envs[0].writer.add_scalar("charts/ICM/int_reward", 
                                                       intrinsic_reward.cpu().detach().numpy(), 
                                                       global_step)
                    reward += intrinsic_reward.cpu().detach().numpy()

                    total_loss = inv_loss + fw_loss
                    self.icm_optimizer.zero_grad()
                    total_loss.backward()
                    self.icm_optimizer.step()

                    self.env.envs[0].writer.add_scalar("charts/ICM/total_reward", 
                                                       reward, 
                                                       global_step)

                assert np.isnan(action_clipped).any()==False
                
                # Correct next observation (for vec gym)
                real_next_ob = next_ob.copy()
                for idx, trunc in enumerate(next_truncation):
                    if trunc:
                        real_next_ob[idx] = info["final_observation"][idx]
                next_ob = torch.Tensor(next_ob).to(self.device)
                
                # Collect trajectory
                obs[step] = torch.Tensor(ob).to(self.device)
                next_obs[step] = torch.Tensor(real_next_ob).to(self.device)
                actions[step] = torch.Tensor(action).to(self.device)
                logprobs[step] = torch.Tensor(logprob).to(self.device)
                values[step] = torch.Tensor(value.flatten()).to(self.device)
                next_terminations[step] = torch.Tensor(next_termination).to(self.device)
                next_dones[step] = torch.Tensor(np.logical_or(next_termination, next_truncation)).to(self.device)
                rewards[step] = torch.tensor(reward).to(self.device).view(-1)

                # writing returns episode and length episode
                if "final_info" in info:
                    for info in info["final_info"]:
                        if info and "episode" in info:
                            self.logger.print("info", f"global_step={global_step}, episodic_return={info['episode']['r'].item():.3f}")
                            self.env.envs[0].writer.add_scalar("charts/episodic_return", info["episode"]["r"], global_step)
                            self.env.envs[0].writer.add_scalar("charts/episodic_length", info["episode"]["l"], global_step)
                            if("reset" in self.save_config):
                                self.reset_counter += 1

                # saving model every step
                if("reset" in self.save_config):
                    if (self.reset_counter % self.params.save_step == 0 and self.reset_counter>0):
                        self.reset_counter = 0
                        self.save_iter += 1
                        self.agent.save_model(self.params.save_path,
                                              self.params.exp_name,
                                              int(self.save_iter))
                elif("step" in self.save_config):
                     self.num_timestep += 1
                     if (self.num_timestep % self.params.save_step == 0 and self.num_timestep>=0):
                        self.num_timestep = 0
                        self.save_iter += 1
                        self.agent.save_model(self.params.save_path,
                                              self.params.exp_name,
                                              int(self.save_iter))
                
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
                        next_values[value_mask] = self.agent.get_value(next_obs[t][value_mask]).flatten()
                        next_values[~value_mask] = values[t + 1][~value_mask]
                    delta = rewards[t] + self.params.gamma * next_values * (1 - next_terminations[t]) - values[t]
                    advantages[t] = lastgaelam = delta + self.params.gamma * self.params.gae_lambda * (1 - next_dones[t]) * lastgaelam
                returns = advantages + values

            # flatten the batch
            b_obs = obs.reshape((-1,) + self.env.single_observation_space.shape)
            b_logprobs = logprobs.reshape(-1)
            b_actions = actions.reshape((-1,) + self.env.single_action_space.shape)
            b_advantages = advantages.reshape(-1)
            b_returns = returns.reshape(-1)
            b_values = values.reshape(-1)

            # Optimizing the policy and value network
            b_inds = np.arange(self.batch_size)
            clipfracs = []
            for _ in range(self.params.num_epoch):
                np.random.shuffle(b_inds)
                approx_kl_list = []
                for start in range(0, self.batch_size, self.minibatch_size):
                    end = start + self.minibatch_size
                    mb_inds = b_inds[start:end]
                    # print(f"{b_actions[mb_inds]=}")
                    _, newlogprob, entropy, newvalue = self.agent.get_action_and_value(b_obs[mb_inds], b_actions[mb_inds])
                    logratio = newlogprob - b_logprobs[mb_inds]
                    ratio = logratio.exp()

                    with torch.no_grad():
                        # calculate approx_kl http://joschu.net/blog/kl-approx.html
                        old_approx_kl = (-logratio).mean()
                        approx_kl = ((ratio - 1) - logratio).mean()
                        clipfracs += [((ratio - 1.0).abs() > self.params.clip_coef).float().mean().item()]
                    
                    approx_kl_list.append(approx_kl.item())

                    mb_advantages = b_advantages[mb_inds]
                    if self.params.norm_adv:
                        mb_advantages = (mb_advantages - mb_advantages.mean()) / (mb_advantages.std() + 1e-8)

                    # Policy loss
                    pg_loss1 = -mb_advantages * ratio
                    pg_loss2 = -mb_advantages * torch.clamp(ratio, 1 - self.params.clip_coef, 1 + self.params.clip_coef)
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
                        v_loss = 0.5 * ((newvalue - b_returns[mb_inds]) ** 2).mean()

                    entropy_loss = entropy.mean()
                    loss = pg_loss - self.params.ent_coef * entropy_loss + v_loss * self.params.vf_coef

                    self.optimizer.zero_grad()
                    loss.backward()
                    nn.utils.clip_grad_norm_(self.agent.parameters(), self.params.max_grad_norm)
                    self.optimizer.step()

                # self.logger.print("hidden", "RPO", f"approx kl max:{np.max(approx_kl_list):.3f}, min:{np.min(approx_kl_list):.7f}")
                if self.params.target_kl is not None:
                    if approx_kl > self.params.target_kl:
                        self.logger.print("err", f"approx kl reach target")
                        break

            y_pred, y_true = b_values.cpu().numpy(), b_returns.cpu().numpy()
            var_y = np.var(y_true)
            explained_var = np.nan if var_y == 0 else 1 - np.var(y_true - y_pred) / var_y

            # TRY NOT TO MODIFY: record rewards for plotting purposes
            self.env.envs[0].writer.add_scalar("charts/learning_rate", self.optimizer.param_groups[0]["lr"], global_step)
            self.env.envs[0].writer.add_scalar("losses/value_loss", v_loss.item(), global_step)
            self.env.envs[0].writer.add_scalar("losses/policy_loss", pg_loss.item(), global_step)
            self.env.envs[0].writer.add_scalar("losses/entropy", entropy_loss.item(), global_step)
            self.env.envs[0].writer.add_scalar("losses/old_approx_kl", old_approx_kl.item(), global_step)
            self.env.envs[0].writer.add_scalar("losses/approx_kl", approx_kl.item(), global_step)
            self.env.envs[0].writer.add_scalar("losses/clipfrac", np.mean(clipfracs), global_step)
            self.env.envs[0].writer.add_scalar("losses/explained_variance", explained_var, global_step)
            #print("SPS:", int(global_step / (time.time() - start_time)))
            self.env.envs[0].writer.add_scalar("charts/SPS", int(global_step / (time.time() - start_time)), global_step)

        self.env.envs[0].writer.close()
        self.env.close()

    def eval_once(self, iter):
        # load model based on last training
        self.agent.load_model(path=self.params.save_path, 
                                exp_name=self.params.exp_name, 
                                iter=iter)
        obs, _ = self.env.reset(seed=self.params.seed)
        
        isExit = False
        global_step = 0
        while(isExit==False):
            predict_action = self.agent.get_action(torch.from_numpy(obs).to(self.device), deterministic=True)
            next_obs, rew, term, trunc, info = self.env.step(predict_action.cpu().numpy())
            
            obs = next_obs
            global_step += 1

            if "final_info" in info:
                for info in info["final_info"]:
                    if info and "episode" in info:
                        self.logger.print("info", f"global_step={global_step}, episodic_return={info['episode']['r'].item():.3f}")
                        self.env.envs[0].writer.add_scalar("charts/eval/episodic_return", info["episode"]["r"], global_step)
                        self.env.envs[0].writer.add_scalar("charts/eval/episodic_length", info["episode"]["l"], global_step)
                        isExit=True

        self.env.close()

class PPG():
    """
    Phasic Policy Gradients Algorithm
    """
    def __init__(self,
                 env,
                 params_cfg):
        
        self.env = env
        self.num_envs = 1

        self.params = self.read_param_cfg(params_cfg=params_cfg)

        self.batch_size = self.num_envs * self.params.num_steps 
        self.minibatch_size = self.batch_size // self.params.num_minibatches
        
        # TRY NOT TO MODIFY: seeding
        random.seed(self.params.seed)
        np.random.seed(self.params.seed)
        torch.manual_seed(self.params.seed)
        torch.backends.cudnn.deterministic = self.params.torch_deterministic
        # passing to env
        self.env.seed = self.params.seed
        
        self.device = torch.device("cuda" if torch.cuda.is_available() and self.params.cuda_en else "cpu")
        
        assert isinstance(env.single_action_space, gym.spaces.Box), "only continuous action space is supported"

        # Initialize PPO_Agent_NN Function
        self.agent = PPG_Agent_NN(env,  
                        activation=self.params.activation_fn,
                        device=self.device).to(self.device)
        self.optimizer = Adam(self.agent.parameters(), lr=self.params.learning_rate, eps=1e-8)
        
        self.logger = Logger()

        # if hasattr(self.env.envs[0], 'writer'):
        # if self.env.get_wrapper_attr('writer'):
        try:
            _ = self.env.unwrapped.writer
            self.logger.print("info", "SummaryWriter Found on Env")
            self.env.envs[0].writer.add_text(
                "rl_hyperparameters",
                "|param|value|\n|-|-|\n%s" % ("\n".join([f"|{key}|{value}|" for key, value in vars(self.params).items()])),
            )
        except AttributeError:
            self.logger.print("info", "Create SummaryWriter inside Env")
            self.env.envs[0].writer = SummaryWriter(f"runs/{self.params.exp_name}")

    def read_param_cfg(self, params_cfg, verbose:Optional[bool]=False):
        head_name = "PPG_read_param"
        cfg_namespace = SimpleNamespace(**params_cfg)
        if(verbose):
            for key, val in cfg_namespace.items():
                self.logger.print("info", f"{key} :\t{val}")
        return cfg_namespace 
    
    def train(self, total_timesteps:Optional[int]=int(1e6)):
        self.num_iterations = int(total_timesteps // self.batch_size)
        self.num_phases = int(self.num_iterations // self.params.n_iteration)
        self.aux_batch_rollouts = int(self.num_envs * self.params.n_iteration)
        # ALGO Logic: Storage setup
        obs = torch.zeros((self.params.num_steps, self.num_envs) + self.env.single_observation_space.shape).to(self.device)
        next_obs = torch.zeros((self.params.num_steps, self.num_envs) + self.env.single_observation_space.shape).to(self.device)
        actions = torch.zeros((self.params.num_steps, self.num_envs) + self.env.single_action_space.shape).to(self.device)
        logprobs = torch.zeros((self.params.num_steps, self.num_envs)).to(self.device)
        rewards = torch.zeros((self.params.num_steps, self.num_envs)).to(self.device)
        next_dones = torch.zeros((self.params.num_steps, self.num_envs)).to(self.device)
        next_terminations = torch.zeros((self.params.num_steps, self.num_envs)).to(self.device)
        values = torch.zeros((self.params.num_steps, self.num_envs)).to(self.device)
        aux_obs = torch.zeros((self.params.num_steps, 
                               self.aux_batch_rollouts) + self.env.single_observation_space.shape)  # Saves lot system RAM
        aux_returns = torch.zeros((self.params.num_steps, self.aux_batch_rollouts))

        # TRY NOT TO MODIFY: start the game
        global_step = 0
        start_time = time.time()
        next_ob, _ = self.env.reset(seed=self.params.seed)
        next_ob = torch.Tensor(next_ob).to(self.device)
        next_termination = torch.zeros(self.num_envs).to(self.device)

        for phase in range(1, self.num_phases + 1):

            #Policy Phase
            self.logger.print("hidden", "Policy Phase Training...")
            for iteration in range(1, self.params.n_iteration + 1):
                # Annealing the rate if instructed to do so.
                if self.params.anneal_lr:
                    frac = 1.0 - (iteration - 1.0) / self.num_iterations
                    lrnow = frac * self.params.learning_rate
                    self.optimizer.param_groups[0]["lr"] = lrnow

                for step in range(0, self.params.num_steps):
                    global_step += self.num_envs
                    
                    ob = next_ob
                    # ALGO LOGIC: action logic
                    with torch.no_grad():
                        action, logprob, _, value = self.agent.get_action_and_value(ob)

                    # TRY NOT TO MODIFY: execute the game and log data.
                    next_ob, reward, next_termination, next_truncation, info = self.env.step(action.cpu().numpy())
                    
                    # Correct next observation (for vec gym)
                    real_next_ob = next_ob.copy()
                    for idx, trunc in enumerate(next_truncation):
                        if trunc:
                            real_next_ob[idx] = info["final_observation"][idx]
                    next_ob = torch.Tensor(next_ob).to(self.device)
                    
                    # Collect trajectory
                    obs[step] = torch.Tensor(ob).to(self.device)
                    next_obs[step] = torch.Tensor(real_next_ob).to(self.device)
                    actions[step] = torch.Tensor(action).to(self.device)
                    logprobs[step] = torch.Tensor(logprob).to(self.device)
                    values[step] = torch.Tensor(value.flatten()).to(self.device)
                    next_terminations[step] = torch.Tensor(next_termination).to(self.device)
                    next_dones[step] = torch.Tensor(np.logical_or(next_termination, next_truncation)).to(self.device)
                    rewards[step] = torch.tensor(reward).to(self.device).view(-1)

                    # print(f"{info=}")
                    if "final_info" in info:
                        for info in info["final_info"]:
                            if info and "episode" in info:
                                self.logger.print("info",f"global_step={global_step}, episodic_return={info['episode']['r']}")
                                self.env.envs[0].writer.add_scalar("charts/episodic_return", info["episode"]["r"], global_step)
                                self.env.envs[0].writer.add_scalar("charts/episodic_length", info["episode"]["l"], global_step)

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
                            next_values[value_mask] = self.agent.get_value(next_obs[t][value_mask]).flatten()
                            next_values[~value_mask] = values[t + 1][~value_mask]
                        delta = rewards[t] + self.params.gamma * next_values * (1 - next_terminations[t]) - values[t]
                        advantages[t] = lastgaelam = delta + self.params.gamma * self.params.gae_lambda * (1 - next_dones[t]) * lastgaelam
                    returns = advantages + values

                # flatten the batch
                b_obs = obs.reshape((-1,) + self.env.single_observation_space.shape)
                b_logprobs = logprobs.reshape(-1)
                b_actions = actions.reshape((-1,) + self.env.single_action_space.shape)
                b_advantages = advantages.reshape(-1)
                b_returns = returns.reshape(-1)
                b_values = values.reshape(-1)

                # PPG code does full batch advantage normalization
                if self.params.norm_adv_fullbatch:
                    b_advantages = (b_advantages - b_advantages.mean()) / (b_advantages.std() + 1e-8)

                # Optimizing the policy and value network
                b_inds = np.arange(self.batch_size)
                clipfracs = []
                for _ in range(self.params.e_policy):
                    np.random.shuffle(b_inds)
                    approx_kl_list = []
                    for start in range(0, self.batch_size, self.minibatch_size):
                        end = start + self.minibatch_size
                        mb_inds = b_inds[start:end]

                        _, newlogprob, entropy, newvalue = self.agent.get_action_and_value(b_obs[mb_inds], b_actions[mb_inds])
                        logratio = newlogprob - b_logprobs[mb_inds]
                        ratio = logratio.exp()

                        with torch.no_grad():
                            # calculate approx_kl http://joschu.net/blog/kl-approx.html
                            old_approx_kl = (-logratio).mean()
                            approx_kl = ((ratio - 1) - logratio).mean()
                            clipfracs += [((ratio - 1.0).abs() > self.params.clip_coef).float().mean().item()]
                        
                        approx_kl_list.append(approx_kl.item())

                        mb_advantages = b_advantages[mb_inds]
                        if self.params.norm_adv:
                            mb_advantages = (mb_advantages - mb_advantages.mean()) / (mb_advantages.std() + 1e-8)

                        # Policy loss
                        pg_loss1 = -mb_advantages * ratio
                        pg_loss2 = -mb_advantages * torch.clamp(ratio, 1 - self.params.clip_coef, 1 + self.params.clip_coef)
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
                            v_loss = 0.5 * ((newvalue - b_returns[mb_inds]) ** 2).mean()

                        entropy_loss = entropy.mean()
                        loss = pg_loss - self.params.ent_coef * entropy_loss + v_loss * self.params.vf_coef

                        self.optimizer.zero_grad()
                        loss.backward()
                        nn.utils.clip_grad_norm_(self.agent.parameters(), self.params.max_grad_norm)
                        self.optimizer.step()

                    # self.logger.print("hidden", "RPO", f"approx kl max:{np.max(approx_kl_list):.3f}, min:{np.min(approx_kl_list):.7f}")
                    if self.params.target_kl is not None:
                        if approx_kl > self.params.target_kl:
                            self.logger.print("err", f"approx kl reach target")
                            break

                y_pred, y_true = b_values.cpu().numpy(), b_returns.cpu().numpy()
                var_y = np.var(y_true)
                explained_var = np.nan if var_y == 0 else 1 - np.var(y_true - y_pred) / var_y

                # TRY NOT TO MODIFY: record rewards for plotting purposes
                self.env.envs[0].writer.add_scalar("charts/learning_rate", self.optimizer.param_groups[0]["lr"], global_step)
                self.env.envs[0].writer.add_scalar("losses/value_loss", v_loss.item(), global_step)
                self.env.envs[0].writer.add_scalar("losses/policy_loss", pg_loss.item(), global_step)
                self.env.envs[0].writer.add_scalar("losses/entropy", entropy_loss.item(), global_step)
                self.env.envs[0].writer.add_scalar("losses/old_approx_kl", old_approx_kl.item(), global_step)
                self.env.envs[0].writer.add_scalar("losses/approx_kl", approx_kl.item(), global_step)
                self.env.envs[0].writer.add_scalar("losses/clipfrac", np.mean(clipfracs), global_step)
                self.env.envs[0].writer.add_scalar("losses/explained_variance", explained_var, global_step)
                #print("SPS:", int(global_step / (time.time() - start_time)))
                self.env.envs[0].writer.add_scalar("charts/SPS", int(global_step / (time.time() - start_time)), global_step)

                # PPG Storage - Rollouts are saved without flattening for sampling full rollouts later:
                storage_slice = slice(self.num_envs * (iteration - 1), self.num_envs * iteration)
                aux_obs[:, storage_slice] = obs.cpu().clone()
                aux_returns[:, storage_slice] = returns.cpu().clone()

            # Auxiliary Phase
            self.logger.print("hidden", "Auxiliary Phase Training...")
            aux_inds = np.arange(self.aux_batch_rollouts)

                # Build the old policy on the aux buffer before distilling to the network
            aux_pi_mean = torch.zeros((self.params.num_steps, 
                                self.aux_batch_rollouts, 
                                np.prod(self.env.single_action_space.shape)
                                ))
            aux_pi_std = torch.zeros((self.params.num_steps, 
                                self.aux_batch_rollouts, 
                                np.prod(self.env.single_action_space.shape)
                                ))
            for i, start in enumerate(range(0, self.aux_batch_rollouts, self.params.num_aux_rollouts)):
                end = start + self.params.num_aux_rollouts
                aux_minibatch_ind = aux_inds[start:end]
                m_aux_obs = aux_obs[:, aux_minibatch_ind].to(torch.float32).to(self.device)
                m_obs_shape = m_aux_obs.shape
                # m_aux_obs = flatten01(m_aux_obs)      # uncomment this because obs is 1-d not 2-d
                # print(f"{m_aux_obs=}")
                with torch.no_grad():
                    pi_mean, pi_std = self.agent.get_pi(m_aux_obs)
                # aux_pi[:, aux_minibatch_ind] = unflatten01(pi_logits, m_obs_shape[:2]) # we don't use unflatten because obs is already 1-d
                aux_pi_mean[:, aux_minibatch_ind] = pi_mean.cpu().clone()
                aux_pi_std[:, aux_minibatch_ind] = pi_std.cpu().clone()
                del m_aux_obs

            for auxiliary_update in range(1, self.params.e_auxiliary + 1):
                # print(f"aux epoch {auxiliary_update}")
                np.random.shuffle(aux_inds)
                for i, start in enumerate(range(0, self.aux_batch_rollouts, self.params.num_aux_rollouts)):
                    end = start + self.params.num_aux_rollouts
                    aux_minibatch_ind = aux_inds[start:end]
                    try:
                        m_aux_obs = aux_obs[:, aux_minibatch_ind].to(self.device)
                        m_obs_shape = m_aux_obs.shape
                        # m_aux_obs = flatten01(m_aux_obs)  # Sample full rollouts for PPG instead of random indexes (already 1-d data)
                        m_aux_returns = aux_returns[:, aux_minibatch_ind].to(torch.float32).to(self.device)
                        #m_aux_returns = flatten01(m_aux_returns) already 1-d data not to be flattening
                        # print(f"{aux_returns.shape=}")
                        new_pi_mean, new_pi_std, new_values, new_aux_values = self.agent.get_pi_value_and_aux_value(m_aux_obs)

                        # print(f"{new_values.shape=}")
                        # new_values = new_values.view(-1)
                        # new_aux_values = new_aux_values.view(-1)
                        new_values = new_values.squeeze(-1)
                        new_aux_values = new_aux_values.squeeze(-1)
                        # print(f"{new_values.shape=}")
                        # old_pi_logits = flatten01(aux_pi[:, aux_minibatch_ind]).to(device)
                        old_pi_mean = aux_pi_mean[:, aux_minibatch_ind].to(self.device)
                        old_pi_std = aux_pi_std[:, aux_minibatch_ind].to(self.device)

                        old_pi = Normal(loc=old_pi_mean, scale=old_pi_std)
                        new_pi = Normal(loc=new_pi_mean, scale=new_pi_std)

                        kl_loss = td.kl_divergence(old_pi, new_pi).mean()

                        # print(f"{m_aux_returns.shape=}")
                        real_value_loss = 0.5 * ((new_values - m_aux_returns) ** 2).mean()
                        aux_value_loss = 0.5 * ((new_aux_values - m_aux_returns) ** 2).mean()
                        joint_loss = aux_value_loss + self.params.beta_clone * kl_loss

                        loss = (joint_loss + real_value_loss) / self.params.n_aux_grad_accum
                        loss.backward()

                        if (i + 1) % self.params.n_aux_grad_accum == 0:
                            nn.utils.clip_grad_norm_(self.agent.parameters(), self.params.max_grad_norm)
                            self.optimizer.step()
                            self.optimizer.zero_grad()  # This cannot be outside, else gradients won't accumulate

                    except RuntimeError as e:
                        raise Exception(
                            "if running out of CUDA memory, try a higher --n-aux-grad-accum, which trades more time for less gpu memory"
                        ) from e

                    del m_aux_obs, m_aux_returns
            self.env.envs[0].writer.add_scalar("losses/aux/kl_loss", kl_loss.mean().item(), global_step)
            self.env.envs[0].writer.add_scalar("losses/aux/aux_value_loss", aux_value_loss.item(), global_step)
            self.env.envs[0].writer.add_scalar("losses/aux/real_value_loss", real_value_loss.item(), global_step)

        self.env.envs[0].writer.close()