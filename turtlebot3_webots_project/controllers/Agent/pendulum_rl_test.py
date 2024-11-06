from library.custom_rl_algo import PPG, PPO

import torch as th
import numpy as np

import gymnasium as gym
from gymnasium.wrappers.record_episode_statistics import RecordEpisodeStatistics
from gymnasium.wrappers.rescale_action import RescaleAction

# env = gym.make_vec('MountainCarContinuous-v0', num_envs=1, vectorization_mode="sync")
params_PPO = {"exp_name"            :   "RL_PPO_Gaussian",
              "save_every_reset"    :   True,           # choose one save every reset or save every n step
              "save_every_step"     :   False,
              "save_step"           :   10,             # optional if save every n step or reset true otherwise you can uncomment
              "save_path"           :   "./models/RL_PPO",
              "seed"                :   1,
              "cuda_en"             :   True,
              "torch_deterministic" :   True,
              "num_steps"           :   200,
              "num_minibatches"     :   40,
              "num_epoch"           :   10,
              "learning_rate"       :   3e-4,
              "activation_fn"       :   th.nn.Tanh,
              "anneal_lr"           :   False,
              "gamma"               :   0.99,
              "gae_lambda"          :   0.95,
              "norm_adv"            :   True,
              "clip_coef"           :   0.2,
              "clip_vloss"          :   False,
              "ent_coef"            :   0.005,
              "vf_coef"             :   0.5,
              "max_grad_norm"       :   1.0,
              "target_kl"           :   None,
              "rpo_alpha"           :   0.01,
              "distributions"       :   "Normal",
              "use_tanh_output"     :   False,
              "use_icm"             :   True,
                  }


# params_PPG = {"exp_name"            :   "RL_PPG",
#                   "seed"                :   15,
#                   "cuda_en"             :   True,
#                   "torch_deterministic" :   True,
#                   "num_steps"           :   1024,
#                   "num_minibatches"     :   256,
#                   "learning_rate"       :   7.77e-05,
#                   "activation_fn"       :   th.nn.Tanh,
#                   "anneal_lr"           :   False,
#                   "gamma"               :   0.9999,
#                   "gae_lambda"          :   0.9,
#                   "norm_adv"            :   False,          # select norm_adv or norm_adv_fullbatch only [select only one]
#                   "norm_adv_fullbatch"  :   True,
#                   "clip_coef"           :   0.1,
#                   "clip_vloss"          :   False,
#                   "ent_coef"            :   0.00429,
#                   "vf_coef"             :   0.19,
#                   "max_grad_norm"       :   3.0,
#                   "target_kl"           :   None,
#                   # PPG Specific args
#                   "n_iteration"         :   8,
#                   "e_policy"            :   10,
#                   "v_value"             :   1,
#                   "e_auxiliary"         :   5,
#                   "beta_clone"          :   1.0,            # behavior clonning coefficient
#                   "num_aux_rollouts"    :   4,              # number of minibatch in the auxiliary phase
#                   "n_aux_grad_accum"    :   1,              # number of gradient accumuation in mini batch
#                   }


# for training disable rendering
env = gym.make("Pendulum-v1",render_mode="human")
# env_ = RescaleAction(env, min_action=np.array([-1.25]*env.action_space.shape[0]), max_action=np.array([1.25]*env.action_space.shape[0]))
env__ = gym.vector.SyncVectorEnv([lambda: RecordEpisodeStatistics(env)])

# for eval enable rendering
# env = gym.vector.SyncVectorEnv([lambda: RecordEpisodeStatistics(gym.make("Pendulum-v1",render_mode="human"))])

# model = PPG(env,
#             params_cfg=params_PPG)
model = PPO(env__,
            params_cfg=params_PPO)

# model.train(total_timesteps=int(3e5))

model.eval_once(iter=150)

