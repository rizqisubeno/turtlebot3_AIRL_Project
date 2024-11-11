from library.custom_rl_algo import PPO, SAC

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

params_SAC = {"exp_name"            :   "RL_SAC",
              "save_every_reset"    :   True,           # choose one save every reset or save every n step
              "save_every_step"     :   False,
              "save_step"           :   10,             # optional if save every n step or reset true otherwise you can uncomment
              "save_path"           :   "./models/RL_SAC",
              "seed"                :   1,
              "cuda_en"             :   True,
              "torch_deterministic" :   True,
              "buffer_size"         :   int(1e6),
              "gamma"               :   0.99,
              "tau"                 :   0.005,
              "batch_size"          :   256,
              "learning_starts"     :   400,
              "q_lr"                :   1e-3,
              "policy_lr"           :   3e-4,
              "policy_frequency"    :   2,
              "target_network_frequency" : 1,
              "autotune"            :   True,           # autotune alpha entropy, leave true for default, if false set alpha value
              "alpha"               :   0.2}


# for training disable rendering
env = gym.make("Pendulum-v1",render_mode="human")
env_ = RescaleAction(env, min_action=np.array([-2.00]*env.action_space.shape[0]), max_action=np.array([2.00]*env.action_space.shape[0]))
env__ = gym.vector.SyncVectorEnv([lambda: RecordEpisodeStatistics(env_)])

# for eval enable rendering
# env = gym.vector.SyncVectorEnv([lambda: RecordEpisodeStatistics(gym.make("Pendulum-v1",render_mode="human"))])

# model = PPG(env,
#             params_cfg=params_PPG)
model = SAC(env__,
            params_cfg=params_SAC)

# model.train(total_timesteps=int(3e5))

# model.eval_once(iter=150)
model.eval_once(iter=65)

