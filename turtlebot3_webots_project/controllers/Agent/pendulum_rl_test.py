from library.custom_rl_algo import PPO, SAC

import torch as th
import numpy as np

from stable_baselines3.sac import SAC as SAC_SB3

import gymnasium as gym
from gymnasium.wrappers.record_episode_statistics import RecordEpisodeStatistics
from gymnasium.wrappers.rescale_action import RescaleAction

training = True

# env = gym.make_vec('MountainCarContinuous-v0', num_envs=1, vectorization_mode="sync")


# for training disable rendering
env = gym.make("Pendulum-v1",
               render_mode="human" if not training else "rgb_array",
               seed=1)
env_ = RescaleAction(env, min_action=np.array([-2.00]*env.action_space.shape[0]), max_action=np.array([2.00]*env.action_space.shape[0]))

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

T = env.spec.max_episode_steps
# try heuristic like on paper https://arxiv.org/pdf/2310.16828.pdf
gamma = np.round(np.clip(((T/5)-1)/(T/5), a_min=0.950, a_max=0.995), decimals=3)
params_SAC = {"exp_name"            :   "RL_SAC_MountainCarContinuous",
              "save_every_reset"    :   True,           # choose one save every reset or save every n step
              "save_every_step"     :   False,
              "save_step"           :   10,             # optional if save every n step or reset true otherwise you can uncomment
              "save_path"           :   "./models/RL_SAC_MountainCarContinuous",
              "seed"                :   1,
              "cuda_en"             :   True,
              "torch_deterministic" :   True,
              "buffer_size"         :   int(1e6),
              "gamma"               :   gamma,          
              "tau"                 :   0.005,
              "batch_size"          :   256,
              "learning_starts"     :   T*2,
              "actor_hidden_size"   :   (256, 256),
              "critic_hidden_size"  :   (256, 256),
              "actor_activation"    :   th.nn.ReLU,
              "critic_activation"   :   th.nn.ReLU,
              "q_lr"                :   1e-3,
              "policy_lr"           :   3e-4,
              "policy_frequency"    :   2,
              "target_network_frequency" : 1,
              "ent_coef"            :   "auto",           # autotune alpha entropy coefficient, leave true for default, if false set alpha value
             }

env__ = gym.vector.SyncVectorEnv([lambda: RecordEpisodeStatistics(env_)])

# for eval enable rendering
# env = gym.vector.SyncVectorEnv([lambda: RecordEpisodeStatistics(gym.make("Pendulum-v1",render_mode="human"))])

# model = PPG(env,
#             params_cfg=params_PPG)
model = SAC(env__,
            params_cfg=params_SAC)

# model = SAC_SB3('MlpPolicy', 
#                 env__, 
#                 verbose=1, 
#                 learning_rate=3e-4, 
#                 buffer_size=int(1e6), 
#                 learning_starts=5000, 
#                 batch_size=256, 
#                 gamma=0.995, 
#                 tau=0.005,  
#                 ent_coef=0.1, 
#                 target_update_interval=1,
#                 gradient_steps=1,
#                 tensorboard_log="./runs/RL_SAC_SB3_MountainCarContinuous")
# model.learn(total_timesteps=int(3e5))

# model.train(total_timesteps=int(3e5))
if(training):
    model.train(total_timesteps=int(3e5))
else:
    model.eval_once(iter=150)

