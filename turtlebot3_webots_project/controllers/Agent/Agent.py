"""Agent controller."""
# File         : Agent.cpp
# Date         : 4 Sept 2024
# Description  : Agent RL Controller
# Author       :  DarkStealthX
# Modifications: - Communication using zeromq
#                - Publish start and step message. Subscribe state data from Robot

# TB3 Agent Core
from library.tb3_agent import logger, Agent
from library.tb3_agent_wrapper import *
from library.tb3_agent_callback import *
from library.custom_rl_algo import *
from library.BC_Net import modelNet

# Gymnasium Library module
import gymnasium as gym

import numpy as np

# RL Library
# from stable_baselines3 import PPO, SAC
from sb3_contrib import RecurrentPPO
from stable_baselines3.common.env_util import make_vec_env

from scipy import signal

def scale_value(value, input_min, input_max, output_min, output_max):
    # Scale the value from the input range to the output range
    scaled_value = ((value - input_min) / (input_max - input_min)) * (output_max - output_min) + output_min
    return scaled_value

def create_demonstration():
    agent_settings = {"goal_dist": 0.11, # in meter
                      "collision_dist": 0.12, # in meter
                      "lidar_for_state": 9,  # number of state lidar read (default=10)
                     }  
    
    scene_configuration = {"change_scene_every_goal_reach": 50, # change the scenario every n goal reach
                           "scene_start_from": 1, # scene start from n
                           "random_start": False, # whether start from x and y coordinate random or not
                           "max_steps": 1280,
                          }
    
    tracking_configuration = {"en_tracking": True,
                              "save_tracking":True,
                              "save_folder":"./expert_traj",
                              "save_name":"trajectory",
                             }
    
    roboJoystick = JoystickEnv("demonstration",
                               agent_settings,
                               scene_configuration,
                               tracking_configuration,
                               fixed_steps=True,
                               type_of_trajectory="dict")

    def init_filter():
        #initialize filter
        filter = signal.firwin(numtaps=15, cutoff=0.1)
        z_out_1 = signal.lfilter_zi(filter, 1) * 0.0   # initialize from -1.0
        z_out_2 = signal.lfilter_zi(filter, 1) * 0.0    # initialize from 0.0

        z_out_1 = np.asarray([-1.0 for i in range(len(z_out_1))])   # initialize linear velocity from low = 0.0 (after scale)
        z_out_2 *= 0.0                      # initialize lienar velocity from low = 0.0 (after scale)
        
        return filter, z_out_1, z_out_2

    filter, z_out_1, z_out_2 = init_filter()
    # print(f"{z_out_1}")
    # print(f"{z_out_2}")

    state, info = roboJoystick.reset(seed=15)
    while(roboJoystick.demo_stop==False):

        # on windows with this setup
        if(roboJoystick.joyController.model == "Android Gamepad"):
            forward = -roboJoystick.joyController.getAxisValue(3)    # joystick analog on right (default 2)
            turn = roboJoystick.joyController.getAxisValue(0)        # joystick analog on left (default 1)

        if(forward<0):
            forward = 0

        # check based distance
        if(state[-2]<=agent_settings['goal_dist']):
            roboJoystick.agent.simulationSetMode(2)
            forward = 0       # force to lowing linear vel
            turn = 0          # force to center of turn
        
        # print(forward)
        out_0 = scale_value(forward, 0, 32767, -1, 1)
        out_0 = np.clip(out_0, a_min=-1.000, a_max=1.000)

        out_1 = scale_value(turn, -32767, 32767, -1, 1)
        out_1 = np.clip(out_1, a_min=-1.000, a_max=1.000)

        out_0_filt, z_out_1 = signal.lfilter(filter, 1, [out_0], zi=z_out_1)
        out_1_filt, z_out_2 = signal.lfilter(filter, 1, [out_1], zi=z_out_2)

        out_0_filt = np.clip(out_0_filt, a_min=-1.000, a_max=1.000)
        out_1_filt = np.clip(out_1_filt, a_min=-1.000, a_max=1.000)

        z_out_1 = np.clip(z_out_1, a_min=-1.000, a_max=1.000)
        z_out_2 = np.clip(z_out_2, a_min=-1.000, a_max=1.000)
        # print(out_0_filt, out_1_filt)

        next_state, reward, done, trunc, info = roboJoystick.step([out_0_filt[0], out_1_filt[0]])
        
        state=next_state

        if(roboJoystick.idx+1>scene_configuration['max_steps']):
            # print("wes rampung gan...")
            roboJoystick.agent.simulationSetMode(1)
            state, info = roboJoystick.reset(seed=15)
            filter, z_out_1, z_out_2 = init_filter()
    logger("info", "Py", "end create demonstration")

def create_mapping_ros2():
    agent_settings = {"goal_dist": 0.11, # in meter
                      "collision_dist": 0.12, # in meter
                      "lidar_for_state": 21,  # number of state lidar read (default=10)
                      }  
    
    # because for mapping
    scene_configuration = {"change_scene_every_goal_reach": np.inf, # change the scenario every n goal reach
                           "scene_start_from": 0, # scene start from n
                           "random_start": False, # whether start from x and y coordinate random or not
                           "max_steps": int(99e6),    # set to maximum integer value
                          }
    
    tracking_configuration = {"en_tracking": False,
                              "save_tracking":False,
                              "save_folder":None,
                              "save_name":None,
                              }
    
    roboJoystick = JoystickEnv(agent_settings,
                               scene_configuration,
                               tracking_configuration,
                               fixed_steps=True,
                               type_of_trajectory="imitation_trajectory")

    #initialize filter
    filter = signal.firwin(numtaps=15, cutoff=0.1)
    z_out_1 = signal.lfilter_zi(filter, 1) * 0.0
    z_out_2 = signal.lfilter_zi(filter, 1) * 0.0

    print("start reset")
    state, info = roboJoystick.reset(seed=15)
    end = False
    # end while pressed button_6 (L-key) or button_7 (r-key)
    while(end!=6 and end!=7):

        # on windows with this setup
        if(roboJoystick.joyController.model == "Android Gamepad"):
            forward = -roboJoystick.joyController.getAxisValue(3)     # joystick analog on right (default 2)
            turn = roboJoystick.joyController.getAxisValue(0)         # joystick analog on left (default 1)
            end = roboJoystick.joyController.getPressedButton()

        if(forward<0):
            forward = 0

        out_0 = scale_value(forward, 0, 32767, -1, 1)
        out_1 = scale_value(turn, -32767, 32767, -1, 1)

        out_0_filt, z_out_1 = signal.lfilter(filter, 1, [out_0], zi=z_out_1)
        out_1_filt, z_out_2 = signal.lfilter(filter, 1, [out_1], zi=z_out_2)

        _, _, _, _, _ = roboJoystick.step([out_0_filt[0], out_1_filt[0]])

    logger("info", "Py", "end create demonstration")

def RL_Training():
    agent_settings = {"goal_dist": 0.11, # in meter
                      "collision_dist": 0.12, # in meter
                      "lidar_for_state": 9,  # number of state lidar read (default=10)
                      "gamma": 0.993,
                      }  
    
    scene_configuration = {"change_scene_every_goal_reach": 5, # change the scenario every n goal reach
                           "scene_start_from": 0, # scene start from n
                           "random_start": False, # whether start from x and y coordinate random or not
                           "max_steps": 1280,    # set to maximum integer value
                          }
    
    eval_configuration = {"eval":{"enable":True,            # do the evaluation and print the result
                                  "save_result":True},
                          "save":{"enable":True,            # do save the model every n-step
                                  "path":'./RL_Model',
                                  "name":'rl_ppo'},
                          "every":{"timestep":1e5}}
    
    roboAgent = Agent(agent_settings=agent_settings,
                      scene_configuration=scene_configuration,
                      fixed_steps=True)
    myCallback = Eval_and_Save(env=roboAgent,
                               eval_configuration=eval_configuration)
    
    # policy_kwargs = dict(activation_fn=th.nn.LeakyReLU,
    #                     net_arch=dict(pi=[64, 64], vf=[64, 64]))

    # model = PPO("MlpPolicy",
    #              roboAgent,
    #              seed=12,
    #              learning_rate=3e-4,
    #              n_steps=2560,
    #              batch_size=256,
    #              n_epochs=20,
    #              gae_lambda=0.95,
    #              max_grad_norm=3.0,
    #              clip_range=0.2,
    #              gamma=roboAgent.agent_settings.gamma,
    #             #  policy_kwargs=policy_kwargs,
    #              ent_coef=0.0075,)
    # model = SAC("MlpPolicy",
    #             roboAgent,
    #             seed=17,
    #             learning_starts=1500,
    #             batch_size=256,)
    # print(model.policy)
    # model.learn(total_timesteps=int(10e6), callback=myCallback)
    # model.save(os.path.join(myCallback.eval_cfg.save.path, 
    #                         myCallback.eval_cfg.save.name))
    
    # reset the environment
    roboAgent.agent.simulationReset()

def agent_checker():
    agent_settings = {"goal_dist": 0.11, # in meter
                      "collision_dist": 0.12, # in meter
                      "lidar_for_state": 21,  # number of state lidar read (default=10)
                      "gamma": 0.99,
                      }  
    
    scene_configuration = {"change_scene_every_goal_reach": 10, # change the scenario every n goal reach
                           "scene_start_from": 0, # scene start from n
                           "random_start": False, # whether start from x and y coordinate random or not
                           "max_steps": 1024,    # set to maximum integer value
                          }

    roboAgent = Agent(agent_settings,
                      scene_configuration,
                      verbose=True)
                      
    observation, info = roboAgent.reset()
    # print("reset suceed")
    for _ in range(100000):
        # action = roboAgent.action_space.sample()  # this is where you would insert your policy
        action = [-1.000, 0.25]
        # logger("custom", "PY_OUT", f"action : {action[0]},{action[1]}")
        observation, reward, terminated, truncated, info = roboAgent.step(action)
        # print("step suceed")
        if terminated or truncated: 
            observation, info = roboAgent.reset()
            
def customRLProgram():
    agent_settings = {"goal_dist": 0.11, # in meter
                      "collision_dist": 0.12, # in meter
                      "lidar_for_state": 9,  # number of state lidar read (default=10)
                      "gamma": 0.993,
                      }  
    
    scene_configuration = {"change_scene_every_goal_reach": 5, # change the scenario every n goal reach
                           "scene_start_from": 0, # scene start from n
                           "random_start": False, # whether start from x and y coordinate random or not
                           "max_steps": 1280,    # set to maximum integer value
                          }
    
    eval_configuration = {"eval":{"enable":True,            # do the evaluation and print the result
                                  "save_result":True},
                          "save":{"enable":True,            # do save the model every n-step
                                  "path":'./RL_Model',
                                  "name":'rl_ppo'},
                          "every":{"timestep":1e5}}
    
    # params_cfg = {"exp_name"            :   "RL_PPO_Gaussian",
    #               "save_every_reset"    :   False,           # choose one save every reset or save every n step
    #               "save_every_step"     :   True,
    #               "save_step"           :   1e5,             # optional if save every n step or reset true otherwise you can uncomment
    #               "save_path"           :   "./models/RL_PPO",
    #               "seed"                :   12,
    #               "cuda_en"             :   True,
    #               "torch_deterministic" :   True,
    #               "num_steps"           :   2560,
    #               "num_minibatches"     :   256,
    #               "num_epoch"           :   10,
    #               "learning_rate"       :   3e-4,
    #               "activation_fn"       :   th.nn.ReLU,
    #               "anneal_lr"           :   False,
    #               "gamma"               :   agent_settings["gamma"],
    #               "gae_lambda"          :   0.95,
    #               "norm_adv"            :   True,
    #               "clip_coef"           :   0.2,
    #               "clip_vloss"          :   False,
    #               "ent_coef"            :   0.005,
    #               "vf_coef"             :   0.5,
    #               "max_grad_norm"       :   1.0,
    #               "target_kl"           :   None,
    #               "rpo_alpha"           :   0.01,
    #               "distributions"       :   "Normal",
    #               "use_tanh_output"     :   False,
    #               "use_icm"             :   False,
    #               }

    roboAgent = gym.vector.SyncVectorEnv([lambda: Agent( name_exp="RL_SAC",
                                                         agent_settings=agent_settings,
                                                         scene_configuration=scene_configuration,
                                                         fixed_steps=True,
                                                         logging_reward=True)])
    T = scene_configuration["max_steps"]
    # try heuristic like on paper https://arxiv.org/pdf/2310.16828.pdf
    gamma = np.round(np.clip(((T/5)-1)/(T/5), a_min=0.950, a_max=0.995), decimals=3)
    params_SAC = {"exp_name"          :   "RL_SAC",
                "save_every_reset"    :   True,           # choose one save every reset or save every n step
                "save_every_step"     :   False,
                "save_step"           :   10,             # optional if save every n step or reset true otherwise you can uncomment
                "save_path"           :   "./models/RL_SAC",
                "seed"                :   1,
                "cuda_en"             :   True,
                "torch_deterministic" :   True,
                "buffer_size"         :   int(1e6),
                "gamma"               :   gamma,          
                "tau"                 :   0.005,
                "batch_size"          :   256,
                "learning_starts"     :   T*2,
                "policy_num_blocks"   :   1,
                "critic_num_blocks"   :   2,
                "policy_hidden_size"  :   256,
                "critic_hidden_size"  :   256,
                "q_lr"                :   1e-3,
                "policy_lr"           :   3e-4,
                "policy_frequency"    :   2,
                "target_network_frequency" : 1,
                "ent_coef"            :   "auto",           # autotune alpha entropy coefficient, leave true for default, if false set alpha value
                "use_rsnorm"          :   True,
                }
    # model = PPO(env=roboAgent,
    #             params_cfg=params_cfg)
    model = SAC(env=roboAgent,
                params_cfg=params_SAC)

    model.train(total_timesteps=int(256e4))
    
    # # reset the environment
    # roboAgent.env_fns[0]().agent.simulationReset()

def refining_demonstration():

    agent_settings = {"goal_dist": 0.11, # in meter
                      "collision_dist": 0.12, # in meter
                      "lidar_for_state": 9,  # number of state lidar read (default=10)
                      "gamma": 0.993,
                      }  
    
    scene_configuration = {"change_scene_every_goal_reach": 5, # change the scenario every n goal reach
                           "scene_start_from": 0, # scene start from n
                           "random_start": False, # whether start from x and y coordinate random or not
                           "max_steps": 1280,    # set to maximum integer value
                          }
    
    eval_configuration = {"eval":{"enable":True,            # do the evaluation and print the result
                                  "save_result":True},
                          "save":{"enable":True,            # do save the model every n-step
                                  "path":'./RL_Model',
                                  "name":'rl_ppo'},
                          "every":{"timestep":1e5}}
    
    roboAgent = Agent(name_exp="refine_demo_checker",
                     agent_settings=agent_settings,
                     scene_configuration=scene_configuration,
                     fixed_steps=True,
                     logging_reward=True)
    
    device = "cuda" if th.cuda.is_available() else "cpu"

    model = modelNet(observation_shape=roboAgent.observation_space.shape[0],
                     action_shape=roboAgent.action_space.shape[0]).to(device)

    model.load_model("./", "bc_model")

    model.eval()
    obs, _ = roboAgent.reset()
    
    isExit = False
    global_step = 0
    while(isExit==False):
        predict_action = model(torch.from_numpy(obs).type(torch.float32).to(device))
        act = predict_action.cpu().detach().numpy()
        print(f"{act}")
        next_obs, rew, term, trunc, info = roboAgent.step(act)
        
        obs = next_obs
        global_step += 1  

def check_demonstration(min_scene:Optional[int]=0,
                        max_scene:Optional[int]=1):
    agent_settings = {"goal_dist": 0.11, # in meter
                      "collision_dist": 0.12, # in meter
                      "lidar_for_state": 9,  # number of state lidar read (default=10)
                      "gamma": 0.993,
                      }  
    
    scene_configuration = {"change_scene_every_goal_reach": 50, # change the scenario every n goal reach
                           "scene_start_from": min_scene, # scene start from n
                           "random_start": False, # whether start from x and y coordinate random or not
                           "max_steps": 1280,    # set to maximum integer value
                          }
    
    roboAgent = Agent(name_exp="refine_demo_checker",
                     agent_settings=agent_settings,
                     scene_configuration=scene_configuration,
                     fixed_steps=True,
                     logging_reward=True)

    expert_path = "./expert_traj"
    expert_id = 0
    expert_ep = 0
    traj = np.load(os.path.join(expert_path,f"trajectory_id_{expert_id}.npy"), 
                   allow_pickle = True)

    obs, _ = roboAgent.reset()
    
    isExit = False
    global_step = 0
    idx = 0
    while(isExit==False):
        next_obs, rew, term, trunc, info = roboAgent.step(traj[expert_ep]['acts'][idx])
        
        idx += 1
        if(term or trunc):
            idx = 0
            expert_ep += 1
            if expert_ep>=scene_configuration["change_scene_every_goal_reach"]:
                expert_ep = 0
                expert_id += 1 
                if (expert_id >= max_scene):
                    isExit = True
            
            obs, _ = roboAgent.reset()
                

if __name__ == "__main__":
    
    # for mapping purpose using joystick
    # create_mapping_ros2()

    # for create demonstration using joystick
    # create_demonstration()
    # refining_demonstration()
    # check_demonstration()

    # agent checker
    # agent_checker()

    # RL Training using Advanced Callback to track SR (Success Rate)
    # RL_Training() #not used again please use CustomRLPrograms
    
    customRLProgram()


        
    
