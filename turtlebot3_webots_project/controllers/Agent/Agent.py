"""Agent controller."""

# File         : Agent.cpp
# Date         : 4 Sept 2024
# Description  : Agent RL Controller
# Author       :  DarkStealthX
# Modifications: - Communication using zeromq
#                - Publish start and step message. Subscribe state data from Robot
import os
import sys
from typing import Any

# Gymnasium Library module
import gymnasium as gym
import numpy as np
import scipy.signal as signal
import torch as th
from library.custom_rl_algo import PPO, SAC

# from library.BC_Net import modelNet
# TB3 Agent Core
from library.Imitation.AIRL import AIRL
from library.tb3_agent import Agent, logger
from library.tb3_agent_callback import Eval_and_Save
from library.tb3_agent_wrapper import JoystickEnv, TB3_Agent_Demo
from numpy.typing import NDArray

# RL Library
from stable_baselines3 import PPO as SB3_PPO
from stable_baselines3 import SAC as SB3_SAC


def scale_value(
    value: float | int,
    input_min: float | int,
    input_max: float | int,
    output_min: float | int,
    output_max: float | int,
) -> float:
    # Scale the value from the input range to the output range
    scaled_value = ((value - input_min) / (input_max - input_min)) * (
        output_max - output_min
    ) + output_min
    return scaled_value


def create_demonstration():
    agent_settings = {
        "goal_dist": 0.11,  # in meter
        "collision_dist": 0.12,  # in meter
        "lidar_for_state": 9,  # number of state lidar read (default=10)
    }

    scene_configuration = {
        "change_scene_every_goal_reach": 50,  # change the scenario every n goal reach
        "scene_start_from": 4,  # scene start from n
        "random_start": False,  # whether start from x and y coordinate random or not
        "max_steps": 1280,
    }

    tracking_configuration = {
        "en_tracking": True,
        "save_tracking": True,
        "save_folder": "./expert_traj",
        "save_name": "trajectory",
    }

    roboJoystick = JoystickEnv(
        "demonstration",
        agent_settings,
        scene_configuration,
        tracking_configuration,
        fixed_steps=True,
        type_of_trajectory="dict",
    )

    def init_filter() -> tuple[NDArray[Any], NDArray[Any], NDArray[Any]]:
        # initialize filter
        fir_filter = signal.firwin(numtaps=15, cutoff=0.1)
        z_out_1 = signal.lfilter_zi(fir_filter, 1) * 0.0  # initialize from -1.0
        z_out_2 = signal.lfilter_zi(fir_filter, 1) * 0.0  # initialize from 0.0

        z_out_1 = np.asarray(
            [-1.0 for i in range(len(z_out_1))]
        )  # initialize linear velocity from low = 0.0 (after scale)
        z_out_2 *= 0.0  # initialize lienar velocity from low = 0.0 (after scale)

        return fir_filter, z_out_1, z_out_2

    filter, z_out_1, z_out_2 = init_filter()
    # print(f"{z_out_1}")
    # print(f"{z_out_2}")
    state, info = roboJoystick.reset(seed=15)
    while not roboJoystick.scenario_reach_end:
        roboJoystick.demo_stop = False
        # state, info = roboJoystick.reset(seed=15)
        while not roboJoystick.demo_stop:
            # on windows with this setup
            if roboJoystick.joyController.model == "Android Gamepad":
                forward = -roboJoystick.joyController.getAxisValue(
                    3
                )  # joystick analog on right (default 2)
                turn = roboJoystick.joyController.getAxisValue(
                    0
                )  # joystick analog on left (default 1)

            if forward < 0:
                forward = 0

            # check based distance
            if state[-2] <= agent_settings["goal_dist"]:
                roboJoystick.agent.simulationSetMode(2)
                forward = 0  # force to lowing linear vel
                turn = 0  # force to center of turn

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

            next_state, reward, done, trunc, info = roboJoystick.step(
                [out_0_filt[0], out_1_filt[0]]
            )

            state = next_state

            if roboJoystick.idx + 1 > scene_configuration["max_steps"]:
                # print("fast mode simulation until end steps!...")
                roboJoystick.agent.simulationSetMode(1)
                state, info = roboJoystick.reset(seed=15)
                filter, z_out_1, z_out_2 = init_filter()
        logger("info", "Py", f"end create demonstration for scenario:{roboJoystick.scenario_idx}")


def create_mapping_ros2():
    agent_settings = {
        "goal_dist": 0.11,  # in meter
        "collision_dist": 0.12,  # in meter
        "lidar_for_state": 21,  # number of state lidar read (default=10)
    }

    # because for mapping
    scene_configuration = {
        "change_scene_every_goal_reach": np.inf,  # change the scenario every n goal reach
        "scene_start_from": 0,  # scene start from n
        "random_start": False,  # whether start from x and y coordinate random or not
        "max_steps": int(99e6),  # set to maximum integer value
    }

    tracking_configuration = {
        "en_tracking": False,
        "save_tracking": False,
        "save_folder": None,
        "save_name": None,
    }

    roboJoystick = JoystickEnv(
        "demonstration",
        agent_settings,
        scene_configuration,
        tracking_configuration,
        fixed_steps=True,
        type_of_trajectory="imitation_trajectory",
    )

    # initialize filter
    filter = signal.firwin(numtaps=15, cutoff=0.1)
    z_out_1 = signal.lfilter_zi(filter, 1) * 0.0
    z_out_2 = signal.lfilter_zi(filter, 1) * 0.0

    print("start reset")
    state, info = roboJoystick.reset(seed=15)
    end = False
    # end while pressed button_6 (L-key) or button_7 (r-key)
    while end != 6 and end != 7:
        # on windows with this setup
        if roboJoystick.joyController.model == "Android Gamepad":
            forward = -roboJoystick.joyController.getAxisValue(
                3
            )  # joystick analog on right (default 2)
            turn = roboJoystick.joyController.getAxisValue(
                0
            )  # joystick analog on left (default 1)
            end = roboJoystick.joyController.getPressedButton()

        if forward < 0:
            forward = 0

        out_0 = scale_value(forward, 0, 32767, -1, 1)
        out_1 = scale_value(turn, -32767, 32767, -1, 1)

        out_0_filt, z_out_1 = signal.lfilter(filter, 1, [out_0], zi=z_out_1)
        out_1_filt, z_out_2 = signal.lfilter(filter, 1, [out_1], zi=z_out_2)

        _, _, _, _, _ = roboJoystick.step([out_0_filt[0], out_1_filt[0]])

    logger("info", "Py", "end create demonstration")


def RL_Training(algo_mode: str = "PPO"):
    agent_settings = {
        "goal_dist": 0.11,  # in meter
        "collision_dist": 0.12,  # in meter
        "lidar_for_state": 9,  # number of state lidar read (default=10)
        "gamma": 0.993,
    }

    scene_configuration = {
        "change_scene_every_goal_reach": 5,  # change the scenario every n goal reach
        "scene_start_from": 0,  # scene start from n
        "random_start": False,  # whether start from x and y coordinate random or not
        "max_steps": 1280,  # set to maximum integer value
    }

    eval_configuration = {
        "eval": {
            "enable": True,  # do the evaluation and print the result
            "save_result": True,
        },
        "save": {
            "enable": True,  # do save the model every n-step
            "path": "./RL_Model",
            "name": "rl_ppo",
        },
        "every": {"timestep": 1e5},
    }

    roboAgent = Agent(
        name_exp="sb3_rl_training",
        agent_settings=agent_settings,
        scene_configuration=scene_configuration,
        fixed_steps=True,
    )
    myCallback = Eval_and_Save(env=roboAgent, eval_configuration=eval_configuration)

    if "PPO" in algo_mode:
        policy_kwargs = dict(
            activation_fn=th.nn.LeakyReLU, net_arch=dict(pi=[64, 64], vf=[64, 64])
        )

        model = SB3_PPO(
            "MlpPolicy",
            roboAgent,
            seed=12,
            learning_rate=3e-4,
            n_steps=2560,
            batch_size=256,
            n_epochs=20,
            gae_lambda=0.95,
            max_grad_norm=3.0,
            clip_range=0.2,
            gamma=roboAgent.agent_settings.gamma,
            policy_kwargs=policy_kwargs,
            ent_coef=0.0075,
        )
    elif "SAC" in algo_mode:
        model = SB3_SAC(
            "MlpPolicy",
            roboAgent,
            seed=17,
            learning_starts=1500,
            batch_size=256,
        )

    print(model.policy)
    model.learn(total_timesteps=int(10e6), callback=myCallback)
    model.save(
        os.path.join(myCallback.eval_cfg.save.path, myCallback.eval_cfg.save.name)
    )

    # reset the environment
    roboAgent.agent.simulationReset()


def agent_checker():
    agent_settings = {
        "goal_dist": 0.11,  # in meter
        "collision_dist": 0.12,  # in meter
        "lidar_for_state": 21,  # number of state lidar read (default=10)
        "gamma": 0.99,
    }

    scene_configuration = {
        "change_scene_every_goal_reach": 10,  # change the scenario every n goal reach
        "scene_start_from": 0,  # scene start from n
        "random_start": False,  # whether start from x and y coordinate random or not
        "max_steps": 1024,  # set to maximum integer value
    }

    roboAgent = Agent(
        name_exp="agent_checker",
        agent_settings=agent_settings,
        scene_configuration=scene_configuration,
        verbose=True,
    )

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


def customRLProgram(
    algo: str,
    exp_name: str,
):
    agent_settings = {
        "goal_dist": 0.11,  # in meter
        "collision_dist": 0.12,  # in meter
        "lidar_for_state": 9,  # number of state lidar read (default=10)
        "gamma": 0.993,
    }

    scene_configuration = {
        "change_scene_every_goal_reach": 5,  # change the scenario every n goal reach
        "scene_start_from": 0,  # scene start from n
        "random_start": False,  # whether start from x and y coordinate random or not
        "max_steps": 1280,  # set to maximum integer value
    }

    roboAgent = gym.vector.SyncVectorEnv(
        [
            lambda: Agent(
                name_exp=exp_name,
                agent_settings=agent_settings,
                scene_configuration=scene_configuration,
                fixed_steps=True,
                logging_reward=True,
            )
        ]
    )

    if algo == "PPO":
        model = PPO(env=roboAgent, config_path="./config", config_name=exp_name, bypass_class_cfg=False)
        model.train(total_timesteps=int(256e4))
    elif algo == "SAC":
        model = SAC(env=roboAgent, config_path="./config", config_name=exp_name)
        model.train(total_timesteps=int(256e4))
    else:
        sys.exit("Algorithm not found")


def CustomAIRLProgram(exp_name: str):
    agent_settings = {
        "goal_dist": 0.11,  # in meter
        "collision_dist": 0.12,  # in meter
        "lidar_for_state": 9,  # number of state lidar read (default=10)
        "gamma": 0.993,
    }

    scene_configuration = {
        "change_scene_every_goal_reach": 5,  # change the scenario every n goal reach
        "scene_start_from": 0,  # scene start from n
        "random_start": False,  # whether start from x and y coordinate random or not
        "max_steps": 1280,  # set to maximum integer value
    }

    roboAgent = gym.vector.SyncVectorEnv(
        [
            lambda: Agent(name_exp=exp_name,
                                   agent_settings=agent_settings,
                                   scene_configuration=scene_configuration,
                                   fixed_steps=True,
                                   logging_reward=True,
            )
        ]
    )

    model = AIRL(env=roboAgent, 
                 config_path="config", 
                 config_name=exp_name,
                 expert_path="./expert_traj")
    model.train(num_steps=int(256e4))


def check_demonstration(min_scene: int = 0, max_scene: int = 1):
    agent_settings = {
        "goal_dist": 0.11,  # in meter
        "collision_dist": 0.12,  # in meter
        "lidar_for_state": 9,  # number of state lidar read (default=10)
        "gamma": 0.993,
    }

    scene_configuration = {
        "change_scene_every_goal_reach": 50,  # change the scenario every n goal reach
        "scene_start_from": min_scene,  # scene start from n
        "random_start": False,  # whether start from x and y coordinate random or not
        "max_steps": 1280,  # set to maximum integer value
    }

    roboAgent = Agent(
        name_exp="refine_demo_checker",
        agent_settings=agent_settings,
        scene_configuration=scene_configuration,
        fixed_steps=True,
        logging_reward=True,
    )

    expert_path = "./expert_traj"
    expert_id = 0
    expert_ep = 0
    traj = np.load(
        os.path.join(expert_path, f"trajectory_id_{expert_id}.npy"), allow_pickle=True
    )

    obs, _ = roboAgent.reset()

    isExit = False
    idx = 0
    while not isExit:
        next_obs, rew, term, trunc, info = roboAgent.step(traj[expert_ep]["acts"][idx])

        idx += 1
        if term or trunc:
            idx = 0
            expert_ep += 1
            if expert_ep >= scene_configuration["change_scene_every_goal_reach"]:
                expert_ep = 0
                expert_id += 1
                if expert_id >= max_scene:
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

    # Custom RL Programs using PPO or SAC
    # using exp_name matched the configuration on config folder
    # customRLProgram(algo="SAC",
    #                 exp_name="rl_sac")
    customRLProgram(algo="PPO", exp_name="rl_ppo_gaussian")
    # customRLProgram(algo="PPO",
    #                 exp_name="rl_ppo_clippedgaussian")

    # Here we go, Custom IRL Program (AIRL)
    # CustomAIRLProgram(exp_name="airl")
