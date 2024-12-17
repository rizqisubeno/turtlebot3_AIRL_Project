"""Turtlebot3 Agent controller."""
# File         : tb3_agent.cpp
# Date         : 14 Sept 2024
# Description  : Agent RL Controller
# Author       :  DarkStealthX
# Modifications: - Communication using zeromq
#                - Publish start and step message. Subscribe state data from Robot

# You may need to import some classes of the controller module. Ex:
#  from controller import Robot, Motor, DistanceSensor
import sys
import time
from types import SimpleNamespace
from typing import Optional

# Gymnasium Library module
import gymnasium as gym
import numpy as np
from controller.ansi_codes import AnsiCodes
from controller.supervisor import Supervisor
from library.Agent_Publisher import AgentPublisher
from library.message_vec_pb2 import start_agent_msg, step_agent_msg

# protobuf and zmq library
from library.Robo_Subscriber import RobotSubscriber
from torch.utils.tensorboard.writer import SummaryWriter

# scenario list
from scenario_list import list_robot_scenario, max_robot_scenario


# for prettier logging on terminal
def logger(mode: str = "", head_name: str = "", str_name: str = ""):
    head = ""
    if "info" in mode:
        head = AnsiCodes.BLUE_BACKGROUND
    elif "warn" in mode:
        head = AnsiCodes.YELLOW_BACKGROUND
    elif "err" in mode:
        head = AnsiCodes.RED_BACKGROUND
    elif "custom" in mode:
        head = AnsiCodes.MAGENTA_BACKGROUND
    elif "hidden" in mode:
        head = AnsiCodes.GREEN_BACKGROUND
    elif "none" in mode:
        head = ""
    else:
        head = AnsiCodes.RESET

    if head_name is None:
        print(head + f"{str_name}" + AnsiCodes.RESET)
    else:
        # tab_times = 2-(int((len(head_name))/9))
        tab_times = 2 if len(head_name) < 10 else 1
        print(
            head
            + str(f"[{head_name}]")
            + AnsiCodes.RESET
            + str("\t" * tab_times)
            + str_name
        )


class Agent(gym.Env):
    def __init__(
        self,
        name_exp,
        agent_settings,
        scene_configuration,
        fixed_steps: bool = True,
        logging_reward: bool = False,
        verbose: bool = False,
    ):
        self.seed = None
        self.fixed_steps = fixed_steps
        self.logging_reward = logging_reward
        if self.logging_reward:
            self.reward_list = []

        head_name = "TB3Py_Init"
        self.agent = Supervisor()
        self.timestep = int(self.agent.getBasicTimeStep())
        logger("info", head_name, f"webots timestep: {self.timestep}")

        # read agent settings for supporting algorithm list
        self.agent_settings = self.read_agent_settings(agent_settings, verbose=verbose)

        # read scene configuration
        self.scene_cfg = self.read_scene_configuration(
            scene_configuration, verbose=verbose
        )

        # get info node
        self.agent_node, self.target_node, self.info_node = self.read_webots_node(
            verbose=verbose
        )

        # get info node
        self.world_title = self.info_node.getField("title").getSFString()
        logger("info", "title_init", f"world title : {self.world_title}")

        # gym environment action setup (action space) [linear vel, angular vel]
        action_low = np.array([-1.0] * 2, dtype=np.float32)
        action_high = np.array([1.0] * 2, dtype=np.float32)
        self.action_space = gym.spaces.Box(low=action_low, high=action_high)

        # gym environment state setup (observation space) [lidar_state*10, d_goal, theta_goal] (only 12 state)
        observation_low = np.array(
            [0.12] * self.agent_settings.lidar_for_state + [0.0, -np.pi],
            dtype=np.float32,
        )
        observation_high = np.array(
            [10.0] * self.agent_settings.lidar_for_state + [10.0, np.pi],
            dtype=np.float32,
        )
        self.observation_space = gym.spaces.Box(
            low=observation_low, high=observation_high
        )

        # initialization zmq protocol
        self.robo_subscriber = RobotSubscriber()
        self.agent_publisher = AgentPublisher()
        self.start_idx = 0
        self.step_idx = 0
        self.idx = 0  # idx counter until maxsteps then reset
        self.scenario_idx = self.scene_cfg.scene_start_from
        self.scenario_reach_end = False
        self.max_scenario = max_robot_scenario
        self.success_counter = 0
        self.done = False
        self.eval = False
        # self.state = None
        self.turn_off_act = False
        # while ("start" not in self.agent_publisher.subscriptions or
        #     "step" not in self.agent_publisher.subscriptions):
        #     self.agent_publisher.receiveSubscriptions()
        # wait c++ controller to initialize please...
        # time.sleep(1.0)

        # needed step once before in c++ controller stepping while loop
        # for initilize joystick or not
        # if joystick initialize need first_step false because in their wrapper need stepping webots
        # if not use joystick, initialize this first_step to true because original c++ controller using first step once
        self.first_step = True

        # checker (must-used)
        self.toggle_stable_end_checker = False
        self.toggle_first_print = True
        self.collision_count = 0
        self.sequential_count = 0
        self.toggle_success_counter = True

        self.writer = SummaryWriter(f"runs/{name_exp}")
        self.writer.add_text(
            "Agent_Params",
            "|param|value|\n|-|-|\n%s"
            % (
                "\n".join(
                    [
                        f"|{key}|{value}|"
                        for key, value in vars(self.agent_settings).items()
                    ]
                )
            ),
        )

    def read_webots_node(self, verbose=False):
        head_name = "TB3Py_read_node"
        info_node = self.agent.getFromDef("WorldInfo")
        if info_node is None:
            logger(
                "error",
                head_name,
                "No WorldInfo node found in the current world file\n",
            )
            sys.exit(1)
        else:
            logger("info", head_name, "WorldInfo node def found")
        agent_node = self.agent.getFromDef("Agent")  # def name is Agent
        if agent_node is None:
            logger(
                "error",
                head_name,
                "No DEF Agent node found in the current world file\n",
            )
            sys.exit(1)
        else:
            logger("info", head_name, "Box Agent node def found")
        # initialize target ball node
        target_node = self.agent.getFromDef("target_ball")  # DEF name is target_ball
        if target_node is None:
            logger(
                "error",
                head_name,
                "No target ball node found in the current world file\n",
            )
            sys.exit(1)
        else:
            logger("info", head_name, "target ball node def found")

        return agent_node, target_node, info_node

    def read_agent_settings(self, agent_settings, verbose):
        head_name = "TB3Py_agent_cfg"
        settings_namespace = SimpleNamespace(**agent_settings)
        if verbose:
            for key, val in agent_settings.items():
                logger("info", head_name, f"{key} :\t{val}")
        return settings_namespace

    def read_scene_configuration(self, scene_configuration, verbose):
        head_name = "TB3Py_scene_cfg"
        print(f"{scene_configuration=}")
        scene_config_namespace = SimpleNamespace(**scene_configuration)
        if verbose:
            for key, val in scene_configuration.items():
                logger("info", head_name, f"{key}: \t{val}")
        return scene_config_namespace

    def observe_collision(self, lidar_data):
        min_laser = np.min(lidar_data)
        # print(f"min {min_laser:.3f} on idx : {np.argmin(lidar_data)}")
        if min_laser <= self.agent_settings.collision_dist:
            return True, min_laser
        return False, min_laser

    def calculate_reward(self, distanceToTarget, min_lidar, action):
        info = {}
        c1 = 1.75
        c2 = 0.45
        d_sm = 0.14  # 0.15meter mininum distance to obstacle from robot
        distance = c1 * distanceToTarget
        r_safety = c2 * (min_lidar - d_sm)
        # if reach target
        if distanceToTarget <= self.agent_settings.goal_dist:
            info["target"] = True
            return 10, info
        # if collision obstacle
        elif min_lidar <= self.agent_settings.collision_dist:
            info["collision"] = True
            # if collision near the target give least penalty than collision far from target
            return -10, info
        # if else
        else:
            info["target"] = False
            return (-distance) + (r_safety) + ((action[0] + 1.0) / 3.0) + (
                -abs(action[1]) / 3.0
            ), info
            # forward bonus when acting from low to medium high (0.15 scale) give bonus, if higher than 0.15 no bonus
            # turn penalty, if turn higher than 0.70 give penalty else no penalty
            # return (-distance) + (r_safety) + ((action[0]+1.0) if action[0]<0.15 else (0.00) / 4.0) + (-abs(action[1] if action[1]>=0.70 else 0.0) / 4.0), info

    def step(self, action):
        head_name = "Agent_Step"
        fixed_steps = self.fixed_steps
        if self.first_step:
            self.first_step = False
            self.agent.step(self.timestep)
            time.sleep(0.1)
            # print("ready......")
        while (
            "start" not in self.agent_publisher.subscriptions
            or "step" not in self.agent_publisher.subscriptions
        ):
            self.agent_publisher.receiveSubscriptions()
        if self.turn_off_act:
            # logger("warning", "Py", "act zero set")
            action = [
                -1.000,
                0.000,
            ]  # -1.000 for minimum linear vel & 0.000 for not turn

        step_msg = step_agent_msg()
        step_msg.idx = self.step_idx
        step_msg.act_0 = action[0]
        step_msg.act_1 = action[1]

        self.agent_publisher.step_msg = step_msg
        self.agent_publisher.run(mode="step")
        if self.agent_publisher.is_got_reply:
            self.step_idx += 1
            self.idx += 1
            self.agent_publisher.is_got_reply = False
        # logger("info", "Py", "agent publisher got reply on step")
        # stepping in this area
        self.agent.step()
        if not self.robo_subscriber.get_msg():
            # logger("info", "Py", "robo subscriber fail get msg")
            return None, [0], True, True, {}

        state = self.robo_subscriber.states.lidar_data
        # print(f"state1={state}")
        state = np.asarray(state)
        # print(f"state2={state}")
        state = np.append(
            state,
            np.array(
                [
                    self.robo_subscriber.states.distance_length,
                    self.robo_subscriber.states.angular_length,
                ]
            ),
        )
        # print(f"{state=}")
        collision, min_laser = self.observe_collision(
            state[: self.agent_settings.lidar_for_state]
        )
        # if(collision):
        #     logger("warning", "Py", f"{collision=}")
        reward, info = self.calculate_reward(
            state[self.agent_settings.lidar_for_state], min_laser, action
        )
        if self.logging_reward:
            self.reward_list.append(reward)
        # logger("info", "Py", f"{info=}")
        truncated = True if self.idx >= self.scene_cfg.max_steps else False
        target = [
            True if (key == "target" and value) else False
            for key, value in info.items()
        ][0]
        collision = [
            True if (key == "collision" and value) else False
            for key, value in info.items()
        ][0]

        if target and self.toggle_success_counter:
            self.toggle_success_counter = False
            if not self.eval:
                self.success_counter += 1

        done = (target or collision or truncated) if (not fixed_steps) else truncated
        self.done = done

        # logger("warning", "Py", f"{fixed_steps=}, {target=}, {collision=}")
        if (fixed_steps and target) or (fixed_steps and collision):
            self.turn_off_act = True
            # logger("warning", "Py", f"{self.turn_off_act=}")
            self.toggle_stable_end_checker = True
            self.collision_count += 1
        if self.toggle_first_print and (target or collision):
            logger("warning", "Py", "robot stop")
            if target:
                # because print start from 1 not from 0
                logger(
                    "hidden",
                    "Py",
                    f"Reach Goal!Scene={self.scenario_idx+1}, counter={self.success_counter}",
                )
            self.toggle_first_print = False
        if fixed_steps and truncated:
            self.turn_off_act = False
            # print("dimatiin pas disini")
            # self.idx = 0

        if self.toggle_stable_end_checker:
            self.sequential_count += 1
            if self.sequential_count != self.collision_count:
                self.sequential_count = 0
                self.collision_count = 0
                self.turn_off_act = False
                self.toggle_stable_end_checker = False
                # self.toggle_first_print = True
            else:
                # logger("hidden", "Py", "stable collision")
                pass

        # info = {}
        # print(f"{info=}")
        # masking truncated if target and collision is false because reach terminal state
        truncated = False if (target or collision) and fixed_steps else truncated
        # if (target or collision or truncated) and fixed_steps:
        if (done or truncated) and self.logging_reward:
            logger(
                "info",
                head_name,
                f"change episode scene: {self.scenario_idx+1}, cause: {'target' if target else ('collision' if collision else 'truncated')}",
            )
            reward_total = np.sum(
                np.asarray(
                    [
                        r * (self.agent_settings.gamma**q)
                        for q, r in enumerate(self.reward_list)
                    ]
                )
            )
            info["episode"] = {"r": reward_total, "l": self.idx}
            # adding to tensorboard
            self.writer.add_scalar(
                f"Reward/scene_{self.scenario_idx+1}", reward_total, self.step_idx
            )
            self.reward_list = []
            # print(f"{self.reward_list=}")
            # print(f"{info['episode']=}")
        #     print(f"{truncated=}")
        return state, reward, done, truncated, info

    def reset(self, seed: Optional[int] = 1):
        if self.first_step:
            self.first_step = False
            self.agent.step(self.timestep)
            time.sleep(0.1)
        while (
            "start" not in self.agent_publisher.subscriptions
            or "step" not in self.agent_publisher.subscriptions
        ):
            self.agent_publisher.receiveSubscriptions()
        if (
            self.success_counter >= self.scene_cfg.change_scene_every_goal_reach
            and self.eval == False
        ):
            self.success_counter = 0
            self.scenario_idx += 1
            if self.scenario_idx >= max_robot_scenario:
                self.scenario_idx = 0
                self.scenario_reach_end = True
        start_msg = start_agent_msg()
        start_msg.idx = self.start_idx
        start_msg.agent_state = "r"
        start_msg.x = list_robot_scenario[self.scenario_idx][0]
        start_msg.y = list_robot_scenario[self.scenario_idx][1]
        start_msg.z = list_robot_scenario[self.scenario_idx][2]
        start_msg.angle = list_robot_scenario[self.scenario_idx][3]
        start_msg.target_x = list_robot_scenario[self.scenario_idx][4]
        start_msg.target_y = list_robot_scenario[self.scenario_idx][5]
        start_msg.target_z = list_robot_scenario[self.scenario_idx][6]
        start_msg.max_steps = (
            self.scene_cfg.max_steps if (self.scene_cfg.max_steps >= 1e5) else 1024
        )
        start_msg.lidar_for_state = self.agent_settings.lidar_for_state

        self.agent_publisher.start_msg = start_msg
        self.agent_publisher.run(mode="start")
        self.agent.step()
        # print("this....")
        if self.agent_publisher.is_got_reply:
            # logger("info", "Py", "agent publisher got reply on reset")
            self.start_idx += 1
            self.agent_publisher.is_got_reply = False
        # stepping in this area
        self.agent.step()
        # print("that......")
        if self.robo_subscriber.get_msg() == False:
            # logger("info", "Py", "robo subscriber fail get msg")
            return None, {}
        # else:
        # logger("info", "Py", "agent robo subscriber get msg")

        state = self.robo_subscriber.states.lidar_data[
            : self.agent_settings.lidar_for_state
        ]
        state = np.asarray(state)
        state = np.append(
            state,
            [
                self.robo_subscriber.states.distance_length,
                self.robo_subscriber.states.angular_length,
            ],
        )
        # print(f"{state=}")
        info = {}
        self.toggle_success_counter = True
        self.idx = 0
        self.toggle_first_print = True
        return state, info
