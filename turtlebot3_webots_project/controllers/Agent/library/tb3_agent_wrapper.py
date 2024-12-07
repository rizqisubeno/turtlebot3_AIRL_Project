import os
import sys
from types import SimpleNamespace
from typing import Optional

import numpy as np
from imitation.data import types
from library.tb3_agent import Agent, logger

# from scenario list
from scenario_list import list_robot_scenario


class JoystickEnv(Agent):
    """
    env wrapper for use joystick
    to create expert demonstration,
    to manually read state, reward, info from joy controller
    or to manually mapping tools
    """

    def __init__(
        self,
        name_exp,
        agent_settings,
        scene_configuration,
        tracking_configuration,
        fixed_steps,
        type_of_trajectory: Optional[str] = "imitation_trajectory",
        verbose: Optional[bool] = False,
    ):
        super(JoystickEnv, self).__init__(
            name_exp, agent_settings, scene_configuration, fixed_steps, verbose
        )

        self.head_name = "Joy_Env"
        self.verbose = verbose
        self.type_of_trajectory = type_of_trajectory

        # allowing this string only
        assert (
            type_of_trajectory == "imitation_trajectory" or type_of_trajectory == "dict"
        )

        self.en_tracking = False
        self.save_tracking = False
        self.save_path = None
        self.save_name = None
        for key, item in tracking_configuration.items():
            if "en" in key:
                self.en_tracking = item
            if "save_track" in key:
                self.save_tracking = item
            if "folder" in key:
                if item is not None:
                    self.save_path = item
                    if not os.path.exists(self.save_path):
                        os.makedirs(self.save_path)
                        logger("info", self.head_name, "save path created")

            if "save_name" in key:
                self.save_name = item

        self.demo_stop = False

        # set first_step to false because after this code is stepping the webots once to setup joystick
        self.first_step = False
        try:
            self.joyController = self.agent.getJoystick()
            self.joyController.enable(True)
            self.agent.step(self.timestep)
            logger(
                "info",
                self.head_name,
                f"joystick connected : {self.joyController.model}",
            )
        except:
            logger("err", self.head_name, "joystick not found")
            sys.exit(1)

        if self.en_tracking:
            self.tracking = SimpleNamespace(
                trajectory=[], states=[], actions=[], infos=[]
            )

    def step(self, action):
        if self.turn_off_act:
            # print("nyala kie...")
            action = [-1.0, 0.0]
        # because done only after end episode then just follow the output tuple
        state, reward, done, truncated, info = super(JoystickEnv, self).step(action)

        if self.en_tracking:
            self.tracking.states.append(
                state.tolist() if isinstance(action, np.ndarray) else state
            )
            self.tracking.actions.append(
                action.tolist() if isinstance(action, np.ndarray) else action
            )
            self.tracking.infos.append(info)

        # if(done):
        if self.idx + 1 > self.scene_cfg.max_steps and self.fixed_steps:
            # print("awal done")
            traj_state_tuple = np.array(self.tracking.states)
            traj_action_tuple = np.array(self.tracking.actions)
            traj_info_tuple = np.array(self.tracking.infos)
            if "target" in info:
                if info["target"]:
                    terminal = True
                else:
                    terminal = False
            else:
                terminal = False
            if self.type_of_trajectory == "imitation_trajectory":
                trajectory = types.Trajectory(
                    obs=traj_state_tuple,
                    acts=traj_action_tuple,
                    infos=traj_info_tuple,
                    terminal=terminal,
                )
            elif self.type_of_trajectory == "dict":
                trajectory = {
                    "obs": traj_state_tuple,
                    "acts": traj_action_tuple,
                    "infos": traj_info_tuple,
                    "terminal": terminal,
                }

            self.tracking.trajectory.append(trajectory)

            self.tracking.states = []
            self.tracking.actions = []
            self.tracking.infos = []
            # print("akhir done")

        return state, reward, done, truncated, info

    def reset(self, seed=None):
        # print(f"{self.success_counter=}")
        # print(f"{self.scene_cfg.change_scene_every_goal_reach=}")
        if self.success_counter == self.scene_cfg.change_scene_every_goal_reach:
            print("saving demons")
            np.save(
                f"{self.save_path}/{self.save_name}_id_{self.scenario_idx}.npy",
                self.tracking.trajectory,
                allow_pickle=True,
            )
            if self.scenario_idx + 1 >= 1:
                self.demo_stop = True
            self.tracking.trajectory = []
        # print("masuk reset ni...")
        state, info = super(JoystickEnv, self).reset(seed=seed)
        # print("keluar reset ni...")
        # print(f"{self.demo_stop=}")

        # adding first state when robot start
        if self.en_tracking:
            self.tracking.states.append(
                state.tolist() if isinstance(state, np.ndarray) else state
            )

        return state, info


class TB3_Agent_Demo(Agent):
    def __init__(self,
                 name_exp,
                 agent_settings,
                 scene_configuration,
                 fixed_steps: bool = True,
                 logging_reward: bool = False,
                 verbose: bool = False):
        super(TB3_Agent_Demo, self).__init__(name_exp, 
                                             agent_settings, 
                                             scene_configuration, 
                                             fixed_steps, 
                                             verbose)
        
        self.scenario_now_idx = self.scene_cfg.scene_start_from
        self.spawn_and_change_hidden_ball(self.scene_cfg.scene_start_from)

    def spawn_and_change_hidden_ball(self, 
                                     scenario_idx):
        self.root_node = self.agent.getRoot()
        self.children_field = self.root_node.getField('children')        
        
        j = 0
        for i in range(self.max_scenario):
            if i==scenario_idx:
                pass
            else:
                print("spawn hidden target index : ", i)
                x = list_robot_scenario[i][4]
                y = list_robot_scenario[i][5]
                z = list_robot_scenario[i][6]
                self.children_field.importMFNodeFromString(-1, 
                                                           f"DEF hidden{j} hidden_target_ball {{ translation {x} {y} {z} }}")
                j+=1

    def delete_hidden_ball(self,
                           scenario_idx):
        j = 0
        for i in range(self.max_scenario):
            if i==scenario_idx:
                pass
            else:
                print("delete hidden target index : ", i)
                hidden_node = self.agent.getFromDef(f'hidden{j}')
                hidden_node.remove()
                j+=1

    def step(self, action):
        return super(TB3_Agent_Demo, self).step(action)

    def reset(self, seed: Optional[int] = 1):
        obs, info = super(TB3_Agent_Demo, self).reset(seed)
        if self.scenario_now_idx != self.scenario_idx:
            self.scenario_now_idx = self.scenario_idx
            self.delete_hidden_ball(self.scenario_now_idx)
            self.spawn_and_change_hidden_ball(self.scenario_now_idx)

        return obs, info

