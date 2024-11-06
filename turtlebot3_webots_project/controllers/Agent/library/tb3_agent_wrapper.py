from sympy import true
from library.tb3_agent import logger, Agent

import os, sys
from typing import *
from types import SimpleNamespace

import numpy as np

from imitation.data import types


class JoystickEnv(Agent):
    """
    env wrapper for use joystick
    to create expert demonstration,
    to manually read state, reward, info from joy controller
    or to manually mapping tools
    """

    def __init__(self,
                 name_exp,
                 agent_settings,
                 scene_configuration,
                 tracking_configuration,
                 fixed_steps,
                 type_of_trajectory:Optional[str]="imitation_trajectory",
                 verbose: Optional[bool]=False,):
        
        super(JoystickEnv, self).__init__(name_exp,
                                          agent_settings,
                                          scene_configuration,
                                          fixed_steps,
                                          verbose)
        
        self.head_name = "Joy_Env"
        self.verbose = verbose
        self.type_of_trajectory = type_of_trajectory

        # allowing this string only
        assert type_of_trajectory == "imitation_trajectory" or type_of_trajectory == "dict"
        
        self.en_tracking = False
        self.save_tracking = False
        self.save_path = None
        self.save_name = None
        for key, item in tracking_configuration.items():
            if ("en" in key):
                self.en_tracking = item
            if ("save_track" in key):
                self.save_tracking = item
            if ("folder" in key):
                if(item is not None):
                    self.save_path = item
                    if not os.path.exists(self.save_path):
                        os.makedirs(self.save_path)
                        logger("info", self.head_name, "save path created")

            if ("save_name" in key):
                self.save_name = item

        self.demo_stop = False

        #set first_step to false because after this code is stepping the webots once to setup joystick
        self.first_step = False
        try:
            self.joyController = self.agent.getJoystick()
            self.joyController.enable(True)
            self.agent.step(self.timestep)
            logger("info", self.head_name, f"joystick connected : {self.joyController.model}")
        except:
            logger("err", self.head_name, f"joystick not found")
            sys.exit(1)
        
        if(self.en_tracking):
            self.tracking = SimpleNamespace(trajectory=[],
                                            states=[],
                                            actions=[],
                                            infos=[])
    
    def step(self, action):
        if(self.turn_off_act):
            # print("nyala kie...")
            action = [-1.0, 0.0]
        # because done only after end episode then just follow the output tuple
        state, reward, done, truncated, info = super(JoystickEnv, self).step(action)

        if(self.en_tracking):
            self.tracking.states.append(state.tolist() if type(state)==np.ndarray else state)
            self.tracking.actions.append(action.tolist() if type(action)==np.ndarray else action)
            self.tracking.infos.append(info)

        # if(done):
        if (self.idx+1>self.scene_cfg.max_steps and self.fixed_steps==True):
            # print("awal done")
            traj_state_tuple = np.array(self.tracking.states)
            traj_action_tuple = np.array(self.tracking.actions)
            traj_info_tuple = np.array(self.tracking.infos)
            if "target" in info:
                if(info["target"] == True):
                    terminal = True
                else:
                    terminal = False
            else:
                terminal = False
            if(self.type_of_trajectory=="imitation_trajectory"):
                trajectory = types.Trajectory(obs=traj_state_tuple, acts=traj_action_tuple,
                                            infos = traj_info_tuple, terminal=terminal)
            elif(self.type_of_trajectory=="dict"):
                trajectory = {"obs":traj_state_tuple,
                              "acts":traj_action_tuple,
                              "infos":traj_info_tuple,
                              "terminal":terminal}
            
            self.tracking.trajectory.append(trajectory)

            self.tracking.states = []
            self.tracking.actions = []
            self.tracking.infos = []
            # print("akhir done")

        return state, reward, done, truncated, info
    
    def reset(self, seed=None):  
        # print(f"{self.success_counter=}")
        # print(f"{self.scene_cfg.change_scene_every_goal_reach=}")
        if(self.success_counter == self.scene_cfg.change_scene_every_goal_reach):
            print("saving demons")
            np.save(f"{self.save_path}/{self.save_name}_id_{self.scenario_idx}.npy",
                    self.tracking.trajectory, allow_pickle=True)
            if(self.scenario_idx+1 >= 1):
                self.demo_stop = True
            self.tracking.trajectory = []
        # print("masuk reset ni...")
        state, info = super(JoystickEnv, self).reset(seed=seed)
        # print("keluar reset ni...")
        # print(f"{self.demo_stop=}")

        #adding first state when robot start
        if (self.en_tracking):
            self.tracking.states.append(state.tolist() if type(state)==np.ndarray else state)

        return state, info

