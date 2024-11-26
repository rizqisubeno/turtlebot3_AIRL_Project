import os
from types import SimpleNamespace

import numpy as np
from library.tb3_agent import logger
from stable_baselines3.common.callbacks import BaseCallback


# progress adding evaluation at before saving the model
class Eval_and_Save(BaseCallback):
    def __init__(self,
                 env,
                 eval_configuration):
        super(Eval_and_Save, self).__init__()
        self.env = env
        self.eval_cfg = SimpleNamespace(**{sub_key: SimpleNamespace(**sub_item) for sub_key, sub_item in eval_configuration.items()})

        # for triggering save and eval model
        # this boolean function is to triggering after env done true
        self.trigger = False

        #for independent reward tracking...
        self.reward_list = []

    def _init_callback(self) -> None:
        if self.eval_cfg.save.path is None or self.eval_cfg.save.name is None:
            raise ValueError("Please Specify Save Path and name model for saving model!")
        # Create folder if needed
        if self.eval_cfg.save.path is None:
            os.makedirs(self.eval_cfg.save.path, exist_ok=True)
        if (self.eval_cfg.eval.save_result == True):
            os.makedirs(os.path.join(self.eval_cfg.save.path, "stat"), exist_ok=True)

    def _on_step(self) -> bool:
        if(self.env.done!=True):
            self.reward_list.append(self.locals['rewards'][0])
        else:
            all_reward = np.sum(np.asarray([val*(self.env.agent_settings.gamma**j) for j, val in enumerate(self.reward_list)]))
            self.reward_list = []
            logger("info", "CB", f"Total Reward: {all_reward}")

        if (self.n_calls % self.eval_cfg.every.timestep == 0 and self.eval_cfg.eval.enable==True):
            self.trigger = True
        if(self.env.done == True and self.trigger == True and self.eval_cfg.eval.enable==True):
            self.trigger = False

            # set to eval mode so not tracing change scenario automatically
            self.env.eval = True

            # saving last scene and last success counter
            self.last_scene_idx = self.env.scenario_idx
            self.last_success_counter = self.env.success_counter

            #resetting success counter
            self.env.success_counter = 0

            #evaluate policy
            n_success, ep_rew, ep_time_len = tb3_evaluate_policy(self.env,
                                                                 self.model,
                                                                 deterministic=True)

            #saving statistic reward and time length
            stat = np.vstack([n_success, ep_rew, ep_time_len], dtype=object).T
            os.path.join("./RL_model","stat/")+str("rl_ppo"+"_stat")+str("_step_0")
            with open(os.path.join(self.eval_cfg.save.path,"stat/")+str(self.eval_cfg.save.name+"_stat")+str(f"_step_{self.n_calls}"), 'wb') as f:
                np.save(f, stat)

            #saving RL model
            if (self.eval_cfg.save.enable == True):
                # save if all scenario robot success go to goal point
                # if not have failed tag at the end filename
                print(f"n_success: {np.sum(n_success)}")
                print(f"max_scenario: {self.env.max_scenario}")
                if(np.sum(n_success)==self.env.max_scenario):
                    self.model.save(os.path.join(self.eval_cfg.save.path, self.eval_cfg.save.name)+str(f"_step_{self.n_calls}"))
                else:
                    self.model.save(os.path.join(self.eval_cfg.save.path, self.eval_cfg.save.name)+str(f"_step_{self.n_calls}")+"_failed")

            # reload last scene and last success counter to env
            self.env.scenario_idx = self.last_scene_idx
            self.env.success_counter = self.last_success_counter

            # disable eval mode to change scenario automatically
            self.env.eval = False

            #resetting the environment
            self.env.reset()

        return True

# support only single environment
def tb3_evaluate_policy(env,
                        model,
                        deterministic:bool=True):
    head_name = "Eval_Mode"
    logger("hidden", head_name, "evaluate model policy")

    #for counting
    n_success = np.zeros(shape=env.max_scenario, dtype=np.bool_)
    ep_reward = np.zeros(shape=env.max_scenario, dtype=np.float64)
    ep_time_length = np.zeros(shape=env.max_scenario, dtype=np.float64)

    reward_list = []
    model.policy.set_training_mode(False)
    for i in range(env.max_scenario):
        env.scenario_idx = i
        # resetting every change scenario (forcing...)
        env.success_counter = 0
        logger("hidden", head_name, f"Eval scene-{i+1}")
        observations, _ = env.reset(seed=0)
        states = None
        #counter iteration episode on env.n_step
        # episode_starts = np.ones((env.num_envs,), dtype=bool)
        episode_starts = np.ones((1), dtype=bool)
        reward_list = []
        while (env.idx < env.scene_cfg.max_steps):
            actions, states = model.predict(observations,
                                           state=states,
                                           episode_start=episode_starts,
                                           deterministic=deterministic)

            new_observations, rewards, dones, trunc, infos = env.step(actions)
            reward_list.append(rewards)
            episode_starts[0] = dones

            observations = new_observations

            if (dones or trunc):
                print("infos: ", infos)
                all_reward = np.sum(np.asarray([val*(env.agent_settings.gamma**j) for j, val in enumerate(reward_list)]))
                ep_reward[i] = all_reward
                ep_time_length[i] = env.idx*env.timestep/1000.0
                if "target" in infos:
                    if(infos["target"]==True):
                        n_success[i] = True
                    else:
                        n_success[i] = False    #target false
                elif "collision" in infos:
                    n_success[i] = False
                else:
                    n_success[i] = False
                print("n_success: ", n_success[i])
                break
    model.policy.set_training_mode(True)
    return n_success, ep_reward, ep_time_length


# class AIRL_Standard_Callback(BaseCallback):
#     def __init__(self,
#                  env,
#                  max_success_to_goal: int = 5):
#         self.env = env

#         self.n_calls= 0
#         self.max_success_to_goal = max_success_to_goal

#     def _init_callback(self) -> None:
#         pass

#     def _on_step(self) -> bool:
#         self.n_calls += 1
#         print("~~~~~callback called~~~~~")

#         if(self.env.reset_called==True):
#             self.steps = 0
#             self.env.reset_called=False

#         if (self.env.success_counter >= self.max_success_to_goal):
#             return False                                    # set ending the training AIRL after passing threshold

#         return True
