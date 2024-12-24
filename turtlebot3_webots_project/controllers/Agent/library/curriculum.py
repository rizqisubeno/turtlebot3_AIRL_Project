import numpy as np
from collections import deque
class TeacherExp3(object):
  """Teacher with Exponential-weight algorithm for Exploration and Exploitation.
  https://rlcurriculum.github.io/
  """

  def __init__(self, tasks, gamma=0.3):
    self._tasks = tasks
    self._n_tasks = len(self._tasks)
    self._gamma = gamma
    self._log_weights = np.zeros(self._n_tasks)
    # first scenario sampling is from scenario 0 first
    self.first_scenario = True
    
  @property
  def task_probabilities(self):
    weights = np.exp(self._log_weights - np.sum(self._log_weights))
    probs = ((1 - self._gamma)*weights / np.sum(weights) + self._gamma/self._n_tasks)
    return probs
  
  def get_task(self):
    """Samples a task, according to current Exp3 belief.
    """
    if self.first_scenario:
        task_i = 0
        self.first_scenario = False
    else:
        task_i = np.random.choice(self._n_tasks, p=self.task_probabilities)
    return task_i #self._tasks[task_i]
  
  def update(self, task_i, reward):
    """ Updates the weight of task given current reward observed
    """    
    # task_i = self._tasks.index(task)
    reward_corrected = reward/self.task_probabilities[task_i]
    self._log_weights[task_i] += self._gamma*reward_corrected/self._n_tasks

# class CurriculumLearning:
#     def __init__(self, num_tasks, buffer_size, student_algorithm):
#         """
#         Initialize the curriculum learning sampling algorithm.

#         :param num_tasks: Number of tasks (N)
#         :param buffer_size: Size of the FIFO buffer (K)
#         :param student_algorithm: The RL agent (STUDENT)
#         """
#         self.num_tasks = num_tasks
#         self.buffer_size = buffer_size
#         self.student = student_algorithm
#         self.task_buffers = {task: deque(maxlen=buffer_size) for task in range(num_tasks)}

#     def sample_reward(self, task):
#         """
#         Sample a reward from the task buffer. If the buffer is empty, return a default reward of 1.
#         """
#         if len(self.task_buffers[task]) == 0:
#             return 1  # Default reward for empty buffer
#         return np.random.choice(self.task_buffers[task])

#     def train(self, total_timesteps):
#         """
#         Train the STUDENT using the curriculum sampling algorithm.

#         :param total_timesteps: Total number of training steps (T)
#         """
#         for t in range(total_timesteps):
#             # Step 1: Sample rewards for all tasks
#             sampled_rewards = [self.sample_reward(task) for task in range(self.num_tasks)]

#             # Step 2: Choose the task with the highest sampled reward
#             selected_task = max(range(self.num_tasks), key=lambda task: abs(sampled_rewards[task]))

#             # Step 3: Train the STUDENT on the selected task and observe the reward
#             reward = self.student.train_on_task(selected_task)  # STUDENT method to train on a task

#             # Step 4: Store the observed reward in the task buffer
#             self.task_buffers[selected_task].append(reward)
    