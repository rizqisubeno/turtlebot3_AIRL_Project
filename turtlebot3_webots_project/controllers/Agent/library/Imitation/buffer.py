import os
import numpy as np
import torch


class SerializedBuffer:

    def __init__(self, path, device):
        tmp = torch.load(path)
        # self.buffer_size = self._n = tmp['state'].size(0)
        self.device = device

        self.states = tmp['state'].clone().to(self.device)
        self.actions = tmp['action'].clone().to(self.device)
        self.rewards = tmp['reward'].clone().to(self.device)
        self.dones = tmp['done'].clone().to(self.device)
        self.next_states = tmp['next_state'].clone().to(self.device)

    def sample(self, batch_size):
        idxes = np.random.randint(low=0, high=self._n, size=batch_size)
        return (
            self.states[idxes],
            self.actions[idxes],
            self.rewards[idxes],
            self.dones[idxes],
            self.next_states[idxes]
        )


class Buffer(SerializedBuffer):

    def __init__(self, buffer_size, state_shape, action_shape, device):
        self._n = 0
        self._p = 0
        self.buffer_size = buffer_size
        self.device = device

        self.states = torch.empty(
            (buffer_size, *state_shape), dtype=torch.float, device=device)
        self.actions = torch.empty(
            (buffer_size, *action_shape), dtype=torch.float, device=device)
        self.rewards = torch.empty(
            (buffer_size, 1), dtype=torch.float, device=device)
        self.dones = torch.empty(
            (buffer_size, 1), dtype=torch.float, device=device)
        self.next_states = torch.empty(
            (buffer_size, *state_shape), dtype=torch.float, device=device)

    def append(self, state, action, reward, done, next_state):
        self.states[self._p].copy_(torch.from_numpy(state))
        self.actions[self._p].copy_(torch.from_numpy(action))
        self.rewards[self._p] = float(reward)
        self.dones[self._p] = float(done)
        self.next_states[self._p].copy_(torch.from_numpy(next_state))

        self._p = (self._p + 1) % self.buffer_size
        self._n = min(self._n + 1, self.buffer_size)

    def save(self, path):
        if not os.path.exists(os.path.dirname(path)):
            os.makedirs(os.path.dirname(path))

        torch.save({
            'state': self.states.clone().cpu(),
            'action': self.actions.clone().cpu(),
            'reward': self.rewards.clone().cpu(),
            'done': self.dones.clone().cpu(),
            'next_state': self.next_states.clone().cpu(),
        }, path)

    def load(self, path):
        if not os.path.exists(os.path.dirname(path)):
            os.makedirs(os.path.dirname(path))

        data = torch.load(path,
                          weights_only=True)
        self.states = data['state'].clone().to(self.device)
        self.actions = data['action'].clone().to(self.device)
        self.rewards = data['reward'].clone().to(self.device)
        self.dones = data['dones'].clone().to(self.device)
        self.next_states = data['next_state'].clone().to(self.device)

        self._p = 0
        self._n = torch.prod(torch.as_tensor(data['dones'].shape))

class ModifiedBuffer():
    def __init__(self, device):
        self.device = device
        self.latest_idx = 0
        self.num_ep = 0
        self.choices = []

    def load(self, path):
        data = torch.load(path, 
                          weights_only=False)
        self.num_ep = len(data)
        self.num_ep_traj = len(data[0][0]['action'])
        state_shape = data[0][0]['state'][0].shape
        action_shape = data[0][0]['action'][0].shape
        next_state_shape = data[0][0]['next_state'][0].shape

        self.states = torch.empty(
            (self.num_ep, self.num_ep_traj, *state_shape), dtype=torch.float, device=self.device)
        self.actions = torch.empty(
            (self.num_ep, self.num_ep_traj, *action_shape), dtype=torch.float, device=self.device)
        self.rewards = torch.empty(
            (self.num_ep, self.num_ep_traj, 1), dtype=torch.float, device=self.device)
        # print(f"{self.rewards.shape=}")
        self.dones = torch.empty(
            (self.num_ep, self.num_ep_traj, 1), dtype=torch.float, device=self.device)
        self.next_states = torch.empty(
            (self.num_ep, self.num_ep_traj, *next_state_shape), dtype=torch.float, device=self.device)
        self.rank = torch.empty(
            (self.num_ep, 1), dtype=torch.float, device=self.device)
        
        for i_traj in range(len(data)):
            self.states[i_traj] = data[i_traj][0]['state'].clone().to(self.device)
            self.actions[i_traj] = data[i_traj][0]['action'].clone().to(self.device)
            # print(f"{self.rewards[i_traj].shape=}")
            self.rewards[i_traj] = data[i_traj][0]['reward'].clone().to(self.device)
            self.dones[i_traj] = data[i_traj][0]['dones'].clone().to(self.device)
            self.next_states[i_traj] = data[i_traj][0]['next_state'].clone().to(self.device)

            self.rank[i_traj] = torch.Tensor([data[i_traj][1]['rank']]).to(self.device)

        self.choices = self.sampling_choice()
        #reset every load new expert trajectory
        self.latest_idx = 0
    
    def sampling_choice(self):
        return np.random.choice(self.num_ep, 50, p=self.rank.reshape(-1).cpu().numpy(), replace=False)
    
    # customize in this function
    def sample(self, batch_size):
        if(self.latest_idx>=self.num_ep):
            self.latest_idx = 0
            self.choices = self.sampling_choice()
        id_ep = self.choices[self.latest_idx]
        self.latest_idx+=1
        idxes = torch.randint(low=0, high=self.num_ep_traj, size=(batch_size,))
        return (
            self.states[id_ep][idxes],
            self.actions[id_ep][idxes],
            self.rewards[id_ep][idxes],
            self.dones[id_ep][idxes],
            self.next_states[id_ep][idxes]
        )


class RolloutBuffer():
    def __init__(self,
                 buffer_size,
                 num_envs,
                 state_shape,
                 action_shape,
                 device):
        
        self.idx = 0
        self.min_idx = 0
        self.max_buffer = buffer_size
        self.num_envs = num_envs
        #ALGO Logic: Storage setup
        self.obs = torch.zeros((self.max_buffer, self.num_envs) + state_shape).to(device)
        self.next_obs = torch.zeros((self.max_buffer, self.num_envs) + state_shape).to(device)
        self.actions = torch.zeros((self.max_buffer, self.num_envs) + action_shape).to(device)
        self.logprobs = torch.zeros((self.max_buffer, self.num_envs)).to(device)
        self.rewards = torch.zeros((self.max_buffer, self.num_envs)).to(device)
        self.next_dones = torch.zeros((self.max_buffer, self.num_envs)).to(device)
        self.next_terminations = torch.zeros((self.max_buffer, self.num_envs)).to(device)
        self.values = torch.zeros((self.max_buffer, self.num_envs)).to(device)

    def append(self,
               ob,
               next_ob,
               action,
               logprob,
               value,
               next_termination,
               next_done,
               reward,
               ):
        self.obs[self.idx] = ob
        self.next_obs[self.idx] = next_ob
        self.actions[self.idx] = action
        self.logprobs[self.idx] = logprob
        self.values[self.idx] = value
        self.next_terminations[self.idx] = next_termination
        self.next_dones[self.idx] = next_done
        self.rewards[self.idx] = reward

        self.idx = (self.idx + 1) % self.max_buffer
        self.min_idx = min(self.min_idx+1, self.max_buffer) 

    def get(self):
        assert self.idx % self.max_buffer == 0
        start = (self.idx - self.max_buffer) % self.max_buffer
        idxes = slice(start, start + self.max_buffer)
        return (
            self.obs[idxes],
            self.next_obs[idxes],
            self.actions[idxes],
            self.logprobs[idxes],
            self.values[idxes],
            self.next_terminations[idxes],
            self.next_dones[idxes],
            self.rewards[idxes],
        )

    def set(self,
            obs,
            next_obs,
            actions,
            logprobs,
            values,
            next_terminations,
            next_dones,
            rewards):
        assert self.idx % self.max_buffer == 0
        self.obs = obs
        self.next_obs = next_obs
        self.actions = actions
        self.logprobs = logprobs
        self.values = values
        self.next_terminations = next_terminations
        self.next_dones = next_dones
        self.rewards = rewards

    def sample(self, batch_size):
        # avoid on array not being filled until max size
        assert self.idx % self.max_buffer == 0
        idxes = np.random.randint(low=0, high=self.min_idx, size=batch_size)
        return (
            self.obs[idxes],
            self.next_obs[idxes],
            self.actions[idxes],
            self.logprobs[idxes],
            self.values[idxes],
            self.next_terminations[idxes],
            self.next_dones[idxes],
            self.rewards[idxes],
        )
