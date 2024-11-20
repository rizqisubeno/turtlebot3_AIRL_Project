import os
import numpy as np
import torch


class SerializedBuffer:

    def __init__(self, path, device):
        tmp = torch.load(path)
        self.buffer_size = self._n = tmp['state'].size(0)
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
        self.dones = data['done'].clone().to(self.device)
        self.next_states = data['next_state'].clone().to(self.device)

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

    def sample(self, batch_size):
        # avoid on array not being filled until max size
        assert self.idx % self.max_buffer == 0
        idxes = np.random.randint(low=0, high=self._n, size=batch_size)
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
