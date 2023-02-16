import numpy as np
import random

class RelOvergenEnv:

    def __init__(self, state_dim, 
                 optim_reward=8, optim_diff_coeff=25, 
                 suboptim_reward=0, suboptim_diff_coeff=0.125,
                 save_visited_states=False):
        self.obs_dim = state_dim
        self.act_dim = 3

        self.state_dim = state_dim
        self.unit = 10.0 / state_dim
        self.states = list(np.arange(0.0, 10.0, self.unit))

        self.agents_pos = [0, 0]

        self.optimal_state = [
            int(state_dim / 4) * self.unit, 
            int(state_dim / 5) * self.unit]
        self.suboptimal_state = [
            10.0 - self.optimal_state[0], 
            10.0 - self.optimal_state[1]]
        
        self.optim_reward = optim_reward
        self.optim_diff_coeff = optim_diff_coeff
        self.suboptim_reward = suboptim_reward
        self.suboptim_diff_coeff = suboptim_diff_coeff
        
        self.max_steps = state_dim
        self.current_step = 0

        self.save_visited = save_visited_states
        self.visited_states = []

    def get_obs(self):
        if self.save_visited:
            self.visited_states.append(self.agents_pos[:])
        return [
            np.eye(self.state_dim)[self.agents_pos[0]],
            np.eye(self.state_dim)[self.agents_pos[1]]
        ]

    def reset(self):
        for a_i in range(2):
            self.agents_pos[a_i] = random.randint(0, self.state_dim - 1)
        self.current_step = 0
        return self.get_obs()

    def compute_reward(self):
        opti = self.optim_reward - self.optim_diff_coeff * (
            (self.states[self.agents_pos[0]] - self.optimal_state[0]) ** 2 + 
            (self.states[self.agents_pos[1]] - self.optimal_state[1]) ** 2)
        subopti = self.suboptim_reward - self.suboptim_diff_coeff * (
            (self.states[self.agents_pos[0]] - self.suboptimal_state[0]) ** 2 + 
            (self.states[self.agents_pos[1]] - self.suboptimal_state[1]) ** 2)
        return max(opti, subopti)

    def step(self, actions):
        for a_i in range(2):
            self.agents_pos[a_i] += int(actions[a_i][0])
            self.agents_pos[a_i] -= int(actions[a_i][1])
            if self.agents_pos[a_i] < 0:
                self.agents_pos[a_i] = 0
            elif self.agents_pos[a_i] >= self.state_dim:
                self.agents_pos[a_i] = self.state_dim - 1
        next_states = self.get_obs()

        reward = self.compute_reward()
        rewards = [reward, reward]
        
        self.current_step += 1
        done = float(self.current_step >= self.max_steps)
        dones = [done, done]
        
        return next_states, rewards, dones, None
    
    def close(self):
        pass