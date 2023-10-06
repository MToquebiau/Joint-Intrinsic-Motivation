import numpy as np
import random

from gym import spaces

class RelOvergenEnv:

    def __init__(self, state_dim, n_agents=2,
                 optim_reward=12, optim_diff_coeff=30, 
                 suboptim_reward=0, suboptim_diff_coeff=0.08,
                 save_visited_states=False):
        self.obs_dim = state_dim
        self.act_dim = 3
        self.n_agents = n_agents

        self.state_dim = state_dim
        self.unit = 10.0 / state_dim
        self.states = list(np.arange(0.0, 10.0, self.unit))

        self._obs_high = np.ones(state_dim, dtype=np.float32)
        self._obs_low = np.zeros(state_dim, dtype=np.float32)
        self.observation_space = [
            spaces.Box(self._obs_low, self._obs_high, dtype=np.float32) 
            for a_i in range(self.n_agents)]
        self._shared_obs_high = np.ones(
            state_dim * self.n_agents, dtype=np.float32)
        self._shared_obs_low = np.zeros(
            state_dim * self.n_agents, dtype=np.float32)
        self.shared_observation_space = [
            spaces.Box(self._shared_obs_low, self._shared_obs_high, 
                       dtype=np.float32) 
            for a_i in range(self.n_agents)]
        self.action_space = [
            spaces.Discrete(3) for a_i in range(self.n_agents)]

        self.agents_pos = [0] * self.n_agents

        self.optimal_state = [
            int(state_dim / 4) * self.unit, 
            int(state_dim / 5) * self.unit, 
            int(state_dim / 6) * self.unit, 
            int(state_dim / 5) * self.unit, 
            int(state_dim / 4) * self.unit]
        self.suboptimal_state = [
            10.0 - self.optimal_state[a_i] for a_i in range(self.n_agents)]
        
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
            self.visited_states.append(list(self.agents_pos))
        return [np.eye(self.state_dim)[self.agents_pos[a_i]] 
                for a_i in range(self.n_agents)]

    def reset(self, init_pos=None):
        for a_i in range(self.n_agents):
            self.agents_pos[a_i] = random.randint(0, self.state_dim - 1)
        self.current_step = 0
        return self.get_obs()

    def compute_reward(self):
        opti = self.optim_reward - self.optim_diff_coeff * sum([
            (self.states[self.agents_pos[a_i]] - self.optimal_state[a_i]) ** 2
            for a_i in range(self.n_agents)])
        subopti = self.suboptim_reward - self.suboptim_diff_coeff * sum([
            (self.states[self.agents_pos[a_i]] - self.suboptimal_state[a_i]) ** 2
            for a_i in range(self.n_agents)])
        return max(opti, subopti)

    def step(self, actions):
        for a_i in range(self.n_agents):
            onehot_action = np.eye(3)[int(actions[a_i][0])]
            if self.agents_pos[a_i] < self.state_dim - 1:
                self.agents_pos[a_i] += int(onehot_action[0])
            if self.agents_pos[a_i] > 0:
                self.agents_pos[a_i] -= int(onehot_action[1])
        next_states = self.get_obs()

        reward = self.compute_reward()
        rewards = [reward] * self.n_agents
        
        self.current_step += 1
        done = float(self.current_step >= self.max_steps)
        dones = [done] * self.n_agents
        
        return next_states, rewards, dones, None

    def seed(self, seed):
        random.seed(seed)
        np.random.seed(seed)
    
    def close(self):
        pass