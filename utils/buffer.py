import random
import numpy as np
from torch import Tensor

class ReplayBuffer(object):

    def __init__(self, max_steps, nb_agents, obs_dims, ac_dims):
        """
        Replay buffer class.
        Inputs:
            max_steps (int): Maximum number of timepoints to store in buffer
            nb_agents (int): Number of agents in environment
            obs_dims (list of ints): number of obervation dimensions for each
                                     agent
            ac_dims (list of ints): number of action dimensions for each agent
        """
        self.max_steps = max_steps
        self.nb_agents = nb_agents

        self.obs_buffs = []
        self.act_buffs = []
        self.rew_buffs = []
        self.next_obs_buffs = []
        self.done_buffs = []
        for odim, adim in zip(obs_dims, ac_dims):
            self.obs_buffs.append(np.zeros((max_steps, odim)))
            self.act_buffs.append(np.zeros((max_steps, adim)))
            self.rew_buffs.append(np.zeros(max_steps))
            self.next_obs_buffs.append(np.zeros((max_steps, odim)))
            self.done_buffs.append(np.zeros(max_steps))

        self.filled_i = 0  # index of first empty location in buffer (last index when full)
        self.curr_i = 0  # current index to write to (ovewrite oldest data)

    def __len__(self):
        return self.filled_i

    def push(self, observations, actions, rewards, next_observations, dones):
        nentries = observations.shape[0]  # handle multiple parallel environments
        if self.curr_i + nentries > self.max_steps:
            rollover = self.max_steps - self.curr_i # num of indices to roll over
            for agent_i in range(self.nb_agents):
                self.obs_buffs[agent_i] = np.roll(self.obs_buffs[agent_i],
                                                  rollover, axis=0)
                self.act_buffs[agent_i] = np.roll(self.act_buffs[agent_i],
                                                 rollover, axis=0)
                self.rew_buffs[agent_i] = np.roll(self.rew_buffs[agent_i],
                                                  rollover)
                self.next_obs_buffs[agent_i] = np.roll(
                    self.next_obs_buffs[agent_i], rollover, axis=0)
                self.done_buffs[agent_i] = np.roll(self.done_buffs[agent_i],
                                                   rollover)
            self.curr_i = 0
            self.filled_i = self.max_steps
        for agent_i in range(self.nb_agents):
            self.obs_buffs[agent_i][self.curr_i:self.curr_i + nentries] = np.vstack(
                observations[:, agent_i])
            # actions are already batched by agent, so they are indexed differently
            self.act_buffs[agent_i][self.curr_i:self.curr_i + nentries] = actions[agent_i]
            self.rew_buffs[agent_i][self.curr_i:self.curr_i + nentries] = rewards[:, agent_i]
            self.next_obs_buffs[agent_i][self.curr_i:self.curr_i + nentries] = np.vstack(
                next_observations[:, agent_i])
            self.done_buffs[agent_i][self.curr_i:self.curr_i + nentries] = dones[:, agent_i]
        self.curr_i += nentries
        if self.filled_i < self.max_steps:
            self.filled_i += nentries
        if self.curr_i == self.max_steps:
            self.curr_i = 0

    def sample(self, N, cuda_device=None, norm_rews=True):
        inds = np.random.choice(np.arange(self.filled_i), size=N,
                                replace=False)
        if cuda_device is not None:
            cast = lambda x: Tensor(x).to(cuda_device)
        else:
            cast = lambda x: Tensor(x)
        if norm_rews:
            ret_rews = [cast((self.rew_buffs[i][inds] -
                              self.rew_buffs[i][:self.filled_i].mean()) /
                              self.rew_buffs[i][:self.filled_i].std())
                        for i in range(self.nb_agents)]
        else:
            ret_rews = [cast(self.rew_buffs[i][inds]) for i in range(self.nb_agents)]
        return ([cast(self.obs_buffs[i][inds]) for i in range(self.nb_agents)],
                [cast(self.act_buffs[i][inds]) for i in range(self.nb_agents)],
                ret_rews,
                [cast(self.next_obs_buffs[i][inds]) for i in range(self.nb_agents)],
                [cast(self.done_buffs[i][inds]) for i in range(self.nb_agents)])

    def get_average_rewards(self, N):
        if self.filled_i == self.max_steps:
            inds = np.arange(self.curr_i - N, self.curr_i)  # allow for negative indexing
        else:
            inds = np.arange(max(0, self.curr_i - N), self.curr_i)
        return [self.rew_buffs[i][inds].sum() for i in range(self.nb_agents)]


class RecReplayBuffer:

    def __init__(self, buffer_size, episode_length, 
                 nb_agents, obs_dim, act_dim):
        """
        Replay buffer for recurrent policy, stores complete episode 
        trajectories.
        Inputs:
            buffer_size (int): Max number of episodes to store in buffer.
            episode_length (int): Max length of an episode.
            nb_agents (int): Number of agents.
            obs_dim (int): Dimension of the observations.
            act_dim (int): Dimension of the actions.
        """
        self.buffer_size = buffer_size
        self.episode_length = episode_length
        self.nb_agents = nb_agents
        self.obs_dim = obs_dim
        self.shared_obs_dim = obs_dim * nb_agents
        self.act_dim = act_dim
        
        self.obs_buff = np.zeros(
            (episode_length + 1, buffer_size, nb_agents, self.obs_dim),
            dtype=np.float32
        )
        self.shared_obs_buff = np.zeros(
            (episode_length + 1, buffer_size, nb_agents, self.shared_obs_dim),
            dtype=np.float32
        )
        self.act_buff = np.zeros(
            (episode_length, buffer_size, nb_agents, self.act_dim),
            dtype=np.float32
        )
        self.rew_buff = np.zeros(
            (episode_length, buffer_size, nb_agents, 1),
            dtype=np.float32
        )
        self.done_buff = np.zeros(
            (episode_length, buffer_size, nb_agents, 1),
            dtype=np.float32
        )

        # Index of the first empty location
        self.filled_i = 0
        # Current index to write to
        self.curr_i = 0

    def __len__(self):
        return self.filled_i

    def init_episode_arrays(self, batch_size=1):
        """
        Returns zero-filled arrays to initialise episode trajectories arrays.
        Inputs:
            batch_size (int): Number of episodes contained in the arrays, 
                default 1.
        Outputs:
            ep_obs (numpy.ndarray): Zero-filled array that will contain 
                observations, dim=(ep_length + 1, batch_size, nb_agents, 
                obs_dim).
            ep_shared_obs (numpy.ndarray): Zero-filled array that will contain
                shared observations, dim=(ep_length + 1, batch_size, nb_agents,
                shared_obs_dim).
            ep_acts (numpy.ndarray): Zero-filled array that will contain 
                actions, dim=(ep_length, batch_size, nb_agents, act_dim).
            ep_rews (numpy.ndarray): Zero-filled array that will contain 
                rewards, dim=(ep_length, batch_size, nb_agents, 1).
            ep_dones (numpy.ndarray): Zero-filled array that will contain 
                done states, dim=(ep_length, batch_size, nb_agents, 1).
        """
        ep_obs = np.zeros((
            self.episode_length + 1, 
            batch_size, 
            self.nb_agents,
            self.obs_dim), dtype=np.float32)
        ep_shared_obs = np.zeros((
            self.episode_length + 1, 
            batch_size, 
            self.nb_agents,
            self.shared_obs_dim), dtype=np.float32)
        ep_acts = np.zeros((
            self.episode_length, 
            batch_size, 
            self.nb_agents,
            self.act_dim), dtype=np.float32)
        ep_rews = np.zeros((
            self.episode_length, 
            batch_size, 
            self.nb_agents,
            1), dtype=np.float32)
        ep_dones = np.zeros((
            self.episode_length, 
            batch_size, 
            self.nb_agents,
            1), dtype=np.float32)
        return ep_obs, ep_shared_obs, ep_acts, ep_rews, ep_dones

    def store(self, ep_obs, ep_shared_obs, ep_acts, ep_rews, ep_dones):
        """
        Stores a complete episode sequence in the buffer.
        Inputs:
            ep_obs (numpy.ndarray): List of sequences of observations, 
                one for each agent, shape is defined in 
                self.init_episode_arrays().
            ep_shared_obs (numpy.ndarray): List of sequences of shared
                observations, one for each agent, shape is defined in 
                self.init_episode_arrays().
            ep_acts (numpy.ndarray): List of sequences of actions, 
                one for each agent, shape is defined in 
                self.init_episode_arrays().
            ep_rews (numpy.ndarray): List of sequences of rewards, 
                one for each agent, shape is defined in 
                self.init_episode_arrays().
            ep_dones (numpy.ndarray): List of sequences of done states, 
                one for each agent, shape is defined in 
                self.init_episode_arrays().
        """
        n_entries = ep_obs[0].shape[1]

        # Roll the buffers if needed
        if self.curr_i + n_entries > self.buffer_size:
            n_roll = self.buffer_size - self.curr_i
            self.obs_buff = np.roll(self.obs_buff, n_roll, axis=1)
            self.shared_obs_buff = np.roll(
                self.shared_obs_buff, n_roll, axis=1)
            self.act_buff = np.roll(self.act_buff, n_roll, axis=1)
            self.rew_buff = np.roll(self.rew_buff, n_roll, axis=1)
            self.done_buff = np.roll(self.done_buff, n_roll, axis=1)
            self.filled_i = self.buffer_size
            self.curr_i = 0
        
        # Store episodes
        self.obs_buff[:, self.curr_i:self.curr_i + n_entries, :] = ep_obs
        self.shared_obs_buff[:, self.curr_i:self.curr_i + n_entries, :] = \
            ep_shared_obs
        self.act_buff[:, self.curr_i:self.curr_i + n_entries, :] = ep_acts
        self.rew_buff[:, self.curr_i:self.curr_i + n_entries, :] = ep_rews
        self.done_buff[:, self.curr_i:self.curr_i + n_entries, :] = ep_dones

        self.curr_i += n_entries
        if self.filled_i < self.buffer_size:
            self.filled_i += n_entries
        if self.curr_i == self.buffer_size:
            self.curr_i = 0

    def sample(self, batch_size, device=None, ids=None):
        """
        Returns a batch of experienced episodes.
        Inputs:
            batch_size (int): Number of episodes to sample.
            device (str): Device to put the samples.
            ids (numpy.ndarray): Indexes of transitions to sample (used in 
                prioritized experience replay).
        Outputs:
            obs_batch (torch.Tensor): Batch of observations, 
                dim=(nb_agents, ep_length + 1, batch_size, obs_dim).
            shared_obs_batch (torch.Tensor): Batch of shared observations, 
                dim=(nb_agents, ep_length + 1, batch_size, shared_obs_dim).
            act_batch (torch.Tensor): Batch of actions, 
                dim=(nb_agents, ep_length, batch_size, act_dim).
            rew_batch (torch.Tensor): Batch of rewards, 
                dim=(nb_agents, ep_length, batch_size, 1).
            done_batch (torch.Tensor): Batch of done states, 
                dim=(nb_agents, ep_length, batch_size, 1).
        """
        if ids is None:
            ids = np.random.choice(self.filled_i, batch_size)

        obs_batch = self.obs_buff[:, ids]
        shared_obs_batch = self.shared_obs_buff[:, ids]
        act_batch = self.act_buff[:, ids]
        rew_batch = self.rew_buff[:, ids]
        done_batch = self.done_buff[:, ids]

        # Transform to tensors
        if device is not None:
            cast = lambda x: Tensor(x.transpose(2, 0, 1, 3)).to(device)
        else:
            cast = lambda x: Tensor(x.transpose(2, 0, 1, 3))
        obs_batch = cast(obs_batch)
        shared_obs_batch = cast(shared_obs_batch)
        act_batch = cast(act_batch)
        rew_batch = cast(rew_batch)
        done_batch = cast(done_batch)

        return obs_batch, shared_obs_batch, act_batch, rew_batch, done_batch


class LanguageBuffer:

    def __init__(self, max_steps):
        self.max_steps = max_steps

        self.obs_buffer = []
        self.sent_buffer = []

    def store(self, obs_list, sent_list):
        for obs, sent in zip(obs_list, sent_list):
            if len(self.obs_buffer) == self.max_steps:
                self.obs_buffer.pop(0)
                self.sent_buffer.pop(0)
            self.obs_buffer.append(obs)
            self.sent_buffer.append(" ".join(sent))

    def sample(self, batch_size):
        if batch_size > len(self.obs_buffer):
            batch_size = len(self.obs_buffer)
        obs_batch = []
        sent_batch = []
        nb_sampled = 0
        while nb_sampled < batch_size:
            index = random.randrange(len(self.obs_buffer))
            obs = self.obs_buffer[index]
            sent = self.sent_buffer[index]
            if sent in sent_batch:
                continue
            else:
                obs_batch.append(obs)
                sent_batch.append(sent.split(" "))
                nb_sampled += 1
        return obs_batch, sent_batch