import numpy as np
import torch

from .buffer import ReplayBuffer, RecReplayBuffer


def unique(sorted_array):
    """
    More efficient implementation of np.unique for sorted arrays
    :param sorted_array: (np.ndarray)
    :return:(np.ndarray) sorted_array without duplicate elements
    """
    if len(sorted_array) == 1:
        return sorted_array
    left = sorted_array[:-1]
    right = sorted_array[1:]
    uniques = np.append(right != left, True)
    return sorted_array[uniques]


class SegmentTree(object):
    def __init__(self, capacity, operation, neutral_element):
        """
        Build a Segment Tree data structure.
        https://en.wikipedia.org/wiki/Segment_tree
        Can be used as regular array that supports Index arrays, but with two
        important differences:
            a) setting item's value is slightly slower.
               It is O(lg capacity) instead of O(1).
            b) user has access to an efficient ( O(log segment size) )
               `reduce` operation which reduces `operation` over
               a contiguous subsequence of items in the array.
        :param capacity: (int) Total size of the array - must be a power of two.
        :param operation: (lambda (Any, Any): Any) operation for combining elements (eg. sum, max) must form a
            mathematical group together with the set of possible values for array elements (i.e. be associative)
        :param neutral_element: (Any) neutral element for the operation above. eg. float('-inf') for max and 0 for sum.
        """
        assert capacity > 0 and capacity & (
            capacity - 1) == 0, "capacity must be positive and a power of 2."
        self._capacity = capacity
        self._value = [neutral_element for _ in range(2 * capacity)]
        self._operation = operation
        self.neutral_element = neutral_element

    def _reduce_helper(self, start, end, node, node_start, node_end):
        if start == node_start and end == node_end:
            return self._value[node]
        mid = (node_start + node_end) // 2
        if end <= mid:
            return self._reduce_helper(start, end, 2 * node, node_start, mid)
        else:
            if mid + 1 <= start:
                return self._reduce_helper(start, end, 2 * node + 1, mid + 1, node_end)
            else:
                return self._operation(
                    self._reduce_helper(start, mid, 2 * node, node_start, mid),
                    self._reduce_helper(
                        mid + 1, end, 2 * node + 1, mid + 1, node_end)
                )

    def reduce(self, start=0, end=None):
        """
        Returns result of applying `self.operation`
        to a contiguous subsequence of the array.
            self.operation(arr[start], operation(arr[start+1], operation(... arr[end])))
        :param start: (int) beginning of the subsequence
        :param end: (int) end of the subsequences
        :return: (Any) result of reducing self.operation over the specified range of array elements.
        """
        if end is None:
            end = self._capacity
        if end < 0:
            end += self._capacity
        end -= 1
        return self._reduce_helper(start, end, 1, 0, self._capacity - 1)

    def __setitem__(self, idx, val):
        # indexes of the leaf
        idxs = idx + self._capacity
        self._value[idxs] = val
        if isinstance(idxs, int):
            idxs = np.array([idxs])
        # go up one level in the tree and remove duplicate indexes
        idxs = unique(idxs // 2)
        while len(idxs) > 1 or idxs[0] > 0:
            # as long as there are non-zero indexes, update the corresponding values
            self._value[idxs] = self._operation(
                self._value[2 * idxs],
                self._value[2 * idxs + 1]
            )
            # go up one level in the tree and remove duplicate indexes
            idxs = unique(idxs // 2)

    def __getitem__(self, idx):
        assert np.max(idx) < self._capacity
        assert 0 <= np.min(idx)
        return self._value[self._capacity + idx]


class SumSegmentTree(SegmentTree):
    def __init__(self, capacity):
        super(SumSegmentTree, self).__init__(
            capacity=capacity,
            operation=np.add,
            neutral_element=0.0
        )
        self._value = np.array(self._value)

    def sum(self, start=0, end=None):
        """
        Returns arr[start] + ... + arr[end]
        :param start: (int) start position of the reduction (must be >= 0)
        :param end: (int) end position of the reduction (must be < len(arr), can be None for len(arr) - 1)
        :return: (Any) reduction of SumSegmentTree
        """
        return super(SumSegmentTree, self).reduce(start, end)

    def find_prefixsum_idx(self, prefixsum):
        """
        Find the highest index `i` in the array such that
            sum(arr[0] + arr[1] + ... + arr[i - i]) <= prefixsum for each entry in prefixsum
        if array values are probabilities, this function
        allows to sample indexes according to the discrete
        probability efficiently.
        :param prefixsum: (np.ndarray) float upper bounds on the sum of array prefix
        :return: (np.ndarray) highest indexes satisfying the prefixsum constraint
        """
        if isinstance(prefixsum, float):
            prefixsum = np.array([prefixsum])
        assert 0 <= np.min(prefixsum)
        assert np.max(prefixsum) <= self.sum() + 1e-5
        assert isinstance(prefixsum[0], float)

        idx = np.ones(len(prefixsum), dtype=int)
        cont = np.ones(len(prefixsum), dtype=bool)

        while np.any(cont):  # while not all nodes are leafs
            idx[cont] = 2 * idx[cont]
            prefixsum_new = np.where(
                self._value[idx] <= prefixsum, prefixsum - self._value[idx], prefixsum)
            # prepare update of prefixsum for all right children
            idx = np.where(np.logical_or(
                self._value[idx] > prefixsum, np.logical_not(cont)), idx, idx + 1)
            # Select child node for non-leaf nodes
            prefixsum = prefixsum_new
            # update prefixsum
            cont = idx < self._capacity
            # collect leafs
        return idx - self._capacity


class MinSegmentTree(SegmentTree):
    def __init__(self, capacity):
        super(MinSegmentTree, self).__init__(
            capacity=capacity,
            operation=np.minimum,
            neutral_element=float('inf')
        )
        self._value = np.array(self._value)

    def min(self, start=0, end=None):
        """
        Returns min(arr[start], ...,  arr[end])
        :param start: (int) start position of the reduction (must be >= 0)
        :param end: (int) end position of the reduction (must be < len(arr), can be None for len(arr) - 1)
        :return: (Any) reduction of MinSegmentTree
        """
        return super(MinSegmentTree, self).reduce(start, end)


class PrioritizedRecReplayBuffer(RecReplayBuffer):
    def __init__(self, 
            alpha, buffer_size, episode_length, nb_agents, obs_dim, act_dim):
        """ Prioritized replay buffer class for training RNN policies. """
        super(PrioritizedRecReplayBuffer, self).__init__(
            buffer_size, episode_length, nb_agents, obs_dim, act_dim)
        self.alpha = alpha
        it_capacity = 1
        while it_capacity < buffer_size:
            it_capacity *= 2

        self._it_sums = SumSegmentTree(it_capacity)
        self._it_mins = MinSegmentTree(it_capacity)
        self.max_priorities = 1.0

    def store(self, ep_obs, ep_shared_obs, ep_acts, ep_rews, ep_dones):
        """See parent class."""
        super().store(ep_obs, ep_shared_obs, ep_acts, ep_rews, ep_dones)
        n_entries = ep_obs[0].shape[1]
        id_end = self.buffer_size if self.curr_i == 0 else self.curr_i
        id_start = id_end - n_entries
        if id_start < 0:
            print("LOOOOOOL")
        for idx in range(id_start, id_end):
                self._it_sums[idx] = self.max_priorities ** self.alpha
                self._it_mins[idx] = self.max_priorities ** self.alpha

    def _sample_proportional(self, batch_size):
        total = self._it_sums.sum(0, len(self) - 1)
        mass = np.random.random(size=batch_size) * total
        idx = self._it_sums.find_prefixsum_idx(mass)
        return idx

    def sample(self, batch_size, beta=0.0, device=None):
        """
        Returns a batch of experienced episodes.
        Inputs:
            batch_size (int): Number of episodes to sample.
            beta (float): Amount of prioritization to apply.
            device (str): Device to put the samples.
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
            torch_weights (torch.Tensor): Batch of importance weigths, 
                dim=(batch_size).
            batch_ids (numpy.ndarray): Indexes of the samples in the batch, 
                dim=(batch_size).
        """
        batch_ids = self._sample_proportional(batch_size)

        p_min = self._it_mins.min() / self._it_sums.sum()
        max_weight = (p_min * len(self)) ** (-beta)
        p_sample = self._it_sums[batch_ids] / self._it_sums.sum()
        weights = (p_sample * len(self)) ** (-beta) / max_weight

        obs_batch, shared_obs_batch, act_batch, rew_batch, done_batch = \
            super().sample(batch_size, device, ids=batch_ids)

        torch_weights = torch.Tensor(weights)
        if device is not None:
            torch_weights = torch_weights.to(device)

        return obs_batch, shared_obs_batch, act_batch, rew_batch, done_batch, \
            torch_weights, batch_ids

    def update_priorities(self, ids, priorities):
        """
        Update priorities of sampled transitions.
        Inputs:
            ids (list(int)): List of ids of sampled transitions.
            priorities (list(float)) List of updated priorities of transitions
                at `ids`.
        """
        assert len(ids) == len(priorities)
        assert np.min(priorities) > 0
        assert np.min(ids) >= 0
        assert np.max(ids) < len(self)

        self._it_sums[ids] = priorities ** self.alpha
        self._it_mins[ids] = priorities ** self.alpha

        self.max_priorities = max(self.max_priorities, np.max(priorities))