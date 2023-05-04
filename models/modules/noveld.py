import torch

from torch import nn
from torch.optim import Adam

from .networks import MLPNetwork
from .rnd import RND


class NovelD(RND):
    """ 
    Class implementing the NovelD algorithm for intrinsic reward generation, from
        Zhang et al., "NovelD: A Simple yet Effective Exploration Criterion", in
        NeurIPS 2021.
    """
    def __init__(self, input_dim, enc_dim, hidden_dim, 
                 lr=1e-4, scale_fac=0.5, device="cpu"):
        """
        Inputs:
            input_dim (int): Dimension of the input.
            embed_dim (int): Dimension of the output of RND networks.
            hidden_dim (int): Dimension of the hidden layers in MLPs.
            lr (float): Learning rate for training the predictor
                (default=0.0001).
            scale_fac (float): Scaling factor for computing the reward, 
                noted alpha in the paper, controls how novel we want the states
                to be to generate some reward (in [0,1]) (default=0.5).
            device (str): CUDA device.
        """
        super(NovelD, self).__init__(
            input_dim, enc_dim, hidden_dim, lr, device)
        self.scale_fac = scale_fac
        # Last state novelty
        self.last_nov = None
        # Save count of states encountered during each episode
        self.episode_states_count = {}

    def init_new_episode(self):
        self.last_nov = None
        self.episode_states_count = {}

    def is_empty(self):
        return True if len(self.episode_states_count) == 0 else False

    def get_reward(self, state):
        """
        Get intrinsic reward for this new state.
        Inputs:
            state (torch.Tensor): State from which to generate 
                the reward, dim=(1, state_dim).
        Outputs:
            intrinsic_reward (float): Intrinsic reward for the input state.
        """
        state_key = tuple(state[0])
        if state_key in self.episode_states_count:
            self.episode_states_count[state_key] += 1
        else:
            self.episode_states_count[state_key] = 1

        # Return 0 if state has already been seen during the current episode
        if self.episode_states_count[state_key] > 1:
            return 0.0

        # Get RND reward as novelty
        nov = super().get_reward(state)

        # Compute reward
        if self.last_nov is not None:
            intrinsic_reward = max(nov - self.scale_fac * self.last_nov, 0.0)
        else:
            intrinsic_reward = 0.0

        self.last_nov = nov

        return intrinsic_reward