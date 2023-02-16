import math
import torch
from torch.nn import functional as F

from .networks import MLPNetwork
from .rnd import RND


class E2S_NovelD(RND):
    """ Elliptical Episodic Scaling of NovelD. """

    def __init__(self, input_dim, enc_dim, hidden_dim, 
                 scale_fac=0.5, ridge=0.1, lr=1e-4, device="cpu"):
        super(E2S_NovelD, self).__init__(
            input_dim, enc_dim, hidden_dim, lr, device)
        self.scale_fac = scale_fac
        self.ridge = ridge
        # Last state novelty
        self.last_nov = None
        # Inverse covariance matrix for Elliptical bonus
        self.inv_cov = torch.eye(input_dim).to(device) * (1.0 / self.ridge)
        self.outer_product_buffer = torch.empty(
            input_dim, input_dim).to(device)
    
    def init_new_episode(self):
        self.last_nov = None
        self.inv_cov = torch.eye(self.input_dim).to(self.device)
        self.inv_cov *= (1.0 / self.ridge)

    def set_eval(self, device):
        super().set_eval(device)
        self.inv_cov = self.inv_cov.to(device)
        self.outer_product_buffer = self.outer_product_buffer.to(device)
        
    def get_reward(self, state):
        """
        Get intrinsic reward for the given state.
        Inputs:
            state (torch.Tensor): State from which to generate the reward,
                dim=(1, state_dim).
        Outputs:
            int_reward (float): Intrinsic reward for the input state.
        """
        # Get RND reward as novelty
        nov = super().get_reward(state)

        # Compute reward
        if self.last_nov is not None:
            int_reward = max(nov - self.scale_fac * self.last_nov, 0.0)
        else:
            int_reward = 0.0

        self.last_nov = nov

        # Compute the elliptic scale
        u = torch.mv(self.inv_cov, state.squeeze())
        b = torch.dot(state.squeeze(), u).item()
        # Update covariance matrix
        torch.outer(u, u, out=self.outer_product_buffer)
        torch.add(
            self.inv_cov, self.outer_product_buffer, 
            alpha=-(1. / (1. + b)), out=self.inv_cov)

        elliptic_scale = math.sqrt(2 * b)

        return int_reward * elliptic_scale

class E2S_NovelD_InvDyn(RND):
    """ Elliptical Episodic Scaling of NovelD. """

    def __init__(self, input_dim, act_dim, enc_dim, hidden_dim, 
                 scale_fac=0.5, ridge=0.1, lr=1e-4, device="cpu"):
        super(E2S_NovelD_InvDyn, self).__init__(
            input_dim, enc_dim, hidden_dim, lr, device)
        self.scale_fac = scale_fac
        self.ridge = ridge
        self.enc_dim = enc_dim
        # Last state novelty
        self.last_nov = None
        # State encoder
        self.encoder = MLPNetwork(
            input_dim, enc_dim, hidden_dim, norm_in=False)
        # Inverse dynamics model
        self.inv_dyn = MLPNetwork(
            2 * enc_dim, act_dim, hidden_dim, norm_in=False)
        # Inverse covariance matrix for Elliptical bonus
        self.inv_cov = torch.eye(enc_dim).to(device) * (1.0 / self.ridge)
        self.outer_product_buffer = torch.empty(
            enc_dim, enc_dim).to(device)
         # Optimizers
        self.encoder_optim = torch.optim.Adam(
            self.encoder.parameters(), 
            lr=lr)
        self.inv_dyn_optim = torch.optim.Adam(
            self.inv_dyn.parameters(), 
            lr=lr)
    
    def init_new_episode(self):
        self.last_nov = None
        self.inv_cov = torch.eye(self.enc_dim).to(self.device)
        self.inv_cov *= (1.0 / self.ridge)

    def set_train(self, device):
        super().set_train(device)
        self.encoder.train()
        self.encoder = self.encoder.to(device)
        self.inv_dyn.train()
        self.inv_dyn = self.inv_dyn.to(device)
        self.inv_cov = self.inv_cov.to(device)
        self.outer_product_buffer = self.outer_product_buffer.to(device)
        self.device = device

    def set_eval(self, device):
        super().set_eval(device)
        self.encoder.eval()
        self.encoder = self.encoder.to(device)
        self.inv_cov = self.inv_cov.to(device)
        self.outer_product_buffer = self.outer_product_buffer.to(device)
        
    def get_reward(self, state):
        """
        Get intrinsic reward for the given state.
        Inputs:
            state (torch.Tensor): State from which to generate the reward,
                dim=(1, state_dim).
        Outputs:
            int_reward (float): Intrinsic reward for the input state.
        """
        ## NovelD
        # Get RND reward as novelty
        nov = super().get_reward(state)

        # Compute reward
        if self.last_nov is not None:
            int_reward = max(nov - self.scale_fac * self.last_nov, 0.0)
        else:
            int_reward = 0.0

        self.last_nov = nov

        ## E3B
        # Encode state
        enc_state = self.encoder(state).squeeze().detach()
        # Compute the elliptic scale
        u = torch.mv(self.inv_cov, enc_state)
        b = torch.dot(enc_state, u).item()
        # Update covariance matrix
        torch.outer(u, u, out=self.outer_product_buffer)
        torch.add(
            self.inv_cov, self.outer_product_buffer, 
            alpha=-(1. / (1. + b)), out=self.inv_cov)

        elliptic_scale = math.sqrt(2 * b)

        return int_reward * elliptic_scale

    def train(self, state_batch, act_batch):
        """
        Inputs:
            state_batch (torch.Tensor): Batch of states, dim=(episode_length, 
                batch_size, state_dim).
            act_batch (torch.Tensor): Batch of actions, dim=(episode_length, 
                batch_size, action_dim).
        """
        super().train(state_batch, act_batch)
        # Encode states
        enc_all_states_b = self.encoder(state_batch)
        enc_states_b = enc_all_states_b[:-1]
        enc_next_states_b = enc_all_states_b[1:]
        # Run inverse dynamics model
        inv_dyn_inputs = torch.cat((enc_states_b, enc_next_states_b), dim=-1)
        pred_actions = self.inv_dyn(inv_dyn_inputs)
        # Compute loss
        loss = F.mse_loss(pred_actions, act_batch)
        # Backward pass
        self.encoder_optim.zero_grad()
        self.inv_dyn_optim.zero_grad()
        loss.backward()
        self.encoder_optim.step()
        self.inv_dyn_optim.step()
        return loss
