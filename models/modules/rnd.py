import torch
from torch.nn import functional as F

from .networks import MLPNetwork
from .intrinsic_rewards import IntrinsicReward


class RND(IntrinsicReward):
    """ Random Network Distillation. """

    def __init__(self, 
            input_dim, enc_dim, hidden_dim, 
            lr=1e-4, device="cpu"):
        self.input_dim = input_dim
        self.device = device
        # Random Network Distillation network
        # Fixed target embedding network
        self.target = MLPNetwork(
            input_dim, enc_dim, hidden_dim, n_hidden_layers=3, norm_in=False)
        # Predictor embedding network
        self.predictor = MLPNetwork(
            input_dim, enc_dim, hidden_dim, n_hidden_layers=3, norm_in=False)

        # Fix weights of target
        for param in self.target.parameters():
            param.requires_grad = False
        
        # Optimizers
        self.optim = torch.optim.Adam(self.predictor.parameters(), lr=lr)
    
    def init_new_episode(self):
        pass

    def set_train(self, device):
        self.target.train()
        self.target = self.target.to(device)
        self.predictor.train()
        self.predictor = self.predictor.to(device)
        self.device = device

    def set_eval(self, device):
        self.target.eval()
        self.target = self.target.to(device)
        self.predictor.eval()
        self.predictor = self.predictor.to(device)
        self.device = device
        
    def get_reward(self, state):
        """
        Get intrinsic reward for the given state.
        Inputs:
            state (torch.Tensor): State from which to generate the reward,
                dim=(1, state_dim).
        Outputs:
            int_reward (float): Intrinsic reward for the input state.
        """
        # Compute embeddings
        target = self.target(state)
        pred = self.predictor(state)

        # Compute novelty
        int_reward = torch.norm(
            pred.detach() - target.detach(), dim=1, p=2).item()
        
        return int_reward
    
    def train(self, state_batch, act_batch):
        """
        Inputs:
            state_batch (torch.Tensor): Batch of states, dim=(episode_length, 
                batch_size, state_dim).
        """
        # Encode states
        targets = self.target(state_batch)
        preds = self.predictor(state_batch)
        # Compute loss
        loss = F.mse_loss(preds, targets)
        # Backward pass
        self.optim.zero_grad()
        loss.backward()
        self.optim.step()
        return float(loss)
    
    def get_params(self):
        return {'target': self.target.state_dict(),
                'predictor': self.predictor.state_dict(),
                'optim': self.optim.state_dict()}

    def load_params(self, params):
        self.target.load_state_dict(params['target'])
        self.predictor.load_state_dict(params['predictor'])
        self.optim.load_state_dict(params['optim'])