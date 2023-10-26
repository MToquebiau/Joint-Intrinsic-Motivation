from abc import ABC, abstractmethod

class IntrinsicReward(ABC):
    """ Abstract class for an Intrinsic Reward Model. """
    
    @abstractmethod
    def init_new_episode(self):
        """
        Initialise model at start of new episode.
        """
        raise NotImplementedError
    
    @abstractmethod
    def set_train(self, device):
        """
        Set to training mode and put networks on given device.
        Inputs:
            device (str): CUDA device.
        """
        raise NotImplementedError
    
    @abstractmethod
    def set_eval(self, device):
        """
        Set to evaluation mode and put networks on given device.
        Inputs:
            device (str): CUDA device.
        """
        raise NotImplementedError
    
    @abstractmethod
    def get_reward(self, state):
        """
        Returns the reward computed from given state.
        Inputs:
            state (torch.Tensor): State used for computing reward, 
                dim=(1, state_dim).
        """
        raise NotImplementedError
    
    @abstractmethod
    def train(self, state_batch, act_batch):
        """
        Set to evaluation mode and put networks on given device.
        """
        raise NotImplementedError
        
    @abstractmethod
    def get_params(self):
        """
        Returns state dicts of networks and optimizers.
        """
        raise NotImplementedError
        
    @abstractmethod
    def load_params(self, params):
        """
        Load parameters in networks and optimizers.
        Inputs:
            params (dict): Dictionary of state dicts.
        """
        raise NotImplementedError


class NoIntrinsicReward(IntrinsicReward):
    """ Placeholder class for models with no intrinsic reward. """
    def __init__(self, *args, **kwargs):
        pass
    
    def init_new_episode(self):
        pass

    def set_train(self, device):
        pass

    def set_eval(self, device):
        pass
        
    def get_reward(self, state):
        return 0.0
    
    def train(self, *args):
        return 0.0
    
    def get_params(self):
        return {}

    def load_params(self, params):
        pass