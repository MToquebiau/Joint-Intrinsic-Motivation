import torch
import numpy as np

from .modules.qmix import QMIX
from .modules.intrinsic_rewards import NoIntrinsicReward
from .modules.noveld import NovelD
from .modules.rnd import RND
from .modules.e3b import E3B
from .modules.e2s_rnd import E2S_RND
from .modules.e2s_noveld import E2S_NovelD_InvDyn

IR_MODELS = {
    "none": NoIntrinsicReward,
    "noveld": NovelD,
    "rnd": RND,
    "e3b": E3B,
    "e2srnd": E2S_RND,
    "e2snoveld": E2S_NovelD_InvDyn
}

class QMIX_IR(QMIX):
    """ 
    Class impelementing QMIX with Intrinsic Rewards, either central or local. 
    """
    def __init__(self, nb_agents, obs_dim, act_dim, lr, 
                 gamma=0.99, tau=0.01, hidden_dim=64, shared_params=False, 
                 init_explo_rate=1.0, max_grad_norm=None, device="cpu", 
                 use_per=False, per_nu=0.9, per_eps=1e-6, 
                 intrinsic_reward_mode="central", intrinsic_reward_algo="none",
                 intrinsic_reward_params={}):
        super(QMIX_IR, self).__init__(
            nb_agents, obs_dim, act_dim, lr, gamma, tau, hidden_dim, 
            shared_params, init_explo_rate, max_grad_norm, device, use_per, 
            per_nu, per_eps)

        self.ir_mode = intrinsic_reward_mode
        self.intrinsic_reward_algo = intrinsic_reward_algo
        self.intrinsic_reward_params = intrinsic_reward_params
        if self.ir_mode == "central":
            self.int_rew = IR_MODELS[intrinsic_reward_algo](
                nb_agents * obs_dim, **intrinsic_reward_params)
        elif self.ir_mode == "local":
            self.int_rew = [
                IR_MODELS[intrinsic_reward_algo](
                    obs_dim, **intrinsic_reward_params)
                for a_i in range(self.nb_agents)]

    def get_intrinsic_rewards(self, next_obs_list):
        """
        Get intrinsic reward of the multi-agent system.
        Inputs:
            next_obs_list (list): List of agents' observations at next 
                step.
        Outputs:
            int_rewards (list): List of agents' intrinsic rewards.
        """
        if self.ir_mode == "central":
            # Concatenate observations
            cat_obs = torch.Tensor(
                np.concatenate(next_obs_list)).unsqueeze(0).to(self.device)
            # Get reward
            int_reward = self.int_rew.get_reward(cat_obs)
            int_rewards = [int_reward] * self.nb_agents
        elif self.ir_mode == "local":
            int_rewards = []
            for a_i in range(self.nb_agents):
                obs = torch.Tensor(
                    next_obs_list[a_i]).unsqueeze(0).to(self.device)
                int_rewards.append(self.int_rew[a_i].get_reward(obs))
        return int_rewards
    
    def train(self, batch):
        """
        Update all agents and Intrinsic reward model.
        Inputs:
            batch (tuple(torch.Tensor)): Tuple of batches of experiences for
                the agents to train on.
        Outputs:
            qtot_loss (float): QMIX loss.
            int_rew_loss (float): Intrinsic reward loss.
        """
        qtot_loss, new_priorities = super().train_on_batch(batch)

        if self.use_per:
            obs_b, shared_obs_b, act_b, _, _, _, _ = batch
        else:
            obs_b, shared_obs_b, act_b, _, _ = batch

        # Intrinsic reward model update
        if self.ir_mode == "central":
            act_b = torch.cat(tuple(act_b), dim=-1)
            int_rew_loss = self.int_rew.train(shared_obs_b[0], act_b)
        elif self.ir_mode == "local":
            losses = [
                self.int_rew[a_i].train(obs_b[a_i], act_b[a_i])
                for a_i in range(self.nb_agents)]
            int_rew_loss = sum(losses) / self.nb_agents

        return qtot_loss, float(int_rew_loss), new_priorities

    def reset_int_reward(self, obs_list):
        if self.ir_mode == "central":
            # Reset intrinsic reward model
            self.int_rew.init_new_episode()
            # Initialise intrinsic reward model with first observation
            cat_obs = torch.Tensor(
                np.concatenate(obs_list)).unsqueeze(0).to(self.device)
            self.int_rew.get_reward(cat_obs.view(1, -1))
        elif self.ir_mode == "local":
            for a_i in range(self.nb_agents):
                # Reset intrinsic reward model
                self.int_rew[a_i].init_new_episode()
                # Initialise intrinsic reward model with first observation
                obs = torch.Tensor(obs_list[a_i]).unsqueeze(0).to(self.device)
                self.int_rew[a_i].get_reward(obs)
    
    def prep_training(self, device='cpu'):
        super().prep_training(device)
        if self.ir_mode == "central":
            self.int_rew.set_train(device)
        elif self.ir_mode == "local":
            for a_int_rew in self.int_rew:
                a_int_rew.set_train(device)
    
    def prep_rollouts(self, device='cpu'):
        super().prep_rollouts(device)
        if self.ir_mode == "central":
            self.int_rew.set_eval(device)
        elif self.ir_mode == "local":
            for a_int_rew in self.int_rew:
                a_int_rew.set_eval(device)

    def _get_ir_params(self):
        if self.ir_mode == "central":
            return self.int_rew.get_params()
        elif self.ir_mode == "local":
            return [a_int_rew.get_params() for a_int_rew in self.int_rew]

    def _load_ir_params(self, params):
        if self.ir_mode == "central":
            self.int_rew.load_params(params)
        elif self.ir_mode == "local":
            for a_int_rew, param in zip(self.int_rew, params):
                a_int_rew.load_params(param)

    def save(self, filename):
        self.prep_training(device='cpu')
        save_dict = {
            'nb_agents': self.nb_agents,
            'obs_dim': self.obs_dim,
            'act_dim': self.act_dim,
            'lr': self.lr,
            'gamma': self.gamma,
            'tau': self.tau,
            'hidden_dim': self.hidden_dim,
            'shared_params': self.shared_params,
            'max_grad_norm': self.max_grad_norm,
            "intrinsic_reward_mode": self.ir_mode,
            "intrinsic_reward_algo": self.intrinsic_reward_algo, 
            "intrinsic_reward_params": self.intrinsic_reward_params,
            'agent_params': [a.get_params() for a in self.agents],
            'mixer_params': self.mixer.state_dict(),
            'target_mixer_params': self.target_mixer.state_dict(),
            'optimizer': self.optimizer.state_dict(),
            'int_reward_params': self._get_ir_params()
        }
        torch.save(save_dict, filename)

    @classmethod
    def init_from_save(cls, filename):
        """
        Instantiate instance of this class from file created by 'save' method
        """
        save_dict = torch.load(filename, map_location=torch.device('cpu'))
        agent_params = save_dict.pop("agent_params")
        mixer_params = save_dict.pop("mixer_params")
        target_mixer_params = save_dict.pop("target_mixer_params")
        optimizer = save_dict.pop("optimizer")
        int_rew_params = save_dict.pop("int_rew_params")
        instance = cls(**save_dict)
        for a, params in zip(instance.agents, agent_params):
            a.load_params(params)
        instance.mixer.load_state_dict(mixer_params)
        instance.target_mixer.load_state_dict(target_mixer_params)
        instance.optimizer.load_state_dict(optimizer)
        instance._load_ir_params(int_rew_params)
        return instance