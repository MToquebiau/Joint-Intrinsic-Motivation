import copy
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

from torch.distributions import Categorical

from .networks import MLPNetwork, get_init_linear
from .utils import soft_update


class DRQNetwork(nn.Module):

    def __init__(self, input_dim, output_dim, hidden_dim=64):
        super(DRQNetwork, self).__init__()
        
        self.mlp_in = get_init_linear(input_dim, hidden_dim)

        self.rnn = nn.GRU(hidden_dim, hidden_dim)

        self.mlp_out = get_init_linear(hidden_dim, output_dim)

    def forward(self, obs, rnn_states):
        """
        Compute q values for every action given observations and rnn states.
        Inputs:
            obs (torch.Tensor): Observations from which to compute q-values,
                dim=(seq_len, batch_size, obs_dim).
            rnn_states (torch.Tensor): Hidden states with which to initialise
                the RNN, dim=(1, batch_size, hidden_dim).
        Outputs:
            q_outs (torch.Tensor): Q-values for every action, 
                dim=(seq_len, batch_size, act_dim).
            new_rnn_states (torch.Tensor): Final hidden states of the RNN, 
                dim=(1, batch_size, hidden_dim).
        """
        self.rnn.flatten_parameters()
        rnn_in = self.mlp_in(obs)

        rnn_outs, new_rnn_states = self.rnn(rnn_in, rnn_states)

        q_outs = self.mlp_out(rnn_outs)

        return q_outs, new_rnn_states


class QMixer(nn.Module):

    def __init__(self, nb_agents, input_dim,
            mixer_hidden_dim=32, hypernet_hidden_dim=64, device="cpu"):
        super(QMixer, self).__init__()
        self.nb_agents = nb_agents
        self.input_dim = input_dim
        self.device = device
        self.mixer_hidden_dim = mixer_hidden_dim
        self.hypernet_hidden_dim = hypernet_hidden_dim

        # Hypernets
        # self.hypernet_weights1 = MLPNetwork(
        #     input_dim, n_agents * mixer_hidden_dim, hypernet_hidden_dim, 0)
        self.hypernet_weights1 = get_init_linear(
            input_dim, nb_agents * mixer_hidden_dim).to(device)
        self.hypernet_bias1 = get_init_linear(
            input_dim, mixer_hidden_dim).to(device)
        # self.hypernet_weights2 = MLPNetwork(
        #     input_dim, mixer_hidden_dim, hypernet_hidden_dim, 0)
        self.hypernet_weights2 = get_init_linear(
            input_dim, mixer_hidden_dim).to(device)
        self.hypernet_bias2 = MLPNetwork(
            input_dim, 1, hypernet_hidden_dim, 0, norm_in=False).to(device)

    def forward(self, local_qs, state):
        """
        Computes Q_tot using local agent q-values and global state.
        Inputs:
            local_qs (torch.Tensor): Local agent q-values, dim=(episode_length, 
                batch_size, nb_agents).
            state (torch.Tensor): Global state, i.e. concatenated local 
                observations, dimension=(episode_length, batch_size, 
                nb_agents * obs_dim)
        Outputs:
            Q_tot (torch.Tensor): Global Q-value computed by the mixer, 
                dim=(episode_length, batch_size, 1, 1).
        """
        batch_size = local_qs.size(1)
        state = state.view(-1, batch_size, self.input_dim).float()
        local_qs = local_qs.view(-1, batch_size, 1, self.nb_agents)

        # First layer forward pass
        w1 = torch.abs(self.hypernet_weights1(state))
        b1 = self.hypernet_bias1(state)
        w1 = w1.view(-1, batch_size, self.nb_agents, self.mixer_hidden_dim)
        b1 = b1.view(-1, batch_size, 1, self.mixer_hidden_dim)
        hidden_layer = F.elu(torch.matmul(local_qs, w1) + b1)

        # Second layer forward pass
        w2 = torch.abs(self.hypernet_weights2(state))
        b2 = self.hypernet_bias2(state)
        w2 = w2.view(-1, batch_size, self.mixer_hidden_dim, 1)
        b2 = b2.view(-1, batch_size, 1, 1)
        out = torch.matmul(hidden_layer, w2) + b2
        # q_tot = out.view(-1, batch_size, 1)
        q_tot = out.squeeze(-1)

        return q_tot


class QMIXAgent:

    def __init__(self, obs_dim, act_dim, 
                 hidden_dim=64, init_explo=1.0, device="cpu"):
        self.epsilon = init_explo
        self.act_dim = act_dim
        self.hidden_dim = hidden_dim
        self.device = device

        # Q function
        self.q_net = DRQNetwork(obs_dim + act_dim, act_dim, hidden_dim).to(device)
        # Target Q function
        self.target_q_net = copy.deepcopy(self.q_net)

    def set_explo_rate(self, explo_rate):
        self.epsilon = explo_rate

    def get_init_hidden(self, batch_size, device="cpu"):
        """
        Returns a zero tensor for initialising the hidden state of the 
        Q-network.
        Inputs:
            batch_size (int): Batch size needed for the tensor.
            device (str): CUDA device to put the tensor on.
        Outputs:
            init_hidden (torch.Tensor): Batch of zero-filled hidden states,
                dim=(1, batch_size, hidden_dim).
        """
        return torch.zeros((1, batch_size, self.hidden_dim), device=device)

    def q_values_from_actions(self, q_batch, action_batch):
        """
        Get Q-values corresponding to actions.
        Inputs:
            q_batch (torch.Tensor): Batch of Q-values, dim=(seq_len, 
                batch_size, act_dim).
            action_batch (torch.Tensor): Batch of one-hot actions taken by the
                agent, dim=(seq_len, batch_size, act_dim).
        Output:
            q_values (torch.Tensor): Q-values in q_batch corresponding to 
                actions in action_batch, dim=(seq_len, batch_size, 1).
        """
        # Convert one-hot actions to index
        action_ids = action_batch.max(dim=-1)[1]
        # Get corresponding Q-values
        q_values = torch.gather(q_batch, 2, action_ids.unsqueeze(dim=-1))
        return q_values

    def get_q_values(self, obs, last_acts, qnet_rnn_states, target=False):
        """
        Returns Q-values computes from given inputs.
        Inputs:
            obs (torch.Tensor): Agent's observation batch, dim=([seq_len], 
                batch_size, obs_dim).
            last_acts (torch.Tensor): Agent's last action batch, 
                dim=([seq_len], batch_size, act_dim).
            qnet_rnn_states (torch.Tensor): Agents' Q-network hidden states
                batch, dim=(1, batch_size, hidden_dim).
            target (bool): Whether to use the target network to compute the 
                Q-values.
        Output:
            q_values (torch.Tensor): Q_values, dim=([seq_len], batch_size, 
                act_dim).
            new_qnet_rnn_states (torch.Tensor): New hidden states of the 
                Q-network, dim=(1, batch_size, hidden_dim).
        """
        # Check if input is a sequence of observations
        no_seq = len(obs.shape) == 2

        # Concatenate observation and last actions
        qnet_input = torch.cat((obs, last_acts), dim=-1)

        if no_seq:
            qnet_input = qnet_input.unsqueeze(0)

        # Get Q-values
        net = self.target_q_net if target else self.q_net
        q_values, new_qnet_rnn_states = net(qnet_input, qnet_rnn_states)

        if no_seq:
            q_values = q_values.squeeze(0)

        return q_values, new_qnet_rnn_states

    def actions_from_q(self, q_values, explore=False):
        """
        Choose actions to take from q_values.
        Inputs:
            q_values (torch.Tensor): Q_values, dim=([seq_len], batch_size, 
                act_dim).
            explore (bool): Whether to perform exploration or exploitation.
        Outputs:
            onehot_actions (torch.Tensor): Chosen actions, dim=([seq_len], 
                batch_size, act_dim).
            greedy_Qs (torch.Tensor): Q-values corresponding to greedy actions,
                dim=([seq_len], batch_size).
        """
        batch_size = q_values.shape[-2]
        # Choose actions
        greedy_Qs, greedy_actions = q_values.max(dim=-1)
        if explore:
            # Sample random number for each action
            rands = torch.rand(batch_size)
            take_random = (rands < self.epsilon).int().to(self.device)
            # Get random actions
            rand_actions = Categorical(
                logits=torch.ones(batch_size, self.act_dim)
            ).sample().to(self.device)
            # Choose actions
            actions = (1 - take_random) * greedy_actions + \
                      take_random * rand_actions
            onehot_actions = torch.eye(self.act_dim)[actions.to("cpu")]
        else:
            onehot_actions = torch.eye(self.act_dim)[greedy_actions.to("cpu")]
        
        return onehot_actions.to(self.device), greedy_Qs

    def get_actions(self, obs, last_acts, qnet_rnn_states, explore=False):
        """
        Returns an action chosen using the Q-network.
        Inputs:
            obs (torch.Tensor): Agent's observation batch, dim=([seq_len], 
                batch_size, obs_dim).
            last_acts (torch.Tensor): Agent's last action batch, 
                dim=([seq_len], batch_size, act_dim).
            qnet_rnn_states (torch.Tensor): Agents' Q-network hidden states
                batch, dim=(1, batch_size, hidden_dim).
            explore (bool): Whether to perform exploration or exploitation.
        Output:
            onehot_actions (torch.Tensor): Chosen actions, dim=([seq_len], 
                batch_size, act_dim).
            greedy_Qs (torch.Tensor): Q-values corresponding to greedy actions,
                dim=([seq_len], batch_size).
            new_qnet_rnn_states (torch.Tensor): New agent's Q-network hidden 
                states dim=(1, batch_size, hidden_dim).
        """
        # Compute Q-values
        q_values, new_qnet_rnn_states = self.get_q_values(
            obs, last_acts, qnet_rnn_states)

        onehot_actions, greedy_Qs = self.actions_from_q(q_values, explore)
        return onehot_actions, greedy_Qs, new_qnet_rnn_states

    def get_params(self):
        return {'q_net': self.q_net.state_dict(),
                'target_q_net': self.target_q_net.state_dict()}

    def load_params(self, params):
        self.q_net.load_state_dict(params['q_net'])
        self.target_q_net.load_state_dict(params['target_q_net'])


class QMIX:
    """
    Class implementing the QMIX algorithm, from Rashid et al., "QMIX: Monotonic
        Value Function Factorisation for Deep Multi-Agent Reinforcement 
        Learning", in ICML 2018.
    """
    def __init__(self, nb_agents, obs_dim, act_dim, lr, 
                 gamma=0.99, tau=0.01, hidden_dim=64, shared_params=False, 
                 init_explo_rate=1.0, max_grad_norm=None, device="cpu",
                 use_per=False, per_nu=0.9, per_eps=1e-6):
        self.nb_agents = nb_agents
        self.obs_dim = obs_dim
        self.act_dim = act_dim
        self.lr = lr
        self.gamma = gamma
        self.tau = tau
        self.hidden_dim = hidden_dim
        self.shared_params = shared_params
        self.max_grad_norm = max_grad_norm
        self.device = device
        self.use_per = use_per
        self.per_nu = per_nu
        self.per_eps = per_eps

        # Create agent policies
        if not shared_params:
            self.agents = [QMIXAgent(
                    obs_dim, 
                    act_dim,
                    hidden_dim, 
                    init_explo_rate,
                    device)
                for _ in range(nb_agents)]
        else:
            self.agents = [QMIXAgent(
                    obs_dim, 
                    act_dim,
                    hidden_dim, 
                    init_explo_rate,
                    device)]

        # Create Q-mixer network
        mixer_in_dim = nb_agents * obs_dim
        self.mixer = QMixer(nb_agents, mixer_in_dim, device=device)
        # Target Q-mixer
        self.target_mixer = copy.deepcopy(self.mixer)

        # Initiate optimiser with all parameters
        self.parameters = []
        for ag in self.agents:
            self.parameters += ag.q_net.parameters()
        self.parameters += self.mixer.parameters()
        self.optimizer = torch.optim.RMSprop(self.parameters, lr)

    def get_actions(self, 
            obs_list, last_actions, qnets_hidden_states, explore=False):
        """
        Returns each agent's action given their observation.
        Inputs:
            obs_list (list(numpy.ndarray)): List of agent observations.
            explore (bool): Whether to explore or not.
            last_actions (list(torch.Tensor)): List of last actions.
            qnets_hidden_states (torch.Tensor)): List of agents' Q-network 
                hidden states.
        Outputs:
            actions (list(torch.Tensor)): Each agent's chosen action.
            new_qnets_hidden_states (list(torch.Tensor)): New hidden states.
        """
        if self.shared_params:
            obs = torch.Tensor(np.array(obs_list)).to(self.device)
            last_actions = torch.cat(last_actions)
            qnets_hidden_states = torch.cat(qnets_hidden_states)
            actions_batch, _, new_qnets_hidden_states = self.agents[0].get_actions(
                obs, last_actions, qnets_hidden_states, explore)
            actions = [actions_batch[a_i] for a_i in range(self.nb_agents)]
            self.last_actions = actions_batch
            self.qnets_hidden_states = new_qnets_hidden_states
        else:
            actions = []
            new_qnets_hidden_states = []
            for a_i in range(self.nb_agents):
                obs = torch.Tensor(obs_list[a_i]).unsqueeze(0).to(self.device)
                action, _, new_qnet_hidden_state = self.agents[a_i].get_actions(
                    obs, 
                    last_actions[a_i], 
                    qnets_hidden_states[a_i],
                    explore
                )
                actions.append(action)
                new_qnets_hidden_states.append(new_qnet_hidden_state)
        return actions, new_qnets_hidden_states

    def train_on_batch(self, batch):
        if self.use_per:
            obs_b, shared_obs_b, act_b, rew_b, done_b, imp_wght_b, ids = batch
        else:
            obs_b, shared_obs_b, act_b, rew_b, done_b = batch

        batch_size = obs_b.shape[2]

        agent_qs = []
        agent_nqs = []
        for a_i in range(self.nb_agents):
            agent = self.agents[0] if self.shared_params else self.agents[a_i]

            obs_ag = obs_b[a_i]
            shared_obs_ag = shared_obs_b[a_i]
            act_ag = act_b[a_i]
            rew_ag = rew_b[a_i]
            done_ag = done_b[a_i]

            prev_act_ag = torch.cat((
                torch.zeros(1, batch_size, self.act_dim).to(self.device),
                act_ag
            ))

            q_values, _ = agent.get_q_values(
                obs_ag, 
                prev_act_ag, 
                agent.get_init_hidden(batch_size, self.device)
            )
            
            action_qs = agent.q_values_from_actions(q_values, act_ag)
            agent_qs.append(action_qs)

            # Get Q-values of next state following the Double Q-Network update
            # rule: get greedy action from main Q-net and estimate its value
            # with the target Q-net.
            with torch.no_grad():
                greedy_actions, _ = agent.actions_from_q(q_values)
                target_qs, _ = agent.get_q_values(
                    obs_ag, 
                    prev_act_ag, 
                    agent.get_init_hidden(batch_size, self.device),
                    target=True
                )
                target_next_action_qs = agent.q_values_from_actions(
                    target_qs, greedy_actions)[1:]
                agent_nqs.append(target_next_action_qs)

        # Combine agent Q-value batchs to feed in the mixer
        agent_qs = torch.cat(agent_qs, dim=-1)
        agent_nqs = torch.cat(agent_nqs, dim=-1)

        # Get current and next step Q_tot
        Q_tot = self.mixer(agent_qs, shared_obs_ag[:-1])
        next_Q_tot = self.target_mixer(agent_nqs, shared_obs_ag[1:])

        # Compute mean between agents' individual rewards
        global_rew_b = torch.mean(rew_b, dim=0)

        # Compute Q-targets
        Q_tot_targets = global_rew_b + self.gamma * next_Q_tot
        # Compute Bellman error
        error = Q_tot - Q_tot_targets.detach()

        # Compute MSE loss
        if self.use_per:
            mse_error = (error ** 2).sum(dim=0).flatten()
            imp_wght_error = mse_error * imp_wght_b
            loss = imp_wght_error.sum()
            err = error.abs().cpu().detach().numpy()
            new_priorities = ((1 - self.per_nu) * err.mean(axis=0) +
                self.per_nu * err.max(axis=0)).flatten() + self.per_eps
        else:
            loss = (error ** 2).sum()
            new_priorities = None

        # Backward and gradient step
        self.optimizer.zero_grad()
        loss.backward()
        if self.max_grad_norm is not None:
            torch.nn.utils.clip_grad_norm_(
                self.parameters, self.max_grad_norm)
        self.optimizer.step()

        return loss.item(), new_priorities

    def get_init_model_inputs(self):
        """ 
        Returns zero-filled tensord for last actions and Q-network hidden 
        states.
        """
        last_actions = [
            torch.zeros(1, self.act_dim, device=self.device)
        ] * self.nb_agents
        qnets_hidden_states = [
            self.agents[0].get_init_hidden(1, self.device)
        ] * self.nb_agents
        return last_actions, qnets_hidden_states

    def set_explo_rate(self, explo_rate):
        """
        Set exploration rate for each agent
        Inputs:
            explo_rate (float): New exploration rate.
        """
        for a in self.agents:
            a.set_explo_rate(explo_rate)

    def prep_training(self, device='cpu'):
        for a in self.agents:
            a.device = device
            a.q_net.train()
            a.q_net = a.q_net.to(device)
        self.device = device

    def prep_rollouts(self, device='cpu'):
        for a in self.agents:
            a.device = device
            a.q_net.eval()
            a.q_net = a.q_net.to(device)
        self.device = device

    def update_all_targets(self):
        """ Soft update the target networks. """
        for a in self.agents:
            soft_update(a.target_q_net, a.q_net, self.tau)
        soft_update(self.target_mixer, self.mixer, self.tau)

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
            'agent_params': [a.get_params() for a in self.agents],
            'mixer_params': self.mixer.state_dict(),
            'target_mixer_params': self.target_mixer.state_dict(),
            'optimizer': self.optimizer.state_dict()
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
        instance = cls(**save_dict)
        for a, params in zip(instance.agents, agent_params):
            a.load_params(params)
        instance.mixer.load_state_dict(mixer_params)
        instance.target_mixer.load_state_dict(target_mixer_params)
        instance.optimizer.load_state_dict(optimizer)
        return instance
