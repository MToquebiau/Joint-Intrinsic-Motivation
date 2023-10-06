import os
import imp
import git
import json
import time
import torch
import random
import argparse
import numpy as np
import pandas as pd

from tensorboardX import SummaryWriter
from tqdm import tqdm

from models.qmix_intrinsic import QMIX_IR
from utils.buffer import RecReplayBuffer
from utils.prio_buffer import PrioritizedRecReplayBuffer
from utils.make_env import make_env, make_env_parser
from utils.utils import get_paths, load_scenario_config, write_params
from utils.eval import perform_eval_scenar
from utils.decay import ParameterDecay

def run(cfg):
    # Get paths for saving logs and model
    run_dir, model_cp_path, log_dir = get_paths(cfg)
    print("Saving model in dir", run_dir)

    # Init summary writer
    logger = SummaryWriter(str(log_dir))

    # Load scenario config
    sce_conf = load_scenario_config(cfg, run_dir)

    torch.manual_seed(cfg.seed)
    np.random.seed(cfg.seed)
    random.seed(cfg.seed)

    # Set training device
    if torch.cuda.is_available():
        if cfg.cuda_device is None:
            device = 'cuda'
        else:
            device = torch.device(cfg.cuda_device)
    else:
        device = 'cpu'

    # Create environment
    if "rel_overgen.py" in cfg.env_path:
        env = imp.load_source('', cfg.env_path).RelOvergenEnv(
            cfg.state_dim,
            cfg.ro_n_agents,
            cfg.optimal_reward,
            cfg.optimal_diffusion_coeff,
            cfg.suboptimal_reward,
            cfg.suboptimal_diffusion_coeff,
            cfg.save_visited_states)
        obs_dim = env.obs_dim
        act_dim = env.act_dim
        nb_agents = cfg.ro_n_agents
    else:
        env = make_env(cfg, sce_conf, discrete_action=True)
        obs_dim = env.observation_space[0].shape[0]
        act_dim = env.action_space[0].n
        nb_agents = env.n_agents if hasattr(env, 'n_agents') \
            else sce_conf["nb_agents"]

    # Save args in txt file
    write_params(run_dir, cfg, env)

    # Create model
    if cfg.intrinsic_reward_algo == "none":
        intrinsic_reward_params = {}
    elif "noveld" == cfg.intrinsic_reward_algo:
        intrinsic_reward_params = {
            "enc_dim": cfg.int_rew_enc_dim,
            "hidden_dim": cfg.int_rew_hidden_dim,
            "lr": cfg.int_rew_lr,
            "scale_fac": cfg.scale_fac,
            "device": device}
    elif "rnd" == cfg.intrinsic_reward_algo:
        intrinsic_reward_params = {
            "enc_dim": cfg.int_rew_enc_dim,
            "hidden_dim": cfg.int_rew_hidden_dim,
            "lr": cfg.int_rew_lr,
            "device": device}
    elif "e3b" == cfg.intrinsic_reward_algo:
        if cfg.intrinsic_reward_mode == "central":
            ir_act_dim = nb_agents * act_dim
        else:
            ir_act_dim = act_dim
        intrinsic_reward_params = {
            "act_dim": ir_act_dim,
            "enc_dim": cfg.int_rew_enc_dim,
            "hidden_dim": cfg.int_rew_hidden_dim,
            "ridge": cfg.ridge,
            "lr": cfg.int_rew_lr,
            "device": device}
    elif "e2srnd" == cfg.intrinsic_reward_algo:
        intrinsic_reward_params = {
            "enc_dim": cfg.int_rew_enc_dim,
            "hidden_dim": cfg.int_rew_hidden_dim,
            "ridge": cfg.ridge,
            "lr": cfg.int_rew_lr,
            "device": device}
    elif "e2snoveld" in cfg.intrinsic_reward_algo:
        if cfg.intrinsic_reward_mode == "central":
            ir_act_dim = nb_agents * act_dim
        else:
            ir_act_dim = act_dim
        intrinsic_reward_params = {
            "enc_dim": cfg.int_rew_enc_dim,
            "act_dim": ir_act_dim,
            "hidden_dim": cfg.int_rew_hidden_dim,
            "ridge": cfg.ridge,
            "scale_fac": cfg.scale_fac,
            "lr": cfg.int_rew_lr,
            "device": device}
        if "-llec" in cfg.intrinsic_reward_algo:
            intrinsic_reward_params["ablation"] = "LLEC"
            cfg.intrinsic_reward_algo = "e2snoveld"
        elif "-eec" in cfg.intrinsic_reward_algo:
            intrinsic_reward_params["ablation"] = "EEC"
            cfg.intrinsic_reward_algo = "e2snoveld"
    qmix = QMIX_IR(nb_agents, obs_dim, act_dim, cfg.lr, cfg.gamma, cfg.tau, 
            cfg.hidden_dim, cfg.shared_params, cfg.init_explo_rate,
            cfg.max_grad_norm, device, cfg.use_per, cfg.per_nu, cfg.per_eps,
            cfg.intrinsic_reward_mode, cfg.intrinsic_reward_algo, 
            intrinsic_reward_params)
    qmix.prep_rollouts(device=device)
    
    # Intrinsic reward coefficient
    if cfg.int_reward_decay_fn != "constant":
        int_reward_coeff = ParameterDecay(
            cfg.int_reward_coeff, 
            cfg.int_reward_coeff / 10, 
            cfg.n_frames, 
            cfg.int_reward_decay_fn, 
            cfg.int_reward_decay_smooth)

    # Create replay buffer
    if cfg.use_per:
        beta = ParameterDecay(cfg.per_beta_start, 1.0, cfg.n_frames)
        buffer = PrioritizedRecReplayBuffer(
            cfg.per_alpha, 
            cfg.buffer_length, 
            cfg.episode_length, 
            nb_agents, 
            obs_dim, 
            act_dim)
    else:
        buffer = RecReplayBuffer(
            cfg.buffer_length, 
            cfg.episode_length, 
            nb_agents, 
            obs_dim, 
            act_dim)

    # Get number of exploration steps
    if cfg.n_explo_frames is None:
        cfg.n_explo_frames = cfg.n_frames
    # Set epsilon decay function
    eps_decay = ParameterDecay(
        cfg.init_explo_rate, cfg.final_explo_rate, 
        cfg.n_explo_frames, cfg.epsilon_decay_fn)

    # Setup evaluation scenario
    if cfg.eval_every is not None:
        if cfg.eval_scenar_file is not None:
            # Load evaluation scenario
            with open(cfg.eval_scenar_file, 'r') as f:
                eval_scenar = json.load(f)
        else:
            eval_scenar = [None]
        eval_data_dict = {
            "Step": [],
            "Mean return": [],
            "Success rate": [],
            "Mean episode length": []
        }

    # Start training
    print(f"Starting training for {cfg.n_frames} frames")
    print(f"                  updates every {cfg.frames_per_update} frames")
    print(f"                  with seed {cfg.seed}")
    train_data_dict = {
        "Step": [],
        "Episode return": [],
        "Episode extrinsic return": [],
        "Episode intrinsic return": [],
        "Success": [],
        "Episode length": []
    }
    # Reset episode data and environment
    ep_step_i = 0
    ep_ext_returns = np.zeros(nb_agents)
    ep_int_returns = np.zeros(nb_agents)
    ep_success = False
    obs = env.reset()
    qmix.reset_int_reward(obs)
    # Init episode data for saving in replay buffer
    ep_obs, ep_shared_obs, ep_acts, ep_rews, ep_dones = \
        buffer.init_episode_arrays()
    # Get initial last actions and hidden states
    last_actions, qnets_hidden_states = qmix.get_init_model_inputs()
    for step_i in tqdm(range(cfg.n_frames), ncols=0):
        qmix.set_explo_rate(eps_decay.get_param(step_i))

        # Get actions
        actions, qnets_hidden_states = qmix.get_actions(
            obs, last_actions, qnets_hidden_states, explore=True)
        last_actions = actions
        if "magym" in cfg.env_path:
            actions = [a.cpu().argmax(-1) for a in actions]
        else:
            actions = [a.cpu().squeeze().data.numpy() for a in actions]
        next_obs, ext_rewards, dones, _ = env.step(actions)

        # Compute intrinsic rewards
        int_rewards = qmix.get_intrinsic_rewards(next_obs)
        if cfg.int_reward_decay_fn == "constant":
            coeff = cfg.int_reward_coeff
        else:
            coeff = int_reward_coeff.get_param(step_i)
        rewards = np.array([ext_rewards]) + coeff * np.array([int_rewards])
        rewards = rewards.T

        # Save experience for replay buffer
        ep_obs[ep_step_i, 0, :] = np.stack(obs)
        ep_shared_obs[ep_step_i, 0, :] = np.tile(
            np.concatenate(obs), (nb_agents, 1))
        ep_acts[ep_step_i, 0, :] = np.stack(actions)
        ep_rews[ep_step_i, 0, :] = rewards
        ep_dones[ep_step_i, 0, :] = np.vstack(dones)

        ep_ext_returns += ext_rewards
        ep_int_returns += int_rewards
        if any(dones):
            ep_success = True
        
        # Check for end of episode
        if ep_success or ep_step_i + 1 == cfg.episode_length:
            # Store next state observations for last step
            ep_obs[ep_step_i + 1, 0, :] = np.stack(next_obs)
            ep_shared_obs[ep_step_i + 1, 0, :] = np.tile( 
                np.concatenate(next_obs), (nb_agents, 1))
            # Store episode in replay buffer
            buffer.store(ep_obs, ep_shared_obs, ep_acts, ep_rews, ep_dones)
            # Log training data
            train_data_dict["Step"].append(step_i)
            train_data_dict["Episode return"].append(
                np.sum(ep_rews) / nb_agents)
            train_data_dict["Episode extrinsic return"].append(
                np.mean(ep_ext_returns))
            train_data_dict["Episode intrinsic return"].append(
                np.mean(ep_int_returns))
            train_data_dict["Success"].append(int(ep_success))
            train_data_dict["Episode length"].append(ep_step_i + 1)
            # Log Tensorboard
            logger.add_scalar(
                'agent0/episode_return', 
                train_data_dict["Episode return"][-1], 
                train_data_dict["Step"][-1])
            logger.add_scalar(
                'agent0/episode_ext_return', 
                train_data_dict["Episode extrinsic return"][-1], 
                train_data_dict["Step"][-1])
            logger.add_scalar(
                'agent0/episode_int_return', 
                train_data_dict["Episode intrinsic return"][-1], 
                train_data_dict["Step"][-1])
            # Reset episode data
            ep_ext_returns = np.zeros(nb_agents)
            ep_int_returns = np.zeros(nb_agents)
            ep_step_i = 0
            ep_success = False
            # Init episode data for saving in replay buffer
            ep_obs, ep_shared_obs, ep_acts, ep_rews, ep_dones = \
                buffer.init_episode_arrays()
            # Get initial last actions and hidden states
            last_actions, qnets_hidden_states = qmix.get_init_model_inputs()
            # Reset environment
            obs = env.reset()
            qmix.reset_int_reward(obs)
        else:
            ep_step_i += 1
            obs = next_obs

        # Training
        if ((step_i + 1) % cfg.frames_per_update == 0 and
                len(buffer) >= cfg.batch_size):
            qmix.prep_training(device=device)
            # Get samples
            if cfg.use_per:
                b = beta.get_param(step_i)
                sample_batch = buffer.sample(cfg.batch_size, b, device)
            else:
                sample_batch = buffer.sample(cfg.batch_size, device)
            # Train
            qmix_loss, int_reward_loss, new_prio = qmix.train(sample_batch)

            if cfg.use_per:
                buffer.update_priorities(sample_batch[-1], new_prio)

            loss_dict = {
                "qtot_loss": qmix_loss,
                "int_reward_loss": int_reward_loss}
            # Log
            logger.add_scalars('agent0/losses', loss_dict, step_i)
            qmix.update_all_targets()
            qmix.prep_rollouts(device=device)
            
        # Evaluation
        if cfg.eval_every is not None and (step_i + 1) % cfg.eval_every == 0:
            eval_return, eval_success_rate, eval_ep_len = perform_eval_scenar(
                cfg, env, qmix, recurrent=True)
            eval_data_dict["Step"].append(step_i + 1)
            eval_data_dict["Mean return"].append(eval_return)
            eval_data_dict["Success rate"].append(eval_success_rate)
            eval_data_dict["Mean episode length"].append(eval_ep_len)
            # Save eval data
            eval_df = pd.DataFrame(eval_data_dict)
            eval_df.to_csv(str(run_dir / 'evaluation_data.csv'))
            # Reset environment
            obs = env.reset()
            qmix.reset_int_reward(obs)

        # Save model
        if (step_i + 1) % cfg.save_interval == 0:
            os.makedirs(run_dir / 'incremental', exist_ok=True)
            qmix.save(run_dir / 'incremental' / ('model_ep%i.pt' % (step_i)))
            qmix.save(model_cp_path)
            qmix.prep_rollouts(device=device)
            # Save training data
            train_df = pd.DataFrame(train_data_dict)
            train_df.to_csv(str(run_dir / 'training_data.csv'))

    env.close()
    # Save model
    qmix.save(model_cp_path)
    # Log Tensorboard
    logger.export_scalars_to_json(str(log_dir / 'summary.json'))
    logger.close()
    # Save training and eval data
    train_df = pd.DataFrame(train_data_dict)
    train_df.to_csv(str(run_dir / 'training_data.csv'))
    if cfg.eval_every is not None:
        eval_df = pd.DataFrame(eval_data_dict)
        eval_df.to_csv(str(run_dir / 'evaluation_data.csv'))
    if "rel_overgen.py" in cfg.env_path and cfg.save_visited_states:
        with open(str(run_dir / "visited_states.json"), 'w') as f:
            json.dump(env.visited_states[:-1], f)
    print("Model saved in dir", run_dir)



if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--env_path", type=str, help="Path to the environment",
                    default="algorithms/JIM/scenarios/push_buttons.py")
    parser.add_argument("--model_name", type=str, default="qmix_TEST",
                        help="Name of directory to store model/training contents")
    parser.add_argument("--seed", default=1, type=int, help="Random seed")
    # Environment
    parser.add_argument("--episode_length", default=100, type=int)
    parser.add_argument("--sce_conf_path", type=str, 
                        default="configs/2a_pol.json",
                        help="Path to the scenario config file")
    # Training
    parser.add_argument("--n_frames", default=2500, type=int,
                        help="Number of training frames to perform")
    parser.add_argument("--frames_per_update", default=100, type=int)
    parser.add_argument("--batch_size", default=32, type=int,
                        help="Number of episodes to sample from replay buffer for training.")
    parser.add_argument("--save_interval", default=100000, type=int)
    # Replay Buffer
    parser.add_argument("--buffer_length", default=5000, type=int,
                        help="Max number of episodes stored in replay buffer.")
    parser.add_argument("--use_per", action="store_true", default=False,
                        help="Whether to use Prioritized Experience Replay.")
    parser.add_argument("--per_alpha", default=0.6, type=float)
    parser.add_argument("--per_nu", default=0.9, type=float,
                        help="Weight of max TD error in formation of PER weights.")
    parser.add_argument("--per_eps", default=1e-6, type=float)
    parser.add_argument("--per_beta_start", default=0.4, type=float)
    # Exploration
    parser.add_argument("--n_explo_frames", default=None, type=int,
                        help="Number of frames where agents explore, if None then equal to n_frames")
    parser.add_argument("--init_explo_rate", default=1.0, type=float)
    parser.add_argument("--final_explo_rate", default=0.001, type=float)
    parser.add_argument("--epsilon_decay_fn", default="linear", type=str,
                        choices=["linear", "exp"])
    # Evalutation
    parser.add_argument("--eval_every", type=int, default=None)
    parser.add_argument("--eval_scenar_file", type=str, default=None)
    # Model hyperparameters
    parser.add_argument("--hidden_dim", default=64, type=int)
    parser.add_argument("--lr", default=0.0007, type=float)
    parser.add_argument("--tau", default=0.005, type=float)
    parser.add_argument("--gamma", default=0.99, type=float)
    parser.add_argument("--shared_params", action='store_true')
    parser.add_argument("--max_grad_norm", type=float, default=0.5,
                        help='Max norm of gradients (default: 0.5)')
    # Intrinsic reward hyperparameters
    parser.add_argument("--intrinsic_reward_mode", default="central",
                        choices=["central", "local"])
    parser.add_argument("--intrinsic_reward_algo", default='none', 
                        choices=['none', 'noveld', 'rnd', 'e3b', 'e2srnd', 'e2snoveld', 'e2snoveld-llec', 'e2snoveld-eec'])
    parser.add_argument("--int_reward_decay_fn", default="constant", type=str, 
                        choices=["constant", "linear", "sigmoid"])
    parser.add_argument("--int_reward_coeff", default=0.1, type=float)
    parser.add_argument("--int_reward_decay_smooth", type=float, default=1.5)
    parser.add_argument("--int_rew_lr", default=1e-4, type=float)
    parser.add_argument("--int_rew_hidden_dim", default=128, type=int)
    parser.add_argument("--int_rew_enc_dim", default=16, type=int)
    parser.add_argument("--scale_fac", default=0.5, type=float)
    parser.add_argument("--ridge", default=0.1)
    # Cuda
    parser.add_argument("--cuda_device", default=None, type=str)
    # Relative Overgeneralisation environment
    parser.add_argument("--state_dim", type=int, default=50)
    parser.add_argument("--ro_n_agents", type=int, default=2)
    parser.add_argument("--optimal_reward", type=float, default=12.0)
    parser.add_argument("--optimal_diffusion_coeff", type=float, default=30.0)
    parser.add_argument("--suboptimal_reward", type=float, default=0.0)
    parser.add_argument("--suboptimal_diffusion_coeff", type=float, 
                        default=0.08)
    parser.add_argument("--save_visited_states", action="store_true",
                         default=False)

    # MA_GYM parameters
    parser.add_argument("--magym_n_agents", type=int, default=4)
    parser.add_argument("--magym_env_size", type=int, default=7)
    parser.add_argument("--magym_obs_range", type=int, default=5)
    parser.add_argument("--magym_n_preys", type=int, default=2)

    config = parser.parse_args()

    run(config)