import time
import torch
import numpy as np


def perform_eval_scenar(
        cfg, env, model, init_pos_list=[None], 
        recurrent=False):
    rollout_fn = rnn_eval_episode if recurrent else eval_episode
    tot_return = 0.0
    n_success = 0.0
    tot_ep_length = 0.0
    for ep_i in range(len(init_pos_list)):
        ep_return, ep_length, ep_success, _ = rollout_fn(
            cfg,
            env, 
            model,
            init_pos_list[ep_i])
        tot_return += ep_return
        tot_ep_length += ep_length
        n_success += int(ep_success)
    mean_return = tot_return / len(init_pos_list)
    success_rate = n_success / len(init_pos_list)
    mean_ep_length = tot_ep_length / len(init_pos_list)
    return mean_return, success_rate, mean_ep_length

@torch.no_grad()
def eval_episode(cfg, env, model, init_pos=None, render=False, 
                 step_time=0, verbose=False):
    ep_return = 0.0
    ep_length = cfg.episode_length
    ep_success = False
    traj = []
    # Reset environment with initial positions
    obs = env.reset()#init_pos=init_pos)
    for step_i in range(cfg.episode_length):
        traj.append([list(obs[0][0:2]), list(obs[1][0:2])])
        # rearrange observations to be per agent
        torch_obs = torch.Tensor(np.array(obs))
        actions = model.step(torch_obs)
        if "magym" in cfg.env_path:
            actions = [a.cpu().argmax(-1) for a in actions]
        else:
            actions = [a.cpu().squeeze().data.numpy() for a in actions]
        next_obs, rewards, dones, _ = env.step(actions)
        if verbose:
            print("Obs", obs)
            print("Action", actions)
            print("Rewards", rewards)

        if render:
            time.sleep(step_time)
            env.render()

        ep_return += rewards[0]
        if dones[0]:
            ep_length = step_i + 1
            ep_success = True
            break
        obs = next_obs

    return ep_return, ep_length, ep_success, traj

@torch.no_grad()
def rnn_eval_episode(cfg, env, model, init_pos=None, render=False,
        step_time=0, verbose=False):
    ep_return = 0.0
    ep_length = cfg.episode_length
    ep_success = False
    traj = []
    # Get initial last actions and hidden states
    last_actions, qnets_hidden_states = model.get_init_model_inputs()
    # Reset environment with initial positions
    obs = env.reset()#init_pos=init_pos)
    for step_i in range(cfg.episode_length):
        traj.append([list(obs[0][0:2]), list(obs[1][0:2])])
        # Get actions
        actions, qnets_hidden_states = model.get_actions(
            obs, last_actions, qnets_hidden_states)
        last_actions = actions
        if "magym" in cfg.env_path:
            actions = [a.cpu().argmax(-1) for a in actions]
        else:
            actions = [a.cpu().squeeze().data.numpy() for a in actions]
        next_obs, rewards, dones, _ = env.step(actions)
        if verbose:
            print("Obs", obs)
            print("Action", actions)
            print("Rewards", rewards)

        if render:
            time.sleep(step_time)
            env.render()

        ep_return += rewards[0]
        if any(dones):
            ep_length = step_i + 1
            ep_success = True
            break
        obs = next_obs

    return ep_return, ep_length, ep_success, traj