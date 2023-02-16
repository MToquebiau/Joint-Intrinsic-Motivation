import numpy as np
import json
import imp
import os
import re
from pathlib import Path
from shutil import copyfile

def make_env(scenario_path, sce_conf={}, discrete_action=False):
    '''
    Creates a MultiAgentEnv object as env. This can be used similar to a gym
    environment by calling env.reset() and env.step().
    Use env.render() to view the environment on the screen.

    Input:
        scenario_path   :   path of the scenario script
                            (without the .py extension)
        benchmark       :   whether you want to produce benchmarking data
                            (usually only done during evaluation)

    Some useful env properties (see environment.py):
        .observation_space  :   Returns the observation space for each agent
        .action_space       :   Returns the action space for each agent
        .n                  :   Returns the number of Agents
    '''
    from multiagent.environment import MultiAgentEnv

    # load scenario from script
    scenario = imp.load_source('', scenario_path).Scenario()
    # create world
    world = scenario.make_world(**sce_conf)
    # create multiagent environment
    env = MultiAgentEnv(
        world, scenario.reset_world, scenario.reward, scenario.observation, 
        done_callback=scenario.done if hasattr(scenario, "done") else None,
        discrete_action=discrete_action)
    return env

def make_env_parser(scenario_path, sce_conf={}, discrete_action=False):
    from multiagent.environment import MultiAgentEnv

    # load scenario from script
    scenar_lib = imp.load_source('', scenario_path)
    scenario = scenar_lib.Scenario()

    # create world
    world = scenario.make_world(**sce_conf)
    # create multiagent environment
    env = MultiAgentEnv(
        world, scenario.reset_world, scenario.reward, scenario.observation, 
        done_callback=scenario.done if hasattr(scenario, "done") else None,
        discrete_action=discrete_action)

    # Create parser
    parser_args = [
        sce_conf['nb_agents'], 
        sce_conf['nb_objects'], 
        0.0]
    parser = scenar_lib.ObservationParser(*parser_args)

    return env, parser

def get_paths(config):
    # Get environment name from script path
    env_name = re.findall("\/?([^\/.]*)\.py", config.env_path)[0]
    # Get path of the run directory
    model_dir = Path('./models') / env_name / config.model_name
    if not model_dir.exists():
        curr_run = 'run1'
    else:
        exst_run_nums = [int(str(folder.name).split('run')[1]) for folder in
                         model_dir.iterdir() if
                         str(folder.name).startswith('run')]
        if len(exst_run_nums) == 0:
            curr_run = 'run1'
        else:
            curr_run = 'run%i' % (max(exst_run_nums) + 1)
    run_dir = model_dir / curr_run
    model_cp_path = run_dir / 'model.pt'
    log_dir = run_dir / 'logs'
    if not log_dir.exists():
        os.makedirs(log_dir)

    return run_dir, model_cp_path, log_dir

def load_scenario_config(config, run_dir):
    sce_conf = {}
    if config.sce_conf_path is not None:
        copyfile(config.sce_conf_path, run_dir / 'sce_config.json')
        with open(config.sce_conf_path) as cf:
            sce_conf = json.load(cf)
            print('Special config for scenario:', config.env_path)
            print(sce_conf, '\n')
    return sce_conf