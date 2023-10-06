import imp

def make_env(cfg, sce_conf={}, discrete_action=False):
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
    if cfg.env_path.endswith("magym_PredPrey.py"):
        env = imp.load_source('', cfg.env_path).PredatorPrey(
            n_agents=cfg.magym_n_agents, 
            grid_shape=(cfg.magym_env_size, cfg.magym_env_size), 
            agent_view_mask=(cfg.magym_obs_range, cfg.magym_obs_range),
            n_preys=cfg.magym_n_preys, 
            max_steps=cfg.episode_length)
    else:
        from multiagent.environment import MultiAgentEnv

        # load scenario from script
        scenario = imp.load_source('', cfg.env_path).Scenario()
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
