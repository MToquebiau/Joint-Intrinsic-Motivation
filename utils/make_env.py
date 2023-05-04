import imp

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
