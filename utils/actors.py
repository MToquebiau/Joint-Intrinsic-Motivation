import keyboard
import numpy as np

from abc import ABC, abstractmethod


class Actor(ABC):
    """ Abstract actor in the Cooperative Push Environment. """
    @abstractmethod
    def get_action(self):
        """
        Returns actions to perform in the environment
        """
        raise NotImplementedError


class KeyboardActor(Actor):
    """
    Keyboard actor, acts by pressing keys (z/q/s/d and arrows)
    :param n_agents: (int) Number of agents in the scenario
    """
    def __init__(self, n_agents):
        self.n_agents = n_agents

    def get_action(self):
        actions = np.zeros((self.n_agents, 2))

        if keyboard.is_pressed('z'):
            actions[0] += np.array([0.0, 0.5])
        if keyboard.is_pressed('s'):
            actions[0] += np.array([0.0, -0.5])
        if keyboard.is_pressed('q'):
            actions[0] += np.array([-0.5, 0])
        if keyboard.is_pressed('d'):
            actions[0] += np.array([0.5, 0.0])
        if keyboard.is_pressed('up arrow'):
            actions[1] += np.array([0.0, 0.5])
        if keyboard.is_pressed('down arrow'):
            actions[1] += np.array([0.0, -0.5])
        if keyboard.is_pressed('left arrow'):
            actions[1] += np.array([-0.5, 0])
        if keyboard.is_pressed('right arrow'):
            actions[1] += np.array([0.5, 0.0])

        return actions

class RandomActor(Actor):
    """
    Random actor, acts randomly
    :param n_agents: (int) Number of agents in the scenario
    :param dim_action: (int) Dimension of the actions
    """
    def __init__(self, n_agents, dim_action=2):
        self.n_agents = n_agents
        self.dim_action = dim_action

    def get_action(self):
        if self.dim_action == 2:
            return np.random.uniform(-1, 1, size=(self.n_agents, self.dim_action))
        else:
            return None