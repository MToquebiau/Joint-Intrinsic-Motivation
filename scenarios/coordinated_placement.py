import numpy as np
import random

from multiagent.scenario import BaseScenario
from multiagent.core import Walled_World, Agent, Landmark, Action, Entity

AGENT_RADIUS = 0.045
AGENT_MASS = 0.4

def get_dist(pos1, pos2, squared=False):
    dist = np.sum(np.square(pos1 - pos2))
    if squared:
        return dist
    else:
        return np.sqrt(dist)
        

class Button(Landmark):

    def __init__(self, radius):
        super(Button, self).__init__()
        self.collide = False
        self.color_name = None
        self.radius = radius

    def is_pushing(self, agent_pos):
        return get_dist(agent_pos, self.state.p_pos) < self.radius


class PushButtons(Walled_World):

    colors = {
        "red": np.array([0.835, 0.369, 0.0]),
        "green": np.array([0.337, 0.706, 0.91]),
        "blue": np.array([0.925, 0.878, 0.196]),
        "yellow": np.array([1.0, 1.0, 0.0]),
        "pink": np.array([0.0, 1.0, 1.0])
    }

    def __init__(self, nb_agents, nb_buttons, button_radius, scenario_params):
        super(PushButtons, self).__init__()
        # add agent
        self.nb_agents = nb_agents
        self.agents = [Agent() for i in range(nb_agents)]
        # Buttons
        self.nb_buttons = nb_buttons
        self.buttons = [Button(button_radius) for i in range(nb_buttons * 2)]
        self.colors_pushed = {
            list(self.colors.keys())[i]: 0 for i in range(nb_buttons)}
        # Control inertia
        self.damping = 0.8
        # Global reward at each step
        self.global_reward = 0.0
        self.scenario_params = scenario_params

    @property
    def entities(self):
        return self.agents + self.buttons

    def step(self):
        super().step()
        self.colors_pushed = {
            list(self.colors.keys())[i]: 0 for i in range(self.nb_buttons)}
        for b in self.buttons:
            for a in self.agents:
                if b.is_pushing(a.state.p_pos):
                    self.colors_pushed[b.color_name] += 1
                    break


class Scenario(BaseScenario):

    def make_world(self, nb_agents=2, nb_buttons=3, obs_range=2.83, 
                   collision_pen=3.0, button_radius=0.1):
        self.nb_agents = nb_agents
        self.nb_buttons = nb_buttons
        self.obs_range = obs_range
        self.collision_pen = collision_pen
        self.agent_radius = AGENT_RADIUS
        self.button_radius = button_radius;
        # Create world
        world = PushButtons(nb_agents, self.nb_buttons, self.button_radius, self.get_params())
        # Init world entities
        for i, agent in enumerate(world.agents):
            agent.name = 'agent %d' % i
            agent.silent = True
            agent.size = self.agent_radius
            agent.initial_mass = AGENT_MASS
            agent.accel = 4.0
            agent.color = np.array([0.0, 0.0, 0.0])
            agent.color += i / nb_agents
        for button in world.buttons:
            button.size = self.button_radius
        # make initial conditions
        self.reset_world(world)
        return world
    
    def get_params(self):
        return {
            "nb_agents": self.nb_agents,
            "nb_buttons": self.nb_buttons,
            "obs_range": self.obs_range,
            "collision_pen": self.collision_pen,
            "agent_radius": self.agent_radius,
            "button_radius": self.button_radius
        }

    def done(self, agent, world):
        return False

    def reset_world(self, world, seed=None, init_pos=None):
        if seed is not None:
            np.random.seed(seed)

        # Agents' initial pos
        # agent_positions = [
        #     [-0.2, 0.0],
        #     [0.0, 0.0],
        #     [0.2, 0.0]]
        agent_positions = [
            [random.uniform(-1 + self.agent_radius, -self.agent_radius), 0.0],
            [random.uniform(self.agent_radius, 1 - self.agent_radius), 0.0],
            [0.2, 0.0]]
        for i, agent in enumerate(world.agents):
            agent.state.p_pos = np.array(agent_positions[i])
            agent.state.c = np.zeros(world.dim_c)
        # Buttons
        button_positions = [-0.5, 0.0, 0.5]
        # colors = random.sample([0, 1, 2], 3) + random.sample([0, 1, 2], 3)
        colors = [0, 1, 2, 2, 0, 1]
        for i, button in enumerate(world.buttons):
            y = -0.5 if i < 3 else 0.5
            button.state.p_pos = np.array([button_positions[i % 3], y])
            button.color_name = list(world.colors.keys())[colors[i]]
            button.color = world.colors[button.color_name]
        # Set initial velocity
        for entity in world.entities:
            entity.state.p_vel = np.zeros(world.dim_p)

    def reward(self, agent, world):
        rew = 0.0
        # Red bonus
        if world.colors_pushed["red"] == 2:
            rew += 10.0
        elif world.colors_pushed["green"] == 2:
            rew += 2.0
        elif world.colors_pushed["blue"] == 2:
            rew += 1.0
        # elif world.colors_pushed["red"] == 1:
        #     rew += -1.0
        elif world.colors_pushed["green"] == 1:
            rew += 0.5
        elif world.colors_pushed["blue"] == 1:
            rew += 0.5
        return rew

    def observation(self, agent, world):
        obs = [agent.state.p_pos, agent.state.p_vel]

        for ag in world.agents:
            if ag is agent: continue
            if get_dist(agent.state.p_pos, ag.state.p_pos) <= self.obs_range:
                obs.append(np.concatenate((
                    [1.0],
                    (ag.state.p_pos - agent.state.p_pos) / self.obs_range, # Relative position normailised into [0, 1]
                    ag.state.p_vel # Velocity
                )))
            else:
                obs.append(np.array([0.0, 1.0, 1.0, 0.0, 0.0]))
        for b in world.buttons:
            if get_dist(agent.state.p_pos, b.state.p_pos) <= self.obs_range:
                obs.append(np.concatenate((
                    [1.0], 
                    (b.state.p_pos - agent.state.p_pos) / self.obs_range, # Relative position normailised into [0, 1]
                    b.color
                )))
            else:
                obs.append(np.array([0.0, 1.0, 1.0, 0.0, 0.0, 0.0]))

        return np.concatenate(obs)