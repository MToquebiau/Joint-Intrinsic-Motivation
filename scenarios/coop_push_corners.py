import random
import numpy as np

from multiagent.scenario import BaseScenario
from multiagent.core import World, Agent, Landmark, Action, Entity

LANDMARK_RADIUS = 0.5
OBJECT_RADIUS = 0.15
OBJECT_MASS = 1.0
AGENT_RADIUS = 0.04
AGENT_MASS = 0.4

def get_dist(pos1, pos2, squared=False):
    dist = np.sum(np.square(pos1 - pos2))
    if squared:
        return dist
    else:
        return np.sqrt(dist)

def obj_callback(agent, world):
    action = Action()
    action.u = np.zeros((world.dim_p))
    action.c = np.zeros((world.dim_c))
    return action

class Object(Entity):
    def __init__(self):
        super(Object, self).__init__()
        # Objects are movable
        self.movable = True

class PushWorld(World):

    corners = [
        np.array([-1.0, -1.0]),
        np.array([1.0, -1.0]),
        np.array([1.0, 1.0]),
        np.array([-1.0, 1.0])
    ]

    def __init__(self, nb_agents, nb_objects, scenario_params):
        super(PushWorld, self).__init__()
        # add agent
        self.nb_agents = nb_agents
        self.agents = [Agent() for i in range(self.nb_agents)]
        # List of objects to push
        self.nb_objects = nb_objects
        self.objects = [Object() for _ in range(self.nb_objects)]
        # Corresponding landmarks
        self.landmarks = [Landmark() for _ in range(self.nb_objects)]
        self.lm_corners = [None] * self.nb_objects
        # Control inertia
        self.damping = 0.8

        self.scenario_params = scenario_params

    @property
    def entities(self):
        return self.agents + self.objects + self.landmarks

    def reset(self):
        for i in range(self.nb_objects):
            self.init_object(i)

    def init_object(self, obj_i, min_dist=0.2, max_dist=1.5):
        # Random color for both entities
        color = np.random.uniform(0, 1, self.dim_color)
        # Object
        self.objects[obj_i].name = 'object %d' % len(self.objects)
        self.objects[obj_i].color = color
        self.objects[obj_i].size = OBJECT_RADIUS
        self.objects[obj_i].initial_mass = OBJECT_MASS
        # Landmark
        self.landmarks[obj_i].name = 'landmark %d' % len(self.landmarks)
        self.landmarks[obj_i].collide = False
        self.landmarks[obj_i].color = color
        self.landmarks[obj_i].size = LANDMARK_RADIUS
        # Set initial positions
        # # Fixed initial pos
        # self.objects[obj_i].state.p_pos = np.zeros(2)
        # self.landmarks[obj_i].state.p_pos = np.array([-0.5, -0.5])
        # return
        if min_dist is not None:
            while True:
                self.objects[obj_i].state.p_pos = np.random.uniform(
                    -1 + OBJECT_RADIUS, 1 - OBJECT_RADIUS, self.dim_p)
                self.landmarks[obj_i].state.p_pos = np.random.uniform(
                    -1 + OBJECT_RADIUS, 1 - OBJECT_RADIUS, self.dim_p)
                dist = get_dist(self.objects[obj_i].state.p_pos, 
                                self.landmarks[obj_i].state.p_pos)
                if dist > min_dist and dist < max_dist:
                    break

    # Integrate state with walls blocking entities on each side
    def integrate_state(self, p_force):
        for i,entity in enumerate(self.entities):
            if not entity.movable: continue
            entity.state.p_vel = entity.state.p_vel * (1 - self.damping)
            if (p_force[i] is not None):
                entity.state.p_vel += (p_force[i] / entity.mass) * self.dt
            if entity.max_speed is not None:
                speed = np.sqrt(np.square(entity.state.p_vel[0]) \
                        + np.square(entity.state.p_vel[1]))
                if speed > entity.max_speed:
                    entity.state.p_vel = entity.state.p_vel / np.sqrt(np.square(entity.state.p_vel[0]) +
                                                                  np.square(entity.state.p_vel[1])) * entity.max_speed
            # Check for wall collision
            temp_pos = entity.state.p_pos + entity.state.p_vel * self.dt
            # West wall
            if temp_pos[0] - entity.size < -1:
                entity.state.p_vel[0] = 0.0
                entity.state.p_pos[0] = -1.0 + entity.size
            # East wall
            if temp_pos[0] + entity.size > 1:
                entity.state.p_vel[0] = 0.0
                entity.state.p_pos[0] = 1.0 - entity.size
            # North wall
            if temp_pos[1] - entity.size < -1:
                entity.state.p_vel[1] = 0.0
                entity.state.p_pos[1] = -1.0 + entity.size
            # South wall
            if temp_pos[1] + entity.size > 1:
                entity.state.p_vel[1] = 0.0
                entity.state.p_pos[1] = 1.0 - entity.size
            entity.state.p_pos += entity.state.p_vel * self.dt
                
        

class Scenario(BaseScenario):

    def make_world(self, nb_agents=4, nb_objects=1, obs_range=0.4, 
                   collision_pen=2.0, reward_done=100, step_penalty=1.0, 
                   obj_lm_dist_range=[1.0, 2.0]):
        self.nb_agents = nb_agents
        self.nb_objects = nb_objects
        self.obs_range = obs_range
        self.collision_pen = collision_pen
        self.reward_done = reward_done
        self.step_penalty = step_penalty
        self.agent_radius = AGENT_RADIUS
        self.object_radius = OBJECT_RADIUS
        self.landmark_radius = LANDMARK_RADIUS
        self.obj_lm_dist_range = obj_lm_dist_range
        world = PushWorld(nb_agents, nb_objects, self.get_params())
        # add agent
        self.nb_agents = nb_agents
        for i, agent in enumerate(world.agents):
            agent.name = 'agent %d' % i
            agent.silent = True
            agent.size = AGENT_RADIUS
            agent.initial_mass = AGENT_MASS
            agent.color = np.array([0.0,0.0,0.0])
            agent.color[i % 3] = 1.0
        # Objects and landmarks
        self.nb_objects = nb_objects
        for i, object in enumerate(world.objects):
            # Random color for both entities
            color = np.random.uniform(0, 1, world.dim_color)
            object.name = 'object %d' % i
            object.color = color
            object.size = OBJECT_RADIUS
            object.initial_mass = OBJECT_MASS
            # Corresponding Landmarks
            world.landmarks[i].name = 'landmark %d' % i
            world.landmarks[i].collide = False
            world.landmarks[i].color = color
            world.landmarks[i].size = LANDMARK_RADIUS
        # Flag for end of episode
        self._done_flag = False
        # make initial conditions
        self.reset_world(world)
        return world
    
    def get_params(self):
        return {
            "nb_agents": self.nb_agents,
            "nb_objects": self.nb_objects,
            "obs_range": self.obs_range,
            "collision_pen": self.collision_pen,
            "reward_done": self.reward_done,
            "step_penalty": self.step_penalty,
            "agent_radius": self.agent_radius,
            "object_radius": self.object_radius,
            "landmark_radius": self.landmark_radius,
            "obj_lm_dist_range": self.obj_lm_dist_range
        }

    def done(self, agent, world):
        # Done if all objects are on their landmarks
        return self._done_flag

    def reset_world(self, world, seed=None, init_pos=None):
        if seed is not None:
            np.random.seed(seed)

        # Check if init positions are valid
        if init_pos is not None:
            if (len(init_pos["agents"]) != self.nb_agents or 
                len(init_pos["objects"]) != self.nb_objects or
                len(init_pos["landmarks"]) != self.nb_objects):
                print("ERROR: The initial positions {} are not valid.".format(
                    init_pos))
                exit(1)

        # Agents' initial pos
        for i, agent in enumerate(world.agents):
            if init_pos is None:
                agent.state.p_pos = np.random.uniform(
                    -1 + agent.size, 1 - agent.size, world.dim_p)
            else:
                agent.state.p_pos = np.array(init_pos["agents"][i])
            agent.state.c = np.zeros(world.dim_c)
        # Objects and landmarks' initial pos
        for i, object in enumerate(world.objects):
            if init_pos is None:
                while True:
                    # Pick a corner randomly
                    c = random.randint(0, 3)
                    world.lm_corners[i] = c
                    world.landmarks[i].state.p_pos = world.corners[c]
                    # Place object
                    object.state.p_pos = np.random.uniform(
                        -1 + OBJECT_RADIUS, 1 - OBJECT_RADIUS, world.dim_p)
                    dist = get_dist(object.state.p_pos, 
                                    world.landmarks[i].state.p_pos)
                    if (self.obj_lm_dist_range is None  or 
                        (dist > self.obj_lm_dist_range[0] and 
                         dist < self.obj_lm_dist_range[1])):
                        break
            else:
                object.state.p_pos = np.array(init_pos["objects"][i])
                world.landmarks[i].state.p_pos = np.array(init_pos["landmarks"][i])
        # Set initial velocity
        for entity in world.entities:
            entity.state.p_vel = np.zeros(world.dim_p)
        self._done_flag = False

    def reward(self, agent, world):
        dists = [get_dist(obj.state.p_pos, 
                          world.landmarks[i].state.p_pos)
                    for i, obj in enumerate(world.objects)]

        rew = -self.step_penalty

        # Reward if task complete
        self._done_flag = all(d <= LANDMARK_RADIUS for d in dists)
        if self._done_flag:
            rew += self.reward_done

        # Penalty for collision between agents
        if agent.collide:
            for other_agent in world.agents:
                if other_agent is agent: continue
                dist = get_dist(agent.state.p_pos, other_agent.state.p_pos)
                dist_min = agent.size + other_agent.size
                if dist <= dist_min:
                    rew -= self.collision_pen

        return rew
        
    def observation(self, agent, world):
        """
        Observation:
         - Agent state: position, velocity
         - Other agents: [distance x, distance y, v_x, v_y]
         - Other agents and objects:
            - If in sight: [1, distance x, distance y, v_x, v_y]
            - If not: [0, 0, 0, 0, 0]
         - Landmarks:
            - If in sight: [1, distance x, distance y]
            - If not: [0, 0, 0]
        => Full observation dim = 2 + 2 + 4 x (nb_agents - 1) + 5 x (nb_objects - 1) + 3 x (nb_landmarks)
        All distances are divided by max_distance to be in [0, 1]
        """
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
        for obj in world.objects:
            if get_dist(agent.state.p_pos, obj.state.p_pos) <= self.obs_range:
                obs.append(np.concatenate((
                    [1.0], # Bit saying entity is observed
                    (obj.state.p_pos - agent.state.p_pos) / self.obs_range, # Relative position normalised into [0, 1]
                    obj.state.p_vel # Velocity
                )))
            else:
                obs.append(np.array([0.0, 1.0, 1.0, 0.0, 0.0]))
        for lm_i, lm in enumerate(world.landmarks):
            if get_dist(agent.state.p_pos, lm.state.p_pos) <= self.obs_range + LANDMARK_RADIUS:
                obs.append(np.array([1.0, float(world.lm_corners[lm_i] + 1)], 
                      #(lm.state.p_pos - agent.state.p_pos) / self.obs_range, # Relative position normailised into [0, 1]
                ))
            else:
                obs.append(np.array([0.0, 0.0]))

        return np.concatenate(obs)