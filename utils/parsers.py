import numpy as np


# Mother classes
class Parser:
    """ Base Parser """

    def __init__(self, scenar):
        self.scenario = scenar
    
    def parse_obs(self, obs):
        """
        Returns a sentence generated based on the actions of the agent
        """
        raise NotImplementedError

    def position_agent(self, obs):
        sentence = []
        # Position of the agent (at all time)
        sentence.append("Located")
        
        # North / South
        if  obs[1] >= 0.33:
            sentence.append("North")
        elif  obs[1] < -0.33:
            sentence.append("South")
        
        # West / East
        if  obs[0] >= 0.33:
            sentence.append("East")
        elif  obs[0] < -0.33:
            sentence.append("West")
        
        # Center
        if len(sentence) == 1:
            sentence.append("Center")

        return sentence
    
    def get_descriptions(self, obs_list):
        """
        Returns descriptions of all agents' observations.
        Inputs:
            obs_list (list(np.array)): List of observations, one for each agent.
        Output:
            descrs (list(list(str))): List of descriptions, one sentence for 
                each agent.
        """
        descr = [
            self.parse_obs(obs_list[a_i]) 
            for a_i in range(self.nb_agents)]
        return descr


class ColorParser(Parser):
    """ Base class for parser for environment with colors. """  

    colors = ["Red", "Bleu", "Green"]

    def __init__(self, scenar):
        super(ColorParser, self).__init__(scenar)

    # Get the color based on its one-hot array
    def array_to_color(self, array):
        # Get the color based on the array
        idx = np.where(array == 1)[0]
        return self.colors[idx]

