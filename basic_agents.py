from agent import Agent

class LeftAgent(Agent):
    def action(self, observation, reward=None):
        return 0

class RightAgent(Agent):
    def action(self, observation, reward=None):
        return 0

class RandomAgent(Agent):
    def action(self, observation, reward=None):
        return self.env.action_space.sample()

class WiggleAgent(Agent):
    def action(self, observation, reward=None):
        if observation[2] > 0:
            return 1
        else:
            return 0