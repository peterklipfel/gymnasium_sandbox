from abc import ABC, abstractmethod

class Agent(ABC):
    def __init__(self, env):
        super().__init__()
        self.env = env

    @abstractmethod
    def action(observation, reward=None):
        raise(f'{cls} does not implement #action method')
