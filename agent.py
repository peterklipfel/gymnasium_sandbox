from abc import ABC, abstractmethod

class Agent(ABC):
    def __init__(self, env):
        self.env = env
        super().__init__()

    @abstractmethod
    def action(observation, reward=None):
        raise(f'{cls} does not implement #action method')
