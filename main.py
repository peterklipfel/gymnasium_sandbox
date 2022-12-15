# SNN-related imports
import snntorch as snn
from snntorch import spikeplot as splt
from snntorch import spikegen

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

import matplotlib.pyplot as plt
import numpy as np
import itertools

# Environment imports
import gymnasium as gym

## OOPy bits
from abc import ABC, abstractmethod


# Network Architecture
num_inputs = 4
num_hidden = 1000
num_outputs = 2

# Temporal Dynamics
num_steps = 25
beta = 0.95


# ONLY_LEFT, ONLY_RIGHT, RANDOM, WIGGLE = ['left', 'right', 'random', 'wiggle']
# run_lengths = {ONLY_LEFT: [], ONLY_RIGHT: [], RANDOM: [], WIGGLE: []}

ONLY_LEFT, ONLY_RIGHT, RANDOM, WIGGLE = [0, 1, 2, 3]

class Agent(ABC):
    def __init__(self, env):
        self.env = env
        super().__init__()

    @abstractmethod
    def action(observation, reward=None):
        raise(f'{cls} does not implement #action method')


class SNNAgent(Agent):
    def action(self, observation, reward=None):
        return 1

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

def run_env_with_agent(env, agent_class):
    agent = agent_class(env)

    # the reward algo for cartpole is +1 for every tick that the pole is not
    # reset. We need to track this manually. I would have expected gymnasium
    # to do that for us, but it doesn't
    total_reward_for_run = 0 

    run_lengths = []
    observation, info = env.reset()
    reward = None

    for _ in range(1000):

        action = agent.action(observation, reward)

        # print(env.observation_space.sample())
        observation, reward, terminated, truncated, info = env.step(action)

        total_reward_for_run += reward
        if terminated:
            run_lengths.append(total_reward_for_run)
            total_reward_for_run = 0
        
        if terminated or truncated:
            observation, info = env.reset()
    return run_lengths

def multi_run(agent_classes):
    env = gym.make("CartPole-v1")
    run_lengths = []

    for agent_class in agent_classes:
        agent_run_lengths = run_env_with_agent(env, agent_class)
        run_lengths.append(agent_run_lengths)

    print(run_lengths)

    env.close()

multi_run([LeftAgent, RightAgent, RandomAgent, WiggleAgent])