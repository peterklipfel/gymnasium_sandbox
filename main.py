# SNN-related imports
import snntorch as snn
from snntorch import spikeplot as splt
from snntorch import spikegen
from snntorch import utils
from snntorch import surrogate
import snntorch.functional as SF

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

import matplotlib.pyplot as plt
import numpy as np
import itertools

# Environment imports
import gymnasium as gym

from basic_agents import LeftAgent, RightAgent, WiggleAgent, RandomAgent
from agent import Agent


def run_env_with_agent(env, agent_class):
    agent = agent_class(env)

    # the reward algo for cartpole is +1 for every tick that the pole is not
    # reset. We need to track this manually. I would have expected gymnasium
    # to do that for us, but it doesn't
    total_reward_for_run = 0 

    run_lengths = []
    observation, info = env.reset()

    for _ in range(1000):

        action = agent.action(observation, total_reward_for_run)

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
