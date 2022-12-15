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
    @abstractmethod
    def action(observation, reward=None):
        raise(f'{cls} does not implement #action method')


class SNNAgent(Agent):
    def action(obserfation, reward=None):
        return 1

def run_env_with_policy(env, policy):
    # the reward algo for cartpole is +1 for every tick that the pole is not
    # reset. We need to track this manually. I would have expected gymnasium
    # to do that for us, but it doesn't
    total_reward_for_run = 0 

    run_lengths = []

    observation, info = env.reset()

    for _ in range(1000):

        if policy == ONLY_LEFT:
            action = 0
        elif policy == ONLY_RIGHT:
            action = 1
        elif policy == RANDOM:
            action = env.action_space.sample()
        elif policy == WIGGLE:
            if observation[2] > 0:
                action = 1
            else:
                action = 0
        else:
            raise("Unknown policy")

        # print(env.observation_space.sample())
        observation, reward, terminated, truncated, info = env.step(action)

        total_reward_for_run += reward
        if terminated:
            run_lengths.append(total_reward_for_run)
            total_reward_for_run = 0
        
        if terminated or truncated:
            observation, info = env.reset()
    return run_lengths

def multi_run(policies):
    env = gym.make("CartPole-v1")
    run_lengths = []

    for policy in policies:
        policy_run_lengths = run_env_with_policy(env, policy)
        run_lengths.append(policy_run_lengths)

    print(run_lengths)

    env.close()

multi_run([ONLY_LEFT, ONLY_RIGHT, RANDOM, WIGGLE])