import matplotlib.pyplot as plt
import numpy as np
import itertools

# Environment imports
import gymnasium as gym

from basic_agents import LeftAgent, RightAgent, WiggleAgent, RandomAgent
from dqn_agent import DQNAgent


def run_env_with_agent(env, agent_class):
    agent = agent_class(env)

    # the reward algo for cartpole is +1 for every tick that the pole is not
    # reset. We need to track this manually. I would have expected gymnasium
    # to do that for us, but it doesn't
    steps_in_current_run = 0

    run_lengths = []
    observation, info = env.reset()

    for _ in range(100000):
        steps_in_current_run += 1

        action = agent.action(observation)

        next_obs, reward, terminated, truncated, info = env.step(action)

        if terminated or truncated:
            run_lengths.append(steps_in_current_run)
            steps_in_current_run = 0
            reward = -reward # ensure negative reinforcement
        
        agent.save_observation(observation, action, reward, next_obs, terminated)

        observation = next_obs

        if terminated or truncated:
            observation, info = env.reset()
            print(f"average: {np.average(run_lengths)} over {len(run_lengths)} runs")
        agent.train()
        
    return run_lengths

def multi_run(agent_classes):
    env = gym.make("CartPole-v1")
    run_lengths = []

    for agent_class in agent_classes:
        agent_run_lengths = run_env_with_agent(env, agent_class)
        run_lengths.append(agent_run_lengths)

    print(run_lengths)

    env.close()

# multi_run([LeftAgent, RightAgent, RandomAgent, WiggleAgent])

multi_run([DQNAgent])

# multi_run([WiggleAgent])

# from tensorflow.python.client import device_lib
# print(device_lib.list_local_devices())