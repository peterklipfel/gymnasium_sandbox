import random
import time

import gym
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from stable_baselines3.common.buffers import ReplayBuffer
# from torch.utils.tensorboard import SummaryWriter

import torch
import torch.nn as nn

from agent import Agent

class DQNAgent(Agent, nn.Module):
    def __init__(self, env):
        super().__init__(env)

        self.network = nn.Sequential(
            nn.Linear(np.array(env.observation_space.shape).prod(), 120),
            nn.ReLU(),
            nn.Linear(120, 84),
            nn.ReLU(),
            nn.Linear(84, env.action_space.n),
        )

    def forward(self, x):
        return self.network(x)
    
    def action(observation, reward=None):
        return 0

def linear_schedule(start_e: float, end_e: float, duration: int, t: int):
    slope = (end_e - start_e) / duration
    return max(slope * t + start_e, end_e)


SEED = 1337
LEARNING_RATE = 2.5e-4
BUFFER_SIZE = 10000
START_E = 1
END_E = 0.05
TIME_STEPS = 500000
EXPLORATION_FRACTION = 0.5
LEARNING_STARTS = 10000
TRAIN_FREQUENCY = 10
GAMMA = 0.99
TARGET_NETWORK_FREQUENCY = 500
BATCH_SIZE = 128

if __name__ == "__main__":
    run_name = f"cartpole__{SEED}__{int(time.time())}"

    # TRY NOT TO MODIFY: seeding
    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    # torch.backends.cudnn.deterministic = args.torch_deterministic

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # env setup
    # env = gym.make("CartPole-v1", render_mode="human")
    env = gym.make("CartPole-v1")

    q_network = DQNAgent(env).to(device)
    optimizer = optim.Adam(q_network.parameters(), lr=LEARNING_RATE)
    target_network = DQNAgent(env).to(device)
    target_network.load_state_dict(q_network.state_dict())

    rb = ReplayBuffer(
        BUFFER_SIZE,
        env.observation_space,
        env.action_space,
        device,
        handle_timeout_termination=True,
    )
    start_time = time.time()

    # TRY NOT TO MODIFY: start the game
    obs, info = env.reset()
    for global_step in range(TIME_STEPS):
        # ALGO LOGIC: put action logic here
        epsilon = linear_schedule(START_E, END_E, EXPLORATION_FRACTION * TIME_STEPS, global_step)
        if random.random() < epsilon:
            action = env.action_space.sample()
        else:
            q_values = q_network(torch.Tensor(obs).to(device))
            action = torch.argmax(q_values, dim=0).cpu().item()

        # TRY NOT TO MODIFY: execute the game and log data.
        next_obs, reward, terminated, done, info = env.step(action)

        # TRY NOT TO MODIFY: save data to reply buffer; handle `terminal_observation`
        real_next_obs = next_obs.copy()
        if(terminated or done):
            real_next_obs, info = env.reset()
        # if done:
        #     print(info)
        #     real_next_obs = info["terminal_observation"]
        
        rb.add(obs, real_next_obs, action, reward, done, [info]) # `info` is in an array because this version of replayBuffer expects multiple parallel runs

        # TRY NOT TO MODIFY: CRUCIAL step easy to overlook
        obs = next_obs

        # ALGO LOGIC: training.
        if global_step > LEARNING_STARTS:
            if global_step % TRAIN_FREQUENCY == 0:
                data = rb.sample(BATCH_SIZE)
                with torch.no_grad():
                    import pdb; pdb.set_trace()
                    target_max, _ = target_network(data.next_observations).max(dim=0)
                    print(target_max)
                    td_target = data.rewards + GAMMA * target_max * (1 - data.dones)
                    # print(target_network(data.next_observations))
                old_val = q_network(data.observations).gather(1, data.actions).squeeze()
                # print(f'{td_target} ----- {old_val}')
                loss = F.mse_loss(td_target, old_val)

                # optimize the model
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

            # update the target network
            if global_step % TARGET_NETWORK_FREQUENCY == 0:
                target_network.load_state_dict(q_network.state_dict())

    env.close()
    # writer.close()