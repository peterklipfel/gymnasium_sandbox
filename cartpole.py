if __name__ == "__main__":
    env = gym.make("CartPole-v1", render_mode="human")

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
            print(steps_in_current_run)
            steps_in_current_run = 0
            reward = -reward # ensure negative reinforcement
        
        agent.save_observation(observation, action, reward, next_obs, terminated)

        observation = next_obs

        if terminated or truncated:
            observation, info = env.reset()
            agent.train()
