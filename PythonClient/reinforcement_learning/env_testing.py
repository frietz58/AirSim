import setup_path
import gym
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.monitor import Monitor
import numpy as np

# Create a DummyVecEnv for main airsim gym env
env = Monitor(
            gym.make(
                "airgym:airsim-my-drone-env-v0",
                ip_address="127.0.0.1",
                step_length=0.25,
            )
    )

obs = env.reset()

for _ in range(100):
    action = env.action_space.sample()
    # action = 1
    new_obs, reward, done, state = env.step(action)  # take a random action
    dp = [round(p, 3) for p in new_obs[0:3]]
    dv = [round(v, 3) for v in new_obs[3:6]]
    print("A: {} | R: {} | D: {} | NP: {} | NV: {}".format(action, reward, done, dp, dv))
    # print(new_obs)
    obs = new_obs

    if done:
        obs = env.reset()


env.close()

