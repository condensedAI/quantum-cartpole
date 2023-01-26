from stable_baselines3 import PPO

import os, sys, h5py, time

import gym
import gym_qcart
import numpy as np
import matplotlib.pyplot as plt

from stable_baselines3.common import results_plotter
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.results_plotter import load_results, ts2xy, plot_results
from stable_baselines3.common.callbacks import BaseCallback

from stable_baselines3.common.evaluation import evaluate_policy

if __name__ == "__main__":

    # log_dir = f"data/"
    env = gym.make('qcart-v0', types = 'lqr')
    # file_name = f"ts{timesteps}_sigma{sigma}_dt{dt}/"
    # model = PPO.load(f"{log_dir}/{file_name}/{file_name}")

    obs = env.reset()
    reward_mean = 0
    for i in range(10):
        env.reset()
        for j in range(env.termination):
            # action, _state = model.predict(obs, deterministic=True)
            obs ,reward, done, something = env.step(0)
            env.render()

            if done == True:
                break