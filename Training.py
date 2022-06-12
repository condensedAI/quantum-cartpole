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

def create_folder(name):
    # Create folder if needed
    if name is not None:
        os.makedirs(name, exist_ok=True)

class SaveOnBestTrainingRewardCallback(BaseCallback):
    """
    Callback for saving a model (the check is done every ``check_freq`` steps)
    based on the training reward (in practice, we recommend using ``EvalCallback``).

    :param check_freq:
    :param log_dir: Path to the folder where the model will be saved.
      It must contains the file created by the ``Monitor`` wrapper.
    :param verbose: Verbosity level.
    """
    def __init__(self, check_freq: int, log_dir: str, file_name: str = "best_model", verbose: int = 1):
        super(SaveOnBestTrainingRewardCallback, self).__init__(verbose)
        self.check_freq = check_freq
        self.log_dir = log_dir
        self.save_path = os.path.join(log_dir, file_name)
        self.best_mean_reward = -np.inf

    def _on_step(self) -> bool:
        if self.n_calls % self.check_freq == 0:

          # Retrieve training reward
          x, y = ts2xy(load_results(self.log_dir), 'timesteps')
          if len(x) > 0:
              # Mean training reward over the last 100 episodes
              mean_reward = np.mean(y[-100:])
              if self.verbose > 0:
                print(f"Num timesteps: {self.num_timesteps}")
                print(f"Best mean reward: {self.best_mean_reward:.2f} - Last mean reward per episode: {mean_reward:.2f}")

              # New best model, you could save the agent here
              if mean_reward > self.best_mean_reward:
                  self.best_mean_reward = mean_reward
                  # Example for saving best model
                  if self.verbose > 0:
                    print(f"Saving new best model to {self.save_path}")
                  self.model.save(self.save_path)

        return True


if __name__ == "__main__":
    timesteps, sigma, dt = int(sys.argv[1]), float(sys.argv[2]), float(sys.argv[3])
    log_dir = f"data/"
    env = gym.make('qcart-v0', sigma = sigma, dt = dt)
    file_name = f"ts{timesteps}_sigma{sigma}_dt{dt}/"
    create_folder(log_dir + file_name)
    
    
    env = Monitor(env, log_dir + file_name)
    model = PPO('MlpPolicy', env, verbose=0)
    callback = SaveOnBestTrainingRewardCallback(check_freq=1000, log_dir=log_dir + file_name, file_name = file_name)
    model.learn(total_timesteps=timesteps, callback=callback)
