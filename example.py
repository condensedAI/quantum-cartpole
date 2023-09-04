from stable_baselines3 import PPO

import os, sys

import gym
import gym_qcart
import numpy as np
import matplotlib.pyplot as plt
import torch as th

from stable_baselines3.common.results_plotter import load_results, ts2xy, plot_results
from stable_baselines3.common.callbacks import BaseCallback

def create_folder(name):
    # Create folder if needed
    if name is not None:
        os.makedirs(name, exist_ok=True)

class CustomSaveOnBestNSteps(BaseCallback):
    """
    saves the run with the highest mean reward
    Callback for saving a model (the check is done every ``check_freq`` steps)
    based on the training reward (in practice, we recommend using ``EvalCallback``).

    :param check_freq:
    :param log_dir: Path to the folder where the model will be saved.
      It must contains the file created by the ``Monitor`` wrapper.
    :param verbose: Verbosity level.
    """
    def __init__(self, check_freq: int, log_dir: str,  env, mean : int, file_name: str = "best_model", verbose: int = 1):
        super(CustomSaveOnBestNSteps, self).__init__(verbose)
        self.check_freq = check_freq
        self.log_dir = log_dir
        self.save_path = os.path.join(log_dir, file_name)
        self.best_mean_reward = -np.inf
        self.mean = mean
        self.last = 0
        self.env = env
        self.continue_training = True

    def _on_step(self) -> bool:
        if self.n_calls % self.check_freq == 0:
        #   weights = model.policy.state_dict()
        #   print("start:", weights['value_net.weight'].numpy()[0, 0:10])          
          mean_reward = env.rewards
          run_count = env.run_count
          env.reset_train()

          with open(f"{self.log_dir}/monitor.npy", 'rb') as f2:
              save_mean = np.load(f2)
          with open(f"{self.log_dir}/monitor.npy", 'rb+') as f2:
              np.save(f2, np.append(save_mean, mean_reward))

          if self.verbose > 0:
                print(f"Num timesteps: {self.num_timesteps}")
                print(f"Best mean reward: {self.best_mean_reward:.4f} - Last mean reward per episode: {mean_reward:.4f} - Run Count: {run_count:.4f}")

          self.model.save(self.save_path + '_last')
          if mean_reward > self.best_mean_reward:
              self.best_mean_reward = mean_reward
              # Example for saving best model
              if self.verbose > 0:
                print(f"Saving new best model to {self.save_path}")
              self.model.save(self.save_path)

          if mean_reward < -10000:
              self.continue_training = False

          env.reset()

        return self.continue_training

class CustomSaveOnBestFilter(BaseCallback):
    """
    saves the run with the highest mean reward
    Callback for saving a model (the check is done every ``check_freq`` steps)
    based on the training reward (in practice, we recommend using ``EvalCallback``).

    :param check_freq:
    :param log_dir: Path to the folder where the model will be saved.
      It must contains the file created by the ``Monitor`` wrapper.
    :param verbose: Verbosity level.
    """
    def __init__(self, check_freq: int, log_dir: str,  env, mean : int, file_name: str = "best_model", verbose: int = 1):
        super(CustomSaveOnBestFilter, self).__init__(verbose)
        self.check_freq = check_freq
        self.log_dir = log_dir
        self.save_path = os.path.join(log_dir, file_name)
        self.best_mean_reward = -np.inf
        self.mean = mean
        self.last = 0
        self.env = env
        self.continue_training = True

    def _on_step(self) -> bool:
        if self.n_calls % self.check_freq == 0:
        #   weights = model.policy.state_dict()
        #   print("start:", weights['value_net.weight'].numpy()[0, 0:10])          
          mean_reward = env.rewards
          run_count = env.run_count
          env.reset_train()

          with open(f"{self.log_dir}/monitor.npy", 'rb') as f2:
              save_mean = np.load(f2)
          with open(f"{self.log_dir}/monitor.npy", 'rb+') as f2:
              np.save(f2, np.append(save_mean, mean_reward))

          if self.verbose > 0:
                print(f"Num timesteps: {self.num_timesteps}")
                print(f"Best mean reward: {self.best_mean_reward:.4f} - Last mean reward per episode: {mean_reward:.4f} - Run Count: {run_count:.4f}")

          if mean_reward > self.best_mean_reward:
              self.best_mean_reward = mean_reward
              # Example for saving best model
              if self.verbose > 0:
                print(f"Saving new best model to {self.save_path}")
              self.model.save(self.save_path)
          
          
        #   env.seed(2)
          env.reset()

        #   weights = model.policy.state_dict()
        #   print("end", weights['value_net.weight'].numpy()[0, 0:10])

        return self.continue_training

if __name__ == "__main__":
    log_dir = f"data/"


    ### Training of the RLE 
    env = gym.make('qcart-v0', potential = 'quadratic', k = np.pi, system = 'classical', controller = 'random', estimator = 'estimator_none', state_return = 'state_estimator', filter_model = None)
    file_name = 'filter'
    filepath = log_dir + file_name 
    create_folder(filepath)
    with open(filepath + "/monitor.npy", 'wb') as f2:
        np.save(f2, np.array([]))

    
    clip_range, target_kl, batchsize, n_epochs, learning_rate, n_steps = 0.5, 0.5, 1024, 10, 0.00003, int(1e4)//1024 *1024
    policy_kwargs = dict(activation_fn = th.nn.Tanh, net_arch=[dict(pi=[32,32], vf=[32,32])])

    filter_model = PPO('MlpPolicy', env, verbose=0, use_sde = False,sde_sample_freq=-1, batch_size=batchsize, n_epochs = n_epochs, clip_range=clip_range, learning_rate=learning_rate, n_steps = n_steps, policy_kwargs=policy_kwargs, target_kl = target_kl)#

    callback = CustomSaveOnBestFilter(check_freq=n_steps, log_dir= filepath, env = env, file_name = file_name, mean = 1, verbose = 1)
    filter_model.learn(total_timesteps=5*n_steps, callback=callback)


    ### Training of the RLEC
    env = gym.make('qcart-v0', potential = 'quadratic', k = np.pi, system = 'classical', controller = 'rlc', estimator = 'rle', filter_model = filter_model)
    file_name = 'rlc'
    filepath = log_dir + file_name 
    create_folder(filepath)
    with open(filepath + "/monitor.npy", 'wb') as f2:
        np.save(f2, np.array([]))

    clip_range, target_kl, batchsize, n_epochs, learning_rate, n_steps = 0.5, 0.5, 1024, 10, 0.00003, int(1e4)//1024 *1024
    policy_kwargs = dict(activation_fn = th.nn.Tanh, net_arch=[dict(pi=[32,32], vf=[32,32])])

    filter_model = PPO('MlpPolicy', env, verbose=0, use_sde = False,sde_sample_freq=-1, batch_size=batchsize, n_epochs = n_epochs, clip_range=clip_range, learning_rate=learning_rate, n_steps = n_steps, policy_kwargs=policy_kwargs, target_kl = target_kl)#

    callback = CustomSaveOnBestNSteps(check_freq=n_steps, log_dir= filepath, env = env, file_name = file_name, mean = 1, verbose = 1)
    filter_model.learn(total_timesteps=100*n_steps, callback=callback)
