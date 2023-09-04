
import math, warnings, os, time

warnings.filterwarnings('ignore', category=DeprecationWarning)
warnings.filterwarnings('ignore', category=FutureWarning)
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
from tensorflow.python.util import deprecation
deprecation._PRINT_DEPRECATION_WARNINGS = False
import tensorflow as tf

# import qcartpole2 as qcart
from stable_baselines3 import PPO
# import imageio
import time, os, sys, h5py
import numpy as np
import gym
import gym_qcart
# from sb3_contrib import RecurrentPPO, TRPO

if __name__ == "__main__":
    N_meas = int(sys.argv[1])
    start = time.time()

    log_dir = f"D:/Programming/Data/qcart/paper5/124/"
    # log_dir = f"/Users/kai/Desktop/Work/data/qcart/paper4/{folder}/Models"
    # log_dir = f"C:/Users/Kaime/Desktop/Work/Data/QCart/paper5/174/"
    # log_dir= "/home/kmeinerz/Desktop/Data/qcart/paper5/128/"
    filter_model_name = f'c_filter_ppo_sig0.7_dt0.0032_L0.05_Nmeas1_factor0_kickfactor1.0_mem1_r1_0'#f'{env_string}' + 
    if os.path.exists(f"{log_dir}/FilterModels/{filter_model_name}/{filter_model_name}") == True:
        filter_model = PPO.load(f"{log_dir}/FilterModels/{filter_model_name}/{filter_model_name}")
    else:
        filter_model = None

    # env = gym.make('qcart-v1', potential = 'cosine', k = 67, system = 'quantum', controller = 'rlc', estimator = 'rle', filter_model = filter_model)
    env = gym.make('qcart-v0', potential = 'quartic', k = np.pi/100, system = 'quantum', controller = 'rlc', estimator = 'rle', filter_model = filter_model)
    # env = gym.make('qcart-v1', potential = 'quadratic', k = np.pi, system = 'quantum', controller = 'rlc', estimator = 'rle', filter_model = filter_model)    

    # log_dir = f"D:/Programming/Data/qcart/paper5/177/"
    file_name = f'ppo_sig0.7_dt0.0032_L0.05_Nmeas{N_meas}_factor0_kickfactor1.0_mem1_r1_0_0'
    model = PPO.load(f"{log_dir}/Models_cfm/{N_meas}/{file_name}/{file_name}")

    

    index = 0
    baseline = []
    obs = env.reset()    
    obs, reward, done, info = env.step(0)
    # env.seed(3)
    obs = env.reset()
    start = time.time()
    

    # lstm_states = None
    # num_envs = 1
    # # Episode start signals are used to reset the lstm states
    # episode_starts = np.ones((num_envs,), dtype=bool)

    resets = []
    for i in range(int(100000//N_meas)):

        action, _state = model.predict(obs, deterministic=True)
        obs, reward, done, info = env.step(action)
        # obs, reward, done, info = env.step(0)
        index += 1

        if done:
            baseline.append(env.number_of_steps_taken)
            resets.append(env.number_of_steps_taken)
            obs = env.reset()
            index = 0
            

    print(time.time() - start, np.mean(resets), np.std(resets)/np.sqrt(len(resets)), len(resets))#, env.testing)