
import  warnings, os
from stable_baselines3 import PPO
import os, sys
import numpy as np
import gym
import gym_qcart

if __name__ == "__main__":
    potential = str(sys.argv[1])

    log_dir = f"example/data"
    model = PPO.load(f"{log_dir}/{potential}_1Meas")
    filter_model = PPO.load(f"{log_dir}/{potential}_filter")

    if potential == "linear":
        env = gym.make('qcart-v0', potential = 'quadratic', k = np.pi, system = 'quantum', controller = 'rlc', estimator = 'rle', filter_model = filter_model)    
    elif potential == "cosine":        
        env = gym.make('qcart-v0', potential = 'cosine', k = 67, system = 'quantum', controller = 'rlc', estimator = 'rle', filter_model = filter_model)
    elif potential == "quartic":
        env = gym.make('qcart-v0', potential = 'quartic', k = np.pi/100, system = 'quantum', controller = 'rlc', estimator = 'rle', filter_model = filter_model)

    index = 0
    baseline = []
    obs = env.reset()    

    resets = []
    for i in range(int(10000)):
        action, _state = model.predict(obs, deterministic=True)
        obs, reward, done, info = env.step(action)
        index += 1

        if done:
            baseline.append(env.number_of_steps_taken)
            resets.append(env.number_of_steps_taken)
            obs = env.reset()
            index = 0
            

    print(f"time steps mean: {np.mean(resets)} and variance: {np.std(resets)/np.sqrt(len(resets))}")
    #python -m example.testc linear