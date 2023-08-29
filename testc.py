
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

    log_dir = f"D:/Programming/Data/qcart/paper5/177/"
    # log_dir = f"/Users/kai/Desktop/Work/data/qcart/paper4/{folder}/Models"
    # log_dir = f"C:/Users/Kaime/Desktop/Work/Data/QCart/paper5/174/"
    # log_dir= "/home/kmeinerz/Desktop/Data/qcart/paper5/128/"
    filter_model_name = f'c_filter_ppo_sig0.5_dt0.003183_L0.05_Nmeas1_factor0_kickfactor1.0_mem1_r1_0_41'#f'{env_string}' + 
    # # print(filter_model_name)
    if os.path.exists(f"{log_dir}/FilterModels/{filter_model_name}/{filter_model_name}") == True:
        filter_model = PPO.load(f"{log_dir}/FilterModels/{filter_model_name}/{filter_model_name}")
    else:
        filter_model = None

    env = gym.make('qcart-v1', system = 'classical', controller = 'lqr', estimator = 'kalman')



    # log_dir = f"/Users/kai/Desktop/Work/data/qcart/paper5/80/Models/"
    file_name = f'ppo_sig0.5_dt0.003183_L0.05_Nmeas{N_meas}_factor0_kickfactor1.0_mem1_r1_0_0'
    model = PPO.load(f"{log_dir}/Models_cnm2/{N_meas}/{file_name}/{file_name}")
    # model = RecurrentPPO.load(f"{log_dir}/{file_name}/{file_name}")

    

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

    tracking = np.zeros((7, 1000))
    # tracking = np.zeros(48)
    max_force = 0
    resets = []
    # if os.path.exists(f"Test2.npy") == True:
    #     with open(f"Test2.npy", 'rb') as f2:
    #         tracking = np.load(f2) 
    # infer = eager_model.signatures["serving_default"]           
    # print(infer)
    for i in range(int(1000000//N_meas)):
    # for i in range(200000):
        # print(i)
        # action, _state = filter_model.predict(obs, deterministic=True)
        action, _state = model.predict(obs, deterministic=True)
        # print(obs, action)
        # print(action, obs)
        # print(obs, action / 0.05 * 2 * 8)
        # print(obs, action)
        # action, _state = model.predict(obs, state=lstm_states, episode_start=episode_starts, deterministic=True)
        # print("action in: ", action)
        # action = infer(tf.convert_to_tensor(env.lstm_input[:,-2:-1,:]))['output_1'].numpy()[0,:]
        # action = infer(tf.convert_to_tensor(env.lstm_input))['output_1'].numpy()[0,-1,:]
        obs, reward, done, info = env.step(action)
        # obs, reward, done, info = env.step(0)
        # print(env.actual_pos)

        # print(action)
        # tracking[:,i] = env.tracking
        # print(env.tracking)
        # tracking[1,i] = env.mom_meas
        # # tracking[i] = env.control[1]
        # tracking[2:4,i] = env.actual_pos 
        # tracking[0,i] = env.estimate_mean[0]
        # tracking[1,i] = env.estimate_mean[1]  
        # tracking[4,i] = env.measurement[0]
        # tracking[5,i] = env.measurement[1]                
        # tracking[6,i] = env.control[1]
        # tracking[7,i] = env.back[1]
        # obs, reward, done, info = env.step(i)
        # max_force = np.maximum(max_force, np.abs(env.force_output) )
        # if env.number_of_steps_taken > 10000:
        # env.render(mode='rgb_array')
        # images.append(img)
        index += 1
        # forces[i] = env.force_output
        # if i%1000 == 0:
        #     print(i)
        if done:
            # print("Reset at", i, (env.number_of_steps_taken*env.N_meas*env.dt)/env.period )
            # print("Reset at", index*N_meas)

            baseline.append(env.number_of_steps_taken)
            resets.append(env.number_of_steps_taken)
            obs = env.reset()
            index = 0
            # if len(resets) == 1000:
            # break
    # print(time.time() - start, np.mean(resets)*N_meas, np.std(resets)/np.sqrt(len(resets)), len(resets))#, env.testing)
    # tracking[N_meas - 1] = np.mean(resets)
    print(time.time() - start, np.mean(resets), np.std(resets)/np.sqrt(len(resets)), len(resets))#, env.testing)

    # print(env.bench)
    # with open(f"Test_{N_meas}.npy", 'wb') as f2:
    #     np.save(f2, env.tracking, allow_pickle = False)

    # print(env.benchmark)
    # print(np.mean(baseline), np.std(baseline)/np.sqrt(len(baseline)))
    # imageio.mimsave('Ccart_sig05.gif', [np.array(img) for i, img in enumerate(images) if i%2 == 0], fps=30)     


## Mach training test, 44 qcart_Saving3, 45 qcart_Saving45

# Ask Mark about Strong Measurement and Xeno effect
# discuss the binning in quartic potential