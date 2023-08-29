from gym.envs.registration import register
import numpy as np

register(
    id='qcart-v0',
    entry_point='gym_qcart.envs:QuantumCartPoleEnvV0',
    kwargs={'Nx': 1001, 'Nq': 1001, 'sigma' : 0.7, 'dt' : 0.01/np.pi, 'L' : 0.05, 'm' : 1/np.pi, 'mu' : 1.0, "N_meas": 5, "q_border": 5, "factor": 0., "kickfactor": 5, "termination": 5000000, "types" : "rl", "rewards" : 0, 'input_type' : 0, 'output_type' : 0},
)

register(
    id='qcart-v1',
    entry_point='gym_qcart.envs:QuantumCartPoleEnvV1',
    kwargs={},
)


############################################# Game implementation

register(
    id='qgame-v0',
    entry_point='gym_qcart.envs:QuantumCartGameEnv',
    kwargs={'sigma' : 0.1, 'dt' : 0.01, 'L' : 0.02, 'mu' : 2.0},
)