from gym.envs.registration import register

register(
    id='qcart-v0',
    entry_point='gym_qcart.envs:QuantumCartPoleEnvV0',
    kwargs={'Nx': 1001, 'Nq': 1001, 'sigma' : 0.7, 'dt' : 0.01, 'L' : 0.05, 'mu' : 2.0, "N_meas": 5, "q_border": 5, "factor": 0., "kickfactor": 1, "termination": 2500},
)

############################################# Game implementation

register(
    id='qgame-v0',
    entry_point='gym_qcart.envs:QuantumCartGameEnv',
    kwargs={'sigma' : 0.1, 'dt' : 0.01, 'L' : 0.02, 'mu' : 2.0},
)