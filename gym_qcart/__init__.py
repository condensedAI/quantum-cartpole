from gym.envs.registration import register
import numpy as np

register(
    id='qcart-v0',
    entry_point='gym_qcart.envs:QuantumCartPoleEnvV0',
    kwargs={},
)


############################################# Game implementation

register(
    id='qgame-v0',
    entry_point='gym_qcart.envs:QuantumCartGameEnv',
    kwargs={'sigma' : 0.1, 'dt' : 0.01, 'L' : 0.02, 'mu' : 2.0},
)