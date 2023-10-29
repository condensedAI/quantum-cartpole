from gym.envs.registration import register
import numpy as np

register(
    id='qcart-v0',
    entry_point='gym_qcart.envs:QuantumCartPoleEnvV0',
    kwargs={},
)