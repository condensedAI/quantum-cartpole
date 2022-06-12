import numpy as np
import gym
import scipy.linalg as linalg
import gym_qcart.utility.utilv0 as utilv0
from scipy.signal import savgol_filter
from os import path

class QuantumCartGameEnvV0(gym.Env):
    """
    Description:
        A pole is attached by an un-actuated joint to a cart, which moves along
        a frictionless track. The pendulum starts upright, and the goal is to
        prevent it from falling over by increasing and reducing the cart's
        velocity.

    Observation:
        Type: Box(2)
        Num     Observation                                 Min                     Max
        0       Mean Weak Measurement Position             -x_border                x_border
        1       Mean Weak Measurement Momentum             -x_border                x_border

    Actions:
        Type: Discrete(Na)
        Num   Action
        0 - (N/2 - 1)     Push wavefunction to the left using a kick operator
        N/2               Nothing               
        (N/2 + 1) - N     Push wavefunction to the right using a kick operator
        Note: The strength of the kick is fixed at the beginning
    Reward:
        Reward is 1 for every step taken, including the termination step
    Episode Termination:
        When more then 50% of the probability distribution is outside of the threshold area
        Episode length is greater than 2000.
    """

    metadata = {'render.modes': ['human']}
    reward_range = (-float("inf"), float("inf"))

    def __init__(self, sigma, dt, L, mu):

        ### System definition
        self.Nx = 1001
        self.x_border = 15.0
        self.xarr = np.linspace(-self.x_border,self.x_border,self.Nx)
        self.delta_x = self.xarr[1] - self.xarr[0]
        self.x_threshold = 5.0 #threshold area from -x to x
        self.mu = mu
        self.sig = 1.0
        self.p0 = 0.0
        self.done = False

        ### Ancilla space definition
        self.Nq = 1001
        self.N_meas = 5
        self.q_border = 5*sigma
        self.qarr = np.linspace(-self.q_border,self.q_border,self.Nq)
        self.delta_q = self.qarr[1] - self.qarr[0]
        self.phi_0 = utilv0.gaussian_wavepaket(self.qarr, 0, 0, sigma)
        self.phi_1 = -1j*utilv0.gaussian_wavepaket_1deriv(self.qarr, 0, 0, sigma)
        self.phi_2 = -1*utilv0.gaussian_wavepaket_2deriv(self.qarr, 0, 0, sigma)

        self.L = L # interactions strength of the measurement

        ### Actions definition
        self.dt = dt
        self.action_number = 5
        self.kick_strength = 0.5
        self.kick_array = np.linspace(-self.kick_strength, self.kick_strength, 3)
        self.probability_threshold = 0.5
        self.termination = 400
        self.action_space = gym.spaces.Discrete(self.action_number)
        self.observation_space = gym.spaces.Box(np.array([-self.q_border, -self.q_border], dtype=np.float32), np.array([self.q_border, self.q_border], dtype=np.float32), dtype=np.float32)
        # self.observation_space = gym.spaces.Box(np.array([-0.11], dtype=np.float32), np.array([0.11], dtype=np.float32), dtype=np.float32)
        self.state = None

        self.actual_state = None
        #Calculate the hamiltonian
        A = np.zeros((self.Nx,self.Nx)) 
        for i in range(self.Nx-1):
            A[i, i+1] = 1

        T2 = (A + A.T - 2 * np.identity(self.Nx)) * (-1 / (2 * (self.delta_x)**2)) #kinetic term with a second derivative:
        V = np.diag(self.potential(self.xarr))
        self.H = T2 + V

        # Time evolution operator
        self.U_time = linalg.expm(-1j * self.H * self.dt) #time evolution
        self.kicks = [self.kick_operator(self.kick_array[i]) for i in range(3)]

        self.number_of_steps_taken = 0
        self.viewer = None
        self.reset()


    def potential(self, x):
        return - 1/2 * x **2

    def seed(self, seed=None):
        np.random.seed(seed)
        return [seed]

    def kick_operator(self, strength):
        return np.exp(1j*self.xarr * strength)

    def step(self, action):
        q_meas_n, p_meas_n = 0, 0
        self.psi = np.dot(self.U_time, self.psi)
        if action == 1:
            self.psi = self.kicks[0]*self.psi
        elif action == 2: 
            self.psi = self.kicks[2]*self.psi
        elif action == 3:
            q_meas_n, self.psi, q_act = utilv0.wm_pos(self.xarr, self.Nq, self.qarr, self.psi, self.phi_0, self.phi_1, self.phi_2, self.L)
            self.psi = savgol_filter(np.real(self.psi), 11, 4) + 1j*savgol_filter(np.imag(self.psi), 11, 4)
            psi1 = utilv0.deriv_first(self.xarr, self.psi, self.delta_x)
            psi2 = utilv0.deriv_first(self.xarr, psi1, self.delta_x)          
            p_meas_n, self.psi, p_act = utilv0.wm_mom(self.xarr, self.Nq, self.qarr, self.psi, psi1, psi2,self.phi_0, self.phi_1, self.phi_2, self.L)
            self.psi = savgol_filter(np.real(self.psi), 11, 4) + 1j*savgol_filter(np.imag(self.psi), 11, 4)
        elif action == 4:
            self.done = True

        self.state = (q_meas_n, p_meas_n)
        
        # Calculate the total probability to find the particle between - x and x
        self.probability_distribution = utilv0.compute_probability(self.psi)
        self.total_probability = utilv0.compute_total_probability(self.probability_distribution[int(self.Nx/2 - 0.5*self.Nx*self.x_threshold/self.x_border) : int(self.Nx/2 + 0.5*self.Nx*self.x_threshold/self.x_border)], self.delta_x)

        if self.total_probability < self.probability_threshold or self.done == True:
            return np.array(self.state, dtype=np.float32), 1, True, {}   
        else:
            return np.array(self.state, dtype=np.float32), 1, False, {}        


    def reset(self):
        # Reset the wavefunction to the initial gaussian
        self.done = False
        x_0 = np.random.uniform(-self.mu, self.mu)
        self.psi = utilv0.gaussian_wavepaket(self.xarr, x_0, self.p0, self.sig)
        self.probability_distribution = utilv0.compute_probability(self.psi)
        self.total_probability = utilv0.compute_total_probability(self.probability_distribution[int(self.Nx/2 - 0.5*self.Nx*self.x_threshold/self.x_border) : int(self.Nx/2 + 0.5*self.Nx*self.x_threshold/self.x_border)], self.delta_x)
        self.number_of_steps_taken = 0
        self.state = (0.,0.)
        return np.array(self.state, dtype=np.float32)

    def render(self):
        pass

    def close(self):
        if self.viewer:
            self.viewer.close()
            self.viewer = None
