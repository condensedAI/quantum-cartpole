import numpy as np
import gym
import scipy.linalg as linalg
import gym_qcart.utility.util as util
from os import path
import time
import numba as nb

def rl_action_dis(env, action):
    return env.kicks[action]*env.psi_0

def rl_action_con(env, action):
    return env.kick_operator(env.force * action) *env.psi_0    

def lqr_action(env, x, p):
    lgc = -1*(x + p*env.dt/env.m + p + env.k*x*env.dt)/env.dt
    force = np.sign(lgc)*min(np.abs(lgc),env.Fmax)
    env.action = int(np.round(force/env.Fmax, decimals = 0)) + 3
    return env.kick_operator(force)*env.psi_0

@nb.jit
def d_dxdx(phi, deltax):
    dphi_dxdx = -2*phi
    dphi_dxdx[:-1] += phi[1:]
    dphi_dxdx[1:] += phi[:-1]
    return dphi_dxdx/(deltax**2)

@nb.jit
def d_dt(phi,m,V, deltax):
    return 1j*1/2/m * d_dxdx(phi, deltax) - 1j*V*phi/1

@nb.jit## use runge kutta for time evolution
def rk4(phi, dt, m, V, deltax):
    k1 = d_dt(phi, m, V, deltax)
    k2 = d_dt(phi+dt/2*k1, m, V, deltax)
    k3 = d_dt(phi+dt/2*k2, m, V, deltax)
    k4 = d_dt(phi+dt*k3, m, V, deltax)        
    return phi + dt/6*(k1+2*k2+2*k3+k4)
    

class QuantumCartPoleEnvV0(gym.Env):

    metadata = {'render.modes': ['human']}
    reward_range = (-float("inf"), float("inf"))
    def __init__(self, Nx, Nq, sigma, dt, L, m, mu, N_meas, q_border, factor, kickfactor, termination, types, rewards, input_type, output_type):
        # Unit deefinition according to the other qcart paper
        self.w_c, self.m_c, self.w, self.m, self.hbar,  self.k, self.Fmax, self.x_threshold = 1, 1, np.pi, m, 1, np.pi, N_meas*kickfactor*np.pi*8, 8
        self.period = 2/self.w_c
        self.factor = factor

        self.types = types
        self.N_meas = N_meas
        self.Nx = Nx
        self.x_border = 20.0
        self.xarr = np.linspace(-self.x_border,self.x_border, self.Nx)
        self.dx = (self.xarr[1] - self.xarr[0])
        self.mu = mu
        self.psi_0 = np.zeros(self.Nx, dtype = np.csingle)
        self.psi_1 = np.zeros(self.Nx, dtype = np.csingle)
        self.psi_2 = np.zeros(self.Nx, dtype = np.csingle)
        self.V = self.potential(self.xarr)
        ### Ancilla space definition
        self.Nq = Nq
        self.q_border = q_border
        qarr = np.linspace(-self.q_border, self.q_border, self.Nq)
        self.qarr = qarr
        self.dq = qarr[1] - qarr[0]
        self.phi_0 = util.gaussian_wavepaket(self.qarr, 0, 0, sigma)
        self.phi_1 = -1j*util.gaussian_wavepaket_1deriv(self.qarr, 0, 0, sigma)
        self.phi_2 = -1*util.gaussian_wavepaket_2deriv(self.qarr, 0, 0, sigma)

        self.position = 0
        self.momentum = 0

        self.L = L # interactions strength of the measurement
        self.psi_0 = util.gaussian_wavepaket(self.xarr, 0, 0, 1)
        self.psi_1 = util.deriv_first(self.xarr, self.psi_0)
        self.psi_2 = util.deriv_first(self.xarr, self.psi_1)        

        ### measurement probability stays constant (only shift in x), so we can compute it once in the beginning
        self.example_pos = util.example_meas_pos(self.xarr, self.qarr, self.psi_0, self.phi_0, self.phi_1, self.phi_2, self.L)
        self.example_mom = util.example_meas_mom(self.qarr, self.psi_0, self.psi_1, self.psi_2, self.phi_0, self.phi_1, self.phi_2, self.L)

        ### Actions definition
        self.dt = dt
        self.probability_threshold = 0.5
        self.termination = int(termination/N_meas)

        self.reward_type = rewards
        self.output_type = output_type
        self.input_type = input_type
        self.action_number = 7
        if self.input_type == 0:
            self.low = np.array([-1, -1], dtype=np.float32)
            self.high = np.array([1, 1], dtype=np.float32)
            self.state = np.array((0., 0.), dtype = np.float32)
        elif self.input_type == 1:
            self.state = np.array((0., 0., 0.), dtype = np.float32)
            if self.output_type == 0:
                self.action_number = 7
                self.low = np.array([-1, -1, 0], dtype=np.float32)
                self.high = np.array([1, 1, self.action_number-1], dtype=np.float32)
            elif self.output_type == 1:
                self.action_number = 7
                self.low = np.array([-1, -1, -1], dtype=np.float32)
                self.high = np.array([1, 1, 1], dtype=np.float32)
        self.observation_space = gym.spaces.Box(self.low, self.high, dtype=np.float32)

        self.force = N_meas*kickfactor*np.pi*8
        if self.output_type == 0:
            self.forces = np.linspace(-self.force, self.force, self.action_number)
            self.action_space = gym.spaces.Discrete(self.action_number)
            self.kick_array = np.linspace(-self.Fmax, self.Fmax, self.action_number)        
            self.kicks = [self.kick_operator(self.kick_array[i]) for i in range(self.action_number)]            
        elif self.output_type ==1:
            self.action_space = gym.spaces.Box(np.array([-1.]), np.array([1.]), dtype=np.float64)
        
        self.action = 3
    
        self.number_of_steps_taken = 0
        self.viewer = None

        self.reset()

    def seed(self, seed=None):
        np.random.seed(seed)
        return [seed]

    def potential(self, action):
        return - self.k/2 * self.xarr **2

    def kick_operator(self, force):
        return np.exp(-1j*(- force*self.xarr)*self.dt)

    def step(self, action):
        q_meas, p_meas = 0,0
        for it in range(self.N_meas):
            self.action = action
            
            for _ in range(5):
                self.psi_0 = rk4(self.psi_0, self.dt/5, self.m, self.V, self.dx) #time evolution

            if it == 0:
                if self.types == "rl":
                    if self.output_type == 0:
                        self.psi_0 = rl_action_dis(self, action)
                    elif self.output_type == 1:
                        self.psi_0 = rl_action_con(self, action)

                elif self.types == "lqr":
                    self.psi_0 = lqr_action(self, self.state[0]/self.L, self.state[1]/self.L)


            q_meas_n, self.psi_0 = util.wm_pos(self)
            self.psi_1 = util.deriv_first(self.xarr, self.psi_0)
            self.psi_2 = util.deriv_first(self.xarr, self.psi_1)
            p_meas_n, self.psi_0 = util.wm_mom(self)
            
            q_meas += q_meas_n
            p_meas += p_meas_n

        self.state[0] = self.state[0]*self.factor +  (q_meas/(self.N_meas)) 
        self.state[1] = self.state[1]*self.factor +  (p_meas/(self.N_meas))
        if self.input_type == 1:
            self.state[2] = action

        self.number_of_steps_taken += 1
        self.psi_1 = util.deriv_first(self.xarr, self.psi_0)
        pos = np.real(np.sum(self.psi_0*self.xarr*np.conj(self.psi_0)*self.dx))     
        self.probability_distribution = util.compute_probability(self.psi_0)## onlz need for plotting        

        if np.abs(pos) > self.x_threshold:
            self.number_of_steps_taken = 0
            return self.state, 0, True, {}
        elif self.number_of_steps_taken >= self.termination:
            self.number_of_steps_taken = 0
            return self.state, 1, True, {}            
        else:
            return self.state, 1, False, {}        

                   


    def reset(self):
        # Reset the wavefunction to the initial gaussian
        p_0 = np.random.uniform(-self.mu, self.mu)
        self.psi_0 = util.gaussian_wavepaket(self.xarr, 0, p_0, 1)
        self.probability_distribution = util.compute_probability(self.psi_0)
        self.total_probability = util.compute_total_probability(self.probability_distribution[int(self.Nx/2 - 0.5*self.Nx*self.x_threshold/self.x_border) : int(self.Nx/2 + 0.5*self.Nx*self.x_threshold/self.x_border)], self.dx)
        self.number_of_steps_taken = 0

        if self.input_type == 0:
            self.state = np.array((0, 0), dtype=np.float32)
        elif self.input_type == 1:
            self.state = np.array((0, 0, 0), dtype=np.float32)
   
        return self.state

    def render(self, mode='both'):
        from gym.envs.classic_control import rendering
        screen_width = 600
        screen_height = 400

        scaleX = screen_width/(2*self.x_border)
        scaleY = screen_height*0.50 #Use only 80% of screen height

        if self.viewer is None:
            self.viewer = rendering.Viewer(screen_width, screen_height)
            self.th_left = rendering.Line(((-self.x_threshold + self.x_border)*scaleX, 0), ((-self.x_threshold + self.x_border)*scaleX, screen_height))
            self.th_right = rendering.Line(((self.x_threshold + self.x_border)*scaleX, 0), ((self.x_threshold + self.x_border)*scaleX, screen_height))
            self.viewer.add_geom(self.th_left)# Add threshold markers
            self.viewer.add_geom(self.th_right)

            fname = path.join(path.dirname(__file__), "assets/arrow.png")
            self.img = rendering.Image(fname, 100.0, 100.0)
            self.imgtrans = rendering.Transform()
            self.img.add_attr(self.imgtrans)
            self.imgtrans.translation = (100, 100)

        self.curve_real = rendering.make_polyline(zip( (self.xarr + self.x_border)*scaleX, np.real(self.psi_0)*scaleY + screen_height*0.5))
        self.curve_im = rendering.make_polyline(zip( (self.xarr + self.x_border)*scaleX, np.imag(self.psi_0)*scaleY + screen_height*0.5))
        self.curve_prob = rendering.make_polyline(zip( (self.xarr + self.x_border)*scaleX, self.probability_distribution*scaleY + screen_height*0.1))
        self.curve_potential = rendering.make_polyline(zip( (self.xarr + self.x_border)*scaleX, (self.potential(self.action))*scaleY/np.max(self.potential(self.action)) + screen_height*0.1))

        self.curve_real.set_linewidth(2)
        self.curve_im.set_linewidth(2)
        self.curve_prob.set_linewidth(2)
        self.curve_potential.set_linewidth(2)

        self.curve_real.set_color(255/255, 51/255, 51/255)
        self.curve_im.set_color(51/255, 133/255, 255/255)

        self.viewer.add_geom(self.curve_im)     
        self.viewer.add_geom(self.curve_real)
        self.viewer.add_geom(self.curve_prob)
        self.viewer.add_geom(self.curve_potential)

        
        self.imgtrans.scale = (2*(self.action/(self.action_number-1) - 0.5) , 2*np.abs(self.action/(self.action_number-1) - 0.5))
        self.viewer.add_onetime(self.img)
        return self.viewer.render(return_rgb_array=mode == "rgb_array")

    def close(self):
        if self.viewer:
            self.viewer.close()
            self.viewer = None
