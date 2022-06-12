import numpy as np
import gym
import scipy.linalg as linalg
import gym_qcart.utility.utilv0 as utilt
from scipy.signal import savgol_filter
from os import path

class QuantumCartPoleEnvV0(gym.Env):
    """
    Add additional support for statistics.
    """

    metadata = {'render.modes': ['human']}
    reward_range = (-float("inf"), float("inf"))
    def __init__(self, Nx, Nq, sigma, dt, L, mu, N_meas, q_border, factor, kickfactor, termination):
        # Unit definition according to the other qcart paper
        self.w_c, self.m_c, self.w, self.m, self.hbar,  self.k, self.Fmax, self.x_threshold = 1, 1, np.pi, 1/np.pi, 1, np.pi, N_meas*kickfactor*np.pi*8, 8 #For quadratic potential
        self.period = 2/self.w_c
        self.factor = factor

        # System definition
        self.N_meas = N_meas
        self.Nx = Nx
        self.x_border = 20.0
        self.xarr = np.linspace(-self.x_border,self.x_border, self.Nx)
        self.dx = (self.xarr[1] - self.xarr[0])
        self.mu = mu
        self.psi_0 = np.zeros(self.Nx, dtype = np.cdouble)
        self.psi_1 = np.zeros(self.Nx, dtype = np.cdouble)
        self.psi_2 = np.zeros(self.Nx, dtype = np.cdouble)

        ### Ancilla space definition
        self.Nq = Nq
        self.q_border = q_border
        qarr = np.linspace(-self.q_border, self.q_border, self.Nq)
        self.qarr = qarr
        self.dq = qarr[1] - qarr[0]
        self.phi_0 = utilt.gaussian_wavepaket(self.qarr, 0, 0, sigma)
        self.phi_1 = -1j*utilt.gaussian_wavepaket_1deriv(self.qarr, 0, 0, sigma)
        self.phi_2 = -1*utilt.gaussian_wavepaket_2deriv(self.qarr, 0, 0, sigma)

        self.L = L # interactions strength of the measurement

        ### Actions definition
        self.dt = dt
        self.action_number = 7
        self.action = self.action_number//2
        self.probability_threshold = 0.5
        self.termination = int(termination/N_meas)
        self.action_space = gym.spaces.Discrete(self.action_number)

        # self.observation_space = gym.spaces.Box(np.array([-self.q_border, -self.q_border], dtype=np.float32), np.array([self.q_border, self.q_border], dtype=np.float32), dtype=np.float32)
        # self.observation_space = gym.spaces.Box(np.array([-2.75, -2.75], dtype=np.float32), np.array([2.75, 2.75], dtype=np.float32), dtype=np.float32)
        self.observation_space = gym.spaces.Box(np.array([-1., -1.], dtype=np.float32), np.array([1., 1.], dtype=np.float32), dtype=np.float32)#normalized
        self.state = np.array((0., 0.), dtype = np.float32)

        self.actual_state = None
        A = np.zeros((self.Nx,self.Nx)) 
        for i in range(self.Nx-1):
            A[i, i+1] = 1

        T2 = - self.hbar**2 / (2*self.m) *   (A + A.T - 2 * np.identity(self.Nx)) / (self.dx**2) #kinetic term with a second derivative:
        V = np.diag(self.potential(self.xarr))
        H = T2 + V
        self.U_time = linalg.expm(-1j * H * self.dt) #time evolution

        # Kicks
        self.kick_array = np.linspace(-self.Fmax, self.Fmax, self.action_number)        
        self.kicks = [self.kick_operator(i) for i in range(self.action_number)]
        self.number_of_steps_taken = 0
        self.viewer = None
        self.reset()


    def seed(self, seed=None):
        np.random.seed(seed)
        return [seed]

    #quadratic potential
    def potential(self, action):
        return - self.k/2 * self.xarr **2

    def kick_operator(self, action):
        return np.exp(-1j*(- self.kick_array[action]*self.xarr)*self.dt)

    def step(self, action):
        q_meas, p_meas = 0,0
        for it in range(self.N_meas):
            
            #kick at the beginning
            self.action = action
            if it == 0:
                self.psi_0 = self.kicks[action]*self.psi_0
            self.psi_0 = np.dot(self.U_time, self.psi_0)    
            
            q_meas_n, self.psi_0 = utilt.wm_pos(self)
            q_meas += q_meas_n
            self.psi_0 = savgol_filter(np.real(self.psi_0), 11, 4) + 1j*savgol_filter(np.imag(self.psi_0), 11, 4)
            self.psi_1 = utilt.deriv_first(self.xarr, self.psi_0)
            # self.psi_1 = savgol_filter(np.real(self.psi_1), 11, 4) + 1j*savgol_filter(np.imag(self.psi_1), 11, 4)
            self.psi_2 = utilt.deriv_first(self.xarr, self.psi_1)
            # self.psi_2 = savgol_filter(np.real(self.psi_2), 11, 4) + 1j*savgol_filter(np.imag(self.psi_2), 11, 4)      
            p_meas_n, self.psi_0 = utilt.wm_mom(self)
            self.psi_0 = savgol_filter(np.real(self.psi_0), 11, 4) + 1j*savgol_filter(np.imag(self.psi_0), 11, 4)
            p_meas += p_meas_n

        # mean measurement output normalized
        self.state[0] = (q_meas/(self.N_meas * self.q_border)) 
        self.state[1] = (p_meas/(self.N_meas * self.q_border))
        self.number_of_steps_taken += 1

        # Calculate the total probability to find the particle between - x and x
        self.probability_distribution = utilt.compute_probability(self.psi_0)
        self.total_probability =utilt.compute_total_probability(self.probability_distribution[int(self.Nx/2 - 0.5*self.Nx*self.x_threshold/self.x_border) : int(self.Nx/2 + 0.5*self.Nx*self.x_threshold/self.x_border)], self.dx)
        

        if self.total_probability < self.probability_threshold:
            return self.state, 0, True, {}
        elif self.number_of_steps_taken >= self.termination:
            return self.state, 1, True, {}            
        else:
            return self.state, 1, False, {}        


    def reset(self):
        # Reset the wavefunction to the initial gaussian
        x_0 = np.random.uniform(-self.mu, self.mu)
        self.psi_0 = utilt.gaussian_wavepaket(self.xarr, x_0, 0, 1)
        self.probability_distribution = utilt.compute_probability(self.psi_0)
        self.total_probability = utilt.compute_total_probability(self.probability_distribution[int(self.Nx/2 - 0.5*self.Nx*self.x_threshold/self.x_border) : int(self.Nx/2 + 0.5*self.Nx*self.x_threshold/self.x_border)], self.dx)
        self.number_of_steps_taken = 0
        self.state = np.array((0.,0.), dtype=np.float32)
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

