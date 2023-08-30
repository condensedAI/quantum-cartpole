import numpy as np
import gym
import scipy.linalg as linalg
import gym_qcart.utility.util as utilt
from os import path
import time
import numba as nb
from scipy.optimize import curve_fit
import control as ct
from gym import spaces

#Runge Kuttta for time evoultion of wavefunction
@nb.jit
def d_dxdx(phi, deltax):
    dphi_dxdx = -2*phi
    dphi_dxdx[:-1] += phi[1:]
    dphi_dxdx[1:] += phi[:-1]
    return dphi_dxdx/(deltax**2)

@nb.jit
def d_dt(phi,m,V, deltax):
    return 1j*1/2/m * d_dxdx(phi, deltax) - 1j*V*phi/1

@nb.jit
def rk4(phi, dt, m, V, deltax):
    k1 = d_dt(phi, m, V, deltax)
    k2 = d_dt(phi+dt/2*k1, m, V, deltax)
    k3 = d_dt(phi+dt/2*k2, m, V, deltax)
    k4 = d_dt(phi+dt*k3, m, V, deltax)        
    return phi + dt/6*(k1+2*k2+2*k3+k4)

def kick_operator(env, force):
    return np.exp(-1j*(-force*env.xarr)*env.dt)

#The system noise approximeted to be a linear combination of the emasurement
@nb.jit
def backaction(x_dif, p_dif, weights):
    return weights[0]* x_dif + weights[2] * p_dif, weights[1]* x_dif + weights[2] * p_dif, 

#Classical time steps depending on the potential, used in classicla time evoultion and for kalman filter
def quadratic(env, sys):
    x = sys[0] + sys[1]/env.m * env.dt
    p = sys[1] + env.k*sys[0] * env.dt
    return np.array([x, p])

def quartic(env, sys):
    x = sys[0] + sys[1]/env.m * env.dt
    p = sys[1] + 4*env.k*sys[0]**3 * env.dt
    return np.array([x, p])

def cosine(env, sys):
    x = sys[0] + sys[1]/env.m * env.dt
    p = sys[1] + env.k*(np.pi/(1.5*env.max_position)) * np.sin(np.pi * sys[0]/(1.5*env.max_position))  * env.dt
    return np.array([x, p])

#Different potential for the systems
def quadraticV(env):
    return -0.5 * env.k * env.xarr ** 2

def quarticV(env):
    return -env.k * env.xarr ** 4

def cosineV(env):
    return env.k * (np.cos(np.pi*env.xarr/(1.5*env.max_position) ) - 1 )

#Jacobian used for extended kalman filter and lqr. quadratic jacobian is just a pass, since it is not needed
def Jacobian_quadratic(env):
    pass

def Jacobian_quartic(env):
    env.A = np.array([[1 , (env.dt) /env.m], [12 * env.k*env.dt * env.estimate[0]**2, 1]]) 
    env.A2 = np.array([[1 , (env.dt * env.N_meas) /env.m], [12 * env.k*(env.dt * env.N_meas) * env.estimate[0]**2, 1]]) 
    env.Q2 = np.diag([env.k * env.estimate[0] **2, 1/(env.m*2)]) 

def Jacobian_cosine(env):
    env.A = np.array([[1 , (env.dt) /env.m], [env.k*env.dt*((np.pi/(1.5*env.max_position) )**2) * np.cos(np.pi*env.estimate[0]/(1.5*env.max_position) ), 1]]) 
    env.A2 = np.array([[1 , (env.dt * env.N_meas) /env.m], [env.k*(env.dt * env.N_meas)*((np.pi/(1.5*env.max_position) )**2) * np.cos(np.pi*env.estimate[0]/(1.5*env.max_position) ), 1]])         
    env.Q2 = np.diag([1, 1])


# Control functions
def control_random(env):
    env.estimate += env.action
    env.difference = np.abs(((env.estimate - env.measurement/env.L) / ( env.max_position * 2 ) ) )
    lgc = (np.random.rand()*2 - 1) * env.Fmax
    return np.sign(lgc)*min(np.abs(lgc), env.Fmax, key=abs)

def control_lqr(env):
    lgc = -np.sum(np.dot(env.K, env.estimate ))
    return np.sign(lgc)*min(np.abs(lgc), env.Fmax, key=abs)

def control_elqr(env):
    env.K, _, _ = ct.dlqr(env.A2, env.B2, env.Q2, env.R2)
    lgc =  -1*np.sum(np.dot(env.K, env.estimate))
    return np.sign(lgc)*min(np.abs(lgc), env.Fmax, key=abs)

def control_rlc(env):
    lgc =  -env.action * env.Fmax
    return np.sign(lgc)*min(np.abs(lgc), env.Fmax, key=abs)

# Time Evolutions function & Measurement
def time_evolution_classical(env):
    env.actual_pos = env.time_step(env, env.actual_pos) + env.B.dot(env.control)
    meas = np.random.normal(0, env.sigma , 2)
    env.back = backaction(meas[0]/env.L, meas[1]/env.L, env.sigma_sys)
    env.measurement = env.C.dot(env.actual_pos) + meas
    env.actual_pos += env.back

def time_evolution_quantum(env):
    
    #time evoultionn on the potential
    for _ in range(7):
        env.psi_0 = rk4(env.psi_0, env.dt/7, env.m, env.V, env.dx)
    env.psi_0 = kick_operator(env, env.control[1])*env.psi_0
    
    # Weak Measurements    
    q_meas_n, env.psi_0 = utilt.wm_pos(env)
    env.psi_1 = utilt.deriv_first(env.xarr, env.psi_0)
    env.psi_2 = utilt.deriv_first(env.xarr, env.psi_1)
    p_meas_n, env.psi_0 = utilt.wm_mom(env)         
    env.measurement[0] = q_meas_n
    env.measurement[1] = p_meas_n

    #Calculating actual <x> value, for termination condition
    env.actual_pos[0] = np.real(np.sum(env.psi_0*env.xarr*np.conj(env.psi_0)*env.dx))
    #env.actual_pos[1] = np.real(np.sum(env.psi_1*np.conj(env.psi_0)*env.dx))

#State Estimation functions
def kalman(env):    
    env.measurement_mean += env.measurement/env.N_meas
    if env.number_of_steps_taken%env.N_meas == env.N_meas-1:
        x_pred = env.A.dot(env.estimate)  + env.B.dot(env.control) ### xpred
        env.estimate = x_pred + env.kalman_gain.dot(env.measurement_mean -  env.C.dot(x_pred))
        env.measurement_mean *= 0

def extended_kalman(env):
    env.measurement_mean += env.measurement/env.N_meas
    if env.number_of_steps_taken%env.N_meas == env.N_meas-1:    
        x_pred = env.time_step(env, env.estimate) + env.B.dot(env.control)          ### state prediction
        y_pred = env.C.dot(x_pred)                                                  ### measurement prediction
        y_res = env.measurement_mean - y_pred                                            ### measurement residual
        env.P = env.A.dot(env.P).dot(np.transpose(env.A)) + env.Q                   ### State prediciton covariance
        S_pred = env.R + env.C.dot(env.P).dot(np.transpose(env.C))                  ### Innovation Covariance
        kalman_gain = env.P.dot(np.transpose(env.C)).dot( np.linalg.inv(S_pred) )   ### Filter gain
        env.estimate = x_pred + kalman_gain.dot(y_res)
        env.P = env.P - kalman_gain.dot(S_pred).dot(np.transpose(kalman_gain))
        env.measurement_mean *= 0

def rle(env):
    env.measurement_mean += env.measurement/env.N_meas
    if env.number_of_steps_taken%env.N_meas == env.N_meas-1:    
        env.estimate += env.filter_model.predict(env.state_est, deterministic = True)[0]
        env.measurement_mean *= 0

def estimator_none(env):
    pass   

# Reward Function
def reward_running(env):
    if np.abs(env.actual_pos[0]) > env.max_position:
        reward = env.number_of_steps_taken/env.termination
    else:
        reward = 0
    return reward

def reward_filter(env):
    return -1*np.mean(env.difference)

# State returns, state_estimator is only used when training an estimator model
def state_running(env):
    env.state[:] = env.estimate_mean    
    env.state_est[:2] = env.measurement / env.L
    env.state_est[2:4] = env.estimate
    env.state_est[4] = env.B.dot(env.control)[1]    

def state_estimator(env):
    env.state[:2] = env.measurement / env.L
    env.state[2:4] = env.estimate
    env.state[4] = env.B.dot(env.control)[1]

##############################################
def sigma_func1(x, a, b):
    return a * x + b   

def sigma_func2(x, a, b, c):
    return a * np.exp(x*b) + c   

class QuantumCartPoleEnvV1(gym.Env):
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

    Actions RLC:
        Type: Continuous
        Num     Action                                  Min                     Max 
        0       push the wavefunction                   -1                      +1
        Note: The max stregth of the kick is fixed in the beginning

    Actions RLE:
        Type: Continuous
        Num     Action                                  Min                     Max 
        0       adjust the position estimation          -1                      +1 
        1       adjust the momentum estimation          -1                      +1        

    Reward:
        RLC: +number_of_steps/termination
        RLE: -np.mean(abs(esitmation - measurement))

    Episode Termination:
        When more then 50% of the probability distribution is outside of the threshold area
        Episode length is greater than 1e6.
    """    

    metadata = {'render.modes': ['human']}
    reward_range = (-float("inf"), float("inf"))
    
    def __init__(self, N_meas = 1, k = np.pi, system = 'quantum', potential = 'quadratic', controller = 'rlc', estimator = 'rle', state_return = 'state_running',filter_model = None):
        
        ######################################Variables you always need
        self.N_meas = N_meas
        self.k = k
        self.max_position = 8
        self.dt = 0.01/np.pi
        self.L = 0.05
        self.mu = np.array([0. , 1.0], dtype=np.float64)
        self.m = 1/np.pi
        self.termination = 1e6
        self.Fmax = 8*np.pi
        self.sigma = 0.7
        self.system = system

        self.rewards = 0
        self.run_count = 1
        self.actual_pos = np.array([0. , 0.], dtype=np.float64)
        self.number_of_steps_taken = 0
        self.control = np.zeros(2, dtype=np.float64)
        self.estimate = np.zeros(2, dtype=np.float64)
        self.estimate_mean = np.zeros(2, dtype=np.float64)
        self.measurement = np.zeros(2, dtype=np.float64)
        self.measurement_mean = np.zeros(2, dtype=np.float64)
        self.viewer = None

        self.action_space = spaces.Box(np.array([-1.]), np.array([1.]), dtype=np.float64)      
        border = np.ones(2, dtype = np.float64) * self.max_position*2
        self.observation_space = spaces.Box(-1*border, border, dtype=np.float64)  
        self.state = np.zeros(2, dtype = np.float64)
        self.state_est = np.zeros(5, dtype = np.float64)
        self.reward_func = reward_running
        self.time_step = eval(potential)
        self.return_output = eval(state_return)

        self.A = np.array([[1 , (self.dt * N_meas) /self.m], [self.k*(self.dt * N_meas) , 1]])
        self.B = np.array([[0, 0], [0, self.dt * N_meas]])
        self.C = np.diag([self.L, self.L])

        if potential == 'quadratic':
            self.P = np.array([[1 , 0], [0, 1]])
            self.P_reset = np.array([[1 , 0], [0, 1]])                
            self.sigma_sys = np.array([sigma_func1(self.sigma, -0.00433991, 0.01332713), 
                                        sigma_func2(self.sigma, -1.34074255e-05, 4.83173413e+00, 1.02313236e-02),
                                        sigma_func2(self.sigma, -6.87118142e-05, 3.30739658e+00, 1.03732448e-02), 
                                        sigma_func1(self.sigma, -0.00504242, 0.01357103)])                                    
        elif potential == 'quartic':
            self.P = np.array([[610 , 280], [180, 130]])
            self.P_reset = np.array([[610 , 280], [180, 130]])                
            self.sigma_sys = np.array([0.008086549851490932, 0.005606322236226655, 0.005601178239519393, 0.006012663950129217])                
        elif potential == 'cosine':
            self.P = np.array([[769 , 621], [620, 769]])
            self.P_reset = np.array([[769 , 621], [620, 769]])                
            self.sigma_sys = np.array([0.01214231068233882, 0.010957951536296766, 0.00955755313643539, 0.010540234359464778])
            # self.sigma_sys = np.array([0.0095, 0.0095, 0.0095, 0.0095])#########Das hier machen


        if system == 'classical':
            self.time_evolution = time_evolution_classical
            self.back = np.zeros(2)

        elif system == 'quantum':
            self.Nq = 10001
            self.xarr = np.linspace(-2*self.max_position,2*self.max_position, 1001)
            self.qarr = np.linspace(-5,5, self.Nq)
            self.dx, self.dq = self.xarr[1] - self.xarr[0], self.qarr[1] - self.qarr[0]
            self.time_evolution = time_evolution_quantum

            self.psi_0 = utilt.gaussian_wavepaket(self.xarr, 0, 0, 1.0)
            self.psi_1 = utilt.deriv_first(self.xarr, self.psi_0)
            self.psi_2 = utilt.deriv_first(self.xarr, self.psi_1)
            self.phi_0 = utilt.gaussian_wavepaket(self.qarr, 0, 0, self.sigma)
            self.phi_1 = -1j*utilt.gaussian_wavepaket_1deriv(self.qarr, 0, 0, self.sigma)
            self.phi_2 = -1*utilt.gaussian_wavepaket_2deriv(self.qarr, 0, 0, self.sigma)

            self.example_pos = utilt.example_meas_pos(self.xarr, self.qarr, self.psi_0, self.phi_0, self.phi_1, self.phi_2, self.L)
            self.example_mom = utilt.example_meas_mom(self.qarr, self.psi_0, self.psi_1, self.psi_2, self.phi_0, self.phi_1, self.phi_2, self.L) 
            self.V = eval(potential + 'V(self)')
         

        if estimator == "kalman" or estimator == "extended_kalman":
            self.R = np.diag([ (self.sigma/self.L/np.sqrt(N_meas))**2, (self.sigma/self.L/np.sqrt(N_meas))**2 ])
            self.R_inv = np.linalg.inv(np.diag([ (self.sigma/self.L)**2, (self.sigma/self.L)**2 ]))
            self.Q = np.ones((2,2)) * 2*(self.sigma/self.L * (self.L**2/(2*self.sigma**2)   / (1/(2*1.41**2))))**2 /np.sqrt(N_meas)
            self.U = np.ones((2,2)) * (self.sigma/self.L)**2/100 /np.sqrt(N_meas)

            self.T = self.U.dot(np.linalg.inv(self.R))
            self.A_mod = self.A - self.T.dot(self.C)
            self.Q_mod = self.Q - self.T.dot(np.transpose(self.U))
            self.kalman_gain , _, _= ct.dlqe(self.A_mod, np.diag([1,1]), self.C, self.Q_mod, self.R)

        if controller == "lqr" or controller == "elqr":
            self.A2 = np.array([[1 , (self.dt * self.N_meas) /self.m], [self.k*(self.dt * self.N_meas), 1]])
            self.B2 = np.array([[0], [(self.dt * self.N_meas)]])         
            self.Q2 = np.diag([self.k/2, 1/(self.m*2)])
            self.R2 = np.diag([0.])
            self.K, _, _ = ct.dlqr(self.A2, self.B2, self.Q2, self.R2)

        self.estimator = eval(estimator)
        self.controller = eval('control_' + controller)        
        self.filter_model = filter_model
        self.Jacobian = eval('Jacobian_' + potential)
        self.Jacobian(self)

        self.reset()
    

    def step(self, action):
        self.action = action        
        self.state[:] = 0.
        reward = 0
        
        for i in range(self.N_meas):
            self.Jacobian(self)
            self.number_of_steps_taken += 1

            if i == 0:
                self.control[1] = self.controller(self) 
            self.time_evolution(self) 
            self.estimator(self) 
            reward += self.reward_func(self)
            self.return_output(self)

            if np.abs(self.actual_pos[0]) > self.max_position:
                break

        # print(self.actual_pos)
        if np.abs(self.actual_pos[0]) > self.max_position:
            self.rewards += reward
            self.run_count += 1
            return self.state,  reward, True, {} 
        elif self.number_of_steps_taken >= self.termination:
            self.rewards +=  reward
            self.run_count += 1                
            return self.state, reward  , True, {}
        else:
            self.rewards += reward
            return self.state, reward , False, {} 

    def reset(self):

        if self.system == 'classical':
            # self.actual_pos[0] = np.random.normal(0, self.mu[0], 1)
            # self.actual_pos[1] = np.random.normal(0, self.mu[1], 1)
            self.actual_pos[0] = np.random.uniform(low=-self.mu[0], high=self.mu[0])
            self.actual_pos[1] = np.random.uniform(low=-self.mu[1], high=self.mu[1])            
        elif self.system == 'quantum':
            self.psi_0 = utilt.gaussian_wavepaket(self.xarr, 0, np.random.normal(0, self.mu[1], 1), 1)

        self.P = 1. * self.P_reset
        self.estimate[:] = 0.
        self.estimate_mean[:] = 0.
        self.control[:] = 0.
        self.state[:] = 0.
        self.measurement[:] = 0.
        # self.rewards = 0
        self.number_of_steps_taken = 0
        return self.state
    
    def reset_train(self):
        self.run_count = 0
        self.rewards = 0

    def seed(self, seed=None):
        np.random.seed(seed)
        return [seed]


    def render(self, mode='both'):
        from gym.envs.classic_control import rendering
        screen_width = 600
        screen_height = 400

        scaleX = screen_width/(2*8)
        scaleY = screen_height*0.50 #Use only 80% of screen height

        if self.viewer is None:
            self.viewer = rendering.Viewer(screen_width, screen_height)
            self.th_left = rendering.Line(((-self.max_position + 8)*scaleX, 0), ((-self.max_position + 8)*scaleX, screen_height))
            self.th_right = rendering.Line(((self.max_position + 8)*scaleX, 0), ((self.max_position + 8)*scaleX, screen_height))
            self.viewer.add_geom(self.th_left)# Add threshold markers
            self.viewer.add_geom(self.th_right)

            fname = path.join(path.dirname(__file__), "assets/arrow.png")
            self.img = rendering.Image(fname, 100.0, 100.0)
            self.imgtrans = rendering.Transform()
            self.img.add_attr(self.imgtrans)
            self.imgtrans.translation = (100, 100)

        self.curve_real = rendering.make_polyline(zip( (self.xarr + 8)*scaleX, np.real(self.psi_0)*scaleY + screen_height*0.5))
        self.curve_im = rendering.make_polyline(zip( (self.xarr + 8)*scaleX, np.imag(self.psi_0)*scaleY + screen_height*0.5))
        # self.curve_prob = rendering.make_polyline(zip( (self.xarr + 8)*scaleX, self.probability_distribution*scaleY + screen_height*0.1))
        # self.curve_potential = rendering.make_polyline(zip( (self.xarr + 8)*scaleX, (self.potential(self.action))*scaleY/np.max(self.potential(self.action)) + screen_height*0.1))

        self.curve_real.set_linewidth(2)
        self.curve_im.set_linewidth(2)
        # self.curve_prob.set_linewidth(2)
        # self.curve_potential.set_linewidth(2)

        self.curve_real.set_color(255/255, 51/255, 51/255)
        self.curve_im.set_color(51/255, 133/255, 255/255)

        self.viewer.add_geom(self.curve_im)     
        self.viewer.add_geom(self.curve_real)
        # self.viewer.add_geom(self.curve_prob)
        # self.viewer.add_geom(self.curve_potential)

        
        # self.imgtrans.scale = (2*(self.action/(self.action_number-1) - 0.5) , 2*np.abs(self.action/(self.action_number-1) - 0.5))
        self.viewer.add_onetime(self.img)
        return self.viewer.render(return_rgb_array=mode == "rgb_array")

    def close(self):
        if self.viewer:
            self.viewer.close()
            self.viewer = None