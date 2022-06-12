import numpy as np
import numba as nb
from scipy.signal import savgol_filter
from scipy.optimize import curve_fit

@nb.jit
def gaussian_wavepaket(x, x0, p0, sig):
    return 1/(2*np.pi*sig**2)**(1/4) *  np.exp(-1/4 * (x - x0)**2/sig**2) * np.exp(1j * x*p0)

@nb.jit
def gaussian_wavepaket_1deriv(x, x0, p0, sig):### analytical derivitav for p0 = 0
    return 1/(2*np.pi*sig**2)**(1/4) *  np.exp(-1/4 * (x - x0)**2/sig**2) * (-1/2 * (x - x0)/sig**2)

@nb.jit
def gaussian_wavepaket_2deriv(x, x0, p0, sig):### analytical derivitav for p0 = 0
    return 1/(2*np.pi*sig**2)**(1/4) *  np.exp(-1/4 * (x - x0)**2/sig**2) * ( (-1/2 * (x - x0)/sig**2)**2 -1/2*1/sig**2)

@nb.jit()
def deriv_first(x, y):
    deriv = np.zeros(y.size, dtype = np.cdouble)
    for i in range(1, y.size-1):
        deriv[i] = np.real(y[i-1])*(2*x[i]-x[i] - x[i+1])/((x[i-1] - x[i])*(x[i-1] - x[i+1])) + \
                    np.real(y[i])*(2*x[i]-x[i-1] - x[i+1])/((x[i] - x[i-1])*(x[i] - x[i+1])) + \
                    np.real(y[i+1])*(2*x[i]-x[i-1] - x[i])/((x[i+1] - x[i-1])*(x[i+1] - x[i]))
        deriv[i] += 1j*(np.imag(y[i-1])*(2*x[i]-x[i] - x[i+1])/((x[i-1] - x[i])*(x[i-1] - x[i+1])) + \
                    np.imag(y[i])*(2*x[i]-x[i-1] - x[i+1])/((x[i] - x[i-1])*(x[i] - x[i+1])) + \
                    np.imag(y[i+1])*(2*x[i]-x[i-1] - x[i])/((x[i+1] - x[i-1])*(x[i+1] - x[i])))        
    return -1j*deriv

@nb.jit
def compute_probability(psi):
    return np.real(np.conj(psi)*psi)

@nb.jit
def compute_total_probability(probability_distribution, dx):
    return np.sum(dx * probability_distribution)

@nb.jit
def fourier(x,p,y):
    transform = np.zeros(y.size, dtype = np.cdouble)
    dx = x[1] - x[0]
    for i in range(p.size):
        transform[i] = 1/np.sqrt(2*np.pi)*np.sum(y * np.exp(-1j*x*p[i])) * dx 
    return transform

#### Weak measurements
@nb.jit
def m_op(x, x1, x2, y, y1, y2, L):
    return x*y - 1j*L*x1*y1 - L**2*0.5*x2*y2

@nb.jit
def pos_meas(x_space, q_space, psi, phi0, phi1, phi2, L):    
    meas = np.zeros(q_space.size, dtype = np.cdouble)
    for i in range(q_space.size):
        post = m_op(psi,x_space*psi,x_space**2*psi,phi0[i],phi1[i],phi2[i],L)
        meas[i] = np.sum(np.conj(post)*post)
    return meas

# @nb.jit
def wm_pos(env):
    meas = pos_meas(env.xarr, env.qarr, env.psi_0, env.phi_0, env.phi_1, env.phi_2, env.L)
    q_max_index = np.random.choice(env.Nq, 1, p=np.real(meas/np.sum(meas)))[0]
    # print(env.qarr[np.argmax(meas)])
    post = m_op(env.psi_0,env.xarr*env.psi_0,env.xarr**2*env.psi_0,env.phi_0[q_max_index],env.phi_1[q_max_index],env.phi_2[q_max_index],env.L)
    return env.qarr[q_max_index], post / np.sqrt(np.sum(np.conj(post)*post*env.dx))

@nb.jit()
def mom_meas(q_space, psi0, psi1, psi2, phi0, phi1, phi2, L):
    meas = np.zeros(q_space.size, dtype = np.cdouble)
    for i in range(q_space.size):
        post = m_op(psi0, psi1, psi2, phi0[i], phi1[i], phi2[i], L)
        meas[i] = np.sum(np.conj(post)*post)

    return meas

# @nb.jit()
def wm_mom(env):
    meas = mom_meas(env.qarr, env.psi_0, env.psi_1, env.psi_2, env.phi_0, env.phi_1, env.phi_2, env.L)
    q_max_index = np.random.choice(env.Nq, 1, p=np.real(meas/np.sum(meas)))[0]
    # print(env.qarr[np.argmax(meas)])
    post = m_op(env.psi_0, env.psi_1, env.psi_2, env.phi_0[q_max_index], env.phi_1[q_max_index], env.phi_2[q_max_index], env.L)
    return env.qarr[q_max_index],  post/np.sqrt(np.sum(np.conj(post)*post*env.dx))