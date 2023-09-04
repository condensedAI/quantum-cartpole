import numpy as np
import numba as nb

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
def deriv_first(x, y): ##polynomial fit of 3 points then derivation (compare with simpson's rule, but derivation)
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

@nb.jit
def example_meas_pos(x_space, q_space, psi, phi0, phi1, phi2, L):    
    meas = np.zeros(q_space.size, dtype = np.cdouble)
    for i in range(q_space.size):
        post = m_op(psi,x_space*psi,x_space**2*psi,phi0[i],phi1[i],phi2[i],L)
        meas[i] = np.sum(np.conj(post)*post)
    return meas

@nb.jit
def example_meas_mom(q_space, psi0, psi1, psi2, phi0, phi1, phi2, L):    
    meas = np.zeros(q_space.size, dtype = np.cdouble)
    for i in range(q_space.size):
        post = m_op(psi0, psi1, psi2, phi0[i], phi1[i], phi2[i], L)
        meas[i] = np.sum(np.conj(post)*post)
    return meas

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

def wm_pos(env):
    meas = env.example_pos
    x0 = env.xarr[np.argmax(np.real(np.conj(env.psi_0)*env.psi_0))]
    q_max_index = np.clip(np.random.choice(env.Nq, 1, p=np.real(meas/np.sum(meas)))[0] + int(np.round(x0*env.L/env.dq, 0)),0,env.Nq-1)
    post = m_op(env.psi_0,env.xarr*env.psi_0,env.xarr**2*env.psi_0,env.phi_0[q_max_index],env.phi_1[q_max_index],env.phi_2[q_max_index],env.L)
    return env.qarr[q_max_index], post / np.sqrt(np.sum(np.conj(post)*post*env.dx))

@nb.jit()
def mom_meas(q_space, psi0, psi1, psi2, phi0, phi1, phi2, L):
    meas = np.zeros(q_space.size, dtype = np.cdouble)
    for i in range(q_space.size):
        post = m_op(psi0, psi1, psi2, phi0[i], phi1[i], phi2[i], L)
        meas[i] = np.sum(np.conj(post)*post)

    return meas

def wm_mom(env):
    meas = env.example_mom
    p0 = np.real(np.sum(env.psi_1*np.conj(env.psi_0)*env.dx))
    q_max_index = np.clip(np.random.choice(env.Nq, 1, p=np.real(meas/np.sum(meas)))[0] + int(np.round(p0*env.L/env.dq, 0)),0,env.Nq-1)
    post = m_op(env.psi_0, env.psi_1, env.psi_2, env.phi_0[q_max_index], env.phi_1[q_max_index], env.phi_2[q_max_index], env.L)
    return env.qarr[q_max_index],  post/np.sqrt(np.sum(np.conj(post)*post*env.dx))