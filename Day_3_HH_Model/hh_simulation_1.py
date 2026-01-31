import numpy as np
import matplotlib.pyplot as plt
import tensorflow.compat.v1 as tf
tf.disable_eager_execution()



def tf_type_check(t, y0):
    if not (y0.dtype.is_floating and t.dtype.is_floating):
        raise TypeError("t and y0 must be floating point tensors")
    

class _Tf_Integrator_functions():

    def integrate(self, func, y0, t):

        dt_grid = t[1:] - t[:-1]

        def scan_loop(y_prev, dt):
            
            t_curr, dt = dt
            y_prev = tf.reshape(y_prev, [-1])
            dy = self.step_fn(func, y_prev, t_curr , dt)
            return y_prev + dy
    
        y = tf.scan(scan_loop, (t[:-1], dt_grid), initializer=y0)
        return tf.concat([[y0], y], axis=0)

    def step_fn(self, func, y, t, dt):
        
        k1 = func(y, t)
        half_step = t + dt / 2
        dt_cast = tf.cast(dt, dtype=y.dtype) 

        k2 = func(y + dt_cast / 2 * k1, half_step)
        k3 = func(y + dt_cast / 2 * k2, half_step)
        k4 = func(y + dt_cast * k3, t + dt)
        dy = tf.add_n([k1, 2 * k2, 2 * k3, k4]) * (dt_cast / 6)
        return dy

def final_ode_int(func, y0, t):

    t = tf.convert_to_tensor(t, name = 't')
    y0 = tf.convert_to_tensor(y0, name = 'y0')
    tf_type_check(t, y0)
    return _Tf_Integrator_functions().integrate(func, y0, t)


#Implementing Hodgkin Huxley Model using the ODE integrator

#parameters
C_m  =   1.0  # membrane capacitance, in uF/cm^2
g_Na = 120.0  # maximum conducances, in mS/cm^2
g_K  =  36.0  # maximum conducances, in mS/cm^2
g_L  =   0.3  # maximum conducances, in mS/cm^2
E_Na =  50.0  # Nernst reversal potentials, in mV
E_K  = -77.0  # Nernst reversal potentials, in mV
E_L  = -54.387  # Nernst reversal potentials, in mV

def ion_K_prop(V):

    T = 22 # temperature in Celsius
    phi = 3.0**((T - 36.0) / 10) # temperature scaling factor
    V_ = V - (-50) # voltage baseline shift


    alpha_n = 0.02*(15.0 - V_)/(tf.exp((15.0 - V_)/5.0) - 1.0) # Alpha for the K-channel gating variable n
    beta_n = 0.5*tf.exp((10.0 - V_)/40.0) # Beta for the K-channel gating variable n

    t_n = 1.0/((alpha_n+beta_n)*phi) # Time constant for the K-channel gating variable n
    n_0 = alpha_n/(alpha_n+beta_n) # Steady-state value for the K-channel gating variable n

    return n_0, t_n

def ion_Na_prop(V):

    T = 22 # Temperature in Celsius
    phi = 3.0**((T-36)/10)  # Temperature scaling factor
    V_ = V-(-50) # Voltage baseline shift

    alpha_m = 0.32*(13.0 - V_)/(tf.exp((13.0 - V_)/4.0) - 1.0) # Alpha for the Na-channel gating variable m
    beta_m = 0.28*(V_ - 40.0)/(tf.exp((V_ - 40.0)/5.0) - 1.0) # Beta for the Na-channel gating variable m

    alpha_h = 0.128*tf.exp((17.0 - V_)/18.0) # Alpha for the Na-channel gating variable h
    beta_h = 4.0/(tf.exp((40.0 - V_)/5.0) + 1.0) # Beta for the Na-channel gating variable h

    t_m = 1.0/((alpha_m+beta_m)*phi) # Time constant for the Na-channel gating variable m
    t_h = 1.0/((alpha_h+beta_h)*phi) # Time constant for the Na-channel gating variable h

    m_0 = alpha_m/(alpha_m+beta_m) # Steady-state value for the Na-channel gating variable m
    h_0 = alpha_h/(alpha_h+beta_h) # Steady-state value for the Na-channel gating variable h

    return m_0, t_m, h_0, t_h


def I_ionK(V, n):

    return g_K * n**4 * (V - E_K)

def I_ionNa(V, m, h):
    
    return g_Na * m**3 * h * (V - E_Na)

def I_L(V):

    return g_L * (V - E_L)

def dx_dt(X, t):
    
    V= X[0:1]
    n= X[1:2]
    m= X[2:3]
    h= X[3:4]

    dVdt = (5- I_ionNa(V, m, h) - I_ionK(V, n) - I_L(V)) / C_m

    n_0, t_n = ion_K_prop(V)
    m_0, t_m, h_0, t_h = ion_Na_prop(V)

    I_ext = 5.0
    dVdt = (I_ext - I_ionNa(V, m, h) - I_ionK(V, n) - I_L(V)) / C_m
    dmdt = - (1.0/t_m)*(m-m_0)
    dhdt = - (1.0/t_h)*(h-h_0)
    dndt = - (1.0/t_n)*(n-n_0)

    return tf.concat([dVdt, dndt, dmdt, dhdt], axis=0)

# Initial conditions
y0 = tf.constant([-71.0, 0.0, 0.0 ,0.0], dtype=tf.float64) # Initial conditions

epsilon = 0.01 # The step size for the numerical integration
t = np.arange(0,200,epsilon) # The time points at which the numerical integration is being performed

state = final_ode_int(dx_dt,y0,t) # Solve the differential equation

with tf.Session() as sess:
    state = sess.run(state) # Run the session


plt.plot(t,state.T[0,:])
plt.xlabel("Time (in ms)")
plt.ylabel("Voltage (in mV)")
plt.title("Hodgkin-Huxley Neuron Model Simulation")
plt.savefig("hh_model_simulation_1.png")
plt.show()