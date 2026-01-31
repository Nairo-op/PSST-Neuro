import numpy as np
import matplotlib.pyplot as plt
import tensorflow.compat.v1 as tf
tf.disable_eager_execution()



def tf_check_type(t, y0): # Ensure Input is Correct
    """
    This function checks the type of the input to ensure that it is a floating point number.
    """
    if not (y0.dtype.is_floating and t.dtype.is_floating):
        raise TypeError('Error: y0 and t must be floating point numbers.')

class _Tf_Integrator():
    """
    This class implements the Runge-Kutta 4th order method in TensorFlow.
    """
    def integrate(self, func, y0, t):
        """
        This function integrates a function func using the Runge-Kutta 4th order method in TensorFlow.

        Parameters:
        -----------
        func: function
            The function to be integrated.
        y0: float
            The initial condition.
        t: numpy array
            The time array.
        """
        time_delta_grid = t[1:] - t[:-1] # define the time step at each point

        def scan_func(y, t_dt): # define the scan function that performs the integration step
            """
            This function performs the integration step.

            Parameters:
            -----------
            y: float
                The value of y at which the function is being evaluated.
            t_dt: (float, float)
                The time point and time step at which the function is being evaluated.
            """
            t, dt = t_dt # unpack the time point and time step
            dy = self._step_func(func,t,dt,y) # Make code more modular.
            return y + dy

        y = tf.scan(scan_func, (t[:-1], time_delta_grid),y0)
        return tf.concat([[y0], y], axis=0)

    def _step_func(self, func, t, dt, y):
        """
        This function determines the value of the integration step.

        Parameters:
        -----------
        func: function
            The function to be integrated.
        t: float
            The time point at which the function is being evaluated.
        dt: float
            The time step at which the function is being integrated.
        y: float
            The value of y at which the function is being evaluated.
        """
        k1 = func(y, t)
        half_step = t + dt / 2
        dt_cast = tf.cast(dt, y.dtype) # Failsafe

        k2 = func(y + dt_cast * k1 / 2, half_step)
        k3 = func(y + dt_cast * k2 / 2, half_step)
        k4 = func(y + dt_cast * k3, t + dt)
        return tf.add_n([k1, 2 * k2, 2 * k3, k4]) * (dt_cast / 6) # add all update terms

def odeint(func, y0, t):
    """
    This function integrates the function func using the Runge-Kutta 4th order method implemented in the _Tf_Integrator class.

    Parameters:
    -----------
    func: function
        The function to be integrated.
    y0: float
        The initial condition.
    t: numpy array
        The time array.
    """
    # Ensure Input is in the form of TensorFlow Tensors
    t = tf.convert_to_tensor(t, name='t')
    y0 = tf.convert_to_tensor(y0, name='y0')
    tf_check_type(y0,t) # Ensure Input is of the correct type
    return _Tf_Integrator().integrate(func,y0,t)

C_m = 1     # Membrane Capacitance

g_K = 10    # K-channel Conductance
E_K = -95   # K-channel Reversal Potential

g_Na = 100  # Na-channel Conductance
E_Na = 50   # Na-channel Reversal Potential

g_L = 0.15  # Leak Conductance
E_L = -55   # Leak Reversal Potential

def K_prop(V):
    """
    This function determines the K-channel gating dynamics.

    Parameters:
    -----------
    V: float
        The membrane potential.
    """
    T = 22 # Temperature
    phi = 3.0**((T-36.0)/10) # Temperature-correction factor
    V_ = V-(-50) # Voltage baseline shift

    alpha_n = 0.02*(15.0 - V_)/(tf.exp((15.0 - V_)/5.0) - 1.0) # Alpha for the K-channel gating variable n
    beta_n = 0.5*tf.exp((10.0 - V_)/40.0) # Beta for the K-channel gating variable n

    t_n = 1.0/((alpha_n+beta_n)*phi) # Time constant for the K-channel gating variable n
    n_0 = alpha_n/(alpha_n+beta_n) # Steady-state value for the K-channel gating variable n

    return n_0, t_n


def Na_prop(V):
    """
    This function determines the Na-channel gating dynamics.

    Parameters:
    -----------
    V: float
        The membrane potential.
    """
    T = 22 # Temperature
    phi = 3.0**((T-36)/10)  # Temperature-correction factor
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

def I_K(V, n):
    """
    This function determines the K-channel current.

    Parameters:
    -----------
    V: float
        The membrane potential.
    n: float
        The K-channel gating variable n.
    """
    return g_K  * n**4 * (V - E_K)

def I_Na(V, m, h):
    """
    This function determines the Na-channel current.

    Parameters:
    -----------
    V: float
        The membrane potential.
    m: float
        The Na-channel gating variable m.
    h: float
        The Na-channel gating variable h.
    """
    return g_Na * m**3 * h * (V - E_Na)

def I_L(V):
    """
    This function determines the leak current.

    Parameters:
    -----------
    V: float
        The membrane potential.
    """
    return g_L * (V - E_L)

def dXdt(X, t):
    """
    This function determines the derivatives of the membrane voltage and gating variables for a single neuron.

    Parameters:
    -----------
    X: float
        The state vector given by the [V, n, m, h] where V is the membrane potential, n is the K-channel gating variable, m and h are the Na-channel gating variables.
    t: float
        The time points at which the derivatives are being evaluated.
    """
    V = X[0:1] # The first element of the state vector is the membrane potential
    m = X[1:2] # The second element of the state vector is the Na-channel gating variable m
    h = X[2:3] # The third element of the state vector is the Na-channel gating variable h
    n = X[3:4] # The fourth element of the state vector is the K-channel gating variable n

    # Note that here we dont index the elements directly because we want the values as a tensor rather than a single value

    dVdt = (5 - I_Na(V, m, h) - I_K(V, n) - I_L(V)) / C_m  # The derivative of the membrane potential
    # Here the current injection I_injected = 5 uA

    m0,tm,h0,th = Na_prop(V) # Calculate the dynamics of the Na-channel gating variables
    n0,tn = K_prop(V) # Calculate the dynamics of the K-channel gating variables

    dmdt = - (1.0/tm)*(m-m0) # The derivative of the Na-channel gating variable m
    dhdt = - (1.0/th)*(h-h0) # The derivative of the Na-channel gating variable h
    dndt = - (1.0/tn)*(n-n0) # The derivative of the K-channel gating variable n

    out = tf.concat([dVdt,dmdt,dhdt,dndt],0) # Concatenate the derivatives into a single tensor
    return out

y0 = tf.constant([-71,0,0,0], dtype=tf.float64) # Initial conditions

epsilon = 0.01 # The step size for the numerical integration
t = np.arange(0,200,epsilon) # The time points at which the numerical integration is being performed

state = odeint(dXdt,y0,t) # Solve the differential equation

with tf.Session() as sess:
    state = sess.run(state) # Run the session

# Plot the membrane potential

plt.plot(t,state.T[0,:])
plt.xlabel("Time (in ms)")
plt.ylabel("Voltage (in mV)")
plt.title("Hodgkin-Huxley Neuron Model Simulation")
plt.savefig("hh_model_simulation.png")
plt.show()