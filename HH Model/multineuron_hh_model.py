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


n_n = 20

C_m = [1.0]*n_n
g_K = [10.0]*n_n
E_K = [-95.0]*n_n

g_Na = [100]*n_n
E_Na = [50]*n_n

g_L = [0.15]*n_n
E_L = [-55.0]*n_n


def K_prop(V):

    T = 22
    phi = 3.0**((T-36.0)/10)
    V_ = V-(-50)

    alpha_n = 0.02*(15.0 - V_)/(tf.exp((15.0 - V_)/5.0) - 1.0)
    beta_n = 0.5*tf.exp((10.0 - V_)/40.0)

    t_n = 1.0/((alpha_n+beta_n)*phi)
    n_0 = alpha_n/(alpha_n+beta_n)

    return n_0, t_n


def Na_prop(V):

    T = 22
    phi = 3.0**((T-36)/10)
    V_ = V-(-50)

    alpha_m = 0.32*(13.0 - V_)/(tf.exp((13.0 - V_)/4.0) - 1.0)
    beta_m = 0.28*(V_ - 40.0)/(tf.exp((V_ - 40.0)/5.0) - 1.0)

    alpha_h = 0.128*tf.exp((17.0 - V_)/18.0)
    beta_h = 4.0/(tf.exp((40.0 - V_)/5.0) + 1.0)

    t_m = 1.0/((alpha_m+beta_m)*phi)
    t_h = 1.0/((alpha_h+beta_h)*phi)

    m_0 = alpha_m/(alpha_m+beta_m)
    h_0 = alpha_h/(alpha_h+beta_h)

    return m_0, t_m, h_0, t_h

def I_K(V, n):

    return g_K  * n**4 * (V - E_K)

def I_Na(V, m, h):

    return g_Na * m**3 * h * (V - E_Na)

def I_L(V):

    return g_L * (V - E_L)

def dXdt(X, t):

    V = X[:1*n_n]
    m = X[1*n_n:2*n_n]
    h = X[2*n_n:3*n_n]
    n = X[3*n_n:]

    dVdt = (np.linspace(0,10,n_n) - I_Na(V, m, h) - I_K(V, n) -I_L(V)) / C_m

    m0,tm,h0,th = Na_prop(V)
    n0,tn = K_prop(V)

    dmdt = - (1.0/tm)*(m-m0)
    dhdt = - (1.0/th)*(h-h0)
    dndt = - (1.0/tn)*(n-n0)

    out = tf.concat([dVdt,dmdt,dhdt,dndt],0)
    return out


y0 = tf.constant([-71]*n_n+[0,0,0]*n_n, dtype=tf.float64)

epsilon = 0.01
t = np.arange(0,200,epsilon)

state = final_ode_int(dXdt,y0,t)

with tf.Session() as sess:
    state = sess.run(state)


# Plot the membrane potentials

plt.figure(figsize=(12,17))
for i in range(20):
    plt.subplot(10,2,i+1)
    plt.plot(t,state[:,i])
    plt.title("Injected Current = {:0.1f}".format(i/2))
    plt.ylim([-90,60])
    plt.xlabel("Time (in ms)")
    plt.ylabel("Voltage (in mV)")
plt.tight_layout()
plt.savefig("multineuron_hh_model.png")
plt.show()