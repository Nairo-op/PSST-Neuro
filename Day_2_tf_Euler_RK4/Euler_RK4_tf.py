import numpy as np
import matplotlib.pyplot as plt
import tensorflow.compat.v1 as tf
from mpl_toolkits import mplot3d

#disable eager execution for TF1 compatibility
tf.disable_eager_execution()

#time parameters
epsilon  = 0.01 
end_time = 50

#time_steps array
dt_loop_array = np.ones(int(end_time/epsilon), dtype=np.float32) * epsilon
t = tf.constant(dt_loop_array)

#Lorentz system parameters and initial conditions
sigma = 10 
beta = 8/3 
rho = 28  
x0, y0, z0 = 1, 1, 1

#define type check function
def tf_check_type(state, t):
    if not (state.dtype.is_floating and t.dtype.is_floating):
        # The datatype of any tensor t is accessed by t.dtype
        raise TypeError('Error: state and t must be floating point numbers')


#define Lorentz system of equations
def lorentz_deriv(X):
    x, y, z = X[0], X[1], X[2]
    dxdt = sigma * (y - x)
    dydt = x * (rho - z) - y
    dzdt = x * y - beta * z
    return tf.stack([dxdt, dydt, dzdt])


#define Euler method
def euler_method(prev_state, h):

    #here, h is a scalar time step
    return prev_state + h * lorentz_deriv(prev_state)


#define RK4 method
def rk4_method(prev_state, h):

    #here too, h is a scalar time step
    k1 = lorentz_deriv(prev_state)
    k2 = lorentz_deriv(prev_state + 0.5 * h * k1)
    k3 = lorentz_deriv(prev_state + 0.5 * h * k2)
    k4 = lorentz_deriv(prev_state + h * k3)
    return prev_state + (h / 6.0) * (k1 + 2*k2 + 2*k3 + k4)


# Euler Implementation
tf.scan_euler = tf.scan(fn=euler_method,
                         elems=t,
                         initializer=tf.constant([x0, y0, z0], dtype=tf.float32))

# RK4 Implementation
tf.scan_rk4 = tf.scan(fn=rk4_method,
                      elems=t,
                      initializer=tf.constant([x0, y0, z0], dtype=tf.float32))


#run session to compute results
with tf.Session() as sess:
    tf_check_type(tf.scan_euler, t)
    tf_check_type(tf.scan_rk4, t)
    euler_result = sess.run(tf.scan_euler)
    rk4_result = sess.run(tf.scan_rk4)

#plot results
#plot Euler and RK4 results side by side
fig = plt.figure(figsize=(12, 5))

#plot Euler
ax1 = fig.add_subplot(121, projection='3d')
ax1.plot(euler_result[:, 0], euler_result[:, 1], euler_result[:, 2], color='blue')
ax1.set_title('Euler Method', fontsize=15)
ax1.set_xlabel('X Axis')
ax1.set_ylabel('Y Axis')
ax1.set_zlabel('Z Axis')

#Plot RK4
ax2 = fig.add_subplot(122, projection='3d')
ax2.plot(rk4_result[:, 0], rk4_result[:, 1], rk4_result[:, 2], color='red')
ax2.set_title('RK4 Method', fontsize=15)
ax2.set_xlabel('X Axis')
ax2.set_ylabel('Y Axis')
ax2.set_zlabel('Z Axis')

#save figure
plt.tight_layout()
plt.savefig('lorentz_euler_rk4_tf.png')


fig1 = plt.figure()
ax = plt.axes(projection='3d')
ax.plot3D(euler_result[:, 0], euler_result[:, 1], euler_result[:, 2], label="Euler Solution for x,y,z in 3D")
plt.xlabel("t")
plt.ylabel("theta/omega")
plt.legend()
plt.show()
plt.savefig('lorentz_euler_tf.png')

fig2 = plt.figure()
ax = plt.axes(projection='3d')
ax.plot3D(rk4_result[:, 0], rk4_result[:, 1], rk4_result[:, 2], label="RK4 Solution for x,y,z in 3D")
plt.xlabel("t")
plt.ylabel("theta/omega")
plt.legend()
plt.show()
plt.savefig('lorentz_rk4_tf.png')
