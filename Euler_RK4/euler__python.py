import numpy as np
import matplotlib.pyplot as plt

#defining initial conditions and parameters
delta_t = 0.001
theta_0 = 0.1
omega_0 = 0.0
g = 10.0
L= 10.0
time_end = 10.0


def func_pendulum(theta, omega):
    dtheta_dt = omega
    domega_dt = -g/L * np.sin(theta)  #linearized for small angles
    
    return np.array([ 
        dtheta_dt,
        domega_dt
        ])


#setting up arrays to hold time, angle, and angular velocity
t = np.arange(0, time_end, delta_t)
y = np.zeros((t.shape[0], 2))
dydt = np.zeros((t.shape[0], 2))


#initializing the first values
y_0 = np.array([theta_0, omega_0])
y[0, :] = y_0
dydt[0, :] = func_pendulum(theta_0, omega_0)

#Euler's method to solve the ODEs
for i in range(1, t.shape[0]):
    y[i, :] = y[i-1, :] + dydt[i-1, :] * delta_t
    dydt[i, :] = func_pendulum(y[i, 0], y[i, 1])

#plotting the results
plt.figure()
plt.plot(t, y[:, 0], label='Angle (theta)')
plt.plot(t, y[:, 1], label='Angular Velocity (omega)')
plt.xlabel('Time (s)')
plt.ylabel('Values')
plt.title('Simple Pendulum Motion using Euler\'s Method')
plt.legend()
plt.grid()
plt.savefig('euler_pendulum.png')
plt.show()

energy = 0.5 * (L**2) * y[:,1]**2 + g * L * (1 - np.cos(y[:,0]))
plt.figure()
plt.plot(t, energy)
plt.xlabel("Time (s)")
plt.ylabel("Energy")
plt.title("Energy Conservation Check")
plt.grid()
plt.savefig('euler_energy.png')
plt.show()