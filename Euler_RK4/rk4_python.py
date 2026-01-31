import numpy as np
import matplotlib.pyplot as plt

#setting initial conditions and parameters
delta_t = 0.001
theta_0 = 0.1
omega_0 = 0.0
g = 10.0
L= 10.0
time_end = 10.0

def func_pendulum(y):
    theta, omega = y
    dtheta_dt = omega
    domega_dt = -g/L * np.sin(theta)
    
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
dydt[0, :] = np.array([omega_0, -(g/L)*np.sin(theta_0)])

for i in range(1, t.shape[0]):
    
    k1 = func_pendulum(y[i-1])
    k2 = func_pendulum(y[i-1] + 0.5 * k1 * delta_t)
    k3 = func_pendulum(y[i-1] + 0.5 * k2 * delta_t)
    k4 = func_pendulum(y[i-1] + k3 * delta_t)

    #Updating the values using RK4 formula
    y[i, :] = y[i-1, :] + (delta_t / 6.0) * (k1 + 2*k2 + 2*k3 + k4)
    dydt[i, :] = func_pendulum(y[i, :])
#plotting the results
plt.figure()
plt.plot(t, y[:, 0], label='Angle (theta)')
plt.plot(t, y[:, 1], label='Angular Velocity (omega)')
plt.xlabel('Time (s)')
plt.ylabel('Values')
plt.title('Simple Pendulum Motion using RK4 Method')
plt.legend()
plt.grid()
plt.savefig('rk4_pendulum.png')
plt.show()

energy = 0.5 * (L**2) * y[:,1]**2 + g * L * (1 - np.cos(y[:,0]))

plt.figure()
plt.plot(t, energy)
plt.xlabel("Time (s)")
plt.ylabel("Energy")
plt.title("Energy Conservation Check")
plt.grid()
plt.savefig('rk4_energy.png')
plt.show()