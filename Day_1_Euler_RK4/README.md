# Day 1: Numerical Integration Fundamentals

This module focuses on the manual implementation of **Ordinary Differential Equations (ODEs)** using Python. Before introducing TensorFlow or parallelization, this exercise establishes the baseline for how dynamical systems are simulated in code using discrete time steps.

## 📝 Problem Statement: The Simple Pendulum

We model a simple gravity pendulum with length  and mass . The state of the system is defined by the angle  and angular velocity .

The equations of motion are derived from Newton's laws:

## ⚙️ Implementation Details

### The Solver: Euler's Method

This code implements **Euler's Method**, a first-order numerical procedure for solving ODEs with a given initial value. It approximates the curve by using the tangent line at the current point.

For a state vector , the update rule is:

**Parameters used:**

* **Time step ():** `0.001` seconds (High resolution to minimize error)
* **Gravity ():** `10.0` m/s²
* **Length ():** `10.0` m
* **Initial Angle ():** `0.1` rad (Small angle approximation regime)

### Energy Conservation Check

A critical part of this exercise is verifying the accuracy of the integrator. For a frictionless pendulum, total mechanical energy should remain constant.

The code calculates this energy at every time step to monitor numerical drift.

## 📂 File Structure

* `day1_pendulum.py`: The core script containing the derivative function and the simulation loop.
* `euler_pendulum.png`: Visualization of the angle and angular velocity over time.
* `euler_energy.png`: Visualization of the system's total energy over time.

## 📊 Results & Analysis

### 1. Motion Trajectory

The simulation produces sinusoidal motion characteristic of a pendulum.
*(Note: As seen in the plot,  and  are 90 degrees out of phase, as expected.)*

### 2. Numerical Instability (The "Energy Drift")

One of the key takeaways from Day 1 is that **Euler's method is not energy conserving**. Even with a small , the global truncation error accumulates.

* **Observation:** The energy plot (above) will show a gradual increase over time. The system artificially gains energy due to the linear approximation errors, effectively causing the pendulum to swing higher and higher forever. This motivates the need for higher-order solvers like **Runge-Kutta 4 (RK4)** used in Day 2.

## 🚀 Usage

Run the script directly via terminal:

```bash
python day1_pendulum.py

```

Dependencies:

* `numpy`
* `matplotlib`