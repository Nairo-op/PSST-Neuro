# PSST: Parallel Scalable Simulations of Biological Neural Networks

### Implementation & Exercises (Days 1-3)

This repository contains my implementation of the first three modules of the **PSST (Parallel Scalable Simulations with TensorFlow)** tutorial series. The project focuses on building a foundation for computational neuroscience by implementing numerical integration methods from scratch and utilizing TensorFlow's data-flow paradigm to simulate biological neuronal dynamics.

## 📂 Project Overview

The core objective of this work was to transition from standard procedural programming to **vectorized, graph-based computation** suitable for simulating large-scale biological networks.

### Modules Completed

#### **Day 1: Numerical Integration Fundamentals**

* **Core Concepts:** Ordinary Differential Equations (ODEs), discretization of time, and numerical stability.
* **Implementation:**
* Developed a manual implementation of **Euler's Method** in native Python.
* Simulated simple 1D dynamical systems (e.g., exponential growth/decay) to visualize integration error accumulation over time.



#### **Day 2: TensorFlow for Differential Equations**

* **Core Concepts:** Computational graphs, tensors, and parallelized operations on GPUs/CPUs.
* **Implementation:**
* Translated the Euler integrator into a **TensorFlow computational graph**.
* Implemented the **Runge-Kutta 4 (RK4)** method for higher-precision integration.


* **Showcase Exercise (Lorenz Attractor):**
* Simulated the chaotic **Lorenz Attractor** system using the custom TensorFlow integrators.
* Modeled the coupled non-linear equations:


* *Output:* Generated 3D phase-space trajectories demonstrating sensitivity to initial conditions.



#### **Day 3: Cells in Silicon (Biophysical Modeling)**

* **Core Concepts:** Biophysics of excitable membranes, Nernst potentials, and ion channel gating.
* **Implementation:**
* Modeled the passive properties of a cell membrane (RC circuit equivalent).
* Simulated the dynamics of a **Single-Compartment Neuron** (e.g., Hodgkin-Huxley type dynamics) by integrating conductance-based equations.
* Visualized membrane potential () spikes and ionic currents over time.



## 🛠️ Tech Stack

* **Language:** Python 3.x
* **Libraries:** TensorFlow (v1.x/v2.x compat), NumPy, Matplotlib
* **Methods:** Euler Integration, Runge-Kutta 4 (RK4)

## 📊 Visualizations


| Lorenz Attractor (Day 2) |
| <img width="1200" height="500" alt="lorentz_euler_rk4_tf" src="https://github.com/user-attachments/assets/7612eaaf-81ec-4b87-a1d6-a820697d1ec7" /> |
| -------------------------------------------------------------------------------------------------------------------------------------------------- |
| Neuron Spike Train (Day 3)                                                                                                                         |
 <img width="1200" height="1700" alt="image" src="https://github.com/user-attachments/assets/056e19e9-cafc-4e66-84ac-8bf7f141d54e" />
 |

## 🔗 References & Acknowledgements

This work is based on the **PSST** tutorial and the associated paper by the **Theoretical & Computational Neuroscience Lab (IISER Pune)**.

* **Original Repository:** [neurorishika/PSST](https://github.com/neurorishika/PSST)
* **Paper:**
> Mohanta, R., & Assisi, C. (2019). **Parallel scalable simulations of biological neural networks using TensorFlow: A beginner's guide.** *arXiv preprint arXiv:1906.03958*. [Link to Paper](https://arxiv.org/abs/1906.03958)
