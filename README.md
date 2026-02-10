# PSST — Parallel Scalable Simulations of Biological Neural Networks

**Implementation Notes & Exercises (Days 1–3)**

This repository documents my implementation of the first three modules of the **PSST (Parallel Scalable Simulations with TensorFlow)** tutorial series.

The objective of this project is to transition from **loop-based Python simulations** to **vectorized, graph-based computation** using TensorFlow, with a focus on **numerical methods and biophysical neural modeling** relevant to computational neuroscience.

Rather than treating neural networks as abstract ML objects, this work enabled me to to approach them as **dynamical systems governed by differential equations**, grounded in physics and biology.

---

## Project Overview

### Day 1 — Mathematical Foundations: Time, Stability, and Error

Before modeling neurons, I focused on the numerical foundations.

* Implemented **Euler’s Method** from scratch in native Python
* Simulated simple 1D ODEs (exponential growth/decay)
* Studied **error accumulation**, timestep sensitivity, and numerical stability

I was able to build intuition for why simple discretization fails in long-term dynamical simulations.

---

### Day 2 — Differential Equations in TensorFlow

This stage helped me introduce **graph-based computation** and parallelism.

* Reimplemented the Euler integrator using **TensorFlow computational graphs**
* Implemented **Runge–Kutta 4 (RK4)** for higher-order accuracy
* Compared integrators under identical timesteps

#### Lorenz Attractor (Chaotic System)

To test the integrators further, I modeled the **Lorenz system**, a classic chaotic dynamical system defined by coupled non-linear ODEs.

* Demonstrates **sensitivity to initial conditions**
* Highlights the importance of integrator choice
* Visualized 3D phase-space trajectories

**Outcome:** Clear divergence between Euler and RK4 under chaos, validating numerical expectations.

<p align="center">
  <img width="100%" alt="Lorenz Attractor" src="https://github.com/user-attachments/assets/7612eaaf-81ec-4b87-a1d6-a820697d1ec7">
</p>

---

### Day 3 — Cells in Silicon: Biophysical Neuron Modeling

Numerical tools were then applied to biological systems.

#### Membrane as an RC Circuit

* Modeled the neuronal membrane using an **equivalent RC circuit**
* Simulated charge accumulation and leakage under current injection

#### Single-Compartment Neuron

* Implemented a **conductance-based neuron model** (Hodgkin–Huxley–type dynamics)
* Integrated ionic currents and membrane potential over time
* Generated **action potentials (spikes)** in response to external input
* Visualized membrane voltage and ionic currents

<p align="center">
  <img width="100%" alt="Neuron Spike Train" src="https://github.com/user-attachments/assets/056e19e9-cafc-4e66-84ac-8bf7f141d54e">
</p>

---

## Key Takeaways

* Neural simulations are fundamentally **numerical integration problems**
* Stability and timestep choice are as important as biological realism
* TensorFlow enables scalable, parallel simulation of dynamical systems
* Conductance-based neuron models naturally emerge from physical principles

This project served as a foundation for extending simulations from **single neurons** to **large-scale spiking neural networks**.

---

## My Computational Setup

All simulations were executed **locally** on my personal laptop using **TensorFlow with NVIDIA CUDA support**, which helped enabling GPU-accelerated computation.

* **GPU:** NVIDIA RTX 3050 (Laptop)
* **Acceleration:** CUDA-enabled TensorFlow
* **Execution:** Identical code can run on CPU or GPU without modification

This validates the portability and scalability of the TensorFlow-based simulation approach.

---

## Tech Stack

* **Language:** Python 3.x
* **Numerical & Graph Computation:** TensorFlow
* **Scientific Computing:** NumPy
* **Visualization:** Matplotlib

---

## References & Credits

This work is based on the **PSST tutorial series** by the
**Theoretical & Computational Neuroscience Lab, IISER Pune**.

* Original Repository: [https://github.com/neurorishika/PSST](https://github.com/neurorishika/PSST)
* Paper:
  Mohanta, R., & Assisi, C. (2019).
  *Parallel scalable simulations of biological neural networks using TensorFlow: A beginner’s guide.*
  [https://arxiv.org/abs/1906.03958](https://arxiv.org/abs/1906.03958)

---
