# Quantum Particle Detection with Absorbing Boundary Condition

This project provides a visual for the evolution of a particle's wave function in one space dimension with hard detectors represented mathematically as absorbing boundary conditions in a 1D Schrödinger model. We are currently working on extending this to two dimensions. This idea was introduced [here](https://arxiv.org/abs/1911.12730).

### Absorbing boundary condition
On the [0,1] interval, we use this absorbing boundary condition on the endpoints. For k > 0, probability "flows out" from [0,1].

<img src="visuals/abc.png" alt="Absorbing boundary condition" width="150">

## Methodology

We use the Crank-Nicolson method to solve a 1D time-independent Schrödinger equation with the above absorbing boundary conditions. The initial wave function is a Gaussian wave packet centered as x = 0.5.

## Animation

![Animation](visuals/0.125x_-4KSidebySidePsi2.gif)
