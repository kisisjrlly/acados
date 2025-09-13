Collecting workspace informationFiltering to most relevant information# Moving Horizon Estimation (MHE) Problem for Pendulum-on-Cart System

Based on the code in your workspace, I'll describe the Moving Horizon Estimation (MHE) problem with parameter estimation that's being solved using acados.

## System Dynamics Model

The system is a pendulum attached to a cart, where we're trying to estimate both the states and the pendulum length parameter `l`. The standard states are:
- Cart position
- Pendulum angle
- Cart velocity
- Angular velocity

For the MHE problem, the state vector is augmented with the parameter `l`:
```
x_augmented = [position, angle, velocity, angular_velocity, pendulum_length]
```

The dynamics are defined in the following files:
- Standard OCP model: `export_pendulum_ode_model()`
- Augmented MHE model: `export_mhe_ode_model_with_param()`

## MHE Objective Function

The MHE objective function minimizes the weighted sum of:
1. Measurement fit errors
2. Process noise terms
3. Arrival cost (prior on initial state)

Mathematically:
$$\min_{x_0,...,x_N,w_0,...w_{N-1}} \sum_{k=0}^{N-1} \|y_k - h(x_k)\|^2_{R^{-1}} + \|w_k\|^2_{Q^{-1}} + \|x_0 - \bar{x}_0\|^2_{Q_0^{-1}}$$

Subject to: $x_{k+1} = f(x_k, w_k, u_k)$

Where:
- $x_k$ are the augmented states
- $w_k$ are the process noises
- $y_k$ are the measurements
- $u_k$ are control inputs
- $\bar{x}_0$ is the prior estimate
- $R$, $Q$, and $Q_0$ are weighting matrices

The weights are defined in the code:
```python
Q0_mhe = np.diag([0.1, 0.01, 0.01, 0.01, 10])  # Arrival cost
Q_mhe = 10.*np.diag([0.1, 0.01, 0.01, 0.01])   # Process noise
R_mhe = 0.1*np.diag(1./v_stds**2)              # Measurement noise
```

## Solver Configuration

The acados solver is configured in export_mhe_solver_with_param.py:

```python
ocp_mhe.solver_options.qp_solver = 'FULL_CONDENSING_QPOASES'
ocp_mhe.solver_options.hessian_approx = 'GAUSS_NEWTON'
ocp_mhe.solver_options.integrator_type = 'ERK'
ocp_mhe.solver_options.nlp_solver_type = 'SQP'
ocp_mhe.solver_options.nlp_solver_max_iter = 200
```

## Solution Process

The code follows these steps:

1. **Generate nominal trajectory** using OCP solver:
```python
status = acados_solver_ocp.solve()
for i in range(N):
    simX[i,:] = acados_solver_ocp.get(i, "x")
    simU[i,:] = acados_solver_ocp.get(i, "u")
    simY[i,:] = simX[i,:] + np.transpose(np.diag(v_stds) @ np.random.standard_normal((nx, 1)))
```

2. **Set up the MHE problem** with simulated measurements:
```python
# Set arrival cost and first measurement
yref_0 = np.zeros((2*nx + nx_augmented, ))
yref_0[:nx] = simY[0, :]
yref_0[2*nx:] = x0_bar
acados_solver_mhe.set(0, "yref", yref_0)
acados_solver_mhe.set(0, "p", simU[0,:])

# Set measurements for remaining stages
for j in range(1, N):
    yref[:nx] = simY[j, :]
    acados_solver_mhe.set(j, "yref", yref)
    acados_solver_mhe.set(j, "p", simU[j,:])
```

3. **Solve the MHE problem**:
```python
status = acados_solver_mhe.solve()
```

4. **Extract the solution** (estimated states and parameter):
```python
for i in range(N):
    x_augmented = acados_solver_mhe.get(i, "x")
    simXest[i,:] = x_augmented[0:nx]
    sim_l_est[i,:] = x_augmented[nx]
    simWest[i,:] = acados_solver_mhe.get(i, "u")
```

## Parameter Estimation

The true value of parameter `l` is 0.8:
```python
l_true = 0.8
```

The initial guess is 1.0 (from `x0_bar = np.array([0.0, np.pi, 0.0, 0.0, 1])`).

After solving, the code evaluates the estimation quality:
```python
print('difference |l_est - l_true|', np.abs(sim_l_est[0] - l_true))
```

This MHE formulation successfully demonstrates joint state and parameter estimation for a pendulum-on-cart system using acados.

Similar code found with 1 license type