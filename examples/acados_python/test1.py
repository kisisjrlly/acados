from acados_template import AcadosModel, AcadosOcp, AcadosOcpSolver
import casadi as ca
import numpy as np
import scipy.linalg

# --- Model definition ---
def export_pendulum_ode_model_with_cost_params() -> AcadosModel:
    model_name = 'pendulum_ode_time_varying_params'

    # Constants
    M = 1.      # mass of the cart [kg]
    m = 0.1     # mass of the ball [kg]
    l = 0.8     # length of the rod [m]
    g = 9.81    # gravity constant [m/s^2]

    # Set up states & controls
    x1 = ca.SX.sym('x1')        # cart position [m]
    theta = ca.SX.sym('theta')  # pendulum angle [rad]
    v1 = ca.SX.sym('v1')        # cart velocity [m/s]
    dtheta = ca.SX.sym('dtheta')# pendulum angular velocity [rad/s]

    x = ca.vertcat(x1, theta, v1, dtheta)
    u = ca.SX.sym('F')          # force on cart [N]

    # Dynamics
    x1_dot = v1
    theta_dot = dtheta
    v1_dot = (l*m*ca.sin(theta)*dtheta**2 + u + g*m*ca.sin(theta)*ca.cos(theta))/(M + m - m*ca.cos(theta)**2)
    dtheta_dot = -(l*m*ca.cos(theta)*ca.sin(theta)*dtheta**2 + u*ca.cos(theta) + (M+m)*g*ca.sin(theta))/(l*(M + m - m*ca.cos(theta)**2))
    f_expl = ca.vertcat(x1_dot, theta_dot, v1_dot, dtheta_dot)

    model = AcadosModel()
    model.x = x
    model.u = u
    model.f_expl_expr = f_expl
    
    nx = model.x.rows()
    nu = model.u.rows()

    model.xdot = ca.SX.sym('xdot', nx, 1)
    model.f_impl_expr = model.xdot - f_expl
    model.name = model_name

    # --- Define symbolic parameters for Q and R matrices ---
    # These parameters will be set at each time stage using acados_solver.set(stage, "p", ...)
    p_Q_flat = ca.SX.sym('p_Q_flat', nx * nx) # Flattened Q matrix
    p_R_flat = ca.SX.sym('p_R_flat', nu * nu) # Flattened R matrix
    model.p = ca.vertcat(p_Q_flat, p_R_flat) # Stage parameters

    # --- Define external cost expressions using these parameters ---
    Q_sym = ca.reshape(p_Q_flat, (nx, nx))
    R_sym = ca.reshape(p_R_flat, (nu, nu))

    # Cost for intermediate stages (0 to N-1): 0.5 * (x'Qx + u'Ru)
    model.cost_expr_ext_cost = 0.5 * (model.x.T @ Q_sym @ model.x + model.u.T @ R_sym @ model.u)
    
    # Cost for terminal stage (N): 0.5 * x'Qx (uses the Q part of model.p)
    # The R part of model.p for the terminal stage will be present in the parameter vector 
    # but ignored by this specific cost_expr_ext_cost_e.
    model.cost_expr_ext_cost_e = 0.5 * (model.x.T @ Q_sym @ model.x)

    return model

# --- OCP Definition and Solver ---
def main():
    # Create OCP object
    ocp = AcadosOcp()

    # Export model with cost parameters
    model = export_pendulum_ode_model_with_cost_params()
    ocp.model = model

    nx = model.x.rows()
    nu = model.u.rows()
    
    # Set dimensions
    N_horizon = 20
    Tf = 1.0  # Prediction horizon time
    ocp.dims.N = N_horizon

    # Set cost type to EXTERNAL
    # This means acados will use model.cost_expr_ext_cost for stages 0 to N-1
    ocp.cost.cost_type = 'EXTERNAL'
    # This means acados will use model.cost_expr_ext_cost_e for stage N
    ocp.cost.cost_type_e = 'EXTERNAL' 

    # Set constraints
    Fmax = 80.0
    ocp.constraints.lbu = np.array([-Fmax])
    ocp.constraints.ubu = np.array([+Fmax])
    ocp.constraints.idxbu = np.array([0])

    x0 = np.array([0.0, np.pi, 0.0, 0.0]) # Initial state: pendulum upright, at rest
    ocp.constraints.x0 = x0

    # Set solver options
    ocp.solver_options.qp_solver = 'PARTIAL_CONDENSING_HPIPM'
    ocp.solver_options.hessian_approx = 'EXACT' # CasADi computes exact Hessian for EXTERNAL cost
    ocp.solver_options.integrator_type = 'IRK'
    ocp.solver_options.nlp_solver_type = 'SQP'
    ocp.solver_options.tf = Tf
    # ocp.solver_options.print_level = 1 # Uncomment for more verbose output

    # --- Initialize ocp.parameter_values before creating the solver ---
    # The shape must be (N_horizon + 1, number_of_parameters_in_model.p)
    # model.p consists of flattened Q (nx*nx) and R (nu*nu)
    np_cost = nx * nx + nu * nu
    ocp.parameter_values = np.zeros((np_cost))

    # Create AcadosOcpSolver
    # IMPORTANT: We are NOT setting ocp.parameter_values before creating the solver.
    # The parameters for each stage will be set individually later.
    acados_solver = AcadosOcpSolver(ocp, json_file=f'{model.name}.json')

    # --- Set time-varying Q and R matrices using acados_solver.set() for each stage ---
    # The field "p" corresponds to ocp.model.p for that specific stage.
    # We iterate from stage 0 to N_horizon (which is stage N in acados).
    for k in range(N_horizon + 1): # Iterate over stages 0, 1, ..., N
        params_for_stage_k_flat = np.zeros(nx*nx + nu*nu)
        if k < N_horizon: # Stages 0 to N-1 (intermediate stages)
            # Example: Make Q and R slightly different at each stage
            Q_k = np.diag([1e3 * (1 + 0.05*k), 1e3 * (1 + 0.05*k), 1e-2, 1e-2])
            R_k = np.diag([1e-2 * (1 + 0.02*k)])
            params_for_stage_k_flat[:nx*nx] = Q_k.flatten()
            params_for_stage_k_flat[nx*nx:] = R_k.flatten()
        else: # Stage N_horizon (terminal stage N)
            # For the terminal stage, cost_expr_ext_cost_e is used.
            # It only depends on Q in our example.
            Q_N = np.diag([2e3, 2e3, 2e-2, 2e-2]) # Different terminal Q
            R_N_dummy = np.zeros((nu, nu))       # R part is needed for consistent param vector size
                                                 # even if not used by cost_expr_ext_cost_e.
            params_for_stage_k_flat[:nx*nx] = Q_N.flatten()
            params_for_stage_k_flat[nx*nx:] = R_N_dummy.flatten()
        
        # Set the parameters 'p' for stage 'k'
        acados_solver.set(k, "p", params_for_stage_k_flat)

    print(f"Solving OCP for {model.name} with time-varying Q & R (EXTERNAL cost, params set after solver creation)...")
    status = acados_solver.solve()

    if status != 0:
        print(f'acados_solver.solve() returned status {status}. Printing statistics:')
        acados_solver.print_statistics()
        return
    else:
        print(f'Solution found successfully!')

    acados_solver.print_statistics()

    # Get solution
    simX = np.zeros((N_horizon + 1, nx))
    simU = np.zeros((N_horizon, nu))
    for i in range(N_horizon):
        simX[i,:] = acados_solver.get(i, "x")
        simU[i,:] = acados_solver.get(i, "u")
    simX[N_horizon,:] = acados_solver.get(N_horizon, "x")

    print("\n--- First few states and controls ---")
    print("X[0:3,:]:\n", simX[0:3,:])
    print("U[0:3,:]:\n", simU[0:3,:])

    # --- Example of updating parameters at runtime for stage 0 and re-solving ---
    print("\n--- Updating Q and R for stage 0 and re-solving ---")
    Q_0_new = np.diag([5e3, 5e3, 5e-2, 5e-2]) # New Q for stage 0
    R_0_new = np.diag([5e-3])                 # New R for stage 0
    param_0_new_flat = np.concatenate((Q_0_new.flatten(), R_0_new.flatten()))

    # Set the parameters 'p' specifically for stage 0
    acados_solver.set(0, "p", param_0_new_flat)

    status_update = acados_solver.solve()

    if status_update != 0:
        print(f'acados_solver.solve() after update returned status {status_update}.')
    else:
        print(f'Solution found successfully after update!')
    
    u0_updated = acados_solver.get(0, "u")
    print(f"New control u[0] after update: {u0_updated}")

if __name__ == '__main__':
    main()