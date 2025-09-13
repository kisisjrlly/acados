from acados_template import AcadosModel, AcadosOcp, AcadosOcpSolver
import casadi as ca
import numpy as np
import time
import matplotlib.pyplot as plt

# 1. 定义模型 (AcadosModel)
model = AcadosModel()
model.name = "mass"

ocp = AcadosOcp()

# 定义状态变量 (position, velocity)
q = ca.SX.sym('q')
q_dot = ca.SX.sym('q_dot')
x = ca.vertcat(q, q_dot)
model.x = x

# 定义控制输入 (force)
F = ca.SX.sym('F')
u = ca.vertcat(F)
model.u = u

# 定义局部参数 (control weight) - np is the number of local parameters
control_weight = ca.SX.sym('control_weight')
model.p = ca.vertcat(control_weight)
ocp.dims.np = 1  # Set the number of local parameters

# 定义模型参数 (mass) - now a local parameter
mass = ca.SX.sym('mass')
# We won't use p_global in this example of time-varying parameters
# model.p_global = ca.vertcat([mass])
ocp.dims.np += 1 # Increase local parameter count for mass

# Define the dynamics with the local mass parameter
q_ddot = F / model.p[1]
x_dot = ca.vertcat(q_dot, q_ddot)

# 设置显式和隐式微分方程
model.f_expl = x_dot
model.f_impl = x - (x + model.T * x_dot)
model.T = 0.1

# 2. 定义 OCP (AcadosOcp)

ocp.model = model
ocp.dims.N = 20
ocp.dims.nx = model.x.shape[0]
ocp.dims.nu = model.u.shape[0]
# ocp.dims.np_global = 0 # Not using global parameters in this time-varying example

# 设置初始状态
ocp.constraints.x0 = np.array([0.0, 0.0])

# 设置控制输入约束
ocp.constraints.lbu = np.array([-10.0])
ocp.constraints.ubu = np.array([10.0])
ocp.constraints.idxbu = np.array([0])

# 设置代价函数 (使用 EXTERNAL cost_type and local parameter)
ocp.cost.cost_type = 'EXTERNAL'
ocp.cost.cost_type_e = 'EXTERNAL'

# 定义外部阶段代价函数 (using local parameter p[0] for control weight)
def external_cost(x, u, p):
    return 1.0 * (x[0] - 2.0)**2 + 0.1 * x[1]**2 + p[0] * u[0]**2

ocp.cost.cost_ext_cost = external_cost

# 定义外部终点代价函数
def external_cost_e(x, p):
    return 10.0 * (x[0] - 2.0)**2 + 0.5 * x[1]**2

ocp.cost.cost_ext_cost_e = external_cost_e

# (可选) 提供外部代价函数的雅可比矩阵
def external_cost_jac(x, u, p):
    grad_x = ca.SX.horzcat([2.0 * (x[0] - 2.0), 0.2 * x[1]])
    grad_u = ca.SX.horzcat([2.0 * p[0] * u[0]])
    grad_p = ca.SX.horzcat([u[0]**2])
    return ca.SX.horzcat([grad_x, grad_u, grad_p])

ocp.cost.cost_ext_cost_jac = external_cost_jac

def external_cost_jac_e(x, p):
    grad_x = ca.SX.horzcat([20.0 * (x[0] - 2.0), 1.0 * x[1]])
    grad_p = ca.SX.horzcat([0.0])
    return ca.SX.horzcat([grad_x, grad_p])

ocp.cost.cost_ext_cost_jac_e = external_cost_jac_e

# 3. 配置求解器 (AcadosOcpSolverConfig)

ocp.solver_options.qp_solver = 'PARTIAL_CONDENSING_HPIPM'
ocp.solver_options.hessian_approx = 'GAUSS_NEWTON'
ocp.solver_options.integrator_type = 'ERK'
ocp.solver_options.nlp_solver_type = 'SQP'

# 4. 创建求解器 (AcadosOcpSolver)
ocp_solver = AcadosOcpSolver(ocp)

# 5. 设置 time-varying local parameters
time_steps = np.linspace(0, ocp.dims.N * model.T, ocp.dims.N + 1)
control_weights = 0.1 + 2 * np.sin(2 * np.pi * time_steps / (ocp.dims.N * model.T))
mass_value = 1.0 # Constant mass for this example

for i in range(ocp.dims.N):
    # Set the control weight (p[0]) for the current stage
    ocp_solver.set(i, 'p', np.array([control_weights[i], mass_value]))

# Set the parameters for the terminal cost (if it used them)
ocp_solver.set(ocp.dims.N, 'p', np.array([control_weights[ocp.dims.N], mass_value]))

# 6. 求解 OCP
simX = np.ndarray((ocp.dims.N + 1, ocp.dims.nx))
simU = np.ndarray((ocp.dims.N, ocp.dims.nu))
times = np.ndarray((ocp.dims.N + 1,))

start_time = time.time()

status = ocp_solver.solve()

if status != 0:
    raise Exception(f'acados returned status {status}.')

# 获取结果
simX[:, :] = ocp_solver.get_prediction_states()
simU[:, :] = ocp_solver.get_prediction_controls()
times[:] = np.array([model.T * k for k in range(ocp.dims.N + 1)])

end_time = time.time()
solve_time = end_time - start_time
print(f"Solve time: {solve_time:.4f} seconds")

# 绘制结果
plt.figure()
plt.subplot(2, 1, 1)
plt.plot(times, simX[:, 0], label='position (q)')
plt.plot(times, simX[:, 1], label='velocity (q_dot)')
plt.ylabel('States')
plt.legend()
plt.grid(True)
plt.title('Simulation with Time-Varying Control Weight')

plt.subplot(2, 1, 2)
plt.step(times[:-1], simU[:, 0], label='force (F)', where='post')
plt.ylabel('Control Input')
plt.xlabel('Time (s)')
plt.legend()
plt.grid(True)

plt.figure()
plt.plot(times[:-1], control_weights[:-1], label='Control Weight (p[0])')
plt.xlabel('Time (s)')
plt.ylabel('Control Weight')
plt.legend()
plt.grid(True)
plt.title('Time-Varying Control Weight')

plt.show()