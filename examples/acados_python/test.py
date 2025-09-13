from acados_template import AcadosOcp, AcadosOcpSolver
import numpy as np
import scipy.linalg
import matplotlib.pyplot as plt
import casadi as ca

def create_ocp_solver(N_horizon, nx, nu, time_varying_Q=True):
    # 创建OCP对象
    ocp = AcadosOcp()
    
    # 设置基本参数
    ocp.model_name = 'mpc_example'
    
    # 状态和控制变量
    x = ca.SX.sym('x', nx)
    u = ca.SX.sym('u', nu)
    
    # 创建参数符号 - 用于时变Q矩阵
    p = ca.SX.sym('p', nx*nx)
    
    # 系统参数和状态空间方程 (简单的双重积分器系统)
    A = np.array([
        [1.0, 0.1],
        [0.0, 1.0]
    ])
    B = np.array([
        [0.005],
        [0.1]
    ])
    
    # 使用CasADi定义系统动态方程
    x_next = ca.mtimes(A, x) + ca.mtimes(B, u)
    
    # 设置OCP模型
    ocp.model.x = x
    ocp.model.u = u
    ocp.model.p = p  # 设置参数
    ocp.model.disc_dyn_expr = x_next  # 离散动力学
    
    # 设置预测时域长度
    ocp.dims.N = N_horizon
    
    # 初始条件约束
    ocp.constraints.x0 = np.array([1.0, 0.0])
    
    # 控制输入约束
    ocp.constraints.lbu = np.array([-10.0])  # 下限
    ocp.constraints.ubu = np.array([10.0])   # 上限
    ocp.constraints.idxbu = np.array([0])    # 施加约束的控制输入索引
    # ...existing code...
    # 成本函数设置 - 使用参数化的Q矩阵
    if time_varying_Q:
        # 非线性最小二乘成本函数 - 允许时变Q
        ocp.cost.cost_type = 'NONLINEAR_LS'
        ocp.cost.cost_type_e = 'NONLINEAR_LS'
        
        # 创建参数化的二次型成本
        Q_param = ca.reshape(p, (nx, nx))  # 将参数p重构为矩阵形式
        R_numpy = np.eye(nu) * 0.1 # nu x nu numpy matrix
        
        # 定义残差函数
        # cost_y_expr 的长度是 1 (来自状态项) + nu (来自控制项)
        ocp.model.cost_y_expr = ca.vertcat(
            ca.sqrt(ca.mtimes(ca.mtimes(x.T, Q_param), x)),  # 权重化状态 (标量)
            ca.mtimes(ca.sqrt(R_numpy), u)                   # 权重化输入 (nu x 1 向量)
        )
        # cost_y_expr_e 的长度是 1 (来自状态项)
        ocp.model.cost_y_expr_e = ca.sqrt(ca.mtimes(ca.mtimes(x.T, Q_param), x))
        
        # 定义输出维度
        ocp.dims.ny = 1 + nu # <--- 修改处
        ocp.dims.ny_e = 1    # <--- 修改处
        
        # 参考轨迹
        ocp.cost.yref = np.zeros(1 + nu) # <--- 修改处
        ocp.cost.yref_e = np.zeros(1)    # <--- 修改处
        
        # 权重矩阵 - 对于非线性最小二乘,权重已经包含在残差中
        # W 和 W_e 应该是单位矩阵，如果成本已经通过 cost_y_expr 正确加权
        ocp.cost.W = np.eye(1 + nu) # <--- 修改处
        ocp.cost.W_e = np.eye(1)    # <--- 修改处
    else:
        # 标准线性最小二乘成本函数 - 使用相同的Q矩阵
        ocp.cost.cost_type = 'LINEAR_LS'
        ocp.cost.cost_type_e = 'LINEAR_LS'
        
        # 设置成本矩阵
        Q = np.eye(nx) * 10.0
        R = np.eye(nu) * 0.1
        
        # 设置输出矩阵
        ocp.cost.Vx = np.zeros((nx+nu, nx))
        ocp.cost.Vx[:nx, :nx] = np.eye(nx)
        
        ocp.cost.Vu = np.zeros((nx+nu, nu))
        ocp.cost.Vu[nx:, :] = np.eye(nu)
        
        ocp.cost.Vx_e = np.eye(nx)
        
        # 定义输出维度
        ocp.dims.ny = nx + nu
        ocp.dims.ny_e = nx
        
        # 参考轨迹
        ocp.cost.yref = np.zeros(nx + nu)
        ocp.cost.yref_e = np.zeros(nx)
        
        # 权重矩阵
        ocp.cost.W = np.block([
            [Q, np.zeros((nx, nu))],
            [np.zeros((nu, nx)), R]
        ])
        ocp.cost.W_e = Q * 5.0  # 终端成本权重更高
    
    # 求解器设置
    ocp.solver_options.qp_solver = 'PARTIAL_CONDENSING_HPIPM'
    ocp.solver_options.hessian_approx = 'GAUSS_NEWTON'
    ocp.solver_options.integrator_type = 'ERK'
    ocp.solver_options.nlp_solver_type = 'SQP'
    ocp.solver_options.qp_solver_cond_N = N_horizon

    # 初始化参数值 - 为参数p提供初始值
    ocp.parameter_values = np.zeros((nx*nx,))
    # ...existing code...
    ocp.model.p = p  # 设置参数
    ocp.model.disc_dyn_expr = x_next  # 离散动力学
    
    # 设置预测时域长度
    ocp.dims.N = N_horizon
    # 设置总时间范围
    dt = 0.1  # 每个离散步骤的时间长度 (例如 0.1 秒)
    ocp.solver_options.tf = N_horizon * dt # 总时间范围
    
    # 初始条件约束
    ocp.constraints.x0 = np.array([1.0, 0.0])
    # ...existing code...
    ocp.solver_options.integrator_type = 'ERK'
    ocp.solver_options.nlp_solver_type = 'SQP'
    ocp.solver_options.qp_solver_cond_N = N_horizon
    # ocp.solver_options.tf = N_horizon * 0.1 # 设置时间范围，例如每个步长0.1秒

    # 初始化参数值 - 为参数p提供初始值
    ocp.parameter_values = np.zeros((nx*nx,))
    # ...existing code...

    # 创建并返回求解器
    return AcadosOcpSolver(ocp, json_file='acados_ocp.json'), ocp

def main():
    # 设置问题维度
    N_horizon = 20  # 预测时域长度
    nx = 2          # 状态维度
    nu = 1          # 控制输入维度
    
    # 创建求解器
    solver, ocp = create_ocp_solver(N_horizon, nx, nu, time_varying_Q=True)
    
    # 创建不同的Q矩阵(随时间变化)
    Q_matrices = []
    for i in range(N_horizon + 1):  # 包括终端成本
        # 随步数增加的Q矩阵 - 终端成本权重最大
        weight = 1.0 + 9.0 * i / N_horizon  # 从1.0线性增加到10.0
        Q = np.eye(nx) * weight
        Q_matrices.append(Q.flatten())  # 将矩阵展平为向量
    
    # 初始状态
    x0 = np.array([1.0, 0.0])
    
    # 求解MPC问题
    solver.set('constr_x0', x0)
    
    # 设置每个阶段的Q矩阵
    for i in range(N_horizon + 1):
        solver.set('p', Q_matrices[i], i)
    
    # 求解OCP
    status = solver.solve()
    
    # 获取结果
    state_traj = np.zeros((N_horizon+1, nx))
    control_traj = np.zeros((N_horizon, nu))
    
    for i in range(N_horizon):
        state_traj[i, :] = solver.get('x', i)
        control_traj[i, :] = solver.get('u', i)
    
    state_traj[N_horizon, :] = solver.get('x', N_horizon)
    
    # 打印统计信息
    print(f"求解状态: {status}")
    print(f"总成本: {solver.get_cost()}")
    
    # 绘制结果
    time_steps = np.arange(N_horizon + 1)
    plt.figure(figsize=(12, 8))
    
    plt.subplot(3, 1, 1)
    plt.step(time_steps, np.append(control_traj, control_traj[-1]), where='post')
    plt.ylabel('控制输入')
    plt.title('MPC求解结果 - 时变Q矩阵')
    plt.grid(True)
    
    plt.subplot(3, 1, 2)
    plt.plot(time_steps, state_traj[:, 0])
    plt.ylabel('位置')
    plt.grid(True)
    
    plt.subplot(3, 1, 3)
    plt.plot(time_steps, state_traj[:, 1])
    plt.ylabel('速度')
    plt.xlabel('预测步数')
    plt.grid(True)
    
    plt.tight_layout()
    plt.show()

    # 展示Q矩阵的变化
    Q_values = np.array([Q_matrices[i][0] for i in range(N_horizon + 1)])
    plt.figure()
    plt.plot(time_steps, Q_values)
    plt.title('Q矩阵权重随时间变化')
    plt.xlabel('预测步数')
    plt.ylabel('Q矩阵(1,1)元素值')
    plt.grid(True)
    plt.show()

if __name__ == "__main__":
    main()