#
# Copyright (c) The acados authors.
#
# This file is part of acados.
#
# The 2-Clause BSD License
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:
#
# 1. Redistributions of source code must retain the above copyright notice,
# this list of conditions and the following disclaimer.
#
# 2. Redistributions in binary form must reproduce the above copyright notice,
# this list of conditions and the following disclaimer in the documentation
# and/or other materials provided with the distribution.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
# ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE
# LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
# CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
# SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
# INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
# CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
# ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
# POSSIBILITY OF SUCH DAMAGE.;
#


# 导入NumPy库用于数值计算和数组操作
import numpy as np
# 导入块对角矩阵函数，用于构建权重矩阵
from scipy.linalg import block_diag
# 导入acados优化控制问题类和求解器
from acados_template import AcadosOcpSolver, AcadosOcp
# 导入CasADi的垂直连接函数，用于构建代价函数表达式
from casadi import vertcat


def export_mhe_solver_with_param(model, N, h, Q, Q0, R, use_cython=False):
    """
    创建并配置移动视界估计(MHE)求解器。
    
    参数:
        model: CasADi模型对象，包含系统动力学方程和变量定义
        N: int，估计窗口长度（离散时间步数）
        h: float，采样时间步长
        Q: numpy.ndarray，过程噪声权重矩阵
        Q0: numpy.ndarray，到达代价（初始状态）权重矩阵
        R: numpy.ndarray，测量噪声权重矩阵
        use_cython: bool，是否使用Cython加速（默认False）
        
    返回:
        acados_solver_mhe: AcadosOcpSolver，配置好的MHE求解器实例
    """
    
    # 创建一个新的acados优化控制问题对象
    ocp_mhe = AcadosOcp()
    
    # 设置模型
    ocp_mhe.model = model
    
    # 提取增广状态维度（通常包括实际状态加一个积分状态）
    nx_augmented = model.x.rows()
    # 提取参数维度
    nparam = model.p.rows()
    # 计算原始状态维度（去除增广部分）
    nx = nx_augmented-1
    
    # 计算中间阶段输出向量维度（测量 + 过程噪声）
    ny = R.shape[0] + Q.shape[0]                    # h(x), w
    # 设置终端阶段输出向量维度为0（不使用终端代价）
    ny_e = 0
    # 计算初始阶段输出向量维度（测量 + 过程噪声 + 到达代价）
    ny_0 = R.shape[0] + Q.shape[0] + Q0.shape[0]    # h(x), w and arrival cost
    
    # 设置估计窗口长度（离散时间步数）
    ocp_mhe.solver_options.N_horizon = N
    
    # 获取模型中的状态和控制输入变量引用，便于后续使用
    x = ocp_mhe.model.x
    u = ocp_mhe.model.u
    
    # 设置代价函数类型
    # 中间阶段使用非线性最小二乘
    ocp_mhe.cost.cost_type = 'NONLINEAR_LS'
    # 终端阶段使用线性最小二乘（实际上这里ny_e=0，所以不会使用）
    ocp_mhe.cost.cost_type_e = 'LINEAR_LS'
    # 初始阶段使用非线性最小二乘
    ocp_mhe.cost.cost_type_0 = 'NONLINEAR_LS'
    
    # 设置初始阶段权重矩阵（R:测量噪声, Q:过程噪声, Q0:到达代价）
    ocp_mhe.cost.W_0 = block_diag(R, Q, Q0)
    # 设置初始阶段代价函数表达式（状态前nx个分量、输入和完整状态）
    ocp_mhe.model.cost_y_expr_0 = vertcat(x[:nx], u, x)
    # 设置初始阶段参考值为零向量（最小化偏差）
    ocp_mhe.cost.yref_0 = np.zeros((ny_0,))
    
    # 设置中间阶段权重矩阵（R:测量噪声, Q:过程噪声）
    ocp_mhe.cost.W = block_diag(R, Q)
    # 设置中间阶段代价函数表达式（状态前nx个分量和输入）
    ocp_mhe.model.cost_y_expr = vertcat(x[0:nx], u)
    
    # 初始化参数值为零向量
    ocp_mhe.parameter_values = np.zeros((nparam, ))
    
    # 为所有阶段设置参考值
    # 中间阶段参考值
    ocp_mhe.cost.yref  = np.zeros((ny,))
    # 终端阶段参考值（虽然ny_e=0）
    ocp_mhe.cost.yref_e = np.zeros((ny_e, ))
    # 初始阶段参考值（再次设置以确保正确初始化）
    ocp_mhe.cost.yref_0  = np.zeros((ny_0,))
    
    # 设置QP求解器选项
    # 另一种可选的求解器选项（注释掉了）
    # ocp_mhe.solver_options.qp_solver = 'PARTIAL_CONDENSING_HPIPM'
    # 设置为使用完全凝聚的qpOASES求解器
    ocp_mhe.solver_options.qp_solver = 'FULL_CONDENSING_QPOASES'
    # 使用Gauss-Newton方法近似Hessian矩阵
    ocp_mhe.solver_options.hessian_approx = 'GAUSS_NEWTON'
    # 使用显式Runge-Kutta积分器离散化系统动力学
    ocp_mhe.solver_options.integrator_type = 'ERK'
    # 所有阶段的代价函数缩放因子设为1（不缩放）
    ocp_mhe.solver_options.cost_scaling = np.ones((N+1, ))
    
    # 设置预测时域长度（总时间 = 步数 × 时间步长）
    ocp_mhe.solver_options.tf = N*h
    
    # 设置非线性规划(NLP)求解器为序列二次规划(SQP)
    ocp_mhe.solver_options.nlp_solver_type = 'SQP'
    # 设置NLP求解器最大迭代次数
    ocp_mhe.solver_options.nlp_solver_max_iter = 200
    # 设置C代码生成目录
    ocp_mhe.code_export_directory = 'mhe_generated_code'
    
    # 基于use_cython参数决定使用哪种求解器实现
    if use_cython:
        # 生成代码并保存配置到JSON文件
        AcadosOcpSolver.generate(ocp_mhe, json_file='acados_ocp_mhe.json')
        # 构建C代码并编译为Cython模块（性能更好）
        AcadosOcpSolver.build(ocp_mhe.code_export_directory, with_cython=True)
        # 创建Cython版本的求解器实例
        acados_solver_mhe = AcadosOcpSolver.create_cython_solver('acados_ocp_mhe.json')
    else:
        # 创建标准Python版本的求解器实例
        acados_solver_mhe = AcadosOcpSolver(ocp_mhe, json_file = 'acados_ocp_mhe.json')
    
    # 在求解器实例中显式设置初始阶段到达代价权重矩阵
    # 这确保了到达代价被正确应用，即使求解器创建过程中有潜在问题
    acados_solver_mhe.cost_set(0, "W", block_diag(R, Q, Q0))
    
    # 返回配置好的MHE求解器
    return acados_solver_mhe