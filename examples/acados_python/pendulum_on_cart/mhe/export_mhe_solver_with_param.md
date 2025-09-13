# acados OCP配置项详解

下面是代码中`ocp_mhe`对象的各个配置项及其详细含义：

## 1. 基本模型设置
```python
ocp_mhe.model = model
```
- **含义**：设置优化问题的系统模型，包含系统动力学、状态变量、控制输入和参数定义

## 2. 时域设置
```python
ocp_mhe.solver_options.N_horizon = N
ocp_mhe.solver_options.tf = N*h
```
- **N_horizon**：设置MHE问题的视界长度（估计窗口中的离散时间步数）
- **tf**：总预测时间长度（N乘以时间步长h）

## 3. 代价函数类型
```python
ocp_mhe.cost.cost_type = 'NONLINEAR_LS'
ocp_mhe.cost.cost_type_e = 'LINEAR_LS'
ocp_mhe.cost.cost_type_0 = 'NONLINEAR_LS'
```
- **cost_type**：中间阶段(1到N-1)的代价函数类型，使用非线性最小二乘
- **cost_type_e**：终端阶段(N)的代价函数类型，使用线性最小二乘
- **cost_type_0**：初始阶段(0)的代价函数类型，使用非线性最小二乘

## 4. 权重矩阵设置
```python
ocp_mhe.cost.W_0 = block_diag(R, Q, Q0)
ocp_mhe.cost.W = block_diag(R, Q)
```
- **W_0**：初始阶段代价函数的权重矩阵，结合了测量、过程和初始状态权重
- **W**：中间阶段代价函数的权重矩阵，只包含测量和过程权重

## 5. 代价函数表达式
```python
ocp_mhe.model.cost_y_expr_0 = vertcat(x[:nx], u, x)
ocp_mhe.model.cost_y_expr = vertcat(x[0:nx], u)
```
- **cost_y_expr_0**：初始阶段代价函数中要最小化的表达式
- **cost_y_expr**：中间阶段代价函数中要最小化的表达式

## 6. 参考值设置
```python
ocp_mhe.cost.yref_0 = np.zeros((ny_0,))
ocp_mhe.cost.yref = np.zeros((ny,))
ocp_mhe.cost.yref_e = np.zeros((ny_e, ))
```
- **yref_0**：初始阶段的参考值，设为零表示最小化偏差
- **yref**：中间阶段的参考值
- **yref_e**：终端阶段的参考值

## 7. 参数值设置
```python
ocp_mhe.parameter_values = np.zeros((nparam, ))
```
- **parameter_values**：模型参数的初始值，创建求解器后可更新

## 8. QP求解器设置
```python
ocp_mhe.solver_options.qp_solver = 'FULL_CONDENSING_QPOASES'
ocp_mhe.solver_options.hessian_approx = 'GAUSS_NEWTON'
ocp_mhe.solver_options.integrator_type = 'ERK'
ocp_mhe.solver_options.cost_scaling = np.ones((N+1, ))
```
- **qp_solver**：二次规划子问题的求解器类型，使用完全凝聚+QPOASES
- **hessian_approx**：Hessian矩阵的近似方法，使用Gauss-Newton
- **integrator_type**：用于离散化连续动力学的积分器类型，使用显式Runge-Kutta
- **cost_scaling**：各阶段代价函数的缩放因子

## 9. NLP求解器设置
```python
ocp_mhe.solver_options.nlp_solver_type = 'SQP'
ocp_mhe.solver_options.nlp_solver_max_iter = 200
```
- **nlp_solver_type**：非线性规划求解类型，使用序列二次规划
- **nlp_solver_max_iter**：求解器的最大迭代次数

## 10. 代码生成设置
```python
ocp_mhe.code_export_directory = 'mhe_generated_code'
```
- **code_export_directory**：生成C代码的目标目录

这些配置共同定义了一个完整的移动视界估计问题，包括其动力学模型、代价函数结构、求解器特性和代码生成选项。