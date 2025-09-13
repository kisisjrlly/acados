# ACADOS最优控制问题详解 - 完整学习笔记

## 1. 什么是最优控制问题？

最优控制问题就是在满足某些约束条件下，找到一个控制策略，使得系统的性能指标达到最优。

**生活中的例子**：
想象你开车从A点到B点：
- **状态变量**：你的位置、速度
- **控制变量**：油门、刹车
- **目标**：在有限时间内安全到达，同时节省油耗
- **约束**：不能超速，不能撞车

## 2. 数学建模

### 2.1 系统状态方程

这个代码求解的是一个**离散时间线性系统**：

$$x_{k+1} = A \cdot x_k + B \cdot u_k + b$$

其中：
- $x_k \in \mathbb{R}^2$：第k步的状态向量（2维）
- $u_k \in \mathbb{R}^1$：第k步的控制输入（1维）
- $A \in \mathbb{R}^{2 \times 2}$：状态转移矩阵
- $B \in \mathbb{R}^{2 \times 1}$：控制输入矩阵
- $b \in \mathbb{R}^{2 \times 1}$：常数偏移向量

### 2.2 具体参数值

```python
# 从代码中提取的参数
A = [[1.0, 0.25],    # 状态转移矩阵
     [0.0, 1.0]]
     
B = [[0.03125],      # 控制矩阵
     [0.25]]
     
b = [[0.2],          # 偏移向量
     [0.2]]
```

**物理意义**：
- 状态可能表示位置和速度：$x = [position, velocity]^T$
- 控制可能表示加速度：$u = [acceleration]$
- 系统演化：下一步的位置和速度取决于当前状态和控制输入

### 2.3 成本函数

我们要最小化总成本：

$$J = \sum_{k=0}^{N-1} L_k(x_k, u_k) + L_N(x_N)$$

其中：
- **阶段成本**：$L_k(x_k, u_k) = \frac{1}{2}(x_k^T Q x_k + u_k^T R u_k + f^T [x_k; u_k])$
- **终端成本**：$L_N(x_N) = \frac{1}{2} x_N^T Q x_N$

参数含义：
- $Q$：状态权重矩阵（惩罚偏离期望状态）
- $R$：控制权重矩阵（惩罚过大的控制输入）
- $f$：线性项系数

```python
Q = [[1.0, 0.0],     # 状态权重矩阵（单位矩阵）
     [0.0, 1.0]]
     
R = [1.0]            # 控制权重（标量）

f = [0.0, 0.0, 0.0]  # 线性项（此例中为0）
```

### 2.4 约束条件

**状态约束**：
$$-1.0 \leq x_{k,1} \leq 1.0$$
$$-1.0 \leq x_{k,2} \leq 1.0$$

**控制约束**：
$$-1.0 \leq u_k \leq 1.0$$

## 3. 代码结构解析

### 3.1 模型设置函数 (`export_parametric_ocp`)

```python
def export_parametric_ocp(param, name="lti", learnable_params=None):
    ocp = AcadosOcp()  # 创建优化控制问题对象
    
    # 1. 定义状态和控制变量
    ocp.model.x = ca.SX.sym("x", 2)  # 2维状态变量
    ocp.model.u = ca.SX.sym("u", 1)  # 1维控制变量
    
    # 2. 设置求解器选项
    ocp.solver_options.N_horizon = 4     # 预测步数
    ocp.solver_options.tf = 8            # 总时间
    ocp.solver_options.integrator_type = 'DISCRETE'  # 离散系统
    
    # 3. 设置参数
    # learnable_params: 可学习的参数（如A, Q, b）
    # non_learnable_params: 固定参数
```

**关键概念**：
- **预测步数N=4**：算法会预测未来4步的最优控制
- **总时间tf=8**：每步时间间隔为8/4=2秒
- **参数化**：某些系统参数可以变化（learnable）

### 3.2 动力学方程实现

```python
def disc_dyn_expr(model):
    param = find_param_in_p_or_p_global(["A", "B", "b"], model)
    return param["A"] @ model.x + param["B"] @ model.u + param["b"]
```

这直接实现了状态方程：$x_{k+1} = Ax_k + Bu_k + b$

### 3.3 成本函数实现

```python
def cost_expr_ext_cost(model):
    x = model.x
    u = model.u
    param = find_param_in_p_or_p_global(["Q", "R", "f"], model)
    
    return 0.5 * (
        ca.transpose(x) @ param["Q"] @ x +      # 状态惩罚项
        ca.transpose(u) @ param["R"] @ u +      # 控制惩罚项
        ca.transpose(param["f"]) @ ca.vertcat(x, u)  # 线性项
    )
```

这实现了二次成本函数：$\frac{1}{2}(x^T Q x + u^T R u + f^T [x; u])$

## 4. 顺序求解 vs 批量求解 - 核心差别

### 4.1 它们求解的是什么问题？

**答案：它们求解的是一系列相关但不同的问题**

| 方面 | 顺序求解 | 批量求解 |
|------|----------|----------|
| **问题性质** | 动态连续的MPC过程 | 静态独立的问题集合 |
| **时间依赖** | 第i步的解影响第i+1步的初始状态 | 所有问题相互独立 |
| **计算方式** | 串行，一个接一个 | 并行，同时计算 |
| **用途** | 真实的控制仿真 | 验证正确性 + 性能测试 |

### 4.2 顺序求解 (`main_sequential`) - MPC仿真过程

```python
def main_sequential(x0, N_sim):
    solver = AcadosOcpSolver(ocp, verbose=False)
    
    for i in range(N_sim):  # N_sim = 128
        # 每次都是一个新的优化问题！
        
        # 1. 重置求解器
        solver.reset()
        
        # 2. 改变参数（模拟参数变化）
        param = ocp.p_global_values.copy()
        param[i % param.shape[0]] += 0.05  # 参数在变化！
        solver.set_p_global_and_precompute_dependencies(param)
        
        # 3. 求解最优控制（从当前状态开始）
        simU[i,:] = solver.solve_for_x0(x0_bar=simX[i, :])
        
        # 4. 获取下一步状态（系统演化）
        simX[i+1,:] = solver.get(1, "x")
        
        # 5. 计算敏感性（梯度）
        sens_adj = solver.eval_adjoint_solution_sensitivity(...)
```

**MPC过程说明**：
- 第1步：从状态 $x_0$ 开始，参数为 $\theta_0$，求解最优控制
- 第2步：从状态 $x_1$ 开始，参数为 $\theta_1$，求解最优控制  
- ...
- 第128步：从状态 $x_{127}$ 开始，参数为 $\theta_{127}$，求解最优控制

这是**Model Predictive Control (MPC)** 策略：
1. 每一步都求解一个有限时域优化问题
2. 只执行第一个控制动作
3. 状态向前演化，重复这个过程

### 4.3 批量求解 (`main_batch`) - 并行验证

```python
def main_batch(Xinit, simU, param_vals, adjoints_ref, tol, num_threads_in_batch_solve=1):
    batch_solver = AcadosOcpBatchSolver(ocp, N_batch, num_threads_in_batch_solve=num_threads_in_batch_solve)
    
    # 并行设置多个问题
    for n in range(N_batch):  # N_batch = 128
        # 设置第n个问题的初始状态（来自顺序求解）
        batch_solver.ocp_solvers[n].constraints_set(0, "lbx", Xinit[n])
        batch_solver.ocp_solvers[n].constraints_set(0, "ubx", Xinit[n])
        batch_solver.ocp_solvers[n].reset()
        # 设置第n个问题的参数（来自顺序求解）
        batch_solver.ocp_solvers[n].set_p_global_and_precompute_dependencies(param_vals[n])
    
    # 一次性并行求解所有128个问题
    batch_solver.solve(N_batch)
    
    # 验证结果一致性
    for n in range(N_batch):
        u = batch_solver.ocp_solvers[n].get(0, "u")
        diff = np.linalg.norm(u-simU[n])
        if not diff < tol:
            raise Exception(f"solution should match sequential call up to {tol}")
```

### 4.4 图形化理解

```
顺序求解（MPC仿真）：
时刻 0: [状态x₀, 参数θ₀] → 求解OCP₀ → 控制u₀ → 状态演化
       ↓
时刻 1: [状态x₁, 参数θ₁] → 求解OCP₁ → 控制u₁ → 状态演化  
       ↓
时刻 2: [状态x₂, 参数θ₂] → 求解OCP₂ → 控制u₂ → 状态演化
       ↓
...    ↓
时刻127: [状态x₁₂₇, 参数θ₁₂₇] → 求解OCP₁₂₇ → 控制u₁₂₇

批量求解（并行验证）：
┌─ [状态x₀, 参数θ₀] → 求解OCP₀ ─┐
├─ [状态x₁, 参数θ₁] → 求解OCP₁ ─┤
├─ [状态x₂, 参数θ₂] → 求解OCP₂ ─┤ 并行执行
├─ ...                        ─┤
└─ [状态x₁₂₇, 参数θ₁₂₇] → 求解OCP₁₂₇ ─┘
```

## 5. 敏感性分析

### 5.1 什么是敏感性？

敏感性分析回答这个问题：**如果参数稍微改变，最优解会如何变化？**

数学上，我们计算：
$$\frac{\partial u^*}{\partial \theta}$$

其中$u^*$是最优解，$\theta$是参数。

### 5.2 伴随方法 (Adjoint Method)

```python
sens_adj = solver.eval_adjoint_solution_sensitivity(
    [(1, np.ones((ocp.dims.nx, 1)))],  # 对状态的权重
    [(1, np.ones((ocp.dims.nu, 1)))]   # 对控制的权重
)
```

伴随方法通过求解伴随方程来高效计算梯度，这对于梯度下降优化算法非常重要。

## 6. 为什么要设计两种求解方式？

### 6.1 验证正确性
确保批量求解的结果与顺序求解完全一致：
```python
for n in range(N_batch):
    u = batch_solver.ocp_solvers[n].get(0, "u")
    diff = np.linalg.norm(u-simU[n])
    if not diff < tol:
        raise Exception(f"solution should match sequential call up to {tol}")
```

### 6.2 性能对比
典型输出例子：
```
main_sequential, reset, set p_global, solve, adjoints and get: 234.567 ms

main_batch: with 1 threads, reset, set x_0 and p_global: 45.123 ms
main_batch: with 1 threads, solve: 178.456 ms

main_batch: with 4 threads, reset, set x_0 and p_global: 45.123 ms  
main_batch: with 4 threads, solve: 56.789 ms
```

**批量求解的优势**：
- 利用多线程并行计算
- 适合训练机器学习模型时的批量处理
- 大幅提升计算效率

### 6.3 机器学习应用
在强化学习或参数学习中，我们经常需要：
1. 收集大量的(状态, 参数, 最优控制)三元组数据
2. 计算关于参数的梯度
3. 批量处理以提高效率

## 7. 代码中的关键技巧

### 7.1 参数分类
```python
learnable_params = ["A", "Q", "b"]  # 可学习的参数
# 其他参数保持固定
```
这样设计是为了机器学习场景，某些系统参数需要通过数据学习。

### 7.2 参数查找函数
```python
def find_param_in_p_or_p_global(param_name: list[str], model: AcadosModel) -> list:
    if model.p == []:
        return {key: model.p_global[key] for key in param_name}
    elif model.p_global is None:
        return {key: model.p[key] for key in param_name}
    else:
        return {
            key: (model.p[key] if key in model.p.keys() else model.p_global[key])
            for key in param_name
        }
```
这个函数智能地在不同参数集合中查找参数，支持灵活的参数配置。

### 7.3 软约束设计
```python
ocp.constraints.idxsbx = np.array([0])  # 软约束索引
ocp.cost.zl = np.array([1e2])           # 软约束惩罚
ocp.cost.zu = np.array([1e2])
```
软约束允许违反某些约束，但会付出代价，增加求解的鲁棒性。

## 8. 实际应用场景

### 8.1 顺序求解适用场景
- **实时MPC控制器**：机器人、无人驾驶车辆的在线控制
- **仿真验证**：验证控制策略的有效性
- **系统辨识**：通过观测数据估计系统参数

### 8.2 批量求解适用场景
- **强化学习训练**：批量计算策略梯度
- **参数优化**：大规模参数搜索和优化
- **离线分析**：批量处理历史数据
- **并行计算**：充分利用多核CPU或GPU

### 8.3 具体应用领域
1. **机器人控制**：机器人路径规划和跟踪
2. **无人驾驶**：车辆轨迹规划和避障
3. **无人机控制**：飞行路径优化和编队控制
4. **强化学习**：模型预测控制作为策略
5. **经济调控**：宏观经济政策优化
6. **化工过程控制**：反应器温度和压力控制

## 9. 总结

### 9.1 核心概念回顾
这个代码实现了一个完整的**参数化模型预测控制**系统：

1. **定义问题**：线性离散系统的最优控制
2. **设置参数**：区分可学习和固定参数  
3. **求解策略**：MPC滚动时域优化
4. **高效计算**：批量并行求解
5. **敏感性分析**：为参数优化提供梯度信息

### 9.2 两种求解方式的本质
- **顺序求解**：模拟真实的MPC过程，问题之间有时序依赖关系
- **批量求解**：将多个问题作为独立的问题并行求解，主要用于验证和性能测试

它们的解应该完全一致（在数值精度范围内），但求解方式和应用场景不同。

### 9.3 现代意义
这种方法在现代智能控制系统中有广泛应用，特别是需要处理不确定性和参数变化的复杂系统。随着机器学习和人工智能的发展，参数化的优化控制方法越来越重要，能够实现自适应和学习型的控制系统。

非常好的问题！这是理解MPC核心概念的关键。让我详细解释一下。

## 是的！每次求解都是一个序列控制问题

### 1. 什么是"一个序列控制问题"？

每次调用`solver.solve_for_x0()`时，求解器实际上在求解这样一个问题：

```
给定当前状态 x₀，找到一个控制序列 [u₀, u₁, u₂, u₃]，
使得预测的状态轨迹 [x₀, x₁, x₂, x₃, x₄] 的总成本最小
```

### 2. 具体的优化问题

在这个代码中，`N_horizon = 4`，所以每次求解的是：

**优化变量**：
$$\mathbf{U} = [u_0, u_1, u_2, u_3]^T \quad \text{(4个控制输入)}$$

**预测的状态序列**：
$$\begin{align}
x_1 &= A x_0 + B u_0 + b \\
x_2 &= A x_1 + B u_1 + b \\
x_3 &= A x_2 + B u_2 + b \\
x_4 &= A x_3 + B u_3 + b
\end{align}$$

**目标函数**：
$$\min_{\mathbf{U}} \quad J = L_0(x_0, u_0) + L_1(x_1, u_1) + L_2(x_2, u_2) + L_3(x_3, u_3) + L_4(x_4)$$

其中：
- $L_k(x_k, u_k) = \frac{1}{2}(x_k^T Q x_k + u_k^T R u_k)$ （阶段成本）
- $L_4(x_4) = \frac{1}{2}x_4^T Q x_4$ （终端成本）

**约束条件**：
$$\begin{align}
&-1.0 \leq u_k \leq 1.0, \quad k = 0,1,2,3 \\
&-1.0 \leq x_{k,i} \leq 1.0, \quad k = 1,2,3,4, \quad i = 1,2
\end{align}$$

### 3. 图形化表示

```
时刻 t: 当前状态 x₀ (已知)
         ↓ 求解优化问题
    找到最优控制序列: [u₀*, u₁*, u₂*, u₃*]
    预测状态轨迹:     [x₀, x₁*, x₂*, x₃*, x₄*]
         ↓ MPC策略：只执行第一个控制
    实际执行: u₀*
    系统演化: x₀ → x₁ (真实的下一个状态)

时刻 t+1: 当前状态 x₁ (新的起点)
         ↓ 重新求解优化问题  
    找到新的最优控制序列: [u₀'*, u₁'*, u₂'*, u₃'*]
    预测新的状态轨迹:     [x₁, x₂'*, x₃'*, x₄'*, x₅'*]
         ↓ 只执行第一个控制
    实际执行: u₀'*
    系统演化: x₁ → x₂
```

### 4. 代码中的体现

```python
def main_sequential(x0, N_sim):
    for i in range(N_sim):  # 128次MPC循环
        # 每次都是求解一个4步预测的序列控制问题！
        
        # 当前状态作为起点
        current_state = simX[i, :]  
        
        # 求解从当前状态开始的4步序列优化问题
        # 返回的是最优控制序列的第一个元素
        optimal_u0 = solver.solve_for_x0(x0_bar=current_state)
        
        # 只执行第一个控制动作
        simU[i,:] = optimal_u0
        
        # 获取系统演化后的下一个状态（成为下次的起点）
        simX[i+1,:] = solver.get(1, "x")  # 这是预测的x₁
```

### 5. 为什么这样设计？

#### 5.1 滚动时域优化 (Receding Horizon)

MPC的核心思想是"滚动时域"：
- **计划长远**：每次都考虑未来4步的影响
- **执行谨慎**：只执行第一步，然后重新规划
- **自适应**：根据新信息不断更新计划

#### 5.2 处理不确定性

```python
# 参数在每次循环中都在变化
param = ocp.p_global_values.copy()
param[i % param.shape[0]] += 0.05  # 模拟系统参数的变化
```

实际系统中，参数可能因为：
- 环境变化（风阻、路面条件）
- 系统磨损（执行器老化）
- 外部干扰（突然的负载变化）

每次重新求解序列问题，可以适应这些变化。

### 6. 与其他控制方法的对比

| 控制方法 | 每次计算 | 特点 |
|----------|----------|------|
| **PID控制** | 一个控制值 | 简单，但无法预测未来 |
| **LQR控制** | 一个控制值 | 最优，但需要线性化 |
| **MPC控制** | 一个控制序列（但只执行第一个） | 能处理约束和非线性，有预测能力 |

### 7. 计算复杂度的考虑

每次求解序列问题的成本：
- **优化变量数量**：4个控制变量（N_horizon = 4）
- **约束数量**：状态约束 + 控制约束 + 动力学约束
- **计算时间**：比PID复杂得多，但现代计算机可以实时处理

这就是为什么代码要测试批量求解的性能——在某些应用中需要快速求解很多这样的序列问题。

### 8. 实际应用示例

**无人驾驶汽车**：
- 每0.1秒重新规划未来2秒的轨迹（20步预测）
- 考虑速度、转向、刹车的组合控制
- 满足道路约束、舒适性约束、安全约束
- 但只执行第一个0.1秒的控制指令

## 总结

是的，**每次求解都是一个序列控制问题**！具体来说：

1. **每次求解**：找到一个4步的最优控制序列
2. **只执行第一步**：保持灵活性和适应性
3. **滚动规划**：用新的状态信息重新规划
4. **处理约束**：序列优化能自然处理各种约束
5. **预测能力**：考虑当前决策对未来的影响

这种"计划长远，执行谨慎"的策略是MPC方法的精髓，也是它在复杂控制问题中如此有效的原因。

在 acados 中，`eval_adjoint_solution_sensitivity` 计算的“灵敏度”指的是一个标量函数 `L` 对参数 `p` 的导数，即 `dL/dp`。这个标量函数 `L` 是通过用户提供的种子（seeds）`seed_x` 和 `seed_u` 隐式定义的，这些种子代表了 `L` 分别对状态 `x` 和控制 `u` 的偏导数 (`∂L/∂x` 和 `∂L/∂u`)。

**返回值的维度为什么是 `(n_batch, n_seeds, np_global)` 而不是 `(n_batch, nx+nu, np_global)`？**

1.  **`n_batch`**: 表示批处理中 OCP (Optimal Control Problem) 的数量。对于批处理中的每一个问题，都会计算一次灵敏度。

2.  **`np_global`**: 表示全局参数 `p` 的维度。灵敏度是针对这 `np_global` 个参数中的每一个计算的。因此，对于每一个 OCP 问题和每一个种子组合，你会得到一个长度为 `np_global` 的向量，表示 `L` 对每个参数的敏感程度。

3.  **`n_seeds`**: 这个维度允许你同时为多个不同的标量目标函数 `L`（由不同的种子定义）计算灵敏度。
    *   你提供的 `seed_x` 的维度是 `(n_batch, nx, n_seeds)`，`seed_u` 的维度是 `(n_batch, nu, n_seeds)`。这里的 `n_seeds` 决定了你希望评估多少个不同的“目标函数”的灵敏度。
    *   对于每一个种子集（即 `seed_x[:, :, i]` 和 `seed_u[:, :, i]`，其中 `i` 是种子索引），都会计算出一个对应的 `dL_i/dp`。

**为什么不是 `(n_batch, nx+nu, np_global)`？**

*   该方法计算的是**标量函数 `L`** 对参数 `p` 的灵敏度 `dL/dp`。即使原始解 `x` 和 `u` 是向量，通过种子 `seed_x` (即 `∂L/∂x`) 和 `seed_u` (即 `∂L/∂u`)，我们将状态和控制的维度“投影”或“加权求和”到了一个标量上。
*   根据链式法则，`dL/dp = (∂L/∂x)^T * (∂x/∂p) + (∂L/∂u)^T * (∂u/∂p)`。
    *   `∂x/∂p` 的维度与 `(nx, np_global)` 相关。
    *   `∂u/∂p` 的维度与 `(nu, np_global)` 相关。
    *   `(∂L/∂x)^T` 对于单个种子和单个批处理项，其维度是 `(1, nx)`。
    *   `(∂L/∂u)^T` 对于单个种子和单个批处理项，其维度是 `(1, nu)`。
*   因此，`(∂L/∂x)^T * (∂x/∂p)` 的结果是一个维度为 `(1, np_global)` 的向量。同样，`(∂L/∂u)^T * (∂u/∂p)` 的结果也是一个维度为 `(1, np_global)` 的向量。
*   将这两部分（以及所有阶段的贡献，这是伴随方法内部处理的）相加，最终得到的 `dL/dp` 对于每个种子和每个批处理问题仍然是一个维度为 `(1, np_global)` 的向量。

所以，返回值的维度是 `(n_batch, n_seeds, np_global)`，因为对于批处理中的每个问题 (`n_batch`) 和每个定义的种子/目标 (`n_seeds`)，都会计算出一个关于所有全局参数 (`np_global`) 的灵敏度向量。

如果你需要的是 `∂x/∂p` 或 `∂u/∂p`（即解的每个分量对每个参数的灵敏度），那将是另一种类型的灵敏度分析（例如前向灵敏度或完整的雅可比矩阵），其结果维度才会包含 `nx` 或 `nu`。而 `eval_adjoint_solution_sensitivity` 专注于计算由种子定义的标量损失函数对参数的梯度。


