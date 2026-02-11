# 连杆机器人运动学仿真器 (2-Link Robot Arm Kinematics Simulator)

这是一个基于 Python 的 2 连杆平面机器人手臂交互式仿真程序。该项目展示了机器人学中的核心概念：**正向运动学 (FK)** 和 **逆向运动学 (IK)**。

---

## 🚀 功能特性

* **实时交互**：点击工作空间内的任意位置，机械臂将实时移动。
* **逆向运动学 (IK)**：使用几何法精确计算关节角度。
* **多解展示**：同时显示 **肘部向上 (Elbow Up)** 和 **肘部向下 (Elbow Down)** 两种配置。
* **正向运动学 (FK)**：根据计算出的角度渲染机器人的物理状态。
* **实时数据监控**：侧边栏面板以度为单位显示目标坐标和关节角度。
* **工作空间验证**：自动检测目标点是否超出可达范围。

---

## 🛠️ 环境要求

* **Python 3.x**
* **NumPy**
* **Matplotlib**

安装依赖库：
\```bash
pip install numpy matplotlib
\```

---

## 📖 数学原理

### 1. 正向运动学 (Forward Kinematics, FK)
已知关节角度 $\theta_1$ 和 $\theta_2$，末端执行器的位置 $(x, y)$ 公式为：

$$x = L_1 \cos(\theta_1) + L_2 \cos(\theta_1 + \theta_2)$$
$$y = L_1 \sin(\theta_1) + L_2 \sin(\theta_1 + \theta_2)$$



### 2. 逆向运动学 (Inverse Kinematics, IK)
通过目标位置 $(x, y)$ 利用 **余弦定理** 求解关节角：

**步骤 A：求解 $\theta_2$**
$$\cos(\theta_2) = \frac{x^2 + y^2 - L_1^2 - L_2^2}{2 L_1 L_2}$$

**步骤 B：求解 $\theta_1$**
$$\theta_1 = \operatorname{atan2}(y, x) - \operatorname{atan2}(L_2 \sin\theta_2, L_1 + L_2 \cos\theta_2)$$



---

## 🎮 使用方法

1.  **运行脚本**: 在终端输入 `python main.py`。
2.  **交互**: 在弹出的窗口内点击灰色圆圈内的任意位置。
3.  **观察**: 
    * **红色实线**: 肘部向上解。
    * **蓝色虚线**: 肘部向下解。
    * **右侧面板**: 实时更新的数据监控。

---

## 📂 项目结构
* `main.py`: 核心仿真源代码。
* `README.md`: 项目说明文档（本文件）。
