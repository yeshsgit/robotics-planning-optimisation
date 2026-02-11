--- START ---

# 2-Link Robot Arm Kinematics Simtlator

This is an interactive Python-based simulation of a 2-link planar robotic arm. This project demonstrates the core concepts of robotics: **Forward Kinematics (FK)** and **Inverse Kinematics (IK)**.

---

## 🚀 Features

* **Real-time Interaction**: Click anywhere within the workspace to move the arm instantly.
* **Inverse Kinematics (IK)**: Calculates joint angles accurately using the geometric approach (Law of Cosines).
* **Multiple Solutions**: Simultaneously displays both **Elbow Up** and **Elbow Down** configurations.
* **Forward Kinematics (FK)**: Renders the physical state of the robot based on the calculated angles.
* **Live Data Monitor**: A dedicated side-panel displaying real-time target coordinates and joint angles in degrees.
* **Workspace Validation**: Automatically detects if a target point is outside the reachable workspace.

---

## 🛠 Requirements

* Python 3.x
* NumPy (for matrix and trigonometric calculations)
* Matplotlib (for the interactive GUI and rendering)

Install the dependencies:

```bash
pip install numpy matplotlib
```

---

## 📖 Mathematical Foundation

### 1. Forward Kinematics (FK)
Given the joint angles $\theta_1$ and $\theta_2$ the end-effector position $(x, y)$ is calculated by summing the vectors of each link:

$$x = L_1 \cos(\theta_1) + L_2 \cos(\theta_1 + \theta_2)$$
$$y = L_1 \sin(\theta_1) + L_2 \sin(\theta_1 + \theta_2)$$

### 2. Inverse Kinematics (IK)
Given a target position $(x, y)$, the joint angles are derived using the **Law of Cosines** to solve the triangle formed by $L_1$, $L_2$, and the distance to the target $d = \sqrt{x^2 + y^2}$.

**Step A: Solve for $\theta_2$**
$$\cos(\theta_2) = \frac{x^2 + y^2 - L_1^2 - L_2^2}{2 L_1 L_2}$$

**Step B: Solve for $\theta_1$**
$$\theta_1 = \operatorname{atan2}(y, x) - \operatorname{atan2}(L_2 \sin\theta_2, L_1 + L_2 \cos\theta_2)$$

---

## 🎮 How th Use

1. **Run the script**: Enter `python main.py` in your terminal.
2. **Interact**: A window titled "2-Link Robot Arm IK/FK Controller" will appear.
3. **Click**: Click anywhere inside the dashed gray circle (the reachable workspace).
    * **Red Solid Line**: Represents the "Elbow Up" configuration.
    * **Blue Dashed Line**: Represents the "Elbow Down" configuration.
4. **Monitor**: View the live angle updates and coordinate data in the monitor box on the right side.

---

## 📂 Project Structure
* `main.py`: Core simulation source code.
* `README.md`: Project documentation (this file).
