

# 2-Link Robot Arm Kinematics Simulator

An interactive Python-based simulator of a 2-link planar robotic arm. This project demonstrates the core concepts of robotics: **Forward Kinematics (FK)** and **Inverse Kinematics (IK)**.

---

## 🚀 Features

* **Real-time Interactivity**: Click anywhere within the workspace to move the arm instantly.
* **Inverse Kinematics (IK)**: Calculates joint angles accurately using the geometric approach.
* **Multiple Solutions**: Simultaneously displays both **Elbow Up** and **Elbow Down** configurations.
* **Forward Kinematics (FK)**: Renders the physical state of the robot based on the calculated angles.
* **Live Data Monitor**: A dedicated side-panel displaying real-time data.
* **Workspace Validation**: Automatically detects if a target is out of range.

---

## 🛠 Requirements

* Python 3.x
* NumPy, Matplotlib

```bash
pip install numpy matplotlib
```

---

## 📖 Mathematical Foundation

### 1. Forward Kinematics (FK)
$$x = L_1 \cos(\theta_1) + L_2 \cos(\theta_1 + \theta_2)$$
$$y = L_1 \sin(\theta_1) + L_2 \sin(\theta_1 + \theta_2)$$

### 2. Inverse Kinematics (IK)
**Step A: Solve for $\theta_2$**
$$\cos(\theta_2) = \frac{x^2 + y^2 - L_1^2 - L_2^2}{2 L_1 L_2}$$

**Step B: Solve for $\theta_1$**
$$\theta_1 = \text{atan2}(y, x) - \text{atan2}(L_2 \sin\theta_2, L_1 + L_2 \cos\theta_2)$$

---

## 🎮 How to Use

1. Click inside the workspace circle.
2. Red line = Elbow Up, Blue dashed = Elbow Down.
3. Check the right panel for live angles.

---

## 📂 Project Structure
* `main.py`: Core simulation code.
* `README.md`: Documentation.