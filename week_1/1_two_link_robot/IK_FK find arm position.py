import matplotlib
matplotlib.use('TkAgg')

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Button, Slider

class RobotArmApp:
    def __init__(self, l1=2.0, l2=1.5):
        self.l1 = l1
        self.l2 = l2
        self.mode = "MENU"

        self.start_point = None
        self.goal_point = None
        self.rrt_lines = []
        self.rrt_nodes = []  # Store node scatter plots
        self.rrt_markers = []  # Store start/goal markers

        # === Figure & workspace ===
        self.fig, self.ax = plt.subplots(figsize=(10, 6))
        plt.subplots_adjust(left=0.05, right=0.68, bottom=0.15)

        self.setup_workspace()
        self.create_right_panel()

        self.fig.canvas.mpl_connect('button_press_event', self.on_click)
        plt.show(block=True)

    # ================= WORKSPACE =================
    def setup_workspace(self):
        limit = (self.l1 + self.l2) * 1.2
        self.ax.set_xlim(-limit, limit)
        self.ax.set_ylim(-limit, limit)
        self.ax.set_aspect('equal')
        self.ax.grid(True, linestyle='--', alpha=0.5)

        circle = plt.Circle((0, 0), self.l1 + self.l2,
                            fill=False, linestyle='--', alpha=0.3)
        self.ax.add_artist(circle)

        # Primary solution (solid line)
        self.arm_line, = self.ax.plot([], [], 'r-o', linewidth=4, markersize=8, label='Solution 1')
        self.target_marker, = self.ax.plot([], [], 'g*', markersize=15)
        
        # Secondary solution (dotted line)
        self.arm_line2, = self.ax.plot([], [], 'b--o', linewidth=3, markersize=6, alpha=0.7, label='Solution 2')
        self.target_marker2, = self.ax.plot([], [], 'c*', markersize=12, alpha=0.7)

        self.ax.set_title("2-Link Robot Arm Simulator")

    # ================= RIGHT PANEL =================
    def create_right_panel(self):
        # --- Buttons ---
        self.btn1_ax = plt.axes([0.72, 0.85, 0.25, 0.07])
        self.btn2_ax = plt.axes([0.72, 0.76, 0.25, 0.07])
        self.btn3_ax = plt.axes([0.72, 0.67, 0.25, 0.07])
        self.btn_back_ax = plt.axes([0.72, 0.55, 0.25, 0.07])  # Better spacing

        self.btn1 = Button(self.btn1_ax, "")
        self.btn2 = Button(self.btn2_ax, "")
        self.btn3 = Button(self.btn3_ax, "")
        self.btn_back = Button(self.btn_back_ax, "Back")

        self.btn1.on_clicked(self.button1_action)
        self.btn2.on_clicked(self.button2_action)
        self.btn3.on_clicked(self.button3_action)
        self.btn_back.on_clicked(lambda e: self.enter_mode("MENU"))

        # --- Info panel (moved down to avoid overlap) ---
        self.info_ax = self.fig.add_axes([0.72, 0.28, 0.25, 0.22])
        self.info_ax.axis('off')
        self.info_text = self.info_ax.text(
            0.05, 0.95, "MAIN MENU",
            verticalalignment='top',
            fontsize=12,
            fontfamily='monospace',
            bbox=dict(facecolor='#F0F0F0',
                      edgecolor='black',
                      boxstyle='round,pad=1')
        )

        # --- FK sliders ---
        self.slider1_ax = plt.axes([0.67, 0.18, 0.25, 0.04])
        self.slider2_ax = plt.axes([0.67, 0.12, 0.25, 0.04])

        self.slider_t1 = Slider(self.slider1_ax, 'Theta1 (°)', -180, 180, valinit=0)
        self.slider_t2 = Slider(self.slider2_ax, 'Theta2 (°)', -180, 180, valinit=0)

        self.slider_t1.on_changed(self.update_fk)
        self.slider_t2.on_changed(self.update_fk)

        # Start in MENU mode
        self.enter_mode("MENU")

    # ================= BUTTON ACTIONS =================
    def button1_action(self, event):
        if self.mode == "MENU":
            self.enter_mode("IK")

    def button2_action(self, event):
        if self.mode == "MENU":
            self.enter_mode("FK")

    def button3_action(self, event):
        if self.mode == "MENU":
            self.enter_mode("RRT")

    # ================= MODE MANAGEMENT =================
    def enter_mode(self, mode):
        # --- Clear previous state ---
        self.clear_rrt()
        self.clear_arm()
        self.start_point = None
        self.goal_point = None

        self.mode = mode

        # Hide sliders and buttons first
        self.slider1_ax.set_visible(False)
        self.slider2_ax.set_visible(False)

        # MENU mode
        if mode == "MENU":
            self.info_text.set_text("MAIN MENU")
            self.btn1_ax.set_visible(True)
            self.btn2_ax.set_visible(True)
            self.btn3_ax.set_visible(True)
            self.btn1.label.set_text("IK Mode")
            self.btn2.label.set_text("FK Mode")
            self.btn3.label.set_text("RRT Mode")
            self.btn_back_ax.set_visible(False)

        else:
            # Hide menu buttons
            self.btn1_ax.set_visible(False)
            self.btn2_ax.set_visible(False)
            self.btn3_ax.set_visible(False)
            self.btn_back_ax.set_visible(True)

            if mode == "IK":
                self.info_text.set_text("IK MODE\nClick workspace\n(Red=elbow down\nBlue=elbow up)")
            elif mode == "FK":
                self.info_text.set_text("FK MODE\nUse sliders")
                self.slider1_ax.set_visible(True)
                self.slider2_ax.set_visible(True)
                self.update_fk(None)
            elif mode == "RRT":
                self.info_text.set_text("RRT MODE\nClick START then GOAL")

        self.fig.canvas.draw()

    # ================= IK =================
    def solve_ik(self, x, y):
        """
        Solves inverse kinematics for 2-link arm.
        Returns both elbow-down and elbow-up solutions if they exist.
        Returns: (sol1, sol2) where each is (t1, t2) or None
        """
        d_sq = x**2 + y**2
        d = np.sqrt(d_sq)
        
        # Check if point is reachable
        if d > (self.l1 + self.l2) or d < abs(self.l1 - self.l2):
            return None, None
        
        # Calculate theta2 for both solutions
        cos_t2 = (d_sq - self.l1**2 - self.l2**2) / (2 * self.l1 * self.l2)
        cos_t2 = np.clip(cos_t2, -1, 1)
        
        # Two solutions for theta2: positive and negative
        t2_1 = np.arccos(cos_t2)   # Elbow down
        t2_2 = -np.arccos(cos_t2)  # Elbow up
        
        # Calculate corresponding theta1 values
        t1_1 = np.atan2(y, x) - np.atan2(self.l2 * np.sin(t2_1),
                                         self.l1 + self.l2 * np.cos(t2_1))
        t1_2 = np.atan2(y, x) - np.atan2(self.l2 * np.sin(t2_2),
                                         self.l1 + self.l2 * np.cos(t2_2))
        
        sol1 = (t1_1, t2_1)
        sol2 = (t1_2, t2_2)
        
        # Only return different solutions (avoid duplicates when point is on straight line)
        if abs(t2_1 - t2_2) < 0.01:  # Nearly the same solution
            return sol1, None
        
        return sol1, sol2

    # ================= FK =================
    def update_fk(self, val):
        if self.mode != "FK":
            return
        t1 = np.radians(self.slider_t1.val)
        t2 = np.radians(self.slider_t2.val)
        self.draw_arm(t1, t2)

    # ================= RRT =================
    def sample_random(self):
        """Sample random point in workspace"""
        limit = self.l1 + self.l2
        return np.random.uniform(-limit, limit, 2)
    
    def nearest(self, tree, q_rand):
        """Find nearest node in tree to random sample"""
        return min(tree, key=lambda node: np.linalg.norm(node - q_rand))
    
    def steer(self, q_near, q_rand, step=0.3):
        """Steer from q_near toward q_rand by step distance"""
        direction = q_rand - q_near
        dist = np.linalg.norm(direction)
        if dist < step:
            return q_rand
        return q_near + (direction / dist) * step
    
    def is_goal_reached(self, point, goal, tol=0.3):
        """Check if point is close enough to goal"""
        return np.linalg.norm(point - goal) < tol
    
    def run_rrt(self, start, goal, max_iter=5000):
        """
        Simplified RRT algorithm based on working 3D implementation.
        Tree grows in workspace (x,y) coordinates.
        """
        print(f"\n=== RRT START ===")
        print(f"Start: {start}")
        print(f"Goal: {goal}")
        print(f"Max iterations: {max_iter}")
        
        tree = [start]
        parent = {tuple(start): None}
        
        # Draw the start node of the tree
        try:
            start_node = self.ax.scatter([start[0]], [start[1]], c='green', s=100, 
                                         zorder=5, edgecolors='darkgreen', linewidths=2)
            self.rrt_nodes.append(start_node)
            print("Start node drawn")
        except Exception as e:
            print(f"Error drawing start node: {e}")
        
        for i in range(max_iter):
            # Update display periodically
            if i % 100 == 0:
                print(f"Iteration {i}, Nodes: {len(tree)}")
                try:
                    plt.pause(0.001)
                    self.info_text.set_text(f"RRT MODE\nIter: {i}/{max_iter}\nNodes: {len(tree)}")
                    self.fig.canvas.draw_idle()
                except Exception as e:
                    print(f"Error updating display: {e}")
            
            # 10% of the time, bias toward goal
            if np.random.rand() < 0.1:
                q_rand = goal
            else:
                q_rand = self.sample_random()
            
            # Find nearest node and steer toward random sample
            try:
                q_near = self.nearest(tree, q_rand)
                q_new = self.steer(q_near, q_rand)
            except Exception as e:
                print(f"Error in nearest/steer at iteration {i}: {e}")
                continue
            
            # Skip if already in tree
            if tuple(q_new) in parent:
                if i % 100 == 0:
                    print(f"Skipping duplicate at iteration {i}")
                continue
            
            # Check if reachable by IK (within workspace)
            limit = self.l1 + self.l2
            if np.linalg.norm(q_new) > limit:
                if i % 100 == 0:
                    print(f"Out of bounds at iteration {i}")
                continue
            
            # Add to tree
            tree.append(q_new)
            parent[tuple(q_new)] = tuple(q_near)
            
            # Draw edge
            try:
                line, = self.ax.plot([q_near[0], q_new[0]], [q_near[1], q_new[1]],
                                     'lightblue', linewidth=1, alpha=0.5)
                self.rrt_lines.append(line)
            except Exception as e:
                print(f"Error drawing edge at iteration {i}: {e}")
            
            # Check if goal reached - STOP IMMEDIATELY
            if self.is_goal_reached(q_new, goal):
                print(f"GOAL REACHED at iteration {i}!")
                parent[tuple(goal)] = tuple(q_new)
                self.info_text.set_text(f"RRT MODE\nGoal reached!\nNodes: {len(tree)}")
                self.fig.canvas.draw_idle()
                
                # Draw all tree nodes at once at the end
                try:
                    tree_array = np.array(tree)
                    nodes = self.ax.scatter(tree_array[:, 0], tree_array[:, 1], 
                                           c='cyan', s=20, zorder=3, alpha=0.6)
                    self.rrt_nodes.append(nodes)
                    print("Tree nodes drawn successfully")
                except Exception as e:
                    print(f"Error drawing tree nodes: {e}")
                
                print("=== RRT SUCCESS ===\n")
                return parent, goal
        
        # Failed - draw tree nodes
        print(f"RRT FAILED after {max_iter} iterations")
        try:
            tree_array = np.array(tree)
            nodes = self.ax.scatter(tree_array[:, 0], tree_array[:, 1], 
                                   c='cyan', s=20, zorder=3, alpha=0.6)
            self.rrt_nodes.append(nodes)
        except Exception as e:
            print(f"Error drawing final tree nodes: {e}")
        
        self.info_text.set_text(f"RRT MODE\nFailed\nNodes: {len(tree)}")
        self.fig.canvas.draw_idle()
        print("=== RRT FAILED ===\n")
        return None, None

    def reconstruct_path(self, parent, goal):
        """Reconstruct path from start to goal using parent pointers"""
        print(f"Reconstructing path from goal: {goal}")
        path = []
        node = tuple(goal)
        visited = set()  # Prevent infinite loops
        max_iterations = 10000  # Safety limit
        
        iteration = 0
        while node is not None and iteration < max_iterations:
            if node in visited:
                print(f"ERROR: Circular reference detected at node {node}!")
                break
            
            visited.add(node)
            path.append(np.array(node))
            
            if iteration % 100 == 0:
                print(f"Path reconstruction iteration {iteration}, current node: {node}")
            
            parent_node = parent.get(node)
            if parent_node is not None:
                # Convert to tuple for consistency
                node = tuple(parent_node) if isinstance(parent_node, np.ndarray) else parent_node
            else:
                node = None
            
            iteration += 1
        
        if iteration >= max_iterations:
            print(f"ERROR: Path reconstruction exceeded max iterations!")
        
        path.reverse()
        print(f"Path reconstruction complete: {len(path)} points")
        return path

    # ================= DRAW =================
    def draw_arm(self, t1, t2, secondary=False):
        """Draw arm configuration. If secondary=True, use dotted line."""
        x1 = self.l1 * np.cos(t1)
        y1 = self.l1 * np.sin(t1)
        x2 = x1 + self.l2 * np.cos(t1 + t2)
        y2 = y1 + self.l2 * np.sin(t1 + t2)
        
        if secondary:
            self.arm_line2.set_data([0, x1, x2], [0, y1, y2])
            self.target_marker2.set_data([x2], [y2])
        else:
            self.arm_line.set_data([0, x1, x2], [0, y1, y2])
            self.target_marker.set_data([x2], [y2])
        
        self.fig.canvas.draw_idle()

    def clear_arm(self):
        self.arm_line.set_data([], [])
        self.target_marker.set_data([], [])
        self.arm_line2.set_data([], [])
        self.target_marker2.set_data([], [])

    def clear_rrt(self):
        # Remove tree edges
        for line in self.rrt_lines:
            line.remove()
        self.rrt_lines.clear()
        
        # Remove tree nodes
        for node in self.rrt_nodes:
            node.remove()
        self.rrt_nodes.clear()
        
        # Remove start/goal markers
        for marker in self.rrt_markers:
            marker.remove()
        self.rrt_markers.clear()

    # ================= CLICK =================
    def on_click(self, event):
        if event.inaxes != self.ax:
            return
        
        if self.mode == "IK":
            sol1, sol2 = self.solve_ik(event.xdata, event.ydata)
            if sol1:
                self.draw_arm(*sol1, secondary=False)
                
                # Display angles for solution 1
                t1_deg = np.degrees(sol1[0])
                t2_deg = np.degrees(sol1[1])
                info_text = f"IK MODE\nClick workspace\n\nElbow Down (Red):\nθ1: {t1_deg:.1f}°\nθ2: {t2_deg:.1f}°"
                
                if sol2:
                    self.draw_arm(*sol2, secondary=True)
                    # Add angles for solution 2
                    t1_deg2 = np.degrees(sol2[0])
                    t2_deg2 = np.degrees(sol2[1])
                    info_text += f"\n\nElbow Up (Blue):\nθ1: {t1_deg2:.1f}°\nθ2: {t2_deg2:.1f}°"
                else:
                    # Clear secondary arm if no second solution
                    self.arm_line2.set_data([], [])
                    self.target_marker2.set_data([], [])
                
                self.info_text.set_text(info_text)
                self.fig.canvas.draw_idle()
            else:
                self.info_text.set_text("IK MODE\nPoint unreachable!")
                self.fig.canvas.draw_idle()
                
        elif self.mode == "RRT":
            if self.start_point is None:
                self.start_point = np.array([event.xdata, event.ydata])
                # Draw start marker
                marker = self.ax.scatter([event.xdata], [event.ydata], c='green', 
                                        s=150, marker='o', zorder=10, 
                                        edgecolors='darkgreen', linewidths=2,
                                        label='Start')
                self.rrt_markers.append(marker)
                self.info_text.set_text("RRT MODE\nStart set\nClick GOAL")
                self.fig.canvas.draw_idle()
                print(f"Start point set: {self.start_point}")
            else:
                self.goal_point = np.array([event.xdata, event.ydata])
                
                # Check if start and goal are too close
                if np.linalg.norm(self.goal_point - self.start_point) < 0.1:
                    print("ERROR: Goal too close to start!")
                    self.info_text.set_text("RRT MODE\nGoal too close!\nClick elsewhere")
                    self.fig.canvas.draw_idle()
                    self.goal_point = None
                    return
                
                # Draw goal marker
                marker = self.ax.scatter([event.xdata], [event.ydata], c='red', 
                                        s=150, marker='*', zorder=10,
                                        edgecolors='darkred', linewidths=2,
                                        label='Goal')
                self.rrt_markers.append(marker)
                self.info_text.set_text("RRT MODE\nRunning RRT...")
                self.fig.canvas.draw_idle()
                print(f"Goal point set: {self.goal_point}")
                
                # Run RRT - tree grows from start toward goal
                print("Calling run_rrt...")
                parent, goal_node = self.run_rrt(self.start_point, self.goal_point)
                print(f"run_rrt returned: parent={'dict' if parent else 'None'}, goal_node={goal_node}")
                
                if parent is not None:
                    print("Reconstructing path...")
                    path = self.reconstruct_path(parent, goal_node)
                    print(f"Path length: {len(path)}")
                    
                    # Highlight the solution path
                    if len(path) > 1:
                        print("Drawing solution path...")
                        path_array = np.array(path)
                        line, = self.ax.plot(path_array[:, 0], path_array[:, 1],
                                            'yellow', linewidth=3, zorder=4,
                                            label='Solution Path', alpha=0.9)
                        self.rrt_lines.append(line)
                        
                        # Draw path nodes
                        nodes = self.ax.scatter(path_array[:, 0], path_array[:, 1],
                                              c='orange', s=50, zorder=5,
                                              edgecolors='darkorange', linewidths=1.5)
                        self.rrt_nodes.append(nodes)
                        print("Solution path drawn")
                    
                    # Animate the arm following the path
                    print("Starting animation...")
                    for idx, p in enumerate(path):
                        if idx % 5 == 0:
                            print(f"Animating point {idx}/{len(path)}")
                        sol1, sol2 = self.solve_ik(p[0], p[1])
                        if sol1:
                            self.draw_arm(*sol1)
                            plt.pause(0.05)
                    print("Animation complete")
                    
                    self.info_text.set_text(f"RRT MODE\nPath: {len(path)} points\nClick to try again")
                else:
                    print("No path found")
                    self.info_text.set_text("RRT MODE\nNo path found\nClick to try again")
                
                self.fig.canvas.draw_idle()
                self.start_point = None
                self.goal_point = None
                print("RRT cycle complete\n")


if __name__ == "__main__":
    RobotArmApp()