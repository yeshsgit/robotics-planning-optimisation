import matplotlib

matplotlib.use('TkAgg')

import numpy as np
import matplotlib.pyplot as plt


class RobotArmIK:
    def __init__(self, l1=2.0, l2=1.5):
        self.l1 = l1
        self.l2 = l2

        self.fig, self.ax = plt.subplots(figsize=(8, 5))

        plt.subplots_adjust(left=0.05, right=0.7)

        self.setup_ui()
        self.fig.canvas.mpl_connect('button_press_event', self.on_click)
        plt.show(block=True)

    def setup_ui(self):
        limit = (self.l1 + self.l2) * 1.2
        self.ax.set_xlim(-limit, limit)
        self.ax.set_ylim(-limit, limit)
        self.ax.set_aspect('equal')
        self.ax.grid(True, linestyle='--', alpha=0.5)

        workspace_circle = plt.Circle((0, 0), self.l1 + self.l2, color='gray', fill=False, linestyle='--', alpha=0.3)
        self.ax.add_artist(workspace_circle)

        self.line_up, = self.ax.plot([], [], 'r-o', linewidth=4, markersize=8, label='Sol 1: Elbow Up')
        self.line_down, = self.ax.plot([], [], 'b--o', linewidth=2, markersize=6, alpha=0.6, label='Sol 2: Elbow Down')
        self.target_marker, = self.ax.plot([], [], 'g*', markersize=15, label='Target')

        self.angle_text = self.ax.text(1.05, 0.95, "WAITING FOR CLICK...",
                                       transform=self.ax.transAxes,
                                       verticalalignment='top',
                                       fontsize=11,
                                       fontfamily='monospace',
                                       bbox=dict(facecolor='#F0F0F0', alpha=1.0, edgecolor='black',
                                                 boxstyle='round,pad=1'))

        self.ax.set_title("2-Link Robot Arm IK/FK (Control Panel)", pad=20)
        self.ax.legend(loc='lower left', fontsize='small')

    def solve_ik(self, x, y):
        d_sq = x ** 2 + y ** 2
        d = np.sqrt(d_sq)
        if d > (self.l1 + self.l2) or d < abs(self.l1 - self.l2):
            self.angle_text.set_text("STATUS:\nOUT OF REACH")
            return None
        cos_t2 = (d_sq - self.l1 ** 2 - self.l2 ** 2) / (2 * self.l1 * self.l2)
        cos_t2 = np.clip(cos_t2, -1.0, 1.0)
        t2_sol1 = np.arccos(cos_t2)
        t2_sol2 = -t2_sol1

        def get_t1(t2):
            return np.atan2(y, x) - np.atan2(self.l2 * np.sin(t2), self.l1 + self.l2 * np.cos(t2))

        return [(get_t1(t2_sol1), t2_sol1), (get_t1(t2_sol2), t2_sol2)]

    def draw_arm(self, t1, t2, plot_handle):
        x1, y1 = self.l1 * np.cos(t1), self.l1 * np.sin(t1)
        x2, y2 = x1 + self.l2 * np.cos(t1 + t2), y1 + self.l2 * np.sin(t1 + t2)
        plot_handle.set_data([0, x1, x2], [0, y1, y2])

    def on_click(self, event):
        if event.inaxes != self.ax: return

        solutions = self.solve_ik(event.xdata, event.ydata)
        if solutions:
            self.draw_arm(solutions[0][0], solutions[0][1], self.line_up)
            self.draw_arm(solutions[1][0], solutions[1][1], self.line_down)
            self.target_marker.set_data([event.xdata], [event.ydata])

            s1_t1, s1_t2 = np.degrees(solutions[0][0]), np.degrees(solutions[0][1])
            s2_t1, s2_t2 = np.degrees(solutions[1][0]), np.degrees(solutions[1][1])

            info = (f"  DATA MONITOR\n"
                    f"====================\n"
                    f"Target X: {event.xdata:>6.2f}\n"
                    f"Target Y: {event.ydata:>6.2f}\n"
                    f"====================\n"
                    f"SOL 1 (Elbow Up):\n"
                    f"  Th 1: {s1_t1:>7.2f}°\n"
                    f"  Th 2: {s1_t2:>7.2f}°\n\n"
                    f"SOL 2 (Elbow Down):\n"
                    f"  Th 1: {s2_t1:>7.2f}°\n"
                    f"  Th 2: {s2_t2:>7.2f}°\n"
                    f"====================")
            self.angle_text.set_text(info)
            self.fig.canvas.draw_idle()


if __name__ == "__main__":
    RobotArmIK(l1=2.0, l2=1.5)