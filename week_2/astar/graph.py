import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

# Import your existing functions
from planner import astar, fk, ik, space1, space2, l1, l2
import planner

from collision_api import RECTS
import matplotlib.patches as patches

start_xy = (15, 0)

def compute_collision_map():

    collision_map = np.zeros((planner.angle1Division,
                              planner.angle2Division))

    for i in range(planner.angle1Division):
        for j in range(planner.angle2Division):

            theta1 = planner.space1[i]
            theta2 = planner.space2[j]

            if planner.collision(theta1, l1, theta2, l2):
                collision_map[i, j] = 1

    return collision_map

# ----------------------------
# Figure setup
# ----------------------------
fig = plt.figure(figsize=(12, 6))

ax_workspace = fig.add_subplot(121)
ax_jointspace = fig.add_subplot(122)

ax_workspace.set_title("Workspace (x, y)")
ax_workspace.set_xlim(-l1-l2-2, l1+l2+2)
ax_workspace.set_ylim(-l1-l2-2, l1+l2+2)
ax_workspace.set_aspect('equal')

# ----------------------------
# Draw obstacle polygons
# ----------------------------
for rect in RECTS:

    polygon = patches.Polygon(
        rect,
        closed=True,
        edgecolor='black',
        facecolor='gray',
        alpha=0.6
    )

    ax_workspace.add_patch(polygon)

ax_jointspace.set_title("Joint Space (θ1, θ2)")
ax_jointspace.set_xlim(0, 2*np.pi)
ax_jointspace.set_ylim(0, 2*np.pi)

# Force exact limits
ax_jointspace.set_xticks([0, np.pi, 2*np.pi])
ax_jointspace.set_yticks([0, np.pi, 2*np.pi])
ax_jointspace.set_xticklabels(["0", "π", "2π"])
ax_jointspace.set_yticklabels(["0", "π", "2π"])

robot_line, = ax_workspace.plot([], [], 'o-', lw=4)
ee_point, = ax_workspace.plot([], [], 'ro')

collision_map = compute_collision_map()

from matplotlib.colors import ListedColormap

ax_jointspace.imshow(
    collision_map.T,
    origin='lower',
    extent=[0, 2*np.pi, 0, 2*np.pi],
    alpha=0.4,
    cmap = ListedColormap(["white", "red"])
)

start_point_joint, = ax_jointspace.plot([], [], 'go', markersize=8, label="Start")
goal_point_joint, = ax_jointspace.plot([], [], 'mo', markersize=8, label="Goal")

joint_path_line, = ax_jointspace.plot([], [], 'b-')
joint_current_point, = ax_jointspace.plot([], [], 'ro')

angle_text = ax_workspace.text(
    0.02, 0.95, "",
    transform=ax_workspace.transAxes,
    verticalalignment='top',
    fontsize=12,
    bbox=dict(boxstyle="round", facecolor="white")
)

current_path = []
animation = None


# ----------------------------
# Break wrap-around jumps
# ----------------------------
def remove_wrap_jumps(theta1_list, theta2_list):

    clean_t1 = [theta1_list[0]]
    clean_t2 = [theta2_list[0]]

    for i in range(1, len(theta1_list)):
        d1 = abs(theta1_list[i] - theta1_list[i-1])
        d2 = abs(theta2_list[i] - theta2_list[i-1])

        # If jump larger than π → wrap happened
        if d1 > np.pi or d2 > np.pi:
            clean_t1.append(np.nan)
            clean_t2.append(np.nan)

        clean_t1.append(theta1_list[i])
        clean_t2.append(theta2_list[i])

    return clean_t1, clean_t2


# ----------------------------
# Draw robot
# ----------------------------
def draw_robot(theta1, theta2):

    x0, y0 = 0, 0

    x1 = l1 * np.cos(theta1)
    y1 = l1 * np.sin(theta1)

    x2 = x1 + l2 * np.cos(theta1 + theta2)
    y2 = y1 + l2 * np.sin(theta1 + theta2)

    robot_line.set_data([x0, x1, x2], [y0, y1, y2])
    ee_point.set_data([x2], [y2])

    angle_text.set_text(
        f"θ1 = {theta1:.4f} rad\nθ2 = {theta2:.4f} rad"
    )


# ----------------------------
# Animation
# ----------------------------
def animate(i):
    theta1, theta2 = current_path[i]

    draw_robot(theta1, theta2)
    joint_current_point.set_data([theta1], [theta2])

    return robot_line, ee_point, joint_current_point, angle_text


# ----------------------------
# Click handler
# ----------------------------
def onclick(event):
    global current_path, animation

    if event.inaxes != ax_workspace:
        return

    goal_xy = (event.xdata, event.ydata)
    print("Goal selected:", goal_xy)

    # --- Get start joint config ---
    start_angles, start_angles2 = ik(start_xy)
    theta1_s = start_angles[0] % (2*np.pi)
    theta2_s = start_angles[1] % (2*np.pi)

    # --- Get goal joint config ---
    goal_angles, goal_angles2 = ik(goal_xy)
    theta1_g = goal_angles[0] % (2*np.pi)
    theta2_g = goal_angles[1] % (2*np.pi)

    # 🔵 Plot start & goal immediately (always)
    start_point_joint.set_data([theta1_s], [theta2_s])
    goal_point_joint.set_data([theta1_g], [theta2_g])

    plt.draw()

    # --- Run A* ---
    path = astar(start_xy, goal_xy, final_angles=goal_angles)

    # Clear previous path drawing
    joint_path_line.set_data([], [])
    joint_current_point.set_data([], [])

    if animation is not None:
        try:
            animation.event_source.stop()
        except:
            pass

    if path is None:
        path = astar(start_xy, goal_xy, final_angles=goal_angles2)

        if path is None:
            print("No path found.")
            return

    current_path = path

    theta1_list = [p[0] for p in path]
    theta2_list = [p[1] for p in path]

    theta1_plot, theta2_plot = remove_wrap_jumps(theta1_list, theta2_list)
    joint_path_line.set_data(theta1_plot, theta2_plot)

    animation = FuncAnimation(
        fig,
        animate,
        frames=len(current_path),
        interval=100,
        repeat=False
    )

    plt.draw()

# ----------------------------
# Initialize
# ----------------------------
start_angles, _ = ik(start_xy)
draw_robot(start_angles[0], start_angles[1])

fig.canvas.mpl_connect('button_press_event', onclick)

plt.show()