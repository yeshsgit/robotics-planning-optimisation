import matplotlib

matplotlib.use('TkAgg')
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Button
import time, random, heapq


# ==========================================
# 1. 地图与碰撞引擎
# ==========================================
class MapEngine:
    def __init__(self, size=100):
        self.size = size
        self.grid = np.zeros((size, size))

    def set_map(self, name):
        self.grid = np.zeros((self.size, self.size))
        if name == "Forest":
            for _ in range(18):
                r, c = random.randint(10, 80), random.randint(10, 80)
                self.grid[r:r + 12, c:c + 12] = 1
        elif name == "Deep Trap":
            self.grid[20:80, 70:75] = 1  # 坑底
            self.grid[20:25, 30:75] = 1  # 上壁
            self.grid[75:80, 30:75] = 1  # 下壁
        elif name == "The Needle":
            self.grid[0:49, 45:55] = 1  # 下方屏障
            self.grid[51:100, 45:55] = 1  # 上方屏障，中间留2像素(0.2单位)缝隙
        return self.grid


def is_colliding(grid, x, y):
    r, c = int(y * 10), int(x * 10)
    if 0 <= r < 100 and 0 <= c < 100:
        return grid[r, c] == 1
    return True


def path_ok(grid, p1, p2):
    for t in np.linspace(0, 1, 15):
        if is_colliding(grid, p1[0] * (1 - t) + p2[0] * t, p1[1] * (1 - t) + p2[1] * t): return False
    return True


# ==========================================
# 2. 算法实现 (A*, RRT, RRT*)
# ==========================================
class Planners:
    @staticmethod
    def a_star(start, goal, grid):
        res = 0.25  # 分辨率临界点
        open_list = [(0, tuple(start))]
        came_from, g_score = {}, {tuple(start): 0}
        while open_list:
            _, curr = heapq.heappop(open_list)
            if np.linalg.norm(np.array(curr) - goal) < 0.5:
                path = []
                while curr in came_from: path.append(curr); curr = came_from[curr]
                return path[::-1], True
            for d in [(-res, 0), (res, 0), (0, -res), (0, res)]:
                nb = (round(curr[0] + d[0], 1), round(curr[1] + d[1], 1))
                if 0 <= nb[0] <= 10 and 0 <= nb[1] <= 10 and not is_colliding(grid, *nb):
                    tg = g_score[curr] + res
                    if nb not in g_score or tg < g_score[nb]:
                        came_from[nb] = curr;
                        g_score[nb] = tg
                        f = tg + np.linalg.norm(np.array(nb) - goal)
                        heapq.heappush(open_list, (f, nb))
        return [], False

    @staticmethod
    def rrt_variants(start, goal, grid, is_star=False):
        nodes = [np.array(start)];
        parents = {0: None};
        costs = {0: 0}
        max_iter = 600  # 较低的迭代上限以产生失败(Succ Rate < 100%)
        step = 0.5
        for _ in range(max_iter):
            q_rand = goal if random.random() < 0.1 else np.array([random.uniform(0, 10), random.uniform(0, 10)])
            dists = [np.linalg.norm(n - q_rand) for n in nodes]
            idx_near = np.argmin(dists)
            q_near = nodes[idx_near]
            q_new = q_near + (q_rand - q_near) / (dists[idx_near] + 1e-6) * step

            if path_ok(grid, q_near, q_new):
                new_idx = len(nodes);
                nodes.append(q_new)
                if not is_star:
                    parents[new_idx] = idx_near
                else:
                    best_idx, min_c = idx_near, costs[idx_near] + step
                    for i, n in enumerate(nodes[:-1]):
                        if np.linalg.norm(n - q_new) < 1.2 and path_ok(grid, n, q_new):
                            if costs[i] + np.linalg.norm(n - q_new) < min_c:
                                min_c = costs[i] + np.linalg.norm(n - q_new);
                                best_idx = i
                    parents[new_idx] = best_idx;
                    costs[new_idx] = min_c
                if np.linalg.norm(q_new - goal) < 0.6:
                    path = [];
                    c = new_idx
                    while c is not None: path.append(nodes[c]); c = parents[c]
                    return path[::-1], True
        return [], False


# ==========================================
# 3. 完整 UI 演示界面
# ==========================================
class PBLComparisonApp:
    def __init__(self):
        self.fig, self.ax = plt.subplots(figsize=(12, 7))
        plt.subplots_adjust(left=0.05, bottom=0.2, right=0.72)

        self.engine = MapEngine()
        self.maps = ["Forest", "Deep Trap", "The Needle"]
        self.current_map = "Forest"
        self.grid = self.engine.set_map(self.current_map)
        self.start, self.goal = np.array([1, 1]), np.array([9, 9])

        self.info_box = self.fig.text(0.75, 0.5, "Press '10x TEST'\nto start benchmarking",
                                      fontsize=11, verticalalignment='center',
                                      bbox=dict(facecolor='white', alpha=0.8, boxstyle='round,pad=1'))
        self.create_widgets()
        self.refresh_ui()
        plt.show(block=True)

    def create_widgets(self):
        self.m_btns = []
        for i, name in enumerate(self.maps):
            ax_m = plt.axes([0.05 + i * 0.11, 0.05, 0.1, 0.06])
            btn = Button(ax_m, name);
            btn.on_clicked(lambda e, n=name: self.change_map(n))
            self.m_btns.append(btn)

        ax_run = plt.axes([0.45, 0.05, 0.15, 0.06])
        self.btn_run = Button(ax_run, '10x TEST', color='orange')
        self.btn_run.on_clicked(self.run_benchmark)

    def change_map(self, name):
        self.current_map = name
        self.grid = self.engine.set_map(name)
        self.refresh_ui()

    def refresh_ui(self):
        self.ax.clear()
        # 黑色=墙(1), 白色=路(0)
        self.ax.imshow(self.grid, cmap='gray_r', origin='lower', extent=[0, 10, 0, 10], zorder=1)
        self.ax.scatter(*self.start, c='lime', s=100, edgecolors='black', label='START', zorder=5)
        self.ax.scatter(*self.goal, c='red', marker='*', s=200, edgecolors='black', label='GOAL', zorder=5)
        self.ax.set_title(f"Benchmarking: {self.current_map} (n=10)")
        self.ax.legend(loc='upper left')
        plt.draw()

    def run_benchmark(self, event):
        self.refresh_ui()
        num_trials = 10
        results_text = f"RESULTS: {self.current_map}\n" + "-" * 20 + "\n"

        planners = [
            ("A*", 'blue', lambda: Planners.a_star(self.start, self.goal, self.grid)),
            ("RRT", 'green', lambda: Planners.rrt_variants(self.start, self.goal, self.grid, False)),
            ("RRT*", 'red', lambda: Planners.rrt_variants(self.start, self.goal, self.grid, True))
        ]

        for name, color, func in planners:
            self.ax.set_title(f"Calculating {name}...");
            plt.draw();
            plt.pause(0.01)
            success_count, total_time, total_dist = 0, 0, 0
            last_path = None

            for _ in range(num_trials):
                t0 = time.perf_counter()
                path, ok = func()
                dt = (time.perf_counter() - t0) * 1000
                if ok:
                    success_count += 1
                    total_time += dt
                    total_dist += sum(np.linalg.norm(np.array(path[i]) - path[i + 1]) for i in range(len(path) - 1))
                    last_path = path

            sr = (success_count / num_trials) * 100
            avg_t = total_time / success_count if success_count > 0 else 0
            avg_d = total_dist / success_count if success_count > 0 else 0

            if last_path:
                p = np.array(last_path)
                self.ax.plot(p[:, 0], p[:, 1], color=color, lw=2.5, zorder=4, label=f"{name} (Succ: {sr:.0f}%)")

            results_text += f"● {name}\n Succ: {sr:.0f}%\n Time: {avg_t:.1f}ms\n Dist: {avg_d:.2f}m\n\n"

        self.info_box.set_text(results_text)
        self.ax.set_title(f"Test Complete: {self.current_map}")
        self.ax.legend(loc='upper left', fontsize='small')
        plt.draw()


if __name__ == "__main__":
    PBLComparisonApp()