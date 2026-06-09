import matplotlib.pyplot as plt
import numpy as np
import time

class DebugVisualizer:
    def __init__(self, world_size=20.0, resolution=6):
        self.world_size_m = world_size
        self.resolution = resolution
        self.half_world = self.world_size_m / 2.0

        self.last_draw_time = 0.0
        self.min_interval = 0.04  # 限制最高 10 帧/秒，防止卡顿

        plt.ion()
        self.fig, self.ax = plt.subplots(figsize=(8, 8))
        plt.show(block=False)

        # 1:1 对齐参考文件：创建空白概率网格（初始化为0.5未知区域）
        initial_prob_grid = np.ones((int(self.world_size_m * self.resolution), 
                                     int(self.world_size_m * self.resolution))) * 0.5
                                     
        # 【核心修正】：严格使用参考代码中的 extent 参数和默认绘图方向
        self.im = self.ax.imshow(initial_prob_grid, cmap='gray_r',
                                 vmin=0, vmax=1,
                                 extent=[-self.half_world, self.half_world, -self.half_world, self.half_world])
        
        self.ax.set_title("Global Lidar Map (20x20m)")
        self.ax.set_xlabel("X (meters)")
        self.ax.set_ylabel("Y (meters)")

        # 1:1 对齐参考代码的 Patch 定义
        self.robot_patch = self.ax.add_patch(plt.Circle((0, 0), 0.2, color='blue', zorder=5))
        self.robot_arrow = None  # 动态生成箭头，防闪退
        self.goal_patch, = self.ax.plot([], [], '*', color='red', markersize=15, zorder=4)
        self.path_patch, = self.ax.plot([], [], '-', color='cyan', linewidth=2, zorder=3)
        
        self.landmark_patches = {}
        self.frontier_scatter = self.ax.scatter([], [], c='green', s=10, zorder=2, alpha=0.7)

        plt.pause(0.1)

    def update(self,
               global_grid_map=None,
               robot_pos=None,
               robot_yaw=None,
               goal_pos=None,
               path=None,
               landmark_positions=None,
               frontier_points=None,
               title_extra='',
               force_update=False):

        # 跳帧限制
        current_time = time.time()
        if not force_update and (current_time - self.last_draw_time < self.min_interval):
            return  
        self.last_draw_time = current_time

        if not plt.fignum_exists(self.fig.number):
            return

        # 1. 严格使用环境类自带的 _log_odds_to_prob 来转概率，没有任何花哨的操作
        if global_grid_map is not None and hasattr(global_grid_map, 'grid'):
            if hasattr(global_grid_map, '_log_odds_to_prob'):
                prob_map = global_grid_map._log_odds_to_prob(global_grid_map.grid)
            else:
                prob_map = 1.0 - 1.0 / (1.0 + np.exp(global_grid_map.grid))
            self.im.set_data(prob_map)

        # 2. 更新机器人（严格对齐环境的 X, Y）
        if robot_pos is not None:
            self.robot_patch.center = (robot_pos[0], robot_pos[1])
            if robot_yaw is not None:
                if self.robot_arrow is not None:
                    self.robot_arrow.remove()
                arrow_dx = 0.5 * np.cos(robot_yaw)
                arrow_dy = 0.5 * np.sin(robot_yaw)
                self.robot_arrow = plt.Arrow(robot_pos[0], robot_pos[1], arrow_dx, arrow_dy, width=0.2, color='blue', zorder=5)
                self.ax.add_patch(self.robot_arrow)

        # 3. 目标点与路径
        if goal_pos is not None:
            self.goal_patch.set_data([goal_pos[0]], [goal_pos[1]])

        if path is not None and len(path) > 0:
            self.path_patch.set_data(path[:, 0], path[:, 1])
        else:
            self.path_patch.set_data([], [])

        # 4. 其他图层更新
        if landmark_positions:
            for lm_id, pos in landmark_positions.items():
                if lm_id not in self.landmark_patches:
                    color = lm_id.split('_')[1] if '_' in lm_id else 'gray' 
                    self.landmark_patches[lm_id] = self.ax.add_patch(
                        plt.Circle((pos[0], pos[1]), 0.4, color=color, alpha=0.6, zorder=4)
                    )
                    self.ax.text(pos[0], pos[1] + 0.5, lm_id.replace("landmark_", ""), ha='center', va='bottom', fontsize=9)
                else:
                    self.landmark_patches[lm_id].center = (pos[0], pos[1])

        if frontier_points is not None and len(frontier_points) > 0:
            self.frontier_scatter.set_offsets(frontier_points)
        else:
            self.frontier_scatter.set_offsets(np.empty((0, 2)))

        self.ax.set_title(f"Global Lidar Map ({title_extra})")
        self.fig.canvas.draw_idle()
        self.fig.canvas.flush_events()
        plt.pause(0.001)

    def close(self):
        if self.fig is not None and plt.fignum_exists(self.fig.number):
            plt.close(self.fig)