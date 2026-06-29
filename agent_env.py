import numpy as np
import mujoco
import mujoco.viewer
import cv2
import sys
import math
import types
import time
import json
import textwrap
from pathlib import Path
from queue import Queue
import gymnasium as gym
from scipy.spatial.transform import Rotation as R
from skimage.draw import line as skimage_line 
from scipy.ndimage import map_coordinates, binary_opening, label, find_objects

try:
    from semantic_segmenter import SegFormerSemanticSegmenter
except ImportError:
    SegFormerSemanticSegmenter = None

# Prefer the bundled random-map training environment so dynamics and observation
# format match the policy checkpoint used by this demo.
local_visual_train = Path(__file__).with_name("visual_train")
if local_visual_train.exists():
    sys.path.insert(0, str(local_visual_train))

try:
    from robot_visual_env_random_map import RobotVisualEnv, GlobalGridMap
except ImportError:
    raise


class MazeGridGenerator:
    def __init__(self, world_size=20.0, grid_dim=8, remove_wall_prob=0.25):
        self.world_size = world_size
        self.grid_dim = grid_dim
        self.cell_size = world_size / grid_dim
        self.remove_wall_prob = remove_wall_prob
        self.wall_thickness = 0.1 

    def generate(self, np_random):
        visited = np.zeros((self.grid_dim, self.grid_dim), dtype=bool)
        stack =[(0, 0)]
        visited[0, 0] = True
        passages = set()
        
        while stack:
            r, c = stack[-1]
            neighbors =[]
            for dr, dc in[(-1, 0), (1, 0), (0, -1), (0, 1)]:
                nr, nc = r + dr, c + dc
                if 0 <= nr < self.grid_dim and 0 <= nc < self.grid_dim and not visited[nr, nc]:
                    neighbors.append((nr, nc))
            
            if neighbors:
                idx = np_random.integers(0, len(neighbors))
                nr, nc = neighbors[idx]
                p1, p2 = (r, c), (nr, nc)
                if p1 > p2: p1, p2 = p2, p1
                passages.add((p1, p2))
                visited[nr, nc] = True
                stack.append((nr, nc))
            else:
                stack.pop()
        
        walls =[]
        for r in range(self.grid_dim):
            for c in range(self.grid_dim - 1):
                p1, p2 = (r, c), (r, c+1)
                if (p1, p2) not in passages:
                    if np_random.random() > self.remove_wall_prob:
                        center_x = (c + 1) * self.cell_size - self.world_size/2.0
                        center_y = (self.world_size/2.0) - (r + 0.5) * self.cell_size 
                        walls.append({
                            'pos': np.array([center_x, center_y, 1.0]),
                            'is_vertical': True,
                            'size': np.array([self.wall_thickness, self.cell_size/2.0 + 0.05, 1.0])
                        })

        for r in range(self.grid_dim - 1):
            for c in range(self.grid_dim):
                p1, p2 = (r, c), (r+1, c)
                if (p1, p2) not in passages:
                    if np_random.random() > self.remove_wall_prob:
                        center_x = (c + 0.5) * self.cell_size - self.world_size/2.0
                        center_y = (self.world_size/2.0) - (r + 1) * self.cell_size
                        walls.append({
                            'pos': np.array([center_x, center_y, 1.0]),
                            'is_vertical': False,
                            'size': np.array([self.cell_size/2.0 + 0.05, self.wall_thickness, 1.0])
                        })
        return walls

class AgentVisualEnv(RobotVisualEnv):
    def __init__(self, *args, **kwargs):
        enable_mujoco_viewer = bool(kwargs.pop("enable_mujoco_viewer", False))
        super().__init__(*args, **kwargs)
        
        if hasattr(self, 'grid_map') and self.grid_map.world_size_m < 20.0:
            self.grid_map = GlobalGridMap(world_size_m=20.0, local_map_size_m=4.0, resolution=6)

        # Keep the policy's 180-ray lidar layout; cache ray directions to cut per-frame cost.
        self.lidar_num_rays = 180
        self.lidar_angles = np.linspace(-self.lidar_fov / 2, self.lidar_fov / 2, self.lidar_num_rays)
        self._lidar_local_rays_cache = np.column_stack([
            np.cos(self.lidar_angles),
            np.sin(self.lidar_angles),
            np.zeros_like(self.lidar_angles),
        ]).astype(np.float64)
        self._lidar_geomgroup = np.array([1, 1, 1, 1, 1, 1], dtype=np.uint8)
        self._lidar_geomid_out = np.zeros(1, dtype=np.int32)
        self._perception_update_stride = 1
        self._perception_update_counter = 0
        self._visual_coverage_update_interval = 0.10
        self._last_visual_coverage_update_time = 0.0
        self._last_perception_pose = None
        self._last_perception_yaw = 0.0
        self._last_perception_map_time = 0.0
        self._perception_pose_min_translation = 0.045
        self._perception_pose_min_yaw = np.deg2rad(3.0)
        self._perception_max_skip_interval = 0.18
        self.runtime_fast_step = True
            
        # Stable-Baselines loads the checkpoint against the training observation shape.
        self.obs_num_cells_local = 24
        half_local = self.obs_num_cells_local / 2.0
        local_x, local_y = np.meshgrid(np.arange(self.obs_num_cells_local), np.arange(self.obs_num_cells_local))
        self.obs_local_coords_base = np.stack((local_x.flatten() - half_local, local_y.flatten() - half_local), axis=1)

        self.observation_space = gym.spaces.Dict({
            "grid_map": gym.spaces.Box(low=0, high=255, shape=(2, self.obs_num_cells_local, self.obs_num_cells_local), dtype=np.uint8),
            "state_history": gym.spaces.Box(low=-np.inf, high=np.inf, shape=(self.history_length * self.state_feature_dim,), dtype=np.float32)
        })
        
        self.agent_assigned_target = None
        self.current_frontiers =[]
        self._dashboard_fig = None
        self._dashboard_renderer = None
        self._dashboard_last_draw_time = 0.0
        self._dashboard_min_interval = 0.12
        self._dashboard_scene_interval = 0.50
        self._dashboard_map_interval = 0.25
        self._dashboard_last_scene_time = 0.0
        self._dashboard_last_map_time = 0.0
        self._dashboard_render_size = (480, 640)
        self._dashboard_scene_camera = None
        self._dashboard_scene_distance = 27.5
        self._dashboard_scene_lookat = np.array([0.0, 0.0, 0.0], dtype=np.float64)
        self._dashboard_layout_path = Path(__file__).with_name("dashboard_layout.json")
        self._dashboard_ax_log = None
        self._dashboard_ax_input = None
        self._dashboard_ax_submit = None
        self._dashboard_input_box = None
        self._dashboard_submit_button = None
        self._dashboard_command_queue = Queue()
        self._dashboard_log_text = None
        self._dashboard_log_lines = []
        self._dashboard_log_max_lines = 5
        self._dashboard_font_props = None
        self.enable_mujoco_viewer = enable_mujoco_viewer
        self._mujoco_viewer = None
        self._mujoco_viewer_last_sync = 0.0
        self._mujoco_viewer_min_interval = 0.08
        self.rgb_camera_name = "d435i_depth"
        self.rgb_camera_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_CAMERA, self.rgb_camera_name)
        if self.rgb_camera_id == -1:
            self.rgb_camera_id = getattr(self, "camera_id", -1)
        self._vision_rgb_renderer = None
        self._vision_depth_renderer = None
        self._vision_size = self._dashboard_render_size
        self._vision_last_time = 0.0
        self._vision_min_interval = 0.10
        self._vision_display_min_interval = 0.10
        self._vision_detect_min_interval = 0.35
        self._command_vision_min_interval = 1.10
        self._vision_last_display_time = 0.0
        self._vision_last_detect_time = 0.0
        self._vision_last_rgb = None
        self._vision_last_depth = None
        self._vision_last_annotated_rgb = None
        self._perf_render_ema = 0.0
        self._perf_vision_ema = 0.0
        self._perf_last_log_time = 0.0
        self._fast_dashboard_window = "Agent Fast Dashboard"
        self._fast_dashboard_ready = False
        self._fast_scene_img = None
        self._fast_last_canvas = None
        self._fast_last_scene_time = 0.0
        self._fast_scene_interval = 0.50
        self._fast_last_log_time = 0.0
        self._is_resetting = False
        self.visual_detections = []
        self.visual_landmarks_world = {}
        self.visual_seen_grid = None
        self._notified_visual_landmarks = set()
        self._qt_force_vision_requested = False
        self._qt_force_vision_request_time = 0.0
        self._qt_force_vision_done_time = 0.0
        self.enable_semantic_segmentation = False
        self.semantic_segmenter = None
        self.semantic_facts = {
            "wall_ratio": 0.0,
            "floor_ratio": 0.0,
            "status": "disabled",
        }

        if not hasattr(self, 'obstacle_geom_ids'):
            self.obstacle_geom_ids = set()
        for i in range(self.model.ngeom):
            geom_name = mujoco.mj_id2name(self.model, mujoco.mjtObj.mjOBJ_GEOM, i)
            if geom_name and ('gen_wall' in geom_name or 'boundary' in geom_name or 'furniture_' in geom_name):
                self.obstacle_geom_ids.add(i)

        self.landmark_names =[
            "landmark_red_body", "landmark_blue_body",
            "landmark_green_body", "landmark_yellow_body",
        ]
        self.landmark_mocap_ids = {}
        for name in self.landmark_names:
            body_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, name)
            if body_id != -1:
                self.landmark_mocap_ids[name] = self.model.body_mocapid[body_id]

        def persistent_update_from_lidar(self_map, robot_pos, hit_points, valid_mask):
            hit_points = np.asarray(hit_points, dtype=np.float32)
            valid_mask = np.asarray(valid_mask, dtype=bool)
            if hit_points.size == 0:
                return

            origin = np.asarray(robot_pos, dtype=np.float32)
            vectors = hit_points - origin[None, :]
            ray_lengths = np.linalg.norm(vectors, axis=1)
            nonzero = ray_lengths > 1e-4
            directions = np.zeros_like(vectors, dtype=np.float32)
            directions[nonzero] = vectors[nonzero] / ray_lengths[nonzero, None]

            step_m = max(0.12, 1.0 / self_map.resolution)
            free_lengths = np.maximum(0.0, ray_lengths - valid_mask.astype(np.float32) * step_m)
            max_steps = int(np.ceil(float(np.max(free_lengths)) / step_m)) if len(free_lengths) else 0
            if max_steps > 0:
                sample_dists = step_m * np.arange(1, max_steps + 1, dtype=np.float32)
                sample_mask = sample_dists[None, :] <= free_lengths[:, None]
                if np.any(sample_mask):
                    sample_x = origin[0] + directions[:, 0:1] * sample_dists[None, :]
                    sample_y = origin[1] + directions[:, 1:2] * sample_dists[None, :]
                    c_free = ((sample_x[sample_mask] + self_map.world_origin_offset_m[0]) * self_map.resolution).astype(np.int32)
                    r_free = self_map.num_cells_world - 1 - (
                        (sample_y[sample_mask] + self_map.world_origin_offset_m[1]) * self_map.resolution
                    ).astype(np.int32)
                    valid_free = (
                        (r_free >= 0) & (r_free < self_map.num_cells_world)
                        & (c_free >= 0) & (c_free < self_map.num_cells_world)
                    )
                    if np.any(valid_free):
                        free_update_mask = getattr(self_map, "_free_update_mask", None)
                        if free_update_mask is None or free_update_mask.shape != self_map.grid.shape:
                            free_update_mask = np.zeros_like(self_map.grid, dtype=bool)
                            self_map._free_update_mask = free_update_mask
                        else:
                            free_update_mask.fill(False)
                        free_update_mask[r_free[valid_free], c_free[valid_free]] = True
                        update_mask = free_update_mask & (self_map.grid < 0.5)
                        self_map.grid[update_mask] = (
                            self_map.grid[update_mask] * self_map.decay_rate
                            + self_map.log_odds_miss
                        )

            hit_points_valid = hit_points[valid_mask]
            if len(hit_points_valid) > 0:
                hits_indices = ((hit_points_valid + self_map.world_origin_offset_m) * self_map.resolution).astype(int)
                r_hits = self_map.num_cells_world - 1 - hits_indices[:, 1]
                c_hits = hits_indices[:, 0]
                valid_hits = (r_hits >= 0) & (r_hits < self_map.num_cells_world) & (c_hits >= 0) & (c_hits < self_map.num_cells_world)
                r_hits, c_hits = r_hits[valid_hits], c_hits[valid_hits]
                np.add.at(self_map.grid, (r_hits, c_hits), self_map.log_odds_hit)

            np.clip(self_map.grid, self_map.log_odds_min, self_map.log_odds_max, out=self_map.grid)

        self.grid_map.update_from_lidar = types.MethodType(persistent_update_from_lidar, self.grid_map)
        
    def _randomize_map(self):
        generator = MazeGridGenerator(world_size=20.0, grid_dim=8)
        generated_walls = generator.generate(self.np_random if hasattr(self, 'np_random') else np.random)
        
        wall_idx = 0
        self.static_walls_info =[]
        
        for i in range(self.model.ngeom):
            geom_name = mujoco.mj_id2name(self.model, mujoco.mjtObj.mjOBJ_GEOM, i)
            if geom_name and ('static_wall' in geom_name or 'gen_wall' in geom_name):
                if wall_idx < len(generated_walls):
                    wall_data = generated_walls[wall_idx]
                    
                    body_id = self.model.geom_bodyid[i]
                    if body_id != 0:
                        self.model.body_pos[body_id] = wall_data['pos']
                    else:
                        self.model.geom_pos[i] = wall_data['pos']
                        
                    self.model.geom_size[i] = wall_data['size']
                    self.static_walls_info.append({
                        'pos': wall_data['pos'].copy(), 
                        'size': wall_data['size'].copy()
                    })
                    wall_idx += 1
                else:
                    body_id = self.model.geom_bodyid[i]
                    if body_id != 0:
                        self.model.body_pos[body_id] = np.array([0, 0, -100])
                    else:
                        self.model.geom_pos[i] = np.array([0, 0, -100])
                    self.model.geom_size[i] = np.array([0.1, 0.1, 0.1])
                    
        print(f"\n[系统] 迷宫需要 {len(generated_walls)} 堵墙，物理引擎提取了 {wall_idx} 堵。")

    def _is_valid_spawn_pos(self, pos, min_dist_to_wall=0.8, min_dist_to_goal=3.0):
        half_size = 9.5 
        if abs(pos[0]) > half_size or abs(pos[1]) > half_size: return False
        robot_radius = 0.4
        safe_margin = robot_radius + min_dist_to_wall 
        for wall in self.static_walls_info:
            w_pos, w_size = wall['pos'], wall['size']
            if (w_pos[0] - w_size[0] - safe_margin) < pos[0] < (w_pos[0] + w_size[0] + safe_margin):
                if (w_pos[1] - w_size[1] - safe_margin) < pos[1] < (w_pos[1] + w_size[1] + safe_margin):
                    return False 
        for furniture in self._get_furniture_collision_info():
            f_pos, f_size = furniture['pos'], furniture['size']
            if (f_pos[0] - f_size[0] - safe_margin) < pos[0] < (f_pos[0] + f_size[0] + safe_margin):
                if (f_pos[1] - f_size[1] - safe_margin) < pos[1] < (f_pos[1] + f_size[1] + safe_margin):
                    return False
        if getattr(self, 'goal_pos', None) is not None and np.linalg.norm(pos - self.goal_pos) < min_dist_to_goal:
            return False
        return True

    def _get_furniture_collision_info(self):
        furniture_info = []
        for i in range(self.model.ngeom):
            geom_name = mujoco.mj_id2name(self.model, mujoco.mjtObj.mjOBJ_GEOM, i)
            if not geom_name or 'furniture_' not in geom_name or 'collision' not in geom_name:
                continue
            furniture_info.append({
                'pos': self.data.geom_xpos[i].copy(),
                'size': self.model.geom_size[i].copy(),
            })
        return furniture_info

    def _set_mocap_body_pose(self, body_name, pos, yaw):
        body_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, body_name)
        if body_id == -1:
            return False
        mocap_id = self.model.body_mocapid[body_id]
        if mocap_id < 0:
            return False
        quat_xyzw = R.from_euler('z', yaw).as_quat()
        self.data.mocap_pos[mocap_id] = [pos[0], pos[1], pos[2]]
        self.data.mocap_quat[mocap_id] = [quat_xyzw[3], quat_xyzw[0], quat_xyzw[1], quat_xyzw[2]]
        return True

    def _randomize_furniture(self):
        """将椅子随机放置在一面墙旁边"""
        body_name = "furniture_chair_0"
        body_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, body_name)
        if body_id < 0:
            return
        mocap_id = self.model.body_mocapid[body_id]
        if mocap_id < 0:
            return

        if not self.static_walls_info:
            return

        chair_half_xy = 0.28
        chair_height = 0.52

        for _ in range(50):
            wall = self.static_walls_info[self.np_random.integers(0, len(self.static_walls_info))]
            wpos = wall['pos']
            wsize = wall['size']
            is_vertical = wsize[0] < wsize[1]
            side = self.np_random.choice([-1, 1])

            if is_vertical:
                offset = wsize[0] + chair_half_xy + 0.08
                x = wpos[0] + side * offset
                y = wpos[1] + self.np_random.uniform(-max(wsize[1] - 0.4, 0), max(wsize[1] - 0.4, 0))
                yaw = math.pi / 2 if side > 0 else -math.pi / 2
            else:
                offset = wsize[1] + chair_half_xy + 0.08
                x = wpos[0] + self.np_random.uniform(-max(wsize[0] - 0.4, 0), max(wsize[0] - 0.4, 0))
                y = wpos[1] + side * offset
                yaw = math.pi if side > 0 else 0

            collision = False
            for w in self.static_walls_info:
                wp = w['pos']
                ws = w['size']
                if abs(x - wp[0]) < chair_half_xy + ws[0] + 0.05 and abs(y - wp[1]) < chair_half_xy + ws[1] + 0.05:
                    collision = True
                    break
            if collision:
                continue

            self.data.mocap_pos[mocap_id] = [x, y, 0.0]
            quat_xyzw = R.from_euler('z', yaw).as_quat()
            self.data.mocap_quat[mocap_id] = [quat_xyzw[3], quat_xyzw[0], quat_xyzw[1], quat_xyzw[2]]
            break

        mujoco.mj_forward(self.model, self.data)

    def _reset_goal(self):
        if getattr(self, 'agent_assigned_target', None) is not None:
            self.goal_pos = self.agent_assigned_target.copy()
        else:
            for _ in range(200):
                x = self.np_random.uniform(-8.5, 8.5)
                y = self.np_random.uniform(-8.5, 8.5)
                candidate = np.array([x, y])
                if self._is_valid_spawn_pos(candidate, min_dist_to_wall=0.6, min_dist_to_goal=0.0):
                    self.goal_pos = candidate
                    break
                    
        target_site_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_SITE, 'target_goal')
        if target_site_id != -1:
            self.model.site_pos[target_site_id][:2] = self.goal_pos

    def set_agent_goal(self, target_x, target_y):
        self.agent_assigned_target = np.array([target_x, target_y], dtype=np.float32)
        self.goal_pos = self.agent_assigned_target.copy()

        target_site_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_SITE, 'target_goal')
        if target_site_id != -1:
            self.model.site_pos[target_site_id][:2] = self.goal_pos

        self.path_update_counter = 0
        self._update_path()

        # If A* cannot find a path yet, use a short forward segment instead of
        # sending the policy a straight-line target through unknown space.
        if self.current_path is None:
            robot_pos = self.data.xpos[self.robot_base_body_id][:2].copy()
            mat = self.data.xmat[self.robot_base_body_id].reshape(3, 3)
            yaw = np.arctan2(mat[1, 0], mat[0, 0])
            forward = np.array([np.cos(yaw), np.sin(yaw)])
            end_pos = robot_pos + forward * 1.5
            end_pos = np.clip(end_pos, -9.0, 9.0)
            self.current_path = np.array([robot_pos, end_pos])
            self.current_waypoint_index = 1
            self.current_target_waypoint = end_pos
            self._path_fallback = True
        else:
            self._path_fallback = False

        self.current_step = 0
        robot_pos = self.data.xpos[self.robot_base_body_id][:2]
        self.dist_to_goal_start = np.linalg.norm(robot_pos - self.goal_pos)
        if self.dist_to_goal_start < 0.1: self.dist_to_goal_start = 0.1
        self.prev_dist_to_goal = self.dist_to_goal_start

        # Keep a few recent GRU frames so a target switch is closer to the
        # training distribution than an all-zero history reset.
        keep_steps = min(5, len(self.state_history_buffer))
        if keep_steps > 0:
            recent_history = list(self.state_history_buffer)[-keep_steps:]
            self.state_history_buffer.clear()
            for h in recent_history:
                self.state_history_buffer.append(h)
        else:
            self.state_history_buffer.clear()

        self._goal_switch_step = self.current_step
        self._prev_goal_pos = getattr(self, 'goal_pos', None)

        # Reset low-level controller state so old actions and smoothers do not
        # bleed into the next navigation target.
        if hasattr(self, 'locomotion_controller'):
            self.locomotion_controller.reset()

    def _update_path(self):
        robot_pos = self.data.xpos[self.robot_base_body_id][:2]
        path = self.path_planner.find_path(robot_pos, self.goal_pos)

        if path is not None and len(path) > 1:
            self.current_path = path
            self.current_waypoint_index = 1
        else:
            self.current_path = None

    def reset(self, seed=None, options=None):
        self._is_resetting = True
        for mocap_id in self.landmark_mocap_ids.values():
            self.data.mocap_pos[mocap_id] =[100, 100, -10]

        gym.Env.reset(self, seed=seed)
        self._randomize_map()

        if getattr(self, 'obstacle_controller', None) is not None:
            if not hasattr(self.obstacle_controller, 'update_walls'):
                def update_walls(self_obj, walls):
                    self_obj.cost_grid = self_obj._create_cost_grid_from_walls(walls)
                    self_obj.free_indices = np.argwhere(np.isfinite(self_obj.cost_grid))
                self.obstacle_controller.update_walls = types.MethodType(update_walls, self.obstacle_controller)
            self.obstacle_controller.update_walls(self.static_walls_info)

        mujoco.mj_resetData(self.model, self.data)
        self.grid_map.reset()
        self.locomotion_controller.reset()
        self.state_history_buffer.clear()
        self.visual_detections = []
        self.visual_landmarks_world.clear()
        self.visual_seen_grid = np.zeros_like(self.grid_map.grid, dtype=np.float32)
        self._notified_visual_landmarks.clear()
        self._qt_force_vision_requested = False
        self._qt_force_vision_request_time = 0.0
        self._qt_force_vision_done_time = 0.0
        self._vision_last_rgb = None
        self._vision_last_depth = None
        self._vision_last_annotated_rgb = None
        self._vision_last_time = 0.0
        self._last_perception_pose = None
        self._last_perception_yaw = 0.0
        self._last_perception_map_time = 0.0
        self.current_step = 0
        self.last_applied_action.fill(0)
        
        self.agent_assigned_target = None
        self._reset_goal()

        if getattr(self, 'obstacle_controller', None) is not None:
            self.obstacle_controller.reset()
            initial_obs_pos_2d = self.obstacle_controller.get_current_position()
            self.data.mocap_pos[self.dyn_obs_mocap_id] =[initial_obs_pos_2d[0], initial_obs_pos_2d[1], 0.5]
        elif getattr(self, 'dyn_obs_mocap_id', -1) != -1:
            self.data.mocap_pos[self.dyn_obs_mocap_id] =[0.0, 0.0, -10.0]

        self._randomize_furniture()
        mujoco.mj_forward(self.model, self.data)

        robot_spawn_pos = None; robot_spawn_yaw = 0.0; valid_pos_found = False
        for _ in range(300): 
            x = self.np_random.uniform(-8.5, 8.5)
            y = self.np_random.uniform(-8.5, 8.5)
            candidate_pos = np.array([x, y])
            if self._is_valid_spawn_pos(candidate_pos, min_dist_to_wall=0.8):
                robot_spawn_pos = candidate_pos
                robot_spawn_yaw = self.np_random.uniform(-math.pi, math.pi)
                valid_pos_found = True
                break
        
        if not valid_pos_found: 
            self._is_resetting = False
            return self.reset(seed=seed, options=options)
        
        self.data.qpos[0] = robot_spawn_pos[0]
        self.data.qpos[1] = robot_spawn_pos[1]
        self.data.qpos[2] = 0.05 
        quat_xyzw = R.from_euler('z', robot_spawn_yaw).as_quat()
        self.data.qpos[3:7] =[quat_xyzw[3], quat_xyzw[0], quat_xyzw[1], quat_xyzw[2]]
        self.data.qpos[-28:] = np.array([
            0.0, 0.0, 0.0, 0.0, -0.26, 0.0, 0.0, 0.52, 0.0, 0.0, -0.26, 0.0, 0.0, 0.0,
            0.0, 0.0, 0.0, 0.0, -0.26, 0.0, 0.0, 0.52, 0.0, 0.0, -0.26, 0.0, 0.0, 0.0
        ], dtype=np.double)
        self.data.qvel[:] = 0.0
        self.data.ctrl[:] = 0.0

        mujoco.mj_forward(self.model, self.data)
        
        try:
            for _ in range(20):
                self.data.ctrl[:] = 0
                mujoco.mj_step(self.model, self.data)
        except Exception: 
            self._is_resetting = False
            return self.reset(seed=seed, options=options)

        is_colliding = self._check_collision()
        robot_quat = self.data.qpos[3:7]
        r = R.from_quat([robot_quat[1], robot_quat[2], robot_quat[3], robot_quat[0]])
        z_axis = r.apply([0, 0, 1])
        is_fallen = z_axis[2] < 0.5

        if is_colliding or is_fallen: 
            self._is_resetting = False
            return self.reset(seed=seed, options=options)
        
        self.dist_to_goal_start = np.linalg.norm(self.data.qpos[:2] - self.goal_pos)
        if self.dist_to_goal_start < 0.1: self.dist_to_goal_start = 0.1
        self.prev_dist_to_goal = self.dist_to_goal_start 

        self._randomize_landmarks()
        mujoco.mj_forward(self.model, self.data)
        self._update_perception()
        self.path_update_counter = 0
        self._update_path()
        
        if self.current_path is None: 
            self._is_resetting = False
            return self.reset(seed=seed, options=options)
        self._is_resetting = False
        return self._get_obs(), {}

    def _randomize_landmarks(self):
        placed_positions =[]
        robot_pos = self.data.xpos[self.robot_base_body_id][:2]

        for name, mocap_id in self.landmark_mocap_ids.items():
            placed = False
            for _ in range(300):
                x = np.random.uniform(-8.5, 8.5)
                y = np.random.uniform(-8.5, 8.5)
                candidate = np.array([x, y])

                valid = True
                for wall in self.static_walls_info:
                    p, s = wall['pos'], wall['size']
                    if abs(candidate[0] - p[0]) < s[0] + 0.8 and abs(candidate[1] - p[1]) < s[1] + 0.8:
                        valid = False
                        break
                if valid:
                    for furniture in self._get_furniture_collision_info():
                        p, s = furniture['pos'], furniture['size']
                        if abs(candidate[0] - p[0]) < s[0] + 0.8 and abs(candidate[1] - p[1]) < s[1] + 0.8:
                            valid = False
                            break

                if valid:
                    for prev in placed_positions:
                        if np.linalg.norm(candidate - prev) < 3.0:
                            valid = False
                            break

                if valid and np.linalg.norm(candidate - robot_pos) < 2.0:
                    valid = False

                if valid:
                    self.data.mocap_pos[mocap_id] =[x, y, 0.5]
                    placed_positions.append(candidate)
                    placed = True
                    break

            if not placed:
                fx, fy = np.random.uniform(-6, 6), np.random.uniform(-6, 6)
                self.data.mocap_pos[mocap_id] =[fx, fy, 0.5]
                placed_positions.append(np.array([fx, fy]))

    def get_landmark_positions_debug(self):
        positions = {}
        for name, mocap_id in self.landmark_mocap_ids.items():
            pos = self.data.mocap_pos[mocap_id][:2].copy()
            landmark_id = name.replace("_body", "")
            positions[landmark_id] = pos
        return positions

    def _sense_lidar(self):
        if self.camera_id != -1:
            lidar_origin = self.data.cam_xpos[self.camera_id].copy()
        else:
            lidar_origin = self.data.xpos[self.robot_base_body_id].copy()
            lidar_origin[2] += 0.2

        body_mat = self.data.xmat[self.robot_base_body_id].reshape(3, 3)
        global_rays = self._lidar_local_rays_cache @ body_mat.T

        hit_points = np.empty((self.lidar_num_rays, 2), dtype=np.float32)
        valid_mask = np.zeros(self.lidar_num_rays, dtype=bool)
        body_exclude = int(self.robot_base_body_id)

        for i in range(self.lidar_num_rays):
            vec = global_rays[i]
            dist = mujoco.mj_ray(
                self.model,
                self.data,
                lidar_origin,
                vec,
                self._lidar_geomgroup,
                1,
                body_exclude,
                self._lidar_geomid_out,
            )
            if dist != -1 and dist < self.lidar_max_range:
                endpoint = lidar_origin[:2] + vec[:2] * dist
                valid_mask[i] = True
            else:
                endpoint = lidar_origin[:2] + vec[:2] * self.lidar_max_range
            hit_points[i] = endpoint

        return hit_points, valid_mask, lidar_origin
        
    def _update_perception(self):
        if self.render_mode == 'qt_dashboard':
            self._perception_update_counter += 1
            now = time.time()
            robot_pos, robot_yaw = self._get_robot_pose()
            should_update_map = self._last_perception_pose is None
            if not should_update_map:
                moved = float(np.linalg.norm(robot_pos - self._last_perception_pose))
                yaw_delta = abs((robot_yaw - self._last_perception_yaw + np.pi) % (2.0 * np.pi) - np.pi)
                stale = now - self._last_perception_map_time >= self._perception_max_skip_interval
                should_update_map = (
                    moved >= self._perception_pose_min_translation
                    or yaw_delta >= self._perception_pose_min_yaw
                    or stale
                )
            if should_update_map and self._perception_update_counter % self._perception_update_stride == 0:
                super()._update_perception()
                self._last_perception_pose = robot_pos
                self._last_perception_yaw = robot_yaw
                self._last_perception_map_time = now
            if now - self._last_visual_coverage_update_time >= self._visual_coverage_update_interval:
                self._mark_visual_coverage()
                self._last_visual_coverage_update_time = now
            return

        super()._update_perception()
        if self.render_mode == 'dashboard':
            self._update_dashboard()
            return
        if self.render_mode == 'fast_dashboard':
            if not self._is_resetting:
                self._update_fast_dashboard()
            return
        if self.render_mode == 'qt_dashboard':
            return

        if self.render_mode == 'human' and self.grid_map.fig is not None:
            ax = self.grid_map.ax
            
            if not hasattr(self.grid_map, 'landmark_patches'):
                self.grid_map.landmark_patches = {}
                self.grid_map.landmark_texts = {}
                import matplotlib.pyplot as plt
                for lm_id in self._landmark_color_specs().keys():
                    color = lm_id.split('_')[1] if '_' in lm_id else 'gray'
                    patch = plt.Circle((0, 0), 0.4, color=color, alpha=0.6, zorder=6)
                    self.grid_map.landmark_patches[lm_id] = ax.add_patch(patch)
                    self.grid_map.landmark_texts[lm_id] = ax.text(0, 0, lm_id.replace("landmark_", ""), ha='center', va='bottom', fontsize=9, zorder=7)
                    patch.set_visible(False)
                    self.grid_map.landmark_texts[lm_id].set_visible(False)
            
            for lm_id in self._landmark_color_specs().keys():
                if lm_id in self.visual_landmarks_world:
                    pos = self.visual_landmarks_world[lm_id]["pos"]
                    self.grid_map.landmark_patches[lm_id].center = (pos[0], pos[1])
                    self.grid_map.landmark_texts[lm_id].set_position((pos[0], pos[1] + 0.5))
                    self.grid_map.landmark_patches[lm_id].set_visible(True)
                    self.grid_map.landmark_texts[lm_id].set_visible(True)
                else:
                    self.grid_map.landmark_patches[lm_id].set_visible(False)
                    self.grid_map.landmark_texts[lm_id].set_visible(False)

            if not hasattr(self.grid_map, 'frontier_scatter'):
                self.grid_map.frontier_scatter = ax.scatter([],[], c='green', s=15, zorder=5, alpha=0.8)
                
            if hasattr(self, 'current_frontiers') and len(self.current_frontiers) > 0:
                self.grid_map.frontier_scatter.set_offsets(self.current_frontiers)
                self.grid_map.frontier_scatter.set_visible(True)
            else:
                self.grid_map.frontier_scatter.set_visible(False)

            self.grid_map.fig.canvas.draw_idle()
            self.grid_map.fig.canvas.flush_events()

    def _ensure_dashboard(self):
        if self._dashboard_fig is not None:
            import matplotlib.pyplot as plt
            if plt.fignum_exists(self._dashboard_fig.number):
                return True

        import matplotlib.pyplot as plt
        plt.ion()

        layout = self._load_dashboard_layout()
        self._dashboard_fig = plt.figure(figsize=layout["figure_size"])
        self._dashboard_fig.canvas.manager.set_window_title("Agent System Dashboard")
        self._dashboard_ax_scene = self._dashboard_fig.add_axes(self._axes_rect(layout, "scene"))
        self._dashboard_ax_camera = self._dashboard_fig.add_axes(self._axes_rect(layout, "camera"))
        self._dashboard_ax_map = self._dashboard_fig.add_axes(self._axes_rect(layout, "map"))
        self._dashboard_ax_log = self._dashboard_fig.add_axes(self._axes_rect(layout, "log"))

        render_h, render_w = self._dashboard_render_size
        blank_scene = np.zeros((render_h, render_w, 3), dtype=np.uint8)
        blank_camera = np.zeros((render_h, render_w, 3), dtype=np.uint8)
        self._dashboard_scene_im = self._dashboard_ax_scene.imshow(blank_scene)
        self._dashboard_camera_im = self._dashboard_ax_camera.imshow(blank_camera)
        self._dashboard_ax_scene.set_aspect('equal', adjustable='box')

        half_world = self.grid_map.world_size_m / 2.0
        prob_grid = 1.0 - 1.0 / (1.0 + np.exp(self.grid_map.grid))
        self._dashboard_map_im = self._dashboard_ax_map.imshow(
            prob_grid,
            cmap='gray_r',
            vmin=0,
            vmax=1,
            extent=[-half_world, half_world, -half_world, half_world],
            origin='upper'
        )
        self._dashboard_robot_patch = plt.Circle((0, 0), 0.2, color='tab:blue', zorder=6)
        self._dashboard_ax_map.add_patch(self._dashboard_robot_patch)
        self._dashboard_robot_arrow = plt.Arrow(0, 0, 0.5, 0.0, width=0.18, color='tab:blue', zorder=7)
        self._dashboard_ax_map.add_patch(self._dashboard_robot_arrow)
        self._dashboard_goal_patch, = self._dashboard_ax_map.plot([], [], '*', color='red', markersize=12, zorder=7)
        self._dashboard_path_patch, = self._dashboard_ax_map.plot([], [], '-', color='cyan', linewidth=1.6, zorder=5)
        self._dashboard_frontier_scatter = self._dashboard_ax_map.scatter([], [], c='lime', s=8, zorder=4, alpha=0.8)
        self._dashboard_landmark_patches = {}
        self._dashboard_landmark_texts = {}
        for landmark_id in self._landmark_color_specs().keys():
            color = landmark_id.split('_')[1] if '_' in landmark_id else 'gray'
            patch = plt.Circle((0, 0), 0.35, color=color, alpha=0.75, zorder=6)
            self._dashboard_landmark_patches[landmark_id] = self._dashboard_ax_map.add_patch(patch)
            self._dashboard_landmark_texts[landmark_id] = self._dashboard_ax_map.text(
                0, 0, landmark_id.replace("landmark_", ""),
                ha='center', va='bottom', fontsize=8, zorder=7
            )
            patch.set_visible(False)
            self._dashboard_landmark_texts[landmark_id].set_visible(False)

        self._dashboard_ax_scene.set_title("Simulation  +/- zoom  arrows pan  f furniture  0 reset")
        self._dashboard_ax_camera.set_title("Robot Camera")
        self._dashboard_ax_map.set_title("Lidar Map")
        for ax in (self._dashboard_ax_scene, self._dashboard_ax_camera):
            ax.set_xticks([])
            ax.set_yticks([])
        self._dashboard_ax_log.set_xticks([])
        self._dashboard_ax_log.set_yticks([])
        self._dashboard_ax_log.set_facecolor("#f6f6f6")
        self._dashboard_ax_log.set_title("LLM Reasoning", fontsize=9, pad=2)
        self._dashboard_log_text = self._dashboard_ax_log.text(
            0.01, 0.92, "",
            transform=self._dashboard_ax_log.transAxes,
            ha='left', va='top',
            fontsize=7.5,
            fontproperties=self._get_dashboard_font_props(),
            color="#1f1f1f",
        )
        self._refresh_dashboard_log()
        self._dashboard_ax_map.set_xlabel("X (m)")
        self._dashboard_ax_map.set_ylabel("Y (m)")
        self._dashboard_ax_map.set_xlim(-half_world, half_world)
        self._dashboard_ax_map.set_ylim(-half_world, half_world)
        self._dashboard_ax_map.set_aspect('equal', adjustable='box')

        self._dashboard_scene_camera = mujoco.MjvCamera()
        self._dashboard_scene_camera.type = mujoco.mjtCamera.mjCAMERA_FREE
        self._dashboard_scene_camera.distance = self._dashboard_scene_distance
        self._dashboard_scene_camera.azimuth = 90.0
        self._dashboard_scene_camera.elevation = -89.0
        self._dashboard_scene_camera.lookat[:] = self._dashboard_scene_lookat
        self._dashboard_fig.canvas.mpl_connect("key_press_event", self._on_dashboard_key_press)

        if self._dashboard_renderer is None:
            try:
                self._dashboard_renderer = mujoco.Renderer(self.model, height=render_h, width=render_w)
            except Exception as exc:
                print(f"[Dashboard] MuJoCo offscreen renderer 初始化失败: {exc}")
                self._dashboard_renderer = False

        self._dashboard_fig.show()
        self._make_dashboard_topmost()
        return True

    def _create_dashboard_input_widgets(self):
        try:
            from matplotlib.widgets import TextBox, Button
            self._dashboard_input_box = TextBox(
                self._dashboard_ax_input,
                "",
                initial="",
                textalignment="left",
                color="#ffffff",
                hovercolor="#f0f0f0",
            )
            self._dashboard_input_box.label.set_text("")
            self._dashboard_input_box.text_disp.set_fontproperties(self._get_dashboard_font_props())
            self._dashboard_input_box.text_disp.set_fontsize(8)
            self._dashboard_input_box.on_submit(self._submit_dashboard_command)

            self._dashboard_submit_button = Button(
                self._dashboard_ax_submit,
                "Send",
                color="#e8e8e8",
                hovercolor="#d8d8d8",
            )
            self._dashboard_submit_button.on_clicked(lambda event: self._submit_dashboard_command(self._dashboard_input_box.text))
        except Exception as exc:
            print(f"[Dashboard] 输入框初始化失败: {exc}")

    def _submit_dashboard_command(self, text):
        command = str(text).strip()
        if not command:
            return
        self._dashboard_command_queue.put(command)
        if self._dashboard_input_box is not None:
            self._dashboard_input_box.set_val("")

    def pop_dashboard_command(self):
        try:
            return self._dashboard_command_queue.get_nowait()
        except Exception:
            return None

    def _apply_dashboard_scene_camera(self):
        if self._dashboard_scene_camera is None:
            return
        self._dashboard_scene_camera.distance = float(self._dashboard_scene_distance)
        self._dashboard_scene_camera.lookat[:] = self._dashboard_scene_lookat

    def _on_dashboard_key_press(self, event):
        key = event.key
        if key in ("+", "=", "up"):
            if key in ("+", "="):
                self._dashboard_scene_distance = max(4.0, self._dashboard_scene_distance * 0.82)
            else:
                self._dashboard_scene_lookat[1] += max(0.25, self._dashboard_scene_distance * 0.04)
        elif key in ("-", "_", "down"):
            if key in ("-", "_"):
                self._dashboard_scene_distance = min(45.0, self._dashboard_scene_distance * 1.22)
            else:
                self._dashboard_scene_lookat[1] -= max(0.25, self._dashboard_scene_distance * 0.04)
        elif key == "left":
            self._dashboard_scene_lookat[0] -= max(0.25, self._dashboard_scene_distance * 0.04)
        elif key == "right":
            self._dashboard_scene_lookat[0] += max(0.25, self._dashboard_scene_distance * 0.04)
        elif key == "0":
            self._dashboard_scene_distance = 27.5
            self._dashboard_scene_lookat[:] = [0.0, 0.0, 0.0]
        elif key == "f":
            furniture = self._get_furniture_collision_info()
            if furniture:
                self._dashboard_scene_lookat[:2] = furniture[0]["pos"][:2]
                self._dashboard_scene_lookat[2] = 0.0
                self._dashboard_scene_distance = 8.0
        else:
            return

        self._apply_dashboard_scene_camera()
        self.append_dashboard_log(
            f"视角: distance={self._dashboard_scene_distance:.1f}, "
            f"lookat=({self._dashboard_scene_lookat[0]:.1f},{self._dashboard_scene_lookat[1]:.1f})"
        )
        self._update_dashboard(force=True)

    def _load_dashboard_layout(self):
        default_layout = {
            "figure_size": [12.0, 7.2],
            "axes": {
                "scene": {"left": 0.0032, "bottom": 0.0032, "width": 0.5548, "height": 0.8568},
                "camera": {"left": 0.5688, "bottom": 0.5552, "width": 0.4290, "height": 0.4419},
                "map": {"left": 0.5645, "bottom": 0.0014, "width": 0.4333, "height": 0.5447},
                "log": {"left": 0.0032, "bottom": 0.8672, "width": 0.5602, "height": 0.1281},
            },
        }
        if not self._dashboard_layout_path.exists():
            return default_layout
        try:
            layout = json.loads(self._dashboard_layout_path.read_text(encoding="utf-8"))
            axes = layout.get("axes", {})
            if not all(key in axes for key in ("scene", "camera", "map", "log")):
                return default_layout
            return {
                "figure_size": layout.get("figure_size", default_layout["figure_size"]),
                "axes": axes,
            }
        except Exception:
            return default_layout

    def _get_dashboard_font_props(self):
        if self._dashboard_font_props is not None:
            return self._dashboard_font_props
        try:
            from matplotlib import font_manager
            font_paths = [
                "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc",
                "/usr/share/fonts/truetype/droid/DroidSansFallbackFull.ttf",
                "/usr/share/fonts/truetype/arphic/uming.ttc",
            ]
            for font_path in font_paths:
                if Path(font_path).exists():
                    self._dashboard_font_props = font_manager.FontProperties(fname=font_path)
                    return self._dashboard_font_props
        except Exception:
            pass
        self._dashboard_font_props = None
        return None

    @staticmethod
    def _landmark_color_specs():
        return {
            "landmark_red": {
                "box_color": (255, 40, 40),
                "mask": lambda rgb: (rgb[..., 0] > 120) & (rgb[..., 0] > rgb[..., 1] * 1.35) & (rgb[..., 0] > rgb[..., 2] * 1.35),
                "hsv": ((0, 55, 45), (10, 255, 255), (170, 55, 45), (179, 255, 255)),
            },
            "landmark_blue": {
                "box_color": (255, 40, 230),
                "mask": lambda rgb: (
                    (rgb[..., 0] > 105)
                    & (rgb[..., 2] > 95)
                    & (rgb[..., 1] < 130)
                    & (rgb[..., 0] > rgb[..., 1] * 1.35)
                    & (rgb[..., 2] > rgb[..., 1] * 1.25)
                ),
                "hsv": ((138, 45, 45), (172, 255, 255)),
            },
            "landmark_green": {
                "box_color": (30, 200, 70),
                "mask": lambda rgb: (rgb[..., 1] > 95) & (rgb[..., 1] > rgb[..., 0] * 1.25) & (rgb[..., 1] > rgb[..., 2] * 1.25),
                "hsv": ((40, 45, 35), (85, 255, 255)),
            },
            "landmark_yellow": {
                "box_color": (255, 220, 30),
                "mask": lambda rgb: (rgb[..., 0] > 130) & (rgb[..., 1] > 120) & (rgb[..., 2] < 120) & (np.abs(rgb[..., 0] - rgb[..., 1]) < 95),
                "hsv": ((18, 45, 45), (38, 255, 255)),
            },
        }

    def _get_vision_renderer(self, depth=False):
        render_h, render_w = self._vision_size
        if depth:
            if self._vision_depth_renderer is None:
                self._vision_depth_renderer = mujoco.Renderer(self.model, height=render_h, width=render_w)
                self._vision_depth_renderer.enable_depth_rendering()
            return self._vision_depth_renderer

        if self._dashboard_renderer not in (None, False):
            return self._dashboard_renderer
        if self._vision_rgb_renderer is None:
            self._vision_rgb_renderer = mujoco.Renderer(self.model, height=render_h, width=render_w)
        return self._vision_rgb_renderer

    def _render_vision_rgb_depth(self):
        if self.rgb_camera_id == -1:
            return None, None
        try:
            start = time.perf_counter()
            rgb_renderer = self._get_vision_renderer(depth=False)
            rgb_renderer.update_scene(self.data, camera=self.rgb_camera_id)
            rgb = rgb_renderer.render().copy()

            depth_renderer = self._get_vision_renderer(depth=True)
            depth_renderer.update_scene(self.data, camera=self.rgb_camera_id)
            depth = depth_renderer.render().copy()
            elapsed = time.perf_counter() - start
            self._perf_vision_ema = elapsed if self._perf_vision_ema == 0.0 else 0.9 * self._perf_vision_ema + 0.1 * elapsed
            return rgb, depth
        except Exception as exc:
            self.append_dashboard_log(f"视觉: RGB-D 渲染失败 {exc}")
            return None, None

    def _render_vision_rgb_only(self):
        if self.rgb_camera_id == -1:
            return None
        try:
            rgb_renderer = self._get_vision_renderer(depth=False)
            rgb_renderer.update_scene(self.data, camera=self.rgb_camera_id)
            return rgb_renderer.render().copy()
        except Exception as exc:
            self.append_dashboard_log(f"视觉: RGB 渲染失败 {exc}")
            return None

    def _pixel_depth_to_world(self, u, v, depth_m):
        if self.rgb_camera_id == -1 or not np.isfinite(depth_m) or depth_m <= 0:
            return None
        height, width = self._vision_size
        fovy = float(self.model.cam_fovy[self.rgb_camera_id])
        focal_y = height / (2.0 * np.tan(np.deg2rad(fovy) / 2.0))
        focal_x = (width / height) * focal_y
        cx, cy = width / 2.0, height / 2.0

        x_cam = (u - cx) * depth_m / focal_x
        y_cam = (v - cy) * depth_m / focal_y
        point_optical = np.array([x_cam, y_cam, depth_m], dtype=np.float64)

        optical_to_mujoco = R.from_euler('x', 180, degrees=True).as_matrix()
        point_camera = optical_to_mujoco @ point_optical
        cam_pos = self.data.cam_xpos[self.rgb_camera_id]
        cam_rot = self.data.cam_xmat[self.rgb_camera_id].reshape(3, 3)
        return cam_rot @ point_camera + cam_pos

    def _mark_visual_coverage(self, max_range_m=6.0):
        if self.visual_seen_grid is None or not hasattr(self, "grid_map"):
            return
        robot_pos, robot_yaw = self._get_robot_pose()
        grid_map = self.grid_map
        h, w = self.visual_seen_grid.shape
        prob_grid = 1.0 - 1.0 / (1.0 + np.exp(grid_map.grid))
        fov = np.deg2rad(70.0)
        if self.rgb_camera_id != -1:
            try:
                fov = np.deg2rad(float(self.model.cam_fovy[self.rgb_camera_id])) * 1.25
            except Exception:
                pass

        self.visual_seen_grid *= 0.9995
        n_rays = 49
        step_m = max(0.08, 0.5 / grid_map.resolution)
        for angle in np.linspace(robot_yaw - fov * 0.5, robot_yaw + fov * 0.5, n_rays):
            direction = np.array([np.cos(angle), np.sin(angle)], dtype=np.float32)
            for dist in np.arange(0.25, max_range_m + step_m, step_m):
                point = robot_pos + direction * dist
                c = int((point[0] + grid_map.world_origin_offset_m[0]) * grid_map.resolution)
                r = grid_map.num_cells_world - 1 - int((point[1] + grid_map.world_origin_offset_m[1]) * grid_map.resolution)
                if not (0 <= r < h and 0 <= c < w):
                    break
                self.visual_seen_grid[r, c] = 1.0
                if prob_grid[r, c] > 0.65:
                    break

    @staticmethod
    def _draw_detection_box(image, bbox, color):
        x0, y0, x1, y1 = [int(v) for v in bbox]
        h, w = image.shape[:2]
        x0, x1 = np.clip([x0, x1], 0, w - 1)
        y0, y1 = np.clip([y0, y1], 0, h - 1)
        image[y0:min(y0 + 3, h), x0:x1 + 1] = color
        image[max(y1 - 2, 0):y1 + 1, x0:x1 + 1] = color
        image[y0:y1 + 1, x0:min(x0 + 3, w)] = color
        image[y0:y1 + 1, max(x1 - 2, 0):x1 + 1] = color

    def _detect_colored_landmarks_from_rgbd(self, rgb, depth):
        if rgb is None or depth is None:
            return [], rgb
        rgb_float = rgb.astype(np.float32)
        hsv = cv2.cvtColor(rgb, cv2.COLOR_RGB2HSV)
        annotated = rgb.copy()
        detections = []
        min_area = max(30, int(rgb.shape[0] * rgb.shape[1] * 0.00025))
        max_area = int(rgb.shape[0] * rgb.shape[1] * 0.65)
        max_depth = 12.0

        for landmark_id, spec in self._landmark_color_specs().items():
            mask = spec["mask"](rgb_float)
            hsv_spec = spec.get("hsv")
            if hsv_spec is not None:
                if len(hsv_spec) == 2:
                    hsv_mask = cv2.inRange(hsv, np.array(hsv_spec[0], dtype=np.uint8), np.array(hsv_spec[1], dtype=np.uint8)) > 0
                else:
                    hsv_mask = (
                        (cv2.inRange(hsv, np.array(hsv_spec[0], dtype=np.uint8), np.array(hsv_spec[1], dtype=np.uint8)) > 0)
                        | (cv2.inRange(hsv, np.array(hsv_spec[2], dtype=np.uint8), np.array(hsv_spec[3], dtype=np.uint8)) > 0)
                    )
                mask = mask | hsv_mask
            mask = binary_opening(mask, structure=np.ones((3, 3), dtype=bool))
            labeled, count = label(mask)
            if count == 0:
                continue

            best = None
            area_min = min_area
            min_box = 4
            aspect_min = 0.12 if landmark_id == "landmark_blue" else 0.18
            aspect_max = 4.0 if landmark_id == "landmark_blue" else 3.2
            for label_idx, obj_slice in enumerate(find_objects(labeled), start=1):
                if obj_slice is None:
                    continue
                ys, xs = obj_slice
                component = labeled[obj_slice] == label_idx
                area = int(np.sum(component))
                if area < area_min:
                    continue
                y0, y1 = ys.start, ys.stop - 1
                x0, x1 = xs.start, xs.stop - 1
                box_w = x1 - x0 + 1
                box_h = y1 - y0 + 1
                if box_w < min_box or box_h < min_box:
                    continue
                if area > max_area:
                    continue
                aspect = box_w / max(box_h, 1)
                if aspect > aspect_max or aspect < aspect_min:
                    continue
                # Filter only low, flat blobs; close landmarks can legitimately fill the frame.
                if y1 > rgb.shape[0] * 0.985 and aspect > 1.8 and box_h < rgb.shape[0] * 0.35:
                    continue
                if best is None or area > best["area"]:
                    best = {"area": area, "bbox": (x0, y0, x1, y1), "component": component, "slice": obj_slice}

            if best is None:
                continue

            obj_slice = best["slice"]
            component = best["component"]
            local_depth = depth[obj_slice][component]
            valid_depth = local_depth[np.isfinite(local_depth) & (local_depth > 0.05) & (local_depth < max_depth)]
            if valid_depth.size < 5:
                continue
            depth_m = float(np.median(valid_depth))

            yy, xx = np.nonzero(component)
            x0, y0, x1, y1 = best["bbox"]
            sample_indices = []
            for frac_y in (0.25, 0.40, 0.55, 0.70):
                sample_indices.append((float((x0 + x1) * 0.5), float(y0 + (y1 - y0) * frac_y)))
            sample_indices.append((float(x0 + np.mean(xx)), float(y0 + np.mean(yy))))
            world_samples = []
            for u, v in sample_indices:
                px = int(np.clip(round(u), 0, depth.shape[1] - 1))
                py = int(np.clip(round(v), 0, depth.shape[0] - 1))
                sample_depth = depth[py, px]
                if not np.isfinite(sample_depth) or sample_depth <= 0.05 or sample_depth >= max_depth:
                    sample_depth = depth_m
                world = self._pixel_depth_to_world(u, v, float(sample_depth))
                if world is not None and np.all(np.isfinite(world)):
                    world_samples.append(world)
            if not world_samples:
                continue
            world_samples = np.asarray(world_samples)
            z_values = world_samples[:, 2]
            # Reject floor-colored false positives by checking reconstructed height.
            min_z = 0.16 if landmark_id == "landmark_blue" else 0.18
            max_z = 1.45 if landmark_id == "landmark_blue" else 1.35
            max_z_span = 0.95 if landmark_id == "landmark_blue" else 0.90
            if np.median(z_values) < min_z or np.median(z_values) > max_z:
                continue
            if np.max(z_values) - np.min(z_values) > max_z_span:
                continue
            world = np.median(world_samples, axis=0)
            # RGB-D gives the visible surface of the cylinder. Navigate to the
            # landmark center instead, otherwise the first observation can be
            # too close and the robot will declare arrival early.
            cam_xy = self.data.cam_xpos[self.rgb_camera_id][:2]
            surface_to_center = world[:2] - cam_xy
            surface_norm = np.linalg.norm(surface_to_center)
            if surface_norm > 1e-4:
                world[:2] += surface_to_center / surface_norm * 0.26

            detection = {
                "id": landmark_id,
                "pos": world[:2].astype(np.float32),
                "pos3d": world.astype(np.float32),
                "dist": depth_m,
                "bbox": best["bbox"],
                "area": best["area"],
            }
            detections.append(detection)
            self._draw_detection_box(annotated, best["bbox"], spec["box_color"])

        return detections, annotated

    def detect_visual_landmarks(self, force=False):
        now = time.time()
        if (
            not force
            and self._vision_last_annotated_rgb is not None
            and now - self._vision_last_detect_time < self._vision_detect_min_interval
        ):
            return self.visual_detections

        rgb, depth = self._render_vision_rgb_depth()
        self._vision_last_time = now
        self._vision_last_detect_time = now
        self._vision_last_rgb = rgb
        self._vision_last_depth = depth
        self._mark_visual_coverage()
        detections, annotated = self._detect_colored_landmarks_from_rgbd(rgb, depth)
        if bool(getattr(self, "debug_timing", False)) and force:
            print(f"[VISION] forced rgbd detections={len(detections)}", flush=True)
        self.visual_detections = detections
        meaning_map = {
            "landmark_red": "红色方块(可能是老板办公室)",
            "landmark_blue": "品红色圆柱体/方块(可能是会议室)",
            "landmark_green": "绿色方块(可能是大门)",
            "landmark_yellow": "黄色方块(可能是茶水间)",
        }
        for item in detections:
            previous = self.visual_landmarks_world.get(item["id"])
            if previous is not None:
                fused = item.copy()
                pos_delta = float(np.linalg.norm(previous["pos"] - item["pos"]))
                previous_weight = 0.35 if pos_delta > 0.75 else 0.65
                new_weight = 1.0 - previous_weight
                fused["pos"] = (previous_weight * previous["pos"] + new_weight * item["pos"]).astype(np.float32)
                if "pos3d" in previous and "pos3d" in item:
                    fused["pos3d"] = (previous_weight * previous["pos3d"] + new_weight * item["pos3d"]).astype(np.float32)
                self.visual_landmarks_world[item["id"]] = fused
                item["pos"] = fused["pos"]
                if "pos3d" in fused:
                    item["pos3d"] = fused["pos3d"]
            else:
                self.visual_landmarks_world[item["id"]] = item
            if item["id"] not in self._notified_visual_landmarks:
                self._notified_visual_landmarks.add(item["id"])
                meaning = meaning_map.get(item["id"], item["id"])
                pos = item.get("pos")
                if pos is not None:
                    self.append_dashboard_log(
                        f"视觉检测: 看到 {meaning}，估计坐标 ({pos[0]:.1f},{pos[1]:.1f})"
                    )
        self._vision_last_annotated_rgb = annotated if annotated is not None else rgb
        return detections

    def get_annotated_camera_image(self):
        now = time.time()
        if getattr(self, "_qt_command_busy", False):
            if now - self._vision_last_display_time >= max(self._vision_display_min_interval, 0.20):
                rgb = self._render_vision_rgb_only()
                if rgb is not None:
                    self._vision_last_rgb = rgb
                    self._vision_last_annotated_rgb = rgb
                self._vision_last_display_time = now
            if self._vision_last_annotated_rgb is not None:
                return self._vision_last_annotated_rgb
            render_h, render_w = self._vision_size
            return np.zeros((render_h, render_w, 3), dtype=np.uint8)
        if self._vision_last_annotated_rgb is None:
            self.detect_visual_landmarks()
            self._vision_last_display_time = now
        elif now - self._vision_last_display_time >= self._vision_display_min_interval:
            if now - self._vision_last_detect_time >= self._vision_detect_min_interval:
                self.detect_visual_landmarks()
            else:
                rgb = self._render_vision_rgb_only()
                if rgb is not None:
                    self._vision_last_rgb = rgb
                    self._vision_last_annotated_rgb = rgb
            self._vision_last_display_time = now
        if self._vision_last_annotated_rgb is not None:
            return self._vision_last_annotated_rgb
        render_h, render_w = self._vision_size
        return np.zeros((render_h, render_w, 3), dtype=np.uint8)

    def append_dashboard_log(self, message):
        text = " ".join(str(message).split())
        if not text:
            return
        if len(text) > 260:
            text = text[:257] + "..."
        timestamp = time.strftime("%H:%M:%S")
        self._dashboard_log_lines.append(f"[{timestamp}] {text}")
        self._dashboard_log_lines = self._dashboard_log_lines[-self._dashboard_log_max_lines:]
        self._refresh_dashboard_log()

    def _refresh_dashboard_log(self):
        if self._dashboard_log_text is None:
            return
        wrapped_lines = []
        for line in self._dashboard_log_lines:
            wrapped_lines.extend(textwrap.wrap(line, width=92, subsequent_indent="  ") or [""])
        self._dashboard_log_text.set_text("\n".join(wrapped_lines[-7:]))

    @staticmethod
    def _axes_rect(layout, key):
        rect = layout["axes"][key]
        return [rect["left"], rect["bottom"], rect["width"], rect["height"]]

    def _make_dashboard_topmost(self):
        try:
            manager = self._dashboard_fig.canvas.manager
            window = getattr(manager, 'window', None)
            if window is None:
                return
            if hasattr(window, 'wm_attributes'):
                window.wm_attributes('-topmost', 1)
            elif hasattr(window, 'setWindowFlag'):
                try:
                    from PyQt5 import QtCore
                    window.setWindowFlag(QtCore.Qt.WindowStaysOnTopHint, True)
                    window.show()
                except Exception:
                    try:
                        from PySide6 import QtCore
                        window.setWindowFlag(QtCore.Qt.WindowStaysOnTopHint, True)
                        window.show()
                    except Exception:
                        pass
            elif hasattr(window, 'SetWindowStyle'):
                import wx
                window.SetWindowStyle(window.GetWindowStyle() | wx.STAY_ON_TOP)
        except Exception:
            pass

    def _render_dashboard_view(self, camera=None):
        if self._dashboard_renderer is False:
            render_h, render_w = self._dashboard_render_size
            return np.zeros((render_h, render_w, 3), dtype=np.uint8)

        renderer = self._dashboard_renderer
        try:
            if camera is None:
                scene_camera = self._dashboard_scene_camera
                renderer.update_scene(self.data, camera=scene_camera)
            else:
                renderer.update_scene(self.data, camera=camera)
            return renderer.render().copy()
        except TypeError:
            try:
                renderer.update_scene(self.data, camera_id=camera)
                return renderer.render().copy()
            except Exception:
                return np.zeros((480, 640, 3), dtype=np.uint8)
        except Exception:
            return np.zeros((480, 640, 3), dtype=np.uint8)

    def _update_dashboard(self, force=False):
        now = time.time()
        if not force and now - self._dashboard_last_draw_time < self._dashboard_min_interval:
            return
        perf_start = time.perf_counter()
        self._dashboard_last_draw_time = now

        if not self._ensure_dashboard():
            return

        import matplotlib.pyplot as plt
        if self._dashboard_fig is None or not plt.fignum_exists(self._dashboard_fig.number):
            return

        update_scene = force or now - self._dashboard_last_scene_time >= self._dashboard_scene_interval
        update_map = force or now - self._dashboard_last_map_time >= self._dashboard_map_interval

        if update_scene:
            scene_img = self._render_dashboard_view(camera=None)
            self._dashboard_scene_im.set_data(scene_img)
            self._dashboard_last_scene_time = now

        camera_img = self.get_annotated_camera_image()
        if self.enable_semantic_segmentation and SegFormerSemanticSegmenter is not None:
            if self.semantic_segmenter is None:
                self.semantic_segmenter = SegFormerSemanticSegmenter()
            camera_img, self.semantic_facts = self.semantic_segmenter.process(camera_img)
        else:
            self.semantic_facts = {
                "wall_ratio": 0.0,
                "floor_ratio": 0.0,
                "status": "disabled",
            }
        self._dashboard_camera_im.set_data(camera_img)
        self._update_semantic_title()

        if update_map:
            prob_grid = 1.0 - 1.0 / (1.0 + np.exp(self.grid_map.grid))
            self._dashboard_map_im.set_data(prob_grid)
            self._dashboard_last_map_time = now

        robot_pos, robot_yaw = self._get_robot_pose()
        self._dashboard_robot_patch.center = (robot_pos[0], robot_pos[1])
        self._dashboard_robot_arrow.set_data(
            x=robot_pos[0],
            y=robot_pos[1],
            dx=0.5 * np.cos(robot_yaw),
            dy=0.5 * np.sin(robot_yaw),
            width=0.18,
        )

        self._dashboard_goal_patch.set_data([self.goal_pos[0]], [self.goal_pos[1]])
        if self.current_path is not None and len(self.current_path) > 0:
            dists = np.linalg.norm(self.current_path - robot_pos, axis=1)
            closest_idx = int(np.argmin(dists))
            display_path = self.current_path[closest_idx:]
            if len(display_path) < 2:
                display_path = self.current_path
            self._dashboard_path_patch.set_data(display_path[:, 0], display_path[:, 1])
        else:
            self._dashboard_path_patch.set_data([], [])

        if hasattr(self, 'current_frontiers') and len(self.current_frontiers) > 0:
            self._dashboard_frontier_scatter.set_offsets(self.current_frontiers)
            self._dashboard_frontier_scatter.set_visible(True)
        else:
            self._dashboard_frontier_scatter.set_visible(False)

        seen_landmarks = set(self.visual_landmarks_world.keys())
        for landmark_id in self._landmark_color_specs().keys():
            patch = self._dashboard_landmark_patches[landmark_id]
            text = self._dashboard_landmark_texts[landmark_id]
            if landmark_id in seen_landmarks:
                pos = self.visual_landmarks_world[landmark_id]["pos"]
                patch.center = (pos[0], pos[1])
                patch.set_visible(True)
                text.set_position((pos[0], pos[1] + 0.45))
                text.set_visible(True)
            else:
                patch.set_visible(False)
                text.set_visible(False)

        self._refresh_dashboard_log()
        self._dashboard_fig.canvas.draw_idle()
        self._dashboard_fig.canvas.flush_events()
        elapsed = time.perf_counter() - perf_start
        self._perf_render_ema = elapsed if self._perf_render_ema == 0.0 else 0.9 * self._perf_render_ema + 0.1 * elapsed
        if now - self._perf_last_log_time > 5.0:
            self._perf_last_log_time = now
            self.append_dashboard_log(
                f"性能: dashboard {self._perf_render_ema * 1000:.0f}ms, RGB-D {self._perf_vision_ema * 1000:.0f}ms"
            )

    def _update_semantic_title(self):
        status = self.semantic_facts.get("status", "disabled")
        if status == "ok":
            wall_pct = self.semantic_facts.get("wall_ratio", 0.0) * 100.0
            floor_pct = self.semantic_facts.get("floor_ratio", 0.0) * 100.0
            self._dashboard_ax_camera.set_title(
                f"Robot Camera + Semantics  floor {floor_pct:.0f}%  wall {wall_pct:.0f}%"
            )
        elif status == "disabled":
            self._dashboard_ax_camera.set_title("Robot Camera")
        else:
            self._dashboard_ax_camera.set_title("Robot Camera + Semantics unavailable")

    def _world_to_fast_map_px(self, pos, width, height):
        half_world = self.grid_map.world_size_m / 2.0
        x = int((pos[0] + half_world) / self.grid_map.world_size_m * width)
        y = int((half_world - pos[1]) / self.grid_map.world_size_m * height)
        return int(np.clip(x, 0, width - 1)), int(np.clip(y, 0, height - 1))

    def _render_fast_map(self, width, height):
        prob_grid = 1.0 - 1.0 / (1.0 + np.exp(self.grid_map.grid))
        gray = np.clip((1.0 - prob_grid) * 255.0, 0, 255).astype(np.uint8)
        map_img = cv2.resize(gray, (width, height), interpolation=cv2.INTER_NEAREST)
        map_img = cv2.cvtColor(map_img, cv2.COLOR_GRAY2BGR)
        if self.visual_seen_grid is not None:
            visual = cv2.resize(self.visual_seen_grid.astype(np.float32), (width, height), interpolation=cv2.INTER_NEAREST)
            visual_mask = visual > 0.2
            if np.any(visual_mask):
                overlay = map_img.copy()
                overlay[visual_mask] = (255, 210, 60)
                map_img = cv2.addWeighted(overlay, 0.30, map_img, 0.70, 0)

        robot_pos, robot_yaw = self._get_robot_pose()
        rx, ry = self._world_to_fast_map_px(robot_pos, width, height)
        cv2.circle(map_img, (rx, ry), 6, (255, 80, 30), -1)
        ax = int(rx + 18 * np.cos(robot_yaw))
        ay = int(ry - 18 * np.sin(robot_yaw))
        cv2.arrowedLine(map_img, (rx, ry), (ax, ay), (255, 80, 30), 2, tipLength=0.35)

        gx, gy = self._world_to_fast_map_px(self.goal_pos, width, height)
        cv2.drawMarker(map_img, (gx, gy), (0, 0, 255), cv2.MARKER_STAR, 18, 2)

        if self.current_path is not None and len(self.current_path) > 1:
            pts = np.array([self._world_to_fast_map_px(p, width, height) for p in self.current_path], dtype=np.int32)
            cv2.polylines(map_img, [pts], False, (255, 255, 0), 2)

        if hasattr(self, 'current_frontiers') and len(self.current_frontiers) > 0:
            for point in self.current_frontiers[::max(1, len(self.current_frontiers) // 80)]:
                px, py = self._world_to_fast_map_px(point, width, height)
                cv2.circle(map_img, (px, py), 2, (0, 255, 0), -1)

        color_map = {
            "landmark_red": (0, 0, 255),
            "landmark_blue": (255, 0, 255),
            "landmark_green": (0, 180, 0),
            "landmark_yellow": (0, 220, 255),
        }
        for landmark_id, item in self.visual_landmarks_world.items():
            px, py = self._world_to_fast_map_px(item["pos"], width, height)
            cv2.circle(map_img, (px, py), 8, color_map.get(landmark_id, (200, 200, 200)), -1)

        cv2.putText(map_img, "Lidar Map", (10, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (40, 40, 40), 2, cv2.LINE_AA)
        cv2.putText(map_img, "cyan overlay = camera seen", (10, height - 12), cv2.FONT_HERSHEY_SIMPLEX, 0.48, (80, 80, 80), 1, cv2.LINE_AA)
        return map_img

    def _compose_fast_dashboard_frame(self, force=False):
        now = time.time()
        if not force and now - self._dashboard_last_draw_time < self._dashboard_min_interval:
            return self._fast_last_canvas
        perf_start = time.perf_counter()
        self._dashboard_last_draw_time = now

        if not self._ensure_dashboard_renderer_only():
            if self._fast_last_canvas is not None:
                return self._fast_last_canvas
            return np.zeros((540, 960, 3), dtype=np.uint8)

        if force or self._fast_scene_img is None or now - self._fast_last_scene_time >= self._fast_scene_interval:
            self._fast_scene_img = self._render_dashboard_view(camera=None)
            self._fast_last_scene_time = now

        scene_img = cv2.cvtColor(self._fast_scene_img, cv2.COLOR_RGB2BGR)
        camera_img = cv2.cvtColor(self.get_annotated_camera_image(), cv2.COLOR_RGB2BGR)

        canvas_h, canvas_w = 540, 960
        canvas = np.full((canvas_h, canvas_w, 3), 245, dtype=np.uint8)
        left_w = 590
        right_w = canvas_w - left_w
        top_h = canvas_h // 2

        scene_view = cv2.resize(scene_img, (left_w, canvas_h), interpolation=cv2.INTER_AREA)
        cam_view = cv2.resize(camera_img, (right_w, top_h), interpolation=cv2.INTER_AREA)
        map_view = self._render_fast_map(right_w, canvas_h - top_h)

        canvas[:, :left_w] = scene_view
        canvas[:top_h, left_w:] = cam_view
        canvas[top_h:, left_w:] = map_view
        cv2.line(canvas, (left_w, 0), (left_w, canvas_h), (60, 60, 60), 1)
        cv2.line(canvas, (left_w, top_h), (canvas_w, top_h), (60, 60, 60), 1)
        cv2.putText(canvas, "Simulation", (12, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2, cv2.LINE_AA)
        cv2.putText(canvas, "Robot Camera", (left_w + 10, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2, cv2.LINE_AA)
        if self._dashboard_log_lines:
            log_y = canvas_h - 64
            cv2.rectangle(canvas, (8, log_y - 18), (left_w - 8, canvas_h - 8), (20, 20, 20), -1)
            for i, line in enumerate(self._dashboard_log_lines[-2:]):
                ascii_line = line.encode("ascii", "ignore").decode("ascii")
                cv2.putText(canvas, ascii_line[-95:], (16, log_y + i * 24), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (230, 230, 230), 1, cv2.LINE_AA)

        elapsed = time.perf_counter() - perf_start
        self._perf_render_ema = elapsed if self._perf_render_ema == 0.0 else 0.9 * self._perf_render_ema + 0.1 * elapsed
        if now - self._perf_last_log_time > 5.0:
            self._perf_last_log_time = now
            self.append_dashboard_log(
                f"性能: fast dashboard {self._perf_render_ema * 1000:.0f}ms, RGB-D {self._perf_vision_ema * 1000:.0f}ms"
            )
        self._fast_last_canvas = canvas
        return canvas

    def get_fast_dashboard_frame_rgb(self, force=False):
        canvas = self._compose_fast_dashboard_frame(force=force)
        if canvas is None:
            canvas = np.zeros((720, 1280, 3), dtype=np.uint8)
        return cv2.cvtColor(canvas, cv2.COLOR_BGR2RGB)

    def _update_fast_dashboard(self, force=False):
        canvas = self._compose_fast_dashboard_frame(force=force)
        if canvas is None:
            return

        canvas_h, canvas_w = canvas.shape[:2]
        if not self._fast_dashboard_ready:
            cv2.namedWindow(self._fast_dashboard_window, cv2.WINDOW_NORMAL)
            cv2.resizeWindow(self._fast_dashboard_window, canvas_w, canvas_h)
            self._fast_dashboard_ready = True
        cv2.imshow(self._fast_dashboard_window, canvas)
        key = cv2.waitKey(1) & 0xFF
        if key in (27, ord("q")):
            raise KeyboardInterrupt
        try:
            if cv2.getWindowProperty(self._fast_dashboard_window, cv2.WND_PROP_VISIBLE) < 1:
                raise KeyboardInterrupt
        except cv2.error:
            pass

    def _ensure_dashboard_renderer_only(self):
        if self._dashboard_scene_camera is None:
            self._dashboard_scene_camera = mujoco.MjvCamera()
            self._dashboard_scene_camera.type = mujoco.mjtCamera.mjCAMERA_FREE
            self._dashboard_scene_camera.distance = self._dashboard_scene_distance
            self._dashboard_scene_camera.azimuth = 90.0
            self._dashboard_scene_camera.elevation = -89.0
            self._dashboard_scene_camera.lookat[:] = self._dashboard_scene_lookat
        if self._dashboard_renderer is None:
            render_h, render_w = self._dashboard_render_size
            try:
                self._dashboard_renderer = mujoco.Renderer(self.model, height=render_h, width=render_w)
            except Exception as exc:
                print(f"[Dashboard] MuJoCo offscreen renderer 初始化失败: {exc}")
                self._dashboard_renderer = False
        return self._dashboard_renderer is not False

    def _sync_mujoco_viewer(self):
        if not self.enable_mujoco_viewer:
            return
        now = time.time()
        if now - self._mujoco_viewer_last_sync < self._mujoco_viewer_min_interval:
            return
        try:
            if self._mujoco_viewer is None or not self._mujoco_viewer.is_running():
                self._mujoco_viewer = mujoco.viewer.launch_passive(self.model, self.data)
                if hasattr(self._mujoco_viewer, "opt"):
                    self._mujoco_viewer.opt.geomgroup[5] = 0
            self._mujoco_viewer.sync()
            self._mujoco_viewer_last_sync = now
        except Exception as exc:
            self.enable_mujoco_viewer = False
            self.append_dashboard_log(f"MuJoCo viewer 打开失败: {exc}")

    def render(self):
        if self.render_mode == 'dashboard':
            self._update_dashboard(force=False)
            self._sync_mujoco_viewer()
        elif self.render_mode == 'fast_dashboard':
            self._update_fast_dashboard(force=False)
        elif self.render_mode == 'qt_dashboard':
            return None
        elif hasattr(super(), 'render'):
            return super().render()

    def close(self):
        if self._mujoco_viewer is not None:
            try:
                if self._mujoco_viewer.is_running():
                    self._mujoco_viewer.close()
            except Exception:
                pass
            self._mujoco_viewer = None
        for renderer_name in ("_vision_rgb_renderer", "_vision_depth_renderer"):
            renderer = getattr(self, renderer_name, None)
            if renderer not in (None, False):
                try:
                    renderer.close()
                except Exception:
                    pass
                setattr(self, renderer_name, None)
        if self._dashboard_renderer not in (None, False):
            try:
                self._dashboard_renderer.close()
            except Exception:
                pass
            self._dashboard_renderer = None
        if self._dashboard_fig is not None:
            try:
                import matplotlib.pyplot as plt
                if plt.fignum_exists(self._dashboard_fig.number):
                    plt.close(self._dashboard_fig)
            except Exception:
                pass
            self._dashboard_fig = None
        if self._fast_dashboard_ready:
            try:
                cv2.destroyWindow(self._fast_dashboard_window)
            except Exception:
                pass
            self._fast_dashboard_ready = False
        super().close()

    def _get_obs(self):
        base_obs = super()._get_obs()
        if isinstance(base_obs, dict) and "state_history" in base_obs:
            state_history = base_obs["state_history"]
        else:
            state_history = np.concatenate(list(self.state_history_buffer)).astype(np.float32)
        
        robot_pos_2d, robot_yaw = self._get_robot_pose()
        c, s = np.cos(-robot_yaw), np.sin(-robot_yaw)
        rotation_matrix = np.array([[c, -s],[s, c]])
        world_aligned_local_coords = self.obs_local_coords_base @ rotation_matrix.T
        
        grid_x_center = (robot_pos_2d[0] + self.grid_map.world_origin_offset_m[0]) * self.grid_map.resolution
        grid_y_center_inverted = self.grid_map.num_cells_world - 1 - (robot_pos_2d[1] + self.grid_map.world_origin_offset_m[1]) * self.grid_map.resolution
        
        sampling_cols = grid_x_center + world_aligned_local_coords[:, 0]
        sampling_rows = grid_y_center_inverted - world_aligned_local_coords[:, 1] 
        sampling_coords = np.stack([sampling_rows, sampling_cols])

        local_map_flat_log_odds = map_coordinates(
            self.grid_map.grid, sampling_coords, order=0, cval=0.0, prefilter=False 
        )
        local_map_prob = 1.0 - 1.0 / (1.0 + np.exp(local_map_flat_log_odds))
        local_map_prob = local_map_prob.reshape((self.obs_num_cells_local, self.obs_num_cells_local))
        
        if hasattr(self.grid_map, 'visited_grid'):
            local_visited_flat = map_coordinates(
                self.grid_map.visited_grid, sampling_coords, order=0, cval=0.0, prefilter=False
            )
            local_visited_map = local_visited_flat.reshape((self.obs_num_cells_local, self.obs_num_cells_local))
        else:
            local_visited_map = np.zeros((self.obs_num_cells_local, self.obs_num_cells_local), dtype=np.float32)
        
        local_grid_map = np.stack([local_map_prob, local_visited_map], axis=0)
        local_grid_map_uint8 = (local_grid_map * 255.0).astype(np.uint8)
        
        return {
            "grid_map": local_grid_map_uint8, 
            "state_history": state_history
        }
