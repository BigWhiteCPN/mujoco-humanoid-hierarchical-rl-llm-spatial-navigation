# --- START OF FILE skills.py ---
import numpy as np
import mujoco
import threading
import time
from contextlib import nullcontext
from scipy.spatial.transform import Rotation as R
from stable_baselines3 import SAC
from scipy.ndimage import binary_dilation

import torch
import torch.nn as nn
import gymnasium as gym
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
from sklearn.cluster import DBSCAN

# ==============================================================================
# 🌟 核心：为了加载新策略所需的自定义网络结构 (CNN + GRU + ReLU对齐)
# ==============================================================================
class SequenceFusionExtractor(BaseFeaturesExtractor):
    def __init__(self, observation_space: gym.spaces.Dict,
                 state_feature_dim=9,
                 history_length=15,
                 d_model=128):
        features_dim = 256
        super().__init__(observation_space, features_dim)

        # 1. 栅格地图处理 CNN
        n_input_channels = observation_space['grid_map'].shape[0] 
        
        self.cnn_body = nn.Sequential(
            nn.Conv2d(n_input_channels, 16, kernel_size=5, stride=2, padding=2),
            nn.ReLU(),
            nn.Conv2d(16, 32, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            nn.Flatten(),
        )
        
        with torch.no_grad():
            sample_input = torch.zeros(1, *observation_space['grid_map'].shape)
            n_flatten = self.cnn_body(sample_input).shape[1]
            
        self.map_cnn = nn.Sequential(
            self.cnn_body,
            nn.Linear(n_flatten, 128),
            nn.LayerNorm(128),
            nn.ReLU()
        )

        # 2. 状态序列处理 GRU
        self.state_sub_dim = state_feature_dim
        self.seq_len = history_length
        
        self.state_embedding = nn.Linear(state_feature_dim, d_model)
        self.gru = nn.GRU(input_size=d_model, hidden_size=d_model, num_layers=2, batch_first=True)
        
        # 3. 融合层
        self.fusion_layer = nn.Sequential(
            nn.Linear(128 + d_model, features_dim),
            nn.LayerNorm(features_dim),
            nn.ReLU()
        )

    def forward(self, observations):
        # 必须转成 float 并除以 255.0，与训练环境严格对齐
        grid_map = observations['grid_map'].float() / 255.0 
        map_feat = self.map_cnn(grid_map)
        
        batch_size = observations['state_history'].shape[0]
        state_seq = observations['state_history'].view(batch_size, self.seq_len, self.state_sub_dim)
        
        x = self.state_embedding(state_seq)
        x = torch.relu(x)
        gru_out, _ = self.gru(x)
        temporal_feat = gru_out[:, -1, :] 
        
        combined = torch.cat([map_feat, temporal_feat], dim=1)
        output = self.fusion_layer(combined)
        return output

# ==============================================================================
# 心智模拟器 (Mental Simulator)
# ==============================================================================
class MentalSimulator:
    """心智模拟器：复用环境路径规划器，在执行物理导航前预演路径"""
    def __init__(self, env, robot_radius_m=0.35):
        self.env = env
        self.robot_radius_m = robot_radius_m

    def simulate_path(self, start_pos, goal_pos):
        if not hasattr(self.env, 'grid_map') or not hasattr(self.env, 'path_planner'):
            return {"feasible": False, "reason": "我还未建立地图空间感知"}

        path = self.env.path_planner.find_path(np.asarray(start_pos), np.asarray(goal_pos))
        if path is None or len(path) < 2:
            return {"feasible": False, "reason": "环境路径规划器无法找到连通路径"}

        path = np.asarray(path, dtype=np.float32)
        segment_lengths = np.linalg.norm(np.diff(path, axis=0), axis=1)
        path_length_m = float(np.sum(segment_lengths))
        return {
            "feasible": True,
            "path_length_m": path_length_m,
            "estimated_steps": int(path_length_m / 0.5),
            "reason": "路径畅通"
        }


# ==============================================================================
# 技能模块：Perception & Navigation
# ==============================================================================
class PerceptionSkill:
    def __init__(self, env, memory):
        self.env = env
        self.memory = memory

    def _request_qt_visual_detection(self, timeout_s=0.8, wait=True):
        now = time.time()
        min_interval = float(getattr(self.env, "_command_vision_min_interval", 1.0))
        last_detect = float(getattr(self.env, "_vision_last_detect_time", 0.0))
        if (
            getattr(self.env, "_vision_last_annotated_rgb", None) is not None
            and now - last_detect < min_interval
        ):
            return list(getattr(self.env, "visual_detections", []))

        if getattr(self.env, "_qt_force_vision_requested", False):
            return list(getattr(self.env, "visual_detections", []))

        request_time = time.time()
        self.env._qt_force_vision_request_time = request_time
        self.env._qt_force_vision_waiting = bool(wait)
        self.env._qt_force_vision_requested = True

        progress_callback = getattr(self.env, "_qt_progress_callback", None)
        if progress_callback is not None:
            progress_callback(force=True)

        if not wait:
            return list(getattr(self.env, "visual_detections", []))

        deadline = time.time() + timeout_s
        while time.time() < deadline:
            if getattr(self.env, "_shutdown_requested", False):
                break
            if getattr(self.env, "_qt_force_vision_done_time", 0.0) >= request_time:
                break
            time.sleep(0.02)
        return list(getattr(self.env, "visual_detections", []))

    def _has_line_of_sight(self, start_pos_2d, target_pos_2d):
        start_3d = np.array([start_pos_2d[0], start_pos_2d[1], 0.3])
        target_3d = np.array([target_pos_2d[0], target_pos_2d[1], 0.3])

        vec = target_3d - start_3d
        dist = np.linalg.norm(vec)

        if dist < 0.1: return True

        vec_norm = vec / dist
        geomgroup = np.array([1, 1, 1, 1, 1, 1], dtype=np.uint8)
        flg_static = 1
        body_exclude = int(self.env.robot_base_body_id)
        geomid_out = np.zeros(1, dtype=np.int32)

        hit_dist = mujoco.mj_ray(
            self.env.model, self.env.data,
            start_3d, vec_norm,
            geomgroup, flg_static, body_exclude, geomid_out
        )

        if hit_dist != -1 and hit_dist < (dist - 0.4): return False
        return True

    def scan_and_remember(self, wait_for_qt=True):
        detected = []
        if not hasattr(self.env, "detect_visual_landmarks"):
            return detected

        qt_render_thread_id = getattr(self.env, "_qt_render_thread_id", None)
        if qt_render_thread_id is not None and threading.get_ident() != qt_render_thread_id:
            visual_detections = self._request_qt_visual_detection(wait=wait_for_qt)
        else:
            now = time.time()
            min_interval = float(getattr(self.env, "_command_vision_min_interval", 1.0))
            last_detect = float(getattr(self.env, "_vision_last_detect_time", 0.0))
            if (
                getattr(self.env, "_vision_last_annotated_rgb", None) is not None
                and now - last_detect < min_interval
            ):
                visual_detections = list(getattr(self.env, "visual_detections", []))
            else:
                visual_detections = self.env.detect_visual_landmarks(force=True)
        for item in visual_detections:
            landmark_id = item["id"]
            lm_pos = np.asarray(item["pos"], dtype=np.float32)
            dist = float(item.get("dist", np.linalg.norm(lm_pos - self.env.data.xpos[self.env.robot_base_body_id][:2])))
            was_known = landmark_id in self.memory.memory_db
            self.memory.add_memory(landmark_id, lm_pos[0], lm_pos[1], confidence=0.8)
            if not was_known and hasattr(self.env, "append_dashboard_log"):
                meaning = self.memory.feature_meaning.get(landmark_id, landmark_id)
                self.env.append_dashboard_log(
                    f"视觉发现: {meaning}，距离约 {dist:.1f}m，坐标 ({lm_pos[0]:.1f},{lm_pos[1]:.1f})"
                )
            detected.append({'id': landmark_id, 'pos': lm_pos, 'dist': dist})

        return detected

class NavigationSkill:
    def __init__(self, env, sac_model_path):
        self.env = env
        print(f"[导航] 准备加载包含 GRU 记忆的全新 SAC 模型: {sac_model_path}")
        
        # 挂载自定义模型架构配置，解决找不到类的问题
        device = "cuda" if torch.cuda.is_available() else "cpu"
        custom_objects = {
            "SequenceFusionExtractor": SequenceFusionExtractor,
            "learning_rate": 0.0, 
            "lr_schedule": lambda _: 0.0,
            "clip_range": lambda _: 0.0,
        }
        
        self.sac_model = SAC.load(
            sac_model_path, 
            env=env, 
            device=device, 
            custom_objects=custom_objects
        )
        print("[导航] SAC 模型加载完成！")

    def _validate_goal(self, target_x, target_y):
        """检查目标点是否在墙里，如果是则挪到最近的空地"""
        grid_map = self.env.grid_map
        prob_grid = 1.0 - 1.0 / (1.0 + np.exp(grid_map.grid))
        c = int((target_x + grid_map.world_origin_offset_m[0]) * grid_map.resolution)
        r = grid_map.num_cells_world - 1 - int((target_y + grid_map.world_origin_offset_m[1]) * grid_map.resolution)
        r = np.clip(r, 0, grid_map.num_cells_world - 1)
        c = np.clip(c, 0, grid_map.num_cells_world - 1)
        if prob_grid[r, c] < 0.65:
            return target_x, target_y
        # 目标在障碍物里，搜索最近的空地
        for radius in range(1, 15):
            for dr in range(-radius, radius + 1):
                for dc in range(-radius, radius + 1):
                    if abs(dr) != radius and abs(dc) != radius: continue
                    nr, nc = r + dr, c + dc
                    if 0 <= nr < grid_map.num_cells_world and 0 <= nc < grid_map.num_cells_world:
                        if prob_grid[nr, nc] < 0.4:
                            wx = nc / grid_map.resolution - grid_map.world_origin_offset_m[0]
                            wy = (grid_map.num_cells_world - 1 - nr) / grid_map.resolution - grid_map.world_origin_offset_m[1]
                            print(f"  [导航] 目标 ({target_x:.1f},{target_y:.1f}) 在墙内，调整到 ({wx:.1f},{wy:.1f})")
                            return wx, wy
        return target_x, target_y

    def go_to(
        self,
        target_x,
        target_y,
        max_steps=8000,
        success_dist=1.0,
        step_callback=None,
        callback_freq=50,
        track_landmark_id=None,
    ):
        lock = getattr(self.env, "_qt_env_lock", None)
        lock_context = lock if lock is not None else nullcontext()
        realtime_pacing = lock is not None
        try:
            step_dt = (
                float(self.env.action_repeat)
                * float(self.env.locomotion_controller.cfg.sim_config.decimation)
                * float(self.env.model.opt.timestep)
            )
        except Exception:
            step_dt = 1.0 / 25.0
        step_dt = float(np.clip(step_dt, 1.0 / 40.0, 1.0 / 15.0))
        next_step_time = time.perf_counter()
        progress_callback = getattr(self.env, "_qt_progress_callback", None)
        debug_timing = bool(getattr(self.env, "debug_timing", False))
        debug_last_print_time = 0.0
        debug_prev_robot_pos = None

        def debug_print(message, force=False):
            nonlocal debug_last_print_time
            if not debug_timing:
                return
            now_dbg = time.perf_counter()
            if force or now_dbg - debug_last_print_time >= 0.25:
                print(message, flush=True)
                debug_last_print_time = now_dbg

        def pace_step():
            nonlocal next_step_time
            if progress_callback is not None:
                progress_callback()
            if not realtime_pacing:
                return
            next_step_time += step_dt
            sleep_dt = next_step_time - time.perf_counter()
            if sleep_dt > 0:
                time.sleep(sleep_dt)
            elif sleep_dt < -step_dt:
                next_step_time = time.perf_counter()
                time.sleep(0.001)
            else:
                time.sleep(0.001)

        with lock_context:
            target_x, target_y = self._validate_goal(target_x, target_y)
            self.env.set_agent_goal(target_x, target_y)
            obs = self.env._get_obs()
        goal = np.array([target_x, target_y])
        last_tracked_goal_update = 0.0
        last_tracked_log_time = 0.0

        # 如果路径规划用了兜底（沿朝向走1.5m），说明地图不够，不盲目导航
        if getattr(self.env, '_path_fallback', False):
            with lock_context:
                robot_pos = self.env.data.xpos[self.env.robot_base_body_id][:2].copy()
            dist = np.linalg.norm(robot_pos - goal)
            print(f"  -> 路径规划失败（地图未充分探索），跳过导航 dist={dist:.2f}")
            return False, dist

        # 撞墙脱困：记录历史位置，检测卡住
        pos_history = []
        stuck_threshold = 0.12  # 窗口内移动不到 0.12m 才判定为真正卡住，减少慢速转弯误判。
        stuck_window = 75
        stuck_count = 0
        max_stuck_recoveries = 3

        for step in range(max_steps):
            if getattr(self.env, "_shutdown_requested", False):
                with lock_context:
                    robot_pos = self.env.data.xpos[self.env.robot_base_body_id][:2].copy()
                dist = float(np.linalg.norm(robot_pos - goal))
                print(f"  -> 收到停止请求，中断导航 dist={dist:.2f}")
                return False, dist
            loop_t0 = time.perf_counter()
            pred_t0 = time.perf_counter()
            action, _ = self.sac_model.predict(obs, deterministic=True)
            pred_ms = (time.perf_counter() - pred_t0) * 1000.0

            lock_wait_ms = 0.0
            step_t0 = time.perf_counter()
            if lock is not None:
                lock_t0 = time.perf_counter()
                lock.acquire()
                lock_wait_ms = (time.perf_counter() - lock_t0) * 1000.0
                try:
                    env_step_t0 = time.perf_counter()
                    obs, reward, terminated, truncated, info = self.env.step(action)
                    env_step_ms = (time.perf_counter() - env_step_t0) * 1000.0
                finally:
                    lock.release()
            else:
                env_step_t0 = time.perf_counter()
                obs, reward, terminated, truncated, info = self.env.step(action)
                env_step_ms = (time.perf_counter() - env_step_t0) * 1000.0
            step_block_ms = (time.perf_counter() - step_t0) * 1000.0

            if progress_callback is not None and step % max(1, callback_freq // 2) == 0:
                progress_callback()

            callback_ms = 0.0
            if step_callback is not None and step % callback_freq == 0:
                cb_t0 = time.perf_counter()
                stop_early = step_callback()
                callback_ms = (time.perf_counter() - cb_t0) * 1000.0
                if stop_early:
                    print("  -> 触发新发现！立即中断当前盲目导航！")
                    with lock_context:
                        robot_pos = self.env.data.xpos[self.env.robot_base_body_id][:2].copy()
                    return True, np.linalg.norm(robot_pos - goal)

            with lock_context:
                robot_pos = self.env.data.xpos[self.env.robot_base_body_id][:2].copy()
            dist = np.linalg.norm(robot_pos - goal)
            pos_delta = 0.0
            if debug_prev_robot_pos is not None:
                pos_delta = float(np.linalg.norm(robot_pos - debug_prev_robot_pos))
            debug_prev_robot_pos = robot_pos.copy()

            if track_landmark_id is not None:
                now = time.time()
                landmark_item = getattr(self.env, "visual_landmarks_world", {}).get(track_landmark_id)
                if landmark_item is not None and "pos" in landmark_item and now - last_tracked_goal_update > 0.8:
                    live_goal = np.asarray(landmark_item["pos"], dtype=np.float32)
                    if np.linalg.norm(live_goal - goal) > 0.45:
                        with lock_context:
                            live_x, live_y = self._validate_goal(float(live_goal[0]), float(live_goal[1]))
                            self.env.set_agent_goal(live_x, live_y)
                            obs = self.env._get_obs()
                        goal = np.array([live_x, live_y], dtype=np.float32)
                        last_tracked_goal_update = now
                        dist = np.linalg.norm(robot_pos - goal)
                        if hasattr(self.env, "append_dashboard_log") and now - last_tracked_log_time > 2.0:
                            self.env.append_dashboard_log(
                                f"导航: 根据实时视觉修正目标到 ({goal[0]:.1f},{goal[1]:.1f})"
                            )
                            last_tracked_log_time = now

            if dist < success_dist:
                if track_landmark_id is not None:
                    landmark_item = getattr(self.env, "visual_landmarks_world", {}).get(track_landmark_id)
                    if landmark_item is not None and "pos" in landmark_item:
                        live_goal = np.asarray(landmark_item["pos"], dtype=np.float32)
                        live_dist = np.linalg.norm(robot_pos - live_goal)
                        if live_dist > success_dist + 0.25:
                            goal = live_goal.copy()
                            with lock_context:
                                live_x, live_y = self._validate_goal(float(goal[0]), float(goal[1]))
                                self.env.set_agent_goal(live_x, live_y)
                                obs = self.env._get_obs()
                            goal = np.array([live_x, live_y], dtype=np.float32)
                            dist = np.linalg.norm(robot_pos - goal)
                            print(f"  -> 视觉目标仍在前方，继续靠近 live_dist={live_dist:.2f}")
                            continue
                print(f"  -> 到达目标! dist={dist:.2f}, steps={step}")
                return True, dist

            if terminated or truncated:
                print(f"  -> 导航中断 terminated={terminated} truncated={truncated}, dist={dist:.2f}")
                return False, dist

            loop_ms = (time.perf_counter() - loop_t0) * 1000.0
            if (
                debug_timing
                and (
                    loop_ms > 70.0
                    or env_step_ms > 45.0
                    or lock_wait_ms > 20.0
                    or callback_ms > 30.0
                    or pred_ms > 20.0
                    or pos_delta > 0.18
                )
            ):
                tag = "[JUMP NAV]" if pos_delta > 0.18 else "[PERF NAV]"
                debug_print(
                    f"{tag} step={step} loop={loop_ms:.1f}ms pred={pred_ms:.1f}ms "
                    f"lock_wait={lock_wait_ms:.1f}ms env_step={env_step_ms:.1f}ms "
                    f"step_block={step_block_ms:.1f}ms callback={callback_ms:.1f}ms "
                    f"pos_delta={pos_delta:.3f}m dist={dist:.2f}",
                    force=pos_delta > 0.18,
                )

            # 撞墙脱困逻辑
            pos_history.append(robot_pos.copy())
            if len(pos_history) > stuck_window:
                pos_history.pop(0)

            if len(pos_history) == stuck_window:
                moved = np.linalg.norm(pos_history[-1] - pos_history[0])
                if moved >= stuck_threshold:
                    stuck_count = 0  # 正常移动，重置脱困计数
                else:
                    stuck_count += 1
                    if stuck_count > max_stuck_recoveries:
                        print(f"  -> 脱困失败 {max_stuck_recoveries} 次，放弃导航, dist={dist:.2f}")
                        return False, dist

                    print(f"  -> 检测到卡住 (移动{moved:.2f}m)，温和脱困 #{stuck_count}...")
                    recovery_action = np.zeros_like(action, dtype=np.float32)
                    recovery_action[0] = -0.12
                    if recovery_action.shape[0] >= 3:
                        recovery_action[2] = 0.18 if stuck_count % 2 else -0.18
                    for _ in range(4):
                        with lock_context:
                            obs, _, term, trunc, _ = self.env.step(recovery_action)
                        pace_step()
                        if term or trunc:
                            break
                    with lock_context:
                        self.env._update_path()
                    pos_history.clear()
                    continue

            pace_step()

        print(f"  -> 导航超时, dist={dist:.2f}")
        return False, dist


# ==============================================================================
# 技能模块：Frontier 主动探索 (🌟 融入主动推断与香农熵的全新数学版本)
# ==============================================================================
class FrontierExplorationSkill:
    def __init__(self, env, nav_skill, percept_skill, topo_map=None):
        self.env = env
        self.nav_skill = nav_skill
        self.percept_skill = percept_skill
        self.topo_map = topo_map
        self.visited_frontiers = []
        self.failed_frontiers =[]
        self.trajectory_history =[]
        self._trajectory_maxlen = 5000
        self._has_initial_scanned = False
        self._last_frontier_center = None
        self._visual_revisit_attempts = {}
        self._path_cache = {}
        self._entropy_cache = None
        self._entropy_cache_gen = -1
        self._visited_frontier_keys = set()
        self._failed_frontier_keys = set()
        self._local_frontier_path_horizon_m = 6.0
        self._local_frontier_path_slack_m = 3.0
        self._near_frontier_path_slack_m = 1.4
        self._near_frontier_far_slack_m = 2.2
        self._near_frontier_tier_threshold_m = 5.0
        self._detour_ratio_soft_limit = 1.8
        self._detour_ratio_prefer_limit = 2.4
        self._local_completion_announced_key = None

    def _visual_sweep_and_remember(self, turns=6):
        detected_total = []
        lock = getattr(self.env, "_qt_env_lock", None)
        lock_context = lock if lock is not None else nullcontext()
        progress_callback = getattr(self.env, "_qt_progress_callback", None)
        action = np.zeros(self.env.action_space.shape, dtype=np.float32)
        if action.shape[0] >= 3:
            action[2] = 0.22
        for _ in range(turns):
            if getattr(self.env, "_shutdown_requested", False):
                break
            with lock_context:
                self.env.step(action)
            if progress_callback is not None:
                progress_callback()
                time.sleep(0.08)
            detected_total.extend(self.percept_skill.scan_and_remember())
            time.sleep(0.04)
        detected_total.extend(self.percept_skill.scan_and_remember())
        unique = {}
        for item in detected_total:
            unique[item["id"]] = item
        return list(unique.values())

    def _world_to_grid(self, world_pos):
        grid_map = self.env.grid_map
        c = int((world_pos[0] + grid_map.world_origin_offset_m[0]) * grid_map.resolution)
        r = grid_map.num_cells_world - 1 - int((world_pos[1] + grid_map.world_origin_offset_m[1]) * grid_map.resolution)
        return r, c

    def _grid_to_world(self, r, c):
        grid_map = self.env.grid_map
        y_grid = grid_map.num_cells_world - 1 - r
        x_world = c / grid_map.resolution - grid_map.world_origin_offset_m[0]
        y_world = y_grid / grid_map.resolution - grid_map.world_origin_offset_m[1]
        return np.array([x_world, y_world])

    def _get_frontiers(self):
        if not hasattr(self.env, 'grid_map'): return[]
        grid_map = self.env.grid_map
        prob_grid = 1.0 - 1.0 / (1.0 + np.exp(grid_map.grid))

        free_mask = prob_grid < 0.4
        unknown_mask = (prob_grid >= 0.4) & (prob_grid <= 0.6)
        occupied_mask = prob_grid > 0.7

        dilated_free = binary_dilation(free_mask, iterations=1)
        free_border = dilated_free & ~free_mask
        frontier_mask = free_border & unknown_mask

        dilated_occupied = binary_dilation(occupied_mask, iterations=2)
        frontier_mask_safe = frontier_mask & ~dilated_occupied
        frontier_indices = np.argwhere(frontier_mask_safe)

        if len(frontier_indices) == 0:
            dilated_occupied_1 = binary_dilation(occupied_mask, iterations=1)
            frontier_mask_fallback = frontier_mask & ~dilated_occupied_1
            frontier_indices = np.argwhere(frontier_mask_fallback)

        if len(frontier_indices) == 0: frontier_indices = np.argwhere(frontier_mask)
        if len(frontier_indices) == 0: return []

        r_arr = frontier_indices[:, 0]
        c_arr = frontier_indices[:, 1]
        grid_map = self.env.grid_map
        border_margin = max(1, int(0.8 * grid_map.resolution))
        inside_border = (
            (r_arr >= border_margin)
            & (r_arr < grid_map.num_cells_world - border_margin)
            & (c_arr >= border_margin)
            & (c_arr < grid_map.num_cells_world - border_margin)
        )
        r_arr = r_arr[inside_border]
        c_arr = c_arr[inside_border]
        if len(r_arr) == 0:
            return []
        y_grid = grid_map.num_cells_world - 1 - r_arr
        x_world = c_arr / grid_map.resolution - grid_map.world_origin_offset_m[0]
        y_world = y_grid / grid_map.resolution - grid_map.world_origin_offset_m[1]
        return np.column_stack([x_world, y_world])

    def _cluster_frontiers(self, frontier_points, cluster_dist=2.5):
        if len(frontier_points) == 0: return []
        if len(frontier_points) < 2:
            return [{'center': frontier_points[0], 'size': 1, 'points': frontier_points}]

        db = DBSCAN(eps=cluster_dist, min_samples=3).fit(frontier_points)
        clusters = []
        for label_id in set(db.labels_):
            if label_id == -1: continue
            mask = db.labels_ == label_id
            cluster_pts = frontier_points[mask]
            mean_pt = cluster_pts.mean(axis=0)
            dists = np.linalg.norm(cluster_pts - mean_pt, axis=1)
            mid_point = cluster_pts[np.argmin(dists)]
            clusters.append({'center': mid_point, 'size': len(cluster_pts), 'points': cluster_pts})
        clusters.sort(key=lambda c: c['size'], reverse=True)
        return clusters

    def _frontier_key(self, point, cell_size=2.5):
        """将坐标量化为网格 cell key，用于 O(1) 回访检测"""
        return (int(round(float(point[0]) / cell_size)), int(round(float(point[1]) / cell_size)))

    def _clear_frontier_history(self):
        """清空已访问/失败前沿记录（列表 + 哈希集合）"""
        self.visited_frontiers.clear()
        self.failed_frontiers.clear()
        self._visited_frontier_keys.clear()
        self._failed_frontier_keys.clear()

    def _is_frontier_visited(self, center, threshold=1.2):
        center = np.asarray(center, dtype=np.float32)
        if self.visited_frontiers:
            visited = np.asarray(self.visited_frontiers, dtype=np.float32)
            if np.any(np.linalg.norm(visited - center, axis=1) < threshold):
                return True
        if self.failed_frontiers:
            failed = np.asarray(self.failed_frontiers, dtype=np.float32)
            if np.any(np.linalg.norm(failed - center, axis=1) < threshold * 1.4):
                return True

        key = self._frontier_key(center, cell_size=threshold)
        if key in self._visited_frontier_keys or key in self._failed_frontier_keys:
            return True
        return False

    def _compute_nav_target(self, frontier_center):
        if not hasattr(self.env, 'grid_map'): return frontier_center
        grid_map = self.env.grid_map
        prob_grid = 1.0 - 1.0 / (1.0 + np.exp(grid_map.grid))
        free_mask = prob_grid < 0.4
        robot_pos = self.env.data.xpos[self.env.robot_base_body_id][:2].copy()

        direction = robot_pos - frontier_center
        dist = np.linalg.norm(direction)
        if dist < 0.1: return frontier_center

        direction_norm = direction / dist
        for step_dist in np.arange(0.4, min(1.5, dist) + 0.1, 0.2):
            candidate = frontier_center + direction_norm * step_dist
            r, c = self._world_to_grid(candidate)
            if not (0 <= r < grid_map.num_cells_world and 0 <= c < grid_map.num_cells_world): continue
            if free_mask[r, c]:
                r_min, r_max = max(0, r - 1), min(grid_map.num_cells_world, r + 2)
                c_min, c_max = max(0, c - 1), min(grid_map.num_cells_world, c + 2)
                if np.mean(free_mask[r_min:r_max, c_min:c_max]) > 0.5: return candidate
        return frontier_center + direction_norm * 0.3

    def _has_grid_line_of_sight(self, start_pos, target_pos, max_prob=0.68):
        if not hasattr(self.env, 'grid_map'):
            return True
        grid_map = self.env.grid_map
        prob_grid = 1.0 - 1.0 / (1.0 + np.exp(grid_map.grid))
        start_pos = np.asarray(start_pos, dtype=np.float32)
        target_pos = np.asarray(target_pos, dtype=np.float32)
        dist = float(np.linalg.norm(target_pos - start_pos))
        if dist < 0.15:
            return True
        n_samples = max(3, int(dist * grid_map.resolution * 1.5))
        for t in np.linspace(0.0, 1.0, n_samples):
            point = start_pos * (1.0 - t) + target_pos * t
            r, c = self._world_to_grid(point)
            if not (0 <= r < grid_map.num_cells_world and 0 <= c < grid_map.num_cells_world):
                return False
            if prob_grid[r, c] > max_prob:
                return False
        return True

    def _frontier_target_has_access(self, nav_candidate, frontier_center):
        nav_candidate = np.asarray(nav_candidate, dtype=np.float32)
        frontier_center = np.asarray(frontier_center, dtype=np.float32)
        dist = float(np.linalg.norm(frontier_center - nav_candidate))
        if dist < 0.15:
            return True

        start_3d = np.array([nav_candidate[0], nav_candidate[1], 0.45], dtype=np.float64)
        target_3d = np.array([frontier_center[0], frontier_center[1], 0.45], dtype=np.float64)
        direction = target_3d - start_3d
        direction_norm = direction / max(np.linalg.norm(direction), 1e-6)

        try:
            geomgroup = np.array([1, 1, 1, 1, 1, 1], dtype=np.uint8)
            geomid_out = np.zeros(1, dtype=np.int32)
            body_exclude = int(getattr(self.env, "robot_base_body_id", -1))
            hit_dist = mujoco.mj_ray(
                self.env.model,
                self.env.data,
                start_3d,
                direction_norm,
                geomgroup,
                1,
                body_exclude,
                geomid_out,
            )
            if hit_dist != -1 and hit_dist < dist - 0.20:
                return False
        except Exception:
            pass

        return self._has_grid_line_of_sight(nav_candidate, frontier_center)

    def _is_in_explored_corridor(self, point, threshold_ratio=0.12):
        if not hasattr(self.env, 'grid_map'): return False
        grid_map = self.env.grid_map
        prob_grid = 1.0 - 1.0 / (1.0 + np.exp(grid_map.grid))
        unknown_mask = (prob_grid >= 0.4) & (prob_grid <= 0.6)

        r_center, c_center = self._world_to_grid(point)
        check_radius = int(2.0 * grid_map.resolution)
        r_min, r_max = max(0, r_center - check_radius), min(grid_map.num_cells_world, r_center + check_radius)
        c_min, c_max = max(0, c_center - check_radius), min(grid_map.num_cells_world, c_center + check_radius)

        total_cells = (r_max - r_min) * (c_max - c_min)
        if total_cells == 0: return False
        return (np.sum(unknown_mask[r_min:r_max, c_min:c_max]) / total_cells) < threshold_ratio

    def _estimate_local_exploration_status(self, robot_pos, frontier_points, radius_m=4.0):
        if not hasattr(self.env, 'grid_map'):
            return False, {}

        grid_map = self.env.grid_map
        prob_grid = 1.0 - 1.0 / (1.0 + np.exp(grid_map.grid))
        unknown_mask = (prob_grid >= 0.4) & (prob_grid <= 0.6)
        occupied_mask = prob_grid > 0.7

        r_center, c_center = self._world_to_grid(robot_pos)
        r_px = max(1, int(radius_m * grid_map.resolution))
        r_min, r_max = max(0, r_center - r_px), min(grid_map.num_cells_world, r_center + r_px + 1)
        c_min, c_max = max(0, c_center - r_px), min(grid_map.num_cells_world, c_center + r_px + 1)

        yy, xx = np.ogrid[r_min:r_max, c_min:c_max]
        disk_mask = (yy - r_center) ** 2 + (xx - c_center) ** 2 <= r_px ** 2
        traversable_context = disk_mask & ~occupied_mask[r_min:r_max, c_min:c_max]
        context_cells = int(np.sum(traversable_context))
        unknown_cells = int(np.sum(unknown_mask[r_min:r_max, c_min:c_max] & traversable_context))
        unknown_ratio = unknown_cells / max(context_cells, 1)

        if len(frontier_points) > 0:
            local_frontier_count = int(np.sum(np.linalg.norm(frontier_points - robot_pos, axis=1) <= radius_m))
        else:
            local_frontier_count = 0

        done = local_frontier_count <= 2 and unknown_ratio < 0.08
        stats = {
            'unknown_ratio': unknown_ratio,
            'local_frontier_count': local_frontier_count,
            'unknown_cells': unknown_cells,
            'context_cells': context_cells,
        }
        return done, stats

    def _record_trajectory(self):
        self.trajectory_history.append(self.env.data.xpos[self.env.robot_base_body_id][:2].copy())
        if len(self.trajectory_history) > self._trajectory_maxlen:
            self.trajectory_history = self.trajectory_history[-self._trajectory_maxlen:]

    def _trajectory_density_at(self, point, radius=2.5):
        if len(self.trajectory_history) == 0: return 0
        distances = np.linalg.norm(np.array(self.trajectory_history) - point, axis=1)
        return int(np.sum(distances < radius))

    def _initial_scan(self):
        print("[探索] 原地扫描中，建立初始地图...")
        robot_pos = self.env.data.xpos[self.env.robot_base_body_id][:2].copy()
        for offset in [[1.0, 0], [-1.0, 0],[0, 1.0],[0, -1.0]]:
            if getattr(self.env, "_shutdown_requested", False):
                return
            target = np.clip(robot_pos + np.array(offset), -9.0, 9.0)
            self.nav_skill.go_to(target[0], target[1], max_steps=100, success_dist=0.8)
            self.percept_skill.scan_and_remember()
        # 拓扑地图：记录起始位置
        if self.topo_map is not None:
            robot_pos_now = self.env.data.xpos[self.env.robot_base_body_id][:2].copy()
            detected = self._visual_sweep_and_remember(turns=4)
            lm_ids = [d['id'] for d in detected] if detected else []
            self.topo_map.visit(self.env.grid_map, robot_pos_now, landmarks=lm_ids, visual_scan=True)
        print("[探索] 初始扫描完成，开始 Frontier 探索。")

    @staticmethod
    def _parse_direction_hint(hint):
        if not hint: return None
        hint = hint.strip().lower()
        d_map = {
            "north":[0.0, 1.0], "south": [0.0, -1.0], "east":[1.0, 0.0], "west": [-1.0, 0.0],
            "northeast":[0.707, 0.707], "northwest":[-0.707, 0.707], 
            "southeast":[0.707, -0.707], "southwest": [-0.707, -0.707],
            "北":[0.0, 1.0], "南":[0.0, -1.0], "东":[1.0, 0.0], "西":[-1.0, 0.0]
        }
        for k, v in d_map.items():
            if k in hint: return np.array(v)
        return None

    def _compute_entropy_map(self):
        """计算贝叶斯概率地图的香农熵分布 (Shannon Entropy)，带缓存"""
        if not hasattr(self.env, 'grid_map'): return None
        grid = self.env.grid_map.grid
        # 地图会在探索过程中原地更新；用轻量级统计量做脏检测，避免复用旧熵图。
        grid_id = (id(grid), grid.shape, float(np.sum(grid)), float(np.mean(grid)))
        if self._entropy_cache is not None and self._entropy_cache_gen == grid_id:
            return self._entropy_cache
        prob_grid = 1.0 - 1.0 / (1.0 + np.exp(grid))
        p = np.clip(prob_grid, 1e-5, 1.0 - 1e-5)
        self._entropy_cache = -p * np.log2(p) - (1 - p) * np.log2(1 - p)
        self._entropy_cache_gen = grid_id
        return self._entropy_cache

    def _estimate_epistemic_value(self, cluster_center, entropy_map, radius=4.0):
        if entropy_map is None: return 0
        grid_map = self.env.grid_map
        r_center, c_center = self._world_to_grid(cluster_center)
        r_px = int(radius * grid_map.resolution)
        
        r_min, r_max = max(0, r_center - r_px), min(grid_map.num_cells_world, r_center + r_px)
        c_min, c_max = max(0, c_center - r_px), min(grid_map.num_cells_world, c_center + r_px)
        
        return np.sum(entropy_map[r_min:r_max, c_min:c_max]) / (grid_map.resolution ** 2)

    def _estimate_path_length(self, start_pos, goal_pos):
        # 量化到 0.5m 网格做缓存 key，避免浮点精度问题
        key = (round(float(start_pos[0]) * 2), round(float(start_pos[1]) * 2),
               round(float(goal_pos[0]) * 2), round(float(goal_pos[1]) * 2))
        cached = self._path_cache.get(key)
        if cached is not None:
            return cached
        if not hasattr(self.env, "path_planner"):
            result = float(np.linalg.norm(goal_pos - start_pos))
        else:
            try:
                path = self.env.path_planner.find_path(np.asarray(start_pos), np.asarray(goal_pos))
            except Exception:
                path = None
            if path is None or len(path) < 2:
                result = float("inf")
            else:
                path = np.asarray(path, dtype=np.float32)
                result = float(np.sum(np.linalg.norm(np.diff(path, axis=0), axis=1)))
        self._path_cache[key] = result
        return result

    def _evaluate_efe(self, cluster_center, robot_pos, entropy_map, direction_bias=None):
        epistemic_value = self._estimate_epistemic_value(cluster_center, entropy_map, radius=4.0)
        dist_to_robot = np.linalg.norm(cluster_center - robot_pos)
        path_length = self._estimate_path_length(robot_pos, cluster_center)
        if not np.isfinite(path_length):
            return -np.inf, epistemic_value, float("inf")
        traj_density = self._trajectory_density_at(cluster_center, radius=2.5)
        local_switch_cost = 0.0
        if self._last_frontier_center is not None:
            local_switch_cost = np.linalg.norm(cluster_center - self._last_frontier_center)
        pragmatic_cost = path_length + 0.35 * dist_to_robot + traj_density * 2.0 + local_switch_cost * 1.25
        
        if direction_bias is not None:
            direction_to_cluster = cluster_center - robot_pos
            dir_norm = np.linalg.norm(direction_to_cluster)
            if dir_norm > 0.1:
                alignment = np.dot(direction_to_cluster / dir_norm, direction_bias)
                pragmatic_cost -= alignment * 5.0 

        alpha = 0.9 
        EFE = pragmatic_cost - alpha * epistemic_value
        return -EFE, epistemic_value, pragmatic_cost

    def _score_frontier_candidate(self, cluster_center, nav_candidate, robot_pos, entropy_map, direction_bias=None):
        path_length = self._estimate_path_length(robot_pos, nav_candidate)
        if not np.isfinite(path_length):
            return None

        direct_dist = float(np.linalg.norm(nav_candidate - robot_pos))
        detour_ratio = path_length / max(direct_dist, 0.5)
        epistemic_value = self._estimate_epistemic_value(nav_candidate, entropy_map, radius=3.0)
        if epistemic_value < 0.5:
            return None

        traj_density = self._trajectory_density_at(cluster_center, radius=2.5)
        local_switch_cost = 0.0
        if self._last_frontier_center is not None:
            local_switch_cost = np.linalg.norm(cluster_center - self._last_frontier_center)

        detour_penalty = max(0.0, detour_ratio - self._detour_ratio_soft_limit) * 6.0
        far_penalty = max(0.0, path_length - self._local_frontier_path_horizon_m) ** 2 * 0.35
        pragmatic_cost = (
            path_length * 1.8
            + direct_dist * 0.25
            + detour_penalty
            + far_penalty
            + traj_density * 1.5
            + local_switch_cost * 0.7
        )

        if direction_bias is not None:
            direction_to_cluster = cluster_center - robot_pos
            dir_norm = np.linalg.norm(direction_to_cluster)
            if dir_norm > 0.1:
                alignment = np.dot(direction_to_cluster / dir_norm, direction_bias)
                pragmatic_cost -= alignment * 3.0

        epistemic_reward = 3.0 * np.log1p(epistemic_value)
        score = epistemic_reward - pragmatic_cost
        return {
            'score': score,
            'epistemic_val': epistemic_value,
            'prag_cost': pragmatic_cost,
            'path_length': path_length,
            'direct_dist': direct_dist,
            'detour_ratio': detour_ratio,
        }

    def _get_visual_revisit_target(self, robot_pos, target_place=None, max_attempts=2):
        if self.topo_map is None or not self.topo_map.nodes:
            return None
        candidates = []
        for idx, node in enumerate(self.topo_map.nodes):
            if node.get("landmarks_seen"):
                continue
            if node.get("visual_scan_count", 0) > 0:
                continue
            attempts = self._visual_revisit_attempts.get(idx, 0)
            if attempts >= max_attempts:
                continue
            pos = np.asarray(node["pos"], dtype=np.float32)
            path_length = self._estimate_path_length(robot_pos, pos)
            if not np.isfinite(path_length):
                continue
            # Prefer nearby under-observed places; older nodes get a small bonus so they are eventually checked.
            age_bonus = min(3.0, max(0.0, time.time() - float(node.get("last_seen", time.time()))) / 120.0)
            score = path_length + attempts * 4.0 - age_bonus
            candidates.append((score, idx, pos))
        if not candidates:
            return None
        candidates.sort(key=lambda item: item[0])
        _, idx, pos = candidates[0]
        self._visual_revisit_attempts[idx] = self._visual_revisit_attempts.get(idx, 0) + 1
        return idx, pos

    def _random_walk_target(self, robot_pos, direction_bias=None):
        if direction_bias is not None:
            base_angle = np.arctan2(direction_bias[1], direction_bias[0])
            random_angle = base_angle + np.random.uniform(-np.pi/3, np.pi/3)
        else: random_angle = np.random.uniform(-np.pi, np.pi)
        r_target = robot_pos + np.array([np.cos(random_angle)*4.5, np.sin(random_angle)*4.5])
        return np.clip(r_target, -8.5, 8.5)

    def execute(self, max_rounds=20, nav_steps_per_round=5000, target_place=None, direction_hint=None):
        if target_place and isinstance(target_place, list) and not target_place: target_place = None
        initial_mem_keys = set(self.percept_skill.memory.memory_db.keys())
        journey_log = ["开始执行探索任务。"]
        self._path_cache.clear()
        consecutive_failures = 0
        direction_bias = self._parse_direction_hint(direction_hint)
        if direction_bias is not None: journey_log.append(f"LLM 常识提示：注入优先向 {direction_hint} 探索的先验信念。")

        self.trajectory_history.clear()
        if not self._has_initial_scanned:
            self._initial_scan()
            self._has_initial_scanned = True

        def dynamic_frontier_update():
            if getattr(self.env, "_shutdown_requested", False):
                return True
            if hasattr(self.env, 'current_frontiers'): self.env.current_frontiers = self._get_frontiers()
            self._record_trajectory()
            self.percept_skill.scan_and_remember(wait_for_qt=False)
            progress_callback = getattr(self.env, "_qt_progress_callback", None)
            if progress_callback is not None:
                progress_callback()
            if target_place:
                targets =[target_place] if isinstance(target_place, str) else target_place
                if sum(1 for t in targets if self.percept_skill.memory.get_location_by_meaning(t)) == len(targets): return True
            else:
                if len(set(self.percept_skill.memory.memory_db.keys())) > len(initial_mem_keys): return True
            return False

        for round_idx in range(max_rounds):
            if getattr(self.env, "_shutdown_requested", False):
                journey_log.append("收到停止请求，已中断探索。")
                break
            self._record_trajectory()
            self._path_cache.clear()
            robot_pos = self.env.data.xpos[self.env.robot_base_body_id][:2].copy()
            missing_targets = []
            if target_place:
                targets_list = [target_place] if isinstance(target_place, str) else target_place
                missing_targets = [t for t in targets_list if not self.percept_skill.memory.get_location_by_meaning(t)]

            if consecutive_failures >= 5:
                journey_log.append(f"连续 {consecutive_failures} 次导航失败，终止本轮探索。")
                break

            if missing_targets and round_idx > 0 and round_idx % 3 == 0:
                revisit = self._get_visual_revisit_target(robot_pos, target_place=missing_targets)
                if revisit is not None:
                    node_idx, revisit_pos = revisit
                    journey_log.append(
                        f"第{round_idx + 1}轮：目标 {missing_targets} 仍未视觉确认，"
                        f"回访拓扑节点 #{node_idx} 做视觉复扫 ({revisit_pos[0]:.1f}, {revisit_pos[1]:.1f})。"
                    )
                    self.nav_skill.go_to(
                        revisit_pos[0], revisit_pos[1], max_steps=1200, success_dist=1.2,
                        step_callback=dynamic_frontier_update, callback_freq=40
                    )
                    detected = self._visual_sweep_and_remember(turns=6)
                    if self.topo_map is not None:
                        robot_pos_now = self.env.data.xpos[self.env.robot_base_body_id][:2].copy()
                        self.topo_map.visit(
                            self.env.grid_map,
                            robot_pos_now,
                            landmarks=[d["id"] for d in detected] if detected else [],
                            visual_scan=True
                        )
                    if all(self.percept_skill.memory.get_location_by_meaning(t) for t in missing_targets):
                        journey_log.append(f"✅ 回访复扫后找齐目标 {missing_targets}。")
                        break

            frontier_points = self._get_frontiers()
            if len(frontier_points) == 0:
                if target_place:
                    targets = [target_place] if isinstance(target_place, str) else target_place
                    missing =[t for t in targets if not self.percept_skill.memory.get_location_by_meaning(t)]
                    if missing:
                        rt = self._random_walk_target(robot_pos, direction_bias)
                        journey_log.append(f"边界耗尽但 {missing} 未找齐，随机探索方向 ({rt[0]:.1f}, {rt[1]:.1f})")
                        self.nav_skill.go_to(rt[0], rt[1], max_steps=2000, success_dist=1.5, step_callback=dynamic_frontier_update, callback_freq=60)
                        self._clear_frontier_history()
                        if not[t for t in targets if not self.percept_skill.memory.get_location_by_meaning(t)]: break
                        continue
                journey_log.append("地图已完全探索，无更多可达区域。"); break

            local_radius_m = 4.0
            local_done, local_stats = self._estimate_local_exploration_status(
                robot_pos, frontier_points, radius_m=local_radius_m
            )
            outside_frontier_exists = bool(
                len(frontier_points) > 0
                and np.any(np.linalg.norm(frontier_points - robot_pos, axis=1) > local_radius_m)
            )
            if local_done:
                local_key = self._frontier_key(robot_pos, cell_size=3.0)
                if local_key != self._local_completion_announced_key:
                    journey_log.append(
                        f"当前局部空间已探索完成：附近前沿={local_stats['local_frontier_count']}，"
                        f"未知比例={local_stats['unknown_ratio'] * 100:.1f}%；转向下一个未探索区域。"
                    )
                    self._local_completion_announced_key = local_key

            clusters = self._cluster_frontiers(frontier_points)
            target_cluster = None
            current_entropy_map = self._compute_entropy_map()
            candidates = []

            for cluster in clusters:
                center = cluster['center']
                if self._is_frontier_visited(center): continue
                if np.linalg.norm(center - robot_pos) < 1.0: continue
                if local_done and outside_frontier_exists and np.linalg.norm(center - robot_pos) <= local_radius_m: continue
                if self._is_in_explored_corridor(center, threshold_ratio=0.12): continue
                nav_candidate = self._compute_nav_target(center)
                if not self._frontier_target_has_access(nav_candidate, center): continue
                metrics = self._score_frontier_candidate(
                    cluster_center=center,
                    nav_candidate=nav_candidate,
                    robot_pos=robot_pos,
                    entropy_map=current_entropy_map,
                    direction_bias=direction_bias,
                )
                if metrics is None:
                    continue
                candidate = dict(cluster)
                candidate['nav_candidate'] = nav_candidate
                candidate.update(metrics)
                candidates.append(candidate)

            if candidates:
                reasonable_detour = [
                    c for c in candidates
                    if c['detour_ratio'] <= self._detour_ratio_prefer_limit or c['path_length'] <= 4.0
                ]
                if reasonable_detour:
                    candidates = reasonable_detour
                candidates.sort(key=lambda c: (c['path_length'], -c['epistemic_val']))
                min_path = candidates[0]['path_length']
                near_slack = (
                    self._near_frontier_path_slack_m
                    if min_path <= self._near_frontier_tier_threshold_m
                    else self._near_frontier_far_slack_m
                )
                near_path_limit = min_path + near_slack
                local_limit = min(
                    self._local_frontier_path_horizon_m,
                    max(4.0, near_path_limit)
                )
                near_candidates = [c for c in candidates if c['path_length'] <= near_path_limit]
                local_candidates = [c for c in near_candidates if c['path_length'] <= local_limit]
                candidate_pool = local_candidates if local_candidates else near_candidates
                target_cluster = max(candidate_pool, key=lambda c: c['score'])
                best_score = target_cluster['score']

            if target_cluster is None:
                if target_place:
                    targets = [target_place] if isinstance(target_place, str) else target_place
                    if[t for t in targets if not self.percept_skill.memory.get_location_by_meaning(t)]:
                        rt = self._random_walk_target(robot_pos, direction_bias)
                        self.nav_skill.go_to(rt[0], rt[1], max_steps=600, success_dist=1.5, step_callback=dynamic_frontier_update, callback_freq=40)
                        self._clear_frontier_history()
                        continue
                journey_log.append("剩余未知区域暂时无法到达..."); self._clear_frontier_history()
                continue

            raw_target = target_cluster['center']
            nav_target = target_cluster.get('nav_candidate', self._compute_nav_target(raw_target))
            journey_log.append(
                f"第{round_idx + 1}轮：基于自由能最小化决策 → "
                f"目标=({nav_target[0]:.1f}, {nav_target[1]:.1f})，"
                f"预期消除香农熵(Epistemic)={target_cluster['epistemic_val']:.1f}，"
                f"路径={target_cluster['path_length']:.1f}m，绕路倍率={target_cluster['detour_ratio']:.1f}x，"
                f"物理与先验消耗(Pragmatic)={target_cluster['prag_cost']:.1f}，最终EFE评分={best_score:.1f}"
            )

            success, dist = self.nav_skill.go_to(
                nav_target[0], nav_target[1], max_steps=nav_steps_per_round, success_dist=1.5,
                step_callback=dynamic_frontier_update, callback_freq=40
            )
            self.visited_frontiers.append(raw_target.copy())
            self._visited_frontier_keys.add(self._frontier_key(raw_target, cell_size=1.2))
            if success:
                self._last_frontier_center = raw_target.copy()

            if not success:
                consecutive_failures += 1; self.failed_frontiers.append(raw_target.copy())
                self._failed_frontier_keys.add(self._frontier_key(raw_target, cell_size=1.2))
                journey_log.append(f"  → 导航受阻 (dist={dist:.1f})，记录拓扑失败节点。")
                # 连续失败时清空已访问边界，强制重新评估（碰撞后机器人位置已变）
                if consecutive_failures >= 2:
                    journey_log.append(f"  ⚠️ 连续 {consecutive_failures} 次失败，强制重新评估边界。")
                    self.visited_frontiers.clear()
                    self._visited_frontier_keys.clear()
                    # 重新扫描感知
                    self.percept_skill.scan_and_remember()
            else:
                consecutive_failures = 0; journey_log.append(f"  → 成功到达认知目标点。")

            detected_landmarks = self._visual_sweep_and_remember(turns=4) if target_place else self.percept_skill.scan_and_remember()

            # 拓扑地图：记录当前位置
            if self.topo_map is not None:
                robot_pos_now = self.env.data.xpos[self.env.robot_base_body_id][:2].copy()
                lm_ids = [d['id'] for d in detected_landmarks] if detected_landmarks else []
                node, is_new = self.topo_map.visit(
                    self.env.grid_map,
                    robot_pos_now,
                    landmarks=lm_ids,
                    visual_scan=bool(target_place)
                )
                if not is_new and node['visit_count'] >= 3:
                    journey_log.append(f"  🔁 识别到重复访问！这里已经是第 {node['visit_count']} 次来了。")

            current_keys = set(self.percept_skill.memory.memory_db.keys())
            if current_keys != initial_mem_keys:
                for new_key in (current_keys - initial_mem_keys):
                    journey_log.append(f"  ★ 发现新地点：{self.percept_skill.memory.feature_meaning.get(new_key, new_key)}")
                initial_mem_keys = current_keys
            progress_callback = getattr(self.env, "_qt_progress_callback", None)
            if progress_callback is not None:
                progress_callback(force=True)

            if target_place:
                targets_list = [target_place] if isinstance(target_place, str) else target_place
                if all(self.percept_skill.memory.get_location_by_meaning(t) for t in targets_list):
                    journey_log.append(f"✅ 目标集 {target_place} 已全部找齐！"); break

        self.env.current_frontiers =[]
        return "\n".join(journey_log)
