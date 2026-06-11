import json
import os
import time
import numpy as np
from scipy.spatial.distance import cosine
from skimage.transform import resize as skimage_resize

class SpatialMemory:
    def __init__(self, save_dir="memory_logs", max_log_size=10000):
        self.memory_db = {}
        self.save_dir = save_dir
        os.makedirs(save_dir, exist_ok=True)
        self.max_log_size = max_log_size

        # Bounded runtime logs; save_to_file persists them per session.
        self.odometry_log = []
        self.feature_log = []
        
        # Domain labels used when reporting color landmarks to the model/user.
        self.feature_meaning = {
            "landmark_red": "红色方块(可能是老板办公室)",
            "landmark_blue": "品红色圆柱体/方块(可能是会议室)",
            "landmark_green": "绿色方块(可能是大门)",
            "landmark_yellow": "黄色方块(可能是茶水间)"
        }
        
    def add_memory(self, feature_id, x, y, confidence=1.0):
        """
        写入或更新一个地标的记忆坐标。
        如果之前已经观测过，使用加权平均融合多次观测。
        """
        if feature_id not in self.memory_db:
            self.memory_db[feature_id] = {
                "x": float(x),
                "y": float(y),
                "observations": 1,
                "confidence": confidence
            }
            meaning = self.feature_meaning.get(feature_id, feature_id)
            print(f"\n[记忆] 发现 {meaning}，估计坐标: ({x:.2f}, {y:.2f})")
        else:
            entry = self.memory_db[feature_id]
            n = entry["observations"]
            delta = np.linalg.norm([float(x) - entry["x"], float(y) - entry["y"]])
            alpha = 0.55 if delta > 0.75 else 1.0 / (n + 1)
            entry["x"] = entry["x"] * (1 - alpha) + float(x) * alpha
            entry["y"] = entry["y"] * (1 - alpha) + float(y) * alpha
            entry["observations"] = n + 1
            entry["confidence"] = min(1.0, entry["confidence"] + 0.1)
        
        self.feature_log.append({
            "timestamp": time.time(),
            "feature_id": feature_id,
            "observed_x": float(x),
            "observed_y": float(y),
            "fused_x": self.memory_db[feature_id]["x"],
            "fused_y": self.memory_db[feature_id]["y"],
            "total_obs": self.memory_db[feature_id]["observations"]
        })
        if len(self.feature_log) > self.max_log_size:
            self.feature_log = self.feature_log[-self.max_log_size:]

    def log_odometry(self, robot_x, robot_y, robot_yaw):
        """Append one odometry sample to the bounded in-memory log."""
        self.odometry_log.append({
            "timestamp": time.time(),
            "x": float(robot_x),
            "y": float(robot_y),
            "yaw": float(robot_yaw)
        })
        if len(self.odometry_log) > self.max_log_size:
            self.odometry_log = self.odometry_log[-self.max_log_size:]

    def get_spatial_report(self, robot_pos):
        """Return a short relative-position summary for known landmarks."""
        if not self.memory_db:
            return "当前没有已知地标。"
        
        report = ["根据我的空间感知，目前已知地标的相对位置如下："]
        for fid, entry in self.memory_db.items():
            meaning = self.feature_meaning.get(fid, fid)
            dx = entry["x"] - robot_pos[0]
            dy = entry["y"] - robot_pos[1]
            dist = np.sqrt(dx**2 + dy**2)
            
            angle = np.arctan2(dy, dx)
            if -np.pi/8 <= angle < np.pi/8: dir_str = "正东方"
            elif np.pi/8 <= angle < 3*np.pi/8: dir_str = "东北方"
            elif 3*np.pi/8 <= angle < 5*np.pi/8: dir_str = "正北方"
            elif 5*np.pi/8 <= angle < 7*np.pi/8: dir_str = "西北方"
            elif angle >= 7*np.pi/8 or angle < -7*np.pi/8: dir_str = "正西方"
            elif -3*np.pi/8 <= angle < -np.pi/8: dir_str = "东南方"
            elif -5*np.pi/8 <= angle < -3*np.pi/8: dir_str = "正南方"
            else: dir_str = "西南方"
            
            report.append(f"- {meaning}：位于我当前位置的{dir_str}，直线距离约{dist:.1f}米。")
        return "\n".join(report)
    
    def _find_single_location(self, keyword):
        """Find one known landmark by id or display label."""
        for fid, entry in self.memory_db.items():
            meaning = self.feature_meaning.get(fid, "")
            if keyword in meaning or keyword in fid:
                return [entry["x"], entry["y"]]
        return None

    def get_feature_id_by_meaning(self, keyword):
        """返回语义关键词对应的已观测地标 id。"""
        for fid in self.memory_db.keys():
            meaning = self.feature_meaning.get(fid, "")
            if keyword in meaning or keyword in fid:
                return fid
        return None
    
    def get_location_by_meaning(self, keyword):
        """
        Look up a landmark coordinate by semantic keyword, for example
        "会议室" -> [x, y]. Returns None when no observed landmark matches.
        """
        result = self._find_single_location(keyword)
        if result:
            return result
        
        for fid, entry in self.memory_db.items():
            meaning = self.feature_meaning.get(fid, "")
            if keyword in meaning or keyword in fid:
                return [entry["x"], entry["y"]]
            # Keep a fallback for labels like "红色方块(可能是老板办公室)".
            for segment in meaning.replace("(", "").replace(")", "").replace("可能是", "").split():
                if segment and segment in keyword:
                    return [entry["x"], entry["y"]]
        
        return None

    def get_all_known_places(self):
        """Return known places in a compact text format for prompting."""
        places = []
        for fid, entry in self.memory_db.items():
            meaning = self.feature_meaning.get(fid, fid)
            places.append(f"{meaning} (约{entry['x']:.1f}, {entry['y']:.1f}), 观测{entry['observations']}次")
        return places if places else ["这里是一片未知区域，我什么都没发现。"]

    def save_to_file(self, session_id=None):
        """Persist landmark memory and runtime logs for one session."""
        if session_id is None:
            session_id = time.strftime("%Y%m%d_%H%M%S")
        
        save_path = os.path.join(self.save_dir, f"session_{session_id}")
        os.makedirs(save_path, exist_ok=True)
        
        memory_file = os.path.join(save_path, "spatial_memory.json")
        with open(memory_file, "w", encoding="utf-8") as f:
            json.dump(self.memory_db, f, ensure_ascii=False, indent=2)
        
        odom_file = os.path.join(save_path, "odometry.json")
        with open(odom_file, "w", encoding="utf-8") as f:
            json.dump(self.odometry_log, f, indent=2)
        
        feature_file = os.path.join(save_path, "feature_observations.json")
        with open(feature_file, "w", encoding="utf-8") as f:
            json.dump(self.feature_log, f, ensure_ascii=False, indent=2)
        
        print(f"\n[记忆] 已保存到: {save_path}/")
        return save_path

    def load_from_file(self, session_path):
        """Load landmark memory from a saved session directory."""
        memory_file = os.path.join(session_path, "spatial_memory.json")
        if os.path.exists(memory_file):
            with open(memory_file, "r", encoding="utf-8") as f:
                self.memory_db = json.load(f)
            print(f"[记忆] 已加载 {len(self.memory_db)} 个地标记忆")

    def clear(self):
        """Clear per-map memory and logs."""
        self.memory_db.clear()
        self.odometry_log.clear()
        self.feature_log.clear()

    def save_visited_map(self, grid_map, session_id=None):
        """Persist occupancy and visited grids for map reuse/debugging."""
        if session_id is None:
            session_id = time.strftime("%Y%m%d_%H%M%S")
        save_path = os.path.join(self.save_dir, f"session_{session_id}")
        os.makedirs(save_path, exist_ok=True)
        np.save(os.path.join(save_path, "occupancy_grid.npy"), grid_map.grid)
        np.save(os.path.join(save_path, "visited_grid.npy"), grid_map.visited_grid)
        print(f"[记忆] 地图已保存: {save_path}/")
        return save_path

    def load_visited_map(self, grid_map, session_path):
        """Merge a saved occupancy map into the current grid."""
        occ_file = os.path.join(session_path, "occupancy_grid.npy")
        vis_file = os.path.join(session_path, "visited_grid.npy")
        if not os.path.exists(occ_file) or not os.path.exists(vis_file):
            print("[记忆] 未找到历史地图文件")
            return False
        saved_occ = np.load(occ_file)
        saved_vis = np.load(vis_file)
        if saved_occ.shape != grid_map.grid.shape:
            print("[记忆] 历史地图尺寸不匹配，跳过")
            return False
        visited_mask = saved_vis > 0.3
        grid_map.grid[visited_mask] = saved_occ[visited_mask] * 0.7 + grid_map.grid[visited_mask] * 0.3
        grid_map.visited_grid = np.maximum(grid_map.visited_grid, saved_vis * 0.8)
        count = int(np.sum(visited_mask))
        print(f"[记忆] 已融合历史地图，覆盖 {count} 个格子 ({count / visited_mask.size * 100:.1f}%)")
        return True


class TopologicalMap:
    """
    Place recognition from local lidar occupancy fingerprints.
    Each node stores one visited position, a downsampled map patch, visit count,
    and landmarks seen there.
    """

    def __init__(self, fingerprint_radius_m=3.0, fingerprint_size=8, match_threshold=0.85):
        self.nodes = []
        self.fingerprint_radius_m = fingerprint_radius_m
        self.fingerprint_size = fingerprint_size
        self.match_threshold = match_threshold
        self.current_node_id = None

    def _world_to_grid(self, world_pos, grid_map):
        c = int((world_pos[0] + grid_map.world_origin_offset_m[0]) * grid_map.resolution)
        r = grid_map.num_cells_world - 1 - int((world_pos[1] + grid_map.world_origin_offset_m[1]) * grid_map.resolution)
        return r, c

    def compute_fingerprint(self, grid_map, robot_pos):
        """Extract a fixed-size occupancy patch around the robot."""
        r, c = self._world_to_grid(robot_pos, grid_map)
        r_px = int(self.fingerprint_radius_m * grid_map.resolution)
        r_min = max(0, r - r_px)
        r_max = min(grid_map.num_cells_world, r + r_px)
        c_min = max(0, c - r_px)
        c_max = min(grid_map.num_cells_world, c + r_px)
        patch = grid_map.grid[r_min:r_max, c_min:c_max]
        if patch.size == 0:
            return np.zeros(self.fingerprint_size * self.fingerprint_size)
        prob_patch = 1.0 - 1.0 / (1.0 + np.exp(patch))
        resized = skimage_resize(prob_patch, (self.fingerprint_size, self.fingerprint_size), anti_aliasing=False)
        return resized.flatten()

    def recognize_place(self, fingerprint):
        """Match a fingerprint against existing nodes using cosine similarity."""
        if len(self.nodes) == 0:
            return None, -1, 0.0
        fp_matrix = np.array([node['fingerprint'] for node in self.nodes])
        fp_norms = np.linalg.norm(fp_matrix, axis=1)
        query_norm = np.linalg.norm(fingerprint)
        denom = fp_norms * query_norm
        denom = np.where(denom < 1e-10, 1e-10, denom)
        similarities = fp_matrix @ fingerprint / denom
        best_idx = int(np.argmax(similarities))
        best_sim = float(similarities[best_idx])
        if best_sim > self.match_threshold:
            return self.nodes[best_idx], best_idx, best_sim
        return None, -1, best_sim

    def visit(self, grid_map, robot_pos, landmarks=None, visual_scan=False):
        """
        Record one visit and return (node, is_new).
        """
        fingerprint = self.compute_fingerprint(grid_map, robot_pos)
        matched_node, matched_idx, sim = self.recognize_place(fingerprint)

        if matched_node is not None:
            matched_node['visit_count'] += 1
            matched_node['last_seen'] = time.time()
            matched_node['pos'] = robot_pos.copy()
            if visual_scan:
                matched_node['visual_scan_count'] = matched_node.get('visual_scan_count', 0) + 1
                matched_node['last_visual_scan'] = time.time()
            if landmarks:
                for lm in landmarks:
                    if lm not in matched_node['landmarks_seen']:
                        matched_node['landmarks_seen'].append(lm)
            self.current_node_id = matched_idx
            print(f"[拓扑] 匹配节点 #{matched_idx}，"
                  f"第 {matched_node['visit_count']} 次访问，相似度 {sim:.3f}")
            return matched_node, False
        else:
            new_node = {
                'pos': robot_pos.copy(),
                'fingerprint': fingerprint,
                'visit_count': 1,
                'first_seen': time.time(),
                'last_seen': time.time(),
                'visual_scan_count': 1 if visual_scan else 0,
                'last_visual_scan': time.time() if visual_scan else 0.0,
                'landmarks_seen': list(landmarks) if landmarks else []
            }
            self.nodes.append(new_node)
            self.current_node_id = len(self.nodes) - 1
            print(f"[拓扑] 新建节点 #{self.current_node_id}，"
                  f"位置 ({robot_pos[0]:.1f}, {robot_pos[1]:.1f})")
            return new_node, True

    def get_explored_summary(self):
        """Return a compact summary of visited topological nodes."""
        if not self.nodes:
            return "还没有任何拓扑记忆。"
        lines = [f"共访问过 {len(self.nodes)} 个不同位置："]
        for i, node in enumerate(self.nodes):
            pos = node['pos']
            lms = ", ".join(node['landmarks_seen']) if node['landmarks_seen'] else "无地标"
            visual_count = node.get('visual_scan_count', 0)
            lines.append(f"  #{i}: ({pos[0]:.1f}, {pos[1]:.1f})，"
                         f"访问 {node['visit_count']} 次，视觉扫描 {visual_count} 次，可见地标: {lms}")
        return "\n".join(lines)

    def get_revisit_candidates(self, min_visits=2):
        """Return nodes visited at least min_visits times."""
        return [n for n in self.nodes if n['visit_count'] >= min_visits]

    def save_to_file(self, save_path):
        """Save topology with numpy arrays converted to lists."""
        serializable = []
        for node in self.nodes:
            serializable.append({
                'pos': node['pos'].tolist(),
                'fingerprint': node['fingerprint'].tolist(),
                'visit_count': node['visit_count'],
                'first_seen': node['first_seen'],
                'last_seen': node['last_seen'],
                'visual_scan_count': node.get('visual_scan_count', 0),
                'last_visual_scan': node.get('last_visual_scan', 0.0),
                'landmarks_seen': node['landmarks_seen']
            })
        topo_file = os.path.join(save_path, "topological_map.json")
        with open(topo_file, "w", encoding="utf-8") as f:
            json.dump(serializable, f, ensure_ascii=False, indent=2)
        print(f"[拓扑] 已保存 {len(self.nodes)} 个节点到: {topo_file}")

    def load_from_file(self, save_path):
        """Load topology from a saved session directory."""
        topo_file = os.path.join(save_path, "topological_map.json")
        if not os.path.exists(topo_file):
            return False
        with open(topo_file, "r", encoding="utf-8") as f:
            data = json.load(f)
        self.nodes = []
        for item in data:
            self.nodes.append({
                'pos': np.array(item['pos']),
                'fingerprint': np.array(item['fingerprint']),
                'visit_count': item['visit_count'],
                'first_seen': item['first_seen'],
                'last_seen': item['last_seen'],
                'visual_scan_count': item.get('visual_scan_count', 0),
                'last_visual_scan': item.get('last_visual_scan', 0.0),
                'landmarks_seen': item['landmarks_seen']
            })
        print(f"[拓扑] 已加载 {len(self.nodes)} 个拓扑节点")
        return True

    def clear(self):
        self.nodes.clear()
        self.current_node_id = None
