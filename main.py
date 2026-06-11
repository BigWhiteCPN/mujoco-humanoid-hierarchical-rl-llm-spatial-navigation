import os
import sys
import time
import threading
from queue import Queue

os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"
os.environ["VECLIB_MAXIMUM_THREADS"] = "1"
os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")
# The default GLFW backend fails under the current Wayland session and can leave
# MuJoCo offscreen frames black. Set this before importing anything that imports
# mujoco.
os.environ.setdefault("MUJOCO_GL", "egl")

import matplotlib.pyplot as plt

from agent_env import AgentVisualEnv
from memory import SpatialMemory, TopologicalMap
from skills import NavigationSkill, PerceptionSkill, FrontierExplorationSkill
from llm_brain import RobotAgent
from realtime_runner import RealtimeRunner


def dashboard_log(env, message):
    if hasattr(env, "append_dashboard_log"):
        env.append_dashboard_log(message)


class CommandWindow:
    def __init__(self):
        import tkinter as tk
        self.tk = tk
        self.queue = Queue()
        self.root = tk.Toplevel(tk._default_root) if tk._default_root is not None else tk.Tk()
        self.root.title("Robot Command")
        self.root.geometry("560x84")
        try:
            self.root.attributes("-topmost", True)
        except Exception:
            pass

        self.var = tk.StringVar()
        frame = tk.Frame(self.root, padx=8, pady=8)
        frame.pack(fill=tk.BOTH, expand=True)
        self.entry = tk.Entry(frame, textvariable=self.var, font=("Noto Sans CJK SC", 12))
        self.entry.pack(side=tk.LEFT, fill=tk.X, expand=True)
        self.entry.bind("<Return>", lambda event: self.submit())
        button = tk.Button(frame, text="发送", width=8, command=self.submit)
        button.pack(side=tk.LEFT, padx=(8, 0))
        self.entry.focus_set()
        self._last_raise_time = 0.0

    def submit(self):
        text = self.var.get().strip()
        if text:
            self.queue.put(text)
            self.var.set("")

    def poll(self):
        now = time.time()
        if now - self._last_raise_time > 1.0:
            try:
                self.root.lift()
                self.entry.focus_set()
            except Exception:
                pass
            self._last_raise_time = now
        self.root.update_idletasks()
        self.root.update()

    def get_command(self):
        try:
            return self.queue.get_nowait()
        except Exception:
            return None

    def close(self):
        try:
            self.root.destroy()
        except Exception:
            pass


class TerminalCommandInput:
    uses_terminal = True

    def __init__(self):
        self.queue = Queue()
        self._closed = False
        self._thread = threading.Thread(target=self._read_loop, daemon=True)
        self._thread.start()

    def _read_loop(self):
        while not self._closed:
            try:
                text = input()
            except (EOFError, KeyboardInterrupt):
                break
            if text.strip():
                self.queue.put(text.strip())

    def poll(self):
        return

    def get_command(self):
        try:
            return self.queue.get_nowait()
        except Exception:
            return None

    def close(self):
        self._closed = True


def input_with_idle(prompt, env, realtime_runner, command_window):
    print(prompt, end="", flush=True)
    if getattr(command_window, "uses_terminal", False):
        dashboard_log(env, "输入: 请在终端输入命令并回车。")
    else:
        dashboard_log(env, "输入: 请在 Robot Command 窗口输入命令并点击发送/回车。")
    if getattr(env, "render_mode", None) == "dashboard" and hasattr(env, "_ensure_dashboard"):
        env._ensure_dashboard()

    while True:
        command_window.poll()
        command = command_window.get_command()
        if command is not None:
            print(command)
            return command
        realtime_runner.step_idle()
        time.sleep(0.001)


def main():
    model_xml = os.environ.get(
        "ROBOT_MODEL_XML",
        "/home/chen/code/IsaacLabExtensionTemplate/scripts/resources/mjcf/Linnxil_fifteen_angle_bs_copy_20260302_copy.xml",
    )
    low_level_policy_path = os.environ.get("LOW_LEVEL_POLICY_PATH", "/home/chen/policy/policy_20251026.pt")
    sac_model_path = os.environ.get(
        "SAC_MODEL_PATH",
        "/home/chen/code/IsaacLabExtensionTemplate/sac_lidar_logs_random/sac_lidar_interrupted_good3_0.91.zip",
    )

    print("=== 启动 MuJoCo 导航演示 ===")
    print("[DEBUG] CPU 数值库线程限制为 1，降低 Qt/MuJoCo/策略推理抢占抖动")

    env_kwargs = {
        "model_path": model_xml,
        "low_level_policy_path": low_level_policy_path,
        "render_mode": 'qt_dashboard',
        "render_decimation": 50,
        "action_repeat": 4,
        "history_length": 15,
        "enable_dynamic_obstacles": False,
        "enable_mujoco_viewer": False,
    }
    env = AgentVisualEnv(**env_kwargs)
    env.debug_timing = True
    print("[DEBUG] timing 日志已开启：关注 [PERF NAV] / [PERF UI] / [JUMP NAV]")
    realtime_runner = RealtimeRunner(env, idle_hz=50.0, perception_hz=5.0, render_hz=15.0, max_catchup_steps=1)
    command_window = None if env_kwargs["render_mode"] == "qt_dashboard" else CommandWindow()

    print("\n[系统] 正在生成随机迷宫并放置地标...")
    env.reset()

    debug_pos = env.get_landmark_positions_debug()
    print("[DEBUG] 地标真实位置:")
    for lm_id, pos in debug_pos.items():
        print(f"  {lm_id}: ({pos[0]:.2f}, {pos[1]:.2f})")

    memory = SpatialMemory(save_dir="memory_logs")
    topo_map = TopologicalMap(fingerprint_radius_m=3.0, fingerprint_size=8, match_threshold=0.85)
    nav_skill = NavigationSkill(env, sac_model_path)
    percept_skill = PerceptionSkill(env, memory)
    explore_skill = FrontierExplorationSkill(env, nav_skill, percept_skill, topo_map=topo_map)
    agent = RobotAgent(nav_skill, memory, explore_skill, topo_map=topo_map)

    print("\n=======================================================")
    print("系统已启动：随机迷宫、Lidar 栅格地图和导航策略已初始化。")
    print("示例指令：")
    print("   '去会议室看看'  - 触发 Frontier 探索")
    print("   '你发现了什么'  - 查看记忆")
    print("   '回忆一下'      - 查看拓扑空间记忆（去过哪里、走了几次）")
    print("   '加载记忆'      - 加载上次的地图和拓扑记忆")
    print("   '保存记忆'      - 持久化到文件")
    print("   '重置地图'      - 新迷宫 + 清空记忆")
    print("   '退出'          - 保存并退出")
    print("=======================================================\n")
    dashboard_log(env, "系统: 智能体启动完毕，等待用户指令。")

    session_id = time.strftime("%Y%m%d_%H%M%S")
    last_session_path = None

    # 供“加载记忆”使用：默认选择最近一次保存的 session。
    if os.path.exists(memory.save_dir):
        sessions = sorted([d for d in os.listdir(memory.save_dir) if d.startswith("session_")])
        if sessions:
            last_session_path = os.path.join(memory.save_dir, sessions[-1])

    closed = False

    def shutdown(reason="系统: 正在保存记忆并退出。"):
        nonlocal closed
        if closed:
            return
        closed = True
        dashboard_log(env, reason)
        save_path = memory.save_to_file(session_id)
        memory.save_visited_map(env.grid_map, session_id)
        topo_map.save_to_file(save_path)
        if command_window is not None:
            try:
                command_window.close()
            except Exception:
                pass
        if hasattr(env, 'close'):
            env.close()
        plt.close('all')
        print("[系统] 已退出")

    def handle_command(user_cmd):
        nonlocal session_id, last_session_path
        command = user_cmd.strip()
        if command in ['退出', 'exit', 'quit']:
            dashboard_log(env, "系统: 收到退出指令。")
            return True

        if not command:
            return False

        if command in ['保存记忆', 'save']:
            dashboard_log(env, "用户: 保存记忆")
            save_path = memory.save_to_file(session_id)
            memory.save_visited_map(env.grid_map, session_id)
            topo_map.save_to_file(save_path)
            last_session_path = save_path
            dashboard_log(env, "结果: 记忆、访问地图和拓扑记忆已保存。")
            return False

        if command in ['加载记忆', 'load']:
            dashboard_log(env, "用户: 加载记忆")
            if last_session_path and os.path.exists(last_session_path):
                memory.load_from_file(last_session_path)
                memory.load_visited_map(env.grid_map, last_session_path)
                topo_map.load_from_file(last_session_path)
                print(f"[系统] 已加载 session: {os.path.basename(last_session_path)}")
                dashboard_log(env, f"结果: 已加载 session {os.path.basename(last_session_path)}。")
            else:
                print("[系统] 未找到历史记忆。")
                dashboard_log(env, "结果: 未找到历史记忆。")
            return False

        if command in ['回忆', 'recall']:
            dashboard_log(env, "用户: 回忆")
            robot_pos = env.data.xpos[env.robot_base_body_id][:2]
            spatial_report = memory.get_spatial_report(robot_pos)
            topo_report = topo_map.get_explored_summary()
            print(f"\n[空间记忆]\n{spatial_report}")
            print(f"\n[拓扑记忆]\n{topo_report}")
            dashboard_log(env, f"结果: {spatial_report} {topo_report}")
            hotspots = topo_map.get_revisit_candidates(min_visits=2)
            if hotspots:
                print("\n[重复访问位置]")
                for i, node in enumerate(hotspots):
                    pos = node['pos']
                    print(f"  节点{i+1}: ({pos[0]:.1f}, {pos[1]:.1f})，访问 {node['visit_count']} 次")
            return False

        if command in ['重置地图', 'reset map']:
            print("[系统] 正在生成新迷宫...")
            dashboard_log(env, "用户: 重置地图")
            env.reset()
            memory.clear()
            topo_map.clear()
            explore_skill.visited_frontiers.clear()
            explore_skill._has_initial_scanned = False
            session_id = time.strftime("%Y%m%d_%H%M%S")
            debug_pos = env.get_landmark_positions_debug()
            print("[DEBUG] 新地标位置:")
            for lm_id, pos in debug_pos.items():
                print(f"  {lm_id}: ({pos[0]:.2f}, {pos[1]:.2f})")
            print("[系统] 新迷宫已生成，记忆已清空。")
            dashboard_log(env, "结果: 新迷宫已生成，记忆已清空。")
            return False

        reply = agent.chat_and_execute(command)
        print(f"[机器人] {reply}")
        dashboard_log(env, f"机器人: {reply}")
        return False

    if env_kwargs["render_mode"] == "qt_dashboard":
        from qt_dashboard import run_qt_dashboard
        try:
            run_qt_dashboard(env, realtime_runner, handle_command, shutdown)
        except KeyboardInterrupt:
            shutdown("系统: 强制中断，正在保存记忆。")
        return

    while True:
        try:
            user_cmd = input_with_idle("\n[用户] 请输入指令: ", env, realtime_runner, command_window)
            if handle_command(user_cmd):
                shutdown()
                break
        except KeyboardInterrupt:
            shutdown("系统: 强制中断，正在保存记忆。")
            sys.exit(0)

if __name__ == '__main__':
    main()
