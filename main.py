import argparse
import logging
import os
import sys
import time
import threading
from queue import Queue

LOGGER = logging.getLogger(__name__)

DEFAULT_MODEL_XML = "resources/mjcf/Linnxil_fifteen_angle_bs_copy_20260302_copy.xml"
DEFAULT_LOW_LEVEL_POLICY = "models/policy_20251026.pt"
DEFAULT_SAC_MODEL = "models/sac_lidar_interrupted_good3_0.91.zip"


def env_bool(name, default=False):
    raw_value = os.environ.get(name)
    if raw_value is None:
        return default
    return raw_value.strip().lower() in {"1", "true", "yes", "y", "on"}


def parse_env_file_argument():
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--env-file", default=os.environ.get("ENV_FILE", ".env"))
    args, _ = parser.parse_known_args()
    return args.env_file


def build_arg_parser():
    parser = argparse.ArgumentParser(
        description="Run the MuJoCo robot navigation project.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--env-file", default=os.environ.get("ENV_FILE", ".env"))
    parser.add_argument("--robot-model-xml", default=os.environ.get("ROBOT_MODEL_XML", DEFAULT_MODEL_XML))
    parser.add_argument("--low-level-policy", default=os.environ.get("LOW_LEVEL_POLICY_PATH", DEFAULT_LOW_LEVEL_POLICY))
    parser.add_argument("--sac-model", default=os.environ.get("SAC_MODEL_PATH", DEFAULT_SAC_MODEL))
    parser.add_argument("--memory-dir", default=os.environ.get("MEMORY_DIR", "memory_logs"))

    parser.add_argument(
        "--render-mode",
        default=os.environ.get("RENDER_MODE", "qt_dashboard"),
        choices=["qt_dashboard", "dashboard", "fast_dashboard", "human", "rgb_array"],
    )
    parser.add_argument("--render-decimation", type=int, default=int(os.environ.get("RENDER_DECIMATION", "50")))
    parser.add_argument("--action-repeat", type=int, default=int(os.environ.get("ACTION_REPEAT", "4")))
    parser.add_argument("--history-length", type=int, default=int(os.environ.get("HISTORY_LENGTH", "15")))
    parser.add_argument("--idle-hz", type=float, default=float(os.environ.get("IDLE_HZ", "50.0")))
    parser.add_argument("--perception-hz", type=float, default=float(os.environ.get("PERCEPTION_HZ", "5.0")))
    parser.add_argument("--render-hz", type=float, default=float(os.environ.get("DASHBOARD_RENDER_HZ", "15.0")))
    parser.add_argument("--max-catchup-steps", type=int, default=int(os.environ.get("MAX_CATCHUP_STEPS", "1")))

    parser.add_argument(
        "--dynamic-obstacles",
        action=argparse.BooleanOptionalAction,
        default=env_bool("ENABLE_DYNAMIC_OBSTACLES", False),
        help="Enable moving obstacles in the random-map environment.",
    )
    parser.add_argument(
        "--mujoco-viewer",
        action=argparse.BooleanOptionalAction,
        default=env_bool("ENABLE_MUJOCO_VIEWER", False),
        help="Open MuJoCo's native viewer when supported by the render mode.",
    )
    parser.add_argument(
        "--debug-timing",
        action=argparse.BooleanOptionalAction,
        default=env_bool("DEBUG_TIMING", False),
        help="Print timing diagnostics from navigation and dashboard loops.",
    )
    parser.add_argument(
        "--show-landmark-debug",
        action=argparse.BooleanOptionalAction,
        default=env_bool("SHOW_LANDMARK_DEBUG", False),
        help="Print ground-truth landmark positions after map reset.",
    )

    parser.add_argument(
        "--llm",
        action=argparse.BooleanOptionalAction,
        default=not env_bool("DISABLE_LLM", False),
        help="Enable natural-language commands through the configured chat endpoint.",
    )
    parser.add_argument("--llm-base-url", default=os.environ.get("LLM_BASE_URL", "https://api.siliconflow.cn/v1"))
    parser.add_argument("--llm-model", default=os.environ.get("LLM_MODEL", "Pro/zai-org/GLM-4.7"))
    parser.add_argument("--llm-timeout-s", type=float, default=float(os.environ.get("LLM_TIMEOUT_SECONDS", "60.0")))

    parser.add_argument("--cpu-threads", type=int, default=int(os.environ.get("DEMO_CPU_THREADS", "1")))
    parser.add_argument("--mujoco-gl", default=os.environ.get("MUJOCO_GL", "egl"))
    parser.add_argument("--matplotlib-cache-dir", default=os.environ.get("MPLCONFIGDIR", "/tmp/matplotlib"))
    parser.add_argument("--log-level", default=os.environ.get("LOG_LEVEL", "INFO"))
    return parser


def configure_logging(log_level):
    level_name = str(log_level).upper()
    level = getattr(logging, level_name, logging.INFO)
    logging.basicConfig(level=level, format="[%(levelname)s] %(name)s: %(message)s")
    logging.getLogger("matplotlib").setLevel(logging.WARNING)


def configure_runtime_environment(args):
    thread_count = str(max(1, int(args.cpu_threads)))
    for name in (
        "OMP_NUM_THREADS",
        "MKL_NUM_THREADS",
        "OPENBLAS_NUM_THREADS",
        "NUMEXPR_NUM_THREADS",
        "VECLIB_MAXIMUM_THREADS",
    ):
        os.environ[name] = thread_count
    os.environ["MPLCONFIGDIR"] = args.matplotlib_cache_dir
    # The default GLFW backend can fail under Wayland and leave MuJoCo offscreen
    # frames black. Set this before importing modules that import mujoco.
    os.environ["MUJOCO_GL"] = args.mujoco_gl


def dashboard_log(env, message):
    if hasattr(env, "append_dashboard_log"):
        env.append_dashboard_log(message)


def load_local_env(project_root, env_file=".env"):
    env_path = env_file if os.path.isabs(env_file) else os.path.join(project_root, env_file)
    if not os.path.exists(env_path):
        return
    with open(env_path, "r", encoding="utf-8") as f:
        for raw_line in f:
            line = raw_line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            key, value = line.split("=", 1)
            key = key.strip()
            value = value.strip().strip('"').strip("'")
            if key:
                os.environ.setdefault(key, value)


def resolve_project_path(project_root, env_name, default_relative_path):
    raw_path = os.environ.get(env_name, default_relative_path)
    return resolve_path(project_root, raw_path)


def resolve_path(project_root, raw_path):
    if os.path.isabs(raw_path):
        return raw_path
    return os.path.join(project_root, raw_path)


def ensure_required_files(paths):
    missing = [path for path in paths if not os.path.exists(path)]
    if missing:
        missing_text = "\n".join(f"  - {path}" for path in missing)
        raise FileNotFoundError(f"Required demo assets are missing:\n{missing_text}")


def print_landmark_debug(env):
    debug_pos = env.get_landmark_positions_debug()
    LOGGER.debug("Ground-truth landmark positions:")
    for lm_id, pos in debug_pos.items():
        LOGGER.debug("  %s: (%.2f, %.2f)", lm_id, pos[0], pos[1])


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
    project_root = os.path.dirname(os.path.abspath(__file__))
    env_file = parse_env_file_argument()
    load_local_env(project_root, env_file)
    parser = build_arg_parser()
    args = parser.parse_args()
    configure_logging(args.log_level)
    configure_runtime_environment(args)

    import matplotlib.pyplot as plt

    from agent_env import AgentVisualEnv
    from memory import SpatialMemory, TopologicalMap
    from skills import NavigationSkill, PerceptionSkill, FrontierExplorationSkill
    from llm_brain import RobotAgent
    from realtime_runner import RealtimeRunner

    model_xml = resolve_path(project_root, args.robot_model_xml)
    low_level_policy_path = resolve_path(project_root, args.low_level_policy)
    sac_model_path = resolve_path(project_root, args.sac_model)
    memory_dir = resolve_path(project_root, args.memory_dir)
    ensure_required_files([model_xml, low_level_policy_path, sac_model_path])

    print("=== 启动 MuJoCo 导航演示 ===")
    LOGGER.info("CPU numeric library threads set to %s", max(1, int(args.cpu_threads)))
    LOGGER.info("MuJoCo GL backend: %s", args.mujoco_gl)

    env_kwargs = {
        "model_path": model_xml,
        "low_level_policy_path": low_level_policy_path,
        "render_mode": args.render_mode,
        "render_decimation": args.render_decimation,
        "action_repeat": args.action_repeat,
        "history_length": args.history_length,
        "enable_dynamic_obstacles": args.dynamic_obstacles,
        "enable_mujoco_viewer": args.mujoco_viewer,
    }
    env = AgentVisualEnv(**env_kwargs)
    env.debug_timing = args.debug_timing
    if args.debug_timing:
        LOGGER.info("Timing diagnostics enabled: watch [PERF NAV] / [PERF UI] / [JUMP NAV]")
    realtime_runner = RealtimeRunner(
        env,
        idle_hz=args.idle_hz,
        perception_hz=args.perception_hz,
        render_hz=args.render_hz,
        max_catchup_steps=args.max_catchup_steps,
    )
    command_window = None if env_kwargs["render_mode"] == "qt_dashboard" else CommandWindow()

    print("\n[系统] 正在生成随机迷宫并放置地标...")
    env.reset()

    if args.show_landmark_debug:
        print_landmark_debug(env)

    memory = SpatialMemory(save_dir=memory_dir)
    topo_map = TopologicalMap(fingerprint_radius_m=3.0, fingerprint_size=8, match_threshold=0.85)
    nav_skill = NavigationSkill(env, sac_model_path)
    percept_skill = PerceptionSkill(env, memory)
    explore_skill = FrontierExplorationSkill(env, nav_skill, percept_skill, topo_map=topo_map)
    agent = RobotAgent(
        nav_skill,
        memory,
        explore_skill,
        topo_map=topo_map,
        llm_base_url=args.llm_base_url,
        llm_model=args.llm_model,
        llm_timeout_s=args.llm_timeout_s,
        llm_enabled=args.llm,
    )

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
            if args.show_landmark_debug:
                print_landmark_debug(env)
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
