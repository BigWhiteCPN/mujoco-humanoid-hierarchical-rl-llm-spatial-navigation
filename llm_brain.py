# --- START OF FILE llm_brain.py ---
import os
import json
import time
import httpx
import threading
from contextlib import nullcontext
from queue import Queue
import numpy as np          
from openai import OpenAI

# 🌟 引入心智模拟器
from skills import MentalSimulator

class RobotAgent:
    def __init__(self, nav_skill, memory, explore_skill, topo_map=None):
        self.nav_skill = nav_skill
        self.memory = memory
        self.explore_skill = explore_skill
        self.topo_map = topo_map

        # 🌟 初始化具身心智预演器 (给 LLM 一个强大的空间“想象力”)
        self.mental_simulator = MentalSimulator(self.nav_skill.env, robot_radius_m=0.4)

        self.client = OpenAI(
            base_url="https://api.siliconflow.cn/v1",
            api_key=os.environ.get("SILICONFLOW_API_KEY"),
            http_client=httpx.Client(trust_env=False)
        )
        self.model_name = "Pro/zai-org/GLM-4.7"

        self.tools =[
            {
                "type": "function",
                "function": {
                    "name": "navigate_to_place",
                    "description": "导航到记忆库中已知的地点。仅在你需要单独前往某个已知位置时使用。",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "place_name": {
                                "type": "string",
                                "description": "目标地点的名称，例如：大门、会议室、老板办公室、茶水间"
                            }
                        },
                        "required":["place_name"]
                    }
                }
            },
            {
                "type": "function",
                "function": {
                    "name": "explore_environment",
                    "description": (
                        "前往地图中未知区域探索。如果人类要你寻找某个具体地点，必须填入 target_place。"
                        "探索完成后，系统会自动按 target_place 列表的顺序依次前往每个目标。"
                        "因此调用此工具后，不需要再额外调用 navigate_to_place。"
                    ),
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "target_place": {
                                "type": "array",
                                "items": {"type": "string"},
                                "description": (
                                    "要定向寻找的具体地点名称列表，按访问顺序排列。"
                                    "例如['老板办公室', '会议室'] 表示先去老板办公室再去会议室。"
                                    "如果是随便逛逛，留空。"
                                )
                            },
                            "direction_hint": {
                                "type": "string",
                                "description": (
                                    "基于你的常识推理，猜测目标可能在哪个方位。"
                                    "可选值：north, south, east, west, northeast, northwest, southeast, southwest。"
                                    "如果无法推断，留空即可。"
                                )
                            }
                        }
                    }
                }
            },
            {
                "type": "function",
                "function": {
                    "name": "recall_spatial_memory",
                    "description": (
                        "回忆空间记忆：报告我到过哪些地方、哪些地方去了多次（可能是关键路口或兴趣点）。"
                        "当用户问'你去过哪里''你还记得路吗''我们走过哪些地方'时调用。"
                    ),
                    "parameters": {
                        "type": "object",
                        "properties": {}
                    }
                }
            }
        ]

        self._llm_result_queue = Queue()

    def _log_ui(self, message):
        env = getattr(self.nav_skill, "env", None)
        if env is None or not hasattr(env, "append_dashboard_log"):
            return
        try:
            env.append_dashboard_log(message)
        except Exception:
            pass

    @staticmethod
    def _shorten_for_ui(message, max_chars=220):
        text = " ".join(str(message).split())
        if len(text) <= max_chars:
            return text
        return text[:max_chars - 3] + "..."

    def _execute_tool_call(self, func_name, args):
        """统一执行工具调用，返回工具结果字符串"""
        if getattr(self.nav_skill.env, "_shutdown_requested", False):
            return "任务已被用户中断。"

        if func_name == "explore_environment":
            targets = args.get("target_place", [])
            if isinstance(targets, str): targets = [targets]
            if not targets: targets = None

            direction_hint = args.get("direction_hint", None)
            if direction_hint and isinstance(direction_hint, str) and direction_hint.strip():
                direction_hint = direction_hint.strip()
            else:
                direction_hint = None

            print(f"[身体 🦾] 收到大脑指令：开始定向寻找目标序列 {targets}，方位偏好: {direction_hint}")

            journey_log = self.explore_skill.execute(
                max_rounds=20,
                target_place=targets,
                direction_hint=direction_hint
            )

            final_action_summary = ""
            if targets:
                missing_targets =[]
                for t in targets:
                    coords = self.memory.get_location_by_meaning(t)
                    if coords:
                        landmark_id = self.memory.get_feature_id_by_meaning(t) if hasattr(self.memory, "get_feature_id_by_meaning") else None
                        print(f"[身体 🦾] 按顺序前往: {t} ({coords[0]:.1f}, {coords[1]:.1f})...")
                        success, dist = self.nav_skill.go_to(
                            coords[0],
                            coords[1],
                            max_steps=2000,
                            success_dist=0.65,
                            track_landmark_id=landmark_id,
                        )
                        if success:
                            final_action_summary += f"\n✅ 已按顺序成功到达【{t}】。"
                        else:
                            final_action_summary += f"\n⚠️ 尝试前往【{t}】但未能完全到达（距离{dist:.1f}米）。"
                    else:
                        missing_targets.append(t)

                if missing_targets:
                    final_action_summary += f"\n❌ 未能找到以下地点：{missing_targets}"

            tool_result = (
                f"【底层身体传回的探索日记】:\n{journey_log}\n"
                f"\n【自动导航结果】:{final_action_summary}\n"
                f"\n⚠️ 注意：系统已按照 target_place 列表的顺序自动完成了所有导航，"
                f"你无需再调用 navigate_to_place。请直接用自然语言向用户汇报结果。"
            )
            return tool_result

        # ==========================================
        # 🌟 动作 2：直接导航（加入 A* 心智预演拦截机制）
        # ==========================================
        elif func_name == "navigate_to_place":
            place_name = args.get('place_name')
            print(f"[大脑 🧠] 正在脑海中预演前往【{place_name}】的路径...")

            coords = self.memory.get_location_by_meaning(place_name)
            if not coords:
                return f"错误：记忆库里找不到 {place_name} 的具体坐标。建议先探索环境。"
            landmark_id = self.memory.get_feature_id_by_meaning(place_name) if hasattr(self.memory, "get_feature_id_by_meaning") else None
            
            robot_pos = self.nav_skill.env.data.xpos[self.nav_skill.env.robot_base_body_id][:2]
            
            # 🌟 核心：启动心智模拟器进行预演！
            sim_result = self.mental_simulator.simulate_path(
                start_pos=robot_pos, 
                goal_pos=np.array(coords)
            )
            
            # 🌟 如果预演发现路被堵死，直接拦截！不执行底层强化学习 SAC
            if not sim_result["feasible"]:
                print(f"[内心独白 💭] 哎呀，拓扑预演失败：{sim_result['reason']}")
                return (f"【路径预演失败】：物理空间阻断，无法前往 {place_name}。原因：{sim_result['reason']}。"
                        f"请使用自然语言向人类解释路被堵死了，不要让机器傻站着。")
            
            # 🌟 预演成功，放行，执行真实物理导航
            print(f"[内心独白 💭] 预演成功！路程约 {sim_result['path_length_m']:.1f}米，预计 {sim_result['estimated_steps']} 步。开始注入底层物理执行...")
            
            success, dist = self.nav_skill.go_to(
                coords[0],
                coords[1],
                success_dist=0.65,
                track_landmark_id=landmark_id,
            )
            
            if success:
                # 把预演得到的数据返回给 LLM，让 LLM 的回答显得有距离感知
                return (f"导航任务完美完成！【预演辅助数据：本次路径长 {sim_result['path_length_m']:.1f}米，大约行驶了 {sim_result['estimated_steps']}步】。"
                        f"目前位置：{place_name} 前方。请在回复时，像人类一样顺便提及这段路程的距离和步数感觉。")
            else:
                return f"导航失败，虽然脑海中预演通畅，但底层物理引擎执行时遭遇动态意外（距离目标{dist:.1f}米）。"

        elif func_name == "recall_spatial_memory":
            spatial_report = self.memory.get_spatial_report(
                self.nav_skill.env.data.xpos[self.nav_skill.env.robot_base_body_id][:2]
            )
            topo_report = self.topo_map.get_explored_summary() if self.topo_map else "拓扑记忆不可用。"
            revisit_info = ""
            if self.topo_map:
                hotspots = self.topo_map.get_revisit_candidates(min_visits=2)
                if hotspots:
                    revisit_info = "\n\n🔥 多次访问的热点位置（可能是关键路口）：\n"
                    for i, node in enumerate(hotspots):
                        pos = node['pos']
                        revisit_info += (f"  热点{i+1}: ({pos[0]:.1f}, {pos[1]:.1f})，"
                                         f"访问 {node['visit_count']} 次\n")
            return f"【空间记忆报告】\n{spatial_report}\n\n【拓扑探索记录】\n{topo_report}{revisit_info}"

        else:
            return f"未知工具: {func_name}"

    def _run_llm_inference(self, messages, use_tools=True):
        try:
            if use_tools:
                response = self.client.chat.completions.create(
                    model=self.model_name,
                    messages=messages,
                    tools=self.tools,
                    tool_choice="auto"
                )
            else:
                response = self.client.chat.completions.create(
                    model=self.model_name,
                    messages=messages
                )
            self._llm_result_queue.put(("ok", response))
        except Exception as e:
            self._llm_result_queue.put(("error", e))

    def _wait_for_llm_with_rendering(self, messages, use_tools=True):
        thread = threading.Thread(
            target=self._run_llm_inference,
            args=(messages, use_tools),
            daemon=True
        )
        thread.start()

        env = self.nav_skill.env
        last_perception_update = 0.0
        progress_callback = getattr(env, "_qt_progress_callback", None)
        lock = getattr(env, "_qt_env_lock", None)
        lock_context = lock if lock is not None else nullcontext()
        try:
            idle_dt = float(env.locomotion_controller.cfg.sim_config.decimation) * float(env.model.opt.timestep)
        except Exception:
            idle_dt = 0.02
        idle_dt = float(np.clip(idle_dt, 0.01, 0.04))
        next_idle_time = time.perf_counter()
        while thread.is_alive():
            if getattr(env, "_shutdown_requested", False):
                self._log_ui("系统: 已取消等待大脑响应。")
                raise RuntimeError("任务已被用户中断")
            now = time.time()
            perf_now = time.perf_counter()
            try:
                should_update = now - last_perception_update > 0.1
                should_idle = perf_now >= next_idle_time
                if should_idle or should_update:
                    with lock_context:
                        if should_idle:
                            self._idle_step_env(env)
                            next_idle_time += idle_dt
                            if perf_now - next_idle_time > idle_dt:
                                next_idle_time = perf_now + idle_dt
                        if should_update:
                            if hasattr(env, "_update_perception"):
                                env._update_perception()
                            if hasattr(env, "render"):
                                env.render()
                            last_perception_update = now
                if should_update:
                    if progress_callback is not None:
                        progress_callback()
            except Exception:
                pass
            time.sleep(0.005)
            thread.join(timeout=0)

        thread.join()

        status, result = self._llm_result_queue.get()
        if status == "error":
            raise result
        return result

    @staticmethod
    def _idle_step_env(env):
        """Keep the robot physically alive while the LLM is thinking."""
        if hasattr(env, "locomotion_controller"):
            env.locomotion_controller.step(0.0, 0.0, 0.0)
        if hasattr(env, "robot_base_body_id") and hasattr(env, "grid_map"):
            robot_pos = env.data.xpos[env.robot_base_body_id][:2]
            if hasattr(env.grid_map, "update_visited_footprint"):
                env.grid_map.update_visited_footprint(robot_pos)

    def chat_and_execute(self, user_command):
        print(f"\n[人类 🗣️] 指令: {user_command}")
        self._log_ui(f"用户: {user_command}")

        robot_pos = self.nav_skill.env.data.xpos[self.nav_skill.env.robot_base_body_id][:2]
        spatial_report = self.memory.get_spatial_report(robot_pos)
        unexplored_hint = self._get_unexplored_directions_hint()

        # 拓扑记忆摘要
        topo_hint = ""
        if self.topo_map and self.topo_map.nodes:
            n_nodes = len(self.topo_map.nodes)
            hotspots = self.topo_map.get_revisit_candidates(min_visits=2)
            topo_hint = f"【拓扑记忆】：已探索 {n_nodes} 个不同位置"
            if hotspots:
                topo_hint += f"，其中 {len(hotspots)} 个位置被多次访问（可能是关键路口）。"
            else:
                topo_hint += "。"

        system_prompt = (
            "你是一个具备高级人类空间逻辑思维的具身探索机器人。\n"
            f"【当前空间感知状态】:\n{spatial_report}\n\n"
            f"【未探索区域分析】:\n{unexplored_hint}\n\n"
            f"{topo_hint}\n\n"
            "【行为逻辑】：\n"
            "1. 理解多步指令：如果人类要求'先去A再去B'，检查A和B是否都在感知状态中。\n"
            "2. 混合决策：如果有地点不在记忆里，调用 explore_environment 寻找，按用户要求的访问顺序排列。\n"
            "3. ⚠️ explore_environment 会在探索完成后自动按顺序依次导航。调用 explore_environment 后，绝对不要再调用 navigate_to_place。\n"
            "4. 只有在用户仅要求去一个**已知地点**时，才使用 navigate_to_place。\n"
            "5. ⚠️ 调用 explore_environment 时，请根据常识和【未探索区域分析】，给出 direction_hint。\n"
            "6. 动作串联：当工具返回执行成功的日记后，结合日记内容用第一人称自然汇报。\n"
            "7. ⚠️ 绝不允许在任务完成前轻易放弃！\n"
            "8. 当 navigate_to_place 返回了路程米数和步数时，要在回答里表现出来，增加你的真实具身感！"
        )

        messages =[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_command}
        ]

        print("[大脑 🧠] 正在进行空间推理与认知推演...")
        self._log_ui("思考提示: 读取空间记忆、未探索区域和拓扑热点，判断是否需要调用导航/探索工具。")

        max_tool_rounds = 5
        for tool_round in range(max_tool_rounds):
            self._log_ui(f"大脑: 第{tool_round + 1}轮空间推理中...")
            response = self._wait_for_llm_with_rendering(messages, use_tools=True)
            response_msg = response.choices[0].message
            messages.append(response_msg)

            if not response_msg.tool_calls:
                answer = response_msg.content
                self._log_ui(f"结果: {self._shorten_for_ui(answer)}")
                return answer

            for tool_call in response_msg.tool_calls:
                if getattr(self.nav_skill.env, "_shutdown_requested", False):
                    return "任务已被用户中断。"
                func_name = tool_call.function.name
                tool_call_id = tool_call.id
                args = json.loads(tool_call.function.arguments) if tool_call.function.arguments else {}

                print(f"[大脑 🧠] 第{tool_round + 1}轮发起技能调用: {func_name}({args})")
                self._log_ui(f"工具调用: {func_name}({self._shorten_for_ui(args, max_chars=120)})")

                tool_result = self._execute_tool_call(func_name, args)
                self._log_ui(f"工具结果: {self._shorten_for_ui(tool_result)}")

                messages.append({
                    "role": "tool",
                    "content": tool_result,
                    "tool_call_id": tool_call_id
                })

        print("[大脑 🧠] 认知推演结束，生成最终意识流汇报...")
        if getattr(self.nav_skill.env, "_shutdown_requested", False):
            return "任务已被用户中断。"
        self._log_ui("大脑: 工具轮次结束，生成最终汇报...")
        final_response = self._wait_for_llm_with_rendering(messages, use_tools=False)
        answer = final_response.choices[0].message.content
        self._log_ui(f"结果: {self._shorten_for_ui(answer)}")
        return answer

    def _get_unexplored_directions_hint(self):
        env = self.nav_skill.env
        if not hasattr(env, 'grid_map'):
            return "无法获取地图数据。"

        grid_map = env.grid_map
        prob_grid = 1.0 - 1.0 / (1.0 + np.exp(grid_map.grid))
        unknown_mask = (prob_grid >= 0.4) & (prob_grid <= 0.6)

        h, w = unknown_mask.shape
        mid_r, mid_c = h // 2, w // 2

        quadrants = {
            "东北 (northeast)": unknown_mask[:mid_r, mid_c:],
            "西北 (northwest)": unknown_mask[:mid_r, :mid_c],
            "东南 (southeast)": unknown_mask[mid_r:, mid_c:],
            "西南 (southwest)": unknown_mask[mid_r:, :mid_c],
        }

        total_unknown = int(np.sum(unknown_mask))
        if total_unknown == 0:
            return "地图已基本完全探索，各方向均无大片未知区域。"

        results =[]
        for name, region in quadrants.items():
            count = int(np.sum(region))
            pct = count / max(total_unknown, 1) * 100
            if pct > 5:  
                results.append(f"- {name}：约{pct:.0f}%的未知区域集中在这里")

        if not results:
            return "未知区域分布较均匀，无明显方位偏好。"

        results.sort(key=lambda x: float(x.split("约")[1].split("%")[0]), reverse=True)
        return "以下方位存在较多未探索区域：\n" + "\n".join(results)
