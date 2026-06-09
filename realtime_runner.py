import time


class RealtimeRunner:
    """Small realtime scheduler for idle locomotion, perception and dashboard refresh."""

    def __init__(
        self,
        env,
        idle_hz=50.0,
        perception_hz=5.0,
        render_hz=15.0,
        max_catchup_steps=3,
    ):
        self.env = env
        self.idle_dt = 1.0 / idle_hz
        self.perception_dt = 1.0 / perception_hz
        self.render_dt = 1.0 / render_hz
        self.max_catchup_steps = max_catchup_steps
        now = time.perf_counter()
        self._next_idle_time = now
        self._last_perception_time = 0.0
        self._last_render_time = 0.0

    def spin_until(self, done_fn, poll_sleep=0.001):
        while not done_fn():
            self.step_idle()
            time.sleep(poll_sleep)

    def step_idle(self):
        now = time.perf_counter()
        steps = 0
        while now >= self._next_idle_time and steps < self.max_catchup_steps:
            self._idle_step_env()
            self._next_idle_time += self.idle_dt
            steps += 1

        # If rendering or Python stalls badly, drop missed idle slots instead of blocking the UI.
        if now - self._next_idle_time > self.idle_dt * self.max_catchup_steps:
            self._next_idle_time = now + self.idle_dt

        if now - self._last_perception_time >= self.perception_dt:
            if hasattr(self.env, "_update_perception"):
                self.env._update_perception()
            self._last_perception_time = now

        if now - self._last_render_time >= self.render_dt:
            if hasattr(self.env, "render"):
                self.env.render()
            self._last_render_time = now

    def _idle_step_env(self):
        if hasattr(self.env, "locomotion_controller"):
            self.env.locomotion_controller.step(0.0, 0.0, 0.0)
        if hasattr(self.env, "robot_base_body_id") and hasattr(self.env, "grid_map"):
            robot_pos = self.env.data.xpos[self.env.robot_base_body_id][:2]
            if hasattr(self.env.grid_map, "update_visited_footprint"):
                self.env.grid_map.update_visited_footprint(robot_pos)
