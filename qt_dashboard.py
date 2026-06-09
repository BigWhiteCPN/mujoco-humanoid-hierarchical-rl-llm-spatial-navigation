import queue
import threading
import traceback
import time

import numpy as np
from PySide6 import QtCore, QtGui, QtWidgets


class QtAgentDashboard(QtWidgets.QWidget):
    def __init__(self, env, realtime_runner, command_handler, close_handler):
        super().__init__()
        self.env = env
        self.realtime_runner = realtime_runner
        self.command_handler = command_handler
        self.close_handler = close_handler
        self._result_queue = queue.Queue()
        self._frame_queue = queue.Queue(maxsize=2)
        self._env_lock = threading.RLock()
        self._worker = None
        self._busy = False
        self._closing = False
        self._close_requested = False
        self._close_request_time = 0.0
        self._last_log_text = ""
        self._last_progress_frame_time = 0.0
        self._progress_requested = False
        self._last_frame = None
        self._debug_last_ui_log_time = 0.0
        self._debug_lock_skip_count = 0
        setattr(self.env, "_qt_env_lock", self._env_lock)
        setattr(self.env, "_qt_render_thread_id", threading.get_ident())
        setattr(self.env, "_qt_progress_callback", self._queue_progress_frame)
        setattr(self.env, "_shutdown_requested", False)
        self.env._dashboard_render_size = (360, 480)
        self.env._vision_size = self.env._dashboard_render_size
        self.env._dashboard_min_interval = 1.0 / 24.0
        self.env._fast_scene_interval = 1.0 / 14.0
        self.env._vision_min_interval = 1.0 / 8.0
        self.env._vision_display_min_interval = 1.0 / 6.0
        self.env._vision_detect_min_interval = 2.0
        self.env._command_vision_min_interval = 1.60

        self.setWindowTitle("Agent System Dashboard")
        self.resize(1280, 820)
        self._build_ui()

        self.frame_timer = QtCore.QTimer(self)
        self.frame_timer.setTimerType(QtCore.Qt.PreciseTimer)
        self.frame_timer.setInterval(33)
        self.frame_timer.timeout.connect(self._tick)
        self.frame_timer.start()

        self.result_timer = QtCore.QTimer(self)
        self.result_timer.setInterval(50)
        self.result_timer.timeout.connect(self._poll_command_result)
        self.result_timer.start()

    def _build_ui(self):
        self.image_label = QtWidgets.QLabel()
        self.image_label.setMinimumSize(960, 540)
        self.image_label.setAlignment(QtCore.Qt.AlignCenter)
        self.image_label.setStyleSheet("background: #111;")

        self.log_view = QtWidgets.QPlainTextEdit()
        self.log_view.setReadOnly(True)
        self.log_view.setMaximumBlockCount(200)
        self.log_view.setMinimumWidth(340)

        self.command_input = QtWidgets.QLineEdit()
        self.command_input.setPlaceholderText("输入命令后回车")
        self.command_input.returnPressed.connect(self._submit_command)

        self.send_button = QtWidgets.QPushButton("发送")
        self.send_button.clicked.connect(self._submit_command)

        self.status_label = QtWidgets.QLabel("就绪")
        self.status_label.setMinimumWidth(120)

        input_row = QtWidgets.QHBoxLayout()
        input_row.addWidget(self.command_input, 1)
        input_row.addWidget(self.send_button)
        input_row.addWidget(self.status_label)

        left = QtWidgets.QVBoxLayout()
        left.addWidget(self.image_label, 1)
        left.addLayout(input_row)

        right = QtWidgets.QVBoxLayout()
        title = QtWidgets.QLabel("运行日志")
        title.setStyleSheet("font-weight: 600;")
        right.addWidget(title)
        right.addWidget(self.log_view, 1)

        root = QtWidgets.QHBoxLayout(self)
        root.addLayout(left, 4)
        root.addLayout(right, 1)

    def _tick(self):
        if self._closing:
            return
        try:
            if not self._busy:
                with self._env_lock:
                    self.realtime_runner.step_idle()
                    self._handle_forced_vision_request()
                    frame = self.env.get_fast_dashboard_frame_rgb()
                self._show_frame(frame)
            else:
                self._show_busy_frame()
            self._sync_log_view()
        except KeyboardInterrupt:
            self.close()
        except Exception as exc:
            self._append_text(f"[UI] 刷新失败: {exc}")

    def _queue_progress_frame(self, force=False):
        if self._closing:
            return
        self._progress_requested = True

    def _handle_forced_vision_request(self):
        if not getattr(self.env, "_qt_force_vision_requested", False):
            return
        try:
            self.env.detect_visual_landmarks(force=True)
        finally:
            self.env._qt_force_vision_done_time = time.time()
            self.env._qt_force_vision_requested = False
            self.env._qt_force_vision_waiting = False

    def _show_busy_frame(self):
        now = time.time()
        if now - self._last_progress_frame_time < 1.0 / 24.0:
            return
        self._last_progress_frame_time = now
        lock_t0 = time.perf_counter()
        acquired = self._env_lock.acquire(timeout=0.018)
        if not acquired:
            self._debug_lock_skip_count += 1
            if bool(getattr(self.env, "debug_timing", False)) and now - self._debug_last_ui_log_time > 1.0:
                print(
                    f"[PERF UI] busy_frame lock skipped {self._debug_lock_skip_count} times in last window",
                    flush=True,
                )
                self._debug_lock_skip_count = 0
                self._debug_last_ui_log_time = now
            if self._last_frame is not None:
                self._show_frame(self._last_frame)
            return
        try:
            lock_wait_ms = (time.perf_counter() - lock_t0) * 1000.0
            render_t0 = time.perf_counter()
            self._handle_forced_vision_request()
            frame = self.env.get_fast_dashboard_frame_rgb(force=False)
            render_ms = (time.perf_counter() - render_t0) * 1000.0
            if bool(getattr(self.env, "debug_timing", False)) and (render_ms > 45.0 or lock_wait_ms > 12.0):
                print(f"[PERF UI] busy_frame wait={lock_wait_ms:.1f}ms render={render_ms:.1f}ms", flush=True)
            self._progress_requested = False
        finally:
            self._env_lock.release()
        self._show_frame(frame)

    def _show_frame(self, frame):
        if frame is None:
            return
        frame = np.ascontiguousarray(frame)
        height, width, channels = frame.shape
        image = QtGui.QImage(
            frame.data,
            width,
            height,
            channels * width,
            QtGui.QImage.Format_RGB888,
        ).copy()
        pixmap = QtGui.QPixmap.fromImage(image)
        pixmap = pixmap.scaled(
            self.image_label.size(),
            QtCore.Qt.KeepAspectRatio,
            QtCore.Qt.FastTransformation,
        )
        self.image_label.setPixmap(pixmap)
        self._last_frame = frame

    def _sync_log_view(self):
        lines = getattr(self.env, "_dashboard_log_lines", [])
        text = "\n".join(lines)
        if text == self._last_log_text:
            return
        self._last_log_text = text
        self.log_view.setPlainText(text)
        scrollbar = self.log_view.verticalScrollBar()
        scrollbar.setValue(scrollbar.maximum())

    def _append_text(self, text):
        self.log_view.appendPlainText(text)
        scrollbar = self.log_view.verticalScrollBar()
        scrollbar.setValue(scrollbar.maximum())

    def _submit_command(self):
        text = self.command_input.text().strip()
        if not text or self._busy:
            return
        self.command_input.clear()
        self._busy = True
        self._close_requested = False
        setattr(self.env, "_shutdown_requested", False)
        self._progress_requested = True
        setattr(self.env, "_qt_command_busy", True)
        self.command_input.setEnabled(False)
        self.send_button.setEnabled(False)
        self.status_label.setText("执行中")
        if hasattr(self.env, "append_dashboard_log"):
            self.env.append_dashboard_log(f"用户: {text}")

        self._worker = threading.Thread(target=self._run_command, args=(text,), daemon=True)
        self._worker.start()

    def _run_command(self, text):
        try:
            should_exit = bool(self.command_handler(text))
            self._result_queue.put(("ok", should_exit, None))
        except Exception:
            self._result_queue.put(("error", False, traceback.format_exc()))

    def _poll_command_result(self):
        try:
            status, should_exit, details = self._result_queue.get_nowait()
        except queue.Empty:
            return

        self._busy = False
        setattr(self.env, "_qt_command_busy", False)
        self.command_input.setEnabled(True)
        self.send_button.setEnabled(True)
        self.command_input.setFocus()
        self.status_label.setText("就绪")

        if status == "error":
            self._append_text(details)
            if hasattr(self.env, "append_dashboard_log"):
                self.env.append_dashboard_log("命令执行失败，详见右侧日志。")
            if self._close_requested:
                self.close()
            return
        if should_exit:
            self.close()
        elif self._close_requested:
            self.close()

    def closeEvent(self, event):
        if self._closing:
            event.accept()
            return
        worker = self._worker
        if worker is not None and worker.is_alive():
            setattr(self.env, "_shutdown_requested", True)
            if not self._close_requested:
                self._close_requested = True
                self._close_request_time = time.time()
                self.status_label.setText("停止中")
                self._append_text("[UI] 已请求停止当前任务，等待导航线程退出后关闭。")
                if hasattr(self.env, "append_dashboard_log"):
                    self.env.append_dashboard_log("系统: 正在停止当前任务并关闭界面。")
            elif time.time() - self._close_request_time > 3.0:
                self._append_text("[UI] 仍在等待后台任务退出；请稍等几秒。")
                self._close_request_time = time.time()
            QtCore.QTimer.singleShot(250, self.close)
            event.ignore()
            return
        self._closing = True
        setattr(self.env, "_shutdown_requested", True)
        self.frame_timer.stop()
        self.result_timer.stop()
        try:
            with self._env_lock:
                self.close_handler()
        finally:
            if hasattr(self.env, "_qt_progress_callback"):
                try:
                    delattr(self.env, "_qt_progress_callback")
                except Exception:
                    pass
            if hasattr(self.env, "_qt_env_lock"):
                try:
                    delattr(self.env, "_qt_env_lock")
                except Exception:
                    pass
            if hasattr(self.env, "_qt_render_thread_id"):
                try:
                    delattr(self.env, "_qt_render_thread_id")
                except Exception:
                    pass
            event.accept()
            QtWidgets.QApplication.quit()


def run_qt_dashboard(env, realtime_runner, command_handler, close_handler):
    app = QtWidgets.QApplication.instance()
    if app is None:
        app = QtWidgets.QApplication([])
    window = QtAgentDashboard(env, realtime_runner, command_handler, close_handler)
    window.show()
    return app.exec()
