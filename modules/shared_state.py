# 模块提供用于处理日期和时间的类 https://docs.python.org/zh-cn/3/library/datetime.html
import datetime
# 日志记录工具 https://docs.python.org/zh-cn/3/library/logging.html
import logging
import threading
import time

from modules import errors, shared, devices
# 对类型提示的支持 https://docs.python.org/zh-cn/3/library/typing.html#typing.Optional
from typing import Optional
# __name__ 模块的完整限定名称
# 返回具有指定 name 的日志记录器
log = logging.getLogger(__name__)


class State:
    skipped = False
    interrupted = False
    job = ""
    job_no = 0
    job_count = 0
    processing_has_refined_job_count = False
    job_timestamp = '0'
    sampling_step = 0
    sampling_steps = 0
    current_latent = None
    current_image = None
    current_image_sampling_step = 0
    id_live_preview = 0
    textinfo = None
    time_start = None
    server_start = None
    # Event 这是线程之间通信的最简单机制之一：一个线程发出事件信号，而其他线程等待该信号。
    _server_command_signal = threading.Event()
    _server_command: Optional[str] = None

    def __init__(self):
        # 记录服务器启动时间戳
        self.server_start = time.time()

    @property
    def need_restart(self) -> bool:
        # Compatibility getter for need_restart.
        return self.server_command == "restart"

    @need_restart.setter
    def need_restart(self, value: bool) -> None:
        # Compatibility setter for need_restart.
        if value:
            self.server_command = "restart"

    @property
    def server_command(self):
        return self._server_command

    @server_command.setter
    def server_command(self, value: Optional[str]) -> None:
        """
        Set the server command to `value` and signal that it's been set.
        """
        self._server_command = value
        self._server_command_signal.set()

    def wait_for_server_command(self, timeout: Optional[float] = None) -> Optional[str]:
        """
        Wait for server command to get set; return and clear the value and signal.
        """
        if self._server_command_signal.wait(timeout):
            self._server_command_signal.clear()
            req = self._server_command
            self._server_command = None
            return req
        return None

    def request_restart(self) -> None:
        self.interrupt()
        self.server_command = "restart"
        log.info("Received restart request")

    def skip(self):
        self.skipped = True
        log.info("Received skip request")

    def interrupt(self):
        self.interrupted = True
        log.info("Received interrupt request")

    def nextjob(self):
        if shared.opts.live_previews_enable and shared.opts.show_progress_every_n_steps == -1:
            self.do_set_current_image()

        self.job_no += 1
        self.sampling_step = 0
        self.current_image_sampling_step = 0

    def dict(self):
        obj = {
            "skipped": self.skipped,
            "interrupted": self.interrupted,
            "job": self.job,
            "job_count": self.job_count,
            "job_timestamp": self.job_timestamp,
            "job_no": self.job_no,
            "sampling_step": self.sampling_step,
            "sampling_steps": self.sampling_steps,
        }

        return obj

    def begin(self, job: str = "(unknown)"):
        self.sampling_step = 0
        self.job_count = -1
        self.processing_has_refined_job_count = False
        self.job_no = 0
        self.job_timestamp = datetime.datetime.now().strftime("%Y%m%d%H%M%S")
        self.current_latent = None
        self.current_image = None
        self.current_image_sampling_step = 0
        self.id_live_preview = 0
        self.skipped = False
        self.interrupted = False
        self.textinfo = None
        self.time_start = time.time()
        self.job = job
        devices.torch_gc()
        log.info("Starting job %s", job)

    def end(self):
        duration = time.time() - self.time_start
        log.info("Ending job %s (%.2f seconds)", self.job, duration)
        self.job = ""
        self.job_count = 0

        devices.torch_gc()

    def set_current_image(self):
        """if enough sampling steps have been made after the last call to this, sets self.current_image from self.current_latent, and modifies self.id_live_preview accordingly"""
        if not shared.parallel_processing_allowed:
            return

        if self.sampling_step - self.current_image_sampling_step >= shared.opts.show_progress_every_n_steps and shared.opts.live_previews_enable and shared.opts.show_progress_every_n_steps != -1:
            self.do_set_current_image()

    def do_set_current_image(self):
        if self.current_latent is None:
            return

        import modules.sd_samplers

        try:
            if shared.opts.show_progress_grid:
                self.assign_current_image(modules.sd_samplers.samples_to_image_grid(self.current_latent))
            else:
                self.assign_current_image(modules.sd_samplers.sample_to_image(self.current_latent))

            self.current_image_sampling_step = self.sampling_step

        except Exception:
            # when switching models during genration, VAE would be on CPU, so creating an image will fail.
            # we silently ignore this error
            errors.record_exception()

    def assign_current_image(self, image):
        self.current_image = image
        self.id_live_preview += 1
