import threading
import time
from collections import defaultdict

import torch


# 内存使用监视器
class MemUsageMonitor(threading.Thread):
    run_flag = None
    device = None
    disabled = False
    opts = None
    data = None

    def __init__(self, name, device, opts):
        threading.Thread.__init__(self)
        self.name = name
        self.device = device
        self.opts = opts

        self.daemon = True
        self.run_flag = threading.Event()
        self.data = defaultdict(int)

        try:
            self.cuda_mem_get_info()
            # 返回给定设备的CUDA内存分配器统计信息的字典。
            torch.cuda.memory_stats(self.device)
        except Exception as e:  # AMD or whatever
            print(f"Warning: caught exception '{e}', memory monitor disabled")
            self.disabled = True

    def cuda_mem_get_info(self):
        # torch.cuda.current_device() 返回当前所选设备的索引
        index = self.device.index if self.device.index is not None else torch.cuda.current_device()
        # 使用cudaMemGetInfo返回给定设备的全局空闲和总GPU内存。
        return torch.cuda.mem_get_info(index)

    def run(self):
        if self.disabled:
            return

        while True:
            self.run_flag.wait()
            # 重置CUDA内存分配器所追踪的“峰值”统计信息
            torch.cuda.reset_peak_memory_stats()
            self.data.clear()

            if self.opts.memmon_poll_rate <= 0:
                self.run_flag.clear()
                continue

            self.data["min_free"] = self.cuda_mem_get_info()[0]

            # 每八分之一秒获取一下内存信息并记录
            while self.run_flag.is_set():
                free, total = self.cuda_mem_get_info()
                self.data["min_free"] = min(self.data["min_free"], free)
                # 在生成过程中每秒进行一次VRAM使用情况的调查
                # config.json 配置memmon_poll_rate = 8
                time.sleep(1 / self.opts.memmon_poll_rate)

    def dump_debug(self):
        print(self, 'recorded data:')
        for k, v in self.read().items():
            print(k, -(v // -(1024 ** 2)))

        print(self, 'raw torch memory stats:')
        tm = torch.cuda.memory_stats(self.device)
        for k, v in tm.items():
            if 'bytes' not in k:
                continue
            print('\t' if 'peak' in k else '', k, -(v // -(1024 ** 2)))

        print(torch.cuda.memory_summary())

    def monitor(self):
        self.run_flag.set()

    def read(self):
        if not self.disabled:
            free, total = self.cuda_mem_get_info()
            self.data["free"] = free
            self.data["total"] = total

            torch_stats = torch.cuda.memory_stats(self.device)
            self.data["active"] = torch_stats["active.all.current"]
            self.data["active_peak"] = torch_stats["active_bytes.all.peak"]
            self.data["reserved"] = torch_stats["reserved_bytes.all.current"]
            self.data["reserved_peak"] = torch_stats["reserved_bytes.all.peak"]
            self.data["system_peak"] = total - self.data["min_free"]

        return self.data

    def stop(self):
        self.run_flag.clear()
        return self.read()
