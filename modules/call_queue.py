from functools import wraps
import html
import time

from modules import shared, progress, errors, devices, fifo_lock

queue_lock = fifo_lock.FIFOLock()


def wrap_queued_call(func):
    """ 排队调用装饰器函数"""
    # 定义一个新的函数，这是装饰器返回的函数
    def f(*args, **kwargs):
        # 使用队列锁保护下面的操作
        with queue_lock:
            # 调用原始函数并获取结果
            res = func(*args, **kwargs)

        return res

    return f


def wrap_gradio_gpu_call(func, extra_outputs=None):
    @wraps(func)
    def f(*args, **kwargs):
        # 受任意数量的位置参数（*args）和关键字参数（**kwargs）
        # if the first argument is a string that says "task(...)", it is treated as a job id
        # 检查函数的第一个参数是否是一个以"task("开头，以")"结尾的字符串。如果是，它将此参数视为一个任务ID，并将其添加到一个任务队列中。
        if args and type(args[0]) == str and args[0].startswith("task(") and args[0].endswith(")"):
            id_task = args[0]
            progress.add_task_to_queue(id_task)
        else:
            id_task = None

        # 使用queue_lock上下文管理器来确保线程安全
        with queue_lock:
            # 将job记录共享状态上
            shared.state.begin(job=id_task)
            # 开始任务，将此 task 设置为 current_task
            progress.start_task(id_task)

            try:
                # 执行实际的函数并传入参数
                res = func(*args, **kwargs)
                # 记录执行结果
                progress.record_results(id_task, res)
            finally:
                # 无论是否发生异常都要将任务添加到 已完成列表
                progress.finish_task(id_task)
            # 更改任务状态为结束
            shared.state.end()

        return res

    return wrap_gradio_call(f, extra_outputs=extra_outputs, add_stats=True)


def wrap_gradio_call(func, extra_outputs=None, add_stats=False):
    @wraps(func)
    def f(*args, extra_outputs_array=extra_outputs, **kwargs):
        run_memmon = shared.opts.memmon_poll_rate > 0 and not shared.mem_mon.disabled and add_stats
        if run_memmon:
            shared.mem_mon.monitor()
        t = time.perf_counter()

        try:
            res = list(func(*args, **kwargs))
        except Exception as e:
            # When printing out our debug argument list,
            # do not print out more than a 100 KB of text
            max_debug_str_len = 131072
            message = "Error completing request"
            arg_str = f"Arguments: {args} {kwargs}"[:max_debug_str_len]
            if len(arg_str) > max_debug_str_len:
                arg_str += f" (Argument list truncated at {max_debug_str_len}/{len(arg_str)} characters)"
            errors.report(f"{message}\n{arg_str}", exc_info=True)

            shared.state.job = ""
            shared.state.job_count = 0

            if extra_outputs_array is None:
                extra_outputs_array = [None, '']

            error_message = f'{type(e).__name__}: {e}'
            res = extra_outputs_array + [f"<div class='error'>{html.escape(error_message)}</div>"]

        devices.torch_gc()

        shared.state.skipped = False
        shared.state.interrupted = False
        shared.state.job_count = 0

        if not add_stats:
            return tuple(res)

        elapsed = time.perf_counter() - t
        elapsed_m = int(elapsed // 60)
        elapsed_s = elapsed % 60
        elapsed_text = f"{elapsed_s:.1f} sec."
        if elapsed_m > 0:
            elapsed_text = f"{elapsed_m} min. "+elapsed_text

        if run_memmon:
            mem_stats = {k: -(v//-(1024*1024)) for k, v in shared.mem_mon.stop().items()}
            active_peak = mem_stats['active_peak']
            reserved_peak = mem_stats['reserved_peak']
            sys_peak = mem_stats['system_peak']
            sys_total = mem_stats['total']
            sys_pct = sys_peak/max(sys_total, 1) * 100

            toltip_a = "Active: peak amount of video memory used during generation (excluding cached data)"
            toltip_r = "Reserved: total amout of video memory allocated by the Torch library "
            toltip_sys = "System: peak amout of video memory allocated by all running programs, out of total capacity"

            text_a = f"<abbr title='{toltip_a}'>A</abbr>: <span class='measurement'>{active_peak/1024:.2f} GB</span>"
            text_r = f"<abbr title='{toltip_r}'>R</abbr>: <span class='measurement'>{reserved_peak/1024:.2f} GB</span>"
            text_sys = f"<abbr title='{toltip_sys}'>Sys</abbr>: <span class='measurement'>{sys_peak/1024:.1f}/{sys_total/1024:g} GB</span> ({sys_pct:.1f}%)"

            vram_html = f"<p class='vram'>{text_a}, <wbr>{text_r}, <wbr>{text_sys}</p>"
        else:
            vram_html = ''

        # last item is always HTML
        res[-1] += f"<div class='performance'><p class='time'>Time taken: <wbr><span class='measurement'>{elapsed_text}</span></p>{vram_html}</div>"

        return tuple(res)

    return f
