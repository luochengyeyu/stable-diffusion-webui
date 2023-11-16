import base64
import io
import time

import gradio as gr
from pydantic import BaseModel, Field

from modules.shared import opts

import modules.shared as shared


current_task = None
pending_tasks = {}
finished_tasks = []
recorded_results = []
recorded_results_limit = 2


def start_task(id_task):
    """
    将id_task赋值给全局变量current_task，并从pending_tasks移除
    """
    global current_task

    current_task = id_task
    pending_tasks.pop(id_task, None)


def finish_task(id_task):
    global current_task

    if current_task == id_task:
        current_task = None

    finished_tasks.append(id_task)
    if len(finished_tasks) > 16:
        finished_tasks.pop(0)


def record_results(id_task, res):
    """
    记录（id_task,res)元组到 recorded_results 列表
    """
    recorded_results.append((id_task, res))
    # recorded_results列表长度大于2时移除列表 index=0 的元素
    if len(recorded_results) > recorded_results_limit:
        recorded_results.pop(0)


def add_task_to_queue(id_job):
    """
    添加任务到队列 pending_tasks，key位id_job，value为时间
    """
    pending_tasks[id_job] = time.time()


class ProgressRequest(BaseModel):
    # 要获取进度的任务的 ID
    id_task: str = Field(default=None, title="Task ID", description="id of the task to get progress for")
    # 上次接收到的最后一张预览图像的id
    id_live_preview: int = Field(default=-1, title="Live preview image ID", description="id of last received last preview image")
    # 是否包含实时预览图像
    live_preview: bool = Field(default=True, title="Include live preview", description="boolean flag indicating whether to include the live preview image")


class ProgressResponse(BaseModel):
    # 任务是否正在进行中
    active: bool = Field(title="Whether the task is being worked on right now")
    # 任务是否在队列中
    queued: bool = Field(title="Whether the task is in queue")
    # 任务是否已经完成
    completed: bool = Field(title="Whether the task has already finished")
    # 进度取值范围为0到1
    progress: float = Field(default=None, title="Progress", description="The progress with a range of 0 to 1")
    # 预计剩余时间（秒）
    eta: float = Field(default=None, title="ETA in secs")
    # 当前实时预览；一个数据：uri
    live_preview: str = Field(default=None, title="Live preview image", description="Current live preview; a data: uri")
    # 将此内容与下一个请求一起发送，以防止收到相同的图像
    id_live_preview: int = Field(default=None, title="Live preview image ID", description="Send this together with next request to prevent receiving same image")
    # WebUI 使用的信息文本
    textinfo: str = Field(default=None, title="Info text", description="Info text used by WebUI.")


def setup_progress_api(app):
    """
    为应用添加一个新的路由，用于接收并处理报告进度的POST请求，并返回一个ProgressResponse
    """
    return app.add_api_route("/internal/progress", progressapi, methods=["POST"], response_model=ProgressResponse)


def progressapi(req: ProgressRequest):
    active = req.id_task == current_task
    queued = req.id_task in pending_tasks
    completed = req.id_task in finished_tasks

    # 任务尚未开始
    if not active:
        textinfo = "Waiting..."
        # 任务在队列中
        if queued:
            # 对等待任务队列进行排序
            sorted_queued = sorted(pending_tasks.keys(), key=lambda x: pending_tasks[x])
            # 获取在队列中的下标
            queue_index = sorted_queued.index(req.id_task)
            # 组装当前任务在队列中位置的文本信息，eg：In queue： 2/5
            textinfo = "In queue: {}/{}".format(queue_index + 1, len(sorted_queued))
        return ProgressResponse(active=active, queued=queued, completed=completed, id_live_preview=-1, textinfo=textinfo)

    # 初始化进度为0
    progress = 0

    # 从shared.state获取当前的job数量以及job号
    job_count, job_no = shared.state.job_count, shared.state.job_no
    # 获取采样总步数，以及当前步数
    sampling_steps, sampling_step = shared.state.sampling_steps, shared.state.sampling_step

    if job_count > 0:
        progress += job_no / job_count
    if sampling_steps > 0 and job_count > 0:
        progress += 1 / job_count * sampling_step / sampling_steps

    progress = min(progress, 1)

    # 计算从开始到现在的时间差
    elapsed_since_start = time.time() - shared.state.time_start
    # 预估时间
    predicted_duration = elapsed_since_start / progress if progress > 0 else None
    # 计算剩余时间
    eta = predicted_duration - elapsed_since_start if predicted_duration is not None else None

    live_preview = None
    id_live_preview = req.id_live_preview

    # 设置中 显示所创建图像的实时预览 选项是开启状态，并且请求参数live_preview 为True
    if opts.live_previews_enable and req.live_preview:
        shared.state.set_current_image()
        if shared.state.id_live_preview != req.id_live_preview:
            image = shared.state.current_image
            if image is not None:
                buffered = io.BytesIO()

                if opts.live_previews_image_format == "png":
                    # using optimize for large images takes an enormous amount of time
                    if max(*image.size) <= 256:
                        save_kwargs = {"optimize": True}
                    else:
                        save_kwargs = {"optimize": False, "compress_level": 1}

                else:
                    save_kwargs = {}

                image.save(buffered, format=opts.live_previews_image_format, **save_kwargs)
                base64_image = base64.b64encode(buffered.getvalue()).decode('ascii')
                live_preview = f"data:image/{opts.live_previews_image_format};base64,{base64_image}"
                id_live_preview = shared.state.id_live_preview

    return ProgressResponse(active=active, queued=queued, completed=completed, progress=progress, eta=eta, live_preview=live_preview, id_live_preview=id_live_preview, textinfo=shared.state.textinfo)


def restore_progress(id_task):
    # 任务正在执行中或者任务在等待队列中，循环等待
    while id_task == current_task or id_task in pending_tasks:
        time.sleep(0.1)

    # 任务执行完毕获取结果
    res = next(iter([x[1] for x in recorded_results if id_task == x[0]]), None)
    if res is not None:
        return res

    return gr.update(), gr.update(), gr.update(), f"Couldn't restore progress for {id_task}: results either have been discarded or never were obtained"
