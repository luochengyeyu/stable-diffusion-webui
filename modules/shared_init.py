import os

import torch

from modules import shared
from modules.shared import cmd_opts


def initialize():
    """Initializes fields inside the shared module in a controlled manner.

    Should be called early because some other modules you can import mingt need these fields to be already set.
    """
    # 创建hypernetwork目录：models/hypernetworks/
    os.makedirs(cmd_opts.hypernetwork_dir, exist_ok=True)

    from modules import options, shared_options
    # shared选项模板赋值，类型为字典类型
    shared.options_templates = shared_options.options_templates
    #
    shared.opts = options.Options(shared_options.options_templates, shared_options.restricted_opts)
    shared.restricted_opts = shared_options.restricted_opts

    if os.path.exists(shared.config_filename):
        # 若config.json文件存在，调用load方法将配置文件的参数加载到内存中
        shared.opts.load(shared.config_filename)

    # 导入设备模块
    from modules import devices
    devices.device, devices.device_interrogate, devices.device_gfpgan, devices.device_esrgan, devices.device_codeformer = \
        (devices.cpu if any(y in cmd_opts.use_cpu for y in [x, 'all']) else devices.get_optimal_device() for x in
         ['sd', 'interrogate', 'gfpgan', 'esrgan', 'codeformer'])
    # 启动项配置了 no_half 返回 float32 否则返回 float16
    devices.dtype = torch.float32 if cmd_opts.no_half else torch.float16
    devices.dtype_vae = torch.float32 if cmd_opts.no_half or cmd_opts.no_half_vae else torch.float16

    shared.device = devices.device
    # 将 Stable Diffusion checkpoint 权重加载到 VRAM 而不是RAM。
    # VRAM GPU的内存
    # RAM CPU的内存
    shared.weight_load_location = None if cmd_opts.lowram else "cpu"

    # 共享状态
    from modules import shared_state
    shared.state = shared_state.State()

    # 导入样式模块
    from modules import styles
    # 创建样式数据库
    # shared.styles_filename = styles.csv
    shared.prompt_styles = styles.StyleDatabase(shared.styles_filename)

    from modules import interrogate
    shared.interrogator = interrogate.InterrogateModels("interrogate")

    #  进度条模块
    from modules import shared_total_tqdm
    shared.total_tqdm = shared_total_tqdm.TotalTQDM()

    # 启动内存监视器
    from modules import memmon, devices
    shared.mem_mon = memmon.MemUsageMonitor("MemMon", devices.device, shared.opts)
    shared.mem_mon.start()
