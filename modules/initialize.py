import importlib
import logging
import sys
import warnings
from threading import Thread

from modules.timer import startup_timer


def imports():
    logging.getLogger("torch.distributed.nn").setLevel(logging.ERROR)  # sshh...
    logging.getLogger("xformers").addFilter(
        lambda record: 'A matching Triton is not available' not in record.getMessage())

    import torch  # noqa: F401
    startup_timer.record("import torch")
    import pytorch_lightning  # noqa: F401
    startup_timer.record("import torch")
    warnings.filterwarnings(action="ignore", category=DeprecationWarning, module="pytorch_lightning")
    warnings.filterwarnings(action="ignore", category=UserWarning, module="torchvision")

    import gradio  # noqa: F401
    startup_timer.record("import gradio")

    from modules import paths, timer, import_hook, errors  # noqa: F401
    startup_timer.record("setup paths")

    import ldm.modules.encoders.modules  # noqa: F401
    startup_timer.record("import ldm")

    import sgm.modules.encoders.modules  # noqa: F401
    startup_timer.record("import sgm")

    from modules import shared_init
    shared_init.initialize()
    startup_timer.record("initialize shared")

    from modules import processing, gradio_extensons, ui  # noqa: F401
    startup_timer.record("other imports")


# 检查依赖库torch gradio xformers 版本
def check_versions():
    from modules.shared_cmd_options import cmd_opts

    if not cmd_opts.skip_version_check:
        from modules import errors
        errors.check_versions()


def initialize():
    # 导入初始化工具类模块
    from modules import initialize_util
    # 截断 PyTorch 的 nightlylocal build 的版本号，以免导致 CodeFormer 或 Safetensors 出现异常
    initialize_util.fix_torch_version()
    # 安装此策略允许事件 循环将在任何线程上自动创建
    initialize_util.fix_asyncio_event_loop_policy()
    # 校验TLS选项
    initialize_util.validate_tls_options()
    # 配置信号处理程序：使程序在 Ctrl+C 处退出，无需等待任何操作
    initialize_util.configure_sigint_handler()
    # 设置配置项发生变化回调,例如切换大模型
    initialize_util.configure_opts_onchange()
    # 将模型移动到正确的位置
    from modules import modelloader
    modelloader.cleanup_models()

    from modules import sd_models
    # 创建模型目录，开启midas模型自动下载
    sd_models.setup_model()
    startup_timer.record("setup SD model")

    from modules.shared_cmd_options import cmd_opts

    # 导入并设置Codeformer
    from modules import codeformer_model
    # 过滤警告信息
    warnings.filterwarnings(action="ignore", category=UserWarning, module="torchvision.transforms.functional_tensor")
    # codeformer_models_path:  models/Codeformer
    codeformer_model.setup_model(cmd_opts.codeformer_models_path)
    startup_timer.record("setup codeformer")

    # 导入并设置gfpgan
    from modules import gfpgan_model
    gfpgan_model.setup_model(cmd_opts.gfpgan_models_path)
    startup_timer.record("setup gfpgan")

    initialize_rest(reload_script_modules=False)


# 初始化和调用reload时使用次函数
def initialize_rest(*, reload_script_modules=False):
    """
    Called both from initialize() and when reloading the webui.
    """
    from modules.shared_cmd_options import cmd_opts

    from modules import sd_samplers
    sd_samplers.set_samplers()
    startup_timer.record("set samplers")

    from modules import extensions
    extensions.list_extensions()
    startup_timer.record("list extensions")

    from modules import initialize_util
    initialize_util.restore_config_state_file()
    startup_timer.record("restore config state file")

    from modules import shared, upscaler, scripts
    if cmd_opts.ui_debug_mode:
        shared.sd_upscalers = upscaler.UpscalerLanczos().scalers
        scripts.load_scripts()
        return

    # 读取 models/Stable-diffusion的大模型列表
    from modules import sd_models
    sd_models.list_models()
    startup_timer.record("list SD models")

    from modules import localization
    # cmd_opts.localizations_dir = localizations/
    # 读取本地化配置文件
    localization.list_localizations(cmd_opts.localizations_dir)
    startup_timer.record("list localizations")

    with startup_timer.subcategory("load scripts"):
        scripts.load_scripts()

    if reload_script_modules:
        for module in [module for name, module in sys.modules.items() if name.startswith("modules.ui")]:
            importlib.reload(module)
        startup_timer.record("reload script modules")

    # 动态加载 upscalers
    # Upscalers 是指一种用于放大图像的模型或算法。
    from modules import modelloader
    modelloader.load_upscalers()
    startup_timer.record("load upscalers")

    # 刷新vae列表
    from modules import sd_vae
    sd_vae.refresh_vae_list()
    startup_timer.record("refresh VAE")

    # 加载 textual_inversion 模板列表
    from modules import textual_inversion
    textual_inversion.textual_inversion.list_textual_inversion_templates()
    startup_timer.record("refresh textual inversion templates")

    from modules import script_callbacks, sd_hijack_optimizations, sd_hijack
    script_callbacks.on_list_optimizers(sd_hijack_optimizations.list_optimizers)
    sd_hijack.list_optimizers()
    startup_timer.record("scripts list_optimizers")

    from modules import sd_unet
    sd_unet.list_unets()
    startup_timer.record("scripts list_unets")

    def load_model():
        """
        Accesses shared.sd_model property to load model.
        After it's available, if it has been loaded before this access by some extension,
        its optimization may be None because the list of optimizaers has neet been filled
        by that time, so we apply optimization again.
        """

        shared.sd_model  # noqa: B018

        if sd_hijack.current_optimizer is None:
            sd_hijack.apply_optimizations()

        from modules import devices
        devices.first_time_calculation()

    Thread(target=load_model).start()

    from modules import shared_items
    shared_items.reload_hypernetworks()
    startup_timer.record("reload hypernetworks")

    from modules import ui_extra_networks
    ui_extra_networks.initialize()
    ui_extra_networks.register_default_pages()

    from modules import extra_networks
    extra_networks.initialize()
    extra_networks.register_default_extra_networks()
    startup_timer.record("initialize extra networks")
