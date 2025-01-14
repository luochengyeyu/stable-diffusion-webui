import sys  # 系统相关的形参和函数
import contextlib  # 此模块为涉及 with 语句的常见任务提供了实用的工具。
from functools import lru_cache  # LRU缓存

import torch
from modules import errors, shared

if sys.platform == "darwin":
    from modules import mac_specific  # macos特殊处理


def has_mps() -> bool:
    if sys.platform != "darwin":
        return False
    else:
        return mac_specific.has_mps


# 若config.json配置了--device-id就取配置文件里的，否则直接返回默认的"cuda"
def get_cuda_device_string():
    if shared.cmd_opts.device_id is not None:
        return f"cuda:{shared.cmd_opts.device_id}"

    return "cuda"


def get_optimal_device_name():
    if torch.cuda.is_available():  # gpu可用
        return get_cuda_device_string()

    if has_mps():  # macos
        return "mps"

    return "cpu"


# 获得最佳设备
def get_optimal_device():
    return torch.device(get_optimal_device_name())


# --use-cpu   {all, sd, interrogate, gfpgan, bsrgan, esrgan, scunet, codeformer}
# 若task术语use-cpu参数中的一个，返回cpu设备，否则调用get_optimal_device()获取设备
def get_device_for(task):
    if task in shared.cmd_opts.use_cpu:
        return cpu

    return get_optimal_device()


#  torch垃圾回收
def torch_gc():
    if torch.cuda.is_available():
        with torch.cuda.device(get_cuda_device_string()):
            torch.cuda.empty_cache()
            torch.cuda.ipc_collect()

    if has_mps():
        mac_specific.torch_mps_gc()


def enable_tf32():
    if torch.cuda.is_available():

        # enabling benchmark option seems to enable a range of cards to do fp16 when they otherwise can't
        # see https://github.com/AUTOMATIC1111/stable-diffusion-webui/pull/4407
        if any(torch.cuda.get_device_capability(devid) == (7, 5) for devid in range(0, torch.cuda.device_count())):
            torch.backends.cudnn.benchmark = True

        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True


errors.run(enable_tf32, "Enabling TF32")

cpu: torch.device = torch.device("cpu")
device: torch.device = None
device_interrogate: torch.device = None
device_gfpgan: torch.device = None
device_esrgan: torch.device = None
device_codeformer: torch.device = None
dtype: torch.dtype = torch.float16
dtype_vae: torch.dtype = torch.float16
dtype_unet: torch.dtype = torch.float16
unet_needs_upcast = False


def cond_cast_unet(input):
    return input.to(dtype_unet) if unet_needs_upcast else input


def cond_cast_float(input):
    return input.float() if unet_needs_upcast else input


nv_rng = None


def autocast(disable=False):
    if disable:
        # nullcontext()是一个特殊的上下文管理器，它不做任何事情。它通常用于替换预期的上下文管理器，当不需要执行任何操作时。
        return contextlib.nullcontext()

    if dtype == torch.float32 or shared.cmd_opts.precision == "full":
        # dtype是PyTorch的float32类型 或者 命令行参数--precision为“full" 也返回nullcontext
        return contextlib.nullcontext()
    # 返回一个上下文管理器，该管理器将自动将数据类型转换为"cuda"，这是GPU上支持的默认数据类型。
    return torch.autocast("cuda")


def without_autocast(disable=False):
    """
    在不需要自动类型转换的情况下，返回一个不执行任何操作的上下文管理器 contextlib.nullcontext()。\n
    如果需要执行自动类型转换，并且没有禁用它，那么就返回一个上下文管理器，该管理器将数据类型转换为 "cuda"。
    """
    return torch.autocast("cuda", enabled=False) if torch.is_autocast_enabled() and not disable else contextlib.nullcontext()


class NansException(Exception):
    pass


def test_for_nans(x, where):
    if shared.cmd_opts.disable_nan_check:
        return

    if not torch.all(torch.isnan(x)).item():
        return

    if where == "unet":
        message = "A tensor with all NaNs was produced in Unet."

        if not shared.cmd_opts.no_half:
            message += " This could be either because there's not enough precision to represent the picture, or because your video card does not support half type. Try setting the \"Upcast cross attention layer to float32\" option in Settings > Stable Diffusion or using the --no-half commandline argument to fix this."

    elif where == "vae":
        message = "A tensor with all NaNs was produced in VAE."

        if not shared.cmd_opts.no_half and not shared.cmd_opts.no_half_vae:
            message += " This could be because there's not enough precision to represent the picture. Try adding --no-half-vae commandline argument to fix this."
    else:
        message = "A tensor with all NaNs was produced."

    message += " Use --disable-nan-check commandline argument to disable this check."

    raise NansException(message)


@lru_cache
def first_time_calculation():
    """
    just do any calculation with pytorch layers - the first time this is done it allocaltes about 700MB of memory and
    spends about 2.7 seconds doing that, at least wih NVidia.
    使用PyTorch层进行任何计算 - 第一次这样做时，它会分配大约700MB的内存，
    至少在NVidia上，完成该操作需要2.7秒。
    """

    x = torch.zeros((1, 1)).to(device, dtype)
    linear = torch.nn.Linear(1, 1).to(device, dtype)
    linear(x)

    x = torch.zeros((1, 1, 3, 3)).to(device, dtype)
    conv2d = torch.nn.Conv2d(1, 1, (3, 3)).to(device, dtype)
    conv2d(x)
