# 容器数据类型 https://docs.python.org/zh-cn/3/library/collections.html
import collections
# 常用路径操作 https://docs.python.org/zh-cn/3/library/os.path.html
import os.path
import sys
# 垃圾回收器 https://docs.python.org/zh-cn/3/library/gc.html
import gc
# 基于线程的并行 https://docs.python.org/zh-cn/3/library/threading.html
import threading
# Facebook的深度学习框架
import torch
import re
# https://github.com/huggingface/safetensors
import safetensors.torch
# 管理和组织配置信息 https://github.com/omry/omegaconf
from omegaconf import OmegaConf
from os import mkdir
# urllib.request 模块定义了适用于在各种复杂情况下打开 URL（主要为 HTTP）的函数和类
from urllib import request
# Midas是一种机器学习模型，可根据任意输入图像估计深度。
import ldm.modules.midas as midas

from ldm.util import instantiate_from_config

from modules import paths, shared, modelloader, devices, script_callbacks, sd_vae, sd_disable_initialization, errors, \
    hashes, sd_models_config, sd_unet, sd_models_xl, cache, extra_networks, processing, lowvram, sd_hijack
from modules.timer import Timer
# tomesd是一种用于加速深度学习模型训练的技术，它可以在保证图片生成质量的基础上，大幅提升stable diffusion生成图片的速度。它被用于大规模生成图片，节省GPU资源和内存资源。
import tomesd

model_dir = "Stable-diffusion"
model_path = os.path.abspath(os.path.join(paths.models_path, model_dir))

checkpoints_list = {}
checkpoint_aliases = {}
checkpoint_alisases = checkpoint_aliases  # for compatibility with old name
# OrderedDict()返回一个 dict 子类的实例，它具有专门用于重新排列字典顺序的方法。
checkpoints_loaded = collections.OrderedDict()


def replace_key(d, key, new_key, value):
    keys = list(d.keys())

    d[new_key] = value

    if key not in keys:
        return d

    index = keys.index(key)
    keys[index] = new_key

    new_d = {k: d[k] for k in keys}

    d.clear()
    d.update(new_d)
    return d


class CheckpointInfo:
    def __init__(self, filename):
        self.filename = filename
        abspath = os.path.abspath(filename)

        self.is_safetensors = os.path.splitext(filename)[1].lower() == ".safetensors"

        if shared.cmd_opts.ckpt_dir is not None and abspath.startswith(shared.cmd_opts.ckpt_dir):
            name = abspath.replace(shared.cmd_opts.ckpt_dir, '')
        elif abspath.startswith(model_path):
            name = abspath.replace(model_path, '')
        else:
            name = os.path.basename(filename)

        if name.startswith("\\") or name.startswith("/"):
            name = name[1:]

        def read_metadata():
            metadata = read_metadata_from_safetensors(filename)
            self.modelspec_thumbnail = metadata.pop('modelspec.thumbnail', None)

            return metadata

        self.metadata = {}
        if self.is_safetensors:
            try:
                self.metadata = cache.cached_data_for_file('safetensors-metadata', "checkpoint/" + name, filename,
                                                           read_metadata)
            except Exception as e:
                errors.display(e, f"reading metadata for {filename}")

        self.name = name
        self.name_for_extra = os.path.splitext(os.path.basename(filename))[0]
        self.model_name = os.path.splitext(name.replace("/", "_").replace("\\", "_"))[0]
        self.hash = model_hash(filename)

        self.sha256 = hashes.sha256_from_cache(self.filename, f"checkpoint/{name}")
        self.shorthash = self.sha256[0:10] if self.sha256 else None

        self.title = name if self.shorthash is None else f'{name} [{self.shorthash}]'
        self.short_title = self.name_for_extra if self.shorthash is None else f'{self.name_for_extra} [{self.shorthash}]'

        self.ids = [self.hash, self.model_name, self.title, name, self.name_for_extra, f'{name} [{self.hash}]']
        if self.shorthash:
            self.ids += [self.shorthash, self.sha256, f'{self.name} [{self.shorthash}]',
                         f'{self.name_for_extra} [{self.shorthash}]']

    def register(self):
        # title格式：revAnimated_v122.safetensors [4199bcdd14]
        # {title:CheckpointInfo}
        checkpoints_list[self.title] = self
        # ids的数据格式如下：
        # [
        # '893e49b9', 'anything-v5-PrtRE', 'anything-v5-PrtRE.safetensors [7f96a1a9ca]',
        # 'anything-v5-PrtRE.safetensors', 'anything-v5-PrtRE', 'anything-v5-PrtRE.safetensors [893e49b9]',
        # '7f96a1a9ca', '7f96a1a9ca9b3a3242a9ae95d19284f0d2da8d5282b42d2d974398bf7663a252',
        # 'anything-v5-PrtRE.safetensors [7f96a1a9ca]',
        # 'anything-v5-PrtRE [7f96a1a9ca]'
        # ]
        # 遍历ids，将每一项作为key，当前的 CheckpointInfo 作为value存到checkpoint_aliases字典中
        # 目的是：不管用ids里的哪一种形式都可以取到对应的 CheckpointInfo
        for id in self.ids:
            checkpoint_aliases[id] = self

    def calculate_shorthash(self):
        self.sha256 = hashes.sha256(self.filename, f"checkpoint/{self.name}")
        if self.sha256 is None:
            return

        shorthash = self.sha256[0:10]
        if self.shorthash == self.sha256[0:10]:
            return self.shorthash

        self.shorthash = shorthash

        if self.shorthash not in self.ids:
            self.ids += [self.shorthash, self.sha256, f'{self.name} [{self.shorthash}]',
                         f'{self.name_for_extra} [{self.shorthash}]']

        old_title = self.title
        self.title = f'{self.name} [{self.shorthash}]'
        self.short_title = f'{self.name_for_extra} [{self.shorthash}]'

        replace_key(checkpoints_list, old_title, self.title, self)
        self.register()

        return self.shorthash


try:
    # this silences the annoying "Some weights of the model checkpoint were not used when initializing..." message at start.
    from transformers import logging, CLIPModel  # noqa: F401

    logging.set_verbosity_error()
except Exception:
    pass


def setup_model():
    os.makedirs(model_path, exist_ok=True)
    # 开启自动下载dpt的midas库
    enable_midas_autodownload()


def checkpoint_tiles(use_short=False):
    return [x.short_title if use_short else x.title for x in checkpoints_list.values()]


def list_models():
    checkpoints_list.clear()
    checkpoint_aliases.clear()

    cmd_ckpt = shared.cmd_opts.ckpt
    if shared.cmd_opts.no_download_sd_model or cmd_ckpt != shared.sd_model_file or os.path.exists(cmd_ckpt):
        model_url = None
    else:
        model_url = "https://huggingface.co/runwayml/stable-diffusion-v1-5/resolve/main/v1-5-pruned-emaonly.safetensors"
    # 获取\models\Stable-diffusion文件夹下.ckpt和.safetensors模型的path列表
    model_list = modelloader.load_models(model_path=model_path, model_url=model_url,
                                         command_path=shared.cmd_opts.ckpt_dir, ext_filter=[".ckpt", ".safetensors"],
                                         download_name="v1-5-pruned-emaonly.safetensors",
                                         ext_blacklist=[".vae.ckpt", ".vae.safetensors"])
    if os.path.exists(cmd_ckpt):
        checkpoint_info = CheckpointInfo(cmd_ckpt)
        checkpoint_info.register()

        shared.opts.data['sd_model_checkpoint'] = checkpoint_info.title
    elif cmd_ckpt is not None and cmd_ckpt != shared.default_sd_model_file:
        print(f"Checkpoint in --ckpt argument not found (Possible it was moved to {model_path}: {cmd_ckpt}",
              file=sys.stderr)

    # 根据 model_list 生成Checkpoint信息类 并注册
    for filename in model_list:
        checkpoint_info = CheckpointInfo(filename)
        checkpoint_info.register()


re_strip_checksum = re.compile(r"\s*\[[^]]+]\s*$")


def get_closet_checkpoint_match(search_string):
    if not search_string:
        return None

    checkpoint_info = checkpoint_aliases.get(search_string, None)
    if checkpoint_info is not None:
        return checkpoint_info

    found = sorted([info for info in checkpoints_list.values() if search_string in info.title],
                   key=lambda x: len(x.title))
    if found:
        return found[0]

    search_string_without_checksum = re.sub(re_strip_checksum, '', search_string)
    found = sorted([info for info in checkpoints_list.values() if search_string_without_checksum in info.title],
                   key=lambda x: len(x.title))
    if found:
        return found[0]

    return None


def model_hash(filename):
    """old hash that only looks at a small part of the file and is prone to collisions"""

    try:
        with open(filename, "rb") as file:
            import hashlib
            m = hashlib.sha256()

            file.seek(0x100000)
            m.update(file.read(0x10000))
            return m.hexdigest()[0:8]
    except FileNotFoundError:
        return 'NOFILE'


def select_checkpoint():
    """Raises `FileNotFoundError` if no checkpoints are found."""
    model_checkpoint = shared.opts.sd_model_checkpoint

    checkpoint_info = checkpoint_aliases.get(model_checkpoint, None)
    if checkpoint_info is not None:
        return checkpoint_info

    if len(checkpoints_list) == 0:
        error_message = "No checkpoints found. When searching for checkpoints, looked at:"
        if shared.cmd_opts.ckpt is not None:
            error_message += f"\n - file {os.path.abspath(shared.cmd_opts.ckpt)}"
        error_message += f"\n - directory {model_path}"
        if shared.cmd_opts.ckpt_dir is not None:
            error_message += f"\n - directory {os.path.abspath(shared.cmd_opts.ckpt_dir)}"
        error_message += "Can't run without a checkpoint. Find and place a .ckpt or .safetensors file into any of those locations."
        raise FileNotFoundError(error_message)

    checkpoint_info = next(iter(checkpoints_list.values()))
    if model_checkpoint is not None:
        print(f"Checkpoint {model_checkpoint} not found; loading fallback {checkpoint_info.title}", file=sys.stderr)

    return checkpoint_info


checkpoint_dict_replacements = {
    'cond_stage_model.transformer.embeddings.': 'cond_stage_model.transformer.text_model.embeddings.',
    'cond_stage_model.transformer.encoder.': 'cond_stage_model.transformer.text_model.encoder.',
    'cond_stage_model.transformer.final_layer_norm.': 'cond_stage_model.transformer.text_model.final_layer_norm.',
}


def transform_checkpoint_dict_key(k):
    for text, replacement in checkpoint_dict_replacements.items():
        if k.startswith(text):
            k = replacement + k[len(text):]

    return k


# 主要功能是从加载的状态字典中获取模型的状态字典。
def get_state_dict_from_checkpoint(pl_sd):
    # 这行代码从 pl_sd 中移除 "state_dict" 键，并获取它的值。
    # 如果 "state_dict" 键不存在，就返回 pl_sd 本身。这可能是在处理嵌套的状态字典
    pl_sd = pl_sd.pop("state_dict", pl_sd)
    # 这行代码再次尝试从 pl_sd 中移除 "state_dict" 键。
    # 这可能是为了确保 "state_dict" 键被完全移除。
    pl_sd.pop("state_dict", None)

    sd = {}
    # 移除后的字典循环加载到新的数组中
    for k, v in pl_sd.items():
        new_key = transform_checkpoint_dict_key(k)

        if new_key is not None:
            sd[new_key] = v

    pl_sd.clear()
    pl_sd.update(sd)
    # 返回新的字典
    return pl_sd


def read_metadata_from_safetensors(filename):
    import json

    with open(filename, mode="rb") as file:
        metadata_len = file.read(8)
        metadata_len = int.from_bytes(metadata_len, "little")
        json_start = file.read(2)

        assert metadata_len > 2 and json_start in (b'{"', b"{'"), f"{filename} is not a safetensors file"
        json_data = json_start + file.read(metadata_len - 2)
        json_obj = json.loads(json_data)

        res = {}
        for k, v in json_obj.get("__metadata__", {}).items():
            res[k] = v
            if isinstance(v, str) and v[0:1] == '{':
                try:
                    res[k] = json.loads(v)
                except Exception:
                    pass

        return res


# 从指定的文件中加载模型的状态字典
# checkpoint_file 传进来的是模型的路径
def read_state_dict(checkpoint_file, print_global_state=False, map_location=None):
    # 获取 checkpoint_file 的文件扩展名。
    _, extension = os.path.splitext(checkpoint_file)
    # 如果是safetensors结尾 使用 safetensors.torch.load_file 
    # 或 safetensors.torch.load 从文件中加载状态字典，并可能将其移动到指定的设备。 我们可以指定为GPU
    if extension.lower() == ".safetensors":
        device = map_location or shared.weight_load_location or devices.get_optimal_device_name()

        if not shared.opts.disable_mmap_load_safetensors:
            pl_sd = safetensors.torch.load_file(checkpoint_file, device=device)
        else:
            pl_sd = safetensors.torch.load(open(checkpoint_file, 'rb').read())
            pl_sd = {k: v.to(device) for k, v in pl_sd.items()}
    else:
        # 使用torch 加载模型
        pl_sd = torch.load(checkpoint_file, map_location=map_location or shared.weight_load_location)

    if print_global_state and "global_step" in pl_sd:
        print(f"Global Step: {pl_sd['global_step']}")
    # 从加载的状态字典中获取模型的状态字典。
    sd = get_state_dict_from_checkpoint(pl_sd)
    return sd


# 从磁盘或缓存中加载模型的向量数据
def get_checkpoint_state_dict(checkpoint_info: CheckpointInfo, timer):
    # 计算并获取模型的短哈希值。
    sd_model_hash = checkpoint_info.calculate_shorthash()
    timer.record("calculate hash")
    # 判断这个模型师傅已经加载了 如果加载了 直接在缓存中进行加载
    if checkpoint_info in checkpoints_loaded:
        # use checkpoint cache
        print(f"Loading weights [{sd_model_hash}] from cache")
        return checkpoints_loaded[checkpoint_info]

    print(f"Loading weights [{sd_model_hash}] from {checkpoint_info.filename}")
    # 调用从磁盘加载模型向量数据
    res = read_state_dict(checkpoint_info.filename)
    timer.record("load weights from disk")

    return res


class SkipWritingToConfig:
    """This context manager prevents load_model_weights from writing checkpoint name to the config when it loads weight."""

    skip = False
    previous = None

    def __enter__(self):
        self.previous = SkipWritingToConfig.skip
        SkipWritingToConfig.skip = True
        return self

    def __exit__(self, exc_type, exc_value, exc_traceback):
        SkipWritingToConfig.skip = self.previous


# 加载模型想了
def load_model_weights(model, checkpoint_info: CheckpointInfo, state_dict, timer):
    # 先计算info的模型端hash
    sd_model_hash = checkpoint_info.calculate_shorthash()
    timer.record("calculate hash")
    # 把info的title设置到json里
    if not SkipWritingToConfig.skip:
        shared.opts.data["sd_model_checkpoint"] = checkpoint_info.title
    # 如果状态字典是空的 在从缓存或者磁盘中获取一遍  感觉代码冗余 可能是其他场景吧
    if state_dict is None:
        state_dict = get_checkpoint_state_dict(checkpoint_info, timer)
    # 判断模型版本
    model.is_sdxl = hasattr(model, 'conditioner')
    model.is_sd2 = not model.is_sdxl and hasattr(model.cond_stage_model, 'model')
    model.is_sd1 = not model.is_sdxl and not model.is_sd2
    # xl模型的特殊处理
    if model.is_sdxl:
        sd_models_xl.extend_sdxl(model)
    # 模型加载传入的字典
    model.load_state_dict(state_dict, strict=False)
    # 打印
    timer.record("apply weights to model")
    # 如果设置的缓存大于0，则添加到缓存 我们可以不考虑
    if shared.opts.sd_checkpoint_cache > 0:
        # cache newly loaded model
        checkpoints_loaded[checkpoint_info] = state_dict
    # 删除字典 因为已经加载到模型里面
    del state_dict
    # 判断是否打开了opt_channelslast参数
    # 如果打开,将模型模型转换成channels_last内存格式
    if shared.cmd_opts.opt_channelslast:
        model.to(memory_format=torch.channels_last)
        timer.record("apply channels_last")
    # 判断是否打开了--no-half参数 如果开启了则不是用半精度 会拖慢速度
    if shared.cmd_opts.no_half:
        model.float()
        devices.dtype_unet = torch.float32
        timer.record("apply float()")
    else:
        # 保存VAE和depth模型对象
        vae = model.first_stage_model
        depth_model = getattr(model, 'depth_model', None)
        # 如果设置了vae 全精度则改为none
        # with --no-half-vae, remove VAE from model when doing half() to prevent its weights from being converted to float16
        if shared.cmd_opts.no_half_vae:
            model.first_stage_model = None
        # with --upcast-sampling, don't convert the depth model weights to float16
        # 如果upcast_sampling 则也设置为none
        if shared.cmd_opts.upcast_sampling and depth_model:
            model.depth_model = None
        # 设置为半精度
        model.half()
        model.first_stage_model = vae
        if depth_model:
            model.depth_model = depth_model

        devices.dtype_unet = torch.float16
        timer.record("apply half()")
    # 判断是否需要upcast采样
    devices.unet_needs_upcast = shared.cmd_opts.upcast_sampling and devices.dtype == torch.float16 and devices.dtype_unet == torch.float16
    # 将模型的第一段模型也即是VAE模型 加载到驱动上
    model.first_stage_model.to(devices.dtype_vae)
    timer.record("apply dtype to VAE")

    # clean up cache if limit is reached
    # 缓存的模型超了 则需要删除一个 我们不用
    while len(checkpoints_loaded) > shared.opts.sd_checkpoint_cache:
        checkpoints_loaded.popitem(last=False)
    # 属性赋值
    model.sd_model_hash = sd_model_hash
    model.sd_model_checkpoint = checkpoint_info.filename
    model.sd_checkpoint_info = checkpoint_info
    shared.opts.data["sd_checkpoint_hash"] = checkpoint_info.sha256
    # 型对象是否有属性'logvar' 我们应该用不到
    if hasattr(model, 'logvar'):
        model.logvar = model.logvar.to(devices.device)  # fix for training
    # 删除之前保存的base VAE模型
    sd_vae.delete_base_vae()
    # 清空之前加载的VAE信息
    sd_vae.clear_loaded_vae()
    # 根据检查点文件名解析出VAE文件路径和来源
    vae_file, vae_source = sd_vae.resolve_vae(checkpoint_info.filename).tuple()
    # 调用函数加载VAE模型
    sd_vae.load_vae(model, vae_file, vae_source)
    timer.record("load VAE")


def enable_midas_autodownload():
    """
    Gives the ldm.modules.midas.api.load_model function automatic downloading.

    When the 512-depth-ema model, and other future models like it, is loaded,
    it calls midas.api.load_model to load the associated midas depth model.
    This function applies a wrapper to download the model to the correct
    location automatically.
    """

    midas_path = os.path.join(paths.models_path, 'midas')

    # stable-diffusion-stability-ai hard-codes the midas model path to
    # a location that differs from where other scripts using this model look.
    # HACK: Overriding the path here.
    for k, v in midas.api.ISL_PATHS.items():
        file_name = os.path.basename(v)
        midas.api.ISL_PATHS[k] = os.path.join(midas_path, file_name)

    midas_urls = {
        "dpt_large": "https://github.com/intel-isl/DPT/releases/download/1_0/dpt_large-midas-2f21e586.pt",
        "dpt_hybrid": "https://github.com/intel-isl/DPT/releases/download/1_0/dpt_hybrid-midas-501f0c75.pt",
        "midas_v21": "https://github.com/AlexeyAB/MiDaS/releases/download/midas_dpt/midas_v21-f6b98070.pt",
        "midas_v21_small": "https://github.com/AlexeyAB/MiDaS/releases/download/midas_dpt/midas_v21_small-70d6b9c8.pt",
    }

    midas.api.load_model_inner = midas.api.load_model

    def load_model_wrapper(model_type):
        path = midas.api.ISL_PATHS[model_type]
        if not os.path.exists(path):
            if not os.path.exists(midas_path):
                mkdir(midas_path)

            print(f"Downloading midas model weights for {model_type} to {path}")
            request.urlretrieve(midas_urls[model_type], path)
            print(f"{model_type} downloaded")

        return midas.api.load_model_inner(model_type)

    midas.api.load_model = load_model_wrapper


def repair_config(sd_config):
    if not hasattr(sd_config.model.params, "use_ema"):
        sd_config.model.params.use_ema = False

    if hasattr(sd_config.model.params, 'unet_config'):
        if shared.cmd_opts.no_half:
            sd_config.model.params.unet_config.params.use_fp16 = False
        elif shared.cmd_opts.upcast_sampling:
            sd_config.model.params.unet_config.params.use_fp16 = True

    if getattr(sd_config.model.params.first_stage_config.params.ddconfig, "attn_type",
               None) == "vanilla-xformers" and not shared.xformers_available:
        sd_config.model.params.first_stage_config.params.ddconfig.attn_type = "vanilla"

    # For UnCLIP-L, override the hardcoded karlo directory
    if hasattr(sd_config.model.params, "noise_aug_config") and hasattr(sd_config.model.params.noise_aug_config.params,
                                                                       "clip_stats_path"):
        karlo_path = os.path.join(paths.models_path, 'karlo')
        sd_config.model.params.noise_aug_config.params.clip_stats_path = sd_config.model.params.noise_aug_config.params.clip_stats_path.replace(
            "checkpoints/karlo_models", karlo_path)


sd1_clip_weight = 'cond_stage_model.transformer.text_model.embeddings.token_embedding.weight'
sd2_clip_weight = 'cond_stage_model.model.transformer.resblocks.0.attn.in_proj_weight'
sdxl_clip_weight = 'conditioner.embedders.1.model.ln_final.weight'
sdxl_refiner_clip_weight = 'conditioner.embedders.0.model.ln_final.weight'


class SdModelData:
    def __init__(self):
        self.sd_model = None
        self.loaded_sd_models = []
        self.was_loaded_at_least_once = False
        self.lock = threading.Lock()

    def get_sd_model(self):
        if self.was_loaded_at_least_once:
            return self.sd_model

        if self.sd_model is None:
            with self.lock:
                if self.sd_model is not None or self.was_loaded_at_least_once:
                    return self.sd_model

                try:
                    load_model()

                except Exception as e:
                    errors.display(e, "loading stable diffusion model", full_traceback=True)
                    print("", file=sys.stderr)
                    print("Stable diffusion model failed to load", file=sys.stderr)
                    self.sd_model = None

        return self.sd_model

    def set_sd_model(self, v, already_loaded=False):
        self.sd_model = v
        if already_loaded:
            sd_vae.base_vae = getattr(v, "base_vae", None)
            sd_vae.loaded_vae_file = getattr(v, "loaded_vae_file", None)
            sd_vae.checkpoint_info = v.sd_checkpoint_info

        try:
            self.loaded_sd_models.remove(v)
        except ValueError:
            pass

        if v is not None:
            self.loaded_sd_models.insert(0, v)


model_data = SdModelData()


def get_empty_cond(sd_model):
    p = processing.StableDiffusionProcessingTxt2Img()
    extra_networks.activate(p, {})

    if hasattr(sd_model, 'conditioner'):
        d = sd_model.get_learned_conditioning([""])
        return d['crossattn']
    else:
        return sd_model.cond_stage_model([""])


def send_model_to_cpu(m):
    if m.lowvram:
        lowvram.send_everything_to_cpu()
    else:
        m.to(devices.cpu)

    devices.torch_gc()


def model_target_device(m):
    if lowvram.is_needed(m):
        return devices.cpu
    else:
        return devices.device


def send_model_to_device(m):
    lowvram.apply(m)

    if not m.lowvram:
        m.to(shared.device)


def send_model_to_trash(m):
    m.to(device="meta")
    devices.torch_gc()


def load_model(checkpoint_info=None, already_loaded_state_dict=None):
    from modules import sd_hijack
    checkpoint_info = checkpoint_info or select_checkpoint()

    timer = Timer()

    if model_data.sd_model:
        send_model_to_trash(model_data.sd_model)
        model_data.sd_model = None
        devices.torch_gc()

    timer.record("unload existing model")

    if already_loaded_state_dict is not None:
        state_dict = already_loaded_state_dict
    else:
        state_dict = get_checkpoint_state_dict(checkpoint_info, timer)

    checkpoint_config = sd_models_config.find_checkpoint_config(state_dict, checkpoint_info)
    clip_is_included_into_sd = any(
        x for x in [sd1_clip_weight, sd2_clip_weight, sdxl_clip_weight, sdxl_refiner_clip_weight] if x in state_dict)

    timer.record("find config")

    sd_config = OmegaConf.load(checkpoint_config)
    repair_config(sd_config)

    timer.record("load config")

    print(f"Creating model from config: {checkpoint_config}")

    sd_model = None
    try:
        with sd_disable_initialization.DisableInitialization(
                disable_clip=clip_is_included_into_sd or shared.cmd_opts.do_not_download_clip):
            with sd_disable_initialization.InitializeOnMeta():
                sd_model = instantiate_from_config(sd_config.model)

    except Exception as e:
        errors.display(e, "creating model quickly", full_traceback=True)

    if sd_model is None:
        print('Failed to create model quickly; will retry using slow method.', file=sys.stderr)

        with sd_disable_initialization.InitializeOnMeta():
            sd_model = instantiate_from_config(sd_config.model)

    sd_model.used_config = checkpoint_config

    timer.record("create model")

    if shared.cmd_opts.no_half:
        weight_dtype_conversion = None
    else:
        weight_dtype_conversion = {
            'first_stage_model': None,
            '': torch.float16,
        }

    with sd_disable_initialization.LoadStateDictOnMeta(state_dict, device=model_target_device(sd_model),
                                                       weight_dtype_conversion=weight_dtype_conversion):
        load_model_weights(sd_model, checkpoint_info, state_dict, timer)
    timer.record("load weights from state dict")

    send_model_to_device(sd_model)
    timer.record("move model to device")

    sd_hijack.model_hijack.hijack(sd_model)

    timer.record("hijack")

    sd_model.eval()
    model_data.set_sd_model(sd_model)
    model_data.was_loaded_at_least_once = True

    sd_hijack.model_hijack.embedding_db.load_textual_inversion_embeddings(
        force_reload=True)  # Reload embeddings after model load as they may or may not fit the model

    timer.record("load textual inversion embeddings")

    script_callbacks.model_loaded_callback(sd_model)

    timer.record("scripts callbacks")

    with devices.autocast(), torch.no_grad():
        sd_model.cond_stage_model_empty_prompt = get_empty_cond(sd_model)

    timer.record("calculate empty prompt")

    print(f"Model loaded in {timer.summary()}.")

    return sd_model


def reuse_model_from_already_loaded(sd_model, checkpoint_info, timer):
    """
    Checks if the desired checkpoint from checkpoint_info is not already loaded in model_data.loaded_sd_models.
    If it is loaded, returns that (moving it to GPU if necessary, and moving the currently loadded model to CPU if necessary).
    If not, returns the model that can be used to load weights from checkpoint_info's file.
    If no such model exists, returns None.
    Additionaly deletes loaded models that are over the limit set in settings (sd_checkpoints_limit).
    """
    # 初始化一个 already_loaded
    already_loaded = None
    # 遍历 model_data.loaded_sd_models 中的所有模型，检查是否有与 checkpoint_info.filename 匹配的模型。
    # 如果找到了匹配的模型，就将其设置为 already_loaded。
    for i in reversed(range(len(model_data.loaded_sd_models))):
        # 从loaded_sd_models中取出已经加载的模型
        loaded_model = model_data.loaded_sd_models[i]
        # 如果传入的filename和当前已经加载的模型的filename相同则 将换成的模型赋值给already_loaded
        if loaded_model.sd_checkpoint_info.filename == checkpoint_info.filename:
            already_loaded = loaded_model
            continue
        # 检查是否已加载的模型数量超过了 shared.opts.sd_checkpoints_limit 设置的限制。
        # 如果超过了限制，就卸载多余的模型，将其发送到垃圾回收。
        if len(model_data.loaded_sd_models) > shared.opts.sd_checkpoints_limit > 0:
            print(
                f"Unloading model {len(model_data.loaded_sd_models)} over the limit of {shared.opts.sd_checkpoints_limit}: {loaded_model.sd_checkpoint_info.title}")
            model_data.loaded_sd_models.pop()
            send_model_to_trash(loaded_model)
            timer.record("send model to trash")
        # 如果加载的模型设置要在cpu 中则加载模型到cpu  到时候可以去掉
        if shared.opts.sd_checkpoints_keep_in_cpu:
            send_model_to_cpu(sd_model)
            timer.record("send model to cpu")
    # 如果already_loaded 不为空 也就意味着已经加载过了
    if already_loaded is not None:
        # 调用此方法 将此模型数据加载到cpu中
        send_model_to_device(already_loaded)
        # 打印
        timer.record("send model to device")
        # 将当前的模型set到model_data的sd_mode里面
        model_data.set_sd_model(already_loaded, already_loaded=True)
        # 这个设置 不知道在哪里  但是应该是吧当前模型的名字和hash 加载到shared对应的json 里  可以不考虑
        if not SkipWritingToConfig.skip:
            shared.opts.data["sd_model_checkpoint"] = already_loaded.sd_checkpoint_info.title
            shared.opts.data["sd_checkpoint_hash"] = already_loaded.sd_checkpoint_info.sha256
        # 打印重新加载模型完成
        print(f"Using already loaded model {already_loaded.sd_checkpoint_info.title}: done in {timer.summary()}")
        # 重新加载对应的VAE
        sd_vae.reload_vae_weights(already_loaded)
        # 返回sd_model
        return model_data.sd_model
    elif shared.opts.sd_checkpoints_limit > 1 and len(model_data.loaded_sd_models) < shared.opts.sd_checkpoints_limit:
        print(
            f"Loading model {checkpoint_info.title} ({len(model_data.loaded_sd_models) + 1} out of {shared.opts.sd_checkpoints_limit})")

        model_data.sd_model = None
        load_model(checkpoint_info)
        return model_data.sd_model
    elif len(model_data.loaded_sd_models) > 0:
        sd_model = model_data.loaded_sd_models.pop()
        model_data.sd_model = sd_model

        sd_vae.base_vae = getattr(sd_model, "base_vae", None)
        sd_vae.loaded_vae_file = getattr(sd_model, "loaded_vae_file", None)
        sd_vae.checkpoint_info = sd_model.sd_checkpoint_info

        print(f"Reusing loaded model {sd_model.sd_checkpoint_info.title} to load {checkpoint_info.title}")
        return sd_model
    else:
        return None


# 实现模型 weights 参数重新加载的功能
def reload_model_weights(sd_model=None, info=None):
    # 如果传入模型的info 要不然使用选择当前选中的项目的模型的info
    checkpoint_info = info or select_checkpoint()
    # 创建定时器对象开始计时
    timer = Timer()
    # 如果不传入模型,则从model_data中获取全局模型
    if not sd_model:
        sd_model = model_data.sd_model
    # 如果模型为空,说明上次加载失败,当前检查点信息为空
    if sd_model is None:  # previous model load failed
        current_checkpoint_info = None
    else:
        # 如果不为空则 加载当前模型的info
        current_checkpoint_info = sd_model.sd_checkpoint_info
        # 如果文件名相同直接返回模型   这个逻辑没看懂  
        if sd_model.sd_model_checkpoint == checkpoint_info.filename:
            return sd_model
    # 如果参数传入的sd_model为空的时候 根据传入的checkpoint_info从缓存中找是否有对应的模型
    sd_model = reuse_model_from_already_loaded(sd_model, checkpoint_info, timer)
    # 如果在缓存中找到了 则直接返回并且filename 相同
    if sd_model is not None and sd_model.sd_checkpoint_info.filename == checkpoint_info.filename:
        return sd_model
    # 这种情况 没看明白 感觉是没用的代码
    if sd_model is not None:
        sd_unet.apply_unet("None")
        send_model_to_cpu(sd_model)
        sd_hijack.model_hijack.undo_hijack(sd_model)
    # 这种是在缓存的模型中没找到我们对应的info的模型，然后从磁盘或者缓存中加载向量数据，也就是状态字典
    state_dict = get_checkpoint_state_dict(checkpoint_info, timer)
    # 根据状态字典和checkpoint_info 去匹配config文件
    checkpoint_config = sd_models_config.find_checkpoint_config(state_dict, checkpoint_info)
    # 打印
    timer.record("find config")
    # 判断模型没加载成功或者和配置文件不匹配
    if sd_model is None or checkpoint_config != sd_model.used_config:
        # 回收模型
        if sd_model is not None:
            send_model_to_trash(sd_model)
        # 重新加载模型 根据info和字典 暂时先跳过没懂
        load_model(checkpoint_info, already_loaded_state_dict=state_dict)
        return model_data.sd_model

    try:
        # 加载模型的字典和VAE
        load_model_weights(sd_model, checkpoint_info, state_dict, timer)
    except Exception:
        print("Failed to load checkpoint, restoring previous")
        load_model_weights(sd_model, current_checkpoint_info, None, timer)
        raise
    finally:

        sd_hijack.model_hijack.hijack(sd_model)
        timer.record("hijack")

        script_callbacks.model_loaded_callback(sd_model)
        timer.record("script callbacks")

        if not sd_model.lowvram:
            sd_model.to(devices.device)
            timer.record("move model to device")

    print(f"Weights loaded in {timer.summary()}.")

    model_data.set_sd_model(sd_model)
    sd_unet.apply_unet()

    return sd_model


def unload_model_weights(sd_model=None, info=None):
    timer = Timer()

    if model_data.sd_model:
        # 将sd_model移动到CPU
        model_data.sd_model.to(devices.cpu)
        sd_hijack.model_hijack.undo_hijack(model_data.sd_model)
        model_data.sd_model = None
        sd_model = None
        gc.collect()
        devices.torch_gc()

    print(f"Unloaded weights {timer.summary()}.")

    return sd_model


def apply_token_merging(sd_model, token_merging_ratio):
    """
    Applies speed and memory optimizations from tomesd.
    """

    current_token_merging_ratio = getattr(sd_model, 'applied_token_merged_ratio', 0)

    if current_token_merging_ratio == token_merging_ratio:
        return

    if current_token_merging_ratio > 0:
        tomesd.remove_patch(sd_model)

    if token_merging_ratio > 0:
        tomesd.apply_patch(
            sd_model,
            ratio=token_merging_ratio,
            use_rand=False,  # can cause issues with some samplers
            merge_attn=True,
            merge_crossattn=False,
            merge_mlp=False
        )

    sd_model.applied_token_merged_ratio = token_merging_ratio
