from modules import sd_samplers_kdiffusion, sd_samplers_timesteps, shared

# imports for functions that previously were here and are used by other modules
from modules.sd_samplers_common import samples_to_image_grid, sample_to_image  # noqa: F401

# * 操作符用来展开列表或者元组
# 所以 all_samplers 的值是 all_samplers内部两个列表元素合并后组成的列表。
all_samplers = [
    *sd_samplers_kdiffusion.samplers_data_k_diffusion,
    *sd_samplers_timesteps.samplers_data_timesteps,
]
all_samplers_map = {x.name: x for x in all_samplers}

samplers = []
samplers_for_img2img = []
samplers_map = {}
samplers_hidden = {}


def find_sampler_config(name):
    """
    根据name查找采样器
    """
    if name is not None:
        config = all_samplers_map.get(name, None)
    else:
        config = all_samplers[0]

    return config


def create_sampler(name, model):
    config = find_sampler_config(name)
    # 根据name没找到采样器 会抛出AssertionError，异常信息是'bad sampler name: {name}'。
    assert config is not None, f'bad sampler name: {name}'

    # 检查模型是否为SDXL模型，并且在配置中是否指定了"no_sdxl"。
    if model.is_sdxl and config.options.get("no_sdxl", False):
        # 抛出异常，表示该采样器不支持在SDXL模型中使用。
        raise Exception(f"Sampler {config.name} is not supported for SDXL")

    sampler = config.constructor(model)
    sampler.config = config

    return sampler


def set_samplers():
    global samplers, samplers_for_img2img, samplers_hidden
    # shared.opts.hide_samplers 用户在 设置界面 勾选隐藏的 采样器
    samplers_hidden = set(shared.opts.hide_samplers)
    samplers = all_samplers
    samplers_for_img2img = all_samplers

    samplers_map.clear()
    # 将采样器名字|别名小写作为key，采样器名字作为value存入samplers_map字典
    for sampler in all_samplers:
        samplers_map[sampler.name.lower()] = sampler.name
        for alias in sampler.aliases:
            samplers_map[alias.lower()] = sampler.name


# 获取未被用户设置为隐藏的采样器名字
def visible_sampler_names():
    return [x.name for x in samplers if x.name not in samplers_hidden]


# 导入该模块时执行该方法
set_samplers()
