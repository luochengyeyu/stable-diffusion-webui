from __future__ import annotations
import json
import logging
import math
import os
import sys
import hashlib
from dataclasses import dataclass, field

import torch
import numpy as np
from PIL import Image, ImageOps
import random
import cv2
from skimage import exposure
from typing import Any

import modules.sd_hijack
from modules import devices, prompt_parser, masking, sd_samplers, lowvram, generation_parameters_copypaste, \
    extra_networks, sd_vae_approx, scripts, sd_samplers_common, sd_unet, errors, rng
from modules.rng import slerp  # noqa: F401
from modules.sd_hijack import model_hijack
from modules.sd_samplers_common import images_tensor_to_samples, decode_first_stage, approximation_indexes
from modules.shared import opts, cmd_opts, state
import modules.shared as shared
import modules.paths as paths
import modules.face_restoration
import modules.images as images
import modules.styles
import modules.sd_models as sd_models
import modules.sd_vae as sd_vae
from ldm.data.util import AddMiDaS
from ldm.models.diffusion.ddpm import LatentDepth2ImageDiffusion

from einops import repeat, rearrange
from blendmodes.blend import blendLayers, BlendType

# some of those options should not be changed at all because they would break the model, so I removed them from options.
opt_C = 4
opt_f = 8


def setup_color_correction(image):
    logging.info("Calibrating color correction.")
    correction_target = cv2.cvtColor(np.asarray(image.copy()), cv2.COLOR_RGB2LAB)
    return correction_target


def apply_color_correction(correction, original_image):
    logging.info("Applying color correction.")
    image = Image.fromarray(cv2.cvtColor(exposure.match_histograms(
        cv2.cvtColor(
            np.asarray(original_image),
            cv2.COLOR_RGB2LAB
        ),
        correction,
        channel_axis=2
    ), cv2.COLOR_LAB2RGB).astype("uint8"))

    image = blendLayers(image, original_image, BlendType.LUMINOSITY)

    return image.convert('RGB')


def apply_overlay(image, paste_loc, index, overlays):
    if overlays is None or index >= len(overlays):
        return image

    overlay = overlays[index]

    if paste_loc is not None:
        x, y, w, h = paste_loc
        base_image = Image.new('RGBA', (overlay.width, overlay.height))
        image = images.resize_image(1, image, w, h)
        base_image.paste(image, (x, y))
        image = base_image

    image = image.convert('RGBA')
    image.alpha_composite(overlay)
    image = image.convert('RGB')

    return image


def create_binary_mask(image):
    if image.mode == 'RGBA' and image.getextrema()[-1] != (255, 255):
        image = image.split()[-1].convert("L").point(lambda x: 255 if x > 128 else 0)
    else:
        image = image.convert('L')
    return image


def txt2img_image_conditioning(sd_model, x, width, height):
    if sd_model.model.conditioning_key in {'hybrid', 'concat'}:  # Inpainting models

        # The "masked-image" in this case will just be all 0.5 since the entire image is masked.
        image_conditioning = torch.ones(x.shape[0], 3, height, width, device=x.device) * 0.5
        image_conditioning = images_tensor_to_samples(image_conditioning,
                                                      approximation_indexes.get(opts.sd_vae_encode_method))

        # Add the fake full 1s mask to the first dimension.
        image_conditioning = torch.nn.functional.pad(image_conditioning, (0, 0, 0, 0, 1, 0), value=1.0)
        image_conditioning = image_conditioning.to(x.dtype)

        return image_conditioning

    elif sd_model.model.conditioning_key == "crossattn-adm":  # UnCLIP models

        return x.new_zeros(x.shape[0], 2 * sd_model.noise_augmentor.time_embed.dim, dtype=x.dtype, device=x.device)

    else:
        # Dummy zero conditioning if we're not using inpainting or unclip models.
        # Still takes up a bit of memory, but no encoder call.
        # Pretty sure we can just make this a 1x1 image since its not going to be used besides its batch size.
        return x.new_zeros(x.shape[0], 5, 1, 1, dtype=x.dtype, device=x.device)


@dataclass(repr=False)
class StableDiffusionProcessing:
    sd_model: object = None
    outpath_samples: str = None
    outpath_grids: str = None
    prompt: str = ""
    prompt_for_display: str = None
    negative_prompt: str = ""
    styles: list[str] = None
    seed: int = -1
    subseed: int = -1
    subseed_strength: float = 0
    seed_resize_from_h: int = -1
    seed_resize_from_w: int = -1
    seed_enable_extras: bool = True
    sampler_name: str = None
    batch_size: int = 1
    n_iter: int = 1
    steps: int = 50
    cfg_scale: float = 7.0
    width: int = 512
    height: int = 512
    restore_faces: bool = None
    tiling: bool = None
    do_not_save_samples: bool = False
    do_not_save_grid: bool = False
    extra_generation_params: dict[str, Any] = None
    overlay_images: list = None
    eta: float = None
    do_not_reload_embeddings: bool = False
    denoising_strength: float = 0
    ddim_discretize: str = None
    s_min_uncond: float = None
    s_churn: float = None
    s_tmax: float = None
    s_tmin: float = None
    s_noise: float = None
    override_settings: dict[str, Any] = None
    override_settings_restore_afterwards: bool = True
    sampler_index: int = None
    refiner_checkpoint: str = None
    refiner_switch_at: float = None
    token_merging_ratio = 0
    token_merging_ratio_hr = 0
    disable_extra_networks: bool = False

    scripts_value: scripts.ScriptRunner = field(default=None, init=False)
    script_args_value: list = field(default=None, init=False)
    scripts_setup_complete: bool = field(default=False, init=False)

    cached_uc = [None, None]
    cached_c = [None, None]

    comments: dict = None
    sampler: sd_samplers_common.Sampler | None = field(default=None, init=False)
    is_using_inpainting_conditioning: bool = field(default=False, init=False)
    paste_to: tuple | None = field(default=None, init=False)

    is_hr_pass: bool = field(default=False, init=False)

    c: tuple = field(default=None, init=False)
    uc: tuple = field(default=None, init=False)

    rng: rng.ImageRNG | None = field(default=None, init=False)
    step_multiplier: int = field(default=1, init=False)
    color_corrections: list = field(default=None, init=False)

    all_prompts: list = field(default=None, init=False)
    all_negative_prompts: list = field(default=None, init=False)
    all_seeds: list = field(default=None, init=False)
    all_subseeds: list = field(default=None, init=False)
    iteration: int = field(default=0, init=False)
    main_prompt: str = field(default=None, init=False)
    main_negative_prompt: str = field(default=None, init=False)

    prompts: list = field(default=None, init=False)
    negative_prompts: list = field(default=None, init=False)
    seeds: list = field(default=None, init=False)
    subseeds: list = field(default=None, init=False)
    extra_network_data: dict = field(default=None, init=False)

    user: str = field(default=None, init=False)

    sd_model_name: str = field(default=None, init=False)
    sd_model_hash: str = field(default=None, init=False)
    sd_vae_name: str = field(default=None, init=False)
    sd_vae_hash: str = field(default=None, init=False)

    is_api: bool = field(default=False, init=False)

    # 主要用于创建对象后的参数的校验
    # 它在自动生成的 __init__ 方法完成后立即执行。通常用于进行一些初始化工作，比如参数校验、参数转换等。
    def __post_init__(self):
        # 这行代码检查 sampler_index 属性是否不为 None，如果不为 None，则向标准错误输出打印一条消息，
        # 说明 sampler_index 参数对StableDiffusionProcessing 无效，应该使用 sampler_name。
        if self.sampler_index is not None:
            print("sampler_index argument for StableDiffusionProcessing does not do anything; use sampler_name",
                  file=sys.stderr)
        # 创建一个空字典作为 comments 属性 不是初始化时传入的数据 所以在此方法中进行初始化
        self.comments = {}
        # 如果 styles 属性为 None，则设置为一个空列表。
        if self.styles is None:
            self.styles = []
        # 创建一个 sampler_noise_scheduler_override 属性，并设置为 None。
        self.sampler_noise_scheduler_override = None
        # 如果相应的属性为 None，则从 opts 对象中获取默认值。
        self.s_min_uncond = self.s_min_uncond if self.s_min_uncond is not None else opts.s_min_uncond
        self.s_churn = self.s_churn if self.s_churn is not None else opts.s_churn
        self.s_tmin = self.s_tmin if self.s_tmin is not None else opts.s_tmin
        self.s_tmax = (self.s_tmax if self.s_tmax is not None else opts.s_tmax) or float('inf')
        self.s_noise = self.s_noise if self.s_noise is not None else opts.s_noise
        # 如果相应的属性为 None 或者为假（如空列表、空字典等），则设置为一个新的空字典。
        self.extra_generation_params = self.extra_generation_params or {}
        self.override_settings = self.override_settings or {}
        self.script_args = self.script_args or {}
        # xl的refiner_checkpoint_info 赋值为none    
        self.refiner_checkpoint_info = None
        # seed_enable_extras 为假时执行，默认设置一些与种子（seed）相关的属性
        if not self.seed_enable_extras:
            self.subseed = -1
            self.subseed_strength = 0
            self.seed_resize_from_h = 0
            self.seed_resize_from_w = 0
        # 从 StableDiffusionProcessing 类获取 cached_uc 和 cached_c 属性的值，复制到当前实例。
        self.cached_uc = StableDiffusionProcessing.cached_uc
        self.cached_c = StableDiffusionProcessing.cached_c

    # 方法转属性的标签 也就是说他的sd_model 返回的是shared的sd_model
    @property
    def sd_model(self):
        return shared.sd_model

    # 定义了上面的参数sd_model的set方法 但是这方法其实啥也不做
    @sd_model.setter
    def sd_model(self, value):
        pass

    # 方法转属性的标签 这是他的get方法 也就是说他的scripts 返回的是自己的scripts_value
    @property
    def scripts(self):
        return self.scripts_value

    # 属性scripts的set方法
    @scripts.setter
    def scripts(self, value):
        # 将输入的 value 赋值给 self.scripts_value。
        self.scripts_value = value
        # 这个条件判断 self.scripts_value、self.script_args_value 是否都存在
        # 且 self.scripts_setup_complete 为 False。如果是，执行下一步。
        if self.scripts_value and self.script_args_value and not self.scripts_setup_complete:
            self.setup_scripts()

    @property
    def script_args(self):
        return self.script_args_value

    @script_args.setter
    def script_args(self, value):
        self.script_args_value = value

        if self.scripts_value and self.script_args_value and not self.scripts_setup_complete:
            self.setup_scripts()

    def setup_scripts(self):
        # 将 scripts_setup_complete 属性设置为 True
        self.scripts_setup_complete = True

        self.scripts.setup_scrips(self, is_ui=not self.is_api)

    def comment(self, text):
        self.comments[text] = 1

    def txt2img_image_conditioning(self, x, width=None, height=None):
        self.is_using_inpainting_conditioning = self.sd_model.model.conditioning_key in {'hybrid', 'concat'}

        return txt2img_image_conditioning(self.sd_model, x, width or self.width, height or self.height)

    def depth2img_image_conditioning(self, source_image):
        # Use the AddMiDaS helper to Format our source image to suit the MiDaS model
        transformer = AddMiDaS(model_type="dpt_hybrid")
        transformed = transformer({"jpg": rearrange(source_image[0], "c h w -> h w c")})
        midas_in = torch.from_numpy(transformed["midas_in"][None, ...]).to(device=shared.device)
        midas_in = repeat(midas_in, "1 ... -> n ...", n=self.batch_size)

        conditioning_image = images_tensor_to_samples(source_image * 0.5 + 0.5,
                                                      approximation_indexes.get(opts.sd_vae_encode_method))
        conditioning = torch.nn.functional.interpolate(
            self.sd_model.depth_model(midas_in),
            size=conditioning_image.shape[2:],
            mode="bicubic",
            align_corners=False,
        )

        (depth_min, depth_max) = torch.aminmax(conditioning)
        conditioning = 2. * (conditioning - depth_min) / (depth_max - depth_min) - 1.
        return conditioning

    def edit_image_conditioning(self, source_image):
        conditioning_image = images_tensor_to_samples(source_image * 0.5 + 0.5,
                                                      approximation_indexes.get(opts.sd_vae_encode_method))

        return conditioning_image

    def unclip_image_conditioning(self, source_image):
        c_adm = self.sd_model.embedder(source_image)
        if self.sd_model.noise_augmentor is not None:
            noise_level = 0  # TODO: Allow other noise levels?
            c_adm, noise_level_emb = self.sd_model.noise_augmentor(c_adm, noise_level=repeat(
                torch.tensor([noise_level]).to(c_adm.device), '1 -> b', b=c_adm.shape[0]))
            c_adm = torch.cat((c_adm, noise_level_emb), 1)
        return c_adm

    def inpainting_image_conditioning(self, source_image, latent_image, image_mask=None):
        self.is_using_inpainting_conditioning = True

        # Handle the different mask inputs
        if image_mask is not None:
            if torch.is_tensor(image_mask):
                conditioning_mask = image_mask
            else:
                conditioning_mask = np.array(image_mask.convert("L"))
                conditioning_mask = conditioning_mask.astype(np.float32) / 255.0
                conditioning_mask = torch.from_numpy(conditioning_mask[None, None])

                # Inpainting model uses a discretized mask as input, so we round to either 1.0 or 0.0
                conditioning_mask = torch.round(conditioning_mask)
        else:
            conditioning_mask = source_image.new_ones(1, 1, *source_image.shape[-2:])

        # Create another latent image, this time with a masked version of the original input.
        # Smoothly interpolate between the masked and unmasked latent conditioning image using a parameter.
        conditioning_mask = conditioning_mask.to(device=source_image.device, dtype=source_image.dtype)
        conditioning_image = torch.lerp(
            source_image,
            source_image * (1.0 - conditioning_mask),
            getattr(self, "inpainting_mask_weight", shared.opts.inpainting_mask_weight)
        )

        # Encode the new masked image using first stage of network.
        conditioning_image = self.sd_model.get_first_stage_encoding(
            self.sd_model.encode_first_stage(conditioning_image))

        # Create the concatenated conditioning tensor to be fed to `c_concat`
        conditioning_mask = torch.nn.functional.interpolate(conditioning_mask, size=latent_image.shape[-2:])
        conditioning_mask = conditioning_mask.expand(conditioning_image.shape[0], -1, -1, -1)
        image_conditioning = torch.cat([conditioning_mask, conditioning_image], dim=1)
        image_conditioning = image_conditioning.to(shared.device).type(self.sd_model.dtype)

        return image_conditioning

    def img2img_image_conditioning(self, source_image, latent_image, image_mask=None):
        source_image = devices.cond_cast_float(source_image)

        # HACK: Using introspection as the Depth2Image model doesn't appear to uniquely
        # identify itself with a field common to all models. The conditioning_key is also hybrid.
        if isinstance(self.sd_model, LatentDepth2ImageDiffusion):
            return self.depth2img_image_conditioning(source_image)

        if self.sd_model.cond_stage_key == "edit":
            return self.edit_image_conditioning(source_image)

        if self.sampler.conditioning_key in {'hybrid', 'concat'}:
            return self.inpainting_image_conditioning(source_image, latent_image, image_mask=image_mask)

        if self.sampler.conditioning_key == "crossattn-adm":
            return self.unclip_image_conditioning(source_image)

        # Dummy zero conditioning if we're not using inpainting or depth model.
        return latent_image.new_zeros(latent_image.shape[0], 5, 1, 1)

    def init(self, all_prompts, all_seeds, all_subseeds):
        pass

    def sample(self, conditioning, unconditional_conditioning, seeds, subseeds, subseed_strength, prompts):
        raise NotImplementedError()

    def close(self):
        self.sampler = None
        self.c = None
        self.uc = None
        if not opts.persistent_cond_cache:
            StableDiffusionProcessing.cached_c = [None, None]
            StableDiffusionProcessing.cached_uc = [None, None]

    def get_token_merging_ratio(self, for_hr=False):
        if for_hr:
            return self.token_merging_ratio_hr or opts.token_merging_ratio_hr or self.token_merging_ratio or opts.token_merging_ratio

        return self.token_merging_ratio or opts.token_merging_ratio

    def setup_prompts(self):
        # 先创建正向描述词的列表
        # 判断prompt是否为列表类型 是则直接设置为prompt
        if isinstance(self.prompt, list):
            self.all_prompts = self.prompt
        #  就是正向词不是列表而反向词是的时候 根据批次以及每批次的数量生成对应的关键词列表
        elif isinstance(self.negative_prompt, list):
            self.all_prompts = [self.prompt] * len(self.negative_prompt)
        else:
            # 如果都不是列表,则根据batch_size和n_iter重复prompt创建列表。
            self.all_prompts = self.batch_size * self.n_iter * [self.prompt]
        # 创建反向提示词的列表
        if isinstance(self.negative_prompt, list):
            self.all_negative_prompts = self.negative_prompt
        else:
            self.all_negative_prompts = [self.negative_prompt] * len(self.all_prompts)
        # 判断正向词列表和反向词列表的数组长度是否相等，不相等则抛出异常
        if len(self.all_prompts) != len(self.all_negative_prompts):
            raise RuntimeError(
                f"Received a different number of prompts ({len(self.all_prompts)}) and negative prompts ({len(self.all_negative_prompts)})")
        # 根据用户指定的提示词的格式 进行处理
        self.all_prompts = [shared.prompt_styles.apply_styles_to_prompt(x, self.styles) for x in self.all_prompts]
        self.all_negative_prompts = [shared.prompt_styles.apply_negative_styles_to_prompt(x, self.styles) for x in
                                     self.all_negative_prompts]
        # 将第一个prompt和negative prompt设置为main_prompt和main_negative_prompt
        self.main_prompt = self.all_prompts[0]
        self.main_negative_prompt = self.all_negative_prompts[0]

    def cached_params(self, required_prompts, steps, extra_network_data, hires_steps=None, use_old_scheduling=False):
        """Returns parameters that invalidate the cond cache if changed"""

        return (
            required_prompts,
            steps,
            hires_steps,
            use_old_scheduling,
            opts.CLIP_stop_at_last_layers,
            shared.sd_model.sd_checkpoint_info,
            extra_network_data,
            opts.sdxl_crop_left,
            opts.sdxl_crop_top,
            self.width,
            self.height,
        )

    def get_conds_with_caching(self, function, required_prompts, steps, caches, extra_network_data, hires_steps=None):
        """
        Returns the result of calling function(shared.sd_model, required_prompts, steps)
        using a cache to store the result if the same arguments have been used before.

        cache is an array containing two elements. The first element is a tuple
        representing the previously used arguments, or None if no arguments
        have been used before. The second element is where the previously
        computed result is stored.

        caches is a list with items described above.
        """

        if shared.opts.use_old_scheduling:
            old_schedules = prompt_parser.get_learned_conditioning_prompt_schedules(required_prompts, steps,
                                                                                    hires_steps, False)
            new_schedules = prompt_parser.get_learned_conditioning_prompt_schedules(required_prompts, steps,
                                                                                    hires_steps, True)
            if old_schedules != new_schedules:
                self.extra_generation_params["Old prompt editing timelines"] = True

        cached_params = self.cached_params(required_prompts, steps, extra_network_data, hires_steps,
                                           shared.opts.use_old_scheduling)

        for cache in caches:
            if cache[0] is not None and cached_params == cache[0]:
                return cache[1]

        cache = caches[0]

        with devices.autocast():
            cache[1] = function(shared.sd_model, required_prompts, steps, hires_steps, shared.opts.use_old_scheduling)

        cache[0] = cached_params
        return cache[1]

    def setup_conds(self):
        prompts = prompt_parser.SdConditioning(self.prompts, width=self.width, height=self.height)
        negative_prompts = prompt_parser.SdConditioning(self.negative_prompts, width=self.width, height=self.height,
                                                        is_negative_prompt=True)

        sampler_config = sd_samplers.find_sampler_config(self.sampler_name)
        total_steps = sampler_config.total_steps(self.steps) if sampler_config else self.steps
        self.step_multiplier = total_steps // self.steps
        self.firstpass_steps = total_steps

        self.uc = self.get_conds_with_caching(prompt_parser.get_learned_conditioning, negative_prompts, total_steps,
                                              [self.cached_uc], self.extra_network_data)
        self.c = self.get_conds_with_caching(prompt_parser.get_multicond_learned_conditioning, prompts, total_steps,
                                             [self.cached_c], self.extra_network_data)

    def get_conds(self):
        return self.c, self.uc

    def parse_extra_network_prompts(self):
        self.prompts, self.extra_network_data = extra_networks.parse_prompts(self.prompts)

    def save_samples(self) -> bool:
        """Returns whether generated images need to be written to disk"""
        return opts.samples_save and not self.do_not_save_samples and (
                opts.save_incomplete_images or not state.interrupted and not state.skipped)


class Processed:
    def __init__(self, p: StableDiffusionProcessing, images_list, seed=-1, info="", subseed=None, all_prompts=None,
                 all_negative_prompts=None, all_seeds=None, all_subseeds=None, index_of_first_image=0, infotexts=None,
                 comments=""):
        self.images = images_list
        self.prompt = p.prompt
        self.negative_prompt = p.negative_prompt
        self.seed = seed
        self.subseed = subseed
        self.subseed_strength = p.subseed_strength
        self.info = info
        self.comments = "".join(f"{comment}\n" for comment in p.comments)
        self.width = p.width
        self.height = p.height
        self.sampler_name = p.sampler_name
        self.cfg_scale = p.cfg_scale
        self.image_cfg_scale = getattr(p, 'image_cfg_scale', None)
        self.steps = p.steps
        self.batch_size = p.batch_size
        self.restore_faces = p.restore_faces
        self.face_restoration_model = opts.face_restoration_model if p.restore_faces else None
        self.sd_model_name = p.sd_model_name
        self.sd_model_hash = p.sd_model_hash
        self.sd_vae_name = p.sd_vae_name
        self.sd_vae_hash = p.sd_vae_hash
        self.seed_resize_from_w = p.seed_resize_from_w
        self.seed_resize_from_h = p.seed_resize_from_h
        self.denoising_strength = getattr(p, 'denoising_strength', None)
        self.extra_generation_params = p.extra_generation_params
        self.index_of_first_image = index_of_first_image
        self.styles = p.styles
        self.job_timestamp = state.job_timestamp
        self.clip_skip = opts.CLIP_stop_at_last_layers
        self.token_merging_ratio = p.token_merging_ratio
        self.token_merging_ratio_hr = p.token_merging_ratio_hr

        self.eta = p.eta
        self.ddim_discretize = p.ddim_discretize
        self.s_churn = p.s_churn
        self.s_tmin = p.s_tmin
        self.s_tmax = p.s_tmax
        self.s_noise = p.s_noise
        self.s_min_uncond = p.s_min_uncond
        self.sampler_noise_scheduler_override = p.sampler_noise_scheduler_override
        self.prompt = self.prompt if not isinstance(self.prompt, list) else self.prompt[0]
        self.negative_prompt = self.negative_prompt if not isinstance(self.negative_prompt, list) else \
            self.negative_prompt[0]
        self.seed = int(self.seed if not isinstance(self.seed, list) else self.seed[0]) if self.seed is not None else -1
        self.subseed = int(
            self.subseed if not isinstance(self.subseed, list) else self.subseed[0]) if self.subseed is not None else -1
        self.is_using_inpainting_conditioning = p.is_using_inpainting_conditioning

        self.all_prompts = all_prompts or p.all_prompts or [self.prompt]
        self.all_negative_prompts = all_negative_prompts or p.all_negative_prompts or [self.negative_prompt]
        self.all_seeds = all_seeds or p.all_seeds or [self.seed]
        self.all_subseeds = all_subseeds or p.all_subseeds or [self.subseed]
        self.infotexts = infotexts or [info]

    def js(self):
        obj = {
            "prompt": self.all_prompts[0],
            "all_prompts": self.all_prompts,
            "negative_prompt": self.all_negative_prompts[0],
            "all_negative_prompts": self.all_negative_prompts,
            "seed": self.seed,
            "all_seeds": self.all_seeds,
            "subseed": self.subseed,
            "all_subseeds": self.all_subseeds,
            "subseed_strength": self.subseed_strength,
            "width": self.width,
            "height": self.height,
            "sampler_name": self.sampler_name,
            "cfg_scale": self.cfg_scale,
            "steps": self.steps,
            "batch_size": self.batch_size,
            "restore_faces": self.restore_faces,
            "face_restoration_model": self.face_restoration_model,
            "sd_model_name": self.sd_model_name,
            "sd_model_hash": self.sd_model_hash,
            "sd_vae_name": self.sd_vae_name,
            "sd_vae_hash": self.sd_vae_hash,
            "seed_resize_from_w": self.seed_resize_from_w,
            "seed_resize_from_h": self.seed_resize_from_h,
            "denoising_strength": self.denoising_strength,
            "extra_generation_params": self.extra_generation_params,
            "index_of_first_image": self.index_of_first_image,
            "infotexts": self.infotexts,
            "styles": self.styles,
            "job_timestamp": self.job_timestamp,
            "clip_skip": self.clip_skip,
            "is_using_inpainting_conditioning": self.is_using_inpainting_conditioning,
        }

        return json.dumps(obj)

    def infotext(self, p: StableDiffusionProcessing, index):
        return create_infotext(p, self.all_prompts, self.all_seeds, self.all_subseeds, comments=[],
                               position_in_batch=index % self.batch_size, iteration=index // self.batch_size)

    def get_token_merging_ratio(self, for_hr=False):
        return self.token_merging_ratio_hr if for_hr else self.token_merging_ratio


def create_random_tensors(shape, seeds, subseeds=None, subseed_strength=0.0, seed_resize_from_h=0, seed_resize_from_w=0,
                          p=None):
    g = rng.ImageRNG(shape, seeds, subseeds=subseeds, subseed_strength=subseed_strength,
                     seed_resize_from_h=seed_resize_from_h, seed_resize_from_w=seed_resize_from_w)
    return g.next()


class DecodedSamples(list):
    already_decoded = True


def decode_latent_batch(model, batch, target_device=None, check_for_nans=False):
    samples = DecodedSamples()

    for i in range(batch.shape[0]):
        sample = decode_first_stage(model, batch[i:i + 1])[0]

        if check_for_nans:
            try:
                devices.test_for_nans(sample, "vae")
            except devices.NansException as e:
                if devices.dtype_vae == torch.float32 or not shared.opts.auto_vae_precision:
                    raise e

                errors.print_error_explanation(
                    "A tensor with all NaNs was produced in VAE.\n"
                    "Web UI will now convert VAE into 32-bit float and retry.\n"
                    "To disable this behavior, disable the 'Automatically revert VAE to 32-bit floats' setting.\n"
                    "To always start with 32-bit VAE, use --no-half-vae commandline flag."
                )

                devices.dtype_vae = torch.float32
                model.first_stage_model.to(devices.dtype_vae)
                batch = batch.to(devices.dtype_vae)

                sample = decode_first_stage(model, batch[i:i + 1])[0]

        if target_device is not None:
            sample = sample.to(target_device)

        samples.append(sample)

    return samples


def get_fixed_seed(seed):
    if seed == '' or seed is None:
        seed = -1
    elif isinstance(seed, str):
        try:
            seed = int(seed)
        except Exception:
            seed = -1

    if seed == -1:
        return int(random.randrange(4294967294))

    return seed


def fix_seed(p):
    p.seed = get_fixed_seed(p.seed)
    p.subseed = get_fixed_seed(p.subseed)


def program_version():
    import launch

    res = launch.git_tag()
    if res == "<none>":
        res = None

    return res


def create_infotext(p, all_prompts, all_seeds, all_subseeds, comments=None, iteration=0, position_in_batch=0,
                    use_main_prompt=False, index=None, all_negative_prompts=None):
    if index is None:
        index = position_in_batch + iteration * p.batch_size

    if all_negative_prompts is None:
        all_negative_prompts = p.all_negative_prompts

    clip_skip = getattr(p, 'clip_skip', opts.CLIP_stop_at_last_layers)
    enable_hr = getattr(p, 'enable_hr', False)
    token_merging_ratio = p.get_token_merging_ratio()
    token_merging_ratio_hr = p.get_token_merging_ratio(for_hr=True)

    uses_ensd = opts.eta_noise_seed_delta != 0
    if uses_ensd:
        uses_ensd = sd_samplers_common.is_sampler_using_eta_noise_seed_delta(p)

    generation_params = {
        "Steps": p.steps,
        "Sampler": p.sampler_name,
        "CFG scale": p.cfg_scale,
        "Image CFG scale": getattr(p, 'image_cfg_scale', None),
        "Seed": p.all_seeds[0] if use_main_prompt else all_seeds[index],
        "Face restoration": opts.face_restoration_model if p.restore_faces else None,
        "Size": f"{p.width}x{p.height}",
        "Model hash": p.sd_model_hash if opts.add_model_hash_to_info else None,
        "Model": p.sd_model_name if opts.add_model_name_to_info else None,
        "VAE hash": p.sd_vae_hash if opts.add_model_hash_to_info else None,
        "VAE": p.sd_vae_name if opts.add_model_name_to_info else None,
        "Variation seed": (
            None if p.subseed_strength == 0 else (p.all_subseeds[0] if use_main_prompt else all_subseeds[index])),
        "Variation seed strength": (None if p.subseed_strength == 0 else p.subseed_strength),
        "Seed resize from": (
            None if p.seed_resize_from_w <= 0 or p.seed_resize_from_h <= 0 else f"{p.seed_resize_from_w}x{p.seed_resize_from_h}"),
        "Denoising strength": getattr(p, 'denoising_strength', None),
        "Conditional mask weight": getattr(p, "inpainting_mask_weight",
                                           shared.opts.inpainting_mask_weight) if p.is_using_inpainting_conditioning else None,
        "Clip skip": None if clip_skip <= 1 else clip_skip,
        "ENSD": opts.eta_noise_seed_delta if uses_ensd else None,
        "Token merging ratio": None if token_merging_ratio == 0 else token_merging_ratio,
        "Token merging ratio hr": None if not enable_hr or token_merging_ratio_hr == 0 else token_merging_ratio_hr,
        "Init image hash": getattr(p, 'init_img_hash', None),
        "RNG": opts.randn_source if opts.randn_source != "GPU" else None,
        "NGMS": None if p.s_min_uncond == 0 else p.s_min_uncond,
        "Tiling": "True" if p.tiling else None,
        **p.extra_generation_params,
        "Version": program_version() if opts.add_version_to_infotext else None,
        "User": p.user if opts.add_user_name_to_info else None,
    }

    generation_params_text = ", ".join(
        [k if k == v else f'{k}: {generation_parameters_copypaste.quote(v)}' for k, v in generation_params.items() if
         v is not None])

    prompt_text = p.main_prompt if use_main_prompt else all_prompts[index]
    negative_prompt_text = f"\nNegative prompt: {p.main_negative_prompt if use_main_prompt else all_negative_prompts[index]}" if \
        all_negative_prompts[index] else ""

    return f"{prompt_text}{negative_prompt_text}\n{generation_params_text}".strip()


# 真正的文生图方法
def process_images(p: StableDiffusionProcessing) -> Processed:
    # 如果 p.scripts 不为 None，那么就调用 p.scripts 的 before_process 方法。这可能是在处理图像之前进行一些准备工作。
    if p.scripts is not None:
        p.scripts.before_process(p)
    # 创建一个字典 stored_opts，它保存了 opts.data 中与 p.override_settings 的键相对应的键值对。这可能是在修改 opts.data 之前保存原始值。
    stored_opts = {k: opts.data[k] for k in p.override_settings.keys()}

    try:
        # if no checkpoint override or the override checkpoint can't be found, remove override entry and load opts
        # checkpoint and if after running refiner, the refiner model is not unloaded - webui swaps back to main model
        # here, if model over is present it will be reloaded afterwards 这行代码检查 sd_models.checkpoint_aliases
        # 中是否存在p.override_settings.get('sd_model_checkpoint') 指定的键。 如果不存在，那么就从 p.override_settings 中移除
        # 'sd_model_checkpoint' 键，并重新加载模型权重。
        if sd_models.checkpoint_aliases.get(p.override_settings.get('sd_model_checkpoint')) is None:
            # 那么就从 p.override_settings 中移除 'sd_model_checkpoint' 键
            p.override_settings.pop('sd_model_checkpoint', None)
            # 重新加载模型权重
            sd_models.reload_model_weights()
        # 这个循环遍历 p.override_settings 中的每一个键值对。
        # 对于每一个键值对，它会设置 opts 中相应的值，并可能重新加载模型权重。  理解为需要重新加载的选项
        for k, v in p.override_settings.items():
            opts.set(k, v, is_api=True, run_callbacks=False)

            if k == 'sd_model_checkpoint':
                sd_models.reload_model_weights()

            if k == 'sd_vae':
                sd_vae.reload_vae_weights()
        # 调用 sd_models.apply_token_merging 方法，可能是在处理图像之前调整模型的一些参数。
        sd_models.apply_token_merging(p.sd_model, p.get_token_merging_ratio())
        # 并将结果保存在 res 中。这可能是处理或生成图像的主要步骤。
        res = process_images_inner(p)
    # 之后的代码块在退出函数之前执行一些清理工作，包括恢复 opts 到原始状态，以及可能的重新加载模型权重。
    finally:
        sd_models.apply_token_merging(p.sd_model, 0)

        # restore opts to original state
        if p.override_settings_restore_afterwards:
            for k, v in stored_opts.items():
                setattr(opts, k, v)

                if k == 'sd_vae':
                    sd_vae.reload_vae_weights()

    return res


# 图片生成的核心方法
def process_images_inner(p: StableDiffusionProcessing) -> Processed:
    """this is the main loop that both txt2img and img2img use; it calls func_init once inside all the scopes and
    func_sample once per batch"""
    # 这行代码检查 p.prompt 是否为列表类型，如果是，进一步确认其长度大于0。
    # 如果 p.prompt 是一个空列表，那么 assert 语句会抛出 AssertionError 异常。
    if isinstance(p.prompt, list):
        assert (len(p.prompt) > 0)
    else:
        assert p.prompt is not None
    # 调用 devices.torch_gc 函数。这可能是一个用于清理 PyTorch 产生的 GPU 内存的函数。
    # 在进行一些内存密集型的操作（如训练深度学习模型）之前，清理 GPU 内存可以防止因内存不足而导致的错误。
    devices.torch_gc()
    # 这两行代码调用 get_fixed_seed 函数，分别使用 p.seed 和 p.subseed 作为参数，
    # 获取固定的种子值。种子值通常用于初始化随机数生成器，以确保实验的可重复性。
    seed = get_fixed_seed(p.seed)
    # subseed 子种子 增加额外随机性
    subseed = get_fixed_seed(p.subseed)

    # 是否进行面部修复
    if p.restore_faces is None:
        p.restore_faces = opts.face_restoration
    # 干啥的暂时不知道
    if p.tiling is None:
        p.tiling = opts.tiling
    # xl的refiner 暂时先跳过
    if p.refiner_checkpoint not in (None, "", "None", "none"):
        p.refiner_checkpoint_info = sd_models.get_closet_checkpoint_match(p.refiner_checkpoint)
        if p.refiner_checkpoint_info is None:
            raise Exception(f'Could not find checkpoint with name {p.refiner_checkpoint}')
    # sd_model 是一个WebuiSdModel类   是LatentDiffusion的 详细属性跳转到sd_models_types.py
    # 模型的名字
    p.sd_model_name = shared.sd_model.sd_checkpoint_info.name_for_extra
    # 模型的hash
    p.sd_model_hash = shared.sd_model.sd_model_hash
    # vae的名字
    p.sd_vae_name = sd_vae.get_loaded_vae_name()
    # vae的hash
    p.sd_vae_hash = sd_vae.get_loaded_vae_hash()
    # 感觉是判断当前模型是否在进行循环卷积计算 如果不是则把判断设置成参数 如果是则不设置
    modules.sd_hijack.model_hijack.apply_circular(p.tiling)
    # 清除或重置一个模型对象中的注释(comments)和额外生成参数(extra_generation_params)
    modules.sd_hijack.model_hijack.clear_comments()

    # 这个函数是用来设置用户输入的提示(prompt)和负面提示(negative prompt)的。
    p.setup_prompts()
    # 检查seed是否是一个列表,如果是直接赋值给all_seeds
    if isinstance(seed, list):
        p.all_seeds = seed
    else:
        # 如果seed不是列表,则根据all_prompts列表长度循环生成种子
        p.all_seeds = [int(seed) + (x if p.subseed_strength == 0 else 0) for x in range(len(p.all_prompts))]
    # 检查subseed是否是一个列表,如果是直接赋值给all_seeds
    if isinstance(subseed, list):
        p.all_subseeds = subseed
    else:
        # 如果subseed不是列表,则根据all_prompts列表长度循环生成种子
        p.all_subseeds = [int(subseed) + x for x in range(len(p.all_prompts))]
    # 检查命令行参数embeddings_dir指定的目录是否存在
    # 检查模型配置do_not_reload_embeddings是否为False
    # 如果命中则调用模型的embedding_db对象的load_textual_inversion_embeddings方法  
    if os.path.exists(cmd_opts.embeddings_dir) and not p.do_not_reload_embeddings:
        # 实现了文本反转词向量的重新加载功能，也就是把embeddings中的embedding文件转出向量
        model_hijack.embedding_db.load_textual_inversion_embeddings()
    # 如果使用了脚本 先处理脚本的参数
    if p.scripts is not None:
        p.scripts.process(p)
    # 订阅新的空字典 infotexts和output_images
    infotexts = []
    output_images = []
    # 在不记录历史的情况下进行ema参数更新,节省内存开销，
    # 这段代码通过context管理Real-Time EMA算法的参数更新过程,同时优化内存开销,因为不记录中间节点的运行历史。 
    with torch.no_grad(), p.sd_model.ema_scope():
        # devices.autocast()上下文管理器。
        # autocast可以自动将计算从CPU升级至GPU,提供混合精度计算的能力。
        # devices.autocast()会进入一个上下文中,在这个范围内:计算会自动判断是否可以使用GPU加速 
        with devices.autocast():
            # 是个空方法
            p.init(p.all_prompts, p.all_seeds, p.all_subseeds)
            # 兼容macos的代码  暂时不考虑
            # for OSX, loading the model during sampling changes the generated picture, so it is loaded here
            if shared.opts.live_previews_enable and opts.show_progress_type == "Approx NN":
                sd_vae_approx.model()
            # 判断是否需要重新使用unet 但是这个unet好像对应的是VAE
            sd_unet.apply_unet()
        # 判断当前无任务 则将任务数量设置为 批次的数量
        if state.job_count == -1:
            state.job_count = p.n_iter
        # 循环
        for n in range(p.n_iter):
            # 更新当前批次号
            p.iteration = n
            # 判断当前是否要 跳过 如果是 则设置state.skipped为false
            if state.skipped:
                state.skipped = False
            # 如果已经设置为终止 则直接break 结束跳出循环
            if state.interrupted:
                break
            # 载入模型数据  方法中判断了是否加载
            sd_models.reload_model_weights()  # model can be changed for example by refiner

            p.prompts = p.all_prompts[n * p.batch_size:(n + 1) * p.batch_size]
            p.negative_prompts = p.all_negative_prompts[n * p.batch_size:(n + 1) * p.batch_size]
            p.seeds = p.all_seeds[n * p.batch_size:(n + 1) * p.batch_size]
            p.subseeds = p.all_subseeds[n * p.batch_size:(n + 1) * p.batch_size]

            p.rng = rng.ImageRNG((opt_C, p.height // opt_f, p.width // opt_f), p.seeds, subseeds=p.subseeds,
                                 subseed_strength=p.subseed_strength, seed_resize_from_h=p.seed_resize_from_h,
                                 seed_resize_from_w=p.seed_resize_from_w)

            if p.scripts is not None:
                p.scripts.before_process_batch(p, batch_number=n, prompts=p.prompts, seeds=p.seeds, subseeds=p.subseeds)

            if len(p.prompts) == 0:
                break

            p.parse_extra_network_prompts()

            if not p.disable_extra_networks:
                with devices.autocast():
                    extra_networks.activate(p, p.extra_network_data)

            if p.scripts is not None:
                p.scripts.process_batch(p, batch_number=n, prompts=p.prompts, seeds=p.seeds, subseeds=p.subseeds)

            # params.txt should be saved after scripts.process_batch, since the
            # infotext could be modified by that callback
            # Example: a wildcard processed by process_batch sets an extra model
            # strength, which is saved as "Model Strength: 1.0" in the infotext
            if n == 0:
                with open(os.path.join(paths.data_path, "params.txt"), "w", encoding="utf8") as file:
                    processed = Processed(p, [])
                    file.write(processed.infotext(p, 0))

            p.setup_conds()

            for comment in model_hijack.comments:
                p.comment(comment)

            p.extra_generation_params.update(model_hijack.extra_generation_params)

            if p.n_iter > 1:
                shared.state.job = f"Batch {n + 1} out of {p.n_iter}"

            with devices.without_autocast() if devices.unet_needs_upcast else devices.autocast():
                samples_ddim = p.sample(conditioning=p.c, unconditional_conditioning=p.uc, seeds=p.seeds,
                                        subseeds=p.subseeds, subseed_strength=p.subseed_strength, prompts=p.prompts)

            if getattr(samples_ddim, 'already_decoded', False):
                x_samples_ddim = samples_ddim
            else:
                if opts.sd_vae_decode_method != 'Full':
                    p.extra_generation_params['VAE Decoder'] = opts.sd_vae_decode_method

                x_samples_ddim = decode_latent_batch(p.sd_model, samples_ddim, target_device=devices.cpu,
                                                     check_for_nans=True)

            x_samples_ddim = torch.stack(x_samples_ddim).float()
            x_samples_ddim = torch.clamp((x_samples_ddim + 1.0) / 2.0, min=0.0, max=1.0)

            del samples_ddim

            if lowvram.is_enabled(shared.sd_model):
                lowvram.send_everything_to_cpu()

            devices.torch_gc()

            if p.scripts is not None:
                p.scripts.postprocess_batch(p, x_samples_ddim, batch_number=n)

                p.prompts = p.all_prompts[n * p.batch_size:(n + 1) * p.batch_size]
                p.negative_prompts = p.all_negative_prompts[n * p.batch_size:(n + 1) * p.batch_size]

                batch_params = scripts.PostprocessBatchListArgs(list(x_samples_ddim))
                p.scripts.postprocess_batch_list(p, batch_params, batch_number=n)
                x_samples_ddim = batch_params.images

            def infotext(index=0, use_main_prompt=False):
                return create_infotext(p, p.prompts, p.seeds, p.subseeds, use_main_prompt=use_main_prompt, index=index,
                                       all_negative_prompts=p.negative_prompts)

            save_samples = p.save_samples()

            for i, x_sample in enumerate(x_samples_ddim):
                p.batch_index = i

                x_sample = 255. * np.moveaxis(x_sample.cpu().numpy(), 0, 2)
                x_sample = x_sample.astype(np.uint8)

                if p.restore_faces:
                    if save_samples and opts.save_images_before_face_restoration:
                        images.save_image(Image.fromarray(x_sample), p.outpath_samples, "", p.seeds[i], p.prompts[i],
                                          opts.samples_format, info=infotext(i), p=p, suffix="-before-face-restoration")

                    devices.torch_gc()

                    x_sample = modules.face_restoration.restore_faces(x_sample)
                    devices.torch_gc()

                image = Image.fromarray(x_sample)

                if p.scripts is not None:
                    pp = scripts.PostprocessImageArgs(image)
                    p.scripts.postprocess_image(p, pp)
                    image = pp.image
                if p.color_corrections is not None and i < len(p.color_corrections):
                    if save_samples and opts.save_images_before_color_correction:
                        image_without_cc = apply_overlay(image, p.paste_to, i, p.overlay_images)
                        images.save_image(image_without_cc, p.outpath_samples, "", p.seeds[i], p.prompts[i],
                                          opts.samples_format, info=infotext(i), p=p, suffix="-before-color-correction")
                    image = apply_color_correction(p.color_corrections[i], image)

                image = apply_overlay(image, p.paste_to, i, p.overlay_images)

                if save_samples:
                    images.save_image(image, p.outpath_samples, "", p.seeds[i], p.prompts[i], opts.samples_format,
                                      info=infotext(i), p=p)

                text = infotext(i)
                infotexts.append(text)
                if opts.enable_pnginfo:
                    image.info["parameters"] = text
                output_images.append(image)
                if save_samples and hasattr(p, 'mask_for_overlay') and p.mask_for_overlay and any(
                        [opts.save_mask, opts.save_mask_composite, opts.return_mask, opts.return_mask_composite]):
                    image_mask = p.mask_for_overlay.convert('RGB')
                    image_mask_composite = Image.composite(image.convert('RGBA').convert('RGBa'),
                                                           Image.new('RGBa', image.size),
                                                           images.resize_image(2, p.mask_for_overlay, image.width,
                                                                               image.height).convert('L')).convert(
                        'RGBA')

                    if opts.save_mask:
                        images.save_image(image_mask, p.outpath_samples, "", p.seeds[i], p.prompts[i],
                                          opts.samples_format, info=infotext(i), p=p, suffix="-mask")

                    if opts.save_mask_composite:
                        images.save_image(image_mask_composite, p.outpath_samples, "", p.seeds[i], p.prompts[i],
                                          opts.samples_format, info=infotext(i), p=p, suffix="-mask-composite")

                    if opts.return_mask:
                        output_images.append(image_mask)

                    if opts.return_mask_composite:
                        output_images.append(image_mask_composite)

            del x_samples_ddim

            devices.torch_gc()

            state.nextjob()

        p.color_corrections = None

        index_of_first_image = 0
        unwanted_grid_because_of_img_count = len(output_images) < 2 and opts.grid_only_if_multiple
        if (opts.return_grid or opts.grid_save) and not p.do_not_save_grid and not unwanted_grid_because_of_img_count:
            grid = images.image_grid(output_images, p.batch_size)

            if opts.return_grid:
                text = infotext(use_main_prompt=True)
                infotexts.insert(0, text)
                if opts.enable_pnginfo:
                    grid.info["parameters"] = text
                output_images.insert(0, grid)
                index_of_first_image = 1
            if opts.grid_save:
                images.save_image(grid, p.outpath_grids, "grid", p.all_seeds[0], p.all_prompts[0], opts.grid_format,
                                  info=infotext(use_main_prompt=True), short_filename=not opts.grid_extended_filename,
                                  p=p, grid=True)

    if not p.disable_extra_networks and p.extra_network_data:
        extra_networks.deactivate(p, p.extra_network_data)

    devices.torch_gc()

    res = Processed(
        p,
        images_list=output_images,
        seed=p.all_seeds[0],
        info=infotexts[0],
        subseed=p.all_subseeds[0],
        index_of_first_image=index_of_first_image,
        infotexts=infotexts,
    )

    if p.scripts is not None:
        p.scripts.postprocess(p, res)

    return res


# 旧的高清修复 用户可以通过滑块设置出图的宽高，很大几率不是64的倍数，此方法会对宽高进行处理
def old_hires_fix_first_pass_dimensions(width, height):
    """
        old algorithm for auto-calculating first pass size
        自动计算第一遍尺寸的旧算法

        这个函数的目标是确保调整后的图像尺寸的像素数量接近于所需的最小像素数量（512 x 512）。
        它同时考虑到保持图像的宽高比，并且尺寸是64的倍数，为了方便后续的算法或硬件处理。
    """

    # 所需像素数
    desired_pixel_count = 512 * 512
    # 实际像素数
    actual_pixel_count = width * height
    # 用于计算一个数的平方根。
    scale = math.sqrt(desired_pixel_count / actual_pixel_count)
    # math.ceil 用于对一个浮点数进行上取整。例如，math.ceil(3.14) 的结果为 4。
    width = math.ceil(scale * width / 64) * 64
    height = math.ceil(scale * height / 64) * 64

    return width, height


# 相比普通class，dataclass通常不包含私有属性，数据可以直接访问
# @dataclass是Python 3.7引入的一个内置装饰器，用于自动添加特殊方法，如__init__和__repr__等，到类中，使得我们可以更方便地创建数据类。
# repr=False表示在创建类的实例时，不自动生成__repr__方法。
# StableDiffusionProcessingTxt2Img 继承 StableDiffusionProcessing
@dataclass(repr=False)
class StableDiffusionProcessingTxt2Img(StableDiffusionProcessing):
    # 是否开启高分辨率修复开关
    enable_hr: bool = False
    # 重绘幅度
    denoising_strength: float = 0.75
    # firstphase_width可能用于指示或控制第一阶段处理后的图像宽度。
    firstphase_width: int = 0
    # firstphase_width可能用于指示或控制第一阶段处理后的图像高度。
    firstphase_height: int = 0
    # 放大倍率
    hr_scale: float = 2.0
    # 放大算法：Latent | R-ESRGAN 4+ Anime6B 等算法
    hr_upscaler: str = None
    # 可能用于指示或控制第二阶段生成步骤的数量
    hr_second_pass_steps: int = 0
    # 高分辨率修复控制出图尺寸有两种方式：
    # 1. 设置放大倍率
    # 2. 直接调整狂傲
    # 宽度调整
    hr_resize_x: int = 0
    # 高度调整
    hr_resize_y: int = 0
    hr_checkpoint_name: str = None
    # 采样器名字
    hr_sampler_name: str = None
    # 提示词
    hr_prompt: str = ''
    # 负面提示词
    hr_negative_prompt: str = ''

    cached_hr_uc = [None, None]
    cached_hr_c = [None, None]
    # field() 函数在 Python 3.7 中引入，它是 dataclasses 模块的一部分。此函数用于获取或设置一个数据类实例的字段。
    # default 是字段的默认值。
    # 如果 init 为 true，则该字段将成为类的 __init__（） 函数的参数。
    hr_checkpoint_info: dict = field(default=None, init=False)
    hr_upscale_to_x: int = field(default=0, init=False)
    hr_upscale_to_y: int = field(default=0, init=False)
    truncate_x: int = field(default=0, init=False)
    truncate_y: int = field(default=0, init=False)
    applied_old_hires_behavior_to: tuple = field(default=None, init=False)
    latent_scale_mode: dict = field(default=None, init=False)
    hr_c: tuple | None = field(default=None, init=False)
    hr_uc: tuple | None = field(default=None, init=False)
    all_hr_prompts: list = field(default=None, init=False)
    all_hr_negative_prompts: list = field(default=None, init=False)
    hr_prompts: list = field(default=None, init=False)
    hr_negative_prompts: list = field(default=None, init=False)
    hr_extra_network_data: list = field(default=None, init=False)

    # __post_init__是一个特殊的方法，通常用于初始化类实例。
    # 这个方法通常用于执行一些需要在所有基础构造函数完成之后才开始的初始化工作。
    def __post_init__(self):
        super().__post_init__()

        if self.firstphase_width != 0 or self.firstphase_height != 0:
            self.hr_upscale_to_x = self.width
            self.hr_upscale_to_y = self.height
            self.width = self.firstphase_width
            self.height = self.firstphase_height

        self.cached_hr_uc = StableDiffusionProcessingTxt2Img.cached_hr_uc
        self.cached_hr_c = StableDiffusionProcessingTxt2Img.cached_hr_c

    def calculate_target_resolution(self):
        # 兼容性设置中勾选了 <在高分辨率修复中，使用长宽滑块设置最终分辨率> 选项
        if opts.use_old_hires_fix_width_height and self.applied_old_hires_behavior_to != (self.width, self.height):
            self.hr_resize_x = self.width
            self.hr_resize_y = self.height
            self.hr_upscale_to_x = self.width
            self.hr_upscale_to_y = self.height

            self.width, self.height = old_hires_fix_first_pass_dimensions(self.width, self.height)
            self.applied_old_hires_behavior_to = (self.width, self.height)

        if self.hr_resize_x == 0 and self.hr_resize_y == 0:
            # 高清修复选项里，用户使用放大倍率，没有手动设置宽高
            # 将高清修复放大倍率hr_scale 放到 extra_generation_params中
            self.extra_generation_params["Hires upscale"] = self.hr_scale
            # 宽度 x 放大倍率 得到要生成的图片的宽度
            self.hr_upscale_to_x = int(self.width * self.hr_scale)
            self.hr_upscale_to_y = int(self.height * self.hr_scale)
        else:
            # 用户手动设置高清修复的宽高
            self.extra_generation_params["Hires resize"] = f"{self.hr_resize_x}x{self.hr_resize_y}"

            if self.hr_resize_y == 0:
                self.hr_upscale_to_x = self.hr_resize_x
                # 如果用户未设置高度，则根据原始宽高比去计算高清修复后的高度
                self.hr_upscale_to_y = self.hr_resize_x * self.height // self.width
            elif self.hr_resize_x == 0:
                # 未设置宽度的case
                self.hr_upscale_to_x = self.hr_resize_y * self.width // self.height
                self.hr_upscale_to_y = self.hr_resize_y
            else:
                target_w = self.hr_resize_x
                target_h = self.hr_resize_y
                # 原始宽高比
                src_ratio = self.width / self.height
                # 目标宽高比
                dst_ratio = self.hr_resize_x / self.hr_resize_y

                # 使用户调整后的宽高遵循原始宽高比
                if src_ratio < dst_ratio:
                    self.hr_upscale_to_x = self.hr_resize_x
                    self.hr_upscale_to_y = self.hr_resize_x * self.height // self.width
                else:
                    self.hr_upscale_to_x = self.hr_resize_y * self.width // self.height
                    self.hr_upscale_to_y = self.hr_resize_y

                # 计算由于目标宽高比和原始宽高比不同导致的宽高截断的数值
                self.truncate_x = (self.hr_upscale_to_x - target_w) // opt_f
                self.truncate_y = (self.hr_upscale_to_y - target_h) // opt_f

    def init(self, all_prompts, all_seeds, all_subseeds):
        if self.enable_hr:
            if self.hr_checkpoint_name:
                self.hr_checkpoint_info = sd_models.get_closet_checkpoint_match(self.hr_checkpoint_name)

                if self.hr_checkpoint_info is None:
                    raise Exception(f'Could not find checkpoint with name {self.hr_checkpoint_name}')

                self.extra_generation_params["Hires checkpoint"] = self.hr_checkpoint_info.short_title

            if self.hr_sampler_name is not None and self.hr_sampler_name != self.sampler_name:
                self.extra_generation_params["Hires sampler"] = self.hr_sampler_name

            if tuple(self.hr_prompt) != tuple(self.prompt):
                self.extra_generation_params["Hires prompt"] = self.hr_prompt

            if tuple(self.hr_negative_prompt) != tuple(self.negative_prompt):
                self.extra_generation_params["Hires negative prompt"] = self.hr_negative_prompt

            self.latent_scale_mode = shared.latent_upscale_modes.get(self.hr_upscaler,
                                                                     None) if self.hr_upscaler is not None else shared.latent_upscale_modes.get(
                shared.latent_upscale_default_mode, "nearest")
            if self.enable_hr and self.latent_scale_mode is None:
                if not any(x.name == self.hr_upscaler for x in shared.sd_upscalers):
                    raise Exception(f"could not find upscaler named {self.hr_upscaler}")

            self.calculate_target_resolution()

            if not state.processing_has_refined_job_count:
                if state.job_count == -1:
                    state.job_count = self.n_iter

                shared.total_tqdm.updateTotal(
                    (self.steps + (self.hr_second_pass_steps or self.steps)) * state.job_count)
                state.job_count = state.job_count * 2
                state.processing_has_refined_job_count = True

            if self.hr_second_pass_steps:
                self.extra_generation_params["Hires steps"] = self.hr_second_pass_steps

            if self.hr_upscaler is not None:
                self.extra_generation_params["Hires upscaler"] = self.hr_upscaler

    def sample(self, conditioning, unconditional_conditioning, seeds, subseeds, subseed_strength, prompts):
        self.sampler = sd_samplers.create_sampler(self.sampler_name, self.sd_model)

        x = self.rng.next()
        samples = self.sampler.sample(self, x, conditioning, unconditional_conditioning,
                                      image_conditioning=self.txt2img_image_conditioning(x))
        del x

        if not self.enable_hr:
            return samples

        if self.latent_scale_mode is None:
            decoded_samples = torch.stack(
                decode_latent_batch(self.sd_model, samples, target_device=devices.cpu, check_for_nans=True)).to(
                dtype=torch.float32)
        else:
            decoded_samples = None

        with sd_models.SkipWritingToConfig():
            sd_models.reload_model_weights(info=self.hr_checkpoint_info)

        devices.torch_gc()

        return self.sample_hr_pass(samples, decoded_samples, seeds, subseeds, subseed_strength, prompts)

    def sample_hr_pass(self, samples, decoded_samples, seeds, subseeds, subseed_strength, prompts):
        if shared.state.interrupted:
            return samples

        self.is_hr_pass = True

        target_width = self.hr_upscale_to_x
        target_height = self.hr_upscale_to_y

        def save_intermediate(image, index):
            """saves image before applying hires fix, if enabled in options; takes as an argument either an image or batch with latent space images"""

            if not self.save_samples() or not opts.save_images_before_highres_fix:
                return

            if not isinstance(image, Image.Image):
                image = sd_samplers.sample_to_image(image, index, approximation=0)

            info = create_infotext(self, self.all_prompts, self.all_seeds, self.all_subseeds, [],
                                   iteration=self.iteration, position_in_batch=index)
            images.save_image(image, self.outpath_samples, "", seeds[index], prompts[index], opts.samples_format,
                              info=info, p=self, suffix="-before-highres-fix")

        img2img_sampler_name = self.hr_sampler_name or self.sampler_name

        self.sampler = sd_samplers.create_sampler(img2img_sampler_name, self.sd_model)

        if self.latent_scale_mode is not None:
            for i in range(samples.shape[0]):
                save_intermediate(samples, i)

            samples = torch.nn.functional.interpolate(samples, size=(target_height // opt_f, target_width // opt_f),
                                                      mode=self.latent_scale_mode["mode"],
                                                      antialias=self.latent_scale_mode["antialias"])

            # Avoid making the inpainting conditioning unless necessary as
            # this does need some extra compute to decode / encode the image again.
            if getattr(self, "inpainting_mask_weight", shared.opts.inpainting_mask_weight) < 1.0:
                image_conditioning = self.img2img_image_conditioning(decode_first_stage(self.sd_model, samples),
                                                                     samples)
            else:
                image_conditioning = self.txt2img_image_conditioning(samples)
        else:
            lowres_samples = torch.clamp((decoded_samples + 1.0) / 2.0, min=0.0, max=1.0)

            batch_images = []
            for i, x_sample in enumerate(lowres_samples):
                x_sample = 255. * np.moveaxis(x_sample.cpu().numpy(), 0, 2)
                x_sample = x_sample.astype(np.uint8)
                image = Image.fromarray(x_sample)

                save_intermediate(image, i)

                image = images.resize_image(0, image, target_width, target_height, upscaler_name=self.hr_upscaler)
                image = np.array(image).astype(np.float32) / 255.0
                image = np.moveaxis(image, 2, 0)
                batch_images.append(image)

            decoded_samples = torch.from_numpy(np.array(batch_images))
            decoded_samples = decoded_samples.to(shared.device, dtype=devices.dtype_vae)

            if opts.sd_vae_encode_method != 'Full':
                self.extra_generation_params['VAE Encoder'] = opts.sd_vae_encode_method
            samples = images_tensor_to_samples(decoded_samples, approximation_indexes.get(opts.sd_vae_encode_method))

            image_conditioning = self.img2img_image_conditioning(decoded_samples, samples)

        shared.state.nextjob()

        samples = samples[:, :, self.truncate_y // 2:samples.shape[2] - (self.truncate_y + 1) // 2,
                  self.truncate_x // 2:samples.shape[3] - (self.truncate_x + 1) // 2]

        self.rng = rng.ImageRNG(samples.shape[1:], self.seeds, subseeds=self.subseeds,
                                subseed_strength=self.subseed_strength, seed_resize_from_h=self.seed_resize_from_h,
                                seed_resize_from_w=self.seed_resize_from_w)
        noise = self.rng.next()

        # GC now before running the next img2img to prevent running out of memory
        devices.torch_gc()

        if not self.disable_extra_networks:
            with devices.autocast():
                extra_networks.activate(self, self.hr_extra_network_data)

        with devices.autocast():
            self.calculate_hr_conds()

        sd_models.apply_token_merging(self.sd_model, self.get_token_merging_ratio(for_hr=True))

        if self.scripts is not None:
            self.scripts.before_hr(self)

        samples = self.sampler.sample_img2img(self, samples, noise, self.hr_c, self.hr_uc,
                                              steps=self.hr_second_pass_steps or self.steps,
                                              image_conditioning=image_conditioning)

        sd_models.apply_token_merging(self.sd_model, self.get_token_merging_ratio())

        self.sampler = None
        devices.torch_gc()

        decoded_samples = decode_latent_batch(self.sd_model, samples, target_device=devices.cpu, check_for_nans=True)

        self.is_hr_pass = False

        return decoded_samples

    def close(self):
        super().close()
        self.hr_c = None
        self.hr_uc = None
        if not opts.persistent_cond_cache:
            StableDiffusionProcessingTxt2Img.cached_hr_uc = [None, None]
            StableDiffusionProcessingTxt2Img.cached_hr_c = [None, None]

    def setup_prompts(self):
        super().setup_prompts()

        if not self.enable_hr:
            return

        if self.hr_prompt == '':
            self.hr_prompt = self.prompt

        if self.hr_negative_prompt == '':
            self.hr_negative_prompt = self.negative_prompt

        if isinstance(self.hr_prompt, list):
            self.all_hr_prompts = self.hr_prompt
        else:
            self.all_hr_prompts = self.batch_size * self.n_iter * [self.hr_prompt]

        if isinstance(self.hr_negative_prompt, list):
            self.all_hr_negative_prompts = self.hr_negative_prompt
        else:
            self.all_hr_negative_prompts = self.batch_size * self.n_iter * [self.hr_negative_prompt]

        self.all_hr_prompts = [shared.prompt_styles.apply_styles_to_prompt(x, self.styles) for x in self.all_hr_prompts]
        self.all_hr_negative_prompts = [shared.prompt_styles.apply_negative_styles_to_prompt(x, self.styles) for x in
                                        self.all_hr_negative_prompts]

    def calculate_hr_conds(self):
        if self.hr_c is not None:
            return

        hr_prompts = prompt_parser.SdConditioning(self.hr_prompts, width=self.hr_upscale_to_x,
                                                  height=self.hr_upscale_to_y)
        hr_negative_prompts = prompt_parser.SdConditioning(self.hr_negative_prompts, width=self.hr_upscale_to_x,
                                                           height=self.hr_upscale_to_y, is_negative_prompt=True)

        sampler_config = sd_samplers.find_sampler_config(self.hr_sampler_name or self.sampler_name)
        steps = self.hr_second_pass_steps or self.steps
        total_steps = sampler_config.total_steps(steps) if sampler_config else steps

        self.hr_uc = self.get_conds_with_caching(prompt_parser.get_learned_conditioning, hr_negative_prompts,
                                                 self.firstpass_steps, [self.cached_hr_uc, self.cached_uc],
                                                 self.hr_extra_network_data, total_steps)
        self.hr_c = self.get_conds_with_caching(prompt_parser.get_multicond_learned_conditioning, hr_prompts,
                                                self.firstpass_steps, [self.cached_hr_c, self.cached_c],
                                                self.hr_extra_network_data, total_steps)

    def setup_conds(self):
        if self.is_hr_pass:
            # if we are in hr pass right now, the call is being made from the refiner, and we don't need to setup firstpass cons or switch model
            self.hr_c = None
            self.calculate_hr_conds()
            return

        super().setup_conds()

        self.hr_uc = None
        self.hr_c = None

        if self.enable_hr and self.hr_checkpoint_info is None:
            if shared.opts.hires_fix_use_firstpass_conds:
                self.calculate_hr_conds()

            elif lowvram.is_enabled(
                    shared.sd_model) and shared.sd_model.sd_checkpoint_info == sd_models.select_checkpoint():  # if in lowvram mode, we need to calculate conds right away, before the cond NN is unloaded
                with devices.autocast():
                    extra_networks.activate(self, self.hr_extra_network_data)

                self.calculate_hr_conds()

                with devices.autocast():
                    extra_networks.activate(self, self.extra_network_data)

    def get_conds(self):
        if self.is_hr_pass:
            return self.hr_c, self.hr_uc

        return super().get_conds()

    def parse_extra_network_prompts(self):
        res = super().parse_extra_network_prompts()

        if self.enable_hr:
            self.hr_prompts = self.all_hr_prompts[
                              self.iteration * self.batch_size:(self.iteration + 1) * self.batch_size]
            self.hr_negative_prompts = self.all_hr_negative_prompts[
                                       self.iteration * self.batch_size:(self.iteration + 1) * self.batch_size]

            self.hr_prompts, self.hr_extra_network_data = extra_networks.parse_prompts(self.hr_prompts)

        return res


@dataclass(repr=False)
class StableDiffusionProcessingImg2Img(StableDiffusionProcessing):
    init_images: list = None
    resize_mode: int = 0
    denoising_strength: float = 0.75
    image_cfg_scale: float = None
    mask: Any = None
    mask_blur_x: int = 4
    mask_blur_y: int = 4
    mask_blur: int = None
    inpainting_fill: int = 0
    inpaint_full_res: bool = True
    inpaint_full_res_padding: int = 0
    inpainting_mask_invert: int = 0
    initial_noise_multiplier: float = None
    latent_mask: Image = None

    image_mask: Any = field(default=None, init=False)

    nmask: torch.Tensor = field(default=None, init=False)
    image_conditioning: torch.Tensor = field(default=None, init=False)
    init_img_hash: str = field(default=None, init=False)
    mask_for_overlay: Image = field(default=None, init=False)
    init_latent: torch.Tensor = field(default=None, init=False)

    def __post_init__(self):
        super().__post_init__()

        self.image_mask = self.mask
        self.mask = None
        self.initial_noise_multiplier = opts.initial_noise_multiplier if self.initial_noise_multiplier is None else self.initial_noise_multiplier

    @property
    def mask_blur(self):
        if self.mask_blur_x == self.mask_blur_y:
            return self.mask_blur_x
        return None

    @mask_blur.setter
    def mask_blur(self, value):
        if isinstance(value, int):
            self.mask_blur_x = value
            self.mask_blur_y = value

    def init(self, all_prompts, all_seeds, all_subseeds):
        self.image_cfg_scale: float = self.image_cfg_scale if shared.sd_model.cond_stage_key == "edit" else None

        self.sampler = sd_samplers.create_sampler(self.sampler_name, self.sd_model)
        crop_region = None

        image_mask = self.image_mask

        if image_mask is not None:
            # image_mask is passed in as RGBA by Gradio to support alpha masks,
            # but we still want to support binary masks.
            image_mask = create_binary_mask(image_mask)

            if self.inpainting_mask_invert:
                image_mask = ImageOps.invert(image_mask)

            if self.mask_blur_x > 0:
                np_mask = np.array(image_mask)
                kernel_size = 2 * int(2.5 * self.mask_blur_x + 0.5) + 1
                np_mask = cv2.GaussianBlur(np_mask, (kernel_size, 1), self.mask_blur_x)
                image_mask = Image.fromarray(np_mask)

            if self.mask_blur_y > 0:
                np_mask = np.array(image_mask)
                kernel_size = 2 * int(2.5 * self.mask_blur_y + 0.5) + 1
                np_mask = cv2.GaussianBlur(np_mask, (1, kernel_size), self.mask_blur_y)
                image_mask = Image.fromarray(np_mask)

            if self.inpaint_full_res:
                self.mask_for_overlay = image_mask
                mask = image_mask.convert('L')
                crop_region = masking.get_crop_region(np.array(mask), self.inpaint_full_res_padding)
                crop_region = masking.expand_crop_region(crop_region, self.width, self.height, mask.width, mask.height)
                x1, y1, x2, y2 = crop_region

                mask = mask.crop(crop_region)
                image_mask = images.resize_image(2, mask, self.width, self.height)
                self.paste_to = (x1, y1, x2 - x1, y2 - y1)
            else:
                image_mask = images.resize_image(self.resize_mode, image_mask, self.width, self.height)
                np_mask = np.array(image_mask)
                np_mask = np.clip((np_mask.astype(np.float32)) * 2, 0, 255).astype(np.uint8)
                self.mask_for_overlay = Image.fromarray(np_mask)

            self.overlay_images = []

        latent_mask = self.latent_mask if self.latent_mask is not None else image_mask

        add_color_corrections = opts.img2img_color_correction and self.color_corrections is None
        if add_color_corrections:
            self.color_corrections = []
        imgs = []
        for img in self.init_images:

            # Save init image
            if opts.save_init_img:
                self.init_img_hash = hashlib.md5(img.tobytes()).hexdigest()
                images.save_image(img, path=opts.outdir_init_images, basename=None, forced_filename=self.init_img_hash,
                                  save_to_dirs=False)

            image = images.flatten(img, opts.img2img_background_color)

            if crop_region is None and self.resize_mode != 3:
                image = images.resize_image(self.resize_mode, image, self.width, self.height)

            if image_mask is not None:
                image_masked = Image.new('RGBa', (image.width, image.height))
                image_masked.paste(image.convert("RGBA").convert("RGBa"),
                                   mask=ImageOps.invert(self.mask_for_overlay.convert('L')))

                self.overlay_images.append(image_masked.convert('RGBA'))

            # crop_region is not None if we are doing inpaint full res
            if crop_region is not None:
                image = image.crop(crop_region)
                image = images.resize_image(2, image, self.width, self.height)

            if image_mask is not None:
                if self.inpainting_fill != 1:
                    image = masking.fill(image, latent_mask)

            if add_color_corrections:
                self.color_corrections.append(setup_color_correction(image))

            image = np.array(image).astype(np.float32) / 255.0
            image = np.moveaxis(image, 2, 0)

            imgs.append(image)

        if len(imgs) == 1:
            batch_images = np.expand_dims(imgs[0], axis=0).repeat(self.batch_size, axis=0)
            if self.overlay_images is not None:
                self.overlay_images = self.overlay_images * self.batch_size

            if self.color_corrections is not None and len(self.color_corrections) == 1:
                self.color_corrections = self.color_corrections * self.batch_size

        elif len(imgs) <= self.batch_size:
            self.batch_size = len(imgs)
            batch_images = np.array(imgs)
        else:
            raise RuntimeError(f"bad number of images passed: {len(imgs)}; expecting {self.batch_size} or less")

        image = torch.from_numpy(batch_images)
        image = image.to(shared.device, dtype=devices.dtype_vae)

        if opts.sd_vae_encode_method != 'Full':
            self.extra_generation_params['VAE Encoder'] = opts.sd_vae_encode_method

        self.init_latent = images_tensor_to_samples(image, approximation_indexes.get(opts.sd_vae_encode_method),
                                                    self.sd_model)
        devices.torch_gc()

        if self.resize_mode == 3:
            self.init_latent = torch.nn.functional.interpolate(self.init_latent,
                                                               size=(self.height // opt_f, self.width // opt_f),
                                                               mode="bilinear")

        if image_mask is not None:
            init_mask = latent_mask
            latmask = init_mask.convert('RGB').resize((self.init_latent.shape[3], self.init_latent.shape[2]))
            latmask = np.moveaxis(np.array(latmask, dtype=np.float32), 2, 0) / 255
            latmask = latmask[0]
            latmask = np.around(latmask)
            latmask = np.tile(latmask[None], (4, 1, 1))

            self.mask = torch.asarray(1.0 - latmask).to(shared.device).type(self.sd_model.dtype)
            self.nmask = torch.asarray(latmask).to(shared.device).type(self.sd_model.dtype)

            # this needs to be fixed to be done in sample() using actual seeds for batches
            if self.inpainting_fill == 2:
                self.init_latent = self.init_latent * self.mask + create_random_tensors(self.init_latent.shape[1:],
                                                                                        all_seeds[
                                                                                        0:self.init_latent.shape[
                                                                                            0]]) * self.nmask
            elif self.inpainting_fill == 3:
                self.init_latent = self.init_latent * self.mask

        self.image_conditioning = self.img2img_image_conditioning(image * 2 - 1, self.init_latent, image_mask)

    def sample(self, conditioning, unconditional_conditioning, seeds, subseeds, subseed_strength, prompts):
        x = self.rng.next()

        if self.initial_noise_multiplier != 1.0:
            self.extra_generation_params["Noise multiplier"] = self.initial_noise_multiplier
            x *= self.initial_noise_multiplier

        samples = self.sampler.sample_img2img(self, self.init_latent, x, conditioning, unconditional_conditioning,
                                              image_conditioning=self.image_conditioning)

        if self.mask is not None:
            samples = samples * self.nmask + self.init_latent * self.mask

        del x
        devices.torch_gc()

        return samples

    def get_token_merging_ratio(self, for_hr=False):
        return self.token_merging_ratio or (
                "token_merging_ratio" in self.override_settings and opts.token_merging_ratio) or opts.token_merging_ratio_img2img or opts.token_merging_ratio
