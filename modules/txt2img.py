from contextlib import closing

import modules.scripts
from modules import processing
from modules.generation_parameters_copypaste import create_override_settings_dict
from modules.shared import opts, cmd_opts
import modules.shared as shared
from modules.ui import plaintext_to_html
import gradio as gr

# *args：表示可以接收任意数量的额外参数。
def txt2img(id_task: str, prompt: str, negative_prompt: str, prompt_styles, steps: int, sampler_name: str, n_iter: int, batch_size: int, cfg_scale: float, height: int, width: int, enable_hr: bool, denoising_strength: float, hr_scale: float, hr_upscaler: str, hr_second_pass_steps: int, hr_resize_x: int, hr_resize_y: int, hr_checkpoint_name: str, hr_sampler_name: str, hr_prompt: str, hr_negative_prompt, override_settings_texts, request: gr.Request, *args):
    # 创建了一个字典 override_settings
    override_settings = create_override_settings_dict(override_settings_texts)
    # StableDiffusionProcessingTxt2Img 初始化了一个对象 为p
    p = processing.StableDiffusionProcessingTxt2Img(
        sd_model=shared.sd_model,
        outpath_samples=opts.outdir_samples or opts.outdir_txt2img_samples,
        outpath_grids=opts.outdir_grids or opts.outdir_txt2img_grids,
        prompt=prompt,
        styles=prompt_styles,
        negative_prompt=negative_prompt,
        sampler_name=sampler_name,
        batch_size=batch_size,
        n_iter=n_iter,
        steps=steps,
        cfg_scale=cfg_scale,
        width=width,
        height=height,
        enable_hr=enable_hr,
        denoising_strength=denoising_strength if enable_hr else None,
        hr_scale=hr_scale,
        hr_upscaler=hr_upscaler,
        hr_second_pass_steps=hr_second_pass_steps,
        hr_resize_x=hr_resize_x,
        hr_resize_y=hr_resize_y,
        hr_checkpoint_name=None if hr_checkpoint_name == 'Use same checkpoint' else hr_checkpoint_name,
        hr_sampler_name=None if hr_sampler_name == 'Use same sampler' else hr_sampler_name,
        hr_prompt=hr_prompt,
        hr_negative_prompt=hr_negative_prompt,
        override_settings=override_settings,
    )
    # bylyl: 函数将modules.scripts.scripts_txt2img赋值给p.scripts，并将args赋值给p.script_args。
    p.scripts = modules.scripts.scripts_txt2img
    # bylyl:args赋值给p.script_args 
    p.script_args = args
    #bylyl:将request.username赋值给p.user。
    p.user = request.username
    # bylyl:这段代码是在检查 cmd_opts（启动这个项目时候的配置参数） 对象的 enable_console_prompts 属性是否为真。
    # 如果为真，那么就会在控制台打印出一条包含 prompt 的消息。
    if cmd_opts.enable_console_prompts:
        print(f"\ntxt2img: {prompt}", file=shared.progress_print_out)
    # bylyl:这是一个上下文管理器，它确保在代码块执行完毕后调用 p 对象的 close 方法。
    # p 可能是一个需要在使用完毕后关闭的资源，如文件、网络连接、线程等。
    with closing(p):
        # bylyl:modules.scripts.scripts_txt2img.run 函数，并将结果保存在 processed 变量中。
        # 这个函数可能是用于处理或生成图像的主要功能，p 和 *args 是传递给这个函数的参数。
        processed = modules.scripts.scripts_txt2img.run(p, *args)
        # bylyl:如果为 None，那么可能表示 modules.scripts.scripts_txt2img.run 
        # 函数没有返回任何结果，或者处理过程失败。
        if processed is None:
            # 函数并将结果保存在 processed 中。这可能是一种备用的处理或生成图像的方法。
            processed = processing.process_images(p)
    #bylyl:清除进度条  
    shared.total_tqdm.clear()
    # 这行代码调用 processed 对象的 js 方法，并将返回的结果存储在 generation_info_js 变量中。
    # processed 可能是一个处理或生成图像的结果对象，js 方法可能用于将这个结果对象转化为可以
    # 在 JavaScript 中使用的形式，比如 JSON 格式。
    generation_info_js = processed.js()
    # 如果 opts.samples_log_stdout 为真，那么就将 generation_info_js 打印到标准输出。
    if opts.samples_log_stdout:
        print(generation_info_js)
    # 如果 opts.do_not_show_images 为真，那么就清空 processed 对象的 images 属性，即将其设置为一个空列表。
    # 即可以理解为不展示图片
    if opts.do_not_show_images:
        processed.images = []

    return processed.images, generation_info_js, plaintext_to_html(processed.info), plaintext_to_html(processed.comments, classname="comments")
