# 导入依赖库
from modules import initialize

# 导入依赖库
initialize.imports()
# 检查依赖库版本
initialize.check_versions()
initialize.initialize()


# txt2img
def text2imgapi():
    from modules.api import models
    from contextlib import closing
    import modules.shared as shared
    from modules.shared import opts
    from modules.processing import StableDiffusionProcessingTxt2Img, process_images
    from modules.call_queue import queue_lock

    populate = models.StableDiffusionTxt2ImgProcessingAPI(
        prompt='1girl fullbody',
        negative_prompt='',
        styles=None,
        seed=-1,
        subseed=-1,
        subseed_strength=0,
        seed_resize_from_h=-1,
        seed_resize_from_w=-1,
        sampler_name='Euler',
        batch_size=1,
        n_iter=1,
        steps=10,
        cfg_scale=7.0,
        width=512,
        height=768,
        restore_faces=None,
        tiling=None,
        do_not_save_samples=False,
        do_not_save_grid=False,
        eta=None, denoising_strength=0,
        s_min_uncond=None,
        s_churn=None,
        s_tmax=None,
        s_tmin=None,
        s_noise=None,
        override_settings=None,
        override_settings_restore_afterwards=True,
        refiner_checkpoint=None,
        refiner_switch_at=None,
        disable_extra_networks=False,
        comments=None,
        enable_hr=False,
        firstphase_width=0,
        firstphase_height=0,
        hr_scale=2.0,
        hr_upscaler=None,
        hr_second_pass_steps=0,
        hr_resize_x=0,
        hr_resize_y=0,
        hr_checkpoint_name=None,
        hr_sampler_name=None,
        hr_prompt='',
        hr_negative_prompt=''
    )
    if populate.sampler_name:
        populate.sampler_index = None  # prevent a warning later on

    args = vars(populate)
    args.pop('script_name', None)
    args.pop('script_args', None)  # will refeed them to the pipeline directly after initializing them
    args.pop('alwayson_scripts', None)

    args.pop('send_images', True)
    args.pop('save_images', None)

    with queue_lock:
        with closing(StableDiffusionProcessingTxt2Img(sd_model=shared.sd_model, **args)) as p:
            p.is_api = True
            p.outpath_grids = opts.outdir_txt2img_grids
            p.outpath_samples = opts.outdir_txt2img_samples

            try:
                shared.state.begin(job="scripts_txt2img")
                processed = process_images(p)
            finally:
                shared.state.end()
                shared.total_tqdm.clear()

    processed.images[0].save('output.png')

    # b64images = list(map(encode_pil_to_base64, processed.images)) if send_images else []
    #
    # return models.TextToImageResponse(images=b64images, parameters=vars(txt2imgreq), info=processed.js())


def main():
    # title = shared.sd_model.sd_checkpoint_info.title
    # print(f'model_name = {title}')
    text2imgapi()


if __name__ == "__main__":
    main()
