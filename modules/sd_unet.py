import torch.nn
import ldm.modules.diffusionmodules.openaimodel

from modules import script_callbacks, shared, devices

unet_options = []
current_unet_option = None
current_unet = None


def list_unets():
    new_unets = script_callbacks.list_unets_callback()

    unet_options.clear()
    unet_options.extend(new_unets)


def get_unet_option(option=None):
    option = option or shared.opts.sd_unet

    if option == "None":
        return None

    if option == "Automatic":
        name = shared.sd_model.sd_checkpoint_info.model_name

        options = [x for x in unet_options if x.model_name == name]

        option = options[0].label if options else "None"

    return next(iter([x for x in unet_options if x.label == option]), None)

#这段代码实现了动态加载不同Unet模型的功能: 
def apply_unet(option=None):
    # 定义了当前Unet模型和选项的全局变量
    global current_unet_option
    global current_unet
    # 根据传入的选项获取新的Unet模型对象
    new_option = get_unet_option(option)
    # 判断是否需要更新Unet模型
    if new_option == current_unet_option:
        return
    # 如果需要,打印日志并先卸载旧模型，也就是当前的current_unet不为空则需要卸载
    if current_unet is not None:
        print(f"Dectivating unet: {current_unet.option.label}")
        current_unet.deactivate()
    # 更新全局Unet变量
    current_unet_option = new_option
    # 如果更新后的option为空
    if current_unet_option is None:
        # 设置当前的unet为空
        current_unet = None
        # 如果是低显存模式 需要使用cpu加载model
        if not shared.sd_model.lowvram:
            shared.sd_model.model.diffusion_model.to(devices.device)
        # 结束返回
        return
    #否则 将模型移动到cpu上
    shared.sd_model.model.diffusion_model.to(devices.cpu)
    # 清理内存
    devices.torch_gc()
    # 通过option创建unet模型赋值给当前unet
    current_unet = current_unet_option.create_unet()
    # 设置当前unet的option属性
    current_unet.option = current_unet_option
    print(f"Activating unet: {current_unet.option.label}")
    #激活unet
    current_unet.activate()


class SdUnetOption:
    model_name = None
    """name of related checkpoint - this option will be selected automatically for unet if the name of checkpoint matches this"""

    label = None
    """name of the unet in UI"""

    def create_unet(self):
        """returns SdUnet object to be used as a Unet instead of built-in unet when making pictures"""
        raise NotImplementedError()


class SdUnet(torch.nn.Module):
    def forward(self, x, timesteps, context, *args, **kwargs):
        raise NotImplementedError()

    def activate(self):
        pass

    def deactivate(self):
        pass


def UNetModel_forward(self, x, timesteps=None, context=None, *args, **kwargs):
    if current_unet is not None:
        return current_unet.forward(x, timesteps, context, *args, **kwargs)

    return ldm.modules.diffusionmodules.openaimodel.copy_of_UNetModel_forward_for_webui(self, x, timesteps, context, *args, **kwargs)

