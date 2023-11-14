import torch
from torch.nn.functional import silu
from types import MethodType

from modules import devices, sd_hijack_optimizations, shared, script_callbacks, errors, sd_unet
from modules.hypernetworks import hypernetwork
from modules.shared import cmd_opts
from modules import sd_hijack_clip, sd_hijack_open_clip, sd_hijack_unet, sd_hijack_xlmr, xlmr

import ldm.modules.attention
import ldm.modules.diffusionmodules.model
import ldm.modules.diffusionmodules.openaimodel
import ldm.models.diffusion.ddim
import ldm.models.diffusion.plms
import ldm.modules.encoders.modules

import sgm.modules.attention
import sgm.modules.diffusionmodules.model
import sgm.modules.diffusionmodules.openaimodel
import sgm.modules.encoders.modules

attention_CrossAttention_forward = ldm.modules.attention.CrossAttention.forward
diffusionmodules_model_nonlinearity = ldm.modules.diffusionmodules.model.nonlinearity
diffusionmodules_model_AttnBlock_forward = ldm.modules.diffusionmodules.model.AttnBlock.forward

# new memory efficient cross attention blocks do not support hypernets and we already
# have memory efficient cross attention anyway, so this disables SD2.0's memory efficient cross attention
ldm.modules.attention.MemoryEfficientCrossAttention = ldm.modules.attention.CrossAttention
ldm.modules.attention.BasicTransformerBlock.ATTENTION_MODES["softmax-xformers"] = ldm.modules.attention.CrossAttention

# silence new console spam from SD2
ldm.modules.attention.print = shared.ldm_print
ldm.modules.diffusionmodules.model.print = shared.ldm_print
ldm.util.print = shared.ldm_print
ldm.models.diffusion.ddpm.print = shared.ldm_print

optimizers = []
current_optimizer: sd_hijack_optimizations.SdOptimization = None


def list_optimizers():
    new_optimizers = script_callbacks.list_optimizers_callback()

    new_optimizers = [x for x in new_optimizers if x.is_available()]

    new_optimizers = sorted(new_optimizers, key=lambda x: x.priority, reverse=True)

    optimizers.clear()
    optimizers.extend(new_optimizers)


def apply_optimizations(option=None):
    global current_optimizer

    undo_optimizations()

    if len(optimizers) == 0:
        # a script can access the model very early, and optimizations would not be filled by then
        current_optimizer = None
        return ''

    ldm.modules.diffusionmodules.model.nonlinearity = silu
    ldm.modules.diffusionmodules.openaimodel.th = sd_hijack_unet.th

    sgm.modules.diffusionmodules.model.nonlinearity = silu
    sgm.modules.diffusionmodules.openaimodel.th = sd_hijack_unet.th

    if current_optimizer is not None:
        current_optimizer.undo()
        current_optimizer = None

    selection = option or shared.opts.cross_attention_optimization
    if selection == "Automatic" and len(optimizers) > 0:
        matching_optimizer = next(iter([x for x in optimizers if x.cmd_opt and getattr(shared.cmd_opts, x.cmd_opt, False)]), optimizers[0])
    else:
        matching_optimizer = next(iter([x for x in optimizers if x.title() == selection]), None)

    if selection == "None":
        matching_optimizer = None
    elif selection == "Automatic" and shared.cmd_opts.disable_opt_split_attention:
        matching_optimizer = None
    elif matching_optimizer is None:
        matching_optimizer = optimizers[0]

    if matching_optimizer is not None:
        print(f"Applying attention optimization: {matching_optimizer.name}... ", end='')
        matching_optimizer.apply()
        print("done.")
        current_optimizer = matching_optimizer
        return current_optimizer.name
    else:
        print("Disabling attention optimization")
        return ''


def undo_optimizations():
    ldm.modules.diffusionmodules.model.nonlinearity = diffusionmodules_model_nonlinearity
    ldm.modules.attention.CrossAttention.forward = hypernetwork.attention_CrossAttention_forward
    ldm.modules.diffusionmodules.model.AttnBlock.forward = diffusionmodules_model_AttnBlock_forward

    sgm.modules.diffusionmodules.model.nonlinearity = diffusionmodules_model_nonlinearity
    sgm.modules.attention.CrossAttention.forward = hypernetwork.attention_CrossAttention_forward
    sgm.modules.diffusionmodules.model.AttnBlock.forward = diffusionmodules_model_AttnBlock_forward


def fix_checkpoint():
    """checkpoints are now added and removed in embedding/hypernet code, since torch doesn't want
    checkpoints to be added when not training (there's a warning)"""

    pass


def weighted_loss(sd_model, pred, target, mean=True):
    #Calculate the weight normally, but ignore the mean
    loss = sd_model._old_get_loss(pred, target, mean=False)

    #Check if we have weights available
    weight = getattr(sd_model, '_custom_loss_weight', None)
    if weight is not None:
        loss *= weight

    #Return the loss, as mean if specified
    return loss.mean() if mean else loss

def weighted_forward(sd_model, x, c, w, *args, **kwargs):
    try:
        #Temporarily append weights to a place accessible during loss calc
        sd_model._custom_loss_weight = w

        #Replace 'get_loss' with a weight-aware one. Otherwise we need to reimplement 'forward' completely
        #Keep 'get_loss', but don't overwrite the previous old_get_loss if it's already set
        if not hasattr(sd_model, '_old_get_loss'):
            sd_model._old_get_loss = sd_model.get_loss
        sd_model.get_loss = MethodType(weighted_loss, sd_model)

        #Run the standard forward function, but with the patched 'get_loss'
        return sd_model.forward(x, c, *args, **kwargs)
    finally:
        try:
            #Delete temporary weights if appended
            del sd_model._custom_loss_weight
        except AttributeError:
            pass

        #If we have an old loss function, reset the loss function to the original one
        if hasattr(sd_model, '_old_get_loss'):
            sd_model.get_loss = sd_model._old_get_loss
            del sd_model._old_get_loss

def apply_weighted_forward(sd_model):
    #Add new function 'weighted_forward' that can be called to calc weighted loss
    sd_model.weighted_forward = MethodType(weighted_forward, sd_model)

def undo_weighted_forward(sd_model):
    try:
        del sd_model.weighted_forward
    except AttributeError:
        pass


class StableDiffusionModelHijack:
    fixes = None
    layers = None
    circular_enabled = False
    clip = None
    optimization_method = None

    def __init__(self):
        import modules.textual_inversion.textual_inversion

        self.extra_generation_params = {}
        self.comments = []

        self.embedding_db = modules.textual_inversion.textual_inversion.EmbeddingDatabase()
        self.embedding_db.add_embedding_dir(cmd_opts.embeddings_dir)

    def apply_optimizations(self, option=None):
        try:
            self.optimization_method = apply_optimizations(option)
        except Exception as e:
            errors.display(e, "applying cross attention optimization")
            undo_optimizations()

    def hijack(self, m):
        # 获取模型的条件器conditioner对象
        conditioner = getattr(m, 'conditioner', None)
        # 如果不是空
        if conditioner:
            # 定义一个新的的数组
            text_cond_models = []
            # 循环 conditioner的文本编码器embedders 模型的编码器  设置好对应的编码器  主要是看几个编码器
            for i in range(len(conditioner.embedders)):
                embedder = conditioner.embedders[i]
                # 获取编码器的名称
                typename = type(embedder).__name__
                if typename == 'FrozenOpenCLIPEmbedder':
                    embedder.model.token_embedding = EmbeddingsWithFixes(embedder.model.token_embedding, self)
                    conditioner.embedders[i] = sd_hijack_open_clip.FrozenOpenCLIPEmbedderWithCustomWords(embedder, self)
                    text_cond_models.append(conditioner.embedders[i])
                if typename == 'FrozenCLIPEmbedder':
                    model_embeddings = embedder.transformer.text_model.embeddings
                    model_embeddings.token_embedding = EmbeddingsWithFixes(model_embeddings.token_embedding, self)
                    conditioner.embedders[i] = sd_hijack_clip.FrozenCLIPEmbedderForSDXLWithCustomWords(embedder, self)
                    text_cond_models.append(conditioner.embedders[i])
                if typename == 'FrozenOpenCLIPEmbedder2':
                    embedder.model.token_embedding = EmbeddingsWithFixes(embedder.model.token_embedding, self, textual_inversion_key='clip_g')
                    conditioner.embedders[i] = sd_hijack_open_clip.FrozenOpenCLIPEmbedder2WithCustomWords(embedder, self)
                    text_cond_models.append(conditioner.embedders[i])
            # 将当前的编码器赋值给cond_stage_model
            if len(text_cond_models) == 1:
                m.cond_stage_model = text_cond_models[0]
            else:
                m.cond_stage_model = conditioner
        # 最终根据编码器 获得cond_stage_model和token_embedding的配置
        if type(m.cond_stage_model) == xlmr.BertSeriesModelWithTransformation:
            model_embeddings = m.cond_stage_model.roberta.embeddings
            model_embeddings.token_embedding = EmbeddingsWithFixes(model_embeddings.word_embeddings, self)
            m.cond_stage_model = sd_hijack_xlmr.FrozenXLMREmbedderWithCustomWords(m.cond_stage_model, self)

        elif type(m.cond_stage_model) == ldm.modules.encoders.modules.FrozenCLIPEmbedder:
            model_embeddings = m.cond_stage_model.transformer.text_model.embeddings
            model_embeddings.token_embedding = EmbeddingsWithFixes(model_embeddings.token_embedding, self)
            m.cond_stage_model = sd_hijack_clip.FrozenCLIPEmbedderWithCustomWords(m.cond_stage_model, self)

        elif type(m.cond_stage_model) == ldm.modules.encoders.modules.FrozenOpenCLIPEmbedder:
            m.cond_stage_model.model.token_embedding = EmbeddingsWithFixes(m.cond_stage_model.model.token_embedding, self)
            m.cond_stage_model = sd_hijack_open_clip.FrozenOpenCLIPEmbedderWithCustomWords(m.cond_stage_model, self)
        # 给模型加了个方法有没有用不太知道
        apply_weighted_forward(m)
        # 如果模型有这个 可以进行特殊的劫持 可以先不考虑
        if m.cond_stage_key == "edit":
            sd_hijack_unet.hijack_ddpm_edit()
        # 感觉是个优化方案
        self.apply_optimizations()
        #将模型的价值器 赋值给clip
        self.clip = m.cond_stage_model
        # 将模型层展平成列表
        def flatten(el):
            flattened = [flatten(children) for children in el.children()]
            res = [el]
            for c in flattened:
                res += c
            return res

        self.layers = flatten(m)
        # 首先检查是否已经备份了UNet原始前向函数
        if not hasattr(ldm.modules.diffusionmodules.openaimodel, 'copy_of_UNetModel_forward_for_webui'):
            # 如果没有,进行备份保存
            ldm.modules.diffusionmodules.openaimodel.copy_of_UNetModel_forward_for_webui = ldm.modules.diffusionmodules.openaimodel.UNetModel.forward
        # 替换实际前向调用,从而“劫持”UNet的前向定义
        ldm.modules.diffusionmodules.openaimodel.UNetModel.forward = sd_unet.UNetModel_forward

    def undo_hijack(self, m):
        conditioner = getattr(m, 'conditioner', None)
        if conditioner:
            for i in range(len(conditioner.embedders)):
                embedder = conditioner.embedders[i]
                if isinstance(embedder, (sd_hijack_open_clip.FrozenOpenCLIPEmbedderWithCustomWords, sd_hijack_open_clip.FrozenOpenCLIPEmbedder2WithCustomWords)):
                    embedder.wrapped.model.token_embedding = embedder.wrapped.model.token_embedding.wrapped
                    conditioner.embedders[i] = embedder.wrapped
                if isinstance(embedder, sd_hijack_clip.FrozenCLIPEmbedderForSDXLWithCustomWords):
                    embedder.wrapped.transformer.text_model.embeddings.token_embedding = embedder.wrapped.transformer.text_model.embeddings.token_embedding.wrapped
                    conditioner.embedders[i] = embedder.wrapped

            if hasattr(m, 'cond_stage_model'):
                delattr(m, 'cond_stage_model')

        elif type(m.cond_stage_model) == sd_hijack_xlmr.FrozenXLMREmbedderWithCustomWords:
            m.cond_stage_model = m.cond_stage_model.wrapped

        elif type(m.cond_stage_model) == sd_hijack_clip.FrozenCLIPEmbedderWithCustomWords:
            m.cond_stage_model = m.cond_stage_model.wrapped

            model_embeddings = m.cond_stage_model.transformer.text_model.embeddings
            if type(model_embeddings.token_embedding) == EmbeddingsWithFixes:
                model_embeddings.token_embedding = model_embeddings.token_embedding.wrapped
        elif type(m.cond_stage_model) == sd_hijack_open_clip.FrozenOpenCLIPEmbedderWithCustomWords:
            m.cond_stage_model.wrapped.model.token_embedding = m.cond_stage_model.wrapped.model.token_embedding.wrapped
            m.cond_stage_model = m.cond_stage_model.wrapped

        undo_optimizations()
        undo_weighted_forward(m)

        self.apply_circular(False)
        self.layers = None
        self.clip = None

        ldm.modules.diffusionmodules.openaimodel.UNetModel.forward = ldm.modules.diffusionmodules.openaimodel.copy_of_UNetModel_forward_for_webui
    #它有一个参数enable表示是否启用循环边缘填充。 暂时理解不知道啥意思
    def apply_circular(self, enable):
        # 首先检查当前模型(self)的circular_enabled属性是否等于enable参数,如果等则直接返回,不做额外操作。
        if self.circular_enabled == enable:
            return
        # 否则将self.circular_enabled设置为enable参数值。
        self.circular_enabled = enable
        # 设置为'circular',表示开启循环边缘填充;如果enable为False,则设置为'zeros',表示使用0填充。
        for layer in [layer for layer in self.layers if type(layer) == torch.nn.Conv2d]:
            layer.padding_mode = 'circular' if enable else 'zeros'
    # 这段代码用来清除或重置一个模型对象中的注释(comments)和额外生成参数(extra_generation_params)。
    def clear_comments(self):
        self.comments = []
        self.extra_generation_params = {}

    def get_prompt_lengths(self, text):
        if self.clip is None:
            return "-", "-"

        _, token_count = self.clip.process_texts([text])

        return token_count, self.clip.get_target_prompt_token_count(token_count)

    def redo_hijack(self, m):
        self.undo_hijack(m)
        self.hijack(m)


class EmbeddingsWithFixes(torch.nn.Module):
    def __init__(self, wrapped, embeddings, textual_inversion_key='clip_l'):
        super().__init__()
        self.wrapped = wrapped
        self.embeddings = embeddings
        self.textual_inversion_key = textual_inversion_key

    def forward(self, input_ids):
        batch_fixes = self.embeddings.fixes
        self.embeddings.fixes = None

        inputs_embeds = self.wrapped(input_ids)

        if batch_fixes is None or len(batch_fixes) == 0 or max([len(x) for x in batch_fixes]) == 0:
            return inputs_embeds

        vecs = []
        for fixes, tensor in zip(batch_fixes, inputs_embeds):
            for offset, embedding in fixes:
                vec = embedding.vec[self.textual_inversion_key] if isinstance(embedding.vec, dict) else embedding.vec
                emb = devices.cond_cast_unet(vec)
                emb_len = min(tensor.shape[0] - offset - 1, emb.shape[0])
                tensor = torch.cat([tensor[0:offset + 1], emb[0:emb_len], tensor[offset + 1 + emb_len:]])

            vecs.append(tensor)

        return torch.stack(vecs)


def add_circular_option_to_conv_2d():
    conv2d_constructor = torch.nn.Conv2d.__init__

    def conv2d_constructor_circular(self, *args, **kwargs):
        return conv2d_constructor(self, *args, padding_mode='circular', **kwargs)

    torch.nn.Conv2d.__init__ = conv2d_constructor_circular


model_hijack = StableDiffusionModelHijack()


def register_buffer(self, name, attr):
    """
    Fix register buffer bug for Mac OS.
    """

    if type(attr) == torch.Tensor:
        if attr.device != devices.device:
            attr = attr.to(device=devices.device, dtype=(torch.float32 if devices.device.type == 'mps' else None))

    setattr(self, name, attr)


ldm.models.diffusion.ddim.DDIMSampler.register_buffer = register_buffer
ldm.models.diffusion.plms.PLMSSampler.register_buffer = register_buffer
