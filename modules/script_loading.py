import os
import importlib.util

from modules import errors


def load_module(path):
    """
    使用importlib.util模块来加载指定的Python模块
    """
    # 根据文件位置返回模块规范。
    module_spec = importlib.util.spec_from_file_location(os.path.basename(path), path)
    # 根据提供的规范创建模块。
    module = importlib.util.module_from_spec(module_spec)
    # 行module对象，完成模块的加载。
    module_spec.loader.exec_module(module)
    # 函数返回加载的模块。
    return module


def preload_extensions(extensions_dir, parser, extension_list=None):
    """
    预加载指定目录下的所有扩展模块。
    """
    # 检查extensions_dir是否是一个目录
    if not os.path.isdir(extensions_dir):
        return
    # 若extension_list未传参，就使用extensions_dir获取插件列表
    extensions = extension_list if extension_list is not None else os.listdir(extensions_dir)
    for dirname in sorted(extensions):
        preload_script = os.path.join(extensions_dir, dirname, "preload.py")
        # 对于每一个文件夹，检查是否存在一个名为"preload.py"的文件。
        if not os.path.isfile(preload_script):
            continue

        try:
            # 如果存在名为"preload.py"的文件，使用load_module函数加载这个模块，并检查模块是否有"preload"函数。
            module = load_module(preload_script)
            if hasattr(module, 'preload'):
                # 如果模块是有"preload"函数，调用这个函数并传入parser参数。
                module.preload(parser)

        except Exception:
            errors.report(f"Error running preload() for {preload_script}", exc_info=True)
