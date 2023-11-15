import sys
# 用于对文本进行格式化。这个模块提供了一些函数，可以帮助你控制文本的宽度、填充、折行等。
import textwrap
# 该模块提供了一个标准接口来提取、格式化和打印 Python 程序的堆栈跟踪结果。
# https://docs.python.org/zh-cn/3/library/traceback.html
import traceback

exception_records = []


# 记录异常信息
def record_exception():
    # https://docs.python.org/zh-cn/3/library/sys.html?highlight=sys%20exc_info#sys.exc_info
    # 通过sys.exc_info()获取当前异常的信息，包括异常类型、异常值e和追踪信息tb。
    _, e, tb = sys.exc_info()
    # 检查是否有异常发生。
    if e is None:
        # 没有异常,直接返回
        return
    # 如果存在之前的异常记录，并且新异常与最后一个异常相同，那么函数也直接返回，不再记录新的异常。
    if exception_records and exception_records[-1] == e:
        return

    from modules import sysinfo
    # 使用 sysinfo.format_exception 格式化异常信息，并将结果添加到 exception_records 列表中。
    exception_records.append(sysinfo.format_exception(e, tb))
    # 如果exception_records列表的长度超过了5，那么会从列表的开头移除最早的一个记录。
    if len(exception_records) > 5:
        exception_records.pop(0)


def report(message: str, *, exc_info: bool = False) -> None:
    """
    Print an error message to stderr, with optional traceback.
    """

    record_exception()
    # 根据 \n 进行字符串分割
    for line in message.splitlines():
        print("***", line, file=sys.stderr)
    if exc_info:
        # 当发生异常时，会捕获异常并使用traceback.format_exc()来获取格式化过的异常字符串。
        # 然后，使用textwrap.indent函数为每个行添加四个空格的缩进，
        # 最后将格式化后的异常信息打印到标准错误输出。
        print(textwrap.indent(traceback.format_exc(), "    "), file=sys.stderr)
        print("---", file=sys.stderr)


def print_error_explanation(message):
    record_exception()
    # strip() 去除字符串message的前后空白字符（例如换行符和空格）。
    # 将字符串message分割成多行文本
    lines = message.strip().split("\n")
    # 找出其中最长的行的长度。
    max_len = max([len(x) for x in lines])
    # 打印 max_len 个 = 号
    print('=' * max_len, file=sys.stderr)
    for line in lines:
        print(line, file=sys.stderr)
    print('=' * max_len, file=sys.stderr)


def display(e: Exception, task, *, full_traceback=False):
    record_exception()
    # 若task不存在或者为假，它会被替换为字符串'error'
    # type(e).__name__ 打印出错误类型。
    print(f"{task or 'error'}: {type(e).__name__}", file=sys.stderr)
    te = traceback.TracebackException.from_exception(e)
    if full_traceback:
        # include frames leading up to the try-catch block
        te.stack = traceback.StackSummary(traceback.extract_stack()[:-2] + te.stack)
    print(*te.format(), sep="", file=sys.stderr)

    message = str(e)
    if "copying a param with shape torch.Size([640, 1024]) from checkpoint, the shape in current model is torch.Size([640, 768])" in message:
        print_error_explanation("""
The most likely cause of this is you are trying to load Stable Diffusion 2.0 model without specifying its config file.
See https://github.com/AUTOMATIC1111/stable-diffusion-webui/wiki/Features#stable-diffusion-20 for how to solve this.
        """)


already_displayed = {}


def display_once(e: Exception, task):
    record_exception()
    if task in already_displayed:
        return

    display(e, task)
    already_displayed[task] = 1


def run(code, task):
    """
    尝试运行code函数，并在发生异常时捕获异常，然后调用display函数显示任务和异常。
    """
    try:
        code()
    except Exception as e:
        display(task, e)


def check_versions():
    from packaging import version
    from modules import shared

    import torch
    import gradio

    expected_torch_version = "2.0.0"
    expected_xformers_version = "0.0.20"
    expected_gradio_version = "3.41.2"

    if version.parse(torch.__version__) < version.parse(expected_torch_version):
        print_error_explanation(f"""
You are running torch {torch.__version__}.
The program is tested to work with torch {expected_torch_version}.
To reinstall the desired version, run with commandline flag --reinstall-torch.
Beware that this will cause a lot of large files to be downloaded, as well as
there are reports of issues with training tab on the latest version.

Use --skip-version-check commandline argument to disable this check.
        """.strip())

    if shared.xformers_available:
        import xformers

        if version.parse(xformers.__version__) < version.parse(expected_xformers_version):
            print_error_explanation(f"""
You are running xformers {xformers.__version__}.
The program is tested to work with xformers {expected_xformers_version}.
To reinstall the desired version, run with commandline flag --reinstall-xformers.

Use --skip-version-check commandline argument to disable this check.
            """.strip())

    if gradio.__version__ != expected_gradio_version:
        print_error_explanation(f"""
You are running gradio {gradio.__version__}.
The program is designed to work with gradio {expected_gradio_version}.
Using a different version of gradio is extremely likely to break the program.

Reasons why you have the mismatched gradio version can be:
  - you use --skip-install flag.
  - you use webui.py to start the program instead of launch.py.
  - an extension installs the incompatible gradio version.

Use --skip-version-check commandline argument to disable this check.
        """.strip())
