# 导入 modules\launch_utils 模块
from modules import launch_utils

# 导入启动参数
args = launch_utils.args
python = launch_utils.python
git = launch_utils.git
index_url = launch_utils.index_url
dir_repos = launch_utils.dir_repos

commit_hash = launch_utils.commit_hash
git_tag = launch_utils.git_tag

run = launch_utils.run
is_installed = launch_utils.is_installed
repo_dir = launch_utils.repo_dir

run_pip = launch_utils.run_pip
check_run_python = launch_utils.check_run_python
git_clone = launch_utils.git_clone
git_pull_recursive = launch_utils.git_pull_recursive
list_extensions = launch_utils.list_extensions
run_extension_installer = launch_utils.run_extension_installer
prepare_environment = launch_utils.prepare_environment
configure_for_tests = launch_utils.configure_for_tests
start = launch_utils.start


def main():
    # 如果启动参数包含--dump-sysinfo参数，则获取系统信息并输出到根目录的文件里，程序结束
    if args.dump_sysinfo:
        filename = launch_utils.dump_sysinfo()

        print(f"Sysinfo saved as {filename}. Exiting...")

        exit(0)

    launch_utils.startup_timer.record("initial startup")

    # 创建"prepare environment"计时器，记录准备环境所用的时间
    with launch_utils.startup_timer.subcategory("prepare environment"):
        if not args.skip_prepare_environment:
            # 若命令行参数没有配置跳过准备环境参数则执行 准备环境 操作。
            prepare_environment()

    if args.test_server:
        configure_for_tests()
    # 调用launch_utils里的start方法
    start()


if __name__ == "__main__":
    main()
