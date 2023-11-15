import os
import threading

from modules import shared, errors, cache, scripts
from modules.gitpython_hack import Repo
from modules.paths_internal import extensions_dir, extensions_builtin_dir, script_path  # noqa: F401

extensions = []

os.makedirs(extensions_dir, exist_ok=True)


def active():
    """
    返回可用插件列表
    """
    if shared.cmd_opts.disable_all_extensions or shared.opts.disable_all_extensions == "all":
        # 配置了 禁用所有扩展
        return []
    elif shared.cmd_opts.disable_extra_extensions or shared.opts.disable_all_extensions == "extra":
        # 配置了 禁用所有非内置扩展
        return [x for x in extensions if x.enabled and x.is_builtin]
    else:
        return [x for x in extensions if x.enabled]


class Extension:
    lock = threading.Lock()
    cached_fields = ['remote', 'commit_date', 'branch', 'commit_hash', 'version']

    def __init__(self, name, path, enabled=True, is_builtin=False):
        self.name = name
        self.path = path
        # 是否启用
        self.enabled = enabled
        self.status = ''
        self.can_update = False
        # 是否为内置扩展
        self.is_builtin = is_builtin
        self.commit_hash = ''
        self.commit_date = None
        self.version = ''
        self.branch = None
        self.remote = None
        self.have_info_from_repo = False

    def to_dict(self):
        """
        将对象的某些字段（即cached_fields列表中指定的字段）转换为一个字典。
        """
        return {x: getattr(self, x) for x in self.cached_fields}

    def from_dict(self, d):
        """
        将字典中的值设置到当前对象的字段上（要设置的字段就是cached_fields列表中的那些）
        """
        for field in self.cached_fields:
            setattr(self, field, d[field])

    def read_info_from_repo(self):
        if self.is_builtin or self.have_info_from_repo:
            return

        def read_from_repo():
            with self.lock:
                if self.have_info_from_repo:
                    return

                self.do_read_info_from_repo()

                return self.to_dict()

        try:
            d = cache.cached_data_for_file('extensions-git', self.name, os.path.join(self.path, ".git"), read_from_repo)
            self.from_dict(d)
        except FileNotFoundError:
            pass
        self.status = 'unknown' if self.status == '' else self.status

    def do_read_info_from_repo(self):
        """
        读取仓库信息
        """
        repo = None
        try:
            if os.path.exists(os.path.join(self.path, ".git")):
                repo = Repo(self.path)
        except Exception:
            errors.report(f"Error reading github repository info from {self.path}", exc_info=True)

        if repo is None or repo.bare:
            # repo为空或者repo是一个裸仓库（指没有任何工作目录（即没有文件系统中的文件）的仓库。）
            self.remote = None
        else:
            try:
                self.remote = next(repo.remote().urls, None)
                commit = repo.head.commit
                self.commit_date = commit.committed_date
                if repo.active_branch:
                    self.branch = repo.active_branch.name
                self.commit_hash = commit.hexsha
                self.version = self.commit_hash[:8]

            except Exception:
                errors.report(f"Failed reading extension data from Git repository ({self.name})", exc_info=True)
                self.remote = None

        self.have_info_from_repo = True

    def list_files(self, subdir, extension):
        dirpath = os.path.join(self.path, subdir)
        if not os.path.isdir(dirpath):
            # dirpath不是目录
            return []

        res = []
        # 遍历dirpath下的文件（升序排列）
        for filename in sorted(os.listdir(dirpath)):
            # ScriptFile = namedtuple("ScriptFile", ["basedir", "filename", "path"])
            # 组装 ScriptFile 并添加到 res 列表
            res.append(scripts.ScriptFile(self.path, filename, os.path.join(dirpath, filename)))

        # 过滤列表，只要后缀为extension的文件
        res = [x for x in res if os.path.splitext(x.path)[1].lower() == extension and os.path.isfile(x.path)]

        return res

    def check_updates(self):
        """
        检查插件版本
        """
        repo = Repo(self.path)
        for fetch in repo.remote().fetch(dry_run=True):
            if fetch.flags != fetch.HEAD_UPTODATE:
                self.can_update = True
                self.status = "new commits"
                return

        try:
            origin = repo.rev_parse('origin')
            if repo.head.commit != origin:
                self.can_update = True
                self.status = "behind HEAD"
                return
        except Exception:
            self.can_update = False
            self.status = "unknown (remote error)"
            return

        self.can_update = False
        self.status = "latest"

    def fetch_and_reset_hard(self, commit='origin'):
        repo = Repo(self.path)
        # Fix: `error: Your local changes to the following files would be overwritten by merge`,
        # because WSL2 Docker set 755 file permissions instead of 644, this results to the error.
        repo.git.fetch(all=True)
        repo.git.reset(commit, hard=True)
        self.have_info_from_repo = False


def list_extensions():
    """
    遍历extensions和extensions-builtin目录，返回Extension列表
    不包含禁用的插件
    """
    extensions.clear()

    if not os.path.isdir(extensions_dir):
        return

    if shared.cmd_opts.disable_all_extensions:
        print("*** \"--disable-all-extensions\" arg was used, will not load any extensions ***")
    elif shared.opts.disable_all_extensions == "all":
        print("*** \"Disable all extensions\" option was set, will not load any extensions ***")
    elif shared.cmd_opts.disable_extra_extensions:
        print("*** \"--disable-extra-extensions\" arg was used, will only load built-in extensions ***")
    elif shared.opts.disable_all_extensions == "extra":
        print("*** \"Disable all extensions\" option was set, will only load built-in extensions ***")

    extension_paths = []
    for dirname in [extensions_dir, extensions_builtin_dir]:
        if not os.path.isdir(dirname):
            return

        for extension_dirname in sorted(os.listdir(dirname)):
            path = os.path.join(dirname, extension_dirname)
            if not os.path.isdir(path):
                continue

            extension_paths.append((extension_dirname, path, dirname == extensions_builtin_dir))

    for dirname, path, is_builtin in extension_paths:
        extension = Extension(name=dirname, path=path, enabled=dirname not in shared.opts.disabled_extensions,
                              is_builtin=is_builtin)
        extensions.append(extension)
