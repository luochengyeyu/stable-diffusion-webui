import time
import argparse


# zy 计时器子类别
class TimerSubcategory:
    def __init__(self, timer, category):
        self.timer = timer
        self.category = category
        self.start = None
        self.original_base_category = timer.base_category

    # zy 上下文管理器方法：执行一些预处理工作
    def __enter__(self):
        self.start = time.time()
        self.timer.base_category = self.original_base_category + self.category + "/"
        self.timer.subcategory_level += 1

        if self.timer.print_log:
            print(f"{'  ' * self.timer.subcategory_level}{self.category}:")

    # zy 上下文管理器方法：with语句执行完毕，最后调用此方法，一般用来释放资源等操作
    def __exit__(self, exc_type, exc_val, exc_tb):
        elapsed_for_subcategroy = time.time() - self.start
        self.timer.base_category = self.original_base_category
        self.timer.add_time_to_record(self.original_base_category + self.category, elapsed_for_subcategroy)
        self.timer.subcategory_level -= 1
        self.timer.record(self.category, disable_log=True)


class Timer:
    def __init__(self, print_log=False):
        self.start = time.time()
        self.records = {}
        self.total = 0
        self.base_category = ''
        self.print_log = print_log
        # 用于控制打印几个空格，每多一个level就多打2个空格
        self.subcategory_level = 0

    # zy 当前时间和上次record时间的差值，并更新start为当前时间
    def elapsed(self):
        end = time.time()
        res = end - self.start
        self.start = end
        return res

    # zy 将指定类别的时间存储到records字典中，key为category
    def add_time_to_record(self, category, amount):
        if category not in self.records:
            self.records[category] = 0

        self.records[category] += amount

    # 记录时间
    def record(self, category, extra_time=0, disable_log=False):
        e = self.elapsed()

        self.add_time_to_record(self.base_category + category, e + extra_time)

        self.total += e + extra_time
        # 打印日志
        if self.print_log and not disable_log:
            print(f"{'  ' * self.subcategory_level}{category}: done in {e + extra_time:.3f}s")

    # zy 获取子计时器
    def subcategory(self, name):
        self.elapsed()

        subcat = TimerSubcategory(self, name)
        return subcat

    def summary(self):
        # 保留一位小数的浮点数的单位为s的时间字符串
        res = f"{self.total:.1f}s"
        # zy 从 self.records 字典中筛选出那些 time_taken 大于或等于0.1且 category 中不包含字符'/'的项，
        # 并将这些项作为元组添加到additions列表中。
        additions = [(category, time_taken) for category, time_taken in self.records.items() if
                     time_taken >= 0.1 and '/' not in category]
        if not additions:
            return res

        res += " ("
        res += ", ".join([f"{category}: {time_taken:.1f}s" for category, time_taken in additions])
        res += ")"

        return res

    def dump(self):
        return {'total': self.total, 'records': self.records}

    def reset(self):
        self.__init__()


parser = argparse.ArgumentParser(add_help=False)
parser.add_argument("--log-startup", action='store_true', help="print a detailed log of what's happening at startup")
args = parser.parse_known_args()[0]
# zy args.log_startup为命令行参数--log-startup，用于打印webui启动时的日志信息
startup_timer = Timer(print_log=args.log_startup)

startup_record = None
