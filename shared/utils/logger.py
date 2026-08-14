"""
日志工具
同时输出到终端和文件
终端：美观格式，无时间戳
文件：时间戳记录
"""

import os
import sys
import time
from datetime import datetime


def format_num_params(num):
    """格式化参数数量，同时显示精确值和带单位"""
    if num >= 1e9:
        return f"{num:,} ({num / 1e9:.2f}B)"
    elif num >= 1e6:
        return f"{num:,} ({num / 1e6:.2f}M)"
    elif num >= 1e3:
        return f"{num:,} ({num / 1e3:.2f}K)"
    return f"{num:,} ({num})"


DIVIDER = "# " + "═" * 76


class Logger:
    """同时输出到终端和文件，每行带时间戳"""

    def __init__(self, log_dir, prefix="log", config=None):
        os.makedirs(log_dir, exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.log_file = os.path.join(log_dir, f"{prefix}_{timestamp}.log")
        self.terminal = sys.stdout
        self.line_start = True

        self.file = open(self.log_file, 'w', encoding='utf-8')
        self._write_header(config)

    def _write_header(self, config=None):
        """写入日志头信息"""
        header_lines = [
            DIVIDER,
            f"# 训练日志",
            f"# 创建时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
            DIVIDER,
            "",
        ]
        for line in header_lines:
            self.file.write(line + "\n")
            self.terminal.write(line + "\n")

        if config:
            self.print_config(config)

    def _get_timestamp(self):
        return datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    def _format_config(self, config_dict, title="训练参数配置"):
        """
        格式化配置为美观的多列布局

        Args:
            config_dict: 参数字典
            title: 配置标题

        Returns:
            list: 格式化后的行列表
        """
        if not config_dict:
            return []

        BOX_WIDTH = 70
        lines = []

        lines.append("")
        lines.append("┌" + "─" * (BOX_WIDTH - 2) + "┐")
        lines.append("│" + f"{title}".center(BOX_WIDTH - 2) + "│")
        lines.append("├" + "─" * (BOX_WIDTH - 2) + "┤")

        for key, value in config_dict.items():
            key_str = str(key)
            value_str = str(value)

            max_key_len = 28
            max_value_len = BOX_WIDTH - 2 - max_key_len - 7

            if len(key_str) > max_key_len:
                key_str = key_str[:max_key_len - 3] + "..."

            if len(value_str) > max_value_len:
                value_str = value_str[:max_value_len - 3] + "..."

            line = f"│  {key_str:<{max_key_len}}  :  {value_str:<{max_value_len}} │"
            lines.append(line)

        lines.append("└" + "─" * (BOX_WIDTH - 2) + "┘")
        lines.append("")

        return lines

    def _format_config_grouped(self, config_groups, title="训练参数配置"):
        """
        格式化分组配置为简洁美观的布局

        Args:
            config_groups: 分组字典，格式为 {"组名": {key: value, ...}, ...}
            title: 配置标题

        Returns:
            list: 格式化后的行列表
        """
        if not config_groups:
            return []

        lines = []

        lines.append("")
        lines.append(f"{DIVIDER}")
        lines.append(f"# {title}")
        lines.append(f"{DIVIDER}")
        lines.append("")

        for group_name, group_dict in config_groups.items():
            lines.append(f"  ▸ {group_name}")
            for key, value in group_dict.items():
                lines.append(f"    {key} = {value}")
            lines.append("")

        lines.append(DIVIDER)
        lines.append("")

        return lines

    def print_config(self, config_dict, title="训练参数配置"):
        """
        打印并保存参数配置（美观格式）

        Args:
            config_dict: 参数字典
            title: 配置标题
        """
        lines = self._format_config(config_dict, title)
        for line in lines:
            self.file.write(line + "\n")
            self.terminal.write(line + "\n")

    def print_config_grouped(self, config_groups, title="训练参数配置"):
        """
        打印并保存分组参数配置（美观格式）

        Args:
            config_groups: 分组字典，格式为 {"组名": {key: value, ...}, ...}
            title: 配置标题
        """
        lines = self._format_config_grouped(config_groups, title)
        for line in lines:
            self.file.write(line + "\n")
            # Windows 终端使用 GBK 编码，替换特殊字符避免编码错误
            safe_line = line.replace('▸', '>').replace('═', '=').replace('┌', '+').replace('┐', '+').replace('└', '+').replace('┘', '+').replace('│', '|')
            self.terminal.write(safe_line + "\n")

    def print_title(self, title):
        """
        打印大标题（居中，带装饰线）

        Args:
            title: 标题文本
        """
        BOX_WIDTH = 70
        lines = [
            "",
            "┌" + "═" * (BOX_WIDTH - 2) + "┐",
            "│" + f" {title} ".center(BOX_WIDTH - 2) + "│",
            "└" + "═" * (BOX_WIDTH - 2) + "┘",
            "",
        ]
        for line in lines:
            self.file.write(line + "\n")
            self.terminal.write(line + "\n")

    def print_section(self, title):
        """
        打印章节标题（带装饰线分割）

        Args:
            title: 章节标题
        """
        lines = [
            "",
            DIVIDER,
            f"# {title}",
            DIVIDER,
            "",
        ]
        for line in lines:
            self.file.write(line + "\n")
            self.terminal.write(line + "\n")

    def print_info(self, message):
        """
        打印普通信息（带缩进）

        Args:
            message: 信息文本
        """
        line = f"  {message}"
        self.file.write(line + "\n")
        self.terminal.write(line + "\n")

    def print_metrics(self, metrics_dict, title="评估指标", elapsed_time=None):
        """
        打印指标（特殊格式化，适合数字）

        Args:
            metrics_dict: 指标字典
            title: 标题
            elapsed_time: 耗时（秒），可选
        """
        BOX_WIDTH = 70
        lines = []

        lines.append("")
        lines.append("┌" + "─" * (BOX_WIDTH - 2) + "┐")
        lines.append("│" + f"{title}".center(BOX_WIDTH - 2) + "│")
        lines.append("├" + "─" * (BOX_WIDTH - 2) + "┤")

        for key, value in metrics_dict.items():
            key_str = str(key)
            if isinstance(value, float):
                value_str = f"{value:.4f}"
            else:
                value_str = str(value)

            line = f"│  {key_str:<30}  :  {value_str:>12} │"
            lines.append(line)
        
        if elapsed_time is not None:
            line = f"│  {'Time':<30}  :  {elapsed_time:>12.1f}s │"
            lines.append(line)

        lines.append("└" + "─" * (BOX_WIDTH - 2) + "┘")
        lines.append("")

        for line in lines:
            self.file.write(line + "\n")
            self.terminal.write(line + "\n")

    def print_separator(self, char="─", length=None):
        """
        打印分隔线

        Args:
            char: 分隔线字符
            length: 分隔线长度（默认70）
        """
        if length is None:
            length = 68
        line = "  " + char * length
        self.file.write(line + "\n")
        self.terminal.write(line + "\n")

    def write(self, message):
        """写入日志：终端无时间戳，文件有时间戳"""
        if message:
            if self.line_start:
                timestamp = self._get_timestamp()
                self.file.write(f"[{timestamp}] ")

            self.file.write(message)
            self.terminal.write(message)

            if message.endswith('\n'):
                self.line_start = True
            else:
                self.line_start = False

    def print(self, message):
        """打印日志（自动换行）"""
        self.write(message + "\n")

    def flush(self):
        self.terminal.flush()
        self.file.flush()

    def close(self):
        self.file.close()


class Timer:
    """统一的计时器，用于记录训练/验证时间"""
    def __init__(self):
        self.start_time = None
        self.measurements = {}
    
    def start(self, name="default"):
        """开始计时"""
        self.start_time = time.perf_counter()
        return self
    
    def stop(self, name="default"):
        """停止计时并记录"""
        if self.start_time is None:
            return 0.0
        elapsed = time.perf_counter() - self.start_time
        self.measurements[name] = elapsed
        self.start_time = None
        return elapsed
    
    def get(self, name="default"):
        """获取已记录的时间"""
        return self.measurements.get(name, 0.0)


def setup_logger(log_dir, prefix="log", config=None):
    """
    设置日志记录器

    Args:
        log_dir: 日志保存目录
        prefix: 日志文件前缀
        config: 参数字典（可选），会自动打印到日志

    Returns:
        logger: Logger 实例
    """
    logger = Logger(log_dir, prefix, config)
    sys.stdout = logger
    logger.print(f"日志已保存至: {logger.log_file}")
    return logger


__all__ = ["Logger", "setup_logger"]