"""
路径解析工具
提供权重保存目录与日志名称前缀的统一解析逻辑，供各项目 pipeline/train.py 复用
"""

import os


def resolve_model_dir(model_dir_arg, pipeline_dir, default_dir_name):
    """
    解析权重保存目录

    Args:
        model_dir_arg: 命令行 --model_dir 的值（空字符串表示未指定）
        pipeline_dir: pipeline 目录绝对路径（用于推导默认位置）
        default_dir_name: 默认子目录名（各项目规则不同，由调用方传入）

    Returns:
        绝对路径；--model_dir 优先（支持绝对/相对路径），否则 pipeline/models/{default_dir_name}
    """
    if model_dir_arg:
        # 命令行指定优先：相对路径会基于当前工作目录转为绝对路径
        return os.path.abspath(model_dir_arg)
    # 未指定时按各项目默认规则推导：pipeline/models/{default_dir_name}
    return os.path.join(pipeline_dir, "models", default_dir_name)


def resolve_log_prefix(log_name_arg, default_prefix):
    """
    解析日志名前缀

    Args:
        log_name_arg: 命令行 --log_name 的值（空字符串表示未指定）
        default_prefix: 默认前缀（各项目规则不同，由调用方传入）

    Returns:
        日志名前缀；--log_name 优先，否则默认前缀
    """
    return log_name_arg if log_name_arg else default_prefix


def resolve_cache_parent(default_parent, cli_parent=None):
    """
    解析数据缓存父目录

    优先级：命令行 --cache_parent > 环境变量 SRTP_CACHE_PARENT > default_parent（各项目历史默认）

    注意：本仓库存在两份 persistent_cache_2.0（D:\\ 主项目+5基线体系 与
    Essay\\Data LightSegMamba 体系），两份同为 seed=42 但划分实现不同
    （test 集交集仅 21/188），默认路径与各项目历史划分绑定、不可混用；
    命令行参数/环境变量仅用于整机迁移（换机器时统一重定向）。

    Args:
        default_parent: 默认缓存父目录（各项目历史值，由调用方传入）
        cli_parent: 命令行 --cache_parent 的值（空字符串/None 表示未指定）

    Returns:
        缓存父目录路径
    """
    if cli_parent:
        return os.path.abspath(cli_parent)
    return os.environ.get("SRTP_CACHE_PARENT", default_parent)


def resolve_data_root(default_data_root, cli_data_root=None):
    """
    解析原始 BraTS TrainingData 目录

    优先级：命令行 --data_root > 环境变量 BRATS_DATA_ROOT > default_data_root（历史默认）

    Args:
        default_data_root: 默认数据目录（历史硬编码值，由调用方传入）
        cli_data_root: 命令行 --data_root 的值（空字符串/None 表示未指定）

    Returns:
        数据目录路径
    """
    if cli_data_root:
        return os.path.abspath(cli_data_root)
    return os.environ.get("BRATS_DATA_ROOT", default_data_root)


def resolve_cache_dir(resolution, default_parent, cli_parent=None, max_samples=None):
    """
    解析数据缓存目录（统一拼接规则：{父目录}_{分辨率}[_{采样数}]）

    Args:
        resolution: 数据分辨率（如 2.0）
        default_parent: 默认缓存父目录（各项目历史值，由调用方传入）
        cli_parent: 命令行 --cache_parent 的值（空字符串/None 表示未指定）
        max_samples: LightSegMamba 特有；小于 1251 时目录名追加 _{max_samples}
            （如 persistent_cache_2.0_200），None 或 >=1251 为全量目录

    Returns:
        缓存目录路径，形如 {parent}_{resolution} 或 {parent}_{resolution}_{max_samples}
    """
    parent = resolve_cache_parent(default_parent, cli_parent)
    if max_samples is not None and max_samples < 1251:
        return f"{parent}_{resolution}_{max_samples}"
    return f"{parent}_{resolution}"
