# ...existing code...
"""
config package 初始化。
对外导出 Config、parse_bool，并提供辅助加载函数。
"""
from .config_info import Config, parse_bool  # 引入包内实现

__all__ = [
    "Config",
    "parse_bool",
]

__version__ = "0.0.1"

def load_config(path: str | None = None) -> Config:
    """
    便捷函数：创建 Config 并在提供 path 时读取配置文件。
    使用示例：
        cfg = load_config("/path/to/config.ini")
    """
    cfg = Config()
    if path:
        cfg.read_config_file(path)
    return cfg
# ...existing code...