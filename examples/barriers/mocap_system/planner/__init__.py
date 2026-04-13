# ...existing code...
"""
planner 包初始化：自动发现并容错导入同目录下的子模块/子包。
"""
import logging
import pkgutil
import importlib

logger = logging.getLogger(__name__)

# 动态收集并导入同目录下的子模块/子包，导入失败只做调试级别日志
__all__ = []
for finder, name, ispkg in pkgutil.iter_modules(__path__):
    try:
        module = importlib.import_module('.' + name, __name__)
        globals()[name] = module
        __all__.append(name)
    except Exception as e:
        logger.debug("Failed to import %s.%s: %s", __name__, name, e)

# 便于外部直接 from mocap_system.planner import <submodule>
# 如果需要将子模块中的特定类/函数直接暴露，建议在各自子模块中导出并在此处显式导入。
# ...existing code...