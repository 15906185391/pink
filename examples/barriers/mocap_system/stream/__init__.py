# ...existing code...
"""
stream 包初始化：以容错方式导入子包/常用类型，避免在包导入阶段直接抛出 ImportError。
"""
import logging

logger = logging.getLogger(__name__)

try:
    # 使用相对导入，减少安装路径上下文相关问题
    from . import pico  # noqa: F401
except Exception as e:
    pico = None
    logger.debug("Failed to import mocap_system.stream.pico: %s", e)

try:
    from .pico.streamer import Pico4UltraStreamer  # type: ignore
except Exception as e:
    Pico4UltraStreamer = None
    logger.debug("Failed to import Pico4UltraStreamer: %s", e)

__all__ = []
if pico is not None:
    __all__.append("pico")
if Pico4UltraStreamer is not None:
    __all__.append("Pico4UltraStreamer")
# ...existing code...