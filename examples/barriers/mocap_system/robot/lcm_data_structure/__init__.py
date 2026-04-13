# ...existing code...
# -*- coding: utf-8 -*-
"""
lcm_data_structure 包初始化：容错导入自动生成的 LCM 类型模块并按需导出常用类型。
"""
import logging

logger = logging.getLogger(__name__)

# 尝试导入自动生成的 LCM 类型（失败时只记录日志，不抛出异常）
try:
    from .upper_body_cmd_package import upper_body_cmd_package  # noqa: F401
except Exception as e:
    upper_body_cmd_package = None
    logger.debug("Failed to import upper_body_cmd_package: %s", e)

try:
    from .upper_body_data_package import upper_body_data_package  # noqa: F401
except Exception as e:
    upper_body_data_package = None
    logger.debug("Failed to import upper_body_data_package: %s", e)

try:
    from .t12_command_response import t12_command_response  # noqa: F401
except Exception as e:
    t12_command_response = None
    logger.debug("Failed to import t12_command_response: %s", e)

try:
    from .lcm_command_struct import lcm_command_struct  # noqa: F401
except Exception as e:
    lcm_command_struct = None
    logger.debug("Failed to import lcm_command_struct: %s", e)

try:
    from .lcm_response_lcmt import lcm_response_lcmt  # noqa: F401
except Exception as e:
    lcm_response_lcmt = None
    logger.debug("Failed to import lcm_response_lcmt: %s", e)
    
try:
    from .left_tracker_pos import left_tracker_pos  # noqa: F401
except Exception as e:
    left_tracker_pos = None
    logger.debug("Failed to import left_tracker_pos: %s", e)

# 根据实际可用性导出符号，便于外部 from mocap_system.robot.lcm_data_structure import <Type>
__all__ = []
if upper_body_cmd_package is not None:
    __all__.append("upper_body_cmd_package")
if upper_body_data_package is not None:
    __all__.append("upper_body_data_package")
if t12_command_response is not None:
    __all__.append("t12_command_response")
if lcm_command_struct is not None:
    __all__.append("lcm_command_struct")
if lcm_response_lcmt is not None:
    __all__.append("lcm_response_lcmt")
if left_tracker_pos is not None:
    __all__.append("left_tracker_pos")
# ...existing code...