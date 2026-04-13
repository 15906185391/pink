# ...existing code...
"""
grpc_msg 子包初始化：容错导入由 protobuf/GRPC 生成的模块并按需导出模块名，
避免在包导入阶段直接抛出 ImportError（安装时或生成文件缺失时可容错）。
"""
import logging

logger = logging.getLogger(__name__)

try:
    from . import xrTracking_pb2  # type: ignore
except Exception as e:
    xrTracking_pb2 = None
    logger.debug("mocap_system.stream.pico.grpc_msg: failed to import xrTracking_pb2: %s", e)

try:
    from . import xrTracking_pb2_grpc  # type: ignore
except Exception as e:
    xrTracking_pb2_grpc = None
    logger.debug("mocap_system.stream.pico.grpc_msg: failed to import xrTracking_pb2_grpc: %s", e)

__all__ = []
if xrTracking_pb2 is not None:
    __all__.append("xrTracking_pb2")
if xrTracking_pb2_grpc is not None:
    __all__.append("xrTracking_pb2_grpc")
# ...existing code...