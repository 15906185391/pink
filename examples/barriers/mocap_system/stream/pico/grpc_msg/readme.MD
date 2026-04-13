rm -rf playcontroller_pb2*.py __pycache__/
python3 -m grpc_tools.protoc -I . --python_out=. --grpc_python_out=. ./xrTracking.proto