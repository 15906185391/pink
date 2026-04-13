import numpy as np
import pinocchio as pin
import yaml

# ========== 接口基类 ==========
class Interface:
    """通用接口基类"""
    def __init__(self, name, submodel_name):
        self.name = name
        self.submodel_name = submodel_name


class JointStateInterface(Interface):
    """关节状态接口"""
    def __init__(self, name, submodel_name, dof):
        super().__init__(name, submodel_name)
        self.qpos = [0.0] * dof
        self.qvel = [0.0] * dof

    def update(self, qpos, qvel=None):
        self.qpos = qpos
        if qvel is not None:
            self.qvel = qvel

    def get_data(self):
        return self.qpos, self.qvel


class PoseInterface(Interface):
    """末端或子模型位姿接口"""
    def __init__(self, name, submodel_name):
        super().__init__(name, submodel_name)
        self.pose_matrix = np.eye(4)

    def update(self, pose_matrix):
        self.pose_matrix = np.array(pose_matrix).reshape(4, 4)

    def get_data(self):
        return self.pose_matrix


# ========== SubModel ==========
class SubModel:
    """子模型（机器人子系统，例如左臂、右臂、头、腰）"""
    def __init__(self, name, urdf_path=None, yaml_path=None):
        self.name = name
        self.interfaces = []
        
        if urdf_path is not None:
            self.urdf_path = urdf_path
            self.model = pin.buildModelFromUrdf(urdf_path)
            self.data = self.model.createData()

        if yaml_path is not None:
            self.yaml_path = yaml_path
            with open(yaml_path, 'r') as f:
                data = yaml.safe_load(f)
            
            for key, value in data.items():
                if isinstance(value, dict) and 'value' in value:
                    raw_val = value['value']

                    if isinstance(raw_val, (list, tuple)):
                        processed_val = np.array(raw_val)
                    else:
                        processed_val = raw_val  # 保持原样（float, str 等）

                    setattr(self, key, processed_val)

                else:
                    setattr(self, key, value)


    def add_interface(self, interface):
        self.interfaces.append(interface)

    def get_interfaces(self, name):
        return next((sm for sm in self.interfaces if sm.name == name), None)
    
    def fk(self, q, frame_name):
        pin.forwardKinematics(self.model, self.data, q)
        pin.updateFramePlacement(self.model, self.data, self.model.getFrameId(frame_name))

        return self.data.oMf[self.model.getFrameId(frame_name)]


# ========== Robot 顶层 ==========
class Robot:
    """机器人，由多个 SubModel 组成"""
    def __init__(self, name, urdf_path=None):
        self.name = name
        self.submodels = []

        if urdf_path is not None:
            self.urdf_path = urdf_path
            self.model = pin.buildModelFromUrdf(urdf_path)
            self.data = self.model.createData()

    def add_submodel(self, submodel):
        self.submodels.append(submodel)

    def get_submodels(self):
        return self.submodels

    def get_submodel(self, name):
        return next((sm for sm in self.submodels if sm.name == name), None)


if __name__ == "__main__":
    import os

    current_path = os.path.dirname(__file__)
    urdf_path_l = os.path.join(current_path, "models/l_arm_v1.urdf")
    urdf_path_r = os.path.join(current_path, "models/r_arm_v1.urdf")
    yaml_path = os.path.join(current_path, "config/Gen1.yaml")

    left_arm = SubModel("LeftArm", urdf_path_l, yaml_path)
    right_arm = SubModel("RightArm", urdf_path_r, yaml_path)

    print(right_arm.fk(right_arm.ReadyPos_r, 'link_tcp_r'))
