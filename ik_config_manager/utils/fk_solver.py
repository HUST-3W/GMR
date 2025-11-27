import mujoco
import numpy as np
import xml.etree.ElementTree as ETree
import torch
import copy
from io import BytesIO
from lxml.etree import XMLParser, parse
from collections import OrderedDict


class MuJoCoFK:
    """
    通过 MuJoCo 解析机器人 XML，获取所有链接信息，用于前向运动学计算各 link 的姿态
    """
    def __init__(self, asset_file: str, device=torch.device("cpu")):
        self.mjcf_file = asset_file
        self.model = mujoco.MjModel.from_xml_path(self.mjcf_file)
        self.data  = mujoco.MjData(self.model)

        parser = XMLParser(remove_blank_text=True)
        tree = parse(
            BytesIO(open(self.mjcf_file, "rb").read()),
            parser=parser,
        )
        self.dof_axis = []
        joints = sorted(
            [
                j.attrib["name"]
                for j in tree.getroot().find("worldbody").findall(".//joint")
            ]
        )
        motors = sorted(
            [m.attrib["name"] for m in tree.getroot().find("actuator").getchildren()]
        )
        assert len(motors) > 0, "No motors found in the mjcf file"

        self.num_dof = len(motors)
        self.num_extend_dof = self.num_dof

        self.mjcf_data = mjcf_data = self.from_mjcf(self.mjcf_file)
        self.body_names = copy.deepcopy(mjcf_data["node_names"])
        self._parents = mjcf_data["parent_indices"]
        self._proper_kinematic_structure = copy.deepcopy(mjcf_data["node_names"])
        self._offsets = mjcf_data["local_translation"][None,].to(device)
        self._local_rotation = mjcf_data["local_rotation"][None,].to(device)
        self.actuated_joints_idx = np.array(
            [self.body_names.index(k) for k, v in mjcf_data["body_to_joint"].items()]
        )

        for m in motors:
            if not m in joints:
                print(m)

        if (
            "type" in tree.getroot().find("worldbody").findall(".//joint")[0].attrib
            and tree.getroot().find("worldbody").findall(".//joint")[0].attrib["type"]
            == "free"
        ):
            for j in tree.getroot().find("worldbody").findall(".//joint")[1:]:
                self.dof_axis.append([int(i) for i in j.attrib["axis"].split(" ")])
            self.has_freejoint = True
        elif (
            not "type" in tree.getroot().find("worldbody").findall(".//joint")[0].attrib
        ):
            for j in tree.getroot().find("worldbody").findall(".//joint"):
                self.dof_axis.append([int(i) for i in j.attrib["axis"].split(" ")])
            self.has_freejoint = True
        else:
            for j in tree.getroot().find("worldbody").findall(".//joint")[6:]:
                self.dof_axis.append([int(i) for i in j.attrib["axis"].split(" ")])
            self.has_freejoint = False

        self.dof_axis = torch.tensor(self.dof_axis)
        self.num_bodies = len(self.body_names)
        # 只保留可见刚体：排除 root(id=0)，并保证顺序固定
        self.body_ids = [i for i in range(self.num_bodies) if i != 0]

        self.joints_range = mjcf_data["joints_range"].to(device)
        # 添加关节顺序列表
        self.joint_order = self.get_joint_order()
    
    def from_mjcf(self, path):
        # function from Poselib:
        tree = ETree.parse(path)
        xml_doc_root = tree.getroot()
        xml_world_body = xml_doc_root.find("worldbody")
        if xml_world_body is None:
            raise ValueError("MJCF parsed incorrectly please verify it.")
        # assume this is the root
        xml_body_root = xml_world_body.find("body")
        if xml_body_root is None:
            raise ValueError("MJCF parsed incorrectly please verify it.")

        xml_joint_root = xml_body_root.find("joint")

        node_names = []
        parent_indices = []
        local_translation = []
        local_rotation = []
        joints_range = []
        body_to_joint = OrderedDict()

        # recursively adding all nodes into the skel_tree
        def _add_xml_node(xml_node, parent_index, node_index):
            node_name = xml_node.attrib.get("name")
            # parse the local translation into float list
            pos = np.fromstring(
                xml_node.attrib.get("pos", "0 0 0"), dtype=float, sep=" "
            )
            quat = np.fromstring(
                xml_node.attrib.get("quat", "1 0 0 0"), dtype=float, sep=" "
            )
            node_names.append(node_name)
            parent_indices.append(parent_index)
            local_translation.append(pos)
            local_rotation.append(quat)
            curr_index = node_index
            node_index += 1
            all_joints = xml_node.findall(
                "joint"
            )  # joints need to remove the first 6 joints
            if len(all_joints) == 6:
                all_joints = all_joints[6:]

            for joint in all_joints:
                if not joint.attrib.get("range") is None:
                    joints_range.append(
                        np.fromstring(joint.attrib.get("range"), dtype=float, sep=" ")
                    )
                else:
                    if not joint.attrib.get("type") == "free":
                        joints_range.append([-np.pi, np.pi])
            for joint_node in xml_node.findall("joint"):
                body_to_joint[node_name] = joint_node.attrib.get("name")

            for next_node in xml_node.findall("body"):
                node_index = _add_xml_node(next_node, curr_index, node_index)

            return node_index

        _add_xml_node(xml_body_root, -1, 0)
        assert len(joints_range) == self.num_dof
        return {
            "node_names": node_names,
            "parent_indices": torch.from_numpy(
                np.array(parent_indices, dtype=np.int32)
            ),
            "local_translation": torch.from_numpy(
                np.array(local_translation, dtype=np.float32)
            ),
            "local_rotation": torch.from_numpy(
                np.array(local_rotation, dtype=np.float32)
            ),
            "joints_range": torch.from_numpy(np.array(joints_range)),
            "body_to_joint": body_to_joint,
        }
    
    def get_joint_order(self):
        """获取求解器的关节顺序列表"""
        joint_order = []
        
        # 从执行器关节索引获取关节名称
        for idx in self.actuated_joints_idx:
            if idx < len(self.body_names):
                body_name = self.body_names[idx]
                # 从身体到关节的映射中获取关节名称
                if body_name in self.mjcf_data["body_to_joint"]:
                    joint_name = self.mjcf_data["body_to_joint"][body_name]
                    joint_order.append(joint_name)
                else:
                    # 如果没有找到映射，使用身体名称
                    joint_order.append(body_name)
        
        return joint_order
    
    def get_specific_body_positions(self, qpos_full: np.ndarray, body_names: list):
        """
        获取指定身体部位的位置
        
        Args:
        qpos_full: 完整的机器人姿态
        body_names: 需要获取位置的身体名称列表
        
        Return:
        positions: 指定身体部位的3D位置数组
        rotations: 指定身体部位的旋转矩阵数组
        """
        self.data.qpos[:] = qpos_full
        mujoco.mj_forward(self.model, self.data)
        
        positions = []
        rotations = []
        
        for body_name in body_names:
            try:
                body_id = self.model.body(body_name).id
                pos = np.array(self.data.xpos[body_id], dtype=np.float32)
                rot = np.array(self.data.xmat[body_id], dtype=np.float64).reshape(3, 3)
                positions.append(pos)
                rotations.append(rot)
            except:
                print(f"[Warning] Body '{body_name}' not found in the model")
                # 添加默认位置
                positions.append(np.array([0, 0, 0], dtype=np.float32))
                rotations.append(np.eye(3, dtype=np.float64))
        
        return np.array(positions), np.array(rotations)
