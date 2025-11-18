import numpy as np
from scipy.spatial.transform import Rotation as R

# ===================== 计算四元数偏移 =====================
def compute_quaternion_offsets(human_data, centers, Rs, body_names, ik_config):
    """
    基于offset_human_data逻辑反推rot_offsets
    计算四元数偏移：基于人体根骨骼的统一参考坐标系
    
    Args:
        centers: 机器人连杆在世界坐标系中的位置列表
        Rs: 机器人连杆在世界坐标系中的旋转矩阵列表
        body_names: 机器人连杆名称列表
        scaled_human_data: 缩放后的人体数据
        ik_config: IK配置字典
        root_qoffset_quat: 根的四元数偏移 [w, x, y, z]
        human_root_name: 人体根骨骼名称
        
    Returns:
        tuple: (quat_offsets, missing_links)
        - quat_offsets: 四元数偏移字典 {robot_link: [w, x, y, z]}
        - missing_links: 未找到的机器人连杆列表
    """
    quat_offsets = {}
    missing_links = []
    
    # 构建机器人连杆名称到索引的映射
    robot_link_indices = {}
    for idx, name in enumerate(body_names):
        if not isinstance(name, str):
            name = name.decode("utf-8")
        robot_link_indices[name] = idx
    
    # 遍历两个IK匹配表
    for table_key in ["ik_match_table1", "ik_match_table2"]:
        if table_key not in ik_config:
            continue
            
        table = ik_config[table_key]
        for robot_link, config in table.items():
            if not isinstance(config, list) or len(config) < 1:
                continue
                
            human_bone_name = config[0]  # 对应的人体骨骼名称
            
            # 检查机器人连杆和人体骨骼是否存在
            if (robot_link not in robot_link_indices or 
                human_bone_name not in human_data):
                print(f"[WARN] 无法找到 {robot_link} 或 {human_bone_name}")
                missing_links.append(robot_link)
                continue
            
            # 获取人体骨骼四元数
            human_quat = np.array(human_data[human_bone_name][1], dtype=np.float64)
            
            # 获取机器人连杆旋转矩阵并转换为四元数
            robot_idx = robot_link_indices[robot_link]
            robot_R = Rs[robot_idx]
            robot_rot = R.from_matrix(robot_R)
            
            # 计算旋转偏移: rot_offset = human_rot.inv() * robot_rot
            human_rot = R.from_quat(human_quat, scalar_first=True)
            rot_offset = human_rot.inv() * robot_rot
            
            # 获取四元数偏移
            rot_offset_quat = rot_offset.as_quat(scalar_first=True)
            
            quat_offsets[robot_link] = rot_offset_quat.tolist()
            print(f"  - {robot_link}: 四元数偏移 {rot_offset_quat}")
    
    return quat_offsets, missing_links

# ===================== 计算位置偏移 =====================
def compute_position_offsets(human_data, centers, Rs, body_names, ik_config, rot_offsets):
    """
    基于offset_human_data逻辑反推pos_offsets
    计算位置偏移：在人体骨骼局部坐标系中，从人体骨骼指向机器人连杆的向量
    
    Args:
        centers: 机器人连杆在世界坐标系中的位置列表
        Rs: 机器人连杆在世界坐标系中的旋转矩阵列表
        body_names: 机器人连杆名称列表
        scaled_human_data: 缩放后的人体数据
        ik_config: IK配置字典
        rot_offsets: 四元数偏移字典 {robot_link: [w, x, y, z]}
        
    Returns:
        tuple: (pos_offsets, missing_links)
        - pos_offsets: 位置偏移字典 {robot_link: [dx, dy, dz]}
        - missing_links: 未找到的机器人连杆列表
    """
    pos_offsets = {}
    missing_links = []
    
    # 构建机器人连杆名称到索引的映射
    robot_link_indices = {}
    for idx, name in enumerate(body_names):
        if not isinstance(name, str):
            name = name.decode("utf-8")
        robot_link_indices[name] = idx
    
    # 遍历两个IK匹配表
    for table_key in ["ik_match_table1", "ik_match_table2"]:
        if table_key not in ik_config:
            continue
            
        table = ik_config[table_key]
        for robot_link, config in table.items():
            if not isinstance(config, list) or len(config) < 1:
                continue
                
            human_bone_name = config[0]  # 对应的人体骨骼名称
            
            # 检查机器人连杆和人体骨骼是否存在，以及是否已计算四元数偏移
            if (robot_link not in robot_link_indices or human_bone_name not in human_data or
                robot_link not in rot_offsets):
                print(f"[WARN] 无法找到 {robot_link} 或 {human_bone_name} 或缺少四元数偏移")
                missing_links.append(robot_link)
                continue
            
            # 获取人体骨骼位置和四元数
            human_pos = np.array(human_data[human_bone_name][0], dtype=np.float64)
            human_quat = np.array(human_data[human_bone_name][1], dtype=np.float64)
            
            # 获取机器人连杆位置（已对齐到人体根节点）
            robot_idx = robot_link_indices[robot_link]
            robot_pos = centers[robot_idx]
            
            # 获取该身体部位的四元数偏移
            rot_offset_quat = np.array(rot_offsets[robot_link], dtype=np.float64)
            
            # 计算更新后的人体四元数
            human_rot = R.from_quat(human_quat, scalar_first=True)
            rot_offset = R.from_quat(rot_offset_quat, scalar_first=True)
            updated_rot = human_rot * rot_offset

            # 计算全局位置偏移
            global_pos_offset = robot_pos - human_pos
            
            # 反推pos_offsets
            pos_offset = updated_rot.inv().apply(global_pos_offset)
            pos_offsets[robot_link] = pos_offset.tolist()
            print(f"  - {robot_link}:  位置偏移 = {pos_offset}")
    
    return pos_offsets, missing_links