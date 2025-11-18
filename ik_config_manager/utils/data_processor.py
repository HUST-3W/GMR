import os
import json
import numpy as np
from scipy.spatial.transform import Rotation as R

def load_robot_init(path: str):
    """
    加载机器人初始姿态配置（仅支持特定JSON格式）
    
    支持的JSON格式：
    {
      "root_pos": [x, y, z],
      "root_rot": [w, x, y, z] (wxyz格式),
      "degrees": true/false,
      "joints": {
        "joint_name1": value,
        "joint_name2": value,
        ...
      },
      "joint_match": {
        "robot_joint1": "human_joint1",
        "robot_joint2": "human_joint2",
        ...
      }
    }
    
    Args:
        path: JSON配置文件路径
        
    Returns:
        tuple: (root_pos, root_rot, joints_dict)
        - root_pos: 根位置 (3,) float32
        - root_rot: 根旋转四元数 (4,) float64 (wxyz格式，已归一化)
        - joints_dict: 关节名称到值的字典 (弧度制)
        - joint_match: 机器人关节到人体关节名称的映射字典
    """
    try:
        with open(path, 'r', encoding='utf-8') as f:
            config = json.load(f)
        
        # 提取根位置和旋转
        root_pos = np.array(config['root_pos'], dtype=np.float32).reshape(3)
        root_rot = np.array(config['root_rot'], dtype=np.float64).reshape(4)
        
        # 归一化四元数
        root_rot_norm = np.linalg.norm(root_rot)
        if root_rot_norm > 1e-12:
            root_rot = root_rot / root_rot_norm
        
        # 提取关节配置
        joints_dict = config['joints'].copy()
        
        # 处理角度单位转换
        if config.get('degrees', False):
            # 将度数转换为弧度
            for joint_name, value in joints_dict.items():
                joints_dict[joint_name] = np.deg2rad(float(value))
        else:
            # 已经是弧度，直接转换为float
            for joint_name, value in joints_dict.items():
                joints_dict[joint_name] = float(value)

        # 提取关节映射
        joint_match = config.get('joint_match', {})
        
        return root_pos, root_rot, joints_dict, joint_match
        
    except Exception as e:
        print(f"[ERROR] 加载机器人配置失败: {e}")
        # 返回默认配置
        default_root_pos = np.array([0.0, 0.0, 0.92], dtype=np.float32)
        default_root_rot = np.array([0.70710678, 0.0, 0.0, -0.70710678], dtype=np.float64)
        default_joints = {}
        default_joint_match = {}
        
        return default_root_pos, default_root_rot, default_joints, default_joint_match

def write_all_data_to_ik(ik_config_path: str, output_path: str, human_scale_table: dict = None,
                         pos_offsets: dict = None, quat_offsets: dict = None) -> str:
    """
    将对齐后的所有数据写入IK配置文件
    支持格式：
    {
      "human_scale_table": {
        "human_bone": scale_factor,
        ...
      },
      "ik_match_table1": {
        "robot_link": [human_bone, w_pos, w_rot, [dx,dy,dz], [w,x,y,z]],
        ...
      },
      "ik_match_table2": {
        "robot_link": [human_bone, w_pos, w_rot, [dx,dy,dz], [w,x,y,z]], 
        ...
      }
    }
    包括：
    - human_scale_table: 优化后的人体缩放系数
    - pos_offsets: 位置偏移
    - quat_offsets: 四元数偏移
    
    Args:
        ik_config_path: 输入IK配置路径
        output_path: 输出配置路径
        human_scale_table: 优化后的人体缩放系数字典 {human_bone: scale_factor}
        pos_offset: 位置偏移字典 {robot_link: [dx, dy, dz]}
        quat_offset: 四元数偏移字典 {robot_link: [w, x, y, z]}
        
    Returns:
        输出文件路径，如果失败则返回None
    """
    try:
        # 读取原始配置
        with open(ik_config_path, 'r', encoding='utf-8') as f:
            config = json.load(f)
        
        # 验证配置格式
        if "ik_match_table1" not in config or "ik_match_table2" not in config:
            print("[ERROR] IK配置必须同时包含 ik_match_table1 和 ik_match_table2")
            return None
        
        updated_count = 0
        
        # 1. 更新 human_scale_table
        if human_scale_table is not None:
            if "human_scale_table" in config:
                config["human_scale_table"] = human_scale_table
                updated_count += 1
                print(f"[INFO] 更新了 human_scale_table，包含 {len(human_scale_table)} 个缩放系数")
        
        # 2. 更新位置偏移
        if pos_offsets is not None:
            for table_key in ["ik_match_table1", "ik_match_table2"]:
                table = config[table_key]
                for robot_link, pos_offset in pos_offsets.items():
                    if robot_link in table:
                        entry = table[robot_link]
                        if isinstance(entry, list) and len(entry) >= 4:
                            entry[3] = [float(x) for x in pos_offset]
                            updated_count += 1
        
        # 3. 更新四元数偏移
        if quat_offsets is not None:
            for table_key in ["ik_match_table1", "ik_match_table2"]:
                table = config[table_key]
                for robot_link, quat_offset in quat_offsets.items():
                    if robot_link in table:
                        entry = table[robot_link]
                        if isinstance(entry, list) and len(entry) >= 5:
                            q = np.asarray(quat_offset, dtype=np.float64).reshape(4)
                            q_norm = q / max(np.linalg.norm(q), 1e-12)
                            entry[4] = [float(q_norm[0]), float(q_norm[1]), float(q_norm[2]), float(q_norm[3])]
                            updated_count += 1
        
        # 输出更新统计
        print(f"[INFO] 总共更新了 {updated_count} 个配置项")
        
        # 确定输出路径
        if not output_path:
            base_name = os.path.splitext(ik_config_path)[0]
            output_path = f"{base_name}_auto.json"
        
        # 确保输出目录存在
        output_dir = os.path.dirname(output_path)
        if output_dir and not os.path.exists(output_dir):
            os.makedirs(output_dir, exist_ok=True)
        
        # 写入更新后的配置
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(config, f, indent=2, ensure_ascii=False)
        
        print(f"[SUCCESS] 完整优化配置已保存 -> {output_path}")
        return output_path
        
    except Exception as e:
        print(f"[ERROR] 写入完整优化配置失败: {e}")
        import traceback
        traceback.print_exc()
        return None

def align_robot_data(robot_centers, robot_root_pos, target_root_pos):
    """
    将机器人数据移动到目标根节点位置
    
    Args:
        robot_centers: 机器人各连杆的位置列表
        robot_root_pos: 机器人当前的根节点位置
        target_root_pos: 目标根节点位置
        
    Returns:
        aligned_robot_centers: 对齐后的机器人连杆位置
    """
    # 计算位置偏移
    pos_offset = target_root_pos - robot_root_pos
    
    # 对机器人所有连杆位置应用位置偏移
    aligned_robot_centers = []
    for center in robot_centers:
        aligned_center = center + pos_offset
        aligned_robot_centers.append(aligned_center)
    
    return aligned_robot_centers

def scale_human_data(human_data, human_root_name, human_scale_table):
    """人体数据缩放"""
    human_data_local = {}
    root_pos, root_quat = human_data[human_root_name]
    
    # scale root
    scaled_root_pos = human_scale_table[human_root_name] * root_pos
    
    # scale other body parts in local frame
    for body_name in human_data.keys():
        if body_name not in human_scale_table:
            continue
        if body_name == human_root_name:
            continue
        else:
            # transform to local frame (only position)
            human_data_local[body_name] = (human_data[body_name][0] - root_pos) * human_scale_table[body_name]
        
    # transform the human data back to the global frame
    human_data_global = {human_root_name: (scaled_root_pos, root_quat)}
    for body_name in human_data_local.keys():
        human_data_global[body_name] = (human_data_local[body_name] + scaled_root_pos, human_data[body_name][1])

    return human_data_global

def offset_human_data(human_data, pos_offsets, rot_offsets):
    """the pos offsets are applied in the local frame"""
    offset_human_data = {}
    for body_name in human_data.keys():
        pos, quat = human_data[body_name]
        offset_human_data[body_name] = [pos, quat]
        # apply rotation offset first
        updated_quat = (R.from_quat(quat, scalar_first=True) * rot_offsets[body_name]).as_quat(scalar_first=True)
        offset_human_data[body_name][1] = updated_quat
        local_offset = pos_offsets[body_name]
        # compute the global position offset using the updated rotation
        global_pos_offset = R.from_quat(updated_quat, scalar_first=True).apply(local_offset)
        
        offset_human_data[body_name][0] = pos + global_pos_offset
        
    return offset_human_data