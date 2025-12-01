import torch
import numpy as np
from torch.autograd import Variable
from tqdm import tqdm
from scipy.spatial.transform import Rotation as R
import matplotlib.pyplot as plt
from .data_processor import align_robot_data

def optimize_human_scale_table(human_data, robot_centers, body_names, ik_config, human_root_name="pelvis", 
                               initial_scales=None, bounds=(0.1, 5.0), max_iter=1000, device='cpu', 
                               plot_loss=True, plot_save_path=None):
    """
    基于梯度下降的人体关节的对称形状拟合优化
    策略：
    - 所有中心关节使用与root相同的缩放系数
    - 左右对称关节分别使用相同的缩放系数
    - 在优化过程中，只使用root和其他非中心关节的匹配对计算损失
    
    Args:
        human_data: 原始人体数据
        robot_centers: 机器人连杆位置列表
        body_names: 机器人连杆名称列表
        ik_config: IK配置字典
        human_root_name: 人体根骨骼名称
        initial_scales: 初始缩放系数
        bounds: 缩放系数边界
        max_iter: 最大迭代次数
        device: 计算设备
        plot_loss: 是否绘制损失曲线
        plot_save_path: 损失曲线保存路径
        
    Returns:
        dict: 优化后的 human_scale_table
    """
    # 构建机器人连杆名称到索引的映射
    robot_link_indices = {}
    for idx, name in enumerate(body_names):
        if not isinstance(name, str):
            name = name.decode("utf-8")
        robot_link_indices[name] = idx
    
    # 确定要优化的缩放键（基于人体关节）
    original_scale_keys = list(ik_config.get("human_scale_table", {}).keys())
    if not original_scale_keys:
        print("[WARN] IK配置中没有 human_scale_table，无法优化")
        return {}
    
    # 设置初始缩放系数
    if initial_scales is None:
        initial_scales = ik_config.get("human_scale_table", {})
    
    for key in original_scale_keys:
        if key not in initial_scales:
            initial_scales[key] = 1.0
    
    # 创建基于人体关节的对称分组
    scale_groups, group_keys = create_human_based_symmetric_groups(original_scale_keys)
    
    # 为每个组创建初始值
    initial_group_scales = {}
    for group_key, human_joints in scale_groups.items():
        # 计算组内人体关节初始缩放系数的平均值
        group_initial_values = [initial_scales[human_joint] for human_joint in human_joints]
        initial_group_scales[group_key] = np.mean(group_initial_values)
    
    # 准备优化变量 - 每个组一个变量
    initial_scales_tensor = torch.tensor([initial_group_scales[key] for key in group_keys], 
                                        dtype=torch.float32, device=device)
    group_scale_params = Variable(initial_scales_tensor, requires_grad=True)
    
    # 准备优化所需的固定数据
    fixed_data = prepare_optimization_data(
        human_data, robot_centers, robot_link_indices, ik_config, human_root_name, 
        original_scale_keys, device, scale_groups
    )
    
    # 创建优化器
    optimizer = torch.optim.Adam([group_scale_params], lr=0.1)
    
    # 记录损失历史
    loss_history = []
    
    print(f"[INFO] 开始基于人体关节的对称形状拟合优化")
    
    # 优化循环
    pbar = tqdm(range(max_iter))
    
    for iteration in pbar:
        optimizer.zero_grad()
        
        # 计算损失
        total_loss = compute_shape_fitting_loss(
            group_scale_params, scale_groups, group_keys, fixed_data
        )
        
        # 记录损失
        loss_history.append(total_loss.item())

        # 反向传播
        total_loss.backward()
        optimizer.step()
        
        # 应用边界约束
        with torch.no_grad():
            group_scale_params.data = torch.clamp(group_scale_params.data, bounds[0], bounds[1])
        
        # 更新进度条
        pbar.set_description(f"Loss: {total_loss.item()*1000:5.1f}")
    
    # 将组缩放系数映射回原始人体关节
    group_scale_params = group_scale_params.detach().cpu().numpy()
    optimized_scales_temp = reconstruct_scale_dict_from_groups(group_scale_params, scale_groups, group_keys)

    # 转换为 Float 类型可导入json
    optimized_scales = {}
    for key in original_scale_keys:
        optimized_scales[key] = float(optimized_scales_temp[key])

    # 绘制损失曲线
    if plot_loss:
        plot_loss_curve(loss_history, save_path=plot_save_path, show_plot=True)
    
    print(f"[INFO] 基于人体关节的对称优化完成:")
    print(f"  - 优化后的缩放系数:")
    for group_key, human_joints in scale_groups.items():
        scale_value = optimized_scales[human_joints[0]]
        print(f"    {group_key}: {scale_value:.4f} -> {human_joints}")
    
    return optimized_scales

def create_human_based_symmetric_groups(human_scale_keys):
    """
    基于人体关节创建对称分组，确保：
    1. 左右对称关节使用相同的缩放系数
    2. 根节点和脚部节点使用相同的缩放系数（保证脚始终在地面上）
    3. 中间节点（非左右对称关节）与根节点使用相同的缩放系数

    SMPL标准关节命名示例:
    - 左侧: 'Left_Hip', 'Left_Knee', 'Left_Ankle', 'Left_Shoulder', 'Left_Elbow', 'Left_Wrist'
    - 右侧: 'Right_Hip', 'Right_Knee', 'Right_Ankle', 'Right_Shoulder', 'Right_Elbow', 'Right_Wrist' 
    - 中心: 'Pelvis', 'Spine1', 'Spine2', 'Spine3', 'Neck', 'Head'
    
    这里我们假设human_scale_keys使用类似的命名
    
    Args:
        human_scale_keys: 人体关节名称列表
        
    Returns:
        scale_groups: 分组字典 {group_key: [joint1, joint2, ...]}
        group_keys: 分组键列表
    """
    # 识别左右对称人体关节
    left_joints = []
    right_joints = []
    center_joints = []
    
    # 特别关注脚部关节
    foot_joints = []
    
    for joint in human_scale_keys:
        joint_lower = joint.lower()
        # 检查是否是脚部关节
        # if 'foot' in joint_lower or 'ankle' in joint_lower:
        #     foot_joints.append(joint)
        if joint_lower.startswith('left_') or 'left' in joint_lower:
            left_joints.append(joint)
        elif joint_lower.startswith('right_') or 'right' in joint_lower:
            right_joints.append(joint)
        else:
            center_joints.append(joint)
    
    # 构建分组
    scale_groups = {}

    # 所有中心关节和根节点使用相同的缩放系数
    # 添加脚部关节到中心组（保证脚始终在地面上）
    # for foot_joint in foot_joints:
    #     if foot_joint not in center_joints:
    #         center_joints.append(foot_joint)
    
    if center_joints:
        scale_groups['center_group'] = center_joints
    
    # 提取对称部位名称
    def extract_body_part(joint_name):
        """从关节名中提取身体部位"""
        name_lower = joint_name.lower()
        # 移除左右前缀
        if name_lower.startswith('left_'):
            return name_lower[5:]
        elif name_lower.startswith('right_'):
            return name_lower[6:]
        elif 'left' in name_lower:
            return name_lower.replace('left', '').strip('_')
        elif 'right' in name_lower:
            return name_lower.replace('right', '').strip('_')
        return joint_name
    
    left_parts = {extract_body_part(joint): joint for joint in left_joints}
    right_parts = {extract_body_part(joint): joint for joint in right_joints}
    
    # 找到对称部位
    symmetric_parts = set(left_parts.keys()) & set(right_parts.keys())
    
    # 为每个对称部位创建分组
    for part in symmetric_parts:
        left_joint = left_parts[part]
        right_joint = right_parts[part]
        group_key = part if part else "symmetric_group"  # 使用部位名称作为组键
        scale_groups[group_key] = [left_joint, right_joint]
    
    # 处理未配对的左右关节
    unpaired_left = [j for j in left_joints if j not in sum(scale_groups.values(), [])]
    unpaired_right = [j for j in right_joints if j not in sum(scale_groups.values(), [])]
    
    for joint in unpaired_left + unpaired_right:
        scale_groups[joint] = [joint]
    
    group_keys = list(scale_groups.keys())
    
    print(f"[DEBUG] 基于人体关节的对称分组完成:")
    for group_key, joints in scale_groups.items():
        print(f"  - {group_key}: {joints}")
    
    return scale_groups, group_keys

def prepare_optimization_data(human_data, robot_centers, robot_link_indices, ik_config, human_root_name, 
                              scale_keys, device, scale_groups):
    """
    准备形状拟合优化所需的固定数据，使用对齐后的人体数据
    
    核心功能：
    1. 将人体数据与机器人根节点对齐
    2. 提取人体骨骼的局部位置（相对于根骨骼）
    3. 从机器人前向运动学结果中提取目标位置
    4. 基于IK配置建立人体骨骼与机器人连杆的对应关系
    
    Args:
        human_data: SMPL格式的人体数据，包含各骨骼的位置和旋转
        robot_centers: 通过前向运动学计算的机器人各连杆位置
        robot_link_indices: 机器人连杆名称到索引的映射
        ik_config: IK配置字典，定义人体骨骼与机器人连杆的对应关系
        human_root_name: 人体根骨骼名称
        scale_keys: 需要优化缩放系数的人体骨骼列表
        device: 计算设备
        scale_groups: 分组字典 {group_key: [joint1, joint2, ...]}
        
    Returns:
        dict: 包含优化所需核心数据的字典
    """
    fixed_data = {}
    
    # ========== 人体数据提取 ==========
    # 获取机器人根节点位置（假设第一个body是根节点）
    robot_root_pos = robot_centers[0]  # 根据实际情况调整索引
    
    # 提取根骨骼位置作为参考坐标系原点
    root_pos = np.array(human_data[human_root_name][0], dtype=np.float32)
    fixed_data['root_pos'] = torch.tensor(root_pos, device=device)
    fixed_data['human_root_name'] = human_root_name

    # 将机器人位置移动到根节点与人体数据对齐
    robot_centers=align_robot_data(robot_centers, robot_root_pos, root_pos)
    
    # 计算各骨骼相对于根的局部位置
    human_local_positions = {}
    for body_name, (global_pos, _) in human_data.items():
        if body_name not in scale_keys:
            continue
        local_pos = np.array(global_pos, dtype=np.float32) - root_pos
        human_local_positions[body_name] = local_pos
    
    fixed_data['human_local_positions'] = human_local_positions
    
    # ========== 机器人目标位置提取 ==========
    # 基于IK配置表，建立人体骨骼到机器人连杆的映射
    robot_positions = {}

    # 获取中心关节组中的所有关节（用于过滤）
    center_joints = []
    if 'center_group' in scale_groups:
        center_joints = scale_groups['center_group']
    
    # 记录被忽略的关节
    ignored_joints = []

    # 遍历两个IK匹配表
    for table_key in ["ik_match_table1", "ik_match_table2"]:
        if table_key not in ik_config:
            continue
            
        # 处理每个匹配条目：机器人连杆 -> [人体骨骼名称, ...]
        for robot_link, config in ik_config[table_key].items():
            if not isinstance(config, list) or len(config) < 1:
                continue
                
            human_bone_name = config[0]  # 对应的人体骨骼名称
            
            # 检查这个匹配对是否有效且需要优化
            if (robot_link in robot_link_indices and human_bone_name in scale_keys):
                # 从FK结果中获取机器人连杆位置
                robot_idx = robot_link_indices[robot_link]
                robot_pos = robot_centers[robot_idx]
                # 存储机器人目标位置
                robot_positions[(robot_link, human_bone_name)] = robot_pos
    
    fixed_data['robot_positions'] = robot_positions
    
    return fixed_data

def reconstruct_scale_dict_from_groups(group_scale_params, scale_groups, group_keys):
    """
    将组缩放系数重建为完整的人体关节缩放字典

    Args:
        group_scale_params: 分组缩放系数数组
        scale_groups: 分组字典
        group_keys: 分组键列表
        human_root_name: 根关节名称
        
    Returns:
        dict: 完整的人体关节缩放字典
    """
    # 将分组缩放参数转换为字典格式 {group_key: scale_value}
    group_scale_dict = dict(zip(group_keys, group_scale_params))

    # 为每个人体关节分配对应的缩放系数，同一分组内的关节共享相同的缩放系数（确保对称性）
    scale_dict = {}
    for group_key, human_joints in scale_groups.items():
        scale_value = group_scale_dict[group_key]
        for human_joint in human_joints:
            scale_dict[human_joint] = scale_value
    
    return scale_dict

def compute_shape_fitting_loss(group_scale_params, scale_groups, group_keys, fixed_data):
    """
    计算形状拟合的损失函数
    
    Args:
        group_scale_params: 分组缩放系数张量 [num_groups]
        scale_groups: 分组字典 {group_key: [human_joint1, human_joint2, ...]}
        group_keys: 分组键列表 [group_key1, group_key2, ...]
        fixed_data: 包含优化所需固定数据的字典
        
    Returns:
        torch.Tensor: 平均位置误差损失值
    """
    # ========== 步骤1: 重建完整的人体关节缩放字典 ==========
    scale_dict = reconstruct_scale_dict_from_groups(group_scale_params, scale_groups, group_keys)
    
    # ========== 步骤2: 计算位置误差 ==========
    # 从固定数据中提取必要信息
    root_pos = fixed_data['root_pos']                    # 人体根位置 [3]
    human_root_name = fixed_data['human_root_name']      # 人体根骨骼名称[str]
    human_local_positions = fixed_data['human_local_positions']  # 人体骨骼局部位置 {joint: [x,y,z]}
    robot_positions = fixed_data['robot_positions']      # 机器人连杆位置 {(robot_link, human_bone): [x,y,z]}
    
    total_error = 0.0    # 累计误差
    error_count = 0      # 有效匹配对数量

    # 首先计算缩放后的根节点位置
    root_scale = scale_dict.get(human_root_name, 1.0)
    scaled_root_pos = root_pos * root_scale

    # 遍历所有机器人连杆与人体骨骼的匹配对
    for (robot_link, human_bone), robot_pos in robot_positions.items():
        # 跳过没有缩放系数的人体骨骼与根节点
        if human_bone not in scale_dict or human_bone == human_root_name:
        # if human_bone not in scale_dict:
            continue
            
        # 获取当前人体骨骼的缩放系数
        scale = scale_dict[human_bone]
        
        # 计算缩放后的人体骨骼全局位置:
        if human_bone == human_root_name:
            # 根节点：直接缩放
            human_global_pos = scaled_root_pos
        else:
            # 其他节点：人体全局位置 = 缩放后的根节点位置 + (局部位置 × 缩放系数)
            # 1. 获取人体骨骼相对于根的局部位置
            local_pos = torch.tensor(human_local_positions[human_bone], 
                                    device=scale.device, dtype=torch.float32)
            # 2. 应用缩放系数，转换回全局坐标系
            human_global_pos = scaled_root_pos + local_pos * scale
        
        # 计算位置误差
        robot_pos_tensor = torch.tensor(robot_pos, device=scale.device, dtype=torch.float32)
        error = torch.norm(human_global_pos - robot_pos_tensor)
        
        total_error += error
        error_count += 1
    
    # ========== 步骤3: 返回平均误差 ==========
    # 如果有有效匹配对，返回平均误差；否则返回0
    if error_count > 0:
        return total_error / error_count  # 平均每个匹配对的误差
    else:
        return torch.tensor(0.0, device=group_scale_params.device)

def plot_loss_curve(loss_history, save_path=None, show_plot=True):
    """
    绘制损失函数曲线
    
    Args:
        loss_history: 损失值历史记录列表
        save_path: 图片保存路径，如果为None则不保存
        show_plot: 是否显示图片
    """
    plt.figure(figsize=(10, 6))
    plt.plot(loss_history, 'b-', linewidth=2, label='Loss')
    plt.xlabel('Iteration')
    plt.ylabel('Loss')
    plt.title('Shape Fitting Optimization Loss Curve')
    plt.grid(True, alpha=0.3)
    plt.legend()
    
    # 添加收敛分析
    if len(loss_history) > 10:
        final_loss = loss_history[-1]
        initial_loss = loss_history[0]
        improvement = initial_loss - final_loss
        improvement_ratio = improvement / initial_loss * 100
        
        plt.text(0.02, 0.98, f'Initial Loss: {initial_loss:.4f}\nFinal Loss: {final_loss:.4f}\nImprovement: {improvement_ratio:.1f}%', 
                transform=plt.gca().transAxes, verticalalignment='top',
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"[INFO] 损失曲线已保存到: {save_path}")
    
    if show_plot:
        plt.show()
    else:
        plt.close()
