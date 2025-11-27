import argparse
import pathlib
import os
import time
import json

import numpy as np
from scipy.spatial.transform import Rotation as R

from general_motion_retargeting import GeneralMotionRetargeting as GMR
from general_motion_retargeting import RobotMotionViewer
from general_motion_retargeting.utils.smpl import (
    load_smplx_file,
    get_smplx_data_offline_fast,
)

from rich import print

from utils.optimize_human_scale import optimize_human_scale_table
from utils.fk_solver import MuJoCoFK
from utils.data_processor import load_robot_init, scale_human_data, align_robot_data, offset_human_data, write_all_data_to_ik
from utils.compute_offsets import compute_position_offsets, compute_quaternion_offsets



if __name__ == "__main__":
    HERE = pathlib.Path(__file__).parent

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--smplx_file",
        help="SMPLX motion file to load.",
        type=str,
        required=True,
    )
    
    parser.add_argument(
        "--robot",
        choices=["unitree_g1", "unitree_g1_with_hands", "unitree_h1", "unitree_h1_2",
                 "booster_t1", "booster_t1_29dof","stanford_toddy", "fourier_n1", 
                 "engineai_pm01", "kuavo_s45", "hightorque_hi", "galaxea_r1pro",
                 "berkeley_humanoid_lite", "booster_k1", "pnd_adam_lite",
                 "openloong", "tienkung","joyin","joyin_add"],
        default="unitree_g1",
    )
    
    parser.add_argument(
        "--save_path",
        default=None,
        help="Path to save the robot motion.",
    )
    
    parser.add_argument(
        "--loop",
        default=False,
        action="store_true",
        help="Loop the motion.",
    )

    parser.add_argument(
        "--record_video",
        default=False,
        action="store_true",
        help="Record the video.",
    )

    parser.add_argument(
        "--rate_limit",
        default=False,
        action="store_true",
        help="Limit the rate of the retargeted robot motion to keep the same as the human motion.",
    )

    parser.add_argument("--robot_qpos_init", type=str, default=None,
                        help="Optional: path to .json containing root_pos(3), root_rot(4 wxyz), dof_pos(N) or name->value dict.")


    parser.add_argument("--ik_config_in", type=str, required=True,
                        help="输入 IK 配置路径（支持 ik_match_table1&2 字典结构）。")
    parser.add_argument("--ik_config_out", type=str, required=True,
                        help="写回 qoffset_quat 的输出 IK 路径。")

    args = parser.parse_args()

    SMPLX_FOLDER = HERE / ".." / "assets" / "body_models"

    # Load SMPLX trajectory
    smplx_data, body_model, smplx_output, actual_human_height = load_smplx_file(
        args.smplx_file, SMPLX_FOLDER
    )

    # align fps
    tgt_fps = 30
    smplx_data_frames, aligned_fps = get_smplx_data_offline_fast(smplx_data, body_model, smplx_output, tgt_fps=tgt_fps)

    # 步骤1: 参数生成阶段
    print("=== 开始参数生成阶段 ===")
    
    # 初始化重定向器（使用原始配置）
    retarget = GMR(
        actual_human_height=actual_human_height,
        src_human="smplx",
        tgt_robot=args.robot,
    )

    # 加载IK配置
    with open(args.ik_config_in, "r", encoding="utf-8") as f:
        ik_cfg_tmp = json.load(f)

    # 获取第一帧人体数据用于参数计算
    first_frame_data = smplx_data_frames[0]

    # 加载固定机器人初始姿态
    fixed_root_pos, fixed_root_rot, fixed_dof_pos, joint_match = load_robot_init(args.robot_qpos_init)

    # 初始化FK求解器
    fk_solver = MuJoCoFK(retarget.xml_file)
    joint_order = fk_solver.joint_order

    # 映射关节顺序
    nd_expected = len(joint_order)
    vec = np.zeros(nd_expected, dtype=np.float32)
    assigned = 0

    for i, joint_name in enumerate(joint_order):
        if i < nd_expected and joint_name in fixed_dof_pos.keys():
            vec[i] = fixed_dof_pos[joint_name]
            assigned += 1
    
    print(f"[INFO] 映射关节: {assigned}/{len(fixed_dof_pos)}")
    fixed_dof_pos = vec

    # 计算FK得到机器人T-pose
    qpos_fk = np.concatenate([
        fixed_root_pos.astype(np.float64),
        fixed_root_rot.astype(np.float64),
        fixed_dof_pos.astype(np.float64)
    ], axis=0)
    centers, Rs = fk_solver.get_specific_body_positions(qpos_fk, fk_solver.body_names)

    # 优化缩放系数
    ratio = actual_human_height / ik_cfg_tmp["human_height_assumption"]
    optimized_scales = {}
    print("=== 优化缩放系数 ===")
    optimized_scales = optimize_human_scale_table(
        human_data=first_frame_data,
        robot_centers=centers,
        body_names=fk_solver.body_names,
        ik_config=ik_cfg_tmp,
        human_root_name=retarget.human_root_name,
        initial_scales=ik_cfg_tmp.get("human_scale_table", None),
        bounds=(0.1, 10.0),
        max_iter=1000,
        device='cpu',
        plot_loss=False,
        plot_save_path=None
    )

    # 创建用于GMR的缩放系数
    gmr_scales = {}
    for key, value in optimized_scales.items():
        gmr_scales[key] = float(value / ratio)

    # 缩放人体数据
    scaled_human_data = scale_human_data(
        first_frame_data,
        retarget.human_root_name,
        optimized_scales
    )
    
    missing_links = []
    # 计算四元数偏移
    quat_offsets = {}
    print("=== 计算四元数偏移 ===")
    quat_offsets, missing_links = compute_quaternion_offsets(
        scaled_human_data,
        centers, Rs, fk_solver.body_names, 
        ik_cfg_tmp
    )

    # 对齐机器人数据到人体根节点
    human_root_pos = np.array(scaled_human_data[retarget.human_root_name][0], dtype=np.float64)
    robot_root_name = ik_cfg_tmp.get("robot_root_name", "pelvis")
    
    robot_link_indices = {}
    for idx, name in enumerate(fk_solver.body_names):
        if not isinstance(name, str):
            name = name.decode("utf-8")
        robot_link_indices[name] = idx
    
    if robot_root_name in robot_link_indices:
        robot_root_idx = robot_link_indices[robot_root_name]
        robot_root_pos = centers[robot_root_idx]
    else:
        robot_root_pos = centers[0] if len(centers) > 0 else np.zeros(3)

    aligned_robot_centers = align_robot_data(
        centers, robot_root_pos, human_root_pos
    )

    # 计算位置偏移
    pos_offsets={}
    print("=== 计算位置偏移 ===")
    pos_offsets, missing_links = compute_position_offsets(
        scaled_human_data,
        aligned_robot_centers, Rs, fk_solver.body_names, 
        ik_cfg_tmp, quat_offsets
    )

    # 保存所有优化数据
    print("=== 保存优化配置 ===")
    output_file = write_all_data_to_ik(
        ik_config_path=args.ik_config_in,
        output_path=args.ik_config_out,
        human_scale_table=gmr_scales,
        pos_offsets=pos_offsets,
        quat_offsets=quat_offsets
    )

    if output_file:
        print(f"[SUCCESS] 参数生成完成！")
        print(f"  - 优化了 {len(optimized_scales)} 个缩放系数")
        print(f"  - 计算了 {len(pos_offsets)} 个位置偏移")
        print(f"  - 计算了 {len(quat_offsets)} 个四元数偏移")
        print(f"  - 所有数据已保存到: {output_file}")
    else:
        print("[ERROR] 参数生成失败！")
        exit(1)

    with open(output_file, "r", encoding="utf-8") as f:
        ik_cfg = json.load(f)
        
    ik_match_table1 = ik_cfg.get("ik_match_table1", {})
    rot_offsets={}
    for frame_name, entry in ik_match_table1.items():
        body_name, pos_weight, rot_weight, pos_offset, rot_offset = entry
        if pos_weight != 0 or rot_weight != 0:
            pos_offsets[body_name] = np.array(pos_offset)
            rot_offsets[body_name] = R.from_quat(
                rot_offset, scalar_first=True
            )

    # 得到偏移后的人体数据
    new_human_data = offset_human_data(
        scaled_human_data,
        pos_offsets,
        rot_offsets
    )

    # 步骤2: 可视化阶段（使用新生成的配置）
    print("\n=== 开始可视化阶段 ===")
    
    # 重新初始化重定向器（使用新配置）
    retarget_new = GMR(
        actual_human_height=actual_human_height,
        src_human="smplx",
        tgt_robot=args.robot,
    )

    # 初始化可视化器
    robot_motion_viewer = RobotMotionViewer(
        robot_type=args.robot,
        motion_fps=aligned_fps,
        transparent_robot=0,
        record_video=args.record_video,
        video_path=f"videos/{args.robot}_{pathlib.Path(args.smplx_file).stem}.mp4",
    )

    # 获取机器人连杆名称列表用于可视化
    robot_frames = ik_cfg_tmp.get("ik_match_table1", {}).keys()

    # 渲染 FPS 统计
    fps_counter = 0
    fps_start_time = time.time()
    fps_display_interval = 2.0

    # 可选导出
    if args.save_path is not None:
        save_dir = os.path.dirname(args.save_path)
        if save_dir:
            os.makedirs(save_dir, exist_ok=True)
        qpos_list = []

    desired_dt = 1.0 / float(aligned_fps)
    next_frame_time = time.perf_counter()

    i = 0

    while True:
        if args.rate_limit:
            now = time.perf_counter()
            if now < next_frame_time:
                time.sleep(max(0.0, next_frame_time - now))
            next_frame_time += desired_dt

        if args.loop:
            i = (i + 1) % len(smplx_data_frames)
        else:
            i += 1
            if i >= len(smplx_data_frames):
                break

        fps_counter += 1
        current_time = time.time()
        if current_time - fps_start_time >= fps_display_interval:
            actual_fps = fps_counter / (current_time - fps_start_time)
            print(f"Actual rendering FPS: {actual_fps:.2f}")
            fps_counter = 0
            fps_start_time = current_time

        smplx_frame = smplx_data_frames[i]
        qpos, last_qvel, avg_qvel = retarget_new.retarget(smplx_frame)

        # 可视化
        robot_motion_viewer.step(
            root_pos=scaled_human_data["pelvis"][0],
            root_rot=fixed_root_rot,
            dof_pos=fixed_dof_pos,
            human_motion_data=new_human_data,
            human_pos_offset=np.array([0.0, 0.0, 0.0]),
            show_human_body_name=True,
            robot_frames=robot_frames,
            show_robot_body_name=True,
            rate_limit=args.rate_limit
        )

        if args.save_path is not None:
            qpos_list.append(qpos.copy())

    # 保存运动数据
    if args.save_path is not None and qpos_list:
        np.save(args.save_path, np.array(qpos_list))
        print(f"Motion saved to {args.save_path}")

    robot_motion_viewer.close()
    print("=== 可视化完成 ===")
