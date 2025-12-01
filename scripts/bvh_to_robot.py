import argparse
import pathlib
import time
from general_motion_retargeting import GeneralMotionRetargeting as GMR
from general_motion_retargeting import RobotMotionViewer
from general_motion_retargeting.utils.lafan1 import load_bvh_file
from rich import print
from tqdm import tqdm
import os
import numpy as np
from scipy.spatial.transform import Rotation as R


def _ensure_quat_hemisphere_wxyz(q_prev_wxyz: np.ndarray, q_curr_wxyz: np.ndarray) -> np.ndarray:
    """若两帧四元数（wxyz）点积为负，翻转当前帧到同一半球，避免差分跳变。"""
    if float(np.dot(q_prev_wxyz, q_curr_wxyz)) < 0.0:
        return -q_curr_wxyz
    return q_curr_wxyz


def get_bvh_frame_rate(bvh_file_path):
    """
    从BVH文件中提取帧率信息
    """
    try:
        with open(bvh_file_path, 'r') as f:
            lines = f.readlines()
        
        # 查找Frame Time行
        for line in lines:
            if line.strip().startswith('Frame Time'):
                frame_time = float(line.strip().split()[-1])
                fps = 1.0 / frame_time
                return fps
    except Exception as e:
        print(f"[WARN] Could not extract frame rate from BVH file: {e}")
    
    return None


def check_exported_root_velocities(qpos_seq: np.ndarray,
                                   qvel_seq: np.ndarray,
                                   dt_list: np.ndarray,
                                   lin_tol: float = 5e-3,
                                   ang_tol: float = 5e-3,
                                   strict: bool = False) -> None:
    """
    逐帧验证导出的 root 速度是否与位姿差分一致。
    约定：dt_list[i] 表示区间 (i-1 -> i) 的累计子步时长；dt_list[0] 无意义。
      qpos = [root_pos(3), root_quat(wxyz)(4), dof_pos...]
      qvel = [0:3 线速(world), 3:6 角速(body), 6: 关节速]
    """
    assert qpos_seq.ndim == 2 and qvel_seq.ndim == 2, "qpos/qvel must be (T, ...)"
    T = min(qpos_seq.shape[0], qvel_seq.shape[0], dt_list.shape[0])
    if T < 2:
        print("[check] sequence too short to validate (T < 2), skip.")
        return

    n_bad = 0
    for i in range(1, T):
        dt = float(dt_list[i])
        if not (dt > 0.0 and np.isfinite(dt)):
            print(f"[WARN][frame {i-1}->{i}] invalid dt (dt={dt}), skip check for this interval.")
            continue

        # 位姿
        p_prev = qpos_seq[i-1, :3]
        q_prev_wxyz = qpos_seq[i-1, 3:7]
        p_curr = qpos_seq[i, :3]
        q_curr_wxyz = qpos_seq[i, 3:7]
        q_curr_wxyz = _ensure_quat_hemisphere_wxyz(q_prev_wxyz, q_curr_wxyz)

        # R_{i-1}, R_i（body->world）
        R_prev = R.from_quat(q_prev_wxyz, scalar_first=True).as_matrix()
        R_curr = R.from_quat(q_curr_wxyz, scalar_first=True).as_matrix()

        # —— 帧间差分 —— #
        v_fd_world = (p_curr - p_prev) / dt
        R_delta = R_prev.T @ R_curr
        w_fd_body = R.from_matrix(R_delta).as_rotvec() / dt  # (i-1)body 表达

        # —— qvel —— #
        w_q_body = qvel_seq[i, 3:6]   # 体坐标角速度
        v_q_world = qvel_seq[i, 0:3]  # 世界系线速度

        lin_err = float(np.linalg.norm(v_fd_world - v_q_world))
        ang_err = float(np.linalg.norm(w_fd_body - w_q_body))

        ok = (lin_err <= lin_tol) and (ang_err <= ang_tol)
        tag = "OK  " if ok else "WARN"
        print(f"[{tag}][frame {i-1}->{i}] lin_err={lin_err:.3e}; ang_err={ang_err:.3e}; dt={dt:.6f}")

        if not ok:
            n_bad += 1
            if strict:
                raise RuntimeError(
                    f"[frame {i-1}->{i}] velocity mismatch (lin_tol={lin_tol}, ang_tol={ang_tol})."
                )

    if n_bad == 0:
        print("[check] all frame-to-frame root velocities consistent within tolerance.")
    else:
        print(f"[check] {n_bad} / {T-1} intervals exceeded tolerance "
              f"(lin_tol={lin_tol}, ang_tol={ang_tol}).")


def resample_bvh_data(bvh_data_frames, src_fps, tgt_fps):
    """
    简单的BVH数据重采样（线性插值位置，球面线性插值旋转）
    """
    if abs(src_fps - tgt_fps) < 1e-6:
        return bvh_data_frames
    
    print(f"[INFO] Resampling BVH data from {src_fps:.3f} Hz to {tgt_fps:.3f} Hz")
    
    src_interval = 1.0 / src_fps
    tgt_interval = 1.0 / tgt_fps
    
    # 计算目标时间线
    src_duration = len(bvh_data_frames) * src_interval
    tgt_frames_count = int(src_duration * tgt_fps)
    tgt_times = np.linspace(0, src_duration - src_interval, tgt_frames_count)
    
    resampled_data = []
    
    for t in tgt_times:
        src_frame_idx = t / src_interval
        idx_prev = int(np.floor(src_frame_idx))
        idx_next = min(idx_prev + 1, len(bvh_data_frames) - 1)
        alpha = src_frame_idx - idx_prev
        
        frame_prev = bvh_data_frames[idx_prev]
        frame_next = bvh_data_frames[idx_next]
        
        resampled_frame = {}
        
        for joint_name in frame_prev.keys():
            pos_prev, rot_prev = frame_prev[joint_name]
            pos_next, rot_next = frame_next[joint_name]
            
            # 线性插值位置
            pos_interp = (1 - alpha) * np.array(pos_prev) + alpha * np.array(pos_next)
            
            # 球面线性插值旋转
            rot_prev_r = R.from_quat(rot_prev, scalar_first=True)
            rot_next_r = R.from_quat(rot_next, scalar_first=True)
            rot_interp_r = R.slerp(rot_prev_r, rot_next_r, alpha)
            rot_interp = rot_interp_r.as_quat(scalar_first=True)
            
            resampled_frame[joint_name] = [pos_interp, rot_interp]
        
        resampled_data.append(resampled_frame)
    
    return resampled_data


if __name__ == "__main__":
    
    HERE = pathlib.Path(__file__).parent

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--bvh_file",
        help="BVH motion file to load.",
        required=True,
        type=str,
    )
    
    parser.add_argument(
        "--format",
        choices=["lafan1", "nokov"],
        default="lafan1",
    )
    
    parser.add_argument(
        "--loop",
        default=False,
        action="store_true",
        help="Loop the motion.",
    )
    
    parser.add_argument(
        "--robot",
        choices=["unitree_g1", "unitree_g1_with_hands", "booster_t1", "stanford_toddy", "fourier_n1", "engineai_pm01", "pal_talos"],
        default="unitree_g1",
    )
    
    parser.add_argument(
        "--record_video",
        action="store_true",
        default=False,
    )

    parser.add_argument(
        "--video_path",
        type=str,
        default="videos/example.mp4",
    )

    parser.add_argument(
        "--rate_limit",
        action="store_true",
        default=False,
    )

    parser.add_argument(
        "--save_path",
        default=None,
        help="Path to save the robot motion.",
    )
    
    parser.add_argument(
        "--motion_fps",
        default=None,
        type=float,
        help="Target FPS for retargeting. If not specified, will use source FPS.",
    )
    
    # 校验阈值与严格模式
    parser.add_argument("--vel_check_strict", action="store_true", default=False)
    parser.add_argument("--lin_tol", type=float, default=5e-3)
    parser.add_argument("--ang_tol", type=float, default=5e-3)

    args = parser.parse_args()

    # Load BVH trajectory
    lafan1_data_frames, actual_human_height = load_bvh_file(args.bvh_file, format=args.format)
    
    # ========= 从BVH文件中检测源帧率 =========
    src_fps = get_bvh_frame_rate(args.bvh_file)
    if src_fps is None:
        src_fps = args.motion_fps if args.motion_fps is not None else 30.0
        print(f"[WARN] Could not detect frame rate from BVH file, using {src_fps:.3f} Hz")
    else:
        print(f"[INFO] Detected source FPS from BVH file: {src_fps:.3f} Hz")
    
    # 确定目标帧率
    tgt_fps = args.motion_fps if args.motion_fps is not None else src_fps
    print(f"[INFO] Target FPS: {tgt_fps:.3f} Hz")
    
    # 如果源帧率和目标帧率不同，进行重采样
    if abs(src_fps - tgt_fps) > 1e-6:
        lafan1_data_frames = resample_bvh_data(lafan1_data_frames, src_fps, tgt_fps)
        aligned_fps = tgt_fps
        print(f"[INFO] Resampled to {len(lafan1_data_frames)} frames at {aligned_fps:.3f} Hz")
    else:
        aligned_fps = src_fps
        print(f"[INFO] Using original {len(lafan1_data_frames)} frames at {aligned_fps:.3f} Hz")
    
    # Initialize the retargeting system
    retargeter = GMR(
        src_human=f"bvh_{args.format}",
        tgt_robot=args.robot,
        actual_human_height=actual_human_height,
    )

    robot_motion_viewer = RobotMotionViewer(robot_type=args.robot,
                                            motion_fps=aligned_fps,
                                            transparent_robot=0,
                                            record_video=args.record_video,
                                            video_path=args.video_path,
                                            # video_width=2080,
                                            # video_height=1170
                                            )
    
    # FPS measurement variables
    fps_counter = 0
    fps_start_time = time.time()
    fps_display_interval = 2.0  # Display FPS every 2 seconds
    
    print(f"Final motion FPS: {aligned_fps}")
    
    # 保存容器
    if args.save_path is not None:
        save_dir = os.path.dirname(args.save_path)
        if save_dir:  # Only create directory if it's not empty
            os.makedirs(save_dir, exist_ok=True)
        qpos_list = []
        qvel_list = []
        frame_dt_list = []

    # Create tqdm progress bar for the total number of frames
    pbar = tqdm(total=len(lafan1_data_frames), desc="Retargeting")
    
    # ====== 仅 --rate_limit 时，按墙钟限速 ======
    desired_dt = 1.0 / float(aligned_fps)
    next_frame_time = time.perf_counter()

    # Start the viewer
    i = 0
    
    target_dt = 1.0 / float(aligned_fps)  # 目标帧时长（传给 retarget）

    try:
        # 让 frame_dt 与索引对齐：先放一个占位 0.0（表示第 0 帧不存在的前置区间）
        if args.save_path is not None:
            frame_dt_list.append(0.0)

        while True:
            # 限速（仅开启时）
            if args.rate_limit:
                now = time.perf_counter()
                if now < next_frame_time:
                    time.sleep(next_frame_time - now)
                    next_frame_time += desired_dt
                else:
                    missed = int((now - next_frame_time) // desired_dt) + 1
                    next_frame_time += missed * desired_dt
            
            # FPS measurement
            fps_counter += 1
            current_time = time.time()
            if current_time - fps_start_time >= fps_display_interval:
                actual_fps = fps_counter / (current_time - fps_start_time)
                print(f"Actual rendering FPS: {actual_fps:.2f}")
                fps_counter = 0
                fps_start_time = current_time
                
            # Update progress bar
            pbar.update(1)

            # Update task targets.
            smplx_data = lafan1_data_frames[i]

            # retarget with frame_dt_target
            try:
                ret = retargeter.retarget(smplx_data, frame_dt_target=target_dt)
            except TypeError:
                ret = retargeter.retarget(smplx_data)

            # 解包：支持返回 2 或 3 项
            if isinstance(ret, tuple):
                if len(ret) == 3:
                    qpos, _qvel_last, qvel = ret
                elif len(ret) == 2:
                    qpos, qvel = ret
                    _qvel_last = None
                else:
                    raise RuntimeError(f"Unexpected retarget() return length: {len(ret)}")
            else:
                raise RuntimeError("retarget() should return a tuple")

            # dt：优先取 retarget.last_frame_dt；没有就用目标帧时长
            dt_this = getattr(retargeter, "last_frame_dt", None)
            if not (isinstance(dt_this, (float, np.floating)) and np.isfinite(dt_this) and dt_this > 0.0):
                dt_this = target_dt

            # 获取机器人对应连杆名称列表
            robot_frames = retargeter.ik_match_table1.keys()
            
            # 可视化
            robot_motion_viewer.step(
                root_pos=qpos[:3],
                root_rot=qpos[3:7],
                dof_pos=qpos[7:],
                human_motion_data=retargeter.scaled_human_data,
                human_pos_offset=np.array([0.0, 0.0, 0.0]),
                show_human_body_name=False,
                robot_frames=robot_frames,
                show_robot_body_name=False,
                rate_limit=args.rate_limit,
                # human_pos_offset=np.array([0.0, 0.0, 0.0])
            )

            if args.loop:
                i = (i + 1) % len(lafan1_data_frames)
            else:
                i += 1
                if i >= len(lafan1_data_frames):
                    break
   
            if args.save_path is not None:
                qpos_list.append(qpos)
                qvel_list.append(qvel)
                frame_dt_list.append(float(dt_this))  # 区间 (i-1 -> i) 的 dt
    
    finally:
        # —— 确保渲染与录制干净关闭，避免 GLXBadContext / segfault —— #
        try:
            if getattr(robot_motion_viewer, "stop_recording", None) and args.record_video:
                robot_motion_viewer.stop_recording()
        except Exception as e:
            print(f"[WARN] stop_recording failed: {e}")
        try:
            if getattr(robot_motion_viewer, "close", None):
                robot_motion_viewer.close()
            elif getattr(robot_motion_viewer, "destroy", None):
                robot_motion_viewer.destroy()
        except Exception as e:
            print(f"[WARN] viewer close failed: {e}")
        # 给后台线程（渲染/视频写入）一点收尾时间
        time.sleep(0.05)
    
    # Close progress bar
    pbar.close()
    
    # ====== 导出 ======
    if args.save_path is not None:
        import pickle
        from pathlib import Path

        # 1) 堆成数组
        qpos_arr = np.asarray(qpos_list)
        qvel_arr = np.asarray(qvel_list)
        frame_dt_arr = np.asarray(frame_dt_list, dtype=np.float32)
        # 说明：frame_dt_arr[0] = 0.0（占位）；区间 (i-1->i) 用 frame_dt_arr[i]

        # —— 导出前做逐帧校验（按每帧自己的 dt）——
        try:
            check_exported_root_velocities(
                qpos_seq=qpos_arr,          # (T, 7+Nd)
                qvel_seq=qvel_arr,          # (T, 6+Nd)
                dt_list=frame_dt_arr,       # (T+1,) 但我们只用 1..T 区间
                lin_tol=getattr(args, "lin_tol", 5e-3),
                ang_tol=getattr(args, "ang_tol", 5e-3),
                strict=getattr(args, "vel_check_strict", False),
            )
        except Exception as e:
            print(f"[ERROR] export-time velocity validation failed: {e}")
            # 如需硬失败可改为 raise

        # 2) 拆分（保持你当前的索引解释方式）
        root_pos = qpos_arr[:, :3]                                 # (T,3)
        root_rot_wxyz = qpos_arr[:, 3:7]                           # (T,4) wxyz
        root_rot_xyzw = root_rot_wxyz[:, [1, 2, 3, 0]]             # (T,4) xyzw
        dof_pos  = qpos_arr[:, 7:]                                 # (T,Nd)

        # qvel 0:3 线速(world), 3:6 角速(body)
        root_vel_world = qvel_arr[:, 0:3]                          # (T,3) 世界系线速度
        root_rot_vel   = qvel_arr[:, 3:6]                          # (T,3) 角速度（body）
        dof_vel        = qvel_arr[:, 6:]                           # (T,Nd)

        # 3) 长度对齐（以最短为准）(去前 5 帧异常）
        L = min(root_pos.shape[0], root_rot_xyzw.shape[0], dof_pos.shape[0],
                root_vel_world.shape[0], root_rot_vel.shape[0], dof_vel.shape[0], frame_dt_arr.shape[0])
        if L > 5:  # 只有帧数足够时才去掉前5帧
            root_pos       = root_pos[5:L].astype(np.float32)
            root_rot_wxyz  = root_rot_wxyz[5:L].astype(np.float32)
            root_rot_xyzw  = root_rot_xyzw[5:L].astype(np.float32)
            dof_pos        = dof_pos[5:L].astype(np.float32)
            root_vel_world = root_vel_world[5:L].astype(np.float32)
            root_rot_vel   = root_rot_vel[5:L].astype(np.float32)
            dof_vel        = dof_vel[5:L].astype(np.float32)
            frame_dt_arr   = frame_dt_arr[5:L].astype(np.float32)
        else:
            # 如果帧数太少，使用所有帧
            root_pos       = root_pos.astype(np.float32)
            root_rot_wxyz  = root_rot_wxyz.astype(np.float32)
            root_rot_xyzw  = root_rot_xyzw.astype(np.float32)
            dof_pos        = dof_pos.astype(np.float32)
            root_vel_world = root_vel_world.astype(np.float32)
            root_rot_vel   = root_rot_vel.astype(np.float32)
            dof_vel        = dof_vel.astype(np.float32)
            frame_dt_arr   = frame_dt_arr.astype(np.float32)

        # 4) 计算 root_vel_body：v_body = R^T * v_world （R: body->world, 由 wxyz 四元数得到）
        R_bw = R.from_quat(root_rot_wxyz, scalar_first=True).as_matrix()   # (T,3,3) body->world
        R_wb = np.transpose(R_bw, (0, 2, 1))                               # (T,3,3) world->body
        root_vel_body = np.einsum('tij,tj->ti', R_wb, root_vel_world).astype(np.float32)  # (T,3)

        # 5) 构造导出数据格式
        seq_name = Path(args.bvh_file).stem
        export_data = {
            "root_pos": root_pos,                 # (T,3) world
            "root_vel": root_vel_world,           # (T,3) world（兼容字段）
            "root_vel_body": root_vel_body,       # (T,3) body（新增）
            "root_rot": root_rot_xyzw,            # (T,4) xyzw
            "root_rot_vel": root_rot_vel,         # (T,3) body
            "dof_pos": dof_pos,                   # (T,Nd)
            "dof_vel": dof_vel,                   # (T,Nd)
            "local_body_pos": None,
            "link_body_list": None,
            "frame_rate": float(aligned_fps),
            "fps": float(aligned_fps),
            "meta": {
                "root_rot_convention": "xyzw",
                "root_ang_vel_space": "local",      # 体坐标
                "root_lin_vel_space_world": "world", # 线速度（root_vel）的坐标系
                "root_lin_vel_space_body": "local",  # 线速度（root_vel_body）的坐标系
                # "frame_dt_per_step": frame_dt_arr    # (T,) 每帧累计子步时长（与区间对齐）
            },
        }

        with open(args.save_path, "wb") as f:
            pickle.dump(export_data, f)
        print(f"Saved to {args.save_path}")
