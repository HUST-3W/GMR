import argparse
import pathlib
import os
import time
import numpy as np

from general_motion_retargeting import GeneralMotionRetargeting as GMR
from general_motion_retargeting import RobotMotionViewer
from general_motion_retargeting.utils.smpl import (
    load_smplx_file,
    get_smplx_data_offline_fast,
)
from scipy.spatial.transform import Rotation as R
from rich import print


def _ensure_quat_hemisphere_wxyz(q_prev_wxyz: np.ndarray, q_curr_wxyz: np.ndarray) -> np.ndarray:
    """若两帧四元数（wxyz）点积为负，翻转当前帧到同一半球，避免差分跳变。"""
    if float(np.dot(q_prev_wxyz, q_curr_wxyz)) < 0.0:
        return -q_curr_wxyz
    return q_curr_wxyz


def _extract_scalar(x):
    """把标量/长度为1的 ndarray / list 提取成 float。否则返回 None。"""
    if x is None:
        return None
    if isinstance(x, (int, float, np.floating, np.integer)):
        return float(x)
    if isinstance(x, (list, tuple)) and len(x) == 1:
        v = x[0]
        return float(v) if isinstance(v, (int, float, np.floating, np.integer)) else None
    if isinstance(x, np.ndarray):
        if x.shape == () or x.size == 1:
            return float(x.reshape(-1)[0])
    return None


def _extract_fps_from_obj(obj):
    """从 obj（可能是 dict/npz-like）中尝试读取 fps，返回 float 或 None。"""
    candidate_keys = [
        "frame_rate", "framerate", "fps",
        "FrameRate", "FramesPerSecond",
        "mocap_framerate", "mocap_frame_rate",
        "sampling_rate", "sample_rate",
    ]
    # 直接一级 key
    if hasattr(obj, "keys"):
        try:
            keys = list(obj.keys())
        except Exception:
            keys = []
        for k in candidate_keys:
            if k in keys:
                v = _extract_scalar(obj[k])
                if v is not None and v > 0:
                    return v
    # 常见嵌套 meta/info
    for subkey in ["meta", "Meta", "info", "Info", "header"]:
        try:
            sub = obj.get(subkey, None) if hasattr(obj, "get") else None
        except Exception:
            sub = None
        if sub is not None and hasattr(sub, "keys"):
            for k in candidate_keys:
                if k in sub:
                    v = _extract_scalar(sub[k])
                    if v is not None and v > 0:
                        return v
    return None


def _extract_source_fps(smplx_data, fallback=30.0):
    """
    尝试从 smplx_data（load_smplx_file 的返回之一）里读取原始帧率。
    读取不到则回退到 fallback（默认 30）。
    """
    fps = _extract_fps_from_obj(smplx_data)
    if fps is None:
        # 有些数据会把 fps 放在 'misc' 或类似结构里，这里再多试一手
        for k in ["misc", "extras", "additional", "attributes"]:
            try:
                sub = smplx_data.get(k, None) if hasattr(smplx_data, "get") else None
            except Exception:
                sub = None
            if sub is not None:
                fps = _extract_fps_from_obj(sub)
                if fps is not None:
                    break
    if fps is None:
        fps = float(fallback)
        print(f"[INFO] fps not found in file; fallback to {fps:.3f} Hz")
    else:
        print(f"[INFO] detected source fps from file: {fps:.3f} Hz")
    return float(fps)


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


if __name__ == "__main__":
    HERE = pathlib.Path(__file__).parent

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--smplx_file",
        help="SMPLX motion file to load.",
        type=str,
        default="/home/yanjieze/projects/g1_wbc/GMR/motion_data/ACCAD/Male1General_c3d/General_A1_-_Stand_stageii.npz",
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
    parser.add_argument("--save_path", default=None, help="Path to save the robot motion.")
    parser.add_argument("--loop", default=False, action="store_true", help="Loop the motion.")
    parser.add_argument("--record_video", default=False, action="store_true", help="Record the video.")
    parser.add_argument("--rate_limit", default=False, action="store_true",
                        help="Limit the wall-clock playback rate to the motion FPS.")
    # 校验阈值与严格模式
    parser.add_argument("--vel_check_strict", action="store_true", default=False)
    parser.add_argument("--lin_tol", type=float, default=5e-3)
    parser.add_argument("--ang_tol", type=float, default=5e-3)

    args = parser.parse_args()

    SMPLX_FOLDER = HERE / ".." / "assets" / "body_models"

    # Load SMPLX trajectory
    smplx_data, body_model, smplx_output, actual_human_height = load_smplx_file(
        args.smplx_file, SMPLX_FOLDER
    )

    # ========= 从数据里读取 fps，不到就回退 =========
    src_fps = _extract_source_fps(smplx_data, fallback=30.0)
    tgt_fps = src_fps
    print(f"[INFO] target fps set to: {tgt_fps:.3f} Hz")

    # 对齐到目标 fps
    smplx_data_frames, aligned_fps = get_smplx_data_offline_fast(
        smplx_data, body_model, smplx_output, tgt_fps=tgt_fps
    )
    print(f"[INFO] aligned_fps reported by pipeline: {aligned_fps:.3f} Hz")

    # Initialize the retargeting system
    retarget = GMR(
        actual_human_height=actual_human_height,
        src_human="smplx",
        tgt_robot=args.robot,
    )

    # viewer
    robot_motion_viewer = RobotMotionViewer(
        robot_type=args.robot,
        motion_fps=aligned_fps,
        transparent_robot=0,
        record_video=args.record_video,
        video_path=f"videos/{args.robot}_{pathlib.Path(args.smplx_file).stem}.mp4",
    )

    # FPS measurement (rendering)
    fps_counter = 0
    fps_start_time = time.time()
    fps_display_interval = 2.0  # Display FPS every 2 seconds

    # 保存容器
    if args.save_path is not None:
        save_dir = os.path.dirname(args.save_path)
        if save_dir:
            os.makedirs(save_dir, exist_ok=True)
        qpos_list = []
        qvel_list = []
        frame_dt_list = []

    # ====== 仅 --rate_limit 时，按墙钟限速 ======
    desired_dt = 1.0 / float(aligned_fps)
    next_frame_time = time.perf_counter()

    # 运行主循环
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

            # 帧索引推进
            if args.loop:
                i = (i + 1) % len(smplx_data_frames)
            else:
                i += 1
                if i >= len(smplx_data_frames):
                    break

            # 渲染速率统计
            fps_counter += 1
            current_time = time.time()
            if current_time - fps_start_time >= fps_display_interval:
                actual_fps = fps_counter / (current_time - fps_start_time)
                print(f"Actual rendering FPS: {actual_fps:.2f}")
                fps_counter = 0
                fps_start_time = current_time

            # Update task targets.
            smplx_frame = smplx_data_frames[i]

            # 兼容 retarget 签名
            try:
                ret = retarget.retarget(smplx_frame, frame_dt_target=target_dt)
            except TypeError:
                ret = retarget.retarget(smplx_frame)

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
            dt_this = getattr(retarget, "last_frame_dt", None)
            if not (isinstance(dt_this, (float, np.floating)) and np.isfinite(dt_this) and dt_this > 0.0):
                dt_this = target_dt

            # 获取机器人对应连杆名称列表
            robot_frames = retarget.ik_match_table1.keys()

            # 可视化
            robot_motion_viewer.step(
                root_pos=qpos[:3],
                root_rot=qpos[3:7],
                dof_pos=qpos[7:],
                human_motion_data=retarget.scaled_human_data,
                human_pos_offset=np.array([0.0, 0.0, 0.0]),
                show_human_body_name=False,
                robot_frames=robot_frames,
                show_robot_body_name=False,
                rate_limit=args.rate_limit,
            )

            # 记录
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
        root_pos       = root_pos[5:L].astype(np.float32)
        root_rot_wxyz  = root_rot_wxyz[5:L].astype(np.float32)
        root_rot_xyzw  = root_rot_xyzw[5:L].astype(np.float32)
        dof_pos        = dof_pos[5:L].astype(np.float32)
        root_vel_world = root_vel_world[5:L].astype(np.float32)
        root_rot_vel   = root_rot_vel[5:L].astype(np.float32)
        dof_vel        = dof_vel[5:L].astype(np.float32)
        frame_dt_arr   = frame_dt_arr[5:L].astype(np.float32)

        # 4) 计算 root_vel_body：v_body = R^T * v_world （R: body->world, 由 wxyz 四元数得到）
        R_bw = R.from_quat(root_rot_wxyz, scalar_first=True).as_matrix()   # (T,3,3) body->world
        R_wb = np.transpose(R_bw, (0, 2, 1))                               # (T,3,3) world->body
        root_vel_body = np.einsum('tij,tj->ti', R_wb, root_vel_world).astype(np.float32)  # (T,3)

        # 5) 构造导出数据格式
        seq_name = Path(args.smplx_file).stem
        export_data = {
            "motion_file": seq_name,
            "root_pos": root_pos,                 # (T,3) world
            "root_vel": root_vel_world,           # (T,3) world（兼容字段）
            "root_vel_body": root_vel_body,       # (T,3) body（新增）
            "root_rot": root_rot_xyzw,            # (T,4) xyzw
            "root_rot_vel": root_rot_vel,         # (T,3) body
            "dof_pos": dof_pos,                   # (T,Nd)
            "dof_vel": dof_vel,                   # (T,Nd)
            "local_body_pos": None,
            "link_body_list": None,
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
        print(f"Saved (legacy-compatible) to {args.save_path}")
