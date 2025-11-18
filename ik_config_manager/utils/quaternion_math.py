import numpy as np
from scipy.spatial.transform import Rotation as R


def xyzw_to_wxyz(q_xyzw: np.ndarray) -> np.ndarray:
    q_xyzw = np.asarray(q_xyzw, dtype=np.float64).reshape(4)
    return np.array([q_xyzw[3], q_xyzw[0], q_xyzw[1], q_xyzw[2]], dtype=np.float64)

def wxyz_to_xyzw(q_wxyz: np.ndarray) -> np.ndarray:
    q_wxyz = np.asarray(q_wxyz, dtype=np.float64).reshape(4)
    return np.array([q_wxyz[1], q_wxyz[2], q_wxyz[3], q_wxyz[0]], dtype=np.float64)

def quat_normalize_wxyz(q: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    """归一化四元数（wxyz格式）"""
    n = float(np.linalg.norm(q))
    return q if n < eps else (q / n)

def rot_wxyz_mul(qA: np.ndarray, qB: np.ndarray) -> np.ndarray:
    """四元数乘 (wxyz)：q = qA ⊗ qB"""
    w1, x1, y1, z1 = qA
    w2, x2, y2, z2 = qB
    return np.array([
        w1*w2 - x1*x2 - y1*y2 - z1*z2,
        w1*x2 + x1*w2 + y1*z2 - z1*y2,
        w1*y2 - x1*z2 + y1*w2 + z1*x2,
        w1*z2 + x1*y2 - y1*x2 + z1*w2,
    ], dtype=np.float64)

def quat_conjugate_wxyz(q: np.ndarray) -> np.ndarray:
    """单位四元数的逆（共轭）"""
    return np.array([q[0], -q[1], -q[2], -q[3]], dtype=np.float64)

def R_to_quat_wxyz(Rm: np.ndarray) -> np.ndarray:
    """3x3 -> wxyz"""
    Rm = np.asarray(Rm, dtype=np.float64).reshape(3, 3)
    quat_xyzw = R.from_matrix(Rm).as_quat() # SciPy 默认 xyzw
    quat_xyzw /= max(np.linalg.norm(quat_xyzw), 1e-12)
    return np.array([quat_xyzw[3], quat_xyzw[0], quat_xyzw[1], quat_xyzw[2]], dtype=np.float32)