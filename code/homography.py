# -*- coding: utf-8 -*-
"""
射影矩阵（单应矩阵）自编程实现 —— 核心求解模块
第一章加分作业

坐标约定
    - 像素坐标 (x, y)：x 为列号（向右增大），y 为行号（向下增大）
    - 整点约定：宽 W 的图 x 取 0..W-1，高 H 的图 y 取 0..H-1
    - 点集统一为 shape=(n,2) 的 float64 数组，点序固定 左上→右上→右下→左下

本模块实现（全部自编程，只依赖 numpy 的线性代数原语）
    normalize_points      Hartley 归一化（质心→原点，平均距离→sqrt(2)）
    build_A               构造 A h = 0 的系数矩阵（2n×9）
    solve_homography_svd  主实现：9 未知量齐次 SVD
    solve_homography_8dof 对照实现：固定 h33=1 的 8 元非齐次解
    apply_homography      用 H 映射点
    invert_homography     手写伴随矩阵求逆
    reprojection_error    重投影误差
    homography_distance   尺度/符号无关的 H 比较
    diagnose              条件数、奇异值谱、det 等数值诊断
"""
from __future__ import annotations

import numpy as np

EPS = 1e-12


# ------------------------------------------------------------------ 基础工具

def to_points(pts) -> np.ndarray:
    """转成 (n,2) float64，并做基本合法性检查"""
    p = np.asarray(pts, dtype=np.float64)
    if p.ndim != 2 or p.shape[1] != 2:
        raise ValueError(f"点集形状必须为 (n,2)，当前为 {p.shape}")
    if p.shape[0] < 4:
        raise ValueError(f"至少需要 4 对对应点，当前为 {p.shape[0]} 对")
    if not np.isfinite(p).all():
        raise ValueError("点集包含 NaN/Inf")
    return p


def normalize_points(pts):
    """
    Hartley 归一化
        1. 平移：点集质心移至原点
        2. 缩放：到原点的平均欧氏距离缩放到 sqrt(2)
    返回 (归一化点, 归一化矩阵 T)，满足 pts_n ≈ (T @ [pts,1]^T) 的前两维
    """
    p = to_points(pts)
    c = p.mean(axis=0)                          # 质心
    d = np.sqrt(((p - c) ** 2).sum(axis=1))     # 各点到质心的距离
    mean_d = float(d.mean())
    if mean_d < EPS:
        raise ValueError("点集退化：所有点几乎重合")
    s = np.sqrt(2.0) / mean_d                   # 缩放因子
    T = np.array([[s, 0.0, -s * c[0]],
                  [0.0, s, -s * c[1]],
                  [0.0, 0.0, 1.0]], dtype=np.float64)
    return (p - c) * s, T


def build_A(src, dst):
    """
    构造线性方程组 A h = 0 的系数矩阵，A 为 2n × 9

        A_i = [ x   y   1   0   0   0   -x·x'  -y·x'  -x' ]
              [ 0   0   0   x   y   1   -x·y'  -y·y'  -y' ]

    第 2 行第 3 个元素必须为 0：由 y'(h31·x + h32·y + h33) = h21·x + h22·y + h23
    移项所得方程中不含 h13，故其系数为 0。
    """
    src = to_points(src)
    dst = to_points(dst)
    if src.shape != dst.shape:
        raise ValueError("源点与目标点数量不一致")
    n = src.shape[0]
    x, y = src[:, 0], src[:, 1]
    xp, yp = dst[:, 0], dst[:, 1]

    A = np.zeros((2 * n, 9), dtype=np.float64)
    A[0::2, 0] = x
    A[0::2, 1] = y
    A[0::2, 2] = 1.0
    A[0::2, 6] = -x * xp
    A[0::2, 7] = -y * xp
    A[0::2, 8] = -xp

    A[1::2, 3] = x
    A[1::2, 4] = y
    A[1::2, 5] = 1.0
    A[1::2, 6] = -x * yp
    A[1::2, 7] = -y * yp
    A[1::2, 8] = -yp
    return A


# ------------------------------------------------------------------ H 的尺度处理

def normalize_h(H, mode="fro"):
    """
    尺度规范化
        mode='fro' 用 Frobenius 范数（h33 可能为 0 时唯一安全的选择）
        mode='h33' 用 h33 归一化（便于阅读、与真值直接对比）
    """
    H = np.asarray(H, dtype=np.float64)
    if mode == "h33":
        if abs(H[2, 2]) < 1e-10:
            raise ValueError("H[2,2] ≈ 0：该 H 把源点原点映到无穷远，无法用 h33 归一化")
        return H / H[2, 2]
    nrm = float(np.linalg.norm(H))
    if nrm < EPS:
        raise ValueError("H 为零矩阵")
    return H / nrm


# ------------------------------------------------------------------ 求解器

def solve_homography_svd(src, dst, normalize=True):
    """
    主实现：不固定 h33 的 9 未知量齐次 SVD 解

        1. （默认）对源点和目标点分别做 Hartley 归一化
        2. 构造 A（2n × 9）
        3. SVD：A = U Σ V^T，取 V 的最后一列（最小奇异值对应的右奇异向量）作为 h
        4. 重塑为 3×3 得 H~
        5. 反归一化 H = T'^{-1} H~ T

    返回单位 Frobenius 范数的 H（整体符号仍任意，属齐次坐标固有性质，比较时需取 ±）
    """
    src = to_points(src)
    dst = to_points(dst)
    if src.shape != dst.shape:
        raise ValueError("源点与目标点数量不一致")

    Ts = Td = np.eye(3)
    if normalize:
        src_n, Ts = normalize_points(src)
        dst_n, Td = normalize_points(dst)
    else:
        src_n, dst_n = src, dst

    A = build_A(src_n, dst_n)
    _, _, Vt = np.linalg.svd(A)          # Vt 各行是右奇异向量，按奇异值降序排列
    h = Vt[-1]                           # 最小奇异值对应的右奇异向量
    H = h.reshape(3, 3)

    H = np.linalg.inv(Td) @ H @ Ts       # 反归一化
    return normalize_h(H, "fro")


def solve_homography_8dof(src, dst):
    """
    对照实现：固定 h33 = 1，把 A h = 0 改写为 8 元非齐次方程组 M x = b

        M = A[:, :8]，b = -A[:, 8]，x = [h11 h12 h13 h21 h22 h23 h31 h32]^T

    返回 (H, info)。若 info['linear_residual'] 显著不为 0（或 cond(M) ~ 1e16），
    说明"不存在 h33=1 的解"，即该规范化约定失效（真实 h33=0 的情形）。
    """
    src = to_points(src)
    dst = to_points(dst)
    A = build_A(src, dst)
    M, b = A[:, :8], -A[:, 8]

    cond_M = float(np.linalg.cond(M))
    x, *_ = np.linalg.lstsq(M, b, rcond=None)
    lin_res = float(np.linalg.norm(M @ x - b))
    H = np.append(x, 1.0).reshape(3, 3)
    info = {
        "cond_M": cond_M,
        "linear_residual": lin_res,
        "consistent": bool(lin_res < 1e-8 * max(1.0, float(np.linalg.norm(b)))),
    }
    return H, info


def apply_homography(H, pts):
    """按 H 映射点集：p' ~ H p。落在无穷远的点返回 NaN"""
    H = np.asarray(H, dtype=np.float64)
    p = np.asarray(pts, dtype=np.float64).reshape(-1, 2)
    ph = np.hstack([p, np.ones((p.shape[0], 1))])
    q = (H @ ph.T).T
    w = q[:, 2]
    out = np.full((p.shape[0], 2), np.nan, dtype=np.float64)
    ok = np.abs(w) > 1e-12
    out[ok] = q[ok, :2] / w[ok, None]
    return out


def invert_homography(H):
    """手写伴随矩阵法求 3×3 逆（不调用 np.linalg.inv）"""
    H = np.asarray(H, dtype=np.float64)
    a, b, c = H[0]
    d, e, f = H[1]
    g, h, i = H[2]
    adj = np.array([[e * i - f * h, c * h - b * i, b * f - c * e],
                    [f * g - d * i, a * i - c * g, c * d - a * f],
                    [d * h - e * g, b * g - a * h, a * e - b * d]], dtype=np.float64)
    det = a * (e * i - f * h) - b * (d * i - f * g) + c * (d * h - e * g)
    if abs(det) < 1e-14:
        raise ValueError(f"H 近乎不可逆（det={det:.3e}），四个角点可能共线")
    return adj / det


# ------------------------------------------------------------------ 评估与诊断

def reprojection_error(H, src, dst):
    """重投影误差：返回 (rmse, 每点误差数组)。映射到无穷远的点记为 inf"""
    dst = np.asarray(dst, dtype=np.float64).reshape(-1, 2)
    proj = apply_homography(H, src)
    bad = ~np.isfinite(proj).all(axis=1)
    err = np.full(dst.shape[0], np.inf)
    err[~bad] = np.sqrt(((proj[~bad] - dst[~bad]) ** 2).sum(axis=1))
    finite = err[np.isfinite(err)]
    rmse = float(np.sqrt((finite ** 2).mean())) if finite.size else float("inf")
    return rmse, err


def homography_distance(H1, H2):
    """齐次尺度与整体符号都无关的相对距离：min(‖Ĥ1-Ĥ2‖_F, ‖Ĥ1+Ĥ2‖_F)"""
    A1 = np.asarray(H1, dtype=np.float64)
    A2 = np.asarray(H2, dtype=np.float64)
    A1 = A1 / np.linalg.norm(A1)
    A2 = A2 / np.linalg.norm(A2)
    return float(min(np.linalg.norm(A1 - A2), np.linalg.norm(A1 + A2)))


def diagnose(A, H=None):
    """数值诊断：A 的奇异值谱与条件数；可选给出 H 的行列式与条件数"""
    s = np.linalg.svd(np.asarray(A, dtype=np.float64), compute_uv=False)
    info = {
        "A_shape": list(np.shape(A)),
        "cond_A": float(s[0] / s[-1]) if s[-1] > 0 else float("inf"),
        "sigma_8_over_sigma_9": float(s[-2] / s[-1]) if s[-1] > 0 else float("inf"),
        "singular_values": [float(v) for v in s],
    }
    if H is not None:
        info["cond_H"] = float(np.linalg.cond(H))
        info["det_H"] = float(np.linalg.det(H))
    return info


def check_point_configuration(pts):
    """检查退化配置：任意三点是否接近共线（共线则 A 秩亏，H 不可解）"""
    p = to_points(pts)
    issues = []
    n = p.shape[0]
    for i in range(n):
        for j in range(i + 1, n):
            for k in range(j + 1, n):
                a, b = p[j] - p[i], p[k] - p[i]
                cross = abs(a[0] * b[1] - a[1] * b[0])
                scale = max(float(np.linalg.norm(a)), float(np.linalg.norm(b)), EPS)
                if cross / scale < 1e-6:
                    issues.append(f"点 {i},{j},{k} 接近共线（归一化面积 {cross / scale:.2e}）")
    return issues
