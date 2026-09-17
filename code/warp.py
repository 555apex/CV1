# -*- coding: utf-8 -*-
"""
图像重采样模块 —— 手写逆映射 + 双线性插值
第一章加分作业（不调用 cv2.warpPerspective / cv2.remap）

目标像素坐标 (u, v)：u 为列、v 为行；整点约定 0..W-1 / 0..H-1
逆向映射：src_pt ~ H_inv @ (u, v, 1)

为什么用逆映射
    前向映射（对源像素逐个映射到目标）在放大/斜视时会在目标图留下空洞；
    逆映射对每个目标整数像素反算源坐标采样，天然无空洞。
"""
from __future__ import annotations

import numpy as np


def target_corners(W, H):
    """正视图四角点（固定）：左上、右上、右下、左下"""
    return np.array([[0.0, 0.0], [W - 1.0, 0.0],
                     [W - 1.0, H - 1.0], [0.0, H - 1.0]], dtype=np.float64)


def estimate_target_size(quad, min_side=1):
    """
    由斜图四边形估计正视图尺寸（对边平均长度，保持宽高比，不额外引入各向异性缩放）

        W = (|p1-p0| + |p2-p3|) / 2
        H = (|p3-p0| + |p2-p1|) / 2
    """
    p = np.asarray(quad, dtype=np.float64).reshape(4, 2)
    w_top = float(np.linalg.norm(p[1] - p[0]))
    w_bot = float(np.linalg.norm(p[2] - p[3]))
    h_left = float(np.linalg.norm(p[3] - p[0]))
    h_right = float(np.linalg.norm(p[2] - p[1]))
    W = int(round((w_top + w_bot) / 2.0))
    H = int(round((h_left + h_right) / 2.0))
    return max(W, min_side), max(H, min_side)


def _backward_map(H_inv, W, H):
    """为每个目标像素反算源坐标，返回 (xs, ys, 齐次分量非零掩膜)"""
    H_inv = np.asarray(H_inv, dtype=np.float64)
    uu, vv = np.meshgrid(np.arange(W, dtype=np.float64),
                         np.arange(H, dtype=np.float64))
    w = H_inv[2, 0] * uu + H_inv[2, 1] * vv + H_inv[2, 2]
    xs = np.full(uu.shape, np.nan)
    ys = np.full(uu.shape, np.nan)
    ok = np.abs(w) > 1e-10               # 齐次分量过小 → 映到无穷远，判为无效
    xs[ok] = (H_inv[0, 0] * uu[ok] + H_inv[0, 1] * vv[ok] + H_inv[0, 2]) / w[ok]
    ys[ok] = (H_inv[1, 0] * uu[ok] + H_inv[1, 1] * vv[ok] + H_inv[1, 2]) / w[ok]
    return xs, ys, ok


def warp_bilinear(img, H_inv, W, H, fill=0):
    """逆映射 + 双线性插值。返回 (输出图像, 有效像素掩膜)"""
    img = np.asarray(img, dtype=np.float64)
    squeeze = (img.ndim == 2)
    if squeeze:
        img = img[:, :, None]
    if img.ndim != 3:
        raise ValueError("图像维度必须为 2 或 3")
    Hs, Ws, C = img.shape

    xs, ys, ok = _backward_map(H_inv, W, H)
    ok &= (xs >= 0) & (xs <= Ws - 1) & (ys >= 0) & (ys <= Hs - 1)

    out = np.full((H, W, C), float(fill), dtype=np.float64)
    if ok.any():
        idx = np.where(ok)
        x, y = xs[idx], ys[idx]
        x0 = np.floor(x).astype(np.int64)
        y0 = np.floor(y).astype(np.int64)
        x1 = np.minimum(x0 + 1, Ws - 1)
        y1 = np.minimum(y0 + 1, Hs - 1)
        dx = (x - x0)[:, None]
        dy = (y - y0)[:, None]

        c00 = img[y0, x0]
        c01 = img[y0, x1]
        c10 = img[y1, x0]
        c11 = img[y1, x1]
        out[idx] = ((1 - dx) * (1 - dy) * c00 +
                    dx * (1 - dy) * c01 +
                    (1 - dx) * dy * c10 +
                    dx * dy * c11)
    return (out[:, :, 0] if squeeze else out), ok


def warp_nearest(img, H_inv, W, H, fill=0):
    """逆映射 + 最近邻插值（仅作插值质量对照）"""
    img = np.asarray(img, dtype=np.float64)
    squeeze = (img.ndim == 2)
    if squeeze:
        img = img[:, :, None]
    Hs, Ws, C = img.shape

    xs, ys, ok = _backward_map(H_inv, W, H)
    ok &= (xs >= 0) & (xs <= Ws - 1) & (ys >= 0) & (ys <= Hs - 1)

    out = np.full((H, W, C), float(fill), dtype=np.float64)
    if ok.any():
        idx = np.where(ok)
        xi = np.clip(np.round(xs[idx]).astype(np.int64), 0, Ws - 1)
        yi = np.clip(np.round(ys[idx]).astype(np.int64), 0, Hs - 1)
        out[idx] = img[yi, xi]
    return (out[:, :, 0] if squeeze else out), ok


def measure_collinearity(points):
    """
    共线性度量：给一组理论上共线的点，返回 (最大垂距, 主方向, 中心)
    用于验证"射影变换把直线映成直线"
    """
    p = np.asarray(points, dtype=np.float64)
    c = p.mean(axis=0)
    _, _, vt = np.linalg.svd(p - c)
    d = vt[0]
    nvec = np.array([-d[1], d[0]])
    dev = np.abs((p - c) @ nvec)
    return float(dev.max()), d, c
