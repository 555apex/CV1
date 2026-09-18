# -*- coding: utf-8 -*-
"""
图像预处理与矩阵导出模块

要点
    1. Windows 下 OpenCV 的 imread/imwrite 不支持含中文的路径，
       因此统一用 np.fromfile + cv2.imdecode 读、cv2.imencode + tofile 写。
    2. 降采样后，角点坐标必须以"降采样后的像素坐标系"为准。
       约定：先 resize 再标定，或标定后按 scale 换算（scale_corners）。
    3. 像素整点约定：宽 W 的图 x ∈ 0..W-1，高 H 的图 y ∈ 0..H-1。
"""
from __future__ import annotations

import json
import os

import cv2
import numpy as np


# ------------------------------------------------------------------ 读写（兼容中文路径）

def imread_unicode(path, flags=cv2.IMREAD_COLOR):
    """兼容中文路径的读图"""
    if not os.path.isfile(path):
        raise FileNotFoundError(f"图片不存在：{path}")
    data = np.fromfile(path, dtype=np.uint8)
    if data.size == 0:
        raise ValueError(f"图片为空或不可读：{path}")
    img = cv2.imdecode(data, flags)
    if img is None:
        raise ValueError(f"解码失败：{path}")
    return img


def imwrite_unicode(path, img):
    """兼容中文路径的写图"""
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    ext = os.path.splitext(path)[1] or ".png"
    out = np.clip(np.round(np.asarray(img, dtype=np.float64)), 0, 255).astype(np.uint8)
    ok, buf = cv2.imencode(ext, out)
    if not ok:
        raise IOError(f"编码失败：{path}")
    buf.tofile(path)
    return True


# ------------------------------------------------------------------ 预处理

def load_image(path, max_side=None, color=True):
    """
    读图并（可选）降采样以简化运算

    返回 dict:
        bgr    : 读入的图（color=False 时为灰度）
        gray   : 灰度图（uint8）
        scale  : 相对原图的缩放因子，new_size = round(orig_size * scale)
        orig_size : (W0, H0)
        size      : (W, H) 降采样后尺寸
    注意：返回的坐标一律在"降采样后"的像素坐标系中。
    """
    raw = imread_unicode(path, cv2.IMREAD_COLOR if color else cv2.IMREAD_GRAYSCALE)
    H0, W0 = raw.shape[:2]
    scale = 1.0
    if max_side is not None and max(H0, W0) > max_side:
        scale = float(max_side) / float(max(H0, W0))
        new_size = (int(round(W0 * scale)), int(round(H0 * scale)))
        raw = cv2.resize(raw, new_size, interpolation=cv2.INTER_AREA)

    if color:
        bgr = raw
        gray = cv2.cvtColor(raw, cv2.COLOR_BGR2GRAY)
    else:
        gray = raw
        bgr = cv2.cvtColor(raw, cv2.COLOR_GRAY2BGR)

    H, W = gray.shape[:2]
    return {
        "bgr": bgr,
        "gray": gray,
        "scale": scale,
        "scale_xy": (W / W0, H / H0),
        "orig_size": (W0, H0),
        "size": (W, H),
        "path": path,
    }


def scale_corners(corners, scale):
    """把原图坐标系下的角点换算到降采样后的坐标系"""
    return np.asarray(corners, dtype=np.float64).reshape(-1, 2) * float(scale)


def resize_corners(corners, orig_size, new_size):
    """Transform corners using the pixel-center convention of cv2.resize."""
    p = np.asarray(corners, dtype=np.float64).reshape(-1, 2)
    W0, H0 = map(float, orig_size)
    W1, H1 = map(float, new_size)
    if min(W0, H0, W1, H1) <= 0:
        raise ValueError("resize dimensions must be positive")
    out = p.copy()
    out[:, 0] = (p[:, 0] + 0.5) * (W1 / W0) - 0.5
    out[:, 1] = (p[:, 1] + 0.5) * (H1 / H0) - 0.5
    return out


def validate_quad(corners, tol=1e-9):
    """Validate a tl,tr,br,bl quadrilateral and return issue strings."""
    p = np.asarray(corners, dtype=np.float64)
    if p.shape != (4, 2):
        return [f"corners must have shape (4,2), got {p.shape}"]
    if not np.isfinite(p).all():
        return ["corners contain NaN/Inf"]
    issues = []
    for i in range(4):
        for j in range(i + 1, 4):
            if np.linalg.norm(p[i] - p[j]) <= tol:
                issues.append(f"corner {i} and {j} are coincident or too close")
    cross = []
    for i in range(4):
        a = p[(i + 1) % 4] - p[i]
        b = p[(i + 2) % 4] - p[(i + 1) % 4]
        cross.append(float(a[0] * b[1] - a[1] * b[0]))
    if any(abs(v) <= tol for v in cross):
        issues.append("quadrilateral has a nearly collinear edge")
    if cross and not (all(v > tol for v in cross) or all(v < -tol for v in cross)):
        issues.append("corner order is not a convex, non-self-intersecting polygon")
    area2 = float(sum(p[i, 0] * p[(i + 1) % 4, 1] -
                      p[(i + 1) % 4, 0] * p[i, 1] for i in range(4)))
    if abs(area2) <= tol:
        issues.append("quadrilateral area is too small")
    return issues


def check_corners_in_image(corners, size, tol=1e-6):
    """检查角点是否落在图像范围内，返回越界点列表"""
    W, H = size
    bad = []
    for i, (x, y) in enumerate(np.asarray(corners, dtype=np.float64).reshape(-1, 2)):
        if not (-tol <= x <= W - 1 + tol and -tol <= y <= H - 1 + tol):
            bad.append((i, float(x), float(y)))
    return bad


# ------------------------------------------------------------------ 角点文件

def load_corners(path):
    """
    读角点：支持 .json（{"corners": [[x,y]×4]}）与纯文本（8 个数或 4 行）
    点序必须为 左上 → 右上 → 右下 → 左下
    """
    if not os.path.isfile(path):
        raise FileNotFoundError(f"角点文件不存在：{path}")
    if path.lower().endswith(".json"):
        with open(path, "r", encoding="utf-8") as f:
            obj = json.load(f)
        pts = obj["corners"] if isinstance(obj, dict) else obj
    else:
        with open(path, "r", encoding="utf-8") as f:
            txt = f.read()
        nums = [float(v) for v in txt.replace(",", " ").split()]
        if len(nums) != 8:
            raise ValueError(f"角点文件应含 8 个数值，实际 {len(nums)} 个：{path}")
        pts = np.array(nums, dtype=np.float64).reshape(4, 2)
    pts = np.asarray(pts, dtype=np.float64).reshape(4, 2)
    if not np.isfinite(pts).all():
        raise ValueError(f"角点文件包含 NaN/Inf：{path}")
    return pts


def load_corners_meta(path):
    """Load corners and retain optional JSON provenance metadata."""
    pts = load_corners(path)
    meta = {}
    if path.lower().endswith(".json"):
        with open(path, "r", encoding="utf-8") as f:
            obj = json.load(f)
        if isinstance(obj, dict):
            meta = obj
    return pts, meta


def save_corners(path, corners, meta=None):
    """保存角点（json 格式，附带元信息便于复现）"""
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    obj = {
        "order": "tl, tr, br, bl (左上→右上→右下→左下)",
        "coordinate_system": "降采样后的像素坐标，整点约定 0..W-1 / 0..H-1",
        "corners": np.asarray(corners, dtype=np.float64).reshape(4, 2).tolist(),
    }
    if meta:
        obj.update(meta)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)
    return path


# ------------------------------------------------------------------ 矩阵导出

def matrix_info(m):
    """像素矩阵的元信息（报告里要输出的那些量）"""
    m = np.asarray(m)
    return {
        "shape": list(m.shape),
        "dtype": str(m.dtype),
        "min": float(m.min()),
        "max": float(m.max()),
        "mean": round(float(m.mean()), 6),
        "median": float(np.median(m)),
        "std": round(float(m.std()), 6),
    }


def save_matrix(m, path):
    """保存完整矩阵为 .npy（保留原始数值，可复查）"""
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    np.save(path, np.asarray(m))
    return path


def downsample_matrix(m, size=16):
    """把矩阵降采样成 size×size 的数值表（面积平均），供报告里展示"""
    m = np.asarray(m, dtype=np.float64)
    if m.ndim == 3:
        m = m.mean(axis=2)
    H, W = m.shape
    ys = np.linspace(0, H, size + 1).astype(int)
    xs = np.linspace(0, W, size + 1).astype(int)
    out = np.zeros((size, size), dtype=np.float64)
    for i in range(size):
        for j in range(size):
            block = m[ys[i]:max(ys[i + 1], ys[i] + 1), xs[j]:max(xs[j + 1], xs[j] + 1)]
            out[i, j] = block.mean() if block.size else 0.0
    return out


def save_preview_csv(m, path, size=16):
    """导出降采样后的数值表（CSV），避免生成几十万行的完整矩阵"""
    table = downsample_matrix(m, size=size)
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    np.savetxt(path, np.round(table, 2), fmt="%.2f", delimiter=",")
    return path


def markdown_table(m, size=8, fmt="{:.1f}"):
    """把降采样矩阵渲染成 markdown 表格（写进报告）"""
    table = downsample_matrix(m, size=size)
    lines = ["| " + " | ".join([""] + [f"c{j}" for j in range(size)]) + " |",
             "|" + "---|" * (size + 1)]
    for i in range(size):
        row = [f"r{i}"] + [fmt.format(v) for v in table[i]]
        lines.append("| " + " | ".join(row) + " |")
    return "\n".join(lines)


# ------------------------------------------------------------------ 图像质量指标

def mse(a, b):
    a = np.asarray(a, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)
    return float(((a - b) ** 2).mean())


def psnr(a, b, data_range=255.0):
    e = mse(a, b)
    if e <= 0:
        return float("inf")
    return float(10.0 * np.log10((data_range ** 2) / e))
