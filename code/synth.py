# -*- coding: utf-8 -*-
"""
合成数据模块

为什么需要合成数据
    真实照片只能做"定性"展示——我们不知道它真实的 H。只有合成数据能提供
    解析真值 H0，从而定量回答"解出的 H 对不对、误差多少"。

关键设计：真值 H0 必须解析构造，不能用本项目的求解器反推（否则实验自证）
    H0 = A_out · P · R · A_in
        A_in  : 文档像素坐标 → 中心化归一化坐标
        R     : 绕中心的旋转
        P     : 透视（keystone）矩阵 [[1,0,0],[0,1,0],[p,q,1]]，det=1，恒非奇异
        A_out : 归一化坐标 → 斜视图画布像素坐标（缩放 + 平移，让四边形居中）
    全程只有矩阵乘法，不含任何 DLT/SVD 求解，因此是真正的"外部真值"。

斜视图的生成默认用 cv2.warpPerspective（作为独立实现），
以便后续用它交叉验证我们手写的逆映射重采样。
"""
from __future__ import annotations

import json
import os

import cv2
import numpy as np

from homography import apply_homography, invert_homography, normalize_h
from warp import warp_bilinear

BG = 255


# ------------------------------------------------------------------ 正向文档图

def make_document_image(W=600, H=800):
    """生成一张类文档的正向灰度图（含网格、图形、ASCII 文字、四角标记）"""
    img = np.full((H, W), BG, dtype=np.uint8)

    # 外边框
    cv2.rectangle(img, (12, 12), (W - 13, H - 13), 0, 3)

    # 网格（细线）
    for x in range(40, W - 20, 40):
        cv2.line(img, (x, 20), (x, H - 20), 200, 1)
    for y in range(40, H - 20, 40):
        cv2.line(img, (20, y), (W - 20, y), 200, 1)

    # 标题与文字（Hershey 字体只支持 ASCII，避免中文字体缺失问题）
    cv2.putText(img, "HOMOGRAPHY  DLT  TEST", (60, 90),
                cv2.FONT_HERSHEY_SIMPLEX, 1.0, 0, 2, cv2.LINE_AA)
    cv2.putText(img, "A1  B2  C3  D4", (60, 150),
                cv2.FONT_HERSHEY_SIMPLEX, 1.0, 0, 2, cv2.LINE_AA)
    cv2.putText(img, "0  1  2  3  4  5  6  7  8  9", (60, 210),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, 40, 2, cv2.LINE_AA)
    cv2.putText(img, "perspective  rectification", (60, H - 60),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, 60, 2, cv2.LINE_AA)

    # 棋盘格（一眼看出畸变）
    x0, y0, cell = 60, 260, 30
    for i in range(5):
        for j in range(5):
            if (i + j) % 2 == 0:
                cv2.rectangle(img, (x0 + j * cell, y0 + i * cell),
                              (x0 + (j + 1) * cell, y0 + (i + 1) * cell), 0, -1)

    # 同心圆
    cx, cy = W - 150, 480
    for r in range(20, 110, 20):
        cv2.circle(img, (cx, cy), r, 0 if (r // 20) % 2 else 120, 2)

    # 斜线与水平长线（用于验证"直线仍为直线"）
    for k in range(6):
        cv2.line(img, (60, 620 + k * 14), (W - 60, 560 + k * 14), 0, 1)

    # 四角标记（便于人工核对角点标定）
    for (px, py) in [(0, 0), (W - 1, 0), (W - 1, H - 1), (0, H - 1)]:
        cv2.circle(img, (int(px), int(py)), 9, 0, -1)
        cv2.circle(img, (int(px), int(py)), 13, 0, 2)
    return img


# ------------------------------------------------------------------ 解析真值 H0

def analytic_homography(doc_size, canvas_size=(700, 820), p=0.30, q=0.18,
                        rot_deg=-7.0, margin=48.0):
    """
    解析构造"文档 → 斜视图"的单应矩阵真值 H0（不含任何求解过程）

    返回 (H0, info)
        H0   : 3×3，单位 Frobenius 范数
        info : 含各分块矩阵、斜视图画布尺寸、文档四角与斜视图中四角的坐标
    """
    Wd, Hd = doc_size
    Wc, Hc = canvas_size

    # A_in: 文档像素 → 中心化归一化坐标（以 Wd 为统一尺度，保持宽高比）
    A_in = np.array([[1.0 / Wd, 0.0, -0.5],
                     [0.0, 1.0 / Wd, -0.5 * Hd / Wd],
                     [0.0, 0.0, 1.0]], dtype=np.float64)

    # R: 绕中心旋转
    th = np.deg2rad(rot_deg)
    R = np.array([[np.cos(th), -np.sin(th), 0.0],
                  [np.sin(th), np.cos(th), 0.0],
                  [0.0, 0.0, 1.0]], dtype=np.float64)

    # P: 透视 keystone，det = 1，恒非奇异
    P = np.array([[1.0, 0.0, 0.0],
                  [0.0, 1.0, 0.0],
                  [p, q, 1.0]], dtype=np.float64)

    H_norm = P @ R

    # 先求映射后的归一化四边形，据此确定 A_out（缩放 + 平移）
    doc_corners = np.array([[0.0, 0.0], [Wd - 1.0, 0.0],
                            [Wd - 1.0, Hd - 1.0], [0.0, Hd - 1.0]], dtype=np.float64)
    doc_corners_n = apply_homography(A_in, doc_corners)
    quad_n = apply_homography(H_norm, doc_corners_n)

    xmin, ymin = quad_n.min(axis=0)
    xmax, ymax = quad_n.max(axis=0)
    wn, hn = max(xmax - xmin, 1e-9), max(ymax - ymin, 1e-9)
    k = min((Wc - 2 * margin) / wn, (Hc - 2 * margin) / hn)
    cx_n, cy_n = (xmin + xmax) / 2.0, (ymin + ymax) / 2.0
    tx, ty = Wc / 2.0 - k * cx_n, Hc / 2.0 - k * cy_n

    A_out = np.array([[k, 0.0, tx],
                      [0.0, k, ty],
                      [0.0, 0.0, 1.0]], dtype=np.float64)

    H0 = A_out @ H_norm @ A_in
    H0 = normalize_h(H0, "fro")

    quad_oblique = apply_homography(H0, doc_corners)
    info = {
        "doc_size": [Wd, Hd],
        "canvas_size": [Wc, Hc],
        "params": {"p": p, "q": q, "rot_deg": rot_deg},
        "A_in": A_in.tolist(),
        "R": R.tolist(),
        "P": P.tolist(),
        "A_out": A_out.tolist(),
        "H0": H0.tolist(),
        "doc_corners": doc_corners.tolist(),
        "quad_oblique": quad_oblique.tolist(),
    }
    return H0, info


# ------------------------------------------------------------------ 生成斜视图

def make_oblique_image(doc_img, H0, canvas_size, method="cv2", fill=BG):
    """
    由正向文档图与真值 H0 生成斜视图

    method='cv2'  : 用 cv2.warpPerspective（独立实现，便于交叉验证）
    method='self' : 用本项目手写的逆映射 + 双线性插值
    """
    Wc, Hc = canvas_size
    if method == "cv2":
        M = np.asarray(H0, dtype=np.float64)
        out = cv2.warpPerspective(doc_img, M, (Wc, Hc),
                                  flags=cv2.INTER_LINEAR, borderValue=fill)
        return out
    H_inv = invert_homography(H0)
    out, _ = warp_bilinear(doc_img, H_inv, Wc, Hc, fill=fill)
    return out


def add_gaussian_noise(img, sigma=3.0, rng=None):
    """加高斯噪声，模拟真实拍摄"""
    rng = rng or np.random.default_rng(0)
    out = np.asarray(img, dtype=np.float64) + rng.normal(0.0, sigma, np.asarray(img).shape)
    return np.clip(out, 0, 255)


# ------------------------------------------------------------------ 数据集构建

def build_dataset(root, configs=None, max_side_hint=700):
    """
    生成整套合成数据，落盘到
        data/reference/doc_gray.png      正向文档图
        data/reference/H0.json           真值 H0 及构造分解
        data/oblique/<name>.png          斜视图
        data/corners/<name>.json         斜视图中文档四角坐标（模拟人工标定结果）

    返回 list[dict] 数据集索引
    """
    if configs is None:
        configs = [
            {"name": "synth_oblique", "p": 0.30, "q": 0.18, "rot_deg": -7.0,
             "canvas_size": (700, 820), "noise": 2.0},
            {"name": "synth_oblique_hard", "p": -0.42, "q": 0.34, "rot_deg": 11.0,
             "canvas_size": (720, 800), "noise": 4.0},
            {"name": "synth_oblique_clean", "p": 0.25, "q": 0.15, "rot_deg": -5.0,
             "canvas_size": (900, 1100), "noise": 0.0},
        ]

    ref_dir = os.path.join(root, "data", "reference")
    obl_dir = os.path.join(root, "data", "oblique")
    cor_dir = os.path.join(root, "data", "corners")
    for d in (ref_dir, obl_dir, cor_dir):
        os.makedirs(d, exist_ok=True)

    doc = make_document_image()
    doc_path = os.path.join(ref_dir, "doc_gray.png")
    _imwrite(doc_path, doc)

    index = []
    rng = np.random.default_rng(20260917)
    for cfg in configs:
        H0, info = analytic_homography(
            doc_size=(doc.shape[1], doc.shape[0]),
            canvas_size=cfg["canvas_size"],
            p=cfg["p"], q=cfg["q"], rot_deg=cfg["rot_deg"])
        obl = make_oblique_image(doc, H0, cfg["canvas_size"], method="cv2")
        if cfg.get("noise"):
            obl = add_gaussian_noise(obl, cfg["noise"], rng).astype(np.uint8)
        obl_path = os.path.join(obl_dir, cfg["name"] + ".png")
        _imwrite(obl_path, obl)

        quad = np.asarray(info["quad_oblique"], dtype=np.float64)
        corners_path = os.path.join(cor_dir, cfg["name"] + ".json")
        meta = {
            "image": os.path.relpath(obl_path, root).replace("\\", "/"),
            "image_size": list(cfg["canvas_size"]),
            "source": "synthetic",
            "note": "四角坐标由解析真值 H0 施加于文档四角得到，坐标系为斜视图原始像素坐标",
        }
        with open(corners_path, "w", encoding="utf-8") as f:
            json.dump({"order": "tl, tr, br, bl", "corners": quad.tolist(), **meta},
                      f, ensure_ascii=False, indent=2)

        with open(os.path.join(ref_dir, f"{cfg['name']}_H0.json"), "w", encoding="utf-8") as f:
            json.dump(info, f, ensure_ascii=False, indent=2)

        index.append({
            "name": cfg["name"],
            "image": obl_path,
            "corners": corners_path,
            "canvas_size": list(cfg["canvas_size"]),
            "doc_size": [doc.shape[1], doc.shape[0]],
            "noise": float(cfg.get("noise") or 0.0),
            "H0": np.asarray(H0, dtype=np.float64),
        })
    return {"doc": doc, "doc_path": doc_path, "items": index}


def _imwrite(path, img):
    """兼容中文路径的写图（避免 cv2.imwrite 在中文目录下失败）"""
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    out = np.clip(np.round(np.asarray(img, dtype=np.float64)), 0, 255).astype(np.uint8)
    ok, buf = cv2.imencode(os.path.splitext(path)[1] or ".png", out)
    if not ok:
        raise IOError(f"编码失败：{path}")
    buf.tofile(path)
