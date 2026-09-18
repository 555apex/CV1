# -*- coding: utf-8 -*-
"""
对照实验与指标计算模块

实验清单
    exp1  4 点情形下"齐次 SVD 解"与"h33=1 非齐次解"的等价性
    exp2  坐标归一化对条件数与精度的影响
    exp3  真实 h33=0 的反例：h33=1 约定失效，齐次 SVD 仍然正确
    exp4  n>4 加噪声：齐次 SVD 最小二乘的鲁棒性
    exp5  管线验证：直线保持性、与 cv2 独立实现的交叉验证、正视图与参考图的一致性

所有实验的真值 H0 均由 synth.analytic_homography 解析构造，不使用本项目的求解器。
"""
from __future__ import annotations

import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from homography import (apply_homography, build_A, check_point_configuration,
                        diagnose, homography_distance, invert_homography,
                        normalize_h, normalize_points, reprojection_error,
                        solve_homography_8dof, solve_homography_svd)
from warp import (estimate_target_size, measure_collinearity, target_corners,
                  warp_bilinear, warp_nearest)
import preprocess as pp
import synth as sy


# ------------------------------------------------------------------ 工具

def doc_quad(doc_size):
    Wd, Hd = doc_size
    return np.array([[0.0, 0.0], [Wd - 1.0, 0.0],
                     [Wd - 1.0, Hd - 1.0], [0.0, Hd - 1.0]], dtype=np.float64)


def _score(H, src, dst, H0):
    rmse, per = reprojection_error(H, src, dst)
    return {
        "reproj_rmse_px": rmse,
        "reproj_max_px": float(np.max(per)) if np.isfinite(per).any() else float("inf"),
        "H_distance_to_truth": homography_distance(H, H0),
    }


def _cond_of_A(src, dst, normalize):
    if normalize:
        src_n, _ = normalize_points(src)
        dst_n, _ = normalize_points(dst)
    else:
        src_n, dst_n = src, dst
    A = build_A(src_n, dst_n)
    return A, diagnose(A)["cond_A"]


# ------------------------------------------------------------------ exp1

def exp1_equivalence(doc_size, H0):
    """4 点情形：齐次 SVD（9 未知量）与 h33=1 非齐次解应当给出同一个 H"""
    src = doc_quad(doc_size)
    dst = apply_homography(H0, src)

    H_svd = solve_homography_svd(src, dst, normalize=True)
    H_svd_nonorm = solve_homography_svd(src, dst, normalize=False)
    H_8, info8 = solve_homography_8dof(src, dst, normalize=False)
    H_8_norm, info8_norm = solve_homography_8dof(src, dst, normalize=True)

    A_norm, cond_norm = _cond_of_A(src, dst, True)
    A_raw, cond_raw = _cond_of_A(src, dst, False)

    return {
        "name": "exp1_4点等价性",
        "degenerate_check": check_point_configuration(src),
        "solve_svd_normalized": _score(H_svd, src, dst, H0),
        "solve_svd_raw": _score(H_svd_nonorm, src, dst, H0),
        "solve_8dof": _score(H_8, src, dst, H0),
        "solve_8dof_normalized": _score(H_8_norm, src, dst, H0),
        "consistent_8dof": info8["consistent"],
        "consistent_8dof_normalized": info8_norm["consistent"],
        "cond_M_8dof": info8["cond_M"],
        "cond_M_8dof_normalized": info8_norm["cond_M"],
        "H_svd_vs_8dof_distance": homography_distance(H_svd, H_8),
        "H_svd_raw_vs_8dof_raw_distance": homography_distance(H_svd_nonorm, H_8),
        "H_svd_normalized_vs_8dof_normalized_distance": homography_distance(H_svd, H_8_norm),
        "cond_A_raw": cond_raw,
        "cond_A_normalized": cond_norm,
        "H_truth_h33": float(normalize_h(H0, "fro")[2, 2]),
    }


# ------------------------------------------------------------------ exp2

def exp2_normalization(doc_size, H0):
    """归一化开关对条件数与精度的影响"""
    src = doc_quad(doc_size)
    dst = apply_homography(H0, src)
    rows = []
    for normalize in (False, True):
        H = solve_homography_svd(src, dst, normalize=normalize)
        _, cond = _cond_of_A(src, dst, normalize)
        row = {"normalize": bool(normalize), "cond_A": cond}
        row.update(_score(H, src, dst, H0))
        rows.append(row)
    return {"name": "exp2_归一化影响", "rows": rows}


# ------------------------------------------------------------------ exp3

def build_h33_zero_homography():
    """
    构造一个真实 h33 = 0 的单应矩阵（几何含义：源坐标原点被映到无穷远）
        H = [[1, 0, 0.3], [0, 1, 0.2], [1e-3, 1e-3, 0]]，det ≈ -5e-4 ≠ 0
    注意：h33=0 时源点 (0,0) 无定义，因此点集必须避开原点。
    """
    H = np.array([[1.0, 0.0, 0.3],
                  [0.0, 1.0, 0.2],
                  [1e-3, 1e-3, 0.0]], dtype=np.float64)
    return normalize_h(H, "fro")


def exp3_h33_zero():
    """反例：真实 h33=0 时 h33=1 约定失效，齐次 SVD 不受影响"""
    H0z = build_h33_zero_homography()
    src = np.array([[10.0, 10.0], [110.0, 10.0], [110.0, 90.0], [10.0, 90.0]])
    dst = apply_homography(H0z, src)

    H_svd = solve_homography_svd(src, dst, normalize=True)
    H_8, info8 = solve_homography_8dof(src, dst, normalize=False)

    # h33 归一化在此时应当直接失败
    h33_fail = None
    try:
        normalize_h(H_svd, "h33")
        h33_fail = False
    except ValueError as e:
        h33_fail = str(e)

    return {
        "name": "exp3_h33为0的反例",
        "truth_H_2_2": float(H0z[2, 2]),
        "truth_det": float(np.linalg.det(H0z)),
        "solve_svd": _score(H_svd, src, dst, H0z),
        "solve_svd_h33_component": float(H_svd[2, 2]),
        "solve_8dof": _score(H_8, src, dst, H0z),
        "cond_M_8dof": info8["cond_M"],
        "linear_residual_8dof": info8["linear_residual"],
        "consistent_8dof": info8["consistent"],
        "h33_normalization_error": h33_fail,
    }


# ------------------------------------------------------------------ exp4

def exp4_noisy_least_squares(doc_size, H0, ns=(4, 5, 6, 8, 12, 20, 40),
                             sigmas=(0.0, 0.5, 2.0), trials=50, seed=20260917):
    """n>4 加噪声时齐次 SVD 最小二乘的鲁棒性（真值来自解析 H0）"""
    Wd, Hd = doc_size
    rng = np.random.default_rng(seed)
    out = []
    for sigma in sigmas:
        for n in ns:
            rmses, dists = [], []
            for _ in range(trials):
                # 在文档范围内均匀取 n 个点（避开边缘）
                src = np.column_stack([rng.uniform(20, Wd - 20, n),
                                       rng.uniform(20, Hd - 20, n)])
                dst = apply_homography(H0, src)
                if sigma > 0:
                    dst = dst + rng.normal(0.0, sigma, dst.shape)
                H = solve_homography_svd(src, dst, normalize=True)
                rmse, _ = reprojection_error(H, src, dst)
                rmses.append(rmse)
                dists.append(homography_distance(H, H0))
            def summary(values):
                values = np.asarray(values, dtype=np.float64)
                return {
                    "mean": float(np.mean(values)),
                    "std": float(np.std(values)),
                    "median": float(np.median(values)),
                    "p025": float(np.percentile(values, 2.5)),
                    "p975": float(np.percentile(values, 97.5)),
                }

            out.append({
                "sigma_px": float(sigma),
                "n_points": int(n),
                "reproj_rmse_px_mean": float(np.mean(rmses)),
                "H_distance_mean": float(np.mean(dists)),
                "H_distance_std": float(np.std(dists)),
                "reproj_rmse_px_summary": summary(rmses),
                "H_distance_summary": summary(dists),
            })
    return {"name": "exp4_n大于4加噪声", "trials": trials, "rows": out}


# ------------------------------------------------------------------ exp5

def exp5_pipeline_verification(item, out_dir):
    """
    管线验证
        1. 角点 → H_rect（斜视图 → 正视图）与解析真值 inv(H0) 的一致性
        2. 直线保持性：文档平面上的直线经 H 映射后仍为直线
        3. 与 cv2.warpPerspective 的交叉验证（独立实现）
        4. 正视图与合成参考图的一致性（PSNR）
    """
    img = pp.imread_unicode(item["image"], 0)          # 灰度读入
    quad = pp.load_corners(item["corners"])
    H0 = item["H0"]
    doc = item["doc"]
    doc_size = item["doc_size"]

    # --- 1) 与解析真值比较：目标点直接用文档四角（同尺寸约定）
    H_rect_exact = solve_homography_svd(quad, doc_quad(doc_size), normalize=True)
    H_truth_inv = invert_homography(H0)
    dist_exact = homography_distance(H_rect_exact, H_truth_inv)

    # --- 交付流程：目标尺寸由四边形对边平均长度估计
    W, H = estimate_target_size(quad)
    tc = target_corners(W, H)
    H_rect = solve_homography_svd(quad, tc, normalize=True)
    H_rect_inv = invert_homography(H_rect)

    rect_bilinear, valid = warp_bilinear(img, H_rect_inv, W, H, fill=255)
    rect_nearest, _ = warp_nearest(img, H_rect_inv, W, H, fill=255)
    rmse_corner, _ = reprojection_error(H_rect, quad, tc)

    # --- 2) 直线保持性：文档平面上的水平/竖直/斜线各取样本点，经 H 映射
    lines = {
        "horizontal": np.column_stack([np.linspace(0, doc_size[0] - 1, 40),
                                       np.full(40, (doc_size[1] - 1) * 0.4)]),
        "vertical": np.column_stack([np.full(40, (doc_size[0] - 1) * 0.3),
                                     np.linspace(0, doc_size[1] - 1, 40)]),
        "diagonal": np.column_stack([np.linspace(0, doc_size[0] - 1, 40),
                                     np.linspace(0, doc_size[1] - 1, 40)]),
    }
    line_dev = {}
    for k, pts in lines.items():
        mapped = apply_homography(H0, pts)                # 真值映射：文档 → 斜视图
        dev_truth, _, _ = measure_collinearity(mapped)
        # 再用本项目解出的 H_rect 把斜视图坐标映到正视图坐标，直线仍应是直线
        mapped2 = apply_homography(H_rect, mapped)
        dev_rect, _, _ = measure_collinearity(mapped2)
        line_dev[k] = {"truth_max_dev_px": dev_truth,
                       "via_solved_H_max_dev_px": dev_rect}

    # --- 3) 与 cv2 独立实现交叉验证
    import cv2
    ref_cv2 = cv2.warpPerspective(img, H_rect, (W, H),
                                  flags=cv2.INTER_LINEAR, borderValue=255)
    psnr_vs_cv2 = pp.psnr(rect_bilinear, ref_cv2)
    psnr_nearest_vs_bilinear = pp.psnr(rect_nearest, rect_bilinear)

    # --- 3b) 用解析真值 H 做同样的重采样，与本项目结果比较
    #     该指标把"求解误差"从"插值误差"中隔离出来：两者用同一套重采样代码，
    #     唯一的差别是 H 来自求解还是来自解析真值。
    sx = (W - 1) / (doc_size[0] - 1)
    sy = (H - 1) / (doc_size[1] - 1)
    S = np.array([[sx, 0.0, 0.0], [0.0, sy, 0.0], [0.0, 0.0, 1.0]], dtype=np.float64)
    H_truth_obl2tgt = S @ invert_homography(H0)
    rect_truth, _ = warp_bilinear(img, invert_homography(H_truth_obl2tgt), W, H, fill=255)
    psnr_vs_truth_H = pp.psnr(rect_bilinear, rect_truth)

    # --- 4) 正视图 vs 解析参考图
    # 参考图 = 把文档图按"整点约定"仿射映射到目标尺寸（与 target_corners 完全一致的约定）
    # 注意不要用 cv2.resize：它的半像素约定与整点约定不同，会带来约 0.4px 的系统性错位，
    # 在文字/网格这类高对比边缘上会显著虚低 PSNR。
    doc_resized = cv2.warpAffine(
        doc,
        np.array([[(W - 1) / (doc_size[0] - 1), 0.0, 0.0],
                  [0.0, (H - 1) / (doc_size[1] - 1), 0.0]], dtype=np.float64),
        (W, H), flags=cv2.INTER_LINEAR, borderValue=255)

    psnr_vs_doc = pp.psnr(rect_bilinear, doc_resized)

    y0, x0 = 40, 40
    hh, ww = min(H, doc_resized.shape[0]) - y0, min(W, doc_resized.shape[1]) - x0
    inner_psnr = pp.psnr(rect_bilinear[y0:y0 + hh, x0:x0 + ww],
                         doc_resized[y0:y0 + hh, x0:x0 + ww])

    # --- 出图
    os.makedirs(out_dir, exist_ok=True)
    name = item["name"]
    _fig_overview(img, quad, rect_bilinear, doc_resized, tc,
                  os.path.join(out_dir, f"fig_{name}_overview.png"))
    _fig_grid(img, H0, quad, os.path.join(out_dir, f"fig_{name}_grid.png"), doc_size=doc_size)
    _fig_warp_compare(rect_nearest, rect_bilinear, ref_cv2,
                      os.path.join(out_dir, f"fig_{name}_interp.png"))

    return {
        "name": f"exp5_管线验证_{name}",
        "target_size": [W, H],
        "corner_reproj_rmse_px": rmse_corner,
        "H_rect_vs_truth_inv_distance": dist_exact,
        "line_preservation": line_dev,
        "psnr_rect_vs_cv2": psnr_vs_cv2,
        "psnr_rect_vs_truth_H": psnr_vs_truth_H,
        "psnr_nearest_vs_bilinear": psnr_nearest_vs_bilinear,
        "noise_sigma_px": item.get("noise", 0.0),
        "psnr_rect_vs_reference_full": psnr_vs_doc,
        "psnr_rect_vs_reference_inner": inner_psnr,
        "valid_ratio": float(np.mean(valid)),
        "matrix_info_oblique": pp.matrix_info(img),
        "matrix_info_rectified": pp.matrix_info(rect_bilinear),
    }


# ------------------------------------------------------------------ 出图

def _fig_overview(img, quad, rect, doc_resized, tc, path):
    fig, axes = plt.subplots(1, 3, figsize=(15, 5.2))
    axes[0].imshow(img, cmap="gray", vmin=0, vmax=255)
    q = np.vstack([quad, quad[:1]])
    axes[0].plot(q[:, 0], q[:, 1], "-", color="red", lw=1.4)
    axes[0].plot(quad[:, 0], quad[:, 1], "o", color="red", ms=5)
    axes[0].set_title("Oblique input + calibrated corners")
    axes[1].imshow(rect, cmap="gray", vmin=0, vmax=255)
    axes[1].set_title("Rectified by self-implemented warp")
    axes[2].imshow(doc_resized, cmap="gray", vmin=0, vmax=255)
    axes[2].set_title("Reference (synthetic ground truth)")
    for ax in axes:
        ax.axis("off")
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)


def _fig_grid(img, H0, quad, path, n=12, doc_size=(600, 800)):
    """把规则网格从文档平面映射到斜视图，用于肉眼判断"直线是否仍为直线" """
    fig, ax = plt.subplots(figsize=(6, 7))
    ax.imshow(img, cmap="gray", vmin=0, vmax=255)
    W_doc, H_doc = float(doc_size[0]), float(doc_size[1])
    t = np.linspace(0, 1, 60)
    for i in range(n + 1):
        xs = np.full_like(t, i / n) * (W_doc - 1)
        ys = t * (H_doc - 1)
        pts = apply_homography(H0, np.column_stack([xs, ys]))
        ax.plot(pts[:, 0], pts[:, 1], "-", color="#1f77b4", lw=0.8)
        xs2 = t * (W_doc - 1)
        ys2 = np.full_like(t, i / n) * (H_doc - 1)
        pts2 = apply_homography(H0, np.column_stack([xs2, ys2]))
        ax.plot(pts2[:, 0], pts2[:, 1], "-", color="#1f77b4", lw=0.8)
    q = np.vstack([quad, quad[:1]])
    ax.plot(q[:, 0], q[:, 1], "-", color="red", lw=1.2)
    ax.set_title("Mapped grid: straight lines stay straight")
    ax.axis("off")
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)


def _fig_warp_compare(nearest, bilinear, cv2_ref, path):
    fig, axes = plt.subplots(2, 3, figsize=(13, 9))
    panels = [("Nearest (self)", nearest), ("Bilinear (self)", bilinear),
              ("cv2.warpPerspective", cv2_ref)]
    for ax, (title, im) in zip(axes[0], panels):
        ax.imshow(im, cmap="gray", vmin=0, vmax=255)
        ax.set_title(title)
        ax.axis("off")
    # 局部放大
    H, W = bilinear.shape[:2]
    y0, x0 = int(H * 0.62), int(W * 0.15)
    y1, x1 = min(H, y0 + 120), min(W, x0 + 200)
    for ax, (title, im) in zip(axes[1], panels):
        ax.imshow(im[y0:y1, x0:x1], cmap="gray", vmin=0, vmax=255)
        ax.set_title(f"{title} (zoom)")
        ax.axis("off")
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)


def _fig_exp2(rows, path):
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2))
    labels = ["raw coords", "Hartley normalized"]
    conds = [r["cond_A"] for r in rows]
    rmses = [max(r["reproj_rmse_px"], 1e-18) for r in rows]
    axes[0].bar(labels, conds, color=["#d62728", "#2ca02c"])
    axes[0].set_yscale("log")
    axes[0].set_title("cond(A)")
    axes[0].grid(alpha=0.3, axis="y")
    axes[1].bar(labels, rmses, color=["#d62728", "#2ca02c"])
    axes[1].set_yscale("log")
    axes[1].set_title("Reprojection RMSE (px, lower is better)")
    axes[1].grid(alpha=0.3, axis="y")
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)


def _fig_exp4(rows, path):
    fig, ax = plt.subplots(figsize=(7.5, 4.6))
    sigmas = sorted({r["sigma_px"] for r in rows})
    for s in sigmas:
        sub = [r for r in rows if r["sigma_px"] == s]
        sub.sort(key=lambda r: r["n_points"])
        ax.plot([r["n_points"] for r in sub],
                [max(r["reproj_rmse_px_mean"], 1e-14) for r in sub],
                "o-", label=f"noise σ = {s} px")
    ax.set_xlabel("number of correspondences n")
    ax.set_ylabel("reprojection RMSE (px)")
    ax.set_yscale("log")
    ax.set_title("Homogeneous SVD least squares vs n and noise")
    ax.grid(alpha=0.3, which="both")
    ax.legend()
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)


# ------------------------------------------------------------------ 总入口

def run_all(root, out_root, doc_size=(600, 800)):
    """跑完全部实验，落盘 metrics json 与图，返回汇总字典"""
    metrics_dir = os.path.join(out_root, "metrics")
    images_dir = os.path.join(out_root, "images")
    os.makedirs(metrics_dir, exist_ok=True)
    os.makedirs(images_dir, exist_ok=True)

    H0, info0 = sy.analytic_homography(doc_size=doc_size)
    summary = {"exp1": exp1_equivalence(doc_size, H0),
               "exp2": exp2_normalization(doc_size, H0),
               "exp3": exp3_h33_zero(),
               "exp4": exp4_noisy_least_squares(doc_size, H0)}

    _fig_exp2(summary["exp2"]["rows"], os.path.join(images_dir, "fig_exp2_normalization.png"))
    _fig_exp4(summary["exp4"]["rows"], os.path.join(images_dir, "fig_exp4_noise.png"))

    dataset = sy.build_dataset(root)
    exp5 = []
    for item in dataset["items"]:
        exp5.append(exp5_pipeline_verification({**item, "doc": dataset["doc"]}, images_dir))
    summary["exp5"] = exp5

    with open(os.path.join(metrics_dir, "experiments.json"), "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)
    return summary
