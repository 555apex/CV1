# -*- coding: utf-8 -*-
"""
端到端主流程：斜视图 → 预处理 → 角点 → 求解 H → 正视图 → 输出

用法
    # 单张
    python main.py --image data/oblique/synth_oblique.png \
                   --corners data/corners/synth_oblique.json --max-side 700

    # 批量（data/oblique/*.png 配 data/corners/*.json）
    python main.py --batch

    # 生成合成数据 + 跑全部对照实验
    python main.py --synth --experiments

输出（见 <out>/ 目录）
    images/   斜视图（含角点标记）、正视图（双线性）、最近邻对照、512×512 对照
    matrices/ *.npy 完整像素矩阵
    preview/  *.csv 降采样后的数值表
    metrics/  *.json 分辨率、角点、H、数值诊断、指标、矩阵元信息
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import preprocess as pp
import synth as sy
from homography import (apply_homography, build_A, check_point_configuration,
                        diagnose, homography_distance, invert_homography,
                        normalize_h, normalize_points, reprojection_error,
                        solve_homography_8dof, solve_homography_svd)
from warp import estimate_target_size, target_corners, warp_bilinear, warp_nearest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def _path_label(path, base=ROOT):
    """Return a stable relative path when possible, otherwise an absolute path."""
    path = os.path.abspath(path)
    base = os.path.abspath(base)
    try:
        if os.path.commonpath([path, base]) == base:
            return os.path.relpath(path, base).replace("\\", "/")
    except ValueError:
        pass
    return path.replace("\\", "/")


# ------------------------------------------------------------------ 单张流程

def run_one(image_path, corners_path, out_root, max_side=700, fixed_size=None,
            tag=None, truth=None):
    """
    单张图的完整流程。返回 metrics dict。

    参数
        max_side   : 降采样后最长边（None 表示不降采样）
        fixed_size : (W,H) 强制目标尺寸；None 表示由四边形对边平均长度估计
        truth      : 合成数据的解析真值 (H0, doc_size)，真实照片传 None
    """
    name = tag or os.path.splitext(os.path.basename(image_path))[0]
    prep = pp.load_image(image_path, max_side=max_side)
    img_obl = prep["gray"]
    scale = prep["scale"]
    W_ob, H_ob = prep["size"]

    # 角点：文件里存的是原图坐标，降采样后按 scale 换算
    corners_orig, corner_meta = pp.load_corners_meta(corners_path)
    corners = pp.resize_corners(corners_orig, prep["orig_size"], prep["size"])
    quad_issues = pp.validate_quad(corners)
    if quad_issues:
        raise ValueError("invalid corner quadrilateral: " + "; ".join(quad_issues))
    # Pixel-center resize coordinates can legitimately land half a pixel
    # outside the new image when an original corner lies exactly on the edge.
    bad = pp.check_corners_in_image(corners, prep["size"], tol=0.5)
    if bad:
        raise ValueError(f"角点越界（{bad}）。请确认角点是否在降采样后的坐标系中。")
    degenerate = check_point_configuration(corners)

    # 目标尺寸与固定角点
    metadata_size = corner_meta.get("target_size_WH")
    if fixed_size:
        W_t, H_t = int(fixed_size[0]), int(fixed_size[1])
        size_mode = "fixed"
    elif metadata_size is not None:
        if (not isinstance(metadata_size, (list, tuple)) or len(metadata_size) != 2 or
                any(int(v) != v or int(v) <= 0 for v in metadata_size)):
            raise ValueError("target_size_WH metadata must contain two positive integers")
        W_t, H_t = int(metadata_size[0]), int(metadata_size[1])
        size_mode = "metadata"
    else:
        W_t, H_t = estimate_target_size(corners)
        size_mode = "estimated_from_quad"
    tc = target_corners(W_t, H_t)

    # 求解 H（斜视图 → 正视图）
    H = solve_homography_svd(corners, tc, normalize=True)
    H_nonorm = solve_homography_svd(corners, tc, normalize=False)
    H_8, info8 = solve_homography_8dof(corners, tc)

    # 归一化前后的 A（用同一状态做诊断，κ(A) 才有可比性）
    src_n, _ = normalize_points(corners)
    dst_n, _ = normalize_points(tc)
    A_norm = build_A(src_n, dst_n)
    diag = diagnose(A_norm, H)

    rmse, per_pt = reprojection_error(H, corners, tc)
    rmse_8, _ = reprojection_error(H_8, corners, tc)

    # 重采样（手写逆映射 + 双线性）
    H_inv = invert_homography(H)
    rect, valid = warp_bilinear(img_obl, H_inv, W_t, H_t, fill=255)
    rect_nn, _ = warp_nearest(img_obl, H_inv, W_t, H_t, fill=255)

    # 512×512 对照（固定目标尺寸，便于横向比较）
    H512 = solve_homography_svd(corners, target_corners(512, 512), normalize=True)
    rect512, _ = warp_bilinear(img_obl, invert_homography(H512), 512, 512, fill=255)

    # ---- 落盘
    img_dir = os.path.join(out_root, "images")
    mat_dir = os.path.join(out_root, "matrices")
    pre_dir = os.path.join(out_root, "preview")
    met_dir = os.path.join(out_root, "metrics")

    marked = _draw_corners(prep["bgr"], corners)
    pp.imwrite_unicode(os.path.join(img_dir, f"{name}_oblique.png"), marked)
    pp.imwrite_unicode(os.path.join(img_dir, f"{name}_rectified.png"), rect)
    pp.imwrite_unicode(os.path.join(img_dir, f"{name}_rectified_nearest.png"), rect_nn)
    pp.imwrite_unicode(os.path.join(img_dir, f"{name}_rectified_512.png"), rect512)

    pp.save_matrix(img_obl, os.path.join(mat_dir, f"{name}_oblique_gray.npy"))
    pp.save_matrix(rect, os.path.join(mat_dir, f"{name}_rectified_gray.npy"))
    pp.save_preview_csv(img_obl, os.path.join(pre_dir, f"{name}_oblique_preview.csv"), 16)
    pp.save_preview_csv(rect, os.path.join(pre_dir, f"{name}_rectified_preview.csv"), 16)

    metrics = {
        "name": name,
        "input_image": _path_label(image_path),
        "corners_file": _path_label(corners_path),
        "preprocess": {
            "orig_size_WH": list(prep["orig_size"]),
            "downsample_scale": scale,
            "scale_xy": list(prep["scale_xy"]),
            "size_after_preprocess_WH": [W_ob, H_ob],
            "resolution_after_preprocess": f"{W_ob} x {H_ob}",
        },
        "corners_in_processed_coords": np.asarray(corners).tolist(),
        "corner_metadata": {k: v for k, v in corner_meta.items() if k != "corners"},
        "corners_order": "tl, tr, br, bl (左上→右上→右下→左下)",
        "degenerate_check": degenerate,
        "target": {
            "size_mode": size_mode,
            "size_WH": [W_t, H_t],
            "target_corners": tc.tolist(),
        },
        "H": {
            "normalized_frobenius": H.tolist(),
            "h33_normalized": normalize_h(H, "h33").tolist(),
            "det": float(np.linalg.det(H)),
            "diagnosis": diag,
        },
        "accuracy": {
            "corner_reproj_rmse_px": rmse,
            "corner_reproj_per_point_px": np.asarray(per_pt).tolist(),
            "H_vs_no_normalization_distance": homography_distance(H, H_nonorm),
            "H_vs_8dof_distance": homography_distance(H, H_8),
            "rmse_8dof_px": rmse_8,
            "consistent_8dof": info8["consistent"],
            "cond_M_8dof": info8["cond_M"],
        },
        "matrix_info": {
            "oblique_gray": pp.matrix_info(img_obl),
            "rectified_gray": pp.matrix_info(rect),
            "valid_pixel_ratio": float(np.mean(valid)),
        },
        "outputs": {
            "matrix_npy": {
                "oblique": os.path.relpath(os.path.join(mat_dir, f"{name}_oblique_gray.npy"), out_root).replace("\\", "/"),
                "rectified": os.path.relpath(os.path.join(mat_dir, f"{name}_rectified_gray.npy"), out_root).replace("\\", "/"),
            },
            "preview_csv": {
                "oblique": os.path.relpath(os.path.join(pre_dir, f"{name}_oblique_preview.csv"), out_root).replace("\\", "/"),
                "rectified": os.path.relpath(os.path.join(pre_dir, f"{name}_rectified_preview.csv"), out_root).replace("\\", "/"),
            },
        },
    }

    if truth is not None:
        # 与解析真值比较。注意坐标系：解得的 H 定义在"降采样后"的斜图坐标系，
        # 而真值 H0 定义在原图坐标系，需先把 H 换算回原图坐标系再比较：
        #     H_orig = H @ diag(scale, scale, 1)
        # 真值方面：文档四角经 S 映射到目标四角，而 S ∘ H0^{-1} ∘ H0 = S，
        # 故斜视图(原图坐标) → 正视图 的真值即 H_truth = S @ H0^{-1}
        H0, doc_size = truth
        # Match the pixel-center affine transform used by cv2.resize and
        # pp.resize_corners: p_processed = T_resize @ p_original.
        sx, sy = prep["scale_xy"]
        tx, ty = 0.5 * sx - 0.5, 0.5 * sy - 0.5
        T_ds = np.array([[sx, 0.0, tx], [0.0, sy, ty], [0.0, 0.0, 1.0]])
        H_in_orig = H @ T_ds
        sx = (W_t - 1) / (doc_size[0] - 1)
        sy = (H_t - 1) / (doc_size[1] - 1)
        S = np.array([[sx, 0.0, 0.0], [0.0, sy, 0.0], [0.0, 0.0, 1.0]])
        H_truth = S @ invert_homography(H0)
        metrics["accuracy"]["H_vs_truth_distance"] = homography_distance(H_in_orig, H_truth)
        metrics["accuracy"]["H_original_coords"] = normalize_h(H_in_orig, "h33").tolist()
        metrics["accuracy"]["note"] = "H_vs_truth_distance 为齐次尺度与符号无关的相对距离"

    os.makedirs(met_dir, exist_ok=True)
    with open(os.path.join(met_dir, f"{name}_run.json"), "w", encoding="utf-8") as f:
        json.dump(metrics, f, ensure_ascii=False, indent=2)
    return metrics


def _resolve(path):
    """
    把命令行里的相对路径按"项目根目录"解析。

    这样在 PyCharm 里无论工作目录设成哪里，都可以直接写
        --image data/oblique/xxx.png --corners data/corners/xxx.json
    """
    if not path or os.path.isabs(path):
        return path
    cand = os.path.join(ROOT, path)
    return cand if os.path.exists(cand) else path


def _load_synth_truth(root, name):
    """若存在合成数据的解析真值，则读入 (H0, doc_size) 用于误差评估"""
    p = os.path.join(root, "data", "reference", f"{name}_H0.json")
    if os.path.isfile(p):
        with open(p, "r", encoding="utf-8") as f:
            obj = json.load(f)
        return np.asarray(obj["H0"], dtype=np.float64), tuple(obj["doc_size"])
    return None


def _draw_corners(bgr, corners):
    import cv2
    img = np.asarray(bgr).copy()
    pts = np.asarray(corners, dtype=np.float64).reshape(-1, 2)
    poly = np.round(pts).astype(np.int32).reshape(-1, 1, 2)
    cv2.polylines(img, [poly], True, (0, 0, 255), 2)
    for i, (x, y) in enumerate(pts):
        cv2.circle(img, (int(round(x)), int(round(y))), 5, (0, 255, 255), -1)
        cv2.putText(img, str(i), (int(round(x)) + 6, int(round(y)) - 6),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2, cv2.LINE_AA)
    return img


# ------------------------------------------------------------------ CLI

def main(argv=None):
    ap = argparse.ArgumentParser(description="射影矩阵自编程实现 —— 端到端流程")
    ap.add_argument("--image", help="斜视图路径")
    ap.add_argument("--corners", help="角点文件路径（txt/json）")
    ap.add_argument("--batch", action="store_true", help="批量跑 data/oblique 下所有图")
    ap.add_argument("--synth", action="store_true", help="先生成合成数据")
    ap.add_argument("--experiments", action="store_true", help="跑全部对照实验")
    ap.add_argument("--max-side", type=int, default=700, help="降采样后最长边，0 表示不降采样")
    ap.add_argument("--fixed-size", default=None, help="强制目标尺寸，如 512x512")
    ap.add_argument("--out", default=os.path.join(ROOT, "output"), help="输出目录")
    args = ap.parse_args(argv)
    args.image = _resolve(args.image)
    args.corners = _resolve(args.corners)
    args.out = os.path.abspath(args.out or os.path.join(ROOT, "output"))

    max_side = None if (args.max_side is None or args.max_side <= 0) else args.max_side
    fixed = None
    if args.fixed_size:
        try:
            w, h = args.fixed_size.lower().split("x")
            fixed = (int(w), int(h))
            if fixed[0] <= 0 or fixed[1] <= 0:
                raise ValueError
        except ValueError as e:
            raise SystemExit("--fixed-size must be a positive WxH value, e.g. 512x512") from e

    if args.synth:
        dataset = sy.build_dataset(ROOT, force=True)
        print(f"[合成数据] 参考图: {dataset['doc_path']}")
        for it in dataset["items"]:
            print(f"           {os.path.basename(it['image'])}  角点: {it['corners']}")

    if args.experiments:
        import experiments
        summary = experiments.run_all(ROOT, args.out)
        print("\n[对照实验] 完成，指标见 output/metrics/experiments.json")
        e1 = summary["exp1"]
        print(f"  exp1 齐次SVD重投影RMSE = {e1['solve_svd_normalized']['reproj_rmse_px']:.3e} px")
        print(f"       8元解重投影RMSE   = {e1['solve_8dof']['reproj_rmse_px']:.3e} px")
        print(f"       两解距离          = {e1['H_svd_vs_8dof_distance']:.3e}")
        e3 = summary["exp3"]
        print(f"  exp3 h33=0反例: 齐次解RMSE = {e3['solve_svd']['reproj_rmse_px']:.3e} px, "
              f"8元解RMSE = {e3['solve_8dof']['reproj_rmse_px']:.3e} px, "
              f"cond(M) = {e3['cond_M_8dof']:.3e}")

    if args.batch:
        pairs = []
        image_paths = []
        for ext in ("*.png", "*.jpg", "*.jpeg", "*.bmp"):
            image_paths.extend(glob.glob(os.path.join(ROOT, "data", "oblique", ext)))
        for img_path in sorted(image_paths):
            stem = os.path.splitext(os.path.basename(img_path))[0]
            for ext in (".json", ".txt"):
                cp = os.path.join(ROOT, "data", "corners", stem + ext)
                if os.path.isfile(cp):
                    pairs.append((img_path, cp))
                    break
        if not pairs:
            print("[批量] 未找到 图片 + 角点 的配对，请先 --synth 或把实拍图放进 data/oblique/")
        for img_path, cp in pairs:
            name = os.path.splitext(os.path.basename(img_path))[0]
            m = run_one(img_path, cp, args.out, max_side=max_side, fixed_size=fixed,
                        tag=name, truth=_load_synth_truth(ROOT, name))
            _print_run(m)

    if args.image:
        if not args.corners:
            raise SystemExit("单张模式需要同时提供 --corners")
        name = os.path.splitext(os.path.basename(args.image))[0]
        m = run_one(args.image, args.corners, args.out, max_side=max_side, fixed_size=fixed,
                    tag=name, truth=_load_synth_truth(ROOT, name))
        _print_run(m)
    return 0


def _print_run(m):
    print("-" * 78)
    print(f"[{m['name']}]")
    p = m["preprocess"]
    print(f"  原图分辨率      : {p['orig_size_WH'][0]} x {p['orig_size_WH'][1]}"
          f"   降采样系数 {p['downsample_scale']:.4f}")
    print(f"  预处理后分辨率  : {p['resolution_after_preprocess']}")
    print("  四个角点坐标    :")
    for i, (x, y) in enumerate(m["corners_in_processed_coords"]):
        print(f"      P{i} = ({x:9.3f}, {y:9.3f})")
    print(f"  正视图分辨率    : {m['target']['size_WH'][0]} x {m['target']['size_WH'][1]}"
          f"  ({m['target']['size_mode']})")
    print(f"  H (h33归一化)   :")
    for row in m["H"]["h33_normalized"]:
        print("      [" + ", ".join(f"{v:12.6f}" for v in row) + "]")
    a = m["accuracy"]
    print(f"  角点重投影RMSE  : {a['corner_reproj_rmse_px']:.3e} px")
    diag = m["H"]["diagnosis"]
    print(f"  cond(A)非零谱    : {diag['condition_number_nonzero']:.3e}"
          f"   rank/nullity = {diag['rank']}/{diag['nullity']}")
    print(f"  斜图矩阵        : {m['matrix_info']['oblique_gray']['shape']} "
          f"min/max/mean = {m['matrix_info']['oblique_gray']['min']:.0f}/"
          f"{m['matrix_info']['oblique_gray']['max']:.0f}/{m['matrix_info']['oblique_gray']['mean']:.2f}")
    print(f"  正视图矩阵      : {m['matrix_info']['rectified_gray']['shape']} "
          f"min/max/mean = {m['matrix_info']['rectified_gray']['min']:.0f}/"
          f"{m['matrix_info']['rectified_gray']['max']:.0f}/{m['matrix_info']['rectified_gray']['mean']:.2f}")
    if "H_vs_truth_distance" in a:
        print(f"  与真值H的距离   : {a['H_vs_truth_distance']:.3e}")


if __name__ == "__main__":
    raise SystemExit(main())
