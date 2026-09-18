# -*- coding: utf-8 -*-
"""Compare public-sample rectification with SmartDoc reference images."""
from __future__ import annotations

import argparse
import csv
import json
import os
import sys

import cv2
import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def _read_gray(path: str) -> np.ndarray:
    if not os.path.isfile(path):
        raise FileNotFoundError(path)
    encoded = np.fromfile(path, dtype=np.uint8)
    image = cv2.imdecode(encoded, cv2.IMREAD_GRAYSCALE)
    if image is None:
        raise ValueError(f"unable to decode image: {path}")
    return image.astype(np.float64)


def _compare(a: np.ndarray, b: np.ndarray) -> dict:
    if a.shape != b.shape:
        raise ValueError(f"shape mismatch: {a.shape} vs {b.shape}")
    diff = a - b
    mse = float(np.mean(diff * diff))
    return {
        "mse_gray": mse,
        "mae_gray": float(np.mean(np.abs(diff))),
        "psnr_gray_db": float("inf") if mse == 0 else float(10.0 * np.log10(255.0 ** 2 / mse)),
    }


def evaluate(out_root: str, corners_dir: str, reference_dir: str) -> dict:
    rows = []
    for corner_path in sorted(glob_json(corners_dir, "smartdoc_*.json")):
        name = os.path.splitext(os.path.basename(corner_path))[0]
        run_path = os.path.join(out_root, "metrics", f"{name}_run.json")
        rect_path = os.path.join(out_root, "images", f"{name}_rectified.png")
        with open(run_path, "r", encoding="utf-8") as handle:
            run = json.load(handle)
        rect = _read_gray(rect_path)
        dewarped_path = os.path.join(reference_dir, f"{name.replace('smartdoc_', 'smartdoc_')}_dewarped.png")
        ground_truth_path = os.path.join(reference_dir, f"{name}_ground_truth.png")
        dewarped = _read_gray(dewarped_path)
        ground_truth = _read_gray(ground_truth_path)
        row = {
            "name": name,
            "target_size_WH": run["target"]["size_WH"],
            "corner_reproj_rmse_px": run["accuracy"]["corner_reproj_rmse_px"],
            "valid_pixel_ratio": run["matrix_info"]["valid_pixel_ratio"],
            "condition_number_nonzero": run["H"]["diagnosis"]["condition_number_nonzero"],
            "dewarped": _compare(rect, dewarped),
            "ground_truth": _compare(rect, ground_truth),
            "reference_dewarped": os.path.relpath(dewarped_path, ROOT).replace("\\", "/"),
            "reference_ground_truth": os.path.relpath(ground_truth_path, ROOT).replace("\\", "/"),
        }
        rows.append(row)

    summary = {
        "dataset": "SmartDoc 2017 sample/demo release v1.0",
        "evaluation_scope": {
            "dewarped": "primary geometry/implementation consistency; same source frame after upstream perspective undo",
            "ground_truth": "secondary end-to-end image comparison; includes source-frame blur, illumination, and content differences",
            "analytic_H0": "not available for public samples",
        },
        "samples": rows,
    }
    os.makedirs(os.path.join(out_root, "metrics"), exist_ok=True)
    with open(os.path.join(out_root, "metrics", "public_summary.json"), "w", encoding="utf-8") as handle:
        json.dump(summary, handle, ensure_ascii=False, indent=2)
    _write_csv(rows, os.path.join(out_root, "metrics", "public_summary.csv"))
    _write_markdown(summary, os.path.join(out_root, "metrics", "public_analysis.md"))
    return summary


def glob_json(directory: str, pattern: str) -> list[str]:
    import glob
    return glob.glob(os.path.join(directory, pattern))


def _write_csv(rows: list[dict], path: str) -> None:
    fields = [
        "name", "target_size_WH", "corner_reproj_rmse_px", "valid_pixel_ratio",
        "condition_number_nonzero", "dewarped_mse_gray", "dewarped_psnr_gray_db",
        "ground_truth_mse_gray", "ground_truth_psnr_gray_db",
    ]
    with open(path, "w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({
                "name": row["name"],
                "target_size_WH": "x".join(map(str, row["target_size_WH"])),
                "corner_reproj_rmse_px": row["corner_reproj_rmse_px"],
                "valid_pixel_ratio": row["valid_pixel_ratio"],
                "condition_number_nonzero": row["condition_number_nonzero"],
                "dewarped_mse_gray": row["dewarped"]["mse_gray"],
                "dewarped_psnr_gray_db": row["dewarped"]["psnr_gray_db"],
                "ground_truth_mse_gray": row["ground_truth"]["mse_gray"],
                "ground_truth_psnr_gray_db": row["ground_truth"]["psnr_gray_db"],
            })


def _write_markdown(summary: dict, path: str) -> None:
    lines = [
        "# SmartDoc public-sample analysis",
        "",
        "The primary image metric is comparison with the same-frame `dewarped` reference. "
        "The `ground-truth` comparison is secondary because it also contains blur, illumination, "
        "and source-content differences.",
        "",
        "| sample | target W×H | corner RMSE (px) | dewarped PSNR (dB) | ground-truth PSNR (dB) | valid ratio |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for row in summary["samples"]:
        lines.append(
            f"| {row['name']} | {'×'.join(map(str, row['target_size_WH']))} | "
            f"{row['corner_reproj_rmse_px']:.3e} | "
            f"{row['dewarped']['psnr_gray_db']:.2f} | "
            f"{row['ground_truth']['psnr_gray_db']:.2f} | "
            f"{row['valid_pixel_ratio']:.4f} |"
        )
    lines.extend([
        "",
        "Interpretation: corner RMSE at machine precision validates the four-point homography fit. "
        "Dewarped PSNR measures implementation consistency, while ground-truth PSNR should not be "
        "used alone to attribute error to the rectification algorithm.",
    ])
    with open(path, "w", encoding="utf-8") as handle:
        handle.write("\n".join(lines) + "\n")


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", default=os.path.join(ROOT, "output", "smartdoc_sample"))
    parser.add_argument("--corners", default=os.path.join(ROOT, "data", "corners"))
    parser.add_argument("--reference", default=os.path.join(ROOT, "data", "reference"))
    args = parser.parse_args(argv)
    summary = evaluate(os.path.abspath(args.out), os.path.abspath(args.corners), os.path.abspath(args.reference))
    for row in summary["samples"]:
        print(
            f"{row['name']}: dewarped PSNR={row['dewarped']['psnr_gray_db']:.2f} dB, "
            f"ground-truth PSNR={row['ground_truth']['psnr_gray_db']:.2f} dB"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
