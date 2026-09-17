# -*- coding: utf-8 -*-
"""
鼠标交互式角点标定工具

用法
    python pick_corners.py --image data/oblique/my_photo.jpg --max-side 700 \
                           --out data/corners/my_photo.json

操作
    左键点击 依次标 4 个角点：左上 → 右上 → 右下 → 左下（顺序必须一致）
    u 撤销上一个点      r 清空重来      s 保存并退出      q / Esc 退出

说明
    1. 若指定了 --max-side，程序会先降采样再显示，保存的坐标是【降采样后】的
       像素坐标，与 main.py --max-side 保持一致即可直接使用。
    2. 无图形界面的环境（远程/无显示器）下 cv2.imshow 会失败，
       此时请手动在 data/corners/<name>.txt 中写 8 个数字（4 个点 x,y）。
"""
from __future__ import annotations

import argparse
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import preprocess as pp

ORDER = ["tl 左上", "tr 右上", "br 右下", "bl 左下"]


def main(argv=None):
    import cv2

    ap = argparse.ArgumentParser(description="交互式角点标定")
    ap.add_argument("--image", required=True)
    ap.add_argument("--max-side", type=int, default=700)
    ap.add_argument("--out", default=None)
    args = ap.parse_args(argv)

    max_side = None if args.max_side <= 0 else args.max_side
    prep = pp.load_image(args.image, max_side=max_side)
    img = prep["bgr"].copy()
    scale = prep["scale"]
    W, H = prep["size"]

    out_path = args.out or os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
        "data", "corners",
        os.path.splitext(os.path.basename(args.image))[0] + ".json")

    pts = []

    def on_mouse(event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN and len(pts) < 4:
            pts.append((float(x), float(y)))
            print(f"  点 {len(pts)} = {ORDER[len(pts) - 1]}  ({x}, {y})")

    win = "pick 4 corners: tl -> tr -> br -> bl  (u undo, r reset, s save, q quit)"
    cv2.namedWindow(win, cv2.WINDOW_AUTOSIZE)
    cv2.setMouseCallback(win, on_mouse)

    print(f"图像: {args.image}")
    print(f"原图 {prep['orig_size'][0]}x{prep['orig_size'][1]} → 显示尺寸 {W}x{H}"
          f"（scale={scale:.4f}）")
    print("请依次点击 左上 → 右上 → 右下 → 左下")

    while True:
        canvas = img.copy()
        for i, p in enumerate(pts):
            cv2.circle(canvas, (int(p[0]), int(p[1])), 5, (0, 255, 255), -1)
            cv2.putText(canvas, ORDER[i][:2], (int(p[0]) + 8, int(p[1]) - 8),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2, cv2.LINE_AA)
        if len(pts) >= 2:
            poly = np.round(np.array(pts)).astype(np.int32).reshape(-1, 1, 2)
            cv2.polylines(canvas, [poly], len(pts) == 4, (0, 0, 255), 2)
        cv2.imshow(win, canvas)
        k = cv2.waitKey(20) & 0xFF
        if k == ord("q") or k == 27:
            print("退出，未保存")
            break
        if k == ord("u") and pts:
            pts.pop()
            print("撤销一个点")
        if k == ord("r"):
            pts.clear()
            print("已清空")
        if k == ord("s"):
            if len(pts) != 4:
                print(f"还差 {4 - len(pts)} 个点，无法保存")
                continue
            pp.save_corners(out_path, np.array(pts), meta={
                "image": os.path.relpath(args.image).replace("\\", "/"),
                "image_size": [W, H],
                "downsample_scale": scale,
                "source": "manual",
            })
            print(f"已保存：{out_path}")
            break
    cv2.destroyAllWindows()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
