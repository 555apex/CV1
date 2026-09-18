# -*- coding: utf-8 -*-
"""
环境自检脚本 —— 在 PyCharm 里第一步就该跑这个

检查内容
    1. Python 版本与平台
    2. numpy / opencv-python / matplotlib 是否可导入及版本
    3. 项目目录结构是否完整
    4. 核心算法最小自检：用 4 组点求解 H 并报告重投影误差
    5. 按需生成 / 检查合成数据

用法
    python check_env.py          # 只做检查
    python check_env.py --synth  # 检查后顺便生成合成数据
"""
from __future__ import annotations

import os
import platform
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

OK = "[OK]"
BAD = "[!!]"


def main(argv=None):
    argv = argv if argv is not None else sys.argv[1:]
    print("=" * 70)
    print("第一章加分作业 —— 环境自检")
    print("=" * 70)

    # 1) Python
    print(f"\n[1] Python")
    print(f"    版本   : {sys.version.split()[0]}")
    print(f"    解释器 : {sys.executable}")
    print(f"    平台   : {platform.system()} {platform.release()} ({platform.machine()})")
    if sys.version_info < (3, 8):
        print(f"    {BAD} 需要 Python >= 3.8")

    # 2) 依赖
    print(f"\n[2] 依赖库")
    ok_deps = True
    for mod_name, pkg in (("numpy", "numpy"),
                          ("cv2", "opencv-python"),
                          ("matplotlib", "matplotlib")):
        try:
            mod = __import__(mod_name)
            print(f"    {OK} {pkg:<16} {getattr(mod, '__version__', '未知')}")
        except Exception as e:
            ok_deps = False
            print(f"    {BAD} {pkg:<16} 导入失败：{type(e).__name__}: {e}")
    if not ok_deps:
        print("\n    请先安装依赖（在 PyCharm 的 Terminal 里执行）：")
        print("    python -m pip install -r requirements.txt "
              "-i https://pypi.tuna.tsinghua.edu.cn/simple")
        return 1

    import numpy as np

    # 3) 目录结构
    print(f"\n[3] 项目目录：{ROOT}")
    for rel in ("code", "data", "data/oblique", "data/reference", "data/corners",
                "output", "output/images", "output/matrices",
                "output/preview", "output/metrics"):
        p = os.path.join(ROOT, *rel.split("/"))
        flag = OK if os.path.isdir(p) else "--"
        print(f"    {flag} {rel}")
    for rel in ("第一章加分作业.md", "执行计划.md", "报告.md", "requirements.txt"):
        p = os.path.join(ROOT, rel)
        flag = OK if os.path.isfile(p) else "--"
        print(f"    {flag} {rel}")

    # 4) 核心算法最小自检
    print(f"\n[4] 核心算法自检（不读磁盘，纯数值）")
    import homography as hm
    src = np.array([[0., 0.], [599., 0.], [599., 799.], [0., 799.]])
    # 人为构造一个已知单应：透视 + 缩放 + 平移
    H_true = np.array([[1.02, -0.13, 40.0],
                       [0.07, 1.05, 25.0],
                       [3.0e-4, -2.0e-4, 1.0]])
    dst = hm.apply_homography(H_true, src)

    H = hm.solve_homography_svd(src, dst)
    rmse, _ = hm.reprojection_error(H, src, dst)
    dist = hm.homography_distance(H, H_true)
    H8, info8 = hm.solve_homography_8dof(src, dst)
    print(f"    齐次 SVD 重投影 RMSE : {rmse:.3e} px")
    print(f"    与构造真值的距离     : {dist:.3e}")
    print(f"    8 元解重投影 RMSE    : {hm.reprojection_error(H8, src, dst)[0]:.3e} px"
          f"   cond(M) = {info8['cond_M']:.3e}")
    print(f"    退化检查             : {hm.check_point_configuration(src) or '无任意三点共线'}")
    if rmse < 1e-9 and dist < 1e-9:
        print(f"    {OK} 求解器正常")
    else:
        print(f"    {BAD} 求解结果异常，请检查 homography.py")
        return 1

    # 5) 合成数据
    print(f"\n[5] 合成数据")
    obl = os.path.join(ROOT, "data", "oblique")
    n_img = len([f for f in os.listdir(obl) if f.lower().endswith(".png")]) \
        if os.path.isdir(obl) else 0
    if n_img:
        print(f"    {OK} data/oblique 下有 {n_img} 张图，可以直接跑主流程")
    else:
        print(f"    -- data/oblique 为空")
    if "--synth" in argv:
        import synth as sy
        ds = sy.build_dataset(ROOT, force=True)
        print(f"    {OK} 已生成合成数据：{len(ds['items'])} 组斜视图 + 解析真值 H0")
        for it in ds["items"]:
            print(f"        - {os.path.basename(it['image'])}")

    print("\n" + "=" * 70)
    print("自检通过。下一步在 PyCharm 里运行：")
    print("    main.py --synth --experiments --batch --max-side 700")
    print("=" * 70)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
