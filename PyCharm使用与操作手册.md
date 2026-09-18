# PyCharm 使用与操作手册

> 配套项目：第一章加分作业 —— 射影矩阵自编程实现
> 目标：让你在 PyCharm 里**跑通、看懂、讲清**这套代码
> 环境实测：Windows 11 + Python 3.13 + numpy 2.5.3 + opencv-python 5.0.0 + matplotlib 3.11.2

> 迁移说明：当前 PC 使用本机 Python 3.13 创建项目内 `.venv`；PyCharm 2025.2.2 已安装。不要引用旧 PC 的外部解释器路径。

---

## 0. 先看这张图：你在这里做什么

```
斜向照片  ──预处理──▶  灰度矩阵 + 4 个角点坐标  ──求解 H──▶  射影矩阵  ──逆映射重采样──▶  正视图矩阵
   │                        │                                    │                          │
 data/oblique/          output/preview/                     output/metrics/            output/matrices/
                        output/matrices/                     *_run.json                 *_rectified_gray.npy
```

代码分三层：**算法层**（`homography.py` / `warp.py`，全手写）、**数据层**（`synth.py` / `preprocess.py`）、**验证层**（`experiments.py`）。讲解作业时，重点永远是**算法层**。

---

## 1. 五分钟跑通

### 步骤 1：用 PyCharm 打开项目

`File → Open` → 选择目录：

```
<项目根目录>\ComputerVision - 副本
```

打开后左侧 Project 面板应该能看到 `code/`、`data/`、`output/`、`报告.md` 等。

> 建议在 `Settings → Editor → File Encodings` 里把 Global / Project / Default 三项都设为 **UTF-8**，本项目所有源码和文档都是 UTF-8 编码。

### 步骤 2：配置 Python 解释器

`File → Settings → Project: ComputerVision - 副本 → Python Interpreter → Add Interpreter → Add Local Interpreter → Existing`

选择已有解释器（**推荐，零下载**，因为依赖已经装好了）：

```
<项目根目录>\ComputerVision - 副本\.venv\Scripts\python.exe
```

配置成功后，Interpreter 下方的包列表里应能看到 `numpy`、`opencv-python`、`matplotlib` 三个包。

<details>
<summary>方案 B：想自己建一个干净的虚拟环境（点开）</summary>

1. `Add Interpreter → Add Local Interpreter → Virtualenv Environment → New`
2. Location 填 `<项目根目录>\ComputerVision - 副本\.venv`，Base interpreter 选 Python 3.13
3. 打开 PyCharm 底部 `Terminal`，执行（国内镜像，几十秒）：

```
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
```

</details>

### 插件与终端说明

项目不依赖 Workbuddy 或其他外部 PyCharm 插件。只需确保 PyCharm 自带的 Python 支持已启用；Markdown/图片预览属于可选功能，不影响代码运行。

新 PC 上不要复用旧电脑的解释器路径，统一在项目根目录执行：

```powershell
py -3.13 -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
.\.venv\Scripts\python.exe code\check_env.py
.\.venv\Scripts\python.exe -m unittest discover -s tests -v
```

### 步骤 3：跑环境自检（第一个该跑的东西）

右键 `code/check_env.py` → `Run 'check_env'`，或按 `Ctrl+Shift+F10`。

期望输出：

```
[1] Python        版本 : 3.13.x
[2] 依赖库        [OK] numpy 2.5.3 / opencv-python 5.0.0 / matplotlib 3.11.2
[3] 项目目录      全部 [OK]
[4] 核心算法自检  齐次 SVD 重投影 RMSE : 2.05e-13 px   与构造真值的距离 : 3.39e-15
[5] 合成数据      [OK] data/oblique 下有 3 张图
自检通过。
```

**这一步很关键**：它不读磁盘、纯数值地验证了求解器是否正确。如果第 4 步的 RMSE 不是 `1e-13` 量级，说明解释器或代码被动过，先解决它再往下走。

### 步骤 4：建运行配置，跑完整流程

`Run → Edit Configurations → + → Python`，按下表填：

| 字段 | 值 |
|---|---|
| Name | `all`（随便起） |
| Script path | `$PROJECT_ROOT\code\main.py` |
| Parameters | `--synth --experiments --batch --max-side 700` |
| Working directory | `$PROJECT_ROOT` |
| Python interpreter | 选步骤 2 配好的那个 |

点 Run（`Shift+F10`），约 10~20 秒跑完。

> 关于工作目录：本项目所有路径都是从 `__file__` 推导的绝对路径，**工作目录填哪里都能跑**。填 `code` 只是为了看输出整齐。
>
> 另外，Parameters 里的**相对路径会自动按项目根目录解析**（`main.py` 里的 `_resolve()`），所以可以直接写 `--image data/oblique/xxx.png`，不必从 `code/` 往上退一级。

### 步骤 5：看结果

跑完后在 Project 面板里：

- `output/images/fig_synth_oblique_overview.png` ← **双击，PyCharm 直接看图**（左：斜图+角点；中：本实现正视图；右：参考图）
- `output/images/fig_synth_oblique_interp.png` ← 最近邻 / 双线性 / OpenCV 三种重采样对比
- `output/metrics/experiments.json` ← 所有实验指标
- `output/metrics/synth_oblique_run.json` ← 这张图的完整数据（分辨率、角点、H、误差、矩阵元信息）

在 PyCharm 里查看 PNG：单击文件，如果没显示图像，右键 → `Open In → Explorer`，或用 `Ctrl+Shift+F4` 关闭编辑器后重新双击。

---

## 2. 运行配置清单（一次配好，以后一键跑）

`Run → Edit Configurations` 里建 6 个配置，覆盖全部使用场景：

| 配置名 | Script | Parameters | 用途 |
|---|---|---|---|
| `1_env` | `code/check_env.py` | 留空 | 环境自检 |
| `2_synth` | `code/main.py` | `--synth` | 生成合成数据（3 组斜图 + 真值 H0 + 角点） |
| `3_experiments` | `code/main.py` | `--experiments` | 跑 4 组对照实验 + 出图 |
| `4_batch` | `code/main.py` | `--batch --max-side 700` | 批量处理 `data/oblique/` 下所有图 |
| `5_single` | `code/main.py` | `--image data/oblique/synth_oblique.png --corners data/corners/synth_oblique.json --max-side 700` | 调试单张图 |
| `6_pick` | `code/pick_corners.py` | `--image data/oblique/你的照片.jpg --max-side 700` | 交互标定实拍图角点 |

**一次跑完全部**：Parameters 填 `--synth --experiments --batch --max-side 700`。

### 可选：把 `code` 标记为源码根

右键 `code` 目录 → `Mark Directory as → Sources Root`（变蓝）。这样在 PyCharm 里 `Ctrl+点击` 函数名能跨模块跳转，`homography.solve_homography_svd` 这类引用会正常解析。

> 因为代码里有 `sys.path.insert(0, os.path.dirname(__file__))`，标记与否都能运行，只是为了**阅读体验**。

---

## 3. PyCharm 调试技巧（看代码时最有用）

### 3.1 断点看中间量

在 `code/homography.py:144`（`solve_homography_svd` 里的 `_, _, Vt = np.linalg.svd(A)`）打个断点，用 `5_single` 配置 Debug（`Shift+F9`）。命中后可以检查：

| 变量 | 应该看到什么 |
|---|---|
| `A` | shape `(8, 9)` —— 这就是作业里的 $\mathbf{A}$ |
| `A[0]` | `[x, y, 1, 0, 0, 0, -x*x', -y*x', -x']`，第一行的前三个元素 |
| `A[1]` | `[0, 0, 0, x, y, 1, -x*y', -y*y', -y']`，**第 3 个元素必须是 0** |
| `Vt` | shape `(9, 9)`，`Vt[-1]` 就是解向量 $\vec h$ |
| `H` | 3×3 矩阵 |

在 PyCharm 的 Variables 面板里，数组可以右键 → `View as Array` 用表格查看，比看 `repr` 直观得多。

### 3.2 多行表达式窗口（Evaluate Expression）

调试时按 `Alt+F8`，可以现场敲：

```python
np.linalg.svd(A, compute_uv=False)          # 看奇异值谱
np.linalg.cond(A)                           # 条件数
homography.reprojection_error(H, corners, tc)  # 重投影误差
```

### 3.3 在 Python Console 里交互式探索

`Tools → Python Console`（或右下角 `Python Console`），先切到项目根目录，然后：

```python
import sys; sys.path.insert(0, r'<项目根目录>\ComputerVision - 副本\code')
import numpy as np, homography as hm

# 用 4 组点解一个 H，并检查
src = np.array([[0,0],[599,0],[599,799],[0,799]], float)
H_true = np.array([[1.02,-0.13,40.],[0.07,1.05,25.],[3e-4,-2e-4,1.]])
dst = hm.apply_homography(H_true, src)
H = hm.solve_homography_svd(src, dst)
print(hm.reprojection_error(H, src, dst)[0])   # 应约 1e-13
print(hm.homography_distance(H, H_true))       # 应约 1e-15
print(hm.normalize_h(H, 'h33'))                # 便于阅读的形式
```

### 3.4 查看 `.npy` 像素矩阵

`.npy` 是二进制，PyCharm 不能直接预览。最方便的是建一个运行配置指向 `code/check_env.py` 改的临时脚本，或者直接在 Python Console 里：

```python
import numpy as np
a = np.load(r'output/matrices/synth_oblique_rectified_gray.npy')
print(a.shape, a.dtype, a.min(), a.max(), round(a.mean(),2))
print(a[:5, :8])          # 左上角 5x8 的像素值
```

> 想看得更清楚，用 `output/preview/*_preview.csv` —— 那是降采样后的 16×16 数值表，PyCharm 里双击就是表格视图（CSV 编辑器）。

### 3.5 matplotlib 不出图？

`experiments.py` 开头有 `matplotlib.use("Agg")`，这是**故意**的：让脚本在无图形界面的环境下也能跑，图直接存成 PNG 文件。

如果你想让它弹窗显示，把那一行改成：

```python
# matplotlib.use("Agg")
import matplotlib.pyplot as plt
plt.show()   # 需要自己加在 _fig_* 函数末尾
```

---

## 4. 项目结构导览

```
ComputerVision - 副本/
├─ 第一章加分作业.md     作业原文（已校对修正，作为需求依据）
├─ 执行计划.md           技术决策、验收指标、执行步骤（讲作业时可作"方法论"部分）
├─ 报告.md               ★ 交付主体：原理推导 + 实现 + 4 组实验 + 误差分析
├─ requirements.txt      依赖清单
├─ PyCharm使用与操作手册.md  ← 本文档
├─ code/
│  ├─ check_env.py       环境自检（先跑这个）
│  ├─ homography.py      ★ 核心：A 构造 / SVD 求解 / 8 元对照 / 求逆 / 误差
│  ├─ warp.py            ★ 核心：手写逆映射 + 双线性插值
│  ├─ preprocess.py      读写（兼容中文路径）、降采样、矩阵导出
│  ├─ synth.py           合成数据 + 解析真值 H0（实验的"标准答案"）
│  ├─ experiments.py     4 组对照实验 + 管线验证 + 出图
│  ├─ main.py            端到端主流程（CLI）
│  └─ pick_corners.py    鼠标交互标定角点
├─ data/
│  ├─ reference/         正向文档图 doc_gray.png + <名称>_H0.json（解析真值）
│  ├─ oblique/           斜视图（合成 3 张；实拍照片放这里）
│  └─ corners/           <名称>.json 四角点坐标
└─ output/
   ├─ images/            斜图（含角点标记）、正视图、最近邻、512×512、实验图
   ├─ matrices/          <名称>_oblique_gray.npy / <名称>_rectified_gray.npy
   ├─ preview/           <名称>_*_preview.csv 降采样数值表
   └─ metrics/           experiments.json + <名称>_run.json
```

---

## 5. 代码阅读路线（按调用链读，别按文件顺序读）

### 主线：一张图是怎么被矫正的

```
main.py:47  run_one()                         ← 从这里开始，一个函数读完主流程
  ├─ preprocess.py:50  load_image()           读图 + 灰度 + 降采样 → 得到像素矩阵与 scale
  ├─ preprocess.py:105 load_corners()         读四角点（顺序：左上→右上→右下→左下）
  ├─ preprocess.py:88  scale_corners()        角点坐标换算到降采样坐标系
  ├─ homography.py:244 check_point_configuration()  退化检查（任意 3 点共线则无解）
  ├─ warp.py:24        estimate_target_size() 由对边平均长度定正视图尺寸
  ├─ warp.py:18        target_corners()       固定的目标角点 (0,0)(W-1,0)(W-1,H-1)(0,H-1)
  ├─ homography.py:119 solve_homography_svd() ★求解 H
  ├─ homography.py:208 reprojection_error()   验收指标
  ├─ homography.py:191 invert_homography()    手写伴随矩阵求逆 → H⁻¹
  ├─ warp.py:55        warp_bilinear()        ★逆映射 + 双线性插值 → 正图象素矩阵
  └─ 落盘：images / matrices / preview / metrics
```

### 算法线：`homography.py` 的阅读顺序

| 顺序 | 行号 | 函数 | 一句话理解 |
|---|---|---|---|
| 1 | 63 | `build_A` | 把 4 组点变成 $8\times9$ 矩阵 $\mathbf{A}$，使 $\mathbf{A}\vec h=0$ |
| 2 | 43 | `normalize_points` | Hartley 归一化：质心→原点，平均距离→$\sqrt2$ |
| 3 | 119 | `solve_homography_svd` | **主求解器**：SVD 取 $V$ 最后一列 → 重塑 3×3 → 反归一化 |
| 4 | 152 | `solve_homography_8dof` | 对照解：固定 $h_{33}=1$，解 $8\times8$ 非齐次方程 |
| 5 | 191 | `invert_homography` | 手写伴随矩阵求逆（不调 `np.linalg.inv`） |
| 6 | 208 | `reprojection_error` | 把点映射过去和真值比，得到误差 |
| 7 | 220 | `homography_distance` | 齐次尺度 + 符号无关的 $H$ 距离（比 $H$ 的正确姿势） |
| 8 | 229 | `diagnose` | $\kappa(A)$、奇异值谱、$\det H$ —— 讲"数值稳定性"时用它 |

### 重采样线：`warp.py`

| 行号 | 函数 | 要点 |
|---|---|---|
| 41 | `_backward_map` | 对每个**目标**像素反算源坐标；$|s_w|<\varepsilon$ 判为无穷远点 |
| 55 | `warp_bilinear` | 双线性：$(1-d_x)(1-d_y)I_{00}+d_x(1-d_y)I_{01}+(1-d_x)d_yI_{10}+d_xd_yI_{11}$ |
| 90 | `warp_nearest` | 最近邻，只作质量对照 |
| 110 | `measure_collinearity` | 验证"直线仍为直线" |

### 实验线：`experiments.py`

| 行号 | 函数 | 结论（报告里的第 4 节） |
|---|---|---|
| 67 | `exp1_equivalence` | 4 点下两种解法给出同一个 $H$（距离 $3\times10^{-11}$） |
| 96 | `exp2_normalization` | 归一化把 $\kappa(A)$ 从 $1.09\times10^6$ 降到 3.59 |
| 124 | `exp3_h33_zero` | 真实 $h_{33}=0$ 时 8 元解失效（1.104 px），齐次解正确 |
| 157 | `exp4_noisy_least_squares` | $n>4$ 抗噪：$H$ 距离随点数单调下降 |
| 189 | `exp5_pipeline_verification` | 端到端：与 cv2 一致性 59~66 dB，与真值重采样约 270 dB |

---

## 6. 关键概念速查（讲解时的高频问题）

| 概念 | 一句话 | 代码位置 |
|---|---|---|
| 射影矩阵 $H$ | 3×3 齐次矩阵，$x'=Hx$，把平面到平面的一般射影变换统一表示 | `homography.py:119` |
| 自由度：8 还是 9 | $H$ 是**齐次等价类**（$H$ 与 $\lambda H$ 同一个变换）→ **8 个自由度**；待解向量 $\vec h$ 有 **9 个分量**。$h_{33}=1$ 就是把多出的 1 个尺度自由度固定掉的**规范化约定（gauge fixing）** | `报告.md` 2.1 |
| $\mathbf{A}$ 矩阵 | 每对点贡献 2 行，$n=4$ 时是 $8\times9$。**第 2 行第 3 个元素必须为 0** | `homography.py:63` |
| 为什么取 $V$ 最后一列 | 约束 $\|\vec h\|=1$ 下最小化 $\|A\vec h\|$，解是 $A^TA$ 最小特征值对应的特征向量；而 $A^TA=V\Sigma^2V^T$，故即 $V$ 最后一列 | `homography.py:144` |
| 归一化 DLT | 像素坐标下 $A$ 的元素跨 $1\sim10^6$ 量级，$\kappa(A)\approx10^6$；先归一化再解、最后 $H=T'^{-1}\tilde HT$ 反归一化 | `homography.py:43`（归一化）、`:138`（调用）、`:148`（反归一化） |
| 重投影误差 | 解出的 $H$ 把源点映射过去，与目标点的距离。**注意**：4 点时它恒为 0，不能当唯一验收指标 | `homography.py:208` |
| 逆映射 | 对每个目标像素反算源坐标，避免前向映射留下空洞 | `warp.py:41` |
| 双线性插值 | 用 4 个邻近整数像素按距离加权 | `warp.py:55` |
| 有效掩膜 | 齐次分量 $|s_w|$ 过小（映到无穷远）或源坐标越界的像素要填充背景色 | `warp.py:49` |
| 解析真值 $H_0$ | 用 $A_{out}\cdot P\cdot R\cdot A_{in}$ 乘法构造，**不含任何求解过程**，所以是独立的"标准答案" | `synth.py:85` |

---

## 7. 结果文件说明

| 文件 | 内容 | 怎么用 |
|---|---|---|
| `output/images/<名称>_oblique.png` | 斜图 + 红色四边形 + 黄色角点标记 | 讲解第一步"标定四角点" |
| `output/images/<名称>_rectified.png` | **本实现**输出的正视图 | 核心结果 |
| `output/images/<名称>_rectified_nearest.png` | 最近邻版本 | 与双线性对比，说明插值影响 |
| `output/images/<名称>_rectified_512.png` | 固定 512×512 的目标尺寸版本 | 说明"目标尺寸可以自由指定" |
| `output/images/fig_<名称>_overview.png` | 三联图：斜图 / 正视图 / 参考图 | **讲解时最好用的一张** |
| `output/images/fig_<名称>_grid.png` | 规则网格映射到斜视图 | 直观说明"直线仍是直线" |
| `output/images/fig_<名称>_interp.png` | 最近邻 / 双线性 / OpenCV 对比（含局部放大） | 讲重采样质量 |
| `output/images/fig_exp2_normalization.png` | $\kappa(A)$ 与误差的柱状图 | 讲归一化的必要性 |
| `output/images/fig_exp4_noise.png` | 误差随点数/噪声的曲线 | 讲最小二乘抗噪 |
| `output/matrices/<名称>_*_gray.npy` | 完整像素矩阵 | `<名称>_rectified_gray.npy` 就是作业要的"正图片像素矩阵" |
| `output/preview/<名称>_*_preview.csv` | 16×16 降采样数值表 | 想直接**看到数字**时用这个 |
| `output/metrics/<名称>_run.json` | 单张图全套数据：分辨率、角点、$H$、诊断、误差、矩阵元信息 | 报告里每个数字的出处 |
| `output/metrics/experiments.json` | 4 组实验 + 管线验证的全部指标 | 答辩/提问时的数据来源 |

---

## 8. 常见问题排查

| 现象 | 原因 | 解决 |
|---|---|---|
| `ModuleNotFoundError: No module named 'numpy'` | 解释器选错（选成了系统 Python 3.8） | 按本文档 1.2 重新配置解释器；跑 `check_env.py` 确认 |
| `ModuleNotFoundError: No module named 'homography'` | 直接双击运行了某些文件而没经过 `main.py` | 所有脚本都在开头做了 `sys.path.insert(0, dirname(__file__))`，若仍报错，检查是否复制代码时漏了开头两行 |
| 图片读写报 `None` / 抛 `FileNotFoundError` | 用了中文路径 + `cv2.imread` | 项目已用 `imread_unicode` 绕过，**自己写新代码时不要直接用 `cv2.imread`** |
| 角点报"越界" | 角点是在原图坐标下手填的，但流程开了降采样 | 要么把坐标乘 `scale`，要么用 `main.py --max-side 0` 关闭降采样 |
| 正视图是黑/白一片 | 角点顺序不是 左上→右上→右下→左下 | 顺序错了 $H$ 会算出很怪的结果（不会报错），重新按顺序标定 |
| `cv2.imshow` 报错 / 卡住 | 环境无图形界面 | `pick_corners.py` 只在有桌面环境的本机用；CI/远程请直接手写角点 json |
| matplotlib 弹不出窗口 | 故意设了 `Agg` 后端 | 见本文档 3.5 |
| 输出里出现 `dirname: command not found` | 本机 Git Bash 的 PATH 不完整（无害） | 忽略；这是 PyCharm 之外那个 shell 的环境问题，不影响 Python |
| 结果和上次不一样 | 合成数据用固定随机种子，但如果你手改了 `synth.py` 的 `seed` 或参数 | 恢复 `seed=20260917` 即可复现 |
| `图片不存在：data/oblique/xxx.png` | 相对路径按项目根目录解析（`main.py` 的 `_resolve()`）；只有项目根下确实没有该文件时才会走到这一步 | 确认文件名拼写，或改用绝对路径 |

---

## 9. 讲解作业的建议提纲（约 10 分钟）

**核心策略**：先给结论性数字（让人知道做对了），再讲原理，最后用一个反例证明你理解得比别人深。

| 时间 | 内容 | 展示什么 | 一句话讲法 |
|---|---|---|---|
| 0:00 | 任务与思路 | `报告.md` 第 0 节 | "输入一张斜拍的照片，输出正的像素矩阵；求解 $H$ 必须自编程，所以我把求解器和重采样都手写了一遍。" |
| 0:40 | 预处理 | `fig_<名称>_overview.png` 左图 | "读图→灰度→降采样到 700，得到像素矩阵；四角点按左上→右上→右下→左下标定，目标角点固定为矩形四角。" |
| 1:40 | 原理推导 | 白板 / `报告.md` 2.1–2.2 | 从 $x'=\frac{h_{11}x+h_{12}y+h_{13}}{h_{31}x+h_{32}y+h_{33}}$ 交叉相乘，整理成 $\mathbf{A}\vec h=0$ |
| 3:00 | **识破一个坑** | `homography.py:63`、`报告.md` 2.2 | "作业原文 $A_i$ 第 2 行第 3 个元素写的是 1，应该是 0 —— 因为 $y'$ 那条方程里根本没有 $h_{13}$。这个错很危险：$\mathbf{A}$ 恒为 8×9，SVD **总能**返回一个解，不报错，只能靠重投影误差发现。" |
| 4:00 | 求解 | `homography.py:119` + `fig_exp2_normalization.png` | "SVD 取 $V$ 最后一列；但像素坐标下 $\kappa(A)$ 有 $10^6$，所以必须做 Hartley 归一化，$\kappa$ 降到 3.6，误差从 $10^{-9}$ 降到 $10^{-13}$。" |
| 5:30 | 重采样 | `fig_<名称>_interp.png` | "前向映射会留空洞，所以用逆映射；插值用双线性，和最近邻比 PSNR 差 24~27 dB，边缘平滑度一眼可见。" |
| 6:30 | **结果** | `fig_<名称>_overview.png` 中+右 | "角点重投影误差 $1.7\times10^{-13}$ px；解出的 $H$ 与解析真值距离 $5.6\times10^{-16}$；和 OpenCV 的独立实现一致性 59 dB，和'用真值 $H$ 重采样'的结果一致性 270 dB —— 数值上已经不可区分。" |
| 8:00 | **进阶：反例** | `报告.md` 4.3 | "很多人默认 $h_{33}=1$。我构造了一个 $h_{33}=0$ 的合法射影变换，此时 8 元解直接失效（误差 1.1 px，$\mathrm{cond}(M)=2\times10^{17}$），齐次 SVD 不受影响。所以这不是形式上的'更一般'。" |
| 9:00 | 抗噪与结论 | `fig_exp4_noise.png` | "点越多越抗噪；但注意 4 点时重投影误差恒为 0（过拟合），这时必须看 $H$ 与真值的距离。" |
| 9:40 | 局限 | `报告.md` 第 7 节 | "4 点对单点误差敏感，没做 RANSAC，也没建模径向畸变 —— 实拍图建议标 6~8 个点。" |

### 老师可能追问的问题（提前准备）

| 问题 | 回答要点 |
|---|---|
| 你用了 `cv2.getPerspectiveTransform` 吗？ | 没有。`grep -rn "warpPerspective\|getPerspectiveTransform\|findHomography" code/` 可以看到：`warpPerspective` 只出现在 `synth.py`（造测试数据）和 `experiments.py`（**作为独立实现交叉验证**），交付算法路径 `warp.py`/`homography.py` 完全没用。 |
| 为什么才 4 个点误差就这么小？ | 4 点是精确可解的情形，没有残差可分配，误差只由浮点运算决定（$\sim10^{-13}$）。**这恰恰说明重投影误差不能作为唯一指标** —— 有噪声时它依然是 0，但 $H$ 已经错了 0.13。 |
| 你怎么知道解出来的 $H$ 是对的？ | 三重独立验证：① 与解析构造的真值 $H_0$ 比（$5.6\times10^{-16}$）② 与 OpenCV 独立实现比输出图（59~66 dB）③ 定性检查"直线仍是直线"（偏离 $\sim10^{-13}$ px）。 |
| 归一化为什么是缩放到 $\sqrt2$？ | Hartley 的标准取值：归一化后坐标的均方根恰好为 1，此时 $A$ 各列量级均衡、条件数最小。也可以用别的值，但 $\sqrt2$ 是最优的。 |
| $A$ 为什么是 8×9 而不是 8×8？ | 因为没固定 $h_{33}$。齐次方程的零空间是 1 维的，需要 SVD 求最小奇异值对应的右奇异向量；固定 $h_{33}=1$ 才会变成 8×8 非齐次方程。 |

---

## 10. 复现清单（Checklist）

按顺序打勾，任何一步失败就停下排查：

- [ ] PyCharm 打开项目根目录 `ComputerVision - 副本`
- [ ] 解释器指向项目内 `.venv\Scripts\python.exe`（或使用等价的 Python 3.13 虚拟环境）
- [ ] File Encodings 设为 UTF-8
- [ ] 跑 `code/check_env.py` → 第 4 步 RMSE 为 $10^{-13}$ 量级
- [ ] 建 6 个运行配置（见第 2 节）
- [ ] 跑 `main.py --synth --experiments --batch --max-side 700`
- [ ] 检查 `output/images/` 下 12 个 PNG 是否生成
- [ ] 检查 `output/metrics/experiments.json` 是否有 `exp1`~`exp5`
- [ ] 打开 `fig_synth_oblique_overview.png` 确认正视图是正的
- [ ] 打开 `output/preview/*_rectified_preview.csv` 确认能看到数值
- [ ] （可选）拍一张斜角文档照放 `data/oblique/`，用 `pick_corners.py` 标定后跑单张流程

---

## 11. 想改动/扩展时的入口

| 想做的事 | 改哪里 |
|---|---|
| 换更强的透视角度、加噪声 | `synth.py:180` `build_dataset` 里的 `configs` 列表 |
| 改降采样尺寸 | 运行参数 `--max-side`（0 = 不降采样） |
| 强制目标尺寸（比如 800×1000） | 运行参数加 `--fixed-size 800x1000` |
| 不用对边平均长度估计目标尺寸 | `warp.py:24` `estimate_target_size` |
| 加三次卷积插值 | 在 `warp.py` 里仿照 `warp_bilinear`（`:55`）加一个 `warp_bicubic` |
| 加 RANSAC 鲁棒估计 | 新写一个模块，对 `solve_homography_svd` 做随机采样 + 内点统计 |
| 改验收指标 | `main.py:47` `run_one` 里 `metrics['accuracy']` 部分；`experiments.py:46` `_score` |
| 改报告的实验表格 | 指标都在 `output/metrics/experiments.json`，改 `报告.md` 时从中取数 |
