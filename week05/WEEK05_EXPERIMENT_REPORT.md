# Week05 实验报告：Delta Displacement GASF DDPM

## 1. 总览

Week05 的工作目标是把车辆轨迹从不定长 `(x, y)` 时间序列转成可被 Week04 DDPM 框架训练的固定尺寸 `224 x 224 x 3` float array，并系统评估不同归一化强度和结构辅助 loss 对生成质量的影响。

核心结论：

1. 原始轨迹长度差异很大，不能直接按旧流程假设每条轨迹都有 224 个点；实际只有 40 / 12,061 条轨迹恰好为 224 点。
2. 主训练集采用 bounce / reflect padding 到 224 点，保留全部 12,061 条轨迹；滑窗方案只保留 3,877 个窗口，会丢掉 72.45% 的 source trajectories。
3. 最优模型是 `week05_sig15_mse_diag` 的 40,000 step checkpoint，FID = 44.5496。
4. Diagonal loss 明显改善 FID；symmetry loss 虽然在形式上合理，但在本轮实验中显著伤害生成质量。
5. 后验采样阶段的 GASF consistency guidance 对 FID 和 consistency loss 几乎没有实质影响，说明当前 guidance 目标或施加强度不足以改变样本分布。

## 2. 时间线

| 时间 UTC | 阶段 | 主要文件 / 输出 | 说明 |
|---|---|---|---|
| 2026-06-05 18:25 | 原始数据放入 week05 | `data/vehicle_positions_TS_New.csv` | 2,189,591 行，字段为 `time,id,x,y,type` |
| 2026-06-05 18:33-19:54 | 构建 variable-length delta dataset | `scripts/build_delta_displacement_paired_data.py`, `data/delta_displacement_paired/` | 按车辆 id 排序，保存 absolute、delta displacement、start point 和 train/val/test split |
| 2026-06-05 20:12-20:24 | 尝试 sliding window 固定长度方案 | `scripts/window_existing_delta_dataset.py`, `data/delta_displacement_paired_window224/` | stride=112，只保留长度 >=224 的轨迹窗口 |
| 2026-06-05 20:50-21:53 | 构建 bounce regularized 主数据集 | `scripts/regularize_existing_delta_dataset_to_224_by_bounce.py`, `data/delta_displacement_paired_regularized_bounce224/` | 短轨迹反射填充，长轨迹截断前 224 点 |
| 2026-06-05 22:17-22:57 | 准备并启动 6 组 DDPM 训练 | `scripts/train_week05_ddpm.py`, `slurm/week05_*.slurm` | 2 个 sigmoid k 值 x 3 种 loss |
| 2026-06-06 12:46-13:01 | 6 组训练完成 | `results/*/train_summary.json` | 每组训练到 50,000 step |
| 2026-06-07 12:27-16:30 | checkpoint 采样 | `scripts/sample_week05_checkpoints_to_npy.py`, `results_npy_samples/` | 每组取 40k,42k,44k,46k,48k,50k checkpoint，各生成 1,000 个样本 |
| 2026-06-07 16:53-17:10 | FID 评估 | `scripts/evaluate_week05_npy_fid.py`, `fid_results/week05_fid_summary.csv` | 使用 Inception-v3 feature mean/cov sqrtm，与真实 GASF float arrays 比较 |
| 2026-06-07 19:39-20:50 | 小尺度 consistency guidance | `guidance_consistency_results/` | guidance scale = 0, 0.005, 0.01, 0.02 |
| 2026-06-07 23:00-2026-06-08 00:11 | 大尺度 + grad clip guidance | `guidance_consistency_results_gradclip010/` | guidance scale = 0.05, 0.1, 0.2, 0.5，grad clip=0.1 |

## 3. 实验动机

Week04 的 DDPM 框架面向 `224 x 224 x 3` 图像式输入，而 Week05 希望生成的是车辆轨迹的运动信息。直接生成 absolute position 容易把全局位置分布、起点分布和局部运动模式混在一起；因此 Week05 改用 delta displacement：

```text
delta[0] = 0
delta[t] = absolute[t] - absolute[t-1]
```

这样模型主要学习局部运动增量。为了仍保留轨迹起点信息，第三个通道加入 start-position heatmap。

GASF 表示的动机是把一维时间序列转成二维结构图像，使其能复用现有 DDPM U-Net。`dx` 和 `dy` 分别经过 train split 统计量标准化，再用 sigmoid 映射到 `[0,1]`，最后编码成 GASF：

```text
R = GASF(sigmoid-normalized dx)
G = GASF(sigmoid-normalized dy)
B = start-position heatmap
```

本周主要验证三个问题：

1. 用什么方式把不定长轨迹整理为固定长度 224 最合理。
2. sigmoid 映射强度 `k=1.0` 或 `k=1.5` 哪个更适合。
3. 除普通噪声预测 MSE 外，是否需要用 GASF 的结构性质加入辅助 loss。

## 4. 数据构建

### 4.1 原始数据

原始 CSV：

```text
week05/data/vehicle_positions_TS_New.csv
```

统计：

| 项目 | 数值 |
|---|---:|
| raw rows | 2,189,591 |
| vehicle ids | 12,061 |
| trajectory length min | 13 |
| trajectory length max | 657 |
| trajectory length mean | 181.5431 |
| trajectory length median | 173 |
| exactly 224 points | 40 |
| not 224 points | 12,021 |

### 4.2 Variable-length delta dataset

输出目录：

```text
week05/data/delta_displacement_paired/
```

每个样本保存：

| 文件夹 | 内容 |
|---|---|
| `labels_absolute/*.npy` | 原始 absolute trajectory，shape 为 `(N, 2)` |
| `labels_delta_displacement/*.npy` | 重新计算的 delta displacement，shape 为 `(N, 2)` |
| `starts/*.npy` | 起点 `(x0, y0)` |
| `splits/*.csv` | train/val/test id |
| `normalization_diagnostics.json` | train split 的 delta 统计 |

结果：

| 项目 | 数值 |
|---|---:|
| num_output_samples | 12,061 |
| train / val / test | 9,648 / 1,206 / 1,207 |
| max reconstruction error | 3.18e-12 |
| dx mean / std | 0.0208 / 10.2713 |
| dy mean / std | -0.0289 / 8.5296 |

重建误差接近数值精度，说明 `start + cumulative delta` 能正确还原 absolute trajectory。

### 4.3 Sliding window 方案

输出目录：

```text
week05/data/delta_displacement_paired_window224/
```

设置：`target_length=224`，`stride=112`，`include_tail=false`。

结果：

| 项目 | 数值 |
|---|---:|
| num_source_samples | 12,061 |
| num_output_windows | 3,877 |
| num_skipped_short_samples | 8,738 |
| short_sample_ratio | 0.7245 |
| train / val / test | 3,101 / 387 / 389 |

分析：这个方案保留的都是真实连续 224 点窗口，数据最干净，但数据覆盖很差。由于 72.45% 的 source trajectories 小于 224 点，它更适合作为将来的对照组，不适合作为 Week05 主训练集。

### 4.4 Bounce regularization 主方案

输出目录：

```text
week05/data/delta_displacement_paired_regularized_bounce224/
```

规则：

| 原始长度 | 处理 |
|---|---|
| `N < 224` | bounce / reflect padding |
| `N = 224` | 保持不变 |
| `N > 224` | 截断前 224 点 |

短轨迹的 index pattern：

```text
[0, 1, 2, ..., N-1, N-2, ..., 1] repeated to 224
```

结果：

| 项目 | 数值 |
|---|---:|
| num_source_samples | 12,061 |
| num_output_samples | 12,061 |
| num_skipped_samples | 0 |
| bounce_reflect_padding | 8,738 |
| truncate_first_224 | 3,283 |
| unchanged | 40 |
| train / val / test | 9,648 / 1,206 / 1,207 |
| max reconstruction error | 3.41e-12 |
| dx mean / std | 0.1252 / 10.4037 |
| dy mean / std | -0.0899 / 8.9178 |

选择理由：该方案保留全部车辆轨迹，并避免 zero padding 造成大量静止片段，也避免 interpolation 改变速度模式。代价是短轨迹后半段包含人为反射运动，可能让模型学到折返结构。

## 5. DDPM 输入与训练设置

训练脚本：

```text
week05/scripts/train_week05_ddpm.py
```

数据格式：

```text
float_gasf_generated_on_the_fly_from_delta_npy
```

训练时不读取 PNG，而是在线从 `labels_delta_displacement/*.npy` 生成 `3 x 224 x 224` tensor。输入进入模型前从 `[0,1]` 映射到 `[-1,1]`。

共同训练设置：

| 参数 | 值 |
|---|---|
| data root | `delta_displacement_paired_regularized_bounce224` |
| train samples | 9,648 |
| image size | 224 |
| channels | 3 |
| batch size | 8 |
| gradient accumulation | 2 |
| learning rate | 2e-4 |
| max train steps | 50,000 |
| scheduler | `DDPMScheduler`, 1,000 train timesteps |
| inference checkpoints | 40k, 42k, 44k, 46k, 48k, 50k |
| sample count per checkpoint | 1,000 |
| sampling inference steps | 50 |
| seed | 42 |
| EMA | enabled |

实验矩阵：

| Experiment | Sigmoid k | Loss | Best FID step |
|---|---:|---|---:|
| `week05_sig10_mse` | 1.0 | MSE | 50,000 |
| `week05_sig10_mse_diag` | 1.0 | MSE + diagonal loss | 40,000 |
| `week05_sig10_mse_sym` | 1.0 | MSE + symmetry loss | 50,000 |
| `week05_sig15_mse` | 1.5 | MSE | 48,000 |
| `week05_sig15_mse_diag` | 1.5 | MSE + diagonal loss | 40,000 |
| `week05_sig15_mse_sym` | 1.5 | MSE + symmetry loss | 46,000 |

Loss 定义：

| Loss mode | 公式 / 含义 |
|---|---|
| `none` | 只使用 `MSE(noise_pred, noise)` |
| `diag` | 从预测 `x0` 取前两个 GASF 通道 diagonal，与 clean image diagonal 做 MSE，权重 0.05 |
| `symmetry` | 约束预测 GASF 通道接近转置对称，权重 0.05 |

Diagonal loss 的动机更直接：GASF diagonal 携带可反解回原始 normalized time-series 的关键信息。Symmetry loss 的动机是让生成矩阵满足 GASF 的对称结构，但它只约束形式，不直接约束可恢复的运动序列。

## 6. FID 评估设置

评估脚本：

```text
week05/scripts/evaluate_week05_npy_fid.py
```

设置：

| 项目 | 值 |
|---|---|
| real split | all |
| num real | 12,061 |
| num generated | 1,000 / checkpoint |
| feature extractor | Inception-v3 |
| image resize | 224 -> 299 |
| FID 方法 | empirical mean/cov + `scipy.linalg.sqrtm` |
| real feature cache | 按 `sigmoid_k` 分开缓存 |

注意：FID 是在 GASF float-array 图像空间上计算的，不是直接在反解后的轨迹物理空间上计算的。因此它适合比较本轮模型的图像分布匹配程度，但不能单独证明轨迹物理合理性。

## 7. 实验结果

### 7.1 每组最佳 FID

| Experiment | Best step | Best FID | Final 50k FID |
|---|---:|---:|---:|
| `week05_sig15_mse_diag` | 40,000 | 44.5496 | 52.4143 |
| `week05_sig10_mse_diag` | 40,000 | 52.8173 | 58.3155 |
| `week05_sig15_mse` | 48,000 | 67.9659 | 68.2091 |
| `week05_sig10_mse` | 50,000 | 69.6000 | 69.6000 |
| `week05_sig15_mse_sym` | 46,000 | 129.1660 | 132.7624 |
| `week05_sig10_mse_sym` | 50,000 | 137.4864 | 137.4864 |

排名前 10 的 checkpoint 全部来自 diagonal loss 或其附近结果，其中最好的 6 个都是 `sigmoid_k=1.5 + diag` 的 40k-50k checkpoint。

### 7.2 完整 FID 曲线摘要

| Config | 40k | 42k | 44k | 46k | 48k | 50k |
|---|---:|---:|---:|---:|---:|---:|
| `sig10_mse` | 78.7878 | 79.7034 | 76.4782 | 73.1179 | 71.2523 | 69.6000 |
| `sig10_mse_diag` | 52.8173 | 55.2542 | 55.8920 | 54.0297 | 53.7218 | 58.3155 |
| `sig10_mse_sym` | 143.7027 | 141.7514 | 141.4227 | 139.8515 | 139.3601 | 137.4864 |
| `sig15_mse` | 77.9133 | 77.1691 | 73.7081 | 71.3687 | 67.9659 | 68.2091 |
| `sig15_mse_diag` | 44.5496 | 44.8652 | 47.6087 | 48.0252 | 48.8892 | 52.4143 |
| `sig15_mse_sym` | 141.3163 | 138.4636 | 133.9564 | 129.1660 | 133.0479 | 132.7624 |

### 7.3 训练 loss 观察

所有训练都稳定完成 50,000 step，没有发现非有限 loss 或梯度中断。

50k 最后日志：

| Config | total loss | mse loss | aux loss |
|---|---:|---:|---:|
| `sig10_mse` | 0.000807 | 0.000807 | 0 |
| `sig10_mse_diag` | 0.004644 | 0.001147 | diag=0.069941 |
| `sig10_mse_sym` | 0.000937 | 0.000888 | symmetry=0.000986 |
| `sig15_mse` | 0.000912 | 0.000912 | 0 |
| `sig15_mse_diag` | 0.006084 | 0.001494 | diag=0.091788 |
| `sig15_mse_sym` | 0.001033 | 0.000993 | symmetry=0.000807 |

重要现象：训练 loss 低不等于 FID 好。Symmetry loss 的训练数值很小，但 FID 最差；diagonal loss 的 total loss 更高，但 FID 最好。这说明对 GASF 任务而言，辅助目标的语义比 loss 数值大小更关键。

## 8. Consistency Guidance 追加实验

追加实验使用最优主模型：

```text
week05/results/week05_sig15_mse_diag/checkpoint-40000
```

动机：训练后不改 checkpoint，在 reverse sampling 后半段对 GASF channels 施加可微 consistency guidance，希望生成图像更接近可由一维序列产生的合法 GASF。

设置：

| 项目 | 值 |
|---|---|
| checkpoint | `week05_sig15_mse_diag/checkpoint-40000` |
| sigmoid_k | 1.5 |
| guidance_start_fraction | 0.5 |
| samples | 1,000 |
| inference steps | 50 |
| real split | all |

结果：

| Guidance scale | Grad clip | FID | Mean consistency loss | Max consistency loss |
|---:|---:|---:|---:|---:|
| 0 | none | 44.5504 | 0.03634191 | 0.12904844 |
| 0.005 | none | 44.5488 | 0.03634191 | 0.12904838 |
| 0.01 | none | 44.5498 | 0.03634189 | 0.12904900 |
| 0.02 | none | 44.5501 | 0.03634186 | 0.12904865 |
| 0.05 | 0.1 | 44.5512 | 0.03634173 | 0.12904763 |
| 0.1 | 0.1 | 44.5484 | 0.03634143 | 0.12904617 |
| 0.2 | 0.1 | 44.5496 | 0.03634095 | 0.12904304 |
| 0.5 | 0.1 | 44.5489 | 0.03633951 | 0.12903447 |

结论：guidance 对 mean consistency loss 有极小下降，但幅度约为 `2.4e-6`，FID 基本保持在 44.548-44.551。它没有破坏样本，也没有带来可见的 FID 改善。当前形式更像无效或过弱的后处理约束。

## 9. 分析与结论

### 9.1 数据层面

Bounce regularization 是本周可行的主方案，因为它保留所有 12,061 条轨迹，并让 DDPM 输入满足固定 `224 x 224 x 3`。不过它把 8,738 条短轨迹转成了含反射段的 224 点序列，这会让训练分布包含人为折返。后续如果要生成物理轨迹，需要把 bounce 样本和真实长轨迹窗口分开评估。

Sliding window 数据虽然更真实，但样本数只有 3,877，且短轨迹被系统性排除，可能造成长度和行为模式偏差。它适合作为 “真实 224 点连续片段” 对照实验。

### 9.2 归一化层面

`sigmoid_k=1.5` 整体优于 `sigmoid_k=1.0`：

| Loss | k=1.0 best FID | k=1.5 best FID | 改善 |
|---|---:|---:|---:|
| MSE | 69.6000 | 67.9659 | 1.6341 |
| Diag | 52.8173 | 44.5496 | 8.2677 |
| Symmetry | 137.4864 | 129.1660 | 8.3204 |

`k=1.5` 让归一化后的运动值分布更展开，减少过多值挤在 0.5 附近的问题。对 diagonal loss 尤其有帮助，因为 diagonal loss 直接依赖 GASF diagonal 中可恢复的一维序列信息。

### 9.3 Loss 层面

Diagonal loss 是最有效的改动。相比同 k 的纯 MSE：

| Sigmoid k | MSE best FID | Diag best FID | 改善 |
|---:|---:|---:|---:|
| 1.0 | 69.6000 | 52.8173 | 16.7827 |
| 1.5 | 67.9659 | 44.5496 | 23.4163 |

这说明模型单纯学噪声预测时，可能能生成像图像的 GASF，但不一定保留 GASF diagonal 所承载的时间序列结构。Diagonal loss 把学习目标拉回到 “可解码的 dx/dy 运动序列”。

Symmetry loss 明显失败。可能原因：

1. 训练中的 clean GASF 本身已对称，模型通过 MSE 已经能接触到对称结构，额外 symmetry loss 信息增量有限。
2. Symmetry loss 只约束 `A = A.T`，但大量非 GASF 矩阵也可以对称；它没有约束 diagonal 或角度和结构。
3. 在扩散训练的 noisy / predicted x0 空间中强推对称，可能干扰噪声预测目标，导致 FID 变差。

### 9.4 checkpoint 选择

最优 diagonal 模型在 40k step 已达到最佳 FID，继续训练到 50k 反而变差：

```text
sig15_mse_diag: 44.55 -> 44.87 -> 47.61 -> 48.03 -> 48.89 -> 52.41
```

这说明 diagonal loss 方案可能较早达到最佳图像分布，然后出现过拟合或分布漂移。后续训练应增加中间 checkpoint 的评估密度，例如 30k-44k 每 1k 或 2k 评估。

## 10. 建议后续实验

1. 以 `week05_sig15_mse_diag/checkpoint-40000` 作为 Week05 当前最佳 baseline。
2. 增加 trajectory-space 指标：把 GASF diagonal 反解回 dx/dy，计算 delta 分布、速度分布、累计轨迹边界、起终点位移、异常跳变比例。
3. 单独训练 sliding-window 数据集，作为无 bounce artifact 的对照。
4. 对 diagonal loss 做权重扫描：`0.01, 0.025, 0.05, 0.1`，并重点评估 30k-45k。
5. 放弃或重新设计当前 symmetry loss；如果继续尝试，应改为更强的 GASF legality / cycle consistency 目标，而不是单纯对称性。
6. Guidance 若继续做，应先验证梯度大小和单步样本变化，当前结果显示它几乎不改变采样轨迹。

## 11. 关键文件索引

| 类型 | 路径 |
|---|---|
| 原始数据 | `week05/data/vehicle_positions_TS_New.csv` |
| 主数据集 | `week05/data/delta_displacement_paired_regularized_bounce224/` |
| 数据构建脚本 | `week05/scripts/build_delta_displacement_paired_data.py` |
| 固定长度主方案脚本 | `week05/scripts/regularize_existing_delta_dataset_to_224_by_bounce.py` |
| 训练脚本 | `week05/scripts/train_week05_ddpm.py` |
| 训练结果 | `week05/results/` |
| checkpoint 采样脚本 | `week05/scripts/sample_week05_checkpoints_to_npy.py` |
| 采样结果 | `week05/results_npy_samples/` |
| FID 脚本 | `week05/scripts/evaluate_week05_npy_fid.py` |
| FID 汇总 | `week05/fid_results/week05_fid_summary.csv` |
| Guidance 脚本 | `week05/scripts/sample_week05_gasf_consistency_guidance.py` |
| Guidance 汇总 | `week05/guidance_consistency_results*/guidance_consistency_summary.csv` |

