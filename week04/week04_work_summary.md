# Week04 工作总结：Constrained GASF 表示与 DDPM 消融实验

## 1. 本周核心目标

Week04 的重点从“学习式 decoder 还原轨迹”转向“受约束的 GASF 表示 + 解析式还原轨迹”。核心假设是：如果 DDPM 生成的图像前两个通道保持 GASF 结构，那么可以从 GASF 对角线解析恢复 `dx, dy`，再结合起点热力图积分回轨迹。

新的三通道表示为：

| 通道 | 内容 | 作用 |
| --- | --- | --- |
| R | `GASF(dx)` | 编码 x 方向相邻位移 |
| G | `GASF(dy)` | 编码 y 方向相邻位移 |
| B | initial-point heatmap | 保存绝对起点位置 |

本周主要完成了数据集构建、解析可逆性验证、`[0,1]` 与 `[-1,1]` 两种归一化比较，以及 DDPM 的 MSE / symmetry / GASF structure 消融实验。

## 2. 数据集与表示验证

### 2.1 `[0,1]` constrained GASF 数据集

输出目录：`week04/data_constrained_gasf_dxdy_start`

| 项目 | 数值 |
| --- | ---: |
| 输入轨迹数 | 3159 |
| 成功处理 | 3159 |
| 跳过样本 | 0 |
| 图像大小 | `224 x 224 x 3` |
| heatmap sigma | 3.0 |
| 归一化来源 | train split |
| 表示 | `GASF_dx_GASF_dy_start_heatmap` |

解析一致性检查结果：

| 检查项 | 结果 |
| --- | ---: |
| delta 一致性样本数 | 3159 / 3159 |
| `mean_mse` | 0 |
| `global_max_error` | 0 |

重要结论：`labels_delta_displacement` 与 `absolute_xy[t] - absolute_xy[t-1]` 完全一致，数据标签没有时间对齐问题。

### 2.2 Delta 对齐模式验证

为了把报告中 “Mode A ≈ Mode C，Mode B failed badly” 的结论变成可追溯记录，补充了独立脚本：

```text
week04/scripts/check_alignment_modes.py
```

输出文件：

```text
week04/data_constrained_gasf_dxdy_start/sanity_checks/alignment_modes_metrics.csv
week04/data_constrained_gasf_dxdy_start/sanity_checks/alignment_modes_summary.json
```

比较的三种积分方式：

| 模式 | 定义 |
| --- | --- |
| Mode A | `xy[0]=start; xy[t]=xy[t-1]+delta[t]` |
| Mode B | `xy[0]=start; xy[t]=xy[t-1]+delta[t-1]` |
| Mode C | `xy=start+cumsum(delta)` |

全量 3159 个样本结果：

| 模式 | mean ADE | median ADE | max ADE | mean FDE | 结论 |
| --- | ---: | ---: | ---: | ---: | --- |
| Mode A | `3.37e-17` | 0 | `1.06e-13` | `3.60e-17` | 与真实轨迹一致 |
| Mode B | 9.1718 | 9.2450 | 17.1669 | 9.4136 | 明显 off-by-one |
| Mode C | `5.16e-14` | `1.35e-14` | `9.16e-13` | `1.86e-13` | 与 Mode A 等价到数值误差 |

额外检查：

| 检查项 | 结果 |
| --- | ---: |
| `delta[0]` 最大绝对 x | 0 |
| `delta[0]` 最大绝对 y | 0 |
| `|delta[0]| > 1e-12` 样本数 | 0 |
| Mode A 与 Mode C global max abs diff | `2.44e-12` |

结论：当前数据的 delta convention 是 `delta[t] = xy[t] - xy[t-1]`，且 `delta[0]=0`。因此 Mode A 和 Mode C 都正确；Mode B 使用 `delta[t-1]` 会造成一帧错位，平均 ADE 约 9.17。

### 2.3 PNG 量化误差发现

一开始直接从 PNG 图像抽取 GASF 对角线再解析，轨迹重建 ADE 会达到 `0.5 ~ 10+`，看起来像是表示不可逆。后续把编码前的 float32 数组保存到 `arrays/` 后发现：

| 输入来源 | 重建误差 |
| --- | --- |
| float32 array | 约 `1e-5 ~ 1e-4` |
| uint8 PNG | ADE 可达 `0.5 ~ 10+` |

结论：GASF 对角线解析本身是有效的，主要误差来自 PNG 的 8-bit 量化。后续如果要用解析轨迹指标评估生成结果，应优先使用 float array 训练或保存生成样本的 float 输出，而不是只依赖 PNG。

### 2.4 `[0,1]` 归一化分布问题

`[0,1]` 的优点是对角线可直接无符号歧义地恢复原值：

```text
diag = 2x^2 - 1
x in [0,1] => x = sqrt((diag + 1) / 2)
```

但全局 min-max 归一化后，大量位移集中在 0.5 附近：

| 指标 | dx |
| --- | ---: |
| train 位移数 | 566048 |
| 原始 `dx_min` | -34.42 |
| 原始 `dx_max` | 34.39 |
| `dx_norm_mean` | 0.5114 |
| `dx_norm_std` | 0.1372 |
| `|dx_norm - 0.5| <= 0.01` | 33.25% |
| `|dx_norm - 0.5| <= 0.02` | 40.03% |
| `|dx_norm - 0.5| <= 0.05` | 53.18% |

这说明 `[0,1]` 表示虽然解析可逆，但对 DDPM 来说可能过于集中，容易学到平均化纹理。

### 2.5 `[-1,1]` constrained GASF 数据集

输出目录：`week04/data_constrained_gasf_dxdy_start_neg11`

| 项目 | 数值 |
| --- | ---: |
| 成功处理 | 3159 |
| 跳过样本 | 0 |
| 图像大小 | `224 x 224 x 3` |
| 归一化范围 | `[-1,1]` |
| 表示 | `GASF_dx_GASF_dy_start_heatmap_neg11` |

`[-1,1]` 能把位移中心移动到 0 附近，但 GASF 对角线只保留绝对值：

```text
diag = 2x^2 - 1
x in [-1,1] => diagonal recovers |x| only
```

可逆性检查结果：

| 指标 | dx | dy |
| --- | ---: | ---: |
| `mean_abs_diag_mae` | `1.94e-09` | `1.55e-09` |
| `max_abs_diag_mae` | `4.73e-09` | `3.19e-09` |
| positive branch `mean_mae` | 0.1540 | 0.1326 |
| positive branch `max_error` | 2.0 | 2.0 |

结论：`[-1,1]` 的对角线可以精确恢复 `|dx|, |dy|`，但无法恢复符号，因此不满足严格的 signed analytical inversion。除非额外编码符号，否则它更适合作为生成纹理实验，不适合作为完整解析还原管线。

## 3. DDPM 实验设置

所有主实验使用同一训练脚本：`week04/scripts/train_week04_ddpm.py`。

公共设置：

| 参数 | 设置 |
| --- | --- |
| 模型 | `UNet2DModel`，通过项目已有 `build_unet(image_size)` 构建 |
| scheduler | `DDPMScheduler(num_train_timesteps=1000)` |
| optimizer | `AdamW` |
| learning rate | `2e-4` |
| LR schedule | cosine with warmup |
| warmup steps | 500 |
| batch size | 16 |
| gradient accumulation | 2 |
| effective batch size | 32 |
| max train steps | 50000 |
| image size | 224 |
| save/sample interval | 2000 steps |
| EMA | enabled |
| seed | 42 |
| GASF channels constrained | channels 0 and 1 only |
| channel 2 | start heatmap，不参与 GASF structure loss |

辅助 loss 设计：

| `aux_loss_mode` | total loss |
| --- | --- |
| `none` | `MSE(noise_pred, noise)` |
| `symmetry` | `MSE + 0.01 * symmetry_loss` |
| `gasf_structure` | `MSE + 0.01 * symmetry_loss + 0.01 * consistency_loss` |

其中：

```text
symmetry_loss = mean((GASF - GASF^T)^2)
consistency_loss = mean((GASF - rebuild_from_predicted_diagonal)^2)
```

## 4. 主实验矩阵与指标对比

六个主实验来自两种数据归一化乘以三种 loss：

| 实验 | 数据 | 输入格式 | loss | 状态 | 最后 step | 最后有效 total | 最近 100 条 total 均值 | checkpoint |
| --- | --- | --- | --- | --- | ---: | ---: | ---: | --- |
| `week04_gasf_minmax01_mse` | `[0,1]` | PNG | MSE | 健康，已停止 | 40610 | 0.000578 | 0.000549 | `checkpoint-40000` |
| `week04_gasf_minmax01_symmetry` | `[0,1]` | PNG | MSE + symmetry | 健康，已停止 | 40610 | 0.000609 | 0.000585 | `checkpoint-40000` |
| `week04_gasf_minmax01_gasf_structure` | `[0,1]` | PNG | MSE + symmetry + consistency | NaN，已停止 | 41470 | 0.009001 at step 480 | 0.135167 before NaN | 无有效 checkpoint |
| `week04_gasf_neg11_mse` | `[-1,1]` | PNG | MSE | 健康，已停止 | 40620 | 0.000615 | 0.000529 | `checkpoint-40000` |
| `week04_gasf_neg11_symmetry` | `[-1,1]` | PNG | MSE + symmetry | 健康，已停止 | 40620 | 0.000640 | 0.000562 | `checkpoint-40000` |
| `week04_gasf_neg11_gasf_structure` | `[-1,1]` | PNG | MSE + symmetry + consistency | NaN，已停止 | 41470 | 0.006962 at step 610 | 0.106821 before NaN | 无有效 checkpoint |

更细的 loss 分量对比：

| 实验 | 最后有效 step | MSE | symmetry loss | consistency loss |
| --- | ---: | ---: | ---: | ---: |
| `[0,1]` MSE | 40610 | 0.000578 | 0 | 0 |
| `[0,1]` symmetry | 40610 | 0.000597 | 0.001187 | 0.002574 |
| `[0,1]` gasf_structure | 480 | 0.007969 | 0.060069 | 0.043088 |
| `[-1,1]` MSE | 40620 | 0.000615 | 0 | 0 |
| `[-1,1]` symmetry | 40620 | 0.000632 | 0.000823 | 0.003625 |
| `[-1,1]` gasf_structure | 610 | 0.006345 | 0.035741 | 0.025971 |

观察：

1. MSE-only 和 symmetry-only 都稳定训练到 40k+ step，loss 已在 `5e-4` 附近平台化。
2. symmetry loss 对训练稳定性没有明显破坏，但 total loss 比 pure MSE 略高，因为加入了结构正则项。
3. 两个原始 `gasf_structure` run 都在早期出现 NaN，之后日志持续 NaN，因此 40k 采样只能用于诊断，不能作为有效模型质量指标。
4. `[-1,1]` 的 MSE-only 最近 100 条均值略低于 `[0,1]` MSE-only，但由于 `[-1,1]` 存在符号不可逆，不能只凭训练 loss 判断它更适合最终解析管线。

## 5. float array MSE 实验

为验证 PNG 量化对训练和后续解析评估的影响，额外启动了 float array 输入实验：

| 实验 | 数据 | 输入格式 | loss | 状态 | 当前 step | 最近 120 条 total 均值 |
| --- | --- | --- | --- | --- | ---: | ---: |
| `week04_gasf_minmax01_float_mse` | `[0,1]` | `.npy` float32 arrays | MSE | 运行中 | 33390 | 0.000538 |

对应 Slurm job：`1749335`，当前仍在运行。该实验直接读取 `week04/data_constrained_gasf_dxdy_start/arrays`，避免 PNG 的 uint8 量化误差。它是后续做“生成图像对角线解析成轨迹指标”的更合理 baseline。

## 6. 采样与推理设置

已对 `[0,1]` MSE `checkpoint-40000` 重新采样：

| checkpoint | scheduler | inference steps | 样本数 | 输出目录 |
| --- | --- | ---: | ---: | --- |
| `week04_gasf_minmax01_mse/checkpoint-40000` | DDPM | 100 | 16 | `week04/resampled_samples/minmax01_mse_checkpoint40000/ddpm_100_steps` |
| `week04_gasf_minmax01_mse/checkpoint-40000` | DDPM | 150 | 16 | `week04/resampled_samples/minmax01_mse_checkpoint40000/ddpm_150_steps` |

共生成 32 张样本图。当前这一步主要用于视觉检查和采样步数对比；还没有形成 FID、GASF structural error 或解析轨迹 ADE/FDE 的完整量化评估。

## 7. GASF structure NaN 诊断与修复

原始 `gasf_structure` 的问题出现在从预测对角线重建 GASF 的路径：

```text
diag -> (diag + 1) / 2 -> sqrt -> arccos -> cos(phi_i + phi_j)
```

虽然 forward 做了 clamp，但在 `diag` 接近 `-1` 或 `1` 时，`sqrt` 和 `arccos` 的梯度会非常大或奇异。混合精度和早期噪声预测会进一步放大这个问题，导致非有限梯度，训练进入 NaN。

修复已经加入训练脚本：

| 修复项 | 内容 |
| --- | --- |
| interior clamp | `--gasf-diag-eps` 默认 `1e-4`，把 `(diag+1)/2` clamp 到 `[eps, 1-eps]` |
| float32 path | 对角线到角度重建部分使用 float32 计算 |
| 非有限检查 | loss 或 gradient 非有限时写 `nonfinite_debug_step_*.json` 并立即停止 |

短程验证 run：

| 实验 | 数据 | max steps | 实际最后 step | NaN | 最后 total | 最后 MSE | 最后 symmetry | 最后 consistency |
| --- | --- | ---: | ---: | --- | ---: | ---: | ---: | ---: |
| `minmax01_gasf_structure_safe_700` | `[0,1]` | 700 | 170 | 否 | 0.030051 | 0.026302 | 0.216999 | 0.157852 |
| `neg11_gasf_structure_safe_700` | `[-1,1]` | 700 | 180 | 否 | 0.026800 | 0.023449 | 0.193234 | 0.141861 |

这两个 debug job 被手动取消在 170/180 step，但已经越过了原始 minmax01 failure window 的早期趋势，并且没有出现 NaN。严格来说，`neg11` 原始第一次 NaN 在 step 620，所以仍需要完整跑过 700 step 才能确认修复完全覆盖原失败窗口。

## 8. 本周结论

1. 新的 constrained GASF 表示已经构建完成，`[0,1]` 版本在 float precision 下可以从 GASF 对角线近乎无损解析回 `dx, dy` 和轨迹。
2. Delta 对齐实验已全量落盘：Mode A 与 Mode C 的 ADE 接近 0，Mode B 的 mean ADE 为 9.17，确认没有 timestep alignment 问题。
3. PNG 量化是解析重建误差的主要来源，因此后续轨迹级评估应使用 float array 或保存生成样本的 float tensor。
4. `[0,1]` 表示满足严格解析可逆，但位移分布集中在 0.5 附近，可能不利于 DDPM 学习细粒度动态。
5. `[-1,1]` 表示分布更自然，但 GASF 对角线丢失符号，不满足完整解析可逆；除非增加符号通道或额外约束，否则不能作为最终解析管线。
6. 主 DDPM 消融中，MSE-only 与 symmetry-only 稳定，40k checkpoint 可继续用于采样和评估。
7. 原始 gasf_structure loss 会导致 NaN，原因基本定位为 diagonal-to-angle 重建的边界梯度奇异；已加入 epsilon clamp 和非有限检查，短程 debug run 暂时稳定。

## 9. 下一步建议

1. 等 `week04_gasf_minmax01_float_mse` 完成后，与 PNG MSE run 比较 sample quality、GASF symmetry error、diagonal consistency error 和解析轨迹 ADE/FDE。
2. 为已训练的 40k checkpoints 写统一评估脚本：生成 float tensor、计算 GASF symmetry / self-consistency / diagonal range / start heatmap peak quality。
3. 对 `[0,1]` float MSE 样本做解析轨迹恢复，检查生成结果是否会出现位移坍缩、起点漂移或轨迹尺度异常。
4. 完整重跑修复后的 `gasf_structure` 至至少 700 step，确认越过原始 step 490 / 620 NaN 窗口，再决定是否投入 50k 主训练。
5. 如果继续探索 `[-1,1]`，需要设计符号恢复机制，例如额外 sign channel、GADF 辅助通道，或直接改用可保符号的编码方式。
