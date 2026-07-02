# Week05 Summary: Delta Displacement DDPM Dataset and Training Matrix

## 1. Goal

Week05 的目标是把车辆轨迹数据整理成适合 DDPM 的固定长度训练输入，并基于 Week04 的 DDPM 训练框架建立一组 Week05 消融实验。

核心问题有两个：

1. 原始 `vehicle_positions_TS_New.csv` 中每个 `id` 的轨迹长度不统一，并不是所有轨迹都有 224 个采样点。
2. Week04 的 DDPM 训练脚本面向 `224 x 224 x 3` 的图像/float array 输入；Week05 的轨迹标签是 `224 x 2` 的 delta displacement，需要转换成 DDPM 可训练的 3 通道 float array 表示。

最终采用的主数据集是：

```text
/home/jliu/Thesis/week05/data/delta_displacement_paired_regularized_bounce224
```

训练矩阵使用 2 种 sigmoid normalization 和 3 种 loss，共 6 个实验。

## 2. Source Data

原始 CSV：

```text
/home/jliu/Thesis/week05/data/vehicle_positions_TS_New.csv
```

字段：

```text
time,id,x,y,type
```

原始数据统计：

```text
raw rows: 2,189,591
vehicle ids / source trajectories: 12,061
trajectory length min: 13
trajectory length max: 657
trajectory length mean: 181.5431
trajectory length median: 173
```

对每个 `id` 检查采样点数量后发现：

```text
exactly 224 points: 40 trajectories
not 224 points: 12,021 trajectories
```

因此不能直接假设每个 `id` 都是固定长度 224。

## 3. Variable-Length Delta Dataset

脚本：

```text
week05/scripts/build_delta_displacement_paired_data.py
week05/slurm/build_delta_displacement_paired_data.slurm
```

输出：

```text
week05/data/delta_displacement_paired/
  labels_absolute/*.npy
  labels_delta_displacement/*.npy
  starts/*.npy
  metadata.csv
  skipped_ids.csv
  unique_vehicle_ids.csv
  splits/train_ids.csv
  splits/val_ids.csv
  splits/test_ids.csv
  dataset_summary.json
  normalization_diagnostics.json
  normalization_plots/
```

处理逻辑：

1. 按 `id` 分组。
2. 按 `time` 排序。
3. 保存 absolute trajectory，shape 为 `(N, 2)`。
4. 重新计算 delta displacement：

```python
delta[0] = 0
delta[1:] = absolute[1:] - absolute[:-1]
```

5. 保存 start point：

```python
start = absolute[0]
```

6. 检查 `integrate_delta(start, delta)` 能重建 absolute。

结果：

```text
num_output_samples: 12,061
train: 9,648
val: 1,206
test: 1,207
max_reconstruction_error: 3.18e-12
```

Train split delta normalization 统计：

```text
num_train_values: 1,755,022
dx_min: -37.4000
dx_max: 35.8200
dx_mean: 0.0208
dx_std: 10.2713
dy_min: -37.1100
dy_max: 33.9800
dy_mean: -0.0289
dy_std: 8.5296
dx_norm ratio [0.45,0.55]: 0.4903
dy_norm ratio [0.45,0.55]: 0.5371
```

## 4. Fixed-Length Dataset Option A: Sliding Windows

脚本：

```text
week05/scripts/window_existing_delta_dataset.py
```

输出：

```text
week05/data/delta_displacement_paired_window224/
```

目标：

从已经构建好的 variable-length delta dataset 中切出固定长度 224 的窗口。该流程不回到原始 CSV。

默认配置：

```text
target_length: 224
stride: 112
include_tail: false
```

逻辑：

1. 对每条 source trajectory 读取 `labels_absolute/*.npy`。
2. 如果长度 `< 224`，跳过并记录到 `skipped_samples.csv`。
3. 如果长度 `>= 224`，按 stride 切窗口。
4. 每个窗口重新计算 delta 和 start。
5. 基于 train split 重新计算 normalization diagnostics。

结果：

```text
num_source_samples: 12,061
num_output_windows: 3,877
num_skipped_short_samples: 8,738
short_sample_ratio: 0.7245
train: 3,101
val: 387
test: 389
windows_per_source min: 0
windows_per_source max: 4
windows_per_source mean: 0.3214
```

Train split delta normalization 统计：

```text
num_train_values: 694,624
dx_min: -33.7000
dx_max: 34.7200
dx_mean: 1.6263
dx_std: 9.9268
dy_min: -33.4000
dy_max: 33.3100
dy_mean: 0.0867
dy_std: 7.2104
dx_norm ratio [0.45,0.55]: 0.5237
dy_norm ratio [0.45,0.55]: 0.6355
```

结论：

Sliding window 只保留长度至少 224 的轨迹，因此会丢掉 72.45% 的 source trajectories。它适合作为“只用真实连续 224 点窗口”的对照数据集，但不是主训练数据集。

## 5. Fixed-Length Dataset Option B: Bounce Regularization

脚本：

```text
week05/scripts/regularize_existing_delta_dataset_to_224_by_bounce.py
```

输出：

```text
week05/data/delta_displacement_paired_regularized_bounce224/
```

这是 Week05 DDPM 矩阵使用的主数据集。

目标：

把每条 variable-length trajectory 都统一成长度 224，尽量复刻旧数据的固定长度逻辑，同时避免 zero padding 和 interpolation。

核心规则：

1. `N < 224`：使用 bounce / reflect padding。
2. `N = 224`：保持不变。
3. `N > 224`：默认截断前 224 点。
4. 所有输出都重新计算 delta displacement，不使用旧 delta。

Bounce 逻辑：

```python
base = [0, 1, 2, ..., N-1, N-2, ..., 1]
indices = repeat(base)[:224]
regularized_abs = abs_xy[indices]
```

对于 `N=2`，base 是：

```text
[0, 1]
```

不会出现重复错误。

结果：

```text
num_source_samples: 12,061
num_output_samples: 12,061
num_skipped_samples: 0
target_length: 224
min_length: 2
long_mode: truncate
include_tail: false
```

Regularization method counts：

```text
bounce_reflect_padding: 8,738
truncate_first_224: 3,283
unchanged: 40
```

Split counts：

```text
train: 9,648
val: 1,206
test: 1,207
```

Reconstruction check：

```text
max_reconstruction_error: 3.41e-12
```

Train split delta normalization 统计：

```text
num_train_values: 2,161,152
dx_min: -37.4000
dx_max: 37.4000
dx_mean: 0.1252
dx_std: 10.4037
dy_min: -37.1100
dy_max: 37.1100
dy_mean: -0.0899
dy_std: 8.9178
dx_norm ratio [0.45,0.55]: 0.4811
dy_norm ratio [0.45,0.55]: 0.5427
```

Sigmoid interval ratios on regularized train split：

```text
sigmoid k=1.0:
  dx [0.45,0.55]: 0.3809
  dy [0.45,0.55]: 0.3911

sigmoid k=1.5:
  dx [0.45,0.55]: 0.3122
  dy [0.45,0.55]: 0.3327

sigmoid k=2.0:
  dx [0.45,0.55]: 0.2705
  dy [0.45,0.55]: 0.2954
```

结论：

Bounce regularization 保留了全部 12,061 条 source trajectories，是当前 Week05 DDPM 的主数据源。代价是短轨迹的后半段包含反射运动，delta 分布中会出现由折返带来的方向变化。

## 6. DDPM Input Representation

训练脚本：

```text
week05/scripts/train_week05_ddpm.py
```

该脚本基于 Week04 DDPM 训练脚本改造，但 Week05 不直接读取 PNG，也不读取预生成的 `arrays/*.npy`。

训练时从 dataset root 即时读取：

```text
labels_delta_displacement/<sample_id>.npy
labels_absolute/<sample_id>.npy
starts/<sample_id>.npy
normalization_diagnostics.json
splits/train_ids.csv
```

然后在线生成 `3 x 224 x 224` tensor。

三通道定义：

```text
R channel: GASF(sigmoid-normalized dx)
G channel: GASF(sigmoid-normalized dy)
B channel: start-position heatmap
```

Normalization：

```python
z = (value - train_mean) / train_std
value_norm = sigmoid(k * z)
```

其中 `train_mean/train_std` 来自：

```text
delta_displacement_paired_regularized_bounce224/normalization_diagnostics.json
```

GASF encoding：

```python
phi = arccos(value_norm)
gasf = cos(phi_i + phi_j)
channel = (gasf + 1) / 2
```

DDPM 输入范围：

```python
tensor = float_array * 2 - 1
```

因此模型看到的是 `[-1, 1]` 范围的 3 通道 tensor。

重要修正：

1. `position_stats` 永远使用 train split 计算，不随当前 split 改变。
2. 每次训练会保存 `startup_config.json`。
3. `startup_config.json` 中记录：

```text
data_root
output_dir
normalization_file
normalization_stats
position_stats
num_inputs
sigmoid_k
aux_loss_mode
diag_weight
symmetry_weight
image_size
batch_size
max_train_steps
seed
data_format
png_inputs_used
```

统一 data format 描述：

```text
float_gasf_generated_on_the_fly_from_delta_npy
```

Sanity check：

```text
train samples: 9,648
val samples: 1,206
example tensor shape: (3, 224, 224)
dtype: torch.float32
range: [-1.0, 1.0]
```

## 7. Loss Definitions

Week05 训练矩阵包含三类 loss。

### Pure MSE

CLI：

```text
--aux-loss-mode none
```

Loss：

```python
MSE(noise_pred, noise)
```

### MSE + Diagonal Loss

CLI：

```text
--aux-loss-mode diag
--diag-weight 0.05
```

逻辑：

1. 由 noisy image 和 noise prediction 预测 `x0_pred`。
2. 取 `x0_pred` 前两个 GASF channels 的 diagonal。
3. 和 clean image 前两个 channels 的 diagonal 做 MSE。

Loss：

```python
total = mse_loss + 0.05 * diag_loss
```

动机：

GASF diagonal 携带原始 normalized time-series 的重要信息。对 diagonal 加约束可以鼓励生成结果保留可解码的 dx/dy 序列结构。

### MSE + Symmetry Loss

CLI：

```text
--aux-loss-mode symmetry
--symmetry-weight 0.05
```

逻辑：

GASF 理论上是对称矩阵，因此对预测 `x0_pred` 前两个 channels 加 symmetry loss：

```python
symmetry_loss = mean((gasf - gasf.T)^2)
total = mse_loss + 0.05 * symmetry_loss
```

动机：

鼓励生成的 GASF channels 保持合法结构，而不是只在像素层面匹配噪声预测目标。

## 8. Training Matrix

所有实验使用同一个 dataset root：

```text
/home/jliu/Thesis/week05/data/delta_displacement_paired_regularized_bounce224
```

共同训练配置：

```text
partition: gpu_a40
gpu: 1
cpus-per-task: 4
mem: 16G
time: 1-00:00:00
image_size: 224
channels: 3
batch_size: 8
lr: 2e-4
max_train_steps: 50,000
gradient_accumulation_steps: 2
seed: 42
num_train_timesteps: 1000
lr_warmup_steps: 500
scheduler: DDPMScheduler
model architecture: scripts.train_ddpm.build_unet
sample_every: 2,000
save_every: 2,000
use_ema: true
```

实验列表：

| Experiment | Sigmoid k | Loss | Output |
|---|---:|---|---|
| `week05_sig10_mse` | 1.0 | pure MSE | `week05/results/week05_sig10_mse` |
| `week05_sig10_mse_diag` | 1.0 | MSE + diagonal loss | `week05/results/week05_sig10_mse_diag` |
| `week05_sig10_mse_sym` | 1.0 | MSE + symmetry loss | `week05/results/week05_sig10_mse_sym` |
| `week05_sig15_mse` | 1.5 | pure MSE | `week05/results/week05_sig15_mse` |
| `week05_sig15_mse_diag` | 1.5 | MSE + diagonal loss | `week05/results/week05_sig15_mse_diag` |
| `week05_sig15_mse_sym` | 1.5 | MSE + symmetry loss | `week05/results/week05_sig15_mse_sym` |

Slurm files：

```text
week05/slurm/week05_sig10_mse.slurm
week05/slurm/week05_sig10_mse_diag.slurm
week05/slurm/week05_sig10_mse_sym.slurm
week05/slurm/week05_sig15_mse.slurm
week05/slurm/week05_sig15_mse_diag.slurm
week05/slurm/week05_sig15_mse_sym.slurm
```

Batch submit script：

```text
week05/slurm/submit_week05_matrix.sh
```

提交顺序：

```bash
sbatch /home/jliu/Thesis/week05/slurm/week05_sig10_mse.slurm
sbatch /home/jliu/Thesis/week05/slurm/week05_sig10_mse_diag.slurm
sbatch /home/jliu/Thesis/week05/slurm/week05_sig10_mse_sym.slurm
sbatch /home/jliu/Thesis/week05/slurm/week05_sig15_mse.slurm
sbatch /home/jliu/Thesis/week05/slurm/week05_sig15_mse_diag.slurm
sbatch /home/jliu/Thesis/week05/slurm/week05_sig15_mse_sym.slurm
```

每个 Slurm job 都会打印完整 config，包括：

```text
EXPERIMENT_NAME
DATA_ROOT
NORMALIZATION_FILE
OUTPUT_DIR
SIGMOID_K
AUX_LOSS_MODE
DIAG_WEIGHT
SYMMETRY_WEIGHT
IMAGE_SIZE
CHANNELS
BATCH_SIZE
LR
MAX_TRAIN_STEPS
SEED
NUM_TRAIN_TIMESTEPS
SCHEDULER
MODEL_ARCHITECTURE
DATA_FORMAT
```

日志路径：

```text
week05/logs/<experiment_name>_%j.out
week05/logs/<experiment_name>_%j.err
```

## 9. Important Notes

### DIAG_WEIGHT and SYMMETRY_WEIGHT in pure MSE jobs

所有 Slurm 文件中都 echo 和传入：

```text
DIAG_WEIGHT=0.05
SYMMETRY_WEIGHT=0.05
```

这不影响 pure MSE 实验。

训练脚本逻辑是：

```text
aux_loss_mode=none      -> only MSE, both weights unused
aux_loss_mode=diag      -> uses diag_weight only
aux_loss_mode=symmetry  -> uses symmetry_weight only
```

### Time limit

`gpu_a40` 上已确认 1 天配置曾经可用，因此 Slurm 文件使用：

```text
#SBATCH --time=1-00:00:00
```

之前 `1-12:00:00` 可能触发 `PartitionTimeLimit`。

### Slurm submit status

6 个 Week05 jobs 已通过 `submit_week05_matrix.sh` 提交。实际运行状态以 HPC 上：

```bash
squeue -u jliu
```

为准。

## 10. Generated and Modified Files

Week05 scripts：

```text
week05/scripts/build_delta_displacement_paired_data.py
week05/scripts/window_existing_delta_dataset.py
week05/scripts/regularize_existing_delta_dataset_to_224_by_bounce.py
week05/scripts/train_week05_ddpm.py
```

Week05 Slurm：

```text
week05/slurm/build_delta_displacement_paired_data.slurm
week05/slurm/submit_week05_matrix.sh
week05/slurm/week05_sig10_mse.slurm
week05/slurm/week05_sig10_mse_diag.slurm
week05/slurm/week05_sig10_mse_sym.slurm
week05/slurm/week05_sig15_mse.slurm
week05/slurm/week05_sig15_mse_diag.slurm
week05/slurm/week05_sig15_mse_sym.slurm
```

Week05 data products：

```text
week05/data/delta_displacement_paired/
week05/data/delta_displacement_paired_window224/
week05/data/delta_displacement_paired_regularized_bounce224/
```

Main training data：

```text
week05/data/delta_displacement_paired_regularized_bounce224/
```

Expected training outputs：

```text
week05/results/week05_sig10_mse/
week05/results/week05_sig10_mse_diag/
week05/results/week05_sig10_mse_sym/
week05/results/week05_sig15_mse/
week05/results/week05_sig15_mse_diag/
week05/results/week05_sig15_mse_sym/
```

Each run should contain:

```text
startup_config.json
train_log.jsonl
checkpoint-*/
checkpoint-final/
samples/
train_summary.json
```

## 11. Recommended Next Checks

After jobs start running:

1. Check Slurm logs:

```bash
tail -f week05/logs/week05_sig10_mse_<jobid>.out
```

2. Confirm `startup_config.json` exists in each result directory.
3. Confirm first log line appears in `train_log.jsonl`.
4. Watch for non-finite debug files:

```bash
find week05/results -name 'nonfinite_debug_step_*.json'
```

5. After checkpoints are produced, compare:

```text
pure MSE vs diagonal loss vs symmetry loss
sigmoid k=1.0 vs sigmoid k=1.5
```

Primary evaluation should inspect:

```text
channel distributions
GASF symmetry
diagonal reconstructability
decoded delta trajectory validity
trajectory diversity
nearest-neighbor / memorization diagnostics
```
