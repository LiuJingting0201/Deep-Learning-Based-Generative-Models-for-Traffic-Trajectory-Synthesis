"""Run the original-config no-speed decoder invertibility experiment.

This script is intentionally not about diffusion, GANs, FID, maps, or speed.
It verifies the fixed paired dataset, trains the original-style CNN-FC decoder,
evaluates the best validation checkpoint, and writes thesis-ready diagnostics.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT / "scripts") not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT / "scripts"))

from train_decoder_no_speed_baseline import (  # noqa: E402
    LabelScaler,
    OriginalStyleCNNFCDecoder,
    resolve_device,
    set_seed,
)


class EvalDataset(Dataset):
    def __init__(self, data_root: Path, metadata: pd.DataFrame, scaler: LabelScaler) -> None:
        self.data_root = data_root
        self.metadata = metadata.reset_index(drop=True)
        self.scaler = scaler

    def __len__(self) -> int:
        return len(self.metadata)

    def __getitem__(self, index: int) -> dict[str, object]:
        row = self.metadata.iloc[index]
        image = np.asarray(Image.open(self.data_root / row.image_path).convert("RGB"))
        image_tensor = torch.from_numpy(image.astype(np.float32) / 255.0).permute(2, 0, 1)
        label = np.load(self.data_root / row.label_path).astype(np.float32)
        return {
            "image": image_tensor.contiguous(),
            "label": torch.from_numpy(label),
            "sample_id": str(row.sample_id),
            "vehicle_id": str(row.vehicle_id),
        }


def main() -> None:
    args = parse_args()
    set_seed(args.seed)
    data_root = args.data_root.resolve()
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "evaluation" / "plots").mkdir(parents=True, exist_ok=True)

    metadata, split_metadata = verify_data_and_split(data_root)
    write_ablation_template(output_dir / "ablation_config_template.json")

    if not args.skip_training:
        run_training(args)

    summary = evaluate_best_checkpoint(args, split_metadata)
    write_chinese_report(output_dir, summary)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, default=Path("data_no_speed_paired"))
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("experiments/decoder_no_speed_original_config"),
    )
    parser.add_argument("--epochs", type=int, default=1000)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--weight-decay", type=float, default=0.0)
    parser.add_argument("--optimizer", choices=["adam", "adamw"], default="adam")
    parser.add_argument("--dropout", type=float, default=0.3)
    parser.add_argument("--early-stopping-patience", type=int, default=80)
    parser.add_argument("--early-stopping-min-delta", type=float, default=1e-6)
    parser.add_argument("--metrics-every", type=int, default=5)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--skip-training", action="store_true")
    return parser.parse_args()


def verify_data_and_split(data_root: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    metadata_path = data_root / "metadata.csv"
    split_path = data_root / "splits" / "split_metadata.csv"
    if not metadata_path.exists():
        raise FileNotFoundError(metadata_path)
    if not split_path.exists():
        raise FileNotFoundError(split_path)

    metadata = pd.read_csv(metadata_path)
    split_metadata = pd.read_csv(split_path)
    if len(metadata) != len(split_metadata):
        raise RuntimeError(
            f"metadata rows ({len(metadata)}) != split rows ({len(split_metadata)})"
        )
    if set(metadata["sample_id"]) != set(split_metadata["sample_id"]):
        raise RuntimeError("metadata.csv and split_metadata.csv sample_id sets do not match.")
    if split_metadata["sample_id"].duplicated().any():
        raise RuntimeError("split_metadata.csv contains duplicated sample_id values.")

    split_sets = {
        split: set(split_metadata.loc[split_metadata["split"] == split, "sample_id"])
        for split in ["train", "val", "test"]
    }
    if split_sets["train"] & split_sets["val"] or split_sets["train"] & split_sets["test"] or split_sets["val"] & split_sets["test"]:
        raise RuntimeError("Train/val/test split overlap detected.")
    if set.union(*split_sets.values()) != set(split_metadata["sample_id"]):
        raise RuntimeError("Some samples are not assigned to exactly one split.")

    missing_or_bad: list[dict[str, object]] = []
    label_mins: list[np.ndarray] = []
    label_maxs: list[np.ndarray] = []
    image_min = 255
    image_max = 0
    label_dtype = None

    for row in split_metadata.itertuples(index=False):
        image_path = data_root / row.image_path
        label_path = data_root / row.label_path
        if not image_path.exists():
            missing_or_bad.append({"sample_id": row.sample_id, "reason": "missing_image"})
            continue
        if not label_path.exists():
            missing_or_bad.append({"sample_id": row.sample_id, "reason": "missing_label"})
            continue
        with Image.open(image_path) as image:
            if image.mode != "RGB" or image.size != (224, 224):
                missing_or_bad.append(
                    {"sample_id": row.sample_id, "reason": f"bad_image:{image.mode}:{image.size}"}
                )
            arr = np.asarray(image)
            image_min = min(image_min, int(arr.min()))
            image_max = max(image_max, int(arr.max()))
        label = np.load(label_path)
        label_dtype = str(label.dtype)
        if label.shape != (224, 2):
            missing_or_bad.append({"sample_id": row.sample_id, "reason": f"bad_label:{label.shape}"})
        label_mins.append(label.min(axis=0))
        label_maxs.append(label.max(axis=0))

    if missing_or_bad:
        report_path = data_root / "splits" / "missing_samples_report.csv"
        pd.DataFrame(missing_or_bad).to_csv(report_path, index=False)
        raise RuntimeError(f"Dataset consistency check failed. Report: {report_path}")

    counts = split_metadata["split"].value_counts().reindex(["train", "val", "test"])
    xy_min = np.vstack(label_mins).min(axis=0)
    xy_max = np.vstack(label_maxs).max(axis=0)
    print("数据与 split 校验通过")
    print(f"total samples: {len(split_metadata)}")
    print(f"train count: {int(counts['train'])}")
    print(f"val count: {int(counts['val'])}")
    print(f"test count: {int(counts['test'])}")
    print(f"x/y coordinate min: {xy_min.tolist()}")
    print(f"x/y coordinate max: {xy_max.tolist()}")
    print(f"image channel min/max: {image_min}/{image_max}")
    print(f"label dtype: {label_dtype}")
    return metadata, split_metadata


def run_training(args: argparse.Namespace) -> None:
    command = [
        sys.executable,
        str(PROJECT_ROOT / "scripts" / "train_decoder_no_speed_baseline.py"),
        "--data-root",
        str(args.data_root),
        "--output-dir",
        str(args.output_dir),
        "--epochs",
        str(args.epochs),
        "--batch-size",
        str(args.batch_size),
        "--lr",
        str(args.lr),
        "--weight-decay",
        str(args.weight_decay),
        "--optimizer",
        args.optimizer,
        "--dropout",
        str(args.dropout),
        "--early-stopping-patience",
        str(args.early_stopping_patience),
        "--early-stopping-min-delta",
        str(args.early_stopping_min_delta),
        "--metrics-every",
        str(args.metrics_every),
        "--seed",
        str(args.seed),
        "--device",
        args.device,
        "--num-workers",
        str(args.num_workers),
    ]
    print("开始 original-config 训练：")
    print(" ".join(command))
    subprocess.run(command, check=True, cwd=PROJECT_ROOT)


def evaluate_best_checkpoint(args: argparse.Namespace, split_metadata: pd.DataFrame) -> dict[str, object]:
    data_root = args.data_root.resolve()
    output_dir = args.output_dir.resolve()
    evaluation_dir = output_dir / "evaluation"
    predictions_dir = evaluation_dir / "predictions"
    plots_dir = evaluation_dir / "plots"
    predictions_dir.mkdir(parents=True, exist_ok=True)
    plots_dir.mkdir(parents=True, exist_ok=True)

    scaler_data = json.loads((output_dir / "label_scaler.json").read_text())
    scaler = LabelScaler(mean=scaler_data["mean"], std=scaler_data["std"])
    device = resolve_device(args.device)
    checkpoint_name = getattr(args, "checkpoint", "best_model.pt")
    checkpoint_path = output_dir / "checkpoints" / checkpoint_name
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model = OriginalStyleCNNFCDecoder(dropout=args.dropout).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    rows: list[dict[str, object]] = []
    per_timestep_rows: list[dict[str, object]] = []
    predictions: dict[str, dict[str, np.ndarray]] = {}
    truths: dict[str, dict[str, np.ndarray]] = {}

    for split in ["train", "val", "test"]:
        split_frame = split_metadata[split_metadata["split"] == split].copy()
        dataset = EvalDataset(data_root, split_frame, scaler)
        loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)
        split_predictions, split_truths, split_rows, timestep_ade = predict_split(
            model, loader, scaler, device, predictions_dir, split
        )
        predictions[split] = split_predictions
        truths[split] = split_truths
        rows.extend(split_rows)
        per_timestep_rows.extend(
            {"split": split, "timestep": index, "ade": float(value)}
            for index, value in enumerate(timestep_ade)
        )

    per_sample = pd.DataFrame(rows)
    per_sample.to_csv(evaluation_dir / "per_sample_metrics.csv", index=False)
    per_timestep = pd.DataFrame(per_timestep_rows)
    per_timestep.to_csv(evaluation_dir / "per_timestep_ade.csv", index=False)

    metrics_summary = summarize_metrics(per_sample)
    naive_summary = compute_naive_mean_baseline(data_root, split_metadata)
    variance_summary = compute_variance_diagnostics(predictions, truths)

    (evaluation_dir / "metrics_summary.json").write_text(json.dumps(metrics_summary, indent=2))
    (evaluation_dir / "naive_mean_baseline_metrics.json").write_text(json.dumps(naive_summary, indent=2))
    (evaluation_dir / "prediction_variance_diagnostics.json").write_text(json.dumps(variance_summary, indent=2))

    plot_evaluation_outputs(per_sample, per_timestep, predictions["test"], truths["test"], plots_dir)
    return {
        "checkpoint": checkpoint,
        "metrics_summary": metrics_summary,
        "naive_summary": naive_summary,
        "variance_summary": variance_summary,
    }


def predict_split(
    model: OriginalStyleCNNFCDecoder,
    loader: DataLoader,
    scaler: LabelScaler,
    device: torch.device,
    predictions_dir: Path,
    split: str,
) -> tuple[dict[str, np.ndarray], dict[str, np.ndarray], list[dict[str, object]], np.ndarray]:
    predictions: dict[str, np.ndarray] = {}
    truths: dict[str, np.ndarray] = {}
    rows: list[dict[str, object]] = []
    timestep_error_sum = np.zeros(224, dtype=np.float64)
    total = 0

    with torch.no_grad():
        for batch in loader:
            images = batch["image"].to(device)
            pred_norm = model(images)
            pred = scaler.inverse_transform_tensor(pred_norm).cpu().numpy().astype(np.float32)
            true = batch["label"].cpu().numpy().astype(np.float32)
            for index, sample_id in enumerate(batch["sample_id"]):
                sample_id = str(sample_id)
                vehicle_id = str(batch["vehicle_id"][index])
                sample_pred = pred[index]
                sample_true = true[index]
                np.save(predictions_dir / f"{sample_id}_pred.npy", sample_pred)
                metrics, point_errors = compute_metrics(sample_pred, sample_true)
                rows.append(
                    {
                        "sample_id": sample_id,
                        "vehicle_id": vehicle_id,
                        "split": split,
                        **metrics,
                    }
                )
                predictions[sample_id] = sample_pred
                truths[sample_id] = sample_true
                timestep_error_sum += point_errors
                total += 1

    return predictions, truths, rows, timestep_error_sum / max(1, total)


def compute_metrics(pred: np.ndarray, true: np.ndarray) -> tuple[dict[str, float], np.ndarray]:
    diff = pred - true
    point_errors = np.linalg.norm(diff, axis=1)
    pred_length = trajectory_length(pred)
    true_length = trajectory_length(true)
    return (
        {
            "ADE": float(point_errors.mean()),
            "FDE": float(point_errors[-1]),
            "RMSE": float(np.sqrt(np.mean(diff**2))),
            "MAE": float(np.mean(np.abs(diff))),
            "start_error": float(point_errors[0]),
            "end_error": float(point_errors[-1]),
            "trajectory_length_error": float(abs(pred_length - true_length)),
        },
        point_errors,
    )


def trajectory_length(points: np.ndarray) -> float:
    return float(np.linalg.norm(np.diff(points, axis=0), axis=1).sum())


def summarize_metrics(per_sample: pd.DataFrame) -> dict[str, dict[str, dict[str, float]]]:
    summary: dict[str, dict[str, dict[str, float]]] = {}
    for split, frame in per_sample.groupby("split", sort=False):
        summary[split] = {}
        for metric in ["ADE", "FDE", "RMSE", "MAE", "start_error", "end_error", "trajectory_length_error"]:
            values = frame[metric].to_numpy(dtype=float)
            summary[split][metric] = {
                "mean": float(values.mean()),
                "median": float(np.median(values)),
                "std": float(values.std()),
                "min": float(values.min()),
                "max": float(values.max()),
            }
    return summary


def compute_naive_mean_baseline(data_root: Path, split_metadata: pd.DataFrame) -> dict[str, object]:
    train_frame = split_metadata[split_metadata["split"] == "train"]
    train_labels = np.stack([np.load(data_root / row.label_path) for row in train_frame.itertuples()])
    mean_trajectory = train_labels.mean(axis=0).astype(np.float32)

    summary: dict[str, object] = {}
    for split, frame in split_metadata.groupby("split", sort=False):
        rows = []
        for row in frame.itertuples(index=False):
            true = np.load(data_root / row.label_path).astype(np.float32)
            metrics, _ = compute_metrics(mean_trajectory, true)
            rows.append(metrics)
        split_metrics = pd.DataFrame(rows)
        summary[str(split)] = {
            metric: {
                "mean": float(split_metrics[metric].mean()),
                "median": float(split_metrics[metric].median()),
                "std": float(split_metrics[metric].std(ddof=0)),
                "min": float(split_metrics[metric].min()),
                "max": float(split_metrics[metric].max()),
            }
            for metric in ["ADE", "FDE", "RMSE", "MAE", "start_error", "end_error", "trajectory_length_error"]
        }
    return summary


def compute_variance_diagnostics(
    predictions: dict[str, dict[str, np.ndarray]],
    truths: dict[str, dict[str, np.ndarray]],
) -> dict[str, dict[str, float]]:
    diagnostics: dict[str, dict[str, float]] = {}
    for split in ["train", "val", "test"]:
        pred_stack = np.stack(list(predictions[split].values()))
        true_stack = np.stack(list(truths[split].values()))
        pred_var = float(np.var(pred_stack, axis=0).mean())
        true_var = float(np.var(true_stack, axis=0).mean())
        ratio = pred_var / true_var if true_var > 0 else float("nan")
        diagnostics[split] = {
            "ground_truth_variance": true_var,
            "predicted_variance": pred_var,
            "predicted_to_ground_truth_variance_ratio": ratio,
            "possible_regression_to_mean_collapse": bool(ratio < 0.25),
        }
    return diagnostics


def plot_evaluation_outputs(
    per_sample: pd.DataFrame,
    per_timestep: pd.DataFrame,
    test_predictions: dict[str, np.ndarray],
    test_truths: dict[str, np.ndarray],
    plots_dir: Path,
) -> None:
    test_rows = per_sample[per_sample["split"] == "test"].copy()
    plot_hist(test_rows["ADE"], plots_dir / "hist_test_ADE.png", "Test ADE")
    plot_hist(test_rows["FDE"], plots_dir / "hist_test_FDE.png", "Test FDE")
    plot_scatter(test_rows, plots_dir / "scatter_ADE_vs_FDE.png")
    plot_per_timestep(per_timestep, plots_dir / "per_timestep_ADE_curve.png")
    plot_groups(test_rows, test_predictions, test_truths, plots_dir)


def plot_hist(values: pd.Series, path: Path, title: str) -> None:
    fig, axis = plt.subplots(figsize=(6, 4))
    axis.hist(values, bins=30)
    axis.set_title(title)
    axis.set_xlabel(title)
    axis.set_ylabel("count")
    axis.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def plot_scatter(test_rows: pd.DataFrame, path: Path) -> None:
    fig, axis = plt.subplots(figsize=(5, 5))
    axis.scatter(test_rows["ADE"], test_rows["FDE"], s=16, alpha=0.8)
    axis.set_xlabel("ADE")
    axis.set_ylabel("FDE")
    axis.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def plot_per_timestep(per_timestep: pd.DataFrame, path: Path) -> None:
    fig, axis = plt.subplots(figsize=(7, 4))
    for split, frame in per_timestep.groupby("split", sort=False):
        axis.plot(frame["timestep"], frame["ade"], label=split)
    axis.set_xlabel("timestep")
    axis.set_ylabel("ADE")
    axis.legend()
    axis.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def plot_groups(
    test_rows: pd.DataFrame,
    predictions: dict[str, np.ndarray],
    truths: dict[str, np.ndarray],
    plots_dir: Path,
) -> None:
    sorted_rows = test_rows.sort_values("ADE").reset_index(drop=True)
    groups = {
        "best": sorted_rows.head(10),
        "median": sorted_rows.iloc[
            max(0, len(sorted_rows) // 2 - 5) : min(len(sorted_rows), len(sorted_rows) // 2 + 5)
        ],
        "worst": sorted_rows.tail(10).sort_values("ADE", ascending=False),
    }
    for group_name, frame in groups.items():
        group_dir = plots_dir / group_name
        group_dir.mkdir(parents=True, exist_ok=True)
        panel_paths = []
        for row in frame.itertuples(index=False):
            sample_id = str(row.sample_id)
            path = group_dir / f"{sample_id}_trajectory.png"
            plot_prediction(
                sample_id,
                str(row.vehicle_id),
                truths[sample_id],
                predictions[sample_id],
                float(row.ADE),
                float(row.FDE),
                path,
            )
            panel_paths.append(path)
        make_contact_sheet(panel_paths, plots_dir / f"{group_name}_10_grid.png")


def plot_prediction(
    sample_id: str,
    vehicle_id: str,
    true: np.ndarray,
    pred: np.ndarray,
    ade: float,
    fde: float,
    path: Path,
) -> None:
    fig, axis = plt.subplots(figsize=(5, 5))
    axis.plot(true[:, 0], true[:, 1], label="ground truth", linewidth=1.5)
    axis.plot(pred[:, 0], pred[:, 1], label="prediction", linewidth=1.5)
    axis.scatter(true[0, 0], true[0, 1], s=22, marker="o", label="gt start")
    axis.scatter(true[-1, 0], true[-1, 1], s=22, marker="s", label="gt end")
    axis.scatter(pred[0, 0], pred[0, 1], s=26, marker="x", label="pred start")
    axis.scatter(pred[-1, 0], pred[-1, 1], s=26, marker="^", label="pred end")
    all_xy = np.vstack([true, pred])
    pad_x = max(1.0, float(np.ptp(all_xy[:, 0])) * 0.05)
    pad_y = max(1.0, float(np.ptp(all_xy[:, 1])) * 0.05)
    axis.set_xlim(float(all_xy[:, 0].min() - pad_x), float(all_xy[:, 0].max() + pad_x))
    axis.set_ylim(float(all_xy[:, 1].min() - pad_y), float(all_xy[:, 1].max() + pad_y))
    axis.set_aspect("equal", adjustable="box")
    axis.set_title(f"{sample_id} {vehicle_id} ADE={ade:.2f} FDE={fde:.2f}")
    axis.legend(fontsize=7)
    axis.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def make_contact_sheet(image_paths: list[Path], output_path: Path) -> None:
    if not image_paths:
        return
    thumbs = [Image.open(path).convert("RGB").resize((320, 320)) for path in image_paths]
    cols = 5
    rows = int(np.ceil(len(thumbs) / cols))
    sheet = Image.new("RGB", (cols * 320, rows * 320), "white")
    for index, thumb in enumerate(thumbs):
        sheet.paste(thumb, ((index % cols) * 320, (index // cols) * 320))
    sheet.save(output_path)


def write_ablation_template(path: Path) -> None:
    template = {
        "purpose": "Run one controlled ablation at a time; do not launch a full grid search.",
        "batch_size_options": [8, 16, 32],
        "learning_rate_options": [1e-4, 5e-5, 1e-5],
        "optimizer_options": [
            {"optimizer": "adam", "weight_decay": 0.0},
            {"optimizer": "adamw", "weight_decay": 1e-4},
        ],
        "dropout_options": [0.0, 0.1, 0.3, 0.5],
        "label_normalization_options": ["train_mean_std", "train_min_max"],
        "example_command": (
            "python scripts/train_decoder_no_speed_baseline.py "
            "--data-root data_no_speed_paired "
            "--output-dir experiments/decoder_ablation_lr_5e-5 "
            "--epochs 1000 --batch-size 8 --lr 5e-5 --optimizer adam "
            "--weight-decay 0.0 --dropout 0.3 --early-stopping-patience 80"
        ),
    }
    path.write_text(json.dumps(template, indent=2))


def write_chinese_report(output_dir: Path, summary: dict[str, object]) -> None:
    metrics = summary["metrics_summary"]
    naive = summary["naive_summary"]
    variance = summary["variance_summary"]
    best = json.loads((output_dir / "best_summary.json").read_text())
    train_log = pd.read_csv(output_dir / "train_log.csv")
    final_epoch = int(best["final_epoch"])
    best_epoch = int(best["best_epoch"])
    early_stopped = bool(best["early_stopping_triggered"])

    test_ade = metrics["test"]["ADE"]["mean"]
    test_fde = metrics["test"]["FDE"]["mean"]
    naive_ade = naive["test"]["ADE"]["mean"]
    naive_fde = naive["test"]["FDE"]["mean"]
    ade_improvement = (naive_ade - test_ade) / naive_ade * 100.0
    fde_improvement = (naive_fde - test_fde) / naive_fde * 100.0

    report = f"""
原始配置 CNN-FC decoder 表示可逆性实验报告

1. 收敛情况：best validation loss 出现在 epoch {best_epoch}，总训练到 epoch {final_epoch}。
2. Early stopping：{'已触发' if early_stopped else '未触发'}。
3. DNN ADE/FDE：
   train mean ADE={metrics['train']['ADE']['mean']:.4f}, mean FDE={metrics['train']['FDE']['mean']:.4f}
   val   mean ADE={metrics['val']['ADE']['mean']:.4f}, mean FDE={metrics['val']['FDE']['mean']:.4f}
   test  mean ADE={test_ade:.4f}, mean FDE={test_fde:.4f}
4. 过拟合证据：train/val/test 误差若明显分离才算强过拟合；本次请结合 loss/ADE/FDE 曲线判断。
5. 欠拟合证据：如果 train、val、test 都较高且接近，说明原始 CNN-FC decoder 可能容量/优化不足，或图像表示本身丢失了部分可逆信息。
6. Naive mean baseline：
   naive test mean ADE={naive_ade:.4f}, mean FDE={naive_fde:.4f}
   DNN 相对 naive ADE 改善={ade_improvement:.2f}%，FDE 改善={fde_improvement:.2f}%。
7. Regression-to-mean collapse：
   test predicted/GT variance ratio={variance['test']['predicted_to_ground_truth_variance_ratio']:.4f}。
   若该比例远小于 1，尤其低于 0.25，说明预测多样性不足，可能在向平均轨迹坍缩。
8. 定性图：
   已保存 best/median/worst test 轨迹图和 contact sheet。请重点看 worst 是否形状错误、端点偏移，median 是否只是粗略位置恢复。
9. 结论：
   本实验只回答 no-speed Hilbert/GAF/MTF RGB 表示能否被原始 CNN-FC decoder 解码回 224 点 x/y 轨迹。
   它不评价 FID，也不涉及 diffusion/GAN/map channel/map matching。
"""
    (output_dir / "interpretation_report_zh.txt").write_text(report.strip() + "\n")
    print(report.strip())
    print("训练日志最后几行：")
    print(train_log.tail().to_string(index=False))


if __name__ == "__main__":
    main()
