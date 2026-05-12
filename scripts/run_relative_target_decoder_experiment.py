"""Train/evaluate original CNN-FC decoder with relative trajectory targets."""

from __future__ import annotations

import argparse
import csv
import json
import random
import shutil
import time
from dataclasses import asdict, dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from PIL import Image
from torch import nn
from torch.utils.data import DataLoader, Dataset


@dataclass(frozen=True)
class LabelScaler:
    mean: list[float]
    std: list[float]

    def transform(self, values: np.ndarray) -> np.ndarray:
        return (values - np.asarray(self.mean, dtype=np.float32)) / np.asarray(
            self.std, dtype=np.float32
        )

    def inverse_tensor(self, values: torch.Tensor) -> torch.Tensor:
        mean = torch.tensor(self.mean, dtype=values.dtype, device=values.device).view(1, 1, 2)
        std = torch.tensor(self.std, dtype=values.dtype, device=values.device).view(1, 1, 2)
        return values * std + mean


class RelativeTrajectoryDataset(Dataset):
    def __init__(self, data_root: Path, metadata: pd.DataFrame, scaler: LabelScaler) -> None:
        self.data_root = data_root
        self.metadata = metadata.reset_index(drop=True)
        self.scaler = scaler

    def __len__(self) -> int:
        return len(self.metadata)

    def __getitem__(self, index: int) -> dict[str, object]:
        row = self.metadata.iloc[index]
        image = np.asarray(Image.open(self.data_root / row.image_path).convert("RGB"))
        image = torch.from_numpy(image.astype(np.float32) / 255.0).permute(2, 0, 1)
        absolute = np.load(self.data_root / row.label_path).astype(np.float32)
        start = absolute[0].copy()
        relative = absolute - start
        relative_norm = self.scaler.transform(relative)
        return {
            "image": image.contiguous(),
            "relative_norm": torch.from_numpy(relative_norm.astype(np.float32)),
            "relative": torch.from_numpy(relative.astype(np.float32)),
            "absolute": torch.from_numpy(absolute.astype(np.float32)),
            "start": torch.from_numpy(start.astype(np.float32)),
            "sample_id": str(row.sample_id),
            "vehicle_id": str(row.vehicle_id),
        }


class OriginalStyleCNNFCDecoder(nn.Module):
    def __init__(self, dropout: float = 0.3) -> None:
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 24, kernel_size=3),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Conv2d(24, 48, kernel_size=3),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Conv2d(48, 1024, kernel_size=2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),
        )
        self.regressor = nn.Sequential(
            nn.Linear(1024, 2048),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(2048, 1024),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(1024, 448),
        )

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        return self.regressor(self.features(images)).view(-1, 224, 2)


def main() -> None:
    args = parse_args()
    set_seed(args.seed)
    data_root = args.data_root.resolve()
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "checkpoints").mkdir(exist_ok=True)
    (output_dir / "plots").mkdir(exist_ok=True)

    split_metadata = pd.read_csv(data_root / "splits" / "split_metadata.csv")
    train_frame = split_metadata[split_metadata["split"] == "train"].copy()
    val_frame = split_metadata[split_metadata["split"] == "val"].copy()
    scaler = fit_relative_scaler(data_root, train_frame)
    (output_dir / "label_scaler.json").write_text(json.dumps(asdict(scaler), indent=2))
    shutil.copy2(data_root / "splits" / "split_metadata.csv", output_dir / "split_copy.csv")
    write_config(args, output_dir)

    if not args.skip_training:
        train(args, data_root, output_dir, train_frame, val_frame, scaler)
    evaluate(args, data_root, output_dir, split_metadata, scaler)
    compare_with_absolute(args, data_root, output_dir, split_metadata)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, default=Path("data_no_speed_paired"))
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("experiments/decoder_no_speed_relative_target"),
    )
    parser.add_argument(
        "--absolute-experiment-dir",
        type=Path,
        default=Path("experiments/decoder_no_speed_original_config"),
    )
    parser.add_argument("--epochs", type=int, default=1000)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--weight-decay", type=float, default=0.0)
    parser.add_argument("--dropout", type=float, default=0.3)
    parser.add_argument("--patience", type=int, default=80)
    parser.add_argument("--min-delta", type=float, default=1e-6)
    parser.add_argument("--metrics-every", type=int, default=5)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--skip-training", action="store_true")
    return parser.parse_args()


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def resolve_device(device_arg: str) -> torch.device:
    if device_arg == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(device_arg)


def fit_relative_scaler(data_root: Path, train_frame: pd.DataFrame) -> LabelScaler:
    rels = []
    for row in train_frame.itertuples(index=False):
        absolute = np.load(data_root / row.label_path).astype(np.float32)
        rels.append(absolute - absolute[0])
    stacked = np.concatenate(rels, axis=0)
    std = stacked.std(axis=0)
    std = np.where(std < 1e-6, 1.0, std)
    return LabelScaler(mean=stacked.mean(axis=0).tolist(), std=std.tolist())


def write_config(args: argparse.Namespace, output_dir: Path) -> None:
    config = {k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()}
    config.update(
        {
            "target": "relative_gt = gt - gt[0]",
            "model": "OriginalStyleCNNFCDecoder",
            "optimizer": "Adam",
            "scheduler": "none",
            "loss": "MSELoss in normalized relative-coordinate space",
        }
    )
    (output_dir / "config.json").write_text(json.dumps(config, indent=2))


def train(
    args: argparse.Namespace,
    data_root: Path,
    output_dir: Path,
    train_frame: pd.DataFrame,
    val_frame: pd.DataFrame,
    scaler: LabelScaler,
) -> None:
    device = resolve_device(args.device)
    train_loader = DataLoader(
        RelativeTrajectoryDataset(data_root, train_frame, scaler),
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=device.type == "cuda",
    )
    val_loader = DataLoader(
        RelativeTrajectoryDataset(data_root, val_frame, scaler),
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=device.type == "cuda",
    )
    model = OriginalStyleCNNFCDecoder(dropout=args.dropout).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    criterion = nn.MSELoss()
    best_val = float("inf")
    best_epoch = 0
    bad_epochs = 0
    early_stopped = False
    rows: list[dict[str, object]] = []
    start = time.perf_counter()

    for epoch in range(1, args.epochs + 1):
        train_loss = run_train_epoch(model, train_loader, optimizer, criterion, device)
        compute_metrics = epoch == 1 or epoch % args.metrics_every == 0
        train_eval = evaluate_loader(model, train_loader, criterion, scaler, device, compute_metrics)
        val_eval = evaluate_loader(model, val_loader, criterion, scaler, device, compute_metrics)
        val_loss = val_eval["loss"]
        improved = val_loss < best_val - args.min_delta
        checkpoint = {
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "train_loss": train_loss,
            "val_loss": val_loss,
            "label_scaler": asdict(scaler),
        }
        torch.save(checkpoint, output_dir / "checkpoints" / "last_model.pt")
        if improved:
            best_val = val_loss
            best_epoch = epoch
            bad_epochs = 0
            torch.save(checkpoint, output_dir / "checkpoints" / "best_model.pt")
        else:
            bad_epochs += 1
        rows.append(
            {
                "epoch": epoch,
                "train_loss": train_loss,
                "train_eval_loss": train_eval["loss"],
                "val_loss": val_loss,
                "train_relative_ADE": train_eval["relative_ADE"],
                "train_relative_FDE": train_eval["relative_FDE"],
                "val_relative_ADE": val_eval["relative_ADE"],
                "val_relative_FDE": val_eval["relative_FDE"],
                "learning_rate": args.lr,
                "elapsed_time": time.perf_counter() - start,
            }
        )
        write_train_log(output_dir / "train_log.csv", rows)
        plot_training_curves(rows, output_dir / "plots")
        print(
            f"epoch={epoch:03d}/{args.epochs} train_loss={train_loss:.6f} "
            f"val_loss={val_loss:.6f} val_rel_ADE={fmt(val_eval['relative_ADE'])} "
            f"best_val={best_val:.6f} best_epoch={best_epoch}"
        )
        if bad_epochs >= args.patience:
            early_stopped = True
            print(f"Early stopping triggered after {bad_epochs} epochs without improvement.")
            break

    (output_dir / "best_summary.json").write_text(
        json.dumps(
            {
                "best_epoch": best_epoch,
                "best_val_loss": best_val,
                "final_epoch": int(rows[-1]["epoch"]),
                "early_stopping_triggered": early_stopped,
                "patience": args.patience,
                "min_delta": args.min_delta,
            },
            indent=2,
        )
    )


def run_train_epoch(model, loader, optimizer, criterion, device) -> float:
    model.train()
    total = 0.0
    n = 0
    for batch in loader:
        images = batch["image"].to(device, non_blocking=True)
        labels = batch["relative_norm"].to(device, non_blocking=True)
        pred = model(images)
        loss = criterion(pred, labels)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
        optimizer.step()
        total += float(loss.detach().cpu()) * images.shape[0]
        n += images.shape[0]
    return total / max(1, n)


def evaluate_loader(model, loader, criterion, scaler, device, compute_metrics: bool) -> dict[str, float]:
    model.eval()
    total = 0.0
    n = 0
    ade = 0.0
    fde = 0.0
    with torch.no_grad():
        for batch in loader:
            images = batch["image"].to(device, non_blocking=True)
            labels_norm = batch["relative_norm"].to(device, non_blocking=True)
            pred_norm = model(images)
            loss = criterion(pred_norm, labels_norm)
            bs = images.shape[0]
            total += float(loss.detach().cpu()) * bs
            n += bs
            if compute_metrics:
                pred_rel = scaler.inverse_tensor(pred_norm)
                true_rel = batch["relative"].to(device)
                point_errors = torch.linalg.norm(pred_rel - true_rel, dim=2)
                ade += float(point_errors.mean(dim=1).sum().cpu())
                fde += float(point_errors[:, -1].sum().cpu())
    return {
        "loss": total / max(1, n),
        "relative_ADE": ade / max(1, n) if compute_metrics else float("nan"),
        "relative_FDE": fde / max(1, n) if compute_metrics else float("nan"),
    }


def evaluate(
    args: argparse.Namespace,
    data_root: Path,
    output_dir: Path,
    split_metadata: pd.DataFrame,
    scaler: LabelScaler,
) -> None:
    device = resolve_device(args.device)
    evaluation_dir = output_dir / "evaluation"
    pred_rel_dir = evaluation_dir / "predictions_relative"
    pred_abs_dir = evaluation_dir / "predictions_absolute_oracle_start"
    plots_dir = evaluation_dir / "plots"
    pred_rel_dir.mkdir(parents=True, exist_ok=True)
    pred_abs_dir.mkdir(parents=True, exist_ok=True)
    plots_dir.mkdir(parents=True, exist_ok=True)

    checkpoint = torch.load(output_dir / "checkpoints" / "best_model.pt", map_location=device)
    model = OriginalStyleCNNFCDecoder(dropout=args.dropout).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    rows = []
    test_predictions: dict[str, np.ndarray] = {}
    test_truths: dict[str, np.ndarray] = {}
    for split in ["train", "val", "test"]:
        frame = split_metadata[split_metadata["split"] == split].copy()
        loader = DataLoader(
            RelativeTrajectoryDataset(data_root, frame, scaler),
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=args.num_workers,
        )
        with torch.no_grad():
            for batch in loader:
                images = batch["image"].to(device)
                pred_rel = scaler.inverse_tensor(model(images)).cpu().numpy().astype(np.float32)
                true_rel = batch["relative"].numpy().astype(np.float32)
                true_abs = batch["absolute"].numpy().astype(np.float32)
                starts = batch["start"].numpy().astype(np.float32)
                for i, sample_id in enumerate(batch["sample_id"]):
                    sample_id = str(sample_id)
                    pred_abs = pred_rel[i] + starts[i]
                    np.save(pred_rel_dir / f"{sample_id}_pred_relative.npy", pred_rel[i])
                    np.save(pred_abs_dir / f"{sample_id}_pred_abs_oracle_start.npy", pred_abs)
                    rel_metrics = compute_basic_metrics(pred_rel[i], true_rel[i], "relative")
                    abs_metrics = compute_basic_metrics(pred_abs, true_abs[i], "oracle_start")
                    rows.append(
                        {
                            "sample_id": sample_id,
                            "vehicle_id": str(batch["vehicle_id"][i]),
                            "split": split,
                            **rel_metrics,
                            **abs_metrics,
                        }
                    )
                    if split == "test":
                        test_predictions[sample_id] = pred_abs
                        test_truths[sample_id] = true_abs[i]

    per_sample = pd.DataFrame(rows)
    per_sample.to_csv(evaluation_dir / "per_sample_metrics.csv", index=False)
    summary = summarize(per_sample)
    (evaluation_dir / "metrics_summary.json").write_text(json.dumps(summary, indent=2))
    plot_eval(per_sample, test_predictions, test_truths, plots_dir)
    print("Relative-target evaluation:")
    print(json.dumps({s: summary[s] for s in ["train", "val", "test"]}, indent=2)[:4000])


def compute_basic_metrics(pred: np.ndarray, true: np.ndarray, prefix: str) -> dict[str, float]:
    diff = pred - true
    point_errors = np.linalg.norm(diff, axis=1)
    return {
        f"{prefix}_ADE": float(point_errors.mean()),
        f"{prefix}_FDE": float(point_errors[-1]),
        f"{prefix}_RMSE": float(np.sqrt(np.mean(diff**2))),
        f"{prefix}_MAE": float(np.mean(np.abs(diff))),
    }


def summarize(table: pd.DataFrame) -> dict[str, dict[str, dict[str, float]]]:
    metrics = [
        "relative_ADE",
        "relative_FDE",
        "relative_RMSE",
        "relative_MAE",
        "oracle_start_ADE",
        "oracle_start_FDE",
        "oracle_start_RMSE",
        "oracle_start_MAE",
    ]
    out = {}
    for split, frame in table.groupby("split", sort=False):
        out[split] = {}
        for metric in metrics:
            values = frame[metric].to_numpy(float)
            out[split][metric] = {
                "mean": float(values.mean()),
                "median": float(np.median(values)),
                "std": float(values.std()),
                "min": float(values.min()),
                "max": float(values.max()),
            }
    return out


def plot_eval(
    per_sample: pd.DataFrame,
    test_predictions: dict[str, np.ndarray],
    test_truths: dict[str, np.ndarray],
    plots_dir: Path,
) -> None:
    test = per_sample[per_sample["split"] == "test"].copy()
    plot_hist(test["oracle_start_ADE"], plots_dir / "hist_test_oracle_start_ADE.png", "oracle-start ADE")
    plot_hist(test["relative_ADE"], plots_dir / "hist_test_relative_ADE.png", "relative ADE")
    sorted_test = test.sort_values("oracle_start_ADE").reset_index(drop=True)
    groups = {
        "best": sorted_test.head(10),
        "median": sorted_test.iloc[max(0, len(sorted_test) // 2 - 5) : len(sorted_test) // 2 + 5],
        "worst": sorted_test.tail(10).sort_values("oracle_start_ADE", ascending=False),
    }
    for name, frame in groups.items():
        group_dir = plots_dir / name
        group_dir.mkdir(parents=True, exist_ok=True)
        paths = []
        for row in frame.itertuples(index=False):
            path = group_dir / f"{row.sample_id}_trajectory.png"
            plot_prediction(
                row.sample_id,
                row.vehicle_id,
                test_truths[row.sample_id],
                test_predictions[row.sample_id],
                row.relative_ADE,
                row.oracle_start_ADE,
                row.oracle_start_FDE,
                path,
            )
            paths.append(path)
        make_contact_sheet(paths, plots_dir / f"{name}_10_grid.png")


def plot_hist(values: pd.Series, path: Path, title: str) -> None:
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.hist(values, bins=30)
    ax.set_title(title)
    ax.set_xlabel(title)
    ax.set_ylabel("count")
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def plot_prediction(sample_id, vehicle_id, true, pred, rel_ade, oracle_ade, oracle_fde, path) -> None:
    fig, ax = plt.subplots(figsize=(5, 5))
    ax.plot(true[:, 0], true[:, 1], label="ground truth")
    ax.plot(pred[:, 0], pred[:, 1], label="oracle-start prediction")
    ax.scatter(true[0, 0], true[0, 1], marker="o", s=24, label="start")
    ax.scatter(true[-1, 0], true[-1, 1], marker="s", s=24, label="gt end")
    ax.scatter(pred[-1, 0], pred[-1, 1], marker="^", s=24, label="pred end")
    all_xy = np.vstack([true, pred])
    padx = max(1.0, np.ptp(all_xy[:, 0]) * 0.05)
    pady = max(1.0, np.ptp(all_xy[:, 1]) * 0.05)
    ax.set_xlim(all_xy[:, 0].min() - padx, all_xy[:, 0].max() + padx)
    ax.set_ylim(all_xy[:, 1].min() - pady, all_xy[:, 1].max() + pady)
    ax.set_aspect("equal", adjustable="box")
    ax.set_title(
        f"{sample_id} {vehicle_id} relADE={rel_ade:.2f} "
        f"oracleADE={oracle_ade:.2f} oracleFDE={oracle_fde:.2f}"
    )
    ax.legend(fontsize=7)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def make_contact_sheet(paths: list[Path], output_path: Path) -> None:
    if not paths:
        return
    thumbs = [Image.open(path).convert("RGB").resize((320, 320)) for path in paths]
    sheet = Image.new("RGB", (5 * 320, int(np.ceil(len(thumbs) / 5)) * 320), "white")
    for i, thumb in enumerate(thumbs):
        sheet.paste(thumb, ((i % 5) * 320, (i // 5) * 320))
    sheet.save(output_path)


def compare_with_absolute(
    args: argparse.Namespace,
    data_root: Path,
    output_dir: Path,
    split_metadata: pd.DataFrame,
) -> None:
    comparison_dir = output_dir / "comparison_with_absolute"
    comparison_dir.mkdir(parents=True, exist_ok=True)
    absolute_dir = args.absolute_experiment_dir.resolve()
    abs_metrics = json.loads((absolute_dir / "evaluation" / "metrics_summary.json").read_text())
    rel_metrics = json.loads((output_dir / "evaluation" / "metrics_summary.json").read_text())
    abs_per = pd.read_csv(absolute_dir / "evaluation" / "per_sample_metrics.csv")
    rel_per = pd.read_csv(output_dir / "evaluation" / "per_sample_metrics.csv")

    summary = {
        "absolute_original_test_ADE": abs_metrics["test"]["ADE"]["mean"],
        "absolute_original_test_FDE": abs_metrics["test"]["FDE"]["mean"],
        "relative_oracle_start_test_ADE": rel_metrics["test"]["oracle_start_ADE"]["mean"],
        "relative_oracle_start_test_FDE": rel_metrics["test"]["oracle_start_FDE"]["mean"],
        "relative_shape_test_ADE": rel_metrics["test"]["relative_ADE"]["mean"],
        "relative_shape_test_FDE": rel_metrics["test"]["relative_FDE"]["mean"],
    }
    summary["oracle_start_ADE_improvement_pct"] = (
        (summary["absolute_original_test_ADE"] - summary["relative_oracle_start_test_ADE"])
        / summary["absolute_original_test_ADE"]
        * 100.0
    )
    summary["oracle_start_FDE_improvement_pct"] = (
        (summary["absolute_original_test_FDE"] - summary["relative_oracle_start_test_FDE"])
        / summary["absolute_original_test_FDE"]
        * 100.0
    )
    (comparison_dir / "comparison_summary.json").write_text(json.dumps(summary, indent=2))

    worst_abs = abs_per[abs_per["split"] == "test"].sort_values("ADE", ascending=False).head(10)
    table = worst_abs[["sample_id", "vehicle_id", "ADE", "FDE"]].merge(
        rel_per[["sample_id", "relative_ADE", "relative_FDE", "oracle_start_ADE", "oracle_start_FDE"]],
        on="sample_id",
        how="left",
    )
    table = table.rename(columns={"ADE": "absolute_ADE", "FDE": "absolute_FDE"})
    table.to_csv(comparison_dir / "comparison_table.csv", index=False)
    plot_distribution_compare(abs_per, rel_per, comparison_dir)
    write_comparison_report(summary, table, comparison_dir)


def plot_distribution_compare(abs_per: pd.DataFrame, rel_per: pd.DataFrame, out: Path) -> None:
    abs_test = abs_per[abs_per["split"] == "test"]
    rel_test = rel_per[rel_per["split"] == "test"]
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.hist(abs_test["ADE"], bins=30, alpha=0.55, label="absolute original ADE")
    ax.hist(rel_test["oracle_start_ADE"], bins=30, alpha=0.55, label="relative oracle-start ADE")
    ax.legend()
    ax.set_xlabel("ADE")
    ax.set_ylabel("count")
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(out / "test_ADE_distribution_comparison.png", dpi=150)
    plt.close(fig)


def write_comparison_report(summary: dict[str, float], table: pd.DataFrame, out: Path) -> None:
    if summary["oracle_start_ADE_improvement_pct"] > 20:
        shape_msg = "relative oracle-start 明显改善，说明图像表示对轨迹形状比绝对定位更可靠。"
    else:
        shape_msg = "relative oracle-start 改善有限，说明失败不只是绝对位置问题，形状/拓扑重建也受限。"
    worst_improved = float((table["absolute_ADE"] - table["oracle_start_ADE"]).mean())
    report = f"""
Relative-target 对比报告

1. absolute original test ADE/FDE: {summary['absolute_original_test_ADE']:.4f} / {summary['absolute_original_test_FDE']:.4f}
2. relative oracle-start test ADE/FDE: {summary['relative_oracle_start_test_ADE']:.4f} / {summary['relative_oracle_start_test_FDE']:.4f}
3. relative shape test ADE/FDE: {summary['relative_shape_test_ADE']:.4f} / {summary['relative_shape_test_FDE']:.4f}
4. oracle-start 相对 absolute ADE 改善: {summary['oracle_start_ADE_improvement_pct']:.2f}%
5. oracle-start 相对 absolute FDE 改善: {summary['oracle_start_FDE_improvement_pct']:.2f}%
6. absolute worst 10 在 relative oracle-start 下平均 ADE 改变量: {worst_improved:.4f}
7. 解释：{shape_msg}
8. 对 diffusion-generated images 的含义：后续如果生成图像解码失败，需要区分“形状不可恢复”和“绝对起点/定位不可恢复”；relative target 是一个必要对照，不应过度声称完整绝对轨迹可恢复。
"""
    (out / "comparison_report_zh.txt").write_text(report.strip() + "\n")
    print(report.strip())


def write_train_log(path: Path, rows: list[dict[str, object]]) -> None:
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def plot_training_curves(rows: list[dict[str, object]], plots_dir: Path) -> None:
    for train_key, val_key, name in [
        ("train_loss", "val_loss", "loss_curves.png"),
        ("train_relative_ADE", "val_relative_ADE", "relative_ade_curves.png"),
        ("train_relative_FDE", "val_relative_FDE", "relative_fde_curves.png"),
    ]:
        fig, ax = plt.subplots(figsize=(6, 4))
        xs = [r["epoch"] for r in rows]
        ax.plot(xs, [r[train_key] for r in rows], label="train")
        ax.plot(xs, [r[val_key] for r in rows], label="val")
        ax.legend()
        ax.grid(alpha=0.3)
        ax.set_xlabel("epoch")
        ax.set_ylabel(train_key.replace("train_", ""))
        fig.tight_layout()
        fig.savefig(plots_dir / name, dpi=150)
        plt.close(fig)


def fmt(value: float) -> str:
    return "NA" if np.isnan(value) else f"{value:.3f}"


if __name__ == "__main__":
    main()
