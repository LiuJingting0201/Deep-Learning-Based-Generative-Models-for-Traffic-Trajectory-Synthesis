"""Create best-10 qualitative reconstruction comparisons for Week3 Stage 2 models."""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import torch  # noqa: E402
from PIL import Image  # noqa: E402
from torch.utils.data import DataLoader  # noqa: E402


PROJECT_ROOT = Path(__file__).resolve().parents[2]
WEEK03_SCRIPTS_DIR = PROJECT_ROOT / "week03" / "scripts"
if str(WEEK03_SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(WEEK03_SCRIPTS_DIR))

from evaluate_sequence_latent_alignment import load_checkpoint_config, load_image_encoder, load_teacher  # noqa: E402
from sequence_latent_utils import (  # noqa: E402
    PairedImageDeltaDataset,
    integrated_trajectory,
    load_split_metadata,
    resolve_device,
    set_seed,
)


@dataclass(frozen=True)
class ModelSpec:
    key: str
    model_name: str
    experiment_dir: Path
    checkpoint_path: Path
    output_dir: Path


def main() -> None:
    args = parse_args()
    set_seed(args.seed)
    device = resolve_device(args.device)
    output_root = args.output_dir.resolve()
    comparison_dir = output_root / "comparison_summary"
    comparison_dir.mkdir(parents=True, exist_ok=True)

    specs = [
        ModelSpec(
            key="k7l3",
            model_name="stage2_rowwise_temporalconv_k7_l3_latent64_lr3e4_ep500",
            experiment_dir=args.k7l3_dir.resolve(),
            checkpoint_path=(args.k7l3_dir / "best.pt").resolve(),
            output_dir=output_root / "k7l3",
        ),
        ModelSpec(
            key="transformer",
            model_name="stage2_rowwise_transformer_l2_h4_ff256_latent64_lr1e4_ep500",
            experiment_dir=args.transformer_dir.resolve(),
            checkpoint_path=(args.transformer_dir / "best.pt").resolve(),
            output_dir=output_root / "transformer",
        ),
    ]

    teacher = load_teacher(args.trajectory_ae_checkpoint.resolve(), args.latent_dim, device)
    split_metadata = load_split_metadata(args.data_root.resolve(), seed=args.seed)
    eval_frame = split_metadata[split_metadata["split"] == args.split].copy()
    if eval_frame.empty:
        raise ValueError(f"No samples found for split={args.split}")

    summaries: list[dict[str, Any]] = []
    best10_by_model: dict[str, pd.DataFrame] = {}
    for spec in specs:
        model_summary, best10 = process_model(args, spec, eval_frame, teacher, device)
        summaries.append(model_summary)
        best10_by_model[spec.key] = best10

    summary_frame = pd.DataFrame(summaries)
    summary_frame.to_csv(comparison_dir / "comparison_summary.csv", index=False)
    write_markdown_summary(comparison_dir / "comparison_summary.md", summaries, best10_by_model)
    write_common_sample_comparison(
        comparison_dir / "common_sample_comparison.csv",
        best10_by_model["k7l3"],
        best10_by_model["transformer"],
    )
    print(f"Done. Qualitative comparison outputs saved to {output_root}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, default=Path("/home/jliu/data_no_speed_delta_displacement_paired"))
    parser.add_argument(
        "--trajectory-ae-checkpoint",
        type=Path,
        default=Path("/home/jliu/Thesis/week03/results/sequence_latent/stage1_traj_ae_latent64/best.pt"),
    )
    parser.add_argument(
        "--k7l3-dir",
        type=Path,
        default=Path(
            "/home/jliu/Thesis/week03/results/sequence_latent/"
            "stage2_rowwise_temporalconv_k7_l3_latent64_lr3e4_ep500"
        ),
    )
    parser.add_argument(
        "--transformer-dir",
        type=Path,
        default=Path(
            "/home/jliu/Thesis/week03/results/sequence_latent/"
            "stage2_rowwise_transformer_l2_h4_ff256_latent64_lr1e4_ep500"
        ),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("/home/jliu/Thesis/week03/results/sequence_latent/qualitative_comparison_best10"),
    )
    parser.add_argument("--split", choices=["train", "val", "test"], default="test")
    parser.add_argument("--latent-dim", type=int, default=64)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--top-k", type=int, default=10)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="auto")
    return parser.parse_args()


def process_model(
    args: argparse.Namespace,
    spec: ModelSpec,
    eval_frame: pd.DataFrame,
    teacher: torch.nn.Module,
    device: torch.device,
) -> tuple[dict[str, Any], pd.DataFrame]:
    if not spec.checkpoint_path.exists():
        raise FileNotFoundError(spec.checkpoint_path)
    spec.output_dir.mkdir(parents=True, exist_ok=True)
    plots_dir = spec.output_dir / "best10_plots"
    plots_dir.mkdir(parents=True, exist_ok=True)
    arrays_dir = spec.output_dir / "arrays"
    arrays_dir.mkdir(parents=True, exist_ok=True)

    image_encoder = load_image_encoder(spec.checkpoint_path, args.latent_dim, device)
    checkpoint_config = load_checkpoint_config(spec.checkpoint_path)
    image_channel_mode = str(checkpoint_config.get("image_channel_mode", "rgb"))
    loader = DataLoader(
        PairedImageDeltaDataset(args.data_root.resolve(), eval_frame, image_channel_mode=image_channel_mode),
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=device.type == "cuda",
    )
    rows = evaluate_samples(loader, image_encoder, teacher, device, arrays_dir)
    metrics = pd.DataFrame(rows).sort_values("integrated_ADE", ascending=True).reset_index(drop=True)
    metrics.to_csv(spec.output_dir / "sample_metrics.csv", index=False)

    best10 = metrics.head(args.top_k).copy()
    best10.insert(0, "rank", np.arange(1, len(best10) + 1))
    best10.to_csv(spec.output_dir / "best10_sample_metrics.csv", index=False)
    write_best10_ids(spec.output_dir / "best10_ids.txt", best10)

    plot_paths: list[Path] = []
    for row in best10.itertuples(index=False):
        plot_path = plots_dir / f"rank{int(row.rank):02d}_sample_{safe_filename(str(row.sample_id))}.png"
        plot_reconstruction(
            plot_path,
            spec.model_name,
            str(row.sample_id),
            np.load(row.gt_integrated_path),
            np.load(row.pred_integrated_path),
            float(row.integrated_ADE),
            float(row.integrated_FDE),
        )
        plot_paths.append(plot_path)
    make_contact_sheet(plot_paths, spec.output_dir / "best10_contact_sheet.png")

    best_metrics = read_json_if_exists(spec.experiment_dir / "best_metrics.json")
    summary = {
        "model_name": spec.model_name,
        "checkpoint_path": str(spec.checkpoint_path),
        "experiment_dir": str(spec.experiment_dir),
        "best_metric_name": best_metrics.get("best_metric", ""),
        "best_metric_score": best_metrics.get("best_score", ""),
        "best_epoch": best_metrics.get("best_epoch", ""),
        "mean_integrated_ADE": float(metrics["integrated_ADE"].mean()),
        "mean_integrated_FDE": float(metrics["integrated_FDE"].mean()),
        "mean_delta_ADE": float(metrics["delta_ADE"].mean()),
        "mean_delta_FDE": float(metrics["delta_FDE"].mean()),
        "top10_sample_ids": ", ".join(best10["sample_id"].astype(str).tolist()),
        "top10_integrated_ADE": ", ".join(f"{value:.6f}" for value in best10["integrated_ADE"]),
    }
    return summary, best10


def evaluate_samples(
    loader: DataLoader,
    image_encoder: torch.nn.Module,
    teacher: torch.nn.Module,
    device: torch.device,
    arrays_dir: Path,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with torch.no_grad():
        for batch in loader:
            images = batch["image"].to(device, non_blocking=True)
            delta_gt = batch["delta"].to(device, non_blocking=True)
            z_img = image_encoder(images)
            delta_hat = teacher.decoder(z_img)
            gt_integrated = integrated_trajectory(delta_gt).cpu().numpy().astype(np.float32)
            pred_integrated = integrated_trajectory(delta_hat).cpu().numpy().astype(np.float32)
            delta_gt_np = delta_gt.cpu().numpy().astype(np.float32)
            delta_hat_np = delta_hat.cpu().numpy().astype(np.float32)
            for index, sample_id in enumerate(batch["sample_id"]):
                sample_id = str(sample_id)
                gt_path = arrays_dir / f"{safe_filename(sample_id)}_gt_integrated.npy"
                pred_path = arrays_dir / f"{safe_filename(sample_id)}_pred_integrated.npy"
                np.save(gt_path, gt_integrated[index])
                np.save(pred_path, pred_integrated[index])
                rows.append(
                    {
                        "sample_id": sample_id,
                        "vehicle_id": str(batch["vehicle_id"][index]),
                        **metric_values(delta_hat_np[index], delta_gt_np[index], "delta"),
                        **metric_values(pred_integrated[index], gt_integrated[index], "integrated"),
                        "gt_integrated_path": str(gt_path),
                        "pred_integrated_path": str(pred_path),
                    }
                )
    return rows


def metric_values(pred: np.ndarray, gt: np.ndarray, prefix: str) -> dict[str, float]:
    errors = np.linalg.norm(pred - gt, axis=1)
    return {
        f"{prefix}_ADE": float(errors.mean()),
        f"{prefix}_FDE": float(errors[-1]),
        f"{prefix}_mse": float(np.mean((pred - gt) ** 2)),
    }


def plot_reconstruction(
    path: Path,
    model_name: str,
    sample_id: str,
    gt: np.ndarray,
    pred: np.ndarray,
    integrated_ade: float,
    integrated_fde: float,
) -> None:
    fig, axis = plt.subplots(figsize=(6.2, 5.2))
    axis.plot(gt[:, 0], gt[:, 1], label="Ground Truth", linewidth=2.0)
    axis.plot(pred[:, 0], pred[:, 1], label="Prediction", linewidth=2.0)
    axis.scatter(gt[0, 0], gt[0, 1], s=28, marker="o", label="Start")
    axis.scatter(gt[-1, 0], gt[-1, 1], s=30, marker="s", label="GT End")
    axis.scatter(pred[-1, 0], pred[-1, 1], s=30, marker="x", label="Pred End")
    axis.set_aspect("equal", adjustable="box")
    axis.set_title(
        f"{model_name}\n"
        f"sample_id={sample_id} | integrated_ADE={integrated_ade:.3f} | integrated_FDE={integrated_fde:.3f}",
        fontsize=9,
    )
    axis.grid(alpha=0.25)
    axis.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)


def make_contact_sheet(image_paths: list[Path], output_path: Path, columns: int = 5) -> None:
    if not image_paths:
        return
    images = [Image.open(path).convert("RGB") for path in image_paths]
    thumb_width = 360
    thumb_height = 300
    thumbs = []
    for image in images:
        image.thumbnail((thumb_width, thumb_height))
        canvas = Image.new("RGB", (thumb_width, thumb_height), "white")
        left = (thumb_width - image.width) // 2
        top = (thumb_height - image.height) // 2
        canvas.paste(image, (left, top))
        thumbs.append(canvas)
    rows = int(np.ceil(len(thumbs) / columns))
    sheet = Image.new("RGB", (columns * thumb_width, rows * thumb_height), "white")
    for index, thumb in enumerate(thumbs):
        x = (index % columns) * thumb_width
        y = (index // columns) * thumb_height
        sheet.paste(thumb, (x, y))
    sheet.save(output_path)
    for image in images:
        image.close()


def write_best10_ids(path: Path, best10: pd.DataFrame) -> None:
    lines = [
        f"rank{int(row.rank):02d}\t{row.sample_id}\tintegrated_ADE={float(row.integrated_ADE):.6f}"
        for row in best10.itertuples(index=False)
    ]
    path.write_text("\n".join(lines) + "\n")


def write_markdown_summary(
    path: Path,
    summaries: list[dict[str, Any]],
    best10_by_model: dict[str, pd.DataFrame],
) -> None:
    lines = [
        "# Week3 Stage 2 Best-10 Qualitative Comparison",
        "",
        "Both models are evaluated independently on the test split. The best 10 examples are selected by lowest sample-level integrated ADE for each model.",
        "",
        "## Aggregate Summary",
        "",
        "| Model | Best metric | Best score | Best epoch | Mean integrated ADE | Mean integrated FDE |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for summary in summaries:
        lines.append(
            "| {model_name} | {best_metric_name} | {best_metric_score} | {best_epoch} | "
            "{mean_integrated_ADE:.6f} | {mean_integrated_FDE:.6f} |".format(**summary)
        )
    lines.extend(["", "## Top-10 Samples", ""])
    for summary in summaries:
        model_name = summary["model_name"]
        key = "k7l3" if "k7_l3" in model_name else "transformer"
        lines.extend([f"### {model_name}", "", "| Rank | Sample ID | integrated ADE | integrated FDE |", "|---:|---|---:|---:|"])
        for row in best10_by_model[key].itertuples(index=False):
            lines.append(f"| {int(row.rank)} | {row.sample_id} | {float(row.integrated_ADE):.6f} | {float(row.integrated_FDE):.6f} |")
        lines.append("")
    lines.extend(
        [
            "## Qualitative Note",
            "",
            "The k7/l3 temporal-convolution model uses local residual token communication with a wider temporal receptive field than k5/l2. "
            "The Transformer model uses global token interaction across all timestep tokens. Compare the contact sheets by looking for trajectory-shape fidelity, endpoint drift, and whether local turns are preserved.",
            "",
        ]
    )
    path.write_text("\n".join(lines))


def write_common_sample_comparison(path: Path, k7l3: pd.DataFrame, transformer: pd.DataFrame) -> None:
    k_cols = ["sample_id", "rank", "integrated_ADE", "integrated_FDE", "delta_ADE", "delta_FDE"]
    t_cols = ["sample_id", "rank", "integrated_ADE", "integrated_FDE", "delta_ADE", "delta_FDE"]
    merged = k7l3[k_cols].merge(
        transformer[t_cols],
        on="sample_id",
        how="inner",
        suffixes=("_k7l3", "_transformer"),
    )
    merged.to_csv(path, index=False)


def read_json_if_exists(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    return json.loads(path.read_text())


def safe_filename(value: str) -> str:
    return "".join(char if char.isalnum() or char in {"-", "_"} else "_" for char in value)


if __name__ == "__main__":
    main()
