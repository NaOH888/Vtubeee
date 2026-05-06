from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

import torch
from torch import nn
from torch.utils.data import DataLoader


REPO_ROOT = Path(__file__).resolve().parents[3]
SRC_DIR = REPO_ROOT / "src"
sys.path.insert(0, str(SRC_DIR))

from training.avatar_state import AvatarStateDataset  # noqa: E402
from training.avatar_state.model_builder import build_avatar_state_model  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Validate AvatarState regression model.")
    parser.add_argument("checkpoint", help="训练脚本导出的 checkpoint 路径。")
    parser.add_argument("lmdb", nargs="+", help="验证/推理用 LMDB 路径，可传多个。")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--device", default=None, help="验证设备，例如 cuda:0 或 cpu。")
    parser.add_argument("--output", default=None, help="预测结果 jsonl 输出路径。")
    parser.add_argument("--log-interval", type=int, default=20)
    return parser.parse_args()


def configure_logging() -> logging.Logger:
    """配置终端日志。"""
    logger = logging.getLogger("avatar_state_val")
    logger.setLevel(logging.INFO)
    logger.handlers.clear()
    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(logging.Formatter("%(asctime)s | %(levelname)s | %(message)s"))
    logger.addHandler(handler)
    return logger


def build_loader(
    lmdb_paths: list[str],
    *,
    image_source: str,
    bbox_source: str | None,
    image_size: int,
    padding_ratio: float,
    batch_size: int,
    workers: int,
    state_fields: tuple[str, ...],
) -> DataLoader:
    """构造验证 DataLoader。"""
    dataset = AvatarStateDataset(
        lmdb_paths,
        image_source=image_source,
        bbox_source=bbox_source,
        image_size=image_size,
        padding_ratio=padding_ratio,
        state_fields=state_fields,
    )
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=workers,
        pin_memory=True,
        drop_last=False,
    )


def main() -> None:
    args = parse_args()
    logger = configure_logging()
    checkpoint_path = Path(args.checkpoint).resolve()
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Checkpoint does not exist: {checkpoint_path}")

    device = torch.device(
        args.device if args.device else ("cuda" if torch.cuda.is_available() else "cpu")
    )
    checkpoint = torch.load(checkpoint_path, map_location=device)
    state_fields = tuple(checkpoint["state_fields"])
    logger.info("checkpoint=%s", checkpoint_path)
    logger.info("device=%s", device)
    logger.info("state_fields=%s", state_fields)

    loader = build_loader(
        args.lmdb,
        image_source=checkpoint["image_source"],
        bbox_source=checkpoint["bbox_source"],
        image_size=int(checkpoint["image_size"]),
        padding_ratio=float(checkpoint["padding_ratio"]),
        batch_size=args.batch_size,
        workers=args.workers,
        state_fields=state_fields,
    )
    logger.info("val_samples=%d", len(loader.dataset))

    model, resolved_assets = build_avatar_state_model(
        checkpoint["model_name"],
        output_dim=len(state_fields),
        hidden_dim=int(checkpoint["hidden_dim"]),
        dropout=float(checkpoint["dropout"]),
        pretrained_path=checkpoint.get("local_pretrained_path"),
        use_pretrained_backbone=False,
        device=device,
    )
    logger.info("pretrained_source=%s", resolved_assets.pretrained_source)
    logger.info("local_pretrained_path=%s", resolved_assets.local_pretrained_path)
    # 先构造 head，再加载权重。
    warmup_batch = next(iter(loader))
    warmup_images = warmup_batch["image"].to(device, non_blocking=True)
    with torch.no_grad():
        _ = model(warmup_images[:1])
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    criterion = nn.SmoothL1Loss(reduction="mean")
    total_loss = 0.0
    total_count = 0
    per_field_abs = torch.zeros(len(state_fields), dtype=torch.float64)
    rows: list[dict[str, object]] = []

    with torch.no_grad():
        for step, batch in enumerate(loader, start=1):
            images = batch["image"].to(device, non_blocking=True)
            targets = batch["target"].to(device, non_blocking=True)
            outputs = model(images)
            loss = criterion(outputs, targets)
            batch_size = images.shape[0]
            total_loss += float(loss.item()) * batch_size
            total_count += batch_size
            abs_error = (outputs - targets).abs().detach().cpu()
            per_field_abs += abs_error.sum(dim=0).to(torch.float64)

            predictions = outputs.detach().cpu().tolist()
            targets_cpu = targets.detach().cpu().tolist()
            for item_index, predicted_vector, target_vector in zip(
                batch["index"],
                predictions,
                targets_cpu,
            ):
                rows.append(
                    {
                        "index": int(item_index),
                        "prediction": {
                            field_name: float(value)
                            for field_name, value in zip(state_fields, predicted_vector)
                        },
                        "target": {
                            field_name: float(value)
                            for field_name, value in zip(state_fields, target_vector)
                        },
                    }
                )

            if step % args.log_interval == 0 or step == len(loader):
                logger.info(
                    "step=%d/%d running_loss=%.6f",
                    step,
                    len(loader),
                    total_loss / max(total_count, 1),
                )

    mean_loss = total_loss / max(total_count, 1)
    field_mae = (per_field_abs / max(total_count, 1)).tolist()
    field_mae_map = {
        field_name: field_error
        for field_name, field_error in zip(state_fields, field_mae)
    }
    logger.info("val_loss=%.6f", mean_loss)
    logger.info("field_mae=%s", field_mae_map)

    output_path = None
    if args.output is not None:
        output_path = Path(args.output).resolve()
    else:
        output_path = checkpoint_path.with_name(f"{checkpoint_path.stem}_predictions.jsonl")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    logger.info("predictions_saved=%s", output_path)


if __name__ == "__main__":
    main()
