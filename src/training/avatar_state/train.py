from __future__ import annotations

import argparse
import json
import logging
import random
import sys
import time
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.optim import AdamW
from torch.utils.data import DataLoader


REPO_ROOT = Path(__file__).resolve().parents[3]
SRC_DIR = REPO_ROOT / "src"
sys.path.insert(0, str(SRC_DIR))

from training.avatar_state import (  # noqa: E402
    AvatarStateDataset,
    DEFAULT_AVATAR_STATE_FIELDS,
)
from training.avatar_state.model_builder import (  # noqa: E402
    build_avatar_state_model,
    get_model_spec,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train AvatarState regression model.")
    parser.add_argument(
        "train_lmdb",
        nargs="+",
        help="训练用 LMDB 路径，可传多个。",
    )
    parser.add_argument(
        "--val-lmdb",
        nargs="*",
        default=None,
        help="验证用 LMDB 路径，可传多个。",
    )
    parser.add_argument(
        "--model-name",
        default="timm_convnextv2_tiny",
        help="模型配置项名称。当前默认使用 timm ConvNeXtV2-Tiny 主干。",
    )
    parser.add_argument(
        "--pretrained-path",
        default=None,
        help="本地预训练权重路径。支持 timm/backbone state_dict、本项目 checkpoint、或 safetensors。",
    )
    parser.add_argument(
        "--resume",
        default=None,
        help="从已有训练 checkpoint 继续训练，例如 last.pt。",
    )
    parser.add_argument("--image-source", default="vtube", choices=["vtube", "obs"])
    parser.add_argument(
        "--bbox-source",
        default=None,
        help="bbox 来源，默认与 image_source 相同，可选 vtube 或 obs。",
    )
    parser.add_argument("--image-size", type=int, default=256, help="输入 crop 尺寸。")
    parser.add_argument("--padding-ratio", type=float, default=0.0, help="crop padding 比例。")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--freeze-epochs", type=int, default=3, help="先只训头的 epoch 数。")
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--hidden-dim", type=int, default=512)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default=None, help="训练设备，例如 cuda:0 或 cpu。")
    parser.add_argument("--project", default="outputs/avatar_state_runs")
    parser.add_argument("--name", default="hrnetv2_w18_avatar_state")
    parser.add_argument("--log-interval", type=int, default=20)
    return parser.parse_args()


def configure_logging(run_dir: Path) -> logging.Logger:
    """同时输出到终端和日志文件。"""
    run_dir.mkdir(parents=True, exist_ok=True)
    logger = logging.getLogger("avatar_state_train")
    logger.setLevel(logging.INFO)
    logger.handlers.clear()
    formatter = logging.Formatter("%(asctime)s | %(levelname)s | %(message)s")

    stream_handler = logging.StreamHandler(sys.stdout)
    stream_handler.setFormatter(formatter)
    logger.addHandler(stream_handler)

    file_handler = logging.FileHandler(run_dir / "train.log", encoding="utf-8")
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)
    return logger


def set_seed(seed: int) -> None:
    """固定随机种子。"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def create_loader(
    lmdb_paths: list[str],
    *,
    image_source: str,
    bbox_source: str | None,
    image_size: int,
    padding_ratio: float,
    batch_size: int,
    workers: int,
    shuffle: bool,
) -> DataLoader:
    """构造 AvatarState DataLoader。"""
    dataset = AvatarStateDataset(
        lmdb_paths,
        image_source=image_source,
        bbox_source=bbox_source,
        image_size=image_size,
        padding_ratio=padding_ratio,
        state_fields=DEFAULT_AVATAR_STATE_FIELDS,
    )
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=workers,
        pin_memory=True,
        drop_last=False,
    )


def evaluate(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
) -> tuple[float, list[float]]:
    """在验证集上评估平均 loss 和逐字段 MAE。"""
    model.eval()
    total_loss = 0.0
    total_count = 0
    per_field_abs = torch.zeros(len(DEFAULT_AVATAR_STATE_FIELDS), dtype=torch.float64)
    with torch.no_grad():
        for batch in loader:
            images = batch["image"].to(device, non_blocking=True)
            targets = batch["target"].to(device, non_blocking=True)
            outputs = model(images)
            loss = criterion(outputs, targets)
            batch_size = images.shape[0]
            total_loss += float(loss.item()) * batch_size
            total_count += batch_size
            per_field_abs += (
                (outputs - targets).abs().sum(dim=0).detach().cpu().to(torch.float64)
            )
    if total_count == 0:
        raise ValueError("Validation loader is empty.")
    mean_loss = total_loss / total_count
    field_mae = (per_field_abs / total_count).tolist()
    return mean_loss, field_mae


def save_checkpoint(
    path: Path,
    *,
    model: nn.Module,
    optimizer: AdamW,
    epoch: int,
    best_val_loss: float | None,
    args: argparse.Namespace,
    train_fields: tuple[str, ...],
    resolved_pretrained_source: str | None,
    local_pretrained_path: str | None,
) -> None:
    """保存训练 checkpoint。"""
    payload = {
        "epoch": epoch,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "best_val_loss": best_val_loss,
        "model_name": args.model_name,
        "pretrained_source": resolved_pretrained_source,
        "local_pretrained_path": local_pretrained_path,
        "image_source": args.image_source,
        "bbox_source": args.bbox_source,
        "image_size": args.image_size,
        "padding_ratio": args.padding_ratio,
        "hidden_dim": args.hidden_dim,
        "dropout": args.dropout,
        "state_fields": list(train_fields),
    }
    torch.save(payload, path)


def main() -> None:
    args = parse_args()
    if args.bbox_source not in {None, "vtube", "obs"}:
        raise ValueError("bbox_source must be None, 'vtube' or 'obs'.")
    resume_checkpoint = None
    if args.resume is not None:
        resume_path = Path(args.resume).resolve()
        if not resume_path.exists():
            raise FileNotFoundError(f"Resume checkpoint does not exist: {resume_path}")
        map_location = args.device if args.device else "cpu"
        resume_checkpoint = torch.load(resume_path, map_location=map_location)
    run_dir = (REPO_ROOT / args.project / args.name).resolve()
    logger = configure_logging(run_dir)
    set_seed(args.seed)

    device = torch.device(
        args.device if args.device else ("cuda" if torch.cuda.is_available() else "cpu")
    )
    if resume_checkpoint is not None:
        args.model_name = resume_checkpoint["model_name"]
        logger.info("resume_checkpoint=%s", resume_path)
    model_spec = get_model_spec(args.model_name)
    logger.info("run_dir=%s", run_dir)
    logger.info("device=%s", device)
    logger.info("model_name=%s", model_spec.name)
    logger.info("model_desc=%s", model_spec.description)

    train_loader = create_loader(
        args.train_lmdb,
        image_source=args.image_source,
        bbox_source=args.bbox_source,
        image_size=args.image_size,
        padding_ratio=args.padding_ratio,
        batch_size=args.batch_size,
        workers=args.workers,
        shuffle=True,
    )
    val_loader = None
    if args.val_lmdb:
        val_loader = create_loader(
            args.val_lmdb,
            image_source=args.image_source,
            bbox_source=args.bbox_source,
            image_size=args.image_size,
            padding_ratio=args.padding_ratio,
            batch_size=args.batch_size,
            workers=args.workers,
            shuffle=False,
        )
    logger.info("train_samples=%d", len(train_loader.dataset))
    if val_loader is not None:
        logger.info("val_samples=%d", len(val_loader.dataset))

    model, resolved_assets = build_avatar_state_model(
        args.model_name,
        output_dim=len(DEFAULT_AVATAR_STATE_FIELDS),
        hidden_dim=args.hidden_dim,
        dropout=args.dropout,
        pretrained_path=args.pretrained_path if resume_checkpoint is None else None,
        use_pretrained_backbone=resume_checkpoint is None,
        device=device,
    )
    logger.info("pretrained_source=%s", resolved_assets.pretrained_source)
    logger.info("local_pretrained_path=%s", resolved_assets.local_pretrained_path)
    model.train()
    effective_local_pretrained_path = resolved_assets.local_pretrained_path
    if resume_checkpoint is not None:
        effective_local_pretrained_path = resume_checkpoint.get("local_pretrained_path")

    # 先跑一次前向，确保延迟构造的回归头进入参数列表。
    warmup_batch = next(iter(train_loader))
    warmup_images = warmup_batch["image"].to(device, non_blocking=True)
    with torch.no_grad():
        _ = model(warmup_images[:1])

    if resume_checkpoint is None and args.freeze_epochs > 0 and hasattr(model, "freeze_backbone"):
        model.freeze_backbone()
        logger.info("backbone frozen for first %d epochs", args.freeze_epochs)

    optimizer = AdamW(
        (parameter for parameter in model.parameters() if parameter.requires_grad),
        lr=args.lr,
        weight_decay=args.weight_decay,
    )
    criterion = nn.SmoothL1Loss()

    metrics_path = run_dir / "metrics.jsonl"
    config_path = run_dir / "config.json"
    config_payload = {
        "model_name": args.model_name,
        "pretrained_source": resolved_assets.pretrained_source,
        "local_pretrained_path": effective_local_pretrained_path,
        "train_lmdb": [str(Path(path).resolve()) for path in args.train_lmdb],
        "val_lmdb": [str(Path(path).resolve()) for path in (args.val_lmdb or [])],
        "image_source": args.image_source,
        "bbox_source": args.bbox_source,
        "image_size": args.image_size,
        "padding_ratio": args.padding_ratio,
        "batch_size": args.batch_size,
        "epochs": args.epochs,
        "freeze_epochs": args.freeze_epochs,
        "lr": args.lr,
        "weight_decay": args.weight_decay,
        "workers": args.workers,
        "hidden_dim": args.hidden_dim,
        "dropout": args.dropout,
        "seed": args.seed,
        "device": str(device),
        "state_fields": list(DEFAULT_AVATAR_STATE_FIELDS),
    }
    config_path.write_text(json.dumps(config_payload, ensure_ascii=False, indent=2), encoding="utf-8")

    start_epoch = 1
    best_val_loss: float | None = None
    if resume_checkpoint is not None:
        model.load_state_dict(resume_checkpoint["model_state_dict"])
        optimizer.load_state_dict(resume_checkpoint["optimizer_state_dict"])
        best_val_loss = resume_checkpoint.get("best_val_loss")
        start_epoch = int(resume_checkpoint["epoch"]) + 1
        logger.info("resumed_from_epoch=%d", start_epoch - 1)
        logger.info("best_val_loss=%s", best_val_loss)

    for epoch in range(start_epoch, args.epochs + 1):
        if epoch == args.freeze_epochs + 1 and args.freeze_epochs > 0 and hasattr(model, "unfreeze_backbone"):
            model.unfreeze_backbone()
            optimizer = AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
            logger.info("backbone unfrozen from epoch %d", epoch)

        model.train()
        epoch_start = time.time()
        running_loss = 0.0
        seen_samples = 0

        for step, batch in enumerate(train_loader, start=1):
            images = batch["image"].to(device, non_blocking=True)
            targets = batch["target"].to(device, non_blocking=True)

            optimizer.zero_grad(set_to_none=True)
            outputs = model(images)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()

            batch_size = images.shape[0]
            running_loss += float(loss.item()) * batch_size
            seen_samples += batch_size

            if step % args.log_interval == 0 or step == len(train_loader):
                logger.info(
                    "epoch=%d step=%d/%d train_loss=%.6f",
                    epoch,
                    step,
                    len(train_loader),
                    running_loss / max(seen_samples, 1),
                )

        train_loss = running_loss / max(seen_samples, 1)
        epoch_seconds = time.time() - epoch_start
        metric_row: dict[str, object] = {
            "epoch": epoch,
            "train_loss": train_loss,
            "epoch_seconds": epoch_seconds,
        }

        if val_loader is not None:
            val_loss, field_mae = evaluate(model, val_loader, criterion, device)
            metric_row["val_loss"] = val_loss
            metric_row["field_mae"] = {
                field_name: field_error
                for field_name, field_error in zip(DEFAULT_AVATAR_STATE_FIELDS, field_mae)
            }
            logger.info("epoch=%d train_loss=%.6f val_loss=%.6f", epoch, train_loss, val_loss)
            logger.info("epoch=%d val_field_mae=%s", epoch, metric_row["field_mae"])
            if best_val_loss is None or val_loss < best_val_loss:
                best_val_loss = val_loss
                save_checkpoint(
                    run_dir / "best.pt",
                    model=model,
                    optimizer=optimizer,
                    epoch=epoch,
                    best_val_loss=best_val_loss,
                    args=args,
                    train_fields=DEFAULT_AVATAR_STATE_FIELDS,
                    resolved_pretrained_source=resolved_assets.pretrained_source,
                    local_pretrained_path=effective_local_pretrained_path,
                )
                logger.info("saved new best checkpoint: %s", run_dir / "best.pt")
        else:
            logger.info("epoch=%d train_loss=%.6f", epoch, train_loss)

        with metrics_path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(metric_row, ensure_ascii=False) + "\n")

        save_checkpoint(
            run_dir / "last.pt",
            model=model,
            optimizer=optimizer,
            epoch=epoch,
            best_val_loss=best_val_loss,
            args=args,
            train_fields=DEFAULT_AVATAR_STATE_FIELDS,
            resolved_pretrained_source=resolved_assets.pretrained_source,
            local_pretrained_path=effective_local_pretrained_path,
        )


if __name__ == "__main__":
    main()
