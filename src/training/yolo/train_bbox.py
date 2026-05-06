from __future__ import annotations

import argparse
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train a YOLO bbox detector on the exported VTuber dataset."
    )
    parser.add_argument(
        "--data",
        required=True,
        help="YOLO data.yaml 路径，一般来自 export_lmdb_to_yolo.py 的输出目录。",
    )
    parser.add_argument(
        "--model",
        default="yolov8n.pt",
        help="Ultralytics 检测模型或预训练权重路径。",
    )
    parser.add_argument("--epochs", type=int, default=100, help="训练轮数。")
    parser.add_argument("--imgsz", type=int, default=640, help="训练图像尺寸。")
    parser.add_argument("--batch", type=int, default=16, help="batch size。")
    parser.add_argument(
        "--device",
        default=None,
        help="训练设备，例如 0、0,1、cpu。留空时由 Ultralytics 自行决定。",
    )
    parser.add_argument("--workers", type=int, default=8, help="DataLoader worker 数量。")
    parser.add_argument(
        "--project",
        default="outputs/yolo_runs",
        help="Ultralytics 训练输出目录。",
    )
    parser.add_argument(
        "--name",
        default="two_model_yolo",
        help="本次训练的 run 名称。",
    )
    parser.add_argument(
        "--cache",
        action="store_true",
        help="是否启用 Ultralytics 数据缓存。",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="是否恢复同名训练任务。",
    )
    parser.add_argument(
        "--patience",
        type=int,
        default=50,
        help="早停 patience。",
    )
    parser.add_argument(
        "--single-cls",
        action="store_true",
        help="按单类别检测训练。当前 VTuber bbox 数据集通常建议开启。",
    )
    parser.add_argument(
        "--save-period",
        type=int,
        default=-1,
        help="每隔多少个 epoch 额外保存一次 checkpoint；-1 表示关闭。",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    data_path = Path(args.data).resolve()
    project_path = Path(args.project).resolve()
    if not data_path.exists():
        raise FileNotFoundError(f"data.yaml does not exist: {data_path}")

    try:
        from ultralytics import YOLO
    except ImportError as exc:
        raise RuntimeError(
            "ultralytics is not installed. Install it in the current venv first."
        ) from exc

    model = YOLO(args.model)
    train_kwargs: dict[str, object] = {
        "data": str(data_path),
        "epochs": args.epochs,
        "imgsz": args.imgsz,
        "batch": args.batch,
        "workers": args.workers,
        "project": str(project_path),
        "name": args.name,
        "cache": args.cache,
        "resume": args.resume,
        "patience": args.patience,
        "single_cls": args.single_cls,
        "save_period": args.save_period,
    }
    if args.device:
        train_kwargs["device"] = args.device

    print(f"Training data: {data_path}", flush=True)
    print(f"Model: {args.model}", flush=True)
    print(f"Project: {project_path}", flush=True)
    print(f"Run name: {args.name}", flush=True)
    model.train(**train_kwargs)


if __name__ == "__main__":
    main()
