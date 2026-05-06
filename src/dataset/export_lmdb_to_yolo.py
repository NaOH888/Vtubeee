from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path

import lmdb


REPO_ROOT = Path(__file__).resolve().parents[2]
SRC_DIR = REPO_ROOT / "src"
sys.path.insert(0, str(SRC_DIR))


def read_bytes(txn: lmdb.Transaction, key: bytes) -> bytes:
    """读取 LMDB 字节值。"""
    raw = txn.get(key)
    if raw is None:
        raise KeyError(f"Missing key: {key.decode('utf-8', errors='ignore')}")
    return raw


def read_json(txn: lmdb.Transaction, key: bytes) -> dict:
    """读取 LMDB 中的 JSON 值。"""
    return json.loads(read_bytes(txn, key).decode("utf-8"))


def resolve_path(path_str: str) -> Path:
    """把命令行路径解析到仓库内绝对路径。"""
    path = Path(path_str)
    if path.is_absolute():
        return path
    return REPO_ROOT / path


def ensure_output_dirs(output_dir: Path) -> None:
    """创建 YOLO 标准目录结构。"""
    for relative in (
        "images/train",
        "images/val",
        "labels/train",
        "labels/val",
    ):
        (output_dir / relative).mkdir(parents=True, exist_ok=True)


def write_yaml(output_dir: Path) -> None:
    """写出 YOLO 数据集配置。"""
    yaml_text = (
        f"path: {output_dir.as_posix()}\n"
        "train: images/train\n"
        "val: images/val\n"
        "names:\n"
        "  0: vtuber\n"
    )
    (output_dir / "data.yaml").write_text(yaml_text, encoding="utf-8")


def write_label_file(label_path: Path, yolo_bbox: list[float]) -> None:
    """写 YOLO 单目标标签文件。"""
    x_center, y_center, width, height = yolo_bbox
    label_text = (
        f"0 {x_center:.8f} {y_center:.8f} {width:.8f} {height:.8f}\n"
    )
    label_path.write_text(label_text, encoding="utf-8")


def export_dataset(args: argparse.Namespace) -> None:
    """把一个或多个 LMDB 导出为 YOLO 标准目录。"""
    output_dir = resolve_path(args.output)
    ensure_output_dirs(output_dir)

    rng = random.Random(args.seed)
    export_index = 0
    exported_count = 0
    skipped_no_state = 0
    manifest_path = output_dir / "samples.jsonl"
    manifest_handle = manifest_path.open("w", encoding="utf-8")

    try:
        for lmdb_input in args.inputs:
            lmdb_path = resolve_path(lmdb_input)
            if not lmdb_path.exists():
                raise FileNotFoundError(f"LMDB path does not exist: {lmdb_path}")

            print(f"Exporting from: {lmdb_path}", flush=True)
            env = lmdb.open(
                str(lmdb_path),
                subdir=True,
                readonly=True,
                lock=False,
                readahead=False,
                meminit=False,
            )
            try:
                with env.begin(write=False) as txn:
                    length = int(read_bytes(txn, b"__length__").decode("utf-8"))
                    config = read_json(txn, b"__config__")
                    print(
                        f"  sample_count={length}, profile_id={config.get('profile_id')}",
                        flush=True,
                    )

                    for sample_index in range(length):
                        sample_prefix = f"sample/{sample_index:06d}".encode("utf-8")
                        meta = read_json(txn, sample_prefix + b"/meta")
                        if args.require_state and "state" not in meta:
                            skipped_no_state += 1
                            continue

                        obs_bytes = read_bytes(txn, sample_prefix + b"/obs")
                        split = "val" if rng.random() < args.val_ratio else "train"
                        stem = f"{export_index:08d}"
                        image_relative = Path("images") / split / f"{stem}.jpg"
                        label_relative = Path("labels") / split / f"{stem}.txt"
                        image_path = output_dir / image_relative
                        label_path = output_dir / label_relative
                        image_path.write_bytes(obs_bytes)
                        write_label_file(label_path, list(meta["obs_bbox_yolo"]))

                        manifest_item = {
                            "export_index": export_index,
                            "split": split,
                            "image": image_relative.as_posix(),
                            "label": label_relative.as_posix(),
                            "source_lmdb": str(lmdb_path),
                            "source_index": sample_index,
                            "avatar_id": meta.get("avatar_id"),
                            "profile_id": meta.get("profile_id"),
                            "has_state": "state" in meta,
                        }
                        manifest_handle.write(
                            json.dumps(manifest_item, ensure_ascii=False) + "\n"
                        )
                        export_index += 1
                        exported_count += 1
            finally:
                env.close()
    finally:
        manifest_handle.close()

    write_yaml(output_dir)
    print(f"Output dir: {output_dir}", flush=True)
    print(f"Exported samples: {exported_count}", flush=True)
    print(f"Skipped samples without state: {skipped_no_state}", flush=True)
    print(f"Manifest: {manifest_path}", flush=True)
    print(f"YOLO config: {output_dir / 'data.yaml'}", flush=True)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Export one or more LMDB bbox datasets to YOLO directory format."
    )
    parser.add_argument(
        "inputs",
        nargs="+",
        help="One or more LMDB dataset directories.",
    )
    parser.add_argument(
        "--output",
        default="outputs/yolo_dataset",
        help="Output YOLO dataset directory.",
    )
    parser.add_argument(
        "--val-ratio",
        type=float,
        default=0.1,
        help="Validation split ratio.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed used for train/val split.",
    )
    parser.add_argument(
        "--require-state",
        action="store_true",
        help="Skip old samples that do not contain the `state` field.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.val_ratio < 0.0 or args.val_ratio > 1.0:
        raise ValueError(f"val-ratio must be in [0, 1], got {args.val_ratio}")
    export_dataset(args)


if __name__ == "__main__":
    try:
        main()
    except (FileNotFoundError, KeyError, ValueError, json.JSONDecodeError) as exc:
        print(f"Export error: {exc}", flush=True)
        sys.exit(1)
