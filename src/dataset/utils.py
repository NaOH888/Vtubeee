from __future__ import annotations

import time
from collections.abc import Iterator
from dataclasses import dataclass
from typing import Any, Protocol

import cv2
import numpy as np

from vtuber.avatar.capture import CaptureError, open_directshow_grabber


BBox = tuple[int, int, int, int]


class FrameGrabber(Protocol):
    """双路采样脚本使用的统一抓帧接口。"""

    def capture(self) -> np.ndarray:
        """读取一帧 BGR 图像。"""

    def close(self) -> None:
        """关闭抓帧器。"""


class OpenCVCameraGrabber:
    """OpenCV 摄像头抓帧器。

    OBS Virtual Camera 在当前环境里可以通过 OpenCV 打开，但 pygrabber 的
    DirectShow graph 不能打开；因此数据采样脚本允许对某一路摄像头回退到 OpenCV。
    """

    def __init__(
        self,
        device_index: int,
        timeout_seconds: float,
        *,
        width: int | None = None,
        height: int | None = None,
    ) -> None:
        self.device_index = device_index
        self.timeout_seconds = timeout_seconds
        self._capture = cv2.VideoCapture(device_index, cv2.CAP_DSHOW)
        if not self._capture.isOpened():
            self._capture.release()
            raise CaptureError(f"Cannot open camera index {device_index} via OpenCV.")
        if width is not None and width > 0:
            self._capture.set(cv2.CAP_PROP_FRAME_WIDTH, float(width))
        if height is not None and height > 0:
            self._capture.set(cv2.CAP_PROP_FRAME_HEIGHT, float(height))
        actual_width = int(round(self._capture.get(cv2.CAP_PROP_FRAME_WIDTH)))
        actual_height = int(round(self._capture.get(cv2.CAP_PROP_FRAME_HEIGHT)))
        print(
            f"OpenCV camera index {device_index} actual resolution: "
            f"{actual_width}x{actual_height}",
            flush=True,
        )

    def capture(self) -> np.ndarray:
        """读取一帧 BGR 图像。"""
        deadline = time.monotonic() + self.timeout_seconds
        while time.monotonic() < deadline:
            ok, frame = self._capture.read()
            if ok and frame is not None:
                return frame
            time.sleep(0.03)
        raise CaptureError(f"Timed out while reading camera index {self.device_index}.")

    def close(self) -> None:
        """释放 OpenCV 摄像头。"""
        self._capture.release()


def open_camera_grabber(
    device_index: int,
    timeout_seconds: float,
    backend: str,
    *,
    width: int | None = None,
    height: int | None = None,
) -> FrameGrabber:
    """按指定后端打开摄像头抓帧器。

    `directshow` 使用项目里稳定的 pygrabber 路径；`opencv` 使用 OpenCV DSHOW；
    `auto` 先试 directshow，失败后回退到 opencv。
    """
    if backend == "directshow":
        print(f"Opening camera index {device_index} via DirectShow.", flush=True)
        return open_directshow_grabber(device_index, timeout_seconds)
    if backend == "opencv":
        print(f"Opening camera index {device_index} via OpenCV.", flush=True)
        return OpenCVCameraGrabber(
            device_index,
            timeout_seconds,
            width=width,
            height=height,
        )
    if backend == "auto":
        try:
            print(f"Opening camera index {device_index} via DirectShow.", flush=True)
            return open_directshow_grabber(device_index, timeout_seconds)
        except CaptureError as exc:
            print(f"DirectShow failed for camera index {device_index}: {exc}", flush=True)
            print(f"Opening camera index {device_index} via OpenCV.", flush=True)
            return OpenCVCameraGrabber(
                device_index,
                timeout_seconds,
                width=width,
                height=height,
            )
    raise ValueError(f"Unsupported camera backend: {backend}")


def estimate_background_color(frame: np.ndarray, border_ratio: float) -> np.ndarray:
    """从画面边缘估计背景色。"""
    height, width = frame.shape[:2]
    border = max(1, round(min(width, height) * border_ratio))
    pixels = np.concatenate(
        [
            frame[:border, :, :].reshape(-1, 3),
            frame[-border:, :, :].reshape(-1, 3),
            frame[:, :border, :].reshape(-1, 3),
            frame[:, -border:, :].reshape(-1, 3),
        ],
        axis=0,
    )
    return np.median(pixels, axis=0).astype(np.uint8)


def foreground_mask_from_background(
    frame: np.ndarray,
    background_color: np.ndarray,
    threshold: int,
) -> np.ndarray:
    """根据背景色差异生成前景 mask。"""
    diff = np.abs(frame.astype(np.int16) - background_color.reshape(1, 1, 3).astype(np.int16))
    mask = (diff.max(axis=2) > threshold).astype(np.uint8) * 255
    kernel = np.ones((5, 5), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    return mask


def bbox_from_mask(
    mask: np.ndarray,
    padding_ratio: float,
    min_area_ratio: float,
) -> BBox:
    """从前景 mask 中取最大连通区域并返回带 padding 的外接框。"""
    contours, _hierarchy = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        raise RuntimeError("Cannot find foreground contour from VTube frame.")

    height, width = mask.shape[:2]
    min_area = width * height * min_area_ratio
    contour = max(contours, key=cv2.contourArea)
    area = cv2.contourArea(contour)
    if area < min_area:
        raise RuntimeError(f"Foreground contour is too small: area={area:.1f}, min={min_area:.1f}")

    x, y, box_width, box_height = cv2.boundingRect(contour)
    pad = round(max(box_width, box_height) * padding_ratio)
    x1 = max(0, x - pad)
    y1 = max(0, y - pad)
    x2 = min(width - 1, x + box_width + pad)
    y2 = min(height - 1, y + box_height + pad)
    return x1, y1, x2, y2


def map_bbox_between_frames(
    bbox: BBox,
    source_shape: tuple[int, ...],
    target_shape: tuple[int, ...],
) -> BBox:
    """按分辨率比例把一个画面上的 bbox 映射到另一个画面。"""
    source_height, source_width = source_shape[:2]
    target_height, target_width = target_shape[:2]
    scale_x = target_width / source_width
    scale_y = target_height / source_height
    x1, y1, x2, y2 = bbox
    return (
        round(x1 * scale_x),
        round(y1 * scale_y),
        round(x2 * scale_x),
        round(y2 * scale_y),
    )


def yolo_bbox_from_xyxy(
    bbox: BBox,
    image_shape: tuple[int, ...],
) -> tuple[float, float, float, float]:
    """把 `xyxy` bbox 转换为 YOLO 归一化格式。"""
    height, width = image_shape[:2]
    x1, y1, x2, y2 = bbox
    box_width = max(1, x2 - x1)
    box_height = max(1, y2 - y1)
    x_center = x1 + box_width * 0.5
    y_center = y1 + box_height * 0.5
    return (
        x_center / width,
        y_center / height,
        box_width / width,
        box_height / height,
    )


def bbox_touches_border(bbox: BBox, image_shape: tuple[int, ...]) -> tuple[str, ...]:
    """返回 bbox 是否贴到画面边界。"""
    height, width = image_shape[:2]
    x1, y1, x2, y2 = bbox
    touches: list[str] = []
    if x1 <= 0:
        touches.append("left")
    if y1 <= 0:
        touches.append("top")
    if x2 >= width - 1:
        touches.append("right")
    if y2 >= height - 1:
        touches.append("bottom")
    return tuple(touches)


@dataclass(frozen=True)
class BBoxTransferSample:
    """一次双路采样的结构化结果。"""

    index: int
    vtube_frame: np.ndarray
    obs_frame: np.ndarray
    mask: np.ndarray
    vtube_bbox_xyxy: BBox
    obs_bbox_xyxy: BBox
    background_bgr: tuple[int, int, int]
    bbox_lag_frames: int
    vtube_bbox_source_step: int
    obs_capture_step: int

    def to_metadata(self) -> dict[str, Any]:
        """转换为适合保存到 jsonl 或 npz metadata 的字典。"""
        return {
            "index": self.index,
            "vtube_bbox_xyxy": list(self.vtube_bbox_xyxy),
            "obs_bbox_xyxy": list(self.obs_bbox_xyxy),
            "vtube_shape": list(self.vtube_frame.shape),
            "obs_shape": list(self.obs_frame.shape),
            "background_bgr": list(self.background_bgr),
            "bbox_lag_frames": self.bbox_lag_frames,
            "vtube_bbox_source_step": self.vtube_bbox_source_step,
            "obs_capture_step": self.obs_capture_step,
            "obs_bbox_yolo": list(yolo_bbox_from_xyxy(self.obs_bbox_xyxy, self.obs_frame.shape)),
            "touches_border": list(bbox_touches_border(self.obs_bbox_xyxy, self.obs_frame.shape)),
        }


def collect_bbox_transfer_samples(
    *,
    vtube_grabber: FrameGrabber,
    obs_grabber: FrameGrabber,
    count: int,
    interval: float,
    warmup: int,
    threshold: int,
    padding_ratio: float,
    border_ratio: float,
    min_area_ratio: float,
    bbox_lag_frames: int,
) -> list[BBoxTransferSample]:
    """采集一批双路样本，并把 VTube bbox 按帧延迟映射到 OBS。"""
    return list(
        iter_bbox_transfer_samples(
            vtube_grabber=vtube_grabber,
            obs_grabber=obs_grabber,
            count=count,
            interval=interval,
            warmup=warmup,
            threshold=threshold,
            padding_ratio=padding_ratio,
            border_ratio=border_ratio,
            min_area_ratio=min_area_ratio,
            bbox_lag_frames=bbox_lag_frames,
        )
    )


def iter_bbox_transfer_samples(
    *,
    vtube_grabber: FrameGrabber,
    obs_grabber: FrameGrabber,
    count: int,
    interval: float,
    warmup: int,
    threshold: int,
    padding_ratio: float,
    border_ratio: float,
    min_area_ratio: float,
    bbox_lag_frames: int,
) -> Iterator[BBoxTransferSample]:
    """按生成器方式采集双路样本，避免把所有帧一次性放进内存。"""
    if count <= 0:
        raise ValueError("count must be positive.")
    if interval < 0:
        raise ValueError("interval cannot be negative.")
    if warmup < 0:
        raise ValueError("warmup cannot be negative.")
    if bbox_lag_frames < 0:
        raise ValueError("bbox_lag_frames must be non-negative.")

    for _ in range(warmup):
        vtube_grabber.capture()
        obs_grabber.capture()
        time.sleep(0.03)

    history: list[dict[str, Any]] = []
    total_steps = count + bbox_lag_frames
    sample_index = 0

    for capture_step in range(total_steps):
        vtube_frame = vtube_grabber.capture()
        obs_frame = obs_grabber.capture()
        background_color = estimate_background_color(vtube_frame, border_ratio)
        mask = foreground_mask_from_background(vtube_frame, background_color, threshold)
        vtube_bbox = bbox_from_mask(mask, padding_ratio, min_area_ratio)

        history.append(
            {
                "capture_step": capture_step,
                "vtube_frame": vtube_frame.copy(),
                "mask": mask.copy(),
                "vtube_bbox": vtube_bbox,
                "vtube_shape": vtube_frame.shape,
                "background_bgr": tuple(int(value) for value in background_color.tolist()),
            }
        )
        if len(history) > bbox_lag_frames + 1:
            history.pop(0)

        if len(history) <= bbox_lag_frames:
            print(
                f"Buffered VTube bbox {capture_step + 1}/{bbox_lag_frames + 1} "
                f"before saving OBS samples.",
                flush=True,
            )
            if capture_step + 1 < total_steps:
                time.sleep(interval)
            continue

        source_item = history[-(bbox_lag_frames + 1)]
        obs_bbox = map_bbox_between_frames(
            source_item["vtube_bbox"],
            source_item["vtube_shape"],
            obs_frame.shape,
        )
        sample = BBoxTransferSample(
            index=sample_index,
            vtube_frame=source_item["vtube_frame"],
            obs_frame=obs_frame.copy(),
            mask=source_item["mask"],
            vtube_bbox_xyxy=source_item["vtube_bbox"],
            obs_bbox_xyxy=obs_bbox,
            background_bgr=source_item["background_bgr"],
            bbox_lag_frames=bbox_lag_frames,
            vtube_bbox_source_step=source_item["capture_step"],
            obs_capture_step=capture_step,
        )
        print(
            f"Collected sample {sample.index + 1}/{count}: "
            f"source_step={sample.vtube_bbox_source_step} -> obs_step={sample.obs_capture_step}, "
            f"obs_bbox={sample.obs_bbox_xyxy}",
            flush=True,
        )
        sample_index += 1
        yield sample

        if capture_step + 1 < total_steps:
            time.sleep(interval)
