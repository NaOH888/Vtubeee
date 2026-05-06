from __future__ import annotations

import asyncio
import json
import time
from collections.abc import AsyncIterator, Callable, Iterable
from dataclasses import asdict, dataclass
from pathlib import Path
from threading import Event
from typing import TYPE_CHECKING, Any

import cv2
import numpy as np
from pygrabber.dshow_graph import FilterGraph

from vtuber.avatar.avatar_state import AvatarState

if TYPE_CHECKING:
    from vtuber.avatar.avatar import Avatar


ImageFrame = np.ndarray
ProgressCallback = Callable[["CapturedFrame"], None]


class CaptureError(RuntimeError):
    """avatar 捕获层对外暴露的统一错误。"""


@dataclass(frozen=True)
class CapturedFrame:
    """单帧捕获结果。

    `image` 使用 OpenCV 约定的 BGR 三通道 `numpy.ndarray`。如果后续要交给
    PIL、matplotlib 或深度学习预处理管线，需要按调用方需求转换为 RGB。
    """

    index: int
    time_seconds: float
    state: AvatarState
    image: ImageFrame
    captured_at: float

    def to_metadata(self) -> dict[str, Any]:
        """返回可写入 JSON 的帧级元数据，不包含图像像素本身。"""
        return {
            "frame_index": self.index,
            "time_seconds": self.time_seconds,
            "captured_at": self.captured_at,
            "state": asdict(self.state),
        }


@dataclass(frozen=True)
class CaptureSequenceResult:
    """序列落盘捕获的结果摘要。"""

    output_path: Path
    metadata_path: Path | None
    fps: float
    frame_count: int
    width: int
    height: int


class DirectShowFrameGrabber:
    """复用 DirectShow graph 的虚拟摄像头单帧抓取器。

    VTube Studio 虚拟摄像头在 OpenCV 后端中可能打不开；这里直接搭建
    DirectShow graph，通过 sample grabber 拿到帧。sequence 模式会持续复用同一个
    graph，避免每一帧都重新打开摄像头。
    """

    def __init__(
        self,
        device_index: int,
        timeout_seconds: float,
        *,
        use_null_render: bool = True,
    ) -> None:
        """创建抓帧器。

        Args:
            device_index: DirectShow 摄像头索引。
            timeout_seconds: 单帧抓取超时时间。
            use_null_render: 是否使用 NullRender。启用后不会弹出预览窗口。
        """
        self.device_index = device_index
        self.timeout_seconds = timeout_seconds
        self.use_null_render = use_null_render
        self._frame_ready = Event()
        self._frames: list[ImageFrame] = []
        self._graph: FilterGraph | None = None

    def start(self) -> None:
        """启动 DirectShow graph。"""
        if self._graph is not None:
            return

        graph = FilterGraph()
        graph.add_video_input_device(self.device_index)
        graph.add_sample_grabber(self._on_frame)
        if self.use_null_render:
            graph.add_null_render()
        else:
            graph.add_default_render()
        graph.prepare_preview_graph()
        graph.run()
        self._graph = graph

    def close(self) -> None:
        """停止 DirectShow graph。"""
        if self._graph is not None:
            self._graph.stop()
            self._graph = None

    def capture(self, timeout_seconds: float | None = None) -> ImageFrame:
        """从当前 DirectShow graph 抓取一帧图像。"""
        if self._graph is None:
            raise CaptureError("DirectShow graph is not running.")

        timeout = self.timeout_seconds if timeout_seconds is None else timeout_seconds
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            self._frame_ready.clear()
            self._graph.grab_frame()
            if self._frame_ready.wait(0.12):
                return self._frames[-1]
            time.sleep(0.03)

        render_name = "NullRender" if self.use_null_render else "DefaultRender"
        raise CaptureError(f"Timed out while grabbing a frame via {render_name}.")

    def _on_frame(self, frame: ImageFrame) -> None:
        self._frames.append(frame)
        self._frame_ready.set()


class AvatarCapture:
    """avatar 渲染捕获入口。

    这一层只暴露 `Avatar` 和 `AvatarState` 语义。当前实现通过 VTube Studio
    虚拟摄像头抓取画面；未来替换为 Cubism 离屏渲染时，上层数据集代码仍可继续
    使用同一组接口。
    """

    def __init__(
        self,
        camera: str = "VTubeStudioCam",
        *,
        capture_timeout: float = 3.0,
        settle_seconds: float = 0.25,
        frame_settle_seconds: float = 0.04,
    ) -> None:
        """创建捕获器。

        Args:
            camera: DirectShow 摄像头名称或索引。
            capture_timeout: 单次抓帧超时时间。
            settle_seconds: 单张截图时，设置姿态后等待渲染刷新的时间。
            frame_settle_seconds: sequence 每帧设置姿态后等待渲染刷新的时间。
        """
        if capture_timeout <= 0:
            raise ValueError("capture_timeout must be positive.")
        if settle_seconds < 0:
            raise ValueError("settle_seconds cannot be negative.")
        if frame_settle_seconds < 0:
            raise ValueError("frame_settle_seconds cannot be negative.")

        self.camera = camera
        self.capture_timeout = capture_timeout
        self.settle_seconds = settle_seconds
        self.frame_settle_seconds = frame_settle_seconds

    async def capture_image(self, avatar: Avatar, state: AvatarState) -> CapturedFrame:
        """设置一个姿态并捕获单张图像。

        返回值保留在内存中，适合后续直接进入数据集构造或测试断言；如果需要保存到
        文件，使用 `capture_image_to_file()` 这个便捷接口。
        """
        await avatar.set_state(state)
        await avatar.flush_once()
        await asyncio.sleep(self.settle_seconds)

        device_index = resolve_camera_source(self.camera)
        frame = capture_directshow_frame(device_index, self.capture_timeout)
        return CapturedFrame(
            index=0,
            time_seconds=0.0,
            state=state,
            image=directshow_frame_to_bgr(frame),
            captured_at=time.time(),
        )

    async def capture_image_to_file(
        self,
        avatar: Avatar,
        state: AvatarState,
        output_path: Path | str,
    ) -> CapturedFrame:
        """设置一个姿态、捕获单张图像并保存到文件。"""
        frame = await self.capture_image(avatar, state)
        path = _prepare_output_path(output_path)
        ok = cv2.imwrite(str(path), frame.image)
        if not ok:
            raise CaptureError(f"Failed to write image: {path}")
        return frame

    async def capture_sequence(
        self,
        avatar: Avatar,
        states: Iterable[AvatarState],
        fps: float,
    ) -> AsyncIterator[CapturedFrame]:
        """按固定 FPS 播放逐帧姿态，并异步产出捕获帧。

        这是 sequence 的核心接口：调用方可以一边捕获一边写视频、写数据集或做在线
        检查，不需要把整段视频一次性放进内存。
        """
        if fps <= 0:
            raise ValueError("fps must be positive.")

        device_index = resolve_camera_source(self.camera)
        grabber = open_directshow_grabber(device_index, self.capture_timeout)
        interval = 1.0 / fps
        next_tick = time.monotonic()

        try:
            for frame_index, state in enumerate(states):
                await asyncio.sleep(max(0.0, next_tick - time.monotonic()))
                next_tick += interval

                await avatar.set_state(state)
                await avatar.flush_once()
                if self.frame_settle_seconds > 0:
                    await asyncio.sleep(self.frame_settle_seconds)

                yield CapturedFrame(
                    index=frame_index,
                    time_seconds=frame_index / fps,
                    state=state,
                    image=directshow_frame_to_bgr(grabber.capture()),
                    captured_at=time.time(),
                )
        finally:
            grabber.close()

    async def capture_sequence_to_file(
        self,
        avatar: Avatar,
        states: Iterable[AvatarState],
        output_path: Path | str,
        fps: float,
        *,
        metadata_path: Path | str | None = None,
        progress: ProgressCallback | None = None,
    ) -> CaptureSequenceResult:
        """按逐帧姿态列表录制视频，并保存对应 metadata。

        metadata 只记录每一帧的时间戳和 `AvatarState`，不包含图像像素。图像像素保存在
        输出视频中。
        """
        output = _prepare_output_path(output_path)
        metadata = _resolve_metadata_path(output, metadata_path)
        writer: cv2.VideoWriter | None = None
        frame_metadata: list[dict[str, Any]] = []
        width = 0
        height = 0

        try:
            async for frame in self.capture_sequence(avatar, states, fps):
                if writer is None:
                    height, width = frame.image.shape[:2]
                    writer = _create_video_writer(output, fps, frame.image)
                writer.write(frame.image)
                frame_metadata.append(frame.to_metadata())
                if progress is not None:
                    progress(frame)
        finally:
            if writer is not None:
                writer.release()

        if writer is None:
            raise CaptureError("Cannot write sequence video because no frames were captured.")

        _write_sequence_metadata(
            metadata,
            output=output,
            fps=fps,
            frame_count=len(frame_metadata),
            frames=frame_metadata,
        )
        return CaptureSequenceResult(
            output_path=output,
            metadata_path=metadata,
            fps=fps,
            frame_count=len(frame_metadata),
            width=width,
            height=height,
        )


def list_directshow_cameras() -> list[str]:
    """枚举 Windows DirectShow 视频输入设备名称。"""
    return list(FilterGraph().get_input_devices())


def resolve_camera_name(name: str) -> int:
    """把 DirectShow 摄像头名称解析为设备索引。"""
    devices = list_directshow_cameras()
    for index, device_name in enumerate(devices):
        if device_name == name:
            return index

    lowered = name.lower()
    partial_matches = [
        (index, device_name)
        for index, device_name in enumerate(devices)
        if lowered in device_name.lower()
    ]
    if len(partial_matches) == 1:
        return partial_matches[0][0]

    device_lines = "\n  - ".join(f"{index}: {device_name}" for index, device_name in enumerate(devices))
    raise CaptureError(
        f"Cannot resolve camera name {name!r}. Available DirectShow cameras:\n  - {device_lines}"
    )


def resolve_camera_source(camera: str) -> int:
    """把摄像头参数解析为 DirectShow 设备索引。"""
    camera = camera.strip()
    if camera.isdigit():
        return int(camera)
    return resolve_camera_name(camera)


def open_directshow_grabber(
    device_index: int,
    timeout_seconds: float,
) -> DirectShowFrameGrabber:
    """打开 DirectShow 抓帧器，优先使用不显示窗口的 NullRender。"""
    errors = []
    for use_null_render in (True, False):
        render_name = "NullRender" if use_null_render else "DefaultRender"
        grabber = DirectShowFrameGrabber(
            device_index=device_index,
            timeout_seconds=timeout_seconds,
            use_null_render=use_null_render,
        )
        try:
            grabber.start()
            return grabber
        except Exception as exc:
            grabber.close()
            errors.append(f"{render_name}: {exc}")

    details = "\n  - ".join(errors)
    raise CaptureError(f"Cannot open DirectShow camera graph:\n  - {details}")


def capture_directshow_frame(device_index: int, timeout_seconds: float) -> ImageFrame:
    """从 DirectShow 摄像头抓取单帧图像。"""
    grabber = open_directshow_grabber(device_index, timeout_seconds)
    try:
        return grabber.capture()
    finally:
        grabber.close()


def directshow_frame_to_bgr(frame: ImageFrame) -> ImageFrame:
    """返回 OpenCV 写文件需要的 BGR 帧。

    DirectShow 的 RGB24 媒体类型在 Windows DIB 内存里通常是 BGR 字节顺序；
    pygrabber 返回的是这块像素数据本身，而 OpenCV 的 `imwrite`/`VideoWriter`
    也按 BGR 解释三通道图像。因此这里不能再做 RGB->BGR 转换，否则红蓝通道会
    被交换，画面会明显偏青。
    """
    return frame


def _create_video_writer(output_path: Path, fps: float, frame: ImageFrame) -> cv2.VideoWriter:
    """根据首帧尺寸创建 OpenCV 视频写入器。"""
    height, width = frame.shape[:2]
    suffix = output_path.suffix.lower()
    fourcc_name = "MJPG" if suffix == ".avi" else "mp4v"
    writer = cv2.VideoWriter(
        str(output_path),
        cv2.VideoWriter_fourcc(*fourcc_name),
        fps,
        (width, height),
    )
    if not writer.isOpened():
        raise CaptureError(f"Failed to open video writer: {output_path}")
    return writer


def _prepare_output_path(output_path: Path | str) -> Path:
    """准备输出路径，并确保父目录存在。"""
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    return path


def _resolve_metadata_path(output_path: Path, metadata_path: Path | str | None) -> Path:
    """解析 sequence metadata 输出路径。"""
    if metadata_path is not None:
        return _prepare_output_path(metadata_path)
    return output_path.with_suffix(".json")


def _write_sequence_metadata(
    path: Path,
    *,
    output: Path,
    fps: float,
    frame_count: int,
    frames: list[dict[str, Any]],
) -> None:
    """写入 sequence 帧级 metadata。"""
    with path.open("w", encoding="utf-8") as file:
        json.dump(
            {
                "video": str(output),
                "fps": fps,
                "frame_count": frame_count,
                "frames": frames,
            },
            file,
            ensure_ascii=False,
            indent=2,
        )
