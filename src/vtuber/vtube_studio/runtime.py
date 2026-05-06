from __future__ import annotations

import asyncio
import contextlib
import time
from collections.abc import Callable
from pathlib import Path

from vtuber.vtube_studio.vtube_client import VTubeClient, VTubeError, VTubePlugin


ParameterFrame = dict[str, float]
ParameterPolicy = Callable[[float], ParameterFrame]


DEFAULT_NEUTRAL_PARAMETERS: ParameterFrame = {
    "FaceAngleX": 0.0,
    "FaceAngleY": 0.0,
    "FaceAngleZ": 0.0,
    "MouthOpen": 0.0,
    "MouthSmile": 0.0,
}


class VTubeRuntime:
    """VTube Studio 实时运行时封装。

    `VTubeClient` 是低层 API 客户端，只负责发单次请求；`VTubeRuntime` 则维护当前
    参数状态，并用后台任务按固定 FPS 持续把状态注入到 VTube Studio。上层调用者只
    需要更新当前参数，不需要自己写 30 FPS 发送循环。
    """

    def __init__(
        self,
        host: str = "127.0.0.1",
        port: int = 25565,
        plugin: VTubePlugin | None = None,
        token_path: Path | str = ".vts_token.json",
        fps: float = 30.0,
        neutral_parameters: ParameterFrame | None = None,
        face_found: bool = True,
    ) -> None:
        """创建运行时。

        Args:
            host: VTube Studio API 地址。
            port: VTube Studio Public API 端口。
            plugin: 插件身份信息。
            token_path: 本地认证 token 文件路径。
            fps: 后台参数注入频率。
            neutral_parameters: 中性姿态参数。未指定时使用默认头部居中、闭嘴。
            face_found: 注入参数时传给 VTube Studio 的 faceFound 标记。
        """
        if fps <= 0:
            raise ValueError("fps must be positive.")

        self.client = VTubeClient(host=host, port=port, plugin=plugin, token_path=token_path)
        self.fps = fps
        self.face_found = face_found
        self.neutral_parameters = dict(neutral_parameters or DEFAULT_NEUTRAL_PARAMETERS)
        self.current_parameters: ParameterFrame = dict(self.neutral_parameters)
        self._lock = asyncio.Lock()
        self._sender_task: asyncio.Task[None] | None = None
        self._running = False

    async def __aenter__(self) -> VTubeRuntime:
        await self.start()
        return self

    async def __aexit__(self, *_exc: object) -> None:
        await self.stop()

    async def start(self) -> None:
        """连接、认证并启动后台发送循环。"""
        if self._running:
            return
        await self.client.connect()
        await self.client.authenticate()
        self._running = True
        self._sender_task = asyncio.create_task(self._send_loop())

    async def stop(self) -> None:
        """停止后台发送循环，回到中性姿态并关闭连接。"""
        if not self._running:
            await self.client.close()
            return

        await self.reset_neutral()
        await self.flush_once()
        self._running = False

        if self._sender_task is not None:
            self._sender_task.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await self._sender_task
            self._sender_task = None

        await self.client.close()

    async def ensure_model_loaded(self) -> dict:
        """确认 VTube Studio 当前已经加载 Live2D 模型。

        Returns:
            `Live2DParameterListRequest` 的 data 字段。

        Raises:
            VTubeError: 当前没有加载模型。
        """
        model = await self.client.list_live2d_parameters()
        if not model.get("modelLoaded"):
            raise VTubeError("No Live2D model is loaded in VTube Studio.")
        return model

    @property
    def is_running(self) -> bool:
        """返回 runtime 后台发送循环是否处于运行状态。"""
        return self._running

    async def set_parameter(self, parameter_id: str, value: float) -> None:
        """更新单个 VTube Studio tracking 参数的当前值。

        该方法只更新本地状态；后台发送循环会在下一帧自动注入到 VTube Studio。
        """
        async with self._lock:
            self.current_parameters[parameter_id] = float(value)

    async def set_parameters(self, values: ParameterFrame) -> None:
        """批量更新当前参数值。

        适合每一帧模型输出或动作策略直接写入一组参数。
        """
        async with self._lock:
            for parameter_id, value in values.items():
                self.current_parameters[parameter_id] = float(value)

    async def reset_neutral(self) -> None:
        """把当前参数状态重置为中性姿态。"""
        async with self._lock:
            self.current_parameters = dict(self.neutral_parameters)

    async def flush_once(self) -> None:
        """立即把当前参数状态发送给 VTube Studio。

        后台循环会按 FPS 自动发送参数；这个接口用于截图、录制或测试场景，在
        `set_parameters()` 后希望尽快让 VTube Studio 收到当前帧状态。
        """
        async with self._lock:
            values = dict(self.current_parameters)
        if values:
            await self.client.inject_parameters(values, face_found=self.face_found)

    async def hold(self, seconds: float, parameters: ParameterFrame | None = None) -> None:
        """保持一组参数指定时长。

        Args:
            seconds: 保持时长，单位秒。
            parameters: 要保持的参数；为 `None` 时保持当前参数不变。
        """
        if parameters is not None:
            await self.set_parameters(parameters)
        await asyncio.sleep(max(0.0, seconds))

    async def play(self, policy: ParameterPolicy, seconds: float) -> None:
        """按当前 runtime FPS 播放一个动作策略。

        Args:
            policy: 输入 elapsed 秒数，返回本帧参数字典的函数。
            seconds: 播放时长，单位秒。
        """
        start = time.monotonic()
        interval = 1.0 / self.fps
        next_tick = start

        while True:
            now = time.monotonic()
            elapsed = now - start
            if elapsed >= seconds:
                return
            await self.set_parameters(policy(elapsed))
            next_tick += interval
            await asyncio.sleep(max(0.0, next_tick - time.monotonic()))

    async def _send_loop(self) -> None:
        interval = 1.0 / self.fps
        next_tick = time.monotonic()

        while self._running:
            await self.flush_once()
            next_tick += interval
            await asyncio.sleep(max(0.0, next_tick - time.monotonic()))
