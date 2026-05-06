from __future__ import annotations

import asyncio
import time
from collections.abc import Callable
from pathlib import Path
from typing import Any

from vtuber.avatar.avatar_profile import AvatarProfile
from vtuber.avatar.avatar_state import AvatarState
from vtuber.avatar.avatar_transform import AvatarTransform
from vtuber.vtube_studio.mapper import VTubeParameterMapper
from vtuber.vtube_studio.runtime import VTubeRuntime


VTubeAvatarPolicy = Callable[[float], AvatarState]


class VTubeAvatarDriver:
    """VTube Studio 后端的 AvatarState 驱动器。

    这一层把通用 `AvatarState` 通过 `VTubeParameterMapper` 转为 VTube Studio
    tracking 参数，再交给 `VTubeRuntime` 持续发送。上层通常不直接使用本类，
    而是通过 `vtuber.avatar.Avatar` facade 调用。
    """

    def __init__(
        self,
        host: str = "127.0.0.1",
        port: int = 25565,
        token_path: Path | str = ".vts_token.json",
        fps: float = 30.0,
        mapper: VTubeParameterMapper | None = None,
        profile: AvatarProfile | Path | str | dict[str, Any] | None = None,
        config: dict[str, Any] | None = None,
    ) -> None:
        """创建 VTube Studio avatar driver。

        Args:
            host: VTube Studio API 地址。
            port: VTube Studio Public API 端口。
            token_path: VTube Studio 认证 token 文件位置。
            fps: 参数刷新频率。
            mapper: 可选自定义 mapper。未指定时使用默认 VTube Studio 参数映射。
            profile: 可选 avatar profile。可以是标准 profile JSON、VTube Studio
                `.vtube.json` 路径，或已经构造好的 `AvatarProfile`。
            config: 后端配置预留字段。
        """
        self.config = config or {}
        profile = profile or self.config.get("profile") or self.config.get("profile_path")
        if mapper is None and profile is None:
            raise ValueError("VTubeAvatarDriver requires profile when mapper is not provided.")
        self.mapper = mapper or VTubeParameterMapper(profile=profile)
        self.runtime = VTubeRuntime(
            host=host,
            port=port,
            token_path=token_path,
            fps=fps,
            neutral_parameters=self.mapper.neutral_parameters(),
        )

    async def start(self) -> None:
        """启动 VTube Studio runtime。"""
        await self.runtime.start()

    async def stop(self) -> None:
        """停止 VTube Studio runtime。"""
        await self.runtime.stop()

    async def set_state(self, state: AvatarState) -> None:
        """把通用 avatar 状态映射并写入当前 runtime 状态。"""
        await self.runtime.set_parameters(self.mapper.to_parameters(state))

    async def hold_state(self, state: AvatarState, sec: float) -> None:
        """保持指定 avatar 状态一段时间。"""
        await self.set_state(state)
        await asyncio.sleep(max(0.0, sec))

    async def set_transform(
        self,
        transform: AvatarTransform,
        *,
        time_in_seconds: float = 0.0,
        values_are_relative_to_model: bool = False,
    ) -> None:
        """设置当前模型的整体位置、旋转和缩放。"""
        await self.runtime.client.move_model(
            position_x=transform.position_x,
            position_y=transform.position_y,
            rotation=transform.rotation,
            size=transform.size,
            time_in_seconds=time_in_seconds,
            values_are_relative_to_model=values_are_relative_to_model,
        )

    async def play(self, policy: VTubeAvatarPolicy, sec: float) -> None:
        """按时间策略播放一段 avatar 动作。"""
        start = time.monotonic()
        interval = 1.0 / self.runtime.fps
        next_tick = start

        while True:
            now = time.monotonic()
            elapsed = now - start
            if elapsed >= sec:
                return
            await self.set_state(policy(elapsed))
            next_tick += interval
            await asyncio.sleep(max(0.0, next_tick - time.monotonic()))

    async def reset_neutral(self) -> None:
        """把 VTube Studio runtime 当前状态重置为中性 avatar 状态。"""
        await self.set_state(AvatarState.neutral())

    async def reset_transform(self) -> None:
        """把模型整体位置重置为中性变换。"""
        await self.set_transform(AvatarTransform.neutral())

    async def flush_once(self) -> None:
        """立即把当前 avatar 参数推送到 VTube Studio。

        常规控制依赖 runtime 后台循环；截图和录制场景需要更明确的帧对齐时，可以在
        `set_state()` 后调用本方法。
        """
        await self.runtime.flush_once()

    async def ensure_model_loaded(self) -> dict[str, Any]:
        """确认 VTube Studio 当前已加载 Live2D 模型。"""
        return await self.runtime.ensure_model_loaded()

    async def get_backend_state(self) -> dict[str, Any]:
        """读取 VTube Studio Public API 状态。"""
        return await self.runtime.client.api_state()

    def is_connected(self) -> bool:
        """返回 VTube Studio runtime 是否已启动。"""
        return self.runtime.is_running

    def get_client(self) -> Any:
        """返回底层 VTubeClient，主要用于调试。"""
        return self.runtime.client
