from __future__ import annotations

from collections.abc import Callable
from pathlib import Path
from typing import Any

from vtuber.avatar.avatar_profile import AvatarProfile
from vtuber.avatar.avatar_state import AvatarState
from vtuber.avatar.avatar_transform import AvatarTransform


AvatarPolicy = Callable[[float], AvatarState]


class AvatarError(RuntimeError):
    """avatar facade 层对外暴露的统一错误。"""


class Avatar:
    """项目统一的 avatar 控制入口。

    上层业务只需要使用 `AvatarState` 和这个 facade，不需要直接 import
    `vtube_studio`。当前第一版后端是 VTube Studio；未来接入自研 Cubism runtime
    或其他后端时，可以在这里通过 `backend` 参数切换。
    """

    def __init__(
        self,
        backend: str = "vtube_studio",
        host: str = "127.0.0.1",
        port: int = 25565,
        token_path: Path | str = ".vts_token.json",
        fps: float = 30.0,
        profile: AvatarProfile | Path | str | dict[str, Any] | None = None,
        profile_path: Path | str | None = None,
        config: dict[str, Any] | None = None,
    ) -> None:
        """创建 avatar facade。

        Args:
            backend: 运行时后端名。当前支持 `vtube_studio`。
            host: 后端服务地址。
            port: 后端服务端口。
            token_path: 后端认证 token 文件位置。对 VTube Studio 生效。
            fps: 后端刷新频率。
            profile: 可选 avatar profile。可以传 `AvatarProfile`、标准 profile 字典、
                标准 profile JSON 路径，或 VTube Studio `.vtube.json` 路径。VTube
                Studio 后端必须提供 profile。
            profile_path: `profile` 的路径别名。保留这个参数是为了让调用处更直观地
                表达“构造 Avatar 时传入一个 JSON”。
            config: 后端相关配置。当前保留给后续扩展。
        """
        if profile is not None and profile_path is not None:
            raise ValueError("Use either profile or profile_path, not both.")

        self.backend = backend
        self.host = host
        self.port = port
        self.token_path = token_path
        self.fps = fps
        self.profile = profile if profile is not None else profile_path
        self.config = config or {}
        self._driver = self._create_driver()

    async def __aenter__(self) -> Avatar:
        await self.start()
        return self

    async def __aexit__(self, *_exc: object) -> None:
        await self.stop()

    async def start(self) -> None:
        """启动 avatar 后端连接。"""
        try:
            await self._driver.start()
        except Exception as exc:
            raise AvatarError(str(exc)) from exc

    async def stop(self) -> None:
        """停止 avatar 后端连接，并尽量回到中性状态。"""
        try:
            await self._driver.stop()
        except Exception as exc:
            raise AvatarError(str(exc)) from exc

    async def set_state(self, state: AvatarState) -> None:
        """设置当前 avatar 状态。

        该方法只接收通用 `AvatarState`。具体如何映射到 VTube Studio 或其他运行时，
        由后端 driver 负责。
        """
        try:
            await self._driver.set_state(state)
        except Exception as exc:
            raise AvatarError(str(exc)) from exc

    async def hold_state(self, state: AvatarState, sec: float) -> None:
        """保持指定 avatar 状态一段时间。"""
        try:
            await self._driver.hold_state(state, sec)
        except Exception as exc:
            raise AvatarError(str(exc)) from exc

    async def set_transform(
        self,
        transform: AvatarTransform,
        *,
        time_in_seconds: float = 0.0,
        values_are_relative_to_model: bool = False,
    ) -> None:
        """设置 avatar 在画面中的整体位置、旋转和缩放。"""
        try:
            await self._driver.set_transform(
                transform,
                time_in_seconds=time_in_seconds,
                values_are_relative_to_model=values_are_relative_to_model,
            )
        except Exception as exc:
            raise AvatarError(str(exc)) from exc

    async def reset_transform(self) -> None:
        """把 avatar 整体位置重置为中性变换。"""
        try:
            await self._driver.reset_transform()
        except Exception as exc:
            raise AvatarError(str(exc)) from exc

    async def play(self, policy: AvatarPolicy, sec: float) -> None:
        """按时间策略播放一段 avatar 动作。

        Args:
            policy: 输入 elapsed 秒数，返回该时刻的 `AvatarState`。
            sec: 播放时长，单位秒。
        """
        try:
            await self._driver.play(policy, sec)
        except Exception as exc:
            raise AvatarError(str(exc)) from exc

    async def reset_neutral(self) -> None:
        """把 avatar 重置为中性状态。"""
        try:
            await self._driver.reset_neutral()
        except Exception as exc:
            raise AvatarError(str(exc)) from exc

    async def flush_once(self) -> None:
        """立即把当前 avatar 状态推送到底层后端。

        平时后端 runtime 会按固定 FPS 持续发送；截图、录制和测试时，为了让某一帧
        姿态更快进入后端，可以在 `set_state()` 之后显式调用。
        """
        try:
            await self._driver.flush_once()
        except Exception as exc:
            raise AvatarError(str(exc)) from exc

    async def ensure_model_loaded(self) -> dict[str, Any]:
        """确认后端已经加载可控制的模型，并返回后端模型信息。"""
        try:
            return await self._driver.ensure_model_loaded()
        except Exception as exc:
            raise AvatarError(str(exc)) from exc

    async def get_backend_state(self) -> dict[str, Any]:
        """返回后端运行状态。

        该接口用于调试和状态展示。返回字段由具体后端决定，上层业务不应把它作为
        稳定训练/推理数据结构。
        """
        try:
            return await self._driver.get_backend_state()
        except Exception as exc:
            raise AvatarError(str(exc)) from exc

    def is_connected(self) -> bool:
        """返回当前后端是否处于已连接状态。"""
        return self._driver.is_connected()

    def get_driver(self) -> Any:
        """返回底层 driver，主要用于调试。业务代码通常不应依赖它。"""
        return self._driver

    def get_client(self) -> Any:
        """返回底层 API client，主要用于调试或临时探查后端能力。"""
        return self._driver.get_client()

    def _create_driver(self) -> Any:
        if self.backend == "vtube_studio":
            from vtuber.vtube_studio.avatar_driver import VTubeAvatarDriver

            return VTubeAvatarDriver(
                host=self.host,
                port=self.port,
                token_path=self.token_path,
                fps=self.fps,
                profile=self.profile,
                config=self.config,
            )
        raise ValueError(f"Unsupported avatar backend: {self.backend}")
