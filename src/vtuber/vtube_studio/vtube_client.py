from __future__ import annotations

import asyncio
import json
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import websockets


API_NAME = "VTubeStudioPublicAPI"
API_VERSION = "1.0"


class VTubeError(RuntimeError):
    """VTube Studio 客户端错误基类。"""


class VTubeAPIError(VTubeError):
    """VTube Studio 返回 APIError 响应时抛出。"""

    def __init__(self, error_id: int | None, message: str | None) -> None:
        self.error_id = error_id
        self.message = message or "Unknown VTube Studio API error."
        super().__init__(f"VTube Studio API error {error_id}: {self.message}")


@dataclass(frozen=True)
class VTubePlugin:
    """VTube Studio 插件身份信息，用于申请和复用认证 token。"""

    name: str = "VTuber AI"
    developer: str = "VTuber AI Local"


class VTubeClient:
    """VTube Studio Public API 的薄封装。

    这一层只处理 WebSocket 通讯、认证、请求/响应和参数注入，不包含“摇头”、
    “张嘴”或项目里的 canonical reaction state 等业务语义。
    """

    def __init__(
        self,
        host: str = "127.0.0.1",
        port: int = 25565,
        plugin: VTubePlugin | None = None,
        token_path: Path | str = ".vts_token.json",
    ) -> None:
        """创建客户端。

        Args:
            host: VTube Studio API 监听地址，通常是本机 `127.0.0.1`。
            port: VTube Studio Public API 端口，按当前设置默认为 `25565`。
            plugin: 插件身份信息。VTube Studio 会按这个身份授权并生成 token。
            token_path: 本地 token 存储位置，避免每次运行都重新弹窗授权。
        """
        self.uri = f"ws://{host}:{port}"
        self.plugin = plugin or VTubePlugin()
        self.token_path = Path(token_path)
        self._ws: Any | None = None
        self._request_lock = asyncio.Lock()

    async def __aenter__(self) -> VTubeClient:
        await self.connect()
        return self

    async def __aexit__(self, *_exc: object) -> None:
        await self.close()

    async def connect(self) -> None:
        """建立到 VTube Studio Public API 的 WebSocket 连接。"""
        if self._ws is not None:
            return
        self._ws = await websockets.connect(self.uri)

    async def close(self) -> None:
        """关闭 WebSocket 连接。"""
        if self._ws is None:
            return
        await self._ws.close()
        self._ws = None

    async def request(self, message_type: str, data: dict[str, Any] | None = None) -> dict[str, Any]:
        """发送一条原始 VTube Studio API 请求并返回完整响应。

        Args:
            message_type: VTube Studio API 的消息类型，例如 `APIStateRequest`。
            data: 该消息类型对应的 data 字段；无 data 的请求可传 `None`。

        Raises:
            VTubeError: 尚未连接 WebSocket。
            VTubeAPIError: VTube Studio 返回 APIError。
        """
        if self._ws is None:
            raise VTubeError("VTube Studio WebSocket is not connected.")

        async with self._request_lock:
            payload: dict[str, Any] = {
                "apiName": API_NAME,
                "apiVersion": API_VERSION,
                "requestID": f"vtuber-{uuid.uuid4().hex}",
                "messageType": message_type,
            }
            if data is not None:
                payload["data"] = data

            # websockets 不允许同一个连接上多个协程同时 recv，所以 send/recv 必须串行。
            await self._ws.send(json.dumps(payload))
            response = json.loads(await self._ws.recv())
            if response.get("messageType") == "APIError":
                error = response.get("data", {})
                raise VTubeAPIError(error.get("errorID"), error.get("message"))
            return response

    async def api_state(self) -> dict[str, Any]:
        """读取 API 状态。

        常用于检查 API 是否开启、VTube Studio 版本号，以及当前会话是否已认证。
        """
        response = await self.request("APIStateRequest")
        return response["data"]

    async def authenticate(self) -> dict[str, Any]:
        """完成插件认证。

        优先复用本地保存的 token；如果 token 缺失或已失效，会重新申请 token。
        第一次申请 token 时，VTube Studio 会弹出授权窗口，需要用户手动允许。
        """
        token = self._load_token()
        if token is None:
            token = await self.request_auth_token()
            self._save_token(token)

        data = await self._authenticate_with_token(token)
        if data.get("authenticated"):
            return data

        token = await self.request_auth_token()
        self._save_token(token)
        data = await self._authenticate_with_token(token)
        if not data.get("authenticated"):
            reason = data.get("reason") or data.get("message") or "authentication was rejected"
            raise VTubeError(f"VTube Studio authentication failed: {reason}")
        return data

    async def _authenticate_with_token(self, token: str) -> dict[str, Any]:
        """使用指定 token 发送 AuthenticationRequest。"""
        response = await self.request(
            "AuthenticationRequest",
            {
                "pluginName": self.plugin.name,
                "pluginDeveloper": self.plugin.developer,
                "authenticationToken": token,
            },
        )
        return response["data"]

    async def request_auth_token(self) -> str:
        """向 VTube Studio 申请新的认证 token。

        该请求通常会触发 VTube Studio 授权弹窗。用户同意后，响应中会返回 token。
        """
        response = await self.request(
            "AuthenticationTokenRequest",
            {
                "pluginName": self.plugin.name,
                "pluginDeveloper": self.plugin.developer,
            },
        )
        return response["data"]["authenticationToken"]

    async def list_live2d_parameters(self) -> dict[str, Any]:
        """读取当前加载模型的 Live2D 参数列表。

        返回数据包含模型是否加载、模型名称、以及每个 Live2D 参数的 ID、当前值、
        最小值、最大值和默认值等信息。
        """
        response = await self.request("Live2DParameterListRequest")
        return response["data"]

    async def inject_parameters(
        self,
        values: dict[str, float],
        *,
        face_found: bool = False,
        mode: str = "set",
        weight: float | None = None,
    ) -> None:
        """向 VTube Studio 注入 tracking 参数值。

        Args:
            values: 参数 ID 到数值的映射，例如 `{"FaceAngleX": 20.0}`。
            face_found: 是否告诉 VTube Studio 当前检测到脸。控制默认 face tracking
                参数时通常传 `True`。
            mode: VTube Studio 注入模式，常用 `set`。其他模式需按官方 API 语义使用。
            weight: 可选的参数混合权重。传入后会应用到每个参数值。

        注意：
            需要持续发送参数值。只发送一次通常只能短暂影响模型，随后控制权会回到
            VTube Studio 的默认 tracking、idle animation 或其他 value provider。
        """
        parameter_values: list[dict[str, float | str]] = []
        for parameter_id, value in values.items():
            item: dict[str, float | str] = {"id": parameter_id, "value": float(value)}
            if weight is not None:
                item["weight"] = float(weight)
            parameter_values.append(item)

        await self.request(
            "InjectParameterDataRequest",
            {
                "faceFound": face_found,
                "mode": mode,
                "parameterValues": parameter_values,
            },
        )

    async def move_model(
        self,
        *,
        position_x: float,
        position_y: float,
        rotation: float,
        size: float,
        time_in_seconds: float = 0.0,
        values_are_relative_to_model: bool = False,
    ) -> dict[str, Any]:
        """移动当前加载的模型。

        这里直接封装官方 `MoveModelRequest`。`time_in_seconds=0` 表示瞬时移动；
        如果设置为正数，VTube Studio 会在该时间内平滑移动到目标位置。
        """
        response = await self.request(
            "MoveModelRequest",
            {
                "timeInSeconds": float(max(0.0, time_in_seconds)),
                "valuesAreRelativeToModel": bool(values_are_relative_to_model),
                "positionX": float(position_x),
                "positionY": float(position_y),
                "rotation": float(rotation),
                "size": float(size),
            },
        )
        return response["data"]

    async def live2d_parameters(self) -> dict[str, Any]:
        """旧接口兼容别名；新代码应使用 `list_live2d_parameters()`。"""
        return await self.list_live2d_parameters()

    async def request_token(self) -> str:
        """旧接口兼容别名；新代码应使用 `request_auth_token()`。"""
        return await self.request_auth_token()

    def _load_token(self) -> str | None:
        """从本地 token 文件读取当前插件身份对应的 token。"""
        if not self.token_path.exists():
            return None
        data = json.loads(self.token_path.read_text(encoding="utf-8"))
        if data.get("pluginName") != self.plugin.name:
            return None
        if data.get("pluginDeveloper") != self.plugin.developer:
            return None
        return data.get("authenticationToken")

    def _save_token(self, token: str) -> None:
        """把认证 token 保存到本地。"""
        self.token_path.write_text(
            json.dumps(
                {
                    "pluginName": self.plugin.name,
                    "pluginDeveloper": self.plugin.developer,
                    "authenticationToken": token,
                },
                indent=2,
            ),
            encoding="utf-8",
        )
