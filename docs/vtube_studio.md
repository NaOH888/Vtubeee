# VTube Studio 接入说明

## 当前判断

VTube Studio 可以作为本项目早期的 Live2D 运行端。也就是说，早期不必先自己写
Cubism 渲染程序，可以先让 Python 通过 VTube Studio 的公开插件 API 控制模型。

需要区分两套网络功能：

- **手机/桌面串流网络配置**：用于 iPhone/Android 把面捕数据传给 PC 端。文档中
  默认端口是 `25565`。这不是我们要用的 Python 控制接口。
- **VTube Studio Public API**：用于外部插件或脚本控制 VTube Studio。默认地址是
  `ws://localhost:8001`。这是本项目 Python 端应该使用的接口。

## 适合我们的控制路径

建议早期采用：

```text
Python reaction model / scripted policy
        |
        v
VTube Studio Public API over WebSocket
        |
        v
VTube Studio tracking parameters
        |
        v
VTS Parameter Setup
        |
        v
Live2D Cubism model parameters
```

VTube Studio 里不是直接把外部输入写进每一个 Live2D 参数，而是通常经过一层
tracking parameter。VTube Studio 的模型设置页可以把任意输入参数映射到任意
Live2D 输出参数，并配置输入范围、输出范围、平滑、限幅等。

## VTube Studio 端设置

### 1. 启动插件 API

在 VTube Studio 桌面版中：

1. 打开设置。
2. 找到插件/API 相关选项。
3. 开启 `Allow Plugin API access`。
4. 端口先保持默认 `8001`。

Python 端连接：

```text
ws://localhost:8001
```

如果连接不上，优先检查：

- VTube Studio 是否正在运行。
- 是否开启了 `Allow Plugin API access`。
- 端口是否仍然是 `8001`。
- 防火墙或安全软件是否拦截本地 WebSocket 连接。

### 2. 第一次认证

VTube Studio API 需要插件认证。流程是：

1. Python 发送 `AuthenticationTokenRequest`。
2. VTube Studio 弹窗询问是否允许该插件控制 VTube Studio。
3. 用户点击允许后，Python 得到 `authenticationToken`。
4. 后续连接使用 `AuthenticationRequest` 携带这个 token 完成认证。

token 应保存到本地配置文件中，后续不需要每次弹窗授权。

### 3. 读取当前模型参数

连接并认证后，可以发送：

```text
Live2DParameterListRequest
```

用途：

- 确认当前是否加载了模型。
- 获取模型的 Live2D 参数 ID。
- 获取每个参数的当前值、最小值、最大值、默认值。

这一步对后续构造 canonical state 到 Live2D 参数的映射很重要。

### 4. 注入默认 tracking 参数

可以直接向 VTube Studio 的默认 tracking 参数写值，例如：

```text
FaceAngleX
FaceAngleY
FaceAngleZ
MouthOpen
MouthSmile
EyeOpenLeft
EyeOpenRight
EyeLeftX
EyeLeftY
EyeRightX
EyeRightY
```

使用请求：

```text
InjectParameterDataRequest
```

注意：如果要持续控制某个参数，必须至少每秒重新发送一次。否则 VTube Studio 会
认为该参数输入丢失，并把控制权交还给原来的来源，例如摄像头 tracking、idle
animation 或默认值。

### 5. 创建自定义 tracking 参数

也可以创建我们自己的输入参数，例如：

```text
VTAIHeadYaw
VTAIHeadPitch
VTAIEyeLOpen
VTAIEyeROpen
VTAIMouthOpen
VTAIMouthForm
VTAIBodySway
```

使用请求：

```text
ParameterCreationRequest
```

创建后，这些参数会出现在 VTube Studio 的参数映射 UI 中，可以像默认 face
tracking 参数一样映射到 Live2D 输出参数。

这种方式更适合本项目，因为它把“AI 反应状态”和 VTube Studio 自带面捕输入分开，
调试时更清楚。

## 推荐的第一版实验

第一版不要直接接模型，先做一个脚本化控制实验。

目标：

```text
Python 脚本生成周期性参数
        |
        v
VTube Studio API
        |
        v
模型头部左右摆动、眨眼、张嘴
```

建议控制参数：

- `FaceAngleX`：头部左右。
- `FaceAngleY`：头部上下。
- `FaceAngleZ`：头部倾斜。
- `MouthOpen`：嘴巴开合。
- `EyeOpenLeft` / `EyeOpenRight`：眼睛开合。

如果默认参数可以直接驱动模型，说明 VTube Studio 自动映射或模型配置已经可用。
如果参数值变化但模型不动，就需要进入模型设置的 `VTS Parameter Setup`，检查输入
参数是否映射到了对应 Live2D 参数。

## 推荐的正式路线

正式路线建议使用自定义参数：

```text
canonical reaction state
        |
        v
VTAI* custom tracking parameters
        |
        v
VTS Parameter Setup 手动映射
        |
        v
Live2D Cubism parameters
```

这样做的好处：

- 不污染 VTube Studio 自带的 face tracking 参数。
- 可以同时保留摄像头 tracking 和 AI 控制，用权重或开关做混合。
- 参数命名和项目里的 canonical state 对齐。
- 不同皮套只需要调整 VTube Studio 里的映射，不需要改模型输出空间。

## 参数控制优先级

VTube Studio 内部同一个 Live2D 参数可能被多个来源控制。常见来源包括：

- 默认 Live2D 参数值。
- idle animation。
- face tracking 或 API 注入的 tracking 参数。
- 一次性 animation。
- expression。
- physics。

如果 Python 注入的参数看起来发出去了，但模型没有动，常见原因是：

- 该 tracking 参数没有映射到 Live2D 参数。
- expression 或 animation 正在覆盖该参数。
- physics 覆盖了该参数。
- VTube Studio 端设置了过强的平滑或限幅。
- Python 没有持续发送参数，导致控制权丢失。

## 与项目主线的关系

这一路线对应项目中的 Live2D 运行时通讯支线。

短期目标：

```text
手写策略 -> VTube Studio -> Live2D 模型动起来
```

中期目标：

```text
多模态模型 -> canonical reaction state -> VTube Studio custom parameters
```

长期目标：

```text
如果 VTube Studio API 限制过多，再考虑自研 Cubism runtime
```
