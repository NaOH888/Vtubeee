# VTuber 项目当前实现说明

## 1. 项目目标

当前仓库已经落地的主线不是“直接从直播多模态流到最终 VTuber 行为”，而是先把下列几个基础环节打通：

```text
VTube Studio 控制
-> 虚拟摄像头采集
-> 自动 bbox 标注
-> LMDB 数据集
-> YOLO 检测训练
-> crop -> AvatarState 回归训练
```

这几部分构成了后续多模态主线的前置基础。

当前的实际重点是：

1. 稳定控制 VTube Studio 中的 Live2D 模型。
2. 采集带 bbox 和 AvatarState 标签的数据。
3. 训练检测模型得到角色 bbox。
4. 训练图像 crop 到 AvatarState 的回归模型。

## 2. 当前框架

### 2.1 目录结构

当前主要代码分为三层：

```text
src/vtuber/
    avatar/           通用 Avatar 抽象、状态空间、采集接口
    vtube_studio/     VTube Studio 后端实现

src/dataset/
    bbox_dataset_sampler.py   bbox + state 数据采样
    export_lmdb_to_yolo.py    LMDB -> YOLO 数据集导出
    utils.py                  采样和图像处理工具

src/training/
    yolo/           YOLO 检测训练
    avatar_state/   crop -> AvatarState 回归训练
```

脚本目录：

```text
scripts/
    vts_static_then_motion.py
    vts_virtual_camera_capture.py
    debug_paired_virtual_camera_bbox.py
    inspect_bbox_lmdb.py
```

### 2.2 Avatar 层

`src/vtuber/avatar/` 提供项目统一的抽象。

核心对象包括：

- `AvatarState`
  - 项目内部统一姿态参数空间。
  - 当前已包含头部、眼睛、嘴部、眉毛等字段。
  - 支持 `sample_random()` 生成随机静态姿态样本。

- `AvatarTransform`
  - 模型整体位移、旋转、缩放。
  - 通过 VTube Studio 的 `MoveModelRequest` 控制。

- `AvatarProfile`
  - 不同皮套的参数映射配置。
  - 当前通过 `.vtube.json` 或标准 profile JSON 读取。

- `Avatar`
  - 上层统一调用入口。
  - 负责把 `AvatarState` 和 `AvatarTransform` 发给具体后端。

- `AvatarCapture`
  - 从虚拟摄像头采集单帧或序列。

### 2.3 VTube Studio 后端

`src/vtuber/vtube_studio/` 是当前唯一已接通的运行时后端。

主要职责：

- `vtube_client.py`
  - 与 VTube Studio Public API 通讯。

- `runtime.py`
  - 持续注入参数。

- `mapper.py`
  - `AvatarState -> backend parameter` 映射。

- `avatar_driver.py`
  - VTube Studio 后端驱动。

当前已经打通：

- Public API 连接
- 参数注入
- 模型整体位置/旋转/缩放控制
- 虚拟摄像头采集

### 2.4 数据采样层

`src/dataset/bbox_dataset_sampler.py` 是当前最重要的数据入口。

它做的事情是：

```text
随机 AvatarState
+ 随机 AvatarTransform
-> 驱动 VTube Studio
-> 抓取 VTubeStudioCam 和 OBS Virtual Camera
-> 从干净 VTube 图计算 bbox
-> 写入 LMDB
```

LMDB 中每条样本主要包含：

```text
sample/{id}/obs
sample/{id}/vtube
sample/{id}/meta
```

`meta` 中包含：

- `obs_bbox_xyxy`
- `vtube_bbox_xyxy`
- `state`
- `transform`
- `avatar_id`
- `profile_id`

这意味着同一份数据可以同时服务两条训练线：

1. `OBS image -> bbox`
2. `crop image -> AvatarState`

### 2.5 YOLO 检测训练

`src/training/yolo/train_bbox.py` 用于调用 Ultralytics 训练检测器。

流程是：

```text
LMDB
-> export_lmdb_to_yolo.py
-> 标准 YOLO 目录
-> train_bbox.py
```

当前已经验证：

- 单皮套训练对跨皮套泛化不够。
- 两个皮套训练后，对第三个皮套的泛化已经明显改善。

说明 bbox 检测这条线是可行的，且关键在于皮套多样性和位置分布，而不是过度纠结模型结构。

### 2.6 AvatarState 回归训练

`src/training/avatar_state/` 当前已经完全切到 `timm` 路线，不再依赖 `mmpose`。

当前实现：

- `dataset.py`
  - `AvatarStateDataset`
  - 从 LMDB 读图、裁切、缩放、转 tensor

- `timm_wrapper.py`
  - `timm backbone + MLP head -> AvatarState`

- `model_builder.py`
  - 模型构造入口
  - 当前支持：
    - `timm_convnextv2_tiny`
    - `timm_convnextv2_base`
    - `timm_efficientnetv2_rw_s`

- `train.py`
  - 训练入口
  - 支持本地预训练权重和断点续训

- `val.py`
  - 验证 / 推理入口

当前默认模型是：

```text
timm_convnextv2_tiny
```

## 3. 当前已实现功能

### 3.1 VTube Studio 控制

已实现：

- 连接 VTube Studio API
- 注入参数控制模型
- 控制整体位移/旋转/缩放
- 从虚拟摄像头抓图

### 3.2 数据采样

已实现：

- 主动驱动采样，而不是被动录制
- 自动保存 bbox
- 自动保存 `AvatarState`
- 自动保存 `AvatarTransform`
- LMDB 追加写入

### 3.3 数据检查

已实现：

- `scripts/inspect_bbox_lmdb.py`
  - 查看 LMDB 内容
  - 按索引读取样本
  - 显示 bbox 对齐结果

### 3.4 YOLO 训练

已实现：

- LMDB -> YOLO 导出
- YOLO bbox 训练脚本
- 跨皮套泛化测试

### 3.5 AvatarState 训练

已实现：

- LMDB -> crop -> tensor 数据流
- `timm` 预训练 backbone
- `MLP` 回归头
- 训练日志
- 验证脚本
- `--pretrained-path`
- `--resume`

## 4. 使用说明

### 4.1 环境

当前 `avatar_state` 训练路径建议依赖文件：

- [requirements-avatar-state.txt](E:/pycharmProjs/VTuber/requirements-avatar-state.txt)

其中说明了：

- `numpy`
- `opencv-python`
- `torch / torchvision / torchaudio`
- `timm`

注意：

- `torch / torchvision / torchaudio` 需要从 `cu130` 源安装，保持三件套一致。
- 当前已经不再建议继续走 `mmpose` / OpenMMLab 训练路线。

### 4.2 采集 bbox + state 数据

入口：

- [bbox_dataset_sampler.py](E:/pycharmProjs/VTuber/src/dataset/bbox_dataset_sampler.py)

典型命令示意：

```bash
.\.venv\Scripts\python.exe src\dataset\bbox_dataset_sampler.py ^
  --profile src\vtuber\avatar_profiles\hiyori.vtube.json ^
  --avatar-id hiyori ^
  --count 1000 ^
  --output outputs\hiyori_state.lmdb
```

当前采样内容包括：

- `obs`
- `vtube`
- `bbox`
- `AvatarState`
- `AvatarTransform`

### 4.3 检查 LMDB

入口：

- [inspect_bbox_lmdb.py](E:/pycharmProjs/VTuber/scripts/inspect_bbox_lmdb.py)

查看概要：

```bash
.\.venv\Scripts\python.exe scripts\inspect_bbox_lmdb.py outputs\hiyori_state.lmdb
```

查看某条样本：

```bash
.\.venv\Scripts\python.exe scripts\inspect_bbox_lmdb.py ^
  outputs\hiyori_state.lmdb ^
  --show-fig ^
  --fig-index 0
```

### 4.4 导出 YOLO 数据集

入口：

- [export_lmdb_to_yolo.py](E:/pycharmProjs/VTuber/src/dataset/export_lmdb_to_yolo.py)

示意：

```bash
.\.venv\Scripts\python.exe src\dataset\export_lmdb_to_yolo.py ^
  outputs\hiyori_state.lmdb ^
  --output outputs\yolo_hiyori ^
  --val-ratio 0.1 ^
  --require-state
```

### 4.5 训练 YOLO

入口：

- [train_bbox.py](E:/pycharmProjs/VTuber/src/training/yolo/train_bbox.py)

示意：

```bash
.\.venv\Scripts\python.exe src\training\yolo\train_bbox.py ^
  --data outputs\yolo_hiyori\data.yaml ^
  --model yolov8n.pt ^
  --epochs 100 ^
  --imgsz 640 ^
  --batch 16 ^
  --device 0 ^
  --single-cls
```

### 4.6 训练 AvatarState 回归

入口：

- [train.py](E:/pycharmProjs/VTuber/src/training/avatar_state/train.py)

#### 直接使用在线预训练 backbone

```bash
.\.venv\Scripts\python.exe src\training\avatar_state\train.py ^
  outputs\train.lmdb ^
  --val-lmdb outputs\val.lmdb ^
  --model-name timm_convnextv2_tiny ^
  --image-source vtube ^
  --image-size 384 ^
  --batch-size 16 ^
  --epochs 30 ^
  --freeze-epochs 3 ^
  --device cuda:0 ^
  --name convnextv2_tiny_vtube_384
```

#### 使用本地预训练权重

```bash
.\.venv\Scripts\python.exe src\training\avatar_state\train.py ^
  outputs\train.lmdb ^
  --val-lmdb outputs\val.lmdb ^
  --model-name timm_convnextv2_tiny ^
  --pretrained-path path\to\weights.pth ^
  --device cuda:0
```

#### 从中断点继续训练

```bash
.\.venv\Scripts\python.exe src\training\avatar_state\train.py ^
  outputs\train.lmdb ^
  --val-lmdb outputs\val.lmdb ^
  --resume outputs\avatar_state_runs\convnextv2_tiny_vtube_384\last.pt ^
  --epochs 30 ^
  --device cuda:0
```

### 4.7 验证 / 推理 AvatarState 模型

入口：

- [val.py](E:/pycharmProjs/VTuber/src/training/avatar_state/val.py)

示意：

```bash
.\.venv\Scripts\python.exe src\training\avatar_state\val.py ^
  outputs\avatar_state_runs\convnextv2_tiny_vtube_384\best.pt ^
  outputs\val.lmdb ^
  --device cuda:0
```

输出：

- 总体 `val_loss`
- 每字段 `MAE`
- 逐样本预测 `jsonl`

## 5. 当前已知注意点

### 5.1 crop 默认是正方形拉伸

当前 `AvatarStateDataset` 的 crop 流程是：

```text
按 bbox 裁切
-> resize 到 image_size x image_size
```

默认不是保长宽比，而是直接缩放到正方形。

这意味着：

- 对头部、嘴部、眼睛这些局部几何会有一定形变。
- 当整个人物 bbox 过大时，眉毛等细节分辨率会不足。

因此当前更推荐：

- 提高 `--image-size`，例如 `384`
- 先从 `vtube` 干净图像开始

### 5.2 眉毛等细粒度参数更难

当前输出字段里，眉毛等细粒度参数比头部姿态更难学。

原因主要是：

- 整体 bbox 裁切后人脸区域占比还不够大
- 256 分辨率下细节不足

后续可能要考虑：

- 更大输入分辨率
- face-biased crop
- 多分支结构

### 5.3 旧的 MMPose 路线已废弃

当前仓库不再把 `mmpose` 作为主训练基座。

原因：

- 依赖链过重
- 对 `mmcv/mmengine/mmdet` 版本敏感
- 不适合作为当前项目的长期通用路线

## 6. 下一步建议

当前最值得继续推进的工作：

1. 用多个皮套继续扩充 LMDB 数据。
2. 先稳定训练 `vtube crop -> AvatarState`。
3. 评估 `384` 或更高输入分辨率对眉眼嘴参数的帮助。
4. 之后再切到 `obs crop -> AvatarState`，研究复杂背景域差异。

当前不建议再优先投入：

- `mmpose` / landmark 训练主线
- 复杂多模态端到端模型
- 直接语音生成

当前最重要的是把：

```text
可控采样
可复现数据集
可训练视觉回归模型
```

这三件事做稳。
