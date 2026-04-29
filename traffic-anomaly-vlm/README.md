# traffic-anomaly-vlm

基于“检测/跟踪 + 时序特征 + 事件候选 + 视觉语言模型复核”的交通异常检测原型工程。

当前代码已经形成两条可运行路径：

1. 主干离线流程（`Orchestrator`）：检测跟踪 + 特征打分 + 粗边界候选（span-level VLM 复核当前在入口处默认注释）。
2. Pure VLM Baseline：纯视觉语言模型窗口化打分，支持逐帧分数与区间输出，并带基础评估指标。

## 1. 项目目标

- 输入：视频文件、帧文件夹或帧列表。
- 输出：
  - 帧级异常分数曲线（平滑前/后）
  - 粗异常区间
  - （可选）候选事件证据图与 VLM 复核分数
  - 评估指标（AUC/AP/EER/Accuracy）

## 2. src 架构总览

```text
src/
  core/                 # 通用工具：读视频、采样、日志、json
  perception/           # 感知：YOLO + ByteTrack、结果解析、track id 修复
  features/             # 特征构建：track 特征 + scene 特征融合
  proposals/            # 候选生成：boundary detector + event tree pipeline
  evidence/             # 证据构建：montage 等
  vlm/                  # VLM 加载、推理、prompt、输出解析
  pipeline/             # 主流程 orchestrator / pure_vlm_baseline
  eval/                 # 指标与可视化
  schemas.py            # 核心数据结构（Pydantic）
  settings.py           # 配置加载与实例化
```

## 3. 核心数据结构

定义在 `src/schemas.py`：

- `TrackObject`
  - 单帧单目标的检测/跟踪实体（bbox、类别、track_id、中心点、面积、可选帧尺寸）。
- `WindowFeature`
  - 一个滑窗的聚合特征与触发分数 `trigger_score`。
- `SpanProposal`
  - 扁平候选区间：区间索引/帧号、峰值与均值、先验分、VLM 分与融合分。
- `EvidencePack`
  - VLM 复核所需证据：关键帧路径 + 结构化摘要。

## 4. 两条主流程

### 4.1 主干离线流程（pipeline/orchestrator.py）

`Orchestrator.run_offline(input_video)` 当前执行链路：

1. 读取输入（支持视频文件、帧目录、txt/lst 帧列表）。
2. 按 `fps_sample` 采样。
3. YOLO + ByteTrack 得到 tracks。
4. 可选 track id 修复（`refine_track_ids`）。
5. 构建滑窗特征（track + scene）得到帧级 scores。
6. `BoundaryDetector` 产出粗区间 `coarse_spans`。
7. 输出打分曲线图到结果目录。

说明：span-level 证据构建 + VLM 复核的调用代码已在 `run_offline` 中预留，但默认被注释。

### 4.2 Pure VLM Baseline（pipeline/pure_vlm_baseline.py）

核心思想：

1. 固定窗口（默认 60 帧）滑动。
2. 每窗口分成若干 chunk（默认 4），每 chunk 均匀抽帧。
3. Stage1 prompt：场景理解与风险提示。
4. Stage2 prompt：输出 chunk 异常分数。
5. 将 chunk 分数投影回帧轴，平滑后阈值化得到异常区间。
6. 汇总视频级分数（max/top10）。

该流程由 `scripts/run_pure_vlm_baseline.py` 驱动，支持 manifest 批量评测。

## 5. 特征与打分细节

### 5.1 Track 特征（features/feature_components/track.py）

包含但不限于：

- 速度变化 / 方向变化
- 中心点预测误差、IoU 预测误差
- 中途消失且远离边缘的异常信号
- 下一时刻碰撞风险估计（基于轨迹外推最近距离）

### 5.2 Scene 特征（features/feature_components/scene.py）

包含：

- 计数/密度统计
- 低分辨率视觉变化（lowres）
- 轨迹布局变化（track layout）
- 可选 CLIP 变化、RAFT 变化（按权重与配置启用）

### 5.3 融合（features/feature_builder.py）

- `object_score`：按 `OBJECT_FEATURE_WEIGHTS` 加权归一化。
- `scene_score`：按 `SCENE_FEATURE_WEIGHTS` 加权归一化。
- `trigger_score`：按 `TRIGGER_BRANCH_WEIGHTS` 融合 object/scene。

## 6. 候选生成与树结构

### 6.1 BoundaryDetector（proposals/boundary_detector.py）

支持两种策略：

- `by_peeks`：峰值聚类并扩展为区间
- `by_thres`：自适应双阈值滞回

并支持 Savitzky-Golay 平滑、区间合并、最小长度过滤。

### 6.2 refine_spans_with_vlm（proposals/tree_pipeline.py）

逻辑包含：

1. frame-id 粗区间 -> flat span proposals
2. 每个 span 构建证据图（montage）
3. 调 VLM 单阶段打分（`build_span_score_prompt`）
4. 先验与 VLM 分按权重融合
5. 对阳性 spans 按 frame gap 简单合并

## 7. VLM 子系统

目录：`src/vlm/`

- `model_loader.py`：本地模型加载（默认 Qwen3-VL，Transformers）。
- `infer.py`：多图 + 文本模板推理，含 OOM 降级重试。
- `parser.py`：纯 VLM 两阶段输出解析、span score 解析。
- `prompts/`：stage1/stage2/span_score 的 prompt 构造。

## 8. 配置系统

使用 `src/settings.py`：

- `load_settings(path, overrides)` 读取 yaml 并支持递归覆盖。
- `instantiate_from_config` 将特定 section 映射到 dataclass。

默认配置在 `configs/default.yaml`，关键节包括：

- `video`：采样帧率、窗口大小、读入缩放
- `perception`：YOLO/ByteTrack 参数
- `track_refiner`
- `scene_features`
- `boundary_detector`
- `span_refine_config`
- `pure_vlm_baseline`
- `vlm`

## 9. 快速开始

### 9.1 安装

```bash
pip install -r requirements.txt
```

### 9.2 权重准备

默认代码依赖以下本地权重路径（可在配置里改）：

- YOLO：`weights/yolo26n.pt`
- Qwen3-VL：`weights/qwen3-vl-4bi`
- 可选 CLIP：`weights/clip-vit-base-patch32`
- 可选 RAFT：`weights/raft/raft-things.pth`

### 9.3 运行主干离线流程

```bash
python scripts/run_offline.py \
  --cfg configs/default.yaml \
  --input_video /path/to/video_or_frame_dir_or_list
```

或使用脚本：

```bash
bash scripts/sh/run_offline.sh
```

### 9.4 运行 Pure VLM Baseline

```bash
python scripts/run_pure_vlm_baseline.py \
  --config configs/default.yaml \
  --manifest /path/to/manifest.txt
```

常用覆盖参数：

- `--window-size`
- `--window-stride`
- `--sampled-frames-per-chunk`
- `--max-new-tokens`
- `--vlm-device`
- `--max-image-size`

## 10. 输入输出说明

### 10.1 输入

- 视频文件：如 mp4/avi
- 帧目录：按文件名排序（支持末尾数字解析）
- 帧列表：txt/lst，每行一张图路径

### 10.2 输出

默认写入 `outputs/` 下：

- `results/<video_id>/`：主干流程可视化与结果
- `assets/<video_id>/`：证据图（若启用 tree refine）
- `results/pure_vlm_baseline/`：
  - 每视频 json
  - `metrics_summary.json`
  - `config_used.json`
  - `manifest_used.txt`

## 11. 代码现状与注意事项

1. `Orchestrator.run_offline` 当前默认仅跑到粗边界检测，span-level VLM 细化调用代码在函数内被注释。
2. `scripts/run_offline.py` 仍按旧实验入口打印统计；在当前实现下可能与返回值不一致，需要按你的实际入口方式调整。
3. 原层级候选组件已移除，当前候选复核以 flat span 为单位。
4. 部分模块仍保留 `FIXME` 标记，属于实验阶段参数。

## 12. 推荐调参顺序

1. 先固定 `video.fps_sample` 与 `pure_vlm_baseline.window_size/window_stride`。
2. 调 `boundary_detector` 的 `peak_expand/min_span_len/merge_gap`，观察候选区间稳定性。
3. 再调 `OBJECT_FEATURE_WEIGHTS` 与 `SCENE_FEATURE_WEIGHTS`。
4. 最后再调 `span_refine_config` 的 `min_confidence/positive_threshold/merge_gap`。

## 13. 适合二次开发的位置

- 新增特征：`src/features/feature_components/`
- 新增候选策略：`src/proposals/boundary_detector.py`
- 新增证据形式：`src/evidence/builder.py`
- 新增 prompt 与解析：`src/vlm/prompts/` + `src/vlm/parser.py`
- 新增评估指标：`src/eval/metrics.py`

---

如果你希望，我可以在下一步再补一版“面向论文/汇报”的 README（含方法图、数学符号、实验清单）或者“面向部署”的 README（只保留可运行最短路径）。
