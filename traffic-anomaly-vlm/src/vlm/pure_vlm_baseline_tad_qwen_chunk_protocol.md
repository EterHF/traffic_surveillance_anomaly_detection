# 纯 VLM Baseline 方案说明（TAD + Qwen3-VL / Qwen3.5-VL 风格）

> 目标：构建一个**不依赖 YOLO、光流、图网络或显式触发器**的纯 VLM 基准，用于后续与“触发式 + VLM”正式方法对比。  
> 核心协议：**25 FPS 工作帧率 + 80 帧滑窗 + 4 段 chunk scoring + 两阶段 prompt + 帧级回写**。

---

# 1. 方案定位

这是一个**纯 VLM** 基准：

- 输入：视频帧序列
- 中间：滑窗、采样、两阶段 prompting
- 输出：窗口级异常分数、chunk 级异常分数、帧级异常分数曲线、异常区间

这个基准**不引入**：
- 目标检测
- 跟踪
- 光流
- 图建模
- 手工触发器

因此它的作用是：

1. 给出一个“只靠 VLM 能做到什么程度”的参考线  
2. 为后续正式方法提供公平对比对象  
3. 验证 EventVAD 风格的“短事件单元 + 分层推理”是否优于“整段视频直接问答”

---

# 2. 设计依据

## 2.1 为什么采用短窗口而不是整段视频

长视频直接送入 VLM，会遇到几个问题：

- 时序信息过长，模型容易忽略局部异常
- 输出异常区间不稳定
- 不适合做可回写的帧级评分
- 对异常只占少数帧的视频，整段判断容易被正常片段淹没

因此采用**固定长度重叠滑窗**，让模型处理更短、更一致的局部事件单元。

---

## 2.2 为什么采用两阶段 prompt

采用两阶段 prompt 的原因：

### 第一阶段：中性描述
先让模型描述：
- 主要对象
- 主要运动
- 是否出现明显变化
- 哪些时间块更可疑

### 第二阶段：异常裁决
再让模型根据第一阶段的描述输出：
- 是否异常
- 异常分数
- 异常类型
- 异常 chunk

这样做的优点：

- 降低模型直接打分时的幻觉
- 提高输出结构化程度
- 更贴近 EventVAD 的 hierarchical prompting 思路

---

## 2.3 为什么采用 80 帧窗口

已知你当前对 TAD 的工程恢复假设是 **25 FPS**。

因此：

- 80 帧 ≈ 3.2 秒
- 20 帧 ≈ 0.8 秒
- 16 帧 ≈ 0.64 秒

80 帧窗口的优点：

- 一般足够覆盖一个局部异常过程
- 不至于太长，避免 VLM 时序能力被拉垮
- 便于再切成 4 段 chunk 做粗时间定位

---

# 3. 总体流程

```text
视频帧序列
  -> 固定工作帧率 25 FPS
  -> 80 帧滑窗，步长 20
  -> 每窗切成 4 个 chunk
  -> 每个 chunk 均匀采样 4 帧
  -> 共 16 帧送入 VLM
  -> Prompt Stage 1: 中性描述
  -> Prompt Stage 2: 异常评分
  -> 得到窗口级 overall_score + 4 个 chunk_scores
  -> 将 chunk_scores 回写到帧级
  -> 平滑帧级曲线
  -> 阈值切分异常区间
  -> 输出视频级分数与异常区间
```

---

# 4. 输入数据组织

## 4.1 原始输入

每个视频是一个按时间排序的帧序列：

```python
frames = [I_0, I_1, I_2, ..., I_{T-1}]
```

其中每帧通常是：

```python
I_t.shape == (H, W, 3)
```

建议统一：
- RGB
- uint8
- 文件名按时间顺序排序

---

## 4.2 工作帧率

统一假设：

```python
WORKING_FPS = 25
```

用途：
- 把帧数换算成秒
- 统一窗口、步长、异常区间描述
- 方便后续实验复现

注意：这个 fps 是你对当前 TAD 分发版本的工程性恢复假设，不是必须写进模型逻辑。

---

# 5. 滑窗与 chunk 设计

## 5.1 滑窗参数

```python
WINDOW_SIZE = 80   # 每窗 80 帧
WINDOW_STRIDE = 20 # 相邻窗口步长 20 帧
```

于是第 k 个窗口定义为：

```python
start_k = k * WINDOW_STRIDE
end_k = start_k + WINDOW_SIZE
window_k = frames[start_k:end_k]
```

只保留满足：

```python
len(window_k) == 80
```

---

## 5.2 chunk 划分

每个 80 帧窗口切成 4 段：

```python
chunk_1 = frames[0:20]
chunk_2 = frames[20:40]
chunk_3 = frames[40:60]
chunk_4 = frames[60:80]
```

每段长度 20 帧。

---

## 5.3 每个 chunk 的采样

每个 chunk 均匀采样 4 帧：

```python
sampled_idx = [0, 6, 13, 19]
```

实际可写成通用函数：

```python
def uniform_sample_indices(length: int, n: int) -> list[int]:
    # 例如 length=20, n=4 -> [0, 6, 13, 19]
```

于是一个窗口最终输入模型的总帧数是：

```python
4 chunks * 4 frames = 16 frames
```

---

## 5.4 为什么不是直接采 16 帧

如果直接在 80 帧中均匀采 16 帧，模型知道“这是 16 帧”，但不知道它们的粗时间结构。

先切 4 个 chunk 再各采 4 帧的优点：

- 模型可以更明确地对“前四分之一 / 后四分之一”做判断
- 输出 chunk_scores 时更自然
- 后处理回写到帧级更方便
- 比让模型直接输出“第几帧异常”更稳定

---

# 6. VLM 输入组织

## 6.1 单窗口输入单元

每个窗口输入给模型的是：

- 16 张按时间顺序排列的图像
- 文字 prompt

因此单次推理的逻辑单元是：

```python
{
  "window_id": k,
  "window_start": start_k,
  "window_end": end_k - 1,
  "sampled_frames": [16 images],
  "prompt": "..."
}
```

---

## 6.2 建议的多模态输入格式

推荐用 Transformers 的 chat-template 风格：

```python
messages = [
    {
        "role": "user",
        "content": [
            {"type": "image", "image": img_1},
            {"type": "image", "image": img_2},
            ...
            {"type": "image", "image": img_16},
            {"type": "text", "text": prompt_text}
        ]
    }
]
```

说明：

- 若你实际使用的模型支持 `video` 类型输入，也可以将 16 帧封装为 video
- 但为了更稳定地控制时间顺序和 chunk 对应关系，第一版更推荐**显式多图输入**

---

# 7. 两阶段 Prompt 设计

---

## 7.1 Stage 1：中性描述 Prompt

### 目的
让模型先做“看见了什么”和“哪里可能有变化”的描述，而不是直接打异常分。

### Prompt 模板

```text
You are a traffic surveillance video analyst.

You are given 16 frames in temporal order from a short traffic surveillance clip.
These 16 frames come from 4 consecutive temporal chunks, and each chunk contains 4 sampled frames.

Your task in this step is NOT to output the final anomaly score yet.

Please analyze the clip and output a JSON object with:
- main_objects: major traffic participants visible in the clip
- scene_summary: a concise summary of the whole clip
- chunk_descriptions: a list of 4 short descriptions, one for each temporal chunk
- noticeable_change: whether there is any sudden motion, risky interaction, scene hazard, or abnormal transition
- likely_abnormal_chunks: a list of chunk indices among [1,2,3,4]
- risk_hint: one of ["normal", "possibly_risky", "clearly_risky"]

Rules:
- Output JSON only.
- Do not output anything outside JSON.
```

---

## 7.2 Stage 1 输出格式

```json
{
  "main_objects": ["car", "truck"],
  "scene_summary": "Traffic is mostly normal, but one vehicle shows abrupt deviation later.",
  "chunk_descriptions": [
    "Vehicles move normally.",
    "A car approaches another vehicle.",
    "Interaction becomes closer and less stable.",
    "A risky deviation appears."
  ],
  "noticeable_change": true,
  "likely_abnormal_chunks": [3, 4],
  "risk_hint": "possibly_risky"
}
```

---

## 7.3 Stage 2：异常评分 Prompt

### 目的
根据同一段 16 帧和 Stage 1 的描述，输出：
- 整窗异常分数
- 4 段 chunk 分数
- 异常类型
- 支持证据

### Prompt 模板

```text
You are a traffic anomaly detection judge.

You are given:
1. the same 16-frame traffic surveillance clip
2. a structured neutral description generated in the previous step

The clip is divided into 4 temporal chunks in order.

Please output a JSON object with:
- is_anomaly: true or false
- overall_score: a float in [0,1]
- chunk_scores: a list of 4 floats in [0,1]
- anomaly_type: one of
  ["collision_like", "dangerous_interaction", "abnormal_stop",
   "wrong_direction", "possible_overspeed", "fire_or_smoke",
   "road_obstruction", "normal", "unknown"]
- abnormal_chunks: a list of chunk indices among [1,2,3,4]
- confidence: a float in [0,1]
- short_reason: one sentence
- supporting_evidence: a short list of evidence phrases

Rules:
- If the clip is normal, set anomaly_type to "normal".
- chunk_scores must contain exactly 4 numbers.
- Output JSON only.
```

---

## 7.4 Stage 2 输出格式

```json
{
  "is_anomaly": true,
  "overall_score": 0.81,
  "chunk_scores": [0.08, 0.22, 0.74, 0.86],
  "anomaly_type": "dangerous_interaction",
  "abnormal_chunks": [3, 4],
  "confidence": 0.77,
  "short_reason": "Two vehicles show increasingly risky interaction in the second half of the clip.",
  "supporting_evidence": [
    "relative motion changes sharply",
    "close interaction appears in later chunks"
  ]
}
```

---

# 8. 单窗口推理输出定义

建议保存为：

```python
window_result = {
    "video_id": "...",
    "window_id": k,
    "start_frame": start_k,
    "end_frame": end_k - 1,
    "stage1_output": {...},
    "stage2_output": {
        "is_anomaly": True,
        "overall_score": 0.81,
        "chunk_scores": [0.08, 0.22, 0.74, 0.86],
        "anomaly_type": "dangerous_interaction",
        "abnormal_chunks": [3, 4],
        "confidence": 0.77,
        "short_reason": "...",
        "supporting_evidence": [...]
    }
}
```

---

# 9. 帧级回写（核心后处理）

这是这个 baseline 最关键的环节。

## 9.1 基本思想

模型输出的是窗口级和 chunk 级分数。  
为了得到整段视频的异常区间，需要把 chunk_scores 映射回原始帧。

---

## 9.2 chunk 到原始帧的映射

对于第 k 个窗口：

- 窗口起点：`start_k`
- chunk_scores：`[q1, q2, q3, q4]`

映射关系：

- `q1` -> 帧 `[start_k + 0,  start_k + 19]`
- `q2` -> 帧 `[start_k + 20, start_k + 39]`
- `q3` -> 帧 `[start_k + 40, start_k + 59]`
- `q4` -> 帧 `[start_k + 60, start_k + 79]`

---

## 9.3 多窗口重叠时的融合

由于窗口步长 20，小于窗口长度 80，所以一个帧通常会被多个窗口覆盖。

建议第一版采用：

```python
frame_score[t] = max(all chunk scores written to frame t)
```

即：

\[
a_t = \max q_{k,j}
\]

理由：

- 异常往往只占少量局部
- max pooling 不容易把异常淹没
- 实现简单
- 对 baseline 足够稳

---

## 9.4 伪代码

```python
frame_scores = np.zeros(num_frames, dtype=np.float32)

for result in window_results:
    start = result["start_frame"]
    q = result["stage2_output"]["chunk_scores"]  # [q1, q2, q3, q4]

    for chunk_id in range(4):
        chunk_start = start + chunk_id * 20
        chunk_end = chunk_start + 20
        frame_scores[chunk_start:chunk_end] = np.maximum(
            frame_scores[chunk_start:chunk_end],
            q[chunk_id]
        )
```

---

# 10. 帧级曲线平滑

回写后得到原始帧分数序列：

```python
frame_scores.shape == (T,)
```

但它会比较块状，所以需要平滑。

## 10.1 推荐方式
第一版建议使用：

### 方案 A：简单移动平均
```python
smoothed_scores = moving_average(frame_scores, window=9)
```

### 方案 B：Savitzky-Golay
```python
smoothed_scores = savgol_filter(frame_scores, window_length=11, polyorder=2)
```

更推荐 **Savitzky-Golay**，因为它对局部峰值保留更好。

---

# 11. 异常区间提取

## 11.1 阈值化

设阈值：

```python
THRESHOLD = 0.5
```

则异常帧集合为：

```python
anomaly_mask = smoothed_scores > THRESHOLD
```

---

## 11.2 连通段提取

把连续为 True 的部分提成区间：

```python
intervals = [
    (start_1, end_1),
    (start_2, end_2),
    ...
]
```

---

## 11.3 后处理规则

建议加入以下规则：

### 最短长度过滤
```python
MIN_INTERVAL_LEN = 10  # 帧
```

### 邻近区间合并
若两个区间间距小于：

```python
MERGE_GAP = 8  # 帧
```

则合并。

### 边缘孤立峰去除
过短且低置信的单峰区间去掉。

---

# 12. 视频级分数定义

## 12.1 为什么不能简单平均
如果整段视频只有很短一段异常，平均值会被大量正常片段稀释。

---

## 12.2 推荐定义

### 方案 A：最大值
```python
video_score = smoothed_scores.max()
```

### 方案 B：Top-K 均值
```python
k = max(1, int(0.1 * len(smoothed_scores)))
video_score = np.mean(np.sort(smoothed_scores)[-k:])
```

更建议你同时保存两个值：

- `video_score_max`
- `video_score_top10`

正式视频级分类基准建议优先用 `max`。

---

# 13. 输出文件格式建议

每个视频输出一个 JSON：

```json
{
  "video_id": "01_Accident_001",
  "fps": 25,
  "window_size": 80,
  "window_stride": 20,
  "chunks_per_window": 4,
  "frames_per_chunk": 20,
  "sampled_frames_per_chunk": 4,
  "window_results": [
    {
      "window_id": 0,
      "start_frame": 0,
      "end_frame": 79,
      "stage1_output": {
        "main_objects": ["car", "truck"],
        "scene_summary": "...",
        "chunk_descriptions": ["...", "...", "...", "..."],
        "noticeable_change": true,
        "likely_abnormal_chunks": [3, 4],
        "risk_hint": "possibly_risky"
      },
      "stage2_output": {
        "is_anomaly": true,
        "overall_score": 0.81,
        "chunk_scores": [0.08, 0.22, 0.74, 0.86],
        "anomaly_type": "dangerous_interaction",
        "abnormal_chunks": [3, 4],
        "confidence": 0.77,
        "short_reason": "...",
        "supporting_evidence": ["...", "..."]
      }
    }
  ],
  "frame_scores_raw": [],
  "frame_scores_smooth": [],
  "predicted_intervals": [
    [48, 143]
  ],
  "video_score_max": 0.86,
  "video_score_top10": 0.72
}
```

---

# 14. 建议的代码模块拆分

```text
src/
  data/
    load_tad_frames.py
    window_sampler.py
  vlm/
    model_loader.py
    prompt_stage1.py
    prompt_stage2.py
    run_stage1.py
    run_stage2.py
    parser.py
  postprocess/
    writeback.py
    smooth.py
    intervals.py
    video_score.py
  pipeline/
    run_pure_vlm_baseline.py
```

---

# 15. 推荐配置文件

```yaml
working_fps: 25

window:
  size: 80
  stride: 20
  chunks: 4
  chunk_size: 20
  sampled_frames_per_chunk: 4

vlm:
  model_name_or_path: "Qwen/Qwen3-VL"
  max_new_tokens: 512
  temperature: 0.0
  do_sample: false

postprocess:
  smoothing: "savgol"
  savgol_window: 11
  savgol_polyorder: 2
  threshold: 0.5
  min_interval_len: 10
  merge_gap: 8
  video_score_mode: "max"
```

---

# 16. 端到端伪代码

```python
def run_pure_vlm_baseline(frames, model, processor, cfg):
    windows = build_windows(
        frames,
        window_size=cfg.window.size,
        stride=cfg.window.stride
    )

    window_results = []

    for window_id, window_frames in enumerate(windows):
        chunks = split_into_chunks(window_frames, num_chunks=4)
        sampled_frames = []

        for chunk in chunks:
            sampled_frames.extend(sample_4_frames(chunk))

        # Stage 1
        prompt1 = build_stage1_prompt()
        stage1_raw = run_vlm(sampled_frames, prompt1, model, processor)
        stage1_json = parse_stage1(stage1_raw)

        # Stage 2
        prompt2 = build_stage2_prompt(stage1_json)
        stage2_raw = run_vlm(sampled_frames, prompt2, model, processor)
        stage2_json = parse_stage2(stage2_raw)

        window_results.append({
            "window_id": window_id,
            "start_frame": window_id * cfg.window.stride,
            "end_frame": window_id * cfg.window.stride + cfg.window.size - 1,
            "stage1_output": stage1_json,
            "stage2_output": stage2_json,
        })

    frame_scores = writeback_chunk_scores(window_results, num_frames=len(frames))
    smooth_scores = smooth_frame_scores(frame_scores, cfg.postprocess)
    intervals = extract_intervals(smooth_scores, cfg.postprocess)
    video_score = compute_video_score(smooth_scores, mode=cfg.postprocess.video_score_mode)

    return {
        "window_results": window_results,
        "frame_scores_raw": frame_scores,
        "frame_scores_smooth": smooth_scores,
        "predicted_intervals": intervals,
        "video_score": video_score,
    }
```

---

# 17. 最低可行版本（MVP）

如果你想尽快跑出一个 baseline，建议先做最简版本：

## 先实现
- 25 FPS 假设
- 80 帧滑窗，20 帧步长
- 4 chunk × 4 帧采样
- 两阶段 prompt
- chunk_scores 回写
- max video score
- 简单阈值切段

## 先不实现
- 更复杂的温度采样
- 多 prompt ensemble
- 多模型投票
- 自适应阈值
- 异常类型后校正

---

# 18. 实验建议

## 18.1 视频级基线
使用 TAD 自带 normal/abnormal 标签，计算：
- ROC-AUC
- PR-AUC
- Accuracy / F1（若做阈值分类）

## 18.2 区间定位基线
如果你还没有精标区间，不建议全量评估。
建议先手工标一个小子集，再评：
- 区间 IoU
- Start / End MAE
- chunk hit rate

---

# 19. 这个 baseline 的优点与局限

## 优点
- 纯 VLM，结构简单
- 与 EventVAD 的“短事件单元 + 两阶段推理”思想一致
- 便于与后续正式方法做对比
- 可以得到视频级分数和粗粒度异常区间

## 局限
- 没有显式目标级信息
- 对非常细微的小目标异常可能不稳
- 区间定位只能做到 chunk / 粗帧级
- 依赖 prompt 和输出解析的稳定性

---

# 20. 一句话总结

这个 baseline 的核心思想是：

> **用 25 FPS 的统一时间轴将整段视频切成 80 帧重叠窗口，再把每个窗口切成 4 个 chunk，用 Qwen 风格 VLM 对 16 张采样帧执行两阶段推理，输出 chunk 级异常分数，并通过帧级回写与后处理得到整段视频的异常曲线与异常区间。**

它足够纯、足够简单，也足够适合作为你后续正式方法的对照基线。
