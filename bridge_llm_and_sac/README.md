# MuJoCo Humanoid Hierarchical RL + LLM Spatial Navigation

中文 | [English](#english)

## 中文

### 项目定位

本仓库面向 MuJoCo 人形机器人分层强化学习与 LLM 空间导航任务，提供
LLM 高层任务规划与 SAC 导航控制之间的 bridge/advisor 训练代码。
项目负责运行状态记录、训练数据构建、advisor 训练、离线评估和在线 A/B 实验。

运行系统负责真实运行时的机器人状态、导航回调、空间记忆、拓扑地图和
SAC 导航执行。本项目只研究中间桥接层：如何在大语言模型的高层任务规划
和 SAC 的中层/底层导航控制之间建立更高密度的信息通道。

这个 bridge 不是低层控制器。SAC 仍然执行 `go_to(x, y)` 和局部导航；
bridge 学习的是任务执行监控和重规划触发：

- 当前导航 skill 是否应该继续
- 机器人是否正在卡住、绕路或低收益探索
- 视觉/语义事件是否需要触发扫描
- 候选子目标中哪个更值得交给 SAC
- 什么时候应该让 LLM 重新参与高层规划

### 核心想法

`BridgeNet-v1` 使用多模态编码和交叉注意力融合：

```text
map layers       -> CNN map tokens
robot state      -> MLP state token
TaskSpec ids     -> task token
memory/topology  -> MLP memory token
candidate goals  -> MLP candidate tokens

query tokens:   [CLS, task, state, skill]
context tokens: [map tokens, memory token, candidate tokens]

cross-attention + transformer fusion
        -> GRU temporal head
        -> event / replan / outcome / candidate-score heads
```

它不是端到端控制器，而是一个导航过程监控和子目标决策模块。模型根据地图、
机器人状态、空间记忆和候选子目标预测事件、进展、失败风险和 skill gate；
SAC 保留快速闭环导航能力，LLM 保留高层任务拆解和语义规划能力。

当前运行时状态向量是 22 维：

- 前 16 维：机器人位姿、目标距离、任务/进度等即时特征
- 后 6 维：当前 `go_to` segment 的进度记忆，包括已运行 callback 数、起始距离、
  历史最佳距离、起点到当前的进展、相对最佳点的退化、最近窗口进展

这 6 个 segment 记忆特征是为了补足 LLM 与 SAC 状态交流密度不足的问题，让
bridge 能看到“这段导航是否真的在变好”。

### 文件结构

整理后的根目录只保留项目元信息和核心包：

```text
.
├── bridge/              # 核心模型、数据集、recorder、advisor、hook 与 arbitration
├── scripts/             # 数据生成、采集、训练、评估、A/B 和 smoke 命令入口
├── tests/               # pytest smoke tests
├── README.md
├── requirements.txt
└── .gitignore
```

之前根目录看起来有很多零散文件，是因为研究阶段把每个实验入口都放在根目录：
采集、合并、训练、评估、advisor smoke、A/B 实验、阈值扫描等。现在这些命令
统一收进 `scripts/`，仓库结构更像一个独立研究项目。

`data/`、`runs/`、checkpoint、日志和 `.npz` episode 都被 `.gitignore` 排除。
GitHub 只上传源码和文档，不上传本地训练数据或模型权重。

### 安装

```bash
pip install -r requirements.txt
```

如果系统 Python 没有 Torch，可以使用已有的 MuJoCo/训练环境：

```bash
export MUJOCO_PYTHON=/path/to/mujoco_env/bin/python
```

### 快速自检

生成一个小型合成数据集：

```bash
python -m scripts.make_synthetic_dataset \
  --output-dir data/synthetic \
  --episodes 32 \
  --steps 96
```

训练 3 个 epoch 做 smoke test：

```bash
python -m scripts.train \
  --data-dir data/synthetic \
  --epochs 3 \
  --batch-size 8
```

验证 episode 格式：

```bash
python -m scripts.validate_episode data/synthetic
```

前向和 loss smoke：

```bash
python -m scripts.smoke_forward
```

如果需要用 MuJoCo Python：

```bash
$MUJOCO_PYTHON -m scripts.smoke_forward
```

### 运行时数据采集

不修改外部导航运行系统核心逻辑的运行时采集：

```bash
$MUJOCO_PYTHON -m scripts.collect_runtime \
  --agent-root ../agent_system_complex_version \
  --output-dir data/runtime \
  --max-rounds 2 \
  --nav-steps-per-round 800 \
  --target-place 会议室
```

`collect_runtime` 只包装运行中的 `nav_skill.go_to` 实例，把现有
`step_callback` 同步记录成 bridge snapshot；退出时会恢复原方法。

批量采集：

```bash
$MUJOCO_PYTHON -m scripts.collect_runtime_batch \
  --output-dir data/runtime_v1 \
  --episodes 10 \
  --max-rounds 1 \
  --nav-steps-per-round 400
```

合并多个数据集：

```bash
$MUJOCO_PYTHON -m scripts.merge_episodes \
  --output-dir data/runtime_merged \
  --overwrite \
  data/runtime_v1 data/runtime_v2
```

### 训练

从 runtime 数据训练：

```bash
$MUJOCO_PYTHON -m scripts.train \
  --data-dir data/runtime_merged \
  --output-dir runs/runtime_bridge_v1 \
  --epochs 80 \
  --batch-size 8 \
  --sequence-len 16 \
  --hidden-dim 128 \
  --fusion-layers 2 \
  --num-heads 4 \
  --device cuda \
  --split-by episode \
  --val-fraction 0.2 \
  --balanced-class-loss \
  --patience 12 \
  --lr 0.0001
```

从已有 checkpoint 继续训练：

```bash
$MUJOCO_PYTHON -m scripts.train \
  --data-dir data/runtime_merged \
  --output-dir runs/runtime_bridge_finetune \
  --init-checkpoint runs/pretrain/best.pt \
  --epochs 80 \
  --device cuda
```

评估：

```bash
$MUJOCO_PYTHON -m scripts.evaluate \
  --checkpoint runs/runtime_bridge_v1/best.pt \
  --data-dir data/runtime_merged \
  --split val
```

### Advisor 和 A/B 实验

离线回放一个 episode：

```bash
$MUJOCO_PYTHON -m scripts.smoke_advisor \
  --checkpoint runs/runtime_bridge_v1/best.pt \
  --data-dir data/runtime_merged \
  --device cpu
```

在线采集时启用 advisor：

```bash
$MUJOCO_PYTHON -m scripts.collect_runtime \
  --output-dir data/runtime_advisor \
  --episode-id advisor_001 \
  --max-rounds 1 \
  --nav-steps-per-round 250 \
  --target-place 会议室 \
  --advisor-checkpoint runs/runtime_bridge_v1/best.pt \
  --advisor-device cpu \
  --advisor-control replan \
  --advisor-stop-confidence 0.92 \
  --advisor-stop-consecutive 2 \
  --advisor-warmup-steps 6
```

`--advisor-control off` 表示只观察不介入；`safe` 只允许目标/子目标完成类停止；
`risk` 只使用失败风险类停止；`replan` 允许 stuck、低信息增益和失败风险触发
当前 `go_to` 提前结束，让 frontier 逻辑重新选择子目标。

成对 A/B 实验：

```bash
$MUJOCO_PYTHON -m scripts.run_ab_experiment \
  --output-dir data/ab_runtime_v1 \
  --episodes 5 \
  --seed-base 12000 \
  --max-rounds 1 \
  --nav-steps-per-round 300 \
  --advisor-checkpoint runs/runtime_bridge_v1/best.pt \
  --advisor-device cpu
```

### Episode 格式

训练 episode 是压缩 `.npz`：

```text
data/<dataset_name>/episodes/episode_000001.npz
```

主要数组：

```text
maps: [T, C, H, W] float32
state: [T, state_dim] float32
memory: [T, memory_dim] float32
task: [T, 3] int64
skill: [T] int64
candidates: [T, K, candidate_dim] float32
candidate_mask: [T, K] bool
event: [T] int64
replan: [T] int64
success/stuck/target_found/cost/info_gain: [T] float32
candidate_score_target: [T, K] float32
```

### 当前研究结论

现有实验表明，单纯的 learned advisor 已经能在部分场景中减少无效导航步骤和
timeout，但解析式 progress guard 仍然能捕捉一些 learned model 漏掉的失败模式。

当前更稳妥的方向：

- 保留 hybrid control 作为安全 fallback
- 继续把 risk-head advisor 作为主要学习型中间层方向
- 增加 guard-triggered 和 non-triggered segment 数据，校准失败风险阈值
- 逐步用 learned risk 替换手写 progress reflex

### 开发命令

```bash
python -m py_compile bridge/*.py scripts/*.py tests/*.py
pytest
```

常用数据检查：

```bash
python -m scripts.summarize_episodes data/runtime_merged
python -m scripts.audit_splits --data-dir data/runtime_merged --seed-start 1 --seed-end 30
python -m scripts.sweep_risk_thresholds --data-dir data/runtime_merged
```

---

## English

### Project Scope

This repository contains the bridge/advisor training code for MuJoCo humanoid
hierarchical RL and LLM spatial navigation. It covers runtime state recording,
dataset construction, advisor training, offline evaluation and online A/B
experiments.

The runtime navigation system remains responsible for robot state, navigation
callbacks, spatial memory, topological maps and SAC navigation.
This project focuses only on the middle bridge layer: building a denser
information channel between high-level LLM task planning and SAC-based
navigation execution.

The bridge is not a low-level controller. SAC still executes `go_to(x, y)` and
local navigation. The bridge learns task-level monitoring and replanning gates:

- whether the current navigation skill should continue
- whether navigation is stuck, regressing or low-value
- whether a visual/semantic event should trigger scanning
- which candidate subgoal should be passed to SAC
- when the LLM should be asked for high-level replanning

### Core Idea

`BridgeNet-v1` uses modality-specific encoders and cross-attention fusion:

```text
map layers       -> CNN map tokens
robot state      -> MLP state token
TaskSpec ids     -> task token
memory/topology  -> MLP memory token
candidate goals  -> MLP candidate tokens

query tokens:   [CLS, task, state, skill]
context tokens: [map tokens, memory token, candidate tokens]

cross-attention + transformer fusion
        -> GRU temporal head
        -> event / replan / outcome / candidate-score heads
```

The design is not an end-to-end controller. It is a navigation monitoring and
subgoal decision module. Given maps, robot state, spatial memory and candidate
subgoals, the model predicts events, progress, failure risk and skill gates.
SAC keeps fast closed-loop navigation, while the LLM keeps high-level task
decomposition and semantic planning.

The current runtime state vector is 22-D:

- first 16 dimensions: instantaneous pose, task and progress features
- last 6 dimensions: active `go_to` segment memory, including elapsed callbacks,
  start distance, best distance, progress from start, regret from best distance
  and recent-window progress

These segment-memory features address the sparse communication problem between
LLM-level planning and SAC-level execution.

### Repository Layout

```text
.
├── bridge/              # Core model, dataset, recorder, advisor, hooks, arbitration
├── scripts/             # CLI tools for data, training, evaluation, smoke tests and A/B
├── tests/               # pytest smoke tests
├── README.md
├── requirements.txt
└── .gitignore
```

The project originally had many root-level scripts because each experiment had
its own command entry point. They are now grouped under `scripts/` so the root
stays focused on package code and project metadata.

`data/`, `runs/`, checkpoints, logs and `.npz` episodes are ignored by Git.
Only source code and documentation are uploaded.

### Installation

```bash
pip install -r requirements.txt
```

If the system Python does not include Torch, use the MuJoCo/training Python:

```bash
export MUJOCO_PYTHON=/path/to/mujoco_env/bin/python
```

### Quick Smoke Test

```bash
python -m scripts.make_synthetic_dataset \
  --output-dir data/synthetic \
  --episodes 32 \
  --steps 96

python -m scripts.train \
  --data-dir data/synthetic \
  --epochs 3 \
  --batch-size 8

python -m scripts.validate_episode data/synthetic
python -m scripts.smoke_forward
```

With MuJoCo Python:

```bash
$MUJOCO_PYTHON -m scripts.smoke_forward
```

### Runtime Collection

Collect runtime bridge episodes without modifying the external navigation
runtime:

```bash
$MUJOCO_PYTHON -m scripts.collect_runtime \
  --agent-root ../agent_system_complex_version \
  --output-dir data/runtime \
  --max-rounds 2 \
  --nav-steps-per-round 800 \
  --target-place 会议室
```

`collect_runtime` wraps only the live `nav_skill.go_to` instance and records
bridge snapshots through the existing callback path. The method is restored
when the run exits.

Batch collection:

```bash
$MUJOCO_PYTHON -m scripts.collect_runtime_batch \
  --output-dir data/runtime_v1 \
  --episodes 10 \
  --max-rounds 1 \
  --nav-steps-per-round 400
```

Merge datasets:

```bash
$MUJOCO_PYTHON -m scripts.merge_episodes \
  --output-dir data/runtime_merged \
  --overwrite \
  data/runtime_v1 data/runtime_v2
```

### Training

```bash
$MUJOCO_PYTHON -m scripts.train \
  --data-dir data/runtime_merged \
  --output-dir runs/runtime_bridge_v1 \
  --epochs 80 \
  --batch-size 8 \
  --sequence-len 16 \
  --hidden-dim 128 \
  --fusion-layers 2 \
  --num-heads 4 \
  --device cuda \
  --split-by episode \
  --val-fraction 0.2 \
  --balanced-class-loss \
  --patience 12 \
  --lr 0.0001
```

Continue from a checkpoint:

```bash
$MUJOCO_PYTHON -m scripts.train \
  --data-dir data/runtime_merged \
  --output-dir runs/runtime_bridge_finetune \
  --init-checkpoint runs/pretrain/best.pt \
  --epochs 80 \
  --device cuda
```

Evaluate:

```bash
$MUJOCO_PYTHON -m scripts.evaluate \
  --checkpoint runs/runtime_bridge_v1/best.pt \
  --data-dir data/runtime_merged \
  --split val
```

### Advisor and A/B

Offline replay:

```bash
$MUJOCO_PYTHON -m scripts.smoke_advisor \
  --checkpoint runs/runtime_bridge_v1/best.pt \
  --data-dir data/runtime_merged \
  --device cpu
```

Online advisor control during runtime collection:

```bash
$MUJOCO_PYTHON -m scripts.collect_runtime \
  --output-dir data/runtime_advisor \
  --episode-id advisor_001 \
  --max-rounds 1 \
  --nav-steps-per-round 250 \
  --target-place 会议室 \
  --advisor-checkpoint runs/runtime_bridge_v1/best.pt \
  --advisor-device cpu \
  --advisor-control replan \
  --advisor-stop-confidence 0.92 \
  --advisor-stop-consecutive 2 \
  --advisor-warmup-steps 6
```

Paired A/B experiment:

```bash
$MUJOCO_PYTHON -m scripts.run_ab_experiment \
  --output-dir data/ab_runtime_v1 \
  --episodes 5 \
  --seed-base 12000 \
  --max-rounds 1 \
  --nav-steps-per-round 300 \
  --advisor-checkpoint runs/runtime_bridge_v1/best.pt \
  --advisor-device cpu
```

### Episode Format

Episodes are compressed `.npz` files:

```text
data/<dataset_name>/episodes/episode_000001.npz
```

Main arrays:

```text
maps: [T, C, H, W] float32
state: [T, state_dim] float32
memory: [T, memory_dim] float32
task: [T, 3] int64
skill: [T] int64
candidates: [T, K, candidate_dim] float32
candidate_mask: [T, K] bool
event: [T] int64
replan: [T] int64
success/stuck/target_found/cost/info_gain: [T] float32
candidate_score_target: [T, K] float32
```

### Current Research Takeaway

The learned advisor can already reduce some invalid navigation steps and
timeouts. The analytic progress guard is still useful because it catches failure
modes missed by the learned model.

Near-term direction:

- keep hybrid control as the robust fallback
- continue treating the risk-head advisor as the main learned bridge direction
- collect more guard-triggered and non-triggered segments
- calibrate risk thresholds before replacing the analytic reflex

### Development

```bash
python -m py_compile bridge/*.py scripts/*.py tests/*.py
pytest
```

Useful dataset checks:

```bash
python -m scripts.summarize_episodes data/runtime_merged
python -m scripts.audit_splits --data-dir data/runtime_merged --seed-start 1 --seed-end 30
python -m scripts.sweep_risk_thresholds --data-dir data/runtime_merged
```
