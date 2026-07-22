# 两大类实验运行、续跑与评估手册

最后核对：2026-07-22 21:10（Asia/Shanghai）

本文用于在对话上下文被清空后重新接管实验。它记录实际脚本、依赖关系、当前状态、输出目录、
续跑方法和评估顺序。状态和 PID 是时间点快照；命令、目录和依赖关系才是长期有效信息。

本文把目前工作归为两大类：

1. **跨骨干复现与组件矩阵**：Moment-DETR、QD-DETR、CG-DETR、EaTR 的 plain/GMR/quality/
   dual/counter/HieA2M 训练，包括已经完成、停止和仍在运行的轨迹；
2. **两级判空与学习式选框消融**：以 strict Moment-HieA2M 为父模型，依次运行几何去重、独立
   判空、门限校准、pairwise same-event、软计数和谨慎边界融合。

## 0. 清空上下文后的第一分钟

### 0.1 2026-07-23 持久后台接力

为允许关闭 Codex/SSH 终端，已创建 7 个用户级 systemd transient services；服务器已确认
`Linger=yes`。服务名统一为 `gmr-handoff-*`，分别保护 QD-Counter、canonical QD、canonical
CG、selector main/posw1/seed2025 和 EaTR-DGQC。

```bash
systemctl --user list-units 'gmr-handoff-*' --no-pager
journalctl --user -u gmr-handoff-eatr-dgqc.service -f
tail -f artifacts/persistent_handoff/eatr-dgqc.log
```

接力器先等待原 Codex PID 退出。若原训练进程脱离终端后仍存活，则不重复启动；若被 SSH/Codex
会话清理，则从 checkpoint 恢复。QD/CG/EaTR 使用 optimizer/scheduler exact resume；Moment
selector 使用该阶段已有 best checkpoint恢复后续依赖阶段。实现脚本：

```text
scripts/wait_then_persist_job.sh
scripts/resume_canonical_b128_exact.sh
scripts/recover_selector_branch.sh
scripts/run_eatr_dgqc_transfer.sh resume_full
```

这些是 transient systemd units：能跨终端和 SSH 注销继续运行，但服务器重启后需按本手册重新
创建。不要在服务仍 active 时手工重复运行相同输出目录。

先进入仓库，不要立即重复启动脚本：

```bash
cd /home/guoxiangyu/generalized-moment-retrieval
date '+%F %T %Z'
nvidia-smi --query-gpu=index,memory.used,memory.total,utilization.gpu \
  --format=csv,noheader,nounits
nvidia-smi --query-compute-apps=pid,gpu_uuid,used_memory,process_name \
  --format=csv,noheader
```

对 `nvidia-smi` 给出的每个 PID 查看完整命令：

```bash
ps -p PID -o pid,ppid,lstart,etimes,stat,pcpu,pmem,args
```

然后检查两类实验日志：

```bash
tail -n 2 artifacts/formal/qd_detr/seed2023/qd_dual/train_log.jsonl
tail -n 2 artifacts/formal/qd_detr/seed2023/qd_counter/train_log.jsonl
tail -n 2 artifacts/formal/qd_detr/seed2023/qd_hiea2m/train_log.jsonl

for root in seed2023 seed2024 seed2025 seed2023_posw1 seed2023_posw4; do
  echo "===== $root"
  tr '\r' '\n' \
    < "artifacts/validation_selector_ablation/$root/stage2_zero/stdout.log" \
    | tail -n 12
done
```

接管原则：

- 当前训练采用普通前台进程/受控执行会话，不使用 `nohup` 或 `screen`；
- 所有当前训练的 train/eval batch size 都是 128；
- 发现同一输出目录已有活跃进程时，绝不重复启动；
- 不删除、覆盖或 `git reset` 当前工作树，已有未提交代码和产物都属于实验的一部分；
- 当前只使用 validation 做训练选择和消融，新的 blind test 仍封存；
- legacy-loss、strict-loss、canonical plain、weights-only b128 restart 必须分开报告。

## 1. 共同数据、特征与评估口径

仓库与数据：

```text
/home/guoxiangyu/generalized-moment-retrieval
data/label/Standard/{train,val,test}.jsonl
Soccer-GMR/feature/standard/{clip,slowfast,clip_text}
```

共同设置：

- seed 主线为 2023；新增 selector head 另做 2024、2025 初始化稳定性实验；
- `mAP/mR/mIoU` 只在 positive queries 上统计；
- `AUROC/Rej-F1/G-mIoU` 在完整 positive/null 集合上统计；
- release-compatible 轨保留固定文本布局并使用 2 秒 rounding；
- `--mask-null-vmr-loss` 表示 paper-literal strict null supervision；缺少该参数的 mixed GMR 旧轨
  只能作为 exploratory/legacy 诊断；
- 训练每个 epoch 已自动运行 validation，并保存 best-mAP、best-G-mIoU@3 和 best-joint 产物。

## 2. 第一大类：跨骨干复现与组件训练矩阵

### 2.1 这类实验在回答什么

这类实验先建立各 backbone 的 plain 定位基线，再增加 GMR existence adapter，之后逐项比较：

- `quality`：query 的 IoU-quality 排序；
- `dual` / `phrase`：视频—文本双路时间语义交互；
- `counter`：existence 与正样本条件计数；
- `hiea2m`：把上述组件组合起来。

三套协议不能混写：

1. `artifacts/formal/`：较早的 staged/exploratory 轨；当前 QD 组件进程未传
   `--mask-null-vmr-loss`，因此属于 legacy loss semantic；
2. `artifacts/canonical_b128_restart/`：从原 canonical b32/b64 权重启动的 bsz=128
   **weights-only restart**，不是原训练的 exact resume；
3. `artifacts/formal_strict/`：显式 strict null-loss masking 的 Moment 配对实验。

### 2.2 时间点状态快照

#### 当前仍在运行

| 实验 | PID（快照） | GPU | 最新完整 epoch（约） | 输出目录 | 启动入口 |
|---|---:|---:|---:|---|---|
| QD Dual | 1501042 | 0 | 59 | `artifacts/formal/qd_detr/seed2023/qd_dual` | `start_qd_formal_variant_b128.sh` |
| QD Counter | 1501297 | 0 | 57 | `artifacts/formal/qd_detr/seed2023/qd_counter` | 同上 |
| QD HieA2M | 1501548 | 0 | 63 | `artifacts/formal/qd_detr/seed2023/qd_hiea2m` | 同上 |
| canonical QD b128 restart | 1504620 | 0 | 83 | `artifacts/canonical_b128_restart/qd_detr/seed2023/qd_detr` | `start_b128_restart.sh` |
| canonical CG b128 restart | 1504930 | 0 | 67 | `artifacts/canonical_b128_restart/cg_detr/seed2023/cg_detr` | 同上 |
| canonical EaTR b128 restart | 1505718 | 1 | 117 | `artifacts/canonical_b128_restart/eatr/seed2023/eatr` | 同上 |

PID 只用于核对本次会话；机器或进程重启后必须重新用 `nvidia-smi` 和 `ps` 获取。

#### 已完成或已有可用最终产物

| 实验 | 结论/用途 | 主要目录 |
|---|---|---|
| Moment-DETR-GMR release anchor | 已复现公开 test anchor；后续新候选仍不得读取 blind test | `artifacts/anchors/moment_detr_gmr_release*` |
| staged QD plain | 已有 best checkpoint，研究记录中的 best 为 `mAP 7.32 / G@3 2.60` | `artifacts/formal/qd_detr/seed2023/qd_detr` |
| staged EaTR plain | 已有 best checkpoint，研究记录中的 best 为 `7.92 / 2.99` | `artifacts/formal/eatr/seed2023/eatr` |
| strict Moment GMR v2 | 已结束，是 strict 配对控制 | `artifacts/formal_strict/moment_detr/seed2023/md_gmr_b128_rerun_from_best_v2` |
| strict Moment HieA2M v2 | 已结束，是第二大类实验的父模型 | `artifacts/formal_strict/moment_detr/seed2023/md_hiea2m_b128_rerun_from_best_v2` |

#### 已运行但当前不在进程表中

这些目录都有 checkpoint、validation 预测和日志，但不能只凭文件存在就称为正式完成：

| 实验 | 最后日志 epoch（快照） | 目录 | 处理方式 |
|---|---:|---|---|
| QD-GMR | 71 | `artifacts/formal/qd_detr/seed2023/qd_detr_gmr` | legacy exploratory；需要时 exact resume |
| QD Quality | 54 | `artifacts/formal/qd_detr/seed2023/qd_quality` | legacy exploratory；需要时 exact resume |
| formal CG plain | 132 | `artifacts/formal/cg_detr/seed2023/cg_detr` | 旧 formal 协议，不与 canonical 混表 |
| EaTR-GMR | 78 | `artifacts/formal/eatr/seed2023/eatr_gmr` | legacy exploratory；需要时 exact resume |
| QD invalid all-LR | 14 左右 | `artifacts/formal/qd_detr/seed2023/qd_detr_gmr_invalid_all_lr` | 已判无效，禁止进入结果矩阵 |

### 2.3 第一类脚本索引

| 脚本 | 功能 | 是否可直接重跑 |
|---|---|---|
| `scripts/start_qd_formal_variant_b128.sh` | exact resume `qd_quality/qd_dual/qd_counter/qd_hiea2m` 的 `latest.ckpt` | 仅在对应目录没有活跃进程时 |
| `scripts/start_qd_gmr_b128.sh` | exact resume QD-GMR | 同上 |
| `scripts/start_resumed_b128_formal.sh` | 同时恢复 QD-GMR 和 EaTR-GMR；子进程后台、父脚本前台等待 | 只有两者都确实需要恢复时 |
| `scripts/start_b128_restart.sh` | 从旧 canonical/formal 权重新建 b128 weights-only restart | **只用于首次建轨，不用于当前 restart 目录的二次恢复** |
| `scripts/start_moment_strict_b128.sh` | 从公开 Moment checkpoint 启动 strict GMR/HieA2M | 原始 strict 轨已跑过，不要覆盖 |
| `scripts/rerun_strict_moment_b128_from_best.sh` | 从中断 strict best 权重新建独立 weights-only rerun | v2 已完成；除非明确新建 suffix，否则不要重跑 |
| `scripts/launch_nohup_job.sh` | 历史 nohup 包装器 | 当前运行约定下不使用 |

当前三个 QD 组件的启动/恢复命令：

```bash
bash scripts/start_qd_formal_variant_b128.sh qd_dual 0
bash scripts/start_qd_formal_variant_b128.sh qd_counter 0
bash scripts/start_qd_formal_variant_b128.sh qd_hiea2m 0
```

如要恢复当前未运行的 QD 旧轨：

```bash
bash scripts/start_qd_gmr_b128.sh
bash scripts/start_qd_formal_variant_b128.sh qd_quality 0
```

不要在 GPU0 已有五个大模型任务时机械地追加它们；先确认显存和整体吞吐。

### 2.4 canonical b128 restart 的恢复规则

首次从原 canonical 权重建立新轨时使用：

```bash
bash scripts/start_b128_restart.sh canonical_qd GPU_ID
bash scripts/start_b128_restart.sh canonical_cg GPU_ID
bash scripts/start_b128_restart.sh canonical_eatr GPU_ID
```

这个脚本使用 `--init_checkpoint`，因此只能描述为 weights-only restart。如果
`artifacts/canonical_b128_restart/...` 已经训练过又发生中断，不能再次调用它从旧 b32/b64 权重
开头；应按当前进程完整 argv，把最后的 `--init_checkpoint OLD_PATH` 改为：

```text
QD:   --resume artifacts/canonical_b128_restart/qd_detr/seed2023/qd_detr/latest.ckpt
CG:   --resume artifacts/canonical_b128_restart/cg_detr/seed2023/cg_detr/latest.ckpt
EaTR: --resume artifacts/canonical_b128_restart/eatr/seed2023/eatr/last.pt
```

恢复前还应核对目录内 `config.json` 或 `run.json`，确保 batch size、学习率、特征顺序、
loss semantic 与 checkpoint 一致。QD/CG 是 `SlowFast -> CLIP`；EaTR 通过两个独立参数传入。

### 2.5 第一类训练日志与完成判据

QD/CG 目录通常包含：

```text
latest.ckpt
best_map.ckpt
best_g_miou3.ckpt
best_joint.ckpt
train_log.jsonl
latest_val_metrics.json
best_*_val_metrics.json
optimizer_groups.json
initialization_audit.json        # warm-start 轨
```

EaTR 使用 `last.pt`、`best_map.pt`、`best_gmiou3.pt`、`best_joint.pt`，其余日志语义相同。

判断训练真正运行：

1. PID 在进程表中且状态不是僵尸；
2. `train_log.jsonl` 的 epoch 持续增加；
3. checkpoint/metrics 修改时间更新；
4. GPU 上存在对应 PID；
5. warm-start 组件轨存在 `optimizer_groups.json` 与 `initialization_audit.json`。

判断完成不能只看 `best.*` 是否存在；best checkpoint 在第一个 epoch 后就可能生成。应结合进程已
退出、日志最后 epoch、early-stop/完成消息和 trainer 配置判断。

### 2.6 第一类 validation 复评命令

训练期间每个 epoch 已自动评估 validation。需要独立复评时，只评 val，不要误把 test 路径代入。

QD：

```bash
python -m methods.qd_detr_gmr.evaluate \
  --checkpoint artifacts/formal/qd_detr/seed2023/qd_hiea2m/best.ckpt \
  --eval_annotation data/label/Standard/val.jsonl \
  --output_dir artifacts/eval/qd_hiea2m_val
```

CG：

```bash
python -m methods.cg_detr_gmr.evaluate \
  --checkpoint artifacts/canonical_b128_restart/cg_detr/seed2023/cg_detr/best_map.ckpt \
  --eval_annotation data/label/Standard/val.jsonl \
  --output_dir artifacts/eval/canonical_cg_b128_val
```

EaTR：

```bash
python -m methods.eatr_gmr.evaluate \
  --checkpoint artifacts/canonical_b128_restart/eatr/seed2023/eatr/best_map.pt \
  --annotations data/label/Standard/val.jsonl \
  --slowfast-dir Soccer-GMR/feature/standard/slowfast \
  --clip-dir Soccer-GMR/feature/standard/clip \
  --text-dir Soccer-GMR/feature/standard/clip_text \
  --output-dir artifacts/eval/canonical_eatr_b128_val
```

Moment strict validation：

```bash
/home/guoxiangyu/miniconda3/bin/python -u training/moment_detr_gmr/evaluate.py \
  --model_path artifacts/formal_strict/moment_detr/seed2023/md_hiea2m_b128_rerun_from_best_v2/best_joint.ckpt \
  --split val \
  --eval_path data/label/Standard/val.jsonl \
  --t_feat_dir Soccer-GMR/feature/standard/clip_text \
  --v_feat_dirs Soccer-GMR/feature/standard/clip Soccer-GMR/feature/standard/slowfast \
  --results_dir artifacts/eval/strict_moment_hiea2m_v2_val \
  --device cuda
```

命令参数是否有变动时先运行相应模块的 `--help`；不要根据旧文档猜缩写参数。

### 2.7 EaTR DGQC 跨骨干迁移实验

2026-07-22 22:51 新增一条 strict、matched 的 EaTR 迁移流水线。它用来检验 Moment 上的
HieA2M-DGQC 收益能否迁移到结构不同的 event-aware EaTR。

冻结父模型：

```text
artifacts/eatr_dgqc_transfer/seed2023/frozen_parent/eatr_plain_b128_best_map.pt
SHA256 301c68139d32daa84421a345b227bf74195ee5940111b5ecb388bf1824dcf3bc
validation snapshot: mAP 8.73 / G-mIoU@3 3.51 / mR@5 12.85 / mR+@5 1.37
```

完整执行顺序：

```text
frozen EaTR plain
  -> eatr_gmr_strict
  -> 并行 eatr_quality / eatr_dual / eatr_counter / eatr_hiea2m(DGQC)
```

启动脚本：

```bash
bash scripts/run_eatr_dgqc_transfer.sh full 1
```

也可分别执行 `gmr|quality|dual|counter|hiea2m|children`。所有分支使用 bsz=128、strict null
masking、seed 2023、新模块学习率 `3e-5`、共享 parent 学习率 `3e-6`。四个 child 必须等待 strict
GMR 的 `best.pt`，并以该 GMR 的 best validation 数字作为 matched joint reference。

结果根目录：

```text
artifacts/eatr_dgqc_transfer/seed2023/
```

若 `eatr_hiea2m` 相对匹配的 `eatr_gmr_strict` 同时提升 mAP、G-mIoU@3，并且不降低 multi
acceptance/mR+，它构成跨 DETR backbone 的迁移证据；因为 Moment 与 EaTR 仍使用同一
Soccer-GMR 数据集，所以它不是跨数据集泛化证明。确认信号后还需至少三个完整训练 seed。

## 3. 第二大类：两级判空、学习式去重与软计数选框

### 3.1 算法与阶段依赖

这条路线的完整顺序是：

```text
strict Moment-HieA2M v2 checkpoint
  -> Stage 1: 同候选、同 K 的几何去重基线（无需训练）
  -> Stage 2: 只训练独立 zero verifier
  -> Stage 2 calibration: validation 上校准宽松 gate 与 zero veto
  -> Stage 3: 冻结其他模块，只训练 pairwise same-event head
  -> Stage 4: learned fixed Top-K 与 learned soft-count 网格
  -> Stage 5: complete-link 聚组及谨慎边界融合
```

核心依赖：

- Stage 1 独立，可直接用父模型 raw candidates；
- Stage 2 calibration 必须等待对应 Stage 2 的 validation prediction；
- Stage 3 必须加载同一分支 Stage 2 的 `best.ckpt`；
- Stage 4/5 必须加载同一分支 Stage 3 的 `best.ckpt`；
- 不允许用未训练的 pairwise head 生成 Stage 3–5 结论。

详细算法和消融标准见：

```text
docs/two_stage_learned_selector_ablation_plan_2026-07-22.md
```

### 3.2 已完成的 Stage 1

已完成目录：

```text
artifacts/validation_selector_ablation/seed2023/stage1_geometry/
```

主要结果：

| 固定预算 | direct Top-K mAP | 最好几何方法 | 最好 mAP | 解释 |
|---|---:|---|---:|---|
| Top-3 | 7.57 | complete-link fusion, IoU 0.5 | 7.96 | 小幅有效，但 Top-1 略降 |
| Top-5 | 8.67 | Linear Soft-NMS, IoU 0.5 | 8.84 | 小幅有效 |
| predicted-count | 5.67 | 所有去重近似相同 | 5.67 | 主要瓶颈是计数，不是重复框 |

完整结果：

```text
artifacts/validation_selector_ablation/seed2023/stage1_geometry/dedup_ablation_summary.json
artifacts/validation_selector_ablation/seed2023/stage1_geometry/run.log
```

### 3.3 当前五条 Stage 2 并行分支

所有分支均从同一个 strict seed-2023 HieA2M 父模型初始化：

```text
artifacts/formal_strict/moment_detr/seed2023/
  md_hiea2m_b128_rerun_from_best_v2/best_joint.ckpt
```

| 分支 | selector seed | zero positive weight | PID（快照） | 输出根目录 | 自动后续 |
|---|---:|---:|---:|---|---|
| 主线 | 2023 | 2.0 | 1659161 | `artifacts/validation_selector_ablation/seed2023` | PID 接续器 |
| seed 稳定性 | 2024 | 2.0 | 1662472 | `artifacts/validation_selector_ablation/seed2024` | 完整 branch 脚本 |
| seed 稳定性 | 2025 | 2.0 | 1662765 | `artifacts/validation_selector_ablation/seed2025` | 完整 branch 脚本 |
| 权重消融 | 2023 | 1.0 | 1662771 | `artifacts/validation_selector_ablation/seed2023_posw1` | 完整 branch 脚本 |
| 权重消融 | 2023 | 4.0 | 1662758 | `artifacts/validation_selector_ablation/seed2023_posw4` | 完整 branch 脚本 |

重要解释：seed 2024/2025 只改变新 selector head 的初始化和数据随机性；冻结的父 backbone 仍是
seed-2023 strict HieA2M。因此这两条用于检查“新头训练稳定性”，不能冒充完整 backbone 的三 seed
复现。

截至快照，五条分支均已多次完成 train + validation epoch；每条都生成了
`gradient_audit.json`，只有 `zero_verifier_head` 的 69,399 个参数有梯度，冻结主干梯度为零。

### 3.4 第二类脚本索引

| 脚本 | 功能 |
|---|---|
| `scripts/ablate_temporal_dedup.py` | direct Top-K、Hard/DIoU/Soft-NMS、complete-link 的 Stage 1 离线消融 |
| `scripts/calibrate_two_stage_gate.py` | validation 上扫描 gate/zero/veto/localization 门限 |
| `scripts/ablate_learned_selector.py` | fixed learned Top-K、soft count、停止阈值和融合消融 |
| `scripts/run_learned_selector_stage.sh` | 单独执行 Stage 1、2、校准、3、4/5 |
| `scripts/run_learned_selector_sequence.sh` | 从 Stage 1 开始严格串行跑单一默认分支 |
| `scripts/run_learned_selector_branch.sh` | 从 Stage 2 开始跑一个独立 root，并自动接续校准、3、4/5 |
| `scripts/continue_learned_selector_after_pid.sh` | 等待已启动的 Stage 2 PID 结束，再接续校准、3、4/5 |

脚本采用前台执行，不创建 nohup 或 screen。训练与评估的 batch size 均在脚本中固定为 128。

### 3.5 从头启动一条完整 selector 分支

标准形式：

```bash
bash scripts/run_learned_selector_branch.sh RESULT_ROOT GPU_ID SEED ZERO_POSITIVE_WEIGHT
```

当前四条可复现命令：

```bash
bash scripts/run_learned_selector_branch.sh \
  artifacts/validation_selector_ablation/seed2024 1 2024 2.0

bash scripts/run_learned_selector_branch.sh \
  artifacts/validation_selector_ablation/seed2025 1 2025 2.0

bash scripts/run_learned_selector_branch.sh \
  artifacts/validation_selector_ablation/seed2023_posw1 1 2023 1.0

bash scripts/run_learned_selector_branch.sh \
  artifacts/validation_selector_ablation/seed2023_posw4 1 2023 4.0
```

这些命令只用于全新目录或确认要从 Stage 2 重跑的情况。已有 Stage 2 结果时不要再次运行 branch
脚本，否则会重复训练前置阶段。

### 3.6 按阶段恢复后续工作

先设置分支 root：

```bash
export SELECTOR_ROOT=artifacts/validation_selector_ablation/seed2023
```

逐阶段执行：

```bash
bash scripts/run_learned_selector_stage.sh stage2_zero 1
bash scripts/run_learned_selector_stage.sh stage2_calibrate 1
bash scripts/run_learned_selector_stage.sh stage3_pairwise 1
bash scripts/run_learned_selector_stage.sh stage4_5_selection 1
```

如果 Stage 2 已经在跑、希望它结束后自动接续：

```bash
bash scripts/continue_learned_selector_after_pid.sh \
  STAGE2_PID artifacts/validation_selector_ablation/seed2023 1
```

清空上下文后按“第一个缺失产物”恢复：

1. 有 Stage 2 PID：保持运行，不重复启动；检查是否已有接续器；
2. 无 PID但有 `stage2_zero/best.ckpt`，且没有 `stage2_gate_calibration.json`：运行
   `stage2_calibrate`；
3. 有 calibration、没有 `stage3_pairwise/best.ckpt`：运行 `stage3_pairwise`；
4. 有 Stage 3 best、没有 `stage4_5_selection/learned_selector_ablation_summary.json`：运行
   `stage4_5_selection`；
5. 所有产物齐全：只做汇总，不重新训练。

Moment 的 `train.py --resume` 在这些 head-only 阶段是 weights-only fine-tuning load，不是包含
optimizer/epoch 的 exact resume。若某个 Stage 2/3 在 epoch 中间中断，最清楚的做法是保留旧目录
作为 interrupted 证据，从最佳 checkpoint 新建带 suffix 的独立目录，而不是宣称无缝续训。

### 3.7 第二类日志、产物与完成判据

每条分支的 Stage 2：

```text
ROOT/stage2_zero/stdout.log
ROOT/stage2_zero/train.log
ROOT/stage2_zero/val.log
ROOT/stage2_zero/gradient_audit.json
ROOT/stage2_zero/initialization_audit.json
ROOT/stage2_zero/best.ckpt
ROOT/stage2_zero/best_soccer_gmr_val_preds.jsonl
```

后续关键产物：

```text
ROOT/stage2_gate_calibration.json
ROOT/stage2_gate_calibration.log
ROOT/stage3_pairwise/best.ckpt
ROOT/stage3_pairwise/gradient_audit.json
ROOT/stage3_pairwise/raw_val/
ROOT/stage4_5_selection/learned_selector_ablation_summary.json
ROOT/stage4_5_selection/run.log
```

Stage 2/3 的梯度审计必须满足：目标新头梯度非零，冻结模块梯度为零。Stage 4/5 是同一 raw
validation 前向后的离线网格，不需要为每个参数组合重复训练。

最终比较顺序固定为：

1. direct Top-3 vs learned same-event Top-3，同样输出数量；
2. learned fixed Top-3 vs learned soft-count；
3. soft-count 最佳配置 vs 谨慎融合；
4. 同时检查 mAP、mR@3、重复率、近邻不同事件 recall、G-mIoU 和正样本放行率；
5. 不能只凭 aggregate G-mIoU 采用可能发生 all-empty-like collapse 的配置。

## 4. 哪些事情不能自动做

- 不得把 validation 最优参数直接拿到 test 后继续搜索；
- 不得运行新的 blind test，除非候选完成三 seed、non-collapse、provenance 和 prereg 验收；
- 不得把 `formal/` legacy QD 组件结果标成 strict paper-literal 结果；
- 不得把 seed2024/2025 selector-head 稳定性称为完整 backbone 三 seed；
- 不得用 `best.ckpt` 的存在判断训练已结束；
- 不得在已有输出目录上使用会重新初始化的首次启动脚本；
- 不得因为 GPU 尚有显存就启动依赖于未生成 checkpoint 的 Stage 3/4。

## 5. 建议的后续汇总顺序

当当前进程全部结束后：

1. 汇总第一类每条轨迹的 best-mAP、best-G@3、best-joint、最后 epoch 和停止原因；
2. 把 `formal/legacy`、`canonical b128 restart`、`formal_strict` 分成三张表；
3. 汇总第二类五条分支的 Stage 2 gate/zero calibration，先比较正样本放行率和 veto 错杀；
4. 比较 Stage 3 direct Top-3 与 learned Top-3，决定 pairwise 去重是否值得保留；
5. 再看 soft count 和融合是否在不损害 localization 的情况下提供增益；
6. 更新本文的时间点状态，并同步：
   - `docs/detr_gmr_research_progress_2026-07-22.md`
   - `docs/two_stage_learned_selector_ablation_plan_2026-07-22.md`
   - `PROGRESS.md`

## 6. 最小接管阅读清单

新会话只需依次阅读：

1. 本文：运行状态、脚本、恢复和评估；
2. `docs/two_stage_learned_selector_ablation_plan_2026-07-22.md`：第二类算法与消融标准；
3. `docs/detr_gmr_lessons_and_algorithm_notes_2026-07-22.md`：失败经验和不可重复的坑；
4. `docs/detr_gmr_research_progress_2026-07-22.md`：历史结果和研究结论；
5. `plans/detr_hiea2m_research.md`：更长期的矩阵与 blind-test 约束。
