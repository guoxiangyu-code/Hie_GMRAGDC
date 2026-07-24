# GMR 全部实验清单与上下文恢复总账

更新时间：2026-07-23 16:48 CST  
工程目录：`/home/guoxiangyu/generalized-moment-retrieval`  
数据集：Soccer-GMR Standard validation，465 queries（255 positive / 210 null）

这份文档用于清空对话上下文后的完整接管。它同时覆盖正式训练、探索性训练、
组件消融、后处理消融、跨骨干迁移、smoke test、batch probe、失败/止损轨和
尚未实施的实验。不要只看某个目录存在 `best.ckpt` 就判断实验已完成；必须结合
本文状态、进程、日志末尾和 `runner.status`。

## 0. 新会话接管时先做什么

按以下顺序执行只读检查：

```bash
cd /home/guoxiangyu/generalized-moment-retrieval

ps -eo pid,ppid,sid,etimes,stat,pcpu,pmem,args \
  | rg 'resume_current_interval5|training\.flash_vtg_gmr|methods\.(qd_detr_gmr|cg_detr_gmr|eatr_gmr)\.train|training/moment_detr_gmr'

nvidia-smi

cat artifacts/supplementary_queue/seed2023/queue.status
cat artifacts/qd_fair_ablation/seed2023_bsz32/matrix.status

tail -n 3 artifacts/strict_bsz32/qd_detr/seed2023/qd_dual/train_log.jsonl
tail -n 20 artifacts/flash_vtg_supplement/seed2023_bsz128/flash_vtg_plain/stdout.log
tail -n 20 artifacts/flash_vtg_supplement/seed2023_bsz128/flash_vtg_gmr/stdout.log
```

当前后台协调器及子进程快照：

| 任务 | PID | 快照进度 | 状态 | 输出目录 |
|---|---:|---:|---|---|
| interval-5 coordinator | 2459127 | — | 后台存活 | `artifacts/supplementary_queue/seed2023/interval5_resume` |
| QD-DETR Dual | 2459133 | e84 | 运行中，每 5 epoch val | `artifacts/strict_bsz32/qd_detr/seed2023/qd_dual` |
| Flash-VTG plain | 2459152 | e144 | 运行中，bsz=128，每 5 epoch val | `artifacts/flash_vtg_supplement/seed2023_bsz128/flash_vtg_plain` |
| Flash-VTG GMR | 2459153 | e90+ | 运行中，bsz=128，每 5 epoch val | `artifacts/flash_vtg_supplement/seed2023_bsz128/flash_vtg_gmr` |

PID 会变化，日志是否继续增长比旧 PID 更可靠。当前三个训练均由 PPID 1 下的
协调器管理，关闭终端不会终止。

## 1. 状态标记和实验协议

### 1.1 状态标记

| 标记 | 含义 |
|---|---|
| `PRIMARY` | 当前可用于论文方法讨论的 validation 证据 |
| `COMPLETE` | 已完成并有训练、checkpoint、validation 产物 |
| `RUNNING` | 当前进程存活，日志持续增长 |
| `EXPLORATORY` | 做过且有诊断价值，但协议不满足最终公平比较 |
| `FAILED/STOP` | 已有充分失败证据或主动止损，禁止恢复占资源 |
| `INVALID` | 实验协议错误，结果不得进入任何效果表 |
| `SMOKE` | 只证明代码、梯度、显存或评估路径可运行 |
| `BLOCKED` | 想法已定义，但代码接口或 parent 尚未准备好 |
| `SEALED` | blind test 未授权执行，必须保持封存 |

### 1.2 三种历史协议不能混表

1. `artifacts/formal/`：早期 staged/legacy 轨，部分没有
   `--mask-null-vmr-loss`，只能作探索诊断。
2. `artifacts/canonical_b128_restart/`：从旧 canonical 权重开始的
   bsz=128 weights-only restart，不是 exact resume。
3. `artifacts/formal_strict/`、`artifacts/strict_bsz32/` 和后续 supplementary：
   显式 strict null masking，才适合作为 matched validation 对照。

指标口径也要分清：

- `mAP/mR/mIoU`：只在 positive queries 上统计；
- `AUROC/Rej-F1/G-mIoU`：在完整 positive/null validation 上统计；
- Flash 同一 checkpoint 同时保存 raw view 和 NMS 0.7 view，数字不能混写；
- 所有新方法当前只有 validation 证据，不是论文 blind-test 数字。

## 2. 方法组件字典

| 代码 | 独立组件 | 定义 |
|---|---|---|
| `B` | GMR / existence gate | 判断是否至少存在一个相关时刻 |
| `Q` | Quality head | 预测候选边界 IoU/质量并参与排序 |
| `D` | Dual Grounding | 文本—视频双向语义交互 |
| `D-Phrase` | Phrase Grounding | CG-DETR 上的 phrase/slot grounding 适配 |
| `C` | Hierarchical Counter | existence + positive-conditional `1/2/3/4+` |
| `Z` | Independent Zero Verifier | 独立预测 `P(N=0)`，不使用简单 `1-p_exist` |
| `P` | Pairwise Learned Dedup | 判断候选对是否属于同一真实事件 |
| `SC` | Soft-count prior | 计数只作为软数量先验，不硬截断 |
| `BF` | Boundary Fusion | 对高置信同事件簇进行谨慎边界融合 |

完整旧 HieA2M 不是单个组件：

```text
HieA2M-DGQC = B + Q + D(or Phrase) + C
```

当前论文候选：

```text
U   = B + Q + Z + P
U-D = B + Q + D + Z + P
```

Counter 与 Boundary Fusion 当前不进入默认组合。

## 3. Moment-DETR 全部实验

### 3.1 发布锚点与评估修复

| 实验 | 状态 | 目录 | 说明 |
|---|---|---|---|
| Moment-DETR-GMR release anchor | `COMPLETE` | `artifacts/anchors/moment_detr_gmr_release` | 公开发布 checkpoint/结果锚点 |
| release validation | `COMPLETE` | `artifacts/anchors/moment_detr_gmr_release_val` | 初始 validation 回放 |
| clean release validation | `COMPLETE` | `artifacts/anchors/moment_detr_gmr_release_val_clean` | 修复 NumPy/evaluator 后的干净回放 |

公开 anchor 只用于复现。新方法没有据此重复读取 blind test。

### 3.2 早期 strict 启动尝试

这些目录主要保存 nohup 启动元数据和中断证据，不是最终结果目录：

| Variant | 状态 | 目录 |
|---|---|---|
| `md_gmr` | `EXPLORATORY/INTERRUPTED` | `artifacts/strict/moment_detr/seed2023/md_gmr` |
| `md_gmr_b128` | `EXPLORATORY/INTERRUPTED` | `artifacts/strict/moment_detr/seed2023/md_gmr_b128` |
| `md_hiea2m` | `EXPLORATORY/INTERRUPTED` | `artifacts/strict/moment_detr/seed2023/md_hiea2m` |
| `md_hiea2m_b128` | `EXPLORATORY/INTERRUPTED` | `artifacts/strict/moment_detr/seed2023/md_hiea2m_b128` |

首次启动入口：

```bash
bash scripts/start_moment_strict_b128.sh md_gmr 0
bash scripts/start_moment_strict_b128.sh md_hiea2m 1
```

不要覆盖上述旧目录。

### 3.3 strict matched GMR/HieA2M 与两次独立重跑

| Variant | 状态 | 目录 | 用途 |
|---|---|---|---|
| GMR initial | `COMPLETE/历史` | `artifacts/formal_strict/moment_detr/seed2023/md_gmr_b128` | 初始 strict control |
| GMR rerun | `COMPLETE/历史` | `artifacts/formal_strict/moment_detr/seed2023/md_gmr_b128_rerun_from_best` | 从 best weights-only 重跑 |
| GMR rerun v2 | `COMPLETE` | `artifacts/formal_strict/moment_detr/seed2023/md_gmr_b128_rerun_from_best_v2` | 当前 strict matched control |
| HieA2M initial | `COMPLETE/历史` | `artifacts/formal_strict/moment_detr/seed2023/md_hiea2m_b128` | 初始完整 DGQC |
| HieA2M rerun | `COMPLETE/历史` | `artifacts/formal_strict/moment_detr/seed2023/md_hiea2m_b128_rerun_from_best` | 第一次补跑 |
| HieA2M rerun v2 | `COMPLETE` | `artifacts/formal_strict/moment_detr/seed2023/md_hiea2m_b128_rerun_from_best_v2` | selector/Zero/P 的统一父模型 |

重跑入口：

```bash
bash scripts/rerun_strict_moment_b128_from_best.sh \
  md_gmr 0 PARENT_CKPT NEW_SUFFIX

bash scripts/rerun_strict_moment_b128_from_best.sh \
  md_hiea2m 1 PARENT_CKPT NEW_SUFFIX
```

v2 已完成，除非明确建立新 suffix，不得重复执行。

### 3.4 release-parent 单组件与完整 HieA2M

| 实验 | 状态 | 目录 | 历史 joint-best mAP/G@3 | 判断 |
|---|---|---|---:|---|
| continued GMR control, lr=5e-6 | `COMPLETE` | `artifacts/runs/md_gmr_continue_lr5e6_seed2023` | 约 9.18/34.43（composition view） | 配对续训控制 |
| Quality | `COMPLETE/EXPLORATORY` | `artifacts/runs/md_quality_release_seed2023` | 8.52/35.05 | 定位排序方向有效 |
| DualGround | `COMPLETE/EXPLORATORY` | `artifacts/runs/md_dual_release_seed2023` | 8.32/34.57 | 单独收益较弱 |
| Counter | `COMPLETE/EXPLORATORY` | `artifacts/runs/md_counter_release_seed2023` | 8.58/35.09 | 聚合双升但存在拒答塌缩风险 |
| HieA2M seed2018 | `COMPLETE/DIAGNOSTIC` | `artifacts/runs/md_hiea2m_release_seed2018` | fused 9.70/35.31 | 未过 balanced-G gate |
| HieA2M seed2023 | `COMPLETE/DIAGNOSTIC` | `artifacts/runs/md_hiea2m_release_seed2023` | fused 10.34/36.90 | 未过 balanced-G gate |
| HieA2M seed2024 | `COMPLETE/DIAGNOSTIC` | `artifacts/runs/md_hiea2m_release_seed2024` | fused 9.55/35.01 | 未过 balanced-G gate |

三 seed 汇总：

```text
artifacts/runs/moment_hiea2m_fused_three_seed_summary.json
```

三 seed fused mean 为 mAP 9.86、G@3 35.74，但 balanced-G 约 38–40，
没有通过预注册的 non-collapse 门槛，因此不能进入 blind test。

### 3.5 objective-specific composition、校准和 bootstrap

以下都已做过，属于 Moment HieA2M 的诊断实验：

| 尝试 | 产物 | 结论 |
|---|---|---|
| best-map head + best-joint gate composition | 各 seed 的 `fused_map_joint_val*.json*` | 聚合 mAP/G@3 提升，但需严格 provenance |
| 固定 `tau=0.4` calibration | `md_hiea2m_release_seed2023/calibration_tau0.4.json` | 更强拒答会伤害 multi acceptance |
| vs release paired bootstrap | `fused_map_joint_bootstrap_vs_release.json` | seed2023 两主指标 CI 为正 |
| vs continued GMR bootstrap | `fused_bootstrap_vs_continued_gmr.json` | HieA2M 相对续训控制仍有增益 |
| group diagnostics | `fused_map_joint_group_diagnostics_tau0.4.json` | 揭示 all-empty-like/multi collapse 风险 |

脚本：

```text
scripts/fuse_gmr_heads.py
scripts/calibrate_hiea2m.py
scripts/bootstrap_gmr.py
scripts/diagnose_gmr_groups.py
scripts/summarize_seed_metrics.py
```

### 3.6 解码器实验

| 解码方式 | 状态 | 典型 mAP/G@3 | 结论 |
|---|---|---:|---|
| Full candidates | `COMPLETE` | 8.63/36.59 | 当前最平衡 |
| GREC threshold | `COMPLETE/DIAGNOSTIC` | 6.95/38.96 | G 上升但 mAP 明显下降 |
| HieA2G count-adaptive | `FAILED` | 4.23/38.96 | 硬计数截断严重破坏定位召回 |

### 3.7 Moment Q+D(no Counter)

| 实验 | 状态 | 目录 | AUROC/Rej/mAP/G@3 | 结论 |
|---|---|---|---:|---|
| `md_quality_dual` | `COMPLETE` | `artifacts/cross_backbone_stage_b/seed2023/moment/md_quality_dual` | 71.80/53.40/7.48/23.53 | 未超过 matched baseline 7.77/24.47 |

入口：

```bash
bash scripts/run_stage_b_quality_dual.sh moment 0
```

### 3.8 Moment batch-size probes

| batch | 状态 | 目录 | 结果 |
|---:|---|---|---|
| 1024 | `SMOKE/OK` | `artifacts/batch_probes/moment_hiea2m_b1024` | peak allocated 18.76 GiB |
| 1216 | `SMOKE/OK` | `artifacts/batch_probes/moment_hiea2m_b1216` | peak allocated 22.93 GiB，接近 3090 上限 |
| 1536 | `SMOKE/OOM-or-interrupted` | `artifacts/batch_probes/moment_hiea2m_b1536` | 无 `result.json`，不可称成功 |

入口：`scripts/probe_moment_batch_size.py`。

### 3.9 Moment smoke、冻结评估与校准输入

| 尝试 | 状态 | 目录 | 用途 |
|---|---|---|---|
| clean GMR smoke | `SMOKE` | `artifacts/smoke/md_gmr_clean` | strict null-loss 与梯度路径 |
| HieA2M smoke | `SMOKE` | `artifacts/smoke/md_hiea2m` | Q/D/C 初始化和训练路径 |
| HieA2M full validation smoke | `SMOKE` | `artifacts/smoke/md_hiea2m_full_eval` | 完整 evaluator 输出 |
| seed2018 frozen evaluation | `COMPLETE/辅助` | `artifacts/runs/md_hiea2m_release_seed2018/frozen_eval` | 冻结 checkpoint 复评 |
| seed2024 frozen evaluation | `COMPLETE/辅助` | `artifacts/runs/md_hiea2m_release_seed2024/frozen_eval` | 冻结 checkpoint 复评 |
| seed2023 calibration input | `COMPLETE/辅助` | `artifacts/runs/md_hiea2m_release_seed2023/calibration_input` | 阈值校准输入快照 |

## 4. 两级判空、去重与选框全部实验

统一父模型：

```text
artifacts/formal_strict/moment_detr/seed2023/
  md_hiea2m_b128_rerun_from_best_v2/best_joint.ckpt
```

### 4.1 Stage 1：几何去重

目录：

```text
artifacts/validation_selector_ablation/seed2023/stage1_geometry
artifacts/validation_dedup_ablation/md_hiea2m_b128_rerun_from_best_v2_best_map
```

已比较：

| 方法 | 状态 | 结论 |
|---|---|---|
| Direct Top-3/Top-5/predicted-count | `COMPLETE` | 必须保留的无去重基线 |
| Hard temporal NMS | `COMPLETE` | Top-3 小幅优于 direct |
| 1D DIoU-NMS | `COMPLETE` | 本批结果接近 Hard-NMS |
| Linear Soft-NMS | `COMPLETE` | Top-5 mAP 有小幅提升 |
| Gaussian Soft-NMS | `COMPLETE` | 已纳入网格 |
| complete-link representative/fusion | `COMPLETE` | Top-3 几何方法中最好，mAP 7.96 vs direct 7.57 |
| predicted-count + geometry | `FAILED as solution` | 约 5.67，瓶颈主要在计数而非重复框 |

入口：

```bash
bash scripts/run_learned_selector_stage.sh stage1_geometry 1
```

### 4.2 Stage 2–5：五条完整分支

| 分支 | selector seed | Zero positive weight | 状态 | 根目录 |
|---|---:|---:|---|---|
| 主线 | 2023 | 2.0 | `COMPLETE` | `artifacts/validation_selector_ablation/seed2023` |
| 权重消融 | 2023 | 1.0 | `COMPLETE` | `artifacts/validation_selector_ablation/seed2023_posw1` |
| 权重消融 | 2023 | 4.0 | `COMPLETE` | `artifacts/validation_selector_ablation/seed2023_posw4` |
| selector 稳定性 | 2024 | 2.0 | `COMPLETE` | `artifacts/validation_selector_ablation/seed2024` |
| selector 稳定性 | 2025 | 2.0 | `COMPLETE` | `artifacts/validation_selector_ablation/seed2025` |

每条分支均完成：

1. `stage2_zero`：只训练 Independent Zero Verifier；
2. `stage2_gate_calibration.json`：两级 rescue/veto 校准；
3. `stage3_pairwise`：只训练 same-event pairwise head；
4. `stage4_5_selection`：fixed learned Top-K、soft-count、early stop、fusion 网格。

五条分支的精确阶段目录如下，全部已有最终产物：

```text
artifacts/validation_selector_ablation/seed2023/stage2_zero
artifacts/validation_selector_ablation/seed2023/stage3_pairwise
artifacts/validation_selector_ablation/seed2023/stage3_pairwise/raw_val
artifacts/validation_selector_ablation/seed2023/stage4_5_selection

artifacts/validation_selector_ablation/seed2023_posw1/stage2_zero
artifacts/validation_selector_ablation/seed2023_posw1/stage3_pairwise
artifacts/validation_selector_ablation/seed2023_posw1/stage3_pairwise/raw_val
artifacts/validation_selector_ablation/seed2023_posw1/stage4_5_selection

artifacts/validation_selector_ablation/seed2023_posw4/stage2_zero
artifacts/validation_selector_ablation/seed2023_posw4/stage3_pairwise
artifacts/validation_selector_ablation/seed2023_posw4/stage3_pairwise/raw_val
artifacts/validation_selector_ablation/seed2023_posw4/stage4_5_selection

artifacts/validation_selector_ablation/seed2024/stage2_zero
artifacts/validation_selector_ablation/seed2024/stage3_pairwise
artifacts/validation_selector_ablation/seed2024/stage3_pairwise/raw_val
artifacts/validation_selector_ablation/seed2024/stage4_5_selection

artifacts/validation_selector_ablation/seed2025/stage2_zero
artifacts/validation_selector_ablation/seed2025/stage3_pairwise
artifacts/validation_selector_ablation/seed2025/stage3_pairwise/raw_val
artifacts/validation_selector_ablation/seed2025/stage4_5_selection
```

seed2024/2025 只改变 selector 新头的随机性，父 backbone 仍是 seed2023，
不能冒充完整 backbone 三 seed。

五分支平均结果：

| 选择方法 | AUROC | Rej-F1@0.4 | mAP | G@3 | 平均输出数 |
|---|---:|---:|---:|---:|---:|
| Direct Top-3 | 69.55 | 7.66 | 6.48 | 4.95 | 3.00 |
| Learned Top-3 | 69.55 | 7.66 | 6.96 | 5.06 | 3.00 |
| Learned + soft-count | 69.55 | 7.66 | 6.72 | 7.25 | 5.94 |

判断：

- `P` learned pairwise dedup：相同候选、相同 K 下稳定提高平均 mAP，保留；
- `SC` soft-count：提高 G@3，但平均 mAP 回落，尚未形成稳定净收益；
- `BF` cautious fusion：没有稳定净收益，不进入默认方法。

主线 HieA2M + Independent Zero 的当前关键结果：

| 方法 | AUROC | Rej-F1@0.4 | mAP | G@3 |
|---|---:|---:|---:|---:|
| strict Moment GMR | 70.95 | 61.22 | 8.93 | 30.78 |
| HieA2M-DGQC + Independent Zero | 72.62 | 69.28 | 9.16 | 39.77 |

执行入口：

```bash
bash scripts/run_learned_selector_branch.sh \
  RESULT_ROOT GPU_ID SELECTOR_SEED ZERO_POSITIVE_WEIGHT
```

分阶段入口：

```bash
SELECTOR_ROOT=RESULT_ROOT \
  bash scripts/run_learned_selector_stage.sh stage2_zero 1
SELECTOR_ROOT=RESULT_ROOT \
  bash scripts/run_learned_selector_stage.sh stage2_calibrate 1
SELECTOR_ROOT=RESULT_ROOT \
  bash scripts/run_learned_selector_stage.sh stage3_pairwise 1
SELECTOR_ROOT=RESULT_ROOT \
  bash scripts/run_learned_selector_stage.sh stage4_5_selection 1
```

所有五条分支已经完整，不要重跑。

### 4.3 selector smoke

| 尝试 | 状态 | 目录 |
|---|---|---|
| Zero head b128 smoke v2 | `SMOKE/历史` | `artifacts/smoke/selector_zero_b128_v2` |
| Zero head b128 smoke v3 | `SMOKE/OK` | `artifacts/smoke/selector_zero_b128_v3` |

v3 含 initialization/gradient audit；只用于证明冻结范围和梯度路径正确。

## 5. QD-DETR 全部实验

### 5.1 smoke 与 canonical/plain

| 实验 | 状态 | 目录 | 说明 |
|---|---|---|---|
| forward/eval smoke | `SMOKE` | `artifacts/smoke/qd_detr_gmr` | positive/null/mixed |
| CLI train/eval smoke | `SMOKE` | `artifacts/smoke/qd_detr_gmr_cli` | checkpoint selection 路径 |
| HieA2M smoke | `SMOKE` | `artifacts/smoke/qd_detr_gmr_hiea2m` | 第一版 |
| HieA2M smoke v2 | `SMOKE` | `artifacts/smoke/qd_detr_gmr_hiea2m_v2` | 修复版 |
| HieA2M CLI smoke | `SMOKE` | `artifacts/smoke/qd_hiea2m_cli` | full/threshold/adaptive |
| canonical original | `COMPLETE/历史` | `artifacts/canonical/qd_detr/seed2023/qd_detr` | 原始 plain 协议 |
| canonical b128 restart | `COMPLETE/历史` | `artifacts/canonical_b128_restart/qd_detr/seed2023/qd_detr` | weights-only b128 restart |
| formal plain | `COMPLETE/LEGACY` | `artifacts/formal/qd_detr/seed2023/qd_detr` | staged plain |

### 5.2 formal/legacy GMR 与组件轨

| 实验 | 状态 | 目录 | 处理 |
|---|---|---|---|
| QD-GMR | `EXPLORATORY` | `artifacts/formal/qd_detr/seed2023/qd_detr_gmr` | 旧 loss semantic，不进 strict 表 |
| QD-Quality | `EXPLORATORY` | `artifacts/formal/qd_detr/seed2023/qd_quality` | 旧 staged 组件 |
| QD-Dual | `EXPLORATORY` | `artifacts/formal/qd_detr/seed2023/qd_dual` | 旧 staged 组件 |
| QD-Counter | `EXPLORATORY` | `artifacts/formal/qd_detr/seed2023/qd_counter` | 旧 staged 组件 |
| QD-HieA2M | `EXPLORATORY` | `artifacts/formal/qd_detr/seed2023/qd_hiea2m` | 旧完整组合 |
| QD-GMR all parameters full LR | `INVALID` | `artifacts/formal/qd_detr/seed2023/qd_detr_gmr_invalid_all_lr` | optimizer protocol 错误，禁止引用 |

### 5.3 strict bsz32 组件矩阵

| Variant | 状态 | 最后 epoch | 目录 | 判断 |
|---|---|---:|---|---|
| `qd_detr_gmr` | `COMPLETE` | 109 | `artifacts/strict_bsz32/qd_detr/seed2023/qd_detr_gmr` | strict baseline |
| `qd_quality` | `COMPLETE` | 60 | `artifacts/strict_bsz32/qd_detr/seed2023/qd_quality` | 旧表显示正向，但公平审计后不能单独归因 |
| `qd_dual` | `RUNNING` | 84+ | `artifacts/strict_bsz32/qd_detr/seed2023/qd_dual` | 继续至完成/early stop |
| `qd_counter` | `FAILED/STOP` | 19 | `artifacts/strict_bsz32/qd_detr/seed2023/qd_counter` | 不超过 matched baseline |
| `qd_hiea2m` | `FAILED/STOP` | 12 | `artifacts/strict_bsz32/qd_detr/seed2023/qd_hiea2m` | G@3 峰值约 2.24，完整组合失败 |

入口：

```bash
bash scripts/start_qd_gmr_strict_bsz32.sh
bash scripts/run_components_matrix_strict_bsz32.sh qd 0
```

Counter/HieA2M 禁止恢复。

### 5.4 QD Q+D(no Counter)

| 实验 | 状态 | 目录 | AUROC/Rej/mAP/G@3 | 判断 |
|---|---|---|---:|---|
| `qd_quality_dual` | `COMPLETE` | `artifacts/cross_backbone_stage_b/seed2023/qd/qd_quality_dual` | 72.74/70.26/6.27/42.04 | 拒答强，但 mAP 明显下降 |

### 5.5 QD matched fair continued-control 矩阵

四条均从同一 parent、相同训练长度、相同 interval=5 完成：

| Variant | 状态 | AUROC | Rej-F1@0.4 | mAP | G@3 | 目录 |
|---|---|---:|---:|---:|---:|---|
| Continued Control | `COMPLETE` | 72.02 | 65.96 | 6.91 | 35.23 | `artifacts/qd_fair_ablation/seed2023_bsz32/continued_control` |
| + Quality | `COMPLETE` | 72.10 | 63.35 | 6.54 | 31.75 | `artifacts/qd_fair_ablation/seed2023_bsz32/quality` |
| + Dual | `COMPLETE` | 72.40 | 66.38 | 7.15 | 35.03 | `artifacts/qd_fair_ablation/seed2023_bsz32/dual` |
| + Quality + Dual | `COMPLETE` | 72.74 | 70.26 | 6.27 | 42.04 | `artifacts/qd_fair_ablation/seed2023_bsz32/quality_dual` |

公平审计结论：

- 早期 QD 从 G@3 约 3 跳到 35–42，主要来自继续训练/门限校准；
- Quality 没有超过 continued control，不能把旧轨的全部涨幅归因于 Quality；
- Dual 对 mAP 仅小幅提升，Q+D 用 mAP 换取强拒答。

入口：

```bash
GMR_CPU_THREADS=4 bash scripts/run_qd_fair_control_matrix.sh
```

矩阵已完成，脚本会拒绝覆盖旧目录。

## 6. CG-DETR 全部实验

### 6.1 smoke、canonical 与 formal

| 实验 | 状态 | 目录 |
|---|---|---|
| forward/evaluator smoke | `SMOKE` | `artifacts/smoke/cg_detr_gmr` |
| smoke baseline train | `SMOKE` | `artifacts/smoke/cg_detr_gmr/train_baseline` |
| smoke GMR train | `SMOKE` | `artifacts/smoke/cg_detr_gmr/train_gmr` |
| canonical original | `COMPLETE/历史` | `artifacts/canonical/cg_detr/seed2023/cg_detr` |
| canonical b128 restart | `COMPLETE/历史` | `artifacts/canonical_b128_restart/cg_detr/seed2023/cg_detr` |
| formal plain | `COMPLETE/LEGACY` | `artifacts/formal/cg_detr/seed2023/cg_detr` |
| formal b128 restart | `COMPLETE/历史` | `artifacts/formal_b128_restart/cg_detr/seed2023/cg_detr` |

### 6.2 strict bsz32 矩阵

| Variant | 状态 | 最后 epoch | 目录 | 结论 |
|---|---|---:|---|---|
| `cg_detr_gmr` | `COMPLETE/诊断` | 53 | `artifacts/strict_bsz32/cg_detr/seed2023/cg_detr_gmr` | strict GMR 轨 |
| `cg_quality` | `FAILED/STOP` | 19 | `artifacts/strict_bsz32/cg_detr/seed2023/cg_quality` | 无净收益 |
| `cg_phrase` | `FAILED/STOP` | 19 | `artifacts/strict_bsz32/cg_detr/seed2023/cg_phrase` | 无净收益 |
| `cg_counter` | `FAILED/STOP` | 13 | `artifacts/strict_bsz32/cg_detr/seed2023/cg_counter` | 无净收益 |
| `cg_hiea2m` | `FAILED/STOP` | 12 | `artifacts/strict_bsz32/cg_detr/seed2023/cg_hiea2m` | mAP 约 4.65、G@3 约 1.86 |

CG `Q+Phrase(no Counter)` Stage-B 未启动，skip marker 主动拦截。CG 全线当前
只作为失败移植/适用边界，不再占用 GPU。

入口仅用于追溯：

```bash
bash scripts/start_cg_gmr_strict_bsz32.sh
bash scripts/run_components_matrix_strict_bsz32.sh cg 0
```

## 7. EaTR 全部实验

### 7.1 canonical/formal

| 实验 | 状态 | 目录 |
|---|---|---|
| canonical original | `COMPLETE/历史` | `artifacts/canonical/eatr/seed2023/eatr` |
| canonical b128 restart | `COMPLETE/历史` | `artifacts/canonical_b128_restart/eatr/seed2023/eatr` |
| formal plain | `COMPLETE/LEGACY` | `artifacts/formal/eatr/seed2023/eatr` |
| formal GMR | `EXPLORATORY` | `artifacts/formal/eatr/seed2023/eatr_gmr` |

### 7.2 strict DGQC transfer

统一根目录：

```text
artifacts/eatr_dgqc_transfer/seed2023
```

| Variant | 状态 | 目录 | AUROC/Rej/mAP/G@3 | 结论 |
|---|---|---|---:|---|
| frozen plain parent | `COMPLETE` | `frozen_parent` | parent snapshot mAP/G@3 8.73/3.51 | 固定初始化 |
| `eatr_gmr_strict` | `COMPLETE` | `eatr_gmr_strict` | 71.67/39.35/8.02/16.82 | matched strict baseline |
| `eatr_quality` | `COMPLETE/PRIMARY` | `eatr_quality` | 71.98/44.31/8.24/19.13 | Q 同时提高四项主指标 |
| `eatr_dual` | `COMPLETE/PRIMARY` | `eatr_dual` | 72.05/48.12/8.06/21.10 | 拒答/G@3 明显增强，mAP 基本持平 |
| `eatr_counter` | `FAILED` | `eatr_counter` | mAP/G@3 约 6.16/12.15 | 明显低于 baseline |
| `eatr_hiea2m` | `FAILED` | `eatr_hiea2m` | mAP/G@3 约 7.08/9.56 | Q+D+C 负交互 |

执行入口：

```bash
bash scripts/run_eatr_dgqc_transfer.sh full 1
```

也支持 `gmr|resume_gmr|quality|dual|counter|hiea2m|children|resume_full`。
所有上述分支已完成，不要重跑 Counter/HieA2M。

精确目录：

```text
artifacts/eatr_dgqc_transfer/seed2023/frozen_parent
artifacts/eatr_dgqc_transfer/seed2023/eatr_gmr_strict
artifacts/eatr_dgqc_transfer/seed2023/eatr_quality
artifacts/eatr_dgqc_transfer/seed2023/eatr_dual
artifacts/eatr_dgqc_transfer/seed2023/eatr_counter
artifacts/eatr_dgqc_transfer/seed2023/eatr_hiea2m
```

### 7.3 EaTR Q+D(no Counter)

| 实验 | 状态 | 目录 | AUROC/Rej/mAP/G@3 | 结论 |
|---|---|---|---:|---|
| `eatr_quality_dual` | `COMPLETE` | `artifacts/cross_backbone_stage_b/seed2023/eatr/eatr_quality_dual` | 71.90/41.64/7.85/17.57 | G 略升，mAP -0.17，未晋级 |

## 8. Flash-VTG 全部实验

### 8.1 特征/运行时与 smoke

| 尝试 | 状态 | 目录 | 用途 |
|---|---|---|---|
| CPU/GMR minimal smoke | `SMOKE` | `artifacts/smoke/flash_vtg_gmr/hl-video_tef-cpu_gmr-2026-07-23-12-38-13` | 验证数据和 forward 路径 |
| plain full-null eval | `SMOKE` | `artifacts/smoke/flash_vtg_plain/full_null_eval` | 验证 null evaluation |
| b128 full model | `SMOKE/OK` | `artifacts/smoke/flash_vtg_gmr/b128_full_model` | 证明完整架构 bsz=128 可运行 |
| b128 checkpoint selection | `SMOKE/OK` | `artifacts/smoke/flash_vtg_gmr/b128_checkpoint_selection` | 验证 best-map/G/joint 保存 |
| b128 Q+Z warm-start | `SMOKE/OK` | `artifacts/smoke/flash_vtg_gmr/b128_quality_zero_warmstart` | 验证 Q 冻结、Z 训练与恢复 |

### 8.2 release anchor

| 实验 | 状态 | 目录 | AUROC/Rej/mAP/G@3 |
|---|---|---|---:|
| release GMR validation, raw teacher-report view | `COMPLETE` | `artifacts/flash_vtg_supplement/release_gmr_val` | 73.95/62.53/26.01/33.93 |

NMS 0.7 view 的 mAP/G@3 为约 27.63/34.25。写论文时必须明确使用 raw 还是
NMS view，不能从两列挑最好数字拼接。

### 8.3 release-parent head-only Q/Z/Q+Z

| Variant | 状态 | 目录 | raw AUROC/Rej/mAP/G@3 | 判断 |
|---|---|---|---:|---|
| GMR + Quality | `COMPLETE/PRIMARY` | `flash_vtg_gmr_quality` | 73.95/62.53/26.67/34.03 | Q 提高 mAP/G@3 |
| GMR + Independent Zero | `COMPLETE` | `flash_vtg_gmr_zero` | NMS view 74.29/60.98/27.61/32.01 | 只提高 AUROC，其他下降 |
| GMR + Quality + Zero | `COMPLETE` | `flash_vtg_gmr_quality_zero` | NMS view 74.13/61.12/28.06/33.12 | 未超过 Q-only |

完整路径均位于：

```text
artifacts/flash_vtg_supplement/seed2023_bsz128/
```

精确目录：

```text
artifacts/flash_vtg_supplement/seed2023_bsz128/flash_vtg_gmr_quality
artifacts/flash_vtg_supplement/seed2023_bsz128/flash_vtg_gmr_zero
artifacts/flash_vtg_supplement/seed2023_bsz128/flash_vtg_gmr_quality_zero
```

入口：

```bash
bash scripts/run_flash_vtg_strict.sh \
  gmr_quality GPU SEED 128 OUTPUT_DIR

bash scripts/run_flash_vtg_strict.sh \
  gmr_zero GPU SEED 128 OUTPUT_DIR

FLASH_INIT_CHECKPOINT=QUALITY_CKPT FLASH_FREEZE_QUALITY=1 \
  bash scripts/run_flash_vtg_strict.sh \
  gmr_quality_zero GPU SEED 128 OUTPUT_DIR
```

### 8.4 from-scratch matched pair

| Variant | 状态 | 快照进度 | 目录 | 当前 best-so-far |
|---|---|---:|---|---|
| Flash plain | `RUNNING` | e144 | `flash_vtg_plain` | teacher-report 快照 59.93/41.06/22.00/20.57 |
| Flash GMR | `RUNNING` | e90+ | `flash_vtg_gmr` | teacher-report 快照 70.16/67.07/24.08/39.85 |

四个数字依次为 AUROC/Rej-F1@0.4/mAP/G@3。它们仍会变化，不能作为最终论文数字。

恢复命令：

```bash
FLASH_ALLOW_EXISTING=1 FLASH_EVAL_EPOCH=5 FLASH_PATIENCE=80 \
FLASH_RESUME_CHECKPOINT=artifacts/flash_vtg_supplement/seed2023_bsz128/flash_vtg_plain/model_latest.ckpt \
  bash scripts/run_flash_vtg_strict.sh plain 0 2023 128 \
  artifacts/flash_vtg_supplement/seed2023_bsz128/flash_vtg_plain

FLASH_ALLOW_EXISTING=1 FLASH_EVAL_EPOCH=5 FLASH_PATIENCE=80 \
FLASH_RESUME_CHECKPOINT=artifacts/flash_vtg_supplement/seed2023_bsz128/flash_vtg_gmr/model_latest.ckpt \
  bash scripts/run_flash_vtg_strict.sh gmr 1 2023 128 \
  artifacts/flash_vtg_supplement/seed2023_bsz128/flash_vtg_gmr
```

当前进程存活，不要重复执行。

## 9. 跨骨干 Stage-B 与补充队列

### 9.1 已完成的 Q+D(no Counter)

| 骨干 | 状态 | 输出目录 | 结论 |
|---|---|---|---|
| Moment | `COMPLETE` | `cross_backbone_stage_b/seed2023/moment/md_quality_dual` | 未晋级 |
| QD | `COMPLETE` | `cross_backbone_stage_b/seed2023/qd/qd_quality_dual` | 强拒答、mAP 降低 |
| EaTR | `COMPLETE` | `cross_backbone_stage_b/seed2023/eatr/eatr_quality_dual` | G 略升、mAP 略降 |
| CG | `FAILED/SKIPPED` | 无正式目录 | skip marker 阻止启动 |

统一入口：

```bash
bash scripts/run_stage_b_quality_dual.sh moment GPU
bash scripts/run_stage_b_quality_dual.sh qd GPU
bash scripts/run_stage_b_quality_dual.sh eatr GPU
```

### 9.2 队列/管理脚本

| 脚本 | 作用 | 当前状态 |
|---|---|---|
| `queue_supplementary_experiments.sh` | 原始 wave1–5 总队列 | 历史入口 |
| `run_supplementary_after_wave1.sh` | Flash pair 后继续 Q/Z/Stage-B | 后续任务已被并行预取完成 |
| `run_parallel_ready_experiments.sh` | 并行启动 Flash Q/Z/Q+Z 与三骨干 Stage-B | 已完成 |
| `resume_current_interval5.sh` | 恢复 QD Q/D 与 Flash pair，统一 val interval=5 | 当前协调器正在运行 |
| `launch_nohup_job.sh` | 持久化后台包装器 | 当前 coordinator 使用 |

当前：

```text
artifacts/supplementary_queue/seed2023/queue.status
parallel_prefetch_completed
```

该状态表示预取的 Q/Z/Stage-B 已完成，不代表 Flash plain/GMR 已完成。

管理与日志目录：

```text
artifacts/supplementary_queue/seed2023/interval5_coordinator
artifacts/supplementary_queue/seed2023/interval5_resume
artifacts/supplementary_queue/seed2023/parallel_ready
artifacts/supplementary_queue/seed2023/parallel_ready_manager
```

其中 `interval5_resume/children.pid` 记录当前 wave 的原始子 PID；
`parallel_ready/` 保存各个预取实验的汇总日志；`runner.status` 或进程表优先于
旧 `children.pid`。

## 10. 其他工程与协议实验

### 10.1 后台持久化测试

| 尝试 | 状态 | 目录 |
|---|---|---|
| nohup 30s persistence | `SMOKE` | `artifacts/nohup_smoke/persist_30s` |
| setsid 30s persistence | `SMOKE` | `artifacts/nohup_smoke/setsid_sleep_30s` |
| screen 30s | `SMOKE` | `artifacts/nohup_smoke/screen_30s` |
| absolute screen 30s | `SMOKE` | `artifacts/nohup_smoke/screen_30s_abs` |
| screen 60s log | `SMOKE` | `artifacts/nohup_smoke/gmr_screen_60.log` |

这些只用于确认关闭 Codex/终端后训练能否继续，不是模型实验。

### 10.2 persistent handoff 旧日志

目录 `artifacts/persistent_handoff/` 保存：

- `canonical-cg.log`
- `canonical-qd.log`
- `eatr-dgqc.log`
- `qd-counter.log`
- `selector-main.log`
- `selector-posw1.log`
- `selector-seed2025.log`

它们是历史交接日志，不代表对应任务当前仍在运行。

### 10.3 preregistration 与 blind test

已有脚本：

```text
scripts/preregister_gmr_test.py
scripts/preregister_detr_matrix_test.py
scripts/run_preregistered_detr_matrix_test.py
```

状态：`SEALED`。

- 新方法没有执行 blind test；
- smoke 目录中的 `test4*.jsonl` 是微型合成/调试数据，不是 Soccer-GMR blind test；
- 最终候选、三 seed、阈值、selector、manifest 和 checkpoint 未冻结前，禁止运行 test；
- 正式 test 只允许按 preregistered manifest 执行一次。

## 11. 已失败、无效、禁止恢复总表

| 实验 | 原因 | 处理 |
|---|---|---|
| QD all-parameters full LR | optimizer protocol 错误 | `INVALID`，永不引用 |
| QD Counter strict | 未超过 matched baseline | `STOP` |
| QD HieA2M strict | Q+D+C 完整组合无收益 | `STOP` |
| CG Quality | 无净收益 | `STOP` |
| CG Phrase | 无净收益 | `STOP` |
| CG Counter | 无净收益 | `STOP` |
| CG HieA2M | 全组合无净收益 | `STOP` |
| CG Q+Phrase Stage-B | 前置证据不足 | `SKIPPED` |
| EaTR Counter | mAP/G@3 均下降 | `FAILED` |
| EaTR HieA2M full | 完整 Q+D+C 负交互 | `FAILED` |
| Moment/QD adaptive count truncation | 严重损害 localization recall | 不进主方法 |
| Soft-count | G 提升但 mAP 不稳定 | 暂不采用 |
| Boundary Fusion | 无稳定净收益 | 暂不采用 |
| Flash Z-only | 只提高 AUROC，Rej/mAP/G 不占优 | 不作为 Flash winner |
| Flash Q+Z | 没有超过 Q-only | Flash 当前保留 Q-only |
| Q+D Stage-B 三骨干 | 均未同时满足 mAP 非劣与 G 提升 | D 只保留为可选分支 |

## 12. 尚未做、不能伪称已做的实验

| 优先级 | 实验 | 状态/阻塞 |
|---|---|---|
| P0 | Moment 完全解耦 `Z(no Counter)` | `BLOCKED`：Zero evidence representation 仍来自 Counter 路径 |
| P0 | EaTR Independent Z、Z0–Z4 | `BLOCKED`：尚未接入统一 Zero head |
| P0 | QD Independent Z、Z0–Z4 | `BLOCKED`：尚未接入统一 Zero head |
| P0 | Moment/EaTR/QD 跨骨干 learned P | `BLOCKED`：EaTR/QD 缺 raw-query/pairwise 接口 |
| P1 | Flash learned P | `BLOCKED`：先在三核心骨干证明 P，再移植 |
| P1 | 最终 `U` 与 `U-D` seed2023 | `BLOCKED`：最终组件尚未冻结 |
| P2 | 四骨干完整 B/U seed2024、2025 | `BLOCKED`：seed2023 尚未闭环 |
| P2 | Flash B/U seed2024、2025 | `BLOCKED`：from-scratch seed2023 仍运行 |
| P2 | 正式 test + paired bootstrap | `SEALED` |

注意：已有 Moment HieA2M 三 seed和 selector-head seed2024/2025，但都不能替代
“最终 U 的完整 backbone 三 seed”。

## 13. 当前论文证据应如何表述

### 13.1 可以作为正向证据

1. Moment `HieA2M + Independent Zero`：四项主指标明显高于 strict GMR；
2. Quality：在 EaTR 与 Flash-VTG release parent 上提高 mAP/G@3；
3. DualGround：在 EaTR 上显著提高 Rej-F1/G@3，mAP 基本持平；
4. Learned Pairwise Dedup：五组 selector 设置中，相同 Top-3 预算平均 mAP +0.48；
5. 方法组合存在负交互：Counter/full HieA2M 在 EaTR/QD/CG 上失败，这是重要适用边界。

### 13.2 不能夸大的地方

1. 当前全部是 validation 证据；
2. QD Quality 的早期大涨被 fair continued-control 审计削弱；
3. Moment selector seed2024/2025 不是完整 backbone seed；
4. Flash from-scratch pair 仍在训练；
5. 完整 HieA2M 没有整体跨骨干泛化，真正较稳定的是组件级 Q，D 次之；
6. 最终 `B+Q+Z+P` 尚未在三个骨干完整闭环。

## 14. 报告与指标来源

| 文件 | 用途 |
|---|---|
| `artifacts/GMR_Teacher_Progress_Report_2026-07-23.pdf` | 当前老师汇报版，只保留有意义结果 |
| `artifacts/GMR_Completed_Experiments_Summary_2026-07-23.pdf` | 已完成实验汇总 |
| `artifacts/GMR_Experiment_Master_Summary_Current_2026-07-23.pdf` | 当前总览 |
| `artifacts/GMR_Experiment_Master_Summary_v2.pdf` | 旧版；部分 Moment 数字已过时 |
| `scripts/generate_teacher_progress_report.py` | 老师汇报 PDF 的可追溯生成器 |
| `scripts/generate_completed_experiment_summary.py` | completed summary 生成器 |
| `scripts/generate_current_experiment_master_summary.py` | master summary 生成器 |

数字冲突时优先级：

1. 当前实验目录中的原始 `best_*_metrics.json`；
2. 可追溯生成脚本读取的路径；
3. 本文与 teacher report；
4. 旧 PDF/旧聊天中的手抄数字。

## 15. 脚本索引

### 15.1 训练/恢复

```text
scripts/start_moment_strict_b128.sh
scripts/rerun_strict_moment_b128_from_best.sh
scripts/start_qd_gmr_strict_bsz32.sh
scripts/start_cg_gmr_strict_bsz32.sh
scripts/run_components_matrix_strict_bsz32.sh
scripts/start_qd_formal_variant_b128.sh
scripts/start_qd_gmr_b128.sh
scripts/start_resumed_b128_formal.sh
scripts/start_b128_restart.sh
scripts/resume_canonical_b128_exact.sh
scripts/run_eatr_dgqc_transfer.sh
scripts/run_flash_vtg_strict.sh
scripts/train_flash_vtg_gmr.sh
scripts/train_moment_detr_gmr.sh
scripts/run_stage_b_quality_dual.sh
scripts/run_qd_fair_control_matrix.sh
scripts/resume_current_interval5.sh
scripts/queue_flash_vtg_after_components.sh
scripts/queue_supplementary_experiments.sh
scripts/run_supplementary_after_wave1.sh
scripts/run_parallel_ready_experiments.sh
scripts/launch_nohup_job.sh
scripts/wait_then_persist_job.sh
```

### 15.2 selector/后处理

```text
scripts/ablate_temporal_dedup.py
scripts/calibrate_two_stage_gate.py
scripts/ablate_learned_selector.py
scripts/run_learned_selector_stage.sh
scripts/run_learned_selector_sequence.sh
scripts/run_learned_selector_branch.sh
scripts/continue_learned_selector_after_pid.sh
scripts/recover_selector_branch.sh
```

### 15.3 评估/统计/安全

```text
scripts/calibrate_hiea2m.py
scripts/fuse_gmr_heads.py
scripts/bootstrap_gmr.py
scripts/diagnose_gmr_groups.py
scripts/summarize_seed_metrics.py
scripts/infer_flash_vtg_gmr.sh
scripts/infer_moment_detr_gmr.sh
scripts/preregister_gmr_test.py
scripts/preregister_detr_matrix_test.py
scripts/run_preregistered_detr_matrix_test.py
```

## 16. 清空上下文后的最小阅读顺序

1. 本文：完整实验总账与当前运行状态；
2. `docs/current_unfinished_training_matrix_2026-07-23.md`：未完成任务和恢复命令；
3. `docs/cross_backbone_supplementary_experiment_plan_2026-07-23.md`：最终实验漏斗；
4. `docs/two_stage_learned_selector_ablation_plan_2026-07-22.md`：Z/P/soft-count 细节；
5. `docs/detr_gmr_lessons_and_algorithm_notes_2026-07-22.md`：失败经验；
6. `docs/detr_gmr_research_progress_2026-07-22.md`：较完整历史研究叙事；
7. `plans/detr_hiea2m_research.md`：长期计划与 blind-test 约束。

## 17. 下一步唯一推荐顺序

1. 等当前 QD Dual、Flash plain、Flash GMR 正常结束；
2. 汇总 Flash matched pair，判断 release-parent Q 的收益能否在 from-scratch 下保持；
3. 实现 Moment/EaTR/QD 的完全解耦 Independent Z；
4. 固定相同候选与输出预算，完成 Direct/NMS/learned P 跨骨干比较；
5. 确定最终 `U=B+Q+Z+P`，D 只作为可选 `U-D`；
6. 运行完整 backbone 的 seed2024、2025；
7. 冻结阈值、checkpoint、selector、manifest；
8. 最后一次性执行 blind test 和 paired bootstrap。
