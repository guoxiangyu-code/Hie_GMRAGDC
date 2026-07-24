# 当前未完成与待进行训练矩阵

快照时间：2026-07-23 15:43 CST  
数据集：Standard validation，465 queries（255 positive / 210 null）  
当前统一设置：seed 2023、训练 batch size 尽量 128、每 5 epoch validation。

> 本文只把“仍在运行、中断未完成、尚未实现/尚未启动”的任务列入主矩阵。
> 最近完成的任务及指标放在第 4 节，作为选择下一阶段 parent 的依据。
> 指标均来自 validation，不与论文 test 数字混用。

## 1. 状态摘要

| 状态 | 数量 | 任务 |
|---|---:|---|
| 运行中 | 7 | Flash plain/GMR、原 QD Dual、QD matched fair-ablation 四条 |
| 中断未完成 | 0 | 当前没有可恢复但无人管理的训练 |
| 已完成、等待决策 | 7 | QD Quality；Flash Q/Z/Q+Z；Moment/EaTR/QD Q+D |
| 尚未实现或未启动 | 8 类 | 解耦 Z、Z0–Z4、learned P、最终 U/U-D、多种子、test |
| 禁止恢复 | CG 全线及 Counter 系列 | 已有充分失败证据 |

当前后台状态文件为：

```text
artifacts/supplementary_queue/seed2023/queue.status
```

截至快照时内容为：

```text
parallel_prefetch_completed
```

这表示可并行预取的 Flash Q/Z/Q+Z 与三骨干 Q+D 已完成；Flash plain/GMR
仍由各自 runner 继续训练。

## 2. 正在运行或中断未完成

### 2.1 指标矩阵

| 优先级 | 骨干/Variant | 状态与进度 | AUROC | Rej-F1@0.4 | mAP | G@3 | mR@5 | mR+@5 | 结果口径 |
|---|---|---|---:|---:|---:|---:|---:|---:|---|
| `P0` | Flash-VTG plain | 运行中，e105 | 62.92 | 27.94 | 25.21 | 14.88 | 36.86 | 13.72 | latest + NMS 0.7 |
| `P0` | Flash-VTG GMR | 运行中，e74 | 71.38 | 64.02 | 25.14 | 36.22 | 37.09 | 14.13 | latest + NMS 0.7 |
| `P1` | QD-DETR Dual | 运行中，PID 2459133，e73+ | 72.38 | 67.98 | 6.61 | 38.54 | 10.36 | 0.17 | 历史 best_joint；继续到 early-stop |
| `P0` | QD fair continued control | 运行中，PID 2590572，e58 | 72.02 | 65.96 | 6.91 | 35.23 | 9.95 | 0.67 | 当前 best_joint |
| `P0` | QD fair Quality | 运行中，PID 2590576，e55 | 72.60 | 70.10 | 5.99 | 40.90 | 8.65 | 0.83 | 当前 best_joint |
| `P0` | QD fair Dual | 运行中，PID 2590580，e46 | 72.40 | 66.38 | 7.15 | 35.03 | 10.33 | 0.00 | 当前 best_joint |
| `P0` | QD fair Quality+Dual | 运行中，PID 2590583，e50 | 72.74 | 70.26 | 6.27 | 42.04 | 9.32 | 0.00 | 当前 best_joint |

参考 baseline：

| 骨干 | AUROC | Rej-F1@0.4 | mAP | G@3 | mR@5 | mR+@5 |
|---|---:|---:|---:|---:|---:|---:|
| QD Strict GMR | 72.40 | 3.74 | 7.03 | 3.14 | 9.10 | 0.00 |
| Flash release GMR + NMS | 73.95 | 62.53 | 27.63 | 34.25 | 38.52 | 15.81 |

解释：

- Flash GMR 当前 G@3 已超过 release anchor，但 mAP 尚低 2.49，必须等收敛；
- Flash plain 只是 matched control，不作为论文方法，但必须完成以保证配对；
- 原 QD Dual 已经在继续训练，不需要另开副本；
- matched fair-ablation 四条使用同一 parent、相同训练时长和 interval=5，
  将用于消除旧 QD 实验训练进度不一致的问题。

### 2.2 当前运行与恢复指令

Flash 两项当前正在运行，不要重复启动。完整实际命令保存在：

```text
artifacts/flash_vtg_supplement/seed2023_bsz128/flash_vtg_plain/launch_command.txt
artifacts/flash_vtg_supplement/seed2023_bsz128/flash_vtg_gmr/launch_command.txt
```

若进程中断，使用 latest checkpoint 恢复：

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

QD Dual 当前已经恢复运行。以下只作为中断后的恢复命令，不要重复执行：

```bash
CUDA_VISIBLE_DEVICES=0 /home/guoxiangyu/miniconda3/bin/python -u \
  -m methods.qd_detr_gmr.train \
  --variant qd_dual \
  --output_dir artifacts/strict_bsz32/qd_detr/seed2023/qd_dual \
  --resume artifacts/strict_bsz32/qd_detr/seed2023/qd_dual/latest.ckpt \
  --seed 2023 --epochs 200 --batch_size 32 --eval_bsz 32 \
  --patience 50 --eval_interval 5 --num_workers 0 \
  --lr 3e-5 --backbone_lr_scale 0.1 \
  --train_sample_mode mixed --map_num_workers 1 \
  --reference_map 7.03 --reference_gmiou3 3.14 \
  --train_annotation data/label/Standard/train.jsonl \
  --eval_annotation data/label/Standard/val.jsonl \
  --video_feature_dirs Soccer-GMR/feature/standard/clip \
    Soccer-GMR/feature/standard/slowfast \
  --text_feature_dir Soccer-GMR/feature/standard/clip_text \
  --device cuda --round_to_clip --mask-null-vmr-loss
```

QD matched fair-ablation 当前也已启动，统一入口为：

```bash
GMR_CPU_THREADS=4 bash scripts/run_qd_fair_control_matrix.sh
```

输出与状态：

```text
artifacts/qd_fair_ablation/seed2023_bsz32/matrix.status
artifacts/qd_fair_ablation/seed2023_bsz32/{continued_control,quality,dual,quality_dual}
```

该入口检测到已有输出会拒绝覆盖，因此当前不要再次执行。

## 3. 尚未进行的训练矩阵

`U = B+Q+Z+P`，`U-D = B+Q+D+Z+P`。

| 顺序 | 优先级 | 骨干 | 待进行实验 | Parent 选择 | 当前指标 | 当前阻塞 | 训练指令 |
|---:|---|---|---|---|---|---|---|
| 1 | `P0 ★★★` | Moment-DETR | 解耦 `Z(no Counter)` | Strict B 或 Q parent | — | evidence encoder 仍与 Counter 耦合 | `N/A：先实现统一 Z 接口` |
| 2 | `P0 ★★★` | EaTR | Independent Z | 优先 B+Q；Q+D 未晋级 | — | 尚未实现 Z head | `N/A：实现后生成 run_zero_stage.sh` |
| 3 | `P0 ★★★` | QD-DETR | Independent Z | **QD Quality** | — | 尚未实现 Z head | `N/A：实现后生成 run_zero_stage.sh` |
| 4 | `P0 ★★★` | Moment/EaTR/QD | Z0–Z4 rescue/veto | 各骨干胜出 Z checkpoint | — | 必须先完成 1–3 | `N/A：统一 calibration evaluator 后运行` |
| 5 | `P0 ★★★` | Moment/EaTR/QD | Direct/NMS/geometry/learned P | 胜出 Q+Z parent | — | EaTR/QD 尚无 pairwise head/raw-query 接口 | `N/A：实现 pairwise head 后生成 run_pairwise_stage.sh` |
| 6 | `P1 ★★` | Flash-VTG | learned P | **Flash Quality** 或最终 Q/Z winner | — | 缺 raw queries 与 pairwise 输出 | `N/A：P 在三核心骨干有效后移植` |
| 7 | `P1 ★★` | 三/四核心骨干 | 最终 U 与可选 U-D，seed2023 | 由 Z/P 筛选确定 | — | 最终 variant 未冻结 | `N/A：冻结后生成 run_final_u.sh` |
| 8 | `P2 ★` | Moment/EaTR/QD/必要时 CG | B/U seed2024、2025 | 最终 U | — | seed2023 尚未闭环 | `N/A：生成统一 run_multiseed.sh` |
| 9 | `P2 ★` | Flash-VTG | B/U seed2024、2025 | Flash 最终 U | — | seed2023 尚未闭环 | `N/A：最终配置冻结后运行` |
| 10 | `P2 ★` | 全部最终骨干 | test + paired bootstrap | preregistered checkpoint | — | validation 决策未冻结 | `N/A：只允许最终执行一次` |

这里的 `N/A` 表示代码接口或 parent 尚未确定，当前不存在可诚实执行的训练命令；
不能为了表格完整而伪造命令。完成实现后，应先做 step-zero 等价测试和小数据
smoke，再把正式命令补入本表。

## 4. 最近完成的前置实验结果

### 4.1 Flash-VTG Q/Z/Q+Z

统一采用 best_joint + NMS 0.7：

| Variant | 状态 | AUROC | Rej-F1@0.4 | mAP | G@1 | G@3 | mR@5 | mR+@5 | 判断 |
|---|---|---:|---:|---:|---:|---:|---:|---:|---|
| GMR release anchor | 完成 | 73.95 | 62.53 | 27.63 | 40.02 | 34.25 | 38.52 | 15.81 | 对照 |
| GMR + Q | 完成 | 73.95 | 62.53 | **28.06** | **39.82** | **34.30** | **39.90** | **17.72** | 当前 winner |
| GMR + Z | 完成 | **74.29** | 60.98 | 27.61 | 38.26 | 32.01 | 38.52 | 15.81 | 只提高 AUROC |
| GMR + Q+Z | 完成 | 74.13 | 61.12 | **28.06** | 38.81 | 33.12 | **39.90** | **17.72** | 未超过 Q |

启动器：

```bash
FLASH_EPOCHS=120 FLASH_PATIENCE=30 FLASH_EVAL_EPOCH=5 \
FLASH_SELECTION_METRIC=mAP \
  bash scripts/run_flash_vtg_strict.sh gmr_quality 0 2023 128 \
  artifacts/flash_vtg_supplement/seed2023_bsz128/flash_vtg_gmr_quality

FLASH_EPOCHS=120 FLASH_PATIENCE=30 FLASH_EVAL_EPOCH=5 \
FLASH_SELECTION_METRIC=joint \
FLASH_REFERENCE_MAP=26.01 FLASH_REFERENCE_GMIOU3=33.93 \
  bash scripts/run_flash_vtg_strict.sh gmr_zero 1 2023 128 \
  artifacts/flash_vtg_supplement/seed2023_bsz128/flash_vtg_gmr_zero

FLASH_EPOCHS=120 FLASH_PATIENCE=30 FLASH_EVAL_EPOCH=5 \
FLASH_SELECTION_METRIC=joint \
FLASH_REFERENCE_MAP=26.01 FLASH_REFERENCE_GMIOU3=33.93 \
FLASH_INIT_CHECKPOINT=artifacts/flash_vtg_supplement/seed2023_bsz128/flash_vtg_gmr_quality/model_best_map.ckpt \
FLASH_FREEZE_QUALITY=1 \
  bash scripts/run_flash_vtg_strict.sh gmr_quality_zero 0 2023 128 \
  artifacts/flash_vtg_supplement/seed2023_bsz128/flash_vtg_gmr_quality_zero
```

这些命令对应已完成目录，不得直接重复执行；仅用于记录实验入口。

### 4.2 三骨干 Q+D(no Counter)

| 骨干 | Baseline mAP/G@3 | Q+D mAP | Q+D G@3 | AUROC | Rej-F1@0.4 | mR@5 | mR+@5 | 判断 |
|---|---|---:|---:|---:|---:|---:|---:|---|
| Moment-DETR | 7.77 / 24.47 | 7.48 | 23.53 | 71.80 | 53.40 | 12.34 | 1.67 | 未晋级 |
| EaTR | 8.02 / 16.82 | 7.85 | 17.57 | 71.90 | 41.64 | 12.57 | 3.87 | G@3 提升，mAP -0.17 |
| QD-DETR | 7.03 / 3.14 | 6.27 | 42.04 | 72.74 | 70.26 | 9.32 | 0.00 | 拒答强，mAP 明显下降 |

执行入口：

```bash
bash scripts/run_stage_b_quality_dual.sh moment 0
bash scripts/run_stage_b_quality_dual.sh eatr 1
bash scripts/run_stage_b_quality_dual.sh qd 1
```

CG 对应入口会被 skip marker 拦截，不再训练：

```bash
bash scripts/run_stage_b_quality_dual.sh cg 0
```

### 4.3 QD 单组件

| Variant | 状态 | AUROC | Rej-F1@0.4 | mAP | G@3 | mR@5 | mR+@5 | 判断 |
|---|---|---:|---:|---:|---:|---:|---:|---|
| Strict GMR | 完成 | 72.40 | 3.74 | 7.03 | 3.14 | 9.10 | 0.00 | baseline |
| Quality | 完成/early-stop | **72.50** | **70.24** | **7.56** | **42.36** | **10.80** | **0.67** | 明确晋级 |
| Dual | 中断未收尾 | 72.38 | 67.98 | 6.61 | 38.54 | 10.36 | 0.17 | mAP 不满足门槛 |
| Q+D | 完成 | 72.74 | 70.26 | 6.27 | 42.04 | 9.32 | 0.00 | 不如 Q |

因此，QD 的后续 Z parent 应选择 `qd_quality/best_joint.ckpt`，而不是 Dual
或 Q+D。

## 5. 禁止恢复/不再分配资源

| 骨干 | Variant | 处理 |
|---|---|---|
| EaTR | Counter、HieA2M full | 已完成但明显失败 |
| QD-DETR | Counter、HieA2M full | 已止损 |
| CG-DETR | Quality、Phrase、Counter、HieA2M | 全线止损 |
| CG-DETR | Q+Phrase(no Counter) | skip marker 阻止启动 |

## 6. 推荐资源顺序

1. 先让 Flash plain/GMR 收敛，锁定 matched baseline；
2. 等原 QD Dual 和 matched fair-ablation 四条自动完成，不启动重复副本；
3. 集中实现 Moment/EaTR/QD 的独立 Z；
4. Z0–Z4 校准后，只在胜出 parent 上训练 learned P；
5. 确认 `Q+Z+P` 在三个骨干有效后，再做 Flash P 和多种子；
6. 最后冻结 test，避免 validation/test 泄漏。

## 7. 监控指令

```bash
# 当前 Flash 状态与 epoch
for d in \
  artifacts/flash_vtg_supplement/seed2023_bsz128/flash_vtg_plain \
  artifacts/flash_vtg_supplement/seed2023_bsz128/flash_vtg_gmr
do
  echo "=== $d"
  cat "$d/runner.status"
  tail -n 2 "$d/train.log.txt"
done

# 最近 validation 指标文件
ls -lt \
  artifacts/flash_vtg_supplement/seed2023_bsz128/flash_vtg_plain/*metrics.json \
  artifacts/flash_vtg_supplement/seed2023_bsz128/flash_vtg_gmr/*metrics.json \
  | head

# 后台队列
cat artifacts/supplementary_queue/seed2023/queue.status
tail -n 20 artifacts/supplementary_queue/seed2023/waves.tsv

# QD matched fair-ablation
cat artifacts/qd_fair_ablation/seed2023_bsz32/matrix.status
tail -n 2 artifacts/qd_fair_ablation/seed2023_bsz32/*/train_log.jsonl
```
