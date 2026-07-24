# GMR 未完成实验执行矩阵

更新时间：2026-07-23 14:31 CST  
对应计划：`docs/cross_backbone_supplementary_experiment_plan_2026-07-23.md`

Validation 加速策略（2026-07-23 14:25 更新）：

- 当前 Flash wave 1 和 QD 单组件已经从 latest checkpoint 完整恢复，全部改为
  每 5 epoch validation；
- queue wave 2–5 同样全部为每 5 epoch validation；
- 后续新启动任务把 early-stop patience 从“30/50 次 validation”换算为约
  30/50 个训练 epoch；当前精确恢复任务保留 checkpoint 中的历史状态；
- screening 使用该设置；最终 B/U 多种子必须对配对方法使用完全相同的验证间隔。

CPU 策略（2026-07-23 14:31 更新）：

- 当前存活训练通过 CPU affinity 各限制为 6 个逻辑核；
- 后续所有矩阵任务统一设置 `GMR_CPU_THREADS=6`；
- 启动器同时设置 `OMP_NUM_THREADS`、`MKL_NUM_THREADS`、
  `OPENBLAS_NUM_THREADS`、`NUMEXPR_NUM_THREADS`；
- 允许范围固定为 4–8，超出范围时启动器直接拒绝运行。

并行策略（2026-07-23 14:34 更新）：

- Flash Q、Flash Z、QD Q+D、Moment Q+D、EaTR Q+D 可独立并行；
- Flash Q 完成并生成 `model_best_map.ckpt` 后立即启动 Flash Q+Z；
- 并行批次由 `scripts/run_parallel_ready_experiments.sh` 持久化管理；
- 原 interval-5 协调器完成当前任务后会等待/识别该批次，不会重复启动。

## 1. 优先级定义

| 标记 | 含义 | 资源策略 |
|---|---|---|
| `P0 ★★★` | 直接决定论文主方法组成 | 优先占用两张 GPU，完成前不启动多种子 |
| `P1 ★★` | 验证跨骨干/跨架构泛化 | P0 parent 确定后立即运行 |
| `P2 ★` | 正式多种子和统计检验 | 方法、阈值全部冻结后运行 |
| `STOP` | 已失败或已止损 | 不恢复，不占用资源 |

论文主方法候选为：

```text
U   = B + Q + Z + P
U-D = B + Q + D + Z + P
```

其中 `Q=Quality`、`D=Dual/Phrase`、`Z=Independent Zero`、
`P=Learned Pairwise Dedup`。Counter 和 Boundary Fusion 不进入默认组合。

## 2. 当前运行与已排队矩阵

> 重要：本表中的任务已经由现有进程或持久化队列管理。不要再次执行单项命令，
> 否则会重复训练并覆盖资源。若整台机器重启，才按第 4 节恢复。

| 优先级 | 骨干 | 实验 | 状态 | GPU/调度 | 输出目录 | 执行入口 |
|---|---|---|---|---|---|---|
| `P0 ★★★` | QD-DETR | `Q` | interval-5 恢复已结束 | physical GPU 0 | `artifacts/strict_bsz32/qd_detr/seed2023/qd_quality` | coordinator current wave |
| `P0 ★★★` | QD-DETR | `D` | interval-5 运行，PID 2459133 | GPU 0；CPU 0–5 | `artifacts/strict_bsz32/qd_detr/seed2023/qd_dual` | coordinator current wave |
| `P1 ★★` | Flash-VTG | plain matched control | interval-5 运行，PID 2459152 | GPU 0；CPU 6–11 | `artifacts/flash_vtg_supplement/seed2023_bsz128/flash_vtg_plain` | coordinator current wave |
| `P1 ★★` | Flash-VTG | GMR matched control | interval-5 运行，PID 2459153 | GPU 1；CPU 12–17 | `artifacts/flash_vtg_supplement/seed2023_bsz128/flash_vtg_gmr` | coordinator current wave |
| `P0 ★★★` | Flash-VTG | `Q` | 并行运行，PID 2487301 | GPU 0；6 CPU threads | `artifacts/flash_vtg_supplement/seed2023_bsz128/flash_vtg_gmr_quality` | parallel manager |
| `P0 ★★★` | Flash-VTG | 独立 `Z` | 并行运行，PID 2487303 | GPU 1；6 CPU threads | `artifacts/flash_vtg_supplement/seed2023_bsz128/flash_vtg_gmr_zero` | parallel manager |
| `P0 ★★★` | Flash-VTG | `Q+Z` | 等待 Q checkpoint 后自动启动 | GPU 0；6 CPU threads | `artifacts/flash_vtg_supplement/seed2023_bsz128/flash_vtg_gmr_quality_zero` | parallel manager |
| `P0 ★★★` | QD-DETR | `Q+D(no C)` | 并行运行，PID 2487322 | GPU 1；6 CPU threads | `artifacts/cross_backbone_stage_b/seed2023/qd/qd_quality_dual` | parallel manager |
| `P0 ★★★` | QD-DETR | `FAIR continued control` | 运行中 | GPU 0；4 CPU threads | `artifacts/qd_fair_ablation/seed2023_bsz32/continued_control` | matched fair-ablation matrix |
| `P0 ★★★` | QD-DETR | `FAIR Quality` | 运行中 | GPU 0；4 CPU threads | `artifacts/qd_fair_ablation/seed2023_bsz32/quality` | matched fair-ablation matrix |
| `P0 ★★★` | QD-DETR | `FAIR Dual` | 运行中 | GPU 1；4 CPU threads | `artifacts/qd_fair_ablation/seed2023_bsz32/dual` | matched fair-ablation matrix |
| `P0 ★★★` | QD-DETR | `FAIR Quality+Dual` | 运行中 | GPU 1；4 CPU threads | `artifacts/qd_fair_ablation/seed2023_bsz32/quality_dual` | matched fair-ablation matrix |
| `STOP` | CG-DETR | `Q+Phrase(no C)` | 已从队列跳过 | 不分配 | — | skip marker 生效 |
| `P0 ★★★` | Moment-DETR | `Q+D(no C)` | 并行运行，PID 2487299 | GPU 0；6 CPU threads | `artifacts/cross_backbone_stage_b/seed2023/moment/md_quality_dual` | parallel manager |
| `P0 ★★★` | EaTR | `Q+D(no C)` | 并行运行，PID 2487302 | GPU 1；6 CPU threads | `artifacts/cross_backbone_stage_b/seed2023/eatr/eatr_quality_dual` | parallel manager |

当前持久化协调器：

```bash
# 当前已经运行，PID 2459127；不要重复执行。
bash scripts/launch_nohup_job.sh \
  artifacts/supplementary_queue/seed2023/interval5_coordinator 0 \
  bash scripts/resume_current_interval5.sh
```

当前并行管理器：

```bash
# 当前已经运行，PID 2487256；不要重复执行。
bash scripts/launch_nohup_job.sh \
  artifacts/supplementary_queue/seed2023/parallel_ready_manager 0 \
  bash scripts/run_parallel_ready_experiments.sh
```

## 3. 尚未进行的实验矩阵

| 优先级 | 骨干 | 尚缺实验 | 前置依赖 | 当前是否可执行 | 运行指令 |
|---|---|---|---|---|---|
| `P0 ★★★` | Moment-DETR | 解耦 `Z(no C)` | 拆分 evidence encoder 与 Counter head | 否，尚未实现 | `N/A：先实现统一 Z 接口` |
| `P0 ★★★` | EaTR | 独立 `Z`、Z0–Z4 | 从 Q/Q+D 中选择固定 parent | 否，等待 Stage B | `N/A：parent 固定后生成脚本` |
| `P0 ★★★` | QD-DETR | 独立 `Z`、Z0–Z4 | 从 Q/Q+D 中选择固定 parent | 否，等待 Stage A/B | `N/A：parent 固定后生成脚本` |
| `P2 ★` | CG-DETR | 独立 `Z`、Z0–Z4 | 仅当其他三个骨干证明 Z 有效 | 当前不投入资源 | `N/A：保留为可选外部验证` |
| `P0 ★★★` | Moment/EaTR/QD | Direct/NMS/geometry/learned P | Z 胜出，并固定同一候选文件 | 否，尚未跨骨干接通 P | `N/A：实现 pairwise head/evaluator 后运行` |
| `P1 ★★` | CG-DETR | learned P 公平消融 | 前三个骨干先证明 P 值得使用 | 否 | `N/A：P0 有效后移植` |
| `P1 ★★` | Flash-VTG | learned P 公平消融 | Flash Q+Z 胜出 | 否 | `N/A：补 raw queries 与 pairwise 输出` |
| `P1 ★★` | 四核心骨干 | 最终 U 与 U-D seed2023 对比 | Q/D/Z/P 筛选完成 | 否 | `N/A：最终 variant 尚未冻结` |
| `P2 ★` | 四核心骨干 | B/U seed2024、2025 | 最终 U、阈值和 checkpoint 规则冻结 | 暂不执行 | `N/A：冻结后生成统一多种子脚本` |
| `P2 ★` | Flash-VTG | B/U seed2024、2025 | Flash seed2023 证明有效 | 暂不执行 | `N/A：冻结后运行` |
| `P2 ★` | 全部骨干 | test + paired bootstrap | preregistration 完成 | 禁止提前执行 | `N/A：只允许最终运行一次` |

### P0 集中资源顺序

1. 完成当前 QD 的 Q、D，确定哪些单组件晋级；
2. 完成四骨干 `Q+D(no Counter)` 和 Flash Q/Z/Q+Z；
3. 立即实现三个代表性骨干的解耦 Z：Moment、EaTR、QD；
4. 在相同候选、相同输出预算下完成 Direct Top-K 与 learned P；
5. 只有前三个骨干证明 Z/P 有效，才扩展 CG 和 Flash；
6. 最终 U 确定前，不启动 seed2024/2025。

三个代表性骨干覆盖：

- Moment-DETR：已有最完整的 HieA2M/selector 代码；
- EaTR：当前 Q、D 的正式结果最清楚；
- QD-DETR：用于验证方法不是 Moment/EaTR 特例。

## 4. 可执行命令目录

### 4.1 整体后台队列

仅在确认没有现存 interval-5 coordinator 时执行：

```bash
ps -eo pid,ppid,sid,etimes,stat,args \
  | rg 'resume_current_interval5.sh'

bash scripts/launch_nohup_job.sh \
  artifacts/supplementary_queue/seed2023/interval5_coordinator 0 \
  bash scripts/resume_current_interval5.sh
```

该入口自动持久化，不依赖 Codex 或当前终端继续打开。

### 4.2 Flash-VTG 单项命令

以下命令用于队列故障后的单项恢复；正常情况下由 queue worker 自动执行：

```bash
# matched controls
bash scripts/run_flash_vtg_strict.sh plain 0 2023 128 \
  artifacts/flash_vtg_supplement/seed2023_bsz128/flash_vtg_plain
bash scripts/run_flash_vtg_strict.sh gmr 1 2023 128 \
  artifacts/flash_vtg_supplement/seed2023_bsz128/flash_vtg_gmr

# frozen-parent single heads
FLASH_EPOCHS=120 FLASH_PATIENCE=30 FLASH_SELECTION_METRIC=mAP \
  bash scripts/run_flash_vtg_strict.sh gmr_quality 0 2023 128 \
  artifacts/flash_vtg_supplement/seed2023_bsz128/flash_vtg_gmr_quality

FLASH_EPOCHS=120 FLASH_PATIENCE=30 FLASH_SELECTION_METRIC=joint \
FLASH_REFERENCE_MAP=26.01 FLASH_REFERENCE_GMIOU3=33.93 \
  bash scripts/run_flash_vtg_strict.sh gmr_zero 1 2023 128 \
  artifacts/flash_vtg_supplement/seed2023_bsz128/flash_vtg_gmr_zero

# Q+Z：必须等待 Q 的 model_best_map.ckpt 存在
FLASH_EPOCHS=120 FLASH_PATIENCE=30 FLASH_SELECTION_METRIC=joint \
FLASH_REFERENCE_MAP=26.01 FLASH_REFERENCE_GMIOU3=33.93 \
FLASH_INIT_CHECKPOINT=artifacts/flash_vtg_supplement/seed2023_bsz128/flash_vtg_gmr_quality/model_best_map.ckpt \
FLASH_FREEZE_QUALITY=1 \
  bash scripts/run_flash_vtg_strict.sh gmr_quality_zero 0 2023 128 \
  artifacts/flash_vtg_supplement/seed2023_bsz128/flash_vtg_gmr_quality_zero
```

### 4.3 当前保留的 `Q+D(no Counter)`

正常情况下由 queue worker 自动执行。单项恢复命令：

```bash
bash scripts/run_stage_b_quality_dual.sh qd 1
bash scripts/run_stage_b_quality_dual.sh moment 1
bash scripts/run_stage_b_quality_dual.sh eatr 0
```

CG 已设置 skip marker；下面的命令只会打印跳过原因并正常退出，不会启动训练：

```bash
bash scripts/run_stage_b_quality_dual.sh cg 0
```

不要并行执行写入相同输出目录的两个副本。启动器检测到既有训练产物时会拒绝
覆盖；若任务中断，应先检查 checkpoint 和恢复语义，不能通过删除目录强行重跑。

### 4.4 当前 QD/CG 单组件的实际命令

QD 两项仍在运行；CG 两项已于 e19 止损。可用下列只读命令查看仍在运行或
已经消失的 PID，作为审计依据：

```bash
ps -p 2459132 -o args=  # QD Quality, --eval_interval 5
ps -p 2459133 -o args=  # QD Dual, --eval_interval 5
ps -p 2203245 -o args=  # CG Quality，已停止，应无输出
ps -p 2203247 -o args=  # CG Phrase，已停止，应无输出
```

旧 PID 2202474/2202476 已被 interval-5 PID 2459132/2459133 替代。若新 PID
消失，先检查对应输出目录的 `train_log.jsonl`、latest/best
checkpoint 和最后一个 epoch，再生成只恢复该 variant 的命令；禁止直接执行
`run_components_matrix_strict_bsz32.sh`，因为它会把已经止损的 Counter 和
HieA2M 一并重新启动。

## 5. 已失败实验：禁止重新占用资源

| 骨干 | 实验 | 状态 | 处理 |
|---|---|---|---|
| EaTR | Counter | `STOP` | 已完成但明显低于 baseline |
| EaTR | Q+D+Counter/HieA2M | `STOP` | 已完成，负交互 |
| QD-DETR | Counter | `STOP` | e18 止损，保留产物 |
| QD-DETR | Q+D+Counter/HieA2M | `STOP` | e12 止损，保留产物 |
| CG-DETR | Counter | `STOP` | e13 止损，保留产物 |
| CG-DETR | Q+Phrase+Counter/HieA2M | `STOP` | e12 止损，保留产物 |
| CG-DETR | Quality | `STOP` | e19 止损；mAP 3.97、G@3 1.49 |
| CG-DETR | Phrase | `STOP` | e19 止损；mAP 3.89、G@3 1.41 |
| CG-DETR | Q+Phrase(no Counter) | `STOP` | 已通过 marker 从 queue wave 4 跳过 |

Moment-DETR 旧 release-parent 的 Counter/HieA2M 结果标记为“探索性”，不纳入
严格失败表，但在公共方法确定前也不分配资源重跑。

## 6. 监控指令

```bash
# 队列当前波次
cat artifacts/supplementary_queue/seed2023/queue.status
tail -n 30 artifacts/supplementary_queue/seed2023/waves.tsv

# 当前训练进程
ps -eo pid,ppid,sid,etimes,stat,pcpu,pmem,args \
  | rg 'training.flash_vtg_gmr|methods\\.(qd_detr_gmr|cg_detr_gmr|eatr_gmr)\\.train|training/moment_detr_gmr/train.py'

# GPU
nvidia-smi

# 当前 wave 日志
tail -f artifacts/supplementary_queue/seed2023/wave1_plain.log
tail -f artifacts/supplementary_queue/seed2023/wave1_gmr.log
```

## 7. 资源决策

当前不要再新增训练进程。宿主机保留四个 interval-5 训练任务，协调器会自动进入
后续 P0 波次。GPU 显存未满不代表可以继续加任务：当前瞬时 GPU utilization
较低而 CPU 占用很高，额外并发更可能拖慢数据准备和 validation。

本轮完成后，第一优先不是多种子，而是实现并运行解耦 Z 和 learned P。只有
`Q+Z+P` 在三个代表性骨干通过 mAP 非劣与 G@3 提升门槛，才进入跨五骨干与
seed2024/2025 的正式验证。
