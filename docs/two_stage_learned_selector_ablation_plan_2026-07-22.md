# 两级判空、学习式去重与软计数选框：实现和消融执行方案

日期：2026-07-22

## 1. 研究问题

最终方案固定为：

1. 宽松的第一层 existence gate，目标是高正样本放行率；
2. 独立的 `P(N=0)` 复核头，可救回第一层漏掉的正样本，只在高置信判空且定位证据弱时 veto；
3. 对确认非空的候选，用学习式 `P(same-event)` 抑制重复；
4. 条件计数分布只作为软数量先验，不强制输出恰好 K 个框；
5. 选完事件代表后才允许 complete-link 聚组，并只在边界方差很小时融合。

本轮首要检验不是“去重能否制造一个更高数字”，而是：在相同候选、相同分数、相同 K 下，
学习式去重是否稳定优于直接 Top-K，并且是否保护相邻/重叠的不同真实事件。

## 2. 已实现代码

### 2.1 几何去重基线

- `models/moment_detr_gmr/temporal_dedup.py`
  - direct Top-K / none；
  - Hard temporal NMS；
  - Gaussian Soft-NMS；
  - 1D DIoU-NMS；
  - 高置信重复聚组、边界投票和 Soft-NMS；
  - 固定 `max_output=K`，保证与直接 Top-K 使用同一预算。
- `scripts/ablate_temporal_dedup.py`
  - validation-only；
  - 强制比较 Top-1、Top-3、Top-5 和 predicted-count Top-K；
  - existence/count 字段完全不变；
  - 每个方法保存预测、完整指标、相对 direct Top-K 的差值及重复诊断；
  - 拒绝读取路径或元数据标记为 test 的文件。

### 2.2 独立判空与学习式 pairwise head

- `models/moment_detr_gmr/learned_selector.py`
  - `IndependentZeroVerifier`：输入 counter representation、候选分数/质量统计、熵、候选重叠和中心间隔，
    独立输出 `pred_zero_logits`，不使用 `1-p_exist`；
  - `PairwiseSameEventHead`：对称预测 `P(i,j 指向同一 GT 事件)`；
  - pairwise 输入包含 query 差异和乘积、局部视频特征、tIoU、IoM、中心/边界距离、时长比、
    DIoU 距离、候选分数，以及两个中心之间的平均/峰值帧证据；
  - `two_stage_accept`：实现“双层都判空才拒答、冲突优先放行、高置信零事件且定位弱才 veto”；
  - `learned_mmr_select`：学习式冗余惩罚、熵缩放的 soft-count prior 和提前停止；
  - `cautious_complete_link_fusion`：禁止 single-link chaining，只融合边界稳定的同事件簇。

raw validation 输出额外保存 `all_query_indices`；same-event 矩阵按该 ranking 同步重排，离线脚本
若缺少这个绑定会直接报错，避免把 ranking 顺序窗口与原 query-index 矩阵错位。

新增 Moment-DETR variants：

- `md_hiea2m_zero`：旧 HieA2M + 独立判空头；
- `md_hiea2m_pairwise`：再增加学习式 same-event head。

新增 staged optimizer 范围：

- `--trainable_scope zero`：只训练独立判空头；
- `--trainable_scope pairwise`：只训练 pairwise head；
- `--trainable_scope selection_heads`：后续联合微调选择相关头；
- `--trainable_scope all`：完整联合训练，只作为最后的额外消融。

冻结的 backbone 同时进入 eval mode，避免 frozen feature 被 dropout 随机扰动。

### 2.3 训练监督

- 第一层 existence：全部样本 BCE；
- 独立 `P(N=0)`：全部样本 BCE；正查询权重默认 2.0，使错误 veto 正样本的代价更高；
- 条件计数：只在正样本上训练 `{1,2,3,4+}`；
- localization、IoU、quality、dual grounding：新方案启用 `mask_null_vmr_loss`，null 不参加这些损失；
- pairwise：
  - 所有候选按 tIoU 分配到最近 GT，最低分配 IoU 默认 0.3；
  - 最佳和次佳 GT 的 IoU 差小于 0.05 时视为含糊候选并忽略；
  - 两候选分到同一 GT 为 duplicate positive；
  - 分到不同 GT 为 negative，且时间重叠越高，hard-negative 权重越大；
  - null 或没有可靠分配的候选对不进入 pairwise loss。

## 3. 严格消融顺序

### Stage 1：直接 Top-K 对几何去重

固定同一候选和 K，比较：

1. direct Top-K；
2. Hard-NMS + Top-K；
3. DIoU-NMS + Top-K；
4. Linear/Gaussian Soft-NMS + Top-K；
5. complete-link representative/fusion + Top-K。

这是无训练基线，只回答“候选中是否确实有可利用的重复”。

### Stage 2：独立判空

从 matched strict HieA2M checkpoint 初始化，只训练 `zero_verifier_head`，bsz=128。
训练后在 validation 扫描：

- `gate_recall_thd`: 0.15–0.35；
- `zero_decision_thd`: 0.5–0.8；
- `zero_veto_thd = zero_decision_thd + 0.05/0.10/0.15`；
- `zero_localization_thd`: 0.10–0.25。

主约束是正样本放行率不低于 95%，在满足约束的 Pareto 点中选择最高 G-mIoU@3。
阈值化后的 AUROC 没有解释价值，不参与选择；连续 gate/zero 分数保留用于 AUROC。

### Stage 3：学习式去重与固定 Top-K

从 Stage 2 checkpoint 初始化，只训练 `pairwise_same_event_head`。固定 K=3，比较：

- direct Top-3；
- learned same-event MMR + Top-3。

两者必须输出相同数量，唯一变化是 learned redundancy 是否把重复候选换成另一个事件候选。

### Stage 4：学习式去重与软计数

使用同一个 Stage 3 checkpoint，不再训练，仅在 validation 离线扫描：

- redundancy lambda：0.25 / 0.5 / 1.0 / 2.0；
- count-prior weight：0 / 0.25 / 0.5 / 1.0；
- stop threshold：-2.0 / -1.5 / -1.0 / -0.5。

比较 fixed learned Top-3 与 variable learned soft-count。计数分布熵越大，数量先验自动越弱；
确认非空后至少输出一个框，但不会强制凑满预测数量。

### Stage 5：谨慎边界融合

仅对 Stage 4 最佳配置扫描：

- same-event threshold：0.7 / 0.8 / 0.9；
- normalized boundary std：0.02 / 0.03 / 0.05。

complete-link 要求候选与簇中所有成员都满足同事件阈值。边界方差超过阈值时只保留 medoid，
不做坐标平均。

## 4. 执行脚本和顺序

所有脚本以前台方式运行，不创建 nohup 或 screen；训练 batch size 和 eval batch size 均为 128。

逐阶段运行：

```bash
bash scripts/run_learned_selector_stage.sh stage1_geometry 1
bash scripts/run_learned_selector_stage.sh stage2_zero 1
bash scripts/run_learned_selector_stage.sh stage2_calibrate 1
bash scripts/run_learned_selector_stage.sh stage3_pairwise 1
bash scripts/run_learned_selector_stage.sh stage4_5_selection 1
```

全部严格串行：

```bash
bash scripts/run_learned_selector_sequence.sh 1
```

依赖关系：Stage 2 必须产生 `best.ckpt`；Stage 2 calibration 必须产生冻结门限；Stage 3 才能加载
Stage 2；Stage 4/5 只对 Stage 3 做一次 raw validation 前向，后续网格均为 CPU 离线评估。

默认结果根目录：

```text
artifacts/validation_selector_ablation/seed2023/
├── stage1_geometry/
├── stage2_zero/
├── stage2_gate_calibration.json
├── stage3_pairwise/
│   └── raw_val/
└── stage4_5_selection/
```

每个训练阶段同时记录 `stdout.log`、`train.log`、`val.log`、`gradient_audit.json`、
`initialization_audit.json` 和 objective-specific checkpoints。

## 5. Stage 1 已完成结果

输入为 strict Moment HieA2M best-map validation 候选；existence 和 count 完全不变。

| 固定预算 | 方法 | mAP | mR@3 | G-mIoU@3 | 相对 direct Top-K |
|---|---|---:|---:|---:|---|
| Top-3 | direct Top-K | 7.57 | 10.20 | 8.21 | 基线 |
| Top-3 | Hard-NMS, 0.5 | 7.92 | 11.25 | 8.25 | mAP +0.35，mR@3 +1.05 |
| Top-3 | DIoU-NMS, 0.5 | 7.92 | 11.25 | 8.25 | 与本批 Hard-NMS 相同 |
| Top-3 | Linear Soft-NMS, 0.5 | 7.92 | 11.25 | 8.21 | mAP +0.35，mR@3 +1.05 |
| Top-3 | complete-link fusion, 0.5 | **7.96** | **11.41** | 8.23 | mAP +0.39，mR@3 +1.21 |
| Top-5 | direct Top-K | 8.67 | 10.20 | 8.21 | 基线 |
| Top-5 | Linear Soft-NMS, 0.5 | **8.84** | 11.25 | 8.21 | mAP +0.17，mR@3 +1.05 |
| predicted-count | direct / 所有去重 | 5.67 | 6.50 | 10.82 | 去重无收益 |

当前解释：简单去重确实有小幅价值，但 predicted-count 的主要瓶颈是计数器本身。融合虽然 Top-3
最好，却使 Top-1 mAP 从 5.48 降到 5.37，因此不能直接作为最终方案；这正是继续训练语义
pairwise head、再做谨慎融合的理由。

完整结果和日志：

- `artifacts/validation_selector_ablation/seed2023/stage1_geometry/dedup_ablation_summary.json`
- `artifacts/validation_selector_ablation/seed2023/stage1_geometry/run.log`

## 6. 验证状态与尚未执行部分

- 新增和相关回归测试共 55 项通过；
- 新模型前向已验证会同时输出独立 gate、zero 和对称 `[B,Q,Q]` same-event logits；
- bsz=128 的真实数据 zero-head smoke 已跑过两个 batch；gradient audit 中只有新头有梯度，
  `zero_verifier_head gradient_l2=1.1058`，冻结模块梯度均为零；
- 该 smoke 因服务器已有 9 个 GPU 训练进程导致每 batch 约 12–16 秒，确认反传后人工终止，
  没有把这个不完整 smoke 当成实验结果；
- Stage 2/3 正式训练尚未启动。应在现有 GPU 作业释放一个稳定槽位后按上述顺序运行；不能用
  当前未训练的 pairwise head 生成 Stage 3–5 结论。

## 7. 保留/淘汰标准

学习式方案只有同时满足以下条件才值得保留：

1. 固定 Top-3 下 mAP 和 mR@3 至少不低于 direct Top-3；
2. 同一 GT 重复输出率下降；
3. 高重叠/近邻不同 GT 的 pair recall 不下降；
4. 三个 seed 的收益方向一致，不以单 seed 偶然提升决策；
5. soft count 相对 learned fixed Top-K 提升 G-mIoU，且不能以明显 mAP/mR+ 损失换取；
6. 融合只有在 Top-1、mAP 和 close-event recall 都不恶化时才能进入最终方案。

如果 learned Top-K 不能稳定超过简单 Soft-NMS/Hard-NMS，则不采用复杂 pairwise head；如果
soft count 不能超过 fixed learned Top-K，则最终只保留学习式去重而不使用计数控制。
