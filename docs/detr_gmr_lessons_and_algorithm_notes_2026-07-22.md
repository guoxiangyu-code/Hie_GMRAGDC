# DETR-GMR 实验踩坑与算法修改备忘（短版）

更新时间：2026-07-22（Asia/Shanghai）

## 1. 当前结论

- 已完成 Moment-DETR、QD-DETR、CG-DETR、EaTR 的 GMR/HieA2M 适配与主要诊断工具。
- Moment-DETR 的旧 loss 轨在 validation 上出现 mAP 与 G-mIoU 同时提升，但未通过
  non-collapse 门槛，只能记为 **legacy diagnostic**，不能作为正式候选。
- QD-GMR、EaTR-GMR 的部分高 G-mIoU 主要来自过度拒答；聚合分数高不等于模型有效。
- 目前没有新模型取得 blind-test 资格；除公开 release checkpoint 的复现外，新的 test
  candidate 尚未运行，test 继续封存。

## 2. 踩过的坑与规避方法

### 2.1 评估口径不一致

- 只评 positive queries 会漏掉 GMR 的 null rejection；训练 validation 曾因此沿用 MR-only
  指标。现在所有 backbone 都在完整 465 条 validation 上使用统一 GMR evaluator。
- mAP/mR 只在 positive 上算，AUROC、Rej-F1、G-mIoU 在 positive/null 全集上算；两类指标
  不能互相替代。
- release checkpoint 使用固定 32-token 文本布局与 2 秒 rounding。clean attention mask 和
  continuous-time 是不同协议，必须从 matched baseline 重新训练，不能直接与 release 数字混比。
- NumPy 2.4 移除了 `np.trapz`，曾导致 evaluator 崩溃；兼容实现已修复。

### 2.2 高 G-mIoU 掩盖 all-empty-like collapse

- 仅看 aggregate G-mIoU 会奖励“几乎全部拒绝”的模型。
- QD-GMR 的典型失败点虽有 `mAP 7.73 / G@3 46.25`，但 null/single/multi acceptance 约为
  `1.43% / 49.70% / 2.22%`，multi queries 几乎被全部拒绝。
- 因此正式候选必须同时检查 balanced-G、single/multi acceptance retention、mR+@5 和分组
  指标；不能靠改阈值刚好越过门槛。

### 2.3 null 样本的 loss 语义曾实现错位

- 旧实现让 null 样本继续产生全 background classification 及部分辅助 loss 梯度，这与 GMR
  论文中“null 仅监督 existence、VMR assignment 为空”的语义不一致。
- 已在四个 backbone 加入显式 `--mask-null-vmr-loss`：null 排除 main/aux class、span、GIoU、
  quality、dual/event 等定位相关损失，保留 existence；all-null batch 返回 graph-connected zero。
- 旧 checkpoint 的结果统一标为 legacy-loss；strict-loss 结果必须另目录、matched 重训，二者
  不能混表。

### 2.4 warm-start、优化器和 batch 协议混杂

- 一条 QD 轨错误地让所有参数都使用新模块的全学习率，训练 14 epochs 后被判为无效实验；
  目录保留审计，但不进入结果矩阵。
- staged 轨混用 b128/b64、`1e-4` 与 shared-backbone `0.1x` 学习率；它们只适合机制筛查，
  不能称作严格论文复现。
- 新增模块采用全学习率，共享 backbone 使用明确的缩放组，并把参数分组写入实验日志。
- 修改 batch size 后不能把旧 checkpoint 称为“原样续训”。旧轨如需恢复，应保持原 batch；
  最大 batch 的 strict 轨必须另起 matched GMR/HieA2M 对照。

### 2.5 进程与恢复状态不可靠

- 之前依赖交互终端/PTY 的训练在会话结束后被清理，GPU 作业全部中断。
- 从现在起所有训练统一用 `nohup`，每个 run 独立保存启动命令、PID、stdout/stderr、train/val
  日志和 checkpoint；启动后还要核验 PID、首批日志与 GPU 显存。
- QD/CG/EaTR 旧 checkpoint 没保存 best score 和 early-stop counter，直接 resume 会改变模型
  选择状态。现已增加显式 training state；对旧 checkpoint 则从截至该 epoch 的 JSONL 日志
  确定性恢复，日志缺失时 fail closed。

### 2.6 后处理回放与 blind-test 安全

- 旧 raw-query 先经过 ranking/rounding，丢失原 query index 与未量化 span，无法证明校准结果
  可精确重放。
- 新 contract 需要保存 query index、normalized unrounded span、完整浮点 foreground/quality、
  duration 及可选 existence/count；QD/CG 已部分完成，EaTR 和 apply-only replay 尚待闭环。
- validation 可以搜索参数，test 只能读取冻结配置做无标签 apply-only；prereg/runner 的解释器、
  源码、特征、checkpoint、argv、输出和 ledger 都必须由 hash 闭合。

## 3. 已修改的算法思想

### 3.1 DualGround 时间语义路径

- 将 sentence/phrase 表征真正注入 DETR video memory，而非只增加旁路 proxy loss。
- 包含有效 EOS、learned dummies、RPG、Slot refinement、DQA 与 EOS reconstruction。
- residual gate 在 warm-start 的 step 0 为零，保证 parent 输出不被随机新分支破坏。

### 3.2 Query IoU-quality 校准

- 为每个 decoder query 预测与 matched GT 的 temporal IoU quality。
- 用 `foreground^(1-alpha) * quality^alpha` 排序，目标是缓解分类置信度与边界质量错位，并压低
  重复或低质量窗口。
- `alpha`、diversity 和阈值只能在 validation 冻结，不能根据 test 调整。

### 3.3 Hierarchical counter

- 将 flat `{0,1,2,3,4+}` 改为 existence 与 positive-conditional `{1,2,3,4+}` 两阶段建模。
- existence 在全部样本上监督；count CE、ordinal、contrastive、consistency 仅在 positive 上监督。
- 使用 inverse-sqrt positive count weights，并让稀有 multi positives 直接影响 existence 边界。

### 3.4 集合解码与多目标模型选择

- 实现 `full`、GREC-style `threshold`、HieA2G-style `adaptive` 三种解码，以及 quality/diversity
  ranking。
- 当前 `full` 是 primary；count 不确定时不做硬 top-k 截断，adaptive 只保留为诊断。
- 保存 mAP-best、G-mIoU-best、joint-best checkpoint。Moment 的 objective-specific head
  composition 显示定位与拒答的最佳 epoch 不一致，但现有结果仍未过 non-collapse gate。

## 4. 失败实验留下的有效结论

- fixed n-gram Temporal-HMSA：mAP 仅 `5.87 -> 6.00`，mR+@5 却 `2.33 -> 0.61`；不再扩展。
- flat five-class HMSA+TAGC：`G@3=44.73`，但 mAP `5.87 -> 2.70`；属于 null-dominated collapse。
- Moment threshold/adaptive：分别约 `6.95/38.96` 与 `4.23/38.96`，以明显 mAP 损失换 G 提升；
  不作为 primary。
- 从 all-empty-like parent 继续训练 quality/dual 会继承错误 decision boundary；strict 组件实验
  应从 matched plain/GMR parent 重新启动。
- 仅提高 rejection 已不是主要难点；后续应优先提高 multi-positive acceptance、raw coverage 和
  mR+@5，而不是继续强化“判空”。

## 5. 接下来只做四件事

1. 用真实完整前向/反向实测最大稳定 batch size，在两张 RTX 3090 上以 `nohup` 启动 matched
   strict GMR/HieA2M，并保存完整启动与训练日志。
2. 完成四个 backbone 的 strict 三 seed validation 矩阵，只有双指标提升且每 seed 通过
   non-collapse gate 才保留。
3. 完成 raw-query exact replay、apply-only calibration 与 prereg v3 攻击性回归。
4. 仅对完全冻结且通过门槛的配置执行一次 blind test；若没有合格配置，就报告 validation
   负结果并保持 test 封存。

详细进度与数值见 `docs/detr_gmr_research_progress_2026-07-22.md`。

两级判空、学习式去重、软计数选框的最新实现、严格消融顺序和 Stage 1 结果见
`docs/two_stage_learned_selector_ablation_plan_2026-07-22.md`。
