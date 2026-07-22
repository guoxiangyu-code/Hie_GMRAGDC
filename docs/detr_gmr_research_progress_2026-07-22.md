# DETR-GMR / HieA2M 实验进度、结果与后续研究方向

更新时间：2026-07-22 13:13（Asia/Shanghai）

## 1. 文档目的

本文记录当前针对 Soccer-GMR 的 DETR 系方法改进工作，包括：

- 已尝试的研究思路及其动机；
- 已完成的实现、实验与验证；
- 当前可引用的结果、仅能作为诊断的结果，以及明确失败的结果；
- 正在运行的训练矩阵；
- 进入 blind test 前仍需解决的问题；
- 下一阶段值得优先研究的方向。

最重要的状态结论是：**目前还没有新方法候选具备进入 blind test 的资格**。Moment-DETR
的 validation 聚合指标已有明显双升，但没有通过预先规定的 non-collapse gate；QD/CG/EaTR
矩阵仍在训练。除公开 release checkpoint 的复现外，新的 test candidate 尚未执行。

## 2. 研究目标与覆盖范围

目标是在 Soccer-GMR Standard split 上，对同一 backbone、同一输入特征、同一训练协议的
GMR baseline 同时提升：

1. positive queries 上的定位 `mAP`；
2. 包含 null queries 的端到端 `G-mIoU@3`；
3. multi-moment queries 上的 `mR+@5` 与 positive acceptance；
4. null/single/multi 均衡表现，而不是利用 null 占比制造 all-empty-like 高分。

主矩阵覆盖：

| Backbone | Plain VMR | GMR adapter | HieA2M / 组件消融 |
|---|---:|---:|---:|
| Moment-DETR | 必做 | 必做 | 必做，三 seed 与完整消融 |
| QD-DETR | 必做 | 新增 | 必做 |
| CG-DETR | 必做 | 新增 | 必做，语义路径消融 |
| EaTR | 必做 | 必做 | 必做 |

MS-DETR、LD-DETR 和 Sim-DETR 暂列第二阶段；在上述四个主干完成前，不用扩展方法替代主矩阵。

## 3. 固定数据与评估协议

正式发布资产为：

```text
data/label/Standard/{train,val,test}.jsonl
Soccer-GMR/feature/standard/{slowfast,clip,clip_text}
```

统一评估规则：

- `mAP/mR/mIoU` 仅在 positive queries 上计算；
- `AUROC/Rej-F1/G-mIoU` 在完整 positive/null 集合上计算；
- 主 operating point 为 `tau=0.4`，同时报告 `0.4/0.6/0.8`；
- release anchor 使用 2 秒 clip rounding；continuous-time 只能作为独立协议；
- validation 用于模型、checkpoint、阈值和后处理选择；blind test 只能在全部冻结后执行一次。

文本输入有两个不能混用的协议：

- `release-compatible`：保留公开 checkpoint 使用的固定 32-token 布局；
- `clean-mask`：按保存的 attention mask 去除 padding，必须从 matched baseline 重新训练。

直接把 clean-mask 输入喂给 release checkpoint 会产生分布漂移，不能与论文数字进行同轨比较。

## 4. 已尝试的核心研究思路

### 4.1 发布锚点与统一 GMR evaluator

首先复现 Moment-DETR-GMR 发布 checkpoint，并修复 NumPy 2.4 中 `np.trapz` 被移除导致的
evaluator 兼容问题。随后将所有 backbone 的 validation/test 都统一到完整 465-query GMR
evaluator，避免只评估 positive queries 或沿用 MR-only metric key。

这一阶段的作用是建立可核对的 release anchor，并为后续所有方法提供同一评估口径。

### 4.2 DualGround 时间语义路径

将 Dual Grounding 的思想改造成进入 DETR memory 的可训练时间语义路径：

- sentence path 使用有效 EOS 与 learned dummies；
- phrase path 使用 RPG、Slot refinement、phrase-EOS 与时间编码；
- DQA 与 EOS reconstruction 作为辅助约束；
- sentence/phrase residual gate 在 warm-start 时严格为零，保证 parent 输出不被随机新分支破坏。

该路径与此前失败的 fixed n-gram auxiliary proxy 不同：它直接改变送入 DETR encoder/decoder
的 video memory，并保留可审计的 step-0 parent fidelity。

### 4.3 IoU-quality query calibration

为每个 final decoder query 预测其与 matched GT 的 temporal IoU quality，并用

```text
score = p_foreground^(1-alpha) * p_quality^alpha
```

重新排序 query。目标是缓解 DETR 分类分数与边界质量不一致的问题，同时压低重复或边界较差的
query。`alpha`、temporal diversity 等后处理参数只允许在 validation 上选择。

### 4.4 Hierarchical adaptive counter

将 flat `{0,1,2,3,4+}` 计数分解为：

```text
P(N=0) = 1 - P(exists)
P(N=k) = P(exists) P(N=k | exists), k in {1,2,3,4+}
```

实现包括：

- existence BCE 在全部样本上训练；
- positive-conditional CE、ordinal、contrastive 与 consistency 只在 positive 样本上训练；
- positive count 使用 inverse-sqrt class weights；
- multi positives 的 existence 权重也继承相对长尾权重；
- `full / threshold / adaptive / hard` 多种 decode 分开保存，避免不成熟的 count head 直接破坏
  primary localization recall。

### 4.5 Objective-specific head composition

GMR 的定位 decoder 与 existence adapter 是并行输出头，因此尝试了可审计的双 checkpoint
composition：

- mAP-best checkpoint 只提供 `pred_relevant_windows`；
- joint-best checkpoint 只提供 existence/count 字段；
- 不平均 span，不混合 query score；
- 对 qid 全覆盖与输入/输出 SHA-256 做检查。

这显著提高了 Moment validation 的聚合 mAP/G@3，但当前 provenance 只闭合到 submission hash，
尚未完整绑定 producer checkpoint、精确 argv、source、calibration 与 fusion 全链路。因此它仍是
validation diagnostic，不能直接进入 blind test。

### 4.6 跨 backbone 移植

已经完成 QD-DETR、CG-DETR、EaTR 的隔离 upstream pin、Soccer-GMR dataset/runtime、plain/GMR
以及 quality/dual(or phrase)/counter/full HieA2M 变体。各 backbone 的注入位置不同：

- QD-DETR：DualGround 在 query-dependent video representation 前；
- CG-DETR：复用已有 ACA/dummy sentence path，只新增 phrase temporal path；
- EaTR：DualGround memory 同时进入 event 与 moment reasoning。

所有新增分支都有 checkpoint 结构检测、parent migration 白名单与 staged optimizer group 记录。

### 4.7 论文原意的 strict null supervision

复审 GMR 论文后发现，论文明确要求 null-set 样本的 Hungarian assignment 为空，VMR loss 不产生
梯度，null 只由 existence loss 监督。标准 DETR 的全 background classification 实际仍会产生
梯度，因此此前兼容旧 checkpoint 的训练并不完全符合这一语义。

目前四个 backbone 均已实现显式 `mask_null_vmr_loss`：

- main/aux classification、span、GIoU 排除 null；
- quality、DualGround、EaTR pseudo-event 等 sample-dependent loss 排除 null；
- all-null batch 返回可反传的 graph-connected zero；
- existence 仍监督全部样本；conditional count 仍只监督 positives；
- QD/CG exact resume 会拒绝 strict/legacy loss semantic 漂移。

该开关默认 `false` 只为旧 checkpoint 兼容。后续 paper-literal 正式命令必须显式传
`--mask-null-vmr-loss`。现有正在运行的 staged 轨在进程启动时加载的是旧 semantic，只能作为
exploratory diagnostic。

### 4.8 Validation calibration 与 blind-test preregistration

已实现 raw-query validation calibration 基础，并补充以下安全约束：

- 显式声明 annotation identity 与 `validation` role；
- 校验 annotation byte-level SHA-256；
- 绑定 producer checkpoint、producer argv artifact、source files 与 reference metrics；
- 保存完整 grid records、selected submission path/hash/row count；
- 运行结束前再次检查所有输入 hash，防止校准过程中发生漂移。

同时，七个 allow-listed CLI 已关闭 argparse abbreviation，防止在已绑定的完整参数后追加
`--checkp`、`--eval_annot` 等缩写覆盖 checkpoint 或 annotation。

preregistration v2 的安全复审仍发现 P0：解释器/影子模块未形成输入闭包、自签 seal 可重算、
ledger root 可任意选择、四 backbone roster 不完整、primary metrics/submission 在运行后不解析、
feature/source/provenance 未全部绑定等。目前正在升级 prereg/runner；这些问题关闭前禁止领取
one-shot blind-test claim。

## 5. 当前结果

### 5.1 Moment-DETR-GMR release anchor

公开 release checkpoint 的 test 复现结果：

| AUROC | Rej-F1@0.4 | mAP | mR@5 | mR+@5 | G-mIoU@1 | G-mIoU@3 |
|---:|---:|---:|---:|---:|---:|---:|
| 72.09 | 64.01 | 7.52 | 12.96 | 0.84 | 35.84 | 32.89 |

同一 release-compatible 协议下的 validation anchor：

| mAP | G-mIoU@3 | mR+@5 | balanced-G |
|---:|---:|---:|---:|
| 8.14 | 33.21 | 0.50 | 36.4604 |

这里只复现公开 checkpoint。新的模型候选没有读取或执行 blind test。

### 5.2 Moment 单组件 screening（seed 2023）

以下为 joint-best validation `mAP / G-mIoU@3 / mR+@5`：

| 变体 | mAP | G-mIoU@3 | mR+@5 | 结论 |
|---|---:|---:|---:|---|
| Quality | 8.52 | 35.05 | 2.39 | 聚合双升，定位排序最有价值 |
| Counter | 8.58 | 35.09 | 1.00 | 聚合双升，但需分组防塌缩复核 |
| DualGround | 8.32 | 34.57 | 1.11 | 小幅双升，单独收益较弱 |
| Full HieA2M single checkpoint | 8.52 | 36.65 | 1.44 | G@3 提升较大 |
| Val-calibrated full | 8.63 | 36.59 | 1.44 | `alpha=.25, diversity=.5` |

这些 screening 发生在 strict loss 与若干审计修复之前，只能用于判断组件方向，不能作为最终
paper-literal 结果。

### 5.3 Moment objective-specific composition（三 seed）

固定 `tau=.4` 的 validation 结果：

| seed | mAP | G-mIoU@3 | mR+@5 | balanced-G | Gate 状态 |
|---:|---:|---:|---:|---:|---|
| 2018 | 9.70 | 35.31 | 0.83 | 38.7061 | 未过 `>50` |
| 2023 | 10.34 | 36.90 | 1.17 | 40.4929 | 未过 `>50` |
| 2024 | 9.55 | 35.01 | 1.06 | 38.3898 | 未过 `>50` |
| mean | 9.86 | 35.74 | 1.02 | — | 不具备 test 资格 |

相对 release validation anchor，三 seed mean 为 `+1.72 mAP / +2.53 G@3`。seed 2023 paired
bootstrap 为：

- `Delta mAP = +2.20`，95% CI `[+0.76,+3.70]`；
- `Delta G@3 = +3.69`，95% CI `[+1.87,+5.64]`；
- 两项同时为正的 bootstrap 概率为 `0.9992`。

同 seed、同共享层学习率的纯 GMR 续训控制经相同 composition 后为 `9.18/34.43`；HieA2M
相对该控制仍有 `+1.16/+2.47`，两个 paired CI 均高于零。

但 balanced-G 是预先规定的硬门槛，不能因为聚合指标好看而事后放宽。阈值扫描也不能作为
补救：更高阈值虽会提高 aggregate G@3，却会把 multi acceptance 压到接近 0，且与 matched
threshold anchor 比较后并无稳定优势。因此当前三 seed 全部标记为 **diagnostic**。

### 5.4 解码策略结果

同一 Moment checkpoint 的 validation 结果：

| Decode | mAP | G-mIoU@3 | 结论 |
|---|---:|---:|---|
| full | 8.63 | 36.59 | 当前最平衡的单 checkpoint view |
| GREC threshold | 6.95 | 38.96 | G 提高但 mAP 明显下降 |
| HieA2G adaptive | 4.23 | 38.96 | count 截断严重破坏 localization recall |

因此 full 保留为 primary，threshold/adaptive 只作诊断。后续只有在 count uncertainty 足够可靠时，
adaptive 才可能重新进入候选比较。

### 5.5 跨 backbone staged/exploratory 轨

下表是截至本文更新时间的 validation best-mAP checkpoint；运行中的数字还可能变化：

| Track | 状态/epoch | best mAP | 同 checkpoint G@3 | 解释 |
|---|---:|---:|---:|---|
| QD plain | 已完成 | 7.32 | 2.60 | staged baseline |
| QD-GMR | e67，运行中 | 7.73 | 46.25 | balanced-G 50.54，但 multi acceptance 仅 2.22%，all-empty-like |
| QD-Quality | e11，运行中 | 7.95 | 46.42 | 尚无合格分组诊断，沿用失败 parent，不是候选 |
| QD-Dual | e10，运行中 | 7.14 | 46.40 | 同样受 rejection collapse 风险影响 |
| QD-Counter | e16，运行中 | 7.81 | 2.36 | 当前接近 accept-all，existence 尚未学会 |
| QD-HieA2M | e14，运行中 | 7.26 | 2.07 | 当前接近 accept-all，仍太早 |
| EaTR plain | 已完成 | 7.92 | 2.99 | staged baseline；另有早期高拒答/低 mAP checkpoint |
| EaTR-GMR | e16，运行中 | 7.04 | 44.14 | 高拒答风险，尚未通过分组门槛 |
| CG plain | e130，运行中 | 5.43 | 1.92 | 尚未早停 |

QD-GMR 的具体失败诊断是：null acceptance `1.43%`、single acceptance `49.70%`、multi
acceptance `2.22%`。虽然 `balanced-G=50.5391` 略过 50，但它几乎拒绝全部 multi positives；
这说明 prereg v2 中仅要求 single/multi acceptance `>0` 仍然过弱。v3 必须在查看新结果前
预先定义有意义的 acceptance retention gate，不能用“只剩一两个样本被接受”冒充 non-collapse。

另有一条 QD all-parameters-at-full-LR 的 14-epoch 轨被判定为 optimizer protocol 无效，目录保留
为 `qd_detr_gmr_invalid_all_lr`，不进入任何结果矩阵。

这些 staged 轨混用了 b128/b64、QD/CG `1e-4` baseline 与分阶段 `0.1x` shared LR，主要用于
快速机制诊断，不应称为论文严格复现。

### 5.6 Canonical source-derived baseline 轨

另起 `artifacts/canonical/`，采用：

- 全参数 `3e-5`；
- 200 epochs / patience 200；
- upstream 默认 batch size 32；
- source-derived SlowFast -> CLIP 输入顺序。

其中 b32 来自 upstream 默认，不是论文明确披露值，最终报告必须保留该限定。当前仍处于早期：

| Backbone | 当前 epoch | 当前已见 best-mAP / 同点 G@3 | 状态 |
|---|---:|---:|---|
| QD | 13 | 0.07 / 17.22 | 极早期，不下结论 |
| CG | 11 | 0.01 / 1.55 | 极早期，不下结论 |
| EaTR | 17 | 7.62 / 3.05 | 定位开始恢复，继续训练 |

## 6. 已确认的负结果与风险

### 6.1 方法负结果

- fixed n-gram Temporal-HMSA pilot 只把 mAP 从 `5.87` 提到 `6.00`，却把 `mR+@5` 从
  `2.33` 降到 `0.61`；不再扩大该 proxy。
- flat five-class HMSA+TAGC 曾得到 `G@3=44.73`，但 mAP 从 `5.87` 降到 `2.70`，属于
  null-dominated collapse。
- threshold/adaptive decoder 在当前 Moment checkpoint 上牺牲过多 mAP。
- QD simple-GMR 与 EaTR-GMR 的早期高 G 值主要来自强拒答，不能按 aggregate G 排名。
- 从 all-empty-like parent 启动 quality/dual 消融会继承错误 decision boundary；后续 strict
  组件轨应从合格 parent 或 matched plain checkpoint 重新启动。

### 6.2 实验与工程风险

- legacy-loss 与 strict-loss 结果不能混表；
- warm-start staged track 与 full-parameter paper-literal track 不能混表；
- validation threshold 扫描容易过拟合，若更改主 operating point，需 matched reference 与嵌套
  validation，而不是选择恰好过 gate 的窄阈值；
- QD/CG/EaTR 旧 raw-query 保存顺序经过 ranking/rounding，未保证与原始 query-index、未量化
  span 完全一致；exact replay schema 仍需完成；
- 当前 Moment fusion provenance 未完整绑定 producer checkpoint -> argv/source -> raw output ->
  calibration -> fused output -> metrics/diagnostics；
- prereg v2 尚不能安全承担 one-shot blind test。

## 7. 已完成的验证与代码质量工作

- 四 backbone strict mixed-vs-positive-only loss 等价测试覆盖 main/aux、quality、contrastive、
  event 与 dual；null gradient 为零，existence gradient 保留；
- strict 相关四套 CPU unittest 共 37 项通过；
- argparse abbreviation 安全测试 7/7 通过；
- calibration provenance/split 定向测试 10/10 通过；
- raw-query 基础 shape/range/probability/qid coverage 测试已建立；
- QD/CG resume 会拒绝 loss semantic 漂移，并允许 CPU/CUDA runtime device 解析变化；
- step-0 parent fidelity、checkpoint structure detection、optimizer group 审计已覆盖 QD/CG/EaTR；
- 最新完整集成测试需要在 prereg v3 与 raw exact-replay 修改合并后重新执行，当前不能用旧的
  “全套通过”数字替代最终验收。

## 8. 当前正在进行的工作

### 8.1 GPU 训练

当前共有 10 条 GPU 训练进程：

- staged：CG plain、QD-GMR、QD quality、QD dual、QD counter、QD HieA2M、EaTR-GMR；
- canonical：QD plain、CG plain、EaTR plain。

下一条优先启动的实验是 Moment seed-2023 strict HieA2M matched run；随后依次启动 strict
QD/CG/EaTR GMR。严格轨使用新目录，绝不覆盖 legacy artifacts。

### 8.2 校准与预注册安全升级

正在完成：

1. schema-versioned `{query_index, unrounded_span, foreground, quality}` raw-query contract；
2. validation primary decode 的 byte/metric exact replay equality；
3. calibration apply-only test pipeline，test 阶段不再重新搜索参数；
4. prereg v3 的固定 interpreter/source/feature inventory、四 backbone roster 与完整 static inputs；
5. trusted protocol registry、唯一 ledger root 与不可替代的 manifest digest；
6. 运行后 primary submission qid coverage、metrics finite/schema/protocol 检查；
7. three-seed mean/all-improve/non-collapse gates 与每 seed provenance closure。

## 9. 后续研究方向

### 9.1 最高优先级：真正解决 multi-positive rejection collapse

当前主要瓶颈不是如何让 null rejection 更强，而是如何在拒绝 null 的同时保留 multi positives。
建议按以下顺序研究：

1. 在 strict loss 下重新训练，先确认 null background gradient 是否是定位/decision 冲突来源；
2. 保留现有 inverse-sqrt multi 权重，比较 class-balanced BCE、focal BCE 与 logit adjustment；
3. 加入 group-robust validation objective，显式监控 null rejection、single recall、multi recall 的
   worst-group，而不是只优化 aggregate G@3；
4. 把 existence head 的证据从 max-pooled decoder slots 扩展为 query-set statistics，例如 top-k
   energy、slot entropy、cross-query agreement 与 encoder global evidence；
5. 对 multi positives 增加 set-level existence/coverage consistency，使“存在多个弱 query”不会被
   max pooling 误判为 null。

任何新 loss 都需要 matched ablation，不能同时更改阈值、训练权重与 decoder。

### 9.2 提升 multi-moment localization，而不是依赖 count 截断

可研究：

- Hungarian matching 后的 unmatched-positive coverage loss；
- query pair repulsion / DPP-style diversity，只惩罚高重叠且语义重复的预测；
- coverage-aware quality target，兼顾边界 IoU 与新增 GT 覆盖；
- 让 hierarchical counter 提供 soft cardinality prior，而不是硬截断 top-k；
- 当 count entropy 高时回退 full，只有高置信时才改变集合大小。

目标是提高 `mR+@5` 与 multi raw coverage；如果只提高 null G 而 multi 指标下降，应判为失败。

### 9.3 深化 DualGround，而不是增加更多无约束模块

优先做可归因消融：

1. sentence-only；
2. RPG；
3. RPG + Slot；
4. + DQA；
5. + EOS reconstruction；
6. 不同 phrase 数与 slot iteration。

重点分析哪些路径改善 query ranking、哪些路径改善边界、哪些路径只改变 existence。若某分支在
三个 seed 上无稳定增益，就不进入 full model。

### 9.4 更可靠的 validation calibration

- 使用 nested validation 或固定 calibration/confirmation split，避免同一 val 同时搜索和验收；
- 保存完整 grid，而不是只保存 best config；
- 所有 operating point 都必须与相同阈值协议的 matched reference 比较；
- 只允许 monotonic、可在 test 无标签重放的变换；
- calibration 选择时加入预先定义的 positive acceptance retention，而不是利用刚好大于 0 的
  multi acceptance 过门槛。

### 9.5 多目标 checkpoint selection

现有 objective-specific composition 表明 localization 与 rejection 的最优 epoch 不一致。后续可
比较：

- 单 checkpoint constrained Pareto selection；
- EMA/SWA 是否能减少两个目标的 epoch 偏移；
- 明确冻结的双头 composition；
- 共享 backbone + 独立 head fine-tuning。

双 checkpoint composition 只有在 producer provenance 完整闭合、三个 seed 同规则选择且 test
前冻结时才可作为正式方法。

### 9.6 跨 backbone 与第二阶段扩展

先在 Moment/QD/CG/EaTR 上确定一个真正通过 gate 的 frozen recipe，再考虑 MS-DETR、LD-DETR、
Sim-DETR。跨 backbone 报告应同时包含成功与失败移植，避免只挑最适合该模块的模型。

## 10. 下一阶段验收标准

一个配置只有在以下条件同时满足后，才可进入 blind test preregistration：

1. 三个 seeds 的 validation mean 同时 `Delta mAP > 0`、`Delta G@3 > 0`，目标均至少 `+1.0`；
2. 每个 seed 都通过 non-collapse gate，而不只是平均数通过；
3. `mR+@5` 相对 matched reference 不下降超过 `0.5`；
4. `balanced-G > 50`；
5. single/multi acceptance retention 使用在新结果产生前预先定义的实质门槛；
6. paired bootstrap 的两个主终点 CI 尽量同时高于 0；
7. strict loss、checkpoint、source、feature、argv、calibration、fusion、metrics 与 diagnostics
   provenance 全闭合；
8. prereg/runner 的攻击性回归全部通过；
9. test output root pristine，唯一 registry/ledger 未被领取；
10. 只按冻结 manifest 执行一次，失败、崩溃或中断也消耗该次执行，不按 test 结果换模型。

若最终没有任何配置通过这些条件，正确结论是报告 validation 负结果并保持 test 封存，而不是
降低门槛或挑选一个 aggregate G 较高的 all-empty-like 模型。

## 11. 关键代码与产物位置

- 总体计划：`plans/detr_hiea2m_research.md`
- 进度摘要：`PROGRESS.md`
- Moment 实现：`models/moment_detr_gmr/`、`training/moment_detr_gmr/`
- QD/CG/EaTR 实现：`methods/{qd_detr_gmr,cg_detr_gmr,eatr_gmr}/`
- 统一 evaluator：`eval/eval_main.py`、`eval/metrics.py`
- validation calibration：`scripts/calibrate_hiea2m.py`
- 分组诊断：`scripts/diagnose_gmr_groups.py`
- 双头 composition：`scripts/fuse_gmr_heads.py`
- prereg/runner：`scripts/preregister_detr_matrix_test.py`、
  `scripts/run_preregistered_detr_matrix_test.py`
- release anchor：`artifacts/anchors/moment_detr_gmr_release_val/`
- Moment 三 seed：`artifacts/runs/md_hiea2m_release_seed{2018,2023,2024}/`
- staged 矩阵：`artifacts/formal/`
- canonical 矩阵：`artifacts/canonical/`

