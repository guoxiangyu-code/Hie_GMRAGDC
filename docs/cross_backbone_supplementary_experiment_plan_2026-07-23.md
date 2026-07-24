# GMR 跨骨干补充实验计划

日期：2026-07-23  
目标：找到一套在 Moment-DETR、EaTR、QD-DETR、CG-DETR 上稳定有效，并可进一步迁移到 Flash-VTG 的精简 GMR 方法。

逐项状态、优先级与可执行命令见：
`docs/experiment_execution_matrix_2026-07-23.md`。

状态标记：

- `[完成]`：训练和 validation 评估已完成，且协议可用于当前比较；
- `[运行中]`：当前存在有效训练进程；
- `[已排队]`：代码、parent 和启动脚本已具备，将由持久化队列自动启动；
- `[失败/止损]`：曲线已有足够负面证据，已保留日志/checkpoint 后停止；
- `[探索性]`：已有结果，但 parent 或评估协议与当前严格配对协议不完全一致；
- `[未实现/阻塞]`：仍缺代码，或必须等待前序实验选择固定 parent；
- `[可启动/待资源]`：代码和协议原则上具备，但尚未占用当前训练资源。

## 1. 当前证据与问题

目前最清楚的结果是：

- Quality 在 EaTR 上六项主指标均超过 Strict GMR，在 QD-DETR 的早期训练中也同时提高 mAP、拒答和 G@3。
- Dual/Phrase grounding 在 EaTR 和 QD-DETR 上显著提高拒答与 G@3，但可能牺牲部分定位召回。
- Moment-DETR 的独立 Zero Head 明显提高 mAP、Rej-F1 和 G@3。
- 学习式 pairwise 去重在五组 head 设置中均超过相同候选输入上的 Direct Top-3。
- Counter、Soft Count 和 Boundary Fusion 尚未显示稳定的净收益。
- 完整 Quality + Dual + Counter 在 EaTR 上出现负交互，说明“模块越多越好”不成立。

因此，下一阶段不应继续盲目扩大完整组合，而应寻找跨架构公共核心。

## 2. 模块定义

为了避免把不同功能混在一起，后续统一使用以下缩写：

| 缩写 | 模块 | 作用 |
|---|---|---|
| B | Strict GMR baseline | 当前各骨干严格协议基线 |
| Q | Quality head | 学习候选边界质量，调整候选排序 |
| D | Dual/Phrase grounding | 建模文本片段或 dummy/phrase grounding |
| Z | Independent Zero verifier | 独立预测空查询，而不是复用 `1-p_exist` |
| P | Pairwise learned dedup | 判断两个候选是否属于同一事件 |
| C | Hierarchical Counter | 条件计数与数量先验 |
| F | Boundary Fusion | 同事件候选的边界融合 |

推荐的跨架构公共候选为：

> **U = B + Q + Z + P**

DETR 类骨干的增强候选为：

> **U-D = B + Q + D + Z + P**

Counter 和 Boundary Fusion 暂不进入默认组合，只有通过后述门槛才恢复。

## 3. 统一实验协议

### 3.1 配对原则

每个骨干内部必须满足：

1. 所有子方法从同一个 Strict GMR parent checkpoint 初始化。
2. 使用同一训练集、validation、特征、batch size、训练轮数、优化器和 early-stop 设置。
3. 新增模块采用零残差或等价初始化，保证 step 0 的 parent 输出不被破坏。
4. null 样本不进入 span、GIoU、quality 等定位损失。
5. 相同 seed 的不同 variant 使用相同数据顺序和固定随机种子。
6. 主表只比较 `best_joint`；`best_map` 和 `best_gmiou3` 只作为诊断。

### 3.2 checkpoint 选择

统一使用 baseline 归一化联合分数：

```text
J = sqrt((mAP / mAP_GMR) * (G@3 / G@3_GMR))
```

每个骨干的所有 variant 使用同一个 baseline reference。不能为不同方法手工选择不同的最优口径。

### 3.3 门限校准

门限必须在 validation 上确定并在 test 上冻结：

- `tau_gate`
- `tau_zero`
- `tau_veto`
- selector redundancy lambda
- selector stop threshold
- 是否启用 count/fusion

门限选择采用约束优化：

```text
最大化 validation G@3
约束：正样本放行率 >= 95%
      mAP >= baseline mAP - 0.10
```

同时报告 AUROC，避免结论只依赖某个固定阈值。

## 4. 实验漏斗

## 阶段 A：完成当前组件筛选

截至 2026-07-23 13:27 的实际状态：

| 骨干 | B | Q | D/Phrase | C | Q+D+C |
|---|---|---|---|---|---|
| Moment-DETR | `[完成]` | `[探索性]` | `[探索性]` | `[探索性]` | `[探索性]` |
| EaTR | `[完成]` | `[完成/晋级]` | `[完成/晋级]` | `[失败]` | `[失败]` |
| QD-DETR | `[完成]` | `[运行中]` | `[运行中]` | `[失败/止损]` | `[失败/止损]` |
| CG-DETR | `[完成]` | `[失败/止损]` | `[失败/止损]` | `[失败/止损]` | `[失败/止损]` |

Moment-DETR 的组件结果来自 release-parent 探索运行，不能与当前 strict baseline
直接配对，因此暂不据此宣布单组件正式成功或失败。若最终论文需要完整的四骨干
单组件严格矩阵，再从同一 strict parent 补跑；在 Q+D、Z、P 筛选完成前优先级较低。

阶段 A 的目标不是决定最终方法，而是决定哪些模块值得进入组合搜索。

### 单骨干晋级条件

一个模块进入下一阶段，至少满足：

1. `mAP >= baseline - 0.15`；
2. AUROC 提高至少 0.3，或 Rej-F1 提高至少 2.0；
3. G@3 提高至少 1.0；
4. 没有出现训练塌陷。

### 跨骨干晋级条件

- 至少三个骨干满足单骨干晋级条件；
- 其余骨干不允许 mAP 下降超过 0.20。

预计：

- Q：大概率晋级；
- D：作为拒答增强候选晋级；
- C：大概率淘汰；
- Q+D+C：只有在当前 QD/CG 最终结果反转时才保留。

## 阶段 B：补齐最关键的组合缺口

现有四个骨干均缺少最关键的：

> **Q + D，不含 Counter**

需要新增 variant：

- `md_quality_dual`
- `eatr_quality_dual`
- `qd_quality_dual`
- `cg_quality_phrase`

阶段 B 只使用 seed2023，比较：

| 编号 | Variant | 目的 |
|---|---|---|
| B0 | B | Strict GMR baseline |
| B1 | B+Q | Quality 单独贡献 |
| B2 | B+D | Dual/Phrase 单独贡献 |
| B3 | B+Q+D | 检查 Q 与 D 是否协同 |
| B4 | B+Q+D+C | 检查 Counter 是否造成负交互 |

如果 B3 优于 B1/B2，而 B4 下降，即可证明 Counter 是负交互来源，并将其从主方法移除。

## 阶段 C：移植独立判空 Z

目前只有 Moment-DETR 原生支持 Independent Zero Head。需要在 EaTR、QD-DETR、CG-DETR 上实现相同接口。

代码审计显示，当前 Moment-DETR 的 `IndependentZeroVerifier` 仍硬性读取
`counter_representation`，因此“Z 不含 Counter”不能只增加一个 variant 名称。
实施时必须先把 rich evidence encoder 从 Counter 的分类/序数头中解耦：

- evidence encoder 负责汇聚视频、文本、queries 和候选统计；
- Zero verifier 使用 evidence representation；
- Counter 只作为可选的数量预测 head；
-关闭 Counter 时，Z 的输入和计算图仍然完整；
-使用 stop-gradient 版本作为默认低风险配置。

这一步是验证“独立判空有效”而非“Counter 隐式参与有效”的必要条件。

Z 的输入统一由下列信息组成：

- pooled 视频—文本编码特征；
- decoder/query 特征；
-候选前景分数和 quality 分数统计；
-候选分数熵、top-1/top-2 margin；
-可选的 soft-count 特征，但不得直接把 `1-p_exist` 当作 Z。

先冻结 parent 与定位模块，只训练 Zero Head，降低筛选成本。

比较：

| 编号 | 设置 | 回答的问题 |
|---|---|---|
| Z0 | `1-p_exist` | 旧判空方式 |
| Z1 | Independent Z | 独立判空是否有效 |
| Z2 | Z，无救回 | 提升是否来自独立预测本身 |
| Z3 | Z + rescue | 冲突时救回是否减少正样本误杀 |
| Z4 | Z + rescue + veto | 高置信判空复核是否进一步改善 null |

必须额外报告：

- 正样本误拒率；
- null 拒答率；
- 第一层误拒后被第二层救回的样本数；
- 第一层放行后被 Z veto 的样本数；
- 冲突样本的 mAP 和 G@3；
- AUROC、Rej-F1、mAP、G@3。

阶段 C 最终在每个骨干保留一个 Z 配置。

## 阶段 D：学习式去重 P 的跨骨干验证

所有方法必须使用完全相同的候选输入，比较：

1. Direct Top-K；
2. Hard-NMS；
3. Complete-link geometry；
4. Learned Top-K；
5. Learned Soft Count；
6. Learned Soft Count + Fusion。

当前 learned pairwise selector 只在 Moment-DETR 接通。EaTR、QD-DETR 和
CG-DETR 虽然可以保存 raw queries，但尚不输出
`pred_same_event_probs` 和 `all_query_indices`。因此移植 P 需要补齐：

- pairwise head 与监督匹配；
-pairwise-only/head-only 训练 scope；
-raw-query 输出字段；
-cascade/learned selector evaluator；
-warm-start、模糊 pair 屏蔽和严格 checkpoint 加载测试。

先在每个骨干的最佳 `B+Q+Z` 或 `B+Q+D+Z` checkpoint 上生成固定候选，再训练 P。

P 的晋级条件：

- mAP 高于 Direct Top-K；
- mR@3 不下降；
- 至少三个骨干提升；
- 不允许通过输出更多框伪造 mAP 增益；
- paired query bootstrap 的 mAP 差值 95% CI 在最终 test 上不跨 0。

如果 Soft Count 只提高 mR@5/G@3、但降低 mAP，则保留为召回导向消融，不进入默认 U。

如果 Fusion 与 Soft Count 完全相同，则删除 F。

## 阶段 E：选出跨骨干最终候选

完成 A–D 后，只保留两个候选进入多种子验证：

| 候选 | 组成 | 定位 |
|---|---|---|
| U | B+Q+Z+P | 架构无关公共方法 |
| U-D | B+Q+D+Z+P | DETR 类增强方法 |

选择规则：

1. 每个骨干上 mAP 不劣于 B 超过 0.10；
2. 每个骨干上 G@3 高于 B；
3. 至少三个骨干上 mAP、AUROC、G@3 同时提高；
4. 优先选择最差骨干提升更大的方法，而不是平均值被某一个骨干拉高的方法；
5. 参数量和推理开销作为平局判据。

跨骨干稳健分数采用：

```text
R_b = min(mAP_variant / mAP_B, G@3_variant / G@3_B)
R_global = min over backbones(R_b)
```

先最大化 `R_global`，再比较平均 `J`。该规则会偏好真正跨骨干稳健的组合。

## 阶段 F：多随机种子正式验证

仅对 B 和最终胜出的 U（必要时加 U-D）运行：

- seed 2023
- seed 2024
- seed 2025

正式核心骨干：

1. Moment-DETR
2. EaTR
3. QD-DETR
4. CG-DETR

外部泛化骨干：

5. Flash-VTG

Flash-VTG 已实现并正在筛选 Q 和独立 Z；P 尚未移植。若 Flash-VTG 上最终
U 也成立，可以把结论从“跨 DETR 骨干”提升为“跨 proposal/query 架构”。

### 最低资源方案

四个核心骨干：

```text
4 backbones × 2 methods(B/U) × 2 additional seeds = 16 个新增正式训练
```

Flash-VTG：

```text
1 backbone × 2 methods × 3 seeds = 6 个正式训练
```

组件组合、Z 和 P 的 seed2023 筛选可以采用冻结 parent 的低成本训练。

## 阶段 G：冻结 test

进入 test 前必须生成 preregistration：

- 最终 variant；
- checkpoint 选择规则；
-全部阈值；
-selector 配置；
-seed；
-评估脚本 commit；
-预测文件哈希。

test 只运行冻结后的配置，不允许根据 test：

- 改阈值；
-换 checkpoint；
-改 K；
-启用/关闭模块；
-重新选择 seed。

最终报告：

- 三种子 mean ± std；
-每个 seed 的配对结果；
-query-level paired bootstrap 95% CI；
-mAP、mR@5、mR+@5、AUROC、Rej-F1、G@1/G@3；
-参数量、推理时延、平均输出框数。

## 5. 方法有效性的判定标准

### Quality 有效

满足：

- Q 相对 B 在至少三个核心骨干提高 mAP；
-四个骨干均不下降超过 0.10；
-最终 test 的 pooled paired bootstrap mAP 差值 CI 不跨 0。

可以声称：

> Quality-aware ranking 是跨骨干有效的定位质量建模。

### Independent Zero 有效

满足：

- B+Q+Z 相对 B+Q 提高 AUROC、Rej-F1 和 G@3；
-正样本误拒率不高于预设上限；
-多个骨干出现可重复的 rescue；
-test 上 paired G@3 CI 不跨 0。

可以声称：

> 独立判空复核优于复用 existence 概率，并缓解门限误杀。

### Learned Dedup 有效

满足：

-相同候选、相同输出预算下，P 超过 Direct Top-K；
-至少三个骨干 mAP 提高；
-重复率下降且 mR@3 不下降；
-test paired mAP CI 不跨 0。

可以声称：

> 学习式事件去重优于直接 Top-K 和纯几何抑制。

### Dual/Phrase 有效

满足：

- D 在相同正样本放行率下提高拒答率和 G@3；
-mAP 不下降超过 0.15；
-至少三个骨干成立。

可以声称：

> 细粒度文本 grounding 改善 generalized retrieval 的拒答—定位权衡。

### Counter 有效

只有同时满足下列条件才能保留：

-相对无 C 的相同组合提高 mAP 或 mR+@5；
-G@3 不下降；
-最优 `count_prior_weight > 0`；
-至少两个骨干成立。

否则结论应为：

> 当前 Counter 能学习数量标签，但未转化为稳定检索增益。

## 6. 推荐执行顺序

1. 等当前 QD/CG 八个组件实验完成并统一汇总。
2. 实现四骨干 `Q+D(no Counter)`，先做 step-zero 等价和 smoke test。
3. seed2023 运行 B/Q/D/Q+D/Q+D+C 配对矩阵。
4. 在四骨干移植独立 Z，先冻结 parent 做低成本消融。
5. 选择 `Q+Z` 或 `Q+D+Z` 父模型，进行相同候选输入的 P 消融。
6. 淘汰 C/F，确定 U 与可选 U-D。
7. 对 B/U 补 seed2024、2025。
8. 在 Flash-VTG 做外部架构验证。
9. 冻结配置并 preregister。
10. 一次性运行 test 和统计检验。

## 7. 预期最终结论

如果计划成功，最强且最稳妥的论文主张应为：

> Quality-aware localization、独立判空复核与学习式事件去重构成一个可插拔的 GMR 框架，在多个时刻定位骨干上同时改善定位与 generalized rejection。

若 Dual 只在部分 DETR 骨干有效，则将其定位为增强模块，而不是公共核心。

若 Counter/Fusion 继续失败，应如实作为消融结论删除，避免它们拖累跨骨干主结果。

## 8. 2026-07-23 实施与后台队列状态

### 已完成的代码与验证

- Flash-VTG validation 已改为始终保留 null，plain 对照使用最大候选分数作为
  existence proxy，不再误删 210 条 null validation query。
- Flash-VTG 已接统一 `eval/eval_main.py`，要求 qid 全覆盖，并在每个 epoch
  保存 `best_map`、`best_gmiou3`、`best_joint`。
- 已实现 Flash Candidate Quality head：
  - 监督目标为候选与最近 GT 的 temporal IoU；
  - null 不进入 Quality loss；
  - 排序分数为 foreground 与 quality 的几何校准。
- 已实现 Flash Independent Zero verifier：
  - 直接输出 `P(N=0)`；
  - 不读取 existence logit；
  - 输入视频、文本、候选分布、熵、间隔和候选几何统计。
- 已实现高召回 existence + 独立 Zero 的 rescue/veto cascade，预测文件同时保留
  `pred_exist_score_stage1`、`pred_zero_score` 和最终 `pred_exist_score`。
- 已支持冻结 Flash parent，仅训练 Q/Z；Q+Z 阶段可冻结已训练 Quality，
  只训练 Zero。
- 四个 Stage-B 无 Counter variant 已注册并通过 checkpoint/step-zero 测试：
  - `md_quality_dual`
  - `eatr_quality_dual`
  - `qd_quality_dual`
  - `cg_quality_phrase`
- Flash 完整架构在 RTX 3090 上通过 batch=128 smoke；Q+Z warm-start smoke
  的可训练参数为 0.445M，Quality/Zero loss 均有有效梯度。
- 相关 46 项回归测试全部通过。

### Flash 发布锚点

发布 Standard 特征和发布 GMR checkpoint 的 validation 严格复现为：

| 后处理 | AUROC | Rej-F1@0.4 | mAP | G@1 | G@3 | mR@5 | mR+@5 |
|---|---:|---:|---:|---:|---:|---:|---:|
| Raw | 73.95 | 62.53 | 26.01 | 40.19 | 33.93 | 34.88 | 12.93 |
| Hard-NMS 0.7 | 73.95 | 62.53 | 27.63 | 40.02 | 34.25 | 38.52 | 15.81 |

覆盖为 465/465；该表是 validation 锚点，不与论文 test 数字混用。

输出目录：

```text
artifacts/flash_vtg_supplement/release_gmr_val
```

### 当前运行和持久化执行顺序

当前只继续两个仍有改善价值的 Stage-A 任务：
`qd_quality`、`qd_dual`。CG-DETR 四个单组件/组合均已经止损。独立的持久化进程
`scripts/queue_supplementary_experiments.sh` 已启动，并按以下波次执行：

1. Flash plain / GMR 从头配对训练，seed2023，batch=128；
2. 发布 GMR parent 上冻结主干，分别训练 Flash Q 和 Independent Z；
3. 冻结已训练 Q，训练 Flash Q+Z；并行运行 QD Q+D(no Counter)；
4. 并行运行 CG Q+D(no Counter) 与 Moment Q+D(no Counter)；
5. 运行 EaTR Q+D(no Counter)。

队列状态、PID、各波日志位于：

```text
artifacts/supplementary_queue/seed2023
```

正式输出位于：

```text
artifacts/flash_vtg_supplement/seed2023_bsz128
artifacts/cross_backbone_stage_b/seed2023
```

Learned Dedup、其他骨干的解耦 Zero 和多种子没有盲目排入本轮队列：
它们的 parent 必须由上述 seed2023 筛选结果决定。提前运行会违反固定 parent
和相同候选协议，也会浪费当前已过载的 CPU 资源。

### 资源止损记录（2026-07-23 13:23）

为给 Flash plain/GMR 正式训练释放 CPU/GPU 资源，以下四项在保留全部
checkpoint、逐 epoch 指标和日志后提前停止：

| Variant | 停止时进度 | 单项历史峰值（可来自不同 epoch） | 判定 |
|---|---:|---|---|
| `qd_counter` | e18 | mAP 7.03；G@3 6.01；best joint 0.923 | Counter 未超过 matched baseline joint |
| `qd_hiea2m` | e12 | mAP 6.80；G@3 2.24；best joint 0.713 | Q+D+C 的 G@3 持续低于 baseline 3.14 |
| `cg_counter` | e13 | mAP 4.59；G@3 1.80；best joint 0.946 | mAP、G@3 和 joint 均未超过 baseline |
| `cg_hiea2m` | e12 | mAP 4.65；G@3 1.86；best joint 0.961 | 完整组合持续无净收益 |

继续保留 `qd_quality`、`qd_dual`。Flash plain/GMR 已分别在 GPU 0/1
以 batch=128 正常训练。

## 9. 逐阶段缺口盘点与下一步（2026-07-23 13:27）

### 9.1 已失败并停止的正式筛选实验

以下失败项不再恢复训练；产物没有删除，可以用于负结果分析：

| 骨干 | Variant | 最终处理 | 结论 |
|---|---|---|---|
| EaTR | `eatr_counter` | `[完成/失败]` | mAP 6.16、G@3 12.15，均明显低于 baseline |
| EaTR | `eatr_hiea2m` | `[完成/失败]` | mAP 7.08、G@3 9.56，完整组合出现负交互 |
| QD-DETR | `qd_counter` | `[失败/止损]` | e18 停止，best joint 0.923，未超过 matched baseline |
| QD-DETR | `qd_hiea2m` | `[失败/止损]` | e12 停止，G@3 峰值 2.24，低于 baseline 3.14 |
| CG-DETR | `cg_counter` | `[失败/止损]` | e13 停止，mAP/G@3/joint 均未超过 baseline |
| CG-DETR | `cg_hiea2m` | `[失败/止损]` | e12 停止，完整组合持续无净收益 |
| CG-DETR | `cg_quality` | `[失败/止损]` | e19 停止；latest mAP 3.97、G@3 1.49，best joint 约 0.957 |
| CG-DETR | `cg_phrase` | `[失败/止损]` | e19 停止；latest mAP 3.89、G@3 1.41，best joint 约 0.983 |

Counter/完整组合的六项失败共同支持：它们不应进入默认公共方法。新增的 CG
Quality/Phrase 失败进一步表明当前模块在 CG 骨干上没有迁移收益。
Moment-DETR 的 Counter/完整组合只属于旧 parent 的探索证据，尚不能计入这张
严格失败表。

### 9.2 代码已具备、当前正在运行或已排队

| 阶段 | 实验 | 状态 | 说明 |
|---|---|---|---|
| A | QD `Q`、`D` | `[运行中]` | 继续到 early stop/完成，用于单组件晋级 |
| Flash 配对锚点 | Flash plain、Flash GMR，seed2023 | `[运行中]` | 从头、同 seed、batch=128 的公平配对 |
| Flash Q/Z | Flash `Q`、独立 `Z` | `[已排队]` | release GMR parent 上冻结主干训练 head |
| Flash Q+Z | Flash 解耦 `Q+Z` | `[已排队]` | 只在 Q 和 Z 都产生有效 parent 后启动 |
| B | QD `Q+D(no C)` | `[已排队]` | queue wave 3 |
| B | CG `Q+Phrase(no C)`、Moment `Q+D(no C)` | `[已排队]` | queue wave 4 |
| B | EaTR `Q+D(no C)` | `[已排队]` | queue wave 5 |

因此，当前“已经实现、parent 明确、可公平启动”的 seed2023 核心缺口没有遗漏；
它们已经全部在运行或持久化队列中，不需要再开第二套抢资源的重复进程。

### 9.3 尚未补充、但必须等待前序结果的实验

| 阶段 | 尚缺实验 | 当前状态 | 解锁条件 |
|---|---|---|---|
| C | Moment 解耦 `Z(no C)` | `[未实现/阻塞]` | 拆分 evidence encoder 与 counter head |
| C | EaTR/QD/CG 独立 Z | `[未实现/阻塞]` | 统一 Z 接口，并选定各自 Q 或 Q+D parent |
| C | Z0–Z4 rescue/veto 校准 | `[阻塞]` | 先得到有效 Independent Z checkpoint |
| D | Flash/EaTR/QD/CG learned P | `[未实现/阻塞]` | 补 pairwise 输出与监督，并固定胜出 parent |
| D | 四/五骨干 Direct/NMS/geometry/P 公平消融 | `[阻塞]` | 固定同一候选文件和输出预算 |
| E | U 与 U-D 最终选择 | `[阻塞]` | A–D 的 seed2023 结果齐全 |
| F | 最终 U/U-D 的 seed2024、2025 | `[阻塞]` | 最终 variant 和全部阈值冻结 |
| G | test 与 bootstrap 统计 | `[阻塞]` | preregistration 完成后一次性运行 |

Moment-DETR 已有 seed2023/2024/2025 的旧 selector head-only 结果，可作为 P
的可行性证据，但它读取旧的 Counter representation，不等价于最终
`B+Q+Z+P`，不能代替阶段 D/F。

### 9.4 可以补跑但当前不应抢先启动的实验

严格 baseline 的 seed2024、2025 在 Moment-DETR、EaTR、QD-DETR、CG-DETR
上尚不完整；现有 `validation_selector_ablation/seed2024|2025` 和
`md_hiea2m_release_seed2024` 不能充当四骨干 strict baseline。

这批 baseline 不依赖最终 U，技术上属于 `[可启动/待资源]`。但当前两张 GPU
已经服务 seed2023 筛选和持久化后续波次。建议在当前队列完成、四骨干 strict
baseline 参数核对一致后，作为下一批后台任务统一启动，而不是现在与关键
seed2023 漏斗竞争资源。

### 9.5 下一次唤醒后的执行决策

1. 汇总 QD/CG 的 Q、D/Phrase 最终 `best_joint`，按阶段 A 门槛判定晋级；
2. 检查 Flash Q/Z 与四骨干 Q+D(no C)，淘汰不满足 mAP 非劣约束的组合；
3. 只对胜出 parent 实现和训练解耦 Z，避免在失败 parent 上重复开发；
4. Z 胜出后再移植 P，并完成 Direct Top-K 对照；
5. U 固定后，统一排四核心骨干 B/U 的 seed2024、2025；
6. 最后补 Flash 多种子与冻结 test。
