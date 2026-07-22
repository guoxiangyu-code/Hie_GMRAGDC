# HieA2M-DG：DETR 系 GMR 改进与完整实验计划

最后更新：2026-07-22（Asia/Shanghai）

## 1. 目标与判定标准

目标是在 Soccer-GMR Standard 上，对同一 DETR backbone 的 GMR 基线同时提升：

1. 正查询定位 `mAP`；
2. 全查询端到端 `G-mIoU@1/@3`；
3. 不以牺牲 multi-moment 的 `mR+@5` 换取 null 样本收益。

公平比较对象必须是同 backbone、同输入特征、同数据、同 seed、同训练预算的
`*-GMR`，不是缺少拒答能力的 plain VMR 模型。一个配置只有满足以下条件才进入
test：

- 三个 seeds 的 validation mean 同时满足 `ΔmAP > 0` 与 `ΔG-mIoU@3 > 0`；
- 目标增益为两者均 `>= +1.0` absolute point；
- `mR+@5` 不下降超过 `0.5`；
- paired bootstrap 的两个主终点 95% CI 尽量同时高于 0；
- null/single/multi 分组证明收益不是 all-empty collapse。

test 只在模型、checkpoint 规则、阈值和后处理全部由 validation 冻结后运行一次。

## 2. GMR.pdf 的 DETR 覆盖范围

主表必须覆盖：

| 组别 | Base | GMR Adapter | 本方法 |
|---|---|---|---|
| Moment-DETR | 必做 | 必做 | 必做、完整消融 |
| QD-DETR | 必做 | 论文未报，新增 | 必做 |
| CG-DETR | 必做 | 论文未报，新增 | 必做、语义路径消融 |
| EaTR | 必做 | 必做 | 必做 |

因此论文主表的六个原有 DETR 条目是 Moment-DETR、QD-DETR、CG-DETR、EaTR、
Moment-DETR-GMR、EaTR-GMR；QD-/CG-DETR-GMR 是为了统一 adapter 公平性而新增。

第二阶段覆盖只在相关工作中出现的 MS-DETR、LD-DETR。Sim-DETR 作为强 multi-moment
扩展单列，不能替代 GMR.pdf 主表方法。FlashVTG 没有 DETR decoder，不计入本计划。

## 3. 固定实验协议

### 3.1 数据与特征

正式论文对比轨使用发布资产：

```text
data/label/Standard/{train,val,test}.jsonl
Soccer-GMR/feature/standard/{slowfast,clip,clip_text}
```

发布特征包含额外样本，但每个正式 split 必须验证 0 missing。输入顺序固定为
`[SlowFast(2304) || CLIP(512) || TEF(2)]`，每个视频 75 个 2 秒 token。

`features/f-lighthouse` 是独立重抽特征，不能与发布 checkpoint/论文数字混作同一轨；
如使用，所有 baseline 与方法均须从头同协议训练并单列为 feature-reproduction track。

### 3.2 两种文本输入协议

- `release-compat`：保留公开 checkpoint 使用的 32 行 CLIP token（含 padding），只用于
  精确复现论文 anchor。
- `clean-mask`：按 NPZ `attention_mask` 去除 padding，从头训练 matched baseline 和新方法。

不得把 clean-mask 输入直接喂给 release checkpoint 后与论文数比较；已测得该分布变化会
使公开权重 test 从 `7.52/35.84/32.89` 降到 `6.82/32.75/29.93`。

### 3.3 统一评估

- mAP/mR/mIoU：只在 positive queries 上计算；IoU `0.50:0.05:0.95`。
- AUROC/Rej-F1/G-mIoU：保留全部 null/positive queries。
- 主 operating point `τ=0.4`；`0.4/0.6/0.8` 均报告，但阈值选择只看 val。
- 论文原意的 mixed GMR 训练显式开启 `mask_null_vmr_loss`：null 样本的 Hungarian
  assignment 为空，定位分类/span/GIoU、aux、quality、DualGround 与 EaTR event loss 都乘
  `I(y=1)`；null 只监督 existence。默认 `false` 仅用于旧 checkpoint 兼容，不能把两种
  loss semantic 的结果混作同一轨。
- 同时保存 full-query、existence-gated、adaptive-count 三套解码，区分候选质量、拒答和计数。
- 连续边界与 2 秒 rounding 分别报告；后者用于 release anchor，前者作为预注册消融。

每个 run 保存：配置、命令、环境、git diff/commit、seed、checkpoint、初始化审计、梯度审计、
val/test submission、完整 metrics、计数混淆矩阵和文件 SHA256。

进入 test 前，Moment 双头配置由 `scripts/preregister_gmr_test.py` 冻结；跨骨干主表由
`scripts/preregister_detr_matrix_test.py` 一次性冻结。后者要求 candidate 的 mAP/G@3 双升、
mR+ 守门与非塌缩分组诊断全部通过，并锁定 checkpoint/source hash、精确 inference argv 和
互不重叠的 pristine output root。v2 manifest 将推理写成带依赖的 `execution_steps` DAG，
checkpoint role、test annotation、上游输出及预期输出均与真实 CLI flag 强绑定；Moment 的
localization/decision/fusion 属于同一 entry。冻结后只允许
`scripts/run_preregistered_detr_matrix_test.py` 原子领取全矩阵 one-shot ledger 并执行一次，
无论成功、失败或进程中断均不重跑，也不据 test 结果换权重。

## 4. 方法：HieA2M-DGQC

方法暂名 **Hierarchical Alignment-enhanced Adaptive Moment Grounding with Dual Grounding
and Quality Calibration (HieA2M-DGQC)**。

### 4.1 DualGround 时间化语义路径

在视频/文本 input projection 后、DETR encoder 前：

1. Sentence path：最后有效 CLIP EOS + 3 个 learned dummy；视频 clip 对其做 ACA，dummy
   吸收不相关 attention，只保留 EOS value 的门控贡献，再沿时间 self-attention。
2. Phrase path：3 个 recurrent phrase guides 聚合 lexical tokens；以其初始化 Slot Attention；
   追加 phrase-EOS 做 self-attention；phrase 与每个 clip Hadamard 交互并沿时间编码。
3. 融合：`V* = V + tanh(γs-γs0)Vs + tanh(γp-γp0)Vp`，两个 gate 在初始化时
   **严格为 0** 且导数为 1；公开 baseline warm-start 的 logits/span/existence 在 step-0
   逐位一致。
4. 约束：DQA 正交 phrase-to-word attention；EOS reconstruction 保持局部路径的全局语义。

这是真正进入 DETR memory 的 feature interaction，不是上一轮失败的 fixed n-gram auxiliary proxy。

### 4.2 IoU-quality query calibration

每个 final decoder query 额外预测 temporal IoU quality。Hungarian matched query 的 target 是
其预测 span 与分配 GT 的真实 IoU（target stop-gradient），unmatched query 为 0。推理分数：

```text
score = p_foreground^(1-alpha) * p_quality^alpha
```

它直接修复 DETR 分类置信度与边界质量不一致的问题，目标首先是提升 mAP 排序；同时压低
重复但未匹配的 query。

### 4.3 Hierarchical Adaptive Moment Counter

HieA2G 的 flat `{0,1,2,3,>3}` 在 Soccer train 的分布为
`2002/1423/565/117/31`，早期 pilot 已出现 0 类主导。主方法等价分解为：

```text
P(N=0) = 1 - P(exists)
P(N=k) = P(exists) P(N=k | exists), k in {1,2,3,4+}
```

- existence BCE 在所有 Soccer 样本训练；null 与 single 权重为 1，multi positive 继承相对
  inverse-sqrt count 权重并按 batch 总权重归一化，使拒答边界也收到长尾计数信号；
- conditional CE/ordinal loss 只在 positive 样本训练；
- positive count 使用 inverse-sqrt class weights；
- supervised count contrastive 仅作用于独立 count representation，低权重启用；
- detached soft foreground count 作为已有 detector evidence，低权重监督新 count head；原始
  GT 数量大于 4 时关闭该一致性，避免把 `4+` 错压成 4。

推理先做 quality-aware、temporal-diversity ranking：高置信 `1/2/3` 精确 top-count；`4+`
或不确定 count 回退验证阈值。必须同时比较 full、hard count、entropy-gated adaptive count，避免
错误截断损害 mAP。

### 4.4 GREC 阈值集合与最终分轨

GREC 证明固定 top-k 无法同时表达 no-target 与 multi-target；confidence threshold 才能动态决定
集合大小。因此验证阶段并列比较：

1. `full`：保留全部 DETR queries，仅排序与全局 existence gate；
2. `threshold`：GREC 式逐 query score threshold；
3. `adaptive`：先以 `P(exists)` 阈值判 0/非 0，再在 conditional `{1,2,3,4+}` 上计数，
   禁止对五类 joint posterior 直接 argmax；
4. `hard`：计数诊断上界，不作为默认主提交。

选择以 matched validation baseline 的 `min(mAP/ref_mAP, G@3/ref_G@3)` 为准；阈值协议不同的
结果不得共享同一个 G-mIoU reference。另设不可被该标量覆盖的非塌缩门槛：主候选必须同时
守住 matched baseline 的 mAP 与 mR+@5，`balanced-G` 必须高于全拒答的 50，且 single/multi
positive acceptance 必须显著非零。只靠 null rejection 得到的 harmonic 高分记为失败诊断；
若 simple-GMR 的 harmonic-best 触发该失败模式，下一阶段从通过这些门槛的 best-mAP 权重
warm-start，而不是从 harmonic-best warm-start。

## 5. Moment-DETR 实验顺序

### Stage M0：锚点与 matched baseline

- [x] 发布 Moment-DETR-GMR checkpoint + 官方发布特征 + release input。
- [x] 完整 val anchor：`mAP 8.14 / G-mIoU@3 33.21`（固定 `tau=.4`）。
- [ ] 从头三 seed `MD-base-clean`。
- [ ] 从头三 seed `MD-GMR-clean`，固定 clean-mask/continuous protocol。
- [ ] validation calibration：existence threshold、rounding、NMS/diversity。

### Stage M1：单变量归因

从同一 `MD-GMR-clean` checkpoint 分别训练：

1. `+Quality`；
2. `+Dual sentence only`；
3. `+RPG`；
4. `+RPG+Slot`；
5. `+RPG+Slot+DQA`；
6. `+RPG+Slot+DQA+EOS`（完整 DualGround）；
7. `+Hierarchical counter`，分别 full/hard/adaptive decode。

M1 gate：Quality 或完整 DualGround 至少有一个使 val mAP 提升且 G-mIoU 不明显下降；counter
使 G-mIoU@3 提升且 mAP/mR+ 守门通过。失败模块不进入组合。

当前 release-compatible screening（seed 2023）已得到：Quality `8.52/35.05`、Counter
`8.58/35.09`、DualGround `8.32/34.57` 的 joint-best（`mAP/G@3`）；这些运行发生在若干
审计修复前，作为模块筛选证据，最终组合使用修复后的代码重跑。

### Stage M2：组合与三 seed

- `GMR + successful localization branch + hierarchical counter`；
- phrase 数 `2/3/4`；quality alpha 与 diversity lambda 只在 val 网格搜索；
- 保存 mAP-best、G-mIoU-best、以及以 matched baseline 为 reference 的 Pareto-joint checkpoint；
- 三 seed + paired bootstrap；通过后冻结并运行 test 一次。

当前 validation-only 结果通过了聚合 mAP/G@3 双升，但尚未通过完整非塌缩门槛。由于 GMR 的
定位 decoder 与 existence adapter 本来就是并行预测头，候选 composition 采用：同一 seed
的 mAP-best checkpoint
只提供 ranked windows，joint-best checkpoint 只提供 existence/count；不平均、不替换边界或
分数。`scripts/fuse_gmr_heads.py` 对 qid 全覆盖、字段归属及 SHA256 做强审计。

| seed | mAP | G-mIoU@3 | mR+@5 |
|---:|---:|---:|---:|
| 2018 | 9.70 | 35.31 | 0.83 |
| 2023 | 10.34 | 36.90 | 1.17 |
| 2024 | 9.55 | 35.01 | 1.06 |
| mean | 9.86 | 35.74 | 1.02 |

发布 val anchor 为 `8.14/33.21/0.50`。seed-2023 相对 anchor 的 paired-bootstrap
`ΔmAP=+2.20 [0.76,3.70]`、`ΔG@3=+3.69 [1.87,5.64]`；相对同预算纯 GMR 续训并应用
相同 composition 后仍为 `+1.16/+2.47`，两个 CI 均高于 0。但三个 seed 在 `tau=.4` 的
`balanced-G` 分别只有 `38.7061/40.4929/38.3898`，均低于全拒答参照 50；且这些 checkpoint
沿用旧的 null-background classification semantic。因此它们只保留为筛选证据，不能冻结为
test candidate。下一轮必须使用显式 strict loss，并同时守住 positive acceptance 后再预注册。

## 6. 跨 DETR 移植

每个 backbone 先复现其 GMR.pdf base，再加统一 GMR adapter，然后移植 M2 胜出的组件：

- QD-DETR：DualGround memory 进入 query-dependent video representation 前；counter 接 final slots。
- CG-DETR：复用其 ACA/dummy sentence path，避免重复模块；新增 phrase path、quality、counter。
- EaTR：DualGround memory 进入 event reasoning；counter 接 final event/moment slots。
- MS-/LD-DETR：保持 multi-scale/loop decoder，只改 decoder memory 前与 final slots 后。

每个主表 backbone 至少跑 `base / +GMR / +localization winner / +counter / full`。Moment-DETR
和 CG-DETR 做完整消融，其余移植冻结后的最佳设计。

## 7. 必须报告的诊断

- count accuracy/macro-F1/ECE、每类 recall、预测数量分布、Count-MAE；
- null/single/multi 的 G-mIoU 与 positive acceptance、Null-FPR；
- raw-query oracle coverage、selected full coverage、duplicate rate；
- mAP@0.5/0.75/0.95、mR+@5、短/中/长 moment；
- WorldCup/SportsMoments、action type、query length/style；
- all-empty baseline（约 47.49 G-mIoU）与 Balanced-G，防止把 null collapse 当提升。

## 8. 已知负结果，不重复

相邻 HieA2G 工作区的 seed-2024 pilot 已证明：

- fixed n-gram Temporal-HMSA：`mAP 5.87 -> 6.00`，但 `mR+@5 2.33 -> 0.61`；
- flat five-class HMSA+TAGC：`mAP 5.87 -> 2.70`，虽然 `G-mIoU@3 5.18 -> 44.73`，
  收益几乎全部来自 null，multi coverage 崩溃。

因此本计划不再扩大 fixed n-gram proxy，也不把 flat 0-inclusive count 的总体 G-mIoU 当作成功。
