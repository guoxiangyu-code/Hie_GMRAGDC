# DETR-GMR Research Progress

最后更新：2026-07-22

> 清空对话上下文后的实验接管入口：
> `docs/two_experiment_tracks_runbook_2026-07-22.md`。该文档集中记录两大类实验的脚本、
> 已完成/运行中/待接续状态、输出目录、依赖顺序、续跑与 validation 评估命令。

## 已完成

- 完整审计 GMR/HieA2G/GREC/DualGround 四篇论文。
- 明确 GMR 主表 DETR 范围：Moment-DETR、QD-DETR、CG-DETR、EaTR，以及
  Moment-DETR-GMR/EaTR-GMR；MS-/LD-DETR 属第二阶段，Sim-DETR 单列扩展。
- 找到发布 Soccer-GMR 特征和 Moment-DETR-GMR checkpoint。
- 修复 NumPy 2.4 移除 `np.trapz` 导致的 evaluator 崩溃。
- 严格复现论文 Moment-DETR-GMR test anchor：

```text
AUROC 72.09, Rej-F1@0.4 64.01
mAP 7.52, mR@5 12.96, mR+@5 0.84
G-mIoU@1 35.84, G-mIoU@3 32.89
```

- 实现 DualGround temporal sentence/phrase paths、RPG、Slot refinement、DQA、EOS reconstruction。
- 实现 query IoU-quality head、quality/diversity ranking。
- 实现 hierarchical existence + positive-conditional `{1,2,3,4+}` counter 与 adaptive decoding。
- 实现 GREC `threshold`、HieA2G 两阶段 `adaptive` 与保守 `full` 三路解码；修复 joint-argmax
  导致的二次 null collapse，并加入 raw-count/计数混淆矩阵诊断。
- full-model mixed positive/null forward/backward smoke 通过；新增分支与共享 backbone 梯度均非零。
- 单元测试覆盖 DualGround、counter、empty batch、quality/diversity/adaptive decoding，及
  QD/EaTR/CG 的 step-0 parent 保真、checkpoint 结构检测与分阶段 optimizer 参数归属；
  strict-loss 专项回归进一步覆盖四骨干 main/aux/quality/contrastive/event/dual 与 exact-resume，
  最新 37 项专项测试全部通过。
- 训练 validation 已切换到完整 465 条 GMR evaluator，不再过滤 210 条 null 或使用旧 MR-only key。
- 发布权重完整 val anchor：`mAP 8.14 / G-mIoU@3 33.21`（release-compatible、rounding、tau=.4）。
- Moment-DETR HieA2M seed-2023 joint checkpoint：full `8.52 / 36.65`；仅在 val 上冻结的
  `alpha=.25, diversity=.5, full` 为 `8.63 / 36.59`，相对 anchor 两项同时提升。
- 利用 GMR 定位 decoder 与 existence adapter 的并行结构，新增可审计的 objective-specific
  head composition：mAP-best 只提供 windows，joint-best 只提供 existence/count；脚本严格检查
  qid 并记录输入/输出 SHA256。三个 seeds 的固定 `tau=.4` 结果为
  `9.70/35.31`、`10.34/36.90`、`9.55/35.01`，均值 `9.86/35.74`，相对发布 val anchor
  同时提升 `+1.72/+2.53`。
- seed-2023 paired bootstrap 相对发布 anchor 的增益为 `+2.20 mAP`（95% CI
  `[+0.76,+3.70]`）与 `+3.69 G@3`（`[+1.87,+5.64]`）；两项同时为正概率 0.9992。
- 完成同 seed、同共享层学习率的纯 GMR 续训控制：其相同 head composition 为
  `9.18/34.43`；HieA2M 净增 `+1.16/+2.47`，两个 95% CI 仍均高于 0。
- 同一 checkpoint 的 GREC threshold 最佳为 `6.95 / 38.96`，HieA2G adaptive 为
  `4.23 / 38.96`：二者牺牲 mAP，故只保留为诊断，主结果使用 full。
- 上述三 seed 沿用旧 checkpoint 兼容 loss semantic；论文原意的 strict 路径现已在
  Moment/QD/CG/EaTR 全部实现，正式命令必须显式传 `--mask-null-vmr-loss`。三个 fused 点在
  `tau=.4` 的 `balanced-G` 仅为 `38.7061/40.4929/38.3898`，均未过全拒答参照 50，故降级为
  validation diagnostic，尚不能冻结为 test candidate；门槛不会为迁就现有结果而放宽。
- 完成 QD-DETR、EaTR、CG-DETR 的隔离 MIT upstream pin、Soccer/null-safe baseline/GMR
  与统一 HieA2M 适配；三者的 positive/mixed/all-null GPU smoke、checkpoint 结构检测和
  parent step-0 保真通过。QD-DETR staged baseline 已完成（best-mAP checkpoint
  `7.32/2.60`）；EaTR staged baseline 也已完成（`7.92/2.99`）。QD-DETR-GMR、EaTR-GMR
  与 CG-DETR baseline 仍在运行，canonical b32 三条 plain baseline 也在独立目录继续训练。
- 三个跨骨干 trainer 现在显式记录 `optimizer_groups.json`（EaTR 写入 `run.json`）：GMR
  阶段只有新 existence head 使用全学习率，共享 backbone 使用 `0.1x`；后续组件阶段只把该
  阶段新增的 quality/dual/counter 参数放入全学习率组。一次未应用该分组的 QD 14-epoch
  轨已中止并保留为 `qd_detr_gmr_invalid_all_lr`，不进入结果矩阵。
- QD simple-GMR 已暴露典型 all-empty-like failure：数值 G@3 很高但 multi acceptance 极低；
  因此候选门槛加入 `balanced-G`、single/multi acceptance 与 mR+ 守门。hierarchical counter
  的 existence BCE 现继承相对 inverse-sqrt count 权重，使稀有 multi 正样本也能直接塑造
  拒答边界；null/single 保持单位权重。EaTR 隔离 criterion 遗漏的 count-weight 传递也已
  修正，并由构建级测试确认不再退化为全 1。
- 新增跨骨干 blind-test 预注册器原型。安全复审发现 v2 仍存在 argparse 缩写覆盖、解释器/
  模块与 feature 未闭包、自签 seal、任意 ledger、主输出不解析等 P0；CLI 缩写和 calibration
  provenance 已修，prereg/runner 正在升级，在攻击性回归全部通过前禁止领取 one-shot claim。
- raw-query calibration 已具备严格 shape/range/qid 与 producer/annotation/reference hash 绑定；
  仍需升级为 query-index、未 rounding/未量化的 exact-replay schema，并提供 apply-only test
  pipeline。
- 最新专项回归、`compileall` 与 `git diff --check` 均通过。QD simple-GMR 的
  epoch-20 joint checkpoint 已冻结为跨组件共同父模型（SHA256
  `195fbacd499170bb8205eccbb825bf44fc04161ee6c9b56f4ebc39e96958bdff`）；
  `qd_counter` 与 `qd_hiea2m` 已按同 seed/数据/基线参照启动正式训练。
- 复现口径复审发现原 `formal` 高吞吐轨混用了 b128/b64 与 QD/CG 的 `1e-4`，且
  patience=50；它们现保留为 exploratory/staged 轨。另起不覆盖旧结果的
  `artifacts/canonical/`：采用论文显式 `3e-5` 学习率、三个官方源码共同默认的 b32、
  200 epochs/patience=200，以及 source-derived SlowFast→CLIP 顺序。这里的 b32 来自
  upstream 默认而非论文披露，最终报告必须保留这一限定。QD/CG/EaTR 三条 canonical
  plain baseline 均已启动；后续还会把全参数 `3e-5` 的 paper-literal GMR 与共享层
  `0.1x` 的 staged fine-tune 明确分轨。

## 当前运行契约

- 发布 anchor：release-compatible 32-token padding + 2 秒 rounding。
- 当前可归因 warm-start 轨：官方发布 features + release-compatible text + 2 秒 rounding；
  clean-mask/continuous 必须从 matched clean baseline 起步，单列后续重训轨。
- 新模型对 release checkpoint 的 missing/unexpected keys 采用白名单迁移，并写
  `initialization_audit.json`；共享层任何不匹配均中止。

## 下一步

1. 完成 staged/canonical baseline，并启动显式 strict 的 Moment/QD/CG/EaTR GMR/HieA2M
   matched validation 矩阵；legacy-loss 轨只作诊断。
2. 对跨 backbone 执行 full/adaptive、计数与 null/single/multi 诊断，要求三 seed 聚合双升且
   每 seed 通过 non-collapse gates。
3. 闭合 raw-query exact replay、calibration provenance 与 prereg/runner P0，用攻击性回归证明
   冻结语义不可被缩写、影子模块、换 checkpoint 或换 ledger 绕过。
4. 只为真正通过门槛的配置生成冻结清单并执行一次 blind test；否则报告 validation 负结果并
   保持 test 封存。

完整计划见 `plans/detr_hiea2m_research.md`。
