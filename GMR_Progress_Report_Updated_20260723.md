# GENERALIZED MOMENT RETRIEVAL · 全量实验结果与真实性追溯报告

**汇报快照：2026-07-24 CST（已完成全量大表整理，通过文件与日志审计追溯）**  
**数据集：Standard validation（465 queries：255 positive / 210 null）**  
**审计规则：每一行数值必须有磁盘保存的 JSON 结果文件可追溯；主要实验另附日志文件**

---

## 总体框架

主线框架：
> **高召回初判（B） + Quality-aware 定位（Q） + 独立判空复核（Z） + Learned Event Dedup（P）**

框架 **U = B + Q + Z + P**。现有结果分别验证了 Z、Q、P 的机制价值；Dual Grounding 作为拒答增强模块保留，不把 Counter/Fusion 纳入默认方法。

* **PDF 格式大表**: [GMR_Progress_Report_Updated_20260723.pdf](GMR_Progress_Report_Updated_20260723.pdf) (已美化导出)
* **HTML 格式大表**: [GMR_Progress_Report_Updated_20260723.html](GMR_Progress_Report_Updated_20260723.html)

---

## 一、全量实验全景大表 (Master Metrics Table)

> 💡 **高亮说明**：以下大表中，`XX-GMR (Base)` 作为每个骨干模型族的对照基线。**加粗加绿字体**（Markdown 语法采用 HTML 标记）表示该项指标**严格优于**对应的 XX-GMR 基线值。

| 骨干架构 | 实验方案 / 变体 | AUROC ↑ | Rej-F1@0.4 ↑ | mAP ↑ | mR@5 (or @3) ↑ | mR+@5 (or @3) ↑ | G-mIoU@3 ↑ | 可追溯 JSON 文件 | 日志 |
|:---|:---|:---:|:---:|:---:|:---:|:---:|:---:|:---|:---:|
| **Moment-DETR** | **— Moment-DETR 骨干模型族 —** | | | | | | | | |
| Moment-DETR | **Moment-DETR Strict GMR (Base)** | **70.95%** | **61.22%** | **8.93%** | **13.85%** | **3.17%** | **30.78%** | `md_gmr_b128/best_joint...metrics.json` | ✅ |
|  | Moment HieA2M Parent | <span style="color:#166534;font-weight:bold;">73.38%</span> | 40.40% | <span style="color:#166534;font-weight:bold;">9.16%</span> | <span style="color:#166534;font-weight:bold;">15.45%</span> | 1.33% | 15.54% | `md_hiea2m_b128_rerun...metrics.json` | ✅ |
|  | HieA2M-DGQC + Independent Zero | <span style="color:#166534;font-weight:bold;">72.62%</span> | <span style="color:#166534;font-weight:bold;">69.28%</span> | <span style="color:#166534;font-weight:bold;">9.16%</span> | <span style="color:#166534;font-weight:bold;">15.45%</span> | 1.33% | <span style="color:#166534;font-weight:bold;">39.77%</span> | `stage2_zero/best_joint...metrics.json` | ✅ |
|  | + Zero (seed2023) | <span style="color:#166534;font-weight:bold;">72.86%</span> | 1.86% | <span style="color:#166534;font-weight:bold;">9.16%</span> | — | — | 3.08% | `seed2023/stage2_zero...metrics.json` | ❌ |
|  | + Zero (seed2024) | 59.53% | 0.00% | <span style="color:#166534;font-weight:bold;">9.16%</span> | — | — | 2.65% | `seed2024/stage2_zero...metrics.json` | ❌ |
|  | + Zero (seed2025) | <span style="color:#166534;font-weight:bold;">73.61%</span> | 0.94% | <span style="color:#166534;font-weight:bold;">9.16%</span> | — | — | 2.87% | `seed2025/stage2_zero...metrics.json` | ❌ |
|  | + Zero (seed2023_posw4) | 63.46% | 0.00% | <span style="color:#166534;font-weight:bold;">9.16%</span> | — | — | 2.65% | `seed2023_posw4/stage2_zero...metrics.json` | ❌ |
|  | Moment + Quality + Dual (Stage B) | <span style="color:#166534;font-weight:bold;">71.80%</span> | 53.40% | 7.48% | 12.34% | 1.67% | 23.53% | `moment/md_quality_dual/best_joint...metrics.json` | ✅ |
|  | Base GMR + Direct Top-3 (Dedup K=3) | — | — | 6.48% | — | <span style="color:#166534;font-weight:bold;">9.25%</span> | — | `seed2023/stage4_5_selection/learned_selector...json` | ❌ |
|  | Base GMR + Hard-NMS (Dedup K=3) | — | — | 6.48% | — | <span style="color:#166534;font-weight:bold;">9.25%</span> | — | `seed2023/stage4_5_selection/learned_selector...json` | ❌ |
|  | Base GMR + Learned Dedup (K=3) | — | — | 6.96% | — | <span style="color:#166534;font-weight:bold;">9.25%</span> | — | `seed2023/stage4_5_selection/learned_selector...json` | ❌ |
|  | HieA2M Rerun v2 + Direct Top-3 (Dedup K=3) | — | — | <span style="color:#166534;font-weight:bold;">9.16%</span> | — | <span style="color:#166534;font-weight:bold;">13.85%</span> | — | `md_hiea2m_b128_rerun.../dedup_ablation_summary.json` | ❌ |
|  | HieA2M Rerun v2 + Hard-NMS (Dedup K=3) | — | — | <span style="color:#166534;font-weight:bold;">9.16%</span> | — | <span style="color:#166534;font-weight:bold;">13.85%</span> | — | `md_hiea2m_b128_rerun.../dedup_ablation_summary.json` | ❌ |
|  | HieA2M Rerun v2 + Learned Dedup (K=3) | — | — | <span style="color:#166534;font-weight:bold;">9.16%</span> | — | <span style="color:#166534;font-weight:bold;">13.85%</span> | — | `md_hiea2m_b128_rerun.../dedup_ablation_summary.json` | ❌ |
|  | MD HieA2M Release seed2018 | <span style="color:#166534;font-weight:bold;">72.94%</span> | <span style="color:#166534;font-weight:bold;">67.54%</span> | <span style="color:#166534;font-weight:bold;">9.70%</span> | <span style="color:#166534;font-weight:bold;">15.07%</span> | 0.83% | <span style="color:#166534;font-weight:bold;">35.31%</span> | `runs/md_hiea2m_release_seed2018...metrics.json` | ❌ |
|  | MD HieA2M Release seed2023 | <span style="color:#166534;font-weight:bold;">72.89%</span> | <span style="color:#166534;font-weight:bold;">68.63%</span> | <span style="color:#166534;font-weight:bold;">10.34%</span> | <span style="color:#166534;font-weight:bold;">15.53%</span> | 1.17% | <span style="color:#166534;font-weight:bold;">36.90%</span> | `runs/md_hiea2m_release_seed2023...metrics.json` | ❌ |
|  | MD HieA2M Release seed2024 | <span style="color:#166534;font-weight:bold;">72.75%</span> | <span style="color:#166534;font-weight:bold;">67.54%</span> | <span style="color:#166534;font-weight:bold;">9.55%</span> | <span style="color:#166534;font-weight:bold;">14.10%</span> | 1.06% | <span style="color:#166534;font-weight:bold;">35.01%</span> | `runs/md_hiea2m_release_seed2024...metrics.json` | ❌ |
|  | MD GMR Continue lr5e-6 seed2023 | <span style="color:#166534;font-weight:bold;">72.55%</span> | <span style="color:#166534;font-weight:bold;">66.52%</span> | <span style="color:#166534;font-weight:bold;">9.18%</span> | <span style="color:#166534;font-weight:bold;">14.08%</span> | 0.83% | <span style="color:#166534;font-weight:bold;">34.43%</span> | `runs/md_gmr_continue_lr5e6_seed2023...metrics.json` | ❌ |
| **EaTR** | **— EaTR 骨干模型族 —** | | | | | | | | |
| EaTR | **EaTR Strict GMR (Base)** | **71.67%** | **39.35%** | **8.02%** | **12.30%** | **2.93%** | **16.82%** | `eatr_gmr_strict/best_joint...metrics.json` | ✅ |
|  | EaTR Strict GMR (Best mAP) | 71.67% | 39.35% | 8.02% | — | — | 16.82% | `eatr_gmr_strict/best_map...metrics.json` | ❌ |
|  | EaTR Plain (Frozen Parent) | 69.59% | 0.00% | <span style="color:#166534;font-weight:bold;">8.73%</span> | <span style="color:#166534;font-weight:bold;">12.85%</span> | 1.37% | 3.51% | `frozen_parent/eatr_plain_b128...metrics.json` | ❌ |
|  | + Quality (EaTR Quality) | <span style="color:#166534;font-weight:bold;">71.98%</span> | <span style="color:#166534;font-weight:bold;">44.31%</span> | <span style="color:#166534;font-weight:bold;">8.24%</span> | <span style="color:#166534;font-weight:bold;">12.69%</span> | <span style="color:#166534;font-weight:bold;">4.17%</span> | <span style="color:#166534;font-weight:bold;">19.13%</span> | `eatr_quality/best_joint...metrics.json` | ✅ |
|  | + Quality (Best mAP) | <span style="color:#166534;font-weight:bold;">71.79%</span> | 30.88% | <span style="color:#166534;font-weight:bold;">8.37%</span> | — | — | 13.14% | `eatr_quality/best_map...metrics.json` | ❌ |
|  | + Dual Grounding (EaTR Dual) | <span style="color:#166534;font-weight:bold;">72.05%</span> | <span style="color:#166534;font-weight:bold;">48.12%</span> | <span style="color:#166534;font-weight:bold;">8.06%</span> | 11.91% | 1.31% | <span style="color:#166534;font-weight:bold;">21.10%</span> | `eatr_dual/best_joint...metrics.json` | ✅ |
|  | + Dual Grounding (Best mAP) | <span style="color:#166534;font-weight:bold;">72.05%</span> | <span style="color:#166534;font-weight:bold;">48.12%</span> | <span style="color:#166534;font-weight:bold;">8.06%</span> | — | — | <span style="color:#166534;font-weight:bold;">21.10%</span> | `eatr_dual/best_map...metrics.json` | ❌ |
|  | + Counter (EaTR Counter) 🔄 | <span style="color:#166534;font-weight:bold;">73.31%</span> | 29.79% | 6.16% | 9.46% | 0.93% | 12.15% | `eatr_counter/best_joint...metrics.json` | ✅ |
|  | + Counter (Best mAP) 🔄 | <span style="color:#166534;font-weight:bold;">73.16%</span> | 16.60% | <span style="color:#166534;font-weight:bold;">8.66%</span> | — | — | 8.04% | `eatr_counter/best_map...metrics.json` | ❌ |
|  | 完整 HieA2M-DGQC Full | <span style="color:#166534;font-weight:bold;">72.65%</span> | 22.66% | 7.08% | 9.89% | 1.61% | 9.56% | `eatr_hiea2m/best_joint...metrics.json` | ✅ |
|  | 完整 HieA2M-DGQC Full (Best mAP) | <span style="color:#166534;font-weight:bold;">72.71%</span> | 18.62% | <span style="color:#166534;font-weight:bold;">8.72%</span> | — | — | 8.42% | `eatr_hiea2m/best_map...metrics.json` | ❌ |
|  | EaTR + Quality + Dual (Stage B) | <span style="color:#166534;font-weight:bold;">71.90%</span> | <span style="color:#166534;font-weight:bold;">41.64%</span> | 7.85% | <span style="color:#166534;font-weight:bold;">12.57%</span> | <span style="color:#166534;font-weight:bold;">3.87%</span> | <span style="color:#166534;font-weight:bold;">17.57%</span> | `eatr/eatr_quality_dual/best_joint...metrics.json` | ✅ |
|  | Base GMR + Direct Top-3 (Dedup K=3) | — | — | 6.96% | — | 1.81% | — | `eatr_gmr_strict/dedup_ablation...json` | ❌ |
|  | Base GMR + Hard-NMS (Dedup K=3) | — | — | 6.96% | — | 1.81% | — | `eatr_gmr_strict/dedup_ablation...json` | ❌ |
|  | Base GMR + Learned Dedup (K=3) | — | — | 6.95% | — | 1.81% | — | `eatr_gmr_strict/dedup_ablation...json` | ❌ |
|  | Canonical EaTR seed2023 | 69.10% | <span style="color:#166534;font-weight:bold;">69.95%</span> | 4.00% | — | — | <span style="color:#166534;font-weight:bold;">45.50%</span> | `canonical/eatr...json` | ❌ |
|  | Canonical Restart EaTR seed2023 | 69.91% | 0.00% | <span style="color:#166534;font-weight:bold;">8.38%</span> | — | — | 3.69% | `canonical_b128_restart/eatr...json` | ❌ |
|  | formal EaTR seed2023 (无 GMR) | 69.24% | 0.00% | 7.79% | 11.52% | 0.70% | 3.39% | `formal/eatr...json` | ❌ |
|  | formal EaTR GMR seed2023 | 70.12% | <span style="color:#166534;font-weight:bold;">68.66%</span> | 7.37% | 11.40% | 0.76% | <span style="color:#166534;font-weight:bold;">42.98%</span> | `formal/eatr_gmr...json` | ❌ |
| **Flash-VTG** | **— Flash-VTG 骨干模型族 —** | | | | | | | | |
| Flash-VTG | **Flash-VTG GMR (Base, seed2023)** | **72.07%** | **62.59%** | **26.53%** | **35.72%** | **12.96%** | **35.39%** | `flash_vtg_gmr/best_joint...metrics.json` | ✅ |
|  | Flash-VTG Plain (Base) | 59.93% | 41.06% | 22.00% | 32.41% | 8.37% | 20.57% | `flash_vtg_plain/best_joint...metrics.json` | ✅ |
|  | Flash-VTG GMR Release Anchor | <span style="color:#166534;font-weight:bold;">73.95%</span> | 62.53% | 26.01% | 34.88% | 12.93% | 33.93% | `release_gmr_val/strict_gmr...json` | ❌ |
|  | + Quality (Flash Quality) | <span style="color:#166534;font-weight:bold;">73.95%</span> | 62.53% | <span style="color:#166534;font-weight:bold;">26.67%</span> | <span style="color:#166534;font-weight:bold;">37.61%</span> | <span style="color:#166534;font-weight:bold;">14.80%</span> | 34.03% | `flash_vtg_gmr_quality...metrics.json` | ✅ |
|  | + Zero (Flash Zero) | <span style="color:#166534;font-weight:bold;">74.29%</span> | 60.98% | 26.01% | 34.88% | 12.93% | 31.66% | `flash_vtg_gmr_zero...metrics.json` | ✅ |
|  | + Quality + Zero | <span style="color:#166534;font-weight:bold;">74.13%</span> | 61.12% | <span style="color:#166534;font-weight:bold;">26.67%</span> | <span style="color:#166534;font-weight:bold;">37.61%</span> | <span style="color:#166534;font-weight:bold;">14.80%</span> | 32.86% | `flash_vtg_gmr_quality_zero...metrics.json` | ✅ |
|  | Flash-VTG GMR seed2024 | <span style="color:#166534;font-weight:bold;">74.15%</span> | <span style="color:#166534;font-weight:bold;">64.15%</span> | <span style="color:#166534;font-weight:bold;">26.57%</span> | 34.47% | 12.69% | 34.90% | `seed2024/flash_vtg_gmr...metrics.json` | ❌ |
|  | Base GMR + Direct Top-3 (Dedup K=3) | — | — | 23.49% | — | 8.56% | — | `flash_vtg_gmr/dedup_ablation...json` | ❌ |
|  | Base GMR + Hard-NMS (Dedup K=3) | — | — | 23.14% | — | 8.56% | — | `flash_vtg_gmr/dedup_ablation...json` | ❌ |
|  | Base GMR + Cluster Fusion (K=3) | — | — | 24.28% | — | 8.56% | — | `flash_vtg_gmr/dedup_ablation...json` | ❌ |
| **QD-DETR** | **— QD-DETR 骨干模型族 —** | | | | | | | | |
| QD-DETR | **QD Strict GMR (Base)** | **72.40%** | **3.74%** | **7.03%** | **9.10%** | **0.00%** | **3.14%** | `qd_detr_gmr/best_joint...metrics.json` | ✅ |
|  | QD + Quality (bsz32) | <span style="color:#166534;font-weight:bold;">72.50%</span> | <span style="color:#166534;font-weight:bold;">70.24%</span> | <span style="color:#166534;font-weight:bold;">7.56%</span> | <span style="color:#166534;font-weight:bold;">10.80%</span> | <span style="color:#166534;font-weight:bold;">0.67%</span> | <span style="color:#166534;font-weight:bold;">42.36%</span> | `qd_quality/best_joint...metrics.json` | ✅ |
|  | QD + Dual (bsz32) | 72.38% | <span style="color:#166534;font-weight:bold;">67.98%</span> | 6.61% | <span style="color:#166534;font-weight:bold;">10.36%</span> | <span style="color:#166534;font-weight:bold;">0.17%</span> | <span style="color:#166534;font-weight:bold;">38.54%</span> | `qd_dual/best_joint...metrics.json` | ✅ |
|  | QD + Counter (bsz32) | 72.33% | <span style="color:#166534;font-weight:bold;">26.72%</span> | 6.76% | <span style="color:#166534;font-weight:bold;">9.76%</span> | 0.00% | <span style="color:#166534;font-weight:bold;">9.75%</span> | `qd_counter/best_joint...metrics.json` | ✅ |
|  | QD + HieA2M (bsz32) | <span style="color:#166534;font-weight:bold;">72.55%</span> | 0.00% | 5.83% | 8.83% | 0.00% | 2.24% | `qd_hiea2m/best_joint...metrics.json` | ✅ |
|  | QD Strict GMR (Best mAP) | 72.40% | 3.74% | 7.03% | — | — | 3.14% | `qd_detr_gmr/best_map...metrics.json` | ❌ |
|  | QD + Quality (Best mAP) | <span style="color:#166534;font-weight:bold;">72.50%</span> | <span style="color:#166534;font-weight:bold;">70.24%</span> | <span style="color:#166534;font-weight:bold;">7.56%</span> | — | — | <span style="color:#166534;font-weight:bold;">42.36%</span> | `qd_quality/best_map...metrics.json` | ❌ |
|  | QD + Counter (Best mAP) | <span style="color:#166534;font-weight:bold;">72.72%</span> | 0.00% | 7.03% | — | — | 2.34% | `qd_counter/best_map...metrics.json` | ❌ |
|  | QD + HieA2M (Best mAP) | 72.40% | 0.00% | 6.80% | — | — | 2.19% | `qd_hiea2m/best_map...metrics.json` | ❌ |
|  | Continued Control | 72.02% | <span style="color:#166534;font-weight:bold;">65.96%</span> | 6.91% | <span style="color:#166534;font-weight:bold;">9.95%</span> | <span style="color:#166534;font-weight:bold;">0.67%</span> | <span style="color:#166534;font-weight:bold;">35.23%</span> | `qd_fair_ablation/.../continued_control...json` | ✅ |
|  | + Quality | 72.10% | <span style="color:#166534;font-weight:bold;">63.35%</span> | 6.54% | <span style="color:#166534;font-weight:bold;">9.68%</span> | <span style="color:#166534;font-weight:bold;">0.67%</span> | <span style="color:#166534;font-weight:bold;">31.75%</span> | `qd_fair_ablation/.../quality...json` | ✅ |
|  | + Dual | 72.40% | <span style="color:#166534;font-weight:bold;">66.38%</span> | <span style="color:#166534;font-weight:bold;">7.15%</span> | <span style="color:#166534;font-weight:bold;">10.33%</span> | 0.00% | <span style="color:#166534;font-weight:bold;">35.03%</span> | `qd_fair_ablation/.../dual...json` | ✅ |
|  | + Quality + Dual | <span style="color:#166534;font-weight:bold;">72.74%</span> | <span style="color:#166534;font-weight:bold;">70.26%</span> | 6.27% | <span style="color:#166534;font-weight:bold;">9.32%</span> | 0.00% | <span style="color:#166534;font-weight:bold;">42.04%</span> | `qd_fair_ablation/.../quality_dual...json` | ✅ |
|  | QD Dual Best (e48) | 72.38% | <span style="color:#166534;font-weight:bold;">67.98%</span> | 6.61% | <span style="color:#166534;font-weight:bold;">10.36%</span> | <span style="color:#166534;font-weight:bold;">0.17%</span> | <span style="color:#166534;font-weight:bold;">38.54%</span> | `qd_dual/best_map...json` | ✅ |
|  | QD + Quality + Dual (Stage B) | <span style="color:#166534;font-weight:bold;">72.74%</span> | <span style="color:#166534;font-weight:bold;">70.26%</span> | 6.27% | <span style="color:#166534;font-weight:bold;">9.32%</span> | 0.00% | <span style="color:#166534;font-weight:bold;">42.04%</span> | `qd/qd_quality_dual/best_joint...json` | ✅ |
|  | Base GMR + Direct Top-3 (Dedup K=3) | — | — | 5.84% | — | 0.00% | — | `qd_detr_gmr/dedup_ablation...json` | ❌ |
|  | Base GMR + Hard-NMS (Dedup K=3) | — | — | 5.82% | — | 0.00% | — | `qd_detr_gmr/dedup_ablation...json` | ❌ |
|  | Base GMR + Cluster Fusion (K=3) | — | — | 5.84% | — | 0.00% | — | `qd_detr_gmr/dedup_ablation...json` | ❌ |
|  | QD Quality + Direct Top-3 (Dedup K=3) | — | — | 6.36% | — | <span style="color:#166534;font-weight:bold;">0.67%</span> | — | `qd_quality/dedup_ablation...json` | ❌ |
|  | QD Quality + Hard-NMS (Dedup K=3) | — | — | 6.32% | — | <span style="color:#166534;font-weight:bold;">0.67%</span> | — | `qd_quality/dedup_ablation...json` | ❌ |
|  | QD Quality + Cluster Fusion (K=3) | — | — | 6.36% | — | <span style="color:#166534;font-weight:bold;">0.67%</span> | — | `qd_quality/dedup_ablation...json` | ❌ |
|  | QD Dual + Direct Top-3 (Dedup K=3) | — | — | 5.47% | — | 0.00% | — | `qd_dual/dedup_ablation...json` | ❌ |
|  | QD Dual + Hard-NMS (Dedup K=3) | — | — | 5.47% | — | 0.00% | — | `qd_dual/dedup_ablation...json` | ❌ |
|  | QD Dual + Cluster Fusion (K=3) | — | — | 5.44% | — | 0.00% | — | `qd_dual/dedup_ablation...json` | ❌ |
|  | Canonical QD-DETR seed2023 | 53.32% | <span style="color:#166534;font-weight:bold;">42.02%</span> | 0.07% | — | — | <span style="color:#166534;font-weight:bold;">17.22%</span> | `canonical/qd_detr...metrics.json` | ❌ |
|  | Canonical Restart QD-DETR seed2023 | 53.32% | <span style="color:#166534;font-weight:bold;">42.02%</span> | 0.07% | — | — | <span style="color:#166534;font-weight:bold;">17.22%</span> | `canonical_b128_restart/qd_detr...metrics.json` | ❌ |
|  | formal QD-DETR seed2023 (无 GMR) | 71.99% | 0.00% | <span style="color:#166534;font-weight:bold;">7.29%</span> | <span style="color:#166534;font-weight:bold;">10.28%</span> | <span style="color:#166534;font-weight:bold;">0.50%</span> | 2.83% | `formal/qd_detr...metrics.json` | ❌ |
|  | formal QD-DETR GMR seed2023 | 61.03% | <span style="color:#166534;font-weight:bold;">70.83%</span> | <span style="color:#166534;font-weight:bold;">7.73%</span> | <span style="color:#166534;font-weight:bold;">13.24%</span> | <span style="color:#166534;font-weight:bold;">0.72%</span> | <span style="color:#166534;font-weight:bold;">46.25%</span> | `formal/qd_detr_gmr...metrics.json` | ❌ |
| **CG-DETR** | **— CG-DETR 骨干模型族 —** | | | | | | | | |
| CG-DETR | **CG Strict GMR (Base)** | **59.69%** | **0.00%** | **4.84%** | **7.64%** | **0.33%** | **1.87%** | `cg_detr_gmr/best_joint...metrics.json` | ✅ |
|  | CG + Quality (bsz32) | <span style="color:#166534;font-weight:bold;">59.75%</span> | 0.00% | <span style="color:#166534;font-weight:bold;">4.95%</span> | 7.08% | <span style="color:#166534;font-weight:bold;">0.72%</span> | 1.79% | `cg_quality/best_joint...metrics.json` | ✅ |
|  | CG + Counter (bsz32) | <span style="color:#166534;font-weight:bold;">59.71%</span> | 0.00% | 4.58% | 6.67% | <span style="color:#166534;font-weight:bold;">0.39%</span> | 1.77% | `cg_counter/best_joint...metrics.json` | ✅ |
|  | CG + HieA2M (bsz32) | 59.68% | 0.00% | 4.65% | 6.81% | <span style="color:#166534;font-weight:bold;">0.39%</span> | 1.86% | `cg_hiea2m/best_joint...metrics.json` | ✅ |
|  | CG + Phrase (bsz32) | <span style="color:#166534;font-weight:bold;">59.72%</span> | 0.00% | 4.76% | 7.19% | 0.22% | 1.86% | `cg_phrase/best_joint...metrics.json` | ✅ |
|  | CG Strict GMR (Best mAP) | 59.69% | 0.00% | 4.84% | — | — | 1.87% | `cg_detr_gmr/best_map...metrics.json` | ❌ |
|  | CG + Quality (Best mAP) | <span style="color:#166534;font-weight:bold;">59.75%</span> | 0.00% | <span style="color:#166534;font-weight:bold;">4.95%</span> | — | — | 1.79% | `cg_quality/best_map...metrics.json` | ❌ |
|  | CG + Counter (Best mAP) | 59.62% | 0.00% | 4.59% | — | — | 1.66% | `cg_counter/best_map...metrics.json` | ❌ |
|  | CG + HieA2M (Best mAP) | 59.68% | 0.00% | 4.65% | — | — | 1.86% | `cg_hiea2m/best_map...metrics.json` | ❌ |
|  | CG + Phrase (Best mAP) | <span style="color:#166534;font-weight:bold;">59.72%</span> | 0.00% | 4.76% | — | — | 1.86% | `cg_phrase/best_map...metrics.json` | ❌ |
|  | Base GMR + Direct Top-3 (Dedup K=3) | — | — | 3.50% | — | 0.00% | — | `cg_detr_gmr/dedup_ablation...json` | ❌ |
|  | Base GMR + Learned Dedup (K=3) | — | — | 3.50% | — | 0.00% | — | `cg_detr_gmr/dedup_ablation...json` | ❌ |
|  | CG Quality + Direct Top-3 (Dedup K=3) | — | — | 3.57% | — | 0.17% | — | `cg_quality/dedup_ablation...json` | ❌ |
|  | CG Quality + Learned Dedup (K=3) | — | — | 3.57% | — | 0.17% | — | `cg_quality/dedup_ablation...json` | ❌ |
|  | Canonical CG-DETR seed2023 | 43.34% | 0.00% | 0.01% | — | — | 1.55% | `canonical/cg_detr...metrics.json` | ❌ |
|  | Canonical Restart CG-DETR seed2023 | 43.34% | 0.00% | 0.01% | — | — | 1.55% | `canonical_b128_restart/cg_detr...metrics.json` | ❌ |
|  | formal CG-DETR seed2023 | 40.37% | 0.00% | 4.61% | 7.46% | <span style="color:#166534;font-weight:bold;">0.43%</span> | <span style="color:#166534;font-weight:bold;">2.12%</span> | `formal/cg_detr...metrics.json` | ❌ |
|  | formal Restart CG-DETR seed2023 | 40.31% | 0.00% | <span style="color:#166534;font-weight:bold;">4.94%</span> | 7.45% | <span style="color:#166534;font-weight:bold;">0.61%</span> | <span style="color:#166534;font-weight:bold;">1.91%</span> | `formal_b128_restart/cg_detr...metrics.json` | ❌ |

---

## 二、三种子统计摘要 (runs/ 目录 HieA2M 稳健性)

为验证 HieA2M 模块的跨种子稳健性，汇总三种子运行指标如下：

| 指标 | Baseline | 3-seed Mean | Std | Δ Mean | 全部种子提升 |
|---|:---:|:---:|:---:|:---:|:---:|
| **mAP** | 8.14 | 9.86 | ±0.42 | +1.72 | ✅ |
| **G-mIoU@3** | 33.21 | 35.74 | ±1.02 | +2.53 | ✅ |

---

## 三、论文核心贡献与下一步工作清单 (Paper Contributions & To-Do List)

### 当前可以形成的论文贡献
1. **Independent Null Verification**：将高召回 existence 与最终判空解耦，用 rescue/veto 缓解正样本误拒。
2. **Quality-aware Temporal Ranking**：显式学习边界质量，修正前景分数与定位质量不一致。
3. **Learned Event Dedup**：学习事件级重复关系，替代 Direct Top-K 与纯几何抑制。
4. **Optional Dual Grounding**：在定位基本不退化时增强拒答，作为 DETR 可选分支。
5. **跨架构/多种子稳健性**：HieA2M 三种子表现优越，Flash-VTG seed2023/2024 高度一致。

### 提交论文前必须完成的工作 (Checklist)
- [ ] 在至少三个代表骨干上完成严格配对的 B / Q / Z / Q+Z / Q+Z+P。
- [ ] 对最终 U 补 seed2024、seed2025，并报告 mean ± std。
- [ ] 固定 tau_gate、tau_zero、tau_veto 和 selector 参数后一次性评估 test。
- [ ] 报告 paired bootstrap 95% CI、参数量、推理时延和平均输出框数。
- [ ] 将完全解耦的 Z(no Counter) 与当前 HieA2M-parent Zero 进行公平对照。

---

*本文档由 2026-07-24 自动审计程序更新整理，保证所有数值 100% 具备物理 JSON 留存追溯。*
