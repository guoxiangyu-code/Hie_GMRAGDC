import re

doc_path = '/home/guoxiangyu/generalized-moment-retrieval/GMR_Progress_Report_Updated_20260723.md'
with open(doc_path, 'r', encoding='utf-8') as f:
    content = f.read()

section8_md = """
## 八、事件级去重与集合选择器消融 (Event-Level Deduplication & Selector Ablation)

### 8.1 Moment-DETR 学习式事件去重与策略消融 (Learned Event Selector Ablation)

**设置**：Moment-DETR 固定候选，在 5 组 Selector Head 设置上取平均。

| 选择策略 (Selection Strategy) | AUROC ↑ | Rej-F1@0.4 ↑ | mAP ↑ | mR@3 ↑ | mR@5 ↑ | G-mIoU@3 ↑ | 平均选框数 | 可追溯说明与文件 |
|---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|---|
| **Direct Top-3 (Baseline)** | 69.55 | 7.66 | 6.48 | 8.91 | 8.91 | 4.95 | 3.00 | 直接按分数截断前 3 框 |
| **Learned Dedup Top-3** | 69.55 | 7.66 | **6.96 (+0.48)** | **9.25** | **9.25** | **5.06** | 3.00 | 🎉 **得胜策略**：去重后精度全面提升 |
| **Learned Dedup + Soft Count** | 69.55 | 7.66 | 6.72 | 11.20 | 11.20 | **7.25** | 5.94 | 动态选框，大幅提升召回与 G@3 |
| **Soft Count + Boundary Fusion** | 69.55 | 7.66 | 6.72 | 11.20 | 11.20 | **7.25** | 5.94 | 边界融合保持高召回 |

`artifacts/validation_selector_ablation/*/stage4_5_selection/learned_selector_ablation_summary.json`

---

### 8.2 Moment-DETR 纯几何去重消融 (Geometry-Only Dedup Ablation)

**设置**：Moment HieA2M Rerun v2 固定候选，相同输入。

| 去重策略 (Dedup Method) | mAP ↑ | mR@3 ↑ | G-mIoU@3 ↑ | 选框数 | 可追溯说明与文件 |
|---|:---:|:---:|:---:|:---:|---|
| **Direct Top-3 (Baseline)** | 9.16 | 13.85 | 15.54 | 3.00 | 原始候选分布 |
| **Hard-NMS (IoU=0.5)** | 9.16 | 13.85 | 15.54 | 3.00 | 几何抑制无额外增益 |
| **Linear Soft-NMS (IoU=0.5)** | 9.16 | 13.85 | 15.54 | 3.00 | 权重衰减效果一致 |
| **Complete-link Fusion (IoU=0.5)** | 9.16 | 13.85 | 15.54 | 3.00 | 聚类融合保持最高基线 |

`artifacts/validation_dedup_ablation/md_hiea2m_b128_rerun_from_best_v2_best_map/dedup_ablation_summary.json`

---

### 8.3 跨骨干架构事件去重与泛化性矩阵 (Cross-Backbone Deduplication Ablation)

**对比口径**：固定 Top-3 选框预算（$K=3$），对比 Direct Top-3（无去重）、Hard-NMS（IoU=0.5）、Hard-NMS（IoU=0.7）与 Cluster Fusion（IoU=0.7 聚类去重）。

| 骨干架构 (Backbone) | 去重策略 (Dedup Method) | mAP ↑ | G-mIoU@3 ↑ | mR+@3 ↑ (多事件召回) | 去重特性分析与可追溯文件 |
|---|---|:---:|:---:|:---:|---|
| **Flash-VTG (强 Transformer)** | **Direct Top-3 (无去重基线)** | 23.49 | 35.39 | 8.02 | `artifacts/flash_vtg_supplement/seed2023_bsz128/flash_vtg_gmr/dedup_ablation/dedup_ablation_summary.json` |
| | **Hard-NMS (IoU=0.5)** | 23.14 (↓0.35) | 35.66 | 6.24 (↓1.78) | ⚠️ 纯几何硬阈值误杀多事件候选框 |
| | **Hard-NMS (IoU=0.7)** | 23.88 (+0.39) | 35.65 | 8.13 (+0.11) | 宽松几何阈值 |
| | **Cluster Fusion (IoU=0.7)** | **24.28 (+0.79)** | **35.67 (+0.28)** | **8.56 (+0.54)** | 🎉 **全指标同步提升**：成功区分密集重复框与独立事件 |
| **EaTR (事件感知骨干)** | **Direct Top-3 (无去重基线)** | 6.96 | 16.82 | 1.81 | `artifacts/eatr_dgqc_transfer/seed2023/eatr_gmr_strict/dedup_ablation/dedup_ablation_summary.json` |
| | **Hard-NMS (IoU=0.5)** | 6.96 | 16.82 | 1.81 | 原始输出重叠率极低（<0.15%） |
| | **Cluster Fusion (IoU=0.7)** | 6.95 | 16.82 | 1.81 | 保持稳健精度，避开误杀 |
| **QD-DETR (Query-Dependent)** | **Direct Top-3 (无去重基线)** | 5.84 | 3.14 | 0.00 | `artifacts/strict_bsz32/qd_detr/seed2023/qd_detr_gmr/dedup_ablation/dedup_ablation_summary.json` |
| | **Hard-NMS (IoU=0.5)** | 5.82 (↓0.02) | 3.14 | 0.00 | Hard-NMS 产生轻微误杀 |
| | **Cluster Fusion (IoU=0.7)** | **5.84** | 3.14 | 0.00 | 稳健保留最高精度 |
| **QD-DETR + Quality** | **Direct Top-3 (无去重基线)** | 6.36 | 42.36 | 0.67 | `artifacts/strict_bsz32/qd_detr/seed2023/qd_quality/dedup_ablation/dedup_ablation_summary.json` |
| | **Hard-NMS (IoU=0.5)** | 6.32 (↓0.04) | 42.36 | 0.67 | NMS 产生误杀 |
| | **Cluster Fusion (IoU=0.7)** | **6.36** | 42.36 | 0.67 | 完全保留 Quality 带来的高基线定位收益 |
| **CG-DETR (Chunk-Guided)** | **Direct Top-3 (无去重基线)** | 3.50 | 1.87 | 0.00 | `artifacts/strict_bsz32/cg_detr/seed2023/cg_detr_gmr/dedup_ablation/dedup_ablation_summary.json` |
| | **Cluster Fusion (IoU=0.7)** | 3.50 | 1.87 | 0.00 | 维持平稳性能 |
"""

pattern = r'## 八、事件级去重与集合选择器消融.*?(?=## 三|\Z)|### 8\.1 跨骨干架构事件去重与泛化性矩阵.*?(?=## 三|\Z)'
if re.search(pattern, content, flags=re.DOTALL):
    content = re.sub(pattern, section8_md.strip() + '\n\n', content, flags=re.DOTALL)
else:
    content = content.replace('## 三、严谨性审计与追溯结论', section8_md.strip() + '\n\n## 三、严谨性审计与追溯结论')

with open(doc_path, 'w', encoding='utf-8') as f:
    f.write(content)

print("Section 8 formatted successfully!")
