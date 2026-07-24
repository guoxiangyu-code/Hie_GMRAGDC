import os
import re

repo_root = '/home/guoxiangyu/generalized-moment-retrieval'
doc_path = os.path.join(repo_root, 'GMR_Progress_Report_Updated_20260723.md')

with open(doc_path, 'r', encoding='utf-8') as f:
    text = f.read()

# Extract Section 5 contents to put them into the right places
canonical_rows = [
    '| **Canonical seed2023** | 43.34 | 0.00 | 0.01 | 1.55 | `artifacts/canonical/cg_detr/seed2023/cg_detr/best_joint_val_metrics.json` | ❌ `-` |',
    '| **Canonical seed2023** | 69.10 | 69.95 | 4.00 | 45.50 | `artifacts/canonical/eatr/seed2023/eatr/best_joint_val_metrics.json` | ❌ `-` |',
    '| **Canonical seed2023** | 53.32 | 42.02 | 0.07 | 17.22 | `artifacts/canonical/qd_detr/seed2023/qd_detr/best_joint_val_metrics.json` | ❌ `-` |',
    '| **Canonical Restart seed2023** | 43.34 | 0.00 | 0.01 | 1.55 | `artifacts/canonical_b128_restart/cg_detr/seed2023/cg_detr/best_joint_val_metrics.json` | ❌ `-` |',
    '| **Canonical Restart seed2023** | 69.91 | 0.00 | 8.38 | 3.69 | `artifacts/canonical_b128_restart/eatr/seed2023/eatr/best_joint_val_metrics.json` | ❌ `-` |',
    '| **Canonical Restart seed2023** | 53.32 | 42.02 | 0.07 | 17.22 | `artifacts/canonical_b128_restart/qd_detr/seed2023/qd_detr/best_joint_val_metrics.json` | ❌ `-` |',
]
# Create canonical table
canonical_table = '''
### 0. 严控基线必要性：官方超参 Canonical 坍塌审计

为证明本研究中构建的 Strict 严控基线（统一大 Batch-size、统一学习率）的必要性，我们在原始官方超参下运行了模型，发生大面积坍塌：

| 方案 / 变体 | AUROC ↑ | Rej-F1@0.4 ↑ | mAP ↑ | G-mIoU@3 ↑ | 可追溯 JSON 文件 | 可追溯日志文件 |
|---|---|---|---|---|---|---|
''' + '\n'.join(canonical_rows) + '\n'

# Zero Head Multi-seed
zero_rows = [
    '| **+ Zero (seed2023)** | 72.86 | 1.86 | 9.16 | - | - | 3.08 | `artifacts/validation_selector_ablation/seed2023/stage2_zero/best_joint_soccer_gmr_val_preds_metrics.json` | ❌ `-` |',
    '| **+ Zero (seed2024)** | 59.53 | 0.00 | 9.16 | - | - | 2.65 | `artifacts/validation_selector_ablation/seed2024/stage2_zero/best_joint_soccer_gmr_val_preds_metrics.json` | ❌ `-` |',
    '| **+ Zero (seed2025)** | 73.61 | 0.94 | 9.16 | - | - | 2.87 | `artifacts/validation_selector_ablation/seed2025/stage2_zero/best_joint_soccer_gmr_val_preds_metrics.json` | ❌ `-` |',
    '| **+ Zero (seed2023_posw4)** | 63.46 | 0.00 | 9.16 | - | - | 2.65 | `artifacts/validation_selector_ablation/seed2023_posw4/stage2_zero/best_joint_soccer_gmr_val_preds_metrics.json` | ❌ `-` |',
]

# QD-DETR best_map
qd_rows = [
    '| **QD Strict GMR (Best mAP)** | 72.40 | 3.74 | 7.03 | - | - | 3.14 | `artifacts/strict_bsz32/qd_detr/seed2023/qd_detr_gmr/best_map_val_metrics.json` | ❌ `-` |',
    '| **QD + Quality (Best mAP)** | 72.50 | 70.24 | 7.56 | - | - | 42.36 | `artifacts/strict_bsz32/qd_detr/seed2023/qd_quality/best_map_val_metrics.json` | ❌ `-` |',
    '| **QD + Counter (Best mAP)** | 72.72 | 0.00 | 7.03 | - | - | 2.34 | `artifacts/strict_bsz32/qd_detr/seed2023/qd_counter/best_map_val_metrics.json` | ❌ `-` |',
    '| **QD + HieA2M (Best mAP)** | 72.40 | 0.00 | 6.80 | - | - | 2.19 | `artifacts/strict_bsz32/qd_detr/seed2023/qd_hiea2m/best_map_val_metrics.json` | ❌ `-` |',
]

# CG-DETR best_map
cg_rows = [
    '| **CG Strict GMR (Best mAP)** | 59.69 | 0.00 | 4.84 | - | - | 1.87 | `artifacts/strict_bsz32/cg_detr/seed2023/cg_detr_gmr/best_map_val_metrics.json` | ❌ `-` |',
    '| **CG + Quality (Best mAP)** | 59.75 | 0.00 | 4.95 | - | - | 1.79 | `artifacts/strict_bsz32/cg_detr/seed2023/cg_quality/best_map_val_metrics.json` | ❌ `-` |',
    '| **CG + Counter (Best mAP)** | 59.62 | 0.00 | 4.59 | - | - | 1.66 | `artifacts/strict_bsz32/cg_detr/seed2023/cg_counter/best_map_val_metrics.json` | ❌ `-` |',
    '| **CG + HieA2M (Best mAP)** | 59.68 | 0.00 | 4.65 | - | - | 1.86 | `artifacts/strict_bsz32/cg_detr/seed2023/cg_hiea2m/best_map_val_metrics.json` | ❌ `-` |',
    '| **CG + Phrase (Best mAP)** | 59.72 | 0.00 | 4.76 | - | - | 1.86 | `artifacts/strict_bsz32/cg_detr/seed2023/cg_phrase/best_map_val_metrics.json` | ❌ `-` |',
]

# Insert Canonical table before Section 1
text = text.replace('### 1. 两级判空与 Moment-DETR 主结果 (Moment-DETR Main & Parent)', canonical_table + '\n### 1. 两级判空与 Moment-DETR 主结果 (Moment-DETR Main & Parent)')

# Insert Zero multi-seed rows into Section 1 table
s1_end = text.find('artifacts/validation_selector_ablation/seed2023_posw1/stage2_zero/stdout.log` |')
if s1_end != -1:
    s1_end = text.find('\n', s1_end)
    text = text[:s1_end] + '\n' + '\n'.join(zero_rows) + text[s1_end:]

# Insert QD best_map rows into Section 5 table
s5_end = text.find('artifacts/strict_bsz32/qd_detr/seed2023/qd_hiea2m/train_log.jsonl` |')
if s5_end != -1:
    s5_end = text.find('\n', s5_end)
    text = text[:s5_end] + '\n' + '\n'.join(qd_rows) + text[s5_end:]

# Insert CG best_map rows into Section 6 table
s6_end = text.find('artifacts/strict_bsz32/cg_detr/seed2023/cg_phrase/train_log.jsonl` |')
if s6_end != -1:
    s6_end = text.find('\n', s6_end)
    text = text[:s6_end] + '\n' + '\n'.join(cg_rows) + text[s6_end:]

# Remove Section V (the previously appended unarchived section)
sec_5_idx = text.find('## 五、 artifacts 目录下未正式归档的附加实验发现')
if sec_5_idx != -1:
    text = text[:sec_5_idx].strip() + '\n'

with open(doc_path, 'w', encoding='utf-8') as f:
    f.write(text)

print('Successfully integrated results and removed unarchived section.')
