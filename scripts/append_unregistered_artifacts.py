import os
import json

repo_root = '/home/guoxiangyu/generalized-moment-retrieval'
artifacts_dir = os.path.join(repo_root, 'artifacts')
doc_path = os.path.join(repo_root, 'GMR_Progress_Report_Updated_20260723.md')

with open(doc_path, 'r', encoding='utf-8') as f:
    doc_text = f.read()

core_files = [
    'best_joint_val_metrics.json',
    'best_joint_soccer_gmr_val_preds_metrics.json',
    'best_map_val_metrics.json',
    'strict_gmr_val_metrics_raw.json',
    'best_joint_hl_val_preds_metrics.json'
]

unregistered = []
for root, dirs, files in os.walk(artifacts_dir):
    for file in files:
        if file in core_files:
            full_path = os.path.join(root, file)
            rel_path = os.path.relpath(full_path, repo_root)
            if rel_path not in doc_text:
                unregistered.append(full_path)

def get_brief_data(p):
    try:
        with open(p, encoding='utf-8') as f:
            data = json.load(f)
        b = data.get('brief', data)
        if 'best_by_stage' in data:
            b = data['best_by_stage']['learned_topk']['brief']
        def fmt(k):
            v = b.get(k, b.get('GMR-' + k, b.get('MR-full-' + k, '-')))
            return f'{v:.2f}' if isinstance(v, (int, float)) else str(v)
        return {
            'auroc': fmt('AUROC'),
            'rej': fmt('Rej-F1@0.4'),
            'map': fmt('mAP'),
            'g3': fmt('G-mIoU@3')
        }
    except Exception as e:
        return None

results = []
for full_path in sorted(unregistered):
    if 'smoke' in full_path: continue
    
    if 'canonical' in full_path and 'best_joint' in full_path:
        d = get_brief_data(full_path)
        if d: results.append(('Canonical ' + os.path.basename(os.path.dirname(os.path.dirname(full_path))), d, os.path.relpath(full_path, repo_root)))
    
    if 'strict_bsz32' in full_path and 'best_map' in full_path:
        d = get_brief_data(full_path)
        if d: results.append(('Strict ' + os.path.basename(os.path.dirname(full_path)) + ' (Best mAP Ckpt)', d, os.path.relpath(full_path, repo_root)))
    
    if 'validation_selector_ablation' in full_path and 'stage2_zero' in full_path:
        d = get_brief_data(full_path)
        seed = os.path.basename(os.path.dirname(os.path.dirname(full_path)))
        if d: results.append(('Zero Head ' + seed, d, os.path.relpath(full_path, repo_root)))

md = '''

## 五、 artifacts 目录下未正式归档的附加实验发现

在全局扫描 `artifacts/` 目录时，发现了以下**已完成但未在主汇报表中登记**的补充实验数据（包括 `best_map` 检查点、`canonical` 历史训练版本以及验证集上的其他随机种子）：

| 实验组别 / 检查点特性 | AUROC ↑ | Rej-F1@0.4 ↑ | mAP ↑ | G-mIoU@3 ↑ | 未归档的 JSON 文件路径 |
|---|:---:|:---:|:---:|:---:|---|
'''

for name, d, path in results:
    md += f'| **{name}** | {d["auroc"]} | {d["rej"]} | {d["map"]} | {d["g3"]} | `{path}` |\n'

if '## 五、 artifacts 目录下未正式归档的附加实验发现' not in doc_text:
    with open(doc_path, 'a', encoding='utf-8') as f:
        f.write(md)
    print('Appended unregistered artifacts section.')
else:
    print('Already appended.')
