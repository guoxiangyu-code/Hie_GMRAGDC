#!/usr/bin/env python3
"""Generate a current-state GMR experiment master table (HTML + PDF source).

Completed rows use their saved best checkpoint metrics. Running/interrupted rows
are explicitly labelled as best-so-far so they cannot be mistaken for final
results.
"""

from __future__ import annotations

import html
import json
import re
from datetime import datetime
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
ART = ROOT / "artifacts"
STAMP = datetime.now().astimezone().strftime("%Y-%m-%d %H:%M %Z")
DATE = datetime.now().astimezone().strftime("%Y-%m-%d")


def read_metrics(rel: str | None) -> dict:
    if not rel:
        return {}
    path = ROOT / rel
    if not path.exists():
        return {}
    data = json.loads(path.read_text())
    if isinstance(data.get("GMR-unified"), dict):
        return data["GMR-unified"].get("brief", {})
    brief = data.get("brief", {})
    return brief if isinstance(brief, dict) else {}


def epoch_from(directory: str) -> str:
    root = ROOT / directory
    best = None
    for path in list(root.glob("*.log")) + list(root.glob("*.jsonl")):
        try:
            text = path.read_text(errors="ignore")
        except OSError:
            continue
        vals = [int(x) for x in re.findall(r'"epoch"\s*:\s*(\d+)', text)]
        vals += [int(x) for x in re.findall(r"(?:Epoch|epoch)[\s:=\[]+(\d+)", text)]
        if vals:
            best = max(max(vals), best or 0)
    return f"e{best}" if best is not None else "—"


def mget(m: dict, key: str):
    aliases = {
        "AUROC": ("AUROC", "GMR-AUROC"),
        "Rej-F1@0.4": ("Rej-F1@0.4", "GMR-Rej-F1@0.4"),
        "mAP": ("mAP", "GMR-mAP"),
        "mR@5": ("mR@5", "GMR-mR@5"),
        "mR+@5": ("mR+@5", "GMR-mR+@5"),
        "G-mIoU@3": ("G-mIoU@3", "GMR-G-mIoU@3"),
    }
    for k in aliases[key]:
        if k in m:
            return float(m[k])
    return None


BASE = {
    "Moment-DETR": "artifacts/formal_strict/moment_detr/seed2023/md_gmr_b128/best_joint_soccer_gmr_val_preds_metrics.json",
    "EaTR": "artifacts/eatr_dgqc_transfer/seed2023/eatr_gmr_strict/best_joint_val_metrics.json",
    "QD-DETR": "artifacts/strict_bsz32/qd_detr/seed2023/qd_detr_gmr/best_joint_val_metrics.json",
    "CG-DETR": "artifacts/strict_bsz32/cg_detr/seed2023/cg_detr_gmr/best_joint_val_metrics.json",
    "FlashVTG": "artifacts/flash_vtg_supplement/release_gmr_val/strict_gmr_val_metrics_raw.json",
}


ROWS = [
    # backbone, method, status, metric file, run directory, note
    ("Moment-DETR", "GMR baseline (matched)", "完成", BASE["Moment-DETR"],
     "artifacts/formal_strict/moment_detr/seed2023/md_gmr_b128", "同骨干严格训练基线"),
    ("Moment-DETR", "HieA2M-DGQC", "完成",
     "artifacts/validation_selector_ablation/seed2023_posw1/stage2_zero/stdout.log#epoch13-best-joint",
     "artifacts/validation_selector_ablation/seed2023_posw1/stage2_zero", "当前主方法"),
    ("Moment-DETR", "Quality + Dual gate", "完成",
     "artifacts/cross_backbone_stage_b/seed2023/moment/md_quality_dual/best_joint_soccer_gmr_val_preds_metrics.json",
     "artifacts/cross_backbone_stage_b/seed2023/moment/md_quality_dual", "Stage-B 迁移组合"),
    ("EaTR", "GMR baseline", "完成", BASE["EaTR"],
     "artifacts/eatr_dgqc_transfer/seed2023/eatr_gmr_strict", "同骨干比较基线"),
    ("EaTR", "Quality", "完成",
     "artifacts/eatr_dgqc_transfer/seed2023/eatr_quality/best_joint_val_metrics.json",
     "artifacts/eatr_dgqc_transfer/seed2023/eatr_quality", "单模块"),
    ("EaTR", "Dual gate", "完成",
     "artifacts/eatr_dgqc_transfer/seed2023/eatr_dual/best_joint_val_metrics.json",
     "artifacts/eatr_dgqc_transfer/seed2023/eatr_dual", "双门控"),
    ("EaTR", "Counter", "完成",
     "artifacts/eatr_dgqc_transfer/seed2023/eatr_counter/best_joint_val_metrics.json",
     "artifacts/eatr_dgqc_transfer/seed2023/eatr_counter", "条件计数"),
    ("EaTR", "HieA2M-DGQC", "完成",
     "artifacts/eatr_dgqc_transfer/seed2023/eatr_hiea2m/best_joint_val_metrics.json",
     "artifacts/eatr_dgqc_transfer/seed2023/eatr_hiea2m", "完整方法迁移"),
    ("EaTR", "Quality + Dual gate", "运行中",
     "artifacts/cross_backbone_stage_b/seed2023/eatr/eatr_quality_dual/best_joint_val_metrics.json",
     "artifacts/cross_backbone_stage_b/seed2023/eatr/eatr_quality_dual", "临时最优，最终值待定"),
    ("QD-DETR", "GMR baseline", "完成", BASE["QD-DETR"],
     "artifacts/strict_bsz32/qd_detr/seed2023/qd_detr_gmr", "同骨干比较基线"),
    ("QD-DETR", "Quality", "完成",
     "artifacts/strict_bsz32/qd_detr/seed2023/qd_quality/best_joint_val_metrics.json",
     "artifacts/strict_bsz32/qd_detr/seed2023/qd_quality", "单模块"),
    ("QD-DETR", "Dual gate", "运行中",
     "artifacts/strict_bsz32/qd_detr/seed2023/qd_dual/best_joint_val_metrics.json",
     "artifacts/strict_bsz32/qd_detr/seed2023/qd_dual", "临时最优，逐轮旧任务"),
    ("QD-DETR", "Quality + Dual gate", "完成",
     "artifacts/cross_backbone_stage_b/seed2023/qd/qd_quality_dual/best_joint_val_metrics.json",
     "artifacts/cross_backbone_stage_b/seed2023/qd/qd_quality_dual", "Stage-B 已完成，5 epoch 验证"),
    ("QD-DETR", "Counter", "中止",
     "artifacts/strict_bsz32/qd_detr/seed2023/qd_counter/best_joint_val_metrics.json",
     "artifacts/strict_bsz32/qd_detr/seed2023/qd_counter", "早期走势无优势；数值仅供诊断"),
    ("QD-DETR", "HieA2M", "中止",
     "artifacts/strict_bsz32/qd_detr/seed2023/qd_hiea2m/best_joint_val_metrics.json",
     "artifacts/strict_bsz32/qd_detr/seed2023/qd_hiea2m", "早期走势无优势；数值仅供诊断"),
    ("QD-DETR", "FAIR · Continued control", "运行中",
     "artifacts/qd_fair_ablation/seed2023_bsz32/continued_control/best_joint_val_metrics.json",
     "artifacts/qd_fair_ablation/seed2023_bsz32/continued_control", "公平矩阵：不开新增模块"),
    ("QD-DETR", "FAIR · Quality", "运行中",
     "artifacts/qd_fair_ablation/seed2023_bsz32/quality/best_joint_val_metrics.json",
     "artifacts/qd_fair_ablation/seed2023_bsz32/quality", "公平矩阵：仅 Quality"),
    ("QD-DETR", "FAIR · Dual", "运行中",
     "artifacts/qd_fair_ablation/seed2023_bsz32/dual/best_joint_val_metrics.json",
     "artifacts/qd_fair_ablation/seed2023_bsz32/dual", "公平矩阵：仅 Dual"),
    ("QD-DETR", "FAIR · Quality + Dual", "运行中",
     "artifacts/qd_fair_ablation/seed2023_bsz32/quality_dual/best_joint_val_metrics.json",
     "artifacts/qd_fair_ablation/seed2023_bsz32/quality_dual", "公平矩阵：Quality 与 Dual 组合"),
    ("CG-DETR", "GMR baseline", "完成", BASE["CG-DETR"],
     "artifacts/strict_bsz32/cg_detr/seed2023/cg_detr_gmr", "同骨干比较基线"),
    ("CG-DETR", "Quality", "失败/中止",
     "artifacts/strict_bsz32/cg_detr/seed2023/cg_quality/best_joint_val_metrics.json",
     "artifacts/strict_bsz32/cg_detr/seed2023/cg_quality", "全线无收益，停止占用资源"),
    ("CG-DETR", "Phrase", "失败/中止",
     "artifacts/strict_bsz32/cg_detr/seed2023/cg_phrase/best_joint_val_metrics.json",
     "artifacts/strict_bsz32/cg_detr/seed2023/cg_phrase", "全线无收益，停止占用资源"),
    ("CG-DETR", "Counter", "失败/中止",
     "artifacts/strict_bsz32/cg_detr/seed2023/cg_counter/best_joint_val_metrics.json",
     "artifacts/strict_bsz32/cg_detr/seed2023/cg_counter", "全线无收益，停止占用资源"),
    ("CG-DETR", "HieA2M", "失败/中止",
     "artifacts/strict_bsz32/cg_detr/seed2023/cg_hiea2m/best_joint_val_metrics.json",
     "artifacts/strict_bsz32/cg_detr/seed2023/cg_hiea2m", "全线无收益，停止占用资源"),
    ("FlashVTG", "GMR release anchor", "完成", BASE["FlashVTG"],
     "artifacts/flash_vtg_supplement/release_gmr_val", "迁移实验固定比较锚点"),
    ("FlashVTG", "Plain from scratch", "运行中",
     "artifacts/flash_vtg_supplement/seed2023_bsz128/flash_vtg_plain/best_joint_hl_val_preds_metrics.json",
     "artifacts/flash_vtg_supplement/seed2023_bsz128/flash_vtg_plain", "参照项；不计作 GMR 改进"),
    ("FlashVTG", "GMR from scratch", "运行中",
     "artifacts/flash_vtg_supplement/seed2023_bsz128/flash_vtg_gmr/best_joint_hl_val_preds_metrics.json",
     "artifacts/flash_vtg_supplement/seed2023_bsz128/flash_vtg_gmr", "临时最优，最终值待定"),
    ("FlashVTG", "GMR + Quality", "完成",
     "artifacts/flash_vtg_supplement/seed2023_bsz128/flash_vtg_gmr_quality/best_joint_hl_val_preds_metrics.json",
     "artifacts/flash_vtg_supplement/seed2023_bsz128/flash_vtg_gmr_quality", "早停完成"),
    ("FlashVTG", "GMR + Zero head", "完成",
     "artifacts/flash_vtg_supplement/seed2023_bsz128/flash_vtg_gmr_zero/best_joint_hl_val_preds_metrics.json",
     "artifacts/flash_vtg_supplement/seed2023_bsz128/flash_vtg_gmr_zero", "早停完成"),
    ("FlashVTG", "GMR + Quality + Zero", "完成",
     "artifacts/flash_vtg_supplement/seed2023_bsz128/flash_vtg_gmr_quality_zero/best_joint_hl_val_preds_metrics.json",
     "artifacts/flash_vtg_supplement/seed2023_bsz128/flash_vtg_gmr_quality_zero", "早停完成"),
]

METRIC_OVERRIDES = {
    # The stage2 JSON was overwritten by a later validation after the e13
    # best-joint checkpoint. Preserve the checkpoint-selection record emitted
    # in stdout.log at 2026-07-22 21:52:49.
    ("Moment-DETR", "HieA2M-DGQC"): {
        "AUROC": 73.36, "Rej-F1@0.4": 68.22, "mAP": 9.16,
        "mR@5": 15.45, "mR+@5": 1.33, "G-mIoU@3": 36.40,
    },
}
EPOCH_OVERRIDES = {("Moment-DETR", "HieA2M-DGQC"): "e13"}


def fmt(v):
    return "—" if v is None else f"{v:.2f}"


def comparison(backbone: str, method: str, status: str, metrics: dict):
    if "baseline" in method.lower() or "anchor" in method.lower():
        return "BASE", "base"
    if method == "Plain from scratch":
        return "参考", "neutral"
    base = read_metrics(BASE[backbone])
    dm = None if not metrics or not base else mget(metrics, "mAP") - mget(base, "mAP")
    dg = None if not metrics or not base else mget(metrics, "G-mIoU@3") - mget(base, "G-mIoU@3")
    if dm is None or dg is None:
        return "无结果", "neutral"
    suffix = "（临时）" if "运行" in status else ""
    if dm > 0 and dg > 0:
        return f"双指标超过 baseline{suffix}", "win"
    if dm > 0 or dg > 0:
        return f"单指标超过 / 权衡{suffix}", "trade"
    return f"未超过 baseline{suffix}", "loss"


def status_class(status: str):
    if "运行" in status:
        return "running"
    if "失败" in status or "中止" in status:
        return "stopped"
    return "done"


def build_rows():
    out = []
    for backbone, method, status, metric_path, run_dir, note in ROWS:
        metrics = METRIC_OVERRIDES.get((backbone, method), read_metrics(metric_path))
        base = read_metrics(BASE[backbone])
        vals = {k: mget(metrics, k) for k in ("AUROC", "Rej-F1@0.4", "mAP", "mR@5", "mR+@5", "G-mIoU@3")}
        dm = None
        dg = None
        if metrics and base and "baseline" not in method.lower() and "anchor" not in method.lower() and method != "Plain from scratch":
            dm = vals["mAP"] - mget(base, "mAP")
            dg = vals["G-mIoU@3"] - mget(base, "G-mIoU@3")
        verdict, verdict_cls = comparison(backbone, method, status, metrics)
        out.append({
            "backbone": backbone, "method": method, "status": status,
            "status_cls": status_class(status),
            "epoch": EPOCH_OVERRIDES.get((backbone, method), epoch_from(run_dir)),
            "vals": vals, "dm": dm, "dg": dg, "verdict": verdict,
            "verdict_cls": verdict_cls, "note": note, "metric_path": metric_path,
        })
    return out


rows = build_rows()
wins = [r for r in rows if r["verdict_cls"] == "win" and r["status_cls"] == "done"]
running = [r for r in rows if r["status_cls"] == "running"]
stopped = [r for r in rows if r["status_cls"] == "stopped"]


def delta(v):
    if v is None:
        return "—"
    cls = "pos" if v > 0 else "neg" if v < 0 else ""
    return f'<span class="{cls}">{v:+.2f}</span>'


table_rows = []
last = None
for r in rows:
    group = ""
    if r["backbone"] != last:
        group = f'<tr class="group"><td colspan="14">{html.escape(r["backbone"])}</td></tr>'
        last = r["backbone"]
    v = r["vals"]
    table_rows.append(group + f"""
    <tr class="{r['status_cls']}">
      <td>{html.escape(r['method'])}</td>
      <td><span class="badge {r['status_cls']}">{html.escape(r['status'])}</span></td>
      <td>{r['epoch']}</td>
      <td>{fmt(v['AUROC'])}</td><td>{fmt(v['Rej-F1@0.4'])}</td>
      <td>{fmt(v['mAP'])}</td><td>{fmt(v['mR@5'])}</td><td>{fmt(v['mR+@5'])}</td>
      <td>{fmt(v['G-mIoU@3'])}</td><td>{delta(r['dm'])}</td><td>{delta(r['dg'])}</td>
      <td><span class="verdict {r['verdict_cls']}">{html.escape(r['verdict'])}</span></td>
      <td class="note">{html.escape(r['note'])}</td>
      <td class="source">{html.escape(r['metric_path'])}</td>
    </tr>""")


win_cards = "".join(
    f"""<div class="card win-card"><b>{html.escape(r['backbone'])} · {html.escape(r['method'])}</b>
    <div>mAP {fmt(r['vals']['mAP'])} ({r['dm']:+.2f}) · G-mIoU@3 {fmt(r['vals']['G-mIoU@3'])} ({r['dg']:+.2f})</div></div>"""
    for r in wins
) or '<div class="card">尚无“两个核心指标同时提升”的已完成项。</div>'


html_doc = f"""<!doctype html>
<html lang="zh"><head><meta charset="utf-8">
<title>GMR 当前实验汇总大表 {DATE}</title>
<style>
@page {{ size: A3 landscape; margin: 10mm; }}
* {{ box-sizing: border-box; }}
body {{ font-family: "Noto Sans CJK SC","Microsoft YaHei",Arial,sans-serif; color:#18212f; font-size:11px; }}
h1 {{ font-size:28px; margin:0 0 4px; color:#102a43; }}
h2 {{ font-size:18px; color:#163b65; border-bottom:2px solid #b9cee5; padding-bottom:4px; margin:18px 0 8px; }}
p {{ margin:5px 0; line-height:1.5; }}
.sub {{ color:#52667a; font-size:12px; }}
.summary {{ display:flex; gap:10px; margin:13px 0; }}
.stat {{ flex:1; padding:10px 12px; border-radius:7px; background:#edf4fb; border:1px solid #c8d9ea; }}
.stat b {{ font-size:20px; color:#143c66; display:block; }}
.cards {{ display:grid; grid-template-columns:repeat(3,1fr); gap:7px; }}
.card {{ padding:8px; border:1px solid #cfd8e3; border-radius:6px; background:#f8fafc; }}
.win-card {{ background:#eaf8ef; border-color:#85c99c; }}
.legend {{ padding:8px 10px; background:#fff8df; border-left:4px solid #e3ab2f; margin:8px 0; }}
table {{ width:100%; border-collapse:collapse; table-layout:fixed; font-size:8.5px; }}
th {{ background:#173b63; color:white; padding:5px 3px; border:1px solid #dbe5ef; }}
td {{ padding:4px 3px; border:1px solid #d6dee7; vertical-align:middle; text-align:center; }}
tr:nth-child(even):not(.group) {{ background:#f7f9fc; }}
tr.group td {{ background:#dce9f6; color:#15395f; font-weight:bold; font-size:11px; text-align:left; padding:5px; }}
tr.running {{ background:#fff9e8 !important; }}
tr.stopped {{ background:#f7eeee !important; color:#6f5252; }}
.badge {{ display:inline-block; padding:2px 5px; border-radius:9px; white-space:nowrap; }}
.badge.done {{ background:#dff2e5; color:#176739; }}
.badge.running {{ background:#ffedb8; color:#875e00; }}
.badge.stopped {{ background:#edd9d9; color:#8b3030; }}
.verdict {{ font-weight:bold; }}
.verdict.win {{ color:#08783f; background:#dbf5e6; padding:2px 4px; }}
.verdict.trade {{ color:#976800; }}
.verdict.loss {{ color:#9b3030; }}
.pos {{ color:#08783f; font-weight:bold; }} .neg {{ color:#a12c2c; }}
.note {{ text-align:left; }} .source {{ text-align:left; font-size:7px; color:#52667a; overflow-wrap:anywhere; }}
.matrix td:nth-child(1), .matrix td:nth-child(5) {{ text-align:left; }}
.small {{ color:#52667a; font-size:9px; }}
.pagebreak {{ page-break-before:always; }}
code {{ font-family:monospace; font-size:9px; }}
</style></head><body>
<h1>GMR 当前实验汇总大表</h1>
<p class="sub">快照时间：{STAMP} · Seed 2023 · 当前训练默认 batch size 128；新增/续跑任务每 5 epoch validation</p>
<div class="legend"><b>口径：</b>“完成”行采用保存的 best checkpoint；“运行中”行采用当前 best-so-far，不能作为最终论文数值；
“失败/中止”行仅保留诊断值。Δ 按同 backbone 的 GMR baseline 计算。绿色“超过 baseline”要求 mAP 与 G-mIoU@3 同时提升。</div>
<div class="summary">
 <div class="stat"><b>{len(rows)}</b>已纳入实验行</div>
 <div class="stat"><b>{len(wins)}</b>已完成双指标胜出</div>
 <div class="stat"><b>{len(running)}</b>当前运行中</div>
 <div class="stat"><b>{len(stopped)}</b>失败或主动中止</div>
</div>
<h2>已完成且明确超过各自 GMR baseline</h2>
<div class="cards">{win_cards}</div>

<h2>全量核心结果表</h2>
<table>
<colgroup>
<col style="width:8%"><col style="width:5%"><col style="width:3%">
<col style="width:4%"><col style="width:4%"><col style="width:4%"><col style="width:4%"><col style="width:4%"><col style="width:4%">
<col style="width:4%"><col style="width:4%"><col style="width:10%"><col style="width:11%"><col style="width:31%">
</colgroup>
<thead><tr><th>方法</th><th>状态</th><th>进度</th><th>AUROC</th><th>Rej-F1<br>@0.4</th><th>mAP</th><th>mR@5</th><th>mR+@5</th><th>G-mIoU@3</th><th>ΔmAP</th><th>ΔG@3</th><th>相对 baseline</th><th>说明</th><th>指标来源</th></tr></thead>
<tbody>{''.join(table_rows)}</tbody></table>

<div class="pagebreak"></div>
<h2>结果解读与当前结论</h2>
<table class="matrix">
<thead><tr><th>证据</th><th>当前判断</th><th>可信度</th><th>是否可写入论文主结论</th><th>下一步</th></tr></thead>
<tbody>
<tr><td>Moment-DETR · HieA2M-DGQC</td><td>相对 matched GMR baseline 已形成明确综合提升，是目前最强主结果。</td><td>已完成</td><td>可以</td><td>补多 seed、模块消融与公平训练预算复核。</td></tr>
<tr><td>EaTR 已完成变体</td><td>用于判断 Quality、Dual、Counter 与完整组合在第二骨干上的迁移性；以表中同骨干 Δ 为准。</td><td>已完成</td><td>可作为跨骨干证据，但需按指标权衡表述</td><td>等待 Quality+Dual 最终结果，再决定组合路线。</td></tr>
<tr><td>FlashVTG Quality / Zero / Q+Zero</td><td>两个单模块及其组合均已完成，可直接判断组合是否互补。</td><td>已完成</td><td>可以报告</td><td>以表中的严格同锚点差值选择最终配置。</td></tr>
<tr><td>QD-DETR Quality / Dual / Q+Dual</td><td>Quality 与 Quality+Dual 已完成；旧 Dual 任务仍在训练，现阶段以完成项定稿。</td><td>混合</td><td>完成项可以定稿</td><td>旧 Dual 结束后补齐最后一格。</td></tr>
<tr><td>CG-DETR 全线</td><td>现有 Quality/Phrase/Counter/HieA2M 没有形成有效收益，已停止继续消耗资源。</td><td>失败/中止</td><td>不作为正面主结果</td><td>作为负结果和适用边界，后续不优先。</td></tr>
</tbody></table>

<h2>后续实验矩阵</h2>
<table class="matrix">
<thead><tr><th>优先级</th><th>实验</th><th>当前状态</th><th>判定目标</th><th>处理原则</th></tr></thead>
<tbody>
<tr><td>P0</td><td>FlashVTG GMR + Quality + Zero</td><td>已完成</td><td>组合是否同时优于 release GMR anchor</td><td>结果已冻结；与两个单模块直接比较。</td></tr>
<tr><td>P0</td><td>QD-DETR Quality + Dual gate</td><td>已完成</td><td>两个有效倾向模块是否互补</td><td>结果已冻结；等待旧 Dual 任务补齐矩阵。</td></tr>
<tr><td>P0</td><td>EaTR Quality + Dual gate</td><td>运行中</td><td>组合能否在 EaTR 复现</td><td>完成后决定是否扩展到多 seed。</td></tr>
<tr><td>P1</td><td>Moment / EaTR / QD / FlashVTG 各自最佳方法多 seed</td><td>待运行</td><td>均值、方差与显著性</td><td>只对当前胜出配置投入资源。</td></tr>
<tr><td>P1</td><td>去重：Direct Top-K vs geometry vs learned selector</td><td>已有单骨干消融</td><td>学习式去重是否值得使用</td><td>在最终最佳 backbone 配置上统一复核。</td></tr>
<tr><td>P2</td><td>CG-DETR 组合扩展</td><td>跳过</td><td>现有证据不足</td><td>除非提出针对 CG 结构的新适配，否则不再跑。</td></tr>
</tbody></table>

<h2>审计说明</h2>
<p>1. 除 Moment-DETR HieA2M-DGQC 外，表中数值直接读取对应 JSON 指标文件。该行采用 stdout.log 中 e13 的
<code>Updated best_joint checkpoint</code> 同步指标；其 JSON 后来被后续 validation 覆写，故以 checkpoint 选择日志为准。</p>
<p>2. FlashVTG 使用 <code>GMR-unified.brief</code>；Moment-DETR、EaTR、QD-DETR、CG-DETR 使用 <code>brief</code>。</p>
<p>3. best checkpoint 可能分别由 joint 目标选出，因此表中是“同一个 best-joint checkpoint 的完整指标”，不是把各指标各自最优值拼接。</p>
<p>4. 当前运行中进度来自生成时已有日志；训练继续后 epoch 与 best-so-far 会变化，应重新运行本脚本生成新快照。</p>
</body></html>"""

out = ART / f"GMR_Experiment_Master_Summary_Current_{DATE}.html"
out.write_text(html_doc)
print(out)
