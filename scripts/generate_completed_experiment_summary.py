#!/usr/bin/env python3
"""Generate the traceable completed-experiment summary used for the PDF report."""

from __future__ import annotations

import datetime as dt
import html
import json
import statistics
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
ART = ROOT / "artifacts"
OUT = ART / "GMR_Completed_Experiments_Summary_2026-07-23.html"

METRICS = [
    ("AUROC", "AUROC"),
    ("Rej-F1@0.4", "Rej-F1@0.4"),
    ("mAP", "mAP"),
    ("mR@5", "mR@5"),
    ("mR+@5", "mR+@5"),
    ("G-mIoU@3", "G@3"),
]


def load_json(relative: str) -> dict:
    path = ROOT / relative
    with path.open(encoding="utf-8") as handle:
        return json.load(handle)


def brief(relative: str) -> dict:
    data = load_json(relative)
    return data.get("brief", data)


def fmt(value: float | None) -> str:
    return "—" if value is None else f"{float(value):.2f}"


def metric_cells(values: dict, baseline: dict | None, is_baseline: bool = False) -> str:
    cells: list[str] = []
    for key, _ in METRICS:
        value = values.get(key)
        css = ""
        note = ""
        if is_baseline:
            css = "baseline-cell"
        elif baseline is not None and value is not None and baseline.get(key) is not None:
            delta = float(value) - float(baseline[key])
            if delta > 1e-9:
                css = "win"
                note = f"<span class='delta'>+{delta:.2f}</span>"
            elif delta < -1e-9:
                css = "loss"
                note = f"<span class='delta'>{delta:.2f}</span>"
        cells.append(f"<td class='{css}'><strong>{fmt(value)}</strong>{note}</td>")
    return "".join(cells)


def experiment_table(title: str, subtitle: str, baseline: dict, rows: list[dict]) -> str:
    header = "".join(f"<th>{html.escape(label)}</th>" for _, label in METRICS)
    body = [
        "<tr class='baseline-row'>"
        f"<td><strong>{html.escape(baseline['name'])}</strong><br><span class='tag'>BASELINE GMR</span></td>"
        f"{metric_cells(baseline['metrics'], None, True)}"
        f"<td>{html.escape(baseline['note'])}</td></tr>"
    ]
    for row in rows:
        wins = sum(
            row["metrics"].get(key) is not None
            and baseline["metrics"].get(key) is not None
            and row["metrics"][key] > baseline["metrics"][key]
            for key, _ in METRICS
        )
        badge = "<span class='tag win-tag'>超过 GMR</span>" if wins else ""
        body.append(
            "<tr>"
            f"<td><strong>{html.escape(row['name'])}</strong><br>{badge}</td>"
            f"{metric_cells(row['metrics'], baseline['metrics'])}"
            f"<td>{html.escape(row['note'])}</td></tr>"
        )
    return f"""
    <section>
      <h2>{html.escape(title)}</h2>
      <p class="subtitle">{html.escape(subtitle)}</p>
      <table>
        <thead><tr><th>方案</th>{header}<th>结论</th></tr></thead>
        <tbody>{''.join(body)}</tbody>
      </table>
    </section>
    """


def main() -> None:
    # Moment-DETR: strict baseline and the final independent-zero-head result.
    moment_gmr_path = (
        "artifacts/formal_strict/moment_detr/seed2023/md_gmr_b128/"
        "best_joint_soccer_gmr_val_preds_metrics.json"
    )
    moment_proposed_path = (
        "artifacts/validation_selector_ablation/seed2023_posw1/stage2_zero/"
        "best_joint_soccer_gmr_val_preds_metrics.json"
    )
    moment_base = {
        "name": "Moment-DETR Strict GMR",
        "metrics": brief(moment_gmr_path),
        "note": "同骨干严格 GMR；best-joint",
    }
    moment_rows = [
        {
            "name": "HieA2M-DGQC + 独立 Zero Head",
            "metrics": brief(moment_proposed_path),
            "note": "posw=1；validation best-joint；除 mR+@5 外均超过 GMR",
        }
    ]

    # EaTR matched transfer matrix, all selected by best_joint.
    eatr_root = "artifacts/eatr_dgqc_transfer/seed2023"
    eatr_base_path = f"{eatr_root}/eatr_gmr_strict/best_joint_val_metrics.json"
    eatr_base = {
        "name": "EaTR Strict GMR",
        "metrics": brief(eatr_base_path),
        "note": "EaTR 迁移线配对基线；best-joint",
    }
    eatr_rows = []
    for name, directory, note in [
        ("+ Quality", "eatr_quality", "所有六项均超过 GMR；最均衡"),
        ("+ Dual gate", "eatr_dual", "拒答与 G@3 提升最大，召回略降"),
        ("+ Counter", "eatr_counter", "仅 AUROC 超过 GMR"),
        ("完整 HieA2M-DGQC", "eatr_hiea2m", "仅 AUROC 超过 GMR；完整组合存在负交互"),
    ]:
        eatr_rows.append(
            {
                "name": name,
                "metrics": brief(f"{eatr_root}/{directory}/best_joint_val_metrics.json"),
                "note": note,
            }
        )

    # Strict QD/CG parents are complete; their component matrices are still running.
    qd_path = (
        "artifacts/strict_bsz32/qd_detr/seed2023/qd_detr_gmr/"
        "best_joint_val_metrics.json"
    )
    cg_path = (
        "artifacts/strict_bsz32/cg_detr/seed2023/cg_detr_gmr/"
        "best_joint_val_metrics.json"
    )
    qd = brief(qd_path)
    cg = brief(cg_path)

    # Geometry-only dedup is a separate protocol/input from learned selection.
    geom_path = (
        "artifacts/validation_dedup_ablation/"
        "md_hiea2m_b128_rerun_from_best_v2_best_map/dedup_ablation_summary.json"
    )
    geom = load_json(geom_path)
    geom_by_name = {item["name"]: item for item in geom["methods"]}
    geom_names = [
        ("none__topk3", "Direct Top-3"),
        ("hard_nms__iou0p5__topk3", "Hard-NMS IoU=0.5"),
        ("soft_nms_linear__iou0p5__floor0__topk3", "Linear Soft-NMS IoU=0.5"),
        ("cluster_fusion__iou0p5__complete__topk3", "Complete-link Fusion IoU=0.5"),
    ]
    geom_base = geom_by_name[geom_names[0][0]]["brief"]
    geom_rows = []
    for key, label in geom_names:
        b = geom_by_name[key]["brief"]
        geom_rows.append((label, b["mAP"], b["mR@3"], b["G-mIoU@3"]))

    # Learned selector averages over the five completed head-training settings.
    selector_paths = sorted(
        (ART / "validation_selector_ablation").glob(
            "*/stage4_5_selection/learned_selector_ablation_summary.json"
        )
    )
    selector_stages = [
        ("direct_topk", "Direct Top-3"),
        ("learned_topk", "学习式去重 Top-3"),
        ("learned_soft_count", "学习式去重 + Soft Count"),
        ("learned_soft_count_fusion", "Soft Count + Boundary Fusion"),
    ]
    selector_rows = []
    for stage, label in selector_stages:
        records = [json.loads(path.read_text(encoding="utf-8"))["best_by_stage"][stage] for path in selector_paths]
        selector_rows.append(
            {
                "label": label,
                "map": statistics.mean(record["brief"]["mAP"] for record in records),
                "mr3": statistics.mean(record["brief"]["mR@3"] for record in records),
                "mr5": statistics.mean(record["brief"]["mR@5"] for record in records),
                "g3": statistics.mean(record["brief"]["G-mIoU@3"] for record in records),
                "count": statistics.mean(record["mean_selected_count"] for record in records),
            }
        )
    selector_base = selector_rows[0]

    # Run-level archive: completed repetitions and legacy-protocol runs that are
    # intentionally excluded from the matched main tables.
    inventory: list[dict] = []

    def add_inventory(name: str, path: str, selection: str, role: str) -> None:
        b = brief(path)
        inventory.append(
            {
                "name": name,
                "selection": selection,
                "map": b.get("mAP"),
                "g3": b.get("G-mIoU@3"),
                "auroc": b.get("AUROC"),
                "role": role,
                "path": path,
            }
        )

    for name, directory in [
        ("Moment GMR strict initial", "md_gmr_b128"),
        ("Moment GMR rerun", "md_gmr_b128_rerun_from_best"),
        ("Moment GMR rerun v2", "md_gmr_b128_rerun_from_best_v2"),
        ("Moment HieA2M strict initial", "md_hiea2m_b128"),
        ("Moment HieA2M rerun", "md_hiea2m_b128_rerun_from_best"),
        ("Moment HieA2M rerun v2", "md_hiea2m_b128_rerun_from_best_v2"),
    ]:
        add_inventory(
            name,
            "artifacts/formal_strict/moment_detr/seed2023/"
            f"{directory}/best_joint_soccer_gmr_val_preds_metrics.json",
            "best-joint",
            "重复训练/稳定性记录；主表只选预先声明的配对结果",
        )

    for name, directory in [
        ("QD plain formal", "qd_detr"),
        ("QD GMR legacy", "qd_detr_gmr"),
        ("QD GMR invalid-all-lr", "qd_detr_gmr_invalid_all_lr"),
        ("QD Quality legacy", "qd_quality"),
        ("QD Dual legacy", "qd_dual"),
        ("QD Counter legacy", "qd_counter"),
        ("QD HieA2M legacy", "qd_hiea2m"),
    ]:
        add_inventory(
            name,
            f"artifacts/formal/qd_detr/seed2023/{directory}/best_joint_val_metrics.json",
            "best-joint",
            "旧协议；只归档，不与 strict-bsz32 主表比较",
        )

    for name, directory in [
        ("EaTR plain formal legacy", "eatr"),
        ("EaTR GMR formal legacy", "eatr_gmr"),
    ]:
        add_inventory(
            name,
            f"artifacts/formal/eatr/seed2023/{directory}/best_joint_val_metrics.json",
            "best-joint",
            "旧迁移线；由 matched frozen-parent 实验替代",
        )

    inventory_rows = "".join(
        "<tr>"
        f"<td><strong>{html.escape(item['name'])}</strong></td>"
        f"<td>{html.escape(item['selection'])}</td>"
        f"<td>{fmt(item['map'])}</td><td>{fmt(item['g3'])}</td><td>{fmt(item['auroc'])}</td>"
        f"<td>{html.escape(item['role'])}<br><span class='source'>{html.escape(item['path'])}</span></td>"
        "</tr>"
        for item in inventory
    )

    generated = dt.datetime.now().astimezone().strftime("%Y-%m-%d %H:%M %Z")
    moment_section = experiment_table(
        "1. Moment-DETR：主方法与严格 GMR",
        "Soccer-GMR validation；绿色表示严格高于同骨干 baseline GMR。独立 Zero Head 的结果来自最终 best_joint JSON，而非中间 epoch。",
        moment_base,
        moment_rows,
    )
    eatr_section = experiment_table(
        "2. EaTR：DGQC 跨骨干迁移",
        "全部使用同一 frozen parent、同一 validation 和 best-joint 选择口径。",
        eatr_base,
        eatr_rows,
    )

    geom_body = []
    for index, (label, map_v, mr3, g3) in enumerate(geom_rows):
        def gc(value: float, base: float) -> str:
            delta = value - base
            if index and delta > 1e-9:
                return f"<td class='win'><strong>{value:.2f}</strong><span class='delta'>+{delta:.2f}</span></td>"
            return f"<td class='baseline-cell'><strong>{value:.2f}</strong></td>" if not index else f"<td><strong>{value:.2f}</strong></td>"

        geom_body.append(
            f"<tr><td><strong>{html.escape(label)}</strong>{'<br><span class=\"tag\">BASELINE</span>' if not index else ''}</td>"
            f"{gc(map_v, geom_base['mAP'])}{gc(mr3, geom_base['mR@3'])}{gc(g3, geom_base['G-mIoU@3'])}"
            f"<td>固定 Top-3；相同候选输入</td></tr>"
        )

    learned_body = []
    for index, row in enumerate(selector_rows):
        def lc(value: float, base: float) -> str:
            delta = value - base
            if index and delta > 1e-9:
                return f"<td class='win'><strong>{value:.2f}</strong><span class='delta'>+{delta:.2f}</span></td>"
            if not index:
                return f"<td class='baseline-cell'><strong>{value:.2f}</strong></td>"
            return f"<td class='loss'><strong>{value:.2f}</strong><span class='delta'>{delta:.2f}</span></td>"

        learned_body.append(
            f"<tr><td><strong>{html.escape(row['label'])}</strong>{'<br><span class=\"tag\">BASELINE</span>' if not index else ''}</td>"
            f"{lc(row['map'], selector_base['map'])}"
            f"{lc(row['mr3'], selector_base['mr3'])}"
            f"{lc(row['mr5'], selector_base['mr5'])}"
            f"{lc(row['g3'], selector_base['g3'])}"
            f"<td>{row['count']:.2f}</td></tr>"
        )

    html_doc = f"""<!doctype html>
<html lang="zh-CN">
<head>
<meta charset="utf-8">
<title>GMR 已完成实验汇总</title>
<style>
  @page {{ size: A4 landscape; margin: 11mm 12mm 12mm; }}
  * {{ box-sizing: border-box; }}
  body {{ margin: 0; color: #182233; font-family: "Noto Sans CJK SC", "Microsoft YaHei", sans-serif; font-size: 10.5px; line-height: 1.45; }}
  h1 {{ margin: 0 0 5px; color: #163a68; font-size: 25px; }}
  h2 {{ margin: 0 0 5px; padding-bottom: 5px; border-bottom: 2px solid #2f67a5; color: #193f6d; font-size: 17px; }}
  h3 {{ margin: 14px 0 5px; color: #244f7c; font-size: 13px; }}
  p {{ margin: 4px 0 8px; }}
  .cover {{ padding: 8px 3px 0; }}
  .meta {{ color: #65758a; }}
  .lead {{ margin-top: 12px; padding: 11px 13px; border-left: 5px solid #1a9b69; background: #eef9f4; font-size: 12px; }}
  .key-grid {{ display: table; width: 100%; margin-top: 13px; border-spacing: 9px; }}
  .key-card {{ display: table-cell; width: 33%; padding: 12px; border: 1px solid #ccd8e5; border-radius: 6px; vertical-align: top; }}
  .key-card strong {{ display: block; color: #126b4d; font-size: 18px; }}
  section {{ margin-top: 15px; page-break-inside: avoid; }}
  .page-break {{ page-break-before: always; }}
  .subtitle {{ color: #65758a; }}
  table {{ width: 100%; border-collapse: collapse; table-layout: fixed; }}
  th {{ background: #173b65; color: white; padding: 6px 5px; font-weight: 600; }}
  td {{ border: 1px solid #cdd7e2; padding: 6px 5px; text-align: center; vertical-align: middle; }}
  th:first-child, td:first-child {{ width: 19%; text-align: left; }}
  th:last-child, td:last-child {{ width: 23%; text-align: left; }}
  .baseline-row td {{ background: #edf2f8; }}
  .baseline-cell {{ background: #e9eff6; }}
  .win {{ background: #dff6ea; color: #086a47; }}
  .loss {{ color: #6c7888; }}
  .delta {{ display: block; font-size: 8.5px; font-weight: 600; }}
  .tag {{ display: inline-block; margin-top: 2px; padding: 1px 5px; border-radius: 8px; background: #62748a; color: white; font-size: 7.5px; }}
  .win-tag {{ background: #16875e; }}
  .note {{ padding: 8px 10px; background: #fff7df; border-left: 4px solid #e6a522; }}
  .source {{ font-family: monospace; font-size: 8px; color: #536276; overflow-wrap: anywhere; }}
  .small {{ font-size: 9px; color: #65758a; }}
  ul {{ margin: 5px 0 8px; padding-left: 20px; }}
  li {{ margin: 2px 0; }}
</style>
</head>
<body>
  <div class="cover">
    <h1>GMR 已完成实验结果汇总</h1>
    <p class="meta">Soccer-GMR validation · 数据生成时间：{html.escape(generated)} · 只纳入已经完成且有可追溯指标文件的正式实验</p>
    <div class="lead"><strong>阅读规则：</strong>每张主表以同骨干的 baseline GMR 为基准；绿色单元格表示该指标严格超过 baseline。不同评估集、不同候选输入或不同 checkpoint 选择协议不做直接优劣比较。</div>
    <div class="key-grid">
      <div class="key-card"><strong>Moment 主方法有效</strong>独立判空方案相对 Strict GMR：mAP、拒答和 G@3 同时提高；仅多事件 mR+@5 下降。</div>
      <div class="key-card"><strong>EaTR 可部分泛化</strong>Quality 六项全部超过 EaTR GMR；Dual gate 显著改善拒答与 G@3。完整组合未超过单组件。</div>
      <div class="key-card"><strong>学习式去重有效</strong>五组设置下 mAP 平均 6.48 → 6.96；边界融合没有额外收益，Soft Count 偏向高召回。</div>
    </div>
  </div>

  {moment_section}
  {eatr_section}

  <div class="page-break"></div>
  <section>
    <h2>3. 去重消融：严格分开两种候选输入协议</h2>
    <h3>3.1 纯几何去重（固定候选输入）</h3>
    <p class="subtitle">只与本表 Direct Top-3=7.57 比较；不与后面的 learned-selector Direct Top-3=6.48 混用。</p>
    <table>
      <thead><tr><th>方法</th><th>mAP</th><th>mR@3</th><th>G@3</th><th>协议</th></tr></thead>
      <tbody>{''.join(geom_body)}</tbody>
    </table>

    <h3>3.2 学习式去重（5组 head 训练设置平均）</h3>
    <p class="subtitle">五组为 seed2023、seed2023-posw1、seed2023-posw4、seed2024、seed2025；λ 由 validation 网格选择，多数为2.0，seed2025 为1.0。</p>
    <table>
      <thead><tr><th>方法</th><th>mAP</th><th>mR@3</th><th>mR@5</th><th>G@3</th><th>平均选框数</th></tr></thead>
      <tbody>{''.join(learned_body)}</tbody>
    </table>
    <p class="note"><strong>结论：</strong>学习式去重 Top-3 是当前主选框方案。Soft Count 把 mR@5 提高到 10.69、G@3 提高到 7.25，但平均输出 5.94 个框且 mAP 低于 learned Top-3；Boundary Fusion 与 Soft Count 指标完全相同，没有观察到额外贡献。</p>
  </section>

  <section>
    <h2>4. 已完成但暂时没有组件对照的严格 GMR 父模型</h2>
    <table>
      <thead><tr><th>骨干 / 状态</th><th>mAP</th><th>G@3</th><th>AUROC</th><th>Rej-F1@0.4</th><th>mR@5</th><th>说明</th></tr></thead>
      <tbody>
        <tr><td><strong>QD-DETR Strict GMR bsz32</strong><br><span class="tag">已完成</span></td><td>{fmt(qd.get('mAP'))}</td><td>{fmt(qd.get('G-mIoU@3'))}</td><td>{fmt(qd.get('AUROC'))}</td><td>{fmt(qd.get('Rej-F1@0.4'))}</td><td>{fmt(qd.get('mR@5'))}</td><td>组件矩阵仍在训练，暂不宣称组件增益</td></tr>
        <tr><td><strong>CG-DETR Strict GMR bsz32</strong><br><span class="tag">已完成</span></td><td>{fmt(cg.get('mAP'))}</td><td>{fmt(cg.get('G-mIoU@3'))}</td><td>{fmt(cg.get('AUROC'))}</td><td>{fmt(cg.get('Rej-F1@0.4'))}</td><td>{fmt(cg.get('mR@5'))}</td><td>组件矩阵仍在训练，暂不宣称组件增益</td></tr>
      </tbody>
    </table>
  </section>

  <div class="page-break"></div>
  <section>
    <h2>5. 完成但不进入主结论的诊断实验</h2>
    <table>
      <thead><tr><th>实验</th><th>mAP</th><th>G@3</th><th colspan="4">判定</th><th>处理</th></tr></thead>
      <tbody>
        <tr><td>QD Plain bsz128 Restart</td><td>0.07</td><td>17.22</td><td colspan="4">定位优化塌陷；原因尚未由控制变量实验确定</td><td>不进入方法比较</td></tr>
        <tr><td>CG Plain bsz128 Restart</td><td>0.01</td><td>1.55</td><td colspan="4">定位优化塌陷；原因尚未由控制变量实验确定</td><td>不进入方法比较</td></tr>
        <tr><td>EaTR Plain bsz128</td><td>8.73</td><td>3.51</td><td colspan="4">定位正常，但 Rej-F1@0.4=0；用于说明 GMR 拒答模块的必要性</td><td>只作上下文基线</td></tr>
        <tr><td>QD/CG legacy protocol variants</td><td>—</td><td>—</td><td colspan="4">旧损失/旧空查询处理协议，与 strict 结果不可直接比较</td><td>保留日志，不进主表</td></tr>
      </tbody>
    </table>
    <p class="small">Smoke test、batch-size probe、持久化机制测试只用于工程验证，不作为科学实验结果列入。</p>
  </section>

  <section>
    <h2>6. 当前可支持的研究结论</h2>
    <ul>
      <li><strong>主结果：</strong>Moment-DETR 的“高召回初判 + 独立判空复核”在 validation 上相对 Strict GMR 同时提高 mAP、拒答与 G@3。</li>
      <li><strong>跨骨干：</strong>EaTR Quality 和 Dual gate 均超过 EaTR Strict GMR 的关键指标，支持组件级泛化；完整 DGQC 组合尚未证明跨骨干整体泛化。</li>
      <li><strong>去重：</strong>学习式去重在五组设置中一致提高 mAP；纯几何 Complete-link Fusion 也有小幅增益，但两者来自不同候选输入，不能直接排序。</li>
      <li><strong>计数：</strong>Soft Count 提高召回和 G@3，但牺牲 mAP并输出更多框；现阶段应保留为消融，不作为默认主方案。</li>
      <li><strong>边界融合：</strong>没有观察到相对 Soft Count 的额外提升。</li>
      <li><strong>限制：</strong>以上主表均为 validation；最终论文结论仍需冻结配置后进行 test-set 评估。</li>
    </ul>
  </section>

  <section>
    <h2>7. 指标来源索引</h2>
    <p class="source">M1 {html.escape(moment_gmr_path)}</p>
    <p class="source">M2 {html.escape(moment_proposed_path)}</p>
    <p class="source">E1 {html.escape(eatr_base_path)}</p>
    <p class="source">E2–E5 {html.escape(eatr_root)}/eatr_{{quality,dual,counter,hiea2m}}/best_joint_val_metrics.json</p>
    <p class="source">S1 {html.escape(geom_path)}</p>
    <p class="source">S2 artifacts/validation_selector_ablation/*/stage4_5_selection/learned_selector_ablation_summary.json</p>
    <p class="source">Q1 {html.escape(qd_path)}</p>
    <p class="source">C1 {html.escape(cg_path)}</p>
  </section>

  <div class="page-break"></div>
  <section>
    <h2>附录 A：其余已完成正式运行的逐项归档</h2>
    <p class="subtitle">本表用于做到“已完成实验不遗漏”。其中 legacy、重复续训和无效学习率实验不进入主结论，也不因单项数字较高而标记为超过 strict baseline。</p>
    <table>
      <thead><tr><th>运行</th><th>选择口径</th><th>mAP</th><th>G@3</th><th>AUROC</th><th>用途与来源</th></tr></thead>
      <tbody>{inventory_rows}</tbody>
    </table>
    <p class="note"><strong>未列入：</strong>当前仍在训练的 QD/CG 八个 strict-bsz32 组件实验；smoke test、batch probe、持久化测试等工程检查。它们分别因为“尚未完成”或“不属于科学结果”而不应伪装成已完成实验。</p>
  </section>
</body>
</html>
"""
    OUT.write_text(html_doc, encoding="utf-8")
    print(OUT)


if __name__ == "__main__":
    main()
