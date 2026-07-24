#!/usr/bin/env python3
"""Generate the concise, traceable GMR progress report for advisor review."""

from __future__ import annotations

import datetime as dt
import glob
import html
import json
import statistics
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
ART = ROOT / "artifacts"
OUT_HTML = ART / "GMR_Teacher_Progress_Report_2026-07-23.html"
METRICS = ("AUROC", "Rej-F1@0.4", "mAP", "G-mIoU@3")


SOURCES = {
    "moment_base": (
        "artifacts/formal_strict/moment_detr/seed2023/md_gmr_b128/"
        "best_joint_soccer_gmr_val_preds_metrics.json"
    ),
    "moment_zero": (
        "artifacts/validation_selector_ablation/seed2023_posw1/stage2_zero/"
        "best_joint_soccer_gmr_val_preds_metrics.json"
    ),
    "eatr_base": (
        "artifacts/eatr_dgqc_transfer/seed2023/eatr_gmr_strict/"
        "best_joint_val_metrics.json"
    ),
    "eatr_quality": (
        "artifacts/eatr_dgqc_transfer/seed2023/eatr_quality/"
        "best_joint_val_metrics.json"
    ),
    "eatr_dual": (
        "artifacts/eatr_dgqc_transfer/seed2023/eatr_dual/"
        "best_joint_val_metrics.json"
    ),
    "flash_anchor": (
        "artifacts/flash_vtg_supplement/release_gmr_val/"
        "strict_gmr_val_metrics_raw.json"
    ),
    "flash_quality": (
        "artifacts/flash_vtg_supplement/seed2023_bsz128/"
        "flash_vtg_gmr_quality/best_joint_hl_val_preds_metrics.json"
    ),
    "flash_plain_running": (
        "artifacts/flash_vtg_supplement/seed2023_bsz128/"
        "flash_vtg_plain/best_joint_hl_val_preds_metrics.json"
    ),
    "flash_gmr_running": (
        "artifacts/flash_vtg_supplement/seed2023_bsz128/"
        "flash_vtg_gmr/best_joint_hl_val_preds_metrics.json"
    ),
    "qd_fair_control": (
        "artifacts/qd_fair_ablation/seed2023_bsz32/continued_control/"
        "best_joint_val_metrics.json"
    ),
    "qd_fair_quality": (
        "artifacts/qd_fair_ablation/seed2023_bsz32/quality/"
        "best_joint_val_metrics.json"
    ),
    "qd_fair_dual": (
        "artifacts/qd_fair_ablation/seed2023_bsz32/dual/"
        "best_joint_val_metrics.json"
    ),
    "qd_fair_quality_dual": (
        "artifacts/qd_fair_ablation/seed2023_bsz32/quality_dual/"
        "best_joint_val_metrics.json"
    ),
}


def read_json(relative: str) -> dict:
    with (ROOT / relative).open(encoding="utf-8") as handle:
        return json.load(handle)


def brief(relative: str) -> dict:
    data = read_json(relative)
    unified = data.get("GMR-unified")
    if isinstance(unified, dict) and isinstance(unified.get("brief"), dict):
        return unified["brief"]
    value = data.get("brief", data)
    if not isinstance(value, dict):
        raise TypeError(f"metrics brief is not a mapping: {relative}")
    return value


def select(values: dict) -> dict[str, float]:
    missing = [key for key in METRICS if key not in values]
    if missing:
        raise KeyError(f"missing report metrics {missing}")
    return {key: float(values[key]) for key in METRICS}


def fmt(value: float) -> str:
    return f"{value:.2f}"


def delta(value: float, base: float) -> str:
    change = value - base
    css = "up" if change > 1e-9 else "down" if change < -1e-9 else "flat"
    return f'<span class="{css}">{change:+.2f}</span>'


def metric_header() -> str:
    return "".join(
        f"<th>{label}</th>"
        for label in ("AUROC ↑", "Rej-F1@0.4 ↑", "mAP ↑", "G-mIoU@3 ↑")
    )


def comparison_table(
    title: str,
    subtitle: str,
    base_name: str,
    base: dict[str, float],
    rows: list[tuple[str, dict[str, float], str]],
) -> str:
    body = [
        "<tr class='base'>"
        f"<td><b>{html.escape(base_name)}</b><br><span class='pill'>BASE</span></td>"
        + "".join(f"<td><b>{fmt(base[key])}</b></td>" for key in METRICS)
        + "<td>配对比较基线</td></tr>"
    ]
    for name, values, note in rows:
        body.append(
            "<tr><td><b>"
            + html.escape(name)
            + "</b><br><span class='pill positive'>有效结果</span></td>"
            + "".join(
                f"<td><b>{fmt(values[key])}</b><br>{delta(values[key], base[key])}</td>"
                for key in METRICS
            )
            + f"<td class='left'>{html.escape(note)}</td></tr>"
        )
    return f"""
    <section>
      <h2>{html.escape(title)}</h2>
      <p class="subtitle">{html.escape(subtitle)}</p>
      <table class="metrics">
        <thead><tr><th>方法</th>{metric_header()}<th>可汇报结论</th></tr></thead>
        <tbody>{''.join(body)}</tbody>
      </table>
    </section>
    """


def selector_summary() -> tuple[list[dict], list[str]]:
    paths = sorted(
        glob.glob(
            str(
                ART
                / "validation_selector_ablation"
                / "*"
                / "stage4_5_selection"
                / "learned_selector_ablation_summary.json"
            )
        )
    )
    if not paths:
        raise FileNotFoundError("no learned selector summaries")
    stages = [
        ("direct_topk", "Direct Top-3"),
        ("learned_topk", "Learned Dedup Top-3"),
        ("learned_soft_count", "Learned Dedup + Soft Count"),
    ]
    rows = []
    for stage, label in stages:
        records = [
            json.loads(Path(path).read_text(encoding="utf-8"))["best_by_stage"][stage]
            for path in paths
        ]
        rows.append(
            {
                "name": label,
                "metrics": {
                    key: statistics.mean(float(record["brief"][key]) for record in records)
                    for key in METRICS
                },
                "mean_count": statistics.mean(
                    float(record["mean_selected_count"]) for record in records
                ),
            }
        )
    return rows, paths


def main() -> None:
    values = {name: select(brief(path)) for name, path in SOURCES.items()}
    selector_rows, selector_paths = selector_summary()

    moment = comparison_table(
        "1. 两级判空：当前最强单骨干主结果",
        "Moment-DETR，Standard validation，seed2023，best-joint。第二层 Zero Head 独立输出 P(N=0)，用于 rescue/veto。",
        "Moment-DETR Strict GMR",
        values["moment_base"],
        [
            (
                "HieA2M-DGQC + Independent Zero",
                values["moment_zero"],
                "四项指标全部提升；验证“高召回初判 + 独立判空复核”具有明显潜力。",
            )
        ],
    )

    quality_eatr = comparison_table(
        "2. Quality-aware Ranking：跨骨干最稳定的正向模块",
        "EaTR，matched strict parent，Standard validation，seed2023，best-joint。Quality head 学习候选与最近 GT 的 temporal IoU，并与前景分数联合排序。",
        "EaTR Strict GMR",
        values["eatr_base"],
        [
            (
                "EaTR + Quality",
                values["eatr_quality"],
                "AUROC、拒答、定位和 generalized 指标同时提高，是目前最均衡的迁移结果。",
            )
        ],
    )

    quality_flash = comparison_table(
        "3. Quality 在 Flash-VTG 上的外部架构证据",
        "Flash-VTG 发布 GMR parent 上冻结主干、仅训练 Quality head；Standard validation，seed2023，best-joint。该实验已早停完成。",
        "Flash-VTG GMR Release Anchor",
        values["flash_anchor"],
        [
            (
                "Flash-VTG GMR + Quality",
                values["flash_quality"],
                "不改变判空分数，因此 AUROC/Rej 保持；mAP +0.66、G-mIoU@3 +0.10，支持 Quality 可移植。",
            )
        ],
    )

    dual = comparison_table(
        "4. Dual Grounding：可选的拒答增强模块",
        "EaTR，matched strict parent，Standard validation，seed2023，best-joint。Dual head 建模细粒度文本/虚拟片段 grounding。",
        "EaTR Strict GMR",
        values["eatr_base"],
        [
            (
                "EaTR + Dual Grounding",
                values["eatr_dual"],
                "Rej-F1 +8.77、G-mIoU@3 +4.28；mAP 基本持平，适合作为拒答增强而非公共核心。",
            )
        ],
    )

    selector_base = selector_rows[0]
    selector_body = []
    for index, row in enumerate(selector_rows):
        cls = "base" if index == 0 else ""
        note = (
            "固定候选直接截断。"
            if index == 0
            else "默认去重候选；mAP 最优。"
            if row["name"] == "Learned Dedup Top-3"
            else "召回导向；G-mIoU@3 最优，但 mAP 低于 Learned Top-3。"
        )
        selector_body.append(
            f"<tr class='{cls}'><td><b>{html.escape(row['name'])}</b></td>"
            + "".join(
                (
                    f"<td><b>{fmt(row['metrics'][key])}</b></td>"
                    if index == 0
                    else f"<td><b>{fmt(row['metrics'][key])}</b><br>"
                    f"{delta(row['metrics'][key], selector_base['metrics'][key])}</td>"
                )
                for key in METRICS
            )
            + f"<td>{row['mean_count']:.2f}</td><td class='left'>{note}</td></tr>"
        )

    selector = f"""
    <section>
      <h2>5. Learned Event Dedup：优于直接 Top-K</h2>
      <p class="subtitle">Moment-DETR 固定候选上的 5 组 head 设置平均：
      seed2023、seed2023-posw1、seed2023-posw4、seed2024、seed2025。
      Pairwise head 判断两个候选是否指向同一事件；分类分数固定，因此 AUROC/Rej 不变。</p>
      <table class="metrics">
        <thead><tr><th>选择策略</th>{metric_header()}<th>平均选框数</th><th>结论</th></tr></thead>
        <tbody>{''.join(selector_body)}</tbody>
      </table>
    </section>
    """

    running = comparison_table(
        "附：Flash-VTG 从头配对训练的当前趋势（非最终论文数字）",
        "两项仍在训练，表内为各自 best-joint best-so-far，仅用于说明外部架构可行性。",
        "Flash-VTG Plain",
        values["flash_plain_running"],
        [
            (
                "Flash-VTG GMR",
                values["flash_gmr_running"],
                "当前四项均优于 plain；完成后才冻结为正式配对结果。",
            )
        ],
    )

    qd_names = [
        ("continued_control", "Continued Control", values["qd_fair_control"]),
        ("quality", "+ Quality", values["qd_fair_quality"]),
        ("dual", "+ Dual", values["qd_fair_dual"]),
        ("quality_dual", "+ Quality + Dual", values["qd_fair_quality_dual"]),
    ]
    qd_rows = "".join(
        "<tr>"
        f"<td><b>{html.escape(label)}</b></td>"
        + "".join(f"<td>{fmt(metric[key])}</td>" for key in METRICS)
        + "</tr>"
        for _, label, metric in qd_names
    )

    source_rows = "".join(
        "<div class='source-item'>"
        f"<div class='source-name'>{html.escape(name)}</div>"
        f"<div class='path'>{html.escape(path)}</div>"
        "</div>"
        for name, path in SOURCES.items()
    )
    source_rows += "".join(
        "<div class='source-item'>"
        "<div class='source-name'>selector average input</div>"
        f"<div class='path'>{html.escape(str(Path(path).relative_to(ROOT)))}</div>"
        "</div>"
        for path in selector_paths
    )

    stamp = dt.datetime.now().astimezone().strftime("%Y-%m-%d %H:%M %Z")
    document = f"""<!doctype html>
<html lang="zh-CN"><head><meta charset="utf-8">
<title>GMR 阶段性方法与有效结果汇报</title>
<style>
@page {{ size:A4; margin:13mm 13mm 14mm; }}
* {{ box-sizing:border-box; }}
body {{ font-family:"Noto Sans CJK SC","Microsoft YaHei","WenQuanYi Micro Hei",sans-serif;
       color:#172234; font-size:10.5pt; line-height:1.52; }}
h1 {{ color:#123b64; font-size:25pt; margin:0 0 5mm; }}
h2 {{ color:#164f7c; font-size:15pt; border-bottom:2px solid #b8cee1;
      padding-bottom:2mm; margin:6mm 0 2mm; }}
h3 {{ color:#285b80; margin:4mm 0 1mm; }}
p {{ margin:1.5mm 0; }}
.cover {{ min-height:255mm; display:flex; flex-direction:column; justify-content:center; }}
.kicker {{ color:#3277a8; font-size:12pt; font-weight:bold; letter-spacing:1px; }}
.meta {{ color:#607487; margin-top:3mm; }}
.hero {{ background:linear-gradient(135deg,#edf6fc,#f7fbfe); border-left:5px solid #2778af;
         padding:5mm; margin:8mm 0; border-radius:3px; }}
.grid {{ display:grid; grid-template-columns:1fr 1fr; gap:3mm; margin:4mm 0; }}
.card {{ border:1px solid #cbd9e5; border-radius:5px; padding:3.5mm; background:#f8fbfd; }}
.card b {{ color:#164f7c; }}
.claim {{ background:#e9f7ef; border:1px solid #9bcdb0; padding:3mm; border-radius:4px; }}
.warn {{ background:#fff7df; border-left:4px solid #d6a127; padding:3mm; margin:3mm 0; }}
.audit {{ background:#f4f1fa; border-left:4px solid #7654a7; padding:3mm; margin:3mm 0; }}
.page {{ page-break-before:always; }}
table {{ width:100%; border-collapse:collapse; margin:2.5mm 0 4mm; table-layout:fixed; }}
th {{ background:#174d78; color:white; padding:2.2mm 1.3mm; font-size:8.6pt; }}
td {{ border:1px solid #d2dce5; padding:2mm 1.3mm; text-align:center; font-size:8.6pt; }}
tr:nth-child(even) {{ background:#f7fafc; }}
tr.base {{ background:#e9f0f6; }}
.left {{ text-align:left; }}
.pill {{ display:inline-block; border-radius:9px; padding:0.3mm 1.5mm;
         background:#dbe7f1; color:#31536e; font-size:7.5pt; }}
.pill.positive {{ background:#dff3e6; color:#176b3b; }}
.up {{ color:#08783f; font-weight:bold; }}
.down {{ color:#a73535; font-weight:bold; }}
.flat {{ color:#5f6c78; }}
.subtitle {{ color:#52697d; font-size:9pt; }}
ul {{ margin:2mm 0 2mm 5mm; padding-left:4mm; }}
li {{ margin:1.1mm 0; }}
.path {{ text-align:left; font-family:monospace; font-size:6.8pt; overflow-wrap:anywhere; }}
.source-list {{ margin:3mm 0 4mm; border-top:1px solid #d2dce5; }}
.source-item {{ padding:1.5mm 2mm; border:1px solid #d2dce5; border-top:0;
                page-break-inside:avoid; }}
.source-item:nth-child(even) {{ background:#f7fafc; }}
.source-name {{ color:#164f7c; font-weight:bold; font-size:7.8pt; margin-bottom:0.4mm; }}
.small {{ font-size:8pt; color:#596d7e; }}
.footer-note {{ margin-top:6mm; padding-top:2mm; border-top:1px solid #cbd8e3; color:#617486; }}
</style></head><body>

<div class="cover">
  <div class="kicker">GENERALIZED MOMENT RETRIEVAL · 阶段性研究汇报</div>
  <h1>GMR 阶段性方法与有效结果汇报</h1>
  <p class="meta">汇报快照：{stamp}<br>
  数据集：Standard validation（465 queries：255 positive / 210 null）<br>
  主要口径：AUROC、Rej-F1@0.4、mAP、G-mIoU@3</p>
  <div class="hero">
    <h3>当前最值得形成论文的主线</h3>
    <p><b>高召回初判 + 独立判空复核 + Quality-aware 定位 + Learned Event Dedup</b></p>
    <p>即候选框架 <b>U = B + Q + Z + P</b>。现有结果分别验证了 Z、Q、P 的机制价值；
    Dual Grounding 作为拒答增强模块保留，不把 Counter/Fusion 纳入默认方法。</p>
  </div>
  <div class="grid">
    <div class="card"><b>最强综合结果</b><br>Moment HieA2M-DGQC + Zero：
      mAP 9.16，G-mIoU@3 39.77。</div>
    <div class="card"><b>最稳定跨架构模块</b><br>Quality 在 EaTR 与 Flash-VTG 上均提高 mAP 和 G-mIoU@3。</div>
    <div class="card"><b>拒答增强证据</b><br>EaTR Dual：Rej-F1 39.35→48.12，G-mIoU@3 16.82→21.10。</div>
    <div class="card"><b>集合输出证据</b><br>Learned Dedup 相对 Direct Top-3：平均 mAP +0.48。</div>
  </div>
  <div class="warn"><b>结论边界：</b>当前均为 validation 证据。最终论文仍需统一 parent、
  三随机种子、冻结阈值和一次性 test；训练中结果不作为最终论文数字。</div>
</div>

<div class="page">
  <h1>一、指标与核心有效结果</h1>
  <div class="grid">
    <div class="card"><b>AUROC</b><br>不依赖固定阈值的 positive/null 判别能力。</div>
    <div class="card"><b>Rej-F1@0.4</b><br>existence threshold=0.4 时的拒答 F1。</div>
    <div class="card"><b>mAP</b><br>时间定位平均精度，反映候选边界与排序质量。</div>
    <div class="card"><b>G-mIoU@3</b><br>同时考虑 null 拒答与 top-3 定位的 generalized 指标。</div>
  </div>
  {moment}
  {quality_eatr}
</div>

<div class="page">
  <h1>二、跨架构 Quality 与拒答增强</h1>
  {quality_flash}
  {dual}
  <div class="claim">
    <b>可汇报的阶段性判断：</b>
    Quality-aware ranking 是目前最稳定的可插拔模块；Dual Grounding 对拒答明显有效，
    但更适合作为 DETR 类骨干的可选增强。二者直接组合在 Moment/EaTR/QD 上尚未形成稳定净收益，
    因此论文主方法不采用“模块越多越好”的叙事。
  </div>
</div>

<div class="page">
  <h1>三、学习式事件去重</h1>
  {selector}
  <h3>方法说明</h3>
  <p>对候选对 (i,j) 预测其属于同一真实事件的概率，再按“候选质量 − 重复惩罚”
  增量选框。它不同于固定 IoU NMS：即使两个框重叠，只要内容证据表明属于不同事件，
  仍可同时保留。</p>
  <div class="claim"><b>默认建议：</b>采用 Learned Dedup Top-3。
  Soft Count 可作为召回导向消融，但不进入默认 U；Boundary Fusion 与 Soft Count
  指标相同，没有额外贡献。</div>
  {running}
</div>

<div class="page">
  <h1>四、严谨性审计与论文落地</h1>
  <h2>QD-DETR 公平 continued-control 审计</h2>
  <p class="subtitle">该矩阵用于判断早期 QD 巨大 G-mIoU 增益究竟来自新增模块，还是来自继续训练/门控校准。
  Quality 行仍在训练，因此本表只用于审计，不进入正向主结果。</p>
  <table class="metrics">
    <thead><tr><th>公平设置</th>{metric_header()}</tr></thead>
    <tbody>{qd_rows}</tbody>
  </table>
  <div class="audit"><b>审计结论：</b>Continued Control 本身已达到 G-mIoU@3 35.23，
  因此早期从 3.14 跳到 42 左右不能全部归因于 Quality。公平矩阵中 Quality
  暂未超过 continued control，Dual 仅 mAP 小幅提高。为避免夸大结论，本报告不把
  QD 作为 Quality 的正向跨骨干证据。</div>

  <h2>当前可以形成的论文贡献</h2>
  <ol>
    <li><b>Independent Null Verification：</b>将高召回 existence 与最终判空解耦，
      用 rescue/veto 缓解正样本误拒。</li>
    <li><b>Quality-aware Temporal Ranking：</b>显式学习边界质量，修正前景分数与定位质量不一致。</li>
    <li><b>Learned Event Dedup：</b>学习事件级重复关系，替代 Direct Top-K 与纯几何抑制。</li>
    <li><b>Optional Dual Grounding：</b>在定位基本不退化时增强拒答，作为 DETR 可选分支。</li>
  </ol>

  <h2>提交论文前必须完成</h2>
  <ul>
    <li>在至少三个代表骨干上完成严格配对的 B / Q / Z / Q+Z / Q+Z+P；</li>
    <li>对最终 U 补 seed2024、seed2025，并报告 mean ± std；</li>
    <li>固定 tau_gate、tau_zero、tau_veto 和 selector 参数后一次性评估 test；</li>
    <li>报告 paired bootstrap 95% CI、参数量、推理时延和平均输出框数；</li>
    <li>将完全解耦的 Z(no Counter) 与当前 HieA2M-parent Zero 进行公平对照。</li>
  </ul>
</div>

<div class="page">
  <h1>附录：指标来源与可追溯文件</h1>
  <p>主表所有数字均直接读取下列 JSON；Learned Dedup 为五组已完成设置的算术平均。
  选定方法均具备四项指标，因此本次无需为补字段额外启动训练或评估。</p>
  <div class="source-list">{source_rows}</div>
  <p class="footer-note">报告生成脚本：
  <span class="path">scripts/generate_teacher_progress_report.py</span>。
  本文只保留有方法意义的正向结果和一项必要的公平性审计；CG、Counter、
  完整负交互组合等失败实验不进入老师汇报主表。</p>
</div>

</body></html>"""

    OUT_HTML.write_text(document, encoding="utf-8")
    print(OUT_HTML)


if __name__ == "__main__":
    main()
