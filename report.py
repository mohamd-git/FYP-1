"""
report.py
=========
Generate a professional **PDF inspection report** from the defect register
(``data/agv.db``) produced by an inspection run.

After the AGV finishes traversing the track, this turns the consolidated,
geo-tagged maintenance log into a single hand-it-to-maintenance document:

  * cover page with run metadata and headline figures
  * an auto-written executive summary plus charts
  * a prioritised maintenance schedule (every defect, ranked by urgency)
  * per-defect detail cards with the captured crop image

This is pure software and directly supports Objective 4 (a digitised
maintenance-recording subsystem with review and export). Detection contract and
pipeline are untouched -- this only reads the register.

Run:
    python report.py                                   # config.yaml + data/agv.db
    python report.py --out evaluation/my_report.pdf    # custom output path
"""
from __future__ import annotations

import argparse
import json
import os
from datetime import datetime, timezone
from pathlib import Path

import matplotlib
matplotlib.use("Agg")  # headless
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch

from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER, TA_LEFT, TA_RIGHT
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import mm
from reportlab.platypus import (
    HRFlowable, Image, KeepTogether, PageBreak, Paragraph, SimpleDocTemplate,
    Spacer, Table, TableStyle,
)

from src.config import PROJECT_ROOT, load_config, resolve_path
from src.storage.db import Database

# --------------------------------------------------------------------------- #
# Palette
# --------------------------------------------------------------------------- #
INK    = colors.HexColor("#14233A")   # near-black navy
STEEL  = colors.HexColor("#0E6E8C")   # rail-steel teal (brand accent)
STEEL2 = colors.HexColor("#0A5266")
PAPER  = colors.HexColor("#F4F7FA")
LINE   = colors.HexColor("#D5DEE6")
MUTED  = colors.HexColor("#5B6B7B")
WHITE  = colors.white

SEV_COLOR = {
    "High":   colors.HexColor("#C62828"),
    "Medium": colors.HexColor("#EF6C00"),
    "Low":    colors.HexColor("#F9A825"),
}
BAND_COLOR = {
    "Immediate": colors.HexColor("#B71C1C"),
    "Schedule":  colors.HexColor("#EF6C00"),
    "Routine":   colors.HexColor("#1565C0"),
    "Monitor":   colors.HexColor("#2E7D32"),
}
BAND_LABEL = {
    "Immediate": "Immediate (within 24 h)",
    "Schedule":  "Schedule",
    "Routine":   "Routine",
    "Monitor":   "Monitor",
}
BAND_ORDER = ["Immediate", "Schedule", "Routine", "Monitor"]
SEV_ORDER = ["High", "Medium", "Low"]


def band_of(urgency: float) -> str:
    if urgency >= 75:
        return "Immediate"
    if urgency >= 50:
        return "Schedule"
    if urgency >= 25:
        return "Routine"
    return "Monitor"


def nice_class(cls: str) -> str:
    return cls.replace("_", " ").title()


def short_dt(iso: str) -> str:
    try:
        return datetime.fromisoformat(iso.replace("Z", "+00:00")).strftime("%Y-%m-%d %H:%M:%S")
    except Exception:
        return str(iso)


# --------------------------------------------------------------------------- #
# Charts (matplotlib -> PNG)
# --------------------------------------------------------------------------- #
def _hex(c: colors.Color) -> str:
    return "#%02x%02x%02x" % (int(c.red * 255), int(c.green * 255), int(c.blue * 255))


def chart_severity_donut(defects, out: Path) -> Path:
    counts = {s: sum(1 for d in defects if d["severity"] == s) for s in SEV_ORDER}
    counts = {k: v for k, v in counts.items() if v > 0}
    fig, ax = plt.subplots(figsize=(3.1, 3.1), dpi=150)
    if counts:
        ax.pie(
            list(counts.values()),
            labels=[f"{k}\n{v}" for k, v in counts.items()],
            colors=[_hex(SEV_COLOR[k]) for k in counts],
            startangle=90, counterclock=False,
            wedgeprops=dict(width=0.42, edgecolor="white", linewidth=2),
            textprops=dict(color=_hex(INK), fontsize=9, weight="bold"),
        )
        ax.text(0, 0, f"{sum(counts.values())}\ndefects", ha="center", va="center",
                fontsize=12, weight="bold", color=_hex(INK))
    ax.set_title("By severity", fontsize=11, weight="bold", color=_hex(INK), pad=8)
    fig.tight_layout()
    fig.savefig(out, transparent=True)
    plt.close(fig)
    return out


def chart_band_bar(defects, out: Path) -> Path:
    counts = {b: 0 for b in BAND_ORDER}
    for d in defects:
        counts[band_of(d["urgency_score"])] += 1
    fig, ax = plt.subplots(figsize=(3.6, 3.1), dpi=150)
    ys = list(range(len(BAND_ORDER)))[::-1]
    vals = [counts[b] for b in BAND_ORDER]
    ax.barh(ys, vals, color=[_hex(BAND_COLOR[b]) for b in BAND_ORDER], height=0.62)
    for y, b, v in zip(ys, BAND_ORDER, vals):
        if v:
            ax.text(v + max(vals) * 0.02 + 0.05, y, str(v), va="center",
                    fontsize=10, weight="bold", color=_hex(INK))
    ax.set_yticks(ys)
    ax.set_yticklabels([BAND_LABEL[b] for b in BAND_ORDER], fontsize=9, color=_hex(INK))
    ax.set_xlim(0, max(vals + [1]) * 1.18)
    ax.set_xticks([])
    for sp in ("top", "right", "bottom"):
        ax.spines[sp].set_visible(False)
    ax.spines["left"].set_color(_hex(LINE))
    ax.set_title("Recommended action", fontsize=11, weight="bold", color=_hex(INK), pad=8)
    fig.tight_layout()
    fig.savefig(out, transparent=True)
    plt.close(fig)
    return out


def chart_class_bar(defects, out: Path) -> Path:
    counts: dict[str, int] = {}
    for d in defects:
        counts[d["defect_class"]] = counts.get(d["defect_class"], 0) + 1
    items = sorted(counts.items(), key=lambda kv: kv[1])
    fig, ax = plt.subplots(figsize=(3.6, 3.1), dpi=150)
    ys = list(range(len(items)))
    vals = [v for _, v in items]
    ax.barh(ys, vals, color=_hex(STEEL), height=0.6)
    for y, v in zip(ys, vals):
        ax.text(v + max(vals) * 0.02 + 0.05, y, str(v), va="center",
                fontsize=10, weight="bold", color=_hex(INK))
    ax.set_yticks(ys)
    ax.set_yticklabels([nice_class(k) for k, _ in items], fontsize=9, color=_hex(INK))
    ax.set_xlim(0, max(vals + [1]) * 1.18)
    ax.set_xticks([])
    for sp in ("top", "right", "bottom"):
        ax.spines[sp].set_visible(False)
    ax.spines["left"].set_color(_hex(LINE))
    ax.set_title("By defect type", fontsize=11, weight="bold", color=_hex(INK), pad=8)
    fig.tight_layout()
    fig.savefig(out, transparent=True)
    plt.close(fig)
    return out


def chart_chainage_strip(defects, out: Path) -> Path:
    chmain = [d["chainage_m"] for d in defects]
    end = max(chmain) if chmain else 1.0
    fig, ax = plt.subplots(figsize=(7.3, 1.45), dpi=150)
    ax.hlines(0, 0, end, color=_hex(INK), linewidth=3, zorder=1)
    for d in defects:
        ax.scatter(d["chainage_m"], 0, s=130, zorder=3,
                   color=_hex(SEV_COLOR.get(d["severity"], STEEL)),
                   edgecolor="white", linewidth=1.2)
    ax.set_xlim(-end * 0.02, end * 1.02)
    ax.set_ylim(-1, 1)
    ax.set_yticks([])
    ax.set_xlabel("Chainage along track (m)", fontsize=9, color=_hex(MUTED))
    ax.tick_params(axis="x", colors=_hex(MUTED), labelsize=9)
    for sp in ("top", "right", "left"):
        ax.spines[sp].set_visible(False)
    ax.spines["bottom"].set_color(_hex(LINE))
    ax.set_title("Defect positions along the inspected corridor", fontsize=11,
                 weight="bold", color=_hex(INK), pad=6, loc="left")
    fig.tight_layout()
    fig.savefig(out, transparent=True)
    plt.close(fig)
    return out


# --------------------------------------------------------------------------- #
# Paragraph styles
# --------------------------------------------------------------------------- #
def build_styles():
    ss = getSampleStyleSheet()
    s = {}
    s["title"] = ParagraphStyle("title", parent=ss["Title"], textColor=WHITE,
                                fontSize=24, leading=28, alignment=TA_LEFT, fontName="Helvetica-Bold")
    s["subtitle"] = ParagraphStyle("subtitle", textColor=colors.HexColor("#BfeAf2"),
                                   fontSize=12, leading=15, alignment=TA_LEFT)
    s["h2"] = ParagraphStyle("h2", parent=ss["Heading2"], textColor=INK,
                             fontSize=14, leading=17, spaceBefore=4, spaceAfter=6,
                             fontName="Helvetica-Bold")
    s["body"] = ParagraphStyle("body", parent=ss["BodyText"], textColor=INK,
                               fontSize=10, leading=14.5)
    s["small"] = ParagraphStyle("small", textColor=MUTED, fontSize=8, leading=10.5)
    s["cell"] = ParagraphStyle("cell", fontSize=8.4, leading=10.5, textColor=INK)
    s["cellb"] = ParagraphStyle("cellb", fontSize=8.4, leading=10.5, textColor=INK,
                                fontName="Helvetica-Bold")
    s["cellw"] = ParagraphStyle("cellw", fontSize=8.4, leading=10.5, textColor=WHITE,
                                fontName="Helvetica-Bold", alignment=TA_CENTER)
    s["meta_k"] = ParagraphStyle("meta_k", fontSize=9, leading=12, textColor=MUTED)
    s["meta_v"] = ParagraphStyle("meta_v", fontSize=9, leading=12, textColor=INK,
                                 fontName="Helvetica-Bold")
    s["detail_k"] = ParagraphStyle("detail_k", fontSize=8.2, leading=11, textColor=MUTED)
    s["detail_v"] = ParagraphStyle("detail_v", fontSize=9, leading=11.5, textColor=INK,
                                   fontName="Helvetica-Bold")
    return s


# --------------------------------------------------------------------------- #
# Footer / header on every page
# --------------------------------------------------------------------------- #
def _decorate(canvas, doc):
    canvas.saveState()
    w, h = A4
    # footer line
    canvas.setStrokeColor(LINE)
    canvas.setLineWidth(0.6)
    canvas.line(18 * mm, 14 * mm, w - 18 * mm, 14 * mm)
    canvas.setFont("Helvetica", 7.5)
    canvas.setFillColor(MUTED)
    canvas.drawString(18 * mm, 10 * mm,
                      "AGV Rail Defect Detection — Inspection & Smart Maintenance System")
    canvas.drawRightString(w - 18 * mm, 10 * mm, f"Page {canvas.getPageNumber()}")
    canvas.restoreState()


# --------------------------------------------------------------------------- #
# Build
# --------------------------------------------------------------------------- #
def generate(db_path: Path, out_path: Path, config: dict) -> Path:
    db = Database(db_path).connect()
    defects = db.all_defects()       # already urgency-desc
    db.close()

    S = build_styles()
    tmp = Path(os.environ.get("TEMP", "."))
    story = []

    # ---- derived stats ----
    n = len(defects)
    sev_counts = {s: sum(1 for d in defects if d["severity"] == s) for s in SEV_ORDER}
    band_counts = {b: 0 for b in BAND_ORDER}
    for d in defects:
        band_counts[band_of(d["urgency_score"])] += 1
    chain = [d["chainage_m"] for d in defects]
    dist = (max(chain) - min(chain)) if len(chain) > 1 else (chain[0] if chain else 0.0)
    span_end = max(chain) if chain else 0.0
    avg_conf = (sum(d["confidence"] for d in defects) / n) if n else 0.0
    model = defects[0]["model"] if n else "n/a"
    speed = (config.get("localisation", {}) or {}).get("inspection_speed_mps", 0.4)
    gen_dt = datetime.now().strftime("%d %B %Y, %H:%M")
    report_id = "AGV-" + datetime.now().strftime("%Y%m%d-%H%M%S")

    # =====================================================================
    # COVER
    # =====================================================================
    banner = Table(
        [[Paragraph("TRACK INSPECTION REPORT", S["title"])],
         [Paragraph("AGV for Rail Defect Detection &mdash; Inspection &amp; Smart Maintenance System", S["subtitle"])]],
        colWidths=[174 * mm],
    )
    banner.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, -1), INK),
        ("LEFTPADDING", (0, 0), (-1, -1), 16),
        ("RIGHTPADDING", (0, 0), (-1, -1), 16),
        ("TOPPADDING", (0, 0), (0, 0), 18),
        ("BOTTOMPADDING", (0, 0), (0, 0), 2),
        ("TOPPADDING", (0, 1), (0, 1), 2),
        ("BOTTOMPADDING", (0, 1), (0, 1), 18),
    ]))
    story += [banner, Spacer(1, 8 * mm)]

    # metadata block
    meta_rows = [
        ["Report ID", report_id, "Generated", gen_dt],
        ["Detector model", str(model), "Inference device", "CPU (Coral Edge TPU in hardware phase)"],
        ["Corridor inspected", f"{span_end:,.1f} m", "Inspection speed", f"{speed} m/s"],
        ["Defects logged", str(n), "Avg. confidence", f"{avg_conf:.2f}"],
    ]
    mt = Table(
        [[Paragraph(r[0], S["meta_k"]), Paragraph(r[1], S["meta_v"]),
          Paragraph(r[2], S["meta_k"]), Paragraph(r[3], S["meta_v"])] for r in meta_rows],
        colWidths=[30 * mm, 57 * mm, 30 * mm, 57 * mm],
    )
    mt.setStyle(TableStyle([
        ("LINEBELOW", (0, 0), (-1, -2), 0.5, LINE),
        ("TOPPADDING", (0, 0), (-1, -1), 6),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 6),
        ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
    ]))
    story += [mt, Spacer(1, 9 * mm)]

    # headline stat band
    def stat_cell(value, label, color):
        return Table([[Paragraph(f"<font size=30><b>{value}</b></font>", S["cell"])],
                      [Paragraph(label, ParagraphStyle("sl", fontSize=9, leading=11,
                                 textColor=WHITE, alignment=TA_CENTER))]],
                     colWidths=[56 * mm], rowHeights=[16 * mm, 8 * mm],
                     style=TableStyle([
                         ("BACKGROUND", (0, 0), (-1, -1), color),
                         ("TEXTCOLOR", (0, 0), (-1, -1), WHITE),
                         ("ALIGN", (0, 0), (-1, -1), "CENTER"),
                         ("VALIGN", (0, 0), (0, 0), "BOTTOM"),
                         ("VALIGN", (0, 1), (0, 1), "TOP"),
                     ]))
    band_tbl = Table([[
        stat_cell(n, "TOTAL DEFECTS", STEEL),
        stat_cell(band_counts["Immediate"], "IMMEDIATE (24 h)", BAND_COLOR["Immediate"]),
        stat_cell(sev_counts["High"], "HIGH SEVERITY", SEV_COLOR["High"]),
    ]], colWidths=[58 * mm, 58 * mm, 58 * mm])
    band_tbl.setStyle(TableStyle([("LEFTPADDING", (0, 0), (-1, -1), 1),
                                  ("RIGHTPADDING", (0, 0), (-1, -1), 1),
                                  ("VALIGN", (0, 0), (-1, -1), "MIDDLE")]))
    story += [band_tbl, Spacer(1, 10 * mm)]
    story += [Paragraph(
        "This report is generated automatically from the AGV's on-board defect register at the "
        "end of an inspection run. Every defect below is geo-tagged, severity-graded, and ranked "
        "by a 0&ndash;100 urgency score so that maintenance can be planned condition-first.", S["small"])]
    story += [PageBreak()]

    # =====================================================================
    # EXECUTIVE SUMMARY
    # =====================================================================
    story += [Paragraph("Executive summary", S["h2"]),
              HRFlowable(width="100%", thickness=1.4, color=STEEL, spaceAfter=8)]

    top = defects[0] if defects else None
    narrative = (
        f"This automated inspection traversed approximately <b>{span_end:,.1f} m</b> of track and "
        f"recorded <b>{n}</b> defect{'s' if n != 1 else ''}. "
        f"Of these, <b>{band_counts['Immediate']}</b> require <b>immediate</b> attention (within 24 hours) and "
        f"<b>{sev_counts['High']}</b> {'are' if sev_counts['High'] != 1 else 'is'} graded <b>High</b> severity. "
    )
    if top:
        narrative += (
            f"The most urgent item is a <b>{nice_class(top['defect_class'])}</b> at chainage "
            f"<b>{top['chainage_m']:,.1f} m</b> (urgency <b>{top['urgency_score']}/100</b>, "
            f"confidence {top['confidence']:.2f}). "
        )
    narrative += ("All detections are listed in the prioritised maintenance schedule that follows, "
                  "ranked highest-urgency first, with per-defect evidence on the final pages.")
    story += [Paragraph(narrative, S["body"]), Spacer(1, 6 * mm)]

    # charts row 1: donut + band bar + class bar
    c1 = chart_severity_donut(defects, tmp / "rep_sev.png")
    c2 = chart_band_bar(defects, tmp / "rep_band.png")
    c3 = chart_class_bar(defects, tmp / "rep_class.png")
    charts = Table([[Image(str(c1), width=52 * mm, height=52 * mm),
                     Image(str(c2), width=60 * mm, height=52 * mm),
                     Image(str(c3), width=60 * mm, height=52 * mm)]],
                   colWidths=[54 * mm, 60 * mm, 60 * mm])
    charts.setStyle(TableStyle([("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
                                ("ALIGN", (0, 0), (-1, -1), "CENTER")]))
    story += [charts, Spacer(1, 3 * mm)]

    # chainage strip
    c4 = chart_chainage_strip(defects, tmp / "rep_chain.png")
    story += [Image(str(c4), width=174 * mm, height=34 * mm), Spacer(1, 2 * mm)]
    story += [PageBreak()]

    # =====================================================================
    # PRIORITISED MAINTENANCE SCHEDULE
    # =====================================================================
    story += [Paragraph("Prioritised maintenance schedule", S["h2"]),
              HRFlowable(width="100%", thickness=1.4, color=STEEL, spaceAfter=8)]

    header = [Paragraph(h, S["cellw"]) for h in
              ["#", "Defect", "Severity", "Urgency", "Recommended action", "Chainage", "GPS (lat, lng)", "Conf."]]
    data = [header]
    sev_bg_rows = []
    for i, d in enumerate(defects, start=1):
        sev = d["severity"]
        data.append([
            Paragraph(str(i), S["cell"]),
            Paragraph(nice_class(d["defect_class"]), S["cellb"]),
            Paragraph(sev, S["cellw"]),
            Paragraph(f"{d['urgency_score']}", S["cellb"]),
            Paragraph(d["recommended_action"], S["cell"]),
            Paragraph(f"{d['chainage_m']:,.1f} m", S["cell"]),
            Paragraph(f"{d['lat']:.5f}, {d['lng']:.5f}", S["cell"]),
            Paragraph(f"{d['confidence']:.2f}", S["cell"]),
        ])
        sev_bg_rows.append((i, SEV_COLOR.get(sev, MUTED)))

    tbl = Table(data, colWidths=[7 * mm, 27 * mm, 17 * mm, 14 * mm, 52 * mm, 17 * mm, 28 * mm, 12 * mm],
                repeatRows=1)
    style = [
        ("BACKGROUND", (0, 0), (-1, 0), INK),
        ("TOPPADDING", (0, 0), (-1, -1), 4),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 4),
        ("LEFTPADDING", (0, 0), (-1, -1), 5),
        ("RIGHTPADDING", (0, 0), (-1, -1), 5),
        ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
        ("LINEBELOW", (0, 0), (-1, -1), 0.4, LINE),
        ("ROWBACKGROUNDS", (0, 1), (-1, -1), [WHITE, PAPER]),
    ]
    for ridx, color in sev_bg_rows:
        style.append(("BACKGROUND", (2, ridx), (2, ridx), color))   # severity chip cell
    tbl.setStyle(TableStyle(style))
    story += [tbl]
    story += [PageBreak()]

    # =====================================================================
    # DEFECT DETAIL CARDS (with crop evidence)
    # =====================================================================
    story += [Paragraph("Defect evidence &amp; detail", S["h2"]),
              HRFlowable(width="100%", thickness=1.4, color=STEEL, spaceAfter=8)]

    for i, d in enumerate(defects, start=1):
        sev = d["severity"]
        band = band_of(d["urgency_score"])
        # image cell
        img_path = resolve_path(d["image_ref"]) if d.get("image_ref") else None
        if img_path and Path(img_path).is_file():
            img_flow = Image(str(img_path), width=38 * mm, height=38 * mm, kind="proportional")
        else:
            img_flow = Paragraph("(no crop image)", S["small"])

        # details (key/value grid)
        kv = [
            ["Defect ID", d["defect_key"], "Detector", str(d["model"])],
            ["Severity", sev, "Urgency", f"{d['urgency_score']} / 100"],
            ["Action band", BAND_LABEL[band], "Confidence", f"{d['confidence']:.2f}"],
            ["Chainage", f"{d['chainage_m']:,.1f} m", "GPS", f"{d['lat']:.5f}, {d['lng']:.5f}"],
            ["First seen", short_dt(d["first_seen"]), "Frames", str(d["frame_count"])],
        ]
        kv_tbl = Table(
            [[Paragraph(r[0], S["detail_k"]), Paragraph(str(r[1]), S["detail_v"]),
              Paragraph(r[2], S["detail_k"]), Paragraph(str(r[3]), S["detail_v"])] for r in kv],
            colWidths=[22 * mm, 40 * mm, 20 * mm, 40 * mm],
        )
        kv_tbl.setStyle(TableStyle([
            ("TOPPADDING", (0, 0), (-1, -1), 2.5), ("BOTTOMPADDING", (0, 0), (-1, -1), 2.5),
            ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
        ]))
        title = Table([[Paragraph(f"&nbsp;{i}.&nbsp; {nice_class(d['defect_class'])}",
                                  ParagraphStyle("ct", fontSize=11, textColor=WHITE,
                                                 fontName="Helvetica-Bold", leading=14)),
                        Paragraph(BAND_LABEL[band], S["cellw"])]],
                      colWidths=[100 * mm, 22 * mm])
        title.setStyle(TableStyle([
            ("BACKGROUND", (0, 0), (0, 0), INK),
            ("BACKGROUND", (1, 0), (1, 0), BAND_COLOR[band]),
            ("TOPPADDING", (0, 0), (-1, -1), 5), ("BOTTOMPADDING", (0, 0), (-1, -1), 5),
            ("LEFTPADDING", (0, 0), (-1, -1), 6), ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
        ]))
        action = Paragraph(f"<b>Recommended action:</b> {d['recommended_action']}", S["body"])
        body = Table([[img_flow, kv_tbl]], colWidths=[44 * mm, 124 * mm])
        body.setStyle(TableStyle([
            ("VALIGN", (0, 0), (-1, -1), "TOP"),
            ("BACKGROUND", (0, 0), (-1, -1), PAPER),
            ("BOX", (0, 0), (-1, -1), 0.5, LINE),
            ("LEFTPADDING", (0, 0), (-1, -1), 6), ("RIGHTPADDING", (0, 0), (-1, -1), 6),
            ("TOPPADDING", (0, 0), (-1, -1), 6), ("BOTTOMPADDING", (0, 0), (-1, -1), 6),
        ]))
        card = KeepTogether([title, body,
                             Table([[action]], colWidths=[168 * mm],
                                   style=TableStyle([
                                       ("BACKGROUND", (0, 0), (-1, -1), colors.HexColor("#FFF6E8")),
                                       ("BOX", (0, 0), (-1, -1), 0.5, LINE),
                                       ("LEFTPADDING", (0, 0), (-1, -1), 6),
                                       ("RIGHTPADDING", (0, 0), (-1, -1), 6),
                                       ("TOPPADDING", (0, 0), (-1, -1), 5),
                                       ("BOTTOMPADDING", (0, 0), (-1, -1), 5)])),
                             Spacer(1, 5 * mm)])
        story += [card]

    # ---- render ----
    out_path.parent.mkdir(parents=True, exist_ok=True)
    doc = SimpleDocTemplate(
        str(out_path), pagesize=A4,
        leftMargin=18 * mm, rightMargin=18 * mm, topMargin=16 * mm, bottomMargin=20 * mm,
        title="AGV Track Inspection Report", author="AGV Rail Defect Detection System",
    )
    doc.build(story, onFirstPage=_decorate, onLaterPages=_decorate)
    return out_path


def main() -> int:
    ap = argparse.ArgumentParser(description="Generate a PDF inspection report from the defect register.")
    ap.add_argument("--db", default=None, help="path to the SQLite register (default: from config)")
    ap.add_argument("--out", default=None, help="output PDF path (default: evaluation/inspection_report_<ts>.pdf)")
    args = ap.parse_args()

    config = load_config()
    db_path = Path(args.db) if args.db else Database.from_config(config).path
    if args.out:
        out_path = Path(args.out)
    else:
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        out_path = resolve_path(f"evaluation/inspection_report_{ts}.pdf")

    if not Path(db_path).is_file():
        raise SystemExit(f"No register found at {db_path}. Run an inspection first (python run.py).")

    out = generate(Path(db_path), out_path, config)
    print(f"Inspection report written -> {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
