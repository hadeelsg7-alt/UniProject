"""Executive PDF report built from `insights.compute_insights` (real analysis results only).

Layout: A4 portrait, dark-green and white identity, numbered sections 1-10, bar charts drawn
directly in the PDF (no plotting dependency). Arabic is shaped with HarfBuzz (fpdf2 text
shaping) and every cell or line is set right-to-left when its text is Arabic, so evidence
quotes render correctly. The bundled DejaVu Sans font covers Latin and Arabic.
"""
from __future__ import annotations

import re
from typing import Iterable

from fpdf import FPDF

import config
import tools

FONT_DIR = config.BASE_DIR / "assets" / "fonts"
GREEN, GREEN_2, ACCENT, TINT, RULE = (14, 59, 44), (21, 90, 64), (15, 122, 81), (243, 248, 245), (205, 222, 212)
INK, MUTED, WHITE = (16, 38, 29), (90, 110, 100), (255, 255, 255)
SEV_COL = {"Critical": (208, 59, 59), "High": (236, 131, 90), "Medium": (250, 178, 25), "Low": (12, 163, 12)}
PRIO_COL = {"P1": (13, 54, 107), "P2": (28, 92, 171), "P3": (57, 135, 229), "P4": (134, 182, 239)}
_AR = re.compile(r"[؀-ۿݐ-ݿﭐ-﷿ﹰ-﻿]")
_LATIN = re.compile(r"[A-Za-z]")


def is_arabic(text: str) -> bool:
    """Arabic-dominant text is laid out right-to-left."""
    t = str(text)
    return len(_AR.findall(t)) > len(_LATIN.findall(t))


class ExecutiveReport(FPDF):
    def __init__(self, label: str):
        super().__init__(orientation="P", unit="mm", format="A4")
        self.label = label
        self.add_font("DV", "", str(FONT_DIR / "DejaVuSans.ttf"))
        self.add_font("DV", "B", str(FONT_DIR / "DejaVuSans-Bold.ttf"))
        self.set_auto_page_break(True, margin=18)
        self.set_margins(16, 16, 16)
        self.set_title(f"{config.APP_NAME} — Executive Report")
        self.set_author("Research prototype (multi-agent feedback analysis)")
        self.set_text_shaping(True, direction="ltr")
        self.cover = True

    # ---- page furniture -----------------------------------------------------
    def header(self):
        if self.cover:
            return
        self.set_fill_color(*GREEN)
        self.rect(0, 0, self.w, 9, "F")
        self._font(7.5, color=WHITE)
        self.set_xy(16, 2.6)
        self.cell(0, 4, f"{config.APP_TITLE_EN}  ·  Executive Report")
        self.set_y(16)

    def footer(self):
        self.set_y(-12)
        self.set_draw_color(*RULE)
        self.line(16, self.get_y() - 1.5, self.w - 16, self.get_y() - 1.5)
        self._font(7.5, color=MUTED)
        self.cell(0, 5, self.label)
        self.set_x(16)
        self.cell(0, 5, f"Page {self.page_no()} / {{nb}}", align="R")

    # ---- text helpers -------------------------------------------------------
    def _font(self, size: float, bold: bool = False, color=INK):
        self.set_font("DV", "B" if bold else "", size)
        self.set_text_color(*color)

    def para(self, text: str, size: float = 9.5, bold: bool = False, color=INK, h: float = 5.2, align: str | None = None):
        rtl = is_arabic(text)
        self.set_text_shaping(True, direction="rtl" if rtl else "ltr")
        self._font(size, bold, color)
        self.multi_cell(0, h, text, align=align or ("R" if rtl else "L"), new_x="LMARGIN", new_y="NEXT")
        self.set_text_shaping(True, direction="ltr")

    def bullets(self, items: Iterable[str], size: float = 9.5):
        for it in items:
            self._font(size, color=ACCENT)
            y = self.get_y()
            self.set_xy(self.l_margin, y)
            self.cell(5, 5.2, "•")
            self.set_xy(self.l_margin + 5, y)
            self._font(size)
            self.multi_cell(self.w - self.l_margin - self.r_margin - 5, 5.2, it, new_x="LMARGIN", new_y="NEXT")
            self.ln(0.8)

    def section(self, num: int, title: str, title_ar: str = "", subtitle: str = ""):
        if self.get_y() > self.h - 45:
            self.add_page()
        self.ln(2)
        y = self.get_y()
        self.set_fill_color(*ACCENT)
        self.rect(self.l_margin, y + 0.8, 1.4, 5.6, "F")
        self.set_xy(self.l_margin + 3.5, y)
        self._font(11.5, True, GREEN)
        self.cell(110, 7, f"{num}. {title}")
        if title_ar:
            self.set_text_shaping(True, direction="rtl")
            self._font(10.5, True, ACCENT)
            self.set_xy(self.l_margin, y)
            self.cell(self.w - self.l_margin - self.r_margin, 7, title_ar, align="R")
            self.set_text_shaping(True, direction="ltr")
        self.set_xy(self.l_margin, y + 7.5)
        if subtitle:
            self._font(8, color=MUTED)
            self.set_x(self.l_margin + 3.5)
            self.multi_cell(0, 4.2, subtitle, new_x="LMARGIN", new_y="NEXT")
        self.ln(1)

    def note(self, text: str, fill=TINT, border=ACCENT):
        x, w = self.l_margin, self.w - self.l_margin - self.r_margin
        self._font(8.5, color=INK)
        lines = self.multi_cell(w - 8, 4.6, text, dry_run=True, output="LINES")
        h = 4.6 * len(lines) + 5
        if self.get_y() + h > self.h - 20:
            self.add_page()
        y = self.get_y()
        self.set_fill_color(*fill)
        self.rect(x, y, w, h, "F")
        self.set_fill_color(*border)
        self.rect(x, y, 1.2, h, "F")
        self.set_xy(x + 5, y + 2.5)
        self.multi_cell(w - 8, 4.6, text, new_x="LMARGIN", new_y="NEXT")
        self.set_y(y + h + 3)

    # ---- table with per-cell text direction --------------------------------
    def table_rows(self, headers: list[str], rows: list[list], widths: list[float], size: float = 8.3, aligns: list[str] | None = None):
        total = self.w - self.l_margin - self.r_margin
        widths = [w * total / sum(widths) for w in widths]
        aligns = aligns or ["L"] * len(headers)
        lh, pad = 4.4, 1.6

        def lines_for(text, w, bold=False):
            self.set_text_shaping(True, direction="rtl" if is_arabic(text) else "ltr")
            self._font(size, bold)
            n = len(self.multi_cell(w - 2 * pad, lh, str(text), dry_run=True, output="LINES"))
            self.set_text_shaping(True, direction="ltr")
            return max(n, 1)

        def draw_row(cells, header=False, zebra=False):
            h = max(lines_for(c, w, header) for c, w in zip(cells, widths)) * lh + 2 * pad
            if self.get_y() + h > self.h - 20:
                self.add_page()
                if not header:
                    draw_row(headers, header=True)
            y, x = self.get_y(), self.l_margin
            for c, w, a in zip(cells, widths, aligns):
                if header:
                    self.set_fill_color(*GREEN)
                    self.rect(x, y, w, h, "F")
                elif zebra:
                    self.set_fill_color(*TINT)
                    self.rect(x, y, w, h, "F")
                text = str(c)
                rtl = is_arabic(text)
                self.set_text_shaping(True, direction="rtl" if rtl else "ltr")
                self._font(size, header, WHITE if header else INK)
                self.set_xy(x + pad, y + pad)
                self.multi_cell(w - 2 * pad, lh, text, align="R" if rtl else a, new_x="RIGHT", new_y="TOP")
                x += w
            self.set_text_shaping(True, direction="ltr")
            self.set_draw_color(*RULE)
            self.line(self.l_margin, y + h, self.l_margin + total, y + h)
            self.set_y(y + h)

        def height(cells, header=False):
            return max(lines_for(c, w, header) for c, w in zip(cells, widths)) * lh + 2 * pad

        # keep the header with at least its first row (no orphaned header at the foot of a page)
        if rows and self.get_y() + height(headers, True) + height(rows[0]) > self.h - 20:
            self.add_page()
        draw_row(headers, header=True)
        for k, r in enumerate(rows):
            draw_row(r, zebra=k % 2 == 1)
        self.ln(3)

    # ---- horizontal bar chart -----------------------------------------------
    def hbars(self, title: str, items: list[tuple[str, int, tuple]], unit: str = ""):
        """items: (label, value, rgb). Bars scale to the largest value; values printed at the bar end."""
        label_w, bar_h, gap = 52, 4.2, 1.6
        h = 8 + len(items) * (bar_h + gap)
        if self.get_y() + h > self.h - 20:
            self.add_page()
        self._font(9, True, GREEN)
        self.cell(0, 6, title, new_x="LMARGIN", new_y="NEXT")
        x0 = self.l_margin + label_w
        span = self.w - self.r_margin - x0 - 22
        top = max([v for _, v, _ in items] + [1])
        for label, v, rgb in items:
            y = self.get_y()
            self._font(8.3)
            self.set_xy(self.l_margin, y)
            self.cell(label_w - 2, bar_h, label, align="R")
            self.set_fill_color(235, 241, 237)
            self.rect(x0, y, span, bar_h, "F")
            if v:
                self.set_fill_color(*rgb)
                self.rect(x0, y, max(span * v / top, 0.6), bar_h, "F")
            self._font(8.3, True)
            self.set_xy(x0 + span * v / top + 1.5, y)
            self.cell(20, bar_h, f"{v}{unit}")
            self.set_y(y + bar_h + gap)
        self.ln(3)


def _kpis(pdf: ExecutiveReport, items: list[tuple[str, str, str]]):
    total = pdf.w - pdf.l_margin - pdf.r_margin
    w, gap, y = (total - 3 * 4) / 4, 4, pdf.get_y()
    for k, (label, value, sub) in enumerate(items):
        x = pdf.l_margin + k * (w + gap)
        pdf.set_fill_color(*WHITE)
        pdf.set_draw_color(*RULE)
        pdf.rect(x, y, w, 24, "DF")
        pdf.set_fill_color(*ACCENT)
        pdf.rect(x, y, w, 1.4, "F")
        pdf.set_xy(x + 3, y + 3.5)
        pdf._font(7, True, MUTED)
        pdf.cell(w - 6, 4, label.upper())
        pdf.set_xy(x + 3, y + 8.5)
        pdf._font(16, True, GREEN)
        pdf.cell(w - 6, 8, value)
        pdf.set_xy(x + 3, y + 17.5)
        pdf._font(7, color=MUTED)
        pdf.cell(w - 6, 4, sub)
    pdf.set_y(y + 28)


def build_executive_report(ins: dict) -> bytes:
    """Concise (2-4 page) bilingual executive report. Raises ValueError when nothing has been analysed."""
    cov = ins["coverage"]
    if cov["reviews"] == 0:
        raise ValueError("No analysed reviews: run an analysis before generating the report.")
    pdf = ExecutiveReport(ins["label"])
    pdf.alias_nb_pages()
    pdf.set_auto_page_break(True, margin=16)
    urgent = sum(c for p, c, _ in ins["priority"] if p in ("P1", "P2"))
    hc = sum(c for s_, c, _ in ins["severity"] if s_ in ("High", "Critical"))

    # ---- title band (page 1) -----------------------------------------------------
    pdf.add_page()
    pdf.set_fill_color(*GREEN)
    pdf.rect(0, 0, pdf.w, 46, "F")
    pdf.set_fill_color(*ACCENT)
    pdf.rect(0, 46, pdf.w, 1.6, "F")
    x0 = 16
    if config.LOGO_PATH.exists():
        pdf.image(str(config.LOGO_PATH), x=16, y=8, h=30)
        x0 = 52
    pdf.set_xy(x0, 9)
    avail = pdf.w - x0 - 16
    size = 15.0
    pdf._font(size, True, WHITE)
    while pdf.get_string_width(config.APP_TITLE_EN) > avail and size > 9:  # keep the title inside the band
        size -= 0.5
        pdf._font(size, True, WHITE)
    pdf.cell(avail, 8, config.APP_TITLE_EN, new_x="LMARGIN", new_y="NEXT")
    pdf.set_text_shaping(True, direction="rtl")
    pdf._font(11.5, True, (214, 236, 223))
    pdf.set_x(x0)
    pdf.cell(avail, 7, config.APP_TITLE_AR, align="L", new_x="LMARGIN", new_y="NEXT")
    pdf.set_text_shaping(True, direction="ltr")
    pdf.set_xy(x0, 26)
    pdf._font(9, color=(214, 236, 223))
    pdf.cell(0, 5, f"Executive report on beneficiary feedback · analysis date {cov['analysis_date']}", new_x="LMARGIN", new_y="NEXT")
    lw = pdf.get_string_width(ins["label"]) + 8
    pdf.set_fill_color(*ACCENT)
    pdf.rect(x0, 34, lw, 6.5, "F")
    pdf.set_xy(x0, 34)
    pdf._font(8, True, WHITE)
    pdf.cell(lw, 6.5, ins["label"], align="C")
    pdf.set_y(53)
    _kpis(pdf, [("Reviews analysed", f"{cov['reviews']:,}", f"{cov['apps']} application(s)"),
                ("Service issues", f"{cov['issues']:,}", "evidence-backed"),
                ("High / Critical", f"{hc:,}", "service blocked or disrupted"),
                ("P1 / P2", f"{urgent:,}", "attention first")])

    pdf.section(1, "Executive Summary", "الملخص التنفيذي")
    pdf.bullets(ins["summary"], size=8.8)

    pdf.section(2, "Dataset and Analysis Coverage", "البيانات ونطاق التحليل")
    ex = cov["execution"]
    pdf.table_rows(["Unique reviews in dataset", "Reviews analysed", "Coverage", "Apps", "Execution (LLM / fallback / offline / failed)"],
                   [[f"{cov['dataset_reviews']:,}" if cov.get("dataset_reviews") else "—", f"{cov['reviews']:,}",
                     f"{cov['coverage_pct']}%" if cov.get("coverage_pct") is not None else "—", cov["apps"],
                     f"{ex['LLM']} / {ex['Rule-based fallback']} / {ex['Rule-based (offline)']} / {ex['Failed']}"]],
                   [1.6, 1.3, 1, 1.1, 3], size=8, aligns=["C", "C", "C", "C", "C"])

    pdf.section(3, "Key Service-Quality Findings", "أبرز نتائج جودة الخدمة")
    pdf.hbars("Feedback type (reviews)", [(t, c, ACCENT) for t, c, _ in ins["feedback_types"] if c])
    pdf.hbars("Operational severity (issues; rules R1–R4)", [(t, c, SEV_COL[t]) for t, c, _ in ins["severity"]])

    pdf.section(4, "Top Five Reported Issues", "المشكلات الخمس الأكثر تكرارا")
    if ins["top_issues"]:
        pdf.table_rows(["Issue", "Dimension", "Issues", "P1/P2", "Example evidence (Arabic)"],
                       [[t["issue"], t["dimension"].split(" (")[0], t["count"], t["urgent"],
                         t["examples"][0]["evidence"] if t["examples"] else ""] for t in ins["top_issues"][:5]],
                       [2.4, 1.9, 0.9, 0.9, 4.6], size=7.8, aligns=["L", "L", "C", "C", "R"])
    else:
        pdf.para("No operational issue was extracted from the analysed reviews.", size=9)

    pdf.section(5, "Priority Distribution", "توزيع الأولويات",
                "Research-defined Impact × Urgency matrix informed by ITIL 4 (not prescribed by it). Missing information "
                "never lowers a priority; such cases are marked provisional for human verification.")
    pdf.hbars("Issues by priority", [(p, c, PRIO_COL[p]) for p, c, _ in ins["priority"]])

    pdf.section(6, "Most Affected EGSQUAL Dimensions", "أبعاد جودة الخدمة الأكثر تأثرا",
                "Operational categories with a provisional EGSQUAL dimension (Aljukhadar et al., 2022); definitions not yet verified.")
    if ins["dimensions"]:
        pdf.table_rows(["Dimension", "EGSQUAL (provisional)", "Issues", "Share", "P1/P2", "Most affected app"],
                       [[d["name"].split(" (")[0], d["egsqual"] or "—", d["issues"], f"{d['share']}%", d["urgent"],
                         d["top_apps"][0][0] if d["top_apps"] else "—"] for d in ins["dimensions"][:5]],
                       [2.2, 2.1, 0.9, 0.9, 0.9, 2.6], size=7.8, aligns=["L", "L", "C", "C", "C", "L"])

    pdf.section(7, "Recommended Improvement Actions", "إجراءات التحسين المقترحة", "Derived from Agent 3's decisions on each review.")
    if ins["improvements"]:
        pdf.table_rows(["Action", "Application", "Handling team", "Tickets", "P1/P2"],
                       [[g["action"], g["app"], g["team"], g["tickets"], g["urgent"]] for g in ins["improvements"][:5]],
                       [2.6, 2.2, 2.6, 0.9, 0.9], size=7.8, aligns=["L", "L", "L", "C", "C"])

    pdf.section(8, "Simulated Escalation Summary", "ملخص التصعيد (محاكاة)")
    sla = ins.get("sla_targets") or tools.sla_targets_kb()["targets"]
    esc = dict(ins["escalation"])
    pdf.para(f"Escalation candidates awaiting human approval: {len(ins['escalation_candidates'])} · complaint mentioned but "
             f"unverifiable: {len(ins['escalation_pending'])} · not eligible: {esc.get('NOT_ELIGIBLE', 0)} · not applicable: "
             f"{esc.get('NOT_APPLICABLE', 0)}.", size=8.8)
    pdf.para("Simulated SLA targets (elapsed clock hours, research-defined): "
             + " · ".join(f"{p} respond {sla[p]['response_hours']} h / resolve {sla[p]['resolution_hours']} h" for p in ("P1", "P2", "P3", "P4") if p in sla)
             + ". Nothing is submitted to AMER, the DGA or any government system.", size=8, color=MUTED)

    pdf.section(9, "Management Recommendations", "توصيات للإدارة")
    if ins["recommendations"]:
        for k, r in enumerate(ins["recommendations"], start=1):
            pdf.para(f"{k}. {r['title']}", size=9.2, bold=True, color=GREEN, h=4.8)
            pdf.para(r["basis"], size=8.4, h=4.4)
    else:
        pdf.note(f"Not generated: only {cov['reviews']} review(s) were analysed. Management-level recommendations require at least "
                 f"{ins['min_reviews']} analysed reviews so that they do not generalise from a very small sample.")

    pdf.section(10, "Research Limitations", "حدود البحث")
    pdf.bullets(["Research prototype: the figures count the framework's decisions; accuracy is measured separately against "
                 "human-verified labels.",
                 "Severity, priority, SLA, routing and escalation rules are research-defined; ITIL 4, ISO/IEC 20000-1:2018 and "
                 "ISO 10002:2018 inform the design, with no claim of compliance or official approval.",
                 "EGSQUAL dimension definitions and the crosswalk are unverified until the full paper is reviewed.",
                 f"Coverage is limited to the {cov['reviews']:,} analysed review(s); rule-based fallback results are counted "
                 "separately from LLM results."], size=8.2)
    c = tools.egsqual_reference()["citation"]
    pdf.para(f"Reference: {c['authors']} ({c['year']}). {c['title']}. {c['journal']}, {c['volume']}, {c['article']}. "
             f"https://doi.org/{c['doi']}", size=7.2, color=MUTED, h=3.8)
    return bytes(pdf.output())
