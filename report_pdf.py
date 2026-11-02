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
        self.set_title("Service Quality Executive Report")
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
        self.cell(0, 4, "Service Quality Executive Report  ·  Saudi government mobile applications")
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

    def section(self, num: int, title: str, subtitle: str = ""):
        if self.get_y() > self.h - 60:
            self.add_page()
        self.ln(3)
        self.set_fill_color(*ACCENT)
        self.rect(self.l_margin, self.get_y() + 0.8, 1.6, 6.2, "F")
        self.set_x(self.l_margin + 4)
        self._font(13, True, GREEN)
        self.cell(0, 8, f"{num}. {title}", new_x="LMARGIN", new_y="NEXT")
        if subtitle:
            self._font(8.5, color=MUTED)
            self.set_x(self.l_margin + 4)
            self.multi_cell(0, 4.5, subtitle, new_x="LMARGIN", new_y="NEXT")
        self.ln(2)

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
        label_w, bar_h, gap = 52, 5, 2.2
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
    """Render the 10-section executive report. Raises ValueError when nothing has been analysed."""
    cov = ins["coverage"]
    if cov["reviews"] == 0:
        raise ValueError("No analysed reviews: run an analysis before generating the report.")
    pdf = ExecutiveReport(ins["label"])
    pdf.alias_nb_pages()
    urgent = sum(c for p, c, _ in ins["priority"] if p in ("P1", "P2"))
    hc = sum(c for s, c, _ in ins["severity"] if s in ("High", "Critical"))

    # ---- cover ---------------------------------------------------------------
    pdf.add_page()
    pdf.set_fill_color(*GREEN)
    pdf.rect(0, 0, pdf.w, 112, "F")
    pdf.set_fill_color(*ACCENT)
    pdf.rect(0, 112, pdf.w, 2.5, "F")
    pdf.set_xy(16, 26)
    pdf._font(9, True, (190, 226, 205))
    pdf.cell(0, 6, "EXECUTIVE REPORT", new_x="LMARGIN", new_y="NEXT")
    pdf.set_x(16)
    pdf._font(23, True, WHITE)
    pdf.multi_cell(170, 10.5, "Beneficiary Feedback and Digital Service Quality", align="L", new_x="LMARGIN", new_y="NEXT")
    pdf.ln(1)
    pdf.set_text_shaping(True, direction="rtl")
    pdf._font(14, True, WHITE)
    pdf.set_x(16)
    pdf.multi_cell(pdf.w - 32, 8, "تقرير تنفيذي عن ملاحظات المستفيدين وجودة الخدمات الرقمية", align="R", new_x="LMARGIN", new_y="NEXT")
    pdf.set_text_shaping(True, direction="ltr")
    pdf.ln(3)
    pdf.set_x(16)
    pdf._font(10, color=(214, 236, 223))
    pdf.multi_cell(170, 5.5, "Saudi government mobile applications · analysed by a three-agent decision-support framework",
                   align="L", new_x="LMARGIN", new_y="NEXT")
    pdf.ln(5)
    label_w = pdf.get_string_width(ins["label"]) + 10
    pdf.set_fill_color(*ACCENT)
    pdf.rect(16, pdf.get_y(), label_w, 8, "F")
    pdf.set_x(16)
    pdf._font(8.5, True, WHITE)
    pdf.cell(label_w, 8, ins["label"], align="C", new_x="LMARGIN", new_y="NEXT")
    pdf.set_y(124)
    _kpis(pdf, [("Reviews analysed", f"{cov['reviews']:,}", f"{cov['apps']} application(s)"),
                ("Service issues", f"{cov['issues']:,}", "evidence-backed"),
                ("High / Critical", f"{hc:,}", "service blocked or disrupted"),
                ("P1 / P2", f"{urgent:,}", "require attention first")])
    pdf._font(9, color=MUTED)
    pdf.multi_cell(0, 5, f"Analysis date: {cov['analysis_date']}   ·   Reasoning engine: {', '.join(cov['engines'])}",
                   new_x="LMARGIN", new_y="NEXT")
    pdf.ln(4)
    pdf.note("All figures in this report are counts over the reviews actually analysed by the prototype. They describe what the "
             "framework found; they are not accuracy measurements. Tickets and escalations are simulated: nothing was submitted "
             "to AMER, the DGA or any government system.")
    pdf.cover = False

    # ---- 1 Executive summary ---------------------------------------------------
    pdf.add_page()
    pdf.section(1, "Executive Summary")
    pdf.bullets(ins["summary"])

    # ---- 2 Dataset coverage ----------------------------------------------------
    pdf.section(2, "Dataset Coverage")
    pdf.table_rows(["Measure", "Value"], [
        ["Reviews analysed", f"{cov['reviews']:,}"], ["Applications covered", cov["apps"]],
        ["Service issues extracted", f"{cov['issues']:,}"], ["Analysis date", cov["analysis_date"]],
        ["Reasoning engine", ", ".join(cov["engines"])],
        ["Reviews with two or more issues", cov["multi_issue_reviews"]],
        ["Reviews needing clarification", cov["needs_clarification"]],
        ["Inter-agent review requests", cov["revision_requests"]],
        ["Reviews where an LLM step fell back to rules", cov["llm_fallback_reviews"]],
        ["Synthetic test reviews included", cov["synthetic_reviews"]],
    ], [3, 2])
    if cov["top_apps"]:
        pdf.table_rows(["Application", "Reviews analysed"], [[a, n] for a, n in cov["top_apps"]], [4, 1.4], aligns=["L", "C"])

    # ---- 3 Key indicators --------------------------------------------------------
    pdf.section(3, "Key Service Quality Indicators",
                "Feedback type and sentiment are per review; severity and dimensions are per extracted issue.")
    pdf.hbars("Feedback type (reviews)", [(t, c, ACCENT) for t, c, _ in ins["feedback_types"]])
    pdf.hbars("Sentiment of the text (reviews; reported separately from severity)",
              [(t, c, {"Negative": (208, 59, 59), "Neutral": (150, 165, 158), "Positive": (12, 163, 12)}[t]) for t, c, _ in ins["sentiment"]])
    pdf.hbars("Operational severity (issues)", [(t, c, SEV_COL[t]) for t, c, _ in ins["severity"]])
    if ins["dimensions"]:
        pdf.hbars("Affected service-quality dimensions (issues)", [(d["name"].split(" (")[0], d["issues"], GREEN_2) for d in ins["dimensions"]])

    # ---- 4 Top reported issues ---------------------------------------------------
    pdf.section(4, "Top Reported Issues", "Issues grouped by topic and dimension, with verbatim beneficiary evidence.")
    if ins["top_issues"]:
        rows = []
        for t in ins["top_issues"]:
            ex = t["examples"][0] if t["examples"] else {"evidence": "", "review_id": ""}
            rows.append([t["issue"], t["dimension"], t["count"], t["urgent"], ex["evidence"], ex["review_id"]])
        pdf.table_rows(["Issue", "Dimension", "Issues", "P1/P2", "Example evidence (Arabic)", "Review"], rows,
                       [2.5, 2.3, 1.2, 1.2, 4.0, 1.3], aligns=["L", "L", "C", "C", "R", "L"])
    else:
        pdf.para("No operational issue was extracted from the analysed reviews.")

    # ---- 5 Priority ------------------------------------------------------------
    pdf.section(5, "Priority Distribution", "Assigned by Agent 2's documented rules; never by tone or star rating.")
    pdf.hbars("Issues by priority", [(p, c, PRIO_COL[p]) for p, c, _ in ins["priority"]])
    pdf.table_rows(["Rule", "Definition"], [[r["id"], r["rule"]] for r in tools.severity_rules_kb()["priority_rules"]], [1, 6])
    if ins["high_priority"]:
        hp = ins["high_priority"][:12]
        pdf.para(f"High-priority feedback requiring attention (showing {len(hp)} of {len(ins['high_priority'])} P1/P2 issues):",
                 size=9, bold=True, color=GREEN)
        pdf.ln(1)
        pdf.table_rows(["Priority", "Severity", "Application", "Issue", "Evidence (Arabic)"],
                       [[r["priority"], r["severity"], r["app"], r["description"], r["evidence"]] for r in hp],
                       [1.4, 1.6, 2, 2.8, 4.0], aligns=["C", "C", "L", "L", "R"])

    # ---- 6 EGSQUAL insights --------------------------------------------------------
    ref = tools.egsqual_reference()
    pdf.section(6, "Service Quality Insights (EGSQUAL)",
                f"Operational categories with their provisional EGSQUAL dimension (Aljukhadar et al., {ref['citation']['year']}).")
    if ins["dimensions"]:
        pdf.table_rows(["Dimension", "EGSQUAL (provisional)", "Issues", "Share", "High/Crit.", "P1/P2", "Most affected app"],
                       [[d["name"], d["egsqual"] or "—", d["issues"], f"{d['share']}%", d["high_critical"], d["urgent"],
                         f"{d['top_apps'][0][0]} ({d['top_apps'][0][1]})" if d["top_apps"] else "—"] for d in ins["dimensions"]],
                       [2.5, 2.0, 1.2, 1.2, 1.6, 1.2, 2.3], aligns=["L", "L", "C", "C", "C", "C", "L"])
        ex_rows = [[d["name"], e["evidence"], e["review_id"]] for d in ins["dimensions"][:4] for e in d["examples"][:1]]
        if ex_rows:
            pdf.table_rows(["Dimension", "Supporting evidence (Arabic)", "Review"], ex_rows, [2.4, 5.5, 1.1], aligns=["L", "R", "L"])
    pdf.note("The mapping to EGSQUAL is provisional: the dimension names come from the authors' abstract, and the definitions have "
             "not yet been verified against the full paper. EGSQUAL was developed to measure the quality of governmental websites; "
             "here it is adapted as a coding frame for Arabic app reviews.")

    # ---- 7 Improvement actions ----------------------------------------------------
    pdf.section(7, "Recommended Improvement Actions", "Derived from Agent 3's decisions on each review.")
    pdf.table_rows(["Agent 3 action", "Reviews", "Share"], [[a, c, f"{p}%"] for a, c, p in ins["actions"]], [5, 1, 1],
                   aligns=["L", "C", "C"])
    if ins["improvements"]:
        pdf.table_rows(["Action", "Application", "Dimension", "Tickets", "P1/P2", "Example ticket"],
                       [[g["action"], g["app"], g["dimension"], g["tickets"], g["urgent"], g["example_title"]] for g in ins["improvements"]],
                       [2.4, 1.8, 2.0, 1.3, 1.2, 3.2], aligns=["L", "L", "L", "C", "C", "L"])

    # ---- 8 SLA and escalation ----------------------------------------------------
    pdf.section(8, "Simulated SLA and Escalation Summary")
    pdf.note("RESEARCH-DEFINED SIMULATION. Escalation rules E1–E5 and the SLA values are proposed research rules, not official rules "
             "of the DGA or the AMER / Balaagh Digital service. Every escalation requires human approval, and nothing is submitted.",
             fill=(255, 247, 230), border=(201, 126, 0))
    pdf.table_rows(["Escalation status", "Reviews"], [[s, c] for s, c in ins["escalation"]], [5, 1], aligns=["L", "C"])
    if ins["escalation_candidates"]:
        pdf.table_rows(["Review", "Application", "Complaint ref.", "SLA deadline", "Days overdue"],
                       [[c["review_id"], c["app"], c["reference"], c["sla_deadline"], c["days_overdue"]] for c in ins["escalation_candidates"]],
                       [1.2, 2.2, 1.8, 1.6, 1.2])
    if ins["escalation_pending"]:
        pdf.table_rows(["Review", "Application", "Missing to verify the complaint"],
                       [[p["review_id"], p["app"], p["missing"]] for p in ins["escalation_pending"]], [1.2, 2.2, 4.6])
    pdf.table_rows(["Rule", "Escalation eligibility (proposed)"], [[r["id"], r["description"]] for r in tools.escalation_rules_kb()["rules"]], [1, 7])

    # ---- 9 Management recommendations ----------------------------------------------
    pdf.section(9, "Management Recommendations", "Generated by fixed rules from the counts above; each states its basis.")
    for k, r in enumerate(ins["recommendations"], start=1):
        pdf.para(f"{k}. {r['title']}", size=10, bold=True, color=GREEN)
        pdf.para(r["basis"], size=9, color=INK)
        pdf.ln(1.5)

    # ---- 10 Methodology and limitations ---------------------------------------------
    pdf.section(10, "Methodology and Limitations")
    pdf.para("Method. Each review is processed by three cooperating agents. The Investigation Agent extracts independent issues with "
             "verbatim evidence and maps each to a service-quality dimension. The Impact Agent judges whether the beneficiary could "
             "complete the service and applies severity rules R1–R4 and priority rules P-BASE, P-SCOPE and P-VERIFY. It can send an "
             "issue back to the Investigation Agent for revision (at most two rounds). The Resolution Agent decides the action, "
             "checks complaint history and SLA, and drafts a simulated ticket.", size=9)
    pdf.ln(1.5)
    pdf.para(f"Coverage. This report covers the {cov['reviews']:,} reviews analysed in the session on {cov['analysis_date']}, not the "
             "whole dataset unless all reviews were analysed.", size=9)
    pdf.ln(1.5)
    pdf.para("Limitations:", size=9, bold=True)
    pdf.bullets(["This report counts the framework's decisions; accuracy against human-verified labels is reported separately "
                 "in the Evaluation page.",
                 "Severity, priority and escalation rules are proposed by the researcher and are not official DGA rules.",
                 "Management recommendations are generated by fixed rules from the counts; they are not expert judgements."]
                + ref["limitations"], size=8.5)
    pdf.ln(1)
    c = ref["citation"]
    pdf.para(f"Reference: {c['authors']} ({c['year']}). {c['title']}. {c['journal']}, {c['volume']}, {c['article']}. "
             f"https://doi.org/{c['doi']}", size=8, color=MUTED)
    return bytes(pdf.output())
