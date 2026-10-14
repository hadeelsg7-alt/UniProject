"""Altair chart builders. Colour rules:
- single-series counts use one hue (teal / navy) - axis labels carry identity;
- severity uses the reserved status palette, always with text labels;
- priority uses a validated single-hue ordinal ramp (P4 light -> P1 dark);
- P/R/F1 uses three validated categorical slots with a legend and value labels.
Every chart has hover tooltips; thin bars with 4px rounded data ends."""
from __future__ import annotations

import altair as alt
import pandas as pd

from ui import theme as T

FONT = "Inter, IBM Plex Sans Arabic, sans-serif"


def _style(chart: alt.Chart, height: int) -> alt.Chart:
    return (chart.properties(height=height)
            .configure_view(strokeWidth=0)
            .configure_axis(labelFont=FONT, titleFont=FONT, labelColor=T.INK_2, titleColor=T.INK_2, gridColor="#EDF3EF",
                            domain=False, tickColor="#EDF3EF", labelFontSize=12, titleFontSize=12)
            .configure_legend(labelFont=FONT, titleFont=FONT, labelColor=T.INK_2, titleColor=T.INK_2, orient="top")
            .configure_text(font=FONT))


def _counts(series: pd.Series, order: list[str] | None = None) -> pd.DataFrame:
    vc = series.dropna().value_counts()
    if order:
        vc = vc.reindex(order, fill_value=0)
    df = vc.rename_axis("label").reset_index(name="count")
    total = max(int(df["count"].sum()), 1)
    df["share"] = (df["count"] / total * 100).round(1)
    return df


def single_hue_bars(series: pd.Series, color: str, title: str, order: list[str] | None = None,
                    sort_desc: bool = True, height: int | None = None) -> alt.Chart:
    df = _counts(series, order)
    sort = "-x" if (sort_desc and not order) else order
    base = alt.Chart(df).encode(
        y=alt.Y("label:N", sort=sort, title=None, axis=alt.Axis(labelLimit=260, labelOverlap=False)),
        x=alt.X("count:Q", title=title),
        tooltip=[alt.Tooltip("label:N", title="Category"), alt.Tooltip("count:Q", title="Count"),
                 alt.Tooltip("share:Q", title="Share %")])
    bars = base.mark_bar(color=color, cornerRadiusEnd=4, height={"band": 0.62})
    labels = base.mark_text(align="left", dx=4, color=T.INK_2, fontSize=12).encode(text="count:Q")
    return _style(bars + labels, height or max(150, 42 * len(df)))


def severity_bars(series: pd.Series, include_none: bool = False) -> alt.Chart:
    order = T.SEVERITY_ORDER + (["No severity"] if include_none else [])
    df = _counts(series.fillna("No severity") if include_none else series, order)
    colors = [T.SEVERITY_COLORS[s] for s in T.SEVERITY_ORDER] + (["#C2D1C8"] if include_none else [])
    base = alt.Chart(df).encode(
        y=alt.Y("label:N", sort=order, title=None, axis=alt.Axis(labelOverlap=False)),
        x=alt.X("count:Q", title="Count"),
        tooltip=[alt.Tooltip("label:N", title="Severity"), alt.Tooltip("count:Q", title="Count"), alt.Tooltip("share:Q", title="Share %")])
    bars = base.mark_bar(cornerRadiusEnd=4, height={"band": 0.62}).encode(
        color=alt.Color("label:N", scale=alt.Scale(domain=order, range=colors), legend=None))
    labels = base.mark_text(align="left", dx=4, color=T.INK_2, fontSize=12).encode(text="count:Q")
    return _style(bars + labels, 42 * len(order) + 30)


def priority_bars(series: pd.Series) -> alt.Chart:
    order = T.PRIORITY_ORDER
    df = _counts(series, order)
    base = alt.Chart(df).encode(
        x=alt.X("label:N", sort=order, title=None, axis=alt.Axis(labelAngle=0)),
        y=alt.Y("count:Q", title="Issues"),
        tooltip=[alt.Tooltip("label:N", title="Priority"), alt.Tooltip("count:Q", title="Count"), alt.Tooltip("share:Q", title="Share %")])
    bars = base.mark_bar(cornerRadiusEnd=4, width={"band": 0.55}).encode(
        color=alt.Color("label:N", scale=alt.Scale(domain=order, range=[T.PRIORITY_COLORS[p] for p in order]), legend=None))
    labels = base.mark_text(dy=-8, color=T.INK_2, fontSize=12).encode(text="count:Q")
    return _style(bars + labels, 230)


def severity_by_rating(df: pd.DataFrame) -> alt.Chart:
    """Share of each severity (incl. 'No severity') within each star rating - shows rating != severity."""
    d = df.dropna(subset=["rating"]).copy()
    d["severity"] = d["overall_severity"].fillna("No severity")
    d["rating"] = d["rating"].astype(int).astype(str) + "★"
    g = d.groupby(["rating", "severity"]).size().reset_index(name="count")
    g["share"] = (g["count"] / g.groupby("rating")["count"].transform("sum") * 100).round(1)
    order = T.SEVERITY_ORDER + ["No severity"]
    colors = [T.SEVERITY_COLORS[s] for s in T.SEVERITY_ORDER] + ["#C2D1C8"]
    chart = alt.Chart(g).mark_bar(stroke="#FFFFFF", strokeWidth=2, height={"band": 0.62}).encode(
        y=alt.Y("rating:N", title="Star rating", sort=["1★", "2★", "3★", "4★", "5★"], axis=alt.Axis(labelOverlap=False)),
        x=alt.X("share:Q", stack="normalize", title="Share of reviews", axis=alt.Axis(format="%")),
        color=alt.Color("severity:N", scale=alt.Scale(domain=order, range=colors), title="Overall severity"),
        order=alt.Order("sev_rank:Q"),
        tooltip=[alt.Tooltip("rating:N", title="Rating"), alt.Tooltip("severity:N", title="Severity"),
                 alt.Tooltip("count:Q", title="Reviews"), alt.Tooltip("share:Q", title="Share %")],
    ).transform_calculate(sev_rank=f"indexof({order}, datum.severity)")
    return _style(chart, 260)


def checks_by_scenario(checks: pd.DataFrame) -> alt.Chart:
    g = checks.groupby(["scenario", "result"]).size().reset_index(name="count")
    domain, colors = ["PASS", "FAIL", "N/A"], ["#0CA30C", "#D03B3B", "#C2D1C8"]
    chart = alt.Chart(g).mark_bar(stroke="#FFFFFF", strokeWidth=2, height={"band": 0.6}).encode(
        y=alt.Y("scenario:N", title=None, axis=alt.Axis(labelOverlap=False)),
        x=alt.X("count:Q", title="Behavioural checks"),
        color=alt.Color("result:N", scale=alt.Scale(domain=domain, range=colors), title="Result"),
        order=alt.Order("rank:Q"),
        tooltip=[alt.Tooltip("scenario:N", title="Scenario"), alt.Tooltip("result:N", title="Result"), alt.Tooltip("count:Q", title="Checks")],
    ).transform_calculate(rank=f"indexof({domain}, datum.result)")
    return _style(chart, 40 * checks["scenario"].nunique() + 50)


def prf_bars(table: pd.DataFrame) -> alt.Chart:
    """Grouped Precision / Recall / F1 per label (rows with label not in the summary rows)."""
    d = table[~table["label"].isin(["MACRO AVG", "ACCURACY", "MICRO AVG"])]
    long = d.melt(id_vars=["label", "support"], value_vars=["precision", "recall", "f1"], var_name="metric", value_name="score")
    domain = ["precision", "recall", "f1"]
    base = alt.Chart(long).encode(
        y=alt.Y("label:N", title=None, axis=alt.Axis(labelLimit=260, labelOverlap=False)),
        yOffset=alt.YOffset("metric:N", sort=domain),
        x=alt.X("score:Q", title="Score", scale=alt.Scale(domain=[0, 1.08]), axis=alt.Axis(format=".1f", values=[0, 0.2, 0.4, 0.6, 0.8, 1.0])),
        tooltip=[alt.Tooltip("label:N", title="Class"), alt.Tooltip("metric:N", title="Metric"),
                 alt.Tooltip("score:Q", title="Score", format=".3f"), alt.Tooltip("support:Q", title="Support")])
    bars = base.mark_bar(cornerRadiusEnd=4, height={"band": 0.9}).encode(
        color=alt.Color("metric:N", scale=alt.Scale(domain=domain, range=[T.METRIC_COLORS[m] for m in domain]), title="Metric"))
    labels = base.mark_text(align="left", dx=3, fontSize=11, color=T.INK_2).encode(text=alt.Text("score:Q", format=".2f"))
    return _style(bars + labels, max(180, 72 * d["label"].nunique() + 40))


def confusion_heatmap(conf: pd.DataFrame, color: str, order: list[str] | None = None) -> alt.Chart:
    """Reference (rows) x predicted (columns) counts; single-hue sequential scale, counts printed in each cell."""
    labels = order or sorted(set(conf["reference"]) | set(conf["predicted"]))
    labels = [x for x in labels if x in set(conf["reference"]) | set(conf["predicted"])]
    labels += sorted((set(conf["predicted"]) | set(conf["reference"])) - set(labels))
    full = pd.MultiIndex.from_product([labels, labels], names=["reference", "predicted"])
    d = conf.set_index(["reference", "predicted"]).reindex(full, fill_value=0).reset_index()
    d["diag"] = d["reference"] == d["predicted"]
    base = alt.Chart(d).encode(
        x=alt.X("predicted:N", sort=labels, title="Predicted", axis=alt.Axis(labelAngle=-30, labelOverlap=False, labelLimit=160)),
        y=alt.Y("reference:N", sort=labels, title="Reference label", axis=alt.Axis(labelOverlap=False, labelLimit=160)),
        tooltip=[alt.Tooltip("reference:N", title="Reference"), alt.Tooltip("predicted:N", title="Predicted"),
                 alt.Tooltip("count:Q", title="Reviews")])
    rect = base.mark_rect(stroke="#FFFFFF", strokeWidth=2, cornerRadius=3).encode(
        color=alt.Color("count:Q", scale=alt.Scale(range=["#F4F7F5", color]), legend=None))
    text = base.mark_text(fontSize=12, fontWeight=600).encode(
        text=alt.Text("count:Q"),
        color=alt.condition(alt.datum.count > d["count"].max() / 2, alt.value("#FFFFFF"), alt.value(T.INK)))
    return _style(rect + text, max(170, 38 * len(labels) + 60))


def system_metric_bars(df: pd.DataFrame, value: str, title: str, colors: dict[str, str], fmt: str = ".3f",
                       domain_max: float | None = None) -> alt.Chart:
    """One bar per system (rows: system_key, system, <value>); validated categorical system colours + value labels."""
    d = df[["system_key", "system", value]].dropna(subset=[value])
    keys = d["system_key"].tolist()
    base = alt.Chart(d).encode(
        y=alt.Y("system:N", title=None, sort=d["system"].tolist(), axis=alt.Axis(labelLimit=220, labelOverlap=False)),
        x=alt.X(f"{value}:Q", title=title, scale=alt.Scale(domain=[0, domain_max or float(d[value].max() or 1) * 1.3])),
        tooltip=[alt.Tooltip("system:N", title="System"), alt.Tooltip(f"{value}:Q", title=title, format=fmt)])
    bars = base.mark_bar(cornerRadiusEnd=4, height={"band": 0.6}).encode(
        color=alt.Color("system_key:N", scale=alt.Scale(domain=keys, range=[colors[k] for k in keys]), legend=None))
    labels = base.mark_text(align="left", dx=4, color=T.INK_2, fontSize=12).encode(text=alt.Text(f"{value}:Q", format=fmt))
    return _style(bars + labels, 46 * max(len(d), 1) + 40)


def system_grouped_f1(long: pd.DataFrame, colors: dict[str, str]) -> alt.Chart:
    """Macro F1 per label field, one bar per system (long: field, system_key, system, f1)."""
    keys = list(dict.fromkeys(long["system_key"]))
    names = list(dict.fromkeys(long["system"]))
    base = alt.Chart(long).encode(
        y=alt.Y("field:N", title=None, sort=list(dict.fromkeys(long["field"])), axis=alt.Axis(labelLimit=220, labelOverlap=False)),
        yOffset=alt.YOffset("system:N", sort=names),
        x=alt.X("f1:Q", title="Macro F1", scale=alt.Scale(domain=[0, 1.08]), axis=alt.Axis(format=".1f", values=[0, 0.2, 0.4, 0.6, 0.8, 1.0])),
        tooltip=[alt.Tooltip("field:N", title="Field"), alt.Tooltip("system:N", title="System"),
                 alt.Tooltip("f1:Q", title="Macro F1", format=".3f")])
    bars = base.mark_bar(cornerRadiusEnd=4, height={"band": 0.9}).encode(
        color=alt.Color("system:N", scale=alt.Scale(domain=names, range=[colors[k] for k in keys]), title="System",
                        legend=alt.Legend(orient="top", direction="vertical")))
    labels = base.mark_text(align="left", dx=3, fontSize=11, color=T.INK_2).encode(text=alt.Text("f1:Q", format=".2f"))
    return _style(bars + labels, max(180, 22 * len(long) + 60))
