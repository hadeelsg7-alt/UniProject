"""Run the full system comparison from the command line and write a report.

    python run_comparison.py                                   # default labels file, all systems
    python run_comparison.py --labels my_labels.xlsx --systems multi_agent single_llm_cot
    python run_comparison.py --limit 10                        # quick, cheap trial on 10 reviews

Uses the same code as the Evaluation Dashboard. Needs the reviews dataset (data/reviews.csv) and, for the
LLM baselines, a working ANTHROPIC_API_KEY in .env. Writes to outputs/comparison/:
  report.md, overall.csv, significance.csv, subgroups.csv, ci.csv, predictions.csv
Nothing is simulated: systems that cannot run are reported as failed.
"""
from __future__ import annotations

import argparse
import time
from pathlib import Path

import pandas as pd

import baselines
import config
import evaluation_suite as E
from data_loader import load_reviews
from llm import LLMClient
from workflow import MultiAgentWorkflow

OUT = config.OUTPUT_DIR / "comparison"


def _fmt(v, f=".3f"):
    return "n/a" if v is None or (isinstance(v, float) and pd.isna(v)) else format(v, f)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--labels", default="evaluation_labels/review_labels_50.xlsx")
    ap.add_argument("--dataset", default=str(config.DEFAULT_DATASET_PATH))
    ap.add_argument("--systems", nargs="+", default=list(E.SYSTEMS), choices=list(E.SYSTEMS))
    ap.add_argument("--limit", type=int, default=0, help="only the first N labelled reviews (0 = all)")
    args = ap.parse_args()

    llm = LLMClient()
    print(f"Reasoning engine: {llm.name}" + (f"  ({llm.note})" if llm.note else ""))
    if not llm.available:
        print("WARNING: no LLM available - the LLM baselines will be recorded as FAILED (not simulated).")

    path = Path(args.labels)
    ref, original = E.read_labels_file(path.read_bytes(), path.name)
    labels, cov = E.normalise_reference(ref)
    rs = E.review_status(ref)
    df, _ = load_reviews(args.dataset)
    reviews, missing = E.reviews_for_labels(ref[ref["review_id"].astype(str).str.strip().isin(labels)], df)
    if args.limit:
        reviews = reviews[: args.limit]
        labels = {r.review_id: labels[r.review_id] for r in reviews}
    print(f"{len(reviews)} labelled review(s); systems: {', '.join(args.systems)}"
          + (f"; {len(missing)} id(s) without text skipped" if missing else ""))

    results, prices = {}, {}
    for s in args.systems:
        t0 = time.time()
        prog = lambda k, n, s=s: print(f"\r  {E.SYSTEMS[s]}: {k}/{n}", end="", flush=True)  # noqa: E731
        if s == "multi_agent":
            results[s] = E.run_multi_agent(MultiAgentWorkflow(llm), reviews, progress=prog)
        elif s == "multi_agent_no_loop":
            results[s] = E.run_multi_agent(MultiAgentWorkflow(llm, max_revisions=0), reviews, progress=prog, system=s)
        else:
            results[s] = baselines.run_baseline(s, llm, reviews, progress=prog)
        prices[s] = llm.price_per_mtok()
        failed = sum(p.failed for p in results[s])
        print(f"  done in {time.time() - t0:.0f}s" + (f"  ({failed} failed)" if failed else ""))

    rmap = {r.review_id: r for r in reviews}
    overall = E.overall_table(labels, results, prices, rmap)
    sig = E.significance_table(labels, results)
    sub = E.subgroup_table(labels, results, rmap)
    ci = E.ci_table(labels, results)
    OUT.mkdir(parents=True, exist_ok=True)
    overall.to_csv(OUT / "overall.csv", index=False, encoding="utf-8-sig")
    sig.to_csv(OUT / "significance.csv", index=False, encoding="utf-8-sig")
    sub.to_csv(OUT / "subgroups.csv", index=False, encoding="utf-8-sig")
    ci.to_csv(OUT / "ci.csv", index=False, encoding="utf-8-sig")
    E.predictions_frame(results).to_csv(OUT / "predictions.csv", index=False, encoding="utf-8-sig")

    lines = [f"# System comparison report", "",
             f"- Reasoning engine: `{llm.name}`",
             f"- Labelled reviews: {len(reviews)} (labels file `{path.name}`)",
             f"- Label review status: " + (f"{rs['verified']} of {rs['rows']} rows marked human-verified"
                                           if rs["tracked"] else "not tracked"),
             ""]
    if rs["tracked"] and rs["unreviewed"]:
        lines += ["> **Warning:** some labels are AI-drafted and not human-verified. The framework also uses Claude, "
                  "so these scores are biased and must not be reported as human-verified results.", ""]
    if original is not None:
        ch = E.label_changes(ref, original)
        lines += [f"- Labels changed by the human reviewer: {int(ch['changed_by_reviewer'].sum())} of {int(ch['drafted'].sum())}", ""]
    lines += ["## Overall", "", "| System | Mean Macro F1 | Severe recall | Evidence grounding | Escalation F1 | JSON valid | s/review | est. $/review | failed |",
              "|---|---|---|---|---|---|---|---|---|"]
    for _, r in overall.iterrows():
        lines.append(f"| {r.system} | {_fmt(r.macro_f1_mean)} | {_fmt(r.severe_recall)} | {_fmt(r.evidence_grounding_rate)} | "
                     f"{_fmt(r.escalation_f1)} | {_fmt(r.json_validity_rate)} | {_fmt(r.time_per_review_s, '.2f')} | "
                     f"{_fmt(r.est_cost_per_review_usd, '.4f')} | {int(r.failed_reviews)} |")
    lines += ["", "## Macro F1 per field (95% bootstrap CI)", "", "| System | Field | Macro F1 | CI |", "|---|---|---|---|"]
    for _, r in ci.iterrows():
        lines.append(f"| {r.system} | {r.field} | {_fmt(r.macro_f1)} | [{_fmt(r.ci_low)}, {_fmt(r.ci_high)}] |")
    if not sig.empty:
        lines += ["", "## Significance (exact McNemar, alpha 0.05): Multi-Agent framework vs others", "",
                  "| Compared with | Field | n | Framework acc | Other acc | p | Result |", "|---|---|---|---|---|---|---|"]
        for _, r in sig.iterrows():
            lines.append(f"| {r.compared_with} | {r.field} | {r.n} | {_fmt(r.framework_accuracy)} | {_fmt(r.other_accuracy)} | "
                         f"{r.p_value:.4f} | {r['result (alpha 0.05)']} |")
    if not sub.empty:
        lines += ["", "## Hard cases (mean accuracy over feedback type, severity, action)", "", "| Subgroup | n | System | Mean accuracy |",
                  "|---|---|---|---|"]
        for _, r in sub.iterrows():
            lines.append(f"| {r.subgroup} | {r.n} | {r.system} | {_fmt(r.mean_accuracy)} |")
    lines += ["", "Cost is an estimate: tokens counted from the API responses × configured prices (.env)."]
    (OUT / "report.md").write_text("\n".join(lines), encoding="utf-8")
    print(f"\nReport written to {OUT / 'report.md'}")
    print(overall[["system", "macro_f1_mean", "severe_recall", "evidence_grounding_rate", "time_per_review_s",
                   "est_cost_per_review_usd", "failed_reviews"]].to_string(index=False))


if __name__ == "__main__":
    main()
