# Reference labels

## `specialist_labels_50_balanced.xlsx` (use this one)

A stratified random sample of 50 reviews (seed 42) from the balanced research dataset `balanced_1-5_Arabic_dataset.csv`:
25 one-star and 25 five-star reviews from 24 applications. The label cells are **empty**: specialists in digital service
quality fill them in from the Arabic text only, ideally two annotators independently with disagreements resolved, and set
`label_source` to `human-verified` for each completed row. Upload the file on the Evaluation & Results page.

## Older files (not for the reported results)

`draft_labels_ai_50.csv` and `review_labels_50.xlsx` were sampled from the earlier 6,513-review file (ratings 1-5). Only 16 of
their 50 reviews occur in the balanced dataset, and their labels were drafted by an AI, so they are kept only as a
worked example.


`draft_labels_ai_50.csv` is a stratified sample of 50 reviews (by star rating, seed 42, from
`evaluation_suite.annotation_sheet`). Its labels were **drafted by an AI (Claude)** from the Arabic text only,
without looking at the system's outputs. It is **not** human-verified.

- Cells left empty were too ambiguous to label (for example "جيد ويحتاج تطوير"). They are excluded from the metrics.
- Escalation is `no` for every review, because none of them contains a verifiable previous complaint. Escalation F1
  is therefore undefined (n/a) on this set.
- The framework also uses Claude, so scores against these labels are biased in its favour.

Before you report any result:
1. A human annotator reviews and corrects every row.
2. Ideally a second annotator labels the same rows independently, and disagreements are resolved.
3. Replace `label_source` with, for example, `human-verified`.

## Reviewing the labels (recommended)

Open `review_labels_50.xlsx` in Excel or Numbers. It has three sheets:
- **Labels**: the AI-drafted labels in yellow cells with dropdowns, plus `ai_rationale` (why each label was chosen) and
  `reviewer_notes`. Correct any wrong label and set `label_source` to `human-verified` for each row you have checked.
- **Guide**: the review steps and the allowed values.
- **AI_original**: a locked copy of the draft. The dashboard uses it to report how many labels you changed.

Upload the `.xlsx` (or a CSV) on the **Evaluation Dashboard** page. It reports how many rows are still unreviewed and
how many labels you changed, which you can cite in the methodology.
