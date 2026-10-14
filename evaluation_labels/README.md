# Reference labels

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

Upload the file on the **Evaluation Dashboard** page.
