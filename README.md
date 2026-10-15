# Multi-Agent AI Framework for Arabic Beneficiary Feedback Analysis

**Project title:** *A Multi-Agent AI Framework for Arabic Beneficiary Feedback Analysis to Enhance Digital Service
Quality and Beneficiary Experience in Saudi Government Digital Services.*

A working Python prototype in which **three cooperating agents** turn unstructured Arabic reviews (MSA and Saudi
dialect) of Saudi government mobile apps into:

- service-quality insights mapped to EGSQUAL dimensions,
- operational severity and priority decisions,
- simulated resolution and escalation decisions.

> ⚠️ **Simulation only.** No complaint is ever submitted to AMER, the DGA or any government system. The escalation,
> severity and priority rules are **proposed research rules**, not official DGA rules. A human officer must approve
> before any external escalation.

---

## 1. Quick start

### 1.1 Install (Python 3.11 or newer)

```bash
cd UniProject
python -m venv .venv
# Windows:  .venv\Scripts\activate
# macOS/Linux:
source .venv/bin/activate
pip install -r requirements.txt
```

### 1.2 Add your API key

```bash
cp .env.example .env        # Windows: copy .env.example .env
```

Open `.env` and fill in **one** provider:

```ini
LLM_PROVIDER=anthropic
ANTHROPIC_API_KEY=sk-ant-...        # your key
ANTHROPIC_MODEL=claude-opus-5-5
```

or `LLM_PROVIDER=openai` with `OPENAI_API_KEY` and `OPENAI_MODEL`.

**No key?** The app still runs. It switches to the **offline rule-based reasoner** and says so in the sidebar and
in the decision log. All tests use offline mode, so they need no key.

### 1.3 Add the dataset

Copy your CSV to `data/reviews.csv`, or upload it from the dashboard sidebar.

### 1.4 Run the dashboard

```bash
streamlit run app.py
```

A browser opens at <http://localhost:8501>.

### 1.5 Run the tests and the scenarios

```bash
pytest -q                                   # 62 functional, scenario, regression and UI tests (offline, no API key)
python run_scenarios.py --provider offline  # all scenarios, rule-based reasoner
python run_scenarios.py                     # all scenarios with the LLM from .env
python run_scenarios.py --only S1 S6        # selected scenarios
```

`run_scenarios.py` writes `outputs/scenario_report.md` (a readable report for each scenario),
`outputs/scenario_results.json` (the full shared state) and `outputs/scenario_checks.csv`.

---

## 2. How to test the scenarios

In the dashboard, open the **Test scenarios** page, choose a scenario (S1–S8) and click **Run scenario**. The
**Evaluation** page runs all of them at once and shows the behavioural checks.

| Key | Scenario | Source | What it demonstrates |
|---|---|---|---|
| S1 | Clear technical issue | real, row 705 (Citizen Account) | OTP never sent → login blocked → High/P2 → internal technical ticket |
| S2 | Multiple issues in one review | real, row 2464 (SPL) | 3 independent issues (crashes, login, Arabic UI shows English) with separate dimensions |
| S3 | Ambiguous complaint | real, row 1182 (`لا يعمل`) | Agent2 → Agent1 review request; Agent1 confirms ambiguity → INSUFFICIENT_EVIDENCE, priority capped → request for information |
| S3b | 1-star complaint with no operational issue | real, row 3067 | Strongly negative but **no severity assigned**: a 1-star review is not automatically critical |
| S4 | Positive feedback + suggestion | real, row 786 (GOSI) | Praise and a fingerprint-login suggestion → Low/P4 → service improvement request |
| S5 | High-impact service issue | real, row 77 (SCE) | Fees paid but registration not completed → FINANCIAL risk → Critical/P1 |
| S6 | Conflicting classification | real, row 50 (Watani) | 5★ praise containing a bug: Agent2 detects TYPE_CONFLICT, Agent1 revises Praise → Bug |
| S7 | Previous complaint, details missing | real, row 370 (NWC) | Complaint mentioned with no reference → **PENDING_VERIFICATION**; nothing is invented |
| S8 | **SYNTHETIC** complaint history + SLA | synthetic (labelled) | Record `SYN-2026-0417` is open, the 5-day SLA has passed and severity is Critical → escalation candidate awaiting **human approval** |

Each scenario shows: the original Arabic review, Agent 1/2/3 outputs, the tools used, the inter-agent revision history,
the decision log and the final operational decision. When `data/reviews.csv` is present, every real scenario is checked
to exist verbatim in the dataset (✅ on the Test scenarios page).

---

## 3. Architecture

```
            ┌──────────────────────── shared WorkflowState (schemas.py) ────────────────────────┐
            │ review · agent outputs · inter-agent messages · decision log · tool calls · trace │
            └───────────────────────────────────────────────────────────────────────────────────┘
 Arabic  ─► A1_ANALYZE ─► A2_ASSESS ──(evidence sufficient)──────────────► A3_RESOLVE ─► Human officer
 review          ▲            │                                                (simulated ticket)
                 │            └─(REVIEW_REQUEST, round < 2)─► A1_REVIEW_RESPONSE ─┐
                 └────────────────────(REVISE / CONFIRM)───────────────────────────┘
```

- **`workflow.py`** is a small, explicit state machine. LangGraph was not needed: transitions depend on the agents'
  decisions, the Agent1↔Agent2 loop is capped at `MAX_REVISIONS` (default 2), and every transition is recorded in
  `state.node_trace` and the decision log.
- **The LLM judges, the rules decide.** The LLM (Claude by default) handles language understanding: issue extraction
  and impact facts. Its output is a validated **Pydantic** schema (`client.messages.parse(output_format=...)`).
  The final severity, priority and escalation decisions come from documented, deterministic rule engines
  (`tools.py`), so they are auditable.
- **Tools are agent-specific.** Every tool is registered with an allow-list (`tools.TOOL_REGISTRY`). An agent calling
  another agent's tool raises `PermissionError`. Every call is logged with its reason, input and output.
- **Resilience.** Retries use exponential backoff (`LLM_MAX_RETRIES`). Non-retryable errors (authentication, bad
  request, refusal) are not retried. Any LLM failure falls back to the rule-based reasoner for that step and is
  logged as `LLM_FALLBACK`.

### Files

| File | Purpose |
|---|---|
| `app.py` | Streamlit entry point: page config, sidebar (reasoning mode, settings, dataset upload) and navigation |
| `ui/theme.py` | Design tokens (navy / white / teal), validated chart palettes, global CSS incl. RTL Arabic |
| `ui/components.py` | Reusable components: KPI cards, severity/priority badges, workflow diagram, evidence highlighting, issue and assessment cards, message timeline, escalation checklist, ticket |
| `ui/charts.py` | Altair charts (categories, EGSQUAL dimensions, priority, severity, severity-by-rating, checks, P/R/F1) |
| `ui/result_view.py` | The multi-agent result screen: live run, step-by-step replay, three agent panels, communication, audit trail, export |
| `ui/pages.py` | Pages: Dashboard, Analyse feedback, Test scenarios, Batch analysis, Evaluation, Methodology & rules |
| `.streamlit/config.toml` | Light navy/teal theme for presentations |
| `agents/agent1.py` | Feedback Investigation Agent |
| `agents/agent2.py` | Impact Assessment & Prioritization Agent |
| `agents/agent3.py` | Resolution & Escalation Agent |
| `agents/base.py` | Tool calling with allow-lists, logging, LLM-or-fallback reasoning |
| `workflow.py` | State machine orchestration and the revision loop |
| `tools.py` | All agent tools and rule engines |
| `schemas.py` | Pydantic schemas: LLM outputs, agent outputs, messages, shared state |
| `config.py` | Configuration from environment variables |
| `llm.py` | Anthropic / OpenAI clients with structured outputs and retries; `StubLLM` for tests |
| `heuristics.py`, `lexicon.py`, `arabic_text.py` | Offline Arabic reasoner, MSA/dialect lexicons, normalisation |
| `data_loader.py` | CSV loading: column detection, UTF-8, missing values, duplicates |
| `scenarios.py`, `run_scenarios.py` | Scenario definitions and the command-line runner and report |
| `evaluation.py` | Behavioural checks and optional P/R/F1 |
| `evaluation_suite.py` | Evaluation Dashboard metrics: label normalisation, annotation sheet, per-agent P/R/Macro F1, ticket & escalation F1, JSON validity, time and estimated cost |
| `baselines.py` | Comparison baselines: Single LLM and Sequential LLM pipeline (same knowledge, no tools / verification / loop) |
| `ui/eval_dashboard.py` | The Evaluation Dashboard page |
| `knowledge_base/*.json` | EGSQUAL dimensions, dialect glossary, severity/priority rules, escalation rules, synthetic complaint history |
| `tests/` | Unit, agent, workflow, scenario and LLM-path tests |

---

## 4. The three agents: behaviour beyond ML/NLP classification

A conventional classifier maps text to a label in one pass. Each agent here **plans, uses tools conditionally,
verifies its own output, decides, and talks to the other agents**.

### Agent 1: Feedback Investigation
- **Decomposes** the review into independent issues. One review can produce several issues, each with its own type,
  EGSQUAL dimension and **verbatim evidence**.
- **Uses tools conditionally:**
  - it runs the dialect glossary and adds it to the LLM context only when dialect is found;
  - it looks up every dimension in the EGSQUAL knowledge base and, if the dimension is invalid, searches the KB and
    corrects it;
  - it runs the ambiguity checker only when there are few issues or the text is short.
- **Verifies itself:** each evidence quote must really occur in the review. If it does not, the agent searches the
  review for a supporting span. If none exists, it **rejects the issue** instead of keeping a hallucination.
  Duplicate issues are merged.
- **Decides** `ACCEPTED`, `REVISED` or `NEEDS_CLARIFICATION`, and lists what information is missing.
- **Answers Agent 2's review requests** by re-examining the original text with tools. It then **revises** (changes
  the type, extends the evidence, identifies the service, merges issues) or **confirms** with a justification.

### Agent 2: Impact Assessment & Prioritization
- Judges **operational impact**: blocked, partially blocked, degraded or not affected; how widespread; workaround;
  financial, data, safety or legal risk. **Sentiment is measured separately** and never sets severity (rule S0).
  1-star reviews can be Low or get no severity at all (S3b).
- Applies the **documented rules** R1–R4 (severity) and P-BASE / P-SCOPE / P-VERIFY (priority) in
  `knowledge_base/severity_priority_rules.json`.
- **Verifies the evidence** behind its own decision. For example, "blocked" requires an explicit failure statement.
  It also cross-checks Agent 1. If something is wrong, it sends a **REVIEW_REQUEST** to Agent 1:
  `TYPE_CONFLICT`, `WEAK_EVIDENCE`, `AMBIGUITY_CHECK`, `MISSING_SERVICE_CONTEXT` or `DUPLICATE_ISSUES`.
- After at most 2 rounds, if evidence is still insufficient, it marks the issue `INSUFFICIENT_EVIDENCE`, labels the
  severity **provisional** and caps the priority at P3.

### Agent 3: Resolution & Escalation
- Decides whether the feedback is actionable and picks the action. The options are an internal technical ticket, a
  service improvement request, a request for information, follow-up on an existing complaint, an escalation
  candidate (AMER / Balaagh Digital, **simulated**), or no action. It also lists secondary actions for the other
  issues.
- Runs a **conditional tool chain**: complaint-reference extraction, then history lookup (only if a reference
  exists), then the SLA check (only if a record exists), then the escalation rules E1–E5.
- If the complaint reference, status, dates or SLA are missing, it returns **PENDING_VERIFICATION** and lists the
  missing fields. It **never invents** complaint numbers, dates or statuses.
- Generates a **simulated ticket** (title, description, application, affected service, severity, priority,
  recommended action, justification, evidence) and validates its required fields. **Human approval is always
  required.**

### Proposed rules (summary)

| Rule | Definition |
|---|---|
| R1 Critical | BLOCKED and (financial / personal-data / safety risk, or widespread) |
| R2 High | BLOCKED; or PARTIALLY_BLOCKED with a risk flag |
| R3 Medium | PARTIALLY_BLOCKED; or DEGRADED that is recurring, widespread or risky |
| R4 Low | DEGRADED (individual, no risk); or NOT_AFFECTED |
| Priority | Critical→P1, High→P2, Medium→P3, Low→P4; one level higher if widespread; capped at P3 if evidence is insufficient |
| Escalation | E1 prior complaint exists · E2 unresolved · E3 SLA passed · E4 severity ≥ High · E5 human approval (always) |

You can change all thresholds in `knowledge_base/*.json` without touching the code.

---

## 5. Evaluation

- **Behavioural checks** (`evaluation.py`, Evaluation page, `run_scenarios.py`) test:
  - that evidence is found in the review;
  - that multiple issues are detected;
  - that ambiguity is flagged;
  - that severity and priority follow the rules;
  - that escalation is consistent with the evidence (no invented references, human approval for candidates);
  - that ticket fields are complete;
  - that inter-agent review actually happened;
  - that the loop stayed within its bound.

  These are **rule-compliance checks, not accuracy scores**.
- **Evaluation Dashboard** (`evaluation_suite.py`, `baselines.py`, *Evaluation Dashboard* page) implements the evaluation plan.
  All metrics are computed **only** from human-verified reference labels that you upload. The dataset has no labels, so no
  accuracy figures are claimed.
  1. Export a stratified annotation sheet (blank label columns) and the guidelines. Label it, then upload it.
     The columns are `review_id`, plus optional `text`, and any of `feedback_type`, `dimensions`, `sentiment`, `severity`,
     `priority`, `action`, `ticket` and `escalation`.
  2. The dashboard reports these metrics:
     - **Agent 1:** feedback type and EGSQUAL service domain, with P, R and Macro F1.
     - **Agent 2:** sentiment, severity and priority, with P, R and Macro F1. The star rating is shown only as a supporting
       indicator for sentiment.
     - **Agent 3:** ticket and escalation decisions, with P, R and F1, plus the JSON ticket validity rate.
     - **Statistical evidence:** the framework is compared with a few-shot + chain-of-thought single LLM and with an
       ablation that has no review loop. The comparison uses an exact McNemar test per field, 95% bootstrap CIs of Macro F1,
       severe-case recall, evidence grounding (no invented quotes), and hard-case subgroups (dialect, multi-issue,
       short/vague, rating contradicts content). The summary also states where the framework is *not* better.
     - **Overall:** Multi-Agent vs Single LLM vs Sequential LLM pipeline on the same set. It shows Macro F1, Escalation F1,
       processing time per review, and estimated cost.
  3. Baselines need a working LLM. If they cannot run, they are recorded as failed and never simulated. Cost is an
     estimate: tokens counted from API responses × `ANTHROPIC_PRICE_*` / `OPENAI_PRICE_*` from `.env`.

---

## 6. Dataset handling

- Column detection: `Content` → text, `Abbreviation` → app id, `Source` → app name, `Rating` → rating (see `data/README.md`).
- Text is read as UTF-8 (with fallbacks).
- Empty texts are dropped.
- Duplicates within the same app are removed after Arabic normalisation (159 in the provided file).
  Each kept review records its `duplicate_count`.
- Review ids are `R` plus the original row number (e.g. `R00705`), so every result can be traced back to the CSV.

---

## 7. Knowledge base and honesty notes

- **EGSQUAL dimensions** (`knowledge_base/egsqual_dimensions.json`) are adapted from E-S-QUAL / E-RecS-QUAL
  (Parasuraman et al., 2005) and e-GovQual (Papadomichelaki & Mentzas, 2012). **Check them against the exact EGSQUAL
  source in your thesis** and edit the file if needed.
- The **offline reasoner** is a transparent lexicon and segmentation baseline. It is useful for demos without a key
  and for reproducible tests, but its language understanding is weaker than the LLM's. The **LLM mode is the intended
  configuration** for research results.
- `complaint_history_synthetic.json` contains **synthetic** records only (`SYN-` prefix).

## 8. What was tested and what was not

All of the following were run in this development environment, and only these results are claimed.

- ✅ **Clean install:** a fresh Python 3.11 virtual environment installed from `requirements.txt`.
- ✅ **62 automated tests pass** (`pytest -q`), including 5 UI tests, unit, agent, workflow, revision loop (including a forced
  **2-round** loop that stops at the cap), 9 scenarios, data loader, 10 realistic-review regression cases, and the
  LLM code path (stub LLM plus the real Anthropic SDK structured-output call against a local mock server).
- ✅ **Whole dataset, three-agent integration:** all 6,354 unique reviews were run end to end, and cross-agent rules
  were checked on every run. Agent 2 assesses exactly Agent 1's final issues; every review request is answered; the
  loop never exceeds 2 rounds; ticket severity equals Agent 2's overall severity; no complaint reference is invented;
  every ticket is marked simulated with human approval. **0 violations, 0 runtime errors** (about 2 minutes, offline).
- ✅ **User interface:** launched with `streamlit run app.py` and driven in a headless Chromium browser. Single
  review (typed and from the dataset), scenario S8, batch analysis and evaluation all ran with no exceptions.
- ⚠️ **Offline reasoner quality** (developer's judgement on blind random samples of 20 real reviews; no ground truth):
  13/20 acceptable before fixes, and 16/20 fully plus 2/20 partly acceptable on a new sample after fixes. Known
  weak spots:
  - rhetorical questions ("ليش ...؟") can be typed as Inquiries;
  - some topic confusions (a privacy complaint about location access was read as a report failure);
  - the dialect verb "ما يسوي" (doesn't work) is ambiguous with "ما يسوى" (worthless) and is not used.

  This mode is a transparent baseline, not the intended research configuration.
- ❌ **Not tested: live LLM calls.** No Anthropic or OpenAI API key was available, so no real Claude/OpenAI output
  has been produced or evaluated. The OpenAI path (`responses.parse`) was only checked against the SDK's method
  signature. With a key, run `python run_scenarios.py` and review `outputs/scenario_report.md` before relying on
  LLM-mode results. Each review makes about 2–4 API calls.

### How to tell which mode is running
- **Sidebar and page banner:** a green **"LLM mode · anthropic:claude-opus-5-5"** when a key is configured, or a
  yellow **"Offline rule-based mode · no LLM is being called"** when it is not.
- **Each agent card and panel** says what actually produced its output: `LLM`, `Rule-based (offline)`,
  `Fallback (LLM failed)` or `LLM + fallback`.
- **If any LLM call fails during a run,** a red **"LLM failure · N step(s) used the rule-based fallback"** banner
  lists each failed step and its error. A fallback is never presented as an LLM result. This was tested by pointing
  the app at an unreachable API endpoint; the API key never appears in the page.

The CLI prints `Reasoner: ...` on its first line. Every result stores it in `state.reasoner`, and an LLM failure
on any step is logged as `LLM_FALLBACK`.

## 9. Dashboard guide (for the presentation)

Launch with `streamlit run app.py` and open <http://localhost:8501>.

| Page | What to show |
|---|---|
| **Dashboard** | Dataset KPIs, then **Run sample analysis** to get KPIs and interactive charts. The KPIs are reviews analysed, issues extracted, High/Critical, P1/P2, simulated tickets and revision loops. The charts are feedback categories, EGSQUAL dimensions, priority, severity, and *severity vs star rating*. Everything is computed only from reviews actually analysed in the session. |
| **Analyse feedback** | Type a review (or load a researcher-written example), pick one from the dataset, or upload a CSV. The pipeline shows the active agent live. The result shows the decision summary, a workflow diagram with a revision counter, **step-by-step replay** of the real trace, and tabs for each agent, communication, audit trail and export. |
| **Test scenarios** | S1–S8 cards. **S6** shows a real Agent 2 → Agent 1 revision; **S8** (synthetic) shows an escalation candidate awaiting human approval. |
| **Batch analysis** | Filters, KPIs, charts, a results table, CSV/JSON export and drill-down into any result. |
| **Evaluation** | Behavioural checks on the scenarios (pass/fail chart). |
| **Evaluation Dashboard** | Annotation sheet export, then label upload. It shows per-agent P/R/Macro F1 (framed per agent) with confusion matrices, ticket/escalation F1 and JSON validity, and the Multi-Agent vs Single LLM vs Sequential pipeline comparison (Macro F1, Escalation F1, time, estimated cost). |
| **Methodology & rules** | EGSQUAL definitions, the R1–R4 / P-rules and the E1–E5 escalation rules. |

The step replay and live view are driven by the workflow's own callbacks and trace (`node_log_index`,
`agent1_history`, `agent2_history`), so each step shows only what existed at that point. Revision arrows and
counters appear only when a review request really occurred.
