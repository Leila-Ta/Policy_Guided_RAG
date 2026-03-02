# Policy_Guided_RAG

Segment-level **transformation governance** for Retrieval-Augmented Generation (RAG) in regulated settings.
We attach **sensitivity metadata** to retrieved segments, route queries into an allowed transformation mode
(**Verbatim**, **Combined / controlled synthesis**, **Synthesis**), and audit compliance on the **final output**
using answer-side checks (not model self-reporting).

> This repo releases the **evaluation protocol** (routing rules, prompt templates, and compliance/leakage metrics).
> It does **not** release organisational documents, queries, or sensitive spans due to access restrictions.

---

## What this repo contains
- **Routing policy**: deterministic mapping from segment sensitivity mix → generation mode.
- **Prompt templates**: quote-aware templates for Verbatim / Combined; synthesis template for non-sensitive only.
- **Evaluation code**: metrics for policy compliance, leakage diagnostics, and utility proxies.

> If you are using your own corpus, you only need to reproduce the *data interface* (CSV schema below)
and run the evaluation scripts/notebooks.

---

## Modes
- **Synthesis**: answer from non-sensitive evidence only (no sensitive block provided).
- **Verbatim**: sensitive evidence must be reproduced **exactly** (no paraphrase).
- **Combined (controlled synthesis)**: synthesise non-sensitive content, but quote sensitive spans **verbatim**.

---

## Data interface (CSV schema)
Your evaluation expects (adapt as needed):
- `query`: the question.
- `s_section` / `S_i`: sensitive text block (may be empty for non-applicable rows).
- `ns_section` / `N_i`: non-sensitive text block.
- Model outputs:
  - `answer` / `A_i`: final answer text.
  - `verbatim_quotes_used` / `Q_i` (Verbatim/Combined only): list-like field of declared quotes (optional diagnostics).
  - Optional: `llm_resp_without_quotes`: answer with declared quotes removed (if you precompute it).

Baseline outputs do **not** include `verbatim_quotes_used`.

---

## Metrics (high level)
**Policy compliance** (evaluated when `S_i` is non-empty):
- **VVR** (Verbatim Violation Rate): violations detected on the *final answer* (answer-side checking).
- **AR** (Abstention Rate): explicit refusal/“I can’t answer”.
- **FAR** (False Abstention Rate): abstains despite non-sensitive evidence being present (proxy).

**Leakage diagnostics**:
- **PSR**: paraphrase-style sensitive reuse (TF–IDF / token-overlap proxies on unquoted remainder).
- **Levenshtein proximity**: sentence-level textual proximity (includes compliant verbatim quoting; interpret with VVR/PSR).

**Utility**:
- **Completeness**: coverage proxy (0 for abstentions).
- **SemSim_ctx**: TF–IDF cosine similarity to provided context.

---

## Reproducing the evaluation
1. Prepare your CSVs in the schema above.
2. Run the evaluation notebook/script under `eval/` or `notebooks/` (update paths at the top).
3. Outputs:
   - `*_row_metrics.csv` (row-level)
   - `*_summary.json` (aggregate)
   - `all_system_summaries.csv` (multi-system panel)

---

## Notes on release constraints
This repository intentionally excludes any organisational data. The goal is **protocol + code** that can be run
on another corpus with similar transformation constraints.

---

## Citation
If you use this code/protocol, please cite:

```bibtex
@misc{tavakoli2026policyguidedrag,
  title        = {Policy-Guided RAG: Segment-Level Transformation Governance in Regulated Settings},
  author       = {Tavakoli, Leila and ...},
  year         = {2026},
  note         = {SIGIR Industry Track submission (under review)}
}
