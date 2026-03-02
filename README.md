# Policy_Guided_RAG

> ✅ This repository releases the **evaluation protocol** (prompt templates, and compliance/leakage metrics).  
> 🚫 It does **not** release organisational documents, queries, or sensitive spans due to access restrictions.

---

## What this repo contains
- **Routing policy**: deterministic mapping from sensitivity mix → generation mode.
- **Prompt templates**: quote-aware templates for Verbatim / Combined; synthesis template for non-sensitive only.
- **Evaluation code**: metrics for policy compliance, leakage diagnostics, and utility proxies.
- **Uncertainty utilities**: paired bootstrap CIs over queries (where applicable).

If you are using your own corpus, you only need to reproduce the *data interface* (CSV schema below) and run the evaluation.

---

## Modes
- **Synthesis**: answer from non-sensitive evidence only (no sensitive block provided).
- **Verbatim**: sensitive evidence must be reproduced **exactly** (no paraphrase).
- **Combined (controlled synthesis)**: synthesise non-sensitive content, but quote sensitive spans **verbatim**.

---

## Data interface (CSV schema)
### Required input dataset (per query)
- `query`: the question.
- `s_section` or `S_i`: sensitive text block (may be empty for non-applicable rows).
- `ns_section` or `N_i`: non-sensitive text block.

### Required model outputs (per query)
- `answer` or `A_i`: final answer text.
- `verbatim_quotes_used` or `Q_i` (Verbatim/Combined only): list-like field of declared quotes (optional diagnostics).
- Optional: `llm_resp_without_quotes`: answer with declared quotes removed (if you precompute it).

**Baseline outputs do not include `verbatim_quotes_used`.**

---

## Metrics (high level)
### Policy compliance (evaluated when `S_i` is non-empty)
- **VVR** (Verbatim Violation Rate): violations detected on the *final answer* (answer-side verification).
- **AR** (Abstention Rate): explicit refusal/“I can’t answer”.
- **FAR** (False Abstention Rate): abstains despite non-sensitive evidence being present (proxy).

### Leakage diagnostics
- **PSR**: paraphrase-style sensitive reuse (TF–IDF / token-overlap proxies on the unquoted remainder).
- **Levenshtein proximity**: sentence-level textual proximity.  
  *Note:* includes compliant verbatim quotes, so interpret jointly with **VVR/PSR**.

### Utility (reference-free proxies)
- **Completeness**: coverage proxy (0 for abstentions).
- **SemSim_ctx**: TF–IDF cosine similarity to provided context.

---

## Quickstart (evaluation)
1. **Prepare CSVs** in the schema above:
   - `DATASET_CSV` (query + `s_section` + `ns_section`)
   - `MODEL_OUT_CSV` (query + answer + optional quotes)
2. **Run evaluation** (notebook or script under `eval/` or `notebooks/`):
   - Update paths at the top of the file
   - Run all cells / execute the script
3. **Outputs**
   - `*_row_metrics.csv` (row-level metrics)
   - `*_summary.json` (aggregate metrics)
   - `all_system_summaries.csv` (multi-system panel)

---

## Recommended repo structure (expected by reviewers)
- `prompts/` — prompt templates for Verbatim / Combined / Synthesis  
- `policy/` — routing rules and label→mode mapping  
- `eval/` — metric implementations + CI utilities  
- `examples/` — **synthetic** toy CSVs demonstrating the schema (no real data)  
- `data/` — empty (with a `README.md` warning not to commit sensitive data)

---

## Notes on release constraints
This repository intentionally excludes any organisational data. The goal is **protocol + code** that can be executed
on another corpus with similar transformation constraints.

---

## Citation
If you use this code/protocol, please cite:

```bibtex
@misc{tavakoli2026policyguidedrag,
  title        = {Policy-Guided RAG: Segment-Level Transformation Governance in Regulated Settings},
  author       = {Naghash Asadi, Mina; Tavakoli, Leila and Bilgrami, Mustafa},
  year         = {2026},
  note         = {SIGIR Industry Track submission (under review)}
}
