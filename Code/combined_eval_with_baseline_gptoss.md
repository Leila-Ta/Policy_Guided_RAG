```python
import os, re, json, ast, difflib
from typing import Optional, List, Dict, Any, Tuple
import numpy as np
import pandas as pd
import ast
import json
import math
import re
import difflib
from openai import AzureOpenAI
import time
```


```python
# =========================
# CONFIG: paths & systems
# =========================

# Ground-truth dataset (must have: query, s_section, ns_section)
DATASET_CSV = r".../combined_manual.csv"

# Model outputs for *Combined* mode (must have: query, answer, verbatim_quotes_used)
# Optional but recommended: llm_resp_without_quotes (answer with quoted spans removed)
COMBINED_OUT_CSV = r".../combined_llm_resp_with_quotes_llama4.csv"

# Output folder
OUTPUT_DIR = r".../eval_results_llama4_baseline_vs_combined"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Baseline output file (will be created if missing and GENERATE_BASELINE=True)
BASELINE_OUT_CSV = os.path.join(OUTPUT_DIR, "baseline_llama4_outputs.csv")

# Baseline generation switch
GENERATE_BASELINE = True     # set False if you already have BASELINE_OUT_CSV
REGENERATE_BASELINE = False  # set True to overwrite baseline outputs

# Baseline model settings
BASELINE_MODEL = "Llama-4-Maverick-17B-128E-Instruct-FP8"   # deployment/model name
BASELINE_TEMPERATURE = 0.0
BASELINE_MAX_TOKENS = 600
BASELINE_REQUEST_SLEEP_S = 0.2    # small pause to avoid rate limits

# Systems to evaluate
SYSTEMS = {
    "baseline_llama4": {
        "out_csv": BASELINE_OUT_CSV,
        "answer_col": "answer",
        "quotes_col": None,                 # baseline has no quotes
        "raw_json_col": None,
        "unquoted_answer_col": None,
        "join_on": "query",
    },
    "combined_llama4": {
        "out_csv": COMBINED_OUT_CSV,
        "answer_col": "answer",
        "quotes_col": "verbatim_quotes_used",
        "raw_json_col": None,
        "unquoted_answer_col": "llm_resp_without_quotes",  # optional
        "join_on": "query",
    },
}

# Optional (used for TF-IDF based diagnostics)
try:
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.metrics.pairwise import cosine_similarity
    _HAS_SKLEARN = True
except Exception:
    _HAS_SKLEARN = False
```


```python
# =========================
# Baseline generation using the SAME retrieved context (s_section + ns_section)
# You do NOT need a separate "baseline dataset" — we reuse DATASET_CSV and write BASELINE_OUT_CSV.
# =========================
# initiate azure models
def read_csv_robust(path: str) -> pd.DataFrame:
    """Robust CSV loader to avoid UnicodeDecodeError in CDSW."""
    for enc in ["utf-8", "utf-8-sig", "cp1252", "latin1"]:
        try:
            return pd.read_csv(path, encoding=enc, encoding_errors="replace")
        except Exception:
            pass
    return pd.read_csv(path, engine="python", encoding_errors="replace")

def ensure_baseline_context(df: pd.DataFrame) -> pd.DataFrame:
    """Ensure baseline_context exists as concatenation of s_section + ns_section."""
    if "baseline_context" not in df.columns:
        if "s_section" not in df.columns or "ns_section" not in df.columns:
            raise ValueError("DATASET_CSV must contain s_section and ns_section to build baseline_context.")
        df = df.copy()
        df["baseline_context"] = (df["s_section"].fillna("").astype(str) + "\n\n" + df["ns_section"].fillna("").astype(str)).str.strip()
    return df

def generate_baseline_outputs(df: pd.DataFrame, out_csv: str) -> pd.DataFrame:
    """
    Generate baseline answers for each query using ONLY baseline_context (s+ns concatenated).
    Output schema: query, answer, model
    """
    client = AzureOpenAI(
        api_key=...,
        azure_endpoint=...,
        api_version=...,)

    rows = []
    for i, r in df.iterrows():
        q = str(r.get("query", "")).strip()
        ctx = str(r.get("baseline_context", "")).strip()
        if not q:
            continue

        user_prompt = (
            "Answer the question using ONLY the provided context.\n"
            "If the answer is not supported by the context, reply exactly: I don't know.\n\n"
            f"Question: {q}\n\n"
            f"Context:\n{ctx}"
        )

        # retry loop
        last_err = None
        for attempt in range(5):
            try:
                resp = client.chat.completions.create(
                    model=BASELINE_MODEL,
                    temperature=BASELINE_TEMPERATURE,
                    max_tokens=BASELINE_MAX_TOKENS,
                    messages=[
                        {"role": "system", "content": "You are a helpful assistant."},
                        {"role": "user", "content": user_prompt},
                    ],
                )
                ans = resp.choices[0].message.content if resp.choices else ""
                rows.append({"query": q, "answer": ans, "model": BASELINE_MODEL})
                break
            except Exception as e:
                last_err = e
                time.sleep(1.5 * (attempt + 1))
        else:
            # failed after retries
            rows.append({"query": q, "answer": "", "model": BASELINE_MODEL, "error": str(last_err)})

        time.sleep(BASELINE_REQUEST_SLEEP_S)

    out_df = pd.DataFrame(rows)
    out_df.to_csv(out_csv, index=False, encoding="utf-8")
    return out_df

# --- Create baseline outputs if needed
if GENERATE_BASELINE and (REGENERATE_BASELINE or (not os.path.exists(BASELINE_OUT_CSV))):
    print("Generating baseline outputs:", BASELINE_OUT_CSV)
    ds = read_csv_robust(DATASET_CSV)
    ds = ensure_baseline_context(ds)
    # Only keep the query+context we need
    if "query" not in ds.columns:
        raise ValueError(f"DATASET_CSV must contain a 'query' column. Found: {list(ds.columns)}")
    ds_small = ds[["query","baseline_context"]].copy()
    _ = generate_baseline_outputs(ds_small, BASELINE_OUT_CSV)
    print("Done.")
else:
    print("Skipping baseline generation (using existing file):", BASELINE_OUT_CSV)
```


```python
# Utilities
STOPWORDS = set("""
a an the and or but if then else for to of in on at by with as is are was were be been being
i you he she it we they me him her them my your his its our their this that these those
from into over under between about above below up down out off not no yes do does did doing
can could should would may might will just than too very must done there here has
""".split())
_SENT_SPLIT_RE = re.compile(r"(?<=[\.\?\!])\s+|\n+")
_WS_RE = re.compile(r"\s+")
_WORD_RE = re.compile(r"[A-Za-z0-9]+")
```


```python
def unescape_newlines(x: Any) -> str:
    if x is None or (isinstance(x, float) and np.isnan(x)):
        return ""
    s = str(x)
    # handle literal "\n"
    return s.replace("\\n", "\n").replace("\\t", "\t")
```


```python
def normalize_ws(s: str) -> str:
    return _WS_RE.sub(" ", (s or "").strip())
```


```python
def safe_literal_list(x: Any) -> List[str]:
    """Accept list, JSON list, or python-literal list stored as string."""
    if x is None or (isinstance(x, float) and np.isnan(x)):
        return []
    if isinstance(x, list):
        return [str(i) for i in x]
    if isinstance(x, str):
        x = x.strip()
        if not x:
            return []
        # Try JSON list
        j = safe_json_loads(x)
        if isinstance(j, list):
            return [str(i) for i in j]
        # Try python literal
        try:
            v = ast.literal_eval(x)
            if isinstance(v, list):
                return [str(i) for i in v]
        except Exception:
            pass
    return []
```


```python
def safe_json_loads(s: str) -> Optional[Any]:
    try:
        return json.loads(s)
    except Exception:
        return None
```


```python
def tokenise(s: str) -> List[str]:
    s = normalize_ws(s.lower())
    return _WORD_RE.findall(s)
```


```python
def content_tokens(s: str) -> List[str]:
    toks = [t for t in tokenise(s) if len(t) >= 3 and t not in STOPWORDS]
    return toks
```


```python
def split_spans(text: str) -> List[str]:
    """Heuristic sentence/line segmentation used for diagnostics."""
    t = (text or "").strip()
    if not t:
        return []
    parts = [p.strip() for p in _SENT_SPLIT_RE.split(t) if p and p.strip()]
    spans = []
    for p in parts:
        if len(p) < 5:
            continue
        # avoid tiny fragments like 'or'
        if len(content_tokens(p)) < 3 and len(p) < 30:
            continue
        spans.append(p)
    return spans
```


```python
def token_overlap_recall(answer: str, source: str) -> float:
    """Recall of content tokens from source that appear in answer."""
    src = set(content_tokens(source))
    if not src:
        return np.nan
    ans = set(content_tokens(answer))
    return len(src & ans) / len(src)
```


```python
def novel_token_rate(answer: str, source: str) -> float:
    """Fraction of answer content tokens not present in source content tokens."""
    ans = content_tokens(answer)
    if not ans:
        return np.nan
    src = set(content_tokens(source))
    novel = [t for t in ans if t not in src]
    return len(novel) / max(1, len(ans))
```


```python
def longest_common_substring_len(a: str, b: str) -> int:
    if not a or not b:
        return 0
    return difflib.SequenceMatcher(None, a, b).find_longest_match(0, len(a), 0, len(b)).size
```


```python
def matching_blocks_stats(a: str, b: str, min_block: int = 30) -> Dict[str, int]:
    """Return total and max contiguous matching block sizes >= min_block."""
    a = normalize_ws(a or "")
    b = normalize_ws(b or "")
    if not a or not b:
        return {"total": 0, "max_block": 0, "num_blocks": 0}
    sm = difflib.SequenceMatcher(None, a, b, autojunk=False)
    blocks = [m.size for m in sm.get_matching_blocks() if m.size >= min_block]
    return {
        "total": int(sum(blocks)),
        "max_block": int(max(blocks) if blocks else 0),
        "num_blocks": int(len(blocks)),
    }
```


```python
def seq_ratio(a: str, b: str) -> float:
    """difflib ratio in [0,1]."""
    a = normalize_ws(a)
    b = normalize_ws(b)
    if not a or not b:
        return np.nan
    return difflib.SequenceMatcher(None, a, b, autojunk=False).ratio()
```


```python
def is_abstain(ans: str) -> bool:
    a = normalize_ws(ans).lower()
    if not a:
        return True
    patterns = [
        "i don't know",
        "i do not know",
        "cannot answer",
        "can't answer",
        "i can't help",
        "unable to answer",
        "insufficient information",
        "not enough information",
        "i cannot provide",
        "i'm not able to",
        "i am not able to",
        "no information provided",
    ]
    return any(p in a for p in patterns)
```


```python
def quote_sentence_boundary_error(q: str) -> bool:
    """True if quote contains >1 sentence by naive splitter."""
    sents = split_sentences(q)
    return len(sents) > 1
```


```python
def count_syllables(word: str) -> int:
    w = re.sub(r"[^a-z]", "", word.lower())
    if not w:
        return 0
    count = 0
    prev_v = False
    for ch in w:
        is_v = ch in _VOWELS
        if is_v and not prev_v:
            count += 1
        prev_v = is_v
    # silent e
    if w.endswith("e") and count > 1:
        count -= 1
    return max(1, count)
```


```python
def readability(ans: str) -> Tuple[float, float]:
    """(Flesch Reading Ease, Flesch-Kincaid Grade)"""
    text = normalize_ws(ans)
    if not text:
        return (np.nan, np.nan)
    words = tokenise(text)
    if not words:
        return (np.nan, np.nan)
    sentences = max(1, len(split_sentences(text)))
    n_words = len(words)
    n_syll = sum(count_syllables(w) for w in words)
    # Flesch Reading Ease
    fre = 206.835 - 1.015 * (n_words / sentences) - 84.6 * (n_syll / n_words)
    # Flesch-Kincaid Grade
    fk = 0.39 * (n_words / sentences) + 11.8 * (n_syll / n_words) - 15.59
    return (float(fre), float(fk))
```


```python
def split_sentences(text: str) -> List[str]:
    """Naive sentence splitter (good enough for auditing granularity)."""
    t = normalize_ws(text)
    if not t:
        return []
    # split on punctuation + whitespace
    sents = re.split(r"(?<=[.!?])\s+", t)
    sents = [s.strip() for s in sents if s.strip()]
    return sents
```


```python
def clamp01(v: float) -> float:
    if v is None:
        return np.nan
    try:
        if np.isnan(v):
            return np.nan
    except Exception:
        pass
    return float(max(0.0, min(1.0, v)))
```


```python
def levenshtein_similarity(a: str, b: str) -> float:
    """
    Normalized Levenshtein similarity in [0,1], computed with a DP fallback.
    We compute it between *sentences*, not full paragraphs.
    """
    a = normalize_ws(a)
    b = normalize_ws(b)
    if not a or not b:
        return np.nan

    # Simple DP (fine for sentence-sized strings)
    la, lb = len(a), len(b)
    if la == 0 and lb == 0:
        return 1.0
    if la == 0 or lb == 0:
        return 0.0

    # O(min(la,lb)) space
    if la < lb:
        a, b = b, a
        la, lb = lb, la

    prev = list(range(lb + 1))
    for i in range(1, la + 1):
        cur = [i] + [0] * lb
        ca = a[i - 1]
        for j in range(1, lb + 1):
            cost = 0 if ca == b[j - 1] else 1
            cur[j] = min(
                prev[j] + 1,        # deletion
                cur[j - 1] + 1,     # insertion
                prev[j - 1] + cost  # substitution
            )
        prev = cur

    dist = prev[lb]
    denom = max(la, lb)
    return float(1.0 - (dist / denom))
```


```python
def tfidf_cosine(a: str, b: str) -> float:
    if not _HAS_SKLEARN:
        return np.nan
    a = normalize_ws(a)
    b = normalize_ws(b)
    if not a or not b:
        return np.nan
    try:
        vec = TfidfVectorizer(stop_words="english")
        X = vec.fit_transform([a, b])
        return float(cosine_similarity(X[0], X[1])[0, 0])
    except ValueError:
        return np.nan
```


```python
def best_sentence_match(
    needles: List[str],
    hay_sents: List[str],
    sim_fn
) -> Tuple[float, Tuple[str, str]]:
    """
    For each needle sentence, find best similarity with any answer sentence.
    Return global max similarity and the (needle, best_hay) pair.
    """
    best = -1.0
    best_pair = ("", "")
    for n in needles:
        for h in hay_sents:
            s = sim_fn(n, h)
            if s is None or (isinstance(s, float) and np.isnan(s)):
                continue
            if s > best:
                best = float(s)
                best_pair = (n, h)
    if best < 0:
        return (np.nan, ("", ""))
    return (best, best_pair)
```


```python
_VOWELS = set("aeiouy")
```


```python
# --- Two-pass diagnostic helper: remove quoted spans from response ---
# Supports both: (1) explicit markers: #quote< ... >#quote, and (2) exact quote strings.

_QUOTE_SPAN_RE = re.compile(r"#quote<(?P<q>.*?)>#quote", flags=re.DOTALL | re.IGNORECASE)

def strip_quoted_spans(answer: str, quotes: List[str]) -> Tuple[str, int, int]:
    """Remove marked quote spans and exact occurrences of declared quotes.
    Returns: (remainder_text, n_marked_spans_removed, n_exact_quote_occurrences_removed).
    """
    a = answer or ""
    # 1) Remove explicitly marked spans
    marked = _QUOTE_SPAN_RE.findall(a)
    n_marked = len(marked)
    a2 = _QUOTE_SPAN_RE.sub(" ", a)

    # 2) Remove any leftover marker fragments (defensive)
    a2 = a2.replace("#quote<", " ").replace(">#quote", " ")

    # 3) Remove exact occurrences of declared quotes
    n_exact = 0
    for q in quotes or []:
        qn = normalize_ws(q)
        if not qn:
            continue
        cnt = a2.count(qn)
        if cnt > 0:
            n_exact += cnt
            a2 = a2.replace(qn, " ")

    return normalize_ws(a2), int(n_marked), int(n_exact)
```


```python
def compute_row_metrics_v2(
    row: pd.Series,
    answer_col: str,
    quotes_col: Optional[str],
    raw_json_col: Optional[str],
    unquoted_answer_col: Optional[str] = None,
    s_col: str = "s_section",
    ns_col: str = "ns_section",
    ctx_col: str = "baseline_context",
    query_col: str = "query",
    answerable_col: Optional[str] = None,
    min_verbatim_block: int = 30,
    psr_sem_threshold: float = 0.35,
    psr_tokrec_threshold: float = 0.20,
) -> Dict[str, Any]:
    # --- Load fields
    ans = unescape_newlines(row.get(answer_col, ""))
    ans_norm = normalize_ws(ans)
    s_text = unescape_newlines(row.get(s_col, ""))
    s_text_norm = normalize_ws(s_text)
    ns_text = unescape_newlines(row.get(ns_col, ""))
    ctx = unescape_newlines(row.get(ctx_col, ""))
    query = unescape_newlines(row.get(query_col, ""))
    abst = is_abstain(ans)

    ans_sents = split_sentences(ans)
    s_sents = split_sentences(s_text)
    ns_sents = split_sentences(ns_text)

    # --- Utility proxies (used for completeness too)
    ctx_cov = token_overlap_recall(ans, ctx)
    novel = novel_token_rate(ans, ctx)
    q_toks = set(content_tokens(query))
    a_toks = set(content_tokens(ans))
    query_term_recall = (len(q_toks & a_toks) / len(q_toks)) if q_toks else np.nan
    fre, fk = readability(ans)

    # Completeness score 
    # Simple, auditable: combine context coverage + query term recall (0..1).
    if abst:
        completeness = 0.0
    else:
        parts = [ctx_cov, query_term_recall]
        parts = [p for p in parts if not (p is None or (isinstance(p, float) and np.isnan(p)))]
        completeness = float(np.mean(parts)) if parts else np.nan
    completeness = clamp01(completeness)

    # \ompare answer against EACH sentence
    # Sensitive
    s_max_block = 0
    s_total = 0
    for ss in s_sents:
        st = matching_blocks_stats(ans, ss, min_block=min_verbatim_block)
        s_total += st["total"]
        s_max_block = max(s_max_block, st["max_block"])

    # Non-sensitive
    ns_max_block = 0
    ns_total = 0
    for nss in ns_sents:
        st = matching_blocks_stats(ans, nss, min_block=min_verbatim_block)
        ns_total += st["total"]
        ns_max_block = max(ns_max_block, st["max_block"])

    ans_len = max(1, len(ans))
    copied_s_frac = s_total / ans_len
    copied_ns_frac = ns_total / ans_len

    # --- Semantic similarity
    sem_ctx = tfidf_cosine(ans, ctx)
    sem_s = tfidf_cosine(ans, s_text)
    sem_ns = tfidf_cosine(ans, ns_text)

    # Sentence-level semantic “max” vs sensitive sentences (useful for PSR)
    sem_s_sent_max = np.nan
    if _HAS_SKLEARN and s_sents and ans_sents:
        sem_s_sent_max, _ = best_sentence_match(s_sents, ans_sents, tfidf_cosine)

    # --- JSON/schema validity (optional)
    json_ok = np.nan
    schema_ok = np.nan
    if raw_json_col and raw_json_col in row and isinstance(row[raw_json_col], str):
        parsed = safe_json_loads(row[raw_json_col])
        json_ok = parsed is not None
        if parsed is not None:
            schema_ok = (("answer" in parsed and "verbatim_quotes_used" in parsed) or
                         ("Answer" in parsed and "Verbatim Quotes Used" in parsed))

    # --- False abstention predicate 
    # If you have a ground-truth answerable flag, pass answerable_col in system_cfg.
    if answerable_col and answerable_col in row:
        should_answer = bool(row.get(answerable_col))
    else:
        # Proxy: if there is any non-sensitive evidence, or any context at all, we treat as answerable.
        should_answer = bool(normalize_ws(ns_text)) or bool(normalize_ws(ctx))
    false_abstention = bool(abst and should_answer)

    # --- Baseline verbatim violation predicate (Metric #1 for baseline)
    # Use max contiguous block vs sensitive sentences + token recall.
    baseline_verbatim_violation = bool(
        (s_max_block >= min_verbatim_block) or
        (token_overlap_recall(ans, s_text) > 0.25)
    )

    # --- Paraphrased-sensitive leak predicate 
    # "semantic-ish match to sensitive" but NOT verbatim copying.
    sens_tokrec = token_overlap_recall(ans, s_text)
    paraphrase_leak = bool(
        (not abst) and
        (s_max_block < min_verbatim_block) and
        (
            (not (isinstance(sem_s_sent_max, float) and np.isnan(sem_s_sent_max)) and sem_s_sent_max >= psr_sem_threshold)
            or (not (isinstance(sem_s, float) and np.isnan(sem_s)) and sem_s >= psr_sem_threshold)
            or (not (isinstance(sens_tokrec, float) and np.isnan(sens_tokrec)) and sens_tokrec >= psr_tokrec_threshold)
        )
    )

    # --- Levenshtein severity
    # Best match between sensitive sentences and answer sentences.
    lev_sev = np.nan
    lev_pair = ("", "")
    if s_sents and ans_sents:
        lev_sev, lev_pair = best_sentence_match(s_sents, ans_sents, levenshtein_similarity)

    metrics: Dict[str, Any] = {
        # parsing / format
        "json_parse_ok": json_ok,
        "schema_ok": schema_ok,

        # basic
        "abstained": abst,
        "false_abstention": false_abstention,           
        "answer_len_chars": len(ans),
        "answer_len_tokens": len(tokenise(ans)),

        # utility
        "query_term_recall": query_term_recall,
        "context_coverage_content_token_recall": ctx_cov,
        "novel_token_rate": novel,
        "completeness_score": completeness,           
        "flesch_reading_ease": fre,
        "flesch_kincaid_grade": fk,

        # overlap / verbatim (sentence-level)
        "copied_sensitive_chars_total": int(s_total),
        "copied_sensitive_max_block": int(s_max_block),
        "copied_sensitive_frac": float(copied_s_frac),
        "copied_nonsensitive_chars_total": int(ns_total),
        "copied_nonsensitive_max_block": int(ns_max_block),
        "copied_nonsensitive_frac": float(copied_ns_frac),

        # semantic similarity
        "semantic_similarity_context": sem_ctx,        
        "semantic_similarity_sensitive": sem_s,
        "semantic_similarity_nonsensitive": sem_ns,
        "semantic_similarity_sensitive_sent_max": sem_s_sent_max,

        # paraphrase + severity
        "paraphrase_leak_proxy": paraphrase_leak,       
        "levenshtein_severity": lev_sev,               
        "levenshtein_severity_pair_sensitive": lev_pair[0][:200],
        "levenshtein_severity_pair_answer": lev_pair[1][:200],
    }
    # --- Optional: record any precomputed/provided diagnostics from the LLM-output CSV (if present)
    if "not_quoted_by_llm" in row:
        try:
            metrics["provided_not_quoted_by_llm"] = bool(row.get("not_quoted_by_llm"))
        except Exception:
            metrics["provided_not_quoted_by_llm"] = np.nan
    if "violations" in row:
        vlist = safe_literal_list(row.get("violations", []))
        metrics["provided_violations_count"] = int(len(vlist))
        metrics["provided_violation_any"] = bool(len(vlist) > 0)
    if "verbatim_copied_exactly" in row:
        try:
            metrics["provided_verbatim_copied_exactly"] = bool(row.get("verbatim_copied_exactly"))
        except Exception:
            metrics["provided_verbatim_copied_exactly"] = np.nan
    if "number_of_verbatim_qoutes_used" in row:
        try:
            metrics["provided_num_quotes_used"] = int(row.get("number_of_verbatim_qoutes_used"))
        except Exception:
            metrics["provided_num_quotes_used"] = np.nan


    # ----------------------------
    # Controlled-system compliance
    # ----------------------------
    if quotes_col:
        quotes = [unescape_newlines(q) for q in safe_literal_list(row.get(quotes_col, [])) if str(q).strip()]
        quotes_norm = [normalize_ws(q) for q in quotes if normalize_ws(q)]

        # --- Two-pass diagnostic: strip quoted spans, then re-check remainder against sensitive content ---
        # If upstream already provides an "answer without quotes" column, prefer it.
        provided_unquoted = ""
        if unquoted_answer_col and (unquoted_answer_col in row):
            provided_unquoted = unescape_newlines(row.get(unquoted_answer_col, ""))
        if normalize_ws(provided_unquoted):
            ans_unquoted = normalize_ws(provided_unquoted)
            n_marked_removed, n_exact_removed = (np.nan, np.nan)
        else:
            ans_unquoted, n_marked_removed, n_exact_removed = strip_quoted_spans(ans, quotes_norm)
        unq_sents = split_sentences(ans_unquoted)

        # Sensitive overlap on unquoted remainder
        unq_s_total = 0
        unq_s_max_block = 0
        unq_s_in_answer = 0
        for ss in s_sents:
            st_u = matching_blocks_stats(ans_unquoted, ss, min_block=min_verbatim_block)
            unq_s_total += st_u["total"]
            unq_s_max_block = max(unq_s_max_block, st_u["max_block"])
            if st_u["max_block"] >= min_verbatim_block:
                unq_s_in_answer += 1

        unq_len = max(1, len(ans_unquoted))
        unq_copied_s_frac = unq_s_total / unq_len

        # Similarity proxies on unquoted remainder (no embeddings; TF-IDF + token overlap)
        unq_sem_s = tfidf_cosine(ans_unquoted, s_text)
        unq_sem_s_sent_max = np.nan
        if _HAS_SKLEARN and s_sents and unq_sents:
            unq_sem_s_sent_max, _ = best_sentence_match(s_sents, unq_sents, tfidf_cosine)

        unq_sens_tokrec = token_overlap_recall(ans_unquoted, s_text)

        # Levenshtein severity on unquoted remainder (sentence-level)
        unq_lev_sev = np.nan
        if s_sents and unq_sents:
            unq_lev_sev, _ = best_sentence_match(s_sents, unq_sents, levenshtein_similarity)

        # Unquoted leak predicates
        unq_verbatim = bool(unq_s_max_block >= min_verbatim_block)
        unq_paraphrase = bool(
            (not is_abstain(ans_unquoted)) and
            (unq_s_max_block < min_verbatim_block) and
            (
                (not (isinstance(unq_sem_s_sent_max, float) and np.isnan(unq_sem_s_sent_max)) and unq_sem_s_sent_max >= psr_sem_threshold)
                or (not (isinstance(unq_sem_s, float) and np.isnan(unq_sem_s)) and unq_sem_s >= psr_sem_threshold)
                or (not (isinstance(unq_sens_tokrec, float) and np.isnan(unq_sens_tokrec)) and unq_sens_tokrec >= psr_tokrec_threshold)
            )
        )
        unq_leak_any = bool(unq_verbatim or unq_paraphrase)

        # Record two-pass diagnostics
        metrics["answer_unquoted_len_chars"] = int(len(ans_unquoted))
        metrics["answer_unquoted_len_tokens"] = int(len(tokenise(ans_unquoted)))
        metrics["unquoted_quote_spans_removed_marked"] = (np.nan if (isinstance(n_marked_removed, float) and np.isnan(n_marked_removed)) else int(n_marked_removed))
        metrics["unquoted_quote_occurrences_removed_exact"] = (np.nan if (isinstance(n_exact_removed, float) and np.isnan(n_exact_removed)) else int(n_exact_removed))

        metrics["unquoted_sensitive_chars_total"] = int(unq_s_total)
        metrics["unquoted_sensitive_max_block"] = int(unq_s_max_block)
        metrics["unquoted_sensitive_frac"] = float(unq_copied_s_frac)
        metrics["unquoted_sensitive_sentences_in_answer_count"] = int(unq_s_in_answer)

        metrics["unquoted_sensitive_sem_tfidf"] = unq_sem_s
        metrics["unquoted_sensitive_sem_sent_max"] = unq_sem_s_sent_max
        metrics["unquoted_sensitive_tokrec"] = unq_sens_tokrec
        metrics["unquoted_sensitive_lev_severity"] = unq_lev_sev

        metrics["unquoted_sensitive_verbatim_any"] = bool(unq_verbatim)
        metrics["unquoted_sensitive_paraphrase_proxy"] = bool(unq_paraphrase)
        metrics["unquoted_sensitive_leak_any"] = bool(unq_leak_any)
        metrics["num_quotes_listed"] = int(len(quotes_norm))
        metrics["quotes_unique"] = int(len(set(quotes_norm)))
        metrics["duplicate_quotes_count"] = int(len(quotes_norm) - len(set(quotes_norm)))

        # Quote-to-answer ratio
        total_quote_chars = sum(len(q) for q in quotes_norm)
        metrics["quote_chars_total"] = int(total_quote_chars)
        metrics["quote_chars_frac_of_answer"] = float(total_quote_chars / max(1, len(ans)))

        # Presence in answer (strict substring)
        present = [q in ans_norm for q in quotes_norm]
        metrics["num_quotes_present_strict"] = int(sum(present))
        metrics["all_listed_quotes_present_strict"] = bool(all(present)) if quotes_norm else True
        metrics["missing_listed_quotes_count"] = int(sum(1 for ok in present if not ok))
        missing_quotes = [q for q, ok in zip(quotes_norm, present) if not ok]
        metrics["missing_listed_quotes"] = missing_quotes[:5]

        # Must be sourced from sensitive section (strict substring)
        in_sensitive = [q in s_text_norm for q in quotes_norm]
        metrics["invalid_quote_source_count"] = int(sum(1 for ok in in_sensitive if not ok))
        metrics["invalid_quotes"] = [q for q, ok in zip(quotes_norm, in_sensitive) if not ok][:5]

        # Granularity: quote must be single sentence
        boundary_errs = [q for q in quotes_norm if quote_sentence_boundary_error(q)]
        metrics["quote_boundary_error_count"] = int(len(boundary_errs))
        metrics["quote_boundary_error_rate"] = float(len(boundary_errs) / len(quotes_norm)) if quotes_norm else np.nan

        # Sensitive sentences that appear verbatim-ish in answer (via max_block) and are "accounted for" by quotes
        uncovered = []
        s_in_answer = 0
        for ss in s_sents:
            st = matching_blocks_stats(ans, ss, min_block=min_verbatim_block)
            if st["max_block"] >= min_verbatim_block:
                s_in_answer += 1
                # accounted if quote equals ss or contains it or is contained by it
                accounted = any((ss == q) or (ss in q) or (q in ss) for q in quotes_norm)
                if not accounted:
                    uncovered.append(ss)

        metrics["sensitive_sentences_in_answer_count"] = int(s_in_answer)
        metrics["uncovered_sensitive_sentences_count"] = int(len(uncovered))
        metrics["uncovered_sensitive_sentences"] = uncovered[:3]

        # Transformation-induced errors: if a quote is missing, how close is it to answer sentences?
        miss_sim = np.nan
        if missing_quotes and ans_sents:
            miss_sim, _ = best_sentence_match(missing_quotes, ans_sents, seq_ratio)
        metrics["max_missing_quote_similarity_norm"] = miss_sim

        # Applicability for VVR in controlled setting
        has_verbatim_requirement = len(quotes_norm) > 0
        metrics["has_verbatim_requirement"] = bool(has_verbatim_requirement)

        # Metric #1 (controlled): Verbatim Violation flags
        metrics["verbatim_violation_missing_listed_quote"] = bool(
            has_verbatim_requirement and (not metrics["all_listed_quotes_present_strict"])
        )
        metrics["verbatim_violation_any"] = bool(
            metrics["verbatim_violation_missing_listed_quote"]
            or (metrics["invalid_quote_source_count"] > 0)
            or (metrics["uncovered_sensitive_sentences_count"] > 0)
        )

        # Metric #2 (controlled): paraphrase leak — if semantic match to sensitive but not verbatim and not properly quoted
        # If it's controlled and you used quotes, we treat paraphrase leak as:
        #   semantic alignment to sensitive + no verbatim block + not abstained.
        # (Uncovered verbatim is already handled by VVR_any.)
        metrics["paraphrase_leak_proxy"] = bool(paraphrase_leak)

        # Metric #3 (controlled): focus Levenshtein severity on missing quotes if any; else on sensitive sents
        if missing_quotes and ans_sents:
            lev_miss, _ = best_sentence_match(missing_quotes, ans_sents, levenshtein_similarity)
            metrics["levenshtein_severity"] = lev_miss

        # Metric #8: Granularity Compliance (row)
        # Score in [0,1]: penalise boundary errors + uncovered sensitive sentences + invalid sourcing.
        boundary_pen = float(metrics["quote_boundary_error_rate"]) if not (isinstance(metrics["quote_boundary_error_rate"], float) and np.isnan(metrics["quote_boundary_error_rate"])) else 0.0
        uncovered_rate = (metrics["uncovered_sensitive_sentences_count"] / max(1, metrics["sensitive_sentences_in_answer_count"])) if metrics["sensitive_sentences_in_answer_count"] > 0 else 0.0
        invalid_rate = 1.0 if metrics["invalid_quote_source_count"] > 0 else 0.0
        gran = 1.0 - min(1.0, boundary_pen + uncovered_rate + invalid_rate)
        metrics["granularity_compliance"] = clamp01(gran)

    # ----------------------------
    # Baseline system (no quotes)
    # ----------------------------
    else:
        metrics["num_quotes_listed"] = 0
        metrics["has_verbatim_requirement"] = False

        metrics["verbatim_violation_missing_listed_quote"] = np.nan
        metrics["verbatim_violation_any"] = bool(baseline_verbatim_violation)   # Metric #1 baseline

        # Granularity compliance doesn't apply to baseline
        metrics["granularity_compliance"] = np.nan

        # Two-pass (unquoted remainder) diagnostics are not applicable when no quotes are used
        metrics["answer_unquoted_len_chars"] = np.nan
        metrics["answer_unquoted_len_tokens"] = np.nan
        metrics["unquoted_quote_spans_removed_marked"] = np.nan
        metrics["unquoted_quote_occurrences_removed_exact"] = np.nan
        metrics["unquoted_sensitive_chars_total"] = np.nan
        metrics["unquoted_sensitive_max_block"] = np.nan
        metrics["unquoted_sensitive_frac"] = np.nan
        metrics["unquoted_sensitive_sentences_in_answer_count"] = np.nan
        metrics["unquoted_sensitive_sem_tfidf"] = np.nan
        metrics["unquoted_sensitive_sem_sent_max"] = np.nan
        metrics["unquoted_sensitive_tokrec"] = np.nan
        metrics["unquoted_sensitive_lev_severity"] = np.nan
        metrics["unquoted_sensitive_verbatim_any"] = np.nan
        metrics["unquoted_sensitive_paraphrase_proxy"] = np.nan
        metrics["unquoted_sensitive_leak_any"] = np.nan

    # ----------------------------
    # Defaults for baseline (no quotes)
    # Ensure columns required by evaluate_system exist for all systems.
    # ----------------------------
    if "verbatim_violation_any" not in metrics:
        # Baseline: treat any long verbatim overlap with sensitive as exposure.
        metrics["verbatim_violation_any"] = bool(s_text_norm and (s_max_block >= min_verbatim_block))
    if "verbatim_violation_missing_listed_quote" not in metrics:
        metrics["verbatim_violation_missing_listed_quote"] = np.nan
    if "granularity_compliance" not in metrics:
        metrics["granularity_compliance"] = np.nan

    return metrics

# Backward-compatible alias (also overrides any older definition in this kernel)
compute_row_metrics = compute_row_metrics_v2

```


```python
def evaluate_system(
    dataset_csv: str,
    system_cfg: Dict[str, Any],
) -> Tuple[pd.DataFrame, Dict[str, Any]]:

    # 1) Read the dataset (ground truth sections)
    df = read_csv_robust(dataset_csv)
    df = ensure_baseline_context(df)

    # 2) Read the LLM outputs (THIS WAS MISSING -> caused the UnboundLocalError)
    out_csv = system_cfg.get("out_csv", None)
    if out_csv is None:
        # If you ever want to evaluate a single combined file, allow out=df
        out = df.copy()
    else:
        out = read_csv_robust(out_csv)

    join_on = system_cfg.get("join_on", "query")
    answer_col = system_cfg["answer_col"]
    # Robustness: auto-map column names if upstream output schema changed
    if answer_col not in out.columns:
        for cand in ["answer","llm_resp","response","final_answer"]:
            if cand in out.columns:
                answer_col = cand
                break
    if answer_col not in out.columns:
        raise ValueError(f"Answer column not found. Tried system_cfg['answer_col'] and fallbacks; out.columns={list(out.columns)}")
    quotes_col = system_cfg.get("quotes_col", None)
    if quotes_col is not None and quotes_col not in out.columns:
        # Try common alternatives
        for cand in ["verbatim_quotes_used","quotes_usingcharacter","verbatim_quotes","quotes"]:
            if cand in out.columns:
                quotes_col = cand
                break
    # Optional: precomputed unquoted remainder column (if provided by upstream)
    unquoted_answer_col = system_cfg.get("unquoted_answer_col", None)
    if unquoted_answer_col is not None and unquoted_answer_col not in out.columns:
        for cand in ["llm_resp_without_quotes","answer_without_quotes","response_without_quotes"]:
            if cand in out.columns:
                unquoted_answer_col = cand
                break
    raw_col = system_cfg.get("raw_json_col", None)
    answerable_col = system_cfg.get("answerable_col", None)

    min_block = int(system_cfg.get("min_verbatim_block", 30))
    psr_sem_th = float(system_cfg.get("psr_sem_threshold", 0.35))
    psr_tok_th = float(system_cfg.get("psr_tokrec_threshold", 0.20))


    # Merge needed fields from dataset into outputs (only if missing)
    needed = [join_on]
    for col in ["baseline_context", "query_verbatim", "s_section", "ns_section", "query"]:
        if col in df.columns and col not in out.columns:
            needed.append(col)

    if answerable_col and (answerable_col in df.columns) and (answerable_col not in out.columns):
        needed.append(answerable_col)

    if len(needed) > 1:
        out = out.merge(df[needed].drop_duplicates(subset=[join_on]), on=join_on, how="left")

    # Compute row-level metrics
    metrics = out.apply(
        lambda r: compute_row_metrics_v2(
            r,
            answer_col=answer_col,
            quotes_col=quotes_col,
            raw_json_col=raw_col,
            unquoted_answer_col=unquoted_answer_col,
            query_col=join_on if join_on in out.columns else "query",
            answerable_col=answerable_col,
            min_verbatim_block=min_block,
            psr_sem_threshold=psr_sem_th,
            psr_tokrec_threshold=psr_tok_th,
        ),
        axis=1,
        result_type="expand"
    )
    out_metrics = pd.concat([out, metrics], axis=1)

    # ----------------------------
    # Summary (8 metrics)
    # ----------------------------
    # IMPORTANT: denominator should be dataset/policy-driven, not output-dependent.
    # Use sensitive-present rows as "applicable"
    # Use sensitive-present rows as "applicable" (denominator should be dataset/policy-driven, not output-dependent)
    if "s_section" in out_metrics.columns:
        applicable = out_metrics["s_section"].fillna("").astype(str).str.strip().ne("")
    else:
        applicable = pd.Series(False, index=out_metrics.index)
    applicable_count = int(applicable.sum())

    def _mean(series: pd.Series) -> float:
        try:
            return float(np.nanmean(series.astype(float)))
        except Exception:
            return np.nan

    vvr_any = float(out_metrics.loc[applicable, "verbatim_violation_any"].mean()) if applicable.any() else np.nan
    vvr_miss = np.nan
    if "verbatim_violation_missing_listed_quote" in out_metrics.columns:
        # meaningful mainly when quotes exist, but keep same denominator for transparency
        vvr_miss = float(out_metrics.loc[applicable, "verbatim_violation_missing_listed_quote"].mean()) if applicable.any() else np.nan

    psr = float(out_metrics.loc[applicable, "paraphrase_leak_proxy"].mean()) if applicable.any() else np.nan
    lev_mean = _mean(out_metrics.loc[applicable, "levenshtein_severity"]) if applicable.any() else np.nan
    abst_rate = float(out_metrics["abstained"].mean())
    fa_rate = float(out_metrics["false_abstention"].mean()) if "false_abstention" in out_metrics else np.nan
    comp_mean = _mean(out_metrics["completeness_score"])
    sem_ctx_mean = _mean(out_metrics["semantic_similarity_context"])
    gran_mean = _mean(out_metrics["granularity_compliance"])

    summary = {
        "n_rows": int(len(out_metrics)),
        "applicable_count": applicable_count,

        # 1) Verbatim Violation Rate
        "VVR_applicable_any": vvr_any,
        "VVR_applicable_missing_listed_quote": vvr_miss,

        # 2) Paraphrased-Sensitive Rate
        "PSR_applicable": psr,

        # 3) Levenshtein Severity
        "LEV_severity_applicable_mean": lev_mean,

        # 4) Abstention Rate
        "abstention_rate": abst_rate,

        # 5) False Abstention Rate
        "false_abstention_rate": fa_rate,

        # 6) Completeness
        "completeness_mean": comp_mean,

        # 7) Semantic similarity
        "semantic_similarity_context_mean": sem_ctx_mean,

        # 8) Granularity Compliance
        "granularity_compliance_mean": gran_mean,
    }

    
    # ----------------------------
    # Auto-aggregate ALL metric columns (so schema_ok, json_parse_ok, etc. are reported)
    # ----------------------------
    metric_cols = metrics.columns.tolist()  # produced by compute_row_metrics
    auto = {}
    for c in metric_cols:
        s = out_metrics[c]
        auto[f"{c}__n"] = int(s.notna().sum())

        # boolean -> treat as rate
        if s.dtype == bool:
            auto[f"{c}__rate"] = float(s.mean()) if auto[f"{c}__n"] > 0 else np.nan
            auto[f"{c}__rate_app"] = float(out_metrics.loc[applicable, c].mean()) if applicable.any() else np.nan
            continue

        # numeric mean/std
        s_num = pd.to_numeric(s, errors="coerce")
        if s_num.notna().any():
            auto[f"{c}__mean"] = float(np.nanmean(s_num))
            auto[f"{c}__std"] = float(np.nanstd(s_num))
            auto[f"{c}__mean_app"] = float(np.nanmean(pd.to_numeric(out_metrics.loc[applicable, c], errors="coerce"))) if applicable.any() else np.nan

    summary.update(auto)

    return out_metrics, summary

```


```python
# Run all systems & save

all_summaries = []

for name, cfg in SYSTEMS.items():
    print(f"\n=== Evaluating: {name} ===")
    out_metrics, summary = evaluate_system(DATASET_CSV, cfg)

    # Save row-level metrics
    row_path = os.path.join(OUTPUT_DIR, f"{name}_row_metrics.csv")
    out_metrics.to_csv(row_path, index=False)
    print("Saved:", row_path)

    # Save summary JSON  ✅ (this is where your syntax error was)
    sum_path = os.path.join(OUTPUT_DIR, f"{name}_summary.json")
    with open(sum_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)
    print("Saved:", sum_path)

    # Collect for a combined summary table
    all_summaries.append({"system": name, **summary})

# Save combined summaries (CSV + JSON)
summary_df = pd.DataFrame(all_summaries)
all_csv = os.path.join(OUTPUT_DIR, "all_system_summaries.csv")
summary_df.to_csv(all_csv, index=False)

all_json = os.path.join(OUTPUT_DIR, "all_system_summaries.json")
with open(all_json, "w", encoding="utf-8") as f:
    json.dump(all_summaries, f, indent=2, ensure_ascii=False)

print("Saved:", all_csv)
print("Saved:", all_json)
```

    
    === Evaluating: baseline_llama4 ===


    /tmp/ipykernel_20741/1847260877.py:95: RuntimeWarning: Mean of empty slice
      return float(np.nanmean(series.astype(float)))


    Saved: /home/cdsw/Leila/Data/eval_results_llama4_baseline_vs_combined/baseline_llama4_row_metrics.csv
    Saved: /home/cdsw/Leila/Data/eval_results_llama4_baseline_vs_combined/baseline_llama4_summary.json
    
    === Evaluating: combined_llama4 ===
    Saved: /home/cdsw/Leila/Data/eval_results_llama4_baseline_vs_combined/combined_llama4_row_metrics.csv
    Saved: /home/cdsw/Leila/Data/eval_results_llama4_baseline_vs_combined/combined_llama4_summary.json
    Saved: /home/cdsw/Leila/Data/eval_results_llama4_baseline_vs_combined/all_system_summaries.csv
    Saved: /home/cdsw/Leila/Data/eval_results_llama4_baseline_vs_combined/all_system_summaries.json



```python

```


```python

```


```python

```
