### After running verbatim_llm_resp, this code :
#####   removes the verbatim quotes from LLM response
#####   conducts a similarity comparison between the cleaned LLM response and the verbatim text at sentence level
#####   if a similarity above the given threshold is identified then lists the sentences


```python
from typing import List, Dict, Tuple
import re
import difflib
import pandas as pd
```


```python
import nltk
import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
from rapidfuzz import fuzz
import json
import ast
```


```python
import tokenize
```


```python
#read data
df = pd.read_csv('.../new_verb_output.csv')
```


```python
df.shape
```


```python
MODEL_NAME = ".../all_MiniLM_L6_v2"
model = SentenceTransformer(MODEL_NAME)
```


```python
SEMANTIC_THRESHOLD = 0.85 
LEXICAL_THRESHOLD = 60 # llow lexical similarity
MIN_SENTENCE_LENGTH = 10 # ignore very short sentences - is that ok?
```


```python
#remove verbatim from response block 

def remove_sentences_from_answer(sentences_col, text2):
    
    #ensure sentences are a Python list
    if isinstance(sentences_col, str):
        sentences = ast.literal_eval(sentences_col)
    else:
        sentences = sentences_col

    #convert llm resp to string
    answer = str(text2)

    # remove the quoting sign
    for s in sentences:
        s = s.strip()

        # remove quoted version
        quoted_pattern = (
            r"#quote(?:<|&lt;)\s*"
            + re.escape(s)
            + r"\s*(?:>|&gt;)#quote"
        )
        answer = re.sub(quoted_pattern, "", answer, flags=re.IGNORECASE)

        # remove plain version (in case it appears unquoted)
        answer = re.sub(re.escape(s), "", answer, flags=re.IGNORECASE)

    # cleanup whitespace / newlines
    answer = re.sub(r"\n{2,}", "\n", answer)
    answer = re.sub(r"\s{2,}", " ", answer).strip()
    return answer

```


```python
def split_sentences(text):
    return text.split()

```


```python
def sliding_windows(tokens, window_size=30, stride=10):
    windows = []
    for i in range(0, len(tokens) - window_size + 1, stride):
        chunk = tokens[i:i + window_size]
        windows.append(" ".join(chunk))
    return windows
```


```python
def exact_match(sentence, text):
    return sentence in text

```


```python
def lexical_similarity(a, b):
# Token sort ratio helps normalize word order
    return fuzz.token_sort_ratio(a, b)

```


```python
from transformers import AutoTokenizer
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
```


```python
def detect_paraphrase_violations(verbatim_block, response_text):
    model = SentenceTransformer(MODEL_NAME)

    verbatim_tokens = tokenizer.tokenize(verbatim_block)
    response_tokens = tokenizer.tokenize(response_text)

    
    verbatim_chunks = sliding_windows(verbatim_tokens)
    response_chunks = sliding_windows(response_tokens)

    if not verbatim_chunks or not response_chunks:
        return []

    # Compute embeddings
    verbatim_embeddings = model.encode(verbatim_chunks)
    response_embeddings = model.encode(response_chunks)

    violations = []
    for i, v_sent in enumerate(verbatim_chunks):
        for j, r_sent in enumerate(response_chunks):
            # Skip exact matches (these are legitimate verbatim uses) - alreadyy captured - removing this because they are deleted from the resp now
            #if exact_match(v_sent, response_text):
             #   continue
        
            # Semantic similarity
            sim = cosine_similarity(
            [verbatim_embeddings[i]],
            [response_embeddings[j]]
            )[0][0]

            # Lexical similarity
            lex_sim = lexical_similarity(v_sent, r_sent)

            # Detect paraphrase violation
            if sim >= SEMANTIC_THRESHOLD and lex_sim < LEXICAL_THRESHOLD:
                violations.append({
                "verbatim_sentence": v_sent,
                "response_sentence": r_sent,
                "semantic_similarity": float(sim),
                "lexical_similarity": lex_sim
                })
    return violations

```


```python
def extract_answer_safe(x):
    try:
        return json.loads(x).get("answer")
    except Exception:
        return None
```


```python
#df["answer"] = df["llm_resp"].apply(lambda x: json.loads(x)["answer"])
#df["answer"] = df["llm_resp"].apply(extract_answer_safe)
```


```python
#not needed anymore
#remove verbatim_quotes_used from LLM resp 
#df["llm_clean_up_verbatim_block"] = df["answer"].apply(lambda x: x[:str(x).find('verbatim_quotes_used')])
```


```python
df = df.rename(columns ={'s_text': 's_section'})
```


```python
df.columns
```


```python
#remove the quotes marked in the response - these are alreday captired in df['quotes'] column
df['llm_resp_without_quotes']= df.apply(lambda row: remove_sentences_from_answer(row['quotes_usingcharacter'], row['llm_resp']), axis=1)
```


```python
#detect potential violations
df['violations'] = df.apply(lambda row: detect_paraphrase_violations(row['s_section'], row['llm_resp_without_quotes']), axis=1)
```


```python
df['not_quoted_by_llm'] = df['violations'].apply(lambda x: True if len(x)>0 else False)
```


```python
df['answer'] = df['llm_resp']
```


```python
df[['query', 's_section', 'answer', 'verbatim_quotes_used', 'verbatim_copied_exactly',
       'number_of_verbatim_qoutes_used', 'quotes_usingcharacter',
       'llm_resp_without_quotes', 'violations', 'not_quoted_by_llm']].to_csv('.../verbatim_llm_resp_with_quotes_gptoss.csv')
```


```python

```
