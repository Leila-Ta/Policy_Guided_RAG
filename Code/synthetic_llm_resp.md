### Get the LLM response for the synthesis (non-sensitive) category. Then run find_verbatim_in_answer to extract the quoted sentences


```python
from elasticsearch import Elasticsearch, exceptions, helpers, AuthenticationException
from urllib.request import urlopen
from getpass import getpass
import json
import time
import getpass
import pandas as pd
import os
import re
```


```python
from openai import AzureOpenAI
```


```python
# initiate azure models
client = AzureOpenAI(
        api_key= ...,
        azure_endpoint=...,
        api_version=...,
)
```


```python
def find_quotes(text):
    quotes = re.findall(r"#quote<?\s*(.*?)\s*>?\.?#quote", text, flags=re.DOTALL)
    quotes = [q.strip() for q in quotes]
    return quotes
```


```python
def parse_llm_output(s):
    try:
        obj = json.loads(s)

        answer = obj.get("answer", None)
        quotes = obj.get("verbatim_quotes_used", None)

        compliance = obj.get("compliance", {}) or {}
        copied = compliance.get("verbatim_copied_exactly", None)
        n_quotes = compliance.get("number_of_verbatim_qoutes_used", None)

        return pd.Series({
            "answer": answer,
            "verbatim_quotes_used": quotes,
            "verbatim_copied_exactly": copied,
            "number_of_verbatim_qoutes_used": n_quotes
        })

    except Exception:

        return pd.Series({
            "answer": None,
            "verbatim_quotes_used": None,
            "verbatim_copied_exactly": None,
            "number_of_verbatim_qoutes_used": None
        })

```

#### Synthetic generation


```python
df_synth= pd.read_csv('.../synthetic.csv')
```


```python

json_schema ="""{
  "answer": "<final answer text>",
  "verbatim_quotes_used": ["<exact copied text>"],
  "compliance": {
    "verbatim_copied_exactly": true/false,
      "number_of_verbatim_qoutes_used": int,
  }
}
"""
```


```python
df_synth.columns
```


```python
#iterate through the dataframe and generate response for combined text 
resp = []
for index, row in df_synth.iterrows():
    query = row['queries']
    #verbatim = row['s_section']
    content = row['text_chunk']
    prompt = f"""
           
                  You are a helpful assistant that answers questions ONLY using the provided content.
        
        CRITICAL RULES (TRACEABILITY MODE — paraphrase allowed):
        
        1) ALLOWED WRITING
        - You MAY paraphrase, summarise, and reorganise the provided content for clarity.
        - You MUST NOT introduce facts or details that are not supported by the provided content.
        
        2) VERBATIM QUOTES USED (EVIDENCE QUOTES)
        - Even though you may paraphrase, you MUST provide traceability by quoting the supporting text you relied on.
        - A “verbatim quote” is an EXACT, contiguous excerpt copied from the PROVIDED CONTENT.
        - Quotes may be a full sentence or a short span (recommended: 1–2 sentences).
        - Each quote MUST be copied character-for-character (same punctuation, spacing, casing).
        - Each quote MUST be wrapped in the answer using:
          #quote<QUOTE_TEXT>#quote
          (Use exactly those markers.)
        
        3) LISTING REQUIREMENT (verbatim_quotes_used)
        - Every quote that appears in the answer MUST also appear as an entry in "verbatim_quotes_used".
        - Every entry in "verbatim_quotes_used" MUST appear verbatim somewhere in the "answer" field.
        - Do NOT include any quote in "verbatim_quotes_used" that does not appear exactly in the answer. 
        
        4) MINIMAL QUOTING
        - Use the minimum number of quotes needed to support the key factual statements (typically 1–5).
        - Prefer quotes that directly support the main claims.
        
        5) UNSUPPORTED ANSWERS
        - If the question cannot be answered using the provided content, respond exactly with:
          "I don't know based on the provided content."
        
        6) COMPLIANCE FIELDS
        - Set compliance.verbatim_copied_exactly = true ONLY if ALL quotes used are exact copies from the PROVIDED CONTENT.
        - Set compliance.number_of_verbatim_qoutes_used = the number of items in "verbatim_quotes_used".
        
        INTERNAL CHECK (must be completed before final output):
        - Identify every #quote<...>#quote in the answer.
        - Confirm each quoted text is an exact substring of the PROVIDED CONTENT.
        - Confirm the set of quotes in the answer matches "verbatim_quotes_used" exactly.
        
        OUTPUT FORMAT (JSON only):
        {json_schema}
        
        QUESTION:
        {query}
        
        PROVIDED CONTENT:
        <CONTENT>
        {content}
        </CONTENT>
    """
    messages = [{"role": "user", "content": prompt}]
    response = client.chat.completions.create(
        model='gpt-oss-120b',
        messages=messages,
        temperature=0.0
    )
    resp.append(response.choices[0].message.content)
    
```


```python
df_synth['llm_resp']= resp
```


```python
df_synth[["answer", "verbatim_quotes_used", "verbatim_copied_exactly", "number_of_verbatim_qoutes_used"]] = (
    df_synth["llm_resp"].apply(parse_llm_output)
)

```


```python
df_synth.head(3)
```


```python
#now we need to seperate sentences between <#quote and >#quote
df_synth['quotes_usingcharacter'] = df_synth['answer'].apply(lambda x: find_quotes(x))
```


```python
find_quotes(df_synth['answer'][2])
```


```python
df_synth.to_csv('.../llm_resp_with_quotes_synth.csv')
```


```python
df_combined.shape
```
