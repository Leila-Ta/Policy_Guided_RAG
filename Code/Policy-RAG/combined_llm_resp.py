### Get the LLM response for the combined category. Then run find_verbatim_in_answer to extract the quoted sentences
from elasticsearch import Elasticsearch, exceptions, helpers, AuthenticationException
from urllib.request import urlopen
from getpass import getpass
import json
import time
import getpass
import pandas as pd
import os
import re
from openai import AzureOpenAI
# initiate azure models
client = AzureOpenAI(
        api_key=...,
        azure_endpoint=...,
        api_version=...,
)
def find_quotes(text):
    quotes = re.findall(r"#quote<\s*(.*?)\s*>\.?#quote", text, flags=re.DOTALL)
    quotes = [q.strip() for q in quotes]
    return quotes
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
#### Combined generation
df_combined= pd.read_csv('.../combined.csv')
json_schema ="""{
  "answer": "<final answer text>",
  "verbatim_quotes_used": ["<exact copied text>"],
  "compliance": {
    "verbatim_copied_exactly": true/false,
      "number_of_verbatim_qoutes_used": int,
  }
}
"""
#iterate through the dataframe and generate response for combined text 
resp = []
for index, row in df_combined.iterrows():
    query = row['query']
    verbatim = row['s_section']
    synthesis = row['ns_section']
    prompt = f"""
           
            You are a helpful assistant that answers questions ONLY using the provided content.
            
            CRITICAL RULES:
            
            1) VERBATIM USAGE (STRICT – SENTENCE LEVEL)
            
            - The atomic unit of verbatim text is a COMPLETE, STANDALONE SENTENCE from <VERBATIM_SECTIONS>.
            - You MAY select and use only the sentence(s) from <VERBATIM_SECTIONS> that are relevant to the question. 
            - Anything used from the <VERBATIM_SECTIONs> MUST be listed as verbatim_quotes_used.
            - Any verbatim sentence used MUST be copied character for character.
            - Any verbatim sentence used MUST appear EXACTLY as-is in the final response.
            - Any verbatim sentence used MUST be placed between '#quote<' and '>#quote' in the final response.
            - Any verbatim sentence used MUST be start with '#quote<' and end with '>#quote' in the final response.
            - Do NOT paraphrase, summarise, reorder, truncate, or edit verbatim sentences.
            - Do NOT copy partial sentences, sentence fragments, or substrings.
            - Do NOT merge multiple sentences into one.
            - If any verbatim sentence is used, it MUST be listed in the "verbatim_quotes_used" array.
            - If no verbatim sentence is relevant, do NOT use verbatim text.
            2)COMPLETE, STANDALONE SENTENCE RULES:
            - Conveys a complete, standalone meaning without requiring following text
            - Does NOT introduce, depend on, or imply a list, table, subsection, or continuation
            - Does NOT end with punctuation that signals continuation
            
             3) VERBATIM DELIMITING (SENTENCE LEVEL)
            
            - Each entry in "verbatim_quotes_used" MUST be:
              - Exactly one full sentence (follow COMPLETE, STANDALONE SENTENC rules above)
              - Copied verbatim from <VERBATIM_SECTIONS>
              - Identical in punctuation, spacing, casing, encoding, and line breaks
            - Each sentence listed in "verbatim_quotes_used" MUST appear verbatim in the "answer" field.
            - Do NOT include sentences in "verbatim_quotes_used" that do not appear in the answer.
            
            4) SYNTHESIS USAGE
            
            - Text from <SYNTHESIS_SECTIONS> may be paraphrased for clarity.
            - Do NOT introduce facts not present in VERBATIM or SYNTHESIS.
            
            5) UNSUPPORTED ANSWERS
            
            - If the question cannot be answered using the provided content, respond exactly with:
              "I don't know based on the provided content."
            
            6) CONSISTENCY REQUIREMENT
            
            - If any verbatim sentence appears in the "answer" field but is NOT listed in "verbatim_quotes_used",
              the response is INVALID.
            
            INTERNAL CHECK (must be completed before final output):
            
            - Split <VERBATIM_SECTIONS> into sentences.
            - Identify all verbatim sentences used in the answer.
            - Ensure each used sentence:
              - Matches exactly one sentence from <VERBATIM_SECTIONS>
              - Is copied character-for-character
              - Appears exactly once in "verbatim_quotes_used"
            - If any mismatch exists, revise the response before outputting.
            - set verbatim_copied_exactly=True ONLY if ALL the verbatim sentences are copied character for character ptherwise set to False
            
            OUTPUT FORMAT (JSON only):
            {json_schema}
            
            - If no verbatim text is used, set:
              "verbatim_quotes_used": []
            
            QUESTION:
            {query}
            
            PROVIDED CONTENT:
            <VERBATIM_SECTIONS>
            {verbatim}
            </VERBATIM_SECTIONS>
            
            <SYNTHESIS_SECTIONS>
            {synthesis}
            </SYNTHESIS_SECTIONS>
            """
    messages = [{"role": "user", "content": prompt}]
    response = client.chat.completions.create(
        model='gpt-oss-120b',
        messages=messages,
        temperature=0.0
    )
    resp.append(response.choices[0].message.content)
    
df_combined['llm_resp']= resp
df_combined[["answer", "verbatim_quotes_used", "verbatim_copied_exactly", "number_of_verbatim_qoutes_used"]] = (
    df_combined["llm_resp"].apply(parse_llm_output)
)

df_combined['verbatim_copied_exactly'] = df_combined.apply(lambda row: False if row['number_of_verbatim_qoutes_used']==0 else row['verbatim_copied_exactly'], axis=1)
#now we need to seperate sentences between <#quote and >#quote
df_combined['quotes_usingcharacter'] = df_combined['llm_resp'].apply(lambda x: find_quotes(x))
df_combined.to_csv('.../llm_resp_with_quotes.csv')
df_combined.shape
