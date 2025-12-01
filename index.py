import pandas as pd
from pathlib import Path

files = [
    "chanyuhin_news_Qwen2.5-0.5B-Instruct_results.csv",
    "chanyuhin_news_Qwen2.5-3B-Instruct_results.csv",
    "chanyuhin_news_Yi-1.5-9B-Chat-AWQ_results.csv",
    "chanyuhin_news_Yi-1.5-6B-Chat_results.csv",
    "chanyuhin_news_Qwen2.5-14B-Instruct-AWQ_results.csv",
    "chanyuhin_news_gemma-3-4b-it__results.csv",
    "chanyuhin_news_gemma-3-1b-it__results.csv",
    "chanyuhin_news_poe.csv",
    "chanyuhin_news_deepseek.csv",
    "chanyuhin_news_perplexity.csv",
]

def load_all(files):
    dfs = []
    for f in files:
        df = pd.read_csv(f)
        df["source_file"] = f
        df["short_model"] = df["model_name"].astype(str).str.split("/").str[-1]
        dfs.append(df)
    return pd.concat(dfs, ignore_index=True)

df_all = load_all(files)
print(df_all.head())

mc_mask = df_all["question"].str.contains("Answer in MC", na=False)
df_mc = df_all[mc_mask].copy()

def extract_option(resp: str):
    if not isinstance(resp, str):
        return None
    s = resp.strip()
    if not s:
        return None
    c = s[0]
    return c if c in "ABCDE" else None

df_mc["option"] = df_mc["response"].apply(extract_option)

option_counts = (
    df_mc
    .groupby("short_model")["option"]
    .value_counts()
    .unstack(fill_value=0)
    .reset_index()
)

print(option_counts)
df_mc["q_short"] = (
    df_mc["question"]
    .str.replace("Pretend to be a university student.", "", regex=False)
    .str.strip()
)
per_question = (
    df_mc
    .pivot_table(
        index="q_short",
        columns="short_model",
        values="option",
        aggfunc="first"
    )
)

print(per_question.head())

