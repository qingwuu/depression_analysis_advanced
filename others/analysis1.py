# -*- coding: utf-8 -*-
import pandas as pd
import numpy as np
import re
from datetime import datetime, timezone
from collections import Counter
from glob import glob
import os

# Optional: TF-IDF with scikit-learn (usually available on Kaggle)
try:
    from sklearn.feature_extraction.text import TfidfVectorizer
    SK_AVAILABLE = True
except Exception:
    SK_AVAILABLE = False

# Kaggle/Jupyter-friendly display
try:
    from IPython.display import display
except Exception:
    def display(x):
        print(x if isinstance(x, str) else x.head())

# -----------------------------
# 1) Load data (robust path resolution for Kaggle)
# -----------------------------
# Preferred path from your original script
preferred_path = "/mnt/data/depression-sampled.csv"

# Common Kaggle locations
kaggle_working = "/kaggle/working/depression-sampled.csv"
# Try to auto-locate under /kaggle/input/**/*
auto_candidates = glob("/kaggle/input/**/depression-sampled*.csv", recursive=True)

csv_path = None
for p in [preferred_path, kaggle_working] + auto_candidates:
    if os.path.exists(p):
        csv_path = p
        break

if csv_path is None:
    raise FileNotFoundError(
        "无法找到数据文件 depression-sampled*.csv。\n"
        "请将 CSV 放到以下任一位置后重试：\n"
        "1) /kaggle/working/depression-sampled.csv\n"
        "2) 将数据集添加到本 Notebook，并确保文件名包含 'depression-sampled'。\n"
        "3) /mnt/data/depression-sampled.csv（如果你自定义了该目录）"
    )

df = pd.read_csv(csv_path)

# -----------------------------
# 2) Basic cleanup
# -----------------------------
# unify selftext/title as string; NaNs -> ""
df['selftext'] = df.get('selftext', pd.Series([None]*len(df))).astype(str).fillna("")
df['title'] = df.get('title', pd.Series([None]*len(df))).astype(str).fillna("")
df['author'] = df.get('author', pd.Series([None]*len(df))).astype(str).fillna("")

# remove placeholders from selftext for analysis
REMOVED_SET = {"[removed]", "[deleted]", "nan", ""}
is_removed = df['selftext'].str.strip().str.lower().isin(REMOVED_SET)
selftext_clean = df.loc[~is_removed, 'selftext'].copy()

# parse created_utc (epoch seconds)
if 'created_utc' in df.columns:
    created_dt = pd.to_datetime(df['created_utc'], unit='s', utc=True, errors='coerce')
else:
    created_dt = pd.Series(pd.NaT, index=df.index)

# -----------------------------
# 3) Core metrics requested
# -----------------------------
posts_total = len(df)
unique_authors = df['author'].nunique()

# word count (by whitespace token) for valid selftext rows
word_splitter = re.compile(r"\b[\w']+\b", flags=re.UNICODE)
def count_words(s: str) -> int:
    if not isinstance(s, str) or not s:
        return 0
    return len(word_splitter.findall(s))

word_counts = selftext_clean.apply(count_words)
avg_post_length_words = float(word_counts.mean()) if not word_counts.empty else 0.0
median_post_length_words = float(word_counts.median()) if not word_counts.empty else 0.0
p90_post_length_words = float(word_counts.quantile(0.9)) if not word_counts.empty else 0.0

date_min = pd.to_datetime(created_dt.min()) if created_dt.notna().any() else pd.NaT
date_max = pd.to_datetime(created_dt.max()) if created_dt.notna().any() else pd.NaT
date_range_str = (
    f"{date_min.strftime('%Y-%m-%d')} → {date_max.strftime('%Y-%m-%d')}"
    if pd.notna(date_min) and pd.notna(date_max) else "N/A"
)

# -----------------------------
# 4) Important words (Top 20 from selftext)
# -----------------------------
top_words_df = None

def top_words_frequency(series, topn=20):
    # very light English stoplist
    stop = set("""a an and are as at be by for from has have i i'm im in is it its it's of on or our so that the their there they this to was were what when where who why will with you your youre i've we'll we're don't can't won't didn't isn't wasn't shouldn't shouldnt couldn't couldnt""".replace("’","'").split())
    token_re = re.compile(r"[a-z']{3,}", re.IGNORECASE)
    cnt = Counter()
    for text in series.dropna().astype(str):
        for w in token_re.findall(text.lower()):
            w = w.strip("'")
            if len(w) < 3 or w in stop:
                continue
            cnt[w] += 1
    return pd.DataFrame(cnt.most_common(topn), columns=["word", "count"])

if SK_AVAILABLE:
    try:
        # Use TF-IDF to rank terms by summed tf-idf across docs (better than raw counts)
        # Limit features for performance.
        vectorizer = TfidfVectorizer(
            lowercase=True,
            stop_words='english',
            max_df=0.8,
            min_df=5,
            max_features=5000,
            token_pattern=r"(?u)\b[a-zA-Z][a-zA-Z']{2,}\b",
        )
        X = vectorizer.fit_transform(selftext_clean.tolist())
        # sum tf-idf per term
        term_scores = np.asarray(X.sum(axis=0)).ravel()
        terms = np.array(vectorizer.get_feature_names_out())
        order = np.argsort(-term_scores)[:20]
        top_terms = terms[order]
        top_scores = term_scores[order]
        top_words_df = pd.DataFrame({
            "word": top_terms,
            "tfidf_sum": np.round(top_scores, 6)
        })
        top_words_df.index = np.arange(1, len(top_words_df)+1)
    except Exception:
        # fallback to frequency
        top_words_df = top_words_frequency(selftext_clean, topn=20)
        top_words_df.index = np.arange(1, len(top_words_df)+1)
else:
    # no sklearn -> frequency
    top_words_df = top_words_frequency(selftext_clean, topn=20)
    top_words_df.index = np.arange(1, len(top_words_df)+1)

# -----------------------------
# 5) Extra skills: 
#   - Monthly trend table
#   - Top bigrams via TF-IDF (if available), else frequency
#   - % removed/deleted
# -----------------------------
extras = {}

# Monthly trend
if created_dt.notna().any():
    monthly = (
        pd.DataFrame({"created": created_dt})
        .dropna()
        .set_index("created")
        .resample("MS")
        .size()
        .reset_index(name="post_count")
    )
    monthly['month'] = monthly['created'].dt.strftime("%Y-%m")
    extras['monthly'] = monthly[['month', 'post_count']].copy()
else:
    extras['monthly'] = pd.DataFrame(columns=['month', 'post_count'])

# Bigram TF-IDF (top 15)
def top_bigrams_tfidf(series, topn=15):
    if not SK_AVAILABLE:
        return pd.DataFrame(columns=["bigram","score"])
    vec = TfidfVectorizer(
        lowercase=True,
        stop_words='english',
        ngram_range=(2,2),
        max_df=0.9,
        min_df=8,
        max_features=4000,
        token_pattern=r"(?u)\b[a-zA-Z][a-zA-Z']+\b",
    )
    X = vec.fit_transform(series.tolist())
    scores = np.asarray(X.sum(axis=0)).ravel()
    grams = np.array(vec.get_feature_names_out())
    order = np.argsort(-scores)[:topn]
    return pd.DataFrame({"bigram": grams[order], "tfidf_sum": np.round(scores[order], 6)})

try:
    bigrams_df = top_bigrams_tfidf(selftext_clean, 15)
except Exception:
    bigrams_df = pd.DataFrame(columns=["bigram","tfidf_sum"])

# Removed/Deleted share
removed_ratio = float(is_removed.mean() * 100.0)

# -----------------------------
# 6) Build the requested summary table
# -----------------------------
summary_rows = [
    ["帖子总数", posts_total],
    ["独立作者总数", unique_authors],
    ["帖子平均长度（字数）", round(avg_post_length_words, 2)],
    ["帖子长度中位数（字数）", round(median_post_length_words, 2)],
    ["帖子长度 90 分位（字数）", round(p90_post_length_words, 2)],
    ["数据集的日期范围（UTC）", date_range_str],
    ["正文缺失/被删占比（%）", f"{removed_ratio:.2f}%"]
]
summary_df = pd.DataFrame(summary_rows, columns=["指标", "值"])

# -----------------------------
# 7) Display tables (Kaggle/Jupyter)
# -----------------------------
display(summary_df)

if top_words_df is not None:
    title = "帖子中最重要的 20 个词（TF-IDF）" if 'tfidf_sum' in top_words_df.columns else "帖子中最重要的 20 个词（词频）"
    print("\n" + title)
    display(top_words_df)

if not extras['monthly'].empty:
    print("\n每月发帖数量（UTC）")
    display(extras['monthly'])

if not bigrams_df.empty:
    print("\n重要二元词组（TF-IDF，Top 15）")
    display(bigrams_df)

# -----------------------------
# 8) Save a compact CSV report for download (Kaggle outputs)
# -----------------------------
# Save everything to /kaggle/working so it's downloadable from the "Output" panel
report = summary_df.copy()
report_path = "/kaggle/working/depression_summary_report.csv"
report.to_csv(report_path, index=False)

# Optional: save additional tables for convenience
try:
    top_words_path = "/kaggle/working/top_words.csv"
    top_words_df.to_csv(top_words_path, index=False)
    monthly_path = "/kaggle/working/monthly.csv"
    extras['monthly'].to_csv(monthly_path, index=False)
    bigrams_path = "/kaggle/working/bigrams.csv"
    bigrams_df.to_csv(bigrams_path, index=False)
except Exception:
    pass

# -----------------------------
# 9) Simple matplotlib chart for monthly trend (one plot; no specific colors)
# -----------------------------
import matplotlib.pyplot as plt

plot_path = "N/A"
if not extras['monthly'].empty:
    plt.figure(figsize=(9, 4.5))
    # Construct a datetime index for plotting
    x = pd.to_datetime(extras['monthly']['month'] + "-01", errors='coerce')
    y = extras['monthly']['post_count'].values
    plt.plot(x, y)
    plt.title("Monthly Post Count (UTC)")
    plt.xlabel("Month")
    plt.ylabel("Post Count")
    plt.tight_layout()
    plot_path = "/kaggle/working/monthly_post_count.png"
    plt.savefig(plot_path)
    plt.show()
    plt.close()

# Also print file paths for quick reference in logs
print(f"REPORT_SAVED:{report_path}")
print(f"PLOT_SAVED:{plot_path}")
