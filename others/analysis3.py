# =========================================
# Kaggle - Reddit Depression Language Analysis (Pro Viz Edition)
# =========================================
# ✅ 说明：
# - 交互式可视化使用 Plotly（大气、可悬浮提示、放大缩放）
# - 主题模型用 sklearn LDA，并用热力图展示主题-词强度
# - 使用 TextBlob 做情感极性 & 主观性
# - 使用词云快速概览语义显著词
# - 附带：Top 20 TF-IDF 关键词、Top 20 二元词组、Sankey 大图展示二元词组结构
# - 时间趋势图支持移动均线（3/6 期）以突出长期变化

# =========================================
# 0) 安装依赖
# =========================================
!pip -q install plotly textblob wordcloud

# =========================================
# 1) 导入库
# =========================================
import os
import re
import math
import json
import numpy as np
import pandas as pd
from collections import Counter
from datetime import datetime, timezone

from textblob import TextBlob
from wordcloud import WordCloud

# ML
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import LatentDirichletAllocation

# 可视化
import plotly.express as px
import plotly.graph_objects as go
import plotly.io as pio

# Matplotlib 仅用于保存词云
import matplotlib.pyplot as plt

# 统一 Plotly 风格（深色简洁，可改为 "plotly_white"）
pio.templates.default = "plotly_dark"

# =========================================
# 2) 加载数据
# =========================================
DATA_PATH = "/kaggle/input/depression-sampled-csv/depression-sampled.csv"
df = pd.read_csv(DATA_PATH)

# 基础清洗
df['selftext'] = df.get('selftext', "").astype(str).fillna("")
df['title'] = df.get('title', "").astype(str).fillna("")
df['author'] = df.get('author', "").astype(str).fillna("")

# 去掉被删/空文本
REMOVED = {"[removed]", "[deleted]", "nan", ""}
mask_valid = ~df['selftext'].str.strip().str.lower().isin(REMOVED)
df = df[mask_valid].copy().reset_index(drop=True)

# 时间
if 'created_utc' in df.columns:
    df['created_dt'] = pd.to_datetime(df['created_utc'], unit='s', utc=True, errors='coerce')
else:
    df['created_dt'] = pd.NaT

# =========================================
# 3) 分词、停用词、基础函数
# =========================================
token_re = re.compile(r"[a-z']{3,}", re.IGNORECASE)
stopwords = set("""
a an and are as at be by for from has have i im i'm in is it its it's of on or our so that
the their there they this to was were what when where who why will with you your you're
we we've we'll don't can't won't didn't isn't wasn't shouldnt shouldn't couldnt couldn't
""".replace("’","'").split())

def tokenize(text: str):
    words = token_re.findall(text.lower())
    return [w.strip("'") for w in words if len(w) > 2 and w not in stopwords]

texts = df['selftext'].tolist()

# =========================================
# 4) 词频 & TF-IDF & 二元词组
# =========================================
# 4.1 词频
all_tokens = []
for t in texts:
    all_tokens.extend(tokenize(t))
word_counts = Counter(all_tokens)
top20_freq = pd.DataFrame(word_counts.most_common(20), columns=["word", "count"])

# 4.2 TF-IDF（单词）
tfidf_vec = TfidfVectorizer(
    tokenizer=tokenize,
    max_features=8000,
    max_df=0.8,
    min_df=5
)
X = tfidf_vec.fit_transform(texts)
tfidf_scores = np.asarray(X.sum(axis=0)).ravel()
terms = tfidf_vec.get_feature_names_out()
order = np.argsort(-tfidf_scores)[:20]
top20_tfidf = pd.DataFrame({
    "word": terms[order],
    "tfidf_sum": tfidf_scores[order]
})

# 4.3 二元词组（TF-IDF）
bigram_vec = TfidfVectorizer(
    tokenizer=None,
    lowercase=True,
    stop_words='english',
    ngram_range=(2,2),
    max_df=0.9,
    min_df=8,
    max_features=5000,
    token_pattern=r"(?u)\b[a-zA-Z][a-zA-Z']+\b"
)
X2 = bigram_vec.fit_transform(df['selftext'].tolist())
bigrams = bigram_vec.get_feature_names_out()
bigram_scores = np.asarray(X2.sum(axis=0)).ravel()
order2 = np.argsort(-bigram_scores)[:20]
top20_bigrams = pd.DataFrame({
    "bigram": bigrams[order2],
    "tfidf_sum": bigram_scores[order2]
})
# 拆分为 source/target 以做 Sankey
bt = top20_bigrams.copy()
bt[['src','tgt']] = bt['bigram'].str.split(' ', n=1, expand=True)

# =========================================
# 5) 情感与主观性
# =========================================
df['polarity'] = df['selftext'].apply(lambda x: TextBlob(x).sentiment.polarity)
df['subjectivity'] = df['selftext'].apply(lambda x: TextBlob(x).sentiment.subjectivity)

# =========================================
# 6) 时间趋势（每月帖子数 + 移动均线）
# =========================================
time_df = df[['created_dt']].dropna().copy()
if not time_df.empty:
    monthly = (
        time_df.set_index('created_dt')
        .resample('MS')
        .size()
        .reset_index(name='post_count')
    )
    monthly['month'] = monthly['created_dt'].dt.strftime("%Y-%m")
    monthly['ma3'] = monthly['post_count'].rolling(3, min_periods=1).mean()
    monthly['ma6'] = monthly['post_count'].rolling(6, min_periods=1).mean()
else:
    monthly = pd.DataFrame(columns=['created_dt','post_count','month','ma3','ma6'])

# =========================================
# 7) 主题模型（LDA）
# =========================================
# 主题数可调整
N_TOPICS = 6
lda = LatentDirichletAllocation(n_components=N_TOPICS, random_state=42, learning_method='batch')
lda.fit(X)

# 获取每个主题的前若干关键词（按权重）
def top_terms_for_topic(model, feature_names, n_top_words=12):
    topics = []
    for topic_idx, topic in enumerate(model.components_):
        top_ids = topic.argsort()[:-n_top_words-1:-1]
        topics.append([feature_names[i] for i in top_ids])
    return topics

lda_topics = top_terms_for_topic(lda, terms, n_top_words=12)

# 构建主题-词热力图数据（取每个主题前 20 词）
topic_term_matrix = []
topic_labels = []
term_labels = []
for k in range(N_TOPICS):
    # 取该主题下权重最高的 20 个词
    top_ids = lda.components_[k].argsort()[:-21:-1]
    topic_labels.append(f"Topic {k+1}")
    row_vals = lda.components_[k][top_ids]
    topic_term_matrix.append(row_vals)
    term_labels.append([terms[i] for i in top_ids])

# 让列标签统一（以 Topic 1 的 20 个词为列标签示例，简化显示）
term_axis = term_labels[0] if term_labels else []

# =========================================
# 8) 汇总信息卡片（供参考）
# =========================================
n_posts = len(df)
n_authors = df['author'].nunique()
avg_words = float(np.mean([len(tokenize(t)) for t in df['selftext']])) if n_posts>0 else 0.0
date_min = pd.to_datetime(df['created_dt'].min()).strftime('%Y-%m-%d') if df['created_dt'].notna().any() else "N/A"
date_max = pd.to_datetime(df['created_dt'].max()).strftime('%Y-%m-%d') if df['created_dt'].notna().any() else "N/A"

print("===== Dataset Summary =====")
print(f"Posts: {n_posts}")
print(f"Unique authors: {n_authors}")
print(f"Avg words per post (selftext): {avg_words:.2f}")
print(f"Date range (UTC): {date_min} → {date_max}")

# =========================================
# 9) 可视化（高大上版）
# =========================================

# 9.1 Top 20 TF-IDF 词（水平条形图）
fig_tfidf = px.bar(
    top20_tfidf.sort_values('tfidf_sum'),
    x='tfidf_sum', y='word',
    orientation='h',
    title="Top 20 Keywords by TF-IDF (Selftext)",
    labels={'tfidf_sum':'TF-IDF (sum)', 'word':'Keyword'},
    height=600
)
fig_tfidf.update_layout(xaxis=dict(showgrid=True), yaxis=dict(categoryorder='total ascending'))
fig_tfidf.show()

# 9.2 Top 20 高频词（水平条形图）
fig_freq = px.bar(
    top20_freq.sort_values('count'),
    x='count', y='word',
    orientation='h',
    title="Top 20 Most Frequent Words (Selftext)",
    labels={'count':'Count', 'word':'Word'},
    height=600
)
fig_freq.update_layout(xaxis=dict(showgrid=True), yaxis=dict(categoryorder='total ascending'))
fig_freq.show()

# 9.3 二元词组 Sankey（展示搭配结构）
# 构造 Sankey 节点 & 边
if not bt.empty:
    nodes = sorted(set(bt['src']).union(set(bt['tgt'])))
    idx = {w:i for i,w in enumerate(nodes)}
    link = dict(
        source=[idx[s] for s in bt['src']],
        target=[idx[t] for t in bt['tgt']],
        value=[float(v) for v in bt['tfidf_sum']]
    )
    fig_sankey = go.Figure(data=[go.Sankey(
        node=dict(
            pad=18, thickness=12,
            label=nodes
        ),
        link=link
    )])
    fig_sankey.update_layout(title_text="Top 20 Bigrams (TF-IDF) — Sankey View", height=650)
    fig_sankey.show()

# 9.4 情感分布（箱提琴图）
fig_violin = px.violin(
    df.sample(min(8000, len(df))),  # 采样加速渲染
    y='polarity', box=True, points='suspectedoutliers',
    title="Sentiment Polarity Distribution (Sampled)",
    labels={'polarity':'Polarity [-1~1]'},
    height=500
)
fig_violin.add_hline(y=0, line_dash='dash', line_color='gray')
fig_violin.show()

# 9.5 主观性直方图
fig_subj = px.histogram(
    df, x='subjectivity', nbins=40,
    title="Subjectivity Histogram",
    labels={'subjectivity':'Subjectivity [0~1]','count':'Posts'},
    height=450
)
fig_subj.update_layout(bargap=0.05)
fig_subj.show()

# 9.6 时间趋势（帖子数 + 移动均线）
if not monthly.empty:
    fig_trend = go.Figure()
    fig_trend.add_trace(go.Scatter(
        x=monthly['created_dt'], y=monthly['post_count'],
        mode='lines+markers', name='Monthly Posts'
    ))
    fig_trend.add_trace(go.Scatter(
        x=monthly['created_dt'], y=monthly['ma3'],
        mode='lines', name='MA(3)', line=dict(width=3)
    ))
    fig_trend.add_trace(go.Scatter(
        x=monthly['created_dt'], y=monthly['ma6'],
        mode='lines', name='MA(6)', line=dict(width=3, dash='dash')
    ))
    fig_trend.update_layout(
        title="Monthly Post Count (with Moving Averages)",
        xaxis_title="Month", yaxis_title="Post Count", height=500
    )
    fig_trend.show()

# 9.7 LDA 主题 - 热力图（Topic x Top Terms）
if topic_term_matrix:
    tm = np.vstack(topic_term_matrix)  # shape: (N_TOPICS, 20)
    # 如果列标签 term_axis 为空则兜底
    if len(term_axis) == 0:
        term_axis = [f"term{i+1}" for i in range(tm.shape[1])]

    fig_lda = go.Figure(data=go.Heatmap(
        z=tm,
        x=term_axis,
        y=[f"Topic {i+1}" for i in range(tm.shape[0])],
        colorscale="Viridis",
        colorbar=dict(title="Weight")
    ))
    fig_lda.update_layout(
        title=f"LDA Topics × Top Terms Heatmap (K={N_TOPICS})",
        xaxis_title="Top Terms", yaxis_title="Topics", height=500
    )
    fig_lda.show()

# =========================================
# 10) 词云保存（PNG）
# =========================================
wc = WordCloud(width=1200, height=600, background_color="white").generate(" ".join(all_tokens))
plt.figure(figsize=(12,6))
plt.imshow(wc, interpolation="bilinear")
plt.axis("off")
plt.tight_layout()
plt.savefig("wordcloud_selftext.png", dpi=160)
plt.close()
print("Word cloud saved to: wordcloud_selftext.png")

# =========================================
# 11) 可选：导出关键表（CSV）
# =========================================
top20_freq.to_csv("top20_frequent_words.csv", index=False)
top20_tfidf.to_csv("top20_tfidf_words.csv", index=False)
top20_bigrams.to_csv("top20_bigrams_tfidf.csv", index=False)
if not monthly.empty:
    monthly[['created_dt','post_count','ma3','ma6']].to_csv("monthly_trend.csv", index=False)
print("CSV exported: top20_frequent_words.csv / top20_tfidf_words.csv / top20_bigrams_tfidf.csv / monthly_trend.csv")
