# -*- coding: utf-8 -*-
"""
Reddit Depression Language Analysis — Local VS Code Edition
-----------------------------------------------------------
Usage:
  python analysis3_python.py --csv /path/to/depression-sampled.csv

Options:
  --topics 6                 # LDA 主题数
  --renderer auto            # auto / vscode / notebook / browser
  --no-autoinstall           # 不自动安装缺失依赖
  --no-export-html           # 不导出图表 HTML（默认会导出）
"""

import os
import re
import sys
import argparse
import warnings
from collections import Counter

import numpy as np
import pandas as pd

# ---------------------------
# 0) 依赖管理（可选自动安装）
# ---------------------------
REQ_PKGS = ["plotly", "textblob", "wordcloud", "scikit-learn", "matplotlib"]

def ensure_packages(autoinstall: bool):
    missing = []
    for p in REQ_PKGS:
        try:
            __import__(p if p != "scikit-learn" else "sklearn")
        except ImportError:
            missing.append(p)
    if missing and autoinstall:
        import subprocess
        print(f"[setup] Installing missing packages: {missing}")
        #用当前Python解释器（sys.excutable）执行-m pip install...命令，确保装到同一个解释器环境里
        subprocess.check_call([sys.executable, "-m", "pip", "install", *missing])
    elif missing:
        print("[setup] Missing packages:", missing)
        print("        Install them with:")
        print(f"        {sys.executable} -m pip install " + " ".join(missing))
        #退出程序并返回非零状态码，表示异常结束（依赖不满足）
        sys.exit(1)

# 解析参数（先拿到 --no-autoinstall）
#在终端运行python analysis3_python.py --help时会看到这段描述
parser = argparse.ArgumentParser(description="Reddit Depression Viz (Local)")
parser.add_argument("--csv", type=str, default="depression-sampled.csv",
                    help="Path to depression-sampled.csv")
parser.add_argument("--topics", type=int, default=6, help="Number of LDA topics")
parser.add_argument("--renderer", type=str, default="auto",
                    choices=["auto", "vscode", "notebook", "browser"],
                    help="Plotly renderer")
#store_true的意思是，只要提供了这个参数，就把值设为True，没写就是False
parser.add_argument("--no-autoinstall", action="store_true", help="Disable auto pip installs")
parser.add_argument("--no-export-html", action="store_true", help="Do NOT export figures to HTML")
#真正解析命令行（读取sys.argv），把所有参数装进args对象里
# 例：args.csv、args.topics、args.renderer、args.no_autoinstall、args.no_export_html
args = parser.parse_args()

#调用前面定义的ensure_packages
#autoinstall=not args.no_autoinstall--->没有加no_autoinstall----->autoinstall=True
ensure_packages(autoinstall=not args.no_autoinstall)

# 依赖齐全后再导入这些库
# plotly的三个常用入口：
# px：快速作图（高级API）
# go：底层图元（精细控制）
# pio：I/O与渲染配置（比如到处HTML、设置默认renderer）
import plotly.express as px
import plotly.graph_objects as go
import plotly.io as pio
#TextBlob:用于情感分析等自然语言处理的简易库
#这里主要会用polarity & subjectivity
from textblob import TextBlob
#WorldCloud：生成词云图片的类
from wordcloud import WordCloud
#从scikit-learn里引入：
# TfidVectorizer：把文本转成TF-IDF特征向量
# LatentRirichletAllocation：LAD主题模型
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import LatentDirichletAllocation
import matplotlib.pyplot as plt

#屏蔽非致命告警
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)

# ---------------------------
# 1) Plotly 主题 & 渲染器（兼容各版本）
# ---------------------------
#pio.temolates.default:控制所有Plotly图表的默认主题
# plotly_dark：黑底白字
# plotly_white：白底黑字
# 其他主题还包括 "ggplot2", "seaborn", "simple_white" 等
pio.templates.default = "plotly_dark"  # 可改为 "plotly_white"

#renderer是什么：
# 常见渲染器：
# browser：在默认浏览器里打开图标
# notebook_connected：在jupter notebook里展示
# vscode：在VS Code里展示
# colab：在Google Colab环境里展示
def try_set_renderer(name: str) -> bool:
    """尝试设置渲染器；成功返回 True，失败返回 False（兼容无 .names 的旧版 Plotly）"""
    if not name:
        return False
    try:
        pio.renderers.default = name
        return True
    except Exception:
        return False

#自动选择合适渲染器
def choose_renderer(pref: str = "auto") -> str:
    # 用户显式指定
    #如果用户传了--renderer notebook，会自动映射到notebook_connected
    if pref and pref != "auto":
        alias = {"notebook": "notebook_connected", "jupyter": "notebook_connected"}
        cand = alias.get(pref, pref)
        if try_set_renderer(cand):
            return pio.renderers.default

    # 如果用户没指定，自动探测候选
    for cand in ["vscode", "notebook_connected", "jupyterlab", "colab", "browser", "iframe_connected"]:
        if try_set_renderer(cand):
            return pio.renderers.default

    # 兜底，如果前面都失败，返回当前的默认渲染器
    return pio.renderers.default

renderer_used = choose_renderer(args.renderer)
print(f"[info] Plotly renderer = {renderer_used}")

# 输出目录 & HTML 保存函数
OUTDIR = "outputs"
#如果目录已经存在，就不报错
os.makedirs(OUTDIR, exist_ok=True)

#定义保存图标为HTML的函数
#fig：一个Plotly图标对象
#name：保存的文件名（不带拓展名）
def save_fig_html(fig, name: str):
    #生成文件路径
    html_path = os.path.join(OUTDIR, f"{name}.html")
    # include_plotlyjs="cdn"：使用CDN加载Plotly JS库（文件更小）
    pio.write_html(fig, file=html_path, auto_open=False, include_plotlyjs="cdn")
    return html_path

# ---------------------------
# 2) 加载数据（更稳健）
# ---------------------------
DATA_PATH = args.csv
#检查文件是否存在
#优点：容错性好，不会因为文件名大小或目录变化小问题而直接崩溃
if not os.path.isfile(DATA_PATH):
    # 尝试在当前目录猜测文件名
    candidates = [f for f in os.listdir(".") if f.lower().endswith(".csv") and "depression" in f.lower()]
    if candidates:
        DATA_PATH = candidates[0]
        print(f"[warn] --csv 未找到。改用推测文件: {DATA_PATH}")
    else:
        print(f"[error] CSV 文件不存在：{args.csv}")
        sys.exit(1)

#加载csv数据
print(f"[info] Loading CSV: {DATA_PATH}")
#用Pandas read_csv()读入数据，存放在df（DataFrame）里
df = pd.read_csv(DATA_PATH)

# 统一存在必要列
for col in ["selftext", "title", "author"]:
    if col in df.columns:
        df[col] = df[col].astype(str).fillna("")
    else:
        #新建一列，全为空
        df[col] = ""

# 过滤被删/空文本
REMOVED = {"[removed]", "[deleted]", "nan", ""}
#mask_valid:布尔掩码，True表示“不是无效内容”
# str.strip()去掉空格
# ~取反，表示“有效文本”
mask_valid = ~df["selftext"].str.strip().str.lower().isin(REMOVED)
#只保留有效行
#copy（）避免警告
df = df[mask_valid].copy().reset_index(drop=True)

# 处理时间列
#Reddit数据通常有两种时间戳列：
# created_utc:以秒数表示的Unix时间戳（从 1970-01-01 UTC 开始的秒数）
# created:已经是字符串形式的日期
if "created_utc" in df.columns:
    #errors="coerce"：遇到解析不了的时间自动转成NaT
    df["created_dt"] = pd.to_datetime(df["created_utc"], unit="s", utc=True, errors="coerce")
elif "created" in df.columns:
    df["created_dt"] = pd.to_datetime(df["created"], utc=True, errors="coerce")
else:
    #填充缺失值NaT（Not a Time）
    df["created_dt"] = pd.NaT

# ---------------------------
# 3) 分词/停用词/工具函数
# ---------------------------
token_re = re.compile(r"[a-z']{3,}", re.IGNORECASE)
stopwords = set("""
a about above after again against all am an and any are aren't as at be because been before being below between both but by can't cannot could couldn't did didn't do does doesn't doing don't down during each few for from further had hadn't has hasn't have haven't having he he'd he'll he's her here here's hers herself him himself his how how's i i'd i'll i'm i've if in into is isn't it it's its itself let's me more most mustn't my myself no nor not of off on once only or other ought our ours ourselves out over own same shan't she she'd she'll she's should shouldn't so some such than that that's the their theirs them themselves then there there's these they they'd they'll they're they've this those through to too under until up very was wasn't we we'd we'll we're we've were weren't what what's when when's where where's which while who who's whom why why's with won't would wouldn't you you'd you'll you're you've your yours yourself yourselves
""".replace("’","'").split())

def tokenize(text: str):
    words = token_re.findall(text.lower())
    return [w.strip("'") for w in words if len(w) > 2 and w not in stopwords]

# 每个元素是一篇帖子的正文（字符串）
texts = df["selftext"].tolist()

# ---------------------------
# 4) 词频 / TF-IDF / 二元词组
# ---------------------------
# 4.1 词频
all_tokens = []
for t in texts:
    all_tokens.extend(tokenize(t))
#每个词出现的次数
word_counts = Counter(all_tokens)
top20_freq = pd.DataFrame(word_counts.most_common(20), columns=["word", "count"])

# 4.2 TF-IDF（单词）
tfidf_vec = TfidfVectorizer(
    tokenizer=tokenize,
    max_features=8000,
    #出现在80%以上文档的词会被去掉
    max_df=0.8,
    #出现在少于5篇文档的词会被去掉
    min_df=5
)
#得到稀疏矩阵X，维度：文档数*词表大小，每个值是TF-IDF权重
X = tfidf_vec.fit_transform(texts)
#X.sum(axis=0)：每个词在所有文档中的TF-IDF权重总和
tfidf_scores = np.asarray(X.sum(axis=0)).ravel()
terms = tfidf_vec.get_feature_names_out()
#按权重排序，取前20
order = np.argsort(-tfidf_scores)[:20]
top20_tfidf = pd.DataFrame({
    "word": terms[order],
    "tfidf_sum": tfidf_scores[order]
})

# 4.3 二元词组bigrams（TF-IDF）
bigram_vec = TfidfVectorizer(
    #不用自定义分类器，直接由token_pattern控制
    tokenizer=None,
    lowercase=True,
    #用sklearn内置的英文通用词表
    stop_words="english",
    #只考虑二元词组
    ngram_range=(2, 2),
    max_df=0.9,
    min_df=8,
    max_features=5000,
    #匹配至少两个字母的词
    token_pattern=r"(?u)\b[a-zA-Z][a-zA-Z']+\b"
)
X2 = bigram_vec.fit_transform(df["selftext"].tolist())
bigrams = bigram_vec.get_feature_names_out()
bigram_scores = np.asarray(X2.sum(axis=0)).ravel()
order2 = np.argsort(-bigram_scores)[:20]
top20_bigrams = pd.DataFrame({
    "bigram": bigrams[order2],
    "tfidf_sum": bigram_scores[order2]
})
#拆成前后两个词，方便可视化
#常用于网络图（Network Graph）可视化，把bigram看成“词与词的关系”
bt = top20_bigrams.copy()
if not bt.empty:
    bt[["src", "tgt"]] = bt["bigram"].str.split(" ", n=1, expand=True)

# ---------------------------
# 5) 情感与主观性（TextBlob）
# ---------------------------
#新建情感极性列polarity
#.apply(lambda x: ...)：对每一行执行匿名函数
#TextBlob（x）：把文本包装成TextBlob对象
# .sentiment.polarity：提取情感倾向分数，范围是[-1, 1]
#                       负值：负面情绪  正值：正面情绪  接近0：中性
df["polarity"] = df["selftext"].apply(lambda x: TextBlob(x).sentiment.polarity)
#新建主观性列subjectivity
#.sentiment.subjectivity → 提取主观性分数，范围是 [0, 1]。
# 越接近 1 → 越主观（情绪化、意见）
# 越接近 0 → 越客观（事实描述）
df["subjectivity"] = df["selftext"].apply(lambda x: TextBlob(x).sentiment.subjectivity)

# ---------------------------
# 6) 时间趋势（每月 + 移动均线）
# ---------------------------
#.dropna()：去掉NaT
#.copy()：复制一份，避免Pandas警告
time_df = df[["created_dt"]].dropna().copy()
if not time_df.empty:
    monthly = (
        #设为索引，才能重采样
        time_df.set_index("created_dt")
        #按月初分组统计(Month Start)
        .resample("MS")
        #每个月有多少帖子
        .size()
        #恢复为普通列表，并把计数列命名为post_count
        .reset_index(name="post_count")
    )
    #时间戳更改格式，方便绘图和显示
    monthly["month"] = monthly["created_dt"].dt.strftime("%Y-%m")
    #计算移动平均
    #ma3：短期趋势，波动更敏感
    monthly["ma3"] = monthly["post_count"].rolling(3, min_periods=1).mean()
    #ma5：中期趋势，更平滑但滞后
    monthly["ma6"] = monthly["post_count"].rolling(6, min_periods=1).mean()
else:
    monthly = pd.DataFrame(columns=["created_dt", "post_count", "month", "ma3", "ma6"])

# ---------------------------
# 7) 主题模型（LDA）
# ---------------------------
#从命令行参数topics读取主题数（字符串--->整数），并用max(2, …)保证至少2个主题
N_TOPICS = max(2, int(args.topics))
#learning_method="batch"，另一种是online（适合大语料、增量训练）
#LDA更偏好Count向量（词频）而不是TF-IDF
lda = LatentDirichletAllocation(n_components=N_TOPICS, random_state=42, learning_method="batch")
# X.shape[0]：文档数N   X.shape[1]：词表大小V
#只有N>0 且 V>0 才拟合LDA，否则跳过并将lda=None
#为什么要这样：防御式编程：避免数据全被清洗掉或参数太严导致此表为空时直接报错
#拟合后得到：lda.components_（形状：(K, V)）：每个主题下歌词的“强度/伪计数”，数值越大代表该词在该主题下越重要
if X.shape[0] > 0 and X.shape[1] > 0:
    lda.fit(X)
else:
    print("[warn] 文本稀疏矩阵为空，跳过 LDA。")
    lda = None

def top_terms_for_topic(model, feature_names, n_top_words=12):
    topics = []
    for topic_idx, topic in enumerate(model.components_):
        #topic.argsort()：按照权重从小到大排列后的列索引
        top_ids = topic.argsort()[:-n_top_words-1:-1]
        topics.append([feature_names[i] for i in top_ids])
    return topics

topic_term_matrix = []
term_labels = []
if lda is not None:
    lda_topics = top_terms_for_topic(lda, terms, n_top_words=12)
    for k in range(N_TOPICS):
        # lda.components_[k]：第k个主题对所有词的权重向量（长度V）
        top_ids = lda.components_[k].argsort()[:-21:-1]
        row_vals = lda.components_[k][top_ids]
        #topic_term_matrix：长度 K 的列表，每个元素是长度 20 的数组 → K×20（可直接转 np.array）
        topic_term_matrix.append(row_vals)
        #term_labels：长度 K 的列表，每个元素是长度 20 的字符串列表（对应上面每个数值的词）
        term_labels.append([terms[i] for i in top_ids])
    #取第0个主题的前20歌词作为一个统一的词轴term_axis
    term_axis = term_labels[0] if term_labels else []
else:
    lda_topics = []
    term_axis = []

# ---------------------------
# 8) 汇总信息打印
# ---------------------------
n_posts = len(df)
# nunique() 统计唯一值个数，已在前面将缺失作者填为 ""，因此空字符串也会计入
n_authors = df["author"].nunique()
#这个值受tokenize设计的影响
avg_words = float(np.mean([len(tokenize(t)) for t in df["selftext"]])) if n_posts > 0 else 0.0
date_min = pd.to_datetime(df["created_dt"].min()).strftime("%Y-%m-%d") if df["created_dt"].notna().any() else "N/A"
date_max = pd.to_datetime(df["created_dt"].max()).strftime("%Y-%m-%d") if df["created_dt"].notna().any() else "N/A"

print("\n===== Dataset Summary =====")
print(f"Posts: {n_posts}")
print(f"Unique authors: {n_authors}")
print(f"Avg words per post (selftext): {avg_words:.2f}")
print(f"Date range (UTC): {date_min} → {date_max}\n")

# ---------------------------
# 9) 可视化
# ---------------------------
EXPORT_HTML = not args.no_export_html  # 默认导出 HTML

# 9.1 Top 20 TF-IDF 词
if not top20_tfidf.empty:
    fig_tfidf = px.bar(
        top20_tfidf.sort_values("tfidf_sum"),
        x="tfidf_sum", y="word",
        orientation="h",
        title="Top 20 Keywords by TF-IDF (Selftext)",
        labels={"tfidf_sum": "TF-IDF (sum)", "word": "Keyword"},
        height=600
    )
    fig_tfidf.update_layout(xaxis=dict(showgrid=True), yaxis=dict(categoryorder="total ascending"))
    fig_tfidf.show()
    if EXPORT_HTML:
        path = save_fig_html(fig_tfidf, "top20_tfidf_keywords")
        print(f"[export] {path}")

# 9.2 Top 20 高频词
if not top20_freq.empty:
    fig_freq = px.bar(
        top20_freq.sort_values("count"),
        x="count", y="word",
        orientation="h",
        title="Top 20 Most Frequent Words (Selftext)",
        labels={"count": "Count", "word": "Word"},
        height=600
    )
    fig_freq.update_layout(xaxis=dict(showgrid=True), yaxis=dict(categoryorder="total ascending"))
    fig_freq.show()
    if EXPORT_HTML:
        path = save_fig_html(fig_freq, "top20_frequent_words")
        print(f"[export] {path}")

# 9.3 二元词组 Sankey
#只有当前面以把bigram差分为src/tgt切非空时才绘制Sankey
if "src" in bt.columns and not bt.empty:
    #nodes：所有出现的此节点（来源词+目标此去重后的集合）
    nodes = sorted(set(bt["src"]).union(set(bt["tgt"])))
    #idx：将词映射为节点索引（Sankey需要整数索引）
    idx = {w: i for i, w in enumerate(nodes)}
    #构造Sankey的连线
    link = dict(
        #source：起点节点索引列表，每个bigram的第1个词
        source=[idx[s] for s in bt["src"]],
        #target：中点节点索引列表，第2个词
        target=[idx[t] for t in bt["tgt"]],
        #value：流量，这里用bigram的IF-IDF总和
        value=[float(v) for v in bt["tfidf_sum"]]
    )
    #创建Sankey图
    fig_sankey = go.Figure(data=[go.Sankey(
        #node.pad：节点与节点之间的间距
        #node.thickness：节点条厚度
        #label：节点显示文字
        node=dict(pad=18, thickness=12, label=nodes),
        link=link
    )])
    fig_sankey.update_layout(title_text="Top 20 Bigrams (TF-IDF) — Sankey View", height=650)
    fig_sankey.show()
    if EXPORT_HTML:
        path = save_fig_html(fig_sankey, "bigrams_sankey")
        print(f"[export] {path}")

# 9.4 情感分布（箱提琴图）
if not df.empty:
    fig_violin = px.violin(
        df.sample(min(8000, len(df)), random_state=42),
        # box=True：在小提琴图上叠加箱线图（展示四分位）
        # points="suspectedoutliers"：显示疑似离群点
        y="polarity", box=True, points="suspectedoutliers",
        title="Sentiment Polarity Distribution (Sampled)",
        labels={"polarity": "Polarity [-1~1]"},
        height=500
    )
    # 添加 y=0 的参考虚线（区分正负情感）
    fig_violin.add_hline(y=0, line_dash="dash", line_color="gray")
    fig_violin.show()
    if EXPORT_HTML:
        path = save_fig_html(fig_violin, "polarity_violin")
        print(f"[export] {path}")

# 9.5 主观性直方图
if not df.empty:
    fig_subj = px.histogram(
        df, x="subjectivity", nbins=40,
        title="Subjectivity Histogram",
        labels={"subjectivity": "Subjectivity [0~1]", "count": "Posts"},
        height=450
    )
    fig_subj.update_layout(bargap=0.05)
    fig_subj.show()
    if EXPORT_HTML:
        path = save_fig_html(fig_subj, "subjectivity_hist")
        print(f"[export] {path}")

# 9.6 时间趋势
if not monthly.empty:
    fig_trend = go.Figure()
    fig_trend.add_trace(go.Scatter(
        x=monthly["created_dt"], y=monthly["post_count"],
        mode="lines+markers", name="Monthly Posts"
    ))
    fig_trend.add_trace(go.Scatter(
        x=monthly["created_dt"], y=monthly["ma3"],
        mode="lines", name="MA(3)", line=dict(width=3)
    ))
    fig_trend.add_trace(go.Scatter(
        x=monthly["created_dt"], y=monthly["ma6"],
        mode="lines", name="MA(6)", line=dict(width=3, dash="dash")
    ))
    fig_trend.update_layout(
        title="Monthly Post Count (with Moving Averages)",
        xaxis_title="Month", yaxis_title="Post Count", height=500
    )
    fig_trend.show()
    if EXPORT_HTML:
        path = save_fig_html(fig_trend, "monthly_trend")
        print(f"[export] {path}")

# 9.7 LDA 主题热力图
if topic_term_matrix:
    tm = np.vstack(topic_term_matrix)  # (N_TOPICS, 20)
    term_axis = term_axis if len(term_axis) > 0 else [f"term{i+1}" for i in range(tm.shape[1])]
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
    if EXPORT_HTML:
        path = save_fig_html(fig_lda, "lda_topics_heatmap")
        print(f"[export] {path}")

# ---------------------------
# 10) 词云保存（PNG）
# ---------------------------
if len(all_tokens) > 0:
    wc = WordCloud(width=1200, height=600, background_color="white").generate(" ".join(all_tokens))
    plt.figure(figsize=(12, 6))
    plt.imshow(wc, interpolation="bilinear")
    plt.axis("off")
    plt.tight_layout()
    wc_path = os.path.join(OUTDIR, "wordcloud_selftext.png")
    plt.savefig(wc_path, dpi=160)
    plt.close()
    print(f"[export] {wc_path}")
else:
    print("[warn] 无有效 tokens，跳过词云。")

# ---------------------------
# 11) 导出关键表（CSV）
# ---------------------------
top20_freq.to_csv(os.path.join(OUTDIR, "top20_frequent_words.csv"), index=False)
top20_tfidf.to_csv(os.path.join(OUTDIR, "top20_tfidf_words.csv"), index=False)
top20_bigrams.to_csv(os.path.join(OUTDIR, "top20_bigrams_tfidf.csv"), index=False)
if not monthly.empty:
    monthly[["created_dt", "post_count", "ma3", "ma6"]].to_csv(os.path.join(OUTDIR, "monthly_trend.csv"), index=False)

print("\n[done] CSV exported to outputs/:")
print("       - top20_frequent_words.csv")
print("       - top20_tfidf_words.csv")
print("       - top20_bigrams_tfidf.csv")
if not monthly.empty:
    print("       - monthly_trend.csv")
print("\n[tip] 所有交互式图也已导出为 HTML（在 outputs/ 打开即可用浏览器查看）。")
