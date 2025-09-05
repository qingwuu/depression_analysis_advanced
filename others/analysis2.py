# Kaggle Notebook - Reddit Depression Dataset Language Analysis

# ==============================
# 0) 安装依赖
# ==============================
!pip install textblob wordcloud

# ==============================
# 1) 导入库
# ==============================
import pandas as pd
import numpy as np
import re
from collections import Counter
import matplotlib.pyplot as plt
from wordcloud import WordCloud
from textblob import TextBlob
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import LatentDirichletAllocation

# ==============================
# 2) 加载数据
# ==============================
df = pd.read_csv("/kaggle/input/depression-sampled-csv/depression-sampled.csv")

# 数据清洗
df['selftext'] = df['selftext'].astype(str).fillna("")
REMOVED = {"[removed]", "[deleted]", "nan", ""}
df = df[~df['selftext'].str.strip().str.lower().isin(REMOVED)]
texts = df['selftext'].tolist()

# ==============================
# 3) 分词 & 词频统计
# ==============================
token_re = re.compile(r"[a-z']{3,}", re.IGNORECASE)
stopwords = set("""
a an and are as at be by for from has have i im in is it its of on or our so that 
the their there they this to was were what when where who why will with you your
""".split())

def tokenize(text):
    words = token_re.findall(text.lower())
    return [w.strip("'") for w in words if len(w) > 2 and w not in stopwords]

all_tokens = []
for t in texts:
    all_tokens.extend(tokenize(t))

word_counts = Counter(all_tokens)
print("Top 20 高频词：", word_counts.most_common(20))

# ==============================
# 4) TF-IDF 分析
# ==============================
vectorizer = TfidfVectorizer(
    tokenizer=tokenize,
    max_features=5000,
    max_df=0.8,
    min_df=5
)
X = vectorizer.fit_transform(texts)
tfidf_scores = np.asarray(X.sum(axis=0)).ravel()
terms = vectorizer.get_feature_names_out()
top_idx = np.argsort(-tfidf_scores)[:20]

print("\nTop 20 TF-IDF 词：")
for i in top_idx:
    print(terms[i], tfidf_scores[i])

# ==============================
# 5) 情感分析
# ==============================
df['sentiment'] = df['selftext'].apply(lambda x: TextBlob(x).sentiment.polarity)
df['subjectivity'] = df['selftext'].apply(lambda x: TextBlob(x).sentiment.subjectivity)

print("\n平均情感极性：", df['sentiment'].mean())
print("平均主观性：", df['subjectivity'].mean())

# 可视化情感分布
plt.hist(df['sentiment'], bins=40, color="skyblue", edgecolor="k")
plt.title("情感极性分布 (Polarity)")
plt.xlabel("Polarity [-1=负向, +1=正向]")
plt.ylabel("帖子数")
plt.show()

# ==============================
# 6) 词云
# ==============================
wc = WordCloud(width=800, height=400, background_color="white").generate(" ".join(all_tokens))
plt.figure(figsize=(10,5))
plt.imshow(wc, interpolation="bilinear")
plt.axis("off")
plt.title("Word Cloud - Selftext")
plt.show()

# ==============================
# 7) 主题建模 (LDA)
# ==============================
lda = LatentDirichletAllocation(n_components=5, random_state=42)
lda.fit(X)

terms = vectorizer.get_feature_names_out()
for idx, topic in enumerate(lda.components_):
    print(f"\n主题 {idx+1}:")
    print([terms[i] for i in topic.argsort()[:-11:-1]])
