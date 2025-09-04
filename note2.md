# TF‑IDF 稀疏矩阵：专业补充与最佳实践 (Pro)

> 在原有笔记基础上，补充更系统、更工程化的说明：从 **CSR 内部结构与复杂度**、**词表构建细节**、**TF‑IDF 公式实现差异**，到 **ngram 与分词**、**内存/性能估算**、**常见坑与推荐实践**。

---

## 0. 快速要点（TL;DR）

- **X / X2**：`scipy.sparse.csr_matrix`（CSR，压缩行存储）。行=文档，列=特征（词 / 二元词组）。
- **shape**：`(N, V)`，其中 `N`=文档数；`V`=词表大小（由清洗流程与超参共同决定）。
- **元素含义**：默认 `X[i, j]` 为 **L2 归一化后的 TF‑IDF 权重**（`norm='l2'`）。
- **TF‑IDF 默认实现**：`idf_j = log((1+n)/(1+df_j)) + 1`（平滑），可选 `sublinear_tf=True` 做 `1 + log(tf)`。
- **ngram**：`ngram_range=(2,2)` 仅提取 **连续 bigram**；与 `tokenizer` / `token_pattern` / 停用词策略强相关。
- **内存估算**：CSR 占用 ≈ `data.nbytes + indices.nbytes + indptr.nbytes`，可用 `dtype=float32` 降低内存。

---

## 1. X / X2 的存储与复杂度（以 CSR 为中心）

### 1.1 CSR 的三个核心数组

- **`data`**：长度 = 非零元素个数 `nnz`，保存非零值（如 TF‑IDF 权重）。
- **`indices`**：长度 = `nnz`，每个非零元素对应的 **列索引**（`int32`/`int64`）。
- **`indptr`**：长度 = `N+1`，第 `i` 行的非零段为 `data[indptr[i] : indptr[i+1]]`。

### 1.2 常见操作的时间复杂度

- **按行遍历**：O(该行的 `nnz`) —— 非常快（CSR 优势）。
- **按列取值**：相对不友好（需要转换为 CSC 或使用稀疏算子）。
- **矩阵乘法 / 点积**：对行相关的操作较优；常与线性模型（LogReg/SVM）配合良好。
- **切片**：`X[i]` 或 `X[i:j]` 代价低；`X[:, j]` 更适合用 **CSC**。

> **对比**：
>
> - **CSC**（压缩列存储）：列操作快、适合特征筛选；
> - **COO**：构建友好、便于增量累加，但计算前通常转 CSR/CSC；
> - **LIL/DOK**：渐进式构建容易，**训练前**再转 CSR/CSC 做高效计算。

---

## 2. 维度与词表构建（`V` 从哪里来）

### 2.1 构建流程（以 `TfidfVectorizer` 为例）

1. **预处理**：`lowercase`、`strip_accents`、`preprocessor`（若提供）。
2. **分词/分析**：由 `analyzer` 决定：

   - `analyzer='word'`（默认）：走 `tokenizer` 或 `token_pattern`；
   - `analyzer='char' / 'char_wb'`：基于字符 n‑gram（对拼写/形态鲁棒）。

3. **ngram 生成**：在 token 序列上滑窗拼接（连续 n‑gram）。
4. **筛选词表**：应用 `min_df` / `max_df`、`stop_words`、`max_features` 等。
5. **编号与定序**：形成 `vocabulary_`（`dict: term -> col_index`），`get_feature_names_out()` 给出列名数组。

### 2.2 关键超参的专业细节

- **`min_df` / `max_df`**：

  - `int`：最小/最大 **文档计数阈值**（出现于多少篇文档）；
  - `float`：**文档占比阈值**（0\~1），常用 `max_df=0.9` 去除过泛滥的词。

- **`max_features`**：按 **语料级总词频**（而非 TF‑IDF）选 top‑K 特征；

  - 若与 `min_df/max_df` 冲突，以过滤后的候选集中再截断。

- **`tokenizer` vs `token_pattern`**：

  - 提供 `tokenizer` 时，**覆盖**默认正则；`token_pattern` 将被忽略；
  - `preprocessor` 仍会生效（注意大小写与标点处理顺序）。

- **停用词**：`stop_words='english'` 或自定义列表；

  - 停用词会影响 n‑gram 生成（停用词被移除后，跨停用词的 bigram 不再出现）。

- **`dtype`**：建议 `np.float32`（半内存，速度足够），默认 `float64`。
- **OOV（未登录词）**：`transform()` 时 **不在训练词表的 token 会被忽略**（列固定）。

> **中文/多语注意**：默认 `token_pattern=r"(?u)\b\w\w+\b"` 不适合中文；需自定义 `tokenizer`（如 jieba）或采用字符 n‑gram。

---

## 3. 矩阵的数值含义与归一化影响

- 默认 `norm='l2'`：每行向量单位化，
  $\|x_i\|_2 = 1$.
- **相似度解释**：在 L2 归一化后，两个文档向量 `x`、`y` 的点积即余弦相似度，
  $x\cdot y = \cos(\theta)$.
- **聚合统计注意**：在 L2 归一化后对列求和 `X.sum(axis=0)`，得到的是 **归一化权重的累积**，不再等价于“原始出现频率”。

  - 若要“全局重要性”或“真实频次”，可：

    - 用 `norm=None` 重新构造 TF‑IDF；或
    - 单独用 `CountVectorizer` 得到 `tf` / `df` 再分析。

---

## 4. TF‑IDF 公式与实现细节

### 4.1 默认公式（`use_idf=True`, `smooth_idf=True`）

- **TF**：

  - `sublinear_tf=False`：$tf_{i,j} = \text{count}(t_j \in doc_i)$
  - `sublinear_tf=True`：$tf = 1 + \log(tf)$

- **IDF**（平滑）：

  $$
  idf_j = \log\!\left(\frac{1+n}{1+df_j}\right) + 1
  $$

- **未归一化 TF‑IDF**：$tfidf^*_{i,j} = tf_{i,j}\cdot idf_j$
- **归一化**：若 `norm='l2'`，再做行向量归一化。

> `smooth_idf=False` 时：$idf_j = \log(n/df_j) + 1$。实际使用中，平滑更稳健，特别是 `df=1` 的边界。

### 4.2 为什么 “看不到公式”

`TfidfVectorizer` 内部等价于：`CountVectorizer`（计数） + `TfidfTransformer`（转换 + 归一化），整体封装后对用户隐藏实现细节。

### 4.3 选择与影响

- **`sublinear_tf=True`**：降低长文档/重复词对权重的支配，常对检索/相似度更稳健。
- **`norm=None`**：若要用 **BM25 等检索算法** 的外部实现，或自定义归一化，先取未归一化的 tf‑idf 更合适。

---

## 5. n‑gram 与分词“语法”

### 5.1 `ngram_range` 的严格含义

- `(min_n, max_n)`：对 token 序列生成 **所有长度在区间内的连续 n‑gram**。
- 样例（`"feel very sad"`）：

  - bigram = `["feel very", "very sad"]`（**不跨越**被移除的 token）。

### 5.2 与 `tokenizer` / `token_pattern` / 停用词的交互

- 若传入 `tokenizer`，将覆盖默认正则切分；
- 移除停用词后形成的新序列再做滑窗（因此很多“跨停用词”的短语会消失）；
- `analyzer='char'/'char_wb'` 生成字符 n‑gram，常对噪声、拼写变体更鲁棒。

### 5.3 选择策略

- **仅 bigram (`(2,2)`)**：强调固定短语；召回可能下降（单词本身被丢弃）。
- **`(1,2)`**：兼顾词与短语（工程上更通用），配合 `max_features`、`min_df` 控制规模。

---

## 6. 可靠检查与可视化方式（避免“误用稠密化”）

### 6.1 查看单行/单列

```python
# 单行的稀疏视图（避免 toarray 全量稠密化）
row = 0
r = X.getrow(row)              # 仍是 CSR
idx, val = r.indices, r.data   # 非零列与权重
terms = tfidf_vec.get_feature_names_out()
print(sorted([(terms[j], v) for j, v in zip(idx, val)], key=lambda x: -x[1])[:10])

# 列聚合（全局 top 词）——注意 norm 的影响
import numpy as np
col_sum = np.asarray(X.sum(axis=0)).ravel()  # 稀疏->小密集向量
order = col_sum.argsort()[::-1][:20]
print([(terms[j], col_sum[j]) for j in order])
```

### 6.2 导出小批量可读表

```python
import pandas as pd
rows = [0, 5, 12]
sub = X[rows]
coo = sub.tocoo()
terms = tfidf_vec.get_feature_names_out()
df_view = pd.DataFrame({
    'row': coo.row,
    'term': [terms[j] for j in coo.col],
    'tfidf': coo.data
}).sort_values(['row', 'tfidf'], ascending=[True, False])
print(df_view.head(30))
```

### 6.3 验证 IDF 与未归一化 TF‑IDF

```python
from math import log
n = X.shape[0]
# 近似 df：>0 的文档计数（用布尔化计数矩阵）
df_counts = np.diff((X > 0).astype(np.int32).sum(axis=0).A1)  # 或者：np.asarray((X>0).sum(axis=0)).ravel()
# sklearn 平滑公式
def idf_smooth(df):
    return np.log((1 + n) / (1 + df)) + 1
```

> 工程建议：**大矩阵避免 `X.toarray()` / `X.todense()`**；如需取一小段用于展示，先切片再稠密化。

---

## 7. 内存与性能估算

### 7.1 估算公式

- **内存**：$\text{Mem} \approx \text{data.nbytes} + \text{indices.nbytes} + \text{indptr.nbytes}$
- 设：`nnz=非零总数`，`N=文档数`，`dtype=float32 (4B)`，`indices=int32 (4B)`，`indptr=int32 (4B)`。

  - data ≈ `4 * nnz` 字节
  - indices ≈ `4 * nnz` 字节
  - indptr ≈ `4 * (N+1)` 字节

### 7.2 数值例子

- 若 `N=50,000`，每文档平均 100 个非零 ⇒ `nnz ≈ 5,000,000`：

  - data ≈ 20 MB；indices ≈ 20 MB；indptr ≈ 0.2 MB；**总计 ≈ 40.2 MB**（不含 Python 对象等开销）。

- 改为 `float64`（8B）大致 **翻倍**。

### 7.3 提速与省内存建议

- 设定 `dtype=np.float32`；
- 控制 `max_features`、调高 `min_df`；
- 优先 `(1,1)` 或 `(1,2)`，谨慎提升 n‑gram 上限；
- 训练后持久化用 `joblib.dump`（支持稀疏）并记录版本与超参。

---

## 8. 常见坑 & 最佳实践清单

- **（坑）对稀疏矩阵做 `.toarray()`**：大数据直接 OOM。
- **（坑）归一化后做总和/均值解释**：语义变化，需改用 `norm=None` 或 `CountVectorizer`。
- **（坑）仅用 `(2,2)`**：单词被丢弃，召回显著下降。更常用 `(1,2)`。
- **（坑）中文直接用默认 `token_pattern`**：分词失效，需自定义 tokenizer 或用 char n‑gram。
- **（坑）`max_features` 后特征含义变化**：再次 `fit` 的词表顺序可能不同；需固定语料或保存 `vocabulary_`。
- **（荐）对长文档**：`sublinear_tf=True` 往往更稳健。
- **（荐）下游用余弦相似度/线性分类器**：保留 `norm='l2'` 通常合适；若改用树模型可以考虑 `norm=None`。
- **（荐）可解释性**：导出 `idf_`、列和、Top‑K 特征，结合领域停用词表迭代清洗。

---

## 9. 最小可复现实验（含 uni/bigram）

```python
from sklearn.feature_extraction.text import TfidfVectorizer
import numpy as np

corpus = [
    "I really love deep learning",
    "Deep learning models really love data",
    "I love classical machine learning"
]

# 1) unigram
uni_vec = TfidfVectorizer(ngram_range=(1,1), dtype=np.float32, sublinear_tf=True)
X = uni_vec.fit_transform(corpus)
print("unigram shape:", X.shape, " nnz:", X.nnz)

# 2) bigram
bi_vec = TfidfVectorizer(ngram_range=(2,2), dtype=np.float32, sublinear_tf=True)
X2 = bi_vec.fit_transform(corpus)
print("bigram  shape:", X2.shape, " nnz:", X2.nnz)

# 3) 词表与 top‑k
terms = uni_vec.get_feature_names_out()
col_sum = np.asarray(X.sum(axis=0)).ravel()
order = col_sum.argsort()[::-1][:5]
print("Top features:", [(terms[i], float(col_sum[i])) for i in order])

# 4) 单文档解析
row = 0
r = X.getrow(row)
idx, val = r.indices, r.data
print("Doc0 top uni:", sorted([(terms[j], float(v)) for j, v in zip(idx, val)], key=lambda x: -x[1])[:5])

# 5) 查看 IDF
idf = uni_vec.idf_
print("Max IDF (rarest terms):", sorted([(terms[i], float(idf[i])) for i in range(len(terms))], key=lambda x: -x[1])[:5])
```

---

## 10. 术语小词典（方便查阅）

- **CSR/CSC/COO/LIL/DOK**：稀疏矩阵存储格式；训练/推理首选 CSR/CSC。
- **`nnz`**：non‑zeros，非零元素个数。
- **TF / DF / IDF**：词频 / 文档频次 / 逆文档频率。
- **OOV**：out‑of‑vocabulary，未登录词，不在训练时构建的词表中。
- **`vocabulary_`**：`term -> column index` 的字典映射。

---

### 参考使用建议（综合）

- **文本分类/相似度**：`ngram_range=(1,2)`, `min_df` 合理上调，`sublinear_tf=True`，`norm='l2'`，`dtype=float32`。
- **检索排序**：可输出未归一化 TF‑IDF（`norm=None`），或转 BM25 等专用打分。
- **中文语料**：优先使用 **可靠分词器** 或 **字符 n‑gram**，并维护 **领域停用词表**。
