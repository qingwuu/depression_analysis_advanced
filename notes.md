# 深入理解 argparse、TF-IDF、LDA

## 1. argparse（命令行参数解析器）

### 专业版解释

- **argparse** 是 Python 标准库，用来把命令行参数解析成结构化对象 `args`。
- 使用流程：

  1. 创建解析器：
     ```python
     parser = argparse.ArgumentParser(description="描述信息")
     ```
  2. 添加参数：
     ```python
     parser.add_argument("--csv", type=str, default="data.csv")
     ```
  3. 解析：
     ```python
     args = parser.parse_args()
     ```
  4. 使用：
     ```python
     args.csv
     ```

- **ArgumentParser** 是类，构造出的对象就是“解析器”。
- **你脚本里用到的参数**：
  - `--csv`: 数据路径（字符串，默认 `depression-sampled.csv`）
  - `--topics`: LDA 主题数（整数，默认 6）
  - `--renderer`: Plotly 渲染器（取值限定：`auto/vscode/notebook/browser`）
  - `--no-autoinstall`: 是否禁用自动安装（布尔开关，出现即为 True）
  - `--no-export-html`: 是否禁止导出 HTML（布尔开关）

### 常用参数选项

- `type=...`：参数类型（int、str、float…）
- `default=...`：默认值
- `choices=[...]`：限定取值
- `required=True`：强制要求必须传
- `action="store_true"`：布尔开关（出现时 True，不出现时 False）

### 类比

把 `ArgumentParser` 想成“前台接待”，  
`add_argument` 是“登记你要提供的资料”，  
`parse_args` 是“把资料整理好交给后台（args）”。

---

## 2. TF-IDF（Term Frequency–Inverse Document Frequency）

### 专业版解释

- 目标：把文本转换成数值向量，衡量词语对当前文档的重要性。
- **TF（词频）**：词在文档内的出现频率。
  \[
  \text{tf}\_{t,d} = \frac{\text{count}(t \text{ in } d)}{\text{总词数}}
  \]
- **IDF（逆文档频率）**：词在整个语料库中有多稀有。
  \[
  \text{idf}\_t = \log\left(\frac{1+n}{1+\text{df}\_t}\right) + 1
  \]
  - \(n\)：文档总数
  - \(\text{df}\_t\)：包含词 t 的文档数
- **TF-IDF = TF × IDF**  
  稀有词在某篇文档里频繁出现 → 权重高。

- **TfidfVectorizer（sklearn）** 功能：
  1. 分词并建立词表（vocabulary）
  2. 生成文档 × 词的稀疏矩阵（每个值是 TF-IDF）

### 类比

想象每篇文章是一锅汤，词是调料。

- TF = 这锅汤里某调料加了多少勺
- IDF = 这调料在所有汤里有多罕见
- TF-IDF 高的调料 = 这锅汤的“独特味道”

### 注意

- LDA 更适合用 **CountVectorizer**（计数）而不是 TF-IDF。
- 在实践中：TF-IDF 常用于文本分类、搜索排序；LDA 主题建模多用 Count。

---

## 3. LDA（Latent Dirichlet Allocation，潜在狄利克雷分布）

### 专业版解释

- **核心思想**：每篇文档是若干主题的混合，每个主题是若干词的分布。
- **生成过程**：

  1. 每个主题对应一个“词分布” \(\phi_k\)
  2. 每篇文档有一个“主题分布” \(\theta_d\)
  3. 每个词位置：先选主题，再从该主题的词分布里抽词

- **输入**：文档 × 词矩阵（推荐用词频 Count）
- **输出**：

  - 主题 → 词分布（每个主题的高频词）
  - 文档 → 主题分布（每篇文档的主题比例）

- **sklearn.LDA 重要参数**：
  - `n_components`：主题数 K
  - `doc_topic_prior (α)`：文档–主题稀疏性
  - `topic_word_prior (β)`：主题–词稀疏性
- **重要属性**：
  - `.components_`：主题 × 词矩阵，取 top-N 词即可展示主题。

### 类比

把文档想成一幅画：

- 每幅画由几种颜色（主题）混合（比例不同）。
- 每种颜色的“笔触特征”就是它常见的词。
- LDA 训练的任务就是：**反推**“颜色配比”和“每种颜色的典型笔触”。

### 实用提示

- 脚本里用 **TF-IDF 喂给 LDA**，虽然能跑，但推荐改成 **CountVectorizer**，主题更清晰。
- 常见应用：社交媒体文本 → 主题如“求助”“治疗”“情绪分享”。

---

## 小练习（加深理解）

1. **argparse**：加一个 `--max-features` 参数控制词表大小，运行时传不同值观察效果。
2. **TF-IDF**：自己手算一个小例子，验证和 sklearn 结果一致。
3. **LDA**：跑一次 CountVectorizer+LDA，打印每个主题的 top 词，试着给主题命名。

---
