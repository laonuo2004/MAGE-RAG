# Offline Evidence Graph 数据仓库探索报告

本文基于当前代码仓库 `/root/autodl-tmp/ylz/NeurIPS_2026/code`、方法设计文档 `docs/方法设计方案.md`、MinerU 解析产物与已生成的 evidence graph 产物，对当前 offline graph 的设计、实现路径和部分样例结果进行梳理。结论先行：当前 offline 部分已经实现了一个以页面和版面元素为核心的文档级多模态证据图，节点覆盖 page、title、paragraph、list、image、chart、table、equation、code、algorithm 等类型，边覆盖页面包含关系、阅读顺序、同页左右布局、标题层级与 ColPali MaxSim 语义近邻关系，并为节点补齐了用于 QA 的保守语义摘要和 ColPali token embedding。

## 1. 与 Framework 图的对应关系

设计文档中的 offline 阶段包含三件事：

1. 使用 MinerU 对 PDF 做结构化解析，得到页面、文本、图片、表格等元素。
2. 对必要信息用 LLM 补全，例如图片、表格、页面的详细描述。
3. 构建文档级证据图，并用 encoder 编码，用于相似度计算和后续 online 阶段扩展。

当前代码实现基本对应上述流程：

```text
MinerU outputs
  -> load_mineru_document()
  -> build_nodes()
  -> fill_llm_abstracts()
  -> build_structural_edges()
  -> materialize_node_embeddings()
  -> build_semantic_edges()
  -> write_graph_artifacts()
```

入口脚本是 `benchmarks/scripts/build_evidence_graphs.py`。核心构建函数是 `benchmarks/evidence_graph/builder.py::build_document_graph()`。输出目录为：

- MMLongBench: `benchmarks/mmlongbench/data/processed/evidence_graphs/{doc_key}/`
- LongDocURL: `benchmarks/longdocurl/data/processed/evidence_graphs/{shard}/{doc_no}/`

每篇文档输出三个核心文件：

- `graph.json`: 图级 metadata、构建配置、源路径、统计信息。
- `nodes.jsonl`: 每行一个 EvidenceNode。
- `edges.jsonl`: 每行一个 EvidenceEdge。

## 2. 原始数据组织：MinerU 产物

原始解析数据位于：

- `benchmarks/mmlongbench/data/processed/pdfs_mineru/{doc_key}/`
- `benchmarks/longdocurl/data/processed/pdfs_mineru/4000-4999/{doc_no}/`

典型目录包含：

```text
layout.json
*_content_list.json
*_content_list_v2.json
*_model.json
*_origin.pdf
full.md
images/
```

其中当前建图真正依赖两个文件：

- `layout.json`: 提供 `pdf_info`，每个 page 有 `page_idx`、`page_size`、`preproc_blocks`、`para_blocks` 等。
- `*_content_list_v2.json`: 当前节点构建的主要来源，按 page 组织，每页是一个 block 列表。

以 `mmlongbench/2021-Apple-Catalog` 为例：

- `layout.json` 中共有 55 页，首页 `page_size=[612, 792]`、`page_idx=0`。
- `*_content_list_v2.json` 中也有 55 个 page entry。
- 第 0 页 block 类型为 `title`、`paragraph`、多个 `image`、`page_footer`。
- 第 1 页 block 类型主要为 `title`、`paragraph`。
- 第 2 页包含 `title`、`image`、`paragraph`、`page_header`、`page_number`。

以 `mmlongbench/2005.12872v3` 为例：

- 共 26 页。
- 第 0 页包括论文标题、作者段落、摘要段落、页脚等。
- 第 1 页首个 block 是 `image`，实际对应 DETR 架构图和 caption。
- 后续页面包含 `paragraph`、`title`、`equation_interline`、`table`、`chart`、`code` 等学术论文常见元素。

以 `longdocurl/4004473` 为例：

- 共 72 页。
- 第 0 页为 Cummins 2020 Sustainability Progress Report 封面，包含多个 `title` 和 `paragraph`。
- 第 1 页含标题、段落和图片。
- 第 2 页开始出现多栏布局，多个文本 block 在同页并列。

这说明 MinerU v2 content list 已经把原始 PDF 解析为适合构图的元素序列：每个 block 有 `type`、`bbox`、`content`，其中 `content` 内部再按类型保存文本 span、图片路径、caption、HTML 表格、LaTeX 公式、代码等。

## 3. 节点设计

节点 schema 定义在 `benchmarks/evidence_graph/schema.py::EvidenceNode`。核心字段为：

```text
id, type, doc_id, page_index, index, bbox, abstract,
embedding_path, in_edges, out_edges, metadata, fields
```

当前节点分两层：

1. Page 节点：每页一个，id 形如 `{doc_key}:page:{page_index}`。
2. Element 节点：来自每页 content blocks，id 形如 `{doc_key}:page:{page_index}:block:{block_index}:{block_type}`。

当前会跳过辅助类型：

```text
page_header, page_footer, page_number, page_aside_text, page_footnote
```

保留的主要节点类型包括：

```text
page, title, paragraph, list, image, chart, table,
equation_interline, code, algorithm
```

不同节点类型的字段不同：

- `title`: `title`, `level`
- `paragraph`: `paragraph`
- `list`: `list_type`, `items`
- `image`: `image_path`, `content`, `caption`, `footnote`
- `chart`: `image_path`, `content`, `caption`, `footnote`
- `table`: `image_path`, `html`, `table_type`, `table_nest_level`, `caption`, `footnote`
- `equation_interline`: `latex`, `math_type`, `image_path`
- `code`: `language`, `code`, `caption`
- `algorithm`: `algorithm`, `caption`

每个 element 节点的 `metadata.source_block_index` 保留其在 MinerU page block list 中的原始位置，`metadata.source_text` 保留由 `content.block_text()` 提取出的可读文本。

## 4. 节点摘要：LLM Grounding Abstract

`benchmarks/evidence_graph/summaries.py` 会为节点生成 `abstract`。该模块的设计重点是“保守、忠实、只使用源内容和可见图像信息”，这与后续 QA 场景中的 evidence grounding 目标一致。

LLM 摘要策略有几个特点：

- `page`、`image`、`chart`、`table`、`equation_interline` 会附带图像输入，视觉节点不仅依赖 OCR 文本，也允许描述可见结构、标签、箭头、表格单元格等。
- `paragraph`、`title` 等非视觉节点要求不能比源文本引入更多事实。
- prompt 中明确禁止外部知识、推测、补全缺失上下文。
- 对短片段如作者行、报告编号、页码、标签，要求只做短句复述。

样例结果：

- `2005.12872v3:page:1:block:0:image` 的摘要准确描述 DETR 架构图：输入 image 经 CNN、transformer encoder-decoder 得到 box predictions，再通过 bipartite matching loss 与 ground truth 匹配。
- `2005.12872v3:page:8:block:0:table` 的摘要概括 COCO validation 上 Faster R-CNN 与 DETR 的 AP 指标对比，并保留 DETR 小目标 AP 较低、大目标 AP 较高的显式结论。
- `2021-Apple-Catalog:page:10:block:0:table` 的摘要从表格 HTML 和图像中提取 One55、One65、One60 的功能差异，例如 Lift Length、Enhanced Security、High Security、Data Analytics、Brackets。
- `2021-Apple-Catalog:page:42:block:6:chart` 的摘要描述了 dashboard 上的四个橙色 bar charts 以及多设备展示。

这类 abstract 是当前图中最接近“可直接给 Action Evaluator/Reader 使用的语义证据”的字段。

## 5. 边设计

边 schema 定义在 `EvidenceEdge`：

```text
id, source, target, type, relation, weight, metadata
```

当前边分为五类。

### 5.1 containment

构造位置：`benchmarks/evidence_graph/edges.py::build_structural_edges()`。

每个 page 指向该页所有非辅助元素：

```text
page -> element
type=containment
relation=contains
weight=bbox_area
```

权重是元素 bbox 面积。如果 bbox 缺失，则默认为 1.0。这个边非常适合 online 阶段的 `Open Element`：先激活 page，再枚举 page contains 的候选元素。

### 5.2 reading_order

包含两条链：

- Page 级顺序：`page_i -> page_{i+1}`
- Element 级顺序：按 `page_index` 和 `index` 全文串联所有 readable element

字段为：

```text
type=reading_order
relation=next
weight=1.0
```

这对应 online action 中的 `Turn Page` 和顺序阅读扩展。

### 5.3 layout

同页内根据 bbox 计算左右关系。当前仅建模横向关系：

```text
type=layout
relation=left_of/right_of
weight=1 / (1 + center_distance)
metadata.distance=center_distance
```

每个 source 在同页中最多连接 `layout_k` 个候选；当前已生成图的配置通常为 `layout_k=1`。这个边适合多栏论文、产品手册、报告中的左右邻接扩展。

### 5.4 section_hierarchy

按标题节点的 `level` 构建栈：

- 标题到子标题：`relation=has_subsection`
- 最近标题到普通 block：`relation=contains_block`

当前统计里 `contains_block` 很多，但样本中尚未明显看到大量 `has_subsection`。这与 MinerU 输出的 title level 质量有关：若 title level 不够细，标题层级边会退化为“当前标题包含后续 block”。

### 5.5 semantic

构造位置：`benchmarks/evidence_graph/semantic.py`。

所有节点两两计算 ColPali token embedding 的 MaxSim 分数，每个 source 取 Top-K：

```text
type=semantic
relation=similar_to
weight=maxsim_score
metadata.rank=rank
metadata.scoring=maxsim
```

当前配置通常为 `semantic_k=3`，因此语义边数量约等于 `3 * 节点数`。例如 `2021-Apple-Catalog` 有 652 个节点、1956 条 semantic edges，刚好是 3 倍。

语义边直接支持 framework 中的 `Follow Edge`，也可作为 `Jump` 或 query-driven retrieval 的局部扩展基础。

## 6. Embedding 设计

`benchmarks/evidence_graph/embeddings.py` 负责为节点 materialize ColPali token embedding。

Page 节点不重新编码文本，而是读取已有 page-level PDF image embedding：

```text
colpali_pdf_embeddings_path(benchmark, doc_key, shard)
```

Element 节点分为两类编码路径。

第一类是含有独立图像文件的视觉元素节点：

```text
image, chart, table, equation_interline
```

这些节点会优先读取 `fields.image_path` 指向的图片文件，并发送给 ColPali vLLM `/pooling` 接口生成真正的多模态 token embedding。请求形式与 page-level PDF image embedding 保持一致：

```text
POST {vllm_url}/pooling
model=colpali-v1.3
task=token_embed
messages=[
  {
    role: user,
    content: [
      {type: text, text: "<image>"},
      {type: image_url, image_url: {url: "data:<mime>;base64,<image>"}}
    ]
  }
]
```

因此，`page`、`image`、`chart`、`table`、`equation_interline` 这几类节点的语义边计算都可以基于 ColPali 的视觉/多模态 embedding，而不只是 OCR 文本或 caption。对于表格和公式节点，这一点尤其重要：即使 HTML/LaTeX/OCR 不完整，节点 embedding 仍能利用原始裁剪图像中的版面、单元格、公式符号和视觉结构。

如果视觉元素节点缺少有效 `image_path`，构建逻辑会记录 warning，并回退到文本编码，避免批量建图因为单个图片缺失而中断。

第二类是纯文本或主要文本节点，会按类型拼接文本输入：

- title: title text
- paragraph: paragraph text
- list: items
- code: caption + language + code
- algorithm: caption + algorithm

纯文本节点的编码请求发送到 vLLM `/pooling`：

```text
POST {vllm_url}/pooling
model=colpali-v1.3
task=token_embed
input=[text]
```

生成的节点 embedding 写入：

- MMLongBench: `benchmarks/mmlongbench/data/cache/colpali/node_embeddings/{doc_key}/`
- LongDocURL: `benchmarks/longdocurl/data/cache/colpali/node_embeddings/{shard}/{doc_no}/`

为了避免长文本过长，纯文本节点单次 embedding 输入上限为 6000 字符，超过后按空白、分号、逗号、HTML 边界等切块，再拼接 token embeddings。

注意：早期已生成的部分 node embedding cache 可能是在修正前生成的文本版视觉节点 embedding。若需要让既有图完全符合当前逻辑，应使用 `--overwrite` 重新运行 `benchmarks/scripts/build_evidence_graphs.py`，至少重建包含 `image/chart/table/equation_interline` 的文档节点 embedding 和 semantic edges。

## 6.1 重建与缓存保护策略

由于前期图数据中的视觉元素 ColPali embedding 和基于这些 embedding 建立的 semantic edges 可能不可靠，当前推荐对已有文档执行覆盖式重建。但直接覆盖重建的风险是：LLM 生成的 `abstract` 调用了付费 API，若全部重写会产生不必要成本。

最新代码已经把这两个目标拆开：

- `--overwrite`：允许重写 `graph.json`、`nodes.jsonl`、`edges.jsonl`，并强制重新生成 node-level ColPali embedding cache，从而重建 semantic edges。
- 默认行为：若旧图目录中已有 `nodes.jsonl`，构建器会按稳定的 `node.id` 读取旧节点的 `abstract`，写回新构建的节点，并在 `fill_llm_abstracts()` 阶段保护这些节点，不再调用 LLM。
- `--overwrite-llm-abstracts`：只有显式加上该参数时，才会忽略旧 `abstract` 并重新调用 LLM 生成摘要。

因此，当前重建错误 embedding 和 semantic edges 的推荐命令形态是：

```bash
conda run -n logma-rag-py12 python benchmarks/scripts/build_evidence_graphs.py \
  --benchmark mmlongbench \
  --overwrite \
  --workers 1
```

不要加 `--overwrite-llm-abstracts`，即可保留已经付费生成过的 `abstract`。如果某些节点是新出现的节点，或者旧图中没有对应 `node.id` 的 `abstract`，这些节点仍会按正常逻辑生成摘要；已存在且 id 匹配的节点不会重复生成。

## 7. 已生成图数据概况

截至本次探索，已生成的 graph 数量如下。

### MMLongBench

- 已生成文档数：45
- 总节点数：18,393
- 总边数：108,288

节点类型统计：

```text
paragraph: 9865
title: 3422
image: 1586
page: 1550
table: 736
list: 727
chart: 342
code: 81
equation_interline: 78
algorithm: 6
```

边类型统计：

```text
semantic: 55179
reading_order: 18303
containment: 16843
section_hierarchy: 10144
layout: 7819
```

最大几篇样例：

```text
2311.16502v3: 1399 nodes, 8063 edges
52b3137455e7ca4df65021a200aef724: 1363 nodes, 8505 edges
ADOBE_2015_10K: 1298 nodes, 7068 edges
2024.ug.eprospectus: 1162 nodes, 7749 edges
BESTBUY_2023_10K: 993 nodes, 5421 edges
```

### LongDocURL shard 4000-4999

- 已生成文档数：73
- 总节点数：57,151
- 总边数：333,620

节点类型统计：

```text
paragraph: 29465
title: 9855
page: 5931
image: 5768
list: 2947
table: 2365
chart: 414
equation_interline: 327
code: 67
algorithm: 12
```

边类型统计：

```text
semantic: 171453
reading_order: 57005
containment: 51220
section_hierarchy: 29203
layout: 24739
```

最大几篇样例：

```text
4012073: 2680 nodes, 17375 edges
4009036: 2510 nodes, 16145 edges
4020030: 2415 nodes, 16115 edges
4016274: 2149 nodes, 13765 edges
4009705: 1900 nodes, 12443 edges
```

这些统计能说明两点：

1. 当前 evidence graph 的规模已经明显超过 page-level retrieval，能够细粒度定位到段落、图、表、代码、公式等元素。
2. semantic edge 占边数约一半，因为每个节点默认连 3 个语义近邻；这会让 online expansion 在局部结构边之外拥有跨页跳转能力。

## 8. 具体样例观察

### 8.1 `mmlongbench/2021-Apple-Catalog`

图统计：

```text
nodes: 652
edges: 4001
node types: page 55, title 168, paragraph 129, image 217, list 59, table 23, chart 1
edge types: semantic 1956, reading_order 650, containment 597, section_hierarchy 423, layout 375
```

这是一个产品目录型文档，图中 image 节点很多，table 节点也不少。样例节点显示：

- Page 摘要能够概括整页产品类别，例如 “iPhone & iPad Solutions”。
- Image 节点记录产品图片路径，并用摘要描述可见设备或界面。
- Table 节点保留 HTML，并由 LLM 摘要提取不同型号的功能差异。
- List 节点可保留研发投入、中心面积等 bullet 信息。

这个样例适合测试多模态商品手册问题，例如“某型号是否支持 Data Analytics”“哪个产品 lift length 是 15 inches”等。对应证据通常会落在 table 节点或 page-card + table summary 上。

### 8.2 `mmlongbench/2005.12872v3`

图统计：

```text
nodes: 186
edges: 983
node types: page 26, title 24, paragraph 99, image 15, equation_interline 11, table 5, chart 2, list 3, code 1
edge types: semantic 558, reading_order 184, containment 160, section_hierarchy 73, layout 8
```

这是学术论文型文档。图中覆盖了论文 QA 中常见证据：

- figure/image: DETR 架构图。
- equation: matching loss 公式。
- table: COCO validation 指标。
- chart: decoder layer 与 AP/AP50 曲线。
- code: DETR PyTorch 伪实现。

样例中 `equation_interline` 节点会将 LaTeX 公式转成保守解释，避免把变量解释成未声明领域含义。这个策略对论文问答很重要，因为长文档 QA 中很多错误来自模型看到公式后自行补全语义。

### 8.3 `mmlongbench/05-03-18-political-release`

图统计：

```text
nodes: 201
edges: 1239
node types: page 17, paragraph 134, title 30, chart 18, table 2
edge types: semantic 603, reading_order 199, containment 184, section_hierarchy 128, layout 125
```

这是 Pew Research 报告型文档。特点是 paragraph 和 chart 较多，适合回答政策态度、百分比、党派差异等问题。Page 摘要已经能保留核心调查结论，例如关于 Trump 行政团队伦理标准、经济议题信心等。

这类文档对 chart/table 摘要质量要求高，因为问题往往询问某个百分比或趋势。当前 chart 节点有图像路径和 LLM 摘要，但是否能稳定提取所有数值还需要后续针对评测问题做错误分析。

### 8.4 `longdocurl/4004473`

图统计：

```text
nodes: 1498
edges: 9837
node types: page 72, title 357, paragraph 867, image 126, list 38, table 12, chart 26
edge types: semantic 4494, reading_order 1496, containment 1426, layout 1370, section_hierarchy 1051
```

这是企业 sustainability report，长文档、多图、多标题、多图表。这个样例非常接近 framework 图想解决的“长文档中逐步探索证据”的场景：

- 初始检索可以命中相关 page。
- containment 展开该页标题、段落、图表。
- reading_order 支持跨页连续阅读。
- section_hierarchy 支持从标题进入相关 block。
- semantic edge 支持从某个主题段落跳到跨页相似主题。
- layout edge 对报告中的多栏排版较有价值。

## 9. 当前建图方案的能力边界

当前 offline graph 已经具备以下能力：

1. 粒度细：从 page 下沉到 block-level multimodal nodes。
2. 证据可读：每个节点有 `abstract`，后续 online 阶段不必直接消费原始 OCR/HTML。
3. 多模态可追溯：视觉节点保存 `image_path`，page 节点保存 `image_path`，可在最终 reader 阶段提供 crop 或页面图像。
4. 图结构实用：containment、reading_order、section、layout、semantic 分别服务于 open element、turn page、follow section、layout-aware expansion、semantic jump。
5. embedding 与图绑定：每个节点都有 embedding path，semantic edge 的权重来自统一 MaxSim scoring。

同时也有一些清晰的限制：

1. 当前 `layout` 只建模左右关系，不建模 above/below、caption-near-image、table-near-caption 等更细关系。
2. `section_hierarchy` 完全依赖 MinerU title level；若 title level 粗糙，层级图会比较浅。
3. 目前未看到显式 reference/citation 边，例如 “Fig. 1” 文本段落到对应 figure 节点的边。这类边对 framework 中的 `Follow Edge` 很关键。
4. semantic edge 是全节点 Top-K，规模可控但没有去重或双向合并；online 阶段需要处理相似边带来的冗余。
5. 视觉元素节点 embedding 已改为优先使用图像多模态 ColPali 编码，但既有缓存需要显式 `--overwrite` 才会刷新；否则旧缓存仍可能保留修正前的文本版 embedding。当前 `--overwrite` 默认会复用旧 `abstract`，可以安全用于刷新 embedding 和 semantic edges；只有显式加入 `--overwrite-llm-abstracts` 才会重跑 LLM 摘要。
6. LLM abstract 质量决定 evidence graph 可读性。当前 prompt 很保守，但批量生成中仍需抽样检查 hallucination、漏数字、误读图表、表格 HTML OCR 错误等。由于重建时会按 `node.id` 复用旧摘要，如果 MinerU 解析结果、节点 id 规则或节点粒度发生变化，新节点仍可能需要补充生成 abstract。

## 10. 对后续 Online 阶段的启发

结合当前图结构，online 阶段可以把 action 映射为更具体的图操作：

- Initial Grounding: 用 ColPali page embedding 检索 Top-1 page，激活 `page` 节点。
- Open Element: 从 page 的 `containment/contains` out_edges 中列出候选 element，优先打开与 query 或 active evidence 语义相近的元素。
- Turn Page: 沿 page-level `reading_order/next` 扩展前后页；当前只有正向 next，若需要上一页，可在运行时用 in_edges 反查。
- Follow Edge: 沿 `section_hierarchy`、`layout`、`semantic` 扩展。未来若补 reference 边，可把 “见 Figure/Table/Section” 跳转做成强动作。
- Jump: 用 query 或 evaluator 生成的 evidence gap query 做 page/node retrieval，然后激活新 page 或新节点。
- Prune/Compress: 利用 `abstract`、edge type、edge weight、activation history 压缩 evidence subgraph。

最终 Answering 阶段可以把 active subgraph 序列化为：

```text
Page Card:
  page id, page index, page abstract, page image

Opened Elements:
  node id, type, bbox, abstract, key fields

Graph Context:
  edge relation, neighboring node abstract, source/target provenance
```

对于视觉节点，可进一步使用 `bbox` 和 `image_path` 做 crop，而不是只给整页图像。

## 11. 建议的下一步检查项

1. 抽样做 graph-browser/debug view：输入 doc_key 后显示 page image、节点 bbox、abstract、边列表，便于人工检查。
2. 增加 reference edge：从 paragraph/title/list 中正则识别 `Figure 1`、`Table 2`、`Section 3.1`，再链接到对应 caption/title 节点。
3. 增加 vertical layout/caption proximity edge：特别是图表 caption 与图表主体、表格与脚注。
4. 对 semantic edge 做质量评估：抽样查看 Top-3 是否真的有助于跨页扩展，必要时按节点类型过滤或调整 K。
5. 针对 chart/table QA 做专项抽样：检查 LLM abstract 是否保留足够数值，尤其是百分比、行列标题、单位。
6. 为 online evaluator 设计候选动作 preview 格式：每条动作应包含 action type、target node/page、edge type、preview abstract、预计新增 token/crop 成本。

总体来看，当前 offline graph 已经是一个可用的“文档证据记忆层”：它把长文档拆成可追踪、可读、可检索、可扩展的多模态节点，并提供了足够多的结构边和语义边支撑后续 iterative expansion。后续工作的重点应转向在线阶段如何选择动作、如何避免无效扩展、如何把 active subgraph 压缩成高质量 LVLM 输入。
