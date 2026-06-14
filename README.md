<div align="center">
  <img src="assets/readme/banner.svg" alt="MAGE-RAG animated banner" width="100%">

<h1>MAGE-RAG</h1>

<p>
    <strong>Multigranular Adaptive Graph Evidence for Agentic Multimodal RAG in Long-Document QA</strong>
  </p>

<p>
    <a href="#项目简介"><img alt="Task" src="https://img.shields.io/badge/Task-Long--Document%20Multimodal%20QA-2f6fed"></a>
    <a href="#framework-overview"><img alt="Method" src="https://img.shields.io/badge/Method-Agentic%20Graph%20RAG-16a34a"></a>
    <a href="#experimental-results"><img alt="LongDocURL" src="https://img.shields.io/badge/LongDocURL-52.75%20Acc-0f766e"></a>
    <a href="#experimental-results"><img alt="MMLongBench-Doc" src="https://img.shields.io/badge/MMLongBench--Doc-53.26%20Acc%20%7C%2051.19%20F1-c2410c"></a>
    <a href="LICENSE"><img alt="License" src="https://img.shields.io/badge/License-MIT-111827"></a>
  </p>

<p>
    <a href="#framework-overview">Framework</a> •
    <a href="#experimental-results">Results</a> •
    <a href="#quick-start">Quick Start</a> •
    <a href="#citation">Citation</a>
  </p>
</div>

---

## 项目简介

**MAGE-RAG** 是一个面向长 PDF 文档多模态问答的检索增强生成框架。它关注的问题不是简单地“检索更多页面”，而是：在上下文、图像数量和模型调用预算有限的情况下，系统应该为当前问题选择、展开、保留或移除哪些证据。

长文档 PDF 中的答案线索往往稀疏且异构，可能分布在段落、表格、图表、图片、标题层级、页面布局和跨页上下文中。固定 Top-k 文本块检索容易丢失视觉和版面信息；固定 Top-k 页面检索虽然保留原始页面，但会把大量无关区域交给读者模型，形成覆盖率、噪声和推理成本之间的静态折中。

MAGE-RAG 将长文档多模态 QA 建模为 **budgeted evidence subgraph construction**。系统离线构建由页面节点与页内元素节点组成的多粒度证据图；在线阶段先通过 ColPali 页面级视觉检索获得入口页面，再由证据控制器根据当前问题、证据状态和历史轨迹迭代执行 `ActivatePage`、`ActivateNode`、`OpenNode`、`SearchEvidence` 和 `PruneNode`，最后把非剪枝证据渲染为结构化多模态 reader 输入。

## Highlights

| 亮点                                    | 说明                                                                                                               |
| --------------------------------------- | ------------------------------------------------------------------------------------------------------------------ |
| ✨**多粒度证据图**                | 同时建模 full-page 视觉上下文与页内段落、标题、表格、图片、图表等元素证据。                                        |
| 🧭**Query-time Evidence Control** | 在线控制器维护 `Inactive / Active / Opened / Pruned` 状态，按问题动态扩展和剪枝证据。                            |
| 🔗**结构与语义联合扩展**          | 支持 containment、reading order、layout adjacency、section hierarchy 和 semantic neighbor 等关系。                 |
| 🖼️**结构化多模态渲染**          | 将最终证据按页面层级组织为 XML，并对齐页面图像、局部节点图像或 bbox crop。                                         |
| 📊**统一实验协议**                | 在 LongDocURL 与 MMLongBench-Doc 上对 Direct MLLM、Text RAG、Page-level Visual RAG 和 Graph/Agentic RAG 进行比较。 |
| 🔍**可审计执行轨迹**              | 每次激活、打开、搜索、剪枝和停止决策都会记录 trace，便于复盘错误来源与证据路径。                                   |

## Framework Overview

<div align="center">
  <img src="assets/readme/framework.png" alt="MAGE-RAG framework" width="96%">
</div>

MAGE-RAG 的主流程由四个部分组成：

1. **Offline Multigranular Evidence Graph Construction**
   文档解析后构建页面节点和页内元素节点。节点保存摘要、文本或结构化内容、bbox、页面图像或局部图像引用；边表示 containment、reading order、layout、section hierarchy 和 semantic-neighbor 等关系。对应实现主要位于 `benchmarks/evidence_graph/`。
2. **Page-Level Initial Grounding**
   查询时先使用 ColPali 页面级视觉检索在当前文档允许页范围内选出初始 Top-k 页面，并通过 `ActivatePage` 加入工作证据状态。对应实现位于 `baselines/magerag/retrieval.py`。
3. **Online Evidence Controller**
   控制器把当前证据状态、候选动作和最近 trace 编成 XML，让 evaluator 选择有边际收益的激活动作，或请求打开节点、重新搜索、剪枝低价值证据、停止扩展。状态机与控制器实现位于 `baselines/magerag/state.py`、`baselines/magerag/evaluator.py` 和 `baselines/magerag/builder.py`。
4. **Structured Multimodal Reader Rendering**
   最终状态中的非剪枝页面和元素被渲染为 page-organized XML，并与页面截图、节点图像或裁剪区域交错输入 LVLM reader。对应实现位于 `baselines/magerag/renderer.py`。

> [!TIP]
> MAGE-RAG 的 `k` 不是最终 reader 上下文大小，而是证据图搜索的入口预算。最终上下文由后续图扩展、搜索、剪枝和 rendering 决定。

## Main Contributions

- **问题建模**：将长文档多模态 QA 中的证据组织问题形式化为预算约束下的 query-time evidence subgraph construction。
- **方法框架**：提出结合页面级视觉入口、多粒度文档证据图、在线证据控制器和结构化多模态渲染的 MAGE-RAG。
- **实验与分析**：在 LongDocURL 和 MMLongBench-Doc 上建立统一比较协议，并结合主结果、细粒度 breakdown、预算曲线、消融实验和 trace 统计分析证据组织行为。

## Experimental Results

主实验使用 **Qwen3-VL-8B-Instruct** 作为 reader，在 LongDocURL 与 MMLongBench-Doc 上比较四类方法：Direct MLLM、Text RAG、Page-level Visual RAG、Graph/Agentic RAG。

### Main Results

| Method             | Type                        |  LongDocURL Acc | MMLongBench-Doc Acc | MMLongBench-Doc F1 |
| ------------------ | --------------------------- | --------------: | ------------------: | -----------------: |
| Direct MLLM        | Direct MLLM                 |           50.07 |               43.05 |              43.63 |
| BM25               | Text RAG                    |           36.26 |               31.16 |              22.03 |
| ColBERTv2          | Text RAG                    |           31.74 |               30.56 |              20.43 |
| M3DocRAG           | Page-level Visual RAG       |           49.35 |               38.21 |              37.52 |
| EVisRAG            | Page-level Visual RAG       |           41.84 |               42.33 |              40.65 |
| G2Reader           | Graph/Agentic RAG           |           50.67 |               46.96 |              45.29 |
| MLDocRAG           | Graph/Agentic RAG           |           50.80 |               47.90 |                  - |
| **MAGE-RAG** | **Graph/Agentic RAG** | **52.75** |     **53.26** |    **51.19** |

### Fine-Grained Observations

- 在 **LongDocURL** 上，MAGE-RAG 达到 **52.75 overall accuracy**，并在 layout、figure、table、cross-element 等分组中取得最高或领先结果。
- 在 **MMLongBench-Doc** 上，MAGE-RAG 达到 **53.26 Acc / 51.19 F1**；cross-page accuracy 为 **34.70**，高于 M3DocRAG 的 24.87 和 EVisRAG 的 24.51。
- 预算实验显示，增加初始页面数、控制器轮数或每轮动作数并不总是单调提升，说明性能收益来自证据选择路径，而不是简单扩大上下文。

### Representative Ablations on MMLongBench-Doc

| Controller                     | Graph          |   Acc |    F1 | Single | Cross | Unanswerable |
| ------------------------------ | -------------- | ----: | ----: | -----: | ----: | -----------: |
| Top-k Page Only                | Full Graph     | 48.75 | 45.71 |  55.30 | 24.31 |        73.36 |
| Top-k Page with Node Rendering | Full Graph     | 49.71 | 47.38 |  57.09 | 26.42 |        71.31 |
| Graph Neighbor Expansion       | Full Graph     | 53.88 | 50.20 |  59.83 | 32.11 |        76.23 |
| Dynamic Controller No Search   | Full Graph     | 47.95 | 45.52 |  55.76 | 22.15 |        72.54 |
| Full                           | Page Only      | 52.36 | 49.35 |  58.62 | 31.70 |        72.13 |
| Full                           | Semantic Graph | 53.36 | 49.91 |  60.37 | 29.19 |        75.23 |
| Dynamic Controller No Prune    | Full Graph     | 52.99 | 50.26 |  60.64 | 31.45 |        70.48 |

完整实验表、预算曲线、trace 统计和 case study 可由 `analysis/` 中的脚本与 `results/` 中的实验产物复核；README 中只展示最核心的公开结果摘要。

## Repository Structure

```text
code/
├── main.py                         # Hydra 入口：prepare embedding cache -> run benchmark
├── configs/                        # benchmark、baseline、LiteLLM 等配置
├── baselines/
│   ├── magerag/                    # MAGE-RAG 在线证据控制、状态机、渲染器
│   ├── m3docrag.py                 # M3DocRAG baseline 接入
│   ├── evisrag.py                  # EVisRAG baseline 接入
│   ├── g2reader.py                 # G2Reader baseline 接入
│   └── bm25.py / colbertv2.py      # 文本检索 baseline
├── benchmarks/
│   ├── adapters.py                 # LongDocURL / MMLongBench 评分与样本处理
│   ├── runner.py                   # 统一运行、断点续跑、metrics 写出
│   ├── evidence_graph/             # 离线证据图节点、边、摘要、embedding 构建
│   ├── longdocurl/                 # LongDocURL 数据处理与脚本
│   └── mmlongbench/                # MMLongBench-Doc 数据处理与脚本
├── analysis/                       # 论文结果表、预算分析、trace/case 可视化
├── scripts/                        # 常用构图、服务、实验 sweep 脚本
├── tests/                          # pytest 单元测试与回归测试
└── results/                        # 已生成的代表性实验结果
```

README 使用的展示图片位于 `assets/readme/`。代码仓库保持自包含，图片、脚本与主要结果摘要均可在当前仓库内查看。

## Quick Start

> [!NOTE]
> MAGE-RAG 的完整评测依赖长文档数据、PDF 解析结果、ColPali embeddings、证据图 artifacts，以及 OpenAI-compatible LVLM 服务。仓库已经提供统一入口、配置和脚本；数据与模型权重请按照对应 benchmark 与模型许可自行准备。

### 1. Environment

推荐使用 Python 3.12 环境。当前实验脚本默认使用：

```bash
/root/autodl-tmp/conda/envs/logma-rag-py12/bin/python
```

如需使用其他环境，可通过 `PYTHON_BIN` 覆盖脚本中的 Python 路径。核心运行入口位于 `main.py`，配置由 Hydra 从 `configs/` 加载。

### 2. Services

MAGE-RAG 使用 OpenAI-compatible 接口调用 reader/evaluator。可通过 LiteLLM 启动本地代理：

```bash
bash scripts/serve_litellm.sh configs/litellm_config.yaml
```

在线 `SearchEvidence` 需要 ColPali query embedding 时，可启动 ColPali vLLM pooling 服务：

```bash
bash scripts/serve_colpali_vllm.sh 8k
```

### 3. Build Evidence Graphs

离线证据图由 `benchmarks/scripts/build_evidence_graphs.py` 构建，常用封装脚本如下：

```bash
bash scripts/run_build_longdocurl.sh
bash scripts/run_build_mmlongbench.sh
```

构图流程读取 PDF 解析结果，生成页面/元素节点，补充 LLM 摘要，构建结构边与语义边，并写出 `graph.json`、`nodes.jsonl`、`edges.jsonl` 等 artifacts。

### 4. Run MAGE-RAG

默认 Hydra 配置为 `benchmarks=longdocurl` 与 `baselines=magerag`：

```bash
/root/autodl-tmp/conda/envs/logma-rag-py12/bin/python main.py
```

运行 MMLongBench-Doc：

```bash
/root/autodl-tmp/conda/envs/logma-rag-py12/bin/python main.py benchmarks=mmlongbench baselines=magerag
```

运行 MAGE-RAG sweep：

```bash
bash scripts/run_magerag.sh
```

更细粒度的实验可直接使用 Hydra overrides：

```bash
/root/autodl-tmp/conda/envs/logma-rag-py12/bin/python main.py \
  benchmarks=mmlongbench \
  baselines=magerag \
  baselines.params.top_k=5 \
  baselines.controller.watchdog_iterations=10 \
  baselines.evaluator.max_selected_actions_per_iteration=5
```

### 5. Validate

核心测试使用 `pytest`：

```bash
/root/autodl-tmp/conda/envs/logma-rag-py12/bin/python -m pytest tests
```

与 MAGE-RAG、benchmark adapter 和证据图相关的窄测试：

```bash
/root/autodl-tmp/conda/envs/logma-rag-py12/bin/python -m pytest \
  tests/test_magerag.py \
  tests/test_evidence_graph_builder.py \
  tests/test_benchmark_adapters.py
```

## Usage

MAGE-RAG 的运行链路为：

```text
main.py
  -> benchmarks.wrapper.run_benchmark
  -> benchmarks.runner.run_benchmark_with_adapter
  -> baselines.wrapper.build_context_builder
  -> baselines.magerag.MAGERAGContextBuilder
  -> benchmarks.adapters.{LongDocURLAdapter,MMLongBenchAdapter}
```

关键配置位于 `configs/baselines/magerag.yaml`：

| Config                                           | 作用                                                               |
| ------------------------------------------------ | ------------------------------------------------------------------ |
| `params.top_k`                                 | 初始页面入口预算。                                                 |
| `controller.watchdog_iterations`               | 在线控制器最大迭代轮数。                                           |
| `evaluator.max_selected_actions_per_iteration` | 每轮最多执行的编号激活动作数。                                     |
| `graph.mode`                                   | 图结构模式，如 `full_graph`、`page_only`、`semantic_graph`。 |
| `reader.include_page_images`                   | 是否向 reader 输入页面图像。                                       |
| `trace.save_candidate_actions`                 | 是否保存候选动作，便于审计 trace。                                 |

主要输出包括 JSONL 预测文件、`.metrics.json` 指标文件，以及每条样本中的 `prepare_metadata.magerag` trace、graph stats、reader input summary 和 logical cost。

## Citation

如果本仓库对你的研究有帮助，请引用 MAGE-RAG。正式出版信息确定后，我们会更新 venue、pages、doi/url 等字段。

```bibtex
@misc{zuo2026magerag,
  title        = {MAGE-RAG: Multigranular Adaptive Graph Evidence for Agentic Multimodal RAG in Long-Document QA},
  author       = {Zuo, Yilong and Li, Xunkai and Yuan, Jing and Dai, Qiangqiang and Qin, Hongchao and Li, Ronghua},
  year         = {2026},
  note         = {Manuscript under review},
  howpublished = {\url{https://github.com/laonuo2004/MAGE-RAG}}
}
```

## License

本项目代码采用 [MIT License](LICENSE)。

## Acknowledgements

本项目的实验协议和实现参考并接入了多个长文档多模态 QA 与 RAG 方向的重要工作，包括 LongDocURL、MMLongBench-Doc、ColPali、M3DocRAG、VisRAG/EVisRAG、G2Reader、MLDocRAG、BM25 与 ColBERTv2。感谢这些数据集、模型和开源实现为长文档多模态 RAG 研究提供基础。
