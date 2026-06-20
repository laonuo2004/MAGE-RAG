<div align="center">
  <img src="assets/readme/banner.svg" alt="MAGE-RAG banner" width="100%">

<h1>MAGE-RAG</h1>

<p>
  <strong>Multigranular Adaptive Graph Evidence for Agentic Multimodal RAG in Long-Document QA</strong>
</p>

<p>
  <a href="#项目概览"><img alt="Task" src="https://img.shields.io/badge/Task-Long--Document%20Multimodal%20QA-2f6fed"></a>
  <a href="#方法概览"><img alt="Method" src="https://img.shields.io/badge/Method-Agentic%20Multimodal%20Graph%20RAG-16a34a"></a>
  <a href="LICENSE"><img alt="License" src="https://img.shields.io/badge/License-MIT-111827"></a>
</p>

<p>
  <a href="#主要实验结果"><img alt="LongDocURL" src="https://img.shields.io/badge/LongDocURL-52.75%20Acc-0f766e"></a>
  <a href="#主要实验结果"><img alt="MMLongBench-Doc" src="https://img.shields.io/badge/MMLongBench--Doc-53.26%20Acc%20%7C%2051.19%20F1-c2410c"></a>
</p>

<p>
  <a href="#项目概览">Overview</a> •
  <a href="#快速开始">Quick Start</a> •
  <a href="#数据与缓存准备">Data</a> •
  <a href="#主要实验结果">Results</a> •
  <a href="#引用">Citation</a>
</p>
</div>

---

## 项目概览

MAGE-RAG 是一个面向长 PDF 文档多模态问答的 Agentic Graph RAG 框架。

<div align="center">
  <img src="assets/readme/intro.png" alt="MAGE-RAG introduction" width="70%">
</div>

在长 PDF 文档多模态问答场景当中，答案的证据往往是稀疏且异构的，具体表现为：

1. **稀疏性**：答案证据可能只分布在少数一个或几个页面与页内元素中，而大部分页面和元素可能与问题无关；
2. **异构性**：答案证据所来源的模态多种多样，包括但不限于整体的页面布局与局部的文本段落、表格、图像等多种模态信息；

现有的 RAG 方法通常采用返回固定 Top-k 的检索广度，同时普遍采用 page-level 或 chunk/element-level 的单种检索粒度。固定的 Top-k 检索在 k 值较小时，容易遗漏证据，而当 k 值较大时，又会引入不必要的噪声与成本；chunk/element-level 尽管返回的证据更为细节具体，但是缺乏对全局的把握，且检索的召回率与精确率更低，而 page-level 的检索虽然保留了全局信息，且在检索的召回率与精确率上表现较好，但会把大量无关噪声内容交给 Reader 模型，增大其理解负担。与此同时，当前的众多 RAG 方法大多提供 one-shot context，Reader 模型只能看到零散的证据，而无法得知证据之间的逻辑与扩展关系。

MAGE-RAG 通过有效的状态设计与自适应的证据控制、扩展机制，针对性地解决了上述 RAG 的固有缺陷。MAGE-RAG 离线建立文档级别的证据图，并在在线问答阶段，使用 page-level 检索找到高准确率的证据图入口，随后借助 Controller 选择性地激活证据图当中的部分细粒度节点，最后将被激活的节点包装为结构化的证据输入到 Reader 模型当中，有效地实现了自适应的证据广度、多粒度的证据粒度，以及可归因、可追溯的证据关系。

本仓库包含 MAGE-RAG 的核心实现、LongDocURL 与 MMLongBench-Doc 评测适配、多个 baseline 接入、证据图构建脚本、结果分析脚本，以及论文中主要表格和图的生成逻辑。

## 方法概览

<div align="center">
  <img src="assets/readme/framework.png" alt="MAGE-RAG framework" width="96%">
</div>

MAGE-RAG 的运行流程由四个部分组成：

1. **离线多粒度证据图构建**  
   文档解析后构建页面节点和页内元素节点。节点保存摘要、文本或结构化内容、bbox、页面图像或局部图像引用；边表示 containment、reading order、layout adjacency、section hierarchy 和 semantic neighbor 等关系。实现主要位于 `benchmarks/evidence_graph/`。

2. **页面级初始 grounding**  
   查询时先使用 ColPali 页面级视觉检索，在当前文档允许页范围内选出初始 Top-k 页面，并通过 `ActivatePage` 加入证据状态。实现主要位于 `baselines/magerag/retrieval.py`。

3. **在线证据控制器**  
   控制器维护 `Inactive / Active / Opened / Pruned` 状态，把当前证据、候选动作和最近 trace 编成 XML，由 evaluator 选择 `ActivateNode`、`OpenNode`、`SearchEvidence`、`PruneNode` 或停止扩展。实现主要位于 `baselines/magerag/state.py`、`baselines/magerag/evaluator.py` 和 `baselines/magerag/builder.py`。

4. **结构化多模态 reader 渲染**  
   最终状态中的非剪枝页面和元素被渲染为 page-organized XML，并与页面截图、局部节点图像或 bbox crop 一起输入 LVLM reader。实现主要位于 `baselines/magerag/renderer.py`。

> [!NOTE]
> MAGE-RAG 中的 `top_k` 是证据图搜索的入口页面预算，不等价于最终 reader 上下文大小。最终输入由后续图扩展、在线搜索、剪枝和渲染共同决定。

## 主要特点

| 特点 | 说明 |
| --- | --- |
| 多粒度证据图 | 同时建模 full-page 视觉上下文和页内段落、标题、表格、图片、图表等元素证据。 |
| Query-time evidence control | 根据问题、证据状态和历史 trace 动态扩展、打开、搜索和剪枝证据。 |
| 结构与语义联合扩展 | 支持 containment、reading order、layout、section hierarchy 和 semantic neighbor 等关系。 |
| 结构化多模态输入 | 将最终证据组织为 XML，并对齐页面图像、局部节点图像或 bbox crop。 |
| 统一评测协议 | 在 LongDocURL 和 MMLongBench-Doc 上比较 Direct MLLM、Text RAG、Page-level Visual RAG 和 Graph/Agentic RAG。 |
| 可审计 trace | 每次激活、打开、搜索、剪枝和停止决策都会记录，便于分析错误来源和证据路径。 |

## 仓库结构

```text
MAGE-RAG/
├── main.py                         # Hydra 入口：准备 embedding cache 后运行 benchmark
├── configs/                        # 默认配置、benchmark 配置、baseline 配置、LiteLLM 配置
├── baselines/
│   ├── magerag/                    # MAGE-RAG 状态机、控制器、检索、渲染和图存储
│   ├── bm25.py                     # BM25 文本检索 baseline
│   ├── colbertv2.py                # ColBERTv2 文本检索 baseline
│   ├── image.py                    # 页面级视觉输入 baseline
│   ├── m3docrag.py                 # M3DocRAG 接入
│   ├── evisrag.py                  # EVisRAG 接入
│   └── g2reader.py                 # G2Reader 接入
├── benchmarks/
│   ├── adapters.py                 # LongDocURL / MMLongBench-Doc 样本处理和评分
│   ├── runner.py                   # 统一运行、断点续跑、预测和 metrics 写出
│   ├── wrapper.py                  # benchmark 路由
│   ├── evidence_graph/             # 证据图节点、边、摘要、语义边和 embedding 构建
│   ├── longdocurl/                 # LongDocURL 数据说明、预处理和脚本
│   ├── mmlongbench/                # MMLongBench-Doc 数据说明、样例数据和脚本
│   └── utils/                      # 数据路径、embedding cache、PDF 处理和结果工具
├── analysis/                       # 论文结果表、breakdown、预算、trace、case study 分析
├── scripts/                        # 常用服务、构图、baseline 和 sweep 脚本
├── assets/readme/                  # README 展示图片
└── LICENSE
```

## 如何复现

### 复现环境

// 这里写当前服务器的运行环境+重要版本信息

Python==3.12.13
Pytorch==2.10.0+cu128
vLLM==0.19.1
MinerU: VLM 3.1.8
Colpali: v1.3

### 


## 引用

如果本仓库对你的研究有帮助，请引用 MAGE-RAG。

```bibtex
@article{zuo2026mage,
  title={MAGE-RAG: Multigranular Adaptive Graph Evidence for Agentic Multimodal RAG in Long-Document QA},
  author={Zuo, Yilong and Li, Xunkai and Yuan, Jing and Dai, Qiangqiang and Qin, Hongchao and Li, Ronghua},
  journal={arXiv preprint arXiv:2606.15906},
  year={2026}
}
```

## License

本项目代码采用 [MIT License](LICENSE)。

## Acknowledgements

该项目依赖于以下项目，感谢！

- MinerU：提供 PDF 文档解析服务 (https://github.com/opendatalab/MinerU.git)
- Colpali：用于页面级视觉检索 (https://huggingface.co/vidore/colpali-v1.3-hf)