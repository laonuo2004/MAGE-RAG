<p align="right">
  <a href="README-zh.md">中文</a> | <strong>English</strong>
</p>

<div align="center">
  <img src="assets/readme/banner.svg" alt="MAGE-RAG banner" width="100%">

<h1>MAGE-RAG</h1>

<p>
  <strong>Multigranular Adaptive Graph Evidence for Agentic Multimodal RAG in Long-Document QA</strong>
</p>

<p>
  <a href="https://arxiv.org/abs/2606.15906"><img alt="arXiv" src="https://img.shields.io/badge/arXiv-2606.15906-b31b1b"></a>
  <a href="#overview"><img alt="Task" src="https://img.shields.io/badge/Task-Long--Document%20Multimodal%20QA-2f6fed"></a>
  <a href="#method-overview"><img alt="Method" src="https://img.shields.io/badge/Method-Agentic%20Multimodal%20Graph%20RAG-16a34a"></a>
  <a href="LICENSE"><img alt="License" src="https://img.shields.io/badge/License-MIT-111827"></a>
</p>

<p>
  <a href="#main-results"><img alt="MMLongBench-Doc" src="https://img.shields.io/badge/MMLongBench--Doc-53.26%20Acc%20%7C%2051.19%20F1-c2410c"></a>
  <a href="#main-results"><img alt="LongDocURL" src="https://img.shields.io/badge/LongDocURL-52.75%20Acc-0f766e"></a>
</p>

<p>
  <a href="#overview">Overview</a> •
  <a href="#method-overview">Method</a> •
  <a href="#reproduction">Reproduction</a> •
  <a href="#data-preparation">Data</a> •
  <a href="#citation">Citation</a>
</p>
</div>

---

# Overview

MAGE-RAG is an agentic graph RAG framework for multimodal question answering over long PDF documents.

<div align="center">
  <img src="assets/readme/intro.png" alt="MAGE-RAG introduction" width="60%">
</div>

Evidence in long-document multimodal QA is often both sparse and heterogeneous:

1. **Sparse evidence**: the information needed to answer a question may appear on only one or a few pages, or within a small number of page elements, while most of the document is irrelevant.
2. **Heterogeneous evidence**: useful evidence may come from full-page layouts, local text passages, tables, images, and other modalities.

Most existing RAG systems retrieve a fixed Top-k number of items and operate at a single granularity, typically pages or chunks/elements. A small k can miss relevant evidence, whereas a large k introduces unnecessary noise and cost. Chunk- or element-level retrieval provides focused details but has less global context and can suffer from lower retrieval recall and precision. Page-level retrieval preserves the document's global visual structure and often provides stronger recall and precision, but it also sends substantial irrelevant content to the reader. Moreover, many RAG systems provide a one-shot context: the reader sees isolated evidence without an explicit account of how evidence items are related or how they were discovered.

MAGE-RAG addresses these limitations through explicit evidence states and adaptive evidence control. It constructs a document-level evidence graph offline. At query time, page-level retrieval identifies reliable entry points into the graph, after which a Controller selectively activates fine-grained evidence nodes. The activated nodes are finally organized into structured evidence for the Reader. This design supports adaptive evidence breadth, multiple evidence granularities, and evidence relationships that remain attributable and traceable.

This repository contains the core MAGE-RAG implementation, evaluation adapters for MMLongBench-Doc and LongDocURL, integrations for multiple baselines, evidence-graph construction scripts, result-analysis utilities, and the code used to generate the paper's main tables and figures.

# Method Overview

<div align="center">
  <img src="assets/readme/framework.png" alt="MAGE-RAG framework" width="96%">
</div>

MAGE-RAG consists of four main stages:

1. **Offline multigranular evidence-graph construction**  
   After document parsing, MAGE-RAG creates page nodes and within-page element nodes. Nodes store summaries, text or structured content, bounding boxes, and references to page or local images. Edges represent containment, reading order, layout adjacency, section hierarchy, semantic neighbors, and related document structure. The main implementation is under `benchmarks/evidence_graph/`.

2. **Page-level initial grounding**  
   At query time, ColPali visual retrieval selects the initial Top-k pages from the allowed page range of the current document. These pages enter the evidence state through `ActivatePage`. The implementation is primarily in `baselines/magerag/retrieval.py`.

3. **Online evidence controller**  
   The controller maintains `Inactive / Active / Opened / Pruned` states. It serializes the current evidence, candidate actions, and recent trace into XML, then asks an evaluator to choose among `ActivateNode`, `OpenNode`, `SearchEvidence`, `PruneNode`, or stopping. The main implementation is in `baselines/magerag/state.py`, `baselines/magerag/evaluator.py`, and `baselines/magerag/builder.py`.

4. **Structured multimodal reader rendering**  
   Non-pruned pages and elements in the final state are rendered as page-organized XML and paired with page screenshots, local node images, or bounding-box crops for the LVLM reader. The implementation is primarily in `baselines/magerag/renderer.py`.

> [!NOTE]
> In MAGE-RAG, `top_k` controls the number of entry pages for evidence-graph search. It is not the final reader-context size. The final input is determined jointly by graph expansion, online search, pruning, and rendering.

# Key Features

| Feature | Description |
| --- | --- |
| Multigranular evidence graph | Jointly models full-page visual context and within-page evidence such as paragraphs, headings, tables, images, and charts. |
| Query-time evidence control | Dynamically expands, opens, searches, and prunes evidence according to the question, evidence state, and action history. |
| Structural and semantic expansion | Supports containment, reading order, layout, section hierarchy, semantic-neighbor relations, and other graph connections. |
| Structured multimodal input | Organizes final evidence as XML aligned with page images, local node images, or bounding-box crops. |
| Unified evaluation protocol | Compares Direct MLLM, Text RAG, Page-level Visual RAG, and Graph/Agentic RAG on MMLongBench-Doc and LongDocURL. |
| Auditable traces | Records activation, opening, search, pruning, and stopping decisions for evidence-path inspection and error analysis. |

# Repository Structure

```text
MAGE-RAG/
├── main.py                         # Hydra entrypoint: prepare embedding caches and run benchmarks
├── configs/                        # Default, benchmark, baseline, and LiteLLM configurations
├── baselines/
│   ├── magerag/                    # MAGE-RAG state machine, controller, retrieval, rendering, and graph store
│   ├── bm25.py                     # BM25 text-retrieval baseline
│   ├── colbertv2.py                # ColBERTv2 text-retrieval baseline
│   ├── image.py                    # Page-level visual-input baseline
│   ├── m3docrag.py                 # M3DocRAG integration
│   ├── evisrag.py                  # EVisRAG integration
│   └── g2reader.py                 # G2Reader integration
├── benchmarks/
│   ├── adapters.py                 # MMLongBench-Doc / LongDocURL sample processing and scoring
│   ├── runner.py                   # Unified execution, resumption, prediction, and metric output
│   ├── wrapper.py                  # Benchmark routing
│   ├── evidence_graph/             # Nodes, edges, summaries, semantic edges, and graph embeddings
│   ├── mmlongbench/                # MMLongBench-Doc data notes, samples, and scripts
│   ├── longdocurl/                 # LongDocURL data notes, preprocessing, and scripts
│   └── utils/                      # Data paths, embedding caches, PDF processing, and result utilities
├── analysis/                       # Paper tables, breakdowns, budgets, traces, and case-study analysis
├── scripts/                        # Service, graph-building, baseline, and sweep scripts
├── assets/readme/                  # README assets
└── LICENSE
```

# Reproduction

## Environment

### Reference Experimental Environment

The paper experiments and repository validation were conducted with the following environment:

| Component | Version or configuration |
| --- | --- |
| OS | Ubuntu 22.04.5 LTS, Linux 5.15, x86_64 |
| GPU | 2 × NVIDIA RTX PRO 6000 Blackwell, 96 GiB / GPU |
| NVIDIA Driver | 590.44.01 |
| CUDA | PyTorch CUDA runtime 12.8 |
| Python | 3.12.13 |
| PyTorch | 2.10.0+cu128 |
| TorchVision | 0.25.0 |
| Transformers | 5.5.4 |
| vLLM | 0.19.1 |
| PyMuPDF | 1.27.2.2 |
| MinerU | VLM 3.1.8 |
| ColPali | `vidore/colpali-v1.3-hf` |
| Reader / controller | `Qwen/Qwen3-VL-8B-Instruct` |

### Create the Python Environment and Configure Environment Variables

Create the Python environment:

```bash
uv sync --frozen
```

Copy the `.env` template:

```bash
cp .env.example .env
```

Then edit `.env`. A MinerU API key can be obtained from the [MinerU website](https://mineru.net/apiManage/token).

## Model Services

MAGE-RAG uses two model services:

1. A ColPali pooling service for page, question, and graph-node embeddings. Its default endpoint is `http://127.0.0.1:8020`.
2. An LVLM service for node summarization, evidence control, and final answer generation. Its default OpenAI-compatible endpoint is `http://127.0.0.1:4000/v1`.

Download [Qwen3-VL-8B-Instruct](https://huggingface.co/Qwen/Qwen3-VL-8B-Instruct) and [ColPali v1.3](https://huggingface.co/vidore/colpali-v1.3-hf), then start the two services:

```bash
# Terminal 1: Qwen3-VL reader/controller on port 4000
env \
  MODEL_NAME=/path/to/Qwen3-VL-8B-Instruct \
  VLLM_BIN="$PWD/.venv/bin/vllm" \
  CUDA_VISIBLE_DEVICES=0 \
  PORT=4000 \
  bash scripts/serve_qwen3_vl_vllm.sh 128k

# Terminal 2: ColPali pooling service on its default port 8020
env \
  MODEL_NAME=/path/to/colpali-v1.3-hf \
  SERVED_MODEL_NAME=colpali-v1.3 \
  VLLM_BIN="$PWD/.venv/bin/vllm" \
  CUDA_VISIBLE_DEVICES=1 \
  bash scripts/serve_colpali_vllm.sh 8k
```

We recommend at least 48 GB of GPU memory for Qwen3-VL-8B-Instruct with a 128k context window, and at least 10 GB for ColPali with an 8k context window.

If either service fails to start, adjust the deployment settings as needed:

- If additional GPU memory is available, add `GPU_MEMORY_UTILIZATION=0.9` and adjust the parameters as needed.
- On a multi-GPU machine, change `CUDA_VISIBLE_DEVICES` and add `TENSOR_PARALLEL_SIZE=2`. The tensor-parallel size should match the number of selected GPUs and must be a multiple of 2.
- Reduce deployment parameters such as `max_model_len`, `max-num-seqs`, or `max-num-batched-tokens`.
- See the [vLLM engine arguments documentation](https://docs.vllm.ai/en/stable/configuration/engine_args) for additional settings.

## Data Preparation

The original QA files and PDFs are managed with Git LFS. After cloning the repository, install Git LFS and fetch the data:

```bash
sudo apt-get update
sudo apt-get install -y git-lfs
cd MAGE-RAG
git lfs install
git lfs pull
```

After the download completes, the repository should contain:

```text
benchmarks/
├── mmlongbench/data/raw/
│   ├── samples.json                     # 1,091 QA pairs over 135 documents
│   └── documents/<doc_id>.pdf           # Original PDFs
└── longdocurl/data/raw/
    ├── LongDocURL.jsonl                 # 2,325 QA pairs over 396 documents
    └── pdfs/4000-4999/<doc_no>.pdf      # Original PDFs
```

### 1. Render PDF Pages as PNG Images

Render the MMLongBench-Doc PDFs:

```bash
uv run python benchmarks/scripts/preprocess_documents.py \
  --benchmark mmlongbench \
  --workers 8
```

Following the benchmark protocol, MMLongBench-Doc renders at most the first 120 pages and uses 1-based page numbering.

**Expected output:**

```text
benchmarks/mmlongbench/data/processed/pdf_pngs/
└── <doc_key>/
    └── page_<1-based page number, 4 digits>_dpi144.png
```

The output covers 135 documents. Each document produces `min(total PDF pages, 120)` PNG images at 144 DPI.

---

Render the LongDocURL PDFs:

```bash
uv run python benchmarks/scripts/preprocess_documents.py \
  --benchmark longdocurl \
  --workers 8
```

LongDocURL renders each complete PDF and uses 0-based page numbering.

**Expected output:**

```text
benchmarks/longdocurl/data/processed/pdf_pngs/4000-4999/
└── <first four digits of doc_no>/
    └── <doc_no>_<0-based page index>.png
```

The output contains every page from all 396 documents, with one 144-DPI PNG per PDF page.

Verify that the PNG outputs for both benchmarks are complete and have the expected dimensions:

```bash
for benchmark in mmlongbench longdocurl; do
  uv run python benchmarks/scripts/verify_artifacts.py \
    --benchmark "$benchmark" \
    --stage png
done
```

A successful verification ends with:

```text
Verified mmlongbench png: 135 documents (5784 pages).
Verified longdocurl png: 396 documents (33913 pages).
```

### 2. Parse PDFs with MinerU

```bash
uv run --env-file .env python benchmarks/scripts/extract_mineru.py \
  --benchmark mmlongbench

uv run --env-file .env python benchmarks/scripts/extract_mineru.py \
  --benchmark longdocurl
```

**Expected output:**

```text
benchmarks/mmlongbench/data/processed/pdfs_mineru/<doc_key>/
├── layout.json
├── *_content_list_v2.json
└── images/                       # Created when the document contains extracted images

benchmarks/longdocurl/data/processed/pdfs_mineru/4000-4999/<doc_no>/
├── layout.json
├── *_content_list_v2.json
└── images/                       # Created when the document contains extracted images
```

`layout.json` stores the page-level layout analysis. `*_content_list_v2.json` stores page-organized elements such as text, headings, tables, and images. MMLongBench-Doc and LongDocURL should produce 135 and 396 document directories, respectively. The `images/` directory is present only when images were extracted from a document.

Verify the MinerU outputs:

```bash
for benchmark in mmlongbench longdocurl; do
  uv run python benchmarks/scripts/verify_artifacts.py \
    --benchmark "$benchmark" \
    --stage mineru
done
```

A successful verification ends with:

```text
Verified mmlongbench mineru: 135 documents.
Verified longdocurl mineru: 396 documents.
```

### 3. Generate ColPali Page and Question Embeddings

Keep the ColPali service on port 8020 running, then execute:

```bash
uv run python benchmarks/scripts/generate_colpali_embeddings.py \
  --benchmark mmlongbench

uv run python benchmarks/scripts/generate_colpali_embeddings.py \
  --benchmark longdocurl
```

**Expected output:**

```text
benchmarks/mmlongbench/data/cache/colpali/
├── pdf_embeddings/
│   ├── <doc_key>.safetensors
│   └── manifest.jsonl
└── question_embeddings/
    ├── <question_id>.safetensors
    └── manifest.jsonl

benchmarks/longdocurl/data/cache/colpali/
├── pdf_embeddings/4000-4999/
│   ├── <doc_no>.safetensors
│   └── manifest.jsonl
└── question_embeddings/
    ├── <question_id>.safetensors
    └── manifest.jsonl
```

Each PDF embedding file contains an `embeddings` tensor with shape `[number of pages, page tokens, embedding dimension]`. Each question file contains a `query_embedding` tensor with shape `[query tokens, embedding dimension]`. The `manifest.jsonl` files record input paths, output paths, models, and processing status. A complete run should produce 135 PDF embeddings and 1,091 question embeddings for MMLongBench-Doc, and 396 PDF embeddings and 2,325 question embeddings for LongDocURL.

Verify the PDF and question embeddings:

```bash
for benchmark in mmlongbench longdocurl; do
  uv run python benchmarks/scripts/verify_artifacts.py \
    --benchmark "$benchmark" \
    --stage colpali
done
```

A successful verification ends with:

```text
Verified mmlongbench colpali: 1226 artifacts (135 documents, 1091 questions).
Verified longdocurl colpali: 2721 artifacts (396 documents, 2325 questions).
```

### 4. Build Evidence Graphs

Graph construction generates LLM abstracts, element-level ColPali embeddings, structural edges, layout edges, and semantic edges. This stage requires both the ColPali service on port 8020 and the OpenAI-compatible Qwen3-VL service on port 4000:

```bash
uv run python benchmarks/scripts/build_evidence_graphs.py \
  --benchmark mmlongbench \
  --workers 16 \
  --abstract-processor-path /path/to/Qwen3-VL-8B-Instruct \
  --abstract-context-window 131072

uv run python benchmarks/scripts/build_evidence_graphs.py \
  --benchmark longdocurl \
  --workers 16 \
  --abstract-processor-path /path/to/Qwen3-VL-8B-Instruct \
  --abstract-context-window 131072
```

Adjust `--workers` according to the throughput of the model services. `--abstract-processor-path` should point to a local Qwen3-VL processor used for context-budget validation.

**Expected output:**

```text
benchmarks/mmlongbench/data/processed/evidence_graphs/<doc_key>/
├── graph.json
├── nodes.jsonl
└── edges.jsonl

benchmarks/mmlongbench/data/cache/colpali/node_embeddings/<doc_key>/
└── *.safetensors

benchmarks/longdocurl/data/processed/evidence_graphs/4000-4999/<doc_no>/
├── graph.json
├── nodes.jsonl
└── edges.jsonl

benchmarks/longdocurl/data/cache/colpali/node_embeddings/4000-4999/<doc_no>/
└── *.safetensors
```

`graph.json` stores document metadata, graph-building settings, and node/edge counts. `nodes.jsonl` stores page and within-page element nodes, while `edges.jsonl` stores structural, layout, and semantic relations. Each non-page node has a corresponding `.safetensors` file containing an `embedding` tensor. Page nodes reuse the PDF page embeddings produced in Step 3. A complete build should produce 135 evidence-graph directories for MMLongBench-Doc and 396 for LongDocURL.

Verify the evidence graphs and node embeddings:

```bash
for benchmark in mmlongbench longdocurl; do
  uv run python benchmarks/scripts/verify_artifacts.py \
    --benchmark "$benchmark" \
    --stage graph
done
```

A successful verification ends with:

```text
Verified mmlongbench graph: 135 graphs.
Verified longdocurl graph: 396 graphs.
```

## Online QA and Evaluation

Run both benchmarks:

```bash
uv run python main.py --multirun \
  benchmarks=mmlongbench,longdocurl
```

The runner supports resuming interrupted experiments. Per-sample predictions are written to `results/<benchmark>/magerag/*.jsonl`, aggregate metrics are written to matching `*.metrics.json` files, and Hydra configurations and logs are stored under `outputs/`.

# Citation

If you find this repository useful, please cite [MAGE-RAG](https://arxiv.org/abs/2606.15906).

```bibtex
@article{zuo2026mage,
  title={MAGE-RAG: Multigranular Adaptive Graph Evidence for Agentic Multimodal RAG in Long-Document QA},
  author={Zuo, Yilong and Li, Xunkai and Yuan, Jing and Dai, Qiangqiang and Qin, Hongchao and Li, Ronghua},
  journal={arXiv preprint arXiv:2606.15906},
  year={2026}
}
```

# License

This project is released under the [MIT License](LICENSE).

# Acknowledgements

This project builds on the following projects:

- MinerU: PDF document parsing (https://mineru.net/)
- vLLM: model serving and inference (https://vllm.ai/)
- Qwen3-VL-8B-Instruct: Reader and Controller model (https://huggingface.co/Qwen/Qwen3-VL-8B-Instruct)
- ColPali: page-level visual retrieval (https://huggingface.co/vidore/colpali-v1.3-hf)
