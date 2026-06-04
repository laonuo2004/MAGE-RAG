import json
import time
from pathlib import Path

import pytest
import torch
from safetensors.torch import load_file, save_file

from benchmarks.scripts import build_evidence_graphs
from benchmarks.evidence_graph.builder import EvidenceGraphBuildConfig, build_document_graph
from benchmarks.evidence_graph.edges import build_structural_edges
from benchmarks.evidence_graph.schema import BuildResult, EvidenceNode
from benchmarks.evidence_graph.content import concatenate_spans
from benchmarks.evidence_graph.semantic import (
    _batched_maxsim_score_matrix,
    _batched_maxsim_scores,
    build_semantic_edges,
    maxsim_score,
)
from benchmarks.evidence_graph.embeddings import _encode_image_path_once, _node_embedding_text, materialize_node_embeddings
from benchmarks.evidence_graph.summaries import _node_prompt, _summarize_node
from benchmarks.evidence_graph.paths import GraphPathContext
from benchmarks.evidence_graph.writer import load_graph_artifacts


def _write_json(path: Path, payload):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")


def _fixture_doc(tmp_path: Path) -> GraphPathContext:
    mineru_dir = tmp_path / "mineru" / "sample-doc"
    png_dir = tmp_path / "pngs" / "sample-doc"
    page_embedding_dir = tmp_path / "page_embeddings"
    graph_root = tmp_path / "graphs"
    node_embedding_root = tmp_path / "node_embeddings"

    png_dir.mkdir(parents=True)
    for page_number in (1, 2):
        (png_dir / f"page_{page_number:04d}_dpi144.png").write_bytes(b"fake-png")

    page_embedding_dir.mkdir(parents=True)
    save_file({"embeddings": torch.ones((2, 2, 8), dtype=torch.bfloat16)}, page_embedding_dir / "sample-doc.safetensors")

    _write_json(
        mineru_dir / "layout.json",
        {
            "pdf_info": [
                {"page_idx": 0, "page_size": [100, 200]},
                {"page_idx": 1, "page_size": [100, 200]},
            ]
        },
    )
    _write_json(
        mineru_dir / "sample_content_list_v2.json",
        [
            [
                {
                    "type": "title",
                    "content": {"title_content": [{"type": "text", "content": "Section 1"}], "level": 1},
                    "bbox": [10, 10, 60, 20],
                },
                {
                    "type": "paragraph",
                    "content": {
                        "paragraph_content": [
                            {"type": "text", "content": "See Figure "},
                            {"type": "equation_inline", "content": "1"},
                            {"type": "text", "content": " for details."},
                        ]
                    },
                    "bbox": [10, 30, 80, 50],
                },
                {
                    "type": "page_number",
                    "content": {"page_number_content": [{"type": "text", "content": "1"}]},
                    "bbox": [45, 190, 55, 198],
                },
            ],
            [
                {
                    "type": "image",
                    "content": {
                        "image_source": {"path": "images/figure1.jpg"},
                        "content": "diagram text",
                        "image_caption": [{"type": "text", "content": "Figure 1: Pipeline"}],
                        "image_footnote": [],
                    },
                    "bbox": [10, 10, 90, 90],
                },
                {
                    "type": "table",
                    "content": {
                        "image_source": {"path": "images/table1.jpg"},
                        "table_caption": [{"type": "text", "content": "Table 1: Scores"}],
                        "table_footnote": [],
                        "html": "<table><tr><td>A</td></tr></table>",
                        "table_type": "simple_table",
                        "table_nest_level": 1,
                    },
                    "bbox": [10, 100, 90, 160],
                },
            ],
        ],
    )

    return GraphPathContext(
        benchmark_name="mmlongbench",
        doc_id="sample-doc.pdf",
        doc_key="sample-doc",
        mineru_dir=mineru_dir,
        page_image_paths={
            0: png_dir / "page_0001_dpi144.png",
            1: png_dir / "page_0002_dpi144.png",
        },
        page_embedding_path=page_embedding_dir / "sample-doc.safetensors",
        graph_dir=graph_root / "sample-doc",
        node_embedding_dir=node_embedding_root / "sample-doc",
    )


def test_concatenate_spans_wraps_inline_equations():
    spans = [
        {"type": "text", "content": "Unit A on "},
        {"type": "equation_inline", "content": "15^{\\text{th}}"},
        {"type": "text", "content": " Floor"},
    ]

    assert concatenate_spans(spans) == "Unit A on $15^{\\text{th}}$ Floor"


def test_node_prompt_focuses_on_semantic_content_without_structural_metadata():
    node = EvidenceNode(
        id="doc:page:3:block:4:paragraph",
        type="paragraph",
        doc_id="doc.pdf",
        page_index=3,
        index=4,
        bbox=[1, 2, 3, 4],
        fields={"paragraph": "Revenue increased because demand recovered."},
    )

    prompt = _node_prompt(node, "Revenue increased because demand recovered.")

    assert "Revenue increased because demand recovered." in prompt
    assert "semantic" in prompt.lower()
    assert "Document id" not in prompt
    assert "Page index" not in prompt
    assert "Element index" not in prompt
    assert "BBox" not in prompt
    assert "Node type" not in prompt
    assert "doc.pdf" not in prompt
    assert "<prompt>" in prompt
    assert "<source_content>" in prompt
    assert "Use only facts present in <source_content>" in prompt
    assert "Do not infer" in prompt
    assert "Do not exceed the source content length" in prompt
    assert "If the content is only a label" in prompt
    assert "Return only the abstract text" in prompt


def test_node_prompt_restricts_formula_explanations_to_local_context():
    node = EvidenceNode(
        id="doc:page:2:block:4:equation_interline",
        type="equation_interline",
        doc_id="doc.pdf",
        page_index=2,
        fields={"latex": "S_{wp}=\\frac{\\sum I_l(i,j)}{|C_{wp}|}"},
    )

    prompt = _node_prompt(node, "S_{wp}=\\frac{\\sum I_l(i,j)}{|C_{wp}|}")

    assert "Use nearby textual context" in prompt
    assert "Do not guess a domain" in prompt
    assert "image processing" in prompt
    assert "Variable names alone are not evidence of a domain" in prompt
    assert "Use one compact sentence" in prompt


def test_visual_node_prompt_allows_visual_evidence_but_rejects_external_facts():
    node = EvidenceNode(
        id="doc:page:1:block:0:image",
        type="image",
        doc_id="doc.pdf",
        page_index=1,
        fields={"caption": "Figure 1: Overview of ProgramFC."},
    )

    prompt = _node_prompt(node, "Figure 1: Overview of ProgramFC.")

    assert "may exceed extracted text" in prompt
    assert "Do not add outside facts" in prompt
    assert "born in" in prompt
    assert "visible image structure" in prompt
    assert "describe only its visible role or appearance" in prompt


def test_node_prompt_uses_type_specific_length_policy_for_page_and_list():
    page_node = EvidenceNode(
        id="doc:page:0",
        type="page",
        doc_id="doc.pdf",
        page_index=0,
    )
    list_node = EvidenceNode(
        id="doc:page:0:block:1:list",
        type="list",
        doc_id="doc.pdf",
        page_index=0,
    )

    page_prompt = _node_prompt(page_node, "Long page text about revenue and demand." * 20)
    list_prompt = _node_prompt(list_node, "First item about revenue. Second item about demand." * 3)

    assert "Use 2-4 compact sentences" in page_prompt
    assert "much shorter than the page content" in page_prompt
    assert "Use 1-2 compact sentences" in list_prompt
    assert "Do not enumerate every item unless the list has three or fewer items" in list_prompt


def test_summarize_node_allows_large_outputs():
    class FakeCompletions:
        def __init__(self):
            self.kwargs = None

        def create(self, **kwargs):
            self.kwargs = kwargs
            message = type("Message", (), {"content": "semantic summary"})()
            choice = type("Choice", (), {"message": message})()
            return type("Completion", (), {"choices": [choice]})()

    class FakeClient:
        def __init__(self):
            self.chat = type("Chat", (), {"completions": FakeCompletions()})()

    client = FakeClient()
    node = EvidenceNode(id="n", type="paragraph", doc_id="d", page_index=0)

    summary = _summarize_node(client, "model", node, "important content")

    assert summary == "semantic summary"
    assert client.chat.completions.kwargs["max_tokens"] == 4096


def test_summarize_page_node_sends_text_and_page_image(tmp_path):
    class FakeCompletions:
        def __init__(self):
            self.kwargs = None

        def create(self, **kwargs):
            self.kwargs = kwargs
            message = type("Message", (), {"content": "page semantic summary"})()
            choice = type("Choice", (), {"message": message})()
            return type("Completion", (), {"choices": [choice]})()

    class FakeClient:
        def __init__(self):
            self.chat = type("Chat", (), {"completions": FakeCompletions()})()

    image_path = tmp_path / "page.png"
    image_path.write_bytes(b"fake-page-image")
    client = FakeClient()
    node = EvidenceNode(
        id="doc:page:0",
        type="page",
        doc_id="doc.pdf",
        page_index=0,
        fields={"image_path": str(image_path)},
    )

    summary = _summarize_node(client, "model", node, "important page text")

    content = client.chat.completions.kwargs["messages"][0]["content"]
    assert summary == "page semantic summary"
    assert content[0]["type"] == "image_url"
    assert content[0]["image_url"]["url"].startswith("data:image/png;base64,")
    assert content[1]["type"] == "text"
    assert "important page text" in content[1]["text"]


def test_summarize_node_truncates_source_content_before_request(monkeypatch, caplog):
    class FakeProcessor:
        def apply_chat_template(self, messages, tokenize=False, add_generation_prompt=True):
            parts = []
            for item in messages[0]["content"]:
                if item["type"] == "text":
                    parts.append(item["text"])
                elif item["type"] == "image":
                    parts.append("<image>")
            return "\n".join(parts)

        def __call__(self, *, text, images=None, return_tensors=None):
            token_count = len(text[0].split()) + 100 * len(images or [])
            return {"input_ids": [list(range(token_count))]}

    class FakeCompletions:
        def __init__(self):
            self.kwargs = None

        def create(self, **kwargs):
            self.kwargs = kwargs
            message = type("Message", (), {"content": "short summary"})()
            choice = type("Choice", (), {"message": message})()
            return type("Completion", (), {"choices": [choice]})()

    class FakeClient:
        def __init__(self):
            self.chat = type("Chat", (), {"completions": FakeCompletions()})()

    monkeypatch.setattr("benchmarks.evidence_graph.summaries._load_processor", lambda _path: FakeProcessor())
    client = FakeClient()
    node = EvidenceNode(id="n", type="paragraph", doc_id="d", page_index=0)
    source = " ".join(f"token{i}" for i in range(300))

    with caplog.at_level("WARNING", logger="benchmarks.evidence_graph.summaries"):
        summary = _summarize_node(
            client,
            "model",
            node,
            source,
            abstract_processor_path="processor-path",
            abstract_context_window=600,
            abstract_output_tokens=20,
            abstract_safety_margin=0,
        )

    prompt = client.chat.completions.kwargs["messages"][0]["content"][0]["text"]
    token_count = len(prompt.split())
    assert summary == "short summary"
    assert token_count + 20 <= 600
    assert "token0" in prompt
    assert "token299" in prompt
    assert "token150" not in prompt
    assert "Truncated LLM abstract source for n" in caplog.text
    assert "before_tokens=" in caplog.text
    assert "after_tokens=" in caplog.text


def test_build_document_graph_writes_nodes_edges_from_embedding_caches(tmp_path):
    context = _fixture_doc(tmp_path)
    config = EvidenceGraphBuildConfig(
        semantic_k=1,
        layout_k=1,
        skip_llm=True,
        skip_embeddings=True,
        overwrite=True,
    )

    result = build_document_graph(context, config)
    artifacts = load_graph_artifacts(context.graph_dir)

    assert result.node_count == 6
    assert result.edge_count > 0
    assert context.graph_dir.joinpath("graph.json").exists()
    assert context.graph_dir.joinpath("nodes.jsonl").exists()
    assert context.graph_dir.joinpath("edges.jsonl").exists()
    assert not context.graph_dir.joinpath("semantic_index.safetensors").exists()

    nodes = {node["id"]: node for node in artifacts.nodes}
    assert "sample-doc:page:0" in nodes
    assert "sample-doc:page:0:block:0:title" in nodes
    assert "sample-doc:page:0:block:1:paragraph" in nodes
    assert all(node["type"] != "page_number" for node in artifacts.nodes)

    paragraph = nodes["sample-doc:page:0:block:1:paragraph"]
    assert paragraph["paragraph"] == "See Figure $1$ for details."
    assert paragraph["abstract"] == "See Figure $1$ for details."
    assert paragraph["in_edges"]
    assert paragraph["out_edges"]

    edges = {(edge["type"], edge["relation"], edge["source"], edge["target"]) for edge in artifacts.edges}
    assert ("containment", "contains", "sample-doc:page:0", "sample-doc:page:0:block:0:title") in edges
    assert ("reading_order", "next", "sample-doc:page:0", "sample-doc:page:1") in edges
    assert (
        "reading_order",
        "next",
        "sample-doc:page:0:block:1:paragraph",
        "sample-doc:page:1:block:0:image",
    ) in edges
    assert any(edge[0] == "section_hierarchy" and edge[1] == "contains_block" for edge in edges)
    assert all(edge[0] != "reference" for edge in edges)
    assert all(edge[0] != "semantic" for edge in edges)

    page_embedding = load_file(context.page_embedding_path, device="cpu")["embeddings"]
    assert tuple(page_embedding[0].shape) == (2, 8)
    assert not (context.node_embedding_dir / "page_0_block_0_title.safetensors").exists()
    assert artifacts.metadata["doc_key"] == "sample-doc"
    assert artifacts.metadata["semantic_embeddings"]["enabled"] is False
    assert artifacts.metadata["semantic_embeddings"]["page_embedding_path"] == str(context.page_embedding_path)
    assert artifacts.metadata["semantic_embeddings"]["node_embedding_dir"] == str(context.node_embedding_dir)
    assert artifacts.metadata["counts"]["nodes"] == 6


def test_build_document_graph_caps_pages_when_max_pages_is_configured(tmp_path):
    context = _fixture_doc(tmp_path)
    pages = [{"page_idx": index, "page_size": [100, 200]} for index in range(121)]
    content_pages = [[] for _index in range(121)]
    _write_json(context.mineru_dir / "layout.json", {"pdf_info": pages})
    _write_json(context.mineru_dir / "sample_content_list_v2.json", content_pages)
    save_file({"embeddings": torch.ones((120, 2, 8), dtype=torch.bfloat16)}, context.page_embedding_path)

    result = build_document_graph(
        context,
        EvidenceGraphBuildConfig(
            semantic_k=0,
            layout_k=1,
            skip_llm=True,
            skip_embeddings=False,
            overwrite=True,
            max_pages=120,
        ),
    )
    artifacts = load_graph_artifacts(context.graph_dir)

    assert result.status == "generated"
    assert max(node["page_index"] for node in artifacts.nodes) == 119
    assert "sample-doc:page:120" not in {node["id"] for node in artifacts.nodes}
    assert artifacts.metadata["sources"]["mineru_page_count"] == 121
    assert artifacts.metadata["sources"]["content_page_count"] == 121
    assert artifacts.metadata["config"]["max_pages"] == 120
    assert artifacts.metadata["counts"]["pages_built"] == 120


def test_overwrite_reuses_existing_abstracts_without_calling_llm(tmp_path, monkeypatch):
    context = _fixture_doc(tmp_path)
    build_document_graph(
        context,
        EvidenceGraphBuildConfig(
            semantic_k=1,
            layout_k=1,
            skip_llm=True,
            skip_embeddings=True,
            overwrite=True,
        ),
    )
    artifacts = load_graph_artifacts(context.graph_dir)
    rows = []
    for row in artifacts.nodes:
        row["abstract"] = f"paid abstract for {row['id']}"
        rows.append(row)
    nodes_path = context.graph_dir / "nodes.jsonl"
    nodes_path.write_text("\n".join(json.dumps(row, ensure_ascii=False) for row in rows) + "\n", encoding="utf-8")

    def fail_summarize_node(*_args, **_kwargs):
        raise AssertionError("existing abstracts must be reused instead of regenerating LLM abstracts")

    monkeypatch.setattr("benchmarks.evidence_graph.summaries._summarize_node", fail_summarize_node)

    result = build_document_graph(
        context,
        EvidenceGraphBuildConfig(
            semantic_k=1,
            layout_k=1,
            skip_llm=False,
            skip_embeddings=True,
            overwrite=True,
        ),
    )
    rewritten = load_graph_artifacts(context.graph_dir)

    assert result.status == "generated"
    assert {row["abstract"] for row in rewritten.nodes} == {f"paid abstract for {row['id']}" for row in rows}
    assert rewritten.metadata["config"]["preserve_existing_abstracts"] is True


def test_node_embedding_text_does_not_duplicate_table_source_text():
    html = "<table><tr><td>Revenue</td><td>42</td></tr></table>"
    node = EvidenceNode(
        id="doc:page:0:block:0:table",
        type="table",
        doc_id="doc.pdf",
        page_index=0,
        index=0,
        abstract="old summary",
        fields={
            "html": html,
            "caption": "Financial results",
            "footnote": "",
            "table_type": "simple_table",
            "table_nest_level": 1,
            "image_path": "/tmp/table.png",
        },
        metadata={"source_text": html},
    )

    text = _node_embedding_text(node)

    assert text.count(html) == 1
    assert "old summary" not in text
    assert "image_path" not in text


def test_materialize_node_embeddings_chunks_long_node_text(tmp_path, monkeypatch):
    page_embedding_path = tmp_path / "page.safetensors"
    node_embedding_path = tmp_path / "node.safetensors"
    save_file({"embeddings": torch.ones((1, 3, 128), dtype=torch.bfloat16)}, page_embedding_path)
    long_text = "alpha " * 1500
    nodes = [
        EvidenceNode(
            id="doc:page:0",
            type="page",
            doc_id="doc.pdf",
            page_index=0,
            embedding_path=str(page_embedding_path),
        ),
        EvidenceNode(
            id="doc:page:0:block:0:paragraph",
            type="paragraph",
            doc_id="doc.pdf",
            page_index=0,
            index=0,
            embedding_path=str(node_embedding_path),
            fields={"paragraph": long_text},
        ),
    ]
    calls = []

    def fake_encode_text(*, model, vllm_url, text):
        calls.append(text)
        return torch.ones((2, 128), dtype=torch.float32) * len(calls)

    monkeypatch.setattr("benchmarks.evidence_graph.embeddings.MAX_EMBEDDING_INPUT_CHARS", 2000)
    monkeypatch.setattr("benchmarks.evidence_graph.embeddings._encode_text", fake_encode_text)

    vectors = materialize_node_embeddings(
        nodes,
        model="colpali-v1.3",
        vllm_url="http://localhost:8020",
        skip_embeddings=False,
        overwrite=False,
    )

    assert len(calls) > 1
    assert all(len(call) <= 2000 for call in calls)
    assert "".join(calls).replace("\n", " ").count("alpha") >= 1500
    assert tuple(vectors[1].shape) == (2 * len(calls), 128)
    assert tuple(load_file(node_embedding_path, device="cpu")["embedding"].shape) == (2 * len(calls), 128)


def test_materialize_node_embeddings_uses_image_payload_for_visual_element_nodes(tmp_path, monkeypatch):
    page_embedding_path = tmp_path / "page.safetensors"
    node_embedding_path = tmp_path / "image_node.safetensors"
    image_path = tmp_path / "figure.jpg"
    save_file({"embeddings": torch.ones((1, 3, 128), dtype=torch.bfloat16)}, page_embedding_path)
    image_path.write_bytes(b"fake-jpeg")
    nodes = [
        EvidenceNode(
            id="doc:page:0",
            type="page",
            doc_id="doc.pdf",
            page_index=0,
            embedding_path=str(page_embedding_path),
        ),
        EvidenceNode(
            id="doc:page:0:block:0:image",
            type="image",
            doc_id="doc.pdf",
            page_index=0,
            index=0,
            embedding_path=str(node_embedding_path),
            fields={"image_path": str(image_path), "caption": "Figure 1: Pipeline"},
        ),
    ]
    image_calls = []

    def fake_encode_image_path(*, model, vllm_url, image_path):
        image_calls.append((model, vllm_url, image_path))
        return torch.ones((5, 128), dtype=torch.float32)

    def fail_encode_text(**_kwargs):
        raise AssertionError("visual element nodes with image files must not use text-only embeddings")

    monkeypatch.setattr("benchmarks.evidence_graph.embeddings._encode_image_path", fake_encode_image_path)
    monkeypatch.setattr("benchmarks.evidence_graph.embeddings._encode_text", fail_encode_text)

    vectors = materialize_node_embeddings(
        nodes,
        model="colpali-v1.3",
        vllm_url="http://localhost:8020",
        skip_embeddings=False,
        overwrite=False,
    )

    assert image_calls == [("colpali-v1.3", "http://localhost:8020", str(image_path))]
    assert tuple(vectors[1].shape) == (5, 128)
    assert tuple(load_file(node_embedding_path, device="cpu")["embedding"].shape) == (5, 128)


def test_encode_image_path_once_sends_multimodal_pooling_payload(tmp_path, monkeypatch):
    image_path = tmp_path / "crop.png"
    image_path.write_bytes(b"fake-image")
    requests_seen = []

    class FakeResponse:
        def raise_for_status(self):
            return None

        def json(self):
            return {"data": [{"data": [[1.0, 0.0], [0.0, 1.0]]}]}

    def fake_post(url, json, timeout):
        requests_seen.append((url, json, timeout))
        return FakeResponse()

    monkeypatch.setattr("benchmarks.evidence_graph.embeddings.requests.post", fake_post)

    tensor = _encode_image_path_once(model="colpali-v1.3", vllm_url="http://localhost:8020", image_path=str(image_path))

    url, payload, timeout = requests_seen[0]
    assert url == "http://localhost:8020/pooling"
    assert timeout == 180
    assert tuple(tensor.shape) == (2, 2)
    assert payload["model"] == "colpali-v1.3"
    assert payload["task"] == "token_embed"
    assert "input" not in payload
    content = payload["messages"][0]["content"]
    assert content[0] == {"type": "text", "text": "<image>"}
    assert content[1]["type"] == "image_url"
    assert content[1]["image_url"]["url"].startswith("data:image/png;base64,")


def test_materialize_node_embeddings_regenerates_stale_node_cache_with_wrong_dim(tmp_path, monkeypatch):
    page_embedding_path = tmp_path / "page.safetensors"
    node_embedding_path = tmp_path / "node.safetensors"
    save_file({"embeddings": torch.ones((1, 3, 128), dtype=torch.bfloat16)}, page_embedding_path)
    save_file({"embedding": torch.ones((2, 8), dtype=torch.bfloat16)}, node_embedding_path)

    nodes = [
        EvidenceNode(
            id="doc:page:0",
            type="page",
            doc_id="doc.pdf",
            page_index=0,
            embedding_path=str(page_embedding_path),
        ),
        EvidenceNode(
            id="doc:page:0:block:0:paragraph",
            type="paragraph",
            doc_id="doc.pdf",
            page_index=0,
            index=0,
            embedding_path=str(node_embedding_path),
            fields={"paragraph": "semantic content"},
        ),
    ]

    def fake_encode_text(*, model, vllm_url, text):
        return torch.ones((4, 128), dtype=torch.float32)

    monkeypatch.setattr("benchmarks.evidence_graph.embeddings._encode_text", fake_encode_text)

    vectors = materialize_node_embeddings(
        nodes,
        model="colpali-v1.3",
        vllm_url="http://localhost:8020",
        skip_embeddings=False,
        overwrite=False,
    )

    assert tuple(vectors[0].shape) == (3, 128)
    assert tuple(vectors[1].shape) == (4, 128)
    assert tuple(load_file(node_embedding_path, device="cpu")["embedding"].shape) == (4, 128)


def test_semantic_edges_use_token_level_maxsim():
    nodes = [
        type("Node", (), {"id": "a", "type": "paragraph"})(),
        type("Node", (), {"id": "b", "type": "paragraph"})(),
        type("Node", (), {"id": "c", "type": "paragraph"})(),
    ]
    vectors = [
        torch.tensor([[1.0, 0.0], [0.0, 1.0]]),
        torch.tensor([[1.0, 0.0], [1.0, 0.0]]),
        torch.tensor([[-1.0, 0.0], [-1.0, 0.0]]),
    ]

    edges = build_semantic_edges(nodes, vectors, semantic_k=1, semantic_device="cpu")

    assert maxsim_score(vectors[0], vectors[1]) == 0.5
    assert maxsim_score(vectors[0], vectors[2]) == -0.5
    assert edges[0].source == "a"
    assert edges[0].target == "b"
    assert edges[0].weight == 0.5
    assert edges[0].metadata["scoring"] == "maxsim"


def test_semantic_edges_keep_page_and_element_layers_separate():
    nodes = [
        type("Node", (), {"id": "page-a", "type": "page"})(),
        type("Node", (), {"id": "page-b", "type": "page"})(),
        type("Node", (), {"id": "element-a", "type": "paragraph"})(),
        type("Node", (), {"id": "element-b", "type": "image"})(),
    ]
    vectors = [
        torch.tensor([[1.0, 0.0]]),
        torch.tensor([[0.0, 1.0]]),
        torch.tensor([[1.0, 0.0]]),
        torch.tensor([[0.0, 1.0]]),
    ]

    edges = build_semantic_edges(nodes, vectors, semantic_k=2, semantic_device="cpu")

    node_types = {node.id: node.type for node in nodes}
    assert edges
    assert all(
        (node_types[edge.source] == "page") == (node_types[edge.target] == "page")
        for edge in edges
    )
    assert ("page-a", "element-a") not in {(edge.source, edge.target) for edge in edges}
    assert ("element-a", "page-a") not in {(edge.source, edge.target) for edge in edges}


def test_batched_maxsim_scores_match_pairwise_maxsim_for_variable_length_targets():
    source = torch.tensor([[1.0, 0.0], [0.0, 1.0]])
    targets = [
        torch.tensor([[1.0, 0.0], [1.0, 0.0]]),
        torch.tensor([[-1.0, 0.0], [-1.0, 0.0], [0.0, -1.0]]),
        torch.tensor([[0.0, 1.0]]),
    ]

    batched = _batched_maxsim_scores(source, targets, target_chunk_size=2)
    pairwise = torch.tensor([maxsim_score(source, target) for target in targets])

    assert torch.allclose(batched, pairwise)


def test_batched_maxsim_score_matrix_matches_pairwise_maxsim_for_variable_lengths():
    sources = [
        torch.tensor([[1.0, 0.0], [0.0, 1.0]]),
        torch.tensor([[1.0, 0.0]]),
        torch.tensor([[-1.0, 0.0], [0.0, -1.0], [1.0, 0.0]]),
    ]
    targets = [
        torch.tensor([[1.0, 0.0], [1.0, 0.0]]),
        torch.tensor([[-1.0, 0.0], [-1.0, 0.0], [0.0, -1.0]]),
        torch.tensor([[0.0, 1.0]]),
    ]

    batched = _batched_maxsim_score_matrix(sources, targets)
    pairwise = torch.tensor([[maxsim_score(source, target) for target in targets] for source in sources])

    assert torch.allclose(batched, pairwise)


def test_semantic_chunk_logs_warning_when_cuda_oom_falls_back_to_next_device(caplog, monkeypatch):
    from benchmarks.evidence_graph import semantic

    calls = []
    original = semantic._batched_maxsim_score_matrix

    def fail_once_on_cuda(sources, targets, *, normalize=True, device=None):
        calls.append(str(device))
        if str(device) == "cuda:1":
            raise RuntimeError("CUDA out of memory")
        return original(sources, targets, normalize=normalize, device=device)

    monkeypatch.setattr(semantic, "_batched_maxsim_score_matrix", fail_once_on_cuda)

    with caplog.at_level("WARNING", logger="benchmarks.evidence_graph.semantic"):
        scores = semantic._score_group_chunk(
            [torch.tensor([[1.0, 0.0]])],
            [torch.tensor([[1.0, 0.0]])],
            target_chunk_size=1,
            devices=[torch.device("cuda:1"), torch.device("cuda:0"), torch.device("cpu")],
        )

    assert tuple(scores.shape) == (1, 1)
    assert calls == ["cuda:1", "cuda:0"]
    assert "falling back to cuda:0" in caplog.text


def test_maxsim_score_rejects_mismatched_embedding_dims():
    with pytest.raises(ValueError, match="Embedding dimension mismatch"):
        maxsim_score(torch.ones((2, 8)), torch.ones((3, 128)))

def test_build_many_runs_documents_concurrently_and_preserves_manifest_order(tmp_path, monkeypatch):
    contexts = []
    for index in range(3):
        contexts.append(
            GraphPathContext(
                benchmark_name="mmlongbench",
                doc_id=f"doc-{index}.pdf",
                doc_key=f"doc-{index}",
                mineru_dir=tmp_path / f"mineru/doc-{index}",
                page_image_paths={},
                page_embedding_path=tmp_path / f"page_embeddings/doc-{index}.safetensors",
                graph_dir=tmp_path / f"graphs/doc-{index}",
                node_embedding_dir=tmp_path / f"node_embeddings/doc-{index}",
            )
        )

    starts = {}

    def fake_build_document_graph(context, config):
        starts[context.doc_key] = time.monotonic()
        time.sleep(0.2)
        return BuildResult(context.doc_key, str(context.graph_dir), 1, 1, "generated")

    monkeypatch.setattr(build_evidence_graphs, "build_document_graph", fake_build_document_graph)

    records = build_evidence_graphs.build_many(
        contexts,
        EvidenceGraphBuildConfig(skip_llm=True, skip_embeddings=True),
        workers=3,
    )

    assert [record["doc_key"] for record in records] == ["doc-0", "doc-1", "doc-2"]
    assert max(starts.values()) - min(starts.values()) < 0.15


def test_build_many_preloads_abstract_processor_before_workers(tmp_path, monkeypatch):
    contexts = []
    for index in range(2):
        contexts.append(
            GraphPathContext(
                benchmark_name="mmlongbench",
                doc_id=f"doc-{index}.pdf",
                doc_key=f"doc-{index}",
                mineru_dir=tmp_path / f"mineru/doc-{index}",
                page_image_paths={},
                page_embedding_path=tmp_path / f"page_embeddings/doc-{index}.safetensors",
                graph_dir=tmp_path / f"graphs/doc-{index}",
                node_embedding_dir=tmp_path / f"node_embeddings/doc-{index}",
            )
        )
    events = []

    def fake_warm_abstract_processor(path):
        events.append(("warm", path))

    def fake_build_document_graph(context, config):
        events.append(("build", context.doc_key))
        return BuildResult(context.doc_key, str(context.graph_dir), 1, 1, "generated")

    monkeypatch.setattr(build_evidence_graphs, "warm_abstract_processor", fake_warm_abstract_processor)
    monkeypatch.setattr(build_evidence_graphs, "build_document_graph", fake_build_document_graph)

    build_evidence_graphs.build_many(
        contexts,
        EvidenceGraphBuildConfig(
            skip_llm=False,
            skip_embeddings=True,
            abstract_processor_path="processor-path",
            abstract_context_window=131072,
        ),
        workers=2,
    )

    assert events[0] == ("warm", "processor-path")
    assert [event[0] for event in events].count("warm") == 1


def test_build_script_applies_mmlongbench_only_default_max_pages(monkeypatch):
    configs = []

    def fake_contexts(_args):
        return []

    def fake_build_many(_contexts, config, workers=1):
        configs.append(config)
        return []

    monkeypatch.setattr(build_evidence_graphs, "_contexts", fake_contexts)
    monkeypatch.setattr(build_evidence_graphs, "build_many", fake_build_many)
    monkeypatch.setattr("sys.argv", ["build_evidence_graphs.py", "--benchmark", "mmlongbench"])
    build_evidence_graphs.main()

    monkeypatch.setattr("sys.argv", ["build_evidence_graphs.py", "--benchmark", "longdocurl"])
    build_evidence_graphs.main()

    assert configs[0].max_pages == 120
    assert configs[1].max_pages is None
