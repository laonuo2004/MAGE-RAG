import logging
import base64
import mimetypes
from pathlib import Path

import requests
import torch
from safetensors.torch import load_file, save_file

from benchmarks.evidence_graph.content import normalize_text
from benchmarks.evidence_graph.retry import call_with_retries
from benchmarks.evidence_graph.schema import EvidenceNode

MAX_EMBEDDING_INPUT_CHARS = 6000
VISUAL_NODE_TYPES = {"image", "chart", "table", "equation_interline"}

logger = logging.getLogger(__name__)


def materialize_node_embeddings(
    nodes: list[EvidenceNode],
    *,
    model: str,
    vllm_url: str,
    skip_embeddings: bool,
    overwrite: bool,
) -> list[torch.Tensor]:
    if skip_embeddings:
        return []
    vectors = []
    expected_dim = _infer_embedding_dim(nodes)
    page_embedding_cache: dict[Path, torch.Tensor] = {}
    for node in nodes:
        if node.type == "page":
            vector = _load_page_embedding(node, page_embedding_cache, skip_embeddings=skip_embeddings, expected_dim=expected_dim)
        else:
            vector = _load_or_create_node_embedding(
                node,
                model=model,
                vllm_url=vllm_url,
                skip_embeddings=skip_embeddings,
                overwrite=overwrite,
                expected_dim=expected_dim,
            )
        expected_dim = _validate_embedding_dim(vector, node, expected_dim)
        vectors.append(vector)
    return vectors


def _infer_embedding_dim(nodes: list[EvidenceNode]) -> int | None:
    for node in nodes:
        if node.type != "page" or not node.embedding_path:
            continue
        path = Path(node.embedding_path)
        if not path.exists():
            continue
        payload = load_file(path, device="cpu")
        if "embeddings" in payload:
            tensor = payload["embeddings"]
        elif "embedding" in payload:
            tensor = payload["embedding"]
        else:
            continue
        if tensor.ndim in (2, 3):
            return int(tensor.shape[-1])
    return None


def _validate_embedding_dim(vector: torch.Tensor, node: EvidenceNode, expected_dim: int | None) -> int:
    if vector.ndim != 2:
        raise ValueError(f"Expected token embedding [tokens, dim] for {node.id}, got {tuple(vector.shape)}")
    dim = int(vector.shape[-1])
    if expected_dim is not None and dim != expected_dim:
        raise ValueError(
            f"Embedding dimension mismatch for {node.id}: got {dim}, expected {expected_dim}. "
            "Remove stale node embedding caches or rebuild with --overwrite."
        )
    return dim


def _load_page_embedding(
    node: EvidenceNode,
    page_embedding_cache: dict[Path, torch.Tensor],
    *,
    skip_embeddings: bool,
    expected_dim: int | None,
) -> torch.Tensor:
    if not node.embedding_path:
        raise ValueError(f"Page node {node.id} has no page embedding path")
    path = Path(node.embedding_path)
    if not path.exists():
        raise FileNotFoundError(f"Page embedding file not found for {node.id}: {path}")
    if path not in page_embedding_cache:
        payload = load_file(path, device="cpu")
        if "embeddings" in payload:
            tensor = payload["embeddings"]
        elif "embedding" in payload:
            tensor = payload["embedding"]
        else:
            raise KeyError(f"Page embedding file {path} must contain 'embeddings' or 'embedding'")
        page_embedding_cache[path] = tensor.to(torch.float32)
    embeddings = page_embedding_cache[path]
    if embeddings.ndim == 3:
        if node.page_index >= embeddings.shape[0]:
            raise IndexError(f"Page index {node.page_index} out of range for {path} with shape {tuple(embeddings.shape)}")
        return embeddings[node.page_index].to(torch.float32)
    if embeddings.ndim == 2:
        return embeddings.to(torch.float32)
    raise ValueError(f"Expected page embeddings [pages, tokens, dim] or [tokens, dim], got {tuple(embeddings.shape)}")


def _load_or_create_node_embedding(
    node: EvidenceNode,
    *,
    model: str,
    vllm_url: str,
    skip_embeddings: bool,
    overwrite: bool,
    expected_dim: int | None,
) -> torch.Tensor:
    if not node.embedding_path:
        raise ValueError(f"Element node {node.id} has no node embedding path")
    path = Path(node.embedding_path)
    if path.exists() and not overwrite:
        tensor = _load_single_embedding(path)
        if expected_dim is None or tensor.shape[-1] == expected_dim:
            return tensor

    image_path = _node_embedding_image_path(node)
    if image_path:
        tensor = _encode_image_path(model=model, vllm_url=vllm_url, image_path=image_path)
    else:
        text = _node_embedding_text(node)
        tensor = _encode_node_text(model=model, vllm_url=vllm_url, text=text)
    if expected_dim is not None and tensor.shape[-1] != expected_dim:
        raise ValueError(
            f"Generated embedding for {node.id} has dim {tensor.shape[-1]}, expected {expected_dim}. "
            "Check that page and node embeddings are produced by the same ColPali model."
        )
    path.parent.mkdir(parents=True, exist_ok=True)
    save_file({"embedding": tensor.to(torch.bfloat16)}, path)
    return tensor.to(torch.float32)


def _load_single_embedding(path: Path) -> torch.Tensor:
    payload = load_file(path, device="cpu")
    if "embedding" in payload:
        tensor = payload["embedding"]
    elif "embeddings" in payload:
        tensor = payload["embeddings"]
    else:
        raise KeyError(f"Node embedding file {path} must contain 'embedding' or 'embeddings'")
    if tensor.ndim != 2:
        raise ValueError(f"Expected node embedding [tokens, dim] in {path}, got {tuple(tensor.shape)}")
    return tensor.to(torch.float32)


def _node_embedding_text(node: EvidenceNode) -> str:
    fields = node.fields
    if node.type == "title":
        parts = [fields.get("title")]
    elif node.type == "paragraph":
        parts = [fields.get("paragraph")]
    elif node.type == "list":
        parts = fields.get("items") or []
    elif node.type in {"image", "chart"}:
        parts = [fields.get("caption"), fields.get("content"), fields.get("footnote")]
    elif node.type == "table":
        parts = [fields.get("caption"), fields.get("html"), fields.get("footnote")]
    elif node.type == "equation_interline":
        parts = [fields.get("latex")]
    elif node.type == "code":
        parts = [fields.get("caption"), fields.get("language"), fields.get("code")]
    elif node.type == "algorithm":
        parts = [fields.get("caption"), fields.get("algorithm")]
    elif node.type == "page":
        parts = [node.metadata.get("source_text"), node.abstract]
    else:
        parts = [node.metadata.get("source_text")]
        parts.extend(value for key, value in sorted(fields.items()) if not key.endswith("path"))
    return _join_embedding_parts(parts)


def _node_embedding_image_path(node: EvidenceNode) -> str:
    if node.type not in VISUAL_NODE_TYPES:
        return ""
    image_path = str(node.fields.get("image_path") or "")
    if not image_path:
        return ""
    path = Path(image_path)
    if not path.exists() or not path.is_file():
        logger.warning("Visual node %s missing image file for ColPali embedding, falling back to text: %s", node.id, path)
        return ""
    return str(path)


def _join_embedding_parts(parts) -> str:
    normalized = []
    for part in parts:
        if isinstance(part, list):
            normalized.extend(normalize_text(item) for item in part)
        else:
            normalized.append(normalize_text(part))
    return "\n".join(part for part in normalized if part)


def _encode_node_text(*, model: str, vllm_url: str, text: str) -> torch.Tensor:
    chunks = _split_embedding_text(text or " ")
    tensors = [_encode_text(model=model, vllm_url=vllm_url, text=chunk) for chunk in chunks]
    if not tensors:
        return _encode_text(model=model, vllm_url=vllm_url, text=" ")
    dim = tensors[0].shape[-1]
    for tensor in tensors:
        if tensor.ndim != 2 or tensor.shape[-1] != dim:
            raise ValueError("Chunked ColPali embeddings must all be [tokens, dim] with the same dim")
    return torch.cat(tensors, dim=0)


def _encode_image_path(*, model: str, vllm_url: str, image_path: str) -> torch.Tensor:
    return call_with_retries(
        lambda: _encode_image_path_once(model=model, vllm_url=vllm_url, image_path=image_path),
        operation_name="ColPali /pooling image request",
        logger=logger,
    )


def _split_embedding_text(text: str) -> list[str]:
    if len(text) <= MAX_EMBEDDING_INPUT_CHARS:
        return [text]
    chunks = []
    start = 0
    while start < len(text):
        end = min(start + MAX_EMBEDDING_INPUT_CHARS, len(text))
        if end < len(text):
            split = _last_split_position(text, start, end)
            if split > start:
                end = split
        chunk = text[start:end].strip()
        if chunk:
            chunks.append(chunk)
        start = end
        while start < len(text) and text[start].isspace():
            start += 1
    return chunks or [" "]


def _last_split_position(text: str, start: int, end: int) -> int:
    minimum = start + max(1, MAX_EMBEDDING_INPUT_CHARS // 2)
    for index in range(end, minimum, -1):
        if text[index - 1].isspace() or text[index - 1] in ";,</>":
            return index
    return end


def _encode_text(*, model: str, vllm_url: str, text: str) -> torch.Tensor:
    return call_with_retries(
        lambda: _encode_text_once(model=model, vllm_url=vllm_url, text=text),
        operation_name="ColPali /pooling request",
        logger=logger,
    )


def _encode_text_once(*, model: str, vllm_url: str, text: str) -> torch.Tensor:
    response = requests.post(
        f"{vllm_url.rstrip('/')}/pooling",
        json={"model": model, "task": "token_embed", "input": [text or " "]},
        timeout=180,
    )
    try:
        response.raise_for_status()
    except requests.HTTPError as exc:
        raise RuntimeError(f"ColPali /pooling request failed: {response.text[:1000]}") from exc
    payload = response.json()
    tensor = torch.tensor(payload["data"][0]["data"], dtype=torch.float32)
    if tensor.ndim != 2:
        raise ValueError(f"Expected node embedding [tokens, dim], got {tuple(tensor.shape)}")
    return tensor


def _encode_image_path_once(*, model: str, vllm_url: str, image_path: str) -> torch.Tensor:
    image_url = _image_data_url(image_path)
    response = requests.post(
        f"{vllm_url.rstrip('/')}/pooling",
        json={
            "model": model,
            "task": "token_embed",
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": "<image>"},
                        {"type": "image_url", "image_url": {"url": image_url}},
                    ],
                }
            ],
        },
        timeout=180,
    )
    try:
        response.raise_for_status()
    except requests.HTTPError as exc:
        raise RuntimeError(f"ColPali /pooling image request failed: {response.text[:1000]}") from exc
    payload = response.json()
    tensor = torch.tensor(payload["data"][0]["data"], dtype=torch.float32)
    if tensor.ndim != 2:
        raise ValueError(f"Expected image node embedding [tokens, dim], got {tuple(tensor.shape)}")
    return tensor


def _image_data_url(image_path: str) -> str:
    path = Path(image_path)
    mime_type = mimetypes.guess_type(path.name)[0] or "image/png"
    image_b64 = base64.b64encode(path.read_bytes()).decode("ascii")
    return f"data:{mime_type};base64,{image_b64}"
