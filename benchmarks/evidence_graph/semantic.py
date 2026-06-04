import logging
from time import perf_counter

import torch
from benchmarks.evidence_graph.schema import EvidenceEdge, EvidenceNode

PAGE_TARGET_CHUNK_SIZE = 16
ELEMENT_TARGET_CHUNK_SIZE = 16
PAGE_SOURCE_CHUNK_SIZE = 16
ELEMENT_SOURCE_CHUNK_SIZE = 16
AUTO_SEMANTIC_DEVICE_ORDER = ("cuda:1", "cuda:0", "cpu")

logger = logging.getLogger(__name__)


def build_semantic_edges(
    nodes: list[EvidenceNode],
    vectors: list[torch.Tensor],
    *,
    semantic_k: int,
    semantic_device: str = "auto",
) -> list[EvidenceEdge]:
    if semantic_k <= 0 or len(nodes) <= 1:
        return []
    if len(nodes) != len(vectors):
        raise ValueError(f"Expected one embedding tensor per node, got {len(nodes)} nodes and {len(vectors)} tensors")

    started_at = perf_counter()
    normalized_started_at = perf_counter()
    normalized_vectors = [_normalize_tokens(vector) for vector in vectors]
    normalized_seconds = perf_counter() - normalized_started_at
    device_started_at = perf_counter()
    devices = _resolve_semantic_devices(semantic_device)
    device_seconds = perf_counter() - device_started_at
    logger.info(
        "Building semantic edges with devices=%s requested=%s page_chunk=%sx%s element_chunk=%sx%s nodes=%s",
        ",".join(str(device) for device in devices),
        semantic_device,
        PAGE_SOURCE_CHUNK_SIZE,
        PAGE_TARGET_CHUNK_SIZE,
        ELEMENT_SOURCE_CHUNK_SIZE,
        ELEMENT_TARGET_CHUNK_SIZE,
        len(nodes),
    )
    edges = []
    for group_offsets in _semantic_layer_offsets(nodes):
        edges.extend(
            _build_semantic_edges_for_group(
                nodes,
                normalized_vectors,
                group_offsets,
                semantic_k=semantic_k,
                devices=devices,
            )
        )
    # logger.info(
    #     "Semantic edge timing nodes=%s edges=%s normalize=%.3fs device_resolve=%.3fs build=%.3fs total=%.3fs",
    #     len(nodes),
    #     len(edges),
    #     normalized_seconds,
    #     device_seconds,
    #     perf_counter() - started_at - normalized_seconds - device_seconds,
    #     perf_counter() - started_at,
    # )
    return edges


def _semantic_layer_offsets(nodes: list[EvidenceNode]) -> list[list[int]]:
    page_offsets = []
    element_offsets = []
    for offset, node in enumerate(nodes):
        if node.type == "page":
            page_offsets.append(offset)
        else:
            element_offsets.append(offset)
    return [offsets for offsets in (page_offsets, element_offsets) if len(offsets) > 1]


def _build_semantic_edges_for_group(
    nodes: list[EvidenceNode],
    vectors: list[torch.Tensor],
    group_offsets: list[int],
    *,
    semantic_k: int,
    devices: list[torch.device],
) -> list[EvidenceEdge]:
    edges = []
    k = min(semantic_k, len(group_offsets) - 1)
    if k <= 0:
        return edges
    started_at = perf_counter()
    node_type = nodes[group_offsets[0]].type
    target_vectors = [vectors[offset] for offset in group_offsets]
    source_group_indexes = {source_offset: group_index for group_index, source_offset in enumerate(group_offsets)}
    source_chunk_size = _source_chunk_size(nodes[group_offsets[0]])
    target_chunk_size = _target_chunk_size(nodes[group_offsets[0]])
    scoring_seconds = 0.0
    topk_seconds = 0.0
    edge_seconds = 0.0
    chunk_count = 0
    for source_start in range(0, len(group_offsets), source_chunk_size):
        source_offsets = group_offsets[source_start : source_start + source_chunk_size]
        source_vectors = [vectors[offset] for offset in source_offsets]
        scoring_started_at = perf_counter()
        score_matrix = _score_group_chunk(source_vectors, target_vectors, target_chunk_size=target_chunk_size, devices=devices)
        scoring_seconds += perf_counter() - scoring_started_at
        chunk_count += 1
        for source_row, source_offset in enumerate(source_offsets):
            source = nodes[source_offset]
            source_group_index = source_group_indexes[source_offset]
            scores = score_matrix[source_row]
            scores[source_group_index] = float("-inf")
            topk_started_at = perf_counter()
            top_scores, top_group_indexes = torch.topk(scores, k=k)
            topk_seconds += perf_counter() - topk_started_at
            edge_started_at = perf_counter()
            for rank, (score, target_group_index) in enumerate(zip(top_scores.tolist(), top_group_indexes.tolist(), strict=True)):
                if score == float("-inf"):
                    continue
                target_offset = group_offsets[target_group_index]
                target = nodes[target_offset]
                edges.append(
                    EvidenceEdge(
                        id=f"edge:semantic:similar_to:{source_offset}:{target_offset}:{rank}",
                        source=source.id,
                        target=target.id,
                        type="semantic",
                        relation="similar_to",
                        weight=float(score),
                        metadata={"rank": rank, "scoring": "maxsim"},
                    )
                )
            edge_seconds += perf_counter() - edge_started_at
    logger.info(
        "Semantic group timing layer=%s nodes=%s edges=%s source_chunk=%s target_chunk=%s source_chunks=%s scoring=%.3fs topk=%.3fs edges=%.3fs total=%.3fs",
        "page" if node_type == "page" else "element",
        len(group_offsets),
        len(edges),
        source_chunk_size,
        target_chunk_size,
        chunk_count,
        scoring_seconds,
        topk_seconds,
        edge_seconds,
        perf_counter() - started_at,
    )
    return edges


def _score_group_chunk(
    source_vectors: list[torch.Tensor],
    target_vectors: list[torch.Tensor],
    *,
    target_chunk_size: int,
    devices: list[torch.device],
) -> torch.Tensor:
    score_chunks = []
    started_at = perf_counter()
    for target_start in range(0, len(target_vectors), target_chunk_size):
        target_chunk = target_vectors[target_start : target_start + target_chunk_size]
        score_chunks.append(_score_target_chunk_with_device_fallback(source_vectors, target_chunk, devices))
    result = torch.cat(score_chunks, dim=1)
    # logger.info(
    #     "Semantic score chunk timing device=%s source_chunk=%s target_total=%s target_chunk=%s target_chunks=%s seconds=%.3fs",
    #     device,
    #     len(source_vectors),
    #     len(target_vectors),
    #     target_chunk_size,
    #     len(score_chunks),
    #     perf_counter() - started_at,
    # )
    return result


def _score_target_chunk_with_device_fallback(
    source_vectors: list[torch.Tensor],
    target_chunk: list[torch.Tensor],
    devices: list[torch.device],
) -> torch.Tensor:
    last_oom = None
    for index, device in enumerate(devices):
        try:
            return _batched_maxsim_score_matrix(source_vectors, target_chunk, normalize=False, device=device)
        except RuntimeError as exc:
            if not _is_cuda_out_of_memory(exc) or device.type == "cpu":
                raise
            last_oom = exc
            _safe_empty_cache()
            next_device = devices[index + 1] if index + 1 < len(devices) else None
            logger.warning(
                "CUDA OOM while scoring semantic chunk on %s; falling back to %s for source_chunk=%s target_chunk=%s",
                device,
                next_device or "no remaining device",
                len(source_vectors),
                len(target_chunk),
            )
    if last_oom is not None:
        raise last_oom
    raise RuntimeError("No semantic scoring devices available")


def _safe_empty_cache() -> None:
    try:
        torch.cuda.empty_cache()
    except RuntimeError as exc:
        if not _is_cuda_out_of_memory(exc):
            raise


def _source_chunk_size(source: EvidenceNode) -> int:
    if source.type == "page":
        return PAGE_SOURCE_CHUNK_SIZE
    return ELEMENT_SOURCE_CHUNK_SIZE


def _target_chunk_size(source: EvidenceNode) -> int:
    if source.type == "page":
        return PAGE_TARGET_CHUNK_SIZE
    return ELEMENT_TARGET_CHUNK_SIZE


def _resolve_semantic_devices(semantic_device: str) -> list[torch.device]:
    requested = str(semantic_device or "auto").strip().lower()
    if requested == "auto":
        candidates = AUTO_SEMANTIC_DEVICE_ORDER
    else:
        candidates = tuple(part.strip() for part in requested.split(",") if part.strip())
    devices = []
    for candidate in candidates:
        if candidate == "cpu":
            devices.append(torch.device("cpu"))
            continue
        if candidate.startswith("cuda") and _cuda_device_available(candidate):
            devices.append(torch.device(candidate))
    return devices or [torch.device("cpu")]


def _cuda_device_available(candidate: str) -> bool:
    if not torch.cuda.is_available():
        return False
    if candidate == "cuda":
        return True
    try:
        index = int(candidate.split(":", 1)[1])
    except (IndexError, ValueError):
        return False
    return index < torch.cuda.device_count()


def _is_cuda_out_of_memory(exc: RuntimeError) -> bool:
    message = str(exc).lower()
    return "cuda" in message and "out of memory" in message


def maxsim_score(source: torch.Tensor, target: torch.Tensor) -> float:
    source = _normalize_tokens(source)
    target = _normalize_tokens(target)
    if source.numel() == 0 or target.numel() == 0:
        return 0.0
    if source.shape[-1] != target.shape[-1]:
        raise ValueError(
            f"Embedding dimension mismatch for MaxSim: source dim {source.shape[-1]}, target dim {target.shape[-1]}. "
            "Check for stale node embedding caches or mixed ColPali models."
        )
    token_scores = source @ target.T
    return float(token_scores.max(dim=1).values.mean().item())


def _batched_maxsim_scores(
    source: torch.Tensor,
    targets: list[torch.Tensor],
    *,
    target_chunk_size: int,
    normalize: bool = True,
) -> torch.Tensor:
    if normalize:
        source = _normalize_tokens(source)
    if not targets:
        return torch.empty((0,), dtype=torch.float32)
    if target_chunk_size < 1:
        raise ValueError("target_chunk_size must be >= 1")
    if source.numel() == 0:
        return torch.zeros((len(targets),), dtype=torch.float32)
    scores = []
    for start in range(0, len(targets), target_chunk_size):
        chunk_targets = targets[start : start + target_chunk_size]
        chunk = [_normalize_tokens(target) for target in chunk_targets] if normalize else chunk_targets
        scores.append(_batched_maxsim_scores_for_chunk(source, chunk))
    return torch.cat(scores, dim=0)


def _batched_maxsim_scores_for_chunk(source: torch.Tensor, targets: list[torch.Tensor]) -> torch.Tensor:
    dim = source.shape[-1]
    for target in targets:
        if target.shape[-1] != dim:
            raise ValueError(
                f"Embedding dimension mismatch for MaxSim: source dim {dim}, target dim {target.shape[-1]}. "
                "Check for stale node embedding caches or mixed ColPali models."
            )
    max_target_tokens = max(int(target.shape[0]) for target in targets)
    if max_target_tokens == 0:
        return torch.zeros((len(targets),), dtype=torch.float32)
    padded_targets = source.new_zeros((len(targets), max_target_tokens, dim))
    valid_mask = torch.zeros((len(targets), max_target_tokens), dtype=torch.bool, device=source.device)
    empty_target_indexes = []
    for target_index, target in enumerate(targets):
        token_count = int(target.shape[0])
        if token_count == 0:
            empty_target_indexes.append(target_index)
            continue
        padded_targets[target_index, :token_count] = target
        valid_mask[target_index, :token_count] = True
    token_scores = torch.matmul(source.unsqueeze(0), padded_targets.transpose(1, 2))
    token_scores = token_scores.masked_fill(~valid_mask.unsqueeze(1), float("-inf"))
    scores = token_scores.max(dim=2).values.mean(dim=1)
    if empty_target_indexes:
        scores[empty_target_indexes] = 0.0
    return scores


def _batched_maxsim_score_matrix(
    sources: list[torch.Tensor],
    targets: list[torch.Tensor],
    *,
    normalize: bool = True,
    device: torch.device | str | None = None,
) -> torch.Tensor:
    if not sources:
        return torch.empty((0, len(targets)), dtype=torch.float32)
    if not targets:
        return torch.empty((len(sources), 0), dtype=torch.float32)
    if normalize:
        sources = [_normalize_tokens(source) for source in sources]
        targets = [_normalize_tokens(target) for target in targets]
    target_device = torch.device(device) if device is not None else sources[0].device
    sources = [source.to(target_device, non_blocking=True) for source in sources]
    targets = [target.to(target_device, non_blocking=True) for target in targets]
    dim = sources[0].shape[-1]
    for vector in [*sources, *targets]:
        if vector.shape[-1] != dim:
            raise ValueError(
                f"Embedding dimension mismatch for MaxSim: source dim {dim}, target dim {vector.shape[-1]}. "
                "Check for stale node embedding caches or mixed ColPali models."
            )
    max_source_tokens = max(int(source.shape[0]) for source in sources)
    max_target_tokens = max(int(target.shape[0]) for target in targets)
    if max_source_tokens == 0 or max_target_tokens == 0:
        return torch.zeros((len(sources), len(targets)), dtype=torch.float32)

    padded_sources = sources[0].new_zeros((len(sources), max_source_tokens, dim))
    source_mask = torch.zeros((len(sources), max_source_tokens), dtype=torch.bool, device=sources[0].device)
    for source_index, source in enumerate(sources):
        token_count = int(source.shape[0])
        if token_count == 0:
            continue
        padded_sources[source_index, :token_count] = source
        source_mask[source_index, :token_count] = True

    padded_targets = sources[0].new_zeros((len(targets), max_target_tokens, dim))
    target_mask = torch.zeros((len(targets), max_target_tokens), dtype=torch.bool, device=sources[0].device)
    for target_index, target in enumerate(targets):
        token_count = int(target.shape[0])
        if token_count == 0:
            continue
        padded_targets[target_index, :token_count] = target
        target_mask[target_index, :token_count] = True

    token_scores = torch.einsum("sld,tmd->sltm", padded_sources, padded_targets)
    token_scores = token_scores.masked_fill(~target_mask.view(1, 1, len(targets), max_target_tokens), float("-inf"))
    max_scores = token_scores.max(dim=3).values
    max_scores = max_scores.masked_fill(~source_mask.view(len(sources), max_source_tokens, 1), 0.0)
    source_token_counts = source_mask.sum(dim=1).clamp_min(1).to(max_scores.dtype)
    scores = max_scores.sum(dim=1) / source_token_counts.unsqueeze(1)
    scores[source_token_counts == 0] = 0.0
    empty_targets = target_mask.sum(dim=1) == 0
    if empty_targets.any():
        scores[:, empty_targets] = 0.0
    return scores.to("cpu")


def _normalize_tokens(vector: torch.Tensor) -> torch.Tensor:
    if vector.numel() == 0:
        return vector.to(torch.float32)
    if vector.ndim == 1:
        vector = vector.unsqueeze(0)
    if vector.ndim != 2:
        raise ValueError(f"Expected token embedding [tokens, dim], got {tuple(vector.shape)}")
    return torch.nn.functional.normalize(vector.to(torch.float32), dim=1)
