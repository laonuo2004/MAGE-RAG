from __future__ import annotations

from pathlib import Path

from baselines.aeg_rag.actions import (
    ActivateNode,
    ActivatePage,
    CandidateAction,
    OpenNode,
    PruneNode,
    SearchEvidence,
    SummarizeNodes,
    action_from_candidate,
)
from baselines.aeg_rag.candidate_generator import CandidateGenerator
from baselines.aeg_rag.evaluator import XMLEvaluator
from baselines.aeg_rag.graph_store import EvidenceGraphStore
from baselines.aeg_rag.renderer import ReaderRenderer
from baselines.aeg_rag.retrieval import ColPaliTop1Retriever
from baselines.aeg_rag.state import EvidenceAgentState
from baselines.base import ContextBuilder, ContextMessages
from benchmarks.utils.document_preprocess import allowed_page_indices
from benchmarks.utils.data_utils import mmlongbench_file_id
from utils.config_utils import get_config_value, require_config_value


class AEGRAGContextBuilder(ContextBuilder):
    name = "aeg-rag"

    def __init__(self, cfg=None):
        super().__init__(cfg)
        self.params = dict(get_config_value(cfg, "baselines.params", {}) or {})
        self.graph_escape = bool(get_config_value(cfg, "baselines.params.graph_escape", False))
        self.evaluator_model_name = str(get_config_value(cfg, "baselines.evaluator.model_name", "Qwen3-VL-8B-Instruct"))
        self.evaluator = XMLEvaluator(
            self.evaluator_model_name,
            temperature=float(get_config_value(cfg, "baselines.evaluator.temperature", 0.0)),
            retries=int(get_config_value(cfg, "baselines.evaluator.retries", 1)),
            raw_text_char_limit=int(get_config_value(cfg, "baselines.evaluator.raw_text_char_limit_per_opened_node", 1200)),
            include_images_for_opened_nodes=bool(
                get_config_value(cfg, "baselines.evaluator.include_images_for_opened_nodes", False)
            ),
            max_candidate_actions=int(get_config_value(cfg, "baselines.agent.max_evaluator_candidate_actions", 120)),
            candidate_preview_char_limit=int(get_config_value(cfg, "baselines.evaluator.candidate_preview_char_limit", 160)),
            max_selected_actions_per_iteration=int(
                get_config_value(cfg, "baselines.agent.max_selected_actions_per_iteration", 4)
            ),
        )
        self.max_evaluator_candidate_actions = int(
            get_config_value(cfg, "baselines.agent.max_evaluator_candidate_actions", 120)
        )
        self.max_selected_actions_per_iteration = max(
            1,
            int(get_config_value(cfg, "baselines.agent.max_selected_actions_per_iteration", 4)),
        )
        self.max_total_selected_actions = max(
            1,
            int(get_config_value(cfg, "baselines.agent.max_total_selected_actions", 24)),
        )
        self.watchdog_iterations = int(get_config_value(cfg, "baselines.safety.watchdog_iterations", 30))
        self.watchdog_repeated_noop_rounds = int(get_config_value(cfg, "baselines.safety.watchdog_repeated_noop_rounds", 3))
        self.run_online_agent = bool(get_config_value(cfg, "baselines.agent.run_online", True))
        self.auto_activate_initial_page_nodes = bool(
            get_config_value(cfg, "baselines.agent.auto_activate_initial_page_nodes", False)
        )
        self.auto_open_initial_page_nodes = bool(
            get_config_value(cfg, "baselines.agent.auto_open_initial_page_nodes", False)
        )
        self.auto_open_max_nodes_per_page = int(
            get_config_value(cfg, "baselines.agent.auto_open_max_nodes_per_page", 24)
        )
        self.auto_open_max_nodes_per_page_longdocurl = int(
            get_config_value(cfg, "baselines.agent.auto_open_max_nodes_per_page_longdocurl", self.auto_open_max_nodes_per_page)
        )
        self.auto_open_max_nodes_per_page_mmlongbench = int(
            get_config_value(cfg, "baselines.agent.auto_open_max_nodes_per_page_mmlongbench", self.auto_open_max_nodes_per_page)
        )
        self.final_open_active_nodes = bool(
            get_config_value(cfg, "baselines.agent.final_open_active_nodes", True)
        )
        self.final_open_active_node_limit = int(
            get_config_value(cfg, "baselines.agent.final_open_active_node_limit", 16)
        )
        self.final_open_active_node_limit_longdocurl = int(
            get_config_value(
                cfg,
                "baselines.agent.final_open_active_node_limit_longdocurl",
                self.final_open_active_node_limit,
            )
        )
        self.final_open_active_node_limit_mmlongbench = int(
            get_config_value(
                cfg,
                "baselines.agent.final_open_active_node_limit_mmlongbench",
                self.final_open_active_node_limit,
            )
        )
        self.retriever = ColPaliTop1Retriever(cfg)

    def build_mmlongbench(self, sample, **kwargs):
        doc_key = mmlongbench_file_id(sample["doc_id"])
        graph_root = get_config_value(self.cfg, "benchmarks.evidence_graph_dir")
        graph_dir = Path(str(graph_root)) / doc_key
        benchmark_cfg = require_config_value(self.cfg, "benchmarks")
        try:
            page_count = self.retriever.embedding_page_count("mmlongbench", sample)
        except Exception:
            page_count = int(get_config_value(benchmark_cfg, "max_pages", 120))
        allowed_pages = allowed_page_indices("mmlongbench", sample, benchmark_cfg, page_count)
        return self._build("mmlongbench", sample, doc_key, graph_dir, allowed_pages, kwargs.get("client"))

    def build_longdocurl(self, sample, **kwargs):
        doc_key = str(sample["doc_no"])
        graph_root = get_config_value(self.cfg, "benchmarks.evidence_graph_dir")
        graph_dir = Path(str(graph_root)) / doc_key
        page_count = int(sample.get("total_pages") or 0)
        if page_count <= 0:
            page_count = max(_page_index_from_image(path) for path in sample.get("images", [])) + 1
        benchmark_cfg = require_config_value(self.cfg, "benchmarks")
        allowed_pages = allowed_page_indices("longdocurl", sample, benchmark_cfg, page_count)
        return self._build("longdocurl", sample, doc_key, graph_dir, allowed_pages, kwargs.get("client"))

    def _build(self, benchmark_name, sample, doc_key, graph_dir, allowed_pages, client):
        graph = EvidenceGraphStore(graph_dir, allowed_pages=allowed_pages)
        state = EvidenceAgentState(graph, graph_escape=self.graph_escape)
        initial_pages, retrieval_metadata = self._initial_pages(benchmark_name, sample, allowed_pages)
        for initial_page in initial_pages:
            state.execute(ActivatePage(page_index=initial_page["page_index"], source="initial_retrieval"), iteration=0)
            if self.run_online_agent:
                continue
            if self.auto_open_initial_page_nodes:
                self._open_initial_page_nodes(
                    state,
                    initial_page["page_index"],
                    sample["question"],
                    self._auto_open_limit_for(benchmark_name),
                )
            elif self.auto_activate_initial_page_nodes:
                self._activate_salient_page_nodes(state, initial_page["page_index"])

        stop_reason = self._run_agent(sample["question"], state, client) if self.run_online_agent else "retrieval_only"
        if self.run_online_agent and self.final_open_active_nodes:
            self._final_open_active_nodes(
                benchmark_name,
                sample["question"],
                state,
            )
        renderer = ReaderRenderer(
            self.cfg,
            include_page_images=bool(get_config_value(self.cfg, "baselines.renderer.include_page_images", True)),
            include_opened_node_images=bool(get_config_value(self.cfg, "baselines.renderer.include_opened_node_images", True)),
            raw_text_limit=int(get_config_value(self.cfg, "baselines.renderer.raw_text_char_limit_per_opened_node", 8192)),
        )
        content = renderer.render(benchmark_name, sample, state)
        metadata = self._metadata(
            state,
            doc_key,
            graph_dir,
            allowed_pages,
            initial_pages[0],
            retrieval_metadata,
            stop_reason,
        )
        return ContextMessages([{"role": "user", "content": content}], metadata=metadata)

    def _activate_salient_page_nodes(self, state: EvidenceAgentState, page_index: int):
        salient_types = {"title", "table", "figure", "image", "chart"}
        for node in state.graph.nodes_on_page(page_index):
            if str(node.get("type") or "").lower() not in salient_types:
                continue
            state.execute(ActivateNode(str(node["id"])), iteration=0)

    def _auto_open_limit_for(self, benchmark_name: str) -> int:
        if benchmark_name == "longdocurl":
            return self.auto_open_max_nodes_per_page_longdocurl
        if benchmark_name == "mmlongbench":
            return self.auto_open_max_nodes_per_page_mmlongbench
        return self.auto_open_max_nodes_per_page

    def _final_open_limit_for(self, benchmark_name: str) -> int:
        if benchmark_name == "longdocurl":
            return self.final_open_active_node_limit_longdocurl
        if benchmark_name == "mmlongbench":
            return self.final_open_active_node_limit_mmlongbench
        return self.final_open_active_node_limit

    def _final_open_active_nodes(self, benchmark_name: str, question: str, state: EvidenceAgentState):
        limit = max(0, int(self._final_open_limit_for(benchmark_name)))
        if limit <= 0:
            return
        candidates = []
        for node_id in state.active_node_ids():
            node = state.graph.node(node_id)
            if state.graph.is_page_node(node):
                continue
            score = _node_question_relevance(node, state.graph.node_text(node_id), question)
            candidates.append((score, node_id))
        candidates.sort(key=lambda item: (-item[0], item[1]))
        for _, node_id in candidates[:limit]:
            result = state.execute(OpenNode(node_id), iteration="final")
            if result.ok:
                state.trace.append({
                    "iteration": "final",
                    "action": "FinalOpenActiveNode",
                    "node_id": node_id,
                })

    def _open_initial_page_nodes(self, state: EvidenceAgentState, page_index: int, question: str, max_nodes: int):
        scored_nodes = []
        for node in state.graph.nodes_on_page(page_index):
            if state.graph.is_page_node(node):
                continue
            score = _node_question_relevance(node, state.graph.node_text(str(node["id"])), question)
            if score <= 0:
                continue
            scored_nodes.append((score, str(node["id"])))
        scored_nodes.sort(key=lambda item: (-item[0], item[1]))
        for _, node_id in scored_nodes[: int(max_nodes)]:
            activate_result = state.execute(ActivateNode(node_id), iteration=0)
            if activate_result.ok:
                state.execute(OpenNode(node_id), iteration=0)

    def _initial_pages(self, benchmark_name, sample, allowed_pages):
        try:
            return self.retriever.retrieve_many(benchmark_name, sample, allowed_pages)
        except Exception as exc:
            page_index = int(allowed_pages[0])
            return (
                [{"page_index": page_index, "page_number": page_index + 1, "score": None}],
                {"retrieval_error": str(exc), "retrieved_pages": [], "embedding_paths": {}},
            )

    def _run_agent(self, question: str, state: EvidenceAgentState, client) -> str:
        generator = CandidateGenerator(state.graph)
        repeated_noop_rounds = 0
        selected_action_attempts = 0
        if client is None:
            return "fallback_no_client"
        for iteration in range(1, self.watchdog_iterations + 1):
            candidates = self._select_evaluator_candidates(question, state, generator.generate(state))
            candidate_by_id = {candidate.id: candidate for candidate in candidates}
            candidate_by_index = {index: candidate for index, candidate in enumerate(candidates, start=1)}
            made_progress = False
            try:
                evaluator_input = self.evaluator.trace_input(question, state, candidates)
                decision, raw_response = self.evaluator.call(client, question, state, candidates)
            except Exception as exc:
                state.validation_errors.append({"action_type": "Evaluator", "message": str(exc)})
                return "fallback_invalid_xml"
            state.trace.append({
                "iteration": iteration,
                "action": "EvaluatorDecision",
                "evaluator_input": evaluator_input,
                "raw_response": raw_response,
                "decision": decision.__dict__,
                "candidate_ids": sorted(candidate_by_id),
                "candidate_index_map": {
                    str(index): candidate.id
                    for index, candidate in candidate_by_index.items()
                },
                "selected_action_execution_limit": self.max_selected_actions_per_iteration,
                "selected_action_total_limit": self.max_total_selected_actions,
            })

            if decision.stop and not any([
                decision.selected_actions,
                decision.search_query,
                decision.prune_requests,
                decision.summarize_requests,
            ]):
                return "normal_stop"

            selected_actions = list(decision.selected_actions or [])
            if len(selected_actions) > self.max_selected_actions_per_iteration:
                state.trace.append({
                    "iteration": iteration,
                    "action": "TruncatedSelectedActions",
                    "original_count": len(selected_actions),
                    "executed_count": self.max_selected_actions_per_iteration,
                })
                selected_actions = selected_actions[: self.max_selected_actions_per_iteration]

            for selected in selected_actions:
                selected_action_attempts += 1
                candidate_id = selected.get("candidate_id")
                candidate_index = selected.get("candidate_index")
                candidate = _resolve_candidate_index(candidate_index, candidate_by_index)
                if candidate is None:
                    candidate = _resolve_candidate(candidate_id, candidate_by_id)
                if candidate is None:
                    candidate = _candidate_from_selected_alias(candidate_id, state)
                if candidate is None:
                    if candidate_index is not None and not candidate_by_index and not candidate_id:
                        state.trace.append({
                            "iteration": iteration,
                            "action": "IgnoredEmptyCandidateSelection",
                            "candidate_index": candidate_index,
                        })
                        continue
                    state.validation_errors.append({
                        "action_type": "SelectedAction",
                        "message": f"Invalid candidate selection: index={candidate_index} candidate_id={candidate_id}",
                    })
                    state.trace.append({
                        "iteration": iteration,
                        "action": "InvalidCandidate",
                        "candidate_index": candidate_index,
                        "candidate_id": candidate_id,
                    })
                    continue
                result = state.execute(action_from_candidate(candidate), iteration=iteration)
                if (
                    not result.ok
                    and candidate.action_type == "ActivateNode"
                    and "activate page" in result.message
                ):
                    node_id = str(candidate.payload["node_id"])
                    page_index = state.graph.node_page_index(node_id)
                    page_result = state.execute(ActivatePage(page_index, "relation_target"), iteration=iteration)
                    if page_result.ok:
                        result = state.execute(action_from_candidate(candidate), iteration=iteration)
                made_progress = made_progress or result.ok
            if selected_action_attempts >= self.max_total_selected_actions:
                state.trace.append({
                    "iteration": iteration,
                    "action": "SelectedActionBudgetReached",
                    "executed_total": selected_action_attempts,
                    "limit": self.max_total_selected_actions,
                })
                return "watchdog_total_selected_actions"
            if decision.search_query:
                result = state.execute(SearchEvidence(decision.search_query), iteration=iteration)
                made_progress = made_progress or result.ok
            for request in decision.prune_requests:
                result = state.execute(PruneNode(request["node_id"], request.get("reason", "")), iteration=iteration)
                made_progress = made_progress or result.ok
            for request in decision.summarize_requests:
                result = state.execute(
                    SummarizeNodes(request.get("source_node_ids") or [], request.get("goal", "")),
                    iteration=iteration,
                )
                made_progress = made_progress or result.ok

            if not made_progress:
                repeated_noop_rounds += 1
                if repeated_noop_rounds >= self.watchdog_repeated_noop_rounds:
                    return "watchdog_repeated_noop"
            else:
                repeated_noop_rounds = 0
        return "watchdog_iterations"

    def _select_evaluator_candidates(self, question: str, state: EvidenceAgentState, candidates: list) -> list:
        limit = max(1, int(self.max_evaluator_candidate_actions))
        if len(candidates) <= limit:
            return candidates
        indexed = list(enumerate(candidates))
        indexed.sort(
            key=lambda item: (
                -_candidate_question_relevance(item[1], state, question),
                item[0],
            )
        )
        return [candidate for _, candidate in indexed[:limit]]

    def _metadata(self, state, doc_key, graph_dir, allowed_pages, initial_page, retrieval_metadata, stop_reason):
        return {
            "context_builder": self.name,
            "params": self.params,
            "allowed_pages": list(allowed_pages),
            "graph_dir": str(graph_dir),
            "doc_key": doc_key,
            "initial_retrieval": {
                "initial_page": initial_page,
                **retrieval_metadata,
            },
            "final_node_states": state.final_node_states(),
            "active_node_ids": state.active_node_ids(),
            "opened_node_ids": state.opened_node_ids(),
            "pruned_node_ids": state.pruned_node_ids(),
            "summary_artifacts": list(state.summaries),
            "iteration_trace": list(state.trace),
            "stop_reason": stop_reason,
            "validation_errors": list(state.validation_errors),
            "evaluator_model_name": self.evaluator_model_name,
            "run_online_agent": self.run_online_agent,
            "max_evaluator_candidate_actions": self.max_evaluator_candidate_actions,
            "max_selected_actions_per_iteration": self.max_selected_actions_per_iteration,
            "max_total_selected_actions": self.max_total_selected_actions,
            "graph_escape": self.graph_escape,
        }


def _page_index_from_image(image_path) -> int:
    import os
    import re

    match = re.search(r"_(\d+)\.[^.]+$", os.path.basename(str(image_path)))
    if not match:
        raise ValueError(f"Cannot parse page index from image path: {image_path}")
    return int(match.group(1))


def _node_question_relevance(node: dict, node_text: str, question: str) -> int:
    node_type = str(node.get("type") or "").lower()
    question_text = str(question or "").lower()
    haystack = " ".join([
        str(node.get("abstract") or ""),
        str(node.get("title") or ""),
        str(node.get("caption") or ""),
        str(node_text or ""),
    ]).lower()
    score = 0
    if node_type in {"title", "table", "figure", "image", "chart"}:
        score += 8
    if node_type in {"paragraph", "text", "list"}:
        score += 3
    type_markers = {
        "table": ("table", "tabular", "row", "column"),
        "chart": ("chart", "axis", "line", "bar", "color", "percentage", "percent"),
        "figure": ("figure", "fig.", "image", "map", "diagram", "color"),
        "image": ("figure", "image", "map", "diagram", "color"),
        "title": ("title", "section", "heading", "select titles", "which section", "where can we find"),
    }
    for marker in type_markers.get(node_type, ()):
        if marker in question_text:
            score += 12
            break
    question_terms = {
        term.strip(".,:;!?()[]{}\"'")
        for term in question_text.split()
        if len(term.strip(".,:;!?()[]{}\"'")) >= 4
    }
    if question_terms:
        score += min(10, sum(1 for term in question_terms if term in haystack))
    return score


def _candidate_question_relevance(candidate, state: EvidenceAgentState, question: str) -> int:
    score = 0
    action_type = str(candidate.action_type)
    if action_type in {"OpenNode", "FollowRelation"}:
        score += 20
    elif action_type == "ActivateNode":
        score += 10
    elif action_type == "ActivatePage":
        score += 4
    payload = candidate.payload or {}
    node_id = str(payload.get("node_id") or "")
    if node_id and node_id in state.graph.nodes:
        node = state.graph.node(node_id)
        score += _node_question_relevance(node, state.graph.node_text(node_id), question)
    preview = str(candidate.preview or "").lower()
    question_terms = {
        term.strip(".,:;!?()[]{}\"'")
        for term in str(question or "").lower().split()
        if len(term.strip(".,:;!?()[]{}\"'")) >= 4
    }
    score += min(20, sum(2 for term in question_terms if term in preview))
    return score


def _resolve_candidate(candidate_id, candidate_by_id):
    candidate_id = str(candidate_id or "")
    if candidate_id in candidate_by_id:
        return candidate_by_id[candidate_id]
    aliases = {candidate_id}
    if candidate_id.startswith("act:"):
        without_prefix = candidate_id[len("act:"):]
        aliases.add(without_prefix)
        parts = without_prefix.split(":", 1)
        if len(parts) == 2:
            aliases.add(parts[1])
    else:
        aliases.add(f"act:{candidate_id}")
    for candidate in candidate_by_id.values():
        payload = candidate.payload
        target_values = {
            str(payload.get("node_id") or ""),
            str(payload.get("edge_id") or ""),
            str(payload.get("page_index") or ""),
        }
        readable_suffixes = {
            f'{candidate.action_type}:{value}'
            for value in target_values
            if value
        }
        if aliases & target_values or aliases & readable_suffixes:
            return candidate
    return None


def _resolve_candidate_index(candidate_index, candidate_by_index):
    if candidate_index is None:
        return None
    try:
        index = int(candidate_index)
    except (TypeError, ValueError):
        return None
    return candidate_by_index.get(index)


def _candidate_from_selected_alias(candidate_id, state: EvidenceAgentState):
    candidate_id = str(candidate_id or "")
    if candidate_id in state.graph.edges:
        return CandidateAction(
            id=f"act:FollowRelation:{candidate_id}",
            action_type="FollowRelation",
            payload={"edge_id": candidate_id},
            preview=state.graph.preview_node(str(state.graph.edge(candidate_id).get("target") or "")),
        )
    if candidate_id.startswith("act:FollowRelation:"):
        edge_id = candidate_id[len("act:FollowRelation:"):]
        if edge_id in state.graph.edges:
            return CandidateAction(
                id=candidate_id,
                action_type="FollowRelation",
                payload={"edge_id": edge_id},
                preview=state.graph.preview_node(str(state.graph.edge(edge_id).get("target") or "")),
            )
    if candidate_id in state.graph.nodes:
        action_type = "OpenNode" if state.state_of(candidate_id) == "Active" else "ActivateNode"
        page_index = state.graph.node_page_index(candidate_id)
        if not state.graph.is_page_allowed(page_index, state.graph_escape):
            return None
        return CandidateAction(
            id=f"act:{action_type}:{candidate_id}",
            action_type=action_type,
            payload={"node_id": candidate_id},
            preview=state.graph.preview_node(candidate_id),
        )
    for action_type in ("ActivateNode", "OpenNode"):
        prefix = f"act:{action_type}:"
        if not candidate_id.startswith(prefix):
            continue
        node_id = candidate_id[len(prefix):]
        if node_id not in state.graph.nodes:
            node_id = _nearest_node_alias(node_id, state)
        if node_id is None:
            return None
        page_index = state.graph.node_page_index(node_id)
        if not state.graph.is_page_allowed(page_index, state.graph_escape):
            return None
        return CandidateAction(
            id=candidate_id,
            action_type=action_type,
            payload={"node_id": node_id},
            preview=state.graph.preview_node(node_id),
        )
    return None


def _nearest_node_alias(node_id: str, state: EvidenceAgentState) -> str | None:
    parsed = _parse_graph_node_id(node_id)
    if parsed is None:
        return None
    prefix, page_index, block_index, node_type = parsed
    if not state.graph.is_page_allowed(page_index, state.graph_escape):
        return None
    page_nodes = state.graph.nodes_on_page(page_index)
    if not page_nodes:
        return None
    same_prefix = [node for node in page_nodes if str(node.get("id") or "").startswith(prefix)]
    if not same_prefix:
        same_prefix = page_nodes
    typed = [
        node for node in same_prefix
        if str(node.get("type") or "").lower() == node_type.lower()
        or str(node.get("id") or "").lower().endswith(f":{node_type.lower()}")
    ]
    candidates = typed or same_prefix

    def sort_key(node):
        candidate_id = str(node.get("id") or "")
        candidate_block = _block_index(candidate_id)
        distance = abs(candidate_block - block_index) if candidate_block is not None else 10_000
        state_rank = 0 if state.state_of(candidate_id) == "Active" else 1
        return (distance, state_rank, candidate_id)

    return str(sorted(candidates, key=sort_key)[0].get("id"))


def _parse_graph_node_id(node_id: str) -> tuple[str, int, int, str] | None:
    import re

    match = re.match(r"^(?P<prefix>.+:page:(?P<page>\d+):)block:(?P<block>\d+):(?P<type>[^:]+)$", str(node_id))
    if not match:
        return None
    return (
        match.group("prefix"),
        int(match.group("page")),
        int(match.group("block")),
        match.group("type"),
    )


def _block_index(node_id: str) -> int | None:
    import re

    match = re.search(r":block:(\d+):", str(node_id))
    return int(match.group(1)) if match else None
