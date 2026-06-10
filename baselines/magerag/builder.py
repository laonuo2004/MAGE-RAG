from __future__ import annotations

import logging
from pathlib import Path

from baselines.magerag.actions import (
    ActivateNode,
    ActivatePage,
    CandidateAction,
    OpenNode,
    PruneNode,
    SearchEvidence,
    action_from_candidate,
)
from baselines.magerag.candidate_generator import CandidateGenerator
from baselines.magerag.evaluator import XMLEvaluator
from baselines.magerag.graph_store import EvidenceGraphStore
from baselines.magerag.renderer import ReaderRenderer
from baselines.magerag.retrieval import ColPaliTop1Retriever
from baselines.magerag.state import EvidenceAgentState
from baselines.base import ContextBuilder, ContextMessages, build_context_summary, build_logical_cost, build_retrieval_metadata
from benchmarks.utils.document_preprocess import allowed_page_indices
from benchmarks.utils.data_utils import mmlongbench_file_id
from utils.config_utils import get_config_value, require_config_value

logger = logging.getLogger(__name__)
class MAGERAGContextBuilder(ContextBuilder):
    """
    把 benchmark sample 转成最终 reader messages 的 MAGE-RAG 编排器。

    对应方法图中的 online pipeline：Stage I 初始页面定位，Stage II 迭代扩展证据图，
    Stage III 把 evidence subgraph 渲染给 LVLM reader 回答。
    """

    name = "magerag"

    def __init__(self, cfg=None):
        super().__init__(cfg)
        if get_config_value(cfg, "baselines.agent", None) is not None:
            raise ValueError("MAGE-RAG config contains legacy config sections: baselines.agent")
        self.params = dict(get_config_value(cfg, "baselines.params", {}) or {})
        self.evaluator_model_name = str(get_config_value(cfg, "baselines.models.evaluator", "Qwen3-VL-8B-Instruct"))
        self.evaluator = XMLEvaluator(
            self.evaluator_model_name,
            temperature=get_config_value(cfg, "baselines.evaluator.temperature", 0.0),
            retries=get_config_value(cfg, "baselines.evaluator.retries", 2),
            raw_text_char_limit=get_config_value(cfg, "baselines.evaluator.raw_text_char_limit", 1200),
            include_images_for_opened_nodes=get_config_value(cfg, "baselines.evaluator.include_opened_node_images", False),
            candidate_preview_char_limit=get_config_value(cfg, "baselines.evaluator.candidate_preview_char_limit", 160),
            max_selected_actions_per_iteration=get_config_value(cfg, "baselines.evaluator.max_selected_actions_per_iteration", 4),
            include_few_shot_examples=get_config_value(cfg, "baselines.evaluator.include_few_shot_examples", True),
            recent_trace_limit=get_config_value(cfg, "baselines.evaluator.recent_trace_limit", 25),
        )
        self.max_selected_actions_per_iteration = max(
            1,
            int(get_config_value(cfg, "baselines.evaluator.max_selected_actions_per_iteration", 4)),
        )
        self.max_total_selected_actions = max(
            1,
            int(get_config_value(cfg, "baselines.evaluator.max_total_selected_actions", 24)),
        )
        self.controller_mode = str(get_config_value(cfg, "baselines.controller.mode", "full"))
        self.enable_online_controller = bool(get_config_value(cfg, "baselines.controller.enable_online_controller", True))
        self.enable_search = bool(get_config_value(cfg, "baselines.controller.enable_search", True))
        self.enable_prune = bool(get_config_value(cfg, "baselines.controller.enable_prune", True))
        self.watchdog_iterations = max(0, int(get_config_value(cfg, "baselines.controller.watchdog_iterations", 6)))
        self.watchdog_repeated_noop_rounds = max(1, int(get_config_value(cfg, "baselines.controller.watchdog_repeated_noop_rounds", 2)))
        self.auto_activate_initial_page_nodes = True
        self.auto_open_initial_page_nodes = True
        self.final_open_active_nodes = True
        self.final_open_active_node_limit = int(get_config_value(cfg, "baselines.controller.final_open_active_node_limit", 16))
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
            logger.warning("Page count invalid. Using image paths to estimate page count for longdocurl sample %s", doc_key)
            page_count = max(_page_index_from_image(path) for path in sample.get("images", [])) + 1
        benchmark_cfg = require_config_value(self.cfg, "benchmarks")
        allowed_pages = allowed_page_indices("longdocurl", sample, benchmark_cfg, page_count)
        return self._build("longdocurl", sample, doc_key, graph_dir, allowed_pages, kwargs.get("client"))

    def _build(self, benchmark_name, sample, doc_key, graph_dir, allowed_pages, client):
        graph = EvidenceGraphStore(graph_dir, allowed_pages=allowed_pages)
        state = EvidenceAgentState(graph)

        # Stage I: ColPali 做 page-level grounding，先把最相关页面放入工作记忆。
        initial_pages, retrieval_metadata = self._initial_pages(benchmark_name, sample, allowed_pages)
        for initial_page in initial_pages:
            state.execute(ActivatePage(page_index=initial_page["page_index"], source="initial_retrieval"), iteration=0)

        # Stage II: evaluator 反复选择有边际收益的扩展动作，直到 stop 或 watchdog 触发。
        stop_reason = self._run_agent(benchmark_name, sample["question"], state, client, sample=sample)
        if self.final_open_active_nodes:
            # 最后一轮把仍处于 Active 的高相关节点打开，避免 reader 只看到 abstract。
            self._final_open_active_nodes(
                benchmark_name,
                sample["question"],
                state,
            )
        renderer = ReaderRenderer(
            self.cfg,
            include_page_images=get_config_value(self.cfg, "baselines.reader.include_page_images", True),
            include_opened_node_images=get_config_value(self.cfg, "baselines.reader.include_opened_node_images", True),
        )
        # Stage III: evidence graph state -> LVLM reader messages，同时保留 trace 供分析插件复盘。
        content = renderer.render(benchmark_name, sample, state)
        reader_input = renderer.trace_input(benchmark_name, sample, state, content)
        metadata = self._metadata(
            state,
            doc_key,
            graph_dir,
            allowed_pages,
            initial_pages[0],
            retrieval_metadata,
            stop_reason,
            reader_input,
        )
        return ContextMessages([{"role": "user", "content": content}], metadata=metadata)

    def _activate_salient_page_nodes(self, state: EvidenceAgentState, page_index: int):
        # 只激活视觉/结构性强的节点，作为不运行 online agent 时的轻量 evidence expansion。
        salient_types = {"title", "table", "figure", "image", "chart"}
        for node in state.graph.nodes_on_page(page_index):
            if str(node.get("type") or "").lower() not in salient_types:
                continue
            state.execute(ActivateNode(str(node["id"])), iteration=0)

    def _final_open_active_nodes(self, benchmark_name: str, question: str, state: EvidenceAgentState):
        limit = max(0, int(self.final_open_active_node_limit))
        if limit <= 0:
            return
        opened_count = 0
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
                opened_count += 1
                state.trace.append({
                    "iteration": "final",
                    "action": "FinalOpenActiveNode",
                    "node_id": node_id,
                })
        remaining = max(0, limit - opened_count)
        if remaining <= 0:
            return
        # 如果 evaluator 通过搜索/关系跳转激活了新页面，但还没来得及打开其中节点，
        # final pass 会按问题相关性补开少量页面内证据。
        active_page_indices = [
            state.graph.node_page_index(node_id)
            for node_id in state.active_node_ids()
            if state.graph.is_page_node(state.graph.node(node_id))
        ]
        seen_node_ids = set(state.opened_node_ids())
        for page_index in sorted(dict.fromkeys(active_page_indices)):
            if not _active_page_has_non_initial_source(state, page_index):
                continue
            if remaining <= 0:
                break
            for node_id in _question_page_node_ids(state, page_index, question, remaining):
                if node_id in seen_node_ids:
                    continue
                activate_result = state.execute(ActivateNode(node_id), iteration="final")
                if not activate_result.ok:
                    continue
                open_result = state.execute(OpenNode(node_id), iteration="final")
                if not open_result.ok:
                    continue
                seen_node_ids.add(node_id)
                remaining -= 1
                state.trace.append({
                    "iteration": "final",
                    "action": "FinalOpenActivePageNode",
                    "page_index": page_index,
                    "node_id": node_id,
                })
                if remaining <= 0:
                    break

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

    def _open_question_page_nodes(
        self,
        state: EvidenceAgentState,
        page_index: int,
        question: str,
        max_nodes: int,
        preferred_node_ids: list[str] | None = None,
    ) -> list[str]:
        opened_node_ids = []
        ordered_node_ids = []
        for node_id in preferred_node_ids or []:
            if node_id not in ordered_node_ids:
                ordered_node_ids.append(node_id)
        for node_id in _question_page_node_ids(state, page_index, question, max_nodes):
            if node_id not in ordered_node_ids:
                ordered_node_ids.append(node_id)
        for node_id in ordered_node_ids[: int(max_nodes)]:
            activate_result = state.execute(ActivateNode(node_id), iteration=0)
            if not activate_result.ok:
                continue
            open_result = state.execute(OpenNode(node_id), iteration=0)
            if open_result.ok:
                opened_node_ids.append(node_id)
        return opened_node_ids

    def _initial_pages(self, benchmark_name, sample, allowed_pages):
        try:
            return self.retriever.retrieve_many(benchmark_name, sample, allowed_pages)
        except Exception as exc:
            # 检索 embedding 缺失时仍保留可运行路径；metadata 会记录错误，便于后续排查数据准备问题。
            page_index = int(allowed_pages[0])
            return (
                [{"page_index": page_index, "page_number": page_index + 1, "score": None}],
                {"retrieval_error": str(exc), "retrieved_pages": [], "embedding_paths": {}},
            )

    def _run_agent(
        self,
        benchmark_name: str,
        question: str,
        state: EvidenceAgentState,
        client=None,
        sample: dict | None = None,
    ) -> str:
        generator = CandidateGenerator(state.graph)
        repeated_noop_rounds = 0
        selected_action_attempts = 0
        if client is None:
            return "fallback_no_client"
        if not self.enable_online_controller or self.watchdog_iterations <= 0:
            return "controller_disabled"
        for iteration in range(1, self.watchdog_iterations + 1):
            # 每轮都重新枚举候选，因为上轮动作可能激活页面、打开节点或引入搜索结果。
            candidates = generator.generate(state)
            indexed_candidates = [candidate for candidate in candidates if candidate.action_type == "ActivateNode"]
            candidate_by_index = {index: candidate for index, candidate in enumerate(indexed_candidates, start=1)}
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
                "candidate_ids": sorted(candidate.id for candidate in candidates),
                "candidate_index_map": {
                    str(index): candidate.id
                    for index, candidate in candidate_by_index.items()
                },
                "selected_action_execution_limit": self.max_selected_actions_per_iteration,
                "selected_action_total_limit": self.max_total_selected_actions,
                "state_snapshot_before": state.snapshot(),
            })

            if decision.stop and not any([
                decision.selected_actions,
                decision.open_requests,
                decision.search_query,
                decision.prune_requests,
            ]):
                # 只有“stop 且没有任何待执行动作”才真正结束，避免模型一边要求 stop 一边还给动作。
                return "normal_stop"

            selected_actions = list(decision.selected_actions or [])
            if len(selected_actions) > self.max_selected_actions_per_iteration:
                # 限制单轮动作数，防止 evaluator 一次性扩展过宽导致 reader 上下文被噪声淹没。
                state.trace.append({
                    "iteration": iteration,
                    "action": "TruncatedSelectedActions",
                    "original_count": len(selected_actions),
                    "executed_count": self.max_selected_actions_per_iteration,
                })
                selected_actions = selected_actions[: self.max_selected_actions_per_iteration]

            for selected in selected_actions:
                selected_action_attempts += 1
                candidate_index = selected.get("candidate_index")
                candidate = _resolve_candidate_index(candidate_index, candidate_by_index)
                if candidate is None:
                    state.validation_errors.append({
                        "action_type": "SelectedAction",
                        "message": f"Invalid candidate selection index: {candidate_index}",
                    })
                    state.trace.append({
                        "iteration": iteration,
                        "action": "InvalidCandidate",
                        "candidate_index": candidate_index,
                    })
                    continue
                result = state.execute(action_from_candidate(candidate), iteration=iteration)
                self._annotate_last_action_selection(state, selected, candidate, resolved_by="candidate_index")
                made_progress = made_progress or result.ok
            if selected_action_attempts >= self.max_total_selected_actions:
                # 总预算是防 runaway 的硬上限；正常情况应由 stop 或 no-op watchdog 结束。
                state.trace.append({
                    "iteration": iteration,
                    "action": "SelectedActionBudgetReached",
                    "executed_total": selected_action_attempts,
                    "limit": self.max_total_selected_actions,
                })
                return "watchdog_total_selected_actions"
            for request in decision.open_requests:
                node_id = str(request.get("node_id") or "")
                result = state.execute(OpenNode(node_id), iteration=iteration)
                made_progress = made_progress or result.ok
            if decision.search_query and self.enable_search:
                # SearchEvidence 对应方法图里的 Jump：根据当前证据缺口重新找入口页/节点。
                result = state.execute(SearchEvidence(decision.search_query), iteration=iteration)
                made_progress = made_progress or result.ok
                if result.ok:
                    activated_pages = self._activate_search_pages(
                        benchmark_name,
                        sample,
                        decision.search_query,
                        state,
                        iteration=iteration,
                    )
                    made_progress = made_progress or bool(activated_pages)
            elif decision.search_query:
                state.trace.append({
                    "iteration": iteration,
                    "action": "SkippedSearchEvidence",
                    "query": str(decision.search_query),
                    "reason": "search_disabled",
                })
            for request in (decision.prune_requests if self.enable_prune else []):
                # Prune 不删除节点，只把它从当前证据上下文降权，并保留原因供 trace 分析。
                result = state.execute(PruneNode(request["node_id"], request.get("reason", "")), iteration=iteration)
                made_progress = made_progress or result.ok
            if decision.prune_requests and not self.enable_prune:
                state.trace.append({
                    "iteration": iteration,
                    "action": "SkippedPruneRequests",
                    "count": len(decision.prune_requests),
                    "reason": "prune_disabled",
                })

            if not made_progress:
                repeated_noop_rounds += 1
                if repeated_noop_rounds >= self.watchdog_repeated_noop_rounds:
                    # 连续 no-op 表示 evaluator 给出的动作无法改变状态，继续循环只会浪费调用。
                    return "watchdog_repeated_noop"
            else:
                repeated_noop_rounds = 0
        return "watchdog_iterations"

    def _activate_search_pages(
        self,
        benchmark_name: str,
        sample: dict | None,
        query: str,
        state: EvidenceAgentState,
        iteration,
    ) -> list[int]:
        if sample is None:
            state.validation_errors.append({"action_type": "SearchEvidence", "message": "SearchEvidence requires sample metadata."})
            return []
        active_pages = {
            state.graph.node_page_index(node_id)
            for node_id in state.active_node_ids() + state.opened_node_ids()
            if state.graph.is_page_node(state.graph.node(node_id))
        }
        allowed_pages = sorted(state.graph.allowed_pages or state.graph.page_nodes)
        try:
            pages, metadata = self.retriever.retrieve_from_query(
                benchmark_name,
                sample,
                query,
                allowed_pages,
                excluded_pages=active_pages,
            )
        except Exception as exc:
            state.validation_errors.append({"action_type": "SearchEvidence", "message": str(exc)})
            return []
        activated_pages: list[int] = []
        for page in pages:
            result = state.execute(ActivatePage(page_index=page["page_index"], source="search"), iteration=iteration)
            if result.ok:
                activated_pages.append(int(page["page_index"]))
        state.trace.append({
            "iteration": iteration,
            "action": "SearchEvidenceRetrieval",
            "query": str(query),
            "retrieved_pages": pages,
            "activated_pages": activated_pages,
            "metadata": metadata,
        })
        return activated_pages

    def _annotate_last_action_selection(
        self,
        state: EvidenceAgentState,
        selected: dict,
        candidate: CandidateAction,
        resolved_by: str,
    ):
        if not state.trace:
            return
        state.trace[-1]["selection"] = {
            "candidate_index": selected.get("candidate_index"),
            "resolved_candidate_id": candidate.id,
            "resolved_by": resolved_by,
            "utility": selected.get("utility") or "",
            "candidate_action_type": candidate.action_type,
            "candidate_payload": dict(candidate.payload),
            "candidate_preview": candidate.preview,
        }

    def _metadata(self, state, doc_key, graph_dir, allowed_pages, initial_page, retrieval_metadata, stop_reason, reader_input):
        # metadata 是实验分析的主要入口，尽量保存决策 trace 和 reader 输入引用，而不是只保存最终 prompt。
        activated_pages = sorted(
            state.graph.node_page_index(node_id)
            for node_id, node_state in state.final_node_states().items()
            if node_state in {"Active", "Opened"} and state.graph.is_page_node(state.graph.node(node_id))
        )
        initial_retrieved_pages = [
            int(page["page_index"])
            for page in retrieval_metadata.get("retrieved_pages", [])
            if page.get("page_index") is not None
        ]
        if not initial_retrieved_pages and initial_page:
            initial_retrieved_pages = [int(initial_page["page_index"])]
        retrieval_items = [
            {
                "rank": rank,
                "page_index": int(page["page_index"]),
                "page_number": int(page.get("page_number", int(page["page_index"]) + 1)),
                "score": page.get("score"),
            }
            for rank, page in enumerate(retrieval_metadata.get("retrieved_pages", []) or [initial_page], start=1)
            if page and page.get("page_index") is not None
        ]
        opened_node_ids = state.opened_node_ids()
        active_node_ids = state.active_node_ids()
        trace = list(state.trace)
        graph_stats = self._graph_stats(state)
        trace_summary = self._trace_summary(trace, stop_reason)
        context_page_ids = sorted(set(activated_pages))
        node_ids = sorted(set(active_node_ids + opened_node_ids))
        reader_text_chars = sum(len(str(part)) for part in reader_input.get("text_parts", []))
        reader_image_refs = reader_input.get("image_refs", [])
        return {
            "context_builder": self.name,
            "params": self.params,
            "allowed_pages": list(allowed_pages),
            "activated_pages": activated_pages,
            "graph_dir": str(graph_dir),
            "doc_key": doc_key,
            "initial_retrieval": {
                "initial_page": initial_page,
                **retrieval_metadata,
            },
            "reader_input": reader_input,
            "final_node_states": state.final_node_states(),
            "active_node_ids": active_node_ids,
            "opened_node_ids": opened_node_ids,
            "pruned_node_ids": state.pruned_node_ids(),
            "iteration_trace": trace,
            "stop_reason": stop_reason,
            "validation_errors": list(state.validation_errors),
            "evaluator_model_name": self.evaluator_model_name,
            "max_selected_actions_per_iteration": self.max_selected_actions_per_iteration,
            "max_total_selected_actions": self.max_total_selected_actions,
            "retrieval": build_retrieval_metadata(
                retrieved_items=retrieval_items,
                initial_retrieved_pages=initial_retrieved_pages,
                final_context_pages=context_page_ids,
            ),
            "context_summary": build_context_summary(
                page_ids=context_page_ids,
                node_ids=node_ids,
                node_types=[
                    str(state.graph.node(node_id).get("type") or "")
                    for node_id in node_ids
                    if node_id in state.graph.nodes
                ],
                num_context_pages=len(context_page_ids),
                num_context_nodes=len(node_ids),
                num_text_units=len(reader_input.get("text_parts", [])),
                num_image_units=len(reader_image_refs),
                num_text_chars=reader_text_chars,
            ),
            "logical_cost": build_logical_cost(
                num_llm_calls=trace_summary["num_evaluator_calls"],
                num_retriever_calls=1 + trace_summary["num_search_retrievals"],
                num_input_text_chars=reader_text_chars,
                num_input_images=len(reader_image_refs),
                num_context_pages=len(context_page_ids),
                num_context_nodes=len(node_ids),
                num_retrieved_pages=len(initial_retrieved_pages),
                num_final_evidence_units=len(context_page_ids) + len(opened_node_ids),
            ),
            "magerag": {
                "graph_stats": graph_stats,
                "initial_retrieval": {
                    "initial_page": initial_page,
                    **retrieval_metadata,
                },
                "state_summary": state.snapshot(),
                "trace_summary": trace_summary,
                "iteration_trace": trace,
                "controller": {
                    "mode": self.controller_mode,
                    "enable_online_controller": self.enable_online_controller,
                    "enable_search": self.enable_search,
                    "enable_prune": self.enable_prune,
                    "watchdog_iterations": self.watchdog_iterations,
                    "watchdog_repeated_noop_rounds": self.watchdog_repeated_noop_rounds,
                    "final_open_active_node_limit": self.final_open_active_node_limit,
                },
            },
        }

    def _graph_stats(self, state: EvidenceAgentState) -> dict:
        node_type_counts = {}
        edge_type_counts = {}
        for node in state.graph.nodes.values():
            node_type = str(node.get("type") or "unknown")
            node_type_counts[node_type] = node_type_counts.get(node_type, 0) + 1
        for edge in state.graph.edges.values():
            edge_type = str(edge.get("type") or "unknown")
            edge_type_counts[edge_type] = edge_type_counts.get(edge_type, 0) + 1
        return {
            "num_nodes": len(state.graph.nodes),
            "num_edges": len(state.graph.edges),
            "num_allowed_pages": None if state.graph.allowed_pages is None else len(state.graph.allowed_pages),
            "num_page_nodes": len(state.graph.page_nodes),
            "node_type_counts": dict(sorted(node_type_counts.items())),
            "edge_type_counts": dict(sorted(edge_type_counts.items())),
        }

    def _trace_summary(self, trace: list[dict], stop_reason: str) -> dict:
        action_counts = {}
        iterations = set()
        for item in trace:
            action = str(item.get("action") or "")
            action_counts[action] = action_counts.get(action, 0) + 1
            iteration = item.get("iteration")
            if isinstance(iteration, int) and iteration > 0:
                iterations.add(iteration)
        return {
            "stop_reason": stop_reason,
            "num_iterations": len(iterations),
            "num_trace_events": len(trace),
            "num_evaluator_calls": action_counts.get("EvaluatorDecision", 0),
            "num_search_requests": action_counts.get("SearchEvidence", 0),
            "num_search_retrievals": action_counts.get("SearchEvidenceRetrieval", 0),
            "num_activate_page": action_counts.get("ActivatePage", 0),
            "num_activate_node": action_counts.get("ActivateNode", 0),
            "num_open_node": action_counts.get("OpenNode", 0),
            "num_prune_node": action_counts.get("PruneNode", 0),
            "action_counts": dict(sorted(action_counts.items())),
        }

def _page_index_from_image(image_path) -> int:
    import os
    import re

    match = re.search(r"_(\d+)\.[^.]+$", os.path.basename(str(image_path)))
    if not match:
        raise ValueError(f"Cannot parse page index from image path: {image_path}")
    return int(match.group(1))


def _active_page_has_non_initial_source(state: EvidenceAgentState, page_index: int) -> bool:
    for item in state.trace:
        if item.get("action") != "ActivatePage" or not item.get("ok"):
            continue
        payload = item.get("payload") or {}
        if int(payload.get("page_index", -1)) != int(page_index):
            continue
        if str(payload.get("source") or "") in {"search", "question_page_scope", "relation_target"}:
            return True
    return False


def _question_page_node_ids(
    state: EvidenceAgentState,
    page_index: int,
    question: str,
    max_nodes: int,
) -> list[str]:
    # 页面内自动打开优先考虑题目词重叠和视觉/结构节点；没有命中时才退回普通节点。
    typed_nodes = []
    fallback_nodes = []
    type_priority = {
        "table": 0,
        "chart": 1,
        "figure": 2,
        "image": 3,
        "title": 4,
        "paragraph": 5,
        "text": 5,
    }
    for node in state.graph.nodes_on_page(page_index):
        if state.graph.is_page_node(node):
            continue
        node_id = str(node["id"])
        node_type = str(node.get("type") or "").lower()
        score = _node_question_relevance(node, state.graph.node_text(node_id), question)
        rank = type_priority.get(node_type, 9)
        item = (-score, rank, node_id)
        fallback_nodes.append(item)
        if score > 0 or rank <= 4:
            typed_nodes.append(item)
    chosen = typed_nodes or fallback_nodes
    chosen.sort()
    return [node_id for _, _, node_id in chosen[: int(max_nodes)]]


def _node_question_relevance(node: dict, node_text: str, question: str) -> int:
    node_type = str(node.get("type") or "").lower()
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
    question_terms = {
        term.strip(".,:;!?()[]{}\"'")
        for term in str(question or "").lower().split()
        if len(term.strip(".,:;!?()[]{}\"'")) >= 4
    }
    if question_terms:
        score += min(10, sum(1 for term in question_terms if term in haystack))
    return score


def _resolve_candidate_index(candidate_index, candidate_by_index):
    if candidate_index is None:
        return None
    try:
        index = int(candidate_index)
    except (TypeError, ValueError):
        return None
    return candidate_by_index.get(index)
