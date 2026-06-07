from __future__ import annotations

import re
from pathlib import Path

from baselines.magerag.actions import (
    ActivateNode,
    ActivatePage,
    CandidateAction,
    OpenNode,
    PruneNode,
    SearchEvidence,
    SummarizeNodes,
    action_from_candidate,
)
from baselines.magerag.candidate_generator import CandidateGenerator
from baselines.magerag.evaluator import XMLEvaluator
from baselines.magerag.graph_store import EvidenceGraphStore
from baselines.magerag.renderer import ReaderRenderer
from baselines.magerag.retrieval import ColPaliTop1Retriever
from baselines.magerag.state import EvidenceAgentState
from baselines.base import ContextBuilder, ContextMessages
from benchmarks.utils.document_preprocess import allowed_page_indices
from benchmarks.utils.data_utils import mmlongbench_file_id
from utils.config_utils import get_config_value, require_config_value


LEGACY_CONFIG_SECTIONS = ("agent", "renderer", "safety")


class MAGERAGContextBuilder(ContextBuilder):
    """
    把 benchmark sample 转成最终 reader messages 的 MAGE-RAG 编排器。

    对应方法图中的 online pipeline：Stage I 初始页面定位，Stage II 迭代扩展证据图，
    Stage III 把 evidence subgraph 渲染给 LVLM reader 回答。
    """

    name = "magerag"

    def __init__(self, cfg=None):
        super().__init__(cfg)
        _reject_legacy_config_sections(cfg)
        self.params = dict(get_config_value(cfg, "baselines.params", {}) or {})
        self.evaluator_model_name = str(get_config_value(cfg, "baselines.models.evaluator", "Qwen3-VL-8B-Instruct"))
        self.evaluator = XMLEvaluator(
            self.evaluator_model_name,
            temperature=get_config_value(cfg, "baselines.evaluator.temperature", 0.0),
            retries=get_config_value(cfg, "baselines.evaluator.retries", 2),
            raw_text_char_limit=get_config_value(cfg, "baselines.evaluator.raw_text_char_limit", 1200),
            include_images_for_opened_nodes=get_config_value(cfg, "baselines.evaluator.include_opened_node_images", False),
            max_candidate_actions=get_config_value(cfg, "baselines.evaluator.max_candidate_actions", 120),
            candidate_preview_char_limit=get_config_value(cfg, "baselines.evaluator.candidate_preview_char_limit", 160),
            max_selected_actions_per_iteration=get_config_value(cfg, "baselines.evaluator.max_selected_actions_per_iteration", 4),
            prompt_style=get_config_value(cfg, "baselines.evaluator.prompt_style", "structured"),
            include_few_shot_examples=get_config_value(cfg, "baselines.evaluator.include_few_shot_examples", True),
            reason_max_words=get_config_value(cfg, "baselines.evaluator.reason_max_words", 30),
        )
        self.max_evaluator_candidate_actions = int(get_config_value(cfg, "baselines.evaluator.max_candidate_actions", 120))
        self.max_selected_actions_per_iteration = max(
            1,
            int(get_config_value(cfg, "baselines.evaluator.max_selected_actions_per_iteration", 4)),
        )
        self.max_total_selected_actions = max(
            1,
            int(get_config_value(cfg, "baselines.evaluator.max_total_selected_actions", 24)),
        )
        self.watchdog_iterations = max(1, int(get_config_value(cfg, "baselines.controller.watchdog_iterations", 6)))
        self.watchdog_repeated_noop_rounds = max(1, int(get_config_value(cfg, "baselines.controller.watchdog_repeated_noop_rounds", 2)))
        self.auto_activate_initial_page_nodes = True
        self.auto_open_initial_page_nodes = True
        self.auto_open_max_nodes_per_page = int(get_config_value(cfg, "baselines.controller.auto_open_max_nodes_per_page", 24))
        self.auto_open_max_nodes_per_page_longdocurl = int(
            get_config_value(cfg, "baselines.controller.auto_open_max_nodes_per_page_longdocurl", self.auto_open_max_nodes_per_page)
        )
        self.auto_open_max_nodes_per_page_mmlongbench = int(
            get_config_value(cfg, "baselines.controller.auto_open_max_nodes_per_page_mmlongbench", self.auto_open_max_nodes_per_page)
        )
        self.final_open_active_nodes = True
        self.final_open_active_node_limit = int(get_config_value(cfg, "baselines.controller.final_open_active_node_limit", 16))
        self.final_open_active_node_limit_longdocurl = int(
            get_config_value(cfg, "baselines.controller.final_open_active_node_limit_longdocurl", self.final_open_active_node_limit)
        )
        self.final_open_active_node_limit_mmlongbench = int(
            get_config_value(cfg, "baselines.controller.final_open_active_node_limit_mmlongbench", self.final_open_active_node_limit)
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
        state = EvidenceAgentState(graph)

        # Stage I: ColPali 做 page-level grounding，先把最相关页面放入工作记忆。
        initial_pages, retrieval_metadata = self._initial_pages(benchmark_name, sample, allowed_pages)
        for initial_page in initial_pages:
            state.execute(ActivatePage(page_index=initial_page["page_index"], source="initial_retrieval"), iteration=0)

        # 题目显式指定 page/figure/table/section 时，不完全依赖 top-1 retrieval。
        # 这类 scope hint 更像人类直接翻到被点名的位置。
        self._auto_open_question_pages(
            benchmark_name,
            sample["question"],
            state,
            allowed_pages,
        )
        self._auto_open_named_question_scopes(
            benchmark_name,
            sample["question"],
            state,
            allowed_pages,
        )
        self._run_auto_searches(benchmark_name, sample["question"], state)

        # Stage II: evaluator 反复选择有边际收益的扩展动作，直到 stop 或 watchdog 触发。
        stop_reason = self._run_agent(benchmark_name, sample["question"], state, client)
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
            raw_text_limit=get_config_value(self.cfg, "baselines.reader.raw_text_char_limit", 8192),
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
                if not activate_result.ok and "activate page" in activate_result.message:
                    state.execute(ActivatePage(page_index=page_index, source="relation_target"), iteration="final")
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

    def _auto_open_question_pages(
        self,
        benchmark_name: str,
        question: str,
        state: EvidenceAgentState,
        allowed_pages,
    ):
        page_indices = _question_page_indices(question)
        if not page_indices:
            return
        allowed_page_set = set(int(page) for page in allowed_pages)
        max_nodes = max(1, min(int(self._auto_open_limit_for(benchmark_name)), 8))
        for page_index in page_indices:
            if page_index not in allowed_page_set:
                # 记录失败而不是静默跳过，方便分析“题目要求页不在 allowed_pages”这类 benchmark 约束问题。
                state.trace.append({
                    "iteration": 0,
                    "action": "AutoOpenQuestionPage",
                    "page_index": page_index,
                    "ok": False,
                    "message": "requested page is outside allowed_pages",
                    "opened_node_ids": [],
                })
                continue
            page_result = state.execute(ActivatePage(page_index=page_index, source="question_page_scope"), iteration=0)
            opened_node_ids = self._open_question_page_nodes(state, page_index, question, max_nodes)
            state.trace.append({
                "iteration": 0,
                "action": "AutoOpenQuestionPage",
                "page_index": page_index,
                "ok": page_result.ok,
                "opened_node_ids": opened_node_ids,
            })

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

    def _auto_open_named_question_scopes(
        self,
        benchmark_name: str,
        question: str,
        state: EvidenceAgentState,
        allowed_pages,
    ):
        specs = _question_named_scope_specs(question)
        if not specs:
            return
        allowed_page_set = set(int(page) for page in allowed_pages)
        explicit_page_indices = [
            page_index for page_index in _question_page_indices(question)
            if page_index in allowed_page_set
        ]
        max_scopes = 8
        max_nodes = max(1, min(int(self._auto_open_limit_for(benchmark_name)), 8))
        opened_scopes = 0
        seen_pages: set[int] = set()
        for spec in specs:
            matches = _matching_named_scope_nodes(state, spec, allowed_page_set)
            if not matches:
                state.trace.append({
                    "iteration": 0,
                    "action": "AutoOpenNamedQuestionScope",
                    "kind": spec["kind"],
                    "label": spec["label"],
                    "ok": False,
                    "message": "no matching evidence nodes",
                    "opened_node_ids": [],
                })
                continue
            for node_id in matches:
                page_index = state.graph.node_page_index(node_id)
                if page_index in seen_pages and spec["kind"] in {"figure", "table"}:
                    continue
                seen_pages.add(page_index)
                scope_pages = [page_index]
                if spec["kind"] in {"section", "chapter", "quoted_scope", "faq"}:
                    # section/FAQ 类问题通常跨多个页面；figure/table 则尽量只打开命中的局部页面。
                    if spec["kind"] == "quoted_scope" and explicit_page_indices:
                        scope_pages = explicit_page_indices
                    else:
                        scope_pages = _adjacent_scope_page_indices(
                            page_index,
                            allowed_page_set,
                            max_pages=4 if spec["kind"] == "faq" else 12,
                        )
                opened_node_ids = []
                page_result = None
                for scope_page_index in scope_pages:
                    page_result = state.execute(
                        ActivatePage(page_index=scope_page_index, source="question_page_scope"),
                        iteration=0,
                    )
                    preferred = [node_id] if scope_page_index == page_index else []
                    opened_node_ids.extend(
                        self._open_question_page_nodes(
                            state,
                            scope_page_index,
                            question,
                            max_nodes,
                            preferred_node_ids=preferred,
                        )
                    )
                state.trace.append({
                    "iteration": 0,
                    "action": "AutoOpenNamedQuestionScope",
                    "kind": spec["kind"],
                    "label": spec["label"],
                    "page_index": page_index,
                    "scope_page_indices": scope_pages,
                    "ok": bool(page_result and page_result.ok),
                    "opened_node_ids": opened_node_ids,
                })
                opened_scopes += 1
                if opened_scopes >= max_scopes:
                    return

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

    def _run_auto_searches(self, benchmark_name: str, question: str, state: EvidenceAgentState):
        queries = _auto_search_queries(benchmark_name, question)
        if not queries:
            return
        # 自动搜索只在 sample/benchmark 允许的页面范围内引入额外候选入口。
        merged = []
        seen = set()
        for query in queries:
            results = state.graph.search(query, limit=12)
            for item in results:
                node_id = str(item["node"]["id"])
                if node_id in seen:
                    continue
                seen.add(node_id)
                merged.append(item)
            state.trace.append({
                "iteration": 0,
                "action": "AutoSearchEvidence",
                "query": query,
                "result_count": len(results),
                "result_node_ids": [str(item["node"]["id"]) for item in results],
            })
        if merged:
            state.search_results = merged[:24]

    def _run_agent(
        self,
        benchmark_name: str,
        question: str | EvidenceAgentState | None = None,
        state: EvidenceAgentState | None = None,
        client=None,
    ) -> str:
        if isinstance(question, EvidenceAgentState):
            client = state if client is None else client
            state = question
            question = benchmark_name
            benchmark_name = "mmlongbench"
        if state is None:
            raise TypeError("_run_agent requires an EvidenceAgentState")
        question = str(question or "")
        generator = CandidateGenerator(state.graph)
        repeated_noop_rounds = 0
        selected_action_attempts = 0
        if client is None:
            return "fallback_no_client"
        for iteration in range(1, self.watchdog_iterations + 1):
            # 每轮都重新枚举候选，因为上轮动作可能激活页面、打开节点或引入搜索结果。
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
                "state_snapshot_before": state.snapshot(),
            })

            if decision.stop and not any([
                decision.selected_actions,
                decision.search_query,
                decision.prune_requests,
                decision.summarize_requests,
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
                self._annotate_last_action_selection(state, selected, candidate, resolved_by="candidate_index_or_id")
                if (
                    not result.ok
                    and candidate.action_type == "ActivateNode"
                    and "activate page" in result.message
                ):
                    # LLM 可能选择了页面内节点，但父 page 尚未 Active；这里自动补一次 page activation。
                    node_id = str(candidate.payload["node_id"])
                    page_index = state.graph.node_page_index(node_id)
                    page_result = state.execute(ActivatePage(page_index, "relation_target"), iteration=iteration)
                    if page_result.ok:
                        result = state.execute(action_from_candidate(candidate), iteration=iteration)
                        self._annotate_last_action_selection(state, selected, candidate, resolved_by="activate_parent_page_retry")
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
            if decision.search_query:
                # SearchEvidence 对应方法图里的 Jump：根据当前证据缺口重新找入口页/节点。
                result = state.execute(SearchEvidence(decision.search_query), iteration=iteration)
                made_progress = made_progress or result.ok
                if result.ok:
                    opened_node_ids = self._auto_open_search_results(
                        benchmark_name,
                        question,
                        state,
                        state.search_results,
                        iteration=iteration,
                    )
                    made_progress = made_progress or bool(opened_node_ids)
            for request in decision.prune_requests:
                # Prune 不删除节点，只把它从当前证据上下文降权，并保留原因供 trace 分析。
                result = state.execute(PruneNode(request["node_id"], request.get("reason", "")), iteration=iteration)
                made_progress = made_progress or result.ok
            for request in decision.summarize_requests:
                # Summarize 是压缩工作记忆的接口，当前实现为确定性摘要 artifact。
                result = state.execute(
                    SummarizeNodes(request.get("source_node_ids") or [], request.get("goal", "")),
                    iteration=iteration,
                )
                made_progress = made_progress or result.ok

            if not made_progress:
                repeated_noop_rounds += 1
                if repeated_noop_rounds >= self.watchdog_repeated_noop_rounds:
                    # 连续 no-op 表示 evaluator 给出的动作无法改变状态，继续循环只会浪费调用。
                    return "watchdog_repeated_noop"
            else:
                repeated_noop_rounds = 0
        return "watchdog_iterations"

    def _auto_open_search_results(
        self,
        benchmark_name: str,
        question: str,
        state: EvidenceAgentState,
        search_results: list[dict],
        iteration,
    ) -> list[str]:
        if not search_results:
            return []
        # Search 后主动打开少量高相关结果，减少 evaluator 下一轮只看到搜索 preview 的盲区。
        max_nodes = max(1, min(int(self._auto_open_limit_for(benchmark_name)), 4))
        opened_node_ids: list[str] = []
        seen_pages: set[int] = set()
        for item in search_results:
            if len(opened_node_ids) >= max_nodes:
                break
            node = item.get("node") or {}
            node_id = str(node.get("id") or "")
            if not node_id or node_id not in state.graph.nodes:
                continue
            page_index = state.graph.node_page_index(node_id)
            page_result = state.execute(ActivatePage(page_index=page_index, source="search"), iteration=iteration)
            if not page_result.ok:
                continue
            preferred = [] if state.graph.is_page_node(state.graph.node(node_id)) else [node_id]
            per_page_limit = max(1, max_nodes - len(opened_node_ids))
            page_opened = self._open_question_page_nodes(
                state,
                page_index,
                question,
                per_page_limit,
                preferred_node_ids=preferred,
            )
            opened_node_ids.extend(page_opened)
            if page_opened:
                seen_pages.add(page_index)
                state.trace.append({
                    "iteration": iteration,
                    "action": "AutoOpenSearchResult",
                    "page_index": page_index,
                    "node_id": node_id,
                    "opened_node_ids": page_opened,
                })
            if len(seen_pages) >= 2:
                break
        return opened_node_ids

    def _select_evaluator_candidates(self, question: str, state: EvidenceAgentState, candidates: list) -> list:
        limit = max(1, int(self.max_evaluator_candidate_actions))
        if len(candidates) <= limit:
            return candidates
        # 候选过多时先按问题相关性截断；candidate id 仍保持稳定，不依赖截断后的序号。
        indexed = list(enumerate(candidates))
        indexed.sort(
            key=lambda item: (
                -_candidate_question_relevance(item[1], state, question),
                item[0],
            )
        )
        return [candidate for _, candidate in indexed[:limit]]

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
            "candidate_id": selected.get("candidate_id") or candidate.id,
            "resolved_candidate_id": candidate.id,
            "resolved_by": resolved_by,
            "utility": selected.get("utility") or "",
            "reason": selected.get("reason") or "",
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
            "active_node_ids": state.active_node_ids(),
            "opened_node_ids": state.opened_node_ids(),
            "pruned_node_ids": state.pruned_node_ids(),
            "summary_artifacts": list(state.summaries),
            "iteration_trace": list(state.trace),
            "stop_reason": stop_reason,
            "validation_errors": list(state.validation_errors),
            "evaluator_model_name": self.evaluator_model_name,
            "max_evaluator_candidate_actions": self.max_evaluator_candidate_actions,
            "max_selected_actions_per_iteration": self.max_selected_actions_per_iteration,
            "max_total_selected_actions": self.max_total_selected_actions,
        }


def _reject_legacy_config_sections(cfg) -> None:
    legacy = [
        section
        for section in LEGACY_CONFIG_SECTIONS
        if get_config_value(cfg, f"baselines.{section}", None) is not None
    ]
    if legacy:
        joined = ", ".join(f"baselines.{section}" for section in legacy)
        raise ValueError(
            f"MAGE-RAG no longer supports legacy config sections: {joined}. "
            "Use baselines.params, baselines.models, baselines.evaluator, baselines.controller, and baselines.reader instead."
        )


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


def _question_page_indices(question: str, max_pages_to_open: int = 16) -> list[int]:
    # 把自然语言中的 1-based page/slide 引用转成内部 0-based page_index。
    text = str(question or "")
    indices: list[int] = []
    for match in re.finditer(
        r"\b(?:pages?|slides?)\s+(\d+(?:\s*[-–—]\s*\d+)?(?:\s*(?:,|and|&)\s*\d+(?:\s*[-–—]\s*\d+)?)*)",
        text,
        flags=re.IGNORECASE,
    ):
        page_spec = re.sub(r"\s+(?:and|&)\s+", ",", match.group(1), flags=re.IGNORECASE)
        for part in [item.strip() for item in page_spec.split(",") if item.strip()]:
            range_match = re.fullmatch(r"(\d+)(?:\s*[-–—]\s*(\d+))?", part)
            if not range_match:
                continue
            start = int(range_match.group(1))
            end = int(range_match.group(2) or start)
            if start > end:
                start, end = end, start
            for page_number in range(start, end + 1):
                if page_number <= 0:
                    continue
                indices.append(page_number - 1)
                if len(indices) >= max_pages_to_open:
                    break
            if len(indices) >= max_pages_to_open:
                break
        if len(indices) >= max_pages_to_open:
            break
    return list(dict.fromkeys(indices))


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


def _question_named_scope_specs(question: str) -> list[dict[str, str]]:
    # 抽取题目显式点名的 figure/table/section/FAQ 等 scope，作为检索之外的强导航信号。
    text = str(question or "")
    specs: list[dict[str, str]] = []
    for match in re.finditer(
        r"\b(?P<kind>fig(?:ure)?\.?|table)\s+(?P<label>[A-Za-z0-9]+(?:[.\-]\d+)*)\b",
        text,
        flags=re.IGNORECASE,
    ):
        kind = "figure" if match.group("kind").lower().startswith("fig") else "table"
        if not _valid_figure_table_label(match.group("label")):
            continue
        specs.append({"kind": kind, "label": match.group("label")})
    if re.search(r"\bfaqs?\b|\bfrequently asked questions\b", text, flags=re.IGNORECASE):
        specs.append({"kind": "faq", "label": "frequently asked questions"})
    for match in re.finditer(r"\bsection\s+(\d+(?:\.\d+)*)\b", text, flags=re.IGNORECASE):
        specs.append({"kind": "section", "label": match.group(1)})
    if re.search(r"\bappendix\b", text, flags=re.IGNORECASE):
        specs.append({"kind": "section", "label": "Appendix"})
    for match in re.finditer(r"[\u201c\u201d\"']([^\"'\u201c\u201d]{4,90})[\u201c\u201d\"']", text):
        label = " ".join(match.group(1).split())
        before = text[max(0, match.start() - 40):match.start()].lower()
        after = text[match.end():match.end() + 40].lower()
        if any(marker in before + " " + after for marker in ("section", "chapter", "appendix", "faq", "title", "table", "figure")):
            specs.append({"kind": "quoted_scope", "label": label})
    for match in re.finditer(
        r"\b(?P<kind>section|chapter|faq)\s+(?P<label>[A-Z][A-Za-z0-9&/.,'()\- ]{3,80})",
        text,
        flags=re.IGNORECASE,
    ):
        label = re.split(r"\s+(?:has|have|contains?|include|includes?|where|with|in|on|from|of)\b", match.group("label"), maxsplit=1)[0]
        label = " ".join(label.strip(" .,:;?").split())
        if len(label) >= 4:
            specs.append({"kind": match.group("kind").lower(), "label": label})
    deduped = []
    seen = set()
    for spec in specs:
        key = (spec["kind"], spec["label"].lower())
        if key in seen:
            continue
        seen.add(key)
        deduped.append(spec)
    return deduped


def _valid_figure_table_label(label: str) -> bool:
    raw = str(label or "").strip().strip(".")
    text = raw.lower()
    if not text:
        return False
    if text in {"in", "on", "of", "the", "an", "left", "right", "top", "bottom", "below", "above"}:
        return False
    if text == "a" and raw != "A":
        return False
    return bool(re.fullmatch(r"\d+(?:[.\-]\d+)*|[a-z]|[ivxlcdm]+", text))


def _matching_named_scope_nodes(
    state: EvidenceAgentState,
    spec: dict[str, str],
    allowed_page_set: set[int],
) -> list[str]:
    # 在 evidence graph 中寻找命名 scope 的候选节点，返回的是节点入口而不是最终答案。
    kind = str(spec.get("kind") or "")
    label = str(spec.get("label") or "")
    scored = []
    for node_id, node in state.graph.nodes.items():
        page_index = state.graph.node_page_index(node)
        if page_index not in allowed_page_set:
            continue
        if state.graph.is_page_node(node):
            continue
        node_type = str(node.get("type") or "").lower()
        text = " ".join([
            str(node.get("abstract") or ""),
            str(node.get("title") or ""),
            str(node.get("caption") or ""),
            state.graph.node_text(node_id),
        ])
        if not _node_matches_named_scope(kind, label, node_type, text):
            continue
        type_rank = 0 if node_type in {"title", "table", "figure", "image", "chart"} else 1
        relevance = _scope_match_relevance(label, text)
        scored.append((-relevance, type_rank, page_index, str(node_id)))
    scored.sort(key=lambda item: (item[0], item[1], item[2], item[3]))
    return [node_id for _, _, _, node_id in scored[:8]]


def _node_matches_named_scope(kind: str, label: str, node_type: str, text: str) -> bool:
    normalized = _normalize_scope_text(text)
    normalized_label = _normalize_scope_text(label)
    if not normalized_label:
        return False
    if kind in {"figure", "table"}:
        label_variants = {normalized_label}
        roman_value = _roman_to_int(normalized_label)
        if roman_value is not None:
            label_variants.add(str(roman_value))
        for value in label_variants:
            if re.search(rf"\b(?:{kind}|{'fig' if kind == 'figure' else 'tbl'})\s*\.?\s*{re.escape(value)}\b", normalized):
                return True
        if node_type in ({kind, "image", "chart"} if kind == "figure" else {kind, "chart"}):
            return any(re.search(rf"\b{re.escape(value)}\b", normalized) for value in label_variants)
        return False
    return normalized_label in normalized


def _adjacent_scope_page_indices(
    page_index: int,
    allowed_page_set: set[int],
    max_pages: int,
) -> list[int]:
    page_indices = []
    for offset in range(0, max(1, int(max_pages))):
        scoped_page = int(page_index) + offset
        if scoped_page not in allowed_page_set:
            continue
        page_indices.append(scoped_page)
    return page_indices


def _normalize_scope_text(value: str) -> str:
    return " ".join(str(value or "").lower().replace("_", " ").split())


def _scope_match_relevance(label: str, text: str) -> int:
    normalized_label = _normalize_scope_text(label)
    normalized_text = _normalize_scope_text(text)
    score = normalized_text.count(normalized_label) * 10
    for term in normalized_label.split():
        if len(term) >= 4:
            score += normalized_text.count(term)
    return score


def _roman_to_int(value: str) -> int | None:
    text = str(value or "").strip().lower()
    if not re.fullmatch(r"[ivxlcdm]+", text):
        return None
    values = {"i": 1, "v": 5, "x": 10, "l": 50, "c": 100, "d": 500, "m": 1000}
    total = 0
    previous = 0
    for char in reversed(text):
        current = values[char]
        if current < previous:
            total -= current
        else:
            total += current
            previous = current
    return total if total > 0 else None


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
    # 候选排序只用于截断 evaluator 输入，不代表最终动作选择；最终选择仍由 evaluator 决定。
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


def _auto_search_queries(benchmark_name: str, question: str) -> list[str]:
    # 目前只为 MMLongBench 的显式页/slide 问题生成自动搜索，避免无差别扩大上下文。
    if benchmark_name != "mmlongbench":
        return []
    text = _normalized_question_text(question)
    queries = []
    if re.search(r"\bpages?\s*\d|\bslides?\s*\d|\bpage\s+range\b|\bslide\s+range\b", text):
        scope_terms = re.findall(r"\b(?:pages?|slides?)\s*\d+(?:\s*[-–—]\s*\d+)?", text)
        query = " ".join(scope_terms + [term for term in ("table", "figure", "chart", "image") if term in text])
        if query.strip():
            queries.append(query.strip())
    return list(dict.fromkeys(query for query in queries if query))


def _normalized_question_text(question: str) -> str:
    return " ".join(str(question or "").lower().replace("_", " ").split())


def _resolve_candidate(candidate_id, candidate_by_id):
    # Evaluator 理应返回 action index；这些 alias 兼容用于恢复偶发的 node_id/edge_id 直接引用。
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
    # 当候选列表被截断或模型引用了旧候选 ID 时，尝试从当前 graph state 重建一个等价候选。
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
        if not state.graph.is_page_allowed(page_index):
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
        if not state.graph.is_page_allowed(page_index):
            return None
        return CandidateAction(
            id=candidate_id,
            action_type=action_type,
            payload={"node_id": node_id},
            preview=state.graph.preview_node(node_id),
        )
    return None


def _nearest_node_alias(node_id: str, state: EvidenceAgentState) -> str | None:
    # 解析器/模型有时会保留旧 block id；按同页、同类型、相近 block 近似映射到现有节点。
    parsed = _parse_graph_node_id(node_id)
    if parsed is None:
        return None
    prefix, page_index, block_index, node_type = parsed
    if not state.graph.is_page_allowed(page_index):
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
