import json
import os
import tempfile
import unittest
import xml.etree.ElementTree as ET
from pathlib import Path

from omegaconf import OmegaConf

from baselines.magerag.actions import CandidateAction
from baselines.magerag.actions import ActivateNode, ActivatePage, FollowRelation, OpenNode, PruneNode, SearchEvidence
from baselines.magerag.builder import MAGERAGContextBuilder
from baselines.magerag.builder import _candidate_from_selected_alias
from baselines.magerag.builder import _question_named_scope_specs
from baselines.magerag.builder import _question_page_indices
from baselines.magerag.builder import _resolve_candidate
from baselines.magerag.candidate_generator import CandidateGenerator
from baselines.magerag.evaluator import EvaluatorDecision, XMLEvaluator, parse_agent_decision_xml
from baselines.magerag.graph_store import EvidenceGraphStore
from baselines.magerag.retrieval import ColPaliTop1Retriever
from baselines.magerag.renderer import ReaderRenderer
from baselines.magerag.state import ACTIVE, OPENED, PRUNED, EvidenceAgentState
from baselines.base import ContextMessages
from baselines.wrapper import build_context_builder
from benchmarks.adapters import MMLongBenchAdapter


class MAGERAGTests(unittest.TestCase):
    def test_build_context_builder_routes_magerag(self):
        builder = build_context_builder(OmegaConf.create({"baselines": {"name": "magerag"}}))

        self.assertEqual(builder.name, "magerag")

    def test_build_context_builder_rejects_old_aeg_rag_name(self):
        with self.assertRaisesRegex(ValueError, "Unsupported context_builder: aeg-rag"):
            build_context_builder(OmegaConf.create({"baselines": {"name": "aeg-rag"}}))

    def test_build_context_builder_rejects_delimited_mage_names(self):
        for old_name in ("mage-rag", "mage_rag"):
            with self.subTest(old_name=old_name):
                with self.assertRaisesRegex(ValueError, f"Unsupported context_builder: {old_name}"):
                    build_context_builder(OmegaConf.create({"baselines": {"name": old_name}}))

    def test_builder_rejects_legacy_mage_config_sections(self):
        with self.assertRaisesRegex(ValueError, "legacy config sections: baselines.agent"):
            MAGERAGContextBuilder(OmegaConf.create({
                "baselines": {
                    "name": "magerag",
                    "agent": {"run_online": True},
                }
            }))

    def test_retriever_uses_single_params_top_k_for_all_benchmarks(self):
        retriever = ColPaliTop1Retriever(OmegaConf.create({
            "baselines": {
                "params": {"top_k": 7}
            }
        }))

        self.assertEqual(retriever.top_k_for("longdocurl"), 7)
        self.assertEqual(retriever.top_k_for("mmlongbench"), 7)

    def test_default_mage_config_uses_single_fair_top_k(self):
        cfg = OmegaConf.load("configs/baselines/magerag.yaml")

        retriever = ColPaliTop1Retriever(OmegaConf.create({"baselines": cfg}))

        self.assertEqual(retriever.top_k_for("longdocurl"), 5)
        self.assertEqual(retriever.top_k_for("mmlongbench"), 5)

    def test_default_mage_config_is_compact_and_uses_new_schema(self):
        cfg = OmegaConf.load("configs/baselines/magerag.yaml")

        self.assertEqual(str(cfg.name), "magerag")
        self.assertEqual(int(cfg.params.top_k), 5)
        self.assertNotIn("graph_escape", cfg.params)
        self.assertNotIn("online_agent", cfg.params)
        self.assertEqual(set(cfg.params.keys()), {"top_k"})
        self.assertEqual(str(cfg.models.evaluator), "Qwen3-VL-8B-Instruct")
        self.assertEqual(str(cfg.evaluator.prompt_style), "structured")
        self.assertTrue(bool(cfg.evaluator.include_few_shot_examples))
        self.assertEqual(int(cfg.evaluator.reason_max_words), 30)
        self.assertEqual(int(cfg.evaluator.raw_text_char_limit), 1200)
        self.assertEqual(int(cfg.evaluator.max_candidate_actions), 120)
        self.assertEqual(int(cfg.evaluator.max_selected_actions_per_iteration), 4)
        self.assertEqual(int(cfg.evaluator.max_total_selected_actions), 24)
        self.assertEqual(int(cfg.controller.watchdog_iterations), 6)
        self.assertEqual(int(cfg.controller.watchdog_repeated_noop_rounds), 2)
        self.assertEqual(int(cfg.controller.auto_open_max_nodes_per_page), 24)
        self.assertEqual(int(cfg.controller.final_open_active_node_limit), 16)
        self.assertEqual(str(cfg.reader.mode), "compact")
        self.assertEqual(str(cfg.reader.prompt_style), "structured")
        self.assertEqual(str(cfg.reader.not_answerable_text), "Not answerable.")
        self.assertTrue(bool(cfg.reader.include_self_check_instruction))
        self.assertTrue(bool(cfg.reader.include_image_page_labels))
        self.assertNotIn("mmlongbench_prompt", cfg.reader)
        self.assertNotIn("mmlongbench_include_image_page_labels", cfg.reader)
        self.assertNotIn("agent", cfg)
        self.assertIn("evaluator", cfg)
        self.assertNotIn("safety", cfg)

    def test_graph_store_loads_synthetic_artifacts(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            graph_dir = Path(self._write_graph(tmp_dir))
            graph = EvidenceGraphStore(graph_dir, allowed_pages=[0])

        self.assertIn("n1", graph.nodes)
        self.assertIn("e1", graph.edges)
        self.assertEqual(graph.parent_page_node_id("n1"), "page:0")

    def test_activation_outside_allowed_pages_is_blocked(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            state = EvidenceAgentState(EvidenceGraphStore(self._write_graph(tmp_dir), allowed_pages=[0]))

            result = state.execute(ActivatePage(1, "search"))

        self.assertFalse(result.ok)
        self.assertIn("outside allowed_pages", result.message)

    def test_activate_page_activates_allowed_page(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            state = EvidenceAgentState(EvidenceGraphStore(self._write_graph(tmp_dir), allowed_pages=[0]))

            result = state.execute(ActivatePage(0, "initial_retrieval"))

        self.assertTrue(result.ok)
        self.assertEqual(state.state_of("page:0"), ACTIVE)

    def test_activate_node_requires_parent_page_active(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            state = EvidenceAgentState(EvidenceGraphStore(self._write_graph(tmp_dir), allowed_pages=[0]))

            result = state.execute(ActivateNode("n1"))

        self.assertFalse(result.ok)
        self.assertIn("activate page", result.message)

    def test_activate_node_blocks_node_outside_allowed_pages(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            state = EvidenceAgentState(EvidenceGraphStore(self._write_graph(tmp_dir), allowed_pages=[0]))

            result = state.execute(ActivateNode("n2"))

        self.assertFalse(result.ok)
        self.assertIn("outside allowed_pages", result.message)

    def test_open_node_rejects_inactive_and_pruned_nodes(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            state = EvidenceAgentState(EvidenceGraphStore(self._write_graph(tmp_dir), allowed_pages=[0]))

            inactive_result = state.execute(OpenNode("n1"))
            state.execute(ActivatePage(0, "initial_retrieval"))
            state.execute(ActivateNode("n1"))
            state.execute(PruneNode("n1", "irrelevant"))
            pruned_result = state.execute(OpenNode("n1"))

        self.assertFalse(inactive_result.ok)
        self.assertFalse(pruned_result.ok)
        self.assertEqual(state.state_of("n1"), PRUNED)

    def test_open_node_duplicate_is_noop_without_validation_error(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            state = EvidenceAgentState(EvidenceGraphStore(self._write_graph(tmp_dir), allowed_pages=[0]))
            state.execute(ActivatePage(0, "initial_retrieval"))
            state.execute(ActivateNode("n1"))
            first_result = state.execute(OpenNode("n1"))

            duplicate_result = state.execute(OpenNode("n1"))

        self.assertTrue(first_result.ok)
        self.assertFalse(duplicate_result.ok)
        self.assertEqual(state.state_of("n1"), OPENED)
        self.assertTrue(duplicate_result.payload["already_opened"])
        self.assertEqual(state.validation_errors, [])

    def test_pruned_to_active_works_through_activate_node(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            state = EvidenceAgentState(EvidenceGraphStore(self._write_graph(tmp_dir), allowed_pages=[0]))
            state.execute(ActivatePage(0, "initial_retrieval"))
            state.execute(ActivateNode("n1"))
            state.execute(PruneNode("n1", "irrelevant"))

            result = state.execute(ActivateNode("n1"))

        self.assertTrue(result.ok)
        self.assertTrue(result.payload["reactivated_from_pruned"])
        self.assertEqual(state.state_of("n1"), ACTIVE)

    def test_follow_relation_exposes_target_preview_without_opening_target(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            state = EvidenceAgentState(EvidenceGraphStore(self._write_graph(tmp_dir), allowed_pages=[0, 1]))
            state.execute(ActivatePage(0, "initial_retrieval"))
            state.execute(ActivateNode("n1"))

            result = state.execute(FollowRelation("e1"))

        self.assertTrue(result.ok)
        self.assertEqual(result.payload["target_id"], "n2")
        self.assertIn("needle target", result.payload["target_preview"])
        self.assertNotEqual(state.state_of("n2"), OPENED)

    def test_follow_relation_generates_target_activation_candidate_next_round(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            state = EvidenceAgentState(EvidenceGraphStore(self._write_graph(tmp_dir), allowed_pages=[0, 1]))
            state.execute(ActivatePage(0, "initial_retrieval"))
            state.execute(ActivateNode("n1"))
            state.execute(FollowRelation("e1"))

            candidates = CandidateGenerator(state.graph).generate(state)

        relation_candidates = [
            candidate for candidate in candidates
            if candidate.action_type == "ActivatePage" and candidate.payload["page_index"] == 1
        ]
        self.assertEqual(relation_candidates[0].payload["source"], "relation_target")

    def test_candidate_generator_does_not_repeat_followed_relation(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            state = EvidenceAgentState(EvidenceGraphStore(self._write_graph(tmp_dir), allowed_pages=[0, 1]))
            state.execute(ActivatePage(0, "initial_retrieval"))
            state.execute(ActivateNode("n1"))
            state.execute(FollowRelation("e1"))

            candidates = CandidateGenerator(state.graph).generate(state)

        followed_edges = [
            candidate for candidate in candidates
            if candidate.action_type == "FollowRelation" and candidate.payload["edge_id"] == "e1"
        ]
        self.assertEqual(followed_edges, [])

    def test_candidate_ids_are_stable_when_candidate_list_changes(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            state = EvidenceAgentState(EvidenceGraphStore(self._write_graph(tmp_dir), allowed_pages=[0]))
            state.execute(ActivatePage(0, "initial_retrieval"))
            generator = CandidateGenerator(state.graph)

            first_candidates = generator.generate(state)
            n3_first_id = next(
                candidate.id for candidate in first_candidates
                if candidate.payload.get("node_id") == "n3"
            )
            state.execute(ActivateNode("n1"))
            second_candidates = generator.generate(state)
            n3_second_id = next(
                candidate.id for candidate in second_candidates
                if candidate.payload.get("node_id") == "n3"
            )

        self.assertEqual(n3_first_id, n3_second_id)
        self.assertIn("ActivateNode", n3_first_id)
        self.assertIn("n3", n3_first_id)

    def test_candidate_resolution_accepts_node_id_aliases(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            state = EvidenceAgentState(EvidenceGraphStore(self._write_graph(tmp_dir), allowed_pages=[0]))
            state.execute(ActivatePage(0, "initial_retrieval"))
            candidates = CandidateGenerator(state.graph).generate(state)
            candidate_by_id = {candidate.id: candidate for candidate in candidates}

            plain = _resolve_candidate("n1", candidate_by_id)
            prefixed = _resolve_candidate("act:n1", candidate_by_id)

        self.assertEqual(plain.payload["node_id"], "n1")
        self.assertEqual(prefixed.payload["node_id"], "n1")

    def test_candidate_resolution_accepts_stale_action_type_for_same_node(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            state = EvidenceAgentState(EvidenceGraphStore(self._write_graph(tmp_dir), allowed_pages=[0]))
            state.execute(ActivatePage(0, "initial_retrieval"))
            state.execute(ActivateNode("n1"))
            candidates = CandidateGenerator(state.graph).generate(state)
            candidate_by_id = {candidate.id: candidate for candidate in candidates}

            resolved = _resolve_candidate("act:ActivateNode:n1", candidate_by_id)

        self.assertIsNotNone(resolved)
        self.assertEqual(resolved.action_type, "OpenNode")
        self.assertEqual(resolved.payload["node_id"], "n1")

    def test_selected_alias_can_recover_allowed_node_outside_current_candidate_budget(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            state = EvidenceAgentState(EvidenceGraphStore(self._write_graph(tmp_dir), allowed_pages=[0]))
            state.execute(ActivatePage(0, "initial_retrieval"))

            recovered = _candidate_from_selected_alias("act:ActivateNode:n1", state)

        self.assertIsNotNone(recovered)
        self.assertEqual(recovered.action_type, "ActivateNode")
        self.assertEqual(recovered.payload["node_id"], "n1")

    def test_selected_alias_can_recover_bare_node_id(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            state = EvidenceAgentState(EvidenceGraphStore(self._write_graph(tmp_dir), allowed_pages=[0]))
            state.execute(ActivatePage(0, "initial_retrieval"))

            inactive = _candidate_from_selected_alias("n1", state)
            state.execute(ActivateNode("n1"))
            active = _candidate_from_selected_alias("n1", state)

        self.assertIsNotNone(inactive)
        self.assertEqual(inactive.action_type, "ActivateNode")
        self.assertIsNotNone(active)
        self.assertEqual(active.action_type, "OpenNode")

    def test_selected_alias_recovers_nearest_same_page_node_for_stale_block_id(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            graph_dir = self._write_graph(tmp_dir)
            state = EvidenceAgentState(EvidenceGraphStore(graph_dir, allowed_pages=[0]))
            state.execute(ActivatePage(0, "initial_retrieval"))
            stale_id = "sample:page:0:block:4:paragraph"
            state.graph.nodes["sample:page:0:block:3:paragraph"] = {
                "id": "sample:page:0:block:3:paragraph",
                "type": "paragraph",
                "doc_id": "sample",
                "page_index": 0,
                "abstract": "near paragraph",
                "text": "near paragraph",
            }

            recovered = _candidate_from_selected_alias(f"act:ActivateNode:{stale_id}", state)

        self.assertIsNotNone(recovered)
        self.assertEqual(recovered.action_type, "ActivateNode")
        self.assertEqual(recovered.payload["node_id"], "sample:page:0:block:3:paragraph")

    def test_selected_alias_rejects_nodes_outside_allowed_pages(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            state = EvidenceAgentState(EvidenceGraphStore(self._write_graph(tmp_dir), allowed_pages=[0]))

            recovered = _candidate_from_selected_alias("act:ActivateNode:n2", state)

        self.assertIsNone(recovered)

    def test_search_evidence_returns_only_allowed_page_results(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            state = EvidenceAgentState(EvidenceGraphStore(self._write_graph(tmp_dir), allowed_pages=[0]))

            result = state.execute(SearchEvidence("needle"))

        result_ids = [item["node_id"] for item in result.payload["results"]]
        self.assertIn("n1", result_ids)
        self.assertNotIn("n2", result_ids)

    def test_xml_evaluator_output_parses_into_internal_requests(self):
        decision = parse_agent_decision_xml(
            """
            <agent_decision>
              <stop>false</stop>
              <selected_actions><action candidate_id="act:1" utility="0.9"><reason>open it</reason></action></selected_actions>
              <search_request><query>more evidence</query></search_request>
              <prune_requests><node id="n1"><reason>bad</reason></node></prune_requests>
              <summarize_requests><summary source_node_ids="n1,n2"><goal>combine</goal></summary></summarize_requests>
              <reason>continue</reason>
            </agent_decision>
            """
        )

        self.assertFalse(decision.stop)
        self.assertEqual(decision.selected_actions[0]["candidate_id"], "act:1")
        self.assertIsNone(decision.selected_actions[0]["candidate_index"])
        self.assertEqual(decision.search_query, "more evidence")
        self.assertEqual(decision.prune_requests[0]["node_id"], "n1")
        self.assertEqual(decision.summarize_requests[0]["source_node_ids"], ["n1", "n2"])

    def test_xml_evaluator_accepts_numeric_action_indexes(self):
        decision = parse_agent_decision_xml(
            """
            <agent_decision>
              <selected_actions><action index="2" utility="0.9"><reason>open it</reason></action></selected_actions>
            </agent_decision>
            """
        )

        self.assertEqual(decision.selected_actions[0]["candidate_index"], 2)
        self.assertEqual(decision.selected_actions[0]["candidate_id"], "")

    def test_xml_evaluator_prompt_uses_structured_generic_policy(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            state = EvidenceAgentState(EvidenceGraphStore(self._write_graph(tmp_dir), allowed_pages=[0]))

            prompt = XMLEvaluator("model").build_prompt(
                "How many figures are on pages 400-640?",
                state,
                [],
            )

        self.assertIn("<decision_policy>", prompt)
        self.assertIn("<grounding_policy>", prompt)
        self.assertIn("<output_schema>", prompt)
        self.assertIn("<self_check>", prompt)
        self.assertIn("<few_shot_examples>", prompt)
        self.assertIn("<agent_step_context>", prompt)
        self.assertNotIn("For questions naming specific pages or slides", prompt)
        self.assertNotIn("For list or exhaustive questions", prompt)

    def test_xml_evaluator_prompt_does_not_add_question_keyword_hints(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            state = EvidenceAgentState(EvidenceGraphStore(self._write_graph(tmp_dir), allowed_pages=[0]))

            prompt = XMLEvaluator("model").build_prompt(
                "List all sections and enumerate every color mentioned on slides 1-3.",
                state,
                [],
            )

        self.assertNotIn("For questions naming specific pages or slides", prompt)
        self.assertNotIn("For list or exhaustive questions", prompt)
        self.assertNotIn("For color questions", prompt)
        self.assertNotIn("page/slide and target evidence", prompt)

    def test_xml_evaluator_accepts_direct_action_candidate_output(self):
        decision = parse_agent_decision_xml(
            """
            <agent_decision>
              <action index="3" type="ActivateNode" node_id="n1"/>
            </agent_decision>
            """
        )

        self.assertEqual(decision.selected_actions[0]["candidate_index"], 3)

    def test_xml_evaluator_extracts_decision_from_code_fence(self):
        decision = parse_agent_decision_xml(
            "Here is the XML:\n```xml\n<agent_decision><stop>true</stop><reason>done</reason></agent_decision>\n```"
        )

        self.assertTrue(decision.stop)
        self.assertEqual(decision.reason, "done")

    def test_xml_evaluator_recovers_numeric_actions_from_malformed_xml(self):
        decision = parse_agent_decision_xml(
            """
            <agent_decision>
              <selected_actions>
                <action index="17"/>
              </selected_actions_broken>
            </agent_decision>
            """
        )

        self.assertEqual(decision.selected_actions[0]["candidate_index"], 17)

    def test_xml_evaluator_recovers_search_query_from_malformed_xml(self):
        decision = parse_agent_decision_xml(
            """
            <agent_decision>
              <search_request>
                <query>page 98 file sizes</query>
                <scope><page index="97"/>
              </search_request>
            </agent_decision>
            """
        )

        self.assertEqual(decision.search_query, "page 98 file sizes")

    def test_invalid_candidate_ids_are_rejected_and_traced(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            state = EvidenceAgentState(EvidenceGraphStore(self._write_graph(tmp_dir), allowed_pages=[0]))
            state.execute(ActivatePage(0, "initial_retrieval"))
            builder = MAGERAGContextBuilder(OmegaConf.create({
                "baselines": {
                    "name": "magerag",
                }
            }))
            builder.max_selected_actions_per_iteration = 2
            builder.evaluator.call = lambda client, question, state, candidates: (
                EvaluatorDecision(selected_actions=[{"candidate_id": "missing"}]),
                "<agent_decision/>",
            )

            stop_reason = builder._run_agent("question", state, client=object())

        self.assertEqual(stop_reason, "watchdog_repeated_noop")
        self.assertEqual(state.validation_errors[-1]["action_type"], "SelectedAction")
        self.assertEqual(state.trace[-1]["action"], "InvalidCandidate")

    def test_numeric_candidate_index_executes_matching_action(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            state = EvidenceAgentState(EvidenceGraphStore(self._write_graph(tmp_dir), allowed_pages=[0]))
            state.execute(ActivatePage(0, "initial_retrieval"))
            builder = MAGERAGContextBuilder(OmegaConf.create({
                "baselines": {
                    "name": "magerag",
                }
            }))
            def fake_call(client, question, state, candidates):
                index = next(index for index, candidate in enumerate(candidates, start=1) if candidate.id == "act:ActivateNode:n1")
                return (
                    EvaluatorDecision(selected_actions=[{"candidate_index": index, "candidate_id": ""}]),
                    "<agent_decision/>",
                )

            builder.evaluator.call = fake_call

            builder._run_agent("question", state, client=object())

        self.assertIn("n1", state.active_node_ids())
        decision_trace = next(item for item in state.trace if item.get("action") == "EvaluatorDecision")
        self.assertIn("act:ActivateNode:n1", decision_trace["candidate_index_map"].values())

    def test_agent_truncates_selected_actions_per_iteration(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            state = EvidenceAgentState(EvidenceGraphStore(self._write_graph(tmp_dir), allowed_pages=[0]))
            state.execute(ActivatePage(0, "initial_retrieval"))
            builder = MAGERAGContextBuilder(OmegaConf.create({
                "baselines": {
                    "name": "magerag",
                }
            }))
            builder.max_selected_actions_per_iteration = 2
            call_count = 0

            def fake_call(client, question, state, candidates):
                nonlocal call_count
                call_count += 1
                if call_count > 1:
                    return EvaluatorDecision(stop=True, reason="done"), "<agent_decision><stop>true</stop></agent_decision>"
                return (
                    EvaluatorDecision(selected_actions=[
                        {"candidate_id": candidate_id}
                        for candidate_id in [
                            "act:ActivateNode:n1",
                            "act:ActivateNode:n3",
                            "act:ActivateNode:n_title",
                            "act:ActivateNode:n2",
                            "missing",
                        ]
                    ]),
                    "<agent_decision/>",
                )

            builder.evaluator.call = fake_call

            builder._run_agent("question", state, client=object())

        active_nodes = state.active_node_ids()
        self.assertIn("n1", active_nodes)
        self.assertIn("n3", active_nodes)
        self.assertNotIn("n_title", active_nodes)
        truncation_trace = next(item for item in state.trace if item.get("action") == "TruncatedSelectedActions")
        self.assertEqual(truncation_trace["original_count"], 5)
        self.assertEqual(truncation_trace["executed_count"], 2)
        decision_trace = next(item for item in state.trace if item.get("action") == "EvaluatorDecision")
        self.assertEqual(decision_trace["selected_action_execution_limit"], 2)

    def test_agent_stops_after_total_selected_action_budget(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            state = EvidenceAgentState(EvidenceGraphStore(self._write_graph(tmp_dir), allowed_pages=[0]))
            state.execute(ActivatePage(0, "initial_retrieval"))
            builder = MAGERAGContextBuilder(OmegaConf.create({
                "baselines": {
                    "name": "magerag",
                }
            }))
            builder.max_selected_actions_per_iteration = 2
            builder.max_total_selected_actions = 3
            call_count = 0

            def fake_call(client, question, state, candidates):
                nonlocal call_count
                call_count += 1
                return (
                    EvaluatorDecision(selected_actions=[
                        {"candidate_index": index, "candidate_id": ""}
                        for index in range(1, 4)
                    ]),
                    "<agent_decision/>",
                )

            builder.evaluator.call = fake_call

            stop_reason = builder._run_agent("question", state, client=object())

        self.assertEqual(stop_reason, "watchdog_total_selected_actions")
        self.assertEqual(call_count, 2)
        budget_trace = next(item for item in state.trace if item.get("action") == "SelectedActionBudgetReached")
        self.assertEqual(budget_trace["executed_total"], 4)
        self.assertEqual(budget_trace["limit"], 3)

    def test_empty_candidate_numeric_selection_is_ignored_without_validation_error(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            state = EvidenceAgentState(EvidenceGraphStore(self._write_graph(tmp_dir), allowed_pages=[0]))
            builder = MAGERAGContextBuilder(OmegaConf.create({
                "baselines": {
                    "name": "magerag",
                }
            }))
            builder.evaluator.call = lambda client, question, state, candidates: (
                EvaluatorDecision(selected_actions=[{"candidate_index": 1, "candidate_id": ""}]),
                "<agent_decision/>",
            )

            stop_reason = builder._run_agent("question", state, client=object())

        self.assertEqual(stop_reason, "watchdog_repeated_noop")
        self.assertEqual(state.validation_errors, [])
        self.assertEqual(state.trace[-1]["action"], "IgnoredEmptyCandidateSelection")

    def test_evaluator_decision_trace_records_input_without_base64_images(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            graph_dir = Path(self._write_graph(tmp_dir))
            image_path = os.path.join(tmp_dir, "crop.png")
            with open(image_path, "wb") as handle:
                handle.write(b"fake image")
            lines = (graph_dir / "nodes.jsonl").read_text(encoding="utf-8").splitlines()
            node = json.loads(lines[0])
            node["image_path"] = image_path
            lines[0] = json.dumps(node)
            (graph_dir / "nodes.jsonl").write_text("\n".join(lines) + "\n", encoding="utf-8")
            state = EvidenceAgentState(EvidenceGraphStore(graph_dir, allowed_pages=[0]))
            state.execute(ActivatePage(0, "initial_retrieval"))
            state.execute(ActivateNode("n1"))
            state.execute(OpenNode("n1"))
            builder = MAGERAGContextBuilder(OmegaConf.create({
                "baselines": {
                    "name": "magerag",
                }
            }))
            builder.evaluator.call = lambda client, question, state, candidates: (
                EvaluatorDecision(stop=True, reason="done"),
                "<agent_decision><stop>true</stop><reason>done</reason></agent_decision>",
            )

            builder._run_agent("question", state, client=object())

        decision_trace = next(item for item in state.trace if item.get("action") == "EvaluatorDecision")
        evaluator_input = decision_trace["evaluator_input"]
        self.assertIn("<agent_step_context>", evaluator_input["context_xml"])
        self.assertTrue(evaluator_input["candidate_actions"])
        self.assertEqual(evaluator_input["opened_image_refs"][0]["node_id"], "n1")
        self.assertEqual(evaluator_input["opened_image_refs"][0]["image_path"], image_path)
        serialized = json.dumps(evaluator_input)
        self.assertNotIn("base64", serialized)
        self.assertNotIn("data:image", serialized)

    def test_opened_node_content_truncates_to_configured_char_limit(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            state = EvidenceAgentState(EvidenceGraphStore(self._write_graph(tmp_dir, long_text=True), allowed_pages=[0]))
            state.execute(ActivatePage(0, "initial_retrieval"))
            state.execute(ActivateNode("n1"))
            state.execute(OpenNode("n1"))

            xml = XMLEvaluator("model", raw_text_char_limit=8192).build_context_xml("question", state, [])

        self.assertIn('<content truncated="true">', xml)
        content = ET.fromstring(xml).find(".//opened_nodes/node/content")
        self.assertEqual(len(content.text), 8192)

    def test_online_mode_does_not_auto_open_initial_page_nodes_before_evaluator(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            graph_root = os.path.join(tmp_dir, "graphs")
            self._write_graph(os.path.join(graph_root, "sample"))
            cfg = OmegaConf.create({
                "baselines": {
                    "name": "magerag",
                    "params": {},
                },
                "benchmarks": {
                    "name": "mmlongbench",
                    "evidence_graph_dir": graph_root,
                    "max_pages": 1,
                    "resolution": 144,
                    "pdf_png_dir": os.path.join(tmp_dir, "missing_pngs"),
                },
            })
            builder = MAGERAGContextBuilder(cfg)
            builder.retriever.retrieve_many = lambda benchmark_name, sample, allowed_pages: (
                [{"page_index": 0, "page_number": 1, "score": 2.0}],
                {"retrieved_pages": [{"page_index": 0, "page_number": 1, "score": 2.0}]},
            )
            builder.evaluator.call = lambda client, question, state, candidates: (
                EvaluatorDecision(stop=True, reason="done"),
                "<agent_decision><stop>true</stop><reason>done</reason></agent_decision>",
            )

            messages = builder.build(
                "mmlongbench",
                {"doc_id": "sample.pdf", "question_id": "q1", "question": "What does the source evidence say?"},
                client=object(),
            )

        decision_trace = next(item for item in messages.metadata["iteration_trace"] if item.get("action") == "EvaluatorDecision")
        self.assertEqual(messages.metadata["opened_node_ids"], [])
        self.assertEqual(decision_trace["evaluator_input"]["opened_image_refs"], [])
        self.assertNotIn("<opened_nodes>\n      <node", decision_trace["evaluator_input"]["context_xml"])

    def test_online_build_final_opens_active_nodes_after_evaluator(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            graph_root = os.path.join(tmp_dir, "graphs")
            self._write_graph(os.path.join(graph_root, "sample"))
            cfg = OmegaConf.create({
                "baselines": {
                    "name": "magerag",
                    "params": {},
                },
                "benchmarks": {
                    "name": "mmlongbench",
                    "evidence_graph_dir": graph_root,
                    "max_pages": 1,
                    "resolution": 144,
                    "pdf_png_dir": os.path.join(tmp_dir, "missing_pngs"),
                },
            })
            builder = MAGERAGContextBuilder(cfg)
            builder.retriever.retrieve_many = lambda benchmark_name, sample, allowed_pages: (
                [{"page_index": 0, "page_number": 1, "score": 2.0}],
                {"retrieved_pages": [{"page_index": 0, "page_number": 1, "score": 2.0}]},
            )
            call_count = 0

            def fake_call(client, question, state, candidates):
                nonlocal call_count
                call_count += 1
                if call_count > 1:
                    return (
                        EvaluatorDecision(stop=True, reason="done"),
                        "<agent_decision><stop>true</stop><reason>done</reason></agent_decision>",
                    )
                return (
                    EvaluatorDecision(selected_actions=[{"candidate_id": "act:ActivateNode:n1"}]),
                    "<agent_decision><selected_actions><action candidate_id=\"act:ActivateNode:n1\"/></selected_actions></agent_decision>",
                )

            builder.evaluator.call = fake_call

            messages = builder.build(
                "mmlongbench",
                {"doc_id": "sample.pdf", "question_id": "q1", "question": "What does the source evidence say?"},
                client=object(),
            )

        decision_trace = next(item for item in messages.metadata["iteration_trace"] if item.get("action") == "EvaluatorDecision")
        self.assertEqual(decision_trace["evaluator_input"]["opened_image_refs"], [])
        self.assertIn("n1", messages.metadata["opened_node_ids"])
        self.assertTrue(any(item.get("action") == "FinalOpenActiveNode" for item in messages.metadata["iteration_trace"]))

    def test_online_search_decision_opens_top_search_result(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            graph_root = os.path.join(tmp_dir, "graphs")
            graph_dir = Path(self._write_graph(os.path.join(graph_root, "sample")))
            with open(graph_dir / "nodes.jsonl", "a", encoding="utf-8") as handle:
                handle.write(json.dumps({
                    "id": "searched_answer",
                    "type": "table",
                    "doc_id": "sample",
                    "page_index": 1,
                    "abstract": "wheelchair accessible ESCAPE BRYGGEN answer table",
                    "text": "ESCAPE BRYGGEN is not suitable for wheelchairs.",
                }) + "\n")
            cfg = OmegaConf.create({
                "baselines": {
                    "name": "magerag",
                    "params": {},
                },
                "benchmarks": {
                    "name": "mmlongbench",
                    "evidence_graph_dir": graph_root,
                    "max_pages": 2,
                    "resolution": 144,
                    "pdf_png_dir": os.path.join(tmp_dir, "missing_pngs"),
                },
            })
            builder = MAGERAGContextBuilder(cfg)
            builder.retriever.retrieve_many = lambda benchmark_name, sample, allowed_pages: (
                [{"page_index": 0, "page_number": 1, "score": 2.0}],
                {"retrieved_pages": [{"page_index": 0, "page_number": 1, "score": 2.0}]},
            )
            builder.evaluator.call = lambda client, question, state, candidates: (
                EvaluatorDecision(search_query="ESCAPE BRYGGEN wheelchair"),
                "<agent_decision><search_request><query>ESCAPE BRYGGEN wheelchair</query></search_request></agent_decision>",
            )

            messages = builder.build(
                "mmlongbench",
                {"doc_id": "sample.pdf", "question_id": "q1", "question": "Which attraction is not suitable for wheelchair?"},
                client=object(),
            )

        self.assertIn("searched_answer", messages.metadata["opened_node_ids"])
        self.assertTrue(any(
            item.get("action") == "AutoOpenSearchResult"
            and item.get("node_id") == "searched_answer"
            for item in messages.metadata["iteration_trace"]
        ))

    def test_final_guard_opens_salient_nodes_on_active_pages(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            graph_dir = Path(self._write_graph(tmp_dir))
            with open(graph_dir / "nodes.jsonl", "a", encoding="utf-8") as handle:
                handle.write(json.dumps({
                    "id": "page_only_table",
                    "type": "table",
                    "doc_id": "sample",
                    "page_index": 1,
                    "abstract": "target evidence table with final answer",
                    "text": "The final answer is 42.",
                }) + "\n")
            state = EvidenceAgentState(EvidenceGraphStore(graph_dir, allowed_pages=[0, 1]))
            state.execute(ActivatePage(page_index=1, source="search"), iteration=1)
            builder = MAGERAGContextBuilder(OmegaConf.create({
                "baselines": {
                    "name": "magerag",
                },
            }))

            builder._final_open_active_nodes("mmlongbench", "What is the final answer table value?", state)

        self.assertIn("page_only_table", state.opened_node_ids())
        self.assertTrue(any(
            item.get("action") == "FinalOpenActivePageNode"
            and item.get("node_id") == "page_only_table"
            for item in state.trace
        ))

    def test_opened_node_content_truncates_to_compact_default_for_online_evaluator(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            state = EvidenceAgentState(EvidenceGraphStore(self._write_graph(tmp_dir, long_text=True), allowed_pages=[0]))
            state.execute(ActivatePage(0, "initial_retrieval"))
            state.execute(ActivateNode("n1"))
            state.execute(OpenNode("n1"))

            xml = XMLEvaluator("model").build_context_xml("question", state, [])

        content = ET.fromstring(xml).find(".//opened_nodes/node/content")
        self.assertEqual(len(content.text), 1200)

    def test_evaluator_context_limits_candidate_actions_and_preview_chars(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            state = EvidenceAgentState(EvidenceGraphStore(self._write_graph(tmp_dir), allowed_pages=[0]))
            candidates = [
                CandidateAction(
                    id=f"act:ActivateNode:n{i}",
                    action_type="ActivateNode",
                    payload={"node_id": f"n{i}"},
                    preview="x" * 1000,
                )
                for i in range(25)
            ]

            xml = XMLEvaluator(
                "model",
                max_candidate_actions=10,
                candidate_preview_char_limit=40,
            ).build_context_xml("question", state, candidates)

        root = ET.fromstring(xml)
        actions = root.findall(".//candidate_actions/action")
        self.assertEqual(len(actions), 10)
        self.assertEqual(actions[0].attrib["index"], "1")
        self.assertNotIn("id", actions[0].attrib)
        self.assertTrue(all(len(action.findtext("preview")) <= 43 for action in actions))

    def test_agent_keeps_question_relevant_candidates_when_capping_evaluator_actions(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            state = EvidenceAgentState(EvidenceGraphStore(self._write_graph(tmp_dir), allowed_pages=[0]))
            state.execute(ActivatePage(0, "initial_retrieval"))
            for index in range(40):
                node_id = f"noise_{index:02d}"
                state.graph.nodes[node_id] = {
                    "id": node_id,
                    "type": "paragraph",
                    "doc_id": "sample",
                    "page_index": 0,
                    "abstract": "generic unrelated appendix note",
                    "text": "generic unrelated appendix note",
                }
            state.graph.nodes["answer_node"] = {
                "id": "answer_node",
                "type": "paragraph",
                "doc_id": "sample",
                "page_index": 0,
                "abstract": "needle answer revenue table",
                "text": "needle answer revenue table",
            }
            builder = MAGERAGContextBuilder(OmegaConf.create({
                "baselines": {
                    "name": "magerag",
                }
            }))
            builder.max_evaluator_candidate_actions = 5
            seen_candidate_ids = []

            def fake_call(client, question, state, candidates):
                seen_candidate_ids.extend(candidate.id for candidate in candidates)
                return (
                    EvaluatorDecision(stop=True, reason="done"),
                    "<agent_decision><stop>true</stop><reason>done</reason></agent_decision>",
                )

            builder.evaluator.call = fake_call

            builder._run_agent("What is the needle answer revenue?", state, client=object())

        self.assertLessEqual(len(seen_candidate_ids), 5)
        self.assertIn("act:ActivateNode:answer_node", seen_candidate_ids)

    def test_evaluator_recent_trace_excludes_prior_full_evaluator_inputs(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            state = EvidenceAgentState(EvidenceGraphStore(self._write_graph(tmp_dir), allowed_pages=[0]))
            state.execute(ActivatePage(0, "initial_retrieval"))
            state.trace.append({
                "iteration": 1,
                "action": "EvaluatorDecision",
                "evaluator_input": {"context_xml": "<agent_step_context>" + ("x" * 5000)},
                "raw_response": "<agent_decision>" + ("y" * 5000),
                "decision": {"selected_actions": [{"candidate_id": "act:ActivateNode:n1"}]},
                "candidate_ids": ["act:ActivateNode:n1"],
            })

            xml = XMLEvaluator("model").build_context_xml("question", state, [])

        self.assertNotIn("evaluator_input", xml)
        self.assertNotIn("context_xml", xml)
        self.assertNotIn("raw_response", xml)
        self.assertLess(len(xml), 3000)

    def test_reader_renderer_excludes_pruned_nodes_and_includes_summaries(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            state = EvidenceAgentState(EvidenceGraphStore(self._write_graph(tmp_dir), allowed_pages=[0]))
            state.execute(ActivatePage(0, "initial_retrieval"))
            state.execute(ActivateNode("n1"))
            state.execute(OpenNode("n1"))
            state.execute(PruneNode("n1", "irrelevant"))
            state.summaries.append({"summary_id": "summary:iter1:0", "source_node_ids": ["n1"], "text": "summary text"})

            content = ReaderRenderer(
                OmegaConf.create({"benchmarks": {}, "baselines": {"reader": {"mode": "full"}}}),
                include_page_images=False,
            ).render(
                "mmlongbench",
                {"question": "Q?"},
                state,
            )

        prompt = content[0]["text"]
        self.assertNotIn("[n1] type=", prompt)
        self.assertIn("summary text", prompt)

    def test_reader_renderer_full_mode_uses_general_evidence_id_warning(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            state = EvidenceAgentState(EvidenceGraphStore(self._write_graph(tmp_dir), allowed_pages=[0]))
            state.execute(ActivatePage(0, "initial_retrieval"))
            state.execute(ActivateNode("n_title"))
            state.execute(OpenNode("n_title"))

            content = ReaderRenderer(
                OmegaConf.create({"benchmarks": {}, "baselines": {"reader": {"mode": "full"}}}),
                include_page_images=False,
            ).render(
                "longdocurl",
                {
                    "question": "Which section best matches the description?",
                    "question_type": "summary2title",
                    "task_tag": "Locating",
                },
                state,
            )

        prompt = content[0]["text"]
        self.assertIn("Do not answer with evidence node ids", prompt)
        self.assertIn("Important Section Title", prompt)
        self.assertNotIn("Candidate answer strings", prompt)
        self.assertNotIn("[n_title]", prompt)

    def test_reader_renderer_compact_mode_omits_full_active_graph(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            state = EvidenceAgentState(EvidenceGraphStore(self._write_graph(tmp_dir), allowed_pages=[0]))
            state.execute(ActivatePage(0, "initial_retrieval"))
            state.execute(ActivateNode("n_title"))
            state.execute(ActivateNode("n1"))

            content = ReaderRenderer(
                OmegaConf.create({"benchmarks": {}, "baselines": {"reader": {"mode": "compact"}}}),
                include_page_images=False,
            ).render(
                "longdocurl",
                {"question": "Which section best matches the description?"},
                state,
            )

        prompt = content[0]["text"]
        self.assertIn("<task>", prompt)
        self.assertIn("<question>Which section best matches the description?</question>", prompt)
        self.assertIn("<answer_policy>", prompt)
        self.assertIn("If the answer cannot be found, answer exactly: Not answerable.", prompt)
        self.assertNotIn("Candidate visible labels from retrieved evidence:", prompt)
        self.assertNotIn("Important Section Title", prompt)
        self.assertNotIn("Active evidence graph:", prompt)
        self.assertNotIn("provenance_id=n_title", prompt)
        self.assertNotIn("needle source", prompt)

    def test_reader_renderer_compact_mode_limits_candidate_labels_to_locating_questions(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            state = EvidenceAgentState(EvidenceGraphStore(self._write_graph(tmp_dir), allowed_pages=[0]))
            state.execute(ActivatePage(0, "initial_retrieval"))
            state.execute(ActivateNode("n_title"))

            content = ReaderRenderer(
                OmegaConf.create({"benchmarks": {}, "baselines": {"reader": {"mode": "compact"}}}),
                include_page_images=False,
            ).render(
                "longdocurl",
                {"question": "What is the total amount of liabilities?"},
                state,
            )

        prompt = content[0]["text"]
        self.assertNotIn("Candidate visible labels", prompt)
        self.assertNotIn("Important Section Title", prompt)

    def test_reader_renderer_compact_uses_opened_text_without_table_name_candidates(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            state = EvidenceAgentState(EvidenceGraphStore(self._write_graph(tmp_dir), allowed_pages=[0]))
            state.graph.nodes["n1"]["type"] = "table"
            state.graph.nodes["n1"]["caption"] = "Table 15: Leading destination of exports (UGX Billion): July-June"
            state.execute(ActivatePage(0, "initial_retrieval"))
            state.execute(ActivateNode("n1"))
            state.execute(OpenNode("n1"))

            content = ReaderRenderer(
                OmegaConf.create({"benchmarks": {}, "baselines": {"reader": {"mode": "compact"}}}),
                include_page_images=False,
            ).render(
                "longdocurl",
                {"question": "What's name of the table at the page which contains a figure whose name is Figure 29?"},
                state,
            )

        prompt = content[0]["text"]
        self.assertNotIn("Candidate visible labels from retrieved evidence:", prompt)
        self.assertIn("Table 15: Leading destination of exports (UGX Billion): July-June", prompt)

    def test_reader_renderer_compact_uses_generic_structured_answer_prompt(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            state = EvidenceAgentState(EvidenceGraphStore(self._write_graph(tmp_dir), allowed_pages=[0]))
            state.execute(ActivatePage(0, "initial_retrieval"))

            content = ReaderRenderer(
                OmegaConf.create({
                    "benchmarks": {},
                    "baselines": {"reader": {"mode": "compact"}},
                }),
                include_page_images=False,
            ).render("mmlongbench", {"question": "Which area is not shown?"}, state)

        prompt = content[0]["text"]
        self.assertIn("<task>", prompt)
        self.assertIn("<evidence_policy>", prompt)
        self.assertIn("<answer_policy>", prompt)
        self.assertIn("<self_check>", prompt)
        self.assertIn("If the answer cannot be found, answer exactly: Not answerable.", prompt)
        self.assertIn("Do not answer with None, null, [], or an empty string.", prompt)

    def test_reader_renderer_compact_lists_retrieved_pages_generically(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            state = EvidenceAgentState(EvidenceGraphStore(self._write_graph(tmp_dir), allowed_pages=[0, 1]))
            state.execute(ActivatePage(0, "initial_retrieval"), iteration=0)
            state.execute(ActivatePage(1, "initial_retrieval"), iteration=0)

            content = ReaderRenderer(
                OmegaConf.create({
                    "benchmarks": {},
                    "baselines": {"reader": {"mode": "compact"}},
                }),
                include_page_images=False,
            ).render("mmlongbench", {"question": "Which area is shown?"}, state)

        prompt = content[0]["text"]
        self.assertIn("<retrieved_pages>1, 2</retrieved_pages>", prompt)
        self.assertIn("If the answer cannot be found, answer exactly: Not answerable.", prompt)
        self.assertIn("Return only the final answer.", prompt)

    def test_reader_renderer_compact_no_longer_supports_plain_prompt_mode(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            state = EvidenceAgentState(EvidenceGraphStore(self._write_graph(tmp_dir), allowed_pages=[0]))
            state.execute(ActivatePage(0, "initial_retrieval"))

            content = ReaderRenderer(
                OmegaConf.create({
                    "benchmarks": {},
                    "baselines": {"reader": {"mode": "compact"}},
                }),
                include_page_images=False,
            ).render("mmlongbench", {"question": "Which area is shown?"}, state)

        prompt = content[0]["text"]
        self.assertIn("If the answer cannot be found", prompt)
        self.assertIn("<retrieved_pages>1</retrieved_pages>", prompt)
        self.assertNotIn("For color questions", prompt)

    def test_reader_renderer_compact_omits_color_question_hint(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            state = EvidenceAgentState(EvidenceGraphStore(self._write_graph(tmp_dir), allowed_pages=[0]))
            state.execute(ActivatePage(0, "initial_retrieval"))

            content = ReaderRenderer(
                OmegaConf.create({
                    "benchmarks": {},
                    "baselines": {"reader": {"mode": "compact"}},
                }),
                include_page_images=False,
            ).render("mmlongbench", {"question": "What color is the highlighted area?"}, state)

        prompt = content[0]["text"]
        self.assertNotIn("For color questions, use common color names rather than hex codes.", prompt)
        self.assertIn("If the answer cannot be found", prompt)
        self.assertIn("<retrieved_pages>1</retrieved_pages>", prompt)

    def test_reader_renderer_compact_omits_page_scope_hint(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            state = EvidenceAgentState(EvidenceGraphStore(self._write_graph(tmp_dir), allowed_pages=[0]))
            state.execute(ActivatePage(0, "initial_retrieval"))

            content = ReaderRenderer(
                OmegaConf.create({
                    "benchmarks": {},
                    "baselines": {"reader": {"mode": "compact"}},
                }),
                include_page_images=False,
            ).render(
                "mmlongbench",
                {"question": "How many tables are shown on pages 100-110?"},
                state,
            )

        prompt = content[0]["text"]
        self.assertNotIn("For questions that name specific pages or slides", prompt)
        self.assertNotIn("If the retrieved pages do not include the requested page or slide scope", prompt)

    def test_reader_renderer_can_label_mmlongbench_page_images(self):
        renderer = ReaderRenderer(
            OmegaConf.create({
                "benchmarks": {},
                "baselines": {
                    "reader": {
                        "mode": "compact",
                        "include_image_page_labels": True,
                    }
                },
            }),
            include_page_images=True,
        )

        label = renderer._image_label("mmlongbench", 0, 2, 4)

        self.assertEqual(label, "Document page 5:\n")

    def test_reader_renderer_selects_relevant_opened_node_crops(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            state = EvidenceAgentState(EvidenceGraphStore(self._write_graph(tmp_dir), allowed_pages=[0, 1]))
            state.graph.nodes["n1"]["bbox"] = [10, 10, 100, 100]
            state.graph.nodes["n1"]["abstract"] = "unrelated paragraph"
            state.graph.nodes["n2"]["bbox"] = [20, 20, 160, 160]
            state.graph.nodes["n2"]["abstract"] = "needle chart target"
            state.execute(ActivatePage(0, "initial_retrieval"), iteration=0)
            state.execute(ActivateNode("n1"), iteration=0)
            state.execute(OpenNode("n1"), iteration=0)
            state.execute(ActivatePage(1, "initial_retrieval"), iteration=0)
            state.execute(ActivateNode("n2"), iteration=0)
            state.execute(OpenNode("n2"), iteration=0)

            renderer = ReaderRenderer(
                OmegaConf.create({
                    "benchmarks": {},
                    "baselines": {"reader": {"mmlongbench_max_opened_node_crops": 1}},
                }),
                include_page_images=False,
            )

            node_ids = renderer._candidate_crop_node_ids("Which chart contains the needle target?", state)

        self.assertEqual(node_ids, ["n2"])

    def test_reader_renderer_preserves_activation_page_order(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            state = EvidenceAgentState(EvidenceGraphStore(self._write_graph(tmp_dir), allowed_pages=[0, 1]))
            state.execute(ActivatePage(1, "initial_retrieval"), iteration=0)
            state.execute(ActivatePage(0, "initial_retrieval"), iteration=0)

            page_indices = ReaderRenderer(OmegaConf.create({"benchmarks": {}}), include_page_images=False)._reader_page_indices(state)

        self.assertEqual(page_indices, [1, 0])

    def test_reader_renderer_compact_mmlongbench_includes_page_text_snippets(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            state = EvidenceAgentState(EvidenceGraphStore(self._write_graph(tmp_dir), allowed_pages=[0, 1]))
            state.graph.nodes["page:1"]["text"] = "Page two chart says With family and friends 20%."
            state.graph.nodes["page:0"]["text"] = "Page one unrelated text."
            state.execute(ActivatePage(1, "initial_retrieval"), iteration=0)
            state.execute(ActivatePage(0, "initial_retrieval"), iteration=0)

            content = ReaderRenderer(
                OmegaConf.create({
                    "benchmarks": {},
                    "baselines": {
                        "reader": {
                            "mode": "compact",
                            "mmlongbench_page_text_char_limit": 80,
                            "mmlongbench_page_text_max_pages": 1,
                        }
                    },
                }),
                include_page_images=False,
            ).render("mmlongbench", {"question": "How much time was spent with family?"}, state)

        prompt = content[0]["text"]
        self.assertIn("<retrieved_page_text_snippets>", prompt)
        self.assertIn('<page_text page="2">', prompt)
        self.assertNotIn('<page_text page="1">', prompt)
        self.assertIn("With family and friends 20%", prompt)

    def test_final_context_metadata_contains_trace_and_node_state_fields(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            graph_root = os.path.join(tmp_dir, "graphs")
            self._write_graph(os.path.join(graph_root, "sample"))
            cfg = OmegaConf.create({
                "baselines": {"name": "magerag", "params": {}},
                "benchmarks": {
                    "name": "mmlongbench",
                    "evidence_graph_dir": graph_root,
                    "max_pages": 1,
                    "resolution": 144,
                    "pdf_png_dir": os.path.join(tmp_dir, "missing_pngs"),
                },
            })
            builder = MAGERAGContextBuilder(cfg)

            messages = builder.build("mmlongbench", {"doc_id": "sample.pdf", "question_id": "q1", "question": "Q?"})

        self.assertIsInstance(messages, ContextMessages)
        self.assertEqual(messages.metadata["context_builder"], "magerag")
        self.assertEqual(messages.metadata["allowed_pages"], [0])
        self.assertIn("final_node_states", messages.metadata)
        self.assertIn("iteration_trace", messages.metadata)
        self.assertIn("validation_errors", messages.metadata)

    def test_final_context_metadata_records_reader_input_without_base64_images(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            graph_root = os.path.join(tmp_dir, "graphs")
            self._write_graph(os.path.join(graph_root, "sample"))
            page_dir = Path(tmp_dir) / "pngs" / "sample"
            page_dir.mkdir(parents=True)
            from PIL import Image

            Image.new("RGB", (1, 1), color="white").save(page_dir / "page_0001_dpi144.png")
            cfg = OmegaConf.create({
                "baselines": {
                    "name": "magerag",
                    "params": {},
                    "reader": {"include_page_images": True},
                },
                "benchmarks": {
                    "name": "mmlongbench",
                    "evidence_graph_dir": graph_root,
                    "max_pages": 1,
                    "resolution": 144,
                    "pdf_png_dir": str(Path(tmp_dir) / "pngs"),
                },
            })
            builder = MAGERAGContextBuilder(cfg)

            messages = builder.build("mmlongbench", {"doc_id": "sample.pdf", "question_id": "q1", "question": "Q?"})

        reader_input = messages.metadata["reader_input"]
        self.assertIn("Q?", reader_input["text_parts"][0])
        self.assertEqual(reader_input["content_part_count"], len(messages[0]["content"]))
        self.assertTrue(reader_input["image_refs"])
        self.assertEqual(reader_input["image_refs"][0]["page_index"], 0)
        serialized = json.dumps(reader_input)
        self.assertNotIn("base64", serialized)
        self.assertNotIn("data:image", serialized)

    def test_online_trace_records_selection_and_state_snapshots(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            state = EvidenceAgentState(EvidenceGraphStore(self._write_graph(tmp_dir), allowed_pages=[0]))
            state.execute(ActivatePage(0, "initial_retrieval"))
            builder = MAGERAGContextBuilder(OmegaConf.create({
                "baselines": {
                    "name": "magerag",
                }
            }))
            builder.evaluator.call = lambda client, question, state, candidates: (
                EvaluatorDecision(selected_actions=[{"candidate_index": 1, "candidate_id": "", "reason": "inspect"}]),
                "<agent_decision><selected_actions><action index=\"1\"><reason>inspect</reason></action></selected_actions></agent_decision>",
            )

            builder._run_agent("mmlongbench", "question", state, client=object())

        decision_trace = next(item for item in state.trace if item.get("action") == "EvaluatorDecision")
        self.assertIn("prompt_text", decision_trace["evaluator_input"])
        self.assertIn("<agent_step_context>", decision_trace["evaluator_input"]["prompt_text"])
        self.assertIn("state_snapshot_before", decision_trace)
        executed = next(item for item in state.trace if item.get("action") == "ActivateNode")
        self.assertEqual(executed["selection"]["candidate_index"], 1)
        self.assertEqual(executed["selection"]["reason"], "inspect")
        self.assertIn("state_snapshot_after", executed)
        self.assertIn("n1", executed["state_snapshot_after"]["active_node_ids"])

    def test_initial_retrieval_activates_multiple_top_pages(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            graph_root = os.path.join(tmp_dir, "graphs")
            self._write_graph(os.path.join(graph_root, "sample"))
            cfg = OmegaConf.create({
                "baselines": {
                    "name": "magerag",
                    "params": {},
                },
                "benchmarks": {
                    "name": "mmlongbench",
                    "evidence_graph_dir": graph_root,
                    "max_pages": 2,
                    "resolution": 144,
                    "pdf_png_dir": os.path.join(tmp_dir, "missing_pngs"),
                },
            })
            builder = MAGERAGContextBuilder(cfg)
            builder.retriever.retrieve_many = lambda benchmark_name, sample, allowed_pages: (
                [
                    {"page_index": 0, "page_number": 1, "score": 2.0},
                    {"page_index": 1, "page_number": 2, "score": 1.0},
                ],
                {"retrieved_pages": [
                    {"page_index": 0, "page_number": 1, "score": 2.0},
                    {"page_index": 1, "page_number": 2, "score": 1.0},
                ]},
            )

            messages = builder.build("mmlongbench", {"doc_id": "sample.pdf", "question_id": "q1", "question": "Q?"})

        self.assertEqual(len(messages.metadata["initial_retrieval"]["retrieved_pages"]), 2)
        self.assertIn("page:0", messages.metadata["active_node_ids"])
        self.assertIn("page:1", messages.metadata["active_node_ids"])

    def test_build_runs_online_agent_by_default(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            graph_root = os.path.join(tmp_dir, "graphs")
            self._write_graph(os.path.join(graph_root, "sample"))
            cfg = OmegaConf.create({
                "baselines": {
                    "name": "magerag",
                    "params": {},
                },
                "benchmarks": {
                    "name": "mmlongbench",
                    "evidence_graph_dir": graph_root,
                    "max_pages": 1,
                    "resolution": 144,
                    "pdf_png_dir": os.path.join(tmp_dir, "missing_pngs"),
                },
            })
            builder = MAGERAGContextBuilder(cfg)
            builder.retriever.retrieve_many = lambda benchmark_name, sample, allowed_pages: (
                [{"page_index": 0, "page_number": 1, "score": 2.0}],
                {"retrieved_pages": [{"page_index": 0, "page_number": 1, "score": 2.0}]},
            )
            builder.evaluator.call = lambda client, question, state, candidates: (
                EvaluatorDecision(stop=True, reason="done"),
                "<agent_decision><stop>true</stop><reason>done</reason></agent_decision>",
            )

            messages = builder.build("mmlongbench", {"doc_id": "sample.pdf", "question_id": "q1", "question": "Q?"}, client=object())

        self.assertEqual(messages.metadata["stop_reason"], "normal_stop")
        self.assertIn("EvaluatorDecision", [item.get("action") for item in messages.metadata["iteration_trace"]])
        self.assertNotIn("run_online_agent", messages.metadata)

    def test_initial_page_nodes_are_opened_and_rendered_without_expanding_top_k(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            graph_root = os.path.join(tmp_dir, "graphs")
            self._write_graph(os.path.join(graph_root, "sample"))
            cfg = OmegaConf.create({
                "baselines": {
                    "name": "magerag",
                    "params": {},
                    "reader": {
                        "mode": "compact",
                        "include_opened_node_text": True,
                        "include_opened_node_text_mmlongbench": True,
                        "opened_node_text_char_limit": 200,
                    },
                },
                "benchmarks": {
                    "name": "mmlongbench",
                    "evidence_graph_dir": graph_root,
                    "max_pages": 2,
                    "resolution": 144,
                    "pdf_png_dir": os.path.join(tmp_dir, "missing_pngs"),
                },
            })
            builder = MAGERAGContextBuilder(cfg)
            builder.retriever.retrieve_many = lambda benchmark_name, sample, allowed_pages: (
                [{"page_index": 0, "page_number": 1, "score": 2.0}],
                {"retrieved_pages": [{"page_index": 0, "page_number": 1, "score": 2.0}]},
            )
            call_count = 0

            def fake_call(client, question, state, candidates):
                nonlocal call_count
                call_count += 1
                if call_count == 1:
                    return (
                        EvaluatorDecision(selected_actions=[
                            {"candidate_id": "act:ActivateNode:n_title"},
                            {"candidate_id": "act:ActivateNode:n1"},
                        ]),
                        "<agent_decision><selected_actions><action candidate_id=\"act:ActivateNode:n_title\"/><action candidate_id=\"act:ActivateNode:n1\"/></selected_actions></agent_decision>",
                    )
                return (
                    EvaluatorDecision(stop=True, reason="done"),
                    "<agent_decision><stop>true</stop><reason>done</reason></agent_decision>",
                )

            builder.evaluator.call = fake_call

            messages = builder.build(
                "mmlongbench",
                {"doc_id": "sample.pdf", "question_id": "q1", "question": "What does the source evidence say?"},
                client=object(),
            )

        self.assertEqual(len(messages.metadata["initial_retrieval"]["retrieved_pages"]), 1)
        self.assertIn("n_title", messages.metadata["opened_node_ids"])
        self.assertIn("n1", messages.metadata["opened_node_ids"])
        prompt = messages[0]["content"][0]["text"]
        self.assertIn("<opened_evidence_text>", prompt)
        self.assertIn("needle source evidence", prompt)

    def test_mmlongbench_uses_fixed_default_auto_open_limit(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            graph_root = os.path.join(tmp_dir, "graphs")
            self._write_graph(os.path.join(graph_root, "sample"))
            cfg = OmegaConf.create({
                "baselines": {
                    "name": "magerag",
                    "params": {},
                },
                "benchmarks": {
                    "name": "mmlongbench",
                    "evidence_graph_dir": graph_root,
                    "max_pages": 1,
                    "resolution": 144,
                    "pdf_png_dir": os.path.join(tmp_dir, "missing_pngs"),
                },
            })
            builder = MAGERAGContextBuilder(cfg)
            builder.retriever.retrieve_many = lambda benchmark_name, sample, allowed_pages: (
                [{"page_index": 0, "page_number": 1, "score": 2.0}],
                {"retrieved_pages": [{"page_index": 0, "page_number": 1, "score": 2.0}]},
            )
            call_count = 0

            def fake_call(client, question, state, candidates):
                nonlocal call_count
                call_count += 1
                if call_count == 1:
                    return (
                        EvaluatorDecision(selected_actions=[
                            {"candidate_id": "act:ActivateNode:n_title"},
                            {"candidate_id": "act:ActivateNode:n1"},
                        ]),
                        "<agent_decision><selected_actions><action candidate_id=\"act:ActivateNode:n_title\"/><action candidate_id=\"act:ActivateNode:n1\"/></selected_actions></agent_decision>",
                    )
                return (
                    EvaluatorDecision(stop=True, reason="done"),
                    "<agent_decision><stop>true</stop><reason>done</reason></agent_decision>",
                )

            builder.evaluator.call = fake_call

            messages = builder.build(
                "mmlongbench",
                {"doc_id": "sample.pdf", "question_id": "q1", "question": "What does the source evidence say?"},
                client=object(),
            )

        self.assertIn("n_title", messages.metadata["opened_node_ids"])
        self.assertIn("n1", messages.metadata["opened_node_ids"])

    def test_mmlongbench_opens_nodes_on_explicit_question_page(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            graph_root = os.path.join(tmp_dir, "graphs")
            graph_dir = os.path.join(graph_root, "sample")
            os.makedirs(graph_dir, exist_ok=True)
            with open(os.path.join(graph_dir, "graph.json"), "w", encoding="utf-8") as f:
                json.dump({"doc_id": "sample"}, f)
            nodes = [
                {
                    "id": "page1_text",
                    "type": "paragraph",
                    "doc_id": "sample",
                    "page_index": 0,
                    "abstract": "irrelevant retrieved page",
                    "text": "irrelevant retrieved page",
                },
                {
                    "id": "page98_table",
                    "type": "table",
                    "doc_id": "sample",
                    "page_index": 97,
                    "abstract": "file size table on page 98",
                    "text": "<table><tr><td>file</td><td>size</td></tr></table>",
                },
            ]
            with open(os.path.join(graph_dir, "nodes.jsonl"), "w", encoding="utf-8") as f:
                for node in nodes:
                    f.write(json.dumps(node) + "\n")
            Path(os.path.join(graph_dir, "edges.jsonl")).write_text("", encoding="utf-8")
            cfg = OmegaConf.create({
                "baselines": {
                    "name": "magerag",
                    "params": {},
                },
                "benchmarks": {
                    "name": "mmlongbench",
                    "evidence_graph_dir": graph_root,
                    "max_pages": 100,
                    "resolution": 144,
                    "pdf_png_dir": os.path.join(tmp_dir, "missing_pngs"),
                },
            })
            builder = MAGERAGContextBuilder(cfg)
            builder.retriever.embedding_page_count = lambda benchmark_name, sample: 100
            builder.retriever.retrieve_many = lambda benchmark_name, sample, allowed_pages: (
                [{"page_index": 0, "page_number": 1, "score": 2.0}],
                {"retrieved_pages": [{"page_index": 0, "page_number": 1, "score": 2.0}]},
            )

            messages = builder.build(
                "mmlongbench",
                {
                    "doc_id": "sample.pdf",
                    "question_id": "q1",
                    "question": "What is the sum of the two smallest file sizes in the table on page 98?",
                },
            )

        self.assertIn(97, messages.metadata["activated_pages"])
        self.assertIn("page98_table", messages.metadata["opened_node_ids"])
        self.assertTrue(any(
            item.get("action") == "AutoOpenQuestionPage"
            and item.get("page_index") == 97
            and item.get("opened_node_ids") == ["page98_table"]
            for item in messages.metadata["iteration_trace"]
        ))

    def test_question_page_indices_handles_conjoined_page_numbers(self):
        indices = _question_page_indices("What are the overlapped apps between page 21 and 62?")

        self.assertEqual(indices, [20, 61])

    def test_named_scope_specs_ignore_figure_preposition_as_label(self):
        specs = _question_named_scope_specs(
            "List the number of people in the figure in page 6 and legends in Figure A."
        )

        self.assertNotIn({"kind": "figure", "label": "in"}, specs)
        self.assertIn({"kind": "figure", "label": "A"}, specs)

    def test_named_scope_specs_detect_case_insensitive_section(self):
        section_specs = _question_named_scope_specs(
            "How many website URLs are included in the Section Internet Industry in the slides?"
        )

        self.assertIn({"kind": "section", "label": "Internet Industry"}, section_specs)

    def test_named_scope_specs_detect_numeric_section_and_appendix(self):
        section_specs = _question_named_scope_specs("How many papers are not mentioned in Section 3.4?")
        appendix_specs = _question_named_scope_specs("Among the tables in Appendix, how many columns?")

        self.assertIn({"kind": "section", "label": "3.4"}, section_specs)
        self.assertIn({"kind": "section", "label": "Appendix"}, appendix_specs)

    def test_mmlongbench_opens_nodes_on_conjoined_explicit_question_pages(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            graph_root = os.path.join(tmp_dir, "graphs")
            graph_dir = os.path.join(graph_root, "sample")
            os.makedirs(graph_dir, exist_ok=True)
            with open(os.path.join(graph_dir, "graph.json"), "w", encoding="utf-8") as f:
                json.dump({"doc_id": "sample"}, f)
            nodes = [
                {
                    "id": "page21_apps",
                    "type": "image",
                    "doc_id": "sample",
                    "page_index": 20,
                    "abstract": "VIDEO, SLIDESHARE, CONTENT, TWITTER, LINKEDIN, PODCAST",
                },
                {
                    "id": "page62_apps",
                    "type": "image",
                    "doc_id": "sample",
                    "page_index": 61,
                    "abstract": "SLIDESHARE, TWITTER, LINKEDIN, PODCAST",
                },
            ]
            with open(os.path.join(graph_dir, "nodes.jsonl"), "w", encoding="utf-8") as f:
                for node in nodes:
                    f.write(json.dumps(node) + "\n")
            Path(os.path.join(graph_dir, "edges.jsonl")).write_text("", encoding="utf-8")
            cfg = OmegaConf.create({
                "baselines": {
                    "name": "magerag",
                    "params": {},
                },
                "benchmarks": {
                    "name": "mmlongbench",
                    "evidence_graph_dir": graph_root,
                    "max_pages": 80,
                    "resolution": 144,
                    "pdf_png_dir": os.path.join(tmp_dir, "missing_pngs"),
                },
            })
            builder = MAGERAGContextBuilder(cfg)
            builder.retriever.embedding_page_count = lambda benchmark_name, sample: 80
            builder.retriever.retrieve_many = lambda benchmark_name, sample, allowed_pages: (
                [{"page_index": 0, "page_number": 1, "score": 2.0}],
                {"retrieved_pages": [{"page_index": 0, "page_number": 1, "score": 2.0}]},
            )

            messages = builder.build(
                "mmlongbench",
                {
                    "doc_id": "sample.pdf",
                    "question_id": "q1",
                    "question": "What are the overlapped apps between page 21 and 62?",
                },
            )

        self.assertIn(20, messages.metadata["activated_pages"])
        self.assertIn(61, messages.metadata["activated_pages"])
        self.assertIn("page21_apps", messages.metadata["opened_node_ids"])
        self.assertIn("page62_apps", messages.metadata["opened_node_ids"])

    def test_mmlongbench_opens_named_figure_scope_even_when_initial_retrieval_misses_it(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            graph_root = os.path.join(tmp_dir, "graphs")
            graph_dir = os.path.join(graph_root, "sample")
            os.makedirs(graph_dir, exist_ok=True)
            with open(os.path.join(graph_dir, "graph.json"), "w", encoding="utf-8") as f:
                json.dump({"doc_id": "sample"}, f)
            nodes = [
                {
                    "id": "page1_text",
                    "type": "paragraph",
                    "doc_id": "sample",
                    "page_index": 0,
                    "abstract": "irrelevant retrieved page",
                    "text": "irrelevant retrieved page",
                },
                {
                    "id": "figure5",
                    "type": "figure",
                    "doc_id": "sample",
                    "page_index": 4,
                    "abstract": "Figure 5. Clustering colors with red and blue lines.",
                    "text": "Figure 5. Clustering colors with red and blue lines.",
                },
            ]
            with open(os.path.join(graph_dir, "nodes.jsonl"), "w", encoding="utf-8") as f:
                for node in nodes:
                    f.write(json.dumps(node) + "\n")
            Path(os.path.join(graph_dir, "edges.jsonl")).write_text("", encoding="utf-8")
            cfg = OmegaConf.create({
                "baselines": {
                    "name": "magerag",
                    "params": {},
                },
                "benchmarks": {
                    "name": "mmlongbench",
                    "evidence_graph_dir": graph_root,
                    "max_pages": 8,
                    "resolution": 144,
                    "pdf_png_dir": os.path.join(tmp_dir, "missing_pngs"),
                },
            })
            builder = MAGERAGContextBuilder(cfg)
            builder.retriever.embedding_page_count = lambda benchmark_name, sample: 8
            builder.retriever.retrieve_many = lambda benchmark_name, sample, allowed_pages: (
                [{"page_index": 0, "page_number": 1, "score": 2.0}],
                {"retrieved_pages": [{"page_index": 0, "page_number": 1, "score": 2.0}]},
            )

            messages = builder.build(
                "mmlongbench",
                {
                    "doc_id": "sample.pdf",
                    "question_id": "q1",
                    "question": "Which line color in Figure 5 has no intersection?",
                },
            )

        self.assertIn(4, messages.metadata["activated_pages"])
        self.assertIn("figure5", messages.metadata["opened_node_ids"])
        self.assertTrue(any(
            item.get("action") == "AutoOpenNamedQuestionScope"
            and item.get("kind") == "figure"
            and item.get("label") == "5"
            and item.get("page_index") == 4
            for item in messages.metadata["iteration_trace"]
        ))

    def test_mmlongbench_opens_quoted_section_scope_and_following_page(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            graph_root = os.path.join(tmp_dir, "graphs")
            graph_dir = os.path.join(graph_root, "sample")
            os.makedirs(graph_dir, exist_ok=True)
            with open(os.path.join(graph_dir, "graph.json"), "w", encoding="utf-8") as f:
                json.dump({"doc_id": "sample"}, f)
            nodes = [
                {
                    "id": "page1_text",
                    "type": "paragraph",
                    "doc_id": "sample",
                    "page_index": 0,
                    "abstract": "irrelevant retrieved page",
                    "text": "irrelevant retrieved page",
                },
                {
                    "id": "internet_title",
                    "type": "title",
                    "doc_id": "sample",
                    "page_index": 17,
                    "abstract": "Internet Industry",
                    "text": "Internet Industry",
                },
                {
                    "id": "internet_urls",
                    "type": "table",
                    "doc_id": "sample",
                    "page_index": 18,
                    "abstract": "Website URLs listed for Internet Industry",
                    "text": "example.com other.example",
                },
            ]
            with open(os.path.join(graph_dir, "nodes.jsonl"), "w", encoding="utf-8") as f:
                for node in nodes:
                    f.write(json.dumps(node) + "\n")
            Path(os.path.join(graph_dir, "edges.jsonl")).write_text("", encoding="utf-8")
            cfg = OmegaConf.create({
                "baselines": {
                    "name": "magerag",
                    "params": {},
                },
                "benchmarks": {
                    "name": "mmlongbench",
                    "evidence_graph_dir": graph_root,
                    "max_pages": 24,
                    "resolution": 144,
                    "pdf_png_dir": os.path.join(tmp_dir, "missing_pngs"),
                },
            })
            builder = MAGERAGContextBuilder(cfg)
            builder.retriever.embedding_page_count = lambda benchmark_name, sample: 24
            builder.retriever.retrieve_many = lambda benchmark_name, sample, allowed_pages: (
                [{"page_index": 0, "page_number": 1, "score": 2.0}],
                {"retrieved_pages": [{"page_index": 0, "page_number": 1, "score": 2.0}]},
            )

            messages = builder.build(
                "mmlongbench",
                {
                    "doc_id": "sample.pdf",
                    "question_id": "q1",
                    "question": 'How many website URLs are in the section "Internet Industry"?',
                },
            )

        self.assertIn(17, messages.metadata["activated_pages"])
        self.assertIn("internet_title", messages.metadata["opened_node_ids"])
        self.assertIn("internet_urls", messages.metadata["opened_node_ids"])

    def test_mmlongbench_opens_faq_scope_and_following_question_pages(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            graph_root = os.path.join(tmp_dir, "graphs")
            graph_dir = os.path.join(graph_root, "sample")
            os.makedirs(graph_dir, exist_ok=True)
            with open(os.path.join(graph_dir, "graph.json"), "w", encoding="utf-8") as f:
                json.dump({"doc_id": "sample"}, f)
            nodes = [
                {
                    "id": "page1_text",
                    "type": "paragraph",
                    "doc_id": "sample",
                    "page_index": 0,
                    "abstract": "irrelevant retrieved page",
                    "text": "irrelevant retrieved page",
                },
                {
                    "id": "faq_heading",
                    "type": "title",
                    "doc_id": "sample",
                    "page_index": 23,
                    "abstract": "Frequently Asked Questions section heading.",
                    "text": "Frequently Asked Questions",
                },
                {
                    "id": "first_faq",
                    "type": "title",
                    "doc_id": "sample",
                    "page_index": 24,
                    "abstract": "What happens to my certification?",
                    "text": "What happens to my certification?",
                },
                {
                    "id": "second_faq",
                    "type": "title",
                    "doc_id": "sample",
                    "page_index": 25,
                    "abstract": "Next-Gen Recruiter seems to eliminate Boolean Search.",
                    "text": "Next-Gen Recruiter seems to eliminate Boolean Search.",
                },
            ]
            with open(os.path.join(graph_dir, "nodes.jsonl"), "w", encoding="utf-8") as f:
                for node in nodes:
                    f.write(json.dumps(node) + "\n")
            Path(os.path.join(graph_dir, "edges.jsonl")).write_text("", encoding="utf-8")
            cfg = OmegaConf.create({
                "baselines": {
                    "name": "magerag",
                    "params": {},
                },
                "benchmarks": {
                    "name": "mmlongbench",
                    "evidence_graph_dir": graph_root,
                    "max_pages": 30,
                    "resolution": 144,
                    "pdf_png_dir": os.path.join(tmp_dir, "missing_pngs"),
                },
            })
            builder = MAGERAGContextBuilder(cfg)
            builder.retriever.embedding_page_count = lambda benchmark_name, sample: 30
            builder.retriever.retrieve_many = lambda benchmark_name, sample, allowed_pages: (
                [{"page_index": 0, "page_number": 1, "score": 2.0}],
                {"retrieved_pages": [{"page_index": 0, "page_number": 1, "score": 2.0}]},
            )

            messages = builder.build(
                "mmlongbench",
                {
                    "doc_id": "sample.pdf",
                    "question_id": "q1",
                    "question": "What is the second FAQ shown in this slides?",
                },
            )

        self.assertIn("faq_heading", messages.metadata["opened_node_ids"])
        self.assertIn("first_faq", messages.metadata["opened_node_ids"])
        self.assertIn("second_faq", messages.metadata["opened_node_ids"])
        self.assertTrue(any(
            item.get("action") == "AutoOpenNamedQuestionScope"
            and item.get("kind") == "faq"
            and item.get("scope_page_indices") == [23, 24, 25, 26]
            for item in messages.metadata["iteration_trace"]
        ))

    def test_mmlongbench_limits_quoted_scope_to_explicit_question_page(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            graph_root = os.path.join(tmp_dir, "graphs")
            graph_dir = os.path.join(graph_root, "sample")
            os.makedirs(graph_dir, exist_ok=True)
            with open(os.path.join(graph_dir, "graph.json"), "w", encoding="utf-8") as f:
                json.dump({"doc_id": "sample"}, f)
            nodes = [
                {
                    "id": "page1_text",
                    "type": "paragraph",
                    "doc_id": "sample",
                    "page_index": 0,
                    "abstract": "irrelevant retrieved page",
                    "text": "irrelevant retrieved page",
                },
                {
                    "id": "target_scope",
                    "type": "title",
                    "doc_id": "sample",
                    "page_index": 8,
                    "abstract": "Self-Correction section heading.",
                    "text": "Self-Correction",
                },
                {
                    "id": "later_scope",
                    "type": "title",
                    "doc_id": "sample",
                    "page_index": 11,
                    "abstract": "Self-Correction mentioned again.",
                    "text": "Self-Correction",
                },
                {
                    "id": "later_noise",
                    "type": "paragraph",
                    "doc_id": "sample",
                    "page_index": 12,
                    "abstract": "unrelated later content",
                    "text": "unrelated later content",
                },
            ]
            with open(os.path.join(graph_dir, "nodes.jsonl"), "w", encoding="utf-8") as f:
                for node in nodes:
                    f.write(json.dumps(node) + "\n")
            Path(os.path.join(graph_dir, "edges.jsonl")).write_text("", encoding="utf-8")
            cfg = OmegaConf.create({
                "baselines": {
                    "name": "magerag",
                    "params": {},
                },
                "benchmarks": {
                    "name": "mmlongbench",
                    "evidence_graph_dir": graph_root,
                    "max_pages": 16,
                    "resolution": 144,
                    "pdf_png_dir": os.path.join(tmp_dir, "missing_pngs"),
                },
            })
            builder = MAGERAGContextBuilder(cfg)
            builder.retriever.embedding_page_count = lambda benchmark_name, sample: 16
            builder.retriever.retrieve_many = lambda benchmark_name, sample, allowed_pages: (
                [{"page_index": 0, "page_number": 1, "score": 2.0}],
                {"retrieved_pages": [{"page_index": 0, "page_number": 1, "score": 2.0}]},
            )

            messages = builder.build(
                "mmlongbench",
                {
                    "doc_id": "sample.pdf",
                    "question_id": "q1",
                    "question": 'How many papers are not mentioned in the "Self-Correction" section of page 9?',
                },
            )

        self.assertIn("target_scope", messages.metadata["opened_node_ids"])
        self.assertNotIn("later_noise", messages.metadata["opened_node_ids"])
        self.assertTrue(any(
            item.get("action") == "AutoOpenNamedQuestionScope"
            and item.get("kind") == "quoted_scope"
            and item.get("scope_page_indices") == [8]
            for item in messages.metadata["iteration_trace"]
        ))

    def test_renderer_can_disable_opened_node_text_for_mmlongbench_only(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            state = EvidenceAgentState(EvidenceGraphStore(self._write_graph(tmp_dir), allowed_pages=[0]))
            state.execute(ActivatePage(0, "initial_retrieval"))
            state.execute(ActivateNode("n1"))
            state.execute(OpenNode("n1"))
            cfg = OmegaConf.create({
                "benchmarks": {},
                "baselines": {
                    "reader": {
                        "mode": "compact",
                        "include_opened_node_text": True,
                        "include_opened_node_text_mmlongbench": False,
                    }
                },
            })

            mmlong = ReaderRenderer(cfg, include_page_images=False).render(
                "mmlongbench", {"question": "What does the source evidence say?"}, state
            )
            longdoc = ReaderRenderer(cfg, include_page_images=False).render(
                "longdocurl", {"question": "What does the source evidence say?"}, state
            )

        self.assertNotIn("<opened_evidence_text>", mmlong[0]["text"])
        self.assertIn("<opened_evidence_text>", longdoc[0]["text"])

    def test_reader_renderer_prioritizes_question_scope_pages_for_mmlongbench(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            state = EvidenceAgentState(EvidenceGraphStore(self._write_graph(tmp_dir), allowed_pages=[0, 1]))
            state.execute(ActivatePage(0, "initial_retrieval"))
            state.execute(ActivatePage(1, "question_page_scope"))
            cfg = OmegaConf.create({
                "benchmarks": {},
                "baselines": {"reader": {"mode": "compact"}},
            })

            content = ReaderRenderer(cfg, include_page_images=False).render(
                "mmlongbench",
                {"question": "What is on page 2?"},
                state,
            )

        self.assertIn("<retrieved_pages>2, 1</retrieved_pages>", content[0]["text"])

    def test_mmlongbench_allowed_pages_use_embedding_page_count(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            graph_root = os.path.join(tmp_dir, "graphs")
            self._write_graph(os.path.join(graph_root, "sample"))
            cfg = OmegaConf.create({
                "baselines": {"name": "magerag", "params": {}},
                "benchmarks": {
                    "name": "mmlongbench",
                    "evidence_graph_dir": graph_root,
                    "max_pages": 120,
                    "resolution": 144,
                    "pdf_png_dir": os.path.join(tmp_dir, "missing_pngs"),
                },
            })
            builder = MAGERAGContextBuilder(cfg)
            builder.retriever.embedding_page_count = lambda benchmark_name, sample: 2

            messages = builder.build("mmlongbench", {"doc_id": "sample.pdf", "question_id": "q1", "question": "Q?"})

        self.assertEqual(messages.metadata["allowed_pages"], [0, 1])

    def _write_graph(self, root, long_text=False):
        graph_dir = root
        os.makedirs(graph_dir, exist_ok=True)
        with open(os.path.join(graph_dir, "graph.json"), "w", encoding="utf-8") as f:
            json.dump({"doc_id": "sample"}, f)
        text = ("x" * 9000) if long_text else "needle source evidence"
        nodes = [
            {"id": "n1", "type": "paragraph", "doc_id": "sample", "page_index": 0, "abstract": "needle source", "text": text},
            {"id": "n3", "type": "paragraph", "doc_id": "sample", "page_index": 0, "abstract": "needle sibling", "text": "sibling evidence"},
            {"id": "n_title", "type": "title", "doc_id": "sample", "page_index": 0, "abstract": "Important Section Title", "text": "Important Section Title"},
            {"id": "n2", "type": "table", "doc_id": "sample", "page_index": 1, "abstract": "needle target", "text": "target evidence"},
        ]
        with open(os.path.join(graph_dir, "nodes.jsonl"), "w", encoding="utf-8") as f:
            for node in nodes:
                f.write(json.dumps(node) + "\n")
        with open(os.path.join(graph_dir, "edges.jsonl"), "w", encoding="utf-8") as f:
            f.write(json.dumps({"id": "e1", "source": "n1", "target": "n2", "type": "semantic", "relation": "related"}) + "\n")
        return graph_dir


if __name__ == "__main__":
    unittest.main()
