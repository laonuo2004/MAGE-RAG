import json
import os
import tempfile
import unittest
import xml.etree.ElementTree as ET
from pathlib import Path

import torch
from omegaconf import OmegaConf

from baselines.magerag.actions import ActionResult, CandidateAction
from baselines.magerag.actions import ActivateNode, ActivatePage, OpenNode, PruneNode, SearchEvidence
from baselines.magerag.builder import MAGERAGContextBuilder
from baselines.magerag.candidate_generator import CandidateGenerator
from baselines.magerag.evaluator import EvaluatorDecision, XMLEvaluator, parse_agent_decision_xml
from baselines.magerag.graph_store import EvidenceGraphStore
from baselines.magerag.retrieval import ColPaliTop1Retriever
from baselines.magerag.renderer import ReaderRenderer
from utils.image_crop import normalized_bbox_1000_to_pixel_box
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

        self.assertEqual(retriever.top_k_for("longdocurl"), 3)
        self.assertEqual(retriever.top_k_for("mmlongbench"), 3)

    def test_default_mage_config_is_compact_and_uses_new_schema(self):
        cfg = OmegaConf.load("configs/baselines/magerag.yaml")

        self.assertEqual(str(cfg.name), "magerag")
        self.assertEqual(int(cfg.params.top_k), 3)
        self.assertNotIn("graph_escape", cfg.params)
        self.assertNotIn("online_agent", cfg.params)
        self.assertEqual(set(cfg.params.keys()), {"top_k"})
        self.assertEqual(list(cfg.result_name_params), [
            "params.top_k",
            "controller.mode",
            "controller.watchdog_iterations",
            "evaluator.max_selected_actions_per_iteration",
            "graph.mode",
        ])
        self.assertEqual(str(cfg.models.evaluator), "Qwen3-VL-8B-Instruct")
        self.assertTrue(bool(cfg.evaluator.include_few_shot_examples))
        self.assertNotIn("prompt_style", cfg.evaluator)
        self.assertNotIn("reason_max_words", cfg.evaluator)
        self.assertEqual(int(cfg.evaluator.raw_text_char_limit), 1200)
        self.assertNotIn("max_candidate_actions", cfg.evaluator)
        self.assertEqual(int(cfg.evaluator.max_selected_actions_per_iteration), 5)
        self.assertEqual(int(cfg.evaluator.max_total_selected_actions), 100)
        self.assertEqual(int(cfg.evaluator.recent_trace_limit), 25)
        self.assertEqual(int(cfg.controller.watchdog_iterations), 10)
        self.assertEqual(int(cfg.controller.watchdog_repeated_noop_rounds), 2)
        self.assertEqual(str(cfg.controller.mode), "full")
        self.assertTrue(bool(cfg.controller.enable_online_controller))
        self.assertTrue(bool(cfg.controller.enable_search))
        self.assertTrue(bool(cfg.controller.enable_prune))
        self.assertNotIn("auto_open_max_nodes_per_page", cfg.controller)
        self.assertEqual(int(cfg.controller.final_open_active_node_limit), 100)
        self.assertEqual(str(cfg.graph.mode), "full_graph")
        self.assertIsNone(cfg.graph.enabled_edge_types)
        self.assertEqual(list(cfg.graph.disabled_edge_types), [])
        self.assertIsNone(cfg.graph.max_edges_per_node)
        self.assertTrue(bool(cfg.trace.save_candidate_actions))
        self.assertTrue(bool(cfg.trace.save_action_rationales))
        self.assertEqual(str(cfg.reader.not_answerable_text), "Not answerable.")
        self.assertTrue(bool(cfg.reader.include_self_check_instruction))
        self.assertTrue(bool(cfg.reader.include_page_images))
        self.assertTrue(bool(cfg.reader.include_opened_node_images))
        self.assertTrue(bool(cfg.reader.include_opened_node_crops))
        self.assertEqual(int(cfg.reader.opened_node_text_char_limit), 1200)
        self.assertNotIn("mode", cfg.reader)
        self.assertNotIn("prompt_style", cfg.reader)
        self.assertNotIn("raw_text_char_limit", cfg.reader)
        self.assertNotIn("include_image_page_labels", cfg.reader)
        self.assertNotIn("include_opened_node_text", cfg.reader)
        self.assertNotIn("mmlongbench_prompt", cfg.reader)
        self.assertNotIn("mmlongbench_page_text_char_limit", cfg.reader)
        self.assertNotIn("mmlongbench_page_text_max_pages", cfg.reader)
        self.assertNotIn("mmlongbench_include_opened_node_crops", cfg.reader)
        self.assertNotIn("mmlongbench_max_opened_node_crops", cfg.reader)
        self.assertNotIn("agent", cfg)
        self.assertIn("evaluator", cfg)
        self.assertNotIn("safety", cfg)

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

    def test_activate_node_auto_activates_parent_page(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            state = EvidenceAgentState(EvidenceGraphStore(self._write_graph(tmp_dir), allowed_pages=[0]))

            result = state.execute(ActivateNode("n1"))

        self.assertTrue(result.ok)
        self.assertEqual(state.state_of("page:0"), ACTIVE)
        self.assertEqual(state.state_of("n1"), ACTIVE)

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

    def test_open_node_rejects_active_child_when_parent_page_is_pruned(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            state = EvidenceAgentState(EvidenceGraphStore(self._write_graph(tmp_dir), allowed_pages=[0]))
            state.execute(ActivatePage(0, "initial_retrieval"))
            state.execute(ActivateNode("n1"))
            state.execute(PruneNode("page:0", "irrelevant page"))

            result = state.execute(OpenNode("n1"))

        self.assertFalse(result.ok)
        self.assertEqual(state.state_of("n1"), ACTIVE)

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

    def test_pruning_page_hides_page_and_descendants_from_candidates(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            state = EvidenceAgentState(EvidenceGraphStore(self._write_graph(tmp_dir), allowed_pages=[0, 1]))

            result = state.execute(PruneNode("page:0", "Page is unrelated to the answer."))
            candidates = CandidateGenerator(state.graph).generate(state)
            context_xml = XMLEvaluator("model").build_context_xml("question", state, candidates)

        self.assertTrue(result.ok)
        self.assertEqual(state.state_of("page:0"), PRUNED)
        self.assertFalse(any(candidate.payload.get("node_id") in {"n1", "n3", "n_title"} for candidate in candidates))
        self.assertNotIn('id="page:0"', context_xml)
        self.assertNotIn('id="n1"', context_xml)

    def test_evaluator_context_excludes_pruned_element_nodes(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            state = EvidenceAgentState(EvidenceGraphStore(self._write_graph(tmp_dir), allowed_pages=[0]))
            state.execute(ActivatePage(0, "initial_retrieval"))
            state.execute(ActivateNode("n1"))
            state.execute(OpenNode("n1"))
            state.execute(PruneNode("n1", "irrelevant"))

            context_xml = XMLEvaluator("model").build_context_xml("question", state, [])

        evidence_state = ET.fromstring(context_xml).find("evidence_state")
        evidence_xml = ET.tostring(evidence_state, encoding="unicode")
        self.assertIn('id="page:0"', evidence_xml)
        self.assertNotIn('id="n1"', evidence_xml)
        self.assertNotIn("irrelevant", evidence_xml)

    def test_evaluator_image_refs_exclude_opened_nodes_on_pruned_pages(self):
        from PIL import Image

        with tempfile.TemporaryDirectory() as tmp_dir:
            image_path = os.path.join(tmp_dir, "node.png")
            Image.new("RGB", (4, 4), color=(255, 0, 0)).save(image_path)
            state = EvidenceAgentState(EvidenceGraphStore(self._write_graph(tmp_dir), allowed_pages=[0, 1]))
            state.graph.nodes["n1"]["image_path"] = image_path
            state.execute(ActivatePage(0, "initial_retrieval"))
            state.execute(ActivateNode("n1"))
            state.execute(OpenNode("n1"))
            state.execute(PruneNode("page:0", "irrelevant page"))

            evaluator = XMLEvaluator("model", include_images_for_opened_nodes=True)
            content_parts = evaluator._opened_node_image_parts(state)
            image_refs = evaluator._opened_node_image_refs(state)

        self.assertEqual(content_parts, [])
        self.assertEqual(image_refs, [])

    def test_relation_edges_emit_activate_node_candidates_directly(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            state = EvidenceAgentState(EvidenceGraphStore(self._write_graph(tmp_dir), allowed_pages=[0, 1]))
            state.execute(ActivatePage(0, "initial_retrieval"))
            state.execute(ActivateNode("n1"))

            candidates = CandidateGenerator(state.graph).generate(state)

        relation_candidates = [
            candidate for candidate in candidates
            if candidate.action_type == "ActivateNode" and candidate.payload.get("node_id") == "n2"
        ]
        self.assertEqual(len(relation_candidates), 1)
        self.assertEqual(relation_candidates[0].payload["source_node_id"], "n1")
        self.assertEqual(relation_candidates[0].payload["edge_type"], "semantic")
        self.assertEqual(relation_candidates[0].payload["relation"], "related")

    def test_relation_activate_node_candidate_auto_activates_target_page(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            state = EvidenceAgentState(EvidenceGraphStore(self._write_graph(tmp_dir), allowed_pages=[0, 1]))
            state.execute(ActivatePage(0, "initial_retrieval"))
            state.execute(ActivateNode("n1"))
            candidates = CandidateGenerator(state.graph).generate(state)
            relation_candidate = next(
                candidate for candidate in candidates
                if candidate.action_type == "ActivateNode" and candidate.payload.get("node_id") == "n2"
            )

            result = state.execute(ActivateNode(relation_candidate.payload["node_id"]))

        self.assertTrue(result.ok)
        self.assertIn("page:1", state.active_node_ids())
        self.assertIn("n2", state.active_node_ids())

    def test_candidate_generator_includes_reverse_logically_undirected_edges(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            graph_dir = Path(self._write_graph(tmp_dir))
            with open(graph_dir / "edges.jsonl", "a", encoding="utf-8") as handle:
                handle.write(json.dumps({"id": "read_prev", "source": "n3", "target": "n1", "type": "reading_order", "relation": "next"}) + "\n")
                handle.write(json.dumps({"id": "layout_left", "source": "n3", "target": "n1", "type": "layout", "relation": "left_of"}) + "\n")
                handle.write(json.dumps({"id": "section_parent", "source": "n_title", "target": "n1", "type": "section_hierarchy", "relation": "contains_block"}) + "\n")
                handle.write(json.dumps({"id": "contains_page", "source": "page:0", "target": "n1", "type": "containment", "relation": "contains"}) + "\n")
            state = EvidenceAgentState(EvidenceGraphStore(graph_dir, allowed_pages=[0]))
            state.execute(ActivatePage(0, "initial_retrieval"))
            state.execute(ActivateNode("n1"))

            candidates = CandidateGenerator(state.graph).generate(state)

        relation_edge_ids = {
            candidate.payload["edge_id"]
            for candidate in candidates
            if candidate.action_type == "ActivateNode" and "edge_id" in candidate.payload
        }
        self.assertIn("read_prev", relation_edge_ids)
        self.assertIn("layout_left", relation_edge_ids)
        self.assertIn("section_parent", relation_edge_ids)
        self.assertNotIn("contains_page", relation_edge_ids)

    def test_graph_store_filters_edges_by_graph_mode(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            graph_dir = Path(self._write_graph(tmp_dir))
            with open(graph_dir / "edges.jsonl", "a", encoding="utf-8") as handle:
                handle.write(json.dumps({"id": "layout_left", "source": "n1", "target": "n3", "type": "layout", "relation": "left_of"}) + "\n")
                handle.write(json.dumps({"id": "read_next", "source": "n1", "target": "n3", "type": "reading_order", "relation": "next"}) + "\n")

            semantic_graph = EvidenceGraphStore(graph_dir, allowed_pages=[0, 1], graph_mode="semantic_graph")
            structural_graph = EvidenceGraphStore(graph_dir, allowed_pages=[0, 1], graph_mode="structural_graph")
            page_only_graph = EvidenceGraphStore(graph_dir, allowed_pages=[0, 1], graph_mode="page_only")

        self.assertEqual(set(semantic_graph.edges), {"e1"})
        self.assertEqual(set(structural_graph.edges), {"read_next"})
        self.assertEqual(set(page_only_graph.nodes), {"page:0", "page:1"})
        self.assertEqual(page_only_graph.edges, {})

    def test_graph_store_applies_enabled_disabled_and_per_node_edge_limits(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            graph_dir = Path(self._write_graph(tmp_dir))
            with open(graph_dir / "edges.jsonl", "a", encoding="utf-8") as handle:
                handle.write(json.dumps({"id": "semantic_2", "source": "n1", "target": "n3", "type": "semantic", "relation": "related"}) + "\n")
                handle.write(json.dumps({"id": "layout_left", "source": "n1", "target": "n3", "type": "layout", "relation": "left_of"}) + "\n")

            graph = EvidenceGraphStore(
                graph_dir,
                allowed_pages=[0, 1],
                enabled_edge_types=["semantic", "layout"],
                disabled_edge_types=["layout"],
                max_edges_per_node=1,
            )

        self.assertEqual(set(graph.edges), {"e1"})
        self.assertEqual([edge["id"] for edge in graph.out_edges["n1"]], ["e1"])

    def test_reverse_relation_candidate_targets_source_endpoint(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            graph_dir = Path(self._write_graph(tmp_dir))
            with open(graph_dir / "edges.jsonl", "a", encoding="utf-8") as handle:
                handle.write(json.dumps({"id": "read_prev", "source": "n3", "target": "n1", "type": "reading_order", "relation": "next"}) + "\n")
            state = EvidenceAgentState(EvidenceGraphStore(graph_dir, allowed_pages=[0]))
            state.execute(ActivatePage(0, "initial_retrieval"))
            state.execute(ActivateNode("n1"))

            candidates = CandidateGenerator(state.graph).generate(state)

        self.assertTrue(any(
            candidate.action_type == "ActivateNode" and candidate.payload["node_id"] == "n3"
            for candidate in candidates
        ))

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

    def test_candidate_generator_does_not_emit_open_node_candidates(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            state = EvidenceAgentState(EvidenceGraphStore(self._write_graph(tmp_dir), allowed_pages=[0]))
            state.execute(ActivatePage(0, "initial_retrieval"))
            state.execute(ActivateNode("n1"))
            candidates = CandidateGenerator(state.graph).generate(state)

        self.assertFalse(any(candidate.action_type == "OpenNode" for candidate in candidates))

    def test_search_evidence_records_query_without_lexical_node_results(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            state = EvidenceAgentState(EvidenceGraphStore(self._write_graph(tmp_dir), allowed_pages=[0]))

            result = state.execute(SearchEvidence("needle"))

        self.assertTrue(result.ok)
        self.assertEqual(result.payload["query"], "needle")
        self.assertNotIn("results", result.payload)

    def test_xml_evaluator_output_parses_into_internal_requests(self):
        decision = parse_agent_decision_xml(
            """
            <think>Evaluate the evidence before selecting actions.</think>
            <agent_decision>
              <stop>false</stop>
              <selected_actions><action index="1"/></selected_actions>
              <open_requests><node id="n1"/></open_requests>
              <search_request><query>more evidence</query></search_request>
              <prune_requests><node id="n2"><reason>bad</reason></node></prune_requests>
            </agent_decision>
            """
        )

        self.assertFalse(decision.stop)
        self.assertEqual(decision.selected_actions[0]["candidate_index"], 1)
        self.assertEqual(decision.open_requests[0]["node_id"], "n1")
        self.assertEqual(decision.search_query, "more evidence")
        self.assertEqual(decision.prune_requests[0]["node_id"], "n2")
        self.assertEqual(decision.prune_requests[0]["reason"], "bad")

    def test_xml_evaluator_parses_page_prune_requests(self):
        decision = parse_agent_decision_xml(
            """
            <think>Page 4 is unrelated noise and should be removed from context.</think>
            <agent_decision>
              <prune_requests>
                <page id="doc:page:4"><reason>Unrelated appendix page consumes context.</reason></page>
              </prune_requests>
            </agent_decision>
            """
        )

        self.assertEqual(decision.prune_requests, [{
            "node_id": "doc:page:4",
            "reason": "Unrelated appendix page consumes context.",
        }])

    def test_xml_evaluator_accepts_numeric_action_indexes(self):
        decision = parse_agent_decision_xml(
            """
            <agent_decision>
              <selected_actions><action index="2" utility="0.9"><reason>open it</reason></action></selected_actions>
            </agent_decision>
            """
        )

        self.assertEqual(decision.selected_actions[0]["candidate_index"], 2)
        self.assertNotIn("reason", decision.selected_actions[0])

    def test_xml_evaluator_parses_prune_reason_only(self):
        decision = parse_agent_decision_xml(
            """
            <agent_decision>
              <prune_requests>
                <node id="n1"><reason>Not relevant to the requested adult dose.</reason></node>
              </prune_requests>
            </agent_decision>
            """
        )

        self.assertEqual(decision.prune_requests, [{
            "node_id": "n1",
            "reason": "Not relevant to the requested adult dose.",
        }])

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
        self.assertIn("Select at most 4 numbered ActivateNode actions per round.", prompt)
        self.assertIn("The ActivateNode limit does not constrain OpenNode, SearchEvidenceRequest, or PruneNodeRequest.", prompt)
        self.assertIn("Before deciding, output one &lt;think&gt;...&lt;/think&gt; block with detailed deliberation.", prompt)
        self.assertIn("After &lt;think&gt;, return exactly one &lt;agent_decision&gt; XML document.", prompt)
        self.assertNotIn("<thinking>", prompt)
        self.assertIn("<problem>", prompt)
        self.assertIn("<example_output>", prompt)
        self.assertIn('&lt;prune_requests&gt;&lt;page id="visible page id"&gt;', prompt)
        self.assertIn('&lt;prune_requests&gt;&lt;node id="visible element node id"&gt;', prompt)
        self.assertNotIn("<allowed_scope>", prompt)
        self.assertNotIn("summarize_requests", prompt)
        self.assertNotIn("For questions naming specific pages or slides", prompt)
        self.assertNotIn("For list or exhaustive questions", prompt)

    def test_xml_evaluator_prompt_escapes_xml_sensitive_context_with_xml_blocks(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            state = EvidenceAgentState(EvidenceGraphStore(self._write_graph(tmp_dir), allowed_pages=[0]))
            state.graph.nodes["n1"]["abstract"] = "Revenue < cost & rising"
            state.graph.nodes["n1"]["text"] = "Detailed value is A < B & C."
            state.execute(ActivatePage(0, "initial_retrieval"))
            state.execute(ActivateNode("n1"))
            state.execute(OpenNode("n1"))

            prompt = XMLEvaluator("model").build_prompt(
                "Which value uses < and & symbols?",
                state,
                [CandidateAction("act:OpenNode:n1", "OpenNode", {"node_id": "n1"}, "Open < node & inspect")],
            )

        root = ET.fromstring(f"<root>{prompt}</root>")
        self.assertEqual(root.find(".//agent_step_context/question").text, "Which value uses < and & symbols?")
        self.assertIn("Detailed value is A < B & C.", root.find(".//agent_step_context/evidence_state/page/node/content").text)
        self.assertEqual(root.find(".//candidate_actions/OpenNode/template").text, '<open_requests><node id="active non-page node id"/></open_requests>')
        prune_template = root.find(".//agent_step_context/candidate_actions/PruneNodeRequest/template").text
        self.assertIn('<prune_requests><page id="visible page id">', prune_template)
        self.assertIn('<prune_requests><node id="visible element node id">', prune_template)

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

    def test_evaluator_context_groups_activate_node_candidates_and_hides_internal_ids(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            state = EvidenceAgentState(EvidenceGraphStore(self._write_graph(tmp_dir), allowed_pages=[0, 1]))
            state.execute(ActivatePage(0, "initial_retrieval"))
            state.execute(ActivateNode("n1"))
            candidates = CandidateGenerator(state.graph).generate(state)

            xml = XMLEvaluator("model").build_context_xml("question", state, candidates)

        root = ET.fromstring(xml)
        activate_actions = root.findall(".//candidate_actions/ActivateNode/action")
        self.assertTrue(activate_actions)
        relation_action = next(action for action in activate_actions if action.attrib.get("edge_type") == "semantic")
        self.assertEqual(relation_action.attrib["source"], "n1")
        self.assertEqual(relation_action.attrib["relation"], "related")
        self.assertNotIn("node_id", relation_action.attrib)
        self.assertNotIn("edge_id", relation_action.attrib)
        self.assertNotIn("page_index", relation_action.attrib)
        self.assertNotIn("candidate_id", relation_action.attrib)

    def test_xml_evaluator_extracts_decision_from_code_fence(self):
        decision = parse_agent_decision_xml(
            "Here is the XML:\n```xml\n<think>Enough evidence.</think><agent_decision><stop>true</stop></agent_decision>\n```"
        )

        self.assertTrue(decision.stop)

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

    def test_selected_action_without_candidate_index_is_rejected_and_traced(self):
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
                EvaluatorDecision(selected_actions=[{}]),
                "<agent_decision/>",
            )

            stop_reason = builder._run_agent("mmlongbench", "question", state, client=object())

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
                    EvaluatorDecision(selected_actions=[{"candidate_index": index}]),
                    "<agent_decision/>",
                )

            builder.evaluator.call = fake_call

            builder._run_agent("mmlongbench", "question", state, client=object())

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
                    return EvaluatorDecision(stop=True), "<agent_decision><stop>true</stop></agent_decision>"
                candidate_indexes = [
                    index
                    for index, candidate in enumerate(candidates, start=1)
                    if candidate.id in {
                        "act:ActivateNode:n1",
                        "act:ActivateNode:n3",
                        "act:ActivateNode:n_title",
                        "act:ActivateNode:n2",
                    }
                ]
                return (
                    EvaluatorDecision(selected_actions=[
                        {"candidate_index": candidate_index}
                        for candidate_index in candidate_indexes + [999]
                    ]),
                    "<agent_decision/>",
                )

            builder.evaluator.call = fake_call

            builder._run_agent("mmlongbench", "question", state, client=object())

        active_nodes = state.active_node_ids()
        self.assertIn("n1", active_nodes)
        self.assertIn("n3", active_nodes)
        self.assertNotIn("n_title", active_nodes)
        truncation_trace = next(item for item in state.trace if item.get("action") == "TruncatedSelectedActions")
        self.assertEqual(truncation_trace["original_count"], 4)
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
                        {"candidate_index": index}
                        for index in range(1, 4)
                    ]),
                    "<agent_decision/>",
                )

            builder.evaluator.call = fake_call

            stop_reason = builder._run_agent("mmlongbench", "question", state, client=object())

        self.assertEqual(stop_reason, "watchdog_total_selected_actions")
        self.assertEqual(call_count, 2)
        budget_trace = next(item for item in state.trace if item.get("action") == "SelectedActionBudgetReached")
        self.assertEqual(budget_trace["executed_total"], 4)
        self.assertEqual(budget_trace["limit"], 3)

    def test_empty_candidate_numeric_selection_is_rejected(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            state = EvidenceAgentState(EvidenceGraphStore(self._write_graph(tmp_dir), allowed_pages=[0]))
            builder = MAGERAGContextBuilder(OmegaConf.create({
                "baselines": {
                    "name": "magerag",
                }
            }))
            builder.evaluator.call = lambda client, question, state, candidates: (
                EvaluatorDecision(selected_actions=[{"candidate_index": 1}]),
                "<agent_decision/>",
            )

            stop_reason = builder._run_agent("mmlongbench", "question", state, client=object())

        self.assertEqual(stop_reason, "watchdog_repeated_noop")
        self.assertEqual(state.validation_errors[-1]["action_type"], "SelectedAction")
        self.assertEqual(state.trace[-1]["action"], "InvalidCandidate")

    def test_evaluator_decision_trace_records_input_without_base64_images(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            graph_dir = Path(self._write_graph(tmp_dir))
            image_path = os.path.join(tmp_dir, "crop.png")
            with open(image_path, "wb") as handle:
                handle.write(b"fake image")
            lines = (graph_dir / "nodes.jsonl").read_text(encoding="utf-8").splitlines()
            rows = [json.loads(line) for line in lines]
            row_index = next(index for index, row in enumerate(rows) if row["id"] == "n1")
            node = rows[row_index]
            node["image_path"] = image_path
            rows[row_index] = node
            (graph_dir / "nodes.jsonl").write_text(
                "".join(json.dumps(row) + "\n" for row in rows),
                encoding="utf-8",
            )
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
                EvaluatorDecision(stop=True),
                "<agent_decision><stop>true</stop></agent_decision>",
            )

            builder._run_agent("mmlongbench", "question", state, client=object())

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
        content = ET.fromstring(xml).find(".//evidence_state/page/node/content")
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
                EvaluatorDecision(stop=True),
                "<agent_decision><stop>true</stop></agent_decision>",
            )

            messages = builder.build(
                "mmlongbench",
                {"doc_id": "sample.pdf", "question_id": "q1", "question": "What does the source evidence say?"},
                client=object(),
            )

        decision_trace = next(item for item in messages.metadata["iteration_trace"] if item.get("action") == "EvaluatorDecision")
        self.assertEqual(messages.metadata["opened_node_ids"], [])
        self.assertEqual(decision_trace["evaluator_input"]["opened_image_refs"], [])
        self.assertNotIn('state="opened"', decision_trace["evaluator_input"]["context_xml"])

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
                        EvaluatorDecision(stop=True),
                        "<agent_decision><stop>true</stop></agent_decision>",
                    )
                index = next(index for index, candidate in enumerate(candidates, start=1) if candidate.id == "act:ActivateNode:n1")
                return (
                    EvaluatorDecision(selected_actions=[{"candidate_index": index}]),
                    f"<agent_decision><selected_actions><action index=\"{index}\"/></selected_actions></agent_decision>",
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

    def test_online_search_decision_activates_colpali_pages_without_opening_nodes(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            graph_root = os.path.join(tmp_dir, "graphs")
            graph_dir = Path(self._write_graph(os.path.join(graph_root, "sample")))
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
            search_calls = []

            def fake_retrieve_from_query(benchmark_name, sample, query, allowed_pages, excluded_pages=None):
                search_calls.append((benchmark_name, query, tuple(allowed_pages), tuple(sorted(excluded_pages or []))))
                return (
                    [{"page_index": 1, "page_number": 2, "score": 9.0}],
                    {"online_colpali": {"vllm_url": "http://localhost:8020"}, "retrieved_pages": [{"page_index": 1}]},
                )

            builder.retriever.retrieve_from_query = fake_retrieve_from_query
            call_count = 0

            def fake_call(client, question, state, candidates):
                nonlocal call_count
                call_count += 1
                if call_count == 1:
                    return (
                        EvaluatorDecision(search_query="ESCAPE BRYGGEN wheelchair"),
                        "<agent_decision><search_request><query>ESCAPE BRYGGEN wheelchair</query></search_request></agent_decision>",
                    )
                return (
                    EvaluatorDecision(stop=True),
                    "<agent_decision><stop>true</stop></agent_decision>",
                )

            builder.evaluator.call = fake_call
            builder.final_open_active_nodes = False

            messages = builder.build(
                "mmlongbench",
                {"doc_id": "sample.pdf", "question_id": "q1", "question": "Which attraction is not suitable for wheelchair?"},
                client=object(),
            )

        self.assertEqual(search_calls, [("mmlongbench", "ESCAPE BRYGGEN wheelchair", (0, 1), (0,))])
        self.assertIn("page:1", messages.metadata["active_node_ids"])
        self.assertEqual(messages.metadata["opened_node_ids"], [])
        self.assertTrue(any(
            item.get("action") == "SearchEvidenceRetrieval"
            and item.get("activated_pages") == [1]
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

        content = ET.fromstring(xml).find(".//evidence_state/page/node/content")
        self.assertEqual(len(content.text), 1200)

    def test_evaluator_context_includes_all_candidate_actions_and_limits_preview_chars(self):
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
                candidate_preview_char_limit=40,
            ).build_context_xml("question", state, candidates)

        root = ET.fromstring(xml)
        actions = root.findall(".//candidate_actions/ActivateNode/action")
        self.assertEqual(len(actions), 25)
        self.assertEqual(actions[0].attrib["index"], "1")
        self.assertNotIn("id", actions[0].attrib)
        self.assertTrue(all(len(action.findtext("preview")) <= 43 for action in actions))

    def test_agent_passes_all_candidates_to_evaluator_without_heuristic_capping(self):
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
            seen_candidate_ids = []

            def fake_call(client, question, state, candidates):
                seen_candidate_ids.extend(candidate.id for candidate in candidates)
                return (
                    EvaluatorDecision(stop=True),
                    "<agent_decision><stop>true</stop></agent_decision>",
                )

            builder.evaluator.call = fake_call

            builder._run_agent("mmlongbench", "What is the needle answer revenue?", state, client=object())

        self.assertGreater(len(seen_candidate_ids), 5)
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

    def test_evaluator_recent_trace_limit_and_details_are_rendered(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            state = EvidenceAgentState(EvidenceGraphStore(self._write_graph(tmp_dir), allowed_pages=[0]))
            state.trace.extend([
                {"iteration": 1, "action": "ActivateNode", "ok": True, "payload": {"node_id": "n1", "previous_state": "Inactive"}},
                {"iteration": 1, "action": "OpenNode", "ok": True, "payload": {"node_id": "n1", "previous_state": "Active"}},
                {"iteration": 1, "action": "PruneNode", "ok": True, "payload": {"node_id": "n_title", "previous_state": "Active", "reason": "Title is not needed."}},
                {"iteration": 2, "action": "SearchEvidence", "ok": True, "payload": {"query": "principal author report"}},
                {
                    "iteration": 2,
                    "action": "SearchEvidenceRetrieval",
                    "query": "principal author report",
                    "retrieved_pages": [{"page_index": 0, "page_number": 1, "score": 0.9}],
                    "activated_pages": [0],
                },
            ])

            xml = XMLEvaluator("model", recent_trace_limit=3).build_context_xml("question", state, [])

        self.assertNotIn('action="ActivateNode"', xml)
        self.assertNotIn('action="OpenNode"', xml)
        self.assertIn('action="PruneNode"', xml)
        self.assertIn('target="n_title"', xml)
        self.assertIn("Title is not needed.", xml)
        self.assertIn('action="SearchEvidence"', xml)
        self.assertIn('query="principal author report"', xml)
        self.assertIn('<activated_page_nodes>', xml)
        self.assertIn('page_node="page:0"', xml)
        self.assertNotIn("page_number", xml)

    def test_evaluator_decision_has_no_summary_request_surface(self):
        decision = parse_agent_decision_xml(
            "<agent_decision><stop>true</stop></agent_decision>"
        )

        self.assertFalse(hasattr(decision, "summarize_requests"))

    def test_reader_renderer_excludes_pruned_nodes_without_online_summaries(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            state = EvidenceAgentState(EvidenceGraphStore(self._write_graph(tmp_dir), allowed_pages=[0]))
            state.execute(ActivatePage(0, "initial_retrieval"))
            state.execute(ActivateNode("n1"))
            state.execute(OpenNode("n1"))
            state.execute(PruneNode("n1", "irrelevant"))

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
        self.assertNotIn("<online_summaries>", prompt)

    def test_reader_renderer_excludes_pruned_page_and_its_nodes_from_reader_input(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            state = EvidenceAgentState(EvidenceGraphStore(self._write_graph(tmp_dir), allowed_pages=[0, 1]))
            state.execute(ActivatePage(0, "initial_retrieval"), iteration=0)
            state.execute(ActivateNode("n1"), iteration=0)
            state.execute(OpenNode("n1"), iteration=0)
            state.execute(ActivatePage(1, "initial_retrieval"), iteration=0)
            state.execute(ActivateNode("n2"), iteration=0)
            state.execute(OpenNode("n2"), iteration=0)
            state.execute(PruneNode("page:0", "irrelevant page"), iteration=1)

            renderer = ReaderRenderer(
                OmegaConf.create({"benchmarks": {}, "baselines": {"reader": {}}}),
                include_page_images=False,
            )
            content = renderer.render("mmlongbench", {"question": "Q?"}, state)
            trace = renderer.trace_input("mmlongbench", {"question": "Q?"}, state, content)

        prompt = content[0]["text"]
        self.assertEqual(renderer._reader_page_indices(state), [1])
        self.assertNotIn('<page index="0">', prompt)
        self.assertNotIn("<abstract>Document page 1</abstract>", prompt)
        self.assertNotIn("<abstract>needle source</abstract>", prompt)
        self.assertIn('<page index="1">', prompt)
        self.assertIn("<abstract>needle target</abstract>", prompt)
        self.assertTrue(all(ref.get("page_index") != 0 for ref in trace["image_refs"]))

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
                OmegaConf.create({"benchmarks": {}, "baselines": {"reader": {}}}),
                include_page_images=False,
            ).render(
                "longdocurl",
                {"question": "Which section best matches the description?"},
                state,
            )

        prompt = content[0]["text"]
        self.assertIn("<role>", prompt)
        self.assertIn("<objective>", prompt)
        self.assertIn("<question>Which section best matches the description?</question>", prompt)
        self.assertIn("<answer_policy>", prompt)
        self.assertIn("If the answer cannot be found, answer exactly: Not answerable.", prompt)
        self.assertNotIn("Candidate visible labels from retrieved evidence:", prompt)
        self.assertIn("Important Section Title", prompt)
        self.assertNotIn("Active evidence graph:", prompt)
        self.assertNotIn("provenance_id=n_title", prompt)
        self.assertIn("needle source", prompt)

    def test_reader_renderer_prompt_uses_parseable_xml_blocks_for_sensitive_text(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            state = EvidenceAgentState(EvidenceGraphStore(self._write_graph(tmp_dir), allowed_pages=[0]))
            state.graph.nodes["n1"]["text"] = "Visible answer is A < B & C."
            state.graph.nodes["n1"]["abstract"] = "Visible answer is A < B & C."
            state.execute(ActivatePage(0, "initial_retrieval"))
            state.execute(ActivateNode("n1"))
            state.execute(OpenNode("n1"))

            content = ReaderRenderer(
                OmegaConf.create({"benchmarks": {}, "baselines": {"reader": {}}}),
                include_page_images=False,
            ).render(
                "longdocurl",
                {"question": "Which answer contains < and &?"},
                state,
            )

        prompt = content[0]["text"]
        root = ET.fromstring(prompt)
        self.assertEqual(root.find("question").text, "Which answer contains < and &?")
        self.assertIn("Visible answer is A < B & C.", root.find("./evidence/page/node/abstract").text)
        self.assertIsNone(root.find("./evidence/page/node/content"))

    def test_normalized_bbox_1000_maps_to_image_pixels_with_padding(self):
        self.assertEqual(
            normalized_bbox_1000_to_pixel_box([250, 250, 750, 750], (200, 100)),
            (42, 17, 158, 83),
        )

    def test_reader_renderer_compact_mode_limits_candidate_labels_to_locating_questions(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            state = EvidenceAgentState(EvidenceGraphStore(self._write_graph(tmp_dir), allowed_pages=[0]))
            state.execute(ActivatePage(0, "initial_retrieval"))
            state.execute(ActivateNode("n_title"))

            content = ReaderRenderer(
                OmegaConf.create({"benchmarks": {}, "baselines": {"reader": {}}}),
                include_page_images=False,
            ).render(
                "longdocurl",
                {"question": "What is the total amount of liabilities?"},
                state,
            )

        prompt = content[0]["text"]
        self.assertNotIn("Candidate visible labels", prompt)
        self.assertIn("Important Section Title", prompt)

    def test_reader_renderer_compact_uses_opened_text_without_table_name_candidates(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            state = EvidenceAgentState(EvidenceGraphStore(self._write_graph(tmp_dir), allowed_pages=[0]))
            state.graph.nodes["n1"]["type"] = "table"
            state.graph.nodes["n1"]["caption"] = "Table 15: Leading destination of exports (UGX Billion): July-June"
            state.execute(ActivatePage(0, "initial_retrieval"))
            state.execute(ActivateNode("n1"))
            state.execute(OpenNode("n1"))

            content = ReaderRenderer(
                OmegaConf.create({"benchmarks": {}, "baselines": {"reader": {}}}),
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
                    "baselines": {"reader": {}},
                }),
                include_page_images=False,
            ).render("mmlongbench", {"question": "Which area is not shown?"}, state)

        prompt = content[0]["text"]
        self.assertIn("<role>", prompt)
        self.assertIn("<objective>", prompt)
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
                    "baselines": {"reader": {}},
                }),
                include_page_images=False,
            ).render("mmlongbench", {"question": "Which area is shown?"}, state)

        prompt = content[0]["text"]
        self.assertIn("<retrieved_page_indices>0, 1</retrieved_page_indices>", prompt)
        self.assertIn("If the answer cannot be found, answer exactly: Not answerable.", prompt)
        self.assertIn("&lt;think&gt;...&lt;/think&gt;", prompt)
        self.assertIn("&lt;answer&gt;[final_answer]&lt;/answer&gt;", prompt)
        self.assertIn("Output sequence:", prompt)

    def test_reader_renderer_compact_no_longer_supports_plain_prompt_mode(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            state = EvidenceAgentState(EvidenceGraphStore(self._write_graph(tmp_dir), allowed_pages=[0]))
            state.execute(ActivatePage(0, "initial_retrieval"))

            content = ReaderRenderer(
                OmegaConf.create({
                    "benchmarks": {},
                    "baselines": {"reader": {}},
                }),
                include_page_images=False,
            ).render("mmlongbench", {"question": "Which area is shown?"}, state)

        prompt = content[0]["text"]
        self.assertIn("If the answer cannot be found", prompt)
        self.assertIn("<retrieved_page_indices>0</retrieved_page_indices>", prompt)
        self.assertNotIn("For color questions", prompt)

    def test_reader_renderer_compact_omits_color_question_hint(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            state = EvidenceAgentState(EvidenceGraphStore(self._write_graph(tmp_dir), allowed_pages=[0]))
            state.execute(ActivatePage(0, "initial_retrieval"))

            content = ReaderRenderer(
                OmegaConf.create({
                    "benchmarks": {},
                    "baselines": {"reader": {}},
                }),
                include_page_images=False,
            ).render("mmlongbench", {"question": "What color is the highlighted area?"}, state)

        prompt = content[0]["text"]
        self.assertNotIn("For color questions, use common color names rather than hex codes.", prompt)
        self.assertIn("If the answer cannot be found", prompt)
        self.assertIn("<retrieved_page_indices>0</retrieved_page_indices>", prompt)

    def test_reader_renderer_compact_omits_page_scope_hint(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            state = EvidenceAgentState(EvidenceGraphStore(self._write_graph(tmp_dir), allowed_pages=[0]))
            state.execute(ActivatePage(0, "initial_retrieval"))

            content = ReaderRenderer(
                OmegaConf.create({
                    "benchmarks": {},
                    "baselines": {"reader": {}},
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

    def test_reader_renderer_selects_all_opened_node_crops_without_benchmark_limit(self):
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
                    "baselines": {"reader": {"include_opened_node_crops": True}},
                }),
                include_page_images=False,
            )

            node_ids = renderer._candidate_node_image_ids(state)

        self.assertEqual(node_ids, ["n1", "n2"])

    def test_reader_renderer_builds_opened_node_crops_for_longdocurl(self):
        from PIL import Image

        with tempfile.TemporaryDirectory() as tmp_dir:
            image_dir = Path(tmp_dir) / "images" / "1234"
            image_dir.mkdir(parents=True)
            image_path = image_dir / "123456_0.png"
            Image.new("RGB", (200, 200), color="white").save(image_path)
            state = EvidenceAgentState(EvidenceGraphStore(self._write_graph(tmp_dir), allowed_pages=[0]))
            state.graph.nodes["n1"]["bbox"] = [20, 20, 120, 120]
            state.execute(ActivatePage(0, "initial_retrieval"), iteration=0)
            state.execute(ActivateNode("n1"), iteration=0)
            state.execute(OpenNode("n1"), iteration=0)
            cfg = OmegaConf.create({
                "benchmarks": {"image_prefix": str(Path(tmp_dir) / "images")},
                "baselines": {"reader": {"include_opened_node_crops": True}},
            })
            renderer = ReaderRenderer(cfg, include_page_images=False, include_opened_node_images=True)

            content = renderer.render("longdocurl", {"doc_no": "123456", "question": "What is shown?"}, state)
            trace = renderer.trace_input("longdocurl", {"doc_no": "123456", "question": "What is shown?"}, state, content)

        self.assertTrue(any(part.get("type") == "image_url" for part in content if isinstance(part, dict)))
        self.assertEqual(trace["image_refs"][0]["kind"], "opened_node_crop")
        self.assertEqual(trace["image_refs"][0]["node_id"], "n1")
        self.assertEqual(trace["image_refs"][0]["image_path"], str(image_path))

    def test_reader_renderer_preserves_activation_page_order(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            state = EvidenceAgentState(EvidenceGraphStore(self._write_graph(tmp_dir), allowed_pages=[0, 1]))
            state.execute(ActivatePage(1, "initial_retrieval"), iteration=0)
            state.execute(ActivatePage(0, "initial_retrieval"), iteration=0)

            page_indices = ReaderRenderer(OmegaConf.create({"benchmarks": {}}), include_page_images=False)._reader_page_indices(state)

        self.assertEqual(page_indices, [1, 0])

    def test_reader_renderer_includes_all_active_page_text_without_benchmark_limit(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            state = EvidenceAgentState(EvidenceGraphStore(self._write_graph(tmp_dir), allowed_pages=[0, 1]))
            state.graph.nodes["page:1"]["abstract"] = "Document page 2"
            state.graph.nodes["page:1"]["text"] = "Page two chart says With family and friends 20%."
            state.graph.nodes["page:0"]["abstract"] = "Document page 1"
            state.graph.nodes["page:0"]["text"] = "Page one unrelated text."
            state.execute(ActivatePage(1, "initial_retrieval"), iteration=0)
            state.execute(ActivatePage(0, "initial_retrieval"), iteration=0)

            content = ReaderRenderer(
                OmegaConf.create({
                    "benchmarks": {},
                    "baselines": {"reader": {}},
                }),
                include_page_images=False,
            ).render("longdocurl", {"question": "How much time was spent with family?"}, state)

        prompt = content[0]["text"]
        self.assertIn("<evidence>", prompt)
        self.assertIn('<page index="1">', prompt)
        self.assertIn('<page index="0">', prompt)
        self.assertIn("<abstract>Document page 2</abstract>", prompt)
        self.assertNotIn("<content>Page one unrelated text.</content>", prompt)
        self.assertNotIn("With family and friends 20%", prompt)

    def test_reader_renderer_page_node_does_not_duplicate_identical_abstract_and_content(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            state = EvidenceAgentState(EvidenceGraphStore(self._write_graph(tmp_dir), allowed_pages=[0]))
            state.graph.nodes["page:0"]["abstract"] = "Same page summary"
            state.graph.nodes["page:0"]["text"] = "Same page summary"
            state.execute(ActivatePage(0, "initial_retrieval"), iteration=0)

            content = ReaderRenderer(
                OmegaConf.create({"benchmarks": {}, "baselines": {"reader": {}}}),
                include_page_images=False,
            ).render("mmlongbench", {"question": "What is shown?"}, state)

        prompt = content[0]["text"]
        self.assertIn("<abstract>Same page summary</abstract>", prompt)
        self.assertNotIn("<content>Same page summary</content>", prompt)
        self.assertNotIn("number=", prompt)

    def test_reader_renderer_falls_back_to_bbox_crop_when_node_image_path_is_directory(self):
        import base64
        import re
        from io import BytesIO

        from PIL import Image

        with tempfile.TemporaryDirectory() as tmp_dir:
            page_dir = Path(tmp_dir) / "pngs" / "sample"
            page_dir.mkdir(parents=True)
            page_path = page_dir / "page_0001_dpi144.png"
            Image.new("RGB", (200, 200), color="white").save(page_path)
            bad_image_dir = Path(tmp_dir) / "images"
            bad_image_dir.mkdir()
            state = EvidenceAgentState(EvidenceGraphStore(self._write_graph(tmp_dir), allowed_pages=[0]))
            state.graph.nodes["n1"]["image_path"] = str(bad_image_dir)
            state.graph.nodes["n1"]["bbox"] = [250, 250, 750, 750]
            state.execute(ActivatePage(0, "initial_retrieval"), iteration=0)
            state.execute(ActivateNode("n1"), iteration=0)
            state.execute(OpenNode("n1"), iteration=0)
            renderer = ReaderRenderer(
                OmegaConf.create({
                    "benchmarks": {"pdf_png_dir": str(Path(tmp_dir) / "pngs"), "resolution": 144},
                    "baselines": {"reader": {"include_opened_node_crops": True}},
                }),
                include_page_images=False,
                include_opened_node_images=True,
            )

            content = renderer.render("mmlongbench", {"doc_id": "sample.pdf", "question": "What is shown?"}, state)
            trace = renderer.trace_input("mmlongbench", {"doc_id": "sample.pdf", "question": "What is shown?"}, state, content)

        refs = [ref for ref in trace["image_refs"] if ref.get("node_id") == "n1"]
        self.assertEqual(refs[0]["kind"], "opened_node_crop")
        self.assertNotIn("page_number", refs[0])
        image_part = next(part for part in content if isinstance(part, dict) and part.get("type") == "image_url")
        encoded = re.search(r"base64,(.*)$", image_part["image_url"]["url"]).group(1)
        with Image.open(BytesIO(base64.b64decode(encoded))) as crop:
            self.assertEqual(crop.size, (116, 116))

    def test_reader_renderer_skips_opened_node_image_path_when_it_is_directory(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            state = EvidenceAgentState(EvidenceGraphStore(self._write_graph(tmp_dir), allowed_pages=[0]))
            bad_image_dir = os.path.join(tmp_dir, "images")
            os.makedirs(bad_image_dir)
            state.graph.nodes["n1"]["image_path"] = bad_image_dir
            state.execute(ActivatePage(0, "initial_retrieval"), iteration=0)
            state.execute(ActivateNode("n1"), iteration=0)
            state.execute(OpenNode("n1"), iteration=0)
            renderer = ReaderRenderer(
                OmegaConf.create({"benchmarks": {}, "baselines": {"reader": {}}}),
                include_page_images=False,
                include_opened_node_images=True,
            )

            content = renderer.render("mmlongbench", {"doc_id": "sample.pdf", "question": "What is shown?"}, state)
            trace = renderer.trace_input("mmlongbench", {"doc_id": "sample.pdf", "question": "What is shown?"}, state, content)

        self.assertEqual([ref for ref in trace["image_refs"] if ref.get("node_id") == "n1"], [])
        self.assertNotIn('kind="opened_node_image"', content[0]["text"])
        self.assertIn("<abstract>needle source</abstract>", content[0]["text"])
        self.assertNotIn("needle source evidence", content[0]["text"])

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
        self.assertIn("retrieval", messages.metadata)
        self.assertIn("context_summary", messages.metadata)
        self.assertIn("logical_cost", messages.metadata)
        self.assertEqual(messages.metadata["retrieval"]["initial_retrieved_pages"], [0])
        self.assertEqual(messages.metadata["context_summary"]["num_context_pages"], 1)
        self.assertEqual(messages.metadata["logical_cost"]["num_retriever_calls"], 1)
        self.assertIn("magerag", messages.metadata)
        self.assertEqual(messages.metadata["magerag"]["graph_stats"]["num_nodes"], 6)
        self.assertEqual(messages.metadata["magerag"]["trace_summary"]["stop_reason"], "fallback_no_client")
        self.assertEqual(messages.metadata["magerag"]["iteration_trace"], messages.metadata["iteration_trace"])

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
                EvaluatorDecision(selected_actions=[{"candidate_index": 1}]),
                "<think>Select the first useful node.</think><agent_decision><selected_actions><action index=\"1\"/></selected_actions></agent_decision>",
            )

            builder._run_agent("mmlongbench", "question", state, client=object())

        decision_trace = next(item for item in state.trace if item.get("action") == "EvaluatorDecision")
        self.assertIn("prompt_text", decision_trace["evaluator_input"])
        self.assertIn("<agent_step_context>", decision_trace["evaluator_input"]["prompt_text"])
        self.assertIn("state_snapshot_before", decision_trace)
        executed = next(item for item in state.trace if item.get("action") == "ActivateNode")
        self.assertEqual(executed["selection"]["candidate_index"], 1)
        self.assertNotIn("reason", executed["selection"])
        self.assertIn("state_snapshot_after", executed)
        self.assertIn("n1", executed["state_snapshot_after"]["active_node_ids"])

    def test_run_agent_leaves_parent_page_activation_to_activate_node(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            state = EvidenceAgentState(EvidenceGraphStore(self._write_graph(tmp_dir), allowed_pages=[0, 1]))
            state.execute(ActivatePage(0, "initial_retrieval"))
            state.execute(ActivateNode("n1"))
            original_execute = state.execute

            def execute_without_node_activation_retry(action, iteration=None):
                if isinstance(action, ActivateNode) and action.node_id == "n2":
                    result = ActionResult(False, "ActivateNode", "activate page first")
                    state._record_result(result, action, iteration)
                    return result
                return original_execute(action, iteration)

            state.execute = execute_without_node_activation_retry
            builder = MAGERAGContextBuilder(OmegaConf.create({
                "baselines": {
                    "name": "magerag",
                    "controller": {"watchdog_iterations": 1},
                }
            }))

            def fake_call(client, question, state, candidates):
                target_index = next(
                    index
                    for index, candidate in enumerate(
                        [candidate for candidate in candidates if candidate.action_type == "ActivateNode"],
                        start=1,
                    )
                    if candidate.payload.get("node_id") == "n2"
                )
                return (
                    EvaluatorDecision(selected_actions=[{"candidate_index": target_index}]),
                    "<agent_decision><selected_actions>"
                    f"<action index=\"{target_index}\"/>"
                    "</selected_actions></agent_decision>",
                )

            builder.evaluator.call = fake_call

            builder._run_agent("mmlongbench", "question", state, client=object())

        self.assertNotIn("page:1", state.active_node_ids())
        self.assertFalse(any(
            item.get("action") == "ActivatePage"
            and item.get("payload", {}).get("page_index") == 1
            and item.get("payload", {}).get("source") == "relation_target"
            for item in state.trace
        ))

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
                EvaluatorDecision(stop=True),
                "<agent_decision><stop>true</stop></agent_decision>",
            )

            messages = builder.build("mmlongbench", {"doc_id": "sample.pdf", "question_id": "q1", "question": "Q?"}, client=object())

        self.assertEqual(messages.metadata["stop_reason"], "normal_stop")
        self.assertIn("EvaluatorDecision", [item.get("action") for item in messages.metadata["iteration_trace"]])
        self.assertNotIn("run_online_agent", messages.metadata)

    def test_watchdog_iterations_zero_skips_online_evaluator(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            state = EvidenceAgentState(EvidenceGraphStore(self._write_graph(tmp_dir), allowed_pages=[0]))
            state.execute(ActivatePage(0, "initial_retrieval"))
            builder = MAGERAGContextBuilder(OmegaConf.create({
                "baselines": {
                    "name": "magerag",
                    "controller": {"watchdog_iterations": 0},
                }
            }))
            builder.evaluator.call = lambda *args, **kwargs: (_ for _ in ()).throw(AssertionError("evaluator should not be called"))

            stop_reason = builder._run_agent("mmlongbench", "question", state, client=object())

        self.assertEqual(stop_reason, "controller_disabled")
        self.assertFalse(any(item.get("action") == "EvaluatorDecision" for item in state.trace))

    def test_topk_page_only_mode_skips_node_opening_and_evaluator(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            graph_root = os.path.join(tmp_dir, "graphs")
            self._write_graph(os.path.join(graph_root, "sample"))
            cfg = OmegaConf.create({
                "baselines": {
                    "name": "magerag",
                    "params": {},
                    "controller": {"mode": "topk_page_only"},
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
            builder.evaluator.call = lambda *args, **kwargs: (_ for _ in ()).throw(AssertionError("evaluator should not be called"))

            messages = builder.build("mmlongbench", {"doc_id": "sample.pdf", "question_id": "q1", "question": "needle"}, client=object())

        self.assertEqual(messages.metadata["stop_reason"], "mode_topk_page_only")
        self.assertEqual(messages.metadata["opened_node_ids"], [])
        self.assertFalse(any(item.get("action") == "EvaluatorDecision" for item in messages.metadata["iteration_trace"]))

    def test_topk_page_with_node_rendering_mode_opens_initial_page_nodes_without_evaluator(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            graph_root = os.path.join(tmp_dir, "graphs")
            self._write_graph(os.path.join(graph_root, "sample"))
            cfg = OmegaConf.create({
                "baselines": {
                    "name": "magerag",
                    "params": {},
                    "controller": {"mode": "topk_page_with_node_rendering"},
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
            builder.evaluator.call = lambda *args, **kwargs: (_ for _ in ()).throw(AssertionError("evaluator should not be called"))

            messages = builder.build("mmlongbench", {"doc_id": "sample.pdf", "question_id": "q1", "question": "needle source"}, client=object())

        self.assertEqual(messages.metadata["stop_reason"], "mode_topk_page_with_node_rendering")
        self.assertIn("n1", messages.metadata["opened_node_ids"])
        self.assertFalse(any(item.get("action") == "EvaluatorDecision" for item in messages.metadata["iteration_trace"]))

    def test_graph_neighbor_expansion_mode_expands_one_hop_without_evaluator(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            graph_root = os.path.join(tmp_dir, "graphs")
            self._write_graph(os.path.join(graph_root, "sample"))
            cfg = OmegaConf.create({
                "baselines": {
                    "name": "magerag",
                    "params": {},
                    "controller": {"mode": "graph_neighbor_expansion"},
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
            builder.evaluator.call = lambda *args, **kwargs: (_ for _ in ()).throw(AssertionError("evaluator should not be called"))

            messages = builder.build("mmlongbench", {"doc_id": "sample.pdf", "question_id": "q1", "question": "needle target"}, client=object())

        self.assertEqual(messages.metadata["stop_reason"], "mode_graph_neighbor_expansion")
        self.assertIn("n1", messages.metadata["opened_node_ids"])
        self.assertIn("n2", messages.metadata["opened_node_ids"])
        self.assertFalse(any(item.get("action") == "EvaluatorDecision" for item in messages.metadata["iteration_trace"]))

    def test_initial_page_nodes_are_opened_and_rendered_without_expanding_top_k(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            graph_root = os.path.join(tmp_dir, "graphs")
            self._write_graph(os.path.join(graph_root, "sample"))
            cfg = OmegaConf.create({
                "baselines": {
                    "name": "magerag",
                    "params": {},
                    "reader": {
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
                    selected_indexes = [
                        index
                        for index, candidate in enumerate(candidates, start=1)
                        if candidate.id in {"act:ActivateNode:n_title", "act:ActivateNode:n1"}
                    ]
                    return (
                        EvaluatorDecision(selected_actions=[
                            {"candidate_index": index}
                            for index in selected_indexes
                        ]),
                        "<agent_decision><selected_actions>"
                        + "".join(f"<action index=\"{index}\"/>" for index in selected_indexes)
                        + "</selected_actions></agent_decision>",
                    )
                return (
                    EvaluatorDecision(stop=True),
                    "<agent_decision><stop>true</stop></agent_decision>",
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
        self.assertIn("<evidence>", prompt)
        self.assertIn("<abstract>needle source</abstract>", prompt)
        self.assertNotIn("needle source evidence", prompt)

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
                    selected_indexes = [
                        index
                        for index, candidate in enumerate(candidates, start=1)
                        if candidate.id in {"act:ActivateNode:n_title", "act:ActivateNode:n1"}
                    ]
                    return (
                        EvaluatorDecision(selected_actions=[
                            {"candidate_index": index}
                            for index in selected_indexes
                        ]),
                        "<agent_decision><selected_actions>"
                        + "".join(f"<action index=\"{index}\"/>" for index in selected_indexes)
                        + "</selected_actions></agent_decision>",
                    )
                return (
                    EvaluatorDecision(stop=True),
                    "<agent_decision><stop>true</stop></agent_decision>",
                )

            builder.evaluator.call = fake_call

            messages = builder.build(
                "mmlongbench",
                {"doc_id": "sample.pdf", "question_id": "q1", "question": "What does the source evidence say?"},
                client=object(),
            )

        self.assertIn("n_title", messages.metadata["opened_node_ids"])
        self.assertIn("n1", messages.metadata["opened_node_ids"])

    def test_renderer_uses_same_opened_node_text_policy_for_all_benchmarks(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            state = EvidenceAgentState(EvidenceGraphStore(self._write_graph(tmp_dir), allowed_pages=[0]))
            state.execute(ActivatePage(0, "initial_retrieval"))
            state.execute(ActivateNode("n1"))
            state.execute(OpenNode("n1"))
            cfg = OmegaConf.create({
                "benchmarks": {},
                "baselines": {"reader": {}},
            })

            mmlong = ReaderRenderer(cfg, include_page_images=False).render(
                "mmlongbench", {"question": "What does the source evidence say?"}, state
            )
            longdoc = ReaderRenderer(cfg, include_page_images=False).render(
                "longdocurl", {"question": "What does the source evidence say?"}, state
            )

        self.assertIn("<evidence>", mmlong[0]["text"])
        self.assertIn("<evidence>", longdoc[0]["text"])
        self.assertIn("<abstract>needle source</abstract>", mmlong[0]["text"])
        self.assertIn("<abstract>needle source</abstract>", longdoc[0]["text"])
        self.assertNotIn("needle source evidence", mmlong[0]["text"])
        self.assertNotIn("needle source evidence", longdoc[0]["text"])

    def test_reader_renderer_prioritizes_question_scope_pages_for_mmlongbench(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            state = EvidenceAgentState(EvidenceGraphStore(self._write_graph(tmp_dir), allowed_pages=[0, 1]))
            state.execute(ActivatePage(0, "initial_retrieval"))
            state.execute(ActivatePage(1, "question_page_scope"))
            cfg = OmegaConf.create({
                "benchmarks": {},
                "baselines": {"reader": {}},
            })

            content = ReaderRenderer(cfg, include_page_images=False).render(
                "mmlongbench",
                {"question": "What is on page 2?"},
                state,
            )

        self.assertIn("<retrieved_page_indices>1, 0</retrieved_page_indices>", content[0]["text"])

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
            {"id": "page:0", "type": "page", "doc_id": "sample", "page_index": 0, "abstract": "Document page 1", "text": "Page one evidence"},
            {"id": "page:1", "type": "page", "doc_id": "sample", "page_index": 1, "abstract": "Document page 2", "text": "Page two evidence"},
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
