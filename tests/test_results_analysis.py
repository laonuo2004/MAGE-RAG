import json
from pathlib import Path

from analysis.plugins import AnalysisPlugin, ParameterSpec, get_plugin, registered_plugins
from analysis.plugins.aeg_rag import AEGRAGPlugin
from analysis.results_loader import parse_run_parameters, read_jsonl_cached, scan_runs
from analysis.results_metrics import (
    build_leaderboard,
    flatten_metrics,
    pairwise_comparison,
    parameter_curve_rows,
    retrieval_diagnostics,
)


def _write_json(path: Path, data: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, ensure_ascii=False), encoding="utf-8")


def _write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "\n".join(json.dumps(row, ensure_ascii=False) for row in rows) + "\n",
        encoding="utf-8",
    )


def test_parse_run_parameters_from_known_filename_tokens():
    params = parse_run_parameters(
        Path(
            "res_top_k_5_chunk_size_200_chunk_overlap_20_"
            "bm25_k1_1_5_bm25_b_0_75_max_iterations_3_model.jsonl"
        )
    )

    assert params["top_k"] == 5
    assert params["chunk_size"] == 200
    assert params["chunk_overlap"] == 20
    assert params["bm25_k1"] == 1.5
    assert params["bm25_b"] == 0.75
    assert params["max_iterations"] == 3


def test_plugin_registry_discovers_builtin_and_defaults_unknown_baselines():
    plugin_names = {plugin.name for plugin in registered_plugins()}

    assert "bm25" in plugin_names
    assert "m3docrag_iterate" in plugin_names
    assert "aeg_rag" in plugin_names
    assert get_plugin("bm25").name == "bm25"
    assert get_plugin("aeg-rag").name == "aeg_rag"
    assert get_plugin("aeg_rag").name == "aeg_rag"
    assert get_plugin("toy_reranker").name == "default"


def test_plugin_declared_filename_parameters_include_new_baseline_fields():
    class ToyPlugin(AnalysisPlugin):
        name = "toy"
        baseline_names = ("toy_reranker",)

        def parameter_specs(self):
            return (
                ParameterSpec("rerank_top_k", value_type="int", numeric=True),
                ParameterSpec("fusion_weight", value_type="float", numeric=True),
            )

    params = parse_run_parameters(
        "res_rerank_top_k_20_fusion_weight_0_25_model.jsonl",
        ToyPlugin().parameter_specs(),
    )

    assert params == {"rerank_top_k": 20, "fusion_weight": 0.25}


def test_flatten_metrics_preserves_official_nested_breakdowns():
    rows = flatten_metrics(
        {
            "overall_acc": 0.42,
            "fine_grained_metrics": {
                "Main_Task": {"Understanding": 0.5},
                "Fine_Grained": {
                    "Reasoning": {"Multi_Page": {"Table": 0.25}},
                },
            },
            "breakdowns": {"single_page": {"acc": 0.7, "count": 2}},
        }
    )

    by_path = {row["path"]: row for row in rows}
    assert by_path["overall_acc"]["value"] == 0.42
    assert by_path["fine_grained_metrics.Main_Task.Understanding"]["value"] == 0.5
    assert by_path["fine_grained_metrics.Fine_Grained.Reasoning.Multi_Page.Table"]["value"] == 0.25
    assert by_path["breakdowns.single_page.acc"]["count"] == 2


def test_scan_runs_pairs_metrics_and_jsonl_without_reading_records(tmp_path):
    results_root = tmp_path / "results"
    run = results_root / "longdocurl" / "bm25" / "res_top_k_2_chunk_size_100_model"
    _write_json(
        run.with_suffix(".metrics.json"),
        {"overall_acc": 0.3, "sample_count": 9, "completed_count": 8, "failed_count": 1},
    )
    _write_jsonl(run.with_suffix(".jsonl"), [{"question_id": "q1", "score": 1}])

    runs = scan_runs(results_root)

    assert len(runs) == 1
    assert runs[0].benchmark == "longdocurl"
    assert runs[0].baseline == "bm25"
    assert runs[0].jsonl_path == run.with_suffix(".jsonl")
    assert runs[0].metrics_path == run.with_suffix(".metrics.json")
    assert runs[0].sample_count == 9
    assert runs[0].parameters["top_k"] == 2
    assert runs[0].jsonl_size_bytes > 0


def test_scan_runs_prefers_run_metadata_then_legacy_then_sample_then_filename(tmp_path):
    results_root = tmp_path / "results"
    run = results_root / "mmlongbench" / "toy_reranker" / "res_rerank_top_k_5_fusion_weight_0_5_model"
    _write_json(
        run.with_suffix(".metrics.json"),
        {
            "overall_acc": 0.3,
            "parameters": {"fusion_weight": 0.7, "temperature": 0.2},
            "run_metadata": {
                "baselines": {
                    "params": {
                        "rerank_top_k": 20,
                        "fusion_weight": 0.9,
                    }
                }
            },
        },
    )
    _write_jsonl(
        run.with_suffix(".jsonl"),
        [{"question_id": "q1", "score": 1, "prepare_metadata": {"encoder_name": "toy-encoder"}}],
    )

    runs = scan_runs(results_root)

    assert runs[0].parameters["rerank_top_k"] == 20
    assert runs[0].parameters["fusion_weight"] == 0.9
    assert runs[0].parameters["temperature"] == 0.2
    assert runs[0].parameters["encoder_name"] == "toy-encoder"
    assert runs[0].parameter_sources["rerank_top_k"] == "run_metadata.baselines.params"
    assert runs[0].parameter_sources["temperature"] == "metrics.parameters"
    assert runs[0].parameter_sources["encoder_name"] == "sample.prepare_metadata"


def test_read_jsonl_cached_invalidates_on_file_stat_change(tmp_path):
    path = tmp_path / "run.jsonl"
    cache_root = tmp_path / "cache"
    _write_jsonl(path, [{"question_id": "q1", "score": 0.1}])

    first = read_jsonl_cached(path, cache_root)
    _write_jsonl(path, [{"question_id": "q1", "score": 0.9}])
    second = read_jsonl_cached(path, cache_root)

    assert first[0]["score"] == 0.1
    assert second[0]["score"] == 0.9


def test_read_jsonl_cache_signature_includes_plugin_namespace(tmp_path):
    path = tmp_path / "run.jsonl"
    cache_root = tmp_path / "cache"
    _write_jsonl(path, [{"question_id": "q1", "score": 0.1}])

    read_jsonl_cached(path, cache_root, cache_namespace="bm25")
    read_jsonl_cached(path, cache_root, cache_namespace="default")

    assert len(list(cache_root.glob("*.json"))) == 2


def test_retrieval_diagnostics_handles_page_index_and_page_number_hits():
    rows = retrieval_diagnostics(
        [
            {
                "question_id": "q1",
                "score": 0.8,
                "evidence_pages": [3],
                "prepare_metadata": {
                    "duration_seconds": 1.0,
                    "retrieved_chunks": [
                        {"rank": 1, "page_index": 0, "page_number": 1, "score": 0.1},
                        {"rank": 2, "page_index": 2, "page_number": 3, "score": 0.9},
                    ],
                },
                "generation_metadata": {"duration_seconds": 2.0},
                "extraction_metadata": {"duration_seconds": 3.0},
            },
            {
                "question_id": "q2",
                "score": 0.0,
                "evidence_pages": [9],
                "prepare_metadata": {
                    "retrieved_pages": [{"page_index": 0, "page_number": 1, "score": 0.4}],
                },
            },
        ]
    )

    assert rows[0]["evidence_hit"] is True
    assert rows[0]["first_hit_rank"] == 2
    assert rows[0]["first_hit_score"] == 0.9
    assert rows[0]["total_duration_seconds"] == 6.0
    assert rows[1]["evidence_hit"] is False
    assert rows[1]["first_hit_rank"] is None


def test_leaderboard_and_pairwise_comparison():
    class Run:
        def __init__(self, benchmark, baseline, run_id, metrics):
            self.benchmark = benchmark
            self.baseline = baseline
            self.run_id = run_id
            self.metrics = metrics

    leaderboard = build_leaderboard(
        [
            Run("mmlongbench", "bm25", "low", {"overall_acc": 0.2}),
            Run("mmlongbench", "bm25", "high", {"overall_acc": 0.5}),
            Run("mmlongbench", "image", "image", {"overall_acc": 0.4}),
        ],
        "acc",
    )

    assert [row["run_id"] for row in leaderboard] == ["high", "image"]

    comparison = pairwise_comparison(
        [{"question_id": "q1", "score": 1.0}, {"question_id": "q2", "score": 0.2}],
        [{"question_id": "q1", "score": 0.0}, {"question_id": "q2", "score": 0.2}],
    )

    assert comparison["summary"] == {"a_wins": 1, "b_wins": 0, "ties": 1, "aligned": 2}
    assert comparison["rows"][0]["delta"] == 1.0


def test_parameter_curve_rows_are_chart_spec_driven():
    class Run:
        def __init__(self, baseline, parameters):
            self.benchmark = "mmlongbench"
            self.baseline = baseline
            self.run_id = baseline
            self.metrics = {"overall_acc": 0.4}
            self.parameters = parameters
            self.plugin_name = get_plugin(baseline).name

    rows = parameter_curve_rows(
        [
            Run("bm25", {"top_k": 1, "chunk_size": 100, "chunk_overlap": 20}),
            Run("m3docrag", {"top_k": 5}),
            Run("m3docrag-iterate", {"max_iterations": 3}),
        ],
        "acc",
    )

    by_baseline = {row["baseline"]: row for row in rows}
    assert by_baseline["bm25"]["chart_specs"][0]["kind"] == "heatmap"
    assert by_baseline["m3docrag"]["chart_specs"][0]["x"] == "top_k"
    assert by_baseline["m3docrag-iterate"]["chart_specs"][0]["x"] == "max_iterations"


def test_root_dashboard_entrypoint_uses_absolute_imports():
    source = (Path(__file__).resolve().parents[1] / "results_dashboard.py").read_text(encoding="utf-8")

    assert "from analysis.dashboard_app import main" in source
    assert "sys.path" not in source


def test_aeg_rag_plugin_builds_case_visualization_from_trace_and_graph(tmp_path):
    graph_dir = tmp_path / "graph"
    _write_json(graph_dir / "graph.json", {"doc_key": "doc"})
    _write_jsonl(
        graph_dir / "nodes.jsonl",
        [
            {
                "id": "doc:page:0",
                "type": "page",
                "page_index": 0,
                "abstract": "Cover page",
                "image_path": str(tmp_path / "page0.png"),
            },
            {
                "id": "doc:page:0:block:0:title",
                "type": "title",
                "page_index": 0,
                "abstract": "Important title",
            },
            {
                "id": "doc:page:1",
                "type": "page",
                "page_index": 1,
                "abstract": "Evidence page",
            },
        ],
    )
    _write_jsonl(graph_dir / "edges.jsonl", [])
    record = {
        "question_id": "q1",
        "question": "Which page has the important title?",
        "answer": "page 1",
        "pred": "page 1",
        "score": 1.0,
        "evidence_pages": [1],
        "prepare_metadata": {
            "graph_dir": str(graph_dir),
            "duration_seconds": 2.5,
            "stop_reason": "retrieval_only",
            "allowed_pages": [0, 1],
            "initial_retrieval": {
                "initial_page": {"page_index": 0, "page_number": 1, "score": 0.8},
                "retrieved_pages": [
                    {"rank": 1, "page_index": 0, "page_number": 1, "score": 0.8},
                    {"rank": 2, "page_index": 1, "page_number": 2, "score": 0.7},
                ],
            },
            "final_node_states": {
                "doc:page:0": "Active",
                "doc:page:0:block:0:title": "Opened",
            },
            "active_node_ids": ["doc:page:0"],
            "opened_node_ids": ["doc:page:0:block:0:title"],
            "pruned_node_ids": [],
            "iteration_trace": [
                {
                    "iteration": 0,
                    "action": "ActivatePage",
                    "ok": True,
                    "payload": {"page_index": 0, "node_id": "doc:page:0"},
                },
                {
                    "iteration": 0,
                    "action": "OpenNode",
                    "ok": True,
                    "payload": {"node_id": "doc:page:0:block:0:title", "previous_state": "Active"},
                },
            ],
            "validation_errors": [],
        },
    }

    data = AEGRAGPlugin().case_visualization(record)

    assert data["summary"]["trace_steps"] == 2
    assert data["summary"]["opened_nodes"] == 1
    assert data["summary"]["validation_errors"] == 0
    assert data["action_counts"][0]["action"] == "ActivatePage"
    page0 = next(row for row in data["page_rows"] if row["page_index"] == 0)
    page1 = next(row for row in data["page_rows"] if row["page_index"] == 1)
    assert page0["opened_nodes"] == 1
    assert page0["action_count"] == 2
    assert page0["retrieval_rank"] == 1
    assert page1["is_evidence_page"] is True
    assert data["trace_rows"][1]["state_delta"] == "Active -> Opened"
    assert data["node_rows"][0]["preview"] == "Important title"


def test_aeg_rag_plugin_exposes_reader_evaluator_and_expansion_rows(tmp_path):
    graph_dir = tmp_path / "graph"
    _write_json(graph_dir / "graph.json", {"doc_key": "doc"})
    _write_jsonl(
        graph_dir / "nodes.jsonl",
        [
            {"id": "doc:page:0", "type": "page", "page_index": 0, "abstract": "Page"},
            {"id": "doc:page:0:block:0:title", "type": "title", "page_index": 0, "abstract": "Title"},
        ],
    )
    _write_jsonl(graph_dir / "edges.jsonl", [])
    record = {
        "question_id": "q1",
        "score": 1.0,
        "prepare_metadata": {
            "graph_dir": str(graph_dir),
            "allowed_pages": [0],
            "reader_input": {
                "text_parts": ["Reader prompt"],
                "image_refs": [{"page_index": 0, "image_path": str(tmp_path / "page.png")}],
                "content_part_count": 2,
            },
            "final_node_states": {
                "doc:page:0": "Active",
                "doc:page:0:block:0:title": "Opened",
            },
            "iteration_trace": [
                {
                    "iteration": 1,
                    "action": "EvaluatorDecision",
                    "evaluator_input": {
                        "prompt_text": "controller prompt",
                        "context_xml": "<agent_step_context><question>Q</question></agent_step_context>",
                        "candidate_actions": [{"index": 1, "action_type": "OpenNode"}],
                    },
                    "raw_response": "<agent_decision><stop>false</stop></agent_decision>",
                    "decision": {"stop": False},
                    "state_snapshot_before": {"active_node_ids": ["doc:page:0"], "opened_node_ids": []},
                },
                {
                    "iteration": 1,
                    "action": "OpenNode",
                    "ok": True,
                    "payload": {"node_id": "doc:page:0:block:0:title", "previous_state": "Active"},
                    "selection": {"candidate_index": 1, "candidate_id": "act:OpenNode:test"},
                    "state_snapshot_after": {
                        "active_node_ids": ["doc:page:0"],
                        "opened_node_ids": ["doc:page:0:block:0:title"],
                    },
                },
            ],
        },
    }

    data = AEGRAGPlugin().case_visualization(record)

    assert data["reader_input"]["text_parts"] == ["Reader prompt"]
    assert data["reader_image_refs"][0]["page_index"] == 0
    assert data["evaluator_rows"][0]["prompt_text"] == "controller prompt"
    assert data["evaluator_rows"][0]["candidate_action_count"] == 1
    assert data["expansion_rows"][1]["action"] == "OpenNode"
    assert data["expansion_rows"][1]["selected_candidate_index"] == 1
    assert data["expansion_rows"][1]["opened_nodes"] == 1


def test_aeg_rag_plugin_handles_missing_graph_dir_without_crashing():
    data = AEGRAGPlugin().case_visualization(
        {
            "question_id": "q-missing",
            "score": 0.0,
            "prepare_metadata": {
                "allowed_pages": [3],
                "final_node_states": {"doc:page:3": "Active"},
                "iteration_trace": [{"action": "ActivatePage", "payload": {"page_index": 3}, "ok": True}],
            },
        }
    )

    assert data["graph_available"] is False
    assert data["page_rows"][0]["page_index"] == 3
    assert data["summary"]["trace_steps"] == 1
