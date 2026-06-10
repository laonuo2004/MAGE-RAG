import json
from pathlib import Path

import pandas as pd
import pyarrow as pa

from analysis.plugins import AnalysisPlugin, ParameterSpec, get_plugin, registered_plugins
from analysis.plugins.magerag import MAGERAGPlugin
from analysis.dashboard_app import (
    DEFAULT_RESULT_SUBDIRS,
    _dataframe_for_streamlit,
    _extract_xml_document,
    _image_ref_display_image,
    load_agent_trace_runs,
    load_runs,
)
from analysis.results_loader import parse_run_parameters, read_jsonl_cached, scan_runs
from analysis.results_metrics import (
    build_leaderboard,
    case_rows,
    correction_rows,
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
    assert "magerag" in plugin_names
    assert get_plugin("bm25").name == "bm25"
    assert get_plugin("magerag").name == "magerag"
    assert get_plugin("mage-rag").name == "default"
    assert get_plugin("mage_rag").name == "default"
    assert get_plugin("aeg-rag").name == "default"
    assert get_plugin("aeg_rag").name == "default"
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


def test_scan_runs_can_be_scoped_to_selected_result_subdirectories(tmp_path):
    results_root = tmp_path / "results"
    magerag = results_root / "longdocurl" / "magerag" / "res_top_k_1_model"
    other = results_root / "longdocurl" / "bm25" / "res_top_k_2_model"
    missing = Path("mmlongdoc/magerag")
    _write_jsonl(magerag.with_suffix(".jsonl"), [{"question_id": "q1", "score": 1}])
    _write_jsonl(other.with_suffix(".jsonl"), [{"question_id": "q2", "score": 0}])

    runs = scan_runs(
        results_root,
        included_subdirs=(Path("longdocurl/magerag"), missing),
    )

    assert [run.run_id for run in runs] == ["longdocurl/magerag/res_top_k_1_model"]


def test_dashboard_default_subdirs_are_magerag_only():
    assert Path("longdocurl/magerag") in DEFAULT_RESULT_SUBDIRS
    assert Path("mmlongbench/magerag") in DEFAULT_RESULT_SUBDIRS
    assert Path("mmlongdoc/magerag") in DEFAULT_RESULT_SUBDIRS
    assert all(path.parts[-1] == "magerag" for path in DEFAULT_RESULT_SUBDIRS)


def test_dashboard_load_runs_scans_all_result_subdirectories(tmp_path):
    results_root = tmp_path / "results"
    magerag = results_root / "longdocurl" / "magerag" / "res_top_k_1_model"
    bm25 = results_root / "longdocurl" / "bm25" / "res_top_k_2_model"
    _write_jsonl(magerag.with_suffix(".jsonl"), [{"question_id": "q1", "score": 1}])
    _write_jsonl(bm25.with_suffix(".jsonl"), [{"question_id": "q2", "score": 0}])

    load_runs.clear()
    runs = load_runs(str(results_root))

    assert [run.run_id for run in runs] == [
        "longdocurl/bm25/res_top_k_2_model",
        "longdocurl/magerag/res_top_k_1_model",
    ]


def test_dashboard_agent_trace_runs_are_scoped_to_magerag_subdirectories(tmp_path):
    results_root = tmp_path / "results"
    magerag = results_root / "longdocurl" / "magerag" / "res_top_k_1_model"
    bm25 = results_root / "longdocurl" / "bm25" / "res_top_k_2_model"
    _write_jsonl(magerag.with_suffix(".jsonl"), [{"question_id": "q1", "score": 1}])
    _write_jsonl(bm25.with_suffix(".jsonl"), [{"question_id": "q2", "score": 0}])

    load_agent_trace_runs.clear()
    runs = load_agent_trace_runs(str(results_root))

    assert [run.run_id for run in runs] == ["longdocurl/magerag/res_top_k_1_model"]


def test_dashboard_dataframe_for_streamlit_serializes_mixed_object_columns():
    df = _dataframe_for_streamlit(
        pd.DataFrame(
            [
                {"question_id": "q1", "answer": ["white", "10%"], "score": 0.0},
                {"question_id": "q2", "answer": "less well-off", "score": 1.0},
            ]
        )
    )

    assert df["answer"].tolist() == ['["white", "10%"]', "less well-off"]
    assert df["score"].tolist() == [0.0, 1.0]
    pa.Table.from_pandas(df)


def test_dashboard_extract_xml_document_wraps_reader_think_and_answer_blocks():
    wrapped = _extract_xml_document("<think>Reason.</think><answer>Final.</answer>")

    assert wrapped == "<analysis_xml><think>Reason.</think><answer>Final.</answer></analysis_xml>"


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


def test_correction_rows_classify_before_after_score_outcomes():
    rows = correction_rows(
        [
            {
                "question_id": "q1",
                "answer": "Table 29. 12-bit DAC operating requirements",
                "pred": "Table 29. 12-bit DAC operating requirements",
                "score": 1.0,
                "correction_metadata": {
                    "applied": True,
                    "initial_pred": "Table 29",
                    "initial_pred_format": "String",
                    "initial_score": 0.0,
                    "corrected_pred": "Table 29. 12-bit DAC operating requirements",
                    "corrected_pred_format": "String",
                    "corrected_score": 1.0,
                    "duration_seconds": 2.4,
                    "corrected_extracted_res": "Extracted answer: Table 29. 12-bit DAC operating requirements",
                },
            },
            {
                "question_id": "q2",
                "answer": "A",
                "pred": "B",
                "score": 0.0,
                "correction_metadata": {
                    "applied": False,
                    "initial_pred": "B",
                    "initial_score": 0.0,
                    "corrected_pred": "B",
                    "corrected_score": 0.0,
                },
            },
            {
                "question_id": "q3",
                "answer": "C",
                "pred": "C",
                "score": 1.0,
            },
            {
                "question_id": "q4",
                "answer": "D",
                "pred": "D",
                "score": 1.0,
                "correction_metadata": {
                    "applied": False,
                    "initial_pred": "D",
                    "initial_score": 1.0,
                },
            },
            {
                "question_id": "q5",
                "answer": "E",
                "pred": "F",
                "score": 0.0,
                "correction_metadata": {
                    "applied": False,
                    "initial_pred": "F",
                    "initial_score": 0.0,
                    "error": "Failed to parse corrected extraction",
                },
            },
        ]
    )

    by_id = {row["question_id"]: row for row in rows}
    assert by_id["q1"]["correction_outcome"] == "wrong_to_correct"
    assert by_id["q1"]["score_delta"] == 1.0
    assert by_id["q1"]["pred_changed"] is True
    assert by_id["q2"]["correction_outcome"] == "wrong_to_wrong"
    assert by_id["q3"]["correction_outcome"] == "not_run"
    assert by_id["q4"]["correction_outcome"] == "skipped_initial_correct"
    assert by_id["q5"]["correction_outcome"] == "correction_error"


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


def test_magerag_plugin_builds_case_visualization_from_trace_and_graph(tmp_path):
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

    data = MAGERAGPlugin().case_visualization(record)

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


def test_magerag_plugin_exposes_reader_evaluator_and_expansion_rows(tmp_path):
    graph_dir = tmp_path / "graph"
    _write_json(graph_dir / "graph.json", {"doc_key": "doc"})
    _write_jsonl(
        graph_dir / "nodes.jsonl",
        [
            {"id": "doc:page:0", "type": "page", "page_index": 0, "abstract": "Page"},
            {"id": "doc:page:0:block:0:title", "type": "title", "page_index": 0, "abstract": "Title"},
        ],
    )
    _write_jsonl(
        graph_dir / "edges.jsonl",
        [
            {
                "id": "edge:containment:doc:page:0:title",
                "source": "doc:page:0",
                "target": "doc:page:0:block:0:title",
                "type": "containment",
                "relation": "contains",
            }
        ],
    )
    record = {
        "question_id": "q1",
        "score": 1.0,
        "generation_metadata": {
            "response": "Reader parsed answer",
            "raw_response": "<think>Reason about the evidence.</think><answer>Reader parsed answer</answer>",
        },
        "extraction_metadata": {
            "input_messages": [{"role": "user", "content": [{"type": "text", "text": "Extraction prompt"}]}],
            "extracted_res": "Extracted answer: page 1\nAnswer format: String",
            "model": "extractor",
        },
        "correction_metadata": {
            "input_messages": [{"role": "user", "content": [{"type": "text", "text": "Correction prompt"}]}],
            "corrected_extracted_res": "Extracted answer: page 1\nAnswer format: String",
            "initial_pred": "page one",
            "initial_score": 0.0,
            "corrected_pred": "page 1",
            "corrected_score": 1.0,
            "changed": True,
            "improved": True,
            "applied": True,
        },
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

    data = MAGERAGPlugin().case_visualization(record)

    assert data["reader_input"]["text_parts"] == ["Reader prompt"]
    assert data["reader_output"] == "Reader parsed answer"
    assert data["reader_raw_output"] == "<think>Reason about the evidence.</think><answer>Reader parsed answer</answer>"
    assert data["reader_think_output"] == "Reason about the evidence."
    assert data["reader_image_refs"][0]["page_index"] == 0
    assert data["extraction_io"]["input"]["prompt_text"] == "Extraction prompt"
    assert data["extraction_io"]["output"] == "Extracted answer: page 1\nAnswer format: String"
    assert data["correction_io"]["input"]["prompt_text"] == "Correction prompt"
    assert data["correction_io"]["output"] == "Extracted answer: page 1\nAnswer format: String"
    assert data["correction_io"]["changed"] is True
    assert data["correction_io"]["improved"] is True
    assert data["evaluator_rows"][0]["prompt_text"] == "controller prompt"
    assert data["evaluator_rows"][0]["iteration_label"] == "Iteration 1"
    assert data["evaluator_rows"][0]["candidate_action_count"] == 1
    assert data["expansion_rows"][1]["action"] == "OpenNode"
    assert data["expansion_rows"][1]["selected_candidate_index"] == 1
    assert data["expansion_rows"][1]["opened_nodes"] == 1
    assert data["graph_snapshots"][0]["iteration"] == 1
    assert data["graph_snapshots"][0]["nodes"][0]["state"] == "Active"
    assert data["graph_snapshots"][0]["edges"][0]["edge_type"] == "containment"


def test_agent_trace_rows_use_corrected_score_for_bucket_and_display():
    rows = case_rows(
        [
            {
                "question_id": "q1",
                "question": "Q",
                "answer": "A",
                "pred": "initial",
                "score": 0.0,
                "corrected_score": 1.0,
                "correction_metadata": {
                    "initial_score": 0.0,
                    "corrected_score": 1.0,
                    "corrected_pred": "fixed",
                },
            }
        ]
    )

    assert rows[0]["score"] == 1.0
    assert rows[0]["score_bucket"] == "correct"
    assert rows[0]["initial_score"] == 0.0
    assert rows[0]["corrected_score"] == 1.0


def test_magerag_plugin_summary_and_diagnostics_use_corrected_score():
    record = {
        "question_id": "q1",
        "score": 0.0,
        "corrected_score": 1.0,
        "prepare_metadata": {"iteration_trace": []},
        "correction_metadata": {
            "initial_score": 0.0,
            "corrected_score": 1.0,
            "corrected_pred": "fixed",
        },
    }

    plugin = MAGERAGPlugin()
    data = plugin.case_visualization(record)
    diagnostic_rows = plugin.diagnostic_rows([record])

    assert data["summary"]["score"] == 1.0
    assert diagnostic_rows[0]["score"] == 1.0


def test_magerag_plugin_handles_missing_graph_dir_without_crashing():
    data = MAGERAGPlugin().case_visualization(
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


def test_magerag_plugin_does_not_fake_raw_reader_output_from_parsed_response():
    data = MAGERAGPlugin().case_visualization(
        {
            "question_id": "q-no-raw",
            "generation_metadata": {"response": "Parsed answer only"},
            "prepare_metadata": {},
        }
    )

    assert data["reader_output"] == "Parsed answer only"
    assert data["reader_raw_output"] is None
    assert data["reader_think_output"] is None


def test_dashboard_image_ref_display_crops_opened_node_bbox(tmp_path):
    from PIL import Image

    page_path = tmp_path / "page.png"
    Image.new("RGB", (200, 200), color="white").save(page_path)

    display = _image_ref_display_image(
        {
            "kind": "opened_node_crop",
            "image_path": str(page_path),
            "bbox": [250, 250, 750, 750],
        }
    )

    assert not isinstance(display, str)
    assert display.size == (116, 116)
