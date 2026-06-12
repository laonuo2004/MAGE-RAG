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
from analysis.paper_main_results import main_result_rows, write_main_results_artifacts
from analysis.paper_breakdown import breakdown_rows, write_breakdown_artifacts
from analysis.paper_budget_ablation import magerag_budget_rows, write_budget_ablation_artifacts
from analysis.paper_case_study import magerag_case_rows, write_case_study_artifacts
from analysis.paper_diagnostics import write_paper_diagnostic_artifacts
from analysis.paper_artifacts import write_all_paper_artifacts
from analysis.paper_registry import (
    audit_run,
    normalize_int_list,
    official_paper_runs,
    paper_run_diagnostics,
)
from analysis.paper_trace_statistics import magerag_trace_rows, write_trace_statistics_artifacts
from analysis.paper_efficiency import magerag_efficiency_rows, write_efficiency_artifacts
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


def test_magerag_plugin_parses_controller_budget_and_graph_mode_from_filename(tmp_path):
    results_root = tmp_path / "results"
    run = (
        results_root
        / "mmlongbench"
        / "magerag"
        / "res_top_k_3_mode_dynamic_controller_no_search_watchdog_iterations_10_"
        "max_selected_actions_per_iteration_5_mode_layout_graph_Qwen3_VL_8B_Instruct"
    )
    _write_json(run.with_suffix(".metrics.json"), {"overall_acc": 0.5})

    runs = scan_runs(results_root)

    assert runs[0].parameters["top_k"] == 3
    assert runs[0].parameters["controller_mode"] == "dynamic_controller_no_search"
    assert runs[0].parameters["watchdog_iterations"] == 10
    assert runs[0].parameters["max_selected_actions_per_iteration"] == 5
    assert runs[0].parameters["graph_mode"] == "layout_graph"
    assert runs[0].parameter_sources["graph_mode"] == "filename.magerag_pattern"


def test_paper_budget_ablation_artifacts_filter_zero_completion_runs(tmp_path):
    results_root = tmp_path / "results"
    good = (
        results_root
        / "mmlongbench"
        / "magerag"
        / "res_top_k_3_mode_full_watchdog_iterations_5_"
        "max_selected_actions_per_iteration_3_mode_full_graph_Qwen3_VL_8B_Instruct"
    )
    zero = (
        results_root
        / "mmlongbench"
        / "magerag"
        / "res_top_k_8_mode_full_watchdog_iterations_10_"
        "max_selected_actions_per_iteration_5_mode_full_graph_Qwen3_VL_8B_Instruct"
    )
    _write_json(
        good.with_suffix(".metrics.json"),
        {
            "overall_acc": 0.52,
            "overall_f1": 0.49,
            "breakdowns": {
                "single_page": {"acc": 0.60},
                "cross_page": {"acc": 0.31},
                "unanswerable": {"acc": 0.70},
            },
            "sample_count": 10,
            "completed_count": 10,
            "failed_count": 0,
        },
    )
    _write_json(
        zero.with_suffix(".metrics.json"),
        {
            "overall_acc": 0.0,
            "overall_f1": 0.0,
            "sample_count": 10,
            "completed_count": 0,
            "failed_count": 10,
        },
    )

    rows = magerag_budget_rows(scan_runs(results_root))

    assert len(rows) == 1
    assert rows[0]["top_k"] == 3
    assert rows[0]["overall_acc"] == 0.52


def test_paper_main_results_choose_best_completed_run_and_keep_missing_cells(tmp_path):
    results_root = tmp_path / "results"
    _write_json(
        results_root / "longdocurl" / "bm25" / "res_model.metrics.json",
        {
            "overall_acc": 0.3,
            "fine_grained_metrics": {
                "Main_Task": {"Understanding": 0.4, "Reasoning": 0.2, "Locating": 0.1},
                "Evidence_Pages": {"Single_Page": 0.5, "Multi_Page": 0.25},
            },
            "sample_count": 10,
            "completed_count": 10,
            "failed_count": 0,
        },
    )
    _write_json(
        results_root / "mmlongbench" / "magerag" / (
            "res_top_k_3_mode_full_watchdog_iterations_5_"
            "max_selected_actions_per_iteration_3_mode_full_graph_Qwen3_VL_8B_Instruct.metrics.json"
        ),
        {
            "overall_acc": 0.4,
            "overall_f1": 0.35,
            "breakdowns": {
                "single_page": {"acc": 0.5},
                "cross_page": {"acc": 0.2},
                "unanswerable": {"acc": 0.8},
            },
            "sample_count": 12,
            "completed_count": 12,
            "failed_count": 0,
        },
    )
    _write_json(
        results_root / "mmlongbench" / "magerag" / (
            "res_top_k_8_mode_full_watchdog_iterations_10_"
            "max_selected_actions_per_iteration_5_mode_full_graph_Qwen3_VL_8B_Instruct.metrics.json"
        ),
        {"overall_acc": 0.9, "overall_f1": 0.9, "sample_count": 12, "completed_count": 0, "failed_count": 12},
    )

    rows = main_result_rows(scan_runs(results_root))
    by_method = {row["method"]: row for row in rows}

    assert by_method["BM25"]["longdocurl_overall"] == 0.3
    assert by_method["BM25"]["longdocurl_understanding"] == 0.4
    assert by_method["BM25"]["longdocurl_reasoning"] == 0.2
    assert by_method["BM25"]["longdocurl_locating"] == 0.1
    assert by_method["BM25"]["mmlongbench_acc"] is None
    assert by_method["MAGE-RAG"]["mmlongbench_acc"] == 0.4
    assert by_method["MAGE-RAG"]["mmlongbench_f1"] == 0.35
    assert by_method["MAGE-RAG"]["mmlongbench_done"] == "12/12"


def test_paper_main_results_do_not_select_ablation_only_magerag_variants(tmp_path):
    results_root = tmp_path / "results"
    primary = (
        results_root
        / "mmlongbench"
        / "magerag"
        / "res_top_k_3_mode_full_watchdog_iterations_5_max_selected_actions_per_iteration_3_mode_full_graph_Qwen3_VL_8B_Instruct"
    )
    ablation = (
        results_root
        / "mmlongbench"
        / "magerag"
        / "res_top_k_3_mode_graph_neighbor_expansion_watchdog_iterations_10_max_selected_actions_per_iteration_5_mode_full_graph_Qwen3_VL_8B_Instruct"
    )
    for path, acc in [(primary, 0.52), (ablation, 0.90)]:
        _write_json(
            path.with_suffix(".metrics.json"),
            {
                "overall_acc": acc,
                "overall_f1": acc - 0.03,
                "breakdowns": {
                    "single_page": {"acc": 0.60},
                    "cross_page": {"acc": 0.31},
                    "unanswerable": {"acc": 0.70},
                },
                "sample_count": 10,
                "completed_count": 10,
                "failed_count": 0,
            },
        )

    rows = main_result_rows(scan_runs(results_root))
    magerag = next(row for row in rows if row["method"] == "MAGE-RAG")

    assert magerag["mmlongbench_acc"] == 0.52
    assert magerag["mmlongbench_f1"] == 0.49


def test_paper_registry_excludes_backup_and_classifies_run_statuses(tmp_path):
    results_root = tmp_path / "results"
    complete = results_root / "mmlongbench" / "magerag" / "res_top_k_3_mode_full_watchdog_iterations_5_max_selected_actions_per_iteration_3_mode_full_graph_Qwen3_VL_8B_Instruct"
    empty = results_root / "mmlongbench" / "magerag" / "res_top_k_8_mode_full_watchdog_iterations_5_max_selected_actions_per_iteration_3_mode_full_graph_Qwen3_VL_8B_Instruct"
    backup = results_root / "mmlongbench" / "backup" / "res_top_k_1_model"
    _write_json(complete.with_suffix(".metrics.json"), {"overall_acc": 0.5, "sample_count": 3, "completed_count": 3, "failed_count": 0})
    _write_json(empty.with_suffix(".metrics.json"), {"overall_acc": 0.0, "sample_count": 3, "completed_count": 0, "failed_count": 3})
    _write_json(backup.with_suffix(".metrics.json"), {"overall_acc": 0.9, "sample_count": 3, "completed_count": 3, "failed_count": 0})

    runs = scan_runs(results_root)
    diagnostics = paper_run_diagnostics(runs)
    by_id = {item["run_id"]: item for item in diagnostics}

    assert by_id["mmlongbench/magerag/res_top_k_3_mode_full_watchdog_iterations_5_max_selected_actions_per_iteration_3_mode_full_graph_Qwen3_VL_8B_Instruct"]["status"] == "complete"
    assert by_id["mmlongbench/magerag/res_top_k_8_mode_full_watchdog_iterations_5_max_selected_actions_per_iteration_3_mode_full_graph_Qwen3_VL_8B_Instruct"]["status"] == "empty"
    assert "mmlongbench/backup/res_top_k_1_model" in by_id
    assert by_id["mmlongbench/backup/res_top_k_1_model"]["paper_excluded"] is True

    selected = official_paper_runs(runs)

    assert ("mmlongbench", "magerag") in selected
    assert selected[("mmlongbench", "magerag")].run_id.endswith("mode_full_graph_Qwen3_VL_8B_Instruct")
    assert all(run.baseline != "backup" for run in selected.values())


def test_normalize_int_list_handles_jsonish_strings_and_page_dicts():
    assert normalize_int_list("[101, 3]") == [101, 3]
    assert normalize_int_list("[{'page_number': 4}, {'page_index': 1}]") == [4, 1]
    assert normalize_int_list("[]") == []
    assert normalize_int_list(["5", {"page": 6}, {"bad": "x"}]) == [5, 6]


def test_paper_main_results_artifacts_write_csv_and_latex(tmp_path):
    results_root = tmp_path / "results"
    _write_json(
        results_root / "longdocurl" / "image" / "res_model.metrics.json",
        {
            "overall_acc": 0.5,
            "fine_grained_metrics": {
                "Main_Task": {"Understanding": 0.6, "Reasoning": 0.4, "Locating": 0.3},
                "Evidence_Pages": {"Single_Page": 0.7, "Multi_Page": 0.2},
            },
            "sample_count": 10,
            "completed_count": 9,
            "failed_count": 1,
        },
    )
    _write_json(
        results_root / "mmlongbench" / "image" / "res_model.metrics.json",
        {
            "overall_acc": 0.45,
            "overall_f1": 0.4,
            "breakdowns": {
                "single_page": {"acc": 0.55},
                "cross_page": {"acc": 0.25},
                "unanswerable": {"acc": 0.75},
            },
            "sample_count": 12,
            "completed_count": 12,
            "failed_count": 0,
        },
    )

    artifacts = write_main_results_artifacts(scan_runs(results_root), tmp_path / "cache", tmp_path / "tables")

    assert artifacts.csv_path.exists()
    assert artifacts.tex_path.exists()
    tex = artifacts.tex_path.read_text(encoding="utf-8")
    assert "\\label{tab:main_results}" in tex
    assert "Direct MLLM" in tex
    assert "50.00" in tex
    assert "9/10" in tex
    assert "\\underline{50.00}" in tex
    assert "\\textbf{70.00}" in tex
    assert "\\underline{20.00}" in tex
    assert "\\underline{45.00}" in tex
    assert "\\textbf{55.00}" in tex
    assert "\\underline{25.00}" in tex
    assert "\\textbf{75.00}" in tex


def test_paper_breakdown_rows_reuse_selected_main_result_metrics():
    rows = breakdown_rows(
        [
            {
                "method": "BM25",
                "type": "Text RAG",
                "longdocurl_understanding": 0.40,
                "longdocurl_reasoning": 0.20,
                "longdocurl_locating": 0.10,
                "longdocurl_text": 0.42,
                "longdocurl_layout": 0.32,
                "longdocurl_figure": 0.22,
                "longdocurl_table": 0.12,
                "longdocurl_single_page": 0.50,
                "longdocurl_multi_page": 0.25,
                "mmlongbench_single_page": None,
                "mmlongbench_cross_page": None,
                "mmlongbench_unanswerable": None,
            },
            {
                "method": "MAGE-RAG",
                "type": "Graph/Agentic RAG",
                "longdocurl_understanding": 0.60,
                "longdocurl_reasoning": 0.45,
                "longdocurl_locating": 0.35,
                "longdocurl_text": 0.62,
                "longdocurl_layout": 0.52,
                "longdocurl_figure": 0.42,
                "longdocurl_table": 0.32,
                "longdocurl_single_page": 0.70,
                "longdocurl_multi_page": 0.55,
                "mmlongbench_single_page": 0.65,
                "mmlongbench_cross_page": 0.33,
                "mmlongbench_unanswerable": 0.75,
            },
        ]
    )

    by_key = {(row["method"], row["benchmark"], row["split"]): row for row in rows}
    assert by_key[("BM25", "LongDocURL", "Understanding")]["score"] == 0.40
    assert by_key[("BM25", "LongDocURL", "Text")]["group"] == "Element type"
    assert by_key[("BM25", "LongDocURL", "Multi-page")]["group"] == "Evidence pages"
    assert ("BM25", "MMLongBench-Doc", "Single-page") not in by_key
    assert by_key[("MAGE-RAG", "MMLongBench-Doc", "Cross-page")]["score_pct"] == 33.0


def test_paper_breakdown_artifacts_write_csv_tex_and_figure(tmp_path):
    main_rows = [
        {
            "method": "Direct MLLM",
            "type": "Direct MLLM",
            "longdocurl_understanding": 0.55,
            "longdocurl_reasoning": 0.45,
            "longdocurl_locating": 0.35,
            "longdocurl_text": 0.58,
            "longdocurl_layout": 0.48,
            "longdocurl_figure": 0.38,
            "longdocurl_table": 0.28,
            "longdocurl_single_page": 0.60,
            "longdocurl_multi_page": 0.40,
            "mmlongbench_single_page": 0.50,
            "mmlongbench_cross_page": 0.20,
            "mmlongbench_unanswerable": 0.70,
        },
        {
            "method": "MAGE-RAG",
            "type": "Graph/Agentic RAG",
            "longdocurl_understanding": 0.65,
            "longdocurl_reasoning": 0.50,
            "longdocurl_locating": 0.45,
            "longdocurl_text": 0.68,
            "longdocurl_layout": 0.58,
            "longdocurl_figure": 0.48,
            "longdocurl_table": 0.38,
            "longdocurl_single_page": 0.75,
            "longdocurl_multi_page": 0.55,
            "mmlongbench_single_page": 0.60,
            "mmlongbench_cross_page": 0.35,
            "mmlongbench_unanswerable": 0.80,
        },
    ]

    artifacts = write_breakdown_artifacts(main_rows, tmp_path / "paper_cache", tmp_path / "images", tmp_path / "tables")

    assert artifacts.csv_path.exists()
    assert artifacts.figure_path.exists()
    assert artifacts.tex_path.exists()
    csv = pd.read_csv(artifacts.csv_path)
    assert set(csv["group"]) == {"Task", "Element type", "Evidence pages", "Question type"}
    tex = artifacts.tex_path.read_text(encoding="utf-8")
    assert "\\label{fig:main_breakdown}" in tex
    assert "images/main_breakdown.pdf" in tex


def test_paper_budget_ablation_artifacts_write_csv_tex_and_figure(tmp_path):
    results_root = tmp_path / "results"
    for stem, acc in [
        (
            "res_top_k_3_mode_full_watchdog_iterations_5_"
            "max_selected_actions_per_iteration_3_mode_full_graph_Qwen3_VL_8B_Instruct",
            0.52,
        ),
        (
            "res_top_k_3_mode_topk_page_only_watchdog_iterations_10_"
            "max_selected_actions_per_iteration_5_mode_full_graph_Qwen3_VL_8B_Instruct",
            0.48,
        ),
    ]:
        run = results_root / "mmlongbench" / "magerag" / stem
        _write_json(
            run.with_suffix(".metrics.json"),
            {
                "overall_acc": acc,
                "overall_f1": acc - 0.03,
                "breakdowns": {
                    "single_page": {"acc": acc + 0.05},
                    "cross_page": {"acc": acc - 0.20},
                    "unanswerable": {"acc": acc + 0.15},
                },
                "sample_count": 10,
                "completed_count": 10,
                "failed_count": 0,
            },
        )

    artifacts = write_budget_ablation_artifacts(
        scan_runs(results_root),
        tmp_path / "paper_cache",
        tmp_path / "images",
    )

    assert artifacts.budget_csv.exists()
    assert artifacts.ablation_csv.exists()
    assert artifacts.budget_tex.exists()
    assert artifacts.ablation_tex.exists()
    assert artifacts.budget_figure.exists()
    budget_tex = artifacts.budget_tex.read_text(encoding="utf-8")
    ablation_tex = artifacts.ablation_tex.read_text(encoding="utf-8")
    assert "\\label{tab:magerag_budget}" in budget_tex
    assert "3 & 5 & 3 & 52.00" in budget_tex
    assert "52.00 & 49.00 & 57.00 & 32.00 & 67.00 & 10/10 \\\\" in budget_tex
    assert "Topk Page Only" in ablation_tex


def test_paper_trace_statistics_rows_split_correctness_and_actions():
    rows = magerag_trace_rows(
        [
            {
                "question_id": "q1",
                "score": 1.0,
                "prepare_metadata": {
                    "stop_reason": "controller_stop",
                    "activated_pages": [1, 3],
                    "active_node_ids": ["p1", "p3"],
                    "opened_node_ids": ["n1", "n2"],
                    "pruned_node_ids": ["n3"],
                    "reader_input": {"image_refs": [{"path": "a.png"}], "content_part_count": 4},
                    "logical_cost": {"estimated_input_tokens": 120, "llm_calls": 2, "retriever_calls": 1},
                    "iteration_trace": [
                        {"iteration": 0, "action": "ActivatePage"},
                        {"iteration": 1, "action": "OpenNode"},
                        {"iteration": 1, "action": "SearchEvidence"},
                        {"iteration": 1, "action": "SearchEvidenceRetrieval"},
                        {"iteration": 2, "action": "PruneNode"},
                    ],
                },
            },
            {
                "question_id": "q2",
                "score": 0.0,
                "prepare_metadata": {
                    "stop_reason": "fallback_invalid_xml",
                    "active_node_ids": ["p2"],
                    "opened_node_ids": [],
                    "pruned_node_ids": [],
                    "reader_input": {"image_refs": [], "content_part_count": 1},
                    "iteration_trace": [{"iteration": 0, "action": "ActivatePage"}],
                },
            },
        ]
    )

    by_id = {row["question_id"]: row for row in rows}
    assert by_id["q1"]["correctness"] == "correct"
    assert by_id["q1"]["trace_steps"] == 5
    assert by_id["q1"]["activated_pages"] == 2
    assert by_id["q1"]["opened_nodes"] == 2
    assert by_id["q1"]["search_count"] == 2
    assert by_id["q1"]["prune_count"] == 1
    assert by_id["q1"]["max_iteration"] == 2
    assert by_id["q1"]["reader_images"] == 1
    assert by_id["q1"]["estimated_input_tokens"] == 120
    assert by_id["q2"]["correctness"] == "incorrect"
    assert by_id["q2"]["opened_nodes"] == 0


def test_paper_trace_statistics_rows_prefer_magerag_trace_summary():
    rows = magerag_trace_rows(
        [
            {
                "question_id": "q-summary",
                "score": 1.0,
                "prepare_metadata": {
                    "stop_reason": "legacy_stop",
                    "activated_pages": [0],
                    "active_node_ids": ["p1"],
                    "opened_node_ids": ["n1"],
                    "reader_input": {"image_refs": [{"path": "a.png"}], "content_part_count": 4},
                    "logical_cost": {"num_llm_calls": 7, "num_retriever_calls": 3},
                    "magerag": {
                        "trace_summary": {
                            "stop_reason": "summary_stop",
                            "num_trace_events": 12,
                            "num_iterations": 5,
                            "num_activate_page": 4,
                            "num_open_node": 9,
                            "num_prune_node": 2,
                            "num_search_requests": 3,
                            "num_search_retrievals": 2,
                        }
                    },
                    "iteration_trace": [{"iteration": 0, "action": "ActivatePage"}],
                },
            }
        ]
    )

    row = rows[0]
    assert row["stop_reason"] == "summary_stop"
    assert row["trace_steps"] == 12
    assert row["max_iteration"] == 5
    assert row["activated_pages"] == 4
    assert row["opened_nodes"] == 9
    assert row["pruned_nodes"] == 2
    assert row["search_count"] == 5
    assert row["prune_count"] == 2
    assert row["llm_calls"] == 7
    assert row["retriever_calls"] == 3


def test_paper_trace_statistics_artifacts_write_csv_tex_and_figure(tmp_path):
    jsonl_path = tmp_path / "run.jsonl"
    _write_jsonl(
        jsonl_path,
        [
            {
                "question_id": "q1",
                "score": 1.0,
                "prepare_metadata": {
                    "stop_reason": "done",
                    "active_node_ids": ["p1"],
                    "opened_node_ids": ["n1", "n2"],
                    "pruned_node_ids": [],
                    "reader_input": {"image_refs": [{"path": "a.png"}], "content_part_count": 3},
                    "iteration_trace": [
                        {"iteration": 0, "action": "ActivatePage"},
                        {"iteration": 1, "action": "OpenNode"},
                        {"iteration": 1, "action": "OpenNode"},
                    ],
                },
            },
            {
                "question_id": "q2",
                "score": 0.0,
                "prepare_metadata": {
                    "stop_reason": "done",
                    "active_node_ids": ["p2"],
                    "opened_node_ids": [],
                    "pruned_node_ids": ["n3"],
                    "reader_input": {"image_refs": [], "content_part_count": 1},
                    "iteration_trace": [
                        {"iteration": 0, "action": "ActivatePage"},
                        {"iteration": 1, "action": "PruneNode"},
                    ],
                },
            },
        ],
    )

    artifacts = write_trace_statistics_artifacts(
        jsonl_path,
        tmp_path / "paper_cache",
        tmp_path / "images",
        tmp_path / "tables",
    )

    assert artifacts.trace_csv.exists()
    assert artifacts.summary_csv.exists()
    assert artifacts.summary_tex.exists()
    assert artifacts.figure.exists()
    summary_tex = artifacts.summary_tex.read_text(encoding="utf-8")
    assert "\\label{tab:magerag_trace_stats}" in summary_tex
    assert "Correct" in summary_tex
    assert "Incorrect" in summary_tex
    assert "Correct & 1 & 3.00 & 1.00 & 1.00 & 2.00 & 0.00 & 0.00 & 1.00 \\\\" in summary_tex


def test_paper_efficiency_rows_use_logical_cost_and_reader_fallbacks():
    rows = magerag_efficiency_rows(
        [
            {
                "question_id": "q1",
                "score": 1.0,
                "prepare_metadata": {
                    "active_node_ids": ["p1"],
                    "opened_node_ids": ["n1", "n2"],
                    "reader_input": {"image_refs": [{"path": "a.png"}], "content_part_count": 4},
                    "logical_cost": {
                        "num_input_images": 3,
                        "num_context_pages": 2,
                        "num_context_nodes": 5,
                        "num_retriever_calls": 1,
                        "num_llm_calls": 2,
                        "num_input_text_chars": 1200,
                        "estimated_input_tokens": None,
                    },
                },
            },
            {
                "question_id": "q2",
                "score": 0.0,
                "prepare_metadata": {
                    "active_node_ids": ["p2", "p3"],
                    "opened_node_ids": ["n3"],
                    "reader_input": {"image_refs": [{"path": "b.png"}, {"path": "c.png"}], "content_part_count": 7},
                    "logical_cost": {"num_retriever_calls": 2, "num_llm_calls": 3},
                },
            },
        ]
    )

    by_id = {row["question_id"]: row for row in rows}
    assert by_id["q1"]["correctness"] == "correct"
    assert by_id["q1"]["input_images"] == 3
    assert by_id["q1"]["context_pages"] == 2
    assert by_id["q1"]["context_nodes"] == 5
    assert by_id["q1"]["input_text_chars"] == 1200
    assert by_id["q1"]["estimated_input_tokens"] is None
    assert by_id["q2"]["correctness"] == "incorrect"
    assert by_id["q2"]["input_images"] == 2
    assert by_id["q2"]["context_pages"] == 2
    assert by_id["q2"]["context_nodes"] == 1
    assert by_id["q2"]["content_part_count"] == 7


def test_paper_efficiency_artifacts_write_summary_csv_tex_and_figure(tmp_path):
    jsonl_path = tmp_path / "run.jsonl"
    _write_jsonl(
        jsonl_path,
        [
            {
                "question_id": "q1",
                "score": 1.0,
                "prepare_metadata": {
                    "reader_input": {"image_refs": [{"path": "a.png"}], "content_part_count": 4},
                    "logical_cost": {
                        "num_input_images": 2,
                        "num_context_pages": 2,
                        "num_context_nodes": 4,
                        "num_retriever_calls": 1,
                        "num_llm_calls": 2,
                        "num_input_text_chars": 100,
                    },
                },
            },
            {
                "question_id": "q2",
                "score": 0.0,
                "prepare_metadata": {
                    "reader_input": {"image_refs": [{"path": "b.png"}, {"path": "c.png"}], "content_part_count": 6},
                    "logical_cost": {
                        "num_input_images": 4,
                        "num_context_pages": 3,
                        "num_context_nodes": 8,
                        "num_retriever_calls": 2,
                        "num_llm_calls": 3,
                        "num_input_text_chars": 300,
                    },
                },
            },
        ],
    )

    artifacts = write_efficiency_artifacts(
        jsonl_path,
        tmp_path / "paper_cache",
        tmp_path / "images",
        tmp_path / "tables",
    )

    assert artifacts.rows_csv.exists()
    assert artifacts.summary_csv.exists()
    assert artifacts.summary_tex.exists()
    assert artifacts.figure_path.exists()
    summary = pd.read_csv(artifacts.summary_csv)
    overall = summary[summary["group"] == "overall"].iloc[0]
    assert overall["count"] == 2
    assert overall["acc"] == 0.5
    assert overall["avg_input_images"] == 3.0
    assert abs(overall["score_per_input_image"] - (1 / 6)) < 1e-12
    assert overall["score_per_llm_call"] == 1 / 5
    tex = artifacts.summary_tex.read_text(encoding="utf-8")
    assert "\\label{tab:magerag_efficiency}" in tex
    assert "Overall & 2 & 50.00 & 3.00 & 2.50 & 2.50 & 5.00 & 6.00 & 0.167 & 0.200 \\\\" in tex


def test_paper_case_study_rows_select_rich_correct_and_incorrect_cases():
    records = [
        {
            "question_id": "easy_correct",
            "question": "What is shown?",
            "answer": "A",
            "pred": "A",
            "score": 1.0,
            "prepare_metadata": {
                "stop_reason": "fallback_invalid_xml",
                "active_node_ids": ["p1"],
                "opened_node_ids": [],
                "pruned_node_ids": [],
                "reader_input": {"image_refs": [{"page_index": 0}], "content_part_count": 1},
                "iteration_trace": [{"iteration": 0, "action": "ActivatePage", "payload": {"page_index": 0}}],
            },
        },
        {
            "question_id": "rich_correct",
            "question": "Which restaurants are recommended?",
            "answer": ["Chen Mapo Doufu", "Yu Zhi Lan"],
            "pred": ["Chen Mapo Doufu", "Yu Zhi Lan"],
            "score": 1.0,
            "prepare_metadata": {
                "stop_reason": "watchdog_iterations",
                "active_node_ids": ["p1", "p2"],
                "opened_node_ids": ["n1", "n2", "n3"],
                "pruned_node_ids": ["n4"],
                "reader_input": {"image_refs": [{"page_index": 1}, {"page_index": 2}], "content_part_count": 5},
                "iteration_trace": [
                    {"iteration": 0, "action": "ActivatePage", "payload": {"page_index": 1}},
                    {"iteration": 1, "action": "EvaluatorDecision"},
                    {"iteration": 1, "action": "OpenNode", "payload": {"node_id": "n1", "page_index": 1}},
                    {"iteration": 1, "action": "SearchEvidence"},
                    {"iteration": 2, "action": "PruneNode", "payload": {"node_id": "n4", "page_index": 2}},
                ],
            },
        },
        {
            "question_id": "rich_incorrect",
            "question": "What color is the line?",
            "answer": "red",
            "corrected_pred": "orange",
            "corrected_score": 0.0,
            "prepare_metadata": {
                "stop_reason": "watchdog_iterations",
                "active_node_ids": ["p3"],
                "opened_node_ids": ["n5", "n6"],
                "pruned_node_ids": [],
                "reader_input": {"image_refs": [{"page_index": 4}], "content_part_count": 3},
                "iteration_trace": [
                    {"iteration": 0, "action": "ActivatePage", "payload": {"page_index": 4}},
                    {"iteration": 1, "action": "EvaluatorDecision"},
                    {"iteration": 1, "action": "OpenNode", "payload": {"node_id": "n5", "page_index": 4}},
                    {"iteration": 1, "action": "SearchEvidenceRetrieval"},
                ],
            },
        },
    ]

    rows = magerag_case_rows(records)

    assert [row["case_type"] for row in rows] == ["success", "failure"]
    assert rows[0]["question_id"] == "rich_correct"
    assert rows[0]["opened_nodes"] == 3
    assert rows[0]["search_count"] == 1
    assert rows[0]["trace_excerpt"] == "ActivatePage -> EvaluatorDecision -> OpenNode -> SearchEvidence -> PruneNode"
    assert rows[1]["question_id"] == "rich_incorrect"
    assert rows[1]["pred"] == "orange"


def test_paper_case_study_rows_prefer_answerable_cases_when_available():
    base_metadata = {
        "stop_reason": "watchdog_iterations",
        "active_node_ids": ["p1"],
        "opened_node_ids": ["n1", "n2"],
        "pruned_node_ids": [],
        "reader_input": {"image_refs": [{"page_index": 1}], "content_part_count": 4},
        "iteration_trace": [
            {"iteration": 0, "action": "ActivatePage", "payload": {"page_index": 1}},
            {"iteration": 1, "action": "EvaluatorDecision"},
            {"iteration": 1, "action": "OpenNode", "payload": {"node_id": "n1", "page_index": 1}},
            {"iteration": 1, "action": "SearchEvidence"},
        ],
    }
    records = [
        {
            "question_id": "unanswerable_success",
            "question": "How many users?",
            "answer": "Not answerable",
            "pred": "Not answerable",
            "score": 1.0,
            "prepare_metadata": base_metadata | {"opened_node_ids": ["n1", "n2", "n3", "n4"]},
        },
        {
            "question_id": "answerable_success",
            "question": "Which restaurants are recommended?",
            "answer": ["Chen Mapo Doufu", "Yu Zhi Lan"],
            "pred": ["Chen Mapo Doufu", "Yu Zhi Lan"],
            "score": 1.0,
            "prepare_metadata": base_metadata,
        },
        {
            "question_id": "answerable_failure",
            "question": "What color is the line?",
            "answer": "red",
            "pred": "orange",
            "score": 0.0,
            "prepare_metadata": base_metadata,
        },
    ]

    rows = magerag_case_rows(records)

    assert rows[0]["question_id"] == "answerable_success"


def test_paper_case_study_artifacts_write_csv_tex_and_figure(tmp_path):
    jsonl_path = tmp_path / "run.jsonl"
    _write_jsonl(
        jsonl_path,
        [
            {
                "question_id": "q_success",
                "question": "Which restaurants are recommended?",
                "answer": ["Chen Mapo Doufu", "Yu Zhi Lan"],
                "pred": ["Chen Mapo Doufu", "Yu Zhi Lan"],
                "score": 1.0,
                "prepare_metadata": {
                    "stop_reason": "watchdog_iterations",
                    "active_node_ids": ["p1"],
                    "opened_node_ids": ["n1", "n2"],
                    "pruned_node_ids": ["n3"],
                    "reader_input": {"image_refs": [{"page_index": 1}], "content_part_count": 4},
                    "iteration_trace": [
                        {"iteration": 0, "action": "ActivatePage", "payload": {"page_index": 1}},
                        {"iteration": 1, "action": "EvaluatorDecision"},
                        {"iteration": 1, "action": "OpenNode", "payload": {"node_id": "n1", "page_index": 1}},
                    ],
                },
            },
            {
                "question_id": "q_failure",
                "question": "What color is the line?",
                "answer": "red",
                "pred": "orange",
                "score": 0.0,
                "prepare_metadata": {
                    "stop_reason": "watchdog_iterations",
                    "active_node_ids": ["p2"],
                    "opened_node_ids": ["n4"],
                    "pruned_node_ids": [],
                    "reader_input": {"image_refs": [{"page_index": 2}], "content_part_count": 3},
                    "iteration_trace": [
                        {"iteration": 0, "action": "ActivatePage", "payload": {"page_index": 2}},
                        {"iteration": 1, "action": "EvaluatorDecision"},
                        {"iteration": 1, "action": "OpenNode", "payload": {"node_id": "n4", "page_index": 2}},
                    ],
                },
            },
        ],
    )

    artifacts = write_case_study_artifacts(jsonl_path, tmp_path / "paper_cache", tmp_path / "images", tmp_path / "tables")

    assert artifacts.csv_path.exists()
    assert artifacts.figure_path.exists()
    assert artifacts.tex_path.exists()
    tex = artifacts.tex_path.read_text(encoding="utf-8")
    assert "\\label{fig:magerag_case_study}" in tex
    assert "images/magerag_case_study.pdf" in tex


def test_paper_artifact_pipeline_writes_manifest_and_all_artifacts(tmp_path):
    results_root = tmp_path / "results"
    _write_json(
        results_root / "longdocurl" / "image" / "res_model.metrics.json",
        {
            "overall_acc": 0.5,
            "fine_grained_metrics": {
                "Main_Task": {"Understanding": 0.6, "Reasoning": 0.4, "Locating": 0.3},
                "Evidence_Pages": {"Single_Page": 0.7, "Multi_Page": 0.2},
            },
            "sample_count": 10,
            "completed_count": 10,
            "failed_count": 0,
        },
    )
    _write_json(
        results_root / "mmlongbench" / "image" / "res_model.metrics.json",
        {
            "overall_acc": 0.45,
            "overall_f1": 0.4,
            "breakdowns": {
                "single_page": {"acc": 0.55},
                "cross_page": {"acc": 0.25},
                "unanswerable": {"acc": 0.75},
            },
            "sample_count": 12,
            "completed_count": 12,
            "failed_count": 0,
        },
    )
    budget_stem = (
        "res_top_k_3_mode_full_watchdog_iterations_5_"
        "max_selected_actions_per_iteration_3_mode_full_graph_Qwen3_VL_8B_Instruct"
    )
    _write_json(
        results_root / "mmlongbench" / "magerag" / f"{budget_stem}.metrics.json",
        {
            "overall_acc": 0.52,
            "overall_f1": 0.49,
            "breakdowns": {
                "single_page": {"acc": 0.60},
                "cross_page": {"acc": 0.31},
                "unanswerable": {"acc": 0.70},
            },
            "sample_count": 12,
            "completed_count": 12,
            "failed_count": 0,
        },
    )
    _write_json(
        results_root
        / "mmlongbench"
        / "magerag"
        / "res_top_k_8_mode_full_watchdog_iterations_5_max_selected_actions_per_iteration_3_mode_full_graph_Qwen3_VL_8B_Instruct.metrics.json",
        {"overall_acc": 0.0, "sample_count": 12, "completed_count": 0, "failed_count": 12},
    )
    trace_path = tmp_path / "trace.jsonl"
    _write_jsonl(
        trace_path,
        [
            {
                "question_id": "q_success",
                "question": "What is the value?",
                "answer": "0.45",
                "pred": "0.45",
                "score": 1.0,
                "prepare_metadata": {
                    "stop_reason": "done",
                    "active_node_ids": ["p1"],
                    "opened_node_ids": ["n1"],
                    "pruned_node_ids": [],
                    "reader_input": {"image_refs": [{"page_index": 1}], "content_part_count": 3},
                    "iteration_trace": [
                        {"iteration": 0, "action": "ActivatePage", "payload": {"page_index": 1}},
                        {"iteration": 1, "action": "EvaluatorDecision"},
                        {"iteration": 1, "action": "OpenNode", "payload": {"node_id": "n1", "page_index": 1}},
                    ],
                },
            },
            {
                "question_id": "q_failure",
                "question": "How many people?",
                "answer": "8",
                "pred": "12",
                "score": 0.0,
                "prepare_metadata": {
                    "stop_reason": "done",
                    "active_node_ids": ["p2"],
                    "opened_node_ids": ["n2"],
                    "pruned_node_ids": [],
                    "reader_input": {"image_refs": [{"page_index": 2}], "content_part_count": 3},
                    "iteration_trace": [
                        {"iteration": 0, "action": "ActivatePage", "payload": {"page_index": 2}},
                        {"iteration": 1, "action": "EvaluatorDecision"},
                        {"iteration": 1, "action": "OpenNode", "payload": {"node_id": "n2", "page_index": 2}},
                    ],
                },
            },
        ],
    )

    manifest = write_all_paper_artifacts(
        results_root=results_root,
        output_dir=tmp_path / "paper_cache",
        paper_image_dir=tmp_path / "images",
        paper_table_dir=tmp_path / "tables",
        trace_jsonl_path=trace_path,
        case_jsonl_path=trace_path,
        efficiency_jsonl_path=trace_path,
    )

    manifest_path = tmp_path / "paper_cache" / "paper_artifacts_manifest.json"
    assert manifest_path.exists()
    assert manifest["sources"]["results_root"] == str(results_root)
    assert manifest["sources"]["trace_jsonl_path"] == str(trace_path)
    assert manifest["run_diagnostics"]
    assert manifest["selected_runs"]
    assert manifest["warnings"]
    assert manifest["artifact_count"] == len(manifest["artifacts"])
    artifact_names = {artifact["name"] for artifact in manifest["artifacts"]}
    assert "main_results.csv_path" in artifact_names
    assert "breakdown.figure_path" in artifact_names
    assert "budget_ablation.budget_figure" in artifact_names
    assert "trace_statistics.summary_tex" in artifact_names
    assert "case_study.figure_path" in artifact_names
    assert all(Path(artifact["path"]).exists() for artifact in manifest["artifacts"])


def test_paper_artifact_pipeline_records_skipped_optional_inputs(tmp_path):
    manifest = write_all_paper_artifacts(
        results_root=tmp_path / "missing_results",
        output_dir=tmp_path / "paper_cache",
        paper_image_dir=tmp_path / "images",
        paper_table_dir=tmp_path / "tables",
        trace_jsonl_path=tmp_path / "missing_trace.jsonl",
        case_jsonl_path=tmp_path / "missing_case.jsonl",
        efficiency_jsonl_path=tmp_path / "missing_efficiency.jsonl",
    )

    skipped = {item["name"]: item["reason"] for item in manifest["skipped"]}
    assert skipped["trace_statistics"] == "missing input jsonl"
    assert skipped["case_study"] == "missing input jsonl"
    assert skipped["efficiency"] == "missing input jsonl"
    assert (tmp_path / "paper_cache" / "paper_artifacts_manifest.json").exists()


def test_paper_diagnostic_artifacts_write_pairwise_and_retrieval_csvs(tmp_path):
    results_root = tmp_path / "results"
    magerag = (
        results_root
        / "mmlongbench"
        / "magerag"
        / "res_top_k_3_mode_full_watchdog_iterations_5_max_selected_actions_per_iteration_3_mode_full_graph_Qwen3_VL_8B_Instruct"
    )
    bm25 = results_root / "mmlongbench" / "bm25" / "res_model"
    for path, acc in [(magerag, 0.5), (bm25, 0.25)]:
        _write_json(path.with_suffix(".metrics.json"), {"overall_acc": acc, "sample_count": 2, "completed_count": 2, "failed_count": 0})
    _write_jsonl(
        magerag.with_suffix(".jsonl"),
        [
            {
                "question_id": "q1",
                "score": 1.0,
                "evidence_pages": "[3]",
                "prepare_metadata": {
                    "retrieval": {
                        "retrieved_items": [{"rank": 1, "page_index": 2, "page_number": 3, "score": 0.9}],
                    }
                },
            },
            {"question_id": "q2", "score": 0.0, "evidence_pages": "[4]", "prepare_metadata": {"retrieval": {"retrieved_items": []}}},
        ],
    )
    _write_jsonl(
        bm25.with_suffix(".jsonl"),
        [
            {"question_id": "q1", "score": 0.0, "evidence_pages": "[3]", "prepare_metadata": {"retrieved_chunks": []}},
            {"question_id": "q2", "score": 0.0, "evidence_pages": "[4]", "prepare_metadata": {"retrieved_chunks": []}},
        ],
    )

    artifacts = write_paper_diagnostic_artifacts(scan_runs(results_root), tmp_path / "paper_cache")

    assert artifacts.retrieval_csv.exists()
    assert artifacts.pairwise_csv.exists()
    retrieval = pd.read_csv(artifacts.retrieval_csv)
    pairwise = pd.read_csv(artifacts.pairwise_csv)
    assert retrieval["run_id"].nunique() == 2
    magerag_hit = retrieval[(retrieval["baseline"] == "magerag") & (retrieval["question_id"] == "q1")].iloc[0]
    assert bool(magerag_hit["evidence_hit"]) is True
    assert pairwise.iloc[0]["baseline"] == "bm25"
    assert pairwise.iloc[0]["a_wins"] == 1
    assert pairwise.iloc[0]["aligned"] == 2


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
