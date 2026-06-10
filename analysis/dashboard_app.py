from __future__ import annotations

import json
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Any

import networkx as nx
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from analysis.plugins import get_plugin
from analysis.results_loader import RunRecord, read_jsonl_cached, scan_runs
from analysis.results_metrics import (
    aggregate_retrieval,
    build_leaderboard,
    case_rows,
    correction_rows,
    flatten_metrics,
    official_score,
    pairwise_comparison,
    parameter_curve_rows,
    retrieval_diagnostics,
)
from utils.image_crop import crop_image_to_normalized_bbox_1000

DEFAULT_RESULTS_ROOT = Path("results")
DEFAULT_CACHE_ROOT = Path("analysis_cache/result_analysis")
DEFAULT_RESULT_SUBDIRS = (
    Path("longdocurl/magerag"),
    Path("mmlongbench/magerag"),
    Path("mmlongdoc/magerag"),
)


st.set_page_config(page_title="Results Analysis", layout="wide")


@st.cache_data(show_spinner=False)
def load_runs(results_root: str) -> list[RunRecord]:
    return scan_runs(Path(results_root))


@st.cache_data(show_spinner=False)
def load_agent_trace_runs(results_root: str) -> list[RunRecord]:
    return scan_runs(Path(results_root), included_subdirs=DEFAULT_RESULT_SUBDIRS)


@st.cache_data(show_spinner=True)
def load_records(jsonl_path: str, cache_root: str, cache_namespace: str = "default") -> list[dict[str, Any]]:
    return read_jsonl_cached(Path(jsonl_path), Path(cache_root), cache_namespace=cache_namespace)


def _render_dataframe(df: pd.DataFrame) -> None:
    st.dataframe(_dataframe_for_streamlit(df), use_container_width=True, hide_index=True)


def _dataframe_for_streamlit(df: pd.DataFrame) -> pd.DataFrame:
    display_df = df.copy()
    for column in display_df.columns:
        if display_df[column].dtype == "object":
            display_df[column] = display_df[column].map(_cell_for_streamlit)
    return display_df


def _cell_for_streamlit(value: Any) -> Any:
    if value is None:
        return None
    if isinstance(value, float) and pd.isna(value):
        return value
    if isinstance(value, list | tuple | dict):
        return json.dumps(value, ensure_ascii=False, sort_keys=True, default=str)
    if isinstance(value, set):
        return json.dumps(sorted(value, key=str), ensure_ascii=False, default=str)
    if isinstance(value, str):
        return value
    try:
        if pd.isna(value):
            return None
    except (TypeError, ValueError):
        pass
    return str(value)


def main() -> None:
    st.title("Results Analysis Dashboard")
    with st.sidebar:
        results_root = st.text_input("Results root", value=str(DEFAULT_RESULTS_ROOT))
        cache_root = st.text_input("Cache root", value=str(DEFAULT_CACHE_ROOT))
        metric = st.selectbox("Primary metric", ["acc", "f1", "avg_acc"], index=0)
        refresh = st.button("Refresh inventory")
    if refresh:
        load_runs.clear()
        load_agent_trace_runs.clear()
        load_records.clear()

    runs = load_runs(results_root)
    if not runs:
        st.warning("No result files found.")
        return

    run_lookup = {run.run_id: run for run in runs}
    agent_trace_runs = load_agent_trace_runs(results_root)
    agent_trace_run_lookup = {run.run_id: run for run in agent_trace_runs}
    inventory_df = inventory_dataframe(runs, metric)
    tabs = st.tabs(
        [
            "Inventory",
            "Leaderboard",
            "Parameter Curves",
            "Breakdowns",
            "Retrieval",
            "Case Explorer",
            "Correction",
            "Agent Trace",
            "Pairwise",
        ]
    )

    with tabs[0]:
        render_inventory(inventory_df)
    with tabs[1]:
        render_leaderboard(runs, metric)
    with tabs[2]:
        render_parameter_curves(runs, metric)
    with tabs[3]:
        selected = run_selector(run_lookup, "breakdown_run", "Breakdown run")
        render_breakdowns(selected)
    with tabs[4]:
        selected = run_selector(run_lookup, "retrieval_run", "Retrieval run")
        records = records_for_run(selected, cache_root)
        render_retrieval(selected, records)
    with tabs[5]:
        selected = run_selector(run_lookup, "case_run", "Case run")
        records = records_for_run(selected, cache_root)
        render_case_explorer(records)
    with tabs[6]:
        selected = run_selector(run_lookup, "correction_run", "Correction run")
        records = records_for_run(selected, cache_root)
        render_correction(records)
    with tabs[7]:
        if not agent_trace_run_lookup:
            st.info("No MAGE-RAG agent trace result files found.")
        else:
            selected = run_selector(agent_trace_run_lookup, "agent_trace_run", "Agent trace run")
            records = records_for_run(selected, cache_root)
            render_agent_trace(selected, records)
    with tabs[8]:
        left, right = st.columns(2)
        with left:
            run_a = run_selector(run_lookup, "pairwise_a", "Run A")
        with right:
            run_b = run_selector(run_lookup, "pairwise_b", "Run B")
        records_a = records_for_run(run_a, cache_root)
        records_b = records_for_run(run_b, cache_root)
        render_pairwise(records_a, records_b)


def inventory_dataframe(runs: list[RunRecord], metric: str) -> pd.DataFrame:
    rows = []
    for run in runs:
        row = {
            "benchmark": run.benchmark,
            "baseline": run.baseline,
            "run_id": run.run_id,
            "jsonl": str(run.jsonl_path) if run.jsonl_path else None,
            "metrics": str(run.metrics_path) if run.metrics_path else None,
            "paired": run.jsonl_path is not None and run.metrics_path is not None,
            "jsonl_mb": _mb(run.jsonl_size_bytes),
            "metrics_kb": _kb(run.metrics_size_bytes),
            "sample_count": run.sample_count,
            "completed_count": run.completed_count,
            "failed_count": run.failed_count,
            "score": official_score(run.metrics, metric),
        }
        row.update(run.parameters)
        rows.append(row)
    return pd.DataFrame(rows)


def render_inventory(df: pd.DataFrame) -> None:
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Runs", len(df))
    col2.metric("Benchmarks", df["benchmark"].nunique())
    col3.metric("Baselines", df["baseline"].nunique())
    col4.metric("Paired", int(df["paired"].sum()))
    selected_benchmarks = st.multiselect("Benchmark", sorted(df["benchmark"].dropna().unique()))
    selected_baselines = st.multiselect("Baseline", sorted(df["baseline"].dropna().unique()))
    filtered = df.copy()
    if selected_benchmarks:
        filtered = filtered[filtered["benchmark"].isin(selected_benchmarks)]
    if selected_baselines:
        filtered = filtered[filtered["baseline"].isin(selected_baselines)]
    _render_dataframe(filtered)


def render_leaderboard(runs: list[RunRecord], metric: str) -> None:
    rows = build_leaderboard(runs, metric)
    if not rows:
        st.info("No official metric values available.")
        return
    df = pd.DataFrame(rows)
    st.plotly_chart(
        px.bar(df, x="baseline", y="score", color="benchmark", facet_col="benchmark"),
        use_container_width=True,
    )
    _render_dataframe(df)


def render_parameter_curves(runs: list[RunRecord], metric: str) -> None:
    df = pd.DataFrame(parameter_curve_rows(runs, metric))
    if df.empty:
        st.info("No parameterized metric rows available.")
        return
    benchmark = st.selectbox("Benchmark", sorted(df["benchmark"].unique()), key="curve_benchmark")
    df = df[df["benchmark"] == benchmark]
    specs = _chart_specs_from_rows(df)
    tab_labels = [spec["title"] for spec in specs] + ["All runs"]
    curve_tabs = st.tabs(tab_labels)
    for tab, spec in zip(curve_tabs, specs, strict=False):
        with tab:
            render_chart_spec(df, spec)
    with curve_tabs[-1]:
        _render_dataframe(df.drop(columns=["chart_specs"], errors="ignore"))


def render_chart_spec(df: pd.DataFrame, spec: dict[str, Any]) -> None:
    subset = df[df["chart_specs"].apply(lambda specs: spec in specs if isinstance(specs, list) else False)].copy()
    if subset.empty and spec.get("kind") == "scatter":
        subset = df.copy()
    for filter_name in spec.get("filters") or []:
        if filter_name not in subset.columns:
            continue
        values = sorted(v for v in subset[filter_name].dropna().unique())
        selected = st.selectbox(filter_name, values, key=f"chart_{spec['title']}_{filter_name}") if values else None
        if selected is not None:
            subset = subset[subset[filter_name] == selected]
    kind = spec.get("kind")
    if kind == "line":
        render_line(subset, spec.get("x"), spec.get("color") or "baseline")
        return
    if kind == "heatmap":
        render_heatmap(subset, spec)
        return
    if kind == "scatter":
        render_scatter(subset, spec.get("x"), spec.get("color") or "baseline")
        return
    _render_dataframe(subset.drop(columns=["chart_specs"], errors="ignore"))


def render_line(df: pd.DataFrame, x: str | None, color: str) -> None:
    if not x or x not in df.columns or df.empty:
        st.info(f"No {x or 'parameter'} rows available.")
        return
    plot_df = df.dropna(subset=[x, "score"]).sort_values(x)
    if plot_df.empty:
        st.info(f"No {x} rows available.")
        return
    st.plotly_chart(px.line(plot_df, x=x, y="score", color=color, markers=True), use_container_width=True)
    _render_dataframe(plot_df.drop(columns=["chart_specs"], errors="ignore"))


def render_scatter(df: pd.DataFrame, x: str | None, color: str) -> None:
    if not x or x not in df.columns or df.empty:
        st.info(f"No {x or 'parameter'} rows available.")
        return
    plot_df = df.dropna(subset=[x, "score"])
    if plot_df.empty:
        st.info(f"No {x} rows available.")
        return
    st.plotly_chart(px.scatter(plot_df, x=x, y="score", color=color, hover_data=["run_id"]), use_container_width=True)
    _render_dataframe(plot_df.drop(columns=["chart_specs"], errors="ignore"))


def render_heatmap(df: pd.DataFrame, spec: dict[str, Any]) -> None:
    row = spec.get("row")
    column = spec.get("column")
    if not row or not column or not {row, column, "score"}.issubset(df.columns) or df.empty:
        st.info("No heatmap parameter rows available.")
        return
    heatmap = df.pivot_table(values="score", index=row, columns=column, aggfunc="max")
    if heatmap.empty:
        st.info("No heatmap parameter rows available.")
        return
    st.plotly_chart(px.imshow(heatmap, aspect="auto", text_auto=".3f"), use_container_width=True)
    _render_dataframe(df.drop(columns=["chart_specs"], errors="ignore"))


def _chart_specs_from_rows(df: pd.DataFrame) -> list[dict[str, Any]]:
    specs: list[dict[str, Any]] = []
    seen: set[tuple[Any, ...]] = set()
    if "chart_specs" not in df.columns:
        return specs
    for row_specs in df["chart_specs"]:
        if not isinstance(row_specs, list):
            continue
        for spec in row_specs:
            key = (spec.get("kind"), spec.get("title"), spec.get("x"), spec.get("row"), spec.get("column"))
            if key not in seen:
                specs.append(spec)
                seen.add(key)
    if specs:
        return specs
    reserved = {"score", "sample_count", "completed_count", "failed_count"}
    metadata_cols = {"benchmark", "baseline", "run_id", "plugin_name", "chart_specs"}
    for column in df.columns:
        if column in reserved or column in metadata_cols:
            continue
        if pd.api.types.is_numeric_dtype(df[column]):
            specs.append({"kind": "scatter", "title": f"{column} vs score", "x": column, "y": "score", "color": "baseline", "filters": []})
    return specs


def render_breakdowns(run: RunRecord) -> None:
    rows = flatten_metrics(run.metrics)
    if not rows:
        st.info("No metrics breakdown available.")
        return
    df = pd.DataFrame(rows)
    categories = sorted(df["category"].dropna().unique())
    selected = st.multiselect("Breakdown group", categories, default=categories[:4])
    if selected:
        df = df[df["category"].isin(selected)]
    st.plotly_chart(px.bar(df, x="path", y="value", color="category"), use_container_width=True)
    _render_dataframe(df)


def render_retrieval(run: RunRecord, records: list[dict[str, Any]]) -> None:
    rows = retrieval_diagnostics(records)
    if not rows:
        st.info("No sample records loaded.")
        return
    df = pd.DataFrame(rows)
    summary = aggregate_retrieval(rows)
    col1, col2, col3 = st.columns(3)
    col1.metric("Samples", summary["count"])
    col2.metric("Evidence hit rate", _pct(summary["hit_rate"]))
    col3.metric("Avg first-hit rank", _fmt(summary["avg_first_hit_rank"]))
    if "evidence_hit" in df.columns:
        st.plotly_chart(px.histogram(df, x="score", color="evidence_hit", nbins=20), use_container_width=True)
    duration_cols = [col for col in df.columns if col.endswith("duration_seconds")]
    _render_dataframe(df[["question_id", "score", "evidence_hit", "first_hit_rank", "first_hit_score", *duration_cols]])
    plugin_rows = get_plugin(run.baseline).diagnostic_rows(records)
    if plugin_rows:
        st.subheader("Plugin diagnostics")
        _render_dataframe(pd.DataFrame(plugin_rows))


def render_case_explorer(records: list[dict[str, Any]]) -> None:
    rows = case_rows(records)
    if not rows:
        st.info("No sample records loaded.")
        return
    df = pd.DataFrame(rows)
    col1, col2, col3 = st.columns(3)
    with col1:
        buckets = st.multiselect("Score bucket", sorted(df["score_bucket"].dropna().unique()))
    with col2:
        hit_values = st.multiselect("Retrieval hit", sorted(df["evidence_hit"].dropna().unique())) if "evidence_hit" in df else []
    with col3:
        query = st.text_input("Search")
    filtered = df.copy()
    if buckets:
        filtered = filtered[filtered["score_bucket"].isin(buckets)]
    if hit_values:
        filtered = filtered[filtered["evidence_hit"].isin(hit_values)]
    if query:
        mask = filtered[["question_id", "question", "answer", "pred"]].fillna("").astype(str).agg(" ".join, axis=1).str.contains(query, case=False, regex=False)
        filtered = filtered[mask]
    _render_dataframe(filtered.drop(columns=["images", "metadata"], errors="ignore"))
    if filtered.empty:
        return
    selected_id = st.selectbox("Question", filtered["question_id"].astype(str).tolist())
    selected = next(row for row in rows if str(row.get("question_id")) == selected_id)
    left, right = st.columns([2, 1])
    with left:
        st.subheader("Question")
        st.write(selected.get("question"))
        st.subheader("Answer / Prediction")
        st.write({"answer": selected.get("answer"), "pred": selected.get("pred"), "score": selected.get("score")})
        st.subheader("Metadata")
        st.json(selected.get("metadata") or {})
    with right:
        st.subheader("Pages")
        st.write({"evidence_pages": selected.get("evidence_pages"), "retrieval_hit": selected.get("evidence_hit")})
        for image in selected.get("images", [])[:4]:
            image_path = Path(image)
            if image_path.exists():
                st.image(str(image_path), caption=image_path.name, use_container_width=True)


def render_correction(records: list[dict[str, Any]]) -> None:
    rows = correction_rows(records)
    if not rows:
        st.info("No sample records loaded.")
        return
    df = pd.DataFrame(rows)
    ran_count = int(df["correction_ran"].sum()) if "correction_ran" in df else 0
    applied_count = int(df["correction_applied"].sum()) if "correction_applied" in df else 0
    error_count = int(df.get("error", pd.Series(dtype=object)).notna().sum())
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Samples", len(df))
    col2.metric("Correction ran", ran_count)
    col3.metric("Applied", applied_count)
    col4.metric("Errors", error_count)
    summary_cols = st.columns(2)
    with summary_cols[0]:
        outcome_df = df["correction_outcome"].value_counts().reset_index()
        outcome_df.columns = ["outcome", "count"]
        st.plotly_chart(px.bar(outcome_df, x="outcome", y="count", title="Correction outcomes"), use_container_width=True)
    with summary_cols[1]:
        score_df = df.dropna(subset=["score_delta"]) if "score_delta" in df else pd.DataFrame()
        if score_df.empty:
            st.info("No score delta values available.")
        else:
            st.plotly_chart(px.histogram(score_df, x="score_delta", nbins=30, title="Score delta"), use_container_width=True)
    filters = st.columns(3)
    with filters[0]:
        outcomes = st.multiselect("Outcome", sorted(df["correction_outcome"].dropna().unique()), key="correction_outcome")
    with filters[1]:
        applied = st.multiselect("Applied", [False, True], key="correction_applied")
    with filters[2]:
        query = st.text_input("Search", key="correction_search")
    filtered = df.copy()
    if outcomes:
        filtered = filtered[filtered["correction_outcome"].isin(outcomes)]
    if applied:
        filtered = filtered[filtered["correction_applied"].isin(applied)]
    if query:
        mask = filtered[["question_id", "question", "answer", "initial_pred", "corrected_pred", "final_pred"]].fillna("").astype(str).agg(" ".join, axis=1).str.contains(query, case=False, regex=False)
        filtered = filtered[mask]
    visible_cols = [
        "question_id",
        "correction_outcome",
        "correction_applied",
        "initial_score",
        "corrected_score",
        "score_delta",
        "pred_changed",
        "error",
        "question",
        "answer",
        "initial_pred",
        "corrected_pred",
        "final_pred",
    ]
    _render_dataframe(filtered[[col for col in visible_cols if col in filtered.columns]])
    if filtered.empty:
        return
    selected_id = st.selectbox("Question", filtered["question_id"].astype(str).tolist(), key="correction_question")
    row = next(row for row in rows if str(row.get("question_id")) == selected_id)
    left, right = st.columns(2)
    with left:
        st.subheader("Before correction")
        st.write({"pred": row.get("initial_pred"), "format": row.get("initial_pred_format"), "score": row.get("initial_score")})
    with right:
        st.subheader("After correction")
        st.write({"pred": row.get("corrected_pred"), "format": row.get("corrected_pred_format"), "score": row.get("corrected_score")})
    st.subheader("Gold / final")
    st.write({"answer": row.get("answer"), "final_pred": row.get("final_pred"), "final_score": row.get("final_score")})
    if row.get("corrected_extracted_res"):
        st.subheader("Correction raw output")
        render_xml_or_code(str(row["corrected_extracted_res"]), key=f"correction_raw_{selected_id}")
    with st.expander("Correction input messages"):
        st.json(row.get("input_messages") or [])


def render_agent_trace(run: RunRecord, records: list[dict[str, Any]]) -> None:
    plugin = get_plugin(run.baseline)
    if not plugin.has_case_visualization():
        st.info(f"Baseline {run.baseline} does not provide an agent trace visualization.")
        return
    rows = case_rows(records)
    if not rows:
        st.info("No sample records loaded.")
        return
    df = pd.DataFrame(rows)
    metadata_df = _agent_trace_index(records)
    if not metadata_df.empty:
        df = df.merge(metadata_df, on="question_id", how="left")

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Samples", len(df))
    col2.metric("Avg trace steps", _fmt(df["trace_steps"].dropna().mean() if "trace_steps" in df else None))
    col3.metric("Avg opened nodes", _fmt(df["opened_nodes"].dropna().mean() if "opened_nodes" in df else None))
    col4.metric("Validation errors", int(df["validation_errors"].fillna(0).sum()) if "validation_errors" in df else 0)

    filter_cols = st.columns(4)
    with filter_cols[0]:
        buckets = st.multiselect("Score bucket", sorted(df["score_bucket"].dropna().unique()), key="agent_score_bucket")
    with filter_cols[1]:
        stop_values = sorted(v for v in df.get("stop_reason", pd.Series(dtype=object)).dropna().unique())
        stops = st.multiselect("Stop reason", stop_values, key="agent_stop_reason")
    with filter_cols[2]:
        min_steps = int(df["trace_steps"].dropna().min()) if "trace_steps" in df and not df["trace_steps"].dropna().empty else 0
        max_steps = int(df["trace_steps"].dropna().max()) if "trace_steps" in df and not df["trace_steps"].dropna().empty else 0
        step_range = st.slider("Trace steps", min_steps, max_steps, (min_steps, max_steps), key="agent_trace_steps") if max_steps > min_steps else (min_steps, max_steps)
    with filter_cols[3]:
        query = st.text_input("Search", key="agent_trace_search")

    filtered = df.copy()
    if buckets:
        filtered = filtered[filtered["score_bucket"].isin(buckets)]
    if stops and "stop_reason" in filtered:
        filtered = filtered[filtered["stop_reason"].isin(stops)]
    if "trace_steps" in filtered:
        filtered = filtered[(filtered["trace_steps"].fillna(0) >= step_range[0]) & (filtered["trace_steps"].fillna(0) <= step_range[1])]
    if query:
        mask = filtered[["question_id", "question", "answer", "pred"]].fillna("").astype(str).agg(" ".join, axis=1).str.contains(query, case=False, regex=False)
        filtered = filtered[mask]

    visible_cols = [
        "question_id",
        "score",
        "score_bucket",
        "stop_reason",
        "trace_steps",
        "opened_nodes",
        "active_nodes",
        "pruned_nodes",
        "validation_errors",
        "question",
        "answer",
        "pred",
    ]
    _render_dataframe(filtered[[col for col in visible_cols if col in filtered.columns]])
    if filtered.empty:
        return

    selected_id = st.selectbox("Question", filtered["question_id"].astype(str).tolist(), key="agent_trace_question")
    selected_record = next(record for record in records if str(record.get("question_id")) == selected_id)
    data = plugin.case_visualization(selected_record)
    render_agent_trace_case(selected_record, data)


def render_agent_trace_case(record: dict[str, Any], data: dict[str, Any]) -> None:
    summary = data.get("summary") or {}
    st.subheader("Sample Diagnostics")
    left, right = st.columns([2, 1])
    with left:
        st.write(record.get("question"))
        st.write({"answer": record.get("answer"), "pred": record.get("pred")})
    with right:
        st.write({
            "score": summary.get("score"),
            "stop_reason": summary.get("stop_reason"),
            "duration_seconds": summary.get("duration_seconds"),
            "evaluator_model": summary.get("evaluator_model_name"),
        })

    metric_cols = st.columns(6)
    metric_cols[0].metric("Trace steps", summary.get("trace_steps") or 0)
    metric_cols[1].metric("Opened", summary.get("opened_nodes") or 0)
    metric_cols[2].metric("Active", summary.get("active_nodes") or 0)
    metric_cols[3].metric("Pruned", summary.get("pruned_nodes") or 0)
    metric_cols[4].metric("Errors", summary.get("validation_errors") or 0)
    metric_cols[5].metric("Graph", "yes" if data.get("graph_available") else "no")

    chart_col, page_col = st.columns([1, 2])
    with chart_col:
        action_df = pd.DataFrame(data.get("action_counts") or [])
        if not action_df.empty:
            st.plotly_chart(px.bar(action_df, x="action", y="count", title="Action distribution"), use_container_width=True)
        else:
            st.info("No action trace available.")
    with page_col:
        render_page_board(data.get("page_rows") or [])

    detail_tabs = st.tabs(["Page Details", "Trace Steps", "Evaluator I/O", "Reader I/O", "Graph Expansion", "Images", "Artifacts"])
    with detail_tabs[0]:
        render_page_details(data)
    with detail_tabs[1]:
        render_trace_steps(data)
    with detail_tabs[2]:
        render_evaluator_io(data)
    with detail_tabs[3]:
        render_reader_io(data)
    with detail_tabs[4]:
        render_graph_expansion(data)
    with detail_tabs[5]:
        render_agent_images(data)
    with detail_tabs[6]:
        if data.get("validation_errors"):
            st.subheader("Validation errors")
            st.json(data["validation_errors"])
        if data.get("summary_artifacts"):
            st.subheader("Summary artifacts")
            st.json(data["summary_artifacts"])


def render_page_board(page_rows: list[dict[str, Any]]) -> None:
    if not page_rows:
        st.info("No page-level evidence state available.")
        return
    df = pd.DataFrame(page_rows)
    hover_cols = [
        "page_number",
        "dominant_state",
        "opened_nodes",
        "active_nodes",
        "pruned_nodes",
        "action_count",
        "retrieval_rank",
        "retrieval_score",
        "is_evidence_page",
    ]
    fig = px.scatter(
        df,
        x="page_number",
        y="dominant_state",
        size=df["action_count"].clip(lower=1),
        color="dominant_state",
        symbol="is_evidence_page",
        hover_data=[col for col in hover_cols if col in df.columns],
        title="Page board",
        color_discrete_map={
            "Opened": "#0891b2",
            "Active": "#2563eb",
            "Pruned": "#dc2626",
            "Inactive": "#94a3b8",
        },
    )
    fig.update_layout(yaxis_title="Strongest final state", xaxis_title="Page number")
    st.plotly_chart(fig, use_container_width=True)


def render_page_details(data: dict[str, Any]) -> None:
    pages = data.get("page_rows") or []
    if not pages:
        st.info("No pages available.")
        return
    page_numbers = [int(page["page_number"]) for page in pages]
    selected_number = st.selectbox("Page", page_numbers, key="agent_page_detail")
    page = next(page for page in pages if int(page["page_number"]) == selected_number)
    left, right = st.columns([1, 2])
    with left:
        image_path = page.get("page_image_path")
        if image_path and Path(image_path).exists():
            st.image(str(image_path), caption=f"Page {selected_number}", use_container_width=True)
        else:
            st.info("No page image path available.")
    with right:
        st.write(page)
        node_df = pd.DataFrame(data.get("node_rows") or [])
        if not node_df.empty and "page_index" in node_df:
            page_nodes = node_df[node_df["page_index"] == page.get("page_index")]
            _render_dataframe(page_nodes)


def render_trace_steps(data: dict[str, Any]) -> None:
    trace_df = pd.DataFrame(data.get("trace_rows") or [])
    if trace_df.empty:
        st.info("No trace rows available.")
        return
    actions = st.multiselect("Action", sorted(trace_df["action"].dropna().unique()), key="trace_action_filter")
    if actions:
        trace_df = trace_df[trace_df["action"].isin(actions)]
    display_cols = ["step_index", "iteration", "action", "ok", "page_number", "node_id", "state_delta", "message"]
    _render_dataframe(trace_df[[col for col in display_cols if col in trace_df.columns]])
    if trace_df.empty:
        return
    selected_step = st.selectbox("Step", trace_df["step_index"].astype(int).tolist(), key="agent_trace_step")
    step = next(row for row in data.get("trace_rows") or [] if int(row.get("step_index")) == int(selected_step))
    st.json(step)


def render_evaluator_io(data: dict[str, Any]) -> None:
    evaluator_rows = data.get("evaluator_rows") or [
        row for row in data.get("trace_rows") or []
        if row.get("action") == "EvaluatorDecision" or row.get("evaluator_input") or row.get("raw_response")
    ]
    if not evaluator_rows:
        st.info("No evaluator decision trace recorded for this sample.")
        return
    selected_index = st.selectbox(
        "Evaluator iteration",
        list(range(len(evaluator_rows))),
        format_func=lambda index: evaluator_rows[int(index)].get("iteration_label") or f"Iteration {evaluator_rows[int(index)].get('iteration')}",
        key="agent_evaluator_iteration",
    )
    row = evaluator_rows[int(selected_index)]
    selected_step = row.get("step_index")
    evaluator_input = row.get("evaluator_input") or {}
    if not evaluator_input:
        evaluator_input = row
    if evaluator_input.get("prompt_text"):
        st.subheader("Full evaluator prompt")
        render_xml_or_code(evaluator_input["prompt_text"], key=f"evaluator_prompt_{selected_step}")
    if evaluator_input.get("context_xml"):
        st.subheader("Context XML")
        render_xml_or_code(evaluator_input["context_xml"], key=f"evaluator_context_{selected_step}")
    if evaluator_input.get("candidate_actions"):
        st.subheader("Candidate actions")
        _render_dataframe(pd.DataFrame(evaluator_input["candidate_actions"]))
    if evaluator_input.get("opened_image_refs"):
        st.subheader("Opened image refs")
        _render_dataframe(pd.DataFrame(evaluator_input["opened_image_refs"]))
    if row.get("decision"):
        st.subheader("Parsed decision")
        st.json(row["decision"])
    if row.get("raw_response"):
        st.subheader("Raw response")
        render_xml_or_code(str(row["raw_response"]), key=f"evaluator_raw_{selected_step}")


def render_reader_io(data: dict[str, Any]) -> None:
    reader_input = data.get("reader_input") or {}
    if not reader_input:
        st.info("No reader input trace recorded for this sample.")
    else:
        st.write({
            "content_part_count": reader_input.get("content_part_count"),
            "text_parts": len(reader_input.get("text_parts") or []),
            "image_refs": len(reader_input.get("image_refs") or []),
        })
        render_reader_content_parts(reader_input)
        with st.expander("Sanitized messages"):
            st.json(reader_input.get("messages") or [])
    st.subheader("Reader output")
    output = data.get("reader_output")
    raw_output = data.get("reader_raw_output")
    think_output = data.get("reader_think_output")
    parsed_tab, think_tab, raw_tab = st.tabs(["Parsed answer", "Think", "Raw output"])
    with parsed_tab:
        if output:
            st.code(str(output), language="text")
        else:
            st.info("No parsed reader output recorded for this sample.")
    with think_tab:
        if think_output:
            st.code(str(think_output), language="text")
        elif raw_output:
            st.info("Raw reader output is recorded, but it does not contain a <think> block.")
        else:
            st.info("No raw reader output was recorded for this sample, so <think> cannot be displayed.")
    with raw_tab:
        if raw_output:
            render_xml_or_code(str(raw_output), key="reader_raw_output")
        else:
            st.info("No raw reader output recorded for this sample.")


def render_reader_content_parts(reader_input: dict[str, Any]) -> None:
    messages = reader_input.get("messages") or []
    parts = []
    for message in messages:
        if not isinstance(message, dict):
            continue
        content = message.get("content")
        if isinstance(content, list):
            parts.extend(part for part in content if isinstance(part, dict))
    if not parts:
        text_parts = reader_input.get("text_parts") or []
        image_refs = reader_input.get("image_refs") or []
        for text in text_parts:
            parts.append({"type": "text", "text": text})
        for ref in image_refs:
            parts.append({"type": "image_ref", **ref})
    if not parts:
        return
    image_refs = iter(reader_input.get("image_refs") or [])
    for index, part in enumerate(parts, start=1):
        part_type = part.get("type")
        if part_type == "text":
            st.subheader(f"Input part {index}: text")
            render_xml_or_code(str(part.get("text") or ""), key=f"reader_text_{index}")
            continue
        ref = _reader_part_image_ref(part, image_refs)
        if ref:
            st.subheader(f"Input part {index}: image")
            render_image_ref(ref)


def _reader_part_image_ref(part: dict[str, Any], image_refs: Any) -> dict[str, Any] | None:
    if part.get("image_path"):
        return part
    if part.get("type") in {"image", "image_url", "image_ref"}:
        try:
            ref = next(image_refs)
        except StopIteration:
            ref = {}
        return ref if isinstance(ref, dict) else {}
    return None


def render_image_ref(ref: dict[str, Any]) -> None:
    image_path = ref.get("image_path")
    if not image_path:
        st.json(ref)
        return
    path = Path(str(image_path))
    caption = _image_ref_label(ref)
    if path.exists():
        st.image(_image_ref_display_image(ref), caption=caption, use_container_width=True)
    else:
        st.info(f"Missing image path: {path}")
    st.caption(_image_ref_source_caption(ref))


def _image_ref_display_image(ref: dict[str, Any]) -> Any:
    image_path = ref.get("image_path")
    if not image_path:
        return None
    path = Path(str(image_path))
    if ref.get("kind") != "opened_node_crop":
        return str(path)
    bbox = ref.get("bbox")
    if not isinstance(bbox, list | tuple) or len(bbox) != 4:
        return str(path)
    try:
        from PIL import Image

        with Image.open(path) as image:
            crop = crop_image_to_normalized_bbox_1000(image, bbox)
            if crop is None:
                return str(path)
            return crop.copy()
    except (OSError, TypeError, ValueError):
        return str(path)


def _image_ref_source_caption(ref: dict[str, Any]) -> str:
    image_path = str(ref.get("image_path") or "")
    if ref.get("kind") == "opened_node_crop" and ref.get("bbox") is not None:
        return f"{image_path} | bbox {ref.get('bbox')}"
    return image_path


def render_graph_expansion(data: dict[str, Any]) -> None:
    snapshots = data.get("graph_snapshots") or []
    if snapshots:
        selected_snapshot = st.selectbox(
            "Graph iteration",
            list(range(len(snapshots))),
            format_func=lambda index: snapshots[int(index)].get("iteration_label") or f"Iteration {snapshots[int(index)].get('iteration')}",
            key="expansion_graph_iteration",
        )
        render_graph_snapshot(snapshots[int(selected_snapshot)])
    else:
        st.info("No graph snapshot available for this sample.")

    rows = data.get("expansion_rows") or []
    if not rows:
        st.info("No graph expansion trace available.")
        return
    df = pd.DataFrame(rows)
    actions = st.multiselect("Expansion action", sorted(df["action"].dropna().unique()), key="expansion_action_filter")
    if actions:
        df = df[df["action"].isin(actions)]
    display_cols = [
        "step_index",
        "iteration",
        "action",
        "ok",
        "page_number",
        "node_id",
        "selected_candidate_index",
        "selected_candidate_type",
        "active_nodes",
        "opened_nodes",
        "pruned_nodes",
        "message",
    ]
    _render_dataframe(df[[col for col in display_cols if col in df.columns]])
    if df.empty:
        return
    selected_step = st.selectbox("Expansion step", df["step_index"].astype(int).tolist(), key="agent_expansion_step")
    row = next(row for row in rows if int(row.get("step_index")) == int(selected_step))
    st.json(row)


def render_graph_snapshot(snapshot: dict[str, Any]) -> None:
    nodes = snapshot.get("nodes") or []
    edges = snapshot.get("edges") or []
    if not nodes:
        st.info("No graph nodes available for this iteration.")
        return
    graph = nx.Graph()
    for node in nodes:
        graph.add_node(node["node_id"])
    for edge in edges:
        if edge.get("source") and edge.get("target"):
            graph.add_edge(edge["source"], edge["target"])
    if graph.number_of_nodes() == 1:
        only_node = next(iter(graph.nodes))
        positions = {only_node: (0.0, 0.0)}
    else:
        positions = nx.spring_layout(graph, seed=7, k=0.8)
    node_lookup = {node["node_id"]: node for node in nodes}
    edge_traces = []
    for edge_type in sorted({edge.get("edge_type") or "edge" for edge in edges}):
        edge_x = []
        edge_y = []
        hover = []
        for edge in edges:
            if (edge.get("edge_type") or "edge") != edge_type:
                continue
            source = edge.get("source")
            target = edge.get("target")
            if source not in positions or target not in positions:
                continue
            x0, y0 = positions[source]
            x1, y1 = positions[target]
            edge_x.extend([x0, x1, None])
            edge_y.extend([y0, y1, None])
            hover.append(f"{source} -> {target}<br>{edge_type} {edge.get('relation') or ''}")
        edge_traces.append(
            go.Scatter(
                x=edge_x,
                y=edge_y,
                mode="lines",
                line={"width": 1.5, "color": _edge_color(edge_type), "dash": _edge_dash(edge_type)},
                hoverinfo="skip",
                name=edge_type,
            )
        )
    node_x = []
    node_y = []
    colors = []
    symbols = []
    sizes = []
    labels = []
    hover_text = []
    for node_id, (x, y) in positions.items():
        node = node_lookup.get(node_id, {"node_id": node_id})
        node_x.append(x)
        node_y.append(y)
        state = str(node.get("state") or "Inactive")
        colors.append(_state_color(state))
        symbols.append("square" if node.get("is_page") else "circle")
        sizes.append(18 if node.get("highlight") else 12)
        labels.append(_short_node_label(node_id))
        hover_text.append(
            "<br>".join(
                [
                    str(node_id),
                    f"state: {state}",
                    f"type: {node.get('type') or 'unknown'}",
                    f"page: {node.get('page_number') or 'n/a'}",
                    str(node.get("preview") or ""),
                ]
            )
        )
    node_trace = go.Scatter(
        x=node_x,
        y=node_y,
        mode="markers+text",
        text=labels,
        textposition="top center",
        marker={"color": colors, "symbol": symbols, "size": sizes, "line": {"width": 1, "color": "#111827"}},
        hovertext=hover_text,
        hoverinfo="text",
        name="nodes",
    )
    fig = go.Figure(data=[*edge_traces, node_trace])
    fig.update_layout(
        title=snapshot.get("iteration_label") or "Graph snapshot",
        showlegend=True,
        height=620,
        margin={"l": 10, "r": 10, "t": 48, "b": 10},
        xaxis={"visible": False},
        yaxis={"visible": False},
    )
    st.plotly_chart(fig, use_container_width=True)


def _state_color(state: str) -> str:
    return {
        "Opened": "#0891b2",
        "Active": "#2563eb",
        "Pruned": "#dc2626",
        "Inactive": "#94a3b8",
    }.get(state, "#94a3b8")


def _edge_color(edge_type: str) -> str:
    return {
        "containment": "#64748b",
        "reading_order": "#16a34a",
        "semantic": "#f59e0b",
    }.get(edge_type, "#9ca3af")


def _edge_dash(edge_type: str) -> str:
    return {"semantic": "dot", "reading_order": "dash"}.get(edge_type, "solid")


def _short_node_label(node_id: str) -> str:
    parts = str(node_id).split(":")
    if len(parts) >= 4 and "page" in parts:
        page_index = parts.index("page")
        page = parts[page_index + 1] if page_index + 1 < len(parts) else "?"
        tail = parts[-1]
        return f"p{page}:{tail}"
    return str(node_id)[-18:]


def render_agent_images(data: dict[str, Any]) -> None:
    refs = []
    refs.extend(data.get("reader_image_refs") or [])
    for row in data.get("evaluator_rows") or []:
        refs.extend(row.get("opened_image_refs") or [])
    for row in data.get("page_rows") or []:
        if row.get("page_image_path"):
            refs.append({
                "kind": "page_detail",
                "page_index": row.get("page_index"),
                "page_number": row.get("page_number"),
                "image_path": row.get("page_image_path"),
            })
    for row in data.get("node_rows") or []:
        if row.get("image_path"):
            refs.append({
                "kind": "node",
                "node_id": row.get("node_id"),
                "page_index": row.get("page_index"),
                "page_number": row.get("page_number"),
                "image_path": row.get("image_path"),
            })
    deduped = []
    seen = set()
    for ref in refs:
        path = ref.get("image_path")
        key = (ref.get("kind"), ref.get("node_id"), ref.get("page_index"), path)
        if not path or key in seen:
            continue
        seen.add(key)
        deduped.append(ref)
    if not deduped:
        st.info("No image refs available.")
        return
    _render_dataframe(pd.DataFrame(deduped))
    selected = st.selectbox(
        "Image",
        list(range(len(deduped))),
        format_func=lambda index: _image_ref_label(deduped[int(index)]),
        key="agent_image_ref",
    )
    ref = deduped[int(selected)]
    image_path = Path(str(ref.get("image_path")))
    if image_path.exists():
        st.image(str(image_path), caption=str(image_path), use_container_width=True)
    else:
        st.info(f"Missing image path: {image_path}")


def render_xml_or_code(text: str, key: str) -> None:
    view = st.radio("View", ["Structured", "Raw"], horizontal=True, key=f"{key}_view")
    if view == "Raw":
        st.code(text, language="xml")
        return
    try:
        root = ET.fromstring(_extract_xml_document(text))
    except ET.ParseError:
        st.code(text, language="xml")
        return
    render_xml_node(root, key=key)


def render_xml_node(node: ET.Element, key: str, depth: int = 0) -> None:
    text = " ".join((node.text or "").split())
    label = node.tag
    if node.attrib:
        attrs = " ".join(f'{name}="{value}"' for name, value in node.attrib.items())
        label = f"{label} {attrs}"
    if text:
        label = f"{label}: {text[:80]}"
    children = list(node)
    if children:
        with st.expander(label, expanded=depth < 2):
            if node.attrib:
                st.json(dict(node.attrib))
            if text:
                st.write(text)
            for index, child in enumerate(children):
                render_xml_node(child, key=f"{key}_{depth}_{index}", depth=depth + 1)
        return
    st.write(label)


def _extract_xml_document(text: str) -> str:
    value = str(text or "").strip()
    start = _first_xml_tag_index(value, ["correction_prompt", "evaluator_prompt", "agent_step_context", "think", "answer", "agent_decision"])
    if start is not None:
        value = value[start:]
    top_level_tags = [
        tag
        for tag in ("correction_prompt", "evaluator_prompt", "agent_step_context", "think", "answer", "agent_decision")
        if f"<{tag}" in value
    ]
    if len(top_level_tags) > 1:
        return f"<analysis_xml>{value}</analysis_xml>"
    return value


def _first_xml_tag_index(text: str, tags: list[str]) -> int | None:
    positions = [text.find(f"<{tag}") for tag in tags]
    positions = [position for position in positions if position >= 0]
    return min(positions) if positions else None


def _image_ref_label(ref: dict[str, Any]) -> str:
    pieces = [str(ref.get("kind") or "image")]
    if ref.get("page_index") is not None:
        pieces.append(f"page_index {ref['page_index']}")
    if ref.get("node_id"):
        pieces.append(str(ref["node_id"]))
    return " | ".join(pieces)


def _agent_trace_index(records: list[dict[str, Any]]) -> pd.DataFrame:
    rows = []
    for record in records:
        metadata = record.get("prepare_metadata") or {}
        if not isinstance(metadata, dict):
            metadata = {}
        trace = metadata.get("iteration_trace")
        trace = trace if isinstance(trace, list) else []
        final_states = metadata.get("final_node_states")
        final_states = final_states if isinstance(final_states, dict) else {}
        state_values = list(final_states.values())
        rows.append(
            {
                "question_id": record.get("question_id"),
                "stop_reason": metadata.get("stop_reason"),
                "trace_steps": len(trace),
                "opened_nodes": state_values.count("Opened") or len(metadata.get("opened_node_ids") or []),
                "active_nodes": state_values.count("Active") or len(metadata.get("active_node_ids") or []),
                "pruned_nodes": state_values.count("Pruned") or len(metadata.get("pruned_node_ids") or []),
                "validation_errors": len(metadata.get("validation_errors") or []),
            }
        )
    return pd.DataFrame(rows)


def render_pairwise(records_a: list[dict[str, Any]], records_b: list[dict[str, Any]]) -> None:
    if not records_a or not records_b:
        st.info("Select two runs with JSONL records.")
        return
    comparison = pairwise_comparison(records_a, records_b)
    summary = comparison["summary"]
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Aligned", summary["aligned"])
    col2.metric("A wins", summary["a_wins"])
    col3.metric("B wins", summary["b_wins"])
    col4.metric("Ties", summary["ties"])
    df = pd.DataFrame(comparison["rows"])
    if df.empty:
        return
    outcome = st.multiselect("Outcome", sorted(df["outcome"].unique()))
    if outcome:
        df = df[df["outcome"].isin(outcome)]
    st.plotly_chart(px.histogram(df, x="delta", color="outcome", nbins=30), use_container_width=True)
    _render_dataframe(df)


def run_selector(run_lookup: dict[str, RunRecord], key: str, label: str) -> RunRecord:
    run_ids = sorted(run_lookup)
    selected = st.selectbox(label, run_ids, key=key)
    return run_lookup[selected]


def records_for_run(run: RunRecord, cache_root: str) -> list[dict[str, Any]]:
    if run.jsonl_path is None:
        return []
    plugin = get_plugin(run.baseline)
    return load_records(str(run.jsonl_path), cache_root, f"{plugin.name}:{plugin.version}")


def _mb(value: int | None) -> float | None:
    return round(value / 1024 / 1024, 3) if value is not None else None


def _kb(value: int | None) -> float | None:
    return round(value / 1024, 3) if value is not None else None


def _pct(value: float | None) -> str:
    return "n/a" if value is None else f"{value:.1%}"


def _fmt(value: float | None) -> str:
    return "n/a" if value is None else f"{value:.2f}"


if __name__ == "__main__":
    main()
