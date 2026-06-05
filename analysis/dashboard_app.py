from __future__ import annotations

import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Any

import pandas as pd
import plotly.express as px
import streamlit as st

from analysis.plugins import get_plugin
from analysis.results_loader import RunRecord, read_jsonl_cached, scan_runs
from analysis.results_metrics import (
    aggregate_retrieval,
    build_leaderboard,
    case_rows,
    flatten_metrics,
    official_score,
    pairwise_comparison,
    parameter_curve_rows,
    retrieval_diagnostics,
)

DEFAULT_RESULTS_ROOT = Path("results")
DEFAULT_CACHE_ROOT = Path("analysis_cache/result_analysis")


st.set_page_config(page_title="Results Analysis", layout="wide")


@st.cache_data(show_spinner=False)
def load_runs(results_root: str) -> list[RunRecord]:
    return scan_runs(Path(results_root))


@st.cache_data(show_spinner=True)
def load_records(jsonl_path: str, cache_root: str, cache_namespace: str = "default") -> list[dict[str, Any]]:
    return read_jsonl_cached(Path(jsonl_path), Path(cache_root), cache_namespace=cache_namespace)


def main() -> None:
    st.title("Results Analysis Dashboard")
    with st.sidebar:
        results_root = st.text_input("Results root", value=str(DEFAULT_RESULTS_ROOT))
        cache_root = st.text_input("Cache root", value=str(DEFAULT_CACHE_ROOT))
        metric = st.selectbox("Primary metric", ["acc", "f1", "avg_acc"], index=0)
        refresh = st.button("Refresh inventory")
    if refresh:
        load_runs.clear()
        load_records.clear()

    runs = load_runs(results_root)
    if not runs:
        st.warning("No result files found.")
        return

    run_lookup = {run.run_id: run for run in runs}
    inventory_df = inventory_dataframe(runs, metric)
    tabs = st.tabs(
        [
            "Inventory",
            "Leaderboard",
            "Parameter Curves",
            "Breakdowns",
            "Retrieval",
            "Case Explorer",
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
        selected = run_selector(run_lookup, "agent_trace_run", "Agent trace run")
        records = records_for_run(selected, cache_root)
        render_agent_trace(selected, records)
    with tabs[7]:
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
    st.dataframe(filtered, use_container_width=True, hide_index=True)


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
    st.dataframe(df, use_container_width=True, hide_index=True)


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
        st.dataframe(df.drop(columns=["chart_specs"], errors="ignore"), use_container_width=True, hide_index=True)


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
    st.dataframe(subset.drop(columns=["chart_specs"], errors="ignore"), use_container_width=True, hide_index=True)


def render_line(df: pd.DataFrame, x: str | None, color: str) -> None:
    if not x or x not in df.columns or df.empty:
        st.info(f"No {x or 'parameter'} rows available.")
        return
    plot_df = df.dropna(subset=[x, "score"]).sort_values(x)
    if plot_df.empty:
        st.info(f"No {x} rows available.")
        return
    st.plotly_chart(px.line(plot_df, x=x, y="score", color=color, markers=True), use_container_width=True)
    st.dataframe(plot_df.drop(columns=["chart_specs"], errors="ignore"), use_container_width=True, hide_index=True)


def render_scatter(df: pd.DataFrame, x: str | None, color: str) -> None:
    if not x or x not in df.columns or df.empty:
        st.info(f"No {x or 'parameter'} rows available.")
        return
    plot_df = df.dropna(subset=[x, "score"])
    if plot_df.empty:
        st.info(f"No {x} rows available.")
        return
    st.plotly_chart(px.scatter(plot_df, x=x, y="score", color=color, hover_data=["run_id"]), use_container_width=True)
    st.dataframe(plot_df.drop(columns=["chart_specs"], errors="ignore"), use_container_width=True, hide_index=True)


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
    st.dataframe(df.drop(columns=["chart_specs"], errors="ignore"), use_container_width=True, hide_index=True)


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
    st.dataframe(df, use_container_width=True, hide_index=True)


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
    st.dataframe(df[["question_id", "score", "evidence_hit", "first_hit_rank", "first_hit_score", *duration_cols]], use_container_width=True, hide_index=True)
    plugin_rows = get_plugin(run.baseline).diagnostic_rows(records)
    if plugin_rows:
        st.subheader("Plugin diagnostics")
        st.dataframe(pd.DataFrame(plugin_rows), use_container_width=True, hide_index=True)


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
    st.dataframe(
        filtered.drop(columns=["images", "metadata"], errors="ignore"),
        use_container_width=True,
        hide_index=True,
    )
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
    st.dataframe(filtered[[col for col in visible_cols if col in filtered.columns]], use_container_width=True, hide_index=True)
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

    detail_tabs = st.tabs(["Page Details", "Trace Steps", "Evaluator I/O", "Reader Input", "Graph Expansion", "Images", "Artifacts"])
    with detail_tabs[0]:
        render_page_details(data)
    with detail_tabs[1]:
        render_trace_steps(data)
    with detail_tabs[2]:
        render_evaluator_io(data)
    with detail_tabs[3]:
        render_reader_input(data)
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
            st.dataframe(page_nodes, use_container_width=True, hide_index=True)


def render_trace_steps(data: dict[str, Any]) -> None:
    trace_df = pd.DataFrame(data.get("trace_rows") or [])
    if trace_df.empty:
        st.info("No trace rows available.")
        return
    actions = st.multiselect("Action", sorted(trace_df["action"].dropna().unique()), key="trace_action_filter")
    if actions:
        trace_df = trace_df[trace_df["action"].isin(actions)]
    display_cols = ["step_index", "iteration", "action", "ok", "page_number", "node_id", "state_delta", "message"]
    st.dataframe(trace_df[[col for col in display_cols if col in trace_df.columns]], use_container_width=True, hide_index=True)
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
    selected_step = st.selectbox("Evaluator step", [row["step_index"] for row in evaluator_rows], key="agent_evaluator_step")
    row = next(row for row in evaluator_rows if row["step_index"] == selected_step)
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
        st.dataframe(pd.DataFrame(evaluator_input["candidate_actions"]), use_container_width=True, hide_index=True)
    if evaluator_input.get("opened_image_refs"):
        st.subheader("Opened image refs")
        st.dataframe(pd.DataFrame(evaluator_input["opened_image_refs"]), use_container_width=True, hide_index=True)
    if row.get("decision"):
        st.subheader("Parsed decision")
        st.json(row["decision"])
    if row.get("raw_response"):
        st.subheader("Raw response")
        render_xml_or_code(str(row["raw_response"]), key=f"evaluator_raw_{selected_step}")


def render_reader_input(data: dict[str, Any]) -> None:
    reader_input = data.get("reader_input") or {}
    if not reader_input:
        st.info("No reader input trace recorded for this sample.")
        return
    st.write({
        "content_part_count": reader_input.get("content_part_count"),
        "text_parts": len(reader_input.get("text_parts") or []),
        "image_refs": len(reader_input.get("image_refs") or []),
    })
    text_parts = reader_input.get("text_parts") or []
    if text_parts:
        selected = st.selectbox("Text part", list(range(len(text_parts))), format_func=lambda index: f"Text part {index + 1}", key="reader_text_part")
        render_xml_or_code(str(text_parts[int(selected)]), key=f"reader_text_{selected}")
    image_refs = reader_input.get("image_refs") or []
    if image_refs:
        st.subheader("Reader image refs")
        st.dataframe(pd.DataFrame(image_refs), use_container_width=True, hide_index=True)
    with st.expander("Sanitized messages"):
        st.json(reader_input.get("messages") or [])


def render_graph_expansion(data: dict[str, Any]) -> None:
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
    st.dataframe(df[[col for col in display_cols if col in df.columns]], use_container_width=True, hide_index=True)
    if df.empty:
        return
    selected_step = st.selectbox("Expansion step", df["step_index"].astype(int).tolist(), key="agent_expansion_step")
    row = next(row for row in rows if int(row.get("step_index")) == int(selected_step))
    st.json(row)


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
    st.dataframe(pd.DataFrame(deduped), use_container_width=True, hide_index=True)
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
    if "<agent_step_context" in value:
        return value[value.index("<agent_step_context") :]
    if "<agent_decision" in value:
        return value[value.index("<agent_decision") :]
    return value


def _image_ref_label(ref: dict[str, Any]) -> str:
    pieces = [str(ref.get("kind") or "image")]
    if ref.get("page_number") is not None:
        pieces.append(f"page {ref['page_number']}")
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
    st.dataframe(df, use_container_width=True, hide_index=True)


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
