from typing import Any, Dict
import time
import copy

USAGE: Dict[str, Any] = {
    "prompt_tokens": 0,
    "completion_tokens": 0,
    "total_tokens": 0,
    "embedding_tokens": 0,
    "calls": [],
    "by_stage": {
        "text_analysis": {
            "prompt_tokens": 0,
            "completion_tokens": 0,
            "total_tokens": 0,
            "calls": 0,
            "duration_sec": 0.0,
            "max_single_call_sec": 0.0,
            "avg_single_call_sec": 0.0,
            "_single_call_sum": 0.0,
        },
        "image_analysis": {
            "prompt_tokens": 0,
            "completion_tokens": 0,
            "total_tokens": 0,
            "calls": 0,
            "duration_sec": 0.0,
            "max_single_call_sec": 0.0,
            "avg_single_call_sec": 0.0,
            "_single_call_sum": 0.0,
        },
        "text_embedding": {
            "embedding_tokens": 0,
            "calls": 0,
            "duration_sec": 0.0,
            "max_single_call_sec": 0.0,
            "avg_single_call_sec": 0.0,
            "_single_call_sum": 0.0,
        },
        "image_embedding": {
            "embedding_tokens": 0,
            "calls": 0,
            "duration_sec": 0.0,
            "max_single_call_sec": 0.0,
            "avg_single_call_sec": 0.0,
            "_single_call_sum": 0.0,
        },
        "memory_evolution": {
            "prompt_tokens": 0,
            "completion_tokens": 0,
            "total_tokens": 0,
            "calls": 0,
            "duration_sec": 0.0,
            "max_single_call_sec": 0.0,
            "avg_single_call_sec": 0.0,
            "_single_call_sum": 0.0,
        },
        "re_embedding": {
            "embedding_tokens": 0,
            "calls": 0,
            "duration_sec": 0.0,
            "max_single_call_sec": 0.0,
            "avg_single_call_sec": 0.0,
            "_single_call_sum": 0.0,
        },
    },
}

def _extract(usage: Any) -> Dict[str, int]:
    if usage is None:
        return {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}
    try:
        if isinstance(usage, dict):
            return {
                "prompt_tokens": int(usage.get("prompt_tokens", 0) or 0),
                "completion_tokens": int(usage.get("completion_tokens", 0) or 0),
                "total_tokens": int(usage.get("total_tokens", 0) or 0),
            }
        return {
            "prompt_tokens": int(getattr(usage, "prompt_tokens", 0) or 0),
            "completion_tokens": int(getattr(usage, "completion_tokens", 0) or 0),
            "total_tokens": int(getattr(usage, "total_tokens", 0) or 0),
        }
    except Exception:
        return {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}

def add_chat_usage(usage: Any, meta: Dict[str, Any] | None = None) -> None:
    t = _extract(usage)
    USAGE["prompt_tokens"] += t["prompt_tokens"]
    USAGE["completion_tokens"] += t["completion_tokens"]
    USAGE["total_tokens"] += t["total_tokens"]
    call = {"type": "chat", "timestamp": time.time(), **t}
    if meta:
        call.update(meta)
    USAGE["calls"].append(call)
    
    if meta and "qkind" in meta:
        qkind = meta["qkind"]
        stage = None
        if qkind == "analyze_text":
            stage = "text_analysis"
        elif qkind == "analyze_multimodal":
            stage = "image_analysis"
        elif qkind == "memory_evolution":
            stage = "memory_evolution"
        
        if stage and stage in USAGE["by_stage"]:
            USAGE["by_stage"][stage]["prompt_tokens"] += t["prompt_tokens"]
            USAGE["by_stage"][stage]["completion_tokens"] += t["completion_tokens"]
            USAGE["by_stage"][stage]["total_tokens"] += t["total_tokens"]
            USAGE["by_stage"][stage]["calls"] += 1

def add_embed_usage(usage: Any, meta: Dict[str, Any] | None = None) -> None:
    t = _extract(usage)
    emb = t["total_tokens"] or t["prompt_tokens"]
    USAGE["embedding_tokens"] += emb
    call = {"type": "embed", "timestamp": time.time(), "total_tokens": emb}
    if meta:
        call.update(meta)
    USAGE["calls"].append(call)
    
    if meta and "kind" in meta:
        kind = meta["kind"]
        stage = None
        if kind == "text_embedding":
            stage = "text_embedding"
        elif kind == "image_embedding":
            stage = "image_embedding"
        elif kind == "re_embedding":
            stage = "re_embedding"
        
        if stage and stage in USAGE["by_stage"]:
            USAGE["by_stage"][stage]["embedding_tokens"] += emb
            USAGE["by_stage"][stage]["calls"] += 1

def add_stage_duration(stage: str, duration_sec: float) -> None:
    """Record execution time for a stage"""
    if stage in USAGE["by_stage"]:
        USAGE["by_stage"][stage]["duration_sec"] += duration_sec

def add_single_call_duration(stage: str, call_duration_sec: float) -> None:
    """Record single call duration and update maximum single call duration for the stage"""
    if stage in USAGE["by_stage"]:
        stage_data = USAGE["by_stage"][stage]
        
        current_max = stage_data.get("max_single_call_sec", 0.0)
        if call_duration_sec > current_max:
            stage_data["max_single_call_sec"] = call_duration_sec
        
        stage_data["_single_call_sum"] += call_duration_sec

def reset_usage() -> None:
    USAGE["prompt_tokens"] = 0
    USAGE["completion_tokens"] = 0
    USAGE["total_tokens"] = 0
    USAGE["embedding_tokens"] = 0
    USAGE["calls"] = []
    for stage in USAGE["by_stage"]:
        stage_data = USAGE["by_stage"][stage]
        if "prompt_tokens" in stage_data:
            stage_data["prompt_tokens"] = 0
            stage_data["completion_tokens"] = 0
            stage_data["total_tokens"] = 0
        if "embedding_tokens" in stage_data:
            stage_data["embedding_tokens"] = 0
        stage_data["calls"] = 0
        stage_data["duration_sec"] = 0.0
        stage_data["max_single_call_sec"] = 0.0
        stage_data["avg_single_call_sec"] = 0.0
        stage_data["_single_call_sum"] = 0.0

def get_summary() -> Dict[str, Any]:
    stage_data_copy = copy.deepcopy(USAGE["by_stage"])
    for stage, data in stage_data_copy.items():
        if data["calls"] > 0 and data.get("_single_call_sum", 0) > 0:
            data["avg_single_call_sec"] = data["_single_call_sum"] / data["calls"]
        else:
            data["avg_single_call_sec"] = 0.0
        
        data.pop("_single_call_sum", None)
    
    return {
        "prompt_tokens": USAGE["prompt_tokens"],
        "completion_tokens": USAGE["completion_tokens"],
        "total_tokens": USAGE["total_tokens"],
        "embedding_tokens": USAGE["embedding_tokens"],
        "calls": copy.deepcopy(USAGE["calls"]),
        "by_stage": stage_data_copy,
    }

def get_and_reset() -> Dict[str, Any]:
    s = get_summary()
    reset_usage()
    return s