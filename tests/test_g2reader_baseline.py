import sys
import types

from baselines.g2reader import G2ReaderPaths, G2ReaderRuntime, extract_g2_prediction, g2_processed_record
from baselines.wrapper import build_context_builder


def test_g2reader_processed_record_uses_baseline_id_and_judge_default():
    sample = {
        "question_id": "free_gpt4o_4026369_60_70_12",
        "doc_no": "4026369",
        "question": "Which publication?",
        "answer": "University/Advantsar Communications Project",
    }

    record = g2_processed_record("longdocurl", sample)

    assert record["_id"] == "g2_longdocurl_free_gpt4o_4026369_60_70_12"
    assert record["main_doc"] == "4026369.pdf"
    assert record["documents"] == "['4026369.pdf']"
    assert record["judge"] == ""


def test_g2reader_paths_are_benchmark_cache_not_benchmark_package():
    cfg = {"benchmarks": {"name": "mmlongbench"}}

    paths = G2ReaderPaths.from_cfg(cfg)

    assert str(paths.cache_root) == "benchmarks/mmlongbench/data/cache/g2_reader"
    assert str(paths.manifest_dir) == "benchmarks/mmlongbench/data/cache/g2_reader/manifests"


def test_g2reader_registered_from_baseline_entry():
    cfg = {
        "litellm": {"base_url": "http://localhost:4000/v1", "api_key": "none"},
        "benchmarks": {"name": "longdocurl"},
        "baselines": {
            "name": "g2-reader",
            "models": {"chat": "Qwen3-VL-8B-Instruct", "embed": "/root/autodl-tmp/ylz/models/bge-m3"},
            "params": {},
            "paths": {},
        },
    }

    builder = build_context_builder(cfg)

    assert builder.name == "g2-reader"
    assert builder.__class__.__module__ == "baselines.g2reader"
    assert hasattr(builder, "run_sample")


def test_extract_g2_prediction_prefers_output_tag():
    assert extract_g2_prediction("prefix <output>final answer</output> suffix") == "final answer"
    assert extract_g2_prediction("plain answer") == "plain answer"
    assert extract_g2_prediction("") is None


def test_runtime_patch_updates_g2_config_without_modifying_vendored_package(monkeypatch, tmp_path):
    config_module = types.SimpleNamespace(
        LLM_BASE_URL="old",
        LLM_API_KEY="old-key",
        EMBED_BASE_URL="old-embed",
        EMBED_API_KEY="old-embed-key",
        MODELS={"chat": "old-chat", "embed": "old-embed"},
        MEMORY_SYSTEMS_DIR="old-memory",
        PDF_TMP_DIR="old-pdf",
        DATASETS={},
    )
    monkeypatch.setitem(sys.modules, "config.config", config_module)
    cfg = {
        "litellm": {"base_url": "http://localhost:4000/v1", "api_key": "none"},
        "benchmarks": {"name": "longdocurl", "mineru_dir": str(tmp_path / "mineru")},
        "baselines": {
            "models": {"chat": "Qwen3-VL-8B-Instruct", "embed": "/root/autodl-tmp/ylz/models/bge-m3"},
            "params": {},
            "paths": {"cache_root": str(tmp_path / "cache")},
        },
    }

    G2ReaderRuntime(cfg).patch_g2_config()

    assert config_module.LLM_BASE_URL == "http://localhost:4000/v1"
    assert config_module.MODELS["chat"] == "Qwen3-VL-8B-Instruct"
    assert config_module.MODELS["embed"] == "/root/autodl-tmp/ylz/models/bge-m3"
    assert config_module.MEMORY_SYSTEMS_DIR == str(tmp_path / "cache" / "memory_systems")
    assert "g2_longdocurl" in config_module.DATASETS
