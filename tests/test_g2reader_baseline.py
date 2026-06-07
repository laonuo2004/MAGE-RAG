import asyncio
from concurrent.futures import Future
import sys
import types

from baselines.g2reader import (
    G2ReaderPaths,
    G2ReaderRuntime,
    build_g2_memory_from_cfg,
    extract_g2_prediction,
    g2_document_memory_id,
    g2_memory_id_for_record,
    g2_processed_record,
    normalize_g2_devices,
)
import baselines.g2reader as g2reader_module
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
        LLM_GENERATION={},
        MEMORY_SYSTEMS_DIR="old-memory",
        PDF_TMP_DIR="old-pdf",
        DATASETS={},
        MAX_CONCURRENCY=10,
        PARALLEL_ANALYSIS=False,
    )
    monkeypatch.setitem(sys.modules, "config.config", config_module)
    cfg = {
        "litellm": {"base_url": "http://localhost:4000/v1", "api_key": "none"},
        "benchmarks": {"name": "longdocurl", "mineru_dir": str(tmp_path / "mineru")},
        "baselines": {
            "models": {"chat": "Qwen3-VL-8B-Instruct", "embed": "/root/autodl-tmp/ylz/models/bge-m3"},
            "params": {"memory_llm_concurrency": 64, "memory_parallel_analysis": True},
            "paths": {"cache_root": str(tmp_path / "cache")},
        },
    }

    G2ReaderRuntime(cfg).patch_g2_config()

    assert config_module.LLM_BASE_URL == "http://localhost:4000/v1"
    assert config_module.MODELS["chat"] == "Qwen3-VL-8B-Instruct"
    assert config_module.MODELS["embed"] == "/root/autodl-tmp/ylz/models/bge-m3"
    assert config_module.LLM_GENERATION["max_tokens"] == 32768
    assert config_module.MAX_CONCURRENCY == 64
    assert config_module.PARALLEL_ANALYSIS is True
    assert config_module.MEMORY_SYSTEMS_DIR == str(tmp_path / "cache" / "memory_systems")
    assert "g2_longdocurl" in config_module.DATASETS



def test_build_memory_dry_run_reports_missing_cache_as_pending(tmp_path):
    processed = tmp_path / "processed.jsonl"
    processed.write_text('{"_id": "g2_longdocurl_q1"}\n', encoding="utf-8")
    cfg = {
        "benchmarks": {"name": "longdocurl", "mineru_dir": str(tmp_path / "mineru")},
        "baselines": {"params": {"build_evolve_iters": 1}, "paths": {"cache_root": str(tmp_path / "cache")}},
    }

    result = build_g2_memory_from_cfg(cfg, processed_jsonl=processed, dry_run=True)

    assert result["built"] == 0
    assert result["skipped"] == 0
    assert result["pending"] == 1
    assert result["pending_ids"] == ["g2_longdocurl_q1"]


def test_build_memory_verifies_cache_after_build(monkeypatch, tmp_path):
    processed = tmp_path / "processed.jsonl"
    processed.write_text('{"_id": "g2_longdocurl_q1"}\n', encoding="utf-8")
    cfg = {
        "benchmarks": {"name": "longdocurl", "mineru_dir": str(tmp_path / "mineru")},
        "baselines": {"params": {"build_evolve_iters": 1}, "paths": {"cache_root": str(tmp_path / "cache")}},
    }

    def fake_build(self, record, memory_id=None):
        memory_id = memory_id or record["_id"]
        sample_dir = self.paths.memory_systems_dir / f"{memory_id}_iter_1"
        sample_dir.mkdir(parents=True, exist_ok=True)
        (sample_dir / "memories.pkl").write_bytes(b"memory")
        (sample_dir / "retriever_embeddings.npy").write_bytes(b"embedding")

    monkeypatch.setattr(G2ReaderRuntime, "build_memory_for_record", fake_build)

    result = build_g2_memory_from_cfg(cfg, processed_jsonl=processed)

    assert result["built"] == 1
    assert result["skipped"] == 0
    assert result["pending"] == 0
    assert result["pending_ids"] == []



def test_runtime_patch_mineru_chunking_uses_configured_sizes(tmp_path):
    calls = []

    def original_extract(mineru_path, chunk_size=3000, chunk_overlap=300):
        calls.append((mineru_path, chunk_size, chunk_overlap))
        return ["chunk"]

    amem_new = types.SimpleNamespace(extract_chunk_from_mineru=original_extract)
    cfg = {
        "benchmarks": {"name": "longdocurl", "mineru_dir": str(tmp_path / "mineru")},
        "baselines": {
            "params": {"memory_chunk_size": 777, "memory_chunk_overlap": 77},
            "paths": {"cache_root": str(tmp_path / "cache")},
        },
    }

    G2ReaderRuntime(cfg)._patch_mineru_chunking(amem_new)

    assert amem_new.extract_chunk_from_mineru("doc") == ["chunk"]
    assert calls == [("doc", 777, 77)]



def test_runtime_patch_llm_json_retry_retries_with_compact_system(tmp_path):
    calls = []

    async def original_call(system, user_payload, *, is_multimodal=False):
        calls.append((system, user_payload, is_multimodal))
        if len(calls) == 1:
            raise ValueError("bad json")
        return {"summary": "ok", "keywords": ["k"], "tags": ["t"]}

    amem_new = types.SimpleNamespace(call_llm_json=original_call, asyncio=asyncio)
    cfg = {
        "benchmarks": {"name": "longdocurl", "mineru_dir": str(tmp_path / "mineru")},
        "baselines": {
            "params": {"memory_analysis_retries": 1},
            "paths": {"cache_root": str(tmp_path / "cache")},
        },
    }

    G2ReaderRuntime(cfg)._patch_llm_json_retry(amem_new)
    result = asyncio.run(amem_new.call_llm_json("system", "payload"))

    assert result["summary"] == "ok"
    assert len(calls) == 2
    assert calls[0][0] == "system"
    assert "compact valid JSON" in calls[1][0]



def test_g2_document_memory_id_groups_questions_by_pdf():
    record_a = {"_id": "g2_mmlongbench_q1", "documents": "['shared.pdf']"}
    record_b = {"_id": "g2_mmlongbench_q2", "documents": "['shared.pdf']"}
    cfg = {"baselines": {"params": {"memory_cache_scope": "document"}}}

    assert g2_document_memory_id("mmlongbench", "shared.pdf") == "g2_mmlongbench_doc_shared"
    assert g2_memory_id_for_record("mmlongbench", record_a, cfg) == "g2_mmlongbench_doc_shared"
    assert g2_memory_id_for_record("mmlongbench", record_b, cfg) == "g2_mmlongbench_doc_shared"


def test_build_memory_deduplicates_document_scope(monkeypatch, tmp_path):
    processed = tmp_path / "processed.jsonl"
    processed.write_text(
        '{"_id": "g2_mmlongbench_q1", "documents": "[\\\"shared.pdf\\\"]"}\n'
        '{"_id": "g2_mmlongbench_q2", "documents": "[\\\"shared.pdf\\\"]"}\n',
        encoding="utf-8",
    )
    cfg = {
        "benchmarks": {"name": "mmlongbench", "mineru_dir": str(tmp_path / "mineru")},
        "baselines": {
            "params": {"build_evolve_iters": 1, "memory_cache_scope": "document"},
            "paths": {"cache_root": str(tmp_path / "cache")},
        },
    }
    built_ids = []

    def fake_build(self, record, memory_id=None):
        built_ids.append(memory_id)
        sample_dir = self.paths.memory_systems_dir / f"{memory_id}_iter_1"
        sample_dir.mkdir(parents=True, exist_ok=True)
        (sample_dir / "memories.pkl").write_bytes(b"memory")
        (sample_dir / "retriever_embeddings.npy").write_bytes(b"embedding")

    monkeypatch.setattr(G2ReaderRuntime, "build_memory_for_record", fake_build)

    result = build_g2_memory_from_cfg(cfg, processed_jsonl=processed)

    assert built_ids == ["g2_mmlongbench_doc_shared"]
    assert result["sample_count"] == 2
    assert result["memory_system_count"] == 1
    assert result["built"] == 1
    assert result["pending"] == 0


def test_build_memory_uses_configured_document_workers(monkeypatch, tmp_path):
    processed = tmp_path / "processed.jsonl"
    processed.write_text(
        '{"_id": "g2_longdocurl_q1"}\n'
        '{"_id": "g2_longdocurl_q2"}\n',
        encoding="utf-8",
    )
    cfg = {
        "benchmarks": {"name": "longdocurl", "mineru_dir": str(tmp_path / "mineru")},
        "baselines": {
            "params": {"build_evolve_iters": 1, "memory_build_workers": 2},
            "paths": {"cache_root": str(tmp_path / "cache")},
        },
    }
    captured = {"max_workers": None, "submitted": []}

    class ImmediateProcessPool:
        def __init__(self, max_workers):
            captured["max_workers"] = max_workers

        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, tb):
            return False

        def submit(self, fn, *args, **kwargs):
            captured["submitted"].append(args[1])
            future = Future()
            future.set_result(fn(*args, **kwargs))
            return future

    def fake_worker(worker_cfg, memory_id, record, evolve_iters):
        paths = G2ReaderPaths.from_cfg(worker_cfg)
        sample_dir = paths.memory_systems_dir / f"{memory_id}_iter_{evolve_iters}"
        sample_dir.mkdir(parents=True, exist_ok=True)
        (sample_dir / "memories.pkl").write_bytes(b"memory")
        (sample_dir / "retriever_embeddings.npy").write_bytes(b"embedding")
        return {"memory_id": memory_id, "status": "built"}

    monkeypatch.setattr(g2reader_module, "ProcessPoolExecutor", ImmediateProcessPool)
    monkeypatch.setattr(g2reader_module, "_build_g2_memory_worker", fake_worker)

    result = build_g2_memory_from_cfg(cfg, processed_jsonl=processed)

    assert captured["max_workers"] == 2
    assert captured["submitted"] == ["g2_longdocurl_q1", "g2_longdocurl_q2"]
    assert result["built"] == 2
    assert result["skipped"] == 0
    assert result["failed"] == 0



def test_runtime_patch_analysis_failure_logging_records_failed_text_chunk(tmp_path):
    async def original_analyze_content(payload, *, modality):
        raise ValueError("bad json")

    async def original_analyze_content_mineru(payload, *, modality, context="", caption=""):
        return {"summary": "ok", "keywords": ["k"], "tags": ["t"]}

    amem_new = types.SimpleNamespace(
        analyze_content=original_analyze_content,
        analyze_content_mineru=original_analyze_content_mineru,
        _project_current_memory_id="g2_mmlongbench_doc_shared",
    )
    cfg = {
        "benchmarks": {"name": "mmlongbench", "mineru_dir": str(tmp_path / "mineru")},
        "baselines": {"params": {}, "paths": {"cache_root": str(tmp_path / "cache")}},
    }
    runtime = G2ReaderRuntime(cfg)
    runtime._patch_analysis_failure_logging(amem_new)

    try:
        asyncio.run(amem_new.analyze_content("failed text", modality="text"))
    except ValueError:
        pass

    log_path = tmp_path / "cache" / "memory_systems" / "_debug_logs" / "failed_chunks.jsonl"
    line = log_path.read_text(encoding="utf-8").strip()
    assert '"memory_id": "g2_mmlongbench_doc_shared"' in line
    assert '"payload": "failed text"' in line
    assert '"error_type": "ValueError"' in line



def test_runtime_patch_analysis_generation_and_prompts(tmp_path):
    amem_new = types.SimpleNamespace(
        LLM_GENERATION={"max_tokens": 8192, "temperature": 0.7},
        PROMPTS={"text": "old", "text_keyword": "old", "image_ocr_keyword": "old {context} {caption}"},
    )
    cfg = {
        "benchmarks": {"name": "longdocurl", "mineru_dir": str(tmp_path / "mineru")},
        "baselines": {
            "params": {"memory_analysis_max_tokens": 16000, "memory_compact_analysis_prompts": True},
            "paths": {"cache_root": str(tmp_path / "cache")},
        },
    }
    runtime = G2ReaderRuntime(cfg)

    runtime._patch_analysis_generation(amem_new)
    runtime._patch_analysis_prompts(amem_new)

    assert amem_new.LLM_GENERATION["max_tokens"] == 16000
    assert amem_new.LLM_GENERATION["temperature"] == 0.0
    assert "keywords <= 16" in amem_new.PROMPTS["text"]
    assert "text_content <= 800" in amem_new.PROMPTS["image_ocr_keyword"]


def test_runtime_patch_memory_build_concurrency_uses_configured_limits(tmp_path):
    amem_new = types.SimpleNamespace(MAX_CONCURRENCY=10, PARALLEL_ANALYSIS=False, _llm_sem_by_loop={1: object()})
    memory_layer = types.SimpleNamespace(MAX_CONCURRENCY=10)
    cfg = {
        "benchmarks": {"name": "longdocurl", "mineru_dir": str(tmp_path / "mineru")},
        "baselines": {
            "params": {"memory_llm_concurrency": 64, "memory_parallel_analysis": True},
            "paths": {"cache_root": str(tmp_path / "cache")},
        },
    }

    G2ReaderRuntime(cfg)._patch_memory_build_concurrency(amem_new, memory_layer)

    assert amem_new.MAX_CONCURRENCY == 64
    assert memory_layer.MAX_CONCURRENCY == 64
    assert amem_new.PARALLEL_ANALYSIS is True
    assert amem_new._llm_sem_by_loop == {}


def test_normalize_g2_devices_accepts_ordered_fallback_list():
    assert normalize_g2_devices("cuda:1,cuda:0,cpu") == ["cuda:1", "cuda:0", "cpu"]
    assert normalize_g2_devices(["cuda:1", "cuda:0", "cpu"]) == ["cuda:1", "cuda:0", "cpu"]
    assert normalize_g2_devices("cuda:0") == "cuda:0"


def test_embedder_instance_passes_normalized_device_fallbacks(monkeypatch, tmp_path):
    captured = {}

    class FakeEmbedder:
        def __init__(self, model_path, *, use_fp16=True, devices="cuda:0"):
            captured["model_path"] = model_path
            captured["use_fp16"] = use_fp16
            captured["devices"] = devices

    monkeypatch.setattr(g2reader_module, "LocalBGEM3Embedder", FakeEmbedder)
    cfg = {
        "benchmarks": {"name": "longdocurl", "mineru_dir": str(tmp_path / "mineru")},
        "baselines": {
            "models": {"embed": "/models/bge-m3"},
            "devices": "cuda:1,cuda:0,cpu",
            "use_fp16": True,
            "paths": {"cache_root": str(tmp_path / "cache")},
        },
    }

    G2ReaderRuntime(cfg)._embedder_instance()

    assert captured == {
        "model_path": "/models/bge-m3",
        "use_fp16": True,
        "devices": ["cuda:1", "cuda:0", "cpu"],
    }


def test_local_bgem3_embedder_falls_back_when_gpu_does_not_fit(monkeypatch):
    attempts = []

    class FakeBGEM3FlagModel:
        def __init__(self, model_path, *, use_fp16=True, devices="cuda:0"):
            attempts.append(devices)
            if devices == "cuda:1":
                raise RuntimeError("CUDA out of memory")
            self.devices = devices

    monkeypatch.setitem(sys.modules, "FlagEmbedding", types.SimpleNamespace(BGEM3FlagModel=FakeBGEM3FlagModel))

    embedder = g2reader_module.LocalBGEM3Embedder("/models/bge-m3", devices=["cuda:1", "cuda:0", "cpu"])

    assert attempts == ["cuda:1", "cuda:0"]
    assert embedder.model.devices == "cuda:0"


def test_local_bgem3_embedder_falls_back_to_cpu_when_gpus_do_not_fit(monkeypatch):
    attempts = []

    class FakeBGEM3FlagModel:
        def __init__(self, model_path, *, use_fp16=True, devices="cuda:0"):
            attempts.append(devices)
            if str(devices).startswith("cuda"):
                raise RuntimeError("CUDA out of memory")
            self.devices = devices

    monkeypatch.setitem(sys.modules, "FlagEmbedding", types.SimpleNamespace(BGEM3FlagModel=FakeBGEM3FlagModel))

    embedder = g2reader_module.LocalBGEM3Embedder("/models/bge-m3", devices=["cuda:1", "cuda:0", "cpu"])

    assert attempts == ["cuda:1", "cuda:0", "cpu"]
    assert embedder.model.devices == "cpu"


def test_runtime_patch_text_analysis_split_merges_successful_halves(tmp_path):
    calls = []

    async def original_analyze_content(payload, *, modality):
        calls.append(payload)
        if len(payload) > 3:
            raise ValueError("bad json")
        return {"summary": payload, "keywords": [payload], "tags": ["part"]}

    async def original_analyze_content_mineru(payload, *, modality, context="", caption=""):
        return await original_analyze_content(payload, modality=modality)

    amem_new = types.SimpleNamespace(
        analyze_content=original_analyze_content,
        analyze_content_mineru=original_analyze_content_mineru,
    )
    cfg = {
        "benchmarks": {"name": "longdocurl", "mineru_dir": str(tmp_path / "mineru")},
        "baselines": {
            "params": {
                "memory_analysis_split_max_depth": 3,
                "memory_analysis_split_min_chars": 2,
                "memory_analysis_max_summary_chars": 100,
                "memory_analysis_max_keywords": 10,
            },
            "paths": {"cache_root": str(tmp_path / "cache")},
        },
    }

    G2ReaderRuntime(cfg)._patch_text_analysis_split(amem_new)
    result = asyncio.run(amem_new.analyze_content("abcdef", modality="text"))

    assert calls == ["abcdef", "abc", "def"]
    assert result["summary"] == "abc def"
    assert result["keywords"] == ["abc", "def"]
    assert result["tags"] == ["part"]


def test_runtime_patch_memory_evolve_neighbor_limit(tmp_path):
    class AgenticMemorySystem:
        pass

    async def original_find_related_notes(self, query, k=5, include_neighbors=True, modality="all"):
        return k, []

    AgenticMemorySystem.find_related_notes = original_find_related_notes
    memory_layer = types.SimpleNamespace(AgenticMemorySystem=AgenticMemorySystem)
    cfg = {
        "benchmarks": {"name": "longdocurl", "mineru_dir": str(tmp_path / "mineru")},
        "baselines": {
            "params": {"memory_evolution_neighbor_k": 3},
            "paths": {"cache_root": str(tmp_path / "cache")},
        },
    }

    G2ReaderRuntime(cfg)._patch_memory_evolve_options(memory_layer)
    memory_system = AgenticMemorySystem()

    assert asyncio.run(memory_system.find_related_notes("q", k=5, include_neighbors=False))[0] == 3
    assert asyncio.run(memory_system.find_related_notes("q", k=5, include_neighbors=True))[0] == 5


def test_write_build_metadata_records_effective_build_params(tmp_path):
    cfg = {
        "benchmarks": {"name": "longdocurl", "mineru_dir": str(tmp_path / "mineru")},
        "baselines": {
            "models": {"chat": "Qwen3-VL-8B-Instruct", "embed": "/models/bge-m3"},
            "params": {
                "build_evolve_iters": 1,
                "memory_cache_scope": "document",
                "memory_chunk_size": 2400,
                "memory_analysis_max_tokens": 32768,
                "memory_evolution_neighbor_k": 3,
            },
            "paths": {"cache_root": str(tmp_path / "cache")},
        },
    }
    runtime = G2ReaderRuntime(cfg)
    sample_dir = runtime.paths.memory_systems_dir / "g2_longdocurl_doc_1_iter_1"
    sample_dir.mkdir(parents=True)
    (sample_dir / "memories.pkl").write_bytes(b"memory")
    (sample_dir / "retriever_embeddings.npy").write_bytes(b"embedding")

    runtime._write_build_metadata("g2_longdocurl_doc_1", evolve_iters=1, window_size=2)

    metadata = __import__("json").loads((sample_dir / "build_metadata.json").read_text(encoding="utf-8"))
    assert metadata["memory_id"] == "g2_longdocurl_doc_1"
    assert metadata["build_params"]["memory_chunk_size"] == 2400
    assert metadata["build_params"]["memory_analysis_max_tokens"] == 32768
    assert metadata["build_params"]["memory_evolution_neighbor_k"] == 3



def test_runtime_patch_llm_json_retry_preserves_text_content_for_multimodal(tmp_path):
    calls = []

    async def original_call(system, user_payload, *, is_multimodal=False):
        calls.append((system, user_payload, is_multimodal))
        if len(calls) == 1:
            raise ValueError("bad json")
        return {"summary": "ok", "keywords": ["k"], "tags": ["t"], "text_content": "ocr"}

    amem_new = types.SimpleNamespace(call_llm_json=original_call, asyncio=asyncio)
    cfg = {
        "benchmarks": {"name": "longdocurl", "mineru_dir": str(tmp_path / "mineru")},
        "baselines": {
            "params": {"memory_analysis_retries": 1},
            "paths": {"cache_root": str(tmp_path / "cache")},
        },
    }

    G2ReaderRuntime(cfg)._patch_llm_json_retry(amem_new)
    result = asyncio.run(amem_new.call_llm_json("system", "payload", is_multimodal=True))

    assert result["text_content"] == "ocr"
    assert "text_content" in calls[1][0]


def test_analysis_split_and_failure_logging_are_idempotent(tmp_path):
    async def original_analyze_content(payload, *, modality):
        return {"summary": "ok", "keywords": ["k"], "tags": ["t"]}

    async def original_analyze_content_mineru(payload, *, modality, context="", caption=""):
        return {"summary": "ok", "keywords": ["k"], "tags": ["t"]}

    amem_new = types.SimpleNamespace(
        analyze_content=original_analyze_content,
        analyze_content_mineru=original_analyze_content_mineru,
        _project_current_memory_id="g2_longdocurl_doc_1",
    )
    cfg = {
        "benchmarks": {"name": "longdocurl", "mineru_dir": str(tmp_path / "mineru")},
        "baselines": {"params": {}, "paths": {"cache_root": str(tmp_path / "cache")}},
    }
    runtime = G2ReaderRuntime(cfg)

    runtime._patch_text_analysis_split(amem_new)
    runtime._patch_analysis_failure_logging(amem_new)
    first_analyze = amem_new.analyze_content
    first_mineru = amem_new.analyze_content_mineru

    runtime._patch_text_analysis_split(amem_new)
    runtime._patch_analysis_failure_logging(amem_new)

    assert amem_new.analyze_content is first_analyze
    assert amem_new.analyze_content_mineru is first_mineru
