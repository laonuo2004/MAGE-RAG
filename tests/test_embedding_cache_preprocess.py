from pathlib import Path


def test_main_uses_benchmark_embedding_cache_entrypoint():
    source = Path("main.py").read_text(encoding="utf-8")

    assert "from benchmarks.utils.embedding_cache import prepare_embedding_cache" in source
    assert "from baselines.bgem3_cache" not in source
    assert "from baselines.colbertv2_cache" not in source


def test_legacy_baseline_cache_modules_are_thin_compatibility_exports():
    bgem3_source = Path("baselines/bgem3_cache.py").read_text(encoding="utf-8")
    colbert_source = Path("baselines/colbertv2_cache.py").read_text(encoding="utf-8")

    assert "from benchmarks.utils.embedding_cache import prepare_bgem3_cache" in bgem3_source
    assert "from benchmarks.utils.embedding_cache import prepare_colbertv2_cache" in colbert_source
    assert "BGEM3ContextBuilder" not in bgem3_source
    assert "ColBERTv2ContextBuilder" not in colbert_source


def test_benchmark_embedding_scripts_delegate_to_shared_cli():
    scripts = [
        Path("benchmarks/longdocurl/scripts/generate_bgem3_embeddings.py"),
        Path("benchmarks/longdocurl/scripts/generate_colbertv2_embeddings.py"),
        Path("benchmarks/mmlongbench/scripts/generate_bgem3_embeddings.py"),
        Path("benchmarks/mmlongbench/scripts/generate_colbertv2_embeddings.py"),
    ]

    for script in scripts:
        source = script.read_text(encoding="utf-8")
        assert "run_embedding_cache_cli" in source
        assert "def encode_doc_chunks" not in source
        assert "def encode_query" not in source
        assert "def load_samples" not in source
