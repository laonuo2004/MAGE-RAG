from pathlib import Path


def test_run_build_longdocurl_enables_abstract_budgeting():
    script = Path("scripts/run_build_longdocurl.sh").read_text(encoding="utf-8")

    assert "ABSTRACT_PROCESSOR_PATH" in script
    assert "PYTHON_BIN" in script
    assert "/root/autodl-tmp/conda/envs/logma-rag-py12/bin/python" in script
    assert "/root/autodl-tmp/ylz/models/Qwen3-VL-8B-Instruct" in script
    assert "--abstract-processor-path" in script
    assert "--abstract-context-window" in script
    assert "--abstract-output-tokens" in script
    assert "--abstract-safety-margin" in script
