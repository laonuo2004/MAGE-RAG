from pathlib import Path


def test_g2_longdocurl_pm2_wrapper_uses_safe_resume_defaults():
    script = Path("scripts/run_g2_longdocurl_pm2.sh").read_text(encoding="utf-8")

    assert "WORKERS=\"${WORKERS:-16}\"" in script
    assert "overwrite=false" in script
    assert "benchmarks.workers=${WORKERS}" in script
    assert "full_adjust0_corr_relaxed.jsonl" in script
    assert "cleanup_stale_g2_children" in script
    assert "failed_count" in script
    assert "exit 1" in script
