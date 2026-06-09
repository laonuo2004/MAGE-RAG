import asyncio
from prebuild.amem_new import construct_memory
from pathlib import Path
import json
from loguru import logger
from tqdm import tqdm
from concurrent.futures import ProcessPoolExecutor, as_completed
from typing import List, Any, Tuple
from config.config import MEMORY_SYSTEMS_DIR
import time
from prebuild.usage_tracker import reset_usage, get_and_reset

# -------------------------
# collect q_id
# -------------------------
def collect_targets(base_dir="/data/new"):
    targets = []
    seen = set()
    for p in Path(base_dir).glob("processed*.jsonl"):
        try:
            with p.open("r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    obj = json.loads(line)
                    qid = (
                        obj.get("_id")
                    )
                    if qid and qid not in seen:
                        seen.add(qid)
                        targets.append(qid)
        except Exception as e:
            print(f"Skipping file {p}: {e}")
    return targets


TARGETS = collect_targets()


def _init_logger():
    log_dir = Path("results/logs")
    log_dir.mkdir(parents=True, exist_ok=True)
    logger.remove()
    logger.add(
        log_dir / "build_samples.log",
        rotation="10 MB",
        retention="10 days",
        enqueue=True,
        backtrace=True,
        diagnose=True,
        level="INFO",
        format="<green>{time:YYYY-MM-DD HH:mm:ss}</green> | <level>{level}</level> | {message}",
    )
    from datetime import datetime
    import os
    import json
    from config.config import (
        LLM_BASE_URL, MODELS, LLM_GENERATION, RESPONSE_FORMAT,
        MEMORY_SYSTEMS_DIR, PDF_TMP_DIR, DATASETS, MAX_CONCURRENCY, SAVE_CHECKS
    )

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    mem_tag = os.path.basename(MEMORY_SYSTEMS_DIR.rstrip("/")) or "memory_systems"

    snapshot = {
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "LLM_BASE_URL": LLM_BASE_URL,
        "MODELS": MODELS,
        "LLM_GENERATION": LLM_GENERATION,
        "RESPONSE_FORMAT_keys": list(RESPONSE_FORMAT.keys()),
        "MEMORY_SYSTEMS_DIR": MEMORY_SYSTEMS_DIR,
        "PDF_TMP_DIR": PDF_TMP_DIR,
        "DATASETS_keys": list(DATASETS.keys()),
        "MAX_CONCURRENCY": MAX_CONCURRENCY,
        "SAVE_CHECKS": {k: getattr(SAVE_CHECKS, k) for k in vars(SAVE_CHECKS)},
        "api_key_masked": "****"
    }

    snapshot_path = log_dir / f"{mem_tag}_config_{timestamp}.json"
    with open(snapshot_path, "w", encoding="utf-8") as f:
        json.dump(snapshot, f, ensure_ascii=False, indent=2)

    logger.info(f"Config snapshot saved: {snapshot_path}")
    return logger

# -------------------------
# execute 1 task in a subprocess
# -------------------------
def _build_one_run(qid: str) -> Tuple[str, bool, Any]:
    try:
        logger.info(f"[subprocess] start building: {qid}")
        reset_usage()
        start_t = time.time()
        ms = asyncio.run(construct_memory(qid, evolve_iters=3, window_size=3))
        duration_sec = round(time.time() - start_t, 3)
        usage = get_and_reset()
        sample_dir = Path(MEMORY_SYSTEMS_DIR) / qid
        sample_dir.mkdir(parents=True, exist_ok=True)
        by_stage = usage.get("by_stage", {})
        
        report = {
            "qid": qid,
            "duration_sec": duration_sec,
            "memory_count": len(ms.memories),
            "overall_usage": {
                "prompt_tokens": usage.get("prompt_tokens", 0),
                "completion_tokens": usage.get("completion_tokens", 0),
                "total_tokens": usage.get("total_tokens", 0),
                "embedding_tokens": usage.get("embedding_tokens", 0),
            },
            "stage_wise_usage": {
                "text_analysis": by_stage.get("text_analysis", {}),
                "image_analysis": by_stage.get("image_analysis", {}),
                "text_embedding": by_stage.get("text_embedding", {}),
                "image_embedding": by_stage.get("image_embedding", {}),
                "memory_evolution": by_stage.get("memory_evolution", {}),
                "re_embedding": by_stage.get("re_embedding", {}),
            },
            "full_usage": usage,
            "artifacts": {
                "memories_pkl": str(sample_dir / "memories.pkl"),
                "retriever_embeddings_npy": str(sample_dir / "retriever_embeddings.npy"),
            },
        }
        build_json_path = sample_dir / "build.json"
        if build_json_path.exists():
            logger.info(f"[subprocess] detected {build_json_path} exists, skipping write.")
        else:
            try:
                with build_json_path.open("x", encoding="utf-8") as f:
                    json.dump(report, f, ensure_ascii=False, indent=2)
            except FileExistsError:
                logger.info(f"[subprocess] detected {build_json_path} exists, skipping write.")

        stage_logs = []
        for stage_name, stage_data in by_stage.items():
            if stage_data.get("calls", 0) > 0:
                tokens_info = []
                if "total_tokens" in stage_data:
                    tokens_info.append(f"tokens={stage_data['total_tokens']}")
                if "embedding_tokens" in stage_data:
                    tokens_info.append(f"embed_tokens={stage_data['embedding_tokens']}")
                tokens_info.append(f"calls={stage_data['calls']}")
                tokens_info.append(f"time={stage_data.get('duration_sec', 0):.2f}s")
                stage_logs.append(f"{stage_name}({', '.join(tokens_info)})")
        
        logger.info(
            f"[subprocess] build completed: {qid}, memory count: {len(ms.memories)}, "
            f"total tokens: {usage.get('total_tokens', 0)}, "
            f"total embedding_tokens: {usage.get('embedding_tokens', 0)}, "
            f"total duration: {duration_sec}s"
        )
        if stage_logs:
            logger.info(f"[subprocess] stage statistics: {' | '.join(stage_logs)}")
        return qid, True, {
            "memory_count": len(ms.memories), 
            "metrics": {
                "qid": qid, 
                "duration_sec": duration_sec,
                "overall_usage": {
                    "prompt_tokens": usage.get("prompt_tokens", 0),
                    "completion_tokens": usage.get("completion_tokens", 0),
                    "total_tokens": usage.get("total_tokens", 0),
                    "embedding_tokens": usage.get("embedding_tokens", 0),
                },
                "stage_wise_usage": by_stage,
            }
        }
    except Exception as e:
        logger.error(f"[subprocess] build failed: {qid}: {e}")
        return qid, False, str(e)


# -------------------------
# chunk split (keep)
# -------------------------
def _chunk_targets(targets: List[str], size: int) -> List[List[str]]:
    buf = []
    out = []
    for t in targets:
        buf.append(t)
        if len(buf) >= size:
            out.append(buf)
            buf = []
    if buf:
        out.append(buf)
    return out


# -------------------------
# main function: multi-process serial + no thread
# -------------------------
def main():
    _init_logger()

    PROC_WORKERS = 4
    CHUNK_SIZE = 1 

    logger.info(f"total tasks: {len(TARGETS)}, process number: {PROC_WORKERS}, chunk size: {CHUNK_SIZE}")

    chunks = _chunk_targets(TARGETS, CHUNK_SIZE)

    ok = 0
    fail = 0
    all_metrics = []

    with tqdm(total=len(TARGETS), desc="Prebuilding", ncols=100) as pbar:
        with ProcessPoolExecutor(max_workers=PROC_WORKERS) as pool:
            fut_map = {
                pool.submit(_build_one_run, chunk[0]): chunk
                for chunk in chunks
            }

            for fut in as_completed(fut_map):
                chunk = fut_map[fut]
                qid = chunk[0]

                try:
                    t, success, extra = fut.result()
                    if success:
                        ok += 1
                        if isinstance(extra, dict) and "metrics" in extra:
                            all_metrics.append(extra["metrics"])
                    else:
                        fail += 1
                        logger.error(f"failed: {qid}: {extra}")
                except Exception as e:
                    fail += 1
                    logger.error(f"process execution failed: {qid}: {e}")
                pbar.update(1)
    
    logger.info(f"completed: {ok} success, {fail} failed, total {len(TARGETS)}")
    print(f"\ncompleted: {ok} success, {fail} failed, total {len(TARGETS)}")


if __name__ == "__main__":
    main()