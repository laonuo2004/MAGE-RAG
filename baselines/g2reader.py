from __future__ import annotations

import argparse
import ast
import asyncio
import csv
import json
import re
import subprocess
import sys
import time
from argparse import Namespace
from concurrent.futures import ProcessPoolExecutor, as_completed
from concurrent.futures.process import BrokenProcessPool
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import Any, Iterable

import numpy as np
from tqdm import tqdm

from baselines.base import ContextBuilder
from utils.config_utils import get_config_value, require_config_value


VENDORED_G2_ROOT = Path(__file__).resolve().parent / "g2_reader"


@dataclass(frozen=True)
class G2ReaderPaths:
    benchmark_name: str
    cache_root: Path
    manifest_dir: Path
    memory_systems_dir: Path
    logs_dir: Path

    @classmethod
    def from_cfg(cls, cfg) -> "G2ReaderPaths":
        benchmark_name = str(require_config_value(cfg, "benchmarks.name"))
        configured_cache_root = get_config_value(cfg, "baselines.paths.cache_root")
        cache_root = (
            Path(str(configured_cache_root))
            if configured_cache_root
            else Path(f"benchmarks/{benchmark_name}/data/cache/g2_reader")
        )
        return cls(
            benchmark_name=benchmark_name,
            cache_root=cache_root,
            manifest_dir=cache_root / "manifests",
            memory_systems_dir=cache_root / "memory_systems",
            logs_dir=cache_root / "logs",
        )

    def ensure_dirs(self) -> None:
        for path in (self.manifest_dir, self.memory_systems_dir, self.logs_dir):
            path.mkdir(parents=True, exist_ok=True)


def ensure_vendored_g2_import_path() -> None:
    path = str(VENDORED_G2_ROOT)
    if path in sys.path:
        sys.path.remove(path)
    sys.path.insert(0, path)


def prepare_vendored_g2_imports() -> None:
    import types

    ensure_vendored_g2_import_path()
    # G2 imports top-level packages named config/prebuild/agent_search/utils.
    # The host project also has a regular top-level utils package, so create
    # vendored parent packages explicitly before importing G2 modules.
    for name in ("utils", "config", "prebuild", "agent_search"):
        package_dir = VENDORED_G2_ROOT / name
        if not package_dir.exists():
            continue
        for module_name in list(sys.modules):
            if module_name.startswith(f"{name}."):
                module = sys.modules[module_name]
                module_file = str(getattr(module, "__file__", "") or "")
                if module_file and not module_file.startswith(str(VENDORED_G2_ROOT)):
                    sys.modules.pop(module_name, None)
        module = sys.modules.get(name)
        module_file = str(getattr(module, "__file__", "") or "") if module is not None else ""
        module_paths = [str(item) for item in getattr(module, "__path__", [])] if module is not None else []
        is_vendored = module_file.startswith(str(VENDORED_G2_ROOT)) or any(
            item.startswith(str(VENDORED_G2_ROOT)) for item in module_paths
        )
        if is_vendored:
            continue
        vendored_package = types.ModuleType(name)
        vendored_package.__path__ = [str(package_dir)]
        vendored_package.__package__ = name
        vendored_package.__file__ = str(package_dir / "__init__.py")
        sys.modules[name] = vendored_package


def install_lightweight_g2_visdom_utils() -> None:
    import base64
    import re as re_module
    import types
    from io import BytesIO

    module = types.ModuleType("utils.visdom_utils")
    module.__file__ = str(VENDORED_G2_ROOT / "utils" / "visdom_utils.py")

    async def get_pdf(url: str, filename: str):
        raise RuntimeError(
            "G2-Reader PDF download fallback is disabled in the project wrapper; "
            "use DATASETS + MinerU cache inputs instead."
        )

    def clean_text(text):
        if not text:
            return text
        control_chars = "".join(map(chr, range(0, 32))).replace("\t", "").replace("\n", "").replace("\r", "")
        cleaned = re_module.sub("[%s]" % re_module.escape(control_chars), " ", str(text))
        cleaned = cleaned.replace("\x00", "")
        return re_module.sub(r"\s+", " ", cleaned).strip()

    def encode_image(pil_image):
        buffered = BytesIO()
        pil_image.save(buffered, format="JPEG")
        return base64.b64encode(buffered.getvalue()).decode("utf-8")

    module.get_pdf = get_pdf
    module.clean_text = clean_text
    module.encode_image = encode_image
    sys.modules["utils.visdom_utils"] = module



def install_lightweight_text_splitters() -> None:
    import types

    if "langchain_text_splitters" in sys.modules:
        return

    module = types.ModuleType("langchain_text_splitters")

    class RecursiveCharacterTextSplitter:
        def __init__(self, chunk_size=3000, chunk_overlap=300, separators=None):
            self.chunk_size = int(chunk_size)
            self.chunk_overlap = max(0, int(chunk_overlap))
            self.separators = separators or ["\n\n", "\n", " ", ""]

        def split_text(self, text):
            text = "" if text is None else str(text)
            if len(text) <= self.chunk_size:
                return [text] if text else []
            chunks = []
            start = 0
            while start < len(text):
                end = min(len(text), start + self.chunk_size)
                split_at = end
                window = text[start:end]
                for sep in self.separators:
                    if not sep:
                        continue
                    idx = window.rfind(sep)
                    if idx > 0 and start + idx > start:
                        split_at = start + idx + len(sep)
                        break
                chunk = text[start:split_at].strip()
                if chunk:
                    chunks.append(chunk)
                if split_at >= len(text):
                    break
                start = max(split_at - self.chunk_overlap, start + 1)
            return chunks

    module.RecursiveCharacterTextSplitter = RecursiveCharacterTextSplitter
    sys.modules["langchain_text_splitters"] = module


def install_g2_lightweight_shims() -> None:
    install_lightweight_g2_visdom_utils()
    install_lightweight_text_splitters()



def g2_document_name_for_sample(benchmark_name: str, sample: dict[str, Any]) -> str:
    if benchmark_name == "mmlongbench":
        return PurePosixPath(str(sample["doc_id"])).name
    if benchmark_name == "longdocurl":
        return f"{sample['doc_no']}.pdf"
    raise ValueError(f"Unsupported benchmark for G2-Reader: {benchmark_name}")


def g2_processed_record(benchmark_name: str, sample: dict[str, Any]) -> dict[str, Any]:
    doc_name = g2_document_name_for_sample(benchmark_name, sample)
    question_id = re.sub(r"[^A-Za-z0-9._-]+", "_", str(sample.get("question_id", "sample"))).strip("._") or "sample"
    if benchmark_name == "mmlongbench":
        sample_id = f"g2_mmlongbench_{question_id}"
        domain = "MMLongBench"
    elif benchmark_name == "longdocurl":
        sample_id = f"g2_longdocurl_{question_id}"
        domain = "LongDocURL"
    else:
        raise ValueError(f"Unsupported benchmark for G2-Reader: {benchmark_name}")
    return {
        "_id": sample_id,
        "domain": domain,
        "sub_domain": benchmark_name,
        "question": sample["question"],
        "answer": sample["answer"],
        "judge": "",
        "main_doc": doc_name,
        "documents": repr([doc_name]),
    }



def g2_record_documents(record: dict[str, Any]) -> list[str]:
    documents = record.get("documents", "[]")
    if isinstance(documents, list):
        return [str(item) for item in documents]
    parsed = ast.literal_eval(str(documents))
    if not isinstance(parsed, list):
        raise ValueError(f"G2 documents field must be a list: {documents!r}")
    return [str(item) for item in parsed]


def g2_document_memory_id(benchmark_name: str, document_name: str) -> str:
    stem = PurePosixPath(str(document_name)).name
    stem = re.sub(r"\.pdf$", "", stem, flags=re.IGNORECASE)
    safe_stem = re.sub(r"[^A-Za-z0-9._-]+", "_", stem).strip("._") or "document"
    return f"g2_{benchmark_name}_doc_{safe_stem}"


def g2_memory_cache_scope(cfg) -> str:
    return str(get_config_value(cfg, "baselines.params.memory_cache_scope", "document"))


def g2_memory_id_for_record(benchmark_name: str, record: dict[str, Any], cfg=None) -> str:
    if cfg is not None and g2_memory_cache_scope(cfg) != "document":
        return str(record["_id"])
    documents = g2_record_documents(record)
    if len(documents) != 1:
        return str(record["_id"])
    return g2_document_memory_id(benchmark_name, documents[0])


def g2_memory_jobs_for_records(benchmark_name: str, records: list[dict[str, Any]], cfg) -> list[tuple[str, dict[str, Any]]]:
    jobs: dict[str, dict[str, Any]] = {}
    for record in records:
        memory_id = g2_memory_id_for_record(benchmark_name, record, cfg)
        if memory_id in jobs:
            continue
        if memory_id == str(record["_id"]):
            jobs[memory_id] = record
            continue
        memory_record = dict(record)
        memory_record["_id"] = memory_id
        jobs[memory_id] = memory_record
    return list(jobs.items())


def write_g2_manifest(benchmark_name: str, samples: Iterable[dict[str, Any]], paths: G2ReaderPaths) -> dict[str, Any]:
    paths.ensure_dirs()
    records = [g2_processed_record(benchmark_name, sample) for sample in samples]
    processed_path = paths.manifest_dir / f"processed_{benchmark_name}.jsonl"
    csv_path = paths.manifest_dir / f"{benchmark_name}.csv"
    with processed_path.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")
    with csv_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["q_id", "documents"])
        writer.writeheader()
        for record in records:
            writer.writerow({"q_id": record["_id"], "documents": record["documents"]})
    return {"processed_jsonl": str(processed_path), "csv": str(csv_path), "sample_count": len(records)}




def load_samples_for_g2_cfg(cfg) -> list[dict[str, Any]]:
    benchmark_name = str(require_config_value(cfg, "benchmarks.name"))
    if benchmark_name == "mmlongbench":
        from benchmarks.utils.data_utils import load_mmlongbench_samples

        return load_mmlongbench_samples(require_config_value(cfg, "benchmarks.input_path"))
    if benchmark_name == "longdocurl":
        from benchmarks.utils.data_utils import load_longdocurl_samples

        return load_longdocurl_samples(
            require_config_value(cfg, "benchmarks.qa_file"),
            image_prefix=get_config_value(cfg, "benchmarks.image_prefix"),
        )
    raise ValueError(f"Unsupported benchmark for G2-Reader manifest: {benchmark_name}")


def build_g2_manifest_from_cfg(cfg) -> dict[str, Any]:
    benchmark_name = str(require_config_value(cfg, "benchmarks.name"))
    paths = G2ReaderPaths.from_cfg(cfg)
    return write_g2_manifest(benchmark_name, load_samples_for_g2_cfg(cfg), paths)


def read_g2_processed_jsonl(path: str | Path) -> list[dict[str, Any]]:
    records = []
    with Path(path).open("r", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                records.append(json.loads(line))
    return records


def _g2_worker_cfg(cfg):
    try:
        from omegaconf import OmegaConf

        if OmegaConf.is_config(cfg):
            return OmegaConf.to_container(cfg, resolve=True)
    except Exception:
        pass
    return cfg


def _tail_text(path: Path, max_chars: int = 4000) -> str:
    try:
        text = path.read_text(encoding="utf-8", errors="replace")
    except Exception:
        return ""
    if len(text) <= max_chars:
        return text
    return text[-max_chars:]


def _write_g2_subprocess_result(output_path: str | Path, *, ok: bool, result=None, error: str | None = None) -> None:
    payload = {"ok": ok}
    if ok:
        payload["result"] = result
    else:
        payload["error"] = error or "unknown error"
    Path(output_path).write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")


def _run_g2_sample_child(input_json: str | Path, output_json: str | Path) -> None:
    payload = json.loads(Path(input_json).read_text(encoding="utf-8"))
    cfg = payload["cfg"]
    benchmark_name = payload["benchmark_name"]
    sample = payload["sample"]
    try:
        result = G2ReaderRuntime(cfg).run_sample(benchmark_name, sample, client=None)
    except Exception as exc:
        _write_g2_subprocess_result(output_json, ok=False, error=f"{type(exc).__name__}: {exc}")
        raise
    _write_g2_subprocess_result(output_json, ok=True, result=result)


def run_g2_sample_in_subprocess(cfg, benchmark_name: str, sample: dict[str, Any]) -> dict[str, Any] | None:
    worker_cfg = _g2_worker_cfg(cfg)
    paths = G2ReaderPaths.from_cfg(worker_cfg)
    paths.ensure_dirs()
    safe_qid = re.sub(r"[^A-Za-z0-9._-]+", "_", str(sample.get("question_id", "sample"))).strip("._") or "sample"
    run_root = paths.logs_dir / "subprocess" / f"{benchmark_name}_{safe_qid}_{int(time.time() * 1000)}"
    run_root.mkdir(parents=True, exist_ok=True)
    input_path = run_root / "input.json"
    output_path = run_root / "output.json"
    log_path = run_root / "worker.log"
    input_path.write_text(
        json.dumps(
            {"cfg": worker_cfg, "benchmark_name": benchmark_name, "sample": sample},
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )
    cmd = [
        sys.executable,
        "-m",
        "baselines.g2reader",
        "run-sample",
        "--input-json",
        str(input_path),
        "--output-json",
        str(output_path),
    ]
    with log_path.open("w", encoding="utf-8") as log_handle:
        completed = subprocess.run(
            cmd,
            cwd=str(Path(__file__).resolve().parents[1]),
            stdout=log_handle,
            stderr=subprocess.STDOUT,
            text=True,
            check=False,
        )
    if not output_path.exists():
        tail = _tail_text(log_path)
        raise RuntimeError(
            f"G2-Reader subprocess did not write output for {safe_qid}; "
            f"returncode={completed.returncode}; log={log_path}\n{tail}"
        )
    payload = json.loads(output_path.read_text(encoding="utf-8"))
    if completed.returncode != 0 or not payload.get("ok"):
        tail = _tail_text(log_path)
        raise RuntimeError(
            f"G2-Reader subprocess failed for {safe_qid}; returncode={completed.returncode}; "
            f"error={payload.get('error')}; log={log_path}\n{tail}"
        )
    result = payload.get("result")
    if isinstance(result, dict):
        metadata = result.setdefault("metadata", {})
        metadata["inference_isolation"] = "process"
        metadata["g2_subprocess_dir"] = str(run_root)
        metadata["g2_subprocess_log"] = str(log_path)
    return result


def _build_g2_memory_worker(cfg, memory_id: str, record: dict[str, Any], evolve_iters: int) -> dict[str, Any]:
    runtime = G2ReaderRuntime(cfg)
    if g2_memory_ready(runtime.paths.memory_systems_dir, memory_id, evolve_iters=evolve_iters):
        return {"memory_id": memory_id, "status": "skipped"}
    try:
        runtime.build_memory_for_record(record, memory_id=memory_id)
    except Exception as exc:
        return {"memory_id": memory_id, "status": "failed", "error": str(exc)}
    if g2_memory_ready(runtime.paths.memory_systems_dir, memory_id, evolve_iters=evolve_iters):
        return {"memory_id": memory_id, "status": "built"}
    return {"memory_id": memory_id, "status": "failed", "error": "required cache files were not created"}


def build_g2_memory_from_cfg(
    cfg,
    processed_jsonl: str | Path | None = None,
    *,
    dry_run: bool = False,
    limit: int | None = None,
) -> dict[str, Any]:
    benchmark_name = str(require_config_value(cfg, "benchmarks.name"))
    paths = G2ReaderPaths.from_cfg(cfg)
    paths.ensure_dirs()
    processed_path = Path(processed_jsonl) if processed_jsonl else paths.manifest_dir / f"processed_{benchmark_name}.jsonl"
    records = read_g2_processed_jsonl(processed_path)
    if limit is not None:
        records = records[: int(limit)]
    memory_jobs = g2_memory_jobs_for_records(benchmark_name, records, cfg)
    runtime = G2ReaderRuntime(cfg)
    evolve_iters = int(get_config_value(cfg, "baselines.params.build_evolve_iters", 1))
    workers = int(get_config_value(cfg, "baselines.params.memory_build_workers", 1))
    if workers < 1:
        raise ValueError("baselines.params.memory_build_workers must be >= 1")
    built = 0
    skipped = 0
    pending = []
    failed = []
    interrupted = False
    show_progress = bool(get_config_value(cfg, "baselines.params.show_build_progress", True)) and not dry_run
    progress = tqdm(
        memory_jobs,
        total=len(memory_jobs),
        desc=f"G2 memory build ({benchmark_name})",
        unit="doc",
        dynamic_ncols=True,
        disable=not show_progress,
    )
    build_jobs = []
    for memory_id, record in progress:
        progress.set_postfix_str(f"built={built} skipped={skipped} failed={len(failed)}")
        if g2_memory_ready(paths.memory_systems_dir, memory_id, evolve_iters=evolve_iters):
            skipped += 1
            progress.set_postfix_str(f"built={built} skipped={skipped} failed={len(failed)}")
            continue
        if dry_run:
            pending.append(memory_id)
            continue
        if workers > 1:
            build_jobs.append((memory_id, record))
            continue
        runtime.build_memory_for_record(record, memory_id=memory_id)
        if g2_memory_ready(paths.memory_systems_dir, memory_id, evolve_iters=evolve_iters):
            built += 1
        else:
            failed.append(memory_id)
        progress.set_postfix_str(f"built={built} skipped={skipped} failed={len(failed)}")
    progress.close()

    if workers > 1 and build_jobs:
        worker_cfg = _g2_worker_cfg(cfg)
        progress = tqdm(
            total=len(build_jobs),
            desc=f"G2 memory build workers ({benchmark_name})",
            unit="doc",
            dynamic_ncols=True,
            disable=not show_progress,
        )
        with ProcessPoolExecutor(max_workers=workers) as executor:
            futures = {
                executor.submit(_build_g2_memory_worker, worker_cfg, memory_id, record, evolve_iters): memory_id
                for memory_id, record in build_jobs
            }
            for future in as_completed(futures):
                memory_id = futures[future]
                try:
                    result = future.result()
                except BrokenProcessPool as exc:
                    interrupted = True
                    pending.append(memory_id)
                    pending.extend(
                        other_id
                        for other_future, other_id in futures.items()
                        if other_future is not future and not other_future.done()
                    )
                    print(
                        "[WARNING] G2 memory worker pool stopped, likely because a worker process exited "
                        f"during CUDA/OOM handling: {exc}. Remaining docs stay pending for resume."
                    )
                    break
                except Exception as exc:
                    result = {"memory_id": memory_id, "status": "failed", "error": str(exc)}
                status = result.get("status")
                if status == "built":
                    built += 1
                elif status == "skipped":
                    skipped += 1
                else:
                    failed.append(str(result.get("memory_id") or memory_id))
                progress.set_postfix_str(f"built={built} skipped={skipped} failed={len(failed)}")
                progress.update(1)
        progress.close()

    if failed and not interrupted:
        raise RuntimeError(
            "G2-Reader memory build finished without required cache files for: "
            + ", ".join(failed[:20])
        )

    return {
        "benchmark": benchmark_name,
        "processed_jsonl": str(processed_path),
        "memory_systems_dir": str(paths.memory_systems_dir),
        "memory_cache_scope": g2_memory_cache_scope(cfg),
        "sample_count": len(records),
        "memory_system_count": len(memory_jobs),
        "total": len(memory_jobs),
        "built": built,
        "skipped": skipped,
        "failed": len(failed),
        "interrupted": interrupted,
        "workers": workers,
        "pending": len(pending),
        "pending_ids": pending[:20],
        "dry_run": dry_run,
    }

def g2_memory_ready(memory_systems_dir: str | Path, sample_id: str, evolve_iters: int = 1) -> bool:
    sample_dir = Path(memory_systems_dir) / f"{sample_id}_iter_{int(evolve_iters)}"
    return (sample_dir / "memories.pkl").exists() and (sample_dir / "retriever_embeddings.npy").exists()


def extract_g2_prediction(response: str | None) -> str | None:
    if response is None:
        return None
    text = str(response).strip()
    if not text:
        return None
    match = re.search(r"<output>(.*?)</output>", text, flags=re.DOTALL)
    return match.group(1).strip() if match else text


class LocalBGEM3Embedder:
    def __init__(self, model_path: str, *, use_fp16: bool = True, devices: Any = "cuda:0"):
        from FlagEmbedding import BGEM3FlagModel

        device_candidates = devices if isinstance(devices, list) else [devices]
        last_error = None
        for device in device_candidates:
            try:
                self.model = BGEM3FlagModel(model_path, use_fp16=use_fp16, devices=device)
                self.device = device
                return
            except Exception as error:
                last_error = error
                if not _is_g2_device_capacity_error(error):
                    raise
                print(f"[WARNING] G2 embedding model does not fit on {device}; trying next device.")
        raise RuntimeError(f"G2 embedding model failed to load on all configured devices: {device_candidates}") from last_error

    def encode(self, texts: list[str], *, batch_size: int = 8, max_length: int = 8192) -> list[list[float]]:
        current_batch_size = max(1, int(batch_size))
        last_error = None
        while current_batch_size >= 1:
            try:
                outputs = self.model.encode(
                    texts,
                    batch_size=current_batch_size,
                    max_length=max_length,
                    return_dense=True,
                    return_sparse=False,
                    return_colbert_vecs=False,
                )
                return np.asarray(outputs["dense_vecs"], dtype=np.float32).tolist()
            except Exception as error:
                last_error = error
                if not _is_g2_device_capacity_error(error) or current_batch_size == 1:
                    raise
                next_batch_size = max(1, current_batch_size // 2)
                print(
                    "[WARNING] G2 embedding OOM/capacity error; retrying with smaller batch size "
                    f"{current_batch_size} -> {next_batch_size}: {error}"
                )
                current_batch_size = next_batch_size
        raise last_error


def normalize_g2_devices(devices: Any):
    if devices is None:
        return "cuda:0"
    if isinstance(devices, str):
        parts = [part.strip() for part in devices.split(",") if part.strip()]
        if len(parts) > 1:
            return parts
        return parts[0] if parts else "cuda:0"
    if isinstance(devices, (list, tuple)):
        return [str(device).strip() for device in devices if str(device).strip()]
    return devices


def _is_g2_device_capacity_error(error: Exception) -> bool:
    text = str(error).lower()
    return any(pattern in text for pattern in ("out of memory", "cuda oom", "cublas", "memory"))


class G2ReaderRuntime:
    def __init__(self, cfg):
        self.cfg = cfg
        self.paths = G2ReaderPaths.from_cfg(cfg)
        self._embedder: LocalBGEM3Embedder | None = None

    @property
    def chat_model(self) -> str:
        return str(get_config_value(self.cfg, "baselines.models.chat", "Qwen3-VL-8B-Instruct"))

    @property
    def embed_model(self) -> str:
        return str(get_config_value(self.cfg, "baselines.models.embed", "/root/autodl-tmp/ylz/models/bge-m3"))

    def patch_g2_config(self):
        prepare_vendored_g2_imports()
        import config.config as g2_config

        self.paths.ensure_dirs()
        g2_config.LLM_BASE_URL = str(get_config_value(self.cfg, "litellm.base_url", "http://localhost:4000/v1"))
        g2_config.LLM_API_KEY = str(get_config_value(self.cfg, "litellm.api_key", "none"))
        g2_config.LLM_GENERATION["max_tokens"] = int(
            get_config_value(self.cfg, "baselines.params.memory_analysis_max_tokens", 32768)
        )
        g2_config.EMBED_BASE_URL = "local-bgem3"
        g2_config.EMBED_API_KEY = "local-bgem3"
        g2_config.MODELS["chat"] = self.chat_model
        g2_config.MODELS["embed"] = self.embed_model
        g2_config.MEMORY_SYSTEMS_DIR = str(self.paths.memory_systems_dir)
        g2_config.PDF_TMP_DIR = str(get_config_value(self.cfg, "baselines.paths.pdf_tmp_dir", "/tmp/g2_reader_pdfs"))
        g2_config.MAX_CONCURRENCY = int(get_config_value(self.cfg, "baselines.params.memory_llm_concurrency", 64))
        g2_config.PARALLEL_ANALYSIS = bool(get_config_value(self.cfg, "baselines.params.memory_parallel_analysis", True))
        g2_config.DATASETS.update(self._dataset_entries())
        return g2_config

    def patch_embedding(self) -> None:
        prepare_vendored_g2_imports()
        install_g2_lightweight_shims()
        import prebuild.amem_new as amem_new
        import prebuild.memory_layer as memory_layer

        self._patch_memory_build_concurrency(amem_new, memory_layer)

        async def embed_one(text: str, kind: str = "embedding") -> list[float]:
            return (await self._embed_many([text]))[0]

        async def embed_many(texts: list[str], kind: str = "embedding") -> list[list[float]]:
            return await self._embed_many(texts)

        def search_original(retriever, query: str, k: int = 5):
            query_embedding = self._embedder_instance().encode([query])[0]
            similarities = memory_layer.cosine_similarity([query_embedding], retriever.embeddings)[0]
            return np.argsort(similarities)[-k:][::-1]

        async def search(retriever, query: str, k: int = 5):
            return search_original(retriever, query, k=k)

        amem_new.embed_one = embed_one
        amem_new.embed_many = embed_many
        memory_layer.SimpleEmbeddingRetriever.search_original = search_original
        memory_layer.SimpleEmbeddingRetriever.search = search
        self._patch_analysis_generation(amem_new)
        self._patch_analysis_prompts(amem_new)
        self._patch_memory_evolve_signature(memory_layer)
        self._patch_memory_evolve_options(memory_layer)
        self._patch_llm_json_retry(amem_new)
        self._patch_text_analysis_split(amem_new)
        self._patch_analysis_failure_logging(amem_new)
        self._patch_mineru_chunking(amem_new)
        self._patch_mineru_path_resolution(amem_new)
        self._patch_construct_memory_cache_load(amem_new)

    def _patch_analysis_generation(self, amem_new) -> None:
        max_tokens = int(get_config_value(self.cfg, "baselines.params.memory_analysis_max_tokens", 32768))
        amem_new.LLM_GENERATION["max_tokens"] = max_tokens
        amem_new.LLM_GENERATION["temperature"] = 0.0

    def _patch_analysis_prompts(self, amem_new) -> None:
        if not bool(get_config_value(self.cfg, "baselines.params.memory_compact_analysis_prompts", True)):
            return
        if getattr(amem_new, "_project_compact_prompts_patched", False):
            return

        text_prompt = (
            "Extract compact retrieval metadata from the following text. Return only a valid JSON object "
            "with keys keywords, summary, and tags. Hard limits: keywords <= 16, tags <= 5, "
            "summary <= 60 words. Prefer exact entities, section names, table/figure labels, products, "
            "numbers with units, and distinctive terms. Do not copy long passages or repeat tokens.\n\n"
            "Text:\n"
        )
        image_prompt = (
            "Analyze the provided page/figure/table for retrieval. Use the surrounding context and caption only "
            "when helpful. Return only a valid JSON object with keys keywords, summary, text_content, and tags. "
            "Hard limits: keywords <= 16, tags <= 5, summary <= 70 words, text_content <= 800 characters. "
            "For text_content, include only the most useful readable labels, legends, table headers, values, "
            "and caption snippets; collapse repeated characters or repeated values as '[repeated text omitted]'. "
            "Do not transcribe long dotted leaders or repeated numbers. Escape quotes and newlines.\n\n"
            "Context: {context}\nCaption: {caption}"
        )
        amem_new.PROMPTS["text"] = text_prompt
        amem_new.PROMPTS["text_keyword"] = text_prompt
        amem_new.PROMPTS["image_ocr_keyword"] = image_prompt
        amem_new._project_compact_prompts_patched = True

    def _patch_text_analysis_split(self, amem_new) -> None:
        current = amem_new.analyze_content
        if getattr(current, "_project_split_patched", False):
            return

        original_analyze_content = current
        original_analyze_content_mineru = amem_new.analyze_content_mineru
        max_depth = int(get_config_value(self.cfg, "baselines.params.memory_analysis_split_max_depth", 4))
        min_chars = int(get_config_value(self.cfg, "baselines.params.memory_analysis_split_min_chars", 400))
        max_summary_chars = int(get_config_value(self.cfg, "baselines.params.memory_analysis_max_summary_chars", 700))
        max_keywords = int(get_config_value(self.cfg, "baselines.params.memory_analysis_max_keywords", 24))

        async def analyze_text_with_split(payload: str, analyze_part, *, depth: int = 0):
            try:
                return await analyze_part(payload)
            except Exception as error:
                text = "" if payload is None else str(payload)
                if depth >= max_depth or len(text) <= min_chars:
                    raise
                left, right = self._split_text_for_analysis_retry(text)
                if not left or not right:
                    raise
                print(
                    "[WARNING] text analysis failed; splitting chunk "
                    f"depth={depth + 1}/{max_depth}, chars={len(text)}: {error}"
                )
                parts = [
                    await analyze_text_with_split(part, analyze_part, depth=depth + 1)
                    for part in (left, right)
                ]
                return self._merge_analysis_parts(
                    parts,
                    max_summary_chars=max_summary_chars,
                    max_keywords=max_keywords,
                )

        async def analyze_content_split(payload: str, *, modality: str):
            if modality != "text":
                return await original_analyze_content(payload, modality=modality)

            async def analyze_part(part: str):
                return await original_analyze_content(part, modality="text")

            return await analyze_text_with_split(payload, analyze_part)

        async def analyze_content_mineru_split(payload: str, *, modality: str, context: str = "", caption: str = ""):
            if modality != "text":
                return await original_analyze_content_mineru(
                    payload,
                    modality=modality,
                    context=context,
                    caption=caption,
                )

            async def analyze_part(part: str):
                return await original_analyze_content_mineru(
                    part,
                    modality="text",
                    context=context,
                    caption=caption,
                )

            return await analyze_text_with_split(payload, analyze_part)

        analyze_content_split._project_split_patched = True
        analyze_content_split._project_original_analyze_content = original_analyze_content
        analyze_content_mineru_split._project_split_patched = True
        analyze_content_mineru_split._project_original_analyze_content_mineru = original_analyze_content_mineru
        amem_new.analyze_content = analyze_content_split
        amem_new.analyze_content_mineru = analyze_content_mineru_split

    def _split_text_for_analysis_retry(self, text: str) -> tuple[str, str]:
        midpoint = max(1, len(text) // 2)
        candidates = []
        for sep in ("\n\n", "\n", ". ", "; ", " "):
            left_idx = text.rfind(sep, 0, midpoint)
            right_idx = text.find(sep, midpoint)
            if left_idx > 0:
                candidates.append(left_idx + len(sep))
            if right_idx > 0:
                candidates.append(right_idx + len(sep))
        split_at = min(candidates, key=lambda idx: abs(idx - midpoint)) if candidates else midpoint
        left = text[:split_at].strip()
        right = text[split_at:].strip()
        return left, right

    def _merge_analysis_parts(
        self,
        parts: list[dict[str, Any]],
        *,
        max_summary_chars: int,
        max_keywords: int,
    ) -> dict[str, Any]:
        keywords: list[str] = []
        tags: list[str] = []
        summaries: list[str] = []
        for part in parts:
            if not isinstance(part, dict):
                continue
            summary = str(part.get("summary", "")).strip()
            if summary:
                summaries.append(summary)
            for key in part.get("keywords", []) or []:
                text = str(key).strip()
                if text and text not in keywords:
                    keywords.append(text)
            for tag in part.get("tags", []) or []:
                text = str(tag).strip()
                if text and text not in tags:
                    tags.append(text)
        summary = " ".join(summaries).strip()
        if len(summary) > max_summary_chars:
            summary = summary[: max(0, max_summary_chars - 3)].rstrip() + "..."
        return {
            "keywords": keywords[:max_keywords] or ["unknown"],
            "summary": summary or "content analysis failed",
            "tags": tags[:5] or ["text"],
        }

    def _patch_llm_json_retry(self, amem_new) -> None:
        current = amem_new.call_llm_json
        original_call = getattr(current, "_project_original_call_llm_json", current)
        retries = int(get_config_value(self.cfg, "baselines.params.memory_analysis_retries", 2))

        async def call_llm_json_retry(system: str, user_payload, *, is_multimodal: bool = False):
            last_error = None
            retry_system = system
            for attempt in range(max(1, retries + 1)):
                try:
                    return await original_call(retry_system, user_payload, is_multimodal=is_multimodal)
                except Exception as error:
                    last_error = error
                    if attempt >= retries:
                        break
                    required_keys = "keywords, summary, tags"
                    if is_multimodal:
                        required_keys += ", text_content"
                    retry_system = (
                        system
                        + f"\nReturn a compact valid JSON object only. Required keys: {required_keys}. "
                        + "Keep summary under 80 words, text_content under 800 characters, "
                        + "and each keyword/tag short. Do not repeat tokens or long OCR runs."
                    )
                    print(f"[WARNING] JSON analysis failed, retrying {attempt + 1}/{retries}: {error}")
                    await amem_new.asyncio.sleep(1 + attempt)
            raise last_error

        call_llm_json_retry._project_original_call_llm_json = original_call
        call_llm_json_retry._project_retries = retries
        amem_new.call_llm_json = call_llm_json_retry

    def _patch_analysis_failure_logging(self, amem_new) -> None:
        current = amem_new.analyze_content
        if getattr(current, "_project_failure_logger_patched", False):
            return

        original_analyze_content = current
        original_analyze_content_mineru = amem_new.analyze_content_mineru

        async def analyze_content_logged(payload: str, *, modality: str):
            try:
                return await original_analyze_content(payload, modality=modality)
            except Exception as error:
                self._log_analysis_failure(amem_new, "text_or_image_analysis", modality, payload, error)
                raise

        async def analyze_content_mineru_logged(payload: str, *, modality: str, context: str = "", caption: str = ""):
            try:
                return await original_analyze_content_mineru(
                    payload,
                    modality=modality,
                    context=context,
                    caption=caption,
                )
            except Exception as error:
                self._log_analysis_failure(
                    amem_new,
                    "mineru_analysis",
                    modality,
                    payload,
                    error,
                    context=context,
                    caption=caption,
                )
                raise

        analyze_content_logged._project_failure_logger_patched = True
        analyze_content_logged._project_split_patched = getattr(original_analyze_content, "_project_split_patched", False)
        analyze_content_logged._project_original_analyze_content = original_analyze_content
        analyze_content_mineru_logged._project_failure_logger_patched = True
        analyze_content_mineru_logged._project_split_patched = getattr(
            original_analyze_content_mineru,
            "_project_split_patched",
            False,
        )
        analyze_content_mineru_logged._project_original_analyze_content_mineru = original_analyze_content_mineru
        amem_new.analyze_content = analyze_content_logged
        amem_new.analyze_content_mineru = analyze_content_mineru_logged

    def _log_analysis_failure(
        self,
        amem_new,
        stage: str,
        modality: str,
        payload: str,
        error: Exception,
        *,
        context: str = "",
        caption: str = "",
    ) -> None:
        import hashlib
        from datetime import datetime

        payload_text = "" if payload is None else str(payload)
        log_dir = self.paths.memory_systems_dir / "_debug_logs"
        log_dir.mkdir(parents=True, exist_ok=True)
        record = {
            "time": datetime.now().isoformat(),
            "benchmark": str(require_config_value(self.cfg, "benchmarks.name")),
            "memory_id": getattr(amem_new, "_project_current_memory_id", None),
            "stage": stage,
            "modality": modality,
            "error_type": type(error).__name__,
            "error": str(error),
            "payload_sha1": hashlib.sha1(payload_text.encode("utf-8", errors="ignore")).hexdigest(),
            "payload_chars": len(payload_text),
        }
        if modality == "text":
            record["payload"] = payload_text
        else:
            record["context"] = context
            record["caption"] = caption
        with (log_dir / "failed_chunks.jsonl").open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")

    def _patch_mineru_chunking(self, amem_new) -> None:
        current = amem_new.extract_chunk_from_mineru
        original_extract = getattr(current, "_project_original_extract_chunk", current)
        chunk_size = int(get_config_value(self.cfg, "baselines.params.memory_chunk_size", 1200))
        chunk_overlap = int(get_config_value(self.cfg, "baselines.params.memory_chunk_overlap", 120))

        def extract_chunk_from_mineru(mineru_path, chunk_size=None, chunk_overlap=None):
            return original_extract(
                mineru_path,
                chunk_size=chunk_size if chunk_size is not None else chunk_size_value,
                chunk_overlap=chunk_overlap if chunk_overlap is not None else chunk_overlap_value,
            )

        chunk_size_value = chunk_size
        chunk_overlap_value = chunk_overlap
        extract_chunk_from_mineru._project_original_extract_chunk = original_extract
        extract_chunk_from_mineru._project_chunk_size = chunk_size_value
        extract_chunk_from_mineru._project_chunk_overlap = chunk_overlap_value
        amem_new.extract_chunk_from_mineru = extract_chunk_from_mineru

    def _patch_memory_build_concurrency(self, amem_new, memory_layer) -> None:
        concurrency = int(get_config_value(self.cfg, "baselines.params.memory_llm_concurrency", 64))
        if concurrency < 1:
            raise ValueError("baselines.params.memory_llm_concurrency must be >= 1")
        parallel_analysis = bool(get_config_value(self.cfg, "baselines.params.memory_parallel_analysis", True))

        amem_new.MAX_CONCURRENCY = concurrency
        amem_new.PARALLEL_ANALYSIS = parallel_analysis
        memory_layer.MAX_CONCURRENCY = concurrency

        semaphores = getattr(amem_new, "_llm_sem_by_loop", None)
        if isinstance(semaphores, dict):
            semaphores.clear()

    def _patch_mineru_path_resolution(self, amem_new) -> None:
        current = amem_new.resolve_docs_from_dataset_mineru
        if getattr(current, "_project_mineru_resolution_patched", False):
            return

        def has_mineru_payload(path: str) -> bool:
            try:
                entries = amem_new.os.listdir(path)
            except Exception:
                return False
            return any(item.endswith(".md") for item in entries) and any(
                item.endswith("_content_list.json") for item in entries
            )

        def resolve_docs_from_dataset_mineru(dataset_name: str, q_id: str, limit: int = 5):
            cfg = amem_new.DATASETS[dataset_name]
            df = amem_new.load_dataset_df(dataset_name)
            key, docs_col = cfg["key"], cfg["docs_col"]
            benchmark_name = str(dataset_name).replace("g2_", "", 1)
            matches = amem_new.np.where(df[key] == q_id)[0]
            if len(matches) == 0:
                doc_names_raw = None
                for raw_documents in df[docs_col].dropna().tolist():
                    for document in ast.literal_eval(str(raw_documents)):
                        if g2_document_memory_id(benchmark_name, str(document)) == q_id:
                            doc_names_raw = [str(document)]
                            break
                    if doc_names_raw is not None:
                        break
                if doc_names_raw is None:
                    raise ValueError(
                        f"error: data not found in {dataset_name}.csv for {key}='{q_id}'.\n"
                        f"available {key} values: {df[key].unique()[:10].tolist()}..."
                    )
            else:
                row = df.iloc[matches[0]].to_dict()
                doc_names_raw = list(ast.literal_eval(str(row[docs_col])))[:limit]

            base_dir = cfg["mineru_dir"]
            doc_names = [amem_new.os.path.splitext(doc)[0] for doc in doc_names_raw]
            mineru_paths = []
            for doc in doc_names:
                path = amem_new.find_actual_file(base_dir, doc)
                if not path:
                    print(f"skip file not found: {doc}")
                    continue
                for _ in range(2):
                    if has_mineru_payload(path):
                        break
                    subs = [
                        item
                        for item in amem_new.os.listdir(path)
                        if amem_new.os.path.isdir(amem_new.os.path.join(path, item))
                    ]
                    if len(subs) != 1:
                        break
                    path = amem_new.os.path.join(path, subs[0])
                mineru_paths.append(path)

            print(f"found {len(mineru_paths)}/{len(doc_names)} files")
            return base_dir, mineru_paths

        resolve_docs_from_dataset_mineru._project_mineru_resolution_patched = True
        resolve_docs_from_dataset_mineru._project_original_resolver = current
        amem_new.resolve_docs_from_dataset_mineru = resolve_docs_from_dataset_mineru

    def _patch_construct_memory_cache_load(self, amem_new) -> None:
        current = amem_new.construct_memory
        if getattr(current, "_project_cache_load_patched", False):
            return

        original_construct_memory = current
        paths = self.paths

        def cache_ready(iter_name: str) -> bool:
            sample_dir = paths.memory_systems_dir / iter_name
            return (sample_dir / "memories.pkl").exists() and (sample_dir / "retriever_embeddings.npy").exists()

        async def construct_memory_cache_aware(pdf_path, evolve_iters=1, window_size=2):
            sample_id = str(pdf_path)
            memory_id = self._memory_id_for_g2_link(sample_id, amem_new)
            target_iter = int(evolve_iters)
            iter_name = f"{memory_id}_iter_{target_iter}"
            if cache_ready(iter_name):
                memory_system = amem_new.AgenticMemorySystem(
                    model_name=amem_new.MODELS["embed"],
                    llm_model=amem_new.MODELS["chat"],
                )
                memory_system.load_memory_system(iter_name)
                return memory_system

            resume_iter = None
            for candidate_iter in range(target_iter - 1, -1, -1):
                candidate_name = f"{memory_id}_iter_{candidate_iter}"
                if cache_ready(candidate_name):
                    resume_iter = candidate_iter
                    break
            if resume_iter is not None:
                memory_system = amem_new.AgenticMemorySystem(
                    model_name=amem_new.MODELS["embed"],
                    llm_model=amem_new.MODELS["chat"],
                )
                memory_system.load_memory_system(f"{memory_id}_iter_{resume_iter}")
                max_evolution_memories = int(
                    get_config_value(self.cfg, "baselines.params.memory_evolution_max_memories", 12000)
                )
                memory_count = len(getattr(memory_system, "memories", {}) or {})
                skip_evolution = max_evolution_memories > 0 and memory_count > max_evolution_memories
                if skip_evolution:
                    print(
                        "[WARNING] G2 memory evolution skipped for oversized memory "
                        f"{memory_id}: memories={memory_count}, threshold={max_evolution_memories}. "
                        "Promoting cached iter to target iter."
                    )
                    for iteration in range(resume_iter, target_iter):
                        memory_system.save_memory_system(f"{memory_id}_iter_{iteration + 1}")
                    self._write_build_metadata(
                        memory_id,
                        evolve_iters=target_iter,
                        window_size=int(get_config_value(self.cfg, "baselines.params.build_window_size", 2)),
                        extra={
                            "evolution_skipped": True,
                            "skip_reason": "oversized_memory",
                            "memory_count": memory_count,
                            "memory_evolution_max_memories": max_evolution_memories,
                            "promoted_from_iter": resume_iter,
                        },
                    )
                    return memory_system
                for iteration in range(resume_iter, target_iter):
                    _ = await memory_system.process_memory_all()
                    meta = [
                        note.context + " keywords: " + ", ".join(note.keywords)
                        for note in memory_system.memories.values()
                    ]
                    pre_embeddings = await amem_new.embed_many(meta, kind="re_embedding")
                    memory_system.reset_retriever()
                    memory_system.retriever.add_documents(meta, pre_embeddings)
                    for index, note in enumerate(list(memory_system.memories.values())):
                        note.links = [link for link in note.links if link != index]
                    memory_system.save_memory_system(f"{memory_id}_iter_{iteration + 1}")
                return memory_system
            return await original_construct_memory(
                memory_id,
                evolve_iters=evolve_iters,
                window_size=window_size,
            )

        construct_memory_cache_aware._project_cache_load_patched = True
        construct_memory_cache_aware._project_original_construct_memory = original_construct_memory
        amem_new.construct_memory = construct_memory_cache_aware

    def _patch_memory_evolve_signature(self, memory_layer) -> None:
        current = memory_layer.AgenticMemorySystem._call_llm_evolve
        if getattr(current, "_project_signature_patched", False):
            return

        original_call = current

        async def call_llm_evolve_compat(memory_system, *args, **kwargs):
            kwargs.pop("max_tokens", None)
            return await original_call(memory_system, *args, **kwargs)

        call_llm_evolve_compat._project_signature_patched = True
        call_llm_evolve_compat._project_original_call = original_call
        memory_layer.AgenticMemorySystem._call_llm_evolve = call_llm_evolve_compat

    def _patch_memory_evolve_options(self, memory_layer) -> None:
        current = memory_layer.AgenticMemorySystem.find_related_notes
        if getattr(current, "_project_evolve_options_patched", False):
            return

        original_find_related_notes = current
        neighbor_k = int(get_config_value(self.cfg, "baselines.params.memory_evolution_neighbor_k", 3))

        async def find_related_notes_limited(memory_system, query: str, k: int = 5, include_neighbors: bool = True, modality: str = "all"):
            if not include_neighbors and neighbor_k > 0:
                k = min(int(k), neighbor_k)
            return await original_find_related_notes(
                memory_system,
                query,
                k=k,
                include_neighbors=include_neighbors,
                modality=modality,
            )

        find_related_notes_limited._project_evolve_options_patched = True
        find_related_notes_limited._project_original_find_related_notes = original_find_related_notes
        find_related_notes_limited._project_neighbor_k = neighbor_k
        memory_layer.AgenticMemorySystem.find_related_notes = find_related_notes_limited

    def _memory_id_for_record(self, record: dict[str, Any]) -> str:
        benchmark_name = str(require_config_value(self.cfg, "benchmarks.name"))
        return g2_memory_id_for_record(benchmark_name, record, self.cfg)

    def _memory_id_for_g2_link(self, link: str, amem_new) -> str:
        benchmark_name = str(require_config_value(self.cfg, "benchmarks.name"))
        if g2_memory_cache_scope(self.cfg) != "document":
            return str(link)
        doc_prefix = f"g2_{benchmark_name}_doc_"
        if str(link).startswith(doc_prefix):
            return str(link)
        dataset_name = f"g2_{benchmark_name}"
        try:
            cfg = amem_new.DATASETS[dataset_name]
            df = amem_new.load_dataset_df(dataset_name)
            matches = amem_new.np.where(df[cfg["key"]] == str(link))[0]
            if len(matches) == 0:
                return str(link)
            row = df.iloc[matches[0]].to_dict()
            documents = ast.literal_eval(str(row[cfg["docs_col"]]))
            if len(documents) != 1:
                return str(link)
            return g2_document_memory_id(benchmark_name, str(documents[0]))
        except Exception:
            return str(link)

    def run_sample(self, benchmark_name: str, sample: dict[str, Any], client=None) -> dict[str, Any] | None:
        self.patch_g2_config()
        self.patch_embedding()
        record = g2_processed_record(benchmark_name, sample)
        run_dir = self.paths.logs_dir / "runs" / f"{record['_id']}_{int(time.time() * 1000)}"
        run_dir.mkdir(parents=True, exist_ok=True)
        data_path = run_dir / "input.jsonl"
        data_path.write_text(json.dumps(record, ensure_ascii=False) + "\n", encoding="utf-8")
        self._ensure_memory_ready(record)

        args = self._dag_args(data_path, run_dir)
        prepare_vendored_g2_imports()
        install_g2_lightweight_shims()
        from agent_search.pred_kw import DAGPred

        DAGPred(args).main()
        output = self._read_single_output(run_dir, args)
        if output is None:
            return None
        response = output.get("response")
        return {
            "response": response,
            "pred": output.get("pred") or extract_g2_prediction(response),
            "pred_format": None,
            "usage": output.get("usage"),
            "metadata": {
                "context_builder": "g2-reader",
                "model": self.chat_model,
                "sample_key": sample.get("question_id"),
                "g2_id": record["_id"],
                "g2_run_dir": str(run_dir),
                "process_time": output.get("process_time"),
            },
        }

    def _ensure_memory_ready(self, record: dict[str, Any]) -> None:
        evolve_iters = int(get_config_value(self.cfg, "baselines.params.build_evolve_iters", 1))
        sample_id = str(record["_id"])
        memory_id = self._memory_id_for_record(record)
        if g2_memory_ready(self.paths.memory_systems_dir, memory_id, evolve_iters=evolve_iters):
            return
        if bool(get_config_value(self.cfg, "baselines.runtime.allow_runtime_memory_build", False)):
            return
        raise RuntimeError(
            "G2-Reader memory cache is missing for "
            f"sample {sample_id} (memory {memory_id}). Build it first under {self.paths.memory_systems_dir} "
            "or set baselines.runtime.allow_runtime_memory_build=true for a deliberate lazy build."
        )

    def build_memory_for_record(self, record: dict[str, Any], memory_id: str | None = None) -> None:
        self.patch_g2_config()
        self.patch_embedding()
        prepare_vendored_g2_imports()
        install_g2_lightweight_shims()
        import prebuild.amem_new as amem_new

        evolve_iters = int(get_config_value(self.cfg, "baselines.params.build_evolve_iters", 1))
        window_size = int(get_config_value(self.cfg, "baselines.params.build_window_size", 2))
        memory_id = memory_id or self._memory_id_for_record(record)
        previous_memory_id = getattr(amem_new, "_project_current_memory_id", None)
        amem_new._project_current_memory_id = memory_id
        try:
            asyncio.run(amem_new.construct_memory(memory_id, evolve_iters=evolve_iters, window_size=window_size))
            self._write_build_metadata(memory_id, evolve_iters=evolve_iters, window_size=window_size)
        finally:
            if previous_memory_id is None:
                try:
                    delattr(amem_new, "_project_current_memory_id")
                except AttributeError:
                    pass
            else:
                amem_new._project_current_memory_id = previous_memory_id

    def _write_build_metadata(self, memory_id: str, *, evolve_iters: int, window_size: int, extra: dict[str, Any] | None = None) -> None:
        if not g2_memory_ready(self.paths.memory_systems_dir, memory_id, evolve_iters=evolve_iters):
            return
        sample_dir = self.paths.memory_systems_dir / f"{memory_id}_iter_{int(evolve_iters)}"
        params = get_config_value(self.cfg, "baselines.params", {}) or {}
        metadata = {
            "memory_id": memory_id,
            "benchmark": str(require_config_value(self.cfg, "benchmarks.name")),
            "chat_model": self.chat_model,
            "embed_model": self.embed_model,
            "memory_cache_scope": g2_memory_cache_scope(self.cfg),
            "evolve_iters": int(evolve_iters),
            "window_size": int(window_size),
            "build_params": {
                key: params.get(key)
                for key in (
                    "memory_chunk_size",
                    "memory_chunk_overlap",
                    "memory_analysis_retries",
                    "memory_analysis_max_tokens",
                    "memory_analysis_split_max_depth",
                    "memory_analysis_split_min_chars",
                    "memory_compact_analysis_prompts",
                    "memory_build_workers",
                    "memory_llm_concurrency",
                    "memory_parallel_analysis",
                    "memory_evolution_neighbor_k",
                )
            },
            "created_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        }
        with (sample_dir / "build_metadata.json").open("w", encoding="utf-8") as handle:
            json.dump(metadata, handle, ensure_ascii=False, indent=2)

    def _dataset_entries(self) -> dict[str, dict[str, str]]:
        benchmark_name = str(require_config_value(self.cfg, "benchmarks.name"))
        mineru_dir = str(require_config_value(self.cfg, "benchmarks.mineru_dir"))
        return {
            f"g2_{benchmark_name}": {
                "csv": str(self.paths.manifest_dir / f"{benchmark_name}.csv"),
                "mineru_dir": mineru_dir,
                "key": "q_id",
                "docs_col": "documents",
            }
        }

    def _dag_args(self, data_path: Path, run_dir: Path) -> Namespace:
        params = get_config_value(self.cfg, "baselines.params", {}) or {}
        return Namespace(
            save_dir=str(run_dir),
            model=self.chat_model,
            n_proc=1,
            data_path=str(data_path),
            method="dag",
            rag=1,
            self_refine=False,
            top_k=int(get_config_value(params, "top_k", 5)),
            num_runs=int(get_config_value(params, "num_runs", 1)),
            judge_use=False,
            debug=True,
            top_k_text_kw=int(get_config_value(params, "top_k_text_kw", 3)),
            top_k_image_kw=int(get_config_value(params, "top_k_image_kw", 3)),
            keyword_modality=str(get_config_value(params, "keyword_modality", "all")),
            keyword=True,
            use_dag=bool(get_config_value(params, "use_dag", True)),
        )

    def _read_single_output(self, run_dir: Path, args: Namespace) -> dict[str, Any] | None:
        base_filename = f"{args.model.split('/')[-1]}_{args.method}_rag_{args.rag}"
        output_path = run_dir / f"{base_filename}.jsonl"
        if not output_path.exists():
            return None
        rows = [json.loads(line) for line in output_path.read_text(encoding="utf-8").splitlines() if line.strip()]
        return rows[-1] if rows else None

    def _embedder_instance(self) -> LocalBGEM3Embedder:
        if self._embedder is None:
            self._embedder = LocalBGEM3Embedder(
                self.embed_model,
                use_fp16=bool(get_config_value(self.cfg, "baselines.use_fp16", True)),
                devices=normalize_g2_devices(get_config_value(self.cfg, "baselines.devices", "cuda:0")),
            )
        return self._embedder

    async def _embed_many(self, texts: list[str]) -> list[list[float]]:
        batch_size = int(get_config_value(self.cfg, "baselines.batch_size", 8))
        max_length = int(get_config_value(self.cfg, "baselines.max_length", 8192))
        return self._embedder_instance().encode(
            ["" if text is None else str(text) for text in texts],
            batch_size=batch_size,
            max_length=max_length,
        )


def g2_dependency_preflight() -> dict[str, Any]:
    import importlib.util

    checks = {
        "memory_build_required": [
            "numpy",
            "pandas",
            "openai",
            "sklearn",
            "tqdm",
            "PIL",
            "FlagEmbedding",
        ],
        "inference_required": [
            "tiktoken",
            "openai",
            "tqdm",
        ],
        "shimmed_by_wrapper": [
            "langchain_text_splitters",
            "utils.visdom_utils",
        ],
        "g2_pdf_fallback_not_required": [
            "PyPDF2",
            "pdf2image",
            "pytesseract",
        ],
    }
    result = {"ok": True, "checks": {}}
    for group, modules in checks.items():
        rows = []
        for module in modules:
            if group == "shimmed_by_wrapper":
                present = True
            else:
                present = importlib.util.find_spec(module) is not None
            rows.append({"module": module, "present": present})
            if group in {"memory_build_required", "inference_required"} and not present:
                result["ok"] = False
        result["checks"][group] = rows
    return result



def _default_config_file() -> Path:
    return Path(__file__).resolve().parents[1] / "configs" / "config.yaml"


def _cfg_from_json_or_file(
    config_json: str | None,
    config_file: str | None,
    overrides: list[str] | None = None,
):
    from omegaconf import OmegaConf

    overrides = list(overrides or [])
    if config_json:
        cfg = OmegaConf.create(json.loads(config_json))
        dot_overrides = [item for item in overrides if "=" in item and not item.startswith(("benchmarks=", "baselines=", "ablations="))]
        if dot_overrides:
            cfg = OmegaConf.merge(cfg, OmegaConf.from_dotlist(dot_overrides))
        return cfg

    from hydra import compose, initialize_config_dir

    config_path = Path(config_file) if config_file else _default_config_file()
    config_path = config_path.resolve()
    config_dir = str(config_path.parent)
    config_name = config_path.stem
    with initialize_config_dir(version_base="1.3", config_dir=config_dir):
        return compose(config_name=config_name, overrides=overrides)


def main() -> None:
    parser = argparse.ArgumentParser(description="Project-side G2-Reader baseline utilities")
    subparsers = parser.add_subparsers(dest="command", required=True)

    manifest_parser = subparsers.add_parser("build-manifest")
    manifest_parser.add_argument("--config-json")
    manifest_parser.add_argument("--config-file")
    manifest_parser.add_argument("overrides", nargs=argparse.REMAINDER)

    memory_parser = subparsers.add_parser("build-memory")
    memory_parser.add_argument("--config-json")
    memory_parser.add_argument("--config-file")
    memory_parser.add_argument("--processed-jsonl")
    memory_parser.add_argument("--dry-run", action="store_true")
    memory_parser.add_argument("--limit", type=int)
    memory_parser.add_argument("overrides", nargs=argparse.REMAINDER)

    preflight_parser = subparsers.add_parser("preflight")
    preflight_parser.add_argument("overrides", nargs=argparse.REMAINDER)

    sample_parser = subparsers.add_parser("run-sample")
    sample_parser.add_argument("--input-json", required=True)
    sample_parser.add_argument("--output-json", required=True)

    args = parser.parse_args()
    if args.command == "preflight":
        print(json.dumps(g2_dependency_preflight(), ensure_ascii=False, indent=2))
        return
    if args.command == "run-sample":
        _run_g2_sample_child(args.input_json, args.output_json)
        return

    cfg = _cfg_from_json_or_file(args.config_json, args.config_file, getattr(args, "overrides", []))
    if args.command == "build-manifest":
        result = build_g2_manifest_from_cfg(cfg)
    elif args.command == "build-memory":
        result = build_g2_memory_from_cfg(
            cfg,
            processed_jsonl=args.processed_jsonl,
            dry_run=args.dry_run,
            limit=args.limit,
        )
    else:
        raise ValueError(f"Unsupported command: {args.command}")
    print(json.dumps(result, ensure_ascii=False, indent=2))



class G2ReaderContextBuilder(ContextBuilder):
    name = "g2-reader"

    def run_sample(self, benchmark_name, sample, client=None):
        isolation = str(get_config_value(self.cfg, "baselines.params.inference_isolation", "process")).lower()
        if isolation in {"process", "subprocess"}:
            return run_g2_sample_in_subprocess(self.cfg, benchmark_name, sample)
        if isolation in {"inprocess", "none", "false"}:
            return G2ReaderRuntime(self.cfg).run_sample(benchmark_name, sample, client=client)
        raise ValueError(f"Unsupported G2-Reader inference_isolation: {isolation}")


if __name__ == "__main__":
    main()
