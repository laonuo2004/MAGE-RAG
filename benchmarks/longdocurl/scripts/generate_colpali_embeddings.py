#!/usr/bin/env python3
import argparse
import json
import logging
import re
import sys
from pathlib import Path

from PIL import Image
from tqdm import tqdm


CODE_DIR = Path(__file__).resolve().parents[3]
M3DOCRAG_SRC = CODE_DIR / "baselines" / "m3docrag" / "src"
sys.path.insert(0, str(M3DOCRAG_SRC))


DEFAULT_INPUT_PATH = CODE_DIR / "benchmarks" / "longdocurl" / "data" / "LongDocURL.jsonl"
DEFAULT_IMAGE_ROOT = CODE_DIR / "benchmarks" / "longdocurl" / "data" / "pdf_pngs" / "4000-4999"
DEFAULT_OUTPUT_DIR = CODE_DIR / "benchmarks" / "longdocurl" / "data" / "pdf_embeddings" / "4000-4999"
DEFAULT_BACKBONE_PATH = "/root/autodl-tmp/ylz/models/colpaligemma-3b-mix-448-base"
DEFAULT_ADAPTER_PATH = "/root/autodl-tmp/ylz/models/colpali-v1.2"

logger = logging.getLogger("generate_longdocurl_colpali_embeddings")
PAGE_RE = re.compile(r"^(?P<doc_id>.+)_(?P<page_idx>\d+)\.png$")


def parse_args():
    parser = argparse.ArgumentParser(
        description="Generate one ColPali document embedding safetensors file per LongDocURL PDF."
    )
    parser.add_argument("--input-path", default=str(DEFAULT_INPUT_PATH), help="LongDocURL.jsonl path.")
    parser.add_argument("--image-root", default=str(DEFAULT_IMAGE_ROOT), help="Root PNG directory.")
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT_DIR), help="Directory for .safetensors outputs.")
    parser.add_argument("--backbone-path", default=DEFAULT_BACKBONE_PATH, help="ColPali backbone checkpoint path.")
    parser.add_argument("--adapter-path", default=DEFAULT_ADAPTER_PATH, help="ColPali adapter checkpoint path.")
    parser.add_argument("--device", default="auto", help="Device for the ColPali model: auto, cpu, cuda, cuda:0, etc.")
    parser.add_argument("--batch-size", type=int, default=4, help="Image encoding batch size.")
    parser.add_argument("--limit", type=int, default=None, help="Only process the first N selected documents.")
    parser.add_argument("--doc-id", action="append", default=None, help="Specific doc_no to process. Can be repeated.")
    parser.add_argument("--overwrite", action="store_true", help="Regenerate embeddings that already exist.")
    return parser.parse_args()


def load_doc_ids(input_path, explicit_doc_ids):
    if explicit_doc_ids:
        return sorted(set(explicit_doc_ids))

    doc_ids = set()
    with open(input_path, "r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            sample = json.loads(line)
            doc_ids.add(sample["doc_no"])
    return sorted(doc_ids)


def find_page_images(image_root, doc_id):
    page_dir = image_root / doc_id[:4]
    if not page_dir.exists():
        raise FileNotFoundError(f"Missing PNG directory for doc_id={doc_id}: {page_dir}")

    pages = []
    for path in page_dir.glob(f"{doc_id}_*.png"):
        match = PAGE_RE.match(path.name)
        if not match or match.group("doc_id") != doc_id:
            continue
        pages.append((int(match.group("page_idx")), path))

    if not pages:
        raise FileNotFoundError(f"No PNG pages found for doc_id={doc_id} in {page_dir}")

    pages.sort(key=lambda item: item[0])
    page_indices = [idx for idx, _ in pages]
    expected = list(range(page_indices[-1] + 1))
    if page_indices != expected:
        missing = sorted(set(expected) - set(page_indices))
        raise ValueError(f"Non-contiguous pages for doc_id={doc_id}; missing 0-based page indices: {missing}")
    return pages


def load_images(page_paths):
    images = []
    for path in page_paths:
        with Image.open(path) as image:
            images.append(image.convert("RGB"))
    return images


def write_manifest_record(manifest_path, record):
    with open(manifest_path, "a", encoding="utf-8") as f:
        f.write(json.dumps(record, ensure_ascii=False) + "\n")


def resolve_device(device):
    if device != "auto":
        return device
    import torch

    return "cuda" if torch.cuda.is_available() else "cpu"


def load_retrieval_model(backbone_path, adapter_path, device):
    from m3docrag.retrieval import ColPaliRetrievalModel

    retrieval_model = ColPaliRetrievalModel(
        backbone_name_or_path=backbone_path,
        adapter_name_or_path=adapter_path,
    )
    target_device = resolve_device(device)
    retrieval_model.model.to(target_device)
    logger.info("Loaded ColPali model on device=%s", target_device)
    return retrieval_model


def main():
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    args = parse_args()

    image_root = Path(args.image_root)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = output_dir / "manifest.jsonl"

    doc_ids = load_doc_ids(args.input_path, args.doc_id)
    if args.limit is not None:
        doc_ids = doc_ids[: args.limit]
    logger.info("Selected %s LongDocURL documents.", len(doc_ids))

    doc_pages = {doc_id: find_page_images(image_root, doc_id) for doc_id in doc_ids}
    pending = [
        doc_id
        for doc_id in doc_ids
        if args.overwrite or not (output_dir / f"{doc_id}.safetensors").exists()
    ]

    retrieval_model = None
    if pending:
        retrieval_model = load_retrieval_model(args.backbone_path, args.adapter_path, args.device)
    else:
        logger.info("All selected embeddings already exist; writing skip records only.")

    for doc_id in tqdm(doc_ids, desc="LongDocURL ColPali embeddings"):
        output_path = output_dir / f"{doc_id}.safetensors"
        pages = doc_pages[doc_id]
        page_indices = [idx for idx, _ in pages]
        image_paths = [str(path) for _, path in pages]
        status = "skipped"

        if output_path.exists() and not args.overwrite:
            logger.info("Skipping existing embedding: %s", output_path)
        else:
            import torch
            from safetensors.torch import save_file

            images = load_images([path for _, path in pages])
            doc_embs = retrieval_model.encode_images(
                images=images,
                batch_size=args.batch_size,
                to_cpu=True,
                use_tqdm=False,
            )
            doc_embs = torch.stack(doc_embs, dim=0).to(torch.bfloat16)
            save_file({"embeddings": doc_embs}, output_path)
            status = "generated"
            logger.info("Saved %s with shape %s", output_path, tuple(doc_embs.shape))

        write_manifest_record(
            manifest_path,
            {
                "doc_id": doc_id,
                "embedding_path": str(output_path),
                "page_count": len(page_indices),
                "page_indices": page_indices,
                "image_paths": image_paths,
                "backbone_path": args.backbone_path,
                "adapter_path": args.adapter_path,
                "dtype": "bfloat16",
                "status": status,
            },
        )


if __name__ == "__main__":
    main()
