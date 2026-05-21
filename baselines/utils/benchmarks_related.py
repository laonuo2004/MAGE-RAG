import base64
import json
import os
import re
from collections import OrderedDict
from io import BytesIO
from pathlib import Path

from benchmarks.mmlongbench.utils.preprocess_cache import mmlongbench_ocr_page_path
from utils.config_utils import get_config_value, require_config_value


def encode_pil_image_to_base64(img):
    if img.mode in ('RGBA', 'P'):
        img = img.convert('RGB')
    buffer = BytesIO()
    img.save(buffer, format='JPEG')
    return base64.b64encode(buffer.getvalue()).decode('utf-8')


def encode_image_file_to_base64(image_path):
    if 'https' in image_path:
        import requests

        response = requests.get(image_path)
        return base64.b64encode(response.content).decode('utf-8')
    with open(image_path, 'rb') as image_file:
        return base64.b64encode(image_file.read()).decode('utf-8')


def resolve_embedding_path(cfg, field, benchmark_name, stem):
    roots = require_config_value(cfg, f'baselines.{field}')
    if isinstance(roots, str):
        root = roots
    else:
        root = get_config_value(roots, benchmark_name)
    if not root:
        raise ValueError(f'Baseline requires cfg.baselines.{field}.{benchmark_name}.')
    return os.path.join(str(root), f'{stem}.safetensors')


def allowed_page_indices(benchmark_name, sample, benchmark_cfg, page_count):
    if benchmark_name == 'mmlongbench':
        max_pages = int(require_config_value(benchmark_cfg, 'max_pages'))
        return list(range(min(page_count, max_pages)))
    if benchmark_name == 'longdocurl':
        images = sample.get('images')
        if isinstance(images, str):
            images = [images]
        if not images:
            raise ValueError('LongDocURL retrieval requires sample["images"] to derive the page mask.')

        page_indices = []
        for image_path in images:
            filename = os.path.basename(str(image_path))
            match = re.search(r'_(\d+)\.[^.]+$', filename)
            if not match:
                raise ValueError(f'Cannot parse LongDocURL page index from image path: {image_path}')
            page_index = int(match.group(1))
            if 0 <= page_index < page_count:
                page_indices.append(page_index)
        page_indices = sorted(set(page_indices))
        if not page_indices:
            raise ValueError('LongDocURL image-derived page mask is empty after clipping to embedding page count.')
        return page_indices
    raise ValueError(f'Unsupported benchmark for page mask: {benchmark_name}')


def load_mmlongbench_ocr_pages(sample, benchmark_cfg):
    max_pages = int(require_config_value(benchmark_cfg, 'max_pages'))
    allowed_pages = allowed_page_indices('mmlongbench', sample, benchmark_cfg, max_pages)
    pages = []
    for page_index in allowed_pages:
        page_path = mmlongbench_ocr_page_path(benchmark_cfg, sample['doc_id'], page_index)
        if not os.path.exists(page_path):
            if page_index == 0:
                raise FileNotFoundError(
                    f'Missing preprocessed OCR cache for doc_id={sample["doc_id"]}: {page_path}. '
                    'Run benchmarks/mmlongbench/scripts/preprocess_mmlongbench.py before evaluating OCR-based baselines.'
                )
            break
        with open(page_path, 'r', encoding='utf-8') as f:
            page_payload = json.load(f)
        pages.append({
            'page_index': page_index,
            'page_number': page_index + 1,
            'text': str(page_payload.get('text') or '').strip() or '[EMPTY PAGE]',
        })
    return pages, allowed_pages


def load_longdocurl_ocr_pages(sample, benchmark_cfg):
    from benchmarks.longdocurl.eval.api_models.pure_ocr_utils import (
        build_page_texts_from_contents,
        load_pymupdf_record,
    )

    ocr_json_dir = require_config_value(benchmark_cfg, 'ocr_json_dir')
    record = load_pymupdf_record(sample['doc_no'], ocr_json_dir)
    contents = record['contents']
    page_count = int(sample.get('total_pages') or 0)
    if page_count <= 0 and contents:
        page_count = max(int(item['page_no']) for item in contents) + 1

    allowed_pages = allowed_page_indices('longdocurl', sample, benchmark_cfg, page_count)
    page_texts = build_page_texts_from_contents(contents, allowed_pages)
    pages = [
        {
            'page_index': page_no,
            'page_number': page_no + 1,
            'text': page_text.strip() or '[EMPTY PAGE]',
        }
        for page_no, page_text in page_texts
    ]
    if not pages:
        raise ValueError(f'No LongDocURL OCR pages loaded for doc_no={sample["doc_no"]}.')
    return pages, allowed_pages


def normalize_text_block(text):
    text = "" if text is None else str(text)
    text = text.replace("\u00a0", " ")
    text = re.sub(r"[\x00-\x08\x0b\x0c\x0e-\x1f]+", " ", text)
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def build_token_chunks_from_pages(
    pages,
    tokenize_with_spans,
    chunk_size,
    chunk_overlap,
    *,
    allow_cross_page=False,
    max_cross_pages=None,
):
    if chunk_size <= 0:
        raise ValueError("chunk_size must be > 0")
    if chunk_overlap < 0 or chunk_overlap >= chunk_size:
        raise ValueError("chunk_overlap must satisfy 0 <= chunk_overlap < chunk_size")
    if max_cross_pages is not None and max_cross_pages <= 0:
        raise ValueError("max_cross_pages must be > 0 when set")

    normalized_pages = []
    for page in pages:
        normalized_pages.append({
            **page,
            "text": normalize_text_block(page.get("text")),
        })

    if not allow_cross_page:
        return _build_single_page_chunks(normalized_pages, tokenize_with_spans, chunk_size, chunk_overlap)
    return _build_cross_page_chunks(
        normalized_pages,
        tokenize_with_spans,
        chunk_size,
        chunk_overlap,
        max_cross_pages=max_cross_pages,
    )


def colbertv2_doc_cache_variant(checkpoint, chunk_size, chunk_overlap, allow_cross_page, max_cross_pages):
    checkpoint_name = Path(str(checkpoint)).name or "checkpoint"
    checkpoint_name = re.sub(r"[^A-Za-z0-9._-]+", "_", checkpoint_name).strip("._") or "checkpoint"
    return (
        f"ckpt_{checkpoint_name}"
        f"_chunk_size_{int(chunk_size)}"
        f"_chunk_overlap_{int(chunk_overlap)}"
        f"_allow_cross_page_{bool(allow_cross_page)}"
        f"_max_cross_pages_{max_cross_pages}"
    )


def colbertv2_query_cache_variant(checkpoint):
    checkpoint_name = Path(str(checkpoint)).name or "checkpoint"
    checkpoint_name = re.sub(r"[^A-Za-z0-9._-]+", "_", checkpoint_name).strip("._") or "checkpoint"
    return f"ckpt_{checkpoint_name}"


def bgem3_doc_cache_variant(checkpoint, mode, text_source, chunk_size, chunk_overlap, allow_cross_page, max_cross_pages):
    checkpoint_name = Path(str(checkpoint)).name or "checkpoint"
    checkpoint_name = re.sub(r"[^A-Za-z0-9._-]+", "_", checkpoint_name).strip("._") or "checkpoint"
    return (
        f"ckpt_{checkpoint_name}"
        f"_mode_{mode}"
        f"_text_source_{text_source}"
        f"_chunk_size_{int(chunk_size)}"
        f"_chunk_overlap_{int(chunk_overlap)}"
        f"_allow_cross_page_{bool(allow_cross_page)}"
        f"_max_cross_pages_{max_cross_pages}"
    )


def bgem3_query_cache_variant(checkpoint, mode):
    checkpoint_name = Path(str(checkpoint)).name or "checkpoint"
    checkpoint_name = re.sub(r"[^A-Za-z0-9._-]+", "_", checkpoint_name).strip("._") or "checkpoint"
    return f"ckpt_{checkpoint_name}_mode_{mode}"


def _build_single_page_chunks(pages, tokenize_with_spans, chunk_size, chunk_overlap):
    chunks = []
    step = chunk_size - chunk_overlap
    for page in pages:
        token_spans = tokenize_with_spans(page["text"])
        if not token_spans:
            token_spans = [{"start": 0, "end": len(page["text"])}]
        for chunk_index, start in enumerate(range(0, len(token_spans), step)):
            end = min(start + chunk_size, len(token_spans))
            spans = token_spans[start:end]
            chunk_text = page["text"][spans[0]["start"]:spans[-1]["end"]].strip() or "[EMPTY CHUNK]"
            chunks.append({
                "chunk_id": len(chunks),
                "chunk_index": chunk_index,
                "text": chunk_text,
                "start_page_index": page["page_index"],
                "end_page_index": page["page_index"],
                "start_page_number": page["page_number"],
                "end_page_number": page["page_number"],
                "covered_page_indices": [page["page_index"]],
                "covered_page_numbers": [page["page_number"]],
            })
            if end == len(token_spans):
                break
    return chunks


def _build_cross_page_chunks(pages, tokenize_with_spans, chunk_size, chunk_overlap, *, max_cross_pages=None):
    flat_tokens = []
    for page in pages:
        token_spans = tokenize_with_spans(page["text"])
        if not token_spans:
            token_spans = [{"start": 0, "end": len(page["text"])}]
        for span in token_spans:
            flat_tokens.append({
                "page_index": page["page_index"],
                "page_number": page["page_number"],
                "page_text": page["text"],
                "start": span["start"],
                "end": span["end"],
            })

    if not flat_tokens:
        return []

    chunks = []
    step = chunk_size - chunk_overlap
    for chunk_index, start in enumerate(range(0, len(flat_tokens), step)):
        spans = []
        covered_pages = OrderedDict()
        cursor = start
        while cursor < len(flat_tokens) and len(spans) < chunk_size:
            token = flat_tokens[cursor]
            covered_pages[token["page_index"]] = token["page_number"]
            if max_cross_pages is not None and len(covered_pages) > max_cross_pages:
                break
            spans.append(token)
            cursor += 1
        if not spans:
            continue

        text_parts = []
        page_buffers = OrderedDict()
        for token in spans:
            buffer = page_buffers.setdefault(token["page_index"], {
                "page_number": token["page_number"],
                "page_text": token["page_text"],
                "start": token["start"],
                "end": token["end"],
            })
            buffer["start"] = min(buffer["start"], token["start"])
            buffer["end"] = max(buffer["end"], token["end"])

        for page_index, buffer in page_buffers.items():
            page_text = buffer["page_text"][buffer["start"]:buffer["end"]].strip()
            if page_text:
                text_parts.append(page_text)

        chunk_text = "\n".join(text_parts).strip() or "[EMPTY CHUNK]"
        covered_page_indices = list(page_buffers.keys())
        covered_page_numbers = [buffer["page_number"] for buffer in page_buffers.values()]
        chunks.append({
            "chunk_id": len(chunks),
            "chunk_index": chunk_index,
            "text": chunk_text,
            "start_page_index": covered_page_indices[0],
            "end_page_index": covered_page_indices[-1],
            "start_page_number": covered_page_numbers[0],
            "end_page_number": covered_page_numbers[-1],
            "covered_page_indices": covered_page_indices,
            "covered_page_numbers": covered_page_numbers,
        })
        if cursor >= len(flat_tokens):
            break
    return chunks
