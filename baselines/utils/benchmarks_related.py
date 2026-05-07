import base64
import json
import os
import re
from io import BytesIO

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
