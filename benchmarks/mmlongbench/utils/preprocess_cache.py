import json
import os
import pathlib
import re

from utils.config_utils import get_config_value, require_config_value


def mmlongbench_file_id(doc_id):
    filename = pathlib.PurePosixPath(str(doc_id)).name
    file_id = re.sub(r'\.pdf$', '', filename, flags=re.IGNORECASE)
    return re.sub(r'[^A-Za-z0-9._-]+', '_', file_id).strip('._') or 'document'


def mmlongbench_ocr_dir(benchmark_cfg, doc_id):
    tmp_dir = require_config_value(benchmark_cfg, 'tmp_dir')
    root = get_config_value(
        benchmark_cfg,
        'ocr_json_dir',
        os.path.join(tmp_dir, 'pdf_jsons'),
    )
    return os.path.join(root, mmlongbench_file_id(doc_id))


def mmlongbench_png_dir(benchmark_cfg, doc_id):
    tmp_dir = require_config_value(benchmark_cfg, 'tmp_dir')
    root = get_config_value(
        benchmark_cfg,
        'pdf_png_dir',
        os.path.join(tmp_dir, 'pdf_pngs'),
    )
    return os.path.join(root, mmlongbench_file_id(doc_id))


def mmlongbench_ocr_page_path(benchmark_cfg, doc_id, page_index):
    return os.path.join(mmlongbench_ocr_dir(benchmark_cfg, doc_id), f'page_{page_index + 1:04d}.json')


def mmlongbench_png_page_path(benchmark_cfg, doc_id, page_index):
    resolution = int(require_config_value(benchmark_cfg, 'resolution'))
    return os.path.join(
        mmlongbench_png_dir(benchmark_cfg, doc_id),
        f'page_{page_index + 1:04d}_dpi{resolution}.png',
    )


def write_mmlongbench_ocr_cache(pdf, benchmark_cfg, doc_id):
    output_dir = mmlongbench_ocr_dir(benchmark_cfg, doc_id)
    os.makedirs(output_dir, exist_ok=True)
    max_pages = int(require_config_value(benchmark_cfg, 'max_pages'))
    page_count = min(len(pdf), max_pages)
    for page_index in range(page_count):
        page = pdf[page_index]
        page_text = page.get_text('text').strip() or '[EMPTY PAGE]'
        page_payload = {
            'doc_id': doc_id,
            'file_id': mmlongbench_file_id(doc_id),
            'page_index': page_index,
            'page_number': page_index + 1,
            'text': page_text,
        }
        with open(mmlongbench_ocr_page_path(benchmark_cfg, doc_id, page_index), 'w', encoding='utf-8') as f:
            json.dump(page_payload, f, ensure_ascii=False, indent=2)
    return page_count


def write_mmlongbench_png_cache(pdf, benchmark_cfg, doc_id):
    output_dir = mmlongbench_png_dir(benchmark_cfg, doc_id)
    os.makedirs(output_dir, exist_ok=True)
    max_pages = int(require_config_value(benchmark_cfg, 'max_pages'))
    resolution = int(require_config_value(benchmark_cfg, 'resolution'))
    page_count = min(len(pdf), max_pages)
    for page_index in range(page_count):
        page_path = mmlongbench_png_page_path(benchmark_cfg, doc_id, page_index)
        if os.path.exists(page_path):
            continue
        pixmap = pdf[page_index].get_pixmap(dpi=resolution)
        pixmap.save(page_path)
    return page_count
