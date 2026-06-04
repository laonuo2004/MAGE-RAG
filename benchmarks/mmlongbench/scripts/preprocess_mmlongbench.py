#!/usr/bin/env python3
import argparse
import json
import logging
import os
import pathlib
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from types import SimpleNamespace

CODE_DIR = pathlib.Path(__file__).resolve().parents[3]
sys.path.insert(0, str(CODE_DIR))

from benchmarks.utils.data_utils import (
    mmlongbench_file_id,
    mmlongbench_ocr_dir,
    mmlongbench_png_dir,
    write_mmlongbench_ocr_cache,
    write_mmlongbench_png_cache,
)


DEFAULT_INPUT_PATH = CODE_DIR / 'benchmarks' / 'mmlongbench' / 'data' / 'raw' / 'samples.json'
DEFAULT_DOCUMENT_PATH = CODE_DIR / 'benchmarks' / 'mmlongbench' / 'data' / 'raw' / 'documents'
DEFAULT_TMP_DIR = CODE_DIR / 'benchmarks' / 'mmlongbench' / 'data' / 'cache'

logger = logging.getLogger('preprocess_mmlongbench')


def parse_args():
    parser = argparse.ArgumentParser(
        description='Preprocess MMLongBench PDFs into cached OCR JSON files and rendered PNG pages.'
    )
    parser.add_argument('--input-path', default=str(DEFAULT_INPUT_PATH), help='MMLongBench samples.json path.')
    parser.add_argument('--document-path', default=str(DEFAULT_DOCUMENT_PATH), help='Directory containing PDF files.')
    parser.add_argument('--tmp-dir', default=str(DEFAULT_TMP_DIR), help='Root cache directory.')
    parser.add_argument('--ocr-json-dir', default=None, help='Root OCR cache directory. Defaults to tmp_dir/pdf_jsons.')
    parser.add_argument('--pdf-png-dir', default=None, help='Root PNG cache directory. Defaults to tmp_dir/pdf_pngs.')
    parser.add_argument('--max-pages', type=int, default=120, help='Maximum pages to preprocess per PDF.')
    parser.add_argument('--resolution', type=int, default=144, help='PNG render DPI.')
    parser.add_argument('--mode', choices=('both', 'ocr', 'image'), default='both', help='Which cache type to create.')
    parser.add_argument('--workers', type=int, default=1, help='Number of PDF preprocessing workers.')
    parser.add_argument('--doc-id', action='append', default=None, help='Specific doc_id to preprocess. Can be repeated.')
    return parser.parse_args()


def load_doc_ids(input_path, explicit_doc_ids):
    if explicit_doc_ids:
        return sorted(set(explicit_doc_ids))
    with open(input_path, 'r', encoding='utf-8') as f:
        samples = json.load(f)
    return sorted({sample['doc_id'] for sample in samples})


def build_benchmark_cfg(args):
    values = {
        'tmp_dir': args.tmp_dir,
        'max_pages': args.max_pages,
        'resolution': args.resolution,
    }
    if args.ocr_json_dir:
        values['ocr_json_dir'] = args.ocr_json_dir
    if args.pdf_png_dir:
        values['pdf_png_dir'] = args.pdf_png_dir
    return SimpleNamespace(**values)


def preprocess_doc(doc_id, args, benchmark_cfg):
    import fitz

    pdf_path = os.path.join(args.document_path, doc_id)
    if not os.path.exists(pdf_path):
        raise FileNotFoundError(f'Missing PDF for doc_id={doc_id}: {pdf_path}')

    with fitz.open(pdf_path) as pdf:
        page_count = min(len(pdf), args.max_pages)
        if args.mode in ('both', 'ocr'):
            write_mmlongbench_ocr_cache(pdf, benchmark_cfg, doc_id)
        if args.mode in ('both', 'image'):
            write_mmlongbench_png_cache(pdf, benchmark_cfg, doc_id)

    return {
        'doc_id': doc_id,
        'file_id': mmlongbench_file_id(doc_id),
        'pages': page_count,
        'ocr_dir': mmlongbench_ocr_dir(benchmark_cfg, doc_id) if args.mode in ('both', 'ocr') else None,
        'png_dir': mmlongbench_png_dir(benchmark_cfg, doc_id) if args.mode in ('both', 'image') else None,
    }


def main():
    logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s %(message)s')
    args = parse_args()
    benchmark_cfg = build_benchmark_cfg(args)
    doc_ids = load_doc_ids(args.input_path, args.doc_id)
    logger.info('Preprocessing %s MMLongBench PDFs. mode=%s workers=%s', len(doc_ids), args.mode, args.workers)

    os.makedirs(args.tmp_dir, exist_ok=True)
    results = []
    with ThreadPoolExecutor(max_workers=max(1, args.workers)) as executor:
        futures = {executor.submit(preprocess_doc, doc_id, args, benchmark_cfg): doc_id for doc_id in doc_ids}
        for future in as_completed(futures):
            result = future.result()
            results.append(result)
            logger.info(
                'Cached doc_id=%s pages=%s ocr_dir=%s png_dir=%s',
                result['doc_id'],
                result['pages'],
                result['ocr_dir'],
                result['png_dir'],
            )

    logger.info('Finished preprocessing %s PDFs.', len(results))


if __name__ == '__main__':
    main()
