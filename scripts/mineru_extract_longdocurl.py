#!/usr/bin/env python3
"""
用 MinerU API 批量解析 LongDocURL PDF 文件
- 源PDF目录: code/benchmarks/longdocurl/data/pdfs/4000-4999
- 输出目录: code/benchmarks/longdocurl/data/pdfs_mineru/4000-4999/{doc_no}/{doc_no}.json
- 支持断点续跑、日志、并发

用法：
  python code/scripts/mineru_extract_longdocurl.py --api_key <你的API密钥>

建议后台运行

"""
import os
import json
import argparse
from pathlib import Path
from tqdm import tqdm
import requests
import logging
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime

# ========== 日志设置 ==========
def setup_logger(log_file):
    logger = logging.getLogger('MinerU')
    logger.setLevel(logging.INFO)
    fh = logging.FileHandler(log_file, encoding='utf-8')
    fh.setLevel(logging.INFO)
    ch = logging.StreamHandler()
    ch.setLevel(logging.INFO)
    formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    fh.setFormatter(formatter)
    ch.setFormatter(formatter)
    logger.addHandler(fh)
    logger.addHandler(ch)
    return logger

# ========== MinerU API ==========
class MinerUExtractor:
    def __init__(self, api_key, logger=None):
        self.api_key = api_key
        self.api_url = "https://api.mineru.net/v1/pdf_parse"
        self.session = requests.Session()
        self.logger = logger or logging.getLogger('MinerU')
    def extract(self, pdf_path, max_retries=3, timeout=300):
        for attempt in range(max_retries):
            try:
                with open(pdf_path, 'rb') as f:
                    files = {'file': f}
                    headers = {'Authorization': f'Bearer {self.api_key}'}
                    resp = self.session.post(self.api_url, files=files, headers=headers, timeout=timeout)
                if resp.status_code == 200:
                    return resp.json()
                else:
                    self.logger.warning(f"API错误 {resp.status_code}: {resp.text[:100]}")
            except Exception as e:
                self.logger.warning(f"请求异常: {e}")
            if attempt < max_retries - 1:
                time.sleep(5 * (attempt + 1))
        return None

# ========== 断点续跑 ==========
def get_doc_no(pdf_filename):
    return pdf_filename.replace('.pdf', '')
def create_output_path(pdf_filename, output_root):
    doc_no = get_doc_no(pdf_filename)
    doc_folder = os.path.join(output_root, doc_no)
    os.makedirs(doc_folder, exist_ok=True)
    return os.path.join(doc_folder, f"{doc_no}.json")
def is_already_parsed(output_path):
    return os.path.exists(output_path) and os.path.getsize(output_path) > 100

def main(args):
    logger = setup_logger(args.log_file)
    extractor = MinerUExtractor(args.api_key, logger)
    pdf_dir = args.pdf_dir
    output_dir = args.output_dir
    os.makedirs(output_dir, exist_ok=True)
    pdf_files = sorted([f for f in os.listdir(pdf_dir) if f.endswith('.pdf')])
    logger.info(f"共 {len(pdf_files)} 个PDF")
    # 断点续跑
    pending = []
    for pdf_file in pdf_files:
        out_path = create_output_path(pdf_file, output_dir)
        if not is_already_parsed(out_path):
            pending.append((pdf_file, out_path))
    logger.info(f"待处理: {len(pending)} 个PDF")
    if not pending:
        logger.info("全部已完成，无需处理")
        return
    # 多线程并发
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        futures = {}
        for pdf_file, out_path in pending:
            pdf_path = os.path.join(pdf_dir, pdf_file)
            futures[pool.submit(extractor.extract, pdf_path)] = (pdf_file, out_path)
        for fut in tqdm(as_completed(futures), total=len(futures), desc="解析进度"):
            pdf_file, out_path = futures[fut]
            try:
                result = fut.result()
                if result:
                    with open(out_path, 'w', encoding='utf-8') as f:
                        json.dump(result, f, ensure_ascii=False, indent=2)
                    logger.info(f"✓ {pdf_file}")
                else:
                    logger.warning(f"✗ {pdf_file} 解析失败")
            except Exception as e:
                logger.error(f"✗ {pdf_file} 异常: {e}")
    logger.info("全部任务完成！")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='批量用 MinerU 解析 LongDocURL PDF')
    parser.add_argument('--api_key', type=str, default=os.environ.get('MINERU_API_KEY'), help='MinerU API 密钥')
    parser.add_argument('--pdf_dir', type=str, default='code/benchmarks/longdocurl/data/pdfs/4000-4999', help='PDF目录')
    parser.add_argument('--output_dir', type=str, default='code/benchmarks/longdocurl/data/pdfs_mineru/4000-4999', help='输出目录')
    parser.add_argument('--workers', type=int, default=5, help='并发线程数')
    parser.add_argument('--log_file', type=str, default='mineru_longdocurl.log', help='日志文件')
    args = parser.parse_args()
    if not args.api_key:
        print('请用 --api_key 或 MINERU_API_KEY 环境变量提供 MinerU 密钥')
        exit(1)
    main(args)
