#!/usr/bin/env python3
"""Extract benchmark PDFs with the MinerU API."""
import os
import time
import argparse
import logging
import requests
import zipfile
import io
import hashlib
import tempfile
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed


CODE_DIR = Path(__file__).resolve().parents[2]
MINERU_DATA_ID_MAX_LENGTH = 128
MINERU_BATCH_SIZE = 50
POLL_INTERVAL_SECONDS = 60
BATCH_REQUEST_RETRY_SECONDS = 60
BATCH_REQUEST_MAX_ATTEMPTS = 10
SPECIAL_UPLOAD_PAGE_LIMITS = {
    "longdocurl": {
        "4134756": 74,
    },
}

BENCHMARK_SPECS = {
    "longdocurl": {
        "pdf_dir": CODE_DIR / "benchmarks/longdocurl/data/raw/pdfs/4000-4999",
        "output_dir": CODE_DIR / "benchmarks/longdocurl/data/processed/pdfs_mineru/4000-4999",
        "max_pages": None,
    },
    "mmlongbench": {
        "pdf_dir": CODE_DIR / "benchmarks/mmlongbench/data/raw/documents",
        "output_dir": CODE_DIR / "benchmarks/mmlongbench/data/processed/pdfs_mineru",
        "max_pages": 120,
    },
}


def build_mineru_data_id(doc_no, max_length=MINERU_DATA_ID_MAX_LENGTH):
    if len(doc_no) <= max_length:
        return doc_no

    digest = hashlib.sha1(doc_no.encode("utf-8")).hexdigest()[:12]
    suffix = f"-{digest}"
    prefix_length = max_length - len(suffix)
    return f"{doc_no[:prefix_length]}{suffix}"


def get_upload_max_pages(benchmark, doc_no, default_max_pages):
    special_max_pages = SPECIAL_UPLOAD_PAGE_LIMITS.get(benchmark, {}).get(str(doc_no))
    if special_max_pages is None:
        return default_max_pages
    if default_max_pages is None or default_max_pages <= 0:
        return special_max_pages
    return min(default_max_pages, special_max_pages)


def prepare_upload_pdf(pdf_path, temp_dir, max_pages=200):
    pdf_path = Path(pdf_path)
    if max_pages is None or max_pages <= 0:
        return pdf_path

    import fitz

    with fitz.open(pdf_path) as source_doc:
        if len(source_doc) <= max_pages:
            return pdf_path

        truncated_path = Path(temp_dir) / f"{pdf_path.stem}_first_{max_pages}_pages.pdf"
        truncated_doc = fitz.open()
        truncated_doc.insert_pdf(source_doc, from_page=0, to_page=max_pages - 1)
        truncated_doc.save(truncated_path)
        truncated_doc.close()
        return truncated_path

# ========== 日志设置 ==========
def setup_logger(log_file):
    logger = logging.getLogger('MinerU_Auto')
    logger.setLevel(logging.INFO)
    if not logger.handlers:
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

# ========== 核心处理类 ==========
class MinerUAutomator:
    def __init__(self, benchmark, api_key, pdf_dir, output_dir, workers, logger, max_pages):
        self.benchmark = benchmark
        self.api_key = api_key
        self.pdf_dir = pdf_dir
        self.output_dir = output_dir
        self.workers = workers
        self.logger = logger
        self.max_pages = max_pages
        self.session = requests.Session()
        self.session.headers.update({"Authorization": f"Bearer {self.api_key}"})
        
        self.batch_url = "https://mineru.net/api/v4/file-urls/batch"
        self.status_url = "https://mineru.net/api/v4/extract-results/batch/{batch_id}"
        
        # 内存中记录本轮已下载的文件，避免重复触发下载
        self.downloaded_in_session = set()
        self.batch_info = {} #用于记录每个 batch 应该有多少个文件

    # --- 阶段 1：上传相关逻辑 ---
    def apply_batch_urls(self, files):
        for attempt in range(1, BATCH_REQUEST_MAX_ATTEMPTS + 1):
            response = None
            try:
                response = self.session.post(
                    self.batch_url,
                    json={"files": files, "model_version": "vlm"},
                    timeout=30,
                )
            except requests.RequestException as error:
                self.logger.warning(
                    f"申请批次链接失败 ({attempt}/{BATCH_REQUEST_MAX_ATTEMPTS}): {error}"
                )
            else:
                if response.status_code == 200:
                    result = response.json()
                    if result.get("code") == 0:
                        return result["data"]["batch_id"], result["data"]["file_urls"]
                    raise RuntimeError(f"MinerU 申请批次失败: {result.get('msg')}")

                if response.status_code != 429:
                    raise RuntimeError(
                        f"MinerU 申请批次请求失败: HTTP {response.status_code} {response.text}"
                    )

                retry_after = response.headers.get("Retry-After")
                self.logger.warning(
                    f"MinerU 批次接口限流 (HTTP 429)，将在 "
                    f"{retry_after or BATCH_REQUEST_RETRY_SECONDS} 秒后重试"
                )

            if attempt == BATCH_REQUEST_MAX_ATTEMPTS:
                break
            retry_seconds = BATCH_REQUEST_RETRY_SECONDS
            if response is not None and response.status_code == 429:
                retry_after = response.headers.get("Retry-After")
                if retry_after and retry_after.isdigit():
                    retry_seconds = int(retry_after)
            time.sleep(retry_seconds)

        raise RuntimeError(
            f"连续 {BATCH_REQUEST_MAX_ATTEMPTS} 次无法创建 MinerU 批次，请稍后重试"
        )

    def upload_single_file(self, file_path, upload_url, doc_no):
        for attempt in range(3):
            try:
                with open(file_path, 'rb') as f:
                    res = self.session.put(upload_url, data=f)
                    if res.status_code == 200:
                        self.logger.info(f"↗ 上传成功 | {os.path.basename(file_path)}")
                        return True
            except Exception as e:
                pass
            time.sleep(2 * (attempt + 1))
        self.logger.error(f"✗ 上传彻底失败 | {os.path.basename(file_path)}")
        return False

    # --- 阶段 2：下载相关逻辑 ---
    def download_and_extract(self, zip_url, target_folder):
        try:
            r = requests.get(zip_url, timeout=60)
            if r.status_code == 200:
                with zipfile.ZipFile(io.BytesIO(r.content)) as z:
                    z.extractall(target_folder)
                return True
        except Exception as e:
            self.logger.error(f"下载/解压异常 {target_folder}: {e}")
        return False

    def poll_batch_status(self, batch_id, poll_count=0):
        url = self.status_url.format(batch_id=batch_id)
        try:
            resp = self.session.get(url, timeout=10)
            if resp.status_code != 200:
                return False
            
            res_json = resp.json()
            if res_json.get("code") != 0:
                return False
            
            extract_results = res_json["data"].get("extract_result", [])
            
            # 关键修复：获取当前 Batch 已经处理完成（成功或失败）的总数
            finished_count = 0
            
            for item in extract_results:
                file_name = item["file_name"]
                state = item["state"]
                doc_no = file_name.replace('.pdf', '')
                
                if doc_no in self.downloaded_in_session:
                    finished_count += 1
                    continue

                if state == "done":
                    zip_url = item.get("full_zip_url")
                    target_folder = os.path.join(self.output_dir, doc_no)
                    self.logger.info(f"↙ 开始下载 | {file_name} 解析完成")
                    if self.download_and_extract(zip_url, target_folder):
                        self.logger.info(f"✓ 下载解压成功 | 目标: {doc_no}")
                        self.downloaded_in_session.add(doc_no)
                        finished_count += 1
                elif state == "failed":
                    self.logger.warning(f"⚠ 解析失败 | {file_name} | 原因: {item.get('err_msg')}")
                    self.downloaded_in_session.add(doc_no)
                    finished_count += 1
                # 如果是 processing，不计数

            # 只有当完成数 等于 该批次预期的总数时，才算真正结束
            expected_count = self.batch_info.get(batch_id, 0)
            if finished_count >= expected_count:
                return True
            else:
                if poll_count % 5 == 0: # 减少日志冗余，每5轮打印一次具体进度
                    self.logger.info(f"  > Batch {batch_id} 进度: {finished_count}/{expected_count}")
                return False
        except Exception as e:
            self.logger.error(f"查询 Batch {batch_id} 异常: {e}")
            return False

    def wait_for_batch(self, batch_id):
        poll_count = 0
        while True:
            poll_count += 1
            self.logger.info(f"--- 批次 {batch_id} 轮询检查 #{poll_count} ---")
            if self.poll_batch_status(batch_id, poll_count):
                self.logger.info(f"✓ 批次 {batch_id} 的解析结果已全部下载")
                return
            time.sleep(POLL_INTERVAL_SECONDS)

    def process_batch(self, chunk, temp_dir):
        upload_chunk = []
        for pdf_file, pdf_path, doc_no in chunk:
            upload_max_pages = get_upload_max_pages(
                self.benchmark, doc_no, self.max_pages
            )
            upload_path = prepare_upload_pdf(pdf_path, temp_dir, upload_max_pages)
            if Path(upload_path) != Path(pdf_path):
                self.logger.info(
                    f"裁剪上传 | {pdf_file} 仅上传前 {upload_max_pages} 页"
                )
            upload_chunk.append((pdf_file, upload_path, doc_no))

        files_payload = [
            {"name": pdf_file, "data_id": build_mineru_data_id(doc_no)}
            for pdf_file, _, doc_no in upload_chunk
        ]
        batch_id, urls = self.apply_batch_urls(files_payload)
        self.batch_info[batch_id] = len(upload_chunk)
        self.logger.info(f"获取 Batch ID 成功: {batch_id}，并发上传中...")

        with ThreadPoolExecutor(max_workers=self.workers) as pool:
            futures = [
                pool.submit(self.upload_single_file, pdf_path, urls[index], doc_no)
                for index, (_, pdf_path, doc_no) in enumerate(upload_chunk)
            ]
            upload_results = [future.result() for future in as_completed(futures)]

        if not all(upload_results):
            raise RuntimeError(f"批次 {batch_id} 中存在上传失败的 PDF，请重新运行脚本")

        self.logger.info(f"批次 {batch_id} 上传完成，等待解析并下载结果")
        self.wait_for_batch(batch_id)
        
    # --- 主流程调度 ---
    def run(self, limit=None):
        os.makedirs(self.output_dir, exist_ok=True)
        all_pdfs = sorted([f for f in os.listdir(self.pdf_dir) if f.endswith('.pdf')])
        if limit:
            all_pdfs = all_pdfs[:limit]

        # 1. 扫描未处理的文件
        pending_uploads = []
        for pdf_file in all_pdfs:
            doc_no = pdf_file.replace('.pdf', '')
            doc_folder = os.path.join(self.output_dir, doc_no)
            os.makedirs(doc_folder, exist_ok=True)
            # 如果文件夹内有提取结果，说明已完成
            if not any(fn.endswith(('.json', '.md', '.html')) for fn in os.listdir(doc_folder)):
                pending_uploads.append((pdf_file, os.path.join(self.pdf_dir, pdf_file), doc_no))

        self.logger.info(f"总计 PDF: {len(all_pdfs)} | 待上传/解析: {len(pending_uploads)}")

        if not pending_uploads:
            self.logger.info("所有 PDF 均已完成解析。")
            return

        batch_count = (len(pending_uploads) + MINERU_BATCH_SIZE - 1) // MINERU_BATCH_SIZE
        with tempfile.TemporaryDirectory(prefix="mineru_upload_") as temp_dir:
            for offset in range(0, len(pending_uploads), MINERU_BATCH_SIZE):
                batch_number = offset // MINERU_BATCH_SIZE + 1
                chunk = pending_uploads[offset : offset + MINERU_BATCH_SIZE]
                self.logger.info(
                    f"=== 开始处理批次 ({batch_number}/{batch_count}) ==="
                )
                self.process_batch(chunk, temp_dir)

        self.logger.info("所有批次解析与下载完成。")


def parse_args():
    parser = argparse.ArgumentParser(description="Extract benchmark PDFs with MinerU.")
    parser.add_argument("--benchmark", choices=sorted(BENCHMARK_SPECS), required=True)
    parser.add_argument("--api-key", default=os.environ.get("MINERU_API_KEY"))
    parser.add_argument("--pdf-dir", default=None, help="Override the benchmark PDF directory.")
    parser.add_argument("--output-dir", default=None, help="Override the MinerU output directory.")
    parser.add_argument("--workers", type=int, default=5)
    parser.add_argument("--log-file", default=None)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--max-pages", type=int, default=None)
    return parser.parse_args()


def main():
    args = parse_args()
    if not args.api_key:
        raise ValueError("MINERU_API_KEY is required. Set it in .env or pass --api-key.")

    spec = BENCHMARK_SPECS[args.benchmark]
    pdf_dir = Path(args.pdf_dir or spec["pdf_dir"])
    output_dir = Path(args.output_dir or spec["output_dir"])
    max_pages = spec["max_pages"] if args.max_pages is None else args.max_pages
    log_file = args.log_file or f"mineru_{args.benchmark}.log"

    logger = setup_logger(log_file)
    automator = MinerUAutomator(
        args.benchmark,
        args.api_key,
        pdf_dir,
        output_dir,
        args.workers,
        logger,
        max_pages,
    )
    automator.run(limit=args.limit)


if __name__ == "__main__":
    main()
