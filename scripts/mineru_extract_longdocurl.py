#!/usr/bin/env python3
"""
MinerU API 

- 自动断点续传（跳过已下载的，跳过已上传但在排队的）。
- 上传完毕后进入轮询模式，自动下载解析完毕的 Zip。
"""
import os
import time
import argparse
import logging
import requests
import zipfile
import io
from concurrent.futures import ThreadPoolExecutor, as_completed

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
    def __init__(self, api_key, pdf_dir, output_dir, workers, logger):
        self.api_key = api_key
        self.pdf_dir = pdf_dir
        self.output_dir = output_dir
        self.workers = workers
        self.logger = logger
        self.session = requests.Session()
        self.session.headers.update({"Authorization": f"Bearer {self.api_key}"})
        
        self.batch_url = "https://mineru.net/api/v4/file-urls/batch"
        self.status_url = "https://mineru.net/api/v4/extract-results/batch/{batch_id}"
        
        # 内存中记录本轮已下载的文件，避免重复触发下载
        self.downloaded_in_session = set()
        self.batch_info = {} #用于记录每个 batch 应该有多少个文件

    # --- 阶段 1：上传相关逻辑 ---
    def apply_batch_urls(self, files):
        try:
            resp = self.session.post(self.batch_url, json={"files": files, "model_version": "vlm"})
            if resp.status_code == 200:
                res = resp.json()
                if res.get("code") == 0:
                    return res["data"]["batch_id"], res["data"]["file_urls"]
                self.logger.error(f"申请链接失败: {res.get('msg')}")
            else:
                self.logger.error(f"申请链接请求异常: {resp.status_code}")
        except Exception as e:
            self.logger.error(f"申请链接网络异常: {e}")
        return None, None

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

    def poll_batch_status(self, batch_id):
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

        active_batches = []

        # 2. 分批上传获取 batch_id
        chunk_size = 50
        for i in range(0, len(pending_uploads), chunk_size):
            chunk = pending_uploads[i:i + chunk_size]
            self.logger.info(f"=== 开始上传批次 ({(i//chunk_size)+1}/{(len(pending_uploads)//chunk_size)+1}) ===")
            
            files_payload = [{"name": item[0], "data_id": item[2]} for item in chunk]
            batch_id, urls = self.apply_batch_urls(files_payload)
            
            if not batch_id:
                continue
                
            self.logger.info(f"获取 Batch ID 成功: {batch_id}，并发上传中...")
            active_batches.append(batch_id)
            self.batch_info[batch_id] = len(chunk)
            
            with ThreadPoolExecutor(max_workers=self.workers) as pool:
                futures = []
                for j, (pdf_file, pdf_path, doc_no) in enumerate(chunk):
                    futures.append(pool.submit(self.upload_single_file, pdf_path, urls[j], doc_no))
                for fut in as_completed(futures):
                    fut.result()

        if not active_batches:
            self.logger.info("没有正在活跃的批次任务，程序退出。")
            return

        # 3. 轮询监控状态并下载
        self.logger.info("========================================")
        self.logger.info("所有上传完成，进入后台轮询下载模式 (每60秒检查一次)")
        self.logger.info("========================================")

        poll_count = 0
        while active_batches:
            poll_count += 1
            self.logger.info(f"--- 轮询检查 #{poll_count} | 正在跟踪 {len(active_batches)} 个批次 ---")
            
            batches_to_remove = []
            for b_id in active_batches:
                is_completed = self.poll_batch_status(b_id)
                if is_completed:
                    self.logger.info(f"★ 批次 {b_id} 的所有文件均已处理完毕并脱离追踪队列！")
                    batches_to_remove.append(b_id)
            
            # 移除已全部结束的批次
            for b_id in batches_to_remove:
                active_batches.remove(b_id)
                
            if active_batches:
                time.sleep(60) # 等待 60 秒后再次轮询
                
        self.logger.info("所有批次解析与下载任务全部圆满完成！(撒花🎉)")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='MinerU 自动化批量解析与下载')
    parser.add_argument('--api_key', type=str, default=os.environ.get('MINERU_API_KEY'), help='API Key')
    parser.add_argument('--pdf_dir', type=str, default='code/benchmarks/longdocurl/data/pdfs/4000-4999')
    parser.add_argument('--output_dir', type=str, default='code/benchmarks/longdocurl/data/pdfs_mineru/4000-4999')
    parser.add_argument('--workers', type=int, default=5, help='并发上传线程数')
    parser.add_argument('--log_file', type=str, default='mineru_auto.log', help='日志文件')
    parser.add_argument('--limit', type=int, default=None, help='仅处理前N个PDF用于测试')
    args = parser.parse_args()

    if not args.api_key:
        print('错误：请提供 --api_key')
        exit(1)

    logger = setup_logger(args.log_file)
    automator = MinerUAutomator(args.api_key, args.pdf_dir, args.output_dir, args.workers, logger)
    automator.run(limit=args.limit)
