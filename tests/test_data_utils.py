import json
import subprocess
import tempfile
import unittest
from pathlib import Path

from omegaconf import OmegaConf

from benchmarks.utils.data_utils import (
    benchmark_cache_root,
    bgem3_cache_root,
    build_page_texts_from_contents,
    colbertv2_cache_root,
    colpali_pdf_embeddings_path,
    colpali_question_embeddings_path,
    extract_page_nos_from_images,
    get_pure_ocr_prompt_pymupdf,
    load_longdocurl_samples,
    longdocurl_mineru_dir,
    mmlongbench_file_id,
    mmlongbench_ocr_page_path,
    mmlongbench_png_page_path,
)


class DataUtilsTests(unittest.TestCase):
    def test_mmlongbench_file_id_and_page_paths(self):
        cfg = OmegaConf.create({
            "ocr_json_dir": "/tmp/ocr",
            "pdf_png_dir": "/tmp/png",
            "resolution": 144,
        })

        self.assertEqual(mmlongbench_file_id("folder/My Doc.pdf"), "My_Doc")
        self.assertEqual(mmlongbench_ocr_page_path(cfg, "folder/My Doc.pdf", 0), "/tmp/ocr/My_Doc/page_0001.json")
        self.assertEqual(mmlongbench_png_page_path(cfg, "folder/My Doc.pdf", 1), "/tmp/png/My_Doc/page_0002_dpi144.png")

    def test_longdocurl_load_samples_adds_ids_and_rewrites_images(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            qa_file = Path(tmp_dir) / "qa.jsonl"
            qa_file.write_text(json.dumps({"images": ["/old/1234/doc_0.png"], "question": "q"}) + "\n", encoding="utf-8")
            samples = load_longdocurl_samples(qa_file, image_prefix=Path(tmp_dir) / "images")

        self.assertEqual(samples[0]["question_id"], 0)
        self.assertTrue(samples[0]["images"][0].endswith("images/1234/doc_0.png"))

    def test_longdocurl_ocr_page_extraction(self):
        contents = [
            {"page_no": 2, "block_no": 1, "line_no": 1, "word_no": 2, "word": "world"},
            {"page_no": 2, "block_no": 1, "line_no": 1, "word_no": 1, "word": "hello"},
            {"page_no": 3, "block_no": 1, "line_no": 1, "word_no": 1, "word": "ignored"},
        ]

        self.assertEqual(build_page_texts_from_contents(contents, [2]), [(2, "hello world")])
        self.assertEqual(extract_page_nos_from_images(["/x/doc_2.png", "/x/doc_2.png", "/x/doc_4.png"]), [2, 4])

    def test_pymupdf_prompt_uses_direct_or_nested_ocr_record(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            record_path = Path(tmp_dir) / "123456.json"
            record_path.write_text(json.dumps({"contents": [
                {"page_no": 0, "block_no": 1, "line_no": 1, "word_no": 1, "word": "alpha"},
                {"page_no": 0, "block_no": 1, "line_no": 1, "word_no": 2, "word": "beta"},
            ]}), encoding="utf-8")

            prompt, pages = get_pure_ocr_prompt_pymupdf("123456", ocr_json_dir=tmp_dir, start_page=0, end_page=0)

        self.assertEqual(pages, [0])
        self.assertIn("page_no: 1\nalpha beta", prompt)

    def test_default_benchmark_cache_paths(self):
        self.assertEqual(
            colpali_pdf_embeddings_path("longdocurl", "4000045"),
            benchmark_cache_root("longdocurl", "colpali") / "pdf_embeddings" / "4000-4999" / "4000045.safetensors",
        )
        self.assertEqual(
            colpali_pdf_embeddings_path("mmlongbench", "sample"),
            benchmark_cache_root("mmlongbench", "colpali") / "pdf_embeddings" / "sample.safetensors",
        )
        self.assertEqual(
            colpali_question_embeddings_path("mmlongbench", "q1"),
            benchmark_cache_root("mmlongbench", "colpali") / "question_embeddings" / "q1.safetensors",
        )
        self.assertEqual(
            bgem3_cache_root("mmlongbench", "doc_embeddings"),
            benchmark_cache_root("mmlongbench", "bgem3") / "doc_embeddings",
        )
        self.assertEqual(
            colbertv2_cache_root("mmlongbench", "query_embeddings"),
            benchmark_cache_root("mmlongbench", "colbertv2") / "query_embeddings",
        )
        self.assertTrue(str(longdocurl_mineru_dir()).endswith("benchmarks/longdocurl/data/processed/pdfs_mineru/4000-4999"))

    def test_longdocurl_raw_pdfs_are_not_ignored(self):
        result = subprocess.run(
            ["git", "check-ignore", "-q", "benchmarks/longdocurl/data/raw/pdfs/4000-4999/4000045.pdf"],
            cwd=Path(__file__).resolve().parents[1],
            check=False,
            capture_output=True,
            text=True,
        )

        self.assertEqual(result.stdout, "")
        self.assertEqual(result.stderr, "")
        self.assertEqual(result.returncode, 1)


if __name__ == "__main__":
    unittest.main()
