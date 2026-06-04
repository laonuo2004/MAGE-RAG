import importlib.util
from pathlib import Path


SCRIPT_PATH = (
    Path(__file__).resolve().parents[1]
    / "benchmarks"
    / "longdocurl"
    / "scripts"
    / "mineru_extract_longdocurl.py"
)


def load_mineru_extract_module():
    spec = importlib.util.spec_from_file_location("mineru_extract_longdocurl", SCRIPT_PATH)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_build_mineru_data_id_keeps_short_doc_id():
    module = load_mineru_extract_module()

    assert module.build_mineru_data_id("4001506") == "4001506"


def test_build_mineru_data_id_shortens_long_doc_id_with_hash_suffix():
    module = load_mineru_extract_module()
    doc_id = (
        "csewt7zsecmmbzjufbyx-signature-"
        "24d91a254426c21c3079384270e1f138dc43a271cfe15d6d520d68205855b2a3-"
        "poli-150306115347-conversion-gate01_95"
    )

    data_id = module.build_mineru_data_id(doc_id)

    assert len(data_id) <= 128
    assert data_id.startswith("csewt7zsecmmbzjufbyx-signature-")
    assert data_id != doc_id



def test_get_upload_max_pages_uses_special_limit_for_oversized_longdocurl_pdf():
    module = load_mineru_extract_module()

    assert module.get_upload_max_pages("4134756", 200) == 74
    assert module.get_upload_max_pages("4001506", 200) == 200

def create_pdf(path, page_count):
    import fitz

    doc = fitz.open()
    for _ in range(page_count):
        doc.new_page()
    doc.save(path)
    doc.close()


def page_count(path):
    import fitz

    with fitz.open(path) as doc:
        return len(doc)


def test_prepare_upload_pdf_truncates_to_max_pages(tmp_path):
    module = load_mineru_extract_module()
    source_pdf = tmp_path / "long.pdf"
    create_pdf(source_pdf, 5)

    upload_pdf = module.prepare_upload_pdf(source_pdf, tmp_path, max_pages=3)

    assert upload_pdf != source_pdf
    assert page_count(upload_pdf) == 3
    assert page_count(source_pdf) == 5


def test_prepare_upload_pdf_keeps_short_pdf_path(tmp_path):
    module = load_mineru_extract_module()
    source_pdf = tmp_path / "short.pdf"
    create_pdf(source_pdf, 2)

    upload_pdf = module.prepare_upload_pdf(source_pdf, tmp_path, max_pages=3)

    assert upload_pdf == source_pdf
    assert page_count(upload_pdf) == 2


class FakeLogger:
    def __init__(self):
        self.errors = []
        self.infos = []
        self.warnings = []

    def error(self, message):
        self.errors.append(message)

    def info(self, message):
        self.infos.append(message)

    def warning(self, message):
        self.warnings.append(message)


class FakeResponse:
    status_code = 200

    def json(self):
        return {
            "code": 0,
            "data": {
                "extract_result": [
                    {"file_name": "still_processing.pdf", "state": "processing"}
                ]
            },
        }


class FakeSession:
    def get(self, url, timeout):
        return FakeResponse()


def test_poll_batch_status_processing_state_does_not_log_name_error(tmp_path):
    module = load_mineru_extract_module()
    logger = FakeLogger()
    automator = module.MinerUAutomator("key", str(tmp_path), str(tmp_path), 1, logger)
    automator.session = FakeSession()
    automator.batch_info["batch-1"] = 1

    completed = automator.poll_batch_status("batch-1")

    assert completed is False
    assert logger.errors == []
