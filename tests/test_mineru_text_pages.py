import json
from pathlib import Path

from benchmarks.utils.mineru_text import build_mineru_text_pages
from benchmarks.utils.document_preprocess import load_longdocurl_vlm_text_pages


def _write_json(path: Path, payload):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")


class FakeCompletions:
    def __init__(self):
        self.kwargs = None

    def create(self, **kwargs):
        self.kwargs = kwargs
        message = type("Message", (), {"content": "The chart shows revenue rising from 10 to 20."})()
        choice = type("Choice", (), {"message": message})()
        return type("Completion", (), {"choices": [choice]})()


class FakeClient:
    def __init__(self):
        self.chat = type("Chat", (), {"completions": FakeCompletions()})()


def test_build_mineru_text_pages_enriches_visual_blocks_with_llm_description(tmp_path):
    mineru_dir = tmp_path / "mineru" / "doc-1"
    _write_json(mineru_dir / "layout.json", {"pdf_info": [{"page_idx": 0, "page_size": [100, 200]}]})
    _write_json(
        mineru_dir / "doc-1_content_list_v2.json",
        [
            [
                {
                    "type": "paragraph",
                    "content": {"paragraph_content": [{"type": "text", "content": "Annual results summary."}]},
                },
                {
                    "type": "chart",
                    "content": {
                        "chart_caption": [{"type": "text", "content": "Figure 1: Revenue trend"}],
                        "chart_footnote": [],
                        "content": "",
                    },
                },
                {
                    "type": "table",
                    "content": {
                        "table_caption": [{"type": "text", "content": "Table 1: Scores"}],
                        "html": "<table><tr><td>A</td><td>42</td></tr></table>",
                        "table_footnote": [],
                    },
                },
            ]
        ],
    )

    artifact = build_mineru_text_pages(mineru_dir=mineru_dir, client=FakeClient(), model_name="vlm-model", overwrite=True)

    page_text = artifact["pages"][0]["text"]
    assert "Annual results summary." in page_text
    assert "Figure 1: Revenue trend" in page_text
    assert "The chart shows revenue rising from 10 to 20." in page_text
    assert "<table><tr><td>A</td><td>42</td></tr></table>" in page_text
    assert (mineru_dir / "vlm_text_pages.json").exists()


def test_longdocurl_vlm_text_loader_prefers_enriched_pages_artifact(tmp_path):
    mineru_root = tmp_path / "mineru"
    doc_dir = mineru_root / "123456"
    _write_json(
        doc_dir / "vlm_text_pages.json",
        {
            "pages": [
                {"page_index": 0, "page_number": 1, "text": "unused page"},
                {"page_index": 1, "page_number": 2, "text": "enriched visual text"},
            ]
        },
    )
    sample = {
        "doc_no": "123456",
        "total_pages": 3,
        "images": [str(tmp_path / "123456_1.png")],
    }

    pages, allowed_pages = load_longdocurl_vlm_text_pages(sample, {"mineru_dir": str(mineru_root)})

    assert allowed_pages == [1]
    assert pages == [{"page_index": 1, "page_number": 2, "text": "enriched visual text"}]
