from omegaconf import OmegaConf

from benchmarks.adapters import LongDocURLAdapter, MMLongBenchAdapter


class SelfAnsweringBuilder:
    def __init__(self):
        self.calls = []

    def run_sample(self, benchmark_name, sample, client=None):
        self.calls.append((benchmark_name, sample, client))
        return {
            "response": "<output>answer</output>",
            "pred": sample["answer"],
            "pred_format": None,
            "metadata": {"context_builder": "g2-reader", "sample_key": sample.get("question_id")},
            "usage": {"total_tokens": 3},
        }


def test_longdocurl_process_sample_accepts_self_answering_builder():
    cfg = OmegaConf.create({"benchmarks": {"qa_model_name": "unused", "extractor_model_name": "unused"}})
    sample = {
        "question_id": "q1",
        "question": "q",
        "answer": "yes",
        "answer_format": "String",
    }
    builder = SelfAnsweringBuilder()
    client = object()

    result = LongDocURLAdapter().process_sample(sample, cfg, builder, client)

    assert builder.calls[0][0] == "longdocurl"
    assert builder.calls[0][2] is client
    assert result["pred"] == "yes"
    assert result["score"] == 1.0
    assert result["generation_metadata"]["response"] == "<output>answer</output>"
    assert result["generation_metadata"]["usage"] == {"total_tokens": 3}
    assert result["prepare_metadata"]["context_builder"] == "g2-reader"
    assert "extraction_metadata" not in result


def test_mmlongbench_process_sample_accepts_self_answering_builder():
    cfg = OmegaConf.create({"benchmarks": {"qa_model_name": "unused", "extractor_model_name": "unused"}})
    sample = {
        "doc_id": "doc.pdf",
        "question_id": "q1",
        "question": "q",
        "answer": "answer",
        "answer_format": "String",
    }
    builder = SelfAnsweringBuilder()

    result = MMLongBenchAdapter().process_sample(sample, cfg, builder, object())

    assert builder.calls[0][0] == "mmlongbench"
    assert result["pred"] == "answer"
    assert result["score"] == 1.0
    assert result["prepare_metadata"]["context_builder"] == "g2-reader"
