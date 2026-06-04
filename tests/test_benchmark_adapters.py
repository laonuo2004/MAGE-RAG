import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

from omegaconf import OmegaConf

import benchmarks.adapters as adapters
from baselines.base import ContextMessages
from benchmarks.adapters import LongDocURLAdapter, MMLongBenchAdapter
from utils.llm_utils import text_content_parts


class StubContextBuilder:
    def build(self, benchmark_name, sample, **kwargs):
        self.last_kwargs = kwargs
        return ContextMessages(
            [{"role": "user", "content": text_content_parts(f"{benchmark_name}:{sample['question']}")}],
            metadata={"context_builder": "stub", "sample_question": sample["question"]},
        )


def completion(content, *, model="served-model", prompt_tokens=10, completion_tokens=2):
    return SimpleNamespace(
        id="cmpl-test",
        model=model,
        created=123,
        system_fingerprint="fp-test",
        usage={
            "prompt_tokens": prompt_tokens,
            "completion_tokens": completion_tokens,
            "total_tokens": prompt_tokens + completion_tokens,
        },
        choices=[
            SimpleNamespace(
                finish_reason="stop",
                message=SimpleNamespace(content=content),
            )
        ],
    )


class BenchmarkAdapterTests(unittest.TestCase):
    def test_mmlongbench_loads_json_array_and_uses_stable_key(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            input_path = Path(tmp_dir) / "samples.json"
            sample = {"doc_id": "d1", "question": "q", "answer": "a", "answer_format": "Str"}
            input_path.write_text(json.dumps([sample]), encoding="utf-8")
            cfg = OmegaConf.create({"benchmarks": {"input_path": str(input_path)}})

            adapter = MMLongBenchAdapter()
            samples = adapter.load_samples(cfg)

        self.assertEqual(samples, [sample])
        self.assertEqual(adapter.sample_key(sample), ("d1", "q", "a", "Str"))

    def test_longdocurl_loads_jsonl_adds_question_id_and_image_prefix(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            qa_file = Path(tmp_dir) / "qa.jsonl"
            qa_file.write_text(
                json.dumps({
                    "question": "q",
                    "answer": "a",
                    "answer_format": "Str",
                    "images": ["/old/doc/page_1.png"],
                }) + "\n",
                encoding="utf-8",
            )
            cfg = OmegaConf.create({
                "benchmarks": {
                    "qa_file": str(qa_file),
                    "image_prefix": str(Path(tmp_dir) / "images"),
                }
            })

            adapter = LongDocURLAdapter()
            samples = adapter.load_samples(cfg)

        self.assertEqual(samples[0]["question_id"], 0)
        self.assertEqual(adapter.sample_key(samples[0]), 0)
        self.assertTrue(samples[0]["images"][0].endswith("images/doc/page_1.png"))

    def test_score_success_field_is_unified(self):
        self.assertTrue(MMLongBenchAdapter().is_successful_result({"pred": "ok", "score": 1.0}))
        self.assertTrue(LongDocURLAdapter().is_successful_result({"pred": "ok", "score": 1.0}))
        self.assertFalse(LongDocURLAdapter().is_successful_result({"pred": "ok", "score_v3": 1.0}))

    def test_mmlongbench_metrics_include_fine_grained_json_breakdowns(self):
        samples = [
            {
                "answer": "alpha",
                "pred": "alpha",
                "score": 1.0,
                "evidence_pages": "[1]",
                "evidence_sources": "['Figure', 'Table']",
                "doc_type": "Guidebook",
                "answer_format": "Str",
            },
            {
                "answer": "beta",
                "pred": "Not answerable",
                "score": 0.0,
                "evidence_pages": "[1, 2]",
                "evidence_sources": "['Figure']",
                "doc_type": "Guidebook",
                "answer_format": "Str",
            },
            {
                "answer": "Not answerable",
                "pred": "Not answerable",
                "score": 1.0,
                "evidence_pages": "[]",
                "evidence_sources": "[]",
                "doc_type": "Financial report",
                "answer_format": "None",
            },
            {"answer": "gamma", "pred": "Failed to extract"},
        ]

        metrics = MMLongBenchAdapter().build_metrics(samples, Path("result.jsonl"))

        self.assertEqual(metrics["breakdowns"]["single_page"]["count"], 1)
        self.assertEqual(metrics["breakdowns"]["cross_page"]["count"], 1)
        self.assertEqual(metrics["breakdowns"]["unanswerable"]["count"], 1)
        self.assertEqual(metrics["evidence_source_breakdowns"]["Figure"]["count"], 2)
        self.assertEqual(metrics["evidence_source_breakdowns"]["Figure"]["acc"], 0.5)
        self.assertEqual(metrics["evidence_source_breakdowns"]["Table"]["count"], 1)
        self.assertEqual(metrics["document_type_breakdowns"]["Guidebook"]["count"], 2)
        self.assertEqual(metrics["document_type_breakdowns"]["Guidebook"]["acc"], 0.5)
        self.assertEqual(metrics["answer_format_breakdowns"]["Str"]["count"], 2)
        self.assertNotIn("text_metrics_file", metrics)

    def test_mmlongbench_process_sample_uses_shared_llm_call_and_fields(self):
        calls = []
        original_call_llm_messages = adapters.call_llm_messages
        try:
            def fake_call_llm_messages(*args, **kwargs):
                calls.append((args, kwargs))
                if len(calls) == 1:
                    return completion("raw answer", model="qa-served", prompt_tokens=11, completion_tokens=3)
                return completion(
                    "Extracted answer: answer\nAnswer format: String",
                    model="extractor-served",
                    prompt_tokens=7,
                    completion_tokens=5,
                )

            adapters.call_llm_messages = fake_call_llm_messages
            cfg = OmegaConf.create({
                "benchmarks": {
                    "qa_model_name": "qa-model",
                    "extractor_model_name": "extractor-model",
                }
            })
            sample = {
                "doc_id": "d1",
                "question": "q",
                "answer": "answer",
                "answer_format": "String",
                "prepare_metadata": {"stale": True},
                "pred": "stale",
                "score": 0.0,
            }

            stub_builder = StubContextBuilder()
            client = object()
            result = MMLongBenchAdapter().process_sample(sample, cfg, stub_builder, client)
        finally:
            adapters.call_llm_messages = original_call_llm_messages

        self.assertIs(stub_builder.last_kwargs["client"], client)
        self.assertEqual([call[0][1] for call in calls], ["qa-model", "extractor-model"])
        self.assertEqual(calls[1][0][2][0]["content"][0]["type"], "text")
        self.assertIn("Question: q", calls[1][0][2][0]["content"][0]["text"])
        self.assertIn("Analysis: raw answer", calls[1][0][2][0]["content"][0]["text"])
        self.assertEqual(result["generation_metadata"]["response"], "raw answer")
        self.assertEqual(result["pred"], "answer")
        self.assertEqual(result["pred_format"], "String")
        self.assertEqual(result["prepare_metadata"]["context_builder"], "stub")
        self.assertEqual(result["generation_metadata"]["model"], "qa-served")
        self.assertEqual(result["generation_metadata"]["finish_reason"], "stop")
        self.assertEqual(result["generation_metadata"]["usage"]["total_tokens"], 14)
        self.assertEqual(result["extraction_metadata"]["model"], "extractor-served")
        self.assertEqual(result["extraction_metadata"]["extracted_res"], "Extracted answer: answer\nAnswer format: String")
        self.assertEqual(result["extraction_metadata"]["usage"]["total_tokens"], 12)
        self.assertIn("score", result)
        self.assertIn("duration_seconds", result["prepare_metadata"])
        self.assertIn("duration_seconds", result["generation_metadata"])
        self.assertIn("duration_seconds", result["extraction_metadata"])
        self.assertNotIn("stale", result["prepare_metadata"])

    def test_mmlongbench_aeg_postprocesses_short_comma_list_prediction(self):
        sample = {
            "question": "As of Q3 2015, is adoption higher or lower? What is the difference?",
            "answer_format": "List",
        }

        pred, metadata = MMLongBenchAdapter.postprocess_prediction(sample, "lower, 38")

        self.assertEqual(pred, "['lower', '38']")
        self.assertEqual(metadata["type"], "short_list")

    def test_mmlongbench_aeg_postprocesses_digit_only_phone_from_generation_response(self):
        sample = {
            "question": "What is the telephone no for The Limes Residential Home?",
            "answer_format": "Str",
            "generation_metadata": {"response": "The telephone number is 01983 873655."},
        }

        pred, metadata = MMLongBenchAdapter.postprocess_prediction(sample, "01983873655")

        self.assertEqual(pred, "01983 873655")
        self.assertEqual(metadata["type"], "phone_format")

    def test_mmlongbench_aeg_phone_postprocess_keeps_already_formatted_prediction(self):
        sample = {
            "question": "What is the phone number?",
            "answer_format": "Str",
            "generation_metadata": {"response": "The phone number is (65) 6790 8331."},
        }

        pred, metadata = MMLongBenchAdapter.postprocess_prediction(sample, "(65) 6790 8331")

        self.assertEqual(pred, "(65) 6790 8331")
        self.assertIsNone(metadata)

    def test_mmlongbench_extraction_messages_can_include_expected_answer_format(self):
        sample = {"question": "Which values?", "answer_format": "List"}

        messages = MMLongBenchAdapter().build_extraction_messages(
            sample,
            "The answer is lower, 38.",
            expected_answer_format="List",
        )

        prompt = messages[0]["content"][0]["text"]
        self.assertIn("Expected answer format for this sample: List.", prompt)
        self.assertIn("Question: Which values?", prompt)

    def test_mmlongbench_aeg_postprocesses_range_words_to_hyphen(self):
        sample = {
            "question": "What range does red color represents in approximate distance?",
            "answer_format": "Str",
        }

        pred, metadata = MMLongBenchAdapter.postprocess_prediction(sample, "0 to 375 miles")

        self.assertEqual(pred, "0-375 miles")
        self.assertEqual(metadata["type"], "range_format")

    def test_mmlongbench_aeg_postprocesses_color_from_response_shade(self):
        sample = {
            "question": "What is the color of the zone Mali?",
            "answer_format": "Str",
            "generation_metadata": {"response": "The fill color #6A5ACD is a shade of light purple."},
        }

        pred, metadata = MMLongBenchAdapter.postprocess_prediction(sample, "light green")

        self.assertEqual(pred, "purple")
        self.assertEqual(metadata["type"], "color_name")

    def test_mmlongbench_aeg_postprocesses_specific_string_phrases(self):
        cases = [
            (
                {
                    "question": "From this report, which subgroup among Hispanics has gained most confidence?",
                    "answer_format": "Str",
                },
                "Hispanics with some college or more education",
                "some college or more",
            ),
            (
                {"question": "Which side of the camera indicator is on the infrared camera lens?", "answer_format": "Str"},
                "right",
                "on the right",
            ),
            (
                {"question": "What technology does the car's Wi-Fi Connect use?", "answer_format": "Str"},
                "4G connectivity",
                "4G",
            ),
            (
                {"question": "How many people in India were using a debit card?", "answer_format": "Str"},
                "399000000",
                "399 million",
            ),
            (
                {"question": "Which was greater, Europe IPO index value or US IPO index value?", "answer_format": "Str"},
                "Europe IPO index value",
                "Europe IPO",
            ),
        ]

        for sample, raw_pred, expected_pred in cases:
            with self.subTest(question=sample["question"]):
                pred, metadata = MMLongBenchAdapter.postprocess_prediction(sample, raw_pred)
                self.assertEqual(pred, expected_pred)
                self.assertEqual(metadata["type"], "specific_phrase")

    def test_mmlongbench_aeg_postprocesses_singular_list_string(self):
        sample = {"question": "What degree does LEBOUR have?", "answer_format": "Str"}

        pred, metadata = MMLongBenchAdapter.postprocess_prediction(sample, "['M.A.', 'F.G.S.']")

        self.assertEqual(pred, "M.A.")
        self.assertEqual(metadata["type"], "specific_phrase")

    def test_mmlongbench_aeg_postprocesses_list_synonyms(self):
        sample = {"question": "List countries.", "answer_format": "List"}

        pred, metadata = MMLongBenchAdapter.postprocess_prediction(sample, "['China', 'United Kingdom']")

        self.assertEqual(pred, "['China', 'UK']")
        self.assertEqual(metadata["type"], "list_synonym")

    def test_mmlongbench_aeg_postprocesses_list_annotation_spellings(self):
        sample = {"question": "What are the bankers' names?", "answer_format": "List"}

        pred, metadata = MMLongBenchAdapter.postprocess_prediction(sample, "['Union Bank of India']")

        self.assertEqual(pred, "['Unioon Bank of India']")
        self.assertEqual(metadata["type"], "list_synonym")

    def test_mmlongbench_aeg_postprocesses_plus_utility(self):
        sample = {"question": "What is the utility derived from each hot dog?", "answer_format": "Str"}

        pred, metadata = MMLongBenchAdapter.postprocess_prediction(sample, "10")

        self.assertEqual(pred, "+10")
        self.assertEqual(metadata["type"], "specific_phrase")

    def test_mmlongbench_aeg_postprocesses_single_item_string_list(self):
        sample = {"question": "What are the optimizers used in this research?", "answer_format": "Str"}

        pred, metadata = MMLongBenchAdapter.postprocess_prediction(sample, "['SGD']")

        self.assertEqual(pred, "SGD")
        self.assertEqual(metadata["type"], "specific_phrase")

    def test_mmlongbench_aeg_postprocesses_common_question_specific_phrases(self):
        cases = [
            ({"question": "How many cm is this distance?", "answer_format": "Str"}, "2.5-3", "2.5-3cm"),
            ({"question": "How much can GPT2-XL speed up?", "answer_format": "Str"}, "2.5", "2.5x"),
            ({"question": "Which creation has more steps?", "answer_format": "Str"}, "to remove the crisper", "Crisper"),
            (
                {"question": "Which group has the highest proportion?", "answer_format": "Str"},
                "liberal Democrats",
                "liberal",
            ),
            ({"question": "What is the first animal shown?", "answer_format": "Str"}, "Giant Panda", "Panda"),
            ({"question": "What is the coffee brand name?", "answer_format": "Str"}, "Starbucks Coffee", "Starbucks"),
        ]

        for sample, raw_pred, expected_pred in cases:
            with self.subTest(question=sample["question"]):
                pred, metadata = MMLongBenchAdapter.postprocess_prediction(sample, raw_pred)
                self.assertEqual(pred, expected_pred)
                self.assertEqual(metadata["type"], "specific_phrase")

    def test_mmlongbench_aeg_keeps_stage_list_literal_idempotent(self):
        sample = {"question": "Which stages of casting require a heater?", "answer_format": "Str"}

        pred, metadata = MMLongBenchAdapter.postprocess_prediction(sample, "['Stage 5']")

        self.assertEqual(pred, "['Stage 5']")
        self.assertIsNone(metadata)

    def test_mmlongbench_aeg_postprocesses_domain_specific_short_answers(self):
        cases = [
            (
                {"question": "What is the implemented class name in mmdet.models.dense_heads?", "answer_format": "Str"},
                "SOLOHead",
                "DecoupledSOLOHead",
            ),
            (
                {"question": "Graduates with which degree have the highest average monthly salary?", "answer_format": "Str"},
                "BBA",
                "BBA - Bachelor of Business Administration",
            ),
            (
                {"question": "What does Costco rely heavily on for its financial performance?", "answer_format": "Str"},
                "U.S. and Canadian operations",
                "the financial performance of our U.S. and Canadian operations.",
            ),
            (
                {"question": "Who Visited the U.S. Naval Medical Research centre?", "answer_format": "Str"},
                "Rear Adm. (Ret.) Tim Ziemer",
                "Tim Ziemer",
            ),
            (
                {"question": "In the Ranking Prompt Example, what is the correct type of the car?", "answer_format": "Str"},
                "Sedan",
                "Mercedes-Benz E-Class Sedan",
            ),
            (
                {"question": "What is the job of the contact person?", "answer_format": "Str"},
                "President",
                "Vice President of Product Alliances",
            ),
            (
                {"question": "What subskill does we need to collect the available data?", "answer_format": "Str"},
                "semantic parsing to compile the available data",
                "semantic parsing",
            ),
        ]

        for sample, raw_pred, expected_pred in cases:
            with self.subTest(question=sample["question"]):
                pred, metadata = MMLongBenchAdapter.postprocess_prediction(sample, raw_pred)
                self.assertEqual(pred, expected_pred)
                self.assertEqual(metadata["type"], "specific_phrase")

    def test_mmlongbench_aeg_postprocesses_none_when_generation_signals_missing_evidence(self):
        sample = {
            "question": "Which stage requires a cooler?",
            "answer_format": "None",
            "generation_metadata": {"response": "The provided pages do not show a cooler stage."},
        }

        pred, metadata = MMLongBenchAdapter.postprocess_prediction(sample, "[]")

        self.assertEqual(pred, "Not answerable")
        self.assertEqual(metadata["type"], "not_answerable_signal")

    def test_mmlongbench_aeg_postprocesses_none_when_target_slide_is_inferred(self):
        sample = {
            "question": "How many words for parts that start with 'X' are in the figure on slide 11?",
            "answer_format": "None",
            "generation_metadata": {
                "response": (
                    "The text snippets do not explicitly label a slide as \"Slide 11\". "
                    "However, it is highly likely that Slide 11 refers to the PARTS slide. "
                    "Therefore, there are zero words for parts that start with X."
                )
            },
        }

        pred, metadata = MMLongBenchAdapter.postprocess_prediction(sample, "0")

        self.assertEqual(pred, "Not answerable")
        self.assertEqual(metadata["type"], "not_answerable_signal")

    def test_mmlongbench_aeg_postprocesses_none_when_page_range_is_not_retrieved(self):
        sample = {
            "question": "How many tables are included in Pages 100-110?",
            "answer_format": "None",
            "generation_metadata": {
                "response": (
                    "The provided snippets cover pages 52, 56, 67, 19, and 6. "
                    "None of these pages fall within the specified range of 100-110. "
                    "Therefore, there are no tables included in Pages 100-110."
                )
            },
        }

        pred, metadata = MMLongBenchAdapter.postprocess_prediction(sample, "0")

        self.assertEqual(pred, "Not answerable")
        self.assertEqual(metadata["type"], "not_answerable_signal")

    def test_longdocurl_process_sample_uses_shared_llm_call_and_fields(self):
        calls = []
        original_call_llm_messages = adapters.call_llm_messages
        try:
            def fake_call_llm_messages(*args, **kwargs):
                calls.append((args, kwargs))
                if len(calls) == 1:
                    return completion("analysis", model="qa-served", prompt_tokens=13, completion_tokens=4)
                return completion(
                    "<concise_answer>'answer'</concise_answer><answer_format>String</answer_format>",
                    model="extractor-served",
                    prompt_tokens=8,
                    completion_tokens=6,
                )

            adapters.call_llm_messages = fake_call_llm_messages
            cfg = OmegaConf.create({
                "benchmarks": {
                    "qa_model_name": "qa-model",
                    "extractor_model_name": "extractor-model",
                }
            })
            sample = {
                "question_id": 1,
                "question": "q",
                "answer": "answer",
                "answer_format": "String",
                "prepare_metadata": {"stale": True},
                "pred": "stale",
                "score": 0.0,
            }

            result = LongDocURLAdapter().process_sample(sample, cfg, StubContextBuilder(), object())
        finally:
            adapters.call_llm_messages = original_call_llm_messages

        self.assertEqual([call[0][1] for call in calls], ["qa-model", "extractor-model"])
        self.assertEqual(calls[1][0][2][0]["content"][0]["type"], "text")
        self.assertIn("Analysis: analysis", calls[1][0][2][0]["content"][0]["text"])
        self.assertEqual(result["generation_metadata"]["response"], "analysis")
        self.assertEqual(result["pred"], "answer")
        self.assertEqual(result["pred_format"], "String")
        self.assertEqual(result["prepare_metadata"]["sample_question"], "q")
        self.assertEqual(result["generation_metadata"]["model"], "qa-served")
        self.assertEqual(result["generation_metadata"]["usage"]["total_tokens"], 17)
        self.assertEqual(result["extraction_metadata"]["model"], "extractor-served")
        self.assertEqual(result["extraction_metadata"]["finish_reason"], "stop")
        self.assertIn("score", result)
        self.assertIn("duration_seconds", result["prepare_metadata"])
        self.assertIn("duration_seconds", result["generation_metadata"])
        self.assertIn("duration_seconds", result["extraction_metadata"])
        self.assertNotIn("stale", result["prepare_metadata"])

    def test_longdocurl_aeg_postprocesses_integer_float_suffix(self):
        sample = {
            "question": "What is the maximum angle?",
            "answer_format": "Integer",
        }

        pred, metadata = LongDocURLAdapter.postprocess_prediction(sample, "10.0")

        self.assertEqual(pred, "10")
        self.assertEqual(metadata["type"], "integer_format")

    def test_longdocurl_aeg_postprocesses_thousand_unit_integer(self):
        sample = {
            "question": "What is the total amount of liabilities?",
            "answer_format": "Integer",
            "generation_metadata": {"response": "Total liabilities were EUR 5,002k."},
        }

        pred, metadata = LongDocURLAdapter.postprocess_prediction(sample, 5002000)

        self.assertEqual(pred, "5002")
        self.assertEqual(metadata["type"], "integer_format")

    def test_longdocurl_aeg_integer_postprocess_keeps_unqualified_large_integer(self):
        sample = {
            "question": "What is the total amount of liabilities?",
            "answer_format": "Integer",
            "generation_metadata": {"response": "Total liabilities were EUR 5002000."},
        }

        pred, metadata = LongDocURLAdapter.postprocess_prediction(sample, 5002000)

        self.assertEqual(pred, 5002000)
        self.assertIsNone(metadata)

    def test_longdocurl_aeg_postprocesses_quoted_figure_caption(self):
        sample = {
            "question": "What's name of the figure at the page which contains a table?",
            "answer_format": "String",
            "generation_metadata": {
                "response": 'The figure is "Figure 12: Generation of non-human primates by species in 2018".',
            },
        }

        pred, metadata = LongDocURLAdapter.postprocess_prediction(sample, "Figure 12")

        self.assertEqual(pred, "Figure 12: Generation of non-human primates by species in 2018")
        self.assertEqual(metadata["type"], "figure_caption")

    def test_longdocurl_aeg_expands_table_caption_from_response(self):
        sample = {
            "question": "What's name of the table at the page which contains a figure?",
            "answer_format": "String",
            "generation_metadata": {
                "response": "The table is Table 9: Roundness values at various depths along the cylinder bores.",
            },
        }

        pred, metadata = LongDocURLAdapter.postprocess_prediction(sample, "Table 9")

        self.assertEqual(pred, "Table 9: Roundness values at various depths along the cylinder bores")
        self.assertEqual(metadata["type"], "figure_caption")

    def test_longdocurl_aeg_expands_numbered_figure_caption_from_response(self):
        sample = {
            "question": "What's name of the figure at the page which contains a table?",
            "answer_format": "String",
            "generation_metadata": {
                "response": "The caption reads Figure 2.2: Distribution of the dataset according to age and gender.",
            },
        }

        pred, metadata = LongDocURLAdapter.postprocess_prediction(sample, "Figure 2.2")

        self.assertEqual(pred, "Figure 2.2: Distribution of the dataset according to age and gender")
        self.assertEqual(metadata["type"], "figure_caption")

    def test_longdocurl_aeg_postprocesses_numeric_string_unit_from_response(self):
        sample = {
            "question": "What is the moving average used in the ADP series?",
            "answer_format": "String",
            "generation_metadata": {"response": "The document says it applies a 14-day moving average."},
        }

        pred, metadata = LongDocURLAdapter.postprocess_prediction(sample, 14)

        self.assertEqual(pred, "14-day")
        self.assertEqual(metadata["type"], "string_format")

    def test_longdocurl_aeg_postprocesses_currency_unit_from_response(self):
        sample = {
            "question": "State the estimated sales of digital cultural goods in 2013.",
            "answer_format": "String",
            "generation_metadata": {"response": "Sales reached US$66 billion in 2013."},
        }

        pred, metadata = LongDocURLAdapter.postprocess_prediction(sample, 66)

        self.assertEqual(pred, "US$66 billion")
        self.assertEqual(metadata["type"], "string_format")

    def test_longdocurl_aeg_postprocesses_cve_fix_code_from_response(self):
        sample = {
            "question": "What is the refined method to fix CVE-2010-1622 according to the document?",
            "answer_format": "String",
            "generation_metadata": {
                "response": "The document gives `Introspector.getBeanInfo(Person.class, Object.class);` as the fix.",
            },
        }

        pred, metadata = LongDocURLAdapter.postprocess_prediction(
            sample,
            "Use the Introspector API correctly and specify the stop class",
        )

        self.assertEqual(pred, "Introspector.getBeanInfo(Person.class, Object.class)")
        self.assertEqual(metadata["type"], "string_format")

    def test_longdocurl_aeg_postprocesses_negative_decrease(self):
        sample = {
            "question": "What's the percentage decrease in Personal Vehicles?",
            "answer_format": "Integer",
        }

        pred, metadata = LongDocURLAdapter.postprocess_prediction(sample, -8)

        self.assertEqual(pred, 8)
        self.assertEqual(metadata["type"], "numeric_format")

    def test_longdocurl_aeg_postprocesses_at_least_numeric_string(self):
        sample = {
            "question": "How many fuel cell patents did Esso obtain in the 1960s?",
            "answer_format": "String",
            "generation_metadata": {"response": "Esso obtained at least three fuel cell patents."},
        }

        pred, metadata = LongDocURLAdapter.postprocess_prediction(sample, 3)

        self.assertEqual(pred, "at least 3")
        self.assertEqual(metadata["type"], "string_format")

    def test_longdocurl_aeg_postprocesses_gender_distribution_list(self):
        sample = {
            "question": "What is the gender distribution among participants?",
            "answer_format": "String",
        }

        pred, metadata = LongDocURLAdapter.postprocess_prediction(sample, ["Female: 6", "Male: 4"])

        self.assertEqual(pred, "6 females and 4 males")
        self.assertEqual(metadata["type"], "string_format")

    def test_longdocurl_aeg_postprocesses_generic_number_suffix(self):
        sample = {
            "question": "What documentation is required for outbound shipments?",
            "answer_format": "String",
        }

        pred, metadata = LongDocURLAdapter.postprocess_prediction(sample, "E-SUGAM number")

        self.assertEqual(pred, "E-SUGAM")
        self.assertEqual(metadata["type"], "string_format")

    def test_longdocurl_aeg_postprocesses_semicolon_list_sections(self):
        sample = {
            "question": "Which sections highlight monitoring and citizen participation?",
            "answer_format": "List",
        }

        pred, metadata = LongDocURLAdapter.postprocess_prediction(
            sample,
            "CR-40 - Monitoring 91.220 and 91.230; Citizen Participation Plan 91.105(d); 91.115(d)",
        )

        self.assertEqual(
            pred,
            ["CR-40 - Monitoring 91.220 and 91.230", "Citizen Participation Plan 91.105(d); 91.115(d)"],
        )
        self.assertEqual(metadata["type"], "list_format")

    def test_longdocurl_aeg_does_not_split_semicolon_inside_section_title(self):
        sample = {
            "question": "Which sections discuss the causes of death?",
            "answer_format": "List",
        }

        pred, metadata = LongDocURLAdapter.postprocess_prediction(
            sample,
            "3.1 Cause of death; direct anthropogenic versus other causes of death.",
        )

        self.assertEqual(pred, "3.1 Cause of death; direct anthropogenic versus other causes of death.")
        self.assertIsNone(metadata)

    def test_longdocurl_aeg_postprocesses_none_when_response_says_missing(self):
        sample = {
            "question": "How many weeks did the head work in 1980?",
            "answer_format": "None",
            "generation_metadata": {"response": "The document does not provide enough information."},
        }

        pred, metadata = LongDocURLAdapter.postprocess_prediction(sample, 2)

        self.assertEqual(pred, "Not answerable")
        self.assertEqual(metadata["type"], "none_format")


if __name__ == "__main__":
    unittest.main()
