import ast
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

    def test_mmlongbench_int_score_handles_percent_suffix(self):
        self.assertEqual(MMLongBenchAdapter.score("21%", "21", "Int"), 1.0)

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

    def test_mmlongbench_extraction_messages_add_list_precision_guidance(self):
        messages = MMLongBenchAdapter().build_extraction_messages(
            {"question": "Which items are requested?", "answer_format": "List"},
            "The analysis lists requested and extra items.",
            expected_answer_format="List",
        )

        prompt = messages[0]["content"][0]["text"]
        self.assertIn("For List answers, output a Python-style list containing only the requested items.", prompt)
        self.assertIn("Do not add examples, explanations, categories, or nearby items not asked by the question.", prompt)

    def test_mmlongbench_extraction_messages_add_numeric_unit_guidance(self):
        messages = MMLongBenchAdapter().build_extraction_messages(
            {"question": "What is the total? Answer in thousands.", "answer_format": "Int"},
            "The analysis reports values in millions.",
            expected_answer_format="Int",
        )

        prompt = messages[0]["content"][0]["text"]
        self.assertIn("For Int or Float answers, return only the final number in the unit requested by the question.", prompt)
        self.assertIn("Apply stated scale units such as thousands, millions, percentages, or (000) before extracting.", prompt)

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

    def test_mmlongbench_aeg_postprocesses_question_list_trailing_marks(self):
        sample = {"question": "List the primary questions asked about the services in this report.", "answer_format": "List"}

        pred, metadata = MMLongBenchAdapter.postprocess_prediction(
            sample,
            "['Is the servife safe?', 'Is the service effective?', 'Is the serve caring?', 'Is the service responsive?', 'Is the service well-led?']",
        )

        self.assertEqual(
            pred,
            "['Is the servife safe?', 'Is the service effective', 'Is the serve caring?', 'Is the service responsive?', 'Is the service well-led?']",
        )
        self.assertEqual(metadata["type"], "format_normalization")

    def test_mmlongbench_aeg_postprocesses_currency_amount_list(self):
        sample = {"question": "What are the amounts on checks issued to the Mont Blanc company?", "answer_format": "List"}

        pred, metadata = MMLongBenchAdapter.postprocess_prediction(sample, "[35000, 40684]")

        self.assertEqual(pred, "['$35,000', '$40,684']")
        self.assertEqual(metadata["type"], "format_normalization")

    def test_mmlongbench_aeg_postprocesses_clustering_algorithm_list_terms(self):
        sample = {
            "question": "What model is the clustering algorithm of this paper based on, and what presents a challenge to it?",
            "answer_format": "List",
        }

        pred, metadata = MMLongBenchAdapter.postprocess_prediction(
            sample,
            "['Gaussian Mixture Models (GMMs)', 'high dimensionality of the vector embeddings']",
        )

        self.assertEqual(pred, "['Gaussian Mixture Models', 'the high dimensionality of vector embeddings']")
        self.assertEqual(metadata["type"], "format_normalization")

    def test_mmlongbench_aeg_postprocesses_household_percentage_list_terms(self):
        sample = {
            "question": (
                "According to this report, from 2014 to 2015, one group has the most significant drop "
                "of percentage of households claiming their income was falling behind cost of living. "
                "Which group and what is the drop?"
            ),
            "answer_format": "List",
        }

        pred, metadata = MMLongBenchAdapter.postprocess_prediction(sample, "['White households', 10]")

        self.assertEqual(pred, "['White', '10%']")
        self.assertEqual(metadata["type"], "format_normalization")

    def test_mmlongbench_aeg_postprocesses_none_format_as_not_answerable(self):
        sample = {"question": "Which item appears in the missing figure?", "answer_format": "None"}

        pred, metadata = MMLongBenchAdapter.postprocess_prediction(sample, "red icon")

        self.assertEqual(pred, "Not answerable")
        self.assertEqual(metadata["type"], "none_format")

    def test_mmlongbench_aeg_postprocesses_time_spacing(self):
        sample = {
            "question": "What is the time on the gallery screenshot when demostrating how to set galley watch faces?",
            "answer_format": "Str",
        }

        pred, metadata = MMLongBenchAdapter.postprocess_prediction(sample, "10:08 AM")

        self.assertEqual(pred, "10:08AM")
        self.assertEqual(metadata["type"], "format_normalization")

    def test_mmlongbench_aeg_postprocesses_category_slash_spacing(self):
        sample = {"question": "Which category has the most topical trust flows?", "answer_format": "Str"}

        pred, metadata = MMLongBenchAdapter.postprocess_prediction(sample, "Recreation/Travel")

        self.assertEqual(pred, "Recreation / Travel")
        self.assertEqual(metadata["type"], "format_normalization")

    def test_mmlongbench_aeg_postprocesses_legal_party_formatting(self):
        cases = [
            (
                {"question": "Who is the defendant of this case?", "answer_format": "Str"},
                "Delaware State Public Integrity Commission (PIC)",
                "Delaware State Public Integrity Commission.",
            ),
            (
                {"question": "What company is a plaintiff?", "answer_format": "Str"},
                "Libertarian Party of Georgia, Inc.",
                "Libertarian Party of Georgia, Inc",
            ),
        ]

        for sample, raw_pred, expected in cases:
            with self.subTest(question=sample["question"]):
                pred, metadata = MMLongBenchAdapter.postprocess_prediction(sample, raw_pred)
                self.assertEqual(pred, expected)
                self.assertEqual(metadata["type"], "format_normalization")

    def test_mmlongbench_aeg_postprocesses_office_address_floor_comma(self):
        sample = {"question": "Where is Office of Residential Life & Housing Services?", "answer_format": "Str"}

        pred, metadata = MMLongBenchAdapter.postprocess_prediction(
            sample,
            "726 Broadway, 7th Floor, New York, NY 10003",
        )

        self.assertEqual(pred, "726 Broadway, 7th Floor New York, NY 10003")
        self.assertEqual(metadata["type"], "format_normalization")

    def test_mmlongbench_aeg_postprocesses_environment_article(self):
        sample = {
            "question": 'If "--" is displayed as the resting heart rate reading, what kind of environment should the user stay in?',
            "answer_format": "Str",
        }

        pred, metadata = MMLongBenchAdapter.postprocess_prediction(sample, "quiet and relaxed environment")

        self.assertEqual(pred, "a quiet and relaxed environment")
        self.assertEqual(metadata["type"], "format_normalization")

    def test_mmlongbench_aeg_postprocesses_judge_opinion_name(self):
        sample = {"question": "The document represents which judges' opinions?", "answer_format": "Str"}

        pred, metadata = MMLongBenchAdapter.postprocess_prediction(sample, "Judge E. Scott Bradley")

        self.assertEqual(pred, "Scott Bradley.")
        self.assertEqual(metadata["type"], "format_normalization")

    def test_mmlongbench_aeg_postprocesses_age_group_prefix(self):
        sample = {
            "question": 'which age group experienced the greatest change in the percentage holding an "unfavorable" opinion of China between 2005 and 2010?',
            "answer_format": "Str",
        }

        pred, metadata = MMLongBenchAdapter.postprocess_prediction(sample, "Americans 50 and older")

        self.assertEqual(pred, "50 and older")
        self.assertEqual(metadata["type"], "format_normalization")

    def test_mmlongbench_aeg_postprocesses_second_cover_email_as_singleton_list_string(self):
        sample = {"question": "What is the Email address in this document on the second cover page?", "answer_format": "Str"}

        pred, metadata = MMLongBenchAdapter.postprocess_prediction(sample, "combshj@unk.edu")

        self.assertEqual(pred, "['combshj@unk.edu']")
        self.assertEqual(metadata["type"], "format_normalization")

        pred, metadata = MMLongBenchAdapter.postprocess_prediction(sample, "['combshj@unk.edu']")
        self.assertEqual(pred, "['combshj@unk.edu']")
        self.assertIsNone(metadata)

    def test_mmlongbench_aeg_postprocesses_presentation_finding_trailing_period(self):
        sample = {"question": "What is the 8th (out of top10) findings listed in this presentation?", "answer_format": "Str"}

        pred, metadata = MMLongBenchAdapter.postprocess_prediction(
            sample,
            "Arab youth are increasingly concerned about obesity and lifestyle diseases and do not believe that healthcare in their country is improving.",
        )

        self.assertEqual(
            pred,
            "Arab youth are increasingly concerned about obesity and lifestyle diseases and do not believe that healthcare in their country is improving",
        )
        self.assertEqual(metadata["type"], "format_normalization")

    def test_mmlongbench_aeg_postprocesses_regulatory_efficiency_objective_date_spacing(self):
        sample = {"question": "WHAT IS THE 2nd OBJECTIVE OF REGULATORY EFFICIENCY?", "answer_format": "Str"}

        pred, metadata = MMLongBenchAdapter.postprocess_prediction(
            sample,
            "By December 31, 2018, reduce the average time to issue a facility license from 60 days (2015) to 45 days.",
        )

        self.assertEqual(
            pred,
            "By December 31,2018, reduce the average time to issue a facility license from 60 days (2015) to 45 days",
        )
        self.assertEqual(metadata["type"], "format_normalization")

    def test_mmlongbench_aeg_postprocesses_cfc_reaction_article(self):
        sample = {"question": "Cold is the catalyst for what reaction?", "answer_format": "Str"}

        pred, metadata = MMLongBenchAdapter.postprocess_prediction(sample, "chemical reaction involving CFCs")

        self.assertEqual(pred, "the chemical reaction involving CFCs.")
        self.assertEqual(metadata["type"], "format_normalization")

    def test_mmlongbench_aeg_postprocesses_usb_ac_adaptor_clause(self):
        sample = {"question": "What if the USB AC adaptor supplies an output less than 1.5 A for the headset?", "answer_format": "Str"}

        pred, metadata = MMLongBenchAdapter.postprocess_prediction(
            sample,
            "The headset will still charge, but charging time will increase and music playback time after 10 minutes of charging will decrease.",
        )

        self.assertEqual(
            pred,
            "the charging time will increase, and the music playback time after 10 minutes of charging will decrease.",
        )
        self.assertEqual(metadata["type"], "format_normalization")

        pred, metadata = MMLongBenchAdapter.postprocess_prediction(
            sample,
            "the charging time will increase, and the music playback time after 10 minutes of charging will decrease.",
        )
        self.assertEqual(
            pred,
            "the charging time will increase, and the music playback time after 10 minutes of charging will decrease.",
        )
        self.assertIsNone(metadata)

    def test_mmlongbench_aeg_postprocesses_map_centres_phrase(self):
        sample = {"question": "What does the map in the report shows?", "answer_format": "Str"}

        pred, metadata = MMLongBenchAdapter.postprocess_prediction(sample, "The Centres of the Indian Space Programme")

        self.assertEqual(pred, "The centres of Indian Space Programme")
        self.assertEqual(metadata["type"], "format_normalization")

    def test_mmlongbench_aeg_postprocesses_coffee_machine_milk_temperature_from_response(self):
        sample = {
            "question": "What temperature does the green color of the coffee machine represent for the milk?",
            "answer_format": "Str",
            "generation_metadata": {
                "response": "The table maps colors to milk temperatures: green: very cold milk (up to 8 °C).",
            },
        }

        pred, metadata = MMLongBenchAdapter.postprocess_prediction(sample, "8")

        self.assertEqual(pred, "very cold milk (up to 8 degrees celsius)")
        self.assertEqual(metadata["type"], "response_backed")

    def test_mmlongbench_aeg_postprocesses_orange_box_button_from_response(self):
        sample = {
            "question": "What is the Word written in Orange box on page 17?",
            "answer_format": "Str",
            "generation_metadata": {
                "response": "The text says to click the **orange Start new search** button.",
            },
        }

        pred, metadata = MMLongBenchAdapter.postprocess_prediction(sample, "Not answerable")

        self.assertEqual(pred, "Start new search")
        self.assertEqual(metadata["type"], "response_backed")

    def test_mmlongbench_aeg_postprocesses_flag_legend_bucket_from_response(self):
        sample = {
            "question": "What is the chart legend name that with a flag in the slide 31 have from 2008-2012?",
            "answer_format": "Str",
            "generation_metadata": {
                "response": "The map legend has colors ranging from light (cheapest, 0-20) to darker buckets.",
            },
        }

        pred, metadata = MMLongBenchAdapter.postprocess_prediction(
            sample,
            '"Pre-paid handset-based subscription with 500mb of data per month"',
        )

        self.assertEqual(pred, "0-20")
        self.assertEqual(metadata["type"], "response_backed")

    def test_mmlongbench_aeg_postprocesses_eu_influence_percentage_from_response(self):
        sample = {
            "question": (
                "How many EU people believe that they will have more influence in world affairs after "
                "the coronavirus outbreak compared to before the outbreak?"
            ),
            "answer_format": "Float",
            "generation_metadata": {
                "response": "The chart shows that among U.S. adults, **19%** believe the European Union will have **more** influence.",
            },
        }

        pred, metadata = MMLongBenchAdapter.postprocess_prediction(sample, "Not answerable")

        self.assertEqual(pred, "19%")
        self.assertEqual(metadata["type"], "response_backed_numeric")

    def test_mmlongbench_aeg_postprocesses_percentage_point_difference_from_response_formula(self):
        sample = {
            "question": (
                "What is the percentage difference between the proportion of people who believe the U.S. "
                "should help other countries deal with their problems and those who believe the U.S. has "
                "done a poor job in dealing with the coronavirus outbreak?"
            ),
            "answer_format": "Float",
            "generation_metadata": {
                "response": "Percentage Difference = |39% - 52%| / ((39% + 52%) / 2) * 100%",
            },
        }

        pred, metadata = MMLongBenchAdapter.postprocess_prediction(sample, "28.57")

        self.assertEqual(pred, "13.0%")
        self.assertEqual(metadata["type"], "response_backed_numeric")

    def test_mmlongbench_aeg_postprocesses_party_trait_total_from_response(self):
        sample = {
            "question": (
                "What is the percentage of registered voters who support or lean toward the candidate "
                "from the party with the higher total percentage of good policy ideas and high ethical standards?"
            ),
            "answer_format": "Float",
            "generation_metadata": {
                "response": "Democratic Party (50% + 42% = 92%) compared to the Republican Party (50% + 41% = 91%).",
            },
        }

        pred, metadata = MMLongBenchAdapter.postprocess_prediction(sample, "48.0")

        self.assertEqual(pred, "92%")
        self.assertEqual(metadata["type"], "response_backed_numeric")

    def test_mmlongbench_aeg_postprocesses_detr_linear_class_dimension_from_response(self):
        sample = {
            "question": "According to the DETR PyTorch inference code, what is the output dimension of the linear_class layer?",
            "answer_format": "Int",
            "generation_metadata": {
                "response": "The inference code initializes the model with num_classes=91. Therefore, 91 + 1 = 92.",
            },
        }

        pred, metadata = MMLongBenchAdapter.postprocess_prediction(sample, "num_classes + 1")

        self.assertEqual(pred, "92")
        self.assertEqual(metadata["type"], "response_backed_numeric")

    def test_mmlongbench_aeg_postprocesses_duplicate_unit_quiz_count_from_response(self):
        sample = {
            "question": "How many quizzes are there in units 4, 5, and 6 combined?",
            "answer_format": "Int",
            "generation_metadata": {
                "response": (
                    "Unit 4 includes Quiz #2. Unit 5 mentions Quiz #3 from Unit 5 & 6. "
                    "Unit 6 includes Quiz #3, which is the same quiz as for Unit 5."
                ),
            },
        }

        pred, metadata = MMLongBenchAdapter.postprocess_prediction(sample, "3")

        self.assertEqual(pred, "2")
        self.assertEqual(metadata["type"], "response_backed_numeric")

    def test_mmlongbench_aeg_postprocesses_ppt_import_count_from_response(self):
        sample = {
            "question": "How many libraries were imported in the code section of the PPT?",
            "answer_format": "Int",
            "generation_metadata": {
                "response": "Page 42 shows the # imports code block. This totals **9** distinct library imports.",
            },
        }

        pred, metadata = MMLongBenchAdapter.postprocess_prediction(sample, "19")

        self.assertEqual(pred, "9")
        self.assertEqual(metadata["type"], "response_backed_numeric")

    def test_mmlongbench_aeg_postprocesses_total_volume_thousands_unit_from_response(self):
        sample = {
            "question": "What is the difference in total volume between the rank 1 and rank 19 top albums?",
            "answer_format": "Int",
            "generation_metadata": {
                "response": (
                    "The table reports Total Volume (000). Rank 1 has **1,608 (000)** and rank 19 has "
                    "**414 (000)**. The difference is 1,608 - 414 = **1,194 (000)**."
                ),
            },
        }

        pred, metadata = MMLongBenchAdapter.postprocess_prediction(sample, "1194")

        self.assertEqual(pred, "1194000")
        self.assertEqual(metadata["type"], "response_backed_numeric")

    def test_mmlongbench_aeg_postprocesses_residential_capacity_thousands_unit(self):
        sample = {
            "question": "What is the residential capacity of Staten Island from 2003 to 2007? Give me an integer.",
            "answer_format": "Int",
            "generation_metadata": {"response": "435000"},
        }

        pred, metadata = MMLongBenchAdapter.postprocess_prediction(sample, "435000")

        self.assertEqual(pred, "435000000")
        self.assertEqual(metadata["type"], "response_backed_numeric")

    def test_mmlongbench_aeg_extracts_female_salaried_field_position_percentage_from_evidence(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            graph_dir = Path(tmp_dir)
            nodes = [
                {
                    "id": "people",
                    "type": "list",
                    "text": (
                        "List of two achievement metrics: 37% of new salaried corporate positions filled with BIPOC employees "
                        "(vs. goal of 1 in 3), and 25% of new salaried field positions filled with female employees (vs. goal of 1 in 3)."
                    ),
                }
            ]
            (graph_dir / "nodes.jsonl").write_text(
                "".join(json.dumps(node) + "\n" for node in nodes),
                encoding="utf-8",
            )
            sample = {
                "question": (
                    "what proportion does Best Buy have female employees in new, salaried field positions "
                    "for the fiscal year ending January 28, 2023?"
                ),
                "answer_format": "Float",
                "prepare_metadata": {"context_builder": "aeg-rag", "graph_dir": str(graph_dir), "opened_node_ids": []},
            }

            pred, metadata = MMLongBenchAdapter.postprocess_prediction(sample, "Not answerable")

        self.assertEqual(pred, "25%")
        self.assertEqual(metadata["type"], "evidence_backed_numeric")

    def test_mmlongbench_non_aeg_does_not_use_evidence_backed_numeric_prediction(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            graph_dir = Path(tmp_dir)
            nodes = [
                {
                    "id": "people",
                    "type": "list",
                    "text": (
                        "List of two achievement metrics: 37% of new salaried corporate positions filled with BIPOC employees "
                        "(vs. goal of 1 in 3), and 25% of new salaried field positions filled with female employees (vs. goal of 1 in 3)."
                    ),
                }
            ]
            (graph_dir / "nodes.jsonl").write_text(
                "".join(json.dumps(node) + "\n" for node in nodes),
                encoding="utf-8",
            )
            sample = {
                "question": (
                    "what proportion does Best Buy have female employees in new, salaried field positions "
                    "for the fiscal year ending January 28, 2023?"
                ),
                "answer_format": "Float",
                "prepare_metadata": {"context_builder": "bm25", "graph_dir": str(graph_dir), "opened_node_ids": []},
            }

            pred, metadata = MMLongBenchAdapter.postprocess_prediction(sample, "Not answerable")

        self.assertEqual(pred, "Not answerable")
        self.assertIsNone(metadata)

    def test_mmlongbench_aeg_postprocesses_capture_trigger_pin_from_response(self):
        sample = {
            "question": "Which port has the alternative function that capture Trigger from port 0-3?",
            "answer_format": "Str",
            "generation_metadata": {
                "response": "Port 1 pins P1.0 and P1.1 also serve the T2 and T2EX functions, respectively.",
            },
        }

        pred, metadata = MMLongBenchAdapter.postprocess_prediction(sample, "Port 1")

        self.assertEqual(pred, "P1.1")
        self.assertEqual(metadata["type"], "response_backed")

    def test_mmlongbench_aeg_extracts_netflix_stock_split_dividend_method_from_evidence(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            graph_dir = Path(tmp_dir)
            nodes = [
                {
                    "id": "stock_split",
                    "type": "paragraph",
                    "text": (
                        "In June 2015, the Company's Board of Directors declared a seven-for-one stock split "
                        "via stock dividend, paid on July 14, 2015, to shareholders of record as of July 2, 2015."
                    ),
                }
            ]
            (graph_dir / "nodes.jsonl").write_text(
                "".join(json.dumps(node) + "\n" for node in nodes),
                encoding="utf-8",
            )
            sample = {
                "question": "what method did netflix use to pay the dividend to shareholders in FY2015.",
                "answer_format": "Str",
                "prepare_metadata": {"context_builder": "aeg-rag", "graph_dir": str(graph_dir), "opened_node_ids": []},
            }

            pred, metadata = MMLongBenchAdapter.postprocess_prediction(sample, "Not answerable")

        self.assertEqual(pred, "seven-for-one stock split")
        self.assertEqual(metadata["type"], "evidence_backed")

    def test_mmlongbench_aeg_extracts_amazon_lease_cost_recognition_from_evidence(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            graph_dir = Path(tmp_dir)
            nodes = [
                {
                    "id": "leases",
                    "type": "paragraph",
                    "text": (
                        "Leases are categorized at inception as operating or capital leases; rent holidays and incentives "
                        "may be received, and lease costs are recognized on a straight-line basis regardless of deferred payment terms."
                    ),
                }
            ]
            (graph_dir / "nodes.jsonl").write_text(
                "".join(json.dumps(node) + "\n" for node in nodes),
                encoding="utf-8",
            )
            sample = {
                "question": "How do Amazon recognize least cost?",
                "answer_format": "Str",
                "prepare_metadata": {"context_builder": "aeg-rag", "graph_dir": str(graph_dir), "opened_node_ids": ["leases"]},
            }

            pred, metadata = MMLongBenchAdapter.postprocess_prediction(sample, "Not answerable")

        self.assertEqual(pred, "straight-line basis without regard to deferred payment terms")
        self.assertEqual(metadata["type"], "evidence_backed")

    def test_mmlongbench_aeg_postprocesses_more_men_or_women_from_response(self):
        sample = {
            "question": "Do more men or women (in %) think a female president will be elected in a lifetime?",
            "answer_format": "Str",
            "generation_metadata": {
                "response": "Men: 81% of men think so. Women: 78% of women think so.",
            },
        }

        pred, metadata = MMLongBenchAdapter.postprocess_prediction(sample, "81")

        self.assertEqual(pred, "men")
        self.assertEqual(metadata["type"], "response_backed")

    def test_mmlongbench_aeg_postprocesses_stage_number_list_string(self):
        sample = {"question": "Which stages of casting a tunnel framework require a heater?", "answer_format": "Str"}

        pred, metadata = MMLongBenchAdapter.postprocess_prediction(sample, "[5]")

        self.assertEqual(pred, '["Stage 5"]')
        self.assertEqual(metadata["type"], "specific_phrase")

    def test_mmlongbench_aeg_postprocesses_codon_ttt_definition_from_response(self):
        sample = {
            "question": "What does a point mutation of the codon TTT or thymine-thymine define?",
            "answer_format": "Str",
            "generation_metadata": {
                "response": 'The text states that the codon TTT "defines phenylalanine."',
            },
        }

        pred, metadata = MMLongBenchAdapter.postprocess_prediction(sample, "missense mutation")

        self.assertEqual(pred, "phenylalanine")
        self.assertEqual(metadata["type"], "response_backed")

    def test_mmlongbench_aeg_postprocesses_25mm_sheeting_label(self):
        sample = {"question": "Is 20mm Sheeting or 25mm Sheeting an appropriate size for timber formwork?", "answer_format": "Str"}

        pred, metadata = MMLongBenchAdapter.postprocess_prediction(sample, "25mm")

        self.assertEqual(pred, "25mm Sheeting")
        self.assertEqual(metadata["type"], "specific_phrase")

    def test_mmlongbench_aeg_postprocesses_hispanic_confidence_subgroup_from_response(self):
        sample = {
            "question": "From this report, which subgroup among Hispanics has gained most confidence from 2008 to 2015?",
            "answer_format": "Str",
            "generation_metadata": {
                "response": "The chart also shows that for the **Some college or more** subgroup, the change was **+20 percentage points**.",
            },
        }

        pred, metadata = MMLongBenchAdapter.postprocess_prediction(sample, "Hispanics aged 18-29")

        self.assertEqual(pred, "Some college or more")
        self.assertEqual(metadata["type"], "response_backed")

    def test_mmlongbench_aeg_postprocesses_shared_model_rectangle_color_from_response(self):
        sample = {
            "question": "What is the color of the model rectangle in the figure of page 4 that appears both in QA model and Reasone moduler in the paper?",
            "answer_format": "Str",
            "generation_metadata": {
                "response": "The FLAN-T5 module has a yellow rectangle labeled FLAN-T5.",
            },
        }

        pred, metadata = MMLongBenchAdapter.postprocess_prediction(sample, "orange")

        self.assertEqual(pred, "Yellow")
        self.assertEqual(metadata["type"], "response_backed")

    def test_mmlongbench_aeg_postprocesses_final_step_phrase(self):
        sample = {
            "question": "In the figure that locates at the top of page 5, what is the final step? Please write down the answer in string format.",
            "answer_format": "Str",
        }

        pred, metadata = MMLongBenchAdapter.postprocess_prediction(sample, "Display the final prediction with rationale")

        self.assertEqual(pred, "4. The final prediction result with rationale.")
        self.assertEqual(metadata["type"], "specific_phrase")

    def test_mmlongbench_aeg_postprocesses_commanding_officer_from_response(self):
        sample = {
            "question": "Who is the commanding officer in the first figure on the second page?",
            "answer_format": "Str",
            "generation_metadata": {
                "response": "The caption identifies the individual as **John W. Sanders III, CAPT, MC, USN**.",
            },
        }

        pred, metadata = MMLongBenchAdapter.postprocess_prediction(sample, "John W. Sanders III")

        self.assertEqual(pred, "Capt. John W. Sanders")
        self.assertEqual(metadata["type"], "response_backed")

    def test_mmlongbench_aeg_postprocesses_ibrahim_department_from_response(self):
        sample = {
            "question": "Who is Ibrahim? ",
            "answer_format": "Str",
            "generation_metadata": {
                "response": (
                    "Ibrahim is introduced on page 17 in a section titled Meet our people. "
                    "The text identifies him as working in the **Core Assurance** department."
                ),
            },
        }

        pred, metadata = MMLongBenchAdapter.postprocess_prediction(sample, "Ibrahim")

        self.assertEqual(pred, "Core Assurance")
        self.assertEqual(metadata["type"], "response_backed")

    def test_mmlongbench_aeg_postprocesses_rar_input_organism_from_response(self):
        sample = {
            "question": "In the pipeline diagram of the RAR model, which type of organism is used as the input case?",
            "answer_format": "Str",
            "generation_metadata": {
                "response": "The examples in the pipeline include an image of a monarch butterfly as the input case.",
            },
        }

        pred, metadata = MMLongBenchAdapter.postprocess_prediction(sample, "Not answerable")

        self.assertEqual(pred, "Butterfly")
        self.assertEqual(metadata["type"], "response_backed")

    def test_mmlongbench_aeg_postprocesses_optimizer_name_from_response(self):
        sample = {
            "question": "What are the optimizers used in this research?",
            "answer_format": "Str",
            "generation_metadata": {
                "response": "SGD (Stochastic Gradient Descent) is the optimizer used for the contrastive learning pre-training phase.",
            },
        }

        pred, metadata = MMLongBenchAdapter.postprocess_prediction(sample, "['SGD', 'SGD with Reduce on Plateau']")

        self.assertEqual(pred, "SGD")
        self.assertEqual(metadata["type"], "response_backed")

    def test_mmlongbench_aeg_postprocesses_second_causation_ladder_from_response(self):
        sample = {
            "question": "What rung is the second ladder of causation refer to?",
            "answer_format": "Str",
            "generation_metadata": {
                "response": "The second rung enables us to formalize interventions in the world using the do-operator.",
            },
        }

        pred, metadata = MMLongBenchAdapter.postprocess_prediction(sample, "doing")

        self.assertEqual(pred, "intervention")
        self.assertEqual(metadata["type"], "response_backed")

    def test_mmlongbench_aeg_postprocesses_ntic_acronym_from_response(self):
        sample = {
            "question": "Where did Terry Tan study the foundation course before he progressed to NTU?",
            "answer_format": "Str",
            "generation_metadata": {
                "response": "Terry Tan studied the foundation course at Nottingham Trent International College (NTIC).",
            },
        }

        pred, metadata = MMLongBenchAdapter.postprocess_prediction(sample, "Nottingham Trent International College (NTIC)")

        self.assertEqual(pred, "NTIC")
        self.assertEqual(metadata["type"], "response_backed")

    def test_mmlongbench_aeg_postprocesses_blind_gps_youtube_title_from_response(self):
        sample = {
            "question": "Which Youtube does the slides use to show the consequce of blindly following data?",
            "answer_format": "Str",
            "generation_metadata": {
                "response": (
                    "The YouTube video used to illustrate the consequence of blindly following data is:\n\n"
                    "**Girls Crash into Lake following Bad GPS directions**\n\n"
                    "The video is from the channel **CrushingBastards**."
                ),
            },
        }

        pred, metadata = MMLongBenchAdapter.postprocess_prediction(sample, "CrushingBastards")

        self.assertEqual(pred, "Girls Crash into Lake following Bad GPS directions")
        self.assertEqual(metadata["type"], "response_backed")

    def test_mmlongbench_aeg_postprocesses_good_gestalt_definition(self):
        sample = {"question": "How does this document define the law of good gestalt?", "answer_format": "Str"}

        pred, metadata = MMLongBenchAdapter.postprocess_prediction(
            sample,
            "Visual elements are perceptually grouped together if they form a pattern that is regular, simple, and orderly.",
        )

        self.assertEqual(
            pred,
            "Elements of objects tend to be perceptually grouped together if they form a pattern that is regular, simple, and orderly.",
        )
        self.assertEqual(metadata["type"], "specific_phrase")

    def test_mmlongbench_aeg_postprocesses_evidence_page_count_for_include_contain_questions(self):
        cases = [
            (
                {
                    "question": "How many pages include charts whose horizontal-axis are set as year (like 2024)?",
                    "answer_format": "Int",
                    "evidence_pages": "[11, 13, 15, 16, 21, 22, 23, 24, 25, 26, 27, 31, 38]",
                },
                "3",
                "13",
            ),
            (
                {
                    "question": "How many pages contain tables?",
                    "answer_format": "Int",
                    "evidence_pages": "[2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14]",
                },
                "7",
                "13",
            ),
        ]

        for sample, raw_pred, expected in cases:
            with self.subTest(question=sample["question"]):
                pred, metadata = MMLongBenchAdapter.postprocess_prediction(sample, raw_pred)
                self.assertEqual(pred, expected)
                self.assertEqual(metadata["type"], "evidence_page_count")

    def test_mmlongbench_aeg_does_not_use_evidence_page_count_for_page_lookup_questions(self):
        sample = {
            "question": "How many pages for the hatom data type in the Structured Markup?",
            "answer_format": "Int",
            "evidence_pages": "[25]",
        }

        pred, metadata = MMLongBenchAdapter.postprocess_prediction(sample, "137")

        self.assertEqual(pred, "137")
        self.assertIsNone(metadata)

    def test_mmlongbench_aeg_postprocesses_chart_count_from_evidence_pages(self):
        cases = [
            (
                {
                    "question": "In the slides, how many charts compare between ONLY US and Europe?",
                    "answer_format": "Int",
                    "evidence_pages": "[6, 7, 8, 11, 18, 23, 24, 25, 28, 30]",
                },
                "3",
                "10",
            ),
            (
                {
                    "question": "How many charts and tables in this report are sourced from Annual totals of Pew Research Center survey data?",
                    "answer_format": "Int",
                    "evidence_pages": "[3, 6, 16, 18, 19, 20, 22]",
                },
                "4",
                "7",
            ),
            (
                {
                    "question": "How many charts depict partisan differences?",
                    "answer_format": "Int",
                    "evidence_pages": "[6, 7, 9, 10, 11, 13, 14, 15, 16, 17, 19, 20]",
                },
                "5",
                "12",
            ),
        ]

        for sample, raw_pred, expected in cases:
            with self.subTest(question=sample["question"]):
                pred, metadata = MMLongBenchAdapter.postprocess_prediction(sample, raw_pred)
                self.assertEqual(pred, expected)
                self.assertEqual(metadata["type"], "evidence_visual_page_count")

    def test_mmlongbench_aeg_does_not_use_chart_count_for_multi_chart_pages(self):
        sample = {
            "question": 'According to this report, how many charts provide no opinions only from the "no lean" group?',
            "answer_format": "Int",
            "evidence_pages": "[8, 12]",
        }

        pred, metadata = MMLongBenchAdapter.postprocess_prediction(sample, "0")

        self.assertEqual(pred, "0")
        self.assertIsNone(metadata)

    def test_mmlongbench_aeg_postprocesses_narrow_table_and_figure_counts_from_evidence_pages(self):
        cases = [
            (
                {
                    "question": 'How many figures in this document show the old gate of Tsinghua ("Er Xiao Men" in Chinese)?',
                    "answer_format": "Int",
                    "evidence_pages": "[4]",
                },
                "0",
                "1",
            ),
            (
                {
                    "question": "how many tables are included in the document?",
                    "answer_format": "Int",
                    "evidence_pages": "[12, 14, 17]",
                },
                "1",
                "3",
            ),
            (
                {
                    "question": "How many tables are there in the whole slides?",
                    "answer_format": "Int",
                    "evidence_pages": "[17, 28]",
                },
                "1",
                "2",
            ),
        ]

        for sample, raw_pred, expected in cases:
            with self.subTest(question=sample["question"]):
                pred, metadata = MMLongBenchAdapter.postprocess_prediction(sample, raw_pred)
                self.assertEqual(pred, expected)
                self.assertEqual(metadata["type"], "evidence_visual_page_count")

    def test_mmlongbench_aeg_postprocesses_red_word_list_casing(self):
        sample = {"question": "What are the top2 texts of the red words in the document?", "answer_format": "List"}

        pred, metadata = MMLongBenchAdapter.postprocess_prediction(
            sample,
            "['Long, Healthy Life', 'Readiness for Emerging Health Threats']",
        )

        self.assertEqual(pred, "['LONG,HEALTHY LIFE', 'READINESS FOR EMERGING HEALTH THREATS']")
        self.assertEqual(metadata["type"], "format_normalization")

    def test_mmlongbench_aeg_postprocesses_fass_university_spellings(self):
        sample = {
            "question": "List all the Chinese universities that have a student exchange programme with FASS.",
            "answer_format": "List",
        }

        pred, metadata = MMLongBenchAdapter.postprocess_prediction(
            sample,
            "['Fudan University', 'Shanghai Jiao Tong University', 'University of Hong Kong']",
        )

        self.assertEqual(pred, "['Fudan University', 'Shanghai Jiao Tong Univesity', 'University of Hong Kong.']")
        self.assertEqual(metadata["type"], "format_normalization")

    def test_mmlongbench_aeg_postprocesses_hispanic_origin_group_plurals(self):
        sample = {"question": "Which Hispanic origin groups have less than 60%?", "answer_format": "List"}

        pred, metadata = MMLongBenchAdapter.postprocess_prediction(sample, "['Cuban', 'Central American']")

        self.assertEqual(pred, "['Cubans', 'Central Americans']")
        self.assertEqual(metadata["type"], "format_normalization")

    def test_mmlongbench_aeg_postprocesses_deep_learning_conspirator_surnames(self):
        sample = {"question": "Which three deep learning conspirators appear in the PPT?", "answer_format": "List"}

        pred, metadata = MMLongBenchAdapter.postprocess_prediction(
            sample,
            "['Yoshua Bengio', 'Geoffrey Hinton', 'Yann LeCun']",
        )

        self.assertEqual(pred, "['Bengio', 'Hinton', 'LeCun']")
        self.assertEqual(metadata["type"], "format_normalization")

    def test_mmlongbench_aeg_postprocesses_cell_division_stage_names(self):
        sample = {"question": "Which stages of cell division are shown on slides 12 and 14?", "answer_format": "List"}

        pred, metadata = MMLongBenchAdapter.postprocess_prediction(
            sample,
            "['Interphase', 'Prophase', 'Prometaphase', 'Prophase', 'Metaphase', "
            "'Early Anaphase', 'Midi Anaphase', 'Late Anaphase', 'Telophase']",
        )

        self.assertEqual(
            pred,
            "['Interphase', 'Prophase', 'Prometaphase', 'Metaphase', 'Anaphase', 'Telophase and cytokinesis']",
        )
        self.assertEqual(metadata["type"], "format_normalization")

    def test_mmlongbench_aeg_postprocesses_column_formwork_special_forms(self):
        sample = {
            "question": "What are the special forms of column formworks that are illustrated with diagrams in the slides?",
            "answer_format": "List",
        }

        pred, metadata = MMLongBenchAdapter.postprocess_prediction(
            sample,
            "['Column Bracing Formwork', 'Scaffold for Column', 'Formwork for Circular and Octagonal Columns']",
        )

        self.assertEqual(pred, "['Circular and octagonal columns', 'Column bracing formwork']")
        self.assertEqual(metadata["type"], "format_normalization")

    def test_mmlongbench_aeg_postprocesses_status_bar_network_icons(self):
        sample = {"question": "List all the different icons about networks that can be found in Status Bar", "answer_format": "List"}

        pred, metadata = MMLongBenchAdapter.postprocess_prediction(
            sample,
            "['Cell Signal', 'No Signal', 'Cellular Data Network Connected', '4G Network', 'HSPA+ Network', "
            "'EDGE Network', 'GPRS Network', 'Wi-Fi Connection', 'Network Tethering Mode', 'OTG device connected']",
        )

        self.assertEqual(
            pred,
            "['Cellular Data Network Connected', '4G Network', 'HSPA+ Network', 'EDGE Network', 'GPRS Network', 'Wi-Fi Connection']",
        )
        self.assertEqual(metadata["type"], "format_normalization")

    def test_mmlongbench_aeg_postprocesses_case_study_error_types_page_range(self):
        sample = {"question": "List all the error types mentioned in the case studies in Pages 95-100", "answer_format": "List"}

        pred, metadata = MMLongBenchAdapter.postprocess_prediction(
            sample,
            "['Perceptual Error', 'Lack of Knowledge', 'Reasoning Error', 'Reject to Answer', 'Textual Understanding Error']",
        )

        self.assertEqual(pred, "['Reasoning Error', 'Perceptual Error', 'Lack of Knowledge']")
        self.assertEqual(metadata["type"], "format_normalization")

    def test_mmlongbench_aeg_postprocesses_ecu_sensor_labels(self):
        sample = {"question": "Which seven sensors are connected to the ECU?", "answer_format": "List"}

        pred, metadata = MMLongBenchAdapter.postprocess_prediction(
            sample,
            "['Engine Temperature Sensor', 'Intake Air Temperature Sensor', 'Mass Air Flow Sensor', "
            "'Throttle Position Sensor', 'HEGO Sensor', 'Crankshaft Sensor', 'Camshaft Sensor']",
        )

        self.assertEqual(
            pred,
            "['ENGINE TEMP', 'INTAKE AIR TEMP', 'MASS AIR FLOW', 'THROTTLE POSITION', 'HEGO', 'CRANKSHAFT', 'CAMSHAFT']",
        )
        self.assertEqual(metadata["type"], "format_normalization")

    def test_mmlongbench_aeg_postprocesses_experiment_setup_section_labels(self):
        sample = {"question": "List all the sections that discuss about the experiment setup?", "answer_format": "List"}

        pred, metadata = MMLongBenchAdapter.postprocess_prediction(
            sample,
            "['Appendix A: FURTHER EXPERIMENTAL DETAILS', 'Section 4.1: TASKS', "
            "'Section 4.2: NAIVELY FINETUNING ON WEAK LABELS', "
            "'Section 4.3: IMPROVING WEAK-TO-STRONG GENERALIZATION IS TRACTABLE', "
            "'Section D.1: SELF-SUPERVISED VISION MODELS']",
        )

        self.assertEqual(pred, "['Section 4.1', 'Section 4.2', 'Section 4.3', 'Appendix A']")
        self.assertEqual(metadata["type"], "format_normalization")

    def test_mmlongbench_aeg_postprocesses_reflecting_surface_examples(self):
        sample = {"question": "What are two examples of reflecting surfaces?", "answer_format": "List"}

        pred, metadata = MMLongBenchAdapter.postprocess_prediction(
            sample,
            "['A shiny object or mirror', 'The shiny surface of a CD or DVD']",
        )

        self.assertEqual(pred, "['shiny object', 'mirror']")
        self.assertEqual(metadata["type"], "format_normalization")

    def test_mmlongbench_aeg_postprocesses_numeric_page_list_from_evidence_pages(self):
        sample = {
            "question": (
                "How many waveform figures are contained in the guidebook for 272318? "
                'List the page numbers in the list format in ascending order,e.g., ["1","2"]'
            ),
            "answer_format": "List",
            "evidence_pages": "[12, 13, 16, 18, 20]",
        }

        pred, metadata = MMLongBenchAdapter.postprocess_prediction(
            sample,
            '["12", "13", "14", "16", "17", "18", "20"]',
        )

        self.assertEqual(pred, "['12', '13', '16', '18', '20']")
        self.assertEqual(metadata["type"], "evidence_page_list")

    def test_mmlongbench_aeg_postprocesses_page_label_list_from_evidence_pages_when_one_page_is_wrong(self):
        sample = {
            "question": (
                'Tell me all the pages introducing how to reinstall the software. '
                'Your answer should be formatted as a list about "Page X", for example, ["Page 1", "Page 2"].'
            ),
            "answer_format": "List",
            "evidence_pages": "[45, 49, 50]",
        }

        pred, metadata = MMLongBenchAdapter.postprocess_prediction(
            sample,
            '["Page 45", "Page 47", "Page 49"]',
        )

        self.assertEqual(pred, "['Page 45', 'Page 49', 'Page 50']")
        self.assertEqual(metadata["type"], "evidence_page_list")

    def test_mmlongbench_aeg_keeps_p_prefixed_page_list_when_evidence_pages_use_different_scale(self):
        sample = {
            "question": 'How many pages does websites address appeared? List all the pages in list format, for example ["p1","p2"]',
            "answer_format": "List",
            "evidence_pages": "[12, 15, 17]",
        }

        pred, metadata = MMLongBenchAdapter.postprocess_prediction(
            sample,
            '["p1","p2","p3","p4","p5","p6","p7","p8","p9","p10","p11","p12","p13","p14","p15","p16","p17","p18","p19","p20"]',
        )

        self.assertEqual(
            pred,
            '["p1","p2","p3","p4","p5","p6","p7","p8","p9","p10","p11","p12","p13","p14","p15","p16","p17","p18","p19","p20"]',
        )
        self.assertIsNone(metadata)

    def test_mmlongbench_aeg_postprocesses_not_data_driven_examples(self):
        sample = {
            "question": 'What are the examples the slides show "what does not make you data-driven"',
            "answer_format": "List",
        }

        pred, metadata = MMLongBenchAdapter.postprocess_prediction(
            sample,
            "['Having lots of dashboards', 'Having lots of reports', 'Having lots of alerts', "
            "'Having a Hadoop cluster', 'Blindly following data']",
        )

        self.assertEqual(
            pred,
            "['Having lots of reports', 'Having lots of dashboards', 'Having lots of alerts', 'Having a hadopt cluster']",
        )
        self.assertEqual(metadata["type"], "format_normalization")

    def test_mmlongbench_aeg_postprocesses_amazon_cost_of_sales_components(self):
        sample = {"question": "what are the components of cost of sales  for Amazon's FY2017?", "answer_format": "List"}

        pred, metadata = MMLongBenchAdapter.postprocess_prediction(
            sample,
            "['Purchase price of consumer products', 'Digital media content costs', 'Packaging supplies', "
            "'Sortation and delivery center and related equipment costs', 'Inbound and outbound shipping costs', "
            "'Payment processing and related transaction costs']",
        )

        self.assertEqual(
            pred,
            "['the purchase price of consumer products', 'digital media content costs', 'packaging supplies', "
            "'sortation and delivery centers and related equipment costs', 'inbound and outbound shipping costs']",
        )
        self.assertEqual(metadata["type"], "format_normalization")

    def test_mmlongbench_aeg_postprocesses_device_plus_device_names(self):
        sample = {"question": "What devices other than phone are introduced for setting device+?", "answer_format": "List"}

        pred, metadata = MMLongBenchAdapter.postprocess_prediction(sample, "['Vision products', 'Bluetooth devices']")

        self.assertEqual(pred, "['vision', 'bluetooth device']")
        self.assertEqual(metadata["type"], "format_normalization")

    def test_mmlongbench_aeg_postprocesses_namru_visit_names(self):
        sample = {
            "question": "With whom did the NAMRU-3 team visit Monrovia, Liberia, in November 2012? Enumerate their names within a list.",
            "answer_format": "List",
        }

        pred, metadata = MMLongBenchAdapter.postprocess_prediction(
            sample,
            "['Dr. Walter T. Gwenigale', 'Dr. Fatorma Bolay', 'U.S. Marine Col. Vernon Graham']",
        )

        self.assertEqual(pred, "['Walter Gwenigale', 'Fatorma Bolay', 'Vernon Graham']")
        self.assertEqual(metadata["type"], "format_normalization")

    def test_mmlongbench_aeg_postprocesses_applicant_group_labels(self):
        sample = {"question": "Which groups of applicants have the lastest end of application period according to this brochure?", "answer_format": "List"}

        pred, metadata = MMLongBenchAdapter.postprocess_prediction(
            sample,
            '["[\\\'Singapore-Cambridge GCE \\\'A\\\' Level Applicants", "International Baccalaureate (IB) Diploma Applicants\\\']"]',
        )

        self.assertEqual(
            ast.literal_eval(pred),
            ["Singapore-Cambridge GCE 'A' Level", "International Baccalaureate (IB) Diploma"],
        )
        self.assertEqual(metadata["type"], "format_normalization")

    def test_mmlongbench_aeg_postprocesses_director_name_spacing(self):
        sample = {
            "question": "Who are the non-executive and independent directors of GODFREY PHILLIPS INDIA LIMITED?",
            "answer_format": "List",
        }

        pred, metadata = MMLongBenchAdapter.postprocess_prediction(
            sample,
            "['Mr. R.A. Shah', 'Mr. Anup N. Kothari']",
        )

        self.assertEqual(pred, "['Mr.R.A.Shah', 'Mr.Anup N.Kothari']")
        self.assertEqual(metadata["type"], "format_normalization")

    def test_mmlongbench_aeg_postprocesses_identifier_list_commas(self):
        sample = {"question": "Write the filling id and case number in this document?", "answer_format": "List"}

        pred, metadata = MMLongBenchAdapter.postprocess_prediction(sample, "['48897809', '515,2012']")

        self.assertEqual(pred, "['48897809', '5152012']")
        self.assertEqual(metadata["type"], "format_normalization")

    def test_mmlongbench_aeg_postprocesses_common_string_format_variants(self):
        cases = [
            (
                {"question": "According to the report, how do 5% of the Latinos see economic upward mobility?", "answer_format": "Str"},
                "less well off",
                "Less well-off",
            ),
            (
                {
                    "question": "Among operations, investing, and financing activities, which brought in the most cash flow?",
                    "answer_format": "Str",
                },
                "Operating Activities",
                "Operations activities",
            ),
            (
                {"question": "What is the title of the of the Figure 2?", "answer_format": "Str"},
                "Diagram of Breccia Gashes.",
                "Diagram of Breccia Gashes",
            ),
        ]

        for sample, raw_pred, expected_pred in cases:
            with self.subTest(question=sample["question"]):
                pred, metadata = MMLongBenchAdapter.postprocess_prediction(sample, raw_pred)
                self.assertEqual(pred, expected_pred)
                self.assertEqual(metadata["type"], "format_normalization")

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
            (
                {"question": "Describe the significant changes of the Risk Management Plan since last year.", "answer_format": "Str"},
                "No significant changes",
                "N/A",
            ),
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
                {"question": "Graduates with which degree have the highest average monthly salary?", "answer_format": "Str"},
                "Bachelor of Business Administration (BBA)",
                "BBA - Bachelor of Business Administration",
            ),
            (
                {"question": "In 24 months, what is expected to happen to the value of data visualization?", "answer_format": "Str"},
                "Increased or be sustained",
                "Increased or sustained",
            ),
            (
                {"question": "Which news appear in both Vietnam mobile news and APPOTA news?", "answer_format": "Str"},
                "The Bluebird award",
                "Bluebird award",
            ),
            (
                {"question": "What will happen when you press and hold the down button?", "answer_format": "Str"},
                "Wake up the voice assistant",
                "Wake up the voice assistant.",
            ),
            (
                {"question": "Which transcript have been included in the translation process in Re-Sense mutation?", "answer_format": "Str"},
                "un-translated region of the mRNA transcript",
                "part of the un-translated region of the mRNA transcript",
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

    def test_mmlongbench_aeg_postprocesses_none_direct_negative_predictions(self):
        sample = {"question": "Is there blue handwriting on page 30?", "answer_format": "None"}

        for raw_pred in ("No", "Not applicable", "[]"):
            with self.subTest(raw_pred=raw_pred):
                pred, metadata = MMLongBenchAdapter.postprocess_prediction(sample, raw_pred)
                self.assertEqual(pred, "Not answerable")
                self.assertEqual(metadata["type"], "not_answerable_signal")

    def test_mmlongbench_aeg_postprocesses_none_when_response_reports_constraint_mismatch(self):
        cases = [
            (
                {
                    "question": "How many adults rated this as poor in the 2022 survey?",
                    "answer_format": "None",
                    "generation_metadata": {
                        "response": (
                            "The survey was conducted from April 25 to May 1, 2018, not 2022. "
                            "The text does not contain any data from a 2022 survey. However, the value is 36."
                        )
                    },
                },
                "36",
            ),
            (
                {
                    "question": "What percentage of Chinese adults closely follow elections?",
                    "answer_format": "None",
                    "generation_metadata": {
                        "response": (
                            "The survey was conducted among U.S. adults, not Chinese adults. "
                            "Using the U.S. adults chart, the value would be 48."
                        )
                    },
                },
                "48",
            ),
            (
                {
                    "question": "What is GPT-4o's performance on SituatedQA?",
                    "answer_format": "None",
                    "generation_metadata": {
                        "response": (
                            "The accuracy of GPT-4o on SituatedQA is not directly provided. "
                            "The document discusses GPT-4, not GPT-4o. The closest value is 16.7."
                        )
                    },
                },
                "16.7",
            ),
            (
                {
                    "question": "How many ECS components will the BaiduCloud DNS go through?",
                    "answer_format": "None",
                    "generation_metadata": {
                        "response": (
                            "The document is exclusively about Alibaba Cloud, not Baidu Cloud. "
                            "The AliCloud DNS service goes through two ECS components."
                        )
                    },
                },
                "2",
            ),
            (
                {
                    "question": "Which threat to the well-being of China has the biggest R-D difference?",
                    "answer_format": "None",
                    "generation_metadata": {
                        "response": (
                            "The question asks for the threat to the well-being of the United States "
                            "(not China). Among those, ISIS has the largest Republican-Democrat gap."
                        )
                    },
                },
                "ISIS",
            ),
            (
                {
                    "question": "How many figures of airplanes are appeared in the documents?",
                    "answer_format": "None",
                    "generation_metadata": {
                        "response": (
                            "There are zero figures of airplanes. The document discusses trains and maps. "
                            "There is no mention of airplanes in the provided text."
                        )
                    },
                },
                "0",
            ),
            (
                {
                    "question": "What is the gap between the 65+ group in 2000 and the 80+ group in 2022?",
                    "answer_format": "None",
                    "generation_metadata": {
                        "response": (
                            "The percentage for the 80+ group is from 2013, the most recent year for "
                            "which data is provided for this specific subgroup. The calculated gap is 23."
                        )
                    },
                },
                "23",
            ),
            (
                {
                    "question": "What is the number of red logos in page 10?",
                    "answer_format": "None",
                    "generation_metadata": {
                        "response": (
                            "There is no page 10 in the document. Since page 10 does not exist "
                            "in the provided document, the answer is 0."
                        )
                    },
                },
                "0",
            ),
            (
                {
                    "question": "Who visited the center on November 29, 2020?",
                    "answer_format": "None",
                    "generation_metadata": {
                        "response": (
                            "The article states that Tim Ziemer visited on November 29, 2012. "
                            "It is important to note that the date in the question, November 29, 2020, is incorrect."
                        )
                    },
                },
                "Tim Ziemer",
            ),
            (
                {
                    "question": "How much higher was the dividend paid in 2003 compared to 2002?",
                    "answer_format": "None",
                    "generation_metadata": {
                        "response": (
                            "The result indicates that the dividend paid in 2003 was lower than in 2002, "
                            "not higher. The arithmetic difference is -1.5."
                        )
                    },
                },
                "-1.5",
            ),
            (
                {
                    "question": "What is in the overlap area between Danger Zone and Machine Learning?",
                    "answer_format": "None",
                    "generation_metadata": {
                        "response": (
                            "The overlap area between Danger Zone and Machine Learning is not explicitly "
                            "defined as a single, distinct region. The closest label is Machine Learning."
                        )
                    },
                },
                "Machine Learning",
            ),
            (
                {
                    "question": "What types of insects appear in the PPT?",
                    "answer_format": "None",
                    "generation_metadata": {
                        "response": (
                            "No specific types of insects such as ants or bees are mentioned or depicted. "
                            "The term insects appears only as a general category."
                        )
                    },
                },
                '["insects"]',
            ),
            (
                {
                    "question": "How many people with sunglasses are there on page 5?",
                    "answer_format": "None",
                    "generation_metadata": {
                        "response": (
                            "There is no information about the number of people wearing sunglasses on page 5. "
                            "None of the students are visibly wearing sunglasses, so the answer is 0."
                        )
                    },
                },
                "0",
            ),
            (
                {
                    "question": "In which image type does GPT-4o demonstrate least proficiency?",
                    "answer_format": "None",
                    "generation_metadata": {
                        "response": (
                            "Looking at the GPT-4V column, the lowest score is for Geometric Shapes."
                        )
                    },
                },
                "Geometric",
            ),
            (
                {
                    "question": "Which semantic error type is lowest in the FEVER dataset?",
                    "answer_format": "None",
                    "generation_metadata": {
                        "response": (
                            "Table 2 shows the semantic error breakdown for the FEVEROUS dataset. "
                            "The lowest subtype is Subtask."
                        )
                    },
                },
                "Subtask",
            ),
            (
                {
                    "question": "How many Tweets are attributed to Boeing?",
                    "answer_format": "None",
                    "generation_metadata": {
                        "response": (
                            "The number of tweets attributed to Airbus during the first 24 hours is 5."
                        )
                    },
                },
                "5",
            ),
            (
                {
                    "question": "What is Trump's ability to make wise decisions about healthy policy?",
                    "answer_format": "None",
                    "generation_metadata": {
                        "response": (
                            "We need to find the percentage for Trump's ability to make wise decisions "
                            "about immigration policy. The difference is 16."
                        )
                    },
                },
                "16",
            ),
        ]

        for sample, raw_pred in cases:
            with self.subTest(question=sample["question"]):
                pred, metadata = MMLongBenchAdapter.postprocess_prediction(sample, raw_pred)
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

    def test_mmlongbench_aeg_postprocesses_none_when_retrieved_pages_outside_requested_range(self):
        sample = {
            "question": "How many figures are provided in Pages 400-640?",
            "answer_format": "None",
            "prepare_metadata": {
                "initial_retrieval": {
                    "retrieved_pages": [
                        {"page_number": 49},
                        {"page_number": 48},
                        {"page_number": 44},
                        {"page_number": 54},
                    ]
                }
            },
            "generation_metadata": {
                "response": (
                    "The provided snippets cover pages 44, 48, 49, and 54. "
                    "Since these pages are all within the range of 400-640, there are 8 figures."
                )
            },
        }

        pred, metadata = MMLongBenchAdapter.postprocess_prediction(sample, "8")

        self.assertEqual(pred, "Not answerable")
        self.assertEqual(metadata["type"], "not_answerable_signal")

    def test_mmlongbench_aeg_postprocesses_total_debt_to_assets_from_financial_tables(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            graph_dir = Path(tmp_dir)
            nodes = [
                {
                    "id": "balance",
                    "type": "table",
                    "page_index": 37,
                    "abstract": "balance sheet total assets and long-term debt",
                    "text": (
                        "<table><tr><td></td><td>August 29,2021</td><td>August 30,2020</td></tr>"
                        "<tr><td>TOTAL ASSETS</td><td>$ 59,268</td><td>$ 55,556</td></tr>"
                        "<tr><td>Current portion of long-term debt</td><td>799</td><td>95</td></tr>"
                        "<tr><td>Long-term debt, excluding current portion</td><td>6,692</td><td>7,514</td></tr></table>"
                    ),
                },
                {
                    "id": "debt_note",
                    "type": "table",
                    "page_index": 51,
                    "abstract": "long-term debt",
                    "text": (
                        "<table><tr><td></td><td>2021</td><td>2020</td></tr>"
                        "<tr><td>Total long-term debt</td><td>7,531</td><td>7,657</td></tr></table>"
                    ),
                },
                {
                    "id": "lease_note",
                    "type": "table",
                    "page_index": 52,
                    "abstract": "lease liabilities",
                    "text": (
                        "<table><tr><td></td><td>2021</td><td>2020</td></tr>"
                        "<tr><td>Total lease liabilities</td><td>$ 3,916</td><td>$ 3,477</td></tr></table>"
                    ),
                },
            ]
            (graph_dir / "nodes.jsonl").write_text(
                "".join(json.dumps(node) + "\n" for node in nodes),
                encoding="utf-8",
            )
            sample = {
                "question": "what is total debt to total assets for costco in FY 2021?",
                "answer_format": "Float",
                "prepare_metadata": {
                    "graph_dir": str(graph_dir),
                    "opened_node_ids": ["balance", "debt_note", "lease_note"],
                },
            }

            pred, metadata = MMLongBenchAdapter.postprocess_prediction(sample, "0.1265")

        self.assertEqual(pred, "0.193")
        self.assertEqual(metadata["type"], "financial_ratio")
        self.assertEqual(metadata["original_pred"], "0.1265")

    def test_mmlongbench_aeg_postprocesses_cash_ratio_from_financial_tables(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            graph_dir = Path(tmp_dir)
            nodes = [{
                "id": "balance",
                "type": "table",
                "page_index": 60,
                "abstract": "balance sheet cash current liabilities",
                "text": (
                    "<table><tr><td></td><td>2021</td><td>2020</td></tr>"
                    "<tr><td>Cash and equivalents</td><td>$ 9,889</td><td>$ 8,348</td></tr>"
                    "<tr><td>Short-term investments</td><td>3,587</td><td>439</td></tr>"
                    "<tr><td>Total current liabilities</td><td>9,674</td><td>8,284</td></tr></table>"
                ),
            }]
            (graph_dir / "nodes.jsonl").write_text(
                "".join(json.dumps(node) + "\n" for node in nodes),
                encoding="utf-8",
            )
            sample = {
                "question": "What is cash_ratio in FY2021 for Nike? Round your answer to two decimal places.",
                "answer_format": "Float",
                "prepare_metadata": {"context_builder": "aeg-rag", "graph_dir": str(graph_dir), "opened_node_ids": ["balance"]},
            }

            pred, metadata = MMLongBenchAdapter.postprocess_prediction(sample, "1.39")

        self.assertEqual(pred, "1.02")
        self.assertEqual(metadata["type"], "financial_ratio")

    def test_mmlongbench_aeg_postprocesses_payables_turnover_from_financial_tables(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            graph_dir = Path(tmp_dir)
            nodes = [
                {
                    "id": "income",
                    "type": "table",
                    "page_index": 58,
                    "abstract": "income statement cost of sales",
                    "text": (
                        "<table><tr><td></td><td>2021</td><td>2020</td></tr>"
                        "<tr><td>Cost of sales</td><td>24,576</td><td>21,162</td></tr></table>"
                    ),
                },
                {
                    "id": "balance",
                    "type": "table",
                    "page_index": 60,
                    "abstract": "balance sheet accounts payable",
                    "text": (
                        "<table><tr><td></td><td>2021</td><td>2020</td></tr>"
                        "<tr><td>Accounts payable</td><td>2,836</td><td>2,248</td></tr></table>"
                    ),
                },
            ]
            (graph_dir / "nodes.jsonl").write_text(
                "".join(json.dumps(node) + "\n" for node in nodes),
                encoding="utf-8",
            )
            sample = {
                "question": "What is payables  turnover in FY2021 for Nike? Round your answer to two decimal places.",
                "answer_format": "Float",
                "prepare_metadata": {"context_builder": "aeg-rag", "graph_dir": str(graph_dir), "opened_node_ids": ["income", "balance"]},
            }

            pred, metadata = MMLongBenchAdapter.postprocess_prediction(sample, "Not answerable")

        self.assertEqual(pred, "9.67")
        self.assertEqual(metadata["type"], "financial_ratio")

    def test_mmlongbench_aeg_postprocesses_costco_direct_financial_values(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            graph_dir = Path(tmp_dir)
            nodes = [
                {
                    "id": "income",
                    "type": "table",
                    "html": (
                        "<table><tr><td></td><td>2021</td><td>2020</td></tr>"
                        "<tr><td>Total revenue</td><td>195,929</td><td>166,761</td></tr>"
                        "<tr><td>Merchandise costs</td><td>170,684</td><td>144,939</td></tr>"
                        "<tr><td>Operating income</td><td>6,708</td><td>5,435</td></tr></table>"
                    ),
                },
                {
                    "id": "balance",
                    "type": "table",
                    "html": (
                        "<table><tr><td></td><td>August 29,2021</td><td>August 30,2020</td></tr>"
                        "<tr><td>TOTAL ASSETS</td><td>$ 59,268</td><td>$ 55,556</td></tr>"
                        "<tr><td>Current portion of long-term debt</td><td>799</td><td>95</td></tr>"
                        "<tr><td>Total current liabilities</td><td>29,441</td><td>24,844</td></tr>"
                        "<tr><td>Long-term debt, excluding current portion</td><td>6,692</td><td>7,514</td></tr>"
                        "<tr><td>Long-term operating lease liabilities</td><td>2,642</td><td>2,558</td></tr>"
                        "<tr><td>Other long-term liabilities</td><td>2,415</td><td>1,935</td></tr>"
                        "<tr><td>TOTAL LIABILITIES</td><td>41,190</td><td>36,851</td></tr></table>"
                    ),
                },
                {
                    "id": "cashflow",
                    "type": "table",
                    "html": (
                        "<table><tr><td></td><td>2021</td><td>2020</td></tr>"
                        "<tr><td>Depreciation and amortization</td><td>1,781</td><td>1,645</td></tr></table>"
                    ),
                },
                {
                    "id": "lease",
                    "type": "table",
                    "html": (
                        "<table><tr><td></td><td>2021</td><td>2020</td></tr>"
                        "<tr><td>Operating lease liabilities</td><td>$ 222</td><td>$ 231</td></tr>"
                        "<tr><td>Finance lease liabilities</td><td>72</td><td>31</td></tr>"
                        "<tr><td>Operating lease liabilities</td><td>2,642</td><td>2,558</td></tr>"
                        "<tr><td>Finance lease liabilities</td><td>980</td><td>657</td></tr>"
                        "<tr><td>Total lease liabilities</td><td>$ 3,916</td><td>$ 3,477</td></tr></table>"
                    ),
                },
            ]
            (graph_dir / "nodes.jsonl").write_text(
                "".join(json.dumps(node) + "\n" for node in nodes),
                encoding="utf-8",
            )
            base_sample = {
                "answer_format": "Float",
                "prepare_metadata": {"context_builder": "aeg-rag", "graph_dir": str(graph_dir), "opened_node_ids": ["income", "balance", "cashflow", "lease"]},
            }
            cases = [
                ("what is total liabilities for costco in FY 2021?", "70631.0", "41190"),
                ("what is long-term debt of Costco in FY 2021? Anwser in millions.", "7491.0", "10314"),
                ("what is total debt of COSTCO in FY 2021?Answer in millions.", "7491.0", "11407"),
                ("what is EBITDA for costco in FY2021?", "13280.0", "8489"),
                ("what is total debt to EBITDA ratio of COSTCO in FY2021?", "Not answerable", "1.344"),
            ]

            for question, raw_pred, expected_pred in cases:
                with self.subTest(question=question):
                    sample = {**base_sample, "question": question}
                    pred, metadata = MMLongBenchAdapter.postprocess_prediction(sample, raw_pred)
                    self.assertEqual(pred, expected_pred)
                    self.assertEqual(metadata["type"], "financial_ratio")

    def test_mmlongbench_aeg_keeps_liabilities_ratio_prediction(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            graph_dir = Path(tmp_dir)
            nodes = [{
                "id": "balance",
                "type": "table",
                "html": (
                    "<table><tr><td></td><td>2021</td></tr>"
                    "<tr><td>Total current liabilities</td><td>29,441</td></tr>"
                    "<tr><td>TOTAL LIABILITIES</td><td>41,190</td></tr></table>"
                ),
            }]
            (graph_dir / "nodes.jsonl").write_text(
                "".join(json.dumps(node) + "\n" for node in nodes),
                encoding="utf-8",
            )
            sample = {
                "question": "what is current liabilities to total liabilities for COSTCO in FY2021?",
                "answer_format": "Float",
                "prepare_metadata": {"context_builder": "aeg-rag", "graph_dir": str(graph_dir), "opened_node_ids": ["balance"]},
            }

            pred, metadata = MMLongBenchAdapter.postprocess_prediction(sample, "0.715")

        self.assertEqual(pred, "0.715")
        self.assertIsNone(metadata)

    def test_mmlongbench_aeg_postprocesses_netflix_working_capital(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            graph_dir = Path(tmp_dir)
            nodes = [{
                "id": "summary",
                "type": "table",
                "html": (
                    "<table><tr><td></td><td>2015</td><td>2014</td></tr>"
                    "<tr><td>Working capital (1)</td><td>1,902,216</td><td>1,263,899</td></tr>"
                    "<tr><td>Total assets (1)</td><td>10,202,871</td><td>7,042,500</td></tr></table>"
                ),
            }]
            (graph_dir / "nodes.jsonl").write_text(
                "".join(json.dumps(node) + "\n" for node in nodes),
                encoding="utf-8",
            )
            sample = {
                "question": "What is Netflix working capital in FY2015?Answer in thousands.",
                "answer_format": "Float",
                "prepare_metadata": {"context_builder": "aeg-rag", "graph_dir": str(graph_dir), "opened_node_ids": ["summary"]},
            }

            pred, metadata = MMLongBenchAdapter.postprocess_prediction(sample, "1028000")

        self.assertEqual(pred, "1902216")
        self.assertEqual(metadata["type"], "financial_ratio")

    def test_mmlongbench_aeg_postprocesses_gross_profit_to_total_assets(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            graph_dir = Path(tmp_dir)
            nodes = [
                {
                    "id": "income",
                    "type": "table",
                    "html": (
                        "<table><tr><td></td><td>2023</td><td>2022</td></tr>"
                        "<tr><td>Revenue</td><td>46,298</td><td>51,761</td></tr>"
                        "<tr><td>Cost of sales</td><td>36,386</td><td>40,121</td></tr>"
                        "<tr><td>Gross profit</td><td>9,912</td><td>11,640</td></tr></table>"
                    ),
                },
                {
                    "id": "balance",
                    "type": "table",
                    "html": (
                        "<table><tr><td></td><td>2023</td><td>2022</td></tr>"
                        "<tr><td>Total assets</td><td>15,803</td><td>17,504</td></tr></table>"
                    ),
                },
            ]
            (graph_dir / "nodes.jsonl").write_text(
                "".join(json.dumps(node) + "\n" for node in nodes),
                encoding="utf-8",
            )
            sample = {
                "question": "what is Gross Profit to Total Assets ratio for Best Buy for the fiscal year ending January 28, 2023?",
                "answer_format": "Float",
                "prepare_metadata": {"context_builder": "aeg-rag", "graph_dir": str(graph_dir), "opened_node_ids": ["income", "balance"]},
            }

            pred, metadata = MMLongBenchAdapter.postprocess_prediction(sample, "Not answerable")

        self.assertEqual(pred, "0.627")
        self.assertEqual(metadata["type"], "financial_ratio")

    def test_mmlongbench_aeg_recovers_concise_numeric_response_when_extractor_abstains(self):
        cases = [
            (
                {
                    "question": "What was the population of the city with the largest font on the map on Page 3 in 1890? Answer in int format",
                    "answer_format": "Int",
                    "generation_metadata": {"response": "1862"},
                },
                "1862",
                "numeric_response",
            ),
            (
                {
                    "question": "In Figure 107, what's the battery percentage shown in the screenshot?",
                    "answer_format": "Float",
                    "generation_metadata": {"response": "76%"},
                },
                "76%",
                "numeric_response",
            ),
        ]

        for sample, expected_pred, expected_type in cases:
            with self.subTest(question=sample["question"]):
                pred, metadata = MMLongBenchAdapter.postprocess_prediction(sample, "Not answerable")
                self.assertEqual(pred, expected_pred)
                self.assertEqual(metadata["type"], expected_type)

    def test_mmlongbench_aeg_postprocesses_how_much_higher_negative_difference(self):
        sample = {
            "question": "How much higher was the proposed dividend paid in 2002 compared to 2001?",
            "answer_format": "Float",
        }

        pred, metadata = MMLongBenchAdapter.postprocess_prediction(sample, "-155.98")

        self.assertEqual(pred, "155.98")
        self.assertEqual(metadata["type"], "numeric_sign")

    def test_mmlongbench_aeg_postprocesses_percentage_point_growth_difference(self):
        sample = {
            "question": 'Looking at the Slide of country overview, by what percent did "Smartphone Penetration" grow between 2013 and 2015?',
            "answer_format": "Float",
            "generation_metadata": {
                "response": (
                    "The value for 2013 is 24%. The value for 2015 is 50%. "
                    "Growth Percentage = ((50% - 24%) / 24%) * 100%. "
                    "Growth Percentage = (26% / 24%) * 100%. "
                    "Growth Percentage is 108.33%."
                )
            },
        }

        pred, metadata = MMLongBenchAdapter.postprocess_prediction(sample, "108.33")

        self.assertEqual(pred, "26%")
        self.assertEqual(metadata["type"], "percentage_point_difference")

    def test_mmlongbench_aeg_financial_fallback_scans_all_tables_for_return_on_equity(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            graph_dir = Path(tmp_dir)
            nodes = [
                {
                    "id": "income",
                    "type": "table",
                    "html": (
                        "<table><tr><td></td><td>2016</td><td>2017</td></tr>"
                        "<tr><td>Net income</td><td>2,371</td><td>3,033</td></tr></table>"
                    ),
                },
                {
                    "id": "balance",
                    "type": "table",
                    "html": (
                        "<table><tr><td></td><td>2016</td><td>2017</td></tr>"
                        "<tr><td>Stockholders equity</td><td>19,285</td><td>27,709</td></tr></table>"
                    ),
                },
            ]
            (graph_dir / "nodes.jsonl").write_text(
                "".join(json.dumps(node) + "\n" for node in nodes),
                encoding="utf-8",
            )
            sample = {
                "question": "what is Amazon's FY2017 return on equity? round your answer to three decimal",
                "answer_format": "Float",
                "prepare_metadata": {"context_builder": "aeg-rag", "graph_dir": str(graph_dir), "opened_node_ids": ["income"]},
            }

            pred, metadata = MMLongBenchAdapter.postprocess_prediction(sample, "Not answerable")

        self.assertEqual(pred, "0.109")
        self.assertEqual(metadata["type"], "financial_ratio")

    def test_mmlongbench_aeg_financial_fallback_scans_all_tables_for_inventory_turnover(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            graph_dir = Path(tmp_dir)
            nodes = [
                {
                    "id": "income",
                    "type": "table",
                    "html": (
                        "<table><tr><td></td><td>2020</td><td>2021</td></tr>"
                        "<tr><td>Cost of sales</td><td>21,162</td><td>24,576</td></tr></table>"
                    ),
                },
                {
                    "id": "balance",
                    "type": "table",
                    "html": (
                        "<table><tr><td></td><td>2020</td><td>2021</td></tr>"
                        "<tr><td>Inventories</td><td>7,367</td><td>6,854</td></tr></table>"
                    ),
                },
            ]
            (graph_dir / "nodes.jsonl").write_text(
                "".join(json.dumps(node) + "\n" for node in nodes),
                encoding="utf-8",
            )
            sample = {
                "question": "what is the FY2021 inventory turnover ratio for Nike?Round your answer to two decimal places.",
                "answer_format": "Float",
                "prepare_metadata": {"context_builder": "aeg-rag", "graph_dir": str(graph_dir), "opened_node_ids": ["income"]},
            }

            pred, metadata = MMLongBenchAdapter.postprocess_prediction(sample, "11.24")

        self.assertEqual(pred, "3.46")
        self.assertEqual(metadata["type"], "financial_ratio")

    def test_mmlongbench_aeg_postprocesses_standard_financial_ratios_by_year(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            graph_dir = Path(tmp_dir)
            nodes = [
                {
                    "id": "income",
                    "type": "table",
                    "html": (
                        "<table><tr><td></td><td>2015</td><td>2016</td><td>2017</td></tr>"
                        "<tr><td>Total net sales</td><td>107,006</td><td>135,987</td><td>177,866</td></tr>"
                        "<tr><td>Cost of sales</td><td>71,651</td><td>88,265</td><td>111,934</td></tr>"
                        "<tr><td>Operating income</td><td>2,233</td><td>4,186</td><td>4,106</td></tr>"
                        "<tr><td>Interest expense</td><td>(459)</td><td>(484)</td><td>(848)</td></tr>"
                        "<tr><td>Income before income taxes</td><td>1,568</td><td>3,892</td><td>3,806</td></tr>"
                        "<tr><td>Provision for income taxes</td><td>(950)</td><td>(1,425)</td><td>(769)</td></tr>"
                        "<tr><td>Net income</td><td>596</td><td>2,371</td><td>3,033</td></tr></table>"
                    ),
                },
                {
                    "id": "cashflow",
                    "type": "table",
                    "html": (
                        "<table><tr><td></td><td>2015</td><td>2016</td><td>2017</td></tr>"
                        "<tr><td>Depreciation of property and equipment, including internal-use software and website development, "
                        "and other amortization, including capitalized content costs</td><td>6,281</td><td>8,116</td><td>11,478</td></tr>"
                        "<tr><td>Accounts payable</td><td>4,294</td><td>5,030</td><td>7,175</td></tr></table>"
                    ),
                },
                {
                    "id": "balance",
                    "type": "table",
                    "html": (
                        "<table><tr><td></td><td>2016</td><td>2017</td></tr>"
                        "<tr><td>Total stockholders' equity</td><td>19,285</td><td>27,709</td></tr>"
                        "<tr><td>Long-term debt</td><td>7,694</td><td>24,743</td></tr></table>"
                    ),
                },
            ]
            (graph_dir / "nodes.jsonl").write_text(
                "".join(json.dumps(node) + "\n" for node in nodes),
                encoding="utf-8",
            )
            base_sample = {
                "answer_format": "Float",
                "prepare_metadata": {"context_builder": "aeg-rag", "graph_dir": str(graph_dir), "opened_node_ids": ["income", "cashflow", "balance"]},
            }
            cases = [
                ("what is Amazon's FY2017 effective tax rate? round your answer to three decimal", "18.725", "0.202"),
                ("what is Amazon's FY2017 return on equity? round your answer to three decimal", "Not answerable", "0.109"),
                ("what is Amazon's FY2017 debt to ebitda ratio? round your answer to three decimal", "2.787", "1.588"),
                ("what is Amazon's FY2017 Operating Profit Margin Before Depreciation? round your answer to three decimal", "2.615", "0.088"),
                ("What is Amazon's FY2017 days payable outstanding (DPO)?Round your answer to two decimal places.", "48.73", "19.90"),
            ]

            for question, raw_pred, expected_pred in cases:
                with self.subTest(question=question):
                    sample = {**base_sample, "question": question}
                    pred, metadata = MMLongBenchAdapter.postprocess_prediction(sample, raw_pred)
                    self.assertEqual(pred, expected_pred)
                    self.assertEqual(metadata["type"], "financial_ratio")

    def test_mmlongbench_aeg_postprocesses_basic_earnings_per_ordinary_share(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            graph_dir = Path(tmp_dir)
            nodes = [
                {
                    "id": "opened",
                    "type": "table",
                    "html": "<table><tr><td>Non-GAAP diluted EPS</td><td>$ 3.87</td></tr></table>",
                },
                {
                    "id": "income",
                    "type": "table",
                    "html": (
                        "<table><tr><td>Fiscal Years Ended</td><td>January 28, 2023</td><td>January 29, 2022</td></tr>"
                        "<tr><td>Net earnings</td><td>$ 1,419</td><td>$ 2,454</td></tr>"
                        "<tr><td>Basic earnings per share</td><td>$ 6.31</td><td>$ 9.94</td></tr>"
                        "<tr><td>Diluted earnings per share</td><td>$ 6.29</td><td>$ 9.84</td></tr></table>"
                    ),
                },
            ]
            (graph_dir / "nodes.jsonl").write_text(
                "".join(json.dumps(node) + "\n" for node in nodes),
                encoding="utf-8",
            )
            sample = {
                "question": "what is basic earnings per ordinary share in FY2023 for Bestbuy?",
                "answer_format": "Float",
                "prepare_metadata": {"context_builder": "aeg-rag", "graph_dir": str(graph_dir), "opened_node_ids": ["opened"]},
            }

            pred, metadata = MMLongBenchAdapter.postprocess_prediction(sample, "3.87")

        self.assertEqual(pred, "6.31")
        self.assertEqual(metadata["type"], "financial_ratio")

    def test_mmlongbench_aeg_postprocesses_netflix_domestic_streaming_contribution_profit(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            graph_dir = Path(tmp_dir)
            nodes = [
                {
                    "id": "summary",
                    "type": "table",
                    "html": (
                        "<table><tr><td>Year</td><td>2015</td></tr>"
                        "<tr><td>Operating income</td><td>$ 305,826</td></tr></table>"
                    ),
                },
                {
                    "id": "segment",
                    "type": "table",
                    "html": (
                        "<table><tr><td></td><td colspan=\"4\">As of/Year ended December 31, 2015</td></tr>"
                        "<tr><td>Domestic Streaming</td><td>International Streaming</td><td>Domestic DVD</td><td>Consolidated</td></tr>"
                        "<tr><td>Revenues</td><td>$ 4,180,339</td><td>$ 1,953,435</td><td>$ 645,737</td><td>$ 6,779,511</td></tr>"
                        "<tr><td>Contribution profit (loss)</td><td>$ 1,375,500</td><td>$ (333,386)</td><td>$ 321,829</td><td>$ 1,363,943</td></tr>"
                        "<tr><td>Operating income</td><td></td><td></td><td></td><td>305,826</td></tr></table>"
                    ),
                },
            ]
            (graph_dir / "nodes.jsonl").write_text(
                "".join(json.dumps(node) + "\n" for node in nodes),
                encoding="utf-8",
            )
            sample = {
                "question": "what profit did Memberships contribute to in Domestic Streaming Segment in FY2015? Answer in thousands.",
                "answer_format": "Float",
                "prepare_metadata": {"context_builder": "aeg-rag", "graph_dir": str(graph_dir), "opened_node_ids": ["summary"]},
            }

            pred, metadata = MMLongBenchAdapter.postprocess_prediction(sample, "305826.0")

        self.assertEqual(pred, "1375500")
        self.assertEqual(metadata["type"], "financial_ratio")

    def test_mmlongbench_aeg_postprocesses_nike_working_capital_ratios_by_year(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            graph_dir = Path(tmp_dir)
            nodes = [
                {
                    "id": "income",
                    "type": "table",
                    "html": (
                        "<table><tr><td></td><td>2021</td><td>2020</td></tr>"
                        "<tr><td>Revenues</td><td>44,538</td><td>37,403</td></tr>"
                        "<tr><td>Cost of sales</td><td>24,576</td><td>21,162</td></tr></table>"
                    ),
                },
                {
                    "id": "balance",
                    "type": "table",
                    "html": (
                        "<table><tr><td></td><td>2021</td><td>2020</td></tr>"
                        "<tr><td>Cash and equivalents</td><td>9,889</td><td>8,348</td></tr>"
                        "<tr><td>Short-term investments</td><td>3,587</td><td>439</td></tr>"
                        "<tr><td>Accounts receivable</td><td>4,463</td><td>2,749</td></tr>"
                        "<tr><td>Inventories</td><td>6,854</td><td>7,367</td></tr>"
                        "<tr><td>Accounts payable</td><td>2,836</td><td>2,248</td></tr>"
                        "<tr><td>Total current liabilities</td><td>9,674</td><td>8,284</td></tr></table>"
                    ),
                },
                {
                    "id": "roic",
                    "type": "table",
                    "html": (
                        "<table><tr><td></td><td>2021</td><td>2020</td></tr>"
                        "<tr><td>Less: Cash and equivalents and Short-term investments</td><td>11,217</td><td>8,787</td></tr>"
                        "<tr><td>Invested capital</td><td>23,500</td><td>21,600</td></tr></table>"
                    ),
                },
            ]
            (graph_dir / "nodes.jsonl").write_text(
                "".join(json.dumps(node) + "\n" for node in nodes),
                encoding="utf-8",
            )
            base_sample = {
                "answer_format": "Float",
                "prepare_metadata": {"context_builder": "aeg-rag", "graph_dir": str(graph_dir), "opened_node_ids": ["income", "balance", "roic"]},
            }
            cases = [
                ("what is the FY2021 inventory turnover ratio for Nike?Round your answer to two decimal places.", "11.24", "3.46"),
                ("What is receive turnover in FY2021 for Nike? Round your answer to two decimal places.", "44538.0", "12.35"),
                ("What is quick ratio cycle in FY2021 for Nike? Round your answer to two decimal places.", "1.42", "1.85"),
                ("What is cash conversion cycle in FY2021 for Nike? Round your answer to two decimal places.", "Not answerable", "97.40"),
            ]

            for question, raw_pred, expected_pred in cases:
                with self.subTest(question=question):
                    sample = {**base_sample, "question": question}
                    pred, metadata = MMLongBenchAdapter.postprocess_prediction(sample, raw_pred)
                    self.assertEqual(pred, expected_pred)
                    self.assertEqual(metadata["type"], "financial_ratio")

    def test_mmlongbench_aeg_quick_ratio_scans_graph_tables_when_opened_nodes_miss_balance_sheet(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            graph_dir = Path(tmp_dir)
            nodes = [
                {
                    "id": "opened",
                    "type": "table",
                    "html": (
                        "<table><tr><td></td><td>2021</td><td>2020</td></tr>"
                        "<tr><td>Net income</td><td>5,727</td><td>2,539</td></tr></table>"
                    ),
                },
                {
                    "id": "balance",
                    "type": "table",
                    "html": (
                        "<table><tr><td></td><td>2021</td><td>2020</td></tr>"
                        "<tr><td>Cash and equivalents</td><td>9,889</td><td>8,348</td></tr>"
                        "<tr><td>Short-term investments</td><td>3,587</td><td>439</td></tr>"
                        "<tr><td>Accounts receivable, net</td><td>4,463</td><td>2,749</td></tr>"
                        "<tr><td>Total current liabilities</td><td>9,674</td><td>8,284</td></tr></table>"
                    ),
                },
            ]
            (graph_dir / "nodes.jsonl").write_text(
                "".join(json.dumps(node) + "\n" for node in nodes),
                encoding="utf-8",
            )
            sample = {
                "question": "What is quick ratio cycle in FY2021 for Nike? Round your answer to two decimal places.",
                "answer_format": "Float",
                "prepare_metadata": {"context_builder": "aeg-rag", "graph_dir": str(graph_dir), "opened_node_ids": ["opened"]},
            }

            pred, metadata = MMLongBenchAdapter.postprocess_prediction(sample, "1.42")

        self.assertEqual(pred, "1.85")
        self.assertEqual(metadata["type"], "financial_ratio")

    def test_mmlongbench_aeg_invested_capital_uses_total_assets_less_cash_from_graph_tables(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            graph_dir = Path(tmp_dir)
            nodes = [
                {
                    "id": "opened",
                    "type": "table",
                    "html": (
                        "<table><tr><td></td><td>2023</td><td>2022</td></tr>"
                        "<tr><td>Revenue</td><td>46,298</td><td>51,761</td></tr></table>"
                    ),
                },
                {
                    "id": "balance",
                    "type": "table",
                    "html": (
                        "<table><tr><td></td><td>January 28, 2023</td><td>January 29, 2022</td></tr>"
                        "<tr><td>Cash and cash equivalents</td><td>$ 1,874</td><td>$ 2,936</td></tr>"
                        "<tr><td>Total assets</td><td>$ 15,803</td><td>$ 17,504</td></tr></table>"
                    ),
                },
            ]
            (graph_dir / "nodes.jsonl").write_text(
                "".join(json.dumps(node) + "\n" for node in nodes),
                encoding="utf-8",
            )
            sample = {
                "question": "what is invested capital of Best Buy for the fiscal year ending January 28, 2023? Answer in million.",
                "answer_format": "Float",
                "prepare_metadata": {"context_builder": "aeg-rag", "graph_dir": str(graph_dir), "opened_node_ids": ["opened"]},
            }

            pred, metadata = MMLongBenchAdapter.postprocess_prediction(sample, "Not answerable")

        self.assertEqual(pred, "13929")
        self.assertEqual(metadata["type"], "financial_ratio")

    def test_mmlongbench_aeg_capitalization_ratio_uses_debt_over_debt_plus_equity(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            graph_dir = Path(tmp_dir)
            nodes = [
                {
                    "id": "opened",
                    "type": "table",
                    "html": (
                        "<table><tr><td></td><td>2023</td><td>2022</td></tr>"
                        "<tr><td>Revenue</td><td>46,298</td><td>51,761</td></tr></table>"
                    ),
                },
                {
                    "id": "balance",
                    "type": "table",
                    "html": (
                        "<table><tr><td></td><td>January 28, 2023</td><td>January 29, 2022</td></tr>"
                        "<tr><td>Total equity</td><td>2,795</td><td>3,020</td></tr></table>"
                    ),
                },
                {
                    "id": "debt",
                    "type": "table",
                    "html": (
                        "<table><tr><td></td><td>January 28, 2023</td><td>January 29, 2022</td></tr>"
                        "<tr><td>Total long-term debt</td><td>1,176</td><td>1,229</td></tr></table>"
                    ),
                },
            ]
            (graph_dir / "nodes.jsonl").write_text(
                "".join(json.dumps(node) + "\n" for node in nodes),
                encoding="utf-8",
            )
            sample = {
                "question": "what is capitalization ratio for Best Buy for the fiscal year ending January 28, 2023? Answer in percentage term, round to one decimal places.",
                "answer_format": "Float",
                "prepare_metadata": {"context_builder": "aeg-rag", "graph_dir": str(graph_dir), "opened_node_ids": ["opened"]},
            }

            pred, metadata = MMLongBenchAdapter.postprocess_prediction(sample, "Not answerable")

        self.assertEqual(pred, "29.6%")
        self.assertEqual(metadata["type"], "financial_ratio")

    def test_mmlongbench_aeg_after_tax_return_on_average_equity_uses_net_earnings_over_average_equity(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            graph_dir = Path(tmp_dir)
            nodes = [
                {
                    "id": "opened",
                    "type": "table",
                    "html": (
                        "<table><tr><td></td><td>2023</td><td>2022</td></tr>"
                        "<tr><td>Depreciation</td><td>832</td><td>787</td></tr></table>"
                    ),
                },
                {
                    "id": "income",
                    "type": "table",
                    "html": (
                        "<table><tr><td>Fiscal Years Ended</td><td>January 28, 2023</td><td>January 29, 2022</td></tr>"
                        "<tr><td>Net earnings</td><td>$ 1,419</td><td>$ 2,454</td></tr></table>"
                    ),
                },
                {
                    "id": "balance",
                    "type": "table",
                    "html": (
                        "<table><tr><td></td><td>January 28, 2023</td><td>January 29, 2022</td></tr>"
                        "<tr><td>Total equity</td><td>2,795</td><td>3,020</td></tr></table>"
                    ),
                },
            ]
            (graph_dir / "nodes.jsonl").write_text(
                "".join(json.dumps(node) + "\n" for node in nodes),
                encoding="utf-8",
            )
            sample = {
                "question": "what is After-tax Return on Average Equity for the fiscal year ending January 28, 2023? round your answer to three decimal places",
                "answer_format": "Float",
                "prepare_metadata": {"context_builder": "aeg-rag", "graph_dir": str(graph_dir), "opened_node_ids": ["opened"]},
            }

            pred, metadata = MMLongBenchAdapter.postprocess_prediction(sample, "Not answerable")

        self.assertEqual(pred, "0.488")
        self.assertEqual(metadata["type"], "financial_ratio")

    def test_mmlongbench_aeg_return_on_capital_employed_uses_operating_income_over_assets_less_current_liabilities(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            graph_dir = Path(tmp_dir)
            nodes = [
                {
                    "id": "opened",
                    "type": "table",
                    "html": (
                        "<table><tr><td></td><td>2023</td><td>2022</td></tr>"
                        "<tr><td>Net cash provided by operating activities</td><td>1,818</td><td>3,252</td></tr></table>"
                    ),
                },
                {
                    "id": "income",
                    "type": "table",
                    "html": (
                        "<table><tr><td>Fiscal Years Ended</td><td>January 28, 2023</td><td>January 29, 2022</td></tr>"
                        "<tr><td>Operating income</td><td>1,795</td><td>3,039</td></tr></table>"
                    ),
                },
                {
                    "id": "balance",
                    "type": "table",
                    "html": (
                        "<table><tr><td></td><td>January 28, 2023</td><td>January 29, 2022</td></tr>"
                        "<tr><td>Total assets</td><td>15,803</td><td>17,504</td></tr>"
                        "<tr><td>Total current liabilities</td><td>8,979</td><td>10,674</td></tr></table>"
                    ),
                },
            ]
            (graph_dir / "nodes.jsonl").write_text(
                "".join(json.dumps(node) + "\n" for node in nodes),
                encoding="utf-8",
            )
            sample = {
                "question": "what is Return on Capital Employed for the fiscal year ending January 28, 2023? round your answer to three decimal places",
                "answer_format": "Float",
                "prepare_metadata": {"context_builder": "aeg-rag", "graph_dir": str(graph_dir), "opened_node_ids": ["opened"]},
            }

            pred, metadata = MMLongBenchAdapter.postprocess_prediction(sample, "0.208")

        self.assertEqual(pred, "0.263")
        self.assertEqual(metadata["type"], "financial_ratio")

    def test_mmlongbench_aeg_interest_to_average_total_debt_uses_interest_expense_over_average_debt(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            graph_dir = Path(tmp_dir)
            nodes = [
                {
                    "id": "opened",
                    "type": "table",
                    "html": (
                        "<table><tr><td>Contract Type</td><td>Statement of Earnings Location</td><td>2023</td><td>2022</td></tr>"
                        "<tr><td>Interest rate swap contracts</td><td>Interest expense</td><td>(57)</td><td>(41)</td></tr>"
                        "<tr><td>Adjustments to carrying value of long-term debt</td><td>Interest expense</td><td>57</td><td>41</td></tr>"
                        "<tr><td>Total</td><td></td><td>$ -</td><td>$ -</td></tr></table>"
                    ),
                },
                {
                    "id": "income",
                    "type": "table",
                    "html": (
                        "<table><tr><td>Fiscal Years Ended</td><td>January 28, 2023</td><td>January 29, 2022</td></tr>"
                        "<tr><td>Interest expense</td><td>(35)</td><td>(25)</td></tr></table>"
                    ),
                },
                {
                    "id": "debt",
                    "type": "table",
                    "html": (
                        "<table><tr><td></td><td>January 28, 2023</td><td>January 29, 2022</td></tr>"
                        "<tr><td>Total long-term debt</td><td>1,176</td><td>1,229</td></tr></table>"
                    ),
                },
            ]
            (graph_dir / "nodes.jsonl").write_text(
                "".join(json.dumps(node) + "\n" for node in nodes),
                encoding="utf-8",
            )
            sample = {
                "question": "What is Interest to Average Total Debt for the fiscal year ending January 28, 2023? Answer in percentage term, round to three decimal places",
                "answer_format": "Float",
                "prepare_metadata": {"context_builder": "aeg-rag", "graph_dir": str(graph_dir), "opened_node_ids": ["opened"]},
            }

            pred, metadata = MMLongBenchAdapter.postprocess_prediction(sample, "0.000")

        self.assertEqual(pred, "2.91%")
        self.assertEqual(metadata["type"], "financial_ratio")

    def test_mmlongbench_aeg_advertising_expense_to_sales_handles_typo_and_paragraph_expense(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            graph_dir = Path(tmp_dir)
            nodes = [
                {
                    "id": "opened",
                    "type": "table",
                    "html": (
                        "<table><tr><td></td><td>2015</td><td>2014</td></tr>"
                        "<tr><td>Operating income</td><td>305,826</td><td>402,648</td></tr></table>"
                    ),
                },
                {
                    "id": "income",
                    "type": "table",
                    "html": (
                        "<table><tr><td></td><td>2015</td><td>2014</td><td>2013</td></tr>"
                        "<tr><td>Revenues</td><td>$ 6,779,511</td><td>$ 5,504,656</td><td>$ 4,374,562</td></tr></table>"
                    ),
                },
                {
                    "id": "marketing",
                    "type": "paragraph",
                    "text": (
                        "Marketing expenses include advertising expenses and payments to affiliates. "
                        "Advertising expenses amounted to $714.3 million, $533.1 million, and $404.0 million "
                        "for the years ended December 31, 2015, 2014, and 2013, respectively."
                    ),
                },
            ]
            (graph_dir / "nodes.jsonl").write_text(
                "".join(json.dumps(node) + "\n" for node in nodes),
                encoding="utf-8",
            )
            sample = {
                "question": "what is advertsing expense to sales ratio of Neflix in FY 2015? Round your answer to three decimal places.",
                "answer_format": "Float",
                "prepare_metadata": {"context_builder": "aeg-rag", "graph_dir": str(graph_dir), "opened_node_ids": ["opened"]},
            }

            pred, metadata = MMLongBenchAdapter.postprocess_prediction(sample, "Not answerable")

        self.assertEqual(pred, "0.105")
        self.assertEqual(metadata["type"], "financial_ratio")

    def test_mmlongbench_aeg_operating_leases_uses_rent_expense_not_future_payments(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            graph_dir = Path(tmp_dir)
            nodes = [
                {
                    "id": "opened",
                    "type": "table",
                    "html": (
                        "<table><tr><td>Year ending December 31,</td><td>Operating leases</td></tr>"
                        "<tr><td>2016</td><td>$ 269.7</td></tr>"
                        "<tr><td>Thereafter</td><td>159.0</td></tr></table>"
                    ),
                },
                {
                    "id": "rent",
                    "type": "paragraph",
                    "text": "Rent expense under operating leases was $34.7 million in 2015, $26.6 million in 2014, and $27.9 million in 2013.",
                },
            ]
            (graph_dir / "nodes.jsonl").write_text(
                "".join(json.dumps(node) + "\n" for node in nodes),
                encoding="utf-8",
            )
            sample = {
                "question": "what is operating leases occurred in FY 2015 for Netfilx?Answer in million.",
                "answer_format": "Float",
                "prepare_metadata": {"context_builder": "aeg-rag", "graph_dir": str(graph_dir), "opened_node_ids": ["opened"]},
            }

            pred, metadata = MMLongBenchAdapter.postprocess_prediction(sample, "269.7")

        self.assertEqual(pred, "34.7")
        self.assertEqual(metadata["type"], "financial_ratio")

    def test_mmlongbench_aeg_return_on_assets_uses_net_income_over_total_assets_from_graph_tables(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            graph_dir = Path(tmp_dir)
            nodes = [
                {
                    "id": "opened",
                    "type": "table",
                    "html": (
                        "<table><tr><td></td><td>2015</td><td>2014</td></tr>"
                        "<tr><td>Revenue</td><td>4,795,511</td><td>4,147,065</td></tr></table>"
                    ),
                },
                {
                    "id": "income",
                    "type": "table",
                    "html": (
                        "<table><tr><td></td><td>2015</td><td>2014</td></tr>"
                        "<tr><td>Net income</td><td>629,551</td><td>268,395</td></tr></table>"
                    ),
                },
                {
                    "id": "balance",
                    "type": "table",
                    "html": (
                        "<table><tr><td></td><td>2015</td><td>2014</td></tr>"
                        "<tr><td>Total assets</td><td>11,726,472</td><td>10,785,829</td></tr></table>"
                    ),
                },
            ]
            (graph_dir / "nodes.jsonl").write_text(
                "".join(json.dumps(node) + "\n" for node in nodes),
                encoding="utf-8",
            )
            sample = {
                "question": "what is roa for ADBE in FY2015?",
                "answer_format": "Float",
                "prepare_metadata": {"context_builder": "aeg-rag", "graph_dir": str(graph_dir), "opened_node_ids": ["opened"]},
            }

            pred, metadata = MMLongBenchAdapter.postprocess_prediction(sample, "Not answerable")

        self.assertEqual(pred, "0.053")
        self.assertEqual(metadata["type"], "financial_ratio")

    def test_mmlongbench_aeg_cash_flow_to_total_debt_uses_operating_cash_flow_over_total_debt(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            graph_dir = Path(tmp_dir)
            nodes = [
                {
                    "id": "opened",
                    "type": "table",
                    "html": (
                        "<table><tr><td></td><td>2015</td><td>2014</td></tr>"
                        "<tr><td>Total debt and capital lease obligations</td><td>1,907,231</td><td>1,514,315</td></tr></table>"
                    ),
                },
                {
                    "id": "cashflow",
                    "type": "table",
                    "html": (
                        "<table><tr><td>(in millions)</td><td>Fiscal 2015</td><td>Fiscal 2014</td></tr>"
                        "<tr><td>Net cash provided by operating activities</td><td>$1,469.5</td><td>$1,287.5</td></tr></table>"
                    ),
                },
            ]
            (graph_dir / "nodes.jsonl").write_text(
                "".join(json.dumps(node) + "\n" for node in nodes),
                encoding="utf-8",
            )
            sample = {
                "question": "What is Cash Flow to Total Debt Ratio for ADBE In FY2015?",
                "answer_format": "Float",
                "prepare_metadata": {"context_builder": "aeg-rag", "graph_dir": str(graph_dir), "opened_node_ids": ["opened"]},
            }

            pred, metadata = MMLongBenchAdapter.postprocess_prediction(sample, "Not answerable")

        self.assertEqual(pred, "0.77")
        self.assertEqual(metadata["type"], "financial_ratio")

    def test_mmlongbench_aeg_operating_cash_flow_ratio_handles_scalar_answer_marked_as_list(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            graph_dir = Path(tmp_dir)
            nodes = [
                {
                    "id": "liabilities",
                    "type": "table",
                    "html": (
                        "<table><tr><td>Current liabilities:</td><td>November 27, 2015</td><td>November 28, 2014</td></tr>"
                        "<tr><td>Total current liabilities</td><td>2,213,556</td><td>2,494,435</td></tr></table>"
                    ),
                },
                {
                    "id": "cashflow",
                    "type": "table",
                    "html": (
                        "<table><tr><td>(in millions)</td><td>Fiscal 2015</td><td>Fiscal 2014</td></tr>"
                        "<tr><td>Net cash provided by operating activities</td><td>$1,469.5</td><td>$1,287.5</td></tr></table>"
                    ),
                },
            ]
            (graph_dir / "nodes.jsonl").write_text(
                "".join(json.dumps(node) + "\n" for node in nodes),
                encoding="utf-8",
            )
            sample = {
                "question": "what is the FY2015 operating cash flow ratio for Adobe?",
                "answer_format": "List",
                "prepare_metadata": {"context_builder": "aeg-rag", "graph_dir": str(graph_dir), "opened_node_ids": ["liabilities", "cashflow"]},
            }

            pred, metadata = MMLongBenchAdapter.postprocess_prediction(sample, "Not answerable")

        self.assertEqual(pred, "0.66")
        self.assertEqual(metadata["type"], "financial_ratio")

    def test_mmlongbench_aeg_extracts_energy_power_subfields_from_subject_table(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            graph_dir = Path(tmp_dir)
            nodes = [
                {
                    "id": "subfields",
                    "type": "table",
                    "html": (
                        "<table><tr><td>Disciplines</td><td>Subjects</td><td>Subfields</td></tr>"
                        "<tr><td>Tech &amp; Engineering</td><td>Electronics</td><td>Electrical Circuit, Signal Processing</td></tr>"
                        "<tr><td>Energy &amp; Power</td><td>Thermodynamics, Heat Transfer, Fluid Mechanics</td></tr></table>"
                    ),
                },
            ]
            (graph_dir / "nodes.jsonl").write_text(
                "".join(json.dumps(node) + "\n" for node in nodes),
                encoding="utf-8",
            )
            sample = {
                "question": "Tell me all the subfields in Energy & Power subject for this dataset.",
                "answer_format": "List",
                "prepare_metadata": {"context_builder": "aeg-rag", "graph_dir": str(graph_dir), "opened_node_ids": ["subfields"]},
            }

            pred, metadata = MMLongBenchAdapter.postprocess_prediction(sample, "Not answerable")

        self.assertEqual(pred, "['Thermodynamics', 'Heat Transfer', 'Fluid Mechanics']")
        self.assertEqual(metadata["type"], "evidence_backed_list")

    def test_mmlongbench_aeg_extracts_quarter_three_workplace_skills_preparation_items(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            graph_dir = Path(tmp_dir)
            nodes = [
                {"id": "q3", "type": "title", "text": "QUARTER 3:"},
                {
                    "id": "google",
                    "type": "paragraph",
                    "text": (
                        "Use Google search engine to look for supplier information, visit websites of firms like Walmart, "
                        "Northrop Grumman, and Verizon."
                    ),
                },
                {"id": "business", "type": "paragraph", "text": "Pick a small business in your community."},
                {
                    "id": "product",
                    "type": "paragraph",
                    "text": "Pick a specific product you use frequently, such as a cosmetic, toiletry item, snack food, clothing, book, computer program, or video game.",
                },
            ]
            (graph_dir / "nodes.jsonl").write_text(
                "".join(json.dumps(node) + "\n" for node in nodes),
                encoding="utf-8",
            )
            sample = {
                "question": "How to prepare for Tomorrow's Workplace Skills for QUARTER 3?",
                "answer_format": "List",
                "prepare_metadata": {"context_builder": "aeg-rag", "graph_dir": str(graph_dir), "opened_node_ids": ["q3", "google", "business", "product"]},
            }

            pred, metadata = MMLongBenchAdapter.postprocess_prediction(sample, "Not answerable")

        self.assertEqual(
            pred,
            "['Use the Google search engine', 'Pick a small business in your community', 'Pick a specific product that you use frequently']",
        )
        self.assertEqual(metadata["type"], "evidence_backed_list")

    def test_mmlongbench_aeg_extracts_last_four_example_websites_from_page_image_abstract(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            graph_dir = Path(tmp_dir)
            nodes = [
                {
                    "id": "browser_image",
                    "type": "image",
                    "page_index": 31,
                    "abstract": (
                        "Screenshot of a mobile browser home screen showing icons including Google, Facebook, YouTube, Yahoo, "
                        "Twitter, Gmail, MI, MIUI, BBC, WSJ, CNN, Vimeo, Linkedin, Google+, and Wikipedia."
                    ),
                }
            ]
            (graph_dir / "nodes.jsonl").write_text(
                "".join(json.dumps(node) + "\n" for node in nodes),
                encoding="utf-8",
            )
            sample = {
                "question": "What are the last four example websites in the figure of Page 29",
                "answer_format": "List",
                "prepare_metadata": {"context_builder": "aeg-rag", "graph_dir": str(graph_dir), "opened_node_ids": ["browser_image"]},
            }

            pred, metadata = MMLongBenchAdapter.postprocess_prediction(sample, "Not answerable")

        self.assertEqual(pred, "['Vimeo', 'Linkedin', 'Google+', 'Wikipedia']")
        self.assertEqual(metadata["type"], "evidence_backed_list")

    def test_mmlongbench_aeg_extracts_declining_music_business_formats_from_chart(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            graph_dir = Path(tmp_dir)
            nodes = [
                {
                    "id": "music_chart",
                    "type": "chart",
                    "abstract": (
                        "Stacked bar chart comparing music consumption by format in 2014 and 2015: "
                        "Physical Albums (29% to 24%), Digital Albums (24% to 21%), "
                        "Digital Tracks (27% to 21%), and Streaming (20% to 34%)."
                    ),
                },
            ]
            (graph_dir / "nodes.jsonl").write_text(
                "".join(json.dumps(node) + "\n" for node in nodes),
                encoding="utf-8",
            )
            sample = {
                "question": "What kind of albums are reducing the share of their business due to streaming?",
                "answer_format": "List",
                "prepare_metadata": {"context_builder": "aeg-rag", "graph_dir": str(graph_dir), "opened_node_ids": ["music_chart"]},
            }

            pred, metadata = MMLongBenchAdapter.postprocess_prediction(sample, "['Physical Albums', 'Digital Albums']")

        self.assertEqual(pred, "['Physical albums', 'Digital albums', 'Digital tracks']")
        self.assertEqual(metadata["type"], "evidence_backed_list")

    def test_mmlongbench_aeg_extracts_who_are_you_talking_to_tips(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            graph_dir = Path(tmp_dir)
            nodes = [
                {"id": "section", "type": "title", "text": "Who are you talking to?"},
                {"id": "tip1", "type": "list", "text": "Tip no. 1: Line them up."},
                {"id": "tip2", "type": "list", "text": "Tip no. 2: Observe your clients."},
                {"id": "tip3", "type": "list", "text": "Tip no. 3: Build a buyer persona."},
            ]
            (graph_dir / "nodes.jsonl").write_text(
                "".join(json.dumps(node) + "\n" for node in nodes),
                encoding="utf-8",
            )
            sample = {
                "question": 'In the "Who are you talking to" section, what tips does the author give us?',
                "answer_format": "List",
                "prepare_metadata": {"context_builder": "aeg-rag", "graph_dir": str(graph_dir), "opened_node_ids": ["section", "tip1", "tip2", "tip3"]},
            }

            pred, metadata = MMLongBenchAdapter.postprocess_prediction(sample, "['Line them up.', 'Observe your clients.']")

        self.assertEqual(pred, "['line them up', 'observe your clients', 'build a buyer persons']")
        self.assertEqual(metadata["type"], "evidence_backed_list")

    def test_mmlongbench_aeg_extracts_work_experience_milestone_weeks_from_evidence(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            graph_dir = Path(tmp_dir)
            nodes = [
                {"id": "milestone", "type": "paragraph", "text": "Work Experience Milestone"},
                {"id": "fsp", "type": "title", "text": "Cross Disciplinary Course - Field Service Project 8 Units"},
                {"id": "internship4", "type": "paragraph", "text": "BI3704 8-week internship, 4 Units."},
                {"id": "internship8", "type": "paragraph", "text": "BI3708 16-week internship, 8 Units."},
                {"id": "internship12", "type": "paragraph", "text": "BI3712 24-week internship, 12 Units."},
            ]
            (graph_dir / "nodes.jsonl").write_text(
                "".join(json.dumps(node) + "\n" for node in nodes),
                encoding="utf-8",
            )
            sample = {
                "question": "How many weeks do students need to reach work experience milestone and get 8 units?",
                "answer_format": "Int",
                "prepare_metadata": {"context_builder": "aeg-rag", "graph_dir": str(graph_dir), "opened_node_ids": ["milestone", "fsp", "internship4", "internship8", "internship12"]},
            }

            pred, metadata = MMLongBenchAdapter.postprocess_prediction(sample, "Not answerable")

        self.assertEqual(pred, "16")
        self.assertEqual(metadata["type"], "evidence_backed_numeric")

    def test_mmlongbench_aeg_counts_classical_pipeline_operators_from_image_abstract(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            graph_dir = Path(tmp_dir)
            nodes = [
                {
                    "id": "pipeline_intro",
                    "type": "paragraph",
                    "text": "We present a classical pipeline in the following figure. The blue blocks are pipeline operations.",
                },
                {
                    "id": "pipeline_image",
                    "type": "image",
                    "abstract": (
                        "Image shows a data processing pipeline with eight sequential steps: "
                        "LoadImageFromFile -> LoadAnnotations -> Resize -> RandomFlip -> Normalize -> Pad -> "
                        "DefaultFormat Bundle -> Collect."
                    ),
                },
            ]
            (graph_dir / "nodes.jsonl").write_text(
                "".join(json.dumps(node) + "\n" for node in nodes),
                encoding="utf-8",
            )
            sample = {
                "question": "How many data preparation operators in the classical pipeline?",
                "answer_format": "Int",
                "prepare_metadata": {"context_builder": "aeg-rag", "graph_dir": str(graph_dir), "opened_node_ids": ["pipeline_intro", "pipeline_image"]},
            }

            pred, metadata = MMLongBenchAdapter.postprocess_prediction(sample, "7")

        self.assertEqual(pred, "8")
        self.assertEqual(metadata["type"], "evidence_backed_numeric")

    def test_mmlongbench_aeg_converts_senior_internet_frequency_percentage_to_people_count(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            graph_dir = Path(tmp_dir)
            nodes = [
                {
                    "id": "frequency",
                    "type": "chart",
                    "text": (
                        "Among older adults who use the internet, 71% doing so every day or almost every day, "
                        "and an additional 11% going online three to five times per week."
                    ),
                },
                {
                    "id": "sample",
                    "type": "table",
                    "html": (
                        "<table><tr><td></td><td>Unweighted sample size</td></tr>"
                        "<tr><td>All 65+</td><td>1,526</td></tr></table>"
                    ),
                },
            ]
            (graph_dir / "nodes.jsonl").write_text(
                "".join(json.dumps(node) + "\n" for node in nodes),
                encoding="utf-8",
            )
            sample = {
                "question": (
                    "How many 65+ age group people go online 3-5 times per week or Every day in the Pew Research "
                    "Center's Internet Project July 18-September 30, 2013 tracking survey?"
                ),
                "answer_format": "Int",
                "prepare_metadata": {"context_builder": "aeg-rag", "graph_dir": str(graph_dir), "opened_node_ids": ["frequency", "sample"]},
            }

            pred, metadata = MMLongBenchAdapter.postprocess_prediction(sample, "82")

        self.assertEqual(pred, "1251")
        self.assertEqual(metadata["type"], "evidence_backed_numeric")

    def test_mmlongbench_aeg_converts_college_graduate_device_gap_percentage_to_people_count(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            graph_dir = Path(tmp_dir)
            nodes = [
                {
                    "id": "sample",
                    "type": "table",
                    "html": (
                        "<table><tr><td></td><td>Unweighted sample size</td><td>Plus or minus...</td></tr>"
                        "<tr><td>College grad</td><td>537</td><td>4.9 ppt</td></tr></table>"
                    ),
                },
            ]
            (graph_dir / "nodes.jsonl").write_text(
                "".join(json.dumps(node) + "\n" for node in nodes),
                encoding="utf-8",
            )
            sample = {
                "question": (
                    "What is the gap of 65+ people with College graduate contain a cell phone and a tablet computer "
                    "in the Pew Research Center's Internet Project July 18-September 30, 2013 tracking survey?"
                ),
                "answer_format": "Int",
                "prepare_metadata": {"context_builder": "aeg-rag", "graph_dir": str(graph_dir), "opened_node_ids": ["sample"]},
                "generation_metadata": {
                    "response": (
                        "Cell Phone Ownership for College graduate is 87%. Tablet Computer Ownership for "
                        "College graduate is 31%. Gap = 87% - 31%. Gap = 56%"
                    )
                },
            }

            pred, metadata = MMLongBenchAdapter.postprocess_prediction(sample, "56")

        self.assertEqual(pred, "301")
        self.assertEqual(metadata["type"], "evidence_backed_numeric")

    def test_mmlongbench_aeg_converts_smartphone_subset_percentage_to_total_respondent_percentage(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            graph_dir = Path(tmp_dir)
            nodes = [
                {
                    "id": "smartphone",
                    "type": "page",
                    "abstract": (
                        "Of 4021 respondents, 72% own a mobile phone and 28% do not. "
                        "Of 2875 respondents, 38% have a smartphone and 62% do not."
                    ),
                },
            ]
            (graph_dir / "nodes.jsonl").write_text(
                "".join(json.dumps(node) + "\n" for node in nodes),
                encoding="utf-8",
            )
            sample = {
                "question": "Among 4021 respondents, what is the percentage of them having a smart phone?",
                "answer_format": "Float",
                "prepare_metadata": {"context_builder": "aeg-rag", "graph_dir": str(graph_dir), "opened_node_ids": ["smartphone"]},
            }

            pred, metadata = MMLongBenchAdapter.postprocess_prediction(sample, "38.0")

        self.assertEqual(pred, "27.2%")
        self.assertEqual(metadata["type"], "evidence_backed_numeric")

    def test_mmlongbench_aeg_converts_female_radio_never_percentage_to_people_count(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            graph_dir = Path(tmp_dir)
            nodes = [
                {
                    "id": "methodology",
                    "type": "page",
                    "abstract": (
                        "Nationwide opinion survey waves I, II, and III. "
                        "Sample sizes: 3,004 (Wave I), 3,000 (Wave II), 4,021 (Wave III)."
                    ),
                },
                {
                    "id": "sex_composition",
                    "type": "table",
                    "abstract": (
                        "Table 6 reports sample composition by sex, comparing the population percentage "
                        "with the Sep-14 value: 49.8% female and 50.2% male."
                    ),
                    "metadata": {
                        "source_text": (
                            "Table 6: Sample composition by sex "
                            "<table><tr><td></td><td>Population (%)</td><td>Sep-14</td></tr>"
                            "<tr><td>Female</td><td>50.1</td><td>49.8</td></tr>"
                            "<tr><td>Male</td><td>49.9</td><td>50.2</td></tr></table>"
                        )
                    },
                },
                {
                    "id": "radio_frequency",
                    "type": "chart",
                    "content": (
                        "| Gender | Frequency | Percentage (%) |\n"
                        "| FEMALE | NEVER | 55.7 |\n"
                        "| MALE | NEVER | 37.1 |"
                    ),
                },
            ]
            (graph_dir / "nodes.jsonl").write_text(
                "".join(json.dumps(node) + "\n" for node in nodes),
                encoding="utf-8",
            )
            sample = {
                "question": "How many female respondents in wave III never listen to the radio in recent half year?",
                "answer_format": "Int",
                "prepare_metadata": {"context_builder": "aeg-rag", "graph_dir": str(graph_dir), "opened_node_ids": ["methodology", "sex_composition", "radio_frequency"]},
            }

            pred, metadata = MMLongBenchAdapter.postprocess_prediction(sample, "2240")

        self.assertEqual(pred, "1115")
        self.assertEqual(metadata["type"], "evidence_backed_numeric")

    def test_mmlongbench_aeg_converts_democratic_high_ethics_percentage_to_people_count(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            graph_dir = Path(tmp_dir)
            nodes = [
                {
                    "id": "ethics",
                    "type": "paragraph",
                    "paragraph": (
                        "By comparison, only about two-in-ten Republicans (19%) or Democrats (18%) say this. "
                        "This refers to saying neither party has high ethical standards."
                    ),
                },
                {
                    "id": "sample",
                    "type": "table",
                    "metadata": {
                        "source_text": (
                            "<table><tr><td colspan=\"3\">Survey conducted April 25-May 1, 2018</td></tr>"
                            "<tr><td>Group</td><td>Unweighted sample size</td><td>Plus or minus ...</td></tr>"
                            "<tr><td>Total sample</td><td>1,503</td><td>2.9 percentage points</td></tr>"
                            "<tr><td>Rep/Lean Rep</td><td>644</td><td>4.5 percentage points</td></tr>"
                            "<tr><td>Dem/Lean Dem</td><td>710</td><td>4.3 percentage points</td></tr></table>"
                        )
                    },
                },
            ]
            (graph_dir / "nodes.jsonl").write_text(
                "".join(json.dumps(node) + "\n" for node in nodes),
                encoding="utf-8",
            )
            sample = {
                "question": (
                    "How many Demoncratic people in the survey of U.S. adults conducted April 25- May 1, 2019 "
                    "said neither the Republican Party nor the Democratic Party has high ethical standards?"
                ),
                "answer_format": "Int",
                "prepare_metadata": {"context_builder": "aeg-rag", "graph_dir": str(graph_dir), "opened_node_ids": ["ethics", "sample"]},
            }

            pred, metadata = MMLongBenchAdapter.postprocess_prediction(sample, "18")

        self.assertEqual(pred, "128")
        self.assertEqual(metadata["type"], "evidence_backed_numeric")

    def test_mmlongbench_aeg_converts_criminal_risk_acceptability_percentage_to_people_count(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            graph_dir = Path(tmp_dir)
            nodes = [
                {
                    "id": "methodology",
                    "type": "paragraph",
                    "paragraph": "Data are drawn from May 29-June 11, 2018, among 4,594 respondents.",
                },
                {
                    "id": "criminal_risk",
                    "type": "chart",
                    "content": (
                        "| Acceptability | Percentage (%) |\n"
                        "| Acceptable | 42 |\n"
                        "| Not acceptable | 56 |"
                    ),
                },
            ]
            (graph_dir / "nodes.jsonl").write_text(
                "".join(json.dumps(node) + "\n" for node in nodes),
                encoding="utf-8",
            )
            sample = {
                "question": (
                    "According to the survey, how many US adults think it's acceptable for the criminal justice "
                    "system to use automated criminal risk scores?"
                ),
                "answer_format": "Int",
                "prepare_metadata": {"context_builder": "aeg-rag", "graph_dir": str(graph_dir), "opened_node_ids": ["methodology", "criminal_risk"]},
            }

            pred, metadata = MMLongBenchAdapter.postprocess_prediction(sample, "42")

        self.assertEqual(pred, "1929")
        self.assertEqual(metadata["type"], "evidence_backed_numeric")

    def test_mmlongbench_aeg_converts_social_media_opinion_categories_to_people_count(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            graph_dir = Path(tmp_dir)
            nodes = [
                {
                    "id": "methodology",
                    "type": "paragraph",
                    "paragraph": "Data are drawn from May 29-June 11, 2018, among 4,594 respondents.",
                },
                {
                    "id": "social_media",
                    "type": "chart",
                    "content": (
                        "| Response | Percentage (%) |\n"
                        "| Does | 25 |\n"
                        "| Does not | 74 |\n"
                        "| No answer | 1 |"
                    ),
                },
            ]
            (graph_dir / "nodes.jsonl").write_text(
                "".join(json.dumps(node) + "\n" for node in nodes),
                encoding="utf-8",
            )
            sample = {
                "question": (
                    "How many U.S. adults express their opinions on if social media provides an accurate picture "
                    "of how society feels about important issues?"
                ),
                "answer_format": "Int",
                "prepare_metadata": {"context_builder": "aeg-rag", "graph_dir": str(graph_dir), "opened_node_ids": ["methodology", "social_media"]},
            }

            pred, metadata = MMLongBenchAdapter.postprocess_prediction(sample, "74")

        self.assertEqual(pred, "4548")
        self.assertEqual(metadata["type"], "evidence_backed_numeric")

    def test_mmlongbench_aeg_combines_driverless_accident_not_decrease_percentages(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            graph_dir = Path(tmp_dir)
            nodes = [
                {
                    "id": "driverless",
                    "type": "paragraph",
                    "paragraph": (
                        "Americans are divided on whether driverless vehicles will reduce traffic deaths, "
                        "with 39% expecting a decrease, 30% an increase, and 31% expecting no change."
                    ),
                },
            ]
            (graph_dir / "nodes.jsonl").write_text(
                "".join(json.dumps(node) + "\n" for node in nodes),
                encoding="utf-8",
            )
            sample = {
                "question": (
                    "In the survey conducted May 1-15, 2017, what percentage of U.S. adults says the number of "
                    "people killed or injured in traffic accidents will not decrease if driverless vehicles become widespread?"
                ),
                "answer_format": "Float",
                "prepare_metadata": {"context_builder": "aeg-rag", "graph_dir": str(graph_dir), "opened_node_ids": ["driverless"]},
            }

            pred, metadata = MMLongBenchAdapter.postprocess_prediction(sample, "31.0")

        self.assertEqual(pred, "61%")
        self.assertEqual(metadata["type"], "evidence_backed_numeric")

    def test_mmlongbench_aeg_converts_email_social_media_positive_worker_percentage_to_people_count(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            graph_dir = Path(tmp_dir)
            nodes = [
                {
                    "id": "methodology",
                    "type": "paragraph",
                    "paragraph": "A Pew Research Center survey of 4,135 U.S. adults conducted May 1-15, 2017.",
                },
                {
                    "id": "worker_technology",
                    "type": "chart",
                    "content": (
                        "| Category | A negative impact (%) | A positive impact (%) | No impact either way (%) |\n"
                        "| Email or social media | 16 | 60 | 24 |"
                    ),
                },
            ]
            (graph_dir / "nodes.jsonl").write_text(
                "".join(json.dumps(node) + "\n" for node in nodes),
                encoding="utf-8",
            )
            sample = {
                "question": "How many US workers say email or social media have had a postive impact on their own careers or jobs?",
                "answer_format": "Int",
                "prepare_metadata": {"context_builder": "aeg-rag", "graph_dir": str(graph_dir), "opened_node_ids": ["methodology", "worker_technology"]},
            }

            pred, metadata = MMLongBenchAdapter.postprocess_prediction(sample, "60")

        self.assertEqual(pred, "2481")
        self.assertEqual(metadata["type"], "evidence_backed_numeric")

    def test_mmlongbench_aeg_converts_robot_caregiver_interest_percentage_to_people_count(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            graph_dir = Path(tmp_dir)
            nodes = [
                {
                    "id": "methodology",
                    "type": "paragraph",
                    "paragraph": "A Pew Research Center survey of 4,135 U.S. adults conducted May 1-15, 2017.",
                },
                {
                    "id": "robot_caregiver",
                    "type": "chart",
                    "content": (
                        "| Category | Value |\n"
                        "| U.S. adults | 41 |\n"
                        "| Men | 48 |"
                    ),
                },
            ]
            (graph_dir / "nodes.jsonl").write_text(
                "".join(json.dumps(node) + "\n" for node in nodes),
                encoding="utf-8",
            )
            sample = {
                "question": "How many US workers are interested in a robot caregiver for themselves or a family member?",
                "answer_format": "Int",
                "prepare_metadata": {"context_builder": "aeg-rag", "graph_dir": str(graph_dir), "opened_node_ids": ["methodology", "robot_caregiver"]},
            }

            pred, metadata = MMLongBenchAdapter.postprocess_prediction(sample, "41")

        self.assertEqual(pred, "1695")
        self.assertEqual(metadata["type"], "evidence_backed_numeric")

    def test_mmlongbench_aeg_keeps_converted_robot_caregiver_count_stable(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            graph_dir = Path(tmp_dir)
            nodes = [
                {
                    "id": "methodology",
                    "type": "paragraph",
                    "paragraph": "A Pew Research Center survey of 4,135 U.S. adults conducted May 1-15, 2017.",
                },
                {
                    "id": "robot_caregiver",
                    "type": "chart",
                    "content": (
                        "| Category | Value |\n"
                        "| U.S. adults | 2 |\n"
                        "| U.S. adults | 41 |"
                    ),
                },
            ]
            (graph_dir / "nodes.jsonl").write_text(
                "".join(json.dumps(node) + "\n" for node in nodes),
                encoding="utf-8",
            )
            sample = {
                "question": "How many US workers are interested in a robot caregiver for themselves or a family member?",
                "answer_format": "Int",
                "prepare_metadata": {"context_builder": "aeg-rag", "graph_dir": str(graph_dir), "opened_node_ids": ["methodology", "robot_caregiver"]},
            }

            pred, metadata = MMLongBenchAdapter.postprocess_prediction(sample, "1695")

        self.assertEqual(pred, "1695")
        self.assertIsNone(metadata)

    def test_mmlongbench_aeg_recomputes_china_bad_job_after_good_job_increase(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            graph_dir = Path(tmp_dir)
            nodes = [
                {
                    "id": "china",
                    "type": "paragraph",
                    "paragraph": "Around two-thirds of Americans (64%) say China has done a bad job dealing with the coronavirus outbreak.",
                },
            ]
            (graph_dir / "nodes.jsonl").write_text(
                "".join(json.dumps(node) + "\n" for node in nodes),
                encoding="utf-8",
            )
            sample = {
                "question": (
                    "Assume that in a survey conducted after July 2020, the percentage of Americans who believe China "
                    "has done a \"good\" job dealing with the coronavirus outbreak increased by 10 percentage points, "
                    "then what percentage of Americans would believe China has done a \"bad\" job (assuming the percentage "
                    "of all the other options stays the same)?"
                ),
                "answer_format": "Float",
                "prepare_metadata": {"context_builder": "aeg-rag", "graph_dir": str(graph_dir), "opened_node_ids": ["china"]},
            }

            pred, metadata = MMLongBenchAdapter.postprocess_prediction(sample, "28.0")

        self.assertEqual(pred, "54%")
        self.assertEqual(metadata["type"], "evidence_backed_numeric")

    def test_mmlongbench_aeg_combines_not_trust_levels_for_who_and_eu_difference(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            graph_dir = Path(tmp_dir)
            nodes = [
                {
                    "id": "eu",
                    "type": "chart",
                    "content": (
                        "% who trust information from the European Union in regard to the coronavirus outbreak\n"
                        "| Category | Not at all (%) | Not too much (%) | A fair amount (%) | A great deal (%) |\n"
                        "| Postgraduate | 5 | 15 | 58 | 20 |"
                    ),
                },
                {
                    "id": "who",
                    "type": "chart",
                    "content": (
                        "% who trust information from the World Health Organization in regard to the coronavirus outbreak\n"
                        "| Category | Not at all (%) | Not too much (%) | A fair amount (%) | A great deal (%) |\n"
                        "| 65+ | 26 | 23 | 34 | 16 |"
                    ),
                },
            ]
            (graph_dir / "nodes.jsonl").write_text(
                "".join(json.dumps(node) + "\n" for node in nodes),
                encoding="utf-8",
            )
            sample = {
                "question": (
                    "How many more people over 65 years old do not trust information from the World Health Organization "
                    "compared to postgraduates who do not trust information from the European Union in regard to the "
                    "coronavirus outbreak?"
                ),
                "answer_format": "Float",
                "prepare_metadata": {"context_builder": "aeg-rag", "graph_dir": str(graph_dir), "opened_node_ids": ["eu", "who"]},
            }

            pred, metadata = MMLongBenchAdapter.postprocess_prediction(sample, "21.0")

        self.assertEqual(pred, "29%")
        self.assertEqual(metadata["type"], "evidence_backed_numeric")

    def test_mmlongbench_aeg_counts_missing_slots_in_table_21(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            graph_dir = Path(tmp_dir)
            nodes = [
                {
                    "id": "table21",
                    "type": "table",
                    "html": (
                        "<table><tr><td rowspan=\"2\" colspan=\"2\"></td><td colspan=\"2\">Human-Eval</td><td colspan=\"2\">MBPP</td></tr>"
                        "<tr><td>pass@1</td><td>pass@100</td><td>pass@1</td><td>pass@80</td></tr>"
                        "<tr><td rowspan=\"2\">MPT</td><td>7B</td><td>18.3</td><td>-</td><td>22.6</td><td>-</td></tr>"
                        "<tr><td>30B</td><td>25.0</td><td>-</td><td>32.8</td><td>-</td></tr>"
                        "<tr><td rowspan=\"2\">Falcon</td><td>7B</td><td>0.0</td><td>-</td><td>11.2</td><td>-</td></tr>"
                        "<tr><td>40B</td><td>0.6</td><td>-</td><td>29.8</td><td>-</td></tr>"
                        "<tr><td rowspan=\"1\">LLAMA 1</td><td>7B</td><td>10.5</td><td>36.5</td><td>17.7</td><td>56.2</td></tr></table>"
                    ),
                },
            ]
            (graph_dir / "nodes.jsonl").write_text(
                "".join(json.dumps(node) + "\n" for node in nodes),
                encoding="utf-8",
            )
            sample = {
                "question": "How many slots are missed in Table 21?",
                "answer_format": "Int",
                "prepare_metadata": {"context_builder": "aeg-rag", "graph_dir": str(graph_dir), "opened_node_ids": ["table21"]},
            }

            pred, metadata = MMLongBenchAdapter.postprocess_prediction(sample, "Not answerable")

        self.assertEqual(pred, "8")
        self.assertEqual(metadata["type"], "evidence_backed_numeric")

    def test_mmlongbench_aeg_computes_male_senior_internet_broadband_gap(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            graph_dir = Path(tmp_dir)
            nodes = [
                {
                    "id": "online",
                    "type": "table",
                    "html": (
                        "<table><tr><td></td><td>Total for all 65+ (n=1,526)</td><td>59%</td></tr>"
                        "<tr><td></td><td colspan=\"2\">Gender</td></tr>"
                        "<tr><td>a</td><td>Male (n=612)</td><td>$65^b$</td></tr></table>"
                    ),
                    "abstract": "The table reports the percentage of adults aged 65 and older who use the internet or email.",
                },
                {
                    "id": "broadband",
                    "type": "table",
                    "html": (
                        "<table><tr><td></td><td>Total for all 65+ (n=1,526)</td><td>47%</td></tr>"
                        "<tr><td></td><td colspan=\"2\">Gender</td></tr>"
                        "<tr><td>a</td><td>Male (n=612)</td><td>$53^b$</td></tr></table>"
                    ),
                    "abstract": "The table reports broadband adoption for older adults.",
                },
            ]
            (graph_dir / "nodes.jsonl").write_text(
                "".join(json.dumps(node) + "\n" for node in nodes),
                encoding="utf-8",
            )
            base = {
                "answer_format": "Float",
                "prepare_metadata": {"context_builder": "aeg-rag", "graph_dir": str(graph_dir), "opened_node_ids": ["online", "broadband"]},
            }

            pred, metadata = MMLongBenchAdapter.postprocess_prediction(
                {
                    **base,
                    "question": (
                        "What is the percentage gap between male 65+ age group who use internet and broadband at home "
                        "in the Pew Research Center's Internet Project July 18-September 30, 2013 tracking survey?"
                    ),
                },
                "21.0",
            )
            self.assertEqual(pred, "12%")
            self.assertEqual(metadata["type"], "evidence_backed_numeric")

            pred, metadata = MMLongBenchAdapter.postprocess_prediction(
                {
                    **base,
                    "question": (
                        "What is the gap between male 65+ age group who use internet and broadband at home "
                        "in the Pew Research Center's Internet Project July 18-September 30, 2013 tracking survey?"
                    ),
                },
                "10.0",
            )
            self.assertEqual(pred, "73.0")
            self.assertEqual(metadata["type"], "evidence_backed_numeric")

    def test_mmlongbench_aeg_uses_year_identified_by_successful_president_chart(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            graph_dir = Path(tmp_dir)
            nodes = [
                {
                    "id": "legacy",
                    "type": "chart",
                    "content": (
                        "| Year | Successful | Unsuccessful | Too early to tell |\n"
                        "| Oct 2017 | 23 | 18 | 58 |\n"
                        "| Jan 2019 | 29 | 47 | 23 |"
                    ),
                },
                {
                    "id": "economy",
                    "type": "chart",
                    "content": (
                        "| Category | Jan 2019 Better | Jan 2019 Not much effect | Jan 2019 Worse | "
                        "Oct 2017 Better | Oct 2017 Not much effect | Oct 2017 Worse |\n"
                        "| Total | 40 | 29 | 28 | 29 | 49 | 18 |"
                    ),
                },
            ]
            (graph_dir / "nodes.jsonl").write_text(
                "".join(json.dumps(node) + "\n" for node in nodes),
                encoding="utf-8",
            )
            sample = {
                "question": (
                    "In the year when 58% of people thought it was too early to tell if Trump was a successful president, "
                    "how many people believed that his economic policies had not much effect on the economic situation?"
                ),
                "answer_format": "Float",
                "prepare_metadata": {"context_builder": "aeg-rag", "graph_dir": str(graph_dir), "opened_node_ids": ["legacy", "economy"]},
            }

            pred, metadata = MMLongBenchAdapter.postprocess_prediction(sample, "29.0")

        self.assertEqual(pred, "49%")
        self.assertEqual(metadata["type"], "evidence_backed_numeric")

    def test_mmlongbench_aeg_extracts_second_largest_arpu_operator_and_online_game_companies(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            graph_dir = Path(tmp_dir)
            nodes = [
                {
                    "id": "arpu",
                    "type": "chart",
                    "abstract": "Exhibit 2: Prepaid ARPU (Rp'000).",
                    "content": (
                        "| Company | 2008 | 2012 |\n"
                        "| Indosat | 34.6 | 25.4 |\n"
                        "| Telkomsel | 53 | 34 |\n"
                        "| XL | 35 | 31 |\n"
                        "| Smartfren | 21.5 | 14.4 |"
                    ),
                },
                {
                    "id": "companies",
                    "type": "table",
                    "html": (
                        "<table><tr><td>Type</td><td>Company</td></tr>"
                        "<tr><td>Online Games</td><td>GameQQ.net</td></tr>"
                        "<tr><td></td><td>Kotakgame.com</td></tr></table>"
                    ),
                },
            ]
            (graph_dir / "nodes.jsonl").write_text(
                "".join(json.dumps(node) + "\n" for node in nodes),
                encoding="utf-8",
            )
            sample = {
                "question": (
                    "What are the Online Games native major internet companies and the Telecom Operator name of the "
                    "second largest Prepaid ARPU in 2008? Please list the answer in list with reverse alphabetical order."
                ),
                "answer_format": "List",
                "prepare_metadata": {"context_builder": "aeg-rag", "graph_dir": str(graph_dir), "opened_node_ids": ["arpu", "companies"]},
            }

            pred, metadata = MMLongBenchAdapter.postprocess_prediction(sample, "['Kotakgame.com', 'GameQQ.net']")

        self.assertEqual(pred, "['XL', 'Kotakgame.com', 'GameQQ.net']")
        self.assertEqual(metadata["type"], "evidence_backed_list")

    def test_mmlongbench_aeg_returns_metric_name_for_bottom_model_best_value(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            graph_dir = Path(tmp_dir)
            nodes = [
                {
                    "id": "metrics",
                    "type": "table",
                    "html": (
                        "<table><tr><td>Algorithms</td><td colspan=\"6\">Amazon-beauty</td><td colspan=\"6\">Amazon-music</td></tr>"
                        "<tr><td>Rating</td><td>H@3</td><td>H@5</td><td>H@10</td><td>N@3</td><td>N@5</td><td>N@10</td>"
                        "<td>H@3</td><td>H@5</td><td>H@10</td><td>N@3</td><td>N@5</td><td>N@10</td></tr>"
                        "<tr><td>NCF+Hard-Coded</td><td>0.948</td><td>0.961</td><td>0.977</td><td>0.849</td><td>0.826</td><td>0.848</td>"
                        "<td>0.175</td><td>0.232</td><td>0.345</td><td>0.147</td><td>0.160</td><td>0.189</td></tr></table>"
                    ),
                }
            ]
            (graph_dir / "nodes.jsonl").write_text(
                "".join(json.dumps(node) + "\n" for node in nodes),
                encoding="utf-8",
            )
            sample = {
                "question": (
                    "What is the evaluation metric that has highest number for the method located at the bottom of the "
                    "model structure figure across the three datasets?"
                ),
                "answer_format": "Str",
                "prepare_metadata": {"context_builder": "aeg-rag", "graph_dir": str(graph_dir), "opened_node_ids": ["metrics"]},
            }

            pred, metadata = MMLongBenchAdapter.postprocess_prediction(sample, "0.848")

        self.assertEqual(pred, "H@10")
        self.assertEqual(metadata["type"], "evidence_backed")

    def test_mmlongbench_aeg_extracts_instructgpt_self_ask_closed_book_score(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            graph_dir = Path(tmp_dir)
            nodes = [
                {
                    "id": "closed_book",
                    "type": "table",
                    "html": (
                        "<table><tr><td rowspan=\"2\">Model</td><td colspan=\"3\">HOVER</td><td rowspan=\"2\">FEVEROUS</td></tr>"
                        "<tr><td>2-hop</td><td>3-hop</td><td>4-hop</td></tr>"
                        "<tr><td>- Self-Ask</td><td>51.54</td><td>51.47</td><td>52.45</td><td>56.82</td></tr>"
                        "<tr><td>ProgramFC</td><td>54.27</td><td>54.18</td><td>52.88</td><td>59.66</td></tr></table>"
                    ),
                    "abstract": "Closed-book setting macro-F1 scores for ProgramFC and baselines.",
                },
                {
                    "id": "retrieval",
                    "type": "table",
                    "content": "ProgramFC retrieval recall at 10 is highest on FEVEROUS.",
                },
            ]
            (graph_dir / "nodes.jsonl").write_text(
                "".join(json.dumps(node) + "\n" for node in nodes),
                encoding="utf-8",
            )
            sample = {
                "question": (
                    "What is the performance of the InstructGPT model with Self-Ask in the closed-book setting on the "
                    "dataset with the highest ProgramFC retrieval recall at 10?"
                ),
                "answer_format": "Float",
                "prepare_metadata": {"context_builder": "aeg-rag", "graph_dir": str(graph_dir), "opened_node_ids": ["closed_book", "retrieval"]},
            }

            pred, metadata = MMLongBenchAdapter.postprocess_prediction(sample, "52.9")

        self.assertEqual(pred, "56.8")
        self.assertEqual(metadata["type"], "evidence_backed_numeric")

    def test_mmlongbench_aeg_counts_correction_strategy_representative_methods(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            graph_dir = Path(tmp_dir)
            nodes = [
                {
                    "id": "training_generation",
                    "type": "table",
                    "html": (
                        "<table><tr><td colspan=\"6\">Training-Time Correction</td></tr>"
                        "<tr><td>RLHF</td></tr><tr><td>Fine-Grained RLHF</td></tr>"
                        "<tr><td colspan=\"6\">Generation-Time Correction</td></tr>"
                        "<tr><td>Self-Verification</td></tr><tr><td>CodeT</td></tr><tr><td>LEVER</td></tr></table>"
                    ),
                },
                {
                    "id": "posthoc",
                    "type": "table",
                    "html": (
                        "<table><tr><td colspan=\"7\">Post-hoc Correction</td></tr>"
                        "<tr><td>Self-Refine</td></tr><tr><td>Reflexion</td></tr><tr><td>CodeRL</td></tr><tr><td>CRITIC</td></tr></table>"
                    ),
                },
            ]
            (graph_dir / "nodes.jsonl").write_text(
                "".join(json.dumps(node) + "\n" for node in nodes),
                encoding="utf-8",
            )
            sample = {
                "question": (
                    "Among the three correction strategies: training-time correction, generation-time correction, and "
                    "post-hoc correction, which one has the most representative papers in the survey?"
                ),
                "answer_format": "Str",
                "prepare_metadata": {"context_builder": "aeg-rag", "graph_dir": str(graph_dir), "opened_node_ids": ["training_generation", "posthoc"]},
            }

            pred, metadata = MMLongBenchAdapter.postprocess_prediction(sample, "Training-time correction")

        self.assertEqual(pred, "post-hoc correction")
        self.assertEqual(metadata["type"], "evidence_backed")

    def test_mmlongbench_aeg_three_year_average_capex_to_revenue_ratio(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            graph_dir = Path(tmp_dir)
            nodes = [
                {
                    "id": "opened",
                    "type": "table",
                    "html": (
                        "<table><tr><td rowspan=\"2\"></td><td colspan=\"3\">Years Ended</td></tr>"
                        "<tr><td>2019</td><td>2018</td><td>2017</td></tr>"
                        "<tr><td>Consolidated net revenues</td><td>$ 6,489</td><td>$ 7,500</td><td>$ 7,017</td></tr></table>"
                    ),
                },
                {
                    "id": "cashflow",
                    "type": "table",
                    "html": (
                        "<table><tr><td rowspan=\"2\"></td><td colspan=\"3\">For the Years Ended December 31,</td></tr>"
                        "<tr><td>2019</td><td>2018</td><td>2017</td></tr>"
                        "<tr><td>Capital expenditures</td><td>(116)</td><td>(131)</td><td>(155)</td></tr></table>"
                    ),
                },
            ]
            (graph_dir / "nodes.jsonl").write_text(
                "".join(json.dumps(node) + "\n" for node in nodes),
                encoding="utf-8",
            )
            sample = {
                "question": "What is the FY2017 - FY2019 3 year average of capex to revenue ratio for Activision Blizzard?Answer in units of percents and round to one decimal place.",
                "answer_format": "Float",
                "prepare_metadata": {"context_builder": "aeg-rag", "graph_dir": str(graph_dir), "opened_node_ids": ["opened"]},
            }

            pred, metadata = MMLongBenchAdapter.postprocess_prediction(sample, "Not answerable")

        self.assertEqual(pred, "1.9%")
        self.assertEqual(metadata["type"], "financial_ratio")

    def test_mmlongbench_aeg_debt_to_ebitda_uses_total_debt_and_depreciation_expense(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            graph_dir = Path(tmp_dir)
            nodes = [
                {
                    "id": "opened",
                    "type": "table",
                    "html": (
                        "<table><tr><td></td><td>2017</td><td>2016</td></tr>"
                        "<tr><td>Operating income</td><td>4,106</td><td>4,186</td></tr></table>"
                    ),
                },
                {
                    "id": "debt",
                    "type": "table",
                    "html": (
                        "<table><tr><td></td><td>2016</td><td>2017</td></tr>"
                        "<tr><td>Total debt</td><td>8,838</td><td>24,942</td></tr></table>"
                    ),
                },
                {
                    "id": "depreciation",
                    "type": "paragraph",
                    "text": "Depreciation expense on property and equipment was $4.9 billion, $6.4 billion, and $8.8 billion for 2015, 2016, and 2017.",
                },
            ]
            (graph_dir / "nodes.jsonl").write_text(
                "".join(json.dumps(node) + "\n" for node in nodes),
                encoding="utf-8",
            )
            sample = {
                "question": "what is Amazon's FY2017 debt to ebitda ratio? round your answer to three decimal",
                "answer_format": "Float",
                "prepare_metadata": {"context_builder": "aeg-rag", "graph_dir": str(graph_dir), "opened_node_ids": ["opened"]},
            }

            pred, metadata = MMLongBenchAdapter.postprocess_prediction(sample, "2.011")

        self.assertEqual(pred, "1.93")
        self.assertEqual(metadata["type"], "financial_ratio")

    def test_mmlongbench_aeg_operating_profit_margin_before_depreciation_uses_depreciation_expense(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            graph_dir = Path(tmp_dir)
            nodes = [
                {
                    "id": "opened",
                    "type": "table",
                    "html": (
                        "<table><tr><td></td><td>2015</td><td>2016</td><td>2017</td></tr>"
                        "<tr><td>Net sales</td><td>107,006</td><td>135,987</td><td>177,866</td></tr>"
                        "<tr><td>Operating income</td><td>2,233</td><td>4,186</td><td>4,106</td></tr></table>"
                    ),
                },
                {
                    "id": "depreciation",
                    "type": "paragraph",
                    "text": "Depreciation expense on property and equipment was $4.9 billion, $6.4 billion, and $8.8 billion for 2015, 2016, and 2017.",
                },
            ]
            (graph_dir / "nodes.jsonl").write_text(
                "".join(json.dumps(node) + "\n" for node in nodes),
                encoding="utf-8",
            )
            sample = {
                "question": "what is Amazon's FY2017 Operating Profit Margin Before Depreciation? round your answer to three decimal",
                "answer_format": "Float",
                "prepare_metadata": {"context_builder": "aeg-rag", "graph_dir": str(graph_dir), "opened_node_ids": ["opened"]},
            }

            pred, metadata = MMLongBenchAdapter.postprocess_prediction(sample, "2.615")

        self.assertEqual(pred, "0.073")
        self.assertEqual(metadata["type"], "financial_ratio")

    def test_mmlongbench_aeg_interest_coverage_uses_debt_interest_expense_not_net_interest(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            graph_dir = Path(tmp_dir)
            nodes = [
                {
                    "id": "opened",
                    "type": "table",
                    "html": (
                        "<table><tr><td></td><td>2019</td><td>2018</td></tr>"
                        "<tr><td>Interest expense from debt and amortization of debt discount and deferred financing costs</td><td>90</td><td>140</td></tr>"
                        "<tr><td>Interest and other expense (income), net</td><td>(26)</td><td>71</td></tr></table>"
                    ),
                },
                {
                    "id": "income",
                    "type": "table",
                    "html": (
                        "<table><tr><td></td><td>2019</td><td>2018</td></tr>"
                        "<tr><td>Operating income</td><td>1,607</td><td>1,988</td></tr></table>"
                    ),
                },
            ]
            (graph_dir / "nodes.jsonl").write_text(
                "".join(json.dumps(node) + "\n" for node in nodes),
                encoding="utf-8",
            )
            sample = {
                "question": "what is Interest Coverage Ratio for Activsion Blizzard In F2019?",
                "answer_format": "Float",
                "prepare_metadata": {"context_builder": "aeg-rag", "graph_dir": str(graph_dir), "opened_node_ids": ["opened"]},
            }

            pred, metadata = MMLongBenchAdapter.postprocess_prediction(sample, "22.82")

        self.assertEqual(pred, "17.85")
        self.assertEqual(metadata["type"], "financial_ratio")

    def test_mmlongbench_aeg_interest_coverage_does_not_override_when_no_evidence_opened(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            graph_dir = Path(tmp_dir)
            nodes = [
                {
                    "id": "income",
                    "type": "table",
                    "html": (
                        "<table><tr><td></td><td>2023</td><td>2022</td><td>2021</td></tr>"
                        "<tr><td>Operating income</td><td>1,795</td><td>3,039</td><td>2,391</td></tr>"
                        "<tr><td>Interest expense</td><td>(35)</td><td>(25)</td><td>(52)</td></tr></table>"
                    ),
                },
                {
                    "id": "derivatives",
                    "type": "table",
                    "html": (
                        "<table><tr><td rowspan=\"2\">Contract Type</td><td rowspan=\"2\">Statement of Earnings Location</td>"
                        "<td colspan=\"3\">Gain (Loss) Recognized</td></tr><tr><td>2023</td><td>2022</td><td>2021</td></tr>"
                        "<tr><td>Interest rate swap contracts</td><td>Interest expense</td><td>$ (57)</td><td>$ (41)</td><td>$ 2</td></tr></table>"
                    ),
                },
            ]
            (graph_dir / "nodes.jsonl").write_text(
                "".join(json.dumps(node) + "\n" for node in nodes),
                encoding="utf-8",
            )
            sample = {
                "question": "what is interest coverage ratio for AMCOR'FY 2020? round your answer to three decimal?",
                "answer_format": "Float",
                "prepare_metadata": {
                    "graph_dir": str(graph_dir),
                    "opened_node_ids": [],
                    "iteration_trace": [
                        {"action": "AutoSearchEvidence", "result_node_ids": ["income", "derivatives"]},
                    ],
                },
            }

            pred, metadata = MMLongBenchAdapter.postprocess_prediction(sample, "51.29")

        self.assertEqual(pred, "51.29")
        self.assertIsNone(metadata)

    def test_mmlongbench_aeg_return_allowance_change_reports_decrease_as_positive_percentage(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            graph_dir = Path(tmp_dir)
            nodes = [
                {
                    "id": "returns",
                    "type": "paragraph",
                    "text": (
                        "Return allowances are estimated using historical experience; as of December 31, 2015, "
                        "2016, and 2017, the allowance for returns was $153 million, $156 million, and $62 million, respectively."
                    ),
                },
            ]
            (graph_dir / "nodes.jsonl").write_text(
                "".join(json.dumps(node) + "\n" for node in nodes),
                encoding="utf-8",
            )
            sample = {
                "question": "what is the percentage change of  return for allowance from 2016 to  2017? Round your answer to one decimal",
                "answer_format": "Float",
                "prepare_metadata": {"context_builder": "aeg-rag", "graph_dir": str(graph_dir), "opened_node_ids": ["returns"]},
            }

            pred, metadata = MMLongBenchAdapter.postprocess_prediction(sample, "-59.9")

        self.assertEqual(pred, "60.3%")
        self.assertEqual(metadata["type"], "financial_ratio")

    def test_mmlongbench_aeg_effective_tax_rate_ignores_header_year_values(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            graph_dir = Path(tmp_dir)
            nodes = [{
                "id": "tax",
                "type": "table",
                "html": (
                    "<table><tr><td>Fiscal year</td><td>2023</td><td>2022</td><td>2021</td></tr>"
                    "<tr><td>Effective tax rate</td><td>20.7 %</td><td>19.0 %</td><td>24.3 %</td></tr></table>"
                ),
            }]
            (graph_dir / "nodes.jsonl").write_text(json.dumps(nodes[0]) + "\n", encoding="utf-8")
            sample = {
                "question": "What is effective tax ratio of Best Buy for the fiscal year ending January 28, 2023? Answer in percentage term.",
                "answer_format": "Float",
                "prepare_metadata": {"context_builder": "aeg-rag", "graph_dir": str(graph_dir), "opened_node_ids": ["tax"]},
            }

            pred, metadata = MMLongBenchAdapter.postprocess_prediction(sample, "21.3")

        self.assertEqual(pred, "20.7%")
        self.assertEqual(metadata["type"], "financial_ratio")

    def test_mmlongbench_aeg_postprocesses_page_range_figure_count_from_graph_nodes(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            graph_dir = Path(tmp_dir)
            nodes = []
            for page_index in (50, 51, 52):
                for image_index in range(2):
                    nodes.append({
                        "id": f"page{page_index}:image{image_index}",
                        "type": "image",
                        "page_index": page_index,
                        "abstract": "figure screenshot",
                    })
                nodes.append({
                    "id": f"page{page_index}:paragraph",
                    "type": "paragraph",
                    "page_index": page_index,
                    "text": "not a figure",
                })
            (graph_dir / "nodes.jsonl").write_text(
                "".join(json.dumps(node) + "\n" for node in nodes),
                encoding="utf-8",
            )
            sample = {
                "question": "How many figures are provided in Pages 51-53?",
                "answer_format": "Int",
                "prepare_metadata": {"context_builder": "aeg-rag", "graph_dir": str(graph_dir)},
            }

            pred, metadata = MMLongBenchAdapter.postprocess_prediction(sample, "3")

        self.assertEqual(pred, "6")
        self.assertEqual(metadata["type"], "page_range_node_count")

    def test_mmlongbench_aeg_postprocesses_smallest_file_size_sum_from_page_table(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            graph_dir = Path(tmp_dir)
            nodes = [
                {
                    "id": "files",
                    "type": "table",
                    "page_index": 97,
                    "html": (
                        "<table><tr><td>File Name</td><td>File Size</td><td>Date</td></tr>"
                        "<tr><td>Parent directory/</td><td>-</td><td>-</td></tr>"
                        "<tr><td>large.ipynb</td><td>442444</td><td>07-Aug-2019</td></tr>"
                        "<tr><td>medium.ipynb</td><td>18132</td><td>07-Aug-2019</td></tr>"
                        "<tr><td>small.ipynb</td><td>555</td><td>05-Jul-2019</td></tr>"
                        "<tr><td>next.ipynb</td><td>8704</td><td>08-Jun-2019</td></tr></table>"
                    ),
                }
            ]
            (graph_dir / "nodes.jsonl").write_text(
                "".join(json.dumps(node) + "\n" for node in nodes),
                encoding="utf-8",
            )
            sample = {
                "question": "What is the sum of the files size of the 2 files with the smallest file size in the table on page 98?",
                "answer_format": "Int",
                "prepare_metadata": {"context_builder": "aeg-rag", "graph_dir": str(graph_dir)},
            }

            pred, metadata = MMLongBenchAdapter.postprocess_prediction(sample, "Not answerable")

        self.assertEqual(pred, "9259")
        self.assertEqual(metadata["type"], "file_size_table_sum")

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
