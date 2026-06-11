import ast
import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from omegaconf import OmegaConf

import benchmarks.adapters as adapters
import baselines.magerag.evaluator as magerag_evaluator
from baselines.base import ContextMessages
from baselines.magerag.evaluator import XMLEvaluator
from benchmarks.adapters import LongDocURLAdapter, MMLongBenchAdapter
from utils.llm_utils import text_content_parts


class StubContextBuilder:
    def build(self, benchmark_name, sample, **kwargs):
        self.last_kwargs = kwargs
        return ContextMessages(
            [{"role": "user", "content": text_content_parts(f"{benchmark_name}:{sample['question']}")}],
            metadata={"context_builder": "stub", "sample_question": sample["question"]},
        )


class MAGERAGStubContextBuilder:
    def build(self, benchmark_name, sample, **kwargs):
        self.last_kwargs = kwargs
        return ContextMessages(
            [{"role": "user", "content": text_content_parts(f"magerag:{benchmark_name}:{sample['question']}")}],
            metadata={"context_builder": "magerag", "sample_question": sample["question"]},
        )


class SelfAnsweringStubContextBuilder:
    def run_sample(self, benchmark_name, sample, **kwargs):
        self.last_kwargs = kwargs
        return {
            "response": f"{benchmark_name}:{sample['question']}",
            "pred": "answer",
            "pred_format": "String",
            "usage": {"prompt_tokens": 11, "completion_tokens": 3, "total_tokens": 14},
            "metadata": {
                "context_builder": "g2-reader",
                "model": "g2-model",
                "sample_key": sample["question_id"],
            },
        }


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

    def test_mmlongbench_load_samples_honors_benchmark_limit(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            input_path = Path(tmp_dir) / "samples.json"
            samples = [
                {"doc_id": f"d{idx}", "question": f"q{idx}", "answer": "a", "answer_format": "Str"}
                for idx in range(3)
            ]
            input_path.write_text(json.dumps(samples), encoding="utf-8")
            cfg = OmegaConf.create({"benchmarks": {"input_path": str(input_path), "limit": 2}})

            loaded = MMLongBenchAdapter().load_samples(cfg)

        self.assertEqual([sample["doc_id"] for sample in loaded], ["d0", "d1"])

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

    def test_longdocurl_load_samples_honors_benchmark_limit(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            qa_file = Path(tmp_dir) / "qa.jsonl"
            qa_file.write_text(
                "\n".join(
                    json.dumps({"question": f"q{idx}", "answer": "a", "answer_format": "Str"})
                    for idx in range(3)
                )
                + "\n",
                encoding="utf-8",
            )
            cfg = OmegaConf.create({"benchmarks": {"qa_file": str(qa_file), "limit": 2}})

            samples = LongDocURLAdapter().load_samples(cfg)

        self.assertEqual([sample["question"] for sample in samples], ["q0", "q1"])

    def test_score_success_field_is_unified(self):
        self.assertTrue(MMLongBenchAdapter().is_successful_result({"pred": "ok", "score": 1.0}))
        self.assertTrue(LongDocURLAdapter().is_successful_result({"pred": "ok", "score": 1.0}))
        self.assertFalse(LongDocURLAdapter().is_successful_result({"pred": "ok", "score_v3": 1.0}))
        for adapter in (MMLongBenchAdapter(), LongDocURLAdapter()):
            with self.subTest(adapter=adapter.name):
                self.assertFalse(adapter.is_successful_result({"pred": "Fail to answer", "score": 0.0}))
                self.assertFalse(adapter.is_successful_result({"pred": "Failed: Connection error.", "score": 0.0}))
                self.assertFalse(adapter.is_successful_result({
                    "pred": "Not answerable",
                    "score": 1.0,
                    "generation_metadata": {"response": "Failed: Connection error."},
                }))
                self.assertFalse(adapter.is_successful_result({
                    "pred": "Not answerable",
                    "score": 1.0,
                    "prepare_metadata": {
                        "iteration_trace": [
                            {
                                "action": "EvaluatorDecision",
                                "raw_response": (
                                    "<agent_decision><stop>true</stop>"
                                    "<reason>Failed: Connection error.</reason></agent_decision>"
                                ),
                            }
                        ],
                    },
                }))
                self.assertFalse(adapter.is_successful_result({
                    "pred": "Not answerable",
                    "score": 1.0,
                    "prepare_metadata": {
                        "stop_reason": "fallback_invalid_xml",
                        "validation_errors": [
                            {"action_type": "Evaluator", "message": "Connection error."}
                        ],
                    },
                }))

    def test_generation_failure_is_not_converted_to_model_response(self):
        sample = {}
        def fail_or_raise(*args, **kwargs):
            failure_value = kwargs.get("failure_value")
            if failure_value is not None:
                return failure_value(RuntimeError("Connection error."))
            raise RuntimeError("Connection error.")

        with patch.object(adapters, "call_llm_messages", side_effect=fail_or_raise):
            with self.assertRaisesRegex(RuntimeError, "Connection error"):
                adapters._generate_response(sample, [], "qa-model", object(), "generation")

        self.assertNotIn("generation_metadata", sample)

    def test_magerag_evaluator_failure_is_not_converted_to_stop_decision(self):
        def fail_or_raise(*args, **kwargs):
            failure_value = kwargs.get("failure_value")
            if failure_value is not None:
                return failure_value(RuntimeError("Connection error."))
            raise RuntimeError("Connection error.")

        evaluator = XMLEvaluator("controller-model")
        with patch.object(evaluator, "build_prompt", return_value="prompt"), \
                patch.object(evaluator, "_opened_node_image_parts", return_value=[]), \
                patch.object(magerag_evaluator, "call_llm_messages", side_effect=fail_or_raise):
            with self.assertRaisesRegex(RuntimeError, "Connection error"):
                evaluator.call(object(), "question", object(), [])

    def test_finalize_result_fields_adds_run_envelope_and_orders_metadata(self):
        cfg = OmegaConf.create({
            "benchmarks": {
                "name": "longdocurl",
                "qa_model_name": "qa-model",
                "extractor_model_name": "extractor-model",
            },
            "baselines": {
                "name": "bm25",
                "params": {"top_k": 5},
            },
        })
        sample = {
            "question_id": 7,
            "question": "q",
            "answer": "a",
            "answer_format": "Str",
            "prepare_metadata": {"context_builder": "bm25"},
            "pred": "a",
            "pred_format": "Str",
            "score": 1.0,
        }

        finalized = adapters._finalize_result_fields(sample, cfg=cfg)

        self.assertEqual(finalized["benchmark"], "longdocurl")
        self.assertEqual(finalized["baseline"], "bm25")
        self.assertEqual(finalized["run_config"]["benchmark"]["name"], "longdocurl")
        self.assertEqual(finalized["run_config"]["baseline"]["name"], "bm25")
        self.assertEqual(finalized["run_config"]["baseline"]["params"]["top_k"], 5)
        self.assertLess(list(finalized).index("run_config"), list(finalized).index("prepare_metadata"))

    def test_self_answering_baseline_metadata_uses_unified_schema(self):
        sample = {"question_id": "q1", "question": "What is the answer?"}
        builder = SelfAnsweringStubContextBuilder()

        result = adapters._process_self_answering_sample(sample, builder, "mmlongbench", client=object())

        self.assertEqual(result["prepare_metadata"]["context_builder"], "g2-reader")
        self.assertEqual(result["prepare_metadata"]["retrieval"]["retrieved_items"], [])
        self.assertEqual(result["prepare_metadata"]["context_summary"]["num_context_pages"], 0)
        self.assertEqual(result["prepare_metadata"]["logical_cost"]["num_llm_calls"], 1)
        self.assertEqual(result["prepare_metadata"]["logical_cost"]["estimated_input_tokens"], 11)
        self.assertEqual(result["generation_metadata"]["usage"]["total_tokens"], 14)

    def test_mmlongbench_list_score_splits_comma_phrase_and_percentage_points(self):
        score = MMLongBenchAdapter.score(
            "['White', '10%']",
            "White, 10 percentage points",
            "List",
        )

        self.assertEqual(score, 1.0)

    def test_mmlongbench_list_score_allows_household_category_suffix(self):
        score = MMLongBenchAdapter.score(
            "['White', '10%']",
            "White households, 10 percentage points",
            "List",
        )

        self.assertEqual(score, 1.0)

    def test_mmlongbench_int_score_handles_percent_suffix(self):
        self.assertEqual(MMLongBenchAdapter.score("21%", "21", "Int"), 1.0)

    def test_correction_prompt_includes_only_string_scoring_code_for_string_answers(self):
        prompt = adapters._build_correction_messages(
            MMLongBenchAdapter(),
            "mmlongbench",
            {
                "question": "Which group is identified?",
                "answer": "less well-off",
                "answer_format": "String",
            },
            "The group is less well off.",
            "less well off",
            0.9,
        )[0]["content"][0]["text"]

        self.assertIn("def score_string", prompt)
        self.assertIn("_anls_compute", prompt)
        self.assertIn("hyphenation", prompt)
        self.assertIn("<think>...</think>", prompt)
        self.assertIn("<thinking_policy>", prompt)
        self.assertIn("<output_schema>", prompt)
        self.assertIn("<self_check>", prompt)
        self.assertIn("<corrected_extraction>", prompt)
        self.assertIn("Output sequence:", prompt)
        self.assertNotIn("def score_int", prompt)
        self.assertNotIn("def score_float", prompt)
        self.assertNotIn("def score_list", prompt)

    def test_correction_prompt_selects_float_scoring_code_for_float_answers(self):
        prompt = adapters._build_correction_messages(
            MMLongBenchAdapter(),
            "mmlongbench",
            {
                "question": "What percentage changed?",
                "answer": "10%",
                "answer_format": "Float",
            },
            "The change was 10 percentage points.",
            "10 percentage points",
            0.0,
        )[0]["content"][0]["text"]

        self.assertIn("def score_float", prompt)
        self.assertIn("_is_float_equal", prompt)
        self.assertNotIn("def score_int", prompt)
        self.assertNotIn("def score_string", prompt)
        self.assertNotIn("def score_list", prompt)

    def test_correction_prompt_selects_list_scoring_code_for_list_answers(self):
        prompt = adapters._build_correction_messages(
            MMLongBenchAdapter(),
            "mmlongbench",
            {
                "question": "Which category and percentage?",
                "answer": "['white','10%']",
                "answer_format": "List",
            },
            "White households, 10 percentage points.",
            "White households, 10 percentage points",
            0.0,
        )[0]["content"][0]["text"]

        self.assertIn("def score_list", prompt)
        self.assertIn("_normalize_list_item", prompt)
        self.assertIn("list-equivalent", prompt)
        self.assertNotIn("def score_int", prompt)
        self.assertNotIn("def score_float", prompt)
        self.assertNotIn("def score_string", prompt)

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
        self.assertIn("<question>q</question>", calls[1][0][2][0]["content"][0]["text"])
        self.assertIn("<model_response>raw answer</model_response>", calls[1][0][2][0]["content"][0]["text"])
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

    def test_extraction_prompt_requests_think_then_structured_xml_result(self):
        prompt = MMLongBenchAdapter().build_extraction_messages(
            {"question": "q"},
            "analysis",
        )[0]["content"][0]["text"]

        self.assertIn("<extraction_prompt>", prompt)
        self.assertIn("<role>", prompt)
        self.assertIn("<objective>", prompt)
        self.assertIn("<input_data>", prompt)
        self.assertIn("<extraction_policy>", prompt)
        self.assertIn("<not_answerable_policy>", prompt)
        self.assertIn("<format_policy>", prompt)
        self.assertIn("<thinking_policy>", prompt)
        self.assertIn("<output_schema>", prompt)
        self.assertIn("<few_shot_examples>", prompt)
        self.assertIn("<think>...</think>", prompt)
        self.assertIn("<extraction_result>", prompt)
        self.assertIn("Extracted answer: [answer]", prompt)
        self.assertIn("Answer format: [answer_format]", prompt)
        self.assertIn("Use None as answer format only when the extracted answer is exactly Not answerable or Fail to answer.", prompt)
        self.assertIn("<question>q</question>", prompt)
        self.assertIn("<model_response>analysis</model_response>", prompt)

    def test_longdocurl_extraction_prompt_constrains_none_format_to_unanswerable(self):
        prompt = LongDocURLAdapter().build_extraction_messages(
            {"question": "q"},
            "analysis",
        )[0]["content"][0]["text"]

        self.assertIn("<extraction_prompt>", prompt)
        self.assertIn("<target_answer_formats>Integer, Float, String, List, None</target_answer_formats>", prompt)
        self.assertIn("Use None as answer format only when the extracted answer is exactly Not answerable or Fail to answer.", prompt)
        self.assertIn("If the model response contains a concrete answer, never use None as the answer format.", prompt)

    def test_mmlongbench_extraction_strips_think_and_result_xml_before_parsing(self):
        pred, pred_format = MMLongBenchAdapter.parse_extraction_result(
            "<think>Compare the response to the question and choose the concise answer.</think>\n"
            "<extraction_result>\n"
            "Extracted answer: less well-off\n"
            "Answer format: String\n"
            "</extraction_result>"
        )

        self.assertEqual(pred, "less well-off")
        self.assertEqual(pred_format, "String")

    def test_mmlongbench_extraction_unwraps_answer_tags_and_code_fences(self):
        pred, pred_format = MMLongBenchAdapter.parse_extraction_result(
            "```xml\n"
            "<think>The answer block contains the concise answer.</think>\n"
            "<extraction_result>\n"
            "Extracted answer: <answer>less well-off</answer>\n"
            "Answer format: <answer_format>String</answer_format>\n"
            "</extraction_result>\n"
            "```"
        )

        self.assertEqual(pred, "less well-off")
        self.assertEqual(pred_format, "String")

    def test_longdocurl_extraction_strips_think_and_result_xml_before_parsing(self):
        pred, pred_format = LongDocURLAdapter().parse_extraction_result(
            "<think>The response gives a numeric percentage answer.</think>\n"
            "<extraction_result>\n"
            "Extracted answer: <concise_answer>84 percent</concise_answer>\n"
            "Answer format: <answer_format>String</answer_format>\n"
            "</extraction_result>"
        )

        self.assertEqual(pred, "84 percent")
        self.assertEqual(pred_format, "String")

    def test_magerag_generation_response_unwraps_answer_for_extraction(self):
        calls = []
        original_call_llm_messages = adapters.call_llm_messages
        try:
            def fake_call_llm_messages(*args, **kwargs):
                calls.append((args, kwargs))
                if len(calls) == 1:
                    return completion(
                        "<think>The evidence states the normalized phrase.</think>\n"
                        "<answer>less well-off</answer>",
                        model="qa-served",
                    )
                return completion(
                    "Extracted answer: less well-off\nAnswer format: String",
                    model="extractor-served",
                )

            adapters.call_llm_messages = fake_call_llm_messages
            cfg = OmegaConf.create({
                "benchmarks": {
                    "qa_model_name": "qa-model",
                    "extractor_model_name": "extractor-model",
                }
            })
            sample = {"doc_id": "d1", "question": "q", "answer": "less well-off", "answer_format": "String"}

            result = MMLongBenchAdapter().process_sample(sample, cfg, MAGERAGStubContextBuilder(), object())
        finally:
            adapters.call_llm_messages = original_call_llm_messages

        extraction_prompt = calls[1][0][2][0]["content"][0]["text"]
        self.assertIn("<model_response>less well-off</model_response>", extraction_prompt)
        self.assertNotIn("<answer>less well-off</answer>", extraction_prompt)
        self.assertEqual(result["generation_metadata"]["response"], "less well-off")
        self.assertIn("<think>", result["generation_metadata"]["raw_response"])
        self.assertEqual(result["pred"], "less well-off")

    def test_magerag_generation_records_raw_response_when_answer_tag_is_absent(self):
        calls = []
        original_call_llm_messages = adapters.call_llm_messages
        try:
            def fake_call_llm_messages(*args, **kwargs):
                calls.append((args, kwargs))
                if len(calls) == 1:
                    return completion(
                        "<think>The evidence identifies the two capacity columns.</think>\n"
                        "Installed photovoltaic (PV) power*; Installed wind turbine capacity*",
                        model="qa-served",
                    )
                return completion(
                    "Extracted answer: Installed photovoltaic (PV) power*; Installed wind turbine capacity*\n"
                    "Answer format: String",
                    model="extractor-served",
                )

            adapters.call_llm_messages = fake_call_llm_messages
            cfg = OmegaConf.create({
                "benchmarks": {
                    "qa_model_name": "qa-model",
                    "extractor_model_name": "extractor-model",
                }
            })
            sample = {"doc_id": "d1", "question": "q", "answer": "capacity", "answer_format": "String"}

            result = MMLongBenchAdapter().process_sample(sample, cfg, MAGERAGStubContextBuilder(), object())
        finally:
            adapters.call_llm_messages = original_call_llm_messages

        self.assertEqual(
            result["generation_metadata"]["response"],
            "Installed photovoltaic (PV) power*; Installed wind turbine capacity*",
        )
        self.assertIn("<think>", result["generation_metadata"]["raw_response"])
        extraction_prompt = calls[1][0][2][0]["content"][0]["text"]
        self.assertIn(
            "<model_response>Installed photovoltaic (PV) power*; Installed wind turbine capacity*</model_response>",
            extraction_prompt,
        )

    def test_mmlongbench_adapter_has_no_prediction_postprocess_hook(self):
        self.assertFalse(hasattr(MMLongBenchAdapter, "postprocess_prediction"))

    def test_longdocurl_adapter_has_no_prediction_postprocess_hook(self):
        self.assertFalse(hasattr(LongDocURLAdapter, "postprocess_prediction"))

    def test_mmlongbench_correction_runs_for_non_full_score_and_updates_final_score(self):
        calls = []
        original_call_llm_messages = adapters.call_llm_messages
        try:
            def fake_call_llm_messages(*args, **kwargs):
                calls.append((args, kwargs))
                if len(calls) == 1:
                    return completion(
                        "The total grants made for community engagement in 2020 were $22 million.",
                        model="qa-served",
                    )
                if len(calls) == 2:
                    return completion(
                        "Extracted answer: 22000000\nAnswer format: Integer",
                        model="extractor-served",
                    )
                return completion(
                    "Extracted answer: 22 million\nAnswer format: String",
                    model="correction-served",
                )

            adapters.call_llm_messages = fake_call_llm_messages
            cfg = OmegaConf.create({
                "benchmarks": {
                    "qa_model_name": "qa-model",
                    "extractor_model_name": "extractor-model",
                    "correction_enabled": True,
                    "correction_model_name": "correction-model",
                }
            })
            sample = {
                "doc_id": "d1",
                "question": "What was the total amount of grants made for community engagement in 2020?",
                "answer": "22 million",
                "answer_format": "String",
            }

            result = MMLongBenchAdapter().process_sample(sample, cfg, StubContextBuilder(), object())
        finally:
            adapters.call_llm_messages = original_call_llm_messages

        self.assertEqual([call[0][1] for call in calls], ["qa-model", "extractor-model", "correction-model"])
        correction_prompt = calls[2][0][2][0]["content"][0]["text"]
        self.assertIn("<question>", correction_prompt)
        self.assertIn(sample["question"], correction_prompt)
        self.assertIn("<gold_truth>", correction_prompt)
        self.assertIn("22 million", correction_prompt)
        self.assertIn("<model_response>", correction_prompt)
        self.assertIn("<initial_formatted_extraction>", correction_prompt)
        self.assertIn("def score", correction_prompt)
        self.assertIn("<correction_prompt>", correction_prompt)
        self.assertIn("<input_data>", correction_prompt)
        self.assertIn("<scoring_code>\n<![CDATA[", correction_prompt)
        self.assertIn("<initial_score>0.0</initial_score>", correction_prompt)
        self.assertIn("Use gold_truth as a reference target", correction_prompt)
        self.assertIn("Actively fix representation errors", correction_prompt)
        self.assertIn("benchmark-friendly surface form", correction_prompt)
        self.assertIn("Do not copy gold_truth into the answer unless model_response supports it", correction_prompt)
        self.assertEqual(result["pred"], "22000000")
        self.assertEqual(result["pred_format"], "Integer")
        self.assertEqual(result["score"], 0.0)
        self.assertEqual(result["corrected_pred"], "22 million")
        self.assertEqual(result["corrected_format"], "String")
        self.assertEqual(result["corrected_score"], 1.0)
        self.assertEqual(result["correction_metadata"]["initial_pred"], "22000000")
        self.assertEqual(result["correction_metadata"]["corrected_pred"], "22 million")
        self.assertTrue(result["correction_metadata"]["applied"])

    def test_correction_strips_think_block_before_parsing_corrected_extraction(self):
        calls = []
        original_call_llm_messages = adapters.call_llm_messages
        try:
            def fake_call_llm_messages(*args, **kwargs):
                calls.append((args, kwargs))
                if len(calls) == 1:
                    return completion("The answer is $22 million.", model="qa-served")
                if len(calls) == 2:
                    return completion("Extracted answer: 22000000\nAnswer format: Integer", model="extractor-served")
                return completion(
                    "<think>The model response gives the same amount with a scale word, so normalize it.</think>\n"
                    "<corrected_extraction>\n"
                    "Extracted answer: 22 million\n"
                    "Answer format: String\n"
                    "</corrected_extraction>",
                    model="correction-served",
                )

            adapters.call_llm_messages = fake_call_llm_messages
            cfg = OmegaConf.create({
                "benchmarks": {
                    "qa_model_name": "qa-model",
                    "extractor_model_name": "extractor-model",
                    "correction_enabled": True,
                    "correction_model_name": "correction-model",
                }
            })
            sample = {"doc_id": "d1", "question": "q", "answer": "22 million", "answer_format": "String"}

            result = MMLongBenchAdapter().process_sample(sample, cfg, StubContextBuilder(), object())
        finally:
            adapters.call_llm_messages = original_call_llm_messages

        self.assertEqual(result["pred"], "22000000")
        self.assertEqual(result["pred_format"], "Integer")
        self.assertEqual(result["score"], 0.0)
        self.assertEqual(result["corrected_pred"], "22 million")
        self.assertEqual(result["corrected_format"], "String")
        self.assertEqual(result["corrected_score"], 1.0)
        self.assertEqual(result["correction_metadata"]["corrected_pred"], "22 million")
        self.assertIn("<think>", result["correction_metadata"]["corrected_extracted_res"])

    def test_correction_result_fields_are_flattened_and_ordered_for_comparison(self):
        calls = []
        original_call_llm_messages = adapters.call_llm_messages
        try:
            def fake_call_llm_messages(*args, **kwargs):
                calls.append((args, kwargs))
                if len(calls) == 1:
                    return completion("The answer is $22 million.", model="qa-served")
                if len(calls) == 2:
                    return completion("Extracted answer: 22000000\nAnswer format: Integer", model="extractor-served")
                return completion("Extracted answer: 22 million\nAnswer format: String", model="correction-served")

            adapters.call_llm_messages = fake_call_llm_messages
            cfg = OmegaConf.create({
                "benchmarks": {
                    "qa_model_name": "qa-model",
                    "extractor_model_name": "extractor-model",
                    "correction_enabled": True,
                    "correction_model_name": "correction-model",
                }
            })
            sample = {"doc_id": "d1", "question": "q", "answer": "22 million", "answer_format": "String"}

            result = MMLongBenchAdapter().process_sample(sample, cfg, StubContextBuilder(), object())
        finally:
            adapters.call_llm_messages = original_call_llm_messages

        self.assertEqual(result["pred"], "22000000")
        self.assertEqual(result["pred_format"], "Integer")
        self.assertEqual(result["score"], 0.0)
        self.assertEqual(result["corrected_pred"], "22 million")
        self.assertEqual(result["corrected_format"], "String")
        self.assertEqual(result["corrected_score"], 1.0)
        keys = list(result)
        self.assertEqual(
            keys[keys.index("prepare_metadata"):keys.index("score") + 4],
            [
                "prepare_metadata",
                "generation_metadata",
                "extraction_metadata",
                "correction_metadata",
                "pred",
                "pred_format",
                "score",
                "corrected_pred",
                "corrected_format",
                "corrected_score",
            ],
        )

    def test_longdocurl_correction_reparses_xml_and_preserves_original_extraction_metadata(self):
        calls = []
        original_call_llm_messages = adapters.call_llm_messages
        try:
            def fake_call_llm_messages(*args, **kwargs):
                calls.append((args, kwargs))
                if len(calls) == 1:
                    return completion("The workforce share was 84%.", model="qa-served")
                if len(calls) == 2:
                    return completion(
                        "Extracted answer: <concise_answer>84</concise_answer>\n"
                        "Answer format: <answer_format>Integer</answer_format>",
                        model="extractor-served",
                    )
                return completion(
                    "Extracted answer: <concise_answer>84 percent</concise_answer>\n"
                    "Answer format: <answer_format>String</answer_format>",
                    model="correction-served",
                )

            adapters.call_llm_messages = fake_call_llm_messages
            cfg = OmegaConf.create({
                "benchmarks": {
                    "qa_model_name": "qa-model",
                    "extractor_model_name": "extractor-model",
                    "correction_enabled": True,
                    "correction_model_name": "correction-model",
                }
            })
            sample = {
                "question_id": "q1",
                "question": "What was the percentage of the U.S. workforce employed in service sectors by 2010?",
                "answer": "84 percent",
                "answer_format": "String",
            }

            result = LongDocURLAdapter().process_sample(sample, cfg, StubContextBuilder(), object())
        finally:
            adapters.call_llm_messages = original_call_llm_messages

        self.assertEqual(result["extraction_metadata"]["extracted_res"], (
            "Extracted answer: <concise_answer>84</concise_answer>\n"
            "Answer format: <answer_format>Integer</answer_format>"
        ))
        self.assertEqual(result["pred"], 84)
        self.assertEqual(result["pred_format"], "Integer")
        self.assertEqual(result["score"], 0.0)
        self.assertEqual(result["corrected_pred"], "84 percent")
        self.assertEqual(result["corrected_format"], "String")
        self.assertEqual(result["corrected_score"], 1.0)
        self.assertEqual(result["correction_metadata"]["corrected_score"], 1.0)
        self.assertTrue(result["correction_metadata"]["applied"])

    def test_correction_is_skipped_for_full_initial_score(self):
        calls = []
        original_call_llm_messages = adapters.call_llm_messages
        try:
            def fake_call_llm_messages(*args, **kwargs):
                calls.append((args, kwargs))
                if len(calls) == 1:
                    return completion("answer", model="qa-served")
                return completion("Extracted answer: answer\nAnswer format: String", model="extractor-served")

            adapters.call_llm_messages = fake_call_llm_messages
            cfg = OmegaConf.create({
                "benchmarks": {
                    "qa_model_name": "qa-model",
                    "extractor_model_name": "extractor-model",
                    "correction_enabled": True,
                    "correction_model_name": "correction-model",
                }
            })
            sample = {"doc_id": "d1", "question": "q", "answer": "answer", "answer_format": "String"}

            result = MMLongBenchAdapter().process_sample(sample, cfg, StubContextBuilder(), object())
        finally:
            adapters.call_llm_messages = original_call_llm_messages

        self.assertEqual([call[0][1] for call in calls], ["qa-model", "extractor-model"])
        self.assertNotIn("correction_metadata", result)
        self.assertEqual(result["corrected_pred"], result["pred"])
        self.assertEqual(result["corrected_format"], result["pred_format"])
        self.assertEqual(result["corrected_score"], result["score"])

    def test_correction_parse_failure_preserves_initial_result(self):
        calls = []
        original_call_llm_messages = adapters.call_llm_messages
        try:
            def fake_call_llm_messages(*args, **kwargs):
                calls.append((args, kwargs))
                if len(calls) == 1:
                    return completion("The answer is $22 million.", model="qa-served")
                if len(calls) == 2:
                    return completion("Extracted answer: 22000000\nAnswer format: Integer", model="extractor-served")
                return completion("not parseable", model="correction-served")

            adapters.call_llm_messages = fake_call_llm_messages
            cfg = OmegaConf.create({
                "benchmarks": {
                    "qa_model_name": "qa-model",
                    "extractor_model_name": "extractor-model",
                    "correction_enabled": True,
                    "correction_model_name": "correction-model",
                }
            })
            sample = {"doc_id": "d1", "question": "q", "answer": "22 million", "answer_format": "String"}

            result = MMLongBenchAdapter().process_sample(sample, cfg, StubContextBuilder(), object())
        finally:
            adapters.call_llm_messages = original_call_llm_messages

        self.assertEqual(result["pred"], "22000000")
        self.assertEqual(result["score"], 0.0)
        self.assertEqual(result["corrected_pred"], "22000000")
        self.assertEqual(result["corrected_format"], "Integer")
        self.assertEqual(result["corrected_score"], 0.0)
        self.assertFalse(result["correction_metadata"]["applied"])
        self.assertFalse(result["correction_metadata"]["changed"])
        self.assertFalse(result["correction_metadata"]["improved"])
        self.assertIn("error", result["correction_metadata"])

    def test_correction_connection_error_returns_none_so_result_is_retried(self):
        calls = []
        original_call_llm_messages = adapters.call_llm_messages
        try:
            def fake_call_llm_messages(*args, **kwargs):
                calls.append((args, kwargs))
                if len(calls) == 1:
                    return completion("The answer is $22 million.", model="qa-served")
                if len(calls) == 2:
                    return completion("Extracted answer: 22000000\nAnswer format: Integer", model="extractor-served")
                raise ConnectionError("correction service unavailable")

            adapters.call_llm_messages = fake_call_llm_messages
            cfg = OmegaConf.create({
                "benchmarks": {
                    "qa_model_name": "qa-model",
                    "extractor_model_name": "extractor-model",
                    "correction_enabled": True,
                    "correction_model_name": "correction-model",
                }
            })
            sample = {"doc_id": "d1", "question": "q", "answer": "22 million", "answer_format": "String"}

            result = MMLongBenchAdapter().process_sample(sample, cfg, StubContextBuilder(), object())
        finally:
            adapters.call_llm_messages = original_call_llm_messages

        self.assertIsNone(result)
        self.assertEqual([call[0][1] for call in calls], ["qa-model", "extractor-model", "correction-model"])


if __name__ == "__main__":
    unittest.main()
