import asyncio
from types import SimpleNamespace
from unittest.mock import patch

import utils.llm_utils as llm_utils
from utils.llm_utils import AsyncLLMExecutor, call_llm_messages, xml_block


def completion(content):
    return SimpleNamespace(
        choices=[
            SimpleNamespace(
                message=SimpleNamespace(content=content),
            )
        ],
    )


class FailingCompletions:
    def __init__(self, failures_before_success):
        self.failures_before_success = failures_before_success
        self.calls = 0

    def create(self, **kwargs):
        self.calls += 1
        if self.calls <= self.failures_before_success:
            raise RuntimeError(f"temporary failure {self.calls}")
        return completion("ok")


class AsyncRecordingCompletions:
    def __init__(self):
        self.active = 0
        self.max_active = 0
        self.calls = 0

    async def create(self, **kwargs):
        self.calls += 1
        self.active += 1
        self.max_active = max(self.max_active, self.active)
        try:
            await asyncio.sleep(0.01)
            return completion("ok")
        finally:
            self.active -= 1


def test_async_llm_executor_limits_in_flight_requests():
    completions = AsyncRecordingCompletions()
    client = SimpleNamespace(chat=SimpleNamespace(completions=completions))
    executor = AsyncLLMExecutor(client, concurrency=2)

    async def run_calls():
        return await asyncio.gather(*[
            executor.call_messages(
                "test-model",
                [{"role": "user", "content": f"hello {index}"}],
                retries=1,
            )
            for index in range(5)
        ])

    results = asyncio.run(run_calls())

    assert [result.choices[0].message.content for result in results] == ["ok"] * 5
    assert completions.calls == 5
    assert completions.max_active <= 2


def test_call_llm_messages_waits_with_exponential_backoff_between_retries():
    completions = FailingCompletions(failures_before_success=3)
    client = SimpleNamespace(chat=SimpleNamespace(completions=completions))

    with patch.object(llm_utils.time, "sleep") as sleep:
        result = call_llm_messages(
            client,
            "test-model",
            [{"role": "user", "content": "hello"}],
            retries=4,
        )

    assert result.choices[0].message.content == "ok"
    assert completions.calls == 4
    assert [call.args[0] for call in sleep.call_args_list] == [5, 10, 20]


def test_xml_block_keeps_simple_multiline_block_as_default():
    assert xml_block("question", "What is 1 million?") == "<question>\nWhat is 1 million?\n</question>"
    assert xml_block("empty", None) == "<empty>\n\n</empty>"


def test_xml_block_supports_attributes_and_escaping():
    block = xml_block(
        "field",
        "A < B & C",
        attributes={"name": "gold_truth", "required": True, "skip": None},
        escape=True,
        inline=True,
    )

    assert block == '<field name="gold_truth" required="true">A &lt; B &amp; C</field>'


def test_xml_block_supports_cdata_for_code_like_content():
    block = xml_block("scoring_code", "if a < b:\n    return 1", cdata=True)

    assert block == "<scoring_code>\n<![CDATA[if a < b:\n    return 1]]>\n</scoring_code>"


def test_xml_block_supports_template_positional_and_named_values():
    block = xml_block(
        "instruction",
        "22 million",
        "0.0",
        template="Gold: {0}\nInitial score: {1}\nMode: {mode}",
        template_kwargs={"mode": "conservative"},
    )

    assert block == "<instruction>\nGold: 22 million\nInitial score: 0.0\nMode: conservative\n</instruction>"


def test_xml_block_supports_indentation():
    block = xml_block("child", "value", indent=2)

    assert block == "  <child>\n  value\n  </child>"
