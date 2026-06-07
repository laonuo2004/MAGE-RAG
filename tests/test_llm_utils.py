from types import SimpleNamespace
from unittest.mock import patch

from utils.llm_utils import call_llm_messages


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


def test_call_llm_messages_waits_with_exponential_backoff_between_retries():
    completions = FailingCompletions(failures_before_success=3)
    client = SimpleNamespace(chat=SimpleNamespace(completions=completions))

    with patch("utils.llm_utils.time.sleep") as sleep:
        result = call_llm_messages(
            client,
            "test-model",
            [{"role": "user", "content": "hello"}],
            retries=4,
        )

    assert result.choices[0].message.content == "ok"
    assert completions.calls == 4
    assert [call.args[0] for call in sleep.call_args_list] == [5, 10, 20]
