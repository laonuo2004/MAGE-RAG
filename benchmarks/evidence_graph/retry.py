import logging
import time
from collections.abc import Callable
from typing import TypeVar

T = TypeVar("T")

DEFAULT_RETRY_DELAYS_SECONDS = (5, 10, 20)


def call_with_retries(
    operation: Callable[[], T],
    *,
    operation_name: str,
    logger: logging.Logger | None = None,
    retry_delays_seconds: tuple[int, ...] = DEFAULT_RETRY_DELAYS_SECONDS,
) -> T:
    total_attempts = len(retry_delays_seconds) + 1
    for attempt in range(1, total_attempts + 1):
        try:
            return operation()
        except Exception as exc:
            if attempt == total_attempts:
                raise
            delay = retry_delays_seconds[attempt - 1]
            if logger is not None:
                logger.warning(
                    "%s failed. attempt=%s/%s retry_in=%ss error=%s",
                    operation_name,
                    attempt,
                    total_attempts,
                    delay,
                    exc,
                )
            time.sleep(delay)
    raise RuntimeError(f"{operation_name} retry loop exited unexpectedly")
