import traceback

from fastapi import logger
from openai import OpenAI
import logging
logger = logging.getLogger("mmlongbench.extract_answer")

def build_client(api_key=None, base_url=None):
    kwargs = {}
    # The OpenAI SDK requires a non-empty api_key even for local OpenAI-compatible
    # servers such as vLLM that ignore authentication.
    if api_key:
        kwargs["api_key"] = api_key
    elif base_url:
        kwargs["api_key"] = "EMPTY"
    if base_url:
        kwargs["base_url"] = base_url
    return OpenAI(**kwargs)


def extract_answer(
    question,
    output,
    prompt,
    model_name=None,
    client=None,
):
    messages=[
        {
            "role": "user",
            "content": prompt,
        },
        {
            "role": "assistant",
            "content": "\n\nQuestion:{}\nAnalysis:{}\n".format(question, output),
        },
    ]
    logger.debug("Prompt messages for answer extraction: %s", messages)
    response = None
    max_try = 3
    while response is None and max_try > 0:
        try:
            response = client.chat.completions.create(
                model=model_name,
                messages=messages,
                temperature=0.0,
                max_tokens=1024,
            )
            logger.debug("Raw response from LLM for answer extraction: %s", response.choices[0].message)
            response = response.choices[0].message.content
        except Exception:
            logger.warning("Error during answer extraction: %s, left %s attempts.", traceback.format_exc(), max_try - 1)
            max_try -= 1

    if response is None:
        logger.error("Failed to extract answer after multiple attempts.")

    return response
