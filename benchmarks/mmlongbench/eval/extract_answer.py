import time
import traceback

from fastapi import logger
from openai import OpenAI
import logging
logger = logging.getLogger("mmlongbench.extract_answer")

def extract_answer(
    question,
    output,
    prompt,
    model_name=None,
    client=None,
):
    question = "" if question is None else str(question)
    output = "" if output is None else str(output)
    prompt += "\n\nQuestion: " + question \
            + "\nAnalysis: " + output \
            + "\n\nPlease extract the final answer to the question based on the above analysis. Only provide the answer without any additional explanation."
            
    messages=[
        {
            "role": "user",
            "content": prompt,
        }
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
                max_tokens=256,
            )
            logger.debug("Raw response from LLM for answer extraction: %s", response.choices[0].message)
            response = response.choices[0].message.content
        except Exception:
            logger.warning("Error during answer extraction: %s, left %s attempts.", traceback.format_exc(), max_try - 1)
            max_try -= 1

    if response is None:
        logger.error("Failed to extract answer after multiple attempts.")

    return response
