import logging

from utils.llm_utils import call_llm_messages, text_content_parts

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
            "content": text_content_parts(prompt),
        }
    ]
    logger.debug("Prompt messages for answer extraction: %s", messages)
    response = call_llm_messages(
        client,
        model_name,
        messages,
        temperature=0.0,
        max_tokens=8192,
        retries=3,
        logger=logger,
        log_prefix="MMLongBench answer extraction",
    )

    if response is None:
        logger.error("Failed to extract answer after multiple attempts.")

    return response
