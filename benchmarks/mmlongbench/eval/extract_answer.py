from openai import OpenAI


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
    model_name="gpt-4o",
    client=None,
    api_key=None,
    base_url=None,
):
    if client is None:
        client = build_client(api_key=api_key, base_url=base_url)

    try:
        response = client.chat.completions.create(
            model=model_name,
            messages=[
                {
                    "role": "user",
                    "content": prompt,
                },
                {
                    "role": "assistant",
                    "content": "\n\nQuestion:{}\nAnalysis:{}\n".format(question, output),
                },
            ],
            temperature=0.0,
            max_tokens=256,
            top_p=1,
            frequency_penalty=0,
            presence_penalty=0,
        )
        response = response.choices[0].message.content
    except Exception:
        response = "Failed"

    return response
