qa_reason_visual_prompt_template = """
Please read the following text and the attached images and answer the question below.

<text>
{context}
</text>

What is the correct answer to this question: {question}

Format your response as follows: "<reason>detailed reason for your answer here</reason><answer>the correct answer here</answer>". 
Please make sure that your answer is comprehensive and covers all the important information related to the question. 
Meanwhile, if directly relevant information is not provided, you should try your best to give an answer based on the available context (without mentioning that you do not have the information).
"""

import tiktoken
from openai import OpenAI
base_url = "<YOUR_BASE_URL>"
api_key = "<YOUR_API_KEY>"
client = OpenAI(api_key=api_key, base_url=base_url)
from llm_client import get_default_model

tokenizer= tiktoken.encoding_for_model("gpt-4o-2024-11-20")
max_len = 40000 
max_char = 262000 

DEFAULT_MODEL = get_default_model()

def amem_qa_visual(texts, images, question, model=DEFAULT_MODEL, temperature=0):
    prompt = qa_reason_visual_prompt_template.format(question = question, context = texts)
    user_content = [{"type":"text","text": prompt}]
    if len(images) > 0:
        for image in images:
            user_content.append({"type": "image_url","image_url": {"url": f"data:image/jpeg;base64,{image}"}})
    response = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "user", "content": user_content},
            ],
            temperature=temperature,
            max_tokens=35048
        )
    return {
        "answer": response.choices[0].message.content,
        "usage": {
            "prompt_tokens": response.usage.prompt_tokens,
            "completion_tokens": response.usage.completion_tokens,
            "total_tokens": response.usage.total_tokens
        }
    }