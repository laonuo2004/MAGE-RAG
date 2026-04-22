import base64
from io import BytesIO
from abc import ABC, abstractmethod
from openai import OpenAI
import openai
import requests
import os
from typing import Optional, Sequence, Union
import json

# TODO
project_prefix = "/root/autodl-tmp/ylz/NeurIPS_2026/code/benchmarks/longdocurl/"
config_file = os.path.join(project_prefix, "config/api_config.json")


class APIInferencer(ABC):
    def __init__(self):
        pass
        # uncomment if oss paths are used
        # self.bucket = self.get_alimama_oss_bucket()

    def get_alimama_oss_bucket(self):
        raise NotImplementedError("OSS support is not configured for the API inferencer.")

    @abstractmethod
    def infer(self, prompt: str, image_path: Optional[Union[Sequence[str], str]]) -> str:
        pass

    def load_client(self):
        with open(config_file, "r", encoding="utf-8") as rf:
            config = json.load(rf)
        return OpenAI(api_key=config["api_model"]["access_key"], base_url=config["api_model"]["base_url"])

    def cleanup(self):
        if hasattr(self, 'client'):
            del self.client

    def encode_image_to_base64(self, image_path: str) -> str:
        if 'https' in image_path:
            response = requests.get(image_path)
            img = BytesIO(response.content)
            return base64.b64encode(img.read()).decode('utf-8')

        if image_path.startswith('oss://'):
            return base64.b64encode(self.bucket.get_object(image_path[6:].split("/", 1)[1]).read()).decode("utf-8")

        with open(image_path, "rb") as image_file:
            return base64.b64encode(image_file.read()).decode('utf-8')

    def get_correct_response(self, model_name: str, prompt: str, image_path: Optional[Union[Sequence[str], str]], client: openai.OpenAI) -> str:
        response = self.model_chat(model_name, prompt, image_path, client)
        return response

    def model_chat(self, model_name: str, prompt: str, image_path: Optional[Union[Sequence[str], str]], client: openai.OpenAI) -> str:
        # client = self.load_client()
        messages = [
            {
                "role": "user",
                "content": self.build_message_content(prompt, image_path)
            }
        ]
        max_try = 2
        response = None
        while response is None and max_try > 0:
            try:
                completion = client.chat.completions.create(model=model_name, messages=messages, temperature=0., max_completion_tokens=4096)
                response = completion.choices[0].message.content
            except Exception as e:
                print("exception: ", e)
                max_try -= 1
        return response

    def build_message_content(self, prompt: str, image_path: Optional[Union[Sequence[str], str]]):
        content = [{"type": "text", "text": prompt}]
        if image_path is None:
            return content
        if isinstance(image_path, str):
            image_paths = [image_path]
        elif isinstance(image_path, (list, tuple)):
            image_paths = image_path
        else:
            raise TypeError(f"Unsupported image_path type: {type(image_path)}")
        base64_images = [self.encode_image_to_base64(image_path) for image_path in image_paths]
        for i, base64_image in enumerate(base64_images):
            content += [
                {"type": "text", "text": f"Below is the {i+1}-th image (total {len(base64_images)} images).\n"},
                {
                    "type": "image_url",
                    "image_url": {
                        "url": f"data:image/png;base64,{base64_image}"
                    },
                },
            ]
        return content

# class QwenMaxInferencer(APIInferencer):
#     def infer(self, prompt: str, image_path: str, model_name='qwen_max') -> str:
#         response = self.get_correct_response(model_name, prompt, image_path)
#         return response


# class O1PreviewInferencer(APIInferencer):
#     def infer(self, prompt: str, image_path: str, model_name='o1-preview-0912') -> str:
#         response = self.get_correct_response(model_name, prompt, image_path)
#         return response

# class GPT4oInferencer(APIInferencer):
#     def infer(self, prompt: str, image_path: str, model_name='gpt-4o-0513') -> str:
#         response = self.get_correct_response(model_name, prompt, image_path)
#         # response = self.get_correct_response('gpt-4o', prompt, image_path)
#         return response

# class GPT54Inferencer(APIInferencer):
#     def infer(self, prompt: str, image_path: str, model_name='gpt-5.4') -> str:
#         response = self.get_correct_response(model_name, prompt, image_path)
#         return response

# class ClaudeSonnet46Inferencer(APIInferencer):
#     def infer(self, prompt: str, image_path: str, model_name='claude-sonnet-4-6') -> str:
#         response = self.get_correct_response(model_name, prompt, image_path)
#         return response

# class Gemini15ProInferencer(APIInferencer):
#     def infer(self, prompt: str, image_path: str, model_name='gemini-1.5-pro') -> str:
#         response = self.get_correct_response(model_name, prompt, image_path)
#         return response

# class Gemini31ProInferencer(APIInferencer):
#     def infer(self, prompt: str, image_path: str, model_name='gemini-3.1-pro-preview') -> str:
#         response = self.get_correct_response(model_name, prompt, image_path)
#         return response


# class QwenVLMaxInferencer(APIInferencer):
#     def infer(self, prompt: str, image_path: str, model_name='qwen-vl-max') -> str:
#         response = self.get_correct_response(model_name, prompt, image_path)
#         return response

# class Gemma3_27BInferencer(APIInferencer):
#     def infer(self, prompt: str, image_path: str, model_name: str) -> str:
#         response = self.get_correct_response(model_name, prompt, image_path)
#         return response
    
# class Gemma4_26B_A4BInferencer(APIInferencer):
#     def infer(self, prompt: str, image_path: str, model_name: str) -> str:
#         response = self.get_correct_response(model_name, prompt, image_path)
#         return response

class Inferencer(APIInferencer):
    def infer(self, prompt: str, image_path: Optional[Union[Sequence[str], str]], model_name: str, client: openai.OpenAI) -> str:
        response = self.get_correct_response(model_name, prompt, image_path, client)
        return response
