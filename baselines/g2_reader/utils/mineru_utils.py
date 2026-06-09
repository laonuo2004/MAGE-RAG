

import os
import json
import re
from langchain_text_splitters import RecursiveCharacterTextSplitter

def extract_chunk_from_mineru(mineru_path, chunk_size=3000, chunk_overlap=300):
    """
    Extract and split text chunks from MinerU output
    
    Args:
        mineru_path: MinerU output path (xxx/auto)
    
    Returns:
        List of text chunks after RecursiveCharacterTextSplitter processing
    """

    md_files = [f for f in os.listdir(mineru_path) if f.endswith(".md")]
    if len(md_files) != 1:
        raise ValueError("No unique .md found in mineru folder.")

    md_path = os.path.join(mineru_path, md_files[0])

    with open(md_path, "r", encoding="utf-8") as f:
        full_text = f.read()

    full_text = re.sub(r'!\[\]\([^)]*\)', '', full_text)

    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=[
            "\n\n",
            "\n",
            "。", "！", "？",
            ".", "!", "?",
            " ",
            ""
        ]
    )

    chunks = text_splitter.split_text(full_text)

    return chunks

from PIL import Image

def extract_image_from_mineru(mineru_path, per_side_words=1000):
    """
    Extract images and contexts from MinerU output
    
    Args:
        mineru_path: MinerU output path (typically xxx/auto)
    
    Returns:
        images: List[PIL.Image], contexts: List[str]
    """

    json_file = None
    for f in os.listdir(mineru_path):
        if f.endswith("_content_list.json"):
            json_file = os.path.join(mineru_path, f)
            break

    if json_file is None:
        raise FileNotFoundError("No *_content_list.json found in MinerU output")

    with open(json_file, "r", encoding="utf-8") as f:
        data = json.load(f)

    images = []
    captions = []
    contexts = []

    for idx, item in enumerate(data):
        if item.get("type") != "image" and item.get("type") !="table":
            continue
        if not item.get("img_path"):
            continue

        img_path = os.path.join(mineru_path, item["img_path"])
        if item.get("type") == "image":
            cap_list = (item.get("image_caption", []) or []) + (item.get("image_footnote", []) or [])
        else:
            cap_list = (item.get("table_caption", []) or []) + (item.get("table_footnote", []) or [])
        caption = " ".join(cap_list).strip()
        captions.append(caption)

        try:
            img = Image.open(img_path).convert("RGB")
            images.append(img)
        except Exception:
            continue

        context_words = []
        wc = 0
        i = idx - 1
        while i >= 0 and wc < per_side_words:
            txt = data[i].get("text", "")
            words = txt.split()
            need = per_side_words - wc
            context_words = words[-need:] + context_words
            wc += len(words[-need:])
            i -= 1
        # forward（下方）
        wc = 0
        i = idx + 1
        while i < len(data) and wc < per_side_words:
            txt = data[i].get("text", "")
            words = txt.split()
            need = per_side_words - wc
            context_words = context_words + words[:need]
            wc += len(words[:need])
            i += 1

        context = " ".join(context_words).strip()
        contexts.append(context)

    return images, contexts, captions