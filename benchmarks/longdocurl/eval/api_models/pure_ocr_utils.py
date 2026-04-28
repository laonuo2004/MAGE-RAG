import json
import os
import re
from collections import defaultdict

import logging
logger = logging.getLogger(__name__)

PAGE_NO_PATTERN = re.compile(r"_(\d+)\.png$")
OCR_TEXT_TEMPLATE = "page_no: {}\n{}\n\n"


def record2text_with_layout(record):
    text = ""
    img_width, img_height = record["docInfo"]["pages"][0]["imageWidth"], record["docInfo"]["pages"][0]["imageHeight"]
    for item in record["layouts"]:
        _type, sub_type = item["type"], item["subType"]
        item_text = item["text"]
        x1y1 = item["pos"][0]
        x2y2 = item["pos"][2]
        text += (
            f"(type: {_type}, sub_type: {sub_type}, box: "
            f"{tuple(float(f'{_:.2f}') for _ in (x1y1['x']/img_width, x1y1['y']/img_height, x2y2['x']/img_width, x2y2['y']/img_height))})"
            f" {item_text}\n"
        )

    return text


def record2text(record):
    text = ""
    for item in record["layouts"]:
        item_text = item["text"]
        text += f"{item_text}\n"

    return text


# def get_pure_ocr_prompt_docmind(doc_no: str, **kwargs):
#     zip_no = doc_no[:4]
#     json_path = "/mnt/achao/Downloads/pdf_jsons/{}/{}_docmind_results.json"
#     record = json.load(open(json_path.format(zip_no, doc_no), "r", encoding="utf-8"))["contents"]

#     start_page = kwargs.pop("start_page", 0)
#     end_page = kwargs.pop("end_page", start_page + 1)
#     if "extra_infos" in kwargs and "with_layout" in kwargs["extra_infos"] and kwargs["extra_infos"]["with_layout"]:
#         ocr_texts = [record2text_with_layout(record[f"page_{idx}"]) for idx in range(start_page, end_page + 1) if f"page_{idx}" in record]
#     else:
#         ocr_texts = [record2text(record[f"page_{idx}"]) for idx in range(start_page, end_page + 1) if f"page_{idx}" in record]
#     # print("Number Of Pages Used: ", end_page - start_page + 1)

#     ocr_prompt = "\n\n"
#     for page_no, ocr_text in zip(range(start_page, end_page + 1), ocr_texts):
#         ocr_prompt += OCR_TEXT_TEMPLATE.format(page_no + 1, ocr_text)

#     return ocr_prompt


def extract_page_nos_from_images(images):
    page_nos = []
    seen = set()
    for image_path in images or []:
        match = PAGE_NO_PATTERN.search(os.path.basename(image_path))
        if match is None:
            continue
        page_no = int(match.group(1))
        if page_no not in seen:
            seen.add(page_no)
            page_nos.append(page_no)
    return page_nos


def load_pymupdf_record(doc_no: str, ocr_json_dir: str):
    json_path = os.path.join(ocr_json_dir, doc_no[:4], f"{doc_no}.json")
    with open(json_path, "r", encoding="utf-8") as file:
        return json.load(file)


def build_page_texts_from_contents(contents, selected_pages):
    selected_page_set = set(selected_pages)
    page_map = defaultdict(lambda: defaultdict(lambda: defaultdict(list)))

    for item in contents:
        page_no = item["page_no"]
        if page_no not in selected_page_set:
            continue
        page_map[page_no][item["block_no"]][item["line_no"]].append(item)

    page_texts = []
    for page_no in selected_pages:
        if page_no not in page_map:
            continue

        lines = []
        for block_no in sorted(page_map[page_no]):
            for line_no in sorted(page_map[page_no][block_no]):
                words = sorted(page_map[page_no][block_no][line_no], key=lambda word: word["word_no"])
                line_text = " ".join(word["word"] for word in words if word["word"])
                if line_text:
                    lines.append(line_text)

        page_text = "\n".join(lines).strip()
        if page_text:
            page_texts.append((page_no, page_text))

    return page_texts


def get_pure_ocr_prompt_pymupdf(doc_no: str, images=None, ocr_json_dir=None, **kwargs):
    if ocr_json_dir is None:
        raise ValueError("ocr_json_dir is required for PyMuPDF OCR evaluation")

    selected_pages = extract_page_nos_from_images(images or [])
    if not selected_pages:
        start_page = kwargs.pop("start_page", 0)
        end_page = kwargs.pop("end_page", start_page + 1)
        selected_pages = list(range(start_page, end_page + 1))

    record = load_pymupdf_record(doc_no, ocr_json_dir)
    page_texts = build_page_texts_from_contents(record["contents"], selected_pages)
    pages_used = [page_no for page_no, _ in page_texts]
    logger.debug("number of pages used: {}".format(len(pages_used)))

    ocr_prompt = "\n\n"
    for page_no, page_text in page_texts:
        ocr_prompt += OCR_TEXT_TEMPLATE.format(page_no + 1, page_text)

    return ocr_prompt, pages_used
