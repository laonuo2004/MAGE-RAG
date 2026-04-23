python eval/api_models/eval_api_models.py \
    --qa_file data/LongDocURL.jsonl \
    --process_mode parallel \
    --llm_provider local \
    --input_format ocr \
    --ocr_backend pymupdf \
    --ocr_json_dir /root/autodl-tmp/ylz/NeurIPS_2026/code/benchmarks/longdocurl/data/pdf_jsons/4000-4999 \
    --image_prefix /root/autodl-tmp/ylz/NeurIPS_2026/code/benchmarks/longdocurl/data/pdf_pngs/4000-4999 \
    --model_name Qwen/Qwen2.5-VL-7B-Instruct