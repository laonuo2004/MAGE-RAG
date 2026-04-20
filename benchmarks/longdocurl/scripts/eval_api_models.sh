python eval/api_models/eval_api_models.py \
    --qa_file data/LongDocURL.jsonl \
    # --results_file evaluation_results/api_models/results_gpt4o.jsonl \
    --process_mode serial \
    --image_prefix /root/autodl-tmp/ylz/NeurIPS_2026/code/benchmarks/longdocurl/data/pdf_pngs/4000-4999 \
    --model_name google/gemma-3-27b-it
