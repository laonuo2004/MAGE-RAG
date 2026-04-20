# model_name ["qwen2-vl-7b", "qwen25-vl-7b"]
CUDA_VISIBLE_DEVICES=0,1 python eval/eval_open_lvlms.py \
    --qa_file /root/autodl-tmp/ylz/NeurIPS_2026/code/benchmarks/longdocurl/data/LongDocURL.jsonl \
    --results_file results_qwen2vl_7b_dpi144.jsonl \
    --process_mode serial \
    --image_prefix /root/autodl-tmp/ylz/NeurIPS_2026/code/benchmarks/longdocurl/data/pdf_pngs/4000-4999/ \
    --model_name qwen2-vl-7b
