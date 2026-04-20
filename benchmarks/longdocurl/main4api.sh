# --qa_file /home/dataset-local/data/LongDocURL/LongDocURL_public.jsonl \

# # Top-5 docs, 165 * questions
# python eval/api_models/eval_api_models.py \
#     --qa_file /home/dataset-local/Projects/CodeLib/LongDocURL_dev/LongDocURL_public_top5_docs.jsonl \
#     --results_file results_gemini-3.1-pro-preview.jsonl \
#     --process_mode parallel \
#     --image_prefix /home/dataset-local/data/LongDocURL/pdf_pngs/4000-4999 \
#     --model_name gemini-3.1-pro-preview


# Top-5 docs, 165 * questions
python eval/api_models/eval_api_models.py \
    --qa_file /home/dataset-local/Projects/CodeLib/LongDocURL_dev/LongDocURL_public_top5_docs.jsonl \
    --results_file results_gpt-5.4.jsonl \
    --process_mode parallel \
    --image_prefix /home/dataset-local/data/LongDocURL/pdf_pngs/4000-4999 \
    --model_name gpt-5.4


# Top-5 docs, 165 * questions
python eval/api_models/eval_api_models.py \
    --qa_file /home/dataset-local/Projects/CodeLib/LongDocURL_dev/LongDocURL_public_top5_docs.jsonl \
    --results_file results_claude-sonnet-4-6.jsonl \
    --process_mode parallel \
    --image_prefix /home/dataset-local/data/LongDocURL/pdf_pngs/4000-4999 \
    --model_name claude-sonnet-4-6