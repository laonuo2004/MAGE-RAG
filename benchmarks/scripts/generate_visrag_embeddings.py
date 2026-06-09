#!/usr/bin/env python3
import argparse
import json
import logging
import re
import sys
import traceback
from pathlib import Path

import torch
import torch.nn.functional as F
from PIL import Image
from safetensors.torch import save_file
from tqdm import tqdm


CODE_DIR = Path(__file__).resolve().parents[2]
VENDORED_TIMM_DIR = CODE_DIR / 'baselines' / 'visrag' / 'timm_modified'
if str(CODE_DIR) not in sys.path:
    sys.path.insert(0, str(CODE_DIR))
if VENDORED_TIMM_DIR.exists() and str(VENDORED_TIMM_DIR) not in sys.path:
    sys.path.insert(0, str(VENDORED_TIMM_DIR))

from benchmarks.utils.data_utils import (  # noqa: E402
    LONGDOCURL_DEFAULT_SHARD,
    load_longdocurl_samples,
    load_mmlongbench_samples,
    mmlongbench_file_id,
    visrag_pdf_embeddings_path,
    visrag_question_embeddings_path,
)


DEFAULT_MMLONGBENCH_INPUT_PATH = CODE_DIR / 'benchmarks' / 'mmlongbench' / 'data' / 'raw' / 'samples.json'
DEFAULT_MMLONGBENCH_IMAGE_ROOT = CODE_DIR / 'benchmarks' / 'mmlongbench' / 'data' / 'processed' / 'pdf_pngs'
DEFAULT_LONGDOCURL_INPUT_PATH = CODE_DIR / 'benchmarks' / 'longdocurl' / 'data' / 'raw' / 'LongDocURL.jsonl'
DEFAULT_LONGDOCURL_IMAGE_ROOT = CODE_DIR / 'benchmarks' / 'longdocurl' / 'data' / 'processed' / 'pdf_pngs' / LONGDOCURL_DEFAULT_SHARD
DEFAULT_CHECKPOINT = '/root/autodl-tmp/ylz/models/VisRAG-Ret'
PAGE_RE_TEMPLATE = r'^page_(?P<page_num>\d{{4}})_dpi{dpi}\.png$'
LONGDOC_PAGE_RE = re.compile(r'^(?P<doc_id>.+)_(?P<page_idx>\d+)\.png$')
QUERY_INSTRUCTION = 'Represent this query for retrieving relevant documents: '

logger = logging.getLogger('generate_visrag_embeddings')


def parse_args():
    parser = argparse.ArgumentParser(description='Generate VisRAG-Ret page and question embedding caches.')
    parser.add_argument('--benchmark', choices=['mmlongbench', 'longdocurl'], required=True)
    parser.add_argument('--mode', choices=['pdf', 'question', 'both'], default='both')
    parser.add_argument('--input-path', default=None)
    parser.add_argument('--image-root', default=None)
    parser.add_argument('--checkpoint', default=DEFAULT_CHECKPOINT)
    parser.add_argument('--device', default='cuda')
    parser.add_argument('--dtype', choices=['float32', 'float16', 'bfloat16'], default='bfloat16')
    parser.add_argument('--attn-implementation', choices=['sdpa', 'eager'], default='eager')
    parser.add_argument('--batch-size', type=int, default=4)
    parser.add_argument('--limit', type=int, default=None)
    parser.add_argument('--doc-id', action='append', default=None)
    parser.add_argument('--question-id', action='append', default=None)
    parser.add_argument('--overwrite', action='store_true')
    parser.add_argument('--dpi', type=int, default=144)
    parser.add_argument('--longdocurl-shard', default=LONGDOCURL_DEFAULT_SHARD)
    parser.add_argument('--manifest-dir', default=None, help='Directory for manifest JSONL files. Defaults to cache/visrag/manifests.')
    parser.add_argument('--failed-output', default=None, help='Failed-record JSONL path. Defaults under manifest-dir.')
    parser.add_argument('--keep-going', action='store_true', help='Record failures and continue with remaining docs/questions.')
    parser.add_argument('--retry-failed', default=None, help='Read a failed-record JSONL and process only those doc/question ids.')
    parser.add_argument('--record-traceback', action='store_true', help='Include Python traceback text in failed manifest records.')
    return parser.parse_args()


def weighted_mean_pooling(hidden, attention_mask):
    attention_mask_ = attention_mask * attention_mask.cumsum(dim=1)
    numerator = torch.sum(hidden * attention_mask_.unsqueeze(-1).float(), dim=1)
    denominator = attention_mask_.sum(dim=1, keepdim=True).float()
    return numerator / denominator


def assert_finite_tensor(name, tensor):
    if not torch.isfinite(tensor.float()).all():
        nan_count = torch.isnan(tensor.float()).sum().item()
        raise ValueError(f'{name} contains non-finite values; nan_count={nan_count}, shape={tuple(tensor.shape)}')


@torch.no_grad()
def encode(model, tokenizer, text_or_image_list):
    if not text_or_image_list:
        return torch.empty((0, 0), dtype=torch.float32)
    if isinstance(text_or_image_list[0], str):
        inputs = {
            'text': text_or_image_list,
            'image': [None] * len(text_or_image_list),
            'tokenizer': tokenizer,
        }
    else:
        inputs = {
            'text': [''] * len(text_or_image_list),
            'image': text_or_image_list,
            'tokenizer': tokenizer,
        }
    outputs = model(**inputs)
    reps = weighted_mean_pooling(outputs.last_hidden_state, outputs.attention_mask)
    embeddings = F.normalize(reps, p=2, dim=1).detach().cpu()
    assert_finite_tensor('VisRAG-Ret embeddings', embeddings)
    return embeddings


def install_transformers_compat_shims():
    import transformers.utils.import_utils as import_utils
    from transformers.modeling_utils import PreTrainedModel

    if not hasattr(import_utils, 'is_torch_fx_available'):
        import_utils.is_torch_fx_available = lambda: hasattr(torch, 'fx')
    if not hasattr(PreTrainedModel, 'all_tied_weights_keys'):
        PreTrainedModel.all_tied_weights_keys = {}
    try:
        from transformers.cache_utils import DynamicCache

        if not hasattr(DynamicCache, 'from_legacy_cache'):
            DynamicCache.from_legacy_cache = classmethod(lambda cls, past_key_values=None: cls())
        if not hasattr(DynamicCache, 'get_usable_length'):
            DynamicCache.get_usable_length = lambda self, new_seq_length=None, layer_idx=0: self.get_seq_length(layer_idx)
        if not hasattr(DynamicCache, 'to_legacy_cache'):
            DynamicCache.to_legacy_cache = lambda self: ()
    except Exception:
        pass


def load_model(args):
    install_transformers_compat_shims()
    from transformers import AutoConfig, AutoModel

    dtype = {
        'float32': torch.float32,
        'float16': torch.float16,
        'bfloat16': torch.bfloat16,
    }[args.dtype]
    tokenizer = load_visrag_tokenizer(args.checkpoint)
    config = AutoConfig.from_pretrained(args.checkpoint, trust_remote_code=True)
    rope_scaling = getattr(config, 'rope_scaling', None)
    if isinstance(rope_scaling, dict) and rope_scaling.get('rope_type') == 'default' and 'type' not in rope_scaling:
        config.rope_scaling = None
    config.use_cache = False
    model = AutoModel.from_pretrained(
        args.checkpoint,
        config=config,
        trust_remote_code=True,
        torch_dtype=dtype,
        attn_implementation=args.attn_implementation,
    )
    model.eval()
    model.to(args.device)
    rebuild_minicpm_rope_cache(model)
    return model, tokenizer


def rebuild_minicpm_rope_cache(model):
    """Rebuild non-persistent RoPE buffers after loading VisRAG-Ret remote code.

    Newer transformers versions can leave MiniCPM rotary buffers with garbage
    values because the checkpoint does not persist them. Bad RoPE buffers make
    the first decoder layer return all-NaN hidden states.
    """
    layers = getattr(getattr(getattr(model, 'llm', None), 'model', None), 'layers', [])
    for layer in layers:
        rotary_emb = getattr(getattr(layer, 'self_attn', None), 'rotary_emb', None)
        if rotary_emb is None:
            continue
        device = rotary_emb.cos_cached.device
        inv_freq = 1.0 / (
            rotary_emb.base
            ** (torch.arange(0, rotary_emb.dim, 2, device=device, dtype=torch.float32) / rotary_emb.dim)
        )
        rotary_emb.register_buffer('inv_freq', inv_freq, persistent=False)
        rotary_emb._set_cos_sin_cache(
            seq_len=rotary_emb.max_position_embeddings,
            device=device,
            dtype=torch.float32,
        )
        assert_finite_tensor('MiniCPM RoPE cos_cached', rotary_emb.cos_cached)
        assert_finite_tensor('MiniCPM RoPE sin_cached', rotary_emb.sin_cached)


def default_input_path(args):
    if args.input_path:
        return Path(args.input_path)
    return DEFAULT_MMLONGBENCH_INPUT_PATH if args.benchmark == 'mmlongbench' else DEFAULT_LONGDOCURL_INPUT_PATH


def default_image_root(args):
    if args.image_root:
        return Path(args.image_root)
    return DEFAULT_MMLONGBENCH_IMAGE_ROOT if args.benchmark == 'mmlongbench' else DEFAULT_LONGDOCURL_IMAGE_ROOT


def manifest_dir(args):
    if args.manifest_dir:
        return Path(args.manifest_dir)
    return CODE_DIR / 'benchmarks' / args.benchmark / 'data' / 'cache' / 'visrag' / 'manifests'


def manifest_path(args):
    return manifest_dir(args) / f'{args.mode}_manifest.jsonl'


def failed_output_path(args):
    if args.failed_output:
        return Path(args.failed_output)
    return manifest_dir(args) / f'{args.mode}_failed.jsonl'


def write_jsonl(path, record):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('a', encoding='utf-8') as handle:
        handle.write(json.dumps(record, ensure_ascii=False) + '\n')


def failure_record(args, *, kind, key, output_path, exc):
    record = {
        'benchmark': args.benchmark,
        'kind': kind,
        'key': str(key),
        'status': 'failed',
        'output_path': str(output_path) if output_path is not None else None,
        'error_type': type(exc).__name__,
        'error': str(exc),
    }
    if args.record_traceback:
        record['traceback'] = traceback.format_exc()
    return record


def load_retry_filters(path):
    doc_ids = set()
    question_ids = set()
    if not path:
        return doc_ids, question_ids
    with Path(path).open('r', encoding='utf-8') as handle:
        for line in handle:
            if not line.strip():
                continue
            record = json.loads(line)
            if record.get('status') != 'failed':
                continue
            if record.get('kind') == 'pdf':
                doc_ids.add(str(record.get('key')))
            elif record.get('kind') == 'question':
                question_ids.add(str(record.get('key')))
    return doc_ids, question_ids


def load_visrag_tokenizer(checkpoint):
    checkpoint_path = Path(checkpoint)
    tokenizer_py = checkpoint_path / 'tokenizer.py'
    if tokenizer_py.exists():
        import importlib.util

        spec = importlib.util.spec_from_file_location('visrag_ret_tokenizer', tokenizer_py)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        # The original wrapper reads SentencePiece ids through sp_model methods.
        # With this environment's tokenizer stack, those resolve to property
        # objects, so expose the ids through stable PreTrainedTokenizer fields.
        module.LlamaTokenizerWrapper.bos_id = property(lambda self: self.bos_token_id)
        module.LlamaTokenizerWrapper.eos_id = property(lambda self: self.eos_token_id)
        module.LlamaTokenizerWrapper.unk_id = property(lambda self: self.unk_token_id)
        module.LlamaTokenizerWrapper.im_start_id = property(lambda self: self.convert_tokens_to_ids(self.im_start))
        module.LlamaTokenizerWrapper.im_end_id = property(lambda self: self.convert_tokens_to_ids(self.im_end))
        return module.LlamaTokenizerWrapper.from_pretrained(str(checkpoint_path), use_fast=False)

    from transformers import AutoTokenizer

    return AutoTokenizer.from_pretrained(checkpoint, trust_remote_code=True, use_fast=False)


def load_samples(args):
    if args.benchmark == 'mmlongbench':
        return load_mmlongbench_samples(default_input_path(args))
    return load_longdocurl_samples(default_input_path(args))


def selected_samples(args):
    samples = load_samples(args)
    retry_doc_ids, retry_question_ids = load_retry_filters(args.retry_failed)
    explicit_doc_ids = {str(value) for value in (args.doc_id or [])}
    explicit_question_ids = {str(value) for value in (args.question_id or [])}
    doc_ids = explicit_doc_ids | retry_doc_ids
    question_ids = explicit_question_ids | retry_question_ids
    retry_mode = bool(args.retry_failed)
    selected = []
    for sample in samples:
        doc_key = str(sample['doc_id'] if args.benchmark == 'mmlongbench' else sample['doc_no'])
        doc_file_id = mmlongbench_file_id(doc_key) if args.benchmark == 'mmlongbench' else doc_key
        qid = str(sample['question_id'])
        doc_match = doc_key in doc_ids or doc_file_id in doc_ids
        question_match = qid in question_ids
        if retry_mode:
            if not (doc_match or question_match):
                continue
        else:
            if doc_ids and not doc_match:
                continue
            if question_ids and not question_match:
                continue
        selected.append(sample)
    if args.limit is not None:
        selected = selected[:args.limit]
    return selected


def mmlongbench_doc_pages(image_root, doc_id, dpi):
    file_id = mmlongbench_file_id(doc_id)
    page_dir = image_root / file_id
    if not page_dir.exists():
        raise FileNotFoundError(f'Missing PNG directory for doc_id={doc_id}: {page_dir}')
    page_re = re.compile(PAGE_RE_TEMPLATE.format(dpi=re.escape(str(dpi))))
    pages = []
    for path in page_dir.glob(f'page_*_dpi{dpi}.png'):
        match = page_re.match(path.name)
        if match:
            pages.append((int(match.group('page_num')) - 1, path))
    if not pages:
        raise FileNotFoundError(f'No dpi={dpi} pages found for doc_id={doc_id} in {page_dir}')
    return file_id, sorted(pages, key=lambda item: item[0])


def longdocurl_doc_pages(image_root, doc_no):
    page_dir = image_root / doc_no[:4]
    if not page_dir.exists():
        raise FileNotFoundError(f'Missing PNG directory for doc_no={doc_no}: {page_dir}')
    pages = []
    for path in page_dir.glob(f'{doc_no}_*.png'):
        match = LONGDOC_PAGE_RE.match(path.name)
        if match and match.group('doc_id') == doc_no:
            pages.append((int(match.group('page_idx')), path))
    if not pages:
        raise FileNotFoundError(f'No PNG pages found for doc_no={doc_no} in {page_dir}')
    return doc_no, sorted(pages, key=lambda item: item[0])


def unique_doc_records(args, samples):
    image_root = default_image_root(args)
    records = {}
    for sample in samples:
        if args.benchmark == 'mmlongbench':
            doc_key, pages = mmlongbench_doc_pages(image_root, sample['doc_id'], args.dpi)
        else:
            doc_key, pages = longdocurl_doc_pages(image_root, sample['doc_no'])
        records.setdefault(doc_key, pages)
    return records


def output_doc_path(args, doc_key):
    if args.benchmark == 'longdocurl':
        return visrag_pdf_embeddings_path(args.benchmark, doc_key, shard=args.longdocurl_shard)
    return visrag_pdf_embeddings_path(args.benchmark, doc_key)


def encode_one_doc(args, model, tokenizer, doc_key, pages, output_path):
    output_path.parent.mkdir(parents=True, exist_ok=True)
    embeddings = []
    for start in range(0, len(pages), args.batch_size):
        batch_paths = [path for _, path in pages[start:start + args.batch_size]]
        images = [Image.open(path).convert('RGB') for path in batch_paths]
        try:
            embeddings.append(encode(model, tokenizer, images))
        finally:
            for image in images:
                image.close()
    doc_embs = torch.cat(embeddings, dim=0).to(torch.float32)
    page_indices = [idx for idx, _ in pages]
    expected = list(range(page_indices[-1] + 1))
    if page_indices != expected:
        raise ValueError(f'Non-contiguous pages for doc={doc_key}; got page indices {page_indices[:10]}...')
    save_file({'embeddings': doc_embs}, output_path)
    return tuple(doc_embs.shape)


def encode_docs(args, model, tokenizer, doc_records):
    failures = 0
    for doc_key, pages in tqdm(doc_records.items(), desc=f'{args.benchmark} VisRAG-Ret pages'):
        output_path = output_doc_path(args, doc_key)
        record = {
            'benchmark': args.benchmark,
            'kind': 'pdf',
            'key': str(doc_key),
            'output_path': str(output_path),
            'page_count': len(pages),
        }
        if output_path.exists() and not args.overwrite:
            record['status'] = 'skipped'
            write_jsonl(manifest_path(args), record)
            logger.info('Skipping existing PDF embedding: %s', output_path)
            continue
        try:
            shape = encode_one_doc(args, model, tokenizer, doc_key, pages, output_path)
            record.update({'status': 'generated', 'shape': list(shape)})
            write_jsonl(manifest_path(args), record)
            logger.info('Saved %s with shape %s', output_path, shape)
        except Exception as exc:
            failures += 1
            failed = failure_record(args, kind='pdf', key=doc_key, output_path=output_path, exc=exc)
            write_jsonl(manifest_path(args), failed)
            write_jsonl(failed_output_path(args), failed)
            logger.exception('Failed generating PDF embedding for %s', doc_key)
            if not args.keep_going:
                raise
    return failures


def encode_query_batch(model, tokenizer, batch):
    queries = [QUERY_INSTRUCTION + str(sample['question']) for sample in batch]
    return encode(model, tokenizer, queries).to(torch.float32)


def encode_questions(args, model, tokenizer, samples):
    failures = 0
    for start in tqdm(range(0, len(samples), args.batch_size), desc=f'{args.benchmark} VisRAG-Ret questions'):
        batch = samples[start:start + args.batch_size]
        pending = []
        for sample in batch:
            output_path = visrag_question_embeddings_path(args.benchmark, sample['question_id'])
            record = {
                'benchmark': args.benchmark,
                'kind': 'question',
                'key': str(sample['question_id']),
                'doc_key': str(sample.get('doc_id') or sample.get('doc_no')),
                'output_path': str(output_path),
            }
            if output_path.exists() and not args.overwrite:
                record['status'] = 'skipped'
                write_jsonl(manifest_path(args), record)
                logger.info('Skipping existing question embedding: %s', output_path)
                continue
            pending.append((sample, output_path, record))
        if not pending:
            continue
        try:
            query_embs = encode_query_batch(model, tokenizer, [item[0] for item in pending])
            for (sample, output_path, record), query_emb in zip(pending, query_embs):
                output_path.parent.mkdir(parents=True, exist_ok=True)
                save_file({'query_embedding': query_emb}, output_path)
                record.update({'status': 'generated', 'shape': list(query_emb.shape)})
                write_jsonl(manifest_path(args), record)
                logger.info('Saved %s with shape %s', output_path, tuple(query_emb.shape))
        except Exception as exc:
            failures += len(pending)
            logger.exception('Failed generating question embedding batch starting at %s', start)
            for sample, output_path, _ in pending:
                failed = failure_record(args, kind='question', key=sample['question_id'], output_path=output_path, exc=exc)
                failed['doc_key'] = str(sample.get('doc_id') or sample.get('doc_no'))
                write_jsonl(manifest_path(args), failed)
                write_jsonl(failed_output_path(args), failed)
            if not args.keep_going:
                raise
    return failures


def main():
    logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s %(message)s')
    args = parse_args()
    if args.batch_size <= 0:
        raise ValueError('--batch-size must be > 0')
    samples = selected_samples(args)
    if not samples:
        logger.info('No samples selected.')
        return
    model, tokenizer = load_model(args)
    failures = 0
    if args.mode in {'pdf', 'both'}:
        failures += encode_docs(args, model, tokenizer, unique_doc_records(args, samples))
    if args.mode in {'question', 'both'}:
        failures += encode_questions(args, model, tokenizer, samples)
    if failures:
        logger.warning('Finished with %s failed embedding records. Failed records: %s', failures, failed_output_path(args))
        if not args.keep_going:
            raise RuntimeError(f'Finished with {failures} failed embedding records.')


if __name__ == '__main__':
    main()
