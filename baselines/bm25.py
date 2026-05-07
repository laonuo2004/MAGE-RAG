import math
import re
from collections import Counter

from .base import ContextBuilder, ContextMessages
from baselines.utils.benchmarks_related import load_longdocurl_ocr_pages, load_mmlongbench_ocr_pages
from utils.config_utils import get_config_value, require_config_value


TEXT_SYSTEM_PROMPT = (
    'You are an expert in document question-answering. '
    'Answer the question using only the retrieved OCR/text chunks from the document. '
    'If the answer cannot be found, say Not answerable.\n'
)


class BM25ContextBuilder(ContextBuilder):
    name = 'bm25'

    def __init__(self, cfg=None):
        super().__init__(cfg)
        self.top_k = int(get_config_value(self.cfg, 'baselines.top_k', 5))
        self.chunk_size = int(get_config_value(self.cfg, 'baselines.chunk_size', 200))
        self.chunk_overlap = int(get_config_value(self.cfg, 'baselines.chunk_overlap', 50))
        self.k1 = float(get_config_value(self.cfg, 'baselines.bm25_k1', 1.5))
        self.b = float(get_config_value(self.cfg, 'baselines.bm25_b', 0.75))
        self.lowercase = bool(get_config_value(self.cfg, 'baselines.lowercase', True))
        self.min_token_length = int(get_config_value(self.cfg, 'baselines.min_token_length', 1))
        self.max_chunks_per_page = get_config_value(self.cfg, 'baselines.max_chunks_per_page', None)
        if self.max_chunks_per_page is not None:
            self.max_chunks_per_page = int(self.max_chunks_per_page)
        self.tokenizer = str(get_config_value(self.cfg, 'baselines.tokenizer', 'spacy'))
        self.spacy_model = str(get_config_value(self.cfg, 'baselines.spacy_model', 'en'))
        self.allow_regex_fallback = bool(get_config_value(self.cfg, 'baselines.allow_regex_tokenizer_fallback', True))
        self.max_context_chars = get_config_value(self.cfg, 'baselines.max_context_chars', None)
        if self.max_context_chars is not None:
            self.max_context_chars = int(self.max_context_chars)

        if self.top_k <= 0:
            raise ValueError('BM25 baseline requires cfg.baselines.top_k > 0.')
        if self.chunk_size <= 0:
            raise ValueError('BM25 baseline requires cfg.baselines.chunk_size > 0.')
        if self.chunk_overlap < 0 or self.chunk_overlap >= self.chunk_size:
            raise ValueError('BM25 baseline requires 0 <= cfg.baselines.chunk_overlap < chunk_size.')
        if self.max_chunks_per_page is not None and self.max_chunks_per_page <= 0:
            raise ValueError('BM25 baseline requires cfg.baselines.max_chunks_per_page > 0 when set.')
        self._nlp = self._load_tokenizer()

    def build_mmlongbench(self, sample, **kwargs):
        benchmark_cfg = require_config_value(self.cfg, 'benchmarks')
        pages, allowed_pages = load_mmlongbench_ocr_pages(sample, benchmark_cfg)
        chunks = self._build_chunks(pages)
        retrieval = self._retrieve_chunks(sample['question'], chunks)
        prompt = self._build_prompt(sample['question'], retrieval)
        return ContextMessages(
            [{'role': 'user', 'content': prompt}],
            metadata=self._metadata(retrieval, allowed_pages),
        )

    def build_longdocurl(self, sample, **kwargs):
        benchmark_cfg = require_config_value(self.cfg, 'benchmarks')
        pages, allowed_pages = load_longdocurl_ocr_pages(sample, benchmark_cfg)
        chunks = self._build_chunks(pages)
        retrieval = self._retrieve_chunks(sample['question'], chunks)
        prompt = self._build_prompt(sample['question'], retrieval)
        return ContextMessages(
            [{'role': 'user', 'content': [{'type': 'text', 'text': prompt}]}],
            metadata=self._metadata(retrieval, allowed_pages),
        )

    def _build_chunks(self, pages):
        chunks = []
        for page in pages:
            token_spans = self._tokenize_with_spans(page['text'])
            if not token_spans:
                token_spans = [{'token': 'empty', 'start': 0, 'end': len(page['text'])}]

            step = self.chunk_size - self.chunk_overlap
            for chunk_index, start in enumerate(range(0, len(token_spans), step)):
                end = min(start + self.chunk_size, len(token_spans))
                chunk_tokens = token_spans[start:end]
                text_start = chunk_tokens[0]['start']
                text_end = chunk_tokens[-1]['end']
                chunk_text = page['text'][text_start:text_end].strip() or '[EMPTY CHUNK]'
                chunks.append({
                    'chunk_id': len(chunks),
                    'chunk_index': chunk_index,
                    'page_index': page['page_index'],
                    'page_number': page['page_number'],
                    'text': chunk_text,
                    'tokens': [token['token'] for token in chunk_tokens],
                })
                if end == len(token_spans):
                    break
        if not chunks:
            raise ValueError('BM25 retrieval has no OCR chunks to score.')
        return chunks

    def _retrieve_chunks(self, question, chunks):
        query_tokens = [token['token'] for token in self._tokenize_with_spans(question)]
        if not query_tokens:
            query_tokens = ['empty']

        scores = self._bm25_scores(query_tokens, [chunk['tokens'] for chunk in chunks])
        ranked = sorted(
            enumerate(scores),
            key=lambda item: (item[1], -chunks[item[0]]['chunk_id']),
            reverse=True,
        )
        retrieval = []
        page_counts = Counter()
        for chunk_offset, score in ranked:
            chunk = chunks[chunk_offset]
            if (
                self.max_chunks_per_page is not None
                and page_counts[chunk['page_index']] >= self.max_chunks_per_page
            ):
                continue
            page_counts[chunk['page_index']] += 1
            retrieval.append({
                'rank': len(retrieval) + 1,
                'chunk_id': chunk['chunk_id'],
                'chunk_index': chunk['chunk_index'],
                'page_index': chunk['page_index'],
                'page_number': chunk['page_number'],
                'score': float(score),
                'text': chunk['text'],
            })
            if len(retrieval) >= self.top_k:
                break
        return retrieval

    def _bm25_scores(self, query_tokens, corpus_tokens):
        doc_count = len(corpus_tokens)
        doc_lens = [len(tokens) for tokens in corpus_tokens]
        avg_doc_len = sum(doc_lens) / doc_count if doc_count else 0.0
        document_frequencies = Counter()
        term_frequencies = []
        for tokens in corpus_tokens:
            counts = Counter(tokens)
            term_frequencies.append(counts)
            document_frequencies.update(counts.keys())

        scores = []
        for counts, doc_len in zip(term_frequencies, doc_lens):
            score = 0.0
            for token in query_tokens:
                tf = counts.get(token, 0)
                if tf <= 0:
                    continue
                df = document_frequencies[token]
                idf = math.log(1.0 + (doc_count - df + 0.5) / (df + 0.5))
                denominator = tf + self.k1 * (1.0 - self.b + self.b * doc_len / max(avg_doc_len, 1e-9))
                score += idf * (tf * (self.k1 + 1.0)) / denominator
            scores.append(score)
        return scores

    def _build_prompt(self, question, retrieval):
        chunk_blocks = []
        total_chars = 0
        for item in retrieval:
            block = (
                f'[Page {item["page_number"]} | Chunk {item["chunk_index"] + 1} | '
                f'BM25 score {item["score"]:.4f}]\n{item["text"]}'
            )
            if self.max_context_chars is not None and total_chars + len(block) > self.max_context_chars:
                break
            chunk_blocks.append(block)
            total_chars += len(block)

        retrieved_text = '\n\n'.join(chunk_blocks)
        return (
            TEXT_SYSTEM_PROMPT
            + '\nQuestion:\n'
            + question
            + '\n\nRetrieved OCR/text chunks:\n'
            + retrieved_text
        )

    def _load_tokenizer(self):
        if self.tokenizer != 'spacy':
            if self.tokenizer == 'regex':
                return None
            raise ValueError(f'Unsupported BM25 tokenizer: {self.tokenizer}')
        try:
            import spacy
        except ModuleNotFoundError as exc:
            if self.allow_regex_fallback:
                return None
            raise ModuleNotFoundError(
                'BM25 baseline requires spaCy. Install spacy or set '
                'baselines.allow_regex_tokenizer_fallback=true for local smoke tests.'
            ) from exc
        if self.spacy_model in ('blank', 'en', 'en_blank'):
            return spacy.blank('en')
        return spacy.load(self.spacy_model)

    def _tokenize_with_spans(self, text):
        if self._nlp is None:
            return self._regex_tokenize_with_spans(text)
        tokens = []
        for token in self._nlp(text):
            if token.is_space or token.is_punct or token.is_quote or token.is_bracket:
                continue
            value = token.text.lower() if self.lowercase else token.text
            if not any(char.isalnum() for char in value):
                continue
            if len(value) < self.min_token_length:
                continue
            tokens.append({'token': value, 'start': token.idx, 'end': token.idx + len(token.text)})
        return tokens

    def _regex_tokenize_with_spans(self, text):
        tokens = []
        for match in re.finditer(r'\w+', text):
            value = match.group(0).lower() if self.lowercase else match.group(0)
            if len(value) < self.min_token_length:
                continue
            tokens.append({'token': value, 'start': match.start(), 'end': match.end()})
        return tokens

    def _metadata(self, retrieval, allowed_pages):
        retrieved_pages = []
        best_by_page = {}
        for item in retrieval:
            page_index = item['page_index']
            current = best_by_page.get(page_index)
            if current is None or item['score'] > current['score']:
                best_by_page[page_index] = {
                    'page_index': page_index,
                    'page_number': item['page_number'],
                    'score': item['score'],
                }
        for page_index in sorted(best_by_page):
            retrieved_pages.append(best_by_page[page_index])
        return {
            'context_builder': self.name,
            'retrieved_chunks': retrieval,
            'retrieved_pages': retrieved_pages,
            'allowed_pages': list(allowed_pages),
            'top_k': self.top_k,
            'chunk_size': self.chunk_size,
            'chunk_overlap': self.chunk_overlap,
            'max_chunks_per_page': self.max_chunks_per_page,
            'tokenizer': self.tokenizer if self._nlp is not None else 'regex',
            'bm25_k1': self.k1,
            'bm25_b': self.b,
        }
