import os
import json
import time
import asyncio
from typing import Dict, List, Tuple, Any
import numpy as np
import pandas as pd
from openai import OpenAI,AsyncOpenAI

from config.config import (
    LLM_BASE_URL,
    LLM_API_KEY,
    MODELS,
    LLM_GENERATION,
    PROMPTS,
    MEMORY_SYSTEMS_DIR,
    PDF_TMP_DIR,
    DATASETS,
    MAX_CONCURRENCY,
    PARALLEL_ANALYSIS,
)

from prebuild.memory_layer import AgenticMemorySystem
from utils.visdom_utils import (
    clean_text,
    get_pdf,
    encode_image,
)
from utils.mineru_utils import extract_chunk_from_mineru, extract_image_from_mineru

from prebuild.usage_tracker import add_chat_usage, add_embed_usage, add_stage_duration, add_single_call_duration
# Optional: tqdm_asyncio is nice to have; if not available, fallback to asyncio.gather
try:
    _use_tqdm = True
except Exception:
    _use_tqdm = False

# -----------------------------
# Client
# -----------------------------
qwen_aclient = AsyncOpenAI(api_key=LLM_API_KEY, base_url=LLM_BASE_URL)

_llm_sem_by_loop = {}

def _get_llm_semaphore():
    loop = asyncio.get_running_loop()
    sem = _llm_sem_by_loop.get(id(loop))
    if sem is None:
        sem = asyncio.Semaphore(MAX_CONCURRENCY)
        _llm_sem_by_loop[id(loop)] = sem
    return sem

# -----------------------------
# Utilities
# -----------------------------

def ensure_dir(p: str):
    os.makedirs(p, exist_ok=True)

def find_actual_file(base_dir: str, filename: str) -> str | None:
    """Robust matching for potentially mangled filenames (encoding issues)."""
    direct_path = os.path.join(base_dir, filename)
    if os.path.exists(direct_path):
        return direct_path

    try:
        if not os.path.exists(base_dir):
            print(f"Warning: directory does not exist: {base_dir}")
            return None
        all_files = os.listdir(base_dir)
        filename_base, filename_ext = os.path.splitext(filename)
        for actual in all_files:
            ab, ae = os.path.splitext(actual)
            if ae.lower() != filename_ext.lower():
                continue
            if ab.lower() == filename_base.lower():
                return os.path.join(base_dir, actual)
            ne = filename_base.encode("utf-8", errors="ignore").decode("utf-8", errors="ignore")
            na = ab.encode("utf-8", errors="ignore").decode("utf-8", errors="ignore")
            if na.lower() == ne.lower():
                return os.path.join(base_dir, actual)
        from difflib import SequenceMatcher
        best, score = None, 0.8
        for actual in all_files:
            if os.path.splitext(actual)[1].lower() == filename_ext.lower():
                r = SequenceMatcher(None, filename.lower(), actual.lower()).ratio()
                if r > score:
                    best, score = actual, r
        if best:
            print(f"找到相似文件: {filename} -> {best} (相似度: {score:.2%})")
            return os.path.join(base_dir, best)
        print(f"警告：未找到文件: {filename}")
        return None
    except Exception as e:
        print(f"查找文件时出错 {filename}: {e}")
        return None

# -----------------------------
# Dataset helpers
# -----------------------------
_dataset_cache: Dict[str, pd.DataFrame] = {}

def load_dataset_df(name: str) -> pd.DataFrame:
    cfg = DATASETS[name]
    if name in _dataset_cache:
        return _dataset_cache[name]
    enc = cfg.get("encoding", "utf-8")
    df = pd.read_csv(cfg["csv"], encoding=enc)
    _dataset_cache[name] = df
    return df

# Mineru dir aggregation
def resolve_docs_from_dataset_mineru(dataset_name: str, q_id: str, limit: int = 5) -> Tuple[str, List[str]]:
    cfg = DATASETS[dataset_name]
    df = load_dataset_df(dataset_name)
    key, docs_col = cfg["key"], cfg["docs_col"]

    matches = np.where(df[key] == q_id)[0]
    if len(matches) == 0:
        raise ValueError(
            f"error: data not found in {dataset_name}.csv for {key}='{q_id}'.\n"
            f"available {key} values: {df[key].unique()[:10].tolist()}..."
        )

    row = df.iloc[matches[0]].to_dict()
    base_dir = cfg["mineru_dir"]

    doc_names_raw = list(eval(row[docs_col]))[:limit]
    doc_names = [os.path.splitext(d)[0] for d in doc_names_raw]

    mineru_paths: List[str] = []

    for doc in doc_names:
        p = find_actual_file(base_dir, doc)

        for _ in range(2):
            subs = [d for d in os.listdir(p) if os.path.isdir(os.path.join(p, d))]
            if len(subs) == 1:
                p = os.path.join(p, subs[0])
            else:
                break

        if p:
            mineru_paths.append(p)
        else:
            print(f"skip file not found: {doc}")

    print(f"found {len(mineru_paths)}/{len(doc_names)} files")
    return base_dir, mineru_paths

def _log_failed_response(raw: str, error: Exception, is_multimodal: bool, user_payload):
    """record failed LLM response to log file"""
    from datetime import datetime
    log_dir = os.path.join(MEMORY_SYSTEMS_DIR, "_debug_logs")
    os.makedirs(log_dir, exist_ok=True)
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    log_file = os.path.join(log_dir, f"failed_response_{timestamp}.txt")
    
    with open(log_file, "w", encoding="utf-8") as f:
        f.write(f"=" * 80 + "\n")
        f.write(f"time: {datetime.now().isoformat()}\n")
        f.write(f"type: {'multimodal' if is_multimodal else 'text'}\n")
        f.write(f"error: {type(error).__name__}: {error}\n")
        f.write(f"=" * 80 + "\n\n")
        
        f.write("[raw response]\n")
        f.write("-" * 40 + "\n")
        f.write(raw if raw else "(empty response)")
        f.write("\n" + "-" * 40 + "\n\n")
        
        f.write(f"response length: {len(raw) if raw else 0} characters\n")
        f.write(f"finish_reason: (see error information above)\n\n")
        
        # record input (for text type)
        if not is_multimodal and isinstance(user_payload, str):
            f.write("[input content (first 2000 characters of user_payload)]\n")
            f.write("-" * 40 + "\n")
            f.write(user_payload[:2000])
            f.write("\n" + "-" * 40 + "\n")
    
    print(f"[DEBUG] failed response saved to: {log_file}")
    return log_file


async def call_llm_json(system: str, user_payload, *, is_multimodal: bool = False):
    sem = _get_llm_semaphore()
    async with sem:
        call_start = time.time() 
        try:
            resp = await qwen_aclient.chat.completions.create(
                    model=MODELS["chat"],
                    messages=[
                        {"role": "system", "content": system},
                        {"role": "user", "content": user_payload},
                    ],
                    response_format={"type": "json_object"},
                    **LLM_GENERATION,
                )
        except Exception as err:
                # network / timeout / Azure filtering
            print(f"[ERROR] LLM request failed: {err}")
            await asyncio.sleep(1)
            raise 

        # only record duration for successful calls
        call_duration = time.time() - call_start
        
        try:
            qkind = "analyze_multimodal" if is_multimodal else "analyze_text"
            add_chat_usage(
                getattr(resp, "usage", None),
                {"model": MODELS["chat"], "qkind": qkind}
            )
            # only record duration for successful calls
            stage = "image_analysis" if is_multimodal else "text_analysis"
            add_single_call_duration(stage, call_duration)
        except Exception:
            pass 

        raw = resp.choices[0].message.content
        finish_reason = resp.choices[0].finish_reason if resp.choices else None
        
        # check if truncated
        if finish_reason == "length":
            print(f"[WARNING] response truncated (finish_reason=length),可能导致 JSON 不完整")
        
        try:
            return json.loads(raw)
        except json.JSONDecodeError as e:
            print(f"{e}")
            # try to extract JSON object
            import re
            m = re.search(r"\{[\s\S]*\}", raw)
            if m:
                try:
                    return json.loads(m.group(0))
                except json.JSONDecodeError as e2:
                    # record failed response
                    _log_failed_response(raw, e2, is_multimodal, user_payload)
                    raise
            # record failed response
            _log_failed_response(raw, e, is_multimodal, user_payload)
            raise
    

async def analyze_content(payload: str, *, modality: str) -> Dict[str, any]:
    """Unified analyzer for text or image.
    - modality: 'text' (payload is plain text) or 'image' (payload is base64 string)
    """
    system = "You must respond with a JSON object."
    if modality == "text":
        content = clean_text(payload)
        user = PROMPTS["text_keyword"] + content
        return await call_llm_json(system, user)
    elif modality == "image":
        user = [
            {"type": "text", "text": PROMPTS["image_keyword"]},
            {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{payload}"}},
        ]
        return await call_llm_json(system, user, is_multimodal=True)
    else:
        raise ValueError("modality must be 'text' or 'image'")

async def analyze_content_mineru(payload: str, *, modality: str, context: str = "", caption: str = "") -> Dict[str, any]:
    """Unified analyzer for text or image.
    - modality: 'text' (payload is plain text) or 'image' (payload is base64 string)
    """
    system = "You must respond with a JSON object."
    if modality == "text":
        content = clean_text(payload)
        user = PROMPTS["text"] + content
        return await call_llm_json(system, user)
    elif modality == "image":
        prompt  = PROMPTS["image_ocr_keyword"].replace("{context}", context).replace("{caption}", caption)
        user = [
            {"type": "text", "text": prompt},
            {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{payload}"}},
        ]

        return await call_llm_json(system, user, is_multimodal=True)
    else:
        raise ValueError("modality must be 'text' or 'image'")
# -----------------------------
# Embedding helpers
# -----------------------------
async def embed_one(text: str, kind: str = "embedding") -> List[float]:
    # use separate Embedding async client
    call_start = time.time()  
    try:
        resp = await embed_aclient.embeddings.create(model=MODELS["embed"], input=text)
    except Exception as err:
        # API failed, not record duration for this call
        print(f"[ERROR] Embedding request failed: {err}")
        raise
    
    # only record duration for successful calls
    call_duration = time.time() - call_start
    
    try:
        add_embed_usage(getattr(resp, "usage", None), {"model": MODELS["embed"], "kind": kind})
        # only record duration for successful calls
        add_single_call_duration(kind, call_duration)
    except Exception:
        pass
    return resp.data[0].embedding  # type: ignore


async def embed_many(texts: List[str], kind: str = "embedding") -> List[List[float]]:
    # simple concurrency control to avoid overwhelming server
    sem = asyncio.Semaphore(MAX_CONCURRENCY)

    async def _task(t: str):
        async with sem:
            return await embed_one(t, kind=kind)

    return await asyncio.gather(*[_task(t) for t in texts])


# -----------------------------
# Public API (main entry points)
# -----------------------------
async def construct_memory(
    pdf_path: str,
    *,
    evolve_iters: int = 1,
    window_size: int = 2,
) -> AgenticMemorySystem:
    """
    Build the memory system for a given q_id (dataset) or a remote/local PDF path.

    Keeps visdom_utils & memory_layer imports intact.
    Uses config for URLs, API keys, absolute paths, and prompts.
    """
    print("\n" + "=" * 80)
    print(f"start building Memory System: {pdf_path}")
    print("=" * 80)

    ensure_dir(MEMORY_SYSTEMS_DIR)
    ensure_dir(PDF_TMP_DIR)

    # detect dataset by substring; else treat as URL/local file and download
    detected_dataset = next((n for n in DATASETS.keys() if n in pdf_path), None)

    flag_dataset = detected_dataset is not None
    name = pdf_path 
    if flag_dataset:
        _, pdf_paths = resolve_docs_from_dataset_mineru(detected_dataset, pdf_path)
        print(pdf_paths)
    else:
        print("Downloading pdf")
        filename = time.strftime("%Y%m%d_%H%M%S") + ".pdf"
        out_path = os.path.join(PDF_TMP_DIR, filename)
        try:
            await get_pdf(pdf_path, out_path)
        except Exception as e:
            print(f"Error downloading pdf: {e}")
            raise
        pdf_paths = [out_path]

    # load or initialize memory system
    existing = set(os.listdir(MEMORY_SYSTEMS_DIR))
    if name in existing:
        print(f"Loading existing memory system: {name}")
        ms = AgenticMemorySystem(model_name=MODELS["embed"], llm_model=MODELS["chat"])  # type: ignore
        ms.load_memory_system(name+"_iter_"+str(evolve_iters))
    else:
        print("initialize memory system")
        ms = AgenticMemorySystem(model_name=MODELS["embed"], llm_model=MODELS["chat"])  # type: ignore

        # Extract text & images from MinerU output
        chunks = []
        images_b64 = []
        contexts = []
        captions = []
        for p in pdf_paths:
            chunks.extend(extract_chunk_from_mineru(p))  
            imgs, context, caption = extract_image_from_mineru(p)
            images_b64.extend(encode_image(img) for img in imgs)
            contexts.extend(context)
            captions.extend(caption)


        # analyze needs to be modified TODO
        
        # ============================================================
        # define text processing function (analyze + embed)
        # ============================================================
        async def process_text_content():
            """process all text content: analyze + embed"""
            print(f"Analyzing textual content ({len(chunks)} chunks)...")
            text_analysis_start = time.time()
            text_tasks = [analyze_content(ch, modality="text") for ch in chunks]
             
            try:
                text_results = await asyncio.gather(*text_tasks, return_exceptions=True)
            except Exception as e:
                print(f"error: text analysis batch processing failed: {e}")
                raise
            text_analysis_duration = time.time() - text_analysis_start
            add_stage_duration("text_analysis", text_analysis_duration)
            
            text_contents: Dict[int, Dict[str, Any]] = {}
            failed = 0
            for i, r in enumerate(text_results):
                if isinstance(r, Exception) or not isinstance(r, dict) or "summary" not in r:
                    print(r)
                    failed += 1
                    text_contents[i] = {"summary": "content analysis failed", "keywords": ["unknown"], "tags": ["error"]}
                else:
                    text_contents[i] = r
            if failed:
                print(f"warning: {failed}/{len(chunks)} text chunks analysis failed, using default values")
            if len(text_contents) == 0:
                raise RuntimeError("❌ serious error: all text chunks analysis failed!")
            print(f"text analysis completed: {len(text_contents) - failed}/{len(chunks)} successful")
             
            # Embed text
            print("Embedding textual content")
            text_embedding_start = time.time()
            text_docs = [
                f"{text_contents[i]['summary']} keywords: {', '.join(text_contents[i]['keywords'])}"
                for i in range(len(text_contents))
            ]
            try:
                text_vecs = await embed_many(text_docs, kind="text_embedding")
                for i, emb in enumerate(text_vecs):
                    text_contents[i]["embedding"] = [emb]
                print(f"text embedding completed: {len(text_vecs)} vectors")
            except Exception as e:
                print(f"error: text embedding generation failed: {e}")
                raise
            text_embedding_duration = time.time() - text_embedding_start
            add_stage_duration("text_embedding", text_embedding_duration)
            
            return text_contents
        
        # ============================================================
        # define image processing function (analyze + embed)
        # ============================================================
        async def process_image_content():
            """process all image content: analyze + embed"""
            print(f"Analyzing visual content ({len(images_b64)} images)...")
            image_contents: Dict[int, Dict[str, Any]] = {}
            
            if not images_b64:
                print("skip image analysis: no images extracted from PDF")
                return image_contents
            
            image_analysis_start = time.time()
            img_tasks = [analyze_content_mineru(b64, modality="image", context=contexts[i], caption=captions[i]) for i, b64 in enumerate(images_b64)]
            
            try:
                img_results = await asyncio.gather(*img_tasks, return_exceptions=True)
            except Exception as e:
                print(f"error: image analysis batch processing failed: {e}")
                raise
            image_analysis_duration = time.time() - image_analysis_start
            add_stage_duration("image_analysis", image_analysis_duration)
            
            img_failed = 0
            for i, r in enumerate(img_results):
                if isinstance(r, Exception) or not isinstance(r, dict) or "summary" not in r:
                    img_failed += 1
                    image_contents[i] = {"summary": "image analysis failed", "keywords": ["unknown"], "tags": ["error"]}
                else:
                    image_contents[i] = r
            if img_failed:
                print(f"warning: {img_failed}/{len(images_b64)} image analysis failed, using default values")
            if img_failed == len(images_b64):
                print("warning: all image analysis failed! will skip image notes addition.")
                return {}
            else:
                print(f"image analysis completed: {len(image_contents) - img_failed}/{len(images_b64)} successful")
            
            # Embed images (if any valid)
            if image_contents:
                print("Embedding visual content")
                image_embedding_start = time.time()
                img_docs = [
                    f"{image_contents[i]['summary']} keywords: {', '.join(image_contents[i]['keywords'])}"
                    for i in range(len(image_contents))
                ]
                try:
                    img_vecs = await embed_many(img_docs, kind="image_embedding")
                    for i, emb in enumerate(img_vecs):
                        image_contents[i]["embedding"] = [emb]
                    print(f"image embedding completed: {len(img_vecs)} vectors")
                except Exception as e:
                    print(f"error: image embedding generation failed: {e}")
                    raise
                image_embedding_duration = time.time() - image_embedding_start
                add_stage_duration("image_embedding", image_embedding_duration)
            
            return image_contents
        
        if PARALLEL_ANALYSIS:
            print("🚀 use parallel analysis mode (text and image processing simultaneously)")
            parallel_start = time.time()
            text_contents, image_contents = await asyncio.gather(
                process_text_content(),
                process_image_content()
            )
            parallel_duration = time.time() - parallel_start
            print(f"✓ parallel processing completed, total duration: {parallel_duration:.2f} seconds")
        else:
            print("📋 use serial analysis mode (text and image processing sequentially)")
            text_contents = await process_text_content()
            image_contents = await process_image_content()
         
        # --- Add notes ---
        # filter out the notes with "No meaningful information" in the summary
        to_remove = [i for i, text_content in text_contents.items() if "No meaningful information" in text_content["summary"]]
        print(f"filter out {len(to_remove)} text notes containing 'No meaningful information'")
        
        print("Adding textual notes to memory system")
        ok, bad = 0, 0
        for i, ch in enumerate(chunks):
            if i in to_remove:
                continue
            try:
                ms.add_note(
                    content=ch,
                    context=text_contents[i]["summary"],
                    keywords=text_contents[i]["keywords"],
                    tags=text_contents[i]["tags"],
                    category="text",
                    pre_embeddings=text_contents[i]["embedding"],
                )
                ok += 1
            except Exception as e:
                bad += 1
                if bad <= 3:
                    import traceback; traceback.print_exc()
        print(f"text notes added successfully: {ok} successful, {bad} failed")
        if ok == 0:
            raise RuntimeError("❌ serious error: failed to add any text note!")
         
        print("Adding visual notes to memory system")
        ok_i, bad_i = 0, 0
        for i, b64 in enumerate(images_b64):
            if i not in image_contents:
                continue
            try:
                ms.add_note(
                    content=b64,
                    context=image_contents[i]["summary"],
                    keywords=image_contents[i]["keywords"],
                    tags=image_contents[i]["tags"],
                    text_content = image_contents[i]["text_content"],
                    category="image",
                    visual=True,
                    pre_embeddings=image_contents[i]["embedding"],
                )
                ok_i += 1
            except Exception:
                bad_i += 1
        print(f"image notes added successfully: {ok_i} successful, {bad_i} failed")
         
        # --- Initialize local links for text notes only ---
        text_count = ok
        for i, note in enumerate(list(ms.memories.values())[:text_count]):
            for d in range(-window_size, window_size + 1):
                if d == 0:
                    continue
                j = i + d
                if 0 <= j < text_count:
                    note.links.append(j)
            note.links = list(set(note.links))
        print("links initialized")
        
        
        
        # save the memory system before evolving
        print(f"\nsave Memory System: {name}")
        print(f"  prepare to save {len(ms.memories)} memories...")
        ms.save_memory_system(name+"_iter_0")
         
        # --- Evolve (optional) ---
        for it in range(evolve_iters):
            print(f"evolving memory system: iteration {it + 1}")
            evolution_start = time.time()
            _ = await ms.process_memory_all()
            evolution_duration = time.time() - evolution_start
            add_stage_duration("memory_evolution", evolution_duration)
            
            print(f"re-embedding after evolution iteration {it + 1}")
            re_embedding_start = time.time()
            meta = [n.context + " keywords: " + ", ".join(n.keywords) for n in ms.memories.values()]
            pre = await embed_many(meta, kind="re_embedding")
            ms.reset_retriever()
            ms.retriever.add_documents(meta, pre)
            re_embedding_duration = time.time() - re_embedding_start
            add_stage_duration("re_embedding", re_embedding_duration)
            
            # exclude itself in links
            for i, note in enumerate(list(ms.memories.values())):
                note.links = [j for j in note.links if j != i]
                
            # save the memory system after each iteration
            print(f"\nsave Memory System: {name}")
            print(f"  prepare to save {len(ms.memories)} memories...")
            ms.save_memory_system(name+"_iter_"+str(it+1))

    if not flag_dataset:
        for p in pdf_paths:
            try:
                os.remove(p)
            except Exception:
                pass

    print("\n" + "=" * 80)
    print("Memory System built successfully!")
    print(f"   - name: {name}")
    print(f"   - total memories: {len(ms.memories)}")
    print(f"   - save path: {MEMORY_SYSTEMS_DIR}/{name}/")
    print("=" * 80 + "\n")
    return ms


def search_memory(memory_system: AgenticMemorySystem, query_or_keywords, k: int = 10, modality: str = "all", method: str = "semantic", top_k_text: int = 5, top_k_image: int = 5):
    if method == "semantic":
        _, notes = memory_system.find_related_notes_original(str(query_or_keywords), k=k, modality=modality)
        return notes
    elif method == "keywords":
        return memory_system.search_keyword(keywords=list(query_or_keywords), modality=modality, top_k_text=top_k_text, top_k_image=top_k_image)
    else:
        _, notes = memory_system.find_related_notes_original(str(query_or_keywords), k=k, modality=modality)
        return notes
