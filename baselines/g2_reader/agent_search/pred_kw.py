import os
import sys
import json
import re
import time
import tiktoken
from typing import List, Tuple
from openai import OpenAI
from tqdm import tqdm
import multiprocessing as mp
from pathlib import Path
import asyncio
from prebuild.amem_new import construct_memory, search_memory
from prebuild.usage_tracker import add_chat_usage, reset_usage, get_and_reset

current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.dirname(current_dir))

from agent_search.logging import setup_logging
from agent_search.counter import ProcessSafeCounter
from config.config import LLM_BASE_URL, LLM_API_KEY
from config.config import (
    reasoner_prompt_with_trajectory,
    zero_shot_rag_prompt, 
    decomposer_dag_prompt,
    reasoner_dag_prompt,
    evidence_check_prompt,
    decomposer_dag_after_check_prompt,
)

class DAGPred:
    def __init__(self, args):
        self.args = args
        self.reasoner_prompt_with_trajectory = reasoner_prompt_with_trajectory
        self.zero_shot_rag_prompt = zero_shot_rag_prompt
        self.decomposer_dag_prompt = decomposer_dag_prompt
        self.reasoner_dag_prompt = reasoner_dag_prompt
        self.evidence_checker_prompt = evidence_check_prompt
        self.dag_decomposer_after_check_prompt = decomposer_dag_after_check_prompt
        self.token_count = 0
        self.logger = None
        self.counter = ProcessSafeCounter(args.save_dir)
        self._memory_cache = {}

    def query_llm(self, prompt, model, tokenizer, client=None, temperature=0.0, max_new_tokens=128, stop=None, images: List[str] = None, track_usage: bool = True):
        max_len = 4096
        input_ids = tokenizer.encode(prompt)
        if len(input_ids) > max_len:
            input_ids = input_ids[:max_len//2] + input_ids[-max_len//2:]
            prompt = tokenizer.decode(input_ids)
        tries = 0
        model_name = model
        while tries < 5:
            tries += 1
            try:
                if images and images:
                    content = [
                        {"type": "text", "text": prompt}
                    ]
                    for img_base64 in images:
                        content.append({
                            "type": "image_url",
                            "image_url": {
                                "url": f"data:image/jpeg;base64,{img_base64}"
                            }
                        })
                    messages = [{"role": "user", "content": content}]
                else:
                    messages = [{"role": "user", "content": prompt}]
                
                completion = client.chat.completions.create(
                    model=model_name,
                    messages=messages,
                    temperature=temperature,
                    max_tokens=max_new_tokens,
                )
                if track_usage:
                    try:
                        add_chat_usage(
                            getattr(completion, "usage", None),
                            meta={
                                "source": "agent_search.pred.query_llm",
                                "model": model_name,
                                "has_images": bool(images),
                                "temperature": temperature,
                                "max_tokens": max_new_tokens,
                            }
                        )
                    except Exception:
                        pass
                return completion.choices[0].message.content
            except KeyboardInterrupt as e:
                raise e
            except Exception as e:
                print(f"Error Occurs: \"{str(e)}\"        Retry ...")
                time.sleep(1)
        else:
            print("Max tries. Failed.")
            return ''

    def get_token_length(self, text):
        tokenizer = tiktoken.get_encoding('cl100k_base')
        return len(tokenizer.encode(text))

    def extract_output(self, response):
        output = re.findall(r'<output>(.*?)</output>', response, re.DOTALL)
        if len(output) == 0:
            return None
        return output[0].strip()
    
    def truncate_context(self, context, max_tokens=4096):
        tokens = tiktoken.get_encoding('cl100k_base').encode(context)
        if len(tokens) > max_tokens:
            tokens = tokens[:max_tokens]
            context = tiktoken.get_encoding('cl100k_base').decode(tokens)
        return context  
      
    def retriever(self,query: str, link: str, top_k: int = 5, cache_flag: bool = False, save_retrieval: bool = False, item: dict = None) -> Tuple[str, List[str]]:
        """Retriever - directly calls local memory construction and search, returns (texts, images)"""
        try:
            if not hasattr(self, "_memory_cache"):
                self._memory_cache = {}
            if cache_flag and link in self._memory_cache:
                memory_system = self._memory_cache[link]
            else:
                memory_system = asyncio.run(construct_memory(pdf_path=link))
                if cache_flag:
                    self._memory_cache[link] = memory_system

            notes = search_memory(memory_system, query, k=top_k, modality="all")

            texts = ""
            images: List[str] = []
            for note in notes:
                visual = getattr(note, "visual", False) if not isinstance(note, dict) else note.get("visual", False)
                content = note.content if hasattr(note, "content") else (note.get("content") if isinstance(note, dict) else str(note))
                if visual:
                    images.append(content)
                else:
                    if content:
                        texts += content + "\n"

            text_notes: List[str] = []
            for note in notes:
                visual = getattr(note, "visual", False) if not isinstance(note, dict) else note.get("visual", False)
                content = note.content if hasattr(note, "content") else (note.get("content") if isinstance(note, dict) else str(note))
                if not visual and content and content.strip():
                    text_notes.append(content.strip())

            formatted_texts = ""
            for i, text_chunk in enumerate(text_notes):
                formatted_texts += f"\nRelated Memory [{i}]\n {text_chunk}\n\n"

            if save_retrieval and item:
                self._save_simple_retrieval_results(query, link, notes, text_notes, images, item)

            return formatted_texts.strip(), images

        except Exception as e:
            print(f"Retriever error: {e}")
            return "", []

    def _select_query_keywords(self, question: str, model: str, tokenizer, client, max_k: int = 8) -> List[str]:
        """
        Lightweight keyword routing agent: extracts keywords/phrases from question for retrieval.
        Falls back to simple tokenization with high-frequency words on failure.
        """
        prompt = (
            "Identify the essential keywords and constraints from the user's query for a keyword-based search engine (BM25). "
            "Your goal is to ensure high recall by including all restrictive terms.\n"
            
            "**Guidelines:**\n"
            "1. **Entities & Proper Nouns:** Extract names of people, organizations, locations, and specific works (e.g., 'Karen David', 'Xinhai Revolution').\n"
            "2. **Time & Numbers:** ALWAYS extract specific years, dates, or time periods (e.g., '2007', '2008', '19th century'). These are critical filters.\n"
            "3. **Document References (CRITICAL):** ALWAYS extract specific identifiers for figures, tables, or sections EXACTLY as they appear (e.g., 'Figure 20', 'Fig.5', 'Table 3', 'Section 4.1'). Do not split them; treat 'Fig. 5' as a single entity.\n"
            "4. **Specific Nouns:** Include key subject nouns (e.g., 'movies', 'roles', 'relativity', 'mechanism').\n"
            "5. **Exclusion:** Remove purely functional words, verbs, and broad interrogatives (e.g., 'what', 'how', 'explain', 'difference', 'context').\n"
            
            "Output only a standard JSON array of strings."
            
            "\n\n**Examples:**\n"
            "Input: What are the differences between the political views of Liang Qichao and Kang Youwei?\n"
            "Output: [\"Liang Qichao\", \"Kang Youwei\", \"political views\"]\n\n"
            
            "Input: Explain the historical impact of the 1911 Xinhai Revolution.\n"
            "Output: [\"1911\", \"Xinhai Revolution\", \"historical impact\"]\n\n"
            
            "Input: What is the main idea of Einstein's 1905 relativity papers?\n"
            "Output: [\"Einstein\", \"1905\", \"relativity\"]\n\n"
            
            "Input: What movies and roles did Karen David play in the years 2007 and 2008?\n"
            "Output: [\"Karen David\", \"movies\", \"roles\", \"2007\", \"2008\"]\n\n"
            
            "Input: Look at Figure 20 and Fig.5 to analyze the accuracy trend.\n"
            "Output: [\"Figure 20\", \"Fig.5\", \"accuracy trend\"]\n\n"

            "Input: Compare the results in Table 3 with Section 4.2.\n"
            "Output: [\"Table 3\", \"Section 4.2\", \"results\"]\n\n"

            f"Question: {question}\n\n"
            "Output only the JSON array."
        )
        try:
            resp = self.query_llm(prompt, model, tokenizer, client, temperature=0.0, max_new_tokens=128, track_usage=False)
            import json, re
            try:
                kws = json.loads(resp)
                if isinstance(kws, list):
                    clean = []
                    for w in kws:
                        w = (w or "").strip()
                        if not w:
                            continue
                        if len(w) < 2:
                            continue
                        clean.append(w)
                    return clean[:max_k] if clean else []
            except Exception:
                pass
            m = re.search(r"\[[\s\S]*\]", resp)
            if m:
                try:
                    kws = json.loads(m.group(0))
                    if isinstance(kws, list):
                        clean = [str(w).strip() for w in kws if isinstance(w, (str, int)) and len(str(w).strip()) >= 2]
                        return clean[:max_k] if clean else []
                except Exception:
                    pass
        except Exception:
            pass

        import re
        tokens = re.findall(r"\w+", question.lower())
        stop = {"the","and","of","to","in","on","for","is","are","with","a","an"}
        freq = {}
        for t in tokens:
            if t in stop or len(t) < 2:
                continue
            freq[t] = freq.get(t, 0) + 1
        ranked = sorted(freq.items(), key=lambda x: -x[1])
        return [w for w,_ in ranked[:max_k]]

    from typing import List, Union

    def _phrase_rank_texts_optimized(self, docs: List[str], query_terms: List[str], k: int = 5, return_indices: bool = False) -> Union[List[str], List[int]]:
        """
        Phrase matching scoring (optimized version).
        
        Args:
            docs: List of text documents
            query_terms: List of query terms
            k: Return top K results
            return_indices: 
                - False (default): Return list of matched text strings
                - True: Return list of document indices in docs
                
        Logic: Prioritize matching more diverse keywords.
        """
        import re

        valid_terms = [qt.strip() for qt in (query_terms or []) if qt and qt.strip()]
        valid_terms.sort(key=len, reverse=True) 
        
        if not valid_terms:
            return []

        clean_terms = [re.escape(t) for t in valid_terms]
        pattern_str = r"\b(?:" + "|".join(clean_terms) + r")\b"
        regex = re.compile(pattern_str, re.IGNORECASE)

        scores = []
        for i, doc in enumerate(docs):
            if not doc: continue
            
            matches = regex.findall(doc)
            
            if not matches:
                continue

            unique_hits = {m.lower() for m in matches}
            count_unique = len(unique_hits)
            count_total = len(matches)
            score = count_unique * 1.0 + count_total * 0.01
            
            scores.append((i, score))

        scores = [t for t in scores if t[1] > 0]
        if not scores:
            return [] if return_indices else []
        scores.sort(key=lambda x: x[1], reverse=True)
        
        top_idx = [i for i, _ in scores[:k]]
        
        if return_indices:
            return top_idx
        else:
            return [docs[i] for i in top_idx]

    def _save_simple_retrieval_results(self, query: str, link: str, notes, text_notes, images, item: dict):
        """Save simple retrieval method results"""
        try:
            unique_data_id = str(item.get('_id', 'unknown'))
            data_folder = Path(self.args.save_dir) / "logs" / f"data_{unique_data_id}"
            data_folder.mkdir(parents=True, exist_ok=True)
            
            retrieval_file = data_folder / "retrieval_results.jsonl"
            
            retrieval_results = []
            for idx, note in enumerate(notes):
                visual = getattr(note, "visual", False) if not isinstance(note, dict) else note.get("visual", False)
                content = note.content if hasattr(note, "content") else (note.get("content") if isinstance(note, dict) else str(note))
                
                text_content = getattr(note, "text_content", "") if hasattr(note, "text_content") else ""
                
                retrieval_results.append({
                    "index": idx,
                    "type": "image" if visual else "text",
                    "content": content if not visual else "[IMAGE_BASE64]",
                    "text_content": text_content if visual else content,
                    "visual": visual
                })
            
            retrieval_record = {
                "q_id": unique_data_id,
                "query": query,
                "timestamp": time.time(),
                "retrieval_method": "simple_semantic",
                "results": {
                    "count": len(retrieval_results),
                    "text_count": len(text_notes),
                    "image_count": len(images),
                    "details": retrieval_results
                }
            }
            
            with open(retrieval_file, "a", encoding="utf-8") as f:
                f.write(json.dumps(retrieval_record, ensure_ascii=False) + "\n")
            
            if self.logger:
                self.logger.info(f"Retrieval results saved: {retrieval_file} (total: {len(retrieval_results)}, text: {len(text_notes)}, images: {len(images)})")
        
        except Exception as e:
            if self.logger:
                self.logger.warning(f"Error saving retrieval results: {e}")
    
    def _save_retrieval_results(self, query: str, link: str, keywords: List[str], semantic_notes, all_text_docs, all_image_docs, bm25_top_text, bm25_top_index, merged_texts, images, item: dict, memory_system):
        """保存检索结果，包含 Memory content 和对应的 abstract/metadata"""
        try:
            unique_data_id = str(item.get('_id', 'unknown'))
            data_folder = Path(self.args.save_dir) / "logs" / f"data_{unique_data_id}"
            data_folder.mkdir(parents=True, exist_ok=True)
            
            retrieval_file = data_folder / "retrieval_results.jsonl"
            
            # 收集semantic检索的结果
            semantic_results = []
            for idx, note in enumerate(semantic_notes):
                visual = getattr(note, "visual", False) if not isinstance(note, dict) else note.get("visual", False)
                content = note.content if hasattr(note, "content") else (note.get("content") if isinstance(note, dict) else str(note))
                
                # 获取图片的文字描述（OCR或caption）
                text_content = getattr(note, "text_content", "") if hasattr(note, "text_content") else ""
                       
                semantic_results.append({
                    "index": idx,
                    "type": "image" if visual else "text",
                    "content": content if not visual else "[IMAGE_BASE64]",
                    "text_content": text_content if visual else content,  # 图片显示OCRtext，text显示原文
                    "visual": visual
                })
            
            # 收集 BM25 检索的结果（text）
            bm25_text_results = []
            for idx, text in enumerate(bm25_top_text):
                bm25_text_results.append({
                    "index": idx,
                    "type": "text",
                    "content": text,
                    "source": "bm25_text"
                })
            
            # 收集 BM25 检索的结果（images）
            bm25_image_results = []
            for idx in bm25_top_index:
                if idx < len(all_image_docs):
                    bm25_image_results.append({
                        "index": idx,
                        "type": "image",
                        "text_content": all_image_docs[idx],
                        "source": "bm25_image"
                    })
            
            # 构建完整的检索记录
            retrieval_record = {
                "q_id": unique_data_id,
                "query": query,
                "keywords": keywords,
                "timestamp": time.time(),
                "semantic_retrieval": {
                    "count": len(semantic_results),
                    "results": semantic_results
                },
                "bm25_retrieval": {
                    "text_count": len(bm25_text_results),
                    "image_count": len(bm25_image_results),
                    "text_results": bm25_text_results,
                    "image_results": bm25_image_results
                }
            }
            
            # 追加写入文件
            with open(retrieval_file, "a", encoding="utf-8") as f:
                f.write(json.dumps(retrieval_record, ensure_ascii=False) + "\n")
            
            if self.logger:
                self.logger.info(f" 检索结果saved: {retrieval_file} (semantic: {len(semantic_results)}, BM25text: {len(bm25_text_results)}, BM25images: {len(bm25_image_results)})")
        
        except Exception as e:
            if self.logger:
                self.logger.warning(f" Error saving retrieval results: {e}")
    
    def retriever_split_sem_bm25(self, query: str, link: str, keywords: List[str], k_sem: int = 3, k_bm25: int = 3, cache_flag: bool = True, save_retrieval: bool = True, item: dict = None) -> Tuple[str, List[str]]:
        """
        并行进行：
        - semantic检索 k_sem
        - BM25 关键词检索 k_bm25
        合并text块为 Related Memory 格式，images沿用semantic检索得到的。
        """
        try:
            # 构建或缓存 Memory System
            if not hasattr(self, "_memory_cache"):
                self._memory_cache = {}
            if cache_flag and link in self._memory_cache:
                memory_system = self._memory_cache[link]
            else:
                memory_system = asyncio.run(construct_memory(pdf_path=link))
                if cache_flag:
                    self._memory_cache[link] = memory_system

            # semantic检索（限制为 k_sem）
            semantic_notes = search_memory(memory_system, query, k=max(1, int(k_sem)), modality="all")

            # 拆分出text与images
            semantic_texts: List[str] = []
            images: List[str] = []
            for note in semantic_notes:
                visual = getattr(note, "visual", False) if not isinstance(note, dict) else note.get("visual", False)
                content = note.content if hasattr(note, "content") else (note.get("content") if isinstance(note, dict) else str(note))
                if visual:
                    if content:
                        images.append(content)
                else:
                    if content and content.strip():
                        semantic_texts.append(content.strip())

            # BM25 关键词检索：对整个 memory 的text chunk 做打分（限制为 k_bm25）
            all_text_docs: List[str] = []
            all_image_docs: List[str] = []
            all_image = []
            try:
                for n in getattr(memory_system, "memories", {}).values():
                    v = getattr(n, "visual", False)
                    c = getattr(n, "content", None)
                    d = getattr(n, "text_content", None)
                    if not v and isinstance(c, str) and c.strip():
                        all_text_docs.append(c.strip())
                    if v and isinstance(d, str) and d.strip():
                        all_image_docs.append(d.strip())
                        all_image.append(c)
            except Exception:
                pass

            #TODO
            if k_bm25 > 0:
                bm25_top_text = self._phrase_rank_texts_optimized(all_text_docs, keywords, k=int(k_bm25))
            else:
                bm25_top_text = []

            if k_bm25 > 0:
                bm25_top_index = self._phrase_rank_texts_optimized(all_image_docs, keywords, k=int(k_bm25),return_indices=True)
                images  += [all_image[i] for i in bm25_top_index]
            else:
                pass

            
            
            # 合并 & 去重
            merged_texts = []
            seen = set()
            for t in (semantic_texts + bm25_top_text):
                if t not in seen:
                    merged_texts.append(t)
                    seen.add(t)

            # 格式化为 Related Memory
            formatted_texts = ""
            for i, text_chunk in enumerate(merged_texts):
                formatted_texts += f"\nRelated Memory [{i}]\n {text_chunk}\n\n"

            # 保存检索结果到文件
            if save_retrieval and item:
                self._save_retrieval_results(
                    query=query,
                    link=link,
                    keywords=keywords,
                    semantic_notes=semantic_notes,
                    all_text_docs=all_text_docs,
                    all_image_docs=all_image_docs,
                    bm25_top_text=bm25_top_text,
                    bm25_top_index=bm25_top_index if k_bm25 > 0 else [],
                    merged_texts=merged_texts,
                    images=images,
                    item=item,
                    memory_system=memory_system
                )

            return formatted_texts.strip(), images

        except Exception as e:
            print(f"Retriever (split sem+BM25) error: {e}")
            return "", []
    
    def reasoner_with_trajectory(self, related_contexts, query, trajectorys, model, tokenizer, client, item, top_k=10,  images: List[str] = None):
        reasoner_input = self.reasoner_prompt_with_trajectory.replace("$DOC$", related_contexts).replace("$Q$", query).replace("$TRA$", trajectorys)

        response = self.query_llm(reasoner_input, model, tokenizer, client, temperature=0.0, max_new_tokens=2048,images=images)

        return response, reasoner_input

    def reasoner_dag_node(self, related_contexts, query, model, tokenizer, client, images: List[str] = None):
        reasoner_input = self.reasoner_dag_prompt.replace("$DOC$", related_contexts).replace("$Q$", query)
        response = self.query_llm(reasoner_input, model, tokenizer, client, temperature=0.0, max_new_tokens=2048, images=images)
        return response

    def save_model_responses_to_folders(self, item, response_list, save_dir, model_name, data_id, run_idx: str = None,task: str = None,content: str = None, adjust_round: int = 0):
        """将每个模型的回答保存到对应的文件夹中，类似于main的保存方式，支持Adjustment round区分"""
        # 使用唯一 _id 作为 data_id，确保并行无冲突
        unique_data_id = str(item.get('_id', data_id))  # 回退到传入的 data_id 如果无 _id
        
        # 为每个数据ID创建子文件夹
        data_folder = Path(save_dir) / "logs" / f"data_{unique_data_id}"
        data_folder.mkdir(parents=True, exist_ok=True)
        
        # 构建 run_folder：注入 adjust_round 以区分Round
        if run_idx:
            # 如: round_1_node_root
            prefixed_run_idx = f"round_{adjust_round}_{run_idx}" if adjust_round > 0 else run_idx
            run_folder = data_folder / prefixed_run_idx
        else:
            # 默认 root，无 run_idx 时用 round_0_root
            prefixed_run_idx = f"round_{adjust_round}_root" if adjust_round > 0 else "root"
            run_folder = data_folder / prefixed_run_idx
        
        run_folder.mkdir(exist_ok=True)
        save_path = run_folder
        
        # 为每个响应保存单独的JSON文件
        for idx, response in enumerate(response_list):
            response_file = save_path / f"model_response_{idx + 1}.json"
            response_data = {
                "data_id": unique_data_id,
                "question": task if task else item['question'],
                "content": content if content else "",
                "model": model_name,
                "response": response,
                "extracted_pred": self.extract_output(response),
                "adjust_round": adjust_round  # 新增: 记录Round，便于追踪
            }
            with open(response_file, 'w', encoding='utf-8') as f:
                json.dump(response_data, f, ensure_ascii=False, indent=2)
        
        if self.logger:
            self.logger.info(f"saved {len(response_list)} model responses to folder: {save_path} (Unique ID: {unique_data_id}, Round: {adjust_round}, run_idx: {prefixed_run_idx})")

    def dag_decomposer(self, context, question, model, tokenizer, client, images: List[str] = None, max_depth=4, max_nodes=12, max_retries=5, max_rounds=3, round_index=1):        
        """
        生成分层 DAG 结构（失败时重新调用 dag_decomposer）
        """
        import re
        import json

        self.logger.info(f" [Round {round_index}/{max_rounds}] Starting DAG decomposition...")

        input_prompt = self.decomposer_dag_prompt.replace("$DOC$", context).replace("$Q$", question)

        for attempt in range(max_retries):
            self.logger.info(f" Attempting to generate DAG（ {attempt + 1}/{max_retries} 次）...")
            try:
                response = self.query_llm(
                    input_prompt, model, tokenizer, client,
                    temperature=0.5, max_new_tokens=1000, images=images
                )
                self.logger.info(f" DAG decomposition response generated, length: {len(response)} 字符")
                # 提取 <dag> 标签包围的 JSON
                match = re.search(r'<dag>(.*?)</dag>', response, re.DOTALL)
                if not match:
                    self.logger.warning(" DAG tag not found, retrying...")
                    continue

                dag_str = match.group(1).strip()
                dag = json.loads(dag_str)
                self.logger.info(f" DAG JSON extracted successfully, length: {len(dag_str)} 字符")

                # 验证 DAG 合法性
                if self._validate_dag(dag, max_depth, max_nodes):
                    depth = self._compute_depth(dag)
                    self.logger.info(f" Generated valid DAG: {len(dag['nodes'])} nodes，depth {depth}")
                    return dag
                else:
                    self.logger.warning(" DAG validation failed, retrying...")

            except json.JSONDecodeError as e:
                self.logger.warning(f" JSON parsing failed: {e}，retrying...")
            except Exception as e:
                self.logger.warning(f" Error during DAG parsing: {e}，retrying...")

        # 如果当前轮所有尝试都失败
        if round_index < max_rounds:
            self.logger.warning(f" Round {round_index} All attempts failed, starting next round of DAG retry...")
            return self.dag_decomposer(
                context, question, model, tokenizer, client, images=images,
                max_depth=max_depth, max_nodes=max_nodes,
                max_retries=max_retries, max_rounds=max_rounds,
                round_index=round_index + 1
            )
        else:
            self.logger.error(" All DAG generation attempts failed, giving up。")
            raise RuntimeError("DAG 解析多轮失败，无法Generated valid DAG。")


    def dag_decomposer_after_check(self, context, question, evidence, gaps, model, tokenizer, client, images: List[str] = None, old_dag: dict = None, max_depth=4, max_nodes=12, max_retries=5, max_rounds=3, round_index=1):
        """
        在证据检查后调整 DAG 结构（类似 dag_decomposer，但注入当前证据和 gaps）
        """
        import re
        import json

        self.logger.info(f" [Adjustment Round {round_index}/{max_rounds}] Starting DAG adjustment...")

        # 构建证据字符串
        evidence_str = '\n'.join([f"Sub-task: {qa['q']}\nAnswer: {qa['a']}\n" for qa in evidence])
        gaps_str = '; '.join(gaps) if gaps else "No specific gaps identified; refine for completeness."

        # 构建 old_dag 字符串
        old_dag_str = json.dumps(old_dag, ensure_ascii=False) if old_dag else "{}"

        input_prompt = self.dag_decomposer_after_check_prompt.replace("$DOC$", context).replace("$Q$", question).replace("$EVIDENCE$", evidence_str).replace("$GAPS$", gaps_str).replace("$OLD_DAG$", old_dag_str)

        if old_dag:
            self.logger.info(f" Injecting old DAG: {len(old_dag.get('nodes', []))} nodes")

        for attempt in range(max_retries):
            self.logger.info(f" Attempting to adjust DAG（ {attempt + 1}/{max_retries} 次）...")
            try:
                response = self.query_llm(
                    input_prompt, model, tokenizer, client,
                    temperature=0.5, max_new_tokens=1000, images=images
                )
                self.logger.info(f" DAG adjustment response generated, length: {len(response)} 字符")

                # 提取 <dag> 标签包围的 JSON
                match = re.search(r'<dag>(.*?)</dag>', response, re.DOTALL)
                if not match:
                    self.logger.warning(" DAG tag not found, retrying...")
                    continue

                dag_str = match.group(1).strip()
                dag = json.loads(dag_str)
                self.logger.info(f" Adjusted DAG JSON extracted successfully, length: {len(dag_str)} 字符")

                # 验证 DAG 合法性
                if self._validate_dag(dag, max_depth, max_nodes):
                    depth = self._compute_depth(dag)
                    self.logger.info(f" Generated valid adjusted DAG: {len(dag['nodes'])} nodes，depth {depth}")
                    return dag
                else:
                    self.logger.warning(" Adjusted DAG validation failed, retrying...")

            except json.JSONDecodeError as e:
                self.logger.warning(f" JSON parsing failed: {e}，retrying...")
            except Exception as e:
                self.logger.warning(f" Adjusted Error during DAG parsing: {e}，retrying...")

        # 如果当前轮所有尝试都失败
        if round_index < max_rounds:
            self.logger.warning(f" Adjustment Round {round_index} All attempts failed, starting next round of DAG adjustment...")
            return self.dag_decomposer_after_check(
                context, question, evidence, gaps, model, tokenizer, client, images=images, old_dag=old_dag,
                max_depth=max_depth, max_nodes=max_nodes,
                max_retries=max_retries, max_rounds=max_rounds,
                round_index=round_index + 1
            )
        else:
            self.logger.error(" All adjusted DAG generation attempts failed, giving up。")
            raise RuntimeError("Adjusted DAG 解析多轮失败，无法Generated valid DAG。")


    def _validate_dag(self, dag, max_depth, max_nodes):
        """验证 DAG：无环、depth、nodes数"""
        if len(dag.get('nodes', [])) > max_nodes:
            self.logger.warning(f"nodes数超限: {len(dag['nodes'])} > {max_nodes}")
            return False
        # 简单无环检查：使用 topo sort 尝试
        from collections import deque, defaultdict
        graph = defaultdict(list)
        indegree = {}
        for node in dag['nodes']:
            nid = node['id']
            indegree[nid] = 0
            for child in node.get('children', []):
                graph[nid].append(child)
                indegree[child] = indegree.get(child, 0) + 1
        # Kahn 算法检查
        queue = deque([nid for nid, deg in indegree.items() if deg == 0])
        visited = 0
        while queue:
            node = queue.popleft()
            visited += 1
            for neighbor in graph[node]:
                indegree[neighbor] -= 1
                if indegree[neighbor] == 0:
                    queue.append(neighbor)
        if visited != len(dag['nodes']):
            self.logger.warning("DAG has cycles")
            return False
        return True

    def _compute_depth(self, dag):
        """计算 DAG depth"""
        nodes_dict = {n['id']: n for n in dag['nodes']}
        def get_depth(node_id):
            node = nodes_dict.get(node_id, {})
            if not node.get('children'):
                return 1
            return 1 + max(get_depth(child) for child in node['children'] if child in nodes_dict)
        root = next((n for n in dag['nodes'] if n['id'] == 'root'), None)
        return get_depth('root') if root else 1

    def _build_supplementary_info_from_nodes(self, node_results):
        """从nodes结果构建补充信息"""
        if not node_results:
            return "无子nodes补充信息。"
        parts = []
        for idx, result in enumerate(node_results.values()):
            parts.append(f"=== 子nodes {idx+1} ===")
            parts.append(f"子回答: {result[:500]}...")  # 截断
        return '\n\n'.join(parts)

    def _execute_dag_node(self, node_id, nodes_dict, model, tokenizer, client, args, idx, item, link, adjust_round: int = 0):
        """执行单个DAGnodes，添加详细日志"""
        node = nodes_dict[node_id]
        task = node['task']
        self.logger.info(f" Executing DAG nodes {node_id}: {task[:50]}... (总nodes: {len(nodes_dict)}, Adjustment round: {adjust_round})")
        
        # 检索
        self.logger.info(f" 开始检索nodes {node_id}...")
        texts, images = self.retriever(task, link, top_k=args.top_k, cache_flag=True, save_retrieval=True, item=item)
        context = texts  # 使用格式化的text作为 context
        self.logger.info(f" nodes {node_id} 检索完成: Number of text blocks {len(texts.split('Related Memory')) if texts else 0}, Image count {len(images)}")

        # 增强上下文：Injecting sub-results (递进)
        if node.get('children'):
            sub_results = [self.node_results.get((adjust_round, child), "") for child in node['children']]  # 修改: 用 (round, child) 作为 key 避免覆盖
            sub_results = [r for r in sub_results if r]  # 过滤空值
            if sub_results:
                sub_info = self._build_supplementary_info_from_nodes({k: v for k, v in zip(node['children'], sub_results)})
                context += f"\n\n=== 子nodes结果 (Round {adjust_round}) ===\n{sub_info}"
                self.logger.info(f" nodes {node_id} 注入 {len(sub_results)} 个子结果")
        
        # 推理
        self.logger.info(f" 开始推理nodes {node_id}...")
        # 直接使用响应，不再提取thinking标签

        response = self.reasoner_dag_node(context, task, model, tokenizer, client, images=images)
        
        if response is None:
            response = "No response"
        # 修改: 用 (round, node_id) 作为 key 存储结果，避免内存覆盖
        self.node_results[(adjust_round, node_id)] = response
        self.logger.info(f" nodes {node_id} 推理完成: {response[:100]}... (Round {adjust_round})")

        # 保存nodes响应：注入 adjust_round
        self.save_model_responses_to_folders(item, [response], self.args.save_dir, model, item['_id'], node_id, task=task, content=context, adjust_round=adjust_round)  
        
        return response

    def _check_evidence_sufficiency(self, question, main_context, nodes_dict, node_results, model, tokenizer, client, images: List[str] = None, adjust_round: int = 0):
        """证据检查：评估当前 QA 对和上下文是否sufficient"""
        self.logger.info(f" Starting evidence check (Round {adjust_round})...")
        
        # 构建证据字符串：从当前Round结果中取
        evidence_list = []
        for (rnd, nid), res in node_results.items():
            if rnd == adjust_round and nid in nodes_dict:  # 只取当前Round
                q = nodes_dict[nid]['task']
                evidence_list.append({'q': q, 'a': res})
        evidence_str = '\n'.join([f"Sub-task: {qa['q']}\nAnswer: {qa['a']}\n" for qa in evidence_list])  # 截断以防过长
        
        check_prompt = self.evidence_checker_prompt.replace("$Q$", question).replace("$DOC$", main_context).replace("$EVIDENCE$", evidence_str)
        
        response = self.query_llm(check_prompt, model, tokenizer, client, temperature=0.0, max_new_tokens=216, images=images)
        
        # 提取 <check> JSON
        match = re.search(r'<check>(.*?)</check>', response, re.DOTALL)
        if not match:
            self.logger.warning(f" 证据检查 (Round {adjust_round}) 未找到 <check> 标签，defaulting to insufficient")
            return False, []
        
        try:
            check_json = json.loads(match.group(1).strip())
            sufficient = check_json.get('sufficient', False)
            gaps = check_json.get('gaps', []) if 'gaps' in check_json else []
            self.logger.info(f" Evidence check result (Round {adjust_round}): sufficient={sufficient}, identified gaps={len(gaps)}")
            return sufficient, gaps
        except json.JSONDecodeError:
            self.logger.warning(f" 证据检查 JSON parsing failed (Round {adjust_round})，defaulting to insufficient")
            return False, []

    def get_pred_no_dag(self, data, args, fout, process_id=None, run_idx=0):
        """
        无DAG版本的RAGPrediction：直接检索+推理，不进行DAG分解
        - 保留关键词路由
        - 保留混合检索（semantic+BM25）
        - 保留所有日志和统计功能
        """
        # 设置进程专用日志
        if process_id is not None:
            self.logger = setup_logging(args.save_dir, data[0] if data else None)
        else:
            self.logger = setup_logging(args.save_dir)
        
        model = args.model
        tokenizer = tiktoken.encoding_for_model("gpt-4o-2024-08-06")
        
        URL = LLM_BASE_URL
        API_KEY = LLM_API_KEY
        client = OpenAI(base_url=URL, api_key=API_KEY)
        time_ls = []
        
        self.logger.info(f"Starting to process {len(data)} samples（No-DAG direct inference mode）")
        idx = 0

        for item in tqdm(data):
            # 每samples开始：重置 token 使用统计
            reset_usage()
            start_time = time.time()
            
            ############################## 主检索（与DAG版本相同）##############################
            question = item['question']
            id = item['_id']
            
            self.logger.info(f" Task ID: {id}, Starting initial retrieval...")
            
            # 关键词路由 agent
            try:
                selected_kws = self._select_query_keywords(question, model, tokenizer, client, max_k=8)
                self.logger.info(f" Task ID: {id}, Keyword routing results: {selected_kws}")
            except Exception as e:
                selected_kws = []
                self.logger.warning(f" Task ID: {id}, Keyword routing failed, falling back to empty: {e}")

            # 并行合并检索（semantic + BM25）
            texts, initial_images = self.retriever_split_sem_bm25(
                question, id, selected_kws, 
                k_sem=5, k_bm25=3, 
                cache_flag=True, 
                save_retrieval=True, 
                item=item
            )
            main_context = texts
            self.logger.info(f" Task ID: {id} Initial retrieval complete: Number of text blocks {len(texts.split('Related Memory')) if texts else 0}, Image count {len(initial_images)}")
            
            # 单轮直接推理
            reasoner_input = self.zero_shot_rag_prompt.replace("$DOC$", main_context).replace("$Q$", question)
            response = self.query_llm(
                        reasoner_input, model, tokenizer, client, 
                        temperature=0.2, max_new_tokens=2048, 
                        images=initial_images
                    )
            if response is None or response == "":
                response=f'No output'
            self.logger.info(f" Task ID: {id}, Direct inference complete")
            item['reasoner_input'] = reasoner_input
            item['initial_images'] = initial_images
            item['response'] = response
            item['pred'] = self.extract_output(response)
            end_time = time.time()
            process_time = end_time - start_time
            time_ls.append(process_time)
            item['process_time'] = process_time
            
            current_count = self.counter.increment()
            self.logger.info(f" Task ID: {id}, Prediction: {item['pred']}, Correct answer: {item['answer']},  Time: {process_time:.4f}s, Completed: {current_count}")
            
            # 保存响应
            self.save_model_responses_to_folders(
                item, [response], args.save_dir, model, 
                item['_id'], run_idx="no_dag_single",
                task=question, content=main_context
            )
            
            idx += 1

            # 写入样本级 token 使用统计
            usage_summary = get_and_reset()
            item['usage'] = usage_summary

            unique_data_id = str(item.get('_id', 'unknown'))
            data_folder = Path(args.save_dir) / "logs" / f"data_{unique_data_id}"
            data_folder.mkdir(parents=True, exist_ok=True)
            usage_file = data_folder / "usage_time.jsonl"

            usage_row = {
                "q_id": unique_data_id,
                "duration_total_sec": round(float(item.get("process_time", 0.0)), 6),
                "usage": usage_summary,
            }
            with open(usage_file, "a", encoding="utf-8") as fu:
                fu.write(json.dumps(usage_row, ensure_ascii=False) + "\n")
            if self.logger:
                self.logger.info(f" Usage written: {usage_file} -> {usage_row}")

            fout.write(json.dumps(item, ensure_ascii=False) + "\n")
            fout.flush()
        
        if time_ls:
            avg_time = sum(time_ls) / len(time_ls)
            print(f"Average time per sample: {avg_time:.4f} seconds")
        return time_ls

    def get_pred_dag(self, data, args, fout, process_id=None, run_idx=0):
        """
        DAG版本的RAGPrediction：一步预构建分层DAG，然后 topo 执行层层递进
        - 分解：使用 dag_decomposer 生成 JSON DAG (nodes + edges/children)
        - 执行：Topo sort 顺序，逐nodes检索/推理
        - 聚合：全图结果轨迹 → 主推理
        """
        # 设置进程专用日志
        if process_id is not None:
            self.logger = setup_logging(args.save_dir, data[0] if data else None)
        else:
            self.logger = setup_logging(args.save_dir)
        
        model = args.model

        tokenizer = tiktoken.encoding_for_model("gpt-4o-2024-08-06")
        
        URL = LLM_BASE_URL
        API_KEY = LLM_API_KEY

        client = OpenAI(base_url=URL, api_key=API_KEY)
        time_ls = []
        
        self.logger.info(f"Starting to process {len(data)} samples（DAG hierarchical mode）")
        idx = 0

        for item in tqdm(data):
            # 每samples开始：重置 token 使用统计
            reset_usage()

            start_time = time.time()
            
            ############################## DAG 初始nodes：主检索 ##############################
            question = item['question']

            id = item['_id']  # 统一使用 _id 作为检索 link
            
            self.logger.info(f" Task ID: {id}, Starting initial retrieval...")
            # [NEW] 关键词路由 agent
            try:
                selected_kws = self._select_query_keywords(question, model, tokenizer, client, max_k=8)
                self.logger.info(f" Task ID: {id}, Keyword routing results: {selected_kws}")
            except Exception as e:
                selected_kws = []
                self.logger.warning(f" Task ID: {id}, Keyword routing failed, falling back to empty: {e}")

            # [NEW] 并行合并检索（semantic + BM25）
            texts, initial_images = self.retriever_split_sem_bm25(question, id, selected_kws, k_sem=5, k_bm25=3, cache_flag=True, save_retrieval=True, item=item)
            main_context = texts
            self.logger.info(f" Task ID: {id} Initial retrieval complete: Number of text blocks {len(texts.split('Related Memory')) if texts else 0}, Image count {len(initial_images)}")

            self.logger.info("DAG hierarchical decomposition enabled")
            self.logger.info(f" Task ID: {id}, Starting DAG decomposition")
            dag = self.dag_decomposer(main_context, question, model, tokenizer, client, images=initial_images)
            # 保存DAG到JSON文件
            dag_filename = f"dag_{id}_{int(time.time())}.json"
            dag_filepath = os.path.join(self.args.save_dir, "logs",f"data_{item['_id']}", dag_filename)
            os.makedirs(os.path.dirname(dag_filepath), exist_ok=True)
            with open(dag_filepath, 'w', encoding='utf-8') as f:
                json.dump(dag, f, ensure_ascii=False, indent=2)
            self.logger.info(f" DAG saved to: {dag_filepath}")
            self.logger.info(f" Task ID: {id}, Starting DAG execution")
            self.node_results = {}  # 初始化nodes结果
            response = self._execute_dag(dag, question, model, tokenizer, client, main_context, id, idx ,item, initial_images)
            self.logger.info(f" Task ID: {id}, DAG execution complete")
            
            end_time = time.time()
            process_time = end_time - start_time
            time_ls.append(process_time)
            item['response'] = response
            item['pred'] = self.extract_output(response)
            item['process_time'] = process_time
            
            current_count = self.counter.increment()
            self.logger.info(f" Task ID: {id}, Prediction: {item['pred']}, Correct answer: {item['answer']}, Judgment: {item['judge']}, Time: {process_time:.4f}s, Completed: {current_count}")
            self.save_model_responses_to_folders(item, [response], args.save_dir, model, item['_id'],content=main_context)  
            idx += 1

            # 新增：写入样本级 token 使用统计（并清零）
            usage_summary = get_and_reset()
            item['usage'] = usage_summary

            # 将 usage 和总耗时写入样本专属 logs 目录
            unique_data_id = str(item.get('_id', 'unknown'))
            data_folder = Path(args.save_dir) / "logs" / f"data_{unique_data_id}"
            data_folder.mkdir(parents=True, exist_ok=True)
            usage_file = data_folder / "usage_time.jsonl"

            usage_row = {
                "q_id": unique_data_id,
                "duration_total_sec": round(float(item.get("process_time", 0.0)), 6),
                "usage": usage_summary,
            }
            with open(usage_file, "a", encoding="utf-8") as fu:
                fu.write(json.dumps(usage_row, ensure_ascii=False) + "\n")
            if self.logger:
                self.logger.info(f" Usage written: {usage_file} -> {usage_row}")

            fout.write(json.dumps(item, ensure_ascii=False) + "\n")
            fout.flush()
        
        if time_ls:
            avg_time = sum(time_ls) / len(time_ls)
            print(f"Average time per sample: {avg_time:.4f} seconds")
        return time_ls

    def _execute_dag(self, dag, question, model, tokenizer, client, main_context, link, idx, item, initial_images: List[str] = None, max_adjust_rounds=3):
        """Executing DAG：Topo sort + 层层递进检索/推理 + 证据检查 & 调整"""
        from collections import deque, defaultdict
        
        # Adjustment round计数器 (从 0 开始)
        adjust_round = 0
        # breakpoint()
        while adjust_round <= max_adjust_rounds:
            self.logger.info(f" Executing DAG (Adjustment round {adjust_round + 1}/{max_adjust_rounds + 1})")
            
            # 构建图和 indegree（基于提示 schema：nodes 有 id, task, type, children）
            try:
                if not isinstance(dag, dict) or 'nodes' not in dag or not isinstance(dag['nodes'], list):
                    raise ValueError("DAG structure invalid: missing or invalid 'nodes' key")
                
                # Log sample for debugging
                self.logger.info(f"DAG nodes sample: {[type(n).__name__ for n in dag['nodes'][:3]]}")
                
                nodes_dict = {}
                for n in dag['nodes']:
                    if not isinstance(n, dict) or 'id' not in n or 'task' not in n:
                        raise TypeError(f"Invalid node: {n} - must be dict with 'id' and 'task' keys")
                    # 验证根nodes
                    if n['id'] == 'root' and n.get('type') != 'question':
                        self.logger.warning("Root node type mismatch; assuming valid.")
                    nodes_dict[n['id']] = n
                
                self.logger.info(f"Successfully built nodes_dict with {len(nodes_dict)} nodes")
                
                # 只构建一次图：预初始化入度为 0
                graph = defaultdict(list)
                indegree = {node['id']: 0 for node in dag['nodes']}  # 预初始化所有nodes为 0
                for node in dag['nodes']:
                    nid = node['id']
                    for child in node.get('children', []):
                        if child not in indegree:  # 防止无效 child ID
                            self.logger.warning(f"Invalid child {child} in node {nid}; skipping.")
                            continue
                        graph[nid].append(child)
                        indegree[child] += 1
                
                self.logger.info(f"In-degree calculation complete: {indegree}")  # 调试日志
                
            except Exception as e:
                self.logger.error(f"Failed to parse DAG structure: {e}")
                self.logger.info("Falling back to basic reasoner")
                texts, images = self.retriever(question, link, top_k=10, save_retrieval=True, item=item)
                context = texts
                reasoner_input = self.zero_shot_rag_prompt.replace("$DOC$", context).replace("$Q$", question)
                response = self.query_llm(reasoner_input, model, tokenizer, client, temperature=0.2, max_new_tokens=2048, images=images)
                return response
            
            # Topo sort (Kahn 算法)
            queue = deque([nid for nid, deg in indegree.items() if deg == 0])
            execution_order = []
            while queue:
                node_id = queue.popleft()
                execution_order.append(node_id)
                for neighbor in graph[node_id]:
                    indegree[neighbor] -= 1
                    if indegree[neighbor] == 0:
                        queue.append(neighbor)
            
            if len(execution_order) != len(dag['nodes']):
                self.logger.error(f"DAG has cycles or is invalid，Execution order incomplete: {len(execution_order)}/{len(dag['nodes'])}")
                texts, images = self.retriever(question, link, top_k=10, save_retrieval=True, item=item)
                context = texts
                reasoner_input = self.zero_shot_rag_prompt.replace("$DOC$", context).replace("$Q$", question)
                response = self.query_llm(reasoner_input, model, tokenizer, client, temperature=0.2, max_new_tokens=2048, images=images)
                return response  # 回退
            
            self.logger.info(f" DAG execution order determined: {len(execution_order)} 个nodes - {execution_order}")
            
            # 逐nodes执行：层层递进 (子结果增强父上下文)，注入 adjust_round
            for node_id in execution_order:
                if node_id == 'root':
                    continue  # 根nodes是原问题，不执行
                self._execute_dag_node(node_id, nodes_dict, model, tokenizer, client, self.args, idx, item, link, adjust_round=adjust_round)
            
            self.logger.info(f" 所有nodes执行完成 (Round {adjust_round})，nodes结果数量: {len([k for k in self.node_results if k[0] == adjust_round])}")
            
            # 新增：证据检查，注入 adjust_round
            sufficient, gaps = self._check_evidence_sufficiency(question, main_context, nodes_dict, self.node_results, model, tokenizer, client, initial_images, adjust_round=adjust_round)
            if sufficient:
                self.logger.info(" 证据sufficient，跳过调整，直接聚合")
                break
            else:
                self.logger.info(f" Evidence insufficient (Round {adjust_round})，identified gaps: {gaps}")
                if adjust_round < max_adjust_rounds:
                    adjust_round += 1
                    # 准备调整：收集当前 QA 作为 evidence（只当前Round）
                    current_evidence = [{'q': nodes_dict[nid]['task'], 'a': self.node_results[(adjust_round - 1, nid)]} for nid in nodes_dict if (adjust_round - 1, nid) in self.node_results]
                    # 生成调整后的 DAG，传入当前 dag 作为 old_dag
                    dag = self.dag_decomposer_after_check(main_context, question, current_evidence, gaps, model, tokenizer, client, initial_images, old_dag=dag)
                    # 保存调整 DAG
                    adj_filename = f"adj_dag_round_{adjust_round}.json"
                    adj_filepath = os.path.join(self.args.save_dir, "logs", f"data_{item['_id']}", adj_filename)
                    os.makedirs(os.path.dirname(adj_filepath), exist_ok=True)
                    with open(adj_filepath, 'w', encoding='utf-8') as f:
                        json.dump(dag, f, ensure_ascii=False, indent=2)
                    self.logger.info(f" Adjusted DAG saved到: {adj_filepath}")
                    # 继续循环执行新 DAG（node_results 已用 tuple key，不会覆盖旧Round）
                else:
                    self.logger.warning(" 达到最大Adjustment round，强制聚合")
                    break
        # breakpoint()
        # 最终聚合：全轨迹 → 主推理（使用最后Round结果，截断防溢出）
        self.logger.info(f" Starting DAG final aggregation (最终Round: {adjust_round})")
        if not any(k[0] == adjust_round for k in self.node_results):  # 检查最后Round是否有结果
            self.logger.warning("无nodes结果，回退到基本推理")
            texts, images = self.retriever(question, link, top_k=10, save_retrieval=True, item=item)
            context = texts
            reasoner_input = self.zero_shot_rag_prompt.replace("$DOC$", context).replace("$Q$", question)
            response = self.query_llm(reasoner_input, model, tokenizer, client, temperature=0.2, max_new_tokens=2048, images=images)
            if response is None:
                response = "No response"
            return response
        
        # 只用最后Round的结果构建轨迹
        last_round_results = {nid: res for (rnd, nid), res in self.node_results.items() if rnd == adjust_round}
        trajectory_parts = [f"\nNode {nid}: \nQuestion:\n{nodes_dict[nid]['task']}\nAnswer:\n{res}" for nid, res in last_round_results.items()]
        trajectory_str = '\n\n'.join(trajectory_parts)
        trajectory_str = self.truncate_context(trajectory_str, max_tokens=20000)  # 限 20000 tokens
        self.logger.info(f"Trajectory length (最终Round): {self.get_token_length(trajectory_str)} tokens")
        
        final_response, reasoner_input = self.reasoner_with_trajectory(main_context, question, trajectory_str, model, tokenizer, client, item, images=initial_images)
        
        if final_response is None:
            final_response = "No response"
        self.logger.info(" DAG final aggregation complete")

        # 保存最终响应：注入 adjust_round，形成 run_dag_final_round_x.json
        final_filename = f"run_dag_final_round_{adjust_round}.json"
        save_path = os.path.join(self.args.save_dir, "logs", f"data_{item['_id']}", final_filename)
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        with open(save_path, "w", encoding='utf-8') as f:  # 改为 'w'，覆盖旧的最终（或用 'a' 追加历史）
            json.dump({
                "question": question,
                "final_response": final_response,
                "reasoner_input": reasoner_input,
                "initial_images": initial_images if initial_images else [],  # 保存images以便复现
                "adjust_round": adjust_round,
                "total_rounds": adjust_round + 1
            }, f, ensure_ascii=False, indent=2)
        self.logger.info(f" DAG final response saved to: {save_path} (Unique ID: {item['_id']}, 最终Round: {adjust_round}, Image count: {len(initial_images) if initial_images else 0})")
        return final_response

    def main(self):
        args = self.args
        os.makedirs(args.save_dir, exist_ok=True)
        print(args)
        for run_idx in range(args.num_runs):
            print(f"\n======== 开始 {run_idx + 1} run ========")
            
            # 为每run生成不同的输出文件名
            if args.rag > 0:
                base_filename = args.model.split("/")[-1] + f"_{args.method}" + f"_rag_{str(args.rag)}"
            else:
                base_filename = args.model.split("/")[-1] + f"_{args.method}"
            
            if args.num_runs > 1:
                out_file = os.path.join(args.save_dir, base_filename + f"_run_{run_idx + 1}.jsonl")
            else:
                out_file = os.path.join(args.save_dir, base_filename + ".jsonl")

            # Load processed data from JSONL
            with open(args.data_path, 'r', encoding='utf-8') as f:
                dataset = [json.loads(line) for line in f]
            
            # resume from here
            if os.path.exists(out_file):
                with open(out_file, encoding='utf-8') as f:
                    has_data = {json.loads(line)["_id"]: 0 for line in f}
                dataset = [item for item in dataset if item["_id"] not in has_data]

            data_all = dataset
            # cache
            has_data = {}
            if os.path.exists(out_file):
                with open(out_file, encoding='utf-8') as f:
                    has_data = {json.loads(line)["_id"]: 0 for line in f}
            fout = open(out_file, 'a', encoding='utf-8')
            data = []
            
            for item in data_all:
                if item["_id"] not in has_data:
                    data.append(item)
            data_subsets = [data[i::args.n_proc] for i in range(args.n_proc)]
            processes = []
            
            # 根据 args.use_dag 选择使用哪种模式
            pred_method = self.get_pred_dag if getattr(args, 'use_dag', True) else self.get_pred_no_dag
            method_name = "DAG mode" if getattr(args, 'use_dag', True) else "无DAG mode"
            print(f"Using inference method: {method_name}")
            
            if not args.debug:
                for rank in range(args.n_proc):
                    p = mp.Process(target=pred_method, args=(data_subsets[rank], args, fout, rank, run_idx))
                    p.start()
                    processes.append(p)
                for p in processes:
                    p.join()
            #######################Debug use###########################
            if args.debug:
                rank = 0 
                # breakpoint()
                pred_method(data_subsets[0], args, fout, 0, run_idx)

            fout.close()
            print(f"{run_idx + 1} run完成，结果保存到: {out_file}  ==  ")
        
        print("All tasks completed")