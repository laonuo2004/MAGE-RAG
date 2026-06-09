from dataclasses import dataclass
from typing import Dict, Any
import os

LLM_BASE_URL: str = "http://localhost:00000/v1"
LLM_API_KEY: str = "<Your API Key>"

EMBED_BASE_URL: str = "http://localhost:00000/v1" 
EMBED_API_KEY: str = "<Your API Key>"

MODELS = {
    "chat": "qwen3-vl-32b-instruct",
    "embed": "text-embedding-3-small",
    "eval" : "gpt-4o-mini",
}

LLM_GENERATION = {
    "temperature": 0.0, #0.7,
    "max_tokens": 8192,
}

# JSON schema used for both text and image analysis
RESPONSE_FORMAT: Dict[str, Any] = {
    "type": "json_schema",
    "json_schema": {
        "name": "response",
        "schema": {
            "type": "object",
            "properties": {
                "keywords": {"type": "array", "items": {"type": "string"}},
                "summary": {"type": "string"},
                "tags": {"type": "array", "items": {"type": "string"}},
            },
            "required": ["keywords", "summary", "tags"],
            "additionalProperties": False,
        },
        "strict": True,
    },
}


# -----------------------------
# Paths and Dataset Configuration (hard-coded)
# -----------------------------
MEMORY_SYSTEMS_DIR: str =  "<Your Memory Systems Directory>" 
PDF_TMP_DIR: str = "/tmp/visdom_pdfs"
DATA_ROOT = "/D2-Reader/data"

DATASETS = {
    # name -> {"csv":..., "mineru_dir":..., "key": col name holding q_id, "docs_col": col of doc list}
    "MMLongBench": {
        "csv": f"{DATA_ROOT}/Visdom/MMLongBench/MMLongBench.csv",
        "mineru_dir": f"{DATA_ROOT}/mineru/MMLongBench/",
        "key": "q_id",
        "docs_col": "documents",
    },
    "spiqa": {
        "csv": f"{DATA_ROOT}/Visdom/spiqa/spiqa.csv",
        "mineru_dir": f"{DATA_ROOT}/mineru/spiqa/",
        "key": "q_id",
        "docs_col": "documents",
    },
    "feta_tab": {
        "csv": f"{DATA_ROOT}/Visdom/feta_tab/feta_tab.csv",
        "mineru_dir": f"{DATA_ROOT}/mineru/fetatab/",
        "key": "q_id",
        "docs_col": "documents",
        "encoding": "utf-8",
    },
    "scgqa": {
        "csv": f"{DATA_ROOT}/Visdom/scigraphvqa/scigraphqa.csv",
        "mineru_dir": f"{DATA_ROOT}/mineru/scigraphvqa/",
        "key": "q_id",
        "docs_col": "documents",
    },
    "paper_tab": {
        "csv": f"{DATA_ROOT}/Visdom/paper_tab/paper_tab.csv",
        "mineru_dir": f"{DATA_ROOT}/mineru/papertab/",
        "key": "q_id",
        "docs_col": "documents",
    },
    "slidevqa": {
        "csv": f"{DATA_ROOT}/Visdom/slidevqa/slidevqa.csv",
        "mineru_dir": f"{DATA_ROOT}/mineru/slide/",
        "key": "q_id",
        "docs_col": "documents",
    },
}

_PROMPTS_DIR = os.path.join(os.path.dirname(__file__), "prompts")

def _read_prompt(filename: str) -> str:
    path = os.path.join(_PROMPTS_DIR, filename)
    with open(path, "r", encoding="utf-8") as f:
        return f.read()

reasoner_prompt_with_trajectory = _read_prompt("reasoner_prompt_with_trajectory.txt")
zero_shot_rag_prompt = _read_prompt("0shot_rag.txt")
decomposer_dag_prompt = _read_prompt("decomposer_dag.txt")
reasoner_dag_prompt = _read_prompt("reasoner_dag_node.txt")
evidence_check_prompt = _read_prompt("evidence_check.txt")
decomposer_dag_after_check_prompt = _read_prompt("dag_decomposer_after_check_prompt.txt")
text_analysis_prompt = _read_prompt("text_analysis.txt")
image_analysis_prompt = _read_prompt("image_analysis.txt")
image_analysis_prompt_mineru = _read_prompt("image_analysis_mineru.txt")
hybrid_analysis_prompt = _read_prompt("hybrid_analysis.txt")
memory_evolve_prompt_sjr = _read_prompt("memory_evolve.txt")
evaluate_prompt = _read_prompt("eval.txt")
image_analysis_prompt_keyword = _read_prompt("image_analysis_keyword.txt")
text_analysis_prompt_keyword = _read_prompt("text_analysis_keyword.txt")
image_ocr_keyword_prompt = _read_prompt("image_analysis_ocrkeyword.txt")
query_keywords_prompt = _read_prompt("query_keywords.txt")

PROMPTS = {
    "image": image_analysis_prompt,
    "image_mineru": image_analysis_prompt_mineru,
    "image_keyword": image_analysis_prompt_keyword,
    "image_ocr_keyword": image_ocr_keyword_prompt,
    "text": text_analysis_prompt,
    "text_keyword": text_analysis_prompt_keyword,
    "query_keywords": query_keywords_prompt,
    "hybrid": hybrid_analysis_prompt,
    "evolve": memory_evolve_prompt_sjr,
    "reasoner": reasoner_prompt_with_trajectory,
    "zero_shot_rag": zero_shot_rag_prompt,
    "decomposer_dag": decomposer_dag_prompt,
    "reasoner_dag": reasoner_dag_prompt,
    "evidence_check": evidence_check_prompt,
    "decomposer_dag_after_check": decomposer_dag_after_check_prompt,
    "evaluate": evaluate_prompt,
}

MAX_CONCURRENCY = 10

PARALLEL_ANALYSIS = False

@dataclass
class SaveChecks:
    min_bytes: int = 10

SAVE_CHECKS = SaveChecks()