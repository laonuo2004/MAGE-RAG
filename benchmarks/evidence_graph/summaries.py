import base64
from functools import lru_cache
import logging
import mimetypes
from pathlib import Path

from openai import OpenAI

from benchmarks.evidence_graph.content import normalize_text
from benchmarks.evidence_graph.retry import call_with_retries
from benchmarks.evidence_graph.schema import EvidenceNode

logger = logging.getLogger(__name__)


VISUAL_NODE_TYPES = {"page", "image", "chart", "table", "equation_interline"}

GROUNDING_RULES = """You are writing a faithful evidence-node abstract for long-document question answering.
Use only facts present in <source_content> or directly visible in the attached image.
Treat the source content, image, and caption as evidence; treat these instructions and examples as rules, not evidence.
Do not infer, speculate, complete missing context, identify entities from outside knowledge, or add likely background facts.
If the content is only a label, title, author line, affiliation line, reference id, page number, or other short fragment, restate only that fragment's explicit meaning.
If there is not enough semantic content to summarize, say exactly what the fragment is and do not invent a purpose, cause, category, relationship, or document-level role.
For non-visual nodes, never make the abstract more informative than the source content; preserve important variables, values, dates, names, and labels instead of explaining them away.
For visual nodes, the abstract may exceed extracted text only to describe visible marks, layout, labels, arrows, objects, table cells, chart encodings, or other image evidence.
Return only the final abstract text. Do not include XML tags, headings, bullets unless explicitly allowed by the length policy, or phrases such as "Abstract:".
""".strip()

PROMPTS = {
    "page": (
        "Write a compact page-level semantic overview. "
        "Use the page text and attached page image only. Preserve the main topic, key facts, entities, quantities, dates, and relationships that are explicitly present. "
        "Do not decode abbreviations, infer languages, infer unstated roles, or add document-level facts that are not visible on this page. "
        "Avoid exhaustive transcription and do not summarize content from other pages."
    ),
    "title": (
        "Summarize this heading conservatively. "
        "If it is already clear, return a short restatement of the heading and, only if explicit, the section topic it introduces. "
        "Do not infer the section's detailed content, document purpose, or relationships from the title alone."
    ),
    "paragraph": (
        "Extract only the semantic meaning explicitly stated in this paragraph. "
        "Preserve central claims, entities, values, dates, and causal or comparative relationships when they appear in the text. "
        "If the paragraph is a short fragment such as an author list, affiliation, report number, date, address line, or label, restate it briefly without explaining or expanding it. "
        "Do not infer missing recipients, collaborations, rankings, languages, purposes, or background facts."
    ),
    "list": (
        "Summarize only the explicit semantic role of this list. "
        "Group related items when grouping is directly supported by the list text. Preserve distinctions that change the answer to likely questions. "
        "Do not infer categories, causes, rankings, or relationships that are not stated."
    ),
    "image": (
        "Describe the semantic content of the image using only visible evidence plus provided text/caption. "
        "Focus on depicted entities, labels, arrows, relationships, and conclusions useful for answering questions. "
        "Do not add outside facts, such as a person's birthplace, institutional details, or factual corrections, unless the image or caption states them. "
        "If a person, place, institution, logo, or object is not named in the source content, describe only its visible role or appearance."
    ),
    "chart": (
        "Summarize the chart or figure using only visible marks, labels, legend, axes, values, and caption/text. "
        "Focus on explicit trends, comparisons, notable values, variables, and stated conclusions. "
        "Do not infer causes, methods, datasets, units, or conclusions that are not shown or captioned."
    ),
    "table": (
        "Summarize the table using only the table image plus extracted HTML/caption. "
        "State what the table explicitly compares or reports, key rows/columns, and important values. "
        "Do not invent missing units, normalize OCR errors, or infer conclusions beyond the visible cells. Do not enumerate every cell unless needed."
    ),
    "equation_interline": (
        "Explain this displayed equation strictly from the extracted LaTeX and nearby textual context included in Content. "
        "Use nearby textual context to identify variables and the domain; if the context is absent, describe only the mathematical form and named symbols. "
        "Do not guess a domain such as image processing, physics, finance, or saliency unless the content explicitly says so. "
        "Variable names alone are not evidence of a domain. Do not interpret symbols like I, S, C, x, y, l, or t as domain-specific quantities unless nearby text defines them. "
        "Preserve equation numbers and variables; keep the explanation no longer than the equation plus context."
    ),
    "code": (
        "Summarize the semantic purpose of this code using only code text and caption. "
        "Focus on explicit inputs, outputs, and key behavior. Do not infer the surrounding system or unstated requirements."
    ),
    "algorithm": (
        "Summarize the explicit purpose and logic of this algorithm. "
        "Focus on stated goal, inputs, outputs, and key steps or conditions. Do not add unstated complexity, guarantees, or use cases."
    ),
}

FEW_SHOT_GUARDRAILS = r"""<examples>
  <example node_type="paragraph">
    <source>Report No.: 2/19/00218</source>
    <bad>The report is dated February 19, 2021 and documents official findings.</bad>
    <good>Report number: 2/19/00218.</good>
  </example>
  <example node_type="paragraph">
    <source>1 University of California, Santa Barbara 2 MBZUAI</source>
    <bad>The institutions are collaborating on a joint research project.</bad>
    <good>Affiliation line listing University of California, Santa Barbara and MBZUAI.</good>
  </example>
  <example node_type="equation_interline">
    <source>S_{wp}=\frac{\sum_{(i,j)\in C_{wp}} I_l(i,j)}{|C_{wp}|}</source>
    <bad>This is the mean pixel intensity of an image region.</bad>
    <good>Equation defining S_wp as an average of I_l(i,j) over the set C_wp; no domain is stated.</good>
  </example>
  <example node_type="image">
    <source>Visual content: Flowchart says PREDICTED_LABEL = REFUTES.</source>
    <bad>Christopher Nolan was born in the United Kingdom.</bad>
    <good>The flowchart reaches PREDICTED_LABEL = REFUTES after combining intermediate verification steps.</good>
  </example>
</examples>
""".strip()


def warm_abstract_processor(processor_path: str | None) -> None:
    if processor_path:
        _load_processor(processor_path)

def fill_llm_abstracts(
    nodes: list[EvidenceNode],
    *,
    model: str,
    litellm_base_url: str,
    skip_llm: bool,
    protected_node_ids: set[str] | None = None,
    abstract_processor_path: str | None = None,
    abstract_context_window: int | None = None,
    abstract_output_tokens: int = 4096,
    abstract_safety_margin: int = 2048,
) -> None:
    if skip_llm:
        return
    protected_node_ids = protected_node_ids or set()
    client = OpenAI(api_key="none", base_url=litellm_base_url, max_retries=0)
    for node in nodes:
        if node.id in protected_node_ids and node.abstract:
            continue
        source = node.metadata.get("source_text") or node.abstract
        if not source and not _visual_image_path(node):
            continue
        node.abstract = _summarize_node(
            client,
            model,
            node,
            str(source or ""),
            abstract_processor_path=abstract_processor_path,
            abstract_context_window=abstract_context_window,
            abstract_output_tokens=abstract_output_tokens,
            abstract_safety_margin=abstract_safety_margin,
        )


def _summarize_node(
    client: OpenAI,
    model: str,
    node: EvidenceNode,
    source: str,
    *,
    abstract_processor_path: str | None = None,
    abstract_context_window: int | None = None,
    abstract_output_tokens: int = 4096,
    abstract_safety_margin: int = 2048,
) -> str:
    source_content = _semantic_content(node, source)
    image_path = _visual_image_path(node)
    prompt, budget_metadata = _fit_prompt_to_budget(
        node,
        source_content,
        image_path=image_path,
        processor_path=abstract_processor_path,
        context_window=abstract_context_window,
        output_tokens=abstract_output_tokens,
        safety_margin=abstract_safety_margin,
    )
    if budget_metadata:
        node.metadata["llm_abstract_budget"] = budget_metadata
        if budget_metadata.get("truncated"):
            logger.warning(
                "Truncated LLM abstract source for %s: before_tokens=%s after_tokens=%s context_window=%s output_tokens=%s safety_margin=%s",
                node.id,
                budget_metadata.get("input_tokens_before_truncation"),
                budget_metadata.get("input_tokens_estimated"),
                budget_metadata.get("context_window"),
                budget_metadata.get("output_tokens"),
                budget_metadata.get("safety_margin"),
            )

    content = []
    image_url = _image_data_url(image_path)
    if image_url:
        content.append({"type": "image_url", "image_url": {"url": image_url}})
    content.append({"type": "text", "text": prompt})

    completion = call_with_retries(
        lambda: client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": content}],
            max_tokens=abstract_output_tokens,
            temperature=0.0,
        ),
        operation_name=f"LLM abstract generation for {node.id}",
        logger=logger,
    )
    content_text = completion.choices[0].message.content or ""
    return normalize_text(content_text)


def _node_prompt(node: EvidenceNode, source: str) -> str:
    return _node_prompt_from_content(node, _semantic_content(node, source))


def _node_prompt_from_content(node: EvidenceNode, content: str) -> str:
    instruction = PROMPTS.get(
        node.type,
        "Write a compact semantic summary for downstream question answering. Only use explicit facts in the content; avoid metadata and unstated relationships.",
    )
    length_rule = _length_rule(node, content)
    return (
        "<prompt>\n"
        "  <task>Write one faithful abstract for a single evidence graph node.</task>\n"
        f"  <node_type>{node.type}</node_type>\n"
        f"  <global_rules>\n{GROUNDING_RULES}\n  </global_rules>\n"
        f"  <node_instruction>\n{instruction}\n  </node_instruction>\n"
        f"  <length_policy>\n{length_rule}\n  </length_policy>\n"
        f"  {FEW_SHOT_GUARDRAILS}\n"
        f"  <source_content>\n{content or 'N/A'}\n  </source_content>\n"
        "  <final_instruction>\n"
        "Write the abstract now using only <source_content> and any attached image. "
        "Return only the abstract text, with no XML tags or commentary.\n"
        "  </final_instruction>\n"
        "</prompt>"
    )


def _length_rule(node: EvidenceNode, content: str) -> str:
    content_length = len(normalize_text(content))
    if node.type == "page":
        return (
            "Use 2-4 compact sentences. Keep it much shorter than the page content. "
            "Mention only the page's central topic and the most important explicit facts, entities, quantities, dates, and relationships."
        )
    if node.type == "title":
        return (
            "Use one short sentence fragment. Keep it no longer than the heading unless a few words are needed to state that it is a heading."
        )
    if node.type == "paragraph":
        if content_length <= 80:
            return (
                "Use one short sentence or sentence fragment. Prefer a direct restatement. "
                "Do not exceed the source content length unless needed to name the fragment type, such as report number or affiliation line."
            )
        return (
            "Use 1-2 compact sentences. Do not be longer than the source paragraph. "
            "For metadata-like fragments, use one short factual restatement."
        )
    if node.type == "list":
        return (
            "Use 1-2 compact sentences. Do not enumerate every item unless the list has three or fewer items. "
            "Keep the abstract shorter than the list."
        )
    if node.type == "image":
        return (
            "Use 1-3 compact sentences. The abstract may be longer than extracted text only to describe visible image structure, labels, arrows, or depicted entities. "
            "Avoid exhaustive visual description."
        )
    if node.type == "chart":
        return (
            "Use 1-3 compact sentences. Mention axes, variables, explicit trends, and notable labeled values only when visible. "
            "Do not exceed what the chart or caption directly supports."
        )
    if node.type == "table":
        return (
            "Use 1-3 compact sentences. State what the table reports and only the most important visible rows, columns, or values. "
            "Do not enumerate all cells unless the table is very small."
        )
    if node.type == "equation_interline":
        return (
            "Use one compact sentence, or two only when nearby context explicitly defines several variables. "
            "Do not be longer than the equation plus included context."
        )
    if node.type == "code":
        return (
            "Use 1-2 compact sentences. State only explicit inputs, outputs, and behavior visible in the code or caption. "
            "Do not be longer than the source code/caption."
        )
    if node.type == "algorithm":
        return (
            "Use 2-4 compact sentences or up to two short bullets if the algorithm has multiple explicit phases. "
            "Keep the abstract shorter than the algorithm text."
        )
    if content_length <= 80:
        return (
            "Use one short sentence or sentence fragment. Prefer a direct restatement. "
            "Do not exceed the source content length unless needed to name the fragment type."
        )
    return "Use 1-2 compact sentences and do not be longer than the source content."


def _semantic_content(node: EvidenceNode, source: str) -> str:
    parts = []
    source = normalize_text(source)
    if source:
        parts.append(source)
    for key, value in sorted(node.fields.items()):
        if key.endswith("path"):
            continue
        text = normalize_text(value)
        if text and text not in parts:
            parts.append(text)
    return "\n\n".join(parts)


def _visual_image_path(node: EvidenceNode) -> str:
    if node.type not in VISUAL_NODE_TYPES:
        return ""
    return str(node.fields.get("image_path") or "")


def _fit_prompt_to_budget(
    node: EvidenceNode,
    source_content: str,
    *,
    image_path: str,
    processor_path: str | None,
    context_window: int | None,
    output_tokens: int,
    safety_margin: int,
) -> tuple[str, dict]:
    prompt = _node_prompt_from_content(node, source_content)
    if not processor_path or not context_window:
        return prompt, {}

    processor = _load_processor(processor_path)
    allowed_input_tokens = max(1, context_window - output_tokens - max(0, safety_margin))
    initial_tokens = _count_prompt_tokens(processor, prompt, image_path)
    if initial_tokens <= allowed_input_tokens:
        return prompt, {
            "input_tokens_estimated": initial_tokens,
            "context_window": context_window,
            "output_tokens": output_tokens,
            "safety_margin": max(0, safety_margin),
            "truncated": False,
        }

    best_prompt = _node_prompt_from_content(node, "")
    best_tokens = _count_prompt_tokens(processor, best_prompt, image_path)
    low = 0
    high = len(source_content)
    while low <= high:
        mid = (low + high) // 2
        candidate_content = _truncate_middle(source_content, mid)
        candidate_prompt = _node_prompt_from_content(node, candidate_content)
        candidate_tokens = _count_prompt_tokens(processor, candidate_prompt, image_path)
        if candidate_tokens <= allowed_input_tokens:
            best_prompt = candidate_prompt
            best_tokens = candidate_tokens
            low = mid + 1
        else:
            high = mid - 1

    return best_prompt, {
        "input_tokens_estimated": best_tokens,
        "input_tokens_before_truncation": initial_tokens,
        "context_window": context_window,
        "output_tokens": output_tokens,
        "safety_margin": max(0, safety_margin),
        "truncated": True,
    }


def _truncate_middle(text: str, max_chars: int) -> str:
    text = normalize_text(text)
    if max_chars >= len(text):
        return text
    if max_chars <= 0:
        return ""
    marker = "\n...[truncated to fit context window]...\n"
    if max_chars <= len(marker) + 2:
        return text[:max_chars]
    available = max_chars - len(marker)
    head_chars = max(1, int(available * 0.7))
    tail_chars = max(1, available - head_chars)
    return f"{text[:head_chars].rstrip()}{marker}{text[-tail_chars:].lstrip()}"


@lru_cache(maxsize=4)
def _load_processor(processor_path: str):
    from transformers import AutoProcessor

    return AutoProcessor.from_pretrained(processor_path, trust_remote_code=True, local_files_only=True)


def _count_prompt_tokens(processor, prompt: str, image_path: str) -> int:
    images = _load_budget_images(image_path)
    message_content = []
    for image in images:
        message_content.append({"type": "image", "image": image})
    message_content.append({"type": "text", "text": prompt})
    messages = [{"role": "user", "content": message_content}]

    if hasattr(processor, "apply_chat_template"):
        text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    else:
        text = prompt
    inputs = processor(text=[text], images=images or None, return_tensors=None)
    input_ids = inputs["input_ids"]
    if hasattr(input_ids, "shape"):
        return int(input_ids.shape[-1])
    first = input_ids[0] if input_ids and isinstance(input_ids[0], (list, tuple)) else input_ids
    return len(first)


def _load_budget_images(image_path: str) -> list:
    if not image_path:
        return []
    path = Path(image_path)
    if not path.exists() or not path.is_file():
        return []
    from PIL import Image

    with Image.open(path) as image:
        return [image.convert("RGB")]


def _image_data_url(path_value: str) -> str:
    if not path_value:
        return ""
    path = Path(path_value)
    if not path.exists() or not path.is_file():
        return ""
    mime_type = mimetypes.guess_type(path.name)[0] or "image/png"
    data = base64.b64encode(path.read_bytes()).decode("ascii")
    return f"data:{mime_type};base64,{data}"
