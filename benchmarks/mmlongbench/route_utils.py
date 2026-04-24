import json
import os
import re
from dataclasses import dataclass


@dataclass
class RouteSpec:
    index: int
    label: str
    model_name: str
    base_url: str | None
    api_key: str | None
    max_model_len: int | None = None


def parse_csv(value):
    if value is None:
        return []
    if isinstance(value, list):
        return value
    return [item.strip() for item in str(value).split(",") if item.strip()]


def load_json_file(path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def safe_load_existing_samples(path):
    if not os.path.exists(path):
        return None
    try:
        return load_json_file(path)
    except Exception:
        return None


def build_routes(
    *,
    model_name,
    base_url,
    api_key,
    route_model_names=None,
    route_base_urls=None,
    route_api_keys=None,
    route_labels=None,
    route_max_model_lens=None,
):
    route_base_urls = parse_csv(route_base_urls)
    route_model_names = parse_csv(route_model_names)
    route_api_keys = parse_csv(route_api_keys)
    route_labels = parse_csv(route_labels)
    route_max_model_lens = parse_csv(route_max_model_lens)

    if not route_base_urls:
        return [
            RouteSpec(
                index=0,
                label="default",
                model_name=model_name,
                base_url=base_url,
                api_key=api_key,
                max_model_len=None,
            )
        ]

    routes = []
    for idx, item_base_url in enumerate(route_base_urls):
        item_model_name = route_model_names[idx] if idx < len(route_model_names) else model_name
        item_api_key = route_api_keys[idx] if idx < len(route_api_keys) else api_key
        item_label = route_labels[idx] if idx < len(route_labels) else f"route-{idx}"
        item_max_model_len = None
        if idx < len(route_max_model_lens):
            try:
                item_max_model_len = int(route_max_model_lens[idx])
            except ValueError:
                item_max_model_len = None
        routes.append(
            RouteSpec(
                index=idx,
                label=item_label,
                model_name=item_model_name,
                base_url=item_base_url,
                api_key=item_api_key,
                max_model_len=item_max_model_len,
            )
        )
    return routes


def extract_context_limit(error_message):
    message = str(error_message or "")
    match = re.search(r"maximum context length \((\d+)\)", message)
    if match:
        return int(match.group(1))
    return None


def is_context_overflow_error(error_message):
    message = str(error_message or "").lower()
    patterns = [
        "maximum context length",
        "input length",
        "context length",
        "exceeds model's maximum",
        "too many tokens",
    ]
    return any(pattern in message for pattern in patterns)


def find_next_route_index(routes, current_index, error_message):
    limit = extract_context_limit(error_message)
    if limit is not None:
        for route in routes[current_index + 1 :]:
            if route.max_model_len is None or route.max_model_len > limit:
                return route.index
        return None

    if current_index + 1 < len(routes):
        return current_index + 1
    return None
