import argparse
import json
import os
import random
import sys
from collections import defaultdict
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib import error, request

from env_utils import get_config_value, load_local_env


RETRYABLE_CONTEXT_MARKERS = (
    "maximum context length",
    "input length",
    "context length",
    "prompt is too long",
    "max_model_len",
)


RETRYABLE_CONNECTION_MARKERS = (
    "connection error",
    "connection reset",
    "readerror",
    "timeout",
    "temporarily unavailable",
)


def split_csv(value):
    if not value:
        return []
    return [item.strip() for item in value.split(",") if item.strip()]


def load_router_config(args):
    local_env = load_local_env()
    model_name = get_config_value(args.model_name, "ROUTER_MODEL_NAME", local_env=local_env)
    if not model_name:
        raise ValueError("Missing router model name. Set --model-name or ROUTER_MODEL_NAME.")

    config = {
        "model_name": model_name,
        "upstream_api_key": get_config_value(args.api_key, "ROUTER_UPSTREAM_API_KEY", local_env=local_env, default="EMPTY"),
        "groups": {
            "throughput": split_csv(get_config_value(args.throughput, "ROUTER_THROUGHPUT_BASE_URLS", local_env=local_env, default="")),
            "longctx": split_csv(get_config_value(args.longctx, "ROUTER_LONGCTX_BASE_URLS", local_env=local_env, default="")),
            "maxctx": split_csv(get_config_value(args.maxctx, "ROUTER_MAXCTX_BASE_URLS", local_env=local_env, default="")),
        },
    }
    if not any(config["groups"].values()):
        raise ValueError("No upstream base URLs configured. Set ROUTER_*_BASE_URLS or pass CLI flags.")
    return config


class RouterState:
    def __init__(self, config):
        self.model_name = config["model_name"]
        self.api_key = config["upstream_api_key"]
        self.groups = config["groups"]
        self._rr_index = defaultdict(int)

    def list_models_payload(self):
        return {
            "object": "list",
            "data": [
                {
                    "id": self.model_name,
                    "object": "model",
                    "created": 0,
                    "owned_by": "litellm-router",
                }
            ],
        }

    def ordered_upstreams(self):
        ordered = []
        for group_name in ("throughput", "longctx", "maxctx"):
            urls = self.groups.get(group_name) or []
            if not urls:
                continue
            start = self._rr_index[group_name] % len(urls)
            rotated = urls[start:] + urls[:start]
            self._rr_index[group_name] = (start + 1) % len(urls)
            for url in rotated:
                ordered.append((group_name, url.rstrip("/")))
        return ordered


def should_retry_on_error(message):
    text = str(message or "").lower()
    return any(marker in text for marker in RETRYABLE_CONTEXT_MARKERS + RETRYABLE_CONNECTION_MARKERS)


def read_error_body(exc):
    if isinstance(exc, error.HTTPError):
        try:
            return exc.read().decode("utf-8", errors="replace")
        except Exception:
            return str(exc)
    return str(exc)


def upstream_chat_completion(state, payload):
    attempts = []
    last_status = 502
    last_body = None

    for group_name, base_url in state.ordered_upstreams():
        upstream_url = f"{base_url}/v1/chat/completions"
        req = request.Request(
            upstream_url,
            data=json.dumps(payload).encode("utf-8"),
            headers={
                "Content-Type": "application/json",
                "Authorization": f"Bearer {state.api_key}",
            },
            method="POST",
        )
        try:
            with request.urlopen(req, timeout=600) as resp:
                body = resp.read()
                status = resp.status
                response_json = json.loads(body.decode("utf-8"))
                response_json.setdefault("_router", {})
                response_json["_router"]["selected_group"] = group_name
                response_json["_router"]["selected_base_url"] = base_url
                response_json["_router"]["attempts"] = attempts + [{"group": group_name, "base_url": base_url, "status": status}]
                return status, response_json
        except Exception as exc:
            body_text = read_error_body(exc)
            status = exc.code if isinstance(exc, error.HTTPError) else 502
            attempts.append(
                {
                    "group": group_name,
                    "base_url": base_url,
                    "status": status,
                    "error": body_text,
                }
            )
            last_status = status
            last_body = body_text
            if should_retry_on_error(body_text):
                continue
            break

    error_payload = {
        "error": {
            "message": last_body or "All upstreams failed.",
            "type": "router_error",
            "code": last_status,
        },
        "_router": {
            "attempts": attempts,
        },
    }
    return last_status, error_payload


def make_handler(state):
    class RouterHandler(BaseHTTPRequestHandler):
        server_version = "LiteLLMRouter/0.1"

        def _send_json(self, status, payload):
            body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
            self.send_response(status)
            self.send_header("Content-Type", "application/json; charset=utf-8")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def do_GET(self):
            if self.path == "/v1/models":
                self._send_json(200, state.list_models_payload())
                return
            if self.path == "/health":
                self._send_json(200, {"ok": True})
                return
            self._send_json(404, {"detail": "Not Found"})

        def do_POST(self):
            if self.path != "/v1/chat/completions":
                self._send_json(404, {"detail": "Not Found"})
                return

            try:
                content_length = int(self.headers.get("Content-Length", "0"))
                raw_body = self.rfile.read(content_length)
                payload = json.loads(raw_body.decode("utf-8"))
            except Exception as exc:
                self._send_json(400, {"error": {"message": f"Invalid JSON body: {exc}"}})
                return

            if payload.get("model") != state.model_name:
                payload["model"] = state.model_name

            status, response_payload = upstream_chat_completion(state, payload)
            self._send_json(status, response_payload)

        def log_message(self, fmt, *args):
            sys.stderr.write("%s - - [%s] %s\n" % (self.address_string(), self.log_date_time_string(), fmt % args))

    return RouterHandler


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=4001)
    parser.add_argument("--model-name", dest="model_name", default=None)
    parser.add_argument("--api-key", default=None)
    parser.add_argument("--throughput", default=None, help="Comma-separated base URLs for throughput backends.")
    parser.add_argument("--longctx", default=None, help="Comma-separated base URLs for longctx backends.")
    parser.add_argument("--maxctx", default=None, help="Comma-separated base URLs for maxctx backends.")
    args = parser.parse_args()

    config = load_router_config(args)
    state = RouterState(config)
    handler = make_handler(state)
    server = ThreadingHTTPServer((args.host, args.port), handler)

    print(
        f"LiteLLM router listening on http://{args.host}:{args.port}/v1 "
        f"for model={state.model_name}",
        flush=True,
    )
    print(
        "Backends: "
        f"throughput={len(state.groups['throughput'])}, "
        f"longctx={len(state.groups['longctx'])}, "
        f"maxctx={len(state.groups['maxctx'])}",
        flush=True,
    )
    server.serve_forever()


if __name__ == "__main__":
    main()
