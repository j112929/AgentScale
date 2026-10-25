"""Bounded HTTP integration with vLLM's public inference and metrics endpoints."""

import http.client
import json
import math
import re
import time
from dataclasses import dataclass
from urllib.parse import urlsplit


class RuntimeHTTPError(RuntimeError):
    pass


class MetricsUnavailable(RuntimeError):
    pass


class HTTPTransport:
    def __init__(self, base_url: str, timeout_s: float = 10, api_key: str | None = None):
        url = urlsplit(base_url)
        if (
            url.scheme not in ("http", "https")
            or not url.hostname
            or url.username
            or url.password
            or url.query
            or url.fragment
            or not math.isfinite(timeout_s)
            or timeout_s <= 0
        ):
            raise ValueError(
                "HTTP(S) base URL without credentials/query and positive timeout required"
            )
        self.url, self.timeout_s, self._api_key = url, timeout_s, api_key

    def request(self, method: str, path: str, payload: dict | None = None) -> bytes:
        assert self.url.hostname is not None
        cls = (
            http.client.HTTPSConnection
            if self.url.scheme == "https"
            else http.client.HTTPConnection
        )
        connection = cls(self.url.hostname, self.url.port, timeout=self.timeout_s)
        headers = {"Accept": "application/json"}
        if self._api_key:
            headers["Authorization"] = "Bearer " + self._api_key
        body = None
        if payload is not None:
            body = json.dumps(payload, allow_nan=False).encode()
            headers["Content-Type"] = "application/json"
        try:
            connection.request(
                method, self.url.path.rstrip("/") + "/" + path.lstrip("/"), body, headers
            )
            response = connection.getresponse()
            data = response.read(2 * 1024**2 + 1)
            if len(data) > 2 * 1024**2:
                raise RuntimeHTTPError("runtime response exceeds 2 MiB")
            if not 200 <= response.status < 300:
                # Do not follow redirects or include credentials/response bodies in exceptions.
                raise RuntimeHTTPError(f"runtime returned HTTP {response.status}")
            return data
        except (OSError, http.client.HTTPException) as exc:
            raise RuntimeHTTPError(f"runtime connection failed: {type(exc).__name__}") from exc
        finally:
            connection.close()


@dataclass(frozen=True)
class ChatResult:
    text: str
    model: str
    prompt_tokens: int | None = None
    completion_tokens: int | None = None


@dataclass(frozen=True)
class InferenceMetrics:
    timestamp_s: float
    model: str
    running: int
    waiting: int
    cache_usage_fraction: float | None
    generation_tokens_total: float | None
    source: str = "vllm-prometheus"


_SAMPLE = re.compile(r"^([a-zA-Z_:][a-zA-Z0-9_:]*)(\{.*\})?\s+(\S+)(?:\s+\S+)?$")
_LABEL = re.compile(r'([a-zA-Z_][a-zA-Z0-9_]*)="((?:\\.|[^"\\])*)"')


def parse_metrics(text: str, model: str, now_s: float) -> InferenceMetrics:
    wanted = {
        "vllm:num_requests_running",
        "vllm:num_requests_waiting",
        "vllm:gpu_cache_usage_perc",
        "vllm:kv_cache_usage_perc",
        "vllm:generation_tokens_total",
    }
    values: dict[str, list[float]] = {}
    seen = set()
    for raw in text.splitlines():
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        match = _SAMPLE.fullmatch(line)
        if not match or match[1] not in wanted:
            continue
        labels = {key: json.loads('"' + val + '"') for key, val in _LABEL.findall(match[2] or "")}
        if labels.get("model_name") != model:
            continue
        key = (match[1], tuple(sorted(labels.items())))
        if key in seen:
            raise MetricsUnavailable("duplicate metric series")
        seen.add(key)
        value = float(match[3])
        if not math.isfinite(value) or value < 0:
            raise MetricsUnavailable("invalid nonfinite or negative metric")
        values.setdefault(match[1], []).append(value)

    def count(name: str) -> int:
        if name not in values or any(v != int(v) for v in values[name]):
            raise MetricsUnavailable("missing or invalid request-count metric: " + name)
        return int(sum(values[name]))

    caches = values.get("vllm:kv_cache_usage_perc", values.get("vllm:gpu_cache_usage_perc", []))
    if any(v > 1 for v in caches):
        raise MetricsUnavailable("cache occupancy must be a fraction between zero and one")
    tokens = values.get("vllm:generation_tokens_total")
    return InferenceMetrics(
        now_s,
        model,
        count("vllm:num_requests_running"),
        count("vllm:num_requests_waiting"),
        max(caches) if caches else None,
        sum(tokens) if tokens else None,
    )


class VLLMAdapter:
    """Inference and read-only worker telemetry; GPU reallocation is recommendation-only."""

    def __init__(
        self,
        base_url: str,
        model: str,
        *,
        timeout_s: float = 30,
        api_key: str | None = None,
        transport: HTTPTransport | None = None,
    ):
        if not model:
            raise ValueError("explicit served model name required")
        self.model = model
        self.transport = transport or HTTPTransport(base_url, timeout_s, api_key)

    def health(self) -> bool:
        self.transport.request("GET", "/health")
        return True

    def version(self) -> str:
        return str(json.loads(self.transport.request("GET", "/version"))["version"])

    def collect(self, now_s: float | None = None) -> InferenceMetrics:
        raw = self.transport.request("GET", "/metrics").decode()
        return parse_metrics(raw, self.model, time.monotonic() if now_s is None else now_s)

    def chat(self, messages: list[dict[str, str]], max_tokens: int = 1024) -> ChatResult:
        if type(max_tokens) is not int or max_tokens < 1:
            raise ValueError("max_tokens must be a positive integer")
        response = json.loads(
            self.transport.request(
                "POST",
                "/v1/chat/completions",
                {
                    "model": self.model,
                    "messages": messages,
                    "temperature": 0,
                    "max_tokens": max_tokens,
                    "stream": False,
                },
            )
        )
        try:
            content = response["choices"][0]["message"]["content"]
            if not isinstance(content, str):
                raise TypeError("text response required")
            usage = response.get("usage") or {}
            return ChatResult(
                content,
                response.get("model", self.model),
                usage.get("prompt_tokens"),
                usage.get("completion_tokens"),
            )
        except (KeyError, IndexError, TypeError, ValueError) as exc:
            raise RuntimeHTTPError("malformed chat-completion response") from exc

    def resize(self, gpu_workers: int) -> None:
        raise NotImplementedError(
            "GPU worker changes require a verified deployment/safe-point actuator"
        )
