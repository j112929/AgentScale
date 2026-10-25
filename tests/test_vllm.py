import json
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import pytest

from agentscale.vllm_adapter import (
    HTTPTransport,
    MetricsUnavailable,
    RuntimeHTTPError,
    VLLMAdapter,
    parse_metrics,
)

METRICS = """# HELP vllm:num_requests_waiting Number waiting
vllm:num_requests_waiting{model_name="test",engine="0"} 2
vllm:num_requests_waiting{model_name="test",engine="1"} 3
vllm:num_requests_running{model_name="test",engine="0"} 1
vllm:kv_cache_usage_perc{model_name="test",engine="0"} 0.4
vllm:generation_tokens_total{model_name="test",engine="0"} 123
vllm:num_requests_waiting{model_name="other"} 99
"""


def test_metric_aggregation_and_missing_values():
    m = parse_metrics(METRICS, "test", 42)
    assert m.waiting == 5 and m.running == 1 and m.timestamp_s == 42
    assert m.cache_usage_fraction == 0.4 and m.generation_tokens_total == 123
    with pytest.raises(MetricsUnavailable):
        parse_metrics(METRICS, "missing", 0)
    with pytest.raises(MetricsUnavailable):
        parse_metrics(METRICS + METRICS, "test", 0)


@pytest.mark.parametrize("bad", ["NaN", "+Inf", "-1", "0.5"])
def test_invalid_count_metrics(bad):
    with pytest.raises(MetricsUnavailable):
        parse_metrics(METRICS.replace("} 2\n", "} " + bad + "\n"), "test", 0)


@pytest.fixture
def server():
    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *args):
            pass

        def do_GET(self):
            if self.path == "/metrics":
                body = METRICS.encode()
            elif self.path == "/version":
                body = b'{"version":"0.11.0"}'
            elif self.path == "/health":
                body = b""
            else:
                self.send_response(302)
                self.send_header("Location", "http://127.0.0.1:1/")
                self.end_headers()
                return
            self.send_response(200)
            self.end_headers()
            self.wfile.write(body)

        def do_POST(self):
            assert self.path == "/v1/chat/completions"
            request = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
            assert request["model"] == "test" and request["stream"] is False
            self.send_response(200)
            self.end_headers()
            self.wfile.write(
                json.dumps(
                    {
                        "choices": [{"message": {"content": "hello"}}],
                        "usage": {"completion_tokens": 2},
                    }
                ).encode()
            )

    httpd = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    thread = threading.Thread(target=httpd.serve_forever, daemon=True)
    thread.start()
    try:
        yield f"http://127.0.0.1:{httpd.server_port}"
    finally:
        httpd.shutdown()
        httpd.server_close()
        thread.join()


def test_http_contract_inference_telemetry_and_no_redirects(server):
    adapter = VLLMAdapter(server, "test")
    assert adapter.health() and adapter.version() == "0.11.0"
    assert adapter.collect(0).waiting == 5
    assert adapter.chat([{"role": "user", "content": "hello"}]).text == "hello"
    with pytest.raises(RuntimeHTTPError, match="302"):
        adapter.transport.request("GET", "/redirect")
    with pytest.raises(NotImplementedError):
        adapter.resize(2)


@pytest.mark.parametrize(
    "url",
    [
        "file:///tmp/secret",
        "http://user:pass@localhost",
        "http://localhost?token=x",
        "http://localhost/#x",
    ],
)
def test_invalid_service_urls(url):
    with pytest.raises(ValueError):
        HTTPTransport(url)
