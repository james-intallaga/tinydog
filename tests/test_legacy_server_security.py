from __future__ import annotations

import http.client
import json
import threading
from urllib.parse import urlparse

import pytest

from mini_arithmetic.serve import (
    LOCAL_API_TOKEN_HEADER,
    MAX_JSON_REQUEST_BYTES,
    _LocalOnlyRequestHandler,
    _loopback_host_header,
    _loopback_origin,
    _make_loopback_http_server,
    _validated_bind_host,
)


def _probe_handler(api_token: str = "") -> type[_LocalOnlyRequestHandler]:
    configured_token = api_token

    class ProbeHandler(_LocalOnlyRequestHandler):
        expected_local_api_token = configured_token

        def do_GET(self) -> None:
            if not self._authorized(expect_json=False):
                return
            self._json(200, {"ok": True})

        def do_POST(self) -> None:
            parsed = urlparse(self.path)
            binary = parsed.path in {"/multimodal/frame", "/multimodal/audio"}
            if not self._authorized(expect_json=not binary):
                return
            try:
                if binary:
                    body = self._binary_body(8)
                    self._json(202, {"bytes": len(body)})
                else:
                    self._json(200, {"payload": self._payload()})
            except (ValueError, json.JSONDecodeError) as error:
                self._json(400, {"error": str(error)})

    return ProbeHandler


def _request(
    port: int,
    method: str,
    path: str,
    *,
    body: bytes | None = None,
    headers: dict[str, str] | None = None,
) -> tuple[int, dict[str, object]]:
    connection = http.client.HTTPConnection("127.0.0.1", port, timeout=5)
    try:
        connection.request(method, path, body=body, headers=headers or {})
        response = connection.getresponse()
        payload = json.loads(response.read())
        return response.status, payload
    finally:
        connection.close()


def _start_server(*, api_token: str = "") -> tuple[object, threading.Thread, int]:
    server = _make_loopback_http_server("127.0.0.1", 0, _probe_handler(api_token))
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    return server, thread, int(server.server_address[1])


@pytest.mark.parametrize("host", ["0.0.0.0", "::", "192.168.1.10", "example.com"])
def test_legacy_server_rejects_non_loopback_bind(host: str) -> None:
    with pytest.raises(ValueError, match="bind only"):
        _make_loopback_http_server(host, 0, _probe_handler())


@pytest.mark.parametrize(
    ("host", "expected"),
    [
        ("localhost", "localhost"),
        ("LOCALHOST", "localhost"),
        ("127.0.0.1", "127.0.0.1"),
    ],
)
def test_legacy_server_accepts_only_explicit_bind_names(
    host: str, expected: str
) -> None:
    assert _validated_bind_host(host) == expected


@pytest.mark.parametrize(
    "authority",
    [
        "localhost",
        "localhost:8765",
        "127.0.0.1",
        "127.23.45.67:8765",
        "[::1]:8765",
    ],
)
def test_legacy_host_validation_accepts_literal_loopback(authority: str) -> None:
    assert _loopback_host_header(authority)


@pytest.mark.parametrize(
    "authority",
    [
        "",
        "0.0.0.0:8765",
        "192.168.1.10",
        "localhost.example",
        "localhost@evil.example",
        "localhost/path",
        "[::1",
    ],
)
def test_legacy_host_validation_rejects_untrusted_authority(
    authority: str,
) -> None:
    assert not _loopback_host_header(authority)


@pytest.mark.parametrize(
    "origin",
    [
        "http://localhost:3000",
        "https://127.0.0.1",
        "http://127.2.3.4:8080",
        "https://[::1]:3000",
    ],
)
def test_legacy_origin_validation_accepts_loopback(origin: str) -> None:
    assert _loopback_origin(origin)


@pytest.mark.parametrize(
    "origin",
    [
        "https://evil.example",
        "https://localhost.example",
        "http://localhost@evil.example",
        "http://localhost:3000/path",
        "http://localhost:3000?secret=value",
        "file://localhost",
        "http://[::1",
    ],
)
def test_legacy_origin_validation_rejects_non_loopback_or_malformed_origin(
    origin: str,
) -> None:
    assert not _loopback_origin(origin)


def test_legacy_http_requests_require_loopback_host_header() -> None:
    server, thread, port = _start_server()
    try:
        status, _ = _request(
            port,
            "GET",
            "/status",
            headers={"Host": "evil.example"},
        )
        assert status == 403

        status, _ = _request(
            port,
            "GET",
            "/status",
            headers={"Host": f"localhost:{port}"},
        )
        assert status == 200
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=5)


def test_legacy_http_token_is_required_and_compared_in_constant_time(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    comparisons: list[tuple[str, str]] = []

    def compare_digest(supplied: str, expected: str) -> bool:
        comparisons.append((supplied, expected))
        return supplied == expected

    monkeypatch.setattr("mini_arithmetic.serve.hmac.compare_digest", compare_digest)
    server, thread, port = _start_server(api_token="launch-capability")
    try:
        status, _ = _request(port, "GET", "/status")
        assert status == 403
        status, _ = _request(
            port,
            "GET",
            "/status",
            headers={LOCAL_API_TOKEN_HEADER: "wrong"},
        )
        assert status == 403
        status, _ = _request(
            port,
            "GET",
            "/status",
            headers={
                LOCAL_API_TOKEN_HEADER: "launch-capability",
                "Origin": "https://evil.example",
            },
        )
        assert status == 200
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=5)

    assert comparisons == [
        ("", "launch-capability"),
        ("wrong", "launch-capability"),
        ("launch-capability", "launch-capability"),
    ]


def test_legacy_http_without_token_rejects_non_loopback_browser_origin() -> None:
    server, thread, port = _start_server()
    body = json.dumps({"message": "private lesson"}).encode()
    try:
        status, _ = _request(
            port,
            "POST",
            "/chat",
            body=body,
            headers={
                "Content-Type": "application/json",
                "Origin": "https://evil.example",
            },
        )
        assert status == 403

        status, _ = _request(
            port,
            "POST",
            "/chat",
            body=body,
            headers={
                "Content-Type": "application/json",
                "Origin": "http://localhost:3000",
            },
        )
        assert status == 200

        status, _ = _request(
            port,
            "POST",
            "/chat",
            body=body,
            headers={"Content-Type": "application/json"},
        )
        assert status == 200
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=5)


def test_legacy_json_mutations_require_type_and_bounded_object_body() -> None:
    server, thread, port = _start_server()
    try:
        status, _ = _request(
            port,
            "POST",
            "/chat",
            body=b"{}",
            headers={"Content-Type": "text/plain"},
        )
        assert status == 415

        status, _ = _request(
            port,
            "POST",
            "/chat",
            body=b"[]",
            headers={"Content-Type": "application/json; charset=utf-8"},
        )
        assert status == 400

        status, _ = _request(
            port,
            "POST",
            "/chat",
            body=b"{}",
            headers={
                "Content-Type": "application/json",
                "Content-Length": str(MAX_JSON_REQUEST_BYTES + 1),
            },
        )
        assert status == 400

        status, payload = _request(
            port,
            "POST",
            "/chat",
            body=b'{"message":"hello"}',
            headers={"Content-Type": "application/json"},
        )
        assert status == 200
        assert payload == {"payload": {"message": "hello"}}
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=5)


def test_legacy_media_endpoint_preserves_bounded_binary_content_type() -> None:
    server, thread, port = _start_server()
    try:
        status, payload = _request(
            port,
            "POST",
            "/multimodal/frame",
            body=b"jpeg",
            headers={"Content-Type": "image/jpeg"},
        )
        assert status == 202
        assert payload == {"bytes": 4}

        status, _ = _request(
            port,
            "POST",
            "/multimodal/audio",
            body=b"123456789",
            headers={"Content-Type": "audio/webm"},
        )
        assert status == 400
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=5)
