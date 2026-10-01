"""The two GitHub readers of bump-plugin-pins: release tag and raw file contents."""

from __future__ import annotations

import io
import json
import urllib.error
import urllib.request

import pytest

from scripts import bump_plugin_pins

pytestmark = pytest.mark.unit


class _Response(io.BytesIO):
    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()
        return False


def _serve(monkeypatch, *, status: int, body: bytes = b"", seen: list | None = None):
    def fake_urlopen(request, timeout=None):
        if seen is not None:
            seen.append(request)
        if status != 200:
            raise urllib.error.HTTPError(request.full_url, status, "nope", {}, None)
        return _Response(body)

    monkeypatch.setattr(urllib.request, "urlopen", fake_urlopen)


def test_latest_release_tag_reads_the_tag_name(monkeypatch):
    seen: list = []
    _serve(monkeypatch, status=200, body=json.dumps({"tag_name": "v0.5.1"}).encode(), seen=seen)
    monkeypatch.setenv("GITHUB_TOKEN", "t0k")
    assert bump_plugin_pins._latest_release_tag("org/repo") == "v0.5.1"
    request = seen[0]
    assert request.full_url == "https://api.github.com/repos/org/repo/releases/latest"
    assert request.get_header("Accept") == "application/vnd.github+json"
    assert request.get_header("Authorization") == "Bearer t0k"


def test_latest_release_tag_is_none_without_a_release(monkeypatch):
    _serve(monkeypatch, status=404)
    assert bump_plugin_pins._latest_release_tag("org/repo") is None


def test_fetch_text_returns_the_raw_file_or_none(monkeypatch):
    seen: list = []
    _serve(monkeypatch, status=200, body=b"name: sam3\n", seen=seen)
    monkeypatch.delenv("GITHUB_TOKEN", raising=False)
    monkeypatch.delenv("GH_TOKEN", raising=False)
    assert bump_plugin_pins._fetch_text("org/repo", "plugin.yaml", "v1") == "name: sam3\n"
    request = seen[0]
    assert request.full_url == "https://api.github.com/repos/org/repo/contents/plugin.yaml?ref=v1"
    assert request.get_header("Accept") == "application/vnd.github.raw"
    assert request.get_header("Authorization") is None
    _serve(monkeypatch, status=404)
    assert bump_plugin_pins._fetch_text("org/repo", "plugin.yaml", "v1") is None


def test_other_http_errors_propagate(monkeypatch):
    _serve(monkeypatch, status=500)
    with pytest.raises(urllib.error.HTTPError):
        bump_plugin_pins._latest_release_tag("org/repo")
    with pytest.raises(urllib.error.HTTPError):
        bump_plugin_pins._fetch_text("org/repo", "plugin.yaml", "v1")
