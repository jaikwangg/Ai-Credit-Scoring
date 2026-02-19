"""
Unit tests for src.query Ollama error handling.
These tests mock network/model behavior and never require a live Ollama server.
"""

import sys
import importlib
from pathlib import Path
from unittest.mock import Mock, patch

import httpx
import pytest

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src import query as query_module


def test_settings_prefers_ollama_model_env(monkeypatch):
    monkeypatch.setenv("OLLAMA_MODEL", "qwen3:8b")
    monkeypatch.delenv("LLM_MODEL", raising=False)
    import src.settings as settings_module

    importlib.reload(settings_module)
    assert settings_module.OLLAMA_MODEL == "qwen3:8b"


def test_settings_falls_back_to_legacy_llm_model_env(monkeypatch):
    monkeypatch.delenv("OLLAMA_MODEL", raising=False)
    monkeypatch.setenv("LLM_MODEL", "llama3.1:8b")
    import src.settings as settings_module

    importlib.reload(settings_module)
    assert settings_module.OLLAMA_MODEL == "llama3.1:8b"


def test_get_engine_maps_connect_error_to_friendly_message():
    with patch.object(
        query_module,
        "Ollama",
        side_effect=httpx.ConnectError("connect failed"),
    ):
        with pytest.raises(RuntimeError, match="Cannot connect to Ollama"):
            query_module.get_engine()


def test_explain_case_maps_timeout_to_friendly_message():
    fake_engine = Mock()
    fake_engine.query.side_effect = httpx.ReadTimeout("timed out")

    with patch.object(query_module, "get_engine", return_value=fake_engine):
        with pytest.raises(RuntimeError, match="timed out"):
            query_module.explain_case("hello", {"decision": {}})


def test_explain_case_maps_404_to_model_not_found_message():
    request = httpx.Request("POST", "http://localhost:11434/api/generate")
    response = httpx.Response(404, request=request)
    http_error = httpx.HTTPStatusError("not found", request=request, response=response)

    fake_engine = Mock()
    fake_engine.query.side_effect = http_error

    with patch.object(query_module, "get_engine", return_value=fake_engine):
        with pytest.raises(RuntimeError, match="was not found"):
            query_module.explain_case("hello", {"decision": {}})
