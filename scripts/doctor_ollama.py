#!/usr/bin/env python3
"""
Diagnostic checks for local Ollama connectivity from Python.
"""

from __future__ import annotations

import argparse
import os
import sys
from typing import Any

import requests
from dotenv import load_dotenv


def _print_ok(message: str) -> None:
    print(f"[OK] {message}")


def _print_warn(message: str) -> None:
    print(f"[WARN] {message}")


def _print_fail(message: str) -> None:
    print(f"[FAIL] {message}")


def _normalize_base_url(base_url: str) -> str:
    return base_url.rstrip("/")


def _get_tags(base_url: str, timeout: float) -> dict[str, Any]:
    url = f"{base_url}/api/tags"
    resp = requests.get(url, timeout=timeout)
    resp.raise_for_status()
    return resp.json()


def _post_generate(base_url: str, model: str, timeout: float) -> dict[str, Any]:
    url = f"{base_url}/api/generate"
    payload = {
        "model": model,
        "prompt": "Reply with exactly: ok",
        "stream": False,
    }
    resp = requests.post(url, json=payload, timeout=timeout)
    resp.raise_for_status()
    return resp.json()


def _ollama_python_chat(base_url: str, model: str) -> str:
    from ollama import Client

    client = Client(host=base_url)
    response = client.chat(
        model=model,
        messages=[{"role": "user", "content": "Reply with exactly: ok"}],
    )
    if isinstance(response, dict):
        return str(response.get("message", {}).get("content", "")).strip()
    return str(response).strip()


def main() -> int:
    load_dotenv()

    parser = argparse.ArgumentParser(description="Validate Ollama Python integration.")
    parser.add_argument(
        "--base-url",
        default=os.getenv("OLLAMA_BASE_URL", "http://localhost:11434"),
        help="Ollama base URL (default from OLLAMA_BASE_URL or http://localhost:11434)",
    )
    parser.add_argument(
        "--model",
        default=os.getenv("OLLAMA_MODEL", "qwen3:8b"),
        help="Model name (default from OLLAMA_MODEL or qwen3:8b)",
    )
    parser.add_argument(
        "--timeout",
        type=float,
        default=15.0,
        help="HTTP timeout in seconds (default: 15)",
    )
    args = parser.parse_args()

    base_url = _normalize_base_url(args.base_url)
    model = args.model
    timeout = args.timeout
    failures = 0

    print("=== Ollama Doctor ===")
    print(f"Python: {sys.version.split()[0]}")
    print(f"Executable: {sys.executable}")
    print(f"Ollama URL: {base_url}")
    print(f"Ollama model: {model}")
    print()

    if ".venv" in sys.executable.lower():
        _print_ok("Python executable appears to come from .venv.")
    else:
        failures += 1
        _print_fail(
            "Python executable does not include '.venv'. Activate .venv and re-run."
        )
        _print_warn("Expected: .\\.venv\\Scripts\\Activate.ps1")

    try:
        import ollama  # noqa: F401

        _print_ok("`import ollama` succeeded.")
    except Exception as exc:
        failures += 1
        _print_fail(f"`import ollama` failed: {exc}")
        _print_warn("Run: pip install -r requirements.txt")

    tags_payload: dict[str, Any] | None = None
    available_models: list[str] = []
    try:
        tags_payload = _get_tags(base_url=base_url, timeout=timeout)
        models = tags_payload.get("models", [])
        for m in models:
            name = m.get("name") or m.get("model")
            if name:
                available_models.append(str(name))
        _print_ok(f"Reached {base_url}/api/tags")
        if available_models:
            print("Available models:")
            for name in available_models:
                print(f"  - {name}")
        else:
            _print_warn("No models returned by /api/tags.")
    except requests.exceptions.ConnectionError as exc:
        failures += 1
        _print_fail(f"Could not connect to Ollama server at {base_url}: {exc}")
        _print_warn("Ensure Ollama Desktop is running and listening on the configured URL.")
    except requests.exceptions.Timeout:
        failures += 1
        _print_fail(f"Timeout calling {base_url}/api/tags")
        _print_warn("Check OLLAMA_BASE_URL and local firewall settings.")
    except requests.exceptions.HTTPError as exc:
        failures += 1
        status = exc.response.status_code if exc.response is not None else "unknown"
        _print_fail(f"HTTP error from /api/tags (status={status})")
        _print_warn("Possible wrong base URL. Expected something like http://localhost:11434")
    except ValueError as exc:
        failures += 1
        _print_fail(f"Invalid JSON from /api/tags: {exc}")
        _print_warn("This can indicate a wrong base URL (not an Ollama server).")
    except requests.exceptions.RequestException as exc:
        failures += 1
        _print_fail(f"Request to /api/tags failed: {exc}")

    if tags_payload is not None:
        if available_models and model not in available_models:
            failures += 1
            _print_fail(f"Configured model '{model}' is not in /api/tags.")
            _print_warn("Pull the model in Ollama Desktop, then re-run this script.")

        try:
            gen = _post_generate(base_url=base_url, model=model, timeout=timeout)
            text = str(gen.get("response", "")).strip()
            if text:
                _print_ok("POST /api/generate succeeded.")
                print(f"Generate sample: {text[:120]}")
            else:
                _print_warn("POST /api/generate returned empty response text.")
        except requests.exceptions.HTTPError as exc:
            failures += 1
            status = exc.response.status_code if exc.response is not None else "unknown"
            body = ""
            if exc.response is not None:
                body = exc.response.text[:250]
            _print_fail(f"/api/generate failed (status={status}).")
            if "not found" in body.lower() and "model" in body.lower():
                _print_warn(f"Model '{model}' is likely not pulled yet.")
            elif status == 404:
                _print_warn("Wrong endpoint/base URL. Check OLLAMA_BASE_URL.")
            else:
                _print_warn(f"Response body: {body}")
        except requests.exceptions.ConnectionError as exc:
            failures += 1
            _print_fail(f"Connection error during /api/generate: {exc}")
        except requests.exceptions.Timeout:
            failures += 1
            _print_fail("Timeout during /api/generate.")
        except requests.exceptions.RequestException as exc:
            failures += 1
            _print_fail(f"Request error during /api/generate: {exc}")

        try:
            chat_text = _ollama_python_chat(base_url=base_url, model=model)
            if chat_text:
                _print_ok("ollama Python package chat() succeeded.")
                print(f"Chat sample: {chat_text[:120]}")
            else:
                _print_warn("ollama Python chat() returned empty text.")
        except Exception as exc:
            failures += 1
            _print_fail(f"ollama Python package chat() failed: {exc}")
            _print_warn("Check OLLAMA_BASE_URL, OLLAMA_MODEL, and model availability.")

    print()
    if failures == 0:
        _print_ok("All checks passed.")
        return 0

    _print_fail(f"Completed with {failures} issue(s).")
    print("Next steps:")
    print("  1) Activate venv: .\\.venv\\Scripts\\Activate.ps1")
    print("  2) Install deps: pip install -r requirements.txt")
    print("  3) Confirm Ollama Desktop is running")
    print("  4) Confirm model exists in Ollama Desktop (or update OLLAMA_MODEL)")
    print("  5) Re-run: python scripts\\doctor_ollama.py")
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
