from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from .planning import generate_response


def _load_json(path: str) -> dict:
    file_path = Path(path)
    with file_path.open("r", encoding="utf-8-sig") as f:
        data = json.load(f)
    if not isinstance(data, dict):
        raise ValueError(f"Expected JSON object in {path}, got {type(data).__name__}")
    return data


def _configure_console_utf8() -> None:
    for stream_name in ("stdout", "stderr"):
        stream = getattr(sys, stream_name, None)
        if hasattr(stream, "reconfigure"):
            try:
                stream.reconfigure(encoding="utf-8", errors="replace")
            except Exception:
                pass


def main() -> None:
    _configure_console_utf8()

    parser = argparse.ArgumentParser(description="Planning Engine demo for Credit Scoring Assistant")
    parser.add_argument("--user", required=True, help="Path to user input JSON")
    parser.add_argument("--model", required=True, help="Path to model output JSON")
    parser.add_argument("--shap", required=True, help="Path to SHAP Style-1 JSON")
    args = parser.parse_args()

    user_input = _load_json(args.user)
    model_output = _load_json(args.model)
    shap_json = _load_json(args.shap)

    # Integration note: pass rag_lookup=function_here when wiring into FastAPI endpoint.
    response = generate_response(
        user_input=user_input,
        model_output=model_output,
        shap_json=shap_json,
        rag_lookup=None,
    )

    print(f"Mode: {response.get('mode')}")
    print(response.get("result_th", ""))

    if response.get("mode") == "improvement_plan":
        plan = response.get("plan", {}) or {}
        actions = plan.get("actions", []) or []
        if actions:
            print("\nActions")
            for idx, action in enumerate(actions, start=1):
                print(f"{idx}. {action.get('title_th', '-')}")
                print(f"   - why: {action.get('why_th', '-')}")
                print(f"   - how: {action.get('how_th', '-')}")
                print(f"   - evidence_confidence: {action.get('evidence_confidence', '-')}")
                evidence = action.get("evidence", []) or []
                if evidence:
                    print(
                        f"   - evidence: {evidence[0].get('source_title', 'N/A')} "
                        f"({evidence[0].get('category', 'N/A')}, score={evidence[0].get('score', 0)})"
                    )

    print("\nJSON Output")
    print(json.dumps(response, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
