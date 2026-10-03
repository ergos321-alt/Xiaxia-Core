"""Experiment 01: the existing Xiaxia GET /tabs endpoint, with no browser controls."""

import json
from pathlib import Path
import time

import httpx


def register_browser_tabs(endpoint, token_file, timeout_seconds, http_trace=None):
    from tools.registry import registry

    def browser_tabs(args, **kwargs):
        if args:
            return json.dumps({"error": "browser_tabs accepts no arguments"})
        try:
            token = next(
                line.split("=", 1)[1].strip().strip("\"'")
                for line in Path(token_file).read_text(encoding="utf-8-sig").splitlines()
                if line.startswith("BROWSER_AUTH_TOKEN=")
            )
            with httpx.Client(timeout=timeout_seconds, trust_env=False,
                              follow_redirects=False) as client:
                started = time.perf_counter()
                response = client.get(endpoint, headers={"Authorization": f"Bearer {token}"})
                if http_trace is not None:
                    http_trace({"method": "GET", "endpoint": endpoint,
                                "status": response.status_code,
                                "elapsed_ms": round((time.perf_counter() - started) * 1000, 3)})
                response.raise_for_status()
                return json.dumps(response.json(), ensure_ascii=False)
        except (OSError, StopIteration, ValueError, httpx.HTTPError) as exc:
            return json.dumps({"error": type(exc).__name__, "operation": "GET /tabs"})

    registry.register(
        name="browser_tabs",
        toolset="xiaxia_experiment_readonly",
        schema={
            "name": "browser_tabs",
            "description": "Read the titles, URLs and identifiers of tabs currently open in the user's Xiaxia Chrome browser.",
            "parameters": {"type": "object", "properties": {}, "additionalProperties": False},
        },
        handler=browser_tabs,
    )
