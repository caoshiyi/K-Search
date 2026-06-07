from __future__ import annotations

from pathlib import Path
from typing import Any


GLOBAL_FLOW_POLICY_PATTERNS = (
    "Subagent usage policy:",
    "Initial codegen flow agents",
    "Eval-failure repair flow agents",
    "Required native subagent flow",
    "code-reader -> designer -> codegen -> reviewer",
)

RAW_BENCHMARK_PATTERNS = (
    "[benchmark stderr]",
    "[benchmark stdout]",
)

PROFILING_NOISE_PATTERNS = (
    "profiler.py: Start parsing profiling data",
    "CANN profiling data parsed",
)

TRUNCATION_PATTERNS = (
    "[truncated strategy markdown]",
    "[truncated for agentic prompt budget]",
)

ABSOLUTE_PATH_PATTERNS = (
    "/tmp/",
    "/mnt/",
    "/home/",
    "file://",
)


class PromptHygieneError(RuntimeError):
    def __init__(self, hygiene: dict[str, Any]) -> None:
        failures = [
            name
            for name, failed in hygiene.items()
            if name.startswith("contains_") and bool(failed)
        ]
        message = "prompt hygiene violation"
        if failures:
            readable = ", ".join(name.removeprefix("contains_").replace("_", " ") for name in failures)
            message = f"{message}: {readable}"
        super().__init__(message)
        self.hygiene = dict(hygiene)


def _known_path_tokens(known_paths: list[str | Path] | tuple[str | Path, ...] | None) -> list[str]:
    tokens: list[str] = []
    for item in known_paths or ():
        text = str(item or "").strip()
        if not text:
            continue
        try:
            text = str(Path(text).expanduser().resolve())
        except Exception:
            pass
        if text and text not in tokens:
            tokens.append(text)
    return tokens


def check_prompt_hygiene(
    prompt: str,
    *,
    known_paths: list[str | Path] | tuple[str | Path, ...] | None = None,
) -> dict[str, Any]:
    text = str(prompt or "")
    known_path_hits = [token for token in _known_path_tokens(known_paths) if token and token in text]
    contains_absolute_path = bool(known_path_hits) or any(token in text for token in ABSOLUTE_PATH_PATTERNS)
    hygiene = {
        "contains_absolute_path": contains_absolute_path,
        "contains_raw_benchmark_stderr": any(token in text for token in RAW_BENCHMARK_PATTERNS),
        "contains_permission_mismatch": "Permission mismatch" in text,
        "contains_profiling_noise": any(token in text for token in PROFILING_NOISE_PATTERNS),
        "contains_truncated_strategy": any(token in text for token in TRUNCATION_PATTERNS),
        "contains_global_flow_policy": any(token in text for token in GLOBAL_FLOW_POLICY_PATTERNS),
    }
    hygiene["known_path_hits"] = known_path_hits
    hygiene["has_blocking_violation"] = any(
        bool(value)
        for key, value in hygiene.items()
        if key.startswith("contains_")
    )
    return hygiene


def check_prompt_hygiene_or_raise(
    prompt: str,
    *,
    known_paths: list[str | Path] | tuple[str | Path, ...] | None = None,
) -> dict[str, Any]:
    hygiene = check_prompt_hygiene(prompt, known_paths=known_paths)
    if hygiene["has_blocking_violation"]:
        raise PromptHygieneError(hygiene)
    return hygiene
