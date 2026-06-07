from __future__ import annotations

import re
from typing import Any

from k_search.tasks.task_base import EvalResult


_MARKER_RE = re.compile(r"^\s*\[([^\]]+)\]\s*$")
_ANSI_RE = re.compile(r"\x1b\[[0-9;?]*[A-Za-z]")
_ABS_PATH_RE = re.compile(r"(?:file://)?/(?:tmp|mnt|home)/[^\s`'\"<>)]*")
_SPINNER_RE = re.compile(r"(?:[|/\\-]\x08)+")

_NOISE_TOKENS = (
    "Permission mismatch",
    "profiler.py: Start parsing profiling data",
    "CANN profiling data parsed",
    "All profiling data parsed",
    "All profiling data",
    "profiling data parsed",
    "结果已写入",
)


def _status(eval_result: EvalResult | None) -> str:
    return str(getattr(eval_result, "status", "") or "").strip().lower()


def _ms_to_us(value: Any) -> float | None:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    return round(float(value) * 1000.0, 6)


def _split_marked_sections(text: str) -> dict[str, list[str]]:
    sections: dict[str, list[str]] = {}
    current = "unmarked"
    sections[current] = []
    for line in str(text or "").splitlines():
        match = _MARKER_RE.match(line)
        if match:
            current = match.group(1).strip().lower()
            sections.setdefault(current, [])
            continue
        sections.setdefault(current, []).append(line)
    return sections


def _section_text(text: str, markers: tuple[str, ...]) -> str:
    sections = _split_marked_sections(text)
    selected: list[str] = []
    for key, lines in sections.items():
        if key == "unmarked":
            continue
        if any(marker in key for marker in markers):
            selected.extend(lines)
    if selected:
        return "\n".join(selected)
    return str(text or "")


def _clean_llm_log(text: str, *, max_chars: int = 4000) -> str:
    out_lines: list[str] = []
    text = _ANSI_RE.sub("", str(text or ""))
    text = _SPINNER_RE.sub("", text)
    for raw_line in text.splitlines():
        line = raw_line.strip("\r")
        if not line.strip():
            continue
        if any(token in line for token in _NOISE_TOKENS):
            continue
        line = _ABS_PATH_RE.sub("<PROJECT_PATH>", line)
        if "[workdir]" in line.lower():
            continue
        out_lines.append(line)
    cleaned = "\n".join(out_lines).strip()
    if len(cleaned) > max_chars:
        cleaned = cleaned[: max(0, max_chars - 24)].rstrip() + "\n[diagnostic truncated]"
    return cleaned


def _base_summary(status: str, *, has_prior_candidate_eval: bool) -> dict[str, Any]:
    return {
        "schema_version": 1,
        "eval_context_status": status,
        "has_prior_candidate_eval": has_prior_candidate_eval,
    }


def _perf_value_us(metrics: dict[str, Any], *keys: str) -> float | None:
    for key in keys:
        value = metrics.get(key)
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            continue
        if key.endswith("_us") or key.endswith("latency_us"):
            return round(float(value), 6)
        return _ms_to_us(value)
    return None


def _performance_summary(eval_result: EvalResult) -> dict[str, Any]:
    metrics = getattr(eval_result, "metrics", None)
    metrics = metrics if isinstance(metrics, dict) else {}
    perf: dict[str, Any] = {
        "mean_latency_us": _ms_to_us(getattr(eval_result, "latency_ms", None)),
        "min_latency_us": _perf_value_us(metrics, "min_latency_us", "min_latency_ms"),
        "max_latency_us": _perf_value_us(metrics, "max_latency_us", "max_latency_ms"),
        "num_runs": metrics.get("num_runs"),
        "speedup_vs_original_baseline": getattr(eval_result, "mean_vs_baseline_factor", None),
        "speedup_vs_parent": metrics.get("speedup_vs_parent"),
        "score": metrics.get("score"),
    }
    return {key: value for key, value in perf.items() if value is not None}


def build_eval_context_for_llm(
    *,
    eval_result: EvalResult | None,
    task: Any,
) -> tuple[dict[str, Any], str]:
    if eval_result is None:
        reference_latency_ms = getattr(task, "reference_latency_ms", None)
        baseline_us = _ms_to_us(reference_latency_ms)
        if baseline_us is not None:
            summary = _base_summary("reference_only", has_prior_candidate_eval=False)
            summary.update(
                {
                    "has_eval_log": False,
                    "compile_passed": None,
                    "correctness_passed": None,
                    "performance_available": False,
                    "baseline": {
                        "reference_latency_us": baseline_us,
                        "source": "task.reference_latency_ms",
                    },
                    "message_for_llm": (
                        "No previous candidate evaluation exists. Use STRATEGY.md and task "
                        "specification. Reference latency is baseline context only."
                    ),
                }
            )
        else:
            summary = _base_summary("no_prior_eval", has_prior_candidate_eval=False)
            summary.update(
                {
                    "has_eval_log": False,
                    "compile_passed": None,
                    "correctness_passed": None,
                    "performance_available": False,
                    "message_for_llm": (
                        "No previous candidate evaluation exists. Use STRATEGY.md and task specification."
                    ),
                }
            )
        return summary, (
            "# Evaluation Log\n\n"
            "No previous candidate evaluation exists. Use EVAL_SUMMARY.json and STRATEGY.md.\n"
        )

    status = _status(eval_result)
    raw_log = str(getattr(eval_result, "log_excerpt", "") or "")

    if status == "compile_failed":
        summary = _base_summary("compile_failed", has_prior_candidate_eval=True)
        summary.update(
            {
                "has_eval_log": True,
                "compile_passed": False,
                "correctness_passed": None,
                "performance_available": False,
                "diagnostic_kind": "compile_error",
            }
        )
        diagnostic = _clean_llm_log(_section_text(raw_log, ("build", "compile")))
        return summary, "# Compile Error\n\n" + (diagnostic or "No compiler diagnostic was provided.") + "\n"

    if status in {"failed", "correctness_failed"}:
        summary = _base_summary("correctness_failed", has_prior_candidate_eval=True)
        summary.update(
            {
                "has_eval_log": True,
                "compile_passed": True,
                "correctness_passed": False,
                "performance_available": False,
                "diagnostic_kind": "correctness_failure",
                "failing_cases": [],
            }
        )
        diagnostic = _clean_llm_log(_section_text(raw_log, ("correctness", "test")))
        return summary, "# Correctness Failure\n\n" + (diagnostic or "No correctness diagnostic was provided.") + "\n"

    if status == "passed":
        summary = _base_summary("performance_measured", has_prior_candidate_eval=True)
        summary.update(
            {
                "has_eval_log": False,
                "compile_passed": True,
                "correctness_passed": True,
                "performance_available": True,
                "diagnostic_kind": "performance_result",
                "performance": _performance_summary(eval_result),
            }
        )
        return summary, (
            "# Evaluation Log\n\n"
            "No detailed failure log is needed.\n"
            "The candidate compiled and passed correctness.\n"
            "Use EVAL_SUMMARY.json for performance metrics.\n"
        )

    if status in {"benchmark_failed", "timeout", "performance_eval_failed"}:
        summary = _base_summary("performance_eval_failed", has_prior_candidate_eval=True)
        summary.update(
            {
                "has_eval_log": True,
                "compile_passed": True,
                "correctness_passed": True,
                "performance_available": False,
                "diagnostic_kind": "performance_eval_failure",
            }
        )
        diagnostic = _clean_llm_log(_section_text(raw_log, ("benchmark", "perf", "timeout")))
        return summary, (
            "# Performance Evaluation Failure\n\n"
            + (diagnostic or "No benchmark diagnostic was provided.")
            + "\n"
        )

    summary = _base_summary(status or "unknown", has_prior_candidate_eval=True)
    summary.update(
        {
            "has_eval_log": True,
            "compile_passed": None,
            "correctness_passed": None,
            "performance_available": False,
            "diagnostic_kind": "evaluation_failure",
        }
    )
    diagnostic = _clean_llm_log(raw_log)
    return summary, "# Evaluation Failure\n\n" + (diagnostic or "No diagnostic was provided.") + "\n"
