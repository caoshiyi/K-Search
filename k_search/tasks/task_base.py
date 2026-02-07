"""Task-agnostic types and interfaces used by kernel generators.

This module intentionally contains **no** flashinfer-bench specific imports.
Tasks (flashinfer-bench, gpu_mode, etc.) should return this shared `EvalResult`
so generators can stay evaluator-agnostic.
"""

from __future__ import annotations

import hashlib
from dataclasses import asdict, dataclass, field
from enum import Enum
from typing import Any, Dict, Optional, Protocol


@dataclass
class EvalResult:
    """Minimal evaluation datapoint used by generators and the world model (no Trace dependency)."""

    status: str
    latency_ms: float | None = None
    reference_latency_ms: float | None = None
    # Preferred objective when a baseline exists: mean vs_base (>1 is better).
    mean_vs_baseline_factor: float | None = None
    speedup_factor: float | None = None
    log_excerpt: str = ""

    # Task-defined optional metrics. Generators should prefer methods below over raw fields.
    metrics: dict[str, Any] = field(default_factory=dict)

    def to_dict(
        self,
        *,
        include_log_excerpt: bool = False,
        max_str_chars: int = 2000,
        max_log_chars: int = 800,
    ) -> Dict[str, Any]:
        """
        Generic JSON-friendly serialization.

        Goal: avoid hardcoding field lists at call sites. When EvalResult evolves, the WM serialization
        stays correct automatically.
        """

        def _sanitize(v: Any) -> Any:
            if v is None or isinstance(v, (int, float, bool)):
                return v
            if isinstance(v, str):
                s = v
                if max_str_chars and len(s) > int(max_str_chars):
                    s = s[: int(max_str_chars)] + "...<truncated>..."
                return s
            if isinstance(v, (list, tuple)):
                out = []
                for x in v[:200]:
                    out.append(_sanitize(x))
                return out
            if isinstance(v, dict):
                out: dict[str, Any] = {}
                # Stable ordering not required for dict, but keep key count bounded.
                n = 0
                for k, x in v.items():
                    if n >= 200:
                        break
                    out[str(k)] = _sanitize(x)
                    n += 1
                return out
            # Fallback: stringify unknown objects.
            try:
                return _sanitize(str(v))
            except Exception:
                return None

        d = asdict(self)
        if not include_log_excerpt:
            d.pop("log_excerpt", None)
        else:
            try:
                le = str(d.get("log_excerpt", "") or "")
                if max_log_chars and len(le) > int(max_log_chars):
                    d["log_excerpt"] = le[: int(max_log_chars)] + "...<truncated>..."
            except Exception:
                d["log_excerpt"] = ""
        return _sanitize(d)

    def is_passed(self) -> bool:
        try:
            return str(self.status or "").strip().lower() == "passed"
        except Exception:
            return False

    def status_code(self) -> int:
        """Numeric status code (1=passed, 0=not passed, -1=seeded/unknown)."""
        st = str(self.status or "").strip().lower()
        if st == "passed":
            return 1
        if st in ("seeded", "unknown", ""):
            return -1
        return 0

    def score(self) -> float:
        """Comparable scalar score (higher is better)."""
        try:
            sc = self.metrics.get("score") if isinstance(self.metrics, dict) else None
            if isinstance(sc, (int, float)):
                return float(sc)
        except Exception:
            pass
        if not self.is_passed():
            return -1.0
        try:
            vb = self.mean_vs_baseline_factor
            if isinstance(vb, (int, float)) and float(vb) > 0:
                return float(vb)
        except Exception:
            pass
        try:
            sp = self.speedup_factor
            if isinstance(sp, (int, float)) and float(sp) > 0:
                return float(sp)
        except Exception:
            pass
        try:
            lat = self.latency_ms
            if isinstance(lat, (int, float)) and float(lat) > 0:
                return 1.0 / float(lat)
        except Exception:
            pass
        return -1.0

    def perf_summary_lines(self, *, prefix: str) -> list[str]:
        """Task-agnostic perf summary lines for prompts (only includes fields that exist)."""
        p = str(prefix or "").strip()
        pre = (p + "_") if p else ""
        lines: list[str] = []
        try:
            if self.is_passed() and isinstance(self.latency_ms, (int, float)) and float(self.latency_ms) > 0:
                lines.append(f"- {pre}mean_latency_ms: {float(self.latency_ms):.4f}")
        except Exception:
            pass
        try:
            if (
                self.is_passed()
                and isinstance(self.mean_vs_baseline_factor, (int, float))
                and float(self.mean_vs_baseline_factor) > 0
            ):
                lines.append(f"- {pre}vs_baseline: {float(self.mean_vs_baseline_factor):.3f}x")
        except Exception:
            pass
        try:
            if self.is_passed() and isinstance(self.speedup_factor, (int, float)) and float(self.speedup_factor) > 0:
                lines.append(f"- {pre}speedup_vs_ref: {float(self.speedup_factor):.3f}x")
        except Exception:
            pass
        try:
            sc = self.score()
            if self.is_passed() and isinstance(sc, (int, float)) and float(sc) > 0:
                lines.append(f"- {pre}score: {float(sc):.4f}")
        except Exception:
            pass
        return lines


class SupportedLanguages(str, Enum):
    PYTHON = "python"
    TRITON = "triton"
    CUDA = "cuda"
    CPP = "cpp"


@dataclass
class SourceFile:
    path: str
    content: str


@dataclass
class BuildSpec:
    language: SupportedLanguages
    target_hardware: list[str]
    entry_point: str
    dependencies: list[str] = field(default_factory=list)


@dataclass
class Solution:
    """
    Task-agnostic solution container.

    Tasks are responsible for converting this into their backend-specific solution types.
    """

    name: str
    definition: str
    author: str
    spec: BuildSpec
    sources: list[SourceFile]
    description: Optional[str] = None

    def get_entry_path(self) -> str:
        return str(self.spec.entry_point.split("::")[0])

    def get_entry_symbol(self) -> str:
        return str(self.spec.entry_point.split("::")[-1])

    def get_entry_source(self) -> Optional[SourceFile]:
        entry_path = self.get_entry_path()
        for src in self.sources or []:
            if src.path == entry_path:
                return src
        return None

    def hash(self) -> str:
        """
        Deterministic content hash (matches flashinfer-bench's intent; sha1 over key behavior fields).
        """
        h = hashlib.sha1()
        deps = list(self.spec.dependencies or [])
        for s in (
            str(self.name or ""),
            str(self.definition or ""),
            str(self.spec.language.value if isinstance(self.spec.language, SupportedLanguages) else self.spec.language),
            str(self.spec.entry_point or ""),
            *[str(d or "") for d in deps],
            *(part for src in (self.sources or []) for part in (str(src.path or ""), str(src.content or ""))),
        ):
            h.update(str(s).encode())
        return h.hexdigest()

    def to_dict(self) -> Dict[str, Any]:
        # JSON-friendly dict; keep enum as string.
        try:
            return {
                "name": self.name,
                "definition": self.definition,
                "author": self.author,
                "description": self.description,
                "spec": {
                    "language": str(self.spec.language.value if isinstance(self.spec.language, SupportedLanguages) else self.spec.language),
                    "target_hardware": list(self.spec.target_hardware or []),
                    "entry_point": self.spec.entry_point,
                    "dependencies": list(self.spec.dependencies or []),
                },
                "sources": [{"path": sf.path, "content": sf.content} for sf in (self.sources or [])],
            }
        except Exception:
            # Fallback: dataclasses.asdict, then coerce language.
            d = asdict(self)
            try:
                d["spec"]["language"] = str(d.get("spec", {}).get("language", ""))
            except Exception:
                pass
            return d  # type: ignore[return-value]


def code_from_solution(language: str, solution: Solution) -> tuple[Any, Any]:
    """
    Convert a task_base.Solution back into (current_code, current_raw_code) expected by generators:
    - For CUDA: return ({path: content} dict, reconstructed XML blocks string)
    - For Triton/Python: return (entry_source_content, entry_source_content)
    """
    lang = str(language or "").strip().lower()
    if lang == "cuda":
        code_dict: Dict[str, str] = {sf.path: sf.content for sf in (solution.sources or [])}

        def _xml_block(tag: str, name: str, content: str) -> str:
            return f"<{tag} name=\"{name}\">\n{content}\n</{tag}>"

        h = code_dict.get("kernel.h", "")
        cu = code_dict.get("kernel.cu", "")
        cpp = code_dict.get("main.cpp", "")
        parts: list[str] = []
        if h:
            parts.append(_xml_block("header_file", "kernel.h", h))
        if cu:
            parts.append(_xml_block("cuda_file", "kernel.cu", cu))
        if cpp:
            parts.append(_xml_block("cpp_file", "main.cpp", cpp))
        current_raw_code = "\n\n".join(parts) if parts else ""
        return code_dict, current_raw_code

    entry_src = solution.get_entry_source()
    content = entry_src.content if entry_src else ""
    return content, content


class Task(Protocol):
    """
    Task-agnostic protocol that kernel generators depend on.

    Design intent:
    - Keep this interface small but complete for the generators in `kernel_generator/`.
    - Task implementations may expose additional methods for scripts/CLIs (e.g. final eval),
      but generators should not depend on backend-specific dataset objects.
    """

    @property
    def name(self) -> str: ...

    def get_definition_text(self) -> str: ...

    def get_solution(self, solution_name: str) -> Solution | None: ...

    def run_benchmark(self, *, solution: Solution, config: Any = None, dump_traces: bool = False, round_num: int | None = None) -> EvalResult: ...

    # World-model generators / prompt formatting.
    def code_for_world_model_from_raw(self, *, raw: Any, language: str) -> str: ...

    def seed_eval_for_base_solution(self, *, base_solution: Solution, config: Any = None) -> EvalResult: ...

    # Logging / metadata (must be JSON-friendly primitives only).
    def get_config_for_logging(self) -> Dict[str, Any]: ...

    # Optional (not required by the Protocol; generators use getattr fallbacks):
    # - get_baseline_targets_text() -> str
    # - has_last_round_feedback_trace() -> bool
    # - get_last_round_trace_logs_for_prompt() -> str
    # - get_last_round_passed_count() -> int
    # - get_last_round_total_workloads() -> int


