"""GPUMode Task implementation (TriMul).

This task is intentionally self-contained and can be wired into generators later.
All GPUMode task utilities (spec/prompt text + evaluation) live under `k_search.tasks.gpu_mode`.

Note: the legacy top-level `gpu_mode/` folder is expected to be removed later.
"""

from __future__ import annotations

from dataclasses import dataclass
import os
from pathlib import Path
from typing import Any, Dict, Optional

from k_search.tasks.task_base import (
    BuildSpec,
    EvalResult,
    Solution,
    SourceFile,
    SupportedLanguages,
    load_ksearch_solution_json,
    solution_from_json_dict,
)
from k_search.tasks.gpu_mode.code_utils import normalize_cuda_sources
from k_search.tasks.gpu_mode.evaluator import evaluate_trimul_submission
from k_search.tasks.gpu_mode.trimul.spec import TRIMUL_SPEC_TEXT_CUDA, TRIMUL_SPEC_TEXT_TRITON
from k_search.tasks.gpu_mode import DEFAULT_TRIMUL_TASK_DIR


@dataclass(frozen=True)
class GpuModeTriMulTaskConfig:
    mode: str = "benchmark"
    keep_tmp: bool = False
    task_dir: Path = DEFAULT_TRIMUL_TASK_DIR
    # How many chars to print to the main log as "[gpumode_trimul] Failure excerpt:".
    max_failure_excerpt_chars: int = int(os.getenv("KSEARCH_GPUMODE_FAILURE_EXCERPT_CHARS", "4000"))


class GpuModeTriMulTask:
    """Task wrapper around GPUMode TriMul evaluation."""

    def __init__(
        self,
        *,
        mode: str = "benchmark",
        keep_tmp: bool = False,
        task_dir: str | Path | None = None,
        artifacts_dir: str | None = None,
        name: str = "gpumode_trimul",
    ) -> None:
        self._name = str(name or "gpumode_trimul")
        self._cfg = GpuModeTriMulTaskConfig(
            mode=str(mode or "benchmark"),
            keep_tmp=bool(keep_tmp),
            task_dir=(Path(task_dir).expanduser().resolve() if task_dir is not None else DEFAULT_TRIMUL_TASK_DIR),
        )
        self._ksearch_artifacts_dir: str | None = (str(artifacts_dir) if artifacts_dir is not None else None)
        self._solutions: dict[str, Solution] = {}
        # Last-round cache for prompt feedback (best-effort; generator reads via getattr).
        self._last_round_trace_logs_for_prompt: str = ""
        self._last_round_passed_count: int = 0
        self._last_round_total_workloads: int = 0
        self._last_round_summary_line: str = ""

    @property
    def name(self) -> str:
        return self._name

    def get_definition_text(self, language: str | None = None) -> str:
        """
        Return the language-specific task specification text.

        Note: the Task Protocol requires `get_definition_text()` with no args; we keep
        `language` optional for convenience in scripts/CLIs.
        """
        lang = str(language or "").strip().lower()
        if not lang:
            lang = "triton"
        if lang not in ("triton", "cuda"):
            raise ValueError(f"Unsupported language for gpumode_trimul definition text: {lang!r}")
        return f"{self.get_definition_text_for_language(language=lang)}\n"

    # Optional helper (not part of the Task Protocol): generators/CLIs can use this
    # to get language-specific prompt text.
    def get_definition_text_for_language(self, *, language: str) -> str:
        lang = str(language or "").strip().lower()
        return TRIMUL_SPEC_TEXT_CUDA if lang == "cuda" else TRIMUL_SPEC_TEXT_TRITON

    # Optional (not in Task Protocol): language-specific generation prompt.
    def get_generation_prompt(self, *, language: str, target_gpu: str) -> str:
        lang = str(language or "").strip().lower()
        if lang == "cuda":
            return (
                f"{self.get_definition_text_for_language(language='cuda')}\n\n"
                f"Target GPU: {target_gpu}\n\n"
            )
        # Triton/Python: strict entrypoint.
        return (
            f"{self.get_definition_text_for_language(language=lang)}\n\n"
            f"Target GPU: {target_gpu}\n\n"
        )

    # Optional (not in Task Protocol): language-specific optimization prompt.
    def get_optimization_prompt(
        self,
        *,
        language: str,
        target_gpu: str,
        trace_logs: str,
        current_code: str,
        current_best: str | None = None,
        previous_round_summary: str | None = None,
    ) -> str:
        def _strip_reference_block(txt: str) -> str:
            """
            Remove the large embedded reference submission from the spec to save context.
            We keep everything before the reference marker.
            """
            s = str(txt or "")
            marker = "Reference code (baseline `submission.py`):"
            if marker not in s:
                return s
            return s.split(marker, 1)[0].rstrip() + "\n"

        lang = str(language or "").strip().lower()
        # If we already provide current_best (which includes code + perf), drop the long embedded reference
        # submission block to save prompt context.
        if current_best:
            base_def = _strip_reference_block(self.get_definition_text_for_language(language=lang))
            if lang == "cuda":
                base = (
                    f"{base_def}\n\n"
                    f"Target GPU: {target_gpu}\n\n"
                ).strip()
            else:
                base = (
                    f"{base_def}\n\n"
                    f"Target GPU: {target_gpu}\n\n"
                ).strip()
        else:
            base = self.get_generation_prompt(language=lang, target_gpu=target_gpu).strip()

        parts: list[str] = [base]
        parts.append("\nCurrent implementation:\n" + str(current_code or "").strip())
        if previous_round_summary:
            parts.append("\nPrevious round summary:\n" + str(previous_round_summary).strip())
        if trace_logs:
            parts.append("\nExecution log / feedback:\n" + str(trace_logs).strip())
        if current_best:
            parts.append("\nCurrent best:\n" + str(current_best).strip())
        parts.append(
            "\nBefore changing the code: briefly analyze the likely bottlenecks and any correctness risks based on the current implementation + feedback.\n"
            "Analysis checklist (keep it short):\n"
            "- Identify avoidable large intermediates / extra full-tensor reads+writes.\n"
            "- Identify extra kernel launches / passes over the same data.\n"
            "- Note dtype conversions / precision choices that may hurt speed or correctness.\n"
            "Then implement the optimized version.\n"
        )
        return "\n\n".join([p for p in parts if p.strip()])

    # Optional (not in Task Protocol): task-owned solution construction.
    def make_solution_from_generated_code(
        self,
        *,
        cleaned_code: Any,
        raw_code: Any,
        round_num: int,
        model_name: str,
        target_gpu: str,
        language: str,
    ) -> Solution:
        lang = str(language or "").strip().lower()
        uid = f"r{int(round_num)}"
        sol_name = f"{model_name}_{self._name}_{lang}_{uid}"
        if lang == "cuda":
            if not isinstance(cleaned_code, dict):
                # Fall back to raw xml parse later in eval; store as-is in main.cpp for visibility.
                files = normalize_cuda_sources(raw_code)
            else:
                files = {str(k): str(v) for k, v in cleaned_code.items()}
            sources = [
                SourceFile(path="kernel.h", content=str(files.get("kernel.h", "") or "")),
                SourceFile(path="kernel.cu", content=str(files.get("kernel.cu", "") or "")),
                SourceFile(path="main.cpp", content=str(files.get("main.cpp", "") or "")),
            ]
            spec = BuildSpec(
                language=SupportedLanguages.CUDA,
                target_hardware=[str(target_gpu)],
                entry_point="main.cpp::run",
            )
        else:
            # Prefer cleaned_code from the generator. (KernelGenerator handles fenced-block extraction for
            # non-CUDA languages; task-level evaluator also validates entrypoint.)
            code_text = str(cleaned_code or "")
            if not code_text.strip():
                code_text = str(raw_code or "")
            sources = [SourceFile(path="submission.py", content=str(code_text))]
            spec = BuildSpec(
                language=(SupportedLanguages.TRITON if lang == "triton" else SupportedLanguages.PYTHON),
                target_hardware=[str(target_gpu)],
                entry_point="submission.py::custom_kernel",
            )
        return Solution(
            name=sol_name,
            definition=self._name,
            author=str(model_name),
            spec=spec,
            sources=sources,
            description="GPUMode TriMul submission",
        )

    def register_solution(self, sol: Solution) -> None:
        """Optional convenience: allow external code to register solutions by name."""
        if not isinstance(sol, Solution):
            raise TypeError("register_solution expects a k_search.tasks.task_base.Solution")
        self._solutions[str(sol.name)] = sol

    def get_solution(self, solution_name: str) -> Solution | None:
        name = str(solution_name)
        if name in self._solutions:
            return self._solutions.get(name)
        # Allow resolving from k-search artifacts Solution JSON (by path or by name).
        try:
            d = load_ksearch_solution_json(
                solution_ref=name,
                definition_name=str(self.name or ""),
                artifacts_dir=self._ksearch_artifacts_dir,
            )
            sol = solution_from_json_dict(d)
            if str(sol.definition or "") != str(self.name or ""):
                return None
            # Cache for subsequent lookups.
            self._solutions[str(sol.name)] = sol
            return sol
        except FileNotFoundError:
            return None
        except Exception:
            return None

    def code_for_world_model_from_raw(self, *, raw: Any, language: str) -> str:
        lang = str(language or "").strip().lower()
        if lang == "cuda":
            try:
                files = normalize_cuda_sources(raw)
                return str(files.get("kernel.cu", "") or "")
            except Exception:
                return str(raw or "")
        return str(raw or "")

    def seed_eval_for_base_solution(self, *, base_solution: Solution, config: Any = None) -> EvalResult:
        # For GPUMode we don't have dataset traces to seed from; just run a benchmark once.
        return self.run_benchmark(solution=base_solution, config=config, dump_traces=False, round_num=None)

    def run_benchmark(
        self,
        *,
        solution: Solution,
        config: Any = None,
        dump_traces: bool = False,
        round_num: int | None = None,
    ) -> EvalResult:
        # Convert k-search Solution sources to the evaluator input format.
        lang = str(getattr(solution.spec, "language", "") or "").strip().lower()
        entry_src = solution.get_entry_source()
        if lang == "cuda":
            sources_dict = {sf.path: sf.content for sf in (solution.sources or [])}
            submission_code = normalize_cuda_sources(sources_dict)
        else:
            submission_code = (entry_src.content if entry_src else "") or ""

        try:
            summary = evaluate_trimul_submission(
                submission_code=submission_code,
                mode=self._cfg.mode,
                language=lang or "python",
                verbose=True,
                keep_tmp=bool(self._cfg.keep_tmp),
                task_dir=self._cfg.task_dir,
            )
        except Exception as e:
            # Fail fast but don't crash generator loops: malformed outputs should be treated as failed evals.
            # IMPORTANT: populate last-round caches so the generator can surface the failure in the next prompt/log.
            try:
                self._last_round_trace_logs_for_prompt = f"[gpumode_error] {type(e).__name__}: {e}"
                self._last_round_total_workloads = 1
                self._last_round_passed_count = 0
                rn = str(int(round_num)) if round_num is not None else "?"
                self._last_round_summary_line = (
                    f"[{self._name}] Round {rn}: workloads=0/1 (0.0%) | status=failed | "
                    f"latency=- | score=- | mode={self._cfg.mode}"
                )
                if self._last_round_summary_line.strip():
                    print(self._last_round_summary_line, flush=True)
            except Exception:
                pass
            return EvalResult(
                status="failed",
                latency_ms=None,
                reference_latency_ms=None,
                mean_vs_baseline_factor=None,
                speedup_factor=None,
                log_excerpt=f"[gpumode_error] {type(e).__name__}: {e}",
                metrics={
                    "score_name": "inv_latency_ms",
                    "score": None,
                    "gpumode_mode": str(self._cfg.mode),
                },
            )

        passed = str(getattr(summary, "status", "") or "").strip().lower() == "passed"
        latency_ms = getattr(summary, "latency_ms", None)
        latency_ms_f = float(latency_ms) if isinstance(latency_ms, (int, float)) else None

        # For GPUMode, lower latency is better. Use 1/latency as a simple comparable score.
        score = None
        if passed and isinstance(latency_ms_f, (int, float)) and float(latency_ms_f) > 0:
            score = 1.0 / float(latency_ms_f)

        # Populate last-round feedback hooks for generators (best-effort).
        try:
            # For GPUMode, this is the only place the next-round prompt can see the full traceback.
            # Do not aggressively truncate here; the evaluator already bounds log_excerpt.
            self._last_round_trace_logs_for_prompt = str(getattr(summary, "log_excerpt", "") or "")
            self._last_round_total_workloads = 1
            self._last_round_passed_count = 1 if passed else 0
        except Exception:
            pass

        er = EvalResult(
            status=("passed" if passed else "failed"),
            latency_ms=latency_ms_f,
            reference_latency_ms=None,
            mean_vs_baseline_factor=None,
            speedup_factor=None,
            log_excerpt=str(getattr(summary, "log_excerpt", "") or ""),
            metrics={
                "score_name": "inv_latency_ms",
                "score": score,
                "gpumode_mode": str(self._cfg.mode),
                "gpumode_run_key": getattr(summary, "run_key", None),
                "gpumode_run_success": bool(getattr(summary, "run_success", False)),
                "gpumode_run_passed": bool(getattr(summary, "run_passed", False)),
            },
        )

        # Mimic FlashInferBenchTask: print a single compact summary line per benchmark so logs are actionable.
        try:
            rn = str(int(round_num)) if round_num is not None else "?"
            total = int(getattr(self, "_last_round_total_workloads", 1) or 1)
            pc = int(getattr(self, "_last_round_passed_count", 0) or 0)
            pr = (pc / float(total) * 100.0) if total > 0 else 0.0
            lat_text = f"{float(latency_ms_f):.4f} ms" if isinstance(latency_ms_f, (int, float)) else "-"
            score_name = None
            try:
                score_name = er.metrics.get("score_name") if isinstance(getattr(er, "metrics", None), dict) else None
            except Exception:
                score_name = None
            sc = er.score() if getattr(er, "is_passed", lambda: False)() else None
            score_text = f"{float(sc):.6f}" if isinstance(sc, (int, float)) and float(sc) > 0 else "-"
            score_label = str(score_name or "score")
            self._last_round_summary_line = (
                f"[{self._name}] Round {rn}: workloads={pc}/{total} ({pr:.1f}%) | status={er.status} | "
                f"latency={lat_text} | {score_label}={score_text} | mode={self._cfg.mode}"
            )
            if self._last_round_summary_line.strip():
                print(self._last_round_summary_line, flush=True)
            if not getattr(er, "is_passed", lambda: False)():
                le = str(getattr(er, "log_excerpt", "") or "").strip()
                if le:
                    max_chars = int(getattr(self._cfg, "max_failure_excerpt_chars", 800) or 800)
                    if max_chars <= 0:
                        max_chars = 800
                    if len(le) > max_chars:
                        le = le[:max_chars] + "...<truncated>..."
                    print(f"[{self._name}] Failure excerpt:\n{le}", flush=True)
        except Exception:
            pass

        return er

    # Optional (not in Task Protocol): final eval helper for scripts.
    def run_final_evaluation(
        self,
        *,
        solutions: list[Solution],
        config: Any = None,
        dump_traces: bool = False,
        workload_limit: int | None = None,
    ) -> dict[str, Any]:
        out: list[dict[str, Any]] = []
        for sol in solutions or []:
            if sol is None:
                continue
            er = self.run_benchmark(solution=sol, dump_traces=False, round_num=None)
            out.append(
                {
                    "solution": str(getattr(sol, "name", "") or ""),
                    "status": str(er.status or ""),
                    "latency_ms": er.latency_ms,
                    "score_name": (er.metrics.get("score_name") if isinstance(er.metrics, dict) else None),
                    "score": (er.metrics.get("score") if isinstance(er.metrics, dict) else None),
                }
            )
        return {
            "task": str(self._name),
            "mode": str(self._cfg.mode),
            "solutions": out,
        }

    # Optional feedback hooks used by generators (via getattr).
    def get_last_round_trace_logs_for_prompt(self) -> str:
        return str(getattr(self, "_last_round_trace_logs_for_prompt", "") or "")

    def get_last_round_passed_count(self) -> int:
        try:
            return int(getattr(self, "_last_round_passed_count", 0) or 0)
        except Exception:
            return 0

    def get_last_round_total_workloads(self) -> int:
        try:
            return int(getattr(self, "_last_round_total_workloads", 0) or 0)
        except Exception:
            return 0

    def get_last_round_summary_line(self) -> str:
        """Optional convenience hook (mirrors FlashInferBenchTask behavior)."""
        return str(getattr(self, "_last_round_summary_line", "") or "")

    def get_config_for_logging(self) -> Dict[str, Any]:
        return {
            "task_type": "gpu_mode",
            "task_name": self._name,
            "mode": str(self._cfg.mode),
            "keep_tmp": bool(self._cfg.keep_tmp),
            "task_dir": str(self._cfg.task_dir),
        }


