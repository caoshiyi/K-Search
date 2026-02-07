"""FlashInfer-Bench task adapter.

This module is the *only* place that should directly construct:
- `flashinfer_bench.Benchmark`
- `flashinfer_bench.BenchmarkConfig`
- temporary `flashinfer_bench.TraceSet` objects for evaluation
- `flashinfer_bench.utils.hardware_from_device`

The generator loops call into this adapter so we can later plug in other task
evaluators (e.g. GPUMode) without rewriting the loop logic.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional

from .task_base import BuildSpec as TaskBuildSpec
from .task_base import EvalResult
from .task_base import Solution as TaskSolution
from .task_base import SourceFile as TaskSourceFile
from .task_base import SupportedLanguages as TaskSupportedLanguages


@dataclass(frozen=True)
class FlashInferBenchEvalConfig:
    warmup_runs: int = 10
    iterations: int = 50
    num_trials: int = 3
    rtol: float = 1e-2
    atol: float = 1e-2
    use_isolated_runner: bool = False
    parallel_workloads: bool = False
    max_parallel_workloads: int = 0


class FeedbackTraceSelector:
    """
    Select a single trace to feed into prompts (flashinfer-bench scoped).

    Supported policies:
    - **first**: first failed workload in `selected_workloads` order; else first workload in that order.
    - **random**: if any failure exists, still prefer failures; else choose a random trace.
    """

    def __init__(self, policy: str = "first"):
        self.policy = (policy or "first").strip().lower()
        if self.policy not in ("first", "random"):
            raise ValueError(
                f"Unknown feedback trace policy '{self.policy}'. Supported: first, random."
            )

    def select(
        self,
        *,
        traces: List[Any],
        selected_workloads: List[Any],
        by_wl: Dict[str, List[Any]],
    ) -> Optional[Any]:
        import random

        if not traces:
            return None

        # Prefer any failed trace.
        for t in traces:
            try:
                if not FlashInferBenchTask.is_passed_trace(t):
                    return t
            except Exception:
                continue

        if self.policy == "random":
            return random.choice(traces)

        # "first": first trace in selected_workloads order (if available), else first trace.
        for wl in selected_workloads:
            wl_uuid = getattr(getattr(wl, "workload", None), "uuid", None)
            if wl_uuid is None:
                continue
            wl_traces = by_wl.get(wl_uuid, [])
            if wl_traces:
                return wl_traces[0]
        return traces[0]


class FlashInferBenchTask:
    """Thin wrapper around flashinfer-bench evaluation APIs."""

    def __init__(
        self,
        *,
        traceset: Any,
        definition: Any | None = None,
        feedback_trace_policy: str = "first",
        feedback_trace_selector: Any | None = None,
        num_feedback_workloads: int | None = None,
        feedback_workloads: Optional[list[str]] = None,
        baseline_solution_name: Optional[str] = None,
        eval_config: FlashInferBenchEvalConfig | None = None,
    ):
        # We intentionally keep this typed as Any so the generator can be type-agnostic later.
        self._task_path: str | None = None  # set by factories when constructed from a dataset path
        self._traceset = traceset
        self._definition = definition
        self._feedback_trace_selector = (
            feedback_trace_selector if feedback_trace_selector is not None else FeedbackTraceSelector(feedback_trace_policy)
        )
        self._init_num_feedback_workloads = (
            int(num_feedback_workloads) if num_feedback_workloads is not None else None
        )
        self._init_feedback_workloads = list(feedback_workloads) if feedback_workloads else None
        self._init_baseline_solution_name = (
            str(baseline_solution_name) if baseline_solution_name is not None else None
        )
        self._eval_config: FlashInferBenchEvalConfig = eval_config or FlashInferBenchEvalConfig()
        self._baseline_prepared: bool = False
        # Per-generate context (owned by the task, not the generator).
        self._selected_workloads: list[Any] = []
        self._baseline_latency_by_wl: Dict[str, float] = {}
        self._baseline_targets_text: str = ""
        # Last-round cached info (for the generator loop).
        self._last_round_passed_count: int = 0
        self._last_round_total_workloads: int = 0
        self._last_round_summary_line: str = ""
        self._last_round_feedback_trace: Any = None
        self._last_round_trace_logs_for_prompt: str = ""

        # If workload selection config was provided and we already have a definition, select now.
        if self._definition is not None and (
            self._init_num_feedback_workloads is not None or self._init_feedback_workloads is not None
        ):
            self.prepare_selected_workloads(
                num_feedback_workloads=int(self._init_num_feedback_workloads or 1),
                feedback_workloads=self._init_feedback_workloads,
            )
            # If baseline is configured, prepare it now as well.
            if self._init_baseline_solution_name is not None:
                self._prepare_baseline_if_needed()

    def set_eval_config(self, eval_config: FlashInferBenchEvalConfig) -> None:
        self._eval_config = eval_config or FlashInferBenchEvalConfig()

    # -------- Dataset helpers (kept here so scripts don't import flashinfer-bench) --------
    @staticmethod
    def load_traceset_from_path(dataset_path: str) -> Any:
        """Load a flashinfer-bench TraceSet from path (import stays inside task module)."""
        from flashinfer_bench.data import TraceSet

        return TraceSet.from_path(str(dataset_path))

    @staticmethod
    def list_definition_names_from_path(dataset_path: str) -> list[str]:
        """Convenience for scripts: list definitions without importing flashinfer-bench."""
        ts = FlashInferBenchTask.load_traceset_from_path(str(dataset_path))
        return FlashInferBenchTask.list_definition_names(ts)

    @staticmethod
    def list_definition_names(traceset: Any) -> list[str]:
        try:
            defs = getattr(traceset, "definitions", None)
            if isinstance(defs, dict):
                return [str(k) for k in defs.keys()]
        except Exception:
            pass
        return []

    @staticmethod
    def get_definition(traceset: Any, definition_name: str) -> Any:
        try:
            defs = getattr(traceset, "definitions", None)
            if isinstance(defs, dict):
                return defs.get(definition_name)
        except Exception:
            pass
        return None

    @classmethod
    def from_dataset_path(
        cls,
        *,
        dataset_path: str,
        definition_name: str,
        feedback_trace_policy: str = "first",
        num_feedback_workloads: int | None = None,
        feedback_workloads: Optional[list[str]] = None,
        baseline_solution_name: Optional[str] = None,
        eval_config: FlashInferBenchEvalConfig | None = None,
    ) -> "FlashInferBenchTask":
        """
        Construct a task from a dataset path + definition name (keeps flashinfer-bench imports in this module).
        """
        ts = cls.load_traceset_from_path(str(dataset_path))
        definition = cls.get_definition(ts, str(definition_name))
        if definition is None:
            raise ValueError(f"Definition not found in dataset: {definition_name}")
        obj = cls(
            traceset=ts,
            definition=definition,
            feedback_trace_policy=str(feedback_trace_policy or "first"),
            num_feedback_workloads=num_feedback_workloads,
            feedback_workloads=feedback_workloads,
            baseline_solution_name=baseline_solution_name,
            eval_config=eval_config,
        )
        obj._task_path = str(dataset_path)
        return obj

    @classmethod
    def from_cli_args(
        cls,
        *,
        task_path: str,
        definition_name: str,
        warmup_runs: int,
        iterations: int,
        num_trials: int,
        rtol: float,
        atol: float,
        use_isolated_runner: bool,
        parallel_workloads: bool,
        max_parallel_workloads: int,
        baseline_solution: Optional[str],
        feedback_workloads: Optional[list[str]],
        feedback_trace_policy: str,
        num_feedback_workloads: int,
    ) -> "FlashInferBenchTask":
        """
        Convenience factory for scripts/CLI so task-specific init logic lives in the task module.
        """
        eval_cfg = FlashInferBenchEvalConfig(
            warmup_runs=int(warmup_runs),
            iterations=int(iterations),
            num_trials=int(num_trials),
            rtol=float(rtol),
            atol=float(atol),
            use_isolated_runner=bool(use_isolated_runner),
            parallel_workloads=bool(parallel_workloads),
            max_parallel_workloads=int(max_parallel_workloads),
        )
        return cls.from_dataset_path(
            dataset_path=str(task_path),
            definition_name=str(definition_name),
            feedback_trace_policy=str(feedback_trace_policy or "first"),
            num_feedback_workloads=int(num_feedback_workloads),
            feedback_workloads=feedback_workloads,
            baseline_solution_name=baseline_solution,
            eval_config=eval_cfg,
        )

    def get_config_for_logging(self) -> Dict[str, Any]:
        cfg = self._eval_config or FlashInferBenchEvalConfig()
        return {
            "task_source": "flashinfer",
            "task_path": (str(self._task_path) if self._task_path else None),
            "definition": str(self.name or ""),
            "baseline_solution": (
                str(self._init_baseline_solution_name) if self._init_baseline_solution_name else None
            ),
            "feedback_trace_policy": str(getattr(self._feedback_trace_selector, "policy", "first") or "first"),
            "num_feedback_workloads": (
                int(self._init_num_feedback_workloads) if self._init_num_feedback_workloads is not None else None
            ),
            "feedback_workloads": (list(self._init_feedback_workloads) if self._init_feedback_workloads else None),
            "eval_config": {
                "warmup_runs": int(cfg.warmup_runs),
                "iterations": int(cfg.iterations),
                "num_trials": int(cfg.num_trials),
                "rtol": float(cfg.rtol),
                "atol": float(cfg.atol),
                "use_isolated_runner": bool(cfg.use_isolated_runner),
                "parallel_workloads": bool(cfg.parallel_workloads),
                "max_parallel_workloads": int(cfg.max_parallel_workloads),
            },
        }

    @staticmethod
    def _to_task_language(lang: Any) -> TaskSupportedLanguages:
        try:
            s = getattr(lang, "value", None) if lang is not None else None
            s = s if isinstance(s, str) else (str(lang) if lang is not None else "")
            s = s.strip().lower()
        except Exception:
            s = ""
        if s == "cuda":
            return TaskSupportedLanguages.CUDA
        if s == "triton":
            return TaskSupportedLanguages.TRITON
        if s == "cpp":
            return TaskSupportedLanguages.CPP
        return TaskSupportedLanguages.PYTHON

    @staticmethod
    def _to_backend_language(lang: Any) -> Any:
        # Import locally to keep flashinfer-bench dependency contained.
        from flashinfer_bench.data.solution import SupportedLanguages as FBSupportedLanguages

        try:
            s = lang.value if isinstance(lang, TaskSupportedLanguages) else str(lang)
            s = str(s or "").strip().lower()
        except Exception:
            s = "python"
        if s == "cuda":
            return FBSupportedLanguages.CUDA
        if s == "triton":
            return FBSupportedLanguages.TRITON
        if s == "cpp":
            return FBSupportedLanguages.CPP
        return FBSupportedLanguages.PYTHON

    @classmethod
    def _from_backend_solution(cls, sol: Any) -> TaskSolution:
        """
        Convert flashinfer-bench Solution -> task_base.Solution.
        """
        spec0 = getattr(sol, "spec", None)
        lang0 = getattr(spec0, "language", None) if spec0 is not None else None
        deps0 = getattr(spec0, "dependencies", None) if spec0 is not None else None
        deps_raw = list(deps0) if isinstance(deps0, list) else []
        # flashinfer-bench expects List[str]; older tracesets may contain enums/objects.
        deps = [str(x) for x in deps_raw if x is not None and str(x).strip()]
        tgt0 = getattr(spec0, "target_hardware", None) if spec0 is not None else None
        tgt_raw = list(tgt0) if isinstance(tgt0, list) else []
        tgt = [str(x) for x in tgt_raw if x is not None and str(x).strip()]
        ep = str(getattr(spec0, "entry_point", "") or "") if spec0 is not None else ""
        sources0 = getattr(sol, "sources", None)
        sources: list[TaskSourceFile] = []
        if isinstance(sources0, list):
            for sf in sources0:
                try:
                    sources.append(TaskSourceFile(path=str(sf.path), content=str(sf.content)))
                except Exception:
                    continue
        return TaskSolution(
            name=str(getattr(sol, "name", "") or ""),
            definition=str(getattr(sol, "definition", "") or ""),
            author=str(getattr(sol, "author", "") or ""),
            spec=TaskBuildSpec(
                language=cls._to_task_language(lang0),
                target_hardware=tgt,
                entry_point=ep,
                dependencies=deps,
            ),
            sources=sources or [TaskSourceFile(path="main.py", content="")],
            description=(getattr(sol, "description", None) if isinstance(getattr(sol, "description", None), str) else None),
        )

    @classmethod
    def _to_backend_solution(cls, sol: TaskSolution) -> Any:
        """
        Convert task_base.Solution -> flashinfer-bench Solution.
        """
        from flashinfer_bench.data.solution import BuildSpec as FBBuildSpec
        from flashinfer_bench.data.solution import Solution as FBSolution
        from flashinfer_bench.data.solution import SourceFile as FBSourceFile

        fb_sources: list[Any] = []
        for sf in sol.sources or []:
            p = str(getattr(sf, "path", "") or "")
            c = str(getattr(sf, "content", "") or "")
            # flashinfer-bench uses NonEmptyString for both fields; fail loudly if invalid.
            if not p.strip() or not c.strip():
                continue
            fb_sources.append(FBSourceFile(path=p, content=c))
        if not fb_sources:
            raise ValueError("Cannot convert TaskSolution to flashinfer-bench Solution: empty/invalid sources")
        fb_spec = FBBuildSpec(
            language=cls._to_backend_language(sol.spec.language),
            target_hardware=[
                str(x)
                for x in (list(sol.spec.target_hardware or []) or ["cuda"])
                if x is not None and str(x).strip()
            ],
            entry_point=str(sol.spec.entry_point or "main.py::run"),
            dependencies=[str(x) for x in (sol.spec.dependencies or []) if x is not None and str(x).strip()],
        )
        return FBSolution(
            name=str(sol.name or ""),
            definition=str(sol.definition or ""),
            author=str(sol.author or ""),
            spec=fb_spec,
            sources=fb_sources,
            description=(str(sol.description) if sol.description is not None else None),
        )

    def _require_definition(self) -> Any:
        if self._definition is None:
            raise ValueError("FlashInferBenchTask requires a definition, but none was provided")
        return self._definition

    @property
    def name(self) -> str:
        """Task/definition name (mirrors flashinfer-bench Definition.name)."""
        try:
            return str(getattr(self._require_definition(), "name", "") or "")
        except Exception:
            return ""

    def set_definition(self, definition: Any) -> None:
        self._definition = definition
        # If workload selection config was provided at init, select workloads once we have a definition.
        if not self._selected_workloads and (
            self._init_num_feedback_workloads is not None or self._init_feedback_workloads is not None
        ):
            self.prepare_selected_workloads(
                num_feedback_workloads=int(self._init_num_feedback_workloads or 1),
                feedback_workloads=self._init_feedback_workloads,
            )
        # If baseline is configured, prepare it once we have workloads+definition.
        if self._init_baseline_solution_name is not None:
            self._prepare_baseline_if_needed()

    def _ensure_selected_workloads_prepared(self) -> None:
        """
        Ensure selected workloads are available before any baseline/eval logic.
        Generators should not call workload selection directly.
        """
        if self._selected_workloads:
            return
        if self._init_num_feedback_workloads is None and self._init_feedback_workloads is None:
            raise ValueError(
                "Selected workloads are not prepared. "
                "Initialize FlashInferBenchTask with num_feedback_workloads/feedback_workloads, "
                "or call set_selected_workloads(...) before evaluation."
            )
        self.prepare_selected_workloads(
            num_feedback_workloads=int(self._init_num_feedback_workloads or 1),
            feedback_workloads=self._init_feedback_workloads,
        )

    def _prepare_baseline_if_needed(self) -> None:
        """
        Internal baseline preparation; generators should not call baseline prep directly.
        """
        if self._baseline_prepared:
            return
        self._ensure_selected_workloads_prepared()
        self._baseline_latency_by_wl = {}
        if self._init_baseline_solution_name:
            self._baseline_latency_by_wl = self.compute_baseline_latency_by_workload(
                definition_name=self.name,
                selected_workloads=list(self._selected_workloads),
                baseline_solution=str(self._init_baseline_solution_name),
            )
        self._baseline_targets_text = self.render_baseline_targets_text(
            selected_workloads=list(self._selected_workloads),
            baseline_latency_by_wl=dict(self._baseline_latency_by_wl),
        )
        self._baseline_prepared = True


    def get_definition_text(self) -> str:
        """
        Task-owned definition rendering. Prompts are task-agnostic and only consume this text.
        """
        definition = self._require_definition()

        axes_str = "\nAxes:\n"
        for name, axis in definition.axes.items():
            if hasattr(axis, "value"):
                axes_str += f"  {name}: constant = {axis.value}"
            else:
                axes_str += f"  {name}: variable"
            if axis.description:
                axes_str += f" ({axis.description})"
            axes_str += "\n"

        # Format inputs
        inputs_str = "\nInputs:\n"
        for name, spec in definition.inputs.items():
            shape_str = "scalar" if spec.shape is None else f"[{', '.join(spec.shape)}]"
            inputs_str += f"  {name}: {shape_str} ({spec.dtype})"
            if spec.description:
                inputs_str += f" - {spec.description}"
            inputs_str += "\n"

        outputs_str = "\nOutputs:\n"
        for name, spec in definition.outputs.items():
            shape_str = "scalar" if spec.shape is None else f"[{', '.join(spec.shape)}]"
            outputs_str += f"  {name}: {shape_str} ({spec.dtype})"
            if spec.description:
                outputs_str += f" - {spec.description}"
            outputs_str += "\n"

        constraints_str = ""
        if definition.constraints:
            constraints_str = "\nConstraints:\n"
            for constraint in definition.constraints:
                constraints_str += f"  - {constraint}\n"

        return str(
            f"""Name: {definition.name}
Type: {definition.op_type}
{axes_str}{inputs_str}{outputs_str}{constraints_str}

Reference Implementation:
{definition.reference}"""
            or ""
        ).strip()

    def trace_logs_for_prompt(self, trace: Any, *, omit_when_passed: bool = True) -> str:
        """
        Task-owned trace log formatting for prompts.

        Mirrors existing WM prompt behavior: when a trace PASSED, we usually omit full logs and rely on perf summary.
        """
        if omit_when_passed and self.is_passed_trace(trace):
            return "(omitted; use perf summary)"
        try:
            # Intentionally implemented here (not in prompt modules) so tasks don't import generators.
            if trace.is_workload_trace() or not trace.evaluation:
                return "No evaluation logs available (workload-only trace)"

            eval_info = f"Status: {trace.evaluation.status.value}\n"
            eval_info += f"Timestamp: {trace.evaluation.timestamp}\n"

            if trace.evaluation.log:
                eval_info += f"\nExecution Log:\n{trace.evaluation.log}\n"

            if trace.evaluation.correctness:
                eval_info += f"Max relative error: {trace.evaluation.correctness.max_relative_error}\n"
                eval_info += f"Max absolute error: {trace.evaluation.correctness.max_absolute_error}\n"

            if trace.evaluation.performance:
                eval_info += f"Latency: {trace.evaluation.performance.latency_ms}ms\n"
                eval_info += f"Reference latency: {trace.evaluation.performance.reference_latency_ms}ms\n"
                eval_info += f"Speedup factor: {trace.evaluation.performance.speedup_factor}x\n"

            print(eval_info)
            return str(eval_info or "").strip()
        except Exception:
            # Best effort: do not block prompting.
            return "(no logs)"

    @staticmethod
    def format_workload_axes_inline_for_prompt(wl_trace: Any) -> str:
        """
        Render a compact workload axes summary inline for prompts.
        (Moved from generator to task so other evaluators can implement their own formatting.)
        """
        try:
            w = getattr(wl_trace, "workload", None)
            axes = getattr(w, "axes", None)
            if not isinstance(axes, dict):
                return ""
            if axes:
                ax_items: list[str] = []
                for k in sorted(axes.keys()):
                    v = axes.get(k)
                    if isinstance(v, (int, float, str, bool)):
                        ax_items.append(f"{k}={v}")
                s = ("axes{" + ",".join(ax_items) + "}") if ax_items else ""
            else:
                s = ""
            return s or ""
        except Exception:
            return ""

    def code_for_world_model_from_raw(self, *, raw: Any, language: str) -> str:
        """
        Prefer passing only kernel.cu (full text) to the WM prompts to reduce noise.
        For CUDA, attempt to extract kernel.cu from the generator XML container format.
        """
        try:
            sraw = str(raw or "")
            if not sraw.strip():
                return ""
            if str(language or "").lower() != "cuda":
                return sraw
            # Best-effort extraction from the common XML payload format.
            import re

            m = re.search(
                r'<cuda_file\s+name="kernel\.cu"\s*>([\s\S]*?)</cuda_file>',
                sraw,
                flags=re.IGNORECASE,
            )
            if m:
                cu = m.group(1)
                cu = re.sub(r"^\s*<!\[CDATA\[\s*", "", cu)
                cu = re.sub(r"\s*\]\]>\s*$", "", cu)
                cu = cu.strip()
                if cu:
                    return cu
            return sraw
        except Exception:
            return str(raw or "")

    def seed_eval_for_base_solution(
        self,
        *,
        base_solution: TaskSolution,
        config: FlashInferBenchEvalConfig | None = None,
    ) -> Any:
        """
        Best-effort seed eval for a resume-from-solution path:
        - try to aggregate from dataset traces first
        - if no usable score, run a one-off benchmark
        """

        try:
            seed_eval = self.seed_eval_from_dataset_traces(base_solution=base_solution)
        except Exception:
            seed_eval = EvalResult(status="seeded", log_excerpt="", metrics={})

        try:
            has_score = bool(seed_eval.is_passed() and seed_eval.score() > 0)
        except Exception:
            has_score = False

        if not has_score:
            try:
                # Mirror upstream WM generator logging for resume seeding.
                try:
                    sol_name = str(getattr(base_solution, "name", "") or "")
                except Exception:
                    sol_name = ""
                if sol_name:
                    print(
                        f"\n[STAGE] benchmark continue-from solution to seed base score: {sol_name}",
                        flush=True,
                    )
                seed_eval = self.run_benchmark(
                    solution=base_solution,
                    config=(config or self._eval_config),
                    dump_traces=False,
                    round_num=None,
                )
            except Exception:
                pass
        return seed_eval

    # -------- Dataset access helpers --------
    def _list_workloads(self, *, definition_name: str) -> list[Any]:
        return list(getattr(self._traceset, "workloads", {}).get(definition_name, []) or [])

    def _list_traces(self, *, definition_name: str) -> list[Any]:
        return list(getattr(self._traceset, "traces", {}).get(definition_name, []) or [])

    def get_solution(self, solution_name: str) -> TaskSolution | None:
        # TraceSet.get_solution is the canonical API in flashinfer-bench.
        try:
            sol = self._traceset.get_solution(solution_name)
        except Exception:
            sol = None
        if sol is None:
            return None
        try:
            return self._from_backend_solution(sol)
        except Exception:
            return None

    def to_backend_solution(self, solution: TaskSolution) -> Any:
        """Convert a task_base.Solution to the backend (flashinfer-bench) Solution."""
        return self._to_backend_solution(solution)

    def select_workloads(
        self,
        *,
        definition_name: str,
        num_feedback_workloads: int,
        feedback_workloads: Optional[list[str]],
    ) -> list[Any]:
        """
        Select workloads to use for feedback.

        This preserves current generator behavior:
        - if feedback_workloads is provided: keep that order, filter to existing workloads
        - else: sample up to num_feedback_workloads (at least 1)
        """
        import random

        workloads = self._list_workloads(definition_name=definition_name)
        if not workloads:
            return []
        if feedback_workloads:
            by_uuid = {getattr(getattr(wl, "workload", None), "uuid", None): wl for wl in workloads}
            out: list[Any] = []
            for wl_id in feedback_workloads:
                if wl_id in by_uuid:
                    out.append(by_uuid[wl_id])
            return out
        k = max(1, min(int(num_feedback_workloads), len(workloads)))
        return random.sample(workloads, k=k) if len(workloads) >= k else workloads

    def prepare_selected_workloads(
        self,
        *,
        num_feedback_workloads: int,
        feedback_workloads: Optional[list[str]],
    ) -> list[str]:
        """
        Task-owned workload selection and caching for the generator loop.
        Returns the selected workload UUIDs (for logging only).
        """
        self._selected_workloads = self.select_workloads(
            definition_name=self.name,
            num_feedback_workloads=num_feedback_workloads,
            feedback_workloads=feedback_workloads,
        )
        out: list[str] = []
        for wl in self._selected_workloads:
            try:
                out.append(str(wl.workload.uuid))
            except Exception:
                continue

        if not out:
            # Preserve previous behavior/message when there are no workloads.
            raise ValueError(
                f"No workloads found for definition '{self.name}' in the provided TraceSet"
            )

        print(f"Generating optimized solution for {self.name}")
        print("Using workloads for optimization feedback: " + ", ".join(out))

        return out

    def get_selected_workloads(self) -> list[Any]:
        """Internal accessor; generator should generally not depend on workload objects."""
        return list(self._selected_workloads)

    def set_selected_workloads(self, selected_workloads: list[Any]) -> None:
        """Allow generators to set the workload list explicitly (no internal sampling)."""
        self._selected_workloads = list(selected_workloads or [])

    def set_baseline_solution_name(self, baseline_solution_name: Optional[str]) -> None:
        """Configure baseline reference (task-owned)."""
        self._init_baseline_solution_name = (
            str(baseline_solution_name) if baseline_solution_name is not None else None
        )
        # Invalidate and lazily recompute baseline next time.
        self._baseline_prepared = False

    def get_baseline_targets_text(self) -> str:
        self._prepare_baseline_if_needed()
        return str(self._baseline_targets_text or "").strip()

    def summarize_round_and_select_feedback_trace(
        self,
        *,
        traces: list[Any],
        feedback_trace_selector: Any | None,
    ) -> dict:
        """
        Task-owned round aggregation + feedback-trace selection.

        Returns a dict with:
        - passed_count, total_workloads, all_passed
        - mean_speedup, mean_vs_baseline, mean_latency_ms
        - prev_trace_for_prompt
        - summary_line (ready to `print()`)
        """
        by_wl: Dict[str, List[Any]] = {}
        for t in traces:
            try:
                by_wl.setdefault(t.workload.uuid, []).append(t)
            except Exception:
                continue

        er = self.eval_result_from_traces(
            traces=traces,
            selected_workloads=list(self._selected_workloads),
            baseline_latency_by_wl=dict(self._baseline_latency_by_wl),
        )

        passed_count = 0
        total = len(self._selected_workloads)
        try:
            # Recompute passed_count using the same predicate as before (per-workload pass check).
            for wl in self._selected_workloads:
                wl_uuid = wl.workload.uuid
                wl_traces = by_wl.get(wl_uuid, [])
                passed = [t for t in wl_traces if self.is_passed_trace(t)]
                if passed:
                    passed_count += 1
        except Exception:
            passed_count = 0

        all_passed = (passed_count == total) if total else False
        try:
            mean_speedup = float(getattr(er, "speedup_factor", None) or 0.0)
        except Exception:
            mean_speedup = 0.0
        try:
            mean_vs_baseline = float(getattr(er, "mean_vs_baseline_factor", None) or -1.0)
        except Exception:
            mean_vs_baseline = -1.0
        try:
            mean_latency = float(getattr(er, "latency_ms", None) or 0.0)
        except Exception:
            mean_latency = 0.0

        trace_for_feedback = None
        if feedback_trace_selector is not None:
            try:
                trace_for_feedback = feedback_trace_selector.select(
                    traces=traces,
                    selected_workloads=list(self._selected_workloads),
                    by_wl=by_wl,  # type: ignore[arg-type]
                )
            except Exception:
                trace_for_feedback = None

        summary_line = (
            f"Round summary: passed {passed_count}/{total} | "
            f"mean_vs_baseline={(f'{mean_vs_baseline:.2f}x' if (all_passed and mean_vs_baseline > 0) else '-')} | "
            f"mean_speedup={(f'{mean_speedup:.2f}x' if (all_passed and mean_speedup > 0) else '-')} | "
            f"mean_latency={(f'{mean_latency:.4f} ms' if (all_passed and mean_latency > 0) else '-')}"
        )

        return {
            "passed_count": int(passed_count),
            "total_workloads": int(total),
            "all_passed": bool(all_passed),
            "mean_speedup": float(mean_speedup),
            "mean_vs_baseline": float(mean_vs_baseline),
            "mean_latency_ms": float(mean_latency),
            "prev_trace_for_prompt": trace_for_feedback,
            "summary_line": str(summary_line),
        }

    def get_last_round_feedback_trace(self) -> Any:
        return getattr(self, "_last_round_feedback_trace", None)

    def has_last_round_feedback_trace(self) -> bool:
        return self.get_last_round_feedback_trace() is not None

    def get_last_round_trace_logs_for_prompt(self) -> str:
        return str(getattr(self, "_last_round_trace_logs_for_prompt", "") or "")

    def get_last_round_summary_line(self) -> str:
        return str(getattr(self, "_last_round_summary_line", "") or "")

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

    @staticmethod
    def is_passed_trace(trace: Any) -> bool:
        """
        flashinfer-bench trace predicate used by generator aggregation.
        Kept here so generators don't need to import EvaluationStatus.
        """
        try:
            from flashinfer_bench import EvaluationStatus

            ev = getattr(trace, "evaluation", None)
            st = getattr(ev, "status", None) if ev is not None else None
            if st == EvaluationStatus.PASSED:
                return True
            # Be tolerant to alternative serializations.
            name = getattr(st, "name", None)
            val = getattr(st, "value", None)
            if isinstance(name, str) and name.strip().lower() == "passed":
                return True
            if isinstance(val, str) and val.strip().lower() == "passed":
                return True
            if isinstance(st, str) and st.strip().lower() == "passed":
                return True
            return False
        except Exception:
            return False

    @staticmethod
    def render_baseline_targets_text(
        *,
        selected_workloads: list[Any],
        baseline_latency_by_wl: Dict[str, float],
    ) -> str:
        """
        Render baseline targets text used by the generator loops.
        Kept here so non-TraceSet tasks can render targets in their own format.
        """
        if not baseline_latency_by_wl:
            return ""
        lines: list[str] = []
        for wl in selected_workloads:
            try:
                wl_uuid = wl.workload.uuid
            except Exception:
                continue
            if wl_uuid in baseline_latency_by_wl:
                wl_spec = FlashInferBenchTask.format_workload_axes_inline_for_prompt(wl)
                try:
                    lat = float(baseline_latency_by_wl[wl_uuid])
                except Exception:
                    continue
                # Match upstream WM baseline-hint formatting:
                # - workload <uuid>: target_latency_ms <= <lat:.3f>  (spec: axes{...})
                lines.append(
                    f"- workload {wl_uuid}: target_latency_ms <= {lat:.3f}"
                    + (f"  (spec: {wl_spec})" if wl_spec else "")
                )
        return "\n".join(lines).strip()

    def seed_eval_from_dataset_traces(
        self,
        *,
        base_solution: Any,
        selected_workloads: list[Any] | None = None,
        baseline_latency_by_wl: Dict[str, float] | None = None,
    ) -> Any:
        """
        Best-effort: compute mean metrics across selected workloads from existing dataset traces,
        filtered by base_solution.name. Mirrors the nested helper previously in
        `kernel_generator_world_model.py`.
        """
        # Local import to avoid a hard dependency chain at module import time.
        from flashinfer_bench import EvaluationStatus
        from .task_base import EvalResult

        traces = self._list_traces(definition_name=self.name)
        by_wl: Dict[str, List[Any]] = {}
        for t in traces:
            try:
                if getattr(t, "solution", None) != getattr(base_solution, "name", None):
                    continue
                wl = getattr(t, "workload", None)
                wl_uuid = getattr(wl, "uuid", None) if wl is not None else None
                if not isinstance(wl_uuid, str) or not wl_uuid:
                    continue
                by_wl.setdefault(wl_uuid, []).append(t)
            except Exception:
                continue

        sw = list(selected_workloads) if selected_workloads is not None else list(self._selected_workloads)
        bl = dict(baseline_latency_by_wl) if baseline_latency_by_wl is not None else dict(self._baseline_latency_by_wl)

        passed_count = 0
        speedups: List[float] = []
        vs_base: List[float] = []
        latencies: List[float] = []
        for wl in sw:
            wl_uuid = wl.workload.uuid
            wl_traces = by_wl.get(wl_uuid, [])
            passed = [
                t
                for t in wl_traces
                if getattr(getattr(t, "evaluation", None), "status", None) == EvaluationStatus.PASSED
                and getattr(getattr(getattr(t, "evaluation", None), "performance", None), "latency_ms", None)
                is not None
            ]
            if passed:
                passed_count += 1
                best_t = max(passed, key=lambda t: float(t.evaluation.performance.speedup_factor or 0.0))
                try:
                    sp = float(best_t.evaluation.performance.speedup_factor or 0.0)
                    lat = float(best_t.evaluation.performance.latency_ms or 0.0)
                except Exception:
                    sp, lat = 0.0, 0.0
                if sp > 0:
                    speedups.append(sp)
                if lat > 0:
                    latencies.append(lat)
                    if wl_uuid in bl and lat > 0:
                        try:
                            vs_base.append(float(bl[wl_uuid]) / float(lat))
                        except Exception:
                            pass

        mean_speedup = (sum(speedups) / len(speedups)) if speedups else 0.0
        mean_vs_baseline = (sum(vs_base) / len(vs_base)) if vs_base else -1.0
        mean_latency = (sum(latencies) / len(latencies)) if latencies else 0.0
        all_passed = (passed_count == len(sw)) if sw else False

        vb_val = float(mean_vs_baseline) if (all_passed and mean_vs_baseline > 0) else None
        sp_val = float(mean_speedup) if mean_speedup > 0 else None
        score_name = (
            "mean_vs_baseline"
            if vb_val is not None
            else ("mean_speedup_vs_ref" if sp_val is not None else None)
        )
        score_value = vb_val if vb_val is not None else (sp_val if sp_val is not None else None)
        return EvalResult(
            status=("passed" if all_passed else ("partial" if passed_count > 0 else "unknown")),
            latency_ms=(float(mean_latency) if mean_latency > 0 else None),
            reference_latency_ms=None,
            mean_vs_baseline_factor=vb_val,
            speedup_factor=sp_val,
            log_excerpt="",
            metrics={
                "score": score_value,
                "score_name": score_name,
            },
        )


    # -------- Environment / baseline helpers --------
    @staticmethod
    def current_hardware_key() -> Optional[str]:
        """
        Best-effort current hardware key (lowercased), matching what flashinfer-bench stores in traces.
        """
        try:
            import torch  # type: ignore

            device = "cuda:0" if torch.cuda.is_available() else "cpu"
        except Exception:
            device = "cpu"
        try:
            from flashinfer_bench.utils import hardware_from_device

            hw = hardware_from_device(device)
            return hw.lower() if isinstance(hw, str) else None
        except Exception:
            return None

    def compute_baseline_latency_by_workload(
        self,
        *,
        definition_name: str,
        selected_workloads: list[Any],
        baseline_solution: Optional[str],
    ) -> Dict[str, float]:
        """
        Match existing behavior: read dataset traces and pick the best (min latency) PASSED baseline trace
        per workload, filtered to the current hardware.
        """
        if baseline_solution is None:
            return {}

        # Lazily import status enum so generator code doesn't need it.
        from flashinfer_bench import EvaluationStatus

        current_hw_key = self.current_hardware_key()
        wl_set = {getattr(getattr(wl, "workload", None), "uuid", None) for wl in selected_workloads}
        wl_set = {x for x in wl_set if isinstance(x, str) and x}

        by_wl: Dict[str, float] = {}
        for t in self._list_traces(definition_name=definition_name):
            try:
                if getattr(t, "solution", None) != baseline_solution:
                    continue
                ev = getattr(t, "evaluation", None)
                if ev is None or getattr(ev, "status", None) != EvaluationStatus.PASSED:
                    continue
                # Hardware filter
                hw = getattr(getattr(ev, "environment", None), "hardware", None)
                hw_key = hw.lower() if isinstance(hw, str) else None
                if current_hw_key is not None and hw_key != current_hw_key:
                    continue

                wl_uuid = getattr(getattr(t, "workload", None), "uuid", None)
                if not (isinstance(wl_uuid, str) and wl_uuid in wl_set):
                    continue
                perf = getattr(ev, "performance", None)
                lat = getattr(perf, "latency_ms", None) if perf is not None else None
                if lat is None:
                    continue
                lat_f = float(lat)
                prev = by_wl.get(wl_uuid)
                if prev is None or lat_f < prev:
                    by_wl[wl_uuid] = lat_f
            except Exception:
                continue
        return by_wl

    # -------- Evaluation --------
    def run_benchmark(
        self,
        *,
        solution: Any,
        config: FlashInferBenchEvalConfig | None = None,
        dump_traces: bool = False,
        round_num: int | None = None,
    ) -> Any:
        """
        Benchmark + extract traces + aggregate into EvalResult (and cache last-round feedback/logs).
        This is the only API generators should call.
        """
        from flashinfer_bench import Benchmark, BenchmarkConfig, TraceSet

        self._ensure_selected_workloads_prepared()
        self._prepare_baseline_if_needed()
        # Convert task solution -> backend solution (flashinfer-bench).
        if isinstance(solution, TaskSolution):
            solution_backend = self._to_backend_solution(solution)
        else:
            solution_backend = solution

        definition = self._require_definition()
        def_name = self.name
        wls = list(self._selected_workloads)
        # IMPORTANT: keep dataset root on the TraceSet even when dump_traces=False.
        # flashinfer-bench runners use `trace_set.root` to locate workload blobs / safetensors inputs.
        # Upstream example code always sets root=traceset.root for benchmarking.
        root_for_runner = getattr(self._traceset, "root", None)
        temp_traceset = TraceSet(
            root=root_for_runner,
            definitions={def_name: definition},
            solutions={def_name: [solution_backend]},
            workloads={def_name: list(wls)},
            traces={def_name: []},
        )
        cfg = config or self._eval_config
        bench_cfg = BenchmarkConfig(
            warmup_runs=int(cfg.warmup_runs),
            iterations=int(cfg.iterations),
            num_trials=int(cfg.num_trials),
            rtol=float(cfg.rtol),
            atol=float(cfg.atol),
            use_isolated_runner=bool(cfg.use_isolated_runner),
            parallel_workloads=bool(cfg.parallel_workloads),
            max_parallel_workloads=int(cfg.max_parallel_workloads),
        )
        benchmark = Benchmark(temp_traceset, bench_cfg)
        result_traceset = benchmark.run_all(dump_traces=bool(dump_traces))
        traces = self.extract_traces(result_traceset)

        info = self.summarize_round_and_select_feedback_trace(
            traces=traces,
            feedback_trace_selector=self._feedback_trace_selector,
        )
        self._last_round_passed_count = int(info.get("passed_count", 0))
        self._last_round_total_workloads = int(info.get("total_workloads", 0))
        self._last_round_summary_line = str(info.get("summary_line", "") or "")
        self._last_round_feedback_trace = info.get("prev_trace_for_prompt", None)
        try:
            if self._last_round_feedback_trace is not None:
                self._last_round_trace_logs_for_prompt = self.trace_logs_for_prompt(
                    self._last_round_feedback_trace, omit_when_passed=True
                )
            else:
                self._last_round_trace_logs_for_prompt = ""
        except Exception:
            self._last_round_trace_logs_for_prompt = ""

        # Preserve prior logging: print the summary line once per benchmark.
        try:
            if self._last_round_summary_line.strip():
                print(self._last_round_summary_line, flush=True)
        except Exception:
            pass

        er = self.eval_result_from_traces(traces=traces)
        try:
            all_passed = bool(info.get("all_passed", False))
            if all_passed:
                er.status = "passed"
            else:
                # Prefer the concrete failure status from the selected feedback trace (compile error, runtime error, etc.)
                # so WM "too hard" updates are grounded. Also clear perf fields to avoid partial perf evidence.
                st_detail = None
                try:
                    t = self._last_round_feedback_trace
                    ev = getattr(t, "evaluation", None) if t is not None else None
                    st0 = getattr(ev, "status", None) if ev is not None else None
                    st_detail = getattr(st0, "value", None) if st0 is not None else None
                    if st_detail is None and st0 is not None:
                        st_detail = str(st0)
                except Exception:
                    st_detail = None
                er.status = str(st_detail or "failed")
                er.latency_ms = None
                er.reference_latency_ms = None
                er.mean_vs_baseline_factor = None
                er.speedup_factor = None
        except Exception:
            pass
        try:
            t = self._last_round_feedback_trace
            ev = getattr(t, "evaluation", None) if t is not None else None
            perf = getattr(ev, "performance", None) if ev is not None else None
            ref = getattr(perf, "reference_latency_ms", None) if perf is not None else None
            if ref is not None:
                er.reference_latency_ms = ref
            log = getattr(ev, "log", None) if ev is not None else None
            if log is not None:
                er.log_excerpt = str(log or "")[:800]
        except Exception:
            pass
        return er

    def run_final_evaluation(
        self,
        *,
        solutions: list[TaskSolution],
        config: FlashInferBenchEvalConfig | None = None,
        dump_traces: bool = False,
        workload_limit: int | None = None,
    ) -> dict[str, Any]:
        """
        Final evaluation helper for scripts/CLI.

        - Evaluates the provided solutions over **all** workloads for this definition (optionally limited).
        - Writes traces into dataset root only when dump_traces=True (task-owned).
        - Returns a JSON-friendly report (no W&B types).
        """
        from flashinfer_bench import Benchmark, BenchmarkConfig, TraceSet

        definition = self._require_definition()
        def_name = self.name
        all_wls = list(self._list_workloads(definition_name=def_name))
        if workload_limit is not None and int(workload_limit) > 0:
            all_wls = all_wls[: int(workload_limit)]

        # Convert to backend solutions.
        backend_solutions: list[Any] = []
        requested_names: list[str] = []
        for sol in solutions or []:
            if sol is None:
                continue
            requested_names.append(str(getattr(sol, "name", "") or ""))
            try:
                backend_solutions.append(self._to_backend_solution(sol))
            except Exception:
                backend_solutions.append(sol)

        # Include baseline solution in the eval run (if configured) so vs_base can be computed
        # without touching dataset traces.
        baseline_name = str(self._init_baseline_solution_name or "").strip() if self._init_baseline_solution_name else ""
        baseline_backend = None
        if baseline_name and baseline_name not in requested_names:
            try:
                baseline_backend = self._traceset.get_solution(baseline_name)
            except Exception:
                baseline_backend = None
            if baseline_backend is not None:
                backend_solutions.append(baseline_backend)

        # Keep dataset root so runner can read workload blobs; dump_traces controls persistence.
        root_for_runner = getattr(self._traceset, "root", None)
        temp_traceset = TraceSet(
            root=root_for_runner,
            definitions={def_name: definition},
            solutions={def_name: list(backend_solutions)},
            workloads={def_name: list(all_wls)},
            traces={def_name: []},
        )

        cfg = config or self._eval_config
        bench_cfg = BenchmarkConfig(
            warmup_runs=int(cfg.warmup_runs),
            iterations=int(cfg.iterations),
            num_trials=int(cfg.num_trials),
            rtol=float(cfg.rtol),
            atol=float(cfg.atol),
            use_isolated_runner=bool(cfg.use_isolated_runner),
            parallel_workloads=bool(cfg.parallel_workloads),
            max_parallel_workloads=int(cfg.max_parallel_workloads),
        )
        benchmark = Benchmark(temp_traceset, bench_cfg)
        result_traceset = benchmark.run_all(dump_traces=bool(dump_traces))
        traces = self.extract_traces(result_traceset)

        # Group traces: solution -> workload_uuid -> list[trace]
        by_sol_wl: dict[str, dict[str, list[Any]]] = {}
        for t in traces:
            try:
                sol_name = str(getattr(t, "solution", "") or "")
                wl_uuid = str(getattr(getattr(t, "workload", None), "uuid", "") or "")
                if not sol_name or not wl_uuid:
                    continue
                by_sol_wl.setdefault(sol_name, {}).setdefault(wl_uuid, []).append(t)
            except Exception:
                continue

        # Baseline latencies keyed by (wl_uuid, hw_key).
        baseline_lat_by_key: dict[tuple[str, str | None], float] = {}
        if baseline_name:
            for wl_uuid, ts in (by_sol_wl.get(baseline_name, {}) or {}).items():
                for t in ts:
                    if not self.is_passed_trace(t):
                        continue
                    try:
                        ev = getattr(t, "evaluation", None)
                        perf = getattr(ev, "performance", None) if ev is not None else None
                        lat = getattr(perf, "latency_ms", None) if perf is not None else None
                        if lat is None:
                            continue
                        hw = getattr(getattr(ev, "environment", None), "hardware", None) if ev is not None else None
                        hw_key = hw.lower() if isinstance(hw, str) else None
                        k = (wl_uuid, hw_key)
                        lat_f = float(lat)
                        prev = baseline_lat_by_key.get(k)
                        if prev is None or lat_f < prev:
                            baseline_lat_by_key[k] = lat_f
                    except Exception:
                        continue

        # Workloads meta for report readability.
        workloads_meta: list[dict[str, Any]] = []
        for wt in all_wls:
            try:
                wl = getattr(wt, "workload", None)
                wl_uuid = str(getattr(wl, "uuid", "") or "")
                axes = getattr(wl, "axes", None)
                axes_str = ""
                if isinstance(axes, dict) and axes:
                    try:
                        axes_str = ", ".join(f"{k}={axes[k]}" for k in sorted(axes.keys()))
                    except Exception:
                        axes_str = ""
                workloads_meta.append({"workload_uuid": wl_uuid, "axes": axes_str})
            except Exception:
                continue

        def _status_str(t: Any) -> str:
            try:
                ev = getattr(t, "evaluation", None)
                st0 = getattr(ev, "status", None) if ev is not None else None
                v = getattr(st0, "value", None) if st0 is not None else None
                return str(v if v is not None else (st0 or "")).strip() or "unknown"
            except Exception:
                return "unknown"

        def _hw_str(t: Any) -> str | None:
            try:
                ev = getattr(t, "evaluation", None)
                hw = getattr(getattr(ev, "environment", None), "hardware", None) if ev is not None else None
                return str(hw) if hw is not None else None
            except Exception:
                return None

        def _perf_vals(t: Any) -> tuple[float | None, float | None, float | None]:
            try:
                ev = getattr(t, "evaluation", None)
                perf = getattr(ev, "performance", None) if ev is not None else None
                lat = getattr(perf, "latency_ms", None) if perf is not None else None
                ref = getattr(perf, "reference_latency_ms", None) if perf is not None else None
                sp = getattr(perf, "speedup_factor", None) if perf is not None else None
                return (
                    float(lat) if isinstance(lat, (int, float)) else (float(lat) if lat is not None else None),
                    float(ref) if isinstance(ref, (int, float)) else (float(ref) if ref is not None else None),
                    float(sp) if isinstance(sp, (int, float)) else (float(sp) if sp is not None else None),
                )
            except Exception:
                return (None, None, None)

        # Build per-solution reports (exclude baseline row unless it was explicitly requested).
        out_solutions: list[dict[str, Any]] = []
        total_wl = len([m for m in workloads_meta if m.get("workload_uuid")])
        for sol in solutions or []:
            sol_name = str(getattr(sol, "name", "") or "")
            if not sol_name:
                continue
            per_wl: list[dict[str, Any]] = []
            passed = 0
            speedups: list[float] = []
            latencies: list[float] = []
            vs_base_ratios: list[float] = []

            for m in workloads_meta:
                wl_uuid = str(m.get("workload_uuid", "") or "")
                ts = (by_sol_wl.get(sol_name, {}) or {}).get(wl_uuid, []) if wl_uuid else []
                chosen = None
                if ts:
                    passed_ts = [t for t in ts if self.is_passed_trace(t)]
                    if passed_ts:
                        try:
                            chosen = max(
                                passed_ts,
                                key=lambda t: float(getattr(getattr(getattr(t, "evaluation", None), "performance", None), "speedup_factor", 0.0) or 0.0),
                            )
                        except Exception:
                            chosen = passed_ts[0]
                    else:
                        chosen = ts[0]

                if chosen is None:
                    per_wl.append(
                        {
                            "workload_uuid": wl_uuid,
                            "axes": m.get("axes", ""),
                            "status": "missing",
                            "latency_ms": None,
                            "ref_latency_ms": None,
                            "speedup": None,
                            "hardware": None,
                            "vs_base": None,
                        }
                    )
                    continue

                st = _status_str(chosen)
                hw = _hw_str(chosen)
                lat, ref, sp = _perf_vals(chosen)
                vs_base_ratio: float | None = None
                if str(st).strip().lower() == "passed":
                    passed += 1
                    if isinstance(sp, (int, float)):
                        speedups.append(float(sp))
                    if isinstance(lat, (int, float)) and float(lat) > 0:
                        latencies.append(float(lat))
                        if baseline_name:
                            hw_key = hw.lower() if isinstance(hw, str) else None
                            b = baseline_lat_by_key.get((wl_uuid, hw_key))
                            if b is not None:
                                try:
                                    ratio = float(b) / float(lat)
                                    vs_base_ratio = float(ratio)
                                    vs_base_ratios.append(float(ratio))
                                except Exception:
                                    pass
                per_wl.append(
                    {
                        "workload_uuid": wl_uuid,
                        "axes": m.get("axes", ""),
                        "status": st,
                        "latency_ms": lat,
                        "ref_latency_ms": ref,
                        "speedup": sp,
                        "hardware": hw,
                        "vs_base": vs_base_ratio,
                    }
                )

            mean_speedup = (sum(speedups) / float(len(speedups))) if speedups else None
            mean_latency = (sum(latencies) / float(len(latencies))) if latencies else None
            mean_vs_base = (sum(vs_base_ratios) / float(len(vs_base_ratios))) if vs_base_ratios else None
            out_solutions.append(
                {
                    "solution": sol_name,
                    "passed_workloads": int(passed),
                    "total_workloads": int(total_wl),
                    "mean_speedup": mean_speedup,
                    "mean_latency_ms": mean_latency,
                    "mean_vs_base": mean_vs_base,
                    "vs_base_ratios": list(vs_base_ratios),
                    "workloads": per_wl,
                }
            )

        # Print per-workload table(s) for final eval (kept for parity with previous script behavior).
        for row in out_solutions:
            try:
                sol_name = str(row.get("solution", "") or "")
                wl_rows = list(row.get("workloads", []) or [])
                if not sol_name or not wl_rows:
                    continue
                print(f"[{def_name}] Final eval per-workload results for solution: {sol_name}", flush=True)
                print(
                    "workload_uuid                      | axes                                             | status   | speedup(x) | latency(ms) | ref_latency(ms) | vs_base(x)",
                    flush=True,
                )
                print(
                    "-----------------------------------+--------------------------------------------------+----------+------------+------------+----------------+-----------",
                    flush=True,
                )
                for wr in wl_rows:
                    wl_uuid = str(wr.get("workload_uuid", "") or "")
                    axes_str = str(wr.get("axes", "") or "")
                    st = str(wr.get("status", "") or "")
                    sp = wr.get("speedup", None)
                    lat = wr.get("latency_ms", None)
                    ref = wr.get("ref_latency_ms", None)
                    vb = wr.get("vs_base", None)

                    # Keep table readable.
                    if len(axes_str) > 50:
                        axes_str = axes_str[:50] + "..."

                    sp_s = f"{float(sp):.2f}" if isinstance(sp, (int, float)) else "-"
                    lat_s = f"{float(lat):.3f}" if isinstance(lat, (int, float)) else "-"
                    ref_s = f"{float(ref):.3f}" if isinstance(ref, (int, float)) else "-"
                    vb_s = f"{float(vb):.2f}" if isinstance(vb, (int, float)) else "-"
                    line = (
                        f"{wl_uuid:<35} | {axes_str:<50} | {st:<8} | {sp_s:>10} | {lat_s:>10} | {ref_s:>14} | {vb_s:>9}"
                    )
                    print(line, flush=True)
                print("", flush=True)
            except Exception:
                continue

        # Print a compact per-solution summary line (mirrors prior script behavior).
        for row in out_solutions:
            try:
                sol_name = str(row.get("solution", "") or "")
                passed_workloads = int(row.get("passed_workloads", 0) or 0)
                total_workloads = int(row.get("total_workloads", 0) or 0)
                pass_rate = (passed_workloads / float(total_workloads) * 100.0) if total_workloads > 0 else 0.0
                avg_speedup = row.get("mean_speedup")
                avg_latency = row.get("mean_latency_ms")
                mvb = row.get("mean_vs_base")
                mvb_text = f"{float(mvb):.2f}x" if isinstance(mvb, (int, float)) else "-"
                sp_text = f"{float(avg_speedup):.2f}x" if isinstance(avg_speedup, (int, float)) else "-"
                lat_text = f"{float(avg_latency):.3f} ms" if isinstance(avg_latency, (int, float)) else "-"
                print(
                    f"[{def_name}] Final eval for {sol_name}: workloads={passed_workloads}/{total_workloads} "
                    f"({pass_rate:.1f}%) | mean_speedup={sp_text} | mean_latency={lat_text} | mean_vs_base={mvb_text}",
                    flush=True,
                )
            except Exception:
                continue

        return {
            "definition": def_name,
            "baseline_solution": (baseline_name or None),
            "total_workloads": int(total_wl),
            "workloads": workloads_meta,
            "solutions": out_solutions,
        }

    def extract_traces(self, result_traceset: Any) -> list[Any]:
        """
        Extract the definition-specific traces from a flashinfer-bench TraceSet result.
        Kept behind Task so generators don't reach into result_traceset internals.
        """
        try:
            traces_map = getattr(result_traceset, "traces", None)
            if isinstance(traces_map, dict):
                return list(traces_map.get(self.name, []) or [])
        except Exception:
            pass
        return []

    def eval_result_from_traces(
        self,
        *,
        selected_workloads: list[Any] | None = None,
        baseline_latency_by_wl: Dict[str, float] | None = None,
        traces: list[Any],
        status_if_partial: str = "partial",
        status_if_failed: str = "failed",
    ) -> Any:
        """
        Aggregate flashinfer-bench traces into a task-neutral EvalResult.

        Mirrors the aggregation logic previously embedded in generator loops:
        - consider a workload PASSED if it has >=1 passed trace with perf data
        - pick the best passed trace per workload by speedup_factor
        - aggregate mean latency/speedup across workloads with a PASSED trace
        - compute mean vs baseline factor if baseline latencies are available
        """
        from .task_base import EvalResult

        sw = list(selected_workloads) if selected_workloads is not None else list(self._selected_workloads)
        bl = dict(baseline_latency_by_wl) if baseline_latency_by_wl is not None else dict(self._baseline_latency_by_wl)

        passed_count = 0
        speedups: list[float] = []
        vs_base: list[float] = []
        latencies: list[float] = []

        by_wl: Dict[str, List[Any]] = {}
        for t in traces:
            try:
                by_wl.setdefault(t.workload.uuid, []).append(t)
            except Exception:
                continue

        for wl in sw:
            wl_uuid = wl.workload.uuid
            wl_traces = by_wl.get(wl_uuid, [])
            passed = [t for t in wl_traces if self.is_passed_trace(t)]
            if not passed:
                continue
            passed_count += 1
            best_t = max(passed, key=lambda t: float(t.evaluation.performance.speedup_factor or 0.0))
            try:
                sp = float(best_t.evaluation.performance.speedup_factor)
                lat = float(best_t.evaluation.performance.latency_ms)
            except Exception:
                sp, lat = 0.0, 0.0
            if sp > 0:
                speedups.append(sp)
            if lat > 0:
                latencies.append(lat)
            if wl_uuid in bl and lat > 0:
                try:
                    vs_base.append(float(bl[wl_uuid]) / float(lat))
                except Exception:
                    pass

        mean_speedup = (sum(speedups) / len(speedups)) if speedups else 0.0
        mean_vs_baseline = (sum(vs_base) / len(vs_base)) if vs_base else -1.0
        mean_latency = (sum(latencies) / len(latencies)) if latencies else 0.0
        all_passed = (passed_count == len(sw)) if sw else False

        if all_passed:
            status = "passed"
        else:
            status = (status_if_partial if passed_count > 0 else status_if_failed)

        vb_val = float(mean_vs_baseline) if (all_passed and mean_vs_baseline > 0) else None
        sp_val = float(mean_speedup) if mean_speedup > 0 else None
        score_name = (
            "mean_vs_baseline"
            if vb_val is not None
            else ("mean_speedup_vs_ref" if sp_val is not None else None)
        )
        score_value = vb_val if vb_val is not None else (sp_val if sp_val is not None else None)
        return EvalResult(
            status=str(status),
            latency_ms=(float(mean_latency) if mean_latency > 0 else None),
            reference_latency_ms=None,
            mean_vs_baseline_factor=vb_val,
            speedup_factor=sp_val,
            log_excerpt="",
            metrics={
                "score": score_value,
                "score_name": score_name,
            },
        )


