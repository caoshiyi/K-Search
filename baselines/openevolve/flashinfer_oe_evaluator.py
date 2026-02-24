"""Minimal OpenEvolve evaluator for flashinfer-bench.

OpenEvolve calls a module-level `evaluate(program_text)` function.

This evaluator is intentionally *simple*:
- reads `program_text` either as a file path or inline source text
- wraps it into a single-file `flashinfer_bench.Solution`
- runs `flashinfer_bench.Benchmark` for one dataset definition
- returns a small dict with `success` + `final_score` (mean vs_base)

Configuration is via environment variables (portable across subprocess eval):
- OE_DATASET_PATH (required)
- OE_DEFINITION (required)
- OE_LANGUAGE (optional, default: triton)
- OE_WARMUP_RUNS / OE_ITERATIONS / OE_NUM_TRIALS / OE_RTOL / OE_ATOL
- OE_USE_ISOLATED_RUNNER (0/1)
- OE_NUM_EVAL_WORKLOAD (optional)
- OE_BASELINE_SOLUTION (optional): solution name in the dataset to use as the baseline for vs_base.
"""

from __future__ import annotations

from dataclasses import dataclass
import random
import re
from typing import Any, Dict, List, Optional, Sequence, Tuple
from pathlib import Path

def _maybe_read_text(path_or_text: str) -> str:
    """If the input points to an existing file, read it; otherwise treat as inline text."""
    s = (path_or_text or "").strip()
    if not s:
        return ""
    # If `s` is very long (inline code), Path(s) can raise OSError (e.g. [Errno 36] File name too long).
    try:
        p = Path(s)
        if p.exists() and p.is_file():
            return p.read_text(encoding="utf-8")
    except OSError:
        return path_or_text
    return path_or_text

def _format_trace_log(trace) -> str:
    """Format a flashinfer-bench Trace evaluation into a human-readable log string."""

    # Prefix with a minimal workload spec (uuid/axes/inputs) for context.
    try:
        wl = getattr(trace, "workload", None)
        if wl is not None:
            lines: List[str] = []
            lines.append(f"uuid: {getattr(wl, 'uuid', '')}")
            axes = getattr(wl, "axes", {}) or {}
            if axes:
                axes_str = ", ".join(f"{k}={axes[k]}" for k in sorted(axes.keys()))
                lines.append(f"axes: {axes_str}")
            inputs = getattr(wl, "inputs", {}) or {}
            if inputs:
                lines.append("inputs:")
                for in_name in sorted(inputs.keys()):
                    spec = inputs[in_name]
                    typ = getattr(spec, "type", None)
                    if typ == "scalar":
                        lines.append(f"- {in_name}: scalar(value={getattr(spec, 'value', None)})")
                    elif typ == "random":
                        lines.append(f"- {in_name}: random")
                    elif typ == "safetensors":
                        lines.append(f"- {in_name}: safetensors")
                    else:
                        lines.append(f"- {in_name}: {typ or type(spec).__name__}")
            wl_spec = "\n".join(lines).strip()
        else:
            wl_spec = ""
    except Exception:
        wl_spec = ""

    if getattr(trace, "is_workload_trace", lambda: False)() or not getattr(trace, "evaluation", None):
        base = "No evaluation logs available (workload-only trace)"
        return (wl_spec + "\n\n" + base).strip() if wl_spec else base

    ev = trace.evaluation
    eval_info = f"Status: {ev.status.value}\n"
    eval_info += f"Timestamp: {ev.timestamp}\n"

    if getattr(ev, "log", None):
        eval_info += f"\nExecution Log:\n{ev.log}\n"

    if getattr(ev, "correctness", None):
        eval_info += f"Max relative error: {ev.correctness.max_relative_error}\n"
        eval_info += f"Max absolute error: {ev.correctness.max_absolute_error}\n"

    if getattr(ev, "performance", None):
        eval_info += f"Latency: {ev.performance.latency_ms}ms\n"
        eval_info += f"Reference latency: {ev.performance.reference_latency_ms}ms\n"
        eval_info += f"Speedup factor: {ev.performance.speedup_factor}x\n"

    return (wl_spec + "\n\n" + eval_info).strip() if wl_spec else eval_info


def _evaluation_result(
    *,
    combined_score: float,
    latency_ms: Optional[float],
    num_passed: int,
    trace_log: str,
    extra_metrics: Optional[Dict[str, Any]] = None,
):
    """Create an OpenEvolve EvaluationResult.

    Minimal contract (as requested):
            - metrics: {combined_score, latency_ms, num_passed}
            - artifacts: {'trace_log': ...}
    """

    # Use a dynamic import so this file remains importable even when OpenEvolve
    # isn't installed in the editor/typechecker environment.
    import importlib

    EvaluationResult = importlib.import_module("openevolve.evaluation_result").EvaluationResult

    out_metrics: Dict[str, Any] = {
        "combined_score": float(combined_score),
        "latency_ms": (None if latency_ms is None else float(latency_ms)),
        "num_passed": int(num_passed),
    }
    if extra_metrics:
        out_metrics.update(extra_metrics)
    out_artifacts = {"trace_log": trace_log}
    return EvaluationResult(metrics=out_metrics, artifacts=out_artifacts)


@dataclass(frozen=True)
class FlashInferEvaluatorConfig:
    dataset_path: str
    definition: str
    language: str = "triton"
    # Hardware descriptor used in BuildSpec.target_hardware.
    # For consistency with `KernelGenerator`, this should be a GPU model string like "H100".
    # If unspecified, we fall back to "cuda".
    target_gpu: str = "H100"
    warmup_runs: int = 5
    iterations: int = 100
    num_trials: int = 1
    rtol: float = 1e-2
    atol: float = 1e-2
    use_isolated_runner: bool = False
    # Per-workload evaluation timeout inside flashinfer-bench Benchmark (seconds).
    # IMPORTANT: If this is too large and you evaluate many workloads, the overall evaluator call may exceed
    # OpenEvolve's own evaluator timeout, causing OpenEvolve to mark the eval as (timeout=1) and drop combined_score.
    timeout_seconds: int = 300
    # Enable flashinfer-bench's workload-parallel scheduler (BenchmarkConfig.parallel_workloads).
    # This is particularly useful for OpenEvolve where we typically evaluate only 1 solution.
    parallel_workloads: bool = False
    # Max concurrent workloads when parallel_workloads is enabled (0 => auto based on visible CUDA devices).
    max_parallel_workloads: int = 0
    num_eval_workload: int = 0  # 0 = all
    # If set, compute vs_base = baseline_latency_ms / candidate_latency_ms using dataset traces for this baseline solution.
    baseline_solution: Optional[str] = None
    # If true, print a per-workload performance table (like examples/generate_kernels_and_eval.py).
    verbose_table: bool = False
    # Workload selection (for faster evaluation during evolution)
    feedback_workloads: Optional[List[str]] = None
    num_feedback_workloads: int = 2

    @staticmethod
    def from_config(cfg: Dict[str, Any]) -> "FlashInferEvaluatorConfig":
        """Construct an evaluator config from a mapping.

        Required keys:
          - dataset_path
          - definition

        Notes:
          - We also accept the same keys without underscores for convenience
            (e.g. `datasetpath`).
        """

        def _get(key: str, default: Any = None) -> Any:
            return cfg.get(key, cfg.get(key.replace("_", ""), default))

        dataset_path = _get("dataset_path")
        definition = _get("definition")
        if not dataset_path or not definition:
            missing = []
            if not dataset_path:
                missing.append("dataset_path")
            if not definition:
                missing.append("definition")
            raise ValueError(f"Missing required evaluator config key(s): {', '.join(missing)}")

        feedback_workloads_raw = _get("feedback_workloads", None)
        feedback_workloads = list(feedback_workloads_raw) if feedback_workloads_raw else None
        baseline_solution = _get("baseline_solution", _get("baseline", None))
        verbose_table = bool(_get("verbose_table", _get("verbose", False)))
        timeout_seconds = int(_get("timeout_seconds", _get("timeout", 300)))
        parallel_workloads = bool(_get("parallel_workloads", False))
        max_parallel_workloads = int(_get("max_parallel_workloads", 0))

        return FlashInferEvaluatorConfig(
            dataset_path=str(dataset_path),
            definition=str(definition),
            language=str(_get("language", "triton")),
            target_gpu=str(_get("target_gpu", _get("target_hardware", "H100"))),
            warmup_runs=int(_get("warmup_runs", 5)),
            iterations=int(_get("iterations", 100)),
            num_trials=int(_get("num_trials", 1)),
            rtol=float(_get("rtol", 1e-2)),
            atol=float(_get("atol", 1e-2)),
            use_isolated_runner=bool(_get("use_isolated_runner", False)),
            timeout_seconds=int(timeout_seconds),
            parallel_workloads=bool(parallel_workloads),
            max_parallel_workloads=int(max_parallel_workloads),
            num_eval_workload=int(_get("num_eval_workload", 0)),
            baseline_solution=(str(baseline_solution).strip() if baseline_solution else None),
            verbose_table=bool(verbose_table),
            feedback_workloads=feedback_workloads,
            num_feedback_workloads=int(_get("num_feedback_workloads", 2)),
        )


def _parse_cuda_xml_files(code: str) -> Dict[str, str]:
    """Parse KernelGenerator XML blocks into a {filename: content} dict."""
    patterns = {
        "kernel.h": r'<header_file name="kernel\.h">(.*?)</header_file>',
        "kernel.cu": r'<cuda_file name="kernel\.cu">(.*?)</cuda_file>',
        "main.cpp": r'<cpp_file name="main\.cpp">(.*?)</cpp_file>',
    }
    out: Dict[str, str] = {}
    for fname, pat in patterns.items():
        m = re.search(pat, code, re.DOTALL)
        if m:
            out[fname] = m.group(1).strip()
    return out


def _baseline_latency_by_workload(
    trace_set: Any,
    *,
    definition_name: str,
    baseline_solution: str,
    current_hw_key: Optional[str],
) -> Dict[str, float]:
    """Return {workload_uuid: baseline_latency_ms} for a baseline solution from dataset traces.

    Matches KernelGenerator behavior:
    - only use PASSED traces
    - if current_hw_key is available, require trace.evaluation.environment.hardware to match
    - keep the best (min) baseline latency per workload
    """
    out: Dict[str, float] = {}
    traces = list(getattr(trace_set, "traces", {}).get(definition_name, []) or [])
    for t in traces:
        try:
            if getattr(t, "solution", None) != baseline_solution:
                continue
            ev = getattr(t, "evaluation", None)
            if ev is None:
                continue
            # Only PASSED baselines
            st = getattr(ev, "status", None)
            if str(st) not in ("EvaluationStatus.PASSED", "PASSED"):
                continue
            # Match hardware key if possible
            if current_hw_key:
                hw = getattr(getattr(ev, "environment", None), "hardware", None)
                hw_key = hw.lower() if isinstance(hw, str) else None
                if hw_key != current_hw_key:
                    continue
            perf = getattr(ev, "performance", None) if ev is not None else None
            lat = getattr(perf, "latency_ms", None) if perf is not None else None
            wl = getattr(t, "workload", None)
            wl_uuid = getattr(wl, "uuid", None) if wl is not None else None
            if not wl_uuid or not isinstance(lat, (int, float)) or float(lat) <= 0:
                continue
            prev = out.get(str(wl_uuid))
            out[str(wl_uuid)] = float(lat) if prev is None else min(float(prev), float(lat))
        except Exception:
            continue
    return out


def _score_from_traces(traceset, definition_name: str) -> Tuple[bool, Dict[str, Any]]:
    """Compute a simple aggregate score from TraceSet traces."""

    from flashinfer_bench.data import EvaluationStatus

    traces = traceset.traces.get(definition_name, [])
    total = 0
    passed = 0
    best_by_workload = {}
    for t in traces:
        if not getattr(t, "evaluation", None):
            continue
        total += 1
        if t.evaluation.status != EvaluationStatus.PASSED:
            continue
        passed += 1
        wl_uuid = t.workload.uuid
        prev = best_by_workload.get(wl_uuid)
        if prev is None or t.evaluation.performance.speedup_factor > prev.evaluation.performance.speedup_factor:
            best_by_workload[wl_uuid] = t

    workloads_passed = len(best_by_workload)
    if workloads_passed != total:
        return False, {
            "workloads_passed": 0,
            "mean_speedup": 0.0,
            "mean_latency_ms": None,
            "traces_total": total,
            "traces_passed": passed,
        }

    mean_speedup = sum(t.evaluation.performance.speedup_factor for t in best_by_workload.values()) / workloads_passed
    mean_latency = sum(t.evaluation.performance.latency_ms for t in best_by_workload.values()) / workloads_passed
    return True, {
        "workloads_passed": workloads_passed,
        "mean_speedup": float(mean_speedup),
        "mean_latency_ms": float(mean_latency),
        "traces_total": total,
        "traces_passed": passed,
    }


def _pick_failing_trace(traces: Sequence[Any]) -> Optional[Any]:
    """Pick one representative failing trace for feedback.

    Prefer a trace with a non-empty `evaluation.log` (most actionable), otherwise
    return the first failing trace.
    """

    try:
        failing = [
            t
            for t in (traces or [])
            if getattr(t, "evaluation", None) is not None
            and getattr(getattr(t, "evaluation", None), "status", None) is not None
            and str(getattr(getattr(t, "evaluation", None), "status", ""))
            not in ("EvaluationStatus.PASSED", "PASSED")
        ]
    except Exception:
        failing = []

    if not failing:
        return None

    for t in failing:
        try:
            lg = getattr(getattr(t, "evaluation", None), "log", None)
            if lg:
                return t
        except Exception:
            continue
    return failing[0]


class FlashInferEvaluator:
    """Callable evaluator object for OpenEvolve."""

    def __init__(self, cfg: FlashInferEvaluatorConfig):
        self.cfg = cfg

    # --- utils ---

    def raw_text_to_solution(self, *, raw_text: str) -> Tuple[Any, str]:
        """Convert a raw candidate text into a flashinfer-bench Solution.

        Matches the common formats used by `KernelGenerator._create_solution_from_code`:
        - CUDA: XML blocks for kernel.h/kernel.cu/main.cpp
        - Triton/Python: single-file program text

        Returns (solution, mode).
        """

        from flashinfer_bench.data import BuildSpec, Solution, SourceFile, SupportedLanguages

        lang = (self.cfg.language or "triton").lower()
        supported = {
            "python": SupportedLanguages.PYTHON,
            "triton": SupportedLanguages.TRITON,
            "cuda": SupportedLanguages.CUDA,
        }.get(lang, SupportedLanguages.TRITON)

        # Keep this stable for tracking/debugging in traces.
        author = "openevolve"

        assert lang == "cuda" and ("<cuda_file" in raw_text or "<header_file" in raw_text or "<cpp_file" in raw_text), "Invalid CUDA XML format"
        if lang == "cuda" and ("<cuda_file" in raw_text or "<header_file" in raw_text or "<cpp_file" in raw_text):
            files = _parse_cuda_xml_files(raw_text)
            sources = [SourceFile(path=fname, content=content) for fname, content in files.items()]
            # If main.cpp is missing, Benchmark will likely fail; we keep it minimal and let it fail.
            return (
                Solution(
                    name=f"oe_candidate_{abs(hash(raw_text)) % (10**10)}",
                    definition=self.cfg.definition,
                    author=author,
                    spec=BuildSpec(
                        language=supported,
                        target_hardware=[self.cfg.target_gpu],
                        entry_point="main.cpp::run",
                    ),
                    sources=sources,
                    description="OpenEvolve candidate (cuda xml)",
                ),
                "cuda_xml",
            )

        # Default: single-file
        path = "main.py" if lang in ("python", "triton") else "kernel.cu"
        entry_point = "main.py::run" if path.endswith(".py") else "main.cpp::run"
        return (
            Solution(
                name=f"oe_candidate_{abs(hash(raw_text)) % (10**10)}",
                definition=self.cfg.definition,
                author=author,
                spec=BuildSpec(
                    language=supported,
                    target_hardware=[self.cfg.target_gpu],
                    entry_point=entry_point,
                ),
                sources=[SourceFile(path=path, content=raw_text)],
                description="OpenEvolve candidate (single file)",
            ),
            "single_file",
        )

    def eval_workload_selection(self, trace_set) -> Sequence[Any]:
        """Select workloads to evaluate.

        - If cfg.feedback_workloads is provided: select those workload UUIDs in that order.
        - Else:
          - if cfg.num_feedback_workloads <= 0: evaluate ALL workloads
          - otherwise: randomly sample cfg.num_feedback_workloads from available workloads.
        """

        workloads = list(trace_set.workloads.get(self.cfg.definition, []) or [])
        if not workloads:
            return []

        if self.cfg.feedback_workloads:
            wanted = list(self.cfg.feedback_workloads)
            wanted_set = set(wanted)
            order = {w: i for i, w in enumerate(wanted)}
            selected = [wl for wl in workloads if wl.workload.uuid in wanted_set]
            selected.sort(key=lambda wl: order.get(wl.workload.uuid, 1 << 30))
            return selected

        k_req = int(self.cfg.num_feedback_workloads)
        if k_req <= 0:
            return workloads
        k = min(k_req, len(workloads))
        return random.sample(workloads, k=k) if len(workloads) > k else workloads

    def __call__(self, program_text: str):
        return self.evaluate(program_text)

    def evaluate(self, program_text: str):

        try:
            from flashinfer_bench import Benchmark, BenchmarkConfig
            from flashinfer_bench.data import TraceSet

            trace_set = TraceSet.from_path(self.cfg.dataset_path)
            if self.cfg.definition not in trace_set.definitions:
                print(f"Definition '{self.cfg.definition}' not found in dataset.")
                return _evaluation_result(
                    combined_score=0.0,
                    latency_ms=None,
                    num_passed=0,
                    trace_log=f"Definition '{self.cfg.definition}' not found in dataset.",
                )

            candidate_text = _maybe_read_text(program_text)
            solution, mode = self.raw_text_to_solution(raw_text=candidate_text)

            # Choose workloads to evaluate
            selected_workloads = list(self.eval_workload_selection(trace_set))
            if (not selected_workloads) and self.cfg.num_eval_workload and self.cfg.num_eval_workload > 0:
                # Fallback: take first N
                wl_list = trace_set.workloads.get(self.cfg.definition, [])
                if wl_list:
                    selected_workloads = list(wl_list[: min(len(wl_list), self.cfg.num_eval_workload)])

            # IMPORTANT: build a fresh TraceSet for evaluation (mirrors KernelGenerator)
            temp_traceset = TraceSet(
                root=trace_set.root,
                definitions={self.cfg.definition: trace_set.definitions[self.cfg.definition]},
                solutions={self.cfg.definition: [solution]},
                workloads={self.cfg.definition: list(selected_workloads)},
                traces={self.cfg.definition: []},
            )

            bench_cfg = BenchmarkConfig(
                warmup_runs=self.cfg.warmup_runs,
                iterations=self.cfg.iterations,
                num_trials=self.cfg.num_trials,
                rtol=self.cfg.rtol,
                atol=self.cfg.atol,
                use_isolated_runner=self.cfg.use_isolated_runner,
                timeout_seconds=int(getattr(self.cfg, "timeout_seconds", 300) or 300),
                parallel_workloads=bool(getattr(self.cfg, "parallel_workloads", False)),
                max_parallel_workloads=int(getattr(self.cfg, "max_parallel_workloads", 0) or 0),
                definitions=[self.cfg.definition],
                solutions=[solution.name],
            )

            bench = Benchmark(temp_traceset, bench_cfg)
            result_traceset = bench.run_all(dump_traces=False, resume=False)

            traces = list(result_traceset.traces.get(self.cfg.definition, []) or [])

            # Baseline perf map (optional): baseline_latency / candidate_latency.
            # Compute ONCE and reuse for:
            # - verbose per-workload table (vs_base column)
            # - aggregate mean_vs_base / combined_score
            baseline_name = str(self.cfg.baseline_solution or "").strip()
            current_hw_key: Optional[str] = None
            bl_by_wl: Dict[str, float] = {}
            if baseline_name:
                try:
                    from flashinfer_bench.utils import hardware_from_device

                    import torch  # type: ignore

                    dev = "cuda:0" if torch.cuda.is_available() else "cpu"
                    current_hw_key = hardware_from_device(dev).lower()
                except Exception:
                    current_hw_key = None
                bl_by_wl = _baseline_latency_by_workload(
                    trace_set,
                    definition_name=self.cfg.definition,
                    baseline_solution=baseline_name,
                    current_hw_key=current_hw_key,
                )

            # Candidate latencies per workload (best PASSED only).
            cand_lat_by_wl: Dict[str, float] = {}
            try:
                from flashinfer_bench.data import EvaluationStatus

                for t in traces:
                    ev = getattr(t, "evaluation", None)
                    if ev is None or getattr(ev, "status", None) != EvaluationStatus.PASSED:
                        continue
                    wl_uuid = getattr(getattr(t, "workload", None), "uuid", None)
                    lat = getattr(getattr(ev, "performance", None), "latency_ms", None)
                    if not isinstance(wl_uuid, str) or not wl_uuid.strip():
                        continue
                    if not isinstance(lat, (int, float)) or float(lat) <= 0:
                        continue
                    prev = cand_lat_by_wl.get(wl_uuid)
                    cand_lat_by_wl[wl_uuid] = float(lat) if prev is None else min(float(prev), float(lat))
            except Exception:
                cand_lat_by_wl = {}

            # vs_base per workload (baseline_latency / candidate_latency).
            vs_base_by_wl: Dict[str, float] = {}
            if baseline_name and cand_lat_by_wl and bl_by_wl:
                for wl_uuid, cand_lat in cand_lat_by_wl.items():
                    bl_lat = bl_by_wl.get(wl_uuid)
                    if isinstance(bl_lat, (int, float)) and float(bl_lat) > 0 and float(cand_lat) > 0:
                        vs_base_by_wl[wl_uuid] = float(bl_lat) / float(cand_lat)

            # Optional: verbose per-workload perf table (useful during evolution debugging).
            if bool(self.cfg.verbose_table):
                try:
                    from flashinfer_bench.data import EvaluationStatus

                    # Stable workload order from selection
                    wl_order = [w.workload.uuid for w in selected_workloads if getattr(w, "workload", None)]
                    wl_order = [w for w in wl_order if isinstance(w, str) and w.strip()]
                    wl_order = list(dict.fromkeys(wl_order))

                    # Index traces by workload uuid; keep best passed (max speedup) when multiple exist.
                    by_wl: Dict[str, Any] = {}
                    for t in traces:
                        try:
                            wl_uuid = getattr(getattr(t, "workload", None), "uuid", None)
                            if not isinstance(wl_uuid, str) or not wl_uuid.strip():
                                continue
                            prev = by_wl.get(wl_uuid)
                            if prev is None:
                                by_wl[wl_uuid] = t
                                continue
                            # Prefer PASSED with higher speedup
                            prev_ev = getattr(prev, "evaluation", None)
                            ev = getattr(t, "evaluation", None)
                            prev_pass = bool(prev_ev and prev_ev.status == EvaluationStatus.PASSED)
                            cur_pass = bool(ev and ev.status == EvaluationStatus.PASSED)
                            if cur_pass and not prev_pass:
                                by_wl[wl_uuid] = t
                                continue
                            if cur_pass and prev_pass:
                                prev_sp = float(getattr(getattr(prev_ev, "performance", None), "speedup_factor", 0.0) or 0.0)
                                cur_sp = float(getattr(getattr(ev, "performance", None), "speedup_factor", 0.0) or 0.0)
                                if cur_sp > prev_sp:
                                    by_wl[wl_uuid] = t
                        except Exception:
                            continue

                    print(f"[{self.cfg.definition}] Per-workload performance table (OpenEvolve evaluator)")
                    print(
                        "workload_uuid                      | status        | speedup(x) | latency(ms) | ref_latency(ms) | vs_base(x)"
                    )
                    print(
                        "-----------------------------------+---------------+------------+------------+----------------+----------"
                    )

                    for wl_uuid in wl_order:
                        t = by_wl.get(wl_uuid)
                        if t is None or not getattr(t, "evaluation", None):
                            print(f"{wl_uuid:<35} | N/A           |     -      |     -     |       -        |     -")
                            continue
                        ev = t.evaluation
                        st_val = getattr(ev.status, "value", None)
                        st = str(st_val) if isinstance(st_val, str) and st_val else str(ev.status)
                        perf = getattr(ev, "performance", None)
                        if perf is not None and ev.status == EvaluationStatus.PASSED:
                            sp = getattr(perf, "speedup_factor", None)
                            lat = getattr(perf, "latency_ms", None)
                            rlat = getattr(perf, "reference_latency_ms", None)
                            sp_s = f"{float(sp):.2f}" if isinstance(sp, (int, float)) else "-"
                            lat_s = f"{float(lat):.3f}" if isinstance(lat, (int, float)) else "-"
                            rlat_s = f"{float(rlat):.3f}" if isinstance(rlat, (int, float)) else "-"
                            vsb = "-"
                            if baseline_name:
                                v = vs_base_by_wl.get(str(wl_uuid))
                                if isinstance(v, (int, float)) and float(v) > 0:
                                    vsb = f"{float(v):.2f}"
                            print(
                                f"{wl_uuid:<35} | {st:<13} | {sp_s:>10} | {lat_s:>10} | {rlat_s:>14} | {vsb:>8}"
                            )
                        else:
                            print(f"{wl_uuid:<35} | {st:<13} |     -      |     -     |       -        |     -")
                except Exception:
                    # Do not let verbose logging break evaluation.
                    pass

            ok, metrics = _score_from_traces(result_traceset, self.cfg.definition)
            if not ok:
                print(f"Failed to score traces for definition '{self.cfg.definition}': {metrics}")
                # Mirror KernelGenerator: surface one failing trace's log for feedback.
                failing_trace = _pick_failing_trace(traces)
                if failing_trace is not None:
                    trace_log = _format_trace_log(failing_trace)
                else:
                    trace_log = f"Candidate failed. mode={mode}"
                return _evaluation_result(
                    combined_score=0.0,
                    latency_ms=None,
                    num_passed=0,
                    trace_log=trace_log,
                )

            speedup = float(metrics["mean_speedup"])
            latency_ms = metrics.get("mean_latency_ms", None)

            # Otherwise include logs for all evaluated traces (useful when fully passing).
            num_passed = int(metrics.get("workloads_passed", 0) or 0)
            trace_log = "\n\n".join(_format_trace_log(t) for t in traces) if traces else ""

            # Compute vs_base = baseline_latency / candidate_latency using dataset baseline traces.
            mean_vs_base: Optional[float] = None
            vs_base_note = ""
            if baseline_name:
                # Since ok==True implies all workloads passed, require baseline for every evaluated workload.
                if vs_base_by_wl and len(vs_base_by_wl) == int(num_passed):
                    mean_vs_base = sum(vs_base_by_wl.values()) / float(len(vs_base_by_wl))
                    vs_base_note = (
                        f"mean_vs_base={mean_vs_base} (baseline_solution={baseline_name}, hw_key={current_hw_key})"
                    )
                else:
                    mean_vs_base = None
                    vs_base_note = (
                        f"mean_vs_base=None (missing baseline traces for some workloads; baseline_solution={baseline_name}, hw_key={current_hw_key})"
                    )
            else:
                vs_base_note = "mean_vs_base=None (no baseline_solution configured)"

            # Use scaled vs_base as combined_score when configured.
            # Rationale: OpenEvolve may synthesize timeout metrics around ~0.5 when the evaluator call exceeds its
            # watchdog, which can incorrectly dominate if combined_score is a small ratio (~0.01-0.1). Scaling helps
            # keep successful evaluations clearly above timeout fallbacks.
            if baseline_name:
                combined = (float(mean_vs_base) * 100.0) if mean_vs_base is not None else 0.0
            else:
                combined = float(speedup)

            extra = {
                "mean_speedup": float(speedup),
                "mean_vs_base": (None if mean_vs_base is None else float(mean_vs_base)),
                "combined_score_scale": (100.0 if baseline_name else 1.0),
                "baseline_solution": (baseline_name if baseline_name else None),
            }
            trace_log = trace_log + f"\n\n[aggregate]\n{vs_base_note}\nmean_speedup={speedup}"

            return _evaluation_result(
                combined_score=combined,
                latency_ms=(None if latency_ms is None else float(latency_ms)),
                num_passed=num_passed,
                trace_log=trace_log,
                extra_metrics=extra,
            )

        except Exception as e:
            print(f"Evaluator crashed (caught): {e}")
            return _evaluation_result(
                combined_score=0.0,
                latency_ms=None,
                num_passed=0,
                trace_log=f"Evaluator crashed (caught): {e}",
            )


# Backwards-compatible module-level entrypoint.
# NOTE: this does NOT read environment variables; `run_evolve.py` should instantiate
# `FlashInferEvaluator` with an explicit `FlashInferEvaluatorConfig`.
_DEFAULT_EVALUATOR: Optional[FlashInferEvaluator] = None


def set_default_evaluator(cfg: FlashInferEvaluatorConfig) -> None:
    global _DEFAULT_EVALUATOR
    _DEFAULT_EVALUATOR = FlashInferEvaluator(cfg)


def evaluate(program_text: str):
    if _DEFAULT_EVALUATOR is None:
        print("FlashInferEvaluator not configured; returning default failure result.")
        return _evaluation_result(
            combined_score=0.0,
            latency_ms=None,
            num_passed=0,
            trace_log="Default evaluator not configured. Instantiate FlashInferEvaluator in run_evolve.py.",
        )
    return _DEFAULT_EVALUATOR.evaluate(program_text)
