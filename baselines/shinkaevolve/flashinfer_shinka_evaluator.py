"""
FlashInfer-Bench evaluator utilities for ShinkaEvolve.

This module is the ShinkaEvolve analogue of:
  - `examples/openevolve/flashinfer_oe_evaluator.py`

Key differences vs OpenEvolve:
  - ShinkaEvolve evolves a Python program file (`program_path`) with an EVOLVE-BLOCK.
  - The program's `run_experiment(**kwargs)` can call `evaluate_candidate(...)` here
    to score an evolved kernel candidate.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
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


def _parse_cuda_xml_files(code: str) -> Dict[str, str]:
    """Parse KernelGenerator-style XML blocks into a {filename: content} dict."""
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


def _format_trace_log(trace) -> str:
    """Format a flashinfer-bench Trace evaluation into a human-readable log string."""
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
    eval_info = f"Status: {getattr(getattr(ev, 'status', None), 'value', None) or getattr(ev, 'status', None)}\n"
    eval_info += f"Timestamp: {getattr(ev, 'timestamp', None)}\n"
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


def _pick_failing_trace(traces: Sequence[Any]) -> Optional[Any]:
    """Pick one representative failing trace for feedback."""
    try:
        failing = [
            t
            for t in (traces or [])
            if getattr(t, "evaluation", None) is not None
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


def _baseline_latency_by_workload(
    trace_set: Any,
    *,
    definition_name: str,
    baseline_solution: str,
    current_hw_key: Optional[str],
) -> Dict[str, float]:
    """Return {workload_uuid: baseline_latency_ms} for a baseline solution from dataset traces."""
    out: Dict[str, float] = {}
    traces = list(getattr(trace_set, "traces", {}).get(definition_name, []) or [])
    for t in traces:
        try:
            if getattr(t, "solution", None) != baseline_solution:
                continue
            ev = getattr(t, "evaluation", None)
            if ev is None:
                continue
            st = getattr(ev, "status", None)
            if str(st) not in ("EvaluationStatus.PASSED", "PASSED"):
                continue
            if current_hw_key:
                hw = getattr(getattr(ev, "environment", None), "hardware", None)
                hw_key = hw.lower() if isinstance(hw, str) else None
                if hw_key != current_hw_key:
                    continue
            perf = getattr(ev, "performance", None)
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
    """Compute a strict aggregate: all workloads must PASS; score is mean speedup."""
    from flashinfer_bench.data import EvaluationStatus

    traces = traceset.traces.get(definition_name, [])
    total = 0
    passed = 0
    best_by_workload: Dict[str, Any] = {}
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
    if total == 0 or workloads_passed != total:
        return False, {
            "workloads_passed": int(workloads_passed),
            "mean_speedup": 0.0,
            "mean_latency_ms": None,
            "traces_total": int(total),
            "traces_passed": int(passed),
        }

    mean_speedup = sum(t.evaluation.performance.speedup_factor for t in best_by_workload.values()) / workloads_passed
    mean_latency = sum(t.evaluation.performance.latency_ms for t in best_by_workload.values()) / workloads_passed
    return True, {
        "workloads_passed": int(workloads_passed),
        "mean_speedup": float(mean_speedup),
        "mean_latency_ms": float(mean_latency),
        "traces_total": int(total),
        "traces_passed": int(passed),
    }


@dataclass(frozen=True)
class FlashInferShinkaEvaluatorConfig:
    dataset_path: str
    definition: str
    # Required: score is always computed as mean_vs_base * 100 (OpenEvolve-style).
    # This names a solution present in the dataset traces (e.g. a FlashInfer wrapper).
    baseline_solution: str
    language: str = "cuda"  # triton | cuda | python
    target_gpu: str = "H100"
    warmup_runs: int = 10
    iterations: int = 10
    num_trials: int = 1
    rtol: float = 1e-2
    atol: float = 1e-2
    use_isolated_runner: bool = True
    timeout_seconds: int = 150
    parallel_workloads: bool = True
    max_parallel_workloads: int = 0
    # Workload selection for feedback (fast evolution). If set, use exactly these workload UUIDs.
    feedback_workloads: Optional[List[str]] = None
    num_feedback_workloads: int = 5
    num_eval_workload: int = 0  # 0 = all
    verbose_table: bool = False

    @staticmethod
    def from_config(cfg: Dict[str, Any]) -> "FlashInferShinkaEvaluatorConfig":
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
        if not baseline_solution or not str(baseline_solution).strip():
            raise ValueError(
                "Missing required evaluator config key: baseline_solution "
                "(required for vs-base scoring; score = mean_vs_base * 100)."
            )
        verbose_table = bool(_get("verbose_table", _get("verbose", False)))

        timeout_seconds = int(_get("timeout_seconds", _get("timeout", 150)))
        parallel_workloads = bool(_get("parallel_workloads", True))
        max_parallel_workloads = int(_get("max_parallel_workloads", 0))

        return FlashInferShinkaEvaluatorConfig(
            dataset_path=str(dataset_path),
            definition=str(definition),
            language=str(_get("language", "cuda")),
            target_gpu=str(_get("target_gpu", _get("target_hardware", "H100"))),
            warmup_runs=int(_get("warmup_runs", 10)),
            iterations=int(_get("iterations", 10)),
            num_trials=int(_get("num_trials", 1)),
            rtol=float(_get("rtol", 1e-2)),
            atol=float(_get("atol", 1e-2)),
            use_isolated_runner=bool(_get("use_isolated_runner", True)),
            timeout_seconds=int(timeout_seconds),
            parallel_workloads=bool(parallel_workloads),
            max_parallel_workloads=int(max_parallel_workloads),
            feedback_workloads=feedback_workloads,
            num_feedback_workloads=int(_get("num_feedback_workloads", 5)),
            num_eval_workload=int(_get("num_eval_workload", 0)),
            baseline_solution=str(baseline_solution).strip(),
            verbose_table=bool(verbose_table),
        )


def _select_workloads(trace_set: Any, *, cfg: FlashInferShinkaEvaluatorConfig) -> Sequence[Any]:
    workloads = list(trace_set.workloads.get(cfg.definition, []) or [])
    if not workloads:
        return []

    if cfg.feedback_workloads:
        wanted = list(cfg.feedback_workloads)
        wanted_set = set(wanted)
        order = {w: i for i, w in enumerate(wanted)}
        selected = [wl for wl in workloads if wl.workload.uuid in wanted_set]
        selected.sort(key=lambda wl: order.get(wl.workload.uuid, 1 << 30))
        return selected

    k_req = int(cfg.num_feedback_workloads)
    if k_req <= 0:
        return workloads
    k = min(k_req, len(workloads))
    return random.sample(workloads, k=k) if len(workloads) > k else workloads


def _solution_from_candidate(
    *,
    trace_set: Any,
    cfg: FlashInferShinkaEvaluatorConfig,
    candidate: Dict[str, Any],
) -> Tuple[Any, str]:
    """
    Return (Solution, mode_str).

    Candidate contract:
      - baseline mode:
          {"mode": "baseline", "solution_name": "<existing_solution_name>"}
      - code mode:
          {"mode": "code", "language": "cuda"|"triton"|"python", "code": "<text or path>"}

    For CUDA, `code` can be KernelGenerator XML with kernel.h/kernel.cu/main.cpp blocks.
    """
    from flashinfer_bench.data import BuildSpec, Solution, SourceFile, SupportedLanguages

    mode = str(candidate.get("mode", "")).strip().lower()
    if mode == "baseline":
        sol_name = str(candidate.get("solution_name", "")).strip()
        if not sol_name:
            raise ValueError("baseline candidate requires non-empty solution_name")
        sol = trace_set.get_solution(sol_name)
        if sol is None:
            raise ValueError(f"baseline solution not found in dataset: {sol_name}")
        return sol, f"baseline:{sol_name}"

    if mode != "code":
        raise ValueError("candidate['mode'] must be 'baseline' or 'code'")

    lang = str(candidate.get("language", cfg.language)).strip().lower()
    raw_text = _maybe_read_text(str(candidate.get("code", "")))
    if not raw_text:
        raise ValueError("code candidate requires non-empty code")

    supported = {
        "python": SupportedLanguages.PYTHON,
        "triton": SupportedLanguages.TRITON,
        "cuda": SupportedLanguages.CUDA,
    }.get(lang)
    if supported is None:
        raise ValueError(f"Unsupported language: {lang}")

    author = "shinkaevolve"
    name = f"shinka_candidate_{abs(hash(raw_text)) % (10**10)}"

    if lang == "cuda" and ("<cuda_file" in raw_text or "<header_file" in raw_text or "<cpp_file" in raw_text):
        files = _parse_cuda_xml_files(raw_text)
        sources = [SourceFile(path=fname, content=content) for fname, content in files.items()]
        return (
            Solution(
                name=name,
                definition=cfg.definition,
                author=author,
                spec=BuildSpec(language=supported, target_hardware=[cfg.target_gpu], entry_point="main.cpp::run"),
                sources=sources,
                description="ShinkaEvolve candidate (cuda xml)",
            ),
            "cuda_xml",
        )

    # Single-file program.
    path = "main.py" if lang in ("python", "triton") else "kernel.cu"
    entry_point = "main.py::run" if path.endswith(".py") else "main.cpp::run"
    return (
        Solution(
            name=name,
            definition=cfg.definition,
            author=author,
            spec=BuildSpec(language=supported, target_hardware=[cfg.target_gpu], entry_point=entry_point),
            sources=[SourceFile(path=path, content=raw_text)],
            description="ShinkaEvolve candidate (single file)",
        ),
        "single_file",
    )


def evaluate_candidate(*, cfg: FlashInferShinkaEvaluatorConfig, candidate: Dict[str, Any]) -> Dict[str, Any]:
    """
    Evaluate a candidate kernel (baseline or code) on a subset of workloads.

    Returns a dict meant to be consumed by ShinkaEvolve aggregate/validate logic:
      - combined_score: float  (maximize)
      - mean_speedup: float
      - mean_vs_base: Optional[float]
      - mean_latency_ms: Optional[float]
      - num_passed: int
      - trace_log: str (human-readable feedback; safe to put into text_feedback)
    """
    from flashinfer_bench import Benchmark, BenchmarkConfig
    from flashinfer_bench.data import TraceSet

    trace_set = TraceSet.from_path(cfg.dataset_path)
    if cfg.definition not in trace_set.definitions:
        return {
            "combined_score": 0.0,
            "mean_speedup": 0.0,
            "mean_vs_base": None,
            "mean_latency_ms": None,
            "num_passed": 0,
            "trace_log": f"Definition '{cfg.definition}' not found in dataset.",
        }

    try:
        solution, mode = _solution_from_candidate(trace_set=trace_set, cfg=cfg, candidate=candidate)
    except Exception as e:
        return {
            "combined_score": 0.0,
            "mean_speedup": 0.0,
            "mean_vs_base": None,
            "mean_latency_ms": None,
            "num_passed": 0,
            "trace_log": f"Invalid candidate ({e}).",
        }

    selected_workloads = list(_select_workloads(trace_set, cfg=cfg))
    if (not selected_workloads) and cfg.num_eval_workload and cfg.num_eval_workload > 0:
        wl_list = trace_set.workloads.get(cfg.definition, [])
        if wl_list:
            selected_workloads = list(wl_list[: min(len(wl_list), cfg.num_eval_workload)])

    # IMPORTANT: evaluate on a fresh TraceSet (matches KernelGenerator behavior).
    temp_traceset = TraceSet(
        root=trace_set.root,
        definitions={cfg.definition: trace_set.definitions[cfg.definition]},
        solutions={cfg.definition: [solution]},
        workloads={cfg.definition: list(selected_workloads)},
        traces={cfg.definition: []},
    )

    bench_cfg = BenchmarkConfig(
        warmup_runs=int(cfg.warmup_runs),
        iterations=int(cfg.iterations),
        num_trials=int(cfg.num_trials),
        rtol=float(cfg.rtol),
        atol=float(cfg.atol),
        use_isolated_runner=bool(cfg.use_isolated_runner),
        timeout_seconds=int(cfg.timeout_seconds),
        parallel_workloads=bool(cfg.parallel_workloads),
        max_parallel_workloads=int(cfg.max_parallel_workloads),
        definitions=[cfg.definition],
        solutions=[solution.name],
    )

    bench = Benchmark(temp_traceset, bench_cfg)
    result_traceset = bench.run_all(dump_traces=False, resume=False)
    traces = list(result_traceset.traces.get(cfg.definition, []) or [])

    ok, metrics = _score_from_traces(result_traceset, cfg.definition)
    if not ok:
        failing_trace = _pick_failing_trace(traces)
        trace_log = _format_trace_log(failing_trace) if failing_trace is not None else f"Candidate failed. mode={mode}"
        return {
            "combined_score": 0.0,
            "mean_speedup": 0.0,
            "mean_vs_base": None,
            "mean_latency_ms": None,
            "num_passed": 0,
            "trace_log": trace_log,
        }

    mean_speedup = float(metrics["mean_speedup"])
    mean_latency_ms = metrics.get("mean_latency_ms", None)
    num_passed = int(metrics.get("workloads_passed", 0) or 0)

    # Optional: compute vs_base = baseline_latency / candidate_latency from dataset traces.
    baseline_name = str(cfg.baseline_solution or "").strip()
    mean_vs_base: Optional[float] = None
    # IMPORTANT: do not include baseline solution name in trace_log/text feedback,
    # to avoid leaking it into LLM prompts during evolution.
    vs_base_note = "mean_vs_base=None (no baseline configured)"

    if baseline_name:
        current_hw_key: Optional[str] = None
        try:
            from flashinfer_bench.utils import hardware_from_device
            import torch  # type: ignore

            dev = "cuda:0" if torch.cuda.is_available() else "cpu"
            current_hw_key = hardware_from_device(dev).lower()
        except Exception:
            current_hw_key = None

        bl_by_wl = _baseline_latency_by_workload(
            trace_set,
            definition_name=cfg.definition,
            baseline_solution=baseline_name,
            current_hw_key=current_hw_key,
        )

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

        vs_base_by_wl: Dict[str, float] = {}
        if cand_lat_by_wl and bl_by_wl:
            for wl_uuid, cand_lat in cand_lat_by_wl.items():
                bl_lat = bl_by_wl.get(wl_uuid)
                if isinstance(bl_lat, (int, float)) and float(bl_lat) > 0 and float(cand_lat) > 0:
                    vs_base_by_wl[wl_uuid] = float(bl_lat) / float(cand_lat)

        if vs_base_by_wl and len(vs_base_by_wl) == int(num_passed):
            mean_vs_base = sum(vs_base_by_wl.values()) / float(len(vs_base_by_wl))
            vs_base_note = f"mean_vs_base={mean_vs_base} (hw_key={current_hw_key})"
        else:
            mean_vs_base = None
            vs_base_note = (
                f"mean_vs_base=None (missing baseline traces for some workloads; hw_key={current_hw_key})"
            )

    # Score is always OpenEvolve-style vs-base percentage:
    #   combined_score = mean_vs_base * 100
    # If we cannot compute mean_vs_base (e.g., missing baseline traces for some workloads),
    # return 0.0 so evolution strongly disfavors the candidate.
    combined_score = (float(mean_vs_base) * 100.0) if mean_vs_base is not None and math.isfinite(mean_vs_base) else 0.0

    trace_log = "\n\n".join(_format_trace_log(t) for t in traces) if traces else ""
    trace_log = (trace_log + f"\n\n[aggregate]\n{vs_base_note}\nmean_speedup={mean_speedup}").strip()

    return {
        "combined_score": float(combined_score),
        "mean_speedup": float(mean_speedup),
        "mean_vs_base": (None if mean_vs_base is None else float(mean_vs_base)),
        "mean_latency_ms": (None if mean_latency_ms is None else float(mean_latency_ms)),
        "num_passed": int(num_passed),
        "trace_log": trace_log,
    }

