"""Benchmark one candidate vs a FlashInfer baseline solution (per-workload table).

Supports candidates in two formats:
- OpenEvolve program JSON: {"id": ..., "code": ...}
- flashinfer-bench Solution JSON (e.g. oe_best_*.json saved by run_evolve.py)

Example:
  python3 -u examples/openevolve/bench_single_program_vs_flashinfer.py \
    --evaluator-config examples/openevolve/flashinfer_evaluator_config_moe.yaml \
    --program-json openevolve_output/flashinfer_moe_fp8_ds_routing_topk8/programs/a6447285-1277-43a5-8ab3-40af841f459c.json \
    --baseline-solution-json data/min_flashinfer_trace/solutions/moe/moe_fp8_block_scale_ds_routing_topk8_ng8_kg4_e32_h7168_i2048/flashinfer_wrapper_9sdjf3.json

  python3 -u examples/openevolve/bench_single_program_vs_flashinfer.py \
    --evaluator-config examples/openevolve/flashinfer_evaluator_config_moe.yaml \
    --candidate-solution-json data/min_flashinfer_trace/solutions/moe/moe_fp8_block_scale_ds_routing_topk8_ng8_kg4_e32_h7168_i2048/oe_best_....json \
    --baseline-solution-json data/min_flashinfer_trace/solutions/moe/moe_fp8_block_scale_ds_routing_topk8_ng8_kg4_e32_h7168_i2048/flashinfer_wrapper_9sdjf3.json
"""

from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple


def _load_mapping(path: str) -> Dict[str, Any]:
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(path)
    if p.suffix not in (".yaml", ".yml"):
        raise ValueError("Unsupported evaluator config file type (use .yaml/.yml)")
    try:
        from omegaconf import OmegaConf  # type: ignore
    except Exception as e:  # pragma: no cover
        raise RuntimeError("OmegaConf is required. Install with: pip install omegaconf") from e
    cfg = OmegaConf.load(str(p))
    data = OmegaConf.to_container(cfg, resolve=True)
    if not isinstance(data, dict):
        raise ValueError("Evaluator config must be a mapping")
    return data  # type: ignore[return-value]


def _parse_cuda_xml_files(code: str) -> Dict[str, str]:
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


def _load_program_json(path: str) -> Tuple[str, str]:
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(path)
    data = json.loads(p.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise ValueError(f"Program JSON must be an object: {path}")
    pid = str(data.get("id", p.stem))
    code = data.get("code", "")
    if not isinstance(code, str) or not code.strip():
        raise ValueError("Program JSON missing non-empty 'code' field")
    return pid, code


def _build_candidate_solution(*, program_id: str, code: str, eval_cfg) -> Any:
    from flashinfer_bench.data import BuildSpec, Solution, SourceFile, SupportedLanguages

    lang = (getattr(eval_cfg, "language", None) or "triton").lower()
    target_gpu = getattr(eval_cfg, "target_gpu", None) or "H100"
    supported = {
        "python": SupportedLanguages.PYTHON,
        "triton": SupportedLanguages.TRITON,
        "cuda": SupportedLanguages.CUDA,
    }.get(lang, SupportedLanguages.TRITON)

    name = f"oe_program_{program_id}"
    author = "openevolve"

    if lang == "cuda":
        files = _parse_cuda_xml_files(code)
        if not files:
            raise ValueError("CUDA program text missing expected XML blocks (kernel.h/kernel.cu/main.cpp).")
        sources = [SourceFile(path=fname, content=content) for fname, content in files.items()]
        return Solution(
            name=name,
            definition=eval_cfg.definition,
            author=author,
            spec=BuildSpec(language=supported, target_hardware=[target_gpu], entry_point="main.cpp::run"),
            sources=sources,
            description="OpenEvolve program (cuda xml)",
        )

    # python/triton: single-file
    return Solution(
        name=name,
        definition=eval_cfg.definition,
        author=author,
        spec=BuildSpec(language=supported, target_hardware=[target_gpu], entry_point="main.py::run"),
        sources=[SourceFile(path="main.py", content=code)],
        description="OpenEvolve program (single file)",
    )


def _select_workloads(trace_set, definition: str, uuids: Optional[Sequence[str]]) -> List[Any]:
    workloads_wrapped = list(trace_set.workloads.get(definition, []) or [])
    if not workloads_wrapped:
        return []
    if not uuids:
        return workloads_wrapped
    wanted = set(uuids)
    order = {u: i for i, u in enumerate(uuids)}
    selected = [w for w in workloads_wrapped if w.workload.uuid in wanted]
    selected.sort(key=lambda w: order.get(w.workload.uuid, 1 << 30))
    return selected


def _pick_least_used_cuda_device() -> Optional[int]:
    """
    Best-effort pick a CUDA device index with the lowest reported memory usage.

    We avoid importing torch here so this can run before CUDA visibility is frozen.
    """
    try:
        out = subprocess.check_output(
            [
                "nvidia-smi",
                "--query-gpu=memory.used",
                "--format=csv,nounits,noheader",
            ],
            stderr=subprocess.DEVNULL,
            text=True,
        )
        used = []
        for line in (out or "").strip().splitlines():
            line = line.strip()
            if not line:
                continue
            try:
                used.append(int(line))
            except Exception:
                used.append(1 << 60)
        if not used:
            return None
        return int(min(range(len(used)), key=lambda i: used[i]))
    except Exception:
        return None


def _maybe_pin_cuda_visibility(*, cuda_visible_devices: Optional[str], cuda_device: str) -> Optional[str]:
    """
    Set CUDA_VISIBLE_DEVICES early to reduce cross-job interference.

    Policy:
    - If --cuda-visible-devices is provided, use it verbatim.
    - Else if --cuda-device=keep, do not modify CUDA_VISIBLE_DEVICES.
    - Else if --cuda-device=<int>, pin to that single GPU index.
    - Else if --cuda-device is a comma-separated list (e.g. "0,1,2"), treat it as CUDA_VISIBLE_DEVICES.
    - Else (default: auto), pick the least-used GPU (by nvidia-smi memory.used) and pin to it.
    """
    if cuda_visible_devices is not None:
        os.environ["CUDA_VISIBLE_DEVICES"] = cuda_visible_devices
        return cuda_visible_devices

    mode = (cuda_device or "auto").strip().lower()
    if mode == "keep":
        return None

    # Allow comma-separated list as a shorthand for CUDA_VISIBLE_DEVICES.
    # NOTE: This only controls visibility; the underlying benchmark may still choose how to schedule work.
    if "," in mode:
        parts = [p.strip() for p in mode.split(",") if p.strip()]
        try:
            idxs = [int(p) for p in parts]
        except Exception as e:
            raise SystemExit(
                f"--cuda-device comma list must be integers like '0,1,2'; got: {cuda_device!r}"
            ) from e
        if any(i < 0 for i in idxs):
            raise SystemExit(f"--cuda-device comma list must be >= 0; got: {cuda_device!r}")
        os.environ["CUDA_VISIBLE_DEVICES"] = ",".join(str(i) for i in idxs)
        return os.environ["CUDA_VISIBLE_DEVICES"]

    if mode == "auto":
        # If the caller already constrained visibility, respect it.
        existing = os.environ.get("CUDA_VISIBLE_DEVICES", "").strip()
        if existing:
            return None
        pick = _pick_least_used_cuda_device()
        if pick is None:
            # Can't determine; leave as-is.
            return None
        os.environ["CUDA_VISIBLE_DEVICES"] = str(pick)
        return str(pick)

    try:
        idx = int(mode)
    except Exception as e:
        raise SystemExit(f"--cuda-device must be 'auto', 'keep', or an integer GPU index; got: {cuda_device!r}") from e
    if idx < 0:
        raise SystemExit(f"--cuda-device must be >= 0; got: {idx}")
    os.environ["CUDA_VISIBLE_DEVICES"] = str(idx)
    return str(idx)


def main(argv: Optional[list[str]] = None) -> int:
    ap = argparse.ArgumentParser(description="Benchmark one OpenEvolve program JSON vs FlashInfer baseline")
    ap.add_argument("--evaluator-config", type=str, required=True, help="Evaluator YAML (dataset_path/definition/knobs).")
    ap.add_argument("--baseline-solution-json", type=str, required=True, help="Baseline Solution JSON path (flashinfer wrapper).")

    cand_group = ap.add_mutually_exclusive_group(required=True)
    cand_group.add_argument("--program-json", type=str, default="", help="OpenEvolve program JSON with 'id' and 'code'.")
    cand_group.add_argument(
        "--candidate-solution-json",
        type=str,
        default="",
        help="Candidate Solution JSON path (flashinfer-bench Solution), e.g. oe_best_*.json.",
    )

    ap.add_argument("--all-workloads", action="store_true", help="Run all workloads (default).")
    ap.add_argument("--workload-uuids", type=str, nargs="*", default=None, help="Explicit workload UUIDs to run.")
    ap.add_argument("--num-workloads", type=int, default=0, help="If >0, run only first N selected workloads.")
    ap.add_argument("--bench-timeout-seconds", type=int, default=0, help="Override per-workload timeout in seconds.")
    ap.add_argument(
        "--cuda-device",
        type=str,
        default="auto",
        help="CUDA device selection: 'auto' (default, pick least-used GPU), 'keep' (do not set CUDA_VISIBLE_DEVICES), an integer GPU index (e.g. '0'), or a comma list (e.g. '0,1,2,3').",
    )
    ap.add_argument(
        "--cuda-visible-devices",
        type=str,
        default=None,
        help="Explicit CUDA_VISIBLE_DEVICES value (overrides --cuda-device), e.g. '0' or '0,1,2,3'.",
    )

    args = ap.parse_args(argv)

    pinned = _maybe_pin_cuda_visibility(
        cuda_visible_devices=args.cuda_visible_devices,
        cuda_device=args.cuda_device,
    )
    if pinned is not None:
        print(f"[bench] CUDA_VISIBLE_DEVICES={pinned}")

    base = _load_mapping(args.evaluator_config)
    from flashinfer_oe_evaluator import FlashInferEvaluatorConfig

    eval_cfg = FlashInferEvaluatorConfig.from_config(base)
    if args.bench_timeout_seconds and args.bench_timeout_seconds > 0:
        eval_cfg = eval_cfg.__class__(**{**eval_cfg.__dict__, "timeout_seconds": int(args.bench_timeout_seconds)})  # type: ignore

    from flashinfer_bench import Benchmark, BenchmarkConfig
    from flashinfer_bench.data import TraceSet, load_json_file, Solution, EvaluationStatus
    from flashinfer_bench.utils import hardware_from_device
    import torch  # type: ignore

    baseline_sol: Solution = load_json_file(Solution, Path(args.baseline_solution_json))
    if baseline_sol.definition != eval_cfg.definition:
        raise SystemExit(f"Baseline solution definition mismatch: {baseline_sol.definition} != {eval_cfg.definition}")

    # Candidate solution
    program_id: str
    cand_sol: Solution
    if args.candidate_solution_json:
        cand_sol = load_json_file(Solution, Path(args.candidate_solution_json))
        program_id = cand_sol.name
        if cand_sol.definition != eval_cfg.definition:
            raise SystemExit(f"Candidate solution definition mismatch: {cand_sol.definition} != {eval_cfg.definition}")
    else:
        program_id, program_code = _load_program_json(args.program_json)
        cand_sol = _build_candidate_solution(program_id=program_id, code=program_code, eval_cfg=eval_cfg)
        if cand_sol.definition != eval_cfg.definition:
            raise SystemExit(f"Candidate solution definition mismatch: {cand_sol.definition} != {eval_cfg.definition}")

    trace_set = TraceSet.from_path(eval_cfg.dataset_path)
    if eval_cfg.definition not in trace_set.definitions:
        raise SystemExit(f"Definition not found in dataset: {eval_cfg.definition}")

    # Workloads
    if args.workload_uuids is not None:
        selected_uuids = args.workload_uuids
    else:
        selected_uuids = None  # default all
    workloads_wrapped = _select_workloads(trace_set, eval_cfg.definition, selected_uuids)
    if args.num_workloads and args.num_workloads > 0:
        workloads_wrapped = workloads_wrapped[: args.num_workloads]
    if not workloads_wrapped:
        raise SystemExit("No workloads selected.")

    # Minimal traceset for benchmarking
    temp_traceset = TraceSet(
        root=trace_set.root,
        definitions={eval_cfg.definition: trace_set.definitions[eval_cfg.definition]},
        solutions={eval_cfg.definition: [baseline_sol, cand_sol]},
        workloads={eval_cfg.definition: workloads_wrapped},
        traces={eval_cfg.definition: []},
    )

    bench_cfg = BenchmarkConfig(
        warmup_runs=eval_cfg.warmup_runs,
        iterations=eval_cfg.iterations,
        num_trials=eval_cfg.num_trials,
        rtol=eval_cfg.rtol,
        atol=eval_cfg.atol,
        use_isolated_runner=eval_cfg.use_isolated_runner,
        timeout_seconds=int(getattr(eval_cfg, "timeout_seconds", 300) or 300),
        parallel_workloads=bool(getattr(eval_cfg, "parallel_workloads", False)),
        max_parallel_workloads=int(getattr(eval_cfg, "max_parallel_workloads", 0) or 0),
        definitions=[eval_cfg.definition],
        solutions=[baseline_sol.name, cand_sol.name],
    )

    bench = Benchmark(temp_traceset, bench_cfg)
    result_traceset = bench.run_all(dump_traces=False, resume=False)
    traces = list(result_traceset.traces.get(eval_cfg.definition, []) or [])

    # Index traces by (solution, workload_uuid); keep best PASSED by max speedup
    by_sol_wl: Dict[Tuple[str, str], Any] = {}
    for t in traces:
        ev = getattr(t, "evaluation", None)
        wl_uuid = getattr(getattr(t, "workload", None), "uuid", None)
        sol_name = getattr(t, "solution", None)
        if not isinstance(wl_uuid, str) or not isinstance(sol_name, str):
            continue
        key = (sol_name, wl_uuid)
        prev = by_sol_wl.get(key)
        if prev is None:
            by_sol_wl[key] = t
            continue
        prev_ev = getattr(prev, "evaluation", None)
        prev_pass = bool(prev_ev and prev_ev.status == EvaluationStatus.PASSED)
        cur_pass = bool(ev and ev.status == EvaluationStatus.PASSED)
        if cur_pass and not prev_pass:
            by_sol_wl[key] = t
            continue
        if cur_pass and prev_pass:
            prev_sp = float(getattr(getattr(prev_ev, "performance", None), "speedup_factor", 0.0) or 0.0)
            cur_sp = float(getattr(getattr(ev, "performance", None), "speedup_factor", 0.0) or 0.0)
            if cur_sp > prev_sp:
                by_sol_wl[key] = t

    # Hardware key for baseline matching (informational)
    try:
        dev = "cuda:0" if torch.cuda.is_available() else "cpu"
        hw_key = hardware_from_device(dev).lower()
    except Exception:
        hw_key = None

    def_name = eval_cfg.definition
    print(f"[{def_name}] Full benchmark table for {cand_sol.name}")
    print("workload_uuid                      | status        | speedup(x) | latency(ms) | ref_latency(ms) | vs_base(x)")
    print("-----------------------------------+---------------+------------+------------+----------------+----------")

    cand_latencies: List[float] = []
    vs_base_vals: List[float] = []
    total = 0
    passed = 0

    for wl_wrap in workloads_wrapped:
        wl_uuid = wl_wrap.workload.uuid
        total += 1

        cand_t = by_sol_wl.get((cand_sol.name, wl_uuid))
        base_t = by_sol_wl.get((baseline_sol.name, wl_uuid))

        cand_ev = getattr(cand_t, "evaluation", None) if cand_t is not None else None
        base_ev = getattr(base_t, "evaluation", None) if base_t is not None else None

        status = "N/A"
        sp_s = "-"
        lat_s = "-"
        ref_s = "-"
        vsb_s = "-"

        if cand_ev is not None:
            st_val = getattr(cand_ev.status, "value", None)
            status = str(st_val) if isinstance(st_val, str) and st_val else str(cand_ev.status)

            perf = getattr(cand_ev, "performance", None)
            if perf is not None and cand_ev.status == EvaluationStatus.PASSED:
                passed += 1
                sp = getattr(perf, "speedup_factor", None)
                lat = getattr(perf, "latency_ms", None)
                ref = getattr(perf, "reference_latency_ms", None)
                if isinstance(sp, (int, float)):
                    sp_s = f"{float(sp):.2f}"
                if isinstance(lat, (int, float)):
                    cand_latencies.append(float(lat))
                    lat_s = f"{float(lat):.3f}"
                if isinstance(ref, (int, float)):
                    ref_s = f"{float(ref):.3f}"

        # vs_base = baseline_latency / candidate_latency (PASSED only)
        try:
            if (
                cand_ev is not None
                and base_ev is not None
                and cand_ev.status == EvaluationStatus.PASSED
                and base_ev.status == EvaluationStatus.PASSED
            ):
                cand_lat = float(getattr(getattr(cand_ev, "performance", None), "latency_ms", 0.0) or 0.0)
                base_lat = float(getattr(getattr(base_ev, "performance", None), "latency_ms", 0.0) or 0.0)
                if cand_lat > 0.0 and base_lat > 0.0:
                    vs = base_lat / cand_lat
                    vs_base_vals.append(float(vs))
                    vsb_s = f"{float(vs):.2f}"
        except Exception:
            pass

        print(
            f"{wl_uuid:<35} | {status:<13} | {sp_s:>10} | {lat_s:>10} | {ref_s:>14} | {vsb_s:>8}"
        )

    # Aggregate
    mean_lat = (sum(cand_latencies) / len(cand_latencies)) if cand_latencies else None
    mean_vs = (sum(vs_base_vals) / len(vs_base_vals)) if vs_base_vals else None

    print("\n[aggregate]")
    print(f"workloads_total={total} workloads_passed={passed}")
    print(f"mean_latency_ms={mean_lat}")
    print(f"baseline_solution={baseline_sol.name} hw_key={hw_key}")
    print(f"mean_vs_base={mean_vs}")
    if mean_vs is not None:
        print(f"combined_score(mean_vs_base*100)={mean_vs * 100.0}")
    else:
        print("combined_score(mean_vs_base*100)=None")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

