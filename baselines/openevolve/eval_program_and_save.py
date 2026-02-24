from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Any, Dict, Optional


def _read_program_json(path: Path) -> Dict[str, Any]:
    obj = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(obj, dict):
        raise ValueError(f"Program JSON is not an object: {path}")
    return obj


def _format_float(x: object, *, digits: int = 3) -> str:
    if isinstance(x, (int, float)):
        return f"{float(x):.{digits}f}"
    return "-"


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


def _baseline_latency_by_workload(
    trace_set: Any,
    *,
    definition_name: str,
    baseline_solution: str,
    current_hw_key: Optional[str],
) -> Dict[str, float]:
    """Return {workload_uuid: baseline_latency_ms} for a baseline solution from dataset traces.

    - Only uses PASSED traces
    - If current_hw_key is provided, requires trace.evaluation.environment.hardware to match
    - Keeps the best (min) baseline latency per workload
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
            st = getattr(ev, "status", None)
            if str(st) not in ("EvaluationStatus.PASSED", "PASSED"):
                continue
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


def main(argv: Optional[list[str]] = None) -> int:
    ap = argparse.ArgumentParser(
        description="Export an OpenEvolve program JSON into a flashinfer-bench Solution JSON, then run a full benchmark."
    )
    ap.add_argument(
        "--program-json",
        required=True,
        help="Path to OpenEvolve programs/<uuid>.json (contains a 'code' field with CUDA XML blocks).",
    )
    ap.add_argument("--dataset-path", required=True, help="TraceSet root (e.g. <PATH_TO_FLASHINFER_TRACE_DATASET>)")
    ap.add_argument("--definition", required=True, help="Definition name (e.g. mla_paged_decode_h16_ckv512_kpe64_ps1)")
    ap.add_argument("--solution-name", default="", help="Solution name to write (default: oe_program_<program_id>)")
    ap.add_argument("--output-solution-json", default="", help="Where to write Solution JSON (default: ./oe_solution_<id>.json)")
    ap.add_argument("--baseline-solution", default="", help="Baseline solution name (for vs_base); optional")
    ap.add_argument("--target-gpu", default="H100", help="BuildSpec.target_hardware entry (default: H100)")
    ap.add_argument("--warmup-runs", type=int, default=10)
    ap.add_argument("--iterations", type=int, default=10)
    ap.add_argument("--num-trials", type=int, default=1)
    ap.add_argument("--rtol", type=float, default=1e-2)
    ap.add_argument("--atol", type=float, default=1e-2)
    ap.add_argument("--timeout-seconds", type=int, default=150)
    ap.add_argument("--use-isolated-runner", action="store_true", default=True)
    ap.add_argument("--no-use-isolated-runner", action="store_false", dest="use_isolated_runner")
    ap.add_argument("--parallel-workloads", action="store_true", default=True)
    ap.add_argument("--no-parallel-workloads", action="store_false", dest="parallel_workloads")
    ap.add_argument("--max-parallel-workloads", type=int, default=0)
    args = ap.parse_args(argv)

    program_path = Path(args.program_json).resolve()
    program = _read_program_json(program_path)
    program_id = str(program.get("id") or program_path.stem)
    code = str(program.get("code") or "")
    if not code.strip():
        raise ValueError(f"No 'code' found in {program_path}")

    from flashinfer_bench.data import BuildSpec, Solution, SourceFile, SupportedLanguages, TraceSet
    from flashinfer_bench.data.json_utils import save_json_file
    from flashinfer_bench.utils import hardware_from_device
    import torch  # type: ignore

    files = _parse_cuda_xml_files(code)
    if not files:
        raise ValueError("Could not find any <header_file>/<cuda_file>/<cpp_file> blocks in program code")

    sol_name = (args.solution_name or "").strip() or f"oe_program_{program_id.replace('-', '')[:8]}"
    solution = Solution(
        name=sol_name,
        definition=str(args.definition),
        author="openevolve",
        spec=BuildSpec(
            language=SupportedLanguages.CUDA,
            target_hardware=[str(args.target_gpu)],
            entry_point="main.cpp::run",
        ),
        sources=[SourceFile(path=fname, content=content) for fname, content in files.items()],
        description=f"Exported from OpenEvolve program {program_id}",
    )

    out_sol_path = Path(args.output_solution_json).resolve() if str(args.output_solution_json).strip() else (Path.cwd() / f"oe_solution_{program_id}.json")
    save_json_file(solution, out_sol_path)
    print(f"[export] Wrote Solution JSON: {out_sol_path}")

    # Full benchmark on all workloads for the definition.
    trace_set = TraceSet.from_path(str(args.dataset_path))
    if str(args.definition) not in trace_set.definitions:
        raise ValueError(f"Definition not found in dataset: {args.definition}")

    workloads = list(trace_set.workloads.get(str(args.definition), []) or [])
    if not workloads:
        raise ValueError(f"No workloads found for definition: {args.definition}")

    # Build a minimal traceset for this solution (avoid mutating the dataset).
    temp_traceset = TraceSet(
        root=trace_set.root,
        definitions={str(args.definition): trace_set.definitions[str(args.definition)]},
        solutions={str(args.definition): [solution]},
        workloads={str(args.definition): workloads},
        traces={str(args.definition): []},
    )

    from flashinfer_bench import Benchmark, BenchmarkConfig

    cfg = BenchmarkConfig(
        warmup_runs=int(args.warmup_runs),
        iterations=int(args.iterations),
        num_trials=int(args.num_trials),
        rtol=float(args.rtol),
        atol=float(args.atol),
        use_isolated_runner=bool(args.use_isolated_runner),
        timeout_seconds=int(args.timeout_seconds),
        parallel_workloads=bool(args.parallel_workloads),
        max_parallel_workloads=int(args.max_parallel_workloads),
        definitions=[str(args.definition)],
        solutions=[solution.name],
    )

    bench = Benchmark(temp_traceset, cfg)
    result = bench.run_all(dump_traces=False, resume=False)
    traces = list(result.traces.get(str(args.definition), []) or [])

    # Baseline map for vs_base (baseline_latency / candidate_latency).
    baseline_name = str(args.baseline_solution or "").strip()
    current_hw_key = None
    if torch.cuda.is_available():
        try:
            current_hw_key = hardware_from_device("cuda:0").lower()
        except Exception:
            current_hw_key = None

    bl_by_wl: Dict[str, float] = {}
    if baseline_name:
        bl_by_wl = _baseline_latency_by_workload(
            trace_set,
            definition_name=str(args.definition),
            baseline_solution=baseline_name,
            current_hw_key=current_hw_key,
        )

    # Print table (like evaluator / generate_kernels_and_eval).
    from flashinfer_bench.data import EvaluationStatus

    print(f"\n[{args.definition}] Full benchmark table for {solution.name}")
    print(
        "workload_uuid                      | status        | speedup(x) | latency(ms) | ref_latency(ms) | vs_base(x)"
    )
    print(
        "-----------------------------------+---------------+------------+------------+----------------+----------"
    )

    # Keep stable order from dataset workload list.
    wl_order = [w.workload.uuid for w in workloads if getattr(w, "workload", None)]
    by_wl: Dict[str, Any] = {}
    for t in traces:
        wl_uuid = getattr(getattr(t, "workload", None), "uuid", None)
        if not isinstance(wl_uuid, str) or not wl_uuid.strip():
            continue
        # keep "best" passed by max speedup (mirrors evaluator)
        prev = by_wl.get(wl_uuid)
        if prev is None:
            by_wl[wl_uuid] = t
            continue
        prev_ev = getattr(prev, "evaluation", None)
        ev = getattr(t, "evaluation", None)
        prev_pass = bool(prev_ev and prev_ev.status == EvaluationStatus.PASSED)
        cur_pass = bool(ev and ev.status == EvaluationStatus.PASSED)
        if cur_pass and not prev_pass:
            by_wl[wl_uuid] = t
        elif cur_pass and prev_pass:
            prev_sp = float(getattr(getattr(prev_ev, "performance", None), "speedup_factor", 0.0) or 0.0)
            cur_sp = float(getattr(getattr(ev, "performance", None), "speedup_factor", 0.0) or 0.0)
            if cur_sp > prev_sp:
                by_wl[wl_uuid] = t

    num_passed = 0
    vs_vals = []
    lat_vals = []

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
            num_passed += 1
            sp = getattr(perf, "speedup_factor", None)
            lat = getattr(perf, "latency_ms", None)
            rlat = getattr(perf, "reference_latency_ms", None)
            sp_s = _format_float(sp, digits=2)
            lat_s = _format_float(lat, digits=3)
            rlat_s = _format_float(rlat, digits=3)

            vsb = "-"
            if baseline_name and isinstance(lat, (int, float)) and float(lat) > 0:
                bl = bl_by_wl.get(str(wl_uuid))
                if isinstance(bl, (int, float)) and float(bl) > 0:
                    v = float(bl) / float(lat)
                    vs_vals.append(v)
                    vsb = f"{v:.2f}"
            if isinstance(lat, (int, float)):
                lat_vals.append(float(lat))

            print(f"{wl_uuid:<35} | {st:<13} | {sp_s:>10} | {lat_s:>10} | {rlat_s:>14} | {vsb:>8}")
        else:
            print(f"{wl_uuid:<35} | {st:<13} |     -      |     -     |       -        |     -")

    mean_lat = (sum(lat_vals) / len(lat_vals)) if lat_vals else None
    mean_vs = (sum(vs_vals) / len(vs_vals)) if vs_vals else None
    print("\n[aggregate]")
    print(f"workloads_total={len(wl_order)} workloads_passed={num_passed}")
    print(f"mean_latency_ms={mean_lat}" if mean_lat is not None else "mean_latency_ms=None")
    if baseline_name:
        print(f"baseline_solution={baseline_name} hw_key={current_hw_key}")
        print(f"mean_vs_base={mean_vs}" if mean_vs is not None else "mean_vs_base=None")
        if mean_vs is not None:
            print(f"combined_score(mean_vs_base*100)={mean_vs * 100.0}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())

