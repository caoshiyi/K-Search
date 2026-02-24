"""
Benchmark one ShinkaEvolve "best" program vs a FlashInfer baseline (per-workload table).

This is the ShinkaEvolve analogue of:
  - examples/openevolve/bench_single_program_vs_flashinfer.py

Usage example:
  python3 -u examples/shinkaevolve/bench_best_solution_vs_flashinfer.py \
    --evaluator-config examples/shinkaevolve/flashinfer_evaluator_config_mla_prefill.yaml \
    --program-path results_20260120_080313/best/main.py
"""

from __future__ import annotations

import argparse
import contextlib
import fcntl
import importlib.util
import os
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Dict, IO, List, Optional, Sequence, Tuple


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


def _pick_least_used_cuda_device() -> Optional[int]:
    """Best-effort pick a CUDA device index with the lowest reported memory usage."""
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
        used: List[int] = []
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


def _nvidia_smi_gpu_indices() -> List[int]:
    """Return physical GPU indices from nvidia-smi (best-effort)."""
    try:
        out = subprocess.check_output(
            ["nvidia-smi", "--query-gpu=index", "--format=csv,nounits,noheader"],
            stderr=subprocess.DEVNULL,
            text=True,
        )
        idxs: List[int] = []
        for line in (out or "").strip().splitlines():
            line = line.strip()
            if not line:
                continue
            try:
                idxs.append(int(line))
            except Exception:
                continue
        return sorted(set(idxs))
    except Exception:
        return []


def _baseline_latency_by_workload(
    trace_set: Any,
    *,
    definition_name: str,
    baseline_solution: str,
    current_hw_key: Optional[str],
) -> Dict[str, float]:
    """Return {workload_uuid: baseline_latency_ms} for a baseline solution from dataset traces.

    Matches OpenEvolve evaluator behavior:
    - only use PASSED traces
    - if current_hw_key is available, require trace.evaluation.environment.hardware to match
    - keep the best (min) baseline latency per workload
    """
    from flashinfer_bench.data import EvaluationStatus

    out: Dict[str, float] = {}
    traces = list(getattr(trace_set, "traces", {}).get(definition_name, []) or [])
    for t in traces:
        try:
            if getattr(t, "solution", None) != baseline_solution:
                continue
            ev = getattr(t, "evaluation", None)
            if ev is None or getattr(ev, "status", None) != EvaluationStatus.PASSED:
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
            if not isinstance(wl_uuid, str) or not wl_uuid.strip():
                continue
            if not isinstance(lat, (int, float)) or float(lat) <= 0:
                continue
            prev = out.get(str(wl_uuid))
            out[str(wl_uuid)] = float(lat) if prev is None else min(float(prev), float(lat))
        except Exception:
            continue
    return out


@contextlib.contextmanager
def _reserve_gpus(*, gpu_indices: Sequence[int], lock_dir: Path, timeout_s: float) -> Sequence[int]:
    """
    Cross-process GPU reservation using flock() locks.

    This prevents *multiple benchmark jobs* from concurrently using the same physical GPU(s).
    """
    lock_dir.mkdir(parents=True, exist_ok=True)
    files: List[IO[str]] = []
    locked: List[int] = []
    start = time.time()
    while True:
        # Try to acquire all locks. If any fail, release and retry.
        for i in gpu_indices:
            p = lock_dir / f"gpu_{int(i)}.lock"
            f = open(p, "a+", encoding="utf-8")
            try:
                fcntl.flock(f.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
            except BlockingIOError:
                # Someone else holds it.
                try:
                    f.close()
                except Exception:
                    pass
                # Release any locks we already got in this attempt.
                for ff in files:
                    try:
                        fcntl.flock(ff.fileno(), fcntl.LOCK_UN)
                    except Exception:
                        pass
                    try:
                        ff.close()
                    except Exception:
                        pass
                files = []
                locked = []
                break
            else:
                files.append(f)
                locked.append(int(i))
        else:
            # All acquired.
            break

        if timeout_s <= 0:
            raise SystemExit(f"Timed out acquiring GPU locks in {lock_dir} for GPUs={list(gpu_indices)}")
        if time.time() - start > timeout_s:
            raise SystemExit(f"Timed out acquiring GPU locks in {lock_dir} for GPUs={list(gpu_indices)}")
        time.sleep(0.5)

    try:
        yield locked
    finally:
        for f in files:
            try:
                fcntl.flock(f.fileno(), fcntl.LOCK_UN)
            except Exception:
                pass
            try:
                f.close()
            except Exception:
                pass


def _maybe_pin_cuda_visibility(*, cuda_visible_devices: Optional[str], cuda_device: str) -> Optional[str]:
    """
    Set CUDA_VISIBLE_DEVICES early to reduce cross-job interference.

    Policy:
    - If --cuda-visible-devices is provided, use it verbatim.
    - Else if --cuda-device=keep, do not modify CUDA_VISIBLE_DEVICES.
    - Else if --cuda-device=<int>, pin to that single GPU index.
    - Else if --cuda-device is a comma-separated list (e.g. "0,1,2"), treat it as CUDA_VISIBLE_DEVICES.
    - Else (default: auto), pick the least-used GPU and pin to it.
    """
    if cuda_visible_devices is not None:
        os.environ["CUDA_VISIBLE_DEVICES"] = cuda_visible_devices
        return cuda_visible_devices

    mode = (cuda_device or "auto").strip().lower()
    if mode == "keep":
        return None

    if "," in mode:
        parts = [p.strip() for p in mode.split(",") if p.strip()]
        try:
            idxs = [int(p) for p in parts]
        except Exception as e:
            raise SystemExit(f"--cuda-device comma list must be integers like '0,1,2'; got: {cuda_device!r}") from e
        if any(i < 0 for i in idxs):
            raise SystemExit(f"--cuda-device comma list must be >= 0; got: {cuda_device!r}")
        os.environ["CUDA_VISIBLE_DEVICES"] = ",".join(str(i) for i in idxs)
        return os.environ["CUDA_VISIBLE_DEVICES"]

    if mode == "auto":
        existing = os.environ.get("CUDA_VISIBLE_DEVICES", "").strip()
        if existing:
            return None
        pick = _pick_least_used_cuda_device()
        if pick is None:
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


def _load_program_module(program_path: Path):
    spec = importlib.util.spec_from_file_location("shinka_best_program", str(program_path))
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Could not import program module from: {program_path}")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)  # type: ignore[attr-defined]
    return mod


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


def _solution_from_candidate(*, trace_set, eval_cfg, candidate: Dict[str, Any]):
    """
    Build a flashinfer-bench Solution from a ShinkaEvolve candidate dict.

    Uses the same candidate contract as examples/shinkaevolve/flashinfer_initial.py.
    """
    from flashinfer_bench.data import BuildSpec, Solution, SourceFile, SupportedLanguages
    from examples.shinkaevolve.flashinfer_shinka_evaluator import _parse_cuda_xml_files  # type: ignore

    mode = str(candidate.get("mode", "")).strip().lower()
    if mode == "baseline":
        sol_name = str(candidate.get("solution_name", "")).strip()
        if not sol_name:
            raise ValueError("baseline candidate requires non-empty solution_name")
        sol = trace_set.get_solution(sol_name)
        if sol is None:
            raise ValueError(f"baseline solution not found in dataset: {sol_name}")
        return sol

    if mode != "code":
        raise ValueError("candidate['mode'] must be 'baseline' or 'code'")

    lang = str(candidate.get("language", getattr(eval_cfg, "language", "cuda"))).strip().lower()
    raw_text = str(candidate.get("code", "")).strip()
    if not raw_text:
        raise ValueError("code candidate requires non-empty code")

    supported = {
        "python": SupportedLanguages.PYTHON,
        "triton": SupportedLanguages.TRITON,
        "cuda": SupportedLanguages.CUDA,
    }.get(lang)
    if supported is None:
        raise ValueError(f"Unsupported language: {lang}")

    name = f"shinka_best_candidate_{abs(hash(raw_text)) % (10**10)}"
    target_gpu = getattr(eval_cfg, "target_gpu", None) or "H100"

    if lang == "cuda" and ("<cuda_file" in raw_text or "<header_file" in raw_text or "<cpp_file" in raw_text):
        files = _parse_cuda_xml_files(raw_text)
        if not files:
            raise ValueError("CUDA program text missing expected XML blocks (kernel.h/kernel.cu/main.cpp).")
        sources = [SourceFile(path=fname, content=content) for fname, content in files.items()]
        return Solution(
            name=name,
            definition=eval_cfg.definition,
            author="shinkaevolve",
            spec=BuildSpec(language=supported, target_hardware=[target_gpu], entry_point="main.cpp::run"),
            sources=sources,
            description="ShinkaEvolve best candidate (cuda xml)",
        )

    # Single-file program.
    path = "main.py" if lang in ("python", "triton") else "kernel.cu"
    entry_point = "main.py::run" if path.endswith(".py") else "main.cpp::run"
    return Solution(
        name=name,
        definition=eval_cfg.definition,
        author="shinkaevolve",
        spec=BuildSpec(language=supported, target_hardware=[target_gpu], entry_point=entry_point),
        sources=[SourceFile(path=path, content=raw_text)],
        description="ShinkaEvolve best candidate (single file)",
    )


def main(argv: Optional[List[str]] = None) -> int:
    # Ensure repo root importable (matches evaluate_flashinfer.py behavior).
    repo_root = Path(__file__).resolve().parents[2]
    if str(repo_root) not in sys.path:
        sys.path.insert(0, str(repo_root))

    ap = argparse.ArgumentParser(description="Benchmark ShinkaEvolve best program vs FlashInfer baseline (per-workload).")
    ap.add_argument("--evaluator-config", type=str, required=True, help="Evaluator YAML (dataset_path/definition/knobs).")
    ap.add_argument("--program-path", type=str, required=True, help="Path to results_*/best/main.py")
    ap.add_argument(
        "--baseline-solution",
        type=str,
        default="",
        help="Override baseline solution name (defaults to evaluator-config baseline_solution if present).",
    )
    ap.add_argument("--workload-uuids", type=str, nargs="*", default=None, help="Explicit workload UUIDs to run.")
    ap.add_argument("--num-workloads", type=int, default=0, help="If >0, run only first N selected workloads.")
    ap.add_argument(
        "--reserve-gpus",
        type=str,
        default="all",
        help=(
            "GPU reservation mode to avoid cross-job contention: "
            "'all' (default, reserve ALL physical GPUs and use them), "
            "'visible' (reserve current CUDA_VISIBLE_DEVICES list), "
            "'none' (no reservation)."
        ),
    )
    ap.add_argument(
        "--gpu-lock-dir",
        type=str,
        default="/tmp/flashinfer_gpu_locks",
        help="Directory used for cross-process GPU lock files.",
    )
    ap.add_argument(
        "--gpu-lock-timeout-s",
        type=float,
        default=0.0,
        help="Seconds to wait for GPU locks (0 = wait forever).",
    )
    ap.add_argument(
        "--cuda-device",
        type=str,
        default="auto",
        help="CUDA device selection: 'auto' (default), 'keep', integer index, or comma list.",
    )
    ap.add_argument(
        "--cuda-visible-devices",
        type=str,
        default=None,
        help="Explicit CUDA_VISIBLE_DEVICES value (overrides --cuda-device), e.g. '0' or '0,1,2,3'.",
    )
    args = ap.parse_args(argv)

    reserve_mode = (args.reserve_gpus or "all").strip().lower()
    lock_dir = Path(str(args.gpu_lock_dir)).expanduser().resolve()

    # Decide which GPUs to reserve (physical indices).
    if reserve_mode == "none":
        # User opts out. Still allow pinning.
        pinned = _maybe_pin_cuda_visibility(cuda_visible_devices=args.cuda_visible_devices, cuda_device=args.cuda_device)
        if pinned is not None:
            print(f"[bench] CUDA_VISIBLE_DEVICES={pinned}")
        reserved_ctx = contextlib.nullcontext([])  # type: ignore[var-annotated]
    else:
        if reserve_mode == "visible":
            # Reserve exactly what will be visible after pinning logic.
            pinned = _maybe_pin_cuda_visibility(cuda_visible_devices=args.cuda_visible_devices, cuda_device=args.cuda_device)
            vis = os.environ.get("CUDA_VISIBLE_DEVICES", "").strip()
            if pinned is not None:
                print(f"[bench] CUDA_VISIBLE_DEVICES={pinned}")
            if not vis:
                # If not set, "visible" degenerates to all.
                gpu_idxs = _nvidia_smi_gpu_indices()
            else:
                try:
                    gpu_idxs = [int(x.strip()) for x in vis.split(",") if x.strip()]
                except Exception:
                    raise SystemExit(f"Invalid CUDA_VISIBLE_DEVICES={vis!r}; expected comma-separated ints.")
        elif reserve_mode == "all":
            gpu_idxs = _nvidia_smi_gpu_indices()
        else:
            raise SystemExit(f"--reserve-gpus must be one of: all|visible|none; got: {args.reserve_gpus!r}")

        if not gpu_idxs:
            raise SystemExit("Could not determine GPU indices to reserve (nvidia-smi unavailable?). Use --reserve-gpus none.")

        reserved_ctx = _reserve_gpus(gpu_indices=gpu_idxs, lock_dir=lock_dir, timeout_s=float(args.gpu_lock_timeout_s))

    with reserved_ctx as reserved:
        # If we reserved GPUs, force CUDA_VISIBLE_DEVICES to exactly that set (so this process uses only reserved GPUs).
        if reserved:
            os.environ["CUDA_VISIBLE_DEVICES"] = ",".join(str(i) for i in reserved)
            print(f"[bench] Reserved GPUs={list(reserved)} via locks in {lock_dir}")
            print(f"[bench] CUDA_VISIBLE_DEVICES={os.environ['CUDA_VISIBLE_DEVICES']}")

        base = _load_mapping(args.evaluator_config)
        from examples.shinkaevolve.flashinfer_shinka_evaluator import FlashInferShinkaEvaluatorConfig

        eval_cfg = FlashInferShinkaEvaluatorConfig.from_config(base)
        # Force "all workloads" semantics and per-workload table.
        eval_cfg = eval_cfg.__class__(
            **{
                **eval_cfg.__dict__,
                "feedback_workloads": None,
                "num_feedback_workloads": 0,
                "num_eval_workload": 0,
                "verbose_table": True,
            }
        )  # type: ignore

        program_path = Path(args.program_path).expanduser().resolve()
        mod = _load_program_module(program_path)
        cand_fn = getattr(mod, "candidate_kernel", None)
        if not callable(cand_fn):
            raise SystemExit(f"Program missing callable candidate_kernel(...): {program_path}")

        # Dataset / baseline / workloads
        from flashinfer_bench.data import TraceSet, EvaluationStatus
        from flashinfer_bench import Benchmark, BenchmarkConfig

        trace_set = TraceSet.from_path(eval_cfg.dataset_path)
        if eval_cfg.definition not in trace_set.definitions:
            raise SystemExit(f"Definition not found in dataset: {eval_cfg.definition}")

        baseline_name = (str(args.baseline_solution).strip() if args.baseline_solution else "") or (
            str(getattr(eval_cfg, "baseline_solution", "") or "").strip()
        )
        if not baseline_name:
            raise SystemExit("No baseline solution provided. Set evaluator-config baseline_solution or pass --baseline-solution.")

        baseline_sol = trace_set.get_solution(baseline_name)
        if baseline_sol is None:
            raise SystemExit(f"Baseline solution not found in dataset: {baseline_name}")

        # Match OpenEvolve: baseline latencies come from dataset traces (PASSED, hw-matched, min per workload).
        current_hw_key: Optional[str] = None
        try:
            from flashinfer_bench.utils import hardware_from_device
            import torch  # type: ignore

            dev = "cuda:0" if torch.cuda.is_available() else "cpu"
            current_hw_key = hardware_from_device(dev).lower()
        except Exception:
            current_hw_key = None
        baseline_lat_by_wl = _baseline_latency_by_workload(
            trace_set,
            definition_name=eval_cfg.definition,
            baseline_solution=baseline_sol.name,
            current_hw_key=current_hw_key,
        )

        # Build candidate from the best program's candidate_kernel().
        # Newer programs don't accept baseline_solution (baseline is only used for vs-base scoring).
        try:
            candidate = cand_fn()
        except TypeError:
            # Backwards-compat: older programs may still accept baseline_solution.
            candidate = cand_fn(baseline_solution=baseline_name)
        if not isinstance(candidate, dict):
            raise SystemExit(f"candidate_kernel() must return a dict; got: {type(candidate).__name__}")
        cand_sol = _solution_from_candidate(trace_set=trace_set, eval_cfg=eval_cfg, candidate=candidate)

        workloads_wrapped = _select_workloads(trace_set, eval_cfg.definition, args.workload_uuids)
        if args.num_workloads and args.num_workloads > 0:
            workloads_wrapped = workloads_wrapped[: args.num_workloads]
        if not workloads_wrapped:
            raise SystemExit("No workloads selected.")

        # Benchmark ONLY the candidate; baseline comes from dataset traces (OpenEvolve-style).
        temp_traceset = TraceSet(
            root=trace_set.root,
            definitions={eval_cfg.definition: trace_set.definitions[eval_cfg.definition]},
            solutions={eval_cfg.definition: [cand_sol]},
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
            solutions=[cand_sol.name],
        )

        bench = Benchmark(temp_traceset, bench_cfg)
        result_traceset = bench.run_all(dump_traces=False, resume=False)
        traces = list(result_traceset.traces.get(eval_cfg.definition, []) or [])

    # Index traces by workload uuid; keep best PASSED by max speedup.
    by_wl: Dict[str, Any] = {}
    for t in traces:
        ev = getattr(t, "evaluation", None)
        wl_uuid = getattr(getattr(t, "workload", None), "uuid", None)
        sol_name = getattr(t, "solution", None)
        if not isinstance(wl_uuid, str) or not isinstance(sol_name, str):
            continue
        if sol_name != cand_sol.name:
            continue
        prev = by_wl.get(wl_uuid)
        if prev is None:
            by_wl[wl_uuid] = t
            continue
        prev_ev = getattr(prev, "evaluation", None)
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

    def_name = eval_cfg.definition
    print(f"[{def_name}] Per-workload performance table (OpenEvolve-style vs_base; baseline from dataset traces)")
    print("workload_uuid                      | status        | speedup(x) | latency(ms) | ref_latency(ms) | vs_base(x)")
    print("-----------------------------------+---------------+------------+------------+----------------+----------")

    vs_base_vals: List[float] = []
    total = 0
    passed = 0

    for wl_wrap in workloads_wrapped:
        wl_uuid = wl_wrap.workload.uuid
        total += 1

        cand_t = by_wl.get(wl_uuid)
        cand_ev = getattr(cand_t, "evaluation", None) if cand_t is not None else None

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
                    lat_s = f"{float(lat):.3f}"
                if isinstance(ref, (int, float)):
                    ref_s = f"{float(ref):.3f}"

        # vs_base = baseline_latency(dataset trace) / candidate_latency (PASSED only, both present)
        try:
            if cand_ev is not None and cand_ev.status == EvaluationStatus.PASSED:
                cand_lat = float(getattr(getattr(cand_ev, "performance", None), "latency_ms", 0.0) or 0.0)
                base_lat = float(baseline_lat_by_wl.get(str(wl_uuid), 0.0) or 0.0)
                if cand_lat > 0.0 and base_lat > 0.0:
                    vs = base_lat / cand_lat
                    vs_base_vals.append(float(vs))
                    vsb_s = f"{float(vs):.2f}"
        except Exception:
            pass

        print(
            f"{wl_uuid:<35} | {status:<13} | {sp_s:>10} | {lat_s:>10} | {ref_s:>14} | {vsb_s:>8}"
        )

    # Match OpenEvolve: only compute mean_vs_base if we have baseline for every PASSED workload.
    mean_vs: Optional[float] = None
    if passed > 0 and len(vs_base_vals) == int(passed):
        mean_vs = (sum(vs_base_vals) / len(vs_base_vals)) if vs_base_vals else None
    print("\n[aggregate]")
    print(f"workloads_total={total} workloads_passed={passed}")
    print(f"baseline_solution={baseline_sol.name}")
    print(f"baseline_hw_key={current_hw_key}")
    print(f"mean_vs_base={mean_vs}")
    if mean_vs is not None:
        print(f"combined_score(mean_vs_base*100)={mean_vs * 100.0}")
    else:
        print("combined_score(mean_vs_base*100)=None")
        return 0


if __name__ == "__main__":
    raise SystemExit(main())

