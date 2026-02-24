"""Run OpenEvolve against flashinfer-bench kernels.

This is a thin convenience wrapper around OpenEvolve that:
- loads a small evaluator config (YAML)
- instantiates `FlashInferEvaluator` with that config (no env vars)

OpenEvolve controls LLM prompting/mutation/population; this script only wires
the evaluator.
"""

from __future__ import annotations

import argparse
import asyncio
from dataclasses import replace
import json
from datetime import UTC, datetime
from pathlib import Path
import sys
import os
import multiprocessing as mp
from typing import Any, Dict, List, Optional, Tuple
import uuid

from openevolve import OpenEvolve
from openevolve.config import load_config


def _load_mapping(path: str) -> dict:
    """Load a small YAML mapping file for evaluator settings."""
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(path)

    if p.suffix not in (".yaml", ".yml"):
        raise ValueError("Unsupported evaluator config file type (use .yaml/.yml)")

    try:
        from omegaconf import OmegaConf  # type: ignore
    except Exception as e:  # pragma: no cover
        raise RuntimeError(
            "OmegaConf is required to load evaluator config. Install with: pip install omegaconf"
        ) from e

    cfg = OmegaConf.load(str(p))
    data = OmegaConf.to_container(cfg, resolve=True)
    if not isinstance(data, dict):
        raise ValueError("Evaluator config must be a mapping")
    return data


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


def _resolve_db_path(*, oe_config_path: str) -> Optional[Path]:
    """Best-effort resolve of OpenEvolve database.db_path from the OpenEvolve config YAML."""
    try:
        cfg = _load_mapping(oe_config_path)
    except Exception:
        return None
    db = cfg.get("database", {}) if isinstance(cfg, dict) else {}
    db_path = None
    if isinstance(db, dict):
        db_path = db.get("db_path")
    if not isinstance(db_path, str) or not db_path.strip():
        return None
    p = Path(db_path)
    if not p.is_absolute():
        # OpenEvolve typically interprets relative db_path relative to the process CWD.
        # For robustness, try CWD first, then fall back to config directory.
        from_cwd = (Path.cwd() / p).resolve()
        from_cfg = (Path(oe_config_path).resolve().parent / p).resolve()
        p = from_cwd if from_cwd.exists() else from_cfg
    return p.resolve()


def _find_best_program_file(*, search_root: Path) -> Optional[Path]:
    """Find best program file under a given OpenEvolve run directory.

    OpenEvolve has historically written different filenames depending on version/config:
    - best/best_program.py
    - best/best_program.txt
    """
    try:
        for fname in ("best_program.py", "best_program.txt"):
            cand = search_root / "best" / fname
            if cand.exists() and cand.is_file():
                return cand
    except Exception:
        return None
    return None


def _find_latest_tmp_best_program() -> Optional[Path]:
    """Fallback: pick the newest /tmp/openevolve_*/best/best_program.{py,txt} if present."""
    try:
        cand: List[Path] = []
        cand.extend(Path("/tmp").glob("openevolve_*/best/best_program.py"))
        cand.extend(Path("/tmp").glob("openevolve_*/best/best_program.txt"))
        cand = sorted(cand, key=lambda p: p.stat().st_mtime)
        return cand[-1] if cand else None
    except Exception:
        return None


def _best_program_source(*, oe_config_path: str, result_obj: object) -> Tuple[Optional[str], Optional[str]]:
    """
    Return (path_or_text, debug_hint).

    We prefer a concrete file path because OpenEvolve typically writes best_program.{py,txt}.
    """
    # 0) Preferred: OpenEvolve EvolutionResult interface (newer versions)
    try:
        best_code = getattr(result_obj, "best_code", None)
        if isinstance(best_code, str) and best_code.strip():
            return best_code, "inlined from result.best_code"
    except Exception:
        pass
    # Also support dict-like results (some OpenEvolve wrappers return a dict with best_code/output_dir).
    try:
        if isinstance(result_obj, dict):
            best_code = result_obj.get("best_code")
            if isinstance(best_code, str) and best_code.strip():
                return best_code, "inlined from result['best_code']"
    except Exception:
        pass
    try:
        out_dir = getattr(result_obj, "output_dir", None)
        if isinstance(out_dir, str) and out_dir.strip():
            # Prefer on-disk best program file if present.
            for fname in ("best_program.py", "best_program.txt"):
                p = Path(out_dir).resolve() / "best" / fname
                if p.exists() and p.is_file():
                    return str(p), f"{fname} from result.output_dir={out_dir}"
    except Exception:
        pass

    # 1) Try to find best_program.py under configured db_path
    db_root = _resolve_db_path(oe_config_path=oe_config_path)
    if db_root:
        p = _find_best_program_file(search_root=db_root)
        if p:
            return str(p), f"{p.name} from db_path={db_root}"

    # 2) Try some common attrs/keys on the returned result object (version dependent)
    try:
        # dict-like
        if isinstance(result_obj, dict):
            # Best: a direct code string.
            for k in ("best_code", "best_program", "best_program_text", "program_text", "program"):
                v = result_obj.get(k)
                if isinstance(v, str) and v.strip():
                    return v, f"inlined from result[{k}]"
            # Next: best_program object with a `.code` field (OpenEvolve Program object).
            bp = result_obj.get("best_program")
            code = getattr(bp, "code", None)
            if isinstance(code, str) and code.strip():
                return code, "inlined from result['best_program'].code"
    except Exception:
        pass
    try:
        # attribute-like
        for k in ("best_code", "best_program", "best_program_text", "program_text", "program"):
            v = getattr(result_obj, k, None)
            if isinstance(v, str) and v.strip():
                return v, f"inlined from result.{k}"
        bp = getattr(result_obj, "best_program", None)
        code = getattr(bp, "code", None)
        if isinstance(code, str) and code.strip():
            return code, "inlined from result.best_program.code"
    except Exception:
        pass

    # 3) Fallback: newest /tmp/openevolve_*/best/best_program.py
    p = _find_latest_tmp_best_program()
    if p:
        return str(p), f"{p.name} from tmp={p}"

    return None, "no best program source found"


def _persist_solution(trace_set, definition_name: str, solution) -> Optional[Path]:
    """
    Persist a Solution JSON under solutions/<op_type>/<definition>/<solution>.json if TraceSet has a root.
    Mirrors examples/generate_kernels_and_eval.py behavior.
    """
    from flashinfer_bench.data.json_utils import save_json_file

    if getattr(trace_set, "root", None) is None:
        return None
    try:
        definition = trace_set.definitions[definition_name]
        dest = trace_set.solutions_path / definition.op_type / definition.name / f"{solution.name}.json"
        save_json_file(object=solution, path=dest)
        return dest
    except Exception as e:
        print(f"Error saving solution: {e}")
        import traceback

        traceback.print_exc()
        return None


def _best_program_to_solution(*, eval_cfg, best_program_src: str, hint: str):
    """
    Convert best program (path or inline text) into a flashinfer-bench Solution.
    """
    from flashinfer_bench.data import BuildSpec, Solution, SourceFile, SupportedLanguages

    raw_text = _maybe_read_text(best_program_src)
    lang = (getattr(eval_cfg, "language", None) or "triton").lower()
    target_gpu = getattr(eval_cfg, "target_gpu", None) or "H100"
    supported = {
        "python": SupportedLanguages.PYTHON,
        "triton": SupportedLanguages.TRITON,
        "cuda": SupportedLanguages.CUDA,
    }.get(lang, SupportedLanguages.TRITON)

    author = "openevolve"
    ts = datetime.now(UTC).strftime("%Y%m%d_%H%M%S")
    uid = uuid.uuid4().hex[:8]
    name = f"oe_best_{eval_cfg.definition}_{ts}_{uid}"

    # CUDA XML multi-file format
    if lang == "cuda" and (
        "<cuda_file" in raw_text or "<header_file" in raw_text or "<cpp_file" in raw_text
    ):
        # Reuse the evaluator's parser to keep formats consistent.
        from flashinfer_oe_evaluator import _parse_cuda_xml_files  # type: ignore

        files = _parse_cuda_xml_files(raw_text)
        sources = [SourceFile(path=fname, content=content) for fname, content in files.items()]
        return Solution(
            name=name,
            definition=eval_cfg.definition,
            author=author,
            spec=BuildSpec(
                language=supported,
                target_hardware=[target_gpu],
                entry_point="main.cpp::run",
            ),
            sources=sources,
            description=f"OpenEvolve best program ({hint})",
        )

    # Default: single-file program text
    path = "main.py" if lang in ("python", "triton") else "kernel.cu"
    entry_point = "main.py::run" if path.endswith(".py") else "main.cpp::run"
    return Solution(
        name=name,
        definition=eval_cfg.definition,
        author=author,
        spec=BuildSpec(
            language=supported,
            target_hardware=[target_gpu],
            entry_point=entry_point,
        ),
        sources=[SourceFile(path=path, content=raw_text)],
        description=f"OpenEvolve best program ({hint})",
    )


def _wandb_init(*, enable: bool, project: Optional[str], run_name: Optional[str]):
    if not enable:
        return None
    try:
        import wandb  # type: ignore
    except Exception as e:  # pragma: no cover
        raise RuntimeError("W&B requested but not installed. Install with: pip install wandb") from e
    return wandb.init(project=project, name=run_name)


def _wandb_upload_file(*, wb_run, file_path: Path, artifact_name: str, artifact_type: str = "log") -> None:
    """Upload a file as a W&B artifact (best-effort)."""
    if wb_run is None:
        return
    try:
        import wandb  # type: ignore

        art = wandb.Artifact(name=artifact_name, type=artifact_type)
        art.add_file(str(file_path))
        wb_run.log_artifact(art)
    except Exception:
        pass


def _find_latest_openevolve_log() -> Optional[Path]:
    """Best-effort: find newest OpenEvolve log file under /tmp/openevolve_*/logs/."""
    try:
        cand = list(Path("/tmp").glob("openevolve_*/logs/openevolve_*.log"))
        if not cand:
            return None
        cand.sort(key=lambda p: p.stat().st_mtime)
        return cand[-1]
    except Exception:
        return None


class _TeeIO:
    def __init__(self, *streams):
        self._streams = [s for s in streams if s is not None]

    def write(self, data):
        for s in self._streams:
            try:
                s.write(data)
            except Exception:
                pass
        return len(data)

    def flush(self):
        for s in self._streams:
            try:
                s.flush()
            except Exception:
                pass


def _maybe_capture_stdio_to_file(*, enable: bool, run_name: str) -> Optional[Path]:
    """If enabled, tee stdout/stderr to a local file so W&B can upload it."""
    if not enable:
        return None
    try:
        log_dir = Path(os.environ.get("OE_LOG_DIR", "")).expanduser() if os.environ.get("OE_LOG_DIR") else None
        if not log_dir:
            log_dir = Path.cwd() / "openevolve_logs"
        log_dir.mkdir(parents=True, exist_ok=True)
        ts = datetime.now(UTC).strftime("%Y%m%d_%H%M%S")
        path = (log_dir / f"{run_name}_{ts}.log").resolve()
        f = open(path, "a", encoding="utf-8", buffering=1)
        sys.stdout = _TeeIO(sys.stdout, f)  # type: ignore[assignment]
        sys.stderr = _TeeIO(sys.stderr, f)  # type: ignore[assignment]
        return path
    except Exception:
        return None


def _prepare_initial_program_file(*, initial_program: str, output_dir: Path) -> str:
    """
    OpenEvolve controller API expects an initial program *file path*.

    If `initial_program` is a path to an existing file, return it.
    Otherwise, treat it as inline text and write it under output_dir.
    """
    s = (initial_program or "").strip()
    if not s:
        raise SystemExit("--initial-program is required (path to file, or inline text).")
    try:
        p = Path(s)
        if p.exists() and p.is_file():
            return str(p.resolve())
    except OSError:
        # Very long inline strings may fail Path() conversion; treat as text.
        pass

    output_dir.mkdir(parents=True, exist_ok=True)
    # Keep the suffix stable-ish. Most of our kernels are python/triton text, but
    # OpenEvolve only needs a file to mutate; the evaluator decides how to interpret it.
    out = output_dir / f"initial_program_{uuid.uuid4().hex[:8]}.py"
    code = s
    if "EVOLVE-BLOCK-START" not in code:
        code = f"""# EVOLVE-BLOCK-START
{code}
# EVOLVE-BLOCK-END
"""
    out.write_text(code, encoding="utf-8")
    return str(out.resolve())


def _write_openevolve_evaluator_wrapper(
    *,
    eval_cfg_dict: Dict[str, Any],
    wrapper_dir: Path,
    openevolve_dir: Path,
) -> Path:
    """
    OpenEvolve supports passing an evaluator *file path*.

    This avoids OpenEvolve's "callable evaluator" wrapper which stores the callable in
    `openevolve.api` module globals (works with fork, breaks with spawn).
    """
    wrapper_dir.mkdir(parents=True, exist_ok=True)
    wrapper_path = wrapper_dir / f"flashinfer_oe_eval_{uuid.uuid4().hex[:8]}.py"
    payload = json.dumps(eval_cfg_dict, sort_keys=True)

    wrapper_code = f"""\
# Auto-generated by baselines/openevolve/run_evolve.py
import json
import sys

# Ensure we can import the repo-local evaluator module.
sys.path.insert(0, {json.dumps(str(openevolve_dir.resolve()))})

from flashinfer_oe_evaluator import FlashInferEvaluatorConfig, set_default_evaluator, evaluate as _evaluate

_CFG = json.loads({json.dumps(payload)})
set_default_evaluator(FlashInferEvaluatorConfig.from_config(_CFG))

def evaluate(program_text: str):
    return _evaluate(program_text)
"""
    wrapper_path.write_text(wrapper_code, encoding="utf-8")
    return wrapper_path


def main(argv: Optional[list[str]] = None) -> int:
    # IMPORTANT: OpenEvolve may use a multiprocessing process pool for evaluations.
    # If the start method is "fork" (default on Linux), CUDA can break with:
    #   "Cannot re-initialize CUDA in forked subprocess"
    # Force "spawn" early for CUDA-safe subprocess behavior.
    try:
        mp.set_start_method("spawn", force=True)
    except Exception:
        pass
    try:  # pragma: no cover
        import torch  # type: ignore

        torch.multiprocessing.set_start_method("spawn", force=True)  # type: ignore[attr-defined]
    except Exception:
        pass

    p = argparse.ArgumentParser(description="Evolve FlashInfer kernels with OpenEvolve")
    p.add_argument(
        "--config",
        type=str,
        default=str(Path(__file__).with_name("example_config.yaml")),
        help="Path to an OpenEvolve YAML config.",
    )
    p.add_argument(
        "--initial-program",
        type=str,
        default="",
        help=(
            "Initial candidate program (path to file, or inline text). "
            "Required for the controller-based runner used by this script."
        ),
    )

    # OpenEvolve run control
    p.add_argument(
        "--output-dir",
        type=str,
        default="",
        help=(
            "Directory where OpenEvolve writes logs/checkpoints/best_program. "
            "If empty, defaults to the configured database.db_path (if present), otherwise CWD/openevolve_output."
        ),
    )
    p.add_argument("--iterations", type=int, default=0, help="Max evolution iterations (0 = use config).")
    p.add_argument("--target-score", type=float, default=0.0, help="Stop when best score >= target (0 = disabled).")
    p.add_argument(
        "--checkpoint",
        type=str,
        default="",
        help="Path to a checkpoint directory to resume from (e.g. .../checkpoints/checkpoint_100).",
    )

    p.add_argument(
        "--evaluator-config",
        type=str,
        default="",
        help=(
            "Path to a YAML mapping of evaluator settings. "
            "Keys are passed to the evaluator object (no env vars)."
        ),
    )
    p.add_argument(
        "--dataset-path",
        type=str,
        default="",
        help="Override evaluator dataset_path (instead of the value in --evaluator-config).",
    )

    p.add_argument(
        "--bench-timeout-seconds",
        type=int,
        default=0,
        help=(
            "Per-workload timeout inside flashinfer-bench BenchmarkConfig.timeout_seconds. "
            "If 0, use evaluator-config value (timeout_seconds/timeout) or default (300)."
        ),
    )

    p.add_argument(
        "--final-eval-all-workloads",
        action="store_true",
        help="After evolution completes, run one final evaluation of the best program on ALL workloads for the definition.",
    )

    # W&B (optional)
    p.add_argument("--enable-wandb", action="store_true", help="Enable Weights & Biases logging and artifact upload.")
    p.add_argument("--wandb-project", type=str, default="", help="W&B project name.")
    p.add_argument("--wandb-run-name", type=str, default="", help="W&B run name.")
    p.add_argument(
        "--wandb-log-file",
        type=str,
        action="append",
        default=[],
        help="Path to a log file to upload to W&B as an artifact (can be provided multiple times).",
    )

    args = p.parse_args(argv)

    if not args.evaluator_config:
        raise SystemExit("--evaluator-config is required and must be a .yaml/.yml mapping.")

    base = _load_mapping(args.evaluator_config)
    if args.dataset_path and str(args.dataset_path).strip():
        base["dataset_path"] = str(Path(args.dataset_path).expanduser().resolve())
    if args.bench_timeout_seconds and args.bench_timeout_seconds > 0:
        # FlashInferEvaluatorConfig.from_config accepts both `timeout_seconds` and `timeout`.
        base["timeout_seconds"] = int(args.bench_timeout_seconds)

    # Construct evaluator
    from flashinfer_oe_evaluator import FlashInferEvaluator, FlashInferEvaluatorConfig

    try:
        eval_cfg = FlashInferEvaluatorConfig.from_config(base)
    except ValueError as e:
        raise SystemExit(str(e))
    # IMPORTANT: For CUDA safety OpenEvolve must use spawn workers, but OpenEvolve's
    # "callable evaluator" path relies on fork (it stores the callable in openevolve.api globals).
    # So we pass an evaluator *file path* wrapper instead of a callable.
    evaluator_wrapper_path = _write_openevolve_evaluator_wrapper(
        eval_cfg_dict=base,
        wrapper_dir=Path("/tmp") / "openevolve_eval_wrappers",
        openevolve_dir=Path(__file__).resolve().parent,
    )

    # Start W&B immediately so the run shows up at the beginning of a long evolution.
    wb_run = _wandb_init(
        enable=bool(args.enable_wandb),
        project=(args.wandb_project or None) if args.wandb_project else None,
        run_name=(args.wandb_run_name or None) if args.wandb_run_name else None,
    )

    # If W&B is enabled, capture this script's stdout/stderr automatically.
    captured_log_path = _maybe_capture_stdio_to_file(
        enable=bool(args.enable_wandb),
        run_name=(args.wandb_run_name or "openevolve"),
    )

    if wb_run is not None:
        try:
            wb_run.log(
                {
                    "status": "started",
                    "config/openevolve_config": str(Path(args.config).resolve()),
                    "config/evaluator_config": str(Path(args.evaluator_config).resolve()),
                    "config/definition": str(eval_cfg.definition),
                    "config/language": str(getattr(eval_cfg, "language", "")),
                    "config/target_gpu": str(getattr(eval_cfg, "target_gpu", "")),
                    "config/warmup_runs": int(getattr(eval_cfg, "warmup_runs", 0)),
                    "config/iterations": int(getattr(eval_cfg, "iterations", 0)),
                    "config/num_trials": int(getattr(eval_cfg, "num_trials", 0)),
                    "config/use_isolated_runner": bool(getattr(eval_cfg, "use_isolated_runner", False)),
                    "config/output_dir": (args.output_dir or ""),
                    "config/checkpoint": (args.checkpoint or ""),
                    "config/max_iterations_override": int(args.iterations or 0),
                    "config/target_score_override": float(args.target_score or 0.0),
                }
            )
        except Exception:
            pass

    # Load OpenEvolve config and run controller directly so we can support checkpoints/resume.
    cfg_obj = load_config(args.config)

    # Resolve output dir: prefer CLI, else database.db_path, else CWD/openevolve_output.
    out_dir: Optional[Path] = None
    if args.output_dir:
        out_dir = Path(args.output_dir).expanduser().resolve()
    else:
        db_p = _resolve_db_path(oe_config_path=args.config)
        if db_p is not None:
            out_dir = db_p if db_p.is_dir() else db_p.parent
    if out_dir is None:
        out_dir = (Path.cwd() / "openevolve_output").resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    # If user provides --output-dir, also override OpenEvolve database.db_path to live under it.
    # This makes the DB location deterministic: <output_dir>/oe_db
    try:
        if args.output_dir and str(args.output_dir).strip():
            oe_db_dir = (out_dir / "oe_db").resolve()
            oe_db_dir.mkdir(parents=True, exist_ok=True)
            # OpenEvolve config is a dataclass (openevolve.config.Config)
            if hasattr(cfg_obj, "database") and getattr(cfg_obj, "database") is not None:
                setattr(cfg_obj.database, "db_path", str(oe_db_dir))
                # Persist to disk when db_path is provided.
                if hasattr(cfg_obj.database, "in_memory"):
                    setattr(cfg_obj.database, "in_memory", False)
            print(f"[config] Overrode OpenEvolve database.db_path => {oe_db_dir}")
    except Exception as e:
        print(f"[config] Warning: failed to override database.db_path from --output-dir: {e}")

    program_path = _prepare_initial_program_file(initial_program=args.initial_program, output_dir=out_dir)

    checkpoint_path: Optional[str] = None
    if args.checkpoint and str(args.checkpoint).strip():
        checkpoint_path = str(Path(args.checkpoint).expanduser().resolve())
        if not Path(checkpoint_path).exists():
            raise SystemExit(f"--checkpoint path not found: {checkpoint_path}")

    controller = OpenEvolve(
        initial_program_path=program_path,
        evaluation_file=str(evaluator_wrapper_path),
        config=cfg_obj,
        output_dir=str(out_dir),
    )

    max_iters: Optional[int] = int(args.iterations) if int(args.iterations or 0) > 0 else None
    tgt: Optional[float] = float(args.target_score) if float(args.target_score or 0.0) > 0 else None

    best_program = asyncio.run(
        controller.run(iterations=max_iters, target_score=tgt, checkpoint_path=checkpoint_path)
    )
    # Provide a small result-like object for downstream helpers.
    result: Dict[str, Any] = {
        "best_code": (best_program.code if best_program is not None else ""),
        "best_program": best_program,
        "output_dir": str(out_dir),
    }
    print(result)

    # Upload any explicitly provided log files (e.g. script tee output)
    if wb_run is not None:
        # Upload our auto-captured stdout/stderr log (best-effort).
        if captured_log_path is not None and captured_log_path.exists():
            _wandb_upload_file(
                wb_run=wb_run,
                file_path=captured_log_path,
                artifact_name=f"stdout_{captured_log_path.stem}",
                artifact_type="log",
            )

        for lf in (args.wandb_log_file or []):
            try:
                pth = Path(lf).expanduser().resolve()
                if pth.exists() and pth.is_file():
                    _wandb_upload_file(
                        wb_run=wb_run,
                        file_path=pth,
                        artifact_name=f"run_log_{pth.stem}",
                        artifact_type="log",
                    )
            except Exception:
                continue

        # Best-effort: also upload the newest OpenEvolve internal log file
        try:
            oe_log = _find_latest_openevolve_log()
            if oe_log is not None and oe_log.exists():
                _wandb_upload_file(
                    wb_run=wb_run,
                    file_path=oe_log.resolve(),
                    artifact_name=f"openevolve_log_{oe_log.stem}",
                    artifact_type="log",
                )
        except Exception:
            pass

    # Persist the final winner as a dataset Solution JSON (best-effort).
    try:
        from flashinfer_bench.data import TraceSet

        best_src, hint = _best_program_source(oe_config_path=args.config, result_obj=result)
        if best_src:
            trace_set = TraceSet.from_path(eval_cfg.dataset_path)
            if eval_cfg.definition in trace_set.definitions:
                sol = _best_program_to_solution(eval_cfg=eval_cfg, best_program_src=best_src, hint=hint or "")
                trace_set.solutions.setdefault(eval_cfg.definition, []).append(sol)
                saved_path = _persist_solution(trace_set, eval_cfg.definition, sol)
                if saved_path:
                    print(f"[persist] ✓ Saved best solution to: {saved_path}")
                    # Copy solution to output_dir for artifact upload
                    try:
                        import shutil
                        dest = out_dir / saved_path.name
                        shutil.copy2(saved_path, dest)
                        print(f"[upload] Copied solution to output_dir: {dest}")
                    except Exception as e:
                        print(f"[upload] Warning: Failed to copy solution to output_dir: {e}")
                    if wb_run is not None:
                        _wandb_upload_file(
                            wb_run=wb_run,
                            file_path=Path(saved_path).resolve(),
                            artifact_name=f"best_solution_{sol.name}",
                            artifact_type="solution",
                        )
                else:
                    print("[persist] ✗ Could not save best solution (trace_set.root is None)")
            else:
                print(f"[persist] ✗ Definition '{eval_cfg.definition}' not found in dataset; skipping save.")
        else:
            print(f"[persist] ✗ Could not locate best program ({hint}); skipping save.")
    except Exception as e:
        print(f"[persist] ✗ Failed to persist best solution: {e}")

    if args.final_eval_all_workloads:
        # Find best program and run a full-workload benchmark on it (like generate_kernels_and_eval.py end eval).
        best_src, hint = _best_program_source(oe_config_path=args.config, result_obj=result)
        if not best_src:
            print(f"[final-eval] Could not locate best program ({hint}); skipping.")
            # IMPORTANT: don't early-return; we still want to upload artifacts + finish W&B run.
            best_src = None

        if best_src:
            try:
                full_cfg = replace(
                    eval_cfg,
                    # Evaluator semantics: num_feedback_workloads <= 0 and no feedback_workloads => evaluate ALL workloads.
                    feedback_workloads=None,
                    num_feedback_workloads=0,
                    num_eval_workload=0,
                    # Always print the per-workload perf table for the final winner eval.
                    verbose_table=True,
                )
                full_eval = FlashInferEvaluator(full_cfg)
                print(f"[final-eval] Running best program on ALL workloads ({hint})")
                final_res = full_eval.evaluate(best_src)
                print("[final-eval] Result:")
                print(final_res)

                # Best-effort: also dump a JSON summary under the OpenEvolve run directory.
                try:
                    final_out_dir: Optional[Path] = None
                    try:
                        od = getattr(result, "output_dir", None)
                        if isinstance(od, str) and od.strip():
                            final_out_dir = Path(od).resolve()
                    except Exception:
                        final_out_dir = None
                    if final_out_dir is None:
                        final_out_dir = _resolve_db_path(oe_config_path=args.config)
                    if final_out_dir is None:
                        final_out_dir = Path.cwd()
                    final_out_dir.mkdir(parents=True, exist_ok=True)
                    out_path = final_out_dir / "final_eval.json"
                    payload = {
                        "best_program_source": best_src,
                        "best_program_hint": hint,
                        "definition": eval_cfg.definition,
                        "metrics": getattr(final_res, "metrics", None),
                        "artifacts": getattr(final_res, "artifacts", None),
                    }
                    out_path.write_text(json.dumps(payload, indent=2, sort_keys=True))
                    print(f"[final-eval] Wrote summary: {out_path}")
                    if wb_run is not None:
                        _wandb_upload_file(
                            wb_run=wb_run,
                            file_path=out_path.resolve(),
                            artifact_name=f"final_eval_{eval_cfg.definition}",
                            artifact_type="eval",
                        )
                except Exception:
                    pass
            except Exception as e:
                print(f"[final-eval] Failed: {e}")

    # Upload output_dir as a single artifact. It should contain oe_db/ and any copied oe_*.json solutions.
    if wb_run is not None:
        try:
            import wandb  # type: ignore
            if out_dir.exists() and out_dir.is_dir():
                print(f"[upload] Uploading output_dir to W&B: {out_dir}")
                artifact = wandb.Artifact(name=f"{wb_run.name}-output_dir", type="dataset")
                artifact.add_dir(str(out_dir))
                logged = wb_run.log_artifact(artifact)
                try:
                    logged.wait()
                except Exception:
                    pass
                print("[upload] ✓ Successfully uploaded output_dir to W&B")
            else:
                print(
                    f"[upload] output_dir not found: {out_dir}. "
                    f"Current cfg_obj.database.db_path={getattr(getattr(cfg_obj, 'database', None), 'db_path', None)}"
                )
            
        except Exception as e:
            print(f"[upload] ✗ Failed to upload artifacts: {e}")
    
    if wb_run is not None:
        try:
            wb_run.log({"status": "finished"})
        except Exception:
            pass
        try:
            wb_run.finish()
        except Exception:
            pass
    return 0


if __name__ == "__main__":
    raise SystemExit(main())