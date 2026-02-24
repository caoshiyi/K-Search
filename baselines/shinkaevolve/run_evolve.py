"""
Convenience launcher for ShinkaEvolve (FlashInfer kernel evolution).

This mirrors `baselines/openevolve/run_evolve.py` in spirit:
  - wires an evaluator script + initial program + evaluator-config YAML
  - runs a local evolution loop (best-effort) using ShinkaEvolve's Python API

If ShinkaEvolve API names differ in your installed version, you can still run
`shinka_launch` directly; see `baselines/shinkaevolve/run_evolve.sh`.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any, Dict, Optional
import os
import subprocess
import sys
from datetime import datetime


def _extract_yaml_block_scalar(text: str, *, key_path: list[str]) -> Optional[str]:
    """
    Minimal YAML extractor for this repo's configs.
    Supports a nested mapping path ending in a key whose value is a block scalar:
      prompt:
        system_message: |
          ...
    Returns the dedented block content as a string, or None.
    """
    lines = text.splitlines()

    def _find_mapping_block(start_idx: int, indent: int, key: str) -> Optional[tuple[int, int]]:
        # Find "key:" at current indent, return (line_idx, child_indent)
        prefix = " " * indent + f"{key}:"
        for i in range(start_idx, len(lines)):
            ln = lines[i]
            if not ln.strip() or ln.lstrip().startswith("#"):
                continue
            # stop if indent decreased (we left the parent mapping)
            if len(ln) - len(ln.lstrip(" ")) < indent:
                return None
            if ln.startswith(prefix):
                # child indent is next non-empty line's indent (best-effort)
                return i, indent + 2
        return None

    # Walk down to the parent mapping of the final key
    idx = 0
    indent = 0
    for k in key_path[:-1]:
        found = _find_mapping_block(idx, indent, k)
        if not found:
            return None
        idx, indent = found
        idx += 1

    # Find the final key line
    final_key = key_path[-1]
    prefix = " " * indent + f"{final_key}:"
    key_line_idx = None
    for i in range(idx, len(lines)):
        ln = lines[i]
        if not ln.strip() or ln.lstrip().startswith("#"):
            continue
        if len(ln) - len(ln.lstrip(" ")) < indent:
            return None
        if ln.startswith(prefix):
            key_line_idx = i
            break
    if key_line_idx is None:
        return None

    # Must be a block scalar ("|")
    if "|" not in lines[key_line_idx]:
        return None

    # Capture subsequent indented lines as the block
    block_lines: list[str] = []
    # Block content indent is determined by first non-empty content line
    content_indent: Optional[int] = None
    for j in range(key_line_idx + 1, len(lines)):
        ln = lines[j]
        if not ln.strip():
            block_lines.append("")
            continue
        cur_indent = len(ln) - len(ln.lstrip(" "))
        if cur_indent <= indent:
            break
        if content_indent is None:
            content_indent = cur_indent
        block_lines.append(ln[content_indent:] if cur_indent >= content_indent else ln.lstrip(" "))
    # Trim leading/trailing empty lines
    while block_lines and block_lines[0] == "":
        block_lines.pop(0)
    while block_lines and block_lines[-1] == "":
        block_lines.pop()
    out = "\n".join(block_lines)
    return out if out.strip() else None


def _extract_yaml_scalar(text: str, *, key_path: list[str]) -> Optional[str]:
    """
    Minimal YAML extractor for simple scalar values at a nested key path, e.g.:
      llm:
        primary_model: "gemini-3-pro-preview"
    Returns the raw scalar as a string (quotes stripped), or None.
    """
    lines = text.splitlines()
    idx = 0
    indent = 0
    for k in key_path[:-1]:
        prefix = " " * indent + f"{k}:"
        found = None
        for i in range(idx, len(lines)):
            ln = lines[i]
            if not ln.strip() or ln.lstrip().startswith("#"):
                continue
            if len(ln) - len(ln.lstrip(" ")) < indent:
                return None
            if ln.startswith(prefix):
                found = i
                break
        if found is None:
            return None
        idx = found + 1
        indent += 2

    final_key = key_path[-1]
    prefix = " " * indent + f"{final_key}:"
    for i in range(idx, len(lines)):
        ln = lines[i]
        if not ln.strip() or ln.lstrip().startswith("#"):
            continue
        if len(ln) - len(ln.lstrip(" ")) < indent:
            return None
        if ln.startswith(prefix):
            val = ln.split(":", 1)[1].strip()
            # strip inline comment
            if " #" in val:
                val = val.split(" #", 1)[0].rstrip()
            if (val.startswith('"') and val.endswith('"')) or (val.startswith("'") and val.endswith("'")):
                val = val[1:-1]
            return val if val != "" else None
    return None


def _load_yaml_as_dict(config_path: Path) -> Optional[Dict[str, Any]]:
    """
    Load a YAML file into a plain Python dict.
    Prefers OmegaConf if available (to support interpolation/merges), but falls
    back to PyYAML so this launcher works without extra deps.
    """
    # Try OmegaConf first (best compatibility with OpenEvolve configs)
    try:
        from omegaconf import OmegaConf  # type: ignore

        cfg = OmegaConf.load(str(config_path))
        data = OmegaConf.to_container(cfg, resolve=True)
        return data if isinstance(data, dict) else None
    except Exception:
        pass

    # Fallback: basic YAML load
    try:
        import yaml  # type: ignore

        with open(config_path, "r", encoding="utf-8") as f:
            data = yaml.safe_load(f)
        return data if isinstance(data, dict) else None
    except Exception:
        pass

    # Final fallback: dependency-free extraction for the fields we need.
    try:
        text = config_path.read_text(encoding="utf-8")
    except Exception:
        return None

    sys_msg = _extract_yaml_block_scalar(text, key_path=["prompt", "system_message"])
    llm_primary = _extract_yaml_scalar(text, key_path=["llm", "primary_model"])
    llm_secondary = _extract_yaml_scalar(text, key_path=["llm", "secondary_model"])
    llm_wp = _extract_yaml_scalar(text, key_path=["llm", "primary_model_weight"])
    llm_ws = _extract_yaml_scalar(text, key_path=["llm", "secondary_model_weight"])
    llm_temp = _extract_yaml_scalar(text, key_path=["llm", "temperature"])
    llm_max_tokens = _extract_yaml_scalar(text, key_path=["llm", "max_tokens"])

    out: Dict[str, Any] = {}
    if sys_msg is not None:
        out["prompt"] = {"system_message": sys_msg}
    llm: Dict[str, Any] = {}
    if llm_primary is not None:
        llm["primary_model"] = llm_primary
    if llm_secondary is not None:
        llm["secondary_model"] = llm_secondary
    if llm_wp is not None:
        try:
            llm["primary_model_weight"] = float(llm_wp) if "." in llm_wp else int(llm_wp)
        except Exception:
            llm["primary_model_weight"] = llm_wp
    if llm_ws is not None:
        try:
            llm["secondary_model_weight"] = float(llm_ws) if "." in llm_ws else int(llm_ws)
        except Exception:
            llm["secondary_model_weight"] = llm_ws
    if llm_temp is not None:
        try:
            llm["temperature"] = float(llm_temp)
        except Exception:
            llm["temperature"] = llm_temp
    if llm_max_tokens is not None:
        try:
            llm["max_tokens"] = int(float(llm_max_tokens))
        except Exception:
            llm["max_tokens"] = llm_max_tokens
    if llm:
        out["llm"] = llm
    return out if out else None


def _load_openevolve_prompt_system_message(config_path: Path) -> Optional[str]:
    """
    Mirror OpenEvolve: reuse `prompt.system_message` as ShinkaEvolve's task_sys_msg.
    """
    data = _load_yaml_as_dict(config_path)
    if not isinstance(data, dict):
        return None
    prompt = data.get("prompt")
    if not isinstance(prompt, dict):
        return None
    sm = prompt.get("system_message")
    return str(sm) if isinstance(sm, str) and sm.strip() else None


def _load_openevolve_llm_config(config_path: Path) -> Dict[str, Any]:
    """
    Mirror OpenEvolve: reuse LLM knobs from `llm:` section where possible.
    Returns a dict with keys:
      - models (list[str])
      - model_sample_probs (list[float])  # optional, sums to 1.0
      - temperatures (float)              # optional
      - max_tokens (int)                  # optional
    """
    out: Dict[str, Any] = {}
    data = _load_yaml_as_dict(config_path)
    if not isinstance(data, dict):
        return out
    llm = data.get("llm")
    if not isinstance(llm, dict):
        return out
    primary = llm.get("primary_model")
    secondary = llm.get("secondary_model")
    w_primary = llm.get("primary_model_weight", 1)
    w_secondary = llm.get("secondary_model_weight", 1)

    def _as_clean_str(x: Any) -> Optional[str]:
        return str(x).strip() if isinstance(x, str) and str(x).strip() else None

    def _as_float(x: Any) -> Optional[float]:
        if isinstance(x, bool):
            return None
        if isinstance(x, (int, float)):
            return float(x)
        return None

    m_primary = _as_clean_str(primary)
    m_secondary = _as_clean_str(secondary)
    wp = _as_float(w_primary)
    ws = _as_float(w_secondary)

    models: list[str] = []
    weights: list[float] = []
    if m_primary is not None and (wp is None or wp > 0):
        models.append(m_primary)
        weights.append(float(wp) if wp is not None else 1.0)
    if m_secondary is not None and (ws is None or ws > 0):
        models.append(m_secondary)
        weights.append(float(ws) if ws is not None else 1.0)

    # If both weights were explicitly 0 (or invalid) but primary exists, fall back to primary only.
    if not models and m_primary is not None:
        models = [m_primary]
        weights = [1.0]

    if models:
        out["models"] = models
        if len(models) > 1:
            s = float(sum(weights))
            if s > 0:
                out["model_sample_probs"] = [float(w) / s for w in weights]
    temp = llm.get("temperature")
    if isinstance(temp, (int, float)):
        out["temperatures"] = float(temp)
    mt = llm.get("max_tokens")
    if isinstance(mt, int):
        out["max_tokens"] = int(mt)
    return out


def _make_shinka_task_sys_msg(*, kernel_spec: str) -> str:
    """
    ShinkaEvolve evolves a *Python* file (`main.py`) and applies patches by extracting
    a ```python fenced code block from the LLM response.

    We still want to use the OpenEvolve kernel spec, but as *task content* that the
    Python program should embed into `candidate_kernel()` (typically as a CUDA XML
    string returned in code-mode).
    """
    spec = kernel_spec.strip()
    return (
        "You are evolving a Python program that defines `candidate_kernel()`.\n"
        "\n"
        "ABSOLUTE OUTPUT FORMAT (required for patch application):\n"
        "- You MUST output exactly three top-level tags in this order: <NAME>, <DESCRIPTION>, <CODE>.\n"
        "- Inside <CODE>, you MUST include a fenced code block that starts with ```python and ends with ```.\n"
        "- The ```python block MUST contain a complete runnable Python file.\n"
        "- Do NOT output any other fenced code blocks anywhere (no ```cuda, no ```xml, no bare ```).\n"
        "- Do NOT output CUDA/XML as your top-level response; it must be embedded INSIDE the Python code.\n"
        "\n"
        "Hard requirements:\n"
        "- Only change code between EVOLVE-BLOCK-START and EVOLVE-BLOCK-END.\n"
        "- Keep the file runnable and keep `run_experiment(**kwargs)` intact.\n"
        "- `candidate_kernel()` must return a dict describing the candidate:\n"
        "  - Prefer returning CUDA XML via:\n"
        "    {\"mode\": \"code\", \"language\": \"cuda\", \"code\": \"<...CUDA XML...>\"}\n"
        "  - The CUDA XML must follow the kernel specification below.\n"
        "- For CUDA XML, keep it in a Python string (e.g. `kernel_xml = f\"\"\"...\"\"\"`) and return it via the dict.\n"
        "\n"
        "CRITICAL CUDA-EXTENSION CONSTRAINTS (to avoid benchmark compile errors):\n"
        "- Your CUDA candidate MUST be KernelGenerator XML with exactly these 3 files: kernel.h, kernel.cu, main.cpp.\n"
        "- `main.cpp` is compiled with a C++ compiler (not nvcc). Therefore:\n"
        "  - DO NOT use CUDA kernel launch syntax `<<< >>>` in `main.cpp`.\n"
        "  - DO NOT define `__global__` kernels in `main.cpp`.\n"
        "  - `main.cpp` should only expose `run(...)` and call a host launcher declared in `kernel.h`.\n"
        "- Put all kernel launches and any CUDA runtime calls in `kernel.cu` (compiled by nvcc).\n"
        "- `kernel.h` must declare the host launcher (e.g. `void launch_mla_decode(..., cudaStream_t stream);`).\n"
        "\n"
        "Kernel specification (from OpenEvolve config):\n"
        f"{spec}\n"
    )

def _load_openevolve_max_iterations(config_path: Path) -> Optional[int]:
    """
    OpenEvolve-style configs in this repo use `max_iterations` as the evolution length.
    ShinkaEvolve uses `num_generations`. This helper extracts `max_iterations` so the
    launcher can map it to ShinkaEvolve.
    """
    data = _load_yaml_as_dict(config_path)
    if not isinstance(data, dict):
        return None
    v = data.get("max_iterations", None)
    if isinstance(v, bool):
        return None
    if isinstance(v, int):
        return int(v)
    if isinstance(v, float) and float(v).is_integer():
        return int(v)
    return None


def _make_final_eval_all_workloads_config_text(original_yaml: str) -> str:
    """
    Create a copy of an evaluator-config YAML that forces "all workloads" evaluation:
      - remove `feedback_workloads:` block (if present)
      - set `num_feedback_workloads: 0` (so _select_workloads returns all)
      - set `num_eval_workload: 0`
    This is intentionally text-based (no extra deps).
    """
    import re

    lines = original_yaml.splitlines(True)
    out: list[str] = []

    i = 0
    while i < len(lines):
        line = lines[i]
        if re.match(r"^\s*feedback_workloads\s*:\s*$", line):
            base_indent = len(line) - len(line.lstrip(" "))
            i += 1
            # skip following indented block
            while i < len(lines):
                ln = lines[i]
                if not ln.strip() or ln.lstrip().startswith("#"):
                    i += 1
                    continue
                cur_indent = len(ln) - len(ln.lstrip(" "))
                if cur_indent <= base_indent:
                    break
                i += 1
            continue

        out.append(line)
        i += 1

    # Rewrite/add knobs
    def _set_scalar(key: str, value: str) -> None:
        nonlocal out
        import re

        pat = re.compile(rf"^(\s*){re.escape(key)}\s*:\s*.*$", re.M)
        replaced = False
        new_out: list[str] = []
        for ln in out:
            m = pat.match(ln.rstrip("\n"))
            if m and not replaced:
                indent = m.group(1)
                new_out.append(f"{indent}{key}: {value}\n")
                replaced = True
            else:
                new_out.append(ln)
        out = new_out
        if not replaced:
            if out and not out[-1].endswith("\n"):
                out[-1] = out[-1] + "\n"
            out.append(f"{key}: {value}\n")

    _set_scalar("num_feedback_workloads", "0")
    _set_scalar("num_eval_workload", "0")
    return "".join(out)

def _wandb_init(*, enable: bool, project: str, run_name: str):
    if not enable:
        return None
    try:
        import wandb  # type: ignore
    except Exception as e:
        print("WARN: --enable-wandb set but wandb is not installed. Install with: pip install wandb")
        print("WARN:", repr(e))
        return None
    try:
        return wandb.init(project=(project or None), name=(run_name or None))
    except Exception as e:
        print("WARN: wandb.init failed:", repr(e))
        return None


def _wandb_upload_file(*, wb_run, file_path: Path, artifact_name: str, artifact_type: str = "log") -> None:
    if wb_run is None:
        return
    try:
        import wandb  # type: ignore

        p = Path(file_path).expanduser().resolve()
        if not p.exists() or not p.is_file():
            return
        art = wandb.Artifact(name=artifact_name, type=artifact_type)
        art.add_file(str(p))
        wb_run.log_artifact(art)
    except Exception:
        return


def _wandb_upload_dir(*, wb_run, dir_path: Path, artifact_name: str, artifact_type: str = "run_dir") -> None:
    if wb_run is None:
        return
    try:
        import wandb  # type: ignore

        p = Path(dir_path).expanduser().resolve()
        if not p.exists() or not p.is_dir():
            return
        art = wandb.Artifact(name=artifact_name, type=artifact_type)
        art.add_dir(str(p))
        wb_run.log_artifact(art)
    except Exception:
        return


def main(argv: Optional[list[str]] = None) -> int:
    p = argparse.ArgumentParser(description="Evolve FlashInfer kernels with ShinkaEvolve (best-effort launcher)")
    p.add_argument(
        "--initial-program",
        type=str,
        default=str(Path(__file__).with_name("flashinfer_initial.py")),
        help="Initial ShinkaEvolve program file (contains EVOLVE-BLOCK).",
    )
    p.add_argument(
        "--eval-program",
        type=str,
        default=str(Path(__file__).with_name("evaluate_flashinfer.py")),
        help="Evaluation script path (invoked by Shinka jobs).",
    )
    p.add_argument(
        "--evaluator-config",
        type=str,
        required=True,
        help="YAML mapping passed to run_experiment(**kwargs) (dataset_path, definition, ...).",
    )
    p.add_argument("--num-generations", type=int, default=20)
    p.add_argument("--max-parallel-jobs", type=int, default=1)
    p.add_argument(
        "--patch-type",
        type=str,
        default="full",
        choices=["diff", "full", "cross"],
        help=(
            "ShinkaEvolve patch type to generate. "
            "Supported by upstream ShinkaEvolve: diff, full, cross. "
            "Default: full (more robust than diff for kernel-writing tasks). "
            "Note: this launcher intentionally uses exactly one patch type per run."
        ),
    )
    p.add_argument(
        "--final-eval-all-workloads",
        action="store_true",
        help=(
            "After evolution, run one final evaluation on results_dir/best/main.py "
            "with all workloads (disables feedback_workloads sampling)."
        ),
    )
    p.add_argument("--enable-wandb", action="store_true", help="Enable Weights & Biases logging + artifact upload.")
    p.add_argument("--wandb-project", type=str, default="", help="W&B project name.")
    p.add_argument("--wandb-run-name", type=str, default="", help="W&B run name.")
    p.add_argument(
        "--wandb-log-file",
        type=str,
        action="append",
        default=[],
        help="Path to a log file to upload to W&B as an artifact (can be provided multiple times).",
    )
    p.add_argument(
        "--task-sys-msg",
        type=str,
        default=(
            "Evolve the kernel candidate inside EVOLVE-BLOCK. "
            "Prefer CUDA XML with kernel.h/kernel.cu/main.cpp. "
            "Maximize combined_score while keeping correctness within rtol/atol."
        ),
    )
    p.add_argument(
        "--openevolve-config",
        type=str,
        default="",
        help="Path to an OpenEvolve config.yaml to mirror (uses llm: + prompt.system_message).",
    )
    p.add_argument(
        "--llm-model",
        action="append",
        default=[],
        help=(
            "LLM model name for ShinkaEvolve mutations. Can be passed multiple times. "
            "Examples: azure-gpt-4.1-mini (requires AZURE_OPENAI_API_KEY/AZURE_API_ENDPOINT), "
            "gemini-3-pro-preview (requires GEMINI_API_KEY)."
        ),
    )
    p.add_argument("--db-path", type=str, default=str(Path(__file__).with_name("shinka_output")))
    p.add_argument(
        "--embedding-model",
        type=str,
        default="",
        help=(
            "Program embedding model used by ShinkaEvolve for clustering/novelty. "
            "Example for Ollama/OpenAI-compatible endpoint: "
            "local-qwen3-8b-embedding-https://127.0.0.1:11434/v1"
        ),
    )
    p.add_argument("--num-runs", type=int, default=1, help="Forwarded to evaluate_flashinfer.py --num-runs")
    args = p.parse_args(argv)

    initial_program = Path(args.initial_program).expanduser().resolve()
    eval_program = Path(args.eval_program).expanduser().resolve()
    evaluator_config = Path(args.evaluator_config).expanduser().resolve()
    db_path = Path(args.db_path).expanduser().resolve()
    db_path.mkdir(parents=True, exist_ok=True)

    # Try ShinkaEvolve Python API first (names may vary by version).
    try:
        from shinka.core import EvolutionRunner, EvolutionConfig  # type: ignore
        from shinka.launch import LocalJobConfig  # type: ignore
        from shinka.database import DatabaseConfig  # type: ignore
    except Exception as e:
        print("ERROR: Could not import ShinkaEvolve Python API.")
        print("Install ShinkaEvolve:", "https://github.com/SakanaAI/ShinkaEvolve")
        print("Import error:", repr(e))
        print("Fallback: run `shinka_launch` directly (see baselines/shinkaevolve/run_evolve.sh).")
        return 2

    # NOTE: We keep config construction minimal and pass our evaluator-config via extra_cmd_args.
    evo_cfg = EvolutionConfig(
        init_program_path=str(initial_program),
        task_sys_msg=str(args.task_sys_msg),
        num_generations=int(args.num_generations),
        max_parallel_jobs=int(args.max_parallel_jobs),
    )
    # IMPORTANT: ShinkaEvolve computes/stores program embeddings via `evo_config.embedding_model`
    # (used for novelty + saved as Program.embedding). The DatabaseConfig.embedding_model field
    # is not what drives embedding computation in the core runner.
    if getattr(args, "embedding_model", ""):
        evo_cfg.embedding_model = str(args.embedding_model)
    # MIRROR OPENEvolve when requested: use its prompt.system_message and llm knobs.
    if args.openevolve_config:
        oe_path = Path(args.openevolve_config).expanduser().resolve()
        # Map OpenEvolve's `max_iterations` -> ShinkaEvolve `num_generations` (best-effort).
        # Only override when user didn't explicitly change the default.
        oe_max_iters = _load_openevolve_max_iterations(oe_path)
        if oe_max_iters is not None and int(args.num_generations) == 20:
            evo_cfg.num_generations = int(oe_max_iters)
        oe_sys = _load_openevolve_prompt_system_message(oe_path)
        if oe_sys:
            evo_cfg.task_sys_msg = _make_shinka_task_sys_msg(kernel_spec=oe_sys)
        oe_llm = _load_openevolve_llm_config(oe_path)
        if "models" in oe_llm and not args.llm_model:
            evo_cfg.llm_models = list(oe_llm["models"])
            if "model_sample_probs" in oe_llm:
                evo_cfg.llm_kwargs = dict(getattr(evo_cfg, "llm_kwargs", {}) or {})
                evo_cfg.llm_kwargs["model_sample_probs"] = list(oe_llm["model_sample_probs"])
        if "temperatures" in oe_llm or "max_tokens" in oe_llm:
            evo_cfg.llm_kwargs = dict(getattr(evo_cfg, "llm_kwargs", {}) or {})
            if "temperatures" in oe_llm:
                evo_cfg.llm_kwargs["temperatures"] = oe_llm["temperatures"]
            if "max_tokens" in oe_llm:
                evo_cfg.llm_kwargs["max_tokens"] = oe_llm["max_tokens"]

    # Default to Gemini to match the OpenEvolve MOE example config in this repo.
    # Users can still override by passing one or more --llm-model flags.
    evo_cfg.llm_models = list(args.llm_model) if args.llm_model else getattr(evo_cfg, "llm_models", ["gemini-3-pro-preview"])
    # For kernel-writing tasks, SEARCH/REPLACE diffs are brittle (one whitespace mismatch => 0 patches applied).
    # Prefer full rewrites where ShinkaEvolve will splice only EVOLVE-BLOCK payloads back into the file.
    patch_type = str(getattr(args, "patch_type", "full") or "full").strip()
    patch_types = [patch_type]
    patch_probs = [1.0]
    # Best-effort set: field names are stable in upstream, but keep it robust across versions.
    try:
        evo_cfg.patch_types = patch_types
        evo_cfg.patch_type_probs = patch_probs
    except Exception:
        if hasattr(evo_cfg, "patch_types"):
            try:
                setattr(evo_cfg, "patch_types", patch_types)
            except Exception:
                pass
        if hasattr(evo_cfg, "patch_type_probs"):
            try:
                setattr(evo_cfg, "patch_type_probs", patch_probs)
            except Exception:
                pass
    # Include evaluator text feedback (we populate it with trace logs), so the model can fix compile/runtime issues.
    evo_cfg.use_text_feedback = True

    job_cfg = LocalJobConfig(
        eval_program_path=str(eval_program),
        extra_cmd_args={
            "evaluator-config": str(evaluator_config),
            "num-runs": str(int(args.num_runs)),
        },
    )

    # Best-effort database config: prefer putting outputs under db_path if supported.
    db_kwargs: Dict[str, Any] = {}
    for k in ("db_path", "path", "root_dir", "root"):
        db_kwargs[k] = str(db_path)
    if getattr(args, "embedding_model", ""):
        db_kwargs["embedding_model"] = str(args.embedding_model)
    try:
        db_cfg = DatabaseConfig(**db_kwargs)  # type: ignore[arg-type]
    except Exception:
        db_cfg = DatabaseConfig()  # type: ignore[call-arg]
        # Best-effort fallback if the constructor didn't accept embedding_model.
        try:
            if getattr(args, "embedding_model", "") and hasattr(db_cfg, "embedding_model"):
                setattr(db_cfg, "embedding_model", str(args.embedding_model))
        except Exception:
            pass

    runner = EvolutionRunner(evo_config=evo_cfg, job_config=job_cfg, db_config=db_cfg)

    # W&B (optional)
    wb_run_name = args.wandb_run_name or f"shinkaevolve_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    wb = _wandb_init(enable=bool(args.enable_wandb), project=str(args.wandb_project or ""), run_name=str(wb_run_name))
    if wb is not None:
        try:
            wb.log(
                {
                    "status": "started",
                    "config/openevolve_config": str(args.openevolve_config or ""),
                    "config/evaluator_config": str(Path(args.evaluator_config).resolve()),
                    "config/num_generations": int(getattr(evo_cfg, "num_generations", 0)),
                    "config/max_parallel_jobs": int(getattr(evo_cfg, "max_parallel_jobs", 0)),
                }
            )
        except Exception:
            pass

    runner.run()

    # Resolve results directory for uploads and final eval.
    results_dir = Path(getattr(runner, "results_dir", "")).expanduser()
    results_dir = results_dir if results_dir.is_absolute() else (Path(os.getcwd()) / results_dir)

    # Upload key artifacts (best-effort).
    if wb is not None:
        try:
            _wandb_upload_file(
                wb_run=wb,
                file_path=results_dir / "evolution_run.log",
                artifact_name=f"{wb.name}-evolution_run",
                artifact_type="log",
            )
            _wandb_upload_file(
                wb_run=wb,
                file_path=results_dir / "experiment_config.yaml",
                artifact_name=f"{wb.name}-experiment_config",
                artifact_type="config",
            )
            # User-provided log files (e.g., shell-script tee output)
            for lf in (args.wandb_log_file or []):
                try:
                    _wandb_upload_file(
                        wb_run=wb,
                        file_path=Path(lf),
                        artifact_name=f"{wb.name}-log_{Path(lf).stem}",
                        artifact_type="log",
                    )
                except Exception:
                    continue
            # Best snapshot directory (if present)
            if (results_dir / "best").exists():
                _wandb_upload_dir(
                    wb_run=wb,
                    dir_path=(results_dir / "best"),
                    artifact_name=f"{wb.name}-best",
                    artifact_type="best",
                )
        except Exception:
            pass

    # Optional final evaluation pass (OpenEvolve parity): evaluate the best snapshot on all workloads.
    if bool(getattr(args, "final_eval_all_workloads", False)):
        try:
            best_program = results_dir / "best" / "main.py"
            if not best_program.exists():
                print("WARN: --final-eval-all-workloads set, but no best/main.py found at:", str(best_program))
            else:
                final_dir = results_dir / "final_eval_all_workloads"
                final_dir.mkdir(parents=True, exist_ok=True)
                base_cfg_path = Path(args.evaluator_config).expanduser().resolve()
                base_txt = base_cfg_path.read_text(encoding="utf-8", errors="replace")
                final_txt = _make_final_eval_all_workloads_config_text(base_txt)
                final_cfg_path = final_dir / "evaluator_config_all_workloads.yaml"
                final_cfg_path.write_text(final_txt, encoding="utf-8")

                cmd = [
                    sys.executable,
                    str(Path(args.eval_program).expanduser().resolve()),
                    "--program_path",
                    str(best_program),
                    "--results_dir",
                    str(final_dir),
                    "--evaluator-config",
                    str(final_cfg_path),
                    "--num-runs",
                    str(int(args.num_runs)),
                ]
                print("Running final eval (all workloads):", " ".join(cmd))
                proc = subprocess.run(cmd, check=False)
                print("Final eval exit code:", proc.returncode)

                # Also print an OpenEvolve-style per-workload table for the final best program.
                # This mirrors OpenEvolve's `bench_single_program_vs_flashinfer.py` output style
                # and is useful for quickly spotting outlier workloads.
                try:
                    bench_script = Path(__file__).with_name("bench_best_solution_vs_flashinfer.py").resolve()
                    if bench_script.exists():
                        bench_cmd = [
                            sys.executable,
                            "-u",
                            str(bench_script),
                            "--evaluator-config",
                            str(final_cfg_path),
                            "--program-path",
                            str(best_program),
                            # Avoid auto-pinning to a single GPU; keep current visibility.
                            "--cuda-device",
                            "keep",
                            # Avoid cross-job contention if CUDA_VISIBLE_DEVICES is already set;
                            # if it's not set, this will reserve all physical GPUs.
                            "--reserve-gpus",
                            "visible",
                        ]
                        print("Running final per-workload table:", " ".join(bench_cmd))
                        bench_out = subprocess.run(
                            bench_cmd,
                            check=False,
                            stdout=subprocess.PIPE,
                            stderr=subprocess.STDOUT,
                            text=True,
                        ).stdout
                        out_path = final_dir / "per_workload_table.txt"
                        out_path.write_text(bench_out or "", encoding="utf-8")
                        if bench_out:
                            print(bench_out.rstrip())
                        else:
                            print("WARN: per-workload bench produced no stdout; see:", str(out_path))
                    else:
                        print("WARN: bench_best_solution_vs_flashinfer.py not found at:", str(bench_script))
                except Exception as e:
                    print("WARN: final per-workload table failed:", repr(e))

                if wb is not None:
                    # Upload final eval directory
                    _wandb_upload_dir(
                        wb_run=wb,
                        dir_path=final_dir,
                        artifact_name=f"{wb.name}-final_eval_all_workloads",
                        artifact_type="eval",
                    )
        except Exception as e:
            print("WARN: final eval failed:", repr(e))

    if wb is not None:
        try:
            wb.log({"status": "finished"})
        except Exception:
            pass
        try:
            wb.finish()
        except Exception:
            pass
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

