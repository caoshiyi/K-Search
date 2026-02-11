import random
import re
import tempfile
from pathlib import Path
from typing import Any, Dict, List, Optional
import os

import openai
from .kernel_generator_prompts import (
    get_optimization_prompt_from_definition_text,
    get_prompt_from_definition_text,
)
from k_search.tasks.task_base import BuildSpec, Solution, SourceFile, SupportedLanguages
from k_search.tasks.task_base import Task, code_from_solution

# Optional Weights & Biases support
try:
    import wandb  # type: ignore
except Exception:  # pragma: no cover
    wandb = None

def get_code_from_solution(language: str, solution: Any):
    from k_search.tasks.task_base import code_from_solution
    return code_from_solution(language, solution)


class KernelGenerator:
    def __init__(
        self,
        model_name: str,
        language: str = "triton",
        target_gpu: str = "H100",
        api_key: Optional[str] = None,
        base_url: Optional[str] = None,
        reasoning_effort: str = "medium",  # only used for openai reasoning models
    ):
        """
        Args:
            model_name: Name of the model to use (e.g., "gpt-5")
            language: Programming language for code generation (default: "triton")
            target_gpu: Target GPU architecture (e.g., "H100", "B200", "RTX4090", default: "H100")
            api_key: API key (if None, uses LLM_API_KEY environment variable)
            base_url: Base URL for the API (need to provide for non-openai api models)
            reasoning_effort: Reasoning effort for OpenAI reasoning models ("low", "medium", "high", default: "medium")
        """
        self.model_name = model_name
        self.language = language
        self.target_gpu = target_gpu
        self.reasoning_effort = reasoning_effort

        if api_key is None:
            api_key = os.getenv("LLM_API_KEY")
            if api_key is None:
                raise ValueError(
                    "API key must be provided or set in LLM_API_KEY environment variable"
                )

        client_kwargs = {"api_key": api_key}
        if base_url is not None:
            client_kwargs["base_url"] = base_url

        self.client = openai.OpenAI(**client_kwargs)

    def _get_supported_language(self) -> SupportedLanguages:
        language_map = {
            "python": SupportedLanguages.PYTHON,
            "triton": SupportedLanguages.TRITON,
            "cuda": SupportedLanguages.CUDA,
        }
        if self.language.lower() in language_map:
            return language_map[self.language.lower()]
        else:
            # Default Python
            return SupportedLanguages.PYTHON

    # NOTE: baseline-aware generation lives on `KernelGenerator.generate(task=..., ...)`.

    def _parse_xml_files(self, code: str) -> Dict[str, str]:
        files = {}

        patterns = {
            "kernel.h": r'<header_file name="kernel\.h">(.*?)</header_file>',
            "kernel.cu": r'<cuda_file name="kernel\.cu">(.*?)</cuda_file>',
            "main.cpp": r'<cpp_file name="main\.cpp">(.*?)</cpp_file>',
        }

        for filename, pattern in patterns.items():
            match = re.search(pattern, code, re.DOTALL)
            if match:
                content = match.group(1).strip()
                files[filename] = content
            else:
                print(f"Warning: Could not find {filename} in generated code")

        return files

    def _clean_generated_code(self, code: str) -> str:
        """Clean up generated code. For CUDA, parse XML and return dict. For others, clean Python syntax."""
        if self.language.lower() == "cuda":
            return self._parse_xml_files(code)

        # For non-CUDA languages (triton, python), clean up markdown and hex floats
        if "```" in code:
            # Prefer parsing the first fenced block anywhere in the response. This mirrors the
            # CUDA path's "structured output" parsing and is robust to models emitting extra text.
            m = re.search(r"```[a-zA-Z0-9_+-]*\n([\s\S]*?)\n```", str(code or ""))
            if m:
                code = (m.group(1) or "").strip()
            else:
                # Back-compat fallback: strip leading/trailing fences if present, then remove stray backticks.
                if code.startswith("```"):
                    lines = code.split("\n")
                    if lines[0].startswith("```"):
                        lines = lines[1:]
                    code = "\n".join(lines)

                if code.endswith("```"):
                    lines = code.split("\n")
                    if lines and lines[-1].strip() == "```":
                        lines = lines[:-1]
                    code = "\n".join(lines)

                code = code.replace("```", "")

        hex_float_pattern = r"0x[0-9a-fA-F]*\.[0-9a-fA-F]*p[-+]?\d+"
        hex_floats = re.findall(hex_float_pattern, code)

        for hex_float in hex_floats:
            try:
                if hex_float == "0x1.62e42fefa39efp-1":
                    decimal_val = "0.6931471805599453"
                elif hex_float == "0x1.71547652b82fep0":
                    decimal_val = "2.718281828459045"
                elif hex_float == "0x1.921fb54442d18p1":
                    decimal_val = "3.141592653589793"
                else:
                    decimal_val = "1.0"

                code = code.replace(hex_float, decimal_val)
            except Exception as e:
                print(f"Warning: Could not convert hex float {hex_float}: {e}")
                code = code.replace(hex_float, "1.0")

        return code

    def _generate_code_from_prompt(self, prompt: str):
        # If we fail to parse CUDA XML (missing kernel.h/kernel.cu/main.cpp), retry generation.
        max_parse_retries = 5
        is_cuda = (self.language or "").lower() == "cuda"

        last_err: Exception | None = None
        for attempt in range(1, (max_parse_retries if is_cuda else 1) + 1):
            try:
                effective_prompt = prompt

                if self.model_name.startswith("gpt-5") or self.model_name.startswith("o3"):
                    response = self.client.responses.create(
                        model=self.model_name, input=effective_prompt, reasoning={"effort": self.reasoning_effort}
                    )
                    generated_code = response.output_text.strip()
                else:  # We use the completions api for OpenAI SDK compatible models
                    response = self.client.chat.completions.create(
                        model=self.model_name, messages=[{"role": "user", "content": effective_prompt}]
                    )
                    generated_code = response.choices[0].message.content.strip()

                cleaned_code = self._clean_generated_code(generated_code)

                if is_cuda:
                    # cleaned_code should be a dict of required files for CUDA.
                    if not isinstance(cleaned_code, dict):
                        raise ValueError("CUDA generation did not return a parsed file dict")
                    required = ("kernel.h", "kernel.cu", "main.cpp")
                    missing = [k for k in required if (k not in cleaned_code) or (not str(cleaned_code.get(k, "")).strip())]
                    if missing:
                        raise ValueError(f"missing required XML files: {missing}")

                return {"raw": generated_code, "cleaned": cleaned_code}

            except Exception as e:
                last_err = e
                if is_cuda and attempt < max_parse_retries:
                    print(f"[WARN] CUDA XML parse failed ({e}); retrying generation ({attempt}/{max_parse_retries})...")
                    continue
                print(f"Error while generating code: {e}")
                raise

        # Unreachable, but keeps type-checkers happy.
        assert last_err is not None
        raise last_err

    def _create_solution_from_code(
        self,
        *,
        cleaned_code: Any,
        raw_code: Any,
        task: Task,
        round_num: int,
    ) -> Solution:
        """
        Create a k-search `Solution` from generated code.

        Tasks may override via an optional hook:
          - make_solution_from_generated_code(cleaned_code=..., raw_code=..., round_num=..., model_name=..., target_gpu=..., language=...)
        """
        hook = getattr(task, "make_solution_from_generated_code", None)
        if callable(hook):
            return hook(
                cleaned_code=cleaned_code,
                raw_code=raw_code,
                round_num=int(round_num),
                model_name=str(self.model_name),
                target_gpu=str(self.target_gpu),
                language=str(self.language),
            )

        def_name = str(getattr(task, "name", "") or "").strip() or "__unknown__"

        # Include reasoning effort in name and description for GPT-5 models
        if self.model_name.startswith("gpt-5") or self.model_name.startswith("o3"):
            solution_name = f"{self.model_name}_{def_name}_{self.language}_optimized_r{round_num}_{self.reasoning_effort}"
            solution_description = f"{self.model_name} optimized kernel for {def_name} (round {round_num}, reasoning effort: {self.reasoning_effort})"
        else:
            solution_name = (
                f"{self.model_name}_{def_name}_{self.language}_optimized_r{round_num}"
            )
            solution_description = (
                f"{self.model_name} optimized kernel for {def_name} (round {round_num})"
            )

        # Handle different code formats based on language
        if self.language.lower() == "cuda" and isinstance(cleaned_code, dict):
            # For CUDA, we have multiple files
            sources = []
            for filename, content in cleaned_code.items():
                sources.append(SourceFile(path=filename, content=content))

            entry_point = "main.cpp::run"
        else:
            # For single-file languages (triton, python)
            code_txt = raw_code if isinstance(raw_code, str) and raw_code.strip() else cleaned_code
            if isinstance(code_txt, dict):
                code_txt = next(iter(code_txt.values()))
            sources = [SourceFile(path="main.py", content=str(code_txt or ""))]
            entry_point = "main.py::run"

        solution = Solution(
            name=solution_name,
            definition=def_name,
            author=self.model_name,
            spec=BuildSpec(
                language=self._get_supported_language(),
                target_hardware=[str(self.target_gpu or "H100")],
                entry_point=entry_point,
            ),
            sources=sources,
            description=solution_description,
        )
        return solution
    
    def generate(  # type: ignore[override]
        self,
        task: Task,
        max_opt_rounds: int = 10,
        *,
        continue_from_solution: Optional[str] = None,
    ) -> Solution:
        """
        Generate an optimized solution using baseline performance as the reference target.
        - Precompute baseline latencies before codegen
        - Keep optimizing for max_opt_rounds even if PASSED
        - Use multiple workloads' feedback per round
        """
        get_def = getattr(task, "get_definition_text", None)
        if callable(get_def):
            definition_text = str(get_def(language=str(self.language)) or "").strip()
            if not definition_text:
                raise RuntimeError(
                    f"Task '{getattr(task, 'name', '')}' returned empty definition text; "
                    "cannot build prompts without a definition."
                )
        else:
            raise RuntimeError(
                f"Task '{getattr(task, 'name', '')}' does not provide get_definition_text(); "
                "cannot build prompts without a definition."
            )
        baseline_targets_text = str(getattr(task, "get_baseline_targets_text", lambda: "")() or "").strip()

        def _append_baseline_hint(p: str) -> str:
            if not baseline_targets_text:
                return p
            return (
                p
                + "\n\nPerformance targets (lower is better):\n"
                + baseline_targets_text
                + "\n- Optimize for overall mean latency across the listed workloads while maintaining correctness."
            )

        def _per_task_requirement_text(*, phase: str) -> str:
            hook = getattr(task, "get_per_task_requirement_text", None)
            if callable(hook):
                try:
                    return str(
                        hook(language=str(self.language), target_gpu=str(self.target_gpu), phase=str(phase or ""))
                        or ""
                    ).strip()
                except Exception:
                    return ""
            return ""

        current_code = None
        current_raw_code = None

        # Seed initial code: continue from existing solution if provided; else generate fresh.
        seed_solution: Optional[Solution] = None
        if continue_from_solution:
            base_sol = task.get_solution(continue_from_solution)
            if base_sol is None:
                raise ValueError(f"Solution '{continue_from_solution}' not found")
            if base_sol.definition != task.name:
                raise ValueError(
                    f"Solution '{continue_from_solution}' does not belong to definition '{task.name}'"
                )
            seed_solution = base_sol
            current_code, current_raw_code = code_from_solution(self.language, base_sol)
        else:
            gen_prompt_fn = getattr(task, "get_generation_prompt", None)
            if callable(gen_prompt_fn):
                prompt = str(gen_prompt_fn(language=str(self.language), target_gpu=str(self.target_gpu)) or "")
            else:
                per_req = _per_task_requirement_text(phase="generate")
                prompt = get_prompt_from_definition_text(
                    self.language,
                    definition_text,
                    self.target_gpu,
                    per_task_requirement=per_req,
                )
            prompt = _append_baseline_hint(prompt)
            print(prompt)
            code_result = self._generate_code_from_prompt(prompt)
            current_code = code_result["cleaned"]
            current_raw_code = code_result["raw"]

        best_solution: Optional[Solution] = None
        best_eval = None
        best_score: float = -1.0
        best_raw_code: Optional[str] = None

        for round_num in range(1, max_opt_rounds + 1):
            print(f"\n=== Optimization Round {round_num}/{max_opt_rounds} ===")

            # Use the provided seed solution on the first round if available
            if round_num == 1 and seed_solution is not None:
                solution = seed_solution
            else:
                solution = self._create_solution_from_code(
                    cleaned_code=current_code,
                    raw_code=current_raw_code,
                    task=task,
                    round_num=int(round_num),
                )

            print("Evaluating solution...")
            eval_result = task.run_benchmark(solution=solution, dump_traces=False, round_num=int(round_num))
            all_passed = bool(getattr(eval_result, "is_passed", lambda: False)())
            round_score = float(getattr(eval_result, "score", lambda: -1.0)())

            if all_passed and round_score > best_score:
                best_solution = solution
                best_eval = eval_result
                best_score = float(round_score)
                best_raw_code = str(current_raw_code or "")

            # If all workloads passed in this round, log a W&B artifact containing the generated code for traceability.
            if all_passed and wandb is not None and getattr(wandb, "run", None) is not None:
                try:
                    # Truncate artifact name to satisfy WandB limit.
                    safe_def = str(task.name or "")[:32]
                    safe_sol = str(solution.name or "")[-32:]
                    art_name = f"{safe_def}_r{round_num}_{safe_sol}_code"
                    if len(art_name) > 128:
                        import hashlib

                        h = hashlib.md5(art_name.encode()).hexdigest()[:8]
                        art_name = f"{safe_def}_r{round_num}_{h}_code"

                    artifact = wandb.Artifact(
                        name=art_name,
                        type="generated-code",
                        metadata={
                            "definition": str(task.name or ""),
                            "round": int(round_num),
                            "solution": str(solution.name or ""),
                            "language": str(self.language or ""),
                            "target_gpu": str(self.target_gpu or ""),
                        },
                    )
                    with tempfile.TemporaryDirectory() as tmpdir:
                        tmpdir_p = Path(tmpdir)
                        # Cleaned code files
                        if isinstance(current_code, dict):
                            for filename, content in current_code.items():
                                p = tmpdir_p / str(filename)
                                p.parent.mkdir(parents=True, exist_ok=True)
                                p.write_text(str(content or ""))
                                artifact.add_file(str(p), name=f"clean/{filename}")
                        else:
                            p = tmpdir_p / "main.py"
                            p.parent.mkdir(parents=True, exist_ok=True)
                            p.write_text(str(current_code or ""))
                            artifact.add_file(str(p), name="clean/main.py")

                        # Raw code (as generated from the LLM before cleaning)
                        raw_path = tmpdir_p / "raw_code.txt"
                        raw_path.write_text(str(current_raw_code) if current_raw_code is not None else "")
                        artifact.add_file(str(raw_path), name="raw/raw_code.txt")

                        # Round-level eval summary (best-effort)
                        summary_path = tmpdir_p / "round_summary.txt"
                        try:
                            summary_path.write_text(
                                "\n".join(eval_result.perf_summary_lines(prefix="round")) + "\n"
                            )
                        except Exception:
                            summary_path.write_text("")
                        artifact.add_file(str(summary_path), name="eval/round_summary.txt")

                    wandb.log_artifact(artifact)
                except Exception:
                    pass

            # W&B: log best-so-far only
            if wandb is not None and getattr(wandb, "run", None) is not None:
                try:
                    # Also log this round's score (even if not passed; store None for clarity).
                    try:
                        round_score_name = None
                        try:
                            round_score_name = (
                                eval_result.metrics.get("score_name")
                                if isinstance(getattr(eval_result, "metrics", None), dict)
                                else None
                            )
                        except Exception:
                            round_score_name = None
                        round_key = (
                            f"{task.name}/generate/{round_score_name}"
                            if isinstance(round_score_name, str) and round_score_name
                            else f"{task.name}/generate/round_score"
                        )
                        wandb.log(
                            {round_key: (float(round_score) if all_passed and round_score > 0 else None)},
                            step=round_num,
                        )
                    except Exception:
                        pass

                    score_name = None
                    score_val = None
                    if best_eval is not None:
                        score_name = (
                            best_eval.metrics.get("score_name")
                            if isinstance(getattr(best_eval, "metrics", None), dict)
                            else None
                        )
                        score_val = best_eval.score() if getattr(best_eval, "is_passed", lambda: False)() else None
                    key = (
                        f"{task.name}/generate/best_{score_name}"
                        if isinstance(score_name, str) and score_name
                        else f"{task.name}/generate/best_score"
                    )
                    wandb.log(
                        {key: (float(score_val) if isinstance(score_val, (int, float)) and float(score_val) > 0 else None)},
                        step=round_num,
                    )
                except Exception:
                    pass

            # Prepare next round prompt (even if PASSED)
            if round_num < max_opt_rounds:
                trace_logs = str(getattr(task, "get_last_round_trace_logs_for_prompt", lambda: "")() or "").strip()
                # Build optional extra context similar to the original baseline generator.
                current_best_for_prompt = None
                try:
                    cb_lines: List[str] = []
                    if best_score > 0:
                        try:
                            sn = (
                                best_eval.metrics.get("score_name")
                                if (best_eval is not None and isinstance(getattr(best_eval, "metrics", None), dict))
                                else None
                            )
                        except Exception:
                            sn = None
                        score_label = str(sn or "score")
                        cb_lines.append(f"Performance: {score_label} {best_score:.4f}")
                    if best_raw_code and best_raw_code.strip():
                        cb_lines.append("Code:\n" + str(best_raw_code).strip())
                    current_best_for_prompt = "\n".join(cb_lines).strip() if cb_lines else None
                except Exception:
                    current_best_for_prompt = None

                previous_round_summary_for_prompt = None
                try:
                    passed_count = int(getattr(task, "get_last_round_passed_count", lambda: 0)() or 0)
                    total_workloads = int(getattr(task, "get_last_round_total_workloads", lambda: 0)() or 0)
                    pr_lines: List[str] = []
                    if total_workloads > 0:
                        pr_lines.append(f"Passed {passed_count}/{total_workloads} workloads.")
                    if all_passed and round_score > 0:
                        try:
                            sn = (
                                eval_result.metrics.get("score_name")
                                if isinstance(getattr(eval_result, "metrics", None), dict)
                                else None
                            )
                        except Exception:
                            sn = None
                        score_label = str(sn or "score")
                        pr_lines.append(f"{score_label}: {round_score:.4f}.")
                    # Also include task-agnostic perf lines when available.
                    try:
                        pr_lines.extend([ln.lstrip("- ").strip() for ln in eval_result.perf_summary_lines(prefix="")])
                    except Exception:
                        pass
                    previous_round_summary_for_prompt = (
                        "\n".join(f"- {ln}" for ln in pr_lines).strip() if pr_lines else None
                    )
                except Exception:
                    previous_round_summary_for_prompt = None

                # Always use the optimization prompt template for the next round.
                # Tasks may provide empty trace logs on PASS; we still want to carry forward
                # previous round summary / best-so-far / current code context deterministically.
                opt_prompt_fn = getattr(task, "get_optimization_prompt", None)
                if callable(opt_prompt_fn):
                    opt_prompt = str(
                        opt_prompt_fn(
                            language=str(self.language),
                            target_gpu=str(self.target_gpu),
                            trace_logs=str(trace_logs or ""),
                            current_code=str(current_raw_code or ""),
                            current_best=current_best_for_prompt,
                            previous_round_summary=previous_round_summary_for_prompt,
                        )
                        or ""
                    )
                else:
                    per_req = _per_task_requirement_text(phase="optimize")
                    opt_prompt = get_optimization_prompt_from_definition_text(
                        self.language,
                        definition_text=definition_text,
                        trace_logs=str(trace_logs or ""),
                        current_code=str(current_raw_code or ""),
                        target_gpu=self.target_gpu,
                        current_best=current_best_for_prompt,
                        previous_round_summary=previous_round_summary_for_prompt,
                        per_task_requirement=per_req,
                    )
                opt_prompt = _append_baseline_hint(opt_prompt)
                print(opt_prompt)
                print(f"Generating optimized code for round {round_num + 1}...")
                code_result = self._generate_code_from_prompt(opt_prompt)
                current_code = code_result["cleaned"]
                current_raw_code = code_result["raw"]

        return best_solution or solution