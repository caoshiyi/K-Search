import random
import re
import tempfile
from pathlib import Path
from typing import Any, Dict, List, Optional
import os

from .kernel_generator_prompts import (
    get_optimization_prompt_from_definition_text,
    get_prompt_from_definition_text,
)
from .llm_clients import (
    LLMClient,
    LLMProviderFatalError,
    build_llm_client,
    llm_log_context,
    normalize_llm_provider,
)
from .ascendc_agentic_codegen import (
    AscendCAgenticCodegenRequest,
    AscendCAgenticCodegenResult,
    AscendCAgenticCodegenRunner,
)
from k_search.tasks.task_base import BuildSpec, EvalResult, Solution, SourceFile, SupportedLanguages
from k_search.tasks.task_base import Task, code_from_solution
from k_search.utils.paths import get_run_id

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
        reasoning_effort: str = "medium",  # only used for openai-compatible reasoning models
        llm_provider: str = "openai",
        llm_client: Optional[LLMClient] = None,
        stage_checkpoint_config: Any | None = None,
    ):
        """
        Args:
            model_name: Name of the model to use (e.g., "gpt-5")
            language: Programming language for code generation (default: "triton")
            target_gpu: Target GPU architecture (e.g., "H100", "B200", "RTX4090", default: "H100")
            api_key: API key (if None, uses LLM_API_KEY environment variable)
            base_url: Base URL for the API (need to provide for non-openai api models)
            reasoning_effort: Reasoning effort for OpenAI reasoning models ("low", "medium", "high", default: "medium")
            llm_provider: LLM backend to use ("openai" or "claude-agent")
            llm_client: Optional prebuilt client for tests/custom integrations
        """
        self.model_name = model_name
        self.language = language
        self.target_gpu = target_gpu
        self.reasoning_effort = reasoning_effort
        self.llm_provider = normalize_llm_provider(llm_provider)
        self._stage_checkpoint_config = stage_checkpoint_config
        self.llm_client = llm_client or build_llm_client(
            llm_provider=self.llm_provider,
            model_name=self.model_name,
            api_key=api_key,
            base_url=base_url,
            reasoning_effort=self.reasoning_effort,
        )

    def _get_supported_language(self) -> SupportedLanguages:
        language_map = {
            "python": SupportedLanguages.PYTHON,
            "triton": SupportedLanguages.TRITON,
            "cuda": SupportedLanguages.CUDA,
            "ascendc": SupportedLanguages.ASCENDC,
            "mlx": SupportedLanguages.MLX,
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

    def _generate_code_from_prompt(self, prompt: str, task: Optional[Task] = None):
        # Retry parse failures up to 5 times for languages with structured multi-file responses.
        max_parse_retries = 5
        lang = (self.language or "").lower()
        is_cuda = lang == "cuda"
        is_ascendc = lang == "ascendc"
        should_retry = is_cuda or is_ascendc

        last_err: Exception | None = None
        max_attempts = max_parse_retries if should_retry else 1
        prompt_chars = 0
        prompt_lines = 0
        retry_notes: list[str] = []

        def _line_count(text: str) -> int:
            return (text.count("\n") + 1) if text else 0

        def _short_error(e: Exception, limit: int = 1600) -> str:
            text = str(e).strip()
            if len(text) <= limit:
                return text
            return text[: limit - 3].rstrip() + "..."

        def _with_retry_feedback(base_prompt: str) -> str:
            if not retry_notes:
                return base_prompt
            feedback = "\n".join(f"- {note}" for note in retry_notes[-3:])
            prompt_with_feedback = (
                base_prompt
                + "\n\nPrevious code generation attempts failed before evaluation. "
                "Use this feedback to produce a fresh response that exactly matches the required format:\n"
                + feedback
            )
            if task is not None:
                fmt_hook = getattr(task, "get_code_format_text", None)
                if callable(fmt_hook):
                    try:
                        current_format = str(
                            fmt_hook(language=str(self.language), target_gpu=str(self.target_gpu)) or ""
                        ).strip()
                    except Exception:
                        current_format = ""
                    if current_format:
                        prompt_with_feedback += (
                            "\n\nCurrent required response format for this retry:\n"
                            + current_format
                        )
            return prompt_with_feedback

        for attempt in range(1, max_attempts + 1):
            try:
                effective_prompt = _with_retry_feedback(prompt)
                prompt_text = str(effective_prompt or "")
                prompt_chars = len(prompt_text)
                prompt_lines = _line_count(prompt_text)
                print(
                    f"[LLM] codegen request provider={self.llm_provider} model={self.model_name} "
                    f"language={lang or self.language} attempt={attempt}/{max_attempts} "
                    f"prompt_chars={prompt_chars} prompt_lines={prompt_lines}",
                    flush=True,
                )
                with llm_log_context(
                    operator=(getattr(task, "name", None) if task is not None else None),
                    phase="codegen",
                    language=str(lang or self.language),
                    target_gpu=str(self.target_gpu),
                    attempt=attempt,
                    max_attempts=max_attempts,
                ):
                    generated_code = str(self.llm_client.generate(effective_prompt) or "").strip()
                print(
                    f"[LLM] codegen response provider={self.llm_provider} model={self.model_name} "
                    f"language={lang or self.language} attempt={attempt}/{max_attempts} "
                    f"raw_chars={len(generated_code)} raw_lines={_line_count(generated_code)}",
                    flush=True,
                )

                cleaned_code = self._clean_generated_code(generated_code)

                if is_cuda:
                    # cleaned_code should be a dict of required files for CUDA.
                    if not isinstance(cleaned_code, dict):
                        raise ValueError("CUDA generation did not return a parsed file dict")
                    required = ("kernel.h", "kernel.cu", "main.cpp")
                    missing = [k for k in required if (k not in cleaned_code) or (not str(cleaned_code.get(k, "")).strip())]
                    if missing:
                        raise ValueError(f"missing required XML files: {missing}")
                elif is_ascendc and task is not None:
                    preview = getattr(task, "preview_parse_generated_code", None)
                    if callable(preview):
                        preview(raw_code=generated_code)

                print(
                    f"[LLM] codegen parse ok language={lang or self.language} "
                    f"attempt={attempt}/{max_attempts} cleaned_type={type(cleaned_code).__name__}",
                    flush=True,
                )
                return {"raw": generated_code, "cleaned": cleaned_code}

            except Exception as e:
                last_err = e
                err_type = type(e).__name__
                if isinstance(e, LLMProviderFatalError):
                    print(
                        f"[ERROR] LLM provider fatal error language={lang or self.language} "
                        f"attempt={attempt}/{max_attempts} prompt_chars={prompt_chars} "
                        f"prompt_lines={prompt_lines} error_type={err_type}: {e}",
                        flush=True,
                    )
                    raise
                if isinstance(e, TimeoutError):
                    if should_retry and attempt < max_attempts:
                        retry_notes.append(
                            f"attempt {attempt} timed out: {_short_error(e)}"
                        )
                        print(
                            f"[WARN] LLM codegen timeout language={lang or self.language} "
                            f"attempt={attempt}/{max_attempts} prompt_chars={prompt_chars} "
                            f"prompt_lines={prompt_lines} error_type={err_type}: {e}; "
                            f"retrying generation ({attempt}/{max_attempts})...",
                            flush=True,
                        )
                        continue
                    else:
                        print(
                            f"[ERROR] LLM codegen timeout language={lang or self.language} "
                            f"attempt={attempt}/{max_attempts} prompt_chars={prompt_chars} "
                            f"prompt_lines={prompt_lines} error_type={err_type}: {e}",
                            flush=True,
                        )
                        raise
                if should_retry and attempt < max_attempts:
                    retry_notes.append(
                        f"attempt {attempt} failed with {err_type}: {_short_error(e)}"
                    )
                    tag = "CUDA XML" if is_cuda else "AscendC"
                    print(
                        f"[WARN] {tag} parse failed error_type={err_type} "
                        f"attempt={attempt}/{max_attempts} prompt_chars={prompt_chars} "
                        f"prompt_lines={prompt_lines} ({e}); retrying generation ({attempt}/{max_parse_retries})...",
                        flush=True,
                    )
                    continue
                print(f"Error while generating code: error_type={err_type}: {e}", flush=True)
                raise

        assert last_err is not None
        raise last_err

    def _create_solution_from_code(
        self,
        *,
        cleaned_code: Any,
        raw_code: Any,
        task: Task,
        round_num: int,
    ) -> AscendCAgenticCodegenResult:
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

    def _should_use_ascendc_agentic_codegen(self, task: Any) -> bool:
        if self.llm_provider != "claude-agent":
            return False
        if str(self.language or "").strip().lower() != "ascendc":
            return False
        if os.getenv("KSEARCH_DISABLE_ASCENDC_AGENTIC_CODEGEN", "").strip().lower() in {
            "1",
            "true",
            "yes",
            "on",
        }:
            return False
        return hasattr(task, "make_solution_from_project_dir")

    def _allow_ascendc_agentic_legacy_fallback(self) -> bool:
        return os.getenv("KSEARCH_ASCENDC_AGENTIC_FALLBACK", "").strip().lower() == "legacy"

    def _agentic_runner(self) -> AscendCAgenticCodegenRunner:
        runner = getattr(self, "_ascendc_agentic_runner", None)
        if runner is None:
            runner = AscendCAgenticCodegenRunner(
                model_name=str(self.model_name),
                stage_checkpoint_config=getattr(self, "_stage_checkpoint_config", None),
            )
            self._ascendc_agentic_runner = runner
        return runner

    def _generate_ascendc_solution_agentically(
        self,
        *,
        task: Any,
        action_text: str,
        trace_logs: str,
        perf_summary: str,
        round_num: int,
        attempt_idx: int,
        mode: str,
        base_solution: Optional[Solution],
        max_fix_rounds: int | None = None,
    ) -> Any:
        definition_hook = getattr(task, "get_agentic_definition_text", None)
        if callable(definition_hook):
            definition_text = str(definition_hook(language=str(self.language)) or "").strip()
        else:
            definition_text = str(task.get_definition_text(language=str(self.language)) or "").strip()
        run_id = str(
            getattr(task, "_ksearch_run_id", None)
            or getattr(self, "_ksearch_run_id", None)
            or get_run_id()
        )
        task_name = str(
            getattr(task, "name", "")
            or getattr(task, "definition_name", "")
            or "ascendc"
        ).strip() or "ascendc"
        request = AscendCAgenticCodegenRequest(
            definition_text=definition_text,
            action_text=str(action_text or "").strip(),
            trace_logs=str(trace_logs or "").strip(),
            perf_summary=str(perf_summary or "").strip(),
            target_gpu=str(self.target_gpu),
            round_num=int(round_num),
            attempt_idx=int(attempt_idx),
            mode=str(mode),  # type: ignore[arg-type]
            run_id=run_id,
            task_name=task_name,
        )
        result = self._agentic_runner().run_one_shot_closed(
            task=task,
            request=request,
            base_solution=base_solution,
            max_fix_rounds=int(max_fix_rounds or 0),
        )
        print(
            f"[LLM] agentic ascendc result provider={self.llm_provider} model={self.model_name} "
            f"round={round_num} prompt_chars={result.prompt_chars} "
            f"changed_files={','.join(result.changed_paths)} project_path={result.project_path}",
            flush=True,
        )
        return result

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
        pending_agentic_eval: EvalResult | None = None
        pending_agentic_solution: Solution | None = None
        pending_agentic_result: Any | None = None

        # Seed initial code: continue from existing solution if provided; else generate fresh.
        seed_solution: Optional[Solution] = None
        seed_agentic_result: Any | None = None
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
            if self._should_use_ascendc_agentic_codegen(task):
                try:
                    agentic_result = self._generate_ascendc_solution_agentically(
                        task=task,
                        action_text=(
                            "Create an optimized AscendC candidate from the current project files. "
                            "Keep public interfaces and harness behavior unchanged."
                        ),
                        trace_logs="",
                        perf_summary="",
                        round_num=0,
                        attempt_idx=1,
                        mode="generate",
                        base_solution=None,
                    )
                    solution = agentic_result.solution
                    pending_agentic_eval = agentic_result.eval_result
                    current_code, current_raw_code = code_from_solution(self.language, solution)
                    seed_solution = solution
                    seed_agentic_result = agentic_result
                except Exception as exc:
                    if not self._allow_ascendc_agentic_legacy_fallback():
                        raise
                    print(
                        f"[WARN] agentic AscendC seed codegen failed; falling back to legacy prompt path: "
                        f"{type(exc).__name__}: {exc}",
                        flush=True,
                    )
            if seed_solution is None and current_raw_code is None:
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
                with llm_log_context(
                    operator=str(getattr(task, "name", "") or ""),
                    flow="baseline",
                    round_index=0,
                    stage="initial_codegen",
                    max_rounds=max_opt_rounds,
                    language=str(self.language),
                    target_gpu=str(self.target_gpu),
                ):
                    code_result = self._generate_code_from_prompt(prompt, task=task)
                current_code = code_result["cleaned"]
                current_raw_code = code_result["raw"]

        best_solution: Optional[Solution] = None
        best_eval = None
        best_score: float = -1.0
        best_raw_code: Optional[str] = None

        for round_num in range(1, max_opt_rounds + 1):
            print(f"\n=== Optimization Round {round_num}/{max_opt_rounds} ===")

            # Use the provided seed solution on the first round if available
            round_agentic_result: Any | None = None
            if pending_agentic_solution is not None:
                solution = pending_agentic_solution
                round_agentic_result = pending_agentic_result
                pending_agentic_solution = None
                pending_agentic_result = None
            elif round_num == 1 and seed_solution is not None:
                solution = seed_solution
                round_agentic_result = seed_agentic_result
                seed_agentic_result = None
            else:
                solution = self._create_solution_from_code(
                    cleaned_code=current_code,
                    raw_code=current_raw_code,
                    task=task,
                    round_num=int(round_num),
                )

            print("Evaluating solution...")
            if pending_agentic_eval is not None:
                eval_result = pending_agentic_eval
                pending_agentic_eval = None
            else:
                eval_result = task.run_benchmark(solution=solution, dump_traces=False, round_num=int(round_num))
            all_passed = bool(getattr(eval_result, "is_passed", lambda: False)())
            round_score = float(getattr(eval_result, "score", lambda: -1.0)())

            if all_passed and round_score > best_score:
                best_solution = solution
                best_eval = eval_result
                best_score = float(round_score)
                best_raw_code = str(current_raw_code or "")
                if round_agentic_result is not None:
                    from k_search.kernel_generators.memory import save_code_map_if_adopted, save_knowledge_if_adopted
                    save_code_map_if_adopted(
                        task=task,
                        code_map_text=getattr(round_agentic_result, "code_map_text", None),
                        adopted=True,
                        solution_id=solution.hash() if hasattr(solution, "hash") else None,
                        candidate_id=(
                            getattr(getattr(round_agentic_result, "candidate_patch", None), "candidate_id", None)
                        ),
                        action_node_id=(
                            getattr(getattr(round_agentic_result, "candidate_patch", None), "action_node_id", None)
                        ),
                        eval_status=str(getattr(eval_result, "status", "") or ""),
                        speedup_vs_parent=(
                            getattr(eval_result, "metrics", {}).get("speedup_vs_parent")
                            if isinstance(getattr(eval_result, "metrics", None), dict)
                            else None
                        ),
                        created_round=int(round_num),
                    )
                    save_knowledge_if_adopted(
                        task=task,
                        knowledge_text=getattr(round_agentic_result, "knowledge_text", None),
                        adopted=True,
                    )

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
                # --- Agentic AscendC optimization branch ---
                if self._should_use_ascendc_agentic_codegen(task):
                    base_for_agentic = solution
                    action_text = (
                        "Improve the current AscendC candidate. If the last attempt failed, fix compile, "
                        "runtime, or correctness first. If it passed, reduce measured latency while preserving semantics."
                    )
                    try:
                        agentic_result = self._generate_ascendc_solution_agentically(
                            task=task,
                            action_text=action_text,
                            trace_logs=str(trace_logs or ""),
                            perf_summary=str(previous_round_summary_for_prompt or ""),
                            round_num=round_num + 1,
                            attempt_idx=1,
                            mode="improve",
                            base_solution=base_for_agentic,
                            max_fix_rounds=int(os.getenv("KSEARCH_AGENTIC_MAX_FIX_ROUNDS", "3")),
                        )
                        solution = agentic_result.solution
                        pending_agentic_eval = agentic_result.eval_result
                        pending_agentic_solution = solution
                        pending_agentic_result = agentic_result
                        current_code, current_raw_code = code_from_solution(self.language, solution)
                        continue
                    except Exception as exc:
                        if not self._allow_ascendc_agentic_legacy_fallback():
                            raise
                        print(
                            f"[WARN] agentic AscendC optimization codegen failed; "
                            f"falling back to legacy prompt path: {type(exc).__name__}: {exc}",
                            flush=True,
                        )
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
                with llm_log_context(
                    operator=str(getattr(task, "name", "") or ""),
                    flow="baseline",
                    round_index=round_num + 1,
                    stage="optimization_codegen",
                    max_rounds=max_opt_rounds,
                    language=str(self.language),
                    target_gpu=str(self.target_gpu),
                ):
                    code_result = self._generate_code_from_prompt(opt_prompt, task=task)
                current_code = code_result["cleaned"]
                current_raw_code = code_result["raw"]

        return best_solution or solution
