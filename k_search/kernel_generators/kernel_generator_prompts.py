"""
This file contains the prompts for baseline agent generation.
"""

from __future__ import annotations

# CUDA-specific hints
# Note: keep these hints generic (avoid naming specific low-level instructions).
CUDA_OPTIMIZATION_HINTS = "** You MUST use MMA to utilize the tensor cores on H100! ** For each round, you can see your current best solution and the previous round's summary, therefore you can implement the kernel step by step."

# Triton-appropriate subset
TRITON_OPTIMIZATION_HINTS = """
** For each round, you can see your current best solution and the previous round's summary, therefore you can implement the kernel step by step.
"""


TRITON_PROMPT = """Generate a Triton kernel optimized for {target_gpu} GPU for

{definition}

Triton Version: 3.3.1

{per_task_requirement}

{hints}

Generate the implementation:"""

TRITON_OPTIMIZATION_PROMPT = """You are optimizing a Triton kernel for {target_gpu} GPU. The current implementation has issues that need to be fixed or its performance can be improved.

Original Specification:
{definition}

Current Implementation Status:
{trace_logs}

Current Implementation:
{current_code}

{per_task_requirement}

{hints}

{extra_context}
Generate the corrected and optimized implementation:"""


# CUDA prompt
CUDA_PROMPT = """You are a code generator. Generate a CUDA kernel implementation optimized for {target_gpu} GPU for the following specification.

Specification:
{definition}

{per_task_requirement}

{hints}

Generate the implementation:"""

CUDA_OPTIMIZATION_PROMPT = """You are optimizing a CUDA kernel for {target_gpu} GPU. The current implementation has issues that need to be fixed.

Original Specification:
{definition}

Current Implementation Status:
{trace_logs}

Current Implementation:
{current_code}

{per_task_requirement}

{hints}

{extra_context}

Generate the corrected and optimized implementation:"""


def get_prompt_from_definition_text(
    language: str,
    definition_text: str,
    target_gpu: str = "H100",
    *,
    per_task_requirement: str = "",
) -> str:
    """
    Task-agnostic prompt builder: takes a fully-rendered definition text.
    """
    prompts = {"triton": TRITON_PROMPT, "cuda": CUDA_PROMPT}

    if language not in prompts:
        raise ValueError(f"Unsupported language: {language}")

    # Only Triton/CUDA prompts include advanced hints
    if language == "triton":
        return prompts[language].format(
            definition=str(definition_text or "").strip(),
            target_gpu=target_gpu,
            per_task_requirement=str(per_task_requirement or "").strip(),
            hints=TRITON_OPTIMIZATION_HINTS,
        )
    if language == "cuda":
        return prompts[language].format(
            definition=str(definition_text or "").strip(),
            target_gpu=target_gpu,
            per_task_requirement=str(per_task_requirement or "").strip(),
            hints=CUDA_OPTIMIZATION_HINTS,
        )
    return prompts[language].format(definition=str(definition_text or "").strip(), target_gpu=target_gpu)


def get_optimization_prompt_from_definition_text(
    language: str,
    *,
    definition_text: str,
    trace_logs: str,
    current_code: str,
    target_gpu: str = "H100",
    current_best: str | None = None,
    previous_round_summary: str | None = None,
    per_task_requirement: str = "",
) -> str:
    """
    Task-agnostic optimization prompt builder: takes rendered definition + rendered trace logs.
    """
    optimization_prompts = {"triton": TRITON_OPTIMIZATION_PROMPT, "cuda": CUDA_OPTIMIZATION_PROMPT}

    if language not in optimization_prompts:
        raise ValueError(f"Unsupported language for optimization: {language}")

    extra_context = _build_extra_context(
        current_best=current_best,
        previous_round_summary=previous_round_summary,
    )

    if language == "triton":
        return optimization_prompts[language].format(
            definition=str(definition_text or "").strip(),
            trace_logs=str(trace_logs or "").strip(),
            current_code=current_code,
            target_gpu=target_gpu,
            per_task_requirement=str(per_task_requirement or "").strip(),
            hints=TRITON_OPTIMIZATION_HINTS,
            extra_context=extra_context,
        )
    if language == "cuda":
        return optimization_prompts[language].format(
            definition=str(definition_text or "").strip(),
            trace_logs=str(trace_logs or "").strip(),
            current_code=current_code,
            target_gpu=target_gpu,
            per_task_requirement=str(per_task_requirement or "").strip(),
            hints=CUDA_OPTIMIZATION_HINTS,
            extra_context=extra_context,
        )
    # Python doesn't use this path
    raise ValueError(f"No optimization prompt available for language: {language}")


def _build_extra_context(
    *,
    current_best: str | None,
    previous_round_summary: str | None,
) -> str:
    """
    Optional context that can be injected into optimization prompts.
    Returns an empty string when no context is available.
    """
    parts: list[str] = []

    # Put "best so far" last to avoid interrupting the flow of "what just happened" (prev round + profiling).
    if current_best and current_best.strip():
        parts.append("Current Best Solution So Far (best so far across rounds):\n" + current_best.strip())
      
    if previous_round_summary and previous_round_summary.strip():
        parts.append("Last Round Summary::\n" + previous_round_summary.strip())


    if not parts:
        return ""
    return "\n\n" + "\n\n".join(parts)
