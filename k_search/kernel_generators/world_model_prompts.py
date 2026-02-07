"""
World-model codegen prompts (action application + debugging).

These prompts are WM-specific and intentionally kept separate from the baseline
kernel generator prompts in `kernel_generator_prompts.py`.
"""

from __future__ import annotations

from .kernel_generator_prompts import (
    CUDA_OPTIMIZATION_HINTS,
    TRITON_OPTIMIZATION_HINTS,
)


TRITON_ACTION_PROMPT = """You are implementing a SPECIFIC NEXT ACTION on top of a known-good Triton baseline for {target_gpu}.

Original Specification:
{definition}

Known-Good Base Implementation (start from this; do not include any other previous code):
{base_code}

Chosen Next Action (apply this action):
{action_text}

The wrapper function MUST handle complete device management:
- Move CPU tensors to GPU if needed (use .cuda() when torch.cuda.is_available())
- Raise clear errors if CUDA is not available for GPU tensors
- Call the triton kernel with GPU tensors
- Move results back to original device of input tensors
- Handle both args and kwargs properly
- Preserve original tensor devices and restore them for outputs

IMPORTANT: Use only valid Python/Triton syntax:
- NO hexadecimal float literals (0x1.234p5) - use decimal equivalents
- NO C/CUDA specific syntax - this is Python/Triton code
- Use math.log(2), math.pi, math.e instead of hex literals
- All code must be valid Python that passes ast.parse()

You MUST expose a "run" entry point function that can be called to execute the kernel.

Rules:
- Implement ONLY the chosen action; keep everything else as close as possible to the base implementation.
- Keep changes small and single-iteration implementable.
- Preserve correctness and the function signature / wrapper behavior.
- Return only the full updated code (no explanations, no markdown).

{hints}

Generate the updated implementation:"""


CUDA_ACTION_PROMPT = """You are implementing a SPECIFIC NEXT ACTION on top of a known-good CUDA baseline for {target_gpu}.

Original Specification:
{definition}

Known-Good Base Implementation (start from this; do not include any other previous code):
{base_code}

Chosen Next Action (apply this action):
{action_text}

Rules:
- Implement ONLY the chosen action; keep everything else as close as possible to the base implementation.
- Keep changes small and single-iteration implementable.
- Preserve correctness and the same PyTorch binding / entry points.
- Return only the full updated XML blocks (no explanations, no markdown).

IMPORTANT: Generate code in XML format with exactly 3 files with these strict names:

<header_file name="kernel.h">
- All CUDA kernel function declarations
- Host function declarations
- Any necessary struct/type definitions
- Include guards and necessary headers
</header_file>

<cuda_file name="kernel.cu">
- All __global__ kernel implementations
- All __device__ helper functions
- CUDA-specific optimizations and memory patterns
- Proper error checking and memory management
</cuda_file>

<cpp_file name="main.cpp">
- Host function that launches kernels
- Memory allocation and data transfer management
- Device management and error handling
- Entry point function named "run" that can be called to execute the implementation
- Handle both args and kwargs properly
- Move CPU data to GPU, execute kernels, and return results to CPU
- MUST include PyTorch C++ extension bindings using PYBIND11_MODULE
- The "run" function must be exposed to Python through the binding
- Include proper tensor type conversion between PyTorch tensors and CUDA pointers
- Include all necessary PyTorch headers: #include <torch/extension.h>
</cpp_file>

{hints}

Generate the updated implementation:"""


TRITON_DEBUG_PROMPT = """You are in a debug-and-improve loop for a Triton kernel on {target_gpu}.
The current implementation may be buggy OR already correct-but-slower-than-desired.

Original Specification:
{definition}

Known-Good Base Implementation (reference):
{base_code}

Current Implementation (fix or improve THIS code; keep it aligned with the base and the chosen action):
{buggy_code}

Performance Summary:
{perf_summary}

Failure Logs:
{trace_logs}

Chosen Next Action (still targeting; do not expand scope):
{action_text}

Debug-and-improve round: {debug_round}/{max_rounds}

The wrapper function MUST handle complete device management:
- Move CPU tensors to GPU if needed (use .cuda() when torch.cuda.is_available())
- Raise clear errors if CUDA is not available for GPU tensors
- Call the triton kernel with GPU tensors
- Move results back to original device of input tensors
- Handle both args and kwargs properly
- Preserve original tensor devices and restore them for outputs

IMPORTANT: Use only valid Python/Triton syntax:
- NO hexadecimal float literals (0x1.234p5) - use decimal equivalents
- NO C/CUDA specific syntax - this is Python/Triton code
- Use math.log(2), math.pi, math.e instead of hex literals
- All code must be valid Python that passes ast.parse()

You MUST expose a "run" entry point function that can be called to execute the kernel.

Rules:
- If the current implementation FAILED: fix correctness/compile/runtime issues FIRST.
- If the current implementation PASSED: improve performance while preserving correctness.
- Keep changes minimal; do not introduce extra unrelated optimizations.
- Keep the implementation aligned with the base and the chosen action intent.
- Return only the full corrected code (no explanations, no markdown).

{hints}

Generate the corrected implementation:"""


CUDA_DEBUG_PROMPT = """You are in a debug-and-improve loop for a CUDA kernel on {target_gpu}.
The current implementation may be buggy OR already correct-but-slower-than-desired.

Original Specification:
{definition}

Known-Good Base Implementation (reference):
{base_code}

Current Implementation (fix or improve THIS code; keep it aligned with the base and the chosen action):
{buggy_code}

Performance Summary:
{perf_summary}

Failure Logs:
{trace_logs}

Chosen Next Action (still targeting; do not expand scope):
{action_text}

Debug-and-improve round: {debug_round}/{max_rounds}

Rules:
- If the current implementation FAILED: fix correctness/compile/runtime issues FIRST.
- If the current implementation PASSED: improve performance while preserving correctness.
- Keep changes minimal; do not introduce extra unrelated optimizations.
- Keep the implementation aligned with the base and the chosen action intent.
- Return only the full corrected XML blocks (no explanations, no markdown).

IMPORTANT: Generate code in XML format with exactly 3 files with these strict names:

<header_file name="kernel.h">
- All CUDA kernel function declarations
- Host function declarations
- Any necessary struct/type definitions
- Include guards and necessary headers
</header_file>

<cuda_file name="kernel.cu">
- All __global__ kernel implementations
- All __device__ helper functions
- CUDA-specific optimizations and memory patterns
- Proper error checking and memory management
</cuda_file>

<cpp_file name="main.cpp">
- Host function that launches kernels
- Memory allocation and data transfer management
- Device management and error handling
- Entry point function named "run" that can be called to execute the implementation
- Handle both args and kwargs properly
- Move CPU data to GPU, execute kernels, and return results to CPU
- MUST include PyTorch C++ extension bindings using PYBIND11_MODULE
- The "run" function must be exposed to Python through the binding
- Include proper tensor type conversion between PyTorch tensors and CUDA pointers
- Include all necessary PyTorch headers: #include <torch/extension.h>
</cpp_file>

{hints}

Generate the corrected implementation:"""


TRITON_IMPROVE_PROMPT = """You are improving a Triton kernel on {target_gpu}.
The current implementation may be correct-but-slower-than-desired, or it may have regressed.

Original Specification:
{definition}

Cycle-Best Base Implementation (reference):
{base_code}

Current Implementation (improve THIS code; keep it aligned with the base):
{current_code}

Performance Summary:
{perf_summary}

Recent Logs (only if FAILED):
{trace_logs}

Improve round: {debug_round}/{max_rounds}

The wrapper function MUST handle complete device management:
- Move CPU tensors to GPU if needed (use .cuda() when torch.cuda.is_available())
- Raise clear errors if CUDA is not available for GPU tensors
- Call the triton kernel with GPU tensors
- Move results back to original device of input tensors
- Handle both args and kwargs properly
- Preserve original tensor devices and restore them for outputs

IMPORTANT: Use only valid Python/Triton syntax:
- NO hexadecimal float literals (0x1.234p5) - use decimal equivalents
- NO C/CUDA specific syntax - this is Python/Triton code
- Use math.log(2), math.pi, math.e instead of hex literals
- All code must be valid Python that passes ast.parse()

You MUST expose a "run" entry point function that can be called to execute the kernel.

Rules:
- If the current implementation FAILED: fix correctness/compile/runtime issues FIRST.
- If the current implementation PASSED: improve performance while preserving correctness.
- Keep changes minimal; do not introduce extra unrelated optimizations.
- Return only the full corrected code (no explanations, no markdown).

{hints}

Generate the improved implementation:"""


CUDA_IMPROVE_PROMPT = """You are improving a CUDA kernel on {target_gpu}.
The current implementation may be correct-but-slower-than-desired, or it may have regressed.

Original Specification:
{definition}

Cycle-Best Base Implementation (reference):
{base_code}

Current Implementation (improve THIS code; keep it aligned with the base):
{current_code}

Performance Summary:
{perf_summary}

Recent Logs (only if FAILED):
{trace_logs}

Improve round: {debug_round}/{max_rounds}

Rules:
- If the current implementation FAILED: fix correctness/compile/runtime issues FIRST.
- If the current implementation PASSED: improve performance while preserving correctness.
- Keep changes minimal; do not introduce extra unrelated optimizations.
- Return only the full corrected XML blocks (no explanations, no markdown).

IMPORTANT: Generate code in XML format with exactly 3 files with these strict names:

<header_file name="kernel.h">
- All CUDA kernel function declarations
- Host function declarations
- Any necessary struct/type definitions
- Include guards and necessary headers
</header_file>

<cuda_file name="kernel.cu">
- All __global__ kernel implementations
- All __device__ helper functions
- CUDA-specific optimizations and memory patterns
- Proper error checking and memory management
</cuda_file>

<cpp_file name="main.cpp">
- Host function that launches kernels
- Memory allocation and data transfer management
- Device management and error handling
- Entry point function named "run" that can be called to execute the implementation
- Handle both args and kwargs properly
- Move CPU data to GPU, execute kernels, and return results to CPU
- MUST include PyTorch C++ extension bindings using PYBIND11_MODULE
- The "run" function must be exposed to Python through the binding
- Include proper tensor type conversion between PyTorch tensors and CUDA pointers
- Include all necessary PyTorch headers: #include <torch/extension.h>
</cpp_file>

{hints}

Generate the improved implementation:"""

def get_generate_code_from_action_prompt_from_text(
    language: str,
    *,
    definition_text: str,
    base_code: str,
    action_text: str,
    target_gpu: str = "H100",
) -> str:
    """Task-agnostic variant: accepts rendered definition text."""
    lang = (language or "").lower()
    if lang == "triton":
        return TRITON_ACTION_PROMPT.format(
            definition=str(definition_text or "").strip(),
            base_code=base_code,
            action_text=action_text,
            target_gpu=target_gpu,
            hints=TRITON_OPTIMIZATION_HINTS,
        )
    if lang == "cuda":
        return CUDA_ACTION_PROMPT.format(
            definition=str(definition_text or "").strip(),
            base_code=base_code,
            action_text=action_text,
            target_gpu=target_gpu,
            hints=CUDA_OPTIMIZATION_HINTS,
        )
    raise ValueError(f"Unsupported language for action prompt: {language}")


def get_generate_code_from_spec_with_action_prompt_from_text(
    language: str,
    *,
    definition_text: str,
    action_text: str,
    target_gpu: str = "H100",
) -> str:
    """
    Task-agnostic variant: accepts rendered definition text.
    Used when the chosen action's parent is the WM root: start from spec + action only.
    """
    lang = (language or "").lower()
    if lang == "triton":
        return (
            "You are implementing a SPECIFIC NEXT ACTION starting from the specification.\n\n"
            + TRITON_ACTION_PROMPT.format(
                definition=str(definition_text or "").strip(),
                base_code="(no base code; start from spec)",
                action_text=action_text,
                target_gpu=target_gpu,
                hints=TRITON_OPTIMIZATION_HINTS,
            )
        )
    if lang == "cuda":
        return (
            "You are implementing a SPECIFIC NEXT ACTION starting from the specification.\n\n"
            + CUDA_ACTION_PROMPT.format(
                definition=str(definition_text or "").strip(),
                base_code="(no base code; start from spec)",
                action_text=action_text,
                target_gpu=target_gpu,
                hints=CUDA_OPTIMIZATION_HINTS,
            )
        )
    raise ValueError(f"Unsupported language for spec+action prompt: {language}")


def get_debug_and_improve_from_spec_prompt_from_text(
    language: str,
    *,
    definition_text: str,
    trace_logs: str,
    current_code: str,
    action_text: str,
    debug_round: int,
    max_rounds: int = 5,
    target_gpu: str = "H100",
    perf_summary: str = "",
    base_code: str = "(no base code; start from spec)",
) -> str:
    return get_debug_generated_code_prompt_from_text(
        language,
        definition_text=definition_text,
        trace_logs=trace_logs,
        base_code=base_code,
        buggy_code=current_code,
        action_text=action_text,
        debug_round=debug_round,
        max_rounds=max_rounds,
        target_gpu=target_gpu,
        perf_summary=perf_summary,
    )


def get_debug_generated_code_prompt_from_text(
    language: str,
    *,
    definition_text: str,
    trace_logs: str,
    base_code: str,
    buggy_code: str,
    action_text: str,
    debug_round: int,
    max_rounds: int = 5,
    target_gpu: str = "H100",
    perf_summary: str = "",
) -> str:
    """Task-agnostic variant: accepts rendered definition + rendered trace logs."""
    lang = (language or "").lower()
    dr = int(debug_round)
    if dr < 1:
        dr = 1
    mr = int(max_rounds)
    if mr < 1:
        mr = 1
    if dr > mr:
        dr = mr
    if lang == "triton":
        return TRITON_DEBUG_PROMPT.format(
            definition=str(definition_text or "").strip(),
            base_code=base_code,
            buggy_code=buggy_code,
            perf_summary=str(perf_summary or "").strip() or "(none)",
            trace_logs=str(trace_logs or "").strip() or "(no logs)",
            action_text=action_text,
            debug_round=dr,
            max_rounds=mr,
            target_gpu=target_gpu,
            hints=TRITON_OPTIMIZATION_HINTS,
        )
    if lang == "cuda":
        return CUDA_DEBUG_PROMPT.format(
            definition=str(definition_text or "").strip(),
            base_code=base_code,
            buggy_code=buggy_code,
            perf_summary=str(perf_summary or "").strip() or "(none)",
            trace_logs=str(trace_logs or "").strip() or "(no logs)",
            action_text=action_text,
            debug_round=dr,
            max_rounds=mr,
            target_gpu=target_gpu,
            hints=CUDA_OPTIMIZATION_HINTS,
        )
    raise ValueError(f"Unsupported language for debug prompt: {language}")


def get_improve_from_spec_prompt_from_text(
    language: str,
    *,
    definition_text: str,
    trace_logs: str,
    current_code: str,
    debug_round: int,
    max_rounds: int = 5,
    target_gpu: str = "H100",
    perf_summary: str = "",
    base_code: str = "(no base code; start from spec)",
) -> str:
    return get_improve_generated_code_prompt_from_text(
        language,
        definition_text=definition_text,
        trace_logs=trace_logs,
        base_code=base_code,
        current_code=current_code,
        debug_round=debug_round,
        max_rounds=max_rounds,
        target_gpu=target_gpu,
        perf_summary=perf_summary,
    )


def get_improve_generated_code_prompt_from_text(
    language: str,
    *,
    definition_text: str,
    trace_logs: str,
    base_code: str,
    current_code: str,
    debug_round: int,
    max_rounds: int = 5,
    target_gpu: str = "H100",
    perf_summary: str = "",
) -> str:
    """Task-agnostic variant: accepts rendered definition + rendered trace logs."""
    lang = (language or "").lower()
    dr = int(debug_round)
    if dr < 1:
        dr = 1
    mr = int(max_rounds)
    if mr < 1:
        mr = 1
    if dr > mr:
        dr = mr
    if lang == "triton":
        return TRITON_IMPROVE_PROMPT.format(
            definition=str(definition_text or "").strip(),
            base_code=base_code,
            current_code=current_code,
            perf_summary=str(perf_summary or "").strip() or "(none)",
            trace_logs=str(trace_logs or "").strip() or "(no logs)",
            debug_round=dr,
            max_rounds=mr,
            target_gpu=target_gpu,
            hints=TRITON_OPTIMIZATION_HINTS,
        )
    if lang == "cuda":
        return CUDA_IMPROVE_PROMPT.format(
            definition=str(definition_text or "").strip(),
            base_code=base_code,
            current_code=current_code,
            perf_summary=str(perf_summary or "").strip() or "(none)",
            trace_logs=str(trace_logs or "").strip() or "(no logs)",
            debug_round=dr,
            max_rounds=mr,
            target_gpu=target_gpu,
            hints=CUDA_OPTIMIZATION_HINTS,
        )
    raise ValueError(f"Unsupported language for improve prompt: {language}")


