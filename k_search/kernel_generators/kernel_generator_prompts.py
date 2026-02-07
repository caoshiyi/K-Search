"""
This file contains the prompts for baseline agent generation.
"""

from __future__ import annotations

# CUDA-specific hints
# Note: keep these hints generic (avoid naming specific low-level instructions).
CUDA_OPTIMIZATION_HINTS = "** You MUST use MMA to utilize the tensor cores on H100! ** For each round, you can see your current best solution and the previous round's summary, therefore you can implement the kernel step by step."

# Triton-appropriate subset
TRITON_OPTIMIZATION_HINTS = """
Advanced Optimization Hints (Triton):
- Choose tile shapes/layouts that map cleanly to accelerator-friendly math paths when available.
- Pipeline loads and compute via num_stages; use staged shared-memory buffers to overlap tl.load/tl.store with math.
- Vectorize memory (aligned block pointers for tl.load/tl.store) to reduce transactions and bank conflicts.
- Use tl.multiple_of and tl.max_contiguous to help the compiler generate efficient code.
- Replace costly integer division/mod in hot loops with precomputed reciprocals and multiply-add sequences.
- Tune num_warps and num_stages to balance register pressure vs occupancy and latency hiding.
- Parameterize tile sizes, vector widths, and stage counts to enable autotuning-friendly dispatch.
"""


TRITON_PROMPT = """Generate a Triton kernel optimized for {target_gpu} GPU for

{definition}

Triton Version: 3.3.1

Requirements:
- Write clean, efficient Triton code optimized for {target_gpu} architecture
- Use modern Triton syntax with proper grid computation and language features
- Include necessary imports (torch, triton, triton.language as tl)
- Implement the exact functionality described in the specification
- The reference code provides the mathematical specification but is unoptimized - your Triton implementation should match its computational accuracy while delivering high performance
- Use the definition's tensor shapes, dtypes, and axes information to guide memory access patterns and optimization strategies
- Optimize for {target_gpu} GPU characteristics (memory hierarchy, compute units, etc.)

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

- Expose a "run" entry point function that can be called to execute the kernel
- Return only the code, no explanations or markdown formatting

Generate complete, runnable code only - no framework will add device handling wrapper code.

{hints}

Generate the implementation:"""

TRITON_OPTIMIZATION_PROMPT = """You are optimizing a Triton kernel for {target_gpu} GPU. The current implementation has issues that need to be fixed or its performance can be improved.

Original Specification:
{definition}

Current Implementation Status:
{trace_logs}

Current Implementation:
{current_code}

Optimization Strategy:
1. ENSURE CORRECTNESS: If there are compile errors, runtime errors, or incorrect outputs, focus entirely on fixing these issues
   - Analyze compilation errors and fix syntax/API usage
   - Fix runtime errors like shape mismatches, memory access violations
   - Ensure numerical correctness matches the reference implementation

2. OPTIMIZE PERFORMANCE: if the current kernel is functionally correct, focus on performance optimizations
   - Optimize memory access patterns for {target_gpu}
   - Tune block sizes and grid dimensions
   - Use appropriate Triton language features for vectorization
   - Minimize global memory transactions

Requirements for the optimized implementation:
- Write clean, efficient Triton code optimized for {target_gpu} architecture
- Use modern Triton syntax with proper grid computation and language features
- Include necessary imports (torch, triton, triton.language as tl)
- Fix all identified issues from the feedback
- Maintain or improve computational accuracy
- Preserve the same function signature and device handling as specified

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

- Expose a "run" entry point function that can be called to execute the kernel
- Return only the improved code, no explanations or markdown formatting

{hints}

{extra_context}
Generate the corrected and optimized implementation:"""

PYTHON_PROMPT = """You are a code generator. Generate a Python implementation optimized for {target_gpu} GPU for the following specification.

Specification:
{definition}

Requirements:
- Write clean, efficient Python code optimized for {target_gpu} architecture
- Use PyTorch operations when appropriate, optimized for {target_gpu}
- Include necessary imports
- Implement the exact functionality described in the specification
- Expose a "run" entry point function that can be called to execute the implementation
- Return only the code, no explanations or markdown formatting

Generate the implementation:"""

# CUDA prompt
CUDA_PROMPT = """You are a code generator. Generate a CUDA kernel implementation optimized for {target_gpu} GPU for the following specification.

Specification:
{definition}

Requirements:
- Write clean, efficient CUDA C++ code optimized for {target_gpu} architecture
- Use proper CUDA syntax and memory management optimized for {target_gpu}
- Implement the exact functionality described in the specification
- The reference code provides the mathematical specification but is unoptimized - your CUDA implementation should match its computational accuracy while delivering high performance
- Use the definition's tensor shapes, dtypes, and axes information to guide memory access patterns and optimization strategies
- Optimize for {target_gpu} GPU characteristics (memory hierarchy, compute units, etc.)
- For fixed axis values, optimize specifically for those constants rather than general cases
- You may use 3rd party libraries (cuBLAS, cuDNN, CUTLASS) when beneficial, but custom implementations often perform better for specialized kernels with known axis constraints

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
- Include PyTorch C++ extension bindings using PYBIND11_MODULE
- The "run" function must be exposed to Python through the binding
- Include proper tensor type conversion between PyTorch tensors and CUDA pointers
- Include all necessary PyTorch headers: #include <torch/extension.h>
</cpp_file>

Code Generation Guidelines:
- Use modern CUDA features appropriate for {target_gpu}
- Optimize memory coalescing and reduce bank conflicts
- Utilize shared memory effectively for data reuse
- Consider occupancy and register usage
- Implement proper error checking with cudaGetLastError()
- Use appropriate grid and block dimensions for the problem size
- Leverage constant memory for frequently accessed read-only data
- Use PyTorch tensor API (torch::Tensor) for all tensor arguments in the "run" function
- Convert PyTorch tensors to CUDA pointers using .data_ptr<float>() or similar methods
- Ensure proper CUDA stream synchronization and error handling

{hints}

Generate the implementation:"""

CUDA_OPTIMIZATION_PROMPT = """You are optimizing a CUDA kernel for {target_gpu} GPU. The current implementation has issues that need to be fixed.

Original Specification:
{definition}

Current Implementation Status:
{trace_logs}

Current Implementation:
{current_code}

Optimization Strategy:
1. ENSURE CORRECTNESS: If there are compile errors, runtime errors, or incorrect outputs, focus entirely on fixing these issues
   - Analyze compilation errors and fix syntax/API usage
   - Fix runtime errors like shape mismatches, memory access violations, kernel launch failures
   - Ensure numerical correctness matches the reference implementation
   - Verify proper CUDA memory management and synchronization

2. OPTIMIZE PERFORMANCE: if the current kernel is functionally correct, focus on performance optimizations
   - Optimize memory access patterns and coalescing for {target_gpu}
   - Tune block sizes and grid dimensions for maximum occupancy
   - Utilize shared memory effectively to reduce global memory transactions
   - Optimize register usage and minimize divergent branches
   - Consider using specialized libraries (cuBLAS, cuDNN, CUTLASS) where beneficial
   - Leverage constant axis values for compile-time optimizations

Requirements for the optimized implementation:
- Write clean, efficient CUDA C++ code optimized for {target_gpu} architecture
- Use proper CUDA syntax and modern features appropriate for {target_gpu}
- Fix all identified issues from the feedback
- Maintain or improve computational accuracy
- Preserve the same function signatures and device handling as specified
- For fixed axis values, optimize specifically for those constants rather than general cases

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

Code Generation Guidelines:
- Use modern CUDA features appropriate for {target_gpu}
- Optimize memory coalescing and reduce bank conflicts
- Utilize shared memory effectively for data reuse
- Consider occupancy and register usage
- Implement proper error checking with cudaGetLastError()
- Use appropriate grid and block dimensions for the problem size
- Leverage constant memory for frequently accessed read-only data
- Use PyTorch tensor API (torch::Tensor) for all tensor arguments in the "run" function
- Convert PyTorch tensors to CUDA pointers using .data_ptr<float>() or similar methods
- Ensure proper CUDA stream synchronization and error handling

{hints}

{extra_context}

Generate the corrected and optimized implementation:"""


def get_prompt_from_definition_text(language: str, definition_text: str, target_gpu: str = "H100") -> str:
    """
    Task-agnostic prompt builder: takes a fully-rendered definition text.
    """
    prompts = {"triton": TRITON_PROMPT, "python": PYTHON_PROMPT, "cuda": CUDA_PROMPT}

    if language not in prompts:
        raise ValueError(f"Unsupported language: {language}")

    # Only Triton/CUDA prompts include advanced hints
    if language == "triton":
        return prompts[language].format(
            definition=str(definition_text or "").strip(), target_gpu=target_gpu, hints=TRITON_OPTIMIZATION_HINTS
        )
    if language == "cuda":
        return prompts[language].format(
            definition=str(definition_text or "").strip(), target_gpu=target_gpu, hints=CUDA_OPTIMIZATION_HINTS
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
            hints=TRITON_OPTIMIZATION_HINTS,
            extra_context=extra_context,
        )
    if language == "cuda":
        return optimization_prompts[language].format(
            definition=str(definition_text or "").strip(),
            trace_logs=str(trace_logs or "").strip(),
            current_code=current_code,
            target_gpu=target_gpu,
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
