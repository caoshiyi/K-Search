"""
Per-task prompt blocks for FlashInferBenchTask.

These are injected into the generic generator templates via `{per_task_requirement}` so:
- GPUMode tasks can keep it empty
- FlashInferBenchTask can provide phase-specific guidance without embedding large strings in the task class
"""

from __future__ import annotations


def _triton_wrapper_and_output_guidelines_block() -> str:
    # Starts at the "wrapper function MUST handle complete device management" section.
    return """The wrapper function MUST handle complete device management:
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

Triton output format guidelines:
- You MUST expose a "run" entry point function that can be called to execute the kernel.
- Return only the code (no explanations, no markdown formatting)."""


def triton_requirements_block() -> str:
    return (
        """Requirements:
- Write clean, efficient Triton code optimized for the target GPU
- Use modern Triton syntax with proper grid computation and language features
- Include necessary imports (torch, triton, triton.language as tl)
- Implement the exact functionality described in the specification
- Use the definition's tensor shapes, dtypes, and axes information to guide memory access patterns and optimization strategies"""
        + "\n\n"
        + _triton_wrapper_and_output_guidelines_block()
    )


def triton_optimization_strategy_block(*, target_gpu: str) -> str:
    tg = str(target_gpu or "H100")
    return f"""Optimization Strategy:
1. ENSURE CORRECTNESS: If there are compile errors, runtime errors, or incorrect outputs, focus entirely on fixing these issues
   - Analyze compilation errors and fix syntax/API usage
   - Fix runtime errors like shape mismatches, memory access violations
   - Ensure numerical correctness matches the reference implementation

2. OPTIMIZE PERFORMANCE: if the current kernel is functionally correct, focus on performance optimizations
   - Optimize memory access patterns for {tg}
   - Tune block sizes and grid dimensions
   - Use appropriate Triton language features for vectorization
   - Minimize global memory transactions"""


def code_format_text(*, language: str, target_gpu: str) -> str:
    """
    Format/output guidelines used by world-model prompts (`{code_format}`).
    Keep this small and stable across phases.
    """
    lang = str(language or "").strip().lower()
    tg = str(target_gpu or "H100")
    if lang in ("triton", "python"):
        return _triton_wrapper_and_output_guidelines_block().strip()
    if lang == "cuda":
        return _cuda_xml_and_guidelines_block(target_gpu=tg).strip()
    return ""


def cuda_requirements_block(*, target_gpu: str) -> str:
    tg = str(target_gpu or "H100")
    return f"""Requirements:
- Write clean, efficient CUDA C++ code optimized for {tg}
- Use proper CUDA syntax and memory management optimized for {tg}
- Implement the exact functionality described in the specification
- The reference code provides the mathematical specification but is unoptimized - your CUDA implementation should match its computational accuracy while delivering high performance
- Use the definition's tensor shapes, dtypes, and axes information to guide memory access patterns and optimization strategies
- Optimize for {tg} GPU characteristics (memory hierarchy, compute units, etc.)
- For fixed axis values, optimize specifically for those constants rather than general cases
- You may use 3rd party libraries (cuBLAS, cuDNN, CUTLASS) when beneficial, but custom implementations often perform better for specialized kernels with known axis constraints

{_cuda_xml_and_guidelines_block(target_gpu=tg)}"""


def _cuda_xml_and_guidelines_block(*, target_gpu: str) -> str:
    tg = str(target_gpu or "H100")
    return f"""IMPORTANT: Generate code in XML format with exactly 3 files with these strict names:

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
- Use modern CUDA features appropriate for {tg}
- Optimize memory coalescing and reduce bank conflicts
- Utilize shared memory effectively for data reuse
- Consider occupancy and register usage
- Implement proper error checking with cudaGetLastError()
- Use appropriate grid and block dimensions for the problem size
- Leverage constant memory for frequently accessed read-only data
- Use PyTorch tensor API (torch::Tensor) for all tensor arguments in the "run" function
- Convert PyTorch tensors to CUDA pointers using .data_ptr<float>() or similar methods
- Ensure proper CUDA stream synchronization and error handling"""


def cuda_optimization_strategy_block(*, target_gpu: str) -> str:
    tg = str(target_gpu or "H100")
    return f"""Optimization Strategy:
1. ENSURE CORRECTNESS: If there are compile errors, runtime errors, or incorrect outputs, focus entirely on fixing these issues
   - Analyze compilation errors and fix syntax/API usage
   - Fix runtime errors like shape mismatches, memory access violations, kernel launch failures
   - Ensure numerical correctness matches the reference implementation
   - Verify proper CUDA memory management and synchronization

2. OPTIMIZE PERFORMANCE: if the current kernel is functionally correct, focus on performance optimizations
   - Optimize memory access patterns and coalescing for {tg}
   - Tune block sizes and grid dimensions for maximum occupancy
   - Utilize shared memory effectively to reduce global memory transactions
   - Optimize register usage and minimize divergent branches
   - Consider using specialized libraries (cuBLAS, cuDNN, CUTLASS) where beneficial
   - Leverage constant axis values for compile-time optimizations"""


def cuda_optimization_requirements_block(*, target_gpu: str) -> str:
    tg = str(target_gpu or "H100")
    return f"""Requirements for the optimized implementation:
- Write clean, efficient CUDA C++ code optimized for {tg}
- Use proper CUDA syntax and modern features appropriate for {tg}
- Fix all identified issues from the feedback
- Maintain or improve computational accuracy
- Preserve the same function signatures and device handling as specified
- For fixed axis values, optimize specifically for those constants rather than general cases

{_cuda_xml_and_guidelines_block(target_gpu=tg)}"""


def per_task_requirement_text(*, language: str, target_gpu: str, phase: str = "") -> str:
    lang = str(language or "").strip().lower()
    tg = str(target_gpu or "H100")
    raw_ph = str(phase or "").strip().lower()

    # Normalize to a single canonical key so callers don't need to remember synonyms.
    # Canonical phases:
    # - "generate"
    # - "optimize"
    if raw_ph in ("generate", ""):
        ph = "generate"
    elif raw_ph in ("optimize", "optimization", "improve") or raw_ph.startswith("opt"):
        ph = "optimize"
    else:
        ph = raw_ph

    if lang in ("triton", "python"):
        if ph == "optimize":
            return (triton_optimization_strategy_block(target_gpu=tg) + "\n\n" + triton_requirements_block()).strip()
        return triton_requirements_block().strip()

    if lang == "cuda":
        if ph == "optimize":
            return (
                cuda_optimization_strategy_block(target_gpu=tg)
                + "\n\n"
                + cuda_optimization_requirements_block(target_gpu=tg)
            ).strip()
        return cuda_requirements_block(target_gpu=tg).strip()

    return ""


