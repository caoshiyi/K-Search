"""Task adapters (evaluation backends) for k-search.

Keep this module **lightweight**:
- Importing `k_search.tasks.<something>` triggers `k_search.tasks.__init__` first.
- Avoid importing heavy optional deps (e.g. flashinfer-bench) at import time.
"""

from k_search.tasks.task_base import BuildSpec, EvalResult, Solution, SourceFile, SupportedLanguages, Task, code_from_solution

__all__ = [
    "BuildSpec",
    "EvalResult",
    "Solution",
    "SourceFile",
    "SupportedLanguages",
    "Task",
    "code_from_solution",
]

# Optional task implementations (guarded to avoid hard deps).
try:  # pragma: no cover
    from k_search.tasks.flashinfer_bench_task import FlashInferBenchTask

    __all__.append("FlashInferBenchTask")
except Exception:
    FlashInferBenchTask = None  # type: ignore

try:  # pragma: no cover
    from k_search.tasks.gpu_mode_task import GpuModeTriMulTask

    __all__.append("GpuModeTriMulTask")
except Exception:
    GpuModeTriMulTask = None  # type: ignore


