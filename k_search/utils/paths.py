from __future__ import annotations

from pathlib import Path
from typing import Optional, Union


PathLike = Union[str, Path]


def get_ksearch_artifacts_dir(*, base_dir: Optional[PathLike] = None, task_name: Optional[str] = None) -> Path:
    """
    Default k-search artifacts directory (independent of flashinfer-bench dataset paths).
    """
    root = Path(base_dir) if base_dir else (Path.cwd() / ".ksearch")
    root = root.expanduser().resolve()
    if task_name:
        safe = "".join([c if (c.isalnum() or c in ("-", "_", ".")) else "_" for c in str(task_name)])
        return root / safe
    return root


