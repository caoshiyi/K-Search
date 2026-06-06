from __future__ import annotations

from pathlib import Path


NATIVE_HANDOFF_FILES = {
    "CODE_MAP.md",
    "ASCENDC_DESIGN.md",
    "IMPLEMENTATION_EXECUTION_PLAN.md",
    "IMPLEMENTATION_HANDOFF.md",
    "IMPLEMENTATION_DEVIATIONS.md",
    "REVIEW_NOTES.md",
}

NATIVE_MEMORY_FILES = {
    "KNOWLEDGE.md",
}

NATIVE_DEBUG_EVIDENCE_FILES = {
    "debug_packet.json",
    "debug_log.md",
}

NATIVE_RUNTIME_FILES = NATIVE_HANDOFF_FILES | NATIVE_MEMORY_FILES | NATIVE_DEBUG_EVIDENCE_FILES

NATIVE_RUNTIME_DIRS = {
    ".claude",
    ".ksearch",
}


def is_native_runtime_path(path: str | Path) -> bool:
    rel = str(path or "").replace("\\", "/").strip()
    if not rel:
        return True
    parts = tuple(part for part in rel.split("/") if part)
    if any(part in NATIVE_RUNTIME_DIRS for part in parts):
        return True
    return rel in NATIVE_RUNTIME_FILES
