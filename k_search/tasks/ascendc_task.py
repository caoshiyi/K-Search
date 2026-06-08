"""AscendC task adapter.

This backend is intentionally command-driven. K-Search owns generation, solution
packing, scoring, and logs; the project-specific Ascend/CANN environment owns
the actual build, correctness, and benchmark commands.
"""

from __future__ import annotations

import json
import os
import re
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Any

from k_search.tasks.ascendc_patch import (
    ASCENDC_PATCH_FORMAT_TEXT,
    parse_ascendc_project_patch,
)
from k_search.tasks.task_base import (
    BuildSpec,
    EvalResult,
    Solution,
    SourceFile,
    SupportedLanguages,
    load_ksearch_solution_json,
    solution_from_json_dict,
)
from k_search.kernel_generators.runtime_artifacts import NATIVE_RUNTIME_DIRS, NATIVE_RUNTIME_FILES
from k_search.utils.path_sanitize import sanitize_worktree_paths


_FILE_BLOCK_RE = re.compile(
    r"<file\s+path=\"([^\"]+)\"\s*>\s*(?:<!\[CDATA\[(.*?)\]\]>|(.*?))\s*</file>",
    re.DOTALL,
)


ASCENDC_CODE_FORMAT_TEXT = """Return only this multi-file container, with no markdown or explanations:
<ascendc_project>
<file path="kernel.cpp"><![CDATA[
// AscendC device code
]]></file>
<file path="tiling.cpp"><![CDATA[
// Host tiling code if needed
]]></file>
<file path="CMakeLists.txt"><![CDATA[
# Build file if the task requires changing it
]]></file>
</ascendc_project>
Include every file needed by the candidate. Keep paths relative to the project root."""

VALID_CODEGEN_MODES = ("auto", "full", "patch")
ALLOWED_AGENTIC_SOURCE_SUFFIXES = {
    ".c",
    ".cc",
    ".cpp",
    ".h",
    ".hh",
    ".hpp",
    ".json",
    ".yaml",
    ".yml",
    ".txt",
    ".cmake",
}
ALLOWED_AGENTIC_SOURCE_NAMES = {"CMakeLists.txt"}
FORBIDDEN_AGENTIC_PATH_PARTS = {
    ".git",
    ".claude",
    "__pycache__",
    "build",
    "cmake-build-debug",
    "logs",
    "llm_logs",
    *NATIVE_RUNTIME_DIRS,
}


def _normalize_rel_path(path: str) -> str:
    raw = str(path or "").strip().replace("\\", "/")
    if not raw:
        raise ValueError("empty source path")
    p = Path(raw)
    if p.is_absolute() or ".." in p.parts:
        raise ValueError(f"unsafe source path: {path!r}")
    return raw


def _is_forbidden_agentic_changed_path(path: str) -> bool:
    rel = str(path or "").strip().replace("\\", "/")
    if not rel:
        return True
    parts = tuple(p for p in rel.split("/") if p)
    return rel in NATIVE_RUNTIME_FILES or any(part in FORBIDDEN_AGENTIC_PATH_PARTS for part in parts)


def parse_ascendc_project_files(raw: Any) -> dict[str, str]:
    """Parse K-Search's AscendC multi-file container into path -> content."""
    if isinstance(raw, dict):
        return {_normalize_rel_path(str(k)): str(v or "") for k, v in raw.items()}

    text = str(raw or "")
    files: dict[str, str] = {}
    for match in _FILE_BLOCK_RE.finditer(text):
        path = _normalize_rel_path(match.group(1))
        content = match.group(2) if match.group(2) is not None else match.group(3)
        files[path] = str(content or "").strip("\n")
    if not files:
        raise ValueError("AscendC response did not contain any <file path=\"...\"> blocks")
    return files


def format_ascendc_project_files(files: dict[str, str]) -> str:
    """Format path -> content as the container expected from AscendC prompts."""
    parts = ["<ascendc_project>"]
    for path in sorted(files):
        rel = _normalize_rel_path(path)
        content = str(files[path] or "")
        if "]]>" in content:
            content = content.replace("]]>", "]] >")
        parts.append(f'<file path="{rel}"><![CDATA[\n{content}\n]]></file>')
    parts.append("</ascendc_project>")
    return "\n".join(parts)


def _read_first_existing_text(root: Path, names: list[str]) -> tuple[str, Path | None]:
    for name in names:
        p = root / name
        if p.exists() and p.is_file():
            return p.read_text(encoding="utf-8", errors="replace"), p
    return "", None


def _is_source_candidate(path: Path) -> bool:
    if path.name in ALLOWED_AGENTIC_SOURCE_NAMES:
        return True
    return path.suffix.lower() in ALLOWED_AGENTIC_SOURCE_SUFFIXES


def _collect_project_sources(root: Path, *, max_files: int = 80, max_bytes_per_file: int = 200_000) -> list[SourceFile]:
    if not root.exists() or not root.is_dir():
        return []
    out: list[SourceFile] = []
    for p in sorted(root.rglob("*")):
        rel_path = p.relative_to(root)
        rel = str(rel_path).replace("\\", "/")
        if rel in NATIVE_RUNTIME_FILES or any(part in FORBIDDEN_AGENTIC_PATH_PARTS for part in rel_path.parts):
            continue
        if not p.is_file() or not _is_source_candidate(p):
            continue
        try:
            if p.stat().st_size > max_bytes_per_file:
                continue
            out.append(SourceFile(path=rel, content=p.read_text(encoding="utf-8", errors="replace")))
        except Exception:
            continue
        if len(out) >= max_files:
            break
    return out


def _remove_project_source_candidates(root: Path) -> None:
    if not root.exists() or not root.is_dir():
        return
    for p in sorted(root.rglob("*"), reverse=True):
        try:
            rel_path = p.relative_to(root)
            rel = str(rel_path).replace("\\", "/")
            if rel in NATIVE_RUNTIME_FILES or any(part in FORBIDDEN_AGENTIC_PATH_PARTS for part in rel_path.parts):
                continue
            if p.is_file() and _is_source_candidate(p):
                p.unlink()
        except Exception:
            continue


def _default_entry_point(sources: list[SourceFile]) -> str:
    preferred = ("kernel.cpp", "op_kernel.cpp", "main.cpp")
    paths = {s.path for s in sources}
    for p in preferred:
        if p in paths:
            return f"{p}::run"
    if sources:
        return f"{sources[0].path}::run"
    return "kernel.cpp::run"


def _parse_latency_ms(output: str) -> float | None:
    text = str(output or "")
    try:
        obj = json.loads(text)
        if isinstance(obj, dict):
            for key in ("latency_ms", "mean_latency_ms", "avg_latency_ms"):
                val = obj.get(key)
                if isinstance(val, (int, float)) and float(val) > 0:
                    return float(val)
    except Exception:
        pass
    patterns = (
        r"\b(?:mean_|avg_)?latency_ms\b\s*[:=]\s*([0-9]+(?:\.[0-9]+)?)",
        r"\b(?:mean|avg)?\s*latency\s*[:=]\s*([0-9]+(?:\.[0-9]+)?)\s*ms\b",
        r"\btime_ms\b\s*[:=]\s*([0-9]+(?:\.[0-9]+)?)",
        r"\bmean_ms\b\s*[:=]\s*([0-9]+(?:\.[0-9]+)?)",
        r"\bmean\b\s*[:=]\s*([0-9]+(?:\.[0-9]+)?)\s*ms\b",
        r"\bmean_us\b\s*[:=]\s*([0-9]+(?:\.[0-9]+)?)\b",
        r"\bmean\b\s*[:=]\s*([0-9]+(?:\.[0-9]+)?)\s*us\b",
    )
    for pattern in patterns:
        m = re.search(pattern, text, flags=re.IGNORECASE)
        if not m:
            continue
        try:
            val = float(m.group(1))
            if val > 0:
                # mean_us and mean=xxxus values need conversion from us to ms
                matched_text = m.group(0)
                if "_us" in pattern or ("us" in matched_text and "ms" not in matched_text):
                    return val / 1000.0
                return val
        except Exception:
            continue
    return None


class AscendCTask:
    """Direct AscendC backend using external build/correctness/benchmark commands."""

    def __init__(
        self,
        *,
        task_path: str | Path | None,
        definition_name: str | None = None,
        build_cmd: str | None = None,
        test_cmd: str | None = None,
        bench_cmd: str | None = None,
        reference_latency_ms: float | None = None,
        timeout_seconds: int = 600,
        artifacts_dir: str | None = None,
        codegen_mode: str | None = None,
    ) -> None:
        self.task_path = Path(task_path).expanduser().resolve() if task_path else None
        self._name = str(definition_name or (self.task_path.stem if self.task_path else "ascendc_task")).strip()
        self.build_cmd = str(build_cmd or "").strip()
        self.test_cmd = str(test_cmd or "").strip()
        self.bench_cmd = str(bench_cmd or "").strip()
        self.reference_latency_ms = float(reference_latency_ms) if reference_latency_ms else None
        self.timeout_seconds = int(timeout_seconds or 600)
        self.artifacts_dir = artifacts_dir
        self._last_eval: EvalResult | None = None

        mode = codegen_mode or os.environ.get("KSEARCH_ASCENDC_CODEGEN_MODE") or "auto"
        mode = str(mode).strip().lower()
        if mode not in VALID_CODEGEN_MODES:
            raise ValueError(
                f"invalid codegen_mode={mode!r}; expected one of {VALID_CODEGEN_MODES}"
            )
        self.codegen_mode = mode
        self._last_parsed_files: dict[str, str] | None = None
        self._last_parsed_raw: str | None = None
        self._patch_failure_streak = 0
        self._max_patch_failures = 3

    @property
    def name(self) -> str:
        return self._name

    def get_definition_text(
        self,
        language: str | None = None,
        *,
        include_sources: bool = True,
        include_format: bool = True,
    ) -> str:
        task_path = self.task_path
        spec_text = ""
        spec_path: Path | None = None
        if task_path and task_path.is_file():
            spec_text = task_path.read_text(encoding="utf-8", errors="replace")
            spec_path = task_path
        elif task_path and task_path.is_dir():
            spec_text, spec_path = _read_first_existing_text(
                task_path,
                ["ksearch_task.md", "spec.md", "README.md", "task.yml", "task.yaml"],
            )

        lines = [
            f"Task: {self.name}",
            "Target language: AscendC",
            "Target platform hint: use the CLI --target-gpu value, usually ascend_910b/ascend_310p.",
        ]
        if spec_path is not None:
            lines.append(f"Specification source: {spec_path}")
        if spec_text.strip():
            lines.extend(["", "Specification:", spec_text.strip()])
        else:
            lines.extend(
                [
                    "",
                    "Specification:",
                    "Optimize the provided AscendC operator project while preserving its public inputs, outputs, tiling contract, and build/test harness behavior.",
                ]
            )

        if include_sources and task_path and task_path.is_dir():
            sources = _collect_project_sources(task_path, max_files=20, max_bytes_per_file=40_000)
            if sources:
                lines.append("\nExisting project source excerpts:")
                for src in sources:
                    content = src.content
                    if len(content) > 4000:
                        content = content[:4000] + "\n...<truncated>..."
                    lines.append(f"\n--- {src.path} ---\n{content}")

        if include_format:
            lines.extend(["", ASCENDC_CODE_FORMAT_TEXT])
        return "\n".join(lines).strip()

    def get_generation_prompt(self, *, language: str, target_gpu: str) -> str:
        return f"""You are generating an AscendC multi-file operator project optimized for {target_gpu}.

Original Specification:
{self.get_definition_text(language=language)}

Rules:
- Preserve operator semantics, public entry points, host tiling contract, and correctness harness expectations.
- Prefer small, reviewable AscendC changes over broad rewrites.
- Treat Host tiling, TilingData, blockDim, workspace, TPipe/TQue/TBuf, DataCopy/DataCopyPad, UB/L1/L0, AIC/AIV, Matmul API, and tail/alignment handling as first-class performance surfaces.
- Return only the AscendC multi-file container.

Generate the implementation:"""

    def get_optimization_prompt(
        self,
        *,
        language: str,
        target_gpu: str,
        trace_logs: str,
        current_code: str,
        current_best: str | None = None,
        previous_round_summary: str | None = None,
    ) -> str:
        extra = []
        if previous_round_summary:
            extra.append("Previous Round Summary:\n" + previous_round_summary)
        if current_best:
            extra.append("Current Best Solution So Far:\n" + current_best)
        extra_text = "\n\n".join(extra)

        has_baseline = bool(str(current_code or "").strip())
        mode = self._resolve_codegen_mode(has_baseline=has_baseline)
        format_text = self._format_text_for_mode(mode)

        if mode == "patch":
            response_rule = "- Return only the <ascendc_patch> container (unified diff)."
        else:
            response_rule = "- Return only the full AscendC multi-file container."

        return f"""You are optimizing an AscendC multi-file operator project for {target_gpu}.

Original Specification:
{self.get_definition_text(language=language)}

Current Implementation Status:
{trace_logs or "(no logs)"}

Current Implementation:
{current_code}

{extra_text}

Rules:
- If compilation or correctness failed, fix that first.
- If it passed, improve measured latency while preserving semantics.
- Keep changes small enough for one K-Search round.
{response_rule}

Response format:
{format_text}

Generate the corrected and optimized implementation:"""

    def _resolve_codegen_mode(self, *, has_baseline: bool) -> str:
        """Resolve effective mode for one call.

        - "full" -> always full
        - "patch" -> patch when there is a baseline, else full (cold start safety)
        - "auto" -> patch when there is a baseline, else full
        """
        if self.codegen_mode == "full":
            return "full"
        return "patch" if has_baseline else "full"

    def _format_text_for_mode(self, mode: str) -> str:
        return ASCENDC_PATCH_FORMAT_TEXT if mode == "patch" else ASCENDC_CODE_FORMAT_TEXT

    def get_code_format_text(self, *, language: str, target_gpu: str) -> str:
        """Hook used by the world-model generator to embed a code-format reminder."""
        # The world-model generator asks get_baseline_code_for_codegen() for an
        # explicit disk baseline when no parent solution exists, so patch mode has
        # a concrete base even for root actions.
        return self._format_text_for_mode(
            self._resolve_codegen_mode(has_baseline=True)
        )

    def get_baseline_code_for_codegen(self, language: str | None = None) -> str:
        """Return the current on-disk project as the explicit base for patch prompts."""
        files = self._load_baseline_files_from_disk()
        if not files:
            return ""
        return format_ascendc_project_files(files)

    def get_per_task_requirement_text(self, *, language: str, target_gpu: str, phase: str) -> str:
        if phase == "optimize" and self.codegen_mode != "full":
            return ASCENDC_PATCH_FORMAT_TEXT
        return ASCENDC_CODE_FORMAT_TEXT

    def get_baseline_targets_text(self) -> str:
        if self.reference_latency_ms and self.reference_latency_ms > 0:
            return f"- reference_latency_ms: {self.reference_latency_ms:.6f}"
        return ""

    def code_for_world_model_from_raw(self, *, raw: Any, language: str) -> str:
        if isinstance(raw, dict):
            return format_ascendc_project_files({str(k): str(v or "") for k, v in raw.items()})
        text = str(raw or "")
        # Idempotency: when preview_parse_generated_code has already processed
        # this exact raw text, `_last_parsed_files` holds the post-patch state.
        # Re-applying the patch here would mismatch context against the advanced
        # baseline and silently degrade the WM excerpt to raw diff syntax.
        if self._last_parsed_raw == text and self._last_parsed_files is not None:
            return format_ascendc_project_files(self._last_parsed_files)
        if "<ascendc_patch>" in text or "<patch " in text:
            try:
                base_files = self._resolve_patch_base_files()
                files = parse_ascendc_project_patch(text, base_files=base_files)
                # Cache the applied result so subsequent patches diff against the
                # post-patch state, not the pre-patch baseline. We deliberately do
                # not set _last_parsed_raw — that field is owned by
                # _parse_codegen_response's preview/commit contract.
                self._last_parsed_files = dict(files)
                return format_ascendc_project_files(files)
            except ValueError:
                # Bad patch; let the WM see the raw response (truncated upstream).
                return text
        return text

    def _load_baseline_files_from_disk(self) -> dict[str, str]:
        if self.task_path is None or not self.task_path.is_dir():
            return {}
        sources = _collect_project_sources(self.task_path)
        return {s.path: s.content for s in sources}

    def _resolve_patch_base_files(self) -> dict[str, str]:
        if self._last_parsed_files is not None:
            return dict(self._last_parsed_files)
        return self._load_baseline_files_from_disk()

    def _parse_codegen_response(self, raw: Any) -> dict[str, str]:
        """Try patch first (when allowed), fall back to full container.

        Idempotent on identical input: if we already parsed this exact `raw`
        payload (typical when the retry framework calls preview_parse and then
        make_solution_from_generated_code with the same string), return the cached
        result instead of re-applying the patch (which would now mismatch because
        `_last_parsed_files` has already advanced).
        """
        text = str(raw or "")
        if self._last_parsed_raw == text and self._last_parsed_files is not None:
            return dict(self._last_parsed_files)

        looks_like_patch = "<ascendc_patch>" in text or "<patch " in text
        if self.codegen_mode != "full" and looks_like_patch:
            try:
                base_files = self._resolve_patch_base_files()
                files = parse_ascendc_project_patch(text, base_files=base_files)
                self._patch_failure_streak = 0
                self._last_parsed_files = dict(files)
                self._last_parsed_raw = text
                return files
            except ValueError:
                self._patch_failure_streak += 1
                if (
                    self.codegen_mode == "auto"
                    and self._patch_failure_streak >= self._max_patch_failures
                ):
                    print(
                        f"[WARN] ascendc patch parse failed {self._patch_failure_streak}"
                        f" times in a row; falling back to full codegen mode.",
                        file=sys.stderr,
                    )
                    self.codegen_mode = "full"
                raise

        files = parse_ascendc_project_files(text)
        self._patch_failure_streak = 0
        self._last_parsed_files = dict(files)
        self._last_parsed_raw = text
        return files

    def make_solution_from_generated_code(
        self,
        *,
        cleaned_code: Any,
        raw_code: Any,
        round_num: int,
        model_name: str,
        target_gpu: str,
        language: str,
    ) -> Solution:
        files = self._parse_codegen_response(raw_code if raw_code is not None else cleaned_code)
        sources = [SourceFile(path=path, content=content) for path, content in sorted(files.items())]
        return Solution(
            name=f"{model_name}_{self.name}_ascendc_optimized_r{int(round_num)}",
            definition=self.name,
            author=str(model_name),
            spec=BuildSpec(
                language=SupportedLanguages.ASCENDC,
                target_hardware=[str(target_gpu or "ascend")],
                entry_point=_default_entry_point(sources),
            ),
            sources=sources,
            description=f"{model_name} optimized AscendC project for {self.name} (round {int(round_num)})",
        )

    def get_agentic_definition_text(self, *, language: str) -> str:
        return self.get_definition_text(
            language=str(language),
            include_sources=False,
            include_format=False,
        )

    def overlay_solution_sources(
        self,
        *,
        project_dir: str | Path,
        solution: Solution | None,
    ) -> None:
        if solution is None:
            return
        root = Path(project_dir).expanduser().resolve()
        for src in solution.sources or []:
            rel = _normalize_rel_path(src.path)
            dest = root / rel
            dest.parent.mkdir(parents=True, exist_ok=True)
            dest.write_text(str(src.content or ""), encoding="utf-8")

    def _validate_agentic_changed_paths(
        self,
        *,
        project_dir: str | Path,
        changed_paths: list[str] | None,
    ) -> None:
        root = Path(project_dir).expanduser().resolve(strict=True)
        for path in changed_paths or []:
            rel = _normalize_rel_path(path)
            p = (root / rel).resolve(strict=False)
            if p != root and root not in p.parents:
                raise ValueError(f"changed path escapes project root: {rel}")
            parts = p.relative_to(root).parts
            if rel in NATIVE_RUNTIME_FILES or any(part in FORBIDDEN_AGENTIC_PATH_PARTS for part in parts):
                raise ValueError(f"forbidden agentic changed path: {rel}")
            if p.name not in ALLOWED_AGENTIC_SOURCE_NAMES and p.suffix.lower() not in ALLOWED_AGENTIC_SOURCE_SUFFIXES:
                raise ValueError(f"agentic changed path is not an allowed source/config file: {rel}")

    def _validate_solution_sources_under_root(self, root: Path, sources: list[SourceFile]) -> None:
        resolved_root = root.expanduser().resolve(strict=True)
        for src in sources:
            rel = _normalize_rel_path(src.path)
            p = (resolved_root / rel).resolve(strict=False)
            if p != resolved_root and resolved_root not in p.parents:
                raise ValueError(f"solution source escapes project root: {rel}")
            parts = p.relative_to(resolved_root).parts
            if rel in NATIVE_RUNTIME_FILES or any(part in FORBIDDEN_AGENTIC_PATH_PARTS for part in parts):
                raise ValueError(f"forbidden solution source path: {rel}")
            if p.name not in ALLOWED_AGENTIC_SOURCE_NAMES and p.suffix.lower() not in ALLOWED_AGENTIC_SOURCE_SUFFIXES:
                raise ValueError(f"solution source is not an allowed source/config file: {rel}")

    def make_solution_from_project_dir(
        self,
        *,
        project_dir: str | Path,
        changed_paths: list[str] | None,
        raw_agent_output: str,
        round_num: int,
        model_name: str,
        target_gpu: str,
        language: str,
    ) -> Solution:
        root = Path(project_dir).expanduser().resolve()
        self._validate_agentic_changed_paths(
            project_dir=root,
            changed_paths=changed_paths,
        )
        sources = _collect_project_sources(root)
        if not sources:
            raise ValueError(f"agentic project produced no source files: {root}")
        self._validate_solution_sources_under_root(root, sources)
        return Solution(
            name=f"{model_name}_{self.name}_ascendc_agentic_r{int(round_num)}",
            definition=self.name,
            author=str(model_name),
            spec=BuildSpec(
                language=SupportedLanguages.ASCENDC,
                target_hardware=[str(target_gpu or "ascend")],
                entry_point=_default_entry_point(sources),
            ),
            sources=sources,
            description=(
                f"{model_name} agentic AscendC project for {self.name} "
                f"(round {int(round_num)}): {str(raw_agent_output or '').strip()[:500]}"
            ),
        )

    def solution_from_raw_code_for_agentic(
        self,
        *,
        raw_code: str,
        round_num: int,
        model_name: str,
        target_gpu: str,
        language: str,
    ) -> Solution:
        files = parse_ascendc_project_files(raw_code)
        sources = [SourceFile(path=path, content=content) for path, content in sorted(files.items())]
        return Solution(
            name=f"{model_name}_{self.name}_ascendc_agentic_base_r{int(round_num)}",
            definition=self.name,
            author=str(model_name),
            spec=BuildSpec(
                language=SupportedLanguages.ASCENDC,
                target_hardware=[str(target_gpu or "ascend")],
                entry_point=_default_entry_point(sources),
            ),
            sources=sources,
            description=f"{model_name} AscendC agentic base for {self.name} (round {int(round_num)})",
        )

    def preview_parse_generated_code(self, *, raw_code: str) -> None:
        """Validate that `raw_code` will parse successfully.

        Called by `KernelGenerator._generate_code_from_prompt` immediately after
        `_clean_generated_code` returns. Raises ValueError on bad patch / bad full
        container so the retry framework can re-prompt the LLM. State updates
        (last-parsed-files cache, failure streak) happen inside `_parse_codegen_response`.
        Safe to call multiple times with the same raw_code: the second call will
        succeed-trivially because `_last_parsed_files` was updated by the first.
        """
        self._parse_codegen_response(raw_code)

    def get_solution(self, solution_name: str) -> Solution | None:
        ref = str(solution_name or "").strip()
        if not ref:
            return None
        if ref.lower() in {"base", "baseline", "current"}:
            sources = _collect_project_sources(self.task_path) if self.task_path and self.task_path.is_dir() else []
            if not sources:
                return None
            return Solution(
                name=ref,
                definition=self.name,
                author="baseline",
                spec=BuildSpec(
                    language=SupportedLanguages.ASCENDC,
                    target_hardware=["ascend"],
                    entry_point=_default_entry_point(sources),
                ),
                sources=sources,
                description="Baseline AscendC project loaded from task_path",
            )
        try:
            d = load_ksearch_solution_json(
                solution_ref=ref,
                definition_name=self.name,
                artifacts_dir=self.artifacts_dir,
            )
            return solution_from_json_dict(d)
        except Exception:
            return None

    def _prepare_workdir(self, solution: Solution) -> Path:
        root = Path(tempfile.mkdtemp(prefix=f"ksearch_ascendc_{self.name}_"))
        if self.task_path and self.task_path.is_dir():
            ignore = shutil.ignore_patterns(".git", ".ksearch", "build", "cmake-build-debug", "__pycache__")
            shutil.copytree(self.task_path, root, dirs_exist_ok=True, ignore=ignore)
        for src in solution.sources or []:
            rel = _normalize_rel_path(src.path)
            dest = root / rel
            dest.parent.mkdir(parents=True, exist_ok=True)
            dest.write_text(str(src.content or ""), encoding="utf-8")
        return root

    def _run_shell(self, cmd: str, *, cwd: Path) -> subprocess.CompletedProcess[str] | None:
        c = str(cmd or "").strip()
        if not c:
            return None
        c = self._remap_command_for_workdir(c, cwd=cwd)
        return subprocess.run(
            c,
            cwd=str(cwd),
            shell=True,
            capture_output=True,
            text=True,
            timeout=max(1, int(self.timeout_seconds)),
            check=False,
        )

    def _remap_command_for_workdir(self, cmd: str, *, cwd: Path) -> str:
        """Map original task absolute paths to the candidate/eval workdir.

        AscendC harness commands are often authored as absolute paths to
        ``task_path`` or its parent workspace (for example
        ``/repo/agent_workdir/scripts/evaluate_ascendc.sh``). During K-Search
        evaluation, those paths must resolve to the isolated candidate copy,
        not the original baseline workspace.
        """
        if self.task_path is None:
            return cmd
        try:
            src = self.task_path.resolve()
            dst = Path(cwd).expanduser().resolve()
            stop_src = self._command_remap_stop_path(src)
        except Exception:
            return cmd

        replacements: list[tuple[str, str]] = []
        while True:
            replacements.append((str(src), str(dst)))
            if src == stop_src or src.parent == src or dst.parent == dst:
                break
            src = src.parent
            dst = dst.parent

        rewritten = cmd
        for old, new in sorted(replacements, key=lambda pair: len(pair[0]), reverse=True):
            rewritten = rewritten.replace(old, new)
        return rewritten

    def _command_remap_stop_path(self, task_path: Path) -> Path:
        git_root = self._find_task_git_root(task_path)
        if git_root is not None:
            return git_root
        return task_path.parent

    @staticmethod
    def _find_task_git_root(path: Path) -> Path | None:
        if not path.exists():
            return None
        try:
            proc = subprocess.run(
                ["git", "-C", str(path), "rev-parse", "--show-toplevel"],
                text=True,
                capture_output=True,
                check=False,
            )
        except Exception:
            return None
        if proc.returncode != 0:
            return None
        root_text = (proc.stdout or "").strip()
        if not root_text:
            return None
        root = Path(root_text).expanduser().resolve()
        try:
            path.resolve().relative_to(root)
        except ValueError:
            return None
        return root if root.is_dir() else None

    @staticmethod
    def _append_command_log(logs: list[str], label: str, proc: subprocess.CompletedProcess[str] | None) -> None:
        if proc is None:
            logs.append(f"[{label}] skipped")
            return
        logs.append(f"[{label}] exit_code={proc.returncode}")
        out = (proc.stdout or "").strip()
        err = (proc.stderr or "").strip()
        if out:
            logs.append(f"[{label} stdout]\n{out}")
        if err:
            logs.append(f"[{label} stderr]\n{err}")

    def _run_benchmark_in_workdir(
        self,
        *,
        workdir: Path,
        round_num: int | None = None,
    ) -> EvalResult:
        logs: list[str] = []
        logs.append(f"[workdir] {workdir}")

        try:
            build = self._run_shell(self.build_cmd, cwd=workdir)
            self._append_command_log(logs, "build", build)
            if build is not None and build.returncode != 0:
                return self._record_eval(
                    EvalResult(
                        status="compile_failed",
                        log_excerpt=self._truncate_log(logs),
                        metrics={"workdir": str(workdir), "round": round_num},
                    )
                )

            test = self._run_shell(self.test_cmd, cwd=workdir)
            self._append_command_log(logs, "correctness", test)
            if test is not None and test.returncode != 0:
                return self._record_eval(
                    EvalResult(
                        status="failed",
                        log_excerpt=self._truncate_log(logs),
                        metrics={"workdir": str(workdir), "round": round_num},
                    )
                )

            bench = self._run_shell(self.bench_cmd, cwd=workdir)
            self._append_command_log(logs, "benchmark", bench)
            if bench is not None and bench.returncode != 0:
                return self._record_eval(
                    EvalResult(
                        status="benchmark_failed",
                        log_excerpt=self._truncate_log(logs),
                        metrics={"workdir": str(workdir), "round": round_num},
                    )
                )

            latency_ms = _parse_latency_ms(((bench.stdout or "") + "\n" + (bench.stderr or "")) if bench else "")
            if bench is not None and latency_ms is None:
                return self._record_eval(
                    EvalResult(
                        status="benchmark_failed",
                        log_excerpt=self._truncate_log(logs + ["[benchmark] missing latency_ms in output"]),
                        metrics={"workdir": str(workdir), "round": round_num},
                    )
                )

            score = 0.0
            speedup = None
            score_name = "score"
            if latency_ms and latency_ms > 0:
                if self.reference_latency_ms and self.reference_latency_ms > 0:
                    speedup = float(self.reference_latency_ms) / float(latency_ms)
                    score = speedup
                    score_name = "vs_baseline"
                else:
                    score = 1.0 / float(latency_ms)
                    score_name = "inv_latency"

            return self._record_eval(
                EvalResult(
                    status="passed",
                    latency_ms=latency_ms,
                    reference_latency_ms=self.reference_latency_ms,
                    mean_vs_baseline_factor=speedup,
                    speedup_factor=speedup,
                    log_excerpt=self._truncate_log(logs),
                    metrics={
                        "score": score,
                        "score_name": score_name,
                        "workdir": str(workdir),
                        "round": round_num,
                    },
                )
            )
        except subprocess.TimeoutExpired as e:
            logs.append(f"[timeout] command exceeded {self.timeout_seconds}s: {e}")
            return self._record_eval(
                EvalResult(
                    status="timeout",
                    log_excerpt=self._truncate_log(logs),
                    metrics={"workdir": str(workdir), "round": round_num},
                )
            )
        except Exception as e:
            logs.append(f"[error] {type(e).__name__}: {e}")
            return self._record_eval(
                EvalResult(
                    status="failed",
                    log_excerpt=self._truncate_log(logs),
                    metrics={"workdir": str(workdir), "round": round_num},
                )
            )

    def run_benchmark_in_project_dir(
        self,
        *,
        project_dir: str | Path,
        round_num: int | None = None,
        dump_traces: bool = False,
    ) -> EvalResult:
        del dump_traces
        workdir = Path(project_dir).expanduser().resolve()
        return self._run_benchmark_in_workdir(workdir=workdir, round_num=round_num)

    def run_benchmark(
        self,
        *,
        solution: Solution,
        config: Any = None,
        dump_traces: bool = False,
        round_num: int | None = None,
    ) -> EvalResult:
        del config, dump_traces
        workdir = self._prepare_workdir(solution)
        return self._run_benchmark_in_workdir(workdir=workdir, round_num=round_num)

    def _record_eval(self, result: EvalResult) -> EvalResult:
        self._last_eval = result
        return result

    @staticmethod
    def _truncate_log(logs: list[str], *, max_chars: int = 8000) -> str:
        text = sanitize_worktree_paths("\n\n".join(str(x) for x in logs))
        if len(text) > max_chars:
            return text[:max_chars] + "\n...<truncated>..."
        return text

    def seed_eval_for_base_solution(self, *, base_solution: Solution, config: Any = None) -> EvalResult:
        return self.run_benchmark(solution=base_solution, config=config, round_num=0)

    def get_last_round_trace_logs_for_prompt(self) -> str:
        return self._last_eval.log_excerpt if self._last_eval is not None else ""

    def get_last_round_passed_count(self) -> int:
        return 1 if self._last_eval is not None and self._last_eval.is_passed() else 0

    def get_last_round_total_workloads(self) -> int:
        return 1 if self._last_eval is not None else 0

    def get_config_for_logging(self) -> dict[str, Any]:
        return {
            "task_source": "ascendc",
            "name": self.name,
            "task_path": str(self.task_path) if self.task_path else None,
            "build_cmd": self.build_cmd,
            "test_cmd": self.test_cmd,
            "bench_cmd": self.bench_cmd,
            "reference_latency_ms": self.reference_latency_ms,
            "timeout_seconds": self.timeout_seconds,
        }

    def run_final_evaluation(
        self,
        *,
        solutions: list[Solution],
        config: Any = None,
        dump_traces: bool = False,
        workload_limit: int | None = None,
    ) -> dict[str, Any]:
        results = []
        for idx, sol in enumerate(solutions or [], start=1):
            result = self.run_benchmark(solution=sol, config=config, dump_traces=dump_traces, round_num=idx)
            results.append(
                {
                    "solution": sol.name,
                    "result": result.to_dict(include_log_excerpt=True, max_log_chars=8000),
                }
            )
        best = None
        for item in results:
            score = item["result"].get("metrics", {}).get("score")
            if not isinstance(score, (int, float)):
                continue
            if best is None or score > best.get("score", -1):
                best = {"solution": item["solution"], "score": score}
        return {
            "task": self.name,
            "task_source": "ascendc",
            "results": results,
            "best": best,
        }
