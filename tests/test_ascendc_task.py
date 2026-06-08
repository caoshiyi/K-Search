import shlex
import sys
from pathlib import Path

import pytest

from k_search.tasks.task_base import BuildSpec, Solution, SourceFile, SupportedLanguages, code_from_solution
from k_search.kernel_generators.kernel_generator_prompts import get_prompt_from_definition_text
from k_search.kernel_generators.world_model_prompts import get_generate_code_from_action_prompt_from_text
from k_search.tasks.ascendc_task import (
    ASCENDC_CODE_FORMAT_TEXT,
    AscendCTask,
    format_ascendc_project_files,
    parse_ascendc_project_files,
)


def _py_cmd(code: str) -> str:
    return f"{shlex.quote(sys.executable)} -c {shlex.quote(code)}"


def test_parse_and_format_ascendc_project_files_round_trips_cpp_templates():
    raw = """
Before text is ignored.
<ascendc_project>
<file path="kernel.cpp">
#include <type_traits>
template <typename T>
__aicore__ inline T identity(T x) { return x; }
</file>
<file path="CMakeLists.txt">
cmake_minimum_required(VERSION 3.16)
</file>
</ascendc_project>
"""

    files = parse_ascendc_project_files(raw)

    assert files["kernel.cpp"].startswith("#include <type_traits>")
    assert "__aicore__ inline T identity" in files["kernel.cpp"]
    assert files["CMakeLists.txt"] == "cmake_minimum_required(VERSION 3.16)"

    formatted = format_ascendc_project_files(files)
    reparsed = parse_ascendc_project_files(formatted)

    assert reparsed == files


def test_code_from_solution_reconstructs_ascendc_multifile_container():
    solution = Solution(
        name="candidate",
        definition="vec_add",
        author="test",
        spec=BuildSpec(
            language=SupportedLanguages.ASCENDC,
            target_hardware=["ascend_910b"],
            entry_point="kernel.cpp::AddCustom",
        ),
        sources=[
            SourceFile(path="kernel.cpp", content="void AddCustom() {}"),
            SourceFile(path="tiling.cpp", content="void TilingFunc() {}"),
        ],
    )

    code_dict, raw = code_from_solution("ascendc", solution)

    assert code_dict == {
        "kernel.cpp": "void AddCustom() {}",
        "tiling.cpp": "void TilingFunc() {}",
    }
    assert parse_ascendc_project_files(raw) == code_dict


def test_ascendc_task_makes_solution_from_generated_multifile_project(tmp_path):
    (tmp_path / "spec.md").write_text("Vector add AscendC operator.", encoding="utf-8")
    task = AscendCTask(task_path=tmp_path, definition_name="vec_add")
    raw = format_ascendc_project_files(
        {
            "kernel.cpp": "extern \"C\" __global__ __aicore__ void AddCustom() {}",
            "tiling.cpp": "int TilingFunc() { return 0; }",
        }
    )

    solution = task.make_solution_from_generated_code(
        cleaned_code=raw,
        raw_code=raw,
        round_num=3,
        model_name="model",
        target_gpu="ascend_910b",
        language="ascendc",
    )

    assert solution.definition == "vec_add"
    assert solution.spec.language == SupportedLanguages.ASCENDC
    assert solution.spec.target_hardware == ["ascend_910b"]
    assert solution.get_entry_path() == "kernel.cpp"
    assert {src.path for src in solution.sources} == {"kernel.cpp", "tiling.cpp"}


def test_ascendc_task_run_benchmark_executes_build_test_bench_and_scores_latency(tmp_path):
    solution = Solution(
        name="candidate",
        definition="vec_add",
        author="test",
        spec=BuildSpec(
            language=SupportedLanguages.ASCENDC,
            target_hardware=["ascend_910b"],
            entry_point="kernel.cpp::AddCustom",
        ),
        sources=[SourceFile(path="kernel.cpp", content="int main() { return 0; }\n")],
    )
    task = AscendCTask(
        task_path=tmp_path,
        definition_name="vec_add",
        build_cmd=_py_cmd("from pathlib import Path; assert Path('kernel.cpp').exists(); print('build ok')"),
        test_cmd=_py_cmd("print('correctness passed')"),
        bench_cmd=_py_cmd("print('latency_ms=2.5')"),
        reference_latency_ms=5.0,
        timeout_seconds=30,
    )

    result = task.run_benchmark(solution=solution, round_num=1)

    assert result.status == "passed"
    assert result.latency_ms == 2.5
    assert result.reference_latency_ms == 5.0
    assert result.mean_vs_baseline_factor == 2.0
    assert result.metrics["score"] == 2.0
    assert result.metrics["workdir"]
    assert "build ok" in result.log_excerpt
    assert "correctness passed" in result.log_excerpt


def test_ascendc_task_run_benchmark_in_project_dir_uses_existing_project(tmp_path):
    project = tmp_path / "candidate"
    project.mkdir()
    (project / "kernel.cpp").write_text("int main() { return 0; }\n", encoding="utf-8")
    baseline = tmp_path / "baseline"
    baseline.mkdir()
    (baseline / "kernel.cpp").write_text("broken\n", encoding="utf-8")
    task = AscendCTask(
        task_path=baseline,
        definition_name="vec_add",
        build_cmd=_py_cmd(
            "from pathlib import Path; "
            "assert Path('kernel.cpp').read_text() == 'int main() { return 0; }\\n'; "
            "print('build ok')"
        ),
        test_cmd=_py_cmd("print('correctness passed')"),
        bench_cmd=_py_cmd("print('latency_ms=2.5')"),
        reference_latency_ms=5.0,
        timeout_seconds=30,
    )

    result = task.run_benchmark_in_project_dir(project_dir=project, round_num=7)

    assert result.status == "passed"
    assert result.latency_ms == 2.5
    assert result.metrics["score"] == 2.0
    assert result.metrics["workdir"] == str(project.resolve())
    assert result.metrics["round"] == 7
    assert "build ok" in result.log_excerpt


def test_ascendc_task_command_remap_does_not_rewrite_unrelated_shared_ancestor():
    task = AscendCTask(
        task_path="/mnt/workspace/cv_agent_adv/agent_workdir/flash_attention",
        definition_name="flash_attention",
    )
    command = (
        "bash /mnt/workspace/K-Search/scripts/driver.sh && "
        "bash /mnt/workspace/cv_agent_adv/agent_workdir/scripts/evaluate_ascendc.sh flash_attention basic"
    )

    remapped = task._remap_command_for_workdir(
        command,
        cwd=Path("/tmp/ksearch_eval_ab12cd/project/agent_workdir/flash_attention"),
    )

    assert "/mnt/workspace/K-Search/scripts/driver.sh" in remapped
    assert "/tmp/ksearch_eval_ab12cd/project/agent_workdir/scripts/evaluate_ascendc.sh" in remapped
    assert "/mnt/workspace/cv_agent_adv/agent_workdir/scripts/evaluate_ascendc.sh" not in remapped


def test_ascendc_task_run_benchmark_reports_compile_failure(tmp_path):
    solution = Solution(
        name="candidate",
        definition="vec_add",
        author="test",
        spec=BuildSpec(
            language=SupportedLanguages.ASCENDC,
            target_hardware=["ascend_910b"],
            entry_point="kernel.cpp::AddCustom",
        ),
        sources=[SourceFile(path="kernel.cpp", content="broken\n")],
    )
    task = AscendCTask(
        task_path=tmp_path,
        definition_name="vec_add",
        build_cmd=_py_cmd("import sys; print('compile failed'); sys.exit(2)"),
        test_cmd=_py_cmd("raise SystemExit('should not run')"),
        bench_cmd=_py_cmd("raise SystemExit('should not run')"),
        timeout_seconds=30,
    )

    result = task.run_benchmark(solution=solution, round_num=1)

    assert result.status == "compile_failed"
    assert result.score() == -1.0
    assert "compile failed" in result.log_excerpt


def test_ascendc_prompt_builders_accept_ascendc_language():
    prompt = get_prompt_from_definition_text(
        "ascendc",
        "Implement vector add.",
        "ascend_910b",
        per_task_requirement=ASCENDC_CODE_FORMAT_TEXT,
    )
    action_prompt = get_generate_code_from_action_prompt_from_text(
        "ascendc",
        definition_text="Implement vector add.",
        base_code="<ascendc_project></ascendc_project>",
        action_text="Increase tile length within UB capacity.",
        code_format=ASCENDC_CODE_FORMAT_TEXT,
        target_gpu="ascend_910b",
    )

    assert "AscendC" in prompt
    assert "ascend_910b" in prompt
    assert "<ascendc_project>" in prompt
    assert "Increase tile length" in action_prompt


def test_ascendc_task_defaults_to_auto_codegen_mode():
    task = AscendCTask(task_path=None, definition_name="x")
    assert task.codegen_mode == "auto"


def test_ascendc_task_reads_codegen_mode_from_env(monkeypatch):
    monkeypatch.setenv("KSEARCH_ASCENDC_CODEGEN_MODE", "full")
    task = AscendCTask(task_path=None, definition_name="x")
    assert task.codegen_mode == "full"


def test_ascendc_task_explicit_codegen_mode_overrides_env(monkeypatch):
    monkeypatch.setenv("KSEARCH_ASCENDC_CODEGEN_MODE", "full")
    task = AscendCTask(task_path=None, definition_name="x", codegen_mode="patch")
    assert task.codegen_mode == "patch"


def test_ascendc_task_invalid_codegen_mode_raises():
    with pytest.raises(ValueError):
        AscendCTask(task_path=None, definition_name="x", codegen_mode="bogus")


def test_get_optimization_prompt_uses_patch_format_when_baseline_available(tmp_path):
    (tmp_path / "spec.md").write_text("Vector add.", encoding="utf-8")
    task = AscendCTask(task_path=tmp_path, definition_name="x", codegen_mode="patch")
    prompt = task.get_optimization_prompt(
        language="ascendc",
        target_gpu="ascend_910b",
        trace_logs="ok",
        current_code='<ascendc_project><file path="kernel.cpp">int a=1;</file></ascendc_project>',
    )
    assert "<ascendc_patch>" in prompt
    assert "@@" in prompt
    # We must NOT instruct the model to emit the full container in patch mode.
    assert "Return only the full AscendC multi-file container" not in prompt


def test_get_optimization_prompt_falls_back_to_full_when_current_code_empty(tmp_path):
    (tmp_path / "spec.md").write_text("Vector add.", encoding="utf-8")
    task = AscendCTask(task_path=tmp_path, definition_name="x", codegen_mode="patch")
    prompt = task.get_optimization_prompt(
        language="ascendc",
        target_gpu="ascend_910b",
        trace_logs="ok",
        current_code="",
    )
    assert "<ascendc_project>" in prompt
    assert "<ascendc_patch>" not in prompt


def test_get_optimization_prompt_in_full_mode_never_emits_patch_format(tmp_path):
    (tmp_path / "spec.md").write_text("Vector add.", encoding="utf-8")
    task = AscendCTask(task_path=tmp_path, definition_name="x", codegen_mode="full")
    prompt = task.get_optimization_prompt(
        language="ascendc",
        target_gpu="ascend_910b",
        trace_logs="ok",
        current_code="<ascendc_project></ascendc_project>",
    )
    assert "<ascendc_patch>" not in prompt
    assert "<ascendc_project>" in prompt


def test_get_code_format_text_in_patch_mode_returns_patch_format(tmp_path):
    task = AscendCTask(task_path=tmp_path, definition_name="x", codegen_mode="patch")
    fmt = task.get_code_format_text(language="ascendc", target_gpu="ascend_910b")
    assert "<ascendc_patch>" in fmt


def test_get_baseline_code_for_codegen_formats_disk_project(tmp_path):
    kernel_dir = tmp_path / "kernel"
    kernel_dir.mkdir()
    (kernel_dir / "foo.h").write_text("int a = 1;\n", encoding="utf-8")
    task = AscendCTask(task_path=tmp_path, definition_name="x", codegen_mode="patch")

    baseline_code = task.get_baseline_code_for_codegen(language="ascendc")

    assert "<ascendc_project>" in baseline_code
    assert 'file path="kernel/foo.h"' in baseline_code
    assert "int a = 1;" in baseline_code


def test_world_model_codegen_definition_omits_sources_and_format_when_base_is_explicit(tmp_path):
    from k_search.kernel_generators.kernel_generator_world_model import (
        _definition_text_for_codegen_prompt,
    )

    (tmp_path / "spec.md").write_text("Vector add spec.", encoding="utf-8")
    kernel_dir = tmp_path / "kernel"
    kernel_dir.mkdir()
    (kernel_dir / "foo.h").write_text("int duplicated_source = 1;\n", encoding="utf-8")
    task = AscendCTask(task_path=tmp_path, definition_name="x", codegen_mode="patch")

    definition_text = _definition_text_for_codegen_prompt(
        task,
        language="ascendc",
        has_explicit_base_code=True,
    )

    assert "Vector add spec." in definition_text
    assert "Existing project source excerpts" not in definition_text
    assert "int duplicated_source" not in definition_text
    assert "<ascendc_project>" not in definition_text
    assert "<ascendc_patch>" not in definition_text


def test_make_solution_accepts_patch_response_against_disk_baseline(tmp_path):
    kernel_dir = tmp_path / "kernel"
    kernel_dir.mkdir()
    (kernel_dir / "foo.h").write_text("int a = 1;\nint b = 2;\nint c = 3;\n", encoding="utf-8")
    task = AscendCTask(task_path=tmp_path, definition_name="x", codegen_mode="patch")

    raw_patch = (
        "<ascendc_patch>\n"
        '<patch path="kernel/foo.h">\n'
        "@@ -1,3 +1,3 @@\n"
        " int a = 1;\n"
        "-int b = 2;\n"
        "+int b = 22;\n"
        " int c = 3;\n"
        "</patch>\n"
        "</ascendc_patch>\n"
    )
    solution = task.make_solution_from_generated_code(
        cleaned_code=raw_patch,
        raw_code=raw_patch,
        round_num=2,
        model_name="m",
        target_gpu="ascend_910b",
        language="ascendc",
    )
    foo = next(s for s in solution.sources if s.path == "kernel/foo.h")
    assert "int b = 22;" in foo.content


def test_make_solution_falls_back_to_full_container_when_response_is_not_a_patch(tmp_path):
    kernel_dir = tmp_path / "kernel"
    kernel_dir.mkdir()
    (kernel_dir / "foo.h").write_text("int a = 1;\n", encoding="utf-8")
    task = AscendCTask(task_path=tmp_path, definition_name="x", codegen_mode="auto")

    raw_full = format_ascendc_project_files({"kernel/foo.h": "int a = 999;\n"})
    solution = task.make_solution_from_generated_code(
        cleaned_code=raw_full,
        raw_code=raw_full,
        round_num=2,
        model_name="m",
        target_gpu="ascend_910b",
        language="ascendc",
    )
    foo = next(s for s in solution.sources if s.path == "kernel/foo.h")
    assert foo.content == "int a = 999;"


def test_make_solution_records_patch_failure_streak_and_auto_falls_back(tmp_path):
    kernel_dir = tmp_path / "kernel"
    kernel_dir.mkdir()
    (kernel_dir / "foo.h").write_text("alpha\nbeta\ngamma\n", encoding="utf-8")
    task = AscendCTask(task_path=tmp_path, definition_name="x", codegen_mode="auto")

    bogus_patch = (
        "<ascendc_patch>\n"
        '<patch path="kernel/foo.h">\n'
        "@@ -1,3 +1,3 @@\n"
        " alpha\n"
        "-WRONG_CONTEXT\n"
        "+gamma\n"
        " beta\n"
        "</patch>\n"
        "</ascendc_patch>\n"
    )
    # Three failures should trigger fallback to full mode.
    for _ in range(3):
        with pytest.raises(ValueError):
            task.make_solution_from_generated_code(
                cleaned_code=bogus_patch,
                raw_code=bogus_patch,
                round_num=2,
                model_name="m",
                target_gpu="ascend_910b",
                language="ascendc",
            )
    assert task.codegen_mode == "full"
    assert task._patch_failure_streak >= 3


def test_kernel_generator_retries_ascendc_on_bad_patch_then_succeeds(tmp_path, monkeypatch, capsys):
    """A flaky LLM that returns a bad patch on attempt 1 and a good one on attempt 2
    should produce a parseable solution thanks to the widened retry framework.
    """
    from k_search.kernel_generators.kernel_generator import KernelGenerator

    kernel_dir = tmp_path / "kernel"
    kernel_dir.mkdir()
    (kernel_dir / "foo.h").write_text("alpha\nbeta\ngamma\n", encoding="utf-8")
    task = AscendCTask(task_path=tmp_path, definition_name="x", codegen_mode="patch")

    bad_patch = (
        "<ascendc_patch>\n"
        '<patch path="kernel/foo.h">\n'
        "@@ -1,3 +1,3 @@\n"
        " alpha\n"
        "-WRONG\n"
        "+BETA\n"
        " gamma\n"
        "</patch>\n"
        "</ascendc_patch>\n"
    )
    good_patch = (
        "<ascendc_patch>\n"
        '<patch path="kernel/foo.h">\n'
        "@@ -1,3 +1,3 @@\n"
        " alpha\n"
        "-beta\n"
        "+BETA\n"
        " gamma\n"
        "</patch>\n"
        "</ascendc_patch>\n"
    )

    class FlakyClient:
        def __init__(self):
            self.calls = 0

        def generate(self, prompt):
            self.calls += 1
            return bad_patch if self.calls == 1 else good_patch

    gen = KernelGenerator(
        model_name="fake",
        language="ascendc",
        target_gpu="ascend_910b",
        llm_client=FlakyClient(),
    )
    result = gen._generate_code_from_prompt("ignored prompt", task=task)
    output = capsys.readouterr().out
    assert "[LLM] codegen request" in output
    assert "language=ascendc" in output
    assert "attempt=1/5" in output
    assert "prompt_chars=14" in output
    assert "[WARN] AscendC parse failed" in output
    assert "error_type=ValueError" in output
    assert "BETA" in result["raw"]
    # Calling make_solution_from_generated_code with the same raw must NOT re-fail
    # (idempotency cache).
    sol = task.make_solution_from_generated_code(
        cleaned_code=result["cleaned"],
        raw_code=result["raw"],
        round_num=2,
        model_name="fake",
        target_gpu="ascend_910b",
        language="ascendc",
    )
    foo = next(s for s in sol.sources if s.path == "kernel/foo.h")
    assert "BETA" in foo.content


def test_kernel_generator_retries_ascendc_provider_timeout_before_failing(tmp_path, capsys):
    from k_search.kernel_generators.kernel_generator import KernelGenerator

    task = AscendCTask(task_path=tmp_path, definition_name="x", codegen_mode="patch")

    class TimeoutClient:
        def __init__(self):
            self.calls = 0
            self.prompts = []

        def generate(self, prompt):
            self.calls += 1
            self.prompts.append(prompt)
            raise TimeoutError("Claude Agent SDK provider timed out after 1200s")

    client = TimeoutClient()
    gen = KernelGenerator(
        model_name="fake",
        language="ascendc",
        target_gpu="ascend_910b",
        llm_client=client,
    )

    with pytest.raises(TimeoutError, match="timed out after 1200s"):
        gen._generate_code_from_prompt("ignored prompt", task=task)
    output = capsys.readouterr().out
    assert "[LLM] codegen request" in output
    assert "language=ascendc" in output
    assert "attempt=1/5" in output
    assert "attempt=5/5" in output
    assert "prompt_chars=14" in output
    assert "[ERROR] LLM codegen timeout" in output
    assert "error_type=TimeoutError" in output
    assert client.calls == 5
    assert "Previous code generation attempts failed before evaluation" in client.prompts[1]
    assert "timed out after 1200s" in client.prompts[1]


def test_code_for_world_model_from_raw_returns_applied_code_after_preview_parse(tmp_path):
    """After preview_parse_generated_code processes a patch, code_for_world_model_from_raw
    on the same raw text must NOT re-apply the patch (which would silently fail because
    the cached baseline has advanced) — it must return the post-patch <ascendc_project>
    container so the world model sees expanded code, not diff syntax.
    """
    kernel_dir = tmp_path / "kernel"
    kernel_dir.mkdir()
    (kernel_dir / "foo.h").write_text("alpha\nbeta\ngamma\n", encoding="utf-8")
    task = AscendCTask(task_path=tmp_path, definition_name="x", codegen_mode="patch")

    raw_patch = (
        "<ascendc_patch>\n"
        '<patch path="kernel/foo.h">\n'
        "@@ -1,3 +1,3 @@\n"
        " alpha\n"
        "-beta\n"
        "+BETA\n"
        " gamma\n"
        "</patch>\n"
        "</ascendc_patch>\n"
    )
    # Step 1: retry-framework path — preview parses the patch and caches the result.
    task.preview_parse_generated_code(raw_code=raw_patch)
    # Step 2: WM loop path — same raw text fed to code_for_world_model_from_raw.
    wm_excerpt = task.code_for_world_model_from_raw(raw=raw_patch, language="ascendc")
    # The excerpt must be the applied result rendered as <ascendc_project>, NOT raw diff.
    assert "<ascendc_project>" in wm_excerpt
    assert "<ascendc_patch>" not in wm_excerpt
    assert "BETA" in wm_excerpt


def test_ascendc_task_overlays_solution_sources_into_project_dir(tmp_path):
    task = AscendCTask(task_path=tmp_path, definition_name="x")
    solution = Solution(
        name="candidate",
        definition="x",
        author="test",
        spec=BuildSpec(
            language=SupportedLanguages.ASCENDC,
            target_hardware=["ascend_910b"],
            entry_point="kernel/foo.h::run",
        ),
        sources=[
            SourceFile(path="kernel/foo.h", content="overlaid\n"),
            SourceFile(path="tiling.cpp", content="tiling\n"),
        ],
    )

    task.overlay_solution_sources(project_dir=tmp_path, solution=solution)

    assert (tmp_path / "kernel" / "foo.h").read_text(encoding="utf-8") == "overlaid\n"
    assert (tmp_path / "tiling.cpp").read_text(encoding="utf-8") == "tiling\n"


def test_ascendc_task_overlay_preserves_unlisted_project_files(tmp_path):
    (tmp_path / "kernel").mkdir()
    (tmp_path / "kernel" / "stale.h").write_text("stale\n", encoding="utf-8")
    (tmp_path / "kernel" / "large_header.hpp").write_text("x" * 210_000, encoding="utf-8")
    (tmp_path / "README.md").write_text("keep docs\n", encoding="utf-8")
    task = AscendCTask(task_path=tmp_path, definition_name="x")
    solution = Solution(
        name="candidate",
        definition="x",
        author="test",
        spec=BuildSpec(
            language=SupportedLanguages.ASCENDC,
            target_hardware=["ascend_910b"],
            entry_point="kernel/fresh.h::run",
        ),
        sources=[SourceFile(path="kernel/fresh.h", content="fresh\n")],
    )

    task.overlay_solution_sources(project_dir=tmp_path, solution=solution)

    assert (tmp_path / "kernel" / "stale.h").read_text(encoding="utf-8") == "stale\n"
    assert (tmp_path / "kernel" / "large_header.hpp").exists()
    assert (tmp_path / "kernel" / "fresh.h").read_text(encoding="utf-8") == "fresh\n"
    assert (tmp_path / "README.md").read_text(encoding="utf-8") == "keep docs\n"


def test_make_solution_from_project_dir_scans_sources_and_ignores_build_dirs(tmp_path):
    (tmp_path / "kernel").mkdir()
    (tmp_path / "kernel" / "foo.h").write_text("int x = 1;\n", encoding="utf-8")
    (tmp_path / "kernel.cpp").write_text("void run() {}\n", encoding="utf-8")
    (tmp_path / "build").mkdir()
    (tmp_path / "build" / "generated.cpp").write_text("ignored\n", encoding="utf-8")
    (tmp_path / ".git").mkdir()
    (tmp_path / ".git" / "config").write_text("ignored\n", encoding="utf-8")
    task = AscendCTask(task_path=tmp_path, definition_name="x")

    solution = task.make_solution_from_project_dir(
        project_dir=tmp_path,
        changed_paths=["kernel/foo.h", "kernel.cpp"],
        raw_agent_output="changed kernel files",
        round_num=4,
        model_name="claude-sonnet-4-6",
        target_gpu="ascend_910b",
        language="ascendc",
    )

    assert solution.definition == "x"
    assert solution.spec.language == SupportedLanguages.ASCENDC
    assert solution.spec.target_hardware == ["ascend_910b"]
    assert solution.get_entry_path() == "kernel.cpp"
    assert {src.path for src in solution.sources} == {"kernel/foo.h", "kernel.cpp"}
    assert "changed kernel files" in str(solution.description)


def test_make_solution_from_project_dir_rejects_forbidden_changed_path(tmp_path):
    (tmp_path / "kernel.cpp").write_text("void run() {}\n", encoding="utf-8")
    task = AscendCTask(task_path=tmp_path, definition_name="x")

    with pytest.raises(ValueError, match="forbidden agentic changed path"):
        task.make_solution_from_project_dir(
            project_dir=tmp_path,
            changed_paths=["build/generated.cpp"],
            raw_agent_output="changed build output",
            round_num=1,
            model_name="claude",
            target_gpu="ascend_910b",
            language="ascendc",
        )


def test_get_agentic_definition_text_omits_source_containers(tmp_path):
    (tmp_path / "spec.md").write_text("Vector add spec.", encoding="utf-8")
    (tmp_path / "kernel.cpp").write_text("int source_should_not_appear = 1;\n", encoding="utf-8")
    task = AscendCTask(task_path=tmp_path, definition_name="x")

    text = task.get_agentic_definition_text(language="ascendc")

    assert "Vector add spec." in text
    assert "Existing project source excerpts" not in text
    assert "source_should_not_appear" not in text
    assert "<ascendc_project>" not in text


def test_solution_from_raw_code_for_agentic_parses_full_container_without_advancing_patch_state(tmp_path):
    task = AscendCTask(task_path=tmp_path, definition_name="x")
    raw = format_ascendc_project_files({"kernel.cpp": "void run() {}\n"})

    solution = task.solution_from_raw_code_for_agentic(
        raw_code=raw,
        round_num=8,
        model_name="claude",
        target_gpu="ascend_910b",
        language="ascendc",
    )

    assert solution.definition == "x"
    assert {src.path for src in solution.sources} == {"kernel.cpp"}
    assert task._last_parsed_files is None
    assert task._last_parsed_raw is None
    assert task._patch_failure_streak == 0


def test_truncate_log_sanitizes_worktree_paths():
    logs = [
        "[workdir] /tmp/ksearch_agentic_worktree_02ut4r9r/tile2asc/mqa",
        "-- Build files: /tmp/ksearch_agentic_worktree_02ut4r9r/kernel/build",
    ]
    out = AscendCTask._truncate_log(logs)
    assert "ksearch_agentic_worktree_02ut4r9r" not in out
    assert "<PROJECT_ROOT>" in out


def test_truncate_log_sanitizes_before_truncation():
    # 占位符必须在截断前替换,不能被 max_chars 拦腰截断破坏。
    logs = ["/tmp/ksearch_agentic_worktree_02ut4r9r/" + "x" * 50]
    out = AscendCTask._truncate_log(logs, max_chars=20)
    assert "ksearch_agentic_worktree_02ut4r9r" not in out
    assert out.startswith("<PROJECT_ROOT>")
