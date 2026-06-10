import json
import os
import sys
from types import SimpleNamespace

import pytest

import generate_kernels_and_eval as cli
from generate_kernels_and_eval import (
    _parse_strategy_form,
    _build_task_from_args,
    _resolve_llm_config_from_args,
    generate_and_evaluate,
    main,
)
from k_search.kernel_generators.checkpoint_v3 import StageCheckpointConfig


def test_resolve_llm_config_defaults_to_openai_and_reads_env_key(monkeypatch):
    monkeypatch.setenv("LLM_API_KEY", "env-key")
    args = SimpleNamespace(llm_provider=None, api_key=None)

    llm_provider, api_key = _resolve_llm_config_from_args(args)

    assert llm_provider == "openai"
    assert api_key == "env-key"


def test_resolve_llm_config_openai_requires_key(monkeypatch):
    monkeypatch.delenv("LLM_API_KEY", raising=False)
    args = SimpleNamespace(llm_provider="openai", api_key=None)

    with pytest.raises(ValueError, match="LLM_API_KEY"):
        _resolve_llm_config_from_args(args)


def test_resolve_llm_config_claude_agent_does_not_require_llm_api_key(monkeypatch):
    monkeypatch.delenv("LLM_API_KEY", raising=False)
    args = SimpleNamespace(llm_provider="claude-agent", api_key=None)

    llm_provider, api_key = _resolve_llm_config_from_args(args)

    assert llm_provider == "claude-agent"
    assert api_key is None


def test_resolve_llm_config_rejects_unknown_provider(monkeypatch):
    monkeypatch.setenv("LLM_API_KEY", "env-key")
    args = SimpleNamespace(llm_provider="other", api_key=None)

    with pytest.raises(ValueError, match="Unsupported LLM provider"):
        _resolve_llm_config_from_args(args)


def test_parse_strategy_form_only_accepts_natural_language():
    assert _parse_strategy_form("natural_language") == "natural_language"

    with pytest.raises(Exception, match="Only natural_language strategy form is supported"):
        _parse_strategy_form("dsl")


def test_build_task_from_args_constructs_ascendc_task(tmp_path):
    (tmp_path / "spec.md").write_text("AscendC vector add operator.", encoding="utf-8")
    args = SimpleNamespace(
        task_source="ascendc",
        task_path=str(tmp_path),
        local=None,
        definition="vec_add",
        ascendc_build_cmd="echo build",
        ascendc_test_cmd="echo test",
        ascendc_bench_cmd="echo latency_ms=1.0",
        ascendc_timeout_seconds=12,
        ascendc_reference_latency_ms=2.0,
        artifacts_dir=".ksearch-test",
    )

    task = _build_task_from_args(args)

    assert task.name == "vec_add"
    cfg = task.get_config_for_logging()
    assert cfg["task_source"] == "ascendc"
    assert cfg["build_cmd"] == "echo build"
    assert cfg["reference_latency_ms"] == 2.0


def test_generate_and_evaluate_sets_task_run_id_unconditionally(tmp_path, monkeypatch):
    monkeypatch.setenv("KSEARCH_TASK_ID", "task-meta")
    seen = {}

    class FakeTask:
        name = "lineage_task"

        def get_config_for_logging(self):
            return {}

        def run_final_evaluation(self, *, solutions, config, dump_traces, workload_limit):
            seen["final_eval_run_id"] = getattr(self, "_ksearch_run_id", None)
            return SimpleNamespace()

    class FakeKernelGenerator:
        def __init__(self, **kwargs):
            seen["generator_kwargs"] = kwargs

        def generate(self, *, task, max_opt_rounds, continue_from_solution=None):
            seen["generator_run_id"] = getattr(task, "_ksearch_run_id", None)
            return SimpleNamespace(name="fake_solution", description="")

    monkeypatch.setattr(
        "k_search.kernel_generators.kernel_generator.KernelGenerator",
        FakeKernelGenerator,
    )

    task = FakeTask()
    generate_and_evaluate(
        task,
        model_name="fake",
        base_url=None,
        api_key=None,
        language="ascendc",
        target_gpu="ascend_910b",
        max_opt_rounds=1,
        save_results=False,
        save_solutions=False,
        llm_provider="claude-agent",
        run_id="run-meta",
        artifacts_dir=str(tmp_path / "artifacts"),
    )

    assert seen["generator_run_id"] == "run-meta"
    assert seen["final_eval_run_id"] == "run-meta"
    assert task._ksearch_run_id == "run-meta"
    task_meta_path = tmp_path / "artifacts" / "lineage_task" / "task-meta" / "task_meta.json"
    run_meta_path = task_meta_path.parent / "runs" / "run-meta" / "run_meta.json"
    artifacts_meta_path = run_meta_path.parent / "artifacts" / "run_meta.json"
    task_meta = json.loads(task_meta_path.read_text(encoding="utf-8"))
    assert task_meta["task_id"] == "task-meta"
    assert task_meta["task_name"] == "lineage_task"
    assert task_meta["artifacts_dir"] == str(tmp_path / "artifacts")
    assert task_meta["start_time"]
    run_meta = json.loads(run_meta_path.read_text(encoding="utf-8"))
    assert run_meta["task_id"] == "task-meta"
    assert run_meta["run_id"] == "run-meta"
    assert not artifacts_meta_path.exists()


def test_generate_and_evaluate_saves_solution_under_explicit_run_id(tmp_path, monkeypatch):
    monkeypatch.setenv("KSEARCH_TASK_ID", "solution-task")
    monkeypatch.delenv("KSEARCH_RUN_ID", raising=False)

    class FakeTask:
        name = "solution_task"

        def get_config_for_logging(self):
            return {}

        def run_final_evaluation(self, *, solutions, config, dump_traces, workload_limit):
            return SimpleNamespace()

    class FakeKernelGenerator:
        def __init__(self, **kwargs):
            pass

        def generate(self, *, task, max_opt_rounds, continue_from_solution=None):
            return SimpleNamespace(name="fake_solution", description="")

    monkeypatch.setattr(
        "k_search.kernel_generators.kernel_generator.KernelGenerator",
        FakeKernelGenerator,
    )

    generate_and_evaluate(
        FakeTask(),
        model_name="fake",
        base_url=None,
        api_key=None,
        language="ascendc",
        target_gpu="ascend_910b",
        max_opt_rounds=1,
        save_results=False,
        save_solutions=True,
        llm_provider="claude-agent",
        run_id="explicit-save-run",
        artifacts_dir=str(tmp_path / "artifacts"),
    )

    task_root = tmp_path / "artifacts" / "solution_task" / "solution-task"
    runs = sorted(path.name for path in (task_root / "runs").iterdir() if path.is_dir())
    assert runs == ["explicit-save-run"]
    saved_solutions = list(
        (task_root / "runs" / "explicit-save-run" / "artifacts" / "solutions" / "solution_task").glob(
            "fake_solution_*.json"
        )
    )
    assert len(saved_solutions) == 1


def test_stage_checkpoint_config_from_args_requires_world_model():
    args = SimpleNamespace(
        checkpoint_v3=True,
        task_source="ascendc",
        language="ascendc",
        world_model=False,
        checkpoint_resume_claude_session=False,
        checkpoint_claude_session_required=False,
        checkpoint_subagent_resume=False,
        checkpoint_require_subagent_agent_id=False,
        checkpoint_enable_claude_file_checkpointing=False,
        checkpoint_session_store_kind="none",
        checkpoint_session_store_config=None,
        checkpoint_resume_stage_policy="next-pending",
        checkpoint_file_state_source="project-snapshot",
    )

    with pytest.raises(ValueError, match="--world-model"):
        cli._stage_checkpoint_config_from_args(args, llm_provider="claude-agent")


def test_stage_checkpoint_config_rejects_session_store_with_file_checkpointing():
    args = SimpleNamespace(
        checkpoint_v3=True,
        task_source="ascendc",
        language="ascendc",
        world_model=True,
        checkpoint_resume_claude_session=True,
        checkpoint_claude_session_required=False,
        checkpoint_subagent_resume=False,
        checkpoint_require_subagent_agent_id=False,
        checkpoint_enable_claude_file_checkpointing=True,
        checkpoint_session_store_kind="custom",
        checkpoint_session_store_config="store.json",
        checkpoint_resume_stage_policy="next-pending",
        checkpoint_file_state_source="project-snapshot",
    )

    with pytest.raises(ValueError, match="SessionStore cannot be combined"):
        cli._stage_checkpoint_config_from_args(args, llm_provider="claude-agent")


def test_generate_and_evaluate_passes_stage_checkpoint_config_to_world_model_generator(tmp_path, monkeypatch):
    monkeypatch.setenv("KSEARCH_TASK_ID", "checkpoint-task")
    seen = {}

    class FakeTask:
        name = "checkpoint_task"

        def get_config_for_logging(self):
            return {}

        def run_final_evaluation(self, *, solutions, config, dump_traces, workload_limit):
            return SimpleNamespace()

    class FakeWorldModelGenerator:
        def __init__(self, **kwargs):
            seen["generator_kwargs"] = kwargs

        def generate(
            self,
            *,
            task,
            max_opt_rounds,
            wm_stagnation_window,
            continue_from_solution=None,
            continue_from_world_model=None,
            continue_from_run=None,
            run_id=None,
        ):
            return SimpleNamespace(name="fake_solution", description="")

    monkeypatch.setattr(
        "k_search.kernel_generators.kernel_generator_world_model.WorldModelKernelGeneratorWithBaseline",
        FakeWorldModelGenerator,
    )
    config = StageCheckpointConfig(enabled=True)

    generate_and_evaluate(
        FakeTask(),
        model_name="fake",
        base_url=None,
        api_key=None,
        language="ascendc",
        target_gpu="ascend_910b",
        max_opt_rounds=1,
        save_results=False,
        save_solutions=False,
        llm_provider="claude-agent",
        run_id="checkpoint-run",
        artifacts_dir=str(tmp_path / "artifacts"),
        enable_world_model=True,
        stage_checkpoint_config=config,
    )

    assert seen["generator_kwargs"]["stage_checkpoint_config"] is config


def test_main_pins_task_id_when_absent(tmp_path, monkeypatch):
    captured = {}

    monkeypatch.delenv("KSEARCH_TASK_ID", raising=False)
    monkeypatch.delenv("KSEARCH_TASK_START", raising=False)
    monkeypatch.delenv("KSEARCH_RUN_ID", raising=False)
    monkeypatch.delenv("KSEARCH_RUN_START", raising=False)
    monkeypatch.delenv("KSEARCH_ARTIFACTS_DIR", raising=False)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "generate_kernels_and_eval.py",
            "--model-name",
            "fake-model",
            "--llm-provider",
            "claude-agent",
            "--artifacts-dir",
            str(tmp_path / "artifacts"),
        ],
    )
    monkeypatch.setattr(cli, "_resolve_llm_config_from_args", lambda args: ("claude-agent", None))
    monkeypatch.setattr(cli, "_build_task_from_args", lambda args: SimpleNamespace(name="main_task"))

    def fake_generate_and_evaluate(**kwargs):
        captured["task_id"] = os.environ.get("KSEARCH_TASK_ID")
        captured["run_id"] = os.environ.get("KSEARCH_RUN_ID")
        captured["artifacts_dir"] = os.environ.get("KSEARCH_ARTIFACTS_DIR")

    monkeypatch.setattr(cli, "generate_and_evaluate", fake_generate_and_evaluate)

    main()

    assert captured["task_id"]
    assert captured["run_id"]
    assert captured["artifacts_dir"] == str((tmp_path / "artifacts").resolve())
