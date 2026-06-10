import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from k_search.tasks.task_base import BuildSpec, EvalResult, Solution, SourceFile, SupportedLanguages


def _solution(name: str = "sol", content: str = "kernel") -> Solution:
    return Solution(
        name=name,
        definition="vec_add",
        author="test",
        spec=BuildSpec(
            language=SupportedLanguages.ASCENDC,
            target_hardware=["ascend_910b"],
            entry_point="kernel.cpp::run",
        ),
        sources=[SourceFile(path="kernel.cpp", content=content)],
    )


def _world_model_json() -> str:
    return json.dumps(
        {
            "schema_version": 1,
            "task": {"definition_name": "vec_add"},
            "decision_tree": {
                "root_id": "root",
                "active_leaf_id": "s1",
                "nodes": [
                    {
                        "node_id": "root",
                        "parent_id": None,
                        "children": ["s1"],
                        "status": "open",
                    },
                    {
                        "node_id": "s1",
                        "parent_id": "root",
                        "children": [],
                        "status": "open",
                    },
                ],
            },
        }
    )


def test_cycle_checkpoint_writes_manifest_latest_and_restores_to_target_run(tmp_path):
    from k_search.kernel_generators.checkpoint import CheckpointConfig, CheckpointManager

    source_db = tmp_path / "source_solution_db.jsonl"
    source_db.write_text('{"solution_id": "abc"}\n', encoding="utf-8")
    best_eval = EvalResult(
        status="passed",
        latency_ms=0.5,
        reference_latency_ms=1.0,
        metrics={"score": 2.0, "score_name": "vs_baseline"},
    )

    manager = CheckpointManager(
        artifacts_dir=tmp_path / "artifacts",
        task_name="vec_add",
        task_id="task-1",
        run_id="run-1",
        config=CheckpointConfig(enabled=True, keep=3),
    )

    manifest_path = manager.save_cycle_checkpoint(
        task=SimpleNamespace(
            name="vec_add",
            task_source="ascendc",
            task_path=tmp_path / "op_project",
        ),
        round_index=3,
        cycle_start_round=1,
        action_node_id="s1",
        next_round=4,
        world_model_json=_world_model_json(),
        solution_db_path=source_db,
        best_solution=_solution("best", "best-code"),
        best_eval=best_eval,
        best_score=2.0,
        current_solution=_solution("current", "current-code"),
        current_eval=best_eval,
        llm_provider="claude-agent",
        model_name="claude-sonnet-4-6",
        language="ascendc",
        target_gpu="ascend_910b",
        max_opt_rounds=20,
        wm_stagnation_window=5,
        wm_max_difficulty=4,
    )

    assert manifest_path.name == "manifest.json"
    checkpoint_dir = manifest_path.parent
    assert checkpoint_dir.name == "ckpt_000001_cycle_r0003"
    latest = json.loads((checkpoint_dir.parent / "latest.json").read_text(encoding="utf-8"))
    assert latest["latest_checkpoint_id"] == checkpoint_dir.name
    assert latest["latest_checkpoint_path"] == f"{checkpoint_dir.name}/manifest.json"
    assert not any(path.name.endswith(".tmp") for path in checkpoint_dir.parent.iterdir())

    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    assert manifest["checkpoint_kind"] == "cycle_boundary"
    assert manifest["search"]["round_index"] == 3
    assert manifest["paths"]["world_model"] == "world_model/world_model.json"
    assert "world_model/world_model.json" in manifest["integrity"]["files"]

    ref = manager.resolve("latest")
    restored = manager.restore_to_run(ref, target_run_id="run-2")

    target_artifacts = (
        tmp_path
        / "artifacts"
        / "vec_add"
        / "task-1"
        / "runs"
        / "run-2"
        / "artifacts"
    )
    assert restored.checkpoint_id == checkpoint_dir.name
    assert restored.start_round == 4
    assert restored.resume_in_cycle is False
    assert restored.best_solution is not None
    assert restored.best_solution.name == "best"
    assert restored.current_solution is not None
    assert restored.current_solution.name == "current"
    assert restored.best_eval is not None
    assert restored.best_eval.score() == 2.0
    assert restored.world_model_path == target_artifacts / "world_model" / "world_model.json"
    assert restored.solution_db_path == target_artifacts / "world_model" / "solution_db.jsonl"
    assert restored.world_model_path.read_text(encoding="utf-8") == _world_model_json()
    assert restored.solution_db_path.read_text(encoding="utf-8") == '{"solution_id": "abc"}\n'


def test_attempt_checkpoint_writes_v2_manifest_and_restores_in_cycle(tmp_path):
    from k_search.kernel_generators.checkpoint import CheckpointConfig, CheckpointManager

    source_db = tmp_path / "source_solution_db.jsonl"
    source_db.write_text('{"solution_id": "attempt"}\n', encoding="utf-8")
    attempt_eval = EvalResult(
        status="failed",
        log_excerpt="compile error",
        metrics={"score": -1.0, "score_name": "attempt"},
    )

    manager = CheckpointManager(
        artifacts_dir=tmp_path / "artifacts",
        task_name="vec_add",
        task_id="task-1",
        run_id="run-1",
        config=CheckpointConfig(enabled=True, every="attempt", keep=3),
    )

    manifest_path = manager.save_attempt_checkpoint(
        task=SimpleNamespace(
            name="vec_add",
            task_source="ascendc",
            task_path=tmp_path / "op_project",
        ),
        round_index=7,
        cycle_start_round=6,
        attempt_idx=2,
        action_node_id="s1",
        next_round=8,
        next_attempt_idx=3,
        world_model_json=_world_model_json(),
        solution_db_path=source_db,
        best_solution=_solution("best", "best-code"),
        best_eval=EvalResult(status="passed", latency_ms=0.5, metrics={"score": 2.0}),
        best_score=2.0,
        current_solution=_solution("attempt", "attempt-code"),
        current_eval=attempt_eval,
        last_solution=_solution("last", "last-code"),
        last_eval=attempt_eval,
        llm_provider="claude-agent",
        model_name="claude-sonnet-4-6",
        language="ascendc",
        target_gpu="ascend_910b",
        max_opt_rounds=20,
        wm_stagnation_window=5,
        wm_max_difficulty=4,
    )

    checkpoint_dir = manifest_path.parent
    assert checkpoint_dir.name == "ckpt_000001_attempt_r0007"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    runtime = json.loads((checkpoint_dir / "runtime_state.json").read_text(encoding="utf-8"))
    assert manifest["checkpoint_version"] == "v2"
    assert manifest["checkpoint_kind"] == "attempt_boundary"
    assert manifest["search"]["attempt_idx"] == 2
    assert runtime["resume_action"] == "continue_current_action"
    assert runtime["last_completed_attempt_idx"] == 2
    assert runtime["next_attempt_idx"] == 3
    assert (checkpoint_dir / "solutions" / "last_solution.json").is_file()

    restored = manager.restore_to_run(manager.resolve("latest"), target_run_id="run-2")

    assert restored.checkpoint_version == "v2"
    assert restored.checkpoint_kind == "attempt_boundary"
    assert restored.resume_in_cycle is True
    assert restored.start_round == 8
    assert restored.current_solution is not None
    assert restored.current_solution.name == "attempt"


def test_solution_db_exposes_jsonl_path(tmp_path):
    from k_search.utils.solution_db import SolutionDB

    db_path = tmp_path / "solutions.jsonl"
    db = SolutionDB(jsonl_path=db_path)

    assert db.jsonl_path == db_path


def test_checkpoint_cli_validation_accepts_only_ascendc_claude_world_model():
    import generate_kernels_and_eval as cli

    args = SimpleNamespace(
        checkpoint_enable=True,
        resume_from_checkpoint=None,
        world_model=False,
        llm_provider="claude-agent",
        language="ascendc",
        task_source="ascendc",
    )
    with pytest.raises(ValueError, match="--world-model"):
        cli._validate_checkpoint_args(args, llm_provider="claude-agent")

    args.world_model = True
    args.llm_provider = "openai"
    with pytest.raises(ValueError, match="--llm-provider claude-agent"):
        cli._validate_checkpoint_args(args, llm_provider="openai")

    args.llm_provider = "claude-agent"
    args.language = "cuda"
    with pytest.raises(ValueError, match="--language ascendc"):
        cli._validate_checkpoint_args(args, llm_provider="claude-agent")

    args.language = "ascendc"
    args.task_source = "kernelbench"
    with pytest.raises(ValueError, match="--task-source ascendc"):
        cli._validate_checkpoint_args(args, llm_provider="claude-agent")

    args.task_source = "ascendc"
    cli._validate_checkpoint_args(args, llm_provider="claude-agent")


def test_world_model_generate_uses_checkpoint_restore_as_authoritative_state(tmp_path, monkeypatch):
    from k_search.kernel_generators.checkpoint import CheckpointConfig, RestoredCheckpoint
    from k_search.kernel_generators.kernel_generator_world_model import (
        WorldModelKernelGeneratorWithBaseline,
    )
    import k_search.kernel_generators.kernel_generator_world_model as wm_module

    world_model_path = tmp_path / "restored_world_model.json"
    world_model_path.write_text(_world_model_json(), encoding="utf-8")
    restored = RestoredCheckpoint(
        checkpoint_id="ckpt_000001_cycle_r0003",
        checkpoint_version="v1",
        checkpoint_kind="cycle_boundary",
        manifest={"checkpoint_id": "ckpt_000001_cycle_r0003"},
        runtime_state={"next_round": 4},
        world_model_path=world_model_path,
        solution_db_path=tmp_path / "restored_solution_db.jsonl",
        best_solution=_solution("best", "best-code"),
        best_eval=EvalResult(status="passed", latency_ms=0.5, metrics={"score": 2.0}),
        best_score=2.0,
        current_solution=_solution("current", "current-code"),
        start_round=4,
        resume_in_cycle=False,
    )
    captured = {}

    class FakeCheckpointManager:
        def __init__(self, **kwargs):
            captured["manager_kwargs"] = kwargs

        def resolve(self, ref, *, policy="latest"):
            captured["resolve"] = (ref, policy)
            return SimpleNamespace(checkpoint_id="ckpt_000001_cycle_r0003")

        def restore_to_run(self, ref, *, target_run_id):
            captured["restore"] = (ref.checkpoint_id, target_run_id)
            return restored

    class FakeWorldModel:
        def __init__(self):
            self.set_calls = []

        def set(self, definition_name, world_model_json):
            self.set_calls.append((definition_name, world_model_json))

        def get(self, definition_name):
            return _world_model_json()

        def ensure_initialized(self, **kwargs):
            raise AssertionError("checkpoint restore should skip fresh WM init")

    class FakeTask:
        name = "vec_add"
        task_source = "ascendc"
        task_path = tmp_path / "op_project"

        def get_definition_text(self, language):
            return "spec"

        def code_for_world_model_from_raw(self, *, raw, language):
            return str(raw)

    monkeypatch.setattr(wm_module, "CheckpointManager", FakeCheckpointManager)

    generator = WorldModelKernelGeneratorWithBaseline(
        model_name="claude-sonnet-4-6",
        language="ascendc",
        target_gpu="ascend_910b",
        llm_provider="claude-agent",
        llm_client=SimpleNamespace(generate=lambda prompt: "{}"),
        artifacts_dir=str(tmp_path / "artifacts"),
        checkpoint_config=CheckpointConfig(enabled=True, resume_from="latest"),
    )
    fake_wm = FakeWorldModel()
    generator._wm = fake_wm

    def fake_cycles(**kwargs):
        captured["cycles"] = kwargs
        return kwargs["restored_best_solution"]

    monkeypatch.setattr(generator, "_generate_world_model_cycles_v2", fake_cycles)

    solution = generator.generate(
        task=FakeTask(),
        max_opt_rounds=20,
        wm_stagnation_window=5,
        continue_from_solution="ignored-by-checkpoint",
        continue_from_world_model="also-ignored",
        run_id="run-restored",
    )

    assert solution.name == "best"
    assert captured["resolve"] == ("latest", "latest")
    assert captured["restore"] == ("ckpt_000001_cycle_r0003", "run-restored")
    assert captured["cycles"]["start_round"] == 4
    assert captured["cycles"]["restored_best_solution"].name == "best"
    assert captured["cycles"]["restored_best_eval"].score() == 2.0
    assert captured["cycles"]["restored_best_score"] == 2.0
    assert "current-code" in captured["cycles"]["initial_raw_code"]
    assert fake_wm.set_calls == [("vec_add", _world_model_json())]


def test_world_model_generate_saves_cycle_checkpoint_under_run_artifacts(tmp_path, monkeypatch):
    from k_search.kernel_generators.ascendc_agentic_codegen import AscendCAgenticCodegenResult
    from k_search.kernel_generators.checkpoint import CheckpointConfig
    from k_search.kernel_generators.kernel_generator_world_model import (
        WorldModelKernelGeneratorWithBaseline,
    )

    monkeypatch.setenv("KSEARCH_TASK_ID", "task-save")

    class FakeTask:
        name = "vec_add"
        task_source = "ascendc"
        task_path = tmp_path / "op_project"

        def get_definition_text(self, language):
            return "spec"

        def get_agentic_definition_text(self, *, language):
            return "agentic spec"

        def get_baseline_targets_text(self):
            return ""

        def get_last_round_trace_logs_for_prompt(self):
            return ""

        def code_for_world_model_from_raw(self, *, raw, language):
            return json.dumps(raw, sort_keys=True) if isinstance(raw, dict) else str(raw)

        def make_solution_from_project_dir(self, **kwargs):
            raise AssertionError("fake runner supplies solution directly")

    class FakeWorldModel:
        def ensure_initialized(self, **kwargs):
            return _world_model_json()

        def get(self, definition_name):
            return _world_model_json()

        def propose_action_nodes(self, **kwargs):
            return None

        def get_tree_path_text(self, definition_name):
            return ""

        def choose_next_action_node_id(self, definition_name):
            return "s1"

        def set_active_leaf_id(self, definition_name, node_id):
            return None

        def get_node_obj(self, definition_name, node_id):
            return {
                "node_id": "s1",
                "parent_id": "root",
                "action": {
                    "title": "make faster",
                    "description": "Change kernel.",
                    "difficulty_1_to_5": 1,
                    "expected_vs_baseline_factor": 1.1,
                },
            }

        def get_solution_ref_for_node(self, definition_name, node_id):
            return None

        def attach_solution_to_active_leaf(self, **kwargs):
            return None

        def refine(self, **kwargs):
            return None

        def note_action_too_hard(self, **kwargs):
            return None

    class FakeRunner:
        def open_cycle(self, *, task, request, base_solution):
            class FakeCycle:
                def __init__(self, request):
                    self.request = request

                def __enter__(self):
                    return self

                def __exit__(self, exc_type, exc, tb):
                    return False

                def run_initial(self):
                    sol = _solution("cycle-best", "best-code")
                    raw = {src.path: src.content for src in sol.sources}
                    return AscendCAgenticCodegenResult(
                        solution=sol,
                        eval_result=EvalResult(
                            status="passed",
                            latency_ms=0.5,
                            metrics={"score": 2.0, "score_name": "vs_baseline"},
                        ),
                        raw=json.dumps(raw, sort_keys=True),
                        cleaned=raw,
                        transcript="ok",
                        prompt="prompt",
                        prompt_chars=6,
                        changed_paths=["kernel.cpp"],
                        diff_text="diff --git a/kernel.cpp b/kernel.cpp",
                        project_path=str(tmp_path),
                        artifact_paths={},
                        session_id="session-1",
                    )

            return FakeCycle(request)

    generator = WorldModelKernelGeneratorWithBaseline(
        model_name="claude-sonnet-4-6",
        language="ascendc",
        target_gpu="ascend_910b",
        llm_provider="claude-agent",
        llm_client=SimpleNamespace(generate=lambda prompt: "{}"),
        artifacts_dir=str(tmp_path / "artifacts"),
        checkpoint_config=CheckpointConfig(enabled=True),
    )
    generator._wm = FakeWorldModel()
    generator._ascendc_agentic_runner = FakeRunner()

    solution = generator.generate(
        task=FakeTask(),
        max_opt_rounds=1,
        wm_stagnation_window=1,
        run_id="run-save",
    )

    assert solution.name == "cycle-best"
    checkpoint_root = (
        tmp_path
        / "artifacts"
        / "vec_add"
        / "task-save"
        / "runs"
        / "run-save"
        / "artifacts"
        / "checkpoints"
    )
    latest = json.loads((checkpoint_root / "latest.json").read_text(encoding="utf-8"))
    manifest_path = checkpoint_root / latest["latest_checkpoint_path"]
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    runtime = json.loads((manifest_path.parent / "runtime_state.json").read_text(encoding="utf-8"))
    assert manifest["checkpoint_kind"] == "cycle_boundary"
    assert manifest["llm"]["provider"] == "claude-agent"
    assert manifest["search"]["round_index"] == 1
    assert runtime["next_round"] == 2
    assert (manifest_path.parent / "world_model" / "solution_db.jsonl").is_file()
    assert (manifest_path.parent / "solutions" / "best_solution.json").is_file()


def test_world_model_generate_saves_attempt_checkpoint_under_run_artifacts(tmp_path, monkeypatch):
    from k_search.kernel_generators.ascendc_agentic_codegen import AscendCAgenticCodegenResult
    from k_search.kernel_generators.checkpoint import CheckpointConfig
    from k_search.kernel_generators.kernel_generator_world_model import (
        WorldModelKernelGeneratorWithBaseline,
    )

    monkeypatch.setenv("KSEARCH_TASK_ID", "task-save")

    class FakeTask:
        name = "vec_add"
        task_source = "ascendc"
        task_path = tmp_path / "op_project"

        def get_definition_text(self, language):
            return "spec"

        def get_agentic_definition_text(self, *, language):
            return "agentic spec"

        def get_baseline_targets_text(self):
            return ""

        def get_last_round_trace_logs_for_prompt(self):
            return ""

        def code_for_world_model_from_raw(self, *, raw, language):
            return json.dumps(raw, sort_keys=True) if isinstance(raw, dict) else str(raw)

        def make_solution_from_project_dir(self, **kwargs):
            raise AssertionError("fake runner supplies solution directly")

    class FakeWorldModel:
        def ensure_initialized(self, **kwargs):
            return _world_model_json()

        def get(self, definition_name):
            return _world_model_json()

        def propose_action_nodes(self, **kwargs):
            return None

        def get_tree_path_text(self, definition_name):
            return ""

        def choose_next_action_node_id(self, definition_name):
            return "s1"

        def set_active_leaf_id(self, definition_name, node_id):
            return None

        def get_node_obj(self, definition_name, node_id):
            return {
                "node_id": "s1",
                "parent_id": "root",
                "action": {
                    "title": "make faster",
                    "description": "Change kernel.",
                    "difficulty_1_to_5": 1,
                    "expected_vs_baseline_factor": 1.1,
                },
            }

        def get_solution_ref_for_node(self, definition_name, node_id):
            return None

        def attach_solution_to_active_leaf(self, **kwargs):
            return None

        def refine(self, **kwargs):
            return None

        def note_action_too_hard(self, **kwargs):
            return None

    class FakeRunner:
        def open_cycle(self, *, task, request, base_solution):
            class FakeCycle:
                def __init__(self, request):
                    self.request = request

                def __enter__(self):
                    return self

                def __exit__(self, exc_type, exc, tb):
                    return False

                def run_initial(self):
                    sol = _solution("attempt-best", "best-code")
                    raw = {src.path: src.content for src in sol.sources}
                    return AscendCAgenticCodegenResult(
                        solution=sol,
                        eval_result=EvalResult(
                            status="passed",
                            latency_ms=0.5,
                            metrics={"score": 2.0, "score_name": "vs_baseline"},
                        ),
                        raw=json.dumps(raw, sort_keys=True),
                        cleaned=raw,
                        transcript="ok",
                        prompt="prompt",
                        prompt_chars=6,
                        changed_paths=["kernel.cpp"],
                        diff_text="diff --git a/kernel.cpp b/kernel.cpp",
                        project_path=str(tmp_path),
                        artifact_paths={},
                        session_id="session-1",
                    )

            return FakeCycle(request)

    generator = WorldModelKernelGeneratorWithBaseline(
        model_name="claude-sonnet-4-6",
        language="ascendc",
        target_gpu="ascend_910b",
        llm_provider="claude-agent",
        llm_client=SimpleNamespace(generate=lambda prompt: "{}"),
        artifacts_dir=str(tmp_path / "artifacts"),
        checkpoint_config=CheckpointConfig(enabled=True, every="attempt"),
    )
    generator._wm = FakeWorldModel()
    generator._ascendc_agentic_runner = FakeRunner()

    solution = generator.generate(
        task=FakeTask(),
        max_opt_rounds=1,
        wm_stagnation_window=1,
        run_id="run-save",
    )

    assert solution.name == "attempt-best"
    checkpoint_root = (
        tmp_path
        / "artifacts"
        / "vec_add"
        / "task-save"
        / "runs"
        / "run-save"
        / "artifacts"
        / "checkpoints"
    )
    latest = json.loads((checkpoint_root / "latest.json").read_text(encoding="utf-8"))
    manifest_path = checkpoint_root / latest["latest_checkpoint_path"]
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    runtime = json.loads((manifest_path.parent / "runtime_state.json").read_text(encoding="utf-8"))
    assert manifest["checkpoint_version"] == "v2"
    assert manifest["checkpoint_kind"] == "attempt_boundary"
    assert manifest["search"]["attempt_idx"] == 1
    assert runtime["resume_action"] == "continue_current_action"
    assert runtime["next_attempt_idx"] == 2
    assert (manifest_path.parent / "solutions" / "current_solution.json").is_file()
