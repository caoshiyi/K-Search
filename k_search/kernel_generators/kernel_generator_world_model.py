"""World-model aware kernel generators.

These classes keep all world-model-related logic out of `kernel_generator.py`.
They reuse the base generator helpers (code cleaning, solution creation, trace selection)
and only override prompt construction to inject the persistent world model JSON.
"""

from __future__ import annotations

from typing import Any, Optional

from pathlib import Path

from k_search.kernel_generators.kernel_generator import KernelGenerator
from k_search.tasks.task_base import code_from_solution
from k_search.kernel_generators.kernel_generator_prompts import get_prompt_from_definition_text
from k_search.kernel_generators.world_model_prompts import (
    get_debug_and_improve_from_spec_prompt_from_text,
    get_debug_generated_code_prompt_from_text,
    get_generate_code_from_action_prompt_from_text,
    get_generate_code_from_spec_with_action_prompt_from_text,
    get_improve_from_spec_prompt_from_text,
    get_improve_generated_code_prompt_from_text,
)
from k_search.kernel_generators.world_model_manager import WorldModelConfig, WorldModelManager
from k_search.tasks.task_base import EvalResult
from k_search.kernel_generators.world_model import (
    Prediction,
    render_chosen_action_node_block,
    render_open_action_nodes_block,
    render_world_model_section,
    render_world_model_status,
)
from k_search.utils.solution_db import SolutionDB
from k_search.utils.paths import get_ksearch_artifacts_dir


class WorldModelKernelGeneratorWithBaseline(KernelGenerator):
    """Baseline-aware generator variant that maintains and injects a persistent world model."""

    def __init__(
        self,
        *args,
        enable_world_model: bool = True,
        # Default higher to allow passing full kernel.cu into WM prompts (we avoid truncating code).
        world_model_max_chars: int = 50000,
        artifacts_dir: str | None = None,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self._world_model_max_chars = int(world_model_max_chars)
        self._artifacts_dir = artifacts_dir

        def _llm_call(prompt: str) -> str:
            if self.model_name.startswith("gpt-5") or self.model_name.startswith("o3"):
                response = self.client.responses.create(
                    model=self.model_name,
                    input=prompt,
                    reasoning={"effort": self.reasoning_effort},
                )
                return (response.output_text or "").strip()
            response = self.client.chat.completions.create(
                model=self.model_name, messages=[{"role": "user", "content": prompt}]
            )
            return (response.choices[0].message.content or "").strip()

        self._wm = WorldModelManager(
            llm_call=_llm_call,
            target_gpu=self.target_gpu,
            language=self.language,
            config=WorldModelConfig(
                enabled=bool(enable_world_model),
                max_chars_per_block=self._world_model_max_chars,
            ),
        )
        # Lazy-init; we need TraceSet.root to choose a persistence location.
        self._solution_db: Optional[SolutionDB] = None

    def generate(  # type: ignore[override]
        self,
        task: Any,
        max_opt_rounds: int = 10,
        baseline_solution: Optional[str] = None,  # handled by task; kept for signature compatibility
        *,
        wm_stagnation_window: int = 5,
        num_debug_and_improve_rounds: int = 5,
        continue_from_solution: Optional[str] = None,
        # Workload selection is owned by the Task; configure it when constructing `task`.
    ) -> Any:
        """
        Baseline-aware generator with persistent world model injection/refinement.
        This is a lightly modified copy of KernelGenerator.generate().
        """
        import random
        import time
        try:
            max_dai = int(num_debug_and_improve_rounds)
        except Exception:
            max_dai = 5
        if max_dai < 1:
            max_dai = 1

        def _stage(msg: str) -> None:
            m = (msg or "").strip()
            if not m:
                return
            print(f"\n[STAGE] {m}", flush=True)

        def _emit(text: str) -> None:
            t = (text or "").strip("\n")
            if not t:
                return
            print(t, flush=True)

        def _code_for_wm_from_raw(raw: Any) -> str:
            return task.code_for_world_model_from_raw(raw=raw, language=self.language)

        def _wm_guardrail(s: Any) -> str:
            """
            Emergency-only guardrail: avoid pathological multi-MB prompts.
            This is NOT a normal cap; typical kernel.cu should pass through unchanged.
            """
            try:
                ss = s if isinstance(s, str) else str(s or "")
                if len(ss) > 200000:
                    return ss[:200000] + "\n...<truncated for safety>...\n"
                return ss
            except Exception:
                return str(s or "")

        # Init SolutionDB under k-search artifacts dir (task-agnostic; never use dataset root).
        if self._solution_db is None:
            _stage("init SolutionDB")
            try:
                db_path = (
                    get_ksearch_artifacts_dir(base_dir=self._artifacts_dir, task_name=str(getattr(task, "name", "") or ""))
                    / "world_model"
                    / "solution_db.jsonl"
                )
            except Exception:
                db_path = get_ksearch_artifacts_dir(base_dir=self._artifacts_dir, task_name=None) / "world_model" / "solution_db.jsonl"
            self._solution_db = SolutionDB(
                jsonl_path=db_path,
                max_excerpt_chars=self._world_model_max_chars,
            )

        get_def = getattr(task, "get_definition_text", None)
        if callable(get_def):
            definition_text = str(get_def(language=str(self.language)) or "").strip()
            if not definition_text:
                raise RuntimeError(
                    f"Task '{getattr(task, 'name', '')}' returned empty definition text; "
                    "cannot build world-model prompts without a definition."
                )
        else:
            raise RuntimeError(
                f"Task '{getattr(task, 'name', '')}' does not provide get_definition_text(); "
                "cannot build world-model prompts without a definition."
            )

        # Optional W&B support
        try:
            import wandb  # type: ignore
        except Exception:  # pragma: no cover
            wandb = None

        # Seed initial code
        current_code = None
        current_raw_code = None
        if continue_from_solution:
            _stage(f"resume from solution={continue_from_solution}")
            base_sol = task.get_solution(continue_from_solution)
            if base_sol is None:
                raise ValueError(f"Solution '{continue_from_solution}' not found in TraceSet")
            if base_sol.definition != task.name:
                raise ValueError(
                    f"Solution '{continue_from_solution}' does not belong to definition '{task.name}'"
                )
            current_code, current_raw_code = code_from_solution(self.language, base_sol)

            # Evaluate the continued-from solution first (best-effort), then initialize the WM with
            # root attached to this solution. This makes base-vs-cycle comparisons meaningful immediately
            # and lets the LLM "see" the base code+perf at init time.
            _stage(f"seed eval for continue-from solution: {base_sol.name}")
            seed_eval = task.seed_eval_for_base_solution(
                base_solution=base_sol,
            )

            try:
                if self._solution_db is not None:
                    rec_seed = self._solution_db.add(
                        solution=base_sol,
                        eval_result=seed_eval,
                        code_text=str(current_raw_code or ""),
                        parent_solution_id=None,
                    )
                    # Initialize WM (or seed existing WM) with root attached to the continued-from solution.
                    try:
                        wm_code = _wm_guardrail(_code_for_wm_from_raw(current_raw_code))
                    except Exception:
                        wm_code = None
                    self._wm.ensure_initialized(
                        definition_name=task.name,
                        definition_text=definition_text,
                        current_code_excerpt=(str(wm_code) if isinstance(wm_code, str) and wm_code.strip() else None),
                        eval_result=seed_eval,
                        seed_root_solution_id=str(rec_seed.solution_id),
                        seed_root_solution_name=str(rec_seed.solution_name),
                        seed_root_round_index=0,
                    )
                    _emit("[WM] Initialized+seeded root from continue_from_solution (code+eval).")
                    _emit(render_world_model_status(self._wm.get(task.name)))
            except Exception:
                pass
        else:
            _stage("initialize world model")
            t0 = time.perf_counter()
            wm = self._wm.ensure_initialized(definition_name=task.name, definition_text=definition_text)
            dt = time.perf_counter() - t0
            _emit(render_world_model_status(wm))
            _emit(f"[STAGE] world model init latency: {dt:.2f}s")

            # NOTE: We intentionally do NOT attach `baseline_solution` code to the WM tree.
            # Baseline is used for targets/vs_base evaluation, but should be hidden from the model.
            # The root node's "reference implementation" is embedded as an excerpt in root.notes (spec anchor),
            # and the first PASSED generated kernel becomes the practical base for future actions.
            # Do NOT generate code here; the main loop runs explicit action cycles (N attempts each).
            current_code = None
            current_raw_code = None

        # Use the simpler explicit action-cycle loop (v2). This is much easier to reason about:
        # choose action -> attempt 1 (spec/base + action) -> attempts 2..N (debug_and_improve) -> attach+refine/too-hard.
        return self._generate_world_model_cycles_v2(
            task=task,
            max_opt_rounds=max_opt_rounds,
            wm_stagnation_window=wm_stagnation_window,
            max_dai=max_dai,
            initial_raw_code=(current_raw_code if isinstance(current_raw_code, str) else None),
        )
        # (legacy loop removed; v2 runs all optimization rounds)

    def _generate_world_model_cycles_v2(
        self,
        *,
        task: Any,
        max_opt_rounds: int,
        wm_stagnation_window: int = 5,
        max_dai: int,
        initial_raw_code: Optional[str] = None,
    ) -> Any:
        """
        Simpler state machine:
        - cycle start: pick an open action node
        - attempt 1: generate from (spec+action) if parent=root else (base+action) if parent has solution
        - attempts 2..N: debug_and_improve using logs from the previous attempt
        - cycle end: attach+refine best PASSED in this cycle; else mark action too hard
        """
        get_def = getattr(task, "get_definition_text", None)
        if callable(get_def):
            definition_text = str(get_def(language=str(self.language)) or "").strip()
            if not definition_text:
                raise RuntimeError(
                    f"Task '{getattr(task, 'name', '')}' returned empty definition text; "
                    "cannot build world-model prompts without a definition."
                )
        else:
            raise RuntimeError(
                f"Task '{getattr(task, 'name', '')}' does not provide get_definition_text(); "
                "cannot build world-model prompts without a definition."
            )
        baseline_targets_text = str(getattr(task, "get_baseline_targets_text", lambda: "")() or "").strip()

        try:
            import wandb  # type: ignore
        except Exception:  # pragma: no cover
            wandb = None

        def _stage(msg: str) -> None:
            m = (msg or "").strip()
            if not m:
                return
            print(f"\n[STAGE] {m}", flush=True)

        def _emit(text: str) -> None:
            t = (text or "").strip("\n")
            if not t:
                return
            print(t, flush=True)

        def _append_baseline_hint(p: str) -> str:
            if not baseline_targets_text:
                return p
            return (
                p
                + "\n\nPerformance targets (lower is better):\n"
                + baseline_targets_text
                + "\n- Optimize for overall mean latency across the listed workloads while maintaining correctness."
            )

        def _code_format_text() -> str:
            hook = getattr(task, "get_code_format_text", None)
            if not callable(hook):
                return ""
            try:
                return str(
                    hook(language=str(self.language), target_gpu=str(self.target_gpu)) or ""
                ).strip()
            except Exception:
                return ""

        def _emit_kernel_cu(cleaned: object) -> None:
            """Print the generated CUDA kernel.cu to stdout (bounded for readability)."""
            if (self.language or "").lower() != "cuda":
                return
            if not isinstance(cleaned, dict):
                return
            cu = cleaned.get("kernel.cu")
            if not isinstance(cu, str) or not cu.strip():
                return
            s = cu.strip()
            _emit("\n[GENERATED kernel.cu]\n" + s + "\n[/GENERATED kernel.cu]\n")

        def _code_for_wm_from_raw(raw: Any) -> str:
            return task.code_for_world_model_from_raw(raw=raw, language=self.language)

        def _wm_guardrail(s: Any) -> str:
            """
            Emergency-only guardrail: avoid pathological multi-MB prompts.
            This is NOT a normal cap; typical kernel.cu should pass through unchanged.
            """
            try:
                ss = s if isinstance(s, str) else str(s or "")
                if len(ss) > 200000:
                    return ss[:200000] + "\n...<truncated for safety>...\n"
                return ss
            except Exception:
                return str(s or "")

        best_solution: Optional[Any] = None
        best_eval: Optional[EvalResult] = None
        best_score: float = -1.0

        current_raw_code: Any = str(initial_raw_code or "")
        last_solution: Optional[Any] = None

        # Walk action cycles. Each cycle keeps trying the SAME chosen action node until:
        # - we see no improvements for `stagnation_window` consecutive rounds, OR
        # - we hit max_opt_rounds.
        cycle_start_round = 1
        while cycle_start_round <= max_opt_rounds:
            try:
                stagnation_window = int(wm_stagnation_window)
            except Exception:
                stagnation_window = 5
            if stagnation_window < 1:
                stagnation_window = 1

            _stage(f"world model: select next action (cycle start @ round {cycle_start_round})")
            try:
                wm_code = _wm_guardrail(_code_for_wm_from_raw(current_raw_code))
                self._wm.propose_action_nodes(
                    definition_name=task.name,
                    definition_text=definition_text,
                    current_code_excerpt=(str(wm_code) if str(wm_code).strip() else None),
                    current_tree_path=self._wm.get_tree_path_text(definition_name=task.name),
                    baseline_targets_text=baseline_targets_text,
                    round_index=cycle_start_round,
                )
            except Exception:
                pass
            wm_json = self._wm.get(task.name)
            _emit(render_world_model_status(wm_json))
            _emit(render_open_action_nodes_block(wm_json, max_items=8))

            try:
                chosen_leaf = self._wm.choose_next_action_node_id(definition_name=task.name)
            except Exception:
                chosen_leaf = None
            if not chosen_leaf:
                _emit("[WARN] No executable open action nodes found; stopping.")
                break

            self._wm.set_active_leaf_id(definition_name=task.name, node_id=chosen_leaf)
            node_obj = self._wm.get_node_obj(definition_name=task.name, node_id=chosen_leaf)
            chosen_action_text = None
            blk = render_chosen_action_node_block(node_obj or {})
            if blk.strip():
                chosen_action_text = blk.strip()
                _emit(chosen_action_text)

            parent_id = str((node_obj or {}).get("parent_id") or "root")
            parent_is_root = parent_id == "root"
            base_raw_code = ""
            base_score: float = -1.0  # comparable to cycle_best_score (task-defined score)
            base_eval: Optional[EvalResult] = None
            if self._solution_db is not None:
                sr = self._wm.get_solution_ref_for_node(definition_name=task.name, node_id=parent_id)
                sid = sr.get("solution_id") if isinstance(sr, dict) else None
                # base_score from stored WM eval (no extra benchmarking)
                try:
                    ev = sr.get("eval") if isinstance(sr, dict) else None
                    if isinstance(ev, dict):
                        # Best-effort reconstruct EvalResult so we can surface base perf in prompts.
                        try:
                            base_eval = EvalResult(
                                status=str(ev.get("status", "") or ""),
                                latency_ms=(float(ev["latency_ms"]) if isinstance(ev.get("latency_ms"), (int, float)) else None),
                                reference_latency_ms=(
                                    float(ev["reference_latency_ms"])
                                    if isinstance(ev.get("reference_latency_ms"), (int, float))
                                    else None
                                ),
                                mean_vs_baseline_factor=(
                                    float(ev["mean_vs_baseline_factor"])
                                    if isinstance(ev.get("mean_vs_baseline_factor"), (int, float))
                                    else None
                                ),
                                speedup_factor=(
                                    float(ev["speedup_factor"]) if isinstance(ev.get("speedup_factor"), (int, float)) else None
                                ),
                                log_excerpt=str(ev.get("log_excerpt", "") or ""),
                                metrics=(ev.get("metrics") if isinstance(ev.get("metrics"), dict) else {}),
                            )
                        except Exception:
                            base_eval = None
                        m = ev.get("metrics") if isinstance(ev.get("metrics"), dict) else None
                        sc = m.get("score") if isinstance(m, dict) else None
                        base_score = float(sc) if isinstance(sc, (int, float)) else -1.0
                except Exception:
                    base_score = -1.0
                if isinstance(sid, str) and sid.strip():
                    recb = self._solution_db.get(sid)
                    if recb is not None and recb.code:
                        base_raw_code = recb.code

            prediction = None
            try:
                act = (node_obj or {}).get("action") if isinstance((node_obj or {}).get("action"), dict) else {}
                evb = act.get("expected_vs_baseline_factor", None)
                prediction = (
                    Prediction(
                        expected_vs_baseline_factor=float(evb),
                        confidence=0.5,
                        rationale=str(act.get("rationale") or "").strip(),
                    )
                    if evb is not None
                    else None
                )
            except Exception:
                prediction = None

            cycle_best_solution: Optional[Any] = None
            cycle_best_eval: Optional[EvalResult] = None
            cycle_best_raw: str = ""
            cycle_best_wm_code: str = ""
            cycle_best_score: float = -1.0
            cycle_best_round: int = 0
            # End the cycle only after this many consecutive non-improving rounds.
            no_improve_streak: int = 0
            # End the cycle if we keep failing to beat the parent/base score for too long (once we have any PASSED solution).
            no_improve_over_base_streak: int = 0
            rounds_consumed: int = 0

            # Last attempt summary (for next debug prompt)
            last_eval: Optional[EvalResult] = None

            while True:
                if cycle_start_round + rounds_consumed > max_opt_rounds:
                    break
                attempt_idx = rounds_consumed + 1
                round_num = cycle_start_round + rounds_consumed
                print(f"\n=== Optimization Round {round_num}/{max_opt_rounds} ===")
                _emit(
                    f"[CYCLE] action_node_id={chosen_leaf} attempt={attempt_idx} "
                    f"parent_is_root={'yes' if parent_is_root else 'no'} "
                    f"base_code={'yes' if bool(str(base_raw_code or '').strip()) else 'no'}"
                )

                _stage(
                    f"codegen: attempt {attempt_idx} (round {round_num}) "
                    f"no_improve_streak={no_improve_streak}/{stagnation_window} "
                    f"no_improve_over_base={no_improve_over_base_streak}/{stagnation_window}"
                )
                if not chosen_action_text:
                    raise RuntimeError(
                        "World-model generator expected a chosen action, but chosen_action_text is empty. "
                        f"(round={round_num} attempt={attempt_idx})"
                    )
                elif attempt_idx == 1:
                    # If the action's parent has an attached solution (including root when continuing),
                    # start from that base_code; otherwise fall back to spec+action.
                    if isinstance(base_raw_code, str) and base_raw_code.strip():
                        prompt = get_generate_code_from_action_prompt_from_text(
                            self.language,
                            definition_text=definition_text,
                            base_code=base_raw_code,
                            action_text=chosen_action_text,
                            code_format=_code_format_text(),
                            target_gpu=self.target_gpu,
                        )
                    else:
                        prompt = get_generate_code_from_spec_with_action_prompt_from_text(
                            self.language,
                            definition_text=definition_text,
                            action_text=chosen_action_text,
                            code_format=_code_format_text(),
                            target_gpu=self.target_gpu,
                        )
                else:
                    if parent_is_root or not base_raw_code:
                        has_passed_in_cycle = cycle_best_solution is not None
                        # Reference base shown in prompts: prefer whichever is better by score (base_score vs cycle_best_score).
                        # If parent is root (no base score), fall back to cycle_best if present.
                        base_for_debug = "(no base code; start from spec)"
                        if isinstance(base_raw_code, str) and base_raw_code.strip() and base_score > 0:
                            base_for_debug = base_raw_code
                        if (
                            isinstance(cycle_best_raw, str)
                            and cycle_best_raw.strip()
                            and (base_score <= 0 or cycle_best_score > base_score)
                        ):
                            base_for_debug = cycle_best_raw

                        # Perf summary should match the code we include as `base_code` in the prompt.
                        base_perf_eval: Optional[EvalResult] = None
                        if isinstance(base_for_debug, str) and base_for_debug.strip():
                            if base_for_debug == cycle_best_raw and cycle_best_eval is not None:
                                base_perf_eval = cycle_best_eval
                            elif base_for_debug == base_raw_code and base_eval is not None:
                                base_perf_eval = base_eval

                        perf_summary_lines: list[str] = []
                        if last_eval is not None:
                            perf_summary_lines.extend(last_eval.perf_summary_lines(prefix="last_attempt"))
                        if base_perf_eval is not None:
                            perf_summary_lines.extend(base_perf_eval.perf_summary_lines(prefix="base"))
                        perf_summary = "\n".join(perf_summary_lines).strip()
                        if not has_passed_in_cycle:
                            prompt = get_debug_and_improve_from_spec_prompt_from_text(
                                self.language,
                                definition_text=definition_text,
                                trace_logs=str(getattr(task, "get_last_round_trace_logs_for_prompt", lambda: "")() or ""),
                                current_code=str(current_raw_code or ""),
                                action_text=str(chosen_action_text or ""),
                                code_format=_code_format_text(),
                                debug_round=min(attempt_idx, max_dai),
                                max_rounds=max_dai,
                                target_gpu=self.target_gpu,
                                perf_summary=perf_summary,
                                base_code=base_for_debug,
                            )
                        else:
                            prompt = get_improve_from_spec_prompt_from_text(
                                self.language,
                                definition_text=definition_text,
                                trace_logs=str(getattr(task, "get_last_round_trace_logs_for_prompt", lambda: "")() or ""),
                                current_code=str(current_raw_code or ""),
                                code_format=_code_format_text(),
                                debug_round=min(attempt_idx, max_dai),
                                max_rounds=max_dai,
                                target_gpu=self.target_gpu,
                                perf_summary=perf_summary,
                                base_code=base_for_debug,
                            )
                    else:
                        has_passed_in_cycle = cycle_best_solution is not None
                        # Reference base shown in prompts: prefer whichever is better by score (base_score vs cycle_best_score).
                        base_for_debug = base_raw_code
                        if (
                            isinstance(cycle_best_raw, str)
                            and cycle_best_raw.strip()
                            and (base_score <= 0 or cycle_best_score > base_score)
                        ):
                            base_for_debug = cycle_best_raw

                        base_perf_eval: Optional[EvalResult] = None
                        if isinstance(base_for_debug, str) and base_for_debug.strip():
                            if base_for_debug == cycle_best_raw and cycle_best_eval is not None:
                                base_perf_eval = cycle_best_eval
                            elif base_for_debug == base_raw_code and base_eval is not None:
                                base_perf_eval = base_eval

                        perf_summary_lines: list[str] = []
                        if last_eval is not None:
                            perf_summary_lines.extend(last_eval.perf_summary_lines(prefix="last_attempt"))
                        if base_perf_eval is not None:
                            perf_summary_lines.extend(base_perf_eval.perf_summary_lines(prefix="base"))
                        perf_summary = "\n".join(perf_summary_lines).strip()
                        if not has_passed_in_cycle:
                            prompt = get_debug_generated_code_prompt_from_text(
                                self.language,
                                definition_text=definition_text,
                                trace_logs=str(getattr(task, "get_last_round_trace_logs_for_prompt", lambda: "")() or ""),
                                base_code=base_for_debug,
                                buggy_code=str(current_raw_code or ""),
                                action_text=str(chosen_action_text or ""),
                                code_format=_code_format_text(),
                                debug_round=min(attempt_idx, max_dai),
                                max_rounds=max_dai,
                                target_gpu=self.target_gpu,
                                perf_summary=perf_summary,
                            )
                        else:
                            prompt = get_improve_generated_code_prompt_from_text(
                                self.language,
                                definition_text=definition_text,
                                trace_logs=str(getattr(task, "get_last_round_trace_logs_for_prompt", lambda: "")() or ""),
                                base_code=base_for_debug,
                                current_code=str(current_raw_code or ""),
                                code_format=_code_format_text(),
                                debug_round=min(attempt_idx, max_dai),
                                max_rounds=max_dai,
                                target_gpu=self.target_gpu,
                                perf_summary=perf_summary,
                            )

                prompt = prompt + "\n\n" + render_world_model_section(self._wm.get(task.name), max_chars=self._world_model_max_chars)
                prompt = _append_baseline_hint(prompt)

                code_result = self._generate_code_from_prompt(prompt)
                current_code = code_result["cleaned"]
                current_raw_code = code_result["raw"]
                _emit_kernel_cu(current_code)
                current_wm_code = (
                    (current_code.get("kernel.cu") if isinstance(current_code, dict) else None)
                    if (self.language or "").lower() == "cuda"
                    else None
                )
                if not isinstance(current_wm_code, str) or not current_wm_code.strip():
                    current_wm_code = _code_for_wm_from_raw(current_raw_code)
                current_wm_code = _wm_guardrail(str(current_wm_code or ""))

                _stage(f"create Solution object from current code (round {round_num})")
                solution = self._create_solution_from_code(
                    cleaned_code=current_code,
                    raw_code=current_raw_code,
                    task=task,
                    round_num=int(round_num),
                )
                last_solution = solution
                _stage(f"evaluate solution (round {round_num})")
                round_eval = task.run_benchmark(
                    solution=solution,
                    dump_traces=False,
                    round_num=int(round_num),
                )

                all_passed = bool(getattr(round_eval, "is_passed", lambda: False)())
                round_score = float(getattr(round_eval, "score", lambda: -1.0)())

                # Save as "last attempt" for the next debug prompt
                last_eval = round_eval

                # If all workloads passed in this round, log a W&B artifact containing the generated code
                # (and a WM snapshot) for traceability.
                if all_passed and wandb is not None and getattr(wandb, "run", None) is not None:
                    try:
                        import tempfile

                        art_name = f"r{round_num}_code"
                        artifact = wandb.Artifact(
                            name=art_name,
                            type="generated-code",
                            metadata={
                                "definition": task.name,
                                "round": int(round_num),
                                "solution": solution.name,
                                "language": self.language,
                                "target_gpu": self.target_gpu,
                                "wm_enabled": True,
                            },
                        )
                        with tempfile.TemporaryDirectory() as tmpdir:
                            tmpdir_p = Path(tmpdir)
                            # Cleaned code files
                            if isinstance(current_code, dict):
                                for filename, content in current_code.items():
                                    p = tmpdir_p / filename
                                    p.parent.mkdir(parents=True, exist_ok=True)
                                    p.write_text(str(content or ""))
                                    artifact.add_file(str(p), name=f"clean/{filename}")
                            else:
                                p = tmpdir_p / "main.py"
                                p.parent.mkdir(parents=True, exist_ok=True)
                                p.write_text(str(current_code or ""))
                                artifact.add_file(str(p), name="clean/main.py")

                            # Raw code (as generated from the LLM before cleaning)
                            raw_path = tmpdir_p / "raw_code.txt"
                            raw_path.write_text(str(current_raw_code) if current_raw_code is not None else "")
                            artifact.add_file(str(raw_path), name="raw/raw_code.txt")

                            # World model snapshot (best-effort)
                            wm_path = tmpdir_p / "world_model.json"
                            wm_path.write_text(str(self._wm.get(task.name) or ""))
                            artifact.add_file(str(wm_path), name="wm/world_model.json")

                            # Round-level eval summary (best-effort)
                            summary_path = tmpdir_p / "round_summary.txt"
                            summary_path.write_text(
                                (
                                    f"status={getattr(round_eval, 'status', None)}\n"
                                    f"score_name={round_eval.metrics.get('score_name')}\n"
                                    f"score_value={round_eval.metrics.get('score')}\n"
                                    f"mean_vs_baseline_factor={getattr(round_eval, 'mean_vs_baseline_factor', None)}\n"
                                    f"speedup_factor={getattr(round_eval, 'speedup_factor', None)}\n"
                                    f"latency_ms={getattr(round_eval, 'latency_ms', None)}\n"
                                )
                            )
                            artifact.add_file(str(summary_path), name="eval/round_summary.txt")

                        wandb.log_artifact(artifact)
                    except Exception:
                        pass

                if all_passed and round_score > best_score:
                    best_score = float(round_score)
                    best_eval = round_eval
                    best_solution = solution

                if all_passed:
                    er = round_eval
                    score = float(getattr(er, "score", lambda: -1.0)())
                    if score > cycle_best_score:
                        cycle_best_score = float(score)
                        cycle_best_eval = er
                        cycle_best_solution = solution
                        cycle_best_raw = str(current_raw_code or "")
                        cycle_best_wm_code = str(current_wm_code or "")
                        cycle_best_round = int(round_num)
                        no_improve_streak = 0
                    else:
                        no_improve_streak += 1
                else:
                    # Failed round (or missing perf): count as no improvement for stagnation purposes.
                    no_improve_streak += 1

                # "Can't beat base" streak: only meaningful after we have at least one PASSED solution in this cycle,
                # and only when the parent/base has a meaningful score.
                if cycle_best_solution is not None and base_score > 0:
                    if cycle_best_score > base_score:
                        no_improve_over_base_streak = 0
                    else:
                        no_improve_over_base_streak += 1

                if wandb is not None and getattr(wandb, "run", None) is not None:
                    try:
                        # Log current round score as well (helps debug regressions / oscillations).
                        round_sn = None
                        try:
                            round_sn = (
                                round_eval.metrics.get("score_name")
                                if isinstance(getattr(round_eval, "metrics", None), dict)
                                else None
                            )
                        except Exception:
                            round_sn = None
                        round_key = (
                            f"{task.name}/generate/{round_sn}"
                            if isinstance(round_sn, str) and round_sn
                            else f"{task.name}/generate/round_score"
                        )
                        wandb.log(
                            {round_key: (float(round_score) if (all_passed and round_score > 0) else None)},
                            step=round_num,
                        )

                        # Log best-so-far score (single scalar). Before we have any PASSED solution,
                        # `best_eval` is None, so we log None under a stable key.
                        key = (
                            f"{task.name}/generate/best_{best_eval.metrics['score_name']}"
                            if best_eval is not None
                            else f"{task.name}/generate/best_score"
                        )
                        wandb.log({key: (float(best_score) if best_eval is not None else None)}, step=round_num)
                    except Exception:
                        pass

                rounds_consumed += 1
                if no_improve_streak >= stagnation_window or no_improve_over_base_streak >= stagnation_window:
                    break

            if cycle_best_solution is not None and cycle_best_eval is not None:
                _stage(f"cycle end: attach+refine best PASSED (round {cycle_best_round}, score={cycle_best_score:.3f})")
                if self._solution_db is not None:
                    rec_best = self._solution_db.add(
                        solution=cycle_best_solution,
                        eval_result=cycle_best_eval,
                        code_text=str(cycle_best_raw or ""),
                        parent_solution_id=None,
                    )
                    self._wm.set_active_leaf_id(definition_name=task.name, node_id=chosen_leaf)
                    self._wm.attach_solution_to_active_leaf(
                        definition_name=task.name,
                        solution_id=rec_best.solution_id,
                        solution_name=rec_best.solution_name,
                        eval_result=cycle_best_eval,
                        round_index=cycle_best_round,
                    )
                    _emit(render_world_model_status(self._wm.get(task.name)))

                self._wm.refine(
                    definition_name=task.name,
                    definition_text=definition_text,
                    chosen_action_text=chosen_action_text,
                    current_code_excerpt=_wm_guardrail(str(cycle_best_wm_code or "")),
                    current_tree_path=self._wm.get_tree_path_text(definition_name=task.name),
                    eval_result=cycle_best_eval,
                    prediction=prediction,
                    round_index=cycle_best_round,
                )
                _emit(render_world_model_status(self._wm.get(task.name)))
            else:
                _stage("cycle end: no PASSED solution; mark action too hard")
                try:
                    er_fail = None
                    if round_eval is not None:
                        # Task guarantees failed evals contain only failure status + logs (no partial perf fields).
                        er_fail = round_eval
                    self._wm.note_action_too_hard(
                        definition_name=task.name,
                        definition_text=definition_text,
                        chosen_action_text=chosen_action_text,
                        current_code_excerpt=_wm_guardrail(str(_code_for_wm_from_raw(current_raw_code) or "")),
                        current_tree_path=self._wm.get_tree_path_text(definition_name=task.name),
                        eval_result=er_fail,
                        debug_and_improve_round=min(rounds_consumed, max_dai),
                        debug_and_improve_max_rounds=max_dai,
                        baseline_targets_text=baseline_targets_text,
                        round_index=cycle_start_round + max(0, rounds_consumed - 1),
                    )
                    _emit(render_world_model_status(self._wm.get(task.name)))
                except Exception:
                    pass

            cycle_start_round += max(1, rounds_consumed)

        # Fall back to the best observed solution, else the last attempted.
        if best_solution is not None:
            return best_solution
        if last_solution is not None:
            return last_solution
        raise ValueError(f"[{task.name}] No solution was generated (best_solution and last_solution are None).")


