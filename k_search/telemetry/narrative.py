"""Run-level narrative summary log for the world-model optimization loop.

Produces a single human-readable timeline (`summary.md`) plus a machine-readable
event stream (`events.jsonl`) under the unified run logs dir
`<base>/logs/<task>/<run_id>/`. Long content (full prompts/responses, full diffs,
full eval logs) is intentionally summarized here — drill down into the detailed
`llm/` and `telemetry/` logs for the complete records.

Design rules:
- Never raise into the caller: every public method is wrapped in try/except.
- Append-on-write (no long-lived handles) so a crash still leaves a usable log.
"""

from __future__ import annotations

import json
import os
from datetime import datetime
from pathlib import Path
from typing import Any, Iterable, Mapping, Optional


def _default_excerpt_chars() -> int:
    raw = os.getenv("KSEARCH_SUMMARY_EXCERPT_CHARS", "").strip()
    try:
        n = int(raw)
        return n if n > 0 else 1000
    except (TypeError, ValueError):
        return 1000


def _excerpt(text: Any, max_chars: int) -> str:
    s = str(text if text is not None else "").strip()
    if max_chars and len(s) > int(max_chars):
        return s[: int(max_chars)] + "\n...<truncated>..."
    return s


def _fmt_num(value: Any, fmt: str = "{:.4f}") -> str:
    try:
        if isinstance(value, (int, float)):
            return fmt.format(float(value))
    except Exception:
        pass
    return "—"


class RunNarrativeLogger:
    """Append structured, summarized events to summary.md + events.jsonl."""

    def __init__(self, run_dir: str | Path, *, meta: Optional[Mapping[str, Any]] = None) -> None:
        self.run_dir = Path(run_dir)
        self.summary_path = self.run_dir / "summary.md"
        self.events_path = self.run_dir / "events.jsonl"
        self.meta_path = self.run_dir / "run_meta.json"
        self.excerpt_chars = _default_excerpt_chars()
        self.meta = dict(meta or {})
        self._ok = True
        try:
            self.run_dir.mkdir(parents=True, exist_ok=True)
        except Exception:
            self._ok = False
        if self.meta:
            self._write_meta(self.meta)

    # ------------------------------------------------------------------ internals
    @staticmethod
    def _clock() -> str:
        return datetime.now().strftime("%H:%M:%S")

    def _append_md(self, text: str) -> None:
        if not self._ok:
            return
        try:
            with self.summary_path.open("a", encoding="utf-8") as f:
                f.write(text if text.endswith("\n") else text + "\n")
        except Exception:
            pass

    def _append_event(self, etype: str, fields: Mapping[str, Any]) -> None:
        if not self._ok:
            return
        try:
            rec: dict[str, Any] = {
                "ts": datetime.now().isoformat(timespec="seconds"),
                "type": str(etype),
            }
            for k, v in fields.items():
                rec[str(k)] = v
            with self.events_path.open("a", encoding="utf-8") as f:
                f.write(json.dumps(rec, ensure_ascii=False, default=str) + "\n")
        except Exception:
            pass

    def _write_meta(self, meta: dict[str, Any]) -> None:
        if not self._ok:
            return
        try:
            self.meta_path.write_text(
                json.dumps(meta, ensure_ascii=False, indent=2, default=str), encoding="utf-8"
            )
        except Exception:
            pass

    # ------------------------------------------------------------------ events
    def run_start(self, meta: Optional[Mapping[str, Any]] = None) -> None:
        try:
            m = dict(self.meta)
            if meta:
                m.update(dict(meta))
                self.meta = dict(m)
            if m:
                self._write_meta(m)
            lines = ["# K-Search Run Summary", ""]
            for key in (
                "run_id",
                "task_name",
                "model_name",
                "llm_provider",
                "target_gpu",
                "language",
                "max_opt_rounds",
                "reference_latency_ms",
                "artifacts_dir",
            ):
                if key in m and m[key] is not None:
                    lines.append(f"- {key}: {m[key]}")
            lines.append(f"- started: {datetime.now().isoformat(timespec='seconds')}")
            lines.append("")
            self._append_md("\n".join(lines))
            self._append_event("run_start", m)
        except Exception:
            pass

    def world_model_init(self, *, node_count: Any = None, summary: str = "", actions: Optional[Iterable[Mapping[str, Any]]] = None) -> None:
        try:
            acts = list(actions or [])
            lines = [f"## [{self._clock()}] 世界模型初始化"]
            if node_count is not None:
                lines.append(f"- 节点数: {node_count}")
            for a in acts[:20]:
                nid = a.get("node_id") or a.get("id") or "?"
                title = a.get("title") or a.get("decision") or ""
                diff = a.get("difficulty_1_to_5")
                score = a.get("score_0_to_1")
                lines.append(
                    f"  - [{nid}] {str(title)[:120]} (难度{diff if diff is not None else '?'}, "
                    f"score {_fmt_num(score, '{:.2f}')})"
                )
            if summary and not acts:
                lines.append(_excerpt(summary, self.excerpt_chars))
            lines.append("")
            self._append_md("\n".join(lines))
            self._append_event(
                "world_model_init",
                {"node_count": node_count, "actions": acts[:20]},
            )
        except Exception:
            pass

    def action_selected(
        self,
        *,
        node_id: Any = None,
        title: Any = None,
        decision: Any = None,
        difficulty: Any = None,
        score: Any = None,
        round_num: Any = None,
    ) -> None:
        try:
            head = f"## [{self._clock()}] 选择 action"
            if node_id is not None:
                head += f" [{node_id}]"
            lines = [head]
            if title:
                lines.append(f"- 标题: {str(title)[:200]}")
            if decision:
                lines.append(f"- 决策点: {str(decision)[:200]}")
            meta_bits = []
            if difficulty is not None:
                meta_bits.append(f"难度 {difficulty}")
            if score is not None:
                meta_bits.append(f"score {_fmt_num(score, '{:.2f}')}")
            if meta_bits:
                lines.append("- " + ", ".join(meta_bits))
            lines.append("")
            self._append_md("\n".join(lines))
            self._append_event(
                "action_selected",
                {
                    "node_id": node_id,
                    "title": title,
                    "decision": decision,
                    "difficulty": difficulty,
                    "score": score,
                    "round": round_num,
                },
            )
        except Exception:
            pass

    def llm_codegen(
        self,
        *,
        round_num: Any,
        attempt: Any = None,
        mode: str = "",
        prompt: str = "",
        response: str = "",
        changed_paths: Optional[Iterable[str]] = None,
        diff: str = "",
        detail_paths: Optional[Iterable[str]] = None,
    ) -> None:
        try:
            changed = list(changed_paths or [])
            head = f"### [{self._clock()}] Round {round_num} — LLM codegen"
            extra = []
            if mode:
                extra.append(str(mode))
            if attempt is not None:
                extra.append(f"attempt {attempt}")
            if extra:
                head += " (" + ", ".join(extra) + ")"
            lines = [head]
            if prompt:
                lines += ["- prompt 摘要:", "", "```", _excerpt(prompt, self.excerpt_chars), "```"]
            if response:
                lines += ["- response 摘要:", "", "```", _excerpt(response, self.excerpt_chars), "```"]
            if changed:
                lines.append(f"- 改动文件: {', '.join(str(p) for p in changed)}")
            if diff:
                lines += ["- diff 摘要:", "", "```diff", _excerpt(diff, 2000), "```"]
            for dp in (detail_paths or []):
                lines.append(f"- 详细日志: {dp}")
            lines.append("")
            self._append_md("\n".join(lines))
            self._append_event(
                "llm_codegen",
                {
                    "round": round_num,
                    "attempt": attempt,
                    "mode": mode,
                    "prompt_excerpt": _excerpt(prompt, self.excerpt_chars),
                    "response_excerpt": _excerpt(response, self.excerpt_chars),
                    "changed_paths": changed,
                    "detail_paths": list(detail_paths or []),
                },
            )
        except Exception:
            pass

    def eval_result(self, *, round_num: Any, eval_result: Any) -> None:
        """Record one evaluation outcome (compile / accuracy / perf / vs_baseline)."""
        try:
            ev = eval_result
            status = str(getattr(ev, "status", "") or "")
            is_passed = bool(getattr(ev, "is_passed", lambda: status.lower() == "passed")())
            compile_ok = status.lower() not in {"compile_failed", "timeout"}
            latency = getattr(ev, "latency_ms", None)
            vs_base = getattr(ev, "mean_vs_baseline_factor", None)

            compile_mark = "✅" if compile_ok else "❌"
            acc_mark = "✅" if is_passed else "❌"
            lines = [
                f"### [{self._clock()}] Round {round_num} — 评测",
                f"- 状态: {status}",
                f"- 编译: {compile_mark}  精度: {acc_mark}",
            ]
            if isinstance(latency, (int, float)):
                lines.append(f"- 性能: {_fmt_num(latency)} ms")
            if isinstance(vs_base, (int, float)) and float(vs_base) > 0:
                lines.append(f"- 相对基线: {_fmt_num(vs_base, '{:.3f}')}x")
            try:
                for ln in ev.perf_summary_lines(prefix=""):
                    lines.append(ln if ln.startswith("-") else f"- {ln}")
            except Exception:
                pass
            if not (compile_ok and is_passed):
                excerpt = getattr(ev, "log_excerpt", "") or ""
                if excerpt:
                    label = "编译日志" if not compile_ok else "精度/运行日志"
                    lines += [f"- {label}(摘要):", "", "```", _excerpt(excerpt, 2000), "```"]
            lines.append("")
            self._append_md("\n".join(lines))
            self._append_event(
                "eval_result",
                {
                    "round": round_num,
                    "status": status,
                    "compile_ok": compile_ok,
                    "passed": is_passed,
                    "latency_ms": latency if isinstance(latency, (int, float)) else None,
                    "vs_baseline": vs_base if isinstance(vs_base, (int, float)) else None,
                },
            )
        except Exception:
            pass

    def world_model_update(self, *, kind: str = "", round_num: Any = None, detail: str = "", prediction: Any = None) -> None:
        try:
            head = f"### [{self._clock()}] 世界模型更新"
            if kind:
                head += f" ({kind})"
            lines = [head]
            if detail:
                lines.append(_excerpt(detail, self.excerpt_chars))
            if prediction is not None:
                lines.append(f"- 预测: {str(prediction)[:300]}")
            lines.append("")
            self._append_md("\n".join(lines))
            self._append_event(
                "world_model_update",
                {"kind": kind, "round": round_num, "prediction": prediction},
            )
        except Exception:
            pass

    def run_end(self, *, best_round: Any = None, latency_ms: Any = None, vs_baseline: Any = None, total_rounds: Any = None, note: str = "") -> None:
        try:
            lines = [f"## [{self._clock()}] Run 结束"]
            if best_round is not None:
                lines.append(f"- 最优轮: {best_round}")
            if isinstance(latency_ms, (int, float)):
                lines.append(f"- 最优性能: {_fmt_num(latency_ms)} ms")
            if isinstance(vs_baseline, (int, float)) and float(vs_baseline) > 0:
                lines.append(f"- 相对基线: {_fmt_num(vs_baseline, '{:.3f}')}x")
            if total_rounds is not None:
                lines.append(f"- 总轮数: {total_rounds}")
            if note:
                lines.append(f"- 备注: {note}")
            lines.append(f"- 结束: {datetime.now().isoformat(timespec='seconds')}")
            lines.append("")
            self._append_md("\n".join(lines))
            self._append_event(
                "run_end",
                {
                    "best_round": best_round,
                    "latency_ms": latency_ms if isinstance(latency_ms, (int, float)) else None,
                    "vs_baseline": vs_baseline if isinstance(vs_baseline, (int, float)) else None,
                    "total_rounds": total_rounds,
                },
            )
        except Exception:
            pass
