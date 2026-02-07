"""World model maintenance for iterative kernel generation.

This module maintains a structured "world model" (JSON string) across rounds.
It is intended to be injected into prompts so the LLM can refine its mental
model of the kernel constraints, bottlenecks, and design plan over iterations.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any, Optional

from k_search.tasks.task_base import EvalResult

BASE_DIMENSIONS: tuple[str, ...] = (
    "tiling_policy",
    "warp_scheduling",
    "cuda_core_usage",
    "tensor_core_usage",
    "memory_bandwidth",
    "register_pressure",
    "communication",
)

DIMENSION_ENTRY_DEFAULT: dict[str, Any] = {
    "hypothesis": "",
    "confidence": 0.0,
    "notes": "",
    "next_actions": [],
}


def _eval_status_score_for_prompt(ev: Any) -> dict[str, Any]:
    """
    Minimal eval payload for prompts: include only status and score (plus score_name if available).
    """
    if not isinstance(ev, dict):
        return {}
    out: dict[str, Any] = {}
    try:
        st = ev.get("status")
        out["status"] = (str(st).strip() if st is not None else "")
    except Exception:
        out["status"] = ""
    score = None
    score_name = None
    try:
        metrics = ev.get("metrics") if isinstance(ev.get("metrics"), dict) else None
        if isinstance(metrics, dict):
            score = metrics.get("score")
            score_name = metrics.get("score_name")
    except Exception:
        score = None
        score_name = None
    # Also tolerate flattened fields (best-effort).
    if score is None:
        try:
            score = ev.get("score")
        except Exception:
            score = None
    if score_name is None:
        try:
            score_name = ev.get("score_name")
        except Exception:
            score_name = None
    if score is None or isinstance(score, (int, float, bool, str)):
        out["score"] = score
    if score_name is None or isinstance(score_name, (str, int, float, bool)):
        out["score_name"] = score_name
    return out

WORLD_MODEL_JSON_SCHEMA_GUIDE = {
    # Keep this intentionally small and stable.
    "kernel_summary": "One paragraph summary of the kernel + key constraints.",
    # Candidate plans organized as a decision/prefix tree.
    # A plan is the path from root -> a leaf (sequence of decisions + choices).
    # Branches represent alternatives at some decision point.
    "decision_tree": {
        "root_id": "root",
        "active_leaf_id": "root",
        "nodes": [
            {
                "node_id": "root",
                "parent_id": None,
                "node_type": "root",
                "decision": None,
                "choice": None,
                "solution_ref": {
                    "solution_id": None,
                    "parent_solution_id": None,
                    "eval": {"status": "", "score": None, "score_name": None},
                },
                "impacts": {
                    "memory_bandwidth": {"rating_0_to_10": 0, "risk": "", "notes": ""},
                    "register_pressure": {"rating_0_to_10": 0, "risk": "", "notes": ""},
                    "compute_intensity_and_hw_fit": {
                        "rating_0_to_10": 0,
                        "risk": "",
                        "notes": "",
                        "hw_notes": "Hardware shape/layout constraints (if relevant).",
                    },
                },
                "overall_rating_0_to_10": 0,
                "confidence_0_to_1": 0.0,
                "last_updated_round": 0,
                "notes": "",
                # Optional: if this node represents a concrete "next action" candidate,
                # store structured info here (used to drive codegen directly without a separate run()/ranking stage).
                "action": {
                    "title": "",
                    "description": "",
                    "difficulty_1_to_5": 3,
                    "score_0_to_1": 0.0,
                    "expected_vs_baseline_factor": None,
                    "rationale": "",
                },
            }
        ],
    },
    "open_questions": ["Unknowns that materially affect correctness/perf (3-8 items)."],
    # Machine-filled (deterministic) signals; safe for LLM to read but not required to edit.
    "computed_signals": {
        "round_index": 0,
        "trace": {
            "status": "",
            "latency_ms": None,
            "reference_latency_ms": None,
            "mean_vs_baseline_factor": None,
            "speedup_factor": None,
        },
    },
}


def _truncate(s: str, max_chars: int) -> str:
    if max_chars <= 0:
        return ""
    s = s or ""
    if len(s) <= max_chars:
        return s
    head = max_chars - 40
    if head < 0:
        head = 0
    return s[:head] + "\n...<truncated>...\n" + s[-20:]


def compact_definition_for_wm_prompt(definition_text: str, *, max_ref_lines: int = 40) -> str:
    """
    Produce a compact, structured kernel spec for WM prompts.
    This avoids relying on char truncation by selecting bounded, high-signal sections.
    """
    s = (definition_text or "").strip()
    if not s:
        return ""
    lines = s.splitlines()
    out: list[str] = []

    def _take_section(header: str, *, max_lines: int) -> None:
        nonlocal lines, out
        try:
            idx = lines.index(header)
        except ValueError:
            return
        # include header
        out.append(header)
        i = idx + 1
        taken = 0
        while i < len(lines):
            ln = lines[i]
            # stop at next top-level section header
            if ln in ("Axes:", "Inputs:", "Outputs:", "Constraints:", "Reference Implementation:") and ln != header:
                break
            if taken < max_lines:
                out.append(ln)
            taken += 1
            i += 1
        if taken > max_lines:
            out.append(f"... ({taken - max_lines} lines omitted) ...")

    # Always include top 2 lines (Name/Type) if present.
    for ln in lines[:2]:
        if ln.strip():
            out.append(ln)

    _take_section("Axes:", max_lines=18)
    _take_section("Inputs:", max_lines=18)
    _take_section("Outputs:", max_lines=18)
    _take_section("Constraints:", max_lines=12)

    # Reference implementation is often huge; include only a fixed number of lines.
    try:
        ridx = lines.index("Reference Implementation:")
        out.append("Reference Implementation: (excerpt)")
        ref_lines = lines[ridx + 1 : ridx + 1 + max(0, int(max_ref_lines))]
        out.extend(ref_lines)
        remaining = max(0, len(lines) - (ridx + 1 + len(ref_lines)))
        if remaining:
            out.append(f"... ({remaining} lines omitted) ...")
    except ValueError:
        pass

    # Final cleanup: strip trailing empties.
    while out and not out[-1].strip():
        out.pop()
    return "\n".join(out).strip()


def render_world_model_status(
    world_model_json: Optional[str], *, max_path_nodes: int = 6, max_node_preview: int = 8
) -> str:
    """
    Small, human-readable status for logging (stdout).
    Intentionally bounded; does not dump the full WM.
    """
    s = (world_model_json or "").strip()
    if not s:
        return "WM status: (empty)"
    obj = load_world_model_obj(s)
    if obj is None:
        return "WM status: (unparseable)"
    dt = obj.get("decision_tree")
    if not isinstance(dt, dict):
        return "WM status: (no decision_tree)"
    nodes = dt.get("nodes")
    nodes = nodes if isinstance(nodes, list) else []
    by_id: dict[str, dict[str, Any]] = {}
    for n in nodes:
        if isinstance(n, dict) and n.get("node_id"):
            by_id[str(n["node_id"])] = n
    root_id = str(dt.get("root_id", "") or "root")
    active_id = str(dt.get("active_leaf_id", "") or root_id)
    if active_id not in by_id:
        active_id = root_id

    # Count attached solutions
    attached = 0
    for n in nodes:
        if not isinstance(n, dict):
            continue
        sr = n.get("solution_ref")
        if isinstance(sr, dict) and isinstance(sr.get("solution_id"), str) and str(sr.get("solution_id") or "").strip():
            attached += 1

    # Active path summary (bounded)
    path: list[str] = []
    cur = by_id.get(active_id)
    guard = 0
    while isinstance(cur, dict) and guard < 64:
        nid = str(cur.get("node_id") or "")
        dec = cur.get("decision")
        ch = cur.get("choice")
        if dec is None and ch is None:
            path.append(f"{nid}:<root>")
        else:
            d = str(dec or "").strip()
            c = str(ch or "").strip()
            if d and c:
                path.append(f"{nid}:{d}->{c}")
            elif d:
                path.append(f"{nid}:{d}")
            else:
                path.append(f"{nid}:{c}")
        pid = cur.get("parent_id")
        if pid is None:
            break
        cur = by_id.get(str(pid))
        guard += 1
    path.reverse()
    depth = max(0, len(path) - 1)
    if len(path) > max(0, int(max_path_nodes)):
        keep = max(2, int(max_path_nodes))
        path = path[: max(1, keep - 1)] + ["..."] + path[-1:]

    # Computed signals (last trace)
    cs = obj.get("computed_signals")
    trace = cs.get("trace") if isinstance(cs, dict) else None
    status = ""
    sp = None
    lat = None
    if isinstance(trace, dict):
        status = str(trace.get("status", "") or "").strip().lower()
        sp = trace.get("speedup_factor", None)
        lat = trace.get("latency_ms", None)

    parts = [
        f"WM status: nodes={len(nodes)} attached_solutions={attached} active_leaf_id={active_id} depth={depth}",
    ]
    if status or sp is not None or lat is not None:
        ssp = f"{float(sp):.2f}x" if isinstance(sp, (int, float)) else "-"
        slat = f"{float(lat):.4f}ms" if isinstance(lat, (int, float)) else "-"
        parts.append(f"  last_trace: status={status or '-'} speedup={ssp} latency={slat}")
    if path:
        parts.append("  active_path: " + " | ".join(path))
    # Preview node ids (and very short decision/choice) so it's obvious what the node count refers to.
    try:
        preview: list[str] = []
        for nid, n in list(by_id.items())[: max(0, int(max_node_preview))]:
            dec = n.get("decision")
            ch = n.get("choice")
            if dec is None and ch is None:
                preview.append(f"{nid}:<root>")
            else:
                d = str(dec or "").strip()
                c = str(ch or "").strip()
                if d and c:
                    preview.append(f"{nid}:{_shorten(d, 28)}→{_shorten(c, 36)}")
                elif d:
                    preview.append(f"{nid}:{_shorten(d, 40)}")
                else:
                    preview.append(f"{nid}:{_shorten(c, 40)}")
        if len(by_id) > max(0, int(max_node_preview)):
            preview.append("...")  # indicate more nodes exist
        if preview:
            parts.append("  nodes_preview: " + " | ".join(preview))
    except Exception:
        pass
    return "\n".join(parts).strip()


def _extract_json_object(text: str) -> Optional[dict[str, Any]]:
    """Best-effort extraction of a JSON object from an arbitrary model response."""
    if not text:
        return None
    s = text.strip()
    # Fast path.
    if s.startswith("{") and s.endswith("}"):
        try:
            obj = json.loads(s)
            return obj if isinstance(obj, dict) else None
        except Exception:
            pass
    # Best-effort: find outermost braces.
    # More robust scan: find the first balanced {...} that parses as JSON.
    start = s.find("{")
    if start == -1:
        return None
    depth = 0
    in_str = False
    esc = False
    for i in range(start, len(s)):
        ch = s[i]
        if in_str:
            if esc:
                esc = False
            elif ch == "\\":
                esc = True
            elif ch == '"':
                in_str = False
            continue
        else:
            if ch == '"':
                in_str = True
                continue
            if ch == "{":
                depth += 1
            elif ch == "}":
                depth -= 1
                if depth == 0:
                    candidate = s[start : i + 1]
                    try:
                        obj = json.loads(candidate)
                        return obj if isinstance(obj, dict) else None
                    except Exception:
                        # Keep searching: there may be another JSON object later.
                        nxt = s.find("{", start + 1)
                        if nxt == -1:
                            return None
                        start = nxt
                        depth = 0
                        in_str = False
                        esc = False
                        # Restart scan from the new start
                        return _extract_json_object(s[start:])
    return None


def load_world_model_obj(world_model_json: str) -> Optional[dict[str, Any]]:
    """Parse + normalize a world model JSON string into a Python dict."""
    obj = _extract_json_object(world_model_json or "")
    if obj is None:
        return None
    return _normalize_world_model_obj(obj)


def dump_world_model_obj(obj: dict[str, Any]) -> str:
    """Dump a normalized world model dict to a stable JSON string."""
    norm = _normalize_world_model_obj(obj)
    return json.dumps(norm, indent=2, sort_keys=True)


def render_world_model_section(world_model_json: Optional[str], *, max_chars: int = 6000) -> str:
    s = (world_model_json or "").strip()
    if not s:
        return ""
    # Render a compact view for prompts (token-efficient). The full JSON is still
    # stored in the manager; this is just a prompt-facing projection.
    compact = compact_world_model_json_for_prompt(s, max_chars=int(max_chars))
    return "World Model (persistent across rounds; use it to guide kernel design):\n" + compact


@dataclass
class WorldModelPrompts:
    init_prompt: str


def _shorten(s: str, max_len: int) -> str:
    s = (s or "").strip()
    if max_len <= 0:
        return ""
    if len(s) <= max_len:
        return s
    return s[: max(0, max_len - 20)] + " ... " + s[-12:]


def compact_world_model_json_for_prompt(world_model_json: str, *, max_chars: int = 6000) -> str:
    """
    Produce a compact projection of the world model for prompt injection.

    Strategy:
    - Keep `kernel_summary`, `open_questions`, `computed_signals`
    - Keep a compact view of `decision_tree`: active path root->active leaf + a few alternatives
    - Drop large/verbose fields from nodes
    """
    obj = _extract_json_object(world_model_json or "")
    if obj is None:
        return _truncate(world_model_json or "", max_chars)

    obj = _normalize_world_model_obj(obj)

    # Keep only bounded, high-signal fields.
    oq = obj.get("open_questions", [])
    if not isinstance(oq, list):
        oq = []
    oq_clean = [str(x).strip() for x in oq if str(x).strip()]
    out: dict[str, Any] = {
        "kernel_summary": str(obj.get("kernel_summary", "") or "").strip(),
        "open_questions": oq_clean[:6],
        "computed_signals": obj.get("computed_signals", {}),
    }

    dtree = obj.get("decision_tree")
    if isinstance(dtree, dict):
        root_id = str(dtree.get("root_id", "") or "root")
        active_id = str(dtree.get("active_leaf_id", "") or root_id)
        nodes = dtree.get("nodes")
        nodes = nodes if isinstance(nodes, list) else []
        by_id: dict[str, dict[str, Any]] = {}
        for n in nodes:
            if isinstance(n, dict) and n.get("node_id"):
                by_id[str(n["node_id"])] = n

        # Active path root->leaf
        path_ids: list[str] = []
        cur = active_id
        seen: set[str] = set()
        while cur and cur not in seen and cur in by_id:
            seen.add(cur)
            path_ids.append(cur)
            cur = by_id[cur].get("parent_id") or ""
            cur = str(cur) if cur is not None else ""
        path_ids.reverse()

        # Alternatives: top-rated siblings at the last decision point.
        alt_ids: list[str] = []
        active_node = by_id.get(active_id)
        parent = (
            str(active_node.get("parent_id"))
            if isinstance(active_node, dict) and active_node.get("parent_id") is not None
            else None
        )
        if parent:
            sibs: list[dict[str, Any]] = []
            for n in nodes:
                if not isinstance(n, dict):
                    continue
                if str(n.get("parent_id") or "") == parent and str(n.get("node_id") or "") != active_id:
                    sibs.append(n)
            sibs.sort(key=lambda x: float(x.get("overall_rating_0_to_10", 0.0) or 0.0), reverse=True)
            for s in sibs[:2]:
                nid = str(s.get("node_id") or "")
                if nid:
                    alt_ids.append(nid)

        keep_ids: list[str] = []
        for nid in path_ids + alt_ids:
            if nid and nid not in keep_ids:
                keep_ids.append(nid)

        def _compact_node(n: dict[str, Any]) -> dict[str, Any]:
            # Keep decision structure + numeric ratings; drop verbose notes/risk text.
            sr = n.get("solution_ref") if isinstance(n.get("solution_ref"), dict) else {}
            ev = sr.get("eval") if isinstance(sr.get("eval"), dict) else {}
            act = n.get("action") if isinstance(n.get("action"), dict) else {}
            return {
                "node_id": n.get("node_id"),
                "parent_id": n.get("parent_id"),
                "node_type": n.get("node_type"),
                "decision": (None if n.get("decision") is None else str(n.get("decision") or "").strip()),
                "choice": (str(n.get("choice") or "").strip() if n.get("choice") is not None else None),
                "overall_rating_0_to_10": n.get("overall_rating_0_to_10", 0),
                "confidence_0_to_1": n.get("confidence_0_to_1", 0.0),
                "solution_ref": {
                    "solution_id": sr.get("solution_id"),
                    "parent_solution_id": sr.get("parent_solution_id"),
                    "round_index": sr.get("round_index"),
                    "eval": _eval_status_score_for_prompt(ev),
                },
                "action": {
                    "title": str(act.get("title", "") or "").strip(),
                    "difficulty_1_to_5": act.get("difficulty_1_to_5", act.get("difficulty_0_to_3", 3)),
                    "score_0_to_1": act.get("score_0_to_1", 0.0),
                    "expected_vs_baseline_factor": act.get("expected_vs_baseline_factor", None),
                },
            }

        compact_nodes: list[dict[str, Any]] = []
        for nid in keep_ids:
            n = by_id.get(nid)
            if isinstance(n, dict):
                compact_nodes.append(_compact_node(n))

        out["decision_tree"] = {
            "root_id": root_id,
            "active_leaf_id": active_id,
            "nodes": compact_nodes,
        }

    try:
        rendered = json.dumps(out, indent=2, sort_keys=True)
    except Exception:
        rendered = world_model_json
    return _truncate(rendered, max_chars)

@dataclass
class ActionCandidate:
    """A deterministic action candidate that the LLM can rank (but not rewrite the prompt)."""
    action_id: str
    title: str
    description: str
    base_node_id: str | None = None
    # Optional: point to an EXISTING leaf node to attach the next evaluated solution.
    # This prevents the model from always creating new branches and allows it to "fill" pending nodes
    # created during refine().
    attach_to_node_id: str | None = None
    # Optional fields to help constrain action granularity and guide codegen.
    # Preferred scale: [1..5] (1 easiest, 5 hardest).
    difficulty_1_to_5: int | None = None
    # Backward-compatible (deprecated): [0..3] (0 easiest, 3 hardest).
    difficulty_0_to_3: int | None = None
    # (No requirements/steps field: keep action intents high-level in title/description.)


@dataclass
class ActionRankItem:
    action_id: str
    score: float
    reason: str
    # Routing: where to branch from, and where to attach the next evaluated solution.
    base_node_id: str | None = None
    base_solution_id: str | None = None
    attach_to_node_id: str | None = None


@dataclass
class ActionRanking:
    """Result of WorldModelManager.run(): ranked 5 actions + raw model output (for debugging)."""
    candidates: list[ActionCandidate]
    ranking: list[ActionRankItem]
    prediction: "Prediction | None" = None
    raw_model_output: str = ""


def render_chosen_action_block(r: ActionRanking, *, chosen_rank_index: int = 0) -> str:
    """Deterministic formatting for the SINGLE chosen action (for codegen prompts)."""
    if not r.ranking:
        return ""
    idx = int(chosen_rank_index)
    if idx < 0:
        idx = 0
    if idx >= len(r.ranking):
        idx = 0
    item = r.ranking[idx]
    by_id = {c.action_id: c for c in r.candidates}
    c = by_id.get(item.action_id)
    title = (c.title if c else item.action_id).strip()
    desc = (c.description if c else "").strip()
    base_node_id = (item.base_node_id or (c.base_node_id if c else None) or "").strip()
    lines: list[str] = []
    lines.append("World Model: Chosen Next Action (apply this action)")
    lines.append(f"- action_id: {item.action_id}")
    lines.append(f"- title: {title}")
    if c is not None:
        d15 = getattr(c, "difficulty_1_to_5", None)
        if isinstance(d15, int):
            lines.append(f"- difficulty_1_to_5: {int(d15)}")
        elif isinstance(getattr(c, "difficulty_0_to_3", None), int):
            # Legacy mapping 0..3 -> 1..4
            lines.append(f"- difficulty_1_to_5: {int(getattr(c, 'difficulty_0_to_3')) + 1}")
    if desc:
        lines.append(f"- description: {desc}")
    if base_node_id:
        lines.append(f"- base_node_id: {base_node_id}")
    if item.attach_to_node_id:
        lines.append(f"- attach_to_node_id: {item.attach_to_node_id}")
    if r.prediction is not None:
        p = r.prediction
        vb = "?" if getattr(p, "expected_vs_baseline_factor", None) is None else f"{p.expected_vs_baseline_factor:.2f}x"
        sp = "?" if p.expected_speedup_factor is None else f"{p.expected_speedup_factor:.2f}x"
        lt = "?" if p.expected_latency_ms is None else f"{p.expected_latency_ms:.3f} ms"
        pred_metric = (
            f"vs_baseline={vb}"
            if getattr(p, "expected_vs_baseline_factor", None) is not None
            else f"speedup_vs_ref={sp}"
        )
        lines.append(f"- prediction_if_applied: {pred_metric}, latency={lt}, confidence={p.confidence:.2f}")
        if p.rationale:
            lines.append(f"  - rationale: {p.rationale}")
    # Note: do NOT include the full ranking list; codegen should see only this action.
    if item.reason:
        lines.append(f"- why_this_action: {item.reason}")
    return "\n\n" + "\n".join(lines).strip() + "\n"


 



@dataclass
class Prediction:
    """Prediction produced by `run()` for the next iteration (before eval)."""
    expected_speedup_factor: float | None = None
    expected_latency_ms: float | None = None
    # Preferred objective when a baseline exists: vs_base (>1 is better).
    expected_vs_baseline_factor: float | None = None
    confidence: float = 0.5  # [0,1]
    rationale: str = ""


def render_chosen_action_node_block(node: dict) -> str:
    """
    Deterministic formatting for a SINGLE chosen action node from the decision tree.
    This replaces the old render_chosen_action_block(ActionRanking, ...) path.
    """
    if not isinstance(node, dict):
        return ""
    nid = str(node.get("node_id") or "").strip()
    pid = node.get("parent_id")
    pid_s = "" if pid is None else str(pid).strip()
    act = node.get("action") if isinstance(node.get("action"), dict) else {}
    title = str(act.get("title") or "").strip()
    if not (nid and title):
        return ""
    desc = str(act.get("description") or "").strip()
    diff = act.get("difficulty_1_to_5", act.get("difficulty_0_to_3", None))
    try:
        vb = act.get("expected_vs_baseline_factor", None)
        vb_s = "?" if vb is None else f"{float(vb):.2f}x"
    except Exception:
        vb_s = "?"
    why = str(act.get("rationale") or "").strip()

    lines: list[str] = []
    lines.append("World Model: Chosen Next Action (from decision tree node)")
    lines.append(f"- node_id: {nid}")
    if pid_s:
        lines.append(f"- base_node_id: {pid_s}")
    lines.append(f"- title: {title}")
    if isinstance(diff, int):
        lines.append(f"- difficulty_1_to_5: {int(diff)}")
    if desc:
        lines.append(f"- description: {desc}")
    if vb_s != "?":
        lines.append(f"- expected_vs_baseline_factor: {vb_s}")
    if why:
        lines.append(f"- rationale: {why}")
    return "\n\n" + "\n".join(lines).strip() + "\n"


def render_open_action_nodes_block(
    world_model_json: Optional[str], *, max_items: int = 8
) -> str:
    """
    Deterministic formatting of "candidate actions" when `run()` is deprecated.
    Candidates are OPEN action nodes in the decision tree (node.action.title set, no attached solution_id).
    """
    s = (world_model_json or "").strip()
    if not s:
        return ""
    obj = load_world_model_obj(s)
    if obj is None:
        return ""
    dt = obj.get("decision_tree")
    if not isinstance(dt, dict):
        return ""
    nodes = dt.get("nodes")
    if not isinstance(nodes, list):
        return ""
    root_id = str(dt.get("root_id", "") or "root")
    by_id: dict[str, dict] = {}
    for n in nodes:
        if isinstance(n, dict) and n.get("node_id"):
            by_id[str(n["node_id"])] = n

    def _sid(n: dict) -> Optional[str]:
        sr = n.get("solution_ref")
        if not isinstance(sr, dict):
            return None
        v = sr.get("solution_id")
        return str(v).strip() if isinstance(v, str) and v.strip() else None

    depth_cache: dict[str, int] = {}

    def _depth(nid: str) -> int:
        if nid in depth_cache:
            return depth_cache[nid]
        n = by_id.get(nid)
        if not isinstance(n, dict):
            depth_cache[nid] = 0
            return 0
        pid = n.get("parent_id")
        if pid is None:
            depth_cache[nid] = 0
            return 0
        d = 1 + _depth(str(pid))
        depth_cache[nid] = d
        return d

    def _rating01(n: dict) -> float:
        try:
            return float(n.get("overall_rating_0_to_10", 0.0)) / 10.0
        except Exception:
            return 0.0

    def _conf(n: dict) -> float:
        try:
            return float(n.get("confidence_0_to_1", 0.0) or 0.0)
        except Exception:
            return 0.0

    def _score01(n: dict) -> float:
        act = n.get("action") if isinstance(n.get("action"), dict) else {}
        try:
            s01 = float(act.get("score_0_to_1", 0.0))
        except Exception:
            s01 = 0.0
        if s01 < 0.0:
            s01 = 0.0
        if s01 > 1.0:
            s01 = 1.0
        return s01

    cands: list[dict] = []
    open_total = 0
    for n in nodes:
        if not isinstance(n, dict):
            continue
        if _sid(n) is not None:
            continue
        act = n.get("action")
        if not (isinstance(act, dict) and str(act.get("title") or "").strip()):
            continue
        open_total += 1
        pid = n.get("parent_id")
        if pid is None:
            continue
        pid_s = str(pid)
        if pid_s != root_id:
            parent = by_id.get(pid_s)
            if not isinstance(parent, dict):
                continue
            if _sid(parent) is None:
                continue
        cands.append(n)

    if not cands:
        return "\n\nWorld Model: Open Action Nodes (candidates)\n(none)\n"

    cands.sort(
        key=lambda n: (
            -_rating01(n),
            -_score01(n),
            -_conf(n),
            _depth(str(n.get("node_id") or "")),
            str(n.get("node_id") or ""),
        )
    )

    lines: list[str] = []
    lines.append(f"World Model: Open Action Nodes (candidates) frontier={len(cands)} total_open={open_total}")
    for idx, n in enumerate(cands[: max(1, int(max_items))], start=1):
        nid = str(n.get("node_id") or "").strip()
        pid = str(n.get("parent_id") or "").strip()
        act = n.get("action") if isinstance(n.get("action"), dict) else {}
        title = str(act.get("title") or "").strip()
        desc = str(act.get("description") or "").strip()
        diff = act.get("difficulty_1_to_5", act.get("difficulty_0_to_3", None))
        diff_s = "?"  # 1..5 preferred
        try:
            if diff is not None:
                di = int(diff)
                # Legacy 0..3 -> 1..4
                if "difficulty_1_to_5" not in act and "difficulty_0_to_3" in act:
                    di = di + 1
                if di < 1:
                    di = 1
                if di > 5:
                    di = 5
                diff_s = str(di)
        except Exception:
            diff_s = "?"
        try:
            vb = act.get("expected_vs_baseline_factor", None)
            vb_s = "?" if vb is None else f"{float(vb):.2f}x"
        except Exception:
            vb_s = "?"
        lines.append(
            f"{idx}. rating={float(n.get('overall_rating_0_to_10', 0.0) or 0.0):.1f}/10"
            f" score={_score01(n):.2f} diff={diff_s}/5 conf={_conf(n):.2f} vs_base~{vb_s}"
            f" | node_id={nid} parent_id={pid} | {title}"
            + (f" — {desc}" if desc else "")
        )
    return "\n\n" + "\n".join(lines).strip() + "\n"

@dataclass
class DecisionTreeEditOps:
    """
    LLM-returned edit script for decision_tree updates.
    - ops: list of mutations
    - active_leaf_id: optional override
    """
    ops: list[dict[str, Any]]
    active_leaf_id: Optional[str] = None
    raw_model_output: str = ""


def build_world_model_prompts(
    *,
    definition_text: str,
    target_gpu: str,
    language: str,
    previous_world_model_json: Optional[str],
    current_code_excerpt: Optional[str],
    eval_result: Optional[EvalResult],
    chosen_action_text: Optional[str],
    prediction: Optional[Prediction],
    max_chars_per_block: int = 6000,
) -> WorldModelPrompts:
    """Construct prompts to init/refine the world model as strict JSON."""
    schema = json.dumps(WORLD_MODEL_JSON_SCHEMA_GUIDE, indent=2, sort_keys=True)
    prev = (previous_world_model_json or "").strip()

    init_prompt = (
        "You are a GPU kernel performance engineer.\n"
        "Create an initial WORLD MODEL for the kernel problem below.\n\n"
        f"Target GPU: {target_gpu}\n"
        f"Language: {language}\n\n"
        "Kernel Specification:\n"
        f"{_truncate(definition_text, max_chars_per_block)}\n\n"
        "Return ONLY a single valid JSON object matching this schema guide (keys must exist; fill strings/lists as needed):\n"
        f"{schema}\n\n"
        "Deep analysis requirement (DO THIS FIRST, then write JSON):\n"
        "- You must perform a deep analysis of the specification before proposing action nodes.\n"
        "- You must NOT output the analysis as free-form text. Output JSON ONLY.\n"
        "- Instead, encode the analysis into these JSON fields:\n"
        "  - kernel_summary: 1 compact paragraph capturing the classification + bottleneck hypotheses + key constraints\n"
        "  - open_questions: 3-8 concrete unknowns that most affect correctness/performance\n"
        "  - decision_tree.nodes[0].notes (root node notes): a compact bullet summary of the analysis sections below\n\n"
        "Analysis checklist (use as structure for your root notes; keep it concise):\n"
        "1) Problem classification\n"
        "- Classify the kernel (e.g., matrix-multiply-like, reduction, scan, attention/softmax, elementwise fusion, stencil, gather/scatter, sparse/dynamic indexing, mixed).\n"
        "- Identify primary outputs, whether reductions exist, and whether access is dense vs sparse/indirect.\n"
        "- Identify dimensions that are small&fixed vs large&variable vs runtime-dependent.\n\n"
        "2) Canonical math & dependencies\n"
        "- Rewrite the reference computation in canonical form.\n"
        "- State what is computed per output element, which dims are reduced, and what can run independently.\n"
        "- Identify opportunities for reordering, streaming/online computation, and stable accumulation.\n\n"
        "3) Data layout & access patterns\n"
        "- For each tensor: symbolic shape, dtype, contiguous/strided/transposed vs indirect indexing.\n"
        "- Identify dominant global reads/writes and reuse (thread/warp/block).\n"
        "- Note what can be staged (register/shared) and what is read-once.\n\n"
        "4) Bottleneck hypotheses by regime (>=3 regimes)\n"
        "- Define at least 3 runtime regimes and for each: likely bottleneck (bandwidth/latency/compute/sync) and what triggers it.\n\n"
        "5) Kernel design space (knobs)\n"
        "- Enumerate tunable dimensions: mapping/parallelization, tiling, memory movement, compute strategy, numerics, special-case paths.\n\n"
        "6) High-level kernel skeleton (no code)\n"
        "- Describe phases, what lives in registers vs shared, and where sync is needed.\n\n"
        "7) Candidate kernel families (pruned)\n"
        "- Propose 2-3 families; for each: intended regime, tiling philosophy, memory strategy, strengths/weaknesses, primary limiter.\n\n"
        "- FORBIDDEN: implementation tactics as families (CRITICAL):\n"
        "  - Level-1 Family nodes MUST NOT be named after implementation techniques or optimizations techniques.\n"
        "  - Low-level tactics (e.g., specific instruction names, \"vectorized loads\", \"buffering\", \"pipeline\", \"work stealing\") may appear ONLY as knob choices\n"
        "    inside deeper nodes (depth>=2) and/or inside leaf action notes.\n"
        "Rules:\n"
        "- Output JSON only (no markdown, no commentary).\n"
        "- Be concrete and kernel-specific.\n"
        "- Maintain `decision_tree` (prefix tree): branches are alternatives; each root->leaf path composes a plan.\n"
        "- When you refine/mutate an idea, ADD a new child node (do not delete the parent).\n"
        "- The root node is a DUMMY node: it must have decision=null and choice=null. Real decisions start at depth>=1.\n"
        "- IMPORTANT: Initial action proposal is part of init.\n"
        "  Populate the tree with at least 3 OPEN action nodes (nodes with no attached solution_id but with node.action.title filled).\n"
        "  These actions must be small, single-iteration implementable changes.\n"
        "  For each OPEN action node, fill action.title/action.description concisely.\n"
        "  Avoid hardcoding implementation details like launch/grid/block dims.\n"
        "- Self-check for branching (REQUIRED):\n"
        "  - Whenever you create multiple sibling children under the same parent, treat them as true alternatives.\n"
        "  - In each sibling node's `notes`, add a short `SELF_CHECK` line explaining why this sibling cannot be combined with its siblings.\n"
        "- CRITICAL: Fill ALL numeric fields; do NOT leave placeholders at 0/0.0 unless you truly mean the minimum.\n"
        "  - For EVERY node: set overall_rating_0_to_10 (0..10) and confidence_0_to_1 (0..1).\n"
        "  - For EVERY node: fill impacts.{memory_bandwidth,register_pressure,compute_intensity_and_hw_fit}:\n"
        "    rating_0_to_10 (0..10), risk (non-empty), notes (non-empty).\n"
        "  - For EVERY OPEN action node: set action.score_0_to_1 (0..1) and action.difficulty_1_to_5 (1..5).\n"
        "- IMPORTANT: For each OPEN action node, you MUST set node.action.score_0_to_1 in [0,1].\n"
        "  If unsure, set score_0_to_1 ~= overall_rating_0_to_10 / 10 as a prior.\n"
        "- For each node, fill `impacts` with ratings/risks for:\n"
        "  1) memory_bandwidth, 2) register_pressure, 3) compute_intensity_and_hw_fit (include any relevant hardware shape/layout constraints).\n"
        "- Use overall_rating_0_to_10 to score how good the PARTIAL plan up to that node is.\n"
        "- Update confidence_0_to_1 based on evidence.\n"
        "- Fill open_questions with 3-8 unknowns that would most affect correctness/performance.\n"
    )

    return WorldModelPrompts(init_prompt=init_prompt)


def build_decision_tree_edit_prompt(
    *,
    world_model_json: Optional[str],
    definition_text: str,
    baseline_targets_text: Optional[str] = None,
    debug_and_improve_round: Optional[int] = None,
    debug_and_improve_max_rounds: int = 5,
    target_gpu: str,
    language: str,
    current_code_excerpt: Optional[str],
    current_tree_path: Optional[str],
    wm_status_text: Optional[str] = None,
    open_frontier_nodes_text: Optional[str] = None,
    chosen_action_text: Optional[str],
    prediction: Optional[Prediction],
    eval_result: Optional[EvalResult],
    max_chars: int = 6000,
) -> str:
    """
    Ask the model for a small edit script (ops) to update/insert/split decision tree nodes.
    Output JSON only (no full world model).
    """
    # Budget prompt sections explicitly so total prompt size stays bounded.
    max_chars = int(max_chars)
    # Note: we prioritize showing more of the current code (kernel.cu) and keep other sections tighter.
    def_cap = min(1600, max_chars)
    wm_cap = min(2600, max_chars)
    path_cap = min(800, max_chars)
    code_cap = max_chars
    eval_cap = min(800, max_chars)
    pred_cap = min(800, max_chars)
    status_cap = min(900, max_chars)

    wm_compact = compact_world_model_json_for_prompt(world_model_json or "", max_chars=wm_cap)
    def_s = compact_definition_for_wm_prompt(definition_text or "", max_ref_lines=40)
    pred_s = json.dumps(prediction.__dict__, ensure_ascii=False) if prediction else ""
    eval_s = json.dumps(eval_result.__dict__, ensure_ascii=False) if eval_result else ""
    # Intentionally DO NOT summarize or truncate code here: caller should pass kernel.cu only.
    # (We still keep overall prompt bounded via other sections + the LLM context window.)
    code_s = (current_code_excerpt or "").strip()
    baseline_s = (baseline_targets_text or "").strip()
    baseline_block = (
        "Baseline targets (vs_base objective; higher is better; goal is vs_base>=1):\n"
        + f"{_truncate(baseline_s, 900)}\n\n"
        if baseline_s
        else ""
    )
    frontier_s = (open_frontier_nodes_text or "").strip()
    frontier_block = (
        "Open frontier nodes (highest first): nodes without solution attached but ready to be filled.\n"
        + f"{_truncate(frontier_s, 1200)}\n\n"
        if frontier_s
        else ""
    )
    status_s = (wm_status_text or "").strip()
    status_block = (
        "World model status summary (for planning/exploration):\n" + f"{_truncate(status_s, status_cap)}\n\n"
        if status_s
        else ""
    )
    dai = None
    try:
        dai = int(debug_and_improve_round) if debug_and_improve_round is not None else None
    except Exception:
        dai = None
    dai = dai if isinstance(dai, int) and dai > 0 else None
    try:
        dai_max = int(debug_and_improve_max_rounds)
    except Exception:
        dai_max = 5
    if dai_max < 1:
        dai_max = 1

    schema_hint = (
        "{\n"
        '  "active_leaf_id": "optional node_id",\n'
        '  "ops": [\n'
        '    {"op":"update_node","node_id":"...","patch":{...}},\n'
        '    {"op":"insert_node","parent_id":"...","parent_solution_id":"(required if parent has solution)","node":{...}},\n'
        '    {"op":"split_node","node_id":"...","parent_patch":{...},"children":[{...}, ...]}\n'
        "  ]\n"
        "}\n"
    )

    return (
        "You are the WORLD MODEL module.\n"
        "Output ONLY a JSON edit script (no markdown, no extra text).\n\n"
        f"Target GPU: {target_gpu}\nLanguage: {language}\n\n"
        "Kernel specification (reference):\n"
        f"{def_s}\n\n"
        "Current world model (compact):\n"
        f"{_truncate(wm_compact, wm_cap)}\n\n"
        f"{status_block}"
        f"{frontier_block}"
        "Current solution tree path (root -> ... -> active leaf):\n"
        f"{_truncate(str(current_tree_path or '').strip(), path_cap)}\n\n"
        f"{baseline_block}"
        "Evidence:\n"
        f"- chosen_action:\n{_truncate(str(chosen_action_text or '').strip(), 500)}\n"
        f"- prediction:\n{_truncate(pred_s, pred_cap)}\n"
        f"- eval_result:\n{_truncate(eval_s, eval_cap)}\n"
        f"- debug_and_improve_round: {dai if dai is not None else 'null'} (max {dai_max})\n\n"
        f"- current_code:\n{code_s}\n\n"
        "Reflection (REQUIRED; use this to update ratings/scores):\n"
        "- In your node.notes updates, include:\n"
        "  - CURRENT:\n"
        "    - does: 1-2 sentences on what the current solution does (based on current_code)\n"
        "    - bottleneck: bandwidth vs latency vs compute vs sync vs other\n"
        "  - FOLLOW_THROUGH:\n"
        "    - aligned_with_intent: yes/no (does current_code match the attached node's intended action?)\n"
        "    - if_no: 1 sentence on expected intent vs 1 sentence on what was actually implemented\n"
        "  - UPDATE_BELIEF:\n"
        "    - what changed after eval_result? (1-2 sentences)\n"
        "    - next bet: 1-2 sentences on what to try next and why\n"
        "- You MUST update existing nodes via update_node according to the evidence:\n"
        "  - overall_rating_0_to_10 and confidence_0_to_1\n"
        "  - and node.action.score_0_to_1 (and expected_vs_baseline_factor if a baseline exists)\n"
        "  - and node.action.difficulty_1_to_5 for the relevant OPEN action nodes (especially the chosen/active one)\n"
        "  - and impacts.{memory_bandwidth,register_pressure,compute_intensity_and_hw_fit}.rating_0_to_10/risk/notes for the relevant nodes\n"
        "- CRITICAL: do NOT leave placeholders at 0/0.0. If a value is uncertain, pick a reasonable prior (e.g., rating 4-6, confidence 0.3-0.6).\n"
        "- CRITICAL: you MUST actively revise these numbers when new evidence arrives.\n"
        "  - If eval_result is present, you MUST update: overall_rating_0_to_10, confidence_0_to_1, action.score_0_to_1, action.difficulty_1_to_5,\n"
        "    or impacts.*.rating_0_to_10 for the chosen/active node (and any directly implicated siblings/parents).\n"
        "- If FOLLOW_THROUGH is no, you MUST downgrade the chosen/active node's overall_rating/confidence/score to reflect misalignment.\n"
        "  (Optionally, if you still believe the original intent is good, preserve it as a separate OPEN action node, but keep new nodes <=3.)\n"
        "- If prediction.expected_vs_baseline_factor exists and eval_result.mean_vs_baseline_factor exists and they differ materially,\n"
        "  add a PERF_GAP block to notes with expected vs observed and the hypothesized reason.\n\n"
        "Task:\n"
        "- Propose a SMALL set of edits (update, insert, delete) to improve the tree given the new evidence.\n"
        "- Inserts are capped by the system; keep new nodes <=3 and prefer update_node when possible.\n\n"
        "- Continuation rule (IMPORTANT): if eval_result indicates a PASSED solution was attached to the chosen/active node,\n"
        "  and that node currently has NO child nodes representing a next-step action, then insert AT LEAST ONE child node under it.\n"
        "  This child should be a composable next step (a chain refinement), not an alternative sibling of the parent.\n\n"
        "- Pruning / de-prioritization:\n"
        "  Based on the new evidence, if an OPEN leaf action is now clearly wrong, redundant, dominated, or misaligned, do one of:\n"
        "    (A) downscore it via update_node (lower action.score_0_to_1 and/or overall_rating_0_to_10, and reduce confidence), OR\n"
        "    (B) delete it via delete_node (ONLY allowed for OPEN leaf nodes with NO children and NO attached solution; never delete root).\n"
        "Hard constraints:\n"
        "- Root is dummy: never change root decision/choice.\n"
        "- If inserting under a parent that already has a solution_id, you MUST include parent_solution_id.\n"
        "- New-node cap (HARD): the system will apply at most 3 NEW nodes per edit script.\n"
        "  Count = (#insert_node ops that add a node) + (total children added across all split_node ops).\n"
        "  You MUST propose <= 3 total new nodes. If you have more ideas, pick the best 3 and defer the rest.\n"
        "- Self-check for branching (REQUIRED):\n"
        "  - If you create/maintain multiple sibling children under the same parent, they must be mutually exclusive alternatives.\n"
        "  - When inserting/splitting siblings, update each sibling's `notes` with a short `SELF_CHECK` line explaining why it excludes its siblings.\n"
        "- Sibling constraint (CRITICAL): do NOT keep \"baseline\" and \"optimized\" variants of the SAME design as siblings.\n"
        "  - If an action is a refinement/optimization of another, make it a CHILD under that node.\n"
        "- Kernel families: top-level family nodes (root children) represent end-to-end mapping/ownership plans.\n"
        "  Do NOT name families after tactics; tactics belong in deeper nodes or leaf action notes.\n\n"
        "Output JSON with this shape:\n"
        + schema_hint
    )


def try_parse_decision_tree_edit_ops(text: str) -> Optional[DecisionTreeEditOps]:
    obj = _extract_json_object(text or "")
    if obj is None or not isinstance(obj, dict):
        return None
    ops = obj.get("ops")
    if not isinstance(ops, list):
        return None
    active = obj.get("active_leaf_id")
    active_s = str(active).strip() if isinstance(active, str) and active.strip() else None
    # Keep ops as raw dicts; WorldModelManager will validate/apply.
    return DecisionTreeEditOps(ops=ops, active_leaf_id=active_s, raw_model_output=(text or "").strip())


def _normalize_world_model_obj(obj: dict[str, Any]) -> dict[str, Any]:
    """Backfill required fields while preserving any additional dimensions/fields."""
    if not isinstance(obj, dict):
        return obj

    obj.setdefault("kernel_summary", "")
    obj.setdefault("open_questions", [])
    if not isinstance(obj.get("open_questions"), list):
        obj["open_questions"] = []

    # decision_tree (preferred). Also accept legacy `plan_tree` / `policy_table` / `dimensions` for migration.
    dtree = obj.get("decision_tree")
    if not isinstance(dtree, dict):
        dtree = {}
        obj["decision_tree"] = dtree

    root_id = str(dtree.get("root_id", "") or "").strip() or "root"
    nodes = dtree.get("nodes")
    if not isinstance(nodes, list):
        nodes = []
    active_leaf_id = str(dtree.get("active_leaf_id", "") or "").strip() or root_id

    def _clamp_rating_10(x: Any) -> float:
        try:
            v = float(x)
        except Exception:
            v = 0.0
        if v < 0.0:
            v = 0.0
        if v > 10.0:
            v = 10.0
        return v

    def _clamp_conf(x: Any) -> float:
        try:
            v = float(x)
        except Exception:
            v = 0.0
        if v < 0.0:
            v = 0.0
        if v > 1.0:
            v = 1.0
        return v

    def _normalize_node(n: dict[str, Any], *, fallback_id: str) -> Optional[dict[str, Any]]:
        if not isinstance(n, dict):
            return None
        node_id = str(n.get("node_id", "") or "").strip() or fallback_id
        parent_id = n.get("parent_id", None)
        parent_id = None if parent_id is None else str(parent_id).strip() or None
        # Enforce a single dummy root node. If the model emits extra nodes with parent_id=null,
        # attach them under the canonical root_id instead of creating multiple roots.
        if parent_id is None and node_id != root_id:
            parent_id = root_id
        # Root node is a dummy node (no decision/choice content).
        node_type = "root" if parent_id is None else "decision"
        decision = None if parent_id is None else str(n.get("decision", "") or "").strip()
        # Root node should not have a "choice" (it's the start of the path).
        raw_choice = n.get("choice", None)
        if parent_id is None:
            choice: Optional[str] = None
        else:
            choice = str(raw_choice or "").strip()

        impacts = n.get("impacts")
        if not isinstance(impacts, dict):
            impacts = {}
        def _impact(name: str) -> dict[str, Any]:
            f = impacts.get(name)
            if not isinstance(f, dict):
                f = {}
            out = {
                "rating_0_to_10": _clamp_rating_10(f.get("rating_0_to_10", 0.0)),
                "risk": str(f.get("risk", "") or "").strip(),
                "notes": str(f.get("notes", "") or "").strip(),
            }
            if name == "compute_intensity_and_hw_fit":
                out["hw_notes"] = str(f.get("hw_notes", "") or "").strip()
            return out

        imp_norm = {
            "memory_bandwidth": _impact("memory_bandwidth"),
            "register_pressure": _impact("register_pressure"),
            "compute_intensity_and_hw_fit": _impact("compute_intensity_and_hw_fit"),
        }

        # solution_ref: compact link to SolutionDB
        sr = n.get("solution_ref")
        if not isinstance(sr, dict):
            sr = {}
        sid = sr.get("solution_id", None)
        sid = str(sid).strip() if isinstance(sid, str) and sid.strip() else None
        psid = sr.get("parent_solution_id", None)
        psid = str(psid).strip() if isinstance(psid, str) and psid.strip() else None
        ev = sr.get("eval")
        if not isinstance(ev, dict):
            ev = {}
        sol_eval = dict(ev)
        sol_eval.setdefault("status", str(ev.get("status", "") or "").strip())
        solution_ref = {"solution_id": sid, "parent_solution_id": psid, "eval": sol_eval}

        # Optional: action metadata (structured)
        act = n.get("action")
        act_norm: dict[str, Any] = {
            "title": "",
            "description": "",
            "difficulty_1_to_5": 3,
            "score_0_to_1": 0.0,
            "expected_vs_baseline_factor": None,
            "rationale": "",
        }
        if isinstance(act, dict):
            act_norm["title"] = str(act.get("title", "") or "").strip()
            act_norm["description"] = str(act.get("description", "") or "").strip()
            try:
                # New: difficulty_1_to_5 in [1..5]. Backward compatible with older difficulty_0_to_3 in [0..3].
                d = int(act.get("difficulty_1_to_5", act.get("difficulty_0_to_3", 3)))
            except Exception:
                d = 3
            # If this came from the legacy 0..3 scale, map to 1..4 by +1.
            if "difficulty_1_to_5" not in act and "difficulty_0_to_3" in act:
                try:
                    d = int(act.get("difficulty_0_to_3", 2)) + 1
                except Exception:
                    d = 3
            if d < 1:
                d = 1
            if d > 5:
                d = 5
            act_norm["difficulty_1_to_5"] = d
            try:
                # If the model omits score_0_to_1, keep a deterministic fallback so logs/ranking
                # don't collapse to 0.00 everywhere. We use overall_rating as the prior.
                if "score_0_to_1" in act and act.get("score_0_to_1") is not None:
                    s01 = float(act.get("score_0_to_1", 0.0))
                else:
                    s01 = _clamp_rating_10(n.get("overall_rating_0_to_10", 0.0)) / 10.0
            except Exception:
                s01 = _clamp_rating_10(n.get("overall_rating_0_to_10", 0.0)) / 10.0
            if s01 < 0.0:
                s01 = 0.0
            if s01 > 1.0:
                s01 = 1.0
            act_norm["score_0_to_1"] = s01
            evb = act.get("expected_vs_baseline_factor", None)
            try:
                act_norm["expected_vs_baseline_factor"] = float(evb) if evb is not None else None
            except Exception:
                act_norm["expected_vs_baseline_factor"] = None
            act_norm["rationale"] = str(act.get("rationale", "") or "").strip()

        try:
            lur = int(n.get("last_updated_round", 0))
        except Exception:
            lur = 0

        return {
            "node_id": node_id,
            "parent_id": parent_id,
            "node_type": node_type,
            "decision": decision,
            "choice": choice,
            "solution_ref": solution_ref,
            "impacts": imp_norm,
            "overall_rating_0_to_10": _clamp_rating_10(n.get("overall_rating_0_to_10", 0.0)),
            "confidence_0_to_1": _clamp_conf(n.get("confidence_0_to_1", 0.0)),
            "last_updated_round": lur,
            "notes": str(n.get("notes", "") or "").strip(),
            "action": act_norm,
        }

    norm_nodes: list[dict[str, Any]] = []
    if nodes:
        for i, n in enumerate(nodes):
            nn = _normalize_node(n, fallback_id=f"n{i}")
            if nn is not None:
                norm_nodes.append(nn)
    else:
        # Migration: legacy plan_tree -> each plan node becomes a leaf decision node ("Composite plan").
        legacy_ptree = obj.get("plan_tree")
        if isinstance(legacy_ptree, dict):
            legacy_nodes = legacy_ptree.get("nodes")
            legacy_nodes = legacy_nodes if isinstance(legacy_nodes, list) else []
            # Create a root decision node.
            root = _normalize_node(
                {
                    "node_id": root_id,
                    "parent_id": None,
                    "decision": "Plan root",
                    "choice": None,
                    "overall_rating_0_to_10": 0,
                    "confidence_0_to_1": 0.0,
                },
                fallback_id=root_id,
            )
            if root is not None:
                norm_nodes.append(root)
            for i, pn in enumerate(legacy_nodes[:6]):
                if not isinstance(pn, dict):
                    continue
                plan_name = str(pn.get("plan_name", "") or pn.get("plan_id", "") or f"plan{i}")
                dp = pn.get("dimension_policies") if isinstance(pn.get("dimension_policies"), dict) else {}
                dp = dp if isinstance(dp, dict) else {}
                dp_txt = "; ".join(f"{k}={str(v)[:80]}" for k, v in list(dp.items())[:6] if str(v).strip())
                nn = _normalize_node(
                    {
                        "node_id": f"m{i}",
                        "parent_id": root_id,
                        "decision": "Composite plan (migrated)",
                        "choice": f"{plan_name}: {dp_txt}",
                        "overall_rating_0_to_10": pn.get("overall_rating_0_to_10", 0.0),
                        "confidence_0_to_1": pn.get("confidence_0_to_1", 0.0),
                    },
                    fallback_id=f"m{i}",
                )
                if nn is not None:
                    norm_nodes.append(nn)

        # Migration: legacy policy_table -> root + one child
        legacy_pt = obj.get("policy_table")
        if isinstance(legacy_pt, list) and not norm_nodes:
            root = _normalize_node(
                {"node_id": root_id, "parent_id": None, "decision": "Plan root", "choice": None},
                fallback_id=root_id,
            )
            if root is not None:
                norm_nodes.append(root)
            # Compose a short choice string from best-per-dimension policy_text.
            best_by_dim: dict[str, tuple[float, str]] = {}
            for row in legacy_pt:
                if not isinstance(row, dict):
                    continue
                dim = str(row.get("dimension", "") or "").strip()
                txt = str(row.get("policy_text", "") or "").strip()
                rating = _clamp_rating_10(row.get("rating_0_to_10", 0.0))
                if not dim:
                    continue
                prev = best_by_dim.get(dim)
                if prev is None or rating > prev[0]:
                    best_by_dim[dim] = (rating, txt)
            choice = "; ".join(f"{d}={t[:80]}" for d, (_r, t) in list(best_by_dim.items())[:6] if t)
            child = _normalize_node(
                {"node_id": "migrated_policy", "parent_id": root_id, "decision": "Composite plan (migrated)", "choice": choice},
                fallback_id="migrated_policy",
            )
            if child is not None:
                norm_nodes.append(child)

        # Migration: legacy dimensions -> root + one child
        legacy_dims = obj.get("dimensions")
        if isinstance(legacy_dims, dict) and not norm_nodes:
            root = _normalize_node(
                {"node_id": root_id, "parent_id": None, "decision": "Plan root", "choice": None},
                fallback_id=root_id,
            )
            if root is not None:
                norm_nodes.append(root)
            parts: list[str] = []
            for dim_name, dim_val in list(legacy_dims.items())[:8]:
                if not isinstance(dim_val, dict):
                    continue
                hyp = str(dim_val.get("hypothesis", "") or "").strip()
                if hyp:
                    parts.append(f"{dim_name}={hyp[:80]}")
            child = _normalize_node(
                {"node_id": "migrated_dims", "parent_id": root_id, "decision": "Composite plan (migrated)", "choice": "; ".join(parts)},
                fallback_id="migrated_dims",
            )
            if child is not None:
                norm_nodes.append(child)

    # Ensure root exists.
    ids = {n["node_id"] for n in norm_nodes}
    if root_id not in ids:
        root = _normalize_node(
            {
                "node_id": root_id,
                "parent_id": None,
                "decision": "Plan root",
                "choice": None,
            },
            fallback_id=root_id,
        )
        if root is not None:
            norm_nodes.insert(0, root)
        ids = {n["node_id"] for n in norm_nodes}

    # Fix bad parent pointers.
    for n in norm_nodes:
        if n.get("parent_id") is not None and n["parent_id"] not in ids:
            n["parent_id"] = root_id

    if active_leaf_id not in ids:
        active_leaf_id = root_id

    obj["decision_tree"] = {"root_id": root_id, "active_leaf_id": active_leaf_id, "nodes": norm_nodes}

    # Drop legacy keys; decision_tree is the only maintained representation.
    for k in ("plan_tree", "policy_table", "dimensions", "plan_table"):
        if k in obj:
            try:
                del obj[k]
            except Exception:
                pass

    # Computed signals are machine-generated and can exist even if the LLM omits them.
    cs = obj.get("computed_signals")
    if not isinstance(cs, dict):
        cs = {}
        obj["computed_signals"] = cs
    cs.setdefault("round_index", 0)
    cs.setdefault("trace", {})
    if not isinstance(cs.get("trace"), dict):
        cs["trace"] = {}
    tr = cs["trace"]
    tr.setdefault("status", "")
    tr.setdefault("latency_ms", None)
    tr.setdefault("reference_latency_ms", None)
    tr.setdefault("speedup_factor", None)

    return obj


def try_parse_world_model_json(text: str) -> Optional[str]:
    obj = _extract_json_object(text or "")
    if obj is None:
        return None
    obj = _normalize_world_model_obj(obj)
    try:
        return json.dumps(obj, indent=2, sort_keys=True)
    except Exception:
        return None


def merge_computed_signals(
    *,
    world_model_json: Optional[str],
    round_index: Optional[int],
    eval_result: Optional[EvalResult],
) -> Optional[str]:
    """Merge machine-derived signals into the world model JSON (preserving user/LLM content)."""
    base_obj = _extract_json_object(world_model_json or "")
    if base_obj is None:
        return world_model_json
    base_obj = _normalize_world_model_obj(base_obj)

    cs = base_obj.get("computed_signals")
    if not isinstance(cs, dict):
        cs = {}
        base_obj["computed_signals"] = cs
    cs.setdefault("trace", {})
    if not isinstance(cs.get("trace"), dict):
        cs["trace"] = {}

    if round_index is not None:
        try:
            cs["round_index"] = int(round_index)
        except Exception:
            pass

    t = cs["trace"]
    if eval_result is not None:
        try:
            t["status"] = str(eval_result.status or "")
        except Exception:
            pass
        if eval_result.latency_ms is not None:
            try:
                t["latency_ms"] = float(eval_result.latency_ms)
            except Exception:
                pass
        if eval_result.reference_latency_ms is not None:
            try:
                t["reference_latency_ms"] = float(eval_result.reference_latency_ms)
            except Exception:
                pass
        if eval_result.speedup_factor is not None:
            try:
                t["speedup_factor"] = float(eval_result.speedup_factor)
            except Exception:
                pass

    try:
        return json.dumps(base_obj, indent=2, sort_keys=True)
    except Exception:
        return world_model_json


def build_action_ranking_prompt(
    *,
    definition_text: str,
    baseline_targets_text: Optional[str] = None,
    open_frontier_nodes_text: Optional[str] = None,
    current_code_excerpt: str,
    current_active_node_id: str,
    eval_result: Optional[EvalResult],
    target_gpu: str,
    language: str,
    world_model_json: Optional[str],
    max_chars: int = 6000,
) -> str:
    """
    Ask the model to propose AND rank 5 actions for what to try next.
    We do not hardcode the action list; the model must generate them.
    Output JSON only.
    """
    # Budget prompt sections explicitly so total prompt size stays bounded.
    max_chars = int(max_chars)
    # Prefer showing more of the current code and keep other sections tighter.
    def_cap = min(1600, max_chars)
    wm_cap = min(2600, max_chars)
    code_cap = max_chars
    eval_cap = min(800, max_chars)

    wm = (world_model_json or "").strip()
    wm_compact = compact_world_model_json_for_prompt(wm, max_chars=wm_cap)
    def_s = compact_definition_for_wm_prompt(definition_text or "", max_ref_lines=35)
    eval_s = json.dumps(eval_result.__dict__, ensure_ascii=False) if eval_result else ""
    # Intentionally DO NOT summarize or truncate code here: caller should pass kernel.cu only.
    code_s = (current_code_excerpt or "").strip()
    baseline_s = (baseline_targets_text or "").strip()
    baseline_block = (
        "Baseline target (vs_base objective; higher is better; goal is vs_base>=1):\n"
        + f"{_truncate(baseline_s, 800)}\n\n"
        if baseline_s
        else ""
    )
    frontier_s = (open_frontier_nodes_text or "").strip()
    frontier_block = (
        "Open frontier nodes (highest first): nodes that currently have NO attached solution but are ready to be filled.\n"
        "If you target one of these, set action.attach_to_node_id to that node_id and set base_node_id to its parent.\n"
        + f"{_truncate(frontier_s, 1200)}\n\n"
        if frontier_s
        else ""
    )
    example = {
        "actions": [
            {
                "action_id": "a1",
                "title": "Small: change work partitioning for stage-1 (single change)",
                "description": "Keep math identical; change only ONE partitioning choice in stage-1 and keep everything else the same.",
                "difficulty_1_to_5": 3,
                "base_node_id": "n12",
                "attach_to_node_id": "pending_leaf_1",
            },
            {
                "action_id": "a2",
                "title": "Small: adjust warp scheduling (only) to reduce register pressure",
                "description": "Keep sharding/layout fixed; change only warp scheduling/ordering to reduce live ranges and spills.",
                "difficulty_1_to_5": 2,
                "base_node_id": "n12",
            },
            {
                "action_id": "a3",
                "title": "Pipeline global→shared staging",
                "description": "Increase overlap with multi-stage staging (num_stages) and re-order loads/compute to hide memory latency; watch smem pressure.",
                "base_node_id": "n12",
            },
            {
                "action_id": "a4",
                "title": "Memory coalescing / vectorization",
                "description": "Re-layout or vectorize loads/stores (block pointers, alignment hints) to reduce transactions and bank conflicts.",
                "base_node_id": "n3",
            },
            {
                "action_id": "a5",
                "title": "Free explore (non-incremental)",
                "description": "Try an alternative algorithmic structure (e.g., different staging or decomposition) even if it changes multiple knobs at once.",
                "difficulty_1_to_5": 4,
                "base_node_id": "root",
            },
        ],
        "ranking": [
            {"action_id": "a1", "score": 0.82, "reason": "Current code shows high reg pressure signals; splitting head_dim should reduce spills while keeping reuse."},
            {"action_id": "a2", "score": 0.70, "reason": "If using advanced math paths, operand layout and tile shape are likely limiting throughput."},
            {"action_id": "a3", "score": 0.55, "reason": "Pipelining can help if memory latency dominates, but may increase smem usage."},
            {"action_id": "a4", "score": 0.40, "reason": "Worth trying if loads are not coalesced, but impact uncertain without more evidence."},
            {"action_id": "a5", "score": 0.25, "reason": "Exploration is useful, but prioritize more targeted changes first."},
        ],
        "prediction": {
            "expected_speedup_factor": 1.15,
            "expected_latency_ms": None,
            "expected_vs_baseline_factor": 1.05,
            "confidence": 0.55,
            "rationale": "Reducing register pressure should improve occupancy and reduce spill traffic."
        },
    }

    return (
        "You are the WORLD MODEL module.\n"
        "Your job is to propose AND rank 5 candidate actions for what to try next.\n"
        "Do NOT rewrite the kernel prompt. Do NOT generate code.\n\n"
        f"Target GPU: {target_gpu}\n"
        f"Language: {language}\n\n"
        f"Current active node id: {str(current_active_node_id or '')}\n\n"
        "Kernel specification (reference; do not restate):\n"
        f"{def_s}\n\n"
        "Persistent World Model (COMPACT view):\n"
        f"{_truncate(wm_compact, wm_cap)}\n\n"
        f"{frontier_block}"
        "Current implementation (kernel.cu only if CUDA; full text provided):\n"
        f"{code_s}\n\n"
        f"{baseline_block}"
        "Most recent eval_result for current code (may be empty):\n"
        f"{_truncate(eval_s, eval_cap)}\n\n"
        "Return ONLY valid JSON with:\n"
        "- exactly 5 proposed actions\n"
        "- a 5-item ranking (best-to-worst)\n"
        "- a short prediction of expected outcome if the TOP-1 action is applied next\n"
        "\n"
        "{\n"
        '  "actions": [\n'
        '    {"action_id": "a1", "title": "...", "description": "...", "difficulty_1_to_5": 3, "base_node_id": "node_id_here", "attach_to_node_id": "optional_existing_leaf_node_id"},\n'
        "    ... 5 total ...\n"
        "  ],\n"
        '  "ranking": [\n'
        '    {"action_id": "a1", "score": 0.0, "reason": "..."},\n'
        "    ... 5 total ...\n"
        "  ],\n"
        '  "prediction": {\n'
        '    "expected_speedup_factor": null,\n'
        '    "expected_latency_ms": null,\n'
        '    "expected_vs_baseline_factor": null,\n'
        '    "confidence": 0.5,\n'
        '    "rationale": ""\n'
        "  }\n"
        "}\n"
        "\nExample (valid output):\n"
        f"{json.dumps(example, indent=2, ensure_ascii=False)}\n"
        "Rules:\n"
        "- You MUST create 5 actions. Do not reuse action_ids.\n"
        "- Each ranking entry's action_id must match one of the 5 actions.\n"
        "- Each action.base_node_id must be a node_id that exists in the decision_tree.\n"
        "- If you set action.attach_to_node_id, it MUST be one of the provided Open frontier nodes AND its parent must equal base_node_id.\n"
        "- The 5 actions MAY use 5 DIFFERENT base_node_id values.\n"
        "- Portfolio constraint: include at least 2 exploration/structural actions BEFORE low-level tuning.\n"
        "- At most 2 actions may be pure tuning (hardware-specific micro-tuning).\n"
        "- Each action must be single-iteration implementable and SMALL:\n"
        "  - Prefer a single concrete tweak (one axis split OR one warp-group mapping OR one pipeline stage change OR one layout/vectorization change).\n"
        "  - Avoid bundling multiple large features into one action.\n"
        "  - If you mention any hardware/ISA-specific feature, keep it to ONE minimal step and keep the rest unchanged.\n"
        "- Each action MUST include `difficulty_1_to_5` in [1..5].\n"
        "- CRITICAL: Set action.score_0_to_1 explicitly (0..1). Do NOT leave it at 0.0 unless you truly believe it is the worst option.\n"
        "- Strong preference: propose actions with difficulty_1_to_5 <= 3. Actions with difficulty > 3 will likely be delayed until the base solution improves.\n"
        "- score is a float in [0, 1].\n"
        "- reason is 1-2 sentences, grounded in world model + current code excerpt + eval_result.\n"
        "- Actions should be specific (e.g., propose concrete tiling/scheduling/layout/pipeline changes), not generic advice.\n"
        "- prediction confidence must be in [0,1]; other prediction fields may be null.\n"
        "- If a baseline target is provided, prediction.expected_vs_baseline_factor should be your primary predicted metric.\n"
        "- Output JSON only (no markdown, no backticks, no commentary).\n"
    )


def try_parse_action_ranking_json(
    text: str,
) -> Optional[tuple[list[ActionCandidate], list[ActionRankItem], Prediction | None]]:
    obj = _extract_json_object(text or "")
    if not obj or not isinstance(obj.get("ranking"), list) or not isinstance(obj.get("actions"), list):
        return None
    candidates_all: list[ActionCandidate] = []
    for a in obj.get("actions", []):
        if not isinstance(a, dict):
            continue
        aid = str(a.get("action_id", "") or "").strip()
        title = str(a.get("title", "") or "").strip()
        desc = str(a.get("description", "") or "").strip()
        bn = a.get("base_node_id", None)
        base_node_id = str(bn).strip() if isinstance(bn, str) and bn.strip() else None
        at = a.get("attach_to_node_id", None)
        attach_to_node_id = str(at).strip() if isinstance(at, str) and at.strip() else None
        if not aid:
            continue
        diff15 = a.get("difficulty_1_to_5", None)
        diff03 = a.get("difficulty_0_to_3", None)
        diff_i: int | None = None
        try:
            if diff15 is not None:
                diff_i = int(diff15)
            elif diff03 is not None:
                diff_i = int(diff03) + 1  # legacy 0..3 -> 1..4
        except Exception:
            diff_i = None
        if isinstance(diff_i, int):
            if diff_i < 1:
                diff_i = 1
            if diff_i > 5:
                diff_i = 5
        candidates_all.append(
            ActionCandidate(
                action_id=aid,
                title=title,
                description=desc,
                base_node_id=base_node_id,
                attach_to_node_id=attach_to_node_id,
                difficulty_1_to_5=diff_i,
                difficulty_0_to_3=None,
            )
        )
    # Be tolerant: accept >=5, keep first 5 unique action_ids.
    seen_a: set[str] = set()
    candidates: list[ActionCandidate] = []
    for c in candidates_all:
        if c.action_id in seen_a:
            continue
        seen_a.add(c.action_id)
        candidates.append(c)
        if len(candidates) >= 5:
            break
    if len(candidates) != 5:
        return None
    allowed = {c.action_id for c in candidates}
    out: list[ActionRankItem] = []
    for it in obj["ranking"]:
        if not isinstance(it, dict):
            continue
        aid = str(it.get("action_id", "")).strip()
        if aid not in allowed:
            continue
        try:
            score = float(it.get("score", 0.0))
        except Exception:
            score = 0.0
        if score < 0.0:
            score = 0.0
        if score > 1.0:
            score = 1.0
        reason = str(it.get("reason", "")).strip()
        out.append(
            ActionRankItem(
                action_id=aid,
                score=score,
                reason=reason,
            )
        )
    # Enforce uniqueness + count.
    seen: set[str] = set()
    dedup: list[ActionRankItem] = []
    for r in out:
        if r.action_id in seen:
            continue
        seen.add(r.action_id)
        dedup.append(r)
    if len(dedup) != 5:
        return None

    pred_obj = obj.get("prediction")
    pred: Prediction | None = None
    if isinstance(pred_obj, dict):
        try:
            conf = float(pred_obj.get("confidence", 0.5))
        except Exception:
            conf = 0.5
        if conf < 0.0:
            conf = 0.0
        if conf > 1.0:
            conf = 1.0

        def _f(x):
            try:
                return float(x) if x is not None else None
            except Exception:
                return None

        pred = Prediction(
            expected_speedup_factor=_f(pred_obj.get("expected_speedup_factor")),
            expected_latency_ms=_f(pred_obj.get("expected_latency_ms")),
            expected_vs_baseline_factor=_f(pred_obj.get("expected_vs_baseline_factor")),
            confidence=conf,
            rationale=str(pred_obj.get("rationale", "") or "").strip(),
        )

    return candidates, dedup, pred


def render_action_ranking_block(r: ActionRanking) -> str:
    """Deterministic formatting (no LLM-generated prompt text)."""
    if not r.ranking:
        return ""
    by_id = {c.action_id: c for c in r.candidates}
    lines: list[str] = []
    lines.append("World Model: Ranked Next Actions (LLM-proposed)")
    if r.prediction is not None:
        p = r.prediction
        vb = "?" if getattr(p, "expected_vs_baseline_factor", None) is None else f"{p.expected_vs_baseline_factor:.2f}x"
        sp = "?" if p.expected_speedup_factor is None else f"{p.expected_speedup_factor:.2f}x"
        lt = "?" if p.expected_latency_ms is None else f"{p.expected_latency_ms:.3f} ms"
        pred_metric = (
            f"vs_baseline={vb}"
            if getattr(p, "expected_vs_baseline_factor", None) is not None
            else f"speedup_vs_ref={sp}"
        )
        lines.append(f"Prediction (if apply #1): {pred_metric}, latency={lt}, confidence={p.confidence:.2f}")
        if p.rationale:
            lines.append(f"  - {p.rationale}")
    for idx, item in enumerate(r.ranking, start=1):
        c = by_id.get(item.action_id)
        title = c.title if c else item.action_id
        route_parts: list[str] = []
        if item.base_node_id:
            route_parts.append(f"base_node_id={item.base_node_id}")
        if item.base_solution_id:
            route_parts.append(f"base_solution_id={item.base_solution_id}")
        if item.attach_to_node_id:
            route_parts.append(f"attach_to_node_id={item.attach_to_node_id}")
        route = (" | " + ", ".join(route_parts)) if route_parts else ""
        lines.append(f"{idx}. [{item.score:.2f}] {title} ({item.action_id}){route}")
        if item.reason:
            lines.append(f"   - {item.reason}")
    return "\n\n" + "\n".join(lines)


