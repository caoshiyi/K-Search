from __future__ import annotations

from typing import Any


RELATIVE_TO_VALUES = {
    "original_baseline",
    "parent_strategy",
    "reference_baseline",
    "current_parent_solution",
    "unspecified_legacy",
}


def ms_to_us(value_ms: float | int | None) -> float | None:
    if value_ms is None:
        return None
    if isinstance(value_ms, bool) or not isinstance(value_ms, (int, float)):
        return None
    return float(value_ms) * 1000.0


def validate_expected_speedup(
    raw: dict[str, Any],
    *,
    allow_unspecified_legacy: bool = False,
) -> dict[str, Any]:
    if not isinstance(raw, dict):
        raise ValueError("expected_speedup must be an object")
    out = dict(raw)
    factor = out.get("factor")
    if isinstance(factor, bool) or not isinstance(factor, (int, float)):
        raise ValueError("expected_speedup.factor must be numeric")
    out["factor"] = float(factor)
    relative_to = str(out.get("relative_to") or "").strip()
    if relative_to not in RELATIVE_TO_VALUES:
        raise ValueError("expected_speedup.relative_to is invalid or missing")
    if relative_to == "unspecified_legacy" and not allow_unspecified_legacy:
        raise ValueError("expected_speedup.relative_to must not be unspecified_legacy for new schema")
    out["relative_to"] = relative_to
    if relative_to == "parent_strategy":
        parent_strategy_id = str(out.get("parent_strategy_id") or "").strip()
        if not parent_strategy_id:
            raise ValueError("expected_speedup.parent_strategy_id is required for parent_strategy")
        out["parent_strategy_id"] = parent_strategy_id
    source = str(out.get("source") or "strategy_catalog").strip()
    out["source"] = source
    for key in ("parent_latency_us", "target_latency_us"):
        value = out.get(key)
        if value is None:
            continue
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise ValueError(f"expected_speedup.{key} must be numeric")
        out[key] = float(value)
    return out


def normalize_expected_speedup(
    action_or_entry: dict[str, Any],
    *,
    strategy_requires: tuple[str, ...] | list[str] = (),
    allow_unspecified_legacy: bool = True,
) -> dict[str, Any] | None:
    if not isinstance(action_or_entry, dict):
        return None
    if isinstance(action_or_entry.get("expected_speedup"), dict):
        return validate_expected_speedup(
            action_or_entry["expected_speedup"],
            allow_unspecified_legacy=allow_unspecified_legacy,
        )
    legacy = action_or_entry.get("expected_vs_baseline_factor")
    if legacy is None:
        return None
    if isinstance(legacy, bool) or not isinstance(legacy, (int, float)):
        raise ValueError("expected_vs_baseline_factor must be numeric")
    requires = tuple(str(item).strip() for item in strategy_requires if str(item).strip())
    if requires:
        speedup: dict[str, Any] = {
            "factor": float(legacy),
            "relative_to": "parent_strategy",
            "parent_strategy_id": requires[-1],
            "source": "legacy_expected_vs_baseline_factor",
        }
    else:
        speedup = {
            "factor": float(legacy),
            "relative_to": "unspecified_legacy" if allow_unspecified_legacy else "original_baseline",
            "source": "legacy_expected_vs_baseline_factor",
        }
    return validate_expected_speedup(speedup, allow_unspecified_legacy=allow_unspecified_legacy)
