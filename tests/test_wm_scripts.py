from __future__ import annotations

import os
import subprocess
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def _run_bash(script: str, *, env: dict[str, str] | None = None) -> subprocess.CompletedProcess[str]:
    merged_env = dict(os.environ)
    merged_env.update(
        {
            "ANTHROPIC_AUTH_TOKEN": "dummy-token",
            "ANTHROPIC_BASE_URL": "https://example.invalid",
        }
    )
    if env:
        merged_env.update(env)
    return subprocess.run(
        ["bash", "-c", script],
        cwd=str(ROOT),
        env=merged_env,
        text=True,
        capture_output=True,
        check=False,
    )


def _sourceable_script_copy(
    source: Path,
    tmp_path: Path,
    *,
    stop_before: str | None = None,
    remove_line: str | None = None,
) -> Path:
    text = source.read_text(encoding="utf-8")
    if stop_before is not None:
        text = text.split(stop_before, 1)[0]
    if remove_line is not None:
        text = "\n".join(line for line in text.splitlines() if line.strip() != remove_line)
        text += "\n"
    target = tmp_path / source.name
    target.write_text(text, encoding="utf-8")
    return target


def test_ascendc_fa_script_uses_documented_default_baseline_when_unset(tmp_path):
    script = ROOT / "scripts" / "ascendc_fa_wm.sh"
    sourceable = _sourceable_script_copy(script, tmp_path, stop_before='cd "$KSEARCH_ROOT"')
    proc = _run_bash(
        f"""
        set -euo pipefail
        unset BASELINE_MS
        source {sourceable}
        test "$BASELINE_MS" = "0.394"
        """
    )

    assert proc.returncode == 0, proc.stderr + proc.stdout


def test_flash_attention_script_preserves_user_baseline(tmp_path):
    script = ROOT / "scripts" / "flash_attention_wm.sh"
    sourceable = _sourceable_script_copy(script, tmp_path, remove_line='main "$@"')
    proc = _run_bash(
        f"""
        set -euo pipefail
        export BASELINE_MS=0.5
        source {sourceable}
        test "$BASELINE_MS" = "0.5"
        """
    )

    assert proc.returncode == 0, proc.stderr + proc.stdout


def test_flash_attention_script_does_not_preseed_baseline_when_unset(tmp_path):
    script = ROOT / "scripts" / "flash_attention_wm.sh"
    sourceable = _sourceable_script_copy(script, tmp_path, remove_line='main "$@"')
    proc = _run_bash(
        f"""
        set -euo pipefail
        unset BASELINE_MS
        source {sourceable}
        test -z "${{BASELINE_MS:-}}"
        """
    )

    assert proc.returncode == 0, proc.stderr + proc.stdout
