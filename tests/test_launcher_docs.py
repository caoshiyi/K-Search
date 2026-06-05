import os
import subprocess
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]


def test_mla_decode_launcher_accepts_llm_api_key_fallback(tmp_path):
    fake_bin = tmp_path / "bin"
    fake_bin.mkdir()
    sudo_marker = tmp_path / "sudo-called"
    fake_sudo = fake_bin / "sudo"
    fake_sudo.write_text(
        "#!/bin/sh\n"
        "printf '%s\\n' \"$@\" > \"$SUDO_MARKER\"\n"
        "exit 0\n"
    )
    fake_sudo.chmod(0o755)

    env = os.environ.copy()
    env.pop("API_KEY", None)
    env.update(
        {
            "PATH": f"{fake_bin}{os.pathsep}{env['PATH']}",
            "LLM_PROVIDER": "openai",
            "LLM_API_KEY": "env-key",
            "KSEARCH_ROOT": str(REPO_ROOT),
            "DATASET_ROOT": str(tmp_path / "dataset"),
            "SUDO_MARKER": str(sudo_marker),
        }
    )

    result = subprocess.run(
        ["bash", str(REPO_ROOT / "scripts" / "mla_decode_wm.sh")],
        cwd=REPO_ROOT,
        env=env,
        capture_output=True,
        text=True,
        check=False,
    )

    assert result.returncode == 0, result.stderr
    assert sudo_marker.exists()
    sudo_args = sudo_marker.read_text().splitlines()
    api_key_arg_index = sudo_args.index("--api-key")
    assert sudo_args[api_key_arg_index + 1] == "env-key"


def test_readme_documents_claude_agent_sdk_installation():
    readme = (REPO_ROOT / "README.md").read_text()
    claude_section = readme.split("### Claude Agent SDK Backend", 1)[1]

    assert "uv pip install claude-agent-sdk" in claude_section
    assert "Claude+AscendC uses agentic worktree codegen by default" in claude_section
    assert "KSEARCH_AGENTIC_PROMPT_MAX_CHARS" in claude_section
    assert "KSEARCH_KEEP_AGENTIC_WORKTREES" in claude_section
    assert 'permission_mode="dontAsk"' in claude_section
    assert "`Read`, `Grep`, `Glob`, `Edit`, `Write`, `Skill`, and `Agent`" in claude_section
