import pytest

from k_search.kernel_generators.prompt_hygiene import (
    PromptHygieneError,
    check_prompt_hygiene,
    check_prompt_hygiene_or_raise,
)


def test_prompt_hygiene_blocks_path_leak():
    prompt = "Read /tmp/ksearch_worktrees/abc/kernel.cpp"

    with pytest.raises(PromptHygieneError, match="absolute path"):
        check_prompt_hygiene_or_raise(prompt)


def test_prompt_hygiene_flags_noisy_logs():
    prompt = "Recent failure:\n[benchmark stderr]\nPermission mismatch"

    hygiene = check_prompt_hygiene(prompt)

    assert hygiene["contains_raw_benchmark_stderr"] is True
    assert hygiene["contains_permission_mismatch"] is True


def test_prompt_hygiene_flags_global_flow_policy():
    hygiene = check_prompt_hygiene(
        "Subagent usage policy:\n"
        "- Initial codegen flow agents: code-reader, designer, codegen, reviewer.\n"
    )

    assert hygiene["contains_global_flow_policy"] is True
    with pytest.raises(PromptHygieneError, match="global flow policy"):
        check_prompt_hygiene_or_raise("Required native subagent flow: code-reader -> designer")
