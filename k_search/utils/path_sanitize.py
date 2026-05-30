from __future__ import annotations

import re

# 匹配 ksearch 临时 worktree / fallback 临时 repo 的绝对路径根:
#   /<任意前缀>/ksearch_agentic_worktree_<rand>
#   /<任意前缀>/ksearch_agentic_temp_repo_<rand>
# 捕获到随机根目录为止,其后的相对子路径保留不动。
# 后缀 [A-Za-z0-9]+ 对应 tempfile.mkdtemp 产出的随机名(见 k_search/kernel_generators/agentic_worktree.py 的
# prefix="ksearch_agentic_worktree_" / "ksearch_agentic_temp_repo_",无 suffix= 故为纯字母数字)。
_WORKTREE_ROOT_RE = re.compile(
    r"/[^\s]*?/(?:ksearch_agentic_worktree|ksearch_agentic_temp_repo)_[A-Za-z0-9]+"
)


def sanitize_worktree_paths(
    text: str,
    *,
    placeholder: str = "<PROJECT_ROOT>",
    task_path: str | None = None,
) -> str:
    """把任意 ksearch 临时 worktree / 临时 repo 的绝对路径前缀替换为语义占位符。

    同时,如果 caller 提供了原始任务目录的 task_path,也将其替换为 placeholder。
    这防止 LLM agent 从 prompt 中读出原始绝对路径后,绕过 worktree 去修改源目录。

    用通配正则而非精确字符串,故任意历史轮次的残留路径都会被替换,无需知道
    "当前 worktree 是谁";天然幂等。
    """
    if not text:
        return text
    out = _WORKTREE_ROOT_RE.sub(placeholder, text)
    if task_path:
        # 替换原始任务目录路径(含子路径)为 placeholder + 相对部分。
        # 例如: /home/user/cv_agent/tile2asc/multi_query_attention/ksearch_task.md
        #       → <PROJECT_ROOT>/ksearch_task.md
        task_prefix = str(task_path).rstrip("/")
        # 先精确匹配 task_path 本身,再匹配 task_path/ 子路径。
        # 用 str.replace 简单高效;regex 不必要,因为 task_path 是确定值。
        out = out.replace(task_prefix, placeholder)
    return out