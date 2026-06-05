---
name: ascendc-codegen
description: Use for K-Search AscendC candidate optimization attempts that need code-reader, plan, codegen, and reviewer coordination inside a project worktree.
---

# AscendC Codegen Workflow

You are working inside a K-Search candidate worktree.

Use this workflow:
1. Use code-reader when CODE_MAP.md is missing or stale.
2. Use plan to write IMPLEMENTATION_PLAN.md.
3. Use codegen to edit source files according to IMPLEMENTATION_PLAN.md.
4. Use reviewer to write REVIEW_NOTES.md.
5. Return only a concise final summary to the parent.

Cross-agent state must be written to files:
- CODE_MAP.md
- IMPLEMENTATION_PLAN.md
- REVIEW_NOTES.md

Do not paste those files into final messages.
Do not run Bash.
Do not edit outside the current working directory.
Preserve public entry points, host tiling contract, kernel launch behavior, correctness harness behavior, and build layout.
