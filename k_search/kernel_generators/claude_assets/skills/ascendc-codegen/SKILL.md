---
name: ascendc-codegen
description: Use for K-Search AscendC candidate optimization attempts that need code-reader, designer, codegen, and reviewer coordination inside a project worktree.
---

# AscendC Codegen Workflow

You are working inside a K-Search candidate worktree.

Use this workflow:
1. Use code-reader when CODE_MAP.md is missing or stale.
2. Use designer to write ASCENDC_DESIGN.md as the detailed design.
3. Use codegen to write IMPLEMENTATION_EXECUTION_PLAN.md, edit source files according to ASCENDC_DESIGN.md, and write IMPLEMENTATION_HANDOFF.md.
4. Use reviewer to write REVIEW_NOTES.md.
5. Return only a concise final summary to the parent.

Cross-agent state must be written to files:
- CODE_MAP.md
- ASCENDC_DESIGN.md
- IMPLEMENTATION_EXECUTION_PLAN.md
- IMPLEMENTATION_HANDOFF.md
- IMPLEMENTATION_DEVIATIONS.md when implementation deviates from design
- REVIEW_NOTES.md

Do not paste those files into final messages.
Do not run Bash.
Do not edit outside the current working directory.
Preserve public entry points, host tiling contract, kernel launch behavior, correctness harness behavior, and build layout.
