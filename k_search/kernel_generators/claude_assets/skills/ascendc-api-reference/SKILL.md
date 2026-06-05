---
name: ascendc-api-reference
description: Use when AscendC API signatures, constraints, examples, or anti-patterns are needed from K-Search references/ docs.
---

# AscendC API Reference Lookup

K-Search stores AscendC API documentation under references/.
When strategy text lists a path like references/api_reference_docs/.../RowMuls.md, read that file before editing code that uses the API.

Rules:
- Do not guess AscendC API signatures from general C++ experience.
- Prefer strategy-provided API summaries first, then read full docs for details.
- Treat anti-pattern warnings in strategy text as hard constraints.
- If a signature or constraint is missing, report it in IMPLEMENTATION_PLAN.md instead of inventing a call.
