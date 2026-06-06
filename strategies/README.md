# Strategy Catalogs

K-Search strategy injection now supports one strategy form: natural-language
markdown referenced from a concise JSON catalog.

## Catalog Schema

`mqa_strategies_catalog.json` is a metadata index. It should contain short
summaries and relative markdown references only:

```json
{
  "version": 2,
  "strategy_form": "natural_language",
  "strategies": [
    {
      "id": "reuse_ub_cache",
      "title": "Reuse UB cache to reduce repeated GM loads",
      "summary": "Cache reused tiles in UB/L1 to reduce redundant global-memory reads.",
      "markdown_ref": "mqa_strategies/reuse_ub_cache.md",
      "tags": ["memory", "ub", "data-movement"],
      "difficulty_1_to_5": 3,
      "score_0_to_1": 0.72,
      "expected_vs_baseline_factor": 1.05
    }
  ]
}
```

Rules:

- `id`, `title`, `summary`, and `markdown_ref` are required.
- `markdown_ref` must be a relative `.md` path under this directory.
- `summary` should stay concise; the loader rejects summaries over 1200 chars.
- `structured_params`, `dsl`, and inline `natural_language` fields are rejected by default.

## Markdown Files

The full strategy body lives in markdown and is loaded only after the world
model selects that action. K-Search does not parse markdown sections; it injects
the file text as the selected strategy document.

Example:

```markdown
# Reuse UB cache to reduce repeated GM loads

## Intent
Reduce redundant global-memory reads by keeping frequently reused tensors,
tiles, or intermediate values in UB/L1 when lifetime and capacity allow.

## Implementation checklist
1. Identify the repeated GM read path.
2. Check UB/L1 capacity.
3. Reorder loops or cache the tile.
4. Preserve boundary handling and alignment.
5. Re-run correctness first, then benchmark.

## Risks
- UB overflow.
- Higher register pressure.
- Incorrect lifetime across pipeline stages.
```

## Usage

```bash
python generate_kernels_and_eval.py \
  --task-source ascendc \
  --task-path /path/to/op \
  --model-name claude-sonnet-4-6 \
  --llm-provider claude-agent \
  --language ascendc \
  --world-model \
  --strategy-file strategies/mqa_strategies_catalog.json
```

`--strategy-form` is optional and only accepts `natural_language`; omit it for
new runs.
