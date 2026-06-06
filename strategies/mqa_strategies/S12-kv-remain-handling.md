# KV Remain Handling

## Intent
Handle dim values that are not divisible by BASE_K with kRemain/nRemain loops. In MM1, after the main ki loop (dim/BASE_K iterations), check kRemain = dim % BASE_K. If kRemain > 0, do one extra Mmad iteration with kAligned = AlignUp(kRemain, C0) as the K dimension, setting cmatrix_init_val based on whether kTiles was 0. In MM2, similarly handle nRemain = dim % BASE_K with an extra Mmad iteration for the remaining output columns, using FixpipeNzL0cToNdGm for the non-standard N dimension. This ensures correctness for all dim values (not just multiples of BASE_K) and avoids silent precision loss from ignoring the remainder.

## Implementation checklist
- P0: Add kRemain loop in MM1 for dim % BASE_K handling
- P1: Add nRemain loop in MM2 for dim % BASE_K handling (requires: P0 verified)

## Expected evidence
- Expected speedup interval: min 1.0x, likely 1.02x, max 1.05x.
- Rationale: min assumes no speedup (correctness fix); likely based on typical dim values; max is ideal edge case optimization

## Likely files
- flash_attention_cube.h

## Risks
- Validate correctness before benchmarking.
- Recheck buffer capacity, alignment, and pipeline lifetime constraints after the change.
