# VEC2 Large Chunk

## Intent
Increase VEC2_M_CHUNK from 8 to 64. Vec2 processes O accumulation in row chunks; larger chunks amortize loop overhead and allow RowMuls/RowDivs to operate on more rows at once, improving vector unit utilization. VEC2_M_CHUNK=64 means processing BLOCK_M rows in a single chunk when BLOCK_M=128, or in two chunks when BLOCK_M=128 with subBlockRows_=64. UB budget check: oPrevBuf needs VEC2_M_CHUNK * dimAlign * sizeof(float) = 64*128*4 = 32KB for dim=128, which fits in UB alongside other buffers (total UB ~196KB).

## Implementation checklist
- P0: Increase VEC2_M_CHUNK constant from 8 to 64

## Expected evidence
- Expected speedup interval: min 1.05x, likely 1.1x, max 1.15x.
- Rationale: min assumes partial row processing; likely based on typical UB budget; max is ideal full chunk

## Likely files
- flash_attention_vec.h

## Risks
- Validate correctness before benchmarking.
- Recheck buffer capacity, alignment, and pipeline lifetime constraints after the change.
