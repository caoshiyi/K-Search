# Single KV L1 Buffer

## Intent
Use a single kvBufL1_ (TBuf<A1>) for K and a single vBufL1_ (TBuf<A1>) for V, instead of double-buffered kBufL1_0/1 and vBufL1_0/1. The WorkspaceQueue pattern with targeted sync (PIPE_FIX for S, PIPE_MTE3 for P) ensures that AIC doesn't need to overlap K/V prefetch with computation — the AIC->AIV pipeline sync is handled by the queue. Since each KV tile is loaded, consumed in MM1/MM2, and then the workspace slot is released, there's no need for double-buffering at L1 level. This saves 2*BLOCK_N*dim*sizeof(QType) bytes of L1 space (one slot per buffer type), which can be used for larger Q chunks or P buffers.

## Implementation checklist
- P0: Replace kBufL1_0/1 with single kvBufL1_
- P1: Replace vBufL1_0/1 with single vBufL1_ (requires: P0 verified)

## Expected evidence
- Expected speedup interval: min 1.05x, likely 1.1x, max 1.15x.
- Rationale: min assumes partial L1 savings; likely based on WorkspaceQueue sync; max is ideal full L1 release

## Likely files
- flash_attention_cube.h

## Risks
- Validate correctness before benchmarking.
- Recheck buffer capacity, alignment, and pipeline lifetime constraints after the change.
