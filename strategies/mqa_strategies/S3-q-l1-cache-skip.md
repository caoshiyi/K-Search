# Q L1 Cache Skip

## Intent
Add a Q chunk caching mechanism in the Cube (AIC) side. Track qChunkStart_ (row offset) and qChunkId_ (bz*nq+qHead identifier). When LoadQ is called with the same chunkId and qRowStart, skip the GM->L1 DataCopy entirely and reuse the Q data already in L1. This eliminates redundant Q loads when the same Q tile is reused across multiple KV iterations or micro-M loops. The Q data stays resident in L1 across the entire inner KV loop, reducing GM traffic significantly. For MQA specifically, Q is loaded once per q_outer block but reused for every KV tile, so the savings is proportional to kvSeqLen/BLOCK_N.

## Implementation checklist
- P0: Add Q chunk caching mechanism with qChunkStart_ and qChunkId_ fields
- P1: Implement cache skip logic in LoadQ function (requires: P0 verified)

## Expected evidence
- Expected speedup interval: min 1.05x, likely 1.1x, max 1.15x.
- Rationale: min assumes partial KV reuse; likely based on typical MQA kvSeqLen; max is ideal full reuse

## Likely files
- flash_attention_cube.h

## Risks
- Validate correctness before benchmarking.
- Recheck buffer capacity, alignment, and pipeline lifetime constraints after the change.
