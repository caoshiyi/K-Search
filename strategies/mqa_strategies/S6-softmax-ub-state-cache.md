# Softmax UB State Cache

## Intent
Cache softmax max/sum/exp states per ring slot in UB (maxCacheBuf_, sumCacheBuf_, expCacheBuf_) instead of writing them to GM workspace (wsMeta). Each slot has a stateStride of AlignUp(subBlockRows_*BRCB_NUM*4, 32)/4 float elements. In Vec1, instead of writing meta to wsMetaGm, write to maxCacheUb[slot*stateStride], sumCacheUb[slot*stateStride], expCacheUb[slot*stateStride]. In Vec2, instead of reading from wsMetaGm, read from the cached UB state. This eliminates GM round-trips for softmax state per KV tile, saving ~BLOCK_M*3*4 bytes per tile of GM traffic. The default state buffers (softmaxMaxDefaultBuf_, softmaxSumDefaultBuf_) are initialized with -inf/0 and reused for isFirst=true cases.

## Implementation checklist
- P0: Add maxCacheBuf_, sumCacheBuf_, expCacheBuf_ buffer allocations
- P1: Modify Vec1 to write state to UB cache instead of GM workspace (requires: P0 verified)
- P2: Modify Vec2 to read state from UB cache (requires: P1 verified)

## Expected evidence
- Expected speedup interval: min 1.05x, likely 1.08x, max 1.1x.
- Rationale: min assumes partial GM reduction; likely based on typical state size; max is ideal zero GM round-trips

## Likely files
- flash_attention_vec.h

## Risks
- Validate correctness before benchmarking.
- Recheck buffer capacity, alignment, and pipeline lifetime constraints after the change.
