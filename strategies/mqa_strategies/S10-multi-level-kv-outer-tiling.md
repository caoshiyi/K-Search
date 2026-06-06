# Multi-level KV Outer Tiling

## Intent
Implement a two-level KV sequence tiling scheme with s2OuterBlocks and s2BaseSize. When kvSeqLen is large, split it into s2OuterBlocks outer blocks of size s2BaseSize. Each outer block contains microNLoops = CeilDiv(s2Rows, BLOCK_N) inner micro-tiles. The outer loop iterates over s2OuterBlocks, and the inner loop iterates over microNLoops micro-tiles within each outer block. This enables: (1) Q can be loaded once at the start of each outer block and reused across all inner micro-tiles; (2) tileBase accumulation across outer blocks enables correct Vec1/Vec2 slot indexing; (3) Better L1 reuse patterns because Q stays resident across all KV inner tiles within an outer block. The s2OuterBlocks and s2BaseSize are computed in host tiling based on kvSeqLen and available core count.

## Implementation checklist
- P0: Add s2OuterBlocks and s2BaseSize parameters in host tiling
- P1: Implement outer loop structure in kernel (requires: P0 verified)
- P2: Add Q residency optimization across inner tiles (requires: P1 verified)

## Expected evidence
- Expected speedup interval: min 1.3x, likely 1.5x, max 2.0x.
- Rationale: min assumes partial Q reuse; likely based on typical kvSeqLen; max is ideal full outer block reuse

## Likely files
- flash_attention_tiling.h
- flash_attention_tiling.cpp
- flash_attention_kernel.h
- flash_attention_cube.h

## Risks
- Validate correctness before benchmarking.
- Recheck buffer capacity, alignment, and pipeline lifetime constraints after the change.
