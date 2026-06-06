# Enlarged Tile Sizes

## Intent
Increase BLOCK_M, BLOCK_N, and BASE_K from 64 to 128. Larger tile sizes amortize loop overhead, improve L0A/L0B utilization from 25% to 50%, and reduce the number of Mmad iterations per tile. This is the most impactful single change for bandwidth-bound attention because each Mmad processes more data, reducing per-tile sync and loop overhead. However, larger tiles require more L1 buffer space for Q (128*dim*2 bytes per buffer) and KV (128*dim*2 bytes per buffer). For dim=128, Q buffer = 128*128*2 = 32KB, KV buffer = 128*128*2 = 32KB, which fits comfortably in 512KB L1. For dim=512, Q buffer = 128*512*2 = 128KB, KV buffer = 128*512*2 = 128KB, total ~384KB including P buffer, which may approach L1 limit.

## Implementation checklist
- P0: Change tile size constants (BLOCK_M, BLOCK_N, BASE_K) from 64 to 128
- P1: Adjust L1 buffer allocations for larger tile sizes (requires: P0 verified)
- P2: Reduce loop iteration count and sync overhead (requires: P0+P1 verified)

## Expected evidence
- Expected speedup interval: min 1.3x, likely 1.47x, max 1.5x.
- Rationale: min accounts for dim=512 L1 pressure fallback; likely based on dim=128 basic case实测; max is ideal L0 utilization

## Likely files
- flash_attention_tiling.h
- flash_attention_kernel.h
- flash_attention_cube.h
- flash_attention_vec.h

## Risks
- Validate correctness before benchmarking.
- Recheck buffer capacity, alignment, and pipeline lifetime constraints after the change.
