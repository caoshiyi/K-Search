# MQA Optimization Strategy Catalog
# Derived from Flash Attention vs MQA implementation comparison

## Strategy Index

| ID | Name | Category | Impact | Difficulty |
|----|------|----------|--------|------------|
| S1 | Enlarged Tile Sizes | Tiling | High | 2 |
| S2 | WorkspaceQueue Pattern | Pipeline | High | 3 |
| S3 | Q L1 Cache Skip | Memory | Medium | 2 |
| S4 | Vectorized RowOps | Compute | High | 2 |
| S5 | VEC2 Large Chunk | Compute | Medium | 1 |
| S6 | Softmax UB State Cache | Memory | Medium | 3 |
| S7 | Sub-block Row Handling | Compute | Low | 2 |
| S8 | Targeted SetWaitFlag | Pipeline | High | 3 |
| S9 | oPrev Dedicated TBuf | Memory | Low | 1 |
| S10 | Multi-level KV Outer Tiling | Tiling | High | 4 |
| S11 | Single KV L1 Buffer | Memory | Medium | 2 |
| S12 | KV Remain Handling | Compute | Low | 2 |

## Optimization Strategy Groupings

### Group A: Memory Bandwidth (Primary bottleneck)
- S1 (Enlarged Tiles) + S3 (Q Cache) + S11 (Single KV Buffer) + S10 (Multi-level KV Tiling)
- Expected: 2x-3x speedup on bandwidth-bound shapes

### Group B: Pipeline/Sync Efficiency
- S2 (WorkspaceQueue) + S8 (Targeted WaitFlags)
- Expected: 1.2x-1.5x speedup from pipeline overlap

### Group C: Vector Compute Efficiency
- S4 (Vectorized RowOps) + S5 (VEC2 Chunk) + S7 (Sub-block) + S9 (oPrevBuf)
- Expected: 1.1x-1.3x speedup from vector unit utilization

### Group D: Precision & Portability
- S12 (KV Remain) + runtime core count
- Expected: correctness fix, minor perf gain