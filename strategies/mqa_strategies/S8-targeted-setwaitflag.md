# Targeted SetWaitFlag

## Intent
Replace all PipeBarrier<PIPE_ALL> with targeted SetWaitFlag<HardEvent::X_Y> calls. PipeBarrier<PIPE_ALL> drains all pipeline channels (V, M, MTE1, MTE2, MTE3, FIX), destroying overlap. Targeted barriers only wait for the specific dependency needed. In Cube: use SetWaitFlag<HardEvent::M_MTE1> before loading data from L1 to L0 (depends on MTE1 completing); SetWaitFlag<HardEvent::MTE1_M> before Mmad (depends on MTE1 completing); SetWaitFlag<HardEvent::M_FIX> before Fixpipe (depends on Mmad completing); SetWaitFlag<HardEvent::FIX_MTE2> after Fixpipe (allows L0C reuse). In Vector: use SetWaitFlag<HardEvent::MTE3_MTE2> before DataCopy from GM to UB; SetWaitFlag<HardEvent::MTE2_V> before vector compute; SetWaitFlag<HardEvent::V_MTE3> before DataCopy from UB to GM; SetWaitFlag<HardEvent::MTE3_V> after GM write completion. Each targeted barrier preserves pipeline overlap for unrelated channels.

## Implementation checklist
- P0: Replace PipeBarrier<PIPE_ALL> in Cube with targeted SetWaitFlag
- P1: Replace PipeBarrier<PIPE_ALL> in Vector with targeted SetWaitFlag (requires: P0 verified)

## Expected evidence
- Expected speedup interval: min 1.2x, likely 1.35x, max 1.5x.
- Rationale: min assumes partial overlap; likely based on reference implementation; max is ideal full pipeline overlap

## Likely files
- flash_attention_cube.h
- flash_attention_vec.h

## Risks
- Validate correctness before benchmarking.
- Recheck buffer capacity, alignment, and pipeline lifetime constraints after the change.
