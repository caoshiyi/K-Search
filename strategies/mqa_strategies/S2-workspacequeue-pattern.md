# WorkspaceQueue Pattern

## Intent
Replace raw CrossCoreSetFlag/WaitFlag with a typed WorkspaceQueue class that encapsulates workspace ring-buffer management. WorkspaceQueue provides ProducerAcquire() to get the next slot, ProducerReleaseFix() or ProducerReleaseMte3() to signal completion with the correct pipeline mode, and ConsumerAcquire() which waits with the correct PIPE_MTE2 mode. This ensures that AIC->AIV sync uses the minimal pipeline scope: Fixpipe completion signals S_READY (PIPE_FIX), DataCopy completion signals P_READY (PIPE_MTE3), and AIV consumption uses PIPE_MTE2. This eliminates the mismatch between producer pipeline (FIX for S, MTE3 for P and O) and consumer pipeline (MTE2 for all). The ring buffer slot calculation is encapsulated: slot = (head/tail % RING_SLOTS) * slotSize, eliminating manual offset tracking.

## Implementation checklist
- P0: Add WorkspaceQueue class encapsulating workspace ring-buffer management
- P1: Integrate WorkspaceQueue in Cube side (AIC) (requires: P0 verified)
- P2: Integrate WorkspaceQueue in Vector side (AIV) (requires: P1 verified)

## Expected evidence
- Expected speedup interval: min 1.2x, likely 1.35x, max 1.5x.
- Rationale: min assumes basic sync fix; likely based on reference implementation; max is ideal pipeline overlap

## Likely files
- workspace_queue.h
- flash_attention_cube.h
- flash_attention_vec.h
- flash_attention_kernel.h

## Risks
- Validate correctness before benchmarking.
- Recheck buffer capacity, alignment, and pipeline lifetime constraints after the change.
