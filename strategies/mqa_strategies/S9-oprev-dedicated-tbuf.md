# oPrev Dedicated TBuf

## Intent
Allocate a dedicated TBuf<VECCALC> oPrevBuf_ for Vec2 O accumulation instead of using a TQue slot. This avoids consuming a TQue slot while oNewUb is live, keeping peak UB within limits. The buffer size is VEC2_M_CHUNK * dimAlign * sizeof(float). In Vec2, when !isFirst, DataCopy wsAccOGm to oPrevUb_ (the dedicated TBuf), then RowMuls oPrevUb_ with expState, then Add oNewUb += oPrevUb_. This eliminates the need for double-buffering the accumulation in TQue, freeing UB space for other buffers.

## Implementation checklist
- P0: Allocate dedicated TBuf<VECCALC> oPrevBuf_
- P1: Modify Vec2 to use oPrevBuf_ instead of TQue slot (requires: P0 verified)

## Expected evidence
- Expected speedup interval: min 1.02x, likely 1.03x, max 1.05x.
- Rationale: min assumes basic UB reduction; likely based on typical TQue pressure; max is ideal zero TQue contention

## Likely files
- flash_attention_vec.h

## Risks
- Validate correctness before benchmarking.
- Recheck buffer capacity, alignment, and pipeline lifetime constraints after the change.
