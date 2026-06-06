# Sub-block Row Handling

## Intent
Properly handle AIV sub-block processing. In MIX_AIC_1_2 mode, each Cube core has 2 Vector sub-blocks. Each sub-block processes its own rows: subBlockRows_ = BLOCK_M / subBlockNum_, rowStart_ = subBlockIdx_ * subBlockRows_. In Vec1, each subblock reads only its rows from the S workspace slot (offset = rowStart_ * BLOCK_N), processes softmax on curSubBlockRows_ rows, and writes P back to the corresponding offset. This ensures no cross-subblock data races and proper vector width utilization. The inputQue1/outputQue1 buffer sizes are based on subBlockRows_ instead of BLOCK_M, reducing UB pressure.

## Implementation checklist
- P0: Add subBlockRows_, rowStart_ calculation logic
- P1: Adjust Vec1 buffer sizes for subBlockRows_ (requires: P0 verified)
- P2: Adjust Vec2 buffer sizes for subBlockRows_ (requires: P1 verified)

## Expected evidence
- Expected speedup interval: min 1.02x, likely 1.05x, max 1.08x.
- Rationale: min assumes basic subblock; likely based on MIX_AIC_1_2 mode; max is ideal UB pressure reduction

## Likely files
- flash_attention_vec.h

## Risks
- Validate correctness before benchmarking.
- Recheck buffer capacity, alignment, and pipeline lifetime constraints after the change.
