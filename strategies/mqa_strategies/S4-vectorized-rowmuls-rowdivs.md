# Vectorized RowMuls/RowDivs

## Intent
Replace scalar RowMulsImpl/RowDivsImpl (which use GetValue per-row loops) with vectorized RowMuls/RowDivs helper functions.

Important: RowMuls and RowDivs are NOT AscendC built-in APIs. They are user-defined __aicore__ inline functions that wrap Mul/Div with BinaryRepeatParams to achieve row-wise broadcast scaling at vector width (64 fp16/cycle) instead of scalar width (1/cycle). You must define or import these functions before calling them. Reference implementation: references/row_ops_source_reference/vector_common_row_ops.h

The scalar loops process one element per cycle while the vector unit sits idle. The vectorized helper functions take a scale source tensor and broadcast it across each row of the destination tensor, processing 64 fp16 elements per cycle. Similarly RowDivs divides each row by the corresponding element from a source tensor.

This change requires: (1) In Vec2, prepare a scale tensor (expStateUb) from softmax state cache; (2) Call RowMuls(oPrevUb, oPrevUb, expStateUb, dealRows, dim, actualDim) where the 4th argument is the number of rows, 5th is aligned dim, and 6th is actual dim; (3) Call RowDivs(oNewUb, oNewUb, sumStateUb, dealRows, dim, actualDim) for final normalization. The state tensors must be properly sized: each row's scale factor is at offset row*BRCB_NUM in the state buffer.

## Implementation checklist
- P0: Replace scalar RowMulsImpl loops with vectorized RowMuls helper function
- P1: Replace scalar RowDivsImpl loops with vectorized RowDivs helper function (requires: P0 verified)

## Expected evidence
- Expected speedup interval: min 1.1x, likely 1.2x, max 1.3x.
- Rationale: min assumes partial vectorization; likely based on typical softmax row count; max is ideal full vector width

## Likely files
- flash_attention_vec.h

## Risks
- Validate correctness before benchmarking.
- Recheck buffer capacity, alignment, and pipeline lifetime constraints after the change.
