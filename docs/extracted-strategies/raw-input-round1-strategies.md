# raw_input.tar Strategy Extraction

Source: repository-local `raw_input.tar`

Extraction policy: only artifacts inside the raw round archive are treated as evidence. Existing `strategies/mqa_strategies` files are intentionally ignored because they are not measured evidence for this extraction.

## Summary

The archive supports one high-confidence validated optimization strategy: bidirectional Q/KV two-level tiling with L1 reuse for FlashAttention-like kernels. It also supports one negative boundary: Q-only or incomplete two-level reuse should not be generalized as useful, and historical negative results must carry their variable scope. Two additional design-contract lessons should be injected into future K-Search AscendC Claude Agent SDK design prompts because they directly affect whether a blind implementation agent can safely implement the strategy.

## ATTN-L1-001

- `id`: `ATTN-L1-001`
- `title`: Bidirectional Q/KV two-level tiling with L1 reuse
- `classification`: `validated_strategy`
- `operator_scope`: FlashAttention-like attention kernels with online softmax, GQA/MQA-style repeated KV reuse, fp16, seq/dim regimes similar to `qSeqLen=kvSeqLen=1024`, `dim=128`
- `evidence_strength`: high

### Intent

Increase task granularity in both Q and KV dimensions so that K/V tiles loaded from GM into L1 are reused by multiple Q micro-blocks in the same task, instead of being reloaded for every 128-row Q block. This is a measured optimization, not only a design prediction.

### Applicability Boundary

Measured on FlashAttention shape `4,32,4,1024,1024,128,float16`, from baseline commit `66a35f8`, on a 910B-like target with about 512KB L1 and 192KB UB assumptions in the design document. The measured implementation used `mBaseSize=512`, `s2BaseSize=512`, `BLOCK_M=BLOCK_N=BASE_K=128`, `qOuterBlocks=2`, and `s2OuterBlocks=2`.

Use when the baseline loops over Q blocks outside KV blocks and reloads the same K/V data for multiple Q blocks. Do not blindly apply when Q/KV reuse distance is already small, when L1/UB capacity cannot hold the enlarged tiles, or when the vector-side online softmax/workspace layout cannot be enlarged consistently.

### Mechanism

Use outer big blocks plus inner 128 micro-blocks:

```text
task = (batch, kvHead, group, qOuterIdx)
qRows = min(mBaseSize, qSeqLen - qOuterIdx * mBaseSize)
for each KV outer block:
  s2Rows = min(s2BaseSize, kvSeqLen - s2Start)
  compute S = Q_big @ K_outer^T by reusing each K micro-block across all Q micro-blocks
  compute O += P @ V_outer by accumulating along the KV reduce dimension
  run vector online-softmax and online O accumulation over the enlarged workspace slot
```

The key transfer is not a single file edit. It is the loop ownership change: a task owns several Q micro-blocks and a KV outer block, so a K/V micro-block can stay in L1 while multiple Q micro-blocks consume it. In the measured implementation, using `mBaseSize=512` merged four 128-row Q micro-blocks per task, reducing per-head K/V-128 load count from 8 to 2 for the measured shape.

### Implementation Checklist

1. Add tiling fields for `mBaseSize`, `s2BaseSize`, `qOuterBlocks`, and `s2OuterBlocks`.
2. Select `mBaseSize` and `s2BaseSize` as multiples of `BLOCK_M/BLOCK_N`; for the measured long-Q case use 512/512.
3. Change task mapping from `groupNum * seqBlocks` to `groupNum * qOuterBlocks`.
4. Load the whole Q big block once per task into L1 and address 128-row Q micro-blocks inside it.
5. In MM1, iterate K/KV micro-blocks outside Q micro-blocks so each K load is reused across the task's Q micro-blocks.
6. In MM2, accumulate `P @ V` along the KV reduce dimension in L0C; set `cmatrixInitVal` only for the first reduce micro-block.
7. Enlarge S/P/O/meta/accO workspace slots from `BLOCK_M/BLOCK_N` grain to `mBaseSize/s2BaseSize` grain.
8. Update vector softmax to process `s2BaseSize` columns in row chunks and maintain online state per enlarged Q row range.
9. Run correctness first, then benchmark against the original baseline and main-agent double-check the result.

### Design Contract Requirements

A design document for a blind implementation agent must specify:

- Tile choices and constraints: `mBaseSize % BLOCK_M == 0`, `s2BaseSize % BLOCK_N == 0`, `s2BaseSize <= 1024` for the measured UB/softmax alignment assumption.
- L1 capacity table for Q, K, V, and P blocks; the measured single-buffer peak was under 512KB.
- Workspace slot formulas: `wsS/wsP = RING_SLOTS * mBaseSize * s2BaseSize`, `wsO/wsAccO = RING_SLOTS * mBaseSize * dimAlign`, `wsMeta = RING_SLOTS * mBaseSize * 3`.
- Cube/vector queue slot-size consistency across AIC and AIV.
- Tail handling for Q rows, KV rows, and dim tails using `min` and `AlignUp`.
- Vector row chunk formula derived from `VEC_QUEUE_BUF_SIZE / (s2BaseSize * sizeof(float))`.
- Correctness commands and expected baseline-level numerical tolerance.

### Evidence

- Design intent: `raw_strategy/rounds/round1/design_doc.md` defines Q and KV outer blocks and says the optimization is QKV two-level tiling plus L1 reuse.
- Implementation decision: `raw_strategy/rounds/round1/blind_impl_round1.md` records `mBaseSize=512`, `s2BaseSize=512`, `qOuterBlocks=2`, `s2OuterBlocks=2`, Q big-block L1 residency, and K/V micro-block reuse.
- Diff proof: `raw_strategy/rounds/round1/kernel_changes.diff` shows enlarged queue slots, `LoadQ(... qRowStart, qRows)`, `ComputeMM1(... s2Start, s2Rows, qRows)`, `ComputeMM2(... s2Start, s2Rows, qRows)`, new tiling fields, and vector-side enlarged state/workspace.
- Correctness: blind implementation passed basic correctness with REL `9.77e-04/1.84e-05`, ABS `4.88e-04/9.19e-06`, mismatch `0.00%`; main-agent double check also passed.
- Performance: subagent measured base `2391.5us`, AscendC `1572.1us`, speedup `1.52x`; main-agent double check measured AscendC `1590.964us`, base `2377us`, speedup `1.49x`.
- Conflict resolution: `raw_strategy/lessons_round1.md` says the older "two-level alone has no gain" conclusion was invalid for this design because the older experiment was Q-only/incomplete, while this round was true Q+KV two-level reuse.

### Risks And Failure Modes

- UB overflow if `s2BaseSize` is enlarged without row chunking.
- L1 overwrite races if single-buffer K/V/P buffers are reloaded before previous L0 reads finish.
- Incorrect Q/K/P/V NZ strides causing silent wrong reads.
- AIC/AIV workspace slot mismatch after enlarging slots.
- Misleading performance attribution if Q-only historical results are applied to bidirectional Q/KV reuse.

### Follow-Up Experiments

- Sweep `mBaseSize` and `s2BaseSize` across seq lengths and dims.
- Validate dim > 128 and dim tails.
- Add double-buffer/TQue pipeline on top of the measured two-level structure and compare against this round's 1590us baseline.
- Test short-Q decode regimes where the design's selection function changes `s2BaseSize` and `mBaseSize`.

## ATTN-BND-001

- `id`: `ATTN-BND-001`
- `title`: Do not generalize Q-only two-level L1 reuse as a validated optimization
- `classification`: `negative_or_bounded_strategy`
- `operator_scope`: FlashAttention-like attention kernels
- `evidence_strength`: high for the negative boundary, low as an optimization

### Intent

Preserve a negative boundary: incomplete two-level reuse, especially Q-direction expansion while KV/V reuse remains incomplete or pipeline remains serial, should not be promoted as a generally useful strategy. It is evidence about what not to conclude.

### Applicability Boundary

This boundary applies to the older verification artifact where `mBaseSize=512`, Q direction was enlarged, KV still stepped by 128, and no double buffer/TQue pipeline was introduced. It does not apply to the measured round1 bidirectional Q+KV implementation.

### Mechanism

The older experiment enlarged task/workspace shape but did not achieve the same bidirectional K/V reuse. It measured correctness but no speedup. The important strategy is epistemic: when a result says "two-level tiling has no gain," first inspect whether it was Q-only, KV-only, bidirectional, pipelined, or serial.

### Implementation Checklist

1. Before injecting a historical strategy or negative conclusion, record its changed variables.
2. Distinguish Q-only, KV-only, Q+KV bidirectional, and Q/KV plus pipeline variants.
3. Compare performance only against a baseline with matching variables.
4. Do not use a Q-only negative result to reject Q+KV bidirectional reuse.

### Design Contract Requirements

Future design docs must include a "historical evidence boundary" section that states:

- What tensors were actually reused.
- Whether V reuse was included.
- Whether pipeline/double buffer was included.
- Which baseline latency was used.
- Whether the result is positive, neutral, or negative under that exact variable set.

### Evidence

- `raw_strategy/two_level_tiling_verify.md` measured the older experiment at `2406us` versus baseline `2391us`, about `1.00x`, despite correctness PASS.
- The same file concluded that Q-only two-level reuse without pipeline did not improve end-to-end performance.
- `raw_strategy/lessons_round1.md` explicitly says this old conclusion was misapplied because it was "only Q direction" while round1 was true Q+KV bidirectional reuse.

### Risks And Failure Modes

- Rejecting a valid future strategy because a different, narrower variant failed.
- Claiming "L1 reuse alone has no gain" without specifying whether K and V were both actually saved.
- Comparing a pipelined main implementation to a serial partial implementation and drawing the wrong causal conclusion.

### Follow-Up Experiments

- Run a controlled ablation matrix: Q-only, KV-only, Q+KV, each with and without pipeline.
- Keep all else fixed and record correctness plus latency for each variant.

## ATTN-CONTRACT-001

- `id`: `ATTN-CONTRACT-001`
- `title`: Blind implementation design docs must specify synchronization, stride, and default API choices
- `classification`: `design_contract_lesson`
- `operator_scope`: AscendC attention kernels implemented by a blind coding agent from design docs
- `evidence_strength`: high

### Intent

Improve the success rate of blind implementation agents by turning implicit kernel details into explicit contracts. This is not a direct speedup strategy; it is a strategy for making optimization designs implementable without access to the high-performance source.

### Applicability Boundary

Use whenever a design subagent writes an AscendC operator optimization plan for a separate implementation subagent that cannot inspect the high-performance implementation. It is especially important for L1/UB reuse, online softmax state, WorkspaceQueue slots, and event-flag synchronization.

### Mechanism

The blind implementer succeeded but had to fill five gaps: Q L1 layout, MM1 K loading choice, K/V micro-block split values, single-buffer L1 synchronization, and vector state stride. These are not cosmetic details. They define correctness, buffer lifetime, and whether the implementation stays within known-good API behavior.

### Implementation Checklist

1. For every resident buffer, specify row capacity, `dstNzC0Stride`, logical rows, and micro-block addressing formula.
2. For every reusable single-buffer L1 allocation, specify the event dependency that protects reload-overwrite.
3. If multiple API approaches exist, mark one default path and explain why alternatives are fallback-only.
4. For vector online state, specify per-slot and per-subblock stride formulas using global row indices.
5. For each changed workspace slot, state the old formula and new formula.
6. Require the implementation agent to report any self-filled design gaps as part of the implementation log.

### Design Contract Requirements

The next design document should explicitly include:

- Q L1 single-segment formula: `dstNzC0Stride = qRowsAlign`, micro-block `startIndex = mStart / C0 + i`, `srcStride = qC0Stride / C0`.
- MM1 K loading default: reuse the baseline `LoadData2D` path when it avoids an unverified transpose helper.
- K/V split default: use `BLOCK_N` for `nL1Size` and `KL1_SPLIT` when baseline-equivalent loading is safer; explain that task merging via `mBaseSize` is the measured reuse source.
- Single-buffer synchronization: wait for prior L1 reads before GM reload overwrites the same L1 buffer, e.g. the implemented `MTE1_MTE2` dependency.
- Vector state stride: `stateStride_ = AlignUp(mBaseSize, 32)` and state/meta addresses based on task-global row index after subblock splitting.

### Evidence

- `raw_strategy/rounds/round1/blind_impl_round1.md` lists the implementer's self-filled decisions and reports correctness/performance success.
- `raw_strategy/lessons_round1.md` says the design granularity was still insufficient and identifies synchronization contracts, stride/addressing, and multi-option defaults as the highest-risk missing areas.
- `raw_strategy/rounds/round1/kernel_changes.diff` shows concrete implementation of `qC0Stride`, `MTE1_MTE2`, `stateStride_`, enlarged slot sizes, and task context.

### Risks And Failure Modes

- Correct design intent fails because the implementation agent chooses a risky API variant.
- Silent numerical corruption from wrong state stride after AIV subblock splitting.
- Deadlock or data race from under-specified event flags.
- Buffer overwrite while L0 still reads the previous L1 tile.

### Follow-Up Experiments

- In the next blind round, require the design document to include a "contract checklist" and measure whether the implementation agent still needs self-filled assumptions.
- Add a review pass that flags any optimization design lacking explicit stride and synchronization contracts.

## ATTN-CONTRACT-002

- `id`: `ATTN-CONTRACT-002`
- `title`: Treat two-level tiling as a pipeline substrate only after preserving the measured serial structure
- `classification`: `design_contract_lesson`
- `operator_scope`: FlashAttention-like kernels evolving from serial tiling to double-buffer/TQue pipeline
- `evidence_strength`: medium

### Intent

When moving from measured serial Q+KV two-level reuse to pipeline/double buffer optimization, preserve the verified two-level structure and explicitly state which buffers become multi-buffered. Use the round1 1590us result as the next baseline, not the original 2391us baseline.

### Applicability Boundary

This applies to follow-up rounds that add double buffer, TQue, or overlap between MTE2/MTE1/M/FIX/Vec stages. It is not itself a validated blind implementation strategy in this archive because the measured 4x pipeline result came from aligning to main/high-performance code, not from an isolated blind implementation based on a design doc.

### Mechanism

Round1 proved the two-level structure is beneficial by itself. The older verification also shows a main-like pipelined implementation can reach about 606us/3.92x, but that evidence is contaminated for this extraction because it depends on the high-performance implementation path. Therefore, future SDK planning should use two-level reuse as the substrate and run a new blind pipeline round.

### Implementation Checklist

1. Set the follow-up baseline to the measured Q+KV two-level implementation: about 1590us.
2. Keep task mapping, workspace slot grain, vector state layout, and tail handling unchanged unless the design names the change.
3. Add buffer depth tables for each L1/L0/UB/GM workspace queue.
4. Specify ping-pong indices and SetFlag/WaitFlag order across MTE2, MTE1, M, FIX, and Vec.
5. Measure incremental speedup over 1590us, not over 2391us.

### Design Contract Requirements

The design doc must state:

- Which buffers remain single-buffered.
- Which buffers become double-buffered.
- How buffer depth affects L1/UB capacity.
- CrossCore and same-core event ordering.
- Whether online-softmax state and wsAccO use current slot, previous slot, or ping-pong slot.

### Evidence

- `raw_strategy/lessons_round1.md` recommends keeping the verified two-level structure and adding pipeline on top.
- It also warns that double buffer requires renewed L1/UB capacity accounting and explicit synchronization contracts.
- `raw_strategy/two_level_tiling_verify.md` reports a main-like KV two-level plus pipeline result around `606us`/`3.92x`, but the artifact says this was an entire main kernel checkout, so it is not promoted to `validated_strategy`.

### Risks And Failure Modes

- Regressing by rewriting the verified two-level structure instead of layering pipeline onto it.
- Overstating pipeline contribution by comparing to the old 2391us baseline.
- L1/UB overflow after doubling buffer depth.
- Cross-stage deadlock from under-specified event sequencing.

### Follow-Up Experiments

- Run a blind implementation round whose only new variable is pipeline/double buffer on top of ATTN-L1-001.
- Compare against both 1590us and 2391us, but report incremental contribution versus 1590us as the primary result.

## Not Promoted To Validated Strategy

### Full main-style TQue/double-buffer pipeline

The archive contains evidence that a main-like pipeline implementation can reach about `606us`/`3.92x`, but that path was described as checking out/aliging with the high-performance main kernel. Under the blind-implementation extraction standard, this is a high-priority follow-up hypothesis, not a validated extracted strategy. It should become a strategy only after a design subagent writes an implementable contract and a blind implementation subagent reproduces correctness and speedup without seeing the high-performance source.
