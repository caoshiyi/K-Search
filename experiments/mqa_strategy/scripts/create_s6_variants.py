#!/usr/bin/env python3
"""
创建S6策略（Softmax UB State Cache）的变体文件
用于对照实验，测试memory类策略的效果差异
"""

import json
import os
from pathlib import Path

KSEARCH_ROOT = Path(__file__).resolve().parent.parent
OUTPUT_DIR = str(KSEARCH_ROOT / "experiments" / "mqa_strategy" / "strategies" / "mqa_experiments")

# S6基线策略（从catalog提取）
S6_BASELINE = {
    "strategy_catalog": [{
        "id": "S6-V1",
        "name": "Softmax UB State Cache - Minimal",
        "category": "memory",
        "impact": "medium",
        "difficulty": 3,
        "natural_language": "Cache softmax max/sum/exp states per ring slot in UB (maxCacheBuf_, sumCacheBuf_, expCacheBuf_) instead of writing them to GM workspace (wsMeta). Each slot has a stateStride of AlignUp(subBlockRows_*BRCB_NUM*4, 32)/4 float elements. In Vec1, write state to UB cache instead of wsMetaGm. In Vec2, read from cached UB state. This eliminates GM round-trips for softmax state per KV tile.",
        "structured_params": {
            "parameters": {
                "stateStride": {"formula": "AlignUp(subBlockRows_*BRCB_NUM*4, 32)/4", "unit": "float elements"},
                "RING_SLOTS": 3,
                "default_state": {"max": -1073741824.0, "sum": 0.0}
            },
            "buffer_allocations": {
                "maxCacheBuf_": "TBuf<VECCALC> RING_SLOTS * stateStride * 4",
                "sumCacheBuf_": "TBuf<VECCALC> RING_SLOTS * stateStride * 4",
                "expCacheBuf_": "TBuf<VECCALC> RING_SLOTS * stateStride * 4"
            },
            "expected_speedup": {"min": 1.05, "max": 1.1},
            "affected_files": ["flash_attention_vec.h"]
        },
        "expected_speedup_interval": {"min": 1.05, "likely": 1.08, "max": 1.10}
    }]
}

# S6变体2：添加implementation_priorities
S6_V2_PRIORITIES = {
    "strategy_catalog": [{
        "id": "S6-V2",
        "name": "Softmax UB State Cache - Priorities",
        "category": "memory",
        "impact": "medium",
        "difficulty": 3,
        "natural_language": "Cache softmax max/sum/exp states per ring slot in UB (maxCacheBuf_, sumCacheBuf_, expCacheBuf_) instead of writing them to GM workspace (wsMeta). Each slot has a stateStride of AlignUp(subBlockRows_*BRCB_NUM*4, 32)/4 float elements. In Vec1, write state to UB cache instead of wsMetaGm. In Vec2, read from cached UB state. This eliminates GM round-trips for softmax state per KV tile.",
        "structured_params": {
            "parameters": {
                "stateStride": {"formula": "AlignUp(subBlockRows_*BRCB_NUM*4, 32)/4", "unit": "float elements"},
                "RING_SLOTS": 3,
                "default_state": {"max": -1073741824.0, "sum": 0.0}
            },
            "buffer_allocations": {
                "maxCacheBuf_": "TBuf<VECCALC> RING_SLOTS * stateStride * 4",
                "sumCacheBuf_": "TBuf<VECCALC> RING_SLOTS * stateStride * 4",
                "expCacheBuf_": "TBuf<VECCALC> RING_SLOTS * stateStride * 4"
            },
            "expected_speedup": {"min": 1.05, "max": 1.1},
            "affected_files": ["flash_attention_vec.h"]
        },
        "implementation_priorities": [
            {"priority": "P0", "action": "Add maxCacheBuf_, sumCacheBuf_, expCacheBuf_ buffer allocations in VECCALC position", "dependency": None},
            {"priority": "P1", "action": "Modify Vec1 to write state to UB cache instead of GM workspace", "dependency": "P0 verified"},
            {"priority": "P2", "action": "Modify Vec2 to read state from UB cache", "dependency": "P1 verified"}
        ],
        "expected_speedup_interval": {"min": 1.05, "likely": 1.08, "max": 1.10}
    }]
}

# S6变体3：添加anti_patterns
S6_V3_ANTI_PATTERNS = {
    "strategy_catalog": [{
        "id": "S6-V3",
        "name": "Softmax UB State Cache - Anti Patterns",
        "category": "memory",
        "impact": "medium",
        "difficulty": 3,
        "natural_language": "Cache softmax max/sum/exp states per ring slot in UB (maxCacheBuf_, sumCacheBuf_, expCacheBuf_) instead of writing them to GM workspace (wsMeta). Each slot has a stateStride of AlignUp(subBlockRows_*BRCB_NUM*4, 32)/4 float elements. In Vec1, write state to UB cache instead of wsMetaGm. In Vec2, read from cached UB state. This eliminates GM round-trips for softmax state per KV tile.",
        "structured_params": {
            "parameters": {
                "stateStride": {"formula": "AlignUp(subBlockRows_*BRCB_NUM*4, 32)/4", "unit": "float elements"},
                "RING_SLOTS": 3,
                "default_state": {"max": -1073741824.0, "sum": 0.0}
            },
            "buffer_allocations": {
                "maxCacheBuf_": "TBuf<VECCALC> RING_SLOTS * stateStride * 4",
                "sumCacheBuf_": "TBuf<VECCALC> RING_SLOTS * stateStride * 4",
                "expCacheBuf_": "TBuf<VECCALC> RING_SLOTS * stateStride * 4"
            },
            "expected_speedup": {"min": 1.05, "max": 1.1},
            "affected_files": ["flash_attention_vec.h"]
        },
        "anti_patterns": [
            {
                "id": "ap1",
                "pattern": "Allocate cache buffers in TBuf<UB> instead of TBuf<VECCALC>",
                "reason": "VECCALC position is required for softmax state tensors used in RowMuls/RowDivs operations",
                "correct_approach": "Use TBuf<VECCALC> for maxCacheBuf_, sumCacheBuf_, expCacheBuf_"
            },
            {
                "id": "ap2",
                "pattern": "Forget to initialize default state buffers with -inf/0",
                "reason": "First tile (isFirst=true) needs valid default state. Uninitialized buffers cause NaN",
                "correct_approach": "Init softmaxMaxDefaultBuf_ with -1073741824.0, softmaxSumDefaultBuf_ with 0.0"
            },
            {
                "id": "ap3",
                "pattern": "Incorrect slot offset calculation: slot*subBlockRows_ instead of slot*stateStride",
                "reason": "stateStride is aligned to 32 bytes, not equal to subBlockRows_*BRCB_NUM",
                "correct_approach": "Use slot*stateStride for UB cache indexing"
            }
        ],
        "precautions": [
            "Verify VECCALC buffer budget: 3*RING_SLOTS*stateStride*4 bytes must fit in VECCALC (~32KB)",
            "Test with isFirst=true case to ensure default state is correctly used",
            "Monitor GM traffic reduction with profiling tools",
            "Ensure slot indexing is consistent between Vec1 write and Vec2 read"
        ],
        "expected_speedup_interval": {"min": 1.05, "likely": 1.08, "max": 1.10}
    }]
}

def save_strategy_file(strategy_data, filename):
    filepath = os.path.join(OUTPUT_DIR, filename)
    with open(filepath, 'w', encoding='utf-8') as f:
        json.dump(strategy_data, f, indent=2, ensure_ascii=False)
    return filepath

def main():
    # 创建3个变体
    variants = [
        ("S6-V1.json", S6_BASELINE, "基线版本"),
        ("S6-V2.json", S6_V2_PRIORITIES, "添加implementation_priorities"),
        ("S6-V3.json", S6_V3_ANTI_PATTERNS, "添加anti_patterns")
    ]

    for filename, data, desc in variants:
        filepath = save_strategy_file(data, filename)
        # 计算策略长度
        strategy = data["strategy_catalog"][0]
        length = len(json.dumps(strategy, ensure_ascii=False))
        print(f"[OK] {filename} ({desc}): {length} chars -> {filepath}")

    print("\n[完成] S6策略变体已创建，共3个变体")
    print("V1: 基线")
    print("V2: +implementation_priorities")
    print("V3: +anti_patterns")

if __name__ == "__main__":
    main()