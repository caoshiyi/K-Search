#!/usr/bin/env python3
"""
为剩余策略创建变体文件并批量测试
策略: S3, S5, S7, S9, S11, S12
"""

import json
import os
from pathlib import Path

KSEARCH_ROOT = Path(__file__).resolve().parent.parent
OUTPUT_DIR = str(KSEARCH_ROOT / "experiments" / "mqa_strategy" / "strategies" / "mqa_experiments")

# 简化策略定义（仅基线版本）
STRATEGY_BASELINES = {
    "S3": {
        "id": "S3-V1",
        "name": "Q L1 Cache Skip - Minimal",
        "category": "memory",
        "impact": "medium",
        "difficulty": 2,
        "natural_language": "Add a Q chunk caching mechanism in the Cube (AIC) side. Track qChunkStart_ and qChunkId_. When LoadQ is called with the same chunkId and qRowStart, skip the GM->L1 DataCopy and reuse Q data already in L1.",
        "structured_params": {
            "parameters": {
                "qChunkStart_": {"type": "uint32_t", "initial": "0xffffffff"},
                "qChunkId_": {"type": "uint32_t", "initial": "0xffffffff"}
            },
            "expected_speedup": {"min": 1.05, "max": 1.15},
            "affected_files": ["flash_attention_cube.h"]
        },
        "expected_speedup_interval": {"min": 1.05, "likely": 1.10, "max": 1.15}
    },
    "S5": {
        "id": "S5-V1",
        "name": "VEC2 Large Chunk - Minimal",
        "category": "compute",
        "impact": "medium",
        "difficulty": 1,
        "natural_language": "Increase VEC2_M_CHUNK from 8 to 64. Larger chunks amortize loop overhead and allow RowMuls/RowDivs to operate on more rows at once.",
        "structured_params": {
            "parameters": {
                "VEC2_M_CHUNK": {"old": 8, "new": 64, "type": "uint32_t"}
            },
            "expected_speedup": {"min": 1.05, "max": 1.15},
            "affected_files": ["flash_attention_vec.h"]
        },
        "expected_speedup_interval": {"min": 1.05, "likely": 1.10, "max": 1.15}
    },
    "S7": {
        "id": "S7-V1",
        "name": "Sub-block Row Handling - Minimal",
        "category": "compute",
        "impact": "low",
        "difficulty": 2,
        "natural_language": "Properly handle AIV sub-block processing. Each sub-block processes its own rows: subBlockRows_ = BLOCK_M / subBlockNum_, rowStart_ = subBlockIdx_ * subBlockRows_.",
        "structured_params": {
            "parameters": {
                "subBlockRows_": {"formula": "BLOCK_M / subBlockNum_", "value": 64},
                "rowStart_": {"formula": "subBlockIdx_ * subBlockRows_"}
            },
            "expected_speedup": {"min": 1.02, "max": 1.08},
            "affected_files": ["flash_attention_vec.h"]
        },
        "expected_speedup_interval": {"min": 1.02, "likely": 1.05, "max": 1.08}
    },
    "S9": {
        "id": "S9-V1",
        "name": "oPrev Dedicated TBuf - Minimal",
        "category": "memory",
        "impact": "low",
        "difficulty": 1,
        "natural_language": "Allocate a dedicated TBuf<VECCALC> oPrevBuf_ for Vec2 O accumulation instead of using a TQue slot.",
        "structured_params": {
            "parameters": {
                "oPrevBuf_": {"type": "TBuf<TPosition::VECCALC>", "size": "VEC2_M_CHUNK * dimAlign * sizeof(float)"}
            },
            "expected_speedup": {"min": 1.02, "max": 1.05},
            "affected_files": ["flash_attention_vec.h"]
        },
        "expected_speedup_interval": {"min": 1.02, "likely": 1.03, "max": 1.05}
    },
    "S11": {
        "id": "S11-V1",
        "name": "Single KV L1 Buffer - Minimal",
        "category": "memory",
        "impact": "medium",
        "difficulty": 2,
        "natural_language": "Use a single kvBufL1_ and vBufL1_ instead of double-buffered kBufL1_0/1 and vBufL1_0/1. WorkspaceQueue pattern eliminates need for L1 double-buffering.",
        "structured_params": {
            "parameters": {
                "kvBufL1_": {"type": "TBuf<TPosition::A1>", "size": "BLOCK_N * dim * sizeof(QType)"},
                "vBufL1_": {"type": "TBuf<TPosition::A1>", "size": "BLOCK_N * dim * sizeof(QType)"}
            },
            "expected_speedup": {"min": 1.05, "max": 1.15},
            "affected_files": ["flash_attention_cube.h"]
        },
        "expected_speedup_interval": {"min": 1.05, "likely": 1.10, "max": 1.15}
    },
    "S12": {
        "id": "S12-V1",
        "name": "KV Remain Handling - Minimal",
        "category": "compute",
        "impact": "low",
        "difficulty": 2,
        "natural_language": "Handle dim values not divisible by BASE_K with kRemain/nRemain loops for correctness.",
        "structured_params": {
            "parameters": {
                "kRemain": {"formula": "dim % BASE_K"},
                "nRemain": {"formula": "dim % BASE_K"}
            },
            "expected_speedup": {"min": 1.0, "max": 1.05},
            "affected_files": ["flash_attention_cube.h"]
        },
        "expected_speedup_interval": {"min": 1.00, "likely": 1.02, "max": 1.05}
    }
}

def create_strategy_file(strategy_id, strategy_data):
    """创建策略JSON文件"""
    filepath = os.path.join(OUTPUT_DIR, f"{strategy_data['id']}.json")
    data = {"strategy_catalog": [strategy_data]}
    with open(filepath, 'w', encoding='utf-8') as f:
        json.dump(data, f, indent=2, ensure_ascii=False)
    return filepath

def main():
    print("=" * 60)
    print("# 创建剩余策略的基线变体文件")
    print("# 策略: S3, S5, S7, S9, S11, S12")
    print("=" * 60)
    print()

    for sid, data in STRATEGY_BASELINES.items():
        filepath = create_strategy_file(sid, data)
        length = len(json.dumps(data, ensure_ascii=False))
        print(f"[OK] {data['id']}.json ({data['category']}, difficulty={data['difficulty']}): {length} chars")
        print(f"     预期加速比: {data['expected_speedup_interval']['likely']}")

    print()
    print("[完成] 6个策略的基线变体已创建")

    # 打印汇总
    print()
    print("| 策略 | 类别 | 复杂度 | 预期加速比 |")
    print("|------|------|--------|-----------|")
    for sid, data in STRATEGY_BASELINES.items():
        print(f"| {sid} | {data['category']} | {data['difficulty']} | {data['expected_speedup_interval']['likely']} |")

if __name__ == "__main__":
    main()