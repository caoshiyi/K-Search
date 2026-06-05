# 简单 Vector 算子调试流程

适用于单核、线性数据流的算子（Add、Mul、Softmax、LayerNorm 等）。

核心规则和日志管理见 SKILL.md，此处只列具体步骤。

## 调试步骤

### Step 0：生成 CPU Golden 数据（强制前置）

在 `current_task/artifacts/cpu_golden.py` 中用 PyTorch 计算与 dump 点位对应的中间结果：

```python
import torch

def compute_cpu_golden(input_tensor, scale):
    """打印与 DumpTensor 点位对应的 CPU golden 数据"""
    # 输入
    print(f"[CPU-400] input[:8]: {input_tensor[:8]}")

    # 计算中间
    tmp = input_tensor + 1.0
    print(f"[CPU-500] tmp[:8]: {tmp[:8]}")

    # 输出
    output = tmp * scale
    print(f"[CPU-600] output[:8]: {output[:8]}")
```

### Step 1：验证输入数据（从输入开始，禁止跳步）

在 CopyIn（DataCopy 从 GM 到 LocalTensor）后：

```cpp
LocalTensor<T> inputLocal = inQueue.DeQue<T>();
DumpTensor(inputLocal, 400, 32);
```

运行并采集日志：
```bash
scripts/evaluate_ascendc.sh current_task debug > current_task/artifacts/ascendc_dumplog/dump_01_input.log 2>&1
grep "desc=400" current_task/artifacts/ascendc_dumplog/dump_01_input.log
```

### Step 2：验证计算中间结果

每个 Compute 步骤后：

```cpp
Adds(tmpLocal, inputLocal, 1.0f, tileLength);
DumpTensor(tmpLocal, 500, 32);

Mul(outputLocal, tmpLocal, scaleLocal, tileLength);
DumpTensor(outputLocal, 510, 32);
```

### Step 3：验证输出数据

在 CopyOut（DataCopy 写回 GM）前：

```cpp
DumpTensor(outputLocal, 600, 32);
outQueue.EnQue(outputLocal);
```

## 编号约定

| 范围 | 阶段 | 说明 |
|------|------|------|
| 400-499 | V输入 | CopyIn 后，队列 DeQue 后 |
| 500-599 | V中间 | 每个 Compute 步骤后，步内递增 10 |
| 600-699 | V输出 | CopyOut 前，队列 EnQue 前 |

## 快速定位策略

1. **输入错误** → 检查 DataCopy 参数、GM offset 计算、stride/shape
2. **计算中间错误** → 先回查该步输入 dump 是否正确，再检查 API 参数、顺序、广播、类型转换、同步
3. **输出正确但 GM 写入错误** → 检查 CopyOut offset、stride、EnQue/DeQue 顺序

## 日志分析命令速查

```bash
grep "desc=" current_task/artifacts/ascendc_dumplog/dump_XX.log
grep "desc=400\|desc=500" current_task/artifacts/ascendc_dumplog/dump_XX.log
grep -A2 "desc=" current_task/artifacts/ascendc_dumplog/dump_XX.log
```

## 定位完成后

移除所有 DumpTensor，重新验证。
