# GetSystemCycle(ISASI)

**页面ID:** atlasascendc_api_07_0282  
**来源:** https://www.hiascend.com/document/detail/zh/CANNCommunityEdition/850/API/ascendcopapi/atlasascendc_api_07_0282.html

---

#### 产品支持情况

| 产品 | 是否支持 |
| --- | --- |
| Atlas A3 训练系列产品/Atlas A3 推理系列产品 | √ |
| Atlas A2 训练系列产品/Atlas A2 推理系列产品 | √ |
| Atlas 200I/500 A2 推理产品 | x |
| Atlas 推理系列产品AI Core | x |
| Atlas 推理系列产品Vector Core | x |
| Atlas 训练系列产品 | x |

#### 功能说明

获取当前系统cycle数，若换算成时间需要按照50MHz的频率，时间单位为us，换算公式为：time = (cycle数/50) us 。

#### 函数原型

```
__aicore__ inline int64_t GetSystemCycle()
```

#### 参数说明

无

#### 返回值说明

返回系统cycle数。

#### 约束说明

该接口是PIPE_S流水，若需要测试其他流水的指令时间，需要在调用该接口前通过PipeBarrier插入对应流水的同步，具体请参考调用示例。

#### 调用示例

- 如下示例通过GetSystemCycle获取系统cycle数，并换算成时间（单位：us）。

```
#include "kernel_operator.h"

__aicore__ inline void InitTilingParam(int32_t& totalSize, int32_t& loopSize)
{
    int64_t systemCycleBefore = AscendC::GetSystemCycle(); // 调用GetBlockNum指令前的cycle数
    loopSize = totalSize / AscendC::GetBlockNum();
    int64_t systemCycleAfter = AscendC::GetSystemCycle(); // 调用GetBlockNum指令后的cycle数
    int64_t GetBlockNumCycle = systemCycleAfter - systemCycleBefore; // 执行GetBlockNum指令所用的cycle数
    int64_t CycleToTimeBase = 50; // cycle数转换成时间的基准单位，固定为50
    int64_t GetBlockNumTime = GetBlockNumCycle/CycleToTimeBase; // 执行GetBlockNum指令所用时间，单位为us
};
```

- 如下示例为获取矢量计算Add指令时间的关键代码片段，在调用GetSystemCycle之前，插入了PIPE_ALL同步，可以保证相关指令执行完后再获取cycle数。

```
PipeBarrier<PIPE_ALL>();
int64_t systemCycleBefore = AscendC::GetSystemCycle(); // 调用Add指令前的cycle数
AscendC::Add(dstLocal, src0Local, src1Local, 512);
PipeBarrier<PIPE_ALL>();
int64_t systemCycleAfter = AscendC::GetSystemCycle(); // 调用Add指令后的cycle数
int64_t GetBlockNumCycle = systemCycleAfter - systemCycleBefore; // 执行Add指令所用的cycle数
int64_t CycleToTimeBase = 50; // cycle数转换成时间的基准单位，固定为50
int64_t GetBlockNumTime = GetBlockNumCycle/CycleToTimeBase; // 执行Add指令所用时间，单位为us
```
