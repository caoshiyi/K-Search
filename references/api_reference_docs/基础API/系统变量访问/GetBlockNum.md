# GetBlockNum

**页面ID:** atlasascendc_api_07_0184  
**来源:** https://www.hiascend.com/document/detail/zh/CANNCommunityEdition/850/API/ascendcopapi/atlasascendc_api_07_0184.html

---

#### 产品支持情况

| 产品 | 是否支持 |
| --- | --- |
| Atlas A3 训练系列产品/Atlas A3 推理系列产品 | √ |
| Atlas A2 训练系列产品/Atlas A2 推理系列产品 | √ |
| Atlas 200I/500 A2 推理产品 | √ |
| Atlas 推理系列产品AI Core | √ |
| Atlas 推理系列产品Vector Core | x |
| Atlas 训练系列产品 | √ |

#### 功能说明

获取当前任务配置的核数，用于代码内部的多核逻辑控制等。

#### 函数原型

```
__aicore__ inline int64_t GetBlockNum()
```

#### 参数说明

无

#### 返回值说明

当前任务配置的核数。

#### 约束说明

无。

#### 调用示例

```
#include "kernel_operator.h"
// 在核内做简单的tiling计算时使用block_num，复杂tiling建议在host侧完成
__aicore__ inline void InitTilingParam(int32_t& totalSize, int32_t& loopSize)
{
    loopSize = totalSize / AscendC::GetBlockNum();
};
```
