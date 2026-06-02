# GetBlockIdx

**页面ID:** atlasascendc_api_07_0185  
**来源:** https://www.hiascend.com/document/detail/zh/CANNCommunityEdition/850/API/ascendcopapi/atlasascendc_api_07_0185.html

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

获取当前核的index，用于代码内部的多核逻辑控制及多核偏移量计算等。

#### 函数原型

```
__aicore__ inline int64_t GetBlockIdx()
```

#### 参数说明

无

#### 返回值说明

GetBlockIdx返回当前核的索引，index的范围为[0, 用户配置的BlockDim数量)。

同时启动AIC和AIV的场景：

当AIC和AIV比例为1:2时，AIC上取值范围为[0, 用户配置的BlockDim数量), AIV上取值范围为[0, 2 * 用户配置的BlockDim数量)；

当AIC和AIV比例为1:1时，AIC上取值范围为[0, 用户配置的BlockDim数量)，AIV上取值范围为[0, 用户配置的BlockDim数量)。

#### 约束说明

GetBlockIdx为一个系统内置函数，返回当前核的index。

#### 调用示例

```
#include "kernel_operator.h"
constexpr int32_t SINGLE_CORE_OFFSET = 256;
class KernelGetBlockIdx {
public:
    __aicore__ inline KernelGetBlockIdx () {}
    __aicore__ inline void Init(__gm__ uint8_t* src0Gm, __gm__ uint8_t* src1Gm, __gm__ uint8_t* dstGm)
    {
        // 根据index对每个核进行地址偏移
        src0Global.SetGlobalBuffer((__gm__ float*)src0Gm + AscendC::GetBlockIdx() * SINGLE_CORE_OFFSET);
        src1Global.SetGlobalBuffer((__gm__ float*)src1Gm + AscendC::GetBlockIdx() * SINGLE_CORE_OFFSET);
        dstGlobal.SetGlobalBuffer((__gm__ float*)dstGm + AscendC::GetBlockIdx() * SINGLE_CORE_OFFSET);
        pipe.InitBuffer(inQueueSrc0, 1, 256 * sizeof(float));
        pipe.InitBuffer(inQueueSrc1, 1, 256 * sizeof(float));
        pipe.InitBuffer(selMask, 1, 256);
        pipe.InitBuffer(outQueueDst, 1, 256 * sizeof(float));
    }
    ......
};
```
