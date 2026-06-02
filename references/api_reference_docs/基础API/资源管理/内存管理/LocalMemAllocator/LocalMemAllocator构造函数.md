# LocalMemAllocator构造函数

**页面ID:** atlasascendc_api_07_00097  
**来源:** https://www.hiascend.com/document/detail/zh/CANNCommunityEdition/850/API/ascendcopapi/atlasascendc_api_07_00097.html

---

#### 产品支持情况

| 产品 | 是否支持 |
| --- | --- |
| Atlas A3 训练系列产品/Atlas A3 推理系列产品 | √ |
| Atlas A2 训练系列产品/Atlas A2 推理系列产品 | √ |
| Atlas 200I/500 A2 推理产品 | x |
| Atlas 推理系列产品AI Core | √ |
| Atlas 推理系列产品Vector Core | √ |
| Atlas 训练系列产品 | x |

#### 功能说明

LocalMemAllocator构造函数。

#### 函数原型

```
template <Hardware hard>
__aicore__ inline LocalMemAllocator<hard>::LocalMemAllocator()
```

#### 参数说明

**表1 **模板参数说明

| 参数名 | 描述 |
| --- | --- |
| 用于表示数据的物理位置，Hardware枚举类型，定义如下，合法位置为：UB、L1、L0A、L0B、L0C、BIAS、FIXBUF。 ``` enum class Hardware : uint8_t {  GM,     // Global Memory UB,     // Unified Buffer L1,     // L1 Buffer L0A,    // L0A Buffer L0B,    // L0B Buffer L0C,    // L0C Buffer BIAS,   // BiasTable Buffer FIXBUF, // Fixpipe Buffer MAX }; ``` |  |

#### 约束说明

同一个物理位置的LocalMemAllocator对象，在算子生命周期内只能存在1个。
