# LocalMemAllocator简介

**页面ID:** atlasascendc_api_07_00096  
**来源:** https://www.hiascend.com/document/detail/zh/CANNCommunityEdition/850/API/ascendcopapi/atlasascendc_api_07_00096.html

---

LocalMemAllocator是在使用静态Tensor编程方式时用于内存管理的类，用户无需构建TPipe/TQue，而是直接创建LocalTensor对象（也可以直接通过LocalTensor构造函数进行构造）并开发算子，从而减少运行时的开销，实现更优的性能。

LocalMemAllocator仅支持在Ascend C静态Tensor编程方式中使用，不可以与TPipe等接口混用。

#### 需要包含的头文件

```
#include "kernel_operator.h"
```

#### 原型定义

```
template<Hardware hard = Hardware::UB>
class LocalMemAllocator {
public:
    __aicore__ inline LocalMemAllocator();
    __aicore__ inline uint32_t GetCurAddr() const;
    template <class DataType, uint32_t tileSize> LocalTensor<DataType> __aicore__ inline Alloc();
    template <TPosition pos, class DataType, uint32_t tileSize> __aicore__ inline LocalTensor<DataType> Alloc();
    template <class DataType> LocalTensor<DataType> __aicore__ inline Alloc(uint32_t tileSize);    
    template <TPosition pos, class DataType> LocalTensor<DataType> __aicore__ inline Alloc(uint32_t tileSize);

    template <class TensorTraitType> LocalTensor<TensorTraitType> __aicore__ inline Alloc();

    
};
```

#### 模板参数

**表1 **模板参数说明

| 参数名 | 描述 |
| --- | --- |
| 用于表示数据的物理位置，Hardware枚举类型，定义如下，合法位置为：UB、L1、L0A、L0B、L0C、BIAS、FIXBUF。                                                                                                                           ``` enum class Hardware : uint8_t {  GM,     // Global Memory UB,     // Unified Buffer L1,     // L1 Buffer L0A,    // L0A Buffer L0B,    // L0B Buffer L0C,    // L0C Buffer BIAS,   // BiasTable Buffer FIXBUF, // Fixpipe Buffer MAX }; ``` |  |
