# SetValue

**页面ID:** atlasascendc_api_07_00028  
**来源:** https://www.hiascend.com/document/detail/zh/CANNCommunityEdition/850/API/ascendcopapi/atlasascendc_api_07_00028.html

---

#### 产品支持情况

| 产品 | 是否支持 |
| --- | --- |
| Atlas A3 训练系列产品/Atlas A3 推理系列产品 | √ |
| Atlas A2 训练系列产品/Atlas A2 推理系列产品 | √ |
| Atlas 200I/500 A2 推理产品 | √ |
| Atlas 推理系列产品AI Core | √ |
| Atlas 推理系列产品Vector Core | √ |
| Atlas 训练系列产品 | √ |

#### 功能说明

设置GlobalTensor相应偏移位置的值。

- 因为硬件实现不同，其与通用CPU标量赋值操作存在差异。SetValue赋值操作首先改写的是每个AI Core内部的DCache， 不会立刻写出到Global Memory，且后续写出时以Cache Line（64B）为单位。使用该接口之前必须了解DCache结构和Cache一致性原理（参见DataCacheCleanAndInvalid），否则可能存在误用的情况，**请谨慎使用。**
- 调用SetValue后，首先改写的是每个AI Core内部的DCache，如果需要立即写出到Global Memory，需要在调用此接口后，再调用DataCacheCleanAndInvalid，确保DCache与Global Memory的Cache一致性。
- 多核操作GM地址时，要求不同核操作的地址（通过offset参数设置元素偏移，可以转换为地址）至少有Cache Line大小的偏移，否则会出现多核数据随机覆盖。同时需要考虑地址对齐（64B）的问题。详细内容请参考调用示例。

#### 函数原型

```
__aicore__ inline void SetValue(const uint64_t offset, PrimType value)
```

#### 参数说明

**表1 **参数说明

| 参数名 | 输入/输出 | 描述 |
| --- | --- | --- |
| offset | 输入 | 偏移offset个元素。 |
| 设置值。PrimType类型。 PrimType定义如下： ``` // PrimT用于从T中提取基础数据类型：T传入基础数据类型，直接返回数据类型；T传入为TensorTrait类型时萃取TensorTrait中的LiteType基础数据类型 using PrimType = PrimT<T>; ``` |  |  |

#### 返回值说明

无。

#### 约束说明

无。
