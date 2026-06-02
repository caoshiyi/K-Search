# GetSize

**页面ID:** atlasascendc_api_07_00029  
**来源:** https://www.hiascend.com/document/detail/zh/CANNCommunityEdition/850/API/ascendcopapi/atlasascendc_api_07_00029.html

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

获取GlobalTensor的元素个数。

#### 函数原型

```
__aicore__ inline uint64_t GetSize() const
```

#### 参数说明

无。

#### 返回值说明

GlobalTensor的元素个数。

#### 约束说明

使用仅传入全局数据指针的SetGlobalBuffer接口对GlobalTensor进行初始化，通过本接口获取到的元素个数为0。
