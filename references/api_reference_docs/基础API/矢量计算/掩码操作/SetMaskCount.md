# SetMaskCount

**页面ID:** atlasascendc_api_07_0094  
**来源:** https://www.hiascend.com/document/detail/zh/CANNCommunityEdition/850/API/ascendcopapi/atlasascendc_api_07_0094.html

---

#### 产品支持情况

| 产品 | 是否支持 |
| --- | --- |
| Atlas A3 训练系列产品/Atlas A3 推理系列产品 | √ |
| Atlas A2 训练系列产品/Atlas A2 推理系列产品 | √ |
| Atlas 200I/500 A2 推理产品 | √ |
| Atlas 推理系列产品AI Core | √ |
| Atlas 推理系列产品Vector Core | x |
| Atlas 训练系列产品 | x |

#### 功能说明

设置mask模式为Counter模式。该模式下，不需要开发者去感知迭代次数、处理非对齐的尾块等操作，可直接传入计算数据量，实际迭代次数由Vector计算单元自动推断。

#### 函数原型

```
__aicore__ inline void SetMaskCount()
```

#### 参数说明

无

#### 约束说明

设置为Counter模式的场景需要在矢量计算使用完之后调用SetMaskNorm将mask模式恢复为Normal模式。

#### 调用示例

请参考Counter模式调用示例。
