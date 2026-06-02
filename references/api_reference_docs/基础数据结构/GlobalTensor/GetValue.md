# GetValue

**页面ID:** atlasascendc_api_07_00026  
**来源:** https://www.hiascend.com/document/detail/zh/CANNCommunityEdition/850/API/ascendcopapi/atlasascendc_api_07_00026.html

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

获取GlobalTensor的相应偏移位置的值。

#### 函数原型

```
__aicore__ inline __inout_pipe__(S) PrimType GetValue(const uint64_t offset) const
```

#### 参数说明

**表1 **参数说明

| 参数名 | 输入/输出 | 描述 |
| --- | --- | --- |
| offset | 输入 | 偏移offset个元素。 |

#### 返回值说明

返回PrimType类型的立即数。

#### 约束说明

如果GetValue的Global Memory地址内容存在被外部改写的可能，需要先调用DataCacheCleanAndInvalid，确保Data Cache与Global Memory的Cache一致性，之后再调用此接口。
