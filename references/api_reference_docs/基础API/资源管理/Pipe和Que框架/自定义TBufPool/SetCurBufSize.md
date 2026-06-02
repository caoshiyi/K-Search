# SetCurBufSize

**页面ID:** atlasascendc_api_07_0134  
**来源:** https://www.hiascend.com/document/detail/zh/CANNCommunityEdition/850/API/ascendcopapi/atlasascendc_api_07_0134.html

---

#### 产品支持情况

| 产品 | 是否支持 |
| --- | --- |
| Atlas A3 训练系列产品/Atlas A3 推理系列产品 | √ |
| Atlas A2 训练系列产品/Atlas A2 推理系列产品 | √ |
| Atlas 200I/500 A2 推理产品 | x |
| Atlas 推理系列产品AI Core | √ |
| Atlas 推理系列产品Vector Core | x |
| Atlas 训练系列产品 | √ |

#### 功能说明

设置当前已经被自定义TBufPool分配的内存块个数。

#### 函数原型

```
__aicore__ inline void SetCurBufSize(uint8_t curBufSize)
```

#### 约束说明

无

#### 调用示例

请参考调用示例。
