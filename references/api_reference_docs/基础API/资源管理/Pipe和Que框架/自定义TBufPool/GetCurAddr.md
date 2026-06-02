# GetCurAddr

**页面ID:** atlasascendc_api_07_0133  
**来源:** https://www.hiascend.com/document/detail/zh/CANNCommunityEdition/850/API/ascendcopapi/atlasascendc_api_07_0133.html

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

获取当前已经被自定义TBufPool分配的地址，用户可以从该地址值开始向后分配内存块。

#### 函数原型

```
__aicore__ inline uint32_t GetCurAddr()
```

#### 约束说明

无

#### 返回值说明

当前已分配的内存地址。

#### 调用示例

请参考调用示例。
