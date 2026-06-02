# SetCurAddr

**页面ID:** atlasascendc_api_07_0132  
**来源:** https://www.hiascend.com/document/detail/zh/CANNCommunityEdition/850/API/ascendcopapi/atlasascendc_api_07_0132.html

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

设置自定义TBufPool已经被分配完的地址，比如初始化时一共申请了32k的内存大小，给某一个TQue分配了8K，则需要调用该接口以保证后续的内存块从8K开始分配。

#### 函数原型

```
__aicore__ inline void SetCurAddr(uint32_t curAddr)
```

**表1 **参数说明

| 参数名称 | 输入/输出 | 含义 |
| --- | --- | --- |
| curAddr | 输入 | 已分配的内存地址。 |

#### 约束说明

无

#### 调用示例

请参考调用示例。
