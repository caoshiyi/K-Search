# GetPhyAddr

**页面ID:** atlasascendc_api_07_00111  
**来源:** https://www.hiascend.com/document/detail/zh/CANNCommunityEdition/850/API/ascendcopapi/atlasascendc_api_07_00111.html

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

返回LocalTensor的地址或指定偏移量后的地址。

#### 函数原型

```
__aicore__ inline uint64_t GetPhyAddr() const
__aicore__ inline uint64_t GetPhyAddr(const uint32_t offset) const
```

#### 参数说明

**表1 **参数说明

| 参数名 | 输入/输出 | 描述 |
| --- | --- | --- |
| offset | 输入 | 偏移量。 |

#### 返回值说明

LocalTensor的地址或指定偏移量后的地址。

#### 约束说明

无

#### 调用示例

参考调用示例。
