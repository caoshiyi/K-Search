# Print

**页面ID:** atlasascendc_api_07_00119  
**来源:** https://www.hiascend.com/document/detail/zh/CANNCommunityEdition/850/API/ascendcopapi/atlasascendc_api_07_00119.html

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

只限于CPU调试，在调试窗口中打印LocalTensor数据用于精度调试，每一行打印一个DataBlock（32字节）的数据。

#### 函数原型

```
inline void Print()
inline void Print(uint32_t len)
```

#### 参数说明

**表1 **参数说明

| 参数名 | 输入/输出 | 描述 |
| --- | --- | --- |
| len | 输入 | 打印元素个数。 |

#### 约束说明

无

#### 调用示例

参考调用示例。
