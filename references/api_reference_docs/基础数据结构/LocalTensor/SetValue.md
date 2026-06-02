# SetValue

**页面ID:** atlasascendc_api_07_00102  
**来源:** https://www.hiascend.com/document/detail/zh/CANNCommunityEdition/850/API/ascendcopapi/atlasascendc_api_07_00102.html

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

设置LocalTensor中的某个值。

**该接口仅在LocalTensor的TPosition为VECIN/VECCALC/VECOUT时支持。**

#### 函数原型

```
template <typename T1> __aicore__ inline __inout_pipe__(S)
void SetValue(const uint32_t index, const T1 value) const
```

#### 参数说明

**表1 **参数说明

| 参数名 | 输入/输出 | 描述 |
| --- | --- | --- |
| index | 输入 | LocalTensor索引，单位为元素。 |
| value | 输入 | 待设置的数值。 |

#### 约束说明

不要大量使用SetValue对LocalTensor进行赋值，会使性能下降。若需要大批量赋值，请根据实际场景选择数据填充基础API接口或数据填充高阶API接口（Pad、Broadcast），以及在需要生成递增数列的场景，选择Arange。

#### 调用示例

参考调用示例。
