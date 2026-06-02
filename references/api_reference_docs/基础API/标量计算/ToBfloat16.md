# ToBfloat16

**页面ID:** atlasascendc_api_07_0021  
**来源:** https://www.hiascend.com/document/detail/zh/CANNCommunityEdition/850/API/ascendcopapi/atlasascendc_api_07_0021.html

---

#### 产品支持情况

| 产品 | 是否支持 |
| --- | --- |
| Atlas A3 训练系列产品/Atlas A3 推理系列产品 | √ |
| Atlas A2 训练系列产品/Atlas A2 推理系列产品 | √ |
| Atlas 200I/500 A2 推理产品 | x |
| Atlas 推理系列产品AI Core | x |
| Atlas 推理系列产品Vector Core | x |
| Atlas 训练系列产品 | x |

#### 功能说明

float类型标量数据转换成bfloat16_t类型标量数据。

#### 函数原型

```
__aicore__ inline bfloat16_t ToBfloat16(const float& fVal)
```

#### 参数说明

**表1 **接口参数说明

| 参数名称 | 输入/输出 | 含义 |
| --- | --- | --- |
| fVal | 输入 | float类型标量数据。 |

#### 返回值说明

转换后的bfloat16_t类型标量数据。

#### 约束说明

无

#### 调用示例

```
float m = 3.0f;
bfloat16_t n = AscendC::ToBfloat16(m);
```
