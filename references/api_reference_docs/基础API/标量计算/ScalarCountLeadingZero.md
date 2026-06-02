# ScalarCountLeadingZero

**页面ID:** atlasascendc_api_07_0017  
**来源:** https://www.hiascend.com/document/detail/zh/CANNCommunityEdition/850/API/ascendcopapi/atlasascendc_api_07_0017.html

---

#### 产品支持情况

| 产品 | 是否支持 |
| --- | --- |
| Atlas A3 训练系列产品/Atlas A3 推理系列产品 | √ |
| Atlas A2 训练系列产品/Atlas A2 推理系列产品 | √ |
| Atlas 200I/500 A2 推理产品 | x |
| Atlas 推理系列产品AI Core | √ |
| Atlas 推理系列产品Vector Core | x |
| Atlas 训练系列产品 | x |

#### 功能说明

计算一个uint64_t类型数字前导0的个数（二进制从最高位到第一个1一共有多少个0）。

#### 函数原型

```
__aicore__ inline int64_t ScalarCountLeadingZero(uint64_t valueIn)
```

#### 参数说明

**表1 **参数说明

| 参数名 | 输入/输出 | 描述 |
| --- | --- | --- |
| valueIn | 输入 | 被统计的二进制数字。 |

#### 返回值说明

返回valueIn的前导0的个数。

#### 约束说明

无

#### 调用示例

```
uint64_t valueIn = 0x0fffffffffffffff;
// 输出数据ans：4
int64_t ans = AscendC::ScalarCountLeadingZero(valueIn);
```
