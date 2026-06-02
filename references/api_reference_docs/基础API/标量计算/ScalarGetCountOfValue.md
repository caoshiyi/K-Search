# ScalarGetCountOfValue

**页面ID:** atlasascendc_api_07_0016  
**来源:** https://www.hiascend.com/document/detail/zh/CANNCommunityEdition/850/API/ascendcopapi/atlasascendc_api_07_0016.html

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

获取一个uint64_t类型数字的二进制中0或者1的个数。

#### 函数原型

```
template <int countValue> 
__aicore__ inline int64_t ScalarGetCountOfValue(uint64_t valueIn)
```

#### 参数说明

**表1 **模板参数说明

| 参数名 | 描述 |
| --- | --- |
| countValue | 指定统计0还是统计1的个数。 只能输入0或1。 |

**表2 **参数说明

| 参数名 | 输入/输出 | 描述 |
| --- | --- | --- |
| valueIn | 输入 | 被统计的二进制数字。 |

#### 返回值说明

valueIn中0或者1的个数。

#### 约束说明

无

#### 调用示例

```
uint64_t valueIn = 0xffff;
// 输出数据oneCount: 16
int64_t oneCount = AscendC::ScalarGetCountOfValue<1>(valueIn);
```
