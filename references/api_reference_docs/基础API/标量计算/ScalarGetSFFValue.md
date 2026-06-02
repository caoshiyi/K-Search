# ScalarGetSFFValue

**页面ID:** atlasascendc_api_07_0020  
**来源:** https://www.hiascend.com/document/detail/zh/CANNCommunityEdition/850/API/ascendcopapi/atlasascendc_api_07_0020.html

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

获取一个uint64_t类型数字的二进制表示中从最低有效位开始的第一个0或1出现的位置，如果没找到则返回-1。

#### 函数原型

```
template <int countValue> 
__aicore__ inline int64_t ScalarGetSFFValue(uint64_t valueIn)
```

#### 参数说明

**表1 **模板参数说明

| 参数名 | 描述 |
| --- | --- |
| countValue | 指定要查找的值，0表示查找第一个0的位置，1表示查找第一个1的位置，数据类型是int，只能输入0或1。 |

**表2 **参数说明

| 参数名 | 输入/输出 | 描述 |
| --- | --- | --- |
| valueIn | 输入 | 输入数据，数据类型是uint64_t。 |

#### 返回值说明

int64_t类型的数，valueIn中第一个0或1出现的位置。

#### 约束说明

无。

#### 调用示例

```
uint64_t valueIn = 28;
// 输出数据oneCount：2
int64_t oneCount = AscendC::ScalarGetSFFValue<1>(valueIn);
```
