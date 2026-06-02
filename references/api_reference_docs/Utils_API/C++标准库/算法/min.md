# min

**页面ID:** atlasascendc_api_07_10054  
**来源:** https://www.hiascend.com/document/detail/zh/CANNCommunityEdition/850/API/ascendcopapi/atlasascendc_api_07_10054.html

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

比较相同数据类型的两个数，返回最小值。

#### 函数原型

```
template <typename T, typename U>
__aicore__ inline T min(const T src0, const U src1)
```

#### 参数说明

**表1 **模板参数说明

| 参数名 | 含义 |
| --- | --- |
| T | 输入数据src0的数据类型。当前支持的数据类型为 bool、int8_t、uint8_t、int16_t、uint16_t、int32_t、uint32_t、float、int64_t、uint64_t。 |
| U | 输入数据src1的数据类型。当前支持的数据类型为 bool、int8_t、uint8_t、int16_t、uint16_t、int32_t、uint32_t、float、int64_t、uint64_t。 预留类型，当前必须与T保持一致。 |

**表2 **接口参数说明

| 参数名 | 输入/输出 | 含义 |
| --- | --- | --- |
| src0 | 输入 | 源操作数。参与比较的输入。 |
| src1 | 输入 | 源操作数。参与比较的输入。 |

#### 约束说明

两个源操作数的数据类型必须相同。

#### 返回值说明

两个输入数据中的最小值。

#### 调用示例

```
int64_t src0 = 1;
int64_t src1 = 2;

int64_t result = AscendC::Std::min(src0, src1); 
// result: 1
```
