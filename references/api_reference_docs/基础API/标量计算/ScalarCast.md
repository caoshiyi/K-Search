# ScalarCast

**页面ID:** atlasascendc_api_07_0018  
**来源:** https://www.hiascend.com/document/detail/zh/CANNCommunityEdition/850/API/ascendcopapi/atlasascendc_api_07_0018.html

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

对标量的数据类型进行转换。

#### 函数原型

```
template <typename T, typename U, RoundMode roundMode>
__aicore__ inline U ScalarCast(T valueIn)
```

#### 参数说明

**表1 **模板参数说明

| 参数名 | 描述 |
| --- | --- |
| T | valueIn的数据类型，支持float。 |
| U | 转换后的数据类型，支持half、int32_t。 |
| 精度转换处理模式，类型是RoundMode。 RoundMode为枚举类型，用以控制精度转换处理模式，具体定义为： ``` enum class RoundMode {     CAST_NONE = 0,  // 在转换有精度损失时表示CAST_RINT模式，不涉及精度损失时表示不取整     CAST_RINT,      // rint，四舍六入五成双取整     CAST_FLOOR,     // floor，向负无穷取整     CAST_CEIL,      // ceil，向正无穷取整     CAST_ROUND,     // round，四舍五入取整     CAST_TRUNC,     // trunc，向零取整     CAST_ODD,       // Von Neumann rounding，最近邻奇数舍入 }; ```  对于ScalarCast，转换类型仅支持float转half(f322f16)与float转int32_t(f322s32)，相应支持的RoundMode如下： - f322f16：CAST_ODD；- f322s32：CAST_ROUND、CAST_CEIL、CAST_FLOOR、CAST_RINT。 ScalarCast的精度转换规则与Cast保持一致，具体可参考表1。 |  |

**表2 **参数说明

| 参数名 | 输入/输出 | 描述 |
| --- | --- | --- |
| valueIn | 输入 | 被转换数据类型的标量。 |

#### 返回值说明

U类型的valueIn。

#### 约束说明

无

#### 调用示例

```
float valueIn = 2.5;
// 输出数据valueOut：3
int32_t valueOut = AscendC::ScalarCast<float, int32_t, AscendC::RoundMode::CAST_ROUND>(valueIn);
```
