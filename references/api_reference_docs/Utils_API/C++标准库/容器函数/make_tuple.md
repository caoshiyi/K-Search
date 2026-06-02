# make_tuple

**页面ID:** atlasascendc_api_07_10110  
**来源:** https://www.hiascend.com/document/detail/zh/CANNCommunityEdition/850/API/ascendcopapi/atlasascendc_api_07_10110.html

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

make_tuple是一个实用的函数模板，其作用在于便捷地创建tuple对象。它可以自动推断元素的类型，使代码更简洁，也可以构造元素列表。

#### 函数原型

```
template <typename ...Tps>
__aicore__ inline constexpr tuple<unwrap_decay_t<Tps>...> make_tuple(Tps&& ...args)
```

#### 参数说明

**表1 **模板参数说明

| 参数名 | 含义 |
| --- | --- |
| Tps... | Tps...为传入tuple的模板参数包，代表传递给make_tuple的参数类型，参数个数范围为[0, 64]。 Atlas A3 训练系列产品/Atlas A3 推理系列产品，支持的数据类型为：bool、int4b_t、int8_t、uint8_t、int16_t、uint16_t、half、bfloat16_t、int32_t、uint32_t、float、int64_t、uint64_t、LocalTensor、GlobalTensor。 Atlas A2 训练系列产品/Atlas A2 推理系列产品，支持的数据类型为：bool、int4b_t、int8_t、uint8_t、int16_t、uint16_t、half、bfloat16_t、int32_t、uint32_t、float、int64_t、uint64_t、LocalTensor、GlobalTensor。 |
| args | args...是函数参数包，代表传递给make_tuple的实际参数，参数个数范围为[0, 64]。 |

#### 约束说明

- tuple实例化深度为64，即支持64个元素以内的基础数据类型的聚合。
- make_tuple中需要特别指定数据类型的元素，必须对该元素的数据类型增加强转，否则由编译器自行推断，可能与预期不符。
- 不支持数组等可变长度的数据类型。
- 不支持隐式转换构造函数。

#### 返回值说明

一个tuple对象，此对象包含传递的参数副本。

#### 调用示例

```
AscendC::Std::tuple<uint32_t, float, bool> test = AscendC::Std::make_tuple(22, 3.3, true);
```

更多调用示例请参见示例。
