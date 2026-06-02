# get

**页面ID:** atlasascendc_api_07_10109  
**来源:** https://www.hiascend.com/document/detail/zh/CANNCommunityEdition/850/API/ascendcopapi/atlasascendc_api_07_10109.html

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

get的作用是从tuple容器中提取指定位置的元素。

#### 函数原型

```
template <size_t N, typename ...Tps>
__aicore__ inline typename tuple_element<N, tuple<Tps...> >::type& get(tuple<Tps...>& t) noexcept
```

```
template <size_t N, typename ...Tps>
__aicore__ inline const typename tuple_element<N, tuple<Tps...> >::type& get(const tuple<Tps...>& t) noexcept
```

```
template <size_t N, typename ...Tps>
__aicore__ inline typename tuple_element<N, tuple<Tps...> >::type&& get(tuple<Tps...>&& t) noexcept
```

```
template <size_t N, typename ...Tps>
__aicore__ inline const typename tuple_element<N, tuple<Tps...> >::type&& get(const tuple<Tps...>&& t) noexcept
```

#### 参数说明

**表1 **模板参数说明

| 参数名 | 含义 |
| --- | --- |
| N | N是一个编译时常量，代表要提取元素的索引。索引从0开始，取值范围为[0, 64)。 |
| Tps... | Tps...为传入tuple的模板参数包，tuple参数个数范围为(0, 64]。 Atlas A3 训练系列产品/Atlas A3 推理系列产品，支持的数据类型为：bool、int4b_t、int8_t、uint8_t、int16_t、uint16_t、half、bfloat16_t、int32_t、uint32_t、float、int64_t、uint64_t、LocalTensor、GlobalTensor。 Atlas A2 训练系列产品/Atlas A2 推理系列产品，支持的数据类型为：bool、int4b_t、int8_t、uint8_t、int16_t、uint16_t、half、bfloat16_t、int32_t、uint32_t、float、int64_t、uint64_t、LocalTensor、GlobalTensor。 |
| t | t是tuple对象，可以是左值引用、常量左值引用或右值引用。 |

#### 约束说明

get函数仅支持const和constexpr常量索引，索引的取值范围为0, 64)。

#### 返回值说明

tuple对象中对应位置的元素。

#### 调用示例

```
AscendC::Std::tuple<uint32_t, float, bool> test{11, 2.2, true};
uint32_t const_uint32_t = AscendC::Std::get<0>(test);
```

更多调用示例请参见[示例。
