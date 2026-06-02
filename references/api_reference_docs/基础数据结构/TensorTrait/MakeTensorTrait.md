# MakeTensorTrait

**页面ID:** atlasascendc_api_07_00124  
**来源:** https://www.hiascend.com/document/detail/zh/CANNCommunityEdition/850/API/ascendcopapi/atlasascendc_api_07_00124.html

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

生成TensorTrait实例化对象。

#### 函数原型

```
template <typename T, TPosition pos, typename LayoutType>
__aicore__ inline constexpr auto MakeTensorTrait(const LayoutType& t)
```

#### 参数说明

| 参数名 | 描述 |
| --- | --- |
| T | 只支持如下基础数据类型：int4b_t、uint8_t、int8_t、int16_t、uint16_t、bfloat16_t、int32_t、uint32_t、int64_t、uint64_t、float、half 。 在TensorTrait结构体内部，使用using关键字定义了一个类型别名LiteType，与模板参数T类型一致。 通过TensorTrait定义的LocalTensor/GlobalTensor不包含ShapeInfo信息。 例如：LocalTensor<float>对应的不含ShapeInfo信息的Tensor为LocalTensor<TensorTrait<float>>。 |
| pos | 数据存放的逻辑位置，Tposition类型。 |
| LayoutType | Layout数据类型，输入的数据类型LayoutType，需满足约束说明。 |

#### 返回值说明

返回TensorTrait实例化对象。

#### 约束说明

无

#### 调用示例

见调用示例。
