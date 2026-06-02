# Alloc

**页面ID:** atlasascendc_api_07_00099  
**来源:** https://www.hiascend.com/document/detail/zh/CANNCommunityEdition/850/API/ascendcopapi/atlasascendc_api_07_00099.html

---

#### 产品支持情况

| 产品 | 是否支持 |
| --- | --- |
| Atlas A3 训练系列产品/Atlas A3 推理系列产品 | √ |
| Atlas A2 训练系列产品/Atlas A2 推理系列产品 | √ |
| Atlas 200I/500 A2 推理产品 | x |
| Atlas 推理系列产品AI Core | √ |
| Atlas 推理系列产品Vector Core | √ |
| Atlas 训练系列产品 | x |

#### 功能说明

根据用户指定的逻辑位置、数据类型、数据长度返回对应的LocalTensor对象。

#### 函数原型

- 原型1：tileSize为模板参数

```
// 当tileSize为常量时，建议使用此接口，以获得更优的性能
template <class DataType, uint32_t tileSize> LocalTensor<DataType> __aicore__ inline Alloc()
template <TPosition pos, class DataType, uint32_t tileSize> __aicore__ inline LocalTensor<DataType> Alloc()
```

- 原型2：tileSize为接口入参

```
// 当tileSize为动态参数时使用此接口
template <class DataType> LocalTensor<DataType> __aicore__ inline Alloc(uint32_t tileSize)
template <TPosition pos, class DataType> LocalTensor<DataType> __aicore__ inline Alloc(uint32_t tileSize)
```

- 原型3：使用TensorTrait时使用此接口

```
template <class TensorTraitType> LocalTensor<TensorTraitType> __aicore__ inline Alloc()
```

#### 参数说明

**表1 **原型1和原型2模板参数说明

| 参数名 | 描述 |
| --- | --- |
| pos | TPosition位置，需要符合LocalMemAllocator中指定的Hardware物理位置（静态Tensor编程场景下，此参数可以省略）。 |
| DataType | LocalTensor的数据类型，只支持基础数据类型，不支持TensorTrait类型。 |
| tileSize | LocalTensor的元素个数，其数量不应超过当前物理位置剩余的内存空间。 |

**表2 **原型2参数说明

| 参数名 | 输入/输出 | 描述 |
| --- | --- | --- |
| tileSize | 输入 | LocalTensor的元素个数，其数量不应超过当前物理位置剩余的内存空间。 剩余的内存空间可以通过物理内存最大值与当前可用内存地址（GetCurAddr返回值）的差值来计算。 |

**表3 **原型3模板参数说明

| 参数名 | 描述 |
| --- | --- |
| TensorTraitType | 只支持传入TensorTrait类型，TensorTrait的数据类型/逻辑位置/Shape大小需要匹配LocalMemAllocator中指定的物理位置及其剩余空间。 |

#### 返回值说明

根据用户输入构造的LocalTensor对象。

#### 约束说明

无

#### 调用示例

```
template <uint32_t v>
using UIntImm = Std::integral_constant<uint32_t, v>;
...
AscendC::LocalMemAllocator allocator;
// 原型1：float类型，Tensor中有1024个元素，用户可以指定逻辑位置(或者不指定，由Alloc函数根据物理位置给出默认值，不影响功能)
auto tensor1 = allocator.Alloc<AscendC::TPosition::VECIN, float, 1024>();
auto tensor1 = allocator.Alloc<float, 1024>();

// 原型2：float类型，Tensor中有tileLength个元素，用户可以指定逻辑位置(或者不指定，由Alloc函数根据物理位置给出默认值，不影响功能)
auto tensor1 = allocator.Alloc<AscendC::TPosition::VECIN, float>(tileLength);

// 原型3：用户指定逻辑位置VECIN，数据类型为float，Tensor中元素个数为16*16*16
auto shape = AscendC::MakeShape(UIntImm<16>{}, UIntImm<16>{}, UIntImm<16>{});
auto stride = AscendC::MakeStride(UIntImm<0>{}, UIntImm<0>{}, UIntImm<0>{});
auto layoutMake = AscendC::MakeLayout(shape, stride);
auto tensorTraitMake = AscendC::MakeTensorTrait<float, AscendC::TPosition::VECIN>(layoutMake);
auto tensor3 = allocator.Alloc<decltype(tensorTraitMake)>();
```
