# Muls

**页面ID:** atlasascendc_api_07_0055  
**来源:** https://www.hiascend.com/document/detail/zh/CANNCommunityEdition/850/API/ascendcopapi/atlasascendc_api_07_0055.html

---

#### 产品支持情况

| 产品 | 是否支持 |
| --- | --- |
| Atlas A3 训练系列产品            /             Atlas A3 推理系列产品 | √ |
| Atlas A2 训练系列产品            /             Atlas A2 推理系列产品 | √ |
| Atlas 200I/500 A2 推理产品 | √ |
| Atlas 推理系列产品            AI Core | √ |
| Atlas 推理系列产品            Vector Core | x |
| Atlas 训练系列产品 | √ |

#### 功能说明

矢量内每个元素与标量求积，计算公式如下：

<!-- img2text -->
```
dst_i = src_i * scalar
```

#### 函数原型

- Tensor前n个数据计算

```
template <typename T, bool isSetMask = true>
__aicore__ inline void Muls(const LocalTensor<T>& dst, const LocalTensor<T>& src, const T& scalarValue, const int32_t& count)
```

- Tensor高维切分计算

  - mask逐bit模式

```
template <typename T, bool isSetMask = true>
__aicore__ inline void Muls(const LocalTensor<T>& dst, const LocalTensor<T>& src, const T& scalarValue, uint64_t mask[], const uint8_t repeatTime, const UnaryRepeatParams& repeatParams)
```

  - mask连续模式

```
template <typename T, bool isSetMask = true>
__aicore__ inline void Muls(const LocalTensor<T>& dst, const LocalTensor<T>& src, const T& scalarValue, uint64_t mask, const uint8_t repeatTime, const UnaryRepeatParams& repeatParams)
```

dst和src使用TensorTrait类型时，其数据类型TensorTrait和scalarValue的数据类型（对应TensorTrait中的LiteType类型）不一致。因此新增模板类型U表示scalarValue的数据类型，并通过std::enable_if检查T中萃取出的LiteType和U是否完全一致，一致则接口通过编译，否则编译失败。接口原型定义如下： 

- Tensor前n个数据计算

```
template <typename T, typename U, bool isSetMask = true, typename Std::enable_if<Std::is_same<PrimT<T>, U>::value, bool>::type = true>
__aicore__ inline void Muls(const LocalTensor<T>& dst, const LocalTensor<T>& src, const U& scalarValue, const int32_t& count)
```

- Tensor高维切分计算

  - mask逐bit模式

```
template <typename T, typename U, bool isSetMask = true, typename Std::enable_if<Std::is_same<PrimT<T>, U>::value, bool>::type = true>
__aicore__ inline void Muls(const LocalTensor<T>& dst, const LocalTensor<T>& src, const U& scalarValue, uint64_t mask[], const uint8_t repeatTime, const UnaryRepeatParams& repeatParams)
```

  - mask连续模式

```
template <typename T, typename U, bool isSetMask = true, typename Std::enable_if<Std::is_same<PrimT<T>, U>::value, bool>::type = true>
__aicore__ inline void Muls(const LocalTensor<T>& dst, const LocalTensor<T>& src, const U& scalarValue, uint64_t mask, const uint8_t repeatTime, const UnaryRepeatParams& repeatParams)
```

#### 参数说明

**表1 **模板参数说明

| 参数名 | 描述 |
| --- | --- |
| T | 操作数数据类型。                       Atlas 训练系列产品            ，支持的数据类型为：half/float                       Atlas 推理系列产品            AI Core，支持的数据类型为：half/int16_t/float/int32_t                       Atlas A2 训练系列产品            /             Atlas A2 推理系列产品            ，支持的数据类型为：half/int16_t/float/int32_t                       Atlas A3 训练系列产品            /             Atlas A3 推理系列产品            ，支持的数据类型为：half/int16_t/float/int32_t                       Atlas 200I/500 A2 推理产品            ，支持的数据类型为：half/int16_t/float/int32_t |
| U | scalarValue数据类型。                       Atlas 训练系列产品            ，支持的数据类型为：half/float                       Atlas 推理系列产品            AI Core，支持的数据类型为：half/int16_t/float/int32_t                       Atlas A2 训练系列产品            /             Atlas A2 推理系列产品            ，支持的数据类型为：half/int16_t/float/int32_t                       Atlas A3 训练系列产品            /             Atlas A3 推理系列产品            ，支持的数据类型为：half/int16_t/float/int32_t                       Atlas 200I/500 A2 推理产品            ，支持的数据类型为：half/int16_t/float/int32_t |
| isSetMask | 是否在接口内部设置mask模式和mask值。                     - true，表示在接口内部设置。            tensor高维切分计算API/tensor前n个数据计算API内部使用了mask的Normal模式/Counter模式，一般情况下保持isSetMask默认值即可，表示在API内部进行根据开发者传入的mask/count参数进行mask模式和mask值的设置。           - false，表示在接口外部设置。                           - 针对tensor高维切分计算接口，对性能要求较高的部分场景下，开发者需要使用SetMaskNorm/SetMaskCount设置mask模式，并通过SetVectorMask接口设置mask值。本接口入参中的mask值必须设置为MASK_PLACEHOLDER。               - 针对tensor前n个数据计算接口，对性能要求较高的部分场景下，开发者需要使用SetMaskCount设置mask模式为Counter模式，并通过SetVectorMask接口设置mask值。本接口入参中的count不生效，建议设置成1。                                针对以下型号，tensor前n个数据计算API中的isSetMask参数不生效，保持默认值即可。                     -               Atlas 200I/500 A2 推理产品 |

**表2 **参数说明

| **参数名称** | 输入/输出 | **说明** |
| --- | --- | --- |
| dst | 输出 | 目的操作数。          类型为LocalTensor，支持的TPosition为VECIN/VECCALC/VECOUT。          LocalTensor的起始地址需要32字节对齐。 |
| src | 输入 | 源操作数。          类型为LocalTensor，支持的TPosition为VECIN/VECCALC/VECOUT。          LocalTensor的起始地址需要32字节对齐。          数据类型需要与目的操作数保持一致。 |
| scalarValue | 输入 | 源操作数，数据类型需要与目的操作数Tensor中的元素类型保持一致 |
| count | 输入 | 参与计算的元素个数。 |
| mask/mask[] | 输入 | mask用于控制每次迭代内参与计算的元素。                     - 逐bit模式：可以按位控制哪些元素参与计算，bit位的值为1表示参与计算，0表示不参与。            mask为数组形式，数组长度和数组元素的取值范围和操作数的数据类型有关。当操作数为16位时，数组长度为2，mask[0]、mask[1]∈[0, 264-1]并且不同时为0；当操作数为32位时，数组长度为1，mask[0]∈(0, 264-1]；当操作数为64位时，数组长度为1，mask[0]∈(0, 232-1]。            例如，mask=[8, 0]，8=0b1000，表示仅第4个元素参与计算。                               - 连续模式：表示前面连续的多少个元素参与计算。取值范围和操作数的数据类型有关，数据类型不同，每次迭代内能够处理的元素个数最大值不同。当操作数为16位时，mask∈[1, 128]；当操作数为32位时，mask∈[1, 64]；当操作数为64位时，mask∈[1, 32]。 |
| repeatTime | 输入 | 重复迭代次数。 矢量计算单元，每次读取连续的256Bytes数据进行计算，为完成对输入数据的处理，必须通过多次迭代（repeat）才能完成所有数据的读取与计算。repeatTime表示迭代的次数。          关于该参数的具体描述请参考高维切分API。 |
| repeatParams | 输入 | 元素操作控制结构信息，具体请参考UnaryRepeatParams。 |

#### 约束说明

#### 调用示例

- Tensor高维切分计算样例-mask连续模式

```
uint64_t mask = 128;
int16_t scalar = 2;
// repeatTime = 4, 单次迭代处理128个数，计算512个数需要迭代4次
// dstBlkStride, srcBlkStride = 1, 每个迭代内src0参与计算的数据地址间隔为1个datablock，表示单次迭代内数据连续读取和写入
// dstRepStride, srcRepStride = 8, 相邻迭代间的地址间隔为8个datablock，表示相邻迭代间数据连续读取和写入
AscendC::Muls(dstLocal, srcLocal, scalar, mask, 4, { 1, 1, 8, 8 });
```

- Tensor高维切分计算样例-mask逐bit模式

```
uint64_t mask[2] = { UINT64_MAX, UINT64_MAX };
int16_t scalar = 2;
// repeatTime = 4, 单次迭代处理128个数，计算512个数需要迭代4次
// dstBlkStride, srcBlkStride = 1, 每个迭代内src0参与计算的数据地址间隔为1个datablock，表示单次迭代内数据连续读取和写入
// dstRepStride, srcRepStride = 8, 相邻迭代间的地址间隔为8个datablock，表示相邻迭代间数据连续读取和写入
AscendC::Muls(dstLocal, srcLocal, scalar, mask, 4, {1, 1, 8, 8});
```

- Tensor前n个数据计算样例

```
int16_t scalar = 2;
AscendC::Muls(dstLocal, srcLocal, scalar, 512);
```

     结果示例如下：

```
输入数据srcLocal：[1 2 3 ... 512]
输入数据scalar = 2
输出数据dstLocal：[2 4 6 ... 1024]
```
