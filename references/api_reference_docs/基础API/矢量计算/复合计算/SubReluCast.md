# SubReluCast

**页面ID:** atlasascendc_api_07_0047  
**来源:** https://www.hiascend.com/document/detail/zh/CANNCommunityEdition/850/API/ascendcopapi/atlasascendc_api_07_0047.html

---

#### 产品支持情况

| 产品 | 是否支持 |
| --- | --- |
| Atlas A3 训练系列产品            /             Atlas A3 推理系列产品 | √ |
| Atlas A2 训练系列产品            /             Atlas A2 推理系列产品 | √ |
| Atlas 200I/500 A2 推理产品 | √ |
| Atlas 推理系列产品            AI Core | √ |
| Atlas 推理系列产品            Vector Core | x |
| Atlas 训练系列产品 | x |

#### 功能说明

按元素求差，再进行Relu计算（结果和0对比取较大值），并根据源操作数和目的操作数Tensor的数据类型进行精度转换。计算公式如下，其中dstType表示目的操作数的数据类型：

![](images/atlasascendc_api_07_0047_img_001.png)

#### 函数原型

- tensor前n个数据计算

```
template <typename T, typename U>
__aicore__ inline void SubReluCast(const LocalTensor<T>& dst, const LocalTensor<U>& src0, const LocalTensor<U>& src1, const uint32_t count)
```

- tensor高维切分计算

  - mask逐bit模式

```
template <typename T, typename U, bool isSetMask = true>
__aicore__ inline void SubReluCast(const LocalTensor<T>& dst, const LocalTensor<U>& src0, const LocalTensor<U>& src1, uint64_t mask[], const uint8_t repeatTime, const BinaryRepeatParams& repeatParams)
```

  - mask连续模式

```
template <typename T, typename U, bool isSetMask = true>
__aicore__ inline void SubReluCast(const LocalTensor<T>& dst, const LocalTensor<U>& src0, const LocalTensor<U>& src1, uint64_t mask, const uint8_t repeatTime, const BinaryRepeatParams& repeatParams)
```

#### 参数说明

**表1 **模板参数说明

| 参数名 | 描述 |
| --- | --- |
| T | 目的操作数数据类型。不同数据类型对应的精度转换规则见表3。                       Atlas A3 训练系列产品            /             Atlas A3 推理系列产品            ，支持的数据类型为：int8_t、half。                       Atlas A2 训练系列产品            /             Atlas A2 推理系列产品            ，支持的数据类型为：int8_t、half。                       Atlas 200I/500 A2 推理产品            ，支持的数据类型为int8_t、half。                       Atlas 推理系列产品            AI Core，支持的数据类型为：int8_t、half。 |
| U | 源操作数数据类型。                       Atlas A3 训练系列产品            /             Atlas A3 推理系列产品            ，支持的数据类型为：int16_t、half、float。                       Atlas A2 训练系列产品            /             Atlas A2 推理系列产品            ，支持的数据类型为：int16_t、half、float。                       Atlas 200I/500 A2 推理产品            ，支持的数据类型为half、float。                       Atlas 推理系列产品            AI Core，支持的数据类型为：int16_t、half、float。 |
| isSetMask | 是否在接口内部设置mask。                     - true，表示在接口内部设置mask。           - false，表示在接口外部设置mask，开发者需要使用SetVectorMask接口设置mask值。这种模式下，本接口入参中的mask值必须设置为占位符MASK_PLACEHOLDER。 |

**表2 **参数说明

| 参数名 | 输入/输出 | 描述 |
| --- | --- | --- |
| dst | 输出 | 目的操作数。          类型为LocalTensor，支持的TPosition为VECIN/VECCALC/VECOUT。 |
| src0、src1 | 输入 | 源操作数。          类型为LocalTensor，支持的TPosition为VECIN/VECCALC/VECOUT。 |
| count | 输入 | 参与计算的元素个数。 |
| mask[]/mask | 输入 | mask用于控制每次迭代内参与计算的元素。                     - 逐bit模式：可以按位控制哪些元素参与计算，bit位的值为1表示参与计算，0表示不参与。            mask为数组形式，数组长度和数组元素的取值范围和操作数的数据类型有关。当操作数为16位时，数组长度为2，mask[0]、mask[1]∈[0, 264-1]并且不同时为0；当操作数为32位时，数组长度为1，mask[0]∈(0, 264-1]；当操作数为64位时，数组长度为1，mask[0]∈(0, 232-1]。            例如，mask=[8, 0]，8=0b1000，表示仅第4个元素参与计算。                               - 连续模式：表示前面连续的多少个元素参与计算。取值范围和操作数的数据类型有关，数据类型不同，每次迭代内能够处理的元素个数最大值不同。当操作数为16位时，mask∈[1, 128]；当操作数为32位时，mask∈[1, 64]；当操作数为64位时，mask∈[1, 32]。                    当源操作数和目的操作数位数不同时，以数据类型的字节较大的为准。例如，源操作数为half类型，目的操作数为int8_t类型，计算mask时以half为准。 |
| repeatTime | 输入 | 重复迭代次数。矢量计算单元，每次读取连续的256Bytes数据进行计算，为完成对输入数据的处理，必须通过多次迭代（repeat）才能完成所有数据的读取与计算。repeatTime表示迭代的次数。          关于该参数的具体描述请参考高维切分API。 |
| repeatParams | 输入 | 控制操作数地址步长的参数。BinaryRepeatParams类型，包含操作数相邻迭代间相同datablock的地址步长，操作数同一迭代内不同datablock的地址步长等参数。          相邻迭代间的地址步长参数说明请参考repeatStride；同一迭代内DataBlock的地址步长参数说明请参考dataBlockStride。 |

**表3 **精度转换规则

| 源操作数的数据类型 | 目的操作数的数据类型 | 类型转换模式介绍 |
| --- | --- | --- |
| float | half | 将源操作数按照CAST_NONE模式取到half所能表示的数，以half格式（溢出默认按照饱和处理）存入目的操作数中。 |
| half | int8_t | 将源操作数按照CAST_NONE模式取整，以int8_t格式（溢出默认按照饱和处理）存入目的操作数中。 |
| int16_t | int8_t | 将源操作数按照CAST_NONE模式取到int8_t所能表示的数，以int8_t格式（溢出默认按照饱和处理）存入目的操作数中。 |

#### 调用示例

本样例的srcLocal为half类型，dstLocal为int8_t类型，计算mask时以half为准。

- tensor高维切分计算样例-mask连续模式

```
uint64_t mask = 256 / sizeof(half); // 128
// repeatTime = 4，一次迭代计算128个数，共计算512个数
// dstBlkStride, src0BlkStride, src1BlkStride = 1，单次迭代内数据连续读取和写入
// dstRepStride = 4，src0RepStride, src1RepStride = 8，相邻迭代间数据连续读取和写入
AscendC::SubReluCast(dstLocal, src0Local, src1Local, mask, 4, { 1, 1, 1, 4, 8, 8 });
```

- tensor高维切分计算样例-mask逐bit模式

```
uint64_t mask[2] = { UINT64_MAX, UINT64_MAX };
// repeatTime = 4，一次迭代计算128个数，共计算512个数
// dstBlkStride, src0BlkStride, src1BlkStride = 1，单次迭代内数据连续读取和写入
// dstRepStride = 4，src0RepStride, src1RepStride = 8，相邻迭代间数据连续读取和写入
AscendC::SubReluCast(dstLocal, src0Local, src1Local, mask, 4, { 1, 1, 1, 4, 8, 8 });
```

- tensor前n个数据计算样例

```
AscendC::SubReluCast(dstLocal, src0Local, src1Local, 512);
```

     结果示例如下：

```
输入数据src0Local：[1 2 3 ... 512]
输入数据src1Local：[0 0.5 4 ... 513]
输出数据dstLocal：[1 2 0 ... 0]
```
