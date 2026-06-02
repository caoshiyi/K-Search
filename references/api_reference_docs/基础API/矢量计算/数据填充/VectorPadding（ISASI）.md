# VectorPadding(ISASI)

**页面ID:** atlasascendc_api_07_0221  
**来源:** https://www.hiascend.com/document/detail/zh/CANNCommunityEdition/850/API/ascendcopapi/atlasascendc_api_07_0221.html

---

#### 产品支持情况

| 产品 | 是否支持 |
| --- | --- |
| Atlas A3 训练系列产品            /             Atlas A3 推理系列产品 | x |
| Atlas A2 训练系列产品            /             Atlas A2 推理系列产品 | x |
| Atlas 200I/500 A2 推理产品 | x |
| Atlas 推理系列产品            AI Core | √ |
| Atlas 推理系列产品            Vector Core | x |
| Atlas 训练系列产品 | x |

#### 功能说明

根据padMode（pad模式）与padSide（pad方向）对源操作数按照datablock进行填充操作。

假设源操作数的一个datablock有16个数，datablock[0:15]=a~p：

- padSide==false：从datablock的左边开始填充，即datablock的起始值方向(a->p)

- padSide==true：从datablock的右边开始填充，即datablock的结束值方向(p->a)
- padMode==0：用邻近数作为填充值，例：aaa|abc(padSide=false)、nop|ppp(padSide=true)
- padMode==1：用邻近datablock值对称填充，例：cba|abc(padSide=false)、nop|pon(padSide=true)
- padMode==2：用邻近datablock值填充，偏移一个数，做对称填充，例：

  - padSide=false：xcb|abc，xcb被填充，填充过程描述：a被丢弃，对称填充，x处填充0
  - padSide=true：nop|onx，onx被填充，填充过程描述：p被丢弃，对称填充，x处填充0

#### 函数原型

- tensor前n个数据计算

```
template <typename T>
__aicore__ inline void VectorPadding(const LocalTensor<T>& dst, const LocalTensor<T>& src, const uint8_t padMode, const bool padSide, const uint32_t count)
```

- tensor高维切分计算

  - mask逐bit模式

```
template <typename T, bool isSetMask = true>
__aicore__ inline void VectorPadding(const LocalTensor<T>& dst, const LocalTensor<T>& src, const uint8_t padMode, const bool padSide, const uint64_t mask[], const uint8_t repeatTime, const UnaryRepeatParams& repeatParams)
```

  - mask连续模式

```
template <typename T, bool isSetMask = true>
__aicore__ inline void VectorPadding(const LocalTensor<T>& dst, const LocalTensor<T>& src, const uint8_t padMode, const bool padSide, const uint64_t mask, const uint8_t repeatTime, const UnaryRepeatParams& repeatParams)
```

#### 参数说明

**表1 **模板参数说明

| 参数名 | 描述 |
| --- | --- |
| T | 操作数数据类型。                       Atlas 推理系列产品            AI Core，支持的数据类型为：int16_t/uint16_t/half/int32_t/uint32_t/float |
| isSetMask | 是否在接口内部设置mask。                     - true，表示在接口内部设置mask。           - false，表示在接口外部设置mask，开发者需要使用SetVectorMask接口设置mask值。这种模式下，本接口入参中的mask值必须设置为占位符MASK_PLACEHOLDER。 |

**表2 **参数说明

| 参数名 | 输入/输出 | 描述 |
| --- | --- | --- |
| dst | 输出 | 目的操作数。          类型为LocalTensor，支持的TPosition为VECIN/VECCALC/VECOUT。          LocalTensor的起始地址需要32字节对齐。 |
| src | 输入 | 源操作数。          类型为LocalTensor，支持的TPosition为VECIN/VECCALC/VECOUT。          LocalTensor的起始地址需要32字节对齐。          源操作数的数据类型需要与目的操作数保持一致。 |
| padMode | 输入 | padding模式，类型为uint8_t，取值范围：[0,2]。                     - 0：用邻近数作为填充值。           - 1：用邻近datablock值对称填充。           - 2：用邻近datablock值填充，偏移一个数，做对称填充。 |
| padSide | 输入 | padding的方向，类型为bool。                     - false：左边。           - true：右边。 |
| count | 输入 | 参与计算的元素个数。 |
| mask[]/mask | 输入 | mask用于控制每次迭代内参与计算的元素。                     - 逐bit模式：可以按位控制哪些元素参与计算，bit位的值为1表示参与计算，0表示不参与。            mask为数组形式，数组长度和数组元素的取值范围和操作数的数据类型有关。当操作数为16位时，数组长度为2，mask[0]、mask[1]∈[0, 264-1]并且不同时为0；当操作数为32位时，数组长度为1，mask[0]∈(0, 264-1]；当操作数为64位时，数组长度为1，mask[0]∈(0, 232-1]。            例如，mask=[8, 0]，8=0b1000，表示仅第4个元素参与计算。                               - 连续模式：表示前面连续的多少个元素参与计算。取值范围和操作数的数据类型有关，数据类型不同，每次迭代内能够处理的元素个数最大值不同。当操作数为16位时，mask∈[1, 128]；当操作数为32位时，mask∈[1, 64]；当操作数为64位时，mask∈[1, 32]。 |
| repeatTime | 输入 | 重复迭代次数。矢量计算单元，每次读取连续的256Bytes数据进行计算，为完成对输入数据的处理，必须通过多次迭代（repeat）才能完成所有数据的读取与计算。repeatTime表示迭代的次数。          关于该参数的具体描述请参考高维切分API。 |
| repeatParams | 输入 | 控制操作数地址步长的参数。UnaryRepeatParams类型，包含操作数相邻迭代间相同DataBlock的地址步长，操作数同一迭代内不同DataBlock的地址步长等参数。          相邻迭代间的地址步长参数说明请参考repeatStride；同一迭代内DataBlock的地址步长参数说明请参考dataBlockStride。 |

#### 约束说明

- mask仅控制目的操作数中的哪些元素要写入，源操作数的读取与mask无关。
- count表示写入目的操作数中的元素总数，源操作数的读取与count无关。

#### 调用示例

样例的srcLocal和dstLocal均为half类型。

- tensor高维切分计算样例-mask连续模式

```
uint64_t mask = 256 / sizeof(half);
uint8_t padMode = 0;
bool padSide = false;
// repeatTime = 4, 128 elements one repeat, 512 elements total
// dstBlkStride, srcBlkStride = 1, no gap between blocks in one repeat
// dstRepStride, srcRepStride = 8, no gap between repeats
AscendC::VectorPadding(dstLocal, srcLocal, padMode, padSide, mask, 4, { 1, 1, 8, 8 });
```

- tensor高维切分计算样例-mask逐bit模式

```
uint64_t mask[2] = { UINT64_MAX, UINT64_MAX };
uint8_t padMode = 0;
bool padSide = false;
// repeatTime = 4, 128 elements one repeat, 512 elements total
// dstBlkStride, srcBlkStride = 1, no gap between blocks in one repeat
// dstRepStride, srcRepStride = 8, no gap between repeats
AscendC::VectorPadding(dstLocal, srcLocal, padMode, padSide, mask, 4, { 1, 1, 8, 8 });
```

- tensor前n个数据计算样例

```
uint8_t padMode = 0;
bool padSide = false;
AscendC::VectorPadding(dstLocal, srcLocal, padMode, padSide, 512);
```

     结果示例如下：

```
// 以srcLocal的一个datablock的值为例，有16个数
输入数据(srcLocal): [6.938 -8.86 -0.2263 ... 1.971 1.778]
输出数据(dstLocal): 
[6.938 6.938 6.938 ... 6.938 6.938]
```
