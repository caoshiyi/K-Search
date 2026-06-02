# Duplicate

**页面ID:** atlasascendc_api_07_0088  
**来源:** https://www.hiascend.com/document/detail/zh/CANNCommunityEdition/850/API/ascendcopapi/atlasascendc_api_07_0088.html

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

将一个变量或立即数复制多次并填充到向量中。

#### 函数原型

- tensor前n个数据计算

  - 源操作数为标量

```
template <typename T>
__aicore__ inline void Duplicate(const LocalTensor<T>& dst, const T& scalarValue, const int32_t& count)
```

- tensor高维切分计算

  - mask逐比特模式

```
template <typename T, bool isSetMask = true>
__aicore__ inline void Duplicate(const LocalTensor<T>& dst, const T& scalarValue, uint64_t mask[], const uint8_t repeatTime, const uint16_t dstBlockStride, const uint8_t dstRepeatStride)
```

  - mask连续模式

```
template <typename T, bool isSetMask = true>
__aicore__ inline void Duplicate(const LocalTensor<T>& dst, const T& scalarValue, uint64_t mask, const uint8_t repeatTime, const uint16_t dstBlockStride, const uint8_t dstRepeatStride)
```

#### 参数说明

**表1 **模板参数说明

| 参数名 | 描述 |
| --- | --- |
| T | 操作数数据类型。                       Atlas A3 训练系列产品            /             Atlas A3 推理系列产品            ，支持的数据类型为：int16_t、uint16_t、half、bfloat16_t、int32_t、uint32_t、float。                       Atlas A2 训练系列产品            /             Atlas A2 推理系列产品            ，支持的数据类型为：int16_t、uint16_t、half、bfloat16_t、int32_t、uint32_t、float。                       Atlas 200I/500 A2 推理产品            ，支持的数据类型为：int16_t、uint16_t、half、int32_t、uint32_t、float。                       Atlas 推理系列产品            AI Core，支持的数据类型为：int16_t、uint16_t、half、int32_t、uint32_t、float。                       Atlas 训练系列产品            ，支持的数据类型为：int16_t、uint16_t、half、int32_t、uint32_t、float。 |
| isSetMask | 是否在接口内部设置mask。                     - true，表示在接口内部设置mask。           - false，表示在接口外部设置mask，开发者需要使用SetVectorMask接口设置mask值。这种模式下，本接口入参中的mask值必须设置为占位符MASK_PLACEHOLDER。 |

**表2 **参数说明

| 参数名称 | 输入/输出 | 含义 |
| --- | --- | --- |
| dst | 输出 | 目的操作数。          类型为LocalTensor，支持的TPosition为VECIN/VECCALC/VECOUT。          LocalTensor的起始地址需要32字节对齐。 |
| scalarValue | 输入 | 被复制的源操作数，数据类型需与dst中元素的数据类型保持一致。 |
| count | 输入 | 参与计算的元素个数。 |
| mask/mask[] | 输入 | mask用于控制每次迭代内参与计算的元素。                     - 逐bit模式：可以按位控制哪些元素参与计算，bit位的值为1表示参与计算，0表示不参与。            mask为数组形式，数组长度和数组元素的取值范围和操作数的数据类型有关。当操作数为16位时，数组长度为2，mask[0]、mask[1]∈[0, 264-1]并且不同时为0；当操作数为32位时，数组长度为1，mask[0]∈(0, 264-1]；当操作数为64位时，数组长度为1，mask[0]∈(0, 232-1]。            例如，mask=[8, 0]，8=0b1000，表示仅第4个元素参与计算。                               - 连续模式：表示前面连续的多少个元素参与计算。取值范围和操作数的数据类型有关，数据类型不同，每次迭代内能够处理的元素个数最大值不同。当操作数为16位时，mask∈[1, 128]；当操作数为32位时，mask∈[1, 64]；当操作数为64位时，mask∈[1, 32]。 |
| repeatTime | 输入 | 矢量计算单元，每次读取连续的8个datablock（每个block32Bytes，共256Bytes）数据进行计算，为完成对输入数据的处理，必须通过多次迭代（repeat）才能完成所有数据的读取与计算。repeatTime表示迭代的次数。 |
| dstBlockStride | 输入 | 单次迭代内，矢量目的操作数不同datablock间地址步长。 |
| dstRepeatStride | 输入 | 相邻迭代间，矢量目的操作数相同datablock地址步长。 |

#### 约束说明

#### 调用示例

本示例仅展示Compute流程的部分代码。如需运行，请将代码段复制并粘贴到样例模板中的Compute函数对应位置。

- tensor高维切分计算样例-mask连续模式

```
uint64_t mask = 128;
half scalar = 18.0;
// repeatTime = 2, 128 elements one repeat, 256 elements total
// dstBlkStride = 1, no gap between blocks in one repeat
// dstRepStride = 8, no gap between repeats
AscendC::Duplicate(dstLocal, scalar, mask, 2, 1, 8 );
```

- tensor高维切分计算样例-mask逐bit模式

```
uint64_t mask[2] = { UINT64_MAX, UINT64_MAX };
half scalar = 18.0;
// repeatTime = 2, 128 elements one repeat, 256 elements total
// dstBlkStride = 1, no gap between blocks in one repeat
// dstRepStride = 8, no gap between repeats
AscendC::Duplicate(dstLocal, scalar, mask, 2, 1, 8 );
```

- tensor前n个数据计算样例，源操作数为标量

```
half inputVal(18.0);
AscendC::Duplicate<half>(dstLocal, inputVal, srcDataSize);
```

     结果示例如下：

```
scalar: 18.0
dstLocal: [18.0 18.0 18.0 ... 18.0 18.0]
```

#### 更多样例

您可以参考以下样例，了解如何使用Duplicate指令的tensor高维切分计算接口，进行更灵活的操作、实现更高级的功能。本示例仅展示Compute流程的部分代码。如需运行，请将代码段复制并粘贴到样例模板中的Compute函数对应位置。

- 通过tensor高维切分计算接口中的mask连续模式，实现数据非连续计算。

```
uint64_t mask = 64;  // 每个迭代内只计算前64个数
half scalar = 18.0;
// repeatTime = 2, 128 elements one repeat, 256 elements total
// dstBlkStride = 1, dstRepStride = 8 
AscendC::Duplicate(dstLocal, scalar, mask, 2, 1, 8 );
```

结果示例如下：

```
[18.0 18.0 18.0 ... 18.0  undefined ... undefined 
 18.0 18.0 18.0 ... 18.0 undefined ... undefined ]（每段计算结果或undefined数据长64）
```

- 通过tensor高维切分计算接口中的mask逐bit模式，实现数据非连续计算。

```
uint64_t mask[2] = { UINT64_MAX, 0 };  // mask[0]满，mask[1]空，每次只计算前64个数
half scalar = 18.0;
// repeatTime = 2, 128 elements one repeat, 512 elements total
// dstBlkStride = 1, no gap between blocks in one repeat
// dstRepStride = 8, no gap between repeats
AscendC::Duplicate(dstLocal, scalar, mask, 2, 1, 8);
```

       结果示例：

```
输入数据src0Local: [1.0 2.0 3.0 ... 256.0]
输入数据src1Local: half scalar = 18.0;
输出数据dstLocal: 
[18.0 18.0 18.0 ... 18.0 undefined ... undefined
 18.0 18.0 18.0 ... 18.0 undefined ... undefined]（每段计算结果或undefined数据长64）
```

- 通过控制tensor高维切分计算接口的DataBlock Stride参数，实现数据非连续计算。

```
uint64_t mask = 128;
half scalar = 18.0;
// repeatTime = 1, 128 elements one repeat, 256 elements total
// dstBlkStride = 2, 1 block gap between blocks in one repeat
// dstRepStride = 0, repeatTime = 1
AscendC::Duplicate(dstLocal, scalar, mask, 1, 2, 0);
```

       结果示例：

```
输入数据src0Local: [1.0 2.0 3.0 ... 256.0]
输入数据src1Local: half scalar = 18.0;
输出数据dstLocal: 
[18.0 18.0 18.0 ... 18.0 undefined ... undefined
 18.0 18.0 18.0 ... 18.0 undefined ... undefined
 18.0 18.0 18.0 ... 18.0 undefined ... undefined
 18.0 18.0 18.0 ... 18.0 undefined ... undefined
 18.0 18.0 18.0 ... 18.0 undefined ... undefined
 18.0 18.0 18.0 ... 18.0 undefined ... undefined
 18.0 18.0 18.0 ... 18.0 undefined ... undefined
 18.0 18.0 18.0 ... 18.0 undefined ... undefined]（每段计算结果长16）
```

- 通过控制tensor高维切分计算接口的Repeat Stride参数，实现数据非连续计算。

```
uint64_t mask = 64;
half scalar = 18.0;
// repeatTime = 2, 128 elements one repeat, 256 elements total
// dstBlkStride = 1, no gap between blocks in one repeat
// dstRepStride = 12, 4 blocks gap between repeats
AscendC::Duplicate(dstLocal, scalar, mask, 2, 1, 12);
```

       结果示例：

```
输入数据src0Local: [1.0 2.0 3.0 ... 256.0]
输入数据src1Local: half scalar = 18.0;
输出数据dstLocal: 
[18.0 18.0 18.0 ... 18.0 undefined ... undefined 18.0 18.0 18.0 ... 18.0]（每段计算结果长64，undefined长128）
```

#### 样例模板

```
#include "kernel_operator.h"
class KernelDuplicate {
public:
    __aicore__ inline KernelDuplicate() {}
    __aicore__ inline void Init(__gm__ uint8_t* src, __gm__ uint8_t* dstGm)
    {
        srcGlobal.SetGlobalBuffer((__gm__ half*)src);
        dstGlobal.SetGlobalBuffer((__gm__ half*)dstGm);
        pipe.InitBuffer(inQueueSrc, 1, srcDataSize * sizeof(half));
        pipe.InitBuffer(outQueueDst, 1, dstDataSize * sizeof(half));
    }
    __aicore__ inline void Process()
    {
        CopyIn();
        Compute();
        CopyOut();
    }
private:
    __aicore__ inline void CopyIn()
    {
        AscendC::LocalTensor<half> srcLocal = inQueueSrc.AllocTensor<half>();
        AscendC::DataCopy(srcLocal, srcGlobal, srcDataSize);
        inQueueSrc.EnQue(srcLocal);
    }
    __aicore__ inline void Compute()
    {
        AscendC::LocalTensor<half> srcLocal = inQueueSrc.DeQue<half>();
        AscendC::LocalTensor<half> dstLocal = outQueueDst.AllocTensor<half>();
        half inputVal(18.0);
        AscendC::Duplicate<half>(dstLocal, inputVal, srcDataSize);
        outQueueDst.EnQue<half>(dstLocal);
        inQueueSrc.FreeTensor(srcLocal);
    }
    __aicore__ inline void CopyOut()
    {
        AscendC::LocalTensor<half> dstLocal = outQueueDst.DeQue<half>();
        AscendC::DataCopy(dstGlobal, dstLocal, dstDataSize);
        outQueueDst.FreeTensor(dstLocal);
    }
private:
    AscendC::TPipe pipe;
    AscendC::TQue<AscendC::TPosition::VECIN, 1> inQueueSrc;
    AscendC::TQue<AscendC::TPosition::VECOUT, 1> outQueueDst;
    AscendC::GlobalTensor<half> srcGlobal, dstGlobal;
    int srcDataSize = 256;
    int dstDataSize = 256;
};
extern "C" __global__ __aicore__ void duplicate_kernel(__gm__ uint8_t* src, __gm__ uint8_t* dstGm)
{
    KernelDuplicate op;
    op.Init(src, dstGm);
    op.Process();
}
```
