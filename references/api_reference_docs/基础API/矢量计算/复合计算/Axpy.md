# Axpy

**页面ID:** atlasascendc_api_07_0063  
**来源:** https://www.hiascend.com/document/detail/zh/CANNCommunityEdition/850/API/ascendcopapi/atlasascendc_api_07_0063.html

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

源操作数src中每个元素与标量求积后和目的操作数dst中的对应元素相加，计算公式如下：

<!-- img2text -->
```



      dstLocal[i] = srcLocal[i] × value + dstLocal[i]


```

说明:
- 图中公式表示：源操作数 `src` 中每个元素与标量 `value` 相乘后，再与目的操作数 `dst` 中对应元素相加，结果写入 `dstLocal[i]`
- 可读作：`dstLocal[i] = srcLocal[i] * value + dstLocal[i]`

#### 函数原型

- tensor前n个数据计算

```
template <typename T, typename U>
__aicore__ inline void Axpy(const LocalTensor<T>& dst, const LocalTensor<U>& src, const U& scalarValue, const int32_t& count)
```

- tensor高维切分计算

  - mask逐bit模式

```
template <typename T, typename U, bool isSetMask = true>
__aicore__ inline void Axpy(const LocalTensor<T>& dst, const LocalTensor<U>& src, const U& scalarValue, uint64_t mask[], const uint8_t repeatTime, const UnaryRepeatParams& repeatParams)
```

  - mask连续模式

```
template <typename T, typename U, bool isSetMask = true>
__aicore__ inline void Axpy(const LocalTensor<T>& dst, const LocalTensor<U>& src, const U& scalarValue, uint64_t mask, const uint8_t repeatTime, const UnaryRepeatParams& repeatParams)
```

#### 参数说明

**表1 **模板参数说明

| 参数名 | 描述 |
| --- | --- |
| T | 目的操作数数据类型。目的操作数和源操作数的数据类型约束请参考表3。                       Atlas 训练系列产品            ，支持的数据类型为：half/float                       Atlas 推理系列产品            AI Core，支持的数据类型为：half/float                       Atlas A2 训练系列产品            /             Atlas A2 推理系列产品            ，支持的数据类型为：half/float                       Atlas A3 训练系列产品            /             Atlas A3 推理系列产品            ，支持的数据类型为：half/float                       Atlas 200I/500 A2 推理产品            ，支持的数据类型为：half/float |
| U | 源操作数数据类型。                       Atlas 训练系列产品            ，支持的数据类型为：half/float                       Atlas 推理系列产品            AI Core，支持的数据类型为：half/float                       Atlas A2 训练系列产品            /             Atlas A2 推理系列产品            ，支持的数据类型为：half/float                       Atlas A3 训练系列产品            /             Atlas A3 推理系列产品            ，支持的数据类型为：half/float                       Atlas 200I/500 A2 推理产品            ，支持的数据类型为：half/float |
| isSetMask | 是否在接口内部设置mask。                     - true，表示在接口内部设置mask。           - false，表示在接口外部设置mask，开发者需要使用SetVectorMask接口设置mask值。这种模式下，本接口入参中的mask值必须设置为占位符MASK_PLACEHOLDER。 |

**表2 **参数说明

| **参数名称** | 输入/输出 | **说明** |
| --- | --- | --- |
| dst | 输出 | 目的操作数。          类型为LocalTensor，支持的TPosition为VECIN/VECCALC/VECOUT。          LocalTensor的起始地址需要32字节对齐。 |
| src | 输入 | 源操作数。          类型为LocalTensor，支持的TPosition为VECIN/VECCALC/VECOUT。          LocalTensor的起始地址需要32字节对齐。 |
| scalarValue | 输入 | 源操作数，scalar标量。scalarValue的数据类型需要和src保持一致。 |
| count | 输入 | 参与计算的元素个数。 |
| mask/mask[] | 输入 | mask用于控制每次迭代内参与计算的元素。                     - 逐bit模式：可以按位控制哪些元素参与计算，bit位的值为1表示参与计算，0表示不参与。            mask为数组形式，数组长度和数组元素的取值范围和操作数的数据类型有关。当操作数为16位时，数组长度为2，mask[0]、mask[1]∈[0, 264-1]并且不同时为0；当操作数为32位时，数组长度为1，mask[0]∈(0, 264-1]；当操作数为64位时，数组长度为1，mask[0]∈(0, 232-1]。            例如，mask=[8, 0]，8=0b1000，表示仅第4个元素参与计算。                               - 连续模式：表示前面连续的多少个元素参与计算。取值范围和操作数的数据类型有关，数据类型不同，每次迭代内能够处理的元素个数最大值不同。当操作数为16位时，mask∈[1, 128]；当操作数为32位时，mask∈[1, 64]；当操作数为64位时，mask∈[1, 32]。 |
| repeatTime | 输入 | 重复迭代次数。          矢量计算单元，每次读取连续的256Bytes数据进行计算，为完成对输入数据的处理，必须通过多次迭代（repeat）才能完成所有数据的读取与计算。repeatTime表示迭代的次数。          关于该参数的具体描述请参考高维切分API。 |
| repeatParams | 输入 | 控制操作数地址步长的参数。UnaryRepeatParams类型，包含操作数相邻迭代间相同DataBlock的地址步长，操作数同一迭代内不同DataBlock的地址步长等参数。          相邻迭代间的地址步长参数说明请参考repeatStride；同一迭代内DataBlock的地址步长参数说明请参考dataBlockStride。 |

**表3 **数据类型约束

| src数据类型 | scalar数据类型 | dst数据类型 | PAR | 支持的型号 |
| --- | --- | --- | --- | --- |
| half | half | half | 128 | Atlas 训练系列产品                                   Atlas A2 训练系列产品            /             Atlas A2 推理系列产品                                   Atlas A3 训练系列产品            /             Atlas A3 推理系列产品                                   Atlas 推理系列产品            AI Core                       Atlas 200I/500 A2 推理产品 |
| float | float | float | 64 | Atlas 训练系列产品                                   Atlas A2 训练系列产品            /             Atlas A2 推理系列产品                                   Atlas A3 训练系列产品            /             Atlas A3 推理系列产品                                   Atlas 推理系列产品            AI Core                       Atlas 200I/500 A2 推理产品 |
| half | half | float | 64 | Atlas 训练系列产品                                   Atlas A2 训练系列产品            /             Atlas A2 推理系列产品                                   Atlas A3 训练系列产品            /             Atlas A3 推理系列产品                                   Atlas 推理系列产品            AI Core                       Atlas 200I/500 A2 推理产品 |

#### 约束说明

- 使用tensor高维切分计算接口时，src和scalar的数据类型为half、dst的数据类型为float的情况下，一个迭代处理的源操作数元素个数需要和目的操作数保持一致，所以每次迭代选取前4个datablock参与计算。设置Repeat Stride参数和mask参数以及地址重叠时，需要考虑该限制。

#### 调用示例

本样例中只展示Compute流程中的部分代码。如果您需要运行样例代码，请将该代码段拷贝并替换更多样例完整样例模板中Compute函数的部分代码即可。

- tensor高维切分计算样例-mask连续模式

```
// repeatTime = 4, mask = 128, 128 elements one repeat, 512 elements total
// srcLocal数据类型为half，scalar数据类型为half，dstLocal数据类型为half
// dstBlkStride, srcBlkStride = 1, no gap between blocks in one repeat
// dstRepStride, srcRepStride = 8, no gap between repeats 
AscendC::Axpy(dstLocal, srcLocal, (half)2.0, 128, 4,{ 1, 1, 8, 8 });

// srcLocal数据类型为half，scalar数据类型为half，dstLocal数据类型为float
// repeatTime = 8, mask = 64, 64 elements one repeat, 512 elements total
// dstBlkStride, srcBlkStride = 1, no gap between blocks in one repeat
// dstRepStride = 8, srcRepStride = 4, no gap between repeats 
AscendC::Axpy(dstLocal, srcLocal, (half)2.0, 64, 8,{ 1, 1, 8, 4 }); // 每次迭代选取源操作数前4个datablock参与计算
```

- tensor高维切分计算样例-mask逐bit模式

```
uint64_t mask[2] = { 0xFFFFFFFFFFFFFFFF, 0xFFFFFFFFFFFFFFFF };
// repeatTime = 4, 128 elements one repeat, 512 elements total, half精度组合
// dstBlkStride, srcBlkStride = 1, no gap between blocks in one repeat
// dstRepStride, srcRepStride = 8, no gap between repeats
AscendC::Axpy(dstLocal, srcLocal, (half)2.0, mask, 4,{ 1, 1, 8, 8 });
```

- tensor前n个数据计算样例

```
AscendC::Axpy(dstLocal, src0Local, (half)2.0, 512);// half精度组合
```

#### 更多样例

- 完整样例一：srcLocal、scalar、dstLocal的数据类型均为half。

```
#include "kernel_operator.h"
class KernelAxpy {
public:
    __aicore__ inline KernelAxpy() {}
    __aicore__ inline void Init(__gm__ uint8_t* srcGm, __gm__ uint8_t* dstGm)
    {
        srcGlobal.SetGlobalBuffer((__gm__ half*)srcGm);
        dstGlobal.SetGlobalBuffer((__gm__ half*)dstGm);
        pipe.InitBuffer(inQueueSrc, 1, 512 * sizeof(half));
        pipe.InitBuffer(outQueueDst, 1, 512 * sizeof(half));
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
        AscendC::DataCopy(srcLocal, srcGlobal, 512);
        inQueueSrc.EnQue(srcLocal);
    }
    __aicore__ inline void Compute()
    {
        AscendC::LocalTensor<half> srcLocal = inQueueSrc.DeQue<half>();
        AscendC::LocalTensor<half> dstLocal = outQueueDst.AllocTensor<half>();

        AscendC::Duplicate(dstLocal, (half)0.0, 512);
        AscendC::Axpy(dstLocal, srcLocal, (half)2.0, 512);

        outQueueDst.EnQue<half>(dstLocal);
        inQueueSrc.FreeTensor(srcLocal);
    }
    __aicore__ inline void CopyOut()
    {
        AscendC::LocalTensor<half> dstLocal = outQueueDst.DeQue<half>();
        AscendC::DataCopy(dstGlobal, dstLocal, 512);
        outQueueDst.FreeTensor(dstLocal);
    }
private:
    AscendC::TPipe pipe;
    AscendC::TQue<AscendC::TPosition::VECIN, 1> inQueueSrc;
    AscendC::TQue<AscendC::TPosition::VECOUT, 1> outQueueDst;
    AscendC::GlobalTensor<half> srcGlobal, dstGlobal;
};
extern "C" __global__ __aicore__ void kernel_vec_ternary_scalar_Axpy_half_2_half(__gm__ uint8_t* srcGm, __gm__ uint8_t* dstGm)
{
    KernelAxpy op;
    op.Init(srcGm, dstGm);
    op.Process();
}
```

结果示例如下：

```
输入数据(srcGm):
[1. 1. 1. 1. 1. 1. ... 1.]
输出数据(dstGm):
[2. 2. 2. 2. 2. 2. ... 2.]
```

- 完整样例二：srcLocal、scalar的数据类型为half，dstLocal的数据类型为float。

```
#include "kernel_operator.h"
class KernelAxpy {
public:
    __aicore__ inline KernelAxpy() {}
    __aicore__ inline void Init(__gm__ uint8_t* srcGm, __gm__ uint8_t* dstGm)
    {
        srcGlobal.SetGlobalBuffer((__gm__ half*)srcGm);
        dstGlobal.SetGlobalBuffer((__gm__ float*)dstGm);
        pipe.InitBuffer(outQueueDst, 1, 512 * sizeof(float));
        pipe.InitBuffer(inQueueSrc, 1, 512 * sizeof(half));
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
        AscendC::DataCopy(srcLocal, srcGlobal, 512);
        inQueueSrc.EnQue(srcLocal);
    }
    __aicore__ inline void Compute()
    {
        AscendC::LocalTensor<half> srcLocal = inQueueSrc.DeQue<half>();
        AscendC::LocalTensor<float> dstLocal = outQueueDst.AllocTensor<float>();

        AscendC::Duplicate(dstLocal, 0.0f, 512);
        AscendC::Axpy(dstLocal, srcLocal, (half)2.0, 64, 8, { 1, 1, 8, 4 });

        outQueueDst.EnQue<float>(dstLocal);
        inQueueSrc.FreeTensor(srcLocal);
    }
    __aicore__ inline void CopyOut()
    {
        AscendC::LocalTensor<float> dstLocal = outQueueDst.DeQue<float>();
        AscendC::DataCopy(dstGlobal, dstLocal, 512);
        outQueueDst.FreeTensor(dstLocal);
    }
private:
    AscendC::TPipe pipe;
    AscendC::TQue<AscendC::TPosition::VECIN, 1> inQueueSrc;
    AscendC::TQue<AscendC::TPosition::VECOUT, 1> outQueueDst;
    AscendC::GlobalTensor<half> srcGlobal;
    AscendC::GlobalTensor<float> dstGlobal;
};
extern "C" __global__ __aicore__ void kernel_vec_ternary_scalar_Axpy_half_2_float(__gm__ uint8_t* srcGm, __gm__ uint8_t* dstGm)
{
    KernelAxpy op;
    op.Init(srcGm, dstGm);
    op.Process();
}
```

结果示例如下：

```
输入数据(srcGm):
[1. 1. 1. 1. 1. 1. ... 1.]
输出数据(dstGm):
[2. 2. 2. 2. 2. 2. ... 2.]
```
