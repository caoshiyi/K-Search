# ReduceSum

**页面ID:** atlasascendc_api_07_0078  
**来源:** https://www.hiascend.com/document/detail/zh/CANNCommunityEdition/850/API/ascendcopapi/atlasascendc_api_07_0078.html

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

对所有的输入数据求和。

     ReduceSum的相加方式分为两种：

- 方式一：同一repeat内先按照二叉树累加、不同repeat的结果也按照二叉树累加。

假设源操作数为128个half类型的数据[data0,data1,data2...data127]，一个repeat可以计算完，计算过程如下。

  1. data0和data1相加得到data00，data2和data3相加得到data01，...，data124和data125相加得到data62，data126和data127相加得到data63；
  2. data00和data01相加得到data000，data02和data03相加得到data001，...，data62和data63相加得到data031；
  3. 以此类推，得到目的操作数为1个half类型的数据[data]。

需要注意的是两两相加的计算过程中，计算结果大于65504时结果保存为65504。例如源操作数为[60000,60000,-30000,100]，首先60000+60000溢出，结果为65504，第二步计算-30000+100=-29900，第四步计算65504-29900=35604。

- 方式二：同一repeat内采用二叉树累加，不同repeat的结果按顺序累加。

不同硬件形态对应的ReduceSum相加方式如下：

       Atlas A3 训练系列产品
      /
       Atlas A3 推理系列产品
      tensor前n个数据计算接口采用方式二，tensor高维切分计算接口采用方式一

       Atlas A2 训练系列产品
      /
       Atlas A2 推理系列产品
      tensor前n个数据计算接口采用方式二，tensor高维切分计算接口采用方式一

       Atlas 200I/500 A2 推理产品
      采用方式一

       Atlas 推理系列产品
      AI Core采用方式一

       Atlas 训练系列产品
      采用方式一

- sharedTmpBuffer支持两种处理方式：

  - 方式一：按照如下计算公式计算最小所需空间：

```
// 先定义一个向上取整函数
int RoundUp(int a, int b)
{ 
    return (a + b - 1) / b;
}

// 然后定义参与计算的数据类型
int typeSize = 2;                           // half类型为2Bytes，float类型为4Bytes，按需填入

// 再根据数据类型定义两个单位
int elementsPerBlock = 32 / typeSize;       // 1个datablock存放的元素个数
int elementsPerRepeat = 256 / typeSize;     // 1次repeat可以处理的元素个数

// 最后确定首次最大repeat值
int firstMaxRepeat = repeatTime;           // 此处需要注意：对于tensor高维切分计算接口，firstMaxRepeat就是repeatTime；对于tensor前n个数据计算接口，firstMaxRepeat为count/elementsPerRepeat，比如在half类型下firstMaxRepeat就是count/128，在float类型下为count/64，按需填入，对于count<elementsPerRepeat的场景，firstMaxRepeat就是1

int iter1OutputCount = firstMaxRepeat;                                              // 第一轮操作产生的元素个数
int iter1AlignEnd = RoundUp(iter1OutputCount, elementsPerBlock) * elementsPerBlock; // 第一轮产生的元素个数做向上取整
int finalWorkLocalNeedSize = iter1AlignEnd;                                         // 最终sharedTmpBuffer所需的elements空间大小就是第一轮操作产生元素做向上取整后的结果
```

  - 方式二：传入任意大小的sharedTmpBuffer，sharedTmpBuffer的值不会被改变。

#### 函数原型

- tensor前n个数据计算

```
template <typename T, bool isSetMask = true>
__aicore__ inline void ReduceSum(const LocalTensor<T>& dst, const LocalTensor<T>& src, const LocalTensor<T>& sharedTmpBuffer, const int32_t count)
```

- tensor高维切分计算

  - mask逐bit模式

```
template <typename T>
__aicore__ inline void ReduceSum(const LocalTensor<T>& dst, const LocalTensor<T>& src, const LocalTensor<T>& sharedTmpBuffer, const uint64_t mask[], const int32_t repeatTime, const int32_t srcRepStride)
```

  - mask连续模式

```
template <typename T>
__aicore__ inline void ReduceSum(const LocalTensor<T>& dst, const LocalTensor<T>& src, const LocalTensor<T>& sharedTmpBuffer, const int32_t mask, const int32_t repeatTime, const int32_t srcRepStride)
```

#### 参数说明

**表1 **模板参数说明

| 参数名 | 描述 |
| --- | --- |
| T | 操作数数据类型。                       Atlas A3 训练系列产品            /             Atlas A3 推理系列产品            ，支持的数据类型为：half/float                       Atlas A2 训练系列产品            /             Atlas A2 推理系列产品            ，支持的数据类型为：half/float                       Atlas 200I/500 A2 推理产品            ，支持的数据类型为：half/float                       Atlas 推理系列产品            AI Core，支持的数据类型为：half/float                       Atlas 训练系列产品            ，支持的数据类型为：half |
| isSetMask | 预留参数，为后续的功能做保留。保持默认值即可。 |

**表2 **参数说明

| 参数名称 | 输入/输出 | 含义 |
| --- | --- | --- |
| dst | 输出 | 目的操作数。          类型为LocalTensor，支持的TPosition为VECIN/VECCALC/VECOUT。          LocalTensor的起始地址需要保证2字节对齐（针对half数据类型），4字节对齐（针对float数据类型）。 |
| src | 输入 | 源操作数。          类型为LocalTensor，支持的TPosition为VECIN/VECCALC/VECOUT。          LocalTensor的起始地址需要32字节对齐。          源操作数的数据类型需要与目的操作数保持一致。 |
| sharedTmpBuffer | 输入 | 指令执行期间用于存储中间结果，用于内部计算所需操作空间，需特别注意空间大小，参见约束说明。          类型为LocalTensor，支持的TPosition为VECIN/VECCALC/VECOUT。          LocalTensor的起始地址需要32字节对齐。          数据类型需要与目的操作数保持一致。 |
| count | 输入 | 参与计算的元素个数。          参数取值范围和操作数的数据类型有关，数据类型不同，能够处理的元素个数最大值不同，最大处理的数据量不能超过UB大小限制。 |
| mask/mask[] | 输入 | mask用于控制每次迭代内参与计算的元素。                     - 逐bit模式：可以按位控制哪些元素参与计算，bit位的值为1表示参与计算，0表示不参与。            mask为数组形式，数组长度和数组元素的取值范围和操作数的数据类型有关。当操作数为16位时，数组长度为2，mask[0]、mask[1]∈[0, 264-1]并且不同时为0；当操作数为32位时，数组长度为1，mask[0]∈(0, 264-1]；当操作数为64位时，数组长度为1，mask[0]∈(0, 232-1]。            例如，mask=[8, 0]，8=0b1000，表示仅第4个元素参与计算。                               - 连续模式：表示前面连续的多少个元素参与计算。取值范围和操作数的数据类型有关，数据类型不同，每次迭代内能够处理的元素个数最大值不同。当操作数为16位时，mask∈[1, 128]；当操作数为32位时，mask∈[1, 64]；当操作数为64位时，mask∈[1, 32]。 |
| repeatTime | 输入 | 迭代次数。与通用参数说明中不同的是，支持更大的取值范围，保证不超过int32_t最大值的范围即可。 |
| srcRepStride | 输入 | 源操作数相邻迭代间的地址步长，即源操作数每次迭代跳过的datablock数目。详细说明请参考repeatStride。 |

#### 约束说明

- 该接口内部通过软件仿真来实现ReduceSum功能，某些场景下，性能可能不及直接使用硬件指令实现的BlockReduceSum和WholeReduceSum接口。针对不同场景合理使用归约指令可以带来性能提升，相关介绍请参考选择低延迟指令，优化归约操作性能，具体样例请参考[ReduceCustom](https://gitee.com/ascend/samples/tree/master/operator/ascendc/0_introduction/14_reduce_frameworklaunch/ReduceCustom)。

#### 调用示例

- tensor高维切分计算样例-mask连续模式

```
// dstLocal,srcLocal和sharedTmpBuffer均为half类型,srcLocal的计算数据量为8320,并且连续排布，使用tensor高维切分计算接口，设定repeatTime为65，mask为全部元素参与计算
int32_t mask = 128;
AscendC::ReduceSum<half>(dstLocal, srcLocal, sharedTmpBuffer, mask, 65, 8);
```

- tensor高维切分计算样例-mask逐bit模式

```
// dstLocal,srcLocal和sharedTmpBuffer均为half类型,srcLocal的计算数据量为8320,并且连续排布，使用tensor高维切分计算接口，设定repeatTime为65，mask为全部元素参与计算
uint64_t mask[2] = { 0xFFFFFFFFFFFFFFFF, 0xFFFFFFFFFFFFFFFF };
AscendC::ReduceSum<half>(dstLocal, srcLocal, sharedTmpBuffer, mask, 65, 8);
```

- tensor前n个数据计算样例

```
// dstLocal,srcLocal和sharedTmpBuffer均为half类型,srcLocal的计算数据量为8320,并且连续排布，使用tensor前n个数据计算接口
AscendC::ReduceSum<half>(dstLocal, srcLocal, sharedTmpBuffer, 8320);
```

- tensor高维切分计算接口完整示例:

```
#include "kernel_operator.h"
class KernelReduce {
public:
    __aicore__ inline KernelReduce() {}
    __aicore__ inline void Init(__gm__ uint8_t* src, __gm__ uint8_t* dstGm)
    {
        srcGlobal.SetGlobalBuffer((__gm__ half*)src);
        dstGlobal.SetGlobalBuffer((__gm__ half*)dstGm);
        repeat = srcDataSize / mask;
        pipe.InitBuffer(inQueueSrc, 1, srcDataSize * sizeof(half));
        pipe.InitBuffer(workQueue, 1, 80 * sizeof(half)); // 此处按照公式计算所需的最小work空间为80，也就是160Bytes
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
        AscendC::LocalTensor<half> sharedTmpBuffer = workQueue.AllocTensor<half>();
        // level0
        AscendC::ReduceSum<half>(dstLocal, srcLocal, sharedTmpBuffer, mask, repeat, repStride);
        outQueueDst.EnQue<half>(dstLocal);
        inQueueSrc.FreeTensor(srcLocal);
        workQueue.FreeTensor(sharedTmpBuffer);
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
    AscendC::TQue<AscendC::TPosition::VECOUT, 1> workQueue;
    AscendC::TQue<AscendC::TPosition::VECOUT, 1> outQueueDst;
    AscendC::GlobalTensor<half> srcGlobal, dstGlobal;
    int srcDataSize = 8320;
    int dstDataSize = 16;
    int mask = 128;
    int repStride = 8;
    int repeat = 0;
};
```

示例结果如下：

```
输入数据(src_gm):
[1. 1. 1. ... 1. 1. 1.]
输出数据(dst_gm):
[8320.    0.    0.    0.    0.    0.    0.    0.    0.    0.    0.    0.
    0.    0.    0.    0.]
```

- tensor前n个数据计算接口完整示例:

```
#include "kernel_operator.h"
class KernelReduce {
public:
    __aicore__ inline KernelReduce() {}
    __aicore__ inline void Init(__gm__ uint8_t* src, __gm__ uint8_t* dstGm)
    {
        srcGlobal.SetGlobalBuffer((__gm__ half*)src);
        dstGlobal.SetGlobalBuffer((__gm__ half*)dstGm);
        repeat = srcDataSize / mask;
        pipe.InitBuffer(inQueueSrc, 1, srcDataSize * sizeof(half));
        pipe.InitBuffer(workQueue, 1, 16 * sizeof(half)); // 此处按照公式计算所需的最小work空间为16,也就是32Bytes
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
        AscendC::LocalTensor<half> sharedTmpBuffer = workQueue.AllocTensor<half>();
        AscendC::ReduceSum<half>(dstLocal, srcLocal, sharedTmpBuffer, srcDataSize);
        outQueueDst.EnQue<half>(dstLocal);
        inQueueSrc.FreeTensor(srcLocal);
        workQueue.FreeTensor(sharedTmpBuffer);
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
    AscendC::TQue<AscendC::TPosition::VECOUT, 1> workQueue;
    AscendC::TQue<AscendC::TPosition::VECOUT, 1> outQueueDst;
    AscendC::GlobalTensor<half> srcGlobal, dstGlobal;
    int srcDataSize = 288;
    int dstDataSize = 16;
    int mask = 128;
    int repStride = 8;
    int repeat = 0;
};
```

示例结果如下：

```
输入数据(src_gm):
[1. 1. 1. ... 1. 1. 1.]
输出数据(dst_gm):
[288.   0.   0.   0.   0.   0.   0.   0.   0.   0.   0.   0.   0.   0.   0.   0.]
```
