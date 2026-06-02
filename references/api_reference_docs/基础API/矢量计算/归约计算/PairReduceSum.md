# PairReduceSum

**页面ID:** atlasascendc_api_07_0085  
**来源:** https://www.hiascend.com/document/detail/zh/CANNCommunityEdition/850/API/ascendcopapi/atlasascendc_api_07_0085.html

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

PairReduceSum：相邻两个（奇偶）元素求和，例如（a1, a2, a3, a4, a5, a6...），相邻两个数据求和为（a1+a2, a3+a4, a5+a6, ......）。归约指令的总体介绍请参考如何使用归约计算API。

#### 函数原型

- mask逐bit模式

```
template <typename T, bool isSetMask = true>
__aicore__ inline void PairReduceSum(const LocalTensor<T>& dst, const LocalTensor<T>& src, const int32_t repeatTime, const uint64_t mask[], const int32_t dstRepStride, const int32_t srcBlkStride, const int32_t srcRepStride)
```

- mask连续模式

```
template <typename T, bool isSetMask = true>
__aicore__ inline void PairReduceSum(const LocalTensor<T>& dst, const LocalTensor<T>& src, const int32_t repeatTime, const int32_t mask, const int32_t dstRepStride, const int32_t srcBlkStride, const int32_t srcRepStride)
```

#### 参数说明

**表1 **模板参数说明

| 参数名 | 描述 |
| --- | --- |
| T | 操作数数据类型。                       Atlas A3 训练系列产品            /             Atlas A3 推理系列产品            ，支持的数据类型为：half/float                       Atlas A2 训练系列产品            /             Atlas A2 推理系列产品            ，支持的数据类型为：half/float                       Atlas 200I/500 A2 推理产品            ，支持的数据类型为：half/float                       Atlas 推理系列产品            AI Core，支持的数据类型为：half/float                       Atlas 训练系列产品            ，支持的数据类型为：half |
| isSetMask | 是否在接口内部设置mask。                     - true，表示在接口内部设置mask。           - false，表示在接口外部设置mask，开发者需要使用SetVectorMask接口设置mask值。这种模式下，本接口入参中的mask值必须设置为占位符MASK_PLACEHOLDER。 |

**表2 **参数说明

| 参数名称 | 输入/输出 | 含义 |
| --- | --- | --- |
| dst | 输出 | 目的操作数。          类型为LocalTensor，支持的TPosition为VECIN/VECCALC/VECOUT。          LocalTensor的起始地址需要32字节对齐。 |
| src | 输入 | 源操作数。          类型为LocalTensor，支持的TPosition为VECIN/VECCALC/VECOUT。          LocalTensor的起始地址需要32字节对齐。 |
| repeatTime | 输入 | 迭代次数。取值范围为[0, 255]。          关于该参数的具体描述请参考高维切分API。 |
| mask/mask[] | 输入 | mask用于控制每次迭代内参与计算的元素。                     - 逐bit模式：可以按位控制哪些元素参与计算，bit位的值为1表示参与计算，0表示不参与。            mask为数组形式，数组长度和数组元素的取值范围和操作数的数据类型有关。当操作数为16位时，数组长度为2，mask[0]、mask[1]∈[0, 264-1]并且不同时为0；当操作数为32位时，数组长度为1，mask[0]∈(0, 264-1]；当操作数为64位时，数组长度为1，mask[0]∈(0, 232-1]。            例如，mask=[8, 0]，8=0b1000，表示仅第4个元素参与计算。                               - 连续模式：表示前面连续的多少个元素参与计算。取值范围和操作数的数据类型有关，数据类型不同，每次迭代内能够处理的元素个数最大值不同。当操作数为16位时，mask∈[1, 128]；当操作数为32位时，mask∈[1, 64]；当操作数为64位时，mask∈[1, 32]。 |
| dstRepStride | 输入 | 目的操作数相邻迭代间的地址步长。以一个repeat归约后的长度为单位。PairReduce完成后，一个repeat的长度减半。即单位为128Byte。          注意，此参数值             Atlas 训练系列产品            不支持配置0。 |
| srcBlkStride | 输入 | 单次迭代内datablock的地址步长。详细说明请参考dataBlockStride。 |
| srcRepStride | 输入 | 源操作数相邻迭代间的地址步长，即源操作数每次迭代跳过的datablock数目。详细说明请参考repeatStride。 |

#### 约束说明

- 如果两两相加的两个元素mask位未配置（即当前两个元素不参与运算），对于
        Atlas 200I/500 A2 推理产品
       ，对应的目的操作数中的值会置为0，对于其他产品型号，对应的目的操作数中的值不会变化。比如float场景下对64个数使用当前指令，mask配置为62，表示最后两个元素不参与运算，对于
        Atlas 200I/500 A2 推理产品
       ，目的操作数中最后一个值会返回0；对于其他产品型号，目的操作数中最后一个值不会变化。

#### 调用示例

- PairReduceSum-tensor高维切分计算样例-mask连续模式

```
int32_t mask = 256/sizeof(half);
int repeat = 1;
// repeat = 1, 128 elements one repeat, 128 elements total
// srcBlkStride = 1, no gap between blocks in one repeat
// dstRepStride = 1, srcRepStride = 8, no gap between repeats
AscendC::PairReduceSum<half>(dstLocal, srcLocal, repeat, mask, 1, 1, 8);
```

- PairReduceSum-tensor高维切分计算样例-mask逐bit模式

```
uint64_t mask[2] = { UINT64_MAX, UINT64_MAX };
int repeat = 1;
// repeat = 1, 128 elements one repeat, 128 elements total
// srcBlkStride = 1, no gap between blocks in one repeat
// dstRepStride = 1, srcRepStride = 8, no gap between repeats
AscendC::PairReduceSum<half>(dstLocal, srcLocal, repeat, mask, 1, 1, 8);
```

- 示例结果

```
输入数据src_gm：
[-3.441, 7.246, -0.02759, -6.324, 3.693, -7.984, -4.246, 6.332, -3.734, -2.699, -6.91, 7.887, -3.631, 5.219, 6.539, 8.688, 6.523, -6.789, -8.547, 4.258, 1.344, -8.469, -0.9253, -3.914, 3.293, -9.828, 7.082, 5.961, 2.133, 1.959, 3.928, -1.062, 9.18, -1.725, -3.645, 1.457, -2.328, -0.9487, -0.2849, -2.998, -9.281, 3.137, 0.4028, 5.961, -6.25, 2.406, -6.203, -2.699, 4.914, 1.653, -6.383, 6.855, 9.164, 0.6646, -2.854, 3.18, -0.5884, 0.4258, -5.773, -2.152, 4.258, 4.129, -8.719, -8.828, 6.145, 7.387, 1.386, -4.684, 6.324, -1.275, -1.816, 3.357, 6.832, -1.059, -9.852, -8.539, 2.938, -2.002, 9.625, -4.387, -1.309, 8.289, 2.906, -1.035, 7.723, 4.727, -6.477, 2.389, 6.75, -6.688, -0.04248, -6.613, -3.424, 7.145, 4.836, -5.617, -5.855, -5.234, -9.422, -9.852, -8.531, 2.115, 5.109, -8.094, -6.238, 9.898, -6.848, -6.051, 7.109, 4.227, -0.6187, -3.492, -4.352, 1.344, 1.526, 2.572, 2.16, -1.135, 9.812, 1.426, -8, 3.291, -2.039, 5.93, -5.52, -5.156, -9.422, 0.2236]  
输出数据dst_gm：
[3.805, -6.352, -4.289, 2.086, -6.434, 0.9766, 1.588, 15.23, -0.2656, -4.289, -7.125, -4.84, -6.535, 13.05, 4.094, 2.865, 7.453, -2.188, -3.277, -3.283, -6.145, 6.363, -3.844, -8.906, 6.566, 0.4727, 9.828, 0.3262, -0.1626, -7.926, 8.391, -17.55, 13.53, -3.297, 5.047, 1.541, 5.773, -18.39, 0.9355, 5.238, 6.98, 1.871, 12.45, -4.086, 0.0625, -6.656, 3.721, -0.7812, -11.09, -19.28, -6.414, -2.984, 3.66, -12.9, 11.34, -4.109, -3.008, 4.098, 1.025, 11.23, -4.711, 3.891, -10.67, -9.195]
```

- 完整代码样例

```
#include "kernel_operator.h"
class KernelReduce {
public:
    __aicore__ inline KernelReduce() {}
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
        half zero(0);
        AscendC::Duplicate(dstLocal, zero, dstDataSize);
        //指令执行部分（替换成上述代码）
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
    int srcDataSize = 128;
    int dstDataSize = 64;
};
extern "C" __global__ __aicore__ void reduce_simple_kernel(__gm__ uint8_t* src, __gm__ uint8_t* dstGm)
{
    KernelReduce op;
    op.Init(src, dstGm);
    op.Process();
}
```
