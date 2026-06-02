# Compare

**页面ID:** atlasascendc_api_07_0066  
**来源:** https://www.hiascend.com/document/detail/zh/CANNCommunityEdition/850/API/ascendcopapi/atlasascendc_api_07_0066.html

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

逐元素比较两个tensor大小，如果比较后的结果为真，则输出结果的对应比特位为1，否则为0。

支持多种比较模式：

- LT：小于（less than）
- GT：大于（greater than）

- GE：大于或等于（greater than or equal to）
- EQ：等于（equal to）
- NE：不等于（not equal to）
- LE：小于或等于（less than or equal to）

#### 函数原型

- 整个Tensor参与计算

```
dst = src0 < src1;
dst = src0 > src1;
dst = src0 <= src1;
dst = src0 >= src1;
dst = src0 == src1;
dst = src0 != src1;
```

         Atlas 200I/500 A2 推理产品
        暂不支持整个Tensor参与计算的运算符重载。

- Tensor前n个数据计算

```
template <typename T, typename U>
__aicore__ inline void Compare(const LocalTensor<U>& dst, const LocalTensor<T>& src0, const LocalTensor<T>& src1, CMPMODE cmpMode, uint32_t count)
```

- Tensor高维切分计算

  - mask逐bit模式

```
template <typename T, typename U, bool isSetMask = true>
__aicore__ inline void Compare(const LocalTensor<U>& dst, const LocalTensor<T>& src0, const LocalTensor<T>& src1, CMPMODE cmpMode, const uint64_t mask[], uint8_t repeatTime, const BinaryRepeatParams& repeatParams)
```

  - mask连续模式

```
template <typename T, typename U, bool isSetMask = true>
__aicore__ inline void Compare(const LocalTensor<U>& dst, const LocalTensor<T>& src0, const LocalTensor<T>& src1, CMPMODE cmpMode, const uint64_t mask, uint8_t repeatTime, const BinaryRepeatParams& repeatParams)
```

#### 参数说明

**表1 **模板参数说明

| 参数名 | 描述 |
| --- | --- |
| T | 源操作数数据类型。                       Atlas A3 训练系列产品            /             Atlas A3 推理系列产品            ，支持的数据类型为：half（所有CMPMODE都支持）、float（所有CMPMODE都支持）、 int32_t（只支持CMPMODE::EQ）。                       Atlas A2 训练系列产品            /             Atlas A2 推理系列产品            ，支持的数据类型为：half（所有CMPMODE都支持）、float（所有CMPMODE都支持）、 int32_t（只支持CMPMODE::EQ）。                       Atlas 200I/500 A2 推理产品            ，支持的数据类型为：half、float。                       Atlas 推理系列产品            AI Core，支持的数据类型为：half、float。                       Atlas 训练系列产品            ，支持的数据类型为：half、float。 |
| U | 目的操作数数据类型。                       Atlas A3 训练系列产品            /             Atlas A3 推理系列产品            ，支持的数据类型为：int8_t、uint8_t。                       Atlas A2 训练系列产品            /             Atlas A2 推理系列产品            ，支持的数据类型为：int8_t、uint8_t。                       Atlas 200I/500 A2 推理产品            ，支持的数据类型为：int8_t、uint8_t。                       Atlas 推理系列产品            AI Core，支持的数据类型为：int8_t、uint8_t。                       Atlas 训练系列产品            ，支持的数据类型为：int8_t、uint8_t。 |
| isSetMask | **保留参数，**保持默认值即可。 |

**表2 **接口参数说明

| 参数名称 | 输入/输出 | 含义 |
| --- | --- | --- |
| dst | 输出 | 目的操作数。          类型为LocalTensor，支持的TPosition为VECIN/VECCALC/VECOUT。          LocalTensor的起始地址需要32字节对齐。          dst用于存储比较结果，将dst中uint8_t类型的数据按照bit位展开，由左至右依次表征对应位置的src0和src1的比较结果，如果比较后的结果为真，则对应比特位为1，否则为0。 |
| src0、src1 | 输入 | 源操作数。          类型为LocalTensor，支持的TPosition为VECIN/VECCALC/VECOUT。          LocalTensor的起始地址需要32字节对齐。 |
| cmpMode | 输入 | CMPMODE类型，表示比较模式，包括EQ，NE，GE，LE，GT，LT。                     - LT： src0小于（less than）src1           - GT： src0大于（greater than）src1           - GE：src0大于或等于（greater than or equal to）src1           - EQ：src0等于（equal to）src1           - NE：src0不等于（not equal to）src1           - LE：src0小于或等于（less than or equal to）src1 |
| mask/mask[] | 输入 | mask用于控制每次迭代内参与计算的元素。                       Atlas A3 训练系列产品            /             Atlas A3 推理系列产品            ，保留参数，设置无效。                       Atlas A2 训练系列产品            /             Atlas A2 推理系列产品            ，保留参数，设置无效。                       Atlas 200I/500 A2 推理产品            ，设置有效。                       Atlas 推理系列产品            AI Core，保留参数，设置无效。                       Atlas 训练系列产品            ，保留参数，设置无效。                     - 连续模式：表示前面连续的多少个元素参与计算。取值范围和操作数的数据类型有关，数据类型不同，每次迭代内能够处理的元素个数最大值不同。当操作数为16位时，mask∈[1, 128]；当操作数为32位时，mask∈[1, 64]。                               - 逐bit模式：可以按位控制哪些元素参与计算，bit位的值为1表示参与计算，0表示不参与。参数类型为长度为2或者4的uint64_t类型数组。            例如，mask=[8, 0]，8=0b1000，表示仅第4个元素参与计算。            参数取值范围和操作数的数据类型有关，数据类型不同，每次迭代内能够处理的元素个数最大值不同。当操作数为16位时，mask[0]、mask[1]∈[0, 264-1]并且不同时为0；当操作数为32位时，mask[1]为0，mask[0]∈(0, 264-1]。 |
| repeatTime | 输入 | 重复迭代次数。矢量计算单元，每次读取连续的256Bytes数据进行计算，为完成对输入数据的处理，必须通过多次迭代（repeat）才能完成所有数据的读取与计算。repeatTime表示迭代的次数。          关于该参数的具体描述请参考高维切分API。 |
| repeatParams | 输入 | 控制操作数地址步长的参数。BinaryRepeatParams类型，包含操作数相邻迭代间相同datablock的地址步长，操作数同一迭代内不同datablock的地址步长等参数。          相邻迭代间的地址步长参数说明请参考repeatStride；同一迭代内DataBlock的地址步长参数说明请参考dataBlockStride。 |
| count | 输入 | 参与计算的元素个数。**设置count时，需要保证count个元素所占空间256字节对齐。** |

#### 约束说明

- dst按照小端顺序排序成二进制结果，对应src中相应位置的数据比较结果。
- **使用整个tensor参与计算的运算符重载功能，src0和src1需满足256字节对齐；使用tensor前n个数据参与计算的接口，设置count时，需要保证count个元素所占空间256字节对齐。**

#### 调用示例

本样例中，源操作数src0和src1各存储了256个float类型的数据。样例实现的功能为，逐元素对src0和src1中的数据进行比较，如果src0中的元素小于src1中的元素，dst结果中对应的比特位置1；反之，则置0。dst结果使用uint8_t类型数据存储。

本样例中只展示Compute流程中的部分代码。如果您需要运行样例代码，请将该代码段拷贝并替换样例模板中Compute函数的部分代码即可。

- 整个tensor参与计算

```
dstLocal = src0Local < src1Local;
```

- tensor前n个数据计算

```
AscendC::Compare(dstLocal, src0Local, src1Local, AscendC::CMPMODE::LT, srcDataSize);
```

#### 样例模板

```
#include "kernel_operator.h"
class KernelCmp {
public:
    __aicore__ inline KernelCmp() {}
    __aicore__ inline void Init(__gm__ uint8_t* src0Gm, __gm__ uint8_t* src1Gm, __gm__ uint8_t* dstGm)
    {
        src0Global.SetGlobalBuffer((__gm__ float*)src0Gm);
        src1Global.SetGlobalBuffer((__gm__ float*)src1Gm);
        dstGlobal.SetGlobalBuffer((__gm__ uint8_t*)dstGm);
        pipe.InitBuffer(inQueueSrc0, 1, srcDataSize * sizeof(float));
        pipe.InitBuffer(inQueueSrc1, 1, srcDataSize * sizeof(float));
        pipe.InitBuffer(outQueueDst, 1, dstDataSize * sizeof(uint8_t));
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
        AscendC::LocalTensor<float> src0Local = inQueueSrc0.AllocTensor<float>();
        AscendC::LocalTensor<float> src1Local = inQueueSrc1.AllocTensor<float>();
        AscendC::DataCopy(src0Local, src0Global, srcDataSize);
        AscendC::DataCopy(src1Local, src1Global, srcDataSize);
        inQueueSrc0.EnQue(src0Local);
        inQueueSrc1.EnQue(src1Local);
    }
    __aicore__ inline void Compute()
    {
        AscendC::LocalTensor<float> src0Local = inQueueSrc0.DeQue<float>();
        AscendC::LocalTensor<float> src1Local = inQueueSrc1.DeQue<float>();
        AscendC::LocalTensor<uint8_t> dstLocal = outQueueDst.AllocTensor<uint8_t>();

        // 可根据实际使用接口Compare进行替换
        AscendC::Compare(dstLocal, src0Local, src1Local, AscendC::CMPMODE::LT, srcDataSize);

        outQueueDst.EnQue<uint8_t>(dstLocal);
        inQueueSrc0.FreeTensor(src0Local);
        inQueueSrc1.FreeTensor(src1Local);
    }
    __aicore__ inline void CopyOut()
    {
        AscendC::LocalTensor<uint8_t> dstLocal = outQueueDst.DeQue<uint8_t>();
        AscendC::DataCopy(dstGlobal, dstLocal, dstDataSize);
        outQueueDst.FreeTensor(dstLocal);
    }
private:
    AscendC::TPipe pipe;
    AscendC::TQue<AscendC::TPosition::VECIN, 1> inQueueSrc0, inQueueSrc1;
    AscendC::TQue<AscendC::TPosition::VECOUT, 1> outQueueDst;
    AscendC::GlobalTensor<float> src0Global, src1Global;
    AscendC::GlobalTensor<uint8_t> dstGlobal;
    uint32_t srcDataSize = 256;
    uint32_t dstDataSize = srcDataSize / 8;
};
extern "C" __global__ __aicore__ void main_cpu_cmp_sel_demo(__gm__ uint8_t* src0Gm, __gm__ uint8_t* src1Gm, __gm__ uint8_t* dstGm)
{
    KernelCmp op;
    op.Init(src0Gm, src1Gm, dstGm);
    op.Process();
}
```
