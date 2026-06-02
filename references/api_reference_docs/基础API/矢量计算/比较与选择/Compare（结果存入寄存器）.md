# Compare（结果存入寄存器）

**页面ID:** atlasascendc_api_07_0067  
**来源:** https://www.hiascend.com/document/detail/zh/CANNCommunityEdition/850/API/ascendcopapi/atlasascendc_api_07_0067.html

---

#### 产品支持情况

| 产品 | 是否支持 |
| --- | --- |
| Atlas A3 训练系列产品            /             Atlas A3 推理系列产品 | √ |
| Atlas A2 训练系列产品            /             Atlas A2 推理系列产品 | √ |
| Atlas 200I/500 A2 推理产品 | x |
| Atlas 推理系列产品            AI Core | √ |
| Atlas 推理系列产品            Vector Core | x |
| Atlas 训练系列产品 | x |

#### 功能说明

逐元素比较两个tensor大小，如果比较后的结果为真，则输出结果的对应比特位为1，否则为0。Compare接口需要mask参数时，可以使用此接口。计算结果存入寄存器中。

支持多种比较模式：

- LT：小于（less than）
- GT：大于（greater than）

- GE：大于或等于（greater than or equal to）
- EQ：等于（equal to）
- NE：不等于（not equal to）
- LE：小于或等于（less than or equal to）

#### 函数原型

- mask逐bit模式

```
template <typename T, bool isSetMask = true>
__aicore__ inline void Compare(const LocalTensor<T>& src0, const LocalTensor<T>& src1, CMPMODE cmpMode, const uint64_t mask[], const BinaryRepeatParams& repeatParams)
```

- mask连续模式

```
template <typename T, bool isSetMask = true>
__aicore__ inline void Compare(const LocalTensor<T>& src0, const LocalTensor<T>& src1, CMPMODE cmpMode, const uint64_t mask, const BinaryRepeatParams& repeatParams)
```

#### 参数说明

**表1 **模板参数说明

| 参数名 | 描述 |
| --- | --- |
| T | 源操作数数据类型。                       Atlas A2 训练系列产品            /             Atlas A2 推理系列产品            ，支持的数据类型为：half/float                       Atlas A3 训练系列产品            /             Atlas A3 推理系列产品            ，支持的数据类型为：half/float                       Atlas 推理系列产品            AI Core，支持的数据类型为：half/float |
| isSetMask | 是否在接口内部设置mask。                     - true，表示在接口内部设置mask。           - false，表示在接口外部设置mask，开发者需要使用SetVectorMask接口设置mask值。这种模式下，本接口入参中的mask值必须设置为占位符MASK_PLACEHOLDER。 |

**表2 **接口参数说明

| 参数名称 | 输入/输出 | 含义 |
| --- | --- | --- |
| src0、src1 | 输入 | 源操作数。          类型为LocalTensor，支持的TPosition为VECIN/VECCALC/VECOUT。          LocalTensor的起始地址需要32字节对齐。 |
| cmpMode | 输入 | CMPMODE类型，表示比较模式，包括EQ，NE，GE，LE，GT，LT。                     - LT： src0小于（less than）src1           - GT： src0大于（greater than）src1           - GE：src0大于或等于（greater than or equal to）src1           - EQ：src0等于（equal to）src1           - NE：src0不等于（not equal to）src1           - LE：src0小于或等于（less than or equal to）src1 |
| mask/mask[] | 输入 | mask用于控制每次迭代内参与计算的元素。                     - 逐bit模式：可以按位控制哪些元素参与计算，bit位的值为1表示参与计算，0表示不参与。            mask为数组形式，数组长度和数组元素的取值范围和操作数的数据类型有关。当操作数为16位时，数组长度为2，mask[0]、mask[1]∈[0, 264-1]并且不同时为0；当操作数为32位时，数组长度为1，mask[0]∈(0, 264-1]；当操作数为64位时，数组长度为1，mask[0]∈(0, 232-1]。            例如，mask=[8, 0]，8=0b1000，表示仅第4个元素参与计算。                               - 连续模式：表示前面连续的多少个元素参与计算。取值范围和操作数的数据类型有关，数据类型不同，每次迭代内能够处理的元素个数最大值不同。当操作数为16位时，mask∈[1, 128]；当操作数为32位时，mask∈[1, 64]；当操作数为64位时，mask∈[1, 32]。 |
| repeatParams | 输入 | 控制操作数地址步长的参数。BinaryRepeatParams类型，包含操作数相邻迭代间相同datablock的地址步长，操作数同一迭代内不同datablock的地址步长等参数。          相邻迭代间的地址步长参数说明请参考repeatStride；同一迭代内DataBlock的地址步长参数说明请参考dataBlockStride。 |

#### 约束说明

- 本接口没有repeat输入，repeat默认为1，即一条指令计算256B的数据。
- 本接口将结果写入128bit的cmpMask寄存器中，可以用GetCmpMask接口获取寄存器保存的数据。

#### 调用示例

本样例中，源操作数src0Local和src1Local各存储了64个float类型的数据。样例实现的功能为，逐元素对src0Local和src1Local中的数据进行比较，如果src0Local中的元素小于src1Local中的元素，dstLocal结果中对应的比特位置1；反之，则置0。dstLocal结果使用uint8_t类型数据存储。

本样例中只展示Compute流程中的部分代码。如果您需要运行样例代码，请将该代码段拷贝并替换样例模板中Compute函数的部分代码即可。

- mask连续模式

```
uint64_t mask = 256 / sizeof(float); // 256为每个迭代处理的字节数
AscendC::BinaryRepeatParams repeatParams = { 1, 1, 1, 8, 8, 8 };
// dstBlkStride, src0BlkStride, src1BlkStride = 1, no gap between blocks in one repeat
// dstRepStride, src0RepStride, src1RepStride = 8, no gap between repeats
AscendC::Compare(src0Local, src1Local, AscendC::CMPMODE::LT, mask, repeatParams);
```

- mask逐bit模式

```
uint64_t mask[2] = { UINT64_MAX, 0};
AscendC::BinaryRepeatParams repeatParams = { 1, 1, 1, 8, 8, 8 };
// srcBlkStride, = 1, no gap between blocks in one repeat
// dstRepStride, srcRepStride = 8, no gap between repeats
AscendC::Compare(src0Local, src1Local, AscendC::CMPMODE::LT, mask, repeatParams);
```

     结果示例如下：

```
输入数据(src0_gm): 
[ 86.72287     9.413112   17.033222  -64.10005   -66.2691    -65.57659
  15.898049   94.61241   -68.920685  -36.16883    15.62852    68.078514
 -59.724575   -9.4302225 -64.770935   66.55523   -84.60122    57.331
  60.42026   -86.78856    37.25265     8.356797  -48.544407   16.73616
  15.28083   -21.889254  -67.93181   -41.01825   -68.79465    20.169441
  44.11346   -27.419518   30.452742  -89.30283   -18.590672   32.45831
   8.392082  -57.198048   98.76846   -81.73067   -38.274437  -83.84363
  64.30617     6.028703  -20.77164    93.71867    54.190437   94.98172
 -47.447758  -65.77461    82.21715    59.953922   23.599781  -77.29708
  26.963976  -63.468987   79.97712   -70.47842    39.00433    52.36555
 -63.94925   -65.77033    26.17237   -71.904884 ]
输入数据(src1_gm): 
[  2.2989323  51.8879    -81.49718    41.189415    6.4081917  92.566666
  53.205498  -94.47063   -75.38387    36.464787   85.60772   -28.70681
  42.58504   -76.15293    38.723816   10.006577   74.53035   -78.38537
  71.945404   -4.060528  -14.501523   28.229202   96.87876    41.558033
 -92.623215   43.318684   35.387154  -16.029816   61.544827    3.3527017
  55.806778  -93.242096   22.86275   -87.506584   35.29523     8.405956
  91.03445   -85.29485    34.30078    -3.8019252  93.40503    15.459968
 -57.99712   -74.39948   -59.900818  -43.132637  -13.123036   41.246174
 -93.01083    75.476875  -45.437893  -99.19293    13.543604   76.23386
  46.192528  -39.23934    75.9787    -38.38979     9.807722  -60.610104
 -23.062874   48.1669     89.913376   73.78631  ]
输出数据(dst_gm): 
[122  86 237  94 150   3 226 242]
```

#### 样例模板

```
#include "kernel_operator.h"
template <typename T> class KernelCmpCmpmask {
public:
    __aicore__ inline KernelCmpCmpmask() {}
    __aicore__ inline void Init(__gm__ uint8_t* src0Gm, __gm__ uint8_t* src1Gm, __gm__ uint8_t* dstGm,
        uint32_t dataSize, AscendC::CMPMODE mode)
    {
        srcDataSize = dataSize;
        dstDataSize = 32;
        cmpMode = mode;
        src0Global.SetGlobalBuffer((__gm__ T*)src0Gm);
        src1Global.SetGlobalBuffer((__gm__ T*)src1Gm);
        dstGlobal.SetGlobalBuffer((__gm__ uint8_t*)dstGm);
        pipe.InitBuffer(inQueueSrc0, 1, srcDataSize * sizeof(T));
        pipe.InitBuffer(inQueueSrc1, 1, srcDataSize * sizeof(T));
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
        AscendC::LocalTensor<T> src0Local = inQueueSrc0.AllocTensor<T>();
        AscendC::LocalTensor<T> src1Local = inQueueSrc1.AllocTensor<T>();
        AscendC::DataCopy(src0Local, src0Global, srcDataSize);
        AscendC::DataCopy(src1Local, src1Global, srcDataSize);
        inQueueSrc0.EnQue(src0Local);
        inQueueSrc1.EnQue(src1Local);
    }
    __aicore__ inline void Compute()
    {
        AscendC::LocalTensor<T> src0Local = inQueueSrc0.DeQue<T>();
        AscendC::LocalTensor<T> src1Local = inQueueSrc1.DeQue<T>();
        AscendC::LocalTensor<uint8_t> dstLocal = outQueueDst.AllocTensor<uint8_t>();
        AscendC::Duplicate(dstLocal.ReinterpretCast<float>(), static_cast<float>(0), 8);
        AscendC::BinaryRepeatParams repeatParams;
        uint32_t mask = 256 / sizeof(T);
        AscendC::Compare(src0Local, src1Local, cmpMode, mask, repeatParams);
        AscendC::PipeBarrier<PIPE_V>();
        AscendC::GetCmpMask(dstLocal);
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
    AscendC::GlobalTensor<T> src0Global, src1Global;
    AscendC::GlobalTensor<uint8_t> dstGlobal;
    uint32_t srcDataSize = 0;
    uint32_t dstDataSize = 0;
    AscendC::CMPMODE cmpMode;
};
template <typename T>
__aicore__ void main_cpu_cmp_cmpmask_demo(__gm__ uint8_t* src0Gm, __gm__ uint8_t* src1Gm, __gm__ uint8_t* dstGm, uint32_t dataSize, AscendC::CMPMODE mode)
{
    KernelCmpCmpmask<T> op;
    op.Init(src0Gm, src1Gm, dstGm, dataSize, mode);
    op.Process();
}
extern "C" __global__ __aicore__ void kernel_vec_compare_cmpmask_64_LT_float(GM_ADDR src0_gm, GM_ADDR src1_gm, GM_ADDR dst_gm)
{
    main_cpu_cmp_cmpmask_demo<float>(src0_gm, src1_gm, dst_gm, 64, AscendC::CMPMODE::LT);
}
```
