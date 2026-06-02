# CastDeq

**页面ID:** atlasascendc_api_07_0074  
**来源:** https://www.hiascend.com/document/detail/zh/CANNCommunityEdition/850/API/ascendcopapi/atlasascendc_api_07_0074.html

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

对输入做量化并进行精度转换。不同的数据类型，转换公式不同。

- 在输入类型为int16_t的情况下，对int16_t类型的输入做量化并进行精度转换，得到int8_t/uint8_t类型的数据。使用该接口前需要调用SetDeqScale设置scale、offset、signMode等量化参数。

通过模板参数isVecDeq控制是否选择向量量化模式。

  - 当isVecDeq=false时，根据SetDeqScale设置的scale、offset、signMode，对输入做量化并进行精度转换。计算公式如下：

![](images/atlasascendc_api_07_0074_img_001.png)

  - 当isVecDeq=true时，根据SetDeqScale设置的一段128B的UB上的16组量化参数scale0-scale15、offset0-offset15、signMode0-signMode15，以循环的方式对输入做量化并进行精度转换。计算公式如下：

![](images/atlasascendc_api_07_0074_img_002.png)

- 在输入类型为int32_t的情况下，对int32_t类型的输入做量化并进行精度转换，得到half类型的数据。使用该接口前需要调用SetDeqScale设置scale参数。

.![](images/atlasascendc_api_07_0074_img_003.png)

#### 函数原型

- tensor前n个数据计算

```
template <typename T, typename U, bool isVecDeq = true, bool halfBlock = true>
__aicore__ inline void CastDeq(const LocalTensor<T>& dst, const LocalTensor<U>& src, const uint32_t count)
```

- tensor高维切分计算

  - mask逐bit模式

```
template <typename T, typename U, bool isSetMask = true, bool isVecDeq = true, bool halfBlock = true>
__aicore__ inline void CastDeq(const LocalTensor<T>& dst, const LocalTensor<U>& src, const uint64_t mask[], uint8_t repeatTime, const UnaryRepeatParams& repeatParams)
```

  - mask连续模式

```
template <typename T, typename U, bool isSetMask = true, bool isVecDeq = true, bool halfBlock = true>
__aicore__ inline void CastDeq(const LocalTensor<T>& dst, const LocalTensor<U>& src, const int32_t mask, uint8_t repeatTime, const UnaryRepeatParams& repeatParams)
```

#### 参数说明

**表1 **模板参数说明

| 参数名 | 描述 |
| --- | --- |
| T | 输出Tensor的数据类型。                       Atlas A3 训练系列产品            /             Atlas A3 推理系列产品            ，支持的数据类型为：int8_t/uint8_t/half                       Atlas A2 训练系列产品            /             Atlas A2 推理系列产品            ，支持的数据类型为：int8_t/uint8_t/half                       Atlas 推理系列产品            AI Core，支持的数据类型为：int8_t/uint8_t          和SetDeqScale接口的signMode入参配合使用，当signMode=true时输出数据类型int8_t；signMode=false时输出数据类型uint8_t。 |
| U | 输入Tensor的数据类型。                       Atlas A3 训练系列产品            /             Atlas A3 推理系列产品            ，支持的数据类型为：int16_t/int32_t                       Atlas A2 训练系列产品            /             Atlas A2 推理系列产品            ，支持的数据类型为：int16_t/int32_t                       Atlas 推理系列产品            AI Core，支持的数据类型为：int16_t |
| isSetMask | 是否在接口内部设置mask。                     - true，表示在接口内部设置mask。           - false，表示在接口外部设置mask，开发者需要使用SetVectorMask接口设置mask值。这种模式下，本接口入参中的mask值必须设置为占位符MASK_PLACEHOLDER。 |
| isVecDeq | 控制是否选择向量量化模式。和SetDeqScale(const LocalTensor<T>& src)接口配合使用，当SetDeqScale接口传入Tensor时，isVecDeq必须为true。 |
| halfBlock | 对int16_t类型的输入做量化并进行精度转换得到int8_t/uint8_t类型的数据时，halfBlock参数用于指示输出元素存放在上半还是下半Block。halfBlock=true时，结果存放在下半Block；halfBlock=false时，结果存放在上半Block，如图图1。 |

**表2 **接口参数说明

| 参数名 | 输入/输出 | 描述 |
| --- | --- | --- |
| dst | 输出 | 目的操作数。          类型为LocalTensor，支持的TPosition为VECIN/VECCALC/VECOUT。          LocalTensor的起始地址需要32字节对齐。 |
| src | 输入 | 源操作数。          类型为LocalTensor，支持的TPosition为VECIN/VECCALC/VECOUT。          LocalTensor的起始地址需要32字节对齐。 |
| mask/mask[] | 输入 | mask用于控制每次迭代内参与计算的元素。                     - 逐bit模式：可以按位控制哪些元素参与计算，bit位的值为1表示参与计算，0表示不参与。            mask为数组形式，数组长度和数组元素的取值范围和操作数的数据类型有关。当操作数为16位时，数组长度为2，mask[0]、mask[1]∈[0, 264-1]并且不同时为0；当操作数为32位时，数组长度为1，mask[0]∈(0, 264-1]；当操作数为64位时，数组长度为1，mask[0]∈(0, 232-1]。            例如，mask=[8, 0]，8=0b1000，表示仅第4个元素参与计算。                               - 连续模式：表示前面连续的多少个元素参与计算。取值范围和操作数的数据类型有关，数据类型不同，每次迭代内能够处理的元素个数最大值不同。当操作数为16位时，mask∈[1, 128]；当操作数为32位时，mask∈[1, 64]；当操作数为64位时，mask∈[1, 32]。                    当源操作数和目的操作数位数不同时，以数据类型的字节较大的为准。例如，源操作数为int16_t类型，目的操作数为int8_t类型，计算mask时以int16_t为准。 |
| repeatTime | 输入 | 重复迭代次数。矢量计算单元，每次读取连续的256Bytes数据进行计算，为完成对输入数据的处理，必须通过多次迭代（repeat）才能完成所有数据的读取与计算。repeatTime表示迭代的次数。          关于该参数的具体描述请参考高维切分API。 |
| repeatParams | 输入 | 控制操作数地址步长的参数。UnaryRepeatParams类型，包含操作数相邻迭代间相同DataBlock的地址步长，操作数同一迭代内不同DataBlock的地址步长等参数。          相邻迭代间的地址步长参数说明请参考repeatStride；同一迭代内DataBlock的地址步长参数说明请参考dataBlockStride。 |
| count | 输入 | 参与计算的元素个数。 |

**图1 **halfBlock说明
<!-- img2text -->
```
SRC vector
┌──────────┐ ┌──────────┐ ┌──────────┐ ┌──────────┐ ┌──────────┐ ┌──────────┐ ┌──────────┐ ┌──────────┐
│ BLOCK 0  │ │ BLOCK 1  │ │ BLOCK 2  │ │ BLOCK 3  │ │ BLOCK 4  │ │ BLOCK 5  │ │ BLOCK 6  │ │ BLOCK 7  │
└──────────┘ └──────────┘ └──────────┘ └──────────┘ └──────────┘ └──────────┘ └──────────┘ └──────────┘


dst vector
halfBlock = false
┌──────────┬──────────┐ ┌──────────┬──────────┐ ┌──────────┬──────────┐ ┌──────────┬──────────┐
│  BLOCK0  │//////////│ │  BLOCK1  │//////////│ │  BLOCK2  │//////////│ │  BLOCK3  │//////////│
└──────────┴──────────┘ └──────────┴──────────┘ └──────────┴──────────┘ └──────────┴──────────┘

┌──────────┬──────────┐ ┌──────────┬──────────┐ ┌──────────┬──────────┐ ┌──────────┬──────────┐
│  BLOCK4  │//////////│ │  BLOCK5  │//////////│ │  BLOCK6  │//////////│ │  BLOCK7  │//////////│
└──────────┴──────────┘ └──────────┴──────────┘ └──────────┴──────────┘ └──────────┴──────────┘


dst vector
halfBlock = true
┌──────────┬──────────┐ ┌──────────┬──────────┐ ┌──────────┬──────────┐ ┌──────────┬──────────┐
│//////////│  BLOCK 0 │ │//////////│  BLOCK1  │ │//////////│  BLOCK2  │ │//////////│  BLOCK3  │
└──────────┴──────────┘ └──────────┴──────────┘ └──────────┴──────────┘ └──────────┴──────────┘

┌──────────┬──────────┐ ┌──────────┬──────────┐ ┌──────────┬──────────┐ ┌──────────┬──────────┐
│//////////│  BLOCK4  │ │//////////│  BLOCK5  │ │//////////│  BLOCK6  │ │//////////│  BLOCK7  │
└──────────┴──────────┘ └──────────┴──────────┘ └──────────┴──────────┘ └──────────┴──────────┘
```

#### 约束说明

#### 调用示例

如果您需要运行样例代码，请将该代码段拷贝并替换样例模板中Compute函数的部分代码即可。

- 高维切分计算接口样例-mask连续模式

```
int32_t mask = 256 / sizeof(int16_t);
// repeatTime = 2, 128 elements one repeat, 256 elements total
// dstBlkStride, srcBlkStride = 1, no gap between blocks in one repeat
// dstRepStride, srcRepStride = 8, no gap between repeats
AscendC::CastDeq<uint8_t, int16_t, true, true, true>(dstLocal, srcLocal, mask, 2, { 1, 1, 8, 8 });
```

- 高维切分计算接口样例-mask逐bit模式

```
uint64_t mask[2] = { UINT64_MAX, UINT64_MAX };
// repeatTime = 2, 128 elements one repeat, 256 elements total
// dstBlkStride, srcBlkStride = 1, no gap between blocks in one repeat
// dstRepStride, srcRepStride = 8, no gap between repeats
AscendC::CastDeq<uint8_t, int16_t, true, true, true>(dstLocal, srcLocal, mask, 2, { 1, 1, 8, 8 });
```

- 前n个数计算接口样例

```
AscendC::CastDeq<uint8_t, int16_t, true, true>(dstLocal, srcLocal, 256);
```

结果示例如下：

```
输入数据srcLocal: 
[20 53 26 12 36  6 20 93 66 30 56 99 59 92  7 37 22 47 98 10 85 29 14 46
 17 34 45 17 25 45 82 17 66 94 68 23 67  8 89  8 92  6 10 80 87 20  9 81
 70 62 11 58 38 83 32 14 38 47 41 63 94 26 96 89 88 35 86 55 60 82 15 65
 92 67 83 23 63 25 85 93 50 91 75 60 80 10 55 20 71 14 67 23 31 63  7 93
 69 45 61 23 43 86 11 81 81 36 76 58 53 25 23 51 59 78 82 10 39 40 24 50
 68 49 79 40  4 53 22 38 45 17 29 54  9 66 98 47 12 47 47 20 98  0 59 77
  1 21 39 70 66 20 68  8 77 77 54  0  3 33 37 37 48 60 83 88 27 70 31 49
 75 21 59  3 99 84 92 84 14 44 26 56 72 56 37 52 39 11  2 59 59 65 71 64
 10 65 62 48 42 79 69 69 27 99  8 38 36 77 34 34 60 50 52 50 41 31 95 68
 27 16 42 64 19 47  0 10 36 36 33 62 98 64 32 81 49 53 27 70 35  9 63  7
 10 89  3 39 94 23 89 16 23 60 71 42 46 58 65 90]
输出数据dstLocal: 
[ 0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0 20 53 26 12 36  6 20 93
 66 30 56 99 59 92  7 37  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0
 22 47 98 10 85 29 14 46 17 34 45 17 25 45 82 17  0  0  0  0  0  0  0  0
  0  0  0  0  0  0  0  0 66 94 68 23 67  8 89  8 92  6 10 80 87 20  9 81
  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0 70 62 11 58 38 83 32 14
 38 47 41 63 94 26 96 89  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0
 88 35 86 55 60 82 15 65 92 67 83 23 63 25 85 93  0  0  0  0  0  0  0  0
  0  0  0  0  0  0  0  0 50 91 75 60 80 10 55 20 71 14 67 23 31 63  7 93
  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0 69 45 61 23 43 86 11 81
 81 36 76 58 53 25 23 51  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0
 59 78 82 10 39 40 24 50 68 49 79 40  4 53 22 38  0  0  0  0  0  0  0  0
  0  0  0  0  0  0  0  0 45 17 29 54  9 66 98 47 12 47 47 20 98  0 59 77
  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  1 21 39 70 66 20 68  8
 77 77 54  0  3 33 37 37  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0
 48 60 83 88 27 70 31 49 75 21 59  3 99 84 92 84  0  0  0  0  0  0  0  0
  0  0  0  0  0  0  0  0 14 44 26 56 72 56 37 52 39 11  2 59 59 65 71 64
  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0 10 65 62 48 42 79 69 69
 27 99  8 38 36 77 34 34  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0
 60 50 52 50 41 31 95 68 27 16 42 64 19 47  0 10  0  0  0  0  0  0  0  0
  0  0  0  0  0  0  0  0 36 36 33 62 98 64 32 81 49 53 27 70 35  9 63  7
  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0 10 89  3 39 94 23 89 16
 23 60 71 42 46 58 65 90]
```

#### 样例模板

     为了方便您快速运行指令中的参考样例，本章节提供样例模板。

```
#include "kernel_operator.h"
template <typename srcType, typename dstType>
class KernelCastDeq {
public:
    __aicore__ inline KernelCastDeq() {}
    __aicore__ inline void Init(GM_ADDR src_gm, GM_ADDR dst_gm, uint32_t inputSize, bool halfBlock, bool isVecDeq)
    {
        srcSize = inputSize;
        dstSize = inputSize * 2;
        this->halfBlock = halfBlock;
        this->isVecDeq = isVecDeq;
        src_global.SetGlobalBuffer(reinterpret_cast<__gm__ srcType*>(src_gm), srcSize);
        dst_global.SetGlobalBuffer(reinterpret_cast<__gm__ dstType*>(dst_gm), dstSize);
        pipe.InitBuffer(inQueueX, 1, srcSize * sizeof(srcType));
        pipe.InitBuffer(outQueue, 1, dstSize * sizeof(dstType));
        pipe.InitBuffer(tmpQueue, 1, 128);
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
        AscendC::LocalTensor<srcType> srcLocal = inQueueX.AllocTensor<srcType>();
        AscendC::DataCopy(srcLocal, src_global, srcSize);
        inQueueX.EnQue(srcLocal);
    }
    __aicore__ inline void Compute()
    {
        AscendC::LocalTensor<dstType> dstLocal = outQueue.AllocTensor<dstType>();
        AscendC::LocalTensor<uint64_t> tmpBuffer = tmpQueue.AllocTensor<uint64_t>();
        AscendC::Duplicate(tmpBuffer.ReinterpretCast<int32_t>(), static_cast<int32_t>(0), 32);
        AscendC::PipeBarrier<PIPE_V>();
        AscendC::Duplicate<int32_t>(dstLocal.template ReinterpretCast<int32_t>(), static_cast<int32_t>(0), dstSize / sizeof(int32_t));
        AscendC::PipeBarrier<PIPE_ALL>();
        bool signMode = false;
        if constexpr (AscendC::Std::is_same<dstType, int8_t>::value) {
            signMode = true;
        }
        AscendC::LocalTensor<srcType> srcLocal = inQueueX.DeQue<srcType>();
        if (halfBlock) {
            if (isVecDeq) {
                float vdeqScale[16] = { 1.0 };
                int16_t vdeqOffset[16] = { 0 };
                bool vdeqSignMode[16] = { signMode };
                AscendC::VdeqInfo vdeqInfo(vdeqScale, vdeqOffset, vdeqSignMode);
                AscendC::SetDeqScale(tmpBuffer, vdeqInfo);
                AscendC::CastDeq<dstType, srcType, true, true>(dstLocal, srcLocal, srcSize);
            } else {
                float scale = 1.0;
                int16_t offset = 0;
                AscendC::SetDeqScale(scale, offset, signMode);
                AscendC::CastDeq<dstType, srcType, false, true>(dstLocal, srcLocal, srcSize);
            }
        } else {
            if (isVecDeq) {
                float vdeqScale[16] = { 1.0 };
                int16_t vdeqOffset[16] = { 0 };
                bool vdeqSignMode[16] = { signMode };
                AscendC::VdeqInfo vdeqInfo(vdeqScale, vdeqOffset, vdeqSignMode);
                AscendC::SetDeqScale(tmpBuffer, vdeqInfo);
                AscendC::CastDeq<dstType, srcType, true, false>(dstLocal, srcLocal, srcSize);
            } else {
                float scale = 1.0;
                int16_t offset = 0;
                AscendC::SetDeqScale(scale, offset, signMode);
                AscendC::CastDeq<dstType, srcType, false, false>(dstLocal, srcLocal, srcSize);
            }
        }
        outQueue.EnQue<dstType>(dstLocal);
        tmpQueue.FreeTensor(tmpBuffer);
        inQueueX.FreeTensor(srcLocal);
    }
    __aicore__ inline void CopyOut()
    {
        AscendC::LocalTensor<dstType> dstLocal = outQueue.DeQue<dstType>();
        AscendC::DataCopy(dst_global, dstLocal, dstSize);
        outQueue.FreeTensor(dstLocal);
    }
private:
    AscendC::GlobalTensor<srcType> src_global;
    AscendC::GlobalTensor<dstType> dst_global;
    AscendC::TPipe pipe;
    AscendC::TQue<AscendC::TPosition::VECIN, 1> inQueueX;
    AscendC::TQue<AscendC::TPosition::VECIN, 1> tmpQueue;
    AscendC::TQue<AscendC::TPosition::VECOUT, 1> outQueue;
    bool halfBlock = false;
    bool isVecDeq = false;
    uint32_t srcSize = 0;
    uint32_t dstSize = 0;
};
template <typename srcType, typename dstType>
__aicore__ void kernel_cast_deqscale_operator(GM_ADDR src_gm, GM_ADDR dst_gm, uint32_t dataSize, bool halfBlock, bool isVecDeq)
{
    KernelCastDeq<srcType, dstType> op;
    op.Init(src_gm, dst_gm, dataSize, halfBlock, isVecDeq);
    op.Process();
}
extern "C" __global__ __aicore__ void kernel_cast_deqscale_operator_256_int16_t_uint8_t_true_true(GM_ADDR src_gm, GM_ADDR dst_gm)
{
    kernel_cast_deqscale_operator<int16_t, uint8_t>(src_gm, dst_gm, 256, true, true);
}
```
