# SoftmaxFlash

**页面ID:** atlasascendc_api_07_0756  
**来源:** https://www.hiascend.com/document/detail/zh/CANNCommunityEdition/850/API/ascendcopapi/atlasascendc_api_07_0756.html

---

#### 产品支持情况

| 产品 | 是否支持 |
| --- | --- |
| Atlas A3 训练系列产品/Atlas A3 推理系列产品 | √ |
| Atlas A2 训练系列产品/Atlas A2 推理系列产品 | √ |
| Atlas 200I/500 A2 推理产品 | x |
| Atlas 推理系列产品AI Core | √ |
| Atlas 推理系列产品Vector Core | x |
| Atlas 训练系列产品 | x |

#### 功能说明

**注意：该接口后续即将废弃，请使用精度和性能更好的SoftmaxFlashV2接口**。

Softmax增强版本，除了可以对输入tensor做SoftmaxFlash计算，还可以根据上一次Softmax计算的sum和max来更新本次的Softmax计算结果。last轴切轴的情况，每次计算的reduce结果并非是全轴的，需要根据上一次Softmax计算的sum和max来更新本次的Softmax计算结果，可以使用该增强接口。不支持NZ格式。

当前仅支持传入shape为ND格式，内部的reduce过程都是按last轴进行。不使能update时，该接口等同于SoftMax。

为方便理解，通过Python脚本实现的方式，表达其计算公式如下，其中src、inmax、 insum、update为输入，dst、x_sum、x_max、exp_max为输出。

```
def softmax_flash(src, inmax=None, insum=None, update=None):
    if update == None:
        #基于last轴进行rowmax(按行取最大值)处理
        x_max = np.max(src, axis=-1, keepdims=True)
        x_sub = src - x_max
        x_exp = np.exp(x_sub)
        #基于last轴进行rowsum(按行求和)处理
        x_sum = np.sum(x_exp, axis=-1, keepdims=True)
        dst = x_exp / x_sum
        exp_max = None
        return dst, x_max, x_sum, exp_max
    else:
        #将inmax和src拼接后求rowmax
        x_max = np.max(np.concatenate((inmax, src), axis=-1), axis=-1, keepdims=True)
        x_exp = np.exp(src - x_max)
        x_sum = np.sum(x_exp, axis=-1, keepdims=True)
        exp_max = np.exp(inmax - x_max)
        x_sum = exp_max * insum +  x_sum
        exp_max = exp_max * insum / x_sum
        dst = x_exp / x_sum
        return dst, x_max, x_sum, exp_max
```

#### 函数原型

- 接口框架申请临时空间

```
template <typename T, bool isReuseSource = false, bool isBasicBlock = false>
__aicore__ inline void SoftmaxFlash(const LocalTensor<T> &dstTensor, const LocalTensor<T> &sumTensor, const LocalTensor<T> &maxTensor, const LocalTensor<T> &srcTensor, const LocalTensor<T> &expMaxTensor, const LocalTensor<T> &inSumTensor, const LocalTensor<T> &inMaxTensor, const SoftMaxTiling &tiling, bool isUpdate = false, const SoftMaxShapeInfo &softmaxShapeInfo = {})
```

```
template <typename T, bool isReuseSource = false, bool isBasicBlock = false>
__aicore__ inline void SoftmaxFlash(const LocalTensor<half>& dstTensor, const LocalTensor<float>& sumTensor, const LocalTensor<float>& maxTensor, const LocalTensor<half>& srcTensor, const LocalTensor<half>& expMaxTensor, const LocalTensor<float>& inSumTensor, const LocalTensor<float>& inMaxTensor, const SoftMaxTiling& tiling, bool isUpdate = false, const SoftMaxShapeInfo& softmaxShapeInfo = {})
```

- 通过sharedTmpBuffer入参传入临时空间

```
template <typename T, bool isReuseSource = false, bool isBasicBlock = false>
__aicore__ inline void SoftmaxFlash(const LocalTensor<T>& dstTensor, const LocalTensor<T>& sumTensor, const LocalTensor<T>& maxTensor, const LocalTensor<T>& srcTensor, const LocalTensor<T>& expMaxTensor, const LocalTensor<T>& inSumTensor, const LocalTensor<T>& inMaxTensor, const LocalTensor<uint8_t>& sharedTmpBuffer, const SoftMaxTiling& tiling, bool isUpdate = false, const SoftMaxShapeInfo& softmaxShapeInfo = {})
```

```
template <typename T, bool isReuseSource = false, bool isBasicBlock = false>
__aicore__ inline void SoftmaxFlash(const LocalTensor<half>& dstTensor, const LocalTensor<float>& sumTensor, const LocalTensor<float>& maxTensor, const LocalTensor<half>& srcTensor, const LocalTensor<half>& expMaxTensor, const LocalTensor<float>& inSumTensor, const LocalTensor<float>& inMaxTensor, const LocalTensor<uint8_t>& sharedTmpBuffer, const SoftMaxTiling& tiling, bool isUpdate = false, const SoftMaxShapeInfo& softmaxShapeInfo = {})
```

由于该接口的内部实现中涉及复杂的计算，需要额外的临时空间来存储计算过程中的中间变量。临时空间支持**接口框架申请**和开发者**通过sharedTmpBuffer入参传入**两种方式。

- 接口框架申请临时空间，开发者无需申请，但是需要预留临时空间的大小。

- 通过sharedTmpBuffer入参传入，使用该tensor作为临时空间进行处理，接口框架不再申请。该方式开发者可以自行管理sharedTmpBuffer内存空间，并在接口调用完成后，复用该部分内存，内存不会反复申请释放，灵活性较高，内存利用率也较高。

接口框架申请的方式，开发者需要预留临时空间；通过sharedTmpBuffer传入的情况，开发者需要为tensor申请空间。临时空间大小BufferSize的获取方式如下：通过SoftmaxFlash Tiling接口中提供的GetSoftMaxFlashMaxTmpSize/GetSoftMaxFlashMinTmpSize接口获取所需最大和最小临时空间大小，最小空间可以保证功能正确，最大空间用于提升性能。

#### 参数说明

**表1 **模板参数说明

| 参数名 | 描述 |
| --- | --- |
| T | 操作数的数据类型。 Atlas A3 训练系列产品/Atlas A3 推理系列产品，支持的数据类型为：half、float。 Atlas A2 训练系列产品/Atlas A2 推理系列产品，支持的数据类型为：half、float。 Atlas 推理系列产品AI Core，支持的数据类型为：half、float。 |
| isReuseSource | 该参数预留，传入默认值false即可。 |
| isBasicBlock | srcTensor和dstTensor的shape信息和Tiling切分策略满足基本块要求的情况下，可以使能该参数用于提升性能，默认不使能。是否满足基本块的要求，可以采用如下两种方式之一判断： - srcTensor和dstTensor的shape信息[m,n]需要满足如下条件：  - 尾轴长度n小于2048并且大于等于256/sizeof(T)（即half场景下n最小为128，float场景下n最小为64），同时n是64的倍数；  - 非尾轴长度的乘积m为8的倍数。  - 在Tiling实现中，通过调用IsBasicBlockInSoftMax判断Tiling切分策略是否满足基本块的切分要求。 |

**表2 **接口参数说明

| 参数名 | 输入/输出 | 描述 |
| --- | --- | --- |
| dstTensor | 输出 | 目的操作数。 类型为LocalTensor，支持的TPosition为VECIN/VECCALC/VECOUT。 dstTensor的shape和源操作数srcTensor一致。 |
| sumTensor | 输出 | 目的操作数。 类型为LocalTensor，支持的TPosition为VECIN/VECCALC/VECOUT。 用于保存softmax计算过程中reducesum的结果。 - sumTensor的last轴长度固定为32Byte，即一个datablock长度。该datablock中的所有数据为同一个值，比如half数据类型下，该datablock中的16个数均为相同的reducesum的值。- 非last轴的长度与dstTensor保持一致。 |
| maxTensor | 输出 | 目的操作数。 类型为LocalTensor，支持的TPosition为VECIN/VECCALC/VECOUT。 用于保存softmax计算过程中reducemax的结果。 - maxTensor的last轴长度固定为32Byte，即一个datablock长度。该datablock中的所有数据为同一个值。比如half数据类型下，该datablock中的16个数均为相同的reducemax的值。- 非last轴的长度与dstTensor保持一致。 |
| srcTensor | 输入 | 源操作数。 类型为LocalTensor，支持的TPosition为VECIN/VECCALC/VECOUT。 last轴长度需要32Byte对齐。 |
| expMaxTensor | 输出 | 目的操作数。 类型为LocalTensor，支持的TPosition为VECIN/VECCALC/VECOUT。 - expMaxTensor的last轴长度固定为32Byte，即一个datablock长度。该datablock中的所有数据为同一个值。比如half数据类型下，该datablock中的16个数均为相同的值。- 非last轴的长度与dstTensor保持一致。 |
| inSumTensor | 输入 | 源操作数。 类型为LocalTensor，支持的TPosition为VECIN/VECCALC/VECOUT。 softmax计算所需要的sum值。 - inSumTensor的last轴长度固定为32Byte，即一个datablock长度。该datablock中的所有数据为同一个值，比如half数据类型下，该datablock中的16个数均为相同的值。- 非last轴的长度需要与dstTensor保持一致。 |
| inMaxTensor | 输入 | 源操作数。 类型为LocalTensor，支持的TPosition为VECIN/VECCALC/VECOUT。 softmax计算所需要的max值。 - inMaxTensor的last轴长度固定为32Byte，即一个datablock长度。该datablock中的所有数据为同一个值，比如half数据类型下，该datablock里的16个数均为相同的值。- 非last轴的长度需要与dstTensor保持一致。 |
| sharedTmpBuffer | 输入 | 临时空间。 类型为LocalTensor，支持的TPosition为VECIN/VECCALC/VECOUT。 接口内部复杂计算时用于存储中间变量，由开发者提供。 临时空间大小BufferSize的获取方式请参考SoftmaxFlash Tiling接口。 |
| tiling | 输入 | 接口计算所需tiling信息，Tiling信息的获取请参考SoftmaxFlash Tiling接口。 |
| isUpdate | 输入 | 是否使能update算法。 |
| srcTensor的shape信息。SoftMaxShapeInfo类型，具体定义如下： ``` struct SoftMaxShapeInfo {     uint32_t srcM; // 非尾轴长度的乘积     uint32_t srcK; // 尾轴长度，必须32Byte对齐     uint32_t oriSrcM; // 原始非尾轴长度的乘积     uint32_t oriSrcK;  // 原始尾轴长度 }; ``` |  |  |

#### 约束说明

- srcTensor和dstTensor的空间可以复用，maxTensor和inMaxTensor的空间可以复用，sumTensor和inSumTensor的空间可以复用。
- sumTensor、maxTensor、expMaxTensor、inSumTensor、inMaxTensor的Tensor空间，last轴长度必须固定32Byte。

- 不支持sharedTmpBuffer与源操作数和目的操作数地址重叠。

#### 调用示例

本样例输入src的Shape大小为[80,144]，输出Shape大小dst=[80,144]，输入inExpSumTensor=[80,16]，输入inMaxTensor=[80,16]，输出expMaxTensor=[80,16]，数据类型均为half，update为false。更多算子样例请参考[softmaxflash算子样例](https://gitee.com/ascend/ascendc-api-adv/tree/master/examples/activation/softmaxflash)。

```
#include "kernel_operator.h"

template <typename T>
class KernelSoftmaxFlash {
public:
    __aicore__ inline KernelSoftmaxFlash()
    {}
    __aicore__ inline void Init(
        GM_ADDR srcGm, GM_ADDR inMaxGm, GM_ADDR inSumGm, GM_ADDR dstGm, const SoftMaxTiling &tilingData)
    {
        elementNumPerBlk = 32 / sizeof(T);
        srcGlobal.SetGlobalBuffer((__gm__ T *)srcGm);
        maxGlobal.SetGlobalBuffer((__gm__ T *)inMaxGm);
        sumGlobal.SetGlobalBuffer((__gm__ T *)inSumGm);
        dstGlobal.SetGlobalBuffer((__gm__ T *)dstGm);
        pipe.InitBuffer(inQueueSrc, 1, height * width * sizeof(T));
        pipe.InitBuffer(outQueueDst, 1, height * width * sizeof(T));
        pipe.InitBuffer(inMaxQueue, 1, height * elementNumPerBlk * sizeof(T));
        pipe.InitBuffer(inSumQueue, 1, height * elementNumPerBlk * sizeof(T));
        pipe.InitBuffer(expMaxQueue, 1, height * elementNumPerBlk * sizeof(T));
        tiling = tilingData;
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
        AscendC::LocalTensor<T> srcLocal = inQueueSrc.AllocTensor<T>();
        AscendC::LocalTensor<T> inSumLocal = inSumQueue.AllocTensor<T>();
        AscendC::LocalTensor<T> inMaxLocal = inMaxQueue.AllocTensor<T>();
        AscendC::DataCopy(srcLocal, srcGlobal, height * width);
        AscendC::DataCopy(inSumLocal, sumGlobal, height * elementNumPerBlk);
        AscendC::DataCopy(inMaxLocal, maxGlobal, height * elementNumPerBlk);
        inQueueSrc.EnQue(srcLocal);
        inSumQueue.EnQue(inSumLocal);
        inMaxQueue.EnQue(inMaxLocal);
    }
    __aicore__ inline void Compute()
    {
        AscendC::LocalTensor<T> srcLocal = inQueueSrc.DeQue<T>();
        AscendC::LocalTensor<T> dstLocal = outQueueDst.AllocTensor<T>();

        AscendC::LocalTensor<T> inMaxLocal = inMaxQueue.AllocTensor<T>();
        AscendC::LocalTensor<T> inSumLocal = inSumQueue.AllocTensor<T>();
        AscendC::LocalTensor<T> expMaxTensor = expMaxQueue.AllocTensor<T>();
        AscendC::SoftMaxShapeInfo srcShape = {height, width, height, width};
        AscendC::SoftmaxFlash<T, false>(srcLocal,
            inSumLocal,
            inMaxLocal,
            srcLocal,
            expMaxTensor,
            inSumLocal,
            inMaxLocal,
            tiling,
            false,
            srcShape);

        AscendC::DataCopy(dstLocal, srcLocal, height * width);

        outQueueDst.EnQue<T>(dstLocal);
        inMaxQueue.FreeTensor(inMaxLocal);
        inSumQueue.FreeTensor(inSumLocal);
        inQueueSrc.FreeTensor(srcLocal);

        expMaxQueue.FreeTensor(expMaxTensor);
    }
    __aicore__ inline void CopyOut()
    {
        AscendC::LocalTensor<T> dstLocal = outQueueDst.DeQue<T>();
        AscendC::DataCopy(dstGlobal, dstLocal, height * width);
        outQueueDst.FreeTensor(dstLocal);
    }

private:
    AscendC::TPipe pipe;
    AscendC::TQue<AscendC::TPosition::VECIN, 1> inQueueSrc;
    AscendC::TQue<AscendC::TPosition::VECOUT, 1> outQueueDst;
    AscendC::TQue<AscendC::TPosition::VECIN, 1> inMaxQueue;
    AscendC::TQue<AscendC::TPosition::VECIN, 1> inSumQueue;
    AscendC::TQue<AscendC::TPosition::VECIN, 1> expMaxQueue;

    AscendC::GlobalTensor<T> srcGlobal, dstGlobal;
    AscendC::GlobalTensor<T> maxGlobal, sumGlobal;
    uint32_t elementNumPerBlk = 0;
    uint32_t width = 144;
    uint32_t height = 80;
    SoftMaxTiling tiling;
};

extern "C" __global__ __aicore__ void softmax_flash_kernel_half(GM_ADDR srcGm, GM_ADDR inMaxGm, GM_ADDR inSumGm, GM_ADDR dstGm, GM_ADDR tiling)
{
    GET_TILING_DATA(tilingData, tiling);
    KernelSoftmaxFlash<half> op;
    op.Init(srcGm, inMaxGm, inSumGm, dstGm, tilingData.softmaxTilingData);
    op.Process();
}
```
