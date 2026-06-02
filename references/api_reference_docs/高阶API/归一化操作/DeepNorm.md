# DeepNorm

**页面ID:** atlasascendc_api_07_0808  
**来源:** https://www.hiascend.com/document/detail/zh/CANNCommunityEdition/850/API/ascendcopapi/atlasascendc_api_07_0808.html

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

在深层神经网络训练过程中，执行层LayerNorm归一化时，可以使用DeepNorm进行替代，通过扩大残差连接来提高Transformer的稳定性。

本接口实现了对shape大小为[B，S，H]的输入数据的DeepNorm归一化，其计算公式如下：

DeepNorm(x) = LayerNorm(α * X + SubLayer(X))

SubLayer(X)通常是指在DeepNorm模型中的一个子层（sub-layer），用于实现自注意力机制（self-attention mechanism）。本接口中会整体作为一个输入Tensor传入。

其中LayerNorm的计算公式请参考LayerNorm。

#### 函数原型

- 通过sharedTmpBuffer入参传入临时空间

```
template <typename T, bool isReuseSrc = false, bool isBasicBlock = false>
__aicore__ inline void DeepNorm(const LocalTensor<T>& dstLocal, const LocalTensor<T>& meanLocal, const LocalTensor<T>& rstdLocal, const LocalTensor<T>& srcLocal, const LocalTensor<T>& gxLocal, const LocalTensor<T>& betaLocal, const LocalTensor<T>& gammaLocal, const LocalTensor<uint8_t>& sharedTmpBuffer, const T alpha, const T epsilon, DeepNormTiling& tiling)
```

- 接口框架申请临时空间

```
template <typename T, bool isReuseSrc = false, bool isBasicBlock = false>
__aicore__ inline void DeepNorm(const LocalTensor<T>& dstLocal, const LocalTensor<T>& meanLocal, const LocalTensor<T>& rstdLocal, const LocalTensor<T>& srcLocal, const LocalTensor<T>& gxLocal, const LocalTensor<T>& betaLocal, const LocalTensor<T>& gammaLocal, const T alpha, const T epsilon, DeepNormTiling& tiling)
```

#### 参数说明

**表1 **模板参数说明

| 参数名 | 描述 |
| --- | --- |
| T | 操作数的数据类型。 Atlas A3 训练系列产品/Atlas A3 推理系列产品，支持的数据类型为：half、float。 Atlas A2 训练系列产品/Atlas A2 推理系列产品，支持的数据类型为：half、float。 Atlas 推理系列产品AI Core，支持的数据类型为：half、float。 |
| isReuseSrc | 是否允许修改源操作数，默认值为false。如果开发者允许源操作数被改写，可以使能该参数，使能后能够节省部分内存空间。 设置为**true**，则本接口内部计算时**复用**srcLocal的内存空间，节省内存空间；设置为**false**，则本接口内部计算时**不复用**srcLocal的内存空间。 对于float数据类型输入支持开启该参数，half数据类型输入不支持开启该参数。 isReuseSrc的使用样例请参考更多样例。 |
| isBasicBlock | srcTensor的shape信息满足基本块要求的情况下，可以使能该参数用于提升性能，默认不使能。基本块要求srcTensor的shape需要满足如下条件： - 尾轴即H的长度为64的倍数，但不超过2040；- 非尾轴长度（B*S）为8的倍数。 |

**表2 **接口参数说明

| 参数名 | 输入/输出 | 描述 |
| --- | --- | --- |
| dstLocal | 输出 | 目的操作数。shape为[B，S，H]。H长度不可超过2040。 类型为LocalTensor，支持的TPosition为VECIN/VECCALC/VECOUT。 |
| meanLocal | 输出 | 均值，目的操作数。shape为[B，S]。meanLocal的数据类型需要与dstLocal保持一致。 类型为LocalTensor，支持的TPosition为VECIN/VECCALC/VECOUT。 |
| rstdLocal | 输出 | 方差，目的操作数。shape为[B，S]。rstdLocal的数据类型需要与dstLocal保持一致。 类型为LocalTensor，支持的TPosition为VECIN/VECCALC/VECOUT。 |
| srcLocal | 输入 | 源操作数，shape为[B，S，H]。srcLocal的数据类型需要与目的操作数保持一致，尾轴长度需要32B对齐。H长度不可超过2040。 类型为LocalTensor，支持的TPosition为VECIN/VECCALC/VECOUT。 |
| gxLocal | 输入 | 源操作数，shape为[B，S，H]。gxLocal的数据类型需要与目的操作数保持一致，尾轴长度需要32B对齐。H长度不可超过2040。 类型为LocalTensor，支持的TPosition为VECIN/VECCALC/VECOUT。 该参数对应计算公式中的SubLayer(X)的计算结果。 |
| betaLocal | 输入 | 源操作数，shape为[H]。betaLocal的数据类型需要与目的操作数保持一致，长度需要32B对齐。H长度不可超过2040。 类型为LocalTensor，支持的TPosition为VECIN/VECCALC/VECOUT。 |
| gammaLocal | 输入 | 源操作数，shape为[H]。gammaLocal的数据类型需要与目的操作数保持一致，长度需要32B对齐。H长度不可超过2040。 类型为LocalTensor，支持的TPosition为VECIN/VECCALC/VECOUT。 |
| sharedTmpBuffer | 输入 | 接口内部复杂计算时用于存储中间变量，由开发者提供。 类型为LocalTensor，支持的TPosition为VECIN/VECCALC/VECOUT。 临时空间大小BufferSize的获取方式请参考DeepNorm Tiling。 |
| alpha | 输入 | 权重系数。数据类型需要与目的操作数一致。 |
| epsilon | 输入 | 权重系数， 用来防止除零错误。数据类型需要与目的操作数一致。 |
| tiling | 输入 | DeepNorm计算所需Tiling信息，Tiling信息的获取请参考DeepNorm Tiling。 |

#### 约束说明

- isReuseSrc模板参数为false时，srcLocal和dstLocal的Tensor空间不支持复用。
- 仅支持输入shape为ND格式。
- 输入数据不满足对齐要求时，开发者需要进行补齐，补齐的数据应设置为0，防止出现异常值从而影响网络计算。

#### 调用示例

```
#include "kernel_operator.h"

template <typename dataType, bool isReuseSrc = false, bool isBasicBlock = false>
class KernelDeepNorm {
public:
    __aicore__ inline KernelDeepNorm()
    {}
    __aicore__ inline void Init(GM_ADDR inputGm, GM_ADDR inputGxGm, GM_ADDR betaGm, GM_ADDR gammaGm, GM_ADDR outputGm,
        GM_ADDR outputMeanGm, GM_ADDR outputVarianceGm, const DeepNormCustomTiling &customTiling)
    {
        this->tiling = customTiling.tiling;  // DeepNormTiling
        alpha = customTiling.alpha;
        epsilon = customTiling.epsilon;
        const uint32_t bLength = tiling.bLength;
        const uint32_t sLength = tiling.sLength;
        hLength = tiling.hLength;
        bshLength = bLength * sLength * hLength;
        bsLength = bLength * sLength;
        inputXGlobal.SetGlobalBuffer(reinterpret_cast<__gm__ dataType *>(inputGm), bshLength);
        inputGxGlobal.SetGlobalBuffer(reinterpret_cast<__gm__ dataType *>(inputGxGm), bshLength);
        betaGlobal.SetGlobalBuffer(reinterpret_cast<__gm__ dataType *>(betaGm), hLength);
        gammaGlobal.SetGlobalBuffer(reinterpret_cast<__gm__ dataType *>(gammaGm), hLength);
        outputGlobal.SetGlobalBuffer(reinterpret_cast<__gm__ dataType *>(outputGm), bshLength);
        outputMeanGlobal.SetGlobalBuffer(reinterpret_cast<__gm__ dataType *>(outputMeanGm), bsLength);
        outputVarianceGlobal.SetGlobalBuffer(reinterpret_cast<__gm__ dataType *>(outputVarianceGm), bsLength);
        constexpr uint32_t typeSize = sizeof(dataType);
        pipe.InitBuffer(inQueueX, 1, bshLength * typeSize);
        pipe.InitBuffer(inQueueGx, 1, bshLength * typeSize);
        pipe.InitBuffer(inQueueBeta, 1, hLength * typeSize);
        pipe.InitBuffer(inQueueGamma, 1, hLength * typeSize);
        pipe.InitBuffer(outQueue, 1, bshLength * typeSize);
        pipe.InitBuffer(outMeanQueue, 1, bsLength * typeSize);
        pipe.InitBuffer(outVarianceQueue, 1, bsLength * typeSize);
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
        AscendC::LocalTensor<dataType> inputXLocal = inQueueX.AllocTensor<dataType>();
        AscendC::LocalTensor<dataType> inputGxLocal = inQueueGx.AllocTensor<dataType>();
        AscendC::LocalTensor<dataType> betaLocal = inQueueBeta.AllocTensor<dataType>();
        AscendC::LocalTensor<dataType> gammaLocal = inQueueGamma.AllocTensor<dataType>();
        AscendC::DataCopy(inputXLocal, inputXGlobal, bshLength);
        AscendC::DataCopy(inputGxLocal, inputGxGlobal, bshLength);
        AscendC::DataCopy(betaLocal, betaGlobal, hLength);
        AscendC::DataCopy(gammaLocal, gammaGlobal, hLength);
        inQueueX.EnQue(inputXLocal);
        inQueueGx.EnQue(inputGxLocal);
        inQueueBeta.EnQue(betaLocal);
        inQueueGamma.EnQue(gammaLocal);
    }

    __aicore__ inline void Compute()
    {
        AscendC::LocalTensor<dataType> inputXLocal = inQueueX.DeQue<dataType>();
        AscendC::LocalTensor<dataType> inputGxLocal = inQueueGx.DeQue<dataType>();
        AscendC::LocalTensor<dataType> betaLocal = inQueueBeta.DeQue<dataType>();
        AscendC::LocalTensor<dataType> gammaLocal = inQueueGamma.DeQue<dataType>();
        AscendC::LocalTensor<dataType> outputLocal = outQueue.AllocTensor<dataType>();
        AscendC::LocalTensor<dataType> outputMeanLocal = outMeanQueue.AllocTensor<dataType>();
        AscendC::LocalTensor<dataType> outputVarianceLocal = outVarianceQueue.AllocTensor<dataType>();

        AscendC::DeepNorm<dataType, isReuseSrc, isBasicBlock>(outputLocal,
            outputMeanLocal,
            outputVarianceLocal,
            inputXLocal,
            inputGxLocal,
            betaLocal,
            gammaLocal,
            alpha,
            epsilon,
            tiling);

        inQueueX.FreeTensor(inputXLocal);
        inQueueGx.FreeTensor(inputGxLocal);
        inQueueBeta.FreeTensor(betaLocal);
        inQueueGamma.FreeTensor(gammaLocal);
        outQueue.EnQue(outputLocal);
        outMeanQueue.EnQue(outputMeanLocal);
        outVarianceQueue.EnQue(outputVarianceLocal);
    }

    __aicore__ inline void CopyOut()
    {
        AscendC::LocalTensor<dataType> outputLocal = outQueue.DeQue<dataType>();
        AscendC::LocalTensor<dataType> outputMeanLocal = outMeanQueue.DeQue<dataType>();
        AscendC::LocalTensor<dataType> outputVarianceLocal = outVarianceQueue.DeQue<dataType>();
        AscendC::DataCopy(outputGlobal, outputLocal, bshLength);
        AscendC::DataCopy(outputMeanGlobal, outputMeanLocal, bsLength);
        AscendC::DataCopy(outputVarianceGlobal, outputVarianceLocal, bsLength);
        outQueue.FreeTensor(outputLocal);
        outMeanQueue.FreeTensor(outputMeanLocal);
        outVarianceQueue.FreeTensor(outputVarianceLocal);
    }

private:
    AscendC::GlobalTensor<dataType> inputXGlobal;
    AscendC::GlobalTensor<dataType> inputGxGlobal;
    AscendC::GlobalTensor<dataType> betaGlobal;
    AscendC::GlobalTensor<dataType> gammaGlobal;
    AscendC::GlobalTensor<dataType> outputGlobal;
    AscendC::GlobalTensor<dataType> outputMeanGlobal;
    AscendC::GlobalTensor<dataType> outputVarianceGlobal;

    AscendC::TPipe pipe;
    AscendC::TQue<AscendC::TPosition::VECIN, 1> inQueueX;
    AscendC::TQue<AscendC::TPosition::VECIN, 1> inQueueGx;
    AscendC::TQue<AscendC::TPosition::VECIN, 1> inQueueBeta;
    AscendC::TQue<AscendC::TPosition::VECIN, 1> inQueueGamma;
    AscendC::TQue<AscendC::TPosition::VECOUT, 1> outQueue;
    AscendC::TQue<AscendC::TPosition::VECOUT, 1> outMeanQueue;
    AscendC::TQue<AscendC::TPosition::VECOUT, 1> outVarianceQueue;

    DeepNormTiling tiling;
    uint32_t bshLength;
    uint32_t bsLength;
    uint32_t hLength;
    dataType alpha;
    dataType epsilon;
};

template <typename dataType, bool isReuseSrc = false, bool isBasicBlock = false>
__aicore__ inline void kernel_deepnorm_operator(GM_ADDR inputGm, GM_ADDR inputGxGm, GM_ADDR betaGm, GM_ADDR gammaGm,
    GM_ADDR outputGm, GM_ADDR outputMeanGm, GM_ADDR outputVarianceGm, GM_ADDR customTiling)
{
    GET_TILING_DATA(tilingData, customTiling)
    KernelDeepNorm<dataType, isReuseSrc, isBasicBlock> op;
    op.Init(inputGm, inputGxGm, betaGm, gammaGm, outputGm, outputMeanGm, outputVarianceGm, tilingData);
    op.Process();
}
```
