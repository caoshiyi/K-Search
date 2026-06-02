# WelfordUpdate

**页面ID:** atlasascendc_api_07_0812  
**来源:** https://www.hiascend.com/document/detail/zh/CANNCommunityEdition/850/API/ascendcopapi/atlasascendc_api_07_0812.html

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

Welford是一种在线计算均值和方差的方法。一方面，它可以在不存储所有样本的情况下，逐步计算所有样本的均值和方差，更适合处理海量数据；另一方面，它只需要对数据进行一次遍历，能减少访存次数，提高计算性能。本接口为Welford算法的前处理。

LayerNorm算法中Reduce轴较大的场景，可以通过切分Reduce轴，联合使用本接口与WelfordFinalize，实现等效计算LayerNorm。

如下图所示，切分数据的Reduce轴，假设切分后每块数据的形状为[1, k]，每块数据标号为1，2，3，…，n。

**图1 **Reduce轴切分示意图
<!-- img2text -->
```
                    Reduce轴

┌────────────────┬────────────────┬────────────────┬────────────────┐
│       1        │       2        │       ...      │       n        │
└────────────────┴────────────────┴────────────────┴────────────────┘
```

本接口的计算公式如下。进行上述的数据切分后，分n次调用本接口，切分后的每块数据均完成如下公式的计算。

<!-- img2text -->
```
               x_i - Meant_{i-1}
Meant_i = Meant_{i-1} + ─────────────
                      i
```

<!-- img2text -->
```
Mi = Mi-1 + (xi - Meanti-1)(xi - Meanti)
```

上式中，xi、Meanti、Mi的形状均为[1, k]，xi表示切分后的第i块数据，Meanti表示第i次调用本接口得到的前i块数据的均值，Mi表示第i次调用本接口得到的前i块数据的方差中间结果（即为求方差而保存的中间计算结果，本节后续内容中写作方差中间结果）。其中，第一次调用本接口，即i=1时，公式中的Meant0和M0由用户定义为形状[1, k]、取值全0的数据。

Meantn的计算过程示意如下图，调用n次本接口后，得到形状为[1, k]的Meantn和Mn，Meantn和Mn用于后续WelfordFinalize接口的计算。

**图2 **均值Meantn计算过程示意图
<!-- img2text -->
```text
Xi                               Meanti

  Xi                                  Meanti
  │                                     │
┌───────────────┐                  ┌───────────────┐
│      x1       │ ───────────────→ │    Meant1     │
└───────────────┘                  └───────────────┘

┌───────────────┐                  ┌───────────────┐
│      x2       │ ───────────────→ │    Meant2     │
└───────────────┘                  └───────────────┘

       ...                                  ...
                                            │
                                            ↓

┌───────────────┐                  ┌───────────────┐
│     xn-1      │ ───────────────→ │   Meantn-1    │
└───────────────┘                  └───────────────┘
                                            │
                                            ↓

┌───────────────┐                  ┌───────────────┐
│      xn       │ ───────────────→ │    Meantn     │
└───────────────┘                  └───────────────┘
```

#### 函数原型

- 通过sharedTmpBuffer入参传入临时空间

```
template <typename T, typename U,bool isReuseSource = false, const WelfordUpdateConfig& config = WFUPDATE_DEFAULT_CFG>
__aicore__ inline void WelfordUpdate(const LocalTensor<U>& outputMean, const LocalTensor<U>& outputVariance, const LocalTensor<U>& inputMean, const LocalTensor<U>& inputVariance, const LocalTensor<T>& inputX, const LocalTensor<uint8_t>& sharedTmpBuffer, const WelfordUpdateParam& para)
```

- 接口框架申请临时空间

```
template <typename T, typename U,bool isReuseSource = false, const WelfordUpdateConfig& config = WFUPDATE_DEFAULT_CFG>
__aicore__ inline void WelfordUpdate(const LocalTensor<U>& outputMean, const LocalTensor<U>& outputVariance, const LocalTensor<U>& inputMean, const LocalTensor<U>& inputVariance, const LocalTensor<T>& inputX, const WelfordUpdateParam& para)
```

由于该接口的内部实现中涉及复杂的计算，需要额外的临时空间来存储计算过程中的中间变量。临时空间支持**接口框架申请**和开发者**通过sharedTmpBuffer入参传入**两种方式。

- 接口框架申请临时空间，开发者无需申请，但是需要预留临时空间的大小。

- 通过sharedTmpBuffer入参传入，使用该tensor作为临时空间进行处理，接口框架不再申请。该方式开发者可以自行管理sharedTmpBuffer内存空间，并在接口调用完成后，复用该部分内存，内存不会反复申请释放，灵活性较高，内存利用率也较高。

接口框架申请的方式，开发者需要预留临时空间；通过sharedTmpBuffer传入的情况，开发者需要为tensor申请空间。临时空间大小BufferSize的获取方式如下：通过WelfordUpdate Tiling中提供的GetWelfordUpdateMaxMinTmpSize接口获取所需最大和最小临时空间大小，最小空间可以保证功能正确，最大空间用于提升性能。

#### 参数说明

**表1 **模板参数说明

| 参数名 | 描述 |
| --- | --- |
| T | inputX操作数的数据类型。 Atlas A3 训练系列产品/Atlas A3 推理系列产品，支持的数据类型为：half、float。 Atlas A2 训练系列产品/Atlas A2 推理系列产品，支持的数据类型为：half、float。 Atlas 推理系列产品AI Core，支持的数据类型为：half、float。 |
| U | outputMean、outputVariance、inputMean、inputVariance操作数的数据类型。 Atlas A3 训练系列产品/Atlas A3 推理系列产品，支持的数据类型为：float Atlas A2 训练系列产品/Atlas A2 推理系列产品，支持的数据类型为：float Atlas 推理系列产品AI Core，支持的数据类型为：float |
| isReuseSource | 是否允许修改源操作数，默认值为false。如果开发者允许源操作数被改写，可以使能该参数，使能后能够节省部分内存空间。 设置为**true**，则本接口内部计算时**复用**inputX的内存空间，节省内存空间；设置为**false**，则本接口内部计算时**不复用**inputX的内存空间。 在Atlas 推理系列产品AI Core中，该参数预留，传入默认值false即可。 isReuseSource的使用样例请参考更多样例。 |
| 配置非指定计算范围内的目的操作数与源操作数的复用关系。WelfordUpdateConfig类型，定义如下： ``` struct WelfordUpdateConfig {     bool isInplace = false; // 目的操作数是否复用源操作数。 }; ```  - isInplace：接口参数para中的abComputeLength参数指定了输入数据内层轴的计算长度，在该指定计算长度之外的输出数据具体为何值，通过本参数设置。本参数表示，在指定计算长度之外的目的操作数是否复用源操作数；若复用，对于指定计算长度之外的输出，直接使用对应位置的源操作数代替输出目的操作数；若不复用，则本接口不会输出计算范围外的目的操作数。  - false：默认值。表示目的操作数不复用源操作数。  - true：表示目的操作数复用源操作数。outputMean复用inputMean，outputVariance复用inputVariance。  配置示例如下： ``` constexpr WelfordUpdateConfig WFUPDATE_DEFAULT_CFG = {false}; ```  此参数一般用于配合kernel侧tiling计算的接口使用。 |  |

**表2 **接口参数说明

| 参数名 | 输入/输出 | 描述 |
| --- | --- | --- |
| outputMean | 输出 | 均值目的操作数，对应接口公式中的Meanti。 类型为LocalTensor，支持的TPosition为VECIN/VECCALC/VECOUT。 shape和源操作数inputMean需要保持一致。 |
| outputVariance | 输出 | 方差中间结果目的操作数，对应接口公式中的Mi。 类型为LocalTensor，支持的TPosition为VECIN/VECCALC/VECOUT。 shape和源操作数inputVariance需要保持一致。 |
| inputMean | 输入 | 均值源操作数，对应接口公式中的Meanti-1。 类型为LocalTensor，支持的TPosition为VECIN/VECCALC/VECOUT。 |
| inputVariance | 输入 | 方差中间结果源操作数，对应接口公式中的Mi-1。 类型为LocalTensor，支持的TPosition为VECIN/VECCALC/VECOUT。 |
| inputX | 输入 | 源操作数，对应接口公式中的xi。 类型为LocalTensor，支持的TPosition为VECIN/VECCALC/VECOUT。 |
| sharedTmpBuffer | 输入 | 临时空间。 类型为LocalTensor，支持的TPosition为VECIN/VECCALC/VECOUT。 接口内部复杂计算时用于存储中间变量，由开发者提供。 临时空间大小BufferSize的获取方式请参考WelfordUpdate Tiling。 |
| 计算所需的参数信息。WelfordUpdateParam类型，定义如下。 ``` struct WelfordUpdateParam {     uint32_t rnLength;      uint32_t abLength;      uint32_t abComputeLength;      float nRec; }; ```  - rnLength：预留参数，固定设置为1。- abLength：Reduce轴拆分的大小。- abComputeLength：从输入的起始地址开始的Reduce轴实际计算长度。- nRec：取值为1/i，i为当前调用本接口的累积次数。i的取值范围为[1, n]，n为对输入数据inputX的Reduce轴切分的块数。 各目的操作数和源操作数的shape均为[rnLength, abLength]。 |  |  |

#### 约束说明

- 接口参数para.rnLength当前只支持取值为1；
- 接口参数para.abLength的取值必须为32/sizeof(T)的整数倍；
- 接口参数para.abComputeLength的取值必须大于0。
- 不支持源操作数与目的操作数地址重叠**。**
- 不支持sharedTmpBuffer与源操作数和目的操作数地址重叠。

#### 调用示例

完整的算子样例请参考[welford_update算子样例](https://gitee.com/ascend/ascendc-api-adv/tree/master/examples/normalization/welford_update)。

```
#include "kernel_operator.h"

constexpr AscendC::WelfordUpdateConfig WELFORD_UPDATE_ENABLE_INPLACE_CFG = { true };
constexpr AscendC::WelfordUpdateConfig WELFORD_UPDATE_UNENABLE_INPLACE_CFG = { false };

template <typename dataType, typename dataTypeU, bool isInplace = false> class KernelWelfordUpdate {
public:
    __aicore__ inline KernelWelfordUpdate() {}
    __aicore__ inline void Init(GM_ADDR inputX_gm, GM_ADDR inputmean_gm, GM_ADDR inputvar_gm, GM_ADDR outputMean_gm,
        GM_ADDR outputVariance_gm, uint32_t nLength, uint32_t rLength, uint32_t abComputeLength)
    {
        this->nLength = nLength;
        this->rLength = rLength;
        this->abComputeLength = abComputeLength;
        totalLength = nLength * rLength;

        inputX_global.SetGlobalBuffer(reinterpret_cast<__gm__ dataType *>(inputX_gm), totalLength);
        inputmean_global.SetGlobalBuffer(reinterpret_cast<__gm__ dataTypeU *>(inputmean_gm), totalLength);
        inputvar_global.SetGlobalBuffer(reinterpret_cast<__gm__ dataTypeU *>(inputvar_gm), totalLength);

        outputMean_global.SetGlobalBuffer(reinterpret_cast<__gm__ dataTypeU *>(outputMean_gm), totalLength);
        outputVariance_global.SetGlobalBuffer(reinterpret_cast<__gm__ dataTypeU *>(outputVariance_gm), totalLength);

        pipe.InitBuffer(inQueueX, 1, sizeof(dataType) * totalLength);
        pipe.InitBuffer(inQueueMean, 1, sizeof(dataTypeU) * totalLength);
        pipe.InitBuffer(inQueueVar, 1, sizeof(dataTypeU) * totalLength);
        pipe.InitBuffer(outQueueMean, 1, sizeof(dataTypeU) * totalLength);
        pipe.InitBuffer(outQueueVariance, 1, sizeof(dataTypeU) * totalLength);
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
        AscendC::LocalTensor<dataTypeU> inmeanLocal = inQueueMean.AllocTensor<dataTypeU>();
        AscendC::LocalTensor<dataTypeU> invarLocal = inQueueVar.AllocTensor<dataTypeU>();

        AscendC::DataCopy(inputXLocal, inputX_global, totalLength);
        AscendC::DataCopy(inmeanLocal, inputmean_global, totalLength);
        AscendC::DataCopy(invarLocal, inputvar_global, totalLength);

        inQueueX.EnQue(inputXLocal);
        inQueueMean.EnQue(inmeanLocal);
        inQueueVar.EnQue(invarLocal);
    }
    __aicore__ inline void Compute()
    {
        AscendC::LocalTensor<dataType> inputXLocal = inQueueX.DeQue<dataType>();
        AscendC::LocalTensor<dataTypeU> inmeanLocal = inQueueMean.DeQue<dataTypeU>();
        AscendC::LocalTensor<dataTypeU> invarLocal = inQueueVar.DeQue<dataTypeU>();

        AscendC::LocalTensor<dataTypeU> meanLocal = outQueueMean.AllocTensor<dataTypeU>();
        AscendC::LocalTensor<dataTypeU> varianceLocal = outQueueVariance.AllocTensor<dataTypeU>();

        struct AscendC::WelfordUpdateParam para = { nLength, rLength, abComputeLength, 0.3 };
        if constexpr (isInplace) {
            AscendC::WelfordUpdate<dataType, dataTypeU, false, WELFORD_UPDATE_ENABLE_INPLACE_CFG>(meanLocal, varianceLocal,
                inmeanLocal, invarLocal, inputXLocal, para);
        } else {
            AscendC::WelfordUpdate<dataType, dataTypeU, false, WELFORD_UPDATE_UNENABLE_INPLACE_CFG>(meanLocal, varianceLocal,
                inmeanLocal, invarLocal, inputXLocal, para);
        }

        outQueueMean.EnQue<dataTypeU>(meanLocal);
        outQueueVariance.EnQue<dataTypeU>(varianceLocal);

        inQueueX.FreeTensor(inputXLocal);
        inQueueMean.FreeTensor(inmeanLocal);
        inQueueVar.FreeTensor(invarLocal);
    }
    __aicore__ inline void CopyOut()
    {
        AscendC::LocalTensor<dataTypeU> meanLocal = outQueueMean.DeQue<dataTypeU>();
        AscendC::LocalTensor<dataTypeU> varianceLocal = outQueueVariance.DeQue<dataTypeU>();

        AscendC::DataCopy(outputMean_global, meanLocal, totalLength);
        AscendC::DataCopy(outputVariance_global, varianceLocal, totalLength);

        outQueueMean.FreeTensor(meanLocal);
        outQueueVariance.FreeTensor(varianceLocal);
    }

private:
    AscendC::GlobalTensor<dataType> inputX_global;
    AscendC::GlobalTensor<dataTypeU> inputmean_global;
    AscendC::GlobalTensor<dataTypeU> inputvar_global;
    AscendC::GlobalTensor<dataTypeU> outputMean_global;
    AscendC::GlobalTensor<dataTypeU> outputVariance_global;

    AscendC::TPipe pipe;
    AscendC::TQue<AscendC::TPosition::VECIN, 1> inQueueX;
    AscendC::TQue<AscendC::TPosition::VECIN, 1> inQueueMean;
    AscendC::TQue<AscendC::TPosition::VECIN, 1> inQueueVar;
    AscendC::TQue<AscendC::TPosition::VECOUT, 1> outQueueMean;
    AscendC::TQue<AscendC::TPosition::VECOUT, 1> outQueueVariance;

    uint32_t nLength;
    uint32_t rLength;
    uint32_t abComputeLength;
    uint32_t totalLength;
};
```
