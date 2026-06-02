# Normalize

**页面ID:** atlasascendc_api_07_0810  
**来源:** https://www.hiascend.com/document/detail/zh/CANNCommunityEdition/850/API/ascendcopapi/atlasascendc_api_07_0810.html

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

LayerNorm中，已知均值和方差，计算shape为[A，R]的输入数据的标准差的倒数rstd和y，其计算公式如下：

<!-- img2text -->
```
            x - E
y = ─────────────────── × γ + β
        _____________
       ╱     2
     ╲╱  Var + ε

            1
rstd = ─────────────
         _____________
        ╱     2
      ╲╱  Var + ε
```

![](images/atlasascendc_api_07_0810_img_002.png)

其中，E和Var分别代表输入在R轴的均值，方差，γ为缩放系数，β为平移系数，ε为防除零的权重系数。

#### 函数原型

- 通过sharedTmpBuffer入参传入临时空间

```
template < typename U, typename T, bool isReuseSource = false, const NormalizeConfig& config = NLCFG_NORM>
__aicore__ inline void Normalize(const LocalTensor<T>& output, const LocalTensor<float>& outputRstd, const LocalTensor<float>& inputMean, const LocalTensor<float>& inputVariance, const LocalTensor<T>& inputX, const LocalTensor<U>& gamma, const LocalTensor<U>& beta, const LocalTensor<uint8_t>& sharedTmpBuffer, const float epsilon, const NormalizePara& para)
```

- 接口框架申请临时空间

```
template < typename U, typename T, bool isReuseSource = false, const NormalizeConfig& config = NLCFG_NORM>
__aicore__ inline void Normalize(const LocalTensor<T>& output, const LocalTensor<float>& outputRstd, const LocalTensor<float>& inputMean, const LocalTensor<float>& inputVariance, const LocalTensor<T>& inputX, const LocalTensor<U>& gamma, const LocalTensor<U>& beta, const float epsilon, const NormalizePara& para)
```

由于该接口的内部实现中涉及复杂的计算，需要额外的临时空间来存储计算过程中的中间变量。临时空间支持**接口框架申请**和开发者**通过sharedTmpBuffer入参传入**两种方式。

- 接口框架申请临时空间，开发者无需申请，但是需要预留临时空间的大小。

- 通过sharedTmpBuffer入参传入，使用该tensor作为临时空间进行处理，接口框架不再申请。该方式开发者可以自行管理sharedTmpBuffer内存空间，并在接口调用完成后，复用该部分内存，内存不会反复申请释放，灵活性较高，内存利用率也较高。

接口框架申请的方式，开发者需要预留临时空间；通过sharedTmpBuffer传入的情况，开发者需要为tensor申请空间。临时空间大小BufferSize的获取方式如下：通过Normalize Tiling中提供的GetNormalizeMaxMinTmpSize接口获取所需最大和最小临时空间大小，最小空间可以保证功能正确，最大空间用于提升性能。

#### 参数说明

**表1 **模板参数说明

| 参数名 | 描述 |
| --- | --- |
| U | beta，gamma操作数的数据类型。 Atlas A3 训练系列产品/Atlas A3 推理系列产品，支持的数据类型为: half、float。 Atlas A2 训练系列产品/Atlas A2 推理系列产品，支持的数据类型为: half、float。 Atlas 推理系列产品AI Core，支持的数据类型为: half、float。 |
| T | output，inputX操作数的数据类型。 Atlas A3 训练系列产品/Atlas A3 推理系列产品，支持的数据类型为: half、float。 Atlas A2 训练系列产品/Atlas A2 推理系列产品，支持的数据类型为: half、float。 Atlas 推理系列产品AI Core，支持的数据类型为: half、float。 |
| isReuseSource | 该参数预留，传入默认值false即可。 |
| 配置Normalize接口中输入输出相关信息。NormalizeConfig类型，定义如下。 ``` struct NormalizeConfig {     ReducePattern reducePattern = ReducePattern::AR;     int32_t aLength = -1;     bool isNoBeta = false;     bool isNoGamma = false;     bool isOnlyOutput = false; }; ```  - reducePattern：当前仅支持ReducePattern::AR模式，表示输入的内轴R轴为reduce计算轴。- aLength：用于描述输入的A轴大小。支持的取值如下：  - -1：默认值。取接口参数para中的aLength作为A轴大小。  - 其它值：该值需要与接口参数para中的aLength数值一致。 - isNoBeta：计算时，输入beta是否使用。  - false：默认值，Normalize计算中使用输入beta。  - true：Normalize计算中不使用输入beta。此时，公式中与beta相关的计算被省略。 - isNoGamma：可选输入gamma是否使用。  - false：默认值，Normalize计算中使用可选输入gamma。  - true：Normalize计算中不使用输入gamma。此时，公式中与gamma相关的计算被省略。 - isOnlyOutput：是否只输出y，不输出标准差的倒数rstd。当前该参数仅支持取值为false，表示y和rstd的结果全部输出。 |  |

**表2 **接口参数说明

| 参数名称 | 输入/输出 | 含义 |
| --- | --- | --- |
| output | 输出 | 目的操作数，shape为[A, R]，LocalTensor数据结构的定义请参考LocalTensor。 类型为LocalTensor，支持的TPosition为VECIN/VECCALC/VECOUT。 |
| outputRstd | 输出 | 标准差的倒数，shape为[A]，LocalTensor数据结构的定义请参考LocalTensor。 类型为LocalTensor，支持的TPosition为VECIN/VECCALC/VECOUT。 |
| inputMean | 输入 | 均值，shape为[A]，LocalTensor数据结构的定义请参考LocalTensor。 类型为LocalTensor，支持的TPosition为VECIN/VECCALC/VECOUT。 |
| inputVariance | 输入 | 方差，shape为[A]，LocalTensor数据结构的定义请参考LocalTensor。 类型为LocalTensor，支持的TPosition为VECIN/VECCALC/VECOUT。 |
| inputX | 输入 | 源操作数，shape为[A, R]，LocalTensor数据结构的定义请参考LocalTensor。inputX的数据类型需要与目的操作数保持一致，尾轴长度需要32B对齐。 类型为LocalTensor，支持的TPosition为VECIN/VECCALC/VECOUT。 |
| gamma | 输入 | 缩放系数，shape为[R]，LocalTensor数据结构的定义请参考LocalTensor。gamma的数据类型精度不低于源操作数的数据类型精度。 类型为LocalTensor，支持的TPosition为VECIN/VECCALC/VECOUT。 |
| beta | 输入 | 平移系数，shape为[R]，LocalTensor数据结构的定义请参考LocalTensor。beta的数据类型精度不低于源操作数的数据类型精度。 类型为LocalTensor，支持的TPosition为VECIN/VECCALC/VECOUT。 |
| sharedTmpBuffer | 输入 | 共享缓冲区，用于存放API内部计算产生的临时数据。该方式开发者可以自行管理sharedTmpBuffer内存空间，并在接口调用完成后，复用该部分内存，内存不会反复申请释放，灵活性较高，内存利用率也较高。共享缓冲区大小的获取方式请参考Normalize Tiling。 类型为LocalTensor，支持的TPosition为VECIN/VECCALC/VECOUT。 |
| epsilon | 输入 | 防除零的权重系数。 |
| Normalize计算所需的参数信息。NormalizePara类型，定义如下。 ``` struct NormalizePara {     uint32_t aLength;     uint32_t rLength;     uint32_t rLengthWithPadding; }; ```  - aLength：指定输入inputX的A轴长度。- rLength：指定输入inputX的R轴长度。- rLengthWithPadding：指定输入inputX的R轴对齐后的长度，该值是32B对齐的。 |  |  |

#### 约束说明

- 缩放系数gamma和平移系数beta的数据类型精度必须不低于源操作数inputX的数据类型精度。比如，inputX的数据类型是half，gamma、beta的数据类型可以是half或者float，精度不低于inputX。
- src和dst的Tensor空间不可以复用。
- 输入仅支持ND格式。
- R轴不支持切分。

#### 调用示例

更多调用样例请参考[normalize算子样例](https://gitee.com/ascend/ascendc-api-adv/tree/master/examples/normalization/normalize)。

```
#include "kernel_operator.h"
constexpr int32_t BUFFER_NUM = 1;  // tensor num for each queue

template <const AscendC::NormalizeConfig& CONFIG>
class KernelNormalize {
 public:
  __aicore__ inline KernelNormalize() {}

  __aicore__ inline void Init(GM_ADDR x, GM_ADDR mean, GM_ADDR variance, GM_ADDR gamma, GM_ADDR beta, GM_ADDR rstd, GM_ADDR y, const float epsilon, const AscendC::NormalizePara& para) {
    this->meanRstdSize = (para.aLength + 7) / 8 * 8;  // 此时进行32B对齐处理
    // get start index for current core, core parallel
    xGm.SetGlobalBuffer((__gm__ DTYPE_X*)x, para.aLength * para.rLengthWithPadding);
    meanGm.SetGlobalBuffer((__gm__ float*)mean, this->meanRstdSize);
    varianceGm.SetGlobalBuffer((__gm__ float*)variance, this->meanRstdSize);
    gammaGm.SetGlobalBuffer((__gm__ DTYPE_GAMMA*)gamma, para.rLengthWithPadding);
    betaGm.SetGlobalBuffer((__gm__ DTYPE_BETA*)beta, para.rLengthWithPadding);

    rstdGm.SetGlobalBuffer((__gm__ float*)rstd, this->meanRstdSize);
    yGm.SetGlobalBuffer((__gm__ DTYPE_Y*)y, para.aLength * para.rLengthWithPadding);

    // pipe alloc memory to queue, the unit is Bytes
    pipe.InitBuffer(inQueueX, BUFFER_NUM, para.aLength * para.rLengthWithPadding * sizeof(DTYPE_X));
    pipe.InitBuffer(inQueueMean, BUFFER_NUM, this->meanRstdSize * sizeof(float));
    pipe.InitBuffer(inQueueVariance, BUFFER_NUM, this->meanRstdSize * sizeof(float));
    pipe.InitBuffer(inQueueGamma, BUFFER_NUM, para.rLengthWithPadding * sizeof(DTYPE_GAMMA));
    pipe.InitBuffer(inQueueBeta, BUFFER_NUM, para.rLengthWithPadding * sizeof(DTYPE_BETA));

    pipe.InitBuffer(outQueueRstd, BUFFER_NUM, this->meanRstdSize * sizeof(float));
    pipe.InitBuffer(outQueueY, BUFFER_NUM, para.aLength * para.rLengthWithPadding * sizeof(DTYPE_Y));

    this->epsilon = epsilon;
    this->para = para;
  }

  __aicore__ inline void Compute() {
    AscendC::LocalTensor<DTYPE_X> xLocal = inQueueX.DeQue<DTYPE_X>();
    AscendC::LocalTensor<float> meanLocal = inQueueMean.DeQue<float>();
    AscendC::LocalTensor<float> varianceLocal = inQueueVariance.DeQue<float>();
    AscendC::LocalTensor<DTYPE_GAMMA> gammaLocal = inQueueGamma.DeQue<DTYPE_GAMMA>();
    AscendC::LocalTensor<DTYPE_BETA> betaLocal = inQueueBeta.DeQue<DTYPE_BETA>();

    AscendC::LocalTensor<float> rstdLocal = outQueueRstd.AllocTensor<float>();
    AscendC::LocalTensor<DTYPE_Y> yLocal = outQueueY.AllocTensor<DTYPE_Y>();

    AscendC::Duplicate(rstdLocal, (float)0, this->meanRstdSize);
    AscendC::Duplicate(yLocal, (DTYPE_Y)0, para.aLength * para.rLengthWithPadding);

    AscendC::Normalize<DTYPE_Y, DTYPE_X, false, CONFIG>(yLocal, rstdLocal, meanLocal, varianceLocal, xLocal, gammaLocal, betaLocal, epsilon, para);

    outQueueRstd.EnQue<float>(rstdLocal);
    outQueueY.EnQue<DTYPE_Y>(yLocal);
    inQueueX.FreeTensor(xLocal);
    inQueueMean.FreeTensor(meanLocal);
    inQueueVariance.FreeTensor(varianceLocal);
    inQueueGamma.FreeTensor(gammaLocal);
    inQueueBeta.FreeTensor(betaLocal);

  }
  __aicore__ inline void Process() {
    CopyIn();
    Compute();
    CopyOut();
  }

 private:
  __aicore__ inline void CopyIn() {
    // alloc tensor from queue memory
    AscendC::LocalTensor<DTYPE_X> xLocal = inQueueX.AllocTensor<DTYPE_X>();
    AscendC::LocalTensor<float> meanLocal = inQueueMean.AllocTensor<float>();
    AscendC::LocalTensor<float> varianceLocal = inQueueVariance.AllocTensor<float>();
    AscendC::LocalTensor<DTYPE_GAMMA> gammaLocal = inQueueGamma.AllocTensor<DTYPE_GAMMA>();
    AscendC::LocalTensor<DTYPE_BETA> betaLocal = inQueueBeta.AllocTensor<DTYPE_BETA>();
    // copy progress_th tile from global tensor to local tensor
    AscendC::DataCopy(xLocal, xGm, para.aLength * para.rLengthWithPadding);
    AscendC::DataCopy(meanLocal, meanGm, this->meanRstdSize);
    AscendC::DataCopy(varianceLocal, varianceGm, this->meanRstdSize);
    AscendC::DataCopy(gammaLocal, gammaGm, para.rLengthWithPadding);
    AscendC::DataCopy(betaLocal, betaGm, para.rLengthWithPadding);

    // enque input tensors to VECIN queue
    inQueueX.EnQue(xLocal);
    inQueueMean.EnQue(meanLocal);
    inQueueVariance.EnQue(varianceLocal);
    inQueueGamma.EnQue(gammaLocal);
    inQueueBeta.EnQue(betaLocal);
  }

  __aicore__ inline void CopyOut() {
    // deque output tensor from VECOUT queue
    AscendC::LocalTensor<float> rstdLocal = outQueueRstd.DeQue<float>();
    AscendC::LocalTensor<DTYPE_Y> yLocal = outQueueY.DeQue<DTYPE_Y>();
    // copy progress_th tile from local tensor to global tensor
    AscendC::DataCopy(rstdGm, rstdLocal, this->meanRstdSize);
    AscendC::DataCopy(yGm, yLocal, para.aLength * para.rLengthWithPadding);
    // free output tensor for reuse
    outQueueRstd.FreeTensor(rstdLocal);
    outQueueY.FreeTensor(yLocal);
  }

 private:
  AscendC::TPipe pipe;
  // create queues for input, in this case depth is equal to buffer num
  AscendC::TQue<AscendC::TPosition::VECIN, BUFFER_NUM> inQueueX;
  AscendC::TQue<AscendC::TPosition::VECIN, BUFFER_NUM> inQueueMean;
  AscendC::TQue<AscendC::TPosition::VECIN, BUFFER_NUM> inQueueVariance;
  AscendC::TQue<AscendC::TPosition::VECIN, BUFFER_NUM> inQueueGamma;
  AscendC::TQue<AscendC::TPosition::VECIN, BUFFER_NUM> inQueueBeta;
  // create queue for output, in this case depth is equal to buffer num
  AscendC::TQue<AscendC::TPosition::VECOUT, BUFFER_NUM> outQueueRstd;
  AscendC::TQue<AscendC::TPosition::VECOUT, BUFFER_NUM> outQueueY;

  AscendC::GlobalTensor<float> meanGm;
  AscendC::GlobalTensor<float> varianceGm;
  AscendC::GlobalTensor<DTYPE_X> xGm;
  AscendC::GlobalTensor<DTYPE_GAMMA> gammaGm;
  AscendC::GlobalTensor<DTYPE_BETA> betaGm;

  AscendC::GlobalTensor<float> rstdGm;
  AscendC::GlobalTensor<DTYPE_Y> yGm;

  float epsilon;
  uint32_t meanRstdSize;
  AscendC::NormalizePara para;
};
__aicore__ constexpr AscendC::NormalizeConfig GenConfig(bool isNoBeta, bool isNoGamma)
{
    return {.reducePattern = AscendC::ReducePattern::AR,
        .aLength = -1,
        .isNoBeta = isNoBeta,
        .isNoGamma = isNoGamma,
        .isOnlyOutput = false};
}
// with beta and gamma
constexpr AscendC::NormalizeConfig CONFIG1 = GenConfig(false, false);
constexpr AscendC::NormalizeConfig CONFIG2 = GenConfig(false, true);
constexpr AscendC::NormalizeConfig CONFIG3 = GenConfig(true, false);
constexpr AscendC::NormalizeConfig CONFIG4 = GenConfig(true, true);

extern "C" __global__ __aicore__ void normalize_custom(GM_ADDR x, GM_ADDR mean, GM_ADDR variance, GM_ADDR gamma, GM_ADDR beta, GM_ADDR rstd, GM_ADDR y, GM_ADDR workspace, GM_ADDR tiling) {
    GET_TILING_DATA(tilingData, tiling);
    float epsilon = tilingData.epsilon;
    AscendC::NormalizePara para(tilingData.aLength, tilingData.rLength, tilingData.rLengthWithPadding);
    if (TILING_KEY_IS(1)) {
      if (!tilingData.isNoBeta && !tilingData.isNoGamma) {
          KernelNormalize<CONFIG1> op;
          op.Init(x, mean, variance, gamma, beta, rstd, y, epsilon, para);
          op.Process();
      } else if (!tilingData.isNoBeta && tilingData.isNoGamma) {
          KernelNormalize<CONFIG2> op;
          op.Init(x, mean, variance, gamma, beta, rstd, y, epsilon, para);
          op.Process();
      } else if (tilingData.isNoBeta && !tilingData.isNoGamma) {
          KernelNormalize<CONFIG3> op;
          op.Init(x, mean, variance, gamma, beta, rstd, y, epsilon, para);
          op.Process();
      } else if (tilingData.isNoBeta && tilingData.isNoGamma) {
          KernelNormalize<CONFIG4> op;
          op.Init(x, mean, variance, gamma, beta, rstd, y, epsilon, para);
          op.Process();
      }
    }
  }
```
