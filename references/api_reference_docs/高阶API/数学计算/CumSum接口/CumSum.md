# CumSum

**页面ID:** atlasascendc_api_07_0605  
**来源:** https://www.hiascend.com/document/detail/zh/CANNCommunityEdition/850/API/ascendcopapi/atlasascendc_api_07_0605.html

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

用于对输入张量按行或列进行累加和操作，输出结果中每个元素都是输入张量中对应位置及之前所有行或列的元素累加和。

计算公式如下：

![](images/atlasascendc_api_07_0605_img_001.png)

- 逐行累加算法

  - First轴处理，按行累加和操作，即第一行不变，后面的行依次累加，输出结果的第i行第j列计算公式如下：

![](images/atlasascendc_api_07_0605_img_002.png)

以tensor([[0, 1, 2], [3, 4, 5]])为例，输出结果是tensor([[0, 1, 2], [3, 5, 7]])

  - Last轴处理，按列累加和操作，即第一列不变，后面的列依次累加，输出结果的第i行第j列计算公式如下：

<!-- img2text -->
```
tensor([[0, 1, 2],
        [3, 4, 5]])
```

说明:
- 图中仅可识别出一个 2×3 的张量示意：
  - 第1行: 0, 1, 2
  - 第2行: 3, 4, 5
- 其余内容在图片中无法清晰辨认。

以tensor([[0, 1, 2], [3, 4, 5]])为例，输出结果是tensor([[0, 1, 3], [3, 7, 12]])

#### 函数原型

- 通过sharedTmpBuffer入参传入临时空间

```
template <typename T, const CumSumConfig& config = defaultCumSumConfig>
__aicore__ inline void CumSum(LocalTensor<T>& dstTensor, LocalTensor<T>& lastRowTensor, const LocalTensor<T>& srcTensor, LocalTensor<uint8_t>& sharedTmpBuffer, const CumSumInfo& cumSumInfo)
```

- 接口框架申请临时空间

```
template <typename T, const CumSumConfig& config = defaultCumSumConfig>
__aicore__ inline void CumSum(LocalTensor<T>& dstTensor, LocalTensor<T>& lastRowTensor, const LocalTensor<T>& srcTensor, const CumSumInfo& cumSumInfo)
```

由于该接口的内部实现中涉及精度转换。需要额外的临时空间来存储计算过程中的中间变量。临时空间支持**接口框架申请**和开发者**通过sharedTmpBuffer入参传入**两种方式。

- 接口框架申请临时空间，开发者无需申请，但是需要预留临时空间的大小。

- 通过sharedTmpBuffer入参传入，使用该tensor作为临时空间进行处理，接口框架不再申请。该方式开发者可以自行管理sharedTmpBuffer内存空间，并在接口调用完成后，复用该部分内存，内存不会反复申请释放，灵活性较高，内存利用率也较高。

接口框架申请的方式，开发者需要预留临时空间；通过sharedTmpBuffer传入的情况，开发者需要为tensor申请空间。临时空间大小BufferSize的获取方式如下：通过GetCumSumMaxMinTmpSize中提供的接口获取需要预留空间的大小。

#### 参数说明

**表1 **模板参数说明

| 参数名 | 描述 |
| --- | --- |
| T | 操作数的数据类型。 Atlas A3 训练系列产品/Atlas A3 推理系列产品，支持的数据类型为：half、float。 Atlas A2 训练系列产品/Atlas A2 推理系列产品，支持的数据类型为：half、float。 Atlas 推理系列产品AI Core，支持的数据类型为：half、float。 |
| 定义CumSum接口编译时config参数。 ``` struct CumSumConfig {     bool isLastAxis{true};     bool isReuseSource{false};     bool outputLastRow{false}; }; ```  - isLastAxis：取值为true表示计算按last轴处理，取值为false表示计算按first轴处理；- isReuseSource：是否可以复用srcTensor的内存空间；该参数预留，传入默认值false即可。- outputLastRow：是否输出最后一行数据。 |  |

**表2 **接口参数说明

| 参数名 | 输入/输出 | 描述 |
| --- | --- | --- |
| dstTensor | 输出 | 目的操作数。按first轴或last轴处理，输入元素的累加和。 类型为LocalTensor，支持的TPosition为VECIN/VECCALC/VECOUT。 |
| lastRowTensor | 输出 | 目的操作数。模板参数config中的outputLastRow参数取值为true时，输出的最后一行数据。 类型为LocalTensor，支持的TPosition为VECIN/VECCALC/VECOUT。 |
| srcTensor | 输入 | 源操作数。 类型为LocalTensor，支持的TPosition为VECIN/VECCALC/VECOUT。 |
| sharedTmpBuffer | 输入 | 临时缓存。 类型为LocalTensor，支持的TPosition为VECIN/VECCALC/VECOUT。 用于CumSum内部复杂计算时存储中间变量，由开发者提供。 临时空间大小BufferSize的获取方式请参考GetCumSumMaxMinTmpSize。 |
| srcTensor的shape信息。CumSumInfo类型，具体定义如下： ``` struct CumSumInfo {     uint32_t outter{0};    // 表示输入数据的外轴长度     uint32_t inner{0};     // 表示输入数据的内轴长度 }; ```  注意： - cumSumInfo.outter和cumSumInfo.inner都应大于0。- cumSumInfo.outter * cumSumInfo.inner不能大于dstTensor或srcTensor的大小。- cumSumInfo.inner * sizeof(T)必须是32字节的整数倍。- 当模板参数config中的outputLastRow取值为true时，cumSumInfo.inner不能大于lastRowTensor输出的最后一行数据的大小。 |  |  |

#### 约束说明

- 输入input只支持二维结构。
- cumSumInfo.inner * sizeof(T)必须是32字节的整数倍。

#### 调用示例

```
#include "kernel_operator.h"

template <typename T, const CumSumConfig& CONFIG>
class KernelCumSum
{
public:
    __aicore__ inline KernelCumSum(){}
    __aicore__ inline void Init(
        GM_ADDR srcGm, GM_ADDR dstGm, GM_ADDR lastRowGm, const AscendC::CumSumInfo& cumSumParams)
    {
        outer = cumSumParams.outter;
        inner = cumSumParams.inner;
        srcGlobal.SetGlobalBuffer(reinterpret_cast<__gm__ T *>(srcGm), outer * inner);
        dstGlobal.SetGlobalBuffer(reinterpret_cast<__gm__ T *>(dstGm), outer * inner);
        lastRowGlobal.SetGlobalBuffer(reinterpret_cast<__gm__ T *>(lastRowGm), inner);
        pipe.InitBuffer(inQueueX, 1, outer * inner * sizeof(T));
        pipe.InitBuffer(outQueue, 1, outer * inner * sizeof(T));
        pipe.InitBuffer(lastRowQueue, 1, inner * sizeof(T));
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
        AscendC::LocalTensor<T> srcLocal = inQueueX.AllocTensor<T>();
        AscendC::DataCopy(srcLocal, srcGlobal, outer * inner);
        inQueueX.EnQue(srcLocal);
    }
    __aicore__ inline void Compute()
    {
        AscendC::LocalTensor<T> dstLocal = outQueue.AllocTensor<T>();
        AscendC::LocalTensor<T> lastRowLocal = lastRowQueue.AllocTensor<T>();
        AscendC::LocalTensor<T> srcLocal = inQueueX.DeQue<T>();

        const AscendC::CumSumInfo cumSumInfo{outer, inner};
        AscendC::CumSum<T, CONFIG>(dstLocal, lastRowLocal, srcLocal, cumSumInfo);
        outQueue.EnQue<T>(dstLocal);
        lastRowQueue.EnQue<T>(lastRowLocal);
        inQueueX.FreeTensor(srcLocal);
    }
    __aicore__ inline void CopyOut()
    {
        AscendC::LocalTensor<T> dstLocal = outQueue.DeQue<T>();
        AscendC::DataCopy(dstGlobal, dstLocal, outer * inner);
        outQueue.FreeTensor(dstLocal);
        AscendC::LocalTensor<T> lastRowLocal = lastRowQueue.DeQue<T>();
        AscendC::DataCopy(lastRowGlobal, lastRowLocal, inner);
        lastRowQueue.FreeTensor(lastRowLocal);
    }

private:
    AscendC::GlobalTensor<T> srcGlobal;
    AscendC::GlobalTensor<T> dstGlobal;
    AscendC::GlobalTensor<T> lastRowGlobal;
    AscendC::TPipe pipe;
    AscendC::TQue<AscendC::TPosition::VECIN, 1> inQueueX;
    AscendC::TQue<AscendC::TPosition::VECOUT, 1> outQueue;
    AscendC::TQue<AscendC::TPosition::VECOUT, 1> lastRowQueue;
    uint32_t outer{1};
    uint32_t inner{1};
};

constexpr AscendC::CumSumConfig cumSumConfig{true, false, true};

template <typename T>
__aicore__ inline void kernel_cumsum_operator(
    GM_ADDR srcGm, GM_ADDR dstGm, GM_ADDR lastRowGm, const AscendC::CumSumInfo &cumSumParams)
{
    KernelCumSum<T, cumSumConfig> op;
    op.Init(srcGm, dstGm, lastRowGm, cumSumParams);
    op.Process();
}
```
