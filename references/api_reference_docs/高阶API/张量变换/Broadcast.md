# Broadcast

**页面ID:** atlasascendc_api_07_0853  
**来源:** https://www.hiascend.com/document/detail/zh/CANNCommunityEdition/850/API/ascendcopapi/atlasascendc_api_07_0853.html

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

将输入按照输出shape进行广播。

比如A的shape为(2,1)，广播的目标shape为(2,16)，则会将原来的一列扩展为相同的16列。

```
输入数据： 
[[ 1]
 [ 2]]
输出数据： 
[[ 1  1  1  1  1  1  1  1  1  1  1  1  1  1  1  1]
 [ 2  2  2  2  2  2  2  2  2  2  2  2  2  2  2  2]]
```

#### 实现原理

以float类型，ND格式，[m, 1]广播到[m, k]为例，描述Broadcast高阶API内部算法框图，如下图所示。

**图1 **Broadcast算法框图
<!-- img2text -->
```
                 ┌─────────┐                                    ┌─────────┐
                 │ x[m,1]  │                                    │ x[m,1]  │
                 └────┬────┘                                    └────┬────┘
                      ↓                                                ↓
      ┌──────────────────────────────────────┐        ┌──────────────────────────────────────┐
      │ k对齐场景                            │        │ k非对齐场景                          │
      │                                      │        │                                      │
      │          ┌────────────────┐          │        │          ┌────────────────┐          │
      │          │      Brcb      │          │        │          │      Brcb      │          │
      │          │  ([m,1]->[m,8])│          │        │          │  ([m,1]->[m,8])│          │
      │          └───────┬────────┘          │        │          └───────┬────────┘          │
      │                  ↓                   │        │                  ↓                   │
      │          ┌────────────────┐          │        │          ┌────────────────┐          │
      │          │      Copy      │          │        │          │      Copy      │          │
      │          │  ([m,8]->[m,k])│          │        │          │  ([m,8]->[m,k'])│         │
      │          └───────┬────────┘          │        │          └───────┬────────┘          │
      │                  ↓                   │        │                  ↓                   │
      │                                      │        │          ┌────────────────┐          │
      │                                      │        │          │   GatherMask   │          │
      │                                      │        │          │ ([m,k']->[m,k])│          │
      │                                      │        │          └───────┬────────┘          │
      │                                      │        │                  ↓                   │
      └──────────────────────────────────────┘        └──────────────────────────────────────┘
                      ↓                                                ↓
                 ┌─────────┐                                    ┌─────────┐
                 │ y[m,k]  │                                    │ y[m,k]  │
                 └─────────┘                                    └─────────┘


图示:
输入输出Tensor
vector计算
数据流向
```

计算过程分为如下几步，均在Vector上进行：

1. brcb步骤：将每个元素广播为一个datablock；
2. Copy步骤：将每个datablock均复制为多个datablock，k对齐场景下即为结果y；
3. 对于k非对齐的场景，再使用GatherMask截取[m, k]个元素， 其中k'表示k向上对齐32B的大小。

#### 函数原型

- 通过sharedTmpBuffer入参传入临时空间

```
template <typename T, int32_t dim, int32_t axis, bool isReuseSource = false>
__aicore__ inline void Broadcast(const LocalTensor<T>& dstLocal, const LocalTensor<T>& srcLocal, const uint32_t dstShape[dim], const uint32_t srcShape[dim], LocalTensor<uint8_t>& sharedTmpBuffer)
```

- 接口框架申请临时空间

```
template <typename T, int32_t dim, int32_t axis, bool isReuseSource = false>
__aicore__ inline void Broadcast(const LocalTensor<T>& dstLocal, const LocalTensor<T>& srcLocal, const uint32_t dstShape[dim], const uint32_t srcShape[dim])
```

该接口需要额外的临时空间来存储计算过程中的中间变量。临时空间支持开发者**通过sharedTmpBuffer入参传入**和**接口框架申请**两种方式。

- 通过sharedTmpBuffer入参传入，使用该tensor作为临时空间进行处理，接口框架不再申请。该方式开发者可以自行管理sharedTmpBuffer内存空间，并在接口调用完成后，复用该部分内存，内存不会反复申请释放，灵活性较高，内存利用率也较高。
- 接口框架申请临时空间，开发者无需申请，但是需要预留临时空间的大小。

通过sharedTmpBuffer传入的情况，开发者需要为tensor申请空间；接口框架申请的方式，开发者需要预留临时空间。临时空间大小BufferSize的获取方式如下：通过GetBroadCastMaxMinTmpSize中提供的接口获取需要预留空间范围的大小。

#### 参数说明

**表1 **模板参数说明

| 参数名称 | 功能 |
| --- | --- |
| T | 操作数的数据类型。                       Atlas A3 训练系列产品            /             Atlas A3 推理系列产品            ，支持的数据类型为：int8_t、uint8_t、half、float。                       Atlas A2 训练系列产品            /             Atlas A2 推理系列产品            ，支持的数据类型为：int8_t、uint8_t、half、float。                       Atlas 推理系列产品            AI Core，支持的数据类型为：int8_t、uint8_t、half、float。 |
| dim | 输入/输出tensor的维度，目前仅支持1维和2维。 |
| axis | 要广播的维度，目前仅支持0和1。参数取值为0代表要广播第1维，取值为1代表要广播第2维。 |
| isReuseSource | 是否允许修改源操作数。该参数预留，传入默认值false即可。 |

**表2 **接口参数说明

| 参数名称 | 输入/输出 | 描述 |
| --- | --- | --- |
| dstLocal | 输出 | 目的操作数。          类型为LocalTensor，支持的TPosition为VECIN/VECCALC/VECOUT。 |
| srcLocal | 输入 | 源操作数。          源操作数的数据类型需要与目的操作数保持一致。          类型为LocalTensor，支持的TPosition为VECIN/VECCALC/VECOUT。 |
| dstShape | 输入 | 输出tensor的shape：uint32_t类型的数组，长度为1或者2， 输入/输出的shape维度数目必须一致。 |
| srcShape | 输入 | 输入tensor的shape：uint32_t类型的数组，长度为1或者2， 输入/输出的shape维度数目必须一致。 |
| sharedTmpBuffer | 输入 | 临时缓存。          类型为LocalTensor，支持的TPosition为VECIN/VECCALC/VECOUT。          用于Broadcast内部复杂计算时存储中间变量，由开发者提供。          临时空间大小BufferSize的获取方式请参考GetBroadCastMaxMinTmpSize。 |

#### 约束说明

- **不支持源操作数与目的操作数地址重叠。**
- 当前仅支持ND格式的输入，不支持其他格式。
- dim目前仅支持1或者2， axis目前仅支持0或者1。
- 对于
        Atlas 推理系列产品
       AI Core，在dim=2且axis=1时，srcShape[0]必须为32B对齐，即当输入/输出Tensor的维度为2维，且广播的维度为1时，输入Tensor的第0维数据需为32字节的倍数。
- 在dim=2，axis=0时，要求srcShape[1]必须32B对齐。

#### 调用示例

更多算子样例请参考[broadcast算子样例](https://gitee.com/ascend/ascendc-api-adv/tree/master/examples/pad/broadcast)。

```
#include "kernel_operator.h"

template <typename T, int32_t dim, int32_t axis>
class KernelBroadcast {
public:
    __aicore__ inline KernelBroadcast()
    {}
    __aicore__ inline void Init(
        GM_ADDR srcGm, GM_ADDR dstGm, const uint32_t dstShape[dim], const uint32_t srcShape[dim])
    {
        for (uint32_t i = 0; i < dim; i++) {
            srcSize *= srcShape[i];
            dstSize *= dstShape[i];
        }
        srcGlobal.SetGlobalBuffer(reinterpret_cast<__gm__ T *>(srcGm), srcSize);
        dstGlobal.SetGlobalBuffer(reinterpret_cast<__gm__ T *>(dstGm), dstSize);

        pipe.InitBuffer(inQueueX, 1, srcSize * sizeof(T));
        pipe.InitBuffer(outQueue, 1, dstSize * sizeof(T));
        dstShape_ = dstShape;
        srcShape_ = srcShape;
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
        AscendC::DataCopy(srcLocal, srcGlobal, srcSize);
        inQueueX.EnQue(srcLocal);
    }
    __aicore__ inline void Compute()
    {
        AscendC::LocalTensor<T> dstLocal = outQueue.AllocTensor<T>();
        AscendC::LocalTensor<T> srcLocal = inQueueX.DeQue<T>();
        AscendC::Broadcast<T, dim, axis>(dstLocal, srcLocal, dstShape_, srcShape_);

        outQueue.EnQue<T>(dstLocal);
        inQueueX.FreeTensor(srcLocal);
    }
    __aicore__ inline void CopyOut()
    {
        AscendC::LocalTensor<T> dstLocal = outQueue.DeQue<T>();
        AscendC::DataCopy(dstGlobal, dstLocal, dstSize);
        outQueue.FreeTensor(dstLocal);
    }

private:
    AscendC::GlobalTensor<T> srcGlobal;
    AscendC::GlobalTensor<T> dstGlobal;

    AscendC::TPipe pipe;
    AscendC::TQue<AscendC::TPosition::VECIN, 1> inQueueX;
    AscendC::TQue<AscendC::TPosition::VECOUT, 1> outQueue;
    const uint32_t *dstShape_{nullptr};
    const uint32_t *srcShape_{nullptr};
    int32_t srcSize{1};
    int32_t dstSize{1};
};

template <typename T, int32_t dim, int32_t axis>
__aicore__ void kernel_broadcast_operator(
    GM_ADDR srcGm, GM_ADDR dstGm, const uint32_t dstShape[dim], const uint32_t srcShape[dim])
{
    KernelBroadcast<T, dim, axis> op;
    op.Init(srcGm, dstGm, dstShape, srcShape);
    op.Process();
}
```

     结果示例如下：

```
输入数据（srcLocal）: 
[[ 1]
 [ 2]
 [ 3]
 [ 4]
 [ 5]
 [ 6]
 [ 7]
 [ 8]
 [ 9]
 [10]
 [11]
 [12]
 [13]
 [14]
 [15]
 [16]]
dim：2
axis：1
输出数据（dstLocal）: 
[[ 1  1  1  1  1  1  1  1  1  1  1  1  1  1  1  1]
 [ 2  2  2  2  2  2  2  2  2  2  2  2  2  2  2  2]
 [ 3  3  3  3  3  3  3  3  3  3  3  3  3  3  3  3]
 [ 4  4  4  4  4  4  4  4  4  4  4  4  4  4  4  4]
 [ 5  5  5  5  5  5  5  5  5  5  5  5  5  5  5  5]
 [ 6  6  6  6  6  6  6  6  6  6  6  6  6  6  6  6]
 [ 7  7  7  7  7  7  7  7  7  7  7  7  7  7  7  7]
 [ 8  8  8  8  8  8  8  8  8  8  8  8  8  8  8  8]
 [ 9  9  9  9  9  9  9  9  9  9  9  9  9  9  9  9]
 [10 10 10 10 10 10 10 10 10 10 10 10 10 10 10 10]
 [11 11 11 11 11 11 11 11 11 11 11 11 11 11 11 11]
 [12 12 12 12 12 12 12 12 12 12 12 12 12 12 12 12]
 [13 13 13 13 13 13 13 13 13 13 13 13 13 13 13 13]
 [14 14 14 14 14 14 14 14 14 14 14 14 14 14 14 14]
 [15 15 15 15 15 15 15 15 15 15 15 15 15 15 15 15]
 [16 16 16 16 16 16 16 16 16 16 16 16 16 16 16 16]]
```
