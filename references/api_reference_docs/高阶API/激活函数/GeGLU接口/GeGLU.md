# GeGLU

**页面ID:** atlasascendc_api_07_0786  
**来源:** https://www.hiascend.com/document/detail/zh/CANNCommunityEdition/850/API/ascendcopapi/atlasascendc_api_07_0786.html

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

GeGLU是采用GELU作为激活函数的GLU变体。具体计算公式如下：

<!-- img2text -->
```
                    ┌───────────┐
                    │           │
                    │   GELU    │
                    │           │
                    └───────────┘
                         ▲
                         │
x2 ──────────────────────┘

x1 ────────────────────────────────┐
                                    ▼
                               ┌────────┐
                               │   ×    │
                               └────────┘
                                    │
                                    ▼
                                    y
```

其中GELU激活函数的计算公式如下：

<!-- img2text -->
```


                    x
GELU(x) = ────────────────
                 x
           1 + e^(-1.702x)

```

上述公式中的erf为误差函数：![](images/atlasascendc_api_07_0786_img_003.png)

误差函数没有解析表达式，按照业界普遍使用的tanh近似表达式：![](images/atlasascendc_api_07_0786_img_004.png)

将GELU近似公式代入可得GeGLU表达式为：

![](images/atlasascendc_api_07_0786_img_005.png)

其中*a*=-0.0713548162726, *b*=2.2363860002236e1，x1和x0代表srcTensor1和srcTensor0中的元素。

#### 函数原型

- 通过sharedTmpBuffer入参传入临时空间

  - 源操作数Tensor全部/部分参与计算

```
template <typename T, bool isReuseSource = false>
__aicore__ inline void GeGLU(const LocalTensor<T>& dstTensor, const LocalTensor<T>& srcTensor0, const LocalTensor<T>& srcTensor1, const LocalTensor<uint8_t>& sharedTmpBuffer, uint32_t calCount)
```

  - 源操作数Tensor全部参与计算

```
template <typename T, bool isReuseSource = false>
__aicore__ inline void GeGLU(const LocalTensor<T>& dstTensor, const LocalTensor<T>& srcTensor0, const LocalTensor<T>& srcTensor1, const LocalTensor<uint8_t>& sharedTmpBuffer)
```

- 接口框架申请临时空间

  - 源操作数Tensor全部/部分参与计算

```
template <typename T, bool isReuseSource = false>
__aicore__ inline void GeGLU(const LocalTensor<T>& dstTensor, const LocalTensor<T>& srcTensor0, const LocalTensor<T>& srcTensor1, uint32_t calCount)
```

  - 源操作数Tensor全部参与计算

```
template <typename T, bool isReuseSource = false>
__aicore__ inline void GeGLU(const LocalTensor<T>& dstTensor, const LocalTensor<T>& srcTensor0, const LocalTensor<T>& srcTensor1)
```

由于该接口的内部实现中涉及复杂的数学计算，需要额外的临时空间来存储计算过程中的中间变量。临时空间支持开发者**通过sharedTmpBuffer入参传入**和**接口框架申请**两种方式。

- 通过sharedTmpBuffer入参传入，使用该tensor作为临时空间进行处理，接口框架不再申请。该方式开发者可以自行管理sharedTmpBuffer内存空间，并在接口调用完成后，复用该部分内存，内存不会反复申请释放，灵活性较高，内存利用率也较高。
- 接口框架申请临时空间，开发者无需申请，但是需要预留临时空间的大小。

通过sharedTmpBuffer传入的情况，开发者需要为tensor申请空间；接口框架申请的方式，开发者需要预留临时空间。临时空间大小BufferSize的获取方式如下：通过GetGeGLUMaxMinTmpSize中提供的接口获取需要预留空间范围的大小。

#### 参数说明

**表1 **模板参数说明

| 参数名 | 描述 |
| --- | --- |
| T | 操作数的数据类型。 Atlas A3 训练系列产品/Atlas A3 推理系列产品，支持的数据类型为：half、float。 Atlas A2 训练系列产品/Atlas A2 推理系列产品，支持的数据类型为：half、float。 Atlas 推理系列产品AI Core，支持的数据类型为：half、float。 |
| isReuseSource | 是否允许修改源操作数。该参数预留，传入默认值false即可。 |

**表2 **接口参数说明

| 参数名 | 输入/输出 | 描述 |
| --- | --- | --- |
| dstTensor | 输出 | 目的操作数。 类型为LocalTensor，支持的TPosition为VECIN/VECCALC/VECOUT。 |
| srcTensor0/ srcTensor1 | 输入 | 源操作数。 源操作数的数据类型需要与目的操作数保持一致。 类型为LocalTensor，支持的TPosition为VECIN/VECCALC/VECOUT。 |
| sharedTmpBuffer | 输入 | 临时缓存。 类型为LocalTensor，支持的TPosition为VECIN/VECCALC/VECOUT。 用于GeGLU内部复杂计算时存储中间变量，由开发者提供。 临时空间大小BufferSize的获取方式请参考GetGeGLUMaxMinTmpSize。 |
| calCount | 输入 | 实际计算数据元素个数。 |

#### 约束说明

- **不支持源操作数与目的操作数地址重叠。**
- 当前仅支持ND格式的输入，不支持其他格式。
- 不支持sharedTmpBuffer与源操作数和目的操作数地址重叠。

#### 调用示例

```
#include "kernel_operator.h"

template <typename srcType>
class KernelGeGLU
{
public:
    __aicore__ inline KernelGeGLU() {}
    __aicore__ inline void Init(GM_ADDR src0Gm, GM_ADDR src1Gm, GM_ADDR dstGm, uint32_t inputSize,
                                uint32_t tmpBufSize)
    {
        dataSize = inputSize;
        uint32_t bufSize = 4 * tmpBufSize;
        src0Global.SetGlobalBuffer(reinterpret_cast<__gm__ srcType *>(src0Gm), dataSize);
        src1Global.SetGlobalBuffer(reinterpret_cast<__gm__ srcType *>(src1Gm), dataSize);
        dstGlobal.SetGlobalBuffer(reinterpret_cast<__gm__ srcType *>(dstGm), dataSize);
        pipe.InitBuffer(inQueue0, 1, dataSize * sizeof(srcType));
        pipe.InitBuffer(inQueue1, 1, dataSize * sizeof(srcType));
        pipe.InitBuffer(outQueue, 1, dataSize * sizeof(srcType));
        if ((sizeof(srcType) == sizeof(half)) && (tmpBufSize > 0))
        {
            pipe.InitBuffer(buf, bufSize * sizeof(srcType));
        }
    }
    __aicore__ inline void Process(uint32_t tmpBufSize, uint32_t calCount)
    {
        CopyIn();
        Compute(tmpBufSize, calCount);
        CopyOut();
    }

private:
    __aicore__ inline void CopyIn()
    {
        AscendC::LocalTensor<srcType> src0Local = inQueue0.AllocTensor<srcType>();
        AscendC::LocalTensor<srcType> src1Local = inQueue1.AllocTensor<srcType>();
        AscendC::DataCopy(src0Local, src0Global, dataSize);
        AscendC::DataCopy(src1Local, src1Global, dataSize);
        inQueue0.EnQue(src0Local);
        inQueue1.EnQue(src1Local);
    }
    __aicore__ inline void Compute(uint32_t tmpBufSize, uint32_t calCount)
    {
        AscendC::LocalTensor<srcType> dstLocal = outQueue.AllocTensor<srcType>();
        AscendC::LocalTensor<srcType> src0Local = inQueue0.DeQue<srcType>();
        AscendC::LocalTensor<srcType> src1Local = inQueue1.DeQue<srcType>();
        AscendC::LocalTensor<uint8_t> temp;
        if ((sizeof(srcType) == sizeof(half)) && (tmpBufSize > 0)) {
            temp = buf.Get<uint8_t>();
        }
        if ((tmpBufSize > 0) && (calCount > 0)) {
            AscendC::GeGLU<srcType, false>(dstLocal, src0Local, src1Local, temp, calCount);
        } else if (tmpBufSize > 0) {
            AscendC::GeGLU<srcType, false>(dstLocal, src0Local, src1Local, temp);
        } else if (calCount > 0) {
            AscendC::GeGLU<srcType, false>(dstLocal, src0Local, src1Local, calCount);
        } else {
            AscendC::GeGLU<srcType, false>(dstLocal, src0Local, src1Local);
        }
        outQueue.EnQue<srcType>(dstLocal);
        inQueue0.FreeTensor(src0Local);
        inQueue1.FreeTensor(src1Local);
    }
    __aicore__ inline void CopyOut()
    {
        AscendC::LocalTensor<srcType> dstLocal = outQueue.DeQue<srcType>();
        AscendC::DataCopy(dstGlobal, dstLocal, dataSize);
        outQueue.FreeTensor(dstLocal);
    }

private:
    AscendC::GlobalTensor<srcType> src0Global;
    AscendC::GlobalTensor<srcType> src1Global;
    AscendC::GlobalTensor<srcType> dstGlobal;
    AscendC::TPipe pipe;
    AscendC::TQue<AscendC::TPosition::VECIN, 1> inQueue0;
    AscendC::TQue<AscendC::TPosition::VECIN, 1> inQueue1;
    AscendC::TQue<AscendC::TPosition::VECOUT, 1> outQueue;
    AscendC::TBuf<AscendC::TPosition::VECCALC> buf;
    uint32_t dataSize = 0;
};
template <typename dataType>
__aicore__ void kernel_geglu_operator(GM_ADDR src0Gm, GM_ADDR src1Gm, GM_ADDR dstGm, uint32_t srcSize,
                                      uint32_t tmpBufSize, uint32_t calCount)
{
    KernelGeGLU<dataType> op;
    op.Init(src0Gm, src1Gm, dstGm, srcSize, tmpBufSize);
    op.Process(tmpBufSize, calCount);
}
```

结果示例如下：

```
输入数据(srcTensor0): 
[ 1.6025391   3.4765625   3.4316406   3.7539062  -1.3330078   0.72314453
 -3.0078125   0.85498047 -1.3691406   2.6894531  -2.9101562  -3.6992188
 -2.2734375  -2.859375    2.5683594  -1.7802734 ]

输入数据(srcTensor1)
[-0.6015625  1.9589844  1.9257812  3.8769531  0.5878906  2.9179688
 -1.8847656  3.2304688  2.8945312  2.4550781  1.3730469 -1.9248047
  0.7919922 -2.5332031 -2.1425781 -2.9433594]

输出数据(dstLocal): [-0.263916015625000000 6.640625000000000000 6.429687500000000000
14.554687500000000000 -0.565429687500000000 2.107421875000000000 0.168579101562500000
2.759765625000000000 -3.957031250000000000 6.558593750000000000 -3.656250000000000000
0.192993164062500000 -1.415039062500000000 0.039642333984375000 -0.087890625000000000
0.007740020751953125]
```
