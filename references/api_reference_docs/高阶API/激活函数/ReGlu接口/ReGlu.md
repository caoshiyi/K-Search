# ReGlu

**页面ID:** atlasascendc_api_07_0790  
**来源:** https://www.hiascend.com/document/detail/zh/CANNCommunityEdition/850/API/ascendcopapi/atlasascendc_api_07_0790.html

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

ReGlu是一种GLU变体，使用Relu作为激活函数，计算公式如下：

<!-- img2text -->
```
ReGlu(x, y) = x * max(y, 0)
```

其中Relu激活函数的计算公式如下：

![](images/atlasascendc_api_07_0790_img_002.png)

#### 函数原型

- 通过sharedTmpBuffer入参传入临时空间

```
template <typename T, bool isReuseSource = false>
__aicore__ inline void ReGlu(const LocalTensor<T>& dstTensor, const LocalTensor<T>& srcTensor0, const LocalTensor<T>& srcTensor1, const LocalTensor<uint8_t>& sharedTmpBuffer, const uint32_t calCount)
```

- 接口框架申请临时空间

```
template <typename T, bool isReuseSource = false>
__aicore__ inline void ReGlu(const LocalTensor<T>& dstTensor, const LocalTensor<T>& srcTensor0, const LocalTensor<T>& srcTensor1, const uint32_t calCount)
```

由于该接口的内部实现中涉及复杂的数学计算，需要额外的临时空间来存储计算过程中的中间变量。临时空间支持开发者**通过sharedTmpBuffer入参传入**和**接口框架申请**两种方式。

- 通过sharedTmpBuffer入参传入，使用该tensor作为临时空间进行处理，接口框架不再申请。该方式开发者可以自行管理sharedTmpBuffer内存空间，并在接口调用完成后，复用该部分内存，内存不会反复申请释放，灵活性较高，内存利用率也较高。
- 接口框架申请临时空间，开发者无需申请，但是需要预留临时空间的大小。

通过sharedTmpBuffer传入的情况，开发者需要为tensor申请空间；接口框架申请的方式，开发者需要预留临时空间。临时空间大小BufferSize的获取方式如下：通过GetReGluMaxMinTmpSize中提供的接口获取需要预留空间范围的大小。

#### 参数说明

**表1 **模板参数说明

| 参数名 | 描述 |
| --- | --- |
| T | 操作数的数据类型。 Atlas A3 训练系列产品/Atlas A3 推理系列产品，支持的数据类型为：half、bfloat16_t、float。 Atlas A2 训练系列产品/Atlas A2 推理系列产品，支持的数据类型为：half、bfloat16_t、float。 Atlas 推理系列产品AI Core，支持的数据类型为：half、float。 |
| isReuseSource | 是否允许修改源操作数。该参数预留，传入默认值false即可。 |

**表2 **接口参数说明

| 参数名 | 输入/输出 | 描述 |
| --- | --- | --- |
| dstTensor | 输出 | 目的操作数。 类型为LocalTensor，支持的TPosition为VECIN/VECCALC/VECOUT。 |
| srcTensor0 | 输入 | 源操作数。 类型为LocalTensor，支持的TPosition为VECIN/VECCALC/VECOUT。 源操作数的数据类型需要与目的操作数保持一致。 |
| srcTensor1 | 输入 | 源操作数。 类型为LocalTensor，支持的TPosition为VECIN/VECCALC/VECOUT。 源操作数的数据类型需要与目的操作数保持一致。 |
| sharedTmpBuffer | 输入 | 临时缓存。 类型为LocalTensor，支持的TPosition为VECIN/VECCALC/VECOUT。 用于ReGlu内部复杂计算时存储中间变量，由开发者提供。 临时空间大小BufferSize的获取方式请参考GetReGluMaxMinTmpSize。 |
| calCount | 输入 | 实际计算数据元素个数。 |

#### 约束说明

- **不支持源操作数与目的操作数地址重叠。**
- 不支持sharedTmpBuffer与源操作数和目的操作数地址重叠。
- 当前仅支持ND格式的输入，不支持其他格式。

#### 调用示例

kernel侧调用示例

```
#include "kernel_operator.h"
template <typename srcType>
class KernelReGlu
{
public:
    __aicore__ inline KernelReGlu() {}
    __aicore__ inline void Init(GM_ADDR src0Gm, GM_ADDR src1Gm, GM_ADDR dstGm, uint32_t srcSize)
    {
        dataSize = srcSize;
        src0Global.SetGlobalBuffer(reinterpret_cast<__gm__ srcType *>(src0Gm), dataSize);
        src1Global.SetGlobalBuffer(reinterpret_cast<__gm__ srcType *>(src1Gm), dataSize);
        dstGlobal.SetGlobalBuffer(reinterpret_cast<__gm__ srcType *>(dstGm), dataSize);
        pipe.InitBuffer(inQueueX, 1, dataSize * sizeof(srcType));
        pipe.InitBuffer(inQueueY, 1, dataSize * sizeof(srcType));
        pipe.InitBuffer(outQueue, 1, dataSize * sizeof(srcType));
        if (sizeof(srcType) != sizeof(float))
        {
            pipe.InitBuffer(calcBufs, dataSize * (sizeof(float) / sizeof(uint8_t)) * 3);
        }
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
        AscendC::LocalTensor<srcType> src0Local = inQueueX.AllocTensor<srcType>();
        AscendC::LocalTensor<srcType> src1Local = inQueueY.AllocTensor<srcType>();
        AscendC::DataCopy(src0Local, src0Global, dataSize);
        AscendC::DataCopy(src1Local, src1Global, dataSize);
        inQueueX.EnQue(src0Local);
        inQueueY.EnQue(src1Local);
    }
    __aicore__ inline void Compute()
    {
        AscendC::LocalTensor<srcType> dstLocal = outQueue.AllocTensor<srcType>();
        AscendC::LocalTensor<srcType> src0Local = inQueueX.DeQue<srcType>();
        AscendC::LocalTensor<srcType> src1Local = inQueueY.DeQue<srcType>();
        AscendC::LocalTensor<uint8_t> tmpLocal;
        if (sizeof(srcType) != sizeof(float))
        {
            tmpLocal = calcBufs.Get<uint8_t>();
            AscendC::ReGlu<srcType, false>(dstLocal, src0Local, src1Local, tmpLocal, dataSize);
        }
        else
        {
            AscendC::ReGlu<srcType, false>(dstLocal, src0Local, src1Local, dataSize);
        }
        outQueue.EnQue<srcType>(dstLocal);
        inQueueX.FreeTensor(src0Local);
        inQueueY.FreeTensor(src1Local);
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
    AscendC::TQue<AscendC::TPosition::VECIN, 1> inQueueX;
    AscendC::TQue<AscendC::TPosition::VECIN, 1> inQueueY;
    AscendC::TQue<AscendC::TPosition::VECOUT, 1> outQueue;
    AscendC::TBuf<AscendC::TPosition::VECCALC> calcBufs;
    uint32_t dataSize = 0;
};
template <typename dataType>
__aicore__ void kernel_reglu_operator(GM_ADDR src0Gm, GM_ADDR src1Gm, GM_ADDR dstGm, uint32_t srcSize)
{
    KernelReGlu<dataType> op;
    op.Init(src0Gm, src1Gm, dstGm, srcSize);
    op.Process();
}
```

结果示例如下：

```
输入数据(srcLocal0): 
[ 22.28125    78.375     -10.3515625 -80.75      -22.8125     84.375
  -8.96875    70.5       -51.75       66.875      69.8125      5.2734375
 -51.         50.5       -30.765625  -52.125       8.03125    75.8125
  50.4375    -97.1875    -80.6875     17.125     -30.640625  -13.671875
  92.375      68.8125     53.75        5.1054688  39.6875    -46.71875
  90.25       67.75     ]
输入数据(srcLocal1): 
[ 61.46875   -36.5625    -93.3125    -87.6875    -17.96875   -88.125
 -46.65625   -18.78125    13.4921875 -87.875      65.75      -25.96875
 -44.5625     53.        -69.375      96.5       -24.703125   77.5625
  78.875      -6.0898438 -40.5625    -69.625      57.         18.640625
 -73.875      94.375      91.5        -9.7109375  84.125      79.0625
  88.5        96.3125   ]
输出数据(dstLocal): 
[ 0.0000e+00  0.0000e+00  0.0000e+00 -6.5450e+02  0.0000e+00  1.2544e+02
  3.7880e+03  1.0519e+02 -0.0000e+00 -0.0000e+00 -0.0000e+00  0.0000e+00
 -2.0110e+03  0.0000e+00 -2.8020e+03 -0.0000e+00  0.0000e+00 -2.6120e+03
  6.8840e+03 -0.0000e+00  8.6550e+02 -0.0000e+00  0.0000e+00 -7.4120e+03
 -1.9700e+03  2.3140e+03 -0.0000e+00  0.0000e+00 -0.0000e+00  7.6760e+03
 -4.8828e-01 -0.0000e+00]
```
