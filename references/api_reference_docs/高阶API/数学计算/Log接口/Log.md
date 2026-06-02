# Log

**页面ID:** atlasascendc_api_07_0512  
**来源:** https://www.hiascend.com/document/detail/zh/CANNCommunityEdition/850/API/ascendcopapi/atlasascendc_api_07_0512.html

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

按元素以e、2、10为底做对数运算，计算公式如下：

![](images/atlasascendc_api_07_0512_img_001.png)

![](images/atlasascendc_api_07_0512_img_002.png)

![](images/atlasascendc_api_07_0512_img_003.png)

![](images/atlasascendc_api_07_0512_img_004.png)

![](images/atlasascendc_api_07_0512_img_005.png)

![](images/atlasascendc_api_07_0512_img_006.png)

#### 函数原型

- 以e为底：

  - 源操作数Tensor全部/部分参与计算

```
template<typename T, bool isReuseSource = false>
__aicore__ inline void Log(const LocalTensor<T>& dstTensor, const LocalTensor<T>& srcTensor, uint32_t calCount)
```

  - 源操作数Tensor全部参与计算

```
template<typename T, bool isReuseSource = false>
__aicore__ inline void Log(const LocalTensor<T>& dstTensor, const LocalTensor<T>& srcTensor)
```

- 以2为底

  - 通过sharedTmpBuffer入参传入临时空间

    - 源操作数Tensor全部/部分参与计算

```
template<typename T, bool isReuseSource = false>
__aicore__ inline void Log2(const LocalTensor<T>& dstTensor, const LocalTensor<T>& srcTensor, const LocalTensor<uint8_t>& sharedTmpBuffer, uint32_t calCount)
```

    - 源操作数Tensor全部参与计算

```
template <typename T, bool isReuseSource = false>
__aicore__ inline void Log2(const LocalTensor<T>& dstTensor, const LocalTensor<T>& srcTensor, const LocalTensor<uint8_t>& sharedTmpBuffer)
```

  - 接口框架申请临时空间

    - 源操作数Tensor全部/部分参与计算

```
template<typename T, bool isReuseSource = false>
__aicore__ inline void Log2(const LocalTensor<T>& dstTensor, const LocalTensor<T>& srcTensor, uint32_t calCount)
```

    - 源操作数Tensor全部参与计算

```
template <typename T, bool isReuseSource = false>
__aicore__ inline void Log2(const LocalTensor<T>& dstTensor, const LocalTensor<T>& srcTensor)
```

- 以10为底：

  - 源操作数Tensor全部/部分参与计算

```
template<typename T, bool isReuseSource = false>
__aicore__ inline void Log10(const LocalTensor<T>& dstTensor, const LocalTensor<T>& srcTensor, uint32_t calCount)
```

  - 源操作数Tensor全部参与计算

```
template<typename T, bool isReuseSource = false>
__aicore__ inline void Log10(const LocalTensor<T>& dstTensor, const LocalTensor<T>& srcTensor)
```

由于该接口的内部实现中涉及复杂的数学计算，需要额外的临时空间来存储计算过程中的中间变量。临时空间支持开发者**通过sharedTmpBuffer入参传入**和**接口框架申请**两种方式。

- 通过sharedTmpBuffer入参传入，使用该tensor作为临时空间进行处理，接口框架不再申请。该方式开发者可以自行管理sharedTmpBuffer内存空间，并在接口调用完成后，复用该部分内存，内存不会反复申请释放，灵活性较高，内存利用率也较高。
- 接口框架申请临时空间，开发者无需申请，但是需要预留临时空间的大小。

通过sharedTmpBuffer传入的情况，开发者需要为tensor申请空间；接口框架申请的方式，开发者需要预留临时空间。临时空间大小BufferSize的获取方式如下：通过GetLogMaxMinTmpSize中提供的接口获取需要预留空间范围的大小。

#### 参数说明

**表1 **模板参数说明

| 参数名 | 描述 |
| --- | --- |
| T | 操作数的数据类型。 Atlas A3 训练系列产品/Atlas A3 推理系列产品，支持的数据类型为：half、float。 Atlas A2 训练系列产品/Atlas A2 推理系列产品，支持的数据类型为：half、float。 Atlas 推理系列产品AI Core，支持的数据类型为：half、float。 |
| isReuseSource | 是否允许修改源操作数。该参数预留，传入默认值false即可。 |

**表2 **参数说明

| 参数名 | 输入/输出 | 描述 |
| --- | --- | --- |
| dstTensor | 输出 | 目的操作数。 类型为LocalTensor，支持的TPosition为VECIN/VECCALC/VECOUT。 |
| srcTensor | 输入 | 源操作数。 类型为LocalTensor，支持的TPosition为VECIN/VECCALC/VECOUT。 源操作数的数据类型需要与目的操作数保持一致。 |
| sharedTmpBuffer | 输入 | 临时缓存。 类型为LocalTensor，支持的TPosition为VECIN/VECCALC/VECOUT。 临时空间大小BufferSize的获取方式请参考GetLogMaxMinTmpSize。 |
| calCount | 输入 | 参与计算的元素个数。 |

#### 约束说明

- **不支持源操作数与目的操作数地址重叠。**

#### 调用示例

本样例中只展示Compute流程中的部分代码。如果您需要运行样例代码，请将该代码段拷贝并替换样例模板中Compute函数的部分代码即可。

- Log接口样例

```
Log(dstLocal, srcLocal);
```

结果示例如下：

```
输入数据(srcLocal): [144.22607 9634.764 ... 1835.1245 3145.5125]
输出数据(dstLocal): [4.971382 9.173133 ... 7.514868 8.053732]
```

- Log2接口样例

```
Log2(dstLocal, srcLocal);
```

结果示例如下：

```
输入数据(srcLocal): [6299.54 338.45963 ... 2.853525 5752.1323]
输出数据(dstLocal): [12.621031 8.40284 ... 1.5127451 12.4898815]
```

- Log10接口样例

```
Log10(dstLocal, srcLocal);
```

结果示例如下：

```
输入数据(srcLocal): [712.7535 78.36265 ... 3099.0571 9313.082]
输出数据(dstLocal): [2.8529394 1.8941091 ... 3.4912295 3.9690933]
```

#### 样例模板

```
#include "kernel_operator.h"
template <typename srcType>
class KernelLog
{
public:
    __aicore__ inline KernelLog()
    {
    }
    __aicore__ inline void Init(GM_ADDR srcGm, GM_ADDR dstGm, uint32_t srcSize)
    {
        src_global.SetGlobalBuffer(reinterpret_cast<__gm__ srcType *>(srcGm), srcSize);
        dst_global.SetGlobalBuffer(reinterpret_cast<__gm__ srcType *>(dstGm), srcSize);
        pipe.InitBuffer(inQueueX, 1, srcSize * sizeof(srcType));
        pipe.InitBuffer(outQueue, 1, srcSize * sizeof(srcType));
        bufferSize = srcSize;
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
        AscendC::DataCopy(srcLocal, src_global, bufferSize);
        inQueueX.EnQue(srcLocal);
    }
    __aicore__ inline void Compute()
    {
        AscendC::LocalTensor<srcType> dstLocal = outQueue.AllocTensor<srcType>();
        AscendC::LocalTensor<srcType> srcLocal = inQueueX.DeQue<srcType>();
        AscendC::Log(dstLocal, srcLocal);
        // 或可调用 AscendC::Log10(dstLocal, srcLocal);
        // 或可调用 AscendC::Log2(dstLocal, srcLocal);
        outQueue.EnQue<srcType>(dstLocal);
        inQueueX.FreeTensor(srcLocal);
    }
    __aicore__ inline void CopyOut()
    {
        AscendC::LocalTensor<srcType> dstLocal = outQueue.DeQue<srcType>();
        AscendC::DataCopy(dst_global, dstLocal, bufferSize);
        outQueue.FreeTensor(dstLocal);
    }

private:
    AscendC::GlobalTensor<srcType> src_global;
    AscendC::GlobalTensor<srcType> dst_global;
    AscendC::TPipe pipe;
    AscendC::TQue<AscendC::TPosition::VECIN, 1> inQueueX;
    AscendC::TQue<AscendC::TPosition::VECOUT, 1> outQueue;
    uint32_t bufferSize = 0;
};
template <typename dataType>
__aicore__ void kernel_log_operator(GM_ADDR srcGm, GM_ADDR dstGm, uint32_t srcSize)
{
    KernelLog<dataType> op;
    op.Init(srcGm, dstGm, srcSize);
    op.Process();
}

extern "C" __global__ __aicore__ void log_operator_custom(GM_ADDR srcGm, GM_ADDR dstGm, uint32_t srcSize)
{
    kernel_log_operator<half>(srcGm, dstGm, srcSize); // 传入类型和大小
}
```
