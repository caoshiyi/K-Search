# ReduceMin

**页面ID:** atlasascendc_api_07_0077  
**来源:** https://www.hiascend.com/document/detail/zh/CANNCommunityEdition/850/API/ascendcopapi/atlasascendc_api_07_0077.html

---

#### 产品支持情况

| 产品 | 是否支持 |
| --- | --- |
| Atlas A3 训练系列产品            /             Atlas A3 推理系列产品 | √ |
| Atlas A2 训练系列产品            /             Atlas A2 推理系列产品 | √ |
| Atlas 200I/500 A2 推理产品 | √ |
| Atlas 推理系列产品            AI Core | √ |
| Atlas 推理系列产品            Vector Core | x |
| Atlas 训练系列产品 | √ |

#### 功能说明

在所有的输入数据中找出最小值及最小值对应的索引位置。归约指令的总体介绍请参考如何使用归约计算API。ReduceMin计算原理参考ReduceMax。

#### 函数原型

- tensor前n个数据计算

```
template <typename T>
__aicore__ inline void ReduceMin(const LocalTensor<T>& dst, const LocalTensor<T>& src, const LocalTensor<T>& sharedTmpBuffer, const int32_t count, bool calIndex = 0)
```

- tensor高维切分计算

  - mask逐bit模式

```
template <typename T>
__aicore__ inline void ReduceMin(const LocalTensor<T>& dst, const LocalTensor<T>& src, const LocalTensor<T>& sharedTmpBuffer, const uint64_t mask[], const int32_t repeatTime, const int32_t srcRepStride, bool calIndex = 0)
```

  - mask连续模式

```
template <typename T>
__aicore__ inline void ReduceMin(const LocalTensor<T>& dst, const LocalTensor<T>& src, const LocalTensor<T>& sharedTmpBuffer, const int32_t mask, const int32_t repeatTime, const int32_t srcRepStride, bool calIndex = 0)
```

#### 参数说明

**表1 **模板参数说明

| 参数名 | 描述 |
| --- | --- |
| T | 操作数数据类型。                       Atlas A3 训练系列产品            /             Atlas A3 推理系列产品            ，支持的数据类型为：half/float                       Atlas A2 训练系列产品            /             Atlas A2 推理系列产品            ，支持的数据类型为：half/float                       Atlas 200I/500 A2 推理产品            ，支持的数据类型为：half/float                       Atlas 推理系列产品            AI Core，支持的数据类型为：half/float                       Atlas 训练系列产品            ，支持的数据类型为：half |

**表2 **参数说明

| 参数名称 | 输入/输出 | 含义 |
| --- | --- | --- |
| dst | 输出 | 目的操作数。          类型为LocalTensor，支持的TPosition为VECIN/VECCALC/VECOUT。          LocalTensor的起始地址需要保证4字节对齐（针对half数据类型），8字节对齐（针对float数据类型）。 |
| src | 输入 | 源操作数。          类型为LocalTensor，支持的TPosition为VECIN/VECCALC/VECOUT。          LocalTensor的起始地址需要32字节对齐。          源操作数的数据类型需要与目的操作数保持一致。 |
| sharedTmpBuffer | 输入 | API执行期间，**部分硬件型号**需要一块空间用于存储中间结果，空间大小需要满足最小所需空间的要求，具体计算方法可参考ReduceMax计算示意图中的介绍。          类型为LocalTensor，支持的TPosition为VECIN/VECCALC/VECOUT。          LocalTensor的起始地址需要32字节对齐。          数据类型需要与目的操作数保持一致。                       Atlas A3 训练系列产品            /             Atlas A3 推理系列产品            ，需要使用sharedTmpBuffer。                       Atlas A2 训练系列产品            /             Atlas A2 推理系列产品            ，需要使用sharedTmpBuffer。                       Atlas 200I/500 A2 推理产品            ，需要使用sharedTmpBuffer。                       Atlas 推理系列产品            AI Core，需要使用sharedTmpBuffer                       Atlas 训练系列产品            ，需要使用sharedTmpBuffer。 |
| count | 输入 | 参与计算的元素个数。          参数取值范围和操作数的数据类型有关，数据类型不同，能够处理的元素个数最大值不同，最大处理的数据量不能超过UB大小限制。 |
| mask/mask[] | 输入 | mask用于控制每次迭代内参与计算的元素。                     - 逐bit模式：可以按位控制哪些元素参与计算，bit位的值为1表示参与计算，0表示不参与。            mask为数组形式，数组长度和数组元素的取值范围和操作数的数据类型有关。当操作数为16位时，数组长度为2，mask[0]、mask[1]∈[0, 264-1]并且不同时为0；当操作数为32位时，数组长度为1，mask[0]∈(0, 264-1]；当操作数为64位时，数组长度为1，mask[0]∈(0, 232-1]。            例如，mask=[8, 0]，8=0b1000，表示仅第4个元素参与计算。                               - 连续模式：表示前面连续的多少个元素参与计算。取值范围和操作数的数据类型有关，数据类型不同，每次迭代内能够处理的元素个数最大值不同。当操作数为16位时，mask∈[1, 128]；当操作数为32位时，mask∈[1, 64]；当操作数为64位时，mask∈[1, 32]。 |
| repeatTime | 输入 | 迭代次数。与通用参数说明中不同的是，支持更大的取值范围，保证不超过int32_t最大值的范围即可。 |
| srcRepStride | 输入 | 源操作数相邻迭代间的地址步长，即源操作数每次迭代跳过的datablock数目。详细说明请参考repeatStride。 |
| calIndex | 输入 | 指定是否获取最小值的索引，bool类型，默认值为false，取值：                     - true：同时获取最小值和最小值索引。           - false：不获取索引，只获取最小值。 |

#### 约束说明

- dst结果存储顺序为最小值，最小值索引，若不需要索引，只会存储最小值。返回结果中索引index数据是按照dst的数据类型进行存储的，比如dst使用half类型时，index按照half类型进行存储，如果按照half格式进行读取，index的值是不对的，因此index的读取需要使用reinterpret_cast方法转换到整数类型。若输入数据类型是half，需要使用reinterpret_cast<uint16_t*>，若输入是float，需要使用reinterpret_cast<uint32_t*>。比如tensor高维切分计算接口完整调用示例中，计算结果为[0.01034, 2.104e-05]，2.104e-05需要使用reinterpret_cast方法转换得到索引值353。转换示例如下：

```
float minIndex = dst.GetValue(1);
uint32_t realIndex = *reinterpret_cast<uint32_t*>(&minIndex);
```

- 返回最小值索引时，如果存在多个最小值，返回第一个最小值的索引。
- 当输入类型是half的时候，只支持获取最大不超过65535（uint16_t能表示的最大值）的索引值。

#### 调用示例

- tensor高维切分计算样例-mask连续模式

```
// dstLocal,srcLocal和sharedTmpBuffer均为half类型,srcLocal的计算数据量为8320,并且连续排布，需要索引值，使用tensor高维切分计算接口，设定repeatTime为65，mask为全部元素参与计算
int32_t mask = 128;
AscendC::ReduceMin<half>(dstLocal, srcLocal, sharedTmpBuffer, mask, 65, 8, true);
```

- tensor高维切分计算样例-mask逐bit模式

```
// dstLocal,srcLocal和sharedTmpBuffer均为half类型,srcLocal的计算数据量为8320,并且连续排布，需要索引值，使用tensor高维切分计算接口，设定repeatTime为65,mask为全部元素参与计算
uint64_t mask[2] = { 0xFFFFFFFFFFFFFFFF, 0xFFFFFFFFFFFFFFFF };
AscendC::ReduceMin<half>(dstLocal, srcLocal, sharedTmpBuffer, mask, 65, 8, true);
```

- tensor前n个数据计算样例

```
// dstLocal,srcLocal和sharedTmpBuffer均为half类型,srcLocal的计算数据量为8320,并且连续排布，需要索引值，使用tensor前n个数据计算接口
AscendC::ReduceMin<half>(dstLocal, srcLocal, sharedTmpBuffer, 8320, true);
```

- tensor高维切分计算接口完整调用示例

```
#include "kernel_operator.h"
class KernelReduce {
public:
    __aicore__ inline KernelReduce() {}
    __aicore__ inline void Init(__gm__ uint8_t* src, __gm__ uint8_t* dstGm)
    {
        srcGlobal.SetGlobalBuffer((__gm__ half*)src);
        dstGlobal.SetGlobalBuffer((__gm__ half*)dstGm);
        repeat = srcDataSize / mask;
        pipe.InitBuffer(inQueueSrc, 1, srcDataSize * sizeof(half));
        pipe.InitBuffer(workQueue, 1, 32 * sizeof(half)); // 此处按照公式计算所需的最小work空间为32，也就是64Bytes
        pipe.InitBuffer(outQueueDst, 1, dstDataSize * sizeof(half));
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
        AscendC::LocalTensor<half> srcLocal = inQueueSrc.AllocTensor<half>();
        AscendC::DataCopy(srcLocal, srcGlobal, srcDataSize);
        inQueueSrc.EnQue(srcLocal);
    }
    __aicore__ inline void Compute()
    {
        AscendC::LocalTensor<half> srcLocal = inQueueSrc.DeQue<half>();
        AscendC::LocalTensor<half> dstLocal = outQueueDst.AllocTensor<half>();
        AscendC::LocalTensor<half> sharedTmpBuffer = workQueue.AllocTensor<half>();

        AscendC::ReduceMin<half>(dstLocal, srcLocal, sharedTmpBuffer, mask, repeat, repStride, true);
        outQueueDst.EnQue<half>(dstLocal);
        inQueueSrc.FreeTensor(srcLocal);
        workQueue.FreeTensor(sharedTmpBuffer);
    }
    __aicore__ inline void CopyOut()
    {
        AscendC::LocalTensor<half> dstLocal = outQueueDst.DeQue<half>();
        AscendC::DataCopy(dstGlobal, dstLocal, srcDataSize);
        outQueueDst.FreeTensor(dstLocal);
    }
private:
    AscendC::TPipe pipe;
    AscendC::TQue<AscendC::TPosition::VECIN, 1> inQueueSrc;
    AscendC::TQue<AscendC::TPosition::VECOUT, 1> workQueue;
    AscendC::TQue<AscendC::TPosition::VECOUT, 1> outQueueDst;
    AscendC::GlobalTensor<half> srcGlobal, dstGlobal;
    int srcDataSize = 512;
    int dstDataSize = 512;
    int mask = 128;
    int repStride = 8;
    int repeat = 0;
};
extern "C" __global__ __aicore__ void kernel_ReduceMin_lv0_half_512(__gm__ uint8_t* src, __gm__ uint8_t* dstGm)
{
    KernelReduce op;
    op.Init(src, dstGm);
    op.Process();
}
```

示例结果如下：

```
输入数据(src_gm):
[0.769    0.8584   0.1082   0.2715   0.1759   0.7646   0.6406   0.2944   0.4255   0.927    0.8022   0.04507  0.9688   0.919    0.3008   0.7144   0.3206   0.6753   0.8276
 0.3374   0.4636   0.3591   0.112    0.93     0.822    0.7314   0.01165  0.31     0.5586   0.2808   0.3997   0.04544  0.0931   0.8438   0.612    0.03052  0.3652   0.1153
 0.06213  0.12103  0.4421   0.8003   0.1583   0.845    0.125    0.6934   0.4592   0.871    0.573    0.4133   0.885    0.6875   0.2854   0.7007   0.1294   0.2092   0.3794
 0.7534   0.5923   0.03888  0.2412   0.8584   0.6704   0.429    0.77     0.427    0.6323   0.524    0.0519   0.514    0.2408   0.09357  0.1702   0.3694   0.665    0.2651
 0.9507   0.661    0.459    0.1317   0.7334   0.289    0.0325   0.1187   0.6626   0.2769   0.3083   0.923    0.826    0.7275   0.976    0.4854   0.724    0.7783   0.8022
 0.677    0.2401   0.377    0.839    0.2297   0.54     0.743    0.511    0.1346   0.7183   0.4775   0.3442   0.561    0.2935   0.04065  0.1001   0.753    0.6816   0.8955
 0.07324  0.5947   0.508    0.2229   0.468    0.3135   0.0898   0.5625   0.7407   0.803    0.1071   0.6724   0.797    0.8296   0.807    0.8604   0.7437   0.967    0.4307
 0.3833   0.03394  0.02478  0.9385   0.3105   0.43     0.0706   0.4363   0.05832  0.0812   0.2418   0.03967  0.557    0.2705   0.963    0.8125   0.342    0.8853   0.3047
 0.7197   0.7173   0.02887  0.7695   0.4304   0.691    0.4285   0.9917   0.3994   0.19     0.3984   0.1888   0.83     0.0644   0.9766   0.857    0.09784  0.831    0.224
 0.8228   0.8975   0.1775   0.725    0.882    0.7188   0.3257   0.05347  0.1026   0.05902  0.9697   0.445    0.728    0.626    0.3577   0.711    0.2343   0.3865   0.03888
 0.3318   0.855    0.891    0.3647   0.9297   0.5083   0.7163   0.5737   0.2155   0.804    0.2118   0.525    0.1116   0.558    0.05203  0.6343   0.5796   0.5605   0.449
 0.4475   0.3713   0.3708   0.11017  0.2048   0.087    0.265    0.937    0.933    0.4683   0.5884   0.4312   0.9326   0.839    0.592    0.566    0.4229   0.05493  0.4578
 0.353    0.2915   0.8345   0.888    0.8394   0.8774   0.3582   0.2913   0.798    0.87     0.3372   0.6914   0.9185   0.4368   0.3276   0.8125   0.782    0.885    0.6543
 0.1626   0.0965   0.8247   0.03952  0.459    0.5596   0.694    0.59     0.02153  0.3762   0.2428   0.9727   0.3672   0.732    0.2676   0.2102   0.128    0.5957   0.988
 0.583    0.9097   0.144    0.3845   0.2151   0.327    0.2925   0.974    0.771    0.9224   0.147    0.6206   0.1774   0.1415   0.7637   0.573    0.9736   0.183    0.837
 0.0753   0.098    0.8184   0.08527  0.889    0.528    0.2207   0.1852   0.5903   0.594    0.04865  0.5806   0.6006   0.2048   0.4934   0.1302   0.7217   0.949    0.04105
 0.6875   0.3975   0.845    0.6045   0.4077   0.01927  0.1505   0.4407   0.8457   0.9614   0.4504   0.7134   0.07837  0.3557   0.521    0.545    0.02188  0.581    0.3215
 0.4458   0.853    0.4656   0.928    0.2927   0.3467   0.3516   0.1686   0.88     0.1509   0.2993   0.4006   0.611    0.1251   0.0887   0.896    0.2651   0.5596   0.0359
 0.6895   0.3494   0.871    0.673    0.1486   0.7812   0.0925   0.434    0.09985  0.02402  0.2932   **0.01034**  0.744    0.6357   0.658    0.1487   0.3416   0.1171   0.3088
 0.557    0.837    0.10944  0.7036   0.9097   0.3706   0.73     0.2844   0.78     0.5117   0.5537   0.776    0.6553   0.128    0.3184   0.8022   0.686    0.1785   0.2212
 0.74     0.8955   0.4773   0.6084   0.7827   0.239    0.4849   0.1816   0.2854   0.166    0.012505 0.4421   0.2179   0.06094  0.2124   0.409    0.641    0.1841   0.776
 0.4685   0.2334   0.4094   0.3447   0.6836   0.434    0.10516  0.514    0.8345   0.371    0.8555   0.5396   0.844    0.7554   0.171    0.749    0.7344   0.05936  0.4482
 0.9873   0.3137   0.7627   0.871    0.5503   0.956    0.2607   0.0904   0.535    0.3079   0.762    0.793    0.545    0.889    0.8936   0.6094   0.6533   0.5737   0.945
 0.4434   0.2686   0.05872  0.0776   0.0915   0.5386   0.6777   0.3164   0.8955   0.3398   0.3801   0.3784   0.3904   0.4849   0.816    0.962    0.335    0.705    0.1871
 0.3643   0.7163   0.6484   0.4526   0.8096   0.2408   0.608    0.0215   0.7246   0.412    0.609    0.03342  0.653    0.0424   0.672    0.627    0.3025   0.9424   0.3784
 0.1012   0.4192   0.7695   0.7383   0.9395   0.06494  0.3027   0.11523  0.6035   0.1727   0.4048   0.932    0.4053   0.3528   0.8193   0.0355   0.01953  0.574    0.509
 0.1443   0.0848   0.568    0.8716   0.968    0.613    0.535    0.0389   0.84     0.0655   0.127    0.06104  0.526    0.504    0.4175   0.8027   0.482    0.304   ]
输出数据(dst_gm):
[0.01034,  2.104e-05], 2.104e-05需要使用reinterpret_cast方法转换得到索引值353
```

- tensor前n个数据计算接口完整调用示例：

```
#include "kernel_operator.h"
class KernelReduce {
public:
    __aicore__ inline KernelReduce() {}
    __aicore__ inline void Init(__gm__ uint8_t* src, __gm__ uint8_t* dstGm)
    {
        srcGlobal.SetGlobalBuffer((__gm__ half*)src);
        dstGlobal.SetGlobalBuffer((__gm__ half*)dstGm);
        repeatTime = srcDataSize / mask;
        pipe.InitBuffer(inQueueSrc, 1, srcDataSize * sizeof(half));
        pipe.InitBuffer(workQueue, 1, 32 * sizeof(half)); // 此处按照公式计算所需的最小work空间为32,也就是64Bytes
        pipe.InitBuffer(outQueueDst, 1, dstDataSize * sizeof(half));
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
        AscendC::LocalTensor<half> srcLocal = inQueueSrc.AllocTensor<half>();
        AscendC::DataCopy(srcLocal, srcGlobal, srcDataSize);
        inQueueSrc.EnQue(srcLocal);
    }
    __aicore__ inline void Compute()
    {
        AscendC::LocalTensor<half> srcLocal = inQueueSrc.DeQue<half>();
        AscendC::LocalTensor<half> dstLocal = outQueueDst.AllocTensor<half>();
        AscendC::LocalTensor<half> sharedTmpBuffer = workQueue.AllocTensor<half>();

        // level2
        AscendC::ReduceMin<half>(dstLocal, srcLocal, sharedTmpBuffer, srcDataSize, true);
        outQueueDst.EnQue<half>(dstLocal);
        inQueueSrc.FreeTensor(srcLocal);
        workQueue.FreeTensor(sharedTmpBuffer);
    }
    __aicore__ inline void CopyOut()
    {
        AscendC::LocalTensor<half> dstLocal = outQueueDst.DeQue<half>();
        AscendC::DataCopy(dstGlobal, dstLocal, dstDataSize);
        outQueueDst.FreeTensor(dstLocal);
    }
private:
    AscendC::TPipe pipe;
    AscendC::TQue<AscendC::TPosition::VECIN, 1> inQueueSrc;
    AscendC::TQue<AscendC::TPosition::VECOUT, 1> workQueue;
    AscendC::TQue<AscendC::TPosition::VECOUT, 1> outQueueDst;
    AscendC::GlobalTensor<half> srcGlobal, dstGlobal;
    int srcDataSize = 288;
    int dstDataSize = 16;
    int mask = 128;
    int repStride = 8;
    int repeatTime = 0;
};
extern "C" __global__ __aicore__ void kernel_ReduceMin_lv2_half_288(__gm__ uint8_t* src, __gm__ uint8_t* dstGm)
{
    KernelReduce op;
    op.Init(src, dstGm);
    op.Process();
}
```

示例结果如下：

```
示例结果
输入数据(src_gm):
[0.556    0.5225   0.3623   0.214    0.556    0.0643   0.769    0.594    0.261    0.3652   0.911    0.924    0.386    0.3696   0.2296   0.5957   0.1709   0.79     0.8516
 0.341    0.705    0.728    0.8135   0.7534   0.5874   0.771    0.05835  0.7456   0.1049   0.3105   0.1729   0.9253   0.8003   0.918    0.5005   0.7744   0.688    0.6807
 0.1456   0.4136   0.1055   0.12054  0.275    0.3848   0.08405  0.3843   0.3218   0.6904   0.878    0.3706   0.3586   0.3518   0.429    0.7275   0.6123   0.8096   0.563
 0.54     0.8857   0.8594   0.4143   0.525    0.2744   0.1376   0.382    0.6406   0.1534   0.134    0.2993   0.365    0.8843   0.2986   **0.00393**  0.6577   0.313    0.8164
 0.8706   0.7686   0.873    0.3286   0.03787  0.8145   0.4656   0.66     0.1362   0.1075   0.1376   0.9097   0.9214   0.833    0.3657   0.8438   0.006973 0.2408   0.801
 0.1862   0.864    0.8745   0.1805   0.4324   0.8647   0.844    0.8936   0.8496   0.311    0.0334   0.3967   0.579    0.43     0.2332   0.5366   0.3557   0.3542   0.945
 0.9336   0.252    0.4375   0.9727   0.859    0.6294   0.6787   0.8887   0.1884   0.524    0.787    0.04755  0.3984   0.0508   0.4065   0.716    0.3184   0.21     0.10645
 0.7544   0.2827   0.7856   0.4878   0.5903   0.12146  0.6426   0.8438   0.063    0.7617   0.6396   0.1995   0.6475   0.1464   0.7617   0.514    0.3506   0.2708   0.8643
 0.1204   0.04337  0.21     0.528    0.0644   0.2133   0.0643   0.0125   0.602    0.654    0.866    0.225    0.9473   0.408    0.4597   0.2793   0.11145  0.293    0.04156
 0.7705   0.3555   0.3977   0.7485   0.76     0.9824   0.2832   0.1239   0.4915   0.878    0.5986   0.7217   0.832    0.6206   0.6455   0.0639   0.772    0.01854  0.7437
 0.1962   0.485    0.5483   0.414    0.9253   0.2452   0.2942   0.9478   0.879    0.586    0.659    0.635    0.7197   0.933    0.08905  0.02892  0.74     0.499    0.02054
 0.2241   0.5137   0.8325   0.185    0.6196   0.949    0.935    0.5605   0.04108  0.3672   0.5566   0.3958   0.4565   0.8135   0.3015   0.46     0.1196   0.5044   0.54
 0.05203  0.687    0.8525   0.501    0.3464   0.307    0.804    0.0926   0.202    0.999    0.955    0.581    0.06216  0.271    0.9365   0.854    0.4202   0.269    0.985
 0.04547  1.       0.1208   0.5225   0.00935  0.4128   0.644    0.3826   0.6963   0.2942   0.007626 0.7144   0.609    0.3206   0.694    0.393    0.6265   0.6904   0.2487
 0.9478   0.798    0.891    0.8867   0.9414   0.395    0.11285  0.515    0.919    0.013855 0.749    0.5527   0.465    0.451    0.1458   0.59     0.893    0.0146   0.062
 0.06604  0.934    0.2242  ]
输出数据(dst_gm):
[0.00393,  4.3e-06], 4.3e-06需要使用reinterpret_cast方法转换得到索引值72
```
