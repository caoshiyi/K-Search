# Sort32

**页面ID:** atlasascendc_api_07_0231  
**来源:** https://www.hiascend.com/document/detail/zh/CANNCommunityEdition/850/API/ascendcopapi/atlasascendc_api_07_0231.html

---

#### 产品支持情况

| 产品 | 是否支持 |
| --- | --- |
| Atlas A3 训练系列产品            /             Atlas A3 推理系列产品 | √ |
| Atlas A2 训练系列产品            /             Atlas A2 推理系列产品 | √ |
| Atlas 200I/500 A2 推理产品 | √ |
| Atlas 推理系列产品            AI Core | x |
| Atlas 推理系列产品            Vector Core | x |
| Atlas 训练系列产品 | x |

#### 功能说明

排序函数，一次迭代可以完成32个数的排序，数据需要按如下描述结构进行保存：

score和index分别存储在src0和src1中，按score进行排序（score大的排前面），排序好的score与其对应的index一起以（score, index）的结构存储在dst中。不论score为half还是float类型，dst中的（score, index）结构总是占据8Bytes空间。

如下所示：

- 当score为float，index为uint32_t类型时，计算结果中index存储在高4Bytes，score存储在低4Bytes。

<!-- img2text -->
```
score
┌──────────────────────┐
│       score[0]       │
├──────────────────────┤
│       score[1]       │
├──────────────────────┤
│       score[2]       │
├──────────────────────┤
│          ..          │
├──────────────────────┤
│      score[30]       │
├──────────────────────┤
│      score[31]       │
└──────────────────────┘
<──────────────────────>
          4B

index
┌──────────────────────┐
│       index[0]       │
├──────────────────────┤
│       index[1]       │
├──────────────────────┤
│       index[2]       │
├──────────────────────┤
│          ..          │
├──────────────────────┤
│      index[30]       │
├──────────────────────┤
│      index[31]       │
└──────────────────────┘
<──────────────────────>
          4B

                ─────────────→

result
┌──────────────────────────────┬──────────────────────────────┐
│           Score[5]           │           index[5]           │
├──────────────────────────────┼──────────────────────────────┤
│          score[11]           │          index[11]           │
├──────────────────────────────┼──────────────────────────────┤
│          score[20]           │          index[20]           │
├──────────────────────────────┼──────────────────────────────┤
│              ...             │              ...             │
├──────────────────────────────┼──────────────────────────────┤
│           score[1]           │           index[1]           │
├──────────────────────────────┼──────────────────────────────┤
│           score[8]           │           index[8]           │
└──────────────────────────────┴──────────────────────────────┘
<─────────────────────────────────────────────────────────────>
                              8B

score[0] ↓
score[31] ↓
index[0] ↓
index[31] ↓

score/index 重新排列后对应到 result 中的配对结果：
score[5]  ───────────────────────────────────────────────→ index[5]
score[8]  ───────────────────────────────────────────────→ index[8]
```

说明:
- 左侧两个 4B 列分别为 score 和 index。
- 右侧 8B 结果中，每一行由两部分组成：左侧存储 score，右侧存储 index。
- 图中展示的结果顺序为：`[5]、[11]、[20]、...、[1]、[8]`。
- 根据上下文，此图对应 `score=float`、`index=uint32_t` 时的存储方式：高4Bytes为 `index`，低4Bytes为 `score`。

- 当score为half，index为uint32_t类型时，计算结果中index存储在高4Bytes，score存储在低2Bytes， 中间的2Bytes保留。

<!-- img2text -->
```


┌──────────────┐              ┌──────────────┐                               ┌───────────────────────────────────────┐
│   score[0]   │              │   index[0]   │                               │  Score[5]  │ reserved │   index[5]    │
├──────────────┤              ├──────────────┤                               ├────────────┼──────────┼───────────────┤
│   score[1]   │              │   index[1]   │                               │ score[11]  │ reserved │  index[11]    │
├──────────────┤              ├──────────────┤                               ├────────────┼──────────┼───────────────┤
│   score[2]   │              │   index[2]   │              ┌───────────┐    │ score[20]  │ reserved │  index[20]    │
├──────────────┤              ├──────────────┤              │           │    ├────────────┼──────────┼───────────────┤
│      ...     │              │      ...     │              │     →     │    │     ...    │   ...    │      ...      │
├──────────────┤              ├──────────────┤              │           │    ├────────────┼──────────┼───────────────┤
│  score[30]   │              │  index[30]   │              └───────────┘    │  score[1]  │ reserved │   index[1]    │
├──────────────┤              ├──────────────┤                               ├────────────┼──────────┼───────────────┤
│  score[31]   │              │  index[31]   │                               │  score[8]  │ reserved │   index[8]    │
└──────────────┘              └──────────────┘                               └───────────────────────────────────────┘
      ↓                              ↓                                              ↑                           ↓
      ↓                              ↓                                              └───────────────┬───────────┘
      ↓                              ↓                                                              │
    2B                               4B                                                             8B


score 列内部顺序:
score[0]
score[1]
score[2]
...
score[30]
score[31]

index 列内部顺序:
index[0]
index[1]
index[2]
...
index[30]
index[31]

合并后每项布局:
┌────────────┬──────────┬───────────────┐
│   score    │ reserved │     index     │
└────────────┴──────────┴───────────────┘
     2B          2B            4B

示意映射:
score[5]  ─────────────────────────────────────────────→ Score[5]
index[5]  ─────────────────────────────────────────────→ index[5]

score[8]  ─────────────────────────────────────────────→ score[8]
index[8]  ─────────────────────────────────────────────→ index[8]
```

说明:
- 左侧 score 每个元素大小为 2B。
- 中间 index 每个元素大小为 4B。
- 右侧合并结果每个元素大小为 8B，布局为：低 2Bytes 存 score，中间 2Bytes reserved，高 4Bytes 存 index。
- 右侧示例显示的是重排后的部分元素顺序：Score[5]、score[11]、score[20]、...、score[1]、score[8]，对应 index[5]、index[11]、index[20]、...、index[1]、index[8]。

#### 函数原型

```
template <typename T>
__aicore__ inline void Sort32(const LocalTensor<T>& dst, const LocalTensor<T>& src0, const LocalTensor<uint32_t>& src1, const int32_t repeatTime)
```

#### 参数说明

**表1 **模板参数说明

| 参数名 | 描述 |
| --- | --- |
| T | 操作数数据类型。                       Atlas A3 训练系列产品            /             Atlas A3 推理系列产品            ，支持的数据类型为：half/float                       Atlas A2 训练系列产品            /             Atlas A2 推理系列产品            ，支持的数据类型为：half/float                       Atlas 200I/500 A2 推理产品            ，支持的数据类型为：half/float |

**表2 **参数说明

| 参数名称 | 输入/输出 | 含义 |
| --- | --- | --- |
| dst | 输出 | 目的操作数。          类型为LocalTensor，支持的TPosition为VECIN/VECCALC/VECOUT。          LocalTensor的起始地址需要32字节对齐。 |
| src0 | 输入 | 源操作数。          类型为LocalTensor，支持的TPosition为VECIN/VECCALC/VECOUT。          LocalTensor的起始地址需要32字节对齐。          此源操作数的数据类型需要与目的操作数保持一致。 |
| src1 | 输入 | 源操作数。          类型为LocalTensor，支持的TPosition为VECIN/VECCALC/VECOUT。          LocalTensor的起始地址需要32字节对齐。          此源操作数固定为uint32_t数据类型。 |
| repeatTime | 输入 | 重复迭代次数，int32_t类型，每次迭代完成32个元素的排序，下次迭代src0和src1各跳过32个elements，dst跳过32*8 Byte空间。取值范围：repeatTime∈[0,255]。 |

#### 约束说明

- 当存在score[i]与score[j]相同时，如果i>j，则score[j]将首先被选出来，排在前面。
- 每次迭代内的数据会进行排序，不同迭代间的数据不会进行排序。

#### 调用示例

- 接口使用样例

```
// repeatTime = 4, 对128个数分成4组进行排序，每次完成1组32个数的排序
AscendC::Sort32<float>(dstLocal, srcLocal0, srcLocal1, 4);
```

- 完整样例

```
#include "kernel_operator.h"

class KernelSort32 {
public:
    __aicore__ inline KernelSort32() {}
    __aicore__ inline void Init(__gm__ uint8_t* src0Gm, __gm__ uint8_t* src1Gm, __gm__ uint8_t* dstGm)
    {
        srcGlobal0.SetGlobalBuffer((__gm__ float*)src0Gm);
        srcGlobal1.SetGlobalBuffer((__gm__ uint32_t*)src1Gm);
        dstGlobal.SetGlobalBuffer((__gm__ float*)dstGm);

        repeat = srcDataSize / 32;
        pipe.InitBuffer(inQueueSrc0, 1, srcDataSize * sizeof(float));
        pipe.InitBuffer(inQueueSrc1, 1, srcDataSize * sizeof(uint32_t));
        pipe.InitBuffer(outQueueDst, 1, dstDataSize * sizeof(float));
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
        AscendC::LocalTensor<float> srcLocal0 = inQueueSrc0.AllocTensor<float>();
        AscendC::DataCopy(srcLocal0, srcGlobal0, srcDataSize);
        inQueueSrc0.EnQue(srcLocal0);
        AscendC::LocalTensor<uint32_t> srcLocal1 = inQueueSrc1.AllocTensor<uint32_t>();
        AscendC::DataCopy(srcLocal1, srcGlobal1, srcDataSize);
        inQueueSrc1.EnQue(srcLocal1);
    }
    __aicore__ inline void Compute()
    {
        AscendC::LocalTensor<float> srcLocal0 = inQueueSrc0.DeQue<float>();
        AscendC::LocalTensor<uint32_t> srcLocal1 = inQueueSrc1.DeQue<uint32_t>();
        AscendC::LocalTensor<float> dstLocal = outQueueDst.AllocTensor<float>();

        AscendC::Sort32<float>(dstLocal, srcLocal0, srcLocal1, repeat);

        outQueueDst.EnQue<float>(dstLocal);
        inQueueSrc0.FreeTensor(srcLocal0);
        inQueueSrc1.FreeTensor(srcLocal1);
    }
    __aicore__ inline void CopyOut()
    {
        AscendC::LocalTensor<float> dstLocal = outQueueDst.DeQue<float>();
        AscendC::DataCopy(dstGlobal, dstLocal, dstDataSize);
        outQueueDst.FreeTensor(dstLocal);
    }

private:
    AscendC::TPipe pipe;
    AscendC::TQue<AscendC::TPosition::VECIN, 1> inQueueSrc0;
    AscendC::TQue<AscendC::TPosition::VECIN, 1> inQueueSrc1;
    AscendC::TQue<AscendC::TPosition::VECOUT, 1> outQueueDst;
    AscendC::GlobalTensor<float> srcGlobal0, dstGlobal;
    AscendC::GlobalTensor<uint32_t> srcGlobal1;
    int srcDataSize = 128;
    int dstDataSize = 256;
    int repeat = 0;
};

extern "C" __global__ __aicore__ void vec_sort32_kernel(__gm__ uint8_t* src0Gm, __gm__ uint8_t* src1Gm,
    __gm__ uint8_t* dstGm)
{
    KernelSort32 op;
    op.Init(src0Gm, src1Gm, dstGm);
    op.Process();
}
```

```
示例结果
输入数据src0Gm：128个float类型数据
[7.867878  9.065992  9.374247  1.0911566 9.262053  2.035779  3.747487
 2.9315646 5.237765  5.176559  7.965426  3.2341435 7.203623  1.5736973
 3.386001  5.077001  4.593656  1.8485032 7.8554387 5.1269145 7.223478
 8.259627  5.5502934 8.795028  9.626377  7.7227993 9.505127  6.683293
 6.232041  2.1760664 4.504409  2.906819  9.425597  9.467169  4.990563
 4.609341  1.8662999 3.6319377 3.5542917 8.382838  5.133566  3.1391478
 5.244712  9.330158  2.0394793 5.9761605 4.937267  6.076068  7.5449195
 6.5085726 1.8132887 2.5047603 3.3350103 2.7831945 3.0417829 5.0608244
 3.4855423 2.8485715 4.853921  6.364753  3.1402998 6.052516  3.6143537
 4.0714087 6.8068676 8.625871  8.040528  1.9881475 4.618402  7.0302424
 6.0751796 5.877218  9.256125  4.193431  5.2048235 6.9774013 2.8765092
 5.8294353 8.618196  8.619784  3.9252923 4.491909  6.0063663 2.3781579
 5.8828945 7.269731  6.1864734 8.32413   5.2518435 9.184813  7.9312286
 3.8841062 8.540505  7.611145  8.204335  2.110103  4.1796618 7.2383223
 3.9992998 4.750733  8.650443  7.6469994 6.6126637 8.993322  8.920976
 7.143699  7.0797443 3.3189814 7.3707795 3.26992   8.58087   5.6882014
 2.0333889 6.711474  4.353861  7.946233  4.5678067 6.3354545 4.092168
 2.416961  3.6823056 4.6000533 2.4727547 4.7993317 1.159995  8.025275
 3.3826146 3.8543346]
输入数据src1Gm：
[0,0,0,0,0...0]
输出数据dstGm：
[9.626377  0.        9.505127  0.        9.374247  0.        9.262053
 0.        9.065992  0.        8.795028  0.        8.259627  0.
 7.965426  0.        7.867878  0.        7.8554387 0.        7.7227993
 0.        7.223478  0.        7.203623  0.        6.683293  0.
 6.232041  0.        5.5502934 0.        5.237765  0.        5.176559
 0.        5.1269145 0.        5.077001  0.        4.593656  0.
 4.504409  0.        3.747487  0.        3.386001  0.        3.2341435
 0.        2.9315646 0.        2.906819  0.        2.1760664 0.
 2.035779  0.        1.8485032 0.        1.5736973 0.        1.0911566
 0.        9.467169  0.        9.425597  0.        9.330158  0.
 8.382838  0.        7.5449195 0.        6.5085726 0.        6.364753
 0.        6.076068  0.        6.052516  0.        5.9761605 0.
 5.244712  0.        5.133566  0.        5.0608244 0.        4.990563
 0.        4.937267  0.        4.853921  0.        4.609341  0.
 4.0714087 0.        3.6319377 0.        3.6143537 0.        3.5542917
 0.        3.4855423 0.        3.3350103 0.        3.1402998 0.
 3.1391478 0.        3.0417829 0.        2.8485715 0.        2.7831945
 0.        2.5047603 0.        2.0394793 0.        1.8662999 0.
 1.8132887 0.        9.256125  0.        9.184813  0.        8.625871
 0.        8.619784  0.        8.618196  0.        8.540505  0.
 8.32413   0.        8.204335  0.        8.040528  0.        7.9312286
 0.        7.611145  0.        7.269731  0.        7.0302424 0.
 6.9774013 0.        6.8068676 0.        6.1864734 0.        6.0751796
 0.        6.0063663 0.        5.8828945 0.        5.877218  0.
 5.8294353 0.        5.2518435 0.        5.2048235 0.        4.618402
 0.        4.491909  0.        4.193431  0.        3.9252923 0.
 3.8841062 0.        2.8765092 0.        2.3781579 0.        2.110103
 0.        1.9881475 0.        8.993322  0.        8.920976  0.
 8.650443  0.        8.58087   0.        8.025275  0.        7.946233
 0.        7.6469994 0.        7.3707795 0.        7.2383223 0.
 7.143699  0.        7.0797443 0.        6.711474  0.        6.6126637
 0.        6.3354545 0.        5.6882014 0.        4.7993317 0.
 4.750733  0.        4.6000533 0.        4.5678067 0.        4.353861
 0.        4.1796618 0.        4.092168  0.        3.9992998 0.
 3.8543346 0.        3.6823056 0.        3.3826146 0.        3.3189814
 0.        3.26992   0.        2.4727547 0.        2.416961  0.
 2.0333889 0.        1.159995  0.       ]
```
