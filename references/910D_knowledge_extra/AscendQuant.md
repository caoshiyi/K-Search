# AscendQuant<a name="ZH-CN_TOPIC_0000002554424681"></a>

## 产品支持情况<a name="section1586581915393"></a>

<a name="table169596713360"></a>
<table><thead align="left"><tr id="row129590715369"><th class="cellrowborder" valign="top" width="57.99999999999999%" id="mcps1.1.3.1.1"><p id="p17959971362"><a name="p17959971362"></a><a name="p17959971362"></a><span id="ph895914718367"><a name="ph895914718367"></a><a name="ph895914718367"></a>产品</span></p>
</th>
<th class="cellrowborder" align="center" valign="top" width="42%" id="mcps1.1.3.1.2"><p id="p89594763612"><a name="p89594763612"></a><a name="p89594763612"></a>是否支持</p>
</th>
</tr>
</thead>
<tbody><tr id="row18959673369"><td class="cellrowborder" valign="top" width="57.99999999999999%" headers="mcps1.1.3.1.1 "><p id="p1595910763613"><a name="p1595910763613"></a><a name="p1595910763613"></a><span id="ph1595918753613"><a name="ph1595918753613"></a><a name="ph1595918753613"></a>Ascend 950PR/Ascend 950DT</span></p>
</td>
<td class="cellrowborder" align="center" valign="top" width="42%" headers="mcps1.1.3.1.2 "><p id="p1695957133611"><a name="p1695957133611"></a><a name="p1695957133611"></a>√</p>
</td>
</tr>
</tbody>
</table>

## 功能说明<a name="section618mcpsimp"></a>

按元素做量化计算，比如将half/float数据类型量化为int8\_t数据类型。计算公式如下，cast表示舍入模式：

-   PER\_TENSOR量化：整个srcTensor对应一个量化参数，量化参数的shape为\[1\]。

    <!-- img2text -->
$$dstTensor = cast\left(\frac{srcTensor}{scale} + offset\right)$$

-   PER\_CHANNEL量化：srcTensor的shape为\[m, n\], 每个channel维度对应一个量化参数，量化参数的shape为\[n\]。

    <!-- img2text -->
$$dstTensor_{ij}=\operatorname{round}\left(\frac{srcTensor_{ij}}{scale_j}\right)-offset_j,\quad i\in[0,m),\ j\in[0,n)$$

-   PER\_TOKEN量化：srcTensor的每组token（token为n方向，共有m组token）中的元素共享一组scale和offset参数，srcTensor的shape为\[m, n\]时，scale和offset的shape为\[m, 1\]。offset是可选输入。
-   <!-- img2text -->
$$
\begin{aligned}
dstTensor(i, j) &= \operatorname{cast}\left(\left(srcTensor(i, j) \times scale(j)\right) + offset(j)\right) \\
&\quad i \in [0, m),\ j \in [0, n)
\end{aligned}
$$
-   PER\_GROUP量化：这里定义group的计算方向为k方向，srcTensor在k方向上每groupSize个元素共享一组scale和offset。srcTensor的shape为\[m, n\]时，如果kDim=0，表示k是m方向，scale和offset的shape为\[\(m + groupSize - 1\) / groupSize, n\]；如果kDim=1，表示k是n方向，scale和offset的shape为\[m，\(n + groupSize - 1\) / groupSize\]。offset是可选输入。

    根据输出数据类型的不同，当前PER\_GROUP分为两种场景：fp4x2\_e2m1\_t/fp4x2\_e1m2\_t场景（后续内容中简称为float4场景）和int8\_t/hifloat8\_t/fp8\_e5m2\_t/fp8\_e4m3fn\_t场景（后续内容中简称为b8场景）。

    -   fp4x2\_e2m1\_t/float4\_e1m2场景（float4场景）
        -   kDim = 0:****

            <!-- img2text -->
$$\text{groupNum} = \left\lceil \frac{\text{srcShape}.K \cdot \operatorname{sizeof}(\text{srcType})}{4 \cdot 8 \cdot \text{GROUP\_SIZE}} \right\rceil$$

        -   kDim = 1:

            <!-- img2text -->
$$
\text{dstShape}=(k,\ m,\ n)
$$

    -   int8\_t/hifloat8\_t/fp8\_e5m2\_t/fp8\_e4m3fn\_t场景（b8场景）
        -   kDim=0：

            <!-- img2text -->
$$
srcLocal[x_i] = srcGlobal[(srcOffset_0 + x_i) \times \cdots \times srcOffset_n], \quad 0 \le x_i < count
$$

        -   kDim=1：

            <!-- img2text -->
$$
\text{quantData}_{i,j}=\operatorname{round}\left(\text{input}_{i,j}\times \text{scale}_{j}+\text{offset}_{j}\right)
$$

## 实现原理<a name="section13229175017585"></a>

**图 1**  AscendQuant算法框图scale和offset都是scalar<a name="fig966236152318"></a>  
<!-- img2text -->
```
                                  ┌──────────────┐                                     ┌──────────────┐
                                  │ src_local[n] │                                     │ src_local[n] │
                                  └──────┬───────┘                                     └──────┬───────┘
                                         │                                                      │
                                   half input                                            float input
                                         │                                                      │
                     ┌───────────────────┴───────────────────┐              ┌───────────────────┴───────────────────┐
                     │                                       │              │                                       │
┌──────────┐         │      ┌──────────────────────────┐     │              │      ┌──────────────────────────┐     │
│ scale[1] │ ───────→│      │ Tmp1[n]=Muls(src_local,  │     │              │      │ inputHalf[n]=Cast(src_loca│     │
└──────────┘         │      │          scale)          │     │              │      │ l, float->half)          │     │
                     │      └────────────┬─────────────┘     │              │      └────────────┬─────────────┘     │
                     │                   │                   │              │                   │                   │
┌──────────┐         │      ┌────────────▼─────────────┐     │              │      ┌────────────▼─────────────┐     │
│ offset[1]│ ───────→│      │ Tmp2[n]=Adds(Tmp1,       │     │   ┌──────────┐      │ Tmp1[n]=Muls(inputHalf,   │     │
└──────────┘         │      │          offset)         │     │   │ scale[1] │ ───→ │          scale)          │     │
                     │      └────────────┬─────────────┘     │   └──────────┘      └────────────┬─────────────┘     │
                     │                   │                   │              │                   │                   │
                     │      ┌────────────▼─────────────┐     │              │      ┌────────────▼─────────────┐     │
                     │      │ dst_local[n]=Cast(Tmp2,  │     │   ┌──────────┐      │ Tmp2[n]=Adds(Tmp1,       │     │
                     │      │      half->int8_t)       │     │   │offset[1] │ ───→ │          offset)         │     │
                     │      └────────────┬─────────────┘     │   └──────────┘      └────────────┬─────────────┘     │
                     └───────────────────┼───────────────────┘              │                   │                   │
                                         │                                  │      ┌────────────▼─────────────┐     │
                                         │                                  │      │ dst_local[n]=Cast(Tmp2,  │     │
                                  ┌──────▼───────┐                          │      │      half->int8_t)       │     │
                                  │ dst_local[n] │                          │      └────────────┬─────────────┘     │
                                  └──────────────┘                          └───────────────────┼───────────────────┘
                                                                                                 │
                                                                                          ┌──────▼───────┐
                                                                                          │ dst_local[n] │
                                                                                          └──────────────┘


图示:
输入输出Tensor/Scalar    ┌──────────┐
                        │          │
                        └──────────┘

vector计算             ┌──────────────────────────┐
                       │                          │
                       └──────────────────────────┘

数据流向               ─────────→
```

**图 2**  AscendQuant算法框图scale和offset都是Tensor<a name="fig2405134711019"></a>  
<!-- img2text -->
```
┌───────────────────────────────────────────────────────┐     ┌──────────────────────────────────────────────────────────────┐
│                                                       │     │                                                              │
│     ┌───────────┐                                     │     │             ┌───────────┐                                    │
│     │ scale[n]  │                                     │     │             │ scale[n]  │                                    │
│     └─────┬─────┘                                     │     │             └─────┬─────┘                                    │
│           │                                           │     │                   │                                          │
│           ↓                                           │     │                   ↓                                          │
│   ┌──────────────────────────┐                        │     │   ┌──────────────────────────┐                               │
│   │ scale[m, n]              │                        │     │   │ scale[m, n]              │                               │
│   │ =broadcast(scale[n])     │───→┐                  │     │   │ =broadcast(scale[n])     │───→┐                         │
│   └──────────────────────────┘   │                  │     │   └──────────────────────────┘   │                         │
│                                  │                  │     │                                  │                         │
│                      ┌───────────▼────────────┐     │     │     ┌───────────────┐            │                         │
│     ┌──────────────┐ │ Tmp1[m, n] =           │     │     │     │ src_local[m, n]│            │                         │
│     │ src_local[m, n]│ │ Mul(src_local, scale)│     │     │     └───────┬───────┘            │                         │
│     └──────┬───────┘ └───────────┬────────────┘     │     │             │                    │                         │
│            │                     │                  │     │       float input                │                         │
│       half input                 ↓                  │     │             ↓                    │                         │
│                                  │                  │     │   ┌──────────────────────────┐   │                         │
│   ┌──────────────┐     ┌─────────▼────────────┐     │     │   │ inputHalf[m, n]          │   │                         │
│   │ offset[n]    │     │ Tmp2[m, n] =         │     │     │   │ =Cast(src_local, float->half)│ │                         │
│   └──────┬───────┘     │ Add(Tmp1, offset)    │     │     │   └───────────┬──────────────┘   │                         │
│          │             └─────────┬────────────┘     │     │               │                  │                         │
│          │                       │                  │     │               ↓                  │                         │
│          ↓                       ↓                  │     │      ┌────────▼─────────────┐    │                         │
│   ┌──────────────────────────┐  ┌──────────────────┐│     │      │ Tmp1[m, n] =          │    │                         │
│   │ offset[m, n]             │──→│ dst_local[m, n] ││     │      │ =Mul(inputHalf, scale)│    │                         │
│   │ =broadcast(offset[n])    │   │ =Cast(Tmp2,     ││     │      └────────┬──────────────┘    │                         │
│   └──────────────────────────┘   │ half->int8_t)   ││     │               ↓                   │                         │
│                                  └────────┬────────┘│     │      ┌────────▼─────────────┐    │                         │
│                                           │         │     │      │ Tmp2[m, n] =          │    │                         │
│                                           ↓         │     │      │ =Add(Tmp1, offset)    │    │                         │
│                                  ┌────────────────┐ │     │      └────────┬──────────────┘    │                         │
│                                  │ dst_local[m, n]│ │     │               ↓                   │                         │
│                                  └────────────────┘ │     │      ┌────────▼─────────────┐    │                         │
│                                                       │     │      │ dst_local[m, n]      │    │                         │
└───────────────────────────────────────────────────────┘     │      │ =Cast(Tmp2,          │    │                         │
                                                              │      │ half->int8_t)        │    │                         │
                                                              │      └────────┬─────────────┘    │                         │
                                                              │               │                  │                         │
                                                              │               ↓                  │                         │
                                                              │      ┌────────────────┐          │                         │
                                                              │      │ dst_local[m, n]│          │                         │
                                                              │      └────────────────┘          │                         │
                                                              │                                  │                         │
                                                              │   ┌──────────────────────────┐   │                         │
                                                              │   │ offset[m, n]             │──→│                         │
                                                              │   │ =broadcast(offset[n])    │   │                         │
                                                              │   └──────────┬───────────────┘   │                         │
                                                              │              ↑                   │                         │
                                                              │              │                   │                         │
                                                              │        ┌─────┴─────┐             │                         │
                                                              │        │ offset[n] │             │                         │
                                                              │        └───────────┘             │                         │
                                                              └──────────────────────────────────┘                         │
                                                                                                                           │
                                                                                                                           │
                                                                                     图示：                                │
                                                                                     输入输出Tensor/Scalar  ┌───────────┐   │
                                                                                                             │           │   │
                                                                                                             └───────────┘   │
                                                                                     vector计算            ┌───────────┐   │
                                                                                                             │           │   │
                                                                                                             └───────────┘   │
                                                                                     数据流向                    ─────→     │
```

**图 3**  AscendQuant算法框图scale是Tensor&offset是Scalar<a name="fig6542182812108"></a>  
<!-- img2text -->
```
图 3  AscendQuant算法框图scale是Tensor&offset是Scalar

┌───────────────┐                      ┌────────────────┐
│   scale[n]    │                      │ src_local[m,n] │
└──────┬────────┘                      └──────┬─────────┘
       │                                      │
       │                                      │  half input
       │                                      │
       │      ┌────────────────────────────────────────────────────┐
       │      │                                                    │
       ▼      │                       ▼                            │
┌──────────────────────────┐   ┌──────────────────────────┐        │
│       scale[m,n]         │ → │       Tmp1[m,n]          │        │
│  =broadcast(scale[n])    │   │ =Mul(src_local, scale)   │        │
└──────────────────────────┘   └──────────┬───────────────┘        │
                                          │                        │
                                          ▼                        │
                                   ┌──────────────────────────┐     │
                                   │       Tmp2[m,n]         │     │
                                   │ =Adds(Tmp1, offset)     │     │
                                   └──────────┬───────────────┘     │
                                              │                     │
                                              ▼                     │
                                   ┌──────────────────────────┐     │
                                   │     dst_local[m,n]       │     │
                                   │ =Cast(Tmp2, half->int8_t)│     │
                                   └──────────┬───────────────┘     │
                                              │                     │
┌───────────────┐                              │              ┌──────▼─────────┐
│   offset[1]   │ ────────────────────────────┘              │ dst_local[m,n] │
└───────────────┘                                             └────────────────┘
                 └──────────────────────────────────────────────────────────────┘


┌───────────────┐                      ┌────────────────┐
│   scale[n]    │                      │ src_local[m,n] │
└──────┬────────┘                      └──────┬─────────┘
       │                                      │
       │                                      │  float input
       │                                      │
       │      ┌────────────────────────────────────────────────────┐
       │      │                                                    │
       │      │                       ▼                            │
       │      │        ┌──────────────────────────────┐            │
       │      │        │      inputHalf[m,n]          │            │
       │      │        │ =Cast(src_local, float->half)│            │
       │      │        └─────────────┬────────────────┘            │
       ▼      │                      │                             │
┌──────────────────────────┐         ▼                             │
│       scale[m,n]         │ → ┌──────────────────────────┐        │
│  =broadcast(scale[n])    │   │       Tmp1[m,n]          │        │
└──────────────────────────┘   │ =Mul(inputHalf, scale)   │        │
                               └──────────┬───────────────┘        │
                                          │                        │
                                          ▼                        │
                                   ┌──────────────────────────┐     │
                                   │       Tmp2[m,n]         │     │
                                   │ =Adds(Tmp1, offset)     │     │
                                   └──────────┬───────────────┘     │
                                              │                     │
                                              ▼                     │
                                   ┌──────────────────────────┐     │
                                   │     dst_local[m,n]       │     │
                                   │ =Cast(Tmp2, half->int8_t)│     │
                                   └──────────┬───────────────┘     │
                                              │                     │
┌───────────────┐                              │              ┌──────▼─────────┐
│   offset[1]   │ ────────────────────────────┘              │ dst_local[m,n] │
└───────────────┘                                             └────────────────┘
                 └──────────────────────────────────────────────────────────────┘


图示:
输入输出Tensor/Scalar   ┌───────────────┐
                        │               │
                        └───────────────┘

vector计算             ┌──────────────────────────┐
                        │                          │
                        └──────────────────────────┘

数据流向               ─────────→
```

如上图所示是AscendQuant内部算法框图，计算过程大致描述为如下几步，均在Vector上进行：

1.  精度转换：当输入的src，scale或者offset是float类型时，将其转换为half类型；
2.  broadcast：当输入的scale或者offset是向量时，将其broadcast成和src相同维度；
3.  计算scale：当src和scale为向量时做Mul计算，当scale是scalar时做Muls计算，得到Tmp1；
4.  计算offset：当Tmp1和offset为向量时做Add计算，当offset是scalar时做Adds计算，得到Tmp2；
5.  精度转换：将Tmp2从half转换成int8\_t类型，得到output。

**图 4**  AscendQuant算法框图PER\_TOKEN/PER\_\_GROUP场景scale和offset都是tensor<a name="fig841775615011"></a>  
<!-- img2text -->
```text
                           ┌──────────┐                    ┌────────────────┐
                           │ scale[m] │                    │ src_local[m, n]│
                           └────┬─────┘                    └───────┬────────┘
                                │                                  │
                                │                                  ↓
                                │                            ┌────────────┐
                                │                            │  读取src   │
                                │                            └────┬───────┘
                                │                                 │
                                ↓                                 ↓
                         ┌────────────┐                  ┌──────────────────────┐
                         │  读取scale │                  │      src_vreg =      │
                         └────┬───────┘                  │ Cast(ori_src_vreg)   │
                              │                          └─────────┬────────────┘
                              │                                    │
                              ↓                                    ↓
                     ┌──────────────────────┐             ┌──────────────────────┐
                     │    scale_vreg =      │ ─────────→  │     temp_vreg=       │
                     │ Cast(ori_scale_vreg) │             │ Mul(src_vreg,        │
                     └──────────────────────┘             │     scale_vreg)      │
                                                          └─────────┬────────────┘
                                                                    │
                                                                    ↓
                                                          ┌──────────────────────┐
                                                          │    temp2_vreg=       │ ←──────── ┌──────────────────────┐
                                                          │ Add(temp_vreg,       │           │    offset_vreg =     │
                                                          │     offset_vreg)     │           │ Cast(ori_offset_vreg)│
                                                          └─────────┬────────────┘           └─────────┬────────────┘
                                                                    │                                  ↑
                                                                    │                                  │
                                                                    ↓                            ┌────────────┐
                                                          ┌──────────────────────┐              │  读取offset│
                                                          │      dst_vreg =      │              └────┬───────┘
                                                          │   Cast(temp2_vreg)   │                   │
                                                          └─────────┬────────────┘                   │
                                                                    │                                │
                                                                    ↓                                │
                                                              ┌───────────────┐               ┌──────────┐
                                                              │ dst_local[m, n]│               │ offset[n]│
                                                              └───────────────┘               └──────────┘
```

**图 5**  AscendQuant算法框图PER\_TOKEN/PER\_\_GROUP场景scale是tensor&offset是scalar<a name="fig1452882415581"></a>  
<!-- img2text -->
```text
┌──────────────────────────────────────────────────────────────────────────────────────────────┐
│                                                                                              │
│   ┌───────────────┐                               ┌────────────────┐                         │
│   │   scale[m]    │                               │ src_local[m, n]│                         │
│   └──────┬────────┘                               └───────┬────────┘                         │
│          │                                                │                                  │
│          ↓                                                ↓                                  │
│   ┌────────────────┐                             ┌────────────────┐                         │
│   │    读取scale    │                             │    读取src      │                         │
│   └──────┬─────────┘                             └───────┬────────┘                         │
│          │                                                │                                  │
│          ↓                                                ↓                                  │
│   ┌────────────────────────┐                    ┌────────────────────────┐                  │
│   │      scale_vreg =      │                    │       src_vreg =       │                  │
│   │   Cast(ori_scale_vreg) │                    │   Cast(ori_src_vreg)   │                  │
│   └────────────┬───────────┘                    └────────────┬───────────┘                  │
│                │                                             │                              │
│                └──────────────────────────→ ┌────────────────────────┐                       │
│                                             │      temp_vreg=        │                       │
│                                             │ Mul(src_vreg, scale_vreg)│                      │
│                                             └────────────┬───────────┘                       │
│                                                          │                                   │
│                                                          ↓                                   │
│                                             ┌────────────────────────┐      ┌────────────────────────────┐
│                                             │      temp2_vreg=       │ ←────│        cast_offset =       │
│                                             │ Adds(temp_vreg,        │      │ ToFloat(offset)/           │
│                                             │      cast_offset)      │      │ static_cast(offset)        │
│                                             └────────────┬───────────┘      └────────────┬───────────────┘
│                                                          │                                │               │
│                                                          ↓                                │               │
│                                             ┌────────────────────────┐                     │               │
│                                             │       dst_vreg =       │                     │               │
│                                             │    Cast(temp2_vreg)    │                     │               │
│                                             └────────────┬───────────┘                     │               │
│                                                          │                                 │               │
│                                                          ↓                                 │               │
│                                             ┌────────────────┐                    ┌────────┴───────┐      │
│                                             │ dst_local[m, n]│                    │    offset      │      │
│                                             └────────────────┘                    └────────────────┘      │
│                                                                                              │
└──────────────────────────────────────────────────────────────────────────────────────────────┘
```

PER\_TOKEN/PER\_GROUP场景的计算逻辑如下：

1.  读取数据：连续读取输入src；根据不同的场景，对输入scale和offset，采用不同的读取方式；例如，PER\_TOKEN场景做Broadcast处理，PER\_GROUP场景做Gather处理；
2.  精度转换：根据不同输入的数据类型组合，对src/scale/offset进行相应的数据类型转换；
3.  计算：对类型转换后的数据做乘加操作；
4.  精度转换：将上述乘加操作得到的计算结果转换成dstT类型，得到最终输出。

## 函数原型<a name="section19670529163214"></a>

-   dstTensor为int8\_t数据类型
    -   PER\_TENSOR量化：
        -   通过sharedTmpBuffer入参传入临时空间
            -   源操作数Tensor全部/部分参与计算

                ```
                template <typename T, bool isReuseSource = false, const AscendQuantConfig& config = ASCEND_QUANT_DEFAULT_CFG>
                __aicore__ inline void AscendQuant(const LocalTensor<int8_t>& dstTensor, const LocalTensor<T>& srcTensor, const LocalTensor<uint8_t>& sharedTmpBuffer, const float scale, const float offset, const uint32_t calCount)
                ```

            -   源操作数Tensor全部参与计算

                ```
                template <typename T, bool isReuseSource = false, const AscendQuantConfig& config = ASCEND_QUANT_DEFAULT_CFG>
                __aicore__ inline void AscendQuant(const LocalTensor<int8_t>& dstTensor, const LocalTensor<T>& srcTensor, const LocalTensor<uint8_t>& sharedTmpBuffer, const float scale, const float offset)
                ```

        -   接口框架申请临时空间
            -   源操作数Tensor全部/部分参与计算

                ```
                template <typename T, bool isReuseSource = false, const AscendQuantConfig& config = ASCEND_QUANT_DEFAULT_CFG>
                __aicore__ inline void AscendQuant(const LocalTensor<int8_t>& dstTensor, const LocalTensor<T>& srcTensor, const float scale, const float offset, const uint32_t calCount)
                ```

            -   源操作数Tensor全部参与计算

                ```
                template <typename T, bool isReuseSource = false, const AscendQuantConfig& config = ASCEND_QUANT_DEFAULT_CFG>
                __aicore__ inline void AscendQuant(const LocalTensor<int8_t>& dstTensor, const LocalTensor<T>& srcTensor, const float scale, const float offset)
                ```

    -   PER\_CHANNEL量化：
        -   通过sharedTmpBuffer入参传入临时空间
            -   源操作数Tensor全部/部分参与计算

                ```
                template <typename T, bool isReuseSource = false, const AscendQuantConfig& config = ASCEND_QUANT_DEFAULT_CFG>
                __aicore__ inline void AscendQuant(const LocalTensor<int8_t>& dstTensor, const LocalTensor<T>& srcTensor, const LocalTensor<uint8_t>& sharedTmpBuffer, const LocalTensor<T>& scaleTensor, const T offset, const uint32_t scaleCount, const uint32_t calCount)
                ```

                ```
                template <typename T, bool isReuseSource = false, const AscendQuantConfig& config = ASCEND_QUANT_DEFAULT_CFG>
                __aicore__ inline void AscendQuant(const LocalTensor<int8_t>& dstTensor, const LocalTensor<T>& srcTensor, const LocalTensor<uint8_t>& sharedTmpBuffer, const LocalTensor<T>& scaleTensor, const LocalTensor<T>& offsetTensor, const uint32_t scaleCount, const uint32_t offsetCount, const uint32_t calCount)
                ```

            -   源操作数Tensor全部参与计算

                ```
                template <typename T, bool isReuseSource = false, const AscendQuantConfig& config = ASCEND_QUANT_DEFAULT_CFG>
                __aicore__ inline void AscendQuant(const LocalTensor<int8_t>& dstTensor, const LocalTensor<T>& srcTensor, const LocalTensor<uint8_t>& sharedTmpBuffer, const LocalTensor<T>& scaleTensor, const T offset)
                ```

                ```
                template <typename T, bool isReuseSource = false, const AscendQuantConfig& config = ASCEND_QUANT_DEFAULT_CFG>
                __aicore__ inline void AscendQuant(const LocalTensor<int8_t>& dstTensor, const LocalTensor<T>& srcTensor, const LocalTensor<uint8_t>& sharedTmpBuffer, const LocalTensor<T>& scaleTensor, const LocalTensor<T>& offsetTensor)
                ```

        -   接口框架申请临时空间
            -   源操作数Tensor全部/部分参与计算

                ```
                template <typename T, bool isReuseSource = false, const AscendQuantConfig& config = ASCEND_QUANT_DEFAULT_CFG>
                __aicore__ inline void AscendQuant(const LocalTensor<int8_t>& dstTensor, const LocalTensor<T>& srcTensor, const LocalTensor<T>& scaleTensor, const T offset, const uint32_t scaleCount, const uint32_t calCount)
                ```

                ```
                template <typename T, bool isReuseSource = false, const AscendQuantConfig& config = ASCEND_QUANT_DEFAULT_CFG>
                __aicore__ inline void AscendQuant(const LocalTensor<int8_t>& dstTensor, const LocalTensor<T>& srcTensor, const LocalTensor<T>& scaleTensor, const LocalTensor<T>& offsetTensor, const uint32_t scaleCount, const uint32_t offsetCount, const uint32_t calCount)
                ```

            -   源操作数Tensor全部参与计算

                ```
                template <typename T, bool isReuseSource = false, const AscendQuantConfig& config = ASCEND_QUANT_DEFAULT_CFG>
                __aicore__ inline void AscendQuant(const LocalTensor<int8_t>& dstTensor, const LocalTensor<T>& srcTensor, const LocalTensor<T>& scaleTensor, const T offset)
                ```

                ```
                template <typename T, bool isReuseSource = false, const AscendQuantConfig& config = ASCEND_QUANT_DEFAULT_CFG>
                __aicore__ inline void AscendQuant(const LocalTensor<int8_t>& dstTensor, const LocalTensor<T>& srcTensor, const LocalTensor<T>& scaleTensor, const LocalTensor<T>& offsetTensor)
                ```

-   dstTensor非固定数据类型
    -   PER\_TENSOR量化：
        -   通过sharedTmpBuffer入参传入临时空间
            -   源操作数Tensor全部/部分参与计算

                ```
                template <typename dstT, typename srcT, bool isReuseSource = false>
                __aicore__ inline void AscendQuant(const LocalTensor<dstT>& dstTensor, const LocalTensor<srcT>& srcTensor, const LocalTensor<uint8_t>& sharedTmpBuffer, const float scale, const float offset, const uint32_t calCount)
                ```

            -   源操作数Tensor全部参与计算

                ```
                template <typename dstT, typename srcT, bool isReuseSource = false>
                __aicore__ inline void AscendQuant(const LocalTensor<dstT>& dstTensor, const LocalTensor<srcT>& srcTensor, const LocalTensor<uint8_t>& sharedTmpBuffer, const float scale, const float offset)
                ```

        -   接口框架申请临时空间
            -   源操作数Tensor全部/部分参与计算

                ```
                template <typename dstT, typename srcT, bool isReuseSource = false>
                __aicore__ inline void AscendQuant(const LocalTensor<dstT>& dstTensor, const LocalTensor<srcT>& srcTensor, const float scale, const float offset, const uint32_t calCount)
                ```

            -   源操作数Tensor全部参与计算

                ```
                template <typename dstT, typename srcT, bool isReuseSource = false>
                __aicore__ inline void AscendQuant(const LocalTensor<dstT>& dstTensor, const LocalTensor<srcT>& srcTensor, const float scale, const float offset)
                ```

    -   PER\_CHANNEL量化：
        -   通过sharedTmpBuffer入参传入临时空间
            -   源操作数Tensor全部/部分参与计算

                ```
                template <typename dstT, typename srcT, bool isReuseSource = false>
                __aicore__ inline void AscendQuant(const LocalTensor<dstT>& dstTensor, const LocalTensor<srcT>& srcTensor, const LocalTensor<uint8_t>& sharedTmpBuffer, const LocalTensor<srcT>& scaleTensor, const srcT offset, const uint32_t scaleCount, const uint32_t calCount)
                ```

                ```
                template <typename dstT, typename srcT, bool isReuseSource = false>
                __aicore__ inline void AscendQuant(const LocalTensor<dstT>& dstTensor, const LocalTensor<srcT>& srcTensor, const LocalTensor<uint8_t>& sharedTmpBuffer, const LocalTensor<srcT>& scaleTensor, const LocalTensor<srcT>& offsetTensor, const uint32_t scaleCount, const uint32_t offsetCount, const uint32_t calCount)
                ```

            -   源操作数Tensor全部参与计算

                ```
                template <typename dstT, typename srcT, bool isReuseSource = false>
                __aicore__ inline void AscendQuant(const LocalTensor<dstT>& dstTensor, const LocalTensor<srcT>& srcTensor, const LocalTensor<uint8_t>& sharedTmpBuffer, const LocalTensor<srcT>& scaleTensor, const srcT offset)
                ```

                ```
                template <typename dstT, typename srcT, bool isReuseSource = false>
                __aicore__ inline void AscendQuant(const LocalTensor<dstT>& dstTensor, const LocalTensor<srcT>& srcTensor, const LocalTensor<uint8_t>& sharedTmpBuffer, const LocalTensor<srcT>& scaleTensor, const LocalTensor<srcT>& offsetTensor)
                ```

        -   接口框架申请临时空间
            -   源操作数Tensor全部/部分参与计算

                ```
                template <typename dstT, typename srcT, bool isReuseSource = false>
                __aicore__ inline void AscendQuant(const LocalTensor<dstT>& dstTensor, const LocalTensor<srcT>& srcTensor, const LocalTensor<srcT>& scaleTensor, const srcT offset, const uint32_t scaleCount, const uint32_t calCount)
                ```

                ```
                template <typename dstT, typename srcT, bool isReuseSource = false>
                __aicore__ inline void AscendQuant(const LocalTensor<dstT>& dstTensor, const LocalTensor<srcT>& srcTensor, const LocalTensor<srcT>& scaleTensor, const LocalTensor<srcT>& offsetTensor, const uint32_t scaleCount, const uint32_t offsetCount, const uint32_t calCount)
                ```

            -   源操作数Tensor全部参与计算

                ```
                template <typename dstT, typename srcT, bool isReuseSource = false>
                __aicore__ inline void AscendQuant(const LocalTensor<dstT>& dstTensor, const LocalTensor<srcT>& srcTensor, const LocalTensor<srcT>& scaleTensor, const srcT offset)
                ```

                ```
                template <typename dstT, typename srcT, bool isReuseSource = false>
                __aicore__ inline void AscendQuant(const LocalTensor<dstT>& dstTensor, const LocalTensor<srcT>& srcTensor, const LocalTensor<srcT>& scaleTensor, const LocalTensor<srcT>& offsetTensor)
                ```

    -   PER\_TOKEN/PER\_GROUP量化：
        -   通过sharedTmpBuffer入参传入临时空间
            -   offset操作数类型为Tensor

                ```
                template <typename dstT, typename srcT, typename scaleT, bool isReuseSource = false, const AscendQuantConfig& config, const AscendQuantPolicy& policy>
                __aicore__ inline void AscendQuant(const LocalTensor<dstT>& dstTensor, const LocalTensor<srcT>& srcTensor, const LocalTensor<uint8_t>& sharedTmpBuffer, const LocalTensor<scaleT>& scaleTensor, const LocalTensor<scaleT>& offsetTensor, const AscendQuantParam& para)
                ```

            -   offset操作数类型为scalar

                ```
                template <typename dstT, typename srcT, typename scaleT, bool isReuseSource = false, const AscendQuantConfig& config, const AscendQuantPolicy& policy>
                __aicore__ inline void AscendQuant(const LocalTensor<dstT>& dstTensor, const LocalTensor<srcT>& srcTensor, const LocalTensor<uint8_t>& sharedTmpBuffer, const LocalTensor<scaleT>& scaleTensor,const scaleT offset, const AscendQuantParam& para)
                ```

        -   接口框架申请临时空间
            -   offset操作数类型为Tensor

                ```
                template <typename dstT, typename srcT, typename scaleT, bool isReuseSource = false, const AscendQuantConfig& config, const AscendQuantPolicy& policy>
                __aicore__ inline void AscendQuant(const LocalTensor<dstT>& dstTensor, const LocalTensor<srcT>& srcTensor, const LocalTensor<scaleT>& scaleTensor, const LocalTensor<scaleT>& offsetTensor, const AscendQuantParam& para)
                ```

            -   offset操作数类型为scalar

                ```
                template <typename dstT, typename srcT, typename scaleT, bool isReuseSource = false, const AscendQuantConfig& config, const AscendQuantPolicy& policy>
                __aicore__ inline void AscendQuant(const LocalTensor<dstT>& dstTensor, const LocalTensor<srcT>& srcTensor, const LocalTensor<scaleT>& scaleTensor, const scaleT offset, const AscendQuantParam& para)
                ```

由于该接口的内部实现中涉及复杂的数学计算，需要额外的临时空间来存储计算过程中的中间变量。临时空间支持**接口框架申请**和开发者**通过sharedTmpBuffer入参传入**两种方式。

-   接口框架申请临时空间，开发者无需申请，但是需要预留临时空间的大小。

-   通过sharedTmpBuffer入参传入，使用该tensor作为临时空间进行处理，接口框架不再申请。该方式开发者可以自行管理sharedTmpBuffer内存空间，并在接口调用完成后，复用该部分内存，内存不会反复申请释放，灵活性较高，内存利用率也较高。

接口框架申请的方式，开发者需要预留临时空间；通过sharedTmpBuffer传入的情况，开发者需要为sharedTmpBuffer申请空间。临时空间大小BufferSize的获取方式如下：通过[GetAscendQuantMaxMinTmpSize](GetAscendQuantMaxMinTmpSize.md)中提供的GetAscendQuantMaxMinTmpSize接口获取需要预留空间的范围大小。

需要注意的是，在PER\_TOKEN/PER\_GROUP量化场景，内部实现不需要临时空间Buffer，对应的接口中sharedTempBuffer为预留参数。

## 参数说明<a name="section622mcpsimp"></a>

**表 1**  模板参数说明

<a name="table575571914269"></a>
<table><thead align="left"><tr id="zh-cn_topic_0000001538537601_row18755131942614"><th class="cellrowborder" valign="top" width="19.39%" id="mcps1.2.3.1.1"><p id="zh-cn_topic_0000001538537601_p675519193268"><a name="zh-cn_topic_0000001538537601_p675519193268"></a><a name="zh-cn_topic_0000001538537601_p675519193268"></a>参数名</p>
</th>
<th class="cellrowborder" valign="top" width="80.61%" id="mcps1.2.3.1.2"><p id="zh-cn_topic_0000001538537601_p375511918267"><a name="zh-cn_topic_0000001538537601_p375511918267"></a><a name="zh-cn_topic_0000001538537601_p375511918267"></a>描述</p>
</th>
</tr>
</thead>
<tbody><tr id="zh-cn_topic_0000001538537601_row14755141911264"><td class="cellrowborder" valign="top" width="19.39%" headers="mcps1.2.3.1.1 "><p id="zh-cn_topic_0000001538537601_p47551198266"><a name="zh-cn_topic_0000001538537601_p47551198266"></a><a name="zh-cn_topic_0000001538537601_p47551198266"></a>T</p>
</td>
<td class="cellrowborder" valign="top" width="80.61%" headers="mcps1.2.3.1.2 "><p id="zh-cn_topic_0000001538537601_p125969172719"><a name="zh-cn_topic_0000001538537601_p125969172719"></a><a name="zh-cn_topic_0000001538537601_p125969172719"></a>操作数的数据类型。</p>
</td>
</tr>
<tr id="zh-cn_topic_0000001538537601_row9756719122620"><td class="cellrowborder" valign="top" width="19.39%" headers="mcps1.2.3.1.1 "><p id="zh-cn_topic_0000001538537601_p1682112447268"><a name="zh-cn_topic_0000001538537601_p1682112447268"></a><a name="zh-cn_topic_0000001538537601_p1682112447268"></a>isReuseSource</p>
</td>
<td class="cellrowborder" valign="top" width="80.61%" headers="mcps1.2.3.1.2 "><p id="zh-cn_topic_0000001538537601_p98212044172612"><a name="zh-cn_topic_0000001538537601_p98212044172612"></a><a name="zh-cn_topic_0000001538537601_p98212044172612"></a>是否允许修改源操作数。该参数预留，传入默认值false即可。</p>
</td>
</tr>
<tr id="row1529110458389"><td class="cellrowborder" valign="top" width="19.39%" headers="mcps1.2.3.1.1 "><p id="p2292745133816"><a name="p2292745133816"></a><a name="p2292745133816"></a>config</p>
</td>
<td class="cellrowborder" valign="top" width="80.61%" headers="mcps1.2.3.1.2 "><p id="p1744211392255"><a name="p1744211392255"></a><a name="p1744211392255"></a>结构体模板参数，此参数可选配，AscendQuantConfig类型，具体定义如下。</p>
<a name="screen16476116195910"></a><a name="screen16476116195910"></a><pre class="code_wrap" codetype="Cpp" id="screen16476116195910">struct AscendQuantConfig{
uint32_t calcCount = 0;
uint32_t offsetCount = 0;
uint32_t scaleCount = 0;
uint32_t workLocalSize = 0;
};</pre>
<a name="ul1882194222413"></a><a name="ul1882194222413"></a><ul id="ul1882194222413"><li>calcCount：实际计算数据元素个数。calcCount∈[0, srcTensor.GetSize()]，在调用带有scaleCount入参的接口时，calcCount若取非零值则必须是scaleCount的整数倍。</li><li>offsetCount：实际量化参数元素个数。offsetCount∈[0, offsetTensor.GetSize()]，offsetCount与scaleCount的取值必须相等，要求是32的整数倍。若调用的接口不含offsetCount入参，取值为0即可。</li><li>scaleCount：实际量化参数元素个数。scaleCount∈[0, scaleTensor.GetSize()]，要求是32的整数倍。若调用的接口不含scaleCount入参，取值为0即可。</li><li>workLocalSize：临时缓存sharedTmpBuffer的大小，sharedTmpBuffer的大小/workLocalSize的获取方式请参考<a href="GetAscendQuantMaxMinTmpSize.md">GetAscendQuantMaxMinTmpSize</a>。该参数取值不能大于sharedTmpBuffer的大小。若调用的接口不含sharedTmpBuffer入参，取值为0即可。</li></ul>
<p id="p1189143944315"><a name="p1189143944315"></a><a name="p1189143944315"></a>当上述参数的取值满足如下任一种场景，将使能参数常量化，即编译过程中使用常量化的相关参数，从而减少Scalar计算。</p>
<a name="ul12991368432"></a><a name="ul12991368432"></a><ul id="ul12991368432"><li>若调用的接口不含scaleCount入参，calcCount和workLocalSize取值为非0时，使能参数常量化。</li><li>若调用的接口带有scaleCount入参，scaleCount、calcCount和workLocalSize取值为非0时，使能参数常量化。</li></ul>
<p id="p76421594583"><a name="p76421594583"></a><a name="p76421594583"></a>默认参数的配置示例如下。</p>
<a name="screen19241326175913"></a><a name="screen19241326175913"></a><pre class="code_wrap" codetype="Cpp" id="screen19241326175913">constexpr AscendQuantConfig ASCEND_QUANT_DEFAULT_CFG = {0, 0, 0, 0};</pre>
</td>
</tr>
</tbody>
</table>

**表 2**  dstTensor非固定数据类型的模板参数说明

<a name="table8442308815"></a>
<table><thead align="left"><tr id="row18755131942614"><th class="cellrowborder" valign="top" width="19.37%" id="mcps1.2.3.1.1"><p id="p675519193268"><a name="p675519193268"></a><a name="p675519193268"></a>参数名</p>
</th>
<th class="cellrowborder" valign="top" width="80.63%" id="mcps1.2.3.1.2"><p id="p375511918267"><a name="p375511918267"></a><a name="p375511918267"></a>描述</p>
</th>
</tr>
</thead>
<tbody><tr id="row14755141911264"><td class="cellrowborder" valign="top" width="19.37%" headers="mcps1.2.3.1.1 "><p id="p47551198266"><a name="p47551198266"></a><a name="p47551198266"></a>dstT</p>
</td>
<td class="cellrowborder" valign="top" width="80.63%" headers="mcps1.2.3.1.2 "><p id="p125969172719"><a name="p125969172719"></a><a name="p125969172719"></a>目的操作数的数据类型。</p>
<p id="p66875512307"><a name="p66875512307"></a><a name="p66875512307"></a><span id="ph146879513305"><a name="ph146879513305"></a><a name="ph146879513305"></a>Ascend 950PR/Ascend 950DT</span>，支持的数据类型为：int8_t、fp8_e4m3fn_t、fp8_e5m2_t、hifloat8_t、fp4x2_e1m2_t、fp4x2_e2m1_t。注意，对于fp4x2_e1m2_t、fp4x2_e2m1_t数据类型，仅在PER__GROUP场景下支持。</p>
</td>
</tr>
<tr id="row2885999813"><td class="cellrowborder" valign="top" width="19.37%" headers="mcps1.2.3.1.1 "><p id="p10414151887"><a name="p10414151887"></a><a name="p10414151887"></a>srcT</p>
</td>
<td class="cellrowborder" valign="top" width="80.63%" headers="mcps1.2.3.1.2 "><p id="p7411517814"><a name="p7411517814"></a><a name="p7411517814"></a>源操作数的数据类型。</p>
<p id="p12247148298"><a name="p12247148298"></a><a name="p12247148298"></a><span id="ph152477481396"><a name="ph152477481396"></a><a name="ph152477481396"></a>Ascend 950PR/Ascend 950DT</span>，支持的数据类型为：half、bfloat16_t、float。</p>
</td>
</tr>
<tr id="row9756719122620"><td class="cellrowborder" valign="top" width="19.37%" headers="mcps1.2.3.1.1 "><p id="p1682112447268"><a name="p1682112447268"></a><a name="p1682112447268"></a>isReuseSource</p>
</td>
<td class="cellrowborder" valign="top" width="80.63%" headers="mcps1.2.3.1.2 "><p id="p98212044172612"><a name="p98212044172612"></a><a name="p98212044172612"></a>是否允许修改源操作数。该参数预留，传入默认值false即可。</p>
</td>
</tr>
</tbody>
</table>

**表 3**  PER\_TOKEN/PER\_\_GROUP场景特有模板参数说明

<a name="table589125485010"></a>
<table><thead align="left"><tr id="row489125425015"><th class="cellrowborder" valign="top" width="19.259999999999998%" id="mcps1.2.3.1.1"><p id="p6891105420508"><a name="p6891105420508"></a><a name="p6891105420508"></a>参数名</p>
</th>
<th class="cellrowborder" valign="top" width="80.74%" id="mcps1.2.3.1.2"><p id="p17891145445014"><a name="p17891145445014"></a><a name="p17891145445014"></a>描述</p>
</th>
</tr>
</thead>
<tbody><tr id="row789235465020"><td class="cellrowborder" valign="top" width="19.259999999999998%" headers="mcps1.2.3.1.1 "><p id="p5892135418509"><a name="p5892135418509"></a><a name="p5892135418509"></a>scaleT</p>
</td>
<td class="cellrowborder" valign="top" width="80.74%" headers="mcps1.2.3.1.2 "><p id="p2892165465019"><a name="p2892165465019"></a><a name="p2892165465019"></a>量化参数scale和offset的数据类型。</p>
<p id="p162051891948"><a name="p162051891948"></a><a name="p162051891948"></a><span id="ph142052912417"><a name="ph142052912417"></a><a name="ph142052912417"></a>Ascend 950PR/Ascend 950DT</span>，支持的数据类型为：half、bfloat16_t、float。</p>
</td>
</tr>
<tr id="row089212543508"><td class="cellrowborder" valign="top" width="19.259999999999998%" headers="mcps1.2.3.1.1 "><p id="p15892165425015"><a name="p15892165425015"></a><a name="p15892165425015"></a>config</p>
</td>
<td class="cellrowborder" valign="top" width="80.74%" headers="mcps1.2.3.1.2 "><p id="p13892754135017"><a name="p13892754135017"></a><a name="p13892754135017"></a>量化接口配置参数，AscendQuantConfig类型，具体定义如下：</p>
<a name="screen173444919381"></a><a name="screen173444919381"></a><pre class="screen" codetype="Cpp" id="screen173444919381">struct AscendQuantConfig {
        bool hasOffset;
        int32_t kDim = 1;
        RoundMode roundMode = RoundMode::CAST_RINT;
}</pre>
<a name="ul558513317417"></a><a name="ul558513317417"></a><ul id="ul558513317417"><li>hasOffset：量化参数offset是否参与计算。<a name="ul186211135517"></a><a name="ul186211135517"></a><ul id="ul186211135517"><li>True：表示offset参数参与计算。</li><li>False：表示offset参数不参与计算。</li></ul>
</li><li>kDim：group的计算方向，即k方向。仅在PER__GROUP场景有效，支持的取值如下：<a name="ul1414773219247"></a><a name="ul1414773219247"></a><ul id="ul1414773219247"><li>0：k轴是第0轴，即m方向为group的计算方向；</li><li>1：k轴是第1轴，即n方向为group的计算方向。</li></ul>
</li><li>roundMode：量化过程中，数据由高精度数据类型转换为低精度数据类型的舍入模式，支持的取值有：CAST_NONE、CAST_RINT、CAST_ROUND、CAST_FLOOR、CAST_CEIL、CAST_TRUNC、CAST_HYBRID，各个舍入模式的详细介绍请参考<a href="Cast.md#table235404962912">精度转换规则</a>。不同数据类型的量化支持不同的舍入模式，当量化过程中使用了不支持的舍入模式时，将回退到默认的舍入模式；例如，bfloat16_t数据类型量化为hifloat8_t数据类型时，如果配置的roundMode为不支持的CAST_RINT，实际执行量化时将回退到默认的roundMode（CAST_ROUND）。不同数据类型支持的舍入模式请见<a href="#table158181847102411">表4 PER_TOKEN/PER__GROUP场景支持的数据类型组合</a>。</li></ul>
</td>
</tr>
<tr id="row13892205418504"><td class="cellrowborder" valign="top" width="19.259999999999998%" headers="mcps1.2.3.1.1 "><p id="p10892554165011"><a name="p10892554165011"></a><a name="p10892554165011"></a>policy</p>
</td>
<td class="cellrowborder" valign="top" width="80.74%" headers="mcps1.2.3.1.2 "><p id="p10892185420508"><a name="p10892185420508"></a><a name="p10892185420508"></a>量化策略配置参数，枚举类型，可取值如下：</p>
<a name="screen191844253917"></a><a name="screen191844253917"></a><pre class="screen" codetype="Cpp" id="screen191844253917">enum class AscendQuantPolicy : int32_t {
        PER_TENSOR, // 预留参数，暂不支持
        PER_CHANNEL, // 预留参数，暂不支持
        PER_TOKEN, // 配置为PER_TOKEN场景
        PER_GROUP,  // 配置为PER__GROUP场景
        PER_CHANNEL_PER_GROUP, // 预留参数，暂不支持
        PER_TOKEN_PER_GROUP // 预留参数，暂不支持
}</pre>
</td>
</tr>
</tbody>
</table>

**表 4**  PER\_TOKEN/PER\_\_GROUP场景支持的数据类型组合

<a name="table158181847102411"></a>
<table><thead align="left"><tr id="row381964718248"><th class="cellrowborder" valign="top" width="28.37716228377162%" id="mcps1.2.5.1.1"><p id="p1681934711240"><a name="p1681934711240"></a><a name="p1681934711240"></a>srcDtype</p>
</th>
<th class="cellrowborder" valign="top" width="30.676932306769327%" id="mcps1.2.5.1.2"><p id="p4819184792415"><a name="p4819184792415"></a><a name="p4819184792415"></a>scaleDtype/offsetDtype</p>
</th>
<th class="cellrowborder" valign="top" width="24.417558244175584%" id="mcps1.2.5.1.3"><p id="p48194471241"><a name="p48194471241"></a><a name="p48194471241"></a>dstDtype</p>
</th>
<th class="cellrowborder" valign="top" width="16.52834716528347%" id="mcps1.2.5.1.4"><p id="p665164216619"><a name="p665164216619"></a><a name="p665164216619"></a>roundMode</p>
</th>
</tr>
</thead>
<tbody><tr id="row1881954718248"><td class="cellrowborder" valign="top" width="28.37716228377162%" headers="mcps1.2.5.1.1 "><p id="p1681914742410"><a name="p1681914742410"></a><a name="p1681914742410"></a>half</p>
</td>
<td class="cellrowborder" valign="top" width="30.676932306769327%" headers="mcps1.2.5.1.2 "><p id="p1881984712245"><a name="p1881984712245"></a><a name="p1881984712245"></a>half</p>
</td>
<td class="cellrowborder" rowspan="5" valign="top" width="24.417558244175584%" headers="mcps1.2.5.1.3 "><p id="p1781916472249"><a name="p1781916472249"></a><a name="p1781916472249"></a>fp8_e5m2_t/fp8_e4m3fn_t</p>
<p id="p16819647172419"><a name="p16819647172419"></a><a name="p16819647172419"></a></p>
<p id="p18191747172412"><a name="p18191747172412"></a><a name="p18191747172412"></a></p>
<p id="p15819174712242"><a name="p15819174712242"></a><a name="p15819174712242"></a></p>
<p id="p881904717244"><a name="p881904717244"></a><a name="p881904717244"></a></p>
</td>
<td class="cellrowborder" rowspan="5" valign="top" width="16.52834716528347%" headers="mcps1.2.5.1.4 "><a name="ul7712201815432"></a><a name="ul7712201815432"></a><ul id="ul7712201815432"><li>CAST_RINT（默认）</li></ul>
</td>
</tr>
<tr id="row2819164722415"><td class="cellrowborder" valign="top" headers="mcps1.2.5.1.1 "><p id="p2081974711241"><a name="p2081974711241"></a><a name="p2081974711241"></a>bfloat16_t</p>
</td>
<td class="cellrowborder" valign="top" headers="mcps1.2.5.1.2 "><p id="p2819104711244"><a name="p2819104711244"></a><a name="p2819104711244"></a>bfloat16_t</p>
</td>
</tr>
<tr id="row2819184711242"><td class="cellrowborder" valign="top" headers="mcps1.2.5.1.1 "><p id="p581913477247"><a name="p581913477247"></a><a name="p581913477247"></a>float</p>
</td>
<td class="cellrowborder" valign="top" headers="mcps1.2.5.1.2 "><p id="p178191447122415"><a name="p178191447122415"></a><a name="p178191447122415"></a>float</p>
</td>
</tr>
<tr id="row1481954718242"><td class="cellrowborder" valign="top" headers="mcps1.2.5.1.1 "><p id="p148191847162413"><a name="p148191847162413"></a><a name="p148191847162413"></a>half</p>
</td>
<td class="cellrowborder" valign="top" headers="mcps1.2.5.1.2 "><p id="p98190478243"><a name="p98190478243"></a><a name="p98190478243"></a>float</p>
</td>
</tr>
<tr id="row9819747132418"><td class="cellrowborder" valign="top" headers="mcps1.2.5.1.1 "><p id="p17819134752411"><a name="p17819134752411"></a><a name="p17819134752411"></a>bfloat16_t</p>
</td>
<td class="cellrowborder" valign="top" headers="mcps1.2.5.1.2 "><p id="p148191847132412"><a name="p148191847132412"></a><a name="p148191847132412"></a>float</p>
</td>
</tr>
<tr id="row1181904702416"><td class="cellrowborder" valign="top" width="28.37716228377162%" headers="mcps1.2.5.1.1 "><p id="p10974729172719"><a name="p10974729172719"></a><a name="p10974729172719"></a>half</p>
</td>
<td class="cellrowborder" valign="top" width="30.676932306769327%" headers="mcps1.2.5.1.2 "><p id="p2974122902715"><a name="p2974122902715"></a><a name="p2974122902715"></a>half</p>
</td>
<td class="cellrowborder" rowspan="5" valign="top" width="24.417558244175584%" headers="mcps1.2.5.1.3 "><p id="p38601934122712"><a name="p38601934122712"></a><a name="p38601934122712"></a>hifloat8_t</p>
</td>
<td class="cellrowborder" rowspan="5" valign="top" width="16.52834716528347%" headers="mcps1.2.5.1.4 "><a name="ul1254012103435"></a><a name="ul1254012103435"></a><ul id="ul1254012103435"><li>CAST_ROUND（默认）CAST_HYBRID</li></ul>
</td>
</tr>
<tr id="row12819174742414"><td class="cellrowborder" valign="top" headers="mcps1.2.5.1.1 "><p id="p139741929112715"><a name="p139741929112715"></a><a name="p139741929112715"></a>bfloat16_t</p>
</td>
<td class="cellrowborder" valign="top" headers="mcps1.2.5.1.2 "><p id="p12974329192720"><a name="p12974329192720"></a><a name="p12974329192720"></a>bfloat16_t</p>
</td>
</tr>
<tr id="row12820134710244"><td class="cellrowborder" valign="top" headers="mcps1.2.5.1.1 "><p id="p20974202913273"><a name="p20974202913273"></a><a name="p20974202913273"></a>float</p>
</td>
<td class="cellrowborder" valign="top" headers="mcps1.2.5.1.2 "><p id="p19974132972710"><a name="p19974132972710"></a><a name="p19974132972710"></a>float</p>
</td>
</tr>
<tr id="row1382034714246"><td class="cellrowborder" valign="top" headers="mcps1.2.5.1.1 "><p id="p6974132918277"><a name="p6974132918277"></a><a name="p6974132918277"></a>half</p>
</td>
<td class="cellrowborder" valign="top" headers="mcps1.2.5.1.2 "><p id="p119742029112718"><a name="p119742029112718"></a><a name="p119742029112718"></a>float</p>
</td>
</tr>
<tr id="row16820147142414"><td class="cellrowborder" valign="top" headers="mcps1.2.5.1.1 "><p id="p797413291277"><a name="p797413291277"></a><a name="p797413291277"></a>bfloat16_t</p>
</td>
<td class="cellrowborder" valign="top" headers="mcps1.2.5.1.2 "><p id="p2097432917271"><a name="p2097432917271"></a><a name="p2097432917271"></a>float</p>
</td>
</tr>
<tr id="row6820144717249"><td class="cellrowborder" valign="top" width="28.37716228377162%" headers="mcps1.2.5.1.1 "><p id="p18669744142716"><a name="p18669744142716"></a><a name="p18669744142716"></a>half</p>
</td>
<td class="cellrowborder" valign="top" width="30.676932306769327%" headers="mcps1.2.5.1.2 "><p id="p18669164419276"><a name="p18669164419276"></a><a name="p18669164419276"></a>half</p>
</td>
<td class="cellrowborder" rowspan="5" valign="top" width="24.417558244175584%" headers="mcps1.2.5.1.3 "><p id="p16402047122710"><a name="p16402047122710"></a><a name="p16402047122710"></a>int8_t</p>
</td>
<td class="cellrowborder" rowspan="9" valign="top" width="16.52834716528347%" headers="mcps1.2.5.1.4 "><a name="ul63451314144317"></a><a name="ul63451314144317"></a><ul id="ul63451314144317"><li>CAST_RINT（默认）</li><li>CAST_ROUND</li><li>CAST_FLOOR</li><li>CAST_CEIL</li><li>CAST_TRUNC</li></ul>
</td>
</tr>
<tr id="row38201547172420"><td class="cellrowborder" valign="top" headers="mcps1.2.5.1.1 "><p id="p16669134432717"><a name="p16669134432717"></a><a name="p16669134432717"></a>bfloat16_t</p>
</td>
<td class="cellrowborder" valign="top" headers="mcps1.2.5.1.2 "><p id="p0669154418274"><a name="p0669154418274"></a><a name="p0669154418274"></a>bfloat16_t</p>
</td>
</tr>
<tr id="row138201347102416"><td class="cellrowborder" valign="top" headers="mcps1.2.5.1.1 "><p id="p12669134462710"><a name="p12669134462710"></a><a name="p12669134462710"></a>float</p>
</td>
<td class="cellrowborder" valign="top" headers="mcps1.2.5.1.2 "><p id="p46699449270"><a name="p46699449270"></a><a name="p46699449270"></a>float</p>
</td>
</tr>
<tr id="row1882010471247"><td class="cellrowborder" valign="top" headers="mcps1.2.5.1.1 "><p id="p136691844172712"><a name="p136691844172712"></a><a name="p136691844172712"></a>half</p>
</td>
<td class="cellrowborder" valign="top" headers="mcps1.2.5.1.2 "><p id="p1669184415276"><a name="p1669184415276"></a><a name="p1669184415276"></a>float</p>
</td>
</tr>
<tr id="row188202475248"><td class="cellrowborder" valign="top" headers="mcps1.2.5.1.1 "><p id="p1666914411278"><a name="p1666914411278"></a><a name="p1666914411278"></a>bfloat16_t</p>
</td>
<td class="cellrowborder" valign="top" headers="mcps1.2.5.1.2 "><p id="p176691449279"><a name="p176691449279"></a><a name="p176691449279"></a>float</p>
</td>
</tr>
<tr id="row981617480442"><td class="cellrowborder" valign="top" headers="mcps1.2.5.1.1 "><p id="p1681694819446"><a name="p1681694819446"></a><a name="p1681694819446"></a>half</p>
</td>
<td class="cellrowborder" valign="top" headers="mcps1.2.5.1.2 "><p id="p581684812442"><a name="p581684812442"></a><a name="p581684812442"></a>half</p>
</td>
<td class="cellrowborder" rowspan="4" valign="top" headers="mcps1.2.5.1.3 "><p id="p498117513318"><a name="p498117513318"></a><a name="p498117513318"></a>fp4x2_e1m2_t/fp4x2_e2m1_t</p>
<p id="p1041761134519"><a name="p1041761134519"></a><a name="p1041761134519"></a>（当前均只支持PER__GROUP场景）</p>
</td>
</tr>
<tr id="row10327145374411"><td class="cellrowborder" valign="top" headers="mcps1.2.5.1.1 "><p id="p3328115318444"><a name="p3328115318444"></a><a name="p3328115318444"></a>half</p>
</td>
<td class="cellrowborder" valign="top" headers="mcps1.2.5.1.2 "><p id="p6328145312446"><a name="p6328145312446"></a><a name="p6328145312446"></a>float</p>
</td>
</tr>
<tr id="row5420259134416"><td class="cellrowborder" valign="top" headers="mcps1.2.5.1.1 "><p id="p94211459104415"><a name="p94211459104415"></a><a name="p94211459104415"></a>bfloat16_t</p>
</td>
<td class="cellrowborder" valign="top" headers="mcps1.2.5.1.2 "><p id="p104211759194420"><a name="p104211759194420"></a><a name="p104211759194420"></a>bfloat16_t</p>
</td>
</tr>
<tr id="row11568155611441"><td class="cellrowborder" valign="top" headers="mcps1.2.5.1.1 "><p id="p145681056194411"><a name="p145681056194411"></a><a name="p145681056194411"></a>bfloat16_t</p>
</td>
<td class="cellrowborder" valign="top" headers="mcps1.2.5.1.2 "><p id="p1056875619446"><a name="p1056875619446"></a><a name="p1056875619446"></a>float</p>
</td>
</tr>
</tbody>
</table>

**表 5**  PER\_TENSOR接口参数说明

<a name="table44731299481"></a>
<table><thead align="left"><tr id="row247482914489"><th class="cellrowborder" valign="top" width="16.45%" id="mcps1.2.4.1.1"><p id="p147413295483"><a name="p147413295483"></a><a name="p147413295483"></a>参数名</p>
</th>
<th class="cellrowborder" valign="top" width="11.31%" id="mcps1.2.4.1.2"><p id="p1147432994819"><a name="p1147432994819"></a><a name="p1147432994819"></a>输入/输出</p>
</th>
<th class="cellrowborder" valign="top" width="72.24000000000001%" id="mcps1.2.4.1.3"><p id="p74749297483"><a name="p74749297483"></a><a name="p74749297483"></a>描述</p>
</th>
</tr>
</thead>
<tbody><tr id="row12474329104814"><td class="cellrowborder" valign="top" width="16.45%" headers="mcps1.2.4.1.1 "><p id="p1047411294482"><a name="p1047411294482"></a><a name="p1047411294482"></a>dstTensor</p>
</td>
<td class="cellrowborder" valign="top" width="11.31%" headers="mcps1.2.4.1.2 "><p id="p047412984813"><a name="p047412984813"></a><a name="p047412984813"></a>输出</p>
</td>
<td class="cellrowborder" valign="top" width="72.24000000000001%" headers="mcps1.2.4.1.3 "><p id="p3989161814016"><a name="p3989161814016"></a><a name="p3989161814016"></a>目的操作数。</p>
<p id="p1747492917489"><a name="p1747492917489"></a><a name="p1747492917489"></a><span id="zh-cn_topic_0000002523303824_ph173308471594"><a name="zh-cn_topic_0000002523303824_ph173308471594"></a><a name="zh-cn_topic_0000002523303824_ph173308471594"></a><span id="zh-cn_topic_0000002523303824_ph9902231466"><a name="zh-cn_topic_0000002523303824_ph9902231466"></a><a name="zh-cn_topic_0000002523303824_ph9902231466"></a><span id="zh-cn_topic_0000002523303824_ph1782115034816"><a name="zh-cn_topic_0000002523303824_ph1782115034816"></a><a name="zh-cn_topic_0000002523303824_ph1782115034816"></a>类型为<a href="LocalTensor.md">LocalTensor</a>，支持的TPosition为VECIN/VECCALC/VECOUT。</span></span></span></p>
</td>
</tr>
<tr id="row18474729124817"><td class="cellrowborder" valign="top" width="16.45%" headers="mcps1.2.4.1.1 "><p id="p54741029164810"><a name="p54741029164810"></a><a name="p54741029164810"></a>srcTensor</p>
</td>
<td class="cellrowborder" valign="top" width="11.31%" headers="mcps1.2.4.1.2 "><p id="p144741829194814"><a name="p144741829194814"></a><a name="p144741829194814"></a>输入</p>
</td>
<td class="cellrowborder" valign="top" width="72.24000000000001%" headers="mcps1.2.4.1.3 "><p id="p6914123244017"><a name="p6914123244017"></a><a name="p6914123244017"></a>源操作数。</p>
<p id="p1493334184019"><a name="p1493334184019"></a><a name="p1493334184019"></a><span id="zh-cn_topic_0000002523303824_ph173308471594_1"><a name="zh-cn_topic_0000002523303824_ph173308471594_1"></a><a name="zh-cn_topic_0000002523303824_ph173308471594_1"></a><span id="zh-cn_topic_0000002523303824_ph9902231466_1"><a name="zh-cn_topic_0000002523303824_ph9902231466_1"></a><a name="zh-cn_topic_0000002523303824_ph9902231466_1"></a><span id="zh-cn_topic_0000002523303824_ph1782115034816_1"><a name="zh-cn_topic_0000002523303824_ph1782115034816_1"></a><a name="zh-cn_topic_0000002523303824_ph1782115034816_1"></a>类型为<a href="LocalTensor.md">LocalTensor</a>，支持的TPosition为VECIN/VECCALC/VECOUT。</span></span></span></p>
</td>
</tr>
<tr id="row1747412296483"><td class="cellrowborder" valign="top" width="16.45%" headers="mcps1.2.4.1.1 "><p id="p74741029204817"><a name="p74741029204817"></a><a name="p74741029204817"></a>sharedTmpBuffer</p>
</td>
<td class="cellrowborder" valign="top" width="11.31%" headers="mcps1.2.4.1.2 "><p id="p1747452954810"><a name="p1747452954810"></a><a name="p1747452954810"></a>输入</p>
</td>
<td class="cellrowborder" valign="top" width="72.24000000000001%" headers="mcps1.2.4.1.3 "><p id="p191160465422"><a name="p191160465422"></a><a name="p191160465422"></a>临时缓存。</p>
<p id="p979635010404"><a name="p979635010404"></a><a name="p979635010404"></a><span id="zh-cn_topic_0000002523303824_ph173308471594_2"><a name="zh-cn_topic_0000002523303824_ph173308471594_2"></a><a name="zh-cn_topic_0000002523303824_ph173308471594_2"></a><span id="zh-cn_topic_0000002523303824_ph9902231466_2"><a name="zh-cn_topic_0000002523303824_ph9902231466_2"></a><a name="zh-cn_topic_0000002523303824_ph9902231466_2"></a><span id="zh-cn_topic_0000002523303824_ph1782115034816_2"><a name="zh-cn_topic_0000002523303824_ph1782115034816_2"></a><a name="zh-cn_topic_0000002523303824_ph1782115034816_2"></a>类型为<a href="LocalTensor.md">LocalTensor</a>，支持的TPosition为VECIN/VECCALC/VECOUT。</span></span></span></p>
<p id="p5881016172817"><a name="p5881016172817"></a><a name="p5881016172817"></a>临时空间大小BufferSize的获取方式请参考<a href="GetAscendQuantMaxMinTmpSize.md">GetAscendQuantMaxMinTmpSize</a>。</p>
</td>
</tr>
<tr id="row524952410266"><td class="cellrowborder" valign="top" width="16.45%" headers="mcps1.2.4.1.1 "><p id="p19249424182610"><a name="p19249424182610"></a><a name="p19249424182610"></a>scale</p>
</td>
<td class="cellrowborder" valign="top" width="11.31%" headers="mcps1.2.4.1.2 "><p id="p172491124122610"><a name="p172491124122610"></a><a name="p172491124122610"></a>输入</p>
</td>
<td class="cellrowborder" valign="top" width="72.24000000000001%" headers="mcps1.2.4.1.3 "><p id="p26761711274"><a name="p26761711274"></a><a name="p26761711274"></a>量化参数。</p>
<p id="p3249924172610"><a name="p3249924172610"></a><a name="p3249924172610"></a>类型为Scalar，支持的数据类型为float。</p>
</td>
</tr>
<tr id="row8946172732612"><td class="cellrowborder" valign="top" width="16.45%" headers="mcps1.2.4.1.1 "><p id="p4871237282"><a name="p4871237282"></a><a name="p4871237282"></a>offset</p>
</td>
<td class="cellrowborder" valign="top" width="11.31%" headers="mcps1.2.4.1.2 "><p id="p118710312812"><a name="p118710312812"></a><a name="p118710312812"></a>输入</p>
</td>
<td class="cellrowborder" valign="top" width="72.24000000000001%" headers="mcps1.2.4.1.3 "><p id="p1587193112813"><a name="p1587193112813"></a><a name="p1587193112813"></a>量化参数。</p>
<p id="p13871143192817"><a name="p13871143192817"></a><a name="p13871143192817"></a>类型为Scalar，支持的数据类型为float。</p>
</td>
</tr>
<tr id="row16421712252"><td class="cellrowborder" valign="top" width="16.45%" headers="mcps1.2.4.1.1 "><p id="p1949611581317"><a name="p1949611581317"></a><a name="p1949611581317"></a>calCount</p>
</td>
<td class="cellrowborder" valign="top" width="11.31%" headers="mcps1.2.4.1.2 "><p id="p174961758436"><a name="p174961758436"></a><a name="p174961758436"></a>输入</p>
</td>
<td class="cellrowborder" valign="top" width="72.24000000000001%" headers="mcps1.2.4.1.3 "><p id="p11378261546"><a name="p11378261546"></a><a name="p11378261546"></a>参与计算的元素个数。</p>
</td>
</tr>
</tbody>
</table>

**表 6**  PER\_CHANNEL接口参数说明

<a name="table8690143212334"></a>
<table><thead align="left"><tr id="row969063243317"><th class="cellrowborder" valign="top" width="16.45%" id="mcps1.2.4.1.1"><p id="p11690123212330"><a name="p11690123212330"></a><a name="p11690123212330"></a>参数名</p>
</th>
<th class="cellrowborder" valign="top" width="11.31%" id="mcps1.2.4.1.2"><p id="p769083263318"><a name="p769083263318"></a><a name="p769083263318"></a>输入/输出</p>
</th>
<th class="cellrowborder" valign="top" width="72.24000000000001%" id="mcps1.2.4.1.3"><p id="p19690163216331"><a name="p19690163216331"></a><a name="p19690163216331"></a>描述</p>
</th>
</tr>
</thead>
<tbody><tr id="row1369014325334"><td class="cellrowborder" valign="top" width="16.45%" headers="mcps1.2.4.1.1 "><p id="p1669014322334"><a name="p1669014322334"></a><a name="p1669014322334"></a>dstTensor</p>
</td>
<td class="cellrowborder" valign="top" width="11.31%" headers="mcps1.2.4.1.2 "><p id="p1690183223314"><a name="p1690183223314"></a><a name="p1690183223314"></a>输出</p>
</td>
<td class="cellrowborder" valign="top" width="72.24000000000001%" headers="mcps1.2.4.1.3 "><p id="p0690173203319"><a name="p0690173203319"></a><a name="p0690173203319"></a>目的操作数。</p>
<p id="p3690232203314"><a name="p3690232203314"></a><a name="p3690232203314"></a><span id="zh-cn_topic_0000002523303824_ph173308471594_3"><a name="zh-cn_topic_0000002523303824_ph173308471594_3"></a><a name="zh-cn_topic_0000002523303824_ph173308471594_3"></a><span id="zh-cn_topic_0000002523303824_ph9902231466_3"><a name="zh-cn_topic_0000002523303824_ph9902231466_3"></a><a name="zh-cn_topic_0000002523303824_ph9902231466_3"></a><span id="zh-cn_topic_0000002523303824_ph1782115034816_3"><a name="zh-cn_topic_0000002523303824_ph1782115034816_3"></a><a name="zh-cn_topic_0000002523303824_ph1782115034816_3"></a>类型为<a href="LocalTensor.md">LocalTensor</a>，支持的TPosition为VECIN/VECCALC/VECOUT。</span></span></span></p>
</td>
</tr>
<tr id="row126918321336"><td class="cellrowborder" valign="top" width="16.45%" headers="mcps1.2.4.1.1 "><p id="p2691632133315"><a name="p2691632133315"></a><a name="p2691632133315"></a>srcTensor</p>
</td>
<td class="cellrowborder" valign="top" width="11.31%" headers="mcps1.2.4.1.2 "><p id="p8691193273312"><a name="p8691193273312"></a><a name="p8691193273312"></a>输入</p>
</td>
<td class="cellrowborder" valign="top" width="72.24000000000001%" headers="mcps1.2.4.1.3 "><p id="p176912324339"><a name="p176912324339"></a><a name="p176912324339"></a>源操作数。</p>
<p id="p186912032113317"><a name="p186912032113317"></a><a name="p186912032113317"></a><span id="zh-cn_topic_0000002523303824_ph173308471594_4"><a name="zh-cn_topic_0000002523303824_ph173308471594_4"></a><a name="zh-cn_topic_0000002523303824_ph173308471594_4"></a><span id="zh-cn_topic_0000002523303824_ph9902231466_4"><a name="zh-cn_topic_0000002523303824_ph9902231466_4"></a><a name="zh-cn_topic_0000002523303824_ph9902231466_4"></a><span id="zh-cn_topic_0000002523303824_ph1782115034816_4"><a name="zh-cn_topic_0000002523303824_ph1782115034816_4"></a><a name="zh-cn_topic_0000002523303824_ph1782115034816_4"></a>类型为<a href="LocalTensor.md">LocalTensor</a>，支持的TPosition为VECIN/VECCALC/VECOUT。</span></span></span></p>
</td>
</tr>
<tr id="row569133217332"><td class="cellrowborder" valign="top" width="16.45%" headers="mcps1.2.4.1.1 "><p id="p869113263312"><a name="p869113263312"></a><a name="p869113263312"></a>sharedTmpBuffer</p>
</td>
<td class="cellrowborder" valign="top" width="11.31%" headers="mcps1.2.4.1.2 "><p id="p12691183223312"><a name="p12691183223312"></a><a name="p12691183223312"></a>输入</p>
</td>
<td class="cellrowborder" valign="top" width="72.24000000000001%" headers="mcps1.2.4.1.3 "><p id="p166910324330"><a name="p166910324330"></a><a name="p166910324330"></a>临时缓存。</p>
<p id="p146911332173314"><a name="p146911332173314"></a><a name="p146911332173314"></a><span id="zh-cn_topic_0000002523303824_ph173308471594_5"><a name="zh-cn_topic_0000002523303824_ph173308471594_5"></a><a name="zh-cn_topic_0000002523303824_ph173308471594_5"></a><span id="zh-cn_topic_0000002523303824_ph9902231466_5"><a name="zh-cn_topic_0000002523303824_ph9902231466_5"></a><a name="zh-cn_topic_0000002523303824_ph9902231466_5"></a><span id="zh-cn_topic_0000002523303824_ph1782115034816_5"><a name="zh-cn_topic_0000002523303824_ph1782115034816_5"></a><a name="zh-cn_topic_0000002523303824_ph1782115034816_5"></a>类型为<a href="LocalTensor.md">LocalTensor</a>，支持的TPosition为VECIN/VECCALC/VECOUT。</span></span></span></p>
<p id="p1691163215337"><a name="p1691163215337"></a><a name="p1691163215337"></a>临时空间大小BufferSize的获取方式请参考<a href="GetAscendQuantMaxMinTmpSize.md">GetAscendQuantMaxMinTmpSize</a>。</p>
</td>
</tr>
<tr id="row3691143243310"><td class="cellrowborder" valign="top" width="16.45%" headers="mcps1.2.4.1.1 "><p id="p9691123216338"><a name="p9691123216338"></a><a name="p9691123216338"></a>scaleTensor</p>
</td>
<td class="cellrowborder" valign="top" width="11.31%" headers="mcps1.2.4.1.2 "><p id="p2691332113316"><a name="p2691332113316"></a><a name="p2691332113316"></a>输入</p>
</td>
<td class="cellrowborder" valign="top" width="72.24000000000001%" headers="mcps1.2.4.1.3 "><p id="p66910324338"><a name="p66910324338"></a><a name="p66910324338"></a>量化参数。</p>
<p id="p13772144914345"><a name="p13772144914345"></a><a name="p13772144914345"></a><span id="zh-cn_topic_0000002523303824_ph173308471594_6"><a name="zh-cn_topic_0000002523303824_ph173308471594_6"></a><a name="zh-cn_topic_0000002523303824_ph173308471594_6"></a><span id="zh-cn_topic_0000002523303824_ph9902231466_6"><a name="zh-cn_topic_0000002523303824_ph9902231466_6"></a><a name="zh-cn_topic_0000002523303824_ph9902231466_6"></a><span id="zh-cn_topic_0000002523303824_ph1782115034816_6"><a name="zh-cn_topic_0000002523303824_ph1782115034816_6"></a><a name="zh-cn_topic_0000002523303824_ph1782115034816_6"></a>类型为<a href="LocalTensor.md">LocalTensor</a>，支持的TPosition为VECIN/VECCALC/VECOUT。</span></span></span></p>
</td>
</tr>
<tr id="row11691232163318"><td class="cellrowborder" valign="top" width="16.45%" headers="mcps1.2.4.1.1 "><p id="p1669273263311"><a name="p1669273263311"></a><a name="p1669273263311"></a>offsetTensor</p>
</td>
<td class="cellrowborder" valign="top" width="11.31%" headers="mcps1.2.4.1.2 "><p id="p16692123203315"><a name="p16692123203315"></a><a name="p16692123203315"></a>输入</p>
</td>
<td class="cellrowborder" valign="top" width="72.24000000000001%" headers="mcps1.2.4.1.3 "><p id="p156921323335"><a name="p156921323335"></a><a name="p156921323335"></a>量化参数。</p>
<p id="p8389256113411"><a name="p8389256113411"></a><a name="p8389256113411"></a><span id="zh-cn_topic_0000002523303824_ph173308471594_7"><a name="zh-cn_topic_0000002523303824_ph173308471594_7"></a><a name="zh-cn_topic_0000002523303824_ph173308471594_7"></a><span id="zh-cn_topic_0000002523303824_ph9902231466_7"><a name="zh-cn_topic_0000002523303824_ph9902231466_7"></a><a name="zh-cn_topic_0000002523303824_ph9902231466_7"></a><span id="zh-cn_topic_0000002523303824_ph1782115034816_7"><a name="zh-cn_topic_0000002523303824_ph1782115034816_7"></a><a name="zh-cn_topic_0000002523303824_ph1782115034816_7"></a>类型为<a href="LocalTensor.md">LocalTensor</a>，支持的TPosition为VECIN/VECCALC/VECOUT。</span></span></span></p>
</td>
</tr>
<tr id="row192947255353"><td class="cellrowborder" valign="top" width="16.45%" headers="mcps1.2.4.1.1 "><p id="p97107538356"><a name="p97107538356"></a><a name="p97107538356"></a>scaleCount</p>
</td>
<td class="cellrowborder" valign="top" width="11.31%" headers="mcps1.2.4.1.2 "><p id="p20295162511352"><a name="p20295162511352"></a><a name="p20295162511352"></a>输入</p>
</td>
<td class="cellrowborder" valign="top" width="72.24000000000001%" headers="mcps1.2.4.1.3 "><p id="p573415227366"><a name="p573415227366"></a><a name="p573415227366"></a>实际量化参数元素个数，且scaleCount∈[0, min(scaleTensor.GetSize(),dstTensor.GetSize())]，要求是32的整数倍。</p>
</td>
</tr>
<tr id="row666619297352"><td class="cellrowborder" valign="top" width="16.45%" headers="mcps1.2.4.1.1 "><p id="p1070112544354"><a name="p1070112544354"></a><a name="p1070112544354"></a>offsetCount</p>
</td>
<td class="cellrowborder" valign="top" width="11.31%" headers="mcps1.2.4.1.2 "><p id="p1766742953517"><a name="p1766742953517"></a><a name="p1766742953517"></a>输入</p>
</td>
<td class="cellrowborder" valign="top" width="72.24000000000001%" headers="mcps1.2.4.1.3 "><p id="p4535132373614"><a name="p4535132373614"></a><a name="p4535132373614"></a>实际量化参数元素个数，且offsetCount∈[0, min(offsetTensor.GetSize(),dstTensor.GetSize())]，并且和scaleCount必须相等，要求是32的整数倍。</p>
</td>
</tr>
<tr id="row06925328336"><td class="cellrowborder" valign="top" width="16.45%" headers="mcps1.2.4.1.1 "><p id="p569293273314"><a name="p569293273314"></a><a name="p569293273314"></a>calCount</p>
</td>
<td class="cellrowborder" valign="top" width="11.31%" headers="mcps1.2.4.1.2 "><p id="p8692432163315"><a name="p8692432163315"></a><a name="p8692432163315"></a>输入</p>
</td>
<td class="cellrowborder" valign="top" width="72.24000000000001%" headers="mcps1.2.4.1.3 "><p id="p6692103213316"><a name="p6692103213316"></a><a name="p6692103213316"></a>参与计算的元素个数。calCount必须是scaleCount的整数倍。</p>
</td>
</tr>
</tbody>
</table>

**表 7**  PER\_TOKEN/PER\_GROUP接口参数说明

<a name="table1735215113117"></a>
<table><thead align="left"><tr id="row435295114110"><th class="cellrowborder" valign="top" width="16.661666166616662%" id="mcps1.2.4.1.1"><p id="p1735225111115"><a name="p1735225111115"></a><a name="p1735225111115"></a>参数名</p>
</th>
<th class="cellrowborder" valign="top" width="10.99109910991099%" id="mcps1.2.4.1.2"><p id="p18352115112119"><a name="p18352115112119"></a><a name="p18352115112119"></a>输入/输出</p>
</th>
<th class="cellrowborder" valign="top" width="72.34723472347235%" id="mcps1.2.4.1.3"><p id="p1135235110113"><a name="p1135235110113"></a><a name="p1135235110113"></a>描述</p>
</th>
</tr>
</thead>
<tbody><tr id="row193524511719"><td class="cellrowborder" valign="top" width="16.661666166616662%" headers="mcps1.2.4.1.1 "><p id="p15352951716"><a name="p15352951716"></a><a name="p15352951716"></a>dstTensor</p>
</td>
<td class="cellrowborder" valign="top" width="10.99109910991099%" headers="mcps1.2.4.1.2 "><p id="p1635214511319"><a name="p1635214511319"></a><a name="p1635214511319"></a>输出</p>
</td>
<td class="cellrowborder" valign="top" width="72.34723472347235%" headers="mcps1.2.4.1.3 "><p id="p18204795410"><a name="p18204795410"></a><a name="p18204795410"></a>目的操作数。</p>
<p id="p320416920413"><a name="p320416920413"></a><a name="p320416920413"></a><span id="zh-cn_topic_0000002523303824_ph173308471594_8"><a name="zh-cn_topic_0000002523303824_ph173308471594_8"></a><a name="zh-cn_topic_0000002523303824_ph173308471594_8"></a><span id="zh-cn_topic_0000002523303824_ph9902231466_8"><a name="zh-cn_topic_0000002523303824_ph9902231466_8"></a><a name="zh-cn_topic_0000002523303824_ph9902231466_8"></a><span id="zh-cn_topic_0000002523303824_ph1782115034816_8"><a name="zh-cn_topic_0000002523303824_ph1782115034816_8"></a><a name="zh-cn_topic_0000002523303824_ph1782115034816_8"></a>类型为<a href="LocalTensor.md">LocalTensor</a>，支持的TPosition为VECIN/VECCALC/VECOUT。</span></span></span></p>
</td>
</tr>
<tr id="row123522518116"><td class="cellrowborder" valign="top" width="16.661666166616662%" headers="mcps1.2.4.1.1 "><p id="p63525511816"><a name="p63525511816"></a><a name="p63525511816"></a>srcTensor</p>
</td>
<td class="cellrowborder" valign="top" width="10.99109910991099%" headers="mcps1.2.4.1.2 "><p id="p14352105115117"><a name="p14352105115117"></a><a name="p14352105115117"></a>输入</p>
</td>
<td class="cellrowborder" valign="top" width="72.34723472347235%" headers="mcps1.2.4.1.3 "><p id="p1320459948"><a name="p1320459948"></a><a name="p1320459948"></a>源操作数。</p>
<p id="p82041993416"><a name="p82041993416"></a><a name="p82041993416"></a><span id="zh-cn_topic_0000002523303824_ph173308471594_9"><a name="zh-cn_topic_0000002523303824_ph173308471594_9"></a><a name="zh-cn_topic_0000002523303824_ph173308471594_9"></a><span id="zh-cn_topic_0000002523303824_ph9902231466_9"><a name="zh-cn_topic_0000002523303824_ph9902231466_9"></a><a name="zh-cn_topic_0000002523303824_ph9902231466_9"></a><span id="zh-cn_topic_0000002523303824_ph1782115034816_9"><a name="zh-cn_topic_0000002523303824_ph1782115034816_9"></a><a name="zh-cn_topic_0000002523303824_ph1782115034816_9"></a>类型为<a href="LocalTensor.md">LocalTensor</a>，支持的TPosition为VECIN/VECCALC/VECOUT。</span></span></span></p>
</td>
</tr>
<tr id="row735217510112"><td class="cellrowborder" valign="top" width="16.661666166616662%" headers="mcps1.2.4.1.1 "><p id="p135215512117"><a name="p135215512117"></a><a name="p135215512117"></a>sharedTmpBuffer</p>
</td>
<td class="cellrowborder" valign="top" width="10.99109910991099%" headers="mcps1.2.4.1.2 "><p id="p1635245114114"><a name="p1635245114114"></a><a name="p1635245114114"></a>输入</p>
</td>
<td class="cellrowborder" valign="top" width="72.34723472347235%" headers="mcps1.2.4.1.3 "><p id="p6205129543"><a name="p6205129543"></a><a name="p6205129543"></a>临时缓存。</p>
<p id="p1120549049"><a name="p1120549049"></a><a name="p1120549049"></a><span id="zh-cn_topic_0000002523303824_ph173308471594_10"><a name="zh-cn_topic_0000002523303824_ph173308471594_10"></a><a name="zh-cn_topic_0000002523303824_ph173308471594_10"></a><span id="zh-cn_topic_0000002523303824_ph9902231466_10"><a name="zh-cn_topic_0000002523303824_ph9902231466_10"></a><a name="zh-cn_topic_0000002523303824_ph9902231466_10"></a><span id="zh-cn_topic_0000002523303824_ph1782115034816_10"><a name="zh-cn_topic_0000002523303824_ph1782115034816_10"></a><a name="zh-cn_topic_0000002523303824_ph1782115034816_10"></a>类型为<a href="LocalTensor.md">LocalTensor</a>，支持的TPosition为VECIN/VECCALC/VECOUT。</span></span></span></p>
<p id="p62051192418"><a name="p62051192418"></a><a name="p62051192418"></a>临时空间大小BufferSize的获取方式请参考<a href="GetAscendQuantMaxMinTmpSize.md">GetAscendQuantMaxMinTmpSize</a>。</p>
</td>
</tr>
<tr id="row1235211516111"><td class="cellrowborder" valign="top" width="16.661666166616662%" headers="mcps1.2.4.1.1 "><p id="p5352185119112"><a name="p5352185119112"></a><a name="p5352185119112"></a>scaleTensor</p>
</td>
<td class="cellrowborder" valign="top" width="10.99109910991099%" headers="mcps1.2.4.1.2 "><p id="p153521051117"><a name="p153521051117"></a><a name="p153521051117"></a>输入</p>
</td>
<td class="cellrowborder" valign="top" width="72.34723472347235%" headers="mcps1.2.4.1.3 "><p id="p6205796411"><a name="p6205796411"></a><a name="p6205796411"></a>量化参数scale。</p>
<p id="p1205119748"><a name="p1205119748"></a><a name="p1205119748"></a><span id="zh-cn_topic_0000002523303824_ph173308471594_11"><a name="zh-cn_topic_0000002523303824_ph173308471594_11"></a><a name="zh-cn_topic_0000002523303824_ph173308471594_11"></a><span id="zh-cn_topic_0000002523303824_ph9902231466_11"><a name="zh-cn_topic_0000002523303824_ph9902231466_11"></a><a name="zh-cn_topic_0000002523303824_ph9902231466_11"></a><span id="zh-cn_topic_0000002523303824_ph1782115034816_11"><a name="zh-cn_topic_0000002523303824_ph1782115034816_11"></a><a name="zh-cn_topic_0000002523303824_ph1782115034816_11"></a>类型为<a href="LocalTensor.md">LocalTensor</a>，支持的TPosition为VECIN/VECCALC/VECOUT。</span></span></span></p>
</td>
</tr>
<tr id="row135285114116"><td class="cellrowborder" valign="top" width="16.661666166616662%" headers="mcps1.2.4.1.1 "><p id="p1352175111110"><a name="p1352175111110"></a><a name="p1352175111110"></a>offsetTensor/offset</p>
</td>
<td class="cellrowborder" valign="top" width="10.99109910991099%" headers="mcps1.2.4.1.2 "><p id="p33537511216"><a name="p33537511216"></a><a name="p33537511216"></a>输入</p>
</td>
<td class="cellrowborder" valign="top" width="72.34723472347235%" headers="mcps1.2.4.1.3 "><p id="p132061398413"><a name="p132061398413"></a><a name="p132061398413"></a>量化参数offset。</p>
<a name="ul10486184182917"></a><a name="ul10486184182917"></a><ul id="ul10486184182917"><li>offsetTensor：<p id="p020689945"><a name="p020689945"></a><a name="p020689945"></a><span id="zh-cn_topic_0000002523303824_ph173308471594_12"><a name="zh-cn_topic_0000002523303824_ph173308471594_12"></a><a name="zh-cn_topic_0000002523303824_ph173308471594_12"></a><span id="zh-cn_topic_0000002523303824_ph9902231466_12"><a name="zh-cn_topic_0000002523303824_ph9902231466_12"></a><a name="zh-cn_topic_0000002523303824_ph9902231466_12"></a><span id="zh-cn_topic_0000002523303824_ph1782115034816_12"><a name="zh-cn_topic_0000002523303824_ph1782115034816_12"></a><a name="zh-cn_topic_0000002523303824_ph1782115034816_12"></a>类型为<a href="LocalTensor.md">LocalTensor</a>，支持的TPosition为VECIN/VECCALC/VECOUT。</span></span></span></p>
</li><li>offset：<p id="p7116680304"><a name="p7116680304"></a><a name="p7116680304"></a>类型为Scalar。</p>
</li></ul>
<p id="p122061191946"><a name="p122061191946"></a><a name="p122061191946"></a><span id="ph14206896416"><a name="ph14206896416"></a><a name="ph14206896416"></a>Ascend 950PR/Ascend 950DT</span>，数据类型和scaleTensor保持一致。对于float4场景，offsetTensor/offset不生效。</p>
</td>
</tr>
<tr id="row1235316511515"><td class="cellrowborder" valign="top" width="16.661666166616662%" headers="mcps1.2.4.1.1 "><p id="p13533516110"><a name="p13533516110"></a><a name="p13533516110"></a>para</p>
</td>
<td class="cellrowborder" valign="top" width="10.99109910991099%" headers="mcps1.2.4.1.2 "><p id="p17353115110118"><a name="p17353115110118"></a><a name="p17353115110118"></a>输入</p>
</td>
<td class="cellrowborder" valign="top" width="72.34723472347235%" headers="mcps1.2.4.1.3 "><p id="p535310512112"><a name="p535310512112"></a><a name="p535310512112"></a>量化接口的参数，AscendQuantParam类型，具体定义如下：</p>
<a name="screen1670010177415"></a><a name="screen1670010177415"></a><pre class="screen" codetype="Cpp" id="screen1670010177415">struct AscendQuantParam {
        uint32_t m;
        uint32_t n; 
        uint32_t calCount;  
        uint32_t groupSize = 0;
}</pre>
<a name="ul7297944161517"></a><a name="ul7297944161517"></a><ul id="ul7297944161517"><li>m：m方向元素个数。</li><li>n：n方向元素个数。n值对应的数据大小需满足32B对齐的要求，即shape最后一维为n的输入输出均需要满足该维度上32B对齐的要求。</li><li>calCount:参与计算的元素个数。calCount必须是n的整数倍。</li><li>groupSize：PER_GROUP场景有效，表示groupSize行/列数据共用一个scale/offset。groupSize的取值必须大于0且是32的整倍数。</li></ul>
</td>
</tr>
</tbody>
</table>

## 返回值说明<a name="section38228281712"></a>

无

## 约束说明<a name="section633mcpsimp"></a>

-   源操作数与目的操作数可以复用。
-   操作数地址对齐要求请参见[通用地址对齐约束](通用说明和约束.md#section796754519912)。
-   输入输出操作数参与计算的数据长度要求32B对齐。
-   当Scale为float类型时，其取值范围仍为half类型的取值范围。
-   PER\_TOKEN/PER\_GROUP场景，连续计算方向（即n方向）的数据量要求32B对齐。

## 调用示例<a name="section642mcpsimp"></a>

```
// 输入shape为1024
uint32_t dataSize = 1024; 
// 输入类型为float/half, scale=2.0, offset=0.9，预留临时空间
AscendC::AscendQuant<srcType>(dstLocal, srcLocal, 2.0f, 0.9f, dataSize);
// 使用模板参数使能参数常量化的示例
// static constexpr AscendC::AscendQuantConfig static_config = {1024, 0, 0, 0};
// 使用AscendQuantConfig类型的参数static_config，传入模板参数将参数常量化
// AscendC::AscendQuant<srcType, false, static_config>(dstLocal, srcLocal, 2.0f, 0.9f, dataSize);
```

结果示例如下：

```
输入数据（srcLocal）: 
[-3.22      2.09     -2.025    -2.895    -1.349    -3.336     1.376
  2.453     3.861     1.085    -2.273     0.3923    0.3645   -2.127
 -3.09     -0.002726 -2.783     0.2615   -0.904     1.507    -1.017
  3.568     2.219     0.8643    0.922     1.144    -1.853     2.002
 -1.705     1.675    -3.482     1.519     0.4172    0.4307   -1.228
 -2.62      0.3354   -3.586     2.604     1.688    -3.646    -3.389
 -3.918     3.955     0.7954   -2.562    -1.085     2.91     -0.398
  3.771    -2.914     1.726     3.367     3.482     3.49      1.382
  3.512     0.1938   -0.4087   -3.75      2.873    -2.54      1.826
  3.738     3.188     2.676     0.724    -1.108    -2.682    -0.4783
  2.082    -0.462    -2.955    -2.543     3.98     -1.85      3.018
 -2.688     3.596    -0.799     1.222     1.686    -0.7925    3.295
 -3.568    -0.03836  -2.002    -1.212     1.927    -1.11      1.046
  3.793    -0.6226   -3.494    -3.371    -2.354    -1.7      -0.948
  2.682    -3.344     2.566     2.533    -1.335     1.405     3.867
  3.674     1.359     3.145    -1.22      1.054    -2.492    -1.214
  3.879     2.014     2.664    -2.863    -3.88      2.857     1.695
  2.852     2.893     2.367    -0.1832   -3.254    -1.49      1.13
  0.672    -1.863    -3.547     3.281    -1.573    -1.349    -3.547
 -3.766    -2.99     -3.203    -2.703    -2.793    -1.501     0.4785
 -1.216    -1.205     0.9097   -3.438     0.781    -1.505    -1.982
  0.2037    0.4595    0.759     0.844    -3.396     0.4778   -0.899
 -2.342    -0.961    -2.531    -0.10913  -3.516    -3.66      1.337
 -3.44      0.7495    1.958     2.775     0.0968   -3.       -2.13
 -1.818     2.664     2.066    -1.923     2.97     -2.047    -3.598
  0.1661   -0.179     3.186    -1.247     2.777    -3.344    -3.148
  2.275     2.916    -1.081    -3.213     2.87     -3.12     -3.066
 -0.6      -3.78     -3.012    -3.86     -0.707    -0.2203   -3.338
 -2.273     2.062    -2.422    -0.443    -1.333    -2.2      -1.478
 -2.816     1.134     0.2115   -2.459     3.842    -2.768     2.822
  1.3125   -2.143     1.971    -3.543    -0.07794  -0.1265    0.763
 -3.26      3.514     3.629     0.1902    1.277    -0.1652   -0.006435
 -1.25      2.258    -2.887     3.66      2.729    -3.27     -0.5615
 -3.176    -1.2295    1.556    -0.6626   -2.777     1.946    -0.338
 -2.977    -0.8135   -2.37      0.7764    3.525    -0.6196    2.436
  2.38     -1.708     0.814     0.4688   -1.255     1.04     -1.077
  3.176     1.859     0.9194    2.703     1.436     1.762     2.2
  1.794    -1.234    -2.148    -2.393     2.846     1.854     0.3428
 -2.379     0.2429   -1.561     2.582     0.6836    1.811    -2.53
 -3.951    -2.096    -2.639     2.02      2.799    -0.8936   -1.295
 -3.914    -1.82      2.541    -2.773     1.733     3.955    -3.092
  0.04095   0.82     -1.071     3.93     -3.158    -2.5      -0.5415
 -1.98     -0.1626    3.092    -1.3125    3.387    -2.496     2.355
 -3.033    -3.814    -3.191     2.686     1.377     1.381    -3.047
  2.127    -0.4927   -1.718     2.371    -0.1648    1.885    -0.6826
 -3.121    -2.379    -3.959    -2.164     2.262    -2.973     3.092
  2.111    -0.03732   2.836    -2.725     3.436     1.017     2.877
 -2.926     2.547     0.8574    2.643     2.646    -0.889     3.363
 -0.3147   -0.09546   0.0551   -3.947    -1.434    -0.6104   -3.41
 -2.176    -1.866     3.975    -3.031    -1.25      3.918     3.697
  3.21     -2.436    -3.281    -3.225     0.7856    2.043     1.415
 -2.252    -1.648     0.03824  -3.432     0.3271    1.458    -0.02289
 -0.643     1.441    -0.1847    1.062     3.545     0.367     1.796
 -1.687     2.06      0.2373    3.748    -2.752     2.73     -2.693
 -3.54     -2.275    -3.033    -1.622    -3.936     1.295     2.586
 -2.926    -2.314     2.527    -1.619    -0.04037  -3.225     1.771
  3.064    -1.173    -2.324     3.332    -0.8257    1.075    -3.287
  1.075    -2.262     1.419    -0.344    -0.4988    1.113     3.068
 -1.104     2.531     2.645     0.6333    0.3677   -3.186    -0.3726
  2.549    -0.3347    2.227    -3.963    -2.564     3.656     1.069
 -3.684    -1.388    -0.2568   -0.726     0.4883    1.946    -1.579
 -0.8438   -2.014     2.332     0.306    -3.305    -3.588    -1.038
  3.299     0.832     0.8594   -1.163     1.2705    2.018    -3.352
  2.537     2.111    -3.61      0.645    -2.459    -2.469     1.002
 -3.914     1.079    -0.9214   -2.111    -3.88     -0.5254   -1.908
 -1.19      3.559    -3.285    -2.266     3.672     0.001524 -1.964
 -1.742     1.895     3.887     1.737     0.909     0.5044    2.55
  0.8936    2.139    -3.658     1.828    -3.688    -3.26      1.436
 -1.321    -3.19      2.764    -3.305    -2.52     -2.441    -0.32
 -2.402     2.252    -1.527     0.719     0.2328    0.1766   -2.088
  3.729     0.844    -1.174    -0.7427    0.8296   -0.1885   -0.0379
  2.92      2.502     3.846     1.657    -3.58     -3.352    -3.904
 -2.43      1.159    -1.707     2.21      2.367    -0.5864   -1.647
  1.952   ]
输出数据（dstLocal）: 
[-6  5 -3 -5 -2 -6  4  6  9  3 -4  2  2 -3 -5  1 -5  1 -1  4 -1  8  5  3
  3  3 -3  5 -3  4 -6  4  2  2 -2 -4  2 -6  6  4 -6 -6 -7  9  2 -4 -1  7
  0  8 -5  4  8  8  8  4  8  1  0 -7  7 -4  5  8  7  6  2 -1 -4  0  5  0
 -5 -4  9 -3  7 -4  8 -1  3  4 -1  7 -6  1 -3 -2  5 -1  3  8  0 -6 -6 -4
 -2 -1  6 -6  6  6 -2  4  9  8  4  7 -2  3 -4 -2  9  5  6 -5 -7  7  4  7
  7  6  1 -6 -2  3  2 -3 -6  7 -2 -2 -6 -7 -5 -6 -5 -5 -2  2 -2 -2  3 -6
  2 -2 -3  1  2  2  3 -6  2 -1 -4 -1 -4  1 -6 -6  4 -6  2  5  6  1 -5 -3
 -3  6  5 -3  7 -3 -6  1  1  7 -2  6 -6 -5  5  7 -1 -6  7 -5 -5  0 -7 -5
 -7 -1  0 -6 -4  5 -4  0 -2 -3 -2 -5  3  1 -4  9 -5  7  4 -3  5 -6  1  1
  2 -6  8  8  1  3  1  1 -2  5 -5  8  6 -6  0 -5 -2  4  0 -5  5  0 -5 -1
 -4  2  8  0  6  6 -3  3  2 -2  3 -1  7  5  3  6  4  4  5  4 -2 -3 -4  7
  5  2 -4  1 -2  6  2  5 -4 -7 -3 -4  5  6 -1 -2 -7 -3  6 -5  4  9 -5  1
  3 -1  9 -5 -4  0 -3  1  7 -2  8 -4  6 -5 -7 -5  6  4  4 -5  5  0 -3  6
  1  5  0 -5 -4 -7 -3  5 -5  7  5  1  7 -5  8  3  7 -5  6  3  6  6 -1  8
  0  1  1 -7 -2  0 -6 -3 -3  9 -5 -2  9  8  7 -4 -6 -6  2  5  4 -4 -2  1
 -6  2  4  1  0  4  1  3  8  2  4 -2  5  1  8 -5  6 -4 -6 -4 -5 -2 -7  3
  6 -5 -4  6 -2  1 -6  4  7 -1 -4  8 -1  3 -6  3 -4  4  0  0  3  7 -1  6
  6  2  2 -5  0  6  0  5 -7 -4  8  3 -6 -2  0 -1  2  5 -2 -1 -3  6  2 -6
 -6 -1  7  3  3 -1  3  5 -6  6  5 -6  2 -4 -4  3 -7  3 -1 -3 -7  0 -3 -1
  8 -6 -4  8  1 -3 -3  5  9  4  3  2  6  3  5 -6  5 -6 -6  4 -2 -5  6 -6
 -4 -4  0 -4  5 -2  2  1  1 -3  8  3 -1 -1  3  1  1  7  6  9  4 -6 -6 -7
 -4  3 -3  5  6  0 -2  5]
```

PER\_TOKEN/PER\_GROUP场景调用示例如下。

-   未配置参数AscendQuantConfig的舍入模式roundMode，使用默认配置RoundMode::CAST\_RINT。

    ```
    // 注意m,n需从外部传入
    constexpr static bool isReuseSource = false;
    constexpr static AscendQuantConfig config = {has_offset, 1};
    constexpr static AscendQuantPolicy policy = AscendQuantPolicy::PER_TOKEN; // 可修改枚举值以使能PER_GROUP
    LocalTensor<uint8_t> sharedTmpBuffer = inQueue.AllocTensor<uint8_t>();
    AscendQuantParam para;
    para.m = m;
    para.n = n;
    para.calCount = calCount;
    AscendQuant<dstType, srcType, scaleType, isReuseSource, config, policy>(dstLocal, srcLocal, sharedTmpBuffer, scaleLocal, offsetLocal, para);
    ```

-   主动配置参数AscendQuantConfig的舍入模式roundMode。

    ```
    // 注意m,n需从外部传入
    constexpr static bool isReuseSource = false;
    constexpr static AscendQuantConfig config = {has_offset, 1, RoundMode::CAST_ROUND};
    constexpr static AscendQuantPolicy policy = AscendQuantPolicy::PER_TOKEN; // 可修改枚举值以使能PER_GROUP
    LocalTensor<uint8_t> sharedTmpBuffer = inQueue.AllocTensor<uint8_t>();
    AscendQuantParam para;
    para.m = m;
    para.n = n;
    para.calCount = calCount;
    AscendQuant<dstType, srcType, scaleType, isReuseSource, config, policy>(dstLocal, srcLocal, sharedTmpBuffer, scaleLocal, offsetLocal, para);
    ```

