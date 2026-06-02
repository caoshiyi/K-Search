# AscendAntiQuant<a name="ZH-CN_TOPIC_0000002554423591"></a>

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

按元素做伪量化计算，比如将int8\_t数据类型伪量化为half数据类型，计算公式如下：

-   PER\_CHANNEL场景（按通道量化）
    -   不使能输入转置

        groupSize = src.shape\[0\] / offset.shape\[0\]

        **dst\[i\]\[j\] = scale\[i / groupSize\]\[j\] \* \(src\[i\]\[j\] + offset\[i / groupSize\]\[j\]\)**

    -   使能输入转置

        groupSize = src.shape\[1\] / offset.shape\[1\]

        **dst\[i\]\[j\] = scale\[i\]\[j / groupSize\] \* \(src\[i\]\[j\] + offset\[i\]\[j / groupSize\]\)**

-   PER\_TENSOR场景 （按张量量化）

    **dst\[i\]\[j\] = scale \* \(src\[i\]\[j\] + offset\)**l

-   PER\_TOKEN场景 （按token量化）

    ![](figures/zh-cn_formulaimage_0000002523305596.png)

-   PER\_GROUP场景 （按组量化）

    根据输入数据类型的不同，当前PER\_GROUP分为两种场景：fp4x2\_e2m1\_t/fp4x2\_e1m2\_t场景（后续内容中简称为float4场景）和int8\_t/hifloat8\_t/fp8\_e5m2\_t/fp8\_e4m3fn\_t场景（后续内容中简称为b8场景）。

    -   fp4x2\_e2m1\_t/fp4x2\_e1m2\_t场景（float4场景）
        -   groupSize可配置接口

            定义group的计算方向为k方向，src在k方向上每groupSize个元素共享一组scale。src的shape为\[m, n\]时，如果kDim=0，表示k是m方向，scale的shape为\[\(m + groupSize - 1\) / groupSize, n\]；如果kDim=1，表示k是n方向，scale的shape为\[m，\(n + groupSize - 1\) / groupSize\]。isTranspose为True表示src，scale，dst都是转置的矩阵。

            -   k为m方向，即公式中i轴为group的计算方向：（kDim=0同时isTranspose=False）或者（kDim=1同时isTranspose=True）

                **dst\[i\]\[j\] = scale\[i / groupSize\]\[j\] \* src\[i\]\[j\]**

            -   k为n方向，即公式中j轴为group的计算方向：（kDim=0同时isTranspose=True）或者（kDim=1同时isTranspose=False）

                **dst\[i\]\[j\] = scale\[i\]\[j / groupSize\] \* src\[i\]\[j\]**

        -   groupSize固定为32

            isTranspose为True表示src，scale，dst都是转置的矩阵。

            -   不使能输入转置（isTranspose=False）

                **dst\[i\]\[j\] = scale\[i / groupSize\]\[j\] \* src\[i\]\[j\]**

            -   使能输入转置（isTranspose=True）

                **dst\[i\]\[j\] = scale\[i\]\[j / groupSize\] \* src\[i\]\[j\]**

    -   int8\_t/hifloat8\_t/fp8\_e5m2\_t/fp8\_e4m3fn\_t场景（b8场景）

        定义group的计算方向为k方向，src在k方向上每groupSize个元素共享一组scale和offset。src的shape为\[m, n\]时，如果kDim=0，表示k是m方向，scale和offset的shape为\[\(m + groupSize - 1\) / groupSize, n\]；如果kDim=1，表示k是n方向，scale和offset的shape为\[m，\(n + groupSize - 1\) / groupSize\]。offset是可选输入。isTranspose为True表示src，scale，dst都是转置的矩阵。

        -   k为m方向，即公式中i轴为group的计算方向：（kDim=0同时isTranspose=False）或者（kDim=1同时isTranspose= True）

            <!-- img2text -->
$$
dst_{ij}=src_{ij}\times scale_{\left\lfloor \frac{i}{groupSize}\right\rfloor j}+offset_{\left\lfloor \frac{i}{groupSize}\right\rfloor j}
$$

        -   k为n方向，即公式中j轴为group的计算方向：（kDim=0同时isTranspose=True）或者（kDim=1同时isTranspose =False）

            <!-- img2text -->
$$
\text{groupScale}_{i,j}=\text{scale}_{\left\lfloor \frac{j}{\text{groupSize}} \right\rfloor}
$$

$$
\text{groupOffset}_{i,j}=\text{offset}_{\left\lfloor \frac{j}{\text{groupSize}} \right\rfloor}
$$

## 实现原理<a name="section13229175017585"></a>

**图 1**  AscendAntiQuant算法框图<a name="fig18399121215208"></a>  
<!-- img2text -->
```
                    ┌───────────┐                                   ┌───────────┐
                    │ src[k,n]  │                                   │ src[k,n]  │
                    └─────┬─────┘                                   └─────┬─────┘
                          │                                                 │
                          ↓                                                 ↓
         ┌─────────────────────────────────┐              ┌─────────────────────────────────┐
         │ ┌─────────────────────────────┐ │              │ ┌─────────────────────────────┐ │
         │ │      cast int8->half        │ │              │ │      cast int8->half        │ │
         │ │          x[k,n]             │ │              │ │          x[k,n]             │ │
         │ └──────────────┬──────────────┘ │              │ └──────────────┬──────────────┘ │
         │                │                │              │                │                │
         │                ↓                │              │                ↓                │
         │ ┌─────────────────────────────┐ │  ←─────────  │ ┌─────────────────────────────┐ │  ←─────────
         │ │             add             │ │   offset[1,n]│ │            adds             │ │   offset[1]
         │ │ y[k,n] = x[k,n] + offset[1,n]│ │              │ │   y[k,n] = x[k,n] + offset[1]│ │
         │ └──────────────┬──────────────┘ │              │ └──────────────┬──────────────┘ │
         │                │                │              │                │                │
         │                ↓                │              │                ↓                │
         │ ┌─────────────────────────────┐ │  ←─────────  │ ┌─────────────────────────────┐ │  ←─────────
         │ │             mul             │ │   scale[1,n] │ │            muls             │ │   scale[1]
         │ │ y[k,n] = y[k,n] * scale[1,n]│ │              │ │    y[k,n] = y[k,n] * scale[1]│ │
         │ └──────────────┬──────────────┘ │              │ └──────────────┬──────────────┘ │
         └────────────────┼────────────────┘              └────────────────┼────────────────┘
                          │                                                 │
                          ↓                                                 ↓
                    ┌───────────┐                                   ┌───────────┐
                    │ dst[k,n]  │                                   │ dst[k,n]  │
                    └───────────┘                                   └───────────┘


图示:
输入输出Tensor/Scalar    ┌───────────┐
                        │           │
                        └───────────┘

vector计算            ┌─────────────────────────────┐
                      │                             │
                      └─────────────────────────────┘

数据流向              ─────────→
```

如上图所示，为AscendAntiQuant的典型场景算法框图，计算过程分为如下几步，均在Vector上进行：

1.  精度转换：将输入src转换为half类型；
2.  计算offset：当offset为向量时做Add计算，当offset为scalar时做Adds计算；
3.  计算scale：当scale为向量时做Mul计算，当scale为scalar时做Muls计算。

**图 2**  AscendAntiQuant PER\_TOKEN/PER\_GROUP算法框图<a name="fig585016572279"></a>  
<!-- img2text -->
```
                              ┌──────────────┐
                              │ src_local[m,n] │
                              └──────┬───────┘
                                     │
                                     ▼
┌────────┐                    ┌──────────────┐
│ scale[m] │                  │   读取src     │
└────┬───┘                    └──────┬───────┘
     │                               │
     │                               ▼
     │                    ┌──────────────────────┐
     │                    │ src_vreg =           │
     │                    │ Cast(ori_src_vreg)   │
     │                    └──────┬───────────────┘
     │                           │
     │                           ▼
     │                    ┌──────────────────────┐     ┌──────────────────────────┐
     │                    │ temp_vreg=           │ ◄── │ offset_vreg =            │
     │                    │ Add(src_vreg,        │     │ Cast(ori_offset_vreg)    │
     │                    │ offset_vreg)         │     └──────────┬───────────────┘
     │                    └──────┬───────────────┘                │
     │                           │                                │
     │                           ▼                                │
     │                    ┌──────────────────────┐                │
     │                    │ temp2_vreg=          │                │
     └───────────────►    │ Mul(temp_vreg,       │                │
                          │ scale_vreg)          │                │
┌──────────────┐          └──────┬───────────────┘                │
│   读取scale   │                 │                                │
└──────┬───────┘                 ▼                                │
       │                 ┌──────────────────────┐                 │
       ▼                 │ dst_vreg =           │                 │
┌──────────────────────┐ │ Cast(temp2_vreg)     │                 │
│ scale_vreg =         │ └──────┬───────────────┘                 │
│ Cast(ori_scale_vreg) │        │                                 │
└──────────┬───────────┘        ▼                                 │
           │              ┌──────────────┐                        │
           │              │ dst_local[m,n] │                        │
           │              └──────────────┘                        │
           │                                                       │
           │                                            ┌──────────┴──────────┐
           │                                            │      读取offset      │
           │                                            └──────────┬──────────┘
           │                                                       │
           │                                                       │
           │                                            ┌──────────▼──────────┐
           │                                            │      offset[n]      │
           │                                            └─────────────────────┘
```

PER\_TOKEN/PER\_GROUP b8/float4场景的计算逻辑如下：

1.  读取数据：连续读取输入src；根据不同的场景，对输入scale和offset，采用不同的读取方式；例如，PER\_TOKEN场景做Broadcast处理，PER\_GROUP场景做Gather处理；
2.  精度转换：根据不同输入的数据类型组合，对src/scale/offset进行相应的数据类型转换；
3.  计算：对类型转换后的数据做加乘操作；
4.  精度转换：将上述加乘操作得到的计算结果转换成dstT类型，得到最终输出。

## 函数原型<a name="section620mcpsimp"></a>

-   通过sharedTmpBuffer入参传入临时空间
    -   PER\_CHANNEL场景（按通道量化）

        ```
        template <typename InputDataType, typename OutputDataType, bool isTranspose>
        __aicore__ inline void AscendAntiQuant(const LocalTensor<OutputDataType>& dst, const LocalTensor<InputDataType>& src, const LocalTensor<OutputDataType>& offset, const LocalTensor<OutputDataType>& scale, const LocalTensor<uint8_t>& sharedTmpBuffer, const uint32_t k, const AntiQuantShapeInfo& shapeInfo = {})
        ```

    -   PER\_CHANNEL场景（按通道量化，不带offset）

        ```
        template <typename InputDataType, typename OutputDataType, bool isTranspose>
        __aicore__ inline void AscendAntiQuant(const LocalTensor<OutputDataType>& dst, const LocalTensor<InputDataType>& src, const LocalTensor<OutputDataType>& scale, const LocalTensor<uint8_t>& sharedTmpBuffer, const uint32_t k, const AntiQuantShapeInfo& shapeInfo = {})
        ```

    -   PER\_TENSOR场景 （按张量量化）

        ```
        template <typename InputDataType, typename OutputDataType, bool isTranspose>
        __aicore__ inline void AscendAntiQuant(const LocalTensor<OutputDataType>& dst, const LocalTensor<InputDataType>& src, const OutputDataType offset, const OutputDataType scale, const LocalTensor<uint8_t>& sharedTmpBuffer, const uint32_t k, const AntiQuantShapeInfo& shapeInfo = {})
        ```

    -   PER\_TENSOR场景 （按张量量化，不带offset）

        ```
        template <typename InputDataType, typename OutputDataType, bool isTranspose>
        __aicore__ inline void AscendAntiQuant(const LocalTensor<OutputDataType> &dst, const LocalTensor<InputDataType> &src, const OutputDataType scale, const LocalTensor<uint8_t> &sharedTmpBuffer, const uint32_t k, const AntiQuantShapeInfo& shapeInfo = {})
        ```

    -   PER\_GROUP float4场景（按组量化）

        ```
        template <typename InputDataType, typename OutputDataType, bool isTranspose>
        __aicore__ inline void AscendAntiQuant(const LocalTensor<OutputDataType>& dst, const LocalTensor<InputDataType>& src, const LocalTensor<fp8_e8m0_t>& scale, const LocalTensor<uint8_t>& sharedTmpBuffer, const uint32_t k, const AntiQuantShapeInfo& shapeInfo = {})
        ```

    -   PER\_TOKEN/PER\_GROUP b8/float4场景（按token量化）/（按组量化）

        ```
        template <typename dstT, typename srcT, typename scaleT, const AscendAntiQuantConfig& config, const AscendAntiQuantPolicy& policy>
        __aicore__ inline void AscendAntiQuant(const LocalTensor<dstT>& dstTensor, const LocalTensor<srcT>& srcTensor, const LocalTensor<scaleT>& scaleTensor, const LocalTensor<scaleT>& offsetTensor,const LocalTensor<uint8_t>& sharedTmpBuffer, const AscendAntiQuantParam& para)
        ```

-   接口框架申请临时空间
    -   PER\_CHANNEL场景

        ```
        template <typename InputDataType, typename OutputDataType, bool isTranspose>
        __aicore__ inline void AscendAntiQuant(const LocalTensor<OutputDataType>& dst, const LocalTensor<InputDataType>& src, const LocalTensor<OutputDataType>& offset, const LocalTensor<OutputDataType>& scale, const uint32_t k, const AntiQuantShapeInfo& shapeInfo = {})
        ```

    -   PER\_TENSOR场景

        ```
        template <typename InputDataType, typename OutputDataType, bool isTranspose>
        __aicore__ inline void AscendAntiQuant(const LocalTensor<OutputDataType>& dst, const LocalTensor<InputDataType>& src, const OutputDataType offset, const OutputDataType scale, const uint32_t k, const AntiQuantShapeInfo& shapeInfo = {})
        ```

    -   PER\_GROUP float4场景（groupSize固定为32）

        ```
        template <typename InputDataType, typename OutputDataType, bool isTranspose>
        __aicore__ inline void AscendAntiQuant(const LocalTensor<OutputDataType>& dst, const LocalTensor<InputDataType>& src, const LocalTensor<fp8_e8m0_t>& scale, const uint32_t k, const AntiQuantShapeInfo& shapeInfo = {})
        ```

    -   PER\_TOKEN/PER\_GROUP b8/float4场景（groupSize可配置）

        ```
        template <typename dstT, typename srcT, typename scaleT, const AscendAntiQuantConfig& config, const AscendAntiQuantPolicy& policy>
        __aicore__ inline void AscendAntiQuant(const LocalTensor<dstT>& dstTensor, const LocalTensor<srcT>& srcTensor, const LocalTensor<scaleT>& scaleTensor, const LocalTensor<scaleT>& offsetTensor,const AscendAntiQuantParam& para)
        ```

由于该接口的内部实现中涉及复杂的数学计算，需要额外的临时空间来存储计算过程中的中间变量。临时空间支持**接口框架申请**和开发者**通过sharedTmpBuffer入参传入**两种方式。

-   接口框架申请临时空间，开发者无需申请，但是需要预留临时空间的大小。

-   通过sharedTmpBuffer入参传入，使用该tensor作为临时空间进行处理，接口框架不再申请。该方式开发者可以自行管理sharedTmpBuffer内存空间，并在接口调用完成后，复用该部分内存，内存不会反复申请释放，灵活性较高，内存利用率也较高。

接口框架申请的方式，开发者需要预留临时空间；通过sharedTmpBuffer传入的情况，开发者需要为sharedTmpBuffer申请空间。临时空间大小BufferSize的获取方式如下：通过[GetAscendAntiQuantMaxMinTmpSize](GetAscendAntiQuantMaxMinTmpSize.md)中提供的接口获取需要预留空间的范围大小。

## 参数说明<a name="section622mcpsimp"></a>

**表 1**  模板参数说明

<a name="table575571914269"></a>
<table><thead align="left"><tr id="row18755131942614"><th class="cellrowborder" valign="top" width="19.39%" id="mcps1.2.3.1.1"><p id="p675519193268"><a name="p675519193268"></a><a name="p675519193268"></a>参数名</p>
</th>
<th class="cellrowborder" valign="top" width="80.61%" id="mcps1.2.3.1.2"><p id="p375511918267"><a name="p375511918267"></a><a name="p375511918267"></a>描述</p>
</th>
</tr>
</thead>
<tbody><tr id="row14755141911264"><td class="cellrowborder" valign="top" width="19.39%" headers="mcps1.2.3.1.1 "><p id="p47551198266"><a name="p47551198266"></a><a name="p47551198266"></a>InputDataType</p>
</td>
<td class="cellrowborder" valign="top" width="80.61%" headers="mcps1.2.3.1.2 "><p id="p125969172719"><a name="p125969172719"></a><a name="p125969172719"></a>输入的数据类型。</p>
</td>
</tr>
<tr id="row6356241194912"><td class="cellrowborder" valign="top" width="19.39%" headers="mcps1.2.3.1.1 "><p id="p157510160329"><a name="p157510160329"></a><a name="p157510160329"></a>OutputDataType</p>
</td>
<td class="cellrowborder" valign="top" width="80.61%" headers="mcps1.2.3.1.2 "><p id="p2882344183215"><a name="p2882344183215"></a><a name="p2882344183215"></a>输出的数据类型。</p>
</td>
</tr>
<tr id="row9756719122620"><td class="cellrowborder" valign="top" width="19.39%" headers="mcps1.2.3.1.1 "><p id="p1682112447268"><a name="p1682112447268"></a><a name="p1682112447268"></a>isTranspose</p>
</td>
<td class="cellrowborder" valign="top" width="80.61%" headers="mcps1.2.3.1.2 "><p id="p89797304326"><a name="p89797304326"></a><a name="p89797304326"></a>是否使能输入数据转置。</p>
</td>
</tr>
</tbody>
</table>

**表 2**  PER\_TOKEN/PER\_GROUP b8/float4场景模板参数说明

<a name="table589125485010"></a>
<table><thead align="left"><tr id="row489125425015"><th class="cellrowborder" valign="top" width="19.259999999999998%" id="mcps1.2.3.1.1"><p id="p6891105420508"><a name="p6891105420508"></a><a name="p6891105420508"></a>参数名</p>
</th>
<th class="cellrowborder" valign="top" width="80.74%" id="mcps1.2.3.1.2"><p id="p17891145445014"><a name="p17891145445014"></a><a name="p17891145445014"></a>描述</p>
</th>
</tr>
</thead>
<tbody><tr id="row1589195475011"><td class="cellrowborder" valign="top" width="19.259999999999998%" headers="mcps1.2.3.1.1 "><p id="p188911054155015"><a name="p188911054155015"></a><a name="p188911054155015"></a>dstT</p>
</td>
<td class="cellrowborder" valign="top" width="80.74%" headers="mcps1.2.3.1.2 "><p id="p2182948165115"><a name="p2182948165115"></a><a name="p2182948165115"></a>目的操作数的数据类型。</p>
</td>
</tr>
<tr id="row1891165475020"><td class="cellrowborder" valign="top" width="19.259999999999998%" headers="mcps1.2.3.1.1 "><p id="p68921754135010"><a name="p68921754135010"></a><a name="p68921754135010"></a>srcT</p>
</td>
<td class="cellrowborder" valign="top" width="80.74%" headers="mcps1.2.3.1.2 "><p id="p5182154805113"><a name="p5182154805113"></a><a name="p5182154805113"></a>源操作数的数据类型。</p>
</td>
</tr>
<tr id="row789235465020"><td class="cellrowborder" valign="top" width="19.259999999999998%" headers="mcps1.2.3.1.1 "><p id="p5892135418509"><a name="p5892135418509"></a><a name="p5892135418509"></a>scaleT</p>
</td>
<td class="cellrowborder" valign="top" width="80.74%" headers="mcps1.2.3.1.2 "><p id="p2892165465019"><a name="p2892165465019"></a><a name="p2892165465019"></a>缩放因子scale参数的数据类型。</p>
</td>
</tr>
<tr id="row089212543508"><td class="cellrowborder" valign="top" width="19.259999999999998%" headers="mcps1.2.3.1.1 "><p id="p15892165425015"><a name="p15892165425015"></a><a name="p15892165425015"></a>config</p>
</td>
<td class="cellrowborder" valign="top" width="80.74%" headers="mcps1.2.3.1.2 "><p id="p13892754135017"><a name="p13892754135017"></a><a name="p13892754135017"></a>量化接口配置参数，定义为：</p>
<a name="screen18498115812515"></a><a name="screen18498115812515"></a><pre class="screen" codetype="Cpp" id="screen18498115812515">struct AscendAntiQuantConfig {
        bool hasOffset;
        bool isTranspose;
        int32_t kDim = 1;
 }</pre>
<a name="ul558513317417"></a><a name="ul558513317417"></a><ul id="ul558513317417"><li>hasOffset：量化参数offset是否参与计算。<a name="ul186211135517"></a><a name="ul186211135517"></a><ul id="ul186211135517"><li>True：表示offset参数参与计算。</li><li>False：表示offset参数不参与计算。</li></ul>
</li><li>isTranspose：表示是否使能输入src转置。<a name="ul156001433171912"></a><a name="ul156001433171912"></a><ul id="ul156001433171912"><li>True：表示输入src转置。</li><li>False：表示输入src不转置。</li></ul>
</li><li>kDim：group的计算方向，即k方向。仅在PER_GROUP场景有效，支持的取值如下。<a name="ul980613301459"></a><a name="ul980613301459"></a><ul id="ul980613301459"><li>0：k轴是第0轴，即m方向为group的计算方向；</li><li>1：k轴是第1轴，即n方向为group的计算方向。</li></ul>
</li></ul>
</td>
</tr>
<tr id="row13892205418504"><td class="cellrowborder" valign="top" width="19.259999999999998%" headers="mcps1.2.3.1.1 "><p id="p10892554165011"><a name="p10892554165011"></a><a name="p10892554165011"></a>policy</p>
</td>
<td class="cellrowborder" valign="top" width="80.74%" headers="mcps1.2.3.1.2 "><p id="p10892185420508"><a name="p10892185420508"></a><a name="p10892185420508"></a>量化策略配置参数，枚举类型，可取值如下：</p>
<a name="screen191844253917"></a><a name="screen191844253917"></a><pre class="screen" codetype="Cpp" id="screen191844253917">enum class AscendQuantPolicy : int32_t {
        PER_TENSOR, // 预留参数，暂不支持
        PER_CHANNEL, // 预留参数，暂不支持
        PER_TOKEN, // 配置为PER_TOKEN场景
        PER_GROUP,  // 配置为PER_GROUP场景
        PER_CHANNEL_PER_GROUP, // 预留参数，暂不支持
        PER_TOEKN_PER_GROUP // 预留参数，暂不支持
}</pre>
</td>
</tr>
</tbody>
</table>

**表 3**  PER\_TOKEN/PER\_GROUP b8/float4场景支持的数据类型组合

<a name="table158181847102411"></a>
<table><thead align="left"><tr id="row381964718248"><th class="cellrowborder" valign="top" width="30.683068306830684%" id="mcps1.2.4.1.1"><p id="p1681934711240"><a name="p1681934711240"></a><a name="p1681934711240"></a>srcDtype</p>
</th>
<th class="cellrowborder" valign="top" width="33.17331733173317%" id="mcps1.2.4.1.2"><p id="p4819184792415"><a name="p4819184792415"></a><a name="p4819184792415"></a>scaleDtype/offsetDtype</p>
</th>
<th class="cellrowborder" valign="top" width="36.14361436143614%" id="mcps1.2.4.1.3"><p id="p48194471241"><a name="p48194471241"></a><a name="p48194471241"></a>dstDtype</p>
</th>
</tr>
</thead>
<tbody><tr id="row1881954718248"><td class="cellrowborder" rowspan="5" valign="top" width="30.683068306830684%" headers="mcps1.2.4.1.1 "><p id="p6329646165012"><a name="p6329646165012"></a><a name="p6329646165012"></a>int8_t</p>
</td>
<td class="cellrowborder" valign="top" width="33.17331733173317%" headers="mcps1.2.4.1.2 "><p id="p1881984712245"><a name="p1881984712245"></a><a name="p1881984712245"></a>half</p>
</td>
<td class="cellrowborder" valign="top" width="36.14361436143614%" headers="mcps1.2.4.1.3 "><p id="p15414132419508"><a name="p15414132419508"></a><a name="p15414132419508"></a>half</p>
</td>
</tr>
<tr id="row2819164722415"><td class="cellrowborder" valign="top" headers="mcps1.2.4.1.1 "><p id="p2819104711244"><a name="p2819104711244"></a><a name="p2819104711244"></a>bfloat16_t</p>
</td>
<td class="cellrowborder" valign="top" headers="mcps1.2.4.1.2 "><p id="p341462445013"><a name="p341462445013"></a><a name="p341462445013"></a>bfloat16_t</p>
</td>
</tr>
<tr id="row2819184711242"><td class="cellrowborder" valign="top" headers="mcps1.2.4.1.1 "><p id="p178191447122415"><a name="p178191447122415"></a><a name="p178191447122415"></a>float</p>
</td>
<td class="cellrowborder" valign="top" headers="mcps1.2.4.1.2 "><p id="p13414324155014"><a name="p13414324155014"></a><a name="p13414324155014"></a>float</p>
</td>
</tr>
<tr id="row1481954718242"><td class="cellrowborder" valign="top" headers="mcps1.2.4.1.1 "><p id="p98190478243"><a name="p98190478243"></a><a name="p98190478243"></a>float</p>
</td>
<td class="cellrowborder" valign="top" headers="mcps1.2.4.1.2 "><p id="p64142247502"><a name="p64142247502"></a><a name="p64142247502"></a>half</p>
</td>
</tr>
<tr id="row9819747132418"><td class="cellrowborder" valign="top" headers="mcps1.2.4.1.1 "><p id="p148191847132412"><a name="p148191847132412"></a><a name="p148191847132412"></a>float</p>
</td>
<td class="cellrowborder" valign="top" headers="mcps1.2.4.1.2 "><p id="p64141624175010"><a name="p64141624175010"></a><a name="p64141624175010"></a>bfloat16_t</p>
</td>
</tr>
<tr id="row1181904702416"><td class="cellrowborder" rowspan="5" valign="top" width="30.683068306830684%" headers="mcps1.2.4.1.1 "><p id="p1978211567503"><a name="p1978211567503"></a><a name="p1978211567503"></a>hifloat8_t</p>
</td>
<td class="cellrowborder" valign="top" width="33.17331733173317%" headers="mcps1.2.4.1.2 "><p id="p2974122902715"><a name="p2974122902715"></a><a name="p2974122902715"></a>half</p>
</td>
<td class="cellrowborder" valign="top" width="36.14361436143614%" headers="mcps1.2.4.1.3 "><p id="p1414112412505"><a name="p1414112412505"></a><a name="p1414112412505"></a>half</p>
</td>
</tr>
<tr id="row12819174742414"><td class="cellrowborder" valign="top" headers="mcps1.2.4.1.1 "><p id="p12974329192720"><a name="p12974329192720"></a><a name="p12974329192720"></a>bfloat16_t</p>
</td>
<td class="cellrowborder" valign="top" headers="mcps1.2.4.1.2 "><p id="p1241502411507"><a name="p1241502411507"></a><a name="p1241502411507"></a>bfloat16_t</p>
</td>
</tr>
<tr id="row12820134710244"><td class="cellrowborder" valign="top" headers="mcps1.2.4.1.1 "><p id="p19974132972710"><a name="p19974132972710"></a><a name="p19974132972710"></a>float</p>
</td>
<td class="cellrowborder" valign="top" headers="mcps1.2.4.1.2 "><p id="p84151624115010"><a name="p84151624115010"></a><a name="p84151624115010"></a>float</p>
</td>
</tr>
<tr id="row1382034714246"><td class="cellrowborder" valign="top" headers="mcps1.2.4.1.1 "><p id="p119742029112718"><a name="p119742029112718"></a><a name="p119742029112718"></a>float</p>
</td>
<td class="cellrowborder" valign="top" headers="mcps1.2.4.1.2 "><p id="p1941512410504"><a name="p1941512410504"></a><a name="p1941512410504"></a>half</p>
</td>
</tr>
<tr id="row16820147142414"><td class="cellrowborder" valign="top" headers="mcps1.2.4.1.1 "><p id="p2097432917271"><a name="p2097432917271"></a><a name="p2097432917271"></a>float</p>
</td>
<td class="cellrowborder" valign="top" headers="mcps1.2.4.1.2 "><p id="p14415192420507"><a name="p14415192420507"></a><a name="p14415192420507"></a>bfloat16_t</p>
</td>
</tr>
<tr id="row6820144717249"><td class="cellrowborder" rowspan="5" valign="top" width="30.683068306830684%" headers="mcps1.2.4.1.1 "><p id="p1666914411278"><a name="p1666914411278"></a><a name="p1666914411278"></a>fp8_e5m2_t/fp8_e4m3fn_t</p>
<p id="p20886140149"><a name="p20886140149"></a><a name="p20886140149"></a></p>
<p id="p108861471410"><a name="p108861471410"></a><a name="p108861471410"></a></p>
<p id="p12886642148"><a name="p12886642148"></a><a name="p12886642148"></a></p>
<p id="p488613421414"><a name="p488613421414"></a><a name="p488613421414"></a></p>
</td>
<td class="cellrowborder" valign="top" width="33.17331733173317%" headers="mcps1.2.4.1.2 "><p id="p18669164419276"><a name="p18669164419276"></a><a name="p18669164419276"></a>half</p>
</td>
<td class="cellrowborder" valign="top" width="36.14361436143614%" headers="mcps1.2.4.1.3 "><p id="p144151924185019"><a name="p144151924185019"></a><a name="p144151924185019"></a>half</p>
</td>
</tr>
<tr id="row38201547172420"><td class="cellrowborder" valign="top" headers="mcps1.2.4.1.1 "><p id="p0669154418274"><a name="p0669154418274"></a><a name="p0669154418274"></a>bfloat16_t</p>
</td>
<td class="cellrowborder" valign="top" headers="mcps1.2.4.1.2 "><p id="p241562417509"><a name="p241562417509"></a><a name="p241562417509"></a>bfloat16_t</p>
</td>
</tr>
<tr id="row138201347102416"><td class="cellrowborder" valign="top" headers="mcps1.2.4.1.1 "><p id="p46699449270"><a name="p46699449270"></a><a name="p46699449270"></a>float</p>
</td>
<td class="cellrowborder" valign="top" headers="mcps1.2.4.1.2 "><p id="p10415112416508"><a name="p10415112416508"></a><a name="p10415112416508"></a>float</p>
</td>
</tr>
<tr id="row1882010471247"><td class="cellrowborder" valign="top" headers="mcps1.2.4.1.1 "><p id="p1669184415276"><a name="p1669184415276"></a><a name="p1669184415276"></a>float</p>
</td>
<td class="cellrowborder" valign="top" headers="mcps1.2.4.1.2 "><p id="p941592465012"><a name="p941592465012"></a><a name="p941592465012"></a>half</p>
</td>
</tr>
<tr id="row188202475248"><td class="cellrowborder" valign="top" headers="mcps1.2.4.1.1 "><p id="p176691449279"><a name="p176691449279"></a><a name="p176691449279"></a>float</p>
</td>
<td class="cellrowborder" valign="top" headers="mcps1.2.4.1.2 "><p id="p241614246509"><a name="p241614246509"></a><a name="p241614246509"></a>bfloat16_t</p>
</td>
</tr>
<tr id="row9149111019142"><td class="cellrowborder" rowspan="2" valign="top" width="30.683068306830684%" headers="mcps1.2.4.1.1 "><p id="p18231111621415"><a name="p18231111621415"></a><a name="p18231111621415"></a>fp4x2_e1m2_t/fp4x2_e2m1_t</p>
<p id="p17289125144216"><a name="p17289125144216"></a><a name="p17289125144216"></a>（当前均只支持PER_GROUP场景）</p>
</td>
<td class="cellrowborder" rowspan="2" valign="top" width="33.17331733173317%" headers="mcps1.2.4.1.2 "><p id="p109091731111414"><a name="p109091731111414"></a><a name="p109091731111414"></a>fp8_e8m0_t</p>
</td>
<td class="cellrowborder" valign="top" width="36.14361436143614%" headers="mcps1.2.4.1.3 "><p id="p1314911011145"><a name="p1314911011145"></a><a name="p1314911011145"></a>half</p>
</td>
</tr>
<tr id="row1972181219144"><td class="cellrowborder" valign="top" headers="mcps1.2.4.1.1 "><p id="p37211512111416"><a name="p37211512111416"></a><a name="p37211512111416"></a>bfloat16_t</p>
</td>
</tr>
</tbody>
</table>

**表 4**  接口参数说明

<a name="table44731299481"></a>
<table><thead align="left"><tr id="row247482914489"><th class="cellrowborder" valign="top" width="16.45%" id="mcps1.2.4.1.1"><p id="p147413295483"><a name="p147413295483"></a><a name="p147413295483"></a>参数名</p>
</th>
<th class="cellrowborder" valign="top" width="11.31%" id="mcps1.2.4.1.2"><p id="p1147432994819"><a name="p1147432994819"></a><a name="p1147432994819"></a>输入/输出</p>
</th>
<th class="cellrowborder" valign="top" width="72.24000000000001%" id="mcps1.2.4.1.3"><p id="p74749297483"><a name="p74749297483"></a><a name="p74749297483"></a>描述</p>
</th>
</tr>
</thead>
<tbody><tr id="row12474329104814"><td class="cellrowborder" valign="top" width="16.45%" headers="mcps1.2.4.1.1 "><p id="p1047411294482"><a name="p1047411294482"></a><a name="p1047411294482"></a>dst</p>
</td>
<td class="cellrowborder" valign="top" width="11.31%" headers="mcps1.2.4.1.2 "><p id="p047412984813"><a name="p047412984813"></a><a name="p047412984813"></a>输出</p>
</td>
<td class="cellrowborder" valign="top" width="72.24000000000001%" headers="mcps1.2.4.1.3 "><p id="p3989161814016"><a name="p3989161814016"></a><a name="p3989161814016"></a>目的操作数。</p>
<p id="p1747492917489"><a name="p1747492917489"></a><a name="p1747492917489"></a><span id="zh-cn_topic_0000002523303824_ph173308471594"><a name="zh-cn_topic_0000002523303824_ph173308471594"></a><a name="zh-cn_topic_0000002523303824_ph173308471594"></a><span id="zh-cn_topic_0000002523303824_ph9902231466"><a name="zh-cn_topic_0000002523303824_ph9902231466"></a><a name="zh-cn_topic_0000002523303824_ph9902231466"></a><span id="zh-cn_topic_0000002523303824_ph1782115034816"><a name="zh-cn_topic_0000002523303824_ph1782115034816"></a><a name="zh-cn_topic_0000002523303824_ph1782115034816"></a>类型为<a href="LocalTensor.md">LocalTensor</a>，支持的TPosition为VECIN/VECCALC/VECOUT。</span></span></span></p>
<p id="p12247148298"><a name="p12247148298"></a><a name="p12247148298"></a><span id="ph152477481396"><a name="ph152477481396"></a><a name="ph152477481396"></a>Ascend 950PR/Ascend 950DT</span>，支持的数据类型为：half、bfloat16_t。</p>
</td>
</tr>
<tr id="row18474729124817"><td class="cellrowborder" valign="top" width="16.45%" headers="mcps1.2.4.1.1 "><p id="p54741029164810"><a name="p54741029164810"></a><a name="p54741029164810"></a>src</p>
</td>
<td class="cellrowborder" valign="top" width="11.31%" headers="mcps1.2.4.1.2 "><p id="p144741829194814"><a name="p144741829194814"></a><a name="p144741829194814"></a>输入</p>
</td>
<td class="cellrowborder" valign="top" width="72.24000000000001%" headers="mcps1.2.4.1.3 "><p id="p6914123244017"><a name="p6914123244017"></a><a name="p6914123244017"></a>源操作数。</p>
<p id="p1493334184019"><a name="p1493334184019"></a><a name="p1493334184019"></a><span id="zh-cn_topic_0000002523303824_ph173308471594_1"><a name="zh-cn_topic_0000002523303824_ph173308471594_1"></a><a name="zh-cn_topic_0000002523303824_ph173308471594_1"></a><span id="zh-cn_topic_0000002523303824_ph9902231466_1"><a name="zh-cn_topic_0000002523303824_ph9902231466_1"></a><a name="zh-cn_topic_0000002523303824_ph9902231466_1"></a><span id="zh-cn_topic_0000002523303824_ph1782115034816_1"><a name="zh-cn_topic_0000002523303824_ph1782115034816_1"></a><a name="zh-cn_topic_0000002523303824_ph1782115034816_1"></a>类型为<a href="LocalTensor.md">LocalTensor</a>，支持的TPosition为VECIN/VECCALC/VECOUT。</span></span></span></p>
<p id="p42851823151714"><a name="p42851823151714"></a><a name="p42851823151714"></a><span id="ph142852231172"><a name="ph142852231172"></a><a name="ph142852231172"></a>Ascend 950PR/Ascend 950DT</span>，PER_CHANNEL和PER_TENSOR场景下支持的数据类型为：int8_t、fp8_e4m3fn_t、fp8_e5m2_t、hifloat8_t，PER_GROUP float4场景下支持的数据类型为：fp4x2_e2m1_t、fp4x2_e1m2_t。</p>
</td>
</tr>
<tr id="row2891175273813"><td class="cellrowborder" valign="top" width="16.45%" headers="mcps1.2.4.1.1 "><p id="p118911521386"><a name="p118911521386"></a><a name="p118911521386"></a>offset</p>
</td>
<td class="cellrowborder" valign="top" width="11.31%" headers="mcps1.2.4.1.2 "><p id="p78911552113812"><a name="p78911552113812"></a><a name="p78911552113812"></a>输入</p>
</td>
<td class="cellrowborder" valign="top" width="72.24000000000001%" headers="mcps1.2.4.1.3 "><p id="p1237485874015"><a name="p1237485874015"></a><a name="p1237485874015"></a>输入数据反量化时的偏移量。</p>
<p id="p6374105824017"><a name="p6374105824017"></a><a name="p6374105824017"></a><span id="zh-cn_topic_0000002523303824_ph173308471594_2"><a name="zh-cn_topic_0000002523303824_ph173308471594_2"></a><a name="zh-cn_topic_0000002523303824_ph173308471594_2"></a><span id="zh-cn_topic_0000002523303824_ph9902231466_2"><a name="zh-cn_topic_0000002523303824_ph9902231466_2"></a><a name="zh-cn_topic_0000002523303824_ph9902231466_2"></a><span id="zh-cn_topic_0000002523303824_ph1782115034816_2"><a name="zh-cn_topic_0000002523303824_ph1782115034816_2"></a><a name="zh-cn_topic_0000002523303824_ph1782115034816_2"></a>类型为<a href="LocalTensor.md">LocalTensor</a>，支持的TPosition为VECIN/VECCALC/VECOUT。</span></span></span></p>
<p id="p117351349203919"><a name="p117351349203919"></a><a name="p117351349203919"></a><span id="ph127351049183914"><a name="ph127351049183914"></a><a name="ph127351049183914"></a>Ascend 950PR/Ascend 950DT</span>，支持的数据类型为：half、bfloat16_t。</p>
</td>
</tr>
<tr id="row617218172310"><td class="cellrowborder" valign="top" width="16.45%" headers="mcps1.2.4.1.1 "><p id="p16660132211315"><a name="p16660132211315"></a><a name="p16660132211315"></a>scale</p>
</td>
<td class="cellrowborder" valign="top" width="11.31%" headers="mcps1.2.4.1.2 "><p id="p156601822153115"><a name="p156601822153115"></a><a name="p156601822153115"></a>输入</p>
</td>
<td class="cellrowborder" valign="top" width="72.24000000000001%" headers="mcps1.2.4.1.3 "><p id="p6660152253112"><a name="p6660152253112"></a><a name="p6660152253112"></a>输入数据反量化时的缩放因子。</p>
<p id="p76604221311"><a name="p76604221311"></a><a name="p76604221311"></a><span id="zh-cn_topic_0000002523303824_ph173308471594_3"><a name="zh-cn_topic_0000002523303824_ph173308471594_3"></a><a name="zh-cn_topic_0000002523303824_ph173308471594_3"></a><span id="zh-cn_topic_0000002523303824_ph9902231466_3"><a name="zh-cn_topic_0000002523303824_ph9902231466_3"></a><a name="zh-cn_topic_0000002523303824_ph9902231466_3"></a><span id="zh-cn_topic_0000002523303824_ph1782115034816_3"><a name="zh-cn_topic_0000002523303824_ph1782115034816_3"></a><a name="zh-cn_topic_0000002523303824_ph1782115034816_3"></a>类型为<a href="LocalTensor.md">LocalTensor</a>，支持的TPosition为VECIN/VECCALC/VECOUT。</span></span></span></p>
<p id="p1535511954417"><a name="p1535511954417"></a><a name="p1535511954417"></a><span id="ph6355419154419"><a name="ph6355419154419"></a><a name="ph6355419154419"></a>Ascend 950PR/Ascend 950DT</span>，PER_CHANNEL和PER_TENSOR场景下支持的数据类型为：half、bfloat16_t。PER_GROUP float4场景下支持的数据类型为：fp8_e8m0_t。</p>
</td>
</tr>
<tr id="row1747412296483"><td class="cellrowborder" valign="top" width="16.45%" headers="mcps1.2.4.1.1 "><p id="p74741029204817"><a name="p74741029204817"></a><a name="p74741029204817"></a>sharedTmpBuffer</p>
</td>
<td class="cellrowborder" valign="top" width="11.31%" headers="mcps1.2.4.1.2 "><p id="p1747452954810"><a name="p1747452954810"></a><a name="p1747452954810"></a>输入</p>
</td>
<td class="cellrowborder" valign="top" width="72.24000000000001%" headers="mcps1.2.4.1.3 "><p id="p191160465422"><a name="p191160465422"></a><a name="p191160465422"></a>临时缓存。</p>
<p id="p979635010404"><a name="p979635010404"></a><a name="p979635010404"></a><span id="zh-cn_topic_0000002523303824_ph173308471594_4"><a name="zh-cn_topic_0000002523303824_ph173308471594_4"></a><a name="zh-cn_topic_0000002523303824_ph173308471594_4"></a><span id="zh-cn_topic_0000002523303824_ph9902231466_4"><a name="zh-cn_topic_0000002523303824_ph9902231466_4"></a><a name="zh-cn_topic_0000002523303824_ph9902231466_4"></a><span id="zh-cn_topic_0000002523303824_ph1782115034816_4"><a name="zh-cn_topic_0000002523303824_ph1782115034816_4"></a><a name="zh-cn_topic_0000002523303824_ph1782115034816_4"></a>类型为<a href="LocalTensor.md">LocalTensor</a>，支持的TPosition为VECIN/VECCALC/VECOUT。</span></span></span></p>
<p id="p5881016172817"><a name="p5881016172817"></a><a name="p5881016172817"></a>临时空间大小BufferSize的获取方式请参考<a href="GetAscendAntiQuantMaxMinTmpSize.md">GetAscendAntiQuantMaxMinTmpSize</a>。</p>
</td>
</tr>
<tr id="row13807993916"><td class="cellrowborder" valign="top" width="16.45%" headers="mcps1.2.4.1.1 "><p id="p338118913918"><a name="p338118913918"></a><a name="p338118913918"></a>k</p>
</td>
<td class="cellrowborder" valign="top" width="11.31%" headers="mcps1.2.4.1.2 "><p id="p1516416464400"><a name="p1516416464400"></a><a name="p1516416464400"></a>输入</p>
</td>
<td class="cellrowborder" valign="top" width="72.24000000000001%" headers="mcps1.2.4.1.3 "><p id="p6828105324810"><a name="p6828105324810"></a><a name="p6828105324810"></a>isTranspose为true时，src的shape为[N,K]；isTranspose为false时，src的shape为[K,N]。</p>
<p id="p461124514213"><a name="p461124514213"></a><a name="p461124514213"></a>参数k对应其中的K值。</p>
</td>
</tr>
<tr id="row16581165711547"><td class="cellrowborder" valign="top" width="16.45%" headers="mcps1.2.4.1.1 "><p id="p105824579541"><a name="p105824579541"></a><a name="p105824579541"></a>shapeInfo</p>
</td>
<td class="cellrowborder" valign="top" width="11.31%" headers="mcps1.2.4.1.2 "><p id="p258225710542"><a name="p258225710542"></a><a name="p258225710542"></a>输入</p>
</td>
<td class="cellrowborder" valign="top" width="72.24000000000001%" headers="mcps1.2.4.1.3 "><p id="p180617329136"><a name="p180617329136"></a><a name="p180617329136"></a>设置参数offset和scale的shape信息，仅PER_CHANNEL场景（按通道量化）需要配置。</p>
<p id="p151661918101414"><a name="p151661918101414"></a><a name="p151661918101414"></a>可选参数。在PER_CHANNEL场景，如果未传入该参数或者结构体中数据设置为0，将从offset和scale的<a href="ShapeInfo.md">ShapeInfo</a>中获取offset和scale的shape信息。</p>
<p id="p9967159937"><a name="p9967159937"></a><a name="p9967159937"></a>AntiQuantShapeInfo类型，定义如下：</p>
<a name="screen1823961113712"></a><a name="screen1823961113712"></a><pre class="screen" codetype="Cpp" id="screen1823961113712">struct AntiQuantShapeInfo {
    uint32_t offsetHeight{0};  // offset 的高
    uint32_t offsetWidth{0};  // offset 的宽
    uint32_t scaleHeight{0};  // scale 的高
    uint32_t scaleWidth{0};  // scale 的宽
};</pre>
</td>
</tr>
</tbody>
</table>

**表 5**  PER\_TOKEN/PER\_GROUP b8/float4场景接口参数说明

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
<p id="p320416920413"><a name="p320416920413"></a><a name="p320416920413"></a><span id="zh-cn_topic_0000002523303824_ph173308471594_5"><a name="zh-cn_topic_0000002523303824_ph173308471594_5"></a><a name="zh-cn_topic_0000002523303824_ph173308471594_5"></a><span id="zh-cn_topic_0000002523303824_ph9902231466_5"><a name="zh-cn_topic_0000002523303824_ph9902231466_5"></a><a name="zh-cn_topic_0000002523303824_ph9902231466_5"></a><span id="zh-cn_topic_0000002523303824_ph1782115034816_5"><a name="zh-cn_topic_0000002523303824_ph1782115034816_5"></a><a name="zh-cn_topic_0000002523303824_ph1782115034816_5"></a>类型为<a href="LocalTensor.md">LocalTensor</a>，支持的TPosition为VECIN/VECCALC/VECOUT。</span></span></span></p>
<p id="p112044915418"><a name="p112044915418"></a><a name="p112044915418"></a><span id="ph520420914410"><a name="ph520420914410"></a><a name="ph520420914410"></a>Ascend 950PR/Ascend 950DT</span>，支持的数据类型为：half、bfloat16_t、float。</p>
</td>
</tr>
<tr id="row123522518116"><td class="cellrowborder" valign="top" width="16.661666166616662%" headers="mcps1.2.4.1.1 "><p id="p63525511816"><a name="p63525511816"></a><a name="p63525511816"></a>srcTensor</p>
</td>
<td class="cellrowborder" valign="top" width="10.99109910991099%" headers="mcps1.2.4.1.2 "><p id="p14352105115117"><a name="p14352105115117"></a><a name="p14352105115117"></a>输入</p>
</td>
<td class="cellrowborder" valign="top" width="72.34723472347235%" headers="mcps1.2.4.1.3 "><p id="p1320459948"><a name="p1320459948"></a><a name="p1320459948"></a>源操作数。</p>
<p id="p82041993416"><a name="p82041993416"></a><a name="p82041993416"></a><span id="zh-cn_topic_0000002523303824_ph173308471594_6"><a name="zh-cn_topic_0000002523303824_ph173308471594_6"></a><a name="zh-cn_topic_0000002523303824_ph173308471594_6"></a><span id="zh-cn_topic_0000002523303824_ph9902231466_6"><a name="zh-cn_topic_0000002523303824_ph9902231466_6"></a><a name="zh-cn_topic_0000002523303824_ph9902231466_6"></a><span id="zh-cn_topic_0000002523303824_ph1782115034816_6"><a name="zh-cn_topic_0000002523303824_ph1782115034816_6"></a><a name="zh-cn_topic_0000002523303824_ph1782115034816_6"></a>类型为<a href="LocalTensor.md">LocalTensor</a>，支持的TPosition为VECIN/VECCALC/VECOUT。</span></span></span></p>
<p id="p1204791548"><a name="p1204791548"></a><a name="p1204791548"></a><span id="ph920413915412"><a name="ph920413915412"></a><a name="ph920413915412"></a>Ascend 950PR/Ascend 950DT</span>，支持的数据类型为：int8_t、fp8_e4m3fn_t、fp8_e5m2_t、hifloat8_t、fp4x2_e1m2_t、fp4x2_e2m1_t。注意，对于fp4x2_e1m2_t、fp4x2_e2m1_t数据类型，仅在PER_GROUP场景下支持。</p>
</td>
</tr>
<tr id="row735217510112"><td class="cellrowborder" valign="top" width="16.661666166616662%" headers="mcps1.2.4.1.1 "><p id="p135215512117"><a name="p135215512117"></a><a name="p135215512117"></a>sharedTmpBuffer</p>
</td>
<td class="cellrowborder" valign="top" width="10.99109910991099%" headers="mcps1.2.4.1.2 "><p id="p1635245114114"><a name="p1635245114114"></a><a name="p1635245114114"></a>输入</p>
</td>
<td class="cellrowborder" valign="top" width="72.34723472347235%" headers="mcps1.2.4.1.3 "><p id="p6205129543"><a name="p6205129543"></a><a name="p6205129543"></a>临时缓存。</p>
<p id="p1120549049"><a name="p1120549049"></a><a name="p1120549049"></a><span id="zh-cn_topic_0000002523303824_ph173308471594_7"><a name="zh-cn_topic_0000002523303824_ph173308471594_7"></a><a name="zh-cn_topic_0000002523303824_ph173308471594_7"></a><span id="zh-cn_topic_0000002523303824_ph9902231466_7"><a name="zh-cn_topic_0000002523303824_ph9902231466_7"></a><a name="zh-cn_topic_0000002523303824_ph9902231466_7"></a><span id="zh-cn_topic_0000002523303824_ph1782115034816_7"><a name="zh-cn_topic_0000002523303824_ph1782115034816_7"></a><a name="zh-cn_topic_0000002523303824_ph1782115034816_7"></a>类型为<a href="LocalTensor.md">LocalTensor</a>，支持的TPosition为VECIN/VECCALC/VECOUT。</span></span></span></p>
<p id="p62051192418"><a name="p62051192418"></a><a name="p62051192418"></a>临时空间大小BufferSize的获取方式请参考<a href="GetAscendQuantMaxMinTmpSize.md">GetAscendQuantMaxMinTmpSize</a>。</p>
<p id="p152052091048"><a name="p152052091048"></a><a name="p152052091048"></a><span id="ph142051991410"><a name="ph142051991410"></a><a name="ph142051991410"></a>Ascend 950PR/Ascend 950DT</span>，支持的数据类型为：uint8_t。</p>
</td>
</tr>
<tr id="row1235211516111"><td class="cellrowborder" valign="top" width="16.661666166616662%" headers="mcps1.2.4.1.1 "><p id="p5352185119112"><a name="p5352185119112"></a><a name="p5352185119112"></a>scaleTensor</p>
</td>
<td class="cellrowborder" valign="top" width="10.99109910991099%" headers="mcps1.2.4.1.2 "><p id="p153521051117"><a name="p153521051117"></a><a name="p153521051117"></a>输入</p>
</td>
<td class="cellrowborder" valign="top" width="72.34723472347235%" headers="mcps1.2.4.1.3 "><p id="p6205796411"><a name="p6205796411"></a><a name="p6205796411"></a>量化参数scale。</p>
<p id="p1205119748"><a name="p1205119748"></a><a name="p1205119748"></a><span id="zh-cn_topic_0000002523303824_ph173308471594_8"><a name="zh-cn_topic_0000002523303824_ph173308471594_8"></a><a name="zh-cn_topic_0000002523303824_ph173308471594_8"></a><span id="zh-cn_topic_0000002523303824_ph9902231466_8"><a name="zh-cn_topic_0000002523303824_ph9902231466_8"></a><a name="zh-cn_topic_0000002523303824_ph9902231466_8"></a><span id="zh-cn_topic_0000002523303824_ph1782115034816_8"><a name="zh-cn_topic_0000002523303824_ph1782115034816_8"></a><a name="zh-cn_topic_0000002523303824_ph1782115034816_8"></a>类型为<a href="LocalTensor.md">LocalTensor</a>，支持的TPosition为VECIN/VECCALC/VECOUT。</span></span></span></p>
<p id="p162051891948"><a name="p162051891948"></a><a name="p162051891948"></a><span id="ph142052912417"><a name="ph142052912417"></a><a name="ph142052912417"></a>Ascend 950PR/Ascend 950DT</span>，支持的数据类型为：half、float、bfloat16_t、fp8_e8m0_t。</p>
</td>
</tr>
<tr id="row135285114116"><td class="cellrowborder" valign="top" width="16.661666166616662%" headers="mcps1.2.4.1.1 "><p id="p1352175111110"><a name="p1352175111110"></a><a name="p1352175111110"></a>offsetTensor</p>
</td>
<td class="cellrowborder" valign="top" width="10.99109910991099%" headers="mcps1.2.4.1.2 "><p id="p33537511216"><a name="p33537511216"></a><a name="p33537511216"></a>输入</p>
</td>
<td class="cellrowborder" valign="top" width="72.34723472347235%" headers="mcps1.2.4.1.3 "><p id="p132061398413"><a name="p132061398413"></a><a name="p132061398413"></a>量化参数offset。</p>
<p id="p020689945"><a name="p020689945"></a><a name="p020689945"></a><span id="zh-cn_topic_0000002523303824_ph173308471594_9"><a name="zh-cn_topic_0000002523303824_ph173308471594_9"></a><a name="zh-cn_topic_0000002523303824_ph173308471594_9"></a><span id="zh-cn_topic_0000002523303824_ph9902231466_9"><a name="zh-cn_topic_0000002523303824_ph9902231466_9"></a><a name="zh-cn_topic_0000002523303824_ph9902231466_9"></a><span id="zh-cn_topic_0000002523303824_ph1782115034816_9"><a name="zh-cn_topic_0000002523303824_ph1782115034816_9"></a><a name="zh-cn_topic_0000002523303824_ph1782115034816_9"></a>类型为<a href="LocalTensor.md">LocalTensor</a>，支持的TPosition为VECIN/VECCALC/VECOUT。</span></span></span></p>
<p id="p122061191946"><a name="p122061191946"></a><a name="p122061191946"></a><span id="ph14206896416"><a name="ph14206896416"></a><a name="ph14206896416"></a>Ascend 950PR/Ascend 950DT</span>，支持的数据类型和scaleTensor保持一致。对于float4场景，offsetTensor不生效。</p>
</td>
</tr>
<tr id="row1235316511515"><td class="cellrowborder" valign="top" width="16.661666166616662%" headers="mcps1.2.4.1.1 "><p id="p13533516110"><a name="p13533516110"></a><a name="p13533516110"></a>para</p>
</td>
<td class="cellrowborder" valign="top" width="10.99109910991099%" headers="mcps1.2.4.1.2 "><p id="p17353115110118"><a name="p17353115110118"></a><a name="p17353115110118"></a>输入</p>
</td>
<td class="cellrowborder" valign="top" width="72.34723472347235%" headers="mcps1.2.4.1.3 "><p id="p535310512112"><a name="p535310512112"></a><a name="p535310512112"></a>量化接口的参数，AscendAntiQuantParam类型，具体定义如下：</p>
<a name="screen1110516458366"></a><a name="screen1110516458366"></a><pre class="screen" codetype="Cpp" id="screen1110516458366">struct AscendAntiQuantParam {
        uint32_t m;
        uint32_t n;
        uint32_t calCount;
        uint32_t groupSize = 0;
}</pre>
<a name="ul162019226375"></a><a name="ul162019226375"></a><ul id="ul162019226375"><li>m：m方向元素个数。</li><li>n：n方向元素个数。n值对应的数据大小需满足32B对齐的要求，即shape最后一维为n的输入输出均需要满足该维度上32B对齐的要求。</li><li>calCount:参与计算的元素个数。calCount必须是n的整数倍。</li><li>groupSize：PER_GROUP场景有效，表示groupSize行/列数据共用一个scale/offset。groupSize的取值必须大于0且是32的整倍数。</li></ul>
</td>
</tr>
</tbody>
</table>

## 返回值说明<a name="section38228281712"></a>

无

## 约束说明<a name="section633mcpsimp"></a>

-   **不支持源操作数与目的操作数地址重叠。**
-   操作数地址对齐要求请参见[通用地址对齐约束](通用说明和约束.md#section796754519912)。
-   输入输出操作数参与计算的数据长度要求32B对齐。
-   输入带转置场景，k需要32B对齐。
-   调用接口前，确保输入数据的size正确，offset和scale的size和shape正确。
-   PER\_TOKEN/PER\_GROUP b8/float4场景，连续计算方向（即n方向）的数据量要求32B对齐。

## 调用示例<a name="section642mcpsimp"></a>

完整的调用样例可参考[AscendAntiQuant样例](https://gitcode.com/cann/asc-devkit/tree/master/examples/03_libraries/04_quantization/antiquant)。

```
// dstLocal：结果张量
// srcLocal：量化输入
// offsetLocal：偏移参数
// scaleLocal：缩放参数
// sharedTmpBuffer：开发者管理的临时缓冲区，用于存放内部计算中的中间变量
// k：
// shapeInfo：
AscendC::AntiQuantShapeInfo shapeInfo = {1, elementCountOfOffset, 1, elementCountOfOffset};
AscendC::AscendAntiQuant<InputType, OutType, false>(dstLocal, srcLocal, offsetLocal, scaleLocal, sharedTmpBuffer, k, shapeInfo);
```

结果示例如下：

```
输入数据src（shape为[2,64]，非转置场景）:  
[1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1
 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1
 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1
 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1]
offset（shape为[1,64]）:  
[2. 2. 2. 2. 2. 2. 2. 2. 2. 2. 2. 2. 2. 2. 2. 2. 2. 2. 2. 2. 2. 2. 2. 2.
 2. 2. 2. 2. 2. 2. 2. 2. 2. 2. 2. 2. 2. 2. 2. 2. 2. 2. 2. 2. 2. 2. 2. 2.
 2. 2. 2. 2. 2. 2. 2. 2. 2. 2. 2. 2. 2. 2. 2. 2.]
scale（shape为[1,64]）:  
[3. 3. 3. 3. 3. 3. 3. 3. 3. 3. 3. 3. 3. 3. 3. 3. 3. 3. 3. 3. 3. 3. 3. 3.
 3. 3. 3. 3. 3. 3. 3. 3. 3. 3. 3. 3. 3. 3. 3. 3. 3. 3. 3. 3. 3. 3. 3. 3.
 3. 3. 3. 3. 3. 3. 3. 3. 3. 3. 3. 3. 3. 3. 3. 3.]
输出数据dstLocal（shape为[2,64]）:  
[9. 9. 9. 9. 9. 9. 9. 9. 9. 9. 9. 9. 9. 9. 9. 9. 9. 9. 9. 9. 9. 9. 9. 9.
 9. 9. 9. 9. 9. 9. 9. 9. 9. 9. 9. 9. 9. 9. 9. 9. 9. 9. 9. 9. 9. 9. 9. 9.
 9. 9. 9. 9. 9. 9. 9. 9. 9. 9. 9. 9. 9. 9. 9. 9. 9. 9. 9. 9. 9. 9. 9. 9.
 9. 9. 9. 9. 9. 9. 9. 9. 9. 9. 9. 9. 9. 9. 9. 9. 9. 9. 9. 9. 9. 9. 9. 9.
 9. 9. 9. 9. 9. 9. 9. 9. 9. 9. 9. 9. 9. 9. 9. 9. 9. 9. 9. 9. 9. 9. 9. 9.
 9. 9. 9. 9. 9. 9. 9. 9.]
```

PER\_TOKEN/PER\_GROUP b8场景调用示例如下。

```
// 注意m,n需从外部传入
constexpr static bool isReuseSource = false;
constexpr static AscendAntiQuantConfig config = {has_offset, has_transpose, -1};
constexpr static AscendAntiQuantPolicy policy = AscendAntiQuantPolicy::PER_TOKEN;
AscendAntiQuantParam para;
para.m = m;
para.n = n;
para.calCount = calCount;
AscendAntiQuant<dstType, srcType, scaleType, config, policy>(dstLocal, srcLocal, scaleLocal, offsetLocal, para);
```

