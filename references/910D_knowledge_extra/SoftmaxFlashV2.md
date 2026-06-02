# SoftmaxFlashV2<a name="ZH-CN_TOPIC_0000002554344503"></a>

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

SoftmaxFlash增强版本，对应FlashAttention-2算法。将输入tensor\[m<sub>0</sub>, m<sub>1</sub>, ...m<sub>t</sub>, n\]（t大于等于0）的非尾轴长度相乘的结果看作m，则输入tensor的shape看作\[m, n\]。对输入tensor\[m,n\]按行做如下计算，不同的update值对应不同的计算公式，其中x、inmax和insum为输入，M、S、E均为输出。

-   update为false：

    <!-- img2text -->
$$
M_i = \max_j(x_{ij})
$$

$$
E_{ij} = \exp(x_{ij} - M_i)
$$

$$
S_i = \sum_j E_{ij}
$$

-   update为true：

    <!-- img2text -->
$$
\mathrm{dst} = \frac{\exp(\mathrm{src} - \mathrm{inmax})}{\mathrm{insum}}
$$

当输入shape为ND格式时，内部的reduce过程按last轴进行；当输入shape为NZ格式时，内部的reduce过程按照last轴和first轴进行，reduce过程可以参考[SoftMax](SoftMax.md)中的图示说明。

为方便理解，通过Python脚本实现的方式，表达其计算公式如下，其中src、inmax、 insum、update为输入，dst、x\_sum、x\_max、exp\_max为输出。

```
def softmax_flash_2(src, inmax=None, insum=None, update=None):
    if update == None:
        x_max = np.max(src, axis=-1, keepdims=True)
        x_sub = src - x_max   
        dst = np.exp(x_sub) 
        x_sum = np.sum(dst, axis=-1, keepdims=True)
        exp_max = None
        return dst, x_max, x_sum, exp_max
    else:
        x_max = np.max(np.concatenate((inmax, src), axis=-1), axis=-1, keepdims=True)
        dst = np.exp(src - x_max)
        exp_max = np.exp(inmax - x_max)
        x_sum = np.sum(dst, axis=-1, keepdims=True)
        x_sum = exp_max * insum +  x_sum
        return dst, x_max, x_sum, exp_max
```

## 实现原理<a name="section13229175017585"></a>

以float类型，ND格式，shape为\[m, k\]的输入Tensor为例，描述SoftmaxFlashV2高阶API内部算法框图，如下图所示。

**图 1**  SoftmaxFlashV2算法框图<a name="fig137619450117"></a>  
<!-- img2text -->
```text
                                   ┌─────────┐
                                   │ x[m,k]  │
                                   └────┬────┘
                                        │
                                  ┌─────▼─────┐
                                  │ isUpdate  │
                                  └───┬───┬───┘
                                      │   │
                                False │   │ True
                                      │   │

┌─────────────────────────────────────────────┐   ┌─────────────────────────────────────────────────────┐
│                                             │   │                                                     │
│  ┌───────────────────────────────┐          │   │   ┌───────────────────────────────┐                │
│  │ reducemax                     │          │   │   │ reducemax                     │                │
│  │ ([m,k]->[m,1])                │          │   │   │ ([m,k]->[m,1])                │                │
│  └───────────────┬───────────────┘          │   │   └───────────────┬───────────────┘                │
│                  │                          │   │                   │                                │
│  ┌───────────────▼───────────────┐          │   │   ┌───────────────▼───────────────┐                │
│  │ broadcast                     │──────┐   │   │   │ max=broadcast                 │                │
│  │ ([m,1]->[m,8])                │      │   │   │   │ ([m,1]->[m,8])                │                │
│  └───────────────┬───────────────┘      │   │   │   └───────────────┬───────────────┘                │
│                  │                      │   │   │                   │                                │
│  ┌───────────────▼───────────────┐      │   │   │   ┌───────────────▼───────────────┐                │
│  │ y=sub                        │      │   │   │   │ max                           │──────────────┐ │
│  │ (x[m,k] - max[m,8])          │      │   │   │   │ (max[m,8], inmax[m,8])       │              │ │
│  └───────────────┬───────────────┘      │   │   │   └───────────────┬───────────────┘              │ │
│                  │                      │   │   │                   │                              │ │
│  ┌───────────────▼───────────────┐      │   │   │   ┌───────────────▼───────────────┐              │ │
│  │ exp(y[m,k])                   │──────┼───┼──►│   │ em=sub                        │              │ │
│  └───────────────┬───────────────┘      │   │   │   │ (inmax[m,8] - max[m,8])      │              │ │
│                  │                      │   │   │   └───────────────┬───────────────┘              │ │
│  ┌───────────────▼───────────────┐      │   │   │                   │                              │ │
│  │ reducesum                     │      │   │   │   ┌───────────────▼───────────────┐              │ │
│  │ ([m,k]->[m,1])                │      │   │   │   │ exp                           │───────────┐  │ │
│  └───────────────┬───────────────┘      │   │   │   │ (em[m,8])                     │           │  │ │
│                  │                      │   │   │   └───────────────┬───────────────┘           │  │ │
│  ┌───────────────▼───────────────┐      │   │   │                   │                           │  │ │
│  │ broadcast                     │      │   │   │   ┌───────────────▼───────────────┐           │  │ │
│  │ ([m,1]->[m,8])                │      │   │   │   │ y=sub                        │◄──────────┘  │ │
│  └───────────────┬───────────────┘      │   │   │   │ (x[m,k] - max[m,8])          │◄────────────┘ │
│                  │                      │   │   │   └───────────────┬───────────────┘                │
└──────────────────┼──────────────────────┘   │   │                   │                                │
                   │                          │   │   ┌───────────────▼───────────────┐                │
                   │                          │   │   │ exp(y[m,k])                   │──────────────┐ │
                   │                          │   │   └───────────────┬───────────────┘              │ │
                   │                          │   │                   │                              │ │
                   │                          │   │   ┌───────────────▼───────────────┐              │ │
                   │                          │   │   │ reducesum                     │              │ │
                   │                          │   │   │ ([m,k]->[m,1])                │              │ │
                   │                          │   │   └───────────────┬───────────────┘              │ │
                   │                          │   │                   │                              │ │
                   │                          │   │   ┌───────────────▼───────────────┐              │ │
                   │                          │   │   │ sum=broadcast                 │              │ │
                   │                          │   │   │ ([m,1]->[m,8])                │              │ │
                   │                          │   │   └───────────────┬───────────────┘              │ │
                   │                          │   │                   │                              │ │
                   │                          │   │   ┌───────────────▼──────────────────────────┐   │ │
                   │                          └───┼──►│ s=mul                                     │   │ │
                   │                              │   │ (expmax[m,8] * insum[m,8])               │   │ │
                   │                              │   └───────────────┬──────────────────────────┘   │ │
                   │                              │                   │                              │ │
                   │                              │   ┌───────────────▼───────────────┐◄────────────┘ │
                   │                              │   │ add                           │                │
                   │                              │   │ (s[m,8], sum[m,8])           │                │
                   │                              │   └───────────────┬───────────────┘                │
                   │                              │                   │                                │
                   │                              │                   ▼                                │
                   │                              │             ┌──────────┐                           │
                   │                              │             │ sum[m,8] │                           │
                   │                              │             └──────────┘                           │
                   │                              └─────────────────────────────────────────────────────┘
                   │
                   ▼
             ┌──────────┐
             │ max[m,8] │
             └──────────┘
                   ▲
                   │
             ┌──────────┐
             │ inmax[m,8]│──────────────────────────────────────────────►(输入到右侧 max)
             └──────────┘

             ┌──────────┐
             │ y[m,k]   │───────────────────────────────────────────────►(输入到右侧 y=sub)
             └──────────┘

             ┌──────────┐
             │ sum[m,8] │
             └──────────┘

             ┌──────────┐
             │ insum[m,8]│──────────────────────────────────────────────►(输入到右侧 s=mul)
             └──────────┘


右侧输出:
┌──────────┐
│ max[m,8] │
└──────────┘

┌────────────┐
│ expmax[m,8]│
└────────────┘

┌──────────┐
│ y[m,k]   │
└──────────┘


图示:
输入输出Tensor   ── 圆角矩形
vector计算      ── 矩形
条件判断        ── 菱形
数据流向        ── 箭头
```

计算过程根据isUpdate是否使能分为两个分支处理，均在Vector上进行。

-   当isUpdate为False时，分为如下几步：
    1.  reducemax步骤：对输入x的每一行数据求最大值得到\[m, 1\]，计算结果会保存到一个临时空间temp中；
    2.  broadcast步骤：对temp中的数据\[m, 1\]做一个按datablock为单位的填充，比如float类型下，把\[m, 1\]扩展成\[m, 8\]，同时输出max；
    3.  sub步骤：对输入x的所有数据按行减去max；
    4.  exp步骤：对sub之后的所有数据求exp，并且输出y；
    5.  reducesum步骤：对exp结果的每一行数据求和得到\[m, 1\]，计算结果会保存到临时空间temp中；
    6.  broadcast步骤：对temp\[m, 1\]做一个按datablock为单位的填充，比如float类型下，把\[m, 1\]扩展成\[m, 8\]，同时输出sum。

-   当isUpdate为True时，分为如下几步：
    1.  reducemax步骤：对输入x的每一行数据求最大值得到\[m, 1\]，计算结果会保存到一个临时空间temp中；
    2.  broadcast步骤：对temp中的数据\[m, 1\]做一个按datablock为单位的填充，比如float类型下，把\[m, 1\]扩展成\[m, 8\]，保存为max;
    3.  max步骤：对输入inmax和上一步计算的max做max操作，得到新的max并输出；
    4.  sub步骤：将输入inmax和新的max相减，然后做exp，计算得到expmax并输出；
    5.  sub步骤：将输入x和新的max按行相减；
    6.  exp步骤：对sub之后的所有数据求exp，并且输出y；
    7.  reducesum步骤：对exp结果的每一行数据求和得到\[m, 1\]，计算结果会保存到临时空间temp中；
    8.  broadcast步骤：对temp数据\[m, 1\]做一个按datablock为单位的填充，比如float类型下，把\[m, 1\]扩展成\[m, 8\]，保存到sum中；
    9.  mul步骤：将输入insum和expmax结果相乘；
    10. add步骤：将相乘结果和sum相加，保存到sum并输出。

## 函数原型<a name="section620mcpsimp"></a>

-   接口框架申请临时空间
    -   LocalTensor的数据类型相同，不输出ReduceMax

        ```
        template <typename T, bool isUpdate = false, bool isReuseSource = false, bool isBasicBlock = false, bool isDataFormatNZ = false, const SoftmaxConfig& config = SOFTMAX_DEFAULT_CFG>
        __aicore__ inline void SoftmaxFlashV2(const LocalTensor<T>& dstTensor, const LocalTensor<T>& expSumTensor, const LocalTensor<T>& maxTensor, const LocalTensor<T>& srcTensor, const LocalTensor<T>& expMaxTensor, const LocalTensor<T>& inExpSumTensor, const LocalTensor<T>& inMaxTensor, const SoftMaxTiling& tiling, const SoftMaxShapeInfo& softmaxShapeInfo = {})
        ```

    -   LocalTensor的数据类型相同，且输出ReduceMax

        ```
        template <typename T, bool isUpdate = false, bool isReuseSource = false, bool isBasicBlock = false, bool isDataFormatNZ = false, const SoftmaxConfig& config = SOFTMAX_DEFAULT_CFG>
        __aicore__ inline void SoftmaxFlashV2(const LocalTensor<T>& dstTensor, const LocalTensor<T>& outReduceMax, const LocalTensor<T>& outExpSum, const LocalTensor<T>& outMax, const LocalTensor<T>& srcTensor, const LocalTensor<T>& outExpMax, const LocalTensor<T>& inExpSum, const LocalTensor<T>& inMax, const SoftMaxTiling& tiling, const SoftMaxShapeInfo& softmaxShapeInfo = {})
        ```

    -   LocalTensor的数据类型不同，不输出ReduceMax

        ```
        template <typename T, bool isUpdate = false, bool isReuseSource = false, bool isBasicBlock = false, bool isDataFormatNZ = false, const SoftmaxConfig& config = SOFTMAX_DEFAULT_CFG>
        __aicore__ inline void SoftmaxFlashV2(const LocalTensor<half>& dstTensor, const LocalTensor<float>& expSumTensor, const LocalTensor<float>& maxTensor, const LocalTensor<half>& srcTensor, const LocalTensor<half>& expMaxTensor, const LocalTensor<float>& inExpSumTensor, const LocalTensor<float>& inMaxTensor, const SoftMaxTiling& tiling, const SoftMaxShapeInfo& softmaxShapeInfo = {})
        ```

-   通过sharedTmpBuffer入参传入临时空间
    -   LocalTensor的数据类型相同，不输出ReduceMax

        ```
        template <typename T, bool isUpdate = false, bool isReuseSource = false, bool isBasicBlock = false, bool isDataFormatNZ = false, const SoftmaxConfig& config = SOFTMAX_DEFAULT_CFG>
        __aicore__ inline void SoftmaxFlashV2(const LocalTensor<T>& dstTensor, const LocalTensor<T>& outExpSum, const LocalTensor<T>& outMax, const LocalTensor<T>& srcTensor, const LocalTensor<T>& outExpMax, const LocalTensor<T>& inExpSum, const LocalTensor<T>& inMax, const LocalTensor<uint8_t>& sharedTmpBuffer, const SoftMaxTiling& tiling, const SoftMaxShapeInfo& softmaxShapeInfo = {})
        ```

    -   LocalTensor的数据类型相同，且输出ReduceMax

        ```
        template <typename T, bool isUpdate = false, bool isReuseSource = false, bool isBasicBlock = false, bool isDataFormatNZ = false, const SoftmaxConfig& config = SOFTMAX_DEFAULT_CFG>
        __aicore__ inline void SoftmaxFlashV2(const LocalTensor<T>& dstTensor, const LocalTensor<T>& outReduceMax, const LocalTensor<T>& expSumTensor, const LocalTensor<T>& maxTensor, const LocalTensor<T>& srcTensor, const LocalTensor<T>& expMaxTensor, const LocalTensor<T>& inExpSumTensor, const LocalTensor<T>& inMaxTensor, const LocalTensor<uint8_t>& sharedTmpBuffer, const SoftMaxTiling& tiling, const SoftMaxShapeInfo& softmaxShapeInfo = {})
        ```

    -   LocalTensor的数据类型不同，不输出ReduceMax

        ```
        template <typename T, bool isUpdate = false, bool isReuseSource = false, bool isBasicBlock = false, bool isDataFormatNZ = false, const SoftmaxConfig& config = SOFTMAX_DEFAULT_CFG>
        __aicore__ inline void SoftmaxFlashV2(const LocalTensor<half>& dstTensor, const LocalTensor<float>& expSumTensor, const LocalTensor<float>& maxTensor, const LocalTensor<half>& srcTensor, const LocalTensor<half>& expMaxTensor, const LocalTensor<float>& inExpSumTensor, const LocalTensor<float>& inMaxTensor, const LocalTensor<uint8_t>& sharedTmpBuffer, const SoftMaxTiling& tiling, const SoftMaxShapeInfo& softmaxShapeInfo = {})
        ```

由于该接口的内部实现中涉及复杂的计算，需要额外的临时空间来存储计算过程中的中间变量。临时空间支持**接口框架申请**和开发者**通过sharedTmpBuffer入参传入**两种方式。

-   接口框架申请临时空间，开发者无需申请，但是需要预留临时空间的大小。

-   通过sharedTmpBuffer入参传入，使用该tensor作为临时空间进行处理，接口框架不再申请。该方式开发者可以自行管理sharedTmpBuffer内存空间，并在接口调用完成后，复用该部分内存，内存不会反复申请释放，灵活性较高，内存利用率也较高。

接口框架申请的方式，开发者需要预留临时空间；通过sharedTmpBuffer传入的情况，开发者需要为tensor申请空间。临时空间大小BufferSize的获取方式如下：通过[SoftmaxFlashV2 Tiling接口](SoftmaxFlashV2-Tiling接口.md)中提供的GetSoftMaxFlashV2MinTmpSize/GetSoftMaxFlashV2MaxTmpSize接口获取所需最小和最大临时空间大小，最小空间可以保证功能正确，最大空间用于提升性能。

另外提供了一个kernel侧tiling计算的接口，当kernel侧的输入shape与通过host侧TilingData传入的shape不一致时，可使用该接口在kernel侧重新计算tiling。该接口的参数含义请参考[SoftmaxFlashV2 Tiling接口](SoftmaxFlashV2-Tiling接口.md)。

-   **kernel侧tiling计算接口**

    ```
    __aicore__ inline constexpr SoftMaxTiling SoftMaxFlashV2TilingFunc(const SoftMaxShapeInfo& shapeInfo, const uint32_t dataTypeSize1, const uint32_t dataTypeSize2, const uint32_t localWorkSpaceSize, const bool isUpdate = false, const bool isBasicBlock = false, const bool isDataFormatNZ = false, const bool isFlashOutputBrc = false)
    ```

## 参数说明<a name="section622mcpsimp"></a>

**表 1**  模板参数说明

<a name="table575571914269"></a>
<table><thead align="left"><tr id="row18755131942614"><th class="cellrowborder" valign="top" width="19.39%" id="mcps1.2.3.1.1"><p id="p675519193268"><a name="p675519193268"></a><a name="p675519193268"></a>参数名</p>
</th>
<th class="cellrowborder" valign="top" width="80.61%" id="mcps1.2.3.1.2"><p id="p375511918267"><a name="p375511918267"></a><a name="p375511918267"></a>描述</p>
</th>
</tr>
</thead>
<tbody><tr id="row14755141911264"><td class="cellrowborder" valign="top" width="19.39%" headers="mcps1.2.3.1.1 "><p id="p47551198266"><a name="p47551198266"></a><a name="p47551198266"></a>T</p>
</td>
<td class="cellrowborder" valign="top" width="80.61%" headers="mcps1.2.3.1.2 "><p id="p125969172719"><a name="p125969172719"></a><a name="p125969172719"></a>操作数的数据类型。</p>
<p id="p660618480410"><a name="p660618480410"></a><a name="p660618480410"></a><span id="ph5770145512415"><a name="ph5770145512415"></a><a name="ph5770145512415"></a>Ascend 950PR/Ascend 950DT</span>，支持的数据类型为：half、float。</p>
</td>
</tr>
<tr id="row1783875017236"><td class="cellrowborder" valign="top" width="19.39%" headers="mcps1.2.3.1.1 "><p id="p1838850172311"><a name="p1838850172311"></a><a name="p1838850172311"></a>isUpdate</p>
</td>
<td class="cellrowborder" valign="top" width="80.61%" headers="mcps1.2.3.1.2 "><p id="p1983816503234"><a name="p1983816503234"></a><a name="p1983816503234"></a>是否使能update部分中的计算。</p>
</td>
</tr>
<tr id="row9756719122620"><td class="cellrowborder" valign="top" width="19.39%" headers="mcps1.2.3.1.1 "><p id="p1682112447268"><a name="p1682112447268"></a><a name="p1682112447268"></a>isReuseSource</p>
</td>
<td class="cellrowborder" valign="top" width="80.61%" headers="mcps1.2.3.1.2 "><p id="p1632918814615"><a name="p1632918814615"></a><a name="p1632918814615"></a>该参数预留，传入默认值false即可。</p>
</td>
</tr>
<tr id="row9184124919159"><td class="cellrowborder" valign="top" width="19.39%" headers="mcps1.2.3.1.1 "><p id="p11692440141619"><a name="p11692440141619"></a><a name="p11692440141619"></a>isBasicBlock</p>
</td>
<td class="cellrowborder" valign="top" width="80.61%" headers="mcps1.2.3.1.2 "><p id="p74171158258"><a name="p74171158258"></a><a name="p74171158258"></a>srcTensor和dstTensor的shape信息和Tiling切分策略满足基本块要求的情况下，可以使能该参数用于提升性能，默认不使能。是否满足基本块的要求，可以采用如下两种方式之一判断：</p>
<a name="ul353295811167"></a><a name="ul353295811167"></a><ul id="ul353295811167"><li>srcTensor和dstTensor的shape信息[m,n]需要满足如下条件：<a name="ul09181366549"></a><a name="ul09181366549"></a><ul id="ul09181366549"><li>尾轴长度n小于2048并且大于等于256/sizeof(T)（即half场景下n最小为128，float场景下n最小为64），同时n是64的倍数；</li><li>非尾轴长度的乘积m为8的倍数。</li></ul>
</li></ul>
<a name="ul1941711152254"></a><a name="ul1941711152254"></a><ul id="ul1941711152254"><li>在Tiling实现中，通过调用<a href="IsBasicBlockInSoftMax.md">IsBasicBlockInSoftMax</a>判断Tiling切分策略是否满足基本块的切分要求。</li></ul>
</td>
</tr>
<tr id="row11813131962412"><td class="cellrowborder" valign="top" width="19.39%" headers="mcps1.2.3.1.1 "><p id="p16813119192415"><a name="p16813119192415"></a><a name="p16813119192415"></a>isDataFormatNZ</p>
</td>
<td class="cellrowborder" valign="top" width="80.61%" headers="mcps1.2.3.1.2 "><p id="p2038962913258"><a name="p2038962913258"></a><a name="p2038962913258"></a>当前输入输出的数据格式是否为NZ格式，默认数据格式为ND，即默认取值为false。</p>
</td>
</tr>
<tr id="row996414112248"><td class="cellrowborder" valign="top" width="19.39%" headers="mcps1.2.3.1.1 "><p id="p69641412246"><a name="p69641412246"></a><a name="p69641412246"></a>config</p>
</td>
<td class="cellrowborder" valign="top" width="80.61%" headers="mcps1.2.3.1.2 "><p id="zh-cn_topic_0000001621597661_p1744211392255"><a name="zh-cn_topic_0000001621597661_p1744211392255"></a><a name="zh-cn_topic_0000001621597661_p1744211392255"></a>结构体模板参数，此参数可选配，SoftmaxConfig类型，具体定义如下。</p>
<a name="zh-cn_topic_0000001621597661_screen16476116195910"></a><a name="zh-cn_topic_0000001621597661_screen16476116195910"></a><pre class="screen" codetype="Cpp" id="zh-cn_topic_0000001621597661_screen16476116195910">struct SoftmaxConfig{
    bool isCheckTiling = true; // 是否需要检查shape和tiling的一致性；若不一致，API内会根据shape重新计算所需tiling。默认取值true：API内部会检查一致性
    uint32_t oriSrcM = 0; // 原始非尾轴长度的乘积。设置该参数后，将shape常量化，编译过程中使用常量化的shape
    uint32_t oriSrcK = 0; // 原始尾轴长度。设置该参数后，将shape常量化，编译过程中使用常量化的shape
    SoftmaxMode mode = SoftmaxMode::SOFTMAX_NORMAL; // 输出shape的处理模式
};</pre>
<p id="p105494279362"><a name="p105494279362"></a><a name="p105494279362"></a>其中，参数mode表示输出shape的处理模式，当输入输出的数据格式为NZ格式时，不支持配置mode参数。SoftmaxMode类型，取值如下：</p>
<a name="ul1342184553616"></a><a name="ul1342184553616"></a><ul id="ul1342184553616"><li>SOFTMAX_NORMAL ：默认值，常规模式，对输出数据做Broadcast，使得输出shape由(m, 1)拓展成(m, 8)（输出为float数据类型）或者(m, 16)（输出为half数据类型）。</li><li>SOFTMAX_OUTPUT_WITHOUT_BRC ：非拓展模式，不对输出数据做Broadcast，输出shape均为(m, 1)，相应的输入参数（例如inExpSumTensor、inMaxTensor），shape也均为(m, 1) 。</li></ul>
<p id="zh-cn_topic_0000001621597661_p76421594583"><a name="zh-cn_topic_0000001621597661_p76421594583"></a><a name="zh-cn_topic_0000001621597661_p76421594583"></a>配置示例如下。</p>
<a name="zh-cn_topic_0000001621597661_screen19241326175913"></a><a name="zh-cn_topic_0000001621597661_screen19241326175913"></a><pre class="screen" codetype="Cpp" id="zh-cn_topic_0000001621597661_screen19241326175913">constexpr SoftmaxConfig SOFTMAX_DEFAULT_CFG = {true, 0, 0, SoftmaxMode::SOFTMAX_NORMAL};</pre>
<p id="zh-cn_topic_0000001621597661_p19442739102517"><a name="zh-cn_topic_0000001621597661_p19442739102517"></a><a name="zh-cn_topic_0000001621597661_p19442739102517"></a>此参数一般用于配合kernel侧tiling计算的接口使用。</p>
<p id="p9341337121216"><a name="p9341337121216"></a><a name="p9341337121216"></a>注意：设置了oriSrcM与oriSrcK后，模板参数isBasicBlock不生效，计算数据是否为基本块由API内部判断并处理。</p>
<p id="p9698920448"><a name="p9698920448"></a><a name="p9698920448"></a>针对<span id="ph1527792911411"><a name="ph1527792911411"></a><a name="ph1527792911411"></a>Ascend 950PR/Ascend 950DT</span>，支持该参数。</p>
</td>
</tr>
</tbody>
</table>

**表 2**  接口参数说明

<a name="zh-cn_topic_0235751031_table33761356"></a>
<table><thead align="left"><tr id="zh-cn_topic_0235751031_row27598891"><th class="cellrowborder" valign="top" width="17.77%" id="mcps1.2.4.1.1"><p id="zh-cn_topic_0235751031_p20917673"><a name="zh-cn_topic_0235751031_p20917673"></a><a name="zh-cn_topic_0235751031_p20917673"></a>参数名</p>
</th>
<th class="cellrowborder" valign="top" width="8.07%" id="mcps1.2.4.1.2"><p id="zh-cn_topic_0235751031_p16609919"><a name="zh-cn_topic_0235751031_p16609919"></a><a name="zh-cn_topic_0235751031_p16609919"></a>输入/输出</p>
</th>
<th class="cellrowborder" valign="top" width="74.16%" id="mcps1.2.4.1.3"><p id="zh-cn_topic_0235751031_p59995477"><a name="zh-cn_topic_0235751031_p59995477"></a><a name="zh-cn_topic_0235751031_p59995477"></a>描述</p>
</th>
</tr>
</thead>
<tbody><tr id="row42461942101815"><td class="cellrowborder" valign="top" width="17.77%" headers="mcps1.2.4.1.1 "><p id="p1480238142815"><a name="p1480238142815"></a><a name="p1480238142815"></a>dstTensor</p>
</td>
<td class="cellrowborder" valign="top" width="8.07%" headers="mcps1.2.4.1.2 "><p id="p948033819282"><a name="p948033819282"></a><a name="p948033819282"></a>输出</p>
</td>
<td class="cellrowborder" valign="top" width="74.16%" headers="mcps1.2.4.1.3 "><p id="p242825624218"><a name="p242825624218"></a><a name="p242825624218"></a>目的操作数。</p>
<p id="p16703131355116"><a name="p16703131355116"></a><a name="p16703131355116"></a><span id="zh-cn_topic_0000002523303824_ph173308471594"><a name="zh-cn_topic_0000002523303824_ph173308471594"></a><a name="zh-cn_topic_0000002523303824_ph173308471594"></a><span id="zh-cn_topic_0000002523303824_ph9902231466"><a name="zh-cn_topic_0000002523303824_ph9902231466"></a><a name="zh-cn_topic_0000002523303824_ph9902231466"></a><span id="zh-cn_topic_0000002523303824_ph1782115034816"><a name="zh-cn_topic_0000002523303824_ph1782115034816"></a><a name="zh-cn_topic_0000002523303824_ph1782115034816"></a>类型为<a href="LocalTensor.md">LocalTensor</a>，支持的TPosition为VECIN/VECCALC/VECOUT。</span></span></span></p>
<p id="p1327102402720"><a name="p1327102402720"></a><a name="p1327102402720"></a>dstTensor的shape和源操作数srcTensor一致。</p>
</td>
</tr>
<tr id="row4482101172217"><td class="cellrowborder" valign="top" width="17.77%" headers="mcps1.2.4.1.1 "><p id="p71241319172212"><a name="p71241319172212"></a><a name="p71241319172212"></a>outReduceMax</p>
</td>
<td class="cellrowborder" valign="top" width="8.07%" headers="mcps1.2.4.1.2 "><p id="p14124111932213"><a name="p14124111932213"></a><a name="p14124111932213"></a>输出</p>
</td>
<td class="cellrowborder" valign="top" width="74.16%" headers="mcps1.2.4.1.3 "><p id="p17124131918229"><a name="p17124131918229"></a><a name="p17124131918229"></a>目的操作数。用于保存softmax计算过程中reducemax第一次计算的结果。</p>
<p id="p712471917229"><a name="p712471917229"></a><a name="p712471917229"></a><span id="zh-cn_topic_0000002523303824_ph173308471594_1"><a name="zh-cn_topic_0000002523303824_ph173308471594_1"></a><a name="zh-cn_topic_0000002523303824_ph173308471594_1"></a><span id="zh-cn_topic_0000002523303824_ph9902231466_1"><a name="zh-cn_topic_0000002523303824_ph9902231466_1"></a><a name="zh-cn_topic_0000002523303824_ph9902231466_1"></a><span id="zh-cn_topic_0000002523303824_ph1782115034816_1"><a name="zh-cn_topic_0000002523303824_ph1782115034816_1"></a><a name="zh-cn_topic_0000002523303824_ph1782115034816_1"></a>类型为<a href="LocalTensor.md">LocalTensor</a>，支持的TPosition为VECIN/VECCALC/VECOUT。</span></span></span></p>
<p id="p151241719132219"><a name="p151241719132219"></a><a name="p151241719132219"></a>outReduceMax的shape与目的操作数maxTensor一致。</p>
<p id="p82114142395"><a name="p82114142395"></a><a name="p82114142395"></a>对于输出该结果的接口：</p>
<a name="ul186771932153414"></a><a name="ul186771932153414"></a><ul id="ul186771932153414"><li>模板参数isUpdate为false时，不输出该结果。</li><li>仅支持输入输出的数据格式为ND，模板参数isDataFormatNZ为预留参数，传入默认值false即可。</li><li>模板参数config.isCheckTiling为预留参数，传入默认值false即可。</li><li>模板参数config.mode仅支持配置为非拓展模式SoftmaxMode::SOFTMAX_OUTPUT_WITHOUT_BRC。</li></ul>
</td>
</tr>
<tr id="row2137145181815"><td class="cellrowborder" valign="top" width="17.77%" headers="mcps1.2.4.1.1 "><p id="p20480153802811"><a name="p20480153802811"></a><a name="p20480153802811"></a>expSumTensor、outExpSum</p>
</td>
<td class="cellrowborder" valign="top" width="8.07%" headers="mcps1.2.4.1.2 "><p id="p154801238192812"><a name="p154801238192812"></a><a name="p154801238192812"></a>输出</p>
</td>
<td class="cellrowborder" valign="top" width="74.16%" headers="mcps1.2.4.1.3 "><p id="p1659223718226"><a name="p1659223718226"></a><a name="p1659223718226"></a>目的操作数。用于保存softmax计算过程中reducesum的结果。</p>
<p id="p22921529162311"><a name="p22921529162311"></a><a name="p22921529162311"></a><span id="zh-cn_topic_0000002523303824_ph173308471594_2"><a name="zh-cn_topic_0000002523303824_ph173308471594_2"></a><a name="zh-cn_topic_0000002523303824_ph173308471594_2"></a><span id="zh-cn_topic_0000002523303824_ph9902231466_2"><a name="zh-cn_topic_0000002523303824_ph9902231466_2"></a><a name="zh-cn_topic_0000002523303824_ph9902231466_2"></a><span id="zh-cn_topic_0000002523303824_ph1782115034816_2"><a name="zh-cn_topic_0000002523303824_ph1782115034816_2"></a><a name="zh-cn_topic_0000002523303824_ph1782115034816_2"></a>类型为<a href="LocalTensor.md">LocalTensor</a>，支持的TPosition为VECIN/VECCALC/VECOUT。</span></span></span></p>
<a name="ul12308151213513"></a><a name="ul12308151213513"></a><ul id="ul12308151213513"><li>除模板参数config配置为非拓展模式（SoftmaxMode::SOFTMAX_OUTPUT_WITHOUT_BRC）的场景外，expSumTensor的last轴长度固定为32Byte，即一个<span id="ph16212125511335"><a name="ph16212125511335"></a><a name="ph16212125511335"></a>datablock</span>长度。该<span id="ph1235145803313"><a name="ph1235145803313"></a><a name="ph1235145803313"></a>datablock</span>中的所有数据为同一个值，比如half数据类型下，该<span id="ph16133160163412"><a name="ph16133160163412"></a><a name="ph16133160163412"></a>datablock</span>中的16个数均为相同的reducesum的值。</li><li>非last轴的长度与dstTensor保持一致。</li></ul>
</td>
</tr>
<tr id="row19615183817191"><td class="cellrowborder" valign="top" width="17.77%" headers="mcps1.2.4.1.1 "><p id="p1648015388281"><a name="p1648015388281"></a><a name="p1648015388281"></a>maxTensor、outMax</p>
</td>
<td class="cellrowborder" valign="top" width="8.07%" headers="mcps1.2.4.1.2 "><p id="p14480738162817"><a name="p14480738162817"></a><a name="p14480738162817"></a>输出</p>
</td>
<td class="cellrowborder" valign="top" width="74.16%" headers="mcps1.2.4.1.3 "><p id="p033515495269"><a name="p033515495269"></a><a name="p033515495269"></a>目的操作数。用于保存softmax计算过程中reducemax的结果。</p>
<p id="p1097223511233"><a name="p1097223511233"></a><a name="p1097223511233"></a><span id="zh-cn_topic_0000002523303824_ph173308471594_3"><a name="zh-cn_topic_0000002523303824_ph173308471594_3"></a><a name="zh-cn_topic_0000002523303824_ph173308471594_3"></a><span id="zh-cn_topic_0000002523303824_ph9902231466_3"><a name="zh-cn_topic_0000002523303824_ph9902231466_3"></a><a name="zh-cn_topic_0000002523303824_ph9902231466_3"></a><span id="zh-cn_topic_0000002523303824_ph1782115034816_3"><a name="zh-cn_topic_0000002523303824_ph1782115034816_3"></a><a name="zh-cn_topic_0000002523303824_ph1782115034816_3"></a>类型为<a href="LocalTensor.md">LocalTensor</a>，支持的TPosition为VECIN/VECCALC/VECOUT。</span></span></span></p>
<a name="ul9474401463"></a><a name="ul9474401463"></a><ul id="ul9474401463"><li>除模板参数config配置为非拓展模式（SoftmaxMode::SOFTMAX_OUTPUT_WITHOUT_BRC）的场景外，maxTensor的last轴长度固定为32Byte，即一个<span id="ph1620974113410"><a name="ph1620974113410"></a><a name="ph1620974113410"></a>datablock</span>长度。该<span id="ph473675173413"><a name="ph473675173413"></a><a name="ph473675173413"></a>datablock</span>中的所有数据为同一个值。比如half数据类型下，该<span id="ph17613118113419"><a name="ph17613118113419"></a><a name="ph17613118113419"></a>datablock</span>中的16个数均为相同的reducemax的值。</li><li>非last轴的长度与dstTensor保持一致。</li></ul>
</td>
</tr>
<tr id="row250163914199"><td class="cellrowborder" valign="top" width="17.77%" headers="mcps1.2.4.1.1 "><p id="p204801738152820"><a name="p204801738152820"></a><a name="p204801738152820"></a>srcTensor</p>
</td>
<td class="cellrowborder" valign="top" width="8.07%" headers="mcps1.2.4.1.2 "><p id="p1248073811284"><a name="p1248073811284"></a><a name="p1248073811284"></a>输入</p>
</td>
<td class="cellrowborder" valign="top" width="74.16%" headers="mcps1.2.4.1.3 "><p id="p1159016208298"><a name="p1159016208298"></a><a name="p1159016208298"></a>源操作数。</p>
<p id="p495119425238"><a name="p495119425238"></a><a name="p495119425238"></a><span id="zh-cn_topic_0000002523303824_ph173308471594_4"><a name="zh-cn_topic_0000002523303824_ph173308471594_4"></a><a name="zh-cn_topic_0000002523303824_ph173308471594_4"></a><span id="zh-cn_topic_0000002523303824_ph9902231466_4"><a name="zh-cn_topic_0000002523303824_ph9902231466_4"></a><a name="zh-cn_topic_0000002523303824_ph9902231466_4"></a><span id="zh-cn_topic_0000002523303824_ph1782115034816_4"><a name="zh-cn_topic_0000002523303824_ph1782115034816_4"></a><a name="zh-cn_topic_0000002523303824_ph1782115034816_4"></a>类型为<a href="LocalTensor.md">LocalTensor</a>，支持的TPosition为VECIN/VECCALC/VECOUT。</span></span></span></p>
<p id="p15480438132811"><a name="p15480438132811"></a><a name="p15480438132811"></a>last轴长度需要32Byte对齐。</p>
</td>
</tr>
<tr id="row041413392198"><td class="cellrowborder" valign="top" width="17.77%" headers="mcps1.2.4.1.1 "><p id="p3480143892814"><a name="p3480143892814"></a><a name="p3480143892814"></a>expMaxTensor、outExpMax</p>
</td>
<td class="cellrowborder" valign="top" width="8.07%" headers="mcps1.2.4.1.2 "><p id="p144802038122815"><a name="p144802038122815"></a><a name="p144802038122815"></a>输出</p>
</td>
<td class="cellrowborder" valign="top" width="74.16%" headers="mcps1.2.4.1.3 "><p id="p920126333"><a name="p920126333"></a><a name="p920126333"></a>目的操作数。用于保存inmax与reducemax差值的e的指数幂的结果。</p>
<p id="p19731148142315"><a name="p19731148142315"></a><a name="p19731148142315"></a><span id="zh-cn_topic_0000002523303824_ph173308471594_5"><a name="zh-cn_topic_0000002523303824_ph173308471594_5"></a><a name="zh-cn_topic_0000002523303824_ph173308471594_5"></a><span id="zh-cn_topic_0000002523303824_ph9902231466_5"><a name="zh-cn_topic_0000002523303824_ph9902231466_5"></a><a name="zh-cn_topic_0000002523303824_ph9902231466_5"></a><span id="zh-cn_topic_0000002523303824_ph1782115034816_5"><a name="zh-cn_topic_0000002523303824_ph1782115034816_5"></a><a name="zh-cn_topic_0000002523303824_ph1782115034816_5"></a>类型为<a href="LocalTensor.md">LocalTensor</a>，支持的TPosition为VECIN/VECCALC/VECOUT。</span></span></span></p>
<a name="ul6386131510362"></a><a name="ul6386131510362"></a><ul id="ul6386131510362"><li>除模板参数config配置为非拓展模式（SoftmaxMode::SOFTMAX_OUTPUT_WITHOUT_BRC）的场景外，expMaxTensor的last轴长度固定为32Byte，即一个<span id="ph862151363410"><a name="ph862151363410"></a><a name="ph862151363410"></a>datablock</span>长度。该<span id="ph662931423419"><a name="ph662931423419"></a><a name="ph662931423419"></a>datablock</span>中的所有数据为同一个值，比如half数据类型下，该<span id="ph161371317203414"><a name="ph161371317203414"></a><a name="ph161371317203414"></a>datablock</span>中的16个数均为相同的值。</li><li>非last轴的长度需要与dstTensor保持一致。</li></ul>
</td>
</tr>
<tr id="row17371444131520"><td class="cellrowborder" valign="top" width="17.77%" headers="mcps1.2.4.1.1 "><p id="p548011382287"><a name="p548011382287"></a><a name="p548011382287"></a>inExpSumTensor、inExpSum</p>
</td>
<td class="cellrowborder" valign="top" width="8.07%" headers="mcps1.2.4.1.2 "><p id="p134801438182813"><a name="p134801438182813"></a><a name="p134801438182813"></a>输入</p>
</td>
<td class="cellrowborder" valign="top" width="74.16%" headers="mcps1.2.4.1.3 "><p id="p14729613163018"><a name="p14729613163018"></a><a name="p14729613163018"></a>源操作数。softmax计算所需要的sum值。</p>
<p id="p167015292315"><a name="p167015292315"></a><a name="p167015292315"></a><span id="zh-cn_topic_0000002523303824_ph173308471594_6"><a name="zh-cn_topic_0000002523303824_ph173308471594_6"></a><a name="zh-cn_topic_0000002523303824_ph173308471594_6"></a><span id="zh-cn_topic_0000002523303824_ph9902231466_6"><a name="zh-cn_topic_0000002523303824_ph9902231466_6"></a><a name="zh-cn_topic_0000002523303824_ph9902231466_6"></a><span id="zh-cn_topic_0000002523303824_ph1782115034816_6"><a name="zh-cn_topic_0000002523303824_ph1782115034816_6"></a><a name="zh-cn_topic_0000002523303824_ph1782115034816_6"></a>类型为<a href="LocalTensor.md">LocalTensor</a>，支持的TPosition为VECIN/VECCALC/VECOUT。</span></span></span></p>
<a name="ul7387193685112"></a><a name="ul7387193685112"></a><ul id="ul7387193685112"><li>除模板参数config配置为非拓展模式（SoftmaxMode::SOFTMAX_OUTPUT_WITHOUT_BRC）的场景外，inExpSumTensor的last轴长度固定为32Byte，即一个<span id="ph11711321183412"><a name="ph11711321183412"></a><a name="ph11711321183412"></a>datablock</span>长度。该<span id="ph20484112218343"><a name="ph20484112218343"></a><a name="ph20484112218343"></a>datablock</span>中的所有数据为同一个值，比如half数据类型下，该<span id="ph17413225113410"><a name="ph17413225113410"></a><a name="ph17413225113410"></a>datablock</span>中的16个数均为相同的值。</li><li>非last轴的长度需要与dstTensor保持一致。</li></ul>
</td>
</tr>
<tr id="row1811919157286"><td class="cellrowborder" valign="top" width="17.77%" headers="mcps1.2.4.1.1 "><p id="p184801438162811"><a name="p184801438162811"></a><a name="p184801438162811"></a>inMaxTensor、inMax</p>
</td>
<td class="cellrowborder" valign="top" width="8.07%" headers="mcps1.2.4.1.2 "><p id="p1748023810286"><a name="p1748023810286"></a><a name="p1748023810286"></a>输入</p>
</td>
<td class="cellrowborder" valign="top" width="74.16%" headers="mcps1.2.4.1.3 "><p id="p5531616153011"><a name="p5531616153011"></a><a name="p5531616153011"></a>源操作数。softmax计算所需要的max值。</p>
<p id="p3150558152311"><a name="p3150558152311"></a><a name="p3150558152311"></a><span id="zh-cn_topic_0000002523303824_ph173308471594_7"><a name="zh-cn_topic_0000002523303824_ph173308471594_7"></a><a name="zh-cn_topic_0000002523303824_ph173308471594_7"></a><span id="zh-cn_topic_0000002523303824_ph9902231466_7"><a name="zh-cn_topic_0000002523303824_ph9902231466_7"></a><a name="zh-cn_topic_0000002523303824_ph9902231466_7"></a><span id="zh-cn_topic_0000002523303824_ph1782115034816_7"><a name="zh-cn_topic_0000002523303824_ph1782115034816_7"></a><a name="zh-cn_topic_0000002523303824_ph1782115034816_7"></a>类型为<a href="LocalTensor.md">LocalTensor</a>，支持的TPosition为VECIN/VECCALC/VECOUT。</span></span></span></p>
<a name="ul2353125216375"></a><a name="ul2353125216375"></a><ul id="ul2353125216375"><li>除模板参数config配置为非拓展模式（SoftmaxMode::SOFTMAX_OUTPUT_WITHOUT_BRC）的场景外，inMaxTensor的last轴长度固定为32Byte，即一个<span id="ph0342129153413"><a name="ph0342129153413"></a><a name="ph0342129153413"></a>datablock</span>长度。该<span id="ph172770329347"><a name="ph172770329347"></a><a name="ph172770329347"></a>datablock</span>中的所有数据为同一个值，比如half数据类型下，该<span id="ph4963234153413"><a name="ph4963234153413"></a><a name="ph4963234153413"></a>datablock</span>里的16个数均为相同的值。</li><li>非last轴的长度需要与dstTensor保持一致。</li></ul>
</td>
</tr>
<tr id="row12781171124710"><td class="cellrowborder" valign="top" width="17.77%" headers="mcps1.2.4.1.1 "><p id="p1358462510237"><a name="p1358462510237"></a><a name="p1358462510237"></a>sharedTmpBuffer</p>
</td>
<td class="cellrowborder" valign="top" width="8.07%" headers="mcps1.2.4.1.2 "><p id="p15584152512313"><a name="p15584152512313"></a><a name="p15584152512313"></a>输入</p>
</td>
<td class="cellrowborder" valign="top" width="74.16%" headers="mcps1.2.4.1.3 "><p id="p155841325182310"><a name="p155841325182310"></a><a name="p155841325182310"></a>临时空间。</p>
<p id="p5983205412394"><a name="p5983205412394"></a><a name="p5983205412394"></a><span id="zh-cn_topic_0000002523303824_ph173308471594_8"><a name="zh-cn_topic_0000002523303824_ph173308471594_8"></a><a name="zh-cn_topic_0000002523303824_ph173308471594_8"></a><span id="zh-cn_topic_0000002523303824_ph9902231466_8"><a name="zh-cn_topic_0000002523303824_ph9902231466_8"></a><a name="zh-cn_topic_0000002523303824_ph9902231466_8"></a><span id="zh-cn_topic_0000002523303824_ph1782115034816_8"><a name="zh-cn_topic_0000002523303824_ph1782115034816_8"></a><a name="zh-cn_topic_0000002523303824_ph1782115034816_8"></a>类型为<a href="LocalTensor.md">LocalTensor</a>，支持的TPosition为VECIN/VECCALC/VECOUT。</span></span></span></p>
<p id="p8928165616507"><a name="p8928165616507"></a><a name="p8928165616507"></a>该操作数的数据类型固定uint8_t。</p>
<p id="p104071111204211"><a name="p104071111204211"></a><a name="p104071111204211"></a>接口内部复杂计算时用于存储中间变量，由开发者提供。</p>
<p id="p5881016172817"><a name="p5881016172817"></a><a name="p5881016172817"></a>临时空间大小BufferSize的获取方式请参考<a href="SoftmaxFlashV2-Tiling接口.md">SoftmaxFlashV2 Tiling接口</a>。</p>
</td>
</tr>
<tr id="row1765118177284"><td class="cellrowborder" valign="top" width="17.77%" headers="mcps1.2.4.1.1 "><p id="p54804387283"><a name="p54804387283"></a><a name="p54804387283"></a>tiling</p>
</td>
<td class="cellrowborder" valign="top" width="8.07%" headers="mcps1.2.4.1.2 "><p id="p1048083882812"><a name="p1048083882812"></a><a name="p1048083882812"></a>输入</p>
</td>
<td class="cellrowborder" valign="top" width="74.16%" headers="mcps1.2.4.1.3 "><p id="p16481838112813"><a name="p16481838112813"></a><a name="p16481838112813"></a>softmaxflashv2接口计算所需tiling信息，Tiling信息的获取请参考<a href="SoftmaxFlashV2-Tiling接口.md">SoftmaxFlashV2 Tiling接口</a>。</p>
</td>
</tr>
<tr id="row6303151181211"><td class="cellrowborder" valign="top" width="17.77%" headers="mcps1.2.4.1.1 "><p id="p0734515621"><a name="p0734515621"></a><a name="p0734515621"></a>softmaxShapeInfo</p>
</td>
<td class="cellrowborder" valign="top" width="8.07%" headers="mcps1.2.4.1.2 "><p id="p1973441513219"><a name="p1973441513219"></a><a name="p1973441513219"></a>输入</p>
</td>
<td class="cellrowborder" valign="top" width="74.16%" headers="mcps1.2.4.1.3 "><p id="p1573415151214"><a name="p1573415151214"></a><a name="p1573415151214"></a>srcTensor的shape信息。SoftMaxShapeInfo类型，具体定义如下：</p>
<a name="screen1740392817204"></a><a name="screen1740392817204"></a><pre class="screen" codetype="Cpp" id="screen1740392817204">struct SoftMaxShapeInfo {
uint32_t srcM; // 非尾轴长度的乘积
uint32_t srcK; // 尾轴长度，必须32Byte对齐
uint32_t oriSrcM; // 原始非尾轴长度的乘积
uint32_t oriSrcK;  // 原始尾轴长度
};</pre>
<p id="p1065853610376"><a name="p1065853610376"></a><a name="p1065853610376"></a>需要注意，当输入输出的数据格式为NZ格式时，尾轴长度为reduce轴长度即<a href="SoftMax.md#fig0172155842215">图2</a>中的W0*W1，非尾轴为H0*H1。</p>
</td>
</tr>
</tbody>
</table>

## 返回值说明<a name="section640mcpsimp"></a>

无

## 约束说明<a name="section633mcpsimp"></a>

-   srcTensor和dstTensor的Tensor的空间可以复用，maxTensor和inMaxTensor的空间可以复用，expSumTensor和inExpSumTensor的空间可以复用。
-   除模板参数config配置为非拓展模式（SoftmaxMode::SOFTMAX\_OUTPUT\_WITHOUT\_BRC）的场景外，expSumTensor、maxTensor、expMaxTensor、inExpSumTensor、inMaxTensor的Tensor空间，last轴长度必须固定32Byte。
-   对于输出ReduceMax的接口：
    -   模板参数isReuseSource、isDataFormatNZ、config.isCheckTiling均为预留参数；
    -   config.mode只支持配置为非拓展模式SOFTMAX\_OUTPUT\_WITHOUT\_BRC，其配置为SOFTMAX\_NORMAL模式时，接口功能不执行，不保存各输出；
    -   模板参数isUpdate为false时，outReduceMax不输出；
    -   除outReduceMax外，其余每个输出的计算结果与[不输出ReduceMax的接口](#section620mcpsimp)相同。

-   操作数地址对齐要求请参见[通用地址对齐约束](通用说明和约束.md#section796754519912)。
-   不支持sharedTmpBuffer与源操作数和目的操作数地址重叠。

## 调用示例<a name="section94691236101419"></a>

-   srcK对齐

    本样例中输入srcTensor和输出dstTensor的shape大小为\[320,64\]，输入inSumTensor、inMaxTensor的shape大小为\[320,16\]，输出expMaxTensor的shape大小为\[320,16\]，数据类型均为half，输入输出的数据排布格式为ND，srcTensor和dstTensor空间不复用，不使能基本块，isUpdate为true。算子样例请参考[softmaxflashv2算子样例](https://gitcode.com/cann/asc-devkit/tree/master/examples/03_libraries/02_activation/softmaxflashv2)。

    ```
    #include "kernel_operator.h"
    
    // constexpr AscendC::SoftmaxConfig static_config = {true, 320, 64}; shape常量化使用
    AscendC::LocalTensor<T> srcLocal = inQueueSrc.DeQue<T>();
    AscendC::LocalTensor<T> insumLocal = sumQueue.DeQue<T>();
    AscendC::LocalTensor<T> inmaxLocal = maxQueue.DeQue<T>();
    AscendC::LocalTensor<T> expMaxTensor = expMaxQueue.AllocTensor<T>();
    AscendC::LocalTensor<T> dstLocal = outQueueDst.AllocTensor<T>();
    AscendC::SoftMaxShapeInfo srcShape = {height, width, height, width};
    AscendC::SoftmaxFlashV2<T, true>(
        dstLocal, insumLocal, inmaxLocal, srcLocal, expMaxTensor, insumLocal, inmaxLocal, tiling, srcShape);
    //AscendC::SoftmaxFlashV2<T, true, false, false, false, static_config>(dstLocal, insumLocal, inmaxLocal, srcLocal,
    //expMaxTensor, insumLocal, inmaxLocal, tiling, srcShape);使用SoftmaxConfig类型的参数static_config,传入模板参数将shape常量化
    outQueueDst.EnQue<T>(dstLocal);
    maxQueue.FreeTensor(inmaxLocal);
    sumQueue.FreeTensor(insumLocal);
    inQueueSrc.FreeTensor(srcLocal);
    ```

-   srcK非对齐

    本样例中srcTensor和输出dstTensor的shape大小为\[320,63\]，数据类型均为half，输入输出的数据排布格式为ND，展示非对齐padding补齐的搬入搬出操作和API调用方式。

    ```
    #include "kernel_operator.h"
    // init阶段  height=320, width=63
    padWidth = AlignUp(width * sizeof(T), 32) / sizeof(T);
    // copyin阶段
    AscendC::DataCopyExtParams copyParams{static_cast<uint16_t>(height), static_cast<uint32_t>(width * sizeof(T)), 0, 0, 0};
    AscendC::DataCopyPadExtParams<T> padParam = {true, 0, static_cast<uint8_t>(padWidth - width), 0};
    AscendC::DataCopyPad(srcLocal, srcGlobal, copyParams, padParam);
    AscendC::DataCopy(insumLocal, sumGlobal, height * elementNumPerBlk);
    AscendC::DataCopy(inmaxLocal, maxGlobal, height * elementNumPerBlk);
    inQueueSrc.EnQue(srcLocal);
    sumQueue.EnQue(insumLocal);
    maxQueue.EnQue(inmaxLocal);
    // compute阶段
    AscendC::LocalTensor<T> srcLocal = inQueueSrc.DeQue<T>();
    AscendC::LocalTensor<T> insumLocal = sumQueue.DeQue<T>();
    AscendC::LocalTensor<T> inmaxLocal = maxQueue.DeQue<T>();
    AscendC::LocalTensor<T> expMaxTensor = expMaxQueue.AllocTensor<T>();
    AscendC::LocalTensor<T> dstLocal = outQueueDst.AllocTensor<T>();
    // 由于发生padding，API调用时shape和原始shape发生了不一致
    AscendC::SoftMaxShapeInfo srcShape = {height, padWidth, height, width};
    AscendC::SoftmaxFlashV2<T, true>(
    	dstLocal, insumLocal, inmaxLocal, srcLocal, expMaxTensor, insumLocal, inmaxLocal, tiling, srcShape);
    // copyout阶段
    AscendC::LocalTensor<T> dstLocal = outQueueDst.DeQue<T>();
    AscendC::DataCopyExtParams copyParams{static_cast<uint16_t>(height), static_cast<uint32_t>(width * sizeof(T)), 0, 0, 0};
    AscendC::DataCopyPad(dstGlobal, dstLocal, copyParams);
    outQueueDst.FreeTensor(dstLocal);
    ```

