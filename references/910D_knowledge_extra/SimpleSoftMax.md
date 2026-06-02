# SimpleSoftMax<a name="ZH-CN_TOPIC_0000002523304064"></a>

## 产品支持情况<a name="section1550532418810"></a>

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

将输入tensor\[m<sub>0</sub>, m<sub>1</sub>, ...m<sub>t</sub>, n\]（t大于等于0）的非尾轴长度相乘的结果看作m，则输入tensor的shape看作\[m, n\]。对输入tensor\[m,n\]按行做如下计算，与[SoftMax](SoftMax.md)接口不同，该接口内部没有reduce过程计算sum和max数据，而是使用计算好的sum和max数据对输入tensor做Softmax计算。计算公式如下：

<!-- img2text -->
$$dst_{ij} = \frac{e^{src_{ij} - max_i}}{sum_i}$$

为方便理解，通过Python脚本实现的方式，表达其计算公式如下，其中src、max、sum是源操作数（输入），dst为目的操作数（输出）。

```
def simple_softmax(src, max, sum):
    dst = np.exp(src - max)/sum
    return dst
```

## 实现原理<a name="section13229175017585"></a>

以float类型，ND格式，shape为\[m, k\]的输入Tensor为例，描述SimpleSoftMax高阶API内部算法框图，如下图所示。

**图 1**  SimpleSoftMax算法框图<a name="fig8262115061111"></a>  
<!-- img2text -->
```text
                         ┌────────┐
                         │ x[m,k] │
                         └────────┘
                              │
                              ▼
            ┌──────────────────────────────────────┐
┌──────────┐│                                      │
│ max[m,8] │──→ ┌────────────────────────────────┐ │
└──────────┘│   │             y=sub              │ │
            │   │    (x[m,k] - max[m,8])         │ │
            │   └────────────────────────────────┘ │
            │                  │                   │
            │                  ▼                   │
            │   ┌────────────────────────────────┐ │
            │   │           exp(y[m,k])          │ │
            │   └────────────────────────────────┘ │
            │                  │                   │
┌──────────┐│                  ▼                   │
│ sum[m,8] │──→ ┌────────────────────────────────┐ │
└──────────┘│   │              div               │ │
            │   │      (y[m,k] / sum[m,8])       │ │
            │   └────────────────────────────────┘ │
            └──────────────────┬───────────────────┘
                               │
                               ▼
                         ┌────────┐
                         │ y[m,k] │
                         └────────┘


图示:
输入输出Tensor   ┌──────────┐
                │          │
                └──────────┘

vector计算      ┌────────────────────────────┐
                │                            │
                └────────────────────────────┘

数据流向        ─────→
```

计算过程分为如下几步，均在Vector上进行：

1.sub步骤：对输入x的所有数据按行减去输入的max；

2.exp步骤：对sub之后的所有数据求exp；

3.div步骤：对exp结果的所有数据按行除以输入的sum，得到结果；

## 函数原型<a name="section620mcpsimp"></a>

-   接口框架申请临时空间
    -   LocalTensor的数据类型相同

        ```
        template <typename T, bool isReuseSource = false, bool isBasicBlock = false, bool isDataFormatNZ = false, const SoftmaxConfig& config = SOFTMAX_DEFAULT_CFG> 
        __aicore__ inline void SimpleSoftMax(const LocalTensor<T>& dstTensor, const LocalTensor<T>& inSumTensor, const LocalTensor<T>& inMaxTensor, const LocalTensor<T>& srcTensor, const SoftMaxTiling& tiling, const SoftMaxShapeInfo& softmaxShapeInfo = {})
        ```

    -   LocalTensor的数据类型不同

        ```
        template <typename T, bool isReuseSource = false, bool isBasicBlock = false, bool isDataFormatNZ = false, const SoftmaxConfig& config = SOFTMAX_DEFAULT_CFG>
        __aicore__ inline void SimpleSoftMax(const LocalTensor<half>& dstTensor, const LocalTensor<float>& inSumTensor, const LocalTensor<float>& inMaxTensor, const LocalTensor<half>& srcTensor, const SoftMaxTiling& tiling, const SoftMaxShapeInfo& softmaxShapeInfo = {})
        ```

-   通过sharedTmpBuffer入参传入临时空间
    -   LocalTensor的数据类型相同

        ```
        template <typename T, bool isReuseSource = false, bool isBasicBlock = false, bool isDataFormatNZ = false, const SoftmaxConfig& config = SOFTMAX_DEFAULT_CFG> 
        __aicore__ inline void SimpleSoftMax(const LocalTensor<T>& dstTensor, const LocalTensor<T>& inSumTensor, const LocalTensor<T>& inMaxTensor, const LocalTensor<T>& srcTensor, const LocalTensor<uint8_t>& sharedTmpBuffer, const SoftMaxTiling& tiling, const SoftMaxShapeInfo& softmaxShapeInfo = {})
        ```

    -   LocalTensor的数据类型不同

        ```
        template <typename T, bool isReuseSource = false, bool isBasicBlock = false, bool isDataFormatNZ = false, const SoftmaxConfig& config = SOFTMAX_DEFAULT_CFG>
        __aicore__ inline void SimpleSoftMax(const LocalTensor<half>& dstTensor, const LocalTensor<float>& inSumTensor, const LocalTensor<float>& inMaxTensor, const LocalTensor<half>& srcTensor, const LocalTensor<uint8_t>& sharedTmpBuffer, const SoftMaxTiling& tiling, const SoftMaxShapeInfo& softmaxShapeInfo = {})
        ```

由于该接口的内部实现中涉及复杂的计算，需要额外的临时空间来存储计算过程中的中间变量。临时空间支持**接口框架申请**和开发者**通过sharedTmpBuffer入参传入**两种方式。

-   接口框架申请临时空间，开发者无需申请，但是需要预留临时空间的大小。

-   通过sharedTmpBuffer入参传入，使用该tensor作为临时空间进行处理，接口框架不再申请。该方式开发者可以自行管理sharedTmpBuffer内存空间，并在接口调用完成后，复用该部分内存，内存不会反复申请释放，灵活性较高，内存利用率也较高。

接口框架申请的方式，开发者需要预留临时空间；通过sharedTmpBuffer传入的情况，开发者需要为tensor申请空间。临时空间大小BufferSize的获取方式如下：通过[SoftMax/SimpleSoftMax Tiling](SoftMax-SimpleSoftMax-Tiling.md)中提供的GetSoftMaxMaxTmpSize/GetSoftMaxMinTmpSize接口获取所需最大和最小临时空间大小，最小空间可以保证功能正确，最大空间用于提升性能。

## 参数说明<a name="section622mcpsimp"></a>

**表 1**  模板参数说明

<a name="table575571914269"></a>
<table><thead align="left"><tr id="zh-cn_topic_0000001621597661_row18755131942614"><th class="cellrowborder" valign="top" width="19.39%" id="mcps1.2.3.1.1"><p id="zh-cn_topic_0000001621597661_p675519193268"><a name="zh-cn_topic_0000001621597661_p675519193268"></a><a name="zh-cn_topic_0000001621597661_p675519193268"></a>参数名</p>
</th>
<th class="cellrowborder" valign="top" width="80.61%" id="mcps1.2.3.1.2"><p id="zh-cn_topic_0000001621597661_p375511918267"><a name="zh-cn_topic_0000001621597661_p375511918267"></a><a name="zh-cn_topic_0000001621597661_p375511918267"></a>描述</p>
</th>
</tr>
</thead>
<tbody><tr id="zh-cn_topic_0000001621597661_row14755141911264"><td class="cellrowborder" valign="top" width="19.39%" headers="mcps1.2.3.1.1 "><p id="zh-cn_topic_0000001621597661_p47551198266"><a name="zh-cn_topic_0000001621597661_p47551198266"></a><a name="zh-cn_topic_0000001621597661_p47551198266"></a>T</p>
</td>
<td class="cellrowborder" valign="top" width="80.61%" headers="mcps1.2.3.1.2 "><p id="zh-cn_topic_0000001621597661_p125969172719"><a name="zh-cn_topic_0000001621597661_p125969172719"></a><a name="zh-cn_topic_0000001621597661_p125969172719"></a>操作数的数据类型。</p>
<p id="p12719118196"><a name="p12719118196"></a><a name="p12719118196"></a><span id="ph1839423196"><a name="ph1839423196"></a><a name="ph1839423196"></a>Ascend 950PR/Ascend 950DT</span>，支持的数据类型为：half、float。</p>
</td>
</tr>
<tr id="zh-cn_topic_0000001621597661_row9756719122620"><td class="cellrowborder" valign="top" width="19.39%" headers="mcps1.2.3.1.1 "><p id="zh-cn_topic_0000001621597661_p1682112447268"><a name="zh-cn_topic_0000001621597661_p1682112447268"></a><a name="zh-cn_topic_0000001621597661_p1682112447268"></a>isReuseSource</p>
</td>
<td class="cellrowborder" valign="top" width="80.61%" headers="mcps1.2.3.1.2 "><p id="p595664384319"><a name="p595664384319"></a><a name="p595664384319"></a>该参数预留，传入默认值false即可。</p>
</td>
</tr>
<tr id="zh-cn_topic_0000001621597661_row9184124919159"><td class="cellrowborder" valign="top" width="19.39%" headers="mcps1.2.3.1.1 "><p id="zh-cn_topic_0000001621597661_p11692440141619"><a name="zh-cn_topic_0000001621597661_p11692440141619"></a><a name="zh-cn_topic_0000001621597661_p11692440141619"></a>isBasicBlock</p>
</td>
<td class="cellrowborder" valign="top" width="80.61%" headers="mcps1.2.3.1.2 "><p id="zh-cn_topic_0000001621597661_p1153245821619"><a name="zh-cn_topic_0000001621597661_p1153245821619"></a><a name="zh-cn_topic_0000001621597661_p1153245821619"></a>srcTensor和dstTensor的shape信息和Tiling切分策略满足基本块要求的情况下，可以使能该参数用于提升性能，默认不使能。是否满足基本块的要求，可以采用如下两种方式之一判断：</p>
<a name="ul7845163710593"></a><a name="ul7845163710593"></a><ul id="ul7845163710593"><li>srcTensor和dstTensor的shape信息[m,n]需要满足如下条件：<a name="ul3845133755912"></a><a name="ul3845133755912"></a><ul id="ul3845133755912"><li>尾轴长度n小于2048并且大于等于256/sizeof(T)（即half场景下n最小为128，float场景下n最小为64），同时n是64的倍数；</li><li>非尾轴长度的乘积m为8的倍数。</li></ul>
</li></ul>
<a name="zh-cn_topic_0000001621597661_ul353295811167"></a><a name="zh-cn_topic_0000001621597661_ul353295811167"></a><ul id="zh-cn_topic_0000001621597661_ul353295811167"><li>在Tiling实现中，通过调用<a href="IsBasicBlockInSoftMax.md">IsBasicBlockInSoftMax</a>判断Tiling切分策略是否满足基本块的切分要求。</li></ul>
</td>
</tr>
<tr id="zh-cn_topic_0000001621597661_row1276635121510"><td class="cellrowborder" valign="top" width="19.39%" headers="mcps1.2.3.1.1 "><p id="zh-cn_topic_0000001621597661_p1495244819166"><a name="zh-cn_topic_0000001621597661_p1495244819166"></a><a name="zh-cn_topic_0000001621597661_p1495244819166"></a>isDataFormatNZ</p>
</td>
<td class="cellrowborder" valign="top" width="80.61%" headers="mcps1.2.3.1.2 "><p id="zh-cn_topic_0000001621597661_p207441566176"><a name="zh-cn_topic_0000001621597661_p207441566176"></a><a name="zh-cn_topic_0000001621597661_p207441566176"></a>当前输入输出的数据格式是否为NZ格式，默认数据格式为ND，即默认取值为false。</p>
</td>
</tr>
<tr id="zh-cn_topic_0000001621597661_row1925664115510"><td class="cellrowborder" valign="top" width="19.39%" headers="mcps1.2.3.1.1 "><p id="zh-cn_topic_0000001621597661_p69641412246"><a name="zh-cn_topic_0000001621597661_p69641412246"></a><a name="zh-cn_topic_0000001621597661_p69641412246"></a>config</p>
</td>
<td class="cellrowborder" valign="top" width="80.61%" headers="mcps1.2.3.1.2 "><p id="zh-cn_topic_0000001621597661_p1744211392255"><a name="zh-cn_topic_0000001621597661_p1744211392255"></a><a name="zh-cn_topic_0000001621597661_p1744211392255"></a>结构体模板参数，此参数可选配，SoftmaxConfig类型，具体定义如下。</p>
<a name="zh-cn_topic_0000001621597661_screen16476116195910"></a><a name="zh-cn_topic_0000001621597661_screen16476116195910"></a><pre class="screen" codetype="Cpp" id="zh-cn_topic_0000001621597661_screen16476116195910">struct SoftmaxConfig{
    bool isCheckTiling = true; // 是否需要检查shape和tiling的一致性；若不一致，API内会根据shape重新计算所需tiling。默认取值true：API内部会检查一致性
    uint32_t oriSrcM = 0; // 原始非尾轴长度的乘积。设置该参数后，将shape常量化，编译过程中使用常量化的shape
    uint32_t oriSrcK = 0; // 原始尾轴长度。设置该参数后，将shape常量化，编译过程中使用常量化的shape
};</pre>
<p id="zh-cn_topic_0000001621597661_p76421594583"><a name="zh-cn_topic_0000001621597661_p76421594583"></a><a name="zh-cn_topic_0000001621597661_p76421594583"></a>配置示例如下。</p>
<a name="zh-cn_topic_0000001621597661_screen19241326175913"></a><a name="zh-cn_topic_0000001621597661_screen19241326175913"></a><pre class="screen" codetype="Cpp" id="zh-cn_topic_0000001621597661_screen19241326175913">constexpr SoftmaxConfig SOFTMAX_DEFAULT_CFG = {true, 0, 0};</pre>
<p id="zh-cn_topic_0000001621597661_p19442739102517"><a name="zh-cn_topic_0000001621597661_p19442739102517"></a><a name="zh-cn_topic_0000001621597661_p19442739102517"></a>此参数一般用于配合kernel侧tiling计算的接口使用。</p>
<p id="p1860515359212"><a name="p1860515359212"></a><a name="p1860515359212"></a>注意：config参数生效的优先级低于模板参数isBasicBlock，即使能isBasicBlock参数时，接口内部做基本块的切分优化，config参数的shape常量化不生效。</p>
<p id="p16541183012147"><a name="p16541183012147"></a><a name="p16541183012147"></a><span id="ph254193071416"><a name="ph254193071416"></a><a name="ph254193071416"></a>Ascend 950PR/Ascend 950DT</span>，该参数为预留参数，暂未启用，保持默认值即可。</p>
</td>
</tr>
</tbody>
</table>

**表 2**  接口参数说明

<a name="zh-cn_topic_0235751031_table33761356"></a>
<table><thead align="left"><tr id="zh-cn_topic_0235751031_row27598891"><th class="cellrowborder" valign="top" width="17.77%" id="mcps1.2.4.1.1"><p id="zh-cn_topic_0235751031_p20917673"><a name="zh-cn_topic_0235751031_p20917673"></a><a name="zh-cn_topic_0235751031_p20917673"></a>参数名</p>
</th>
<th class="cellrowborder" valign="top" width="8.08%" id="mcps1.2.4.1.2"><p id="zh-cn_topic_0235751031_p16609919"><a name="zh-cn_topic_0235751031_p16609919"></a><a name="zh-cn_topic_0235751031_p16609919"></a>输入/输出</p>
</th>
<th class="cellrowborder" valign="top" width="74.15%" id="mcps1.2.4.1.3"><p id="zh-cn_topic_0235751031_p59995477"><a name="zh-cn_topic_0235751031_p59995477"></a><a name="zh-cn_topic_0235751031_p59995477"></a>描述</p>
</th>
</tr>
</thead>
<tbody><tr id="row42461942101815"><td class="cellrowborder" valign="top" width="17.77%" headers="mcps1.2.4.1.1 "><p id="p154351349191918"><a name="p154351349191918"></a><a name="p154351349191918"></a>dstTensor</p>
</td>
<td class="cellrowborder" valign="top" width="8.08%" headers="mcps1.2.4.1.2 "><p id="p843620496191"><a name="p843620496191"></a><a name="p843620496191"></a>输出</p>
</td>
<td class="cellrowborder" valign="top" width="74.15%" headers="mcps1.2.4.1.3 "><p id="p194521547192018"><a name="p194521547192018"></a><a name="p194521547192018"></a>目的操作数。</p>
<p id="p16703131355116"><a name="p16703131355116"></a><a name="p16703131355116"></a><span id="zh-cn_topic_0000002523303824_ph173308471594"><a name="zh-cn_topic_0000002523303824_ph173308471594"></a><a name="zh-cn_topic_0000002523303824_ph173308471594"></a><span id="zh-cn_topic_0000002523303824_ph9902231466"><a name="zh-cn_topic_0000002523303824_ph9902231466"></a><a name="zh-cn_topic_0000002523303824_ph9902231466"></a><span id="zh-cn_topic_0000002523303824_ph1782115034816"><a name="zh-cn_topic_0000002523303824_ph1782115034816"></a><a name="zh-cn_topic_0000002523303824_ph1782115034816"></a>类型为<a href="LocalTensor.md">LocalTensor</a>，支持的TPosition为VECIN/VECCALC/VECOUT。</span></span></span></p>
<p id="p1143644911915"><a name="p1143644911915"></a><a name="p1143644911915"></a>dstTensor的shape和源操作数srcTensor一致。</p>
</td>
</tr>
<tr id="row2137145181815"><td class="cellrowborder" valign="top" width="17.77%" headers="mcps1.2.4.1.1 "><p id="p9436204921915"><a name="p9436204921915"></a><a name="p9436204921915"></a>inSumTensor</p>
</td>
<td class="cellrowborder" valign="top" width="8.08%" headers="mcps1.2.4.1.2 "><p id="p1436124951919"><a name="p1436124951919"></a><a name="p1436124951919"></a>输入</p>
</td>
<td class="cellrowborder" valign="top" width="74.15%" headers="mcps1.2.4.1.3 "><p id="p262422213"><a name="p262422213"></a><a name="p262422213"></a>源操作数。</p>
<p id="p19313124162113"><a name="p19313124162113"></a><a name="p19313124162113"></a><span id="zh-cn_topic_0000002523303824_ph173308471594_1"><a name="zh-cn_topic_0000002523303824_ph173308471594_1"></a><a name="zh-cn_topic_0000002523303824_ph173308471594_1"></a><span id="zh-cn_topic_0000002523303824_ph9902231466_1"><a name="zh-cn_topic_0000002523303824_ph9902231466_1"></a><a name="zh-cn_topic_0000002523303824_ph9902231466_1"></a><span id="zh-cn_topic_0000002523303824_ph1782115034816_1"><a name="zh-cn_topic_0000002523303824_ph1782115034816_1"></a><a name="zh-cn_topic_0000002523303824_ph1782115034816_1"></a>类型为<a href="LocalTensor.md">LocalTensor</a>，支持的TPosition为VECIN/VECCALC/VECOUT。</span></span></span></p>
<p id="p76602013123618"><a name="p76602013123618"></a><a name="p76602013123618"></a>softmax计算所需要的sum值。</p>
<a name="ul6386131510362"></a><a name="ul6386131510362"></a><ul id="ul6386131510362"><li>inSumTensor的last轴长度固定为32Byte，即一个<span id="ph885316116307"><a name="ph885316116307"></a><a name="ph885316116307"></a>datablock</span>长度。该<span id="ph774662243116"><a name="ph774662243116"></a><a name="ph774662243116"></a>datablock</span>中的所有数据为同一个值，比如half数据类型下，该<span id="ph275720223362"><a name="ph275720223362"></a><a name="ph275720223362"></a>datablock</span>中的16个数均为相同的值。</li><li>非last轴的长度需要与dstTensor保持一致。</li></ul>
</td>
</tr>
<tr id="row19615183817191"><td class="cellrowborder" valign="top" width="17.77%" headers="mcps1.2.4.1.1 "><p id="p34361549181911"><a name="p34361549181911"></a><a name="p34361549181911"></a>inMaxTensor</p>
</td>
<td class="cellrowborder" valign="top" width="8.08%" headers="mcps1.2.4.1.2 "><p id="p1643616497196"><a name="p1643616497196"></a><a name="p1643616497196"></a>输入</p>
</td>
<td class="cellrowborder" valign="top" width="74.15%" headers="mcps1.2.4.1.3 "><p id="p55755116453"><a name="p55755116453"></a><a name="p55755116453"></a>源操作数。</p>
<p id="p20123173352114"><a name="p20123173352114"></a><a name="p20123173352114"></a><span id="zh-cn_topic_0000002523303824_ph173308471594_2"><a name="zh-cn_topic_0000002523303824_ph173308471594_2"></a><a name="zh-cn_topic_0000002523303824_ph173308471594_2"></a><span id="zh-cn_topic_0000002523303824_ph9902231466_2"><a name="zh-cn_topic_0000002523303824_ph9902231466_2"></a><a name="zh-cn_topic_0000002523303824_ph9902231466_2"></a><span id="zh-cn_topic_0000002523303824_ph1782115034816_2"><a name="zh-cn_topic_0000002523303824_ph1782115034816_2"></a><a name="zh-cn_topic_0000002523303824_ph1782115034816_2"></a>类型为<a href="LocalTensor.md">LocalTensor</a>，支持的TPosition为VECIN/VECCALC/VECOUT。</span></span></span></p>
<p id="p247013293372"><a name="p247013293372"></a><a name="p247013293372"></a>softmax计算所需要的max值。</p>
<a name="ul2353125216375"></a><a name="ul2353125216375"></a><ul id="ul2353125216375"><li>inMaxTensor的last轴长度固定为32Byte，即一个<span id="ph830432833111"><a name="ph830432833111"></a><a name="ph830432833111"></a>datablock</span>长度。该<span id="ph3238123493119"><a name="ph3238123493119"></a><a name="ph3238123493119"></a>datablock</span>中的所有数据为同一个值，比如half数据类型下，该<span id="ph147543818361"><a name="ph147543818361"></a><a name="ph147543818361"></a>datablock</span>里的16个数均为相同的值。</li><li>非last轴的长度需要与dstTensor保持一致。</li></ul>
</td>
</tr>
<tr id="row250163914199"><td class="cellrowborder" valign="top" width="17.77%" headers="mcps1.2.4.1.1 "><p id="p194361349181915"><a name="p194361349181915"></a><a name="p194361349181915"></a>srcTensor</p>
</td>
<td class="cellrowborder" valign="top" width="8.08%" headers="mcps1.2.4.1.2 "><p id="p8436124913195"><a name="p8436124913195"></a><a name="p8436124913195"></a>输入</p>
</td>
<td class="cellrowborder" valign="top" width="74.15%" headers="mcps1.2.4.1.3 "><p id="p135875510456"><a name="p135875510456"></a><a name="p135875510456"></a>源操作数。</p>
<p id="p113692368211"><a name="p113692368211"></a><a name="p113692368211"></a><span id="zh-cn_topic_0000002523303824_ph173308471594_3"><a name="zh-cn_topic_0000002523303824_ph173308471594_3"></a><a name="zh-cn_topic_0000002523303824_ph173308471594_3"></a><span id="zh-cn_topic_0000002523303824_ph9902231466_3"><a name="zh-cn_topic_0000002523303824_ph9902231466_3"></a><a name="zh-cn_topic_0000002523303824_ph9902231466_3"></a><span id="zh-cn_topic_0000002523303824_ph1782115034816_3"><a name="zh-cn_topic_0000002523303824_ph1782115034816_3"></a><a name="zh-cn_topic_0000002523303824_ph1782115034816_3"></a>类型为<a href="LocalTensor.md">LocalTensor</a>，支持的TPosition为VECIN/VECCALC/VECOUT。</span></span></span></p>
<p id="p15436154912196"><a name="p15436154912196"></a><a name="p15436154912196"></a>last轴长度需要32B对齐。</p>
</td>
</tr>
<tr id="row7115123204313"><td class="cellrowborder" valign="top" width="17.77%" headers="mcps1.2.4.1.1 "><p id="p1358462510237"><a name="p1358462510237"></a><a name="p1358462510237"></a>sharedTmpBuffer</p>
</td>
<td class="cellrowborder" valign="top" width="8.08%" headers="mcps1.2.4.1.2 "><p id="p15584152512313"><a name="p15584152512313"></a><a name="p15584152512313"></a>输入</p>
</td>
<td class="cellrowborder" valign="top" width="74.15%" headers="mcps1.2.4.1.3 "><p id="p155841325182310"><a name="p155841325182310"></a><a name="p155841325182310"></a>临时空间。</p>
<p id="p5983205412394"><a name="p5983205412394"></a><a name="p5983205412394"></a><span id="zh-cn_topic_0000002523303824_ph173308471594_4"><a name="zh-cn_topic_0000002523303824_ph173308471594_4"></a><a name="zh-cn_topic_0000002523303824_ph173308471594_4"></a><span id="zh-cn_topic_0000002523303824_ph9902231466_4"><a name="zh-cn_topic_0000002523303824_ph9902231466_4"></a><a name="zh-cn_topic_0000002523303824_ph9902231466_4"></a><span id="zh-cn_topic_0000002523303824_ph1782115034816_4"><a name="zh-cn_topic_0000002523303824_ph1782115034816_4"></a><a name="zh-cn_topic_0000002523303824_ph1782115034816_4"></a>类型为<a href="LocalTensor.md">LocalTensor</a>，支持的TPosition为VECIN/VECCALC/VECOUT。</span></span></span></p>
<p id="p8928165616507"><a name="p8928165616507"></a><a name="p8928165616507"></a>该操作数的数据类型固定uint8_t。</p>
<p id="p104071111204211"><a name="p104071111204211"></a><a name="p104071111204211"></a>接口内部复杂计算时用于存储中间变量，由开发者提供。</p>
<p id="p5881016172817"><a name="p5881016172817"></a><a name="p5881016172817"></a>临时空间大小BufferSize的获取方式请参考<a href="SoftMax-SimpleSoftMax-Tiling.md">SoftMax/SimpleSoftMax Tiling</a>。</p>
</td>
</tr>
<tr id="row041413392198"><td class="cellrowborder" valign="top" width="17.77%" headers="mcps1.2.4.1.1 "><p id="p1743618495191"><a name="p1743618495191"></a><a name="p1743618495191"></a>tiling</p>
</td>
<td class="cellrowborder" valign="top" width="8.08%" headers="mcps1.2.4.1.2 "><p id="p543664917198"><a name="p543664917198"></a><a name="p543664917198"></a>输入</p>
</td>
<td class="cellrowborder" valign="top" width="74.15%" headers="mcps1.2.4.1.3 "><p id="p44361949111915"><a name="p44361949111915"></a><a name="p44361949111915"></a>softmax计算所需tiling信息，Tiling信息的获取请参考<a href="SoftMax-SimpleSoftMax-Tiling.md">SoftMax/SimpleSoftMax Tiling</a>。</p>
</td>
</tr>
<tr id="row1440510124410"><td class="cellrowborder" valign="top" width="17.77%" headers="mcps1.2.4.1.1 "><p id="p0734515621"><a name="p0734515621"></a><a name="p0734515621"></a>softmaxShapeInfo</p>
</td>
<td class="cellrowborder" valign="top" width="8.08%" headers="mcps1.2.4.1.2 "><p id="p1973441513219"><a name="p1973441513219"></a><a name="p1973441513219"></a>输入</p>
</td>
<td class="cellrowborder" valign="top" width="74.15%" headers="mcps1.2.4.1.3 "><p id="p1573415151214"><a name="p1573415151214"></a><a name="p1573415151214"></a>srcTensor的shape信息。SoftMaxShapeInfo类型，具体定义如下：</p>
<a name="screen1740392817204"></a><a name="screen1740392817204"></a><pre class="screen" codetype="Cpp" id="screen1740392817204">struct SoftMaxShapeInfo {
uint32_t srcM; // 非尾轴长度的乘积
uint32_t srcK; // 尾轴长度，必须32Byte对齐
uint32_t oriSrcM; // 原始非尾轴长度的乘积
uint32_t oriSrcK;  // 原始尾轴长度
};</pre>
<p id="p11532172382714"><a name="p11532172382714"></a><a name="p11532172382714"></a>需要注意，当输入输出的数据格式为NZ格式时，尾轴长度为reduce轴长度即<a href="SoftMax.md#fig0172155842215">图2</a>中的W0*W1，非尾轴为H0*H1。</p>
</td>
</tr>
</tbody>
</table>

## 返回值说明<a name="section640mcpsimp"></a>

无

## 约束说明<a name="section633mcpsimp"></a>

-   srcTensor和dstTensor的Tensor空间可以复用。
-   inSumTensor和inMaxTensor为输入，并且last轴长度必须固定32Byte。
-   inSumTensor和inMaxTensor的数据类型需要保持一致。
-   操作数地址对齐要求请参见[通用地址对齐约束](通用说明和约束.md#section796754519912)。
-   不支持sharedTmpBuffer与源操作数和目的操作数地址重叠。

## 调用示例<a name="section94691236101419"></a>

本样例中输入srcTensor和输出dstTensor的shape大小为\[320,64\]，输入inSumTensor和inMaxTensor的shape大小为\[320,16\]，数据类型均为half。算子样例请参考[simplesoftmax算子样例](https://gitcode.com/cann/asc-devkit/tree/master/examples/03_libraries/02_activation/simplesoftmax)。

```
#include "kernel_operator.h"

// constexpr AscendC::SoftmaxConfig static_config = {true, 320, 64}; shape常量化使用
AscendC::LocalTensor<T> srcLocal = inQueueSrc.DeQue<T>();
AscendC::LocalTensor<T> sumTempLocal = sumQueue.DeQue<T>();
AscendC::LocalTensor<T> maxTempLocal = maxQueue.DeQue<T>();
AscendC::LocalTensor<T> dstLocal = outQueueDst.AllocTensor<T>();

AscendC::SoftMaxShapeInfo srcShape = {height, width, height, width};
AscendC::SimpleSoftMax<T>(dstLocal, sumTempLocal, maxTempLocal, srcLocal, tiling, srcShape);
//AscendC::SimpleSoftMax<T, false, false, static_config>(dstLocal, sumTempLocal, maxTempLocal, srcLocal, tiling, //srcShape);使用SoftmaxConfig类型的参数static_config，传入模板参数将shape常量化

outQueueDst.EnQue<T>(dstLocal);
maxQueue.FreeTensor(maxTempLocal);
sumQueue.FreeTensor(sumTempLocal);
inQueueSrc.FreeTensor(srcLocal);
```

