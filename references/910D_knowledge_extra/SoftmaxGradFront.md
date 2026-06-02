# SoftmaxGradFront<a name="ZH-CN_TOPIC_0000002554424601"></a>

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

## 功能说明<a name="section13281349161713"></a>

将输入tensor\[m<sub>0</sub>, m<sub>1</sub>, ...m<sub>t</sub>, n\]（t大于等于0）的非尾轴长度相乘的结果看作m，则输入tensor的shape看作\[m, n\]。对输入tensor\[m,n\]按行做gradfront反向计算，计算公式如下：

<!-- img2text -->
$$
\begin{cases}
d_i = y_i \times \left( dx_i - \sum_{j=1}^{n} \left( y_j \times dx_j \right) \right), & i = 1,2,\ldots,n \\
n \text{：输入 tensor 按行做 gradfront 反向计算时，每行元素个数}
\end{cases}
$$

当输入shape为ND格式时，内部的reduce过程按last轴进行；当输入shape为NZ格式时，内部的reduce过程按照last轴和first轴进行，reduce过程可以参考[SoftMax](SoftMax.md)中的图示说明。

为方便理解，通过Python脚本实现的方式，表达其计算公式如下，其中dx、y是源操作数（输入），d为目的操作数（输出）。

```
def softmax_grad_front(dx, y, is_fp16=False):
    dx = dx.astype(np.float32)
    y = y.astype(np.float32)

    d = (dx * y).sum(axis=-1, keepdims=True)  ###[1024,1]
    if is_fp16:
    d = d.astype(np.float16)
    return d
```

## 实现原理<a name="section13229175017585"></a>

以float类型，ND格式，shape为\[m, k\]的输入Tensor为例，描述SoftmaxGradFront高阶API内部算法框图，如下图所示。

**图 1**  SoftmaxGradFront算法框图<a name="fig149465713201"></a>  
<!-- img2text -->
```text
┌──────────────┐                           ┌──────────────┐
│    x[m,k]    │                           │    y[m,k]    │
└──────────────┘                           └──────────────┘
        │                                         │
        │                                         │
        │    ┌──────────────────────────────────────────────────────┐
        └───→│                                                      │←───┘
             │                  ┌──────────────────────┐            │
             │                  │         mul          │            │
             │                  │  (x[m,k] * y[m,k])   │            │
             │                  └──────────────────────┘            │
             │                             │                        │
             │                             ↓                        │
             │                  ┌──────────────────────┐            │
             │                  │      reducesum       │            │
             │                  │   ([m,k]->[m,1])     │            │
             │                  └──────────────────────┘            │
             │                             │                        │
             │                             ↓                        │
             │                  ┌──────────────────────┐            │
             │                  │      broadcast       │            │
             │                  │   ([m,1]->[m,8])     │            │
             │                  └──────────────────────┘            │
             └───────────────────────────┬──────────────────────────┘
                                         │
                                         ↓
                                  ┌──────────────┐
                                  │    z[m,8]    │
                                  └──────────────┘


图示:
输入输出Tensor   ┌──────────────┐
                │              │
                └──────────────┘

vector计算      ┌──────────────────────┐
                │                      │
                └──────────────────────┘

数据流向        ─────────→
```

计算过程分为如下几步，均在Vector上进行：

1.  mul步骤：对输入x和y所有数据相乘，计算结果会保存到一个临时空间temp中；
2.  reducesum步骤：对temp中的数据\(\[m, k\]\)每一行数据求和得到\[m, 1\]，计算结果保存到临时空间中；
3.  broadcast步骤：对\[m, 1\]做一个按datablock为单位的填充，比如float类型下，把\[m, 1\]扩展成\[m, 8\]，并输出结果z。

## 函数原型<a name="section161381959151619"></a>

-   接口框架申请临时空间

    ```
    template <typename T, bool isBasicBlock = false, bool isDataFormatNZ = false>
    __aicore__ inline void SoftmaxGradFront(const LocalTensor<T>& dstTensor, const LocalTensor<T>& gradTensor, const LocalTensor<T>& srcTensor, const SoftMaxTiling& tiling, const SoftMaxShapeInfo& softmaxShapeInfo = {})
    ```

-   通过sharedTmpBuffer入参传入临时空间

    ```
    template <typename T, bool isBasicBlock = false, bool isDataFormatNZ = false>
    __aicore__ inline void SoftmaxGradFront(const LocalTensor<T>& dstTensor, const LocalTensor<T>& gradTensor, const LocalTensor<T>& srcTensor, const LocalTensor<uint8_t>& sharedTmpBuffer, const SoftMaxTiling& tiling, const SoftMaxShapeInfo& softmaxShapeInfo = {})
    ```

由于该接口的内部实现中涉及复杂的计算，需要额外的临时空间来存储计算过程中的中间变量。临时空间支持**接口框架申请**和开发者**通过sharedTmpBuffer入参传入**两种方式。

-   接口框架申请临时空间，开发者无需申请，但是需要预留临时空间的大小。

-   通过sharedTmpBuffer入参传入，使用该tensor作为临时空间进行处理，接口框架不再申请。该方式开发者可以自行管理sharedTmpBuffer内存空间，并在接口调用完成后，复用该部分内存，内存不会反复申请释放，灵活性较高，内存利用率也较高。

接口框架申请的方式，开发者需要预留临时空间；通过sharedTmpBuffer传入的情况，开发者需要为tensor申请空间。临时空间大小BufferSize的获取方式如下：通过[SoftmaxGrad Tiling接口](SoftmaxGrad-Tiling接口.md)中提供的GetSoftMaxGradMaxTmpSize/GetSoftMaxGradMinTmpSize接口获取所需最小和最大临时空间大小，最小空间可以保证功能正确，最大空间用于提升性能。

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
<p id="p18538146185016"><a name="p18538146185016"></a><a name="p18538146185016"></a><span id="ph19539104619508"><a name="ph19539104619508"></a><a name="ph19539104619508"></a>Ascend 950PR/Ascend 950DT</span>，支持的数据类型为：half、float。</p>
</td>
</tr>
<tr id="row9184124919159"><td class="cellrowborder" valign="top" width="19.39%" headers="mcps1.2.3.1.1 "><p id="p11692440141619"><a name="p11692440141619"></a><a name="p11692440141619"></a>isBasicBlock</p>
</td>
<td class="cellrowborder" valign="top" width="80.61%" headers="mcps1.2.3.1.2 "><p id="p1694110152717"><a name="p1694110152717"></a><a name="p1694110152717"></a>srcTensor和gradTensor的shape信息和Tiling切分策略满足基本块要求的情况下，可以使能该参数用于提升性能，默认不使能。是否满足基本块的要求，可以采用如下两种方式之一判断：</p>
<a name="ul353295811167"></a><a name="ul353295811167"></a><ul id="ul353295811167"><li>srcTensor和dstTensor的shape信息[m,n]需要满足如下条件：<a name="ul09181366549"></a><a name="ul09181366549"></a><ul id="ul09181366549"><li>尾轴长度n小于2048并且大于等于256/sizeof(T)（即half场景下n最小为128，float场景下n最小为64），同时n是64的倍数；</li><li>非尾轴长度的乘积m为8的倍数。</li></ul>
</li></ul>
<a name="ul1994111022714"></a><a name="ul1994111022714"></a><ul id="ul1994111022714"><li>在Tiling实现中，通过调用<a href="IsBasicBlockInSoftMax.md">IsBasicBlockInSoftMax</a>判断Tiling切分策略是否满足基本块的切分要求。</li></ul>
</td>
</tr>
<tr id="row1276635121510"><td class="cellrowborder" valign="top" width="19.39%" headers="mcps1.2.3.1.1 "><p id="p1495244819166"><a name="p1495244819166"></a><a name="p1495244819166"></a>isDataFormatNZ</p>
</td>
<td class="cellrowborder" valign="top" width="80.61%" headers="mcps1.2.3.1.2 "><p id="p5521396273"><a name="p5521396273"></a><a name="p5521396273"></a>当前输入输出的数据格式是否为NZ格式，默认数据格式为ND，即默认取值为false。</p>
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
<tbody><tr id="row42461942101815"><td class="cellrowborder" valign="top" width="17.77%" headers="mcps1.2.4.1.1 "><p id="p132394662514"><a name="p132394662514"></a><a name="p132394662514"></a>dstTensor</p>
</td>
<td class="cellrowborder" valign="top" width="8.07%" headers="mcps1.2.4.1.2 "><p id="p623976112513"><a name="p623976112513"></a><a name="p623976112513"></a>输出</p>
</td>
<td class="cellrowborder" valign="top" width="74.16%" headers="mcps1.2.4.1.3 "><p id="p318727114519"><a name="p318727114519"></a><a name="p318727114519"></a>目的操作数。</p>
<p id="p16703131355116"><a name="p16703131355116"></a><a name="p16703131355116"></a><span id="zh-cn_topic_0000002523303824_ph173308471594"><a name="zh-cn_topic_0000002523303824_ph173308471594"></a><a name="zh-cn_topic_0000002523303824_ph173308471594"></a><span id="zh-cn_topic_0000002523303824_ph9902231466"><a name="zh-cn_topic_0000002523303824_ph9902231466"></a><a name="zh-cn_topic_0000002523303824_ph9902231466"></a><span id="zh-cn_topic_0000002523303824_ph1782115034816"><a name="zh-cn_topic_0000002523303824_ph1782115034816"></a><a name="zh-cn_topic_0000002523303824_ph1782115034816"></a>类型为<a href="LocalTensor.md">LocalTensor</a>，支持的TPosition为VECIN/VECCALC/VECOUT。</span></span></span></p>
<p id="p1143644911915"><a name="p1143644911915"></a><a name="p1143644911915"></a>last轴长度固定32Byte即一个<span id="ph5215105273412"><a name="ph5215105273412"></a><a name="ph5215105273412"></a>datablock</span>长度，并且该<span id="ph20629853153413"><a name="ph20629853153413"></a><a name="ph20629853153413"></a>datablock</span>中的所有数据为同一个值。比如half数据类型下，该<span id="ph1858665514340"><a name="ph1858665514340"></a><a name="ph1858665514340"></a>datablock</span>里的16个数均为相同的值，非last轴长度需要和srcTensor保持一致。</p>
</td>
</tr>
<tr id="row2137145181815"><td class="cellrowborder" valign="top" width="17.77%" headers="mcps1.2.4.1.1 "><p id="p1423915622510"><a name="p1423915622510"></a><a name="p1423915622510"></a>gradTensor</p>
</td>
<td class="cellrowborder" valign="top" width="8.07%" headers="mcps1.2.4.1.2 "><p id="p1723911613251"><a name="p1723911613251"></a><a name="p1723911613251"></a>输入</p>
</td>
<td class="cellrowborder" valign="top" width="74.16%" headers="mcps1.2.4.1.3 "><p id="p423976102519"><a name="p423976102519"></a><a name="p423976102519"></a>源操作数。</p>
<p id="p22831358308"><a name="p22831358308"></a><a name="p22831358308"></a><span id="zh-cn_topic_0000002523303824_ph173308471594_1"><a name="zh-cn_topic_0000002523303824_ph173308471594_1"></a><a name="zh-cn_topic_0000002523303824_ph173308471594_1"></a><span id="zh-cn_topic_0000002523303824_ph9902231466_1"><a name="zh-cn_topic_0000002523303824_ph9902231466_1"></a><a name="zh-cn_topic_0000002523303824_ph9902231466_1"></a><span id="zh-cn_topic_0000002523303824_ph1782115034816_1"><a name="zh-cn_topic_0000002523303824_ph1782115034816_1"></a><a name="zh-cn_topic_0000002523303824_ph1782115034816_1"></a>类型为<a href="LocalTensor.md">LocalTensor</a>，支持的TPosition为VECIN/VECCALC/VECOUT。</span></span></span></p>
<p id="p1943333118619"><a name="p1943333118619"></a><a name="p1943333118619"></a>last轴长度需要32Byte对齐，<span>gradTensor的shape与srcTensor的shape一致</span>。</p>
</td>
</tr>
<tr id="row19615183817191"><td class="cellrowborder" valign="top" width="17.77%" headers="mcps1.2.4.1.1 "><p id="p19239062256"><a name="p19239062256"></a><a name="p19239062256"></a>srcTensor</p>
</td>
<td class="cellrowborder" valign="top" width="8.07%" headers="mcps1.2.4.1.2 "><p id="p52399642514"><a name="p52399642514"></a><a name="p52399642514"></a>输入</p>
</td>
<td class="cellrowborder" valign="top" width="74.16%" headers="mcps1.2.4.1.3 "><p id="p1965113819615"><a name="p1965113819615"></a><a name="p1965113819615"></a>源操作数。</p>
<p id="p7694739163415"><a name="p7694739163415"></a><a name="p7694739163415"></a><span id="zh-cn_topic_0000002523303824_ph173308471594_2"><a name="zh-cn_topic_0000002523303824_ph173308471594_2"></a><a name="zh-cn_topic_0000002523303824_ph173308471594_2"></a><span id="zh-cn_topic_0000002523303824_ph9902231466_2"><a name="zh-cn_topic_0000002523303824_ph9902231466_2"></a><a name="zh-cn_topic_0000002523303824_ph9902231466_2"></a><span id="zh-cn_topic_0000002523303824_ph1782115034816_2"><a name="zh-cn_topic_0000002523303824_ph1782115034816_2"></a><a name="zh-cn_topic_0000002523303824_ph1782115034816_2"></a>类型为<a href="LocalTensor.md">LocalTensor</a>，支持的TPosition为VECIN/VECCALC/VECOUT。</span></span></span></p>
<p id="p26517382613"><a name="p26517382613"></a><a name="p26517382613"></a>last轴长度需要32Byte对齐，<span>srcTensor的shape与gradTensor的shape一致</span>。</p>
</td>
</tr>
<tr id="row9520125782015"><td class="cellrowborder" valign="top" width="17.77%" headers="mcps1.2.4.1.1 "><p id="p1358462510237"><a name="p1358462510237"></a><a name="p1358462510237"></a>sharedTmpBuffer</p>
</td>
<td class="cellrowborder" valign="top" width="8.07%" headers="mcps1.2.4.1.2 "><p id="p15584152512313"><a name="p15584152512313"></a><a name="p15584152512313"></a>输入</p>
</td>
<td class="cellrowborder" valign="top" width="74.16%" headers="mcps1.2.4.1.3 "><p id="p155841325182310"><a name="p155841325182310"></a><a name="p155841325182310"></a>临时空间。</p>
<p id="p5983205412394"><a name="p5983205412394"></a><a name="p5983205412394"></a><span id="zh-cn_topic_0000002523303824_ph173308471594_3"><a name="zh-cn_topic_0000002523303824_ph173308471594_3"></a><a name="zh-cn_topic_0000002523303824_ph173308471594_3"></a><span id="zh-cn_topic_0000002523303824_ph9902231466_3"><a name="zh-cn_topic_0000002523303824_ph9902231466_3"></a><a name="zh-cn_topic_0000002523303824_ph9902231466_3"></a><span id="zh-cn_topic_0000002523303824_ph1782115034816_3"><a name="zh-cn_topic_0000002523303824_ph1782115034816_3"></a><a name="zh-cn_topic_0000002523303824_ph1782115034816_3"></a>类型为<a href="LocalTensor.md">LocalTensor</a>，支持的TPosition为VECIN/VECCALC/VECOUT。</span></span></span></p>
<p id="p8928165616507"><a name="p8928165616507"></a><a name="p8928165616507"></a>该操作数的数据类型固定uint8_t。</p>
<p id="p104071111204211"><a name="p104071111204211"></a><a name="p104071111204211"></a>用于接口内部复杂计算时存储中间变量，由开发者提供。</p>
<p id="p5881016172817"><a name="p5881016172817"></a><a name="p5881016172817"></a>临时空间大小BufferSize的获取方式请参考<a href="SoftmaxGrad-Tiling接口.md">SoftmaxGrad Tiling接口</a>。</p>
</td>
</tr>
<tr id="row266583012111"><td class="cellrowborder" valign="top" width="17.77%" headers="mcps1.2.4.1.1 "><p id="p1743618495191"><a name="p1743618495191"></a><a name="p1743618495191"></a>tiling</p>
</td>
<td class="cellrowborder" valign="top" width="8.07%" headers="mcps1.2.4.1.2 "><p id="p543664917198"><a name="p543664917198"></a><a name="p543664917198"></a>输入</p>
</td>
<td class="cellrowborder" valign="top" width="74.16%" headers="mcps1.2.4.1.3 "><p id="p44361949111915"><a name="p44361949111915"></a><a name="p44361949111915"></a>softmaxgradfront计算所需tiling信息，Tiling信息的获取请参考<a href="SoftmaxGrad-Tiling接口.md">SoftmaxGrad Tiling接口</a>。</p>
</td>
</tr>
<tr id="row138606561892"><td class="cellrowborder" valign="top" width="17.77%" headers="mcps1.2.4.1.1 "><p id="p0734515621"><a name="p0734515621"></a><a name="p0734515621"></a>softmaxShapeInfo</p>
</td>
<td class="cellrowborder" valign="top" width="8.07%" headers="mcps1.2.4.1.2 "><p id="p1973441513219"><a name="p1973441513219"></a><a name="p1973441513219"></a>输入</p>
</td>
<td class="cellrowborder" valign="top" width="74.16%" headers="mcps1.2.4.1.3 "><p id="p1573415151214"><a name="p1573415151214"></a><a name="p1573415151214"></a>srcTensor的shape信息。SoftMaxShapeInfo类型，具体定义如下：</p>
<a name="screen173011433407"></a><a name="screen173011433407"></a><pre class="screen" codetype="Cpp" id="screen173011433407">struct SoftMaxShapeInfo {
    uint32_t srcM; // 非尾轴乘积长度
    uint32_t srcK; // 尾轴长度，必须32Byte对齐
    uint32_t oriSrcM; // 原始非尾轴乘积长度
    uint32_t oriSrcK;  // 原始尾轴长度
};</pre>
<p id="p106970420229"><a name="p106970420229"></a><a name="p106970420229"></a>需要注意，当输入输出的数据格式为NZ格式时，尾轴长度为reduce轴长度即<a href="SoftMax.md#fig0172155842215">图2</a>中的W0*W1，非尾轴为H0*H1。</p>
</td>
</tr>
</tbody>
</table>

## 返回值说明<a name="section640mcpsimp"></a>

无

## 约束说明<a name="section633mcpsimp"></a>

-   操作数地址对齐要求请参见[通用地址对齐约束](通用说明和约束.md#section796754519912)。
-   不支持sharedTmpBuffer与源操作数和目的操作数地址重叠。

## 调用示例<a name="section94691236101419"></a>

本样例输入srcTensor的Shape大小为\[128,64\]，输入gradtensor的Shape大小为\[128,64\]，输出dstTensor的Shape大小为\[128,16\]，数据类型均为half，输入输出的数据排布格式为ND，不使能基本块。算子样例请参考[softmaxgradfront算子样例](https://gitcode.com/cann/asc-devkit/tree/master/examples/03_libraries/02_activation/softmaxgradfront)。

```
#include "kernel_operator.h"

AscendC::LocalTensor<T> srcLocal1 = inQueueSrc1.DeQue<T>();
AscendC::LocalTensor<T> srcLocal2 = inQueueSrc2.DeQue<T>();
AscendC::LocalTensor<T> dstLocal = outQueueDst.AllocTensor<T>();
AscendC::SoftMaxShapeInfo srcShape = { height, width, height, width };
AscendC::SoftmaxGradFront<T>(dstLocal, srcLocal2, srcLocal1, tiling, srcShape);
outQueueDst.EnQue<T>(dstLocal);
inQueueSrc1.FreeTensor(srcLocal1);
inQueueSrc2.FreeTensor(srcLocal2);
```

