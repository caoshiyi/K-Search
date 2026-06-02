# SoftmaxGrad<a name="ZH-CN_TOPIC_0000002523344088"></a>

## 产品支持情况<a name="section24011736183811"></a>

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

将输入tensor\[m<sub>0</sub>, m<sub>1</sub>, ...m<sub>t</sub>, n\]（t大于等于0）的非尾轴长度相乘的结果看作m，则输入tensor的shape看作\[m, n\]。对输入tensor\[m,n\]按行做grad反向计算，计算公式如下：

<!-- img2text -->
$$dx_i = \left( dy_i - \sum_{k=1}^{n} y_k \times dy_k \right) \times y_i,\quad i \in [1,n]$$

当输入shape为ND格式时，内部的reduce过程按last轴进行；当输入shape为NZ格式时，内部的reduce过程按照last轴和first轴进行，reduce过程可以参考[SoftMax](SoftMax.md)中的图示说明。

为方便理解，通过Python脚本实现的方式，表达其计算公式如下，其中src、grad、isFront是源操作数（输入），dst为目的操作数（输出）。

```
def softmax_grad(grad, src, isFront = None):
    dst = grad * src
    dst = np.sum(dst, axis=-1, keepdims=True)
    if isFront :
         return dst
    dst = (grad - dst) * src
    return dst
```

## 实现原理<a name="section13229175017585"></a>

以float类型，ND格式，shape为\[m,k\]的输入Tensor为例，描述SoftmaxGrad高阶API内部算法框图，如下图所示。

**图 1**  SoftmaxGrad算法框图<a name="fig1590310503117"></a>  
<!-- img2text -->
```text
                    ┌─────────┐                               ┌─────────┐
                    │ x[m,k]  │                               │ y[m,k]  │
                    └────┬────┘                               └────┬────┘
                         │                                           │
                         │      ┌───────────────────────────────┐    │
                         └────→ │              mul              │ ←──┘
                                │      (x[m,k] * y[m,k])        │
                                └──────────────┬────────────────┘
                                               │
                                               ↓
                                ┌───────────────────────────────┐
                                │           reducesum           │
                                │        ([m,k]->[m,1])         │
                                └──────────────┬────────────────┘
                                               │
                                               ↓
                                ┌───────────────────────────────┐
                                │           broadcast           │
                                │        ([m,1]->[m,8])         │
                                └──────────────┬────────────────┘
                                               │
                                               ↓
                                         ┌───────────┐
                                         │  isFront  │
                                         └─────┬─────┘
                                      True ────┘ └──── False
                                               │
                                               │
                                               │                      ┌────────┐
                                               ├────────────────────→ │ z[m,8] │
                                               │                      └────────┘
                                               │
                                               ↓
                                ┌───────────────────────────────┐
                                │          t=broadcast          │
                                │        ([m,8]->[m,k])         │
                                └──────────────┬────────────────┘
                                               │
                                               ↓
                    ┌─────────┐    ┌───────────────────────────────┐
                    │ x[m,k]  │ ─→ │             z=sub             │
                    └─────────┘    │       (x[m,k] - t[m,k])       │
                                   └──────────────┬────────────────┘
                                                  │
                                                  ↓
                                   ┌───────────────────────────────┐
                    ┌─────────┐    │              mul              │
                    │ y[m,k]  │ ─→ │       (z[m,k] * y[m,k])       │
                    └─────────┘    └──────────────┬────────────────┘
                                                  │
                                                  ↓
                                              ┌────────┐
                                              │ z[m,k] │
                                              └────────┘


图示:
输入输出Tensor    ┌──────────┐
                 │          │
                 └──────────┘

vector计算       ┌──────────┐
                 │          │
                 └──────────┘

条件判断            ◇

数据流向             ───→
```

计算过程分为如下几步，均在Vector上进行：

1.  mul步骤：对输入x和y所有数据相乘，计算结果会保存到一个临时空间temp中；
2.  reducesum步骤：对temp数据\(\[m, k\]\)每一行求和得到\[m, 1\]，计算结果会保存到临时空间中；
3.  broadcast步骤：对reducesum结果\[m, 1\]的数据做一个按datablock为单位的填充，比如float类型下，把\[m, 1\]扩展成\[m, 8\]；
4.  判断是否isFront模式，如果是，则输出broadcast后的结果，计算结束；如果不是，则继续执行后续步骤；
5.  broadcast步骤：对\[m, 8\]做一个扩维，扩展成\[m, k\]，计算结果会保存到临时空间中；
6.  sub步骤：输入x的所有数据减去上一步broadcast后的结果；
7.  mul步骤：sub后的所有数据和输入y相乘，输出结果z。

## 函数原型<a name="section620mcpsimp"></a>

-   接口框架申请临时空间

    ```
    template <typename T, bool isReuseSource = false, bool isDataFormatNZ = false>
    __aicore__ inline void SoftmaxGrad(const LocalTensor<T>& dstTensor, const LocalTensor<T>& gradTensor, const LocalTensor<T>& srcTensor, const SoftMaxTiling& tiling, bool isFront = false, const SoftMaxShapeInfo& softmaxShapeInfo = {})
    ```

-   通过sharedTmpBuffer入参传入临时空间

    ```
    template <typename T, bool isReuseSource = false, bool isDataFormatNZ = false>
    __aicore__ inline void SoftmaxGrad(const LocalTensor<T>& dstTensor, const LocalTensor<T>& gradTensor, const LocalTensor<T>& srcTensor, const LocalTensor<uint8_t>& sharedTmpBuffer, const SoftMaxTiling& tiling, bool isFront = false, const SoftMaxShapeInfo& softmaxShapeInfo = {})
    ```

由于该接口的内部实现中涉及复杂的计算，需要额外的临时空间来存储计算过程中的中间变量。临时空间支持**接口框架申请**和开发者**通过sharedTmpBuffer入参传入**两种方式。

-   接口框架申请临时空间，开发者无需申请，但是需要预留临时空间的大小。

-   通过sharedTmpBuffer入参传入，使用该tensor作为临时空间进行处理，接口框架不再申请。该方式开发者可以自行管理sharedTmpBuffer内存空间，并在接口调用完成后，复用该部分内存，内存不会反复申请释放，灵活性较高，内存利用率也较高。

接口框架申请的方式，开发者需要预留临时空间；通过sharedTmpBuffer传入的情况，开发者需要为tensor申请空间。临时空间大小BufferSize的获取方式如下：通过[SoftmaxGrad Tiling接口](SoftmaxGrad-Tiling接口.md)中提供的GetSoftMaxGradMaxTmpSize/GetSoftMaxGradMinTmpSize接口获取所需最大和最小临时空间大小，最小空间可以保证功能正确，最大空间用于提升性能。

## 参数说明<a name="section622mcpsimp"></a>

**表 1**  模板参数说明

<a name="table575571914269"></a>
<table><thead align="left"><tr id="row18755131942614"><th class="cellrowborder" valign="top" width="19.36%" id="mcps1.2.3.1.1"><p id="p675519193268"><a name="p675519193268"></a><a name="p675519193268"></a>参数名</p>
</th>
<th class="cellrowborder" valign="top" width="80.64%" id="mcps1.2.3.1.2"><p id="p375511918267"><a name="p375511918267"></a><a name="p375511918267"></a>描述</p>
</th>
</tr>
</thead>
<tbody><tr id="row14755141911264"><td class="cellrowborder" valign="top" width="19.36%" headers="mcps1.2.3.1.1 "><p id="p47551198266"><a name="p47551198266"></a><a name="p47551198266"></a>T</p>
</td>
<td class="cellrowborder" valign="top" width="80.64%" headers="mcps1.2.3.1.2 "><p id="p125969172719"><a name="p125969172719"></a><a name="p125969172719"></a>操作数的数据类型。</p>
<p id="p2422117123915"><a name="p2422117123915"></a><a name="p2422117123915"></a><span id="ph1656602653918"><a name="ph1656602653918"></a><a name="ph1656602653918"></a>Ascend 950PR/Ascend 950DT</span>，支持的数据类型为：half、float。</p>
</td>
</tr>
<tr id="row9756719122620"><td class="cellrowborder" valign="top" width="19.36%" headers="mcps1.2.3.1.1 "><p id="p1682112447268"><a name="p1682112447268"></a><a name="p1682112447268"></a>isReuseSource</p>
</td>
<td class="cellrowborder" valign="top" width="80.64%" headers="mcps1.2.3.1.2 "><p id="p10997182034512"><a name="p10997182034512"></a><a name="p10997182034512"></a>该参数预留，传入默认值false即可。</p>
</td>
</tr>
<tr id="row9184124919159"><td class="cellrowborder" valign="top" width="19.36%" headers="mcps1.2.3.1.1 "><p id="p11692440141619"><a name="p11692440141619"></a><a name="p11692440141619"></a>isDataFormatNZ</p>
</td>
<td class="cellrowborder" valign="top" width="80.64%" headers="mcps1.2.3.1.2 "><p id="p1225514711224"><a name="p1225514711224"></a><a name="p1225514711224"></a>当前输入输出的数据格式是否为NZ格式，默认数据格式为ND，即默认取值为false。</p>
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
<p id="p1143644911915"><a name="p1143644911915"></a><a name="p1143644911915"></a>last轴长度需要32Byte对齐，<span>dstTensor的shape与gradTensor，srcTensor的shape一致</span>。</p>
</td>
</tr>
<tr id="row2137145181815"><td class="cellrowborder" valign="top" width="17.77%" headers="mcps1.2.4.1.1 "><p id="p1423915622510"><a name="p1423915622510"></a><a name="p1423915622510"></a>gradTensor</p>
</td>
<td class="cellrowborder" valign="top" width="8.07%" headers="mcps1.2.4.1.2 "><p id="p1723911613251"><a name="p1723911613251"></a><a name="p1723911613251"></a>输入</p>
</td>
<td class="cellrowborder" valign="top" width="74.16%" headers="mcps1.2.4.1.3 "><p id="p423976102519"><a name="p423976102519"></a><a name="p423976102519"></a>源操作数。</p>
<p id="p172731910162320"><a name="p172731910162320"></a><a name="p172731910162320"></a><span id="zh-cn_topic_0000002523303824_ph173308471594_1"><a name="zh-cn_topic_0000002523303824_ph173308471594_1"></a><a name="zh-cn_topic_0000002523303824_ph173308471594_1"></a><span id="zh-cn_topic_0000002523303824_ph9902231466_1"><a name="zh-cn_topic_0000002523303824_ph9902231466_1"></a><a name="zh-cn_topic_0000002523303824_ph9902231466_1"></a><span id="zh-cn_topic_0000002523303824_ph1782115034816_1"><a name="zh-cn_topic_0000002523303824_ph1782115034816_1"></a><a name="zh-cn_topic_0000002523303824_ph1782115034816_1"></a>类型为<a href="LocalTensor.md">LocalTensor</a>，支持的TPosition为VECIN/VECCALC/VECOUT。</span></span></span></p>
<p id="p1943333118619"><a name="p1943333118619"></a><a name="p1943333118619"></a>last轴长度需要32Byte对齐，<span>gradTensor的shape与dstTensor，srcTensor的shape一致</span>。</p>
</td>
</tr>
<tr id="row19615183817191"><td class="cellrowborder" valign="top" width="17.77%" headers="mcps1.2.4.1.1 "><p id="p19239062256"><a name="p19239062256"></a><a name="p19239062256"></a>srcTensor</p>
</td>
<td class="cellrowborder" valign="top" width="8.07%" headers="mcps1.2.4.1.2 "><p id="p52399642514"><a name="p52399642514"></a><a name="p52399642514"></a>输入</p>
</td>
<td class="cellrowborder" valign="top" width="74.16%" headers="mcps1.2.4.1.3 "><p id="p1965113819615"><a name="p1965113819615"></a><a name="p1965113819615"></a>源操作数。</p>
<p id="p882271372310"><a name="p882271372310"></a><a name="p882271372310"></a><span id="zh-cn_topic_0000002523303824_ph173308471594_2"><a name="zh-cn_topic_0000002523303824_ph173308471594_2"></a><a name="zh-cn_topic_0000002523303824_ph173308471594_2"></a><span id="zh-cn_topic_0000002523303824_ph9902231466_2"><a name="zh-cn_topic_0000002523303824_ph9902231466_2"></a><a name="zh-cn_topic_0000002523303824_ph9902231466_2"></a><span id="zh-cn_topic_0000002523303824_ph1782115034816_2"><a name="zh-cn_topic_0000002523303824_ph1782115034816_2"></a><a name="zh-cn_topic_0000002523303824_ph1782115034816_2"></a>类型为<a href="LocalTensor.md">LocalTensor</a>，支持的TPosition为VECIN/VECCALC/VECOUT。</span></span></span></p>
<p id="p26517382613"><a name="p26517382613"></a><a name="p26517382613"></a>last轴长度需要32Byte对齐，<span>srcTensor的shape与dstTensor，</span><span>gradTensor</span><span>的shape一致</span>。</p>
</td>
</tr>
<tr id="row041413392198"><td class="cellrowborder" valign="top" width="17.77%" headers="mcps1.2.4.1.1 "><p id="p1358462510237"><a name="p1358462510237"></a><a name="p1358462510237"></a>sharedTmpBuffer</p>
</td>
<td class="cellrowborder" valign="top" width="8.07%" headers="mcps1.2.4.1.2 "><p id="p15584152512313"><a name="p15584152512313"></a><a name="p15584152512313"></a>输入</p>
</td>
<td class="cellrowborder" valign="top" width="74.16%" headers="mcps1.2.4.1.3 "><p id="p155841325182310"><a name="p155841325182310"></a><a name="p155841325182310"></a>临时空间。</p>
<p id="p5983205412394"><a name="p5983205412394"></a><a name="p5983205412394"></a><span id="zh-cn_topic_0000002523303824_ph173308471594_3"><a name="zh-cn_topic_0000002523303824_ph173308471594_3"></a><a name="zh-cn_topic_0000002523303824_ph173308471594_3"></a><span id="zh-cn_topic_0000002523303824_ph9902231466_3"><a name="zh-cn_topic_0000002523303824_ph9902231466_3"></a><a name="zh-cn_topic_0000002523303824_ph9902231466_3"></a><span id="zh-cn_topic_0000002523303824_ph1782115034816_3"><a name="zh-cn_topic_0000002523303824_ph1782115034816_3"></a><a name="zh-cn_topic_0000002523303824_ph1782115034816_3"></a>类型为<a href="LocalTensor.md">LocalTensor</a>，支持的TPosition为VECIN/VECCALC/VECOUT。</span></span></span></p>
<p id="p8928165616507"><a name="p8928165616507"></a><a name="p8928165616507"></a>该操作数的数据类型固定uint8_t。</p>
<p id="p104071111204211"><a name="p104071111204211"></a><a name="p104071111204211"></a>接口内部复杂计算时用于存储中间变量，由开发者提供。</p>
<p id="p5881016172817"><a name="p5881016172817"></a><a name="p5881016172817"></a>临时空间大小BufferSize的获取方式请参考<a href="SoftmaxGrad-Tiling接口.md">SoftmaxGrad Tiling接口</a>。</p>
</td>
</tr>
<tr id="row5322195318719"><td class="cellrowborder" valign="top" width="17.77%" headers="mcps1.2.4.1.1 "><p id="p0734515621"><a name="p0734515621"></a><a name="p0734515621"></a>softmaxShapeInfo</p>
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
<p id="p225252394219"><a name="p225252394219"></a><a name="p225252394219"></a>需要注意，当输入输出的数据格式为NZ格式时，尾轴长度为reduce轴长度即<a href="SoftMax.md#fig0172155842215">图2</a>中的W0*W1，非尾轴为H0*H1。</p>
</td>
</tr>
<tr id="row1334965141118"><td class="cellrowborder" valign="top" width="17.77%" headers="mcps1.2.4.1.1 "><p id="p1743618495191"><a name="p1743618495191"></a><a name="p1743618495191"></a>tiling</p>
</td>
<td class="cellrowborder" valign="top" width="8.07%" headers="mcps1.2.4.1.2 "><p id="p543664917198"><a name="p543664917198"></a><a name="p543664917198"></a>输入</p>
</td>
<td class="cellrowborder" valign="top" width="74.16%" headers="mcps1.2.4.1.3 "><p id="p44361949111915"><a name="p44361949111915"></a><a name="p44361949111915"></a>softmaxgrad计算所需tiling信息，Tiling信息的获取请参考<a href="SoftmaxGrad-Tiling接口.md">SoftmaxGrad Tiling接口</a>。</p>
</td>
</tr>
<tr id="row79279624420"><td class="cellrowborder" valign="top" width="17.77%" headers="mcps1.2.4.1.1 "><p id="p89611074441"><a name="p89611074441"></a><a name="p89611074441"></a>isFront</p>
</td>
<td class="cellrowborder" valign="top" width="8.07%" headers="mcps1.2.4.1.2 "><p id="p1896113710445"><a name="p1896113710445"></a><a name="p1896113710445"></a>输入</p>
</td>
<td class="cellrowborder" valign="top" width="74.16%" headers="mcps1.2.4.1.3 "><p id="p196115794419"><a name="p196115794419"></a><a name="p196115794419"></a>是否使能isFront计算，若为True，dstTensor的last轴长度必须固定32Byte。</p>
</td>
</tr>
</tbody>
</table>

## 返回值说明<a name="section640mcpsimp"></a>

无

## 约束说明<a name="section633mcpsimp"></a>

-   srcTensor和dstTensor的Tensor空间可以复用。
-   操作数地址对齐要求请参见[通用地址对齐约束](通用说明和约束.md#section796754519912)。
-   不支持sharedTmpBuffer与源操作数和目的操作数地址重叠。

## 调用示例<a name="section94691236101419"></a>

本样例中输入srcTensor、gradtensor和输出dstTensor的Shape大小均为\[128,64\]，isFront为false，数据类型均为half，输入输出的数据排布格式为ND，srcTensor和dstTensor空间不复用，不使能基本块。算子样例请参考[softmaxgrad算子样例](https://gitcode.com/cann/asc-devkit/tree/master/examples/03_libraries/02_activation/softmaxgrad)。

```
#include "kernel_operator.h"

AscendC::LocalTensor<T> srcLocal1 = inQueueSrc1.DeQue<T>();
AscendC::LocalTensor<T> srcLocal2 = inQueueSrc2.DeQue<T>();
AscendC::LocalTensor<T> dstLocal = outQueueDst.AllocTensor<T>();

AscendC::SoftMaxShapeInfo srcShape = {height, width, height, width};
AscendC::SoftmaxGrad<T>(dstLocal, srcLocal2, srcLocal1, tiling, false, srcShape);

outQueueDst.EnQue<T>(dstLocal);
inQueueSrc1.FreeTensor(srcLocal1);
inQueueSrc2.FreeTensor(srcLocal2);
```

