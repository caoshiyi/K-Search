# LayerNormGradBeta<a name="ZH-CN_TOPIC_0000002523303730"></a>

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

LayerNormGradBeta接口用于获取反向beta/gamma的数值，和LayerNormGrad共同输出pdx, gamma和beta：

算法公式为:

<!-- img2text -->
$$
\text{dgamma}=\sum_{0}^{N-1}(\text{dy}\times(x-\mu)\times\operatorname{rstd})
$$

$$
\text{dbeta}=\sum_{0}^{N-1}\text{dy}
$$

<!-- img2text -->
$$
dbeta = \sum_{i=1}^{M} dy_i
$$

## 函数原型<a name="section1834111321944"></a>

由于该接口的内部实现中涉及复杂的计算，需要额外的临时空间来存储计算过程中的中间变量。临时空间大小BufferSize的获取方法：通过[LayerNormGradBeta Tiling](LayerNormGradBeta-Tiling.md)中提供的GetLayerNormGradBetaMaxMinTmpSize接口获取所需最大和最小临时空间大小，最小空间可以保证功能正确，最大空间用于提升性能。

临时空间支持**接口框架申请**和开发者**通过sharedTmpBuffer入参传入**两种方式，因此LayerNormGradBeta接口的函数原型有两种：

-   通过sharedTmpBuffer入参传入临时空间

    ```
    template <typename T, bool isReuseSource = false>
    __aicore__ inline void LayerNormGradBeta(const LocalTensor<T>& outputPdGamma, const LocalTensor<T>& outputPdBeta, const LocalTensor<T>& resForGamma, const LocalTensor<T>& inputDy, const LocalTensor<uint8_t>& sharedTmpBuffer, const LayerNormGradBetaTiling& tiling)
    ```

    该方式下开发者需自行申请并管理临时内存空间，并在接口调用完成后，复用该部分内存，内存不会反复申请释放，灵活性较高，内存利用率也较高。

-   接口框架申请临时空间

    ```
    template <typename T, bool isReuseSource = false>
    __aicore__ inline void LayerNormGradBeta(const LocalTensor<T>& outputPdGamma, const LocalTensor<T>& outputPdBeta, const LocalTensor<T>& resForGamma, const LocalTensor<T>& inputDy, LayerNormGradBetaTiling& tiling)
    ```

    该方式下开发者无需申请，但是需要预留临时空间的大小。

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
<p id="p3784111051020"><a name="p3784111051020"></a><a name="p3784111051020"></a><span id="ph578461091013"><a name="ph578461091013"></a><a name="ph578461091013"></a>Ascend 950PR/Ascend 950DT</span>，支持的数据类型为：half、float。</p>
</td>
</tr>
<tr id="row9756719122620"><td class="cellrowborder" valign="top" width="19.39%" headers="mcps1.2.3.1.1 "><p id="p1682112447268"><a name="p1682112447268"></a><a name="p1682112447268"></a>isReuseSource</p>
</td>
<td class="cellrowborder" valign="top" width="80.61%" headers="mcps1.2.3.1.2 "><p id="p515452344815"><a name="p515452344815"></a><a name="p515452344815"></a>是否允许修改源操作数，默认值为false。如果开发者允许源操作数被改写，可以使能该参数，使能后能够节省部分内存空间。</p>
<p id="p6154162344810"><a name="p6154162344810"></a><a name="p6154162344810"></a>设置为<strong id="b015417235488"><a name="b015417235488"></a><a name="b015417235488"></a>true</strong>，则本接口内部计算时<strong id="b161541023194816"><a name="b161541023194816"></a><a name="b161541023194816"></a>复用</strong>inputDy的内存空间，节省内存空间；设置为<strong id="b6154142344810"><a name="b6154142344810"></a><a name="b6154142344810"></a>false</strong>，则本接口内部计算时<strong id="b19154152394811"><a name="b19154152394811"></a><a name="b19154152394811"></a>不复用</strong>inputDy的内存空间。</p>
<p id="p171541323194818"><a name="p171541323194818"></a><a name="p171541323194818"></a>对于float数据类型输入支持开启该参数，half数据类型输入不支持开启该参数。</p>
<p id="p62891018544"><a name="p62891018544"></a><a name="p62891018544"></a>isReuseSource的使用样例请参考<a href="更多样例-104.md#section639165323915">更多样例</a>。</p>
</td>
</tr>
</tbody>
</table>

**表 2**  接口参数说明

<a name="table390516122499"></a>
<table><thead align="left"><tr id="row20948201216498"><th class="cellrowborder" valign="top" width="21.08080808080808%" id="mcps1.2.4.1.1"><p id="p189480124492"><a name="p189480124492"></a><a name="p189480124492"></a>参数名称</p>
</th>
<th class="cellrowborder" valign="top" width="15.636363636363637%" id="mcps1.2.4.1.2"><p id="p18948181212492"><a name="p18948181212492"></a><a name="p18948181212492"></a>输入/输出</p>
</th>
<th class="cellrowborder" valign="top" width="63.28282828282828%" id="mcps1.2.4.1.3"><p id="p99481512204915"><a name="p99481512204915"></a><a name="p99481512204915"></a>含义</p>
</th>
</tr>
</thead>
<tbody><tr id="row19481412174914"><td class="cellrowborder" valign="top" width="21.08080808080808%" headers="mcps1.2.4.1.1 "><p id="p109481912164912"><a name="p109481912164912"></a><a name="p109481912164912"></a>outputPdGamma</p>
</td>
<td class="cellrowborder" valign="top" width="15.636363636363637%" headers="mcps1.2.4.1.2 "><p id="p19481912104910"><a name="p19481912104910"></a><a name="p19481912104910"></a>输出</p>
</td>
<td class="cellrowborder" valign="top" width="63.28282828282828%" headers="mcps1.2.4.1.3 "><p id="p446091145118"><a name="p446091145118"></a><a name="p446091145118"></a>目的操作数，shape为[H]，LocalTensor数据结构的定义请参考<a href="LocalTensor.md">LocalTensor</a>。尾轴长度需要32B对齐</p>
<p id="p175792059241"><a name="p175792059241"></a><a name="p175792059241"></a><span id="zh-cn_topic_0000002523303824_ph173308471594"><a name="zh-cn_topic_0000002523303824_ph173308471594"></a><a name="zh-cn_topic_0000002523303824_ph173308471594"></a><span id="zh-cn_topic_0000002523303824_ph9902231466"><a name="zh-cn_topic_0000002523303824_ph9902231466"></a><a name="zh-cn_topic_0000002523303824_ph9902231466"></a><span id="zh-cn_topic_0000002523303824_ph1782115034816"><a name="zh-cn_topic_0000002523303824_ph1782115034816"></a><a name="zh-cn_topic_0000002523303824_ph1782115034816"></a>类型为<a href="LocalTensor.md">LocalTensor</a>，支持的TPosition为VECIN/VECCALC/VECOUT。</span></span></span></p>
</td>
</tr>
<tr id="row129481312114920"><td class="cellrowborder" valign="top" width="21.08080808080808%" headers="mcps1.2.4.1.1 "><p id="p17948171294912"><a name="p17948171294912"></a><a name="p17948171294912"></a>outputPdBeta</p>
</td>
<td class="cellrowborder" valign="top" width="15.636363636363637%" headers="mcps1.2.4.1.2 "><p id="p6948181224910"><a name="p6948181224910"></a><a name="p6948181224910"></a>输出</p>
</td>
<td class="cellrowborder" valign="top" width="63.28282828282828%" headers="mcps1.2.4.1.3 "><p id="p26477112232"><a name="p26477112232"></a><a name="p26477112232"></a>目的操作数，shape为[H]，LocalTensor数据结构的定义请参考<a href="LocalTensor.md">LocalTensor</a>。尾轴长度需要32B对齐</p>
<p id="p11928986247"><a name="p11928986247"></a><a name="p11928986247"></a><span id="zh-cn_topic_0000002523303824_ph173308471594_1"><a name="zh-cn_topic_0000002523303824_ph173308471594_1"></a><a name="zh-cn_topic_0000002523303824_ph173308471594_1"></a><span id="zh-cn_topic_0000002523303824_ph9902231466_1"><a name="zh-cn_topic_0000002523303824_ph9902231466_1"></a><a name="zh-cn_topic_0000002523303824_ph9902231466_1"></a><span id="zh-cn_topic_0000002523303824_ph1782115034816_1"><a name="zh-cn_topic_0000002523303824_ph1782115034816_1"></a><a name="zh-cn_topic_0000002523303824_ph1782115034816_1"></a>类型为<a href="LocalTensor.md">LocalTensor</a>，支持的TPosition为VECIN/VECCALC/VECOUT。</span></span></span></p>
</td>
</tr>
<tr id="row6948512114912"><td class="cellrowborder" valign="top" width="21.08080808080808%" headers="mcps1.2.4.1.1 "><p id="p294810122497"><a name="p294810122497"></a><a name="p294810122497"></a>resForGamma</p>
</td>
<td class="cellrowborder" valign="top" width="15.636363636363637%" headers="mcps1.2.4.1.2 "><p id="p7948161212495"><a name="p7948161212495"></a><a name="p7948161212495"></a>输入</p>
</td>
<td class="cellrowborder" valign="top" width="63.28282828282828%" headers="mcps1.2.4.1.3 "><p id="p132741529191312"><a name="p132741529191312"></a><a name="p132741529191312"></a>源操作数，shape为[B, S, H]，LocalTensor数据结构的定义请参考<a href="LocalTensor.md">LocalTensor</a>。resForGamma的数据类型需要与目的操作数保持一致，尾轴长度需要32B对齐。需提前调用<a href="LayerNormGrad.md">LayerNormGrad</a>接口获取resForGamma参数值。</p>
<p id="p792715112241"><a name="p792715112241"></a><a name="p792715112241"></a><span id="zh-cn_topic_0000002523303824_ph173308471594_2"><a name="zh-cn_topic_0000002523303824_ph173308471594_2"></a><a name="zh-cn_topic_0000002523303824_ph173308471594_2"></a><span id="zh-cn_topic_0000002523303824_ph9902231466_2"><a name="zh-cn_topic_0000002523303824_ph9902231466_2"></a><a name="zh-cn_topic_0000002523303824_ph9902231466_2"></a><span id="zh-cn_topic_0000002523303824_ph1782115034816_2"><a name="zh-cn_topic_0000002523303824_ph1782115034816_2"></a><a name="zh-cn_topic_0000002523303824_ph1782115034816_2"></a>类型为<a href="LocalTensor.md">LocalTensor</a>，支持的TPosition为VECIN/VECCALC/VECOUT。</span></span></span></p>
</td>
</tr>
<tr id="row209485121498"><td class="cellrowborder" valign="top" width="21.08080808080808%" headers="mcps1.2.4.1.1 "><p id="p194851224915"><a name="p194851224915"></a><a name="p194851224915"></a>inputDy</p>
</td>
<td class="cellrowborder" valign="top" width="15.636363636363637%" headers="mcps1.2.4.1.2 "><p id="p20948312164918"><a name="p20948312164918"></a><a name="p20948312164918"></a>输入</p>
</td>
<td class="cellrowborder" valign="top" width="63.28282828282828%" headers="mcps1.2.4.1.3 "><p id="p83698388549"><a name="p83698388549"></a><a name="p83698388549"></a>源操作数，shape为[B, S, H]，LocalTensor数据结构的定义请参考<a href="LocalTensor.md">LocalTensor</a>。inputDy的数据类型需要与目的操作数保持一致，尾轴长度需要32B对齐。</p>
<p id="p11688314152410"><a name="p11688314152410"></a><a name="p11688314152410"></a><span id="zh-cn_topic_0000002523303824_ph173308471594_3"><a name="zh-cn_topic_0000002523303824_ph173308471594_3"></a><a name="zh-cn_topic_0000002523303824_ph173308471594_3"></a><span id="zh-cn_topic_0000002523303824_ph9902231466_3"><a name="zh-cn_topic_0000002523303824_ph9902231466_3"></a><a name="zh-cn_topic_0000002523303824_ph9902231466_3"></a><span id="zh-cn_topic_0000002523303824_ph1782115034816_3"><a name="zh-cn_topic_0000002523303824_ph1782115034816_3"></a><a name="zh-cn_topic_0000002523303824_ph1782115034816_3"></a>类型为<a href="LocalTensor.md">LocalTensor</a>，支持的TPosition为VECIN/VECCALC/VECOUT。</span></span></span></p>
</td>
</tr>
<tr id="row10370172262718"><td class="cellrowborder" valign="top" width="21.08080808080808%" headers="mcps1.2.4.1.1 "><p id="p179192469188"><a name="p179192469188"></a><a name="p179192469188"></a>sharedTmpBuffer</p>
</td>
<td class="cellrowborder" valign="top" width="15.636363636363637%" headers="mcps1.2.4.1.2 "><p id="p209199466187"><a name="p209199466187"></a><a name="p209199466187"></a>输入</p>
</td>
<td class="cellrowborder" valign="top" width="63.28282828282828%" headers="mcps1.2.4.1.3 "><p id="p47801866195"><a name="p47801866195"></a><a name="p47801866195"></a>共享缓冲区，用于存放API内部计算产生的临时数据。该方式开发者可以自行管理sharedTmpBuffer内存空间，并在接口调用完成后，复用该部分内存，内存不会反复申请释放，灵活性较高，内存利用率也较高。共享缓冲区大小的获取方式请参考<a href="LayerNormGradBeta-Tiling.md">LayerNormGradBeta Tiling</a>。</p>
<p id="p167809613199"><a name="p167809613199"></a><a name="p167809613199"></a><span id="zh-cn_topic_0000002523303824_ph173308471594_4"><a name="zh-cn_topic_0000002523303824_ph173308471594_4"></a><a name="zh-cn_topic_0000002523303824_ph173308471594_4"></a><span id="zh-cn_topic_0000002523303824_ph9902231466_4"><a name="zh-cn_topic_0000002523303824_ph9902231466_4"></a><a name="zh-cn_topic_0000002523303824_ph9902231466_4"></a><span id="zh-cn_topic_0000002523303824_ph1782115034816_4"><a name="zh-cn_topic_0000002523303824_ph1782115034816_4"></a><a name="zh-cn_topic_0000002523303824_ph1782115034816_4"></a>类型为<a href="LocalTensor.md">LocalTensor</a>，支持的TPosition为VECIN/VECCALC/VECOUT。</span></span></span></p>
</td>
</tr>
<tr id="row9948161213491"><td class="cellrowborder" valign="top" width="21.08080808080808%" headers="mcps1.2.4.1.1 "><p id="p10948112144914"><a name="p10948112144914"></a><a name="p10948112144914"></a>tiling</p>
</td>
<td class="cellrowborder" valign="top" width="15.636363636363637%" headers="mcps1.2.4.1.2 "><p id="p1394861220490"><a name="p1394861220490"></a><a name="p1394861220490"></a>输入</p>
</td>
<td class="cellrowborder" valign="top" width="63.28282828282828%" headers="mcps1.2.4.1.3 "><p id="p186174619113"><a name="p186174619113"></a><a name="p186174619113"></a>LayerNormGradBeta计算所需Tiling信息，Tiling信息的获取请参考<a href="LayerNormGradBeta-Tiling.md">LayerNormGradBeta Tiling</a>。</p>
</td>
</tr>
</tbody>
</table>

## 返回值说明<a name="section5468191312484"></a>

无

## 约束说明<a name="section633mcpsimp"></a>

-   操作数地址对齐要求请参见[通用地址对齐约束](通用说明和约束.md#section796754519912)。
-   源操作数和目的操作数的Tensor空间可以复用。
-   仅支持输入shape为ND格式。
-   输入数据不满足对齐要求时，开发者需要进行补齐，补齐的数据应设置为0，防止出现异常值从而影响网络计算。
-   不支持对尾轴H轴的切分。

## 调用示例<a name="section642mcpsimp"></a>

完整的调用样例可参考[LayerNormGradBeta样例](https://gitcode.com/cann/asc-devkit/tree/master/examples/03_libraries/03_normalization/layernormgradbeta)。

```
// outputPdGamma: 输出对 gamma 参数的梯度，shape 为 [H]
// outputPdBeta: 输出对 beta 参数的梯度，shape 为 [H]
// resForGamma: 前一步 LayerNormGrad 输出的中间结果，即 normalizedX * inputDy，shape 为 [B, S, H]
// inputDy: 上游传入的梯度，shape 为 [B, S, H]
// tiling: Tiling 调度信息，包含并行划分、块大小等参数

// 使用 LayerNormGradBeta 接口计算 gamma 和 beta 的梯度
AscendC::LayerNormGradBeta<T, isReuseSource>(
    outputPdGamma,   // 输出：gamma 的梯度，shape [H]
    outputPdBeta,    // 输出：beta 的梯度，shape [H]
    resForGamma,     // 输入：中间结果 normalizedX * inputDy，来自 LayerNormGrad
    inputDy,         // 输入：上游梯度 dy，shape [B, S, H]
    tiling           // 输入：Tiling信息
);
```

