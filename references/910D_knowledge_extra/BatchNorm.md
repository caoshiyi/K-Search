# BatchNorm<a name="ZH-CN_TOPIC_0000002523343998"></a>

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

BatchNorm是对于每一层的输入做规范化处理，使得每一层的分布尽可能的相同，从而加速训练过程和提高模型的泛化能力（有效减少梯度消失和梯度爆炸问题）。基本思想是对于每个batch中的样本，对其输入的每个特征在batch的维度上进行归一化。具体来说，对于输入特征x，BatchNorm的计算过程可以表示为：

1.  对输入特征x，在batch维度上计算均值μ和方差σ：

    <!-- img2text -->
$$\mu_{i}=\frac{1}{m}\sum_{j=1}^{m}x_{ji}$$
$$\sigma_{i}^{2}=\frac{1}{m}\sum_{j=1}^{m}(x_{ji}-\mu_{i})^{2}$$

2.  对于每个特征i，对输入特征x进行归一化：

    <!-- img2text -->
$$\hat{x}_i = \frac{x_i - \mu}{\sqrt{\sigma^2 + \epsilon}}$$

3.  对归一化后的特征进行缩放和平移：

    <!-- img2text -->
$$
\text{output} = \hat{x} \times \text{weight} + \text{bias}
$$

## 函数原型<a name="section620mcpsimp"></a>

-   接口框架申请临时空间

    ```
    template <typename T, bool isReuseSource = false, bool isBasicBlock = false>
    __aicore__ inline void BatchNorm(const LocalTensor<T>& output, const LocalTensor<T>& outputMean, const LocalTensor<T>& outputVariance, const LocalTensor<T>& inputX, const LocalTensor<T>& gamm, const LocalTensor<T>& beta, const T epsilon, BatchNormTiling& tiling)
    ```

-   通过sharedTmpBuffer入参传入临时空间

    ```
    template <typename T, bool isReuseSource = false, bool isBasicBlock = false>
    __aicore__ inline void BatchNorm(const LocalTensor<T>& output, const LocalTensor<T>& outputMean, const LocalTensor<T>& outputVariance, const LocalTensor<T>& inputX, const LocalTensor<T>& gamm, const LocalTensor<T>& beta, const LocalTensor<uint8_t>& sharedTmpBuffer, const T epsilon, BatchNormTiling& tiling)
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
<p id="p8700423123617"><a name="p8700423123617"></a><a name="p8700423123617"></a><span id="ph1168842372812"><a name="ph1168842372812"></a><a name="ph1168842372812"></a>Ascend 950PR/Ascend 950DT</span>，支持的数据类型为：half、float。</p>
</td>
</tr>
<tr id="row6356241194912"><td class="cellrowborder" valign="top" width="19.39%" headers="mcps1.2.3.1.1 "><p id="p143561041144915"><a name="p143561041144915"></a><a name="p143561041144915"></a>isReuseSource</p>
</td>
<td class="cellrowborder" valign="top" width="80.61%" headers="mcps1.2.3.1.2 "><p id="p173569413498"><a name="p173569413498"></a><a name="p173569413498"></a>是否允许修改源操作数。该参数预留，传入默认值false即可。</p>
</td>
</tr>
<tr id="row9756719122620"><td class="cellrowborder" valign="top" width="19.39%" headers="mcps1.2.3.1.1 "><p id="p1682112447268"><a name="p1682112447268"></a><a name="p1682112447268"></a>isBasicBlock</p>
</td>
<td class="cellrowborder" valign="top" width="80.61%" headers="mcps1.2.3.1.2 "><p id="p10701171810252"><a name="p10701171810252"></a><a name="p10701171810252"></a>inputX、output的shape信息和Tiling切分策略满足基本块要求的情况下，可以使能该参数用于提升性能，默认不使能。基本块要求如下：</p>
<a name="ul5701918142510"></a><a name="ul5701918142510"></a><ul id="ul5701918142510"><li>originB是8的倍数；</li><li>S*H是64的倍数，但小于2048。</li></ul>
</td>
</tr>
</tbody>
</table>

**表 2**  接口参数说明

<a name="zh-cn_topic_0235751031_table33761356"></a>
<table><thead align="left"><tr id="zh-cn_topic_0235751031_row27598891"><th class="cellrowborder" valign="top" width="17.77%" id="mcps1.2.4.1.1"><p id="zh-cn_topic_0235751031_p20917673"><a name="zh-cn_topic_0235751031_p20917673"></a><a name="zh-cn_topic_0235751031_p20917673"></a>参数名</p>
</th>
<th class="cellrowborder" valign="top" width="9.69%" id="mcps1.2.4.1.2"><p id="zh-cn_topic_0235751031_p16609919"><a name="zh-cn_topic_0235751031_p16609919"></a><a name="zh-cn_topic_0235751031_p16609919"></a>输入/输出</p>
</th>
<th class="cellrowborder" valign="top" width="72.54%" id="mcps1.2.4.1.3"><p id="zh-cn_topic_0235751031_p59995477"><a name="zh-cn_topic_0235751031_p59995477"></a><a name="zh-cn_topic_0235751031_p59995477"></a>描述</p>
</th>
</tr>
</thead>
<tbody><tr id="row42461942101815"><td class="cellrowborder" valign="top" width="17.77%" headers="mcps1.2.4.1.1 "><p id="p8796143175319"><a name="p8796143175319"></a><a name="p8796143175319"></a>output</p>
</td>
<td class="cellrowborder" valign="top" width="9.69%" headers="mcps1.2.4.1.2 "><p id="p186293346150"><a name="p186293346150"></a><a name="p186293346150"></a>输出</p>
</td>
<td class="cellrowborder" valign="top" width="72.54%" headers="mcps1.2.4.1.3 "><p id="p58791759185415"><a name="p58791759185415"></a><a name="p58791759185415"></a>目的操作数，shape为[B，S，H]。</p>
<p id="p54711353133317"><a name="p54711353133317"></a><a name="p54711353133317"></a><span id="zh-cn_topic_0000002523303824_ph173308471594"><a name="zh-cn_topic_0000002523303824_ph173308471594"></a><a name="zh-cn_topic_0000002523303824_ph173308471594"></a><span id="zh-cn_topic_0000002523303824_ph9902231466"><a name="zh-cn_topic_0000002523303824_ph9902231466"></a><a name="zh-cn_topic_0000002523303824_ph9902231466"></a><span id="zh-cn_topic_0000002523303824_ph1782115034816"><a name="zh-cn_topic_0000002523303824_ph1782115034816"></a><a name="zh-cn_topic_0000002523303824_ph1782115034816"></a>类型为<a href="LocalTensor.md">LocalTensor</a>，支持的TPosition为VECIN/VECCALC/VECOUT。</span></span></span></p>
</td>
</tr>
<tr id="row250163914199"><td class="cellrowborder" valign="top" width="17.77%" headers="mcps1.2.4.1.1 "><p id="p48809611553"><a name="p48809611553"></a><a name="p48809611553"></a>outputMean</p>
</td>
<td class="cellrowborder" valign="top" width="9.69%" headers="mcps1.2.4.1.2 "><p id="p1662903414157"><a name="p1662903414157"></a><a name="p1662903414157"></a>输出</p>
</td>
<td class="cellrowborder" valign="top" width="72.54%" headers="mcps1.2.4.1.3 "><p id="p154129195515"><a name="p154129195515"></a><a name="p154129195515"></a>均值，目的操作数，shape为[S，H]。</p>
<p id="p18117165933310"><a name="p18117165933310"></a><a name="p18117165933310"></a><span id="zh-cn_topic_0000002523303824_ph173308471594_1"><a name="zh-cn_topic_0000002523303824_ph173308471594_1"></a><a name="zh-cn_topic_0000002523303824_ph173308471594_1"></a><span id="zh-cn_topic_0000002523303824_ph9902231466_1"><a name="zh-cn_topic_0000002523303824_ph9902231466_1"></a><a name="zh-cn_topic_0000002523303824_ph9902231466_1"></a><span id="zh-cn_topic_0000002523303824_ph1782115034816_1"><a name="zh-cn_topic_0000002523303824_ph1782115034816_1"></a><a name="zh-cn_topic_0000002523303824_ph1782115034816_1"></a>类型为<a href="LocalTensor.md">LocalTensor</a>，支持的TPosition为VECIN/VECCALC/VECOUT。</span></span></span></p>
</td>
</tr>
<tr id="row041413392198"><td class="cellrowborder" valign="top" width="17.77%" headers="mcps1.2.4.1.1 "><p id="p4630634141515"><a name="p4630634141515"></a><a name="p4630634141515"></a>outputVariance</p>
</td>
<td class="cellrowborder" valign="top" width="9.69%" headers="mcps1.2.4.1.2 "><p id="p263018345154"><a name="p263018345154"></a><a name="p263018345154"></a>输出</p>
</td>
<td class="cellrowborder" valign="top" width="72.54%" headers="mcps1.2.4.1.3 "><p id="p16656103755517"><a name="p16656103755517"></a><a name="p16656103755517"></a>方差，目的操作数，shape为[S，H]。</p>
<p id="p11269101103412"><a name="p11269101103412"></a><a name="p11269101103412"></a><span id="zh-cn_topic_0000002523303824_ph173308471594_2"><a name="zh-cn_topic_0000002523303824_ph173308471594_2"></a><a name="zh-cn_topic_0000002523303824_ph173308471594_2"></a><span id="zh-cn_topic_0000002523303824_ph9902231466_2"><a name="zh-cn_topic_0000002523303824_ph9902231466_2"></a><a name="zh-cn_topic_0000002523303824_ph9902231466_2"></a><span id="zh-cn_topic_0000002523303824_ph1782115034816_2"><a name="zh-cn_topic_0000002523303824_ph1782115034816_2"></a><a name="zh-cn_topic_0000002523303824_ph1782115034816_2"></a>类型为<a href="LocalTensor.md">LocalTensor</a>，支持的TPosition为VECIN/VECCALC/VECOUT。</span></span></span></p>
</td>
</tr>
<tr id="row189131716562"><td class="cellrowborder" valign="top" width="17.77%" headers="mcps1.2.4.1.1 "><p id="p14803121835620"><a name="p14803121835620"></a><a name="p14803121835620"></a>inputX</p>
</td>
<td class="cellrowborder" valign="top" width="9.69%" headers="mcps1.2.4.1.2 "><p id="p9961712566"><a name="p9961712566"></a><a name="p9961712566"></a>输入</p>
</td>
<td class="cellrowborder" valign="top" width="72.54%" headers="mcps1.2.4.1.3 "><p id="p13234103535610"><a name="p13234103535610"></a><a name="p13234103535610"></a>源操作数，shape为[B，S，H]。inputX的数据类型需要与目的操作数保持一致，S*H需要32B对齐。支持inputX与output地址重叠。</p>
<p id="p1129114103414"><a name="p1129114103414"></a><a name="p1129114103414"></a><span id="zh-cn_topic_0000002523303824_ph173308471594_3"><a name="zh-cn_topic_0000002523303824_ph173308471594_3"></a><a name="zh-cn_topic_0000002523303824_ph173308471594_3"></a><span id="zh-cn_topic_0000002523303824_ph9902231466_3"><a name="zh-cn_topic_0000002523303824_ph9902231466_3"></a><a name="zh-cn_topic_0000002523303824_ph9902231466_3"></a><span id="zh-cn_topic_0000002523303824_ph1782115034816_3"><a name="zh-cn_topic_0000002523303824_ph1782115034816_3"></a><a name="zh-cn_topic_0000002523303824_ph1782115034816_3"></a>类型为<a href="LocalTensor.md">LocalTensor</a>，支持的TPosition为VECIN/VECCALC/VECOUT。</span></span></span></p>
</td>
</tr>
<tr id="row927571485616"><td class="cellrowborder" valign="top" width="17.77%" headers="mcps1.2.4.1.1 "><p id="p1727531425614"><a name="p1727531425614"></a><a name="p1727531425614"></a>gamm</p>
</td>
<td class="cellrowborder" valign="top" width="9.69%" headers="mcps1.2.4.1.2 "><p id="p1927518141563"><a name="p1927518141563"></a><a name="p1927518141563"></a>输入</p>
</td>
<td class="cellrowborder" valign="top" width="72.54%" headers="mcps1.2.4.1.3 "><p id="p12048552562"><a name="p12048552562"></a><a name="p12048552562"></a>源操作数，shape为[B]。gamm的数据类型需要与目的操作数保持一致，长度需要32B对齐。</p>
<p id="p464020615340"><a name="p464020615340"></a><a name="p464020615340"></a><span id="zh-cn_topic_0000002523303824_ph173308471594_4"><a name="zh-cn_topic_0000002523303824_ph173308471594_4"></a><a name="zh-cn_topic_0000002523303824_ph173308471594_4"></a><span id="zh-cn_topic_0000002523303824_ph9902231466_4"><a name="zh-cn_topic_0000002523303824_ph9902231466_4"></a><a name="zh-cn_topic_0000002523303824_ph9902231466_4"></a><span id="zh-cn_topic_0000002523303824_ph1782115034816_4"><a name="zh-cn_topic_0000002523303824_ph1782115034816_4"></a><a name="zh-cn_topic_0000002523303824_ph1782115034816_4"></a>类型为<a href="LocalTensor.md">LocalTensor</a>，支持的TPosition为VECIN/VECCALC/VECOUT。</span></span></span></p>
</td>
</tr>
<tr id="row14907103561"><td class="cellrowborder" valign="top" width="17.77%" headers="mcps1.2.4.1.1 "><p id="p12490410195613"><a name="p12490410195613"></a><a name="p12490410195613"></a>beta</p>
</td>
<td class="cellrowborder" valign="top" width="9.69%" headers="mcps1.2.4.1.2 "><p id="p3490610195612"><a name="p3490610195612"></a><a name="p3490610195612"></a>输入</p>
</td>
<td class="cellrowborder" valign="top" width="72.54%" headers="mcps1.2.4.1.3 "><p id="p560622519572"><a name="p560622519572"></a><a name="p560622519572"></a>源操作数，shape为[B]。beta的数据类型需要与目的操作数保持一致，长度需要32B对齐。</p>
<p id="p1814171114347"><a name="p1814171114347"></a><a name="p1814171114347"></a><span id="zh-cn_topic_0000002523303824_ph173308471594_5"><a name="zh-cn_topic_0000002523303824_ph173308471594_5"></a><a name="zh-cn_topic_0000002523303824_ph173308471594_5"></a><span id="zh-cn_topic_0000002523303824_ph9902231466_5"><a name="zh-cn_topic_0000002523303824_ph9902231466_5"></a><a name="zh-cn_topic_0000002523303824_ph9902231466_5"></a><span id="zh-cn_topic_0000002523303824_ph1782115034816_5"><a name="zh-cn_topic_0000002523303824_ph1782115034816_5"></a><a name="zh-cn_topic_0000002523303824_ph1782115034816_5"></a>类型为<a href="LocalTensor.md">LocalTensor</a>，支持的TPosition为VECIN/VECCALC/VECOUT。</span></span></span></p>
</td>
</tr>
<tr id="row1421513531574"><td class="cellrowborder" valign="top" width="17.77%" headers="mcps1.2.4.1.1 "><p id="p102155531574"><a name="p102155531574"></a><a name="p102155531574"></a>sharedTmpBuffer</p>
</td>
<td class="cellrowborder" valign="top" width="9.69%" headers="mcps1.2.4.1.2 "><p id="p5654083582"><a name="p5654083582"></a><a name="p5654083582"></a>输入</p>
</td>
<td class="cellrowborder" valign="top" width="72.54%" headers="mcps1.2.4.1.3 "><p id="p104071111204211"><a name="p104071111204211"></a><a name="p104071111204211"></a>接口内部复杂计算时用于存储中间变量，由开发者提供。</p>
<p id="p5983205412394"><a name="p5983205412394"></a><a name="p5983205412394"></a><span id="zh-cn_topic_0000002523303824_ph173308471594_6"><a name="zh-cn_topic_0000002523303824_ph173308471594_6"></a><a name="zh-cn_topic_0000002523303824_ph173308471594_6"></a><span id="zh-cn_topic_0000002523303824_ph9902231466_6"><a name="zh-cn_topic_0000002523303824_ph9902231466_6"></a><a name="zh-cn_topic_0000002523303824_ph9902231466_6"></a><span id="zh-cn_topic_0000002523303824_ph1782115034816_6"><a name="zh-cn_topic_0000002523303824_ph1782115034816_6"></a><a name="zh-cn_topic_0000002523303824_ph1782115034816_6"></a>类型为<a href="LocalTensor.md">LocalTensor</a>，支持的TPosition为VECIN/VECCALC/VECOUT。</span></span></span></p>
<p id="p5881016172817"><a name="p5881016172817"></a><a name="p5881016172817"></a>临时空间大小BufferSize的获取方式请参考<a href="BatchNorm-Tiling.md">BatchNorm Tiling</a>。</p>
</td>
</tr>
<tr id="row88897586577"><td class="cellrowborder" valign="top" width="17.77%" headers="mcps1.2.4.1.1 "><p id="p18889155835712"><a name="p18889155835712"></a><a name="p18889155835712"></a>epsilon</p>
</td>
<td class="cellrowborder" valign="top" width="9.69%" headers="mcps1.2.4.1.2 "><p id="p108890588578"><a name="p108890588578"></a><a name="p108890588578"></a>输入</p>
</td>
<td class="cellrowborder" valign="top" width="72.54%" headers="mcps1.2.4.1.3 "><p id="p854221075919"><a name="p854221075919"></a><a name="p854221075919"></a>防除0的权重系数。数据类型需要与inputX/output保持一致。</p>
</td>
</tr>
<tr id="row188735226595"><td class="cellrowborder" valign="top" width="17.77%" headers="mcps1.2.4.1.1 "><p id="p0277924155919"><a name="p0277924155919"></a><a name="p0277924155919"></a>tiling</p>
</td>
<td class="cellrowborder" valign="top" width="9.69%" headers="mcps1.2.4.1.2 "><p id="p78736227591"><a name="p78736227591"></a><a name="p78736227591"></a>输入</p>
</td>
<td class="cellrowborder" valign="top" width="72.54%" headers="mcps1.2.4.1.3 "><p id="p133169331592"><a name="p133169331592"></a><a name="p133169331592"></a>输入数据的切分信息，Tiling信息的获取请参考<a href="BatchNorm-Tiling.md">BatchNorm Tiling</a>。</p>
</td>
</tr>
</tbody>
</table>

## 返回值说明<a name="section640mcpsimp"></a>

无

## 约束说明<a name="section633mcpsimp"></a>

-   操作数地址对齐要求请参见[通用地址对齐约束](通用说明和约束.md#section796754519912)。
-   当前仅支持ND格式的输入，不支持其他格式。
-   输入数据的S\*H必须满足32B对齐的要求。

## 调用示例<a name="section94691236101419"></a>

完整的调用样例可参考[BatchNorm样例](https://gitcode.com/cann/asc-devkit/tree/master/examples/03_libraries/03_normalization/batchnorm)。

```
// outputLocal：存放BatchNorm计算结果的Tensor
// meanLocal：存放计算结果均值的Tensor
// varianceLocal：存放计算结果方差的Tensor
// inputXLocal：参与计算的输入Tensor
// gammaLocal：输入张量，归一化后数据的缩放系数γ
// betaLocal：输入张量，归一化后数据的平移系数β
// epsilon：防除0的权重系数ε
// batchNormTiling：Tiling数据，从Host侧获取

AscendC::BatchNorm<dataType, isReuseSource, isBasicBlock>(outputLocal, meanLocal,varianceLocal, 
                                                          inputXLocal, gammaLocal, betaLocal, 
                                                          (dataType)epsilon, batchNormTilling); 
```

