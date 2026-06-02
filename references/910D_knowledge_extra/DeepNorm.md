# DeepNorm<a name="ZH-CN_TOPIC_0000002554343761"></a>

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

在深层神经网络训练过程中，执行层LayerNorm归一化时，可以使用DeepNorm进行替代，通过扩大残差连接来提高Transformer的稳定性。

本接口实现了对shape大小为\[B，S，H\]的输入数据的DeepNorm归一化，其计算公式如下：

DeepNorm\(x\) = LayerNorm\(α \* X + SubLayer\(X\)\)

SubLayer\(X\)通常是指在DeepNorm模型中的一个子层（sub-layer），用于实现自注意力机制（self-attention mechanism）。本接口中会整体作为一个输入Tensor传入。

其中LayerNorm的计算公式请参考[LayerNorm](LayerNorm.md#section618mcpsimp)。

## 函数原型<a name="section620mcpsimp"></a>

-   通过sharedTmpBuffer入参传入临时空间

    ```
    template <typename T, bool isReuseSrc = false, bool isBasicBlock = false>
    __aicore__ inline void DeepNorm(const LocalTensor<T>& dstLocal, const LocalTensor<T>& meanLocal, const LocalTensor<T>& rstdLocal, const LocalTensor<T>& srcLocal, const LocalTensor<T>& gxLocal, const LocalTensor<T>& betaLocal, const LocalTensor<T>& gammaLocal, const LocalTensor<uint8_t>& sharedTmpBuffer, const T alpha, const T epsilon, DeepNormTiling& tiling)
    ```

-   接口框架申请临时空间

    ```
    template <typename T, bool isReuseSrc = false, bool isBasicBlock = false>
    __aicore__ inline void DeepNorm(const LocalTensor<T>& dstLocal, const LocalTensor<T>& meanLocal, const LocalTensor<T>& rstdLocal, const LocalTensor<T>& srcLocal, const LocalTensor<T>& gxLocal, const LocalTensor<T>& betaLocal, const LocalTensor<T>& gammaLocal, const T alpha, const T epsilon, DeepNormTiling& tiling)
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
<p id="p565184052211"><a name="p565184052211"></a><a name="p565184052211"></a><span id="ph1168842372812"><a name="ph1168842372812"></a><a name="ph1168842372812"></a>Ascend 950PR/Ascend 950DT</span>，支持的数据类型为：half、float。</p>
</td>
</tr>
<tr id="row6356241194912"><td class="cellrowborder" valign="top" width="19.39%" headers="mcps1.2.3.1.1 "><p id="p143561041144915"><a name="p143561041144915"></a><a name="p143561041144915"></a>isReuseSrc</p>
</td>
<td class="cellrowborder" valign="top" width="80.61%" headers="mcps1.2.3.1.2 "><p id="p10298195117267"><a name="p10298195117267"></a><a name="p10298195117267"></a>是否允许修改源操作数，默认值为false。如果开发者允许源操作数被改写，可以使能该参数，使能后能够节省部分内存空间。</p>
<p id="p42981551112611"><a name="p42981551112611"></a><a name="p42981551112611"></a>设置为<strong id="b16299145110268"><a name="b16299145110268"></a><a name="b16299145110268"></a>true</strong>，则本接口内部计算时<strong id="b1929920514264"><a name="b1929920514264"></a><a name="b1929920514264"></a>复用</strong>srcLocal的内存空间，节省内存空间；设置为<strong id="b2299145172612"><a name="b2299145172612"></a><a name="b2299145172612"></a>false</strong>，则本接口内部计算时<strong id="b729905182619"><a name="b729905182619"></a><a name="b729905182619"></a>不复用</strong>srcLocal的内存空间。</p>
<p id="p4299155115268"><a name="p4299155115268"></a><a name="p4299155115268"></a>对于float数据类型输入支持开启该参数，half数据类型输入不支持开启该参数。</p>
<p id="p62891018544"><a name="p62891018544"></a><a name="p62891018544"></a>isReuseSrc的使用样例请参考<a href="更多样例-104.md#section639165323915">更多样例</a>。</p>
</td>
</tr>
<tr id="row9756719122620"><td class="cellrowborder" valign="top" width="19.39%" headers="mcps1.2.3.1.1 "><p id="p1682112447268"><a name="p1682112447268"></a><a name="p1682112447268"></a>isBasicBlock</p>
</td>
<td class="cellrowborder" valign="top" width="80.61%" headers="mcps1.2.3.1.2 "><p id="p361414011277"><a name="p361414011277"></a><a name="p361414011277"></a>srcTensor的shape信息满足基本块要求的情况下，可以使能该参数用于提升性能，默认不使能。基本块要求srcTensor的shape需要满足如下条件：</p>
<a name="ul76141604271"></a><a name="ul76141604271"></a><ul id="ul76141604271"><li>尾轴即H的长度为64的倍数，但不超过2040；</li><li>非尾轴长度（B*S）为8的倍数。</li></ul>
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
<tbody><tr id="row42461942101815"><td class="cellrowborder" valign="top" width="17.77%" headers="mcps1.2.4.1.1 "><p id="p462911347151"><a name="p462911347151"></a><a name="p462911347151"></a>dstLocal</p>
</td>
<td class="cellrowborder" valign="top" width="9.69%" headers="mcps1.2.4.1.2 "><p id="p186293346150"><a name="p186293346150"></a><a name="p186293346150"></a>输出</p>
</td>
<td class="cellrowborder" valign="top" width="72.54%" headers="mcps1.2.4.1.3 "><p id="p9255193274511"><a name="p9255193274511"></a><a name="p9255193274511"></a>目的操作数。shape为[B，S，H]。H长度不可超过2040。</p>
<p id="p54711353133317"><a name="p54711353133317"></a><a name="p54711353133317"></a><span id="zh-cn_topic_0000002523303824_ph173308471594"><a name="zh-cn_topic_0000002523303824_ph173308471594"></a><a name="zh-cn_topic_0000002523303824_ph173308471594"></a><span id="zh-cn_topic_0000002523303824_ph9902231466"><a name="zh-cn_topic_0000002523303824_ph9902231466"></a><a name="zh-cn_topic_0000002523303824_ph9902231466"></a><span id="zh-cn_topic_0000002523303824_ph1782115034816"><a name="zh-cn_topic_0000002523303824_ph1782115034816"></a><a name="zh-cn_topic_0000002523303824_ph1782115034816"></a>类型为<a href="LocalTensor.md">LocalTensor</a>，支持的TPosition为VECIN/VECCALC/VECOUT。</span></span></span></p>
</td>
</tr>
<tr id="row250163914199"><td class="cellrowborder" valign="top" width="17.77%" headers="mcps1.2.4.1.1 "><p id="p1762920347151"><a name="p1762920347151"></a><a name="p1762920347151"></a>meanLocal</p>
</td>
<td class="cellrowborder" valign="top" width="9.69%" headers="mcps1.2.4.1.2 "><p id="p1662903414157"><a name="p1662903414157"></a><a name="p1662903414157"></a>输出</p>
</td>
<td class="cellrowborder" valign="top" width="72.54%" headers="mcps1.2.4.1.3 "><p id="p18754535134214"><a name="p18754535134214"></a><a name="p18754535134214"></a>均值，目的操作数。shape为[B，S]。meanLocal的数据类型需要与dstLocal保持一致。</p>
<p id="p4335147133511"><a name="p4335147133511"></a><a name="p4335147133511"></a><span id="zh-cn_topic_0000002523303824_ph173308471594_1"><a name="zh-cn_topic_0000002523303824_ph173308471594_1"></a><a name="zh-cn_topic_0000002523303824_ph173308471594_1"></a><span id="zh-cn_topic_0000002523303824_ph9902231466_1"><a name="zh-cn_topic_0000002523303824_ph9902231466_1"></a><a name="zh-cn_topic_0000002523303824_ph9902231466_1"></a><span id="zh-cn_topic_0000002523303824_ph1782115034816_1"><a name="zh-cn_topic_0000002523303824_ph1782115034816_1"></a><a name="zh-cn_topic_0000002523303824_ph1782115034816_1"></a>类型为<a href="LocalTensor.md">LocalTensor</a>，支持的TPosition为VECIN/VECCALC/VECOUT。</span></span></span></p>
</td>
</tr>
<tr id="row041413392198"><td class="cellrowborder" valign="top" width="17.77%" headers="mcps1.2.4.1.1 "><p id="p4630634141515"><a name="p4630634141515"></a><a name="p4630634141515"></a>rstdLocal</p>
</td>
<td class="cellrowborder" valign="top" width="9.69%" headers="mcps1.2.4.1.2 "><p id="p263018345154"><a name="p263018345154"></a><a name="p263018345154"></a>输出</p>
</td>
<td class="cellrowborder" valign="top" width="72.54%" headers="mcps1.2.4.1.3 "><p id="p4615439236"><a name="p4615439236"></a><a name="p4615439236"></a>方差，目的操作数。shape为[B，S]。rstdLocal的数据类型需要与dstLocal保持一致。</p>
<p id="p182141494354"><a name="p182141494354"></a><a name="p182141494354"></a><span id="zh-cn_topic_0000002523303824_ph173308471594_2"><a name="zh-cn_topic_0000002523303824_ph173308471594_2"></a><a name="zh-cn_topic_0000002523303824_ph173308471594_2"></a><span id="zh-cn_topic_0000002523303824_ph9902231466_2"><a name="zh-cn_topic_0000002523303824_ph9902231466_2"></a><a name="zh-cn_topic_0000002523303824_ph9902231466_2"></a><span id="zh-cn_topic_0000002523303824_ph1782115034816_2"><a name="zh-cn_topic_0000002523303824_ph1782115034816_2"></a><a name="zh-cn_topic_0000002523303824_ph1782115034816_2"></a>类型为<a href="LocalTensor.md">LocalTensor</a>，支持的TPosition为VECIN/VECCALC/VECOUT。</span></span></span></p>
</td>
</tr>
<tr id="row17371444131520"><td class="cellrowborder" valign="top" width="17.77%" headers="mcps1.2.4.1.1 "><p id="p1838644151511"><a name="p1838644151511"></a><a name="p1838644151511"></a>srcLocal</p>
</td>
<td class="cellrowborder" valign="top" width="9.69%" headers="mcps1.2.4.1.2 "><p id="p73844410158"><a name="p73844410158"></a><a name="p73844410158"></a>输入</p>
</td>
<td class="cellrowborder" valign="top" width="72.54%" headers="mcps1.2.4.1.3 "><p id="p1976952572418"><a name="p1976952572418"></a><a name="p1976952572418"></a>源操作数，shape为[B，S，H]。srcLocal的数据类型需要与目的操作数保持一致，尾轴长度需要32B对齐。H长度不可超过2040。</p>
<p id="p1194545119358"><a name="p1194545119358"></a><a name="p1194545119358"></a><span id="zh-cn_topic_0000002523303824_ph173308471594_3"><a name="zh-cn_topic_0000002523303824_ph173308471594_3"></a><a name="zh-cn_topic_0000002523303824_ph173308471594_3"></a><span id="zh-cn_topic_0000002523303824_ph9902231466_3"><a name="zh-cn_topic_0000002523303824_ph9902231466_3"></a><a name="zh-cn_topic_0000002523303824_ph9902231466_3"></a><span id="zh-cn_topic_0000002523303824_ph1782115034816_3"><a name="zh-cn_topic_0000002523303824_ph1782115034816_3"></a><a name="zh-cn_topic_0000002523303824_ph1782115034816_3"></a>类型为<a href="LocalTensor.md">LocalTensor</a>，支持的TPosition为VECIN/VECCALC/VECOUT。</span></span></span></p>
</td>
</tr>
<tr id="row1747514515411"><td class="cellrowborder" valign="top" width="17.77%" headers="mcps1.2.4.1.1 "><p id="p1468645141"><a name="p1468645141"></a><a name="p1468645141"></a>gxLocal</p>
</td>
<td class="cellrowborder" valign="top" width="9.69%" headers="mcps1.2.4.1.2 "><p id="p3700202216199"><a name="p3700202216199"></a><a name="p3700202216199"></a>输入</p>
</td>
<td class="cellrowborder" valign="top" width="72.54%" headers="mcps1.2.4.1.3 "><p id="p4783182292513"><a name="p4783182292513"></a><a name="p4783182292513"></a>源操作数，shape为[B，S，H]。gxLocal的数据类型需要与目的操作数保持一致，尾轴长度需要32B对齐。H长度不可超过2040。</p>
<p id="p671765333517"><a name="p671765333517"></a><a name="p671765333517"></a><span id="zh-cn_topic_0000002523303824_ph173308471594_4"><a name="zh-cn_topic_0000002523303824_ph173308471594_4"></a><a name="zh-cn_topic_0000002523303824_ph173308471594_4"></a><span id="zh-cn_topic_0000002523303824_ph9902231466_4"><a name="zh-cn_topic_0000002523303824_ph9902231466_4"></a><a name="zh-cn_topic_0000002523303824_ph9902231466_4"></a><span id="zh-cn_topic_0000002523303824_ph1782115034816_4"><a name="zh-cn_topic_0000002523303824_ph1782115034816_4"></a><a name="zh-cn_topic_0000002523303824_ph1782115034816_4"></a>类型为<a href="LocalTensor.md">LocalTensor</a>，支持的TPosition为VECIN/VECCALC/VECOUT。</span></span></span></p>
<p id="p1710941214403"><a name="p1710941214403"></a><a name="p1710941214403"></a>该参数对应计算公式中的SubLayer(X)的计算结果。</p>
</td>
</tr>
<tr id="row947414514416"><td class="cellrowborder" valign="top" width="17.77%" headers="mcps1.2.4.1.1 "><p id="p1446813458418"><a name="p1446813458418"></a><a name="p1446813458418"></a>betaLocal</p>
</td>
<td class="cellrowborder" valign="top" width="9.69%" headers="mcps1.2.4.1.2 "><p id="p54687451247"><a name="p54687451247"></a><a name="p54687451247"></a>输入</p>
</td>
<td class="cellrowborder" valign="top" width="72.54%" headers="mcps1.2.4.1.3 "><p id="p194686456420"><a name="p194686456420"></a><a name="p194686456420"></a>源操作数，shape为[H]。betaLocal的数据类型需要与目的操作数保持一致，长度需要32B对齐。H长度不可超过2040。</p>
<p id="p4395105919355"><a name="p4395105919355"></a><a name="p4395105919355"></a><span id="zh-cn_topic_0000002523303824_ph173308471594_5"><a name="zh-cn_topic_0000002523303824_ph173308471594_5"></a><a name="zh-cn_topic_0000002523303824_ph173308471594_5"></a><span id="zh-cn_topic_0000002523303824_ph9902231466_5"><a name="zh-cn_topic_0000002523303824_ph9902231466_5"></a><a name="zh-cn_topic_0000002523303824_ph9902231466_5"></a><span id="zh-cn_topic_0000002523303824_ph1782115034816_5"><a name="zh-cn_topic_0000002523303824_ph1782115034816_5"></a><a name="zh-cn_topic_0000002523303824_ph1782115034816_5"></a>类型为<a href="LocalTensor.md">LocalTensor</a>，支持的TPosition为VECIN/VECCALC/VECOUT。</span></span></span></p>
</td>
</tr>
<tr id="row134741451043"><td class="cellrowborder" valign="top" width="17.77%" headers="mcps1.2.4.1.1 "><p id="p64681445649"><a name="p64681445649"></a><a name="p64681445649"></a>gammaLocal</p>
</td>
<td class="cellrowborder" valign="top" width="9.69%" headers="mcps1.2.4.1.2 "><p id="p64685451843"><a name="p64685451843"></a><a name="p64685451843"></a>输入</p>
</td>
<td class="cellrowborder" valign="top" width="72.54%" headers="mcps1.2.4.1.3 "><p id="p86531230165413"><a name="p86531230165413"></a><a name="p86531230165413"></a>源操作数，shape为[H]。gammaLocal的数据类型需要与目的操作数保持一致，长度需要32B对齐。H长度不可超过2040。</p>
<p id="p37906093619"><a name="p37906093619"></a><a name="p37906093619"></a><span id="zh-cn_topic_0000002523303824_ph173308471594_6"><a name="zh-cn_topic_0000002523303824_ph173308471594_6"></a><a name="zh-cn_topic_0000002523303824_ph173308471594_6"></a><span id="zh-cn_topic_0000002523303824_ph9902231466_6"><a name="zh-cn_topic_0000002523303824_ph9902231466_6"></a><a name="zh-cn_topic_0000002523303824_ph9902231466_6"></a><span id="zh-cn_topic_0000002523303824_ph1782115034816_6"><a name="zh-cn_topic_0000002523303824_ph1782115034816_6"></a><a name="zh-cn_topic_0000002523303824_ph1782115034816_6"></a>类型为<a href="LocalTensor.md">LocalTensor</a>，支持的TPosition为VECIN/VECCALC/VECOUT。</span></span></span></p>
</td>
</tr>
<tr id="row1474145340"><td class="cellrowborder" valign="top" width="17.77%" headers="mcps1.2.4.1.1 "><p id="p144696451846"><a name="p144696451846"></a><a name="p144696451846"></a>sharedTmpBuffer</p>
</td>
<td class="cellrowborder" valign="top" width="9.69%" headers="mcps1.2.4.1.2 "><p id="p4469745640"><a name="p4469745640"></a><a name="p4469745640"></a>输入</p>
</td>
<td class="cellrowborder" valign="top" width="72.54%" headers="mcps1.2.4.1.3 "><p id="p155841325182310"><a name="p155841325182310"></a><a name="p155841325182310"></a>接口内部复杂计算时用于存储中间变量，由开发者提供。</p>
<p id="p1226915319367"><a name="p1226915319367"></a><a name="p1226915319367"></a><span id="zh-cn_topic_0000002523303824_ph173308471594_7"><a name="zh-cn_topic_0000002523303824_ph173308471594_7"></a><a name="zh-cn_topic_0000002523303824_ph173308471594_7"></a><span id="zh-cn_topic_0000002523303824_ph9902231466_7"><a name="zh-cn_topic_0000002523303824_ph9902231466_7"></a><a name="zh-cn_topic_0000002523303824_ph9902231466_7"></a><span id="zh-cn_topic_0000002523303824_ph1782115034816_7"><a name="zh-cn_topic_0000002523303824_ph1782115034816_7"></a><a name="zh-cn_topic_0000002523303824_ph1782115034816_7"></a>类型为<a href="LocalTensor.md">LocalTensor</a>，支持的TPosition为VECIN/VECCALC/VECOUT。</span></span></span></p>
<p id="p5881016172817"><a name="p5881016172817"></a><a name="p5881016172817"></a>临时空间大小BufferSize的获取方式请参考<a href="DeepNorm-Tiling.md">DeepNorm Tiling</a>。</p>
</td>
</tr>
<tr id="row1234414448518"><td class="cellrowborder" valign="top" width="17.77%" headers="mcps1.2.4.1.1 "><p id="p163371944652"><a name="p163371944652"></a><a name="p163371944652"></a>alpha</p>
</td>
<td class="cellrowborder" valign="top" width="9.69%" headers="mcps1.2.4.1.2 "><p id="p63851525101917"><a name="p63851525101917"></a><a name="p63851525101917"></a>输入</p>
</td>
<td class="cellrowborder" valign="top" width="72.54%" headers="mcps1.2.4.1.3 "><p id="p621422181413"><a name="p621422181413"></a><a name="p621422181413"></a>权重系数。数据类型需要与目的操作数一致。</p>
</td>
</tr>
<tr id="row10344744456"><td class="cellrowborder" valign="top" width="17.77%" headers="mcps1.2.4.1.1 "><p id="p23373441517"><a name="p23373441517"></a><a name="p23373441517"></a>epsilon</p>
</td>
<td class="cellrowborder" valign="top" width="9.69%" headers="mcps1.2.4.1.2 "><p id="p123370448517"><a name="p123370448517"></a><a name="p123370448517"></a>输入</p>
</td>
<td class="cellrowborder" valign="top" width="72.54%" headers="mcps1.2.4.1.3 "><p id="p13374936101316"><a name="p13374936101316"></a><a name="p13374936101316"></a>权重系数， 用来防止除零错误。数据类型需要与目的操作数一致。</p>
</td>
</tr>
<tr id="row1164114613511"><td class="cellrowborder" valign="top" width="17.77%" headers="mcps1.2.4.1.1 "><p id="p1636134614512"><a name="p1636134614512"></a><a name="p1636134614512"></a>tiling</p>
</td>
<td class="cellrowborder" valign="top" width="9.69%" headers="mcps1.2.4.1.2 "><p id="p10629126111914"><a name="p10629126111914"></a><a name="p10629126111914"></a>输入</p>
</td>
<td class="cellrowborder" valign="top" width="72.54%" headers="mcps1.2.4.1.3 "><p id="p729641616144"><a name="p729641616144"></a><a name="p729641616144"></a>DeepNorm计算所需Tiling信息，Tiling信息的获取请参考<a href="DeepNorm-Tiling.md">DeepNorm Tiling</a>。</p>
</td>
</tr>
</tbody>
</table>

## 返回值说明<a name="section640mcpsimp"></a>

无

## 约束说明<a name="section18375195021515"></a>

-   操作数地址对齐要求请参见[通用地址对齐约束](通用说明和约束.md#section796754519912)。

-   isReuseSrc模板参数为false时，srcLocal和dstLocal的Tensor空间不支持复用。
-   仅支持输入shape为ND格式。
-   输入数据不满足对齐要求时，开发者需要进行补齐，补齐的数据应设置为0，防止出现异常值从而影响网络计算。

## 调用示例<a name="section94691236101419"></a>

完整的调用样例可参考[DeepNorm样例](https://gitcode.com/cann/asc-devkit/tree/master/examples/03_libraries/03_normalization/deepnorm)。

```
// dstLocal: 存放 DeepNorm 计算结果的 Tensor
// meanLocal: 输出均值Tensor
// rstdLocal: 输出方差Tensor
// srcLocal: 输入的主数据 X，shape 为 [B, S, H]
// gxLocal: SubLayer(X) 的输出
// betaLocal: LayerNorm 的偏置系数β，shape 为 [H]
// gammaLocal: LayerNorm 的缩放系数γ，shape 为 [H]
// alpha: 残差连接的缩放系数α
// epsilon: 防除零系数ε
// tiling: Tiling 信息，包含维度、分块等参数

// 使用 DeepNorm 接口实现 DeepNorm(x) = LayerNorm(α * X + SubLayer(X))
// 若尾轴的长度（H）不超过2040且为64的倍数，同时非尾轴长度（B*S）为8的倍数，可设置isBasicBlock = true提升性能
// 若数据类型T为float且允许修改srcLocal，可设置isReuseSrc = true复用srcLocal内存空间以节省内存
AscendC::DeepNorm<T, isReuseSrc, isBasicBlock>(
    dstLocal,     // 输出：归一化后的结果
    meanLocal,    // 输出：均值 mean
    rstdLocal,    // 输出：倒数标准差 rstd
    srcLocal,     // 输入：原始输入 X
    gxLocal,      // 输入：子层输出 SubLayer(X)
    betaLocal,    // 输入：LayerNorm 偏置系数β
    gammaLocal,   // 输入：LayerNorm 缩放系数γ
    alpha,        // 输入：残差路径缩放系数 α
    epsilon,      // 输入：防除零系数 ε
    tiling        // 输入：Tiling 信息
);
```

