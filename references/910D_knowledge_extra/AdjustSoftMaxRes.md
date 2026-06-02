# AdjustSoftMaxRes<a name="ZH-CN_TOPIC_0000002523344552"></a>

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

本接口用于调整SoftMax的计算结果为指定的值。主要用于对SoftMax相关计算结果做后处理。当输入的max中存在指定的值的时候，会调整对应的softmaxres中的结果为输入的自定义的值。以上调整方式为按行进行，即当max某一行的值为某个值时，调整当前softmaxres对应一行的值都为输入的值。

为方便理解，通过Python脚本实现的方式，表达其计算公式如下，其中res是输入也是输出，max\\from\\to\\res\_shape都为输入。

```
def adjust_softmax_res(res, max, from, to, res_shape):
    for i in res_shape[0]:
        if max[i] == from:
            for j in res_shape[1]:
                res[i][j] = to
    return
```

## 函数原型<a name="section161381959151619"></a>

```
template <typename T1, typename T2, bool isDataFormatNZ = false, uint8_t stepSizeMode = 0>
__aicore__ inline bool AdjustSoftMaxRes(const LocalTensor<T1>& softMaxRes, const LocalTensor<T2>& maxTensor, const uint32_t from, const T1 to, const SoftMaxShapeInfo& softmaxShapeInfo)
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
<tbody><tr id="row14755141911264"><td class="cellrowborder" valign="top" width="19.39%" headers="mcps1.2.3.1.1 "><p id="p47551198266"><a name="p47551198266"></a><a name="p47551198266"></a>T1</p>
</td>
<td class="cellrowborder" valign="top" width="80.61%" headers="mcps1.2.3.1.2 "><p id="p125969172719"><a name="p125969172719"></a><a name="p125969172719"></a>softMaxRes的数据类型。</p>
<p id="p1853102419205"><a name="p1853102419205"></a><a name="p1853102419205"></a><span id="ph1949472422013"><a name="ph1949472422013"></a><a name="ph1949472422013"></a>Ascend 950PR/Ascend 950DT</span>，支持的数据类型为：half、float。</p>
</td>
</tr>
<tr id="row197562193260"><td class="cellrowborder" valign="top" width="19.39%" headers="mcps1.2.3.1.1 "><p id="p18961115352610"><a name="p18961115352610"></a><a name="p18961115352610"></a>T2</p>
</td>
<td class="cellrowborder" valign="top" width="80.61%" headers="mcps1.2.3.1.2 "><p id="p1287421219275"><a name="p1287421219275"></a><a name="p1287421219275"></a>maxTensor的数据类型。</p>
<p id="p393965992617"><a name="p393965992617"></a><a name="p393965992617"></a><span id="ph4939135942619"><a name="ph4939135942619"></a><a name="ph4939135942619"></a>Ascend 950PR/Ascend 950DT</span>，支持的数据类型为：half、float。</p>
</td>
</tr>
<tr id="row9756719122620"><td class="cellrowborder" valign="top" width="19.39%" headers="mcps1.2.3.1.1 "><p id="p1954365362720"><a name="p1954365362720"></a><a name="p1954365362720"></a>isDataFormatNZ</p>
</td>
<td class="cellrowborder" valign="top" width="80.61%" headers="mcps1.2.3.1.2 "><p id="p9584312102920"><a name="p9584312102920"></a><a name="p9584312102920"></a>当前输入输出的数据格式是否为NZ格式，默认数据格式为ND，即默认取值为false。</p>
</td>
</tr>
<tr id="row184847156200"><td class="cellrowborder" valign="top" width="19.39%" headers="mcps1.2.3.1.1 "><p id="p1485131512017"><a name="p1485131512017"></a><a name="p1485131512017"></a>stepSizeMode</p>
</td>
<td class="cellrowborder" valign="top" width="80.61%" headers="mcps1.2.3.1.2 "><p id="p2028895682611"><a name="p2028895682611"></a><a name="p2028895682611"></a>maxTensor取元素的步进长度的模式。参数取值如下：</p>
<a name="ul19418153132715"></a><a name="ul19418153132715"></a><ul id="ul19418153132715"><li>0：默认值，每个BlockSize（32字节）内，取第一个元素的数值与输入from的数值作对比。即，maxTensor的数据类型为float时，按照输入shape为(m, 8)的格式，每8个数取一个数，maxTensor的数据类型为half时，按照输入shape为(m, 16)的格式，每16个数取一个数。</li><li>非0：取maxTensor每个元素的数值与输入from的数值作对比。即，按照输入shape为(m, 1)的格式，每次取一个元素的数值与输入from的数值作对比。该参数取值非0时仅支持maxTensor为ND格式。</li></ul>
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
<tbody><tr id="row42461942101815"><td class="cellrowborder" valign="top" width="17.77%" headers="mcps1.2.4.1.1 "><p id="p132394662514"><a name="p132394662514"></a><a name="p132394662514"></a>softMaxRes</p>
</td>
<td class="cellrowborder" valign="top" width="8.07%" headers="mcps1.2.4.1.2 "><p id="p623976112513"><a name="p623976112513"></a><a name="p623976112513"></a>输入/输出</p>
</td>
<td class="cellrowborder" valign="top" width="74.16%" headers="mcps1.2.4.1.3 "><p id="p1262211583124"><a name="p1262211583124"></a><a name="p1262211583124"></a>既是源操作数也是目的操作数。</p>
<p id="p8534719139"><a name="p8534719139"></a><a name="p8534719139"></a><span id="zh-cn_topic_0000002523303824_ph173308471594"><a name="zh-cn_topic_0000002523303824_ph173308471594"></a><a name="zh-cn_topic_0000002523303824_ph173308471594"></a><span id="zh-cn_topic_0000002523303824_ph9902231466"><a name="zh-cn_topic_0000002523303824_ph9902231466"></a><a name="zh-cn_topic_0000002523303824_ph9902231466"></a><span id="zh-cn_topic_0000002523303824_ph1782115034816"><a name="zh-cn_topic_0000002523303824_ph1782115034816"></a><a name="zh-cn_topic_0000002523303824_ph1782115034816"></a>类型为<a href="LocalTensor.md">LocalTensor</a>，支持的TPosition为VECIN/VECCALC/VECOUT。</span></span></span></p>
<p id="p318727114519"><a name="p318727114519"></a><a name="p318727114519"></a>LocalTensor数据结构的定义请参考<a href="LocalTensor.md">LocalTensor</a></p>
<p id="p1143644911915"><a name="p1143644911915"></a><a name="p1143644911915"></a>last轴长度需要32Byte对齐。</p>
<p id="p789510116718"><a name="p789510116718"></a><a name="p789510116718"></a>一般为softmax计算的输出结果。</p>
</td>
</tr>
<tr id="row2137145181815"><td class="cellrowborder" valign="top" width="17.77%" headers="mcps1.2.4.1.1 "><p id="p1423915622510"><a name="p1423915622510"></a><a name="p1423915622510"></a>maxTensor</p>
</td>
<td class="cellrowborder" valign="top" width="8.07%" headers="mcps1.2.4.1.2 "><p id="p1723911613251"><a name="p1723911613251"></a><a name="p1723911613251"></a>输入</p>
</td>
<td class="cellrowborder" valign="top" width="74.16%" headers="mcps1.2.4.1.3 "><p id="p14929162874710"><a name="p14929162874710"></a><a name="p14929162874710"></a>源操作数。</p>
<p id="p106669246207"><a name="p106669246207"></a><a name="p106669246207"></a><span id="zh-cn_topic_0000002523303824_ph173308471594_1"><a name="zh-cn_topic_0000002523303824_ph173308471594_1"></a><a name="zh-cn_topic_0000002523303824_ph173308471594_1"></a><span id="zh-cn_topic_0000002523303824_ph9902231466_1"><a name="zh-cn_topic_0000002523303824_ph9902231466_1"></a><a name="zh-cn_topic_0000002523303824_ph9902231466_1"></a><span id="zh-cn_topic_0000002523303824_ph1782115034816_1"><a name="zh-cn_topic_0000002523303824_ph1782115034816_1"></a><a name="zh-cn_topic_0000002523303824_ph1782115034816_1"></a>类型为<a href="LocalTensor.md">LocalTensor</a>，支持的TPosition为VECIN/VECCALC/VECOUT。</span></span></span></p>
<p id="p296320351261"><a name="p296320351261"></a><a name="p296320351261"></a>softmax计算过程中reducemax的结果。</p>
<a name="ul9474401463"></a><a name="ul9474401463"></a><ul id="ul9474401463"><li>maxTensor的last轴长度固定为32Byte，即一个<span id="ph885316116307"><a name="ph885316116307"></a><a name="ph885316116307"></a>datablock</span>长度。该<span id="ph1180815155353"><a name="ph1180815155353"></a><a name="ph1180815155353"></a>datablock</span>中的所有数据为同一个值。比如half数据类型下，该<span id="ph18505176353"><a name="ph18505176353"></a><a name="ph18505176353"></a>datablock</span>中的16个数均为相同的reducemax的值。</li><li>非last轴的长度与softMaxRes保持一致。</li></ul>
</td>
</tr>
<tr id="row19615183817191"><td class="cellrowborder" valign="top" width="17.77%" headers="mcps1.2.4.1.1 "><p id="p19239062256"><a name="p19239062256"></a><a name="p19239062256"></a>from</p>
</td>
<td class="cellrowborder" valign="top" width="8.07%" headers="mcps1.2.4.1.2 "><p id="p52399642514"><a name="p52399642514"></a><a name="p52399642514"></a>输入</p>
</td>
<td class="cellrowborder" valign="top" width="74.16%" headers="mcps1.2.4.1.3 "><p id="p1965113819615"><a name="p1965113819615"></a><a name="p1965113819615"></a>源操作数，类型为uint32_t。</p>
<p id="p26517382613"><a name="p26517382613"></a><a name="p26517382613"></a>需要判断的maxTensor中的值。需要注意的是，由于maxTensor中的值均为浮点数类型，因此此处需要填入的值为浮点数类型对应十六进制的值。比如当需要判断maxTensor是否有1.0这个值时，from值需要填入1.0对应的十六进制值0x3f800000。</p>
</td>
</tr>
<tr id="row9520125782015"><td class="cellrowborder" valign="top" width="17.77%" headers="mcps1.2.4.1.1 "><p id="p1358462510237"><a name="p1358462510237"></a><a name="p1358462510237"></a>to</p>
</td>
<td class="cellrowborder" valign="top" width="8.07%" headers="mcps1.2.4.1.2 "><p id="p15584152512313"><a name="p15584152512313"></a><a name="p15584152512313"></a>输入</p>
</td>
<td class="cellrowborder" valign="top" width="74.16%" headers="mcps1.2.4.1.3 "><p id="p155841325182310"><a name="p155841325182310"></a><a name="p155841325182310"></a>源操作数，类型和softMaxRes的数据类型保持一致。</p>
<p id="p104071111204211"><a name="p104071111204211"></a><a name="p104071111204211"></a>需要往softMaxRes中填充的值。</p>
</td>
</tr>
<tr id="row138606561892"><td class="cellrowborder" valign="top" width="17.77%" headers="mcps1.2.4.1.1 "><p id="p0734515621"><a name="p0734515621"></a><a name="p0734515621"></a>softmaxShapeInfo</p>
</td>
<td class="cellrowborder" valign="top" width="8.07%" headers="mcps1.2.4.1.2 "><p id="p1973441513219"><a name="p1973441513219"></a><a name="p1973441513219"></a>输入</p>
</td>
<td class="cellrowborder" valign="top" width="74.16%" headers="mcps1.2.4.1.3 "><p id="p1573415151214"><a name="p1573415151214"></a><a name="p1573415151214"></a>softMaxRes的shape信息，结构定义如下：</p>
<a name="screen587041519381"></a><a name="screen587041519381"></a><pre class="screen" codetype="Cpp" id="screen587041519381">struct SoftMaxShapeInfo {
    uint32_t srcM; // 非尾轴乘积长度
    uint32_t srcK; // 尾轴长度，必须32Byte对齐
    uint32_t oriSrcM; // 原始非尾轴乘积长度
    uint32_t oriSrcK;  // 原始尾轴长度
};</pre>
<p id="p106970420229"><a name="p106970420229"></a><a name="p106970420229"></a>需要注意，目前仅支持ND输入。</p>
</td>
</tr>
</tbody>
</table>

## 返回值说明<a name="section640mcpsimp"></a>

bool类型，当返回true时，表示maxTensor中存在需要判断的值，若返回false，则表示maxTensor中不存在需要判断的值。

## 约束说明<a name="section633mcpsimp"></a>

-   操作数地址对齐要求请参见[通用地址对齐约束](通用说明和约束.md#section796754519912)。

## 调用示例<a name="section94691236101419"></a>

本样例中需要对SoftMax计算结果做后处理，判断maxTensor中是否存在0xFF7FFFFF，如果存在刷新对应结果为0。本样例中实现的是固定shape为输入x\[32, 32\]，输出y\[32, 32\]的AdjustSoftMaxResCustom算子。输入softMaxRes的shape大小为\[32,32\]，maxTensor的shape大小为\[32,8\]，数据类型均为float。

完整的调用样例可参考[AdjustSoftMaxRes样例](https://gitcode.com/cann/asc-devkit/tree/master/examples/03_libraries/02_activation/adjustsoftmaxres)。

```
// srcLocal：softmax计算结果
AscendC::SoftMax(srcLocal, ...)
// maxLocal：softmax中间结果，reducemax的结果
// FROM： 判断maxLocal中是否存在值等于FROM的元素
// TO： 当maxLocal中存在值等于FROM的元素时，srcLocal中对应行的元素将被替换为TO
// srcShape：描述srcLocal的shape信息

AscendC::SoftMaxShapeInfo srcShape = {height, width, height, width};
AscendC::AdjustSoftMaxRes<float, float>(srcLocal, maxLocal, FROM, TO, srcShape);
```

