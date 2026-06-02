# Cast<a name="ZH-CN_TOPIC_0000002554343559"></a>

## 产品支持情况<a name="section1550532418810"></a>

<a name="table38301303189"></a>
<table><thead align="left"><tr id="row20831180131817"><th class="cellrowborder" valign="top" width="57.99999999999999%" id="mcps1.1.3.1.1"><p id="p1883113061818"><a name="p1883113061818"></a><a name="p1883113061818"></a><span id="ph20833205312295"><a name="ph20833205312295"></a><a name="ph20833205312295"></a>产品</span></p>
</th>
<th class="cellrowborder" align="center" valign="top" width="42%" id="mcps1.1.3.1.2"><p id="p783113012187"><a name="p783113012187"></a><a name="p783113012187"></a>是否支持</p>
</th>
</tr>
</thead>
<tbody><tr id="row1272474920205"><td class="cellrowborder" valign="top" width="57.99999999999999%" headers="mcps1.1.3.1.1 "><p id="p1622020982912"><a name="p1622020982912"></a><a name="p1622020982912"></a><span id="ph1522010992915"><a name="ph1522010992915"></a><a name="ph1522010992915"></a>Ascend 950PR/Ascend 950DT</span></p>
</td>
<td class="cellrowborder" align="center" valign="top" width="42%" headers="mcps1.1.3.1.2 "><p id="p37256491200"><a name="p37256491200"></a><a name="p37256491200"></a>√</p>
</td>
</tr>
</tbody>
</table>

## 功能说明<a name="section618mcpsimp"></a>

数据类型精度转换，将U类型的源操作数根据指定的转换模式，转换成T类型的目的操作数类型。

## 函数原型<a name="section620mcpsimp"></a>

```
template <typename T = DefaultType, typename U = DefaultType, const CastTrait& trait = castTrait, typename S, typename V>
__simd_callee__ inline void Cast(S& dstReg, V& srcReg, MaskReg& mask)
```

## 参数说明<a name="section622mcpsimp"></a>

**表 1**  模板参数说明

<a name="table4835205712588"></a>
<table><thead align="left"><tr id="row118356578583"><th class="cellrowborder" valign="top" width="18.58%" id="mcps1.2.3.1.1"><p id="p48354572582"><a name="p48354572582"></a><a name="p48354572582"></a>参数名</p>
</th>
<th class="cellrowborder" valign="top" width="81.42%" id="mcps1.2.3.1.2"><p id="p583535795817"><a name="p583535795817"></a><a name="p583535795817"></a>描述</p>
</th>
</tr>
</thead>
<tbody><tr id="row1835857145817"><td class="cellrowborder" valign="top" width="18.58%" headers="mcps1.2.3.1.1 "><p id="p5835457165816"><a name="p5835457165816"></a><a name="p5835457165816"></a>T</p>
</td>
<td class="cellrowborder" valign="top" width="81.42%" headers="mcps1.2.3.1.2 "><p id="p168351657155818"><a name="p168351657155818"></a><a name="p168351657155818"></a>目的操作数数据类型。</p>
</td>
</tr>
<tr id="row1548420123810"><td class="cellrowborder" valign="top" width="18.58%" headers="mcps1.2.3.1.1 "><p id="p04841114381"><a name="p04841114381"></a><a name="p04841114381"></a>U</p>
</td>
<td class="cellrowborder" valign="top" width="81.42%" headers="mcps1.2.3.1.2 "><p id="p19484151193812"><a name="p19484151193812"></a><a name="p19484151193812"></a>源操作数数据类型。</p>
</td>
</tr>
<tr id="row9924958115617"><td class="cellrowborder" valign="top" width="18.58%" headers="mcps1.2.3.1.1 "><p id="p3925158115615"><a name="p3925158115615"></a><a name="p3925158115615"></a>trait</p>
</td>
<td class="cellrowborder" valign="top" width="81.42%" headers="mcps1.2.3.1.2 "><p id="p129251358135613"><a name="p129251358135613"></a><a name="p129251358135613"></a>类型转换模式结构体。</p>
<p id="p1337682765714"><a name="p1337682765714"></a><a name="p1337682765714"></a>包括<a href="RegLayout.md">RegLayout</a>、<a href="SatMode.md">SatMode</a>、<a href="MaskMergeMode.md">MaskMergeMode</a>、<a href="RoundMode.md">RoundMode</a>。</p>
<p id="p173031248202111"><a name="p173031248202111"></a><a name="p173031248202111"></a>使能SatMode生效需与<a href="SetCtrlSpr(ISASI).md">SetCtrlSpr(ISASI)</a>配合使用。</p>
</td>
</tr>
<tr id="row13956192410456"><td class="cellrowborder" valign="top" width="18.58%" headers="mcps1.2.3.1.1 "><p id="p1897872818010"><a name="p1897872818010"></a><a name="p1897872818010"></a>S</p>
</td>
<td class="cellrowborder" valign="top" width="81.42%" headers="mcps1.2.3.1.2 "><p id="p497717282014"><a name="p497717282014"></a><a name="p497717282014"></a>srcReg类型，例如RegTensor&lt;float&gt;，由编译器自动推导，用户不需要填写。</p>
</td>
</tr>
<tr id="row3425134574515"><td class="cellrowborder" valign="top" width="18.58%" headers="mcps1.2.3.1.1 "><p id="p17671757019"><a name="p17671757019"></a><a name="p17671757019"></a>V</p>
</td>
<td class="cellrowborder" valign="top" width="81.42%" headers="mcps1.2.3.1.2 "><p id="p20766185501"><a name="p20766185501"></a><a name="p20766185501"></a>dstReg类型，例如RegTensor&lt;int32_t&gt;，由编译器自动推导，用户不需要填写。</p>
</td>
</tr>
</tbody>
</table>

**表 2**  参数说明

<a name="zh-cn_topic_0235751031_table33761356"></a>
<table><thead align="left"><tr id="zh-cn_topic_0235751031_row27598891"><th class="cellrowborder" valign="top" width="18.54%" id="mcps1.2.4.1.1"><p id="zh-cn_topic_0235751031_p20917673"><a name="zh-cn_topic_0235751031_p20917673"></a><a name="zh-cn_topic_0235751031_p20917673"></a>参数名</p>
</th>
<th class="cellrowborder" valign="top" width="10.05%" id="mcps1.2.4.1.2"><p id="zh-cn_topic_0235751031_p16609919"><a name="zh-cn_topic_0235751031_p16609919"></a><a name="zh-cn_topic_0235751031_p16609919"></a>输入/输出</p>
</th>
<th class="cellrowborder" valign="top" width="71.41%" id="mcps1.2.4.1.3"><p id="zh-cn_topic_0235751031_p59995477"><a name="zh-cn_topic_0235751031_p59995477"></a><a name="zh-cn_topic_0235751031_p59995477"></a>描述</p>
</th>
</tr>
</thead>
<tbody><tr id="row42461942101815"><td class="cellrowborder" valign="top" width="18.54%" headers="mcps1.2.4.1.1 "><p id="p108051250181214"><a name="p108051250181214"></a><a name="p108051250181214"></a>dstReg</p>
</td>
<td class="cellrowborder" valign="top" width="10.05%" headers="mcps1.2.4.1.2 "><p id="p158449584436"><a name="p158449584436"></a><a name="p158449584436"></a>输出</p>
</td>
<td class="cellrowborder" valign="top" width="71.41%" headers="mcps1.2.4.1.3 "><p id="p6389114616514"><a name="p6389114616514"></a><a name="p6389114616514"></a>目的操作数。</p>
<p id="p66093533169"><a name="p66093533169"></a><a name="p66093533169"></a>类型为<a href="RegTensor.md">RegTensor</a>。</p>
<p id="p164006149594"><a name="p164006149594"></a><a name="p164006149594"></a><span id="ph434701113253"><a name="ph434701113253"></a><a name="ph434701113253"></a>Ascend 950PR/Ascend 950DT</span>支持的数据类型参考<a href="#table17752925941">表3</a>、<a href="#table18665350162315">表4</a>、<a href="#table2536411175011">表5</a>、<a href="#table183851919811">表6</a>。</p>
</td>
</tr>
<tr id="row2137145181815"><td class="cellrowborder" valign="top" width="18.54%" headers="mcps1.2.4.1.1 "><p id="p19574165615129"><a name="p19574165615129"></a><a name="p19574165615129"></a>srcReg</p>
</td>
<td class="cellrowborder" valign="top" width="10.05%" headers="mcps1.2.4.1.2 "><p id="p128442058144312"><a name="p128442058144312"></a><a name="p128442058144312"></a>输入</p>
</td>
<td class="cellrowborder" valign="top" width="71.41%" headers="mcps1.2.4.1.3 "><p id="p172083541517"><a name="p172083541517"></a><a name="p172083541517"></a>源操作数。</p>
<p id="p7123111612517"><a name="p7123111612517"></a><a name="p7123111612517"></a>类型为<a href="RegTensor.md">RegTensor</a>。</p>
<p id="p288804212117"><a name="p288804212117"></a><a name="p288804212117"></a><span id="ph18889421413"><a name="ph18889421413"></a><a name="ph18889421413"></a>Ascend 950PR/Ascend 950DT</span>支持的数据类型参考<a href="#table17752925941">表3</a>、<a href="#table18665350162315">表4</a>、<a href="#table2536411175011">表5</a>、<a href="#table183851919811">表6</a>。</p>
</td>
</tr>
<tr id="row19615183817191"><td class="cellrowborder" valign="top" width="18.54%" headers="mcps1.2.4.1.1 "><p id="p1484519586432"><a name="p1484519586432"></a><a name="p1484519586432"></a>mask</p>
</td>
<td class="cellrowborder" valign="top" width="10.05%" headers="mcps1.2.4.1.2 "><p id="p484514581433"><a name="p484514581433"></a><a name="p484514581433"></a>输入</p>
</td>
<td class="cellrowborder" valign="top" width="71.41%" headers="mcps1.2.4.1.3 "><p id="p1337312597287"><a name="p1337312597287"></a><a name="p1337312597287"></a><span id="ph15776181222"><a name="ph15776181222"></a><a name="ph15776181222"></a>源操作数元素操作的有效指示，详细说明请参考<a href="MaskReg.md">MaskReg</a>。</span></p>
<p id="p162828912187"><a name="p162828912187"></a><a name="p162828912187"></a>注意：数据类型转换的mask会按照输入和输出类型中sizeof(dtype)较大的来筛选。</p>
</td>
</tr>
</tbody>
</table>

**表 3**  浮点转整数

<a name="table17752925941"></a>
<table><thead align="left"><tr id="row87528251243"><th class="cellrowborder" valign="top" width="16.666666666666664%" id="mcps1.2.7.1.1"><p id="p1432987510"><a name="p1432987510"></a><a name="p1432987510"></a>src dtype</p>
</th>
<th class="cellrowborder" valign="top" width="16.64667066586683%" id="mcps1.2.7.1.2"><p id="p143248657"><a name="p143248657"></a><a name="p143248657"></a>dst dtype</p>
</th>
<th class="cellrowborder" valign="top" width="16.686662667466507%" id="mcps1.2.7.1.3"><p id="p432682055"><a name="p432682055"></a><a name="p432682055"></a>mode</p>
</th>
<th class="cellrowborder" valign="top" width="16.666666666666664%" id="mcps1.2.7.1.4"><p id="p03220816515"><a name="p03220816515"></a><a name="p03220816515"></a>round mode</p>
</th>
<th class="cellrowborder" valign="top" width="16.666666666666664%" id="mcps1.2.7.1.5"><p id="p1032781351"><a name="p1032781351"></a><a name="p1032781351"></a>sat mode</p>
</th>
<th class="cellrowborder" valign="top" width="16.666666666666664%" id="mcps1.2.7.1.6"><p id="p79509128161"><a name="p79509128161"></a><a name="p79509128161"></a>layout mode</p>
</th>
</tr>
</thead>
<tbody><tr id="row196420518582"><td class="cellrowborder" valign="top" width="16.666666666666664%" headers="mcps1.2.7.1.1 "><p id="p186418525810"><a name="p186418525810"></a><a name="p186418525810"></a>half</p>
</td>
<td class="cellrowborder" valign="top" width="16.64667066586683%" headers="mcps1.2.7.1.2 "><p id="p3641751582"><a name="p3641751582"></a><a name="p3641751582"></a>int4x2_t</p>
</td>
<td class="cellrowborder" rowspan="8" valign="top" width="16.686662667466507%" headers="mcps1.2.7.1.3 "><p id="p76414510588"><a name="p76414510588"></a><a name="p76414510588"></a>MaskMergeMode::ZEROING</p>
</td>
<td class="cellrowborder" rowspan="8" valign="top" width="16.666666666666664%" headers="mcps1.2.7.1.4 "><p id="p56417515816"><a name="p56417515816"></a><a name="p56417515816"></a>RoundMode::CAST_RINT,</p>
<p id="p14321184516"><a name="p14321184516"></a><a name="p14321184516"></a>RoundMode::CAST_ROUND,</p>
<p id="p732588511"><a name="p732588511"></a><a name="p732588511"></a>RoundMode::CAST_FLOOR,</p>
<p id="p18322081512"><a name="p18322081512"></a><a name="p18322081512"></a>RoundMode::CAST_CEIL,</p>
<p id="p33248956"><a name="p33248956"></a><a name="p33248956"></a>RoundMode::CAST_TRUNC</p>
</td>
<td class="cellrowborder" rowspan="7" valign="top" width="16.666666666666664%" headers="mcps1.2.7.1.5 "><p id="p2644555818"><a name="p2644555818"></a><a name="p2644555818"></a>SatMode::NO_SAT</p>
<p id="p1532381551"><a name="p1532381551"></a><a name="p1532381551"></a>SatMode::SAT</p>
</td>
<td class="cellrowborder" valign="top" width="16.666666666666664%" headers="mcps1.2.7.1.6 "><p id="p66416510582"><a name="p66416510582"></a><a name="p66416510582"></a>RegLayout::ZERO, RegLayout::ONE, RegLayout::TWO, RegLayout::THREE</p>
</td>
</tr>
<tr id="row10752925843"><td class="cellrowborder" valign="top" headers="mcps1.2.7.1.1 "><p id="p9321584513"><a name="p9321584513"></a><a name="p9321584513"></a>float</p>
</td>
<td class="cellrowborder" valign="top" headers="mcps1.2.7.1.2 "><p id="p16321481155"><a name="p16321481155"></a><a name="p16321481155"></a>int16_t</p>
</td>
<td class="cellrowborder" rowspan="4" valign="top" headers="mcps1.2.7.1.3 "><p id="p355017811112"><a name="p355017811112"></a><a name="p355017811112"></a>RegLayout::ZERO</p>
<p id="p19550481114"><a name="p19550481114"></a><a name="p19550481114"></a>RegLayout::ONE</p>
</td>
</tr>
<tr id="row137521925045"><td class="cellrowborder" valign="top" headers="mcps1.2.7.1.1 "><p id="p7325819514"><a name="p7325819514"></a><a name="p7325819514"></a>half</p>
</td>
<td class="cellrowborder" valign="top" headers="mcps1.2.7.1.2 "><p id="p732783512"><a name="p732783512"></a><a name="p732783512"></a>int8_t</p>
</td>
</tr>
<tr id="row475312518420"><td class="cellrowborder" valign="top" headers="mcps1.2.7.1.1 "><p id="p1232168659"><a name="p1232168659"></a><a name="p1232168659"></a>half</p>
</td>
<td class="cellrowborder" valign="top" headers="mcps1.2.7.1.2 "><p id="p2322816512"><a name="p2322816512"></a><a name="p2322816512"></a>uint8_t</p>
</td>
</tr>
<tr id="row77537259419"><td class="cellrowborder" valign="top" headers="mcps1.2.7.1.1 "><p id="p176410215618"><a name="p176410215618"></a><a name="p176410215618"></a>float</p>
</td>
<td class="cellrowborder" valign="top" headers="mcps1.2.7.1.2 "><p id="p167641321064"><a name="p167641321064"></a><a name="p167641321064"></a>int64_t</p>
</td>
</tr>
<tr id="row255755719714"><td class="cellrowborder" valign="top" headers="mcps1.2.7.1.1 "><p id="p86323587719"><a name="p86323587719"></a><a name="p86323587719"></a>float</p>
</td>
<td class="cellrowborder" valign="top" headers="mcps1.2.7.1.2 "><p id="p10632858676"><a name="p10632858676"></a><a name="p10632858676"></a>int32_t</p>
</td>
<td class="cellrowborder" rowspan="2" valign="top" headers="mcps1.2.7.1.3 "><p id="p15819184499"><a name="p15819184499"></a><a name="p15819184499"></a>RegLayout::UNKNOWN</p>
</td>
</tr>
<tr id="row18559112614912"><td class="cellrowborder" valign="top" headers="mcps1.2.7.1.1 "><p id="p616514442092"><a name="p616514442092"></a><a name="p616514442092"></a>half</p>
</td>
<td class="cellrowborder" valign="top" headers="mcps1.2.7.1.2 "><p id="p131651844290"><a name="p131651844290"></a><a name="p131651844290"></a>int16_t</p>
</td>
</tr>
<tr id="row19229321793"><td class="cellrowborder" valign="top" headers="mcps1.2.7.1.1 "><p id="p0282218201212"><a name="p0282218201212"></a><a name="p0282218201212"></a>half</p>
</td>
<td class="cellrowborder" valign="top" headers="mcps1.2.7.1.2 "><p id="p9282418181219"><a name="p9282418181219"></a><a name="p9282418181219"></a>int32_t</p>
</td>
<td class="cellrowborder" valign="top" headers="mcps1.2.7.1.3 "><p id="p9282101815127"><a name="p9282101815127"></a><a name="p9282101815127"></a>SatMode::UNKNOWN</p>
</td>
<td class="cellrowborder" valign="top" headers="mcps1.2.7.1.4 "><p id="p25611910121114"><a name="p25611910121114"></a><a name="p25611910121114"></a>RegLayout::ZERO</p>
<p id="p3561810161116"><a name="p3561810161116"></a><a name="p3561810161116"></a>RegLayout::ONE</p>
</td>
</tr>
</tbody>
</table>

**表 4**  浮点转浮点

<a name="table18665350162315"></a>
<table><thead align="left"><tr id="row766575022310"><th class="cellrowborder" valign="top" width="13.0373925214957%" id="mcps1.2.7.1.1"><p id="p1098426162416"><a name="p1098426162416"></a><a name="p1098426162416"></a>src dtype</p>
</th>
<th class="cellrowborder" valign="top" width="13.477304539092183%" id="mcps1.2.7.1.2"><p id="p598102602411"><a name="p598102602411"></a><a name="p598102602411"></a>dst dtype</p>
</th>
<th class="cellrowborder" valign="top" width="20.425914817036595%" id="mcps1.2.7.1.3"><p id="p11981526172413"><a name="p11981526172413"></a><a name="p11981526172413"></a>mode</p>
</th>
<th class="cellrowborder" valign="top" width="19.72605478904219%" id="mcps1.2.7.1.4"><p id="p498526142414"><a name="p498526142414"></a><a name="p498526142414"></a>round mode</p>
</th>
<th class="cellrowborder" valign="top" width="16.666666666666664%" id="mcps1.2.7.1.5"><p id="p59862622413"><a name="p59862622413"></a><a name="p59862622413"></a>sat mode</p>
</th>
<th class="cellrowborder" valign="top" width="16.666666666666664%" id="mcps1.2.7.1.6"><p id="p174039189165"><a name="p174039189165"></a><a name="p174039189165"></a>layout mode</p>
</th>
</tr>
</thead>
<tbody><tr id="row11666155012316"><td class="cellrowborder" valign="top" width="13.0373925214957%" headers="mcps1.2.7.1.1 "><p id="p15991926182419"><a name="p15991926182419"></a><a name="p15991926182419"></a>half</p>
</td>
<td class="cellrowborder" valign="top" width="13.477304539092183%" headers="mcps1.2.7.1.2 "><p id="p99912266246"><a name="p99912266246"></a><a name="p99912266246"></a>float</p>
</td>
<td class="cellrowborder" rowspan="20" valign="top" width="20.425914817036595%" headers="mcps1.2.7.1.3 "><p id="p177234318465"><a name="p177234318465"></a><a name="p177234318465"></a>MaskMergeMode::ZEROING</p>
</td>
<td class="cellrowborder" rowspan="3" valign="top" width="19.72605478904219%" headers="mcps1.2.7.1.4 "><p id="p799122632415"><a name="p799122632415"></a><a name="p799122632415"></a>RoundMode::UNKNOWN</p>
</td>
<td class="cellrowborder" rowspan="3" valign="top" width="16.666666666666664%" headers="mcps1.2.7.1.5 "><p id="p129912268247"><a name="p129912268247"></a><a name="p129912268247"></a>SatMode::UNKNOWN</p>
</td>
<td class="cellrowborder" rowspan="6" valign="top" width="16.666666666666664%" headers="mcps1.2.7.1.6 "><p id="p1123711461116"><a name="p1123711461116"></a><a name="p1123711461116"></a>RegLayout::ZERO</p>
<p id="p1523781411113"><a name="p1523781411113"></a><a name="p1523781411113"></a>RegLayout::ONE</p>
</td>
</tr>
<tr id="row13666105082313"><td class="cellrowborder" valign="top" headers="mcps1.2.7.1.1 "><p id="p87187496248"><a name="p87187496248"></a><a name="p87187496248"></a>bfloat16_t</p>
</td>
<td class="cellrowborder" valign="top" headers="mcps1.2.7.1.2 "><p id="p371816491248"><a name="p371816491248"></a><a name="p371816491248"></a>float</p>
</td>
</tr>
<tr id="row966665072310"><td class="cellrowborder" valign="top" headers="mcps1.2.7.1.1 "><p id="p15262957193212"><a name="p15262957193212"></a><a name="p15262957193212"></a>hifloat8_t</p>
</td>
<td class="cellrowborder" valign="top" headers="mcps1.2.7.1.2 "><p id="p1826265743216"><a name="p1826265743216"></a><a name="p1826265743216"></a>half</p>
</td>
</tr>
<tr id="row5441113338"><td class="cellrowborder" valign="top" headers="mcps1.2.7.1.1 "><p id="p849010358346"><a name="p849010358346"></a><a name="p849010358346"></a>float</p>
</td>
<td class="cellrowborder" valign="top" headers="mcps1.2.7.1.2 "><p id="p8490203512347"><a name="p8490203512347"></a><a name="p8490203512347"></a>bfloat16_t</p>
</td>
<td class="cellrowborder" valign="top" headers="mcps1.2.7.1.3 "><p id="p649033553411"><a name="p649033553411"></a><a name="p649033553411"></a>RoundMode::CAST_RINT,</p>
<p id="p11490935183418"><a name="p11490935183418"></a><a name="p11490935183418"></a>RoundMode::CAST_ROUND,</p>
<p id="p2049013357343"><a name="p2049013357343"></a><a name="p2049013357343"></a>RoundMode::CAST_FLOOR,</p>
<p id="p6490153553418"><a name="p6490153553418"></a><a name="p6490153553418"></a>RoundMode::CAST_CEIL,</p>
<p id="p1349063517346"><a name="p1349063517346"></a><a name="p1349063517346"></a>RoundMode::CAST_TRUNC</p>
</td>
<td class="cellrowborder" rowspan="6" valign="top" headers="mcps1.2.7.1.4 "><p id="p1549019355347"><a name="p1549019355347"></a><a name="p1549019355347"></a>SatMode::NO_SAT</p>
<p id="p18490173510342"><a name="p18490173510342"></a><a name="p18490173510342"></a>SatMode::SAT</p>
</td>
</tr>
<tr id="row1194415506320"><td class="cellrowborder" valign="top" headers="mcps1.2.7.1.1 "><p id="p173371524183511"><a name="p173371524183511"></a><a name="p173371524183511"></a>float</p>
</td>
<td class="cellrowborder" valign="top" headers="mcps1.2.7.1.2 "><p id="p1533732463519"><a name="p1533732463519"></a><a name="p1533732463519"></a>half</p>
</td>
<td class="cellrowborder" valign="top" headers="mcps1.2.7.1.3 "><p id="p1133719242355"><a name="p1133719242355"></a><a name="p1133719242355"></a>RoundMode::CAST_ODD，</p>
<p id="p1879961132213"><a name="p1879961132213"></a><a name="p1879961132213"></a>RoundMode::CAST_RINT,</p>
<p id="p207991711112210"><a name="p207991711112210"></a><a name="p207991711112210"></a>RoundMode::CAST_ROUND,</p>
<p id="p7799121122219"><a name="p7799121122219"></a><a name="p7799121122219"></a>RoundMode::CAST_FLOOR</p>
<p id="p97991711132214"><a name="p97991711132214"></a><a name="p97991711132214"></a>RoundMode::CAST_CEIL,</p>
<p id="p17799201110220"><a name="p17799201110220"></a><a name="p17799201110220"></a>RoundMode::CAST_TRUNC</p>
</td>
</tr>
<tr id="row175711019103612"><td class="cellrowborder" valign="top" headers="mcps1.2.7.1.1 "><p id="p10131028153612"><a name="p10131028153612"></a><a name="p10131028153612"></a>half</p>
</td>
<td class="cellrowborder" valign="top" headers="mcps1.2.7.1.2 "><p id="p813142823618"><a name="p813142823618"></a><a name="p813142823618"></a>hifloat8_t</p>
</td>
<td class="cellrowborder" rowspan="2" valign="top" headers="mcps1.2.7.1.3 "><p id="p1813028103610"><a name="p1813028103610"></a><a name="p1813028103610"></a>RoundMode::CAST_ROUND,</p>
<p id="p71342893617"><a name="p71342893617"></a><a name="p71342893617"></a>RoundMode::CAST_HYBRID</p>
</td>
</tr>
<tr id="row18055213363"><td class="cellrowborder" valign="top" headers="mcps1.2.7.1.1 "><p id="p9615340193617"><a name="p9615340193617"></a><a name="p9615340193617"></a>float</p>
</td>
<td class="cellrowborder" valign="top" headers="mcps1.2.7.1.2 "><p id="p20615184019363"><a name="p20615184019363"></a><a name="p20615184019363"></a>hifloat8_t</p>
</td>
<td class="cellrowborder" rowspan="6" valign="top" headers="mcps1.2.7.1.3 "><p id="p204241525131610"><a name="p204241525131610"></a><a name="p204241525131610"></a>RegLayout::ZERO, RegLayout::ONE, RegLayout::TWO, RegLayout::THREE</p>
</td>
</tr>
<tr id="row842345013360"><td class="cellrowborder" valign="top" headers="mcps1.2.7.1.1 "><p id="p7847313720"><a name="p7847313720"></a><a name="p7847313720"></a>float</p>
</td>
<td class="cellrowborder" valign="top" headers="mcps1.2.7.1.2 "><p id="p1284123173715"><a name="p1284123173715"></a><a name="p1284123173715"></a>fp8_e5m2_t</p>
</td>
<td class="cellrowborder" rowspan="2" valign="top" headers="mcps1.2.7.1.3 "><p id="p987614211379"><a name="p987614211379"></a><a name="p987614211379"></a>RoundMode::CAST_RINT</p>
</td>
</tr>
<tr id="row1587818525360"><td class="cellrowborder" valign="top" headers="mcps1.2.7.1.1 "><p id="p3847313716"><a name="p3847313716"></a><a name="p3847313716"></a>float</p>
</td>
<td class="cellrowborder" valign="top" headers="mcps1.2.7.1.2 "><p id="p14849363717"><a name="p14849363717"></a><a name="p14849363717"></a>fp8_e4m3fn_t</p>
</td>
</tr>
<tr id="row4471838104014"><td class="cellrowborder" valign="top" headers="mcps1.2.7.1.1 "><p id="p16398175316407"><a name="p16398175316407"></a><a name="p16398175316407"></a>hifloat8_t</p>
</td>
<td class="cellrowborder" valign="top" headers="mcps1.2.7.1.2 "><p id="p16398135324018"><a name="p16398135324018"></a><a name="p16398135324018"></a>float</p>
</td>
<td class="cellrowborder" rowspan="3" valign="top" headers="mcps1.2.7.1.3 "><p id="p1787791912411"><a name="p1787791912411"></a><a name="p1787791912411"></a>RoundMode::UNKNOWN</p>
</td>
<td class="cellrowborder" rowspan="7" valign="top" headers="mcps1.2.7.1.4 "><p id="p8234181422116"><a name="p8234181422116"></a><a name="p8234181422116"></a>SatMode::UNKNOWN</p>
</td>
</tr>
<tr id="row377018409403"><td class="cellrowborder" valign="top" headers="mcps1.2.7.1.1 "><p id="p163998535400"><a name="p163998535400"></a><a name="p163998535400"></a>fp8_e4m3fn_t</p>
</td>
<td class="cellrowborder" valign="top" headers="mcps1.2.7.1.2 "><p id="p939955310400"><a name="p939955310400"></a><a name="p939955310400"></a>float</p>
</td>
</tr>
<tr id="row15883442124010"><td class="cellrowborder" valign="top" headers="mcps1.2.7.1.1 "><p id="p8399253204013"><a name="p8399253204013"></a><a name="p8399253204013"></a>fp8_e5m2_t</p>
</td>
<td class="cellrowborder" valign="top" headers="mcps1.2.7.1.2 "><p id="p1139945324013"><a name="p1139945324013"></a><a name="p1139945324013"></a>float</p>
</td>
</tr>
<tr id="row980613492617"><td class="cellrowborder" valign="top" headers="mcps1.2.7.1.1 "><p id="p2080715497620"><a name="p2080715497620"></a><a name="p2080715497620"></a>bfloat16_t</p>
</td>
<td class="cellrowborder" valign="top" headers="mcps1.2.7.1.2 "><p id="p180711492612"><a name="p180711492612"></a><a name="p180711492612"></a>fp4x2_e2m1_t</p>
</td>
<td class="cellrowborder" rowspan="2" valign="top" headers="mcps1.2.7.1.3 "><p id="p936720271495"><a name="p936720271495"></a><a name="p936720271495"></a>RoundMode::CAST_RINT,</p>
<p id="p1136718271699"><a name="p1136718271699"></a><a name="p1136718271699"></a>RoundMode::CAST_ROUND,</p>
<p id="p136711276919"><a name="p136711276919"></a><a name="p136711276919"></a>RoundMode::CAST_FLOOR,</p>
<p id="p153677274920"><a name="p153677274920"></a><a name="p153677274920"></a>RoundMode::CAST_CEIL,</p>
<p id="p1536714271293"><a name="p1536714271293"></a><a name="p1536714271293"></a>RoundMode::CAST_TRUNC</p>
</td>
<td class="cellrowborder" rowspan="6" valign="top" headers="mcps1.2.7.1.4 "><p id="p456643344915"><a name="p456643344915"></a><a name="p456643344915"></a>RegLayout::UNKNOWN</p>
</td>
</tr>
<tr id="row55612525611"><td class="cellrowborder" valign="top" headers="mcps1.2.7.1.1 "><p id="p145665219610"><a name="p145665219610"></a><a name="p145665219610"></a>bfloat16_t</p>
</td>
<td class="cellrowborder" valign="top" headers="mcps1.2.7.1.2 "><p id="p145610527619"><a name="p145610527619"></a><a name="p145610527619"></a>fp4x2_e1m2_t</p>
</td>
</tr>
<tr id="row1246625413613"><td class="cellrowborder" valign="top" headers="mcps1.2.7.1.1 "><p id="p457013335113"><a name="p457013335113"></a><a name="p457013335113"></a>fp4x2_e2m1_t</p>
</td>
<td class="cellrowborder" valign="top" headers="mcps1.2.7.1.2 "><p id="p13466454364"><a name="p13466454364"></a><a name="p13466454364"></a>bfloat16_t</p>
</td>
<td class="cellrowborder" rowspan="2" valign="top" headers="mcps1.2.7.1.3 "><p id="p149337443129"><a name="p149337443129"></a><a name="p149337443129"></a>RoundMode::UNKNOWN</p>
</td>
</tr>
<tr id="row107221256167"><td class="cellrowborder" valign="top" headers="mcps1.2.7.1.1 "><p id="p9634435101115"><a name="p9634435101115"></a><a name="p9634435101115"></a>fp4x2_e1m2_t</p>
</td>
<td class="cellrowborder" valign="top" headers="mcps1.2.7.1.2 "><p id="p1472217561165"><a name="p1472217561165"></a><a name="p1472217561165"></a>bfloat16_t</p>
</td>
</tr>
<tr id="row7203855141314"><td class="cellrowborder" valign="top" headers="mcps1.2.7.1.1 "><p id="p420325520136"><a name="p420325520136"></a><a name="p420325520136"></a>bfloat16_t</p>
</td>
<td class="cellrowborder" valign="top" headers="mcps1.2.7.1.2 "><p id="p12203555131319"><a name="p12203555131319"></a><a name="p12203555131319"></a>half</p>
</td>
<td class="cellrowborder" rowspan="2" valign="top" headers="mcps1.2.7.1.3 "><p id="p88499274172"><a name="p88499274172"></a><a name="p88499274172"></a>RoundMode::CAST_RINT,</p>
<p id="p15849427191716"><a name="p15849427191716"></a><a name="p15849427191716"></a>RoundMode::CAST_ROUND,</p>
<p id="p11849827111711"><a name="p11849827111711"></a><a name="p11849827111711"></a>RoundMode::CAST_FLOOR,</p>
<p id="p1584982714176"><a name="p1584982714176"></a><a name="p1584982714176"></a>RoundMode::CAST_CEIL,</p>
<p id="p17849127191719"><a name="p17849127191719"></a><a name="p17849127191719"></a>RoundMode::CAST_TRUNC</p>
</td>
<td class="cellrowborder" valign="top" headers="mcps1.2.7.1.4 "><p id="p1511414587176"><a name="p1511414587176"></a><a name="p1511414587176"></a>SatMode::NO_SAT</p>
<p id="p141146582176"><a name="p141146582176"></a><a name="p141146582176"></a>SatMode::SAT</p>
</td>
</tr>
<tr id="row215115818132"><td class="cellrowborder" valign="top" headers="mcps1.2.7.1.1 "><p id="p122541291415"><a name="p122541291415"></a><a name="p122541291415"></a>half</p>
</td>
<td class="cellrowborder" valign="top" headers="mcps1.2.7.1.2 "><p id="p333509141411"><a name="p333509141411"></a><a name="p333509141411"></a>bfloat16_t</p>
</td>
<td class="cellrowborder" rowspan="3" valign="top" headers="mcps1.2.7.1.3 "><p id="p62616147230"><a name="p62616147230"></a><a name="p62616147230"></a>SatMode::UNKNOWN</p>
</td>
</tr>
<tr id="row15880194062111"><td class="cellrowborder" valign="top" headers="mcps1.2.7.1.1 "><p id="p168801405213"><a name="p168801405213"></a><a name="p168801405213"></a>bfloat16_t</p>
</td>
<td class="cellrowborder" valign="top" headers="mcps1.2.7.1.2 "><p id="p18880184042112"><a name="p18880184042112"></a><a name="p18880184042112"></a>fp8_e8m0_t</p>
</td>
<td class="cellrowborder" rowspan="2" valign="top" headers="mcps1.2.7.1.3 "><p id="p2168283220"><a name="p2168283220"></a><a name="p2168283220"></a>RoundMode::UNKNOWN</p>
</td>
<td class="cellrowborder" rowspan="2" valign="top" headers="mcps1.2.7.1.4 "><p id="p151951237119"><a name="p151951237119"></a><a name="p151951237119"></a>RegLayout::ZERO</p>
<p id="p71951423141119"><a name="p71951423141119"></a><a name="p71951423141119"></a>RegLayout::ONE</p>
</td>
</tr>
<tr id="row6629433214"><td class="cellrowborder" valign="top" headers="mcps1.2.7.1.1 "><p id="p15237102122216"><a name="p15237102122216"></a><a name="p15237102122216"></a>fp8_e8m0_t</p>
</td>
<td class="cellrowborder" valign="top" headers="mcps1.2.7.1.2 "><p id="p86294302113"><a name="p86294302113"></a><a name="p86294302113"></a>bfloat16_t</p>
</td>
</tr>
</tbody>
</table>

**表 5**  整数转浮点

<a name="table2536411175011"></a>
<table><thead align="left"><tr id="row1853616114502"><th class="cellrowborder" valign="top" width="20%" id="mcps1.2.6.1.1"><p id="p11162827165018"><a name="p11162827165018"></a><a name="p11162827165018"></a>src dtype</p>
</th>
<th class="cellrowborder" valign="top" width="20%" id="mcps1.2.6.1.2"><p id="p716218276501"><a name="p716218276501"></a><a name="p716218276501"></a>dst dtype</p>
</th>
<th class="cellrowborder" valign="top" width="20%" id="mcps1.2.6.1.3"><p id="p61621627185016"><a name="p61621627185016"></a><a name="p61621627185016"></a>mode</p>
</th>
<th class="cellrowborder" valign="top" width="20%" id="mcps1.2.6.1.4"><p id="p1216232765011"><a name="p1216232765011"></a><a name="p1216232765011"></a>round mode</p>
</th>
<th class="cellrowborder" valign="top" width="20%" id="mcps1.2.6.1.5"><p id="p916272725011"><a name="p916272725011"></a><a name="p916272725011"></a>layout mode</p>
</th>
</tr>
</thead>
<tbody><tr id="row3527201119571"><td class="cellrowborder" valign="top" width="20%" headers="mcps1.2.6.1.1 "><p id="p11527111105711"><a name="p11527111105711"></a><a name="p11527111105711"></a>int4x2_t</p>
</td>
<td class="cellrowborder" valign="top" width="20%" headers="mcps1.2.6.1.2 "><p id="p652721135710"><a name="p652721135710"></a><a name="p652721135710"></a>half</p>
</td>
<td class="cellrowborder" rowspan="8" valign="top" width="20%" headers="mcps1.2.6.1.3 "><p id="p1962955019463"><a name="p1962955019463"></a><a name="p1962955019463"></a>MaskMergeMode::ZEROING</p>
<p id="p173744325019"><a name="p173744325019"></a><a name="p173744325019"></a></p>
<p id="p18802729185812"><a name="p18802729185812"></a><a name="p18802729185812"></a></p>
<p id="p188021829115812"><a name="p188021829115812"></a><a name="p188021829115812"></a></p>
<p id="p5415546145619"><a name="p5415546145619"></a><a name="p5415546145619"></a></p>
</td>
<td class="cellrowborder" rowspan="5" valign="top" width="20%" headers="mcps1.2.6.1.4 "><p id="p10983011155419"><a name="p10983011155419"></a><a name="p10983011155419"></a>RoundMode::UNKNOWN</p>
</td>
<td class="cellrowborder" rowspan="2" valign="top" width="20%" headers="mcps1.2.6.1.5 "><p id="p5404141295919"><a name="p5404141295919"></a><a name="p5404141295919"></a>RegLayout::ZERO, RegLayout::ONE, RegLayout::TWO, RegLayout::THREE</p>
</td>
</tr>
<tr id="row2559205445711"><td class="cellrowborder" valign="top" headers="mcps1.2.6.1.1 "><p id="p6302458165710"><a name="p6302458165710"></a><a name="p6302458165710"></a>int4x2_t</p>
</td>
<td class="cellrowborder" valign="top" headers="mcps1.2.6.1.2 "><p id="p6559754145710"><a name="p6559754145710"></a><a name="p6559754145710"></a>bfloat16_t</p>
</td>
</tr>
<tr id="row16536131155019"><td class="cellrowborder" valign="top" headers="mcps1.2.6.1.1 "><p id="p14341443145013"><a name="p14341443145013"></a><a name="p14341443145013"></a>uint8_t</p>
</td>
<td class="cellrowborder" valign="top" headers="mcps1.2.6.1.2 "><p id="p113444305013"><a name="p113444305013"></a><a name="p113444305013"></a>half</p>
</td>
<td class="cellrowborder" rowspan="4" valign="top" headers="mcps1.2.6.1.3 "><p id="p4596132691110"><a name="p4596132691110"></a><a name="p4596132691110"></a>RegLayout::ZERO</p>
<p id="p65969263111"><a name="p65969263111"></a><a name="p65969263111"></a>RegLayout::ONE</p>
</td>
</tr>
<tr id="row115369118509"><td class="cellrowborder" valign="top" headers="mcps1.2.6.1.1 "><p id="p1434243165014"><a name="p1434243165014"></a><a name="p1434243165014"></a>int8_t</p>
</td>
<td class="cellrowborder" valign="top" headers="mcps1.2.6.1.2 "><p id="p153474355013"><a name="p153474355013"></a><a name="p153474355013"></a>half</p>
</td>
</tr>
<tr id="row17536111114502"><td class="cellrowborder" valign="top" headers="mcps1.2.6.1.1 "><p id="p123494316502"><a name="p123494316502"></a><a name="p123494316502"></a>int16_t</p>
</td>
<td class="cellrowborder" valign="top" headers="mcps1.2.6.1.2 "><p id="p133424316504"><a name="p133424316504"></a><a name="p133424316504"></a>float</p>
</td>
</tr>
<tr id="row3295434165520"><td class="cellrowborder" valign="top" headers="mcps1.2.6.1.1 "><p id="p16475114111554"><a name="p16475114111554"></a><a name="p16475114111554"></a>int64_t</p>
</td>
<td class="cellrowborder" valign="top" headers="mcps1.2.6.1.2 "><p id="p047513418557"><a name="p047513418557"></a><a name="p047513418557"></a>float</p>
</td>
<td class="cellrowborder" rowspan="3" valign="top" headers="mcps1.2.6.1.3 "><p id="p14751441135510"><a name="p14751441135510"></a><a name="p14751441135510"></a>RoundMode::CAST_RINT,</p>
<p id="p114753418553"><a name="p114753418553"></a><a name="p114753418553"></a>RoundMode::CAST_ROUND,</p>
<p id="p8475341165513"><a name="p8475341165513"></a><a name="p8475341165513"></a>RoundMode::CAST_FLOOR,</p>
<p id="p1347514418558"><a name="p1347514418558"></a><a name="p1347514418558"></a>RoundMode::CAST_CEIL,</p>
<p id="p114752041135513"><a name="p114752041135513"></a><a name="p114752041135513"></a>RoundMode::CAST_TRUNC</p>
<p id="p204154465568"><a name="p204154465568"></a><a name="p204154465568"></a></p>
</td>
</tr>
<tr id="row13216153645514"><td class="cellrowborder" valign="top" headers="mcps1.2.6.1.1 "><p id="p143520291562"><a name="p143520291562"></a><a name="p143520291562"></a>int16_t</p>
</td>
<td class="cellrowborder" valign="top" headers="mcps1.2.6.1.2 "><p id="p243532975616"><a name="p243532975616"></a><a name="p243532975616"></a>half</p>
</td>
<td class="cellrowborder" rowspan="2" valign="top" headers="mcps1.2.6.1.3 "><p id="p1422103804915"><a name="p1422103804915"></a><a name="p1422103804915"></a>RegLayout::UNKNOWN</p>
</td>
</tr>
<tr id="row15873203719564"><td class="cellrowborder" valign="top" headers="mcps1.2.6.1.1 "><p id="p1869234720564"><a name="p1869234720564"></a><a name="p1869234720564"></a>int32_t</p>
</td>
<td class="cellrowborder" valign="top" headers="mcps1.2.6.1.2 "><p id="p1569384785616"><a name="p1569384785616"></a><a name="p1569384785616"></a>float</p>
</td>
</tr>
</tbody>
</table>

**表 6**  整数转整数

<a name="table183851919811"></a>
<table><thead align="left"><tr id="row638621920118"><th class="cellrowborder" valign="top" width="19.96%" id="mcps1.2.6.1.1"><p id="p1479210241936"><a name="p1479210241936"></a><a name="p1479210241936"></a>src dtype</p>
</th>
<th class="cellrowborder" valign="top" width="20.02%" id="mcps1.2.6.1.2"><p id="p187923241438"><a name="p187923241438"></a><a name="p187923241438"></a>dst dtype</p>
</th>
<th class="cellrowborder" valign="top" width="20.02%" id="mcps1.2.6.1.3"><p id="p1479214244312"><a name="p1479214244312"></a><a name="p1479214244312"></a>mode</p>
</th>
<th class="cellrowborder" valign="top" width="20%" id="mcps1.2.6.1.4"><p id="p679210241232"><a name="p679210241232"></a><a name="p679210241232"></a>sat mode</p>
</th>
<th class="cellrowborder" valign="top" width="20%" id="mcps1.2.6.1.5"><p id="p17921124434"><a name="p17921124434"></a><a name="p17921124434"></a>layout mode</p>
</th>
</tr>
</thead>
<tbody><tr id="row12332362268"><td class="cellrowborder" valign="top" width="19.96%" headers="mcps1.2.6.1.1 "><p id="p7809134710196"><a name="p7809134710196"></a><a name="p7809134710196"></a>uint32_t</p>
</td>
<td class="cellrowborder" valign="top" width="20.02%" headers="mcps1.2.6.1.2 "><p id="p9809147131917"><a name="p9809147131917"></a><a name="p9809147131917"></a>uint8_t</p>
</td>
<td class="cellrowborder" rowspan="19" valign="top" width="20.02%" headers="mcps1.2.6.1.3 "><p id="p827035564617"><a name="p827035564617"></a><a name="p827035564617"></a>MaskMergeMode::ZEROING</p>
</td>
<td class="cellrowborder" rowspan="8" valign="top" width="20%" headers="mcps1.2.6.1.4 "><p id="p1593043725910"><a name="p1593043725910"></a><a name="p1593043725910"></a>SatMode::NO_SAT</p>
<p id="p1193015371598"><a name="p1193015371598"></a><a name="p1193015371598"></a>SatMode::SAT</p>
</td>
<td class="cellrowborder" rowspan="2" valign="top" width="20%" headers="mcps1.2.6.1.5 "><p id="p3246161821310"><a name="p3246161821310"></a><a name="p3246161821310"></a>RegLayout::ZERO, RegLayout::ONE, RegLayout::TWO, RegLayout::THREE</p>
</td>
</tr>
<tr id="row13812194122616"><td class="cellrowborder" valign="top" headers="mcps1.2.6.1.1 "><p id="p789078132012"><a name="p789078132012"></a><a name="p789078132012"></a>int32_t</p>
</td>
<td class="cellrowborder" valign="top" headers="mcps1.2.6.1.2 "><p id="p88901683209"><a name="p88901683209"></a><a name="p88901683209"></a>uint8_t</p>
</td>
</tr>
<tr id="row83868191110"><td class="cellrowborder" valign="top" headers="mcps1.2.6.1.1 "><p id="p16582209411"><a name="p16582209411"></a><a name="p16582209411"></a>uint16_t</p>
</td>
<td class="cellrowborder" valign="top" headers="mcps1.2.6.1.2 "><p id="p1258920145"><a name="p1258920145"></a><a name="p1258920145"></a>uint8_t</p>
</td>
<td class="cellrowborder" rowspan="12" valign="top" headers="mcps1.2.6.1.3 "><p id="p7331173311114"><a name="p7331173311114"></a><a name="p7331173311114"></a>RegLayout::ZERO</p>
<p id="p1833153311114"><a name="p1833153311114"></a><a name="p1833153311114"></a>RegLayout::ONE</p>
<p id="p462311221517"><a name="p462311221517"></a><a name="p462311221517"></a></p>
<p id="p8674191081515"><a name="p8674191081515"></a><a name="p8674191081515"></a></p>
</td>
</tr>
<tr id="row33869191112"><td class="cellrowborder" valign="top" headers="mcps1.2.6.1.1 "><p id="p558112017415"><a name="p558112017415"></a><a name="p558112017415"></a>int16_t</p>
</td>
<td class="cellrowborder" valign="top" headers="mcps1.2.6.1.2 "><p id="p17585208417"><a name="p17585208417"></a><a name="p17585208417"></a>uint8_t</p>
</td>
</tr>
<tr id="row183868199117"><td class="cellrowborder" valign="top" headers="mcps1.2.6.1.1 "><p id="p5581320843"><a name="p5581320843"></a><a name="p5581320843"></a>uint32_t</p>
</td>
<td class="cellrowborder" valign="top" headers="mcps1.2.6.1.2 "><p id="p175811201748"><a name="p175811201748"></a><a name="p175811201748"></a>uint16_t</p>
</td>
</tr>
<tr id="row14387419219"><td class="cellrowborder" valign="top" headers="mcps1.2.6.1.1 "><p id="p658122019411"><a name="p658122019411"></a><a name="p658122019411"></a>uint32_t</p>
</td>
<td class="cellrowborder" valign="top" headers="mcps1.2.6.1.2 "><p id="p5588201543"><a name="p5588201543"></a><a name="p5588201543"></a>int16_t</p>
</td>
</tr>
<tr id="row038791916118"><td class="cellrowborder" valign="top" headers="mcps1.2.6.1.1 "><p id="p6581520949"><a name="p6581520949"></a><a name="p6581520949"></a>int32_t</p>
</td>
<td class="cellrowborder" valign="top" headers="mcps1.2.6.1.2 "><p id="p20581120842"><a name="p20581120842"></a><a name="p20581120842"></a>uint16_t</p>
</td>
</tr>
<tr id="row3387419719"><td class="cellrowborder" valign="top" headers="mcps1.2.6.1.1 "><p id="p1583208419"><a name="p1583208419"></a><a name="p1583208419"></a>int32_t</p>
</td>
<td class="cellrowborder" valign="top" headers="mcps1.2.6.1.2 "><p id="p1586207418"><a name="p1586207418"></a><a name="p1586207418"></a>int16_t</p>
</td>
</tr>
<tr id="row1386482119713"><td class="cellrowborder" valign="top" headers="mcps1.2.6.1.1 "><p id="p164871431487"><a name="p164871431487"></a><a name="p164871431487"></a>uint8_t</p>
</td>
<td class="cellrowborder" valign="top" headers="mcps1.2.6.1.2 "><p id="p114871631784"><a name="p114871631784"></a><a name="p114871631784"></a>uint16_t</p>
</td>
<td class="cellrowborder" rowspan="10" valign="top" headers="mcps1.2.6.1.3 "><p id="p18771471348"><a name="p18771471348"></a><a name="p18771471348"></a>RoundMode::UNKNOWN</p>
<p id="p9830521171317"><a name="p9830521171317"></a><a name="p9830521171317"></a></p>
</td>
</tr>
<tr id="row1624316281579"><td class="cellrowborder" valign="top" headers="mcps1.2.6.1.1 "><p id="p348817311683"><a name="p348817311683"></a><a name="p348817311683"></a>int8_t</p>
</td>
<td class="cellrowborder" valign="top" headers="mcps1.2.6.1.2 "><p id="p848812313816"><a name="p848812313816"></a><a name="p848812313816"></a>int16_t</p>
</td>
</tr>
<tr id="row1853943016717"><td class="cellrowborder" valign="top" headers="mcps1.2.6.1.1 "><p id="p1488133115814"><a name="p1488133115814"></a><a name="p1488133115814"></a>uint16_t</p>
</td>
<td class="cellrowborder" valign="top" headers="mcps1.2.6.1.2 "><p id="p748818311082"><a name="p748818311082"></a><a name="p748818311082"></a>uint32_t</p>
</td>
</tr>
<tr id="row20834193510715"><td class="cellrowborder" valign="top" headers="mcps1.2.6.1.1 "><p id="p104880313811"><a name="p104880313811"></a><a name="p104880313811"></a>int16_t</p>
</td>
<td class="cellrowborder" valign="top" headers="mcps1.2.6.1.2 "><p id="p13488203110815"><a name="p13488203110815"></a><a name="p13488203110815"></a>uint32_t</p>
</td>
</tr>
<tr id="row08668371478"><td class="cellrowborder" valign="top" headers="mcps1.2.6.1.1 "><p id="p14892311786"><a name="p14892311786"></a><a name="p14892311786"></a>int16_t</p>
</td>
<td class="cellrowborder" valign="top" headers="mcps1.2.6.1.2 "><p id="p54891731789"><a name="p54891731789"></a><a name="p54891731789"></a>int32_t</p>
</td>
</tr>
<tr id="row13698517139"><td class="cellrowborder" valign="top" headers="mcps1.2.6.1.1 "><p id="p15151155912135"><a name="p15151155912135"></a><a name="p15151155912135"></a>int32_t</p>
</td>
<td class="cellrowborder" valign="top" headers="mcps1.2.6.1.2 "><p id="p1315175951312"><a name="p1315175951312"></a><a name="p1315175951312"></a>int64_t</p>
</td>
</tr>
<tr id="row107481655131316"><td class="cellrowborder" valign="top" headers="mcps1.2.6.1.1 "><p id="p101806143172"><a name="p101806143172"></a><a name="p101806143172"></a>uint8_t</p>
</td>
<td class="cellrowborder" valign="top" headers="mcps1.2.6.1.2 "><p id="p111801814121717"><a name="p111801814121717"></a><a name="p111801814121717"></a>uint32_t</p>
</td>
<td class="cellrowborder" rowspan="4" valign="top" headers="mcps1.2.6.1.3 "><p id="p125198598318"><a name="p125198598318"></a><a name="p125198598318"></a>RegLayout::ZERO, RegLayout::ONE, RegLayout::TWO, RegLayout::THREE</p>
</td>
</tr>
<tr id="row31051316172"><td class="cellrowborder" valign="top" headers="mcps1.2.6.1.1 "><p id="p971431911175"><a name="p971431911175"></a><a name="p971431911175"></a>int8_t</p>
</td>
<td class="cellrowborder" valign="top" headers="mcps1.2.6.1.2 "><p id="p27142019111711"><a name="p27142019111711"></a><a name="p27142019111711"></a>int32_t</p>
</td>
</tr>
<tr id="row187401344101113"><td class="cellrowborder" valign="top" headers="mcps1.2.6.1.1 "><p id="p187402449112"><a name="p187402449112"></a><a name="p187402449112"></a>int16_t</p>
</td>
<td class="cellrowborder" valign="top" headers="mcps1.2.6.1.2 "><p id="p1774074420115"><a name="p1774074420115"></a><a name="p1774074420115"></a>int4x2_t</p>
</td>
</tr>
<tr id="row58302210131"><td class="cellrowborder" valign="top" headers="mcps1.2.6.1.1 "><p id="p16830132111316"><a name="p16830132111316"></a><a name="p16830132111316"></a>int4x2_t</p>
</td>
<td class="cellrowborder" valign="top" headers="mcps1.2.6.1.2 "><p id="p158306216137"><a name="p158306216137"></a><a name="p158306216137"></a>int16_t</p>
</td>
</tr>
<tr id="row2694113262613"><td class="cellrowborder" valign="top" headers="mcps1.2.6.1.1 "><p id="p21136503297"><a name="p21136503297"></a><a name="p21136503297"></a>int64_t</p>
</td>
<td class="cellrowborder" valign="top" headers="mcps1.2.6.1.2 "><p id="p111345019292"><a name="p111345019292"></a><a name="p111345019292"></a>int32_t</p>
</td>
<td class="cellrowborder" valign="top" headers="mcps1.2.6.1.3 "><p id="p12512114975211"><a name="p12512114975211"></a><a name="p12512114975211"></a>SatMode::NO_SAT</p>
<p id="p851254975218"><a name="p851254975218"></a><a name="p851254975218"></a>SatMode::SAT</p>
</td>
<td class="cellrowborder" valign="top" headers="mcps1.2.6.1.4 "><p id="p2327123814115"><a name="p2327123814115"></a><a name="p2327123814115"></a>RegLayout::ZERO</p>
<p id="p14327203817112"><a name="p14327203817112"></a><a name="p14327203817112"></a>RegLayout::ONE</p>
</td>
</tr>
</tbody>
</table>

**表 7**  浮点转整数转换规则

<a name="table13605442194314"></a>
<table><thead align="left"><tr id="row1160510421435"><th class="cellrowborder" valign="top" width="27%" id="mcps1.2.3.1.1"><p id="p060504214431"><a name="p060504214431"></a><a name="p060504214431"></a>转换模式</p>
</th>
<th class="cellrowborder" valign="top" width="73%" id="mcps1.2.3.1.2"><p id="p1760554215436"><a name="p1760554215436"></a><a name="p1760554215436"></a>转换规则介绍</p>
</th>
</tr>
</thead>
<tbody><tr id="row6605194234311"><td class="cellrowborder" valign="top" width="27%" headers="mcps1.2.3.1.1 "><p id="p1960554244311"><a name="p1960554244311"></a><a name="p1960554244311"></a>CAST_RINT</p>
</td>
<td class="cellrowborder" valign="top" width="73%" headers="mcps1.2.3.1.2 "><p id="p196051742194315"><a name="p196051742194315"></a><a name="p196051742194315"></a>就近舍入，距离相同时向偶数进位。</p>
<p id="p1498023614191"><a name="p1498023614191"></a><a name="p1498023614191"></a>举例：</p>
<p id="p334281614199"><a name="p334281614199"></a><a name="p334281614199"></a>输入3.3，输出3</p>
<p id="p13156193820193"><a name="p13156193820193"></a><a name="p13156193820193"></a>输入5.9，输出6</p>
<p id="p123500495198"><a name="p123500495198"></a><a name="p123500495198"></a>输入5.5，输出6，因为6是偶数，5是奇数，所以向6舍入</p>
<p id="p677122512018"><a name="p677122512018"></a><a name="p677122512018"></a>输入4.5，输出4，4是偶数</p>
<p id="p699394222013"><a name="p699394222013"></a><a name="p699394222013"></a>输入-2.4，输出2</p>
<p id="p151256332114"><a name="p151256332114"></a><a name="p151256332114"></a>输入-3.6，输出-4</p>
</td>
</tr>
<tr id="row16605342114319"><td class="cellrowborder" valign="top" width="27%" headers="mcps1.2.3.1.1 "><p id="p14605134224317"><a name="p14605134224317"></a><a name="p14605134224317"></a>CAST_ROUND</p>
</td>
<td class="cellrowborder" valign="top" width="73%" headers="mcps1.2.3.1.2 "><p id="p1727717196249"><a name="p1727717196249"></a><a name="p1727717196249"></a>就近舍入，距离相同时向远离0的数值进位。</p>
<p id="p12605134218435"><a name="p12605134218435"></a><a name="p12605134218435"></a>举例：</p>
<p id="p8152731164515"><a name="p8152731164515"></a><a name="p8152731164515"></a>输入3.3，输出3</p>
<p id="p15152193164518"><a name="p15152193164518"></a><a name="p15152193164518"></a>输入5.9，输出6</p>
<p id="p131527314456"><a name="p131527314456"></a><a name="p131527314456"></a>输入5.5，输出6，因为6相比5，距离0更远，所以向6舍入</p>
<p id="p1152131154515"><a name="p1152131154515"></a><a name="p1152131154515"></a>输入-2.4，输出2</p>
<p id="p3152131184510"><a name="p3152131184510"></a><a name="p3152131184510"></a>输入-3.6，输出-4</p>
<p id="p84413486466"><a name="p84413486466"></a><a name="p84413486466"></a>输入-6.5，输出-7， 因为-7相比-6，距离0更远</p>
</td>
</tr>
<tr id="row116051742204313"><td class="cellrowborder" valign="top" width="27%" headers="mcps1.2.3.1.1 "><p id="p126061242164312"><a name="p126061242164312"></a><a name="p126061242164312"></a>CAST_FLOOR</p>
</td>
<td class="cellrowborder" valign="top" width="73%" headers="mcps1.2.3.1.2 "><p id="p1860604234315"><a name="p1860604234315"></a><a name="p1860604234315"></a>向负无穷方向舍入。</p>
<p id="p4954658144819"><a name="p4954658144819"></a><a name="p4954658144819"></a>举例：</p>
<p id="p2604183164913"><a name="p2604183164913"></a><a name="p2604183164913"></a>输入3.2，输出3</p>
<p id="p6319215134920"><a name="p6319215134920"></a><a name="p6319215134920"></a>输入7.9，输出7</p>
<p id="p197711300492"><a name="p197711300492"></a><a name="p197711300492"></a>输入-4.6，输出-5</p>
<p id="p17272150184918"><a name="p17272150184918"></a><a name="p17272150184918"></a>输入-3.1，输出-4</p>
<p id="p540715105500"><a name="p540715105500"></a><a name="p540715105500"></a>相比输入，输出更接近负无穷</p>
</td>
</tr>
<tr id="row136061342104317"><td class="cellrowborder" valign="top" width="27%" headers="mcps1.2.3.1.1 "><p id="p1960619425434"><a name="p1960619425434"></a><a name="p1960619425434"></a>CAST_CEIL</p>
</td>
<td class="cellrowborder" valign="top" width="73%" headers="mcps1.2.3.1.2 "><p id="p15606144274319"><a name="p15606144274319"></a><a name="p15606144274319"></a>向正无穷方向舍入。</p>
<p id="p17337303517"><a name="p17337303517"></a><a name="p17337303517"></a>举例：</p>
<p id="p1473363016518"><a name="p1473363016518"></a><a name="p1473363016518"></a>输入3.2，输出4</p>
<p id="p1573314309517"><a name="p1573314309517"></a><a name="p1573314309517"></a>输入7.9，输出8</p>
<p id="p47331309510"><a name="p47331309510"></a><a name="p47331309510"></a>输入-4.6，输出-4</p>
<p id="p273363011518"><a name="p273363011518"></a><a name="p273363011518"></a>输入-3.1，输出-3</p>
<p id="p173311302519"><a name="p173311302519"></a><a name="p173311302519"></a>相比输入，输出更接近正无穷</p>
</td>
</tr>
<tr id="row1160634214437"><td class="cellrowborder" valign="top" width="27%" headers="mcps1.2.3.1.1 "><p id="p12606124284317"><a name="p12606124284317"></a><a name="p12606124284317"></a>CAST_TRUNC</p>
</td>
<td class="cellrowborder" valign="top" width="73%" headers="mcps1.2.3.1.2 "><p id="p560694244312"><a name="p560694244312"></a><a name="p560694244312"></a>向0方向舍入。</p>
<p id="p6382112620537"><a name="p6382112620537"></a><a name="p6382112620537"></a>举例：</p>
<p id="p4382162685317"><a name="p4382162685317"></a><a name="p4382162685317"></a>输入3.2，输出3</p>
<p id="p16382192618535"><a name="p16382192618535"></a><a name="p16382192618535"></a>输入7.9，输出7</p>
<p id="p4382172645317"><a name="p4382172645317"></a><a name="p4382172645317"></a>输入-4.6，输出-4</p>
<p id="p18382126105314"><a name="p18382126105314"></a><a name="p18382126105314"></a>输入-3.1，输出-3</p>
<p id="p0382162617534"><a name="p0382162617534"></a><a name="p0382162617534"></a>相比输入，输出更接近0</p>
</td>
</tr>
</tbody>
</table>

**表 8**  浮点转浮点转换规则

<a name="table7407201112197"></a>
<table><thead align="left"><tr id="row740831181916"><th class="cellrowborder" valign="top" width="20.32203220322032%" id="mcps1.2.4.1.1"><p id="p1740811118190"><a name="p1740811118190"></a><a name="p1740811118190"></a>src dtype</p>
</th>
<th class="cellrowborder" valign="top" width="16.53165316531653%" id="mcps1.2.4.1.2"><p id="p10408131161911"><a name="p10408131161911"></a><a name="p10408131161911"></a>dst dtype</p>
</th>
<th class="cellrowborder" valign="top" width="63.14631463146314%" id="mcps1.2.4.1.3"><p id="p1340818118193"><a name="p1340818118193"></a><a name="p1340818118193"></a>舍入规则</p>
</th>
</tr>
</thead>
<tbody><tr id="row740821191914"><td class="cellrowborder" valign="top" width="20.32203220322032%" headers="mcps1.2.4.1.1 "><p id="p134081811181910"><a name="p134081811181910"></a><a name="p134081811181910"></a>float</p>
</td>
<td class="cellrowborder" valign="top" width="16.53165316531653%" headers="mcps1.2.4.1.2 "><p id="p94081211131910"><a name="p94081211131910"></a><a name="p94081211131910"></a>half</p>
</td>
<td class="cellrowborder" valign="top" width="63.14631463146314%" headers="mcps1.2.4.1.3 "><p id="p1540861115197"><a name="p1540861115197"></a><a name="p1540861115197"></a><strong id="b38381514161311"><a name="b38381514161311"></a><a name="b38381514161311"></a>CAST_ODD</strong>，向奇数方向舍入。</p>
<p id="p930513625818"><a name="p930513625818"></a><a name="p930513625818"></a>half有尾数位10bit，float有尾数位23bit，float转half舍弃精度13bit，float16最低位标记为黄色<span>y</span>，float舍弃精度标记为红色</p>
<p id="p69671151310"><a name="p69671151310"></a><a name="p69671151310"></a>float +/-a.xxxxxxxxx<span>0</span><span>0000100000000</span>，如果<span>y</span> bit位为0偶数，则进位，得到half a.xxxxxxxxx<span>1</span></p>
<p id="p8816182611101"><a name="p8816182611101"></a><a name="p8816182611101"></a>float +/-a.xxxxxxxxx<span>1</span><span>1000100000000</span>，如果<span>y</span> bit位为1奇数，则不进位，得到half a.xxxxxxxxx<span>1</span></p>
<p id="p1626611416128"><a name="p1626611416128"></a><a name="p1626611416128"></a><strong id="b1978745215159"><a name="b1978745215159"></a><a name="b1978745215159"></a>举例</strong></p>
<p id="p66312561122"><a name="p66312561122"></a><a name="p66312561122"></a>输入float 123.23333，二进制为</p>
<p id="p11984102515136"><a name="p11984102515136"></a><a name="p11984102515136"></a>sign:0, exponent:10000101, mantissa:11101100111011101110111</p>
<p id="p178861155131415"><a name="p178861155131415"></a><a name="p178861155131415"></a>float_exponet = 0b10000101 = 133</p>
<p id="p1776105720134"><a name="p1776105720134"></a><a name="p1776105720134"></a>half_exponet - 15 = 133 - 127</p>
<p id="p11403113213156"><a name="p11403113213156"></a><a name="p11403113213156"></a>half_exponet  = 21 = 0b10101</p>
<p id="p593102015168"><a name="p593102015168"></a><a name="p593102015168"></a>float_mantissa 第14位为1奇数，所以不进位，舍弃后面13位</p>
<p id="p196421781177"><a name="p196421781177"></a><a name="p196421781177"></a>111011001<span>1</span>1011101110111</p>
<p id="p1018984491719"><a name="p1018984491719"></a><a name="p1018984491719"></a>得到half_mantissa = 111011001<span>1</span></p>
<p id="p37914513187"><a name="p37914513187"></a><a name="p37914513187"></a>所以half二进制为</p>
<p id="p989710275186"><a name="p989710275186"></a><a name="p989710275186"></a>sign:0, exponent:10101, mantissa:111011001<span>1</span></p>
<p id="p732414111126"><a name="p732414111126"></a><a name="p732414111126"></a>half = 123.2</p>
</td>
</tr>
<tr id="row1240851119193"><td class="cellrowborder" valign="top" width="20.32203220322032%" headers="mcps1.2.4.1.1 "><p id="p13408181114195"><a name="p13408181114195"></a><a name="p13408181114195"></a>float</p>
</td>
<td class="cellrowborder" valign="top" width="16.53165316531653%" headers="mcps1.2.4.1.2 "><p id="p1140816114190"><a name="p1140816114190"></a><a name="p1140816114190"></a>bfloat16_t</p>
</td>
<td class="cellrowborder" valign="top" width="63.14631463146314%" headers="mcps1.2.4.1.3 "><p id="p24081211101917"><a name="p24081211101917"></a><a name="p24081211101917"></a>float 二进制规则为</p>
<p id="p1752119271715"><a name="p1752119271715"></a><a name="p1752119271715"></a>sign:1bit,exponent:8bit mantissa:23bit</p>
<p id="p681771429"><a name="p681771429"></a><a name="p681771429"></a>bfloat16_t二进制规则为</p>
<p id="p102653137211"><a name="p102653137211"></a><a name="p102653137211"></a>sign:1bit,exponent:8bit,mantissa:7bit</p>
<p id="p140583419215"><a name="p140583419215"></a><a name="p140583419215"></a><strong id="b6822156171317"><a name="b6822156171317"></a><a name="b6822156171317"></a>CAST_RINT：</strong>就近舍入，距离相等时向偶数进位</p>
<p id="p14347630853"><a name="p14347630853"></a><a name="p14347630853"></a>举例</p>
<p id="p034613326512"><a name="p034613326512"></a><a name="p034613326512"></a>1、float32_mantissa:001000<span>0</span><span>0010000010000000</span></p>
<p id="p17571201369"><a name="p17571201369"></a><a name="p17571201369"></a>舍弃精度小于0b<span>1000000000000000，等效于[0, 0.5）</span>，则不进位</p>
<p id="p3897111615815"><a name="p3897111615815"></a><a name="p3897111615815"></a>得到bfloat16_mantissa:001000<span>0</span></p>
<p id="p138895104911"><a name="p138895104911"></a><a name="p138895104911"></a>2、float32_mantissa:001100<span>0</span><span>1000000000000000</span></p>
<p id="p126182056669"><a name="p126182056669"></a><a name="p126182056669"></a>舍弃精度等于0b<span>1000000000000000</span>，bf16<span>最低位</span>为偶数时不进位</p>
<p id="p1311613521273"><a name="p1311613521273"></a><a name="p1311613521273"></a>得到bfloat16_mantissa:001100<span>0</span></p>
<p id="p17297149396"><a name="p17297149396"></a><a name="p17297149396"></a>3、float32_mantissa:001100<span>1</span><span>1000000000000000</span></p>
<p id="p56278297100"><a name="p56278297100"></a><a name="p56278297100"></a>舍弃精度等于0b<span>1000000000000000</span>，bf16<span>最低位</span>为奇数时进位</p>
<p id="p76271729131011"><a name="p76271729131011"></a><a name="p76271729131011"></a>得到bfloat16_mantissa:001101<span>0</span></p>
<p id="p18599104219115"><a name="p18599104219115"></a><a name="p18599104219115"></a>4、float32_mantissa:001000<span>0</span><span>1010000010000000</span></p>
<p id="p1962585631114"><a name="p1962585631114"></a><a name="p1962585631114"></a>舍弃精度大于0b<span>1000000000000000，等效于(0.5, 1）</span>，则进位</p>
<p id="p7625165619112"><a name="p7625165619112"></a><a name="p7625165619112"></a>得到bfloat16_mantissa:001000<span>1</span></p>
<p id="p184123219245"><a name="p184123219245"></a><a name="p184123219245"></a><strong id="b129991830131419"><a name="b129991830131419"></a><a name="b129991830131419"></a>CAST_ROUND：</strong>就近舍入，距离相等时向远离0方向进位</p>
<p id="p43233105264"><a name="p43233105264"></a><a name="p43233105264"></a>举例</p>
<p id="p7323111082613"><a name="p7323111082613"></a><a name="p7323111082613"></a>1、float32_mantissa:001000<span>0</span><span>0010000010000000</span></p>
<p id="p16323161011260"><a name="p16323161011260"></a><a name="p16323161011260"></a>舍弃精度小于0b<span>1000000000000000，等效于[0, 0.5）</span>，则不进位</p>
<p id="p1432311102266"><a name="p1432311102266"></a><a name="p1432311102266"></a>得到bfloat16_mantissa:001000<span>0</span></p>
<p id="p83231510202620"><a name="p83231510202620"></a><a name="p83231510202620"></a>2、float32_mantissa:001100<span>0</span><span>1000000000000000</span></p>
<p id="p153234107266"><a name="p153234107266"></a><a name="p153234107266"></a>舍弃精度等于0b<span>1000000000000000</span>，进位后离0越远，所以进位</p>
<p id="p1932341014263"><a name="p1932341014263"></a><a name="p1932341014263"></a>得到bfloat16_mantissa:001100<span>1</span></p>
<p id="p432420107266"><a name="p432420107266"></a><a name="p432420107266"></a>3、float32_mantissa:001000<span>0</span><span>1010000010000000</span></p>
<p id="p173243108264"><a name="p173243108264"></a><a name="p173243108264"></a>舍弃精度大于0b<span>1000000000000000，等效于(0.5, 1）</span>，则进位</p>
<p id="p5324141052613"><a name="p5324141052613"></a><a name="p5324141052613"></a>得到bfloat16_mantissa:001000<span>1</span></p>
<p id="p135931632192810"><a name="p135931632192810"></a><a name="p135931632192810"></a><strong id="b3395337111411"><a name="b3395337111411"></a><a name="b3395337111411"></a>CAST_FLOOR：</strong>向负无穷方向舍入</p>
<p id="p146125340299"><a name="p146125340299"></a><a name="p146125340299"></a>1、float32_mantissa:001000<span>0</span><span>0010000010000000</span>，float32值正数时不进位，得到bfloat16_mantissa:001000<span>0</span></p>
<p id="p16612203410298"><a name="p16612203410298"></a><a name="p16612203410298"></a>2、float32_mantissa:001100<span>0</span><span>1000000000000000</span></p>
<p id="p66121534202910"><a name="p66121534202910"></a><a name="p66121534202910"></a>float32 值负数时进位，得到bfloat16_mantissa:001100<span>1</span></p>
<p id="p19381183512318"><a name="p19381183512318"></a><a name="p19381183512318"></a><strong id="b199311758111411"><a name="b199311758111411"></a><a name="b199311758111411"></a>CAST_CEIL：</strong>向正无穷方向舍入</p>
<p id="p338111358315"><a name="p338111358315"></a><a name="p338111358315"></a>1、float32_mantissa:001000<span>0</span><span>0010000010000000</span>，float32值正数时进位，得到bfloat16_mantissa:001000<span>1</span></p>
<p id="p3381235113113"><a name="p3381235113113"></a><a name="p3381235113113"></a>2、float32_mantissa:001100<span>0</span><span>1000000000000000</span></p>
<p id="p10381173563113"><a name="p10381173563113"></a><a name="p10381173563113"></a>float32 值负数时不进位，得到bfloat16_mantissa:001100<span>0</span></p>
<p id="p1102938183219"><a name="p1102938183219"></a><a name="p1102938183219"></a><strong id="b193505610157"><a name="b193505610157"></a><a name="b193505610157"></a>CAST_TRUNC：</strong>向0方向舍入</p>
<p id="p06391753163219"><a name="p06391753163219"></a><a name="p06391753163219"></a>1、float32_mantissa:001000<span>0</span><span>0010000010000000</span>，都不进位，直接舍弃红色精度，得到bfloat16_mantissa:00100<span>0</span></p>
<p id="p1047416551915"><a name="p1047416551915"></a><a name="p1047416551915"></a><strong id="b18765154221518"><a name="b18765154221518"></a><a name="b18765154221518"></a>完整举例：</strong>输入float32 205.75，二进制为</p>
<p id="p1131614562316"><a name="p1131614562316"></a><a name="p1131614562316"></a>sign:0, exponent:10000110, mantissa:10011011100000000000000</p>
<p id="p11336504152"><a name="p11336504152"></a><a name="p11336504152"></a>CAST_RINT模式</p>
<p id="p117858252384"><a name="p117858252384"></a><a name="p117858252384"></a>float32_mantissa = 0b100110<span>1</span><span>1100000000000000</span></p>
<p id="p1720215126396"><a name="p1720215126396"></a><a name="p1720215126396"></a>舍弃精度大于0b<span>1000000000000000，等效于(0.5, 1）</span>，则进位</p>
<p id="p7633104212399"><a name="p7633104212399"></a><a name="p7633104212399"></a>得到bfloat16_mantinssa = 0b100111<span>0</span></p>
<p id="p958352974015"><a name="p958352974015"></a><a name="p958352974015"></a>所以bfloat16二进制为</p>
<p id="p14583122911401"><a name="p14583122911401"></a><a name="p14583122911401"></a>sign:0, exponent:10000110, mantissa:100111<span>0</span></p>
<p id="p165831829174019"><a name="p165831829174019"></a><a name="p165831829174019"></a>bfloat16 = 206</p>
</td>
</tr>
<tr id="row6408141171917"><td class="cellrowborder" valign="top" width="20.32203220322032%" headers="mcps1.2.4.1.1 "><p id="p154084116195"><a name="p154084116195"></a><a name="p154084116195"></a>bfloat16_t</p>
</td>
<td class="cellrowborder" valign="top" width="16.53165316531653%" headers="mcps1.2.4.1.2 "><p id="p134081811131918"><a name="p134081811131918"></a><a name="p134081811131918"></a>half</p>
</td>
<td class="cellrowborder" valign="top" width="63.14631463146314%" headers="mcps1.2.4.1.3 "><p id="p191834571175"><a name="p191834571175"></a><a name="p191834571175"></a>bfloat16_t二进制规则为</p>
<p id="p9183135717175"><a name="p9183135717175"></a><a name="p9183135717175"></a>sign:1bit,<span>exponent</span>:8bit,mantissa:7bit</p>
<p id="p14790155813177"><a name="p14790155813177"></a><a name="p14790155813177"></a>float16二进制规则为</p>
<p id="p894423635610"><a name="p894423635610"></a><a name="p894423635610"></a>sign:1bit,<span>exponent</span>:5bit,mantissa:10bit</p>
<p id="p52356618261"><a name="p52356618261"></a><a name="p52356618261"></a>尾数位相差<span>3位</span></p>
<p id="p1310919566229"><a name="p1310919566229"></a><a name="p1310919566229"></a>当bfloat16_exponent = <span>exponent</span>-127 &gt;= -14时</p>
<p id="p337012491715"><a name="p337012491715"></a><a name="p337012491715"></a>因为bfloat16精度低，float16精度高，所以bfloat16到float16精度不变</p>
<p id="p985717391252"><a name="p985717391252"></a><a name="p985717391252"></a><strong id="b1225285194612"><a name="b1225285194612"></a><a name="b1225285194612"></a>例1</strong>：bfloat16_mantissa = 0b1011000 -&gt;</p>
<p id="p19964103717418"><a name="p19964103717418"></a><a name="p19964103717418"></a>float16_maintissa = 0b1011000<span>000</span>, 二进制位不发生进位，只需要再低位补3个0</p>
<p id="p14338194712516"><a name="p14338194712516"></a><a name="p14338194712516"></a>当bfloat16_exponent = <span>exponent</span>-127 &lt; -14时</p>
<p id="p3613101018611"><a name="p3613101018611"></a><a name="p3613101018611"></a><strong id="b15241154217462"><a name="b15241154217462"></a><a name="b15241154217462"></a>例2</strong>：当<span>exponent</span> = 108， bfloat16_exponent = <span>exponent</span>-127 = -19&lt; -14时，因为float16_exponent = <span>exponent</span> - 14 最小为-14，无法表示，所以指数需要+5保持和float16_exponent相等，同时尾数位需要除以2的5次方，因为bf16计算公式 = s * (2^(e-127))*(man) =</p>
<p id="p143605231394"><a name="p143605231394"></a><a name="p143605231394"></a>s * (2^(e-127+5))*(man*2^(-5)),指数位乘以一个数，尾数位需要除以相同大小的数，保证最终值不变。假设bfloat16_mantissa = 0b1011011</p>
<p id="p15828152472510"><a name="p15828152472510"></a><a name="p15828152472510"></a>bfloat16 mantissa计算公式为 1+ man/128, 用例二进制表示为</p>
<p id="p1518413108211"><a name="p1518413108211"></a><a name="p1518413108211"></a>0b1.1011011，指数位乘以2的5次方时，尾数要除以2的5次，小数点向左移动5位，所以bfloat16_mantissa = 0b0.000011011<span>0</span><span>11</span>，尾数为12位，因为float16只有10bit尾数位，所以在以下CAST模式中需要舍弃最低2位，并根据舍入模式决定是否舍入，当前例子的<span>舍弃精度中间值</span>为0b<span>10</span></p>
<p id="p59971634115710"><a name="p59971634115710"></a><a name="p59971634115710"></a><strong id="b1099723405713"><a name="b1099723405713"></a><a name="b1099723405713"></a>CAST_RINT：</strong>就近舍入，距离相等时向偶数进位</p>
<p id="p7696142111015"><a name="p7696142111015"></a><a name="p7696142111015"></a>当bfloat16_mantissa尾数小于等于10位时，不需要舍入；</p>
<p id="p273513429362"><a name="p273513429362"></a><a name="p273513429362"></a>当bfloat16_mantissa尾数大于10位时，类似<strong id="b772554764612"><a name="b772554764612"></a><a name="b772554764612"></a>例2</strong>的例子</p>
<p id="p1634810541362"><a name="p1634810541362"></a><a name="p1634810541362"></a>1）被舍弃的精度小于<span>舍弃精度中间值</span>时，不进位，</p>
<p id="p18180299334"><a name="p18180299334"></a><a name="p18180299334"></a>例bfloat16_mantissa = 0b0.000011011<span>0</span><span>0xxx</span></p>
<p id="p995816457350"><a name="p995816457350"></a><a name="p995816457350"></a>0xxx &lt; <span>1000</span> 输出float16_mantissa = 0b0.000011011<span>0</span></p>
<p id="p44375564362"><a name="p44375564362"></a><a name="p44375564362"></a>2）被舍弃的精度大于<span>舍弃精度中间值</span>时，进位</p>
<p id="p277394183718"><a name="p277394183718"></a><a name="p277394183718"></a>例bfloat16_mantissa = 0b0.000011011<span>0</span><span>0xxx</span></p>
<p id="p6773194115378"><a name="p6773194115378"></a><a name="p6773194115378"></a>1x1x &gt; <span>1000</span> 输出float16_mantissa = 0b0.000011011<span>1</span></p>
<p id="p24988117404"><a name="p24988117404"></a><a name="p24988117404"></a>3) 被舍弃的精度等于<span>舍弃精度中间值</span>时，<span>第10位</span>为偶数时不进位</p>
<p id="p2471459184019"><a name="p2471459184019"></a><a name="p2471459184019"></a>例bfloat16_mantissa = 0b0.000011011<span>0</span><span>0xxx</span></p>
<p id="p647959194019"><a name="p647959194019"></a><a name="p647959194019"></a><span>0</span> % 2 == 0 输出float16_mantissa = 0b0.000011011<span>0</span></p>
<p id="p139931425114113"><a name="p139931425114113"></a><a name="p139931425114113"></a>4) 被舍弃的精度等于<span>舍弃精度中间值</span>时，<span>第10位</span>为奇数时进位</p>
<p id="p1899342544115"><a name="p1899342544115"></a><a name="p1899342544115"></a>例bfloat16_mantissa = 0b0.000011000<span>1</span><span>0xxx</span></p>
<p id="p199931825194111"><a name="p199931825194111"></a><a name="p199931825194111"></a><span>1</span> % 2 == 1 输出float16_mantissa = 0b0.000011000<span>1</span></p>
<p id="p5883125765714"><a name="p5883125765714"></a><a name="p5883125765714"></a><strong id="b288365715715"><a name="b288365715715"></a><a name="b288365715715"></a>CAST_ROUND：</strong>就近舍入，距离相等时向远离0方向进位</p>
<p id="p5249165014440"><a name="p5249165014440"></a><a name="p5249165014440"></a>当bfloat16_mantissa尾数小于等于10位时，不需要进位；</p>
<p id="p151428284467"><a name="p151428284467"></a><a name="p151428284467"></a>当bfloat16_mantissa尾数大于10位时，类似<strong id="b250693124713"><a name="b250693124713"></a><a name="b250693124713"></a>例2</strong>的例子</p>
<p id="p9142028114618"><a name="p9142028114618"></a><a name="p9142028114618"></a>1）被舍弃的精度小于<span>舍弃精度中间值</span>时，不进位，</p>
<p id="p314252894617"><a name="p314252894617"></a><a name="p314252894617"></a>例bfloat16_mantissa = 0b0.000011011<span>0</span><span>0xxx</span></p>
<p id="p20142928164614"><a name="p20142928164614"></a><a name="p20142928164614"></a>0xxx &lt; <span>1000</span> 输出float16_mantissa = 0b0.000011011<span>0</span></p>
<p id="p1714210286469"><a name="p1714210286469"></a><a name="p1714210286469"></a>2）被舍弃的精度大于<span>舍弃精度中间值</span>时，进位</p>
<p id="p1814252854619"><a name="p1814252854619"></a><a name="p1814252854619"></a>例bfloat16_mantissa = 0b0.000011011<span>0</span><span>0xxx</span></p>
<p id="p8142228124614"><a name="p8142228124614"></a><a name="p8142228124614"></a>1x1x &gt; <span>1000</span> 输出float16_mantissa = 0b0.000011011<span>1</span></p>
<p id="p51420286462"><a name="p51420286462"></a><a name="p51420286462"></a>3) 被舍弃的精度等于<span>舍弃精度中间值</span>时，进位远离0，所以进位</p>
<p id="p16142028174612"><a name="p16142028174612"></a><a name="p16142028174612"></a>例bfloat16_mantissa = 0b0.000011011<span>0</span><span>0xxx</span></p>
<p id="p61431528134617"><a name="p61431528134617"></a><a name="p61431528134617"></a>输出float16_mantissa = 0b0.000011011<span>1</span></p>
<p id="p351815175815"><a name="p351815175815"></a><a name="p351815175815"></a><strong id="b16510015185819"><a name="b16510015185819"></a><a name="b16510015185819"></a>CAST_FLOOR：</strong>向负无穷方向舍入</p>
<p id="p79216275497"><a name="p79216275497"></a><a name="p79216275497"></a>当bfloat16_mantissa尾数小于等于10位时，不需要进位；</p>
<p id="p492152716498"><a name="p492152716498"></a><a name="p492152716498"></a>当bfloat16_mantissa尾数大于10位时，类似<strong id="b7921227124913"><a name="b7921227124913"></a><a name="b7921227124913"></a>例2</strong>的例子</p>
<p id="p192152715494"><a name="p192152715494"></a><a name="p192152715494"></a>1）符号位S为0，既正数时不舍入，</p>
<p id="p109211127164911"><a name="p109211127164911"></a><a name="p109211127164911"></a>例bfloat16_mantissa = 0b0.000011011<span>0</span><span>0xxx</span></p>
<p id="p392162719498"><a name="p392162719498"></a><a name="p392162719498"></a>输出float16_mantissa = 0b0.000011011<span>0</span></p>
<p id="p79211227204910"><a name="p79211227204910"></a><a name="p79211227204910"></a>2）符号位S为1，既负数时舍入</p>
<p id="p18921527134916"><a name="p18921527134916"></a><a name="p18921527134916"></a>例bfloat16_mantissa = 0b0.000011011<span>0</span><span>0xxx</span></p>
<p id="p292122754916"><a name="p292122754916"></a><a name="p292122754916"></a>输出float16_mantissa = 0b0.000011011<span>1</span></p>
<p id="p13614123385813"><a name="p13614123385813"></a><a name="p13614123385813"></a><strong id="b1216143495816"><a name="b1216143495816"></a><a name="b1216143495816"></a>CAST_CEIL：</strong>向正无穷方向舍入</p>
<p id="p15294726125210"><a name="p15294726125210"></a><a name="p15294726125210"></a>当bfloat16_mantissa尾数小于等于10位时，不需要舍入；</p>
<p id="p829417266523"><a name="p829417266523"></a><a name="p829417266523"></a>当bfloat16_mantissa尾数大于10位时，类似<strong id="b2294426155210"><a name="b2294426155210"></a><a name="b2294426155210"></a>例2</strong>的例子</p>
<p id="p3294102615218"><a name="p3294102615218"></a><a name="p3294102615218"></a>1）符号位S为0，既正数时舍入，</p>
<p id="p72941726155212"><a name="p72941726155212"></a><a name="p72941726155212"></a>例bfloat16_mantissa = 0b0.000011011<span>0</span><span>0xxx</span></p>
<p id="p12294102685218"><a name="p12294102685218"></a><a name="p12294102685218"></a>输出float16_mantissa = 0b0.000011011<span>1</span></p>
<p id="p32941126205210"><a name="p32941126205210"></a><a name="p32941126205210"></a>2）符号位S为1，既负数时不舍入</p>
<p id="p229422610527"><a name="p229422610527"></a><a name="p229422610527"></a>例bfloat16_mantissa = 0b0.000011011<span>0</span><span>0xxx</span></p>
<p id="p2294122675215"><a name="p2294122675215"></a><a name="p2294122675215"></a>输出float16_mantissa = 0b0.000011011<span>0</span></p>
<p id="p15574239105813"><a name="p15574239105813"></a><a name="p15574239105813"></a><strong id="b057419399580"><a name="b057419399580"></a><a name="b057419399580"></a>CAST_TRUNC：</strong>向0方向舍入</p>
<p id="p12779104548"><a name="p12779104548"></a><a name="p12779104548"></a>当bfloat16_mantissa尾数小于等于10位时，不需要舍入；</p>
<p id="p117751095418"><a name="p117751095418"></a><a name="p117751095418"></a>当bfloat16_mantissa尾数大于10位时，类似<strong id="b11771109544"><a name="b11771109544"></a><a name="b11771109544"></a>例2</strong>的例子</p>
<p id="p19344112945412"><a name="p19344112945412"></a><a name="p19344112945412"></a>1）直接舍弃多余精度，不舍入</p>
<p id="p1551302775416"><a name="p1551302775416"></a><a name="p1551302775416"></a>例bfloat16_mantissa = 0b0.000011011<span>0</span><span>0xxx</span></p>
<p id="p8513172717546"><a name="p8513172717546"></a><a name="p8513172717546"></a>输出float16_mantissa = 0b0.000011011<span>0</span></p>
<p id="p5207161418618"><a name="p5207161418618"></a><a name="p5207161418618"></a><strong id="b61451528566"><a name="b61451528566"></a><a name="b61451528566"></a>完整举例</strong>：</p>
<p id="p19694291663"><a name="p19694291663"></a><a name="p19694291663"></a>输入bfloat16为2.90573e-06，二进制为</p>
<p id="p166211313915"><a name="p166211313915"></a><a name="p166211313915"></a>sign:0, exponent:01101100, mantissa:1000011</p>
<p id="p21931943394"><a name="p21931943394"></a><a name="p21931943394"></a>bfloat16_exponent = 108-127 = -19 &lt; -14，float16_exponent最小为-14，所以bfloat16_exponent += 5，bfloat16_mantissa 小数点左移</p>
<p id="p527714569142"><a name="p527714569142"></a><a name="p527714569142"></a>5位，1.1000011 -&gt; 0.000011000<span>0</span><span>11</span></p>
<p id="p16811749171516"><a name="p16811749171516"></a><a name="p16811749171516"></a>CAST_RINT模式</p>
<p id="p181042231613"><a name="p181042231613"></a><a name="p181042231613"></a>舍弃精度<span>11</span>大于<span>舍弃精度中间值10</span>，所以进位</p>
<p id="p967216146178"><a name="p967216146178"></a><a name="p967216146178"></a>bfloat16_mantissa = 0.000011000<span>1</span></p>
<p id="p1175144081720"><a name="p1175144081720"></a><a name="p1175144081720"></a>最终float16二进制为</p>
<p id="p152072981810"><a name="p152072981810"></a><a name="p152072981810"></a>sign:0, exponent: 00000, mantissa = 000011000<span>1</span></p>
<p id="p186211383183"><a name="p186211383183"></a><a name="p186211383183"></a>按照计算公式十进制表示为</p>
<p id="p18468171519192"><a name="p18468171519192"></a><a name="p18468171519192"></a>float16=0.00000293</p>
</td>
</tr>
<tr id="row3408191131910"><td class="cellrowborder" valign="top" width="20.32203220322032%" headers="mcps1.2.4.1.1 "><p id="p7408411101913"><a name="p7408411101913"></a><a name="p7408411101913"></a>half</p>
</td>
<td class="cellrowborder" valign="top" width="16.53165316531653%" headers="mcps1.2.4.1.2 "><p id="p44081211141914"><a name="p44081211141914"></a><a name="p44081211141914"></a>bfloat16_t</p>
</td>
<td class="cellrowborder" valign="top" width="63.14631463146314%" headers="mcps1.2.4.1.3 "><p id="p1566203310518"><a name="p1566203310518"></a><a name="p1566203310518"></a>half二进制规则为</p>
<p id="p14665331754"><a name="p14665331754"></a><a name="p14665331754"></a>sign:1bit,exponent:5bit,mantissa:10bit</p>
<p id="p6663331853"><a name="p6663331853"></a><a name="p6663331853"></a>bfloat16_t二进制规则为</p>
<p id="p878094117514"><a name="p878094117514"></a><a name="p878094117514"></a>sign:1bit,exponent:8bit,mantissa:7bit</p>
<p id="p172420461557"><a name="p172420461557"></a><a name="p172420461557"></a><strong id="b2088013715321"><a name="b2088013715321"></a><a name="b2088013715321"></a>CAST_RINT：</strong>就近舍入，距离相同时向偶数舍入</p>
<p id="p0717339153413"><a name="p0717339153413"></a><a name="p0717339153413"></a>1）当舍弃精度小于<span>舍弃精度中间值100</span>时，不舍入</p>
<p id="p243918073612"><a name="p243918073612"></a><a name="p243918073612"></a>例half_mantinssa = 011101<span>0</span><span>011</span></p>
<p id="p2426744364"><a name="p2426744364"></a><a name="p2426744364"></a>输出bfloat16_mantissa = 011101<span>0</span></p>
<p id="p59967307353"><a name="p59967307353"></a><a name="p59967307353"></a>2）当舍弃精度大于<span>舍弃精度中间值100</span>时，舍入</p>
<p id="p0705153012364"><a name="p0705153012364"></a><a name="p0705153012364"></a>例half_mantinssa = 011101<span>0</span><span>111</span></p>
<p id="p9706103014362"><a name="p9706103014362"></a><a name="p9706103014362"></a>输出bfloat16_mantissa = 011101<span>1</span></p>
<p id="p1299734216366"><a name="p1299734216366"></a><a name="p1299734216366"></a>3）当舍入精度等于<span>舍弃精度中间值100</span>时，bfloat16最低位为偶数，不舍入</p>
<p id="p187582813720"><a name="p187582813720"></a><a name="p187582813720"></a>例half_mantinssa = 011101<span>0</span><span>100</span></p>
<p id="p587532816371"><a name="p587532816371"></a><a name="p587532816371"></a>输出bfloat16_mantissa = 011101<span>0</span></p>
<p id="p3260161504211"><a name="p3260161504211"></a><a name="p3260161504211"></a>4）当舍入精度等于<span>舍弃精度中间值100</span>时，bfloat16最低位为奇数，</p>
<p id="p16719183118389"><a name="p16719183118389"></a><a name="p16719183118389"></a>舍入</p>
<p id="p147191631153813"><a name="p147191631153813"></a><a name="p147191631153813"></a>例half_mantinssa = 011101<span>0</span><span>100</span></p>
<p id="p11720183110389"><a name="p11720183110389"></a><a name="p11720183110389"></a>输出bfloat16_mantissa = 011101<span>1</span></p>
<p id="p117710449306"><a name="p117710449306"></a><a name="p117710449306"></a><strong id="b143741843123218"><a name="b143741843123218"></a><a name="b143741843123218"></a>CAST_ROUND：</strong>就近舍入，距离相同时远离0舍入</p>
<p id="p7819825104116"><a name="p7819825104116"></a><a name="p7819825104116"></a>1）当舍弃精度小于<span>舍弃精度中间值100</span>时，不舍入</p>
<p id="p3819142513419"><a name="p3819142513419"></a><a name="p3819142513419"></a>例half_mantinssa = 011101<span>0</span><span>011</span></p>
<p id="p1581962516412"><a name="p1581962516412"></a><a name="p1581962516412"></a>输出bfloat16_mantissa = 011101<span>0</span></p>
<p id="p581952594113"><a name="p581952594113"></a><a name="p581952594113"></a>2）当舍弃精度大于<span>舍弃精度中间值100</span>时，舍入</p>
<p id="p5819182518416"><a name="p5819182518416"></a><a name="p5819182518416"></a>例half_mantinssa = 011101<span>0</span><span>111</span></p>
<p id="p281942544118"><a name="p281942544118"></a><a name="p281942544118"></a>输出bfloat16_mantissa = 011101<span>1</span></p>
<p id="p28191625114116"><a name="p28191625114116"></a><a name="p28191625114116"></a>3）当舍入精度等于<span>舍弃精度中间值100</span>时，进位远离0，所以舍入</p>
<p id="p18819202524119"><a name="p18819202524119"></a><a name="p18819202524119"></a>例half_mantinssa = 011101<span>0</span><span>100</span></p>
<p id="p1181992514411"><a name="p1181992514411"></a><a name="p1181992514411"></a>输出bfloat16_mantissa = 011101<span>1</span></p>
<p id="p38911357316"><a name="p38911357316"></a><a name="p38911357316"></a><strong id="b133101547183215"><a name="b133101547183215"></a><a name="b133101547183215"></a>CAST_FLOOR：</strong>向负无穷方向舍入</p>
<p id="p177251231174418"><a name="p177251231174418"></a><a name="p177251231174418"></a>1）当输入值为正数时，不舍入</p>
<p id="p16464521442"><a name="p16464521442"></a><a name="p16464521442"></a>例half_mantinssa = 011101<span>0</span><span>011</span></p>
<p id="p9646125224412"><a name="p9646125224412"></a><a name="p9646125224412"></a>输出bfloat16_mantissa = 011101<span>0</span></p>
<p id="p2881145418441"><a name="p2881145418441"></a><a name="p2881145418441"></a>2）当输入值为负数时，舍入</p>
<p id="p1571220624513"><a name="p1571220624513"></a><a name="p1571220624513"></a>例half_mantinssa = 011101<span>0</span><span>011</span></p>
<p id="p1771266184514"><a name="p1771266184514"></a><a name="p1771266184514"></a>输出bfloat16_mantissa = 011101<span>1</span></p>
<p id="p493618163312"><a name="p493618163312"></a><a name="p493618163312"></a><strong id="b129315504320"><a name="b129315504320"></a><a name="b129315504320"></a>CAST_CEIL：</strong>向正无穷方向舍入</p>
<p id="p159119401456"><a name="p159119401456"></a><a name="p159119401456"></a>1）当输入值为正数时，舍入</p>
<p id="p16591740144518"><a name="p16591740144518"></a><a name="p16591740144518"></a>例half_mantinssa = 011101<span>0</span><span>011</span></p>
<p id="p18591840194520"><a name="p18591840194520"></a><a name="p18591840194520"></a>输出bfloat16_mantissa = 011101<span>1</span></p>
<p id="p115918403457"><a name="p115918403457"></a><a name="p115918403457"></a>2）当输入值为负数时，不舍入</p>
<p id="p459134016458"><a name="p459134016458"></a><a name="p459134016458"></a>例half_mantinssa = 011101<span>0</span><span>011</span></p>
<p id="p1559110402457"><a name="p1559110402457"></a><a name="p1559110402457"></a>输出bfloat16_mantissa = 011101<span>0</span></p>
<p id="p159748327319"><a name="p159748327319"></a><a name="p159748327319"></a><strong id="b1632325419327"><a name="b1632325419327"></a><a name="b1632325419327"></a>CAST_TRUNC：</strong>向0方向舍入</p>
<p id="p228073413515"><a name="p228073413515"></a><a name="p228073413515"></a>直接舍弃多余精度</p>
<p id="p046359134614"><a name="p046359134614"></a><a name="p046359134614"></a>例half_mantinssa = 011101<span>0</span><span>011</span></p>
<p id="p1746311918460"><a name="p1746311918460"></a><a name="p1746311918460"></a>输出bfloat16_mantissa = 011101<span>0</span></p>
<p id="p16377173716467"><a name="p16377173716467"></a><a name="p16377173716467"></a><strong id="b19241848164618"><a name="b19241848164618"></a><a name="b19241848164618"></a>完整举例：</strong></p>
<p id="p18337125054615"><a name="p18337125054615"></a><a name="p18337125054615"></a>输入half 0.131</p>
<p id="p618817203473"><a name="p618817203473"></a><a name="p618817203473"></a>二进制为sign:0, exponent:01100, mantissa:0000110001</p>
<p id="p35211039683"><a name="p35211039683"></a><a name="p35211039683"></a>half_exponent = 12-15 = bfloat16_exp - 127</p>
<p id="p179052484720"><a name="p179052484720"></a><a name="p179052484720"></a>bfloat16_exp = 124 = 0b<span>1111100</span></p>
<p id="p1725210317116"><a name="p1725210317116"></a><a name="p1725210317116"></a>CAST_ROUND模式下</p>
<p id="p53398524112"><a name="p53398524112"></a><a name="p53398524112"></a>half_mantissa = 000011<span>0</span><span>001</span></p>
<p id="p7642420161213"><a name="p7642420161213"></a><a name="p7642420161213"></a>舍弃精度小于舍弃中间精度100，不进位，所以得到</p>
<p id="p15267121241314"><a name="p15267121241314"></a><a name="p15267121241314"></a>bfloat16_mantissa = 0000110</p>
<p id="p14392836131320"><a name="p14392836131320"></a><a name="p14392836131320"></a>所以bfloat16二进制为</p>
<p id="p290114217145"><a name="p290114217145"></a><a name="p290114217145"></a>sign:0, exponent:<span>1111100</span>, mantissa: 000011<span>0</span></p>
<p id="p1629435251513"><a name="p1629435251513"></a><a name="p1629435251513"></a>按照计算公式十进制表示为</p>
<p id="p103551853101514"><a name="p103551853101514"></a><a name="p103551853101514"></a>bfloat16=0.130859</p>
</td>
</tr>
<tr id="row87521387222"><td class="cellrowborder" valign="top" width="20.32203220322032%" headers="mcps1.2.4.1.1 "><p id="p197521038182215"><a name="p197521038182215"></a><a name="p197521038182215"></a>float</p>
</td>
<td class="cellrowborder" valign="top" width="16.53165316531653%" headers="mcps1.2.4.1.2 "><p id="p9752193852218"><a name="p9752193852218"></a><a name="p9752193852218"></a>fp8_e4m3fn_t</p>
</td>
<td class="cellrowborder" valign="top" width="63.14631463146314%" headers="mcps1.2.4.1.3 "><p id="p7832184583112"><a name="p7832184583112"></a><a name="p7832184583112"></a>float 二进制规则为</p>
<p id="p883215456314"><a name="p883215456314"></a><a name="p883215456314"></a>sign:1bit,exponent:8bit mantissa:23bit</p>
<p id="p397555417315"><a name="p397555417315"></a><a name="p397555417315"></a>fp8_e4m3fn_t二进制规则为</p>
<p id="p98322458315"><a name="p98322458315"></a><a name="p98322458315"></a>sign:1bit,exponent:4bit,mantissa:3bit</p>
<p id="p3501165715371"><a name="p3501165715371"></a><a name="p3501165715371"></a>需要舍弃20bit，<span>舍弃精度中间值</span>为0b10000000000000000000</p>
<p id="p9752103817226"><a name="p9752103817226"></a><a name="p9752103817226"></a><strong id="b12195244183215"><a name="b12195244183215"></a><a name="b12195244183215"></a>CAST_RINT：</strong>就近舍入，距离相同时向偶数舍入</p>
<p id="p15228104363611"><a name="p15228104363611"></a><a name="p15228104363611"></a>1）当舍弃精度小于<span>舍弃精度中间值</span>时，不舍入</p>
<p id="p1122864319368"><a name="p1122864319368"></a><a name="p1122864319368"></a>例float_mantinssa = 01<span>1</span><span>0010</span><span>011</span><span>0000000000000</span></p>
<p id="p172284433366"><a name="p172284433366"></a><a name="p172284433366"></a>输出fp8_e4m3fn_t_mantissa = 01<span>1</span></p>
<p id="p52281043163611"><a name="p52281043163611"></a><a name="p52281043163611"></a>2）当舍弃精度大于<span>舍弃精度中间值</span>时，舍入</p>
<p id="p722894393615"><a name="p722894393615"></a><a name="p722894393615"></a>例float_mantinssa= 01<span>1</span><span>1010</span><span>011</span><span>0000000000000</span></p>
<p id="p20228114319363"><a name="p20228114319363"></a><a name="p20228114319363"></a>输出fp8_e4m3fn_t_mantissa  = 10<span>0</span></p>
<p id="p9228134314367"><a name="p9228134314367"></a><a name="p9228134314367"></a>3）当舍入精度等于<span>舍弃精度中间值</span>时，fp8_e4m3fn_t最低位为偶数，不舍入</p>
<p id="p12285438366"><a name="p12285438366"></a><a name="p12285438366"></a>例float_mantinssa= 01<span>0</span><span>1000</span><span>000</span><span>0000000000000</span></p>
<p id="p722810436368"><a name="p722810436368"></a><a name="p722810436368"></a>输出fp8_e4m3fn_t_mantissa = 01<span>0</span></p>
<p id="p112287433362"><a name="p112287433362"></a><a name="p112287433362"></a>4）当舍入精度等于<span>舍弃精度中间值</span>时，fp8_e4m3fn_t最低位为奇数，</p>
<p id="p132280437366"><a name="p132280437366"></a><a name="p132280437366"></a>舍入</p>
<p id="p822864319365"><a name="p822864319365"></a><a name="p822864319365"></a>例float_mantinssa= 01<span>1</span><span>1000</span><span>000</span><span>0000000000000</span></p>
<p id="p1722811437369"><a name="p1722811437369"></a><a name="p1722811437369"></a>输出fp8_e4m3fn_t_mantissa = 10<span>0</span></p>
<p id="p1642511525419"><a name="p1642511525419"></a><a name="p1642511525419"></a><strong id="b156202101718"><a name="b156202101718"></a><a name="b156202101718"></a>完整例子：</strong></p>
<p id="p89981021205416"><a name="p89981021205416"></a><a name="p89981021205416"></a>输入float 1.233</p>
<p id="p3686639155614"><a name="p3686639155614"></a><a name="p3686639155614"></a>二进制为</p>
<p id="p799872114548"><a name="p799872114548"></a><a name="p799872114548"></a>sign:0, exponent:01111111, mantissa:00111011101001011110010</p>
<p id="p1599817213540"><a name="p1599817213540"></a><a name="p1599817213540"></a>float_exponent = 127-127 = fp8_e4m3fn_t_exp - 7</p>
<p id="p69983216549"><a name="p69983216549"></a><a name="p69983216549"></a>fp8_e4m3fn_t_exp = 7 = 0b<span>0111</span></p>
<p id="p59981521175416"><a name="p59981521175416"></a><a name="p59981521175416"></a>CAST_RINT模式下</p>
<p id="p199816210542"><a name="p199816210542"></a><a name="p199816210542"></a>float_mantissa = 00<span>1</span><span>11011101001011110010</span></p>
<p id="p19998182112549"><a name="p19998182112549"></a><a name="p19998182112549"></a>舍弃精度大于舍弃中间精度，进位，所以得到</p>
<p id="p1799862115419"><a name="p1799862115419"></a><a name="p1799862115419"></a>fp8_e4m3fn_t_mantissa = 01<span>0</span></p>
<p id="p1199852195414"><a name="p1199852195414"></a><a name="p1199852195414"></a>所以fp8_e4m3fn_t二进制为</p>
<p id="p3998132118544"><a name="p3998132118544"></a><a name="p3998132118544"></a>sign:0, exponent:<span>0111</span>, mantissa: 01<span>0</span></p>
<p id="p1799862185413"><a name="p1799862185413"></a><a name="p1799862185413"></a>按照计算公式十进制表示为</p>
<p id="p17998142185417"><a name="p17998142185417"></a><a name="p17998142185417"></a>fp8_e4m3fn_t=1.25</p>
</td>
</tr>
<tr id="row1761811391119"><td class="cellrowborder" valign="top" width="20.32203220322032%" headers="mcps1.2.4.1.1 "><p id="p2061816396117"><a name="p2061816396117"></a><a name="p2061816396117"></a>float</p>
</td>
<td class="cellrowborder" valign="top" width="16.53165316531653%" headers="mcps1.2.4.1.2 "><p id="p46181139115"><a name="p46181139115"></a><a name="p46181139115"></a>fp8_e5m2_t</p>
</td>
<td class="cellrowborder" valign="top" width="63.14631463146314%" headers="mcps1.2.4.1.3 "><p id="p26646114217"><a name="p26646114217"></a><a name="p26646114217"></a>float 二进制规则为</p>
<p id="p196641912023"><a name="p196641912023"></a><a name="p196641912023"></a>sign:1bit,exponent:8bit mantissa:23bit</p>
<p id="p196641912218"><a name="p196641912218"></a><a name="p196641912218"></a>fp8_e5m2_t二进制规则为</p>
<p id="p1966420115217"><a name="p1966420115217"></a><a name="p1966420115217"></a>sign:1bit,exponent:5bit,mantissa:2bit</p>
<p id="p10664111628"><a name="p10664111628"></a><a name="p10664111628"></a>需要舍弃21bit，<span>舍弃精度中间值</span>为0b100000000000000000000</p>
<p id="p9664611329"><a name="p9664611329"></a><a name="p9664611329"></a><strong id="b1166411115218"><a name="b1166411115218"></a><a name="b1166411115218"></a>CAST_RINT：</strong>就近舍入，距离相同时向偶数舍入</p>
<p id="p176651612211"><a name="p176651612211"></a><a name="p176651612211"></a>1）当舍弃精度小于<span>舍弃精度中间值</span>时，不舍入</p>
<p id="p1366514115214"><a name="p1366514115214"></a><a name="p1366514115214"></a>例float_mantinssa = 0<span>1</span><span>0</span><span>0010</span><span>011</span><span>0000000000000</span></p>
<p id="p146651012217"><a name="p146651012217"></a><a name="p146651012217"></a>输出fp8_e5m2_t_mantissa = 0<span>1</span></p>
<p id="p26653111214"><a name="p26653111214"></a><a name="p26653111214"></a>2）当舍弃精度大于<span>舍弃精度中间值</span>时，舍入</p>
<p id="p1766511118217"><a name="p1766511118217"></a><a name="p1766511118217"></a>例float_mantinssa= 0<span>1</span><span>1010</span><span>011</span><span>0000000000000</span><span>0</span></p>
<p id="p15665181826"><a name="p15665181826"></a><a name="p15665181826"></a>输出fp8_e5m2_t_mantissa  = 1<span>0</span></p>
<p id="p11665141927"><a name="p11665141927"></a><a name="p11665141927"></a>3）当舍入精度等于<span>舍弃精度中间值</span>时，fp8_e5m2_t最低位为偶数，不舍入</p>
<p id="p3665311215"><a name="p3665311215"></a><a name="p3665311215"></a>例float_mantinssa= 0<span>0</span><span>1000</span><span>000</span><span>00000000000000</span></p>
<p id="p26651415211"><a name="p26651415211"></a><a name="p26651415211"></a>输出fp8_e5m2_t_mantissa = 0<span>0</span></p>
<p id="p1666519110217"><a name="p1666519110217"></a><a name="p1666519110217"></a>4）当舍入精度等于<span>舍弃精度中间值</span>时，fp8_e5m2_t最低位为奇数，</p>
<p id="p126651411028"><a name="p126651411028"></a><a name="p126651411028"></a>舍入</p>
<p id="p5665112215"><a name="p5665112215"></a><a name="p5665112215"></a>例float_mantinssa= 0<span>1</span><span>1000</span><span>000</span><span>0000000000000</span></p>
<p id="p36650119211"><a name="p36650119211"></a><a name="p36650119211"></a>输出fp8_e5m2_t_mantissa = 1<span>0</span></p>
<p id="p7665211227"><a name="p7665211227"></a><a name="p7665211227"></a><strong id="b566516113212"><a name="b566516113212"></a><a name="b566516113212"></a>完整例子：</strong></p>
<p id="p566541723"><a name="p566541723"></a><a name="p566541723"></a>输入float32 1.233</p>
<p id="p86661712210"><a name="p86661712210"></a><a name="p86661712210"></a>二进制为</p>
<p id="p966691626"><a name="p966691626"></a><a name="p966691626"></a>sign:0, exponent:01111111, mantissa:00111011101001011110010</p>
<p id="p20666211213"><a name="p20666211213"></a><a name="p20666211213"></a>float_exponent = 127-127 = fp8_e5m2_t_exp - 15</p>
<p id="p566616114214"><a name="p566616114214"></a><a name="p566616114214"></a>fp8_e5m2_t_exp = 15 = 0b<span>01111</span></p>
<p id="p1166601429"><a name="p1166601429"></a><a name="p1166601429"></a>CAST_RINT模式下</p>
<p id="p26667117220"><a name="p26667117220"></a><a name="p26667117220"></a>float_mantissa = 0<span>0</span><span>1</span><span>11011101001011110010</span></p>
<p id="p66661213211"><a name="p66661213211"></a><a name="p66661213211"></a>舍弃精度大于舍弃中间精度，进位，所以得到</p>
<p id="p4666171424"><a name="p4666171424"></a><a name="p4666171424"></a>fp8_e5m2_t_mantissa = 0<span>1</span></p>
<p id="p766671029"><a name="p766671029"></a><a name="p766671029"></a>所以fp8_e5m2_t二进制为</p>
<p id="p56661911023"><a name="p56661911023"></a><a name="p56661911023"></a>sign:0, exponent:<span>01111</span>, mantissa: 0<span>1</span></p>
<p id="p566611322"><a name="p566611322"></a><a name="p566611322"></a>按照计算公式十进制表示为</p>
<p id="p206660115217"><a name="p206660115217"></a><a name="p206660115217"></a>fp8_e5m2_t=1.25</p>
</td>
</tr>
<tr id="row365719428111"><td class="cellrowborder" valign="top" width="20.32203220322032%" headers="mcps1.2.4.1.1 "><p id="p16581542151113"><a name="p16581542151113"></a><a name="p16581542151113"></a>bfloat16_t</p>
</td>
<td class="cellrowborder" valign="top" width="16.53165316531653%" headers="mcps1.2.4.1.2 "><p id="p0658842191111"><a name="p0658842191111"></a><a name="p0658842191111"></a>fp4x2_e2m1_t</p>
</td>
<td class="cellrowborder" valign="top" width="63.14631463146314%" headers="mcps1.2.4.1.3 "><p id="p118345661413"><a name="p118345661413"></a><a name="p118345661413"></a>bfloat16二进制规则为</p>
<p id="p883446141418"><a name="p883446141418"></a><a name="p883446141418"></a>sign:1bit,<span>exponent</span>:8bit,mantissa:7bit</p>
<p id="p1083436111413"><a name="p1083436111413"></a><a name="p1083436111413"></a>fp4x2_e2m1_t二进制规则为</p>
<p id="p98341164149"><a name="p98341164149"></a><a name="p98341164149"></a>sign:1bit,<span>exponent</span>:2bit,mantissa:1bit</p>
<p id="p1783518671418"><a name="p1783518671418"></a><a name="p1783518671418"></a>尾数位相差<span>6位</span></p>
<p id="p883519611416"><a name="p883519611416"></a><a name="p883519611416"></a>当bfloat16_exponent = <span>exponent</span>-127 &lt; 0时</p>
<p id="p1835960148"><a name="p1835960148"></a><a name="p1835960148"></a><strong id="b18835468143"><a name="b18835468143"></a><a name="b18835468143"></a>例1</strong>：当<span>exponent</span> = 124， bfloat16_exponent = <span>exponent</span>-127 = -3 &lt; 0时，因为float4_e2m1_exponent = <span>exponent</span> - 1 最小为0，无法表示，所以指数需要+3保持和float4_e2m1_exponent相等，同时尾数位需要除以2的3次方，因为bf16计算公式 = s * (2^(e-127))*(man) =</p>
<p id="p88351163149"><a name="p88351163149"></a><a name="p88351163149"></a>s * (2^(e-127+3))*(man*2^(-3)),指数位乘以一个数，尾数位需要除以相同大小的数，保证最终值不变。假设bfloat16_mantissa = 0b1011011</p>
<p id="p17835196121411"><a name="p17835196121411"></a><a name="p17835196121411"></a>bfloat16 mantissa计算公式为 1+ man/128, 用例二进制表示为</p>
<p id="p18835861144"><a name="p18835861144"></a><a name="p18835861144"></a>0b1.1011011，指数位乘以2的3次方时，尾数要除以2的3次，小数点向左移动3位，所以bfloat16_mantissa = 0b0.0<span>0110110</span><span>11</span>，因为float4_e2m1只有1bit尾数位，所以在以下CAST模式中需要舍弃最低9位，并根据舍入模式决定是否舍入，当前例子的<span>舍弃精度中间值</span>为0b<span>10</span><span>0000000</span></p>
<p id="p683517631417"><a name="p683517631417"></a><a name="p683517631417"></a><strong id="b18835366141"><a name="b18835366141"></a><a name="b18835366141"></a>CAST_RINT：</strong>就近舍入，距离相等时向偶数进位</p>
<p id="p48352610145"><a name="p48352610145"></a><a name="p48352610145"></a>1）被舍弃的精度小于<span>舍弃精度中间值</span>时，不进位，</p>
<p id="p58351620149"><a name="p58351620149"></a><a name="p58351620149"></a>例bfloat16_mantissa = 0b0.<span>0</span><span>000110xxx</span></p>
<p id="p283520681415"><a name="p283520681415"></a><a name="p283520681415"></a><span>000110xxx</span> &lt; <span>舍弃精度中间值</span> 输出float4_e2m1_mantissa = 0b0.<span>0</span></p>
<p id="p1783517617149"><a name="p1783517617149"></a><a name="p1783517617149"></a>2）被舍弃的精度大于<span>舍弃精度中间值</span>时，进位</p>
<p id="p583514616148"><a name="p583514616148"></a><a name="p583514616148"></a>例bfloat16_mantissa = 0b0.<span>0</span><span>100110xxx</span></p>
<p id="p137931513103111"><a name="p137931513103111"></a><a name="p137931513103111"></a><span>100110xxx</span> &gt; <span>舍弃精度中间值</span> 输出float4_e2m1_mantissa = 0b0.<span>1</span></p>
<p id="p108368611145"><a name="p108368611145"></a><a name="p108368611145"></a>3) 被舍弃的精度等于<span>舍弃精度中间值</span>时，<span>第1位</span>为偶数时不进位</p>
<p id="p1583615617147"><a name="p1583615617147"></a><a name="p1583615617147"></a>例bfloat16_mantissa = 0b0.<span>0</span><span>100000xxx</span></p>
<p id="p2083614681419"><a name="p2083614681419"></a><a name="p2083614681419"></a><span>0</span> % 2 == 0 输出float4_e2m1_mantissa = 0b0.<span>0</span></p>
<p id="p10836146151414"><a name="p10836146151414"></a><a name="p10836146151414"></a>4) 被舍弃的精度等于<span>舍弃精度中间值</span>时，<span>第1位</span>为奇数时进位</p>
<p id="p168360610148"><a name="p168360610148"></a><a name="p168360610148"></a>例bfloat16_mantissa = 0b0.<span>1</span><span>100000xxx</span></p>
<p id="p118360611147"><a name="p118360611147"></a><a name="p118360611147"></a><span>1</span> % 2 == 1 输出float4_e2m1_mantissa = 0b0.<span>1</span></p>
<p id="p1783610671416"><a name="p1783610671416"></a><a name="p1783610671416"></a><strong id="b1083610611149"><a name="b1083610611149"></a><a name="b1083610611149"></a>CAST_ROUND：</strong>就近舍入，距离相等时向远离0方向进位</p>
<p id="p768811420364"><a name="p768811420364"></a><a name="p768811420364"></a>1）被舍弃的精度小于<span>舍弃精度中间值</span>时，不进位，</p>
<p id="p868871413617"><a name="p868871413617"></a><a name="p868871413617"></a>例bfloat16_mantissa = 0b0.<span>0</span><span>000110xxx</span></p>
<p id="p106881014103612"><a name="p106881014103612"></a><a name="p106881014103612"></a><span>000110xxx</span> &lt; <span>舍弃精度中间值</span> 输出float4_e2m1_mantissa = 0b0.<span>0</span></p>
<p id="p86883144361"><a name="p86883144361"></a><a name="p86883144361"></a>2）被舍弃的精度大于<span>舍弃精度中间值</span>时，进位</p>
<p id="p126888147366"><a name="p126888147366"></a><a name="p126888147366"></a>例bfloat16_mantissa = 0b0.<span>0</span><span>100110xxx</span></p>
<p id="p13688141419364"><a name="p13688141419364"></a><a name="p13688141419364"></a><span>100110xxx</span> &gt; <span>舍弃精度中间值</span> 输出float4_e2m1_mantissa = 0b0.<span>1</span></p>
<p id="p10688131413368"><a name="p10688131413368"></a><a name="p10688131413368"></a>3) 被舍弃的精度等于<span>舍弃精度中间值</span>时，进位远离0，所以进位</p>
<p id="p10688181417360"><a name="p10688181417360"></a><a name="p10688181417360"></a>例bfloat16_mantissa = 0b0.<span>0</span><span>100000xxx</span></p>
<p id="p16688814163612"><a name="p16688814163612"></a><a name="p16688814163612"></a>输出float4_e2m1_mantissa = 0b0.<span>1</span></p>
<p id="p983715610143"><a name="p983715610143"></a><a name="p983715610143"></a><strong id="b083716101419"><a name="b083716101419"></a><a name="b083716101419"></a>CAST_FLOOR：</strong>向负无穷方向舍入</p>
<p id="p493581919375"><a name="p493581919375"></a><a name="p493581919375"></a>1）输入值正数时，不进位，</p>
<p id="p1493511198377"><a name="p1493511198377"></a><a name="p1493511198377"></a>例bfloat16_mantissa = 0b0.<span>0</span><span>000110xxx</span></p>
<p id="p7935919183710"><a name="p7935919183710"></a><a name="p7935919183710"></a>输出float4_e2m1_mantissa = 0b0.<span>0</span></p>
<p id="p693641983719"><a name="p693641983719"></a><a name="p693641983719"></a>2）输入值负数时，进位</p>
<p id="p6936319143713"><a name="p6936319143713"></a><a name="p6936319143713"></a>例bfloat16_mantissa = 0b0.<span>0</span><span>100110xxx</span></p>
<p id="p893611911379"><a name="p893611911379"></a><a name="p893611911379"></a>输出float4_e2m1_mantissa = 0b0.<span>1</span></p>
<p id="p58373601415"><a name="p58373601415"></a><a name="p58373601415"></a><strong id="b783796121412"><a name="b783796121412"></a><a name="b783796121412"></a>CAST_CEIL：</strong>向正无穷方向舍入</p>
<p id="p169981437385"><a name="p169981437385"></a><a name="p169981437385"></a>1）输入值正数时，进位，</p>
<p id="p39981434387"><a name="p39981434387"></a><a name="p39981434387"></a>例bfloat16_mantissa = 0b0.<span>0</span><span>000110xxx</span></p>
<p id="p179989316383"><a name="p179989316383"></a><a name="p179989316383"></a>输出float4_e2m1_mantissa = 0b0.<span>1</span></p>
<p id="p2998235386"><a name="p2998235386"></a><a name="p2998235386"></a>2）输入值负数时，不进位</p>
<p id="p169981343813"><a name="p169981343813"></a><a name="p169981343813"></a>例bfloat16_mantissa = 0b0.<span>0</span><span>100110xxx</span></p>
<p id="p69980319380"><a name="p69980319380"></a><a name="p69980319380"></a>输出float4_e2m1_mantissa = 0b0.<span>0</span></p>
<p id="p883810619147"><a name="p883810619147"></a><a name="p883810619147"></a><strong id="b11838106181416"><a name="b11838106181416"></a><a name="b11838106181416"></a>CAST_TRUNC：</strong>向0方向舍入</p>
<p id="p0838266148"><a name="p0838266148"></a><a name="p0838266148"></a>1）直接舍弃多余精度，不进位</p>
<p id="p1120124012381"><a name="p1120124012381"></a><a name="p1120124012381"></a>例bfloat16_mantissa = 0b0.<span>0</span><span>100110xxx</span></p>
<p id="p2201240133820"><a name="p2201240133820"></a><a name="p2201240133820"></a>输出float4_e2m1_mantissa = 0b0.<span>0</span></p>
<p id="p168387681414"><a name="p168387681414"></a><a name="p168387681414"></a><strong id="b683815651415"><a name="b683815651415"></a><a name="b683815651415"></a>完整举例</strong>：</p>
<p id="p8838106141416"><a name="p8838106141416"></a><a name="p8838106141416"></a>输入bfloat16为0.761719，二进制为</p>
<p id="p11838156161415"><a name="p11838156161415"></a><a name="p11838156161415"></a>sign:0, exponent:<span>01111110</span>, mantissa:1000011</p>
<p id="p583816191419"><a name="p583816191419"></a><a name="p583816191419"></a>bfloat16_exponent = <span>126</span>-127 = -1 &lt; 0，float16_exponent最小为0，所以bfloat16_exponent += 1，bfloat16_mantissa 小数点左移</p>
<p id="p78386651417"><a name="p78386651417"></a><a name="p78386651417"></a>1位，1.1000011 -&gt; 0.<span>1</span><span>1000011</span></p>
<p id="p68381063149"><a name="p68381063149"></a><a name="p68381063149"></a>CAST_RINT模式</p>
<p id="p118381462149"><a name="p118381462149"></a><a name="p118381462149"></a>舍弃精度<span>1000011</span>大于<span>舍弃精度中间值1000000</span>，所以进位</p>
<p id="p14838166121412"><a name="p14838166121412"></a><a name="p14838166121412"></a>float4_e2m1_mantissa = 0.<span>1</span></p>
<p id="p12838126111411"><a name="p12838126111411"></a><a name="p12838126111411"></a>最终float4_e2m1二进制为</p>
<p id="p0838196101414"><a name="p0838196101414"></a><a name="p0838196101414"></a>sign:0, exponent: 00, mantissa = <span>1</span></p>
<p id="p128388681417"><a name="p128388681417"></a><a name="p128388681417"></a>按照计算公式十进制表示为</p>
<p id="p1883917618147"><a name="p1883917618147"></a><a name="p1883917618147"></a>float4_e2m1=0.5</p>
</td>
</tr>
<tr id="row1654154318443"><td class="cellrowborder" valign="top" width="20.32203220322032%" headers="mcps1.2.4.1.1 "><p id="p65584344412"><a name="p65584344412"></a><a name="p65584344412"></a>bfloat16_t</p>
</td>
<td class="cellrowborder" valign="top" width="16.53165316531653%" headers="mcps1.2.4.1.2 "><p id="p165517438448"><a name="p165517438448"></a><a name="p165517438448"></a>fp4x2_e1m2_t</p>
</td>
<td class="cellrowborder" valign="top" width="63.14631463146314%" headers="mcps1.2.4.1.3 "><p id="p9903416124512"><a name="p9903416124512"></a><a name="p9903416124512"></a>bfloat16二进制规则为</p>
<p id="p6903111694516"><a name="p6903111694516"></a><a name="p6903111694516"></a>sign:1bit,<span>exponent</span>:8bit,mantissa:7bit</p>
<p id="p10903151619456"><a name="p10903151619456"></a><a name="p10903151619456"></a>fp4x2_e1m2_t二进制规则为</p>
<p id="p790316161450"><a name="p790316161450"></a><a name="p790316161450"></a>sign:1bit,<span>exponent</span>:1bit,mantissa:2bit</p>
<p id="p18903121654519"><a name="p18903121654519"></a><a name="p18903121654519"></a>尾数位相差<span>5位</span></p>
<p id="p13903191612456"><a name="p13903191612456"></a><a name="p13903191612456"></a>当bfloat16_exponent = <span>exponent</span>-127 &lt; 0时，bfloat16_mantissa需要发生位移</p>
<p id="p1590312169455"><a name="p1590312169455"></a><a name="p1590312169455"></a><strong id="b8903216144517"><a name="b8903216144517"></a><a name="b8903216144517"></a>例1</strong>：当<span>exponent</span> = 124， bfloat16_exponent = <span>exponent</span>-127 = -3 &lt; 0时，因为float4_e1m2_exponent = <span>exponent</span> - 1 最小为0，无法表示，所以指数需要+3保持和float4_e2m1_exponent相等，同时尾数位需要除以2的3次方，因为bf16计算公式 = s * (2^(e-127))*(man) =</p>
<p id="p2904111694510"><a name="p2904111694510"></a><a name="p2904111694510"></a>s * (2^(e-127+3))*(man*2^(-3)),指数位乘以一个数，尾数位需要除以相同大小的数，保证最终值不变。假设bfloat16_mantissa = 0b1011011</p>
<p id="p2904916184517"><a name="p2904916184517"></a><a name="p2904916184517"></a>bfloat16 mantissa计算公式为 1+ man/128, 用例二进制表示为</p>
<p id="p8904151613454"><a name="p8904151613454"></a><a name="p8904151613454"></a>0b1.1011011，指数位乘以2的3次方时，尾数要除以2的3次，小数点向左移动3位，所以bfloat16_mantissa = 0b0.00<span>110110</span><span>11</span>，因为float4_e2m1只有2bit尾数位，所以在以下CAST模式中需要舍弃最低8位，并根据舍入模式决定是否舍入，当前例子的<span>舍弃精度中间值</span>为0b<span>10</span><span>000000</span></p>
<p id="p49044165458"><a name="p49044165458"></a><a name="p49044165458"></a><strong id="b1790451620455"><a name="b1790451620455"></a><a name="b1790451620455"></a>CAST_RINT：</strong>就近舍入，距离相等时向偶数进位</p>
<p id="p1390413161454"><a name="p1390413161454"></a><a name="p1390413161454"></a>1）被舍弃的精度小于<span>舍弃精度中间值</span>时，不进位，</p>
<p id="p16904121619453"><a name="p16904121619453"></a><a name="p16904121619453"></a>例bfloat16_mantissa = 0b0.0<span>0</span><span>00011xxx</span></p>
<p id="p6904416174513"><a name="p6904416174513"></a><a name="p6904416174513"></a><span>00011xxx</span> &lt; <span>舍弃精度中间值</span> 输出float4_e1m2_mantissa = 0b0.0<span>0</span></p>
<p id="p119041516124511"><a name="p119041516124511"></a><a name="p119041516124511"></a>2）被舍弃的精度大于<span>舍弃精度中间值</span>时，进位</p>
<p id="p8904121664511"><a name="p8904121664511"></a><a name="p8904121664511"></a>例bfloat16_mantissa = 0b0.0<span>0</span><span>10011xxx</span></p>
<p id="p690431684515"><a name="p690431684515"></a><a name="p690431684515"></a><span>10011xxx</span> &gt; <span>舍弃精度中间值</span> 输出float4_e1m2_mantissa = 0b0.0<span>1</span></p>
<p id="p11904516104516"><a name="p11904516104516"></a><a name="p11904516104516"></a>3) 被舍弃的精度等于<span>舍弃精度中间值</span>时，<span>第2位</span>为偶数时不进位</p>
<p id="p590411164455"><a name="p590411164455"></a><a name="p590411164455"></a>例bfloat16_mantissa = 0b0.0<span>0</span><span>10000xxx</span></p>
<p id="p990431615459"><a name="p990431615459"></a><a name="p990431615459"></a><span>0</span> % 2 == 0 输出float4_e1m2_mantissa = 0b0.0<span>0</span></p>
<p id="p169056166452"><a name="p169056166452"></a><a name="p169056166452"></a>4) 被舍弃的精度等于<span>舍弃精度中间值</span>时，<span>第2位</span>为奇数时进位</p>
<p id="p29057165456"><a name="p29057165456"></a><a name="p29057165456"></a>例bfloat16_mantissa = 0b0.0<span>1</span><span>10000xxx</span></p>
<p id="p209055163457"><a name="p209055163457"></a><a name="p209055163457"></a><span>1</span> % 2 == 1 输出float4_e1m2_mantissa = 0b0.1<span>0</span></p>
<p id="p18905716204511"><a name="p18905716204511"></a><a name="p18905716204511"></a><strong id="b1590516164452"><a name="b1590516164452"></a><a name="b1590516164452"></a>CAST_ROUND：</strong>就近舍入，距离相等时向远离0方向进位</p>
<p id="p490571674515"><a name="p490571674515"></a><a name="p490571674515"></a>1）被舍弃的精度小于<span>舍弃精度中间值</span>时，不进位，</p>
<p id="p490520161455"><a name="p490520161455"></a><a name="p490520161455"></a>例bfloat16_mantissa = 0b0.0<span>0</span><span>00110xxx</span></p>
<p id="p590510162458"><a name="p590510162458"></a><a name="p590510162458"></a><span>00110xxx</span> &lt; <span>舍弃精度中间值</span> 输出float4_e1m2_mantissa = 0b0.0<span>0</span></p>
<p id="p14905131617456"><a name="p14905131617456"></a><a name="p14905131617456"></a>2）被舍弃的精度大于<span>舍弃精度中间值</span>时，进位</p>
<p id="p690581618458"><a name="p690581618458"></a><a name="p690581618458"></a>例bfloat16_mantissa = 0b0.0<span>0</span><span>10110xxx</span></p>
<p id="p1590591611458"><a name="p1590591611458"></a><a name="p1590591611458"></a><span>10110xxx</span> &gt; <span>舍弃精度中间值</span> 输出float4_e1m2_mantissa = 0b0.0<span>1</span></p>
<p id="p15905141634515"><a name="p15905141634515"></a><a name="p15905141634515"></a>3) 被舍弃的精度等于<span>舍弃精度中间值</span>时，进位远离0，所以进位</p>
<p id="p1590514166456"><a name="p1590514166456"></a><a name="p1590514166456"></a>例bfloat16_mantissa = 0b0.<span>00</span><span>10000xxx</span></p>
<p id="p9905516154513"><a name="p9905516154513"></a><a name="p9905516154513"></a>输出float4_e1m2_mantissa = 0b0.0<span>1</span></p>
<p id="p390511614453"><a name="p390511614453"></a><a name="p390511614453"></a><strong id="b139061916164518"><a name="b139061916164518"></a><a name="b139061916164518"></a>CAST_FLOOR：</strong>向负无穷方向舍入</p>
<p id="p129061416174512"><a name="p129061416174512"></a><a name="p129061416174512"></a>1）输入值正数时，不进位，</p>
<p id="p590641664513"><a name="p590641664513"></a><a name="p590641664513"></a>例bfloat16_mantissa = 0b0.0<span>0</span><span>00110xxx</span></p>
<p id="p9906616144512"><a name="p9906616144512"></a><a name="p9906616144512"></a>输出float4_e1m2_mantissa = 0b0.0<span>0</span></p>
<p id="p3906151620454"><a name="p3906151620454"></a><a name="p3906151620454"></a>2）输入值负数时，进位</p>
<p id="p13906516104515"><a name="p13906516104515"></a><a name="p13906516104515"></a>例bfloat16_mantissa = 0b0.0<span>0</span><span>100110xxx</span></p>
<p id="p090615169457"><a name="p090615169457"></a><a name="p090615169457"></a>输出float4_e1m2_mantissa = 0b0.0<span>1</span></p>
<p id="p890651619458"><a name="p890651619458"></a><a name="p890651619458"></a><strong id="b16906616164515"><a name="b16906616164515"></a><a name="b16906616164515"></a>CAST_CEIL：</strong>向正无穷方向舍入</p>
<p id="p129061816164513"><a name="p129061816164513"></a><a name="p129061816164513"></a>1）输入值正数时，进位，</p>
<p id="p1690619161456"><a name="p1690619161456"></a><a name="p1690619161456"></a>例bfloat16_mantissa = 0b0.0<span>0</span><span>00110xxx</span></p>
<p id="p15906416164519"><a name="p15906416164519"></a><a name="p15906416164519"></a>输出float4_e1m2_mantissa = 0b0.0<span>1</span></p>
<p id="p190641612458"><a name="p190641612458"></a><a name="p190641612458"></a>2）输入值负数时，不进位</p>
<p id="p39061167457"><a name="p39061167457"></a><a name="p39061167457"></a>例bfloat16_mantissa = 0b0.0<span>0</span><span>10110xxx</span></p>
<p id="p1490641616454"><a name="p1490641616454"></a><a name="p1490641616454"></a>输出float4_e1m2_mantissa = 0b0.0<span>0</span></p>
<p id="p7906916124518"><a name="p7906916124518"></a><a name="p7906916124518"></a><strong id="b3906171624518"><a name="b3906171624518"></a><a name="b3906171624518"></a>CAST_TRUNC：</strong>向0方向舍入</p>
<p id="p1490661614516"><a name="p1490661614516"></a><a name="p1490661614516"></a>1）直接舍弃多余精度，不进位</p>
<p id="p20906616154510"><a name="p20906616154510"></a><a name="p20906616154510"></a>例bfloat16_mantissa = 0b0.0<span>0</span><span>10110xxx</span></p>
<p id="p10906151684519"><a name="p10906151684519"></a><a name="p10906151684519"></a>输出float4_e1m2_mantissa = 0b0.0<span>0</span></p>
<p id="p149061016134516"><a name="p149061016134516"></a><a name="p149061016134516"></a><strong id="b20907316174512"><a name="b20907316174512"></a><a name="b20907316174512"></a>完整举例</strong>：</p>
<p id="p49071216174514"><a name="p49071216174514"></a><a name="p49071216174514"></a>输入bfloat16为0.761719，二进制为</p>
<p id="p49078166455"><a name="p49078166455"></a><a name="p49078166455"></a>sign:0, exponent:<span>01111110</span>, mantissa:1000011</p>
<p id="p14907151674516"><a name="p14907151674516"></a><a name="p14907151674516"></a>bfloat16_exponent = <span>126</span>-127 = -1 &lt; 0，float16_exponent最小为0，所以bfloat16_exponent += 1，bfloat16_mantissa 小数点左移</p>
<p id="p690717162451"><a name="p690717162451"></a><a name="p690717162451"></a>1位，1.1000011 -&gt; 0.1<span>1</span><span>000011</span></p>
<p id="p1890711664518"><a name="p1890711664518"></a><a name="p1890711664518"></a>CAST_RINT模式</p>
<p id="p11907916184513"><a name="p11907916184513"></a><a name="p11907916184513"></a>舍弃精度<span>000011</span>小于<span>舍弃精度中间值100000</span>，所以不进位</p>
<p id="p1990720169459"><a name="p1990720169459"></a><a name="p1990720169459"></a>float4_e2m1_mantissa = 0.1<span>1</span></p>
<p id="p17907191611451"><a name="p17907191611451"></a><a name="p17907191611451"></a>最终float4_e2m1二进制为</p>
<p id="p290771614511"><a name="p290771614511"></a><a name="p290771614511"></a>sign:0, exponent: 0, mantissa = 1<span>1</span></p>
<p id="p29071916124510"><a name="p29071916124510"></a><a name="p29071916124510"></a>按照计算公式十进制表示为</p>
<p id="p14907016164513"><a name="p14907016164513"></a><a name="p14907016164513"></a>float4_e2m1=0.75</p>
</td>
</tr>
<tr id="row7256205016311"><td class="cellrowborder" valign="top" width="20.32203220322032%" headers="mcps1.2.4.1.1 "><p id="p32561501937"><a name="p32561501937"></a><a name="p32561501937"></a>float</p>
</td>
<td class="cellrowborder" valign="top" width="16.53165316531653%" headers="mcps1.2.4.1.2 "><p id="p1425615501937"><a name="p1425615501937"></a><a name="p1425615501937"></a>hifloat8_t</p>
</td>
<td class="cellrowborder" valign="top" width="63.14631463146314%" headers="mcps1.2.4.1.3 "><p id="p025713506311"><a name="p025713506311"></a><a name="p025713506311"></a>参考<a href="#table1352142520363">表9</a>。</p>
</td>
</tr>
</tbody>
</table>

**表 9**  float to hifloat8\_t类型转换规则

<a name="table1352142520363"></a>
<table><thead align="left"><tr id="row1035382513367"><th class="cellrowborder" colspan="2" valign="top" id="mcps1.2.3.1.1"><p id="p2926133119405"><a name="p2926133119405"></a><a name="p2926133119405"></a>VCVTFF:F322HiF8</p>
</th>
</tr>
</thead>
<tbody><tr id="row13353132573618"><td class="cellrowborder" colspan="2" valign="top" headers="mcps1.2.3.1.1 "><p id="p973652319239"><a name="p973652319239"></a><a name="p973652319239"></a><em id="i1973710231235"><a name="i1973710231235"></a><a name="i1973710231235"></a>tmp</em>2[31 : 0] = <em id="i177371123152313"><a name="i177371123152313"></a><a name="i177371123152313"></a>f32_src_data</em>[<em id="i17210815291"><a name="i17210815291"></a><a name="i17210815291"></a>i</em>][31 : 0];</p>
<p id="p20737132322312"><a name="p20737132322312"></a><a name="p20737132322312"></a><em id="i187371423122316"><a name="i187371423122316"></a><a name="i187371423122316"></a>E<sub id="sub16737102382313"><a name="sub16737102382313"></a><a name="sub16737102382313"></a>v</sub></em> = <em id="i1673722312313"><a name="i1673722312313"></a><a name="i1673722312313"></a>tmp</em>2[30 : 23] - 8'<em id="i147379230231"><a name="i147379230231"></a><a name="i147379230231"></a>b</em>01111111, <em id="i13737112372310"><a name="i13737112372310"></a><a name="i13737112372310"></a>thr</em> = <em id="i573722317236"><a name="i573722317236"></a><a name="i573722317236"></a>tmp</em>2[13 : 0];</p>
<p id="p87371723162315"><a name="p87371723162315"></a><a name="p87371723162315"></a>if (<em id="i167371923182320"><a name="i167371923182320"></a><a name="i167371923182320"></a>E<sub id="sub4737202342318"><a name="sub4737202342318"></a><a name="sub4737202342318"></a>v</sub></em> == 128 &amp;&amp; <em id="i9737122382315"><a name="i9737122382315"></a><a name="i9737122382315"></a>tmp</em>2[22 : 0] != 23'<em id="i1737523132318"><a name="i1737523132318"></a><a name="i1737523132318"></a>b</em>0) then</p>
<p id="p7737423182313"><a name="p7737423182313"></a><a name="p7737423182313"></a><em id="i1773718238239"><a name="i1773718238239"></a><a name="i1773718238239"></a>tmp</em>3[7 : 0] = <em id="i15737132312316"><a name="i15737132312316"></a><a name="i15737132312316"></a>HiF8</em><sub id="sub17737823192319"><a name="sub17737823192319"></a><a name="sub17737823192319"></a><em id="i1515010436291"><a name="i1515010436291"></a><a name="i1515010436291"></a>NAN</em></sub>(8'<em id="i187371323152320"><a name="i187371323152320"></a><a name="i187371323152320"></a>b</em>10000000);</p>
<p id="p1673752314237"><a name="p1673752314237"></a><a name="p1673752314237"></a>else if ((<em id="i187376236239"><a name="i187376236239"></a><a name="i187376236239"></a>E<sub id="sub473762313233"><a name="sub473762313233"></a><a name="sub473762313233"></a>v</sub></em> == 128 &amp;&amp; <em id="i97375237233"><a name="i97375237233"></a><a name="i97375237233"></a>tmp</em>2[22 : 0] == 23'<em id="i1273714233231"><a name="i1273714233231"></a><a name="i1273714233231"></a>b</em>0) || (<em id="i9737192342319"><a name="i9737192342319"></a><a name="i9737192342319"></a>E<sub id="sub19737132317235"><a name="sub19737132317235"></a><a name="sub19737132317235"></a>v</sub></em> &gt; 15)) then</p>
<p id="p1737142317230"><a name="p1737142317230"></a><a name="p1737142317230"></a><em id="i67370234234"><a name="i67370234234"></a><a name="i67370234234"></a>tmp</em>3[7 : 0] = (<em id="i673772315230"><a name="i673772315230"></a><a name="i673772315230"></a>tmp</em>2[31] == 1'<em id="i873719237231"><a name="i873719237231"></a><a name="i873719237231"></a>b</em>0) ? <em id="i473716235236"><a name="i473716235236"></a><a name="i473716235236"></a>HiF8<sub id="sub1873782382318"><a name="sub1873782382318"></a><a name="sub1873782382318"></a>+INF </sub></em>(8'<em id="i107378232235"><a name="i107378232235"></a><a name="i107378232235"></a>b</em>01101111) : <em id="i9737152302315"><a name="i9737152302315"></a><a name="i9737152302315"></a>HiF8<sub id="sub14737102315230"><a name="sub14737102315230"></a><a name="sub14737102315230"></a>-INF</sub></em> (8'<em id="i16737023102316"><a name="i16737023102316"></a><a name="i16737023102316"></a>b</em>11101111);</p>
<p id="p1573711231230"><a name="p1573711231230"></a><a name="p1573711231230"></a>else if (<em id="i373782311239"><a name="i373782311239"></a><a name="i373782311239"></a>E<sub id="sub1973712314234"><a name="sub1973712314234"></a><a name="sub1973712314234"></a>v</sub></em> &lt; -23) then</p>
<p id="p4737323102314"><a name="p4737323102314"></a><a name="p4737323102314"></a><em id="i073772312232"><a name="i073772312232"></a><a name="i073772312232"></a>tmp</em>3[7 : 0] = <em id="i115881521171"><a name="i115881521171"></a><a name="i115881521171"></a>HiF</em>8<em id="i3589192111710"><a name="i3589192111710"></a><a name="i3589192111710"></a><sub id="sub145891291715"><a name="sub145891291715"></a><a name="sub145891291715"></a>ZERO</sub></em> (8'<em id="i673712322311"><a name="i673712322311"></a><a name="i673712322311"></a>b</em>00000000);</p>
<p id="p57371623172317"><a name="p57371623172317"></a><a name="p57371623172317"></a>else if (<em id="i67372235232"><a name="i67372235232"></a><a name="i67372235232"></a>E<sub id="sub8737132362316"><a name="sub8737132362316"></a><a name="sub8737132362316"></a>v</sub></em> == -23) then</p>
<p id="p6737323182311"><a name="p6737323182311"></a><a name="p6737323182311"></a>if ((Half To Away Round) || (Hybrid Round &amp;&amp; {1’<em id="i6737323172312"><a name="i6737323172312"></a><a name="i6737323172312"></a>b</em>1, <em id="i5737132310236"><a name="i5737132310236"></a><a name="i5737132310236"></a>tmp</em>2[22 : 10]} &gt;= <em id="i67371523102314"><a name="i67371523102314"></a><a name="i67371523102314"></a>thr</em>)) then</p>
<p id="p1173762313236"><a name="p1173762313236"></a><a name="p1173762313236"></a><em id="i3737102342318"><a name="i3737102342318"></a><a name="i3737102342318"></a>tmp</em>3[7 : 0] = (<em id="i1373712302316"><a name="i1373712302316"></a><a name="i1373712302316"></a>tmp</em>2[31] == 1'<em id="i1973711235239"><a name="i1973711235239"></a><a name="i1973711235239"></a>b</em>0) ? 8'<em id="i12737102302316"><a name="i12737102302316"></a><a name="i12737102302316"></a>b</em>00000001 : 8'<em id="i16737162362314"><a name="i16737162362314"></a><a name="i16737162362314"></a>b</em>10000001; // min subnormal</p>
<p id="p107371823152312"><a name="p107371823152312"></a><a name="p107371823152312"></a>else</p>
<p id="p11737182314233"><a name="p11737182314233"></a><a name="p11737182314233"></a><em id="i47372230232"><a name="i47372230232"></a><a name="i47372230232"></a>tmp</em>3[7 : 0] = <em id="i473772352316"><a name="i473772352316"></a><a name="i473772352316"></a>HiF8<sub id="sub6737152319231"><a name="sub6737152319231"></a><a name="sub6737152319231"></a>ZERO</sub></em> (8'<em id="i1073782382316"><a name="i1073782382316"></a><a name="i1073782382316"></a>b</em>00000000);</p>
<p id="p12737192320235"><a name="p12737192320235"></a><a name="p12737192320235"></a>end if</p>
<p id="p173732319237"><a name="p173732319237"></a><a name="p173732319237"></a>else</p>
<p id="p473720232230"><a name="p473720232230"></a><a name="p473720232230"></a>if (<em id="i7737172319236"><a name="i7737172319236"></a><a name="i7737172319236"></a>E<sub id="sub3737723122317"><a name="sub3737723122317"></a><a name="sub3737723122317"></a>v</sub></em> == 0) then</p>
<p id="p18737523152319"><a name="p18737523152319"></a><a name="p18737523152319"></a><em id="i2099265312251"><a name="i2099265312251"></a><a name="i2099265312251"></a>M</em> = <em id="i77375238233"><a name="i77375238233"></a><a name="i77375238233"></a>tmp</em>2[22 : 20], <em id="i7737123152310"><a name="i7737123152310"></a><a name="i7737123152310"></a>T</em> <em id="i8737223122318"><a name="i8737223122318"></a><a name="i8737223122318"></a>A_bit</em> = <em id="i1737162382316"><a name="i1737162382316"></a><a name="i1737162382316"></a>tmp</em>2[19], <em id="i14737723192311"><a name="i14737723192311"></a><a name="i14737723192311"></a>frac</em> = <em id="i7737192332314"><a name="i7737192332314"></a><a name="i7737192332314"></a>tmp</em>2[19 : 6];</p>
<p id="p1873717230238"><a name="p1873717230238"></a><a name="p1873717230238"></a>else if (<em id="i673712317238"><a name="i673712317238"></a><a name="i673712317238"></a>E<sub id="sub97371223132315"><a name="sub97371223132315"></a><a name="sub97371223132315"></a>v</sub></em> == &plusmn;1) then</p>
<p id="p167377235237"><a name="p167377235237"></a><a name="p167377235237"></a><em id="i18442152102515"><a name="i18442152102515"></a><a name="i18442152102515"></a>M</em> = <em id="i5737132310230"><a name="i5737132310230"></a><a name="i5737132310230"></a>tmp</em>2[22 : 20], <em id="i12737182316236"><a name="i12737182316236"></a><a name="i12737182316236"></a>T</em> <em id="i1973742362316"><a name="i1973742362316"></a><a name="i1973742362316"></a>A_bit</em> = <em id="i773714231238"><a name="i773714231238"></a><a name="i773714231238"></a>tmp</em>2[19], <em id="i7737162312319"><a name="i7737162312319"></a><a name="i7737162312319"></a>frac</em> = <em id="i1737122342310"><a name="i1737122342310"></a><a name="i1737122342310"></a>tmp</em>2[19 : 6];</p>
<p id="p14737202310233"><a name="p14737202310233"></a><a name="p14737202310233"></a>else if (<em id="i1373712314238"><a name="i1373712314238"></a><a name="i1373712314238"></a>E<sub id="sub073732312318"><a name="sub073732312318"></a><a name="sub073732312318"></a>v</sub></em> == &plusmn;[2, 3]) then</p>
<p id="p4737162320239"><a name="p4737162320239"></a><a name="p4737162320239"></a><em id="i1337195142514"><a name="i1337195142514"></a><a name="i1337195142514"></a>M</em> = <em id="i16737223132318"><a name="i16737223132318"></a><a name="i16737223132318"></a>tmp</em>2[22 : 20], <em id="i12737823152312"><a name="i12737823152312"></a><a name="i12737823152312"></a>T</em> <em id="i1737423152312"><a name="i1737423152312"></a><a name="i1737423152312"></a>A_bit</em> = <em id="i1773712316233"><a name="i1773712316233"></a><a name="i1773712316233"></a>tmp</em>2[19], <em id="i1473722314235"><a name="i1473722314235"></a><a name="i1473722314235"></a>frac</em> = <em id="i16737523132318"><a name="i16737523132318"></a><a name="i16737523132318"></a>tmp</em>2[19:6];</p>
<p id="p27371223112313"><a name="p27371223112313"></a><a name="p27371223112313"></a>else if (<em id="i137372023162320"><a name="i137372023162320"></a><a name="i137372023162320"></a>E<sub id="sub197374231233"><a name="sub197374231233"></a><a name="sub197374231233"></a>v</sub></em> == &plusmn;[4, 7]) then</p>
<p id="p1073742319238"><a name="p1073742319238"></a><a name="p1073742319238"></a><em id="i3671204918259"><a name="i3671204918259"></a><a name="i3671204918259"></a>M</em> = <em id="i18737623142311"><a name="i18737623142311"></a><a name="i18737623142311"></a>tmp</em>2[22 : 21], <em id="i973712233233"><a name="i973712233233"></a><a name="i973712233233"></a>T</em> <em id="i573716232233"><a name="i573716232233"></a><a name="i573716232233"></a>A_bit</em> = <em id="i167379233232"><a name="i167379233232"></a><a name="i167379233232"></a>tmp</em>2[20], <em id="i1073722352314"><a name="i1073722352314"></a><a name="i1073722352314"></a>frac</em> = <em id="i673752302317"><a name="i673752302317"></a><a name="i673752302317"></a>tmp</em>2[20 : 7];</p>
<p id="p1273752317238"><a name="p1273752317238"></a><a name="p1273752317238"></a>else if (<em id="i137371423122313"><a name="i137371423122313"></a><a name="i137371423122313"></a>E<sub id="sub14737823152314"><a name="sub14737823152314"></a><a name="sub14737823152314"></a>v</sub></em> == &plusmn;[8, 15]) then</p>
<p id="p167376233232"><a name="p167376233232"></a><a name="p167376233232"></a><em id="i9109134812511"><a name="i9109134812511"></a><a name="i9109134812511"></a>M</em> = <em id="i873712232237"><a name="i873712232237"></a><a name="i873712232237"></a>tmp</em>2[22], <em id="i8737623142319"><a name="i8737623142319"></a><a name="i8737623142319"></a>T</em> <em id="i19738223142320"><a name="i19738223142320"></a><a name="i19738223142320"></a>A_bit</em> = <em id="i3738112314232"><a name="i3738112314232"></a><a name="i3738112314232"></a>tmp</em>2[21], <em id="i1773812231231"><a name="i1773812231231"></a><a name="i1773812231231"></a>frac</em> = <em id="i5738623172317"><a name="i5738623172317"></a><a name="i5738623172317"></a>tmp</em>2[21 : 8];</p>
<p id="p273812313238"><a name="p273812313238"></a><a name="p273812313238"></a>else if (<em id="i07381623172318"><a name="i07381623172318"></a><a name="i07381623172318"></a>E<sub id="sub187381823152311"><a name="sub187381823152311"></a><a name="sub187381823152311"></a>v</sub></em> == [-16, -22]) then</p>
<p id="p1073842316232"><a name="p1073842316232"></a><a name="p1073842316232"></a><em id="i1673817232233"><a name="i1673817232233"></a><a name="i1673817232233"></a>M</em> = <em id="i2073802320234"><a name="i2073802320234"></a><a name="i2073802320234"></a>E<sub id="sub873832311234"><a name="sub873832311234"></a><a name="sub873832311234"></a>v</sub></em> + 23, <em id="i9738182332316"><a name="i9738182332316"></a><a name="i9738182332316"></a>T</em> <em id="i19738162382319"><a name="i19738162382319"></a><a name="i19738162382319"></a>A_bit</em> = <em id="i12738142317239"><a name="i12738142317239"></a><a name="i12738142317239"></a>tmp</em>2[22], <em id="i1573872362316"><a name="i1573872362316"></a><a name="i1573872362316"></a>frac</em> = <em id="i16738323142320"><a name="i16738323142320"></a><a name="i16738323142320"></a>tmp</em>2[22 : 9]; // subnormal</p>
<p id="p1773892322319"><a name="p1773892322319"></a><a name="p1773892322319"></a>end if</p>
<p id="p14738182310233"><a name="p14738182310233"></a><a name="p14738182310233"></a>if (<em id="i0738162312318"><a name="i0738162312318"></a><a name="i0738162312318"></a>E<sub id="sub18738923132318"><a name="sub18738923132318"></a><a name="sub18738923132318"></a>v</sub></em> == &plusmn;[0, 3]) then</p>
<p id="p1373817238236"><a name="p1373817238236"></a><a name="p1373817238236"></a>if (<em id="i16219137173115"><a name="i16219137173115"></a><a name="i16219137173115"></a>T</em> <em id="i173819238237"><a name="i173819238237"></a><a name="i173819238237"></a>A_bit</em> == 1’<em id="i18738923152317"><a name="i18738923152317"></a><a name="i18738923152317"></a>b</em>1) then</p>
<p id="p177381323122317"><a name="p177381323122317"></a><a name="p177381323122317"></a><em id="i1473882318232"><a name="i1473882318232"></a><a name="i1473882318232"></a>M_tmp</em> = <em id="i8738162332316"><a name="i8738162332316"></a><a name="i8738162332316"></a>M</em> + 1, <em id="i873862316233"><a name="i873862316233"></a><a name="i873862316233"></a>E<sub id="sub1073852314233"><a name="sub1073852314233"></a><a name="sub1073852314233"></a>v</sub></em> = <em id="i8738172302312"><a name="i8738172302312"></a><a name="i8738172302312"></a>E<sub id="sub19738723142316"><a name="sub19738723142316"></a><a name="sub19738723142316"></a>v</sub></em> + <em id="i373882322318"><a name="i373882322318"></a><a name="i373882322318"></a>carry</em> <em id="i1173832317238"><a name="i1173832317238"></a><a name="i1173832317238"></a>of</em> <em id="i19738723112313"><a name="i19738723112313"></a><a name="i19738723112313"></a>M_tmp</em>, <em id="i17738162313234"><a name="i17738162313234"></a><a name="i17738162313234"></a>M</em> = (<em id="i27381237234"><a name="i27381237234"></a><a name="i27381237234"></a>carry of M_tmp</em>) ? 0 : <em id="i67382233236"><a name="i67382233236"></a><a name="i67382233236"></a>M_tmp</em>;</p>
<p id="p4738123132317"><a name="p4738123132317"></a><a name="p4738123132317"></a>end if</p>
<p id="p187381237230"><a name="p187381237230"></a><a name="p187381237230"></a>else if (<em id="i173822311238"><a name="i173822311238"></a><a name="i173822311238"></a>E<sub id="sub573832310231"><a name="sub573832310231"></a><a name="sub573832310231"></a>v</sub></em> == &plusmn;[4, 15]) then</p>
<p id="p873802372317"><a name="p873802372317"></a><a name="p873802372317"></a>if (HALF To Away Round &amp;&amp; <em id="i177381123172318"><a name="i177381123172318"></a><a name="i177381123172318"></a>T</em> <em id="i173822342310"><a name="i173822342310"></a><a name="i173822342310"></a>A_bit</em> == 1’<em id="i3738423192319"><a name="i3738423192319"></a><a name="i3738423192319"></a>b</em>1) || (Hybrid Round &amp;&amp; <em id="i3738223182313"><a name="i3738223182313"></a><a name="i3738223182313"></a>frac</em> &gt;= <em id="i17385238235"><a name="i17385238235"></a><a name="i17385238235"></a>thr</em>) then</p>
<p id="p19738112372310"><a name="p19738112372310"></a><a name="p19738112372310"></a><em id="i17738122314239"><a name="i17738122314239"></a><a name="i17738122314239"></a>M_tmp</em> = <em id="i5913111203211"><a name="i5913111203211"></a><a name="i5913111203211"></a>M</em> + 1, <em id="i4738202332310"><a name="i4738202332310"></a><a name="i4738202332310"></a>E<sub id="sub1073818235239"><a name="sub1073818235239"></a><a name="sub1073818235239"></a>v</sub></em> = <em id="i2738523102320"><a name="i2738523102320"></a><a name="i2738523102320"></a>E<sub id="sub87381923172311"><a name="sub87381923172311"></a><a name="sub87381923172311"></a>v</sub></em> + <em id="i1073812239235"><a name="i1073812239235"></a><a name="i1073812239235"></a>carry</em> <em id="i2738112314232"><a name="i2738112314232"></a><a name="i2738112314232"></a>of</em> <em id="i20738152314233"><a name="i20738152314233"></a><a name="i20738152314233"></a>M_tmp</em>, <em id="i107386237239"><a name="i107386237239"></a><a name="i107386237239"></a>M</em> = (<em id="i207381723102313"><a name="i207381723102313"></a><a name="i207381723102313"></a>carry of M_tmp</em>) ? 0 : <em id="i117382023172315"><a name="i117382023172315"></a><a name="i117382023172315"></a>M_tmp</em>;</p>
<p id="p2837161622714"><a name="p2837161622714"></a><a name="p2837161622714"></a>end if</p>
<p id="p87381523142318"><a name="p87381523142318"></a><a name="p87381523142318"></a>else if (<em id="i14738182382312"><a name="i14738182382312"></a><a name="i14738182382312"></a>E<sub id="sub1273892342310"><a name="sub1273892342310"></a><a name="sub1273892342310"></a>v</sub></em> == [-16, -22]) then</p>
<p id="p1173872316235"><a name="p1173872316235"></a><a name="p1173872316235"></a>if ((Half To Away Round) &amp;&amp; <em id="i47387237234"><a name="i47387237234"></a><a name="i47387237234"></a>T</em> <em id="i16738823142312"><a name="i16738823142312"></a><a name="i16738823142312"></a>A_bit</em> == 1'<em id="i12738112313232"><a name="i12738112313232"></a><a name="i12738112313232"></a>b</em>1) || ((Hybrid Round &amp;&amp; <em id="i5738112392313"><a name="i5738112392313"></a><a name="i5738112392313"></a>frac</em> &gt;= <em id="i2073822382313"><a name="i2073822382313"></a><a name="i2073822382313"></a>thr</em>) then</p>
<p id="p773802311239"><a name="p773802311239"></a><a name="p773802311239"></a><em id="i167382023152311"><a name="i167382023152311"></a><a name="i167382023152311"></a>M</em>_<em id="i1773832322315"><a name="i1773832322315"></a><a name="i1773832322315"></a>tmp</em> = <em id="i12738132311234"><a name="i12738132311234"></a><a name="i12738132311234"></a>E<sub id="sub147381623162310"><a name="sub147381623162310"></a><a name="sub147381623162310"></a>v</sub></em> + 23, <em id="i12738162372316"><a name="i12738162372316"></a><a name="i12738162372316"></a>E<sub id="sub17385239236"><a name="sub17385239236"></a><a name="sub17385239236"></a>v</sub></em> = <em id="i20738023202314"><a name="i20738023202314"></a><a name="i20738023202314"></a>E<sub id="sub12738142313233"><a name="sub12738142313233"></a><a name="sub12738142313233"></a>v</sub></em> + 1, M = (<em id="i27385236231"><a name="i27385236231"></a><a name="i27385236231"></a>E<sub id="sub0738122332312"><a name="sub0738122332312"></a><a name="sub0738122332312"></a>v</sub></em> == -15) ? 0 : <em id="i4738102311237"><a name="i4738102311237"></a><a name="i4738102311237"></a>M_tmp</em>;</p>
<p id="p3738152317230"><a name="p3738152317230"></a><a name="p3738152317230"></a>end if</p>
<p id="p573812319237"><a name="p573812319237"></a><a name="p573812319237"></a>end if</p>
<p id="p1773816236231"><a name="p1773816236231"></a><a name="p1773816236231"></a>encode {<em id="i1731218427320"><a name="i1731218427320"></a><a name="i1731218427320"></a>tmp</em>3[31], <em id="i7738122342317"><a name="i7738122342317"></a><a name="i7738122342317"></a>E<sub id="sub13738152315236"><a name="sub13738152315236"></a><a name="sub13738152315236"></a>v</sub></em>, <em id="i873832302315"><a name="i873832302315"></a><a name="i873832302315"></a>M</em>} to <em id="i473815236232"><a name="i473815236232"></a><a name="i473815236232"></a>tmp</em>3[7 : 0] following HiF encoding rule;</p>
<p id="p27381323192311"><a name="p27381323192311"></a><a name="p27381323192311"></a>end if</p>
<p id="p973862320236"><a name="p973862320236"></a><a name="p973862320236"></a><em id="i97381523112320"><a name="i97381523112320"></a><a name="i97381523112320"></a>result</em>[<em id="i87381923172312"><a name="i87381923172312"></a><a name="i87381923172312"></a>i</em>][7 : 0] = <em id="i5738162312233"><a name="i5738162312233"></a><a name="i5738162312233"></a>saturation</em>(<em id="i873832313234"><a name="i873832313234"></a><a name="i873832313234"></a>tmp</em>3[7 : 0]) according to control bit;</p>
</td>
</tr>
</tbody>
</table>

**表 10**  half to hifloat8\_t类型转换规则

<a name="table163085984311"></a>
<table><thead align="left"><tr id="row231185918436"><th class="cellrowborder" valign="top" width="100%" id="mcps1.2.2.1.1"><p id="p19567167184418"><a name="p19567167184418"></a><a name="p19567167184418"></a>VCVTFF: F162HiF8</p>
</th>
</tr>
</thead>
<tbody><tr id="row6311559174319"><td class="cellrowborder" valign="top" width="100%" headers="mcps1.2.2.1.1 "><p id="p8994018134715"><a name="p8994018134715"></a><a name="p8994018134715"></a><em id="i4994191814712"><a name="i4994191814712"></a><a name="i4994191814712"></a>tmp</em>2[15 : 0] = <em id="i2994171834716"><a name="i2994171834716"></a><a name="i2994171834716"></a>f16_src_data</em>[<em id="i119941818154720"><a name="i119941818154720"></a><a name="i119941818154720"></a>i</em>][15 : 0];</p>
<p id="p1999451894719"><a name="p1999451894719"></a><a name="p1999451894719"></a><em id="i19994218114718"><a name="i19994218114718"></a><a name="i19994218114718"></a>E<sub id="sub19941418164711"><a name="sub19941418164711"></a><a name="sub19941418164711"></a>v</sub></em> = <em id="i4994161816479"><a name="i4994161816479"></a><a name="i4994161816479"></a>tmp</em>2[14 : 10] - 5'<em id="i19941118184718"><a name="i19941118184718"></a><a name="i19941118184718"></a>b</em>01111, <em id="i19941618194717"><a name="i19941618194717"></a><a name="i19941618194717"></a>thr</em> = {<em id="i17994181816471"><a name="i17994181816471"></a><a name="i17994181816471"></a>tmp</em>2[0], 1’b1};</p>
<p id="p799421894712"><a name="p799421894712"></a><a name="p799421894712"></a>if (<em id="i199416180474"><a name="i199416180474"></a><a name="i199416180474"></a>E<sub id="sub8994121817475"><a name="sub8994121817475"></a><a name="sub8994121817475"></a>v</sub></em> == 16 &amp;&amp; <em id="i99941189478"><a name="i99941189478"></a><a name="i99941189478"></a>tmp</em>2[9 : 0] != 10'<em id="i09941718164710"><a name="i09941718164710"></a><a name="i09941718164710"></a>b</em>0) then</p>
<p id="p3994131894710"><a name="p3994131894710"></a><a name="p3994131894710"></a><em id="i99948186477"><a name="i99948186477"></a><a name="i99948186477"></a>tmp</em>3[7 : 0] = <em id="i599461812479"><a name="i599461812479"></a><a name="i599461812479"></a>HiF8<sub id="sub13994141819472"><a name="sub13994141819472"></a><a name="sub13994141819472"></a>NAN</sub></em>(8'<em id="i6994151894715"><a name="i6994151894715"></a><a name="i6994151894715"></a>b</em>10000000);</p>
<p id="p29941018164720"><a name="p29941018164720"></a><a name="p29941018164720"></a>else if (<em id="i11994181884715"><a name="i11994181884715"></a><a name="i11994181884715"></a>E<sub id="sub59941188474"><a name="sub59941188474"></a><a name="sub59941188474"></a>v</sub></em> == 16 &amp;&amp; <em id="i1499481844713"><a name="i1499481844713"></a><a name="i1499481844713"></a>tmp</em>2[9:0] == 10'<em id="i4994151824717"><a name="i4994151824717"></a><a name="i4994151824717"></a>b</em>0) &amp;&amp; (<em id="i16994181874713"><a name="i16994181874713"></a><a name="i16994181874713"></a>E<sub id="sub79941218154710"><a name="sub79941218154710"></a><a name="sub79941218154710"></a>v</sub></em> &gt; 15) then</p>
<p id="p9994118104717"><a name="p9994118104717"></a><a name="p9994118104717"></a><em id="i1899471819477"><a name="i1899471819477"></a><a name="i1899471819477"></a>tmp</em>3[7 : 0] = (<em id="i1199411894715"><a name="i1199411894715"></a><a name="i1199411894715"></a>tmp</em>2[31] == 1'<em id="i3994201824715"><a name="i3994201824715"></a><a name="i3994201824715"></a>b</em>0) ? <em id="i49941818144720"><a name="i49941818144720"></a><a name="i49941818144720"></a>HiF8<sub id="sub20994818104712"><a name="sub20994818104712"></a><a name="sub20994818104712"></a>+INF</sub></em>(8'<em id="i2099451811474"><a name="i2099451811474"></a><a name="i2099451811474"></a>b</em>01101111) : <em id="i189941518174711"><a name="i189941518174711"></a><a name="i189941518174711"></a>HiF8<sub id="sub2994101884715"><a name="sub2994101884715"></a><a name="sub2994101884715"></a>-INF</sub></em>(8'<em id="i399441819472"><a name="i399441819472"></a><a name="i399441819472"></a>b</em>11101111);</p>
<p id="p19994111824719"><a name="p19994111824719"></a><a name="p19994111824719"></a>else if (<em id="i1499417184472"><a name="i1499417184472"></a><a name="i1499417184472"></a>E<sub id="sub1699471818473"><a name="sub1699471818473"></a><a name="sub1699471818473"></a>v</sub></em> &lt; -23) then</p>
<p id="p09940185476"><a name="p09940185476"></a><a name="p09940185476"></a><em id="i12994121814715"><a name="i12994121814715"></a><a name="i12994121814715"></a>tmp</em>3[7 : 0] = <em id="i1999419187475"><a name="i1999419187475"></a><a name="i1999419187475"></a>HiF8<sub id="sub1994218164714"><a name="sub1994218164714"></a><a name="sub1994218164714"></a>ZERO</sub></em>(8'<em id="i18994111812474"><a name="i18994111812474"></a><a name="i18994111812474"></a>b</em>00000000);</p>
<p id="p1799431812471"><a name="p1799431812471"></a><a name="p1799431812471"></a>else if (<em id="i109942018104711"><a name="i109942018104711"></a><a name="i109942018104711"></a>E<sub id="sub199415182476"><a name="sub199415182476"></a><a name="sub199415182476"></a>v</sub></em> == -23) then</p>
<p id="p199461819475"><a name="p199461819475"></a><a name="p199461819475"></a>if ((Half To Away Round) || (Hybrid Round &amp;&amp; {1’<em id="i2994201854710"><a name="i2994201854710"></a><a name="i2994201854710"></a>b</em>1, <em id="i699421819475"><a name="i699421819475"></a><a name="i699421819475"></a>tmp</em>2[9]} &gt;= <em id="i16994101816473"><a name="i16994101816473"></a><a name="i16994101816473"></a>thr</em>)) then</p>
<p id="p17994131874719"><a name="p17994131874719"></a><a name="p17994131874719"></a><em id="i18994181864716"><a name="i18994181864716"></a><a name="i18994181864716"></a>tmp</em>3[7 : 0] = (<em id="i799411884712"><a name="i799411884712"></a><a name="i799411884712"></a>tmp</em>2[31] == 1'<em id="i09942018114718"><a name="i09942018114718"></a><a name="i09942018114718"></a>b</em>0) ? 8'<em id="i39943185479"><a name="i39943185479"></a><a name="i39943185479"></a>b</em>00000001 : 8'<em id="i169944182471"><a name="i169944182471"></a><a name="i169944182471"></a>b</em>10000001; // min subnormal</p>
<p id="p19941318114715"><a name="p19941318114715"></a><a name="p19941318114715"></a>else</p>
<p id="p19941818114716"><a name="p19941818114716"></a><a name="p19941818114716"></a><em id="i89941018114717"><a name="i89941018114717"></a><a name="i89941018114717"></a>tmp</em>3[7 : 0] = <em id="i13994101816476"><a name="i13994101816476"></a><a name="i13994101816476"></a>HiF8<sub id="sub11995131854710"><a name="sub11995131854710"></a><a name="sub11995131854710"></a>ZERO</sub></em>(8'<em id="i15995918114710"><a name="i15995918114710"></a><a name="i15995918114710"></a>b</em>00000000);</p>
<p id="p189951918184712"><a name="p189951918184712"></a><a name="p189951918184712"></a>end if</p>
<p id="p12995418174714"><a name="p12995418174714"></a><a name="p12995418174714"></a>else</p>
<p id="p999511182478"><a name="p999511182478"></a><a name="p999511182478"></a>if (<em id="i59951618144713"><a name="i59951618144713"></a><a name="i59951618144713"></a>E<sub id="sub8995518174711"><a name="sub8995518174711"></a><a name="sub8995518174711"></a>v</sub></em> == 0) then</p>
<p id="p69951518174718"><a name="p69951518174718"></a><a name="p69951518174718"></a><em id="i999521894719"><a name="i999521894719"></a><a name="i999521894719"></a>M</em> = <em id="i89954182473"><a name="i89954182473"></a><a name="i89954182473"></a>tmp</em>2[9 : 7], <em id="i1499571834720"><a name="i1499571834720"></a><a name="i1499571834720"></a>T</em> <em id="i9995141810478"><a name="i9995141810478"></a><a name="i9995141810478"></a>A_bit</em> = <em id="i11995818184719"><a name="i11995818184719"></a><a name="i11995818184719"></a>tmp</em>2[6], <em id="i17995171844717"><a name="i17995171844717"></a><a name="i17995171844717"></a>frac</em> = <em id="i999511185479"><a name="i999511185479"></a><a name="i999511185479"></a>tmp</em>2[6 : 5];</p>
<p id="p3995718184716"><a name="p3995718184716"></a><a name="p3995718184716"></a>else if (<em id="i6995718114717"><a name="i6995718114717"></a><a name="i6995718114717"></a>E<sub id="sub4995151817472"><a name="sub4995151817472"></a><a name="sub4995151817472"></a>v</sub></em> == &plusmn;1) then</p>
<p id="p18995191884714"><a name="p18995191884714"></a><a name="p18995191884714"></a><em id="i999561810477"><a name="i999561810477"></a><a name="i999561810477"></a>M</em> = <em id="i699541884720"><a name="i699541884720"></a><a name="i699541884720"></a>tmp</em>2[9 : 7], <em id="i799551818472"><a name="i799551818472"></a><a name="i799551818472"></a>T</em> <em id="i13995171874718"><a name="i13995171874718"></a><a name="i13995171874718"></a>A_bit</em> = <em id="i1999551814472"><a name="i1999551814472"></a><a name="i1999551814472"></a>tmp</em>2[6], <em id="i1199551824712"><a name="i1199551824712"></a><a name="i1199551824712"></a>frac</em> = <em id="i89951618124716"><a name="i89951618124716"></a><a name="i89951618124716"></a>tmp</em>2[6 : 5];</p>
<p id="p1299519187473"><a name="p1299519187473"></a><a name="p1299519187473"></a>else if (<em id="i139951218164716"><a name="i139951218164716"></a><a name="i139951218164716"></a>E<sub id="sub18995151854714"><a name="sub18995151854714"></a><a name="sub18995151854714"></a>v</sub></em> == &plusmn;[2, 3]) then</p>
<p id="p179951718144710"><a name="p179951718144710"></a><a name="p179951718144710"></a><em id="i179955186479"><a name="i179955186479"></a><a name="i179955186479"></a>M</em> = <em id="i1999541894715"><a name="i1999541894715"></a><a name="i1999541894715"></a>tmp</em>2[9 : 7], <em id="i17995518104718"><a name="i17995518104718"></a><a name="i17995518104718"></a>T</em> <em id="i9995151884715"><a name="i9995151884715"></a><a name="i9995151884715"></a>A_bit</em> = <em id="i6995151854710"><a name="i6995151854710"></a><a name="i6995151854710"></a>tmp</em>2[6], <em id="i109956188472"><a name="i109956188472"></a><a name="i109956188472"></a>frac</em> = <em id="i1199521813477"><a name="i1199521813477"></a><a name="i1199521813477"></a>tmp</em>2[6 : 5];</p>
<p id="p149950187479"><a name="p149950187479"></a><a name="p149950187479"></a>else if (<em id="i1499521812472"><a name="i1499521812472"></a><a name="i1499521812472"></a>E<sub id="sub3995141812472"><a name="sub3995141812472"></a><a name="sub3995141812472"></a>v</sub></em> == &plusmn;[4, 7]) then</p>
<p id="p4995131815472"><a name="p4995131815472"></a><a name="p4995131815472"></a><em id="i1399516182474"><a name="i1399516182474"></a><a name="i1399516182474"></a>M</em> = <em id="i12995141813478"><a name="i12995141813478"></a><a name="i12995141813478"></a>tmp</em>2[9 : 8], <em id="i12995101874715"><a name="i12995101874715"></a><a name="i12995101874715"></a>T</em> <em id="i18995101894718"><a name="i18995101894718"></a><a name="i18995101894718"></a>A_bit</em> = <em id="i19951718174710"><a name="i19951718174710"></a><a name="i19951718174710"></a>tmp</em>2[7], <em id="i14995151814475"><a name="i14995151814475"></a><a name="i14995151814475"></a>frac</em> = <em id="i1995111810471"><a name="i1995111810471"></a><a name="i1995111810471"></a>tmp</em>2[7 : 6];</p>
<p id="p1699591824711"><a name="p1699591824711"></a><a name="p1699591824711"></a>else if (<em id="i899571816475"><a name="i899571816475"></a><a name="i899571816475"></a>E<sub id="sub17995111819474"><a name="sub17995111819474"></a><a name="sub17995111819474"></a>v</sub></em> == &plusmn;[8, 15]) then</p>
<p id="p13995318194720"><a name="p13995318194720"></a><a name="p13995318194720"></a><em id="i1299561874714"><a name="i1299561874714"></a><a name="i1299561874714"></a>M</em> = <em id="i13995161811471"><a name="i13995161811471"></a><a name="i13995161811471"></a>tmp</em>2[9], <em id="i14995201813475"><a name="i14995201813475"></a><a name="i14995201813475"></a>T</em> <em id="i17995191814716"><a name="i17995191814716"></a><a name="i17995191814716"></a>A_bit</em> = <em id="i39951118144710"><a name="i39951118144710"></a><a name="i39951118144710"></a>tmp</em>2[8], <em id="i199591812479"><a name="i199591812479"></a><a name="i199591812479"></a>frac</em> = <em id="i13995121814474"><a name="i13995121814474"></a><a name="i13995121814474"></a>tmp</em>2[8 : 7];</p>
<p id="p299551864716"><a name="p299551864716"></a><a name="p299551864716"></a>else if (<em id="i1699513187477"><a name="i1699513187477"></a><a name="i1699513187477"></a>E<sub id="sub799591812477"><a name="sub799591812477"></a><a name="sub799591812477"></a>v</sub></em> == [-16, -22]) then</p>
<p id="p6995171816475"><a name="p6995171816475"></a><a name="p6995171816475"></a><em id="i1299541811479"><a name="i1299541811479"></a><a name="i1299541811479"></a>M</em> = <em id="i199531817478"><a name="i199531817478"></a><a name="i199531817478"></a>E<sub id="sub1799517185472"><a name="sub1799517185472"></a><a name="sub1799517185472"></a>v</sub></em> + 23, <em id="i1399517181475"><a name="i1399517181475"></a><a name="i1399517181475"></a>T</em> <em id="i7995618194718"><a name="i7995618194718"></a><a name="i7995618194718"></a>A_bit</em> = <em id="i119951818194718"><a name="i119951818194718"></a><a name="i119951818194718"></a>tmp</em>2[9], <em id="i11995318144715"><a name="i11995318144715"></a><a name="i11995318144715"></a>frac</em> = <em id="i79951318114720"><a name="i79951318114720"></a><a name="i79951318114720"></a>tmp</em>2[9 : 8]; // subnormal</p>
<p id="p2995121812473"><a name="p2995121812473"></a><a name="p2995121812473"></a>end if</p>
<p id="p1899518182475"><a name="p1899518182475"></a><a name="p1899518182475"></a>if (<em id="i199591844713"><a name="i199591844713"></a><a name="i199591844713"></a>E<sub id="sub19995151818477"><a name="sub19995151818477"></a><a name="sub19995151818477"></a>v</sub></em> == &plusmn;[0, 3]) then</p>
<p id="p19995218104716"><a name="p19995218104716"></a><a name="p19995218104716"></a>if (T <em id="i17995131814715"><a name="i17995131814715"></a><a name="i17995131814715"></a>A_bit</em> == 1’<em id="i1399521854710"><a name="i1399521854710"></a><a name="i1399521854710"></a>b</em>1) then</p>
<p id="p09951018204720"><a name="p09951018204720"></a><a name="p09951018204720"></a><em id="i7995171824713"><a name="i7995171824713"></a><a name="i7995171824713"></a>M_tmp</em> = <em id="i13995818114713"><a name="i13995818114713"></a><a name="i13995818114713"></a>M</em> + 1, <em id="i1399551815471"><a name="i1399551815471"></a><a name="i1399551815471"></a>E<sub id="sub18995918104718"><a name="sub18995918104718"></a><a name="sub18995918104718"></a>v</sub></em> = <em id="i129956180477"><a name="i129956180477"></a><a name="i129956180477"></a>E<sub id="sub89954181471"><a name="sub89954181471"></a><a name="sub89954181471"></a>v</sub></em> + <em id="i9995131819473"><a name="i9995131819473"></a><a name="i9995131819473"></a>carry</em> <em id="i99951118144712"><a name="i99951118144712"></a><a name="i99951118144712"></a>of</em> <em id="i1499561894713"><a name="i1499561894713"></a><a name="i1499561894713"></a>M_tmp</em>, <em id="i19995181874716"><a name="i19995181874716"></a><a name="i19995181874716"></a>M</em> = (<em id="i79951118144715"><a name="i79951118144715"></a><a name="i79951118144715"></a>carry of M_tmp</em>) ? 0 : <em id="i2995121864717"><a name="i2995121864717"></a><a name="i2995121864717"></a>M_tmp</em>;</p>
<p id="p299531815471"><a name="p299531815471"></a><a name="p299531815471"></a>end if</p>
<p id="p699521815475"><a name="p699521815475"></a><a name="p699521815475"></a>else if (<em id="i10995101811478"><a name="i10995101811478"></a><a name="i10995101811478"></a>E<sub id="sub79951118184718"><a name="sub79951118184718"></a><a name="sub79951118184718"></a>v</sub></em> == &plusmn;[4, 15]) then</p>
<p id="p9995111818479"><a name="p9995111818479"></a><a name="p9995111818479"></a>if (HALF To Away Round &amp;&amp; <em id="i1499501810473"><a name="i1499501810473"></a><a name="i1499501810473"></a>T</em> <em id="i199541817478"><a name="i199541817478"></a><a name="i199541817478"></a>A_bit</em> == 1’<em id="i1899521884712"><a name="i1899521884712"></a><a name="i1899521884712"></a>b</em>1) || (Hybrid Round &amp;&amp; <em id="i599511814715"><a name="i599511814715"></a><a name="i599511814715"></a>frac</em> &gt;= <em id="i15995418114713"><a name="i15995418114713"></a><a name="i15995418114713"></a>thr</em>) then</p>
<p id="p1399516182479"><a name="p1399516182479"></a><a name="p1399516182479"></a><em id="i1995151864718"><a name="i1995151864718"></a><a name="i1995151864718"></a>M_tmp</em> = M + 1, <em id="i12995181884714"><a name="i12995181884714"></a><a name="i12995181884714"></a>E<sub id="sub499581884711"><a name="sub499581884711"></a><a name="sub499581884711"></a>v</sub></em> = <em id="i1599551824710"><a name="i1599551824710"></a><a name="i1599551824710"></a>E<sub id="sub9995111874717"><a name="sub9995111874717"></a><a name="sub9995111874717"></a>v</sub></em> + <em id="i169951189471"><a name="i169951189471"></a><a name="i169951189471"></a>carry</em> <em id="i8995101815479"><a name="i8995101815479"></a><a name="i8995101815479"></a>of</em> <em id="i7995818184719"><a name="i7995818184719"></a><a name="i7995818184719"></a>M_tmp</em>, <em id="i179951018134715"><a name="i179951018134715"></a><a name="i179951018134715"></a>M</em> = (<em id="i199521814710"><a name="i199521814710"></a><a name="i199521814710"></a>carry of M_tmp</em>) ? 0 : <em id="i1099591874714"><a name="i1099591874714"></a><a name="i1099591874714"></a>M_tmp</em>;</p>
<p id="p2099571814473"><a name="p2099571814473"></a><a name="p2099571814473"></a>end if</p>
<p id="p17995141817479"><a name="p17995141817479"></a><a name="p17995141817479"></a>else if (<em id="i13995141816472"><a name="i13995141816472"></a><a name="i13995141816472"></a>E<sub id="sub1799541813478"><a name="sub1799541813478"></a><a name="sub1799541813478"></a>v</sub></em> == [-16, -22]) then</p>
<p id="p15995141810475"><a name="p15995141810475"></a><a name="p15995141810475"></a>if ((Half To Away Round) &amp;&amp; <em id="i16995618114712"><a name="i16995618114712"></a><a name="i16995618114712"></a>T</em> <em id="i9995181824718"><a name="i9995181824718"></a><a name="i9995181824718"></a>A_bit</em> == 1'<em id="i699551813474"><a name="i699551813474"></a><a name="i699551813474"></a>b</em>1) || ((Hybrid Round &amp;&amp; <em id="i99957180474"><a name="i99957180474"></a><a name="i99957180474"></a>frac</em> &gt;= <em id="i99957183473"><a name="i99957183473"></a><a name="i99957183473"></a>thr</em>) then</p>
<p id="p3995718194716"><a name="p3995718194716"></a><a name="p3995718194716"></a><em id="i9995141804714"><a name="i9995141804714"></a><a name="i9995141804714"></a>M</em>_<em id="i109951718174711"><a name="i109951718174711"></a><a name="i109951718174711"></a>tmp</em> = <em id="i89951183470"><a name="i89951183470"></a><a name="i89951183470"></a>E<sub id="sub999517184472"><a name="sub999517184472"></a><a name="sub999517184472"></a>v</sub></em> + 23, <em id="i3995191810475"><a name="i3995191810475"></a><a name="i3995191810475"></a>E<sub id="sub1099541812470"><a name="sub1099541812470"></a><a name="sub1099541812470"></a>v</sub></em> = <em id="i399512184474"><a name="i399512184474"></a><a name="i399512184474"></a>E<sub id="sub99953189478"><a name="sub99953189478"></a><a name="sub99953189478"></a>v</sub></em> + 1, M = (<em id="i1699581819477"><a name="i1699581819477"></a><a name="i1699581819477"></a>E<sub id="sub17995718174714"><a name="sub17995718174714"></a><a name="sub17995718174714"></a>v</sub></em> == -15) ? 0 : <em id="i1995101814471"><a name="i1995101814471"></a><a name="i1995101814471"></a>M_tmp</em>;</p>
<p id="p499571816473"><a name="p499571816473"></a><a name="p499571816473"></a>end if</p>
<p id="p18995151810472"><a name="p18995151810472"></a><a name="p18995151810472"></a>end if</p>
<p id="p129951218114713"><a name="p129951218114713"></a><a name="p129951218114713"></a>encode {<em id="i149951118114710"><a name="i149951118114710"></a><a name="i149951118114710"></a>tmp3</em>[31], <em id="i799571894712"><a name="i799571894712"></a><a name="i799571894712"></a>E<sub id="sub2995131844710"><a name="sub2995131844710"></a><a name="sub2995131844710"></a>v</sub></em>, <em id="i2995718194720"><a name="i2995718194720"></a><a name="i2995718194720"></a>M</em>} to <em id="i159951518154712"><a name="i159951518154712"></a><a name="i159951518154712"></a>tmp</em>3[7 : 0] following HiF encoding rule;</p>
<p id="p149952018104715"><a name="p149952018104715"></a><a name="p149952018104715"></a>end if</p>
<p id="p12995101874720"><a name="p12995101874720"></a><a name="p12995101874720"></a><em id="i899571812479"><a name="i899571812479"></a><a name="i899571812479"></a>result</em>[<em id="i9995171844719"><a name="i9995171844719"></a><a name="i9995171844719"></a>i</em>][7 : 0] = <em id="i17995171804714"><a name="i17995171804714"></a><a name="i17995171804714"></a>saturation</em>(<em id="i99954187478"><a name="i99954187478"></a><a name="i99954187478"></a>tmp</em>3[7 : 0]) according to control bit;</p>
</td>
</tr>
</tbody>
</table>

## 返回值说明<a name="section640mcpsimp"></a>

无

## 约束说明<a name="section633mcpsimp"></a>

-   该指令需配合[SetCtrlSpr\(ISASI\)](SetCtrlSpr(ISASI).md)  指令使用，通过设置寄存器的值控制Cast的饱和和非饱和模式。
-   浮点数转整数：
    -   非饱和模式：输入数据超过输出类型最值时，结果被截断为目标格式的数据宽度，例如输入half值为257, 输出uint8\_t值为1，输入为+/-inf，则返回输出类型的对应最值，输入nan时，返回0。
    -   饱和模式：输入数据超过输出类型最值时，返回输出类型的对应最值，例如输入half值为257, 输出uint8值为255，输入half值为-inf，输出uint8\_t值为0，输入nan时，返回0。

-   浮点数转浮点数：
    -   当前浮点数转浮点数支持饱和模式和非饱和模式，非饱和模式下，输入数据为nan时，输出为nan，输入+/-inf时，输出为+/-inf；饱和模式下，输入为nan时，输出为0，输入数据超过输出类型最值时，返回输出类型的对应最值。
    -   当输出类型float32时，只支持不饱和模式。
    -   当输出类型为fp8\_e4m3fn\_t时，由于fp8\_e4m3fn\_t没有inf表示格式，所以输出为nan。
    -   当输出类型为fp8\_e5m2\_t/fp8\_e4m3fn\_t时，输入nan，默认输出为0。
    -   对于bfloat16 to float4类型，输入bfloat16 inf或超出fp4x2\_e2m1\_t/fp4x2\_e1m2\_t数据最值范围时，会返回对应符号的fp4x2\_e2m1\_t/fp4x2\_e1m2\_t最值；输入nan时，fp4x2\_e2m1\_t/fp4x2\_e1m2\_t输出0。
    -   对于fp4x2\_e2m1\_t/fp4x2\_e1m2\_t类型，cast读写会每2个元素为一对进行读写，mask b16有效位以偶数位为准，例如

        256bit  01 01 00 00 01 00 00 01 等效于 01 01 00 00 01 01 00 00

        <!-- img2text -->
```
┌────┬────┬────┬────┬────┬────┬────┬────┐
│ 0  │ 1  │ 2  │ 3  │ 4  │ 5  │ 6  │ 7  │
└────┴────┴────┴────┴────┴────┴────┴────┘

┌────┬────┬────┬────┬────┬────┬────┬────┐
│ 1  │ 0  │ 0  │ 0  │ 1  │ 1  │ 0  │ 1  │
└────┴────┴────┴────┴────┴────┴────┴────┘

┌────┬────┬────┬────┬────┬────┬────┬────┐
│ 16 │ 16 │ 16 │ 16 │ 16 │ 16 │ 16 │ 16 │
└────┴────┴────┴────┴────┴────┴────┴────┘
                     │
                     ▼
┌──┬──┬──┬──┬──┬──┬──┬──┬──┬──┬──┬──┬──┬──┬──┬──┬──┬──┬──┬──┬──┬──┬──┬──┬──┬──┬──┬──┬──┬──┬──┬──┐
│4 │4 │4 │4 │4 │4 │4 │4 │4 │4 │4 │4 │4 │4 │4 │4 │4 │4 │4 │4 │4 │4 │4 │4 │4 │4 │4 │4 │4 │4 │4 │4 │
└──┴──┴──┴──┴──┴──┴──┴──┴──┴──┴──┴──┴──┴──┴──┴──┴──┴──┴──┴──┴──┴──┴──┴──┴──┴──┴──┴──┴──┴──┴──┴──┘
  1  1                                            1  1

┌──┬──┬──┬──┬──┬──┬──┬──┬──┬──┬──┬──┬──┬──┬──┬──┬──┬──┬──┬──┬──┬──┬──┬──┬──┬──┬──┬──┬──┬──┬──┬──┐
│4 │4 │4 │4 │4 │4 │4 │4 │4 │4 │4 │4 │4 │4 │4 │4 │4 │4 │4 │4 │4 │4 │4 │4 │4 │4 │4 │4 │4 │4 │4 │4 │
└──┴──┴──┴──┴──┴──┴──┴──┴──┴──┴──┴──┴──┴──┴──┴──┴──┴──┴──┴──┴──┴──┴──┴──┴──┴──┴──┴──┴──┴──┴──┴──┘
        1  1                                      1  1

┌──┬──┬──┬──┬──┬──┬──┬──┬──┬──┬──┬──┬──┬──┬──┬──┬──┬──┬──┬──┬──┬──┬──┬──┬──┬──┬──┬──┬──┬──┬──┬──┐
│4 │4 │4 │4 │4 │4 │4 │4 │4 │4 │4 │4 │4 │4 │4 │4 │4 │4 │4 │4 │4 │4 │4 │4 │4 │4 │4 │4 │4 │4 │4 │4 │
└──┴──┴──┴──┴──┴──┴──┴──┴──┴──┴──┴──┴──┴──┴──┴──┴──┴──┴──┴──┴──┴──┴──┴──┴──┴──┴──┴──┴──┴──┴──┴──┘
              1  1                                1  1

┌──┬──┬──┬──┬──┬──┬──┬──┬──┬──┬──┬──┬──┬──┬──┬──┬──┬──┬──┬──┬──┬──┬──┬──┬──┬──┬──┬──┬──┬──┬──┬──┐
│4 │4 │4 │4 │4 │4 │4 │4 │4 │4 │4 │4 │4 │4 │4 │4 │4 │4 │4 │4 │4 │4 │4 │4 │4 │4 │4 │4 │4 │4 │4 │4 │
└──┴──┴──┴──┴──┴──┴──┴──┴──┴──┴──┴──┴──┴──┴──┴──┴──┴──┴──┴──┴──┴──┴──┴──┴──┴──┴──┴──┴──┴──┴──┴──┘
                    1  1                          1  1
```

说明:
- 顶部第1行是索引/位置标号：`0 1 2 3 4 5 6 7`
- 顶部第2行对应值：`1 0 0 0 1 1 0 1`
- 第3行每块均为 `16`
- 向下映射后，得到 4 行、每行 32 个单元、每个单元值为 `4` 的布局
- 图中高亮单元对应关系可读为：
  - 第1行：第1-2列、17-18列
  - 第2行：第3-4列、19-20列
  - 第3行：第5-6列、21-22列
  - 第4行：第7-8列、23-24列
- 原图通过颜色区分高亮位置；ASCII 中用每行下方的 `1  1` 标示对应高亮单元位置

    -   对于fp8\_e8m0\_t类型：

        输入bfloat16\_t +/-inf或绝对值超出fp8\_e8m0\_t类型最值，则返回fp8\_e8m0\_t最大值0b11111110；

        输入bfloat16\_t nan 输出fp8\_e8m0\_t nan = 0b11111111。

-   整数转整数

    不饱和模式：输入数据会截断为目标数据格式，例如，输入int32\_t值为256, 输出uint8\_t值为0

    饱和模式：输入数据超出目标数据范围，会饱和为目标数据最值

    对于窄数据类型例如int16\_t\(2Byte\)转宽数据类型uint32\_t\(4Byte\),只支持饱和模式，输入负数会被饱和成0

## 调用示例<a name="section642mcpsimp"></a>

```
__simd_vf__ inline void CastVF(__ubuf__ int16_t* dstAddr, __ubuf__ float* srcAddr, uint32_t count, uint32_t srcRepeatSize, uint32_t dstRepeatSize, uint16_t repeatTimes)
{
    // castTrait 类变量时需要加static
    static constexpr AscendC::MicroAPI::CastTrait castTrait = 
    {AscendC::MicroAPI::RegLayout::ZERO, AscendC::MicroAPI::SatMode::NO_SAT,AscendC::MicroAPI::MaskMergeMode::ZEROING,AscendC::RoundMode::CAST_RINT};
   
   
    
    AscendC::MicroAPI::RegTensor<float> srcReg;
    AscendC::MicroAPI::RegTensor<int16_t> dstReg;
    AscendC::MicroAPI::MaskReg mask;
    for (uint16_t i = 0; i < repeatTimes; i++) {
        AscendC::MicroAPI::LoadAlign(srcReg, srcAddr + i * srcRepeatSize);
        mask = AscendC::MicroAPI::UpdateMask<float>(count);
        AscendC::MicroAPI::Cast<int16_t, float, castTrait>(dstReg, srcReg, mask);
        AscendC::MicroAPI::StoreAlign(dstAddr + i * dstRepeatSize, dstReg, mask);
    }
}
```

