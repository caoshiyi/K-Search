# GetMMConfig<a name="ZH-CN_TOPIC_0000002554423657"></a>

## 产品支持情况<a name="section1550532418810"></a>

<a name="table1334714391211"></a>
<table><thead align="left"><tr id="row1334743121213"><th class="cellrowborder" valign="top" width="57.99999999999999%" id="mcps1.1.3.1.1"><p id="p834713321216"><a name="p834713321216"></a><a name="p834713321216"></a><span id="ph834783101215"><a name="ph834783101215"></a><a name="ph834783101215"></a>产品</span></p>
</th>
<th class="cellrowborder" align="center" valign="top" width="42%" id="mcps1.1.3.1.2"><p id="p2347234127"><a name="p2347234127"></a><a name="p2347234127"></a>是否支持</p>
</th>
</tr>
</thead>
<tbody><tr id="row113472312122"><td class="cellrowborder" valign="top" width="57.99999999999999%" headers="mcps1.1.3.1.1 "><p id="p234710320128"><a name="p234710320128"></a><a name="p234710320128"></a><span id="ph103471336127"><a name="ph103471336127"></a><a name="ph103471336127"></a>Ascend 950PR/Ascend 950DT</span></p>
</td>
<td class="cellrowborder" align="center" valign="top" width="42%" headers="mcps1.1.3.1.2 "><p id="p4751940181211"><a name="p4751940181211"></a><a name="p4751940181211"></a>√</p>
</td>
</tr>
</tbody>
</table>

## 功能说明<a name="section618mcpsimp"></a>

灵活的自定义Matmul模板参数配置。通过设置[MatmulConfigMode](#table17837129144319)、[MatmulShapeParams](#table16317184295116)、[MatmulQuantParams](#table8313111211573)、[MatmulBatchParams](#table15129204644)、[MatmulFuncParams](#table66217141862)，获取自定义的[MatmulConfig](MatmulConfig.md#table1761013213153)。

MatmulConfigMode指定了获取并要修改的MatmulConfig模板，各模板介绍请参考[模板特性](MatmulConfig.md#table6981133810309)；用户根据使用需求通过设置可变参数，即一个或多个任意顺序的MatmulShapeParams、MatmulQuantParams、MatmulBatchParams、MatmulFuncParams，修改该MatmulConfig模板的相应参数配置。相比[GetNormalConfig](GetNormalConfig.md)、[GetMDLConfig](GetMDLConfig.md)等获取模板的接口，该接口提供了更灵活的自定义Matmul模板参数的配置方式。

## 函数原型<a name="section620mcpsimp"></a>

```
template <MatmulConfigMode configMode, typename... ArgTypes>
__aicore__ inline constexpr MatmulConfig GetMMConfig(ArgTypes&&... args)
```

## 参数说明<a name="section622mcpsimp"></a>

**表 1**  模板参数说明

<a name="table4835205712588"></a>
<table><thead align="left"><tr id="zh-cn_topic_0000001429830437_row118356578583"><th class="cellrowborder" valign="top" width="50%" id="mcps1.2.3.1.1"><p id="zh-cn_topic_0000001429830437_p48354572582"><a name="zh-cn_topic_0000001429830437_p48354572582"></a><a name="zh-cn_topic_0000001429830437_p48354572582"></a>参数名</p>
</th>
<th class="cellrowborder" valign="top" width="50%" id="mcps1.2.3.1.2"><p id="zh-cn_topic_0000001429830437_p583535795817"><a name="zh-cn_topic_0000001429830437_p583535795817"></a><a name="zh-cn_topic_0000001429830437_p583535795817"></a>描述</p>
</th>
</tr>
</thead>
<tbody><tr id="zh-cn_topic_0000001429830437_row1835857145817"><td class="cellrowborder" valign="top" width="50%" headers="mcps1.2.3.1.1 "><p id="zh-cn_topic_0000001429830437_p5835457165816"><a name="zh-cn_topic_0000001429830437_p5835457165816"></a><a name="zh-cn_topic_0000001429830437_p5835457165816"></a>configMode</p>
</td>
<td class="cellrowborder" valign="top" width="50%" headers="mcps1.2.3.1.2 "><p id="zh-cn_topic_0000001429830437_p168351657155818"><a name="zh-cn_topic_0000001429830437_p168351657155818"></a><a name="zh-cn_topic_0000001429830437_p168351657155818"></a>获取的MatmulConfig模板。</p>
</td>
</tr>
<tr id="row1648615377"><td class="cellrowborder" valign="top" width="50%" headers="mcps1.2.3.1.1 "><p id="p1212015191874"><a name="p1212015191874"></a><a name="p1212015191874"></a>ArgTypes</p>
</td>
<td class="cellrowborder" valign="top" width="50%" headers="mcps1.2.3.1.2 "><p id="p1912061914715"><a name="p1912061914715"></a><a name="p1912061914715"></a>可变模板参数。</p>
</td>
</tr>
</tbody>
</table>

**表 2**  参数说明

<a name="table9646134355611"></a>
<table><thead align="left"><tr id="row964714433565"><th class="cellrowborder" valign="top" width="14.99%" id="mcps1.2.4.1.1"><p id="zh-cn_topic_0235751031_p20917673"><a name="zh-cn_topic_0235751031_p20917673"></a><a name="zh-cn_topic_0235751031_p20917673"></a>参数名</p>
</th>
<th class="cellrowborder" valign="top" width="12.02%" id="mcps1.2.4.1.2"><p id="zh-cn_topic_0235751031_p16609919"><a name="zh-cn_topic_0235751031_p16609919"></a><a name="zh-cn_topic_0235751031_p16609919"></a>输入/输出</p>
</th>
<th class="cellrowborder" valign="top" width="72.99%" id="mcps1.2.4.1.3"><p id="zh-cn_topic_0235751031_p59995477"><a name="zh-cn_topic_0235751031_p59995477"></a><a name="zh-cn_topic_0235751031_p59995477"></a>描述</p>
</th>
</tr>
</thead>
<tbody><tr id="row16491543185617"><td class="cellrowborder" valign="top" width="14.99%" headers="mcps1.2.4.1.1 "><p id="p191281537718"><a name="p191281537718"></a><a name="p191281537718"></a>args</p>
</td>
<td class="cellrowborder" valign="top" width="12.02%" headers="mcps1.2.4.1.2 "><p id="p9128195310710"><a name="p9128195310710"></a><a name="p9128195310710"></a>输入</p>
</td>
<td class="cellrowborder" valign="top" width="72.99%" headers="mcps1.2.4.1.3 "><p id="p1128553474"><a name="p1128553474"></a><a name="p1128553474"></a>可变参数，任意顺序传入需要设置的MatmulShapeParams、MatmulQuantParams、MatmulBatchParams、MatmulFuncParams中的一个或多个。</p>
</td>
</tr>
</tbody>
</table>

**表 3**  MatmulConfigMode参数说明

<a name="table17837129144319"></a>
<table><thead align="left"><tr id="row1783811917437"><th class="cellrowborder" valign="top" width="35.85%" id="mcps1.2.3.1.1"><p id="p083889154318"><a name="p083889154318"></a><a name="p083889154318"></a>参数</p>
</th>
<th class="cellrowborder" valign="top" width="64.14999999999999%" id="mcps1.2.3.1.2"><p id="p1183812934313"><a name="p1183812934313"></a><a name="p1183812934313"></a>说明</p>
</th>
</tr>
</thead>
<tbody><tr id="row98385918438"><td class="cellrowborder" valign="top" width="35.85%" headers="mcps1.2.3.1.1 "><p id="p1383879184319"><a name="p1383879184319"></a><a name="p1383879184319"></a>CONFIG_NORM</p>
</td>
<td class="cellrowborder" valign="top" width="64.14999999999999%" headers="mcps1.2.3.1.2 "><p id="p4263424164812"><a name="p4263424164812"></a><a name="p4263424164812"></a>表示设置MatmulConfig默认值为Norm模板</p>
</td>
</tr>
<tr id="row118381498431"><td class="cellrowborder" valign="top" width="35.85%" headers="mcps1.2.3.1.1 "><p id="p138383917430"><a name="p138383917430"></a><a name="p138383917430"></a>CONFIG_MDL</p>
</td>
<td class="cellrowborder" valign="top" width="64.14999999999999%" headers="mcps1.2.3.1.2 "><p id="p5838291437"><a name="p5838291437"></a><a name="p5838291437"></a>表示设置MatmulConfig默认值为MDL模板</p>
</td>
</tr>
<tr id="row12838189124313"><td class="cellrowborder" valign="top" width="35.85%" headers="mcps1.2.3.1.1 "><p id="p38389974312"><a name="p38389974312"></a><a name="p38389974312"></a>CONFIG_SPECIALMDL</p>
</td>
<td class="cellrowborder" valign="top" width="64.14999999999999%" headers="mcps1.2.3.1.2 "><p id="p138385934314"><a name="p138385934314"></a><a name="p138385934314"></a>表示设置MatmulConfig默认值为SpecialMDL模板</p>
</td>
</tr>
<tr id="row5690459484"><td class="cellrowborder" valign="top" width="35.85%" headers="mcps1.2.3.1.1 "><p id="p9690655488"><a name="p9690655488"></a><a name="p9690655488"></a>CONFIG_IBSHARE</p>
</td>
<td class="cellrowborder" valign="top" width="64.14999999999999%" headers="mcps1.2.3.1.2 "><p id="p1569010534815"><a name="p1569010534815"></a><a name="p1569010534815"></a>表示设置MatmulConfig默认值为IBShare模板</p>
</td>
</tr>
</tbody>
</table>

**表 4**  MatmulShapeParams参数说明

<a name="table16317184295116"></a>
<table><thead align="left"><tr id="row23187424510"><th class="cellrowborder" valign="top" width="22.38%" id="mcps1.2.4.1.1"><p id="p831814424516"><a name="p831814424516"></a><a name="p831814424516"></a>参数</p>
</th>
<th class="cellrowborder" valign="top" width="18.29%" id="mcps1.2.4.1.2"><p id="p209933017531"><a name="p209933017531"></a><a name="p209933017531"></a>数据类型</p>
</th>
<th class="cellrowborder" valign="top" width="59.330000000000005%" id="mcps1.2.4.1.3"><p id="p231819424514"><a name="p231819424514"></a><a name="p231819424514"></a>说明</p>
</th>
</tr>
</thead>
<tbody><tr id="row13318144275116"><td class="cellrowborder" valign="top" width="22.38%" headers="mcps1.2.4.1.1 "><p id="p4318114214516"><a name="p4318114214516"></a><a name="p4318114214516"></a>singleCoreM</p>
</td>
<td class="cellrowborder" valign="top" width="18.29%" headers="mcps1.2.4.1.2 "><p id="p499313011537"><a name="p499313011537"></a><a name="p499313011537"></a>uint32_t</p>
</td>
<td class="cellrowborder" valign="top" width="59.330000000000005%" headers="mcps1.2.4.1.3 "><p id="p16121732161510"><a name="p16121732161510"></a><a name="p16121732161510"></a><span id="ph142788325310"><a name="ph142788325310"></a><a name="ph142788325310"></a>单核内M轴shape大小，以元素为单位。</span></p>
</td>
</tr>
<tr id="row3318194295115"><td class="cellrowborder" valign="top" width="22.38%" headers="mcps1.2.4.1.1 "><p id="p731814225110"><a name="p731814225110"></a><a name="p731814225110"></a>singleCoreN</p>
</td>
<td class="cellrowborder" valign="top" width="18.29%" headers="mcps1.2.4.1.2 "><p id="p120710179534"><a name="p120710179534"></a><a name="p120710179534"></a>uint32_t</p>
</td>
<td class="cellrowborder" valign="top" width="59.330000000000005%" headers="mcps1.2.4.1.3 "><p id="p86122032181510"><a name="p86122032181510"></a><a name="p86122032181510"></a><span id="ph1237918363215"><a name="ph1237918363215"></a><a name="ph1237918363215"></a>单核内N轴shape大小，以元素为单位。</span></p>
</td>
</tr>
<tr id="row17318144245111"><td class="cellrowborder" valign="top" width="22.38%" headers="mcps1.2.4.1.1 "><p id="p1531813424510"><a name="p1531813424510"></a><a name="p1531813424510"></a>singleCoreK</p>
</td>
<td class="cellrowborder" valign="top" width="18.29%" headers="mcps1.2.4.1.2 "><p id="p1182871765312"><a name="p1182871765312"></a><a name="p1182871765312"></a>uint32_t</p>
</td>
<td class="cellrowborder" valign="top" width="59.330000000000005%" headers="mcps1.2.4.1.3 "><p id="p961243217151"><a name="p961243217151"></a><a name="p961243217151"></a><span id="ph9173174013214"><a name="ph9173174013214"></a><a name="ph9173174013214"></a>单核内K轴shape大小，以元素为单位。</span></p>
</td>
</tr>
<tr id="row133181942185112"><td class="cellrowborder" valign="top" width="22.38%" headers="mcps1.2.4.1.1 "><p id="p3151255195211"><a name="p3151255195211"></a><a name="p3151255195211"></a>basicM</p>
</td>
<td class="cellrowborder" valign="top" width="18.29%" headers="mcps1.2.4.1.2 "><p id="p1378118135318"><a name="p1378118135318"></a><a name="p1378118135318"></a>uint32_t</p>
</td>
<td class="cellrowborder" valign="top" width="59.330000000000005%" headers="mcps1.2.4.1.3 "><p id="p1565814012286"><a name="p1565814012286"></a><a name="p1565814012286"></a><span id="ph108911580253"><a name="ph108911580253"></a><a name="ph108911580253"></a>与<a href="TCubeTiling结构体.md">TCubeTiling结构体</a>中的baseM参数含义相同，Matmul计算时base块M轴长度，以元素为单位。</span></p>
</td>
</tr>
<tr id="row1094117205532"><td class="cellrowborder" valign="top" width="22.38%" headers="mcps1.2.4.1.1 "><p id="p9941220205312"><a name="p9941220205312"></a><a name="p9941220205312"></a>basicN</p>
</td>
<td class="cellrowborder" valign="top" width="18.29%" headers="mcps1.2.4.1.2 "><p id="p1941102095310"><a name="p1941102095310"></a><a name="p1941102095310"></a>uint32_t</p>
</td>
<td class="cellrowborder" valign="top" width="59.330000000000005%" headers="mcps1.2.4.1.3 "><p id="p46111132121513"><a name="p46111132121513"></a><a name="p46111132121513"></a><span id="ph781319299267"><a name="ph781319299267"></a><a name="ph781319299267"></a>与<a href="TCubeTiling结构体.md">TCubeTiling结构体</a>中的baseN参数含义相同，Matmul计算时base块N轴长度，以元素为单位。</span></p>
</td>
</tr>
<tr id="row17466162419535"><td class="cellrowborder" valign="top" width="22.38%" headers="mcps1.2.4.1.1 "><p id="p5466102410531"><a name="p5466102410531"></a><a name="p5466102410531"></a>basicK</p>
</td>
<td class="cellrowborder" valign="top" width="18.29%" headers="mcps1.2.4.1.2 "><p id="p546632425314"><a name="p546632425314"></a><a name="p546632425314"></a>uint32_t</p>
</td>
<td class="cellrowborder" valign="top" width="59.330000000000005%" headers="mcps1.2.4.1.3 "><p id="p11611932111512"><a name="p11611932111512"></a><a name="p11611932111512"></a><span id="ph186378112276"><a name="ph186378112276"></a><a name="ph186378112276"></a>与<a href="TCubeTiling结构体.md">TCubeTiling结构体</a>中的baseK参数含义相同，Matmul计算时base块K轴长度，以元素为单位。</span></p>
</td>
</tr>
</tbody>
</table>

**表 5**  MatmulQuantParams参数说明

<a name="table8313111211573"></a>
<table><thead align="left"><tr id="row9313151219572"><th class="cellrowborder" valign="top" width="22.38%" id="mcps1.2.4.1.1"><p id="p173131812125712"><a name="p173131812125712"></a><a name="p173131812125712"></a>参数</p>
</th>
<th class="cellrowborder" valign="top" width="18.29%" id="mcps1.2.4.1.2"><p id="p163131812125718"><a name="p163131812125718"></a><a name="p163131812125718"></a>数据类型</p>
</th>
<th class="cellrowborder" valign="top" width="59.330000000000005%" id="mcps1.2.4.1.3"><p id="p83131212135713"><a name="p83131212135713"></a><a name="p83131212135713"></a>说明</p>
</th>
</tr>
</thead>
<tbody><tr id="row12313512205712"><td class="cellrowborder" valign="top" width="22.38%" headers="mcps1.2.4.1.1 "><p id="p1731301220577"><a name="p1731301220577"></a><a name="p1731301220577"></a>isPerTensor</p>
</td>
<td class="cellrowborder" valign="top" width="18.29%" headers="mcps1.2.4.1.2 "><p id="p1231361205716"><a name="p1231361205716"></a><a name="p1231361205716"></a>bool</p>
</td>
<td class="cellrowborder" valign="top" width="59.330000000000005%" headers="mcps1.2.4.1.3 "><p id="p22601175383"><a name="p22601175383"></a><a name="p22601175383"></a><span id="ph346512199361"><a name="ph346512199361"></a><a name="ph346512199361"></a>A矩阵half类型输入且B矩阵int8_t类型输入场景，使能B矩阵量化时是否为per tensor。</span></p>
<a name="ul19611220113813"></a><a name="ul19611220113813"></a><ul id="ul19611220113813"><li><span id="ph022382416360"><a name="ph022382416360"></a><a name="ph022382416360"></a>true：per tensor量化。</span></li><li><span id="ph174121537163315"><a name="ph174121537163315"></a><a name="ph174121537163315"></a>false：per channel量化。</span></li></ul>
<p id="p193136169340"><a name="p193136169340"></a><a name="p193136169340"></a><span id="ph24131137133319"><a name="ph24131137133319"></a><a name="ph24131137133319"></a>注意：MxMatmul场景仅支持false。</span></p>
</td>
</tr>
<tr id="row17313151285719"><td class="cellrowborder" valign="top" width="22.38%" headers="mcps1.2.4.1.1 "><p id="p331451216575"><a name="p331451216575"></a><a name="p331451216575"></a>hasAntiQuantOffset</p>
</td>
<td class="cellrowborder" valign="top" width="18.29%" headers="mcps1.2.4.1.2 "><p id="p10314312165719"><a name="p10314312165719"></a><a name="p10314312165719"></a>bool</p>
</td>
<td class="cellrowborder" valign="top" width="59.330000000000005%" headers="mcps1.2.4.1.3 "><p id="p116141032121517"><a name="p116141032121517"></a><a name="p116141032121517"></a><span id="ph166961850950"><a name="ph166961850950"></a><a name="ph166961850950"></a>A矩阵half类型输入且B矩阵int8_t类型输入场景，使能B矩阵量化时是否使用offset系数。</span></p>
<p id="p142014566541"><a name="p142014566541"></a><a name="p142014566541"></a><span id="ph1697950658"><a name="ph1697950658"></a><a name="ph1697950658"></a>注意：MxMatmul场景仅支持false。</span></p>
</td>
</tr>
</tbody>
</table>

**表 6**  MatmulBatchParams参数说明

<a name="table15129204644"></a>
<table><thead align="left"><tr id="row712911420420"><th class="cellrowborder" valign="top" width="22.38%" id="mcps1.2.4.1.1"><p id="p0129841145"><a name="p0129841145"></a><a name="p0129841145"></a>参数</p>
</th>
<th class="cellrowborder" valign="top" width="18.29%" id="mcps1.2.4.1.2"><p id="p11291541147"><a name="p11291541147"></a><a name="p11291541147"></a>数据类型</p>
</th>
<th class="cellrowborder" valign="top" width="59.330000000000005%" id="mcps1.2.4.1.3"><p id="p16129445415"><a name="p16129445415"></a><a name="p16129445415"></a>说明</p>
</th>
</tr>
</thead>
<tbody><tr id="row1112954642"><td class="cellrowborder" valign="top" width="22.38%" headers="mcps1.2.4.1.1 "><p id="p7129941543"><a name="p7129941543"></a><a name="p7129941543"></a>isNBatch</p>
</td>
<td class="cellrowborder" valign="top" width="18.29%" headers="mcps1.2.4.1.2 "><p id="p31299417414"><a name="p31299417414"></a><a name="p31299417414"></a>bool</p>
</td>
<td class="cellrowborder" valign="top" width="59.330000000000005%" headers="mcps1.2.4.1.3 "><p id="p176131132121514"><a name="p176131132121514"></a><a name="p176131132121514"></a><span id="ph1391761111166"><a name="ph1391761111166"></a><a name="ph1391761111166"></a>是否多Batch输入多Batch输出。仅对BatchMatmul有效，使能该参数后，仅支持Norm模板，且需调用<a href="IterateNBatch.md">IterateNBatch</a>实现多Batch输入多Batch输出。参数取值如下：</span></p>
<a name="ul261310324151"></a><a name="ul261310324151"></a><ul id="ul261310324151"><li><span id="ph7918911141611"><a name="ph7918911141611"></a><a name="ph7918911141611"></a>false：不使能多Batch（默认值）。</span></li><li><span id="ph3919141191616"><a name="ph3919141191616"></a><a name="ph3919141191616"></a>true：使能多Batch。</span></li></ul>
</td>
</tr>
<tr id="row111291541748"><td class="cellrowborder" valign="top" width="22.38%" headers="mcps1.2.4.1.1 "><p id="p4130641046"><a name="p4130641046"></a><a name="p4130641046"></a>batchMode</p>
</td>
<td class="cellrowborder" valign="top" width="18.29%" headers="mcps1.2.4.1.2 "><p id="p11301646418"><a name="p11301646418"></a><a name="p11301646418"></a>BatchMode</p>
</td>
<td class="cellrowborder" valign="top" width="59.330000000000005%" headers="mcps1.2.4.1.3 "><p id="p059314332299"><a name="p059314332299"></a><a name="p059314332299"></a><span id="ph115931633172912"><a name="ph115931633172912"></a><a name="ph115931633172912"></a>BatchMatmul场景中Layout类型为NORMAL时，设置BatchMatmul输入A/B矩阵的多batch数据总和与<span id="zh-cn_topic_0000002523303586_ph6593193372910"><a name="zh-cn_topic_0000002523303586_ph6593193372910"></a><a name="zh-cn_topic_0000002523303586_ph6593193372910"></a>L1 Buffer</span>的大小关系。参数取值如下：</span></p>
<a name="ul10593183316296"></a><a name="ul10593183316296"></a><ul id="ul10593183316296"><li><span id="ph5593733142910"><a name="ph5593733142910"></a><a name="ph5593733142910"></a>BatchMode::BATCH_LESS_THAN_L1：多batch数据总和&lt;<span id="zh-cn_topic_0000002523303586_ph145931033102914"><a name="zh-cn_topic_0000002523303586_ph145931033102914"></a><a name="zh-cn_topic_0000002523303586_ph145931033102914"></a>L1 Buffer</span> Size；</span></li><li><span id="ph25931334295"><a name="ph25931334295"></a><a name="ph25931334295"></a>BatchMode::BATCH_LARGE_THAN_L1：多batch数据总和&gt;<span id="zh-cn_topic_0000002523303586_ph10593333142912"><a name="zh-cn_topic_0000002523303586_ph10593333142912"></a><a name="zh-cn_topic_0000002523303586_ph10593333142912"></a>L1 Buffer</span> Size；</span></li><li><span id="ph9593203310298"><a name="ph9593203310298"></a><a name="ph9593203310298"></a>BatchMode::SINGLE_LARGE_THAN_L1：单batch数据总和&gt;<span id="zh-cn_topic_0000002523303586_ph259363362918"><a name="zh-cn_topic_0000002523303586_ph259363362918"></a><a name="zh-cn_topic_0000002523303586_ph259363362918"></a>L1 Buffer</span> Size。</span></li></ul>
</td>
</tr>
<tr id="row113017353319"><td class="cellrowborder" valign="top" width="22.38%" headers="mcps1.2.4.1.1 "><p id="p17304311330"><a name="p17304311330"></a><a name="p17304311330"></a>isBiasBatch</p>
</td>
<td class="cellrowborder" valign="top" width="18.29%" headers="mcps1.2.4.1.2 "><p id="p82991367333"><a name="p82991367333"></a><a name="p82991367333"></a>bool</p>
</td>
<td class="cellrowborder" valign="top" width="59.330000000000005%" headers="mcps1.2.4.1.3 "><p id="p833813297293"><a name="p833813297293"></a><a name="p833813297293"></a><span id="ph1589155111424"><a name="ph1589155111424"></a><a name="ph1589155111424"></a>批量多Batch的Matmul场景，即BatchMatmul场景，Bias的大小是否带有Batch轴。参数取值如下：</span></p>
<a name="ul333892932912"></a><a name="ul333892932912"></a><ul id="ul333892932912"><li><span id="ph95911051184215"><a name="ph95911051184215"></a><a name="ph95911051184215"></a>true：Bias带有Batch轴，Bias大小为Batch * N（默认值）。</span></li></ul>
<a name="ul5384955133317"></a><a name="ul5384955133317"></a><ul id="ul5384955133317"><li><span id="ph459335174210"><a name="ph459335174210"></a><a name="ph459335174210"></a>false：Bias不带Batch轴，Bias大小为N，多Batch计算Matmul时，会复用Bias。</span><p id="p855417902915"><a name="p855417902915"></a><a name="p855417902915"></a><span id="ph1059585184216"><a name="ph1059585184216"></a><a name="ph1059585184216"></a>注意：BatchMode::SINGLE_LARGE_THAN_L1场景仅支持设置为true。</span></p>
<p id="p12516195543410"><a name="p12516195543410"></a><a name="p12516195543410"></a><span id="ph960534612265"><a name="ph960534612265"></a><a name="ph960534612265"></a><span id="zh-cn_topic_0000002523303586_ph160544610261"><a name="zh-cn_topic_0000002523303586_ph160544610261"></a><a name="zh-cn_topic_0000002523303586_ph160544610261"></a>Ascend 950PR/Ascend 950DT</span>支持该参数。</span></p>
<p id="p14848175223415"><a name="p14848175223415"></a><a name="p14848175223415"></a><span id="ph284875253411"><a name="ph284875253411"></a><a name="ph284875253411"></a></span></p>
<p id="p684825214345"><a name="p684825214345"></a><a name="p684825214345"></a><span id="ph484818522347"><a name="ph484818522347"></a><a name="ph484818522347"></a></span></p>
</li></ul>
</td>
</tr>
<tr id="row2564134173119"><td class="cellrowborder" valign="top" width="22.38%" headers="mcps1.2.4.1.1 "><p id="p145648412315"><a name="p145648412315"></a><a name="p145648412315"></a>bmmOutMode</p>
</td>
<td class="cellrowborder" valign="top" width="18.29%" headers="mcps1.2.4.1.2 "><p id="p2564941314"><a name="p2564941314"></a><a name="p2564941314"></a>BatchOutMode</p>
</td>
<td class="cellrowborder" valign="top" width="59.330000000000005%" headers="mcps1.2.4.1.3 "><p id="p25644414318"><a name="p25644414318"></a><a name="p25644414318"></a>预留参数。</p>
</td>
</tr>
</tbody>
</table>

**表 7**  MatmulFuncParams参数说明

<a name="table66217141862"></a>
<table><thead align="left"><tr id="row06210141562"><th class="cellrowborder" valign="top" width="22.38%" id="mcps1.2.4.1.1"><p id="p3625142619"><a name="p3625142619"></a><a name="p3625142619"></a>参数</p>
</th>
<th class="cellrowborder" valign="top" width="18.29%" id="mcps1.2.4.1.2"><p id="p2624145615"><a name="p2624145615"></a><a name="p2624145615"></a>数据类型</p>
</th>
<th class="cellrowborder" valign="top" width="59.330000000000005%" id="mcps1.2.4.1.3"><p id="p146215141362"><a name="p146215141362"></a><a name="p146215141362"></a>说明</p>
</th>
</tr>
</thead>
<tbody><tr id="row16621014664"><td class="cellrowborder" valign="top" width="22.38%" headers="mcps1.2.4.1.1 "><p id="p862101416616"><a name="p862101416616"></a><a name="p862101416616"></a>intrinsicsLimit</p>
</td>
<td class="cellrowborder" valign="top" width="18.29%" headers="mcps1.2.4.1.2 "><p id="p3629141264"><a name="p3629141264"></a><a name="p3629141264"></a>bool</p>
</td>
<td class="cellrowborder" valign="top" width="59.330000000000005%" headers="mcps1.2.4.1.3 "><p id="p19428173891615"><a name="p19428173891615"></a><a name="p19428173891615"></a><span id="ph31274661320"><a name="ph31274661320"></a><a name="ph31274661320"></a>当左矩阵或右矩阵在单核上内轴（即尾轴）大于等于65535（元素个数）时，是否使能循环执行数据从<span id="zh-cn_topic_0000002523303586_ph610031519596"><a name="zh-cn_topic_0000002523303586_ph610031519596"></a><a name="zh-cn_topic_0000002523303586_ph610031519596"></a>Global Memory</span>到<span id="zh-cn_topic_0000002523303586_ph6551115913423"><a name="zh-cn_topic_0000002523303586_ph6551115913423"></a><a name="zh-cn_topic_0000002523303586_ph6551115913423"></a><span id="zh-cn_topic_0000002523303586_ph455120597421"><a name="zh-cn_topic_0000002523303586_ph455120597421"></a><a name="zh-cn_topic_0000002523303586_ph455120597421"></a>L1 Buffer</span></span>的搬入。例如，左矩阵A[M, K]，单核上的内轴数据singleCoreK大于65535，配置该参数为true后，API内部通过循环执行数据的搬入。参数取值如下：</span></p>
<a name="ul143331631192217"></a><a name="ul143331631192217"></a><ul id="ul143331631192217"><li><span id="ph19135469139"><a name="ph19135469139"></a><a name="ph19135469139"></a>false：当左矩阵或右矩阵在单核上内轴大于等于65535时，不使能循环执行数据的搬入（默认值）。</span></li><li><span id="ph514346131313"><a name="ph514346131313"></a><a name="ph514346131313"></a>true：当左矩阵或右矩阵在单核上内轴大于等于65535时，使能循环执行数据的搬入。</span></li></ul>
<p id="p131018512498"><a name="p131018512498"></a><a name="p131018512498"></a><span id="ph4151946101317"><a name="ph4151946101317"></a><a name="ph4151946101317"></a>注意：MxMatmul场景仅支持false。</span></p>
</td>
</tr>
<tr id="row116319141960"><td class="cellrowborder" valign="top" width="22.38%" headers="mcps1.2.4.1.1 "><p id="p86312141363"><a name="p86312141363"></a><a name="p86312141363"></a>enVecND2NZ</p>
</td>
<td class="cellrowborder" valign="top" width="18.29%" headers="mcps1.2.4.1.2 "><p id="p1963101413615"><a name="p1963101413615"></a><a name="p1963101413615"></a>bool</p>
</td>
<td class="cellrowborder" valign="top" width="59.330000000000005%" headers="mcps1.2.4.1.3 "><p id="p8151145151414"><a name="p8151145151414"></a><a name="p8151145151414"></a><span id="ph443123711176"><a name="ph443123711176"></a><a name="ph443123711176"></a>使能通过vector指令进行ND2NZ。使能时需要设置<a href="SetLocalWorkspace.md">SetLocalWorkspace</a>。参数取值如下：</span></p>
<a name="ul196141232181511"></a><a name="ul196141232181511"></a><ul id="ul196141232181511"><li><span id="ph11451937131714"><a name="ph11451937131714"></a><a name="ph11451937131714"></a>false：不使能通过vector指令进行ND2NZ（默认值）。</span></li><li><span id="ph646193741711"><a name="ph646193741711"></a><a name="ph646193741711"></a>true：使能通过vector指令进行ND2NZ。</span></li></ul>
<p id="p78127493504"><a name="p78127493504"></a><a name="p78127493504"></a><span id="ph348193731711"><a name="ph348193731711"></a><a name="ph348193731711"></a>注意：MxMatmul场景仅支持false。</span></p>
</td>
</tr>
<tr id="row1210515166457"><td class="cellrowborder" valign="top" width="22.38%" headers="mcps1.2.4.1.1 "><p id="p931081724513"><a name="p931081724513"></a><a name="p931081724513"></a>enableDoubleCache</p>
</td>
<td class="cellrowborder" valign="top" width="18.29%" headers="mcps1.2.4.1.2 "><p id="p1731015172455"><a name="p1731015172455"></a><a name="p1731015172455"></a>bool</p>
</td>
<td class="cellrowborder" valign="top" width="59.330000000000005%" headers="mcps1.2.4.1.3 "><p id="p156677274213"><a name="p156677274213"></a><a name="p156677274213"></a><span id="ph137426373192"><a name="ph137426373192"></a><a name="ph137426373192"></a>开启IBShare模板后，在<span id="zh-cn_topic_0000002523303586_ph1899163892912"><a name="zh-cn_topic_0000002523303586_ph1899163892912"></a><a name="zh-cn_topic_0000002523303586_ph1899163892912"></a>L1 Buffer</span>上是否同时缓存两块数据。参数取值如下：</span></p>
<a name="ul991213147241"></a><a name="ul991213147241"></a><ul id="ul991213147241"><li><span id="ph10757237101911"><a name="ph10757237101911"></a><a name="ph10757237101911"></a>false：<span id="zh-cn_topic_0000002523303586_ph14992203818292"><a name="zh-cn_topic_0000002523303586_ph14992203818292"></a><a name="zh-cn_topic_0000002523303586_ph14992203818292"></a>L1 Buffer</span>上同时缓存一块数据（默认值）。</span></li><li><span id="ph6771113713194"><a name="ph6771113713194"></a><a name="ph6771113713194"></a>true：使能<span id="zh-cn_topic_0000002523303586_ph159923388291"><a name="zh-cn_topic_0000002523303586_ph159923388291"></a><a name="zh-cn_topic_0000002523303586_ph159923388291"></a>L1 Buffer</span>上同时缓存两块数据。</span></li></ul>
<p id="p5312142173716"><a name="p5312142173716"></a><a name="p5312142173716"></a><span id="ph2064785683619"><a name="ph2064785683619"></a><a name="ph2064785683619"></a>注意：该参数取值为true时，需要控制基本块大小，防止两块数据的缓存超过<span id="zh-cn_topic_0000002523303586_ph5992173842912"><a name="zh-cn_topic_0000002523303586_ph5992173842912"></a><a name="zh-cn_topic_0000002523303586_ph5992173842912"></a>L1 Buffer</span>大小限制。</span></p>
<p id="p45216710517"><a name="p45216710517"></a><a name="p45216710517"></a><span id="ph14521170515"><a name="ph14521170515"></a><a name="ph14521170515"></a><span id="zh-cn_topic_0000002523303586_ph999243810293"><a name="zh-cn_topic_0000002523303586_ph999243810293"></a><a name="zh-cn_topic_0000002523303586_ph999243810293"></a>Ascend 950PR/Ascend 950DT</span>仅支持取值为false。</span></p>
</td>
</tr>
<tr id="row1122673824514"><td class="cellrowborder" valign="top" width="22.38%" headers="mcps1.2.4.1.1 "><p id="p43711039144512"><a name="p43711039144512"></a><a name="p43711039144512"></a>enableL1CacheUB</p>
</td>
<td class="cellrowborder" valign="top" width="18.29%" headers="mcps1.2.4.1.2 "><p id="p937163915455"><a name="p937163915455"></a><a name="p937163915455"></a>bool</p>
</td>
<td class="cellrowborder" valign="top" width="59.330000000000005%" headers="mcps1.2.4.1.3 "><p id="p113897573386"><a name="p113897573386"></a><a name="p113897573386"></a><span id="ph13491048293"><a name="ph13491048293"></a><a name="ph13491048293"></a>是否使能<span id="zh-cn_topic_0000002523303586_ph134251126162713"><a name="zh-cn_topic_0000002523303586_ph134251126162713"></a><a name="zh-cn_topic_0000002523303586_ph134251126162713"></a>L1 Buffer</span>缓存<span id="zh-cn_topic_0000002523303586_ph1942512616274"><a name="zh-cn_topic_0000002523303586_ph1942512616274"></a><a name="zh-cn_topic_0000002523303586_ph1942512616274"></a>Unified Buffer</span>计算块。建议在MTE3和MTE2流水串行较多的场景使用。参数取值如下：</span></p>
<a name="ul93891572381"></a><a name="ul93891572381"></a><ul id="ul93891572381"><li><span id="ph20492124815913"><a name="ph20492124815913"></a><a name="ph20492124815913"></a>true：使能<span id="zh-cn_topic_0000002523303586_ph17425626102719"><a name="zh-cn_topic_0000002523303586_ph17425626102719"></a><a name="zh-cn_topic_0000002523303586_ph17425626102719"></a>L1 Buffer</span>缓存<span id="zh-cn_topic_0000002523303586_ph8425122642711"><a name="zh-cn_topic_0000002523303586_ph8425122642711"></a><a name="zh-cn_topic_0000002523303586_ph8425122642711"></a>Unified Buffer</span>计算块。</span></li><li><span id="ph19493154816920"><a name="ph19493154816920"></a><a name="ph19493154816920"></a>false：不使能<span id="zh-cn_topic_0000002523303586_ph8425122662719"><a name="zh-cn_topic_0000002523303586_ph8425122662719"></a><a name="zh-cn_topic_0000002523303586_ph8425122662719"></a>L1 Buffer</span>缓存<span id="zh-cn_topic_0000002523303586_ph7425826112719"><a name="zh-cn_topic_0000002523303586_ph7425826112719"></a><a name="zh-cn_topic_0000002523303586_ph7425826112719"></a>Unified Buffer</span>计算块。</span></li></ul>
<p id="p18609101105210"><a name="p18609101105210"></a><a name="p18609101105210"></a><span id="ph194931481792"><a name="ph194931481792"></a><a name="ph194931481792"></a>若要使能<span id="zh-cn_topic_0000002523303586_ph74261826152710"><a name="zh-cn_topic_0000002523303586_ph74261826152710"></a><a name="zh-cn_topic_0000002523303586_ph74261826152710"></a>L1 Buffer</span>缓存<span id="zh-cn_topic_0000002523303586_ph20426192632719"><a name="zh-cn_topic_0000002523303586_ph20426192632719"></a><a name="zh-cn_topic_0000002523303586_ph20426192632719"></a>Unified Buffer</span>计算块，必须在Tiling实现中调用<a href="SetMatmulConfigParams.md">SetMatmulConfigParams</a>接口将参数enableL1CacheUBIn设置为true。</span></p>
<p id="p56625293014"><a name="p56625293014"></a><a name="p56625293014"></a><span id="ph19662529702"><a name="ph19662529702"></a><a name="ph19662529702"></a><span id="zh-cn_topic_0000002523303586_ph164261026142710"><a name="zh-cn_topic_0000002523303586_ph164261026142710"></a><a name="zh-cn_topic_0000002523303586_ph164261026142710"></a>Ascend 950PR/Ascend 950DT</span>不支持该参数。</span></p>
</td>
</tr>
<tr id="row1063514764"><td class="cellrowborder" valign="top" width="22.38%" headers="mcps1.2.4.1.1 "><p id="p11635141360"><a name="p11635141360"></a><a name="p11635141360"></a>doMTE2Preload</p>
</td>
<td class="cellrowborder" valign="top" width="18.29%" headers="mcps1.2.4.1.2 "><p id="p11631314865"><a name="p11631314865"></a><a name="p11631314865"></a>uint32_t</p>
</td>
<td class="cellrowborder" valign="top" width="59.330000000000005%" headers="mcps1.2.4.1.3 "><p id="p13613432131512"><a name="p13613432131512"></a><a name="p13613432131512"></a><span id="ph1798274510297"><a name="ph1798274510297"></a><a name="ph1798274510297"></a>在MTE2流水间隙较大，且M/N数值较大时可通过该参数开启对应M/N方向的预加载功能，开启后能减小MTE2间隙，提升性能。预加载功能仅在MDL模板有效（不支持SpecialMDL模板）。参数取值如下：</span></p>
<a name="ul20613193210152"></a><a name="ul20613193210152"></a><ul id="ul20613193210152"><li><span id="ph99831845122917"><a name="ph99831845122917"></a><a name="ph99831845122917"></a>0：不开启（默认值）。</span></li><li><span id="ph99851455294"><a name="ph99851455294"></a><a name="ph99851455294"></a>1：开启M方向preload。</span></li><li><span id="ph998511455299"><a name="ph998511455299"></a><a name="ph998511455299"></a>2：开启N方向preload。</span></li></ul>
<p id="p126131532141513"><a name="p126131532141513"></a><a name="p126131532141513"></a><span id="ph13986545162913"><a name="ph13986545162913"></a><a name="ph13986545162913"></a>注意：开启M/N方向的预加载功能时需保证K全载且M/N方向开启<a href="DoubleBuffer.md">DoubleBuffer</a>；其中，M方向的K全载条件为：singleCoreK/baseK &lt;= stepKa；N方向的K全载条件为：singleCoreK/baseK &lt;= stepKb。</span></p>
</td>
</tr>
<tr id="row1335742674613"><td class="cellrowborder" valign="top" width="22.38%" headers="mcps1.2.4.1.1 "><p id="p1376723111467"><a name="p1376723111467"></a><a name="p1376723111467"></a>iterateOrder</p>
</td>
<td class="cellrowborder" valign="top" width="18.29%" headers="mcps1.2.4.1.2 "><p id="p157674314460"><a name="p157674314460"></a><a name="p157674314460"></a>IterateOrder</p>
</td>
<td class="cellrowborder" valign="top" width="59.330000000000005%" headers="mcps1.2.4.1.3 "><p id="p13449151032917"><a name="p13449151032917"></a><a name="p13449151032917"></a><span id="ph2044991082912"><a name="ph2044991082912"></a><a name="ph2044991082912"></a>Matmul做矩阵运算的循环迭代顺序，与<a href="TCubeTiling结构体.md#table1563162142915">表1</a>中的iterateOrder参数含义相同。当ScheduleType参数取值为ScheduleType::OUTER_PRODUCT时，本参数生效。参数取值如下：</span></p>
<p id="p18241038144118"><a name="p18241038144118"></a><a name="p18241038144118"></a><span id="ph1939916182436"><a name="ph1939916182436"></a><a name="ph1939916182436"></a>ORDER_M：先往M轴方向偏移再往N轴方向偏移。</span></p>
<p id="p08015513429"><a name="p08015513429"></a><a name="p08015513429"></a><span id="ph16134922144318"><a name="ph16134922144318"></a><a name="ph16134922144318"></a>ORDER_N：先往N轴方向偏移再往M轴方向偏移。</span></p>
<p id="p1259735213413"><a name="p1259735213413"></a><a name="p1259735213413"></a><span id="ph315742574314"><a name="ph315742574314"></a><a name="ph315742574314"></a>UNDEF：当前无效。</span></p>
<p id="p4449161018293"><a name="p4449161018293"></a><a name="p4449161018293"></a><span id="ph18449101011294"><a name="ph18449101011294"></a><a name="ph18449101011294"></a>注：Norm模板的Matmul场景、MDL模板使用时，若IterateOrder取值ORDER_M，<a href="TCubeTiling结构体.md#table1563162142915">TCubeTiling结构</a>中的stepN需要大于1，IterateOrder取值ORDER_N时，TCubeTiling结构中的stepM需要大于1。MxMatmul仅支持MDL模板。</span></p>
</td>
</tr>
<tr id="row975082910462"><td class="cellrowborder" valign="top" width="22.38%" headers="mcps1.2.4.1.1 "><p id="p1976814313468"><a name="p1976814313468"></a><a name="p1976814313468"></a>scheduleType</p>
</td>
<td class="cellrowborder" valign="top" width="18.29%" headers="mcps1.2.4.1.2 "><p id="p77681631124617"><a name="p77681631124617"></a><a name="p77681631124617"></a>ScheduleType</p>
</td>
<td class="cellrowborder" valign="top" width="59.330000000000005%" headers="mcps1.2.4.1.3 "><p id="p032195210336"><a name="p032195210336"></a><a name="p032195210336"></a><span id="ph81739511194"><a name="ph81739511194"></a><a name="ph81739511194"></a>配置Matmul数据搬运模式。参数取值如下：</span></p>
<a name="ul12131154175915"></a><a name="ul12131154175915"></a><ul id="ul12131154175915"><li><span id="ph131749531915"><a name="ph131749531915"></a><a name="ph131749531915"></a>ScheduleType::INNER_PRODUCT：默认模式，在K方向上做MTE1的循环搬运；</span></li><li><span id="ph10151175444910"><a name="ph10151175444910"></a><a name="ph10151175444910"></a>ScheduleType::OUTER_PRODUCT：在M或N方向上做MTE1的循环搬运；使能后，需要与IterateOrder参数配合使用。</span><div class="p" id="p98221854103110"><a name="p98221854103110"></a><a name="p98221854103110"></a><span id="ph1317410519193"><a name="ph1317410519193"></a><a name="ph1317410519193"></a>该配置当前只在BatchMatmul场景（使能Norm模板）或 Matmul场景（使能MDL模板或Norm模板）生效。</span><a name="ul15421439154314"></a><a name="ul15421439154314"></a><ul id="ul15421439154314"><li><span id="ph141752511917"><a name="ph141752511917"></a><a name="ph141752511917"></a>若IterateOrder取值ORDER_M，则N方向循环搬运（在singleCoreN大于baseN场景可能有性能提升），即B矩阵的MTE1搬运并行；</span></li><li><span id="ph11176759196"><a name="ph11176759196"></a><a name="ph11176759196"></a>若IterateOrder取值ORDER_N，则M方向循环搬运（在singleCoreM大于baseM场景可能有性能提升），即A矩阵的MTE1搬运并行；</span></li><li><span id="ph917615151914"><a name="ph917615151914"></a><a name="ph917615151914"></a>不能同时使能M方向和N方向循环搬运；</span></li></ul>
</div>
</li></ul>
<p id="p18454103515913"><a name="p18454103515913"></a><a name="p18454103515913"></a><span id="ph317775141912"><a name="ph317775141912"></a><a name="ph317775141912"></a>注：</span></p>
<a name="ul1682313386598"></a><a name="ul1682313386598"></a><ul id="ul1682313386598"><li><span id="ph31782513195"><a name="ph31782513195"></a><a name="ph31782513195"></a>Norm模板的Batch Matmul场景或者MDL模板中，singleCoreK&gt;baseK时，不能使能ScheduleType::OUTER_PRODUCT取值，需使用默认模式。</span></li><li><span id="ph8741177274"><a name="ph8741177274"></a><a name="ph8741177274"></a>Norm模板或MDL模板的Matmul场景，仅支持在纯Cube模式（只有矩阵计算）下配置ScheduleType::OUTER_PRODUCT。</span></li><li><span id="ph517935141914"><a name="ph517935141914"></a><a name="ph517935141914"></a>MDL模板仅在调用<a href="IterateAll.md">IterateAll</a>计算的场景支持配置ScheduleType::OUTER_PRODUCT。</span></li><li><span id="ph1126191711212"><a name="ph1126191711212"></a><a name="ph1126191711212"></a>仅在C矩阵输出至GM时，支持配置ScheduleType::OUTER_PRODUCT。</span></li></ul>
<p id="p1745211010292"><a name="p1745211010292"></a><a name="p1745211010292"></a><span id="ph1445215105292"><a name="ph1445215105292"></a><a name="ph1445215105292"></a><span id="zh-cn_topic_0000002523303586_ph545215100296"><a name="zh-cn_topic_0000002523303586_ph545215100296"></a><a name="zh-cn_topic_0000002523303586_ph545215100296"></a>Ascend 950PR/Ascend 950DT</span>支持该参数。</span></p>
</td>
</tr>
<tr id="row550891031214"><td class="cellrowborder" valign="top" width="22.38%" headers="mcps1.2.4.1.1 "><p id="p2508201051216"><a name="p2508201051216"></a><a name="p2508201051216"></a>enableReuse</p>
</td>
<td class="cellrowborder" valign="top" width="18.29%" headers="mcps1.2.4.1.2 "><p id="p125081210171210"><a name="p125081210171210"></a><a name="p125081210171210"></a>bool</p>
</td>
<td class="cellrowborder" valign="top" width="59.330000000000005%" headers="mcps1.2.4.1.3 "><p id="p14238268272"><a name="p14238268272"></a><a name="p14238268272"></a><span id="ph14231826102716"><a name="ph14231826102716"></a><a name="ph14231826102716"></a><a href="SetSelfDefineData.md">SetSelfDefineData</a></span><span id="ph1642319261273"><a name="ph1642319261273"></a><a name="ph1642319261273"></a>函数设置的回调函数中的dataPtr是否直接传递计算数据。若未调用SetSelfDefineData设置dataPtr，该参数仅支持默认值true。参数取值如下：</span></p>
<a name="ul14423192611275"></a><a name="ul14423192611275"></a><ul id="ul14423192611275"><li><span id="ph642318266274"><a name="ph642318266274"></a><a name="ph642318266274"></a>true：直接传递计算数据，仅限单个值。</span></li><li><span id="ph2423152612720"><a name="ph2423152612720"></a><a name="ph2423152612720"></a>false：传递GM上存储的数据地址信息。</span></li></ul>
<p id="p12423726112713"><a name="p12423726112713"></a><a name="p12423726112713"></a><span id="ph104237262278"><a name="ph104237262278"></a><a name="ph104237262278"></a>注意：MxMatmul场景仅支持true。</span></p>
</td>
</tr>
<tr id="row484642013125"><td class="cellrowborder" valign="top" width="22.38%" headers="mcps1.2.4.1.1 "><p id="p484615209122"><a name="p484615209122"></a><a name="p484615209122"></a>enableUBReuse</p>
</td>
<td class="cellrowborder" valign="top" width="18.29%" headers="mcps1.2.4.1.2 "><p id="p184662016128"><a name="p184662016128"></a><a name="p184662016128"></a>bool</p>
</td>
<td class="cellrowborder" valign="top" width="59.330000000000005%" headers="mcps1.2.4.1.3 "><p id="p143891657133818"><a name="p143891657133818"></a><a name="p143891657133818"></a><span id="ph101741817077"><a name="ph101741817077"></a><a name="ph101741817077"></a>是否使能<span id="zh-cn_topic_0000002523303586_ph16424826132717"><a name="zh-cn_topic_0000002523303586_ph16424826132717"></a><a name="zh-cn_topic_0000002523303586_ph16424826132717"></a>Unified Buffer</span>复用。在<span id="zh-cn_topic_0000002523303586_ph1142432622711"><a name="zh-cn_topic_0000002523303586_ph1142432622711"></a><a name="zh-cn_topic_0000002523303586_ph1142432622711"></a><span id="zh-cn_topic_0000002523303586_ph042419261271"><a name="zh-cn_topic_0000002523303586_ph042419261271"></a><a name="zh-cn_topic_0000002523303586_ph042419261271"></a>Unified Buffer</span></span>空间足够的条件下（<span id="zh-cn_topic_0000002523303586_ph15424126162717"><a name="zh-cn_topic_0000002523303586_ph15424126162717"></a><a name="zh-cn_topic_0000002523303586_ph15424126162717"></a><span id="zh-cn_topic_0000002523303586_ph16424192614271"><a name="zh-cn_topic_0000002523303586_ph16424192614271"></a><a name="zh-cn_topic_0000002523303586_ph16424192614271"></a>Unified Buffer</span></span>空间大于4倍TCubeTiling的<a href="TCubeTiling结构体.md#p1620315053211">transLength</a>参数），使能该参数后，<span id="zh-cn_topic_0000002523303586_ph34246261279"><a name="zh-cn_topic_0000002523303586_ph34246261279"></a><a name="zh-cn_topic_0000002523303586_ph34246261279"></a><span id="zh-cn_topic_0000002523303586_ph154242264276"><a name="zh-cn_topic_0000002523303586_ph154242264276"></a><a name="zh-cn_topic_0000002523303586_ph154242264276"></a>Unified Buffer</span></span>空间分为互不重叠的两份，分别存储Matmul计算相邻前后两轮迭代的数据，后一轮迭代数据的搬入将不必等待前一轮迭代的<span id="zh-cn_topic_0000002523303586_ph542410265274"><a name="zh-cn_topic_0000002523303586_ph542410265274"></a><a name="zh-cn_topic_0000002523303586_ph542410265274"></a><span id="zh-cn_topic_0000002523303586_ph1424192616274"><a name="zh-cn_topic_0000002523303586_ph1424192616274"></a><a name="zh-cn_topic_0000002523303586_ph1424192616274"></a>Unified Buffer</span></span>空间释放，<span>从而优化流水</span>。参数取值如下：</span></p>
<a name="ul1638914579380"></a><a name="ul1638914579380"></a><ul id="ul1638914579380"><li><span id="ph10175417572"><a name="ph10175417572"></a><a name="ph10175417572"></a>true：使能<span id="zh-cn_topic_0000002523303586_ph11424182614272"><a name="zh-cn_topic_0000002523303586_ph11424182614272"></a><a name="zh-cn_topic_0000002523303586_ph11424182614272"></a>Unified Buffer</span>复用。</span></li><li><span id="ph121766179713"><a name="ph121766179713"></a><a name="ph121766179713"></a>false：不使能<span id="zh-cn_topic_0000002523303586_ph17424526112712"><a name="zh-cn_topic_0000002523303586_ph17424526112712"></a><a name="zh-cn_topic_0000002523303586_ph17424526112712"></a>Unified Buffer</span>复用。</span></li></ul>
<p id="p12425426102711"><a name="p12425426102711"></a><a name="p12425426102711"></a><span id="ph1742511260278"><a name="ph1742511260278"></a><a name="ph1742511260278"></a><span id="zh-cn_topic_0000002523303586_ph8425526132713"><a name="zh-cn_topic_0000002523303586_ph8425526132713"></a><a name="zh-cn_topic_0000002523303586_ph8425526132713"></a>Ascend 950PR/Ascend 950DT</span>不支持该参数。</span></p>
</td>
</tr>
<tr id="row41221548165515"><td class="cellrowborder" valign="top" width="22.38%" headers="mcps1.2.4.1.1 "><p id="p82771410155710"><a name="p82771410155710"></a><a name="p82771410155710"></a>isPartialOutput</p>
</td>
<td class="cellrowborder" valign="top" width="18.29%" headers="mcps1.2.4.1.2 "><p id="p2101154145513"><a name="p2101154145513"></a><a name="p2101154145513"></a>bool</p>
</td>
<td class="cellrowborder" valign="top" width="59.330000000000005%" headers="mcps1.2.4.1.3 "><p id="p6826115611715"><a name="p6826115611715"></a><a name="p6826115611715"></a><span id="ph1057174216711"><a name="ph1057174216711"></a><a name="ph1057174216711"></a>是否开启PartialOutput功能，即控制Matmul顺序输出K方向的基本块计算方式：Matmul一次Iterate计算的K轴是否进行累加计算。参数取值如下：</span></p>
<a name="ul550312572719"></a><a name="ul550312572719"></a><ul id="ul550312572719"><li><span id="ph969194219713"><a name="ph969194219713"></a><a name="ph969194219713"></a>true：开启PartialOutput功能，一次Iterate的K轴不进行累加计算，Matmul每次计算输出局部baseK的baseM * baseN大小的矩阵分片。</span></li><li><span id="ph13787421177"><a name="ph13787421177"></a><a name="ph13787421177"></a>false：不开启PartialOutput功能，一次Iterate的K轴进行累加计算，Matmul每次计算输出SingleCoreK长度的baseM * baseN大小的矩阵分片。</span></li></ul>
<p id="p65031357677"><a name="p65031357677"></a><a name="p65031357677"></a><span id="ph13793426710"><a name="ph13793426710"></a><a name="ph13793426710"></a><span id="zh-cn_topic_0000002523303586_ph203032039133413"><a name="zh-cn_topic_0000002523303586_ph203032039133413"></a><a name="zh-cn_topic_0000002523303586_ph203032039133413"></a>Ascend 950PR/Ascend 950DT</span>支持该参数。</span></p>
</td>
</tr>
<tr id="row1463317147201"><td class="cellrowborder" valign="top" width="22.38%" headers="mcps1.2.4.1.1 "><p id="p167221122161810"><a name="p167221122161810"></a><a name="p167221122161810"></a>isA2B2Shared</p>
</td>
<td class="cellrowborder" valign="top" width="18.29%" headers="mcps1.2.4.1.2 "><p id="p8746152711203"><a name="p8746152711203"></a><a name="p8746152711203"></a>bool</p>
</td>
<td class="cellrowborder" valign="top" width="59.330000000000005%" headers="mcps1.2.4.1.3 "><p id="p1474672732017"><a name="p1474672732017"></a><a name="p1474672732017"></a><span id="ph476825784113"><a name="ph476825784113"></a><a name="ph476825784113"></a>是否开启A2和B2的全局管理，即控制所有Matmul对象是否共用A2和B2的double buffer机制。该配置为全局配置，所有Matmul对象取值必须保持一致。注意，开启时，A矩阵、B矩阵的基本块大小均不能超过32KB。</span></p>
<p id="p10746172712015"><a name="p10746172712015"></a><a name="p10746172712015"></a><span id="ph1077195784120"><a name="ph1077195784120"></a><a name="ph1077195784120"></a>参数取值如下：</span></p>
<a name="ul17746122762016"></a><a name="ul17746122762016"></a><ul id="ul17746122762016"><li><span id="ph177721957104117"><a name="ph177721957104117"></a><a name="ph177721957104117"></a>true：开启。</span></li><li><span id="ph577365719416"><a name="ph577365719416"></a><a name="ph577365719416"></a>false：关闭（默认值）。</span></li></ul>
<p id="p1524135812332"><a name="p1524135812332"></a><a name="p1524135812332"></a><span id="ph1377910579419"><a name="ph1377910579419"></a><a name="ph1377910579419"></a><span id="zh-cn_topic_0000002523303586_ph1024175814333"><a name="zh-cn_topic_0000002523303586_ph1024175814333"></a><a name="zh-cn_topic_0000002523303586_ph1024175814333"></a>Ascend 950PR/Ascend 950DT</span>支持该参数。</span></p>
<p id="p274692719202"><a name="p274692719202"></a><a name="p274692719202"></a><span id="ph9780957144115"><a name="ph9780957144115"></a><a name="ph9780957144115"></a>注意：MxMatmul场景下该参数仅支持false。</span></p>
<p id="p814495612515"><a name="p814495612515"></a><a name="p814495612515"></a><span id="ph1278145713416"><a name="ph1278145713416"></a><a name="ph1278145713416"></a>该参数取值为true时，建议同时设置enUnitFlag参数为true，使搬运与计算流水并行，提高性能。</span></p>
</td>
</tr>
<tr id="row19873121015612"><td class="cellrowborder" valign="top" width="22.38%" headers="mcps1.2.4.1.1 "><p id="p124851975615"><a name="p124851975615"></a><a name="p124851975615"></a>isEnableChannelSplit</p>
</td>
<td class="cellrowborder" valign="top" width="18.29%" headers="mcps1.2.4.1.2 "><p id="p1981104414618"><a name="p1981104414618"></a><a name="p1981104414618"></a>bool</p>
</td>
<td class="cellrowborder" valign="top" width="59.330000000000005%" headers="mcps1.2.4.1.3 "><p id="p132577145518"><a name="p132577145518"></a><a name="p132577145518"></a><span id="ph1991202515416"><a name="ph1991202515416"></a><a name="ph1991202515416"></a>是否使能channel_split功能。正常情况下，Matmul计算出的CubeFormat::NZ格式的C矩阵分形为16*16，假设此时的分形个数为x，channel_split功能是使获得的C矩阵分形为16*8，同时分形个数变为2x。注意，当前仅在Matmul计算结果C矩阵的Format为CubeFormat::NZ，TYPE为float类型，矩阵乘结果CO1为float类型，输出到Global Memory的场景，支持使能该参数。参数取值如下：</span></p>
<a name="ul998819524413"></a><a name="ul998819524413"></a><ul id="ul998819524413"><li><span id="ph79931625175415"><a name="ph79931625175415"></a><a name="ph79931625175415"></a>false：默认值，不使能channel_split功能，输出的分形为16*16。</span></li><li><span id="ph499482519547"><a name="ph499482519547"></a><a name="ph499482519547"></a>true：使能channel_split功能，输出的分形为16*8。</span><p id="p7342144383715"><a name="p7342144383715"></a><a name="p7342144383715"></a><span id="ph43421643153712"><a name="ph43421643153712"></a><a name="ph43421643153712"></a><span id="zh-cn_topic_0000002523303586_ph43421143123716"><a name="zh-cn_topic_0000002523303586_ph43421143123716"></a><a name="zh-cn_topic_0000002523303586_ph43421143123716"></a>Ascend 950PR/Ascend 950DT</span>支持该参数。</span></p>
</li></ul>
</td>
</tr>
<tr id="row15210175918579"><td class="cellrowborder" valign="top" width="22.38%" headers="mcps1.2.4.1.1 "><p id="p521016593574"><a name="p521016593574"></a><a name="p521016593574"></a>enableKdimReorderLoad</p>
</td>
<td class="cellrowborder" valign="top" width="18.29%" headers="mcps1.2.4.1.2 "><p id="p165121018195816"><a name="p165121018195816"></a><a name="p165121018195816"></a>bool</p>
</td>
<td class="cellrowborder" valign="top" width="59.330000000000005%" headers="mcps1.2.4.1.3 "><p id="p95376407818"><a name="p95376407818"></a><a name="p95376407818"></a><span id="ph10909208814"><a name="ph10909208814"></a><a name="ph10909208814"></a>是否使能K轴错峰加载数据。基于相同Tiling参数，执行Matmul计算时，如果多核的左矩阵或者右矩阵相同，且存储于Global Memory，多个核一般会同时访问相同地址以加载矩阵数据，引发同地址访问冲突，影响性能。使能该参数后，多核执行Matmul时，将尽量在相同时间访问矩阵的不同Global Memory地址，减少地址访问冲突概率，提升性能。该参数功能只支持MDL模板，建议K轴较大且左矩阵和右矩阵均非全载场景使能参数。参数取值如下。</span></p>
<a name="ul6537104018815"></a><a name="ul6537104018815"></a><ul id="ul6537104018815"><li><span id="ph053717401382"><a name="ph053717401382"></a><a name="ph053717401382"></a>false：默认值，关闭K轴错峰加载数据的功能。</span></li><li><span id="ph9537040489"><a name="ph9537040489"></a><a name="ph9537040489"></a>true：开启K轴错峰加载数据的功能。</span></li></ul>
<p id="p03118101964"><a name="p03118101964"></a><a name="p03118101964"></a><span id="ph1231111109610"><a name="ph1231111109610"></a><a name="ph1231111109610"></a><span id="zh-cn_topic_0000002523303586_ph5343204318379"><a name="zh-cn_topic_0000002523303586_ph5343204318379"></a><a name="zh-cn_topic_0000002523303586_ph5343204318379"></a>Ascend 950PR/Ascend 950DT</span>支持该参数。</span></p>
</td>
</tr>
<tr id="row677716237224"><td class="cellrowborder" valign="top" width="22.38%" headers="mcps1.2.4.1.1 "><p id="p1977712238226"><a name="p1977712238226"></a><a name="p1977712238226"></a>enableL1BankConflictOptimise</p>
</td>
<td class="cellrowborder" valign="top" width="18.29%" headers="mcps1.2.4.1.2 "><p id="p137773238223"><a name="p137773238223"></a><a name="p137773238223"></a>bool</p>
</td>
<td class="cellrowborder" valign="top" width="59.330000000000005%" headers="mcps1.2.4.1.3 "><p id="p8458135281220"><a name="p8458135281220"></a><a name="p8458135281220"></a><span id="ph1245895281215"><a name="ph1245895281215"></a><a name="ph1245895281215"></a>是否使能L1上的Bank冲突优化。在Tiling侧调用<a href="EnableL1BankConflictOptimise.md">EnableL1BankConflictOptimise</a>接口获取能否使能该参数的结果，并与<a href="基本流程.md#li578045965">TilingKey</a>机制配合使用，在Kernel侧增加代码实现分支。若使能该参数，基于相同Tiling参数执行Matmul计算时，对A、B矩阵和MxMatmul场景的ScaleA、ScaleB矩阵不再连续分配L1 Buffer的空间，在DoubleBuffer场景下，并行计算的数据分别被分配在L1 Buffer的上半部空间和下半部空间，非DoubleBuf场景，数据被分配在L1 Buffer的上半部空间；另外，Bias被分配在L1 Buffer的上半部空间，向量的量化/反量化场景的量化系数被分配在L1 Buffer的下半部空间。参数取值如下。</span></p>
<a name="ul445935211218"></a><a name="ul445935211218"></a><ul id="ul445935211218"><li><span id="ph15311154662213"><a name="ph15311154662213"></a><a name="ph15311154662213"></a>false：默认值，关闭L1 Bank冲突优化。</span></li><li><span id="ph83122046102219"><a name="ph83122046102219"></a><a name="ph83122046102219"></a>true：开启L1 Bank冲突优化。</span></li></ul>
<p id="p145915217126"><a name="p145915217126"></a><a name="p145915217126"></a><span id="ph94591852121214"><a name="ph94591852121214"></a><a name="ph94591852121214"></a><span id="zh-cn_topic_0000002523303586_ph24590524125"><a name="zh-cn_topic_0000002523303586_ph24590524125"></a><a name="zh-cn_topic_0000002523303586_ph24590524125"></a>Ascend 950PR/Ascend 950DT</span>支持该参数。</span></p>
</td>
</tr>
</tbody>
</table>

## 返回值说明<a name="section640mcpsimp"></a>

[MatmulConfig结构体](MatmulConfig.md#table1761013213153)。

## 约束说明<a name="section633mcpsimp"></a>

无

## 调用示例<a name="section116441811350"></a>

```
// 获取MatmulConfig模板为Norm模板
constexpr static MatmulConfigMode configMode = MatmulConfigMode::CONFIG_NORM;
// singleCoreM、singleCoreN、singleCoreK、basicM、basicN、basicK
constexpr static MatmulShapeParams shapeParams = {128, 128, 128, 64, 64, 64};
// B矩阵量化时为per channel且不使用offset系数
constexpr static MatmulQuantParams quantParams = {false, false};
// 不使能多Batch
constexpr static MatmulBatchParams batchParams{false};
// 不进行芯片指令搬运地址偏移量校验，使能通过vector进行ND2NZ
constexpr static MatmulFuncParams funcParams{false, true};
constexpr static MatmulConfig mmConfig = GetMMConfig<configMode>(shapeParams, quantParams, batchParams, funcParams);
```

