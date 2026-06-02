# GetNormalConfig<a name="ZH-CN_TOPIC_0000002554424531"></a>

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

用于配置Norm模板的参数，获取自定义Norm模板。Norm模板的介绍请参考[表 模板特性](MatmulConfig.md#table6981133810309)。

## 函数原型<a name="section620mcpsimp"></a>

```
__aicore__ constexpr MatmulConfig GetNormalConfig(const bool intrinsicsLimit = false, const bool batchLoop = false, const bool isVecND2NZ = false, const BatchMode bmmMode = BatchMode::BATCH_LESS_THAN_L1, const bool isMsgReuse = true, const IterateOrder iterateOrder = IterateOrder::UNDEF, const ScheduleType scheduleType = ScheduleType::INNER_PRODUCT, const bool enUnitFlag = true, const bool enableMixDualMaster = false, const BatchOutMode bmmOutMode = BatchOutMode::SINGLE_BATCH)
```

## 参数说明<a name="section622mcpsimp"></a>

本接口的所有参数用于设置[MatmulConfig结构体](MatmulConfig.md#table1761013213153)中的参数，其中互相对应的参数的功能作用相同。

**表 1**  接口参数说明

<a name="zh-cn_topic_0235751031_table33761356"></a>
<table><thead align="left"><tr id="zh-cn_topic_0235751031_row27598891"><th class="cellrowborder" valign="top" width="17.77%" id="mcps1.2.4.1.1"><p id="zh-cn_topic_0235751031_p20917673"><a name="zh-cn_topic_0235751031_p20917673"></a><a name="zh-cn_topic_0235751031_p20917673"></a>参数名</p>
</th>
<th class="cellrowborder" valign="top" width="11.799999999999999%" id="mcps1.2.4.1.2"><p id="zh-cn_topic_0235751031_p16609919"><a name="zh-cn_topic_0235751031_p16609919"></a><a name="zh-cn_topic_0235751031_p16609919"></a>输入/输出</p>
</th>
<th class="cellrowborder" valign="top" width="70.43%" id="mcps1.2.4.1.3"><p id="zh-cn_topic_0235751031_p59995477"><a name="zh-cn_topic_0235751031_p59995477"></a><a name="zh-cn_topic_0235751031_p59995477"></a>描述</p>
</th>
</tr>
</thead>
<tbody><tr id="row42461942101815"><td class="cellrowborder" valign="top" width="17.77%" headers="mcps1.2.4.1.1 "><p id="p163481714145518"><a name="p163481714145518"></a><a name="p163481714145518"></a>intrinsicsLimit</p>
</td>
<td class="cellrowborder" valign="top" width="11.799999999999999%" headers="mcps1.2.4.1.2 "><p id="p33487148556"><a name="p33487148556"></a><a name="p33487148556"></a>输入</p>
</td>
<td class="cellrowborder" valign="top" width="70.43%" headers="mcps1.2.4.1.3 "><p id="p1147594373717"><a name="p1147594373717"></a><a name="p1147594373717"></a>用于设置参数intrinsicsCheck。</p>
<p id="p19428173891615"><a name="p19428173891615"></a><a name="p19428173891615"></a><span id="ph31274661320"><a name="ph31274661320"></a><a name="ph31274661320"></a>当左矩阵或右矩阵在单核上内轴（即尾轴）大于等于65535（元素个数）时，是否使能循环执行数据从<span id="zh-cn_topic_0000002523303586_ph610031519596"><a name="zh-cn_topic_0000002523303586_ph610031519596"></a><a name="zh-cn_topic_0000002523303586_ph610031519596"></a>Global Memory</span>到<span id="zh-cn_topic_0000002523303586_ph6551115913423"><a name="zh-cn_topic_0000002523303586_ph6551115913423"></a><a name="zh-cn_topic_0000002523303586_ph6551115913423"></a><span id="zh-cn_topic_0000002523303586_ph455120597421"><a name="zh-cn_topic_0000002523303586_ph455120597421"></a><a name="zh-cn_topic_0000002523303586_ph455120597421"></a>L1 Buffer</span></span>的搬入。例如，左矩阵A[M, K]，单核上的内轴数据singleCoreK大于65535，配置该参数为true后，API内部通过循环执行数据的搬入。参数取值如下：</span></p>
<a name="ul143331631192217"></a><a name="ul143331631192217"></a><ul id="ul143331631192217"><li><span id="ph19135469139"><a name="ph19135469139"></a><a name="ph19135469139"></a>false：当左矩阵或右矩阵在单核上内轴大于等于65535时，不使能循环执行数据的搬入（默认值）。</span></li><li><span id="ph514346131313"><a name="ph514346131313"></a><a name="ph514346131313"></a>true：当左矩阵或右矩阵在单核上内轴大于等于65535时，使能循环执行数据的搬入。</span></li></ul>
<p id="p131018512498"><a name="p131018512498"></a><a name="p131018512498"></a><span id="ph4151946101317"><a name="ph4151946101317"></a><a name="ph4151946101317"></a>注意：MxMatmul场景仅支持false。</span></p>
</td>
</tr>
<tr id="row1282014916166"><td class="cellrowborder" valign="top" width="17.77%" headers="mcps1.2.4.1.1 "><p id="p14807165981614"><a name="p14807165981614"></a><a name="p14807165981614"></a>batchLoop</p>
</td>
<td class="cellrowborder" valign="top" width="11.799999999999999%" headers="mcps1.2.4.1.2 "><p id="p0807115911613"><a name="p0807115911613"></a><a name="p0807115911613"></a>输入</p>
</td>
<td class="cellrowborder" valign="top" width="70.43%" headers="mcps1.2.4.1.3 "><p id="p154881241113711"><a name="p154881241113711"></a><a name="p154881241113711"></a>用于设置参数isNBatch。</p>
<p id="p176131132121514"><a name="p176131132121514"></a><a name="p176131132121514"></a><span id="ph1391761111166"><a name="ph1391761111166"></a><a name="ph1391761111166"></a>是否多Batch输入多Batch输出。仅对BatchMatmul有效，使能该参数后，仅支持Norm模板，且需调用<a href="IterateNBatch.md">IterateNBatch</a>实现多Batch输入多Batch输出。参数取值如下：</span></p>
<a name="ul261310324151"></a><a name="ul261310324151"></a><ul id="ul261310324151"><li><span id="ph7918911141611"><a name="ph7918911141611"></a><a name="ph7918911141611"></a>false：不使能多Batch（默认值）。</span></li><li><span id="ph3919141191616"><a name="ph3919141191616"></a><a name="ph3919141191616"></a>true：使能多Batch。</span></li></ul>
</td>
</tr>
<tr id="row4822173173811"><td class="cellrowborder" valign="top" width="17.77%" headers="mcps1.2.4.1.1 "><p id="p198234333811"><a name="p198234333811"></a><a name="p198234333811"></a>isVecND2NZ</p>
</td>
<td class="cellrowborder" valign="top" width="11.799999999999999%" headers="mcps1.2.4.1.2 "><p id="p1420734110385"><a name="p1420734110385"></a><a name="p1420734110385"></a>输入</p>
</td>
<td class="cellrowborder" valign="top" width="70.43%" headers="mcps1.2.4.1.3 "><p id="p98238315389"><a name="p98238315389"></a><a name="p98238315389"></a>用于设置参数enVecND2NZ。</p>
<p id="p11614143215153"><a name="p11614143215153"></a><a name="p11614143215153"></a><span id="ph443123711176"><a name="ph443123711176"></a><a name="ph443123711176"></a>使能通过vector指令进行ND2NZ。使能时需要设置<a href="SetLocalWorkspace.md">SetLocalWorkspace</a>。参数取值如下：</span></p>
<a name="ul196141232181511"></a><a name="ul196141232181511"></a><ul id="ul196141232181511"><li><span id="ph11451937131714"><a name="ph11451937131714"></a><a name="ph11451937131714"></a>false：不使能通过vector指令进行ND2NZ（默认值）。</span></li><li><span id="ph646193741711"><a name="ph646193741711"></a><a name="ph646193741711"></a>true：使能通过vector指令进行ND2NZ。</span></li></ul>
<p id="p78127493504"><a name="p78127493504"></a><a name="p78127493504"></a><span id="ph348193731711"><a name="ph348193731711"></a><a name="ph348193731711"></a>注意：MxMatmul场景仅支持false。</span></p>
</td>
</tr>
<tr id="row13809141083811"><td class="cellrowborder" valign="top" width="17.77%" headers="mcps1.2.4.1.1 "><p id="p7810910113819"><a name="p7810910113819"></a><a name="p7810910113819"></a>bmmMode</p>
</td>
<td class="cellrowborder" valign="top" width="11.799999999999999%" headers="mcps1.2.4.1.2 "><p id="p120714113815"><a name="p120714113815"></a><a name="p120714113815"></a>输入</p>
</td>
<td class="cellrowborder" valign="top" width="70.43%" headers="mcps1.2.4.1.3 "><p id="p481014105388"><a name="p481014105388"></a><a name="p481014105388"></a>用于设置参数batchMode。该参数用于BatchMatmul场景，关于BatchMatmul的介绍请参考<a href="Batch-Matmul基础功能.md">Batch Matmul基础功能</a>。</p>
<p id="p059314332299"><a name="p059314332299"></a><a name="p059314332299"></a><span id="ph115931633172912"><a name="ph115931633172912"></a><a name="ph115931633172912"></a>BatchMatmul场景中Layout类型为NORMAL时，设置BatchMatmul输入A/B矩阵的多batch数据总和与<span id="zh-cn_topic_0000002523303586_ph6593193372910"><a name="zh-cn_topic_0000002523303586_ph6593193372910"></a><a name="zh-cn_topic_0000002523303586_ph6593193372910"></a>L1 Buffer</span>的大小关系。参数取值如下：</span></p>
<a name="ul10593183316296"></a><a name="ul10593183316296"></a><ul id="ul10593183316296"><li><span id="ph5593733142910"><a name="ph5593733142910"></a><a name="ph5593733142910"></a>BatchMode::BATCH_LESS_THAN_L1：多batch数据总和&lt;<span id="zh-cn_topic_0000002523303586_ph145931033102914"><a name="zh-cn_topic_0000002523303586_ph145931033102914"></a><a name="zh-cn_topic_0000002523303586_ph145931033102914"></a>L1 Buffer</span> Size；</span></li><li><span id="ph25931334295"><a name="ph25931334295"></a><a name="ph25931334295"></a>BatchMode::BATCH_LARGE_THAN_L1：多batch数据总和&gt;<span id="zh-cn_topic_0000002523303586_ph10593333142912"><a name="zh-cn_topic_0000002523303586_ph10593333142912"></a><a name="zh-cn_topic_0000002523303586_ph10593333142912"></a>L1 Buffer</span> Size；</span></li><li><span id="ph9593203310298"><a name="ph9593203310298"></a><a name="ph9593203310298"></a>BatchMode::SINGLE_LARGE_THAN_L1：单batch数据总和&gt;<span id="zh-cn_topic_0000002523303586_ph259363362918"><a name="zh-cn_topic_0000002523303586_ph259363362918"></a><a name="zh-cn_topic_0000002523303586_ph259363362918"></a>L1 Buffer</span> Size。</span></li></ul>
</td>
</tr>
<tr id="row154269161386"><td class="cellrowborder" valign="top" width="17.77%" headers="mcps1.2.4.1.1 "><p id="p142615165384"><a name="p142615165384"></a><a name="p142615165384"></a>isMsgReuse</p>
</td>
<td class="cellrowborder" valign="top" width="11.799999999999999%" headers="mcps1.2.4.1.2 "><p id="p18111642173812"><a name="p18111642173812"></a><a name="p18111642173812"></a>输入</p>
</td>
<td class="cellrowborder" valign="top" width="70.43%" headers="mcps1.2.4.1.3 "><p id="p142601617380"><a name="p142601617380"></a><a name="p142601617380"></a>用于设置参数enableReuse。</p>
<p id="p14238268272"><a name="p14238268272"></a><a name="p14238268272"></a><span id="ph14231826102716"><a name="ph14231826102716"></a><a name="ph14231826102716"></a><a href="SetSelfDefineData.md">SetSelfDefineData</a></span><span id="ph1642319261273"><a name="ph1642319261273"></a><a name="ph1642319261273"></a>函数设置的回调函数中的dataPtr是否直接传递计算数据。若未调用SetSelfDefineData设置dataPtr，该参数仅支持默认值true。参数取值如下：</span></p>
<a name="ul14423192611275"></a><a name="ul14423192611275"></a><ul id="ul14423192611275"><li><span id="ph642318266274"><a name="ph642318266274"></a><a name="ph642318266274"></a>true：直接传递计算数据，仅限单个值。</span></li><li><span id="ph2423152612720"><a name="ph2423152612720"></a><a name="ph2423152612720"></a>false：传递GM上存储的数据地址信息。</span></li></ul>
<p id="p12423726112713"><a name="p12423726112713"></a><a name="p12423726112713"></a><span id="ph104237262278"><a name="ph104237262278"></a><a name="ph104237262278"></a>注意：MxMatmul场景仅支持true。</span></p>
</td>
</tr>
<tr id="row188645228388"><td class="cellrowborder" valign="top" width="17.77%" headers="mcps1.2.4.1.1 "><p id="p1586462283818"><a name="p1586462283818"></a><a name="p1586462283818"></a>iterateOrder</p>
</td>
<td class="cellrowborder" valign="top" width="11.799999999999999%" headers="mcps1.2.4.1.2 "><p id="p71112422383"><a name="p71112422383"></a><a name="p71112422383"></a>输入</p>
</td>
<td class="cellrowborder" valign="top" width="70.43%" headers="mcps1.2.4.1.3 "><p id="p108641722113819"><a name="p108641722113819"></a><a name="p108641722113819"></a>用于设置参数iterateOrder。</p>
<p id="p13449151032917"><a name="p13449151032917"></a><a name="p13449151032917"></a><span id="ph2044991082912"><a name="ph2044991082912"></a><a name="ph2044991082912"></a>Matmul做矩阵运算的循环迭代顺序，与<a href="TCubeTiling结构体.md#table1563162142915">表1</a>中的iterateOrder参数含义相同。当ScheduleType参数取值为ScheduleType::OUTER_PRODUCT时，本参数生效。参数取值如下：</span></p>
<p id="p18241038144118"><a name="p18241038144118"></a><a name="p18241038144118"></a><span id="ph1939916182436"><a name="ph1939916182436"></a><a name="ph1939916182436"></a>ORDER_M：先往M轴方向偏移再往N轴方向偏移。</span></p>
<p id="p08015513429"><a name="p08015513429"></a><a name="p08015513429"></a><span id="ph16134922144318"><a name="ph16134922144318"></a><a name="ph16134922144318"></a>ORDER_N：先往N轴方向偏移再往M轴方向偏移。</span></p>
<p id="p1259735213413"><a name="p1259735213413"></a><a name="p1259735213413"></a><span id="ph315742574314"><a name="ph315742574314"></a><a name="ph315742574314"></a>UNDEF：当前无效。</span></p>
<p id="p4449161018293"><a name="p4449161018293"></a><a name="p4449161018293"></a><span id="ph18449101011294"><a name="ph18449101011294"></a><a name="ph18449101011294"></a>注：Norm模板的Matmul场景、MDL模板使用时，若IterateOrder取值ORDER_M，<a href="TCubeTiling结构体.md#table1563162142915">TCubeTiling结构</a>中的stepN需要大于1，IterateOrder取值ORDER_N时，TCubeTiling结构中的stepM需要大于1。MxMatmul仅支持MDL模板。</span></p>
</td>
</tr>
<tr id="row15511132833811"><td class="cellrowborder" valign="top" width="17.77%" headers="mcps1.2.4.1.1 "><p id="p15511162818382"><a name="p15511162818382"></a><a name="p15511162818382"></a>scheduleType</p>
</td>
<td class="cellrowborder" valign="top" width="11.799999999999999%" headers="mcps1.2.4.1.2 "><p id="p1183894215382"><a name="p1183894215382"></a><a name="p1183894215382"></a>输入</p>
</td>
<td class="cellrowborder" valign="top" width="70.43%" headers="mcps1.2.4.1.3 "><p id="p165112284383"><a name="p165112284383"></a><a name="p165112284383"></a>用于设置参数scheduleType。</p>
<p id="p032195210336"><a name="p032195210336"></a><a name="p032195210336"></a><span id="ph81739511194"><a name="ph81739511194"></a><a name="ph81739511194"></a>配置Matmul数据搬运模式。参数取值如下：</span></p>
<a name="ul12131154175915"></a><a name="ul12131154175915"></a><ul id="ul12131154175915"><li><span id="ph131749531915"><a name="ph131749531915"></a><a name="ph131749531915"></a>ScheduleType::INNER_PRODUCT：默认模式，在K方向上做MTE1的循环搬运；</span></li><li><span id="ph10151175444910"><a name="ph10151175444910"></a><a name="ph10151175444910"></a>ScheduleType::OUTER_PRODUCT：在M或N方向上做MTE1的循环搬运；使能后，需要与IterateOrder参数配合使用。</span><div class="p" id="p98221854103110"><a name="p98221854103110"></a><a name="p98221854103110"></a><span id="ph1317410519193"><a name="ph1317410519193"></a><a name="ph1317410519193"></a>该配置当前只在BatchMatmul场景（使能Norm模板）或 Matmul场景（使能MDL模板或Norm模板）生效。</span><a name="ul15421439154314"></a><a name="ul15421439154314"></a><ul id="ul15421439154314"><li><span id="ph141752511917"><a name="ph141752511917"></a><a name="ph141752511917"></a>若IterateOrder取值ORDER_M，则N方向循环搬运（在singleCoreN大于baseN场景可能有性能提升），即B矩阵的MTE1搬运并行；</span></li><li><span id="ph11176759196"><a name="ph11176759196"></a><a name="ph11176759196"></a>若IterateOrder取值ORDER_N，则M方向循环搬运（在singleCoreM大于baseM场景可能有性能提升），即A矩阵的MTE1搬运并行；</span></li><li><span id="ph917615151914"><a name="ph917615151914"></a><a name="ph917615151914"></a>不能同时使能M方向和N方向循环搬运；</span></li></ul>
</div>
</li></ul>
<p id="p18454103515913"><a name="p18454103515913"></a><a name="p18454103515913"></a><span id="ph317775141912"><a name="ph317775141912"></a><a name="ph317775141912"></a>注：</span></p>
<a name="ul1682313386598"></a><a name="ul1682313386598"></a><ul id="ul1682313386598"><li><span id="ph31782513195"><a name="ph31782513195"></a><a name="ph31782513195"></a>Norm模板的Batch Matmul场景或者MDL模板中，singleCoreK&gt;baseK时，不能使能ScheduleType::OUTER_PRODUCT取值，需使用默认模式。</span></li><li><span id="ph8741177274"><a name="ph8741177274"></a><a name="ph8741177274"></a>Norm模板或MDL模板的Matmul场景，仅支持在纯Cube模式（只有矩阵计算）下配置ScheduleType::OUTER_PRODUCT。</span></li><li><span id="ph517935141914"><a name="ph517935141914"></a><a name="ph517935141914"></a>MDL模板仅在调用<a href="IterateAll.md">IterateAll</a>计算的场景支持配置ScheduleType::OUTER_PRODUCT。</span></li><li><span id="ph1126191711212"><a name="ph1126191711212"></a><a name="ph1126191711212"></a>仅在C矩阵输出至GM时，支持配置ScheduleType::OUTER_PRODUCT。</span><p id="p21872165212"><a name="p21872165212"></a><a name="p21872165212"></a><span id="ph418711161214"><a name="ph418711161214"></a><a name="ph418711161214"></a><span id="zh-cn_topic_0000002523303586_ph545215100296"><a name="zh-cn_topic_0000002523303586_ph545215100296"></a><a name="zh-cn_topic_0000002523303586_ph545215100296"></a>Ascend 950PR/Ascend 950DT</span>支持该参数。</span></p>
</li></ul>
</td>
</tr>
<tr id="row20814103513386"><td class="cellrowborder" valign="top" width="17.77%" headers="mcps1.2.4.1.1 "><p id="p4814183503815"><a name="p4814183503815"></a><a name="p4814183503815"></a>enUnitFlag</p>
</td>
<td class="cellrowborder" valign="top" width="11.799999999999999%" headers="mcps1.2.4.1.2 "><p id="p28390429388"><a name="p28390429388"></a><a name="p28390429388"></a>输入</p>
</td>
<td class="cellrowborder" valign="top" width="70.43%" headers="mcps1.2.4.1.3 "><p id="p14814133518382"><a name="p14814133518382"></a><a name="p14814133518382"></a>用于设置参数enUnitFlag。</p>
<p id="p17614103216152"><a name="p17614103216152"></a><a name="p17614103216152"></a><span id="ph1065621641910"><a name="ph1065621641910"></a><a name="ph1065621641910"></a>使能UnitFlag功能，使计算与搬运流水并行，提高性能。Norm, IBShare下默认使能，MDL下默认不使能。参数取值如下：</span></p>
<a name="ul152765814315"></a><a name="ul152765814315"></a><ul id="ul152765814315"><li><span id="ph13657141617192"><a name="ph13657141617192"></a><a name="ph13657141617192"></a>false：不使能UnitFlag功能。</span></li><li><span id="ph166581616141912"><a name="ph166581616141912"></a><a name="ph166581616141912"></a>true：使能UnitFlag功能。</span></li></ul>
<p id="p6593921135211"><a name="p6593921135211"></a><a name="p6593921135211"></a><span id="ph10659101671916"><a name="ph10659101671916"></a><a name="ph10659101671916"></a>注意：MxMatmul场景，仅在NORM/MDL模板、A和scaleA不转置、 B和scaleB转置、C为ND格式，输出到GM场景下，使能UnitFlag功能有性能收益。</span></p>
</td>
</tr>
<tr id="row1356154144611"><td class="cellrowborder" valign="top" width="17.77%" headers="mcps1.2.4.1.1 "><p id="p135611241124617"><a name="p135611241124617"></a><a name="p135611241124617"></a>enableMixDualMaster</p>
</td>
<td class="cellrowborder" valign="top" width="11.799999999999999%" headers="mcps1.2.4.1.2 "><p id="p18561341134613"><a name="p18561341134613"></a><a name="p18561341134613"></a>输入</p>
</td>
<td class="cellrowborder" valign="top" width="70.43%" headers="mcps1.2.4.1.3 "><p id="p1936220475212"><a name="p1936220475212"></a><a name="p1936220475212"></a>用于设置参数enableMixDualMaster。</p>
<p id="p52967361495"><a name="p52967361495"></a><a name="p52967361495"></a><span id="ph54991243205416"><a name="ph54991243205416"></a><a name="ph54991243205416"></a>是否使能MixDualMaster（双主模式）。区别于MIX模式（包含矩阵计算和矢量计算）通过消息机制驱动AIC运行，双主模式为AIC和AIV独立运行代码，不依赖消息驱动，用于提升性能。该参数默认值为false，仅能在以下场景设置为true：</span></p>
<a name="ul1084411362491"></a><a name="ul1084411362491"></a><ul id="ul1084411362491"><li><span id="ph8576175111487"><a name="ph8576175111487"></a><a name="ph8576175111487"></a>核函数的类型为MIX，同时AIC核数 : AIV核数为1:1。</span></li><li><span id="ph1257711515487"><a name="ph1257711515487"></a><a name="ph1257711515487"></a>核函数的类型为MIX，同时AIC核数 : AIV核数为1:2，且A矩阵和B矩阵同时使能<a href="Matmul使用说明.md#table1188045714378">IBSHARE</a>参数。</span></li></ul>
<p id="p5483145714219"><a name="p5483145714219"></a><a name="p5483145714219"></a><span id="ph1757813519483"><a name="ph1757813519483"></a><a name="ph1757813519483"></a>注意，使能MixDualMaster场景，需要满足：</span></p>
<a name="ul154831557184220"></a><a name="ul154831557184220"></a><ul id="ul154831557184220"><li><span id="ph205799517484"><a name="ph205799517484"></a><a name="ph205799517484"></a>同一算子中所有Matmul对象的该参数取值必须保持一致。</span></li><li><span id="ph164746210205"><a name="ph164746210205"></a><a name="ph164746210205"></a>A/B/Bias矩阵只支持从GM搬入。</span></li><li><span id="ph1458055114488"><a name="ph1458055114488"></a><a name="ph1458055114488"></a>获取矩阵计算结果只支持调用<a href="IterateAll.md">IterateAll</a>接口输出到GlobalTensor<span id="zh-cn_topic_0000002523303586_ph20219141053713"><a name="zh-cn_topic_0000002523303586_ph20219141053713"></a><a name="zh-cn_topic_0000002523303586_ph20219141053713"></a>或者LocalTensor</span>，即计算结果放置于Global Memory<span id="zh-cn_topic_0000002523303586_ph3219131053714"><a name="zh-cn_topic_0000002523303586_ph3219131053714"></a><a name="zh-cn_topic_0000002523303586_ph3219131053714"></a><span id="zh-cn_topic_0000002523303586_ph42191810133716"><a name="zh-cn_topic_0000002523303586_ph42191810133716"></a><a name="zh-cn_topic_0000002523303586_ph42191810133716"></a>或者Local Memory </span></span>的地址，不能调用<a href="GetTensorC.md">GetTensorC</a>等接口获取结果。</span></li></ul>
<p id="p1622091073715"><a name="p1622091073715"></a><a name="p1622091073715"></a><span id="ph14220151023713"><a name="ph14220151023713"></a><a name="ph14220151023713"></a><span id="zh-cn_topic_0000002523303586_ph1422071093713"><a name="zh-cn_topic_0000002523303586_ph1422071093713"></a><a name="zh-cn_topic_0000002523303586_ph1422071093713"></a>Ascend 950PR/Ascend 950DT</span>支持该参数。</span></p>
</td>
</tr>
<tr id="row13470184301"><td class="cellrowborder" valign="top" width="17.77%" headers="mcps1.2.4.1.1 "><p id="p16347131818306"><a name="p16347131818306"></a><a name="p16347131818306"></a>bmmOutMode</p>
</td>
<td class="cellrowborder" valign="top" width="11.799999999999999%" headers="mcps1.2.4.1.2 "><p id="p134721813302"><a name="p134721813302"></a><a name="p134721813302"></a>输入</p>
</td>
<td class="cellrowborder" valign="top" width="70.43%" headers="mcps1.2.4.1.3 "><p id="p1534711812302"><a name="p1534711812302"></a><a name="p1534711812302"></a>预留参数。</p>
</td>
</tr>
</tbody>
</table>

## 返回值说明<a name="section640mcpsimp"></a>

[MatmulConfig结构体](MatmulConfig.md#table1761013213153)。

## 约束说明<a name="section633mcpsimp"></a>

无

## 调用示例<a name="section1665082013318"></a>

```
// 配置Norm模板的参数，获取自定义Norm模板
constexpr MatmulConfig MM_CFG = GetNormalConfig(
    /* intrinsicsLimit   */ false,
    /* batchLoop         */ false,
    /* isVecND2NZ        */ false,
    /* bmmMode           */ BatchMode::BATCH_LESS_THAN_L1,
    /* isMsgReuse        */ true,
    /* iterateOrder      */ IterateOrder::UNDEF,
    /* scheduleType      */ ScheduleType::INNER_PRODUCT,
    /* enUnitFlag        */ true
    /* enableMixDualMaster */ false,
    /* bmmOutMode        */ BatchOutMode::SINGLE_BATCH
);
// 常规Matmul计算，最后输出使用自定义Norm模板的计算结果
AscendC::Matmul<A_TYPE, B_TYPE, C_TYPE, BIAS_TYPE, MM_CFG> mm;
REGIST_MATMUL_OBJ(&pipe, GetSysWorkSpacePtr(), mm, &tiling);
mm.SetTensorA(gm_a);
mm.SetTensorB(gm_b);
if (tiling.isBias) {
    mm.SetBias(gmBias);
}
mm.IterateAll(gm_c);
mm.End();
```

