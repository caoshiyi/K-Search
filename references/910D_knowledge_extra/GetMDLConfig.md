# GetMDLConfig<a name="ZH-CN_TOPIC_0000002523343804"></a>

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

用于配置MDL模板的参数，获取自定义MDL模板。MDL模板的介绍请参考[表 模板特性](MatmulConfig.md#table6981133810309)。

## 函数原型<a name="section620mcpsimp"></a>

```
__aicore__ constexpr MatmulConfig GetMDLConfig(const bool intrinsicsLimit = false, const bool batchLoop = false, const uint32_t doMTE2Preload = 0, const bool isVecND2NZ = false, bool isPerTensor = false, bool hasAntiQuantOffset = false, const bool enUnitFlag = false, const bool isMsgReuse = true, const bool enableUBReuse = true, const bool enableL1CacheUB = false, const bool enableMixDualMaster = false, const bool enableKdimReorderLoad = false)
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
<tr id="row034818514267"><td class="cellrowborder" valign="top" width="17.77%" headers="mcps1.2.4.1.1 "><p id="p7348155182618"><a name="p7348155182618"></a><a name="p7348155182618"></a>doMTE2Preload</p>
</td>
<td class="cellrowborder" valign="top" width="11.799999999999999%" headers="mcps1.2.4.1.2 "><p id="p4348175118269"><a name="p4348175118269"></a><a name="p4348175118269"></a>输入</p>
</td>
<td class="cellrowborder" valign="top" width="70.43%" headers="mcps1.2.4.1.3 "><p id="p852877193213"><a name="p852877193213"></a><a name="p852877193213"></a>用于设置参数doMTE2Preload。</p>
<p id="p13613432131512"><a name="p13613432131512"></a><a name="p13613432131512"></a><span id="ph1798274510297"><a name="ph1798274510297"></a><a name="ph1798274510297"></a>在MTE2流水间隙较大，且M/N数值较大时可通过该参数开启对应M/N方向的预加载功能，开启后能减小MTE2间隙，提升性能。预加载功能仅在MDL模板有效（不支持SpecialMDL模板）。参数取值如下：</span></p>
<a name="ul20613193210152"></a><a name="ul20613193210152"></a><ul id="ul20613193210152"><li><span id="ph99831845122917"><a name="ph99831845122917"></a><a name="ph99831845122917"></a>0：不开启（默认值）。</span></li><li><span id="ph99851455294"><a name="ph99851455294"></a><a name="ph99851455294"></a>1：开启M方向preload。</span></li><li><span id="ph998511455299"><a name="ph998511455299"></a><a name="ph998511455299"></a>2：开启N方向preload。</span></li></ul>
<p id="p126131532141513"><a name="p126131532141513"></a><a name="p126131532141513"></a><span id="ph13986545162913"><a name="ph13986545162913"></a><a name="ph13986545162913"></a>注意：开启M/N方向的预加载功能时需保证K全载且M/N方向开启<a href="DoubleBuffer.md">DoubleBuffer</a>；其中，M方向的K全载条件为：singleCoreK/baseK &lt;= stepKa；N方向的K全载条件为：singleCoreK/baseK &lt;= stepKb。</span></p>
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
<tr id="row149492814278"><td class="cellrowborder" valign="top" width="17.77%" headers="mcps1.2.4.1.1 "><p id="p169499882715"><a name="p169499882715"></a><a name="p169499882715"></a>isPerTensor</p>
</td>
<td class="cellrowborder" valign="top" width="11.799999999999999%" headers="mcps1.2.4.1.2 "><p id="p994948122717"><a name="p994948122717"></a><a name="p994948122717"></a>输入</p>
</td>
<td class="cellrowborder" valign="top" width="70.43%" headers="mcps1.2.4.1.3 "><p id="p149497882720"><a name="p149497882720"></a><a name="p149497882720"></a>用于设置参数isPerTensor。</p>
<p id="p22601175383"><a name="p22601175383"></a><a name="p22601175383"></a><span id="ph346512199361"><a name="ph346512199361"></a><a name="ph346512199361"></a>A矩阵half类型输入且B矩阵int8_t类型输入场景，使能B矩阵量化时是否为per tensor。</span></p>
<a name="ul19611220113813"></a><a name="ul19611220113813"></a><ul id="ul19611220113813"><li><span id="ph022382416360"><a name="ph022382416360"></a><a name="ph022382416360"></a>true：per tensor量化。</span></li><li><span id="ph174121537163315"><a name="ph174121537163315"></a><a name="ph174121537163315"></a>false：per channel量化。</span></li></ul>
<p id="p171131416165420"><a name="p171131416165420"></a><a name="p171131416165420"></a><span id="ph24131137133319"><a name="ph24131137133319"></a><a name="ph24131137133319"></a>注意：MxMatmul场景仅支持false。</span></p>
</td>
</tr>
<tr id="row1449551110277"><td class="cellrowborder" valign="top" width="17.77%" headers="mcps1.2.4.1.1 "><p id="p1949551112277"><a name="p1949551112277"></a><a name="p1949551112277"></a>hasAntiQuantOffset</p>
</td>
<td class="cellrowborder" valign="top" width="11.799999999999999%" headers="mcps1.2.4.1.2 "><p id="p18495161142718"><a name="p18495161142718"></a><a name="p18495161142718"></a>输入</p>
</td>
<td class="cellrowborder" valign="top" width="70.43%" headers="mcps1.2.4.1.3 "><p id="p1949591122712"><a name="p1949591122712"></a><a name="p1949591122712"></a>用于设置参数hasAntiQuantOffset。</p>
<p id="p116141032121517"><a name="p116141032121517"></a><a name="p116141032121517"></a><span id="ph166961850950"><a name="ph166961850950"></a><a name="ph166961850950"></a>A矩阵half类型输入且B矩阵int8_t类型输入场景，使能B矩阵量化时是否使用offset系数。</span></p>
<p id="p142014566541"><a name="p142014566541"></a><a name="p142014566541"></a><span id="ph1697950658"><a name="ph1697950658"></a><a name="ph1697950658"></a>注意：MxMatmul场景仅支持false。</span></p>
</td>
</tr>
<tr id="row13253132122812"><td class="cellrowborder" valign="top" width="17.77%" headers="mcps1.2.4.1.1 "><p id="p13190022142812"><a name="p13190022142812"></a><a name="p13190022142812"></a>enUnitFlag</p>
</td>
<td class="cellrowborder" valign="top" width="11.799999999999999%" headers="mcps1.2.4.1.2 "><p id="p181901822122818"><a name="p181901822122818"></a><a name="p181901822122818"></a>输入</p>
</td>
<td class="cellrowborder" valign="top" width="70.43%" headers="mcps1.2.4.1.3 "><p id="p191904228282"><a name="p191904228282"></a><a name="p191904228282"></a>用于设置参数enUnitFlag。</p>
<p id="p2190182211286"><a name="p2190182211286"></a><a name="p2190182211286"></a><span id="ph21901922142815"><a name="ph21901922142815"></a><a name="ph21901922142815"></a>使能UnitFlag功能，使计算与搬运流水并行，提高性能。Norm, IBShare下默认使能，MDL下默认不使能。参数取值如下：</span></p>
<a name="ul191901222287"></a><a name="ul191901222287"></a><ul id="ul191901222287"><li><span id="ph519011223283"><a name="ph519011223283"></a><a name="ph519011223283"></a>false：不使能UnitFlag功能。</span></li><li><span id="ph16190122142816"><a name="ph16190122142816"></a><a name="ph16190122142816"></a>true：使能UnitFlag功能。</span></li></ul>
<p id="p15190222182814"><a name="p15190222182814"></a><a name="p15190222182814"></a><span id="ph61901422102812"><a name="ph61901422102812"></a><a name="ph61901422102812"></a>注意：MxMatmul场景，仅在NORM/MDL模板、A和scaleA不转置、 B和scaleB转置、C为ND格式，输出到GM场景下，使能UnitFlag功能有性能收益。</span></p>
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
<tr id="row128916389284"><td class="cellrowborder" valign="top" width="17.77%" headers="mcps1.2.4.1.1 "><p id="p828983815281"><a name="p828983815281"></a><a name="p828983815281"></a>enableUBReuse</p>
</td>
<td class="cellrowborder" valign="top" width="11.799999999999999%" headers="mcps1.2.4.1.2 "><p id="p928920387288"><a name="p928920387288"></a><a name="p928920387288"></a>输入</p>
</td>
<td class="cellrowborder" valign="top" width="70.43%" headers="mcps1.2.4.1.3 "><p id="p9289838112817"><a name="p9289838112817"></a><a name="p9289838112817"></a>用于设置参数enableUBReuse。</p>
<p id="p143891657133818"><a name="p143891657133818"></a><a name="p143891657133818"></a><span id="ph101741817077"><a name="ph101741817077"></a><a name="ph101741817077"></a>是否使能<span id="zh-cn_topic_0000002523303586_ph16424826132717"><a name="zh-cn_topic_0000002523303586_ph16424826132717"></a><a name="zh-cn_topic_0000002523303586_ph16424826132717"></a>Unified Buffer</span>复用。在<span id="zh-cn_topic_0000002523303586_ph1142432622711"><a name="zh-cn_topic_0000002523303586_ph1142432622711"></a><a name="zh-cn_topic_0000002523303586_ph1142432622711"></a><span id="zh-cn_topic_0000002523303586_ph042419261271"><a name="zh-cn_topic_0000002523303586_ph042419261271"></a><a name="zh-cn_topic_0000002523303586_ph042419261271"></a>Unified Buffer</span></span>空间足够的条件下（<span id="zh-cn_topic_0000002523303586_ph15424126162717"><a name="zh-cn_topic_0000002523303586_ph15424126162717"></a><a name="zh-cn_topic_0000002523303586_ph15424126162717"></a><span id="zh-cn_topic_0000002523303586_ph16424192614271"><a name="zh-cn_topic_0000002523303586_ph16424192614271"></a><a name="zh-cn_topic_0000002523303586_ph16424192614271"></a>Unified Buffer</span></span>空间大于4倍TCubeTiling的<a href="TCubeTiling结构体.md#p1620315053211">transLength</a>参数），使能该参数后，<span id="zh-cn_topic_0000002523303586_ph34246261279"><a name="zh-cn_topic_0000002523303586_ph34246261279"></a><a name="zh-cn_topic_0000002523303586_ph34246261279"></a><span id="zh-cn_topic_0000002523303586_ph154242264276"><a name="zh-cn_topic_0000002523303586_ph154242264276"></a><a name="zh-cn_topic_0000002523303586_ph154242264276"></a>Unified Buffer</span></span>空间分为互不重叠的两份，分别存储Matmul计算相邻前后两轮迭代的数据，后一轮迭代数据的搬入将不必等待前一轮迭代的<span id="zh-cn_topic_0000002523303586_ph542410265274"><a name="zh-cn_topic_0000002523303586_ph542410265274"></a><a name="zh-cn_topic_0000002523303586_ph542410265274"></a><span id="zh-cn_topic_0000002523303586_ph1424192616274"><a name="zh-cn_topic_0000002523303586_ph1424192616274"></a><a name="zh-cn_topic_0000002523303586_ph1424192616274"></a>Unified Buffer</span></span>空间释放，<span>从而优化流水</span>。参数取值如下：</span></p>
<a name="ul1638914579380"></a><a name="ul1638914579380"></a><ul id="ul1638914579380"><li><span id="ph10175417572"><a name="ph10175417572"></a><a name="ph10175417572"></a>true：使能<span id="zh-cn_topic_0000002523303586_ph11424182614272"><a name="zh-cn_topic_0000002523303586_ph11424182614272"></a><a name="zh-cn_topic_0000002523303586_ph11424182614272"></a>Unified Buffer</span>复用。</span></li><li><span id="ph121766179713"><a name="ph121766179713"></a><a name="ph121766179713"></a>false：不使能<span id="zh-cn_topic_0000002523303586_ph17424526112712"><a name="zh-cn_topic_0000002523303586_ph17424526112712"></a><a name="zh-cn_topic_0000002523303586_ph17424526112712"></a>Unified Buffer</span>复用。</span></li></ul>
<p id="p12425426102711"><a name="p12425426102711"></a><a name="p12425426102711"></a><span id="ph1742511260278"><a name="ph1742511260278"></a><a name="ph1742511260278"></a><span id="zh-cn_topic_0000002523303586_ph8425526132713"><a name="zh-cn_topic_0000002523303586_ph8425526132713"></a><a name="zh-cn_topic_0000002523303586_ph8425526132713"></a>Ascend 950PR/Ascend 950DT</span>不支持该参数。</span></p>
</td>
</tr>
<tr id="row770164742811"><td class="cellrowborder" valign="top" width="17.77%" headers="mcps1.2.4.1.1 "><p id="p27034714281"><a name="p27034714281"></a><a name="p27034714281"></a>enableL1CacheUB</p>
</td>
<td class="cellrowborder" valign="top" width="11.799999999999999%" headers="mcps1.2.4.1.2 "><p id="p11705475284"><a name="p11705475284"></a><a name="p11705475284"></a>输入</p>
</td>
<td class="cellrowborder" valign="top" width="70.43%" headers="mcps1.2.4.1.3 "><p id="p170134720282"><a name="p170134720282"></a><a name="p170134720282"></a>用于设置参数enableL1CacheUB 。</p>
<p id="p113897573386"><a name="p113897573386"></a><a name="p113897573386"></a><span id="ph13491048293"><a name="ph13491048293"></a><a name="ph13491048293"></a>是否使能<span id="zh-cn_topic_0000002523303586_ph134251126162713"><a name="zh-cn_topic_0000002523303586_ph134251126162713"></a><a name="zh-cn_topic_0000002523303586_ph134251126162713"></a>L1 Buffer</span>缓存<span id="zh-cn_topic_0000002523303586_ph1942512616274"><a name="zh-cn_topic_0000002523303586_ph1942512616274"></a><a name="zh-cn_topic_0000002523303586_ph1942512616274"></a>Unified Buffer</span>计算块。建议在MTE3和MTE2流水串行较多的场景使用。参数取值如下：</span></p>
<a name="ul93891572381"></a><a name="ul93891572381"></a><ul id="ul93891572381"><li><span id="ph20492124815913"><a name="ph20492124815913"></a><a name="ph20492124815913"></a>true：使能<span id="zh-cn_topic_0000002523303586_ph17425626102719"><a name="zh-cn_topic_0000002523303586_ph17425626102719"></a><a name="zh-cn_topic_0000002523303586_ph17425626102719"></a>L1 Buffer</span>缓存<span id="zh-cn_topic_0000002523303586_ph8425122642711"><a name="zh-cn_topic_0000002523303586_ph8425122642711"></a><a name="zh-cn_topic_0000002523303586_ph8425122642711"></a>Unified Buffer</span>计算块。</span></li><li><span id="ph19493154816920"><a name="ph19493154816920"></a><a name="ph19493154816920"></a>false：不使能<span id="zh-cn_topic_0000002523303586_ph8425122662719"><a name="zh-cn_topic_0000002523303586_ph8425122662719"></a><a name="zh-cn_topic_0000002523303586_ph8425122662719"></a>L1 Buffer</span>缓存<span id="zh-cn_topic_0000002523303586_ph7425826112719"><a name="zh-cn_topic_0000002523303586_ph7425826112719"></a><a name="zh-cn_topic_0000002523303586_ph7425826112719"></a>Unified Buffer</span>计算块。</span></li></ul>
<p id="p18609101105210"><a name="p18609101105210"></a><a name="p18609101105210"></a><span id="ph194931481792"><a name="ph194931481792"></a><a name="ph194931481792"></a>若要使能<span id="zh-cn_topic_0000002523303586_ph74261826152710"><a name="zh-cn_topic_0000002523303586_ph74261826152710"></a><a name="zh-cn_topic_0000002523303586_ph74261826152710"></a>L1 Buffer</span>缓存<span id="zh-cn_topic_0000002523303586_ph20426192632719"><a name="zh-cn_topic_0000002523303586_ph20426192632719"></a><a name="zh-cn_topic_0000002523303586_ph20426192632719"></a>Unified Buffer</span>计算块，必须在Tiling实现中调用<a href="SetMatmulConfigParams.md">SetMatmulConfigParams</a>接口将参数enableL1CacheUBIn设置为true。</span></p>
<p id="p0426182662710"><a name="p0426182662710"></a><a name="p0426182662710"></a><span id="ph144261226202715"><a name="ph144261226202715"></a><a name="ph144261226202715"></a><span id="zh-cn_topic_0000002523303586_ph164261026142710"><a name="zh-cn_topic_0000002523303586_ph164261026142710"></a><a name="zh-cn_topic_0000002523303586_ph164261026142710"></a>Ascend 950PR/Ascend 950DT</span>不支持该参数。</span></p>
</td>
</tr>
<tr id="row1868618161807"><td class="cellrowborder" valign="top" width="17.77%" headers="mcps1.2.4.1.1 "><p id="p135611241124617"><a name="p135611241124617"></a><a name="p135611241124617"></a>enableMixDualMaster</p>
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
<tr id="row16491111317188"><td class="cellrowborder" valign="top" width="17.77%" headers="mcps1.2.4.1.1 "><p id="p849215131188"><a name="p849215131188"></a><a name="p849215131188"></a>enableKdimReorderLoad</p>
</td>
<td class="cellrowborder" valign="top" width="11.799999999999999%" headers="mcps1.2.4.1.2 "><p id="p249241371811"><a name="p249241371811"></a><a name="p249241371811"></a>输入</p>
</td>
<td class="cellrowborder" valign="top" width="70.43%" headers="mcps1.2.4.1.3 "><p id="p172232047663"><a name="p172232047663"></a><a name="p172232047663"></a>用于设置参数enableKdimReorderLoad。</p>
<p id="p95376407818"><a name="p95376407818"></a><a name="p95376407818"></a><span id="ph10909208814"><a name="ph10909208814"></a><a name="ph10909208814"></a>是否使能K轴错峰加载数据。基于相同Tiling参数，执行Matmul计算时，如果多核的左矩阵或者右矩阵相同，且存储于Global Memory，多个核一般会同时访问相同地址以加载矩阵数据，引发同地址访问冲突，影响性能。使能该参数后，多核执行Matmul时，将尽量在相同时间访问矩阵的不同Global Memory地址，减少地址访问冲突概率，提升性能。该参数功能只支持MDL模板，建议K轴较大且左矩阵和右矩阵均非全载场景使能参数。参数取值如下。</span></p>
<a name="ul6537104018815"></a><a name="ul6537104018815"></a><ul id="ul6537104018815"><li><span id="ph053717401382"><a name="ph053717401382"></a><a name="ph053717401382"></a>false：默认值，关闭K轴错峰加载数据的功能。</span></li><li><span id="ph9537040489"><a name="ph9537040489"></a><a name="ph9537040489"></a>true：开启K轴错峰加载数据的功能。</span></li></ul>
<p id="p1234344315377"><a name="p1234344315377"></a><a name="p1234344315377"></a><span id="ph634394333710"><a name="ph634394333710"></a><a name="ph634394333710"></a><span id="zh-cn_topic_0000002523303586_ph5343204318379"><a name="zh-cn_topic_0000002523303586_ph5343204318379"></a><a name="zh-cn_topic_0000002523303586_ph5343204318379"></a>Ascend 950PR/Ascend 950DT</span>支持该参数。</span></p>
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
// 配置MDL模板的参数，获取自定义MDL模板
constexpr MatmulConfig MM_CFG = GetMDLConfig(false, false, 0, false, false, false, false, true, true, false, false, false);
// 常规Matmul计算，最后输出使用自定义MDL模板的计算结果
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

