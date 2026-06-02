# GetIBShareNormConfig<a name="ZH-CN_TOPIC_0000002523344596"></a>

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

用于配置IBShare模板的参数，获取自定义IBShare模板。IBShare模板的介绍请参考[表 模板特性](MatmulConfig.md#table6981133810309)。

## 函数原型<a name="section620mcpsimp"></a>

```
__aicore__ constexpr MatmulConfig GetIBShareNormConfig(const bool intrinsicsLimit = false, const bool batchLoop = false, const bool isVecND2NZ = false, const BatchMode bmmMode = BatchMode::BATCH_LESS_THAN_L1, const bool isDoubleCache = false, const bool enUnitFlag = true)
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
<tr id="row1364824131818"><td class="cellrowborder" valign="top" width="17.77%" headers="mcps1.2.4.1.1 "><p id="p11641524141812"><a name="p11641524141812"></a><a name="p11641524141812"></a>isDoubleCache</p>
</td>
<td class="cellrowborder" valign="top" width="11.799999999999999%" headers="mcps1.2.4.1.2 "><p id="p46442420184"><a name="p46442420184"></a><a name="p46442420184"></a>输入</p>
</td>
<td class="cellrowborder" valign="top" width="70.43%" headers="mcps1.2.4.1.3 "><p id="p164162421819"><a name="p164162421819"></a><a name="p164162421819"></a>用于设置参数enableDoubleCache。</p>
<p id="p156677274213"><a name="p156677274213"></a><a name="p156677274213"></a><span id="ph137426373192"><a name="ph137426373192"></a><a name="ph137426373192"></a>开启IBShare模板后，在<span id="zh-cn_topic_0000002523303586_ph1899163892912"><a name="zh-cn_topic_0000002523303586_ph1899163892912"></a><a name="zh-cn_topic_0000002523303586_ph1899163892912"></a>L1 Buffer</span>上是否同时缓存两块数据。参数取值如下：</span></p>
<a name="ul991213147241"></a><a name="ul991213147241"></a><ul id="ul991213147241"><li><span id="ph10757237101911"><a name="ph10757237101911"></a><a name="ph10757237101911"></a>false：<span id="zh-cn_topic_0000002523303586_ph14992203818292"><a name="zh-cn_topic_0000002523303586_ph14992203818292"></a><a name="zh-cn_topic_0000002523303586_ph14992203818292"></a>L1 Buffer</span>上同时缓存一块数据（默认值）。</span></li><li><span id="ph6771113713194"><a name="ph6771113713194"></a><a name="ph6771113713194"></a>true：使能<span id="zh-cn_topic_0000002523303586_ph159923388291"><a name="zh-cn_topic_0000002523303586_ph159923388291"></a><a name="zh-cn_topic_0000002523303586_ph159923388291"></a>L1 Buffer</span>上同时缓存两块数据。</span></li></ul>
<p id="p5312142173716"><a name="p5312142173716"></a><a name="p5312142173716"></a><span id="ph2064785683619"><a name="ph2064785683619"></a><a name="ph2064785683619"></a>注意：该参数取值为true时，需要控制基本块大小，防止两块数据的缓存超过<span id="zh-cn_topic_0000002523303586_ph5992173842912"><a name="zh-cn_topic_0000002523303586_ph5992173842912"></a><a name="zh-cn_topic_0000002523303586_ph5992173842912"></a>L1 Buffer</span>大小限制。</span></p>
<p id="p45216710517"><a name="p45216710517"></a><a name="p45216710517"></a><span id="ph14521170515"><a name="ph14521170515"></a><a name="ph14521170515"></a><span id="zh-cn_topic_0000002523303586_ph999243810293"><a name="zh-cn_topic_0000002523303586_ph999243810293"></a><a name="zh-cn_topic_0000002523303586_ph999243810293"></a>Ascend 950PR/Ascend 950DT</span>仅支持取值为false。</span></p>
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
</tbody>
</table>

## 返回值说明<a name="section640mcpsimp"></a>

[MatmulConfig结构体](MatmulConfig.md#table1761013213153)。

## 约束说明<a name="section633mcpsimp"></a>

IBShare模板当前仅适用于MIX场景，不支持纯CUBE场景。

## 调用示例<a name="section1665082013318"></a>

```
// 配置IBShare模板的参数，获取自定义IBShare模板。
constexpr MatmulConfig MM_CFG = GetIBShareNormConfig(false, false, false, BatchMode::BATCH_LESS_THAN_L1, false, true);
// 常规Matmul计算，最后输出使用自定义IBShare模板的计算结果
typedef AscendC::MatmulType<AscendC::TPosition::GM, CubeFormat::ND, half> aType; 
typedef AscendC::MatmulType<AscendC::TPosition::GM, CubeFormat::ND, half, true/*开启矩阵转置*/, LayoutMode::NONE/*不使能BatchMatmul*/, true/*使能IBShare*/> bType; 
typedef AscendC::MatmulType<AscendC::TPosition::GM, CubeFormat::ND, float> cType; 
typedef AscendC::MatmulType<AscendC::TPosition::GM, CubeFormat::ND, float> biasType; 
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

