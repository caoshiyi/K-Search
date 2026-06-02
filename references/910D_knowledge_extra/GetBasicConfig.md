# GetBasicConfig<a name="ZH-CN_TOPIC_0000002523344180"></a>

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

用于配置BasicBlock模板的参数，获取自定义BasicBlock模板。BasicBlock模板的介绍请参考[表 模板特性](MatmulConfig.md#table6981133810309)。

使用该接口时可以优先考虑使用模板常量化。相比BasicBlock模板仅实现baseM、baseN、baseK常量化，模板常量化可以在此基础上实现singleCoreM、singleCoreN、singleCoreK、baseM、baseN、baseK的常量化，模板常量化的具体使用方式请参考[Matmul Tiling常量化](GetMatmulApiTiling.md#section618mcpsimp)。

## 函数原型<a name="section620mcpsimp"></a>

```
__aicore__ constexpr MatmulConfig GetBasicConfig(const uint32_t basicM, const uint32_t basicN, const uint32_t basicK, const bool intrinsicsLimit = false, const bool batchLoop = false, const BatchMode bmmMode = BatchMode::BATCH_LESS_THAN_L1)
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
<tbody><tr id="row14726151462416"><td class="cellrowborder" valign="top" width="17.77%" headers="mcps1.2.4.1.1 "><p id="p11726614162415"><a name="p11726614162415"></a><a name="p11726614162415"></a>basicM</p>
</td>
<td class="cellrowborder" valign="top" width="11.799999999999999%" headers="mcps1.2.4.1.2 "><p id="p2072611143247"><a name="p2072611143247"></a><a name="p2072611143247"></a>输入</p>
</td>
<td class="cellrowborder" valign="top" width="70.43%" headers="mcps1.2.4.1.3 "><p id="p57261114122414"><a name="p57261114122414"></a><a name="p57261114122414"></a>用于设置参数basicM。</p>
<p id="p561173251510"><a name="p561173251510"></a><a name="p561173251510"></a><span id="ph108911580253"><a name="ph108911580253"></a><a name="ph108911580253"></a>与<a href="TCubeTiling结构体.md">TCubeTiling结构体</a>中的baseM参数含义相同，Matmul计算时base块M轴长度，以元素为单位。</span></p>
</td>
</tr>
<tr id="row3821161216246"><td class="cellrowborder" valign="top" width="17.77%" headers="mcps1.2.4.1.1 "><p id="p8821141219248"><a name="p8821141219248"></a><a name="p8821141219248"></a>basicN</p>
</td>
<td class="cellrowborder" valign="top" width="11.799999999999999%" headers="mcps1.2.4.1.2 "><p id="p1282141232419"><a name="p1282141232419"></a><a name="p1282141232419"></a>输入</p>
</td>
<td class="cellrowborder" valign="top" width="70.43%" headers="mcps1.2.4.1.3 "><p id="p98211612172418"><a name="p98211612172418"></a><a name="p98211612172418"></a>用于设置参数basicN。</p>
<p id="p46111132121513"><a name="p46111132121513"></a><a name="p46111132121513"></a><span id="ph781319299267"><a name="ph781319299267"></a><a name="ph781319299267"></a>与<a href="TCubeTiling结构体.md">TCubeTiling结构体</a>中的baseN参数含义相同，Matmul计算时base块N轴长度，以元素为单位。</span></p>
</td>
</tr>
<tr id="row156808105249"><td class="cellrowborder" valign="top" width="17.77%" headers="mcps1.2.4.1.1 "><p id="p3680151013243"><a name="p3680151013243"></a><a name="p3680151013243"></a>basicK</p>
</td>
<td class="cellrowborder" valign="top" width="11.799999999999999%" headers="mcps1.2.4.1.2 "><p id="p6680191019241"><a name="p6680191019241"></a><a name="p6680191019241"></a>输入</p>
</td>
<td class="cellrowborder" valign="top" width="70.43%" headers="mcps1.2.4.1.3 "><p id="p156801210162411"><a name="p156801210162411"></a><a name="p156801210162411"></a>用于设置参数basicK。</p>
<p id="p11611932111512"><a name="p11611932111512"></a><a name="p11611932111512"></a><span id="ph186378112276"><a name="ph186378112276"></a><a name="ph186378112276"></a>与<a href="TCubeTiling结构体.md">TCubeTiling结构体</a>中的baseK参数含义相同，Matmul计算时base块K轴长度，以元素为单位。</span></p>
</td>
</tr>
<tr id="row42461942101815"><td class="cellrowborder" valign="top" width="17.77%" headers="mcps1.2.4.1.1 "><p id="p163481714145518"><a name="p163481714145518"></a><a name="p163481714145518"></a>intrinsicsLimit</p>
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
<tr id="row13809141083811"><td class="cellrowborder" valign="top" width="17.77%" headers="mcps1.2.4.1.1 "><p id="p7810910113819"><a name="p7810910113819"></a><a name="p7810910113819"></a>bmmMode</p>
</td>
<td class="cellrowborder" valign="top" width="11.799999999999999%" headers="mcps1.2.4.1.2 "><p id="p120714113815"><a name="p120714113815"></a><a name="p120714113815"></a>输入</p>
</td>
<td class="cellrowborder" valign="top" width="70.43%" headers="mcps1.2.4.1.3 "><p id="p481014105388"><a name="p481014105388"></a><a name="p481014105388"></a>用于设置参数batchMode。该参数用于BatchMatmul场景，关于BatchMatmul的介绍请参考<a href="Batch-Matmul基础功能.md">Batch Matmul基础功能</a>。</p>
<p id="p059314332299"><a name="p059314332299"></a><a name="p059314332299"></a><span id="ph115931633172912"><a name="ph115931633172912"></a><a name="ph115931633172912"></a>BatchMatmul场景中Layout类型为NORMAL时，设置BatchMatmul输入A/B矩阵的多batch数据总和与<span id="zh-cn_topic_0000002523303586_ph6593193372910"><a name="zh-cn_topic_0000002523303586_ph6593193372910"></a><a name="zh-cn_topic_0000002523303586_ph6593193372910"></a>L1 Buffer</span>的大小关系。参数取值如下：</span></p>
<a name="ul10593183316296"></a><a name="ul10593183316296"></a><ul id="ul10593183316296"><li><span id="ph5593733142910"><a name="ph5593733142910"></a><a name="ph5593733142910"></a>BatchMode::BATCH_LESS_THAN_L1：多batch数据总和&lt;<span id="zh-cn_topic_0000002523303586_ph145931033102914"><a name="zh-cn_topic_0000002523303586_ph145931033102914"></a><a name="zh-cn_topic_0000002523303586_ph145931033102914"></a>L1 Buffer</span> Size；</span></li><li><span id="ph25931334295"><a name="ph25931334295"></a><a name="ph25931334295"></a>BatchMode::BATCH_LARGE_THAN_L1：多batch数据总和&gt;<span id="zh-cn_topic_0000002523303586_ph10593333142912"><a name="zh-cn_topic_0000002523303586_ph10593333142912"></a><a name="zh-cn_topic_0000002523303586_ph10593333142912"></a>L1 Buffer</span> Size；</span></li><li><span id="ph9593203310298"><a name="ph9593203310298"></a><a name="ph9593203310298"></a>BatchMode::SINGLE_LARGE_THAN_L1：单batch数据总和&gt;<span id="zh-cn_topic_0000002523303586_ph259363362918"><a name="zh-cn_topic_0000002523303586_ph259363362918"></a><a name="zh-cn_topic_0000002523303586_ph259363362918"></a>L1 Buffer</span> Size。</span></li></ul>
</td>
</tr>
</tbody>
</table>

## 返回值说明<a name="section640mcpsimp"></a>

[MatmulConfig结构体](MatmulConfig.md#table1761013213153)。

## 约束说明<a name="section633mcpsimp"></a>

-   使用本接口时，基本块大小baseM、baseN需满足：singleCoreM能被baseM整除，singleCoreN能被baseN整除。
-   本接口的参数basicM、basicN、basicK应与[TCubeTiling结构体](TCubeTiling结构体.md#p17899165811566)的baseM、baseN、baseK设置保持一致。

## 调用示例<a name="section1665082013318"></a>

BasicBlock模板的完整使用样例请参考[basic\_block\_matmul样例](https://gitcode.com/cann/asc-devkit/tree/master/examples/03_libraries/01_matrix/basic_block_matmul)。

```
// 配置BasicBlock模板的参数，获取自定义BasicBlock模板
constexpr MatmulConfig MM_CFG = GetBasicConfig(128, 256, 64, false, false, BatchMode::BATCH_LESS_THAN_L1); // baseM, baseN, baseK
// 常规Matmul计算，最后输出使用自定义BasicBlock模板的计算结果
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

