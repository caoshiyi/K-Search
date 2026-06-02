# GetSpecialMDLConfig<a name="ZH-CN_TOPIC_0000002554343679"></a>

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

用于配置SpecialMDL模板的参数，获取自定义SpecialMDL模板。SpecialMDL模板的介绍请参考[表 模板特性](MatmulConfig.md#table6981133810309)。

## 函数原型<a name="section620mcpsimp"></a>

```
__aicore__ constexpr MatmulConfig GetSpecialMDLConfig(const bool intrinsicsLimit = false, const bool batchLoop = false, const uint32_t doMTE2Preload = 0, const bool isVecND2NZ = false, bool isPerTensor = false, bool hasAntiQuantOffset = false)
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
<td class="cellrowborder" valign="top" width="70.43%" headers="mcps1.2.4.1.3 "><p id="p98238315389"><a name="p98238315389"></a><a name="p98238315389"></a>预留参数，保持默认值false即可。</p>
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
</tbody>
</table>

## 返回值说明<a name="section640mcpsimp"></a>

[MatmulConfig结构体](MatmulConfig.md#table1761013213153)。

## 约束说明<a name="section633mcpsimp"></a>

无

## 调用示例<a name="section1665082013318"></a>

```
// 配置SpecialMDL模板的参数，获取自定义SpecialMDL模板
constexpr MatmulConfig MM_CFG = GetSpecialMDLConfig(false, false, 0, false, false, false);
// 常规Matmul计算，最后输出使用自定义SpecialMDL模板的计算结果
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

