# CrossCoreWaitFlag\(ISASI\)<a name="ZH-CN_TOPIC_0000002523344838"></a>

## 产品支持情况<a name="section1550532418810"></a>

<a name="table38301303189"></a>
<table><thead align="left"><tr id="row20831180131817"><th class="cellrowborder" valign="top" width="57.99999999999999%" id="mcps1.1.3.1.1"><p id="p1883113061818"><a name="p1883113061818"></a><a name="p1883113061818"></a><span id="ph20833205312295"><a name="ph20833205312295"></a><a name="ph20833205312295"></a>产品</span></p>
</th>
<th class="cellrowborder" align="center" valign="top" width="42%" id="mcps1.1.3.1.2"><p id="p783113012187"><a name="p783113012187"></a><a name="p783113012187"></a>是否支持</p>
</th>
</tr>
</thead>
<tbody><tr id="row1272474920205"><td class="cellrowborder" valign="top" width="57.99999999999999%" headers="mcps1.1.3.1.1 "><p id="p17301775812"><a name="p17301775812"></a><a name="p17301775812"></a><span id="ph2272194216543"><a name="ph2272194216543"></a><a name="ph2272194216543"></a>Ascend 950PR/Ascend 950DT</span></p>
</td>
<td class="cellrowborder" align="center" valign="top" width="42%" headers="mcps1.1.3.1.2 "><p id="p37256491200"><a name="p37256491200"></a><a name="p37256491200"></a>√</p>
</td>
</tr>
</tbody>
</table>

## 功能说明<a name="section618mcpsimp"></a>

面向分离模式的核间同步控制接口。该接口和[CrossCoreSetFlag](CrossCoreSetFlag(ISASI).md)接口配合使用。具体使用方法请参考[CrossCoreSetFlag](CrossCoreSetFlag(ISASI).md)。

## 函数原型<a name="section620mcpsimp"></a>

```
template <uint8_t modeId = 0, pipe_t pipe = PIPE_S>
__aicore__ inline void CrossCoreWaitFlag(uint16_t flagId)
```

## 参数说明<a name="section622mcpsimp"></a>

**表 1**  模板参数说明

<a name="table4835205712588"></a>
<table><thead align="left"><tr id="zh-cn_topic_0000001429830437_row118356578583"><th class="cellrowborder" valign="top" width="18.6%" id="mcps1.2.3.1.1"><p id="zh-cn_topic_0000001429830437_p48354572582"><a name="zh-cn_topic_0000001429830437_p48354572582"></a><a name="zh-cn_topic_0000001429830437_p48354572582"></a>参数名</p>
</th>
<th class="cellrowborder" valign="top" width="81.39999999999999%" id="mcps1.2.3.1.2"><p id="zh-cn_topic_0000001429830437_p583535795817"><a name="zh-cn_topic_0000001429830437_p583535795817"></a><a name="zh-cn_topic_0000001429830437_p583535795817"></a>描述</p>
</th>
</tr>
</thead>
<tbody><tr id="zh-cn_topic_0000001429830437_row1835857145817"><td class="cellrowborder" valign="top" width="18.6%" headers="mcps1.2.3.1.1 "><p id="zh-cn_topic_0000001429830437_p5835457165816"><a name="zh-cn_topic_0000001429830437_p5835457165816"></a><a name="zh-cn_topic_0000001429830437_p5835457165816"></a>modeId</p>
</td>
<td class="cellrowborder" valign="top" width="81.39999999999999%" headers="mcps1.2.3.1.2 "><p id="zh-cn_topic_0000001429830437_p168351657155818"><a name="zh-cn_topic_0000001429830437_p168351657155818"></a><a name="zh-cn_topic_0000001429830437_p168351657155818"></a>核间同步的模式，取值如下：</p>
<a name="ul335269516"></a><a name="ul335269516"></a><ul id="ul335269516"><li>模式0：AI Core核间的同步控制。</li><li>模式1：AI Core内部，Vector核（AIV）之间的同步控制。</li><li>模式2：AI Core内部，Cube核（AIC）与Vector核（AIV）之间的同步控制。</li><li>模式4：AI Core内部，AIC与AIV之间的同步控制。AIV0与AIV1可单独触发AIC等待。</li></ul>
</td>
</tr>
<tr id="row168561422132317"><td class="cellrowborder" valign="top" width="18.6%" headers="mcps1.2.3.1.1 "><p id="p1162623072316"><a name="p1162623072316"></a><a name="p1162623072316"></a>pipe</p>
</td>
<td class="cellrowborder" valign="top" width="81.39999999999999%" headers="mcps1.2.3.1.2 "><p id="p13626330102311"><a name="p13626330102311"></a><a name="p13626330102311"></a>设置这条指令所在的流水类型，流水类型可参考<a href="同步控制简介.md#section1272612276459">硬件流水类型</a>。</p>
<p id="p8834141517517"><a name="p8834141517517"></a><a name="p8834141517517"></a>特别地，PIPE_S流水类型仅<span id="ph1572119291978"><a name="ph1572119291978"></a><a name="ph1572119291978"></a>Ascend 950PR/Ascend 950DT</span>支持。</p>
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
<tbody><tr id="row42461942101815"><td class="cellrowborder" valign="top" width="18.54%" headers="mcps1.2.4.1.1 "><p id="p479605232211"><a name="p479605232211"></a><a name="p479605232211"></a>flagId</p>
</td>
<td class="cellrowborder" valign="top" width="10.05%" headers="mcps1.2.4.1.2 "><p id="p1579635215228"><a name="p1579635215228"></a><a name="p1579635215228"></a>输入</p>
</td>
<td class="cellrowborder" valign="top" width="71.41%" headers="mcps1.2.4.1.3 "><p id="p10993195419912"><a name="p10993195419912"></a><a name="p10993195419912"></a>核间同步的标记。</p>
<p id="p193019567919"><a name="p193019567919"></a><a name="p193019567919"></a><span id="ph9301256291"><a name="ph9301256291"></a><a name="ph9301256291"></a>Ascend 950PR/Ascend 950DT</span>，取值范围如下：</p>
<p id="p4858107124612"><a name="p4858107124612"></a><a name="p4858107124612"></a>AIV0发起的flagId 0-10的CrossCoreSetFlag操作对应AIC CrossCoreWaitFlag中flagId 0-10的操作。</p>
<p id="p1685813713461"><a name="p1685813713461"></a><a name="p1685813713461"></a>AIV1发起的flagId 0-10的CrossCoreSetFlag操作对应AIC CrossCoreWaitFlag中flagId 16-26的操作。</p>
<p id="p4858974461"><a name="p4858974461"></a><a name="p4858974461"></a>AIC发起的flagId 0-10的CrossCoreSetFlag操作对应AIV0 CrossCoreWaitFlag中flagId 0-10的操作。</p>
<p id="p1785837134610"><a name="p1785837134610"></a><a name="p1785837134610"></a>AIC发起的flagId 16-26的CrossCoreSetFlag操作对应AIV1 CrossCoreWaitFlag中flagId 0-10的操作。</p>
</td>
</tr>
</tbody>
</table>

## 返回值说明<a name="section640mcpsimp"></a>

无

## 约束说明<a name="section633mcpsimp"></a>

-   使用该同步接口时，需要按照如下规则[设置Kernel类型](设置Kernel类型.md)：
    -   在纯Vector/Cube场景下，需设置Kernel类型为KERNEL\_TYPE\_MIX\_AIV\_1\_0或KERNEL\_TYPE\_MIX\_AIC\_1\_0。
    -   对于Vector和Cube混合场景，需根据实际情况灵活配置Kernel类型。

-   CrossCoreWaitFlag必须与[CrossCoreSetFlag](CrossCoreSetFlag(ISASI).md)接口配合使用，避免计算核一直处于阻塞阶段。
-   如果执行CrossCoreWaitFlag时该flagId的计数器的值为0，则CrossCoreWaitFlag之后的所有指令都将被阻塞，直到该flagId的计数器的值不为0。同一个flagId的计数器最多设置15次。
-   使用该接口模式0时，建议开启batchmode模式，使算子独占全部所需核资源，否则可能因满足以下条件导致死锁：

    -   多流并发场景（≥2条执行流）。
    -   ≥2个算子并发执行。
    -   所有并发算子的核数总和超过物理核数。
    -   ≥2个并发算子使用了核间同步功能。

    具体而言，在多流场景下，某条流的核间同步算子虽分配到n个物理核，但可能仅有n-m个核先被调度执行，而其余m个核因被其他流的核间同步算子抢占而尚未启动。先启动的n-m个核执行到核间同步时等待剩余m核完成，而剩余m核因被其他流的核间同步算子占用而无法释放，形成死锁。

    Kernel直调场景下通过[\_\_schedmode\_\_\(mode\)](SIMD-BuiltIn关键字和API.md#li1365012910475)限定符来设置batchmode模式；工程化算子开发场景下，通过TilingContext的SetScheduleMode接口来设置batchmode模式，具体请参考《基础数据结构和接口》[《基础数据结构和接口》](zh-cn_topic_0000002554332575.md)。

## 调用示例<a name="section837496171220"></a>

请参考[调用示例](CrossCoreSetFlag(ISASI).md#section837496171220)。

