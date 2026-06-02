# CrossCoreSetFlag\(ISASI\)<a name="ZH-CN_TOPIC_0000002523303700"></a>

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

面向分离模式的核间同步控制接口。

该接口和[CrossCoreWaitFlag](CrossCoreWaitFlag(ISASI).md)接口配合使用。使用时需传入核间同步的标记ID\(flagId\)，每个ID对应一个用于控制同步的计数器。

同步控制分为以下几种模式，如[图1](#fig37581010773)所示：

-   模式0：AI Core核间的同步控制。对于AIC场景，同步所有的AIC核，直到所有的AIC核都执行到CrossCoreSetFlag时，CrossCoreWaitFlag后续的指令才会执行；对于AIV场景，同步所有的AIV核，直到所有的AIV核都执行到CrossCoreSetFlag时，CrossCoreWaitFlag后续的指令才会执行。
-   模式1：AI Core内部，AIV核之间的同步控制。如果两个AIV核都运行了CrossCoreSetFlag，CrossCoreWaitFlag后续的指令才会执行。
-   模式2：AI Core内部，AIC与AIV之间的同步控制。在AIC核执行CrossCoreSetFlag之后， 两个AIV上CrossCoreWaitFlag后续的指令才会继续执行；两个AIV都执行CrossCoreSetFlag后，AIC上CrossCoreWaitFlag后续的指令才能执行。
-   模式4：AI Core内部，AIC与AIV之间的同步控制。AIV0与AIV1可单独触发AIC等待。比如，在AIC核执行CrossCoreSetFlag之后， AIV0上CrossCoreWaitFlag后续的指令才会继续执行；AIV0执行CrossCoreSetFlag后，AIC上CrossCoreWaitFlag后续的指令才能执行。

其中，模式4仅支持Ascend 950PR/Ascend 950DT。

**图 1**  同步控制模式示意图<a name="fig37581010773"></a>  
<!-- img2text -->
```
┌──────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────┐
│ mode 0                          │                              │ mode 2                          │                            │
│                                  │                              │                                  │                            │
│      ┌──────────────┐            │      ┌──────────────┐        │      ┌──────────────┐            │      ┌──────────────┐      │
│      │    AIC 0     │            │      │    AIC 1     │        │      │    AIC 2     │            │      │   AIC N-1    │      │
│      └──────────────┘            │      └──────────────┘        │      └──────────────┘            │      └──────────────┘      │
├──────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────┤
│ mode 0                          │ mode 1                       │                                  │                            │
│                                  │                              │                                  │                            │
│  ┌──────────────┐ ┌──────────────┐│  ┌──────────────┐ ┌──────────────┐│  ┌──────────────┐ ┌──────────────┐│  ┌──────────────┐ ┌──────────────┐│
│  │   AIV 0-0    │ │   AIV 0-1    ││  │   AIV 1-0    │ │   AIV 1-1    ││  │   AIV 2-0    │ │   AIV 2-1    ││  │   AIV N-1-0  │ │   AIV N-1-1  ││
│  └──────────────┘ └──────────────┘│  └──────────────┘ └──────────────┘│  └──────────────┘ └──────────────┘│  └──────────────┘ └──────────────┘│
└──────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────┘
```

## 函数原型<a name="section620mcpsimp"></a>

```
template <uint8_t modeId, pipe_t pipe>
__aicore__ inline void CrossCoreSetFlag(uint16_t flagId)
```

## 参数说明<a name="section622mcpsimp"></a>

**表 1**  模板参数说明

<a name="table4835205712588"></a>
<table><thead align="left"><tr id="zh-cn_topic_0000001429830437_row118356578583"><th class="cellrowborder" valign="top" width="18.8%" id="mcps1.2.3.1.1"><p id="zh-cn_topic_0000001429830437_p48354572582"><a name="zh-cn_topic_0000001429830437_p48354572582"></a><a name="zh-cn_topic_0000001429830437_p48354572582"></a>参数名</p>
</th>
<th class="cellrowborder" valign="top" width="81.2%" id="mcps1.2.3.1.2"><p id="zh-cn_topic_0000001429830437_p583535795817"><a name="zh-cn_topic_0000001429830437_p583535795817"></a><a name="zh-cn_topic_0000001429830437_p583535795817"></a>描述</p>
</th>
</tr>
</thead>
<tbody><tr id="zh-cn_topic_0000001429830437_row1835857145817"><td class="cellrowborder" valign="top" width="18.8%" headers="mcps1.2.3.1.1 "><p id="zh-cn_topic_0000001429830437_p5835457165816"><a name="zh-cn_topic_0000001429830437_p5835457165816"></a><a name="zh-cn_topic_0000001429830437_p5835457165816"></a>modeId</p>
</td>
<td class="cellrowborder" valign="top" width="81.2%" headers="mcps1.2.3.1.2 "><p id="zh-cn_topic_0000001429830437_p168351657155818"><a name="zh-cn_topic_0000001429830437_p168351657155818"></a><a name="zh-cn_topic_0000001429830437_p168351657155818"></a>核间同步的模式，取值如下：</p>
<a name="ul335269516"></a><a name="ul335269516"></a><ul id="ul335269516"><li>模式0：AI Core核间的同步控制。</li><li>模式1：AI Core内部，Vector核（AIV）之间的同步控制。</li><li>模式2：AI Core内部，Cube核（AIC）与Vector核（AIV）之间的同步控制。</li><li>模式4：AI Core内部，AIC与AIV之间的同步控制。AIV0与AIV1可单独触发AIC等待。</li></ul>
</td>
</tr>
<tr id="row168561422132317"><td class="cellrowborder" valign="top" width="18.8%" headers="mcps1.2.3.1.1 "><p id="p1162623072316"><a name="p1162623072316"></a><a name="p1162623072316"></a>pipe</p>
</td>
<td class="cellrowborder" valign="top" width="81.2%" headers="mcps1.2.3.1.2 "><p id="p13626330102311"><a name="p13626330102311"></a><a name="p13626330102311"></a>设置这条指令所在的流水类型，流水类型可参考<a href="同步控制简介.md#section1272612276459">硬件流水类型</a>。</p>
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
<tbody><tr id="row2137145181815"><td class="cellrowborder" valign="top" width="18.54%" headers="mcps1.2.4.1.1 "><p id="p179035252218"><a name="p179035252218"></a><a name="p179035252218"></a>flagId</p>
</td>
<td class="cellrowborder" valign="top" width="10.05%" headers="mcps1.2.4.1.2 "><p id="p7789185214226"><a name="p7789185214226"></a><a name="p7789185214226"></a>输入</p>
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

-   因为[Matmul高阶API](矩阵计算-105.md)内部实现中使用了本接口进行核间同步控制，所以不建议开发者同时使用该接口和Matmul高阶API，否则会有flagID冲突的风险。
-   同一flagId的计数器最多设置15次。
-   使用该接口模式0时，建议开启batchmode模式，使算子独占全部所需核资源，否则可能因满足以下条件导致死锁：

    -   多流并发场景（≥2条执行流）。
    -   ≥2个算子并发执行。
    -   所有并发算子的核数总和超过物理核数。
    -   ≥2个并发算子使用了核间同步功能。

    具体而言，在多流场景下，某条流的核间同步算子虽分配到n个物理核，但可能仅有n-m个核先被调度执行，而其余m个核因被其他流的核间同步算子抢占而尚未启动。先启动的n-m个核执行到核间同步时等待剩余m核完成，而剩余m核因被其他流的核间同步算子占用而无法释放，形成死锁。

    Kernel直调场景下通过[\_\_schedmode\_\_\(mode\)](SIMD-BuiltIn关键字和API.md#li1365012910475)限定符来设置batchmode模式；工程化算子开发场景下，通过TilingContext的SetScheduleMode接口来设置batchmode模式，具体请参考《基础数据结构和接口》[《基础数据结构和接口》](zh-cn_topic_0000002554332575.md)。

## 调用示例<a name="section837496171220"></a>

```
// 使用模式0的方式同步所有的AIV核
if (g_coreType == AscendC::AIV) {
    AscendC::CrossCoreSetFlag<0x0, PIPE_MTE3>(0x8);
    AscendC::CrossCoreWaitFlag(0x8);
}

// 使用模式1的方式同步当前AICore内的所有AIV子核
if (g_coreType == AscendC::AIV) {
    AscendC::CrossCoreSetFlag<0x1, PIPE_MTE3>(0x8);
    AscendC::CrossCoreWaitFlag(0x8);
}

// 注意：如果调用高阶API,无需开发者处理AIC和AIV的同步
// 以Matmul为例：AIC侧做完Matmul计算后通知AIV进行后处理
if (g_coreType == AscendC::AIC) {
    // Matmul处理
    AscendC::CrossCoreSetFlag<0x2, PIPE_FIX>(0x8);
}

// AIV侧等待AIC Set消息, 进行Vector后处理
if (g_coreType == AscendC::AIV) {
    AscendC::CrossCoreWaitFlag(0x8);
    // Vector后处理
}
```

