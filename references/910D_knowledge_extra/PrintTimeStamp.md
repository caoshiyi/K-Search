# PrintTimeStamp<a name="ZH-CN_TOPIC_0000002554423709"></a>

## 产品支持情况<a name="section1550532418810"></a>

<a name="table38301303189"></a>
<table><thead align="left"><tr id="row20831180131817"><th class="cellrowborder" valign="top" width="57.99999999999999%" id="mcps1.1.3.1.1"><p id="p1883113061818"><a name="p1883113061818"></a><a name="p1883113061818"></a><span id="ph20833205312295"><a name="ph20833205312295"></a><a name="ph20833205312295"></a>产品</span></p>
</th>
<th class="cellrowborder" align="center" valign="top" width="42%" id="mcps1.1.3.1.2"><p id="p783113012187"><a name="p783113012187"></a><a name="p783113012187"></a>是否支持</p>
</th>
</tr>
</thead>
<tbody><tr id="row1272474920205"><td class="cellrowborder" valign="top" width="57.99999999999999%" headers="mcps1.1.3.1.1 "><p id="p12300735171314"><a name="p12300735171314"></a><a name="p12300735171314"></a><span id="ph730011352138"><a name="ph730011352138"></a><a name="ph730011352138"></a>Ascend 950PR/Ascend 950DT</span></p>
</td>
<td class="cellrowborder" align="center" valign="top" width="42%" headers="mcps1.1.3.1.2 "><p id="p37256491200"><a name="p37256491200"></a><a name="p37256491200"></a>√</p>
</td>
</tr>
</tbody>
</table>

## 功能说明<a name="section259105813316"></a>

提供时间戳打点功能，用于在算子Kernel代码中标记关键执行点。调用后会打印如下信息：

-   descId:用户自定义标识符，用于区分不同打点位置；
-   rsv ：保留值，默认为0，无需关注；
-   timeStamp :当前系统cycle数，用于计算时间差，时间换算规则可参考[GetSystemCycle\(ISASI\)](GetSystemCycle(ISASI).md)；
-   pcPtr：pc指针数值，若无特殊需求，用户无需关注。

打印示例如下：

```
descId is 65577, rsv is 0, timeStamp is 13806084506158, pcPtr is 20619064414544.
```

> **注意：** 
>该功能主要用于**调试和性能分析**，开启后会对算子性能产生一定影响，**生产环境建议关闭**。
>默认情况下，该功能关闭，开发者可以按需通过增加-DASCENDC\_TIME\_STAMP\_ON编译选项的方式，开启打点功能。

## 函数原型<a name="section2067518173415"></a>

```
__aicore__ inline void PrintTimeStamp(uint32_t descId)
```

## 参数说明<a name="section158061867342"></a>

<a name="zh-cn_topic_0235751031_table33761356"></a>
<table><thead align="left"><tr id="zh-cn_topic_0235751031_row27598891"><th class="cellrowborder" valign="top" width="17.89%" id="mcps1.1.4.1.1"><p id="zh-cn_topic_0235751031_p20917673"><a name="zh-cn_topic_0235751031_p20917673"></a><a name="zh-cn_topic_0235751031_p20917673"></a>参数名</p>
</th>
<th class="cellrowborder" valign="top" width="12.94%" id="mcps1.1.4.1.2"><p id="zh-cn_topic_0235751031_p16609919"><a name="zh-cn_topic_0235751031_p16609919"></a><a name="zh-cn_topic_0235751031_p16609919"></a>输入/输出</p>
</th>
<th class="cellrowborder" valign="top" width="69.17%" id="mcps1.1.4.1.3"><p id="zh-cn_topic_0235751031_p59995477"><a name="zh-cn_topic_0235751031_p59995477"></a><a name="zh-cn_topic_0235751031_p59995477"></a>描述</p>
</th>
</tr>
</thead>
<tbody><tr id="row241512381322"><td class="cellrowborder" valign="top" width="17.89%" headers="mcps1.1.4.1.1 "><p id="p4415838153219"><a name="p4415838153219"></a><a name="p4415838153219"></a>descId</p>
</td>
<td class="cellrowborder" valign="top" width="12.94%" headers="mcps1.1.4.1.2 "><p id="p16415538133215"><a name="p16415538133215"></a><a name="p16415538133215"></a>输入</p>
</td>
<td class="cellrowborder" valign="top" width="69.17%" headers="mcps1.1.4.1.3 "><p id="p10415193810325"><a name="p10415193810325"></a><a name="p10415193810325"></a><span>用户自定义标识符</span>（自定义数字），<span>用于区分不同打点位置</span>。</p>
<div class="caution" id="note726216118599"><a name="note726216118599"></a><a name="note726216118599"></a><span class="cautiontitle"> 注意： </span><div class="cautionbody"><p id="p0341162414018"><a name="p0341162414018"></a><a name="p0341162414018"></a>[0, 0xffff]是预留给Ascend C内部各个模块使用的id值，用户自定义的descId建议使用大于0xffff的数值。</p>
</div></div>
</td>
</tr>
</tbody>
</table>

## 返回值说明<a name="section640mcpsimp"></a>

无

## 约束说明<a name="section794123819592"></a>

-   该功能仅用于NPU上板调试。
-   暂不支持算子入图场景的打印。

-   单次调用本接口打印的数据总量不可超过1MB（还包括少量框架需要的头尾信息，通常可忽略）。使用时应注意，如果超出这个限制，则数据不会被打印。在使用自定义算子工程进行工程化算子开发时，一个算子所有使用Dump功能的接口在每个核上Dump的数据总量不可超过1MB。请开发者自行控制待打印的内容数据量，超出则不会打印。

## 调用示例<a name="section82241477610"></a>

```
AscendC::PrintTimeStamp(65577);
```

打印结果如下（Dump信息头等仅在使用自定义算子工程时才会打印）：

```
opType=AddCustom, DumpHead: AIV-0, CoreType=AIV, block dim=8, total_block_num=8, block_remain_len=1047136, block_initial_space=1048576, rsv=0, magic=5aa5bccd
...// 一些框架内部的打点信息
descId is 65577, rsv is 0, timeStamp is 13806084506158, pcPtr is 20619064414544.
```

