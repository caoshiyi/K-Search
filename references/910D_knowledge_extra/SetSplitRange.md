# SetSplitRange<a name="ZH-CN_TOPIC_0000002523304292"></a>

## 功能说明<a name="section618mcpsimp"></a>

设置baseM/baseN/baseK的最大值和最小值。 目前Tiling暂时不支持该功能。

## 函数原型<a name="section620mcpsimp"></a>

```
int32_t SetSplitRange(int32_t maxBaseM = -1, int32_t maxBaseN = -1, int32_t maxBaseK = -1, int32_t minBaseM = -1, int32_t minBaseN = -1, int32_t minBaseK = -1)
```

## 参数说明<a name="section622mcpsimp"></a>

**表 1**  参数说明

<a name="table9646134355611"></a>
<table><thead align="left"><tr id="row964714433565"><th class="cellrowborder" valign="top" width="14.99%" id="mcps1.2.4.1.1"><p id="zh-cn_topic_0235751031_p20917673"><a name="zh-cn_topic_0235751031_p20917673"></a><a name="zh-cn_topic_0235751031_p20917673"></a>参数名</p>
</th>
<th class="cellrowborder" valign="top" width="12.02%" id="mcps1.2.4.1.2"><p id="zh-cn_topic_0235751031_p16609919"><a name="zh-cn_topic_0235751031_p16609919"></a><a name="zh-cn_topic_0235751031_p16609919"></a>输入/输出</p>
</th>
<th class="cellrowborder" valign="top" width="72.99%" id="mcps1.2.4.1.3"><p id="zh-cn_topic_0235751031_p59995477"><a name="zh-cn_topic_0235751031_p59995477"></a><a name="zh-cn_topic_0235751031_p59995477"></a>描述</p>
</th>
</tr>
</thead>
<tbody><tr id="row106481443135617"><td class="cellrowborder" valign="top" width="14.99%" headers="mcps1.2.4.1.1 "><p id="p7636151954713"><a name="p7636151954713"></a><a name="p7636151954713"></a>maxBaseM</p>
</td>
<td class="cellrowborder" valign="top" width="12.02%" headers="mcps1.2.4.1.2 "><p id="p1263601915472"><a name="p1263601915472"></a><a name="p1263601915472"></a>输入</p>
</td>
<td class="cellrowborder" valign="top" width="72.99%" headers="mcps1.2.4.1.3 "><p id="p56361919194711"><a name="p56361919194711"></a><a name="p56361919194711"></a>设置最大的baseM值，默认值为-1。-1表示不设置指定的baseM最大值，该值由Tiling函数自行计算。</p>
</td>
</tr>
<tr id="row1862281410479"><td class="cellrowborder" valign="top" width="14.99%" headers="mcps1.2.4.1.1 "><p id="p86361019104710"><a name="p86361019104710"></a><a name="p86361019104710"></a>maxBaseN</p>
</td>
<td class="cellrowborder" valign="top" width="12.02%" headers="mcps1.2.4.1.2 "><p id="p5636151916475"><a name="p5636151916475"></a><a name="p5636151916475"></a>输入</p>
</td>
<td class="cellrowborder" valign="top" width="72.99%" headers="mcps1.2.4.1.3 "><p id="p1963614198474"><a name="p1963614198474"></a><a name="p1963614198474"></a>设置最大的baseN值，默认值为-1。-1表示不设置指定的baseN最大值，该值由Tiling函数自行计算。</p>
</td>
</tr>
<tr id="row43114155478"><td class="cellrowborder" valign="top" width="14.99%" headers="mcps1.2.4.1.1 "><p id="p96362198474"><a name="p96362198474"></a><a name="p96362198474"></a>maxBaseK</p>
</td>
<td class="cellrowborder" valign="top" width="12.02%" headers="mcps1.2.4.1.2 "><p id="p263691994712"><a name="p263691994712"></a><a name="p263691994712"></a>输入</p>
</td>
<td class="cellrowborder" valign="top" width="72.99%" headers="mcps1.2.4.1.3 "><p id="p1663615195479"><a name="p1663615195479"></a><a name="p1663615195479"></a>设置最大的baseK值，默认值为-1。-1表示不设置指定的baseK最大值，该值由Tiling函数自行计算。</p>
</td>
</tr>
<tr id="row1816810158478"><td class="cellrowborder" valign="top" width="14.99%" headers="mcps1.2.4.1.1 "><p id="p1263612198477"><a name="p1263612198477"></a><a name="p1263612198477"></a>minBaseM</p>
</td>
<td class="cellrowborder" valign="top" width="12.02%" headers="mcps1.2.4.1.2 "><p id="p8636121974718"><a name="p8636121974718"></a><a name="p8636121974718"></a>输入</p>
</td>
<td class="cellrowborder" valign="top" width="72.99%" headers="mcps1.2.4.1.3 "><p id="p56361019134710"><a name="p56361019134710"></a><a name="p56361019134710"></a>设置最小的baseM值，默认值为-1。-1表示不设置指定的baseM最小值，该值由Tiling函数自行计算。</p>
</td>
</tr>
<tr id="row12312191534716"><td class="cellrowborder" valign="top" width="14.99%" headers="mcps1.2.4.1.1 "><p id="p16368199478"><a name="p16368199478"></a><a name="p16368199478"></a>minBaseN</p>
</td>
<td class="cellrowborder" valign="top" width="12.02%" headers="mcps1.2.4.1.2 "><p id="p1636191911474"><a name="p1636191911474"></a><a name="p1636191911474"></a>输入</p>
</td>
<td class="cellrowborder" valign="top" width="72.99%" headers="mcps1.2.4.1.3 "><p id="p18636161916475"><a name="p18636161916475"></a><a name="p18636161916475"></a>设置最小的baseN值，默认值为-1。-1表示不设置指定的baseN最小值，该值由Tiling函数自行计算。</p>
</td>
</tr>
<tr id="row1745221511475"><td class="cellrowborder" valign="top" width="14.99%" headers="mcps1.2.4.1.1 "><p id="p15636121916475"><a name="p15636121916475"></a><a name="p15636121916475"></a>minBaseK</p>
</td>
<td class="cellrowborder" valign="top" width="12.02%" headers="mcps1.2.4.1.2 "><p id="p863671914479"><a name="p863671914479"></a><a name="p863671914479"></a>输入</p>
</td>
<td class="cellrowborder" valign="top" width="72.99%" headers="mcps1.2.4.1.3 "><p id="p1636111913479"><a name="p1636111913479"></a><a name="p1636111913479"></a>设置最小的baseK值，默认值为-1。-1表示不设置指定的baseK最小值，该值由Tiling函数自行计算。</p>
</td>
</tr>
</tbody>
</table>

## 返回值说明<a name="section640mcpsimp"></a>

-1表示设置失败；0表示设置成功。

## 约束说明<a name="section633mcpsimp"></a>

若baseM/baseN/baseK不满足C0\_size对齐，计算Tiling时会将该值对齐到C0\_size。提示，half/bfloat16\_t数据类型的C0\_size为16，float数据类型的C0\_size为8，int8\_t数据类型的C0\_size为32，int4b\_t数据类型的C0\_size为64。

