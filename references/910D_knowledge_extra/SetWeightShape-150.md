# SetWeightShape<a name="ZH-CN_TOPIC_0000002554424247"></a>

## 功能说明<a name="section618mcpsimp"></a>

设置权重矩阵Weight的形状。

## 函数原型<a name="section620mcpsimp"></a>

```
void SetWeightShape(int64_t cout, int64_t cin, int64_t d, int64_t h, int64_t w)
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
<tbody><tr id="row19977449182915"><td class="cellrowborder" valign="top" width="14.99%" headers="mcps1.2.4.1.1 "><p id="p597874916291"><a name="p597874916291"></a><a name="p597874916291"></a>cout</p>
</td>
<td class="cellrowborder" valign="top" width="12.02%" headers="mcps1.2.4.1.2 "><p id="p397884919296"><a name="p397884919296"></a><a name="p397884919296"></a>输入</p>
</td>
<td class="cellrowborder" valign="top" width="72.99%" headers="mcps1.2.4.1.3 "><p id="p20978114912910"><a name="p20978114912910"></a><a name="p20978114912910"></a>设置GradOutput的Channel值。</p>
</td>
</tr>
<tr id="row19472147192919"><td class="cellrowborder" valign="top" width="14.99%" headers="mcps1.2.4.1.1 "><p id="p7473154742915"><a name="p7473154742915"></a><a name="p7473154742915"></a>cin</p>
</td>
<td class="cellrowborder" valign="top" width="12.02%" headers="mcps1.2.4.1.2 "><p id="p481143015"><a name="p481143015"></a><a name="p481143015"></a>输入</p>
</td>
<td class="cellrowborder" valign="top" width="72.99%" headers="mcps1.2.4.1.3 "><p id="p18165140194616"><a name="p18165140194616"></a><a name="p18165140194616"></a>设置Input的Channel值。</p>
</td>
</tr>
<tr id="row106481443135617"><td class="cellrowborder" valign="top" width="14.99%" headers="mcps1.2.4.1.1 "><p id="p0263115751818"><a name="p0263115751818"></a><a name="p0263115751818"></a>d</p>
</td>
<td class="cellrowborder" valign="top" width="12.02%" headers="mcps1.2.4.1.2 "><p id="p72631357201810"><a name="p72631357201810"></a><a name="p72631357201810"></a>输入</p>
</td>
<td class="cellrowborder" valign="top" width="72.99%" headers="mcps1.2.4.1.3 "><p id="p1726213570185"><a name="p1726213570185"></a><a name="p1726213570185"></a>设置Weight的Depth值。</p>
</td>
</tr>
<tr id="row3462057174316"><td class="cellrowborder" valign="top" width="14.99%" headers="mcps1.2.4.1.1 "><p id="p154627579433"><a name="p154627579433"></a><a name="p154627579433"></a>h</p>
</td>
<td class="cellrowborder" valign="top" width="12.02%" headers="mcps1.2.4.1.2 "><p id="p164621357194319"><a name="p164621357194319"></a><a name="p164621357194319"></a>输入</p>
</td>
<td class="cellrowborder" valign="top" width="72.99%" headers="mcps1.2.4.1.3 "><p id="p18816114011377"><a name="p18816114011377"></a><a name="p18816114011377"></a>设置Weight的Height值。</p>
</td>
</tr>
<tr id="row1245318593437"><td class="cellrowborder" valign="top" width="14.99%" headers="mcps1.2.4.1.1 "><p id="p184534594431"><a name="p184534594431"></a><a name="p184534594431"></a>w</p>
</td>
<td class="cellrowborder" valign="top" width="12.02%" headers="mcps1.2.4.1.2 "><p id="p164539593433"><a name="p164539593433"></a><a name="p164539593433"></a>输入</p>
</td>
<td class="cellrowborder" valign="top" width="72.99%" headers="mcps1.2.4.1.3 "><p id="p144634111375"><a name="p144634111375"></a><a name="p144634111375"></a>设置Weight的Width值。</p>
</td>
</tr>
</tbody>
</table>

## 返回值说明<a name="section640mcpsimp"></a>

无

## 约束说明<a name="section633mcpsimp"></a>

无

## 调用示例<a name="section1665082013318"></a>

```
optiling::Conv3DBackpropFilterTilingData tilingData;
auto ascendcPlatform = platform_ascendc::PlatformAscendCManager::GetInstance();
ConvBackpropApi::Conv3dBpFilterTiling conv3dBpDwTiling(*ascendcPlatform);
conv3dBpDwTiling.SetWeightShape(cout, cin, d, h, w);
```

