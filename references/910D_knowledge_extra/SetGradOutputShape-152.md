# SetGradOutputShape<a name="ZH-CN_TOPIC_0000002554424063"></a>

## 功能说明<a name="section618mcpsimp"></a>

设置GradOutput的形状：Batch、Channel、Depth、Height、Width。

## 函数原型<a name="section620mcpsimp"></a>

```
void SetGradOutputShape(int64_t n, int64_t c, int64_t d, int64_t h, int64_t w)
```

## 参数说明<a name="section622mcpsimp"></a>

**表 1**  参数说明

<a name="table9646134355611"></a>
<table><thead align="left"><tr id="row964714433565"><th class="cellrowborder" valign="top" width="13.68%" id="mcps1.2.4.1.1"><p id="zh-cn_topic_0235751031_p20917673"><a name="zh-cn_topic_0235751031_p20917673"></a><a name="zh-cn_topic_0235751031_p20917673"></a>参数名</p>
</th>
<th class="cellrowborder" valign="top" width="13.33%" id="mcps1.2.4.1.2"><p id="zh-cn_topic_0235751031_p16609919"><a name="zh-cn_topic_0235751031_p16609919"></a><a name="zh-cn_topic_0235751031_p16609919"></a>输入/输出</p>
</th>
<th class="cellrowborder" valign="top" width="72.99%" id="mcps1.2.4.1.3"><p id="zh-cn_topic_0235751031_p59995477"><a name="zh-cn_topic_0235751031_p59995477"></a><a name="zh-cn_topic_0235751031_p59995477"></a>描述</p>
</th>
</tr>
</thead>
<tbody><tr id="row6200102819561"><td class="cellrowborder" valign="top" width="13.68%" headers="mcps1.2.4.1.1 "><p id="p82001289563"><a name="p82001289563"></a><a name="p82001289563"></a>n</p>
</td>
<td class="cellrowborder" valign="top" width="13.33%" headers="mcps1.2.4.1.2 "><p id="p13200828105611"><a name="p13200828105611"></a><a name="p13200828105611"></a>输入</p>
</td>
<td class="cellrowborder" valign="top" width="72.99%" headers="mcps1.2.4.1.3 "><p id="p18165140194616"><a name="p18165140194616"></a><a name="p18165140194616"></a>输入GradOutput的Batch值。</p>
</td>
</tr>
<tr id="row106481443135617"><td class="cellrowborder" valign="top" width="13.68%" headers="mcps1.2.4.1.1 "><p id="p0263115751818"><a name="p0263115751818"></a><a name="p0263115751818"></a>c</p>
</td>
<td class="cellrowborder" valign="top" width="13.33%" headers="mcps1.2.4.1.2 "><p id="p72631357201810"><a name="p72631357201810"></a><a name="p72631357201810"></a>输入</p>
</td>
<td class="cellrowborder" valign="top" width="72.99%" headers="mcps1.2.4.1.3 "><p id="p1726213570185"><a name="p1726213570185"></a><a name="p1726213570185"></a>输入GradOutput的Channel值。</p>
</td>
</tr>
<tr id="row940514418461"><td class="cellrowborder" valign="top" width="13.68%" headers="mcps1.2.4.1.1 "><p id="p114057418469"><a name="p114057418469"></a><a name="p114057418469"></a>d</p>
</td>
<td class="cellrowborder" valign="top" width="13.33%" headers="mcps1.2.4.1.2 "><p id="p11405441184612"><a name="p11405441184612"></a><a name="p11405441184612"></a>输入</p>
</td>
<td class="cellrowborder" valign="top" width="72.99%" headers="mcps1.2.4.1.3 "><p id="p15405134164616"><a name="p15405134164616"></a><a name="p15405134164616"></a>输入GradOutput的Depth值。</p>
</td>
</tr>
<tr id="row97171743104616"><td class="cellrowborder" valign="top" width="13.68%" headers="mcps1.2.4.1.1 "><p id="p171816430463"><a name="p171816430463"></a><a name="p171816430463"></a>h</p>
</td>
<td class="cellrowborder" valign="top" width="13.33%" headers="mcps1.2.4.1.2 "><p id="p571894317464"><a name="p571894317464"></a><a name="p571894317464"></a>输入</p>
</td>
<td class="cellrowborder" valign="top" width="72.99%" headers="mcps1.2.4.1.3 "><p id="p371814314612"><a name="p371814314612"></a><a name="p371814314612"></a>输入GradOutput的Height值。</p>
</td>
</tr>
<tr id="row97241157144614"><td class="cellrowborder" valign="top" width="13.68%" headers="mcps1.2.4.1.1 "><p id="p15725105724616"><a name="p15725105724616"></a><a name="p15725105724616"></a>w</p>
</td>
<td class="cellrowborder" valign="top" width="13.33%" headers="mcps1.2.4.1.2 "><p id="p1072519579466"><a name="p1072519579466"></a><a name="p1072519579466"></a>输入</p>
</td>
<td class="cellrowborder" valign="top" width="72.99%" headers="mcps1.2.4.1.3 "><p id="p672575718462"><a name="p672575718462"></a><a name="p672575718462"></a>输入GradOutput的Width值。</p>
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
auto ascendcPlatform = platform_ascendc::PlatformAscendCManager::GetInstance();
ConvBackpropApi::Conv3dBpFilterTiling conv3dBpDwTiling(*ascendcPlatform);
conv3dBpDwTiling.SetGradOutputShape(n, c, d, h, w);
```

