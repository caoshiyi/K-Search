# SetPadding<a name="ZH-CN_TOPIC_0000002523303542"></a>

## 功能说明<a name="section618mcpsimp"></a>

设置Pad信息。

## 函数原型<a name="section620mcpsimp"></a>

```
void SetPadding(int64_t padFront, int64_t padBack, int64_t padUp, int64_t padDown, int64_t padLeft, int64_t padRight)
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
<tbody><tr id="row106481443135617"><td class="cellrowborder" valign="top" width="14.99%" headers="mcps1.2.4.1.1 "><p id="p0263115751818"><a name="p0263115751818"></a><a name="p0263115751818"></a>padFront</p>
</td>
<td class="cellrowborder" valign="top" width="12.02%" headers="mcps1.2.4.1.2 "><p id="p72631357201810"><a name="p72631357201810"></a><a name="p72631357201810"></a>输入</p>
</td>
<td class="cellrowborder" valign="top" width="72.99%" headers="mcps1.2.4.1.3 "><p id="p1264752220247"><a name="p1264752220247"></a><a name="p1264752220247"></a>卷积正向过程中Input Depth维度的前方向Padding大小。</p>
</td>
</tr>
<tr id="row15231163019481"><td class="cellrowborder" valign="top" width="14.99%" headers="mcps1.2.4.1.1 "><p id="p2023117301486"><a name="p2023117301486"></a><a name="p2023117301486"></a>padBack</p>
</td>
<td class="cellrowborder" valign="top" width="12.02%" headers="mcps1.2.4.1.2 "><p id="p623119305481"><a name="p623119305481"></a><a name="p623119305481"></a>输入</p>
</td>
<td class="cellrowborder" valign="top" width="72.99%" headers="mcps1.2.4.1.3 "><p id="p92647319193"><a name="p92647319193"></a><a name="p92647319193"></a>卷积正向过程中Input Depth维度的后方向Padding大小。</p>
</td>
</tr>
<tr id="row14703232144813"><td class="cellrowborder" valign="top" width="14.99%" headers="mcps1.2.4.1.1 "><p id="p167033325487"><a name="p167033325487"></a><a name="p167033325487"></a>padUp</p>
</td>
<td class="cellrowborder" valign="top" width="12.02%" headers="mcps1.2.4.1.2 "><p id="p107031632194819"><a name="p107031632194819"></a><a name="p107031632194819"></a>输入</p>
</td>
<td class="cellrowborder" valign="top" width="72.99%" headers="mcps1.2.4.1.3 "><p id="p1823015561912"><a name="p1823015561912"></a><a name="p1823015561912"></a>卷积正向过程中Input Height维度的上方向Padding大小。</p>
</td>
</tr>
<tr id="row99271834184818"><td class="cellrowborder" valign="top" width="14.99%" headers="mcps1.2.4.1.1 "><p id="p159271034184817"><a name="p159271034184817"></a><a name="p159271034184817"></a>padDown</p>
</td>
<td class="cellrowborder" valign="top" width="12.02%" headers="mcps1.2.4.1.2 "><p id="p1292763419487"><a name="p1292763419487"></a><a name="p1292763419487"></a>输入</p>
</td>
<td class="cellrowborder" valign="top" width="72.99%" headers="mcps1.2.4.1.3 "><p id="p121491614197"><a name="p121491614197"></a><a name="p121491614197"></a>卷积正向过程中Input Height维度的下方向Padding大小。</p>
</td>
</tr>
<tr id="row713119388486"><td class="cellrowborder" valign="top" width="14.99%" headers="mcps1.2.4.1.1 "><p id="p513110389483"><a name="p513110389483"></a><a name="p513110389483"></a>padLeft</p>
</td>
<td class="cellrowborder" valign="top" width="12.02%" headers="mcps1.2.4.1.2 "><p id="p14131338174817"><a name="p14131338174817"></a><a name="p14131338174817"></a>输入</p>
</td>
<td class="cellrowborder" valign="top" width="72.99%" headers="mcps1.2.4.1.3 "><p id="p065762371714"><a name="p065762371714"></a><a name="p065762371714"></a>卷积正向过程中Input Width维度的左方向Padding大小。</p>
</td>
</tr>
<tr id="row1179574216482"><td class="cellrowborder" valign="top" width="14.99%" headers="mcps1.2.4.1.1 "><p id="p779674217489"><a name="p779674217489"></a><a name="p779674217489"></a>padRight</p>
</td>
<td class="cellrowborder" valign="top" width="12.02%" headers="mcps1.2.4.1.2 "><p id="p379694214813"><a name="p379694214813"></a><a name="p379694214813"></a>输入</p>
</td>
<td class="cellrowborder" valign="top" width="72.99%" headers="mcps1.2.4.1.3 "><p id="p1657223191717"><a name="p1657223191717"></a><a name="p1657223191717"></a>卷积正向过程中Input Width维度的右方向Padding大小。</p>
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
conv3dBpDwTiling.SetPadding(padFront, padBack, padUp, padDown, padLeft, padRight);
```

