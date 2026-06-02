# SetDilation<a name="ZH-CN_TOPIC_0000002554344183"></a>

## 功能说明<a name="section618mcpsimp"></a>

设置Dilation信息，即卷积核Depth/Height/Width方向的扩张大小。

## 函数原型<a name="section620mcpsimp"></a>

```
void SetDilation(int64_t dilationD, int64_t dilationH, int64_t dilationW)
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
<tbody><tr id="row106481443135617"><td class="cellrowborder" valign="top" width="14.99%" headers="mcps1.2.4.1.1 "><p id="p0263115751818"><a name="p0263115751818"></a><a name="p0263115751818"></a>dilationD</p>
</td>
<td class="cellrowborder" valign="top" width="12.02%" headers="mcps1.2.4.1.2 "><p id="p72631357201810"><a name="p72631357201810"></a><a name="p72631357201810"></a>输入</p>
</td>
<td class="cellrowborder" valign="top" width="72.99%" headers="mcps1.2.4.1.3 "><p id="p4921161014141"><a name="p4921161014141"></a><a name="p4921161014141"></a>卷积核Weight的Depth方向扩张大小。</p>
</td>
</tr>
<tr id="row14969828175819"><td class="cellrowborder" valign="top" width="14.99%" headers="mcps1.2.4.1.1 "><p id="p15913445125818"><a name="p15913445125818"></a><a name="p15913445125818"></a>dilationH</p>
</td>
<td class="cellrowborder" valign="top" width="12.02%" headers="mcps1.2.4.1.2 "><p id="p891314451585"><a name="p891314451585"></a><a name="p891314451585"></a>输入</p>
</td>
<td class="cellrowborder" valign="top" width="72.99%" headers="mcps1.2.4.1.3 "><p id="p13913145185812"><a name="p13913145185812"></a><a name="p13913145185812"></a>卷积核Weight的Height方向扩张大小。</p>
</td>
</tr>
<tr id="row11721162535820"><td class="cellrowborder" valign="top" width="14.99%" headers="mcps1.2.4.1.1 "><p id="p14893194616585"><a name="p14893194616585"></a><a name="p14893194616585"></a>dilationW</p>
</td>
<td class="cellrowborder" valign="top" width="12.02%" headers="mcps1.2.4.1.2 "><p id="p8893146155811"><a name="p8893146155811"></a><a name="p8893146155811"></a>输入</p>
</td>
<td class="cellrowborder" valign="top" width="72.99%" headers="mcps1.2.4.1.3 "><p id="p1089310468583"><a name="p1089310468583"></a><a name="p1089310468583"></a>卷积核Weight的Width方向扩张大小。</p>
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
conv3dBpDwTiling.SetDilation(dilationD, dilationH, dilationW);
```

