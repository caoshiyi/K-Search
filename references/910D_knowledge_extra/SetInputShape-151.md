# SetInputShape<a name="ZH-CN_TOPIC_0000002523344764"></a>

## 功能说明<a name="section618mcpsimp"></a>

设置特征矩阵Input的形状：Batch、Channel、Depth、Height、Width。

## 函数原型<a name="section620mcpsimp"></a>

```
void SetInputShape(int64_t n, int64_t c, int64_t d, int64_t h, int64_t w)
```

## 参数说明<a name="section622mcpsimp"></a>

**表 1**  参数说明

<a name="table9646134355611"></a>
<table><thead align="left"><tr id="row964714433565"><th class="cellrowborder" valign="top" width="14.99%" id="mcps1.2.4.1.1"><p id="zh-cn_topic_0235751031_p20917673"><a name="zh-cn_topic_0235751031_p20917673"></a><a name="zh-cn_topic_0235751031_p20917673"></a>参数名</p>
</th>
<th class="cellrowborder" valign="top" width="11.61%" id="mcps1.2.4.1.2"><p id="zh-cn_topic_0235751031_p16609919"><a name="zh-cn_topic_0235751031_p16609919"></a><a name="zh-cn_topic_0235751031_p16609919"></a>输入/输出</p>
</th>
<th class="cellrowborder" valign="top" width="73.4%" id="mcps1.2.4.1.3"><p id="zh-cn_topic_0235751031_p59995477"><a name="zh-cn_topic_0235751031_p59995477"></a><a name="zh-cn_topic_0235751031_p59995477"></a>描述</p>
</th>
</tr>
</thead>
<tbody><tr id="row20537172161716"><td class="cellrowborder" valign="top" width="14.99%" headers="mcps1.2.4.1.1 "><p id="p65378212178"><a name="p65378212178"></a><a name="p65378212178"></a>n</p>
</td>
<td class="cellrowborder" valign="top" width="11.61%" headers="mcps1.2.4.1.2 "><p id="p7537028176"><a name="p7537028176"></a><a name="p7537028176"></a>输入</p>
</td>
<td class="cellrowborder" valign="top" width="73.4%" headers="mcps1.2.4.1.3 "><p id="p165375212178"><a name="p165375212178"></a><a name="p165375212178"></a>输入Input的Batch值。</p>
</td>
</tr>
<tr id="row316416019467"><td class="cellrowborder" valign="top" width="14.99%" headers="mcps1.2.4.1.1 "><p id="p1916410014468"><a name="p1916410014468"></a><a name="p1916410014468"></a>c</p>
</td>
<td class="cellrowborder" valign="top" width="11.61%" headers="mcps1.2.4.1.2 "><p id="p191641007462"><a name="p191641007462"></a><a name="p191641007462"></a>输入</p>
</td>
<td class="cellrowborder" valign="top" width="73.4%" headers="mcps1.2.4.1.3 "><p id="p18165140194616"><a name="p18165140194616"></a><a name="p18165140194616"></a>输入Input的Channel值。</p>
</td>
</tr>
<tr id="row10163172144619"><td class="cellrowborder" valign="top" width="14.99%" headers="mcps1.2.4.1.1 "><p id="p108104576317"><a name="p108104576317"></a><a name="p108104576317"></a>d</p>
</td>
<td class="cellrowborder" valign="top" width="11.61%" headers="mcps1.2.4.1.2 "><p id="p916315214617"><a name="p916315214617"></a><a name="p916315214617"></a>输入</p>
</td>
<td class="cellrowborder" valign="top" width="73.4%" headers="mcps1.2.4.1.3 "><p id="p20647104213916"><a name="p20647104213916"></a><a name="p20647104213916"></a>输入Input的Depth值。</p>
</td>
</tr>
<tr id="row1798064144612"><td class="cellrowborder" valign="top" width="14.99%" headers="mcps1.2.4.1.1 "><p id="p3676901944"><a name="p3676901944"></a><a name="p3676901944"></a>h</p>
</td>
<td class="cellrowborder" valign="top" width="11.61%" headers="mcps1.2.4.1.2 "><p id="p1998014164618"><a name="p1998014164618"></a><a name="p1998014164618"></a>输入</p>
</td>
<td class="cellrowborder" valign="top" width="73.4%" headers="mcps1.2.4.1.3 "><p id="p2734144953914"><a name="p2734144953914"></a><a name="p2734144953914"></a>输入Input的Height值。</p>
</td>
</tr>
<tr id="row688137194619"><td class="cellrowborder" valign="top" width="14.99%" headers="mcps1.2.4.1.1 "><p id="p7904633414"><a name="p7904633414"></a><a name="p7904633414"></a>w</p>
</td>
<td class="cellrowborder" valign="top" width="11.61%" headers="mcps1.2.4.1.2 "><p id="p4881578463"><a name="p4881578463"></a><a name="p4881578463"></a>输入</p>
</td>
<td class="cellrowborder" valign="top" width="73.4%" headers="mcps1.2.4.1.3 "><p id="p612175616391"><a name="p612175616391"></a><a name="p612175616391"></a>输入Input的Width值。</p>
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
conv3dBpDwTiling.SetInputShape(n, c, d, h, w);
```

