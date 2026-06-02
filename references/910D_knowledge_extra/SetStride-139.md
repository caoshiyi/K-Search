# SetStride<a name="ZH-CN_TOPIC_0000002523304620"></a>

## 功能说明<a name="section618mcpsimp"></a>

设置Stride信息。

## 函数原型<a name="section620mcpsimp"></a>

```
void SetStride(int64_t strideD, int64_t strideH, int64_t strideW)
```

## 参数说明<a name="section622mcpsimp"></a>

**表 1**  参数说明

<a name="table9646134355611"></a>
<table><thead align="left"><tr id="row964714433565"><th class="cellrowborder" valign="top" width="14.99%" id="mcps1.2.4.1.1"><p id="zh-cn_topic_0235751031_p20917673"><a name="zh-cn_topic_0235751031_p20917673"></a><a name="zh-cn_topic_0235751031_p20917673"></a>参数名</p>
</th>
<th class="cellrowborder" valign="top" width="12.01%" id="mcps1.2.4.1.2"><p id="zh-cn_topic_0235751031_p16609919"><a name="zh-cn_topic_0235751031_p16609919"></a><a name="zh-cn_topic_0235751031_p16609919"></a>输入/输出</p>
</th>
<th class="cellrowborder" valign="top" width="73%" id="mcps1.2.4.1.3"><p id="zh-cn_topic_0235751031_p59995477"><a name="zh-cn_topic_0235751031_p59995477"></a><a name="zh-cn_topic_0235751031_p59995477"></a>描述</p>
</th>
</tr>
</thead>
<tbody><tr id="row1278945712390"><td class="cellrowborder" valign="top" width="14.99%" headers="mcps1.2.4.1.1 "><p id="p13427205973914"><a name="p13427205973914"></a><a name="p13427205973914"></a>strideD</p>
</td>
<td class="cellrowborder" valign="top" width="12.01%" headers="mcps1.2.4.1.2 "><p id="p134275598397"><a name="p134275598397"></a><a name="p134275598397"></a>输入</p>
</td>
<td class="cellrowborder" valign="top" width="73%" headers="mcps1.2.4.1.3 "><p id="p9427155917394"><a name="p9427155917394"></a><a name="p9427155917394"></a>卷积正向过程中Depth方向Stride的大小。</p>
</td>
</tr>
<tr id="row20251711122515"><td class="cellrowborder" valign="top" width="14.99%" headers="mcps1.2.4.1.1 "><p id="p1925119118257"><a name="p1925119118257"></a><a name="p1925119118257"></a>strideH</p>
</td>
<td class="cellrowborder" valign="top" width="12.01%" headers="mcps1.2.4.1.2 "><p id="p1625111114251"><a name="p1625111114251"></a><a name="p1625111114251"></a>输入</p>
</td>
<td class="cellrowborder" valign="top" width="73%" headers="mcps1.2.4.1.3 "><p id="p7252711102511"><a name="p7252711102511"></a><a name="p7252711102511"></a>卷积正向过程中Height方向Stride的大小。</p>
</td>
</tr>
<tr id="row31397919255"><td class="cellrowborder" valign="top" width="14.99%" headers="mcps1.2.4.1.1 "><p id="p111401912513"><a name="p111401912513"></a><a name="p111401912513"></a>strideW</p>
</td>
<td class="cellrowborder" valign="top" width="12.01%" headers="mcps1.2.4.1.2 "><p id="p151401919256"><a name="p151401919256"></a><a name="p151401919256"></a>输入</p>
</td>
<td class="cellrowborder" valign="top" width="73%" headers="mcps1.2.4.1.3 "><p id="p181403910254"><a name="p181403910254"></a><a name="p181403910254"></a>卷积正向过程中Width方向Stride的大小。</p>
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
ConvBackpropApi::Conv3DBpInputTiling conv3DBpDxTiling(*ascendcPlatform);
conv3DBpDxTiling.SetStride(strideD, strideH, strideW);
```

