# SetWeightShape<a name="ZH-CN_TOPIC_0000002554344133"></a>

## 功能说明<a name="section618mcpsimp"></a>

设置权重矩阵Weight的形状。

## 函数原型<a name="section620mcpsimp"></a>

```
bool SetWeightShape(int64_t cout, int64_t cin, int64_t d, int64_t h, int64_t w)
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
<tbody><tr id="row3963125553813"><td class="cellrowborder" valign="top" width="14.99%" headers="mcps1.2.4.1.1 "><p id="p189631555133813"><a name="p189631555133813"></a><a name="p189631555133813"></a>cout</p>
</td>
<td class="cellrowborder" valign="top" width="12.02%" headers="mcps1.2.4.1.2 "><p id="p10963255183813"><a name="p10963255183813"></a><a name="p10963255183813"></a>输入</p>
</td>
<td class="cellrowborder" valign="top" width="72.99%" headers="mcps1.2.4.1.3 "><p id="p1696375510383"><a name="p1696375510383"></a><a name="p1696375510383"></a>设置卷积正向的输出channel大小，与GradOutput的Channel大小一致。</p>
</td>
</tr>
<tr id="row12659125263813"><td class="cellrowborder" valign="top" width="14.99%" headers="mcps1.2.4.1.1 "><p id="p865985223816"><a name="p865985223816"></a><a name="p865985223816"></a>cin</p>
</td>
<td class="cellrowborder" valign="top" width="12.02%" headers="mcps1.2.4.1.2 "><p id="p1165965215385"><a name="p1165965215385"></a><a name="p1165965215385"></a>输入</p>
</td>
<td class="cellrowborder" valign="top" width="72.99%" headers="mcps1.2.4.1.3 "><p id="p106593526385"><a name="p106593526385"></a><a name="p106593526385"></a>设置卷积正向的输入channel大小，与GradInput的Channel大小一致。</p>
</td>
</tr>
<tr id="row3558530141312"><td class="cellrowborder" valign="top" width="14.99%" headers="mcps1.2.4.1.1 "><p id="p755910306133"><a name="p755910306133"></a><a name="p755910306133"></a>d</p>
</td>
<td class="cellrowborder" valign="top" width="12.02%" headers="mcps1.2.4.1.2 "><p id="p2559330101319"><a name="p2559330101319"></a><a name="p2559330101319"></a>输入</p>
</td>
<td class="cellrowborder" valign="top" width="72.99%" headers="mcps1.2.4.1.3 "><p id="p055911308132"><a name="p055911308132"></a><a name="p055911308132"></a>设置weight的Depth值。</p>
</td>
</tr>
<tr id="row1231453261313"><td class="cellrowborder" valign="top" width="14.99%" headers="mcps1.2.4.1.1 "><p id="p13142322139"><a name="p13142322139"></a><a name="p13142322139"></a>h</p>
</td>
<td class="cellrowborder" valign="top" width="12.02%" headers="mcps1.2.4.1.2 "><p id="p331412328130"><a name="p331412328130"></a><a name="p331412328130"></a>输入</p>
</td>
<td class="cellrowborder" valign="top" width="72.99%" headers="mcps1.2.4.1.3 "><p id="p6314193251313"><a name="p6314193251313"></a><a name="p6314193251313"></a>设置weight的Height值。</p>
</td>
</tr>
<tr id="row106481443135617"><td class="cellrowborder" valign="top" width="14.99%" headers="mcps1.2.4.1.1 "><p id="p184534594431"><a name="p184534594431"></a><a name="p184534594431"></a>w</p>
</td>
<td class="cellrowborder" valign="top" width="12.02%" headers="mcps1.2.4.1.2 "><p id="p164539593433"><a name="p164539593433"></a><a name="p164539593433"></a>输入</p>
</td>
<td class="cellrowborder" valign="top" width="72.99%" headers="mcps1.2.4.1.3 "><p id="p144634111375"><a name="p144634111375"></a><a name="p144634111375"></a>设置weight的Width值。</p>
</td>
</tr>
</tbody>
</table>

## 返回值说明<a name="section640mcpsimp"></a>

true表示设置成功，false表示设置失败。

## 约束说明<a name="section633mcpsimp"></a>

无

## 调用示例<a name="section1665082013318"></a>

```
auto ascendcPlatform = platform_ascendc::PlatformAscendCManager::GetInstance();
ConvBackpropApi::Conv3DBpInputTiling con3dBpDxTiling(*ascendcPlatform);
con3dBpDxTiling.SetWeightShape(cout, cin, d, h, w);
```

