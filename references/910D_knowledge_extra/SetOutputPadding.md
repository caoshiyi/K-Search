# SetOutputPadding<a name="ZH-CN_TOPIC_0000002554424085"></a>

## 功能说明<a name="section618mcpsimp"></a>

构建Conv3DTranspose算子时，设置输出的Padding大小，用于推导输出的形状。在构建Conv3DBackpropInput算子时，此接口无实际意义，请勿使用。

## 函数原型<a name="section620mcpsimp"></a>

```
bool SetOutputPadding(int64_t outputPadD, int64_t outputPadH, int64_t outputPadW)
```

## 参数说明<a name="section622mcpsimp"></a>

**表 1**  参数说明

<a name="table2014052461915"></a>
<table><thead align="left"><tr id="row6152102431917"><th class="cellrowborder" valign="top" width="33.333333333333336%" id="mcps1.2.4.1.1"><p id="p1215282418194"><a name="p1215282418194"></a><a name="p1215282418194"></a>参数名</p>
</th>
<th class="cellrowborder" valign="top" width="33.333333333333336%" id="mcps1.2.4.1.2"><p id="p615292417199"><a name="p615292417199"></a><a name="p615292417199"></a>输入/输出</p>
</th>
<th class="cellrowborder" valign="top" width="33.333333333333336%" id="mcps1.2.4.1.3"><p id="p81525241194"><a name="p81525241194"></a><a name="p81525241194"></a>描述</p>
</th>
</tr>
</thead>
<tbody><tr id="row2061818462519"><td class="cellrowborder" valign="top" width="33.333333333333336%" headers="mcps1.2.4.1.1 "><p id="p661954162515"><a name="p661954162515"></a><a name="p661954162515"></a>outputPadD</p>
</td>
<td class="cellrowborder" valign="top" width="33.333333333333336%" headers="mcps1.2.4.1.2 "><p id="p0619144142519"><a name="p0619144142519"></a><a name="p0619144142519"></a>输入</p>
</td>
<td class="cellrowborder" valign="top" width="33.333333333333336%" headers="mcps1.2.4.1.3 "><p id="p9254125910318"><a name="p9254125910318"></a><a name="p9254125910318"></a>输出在Depth方向的Padding值。</p>
</td>
</tr>
<tr id="row127371872259"><td class="cellrowborder" valign="top" width="33.333333333333336%" headers="mcps1.2.4.1.1 "><p id="p773777172512"><a name="p773777172512"></a><a name="p773777172512"></a>outputPadH</p>
</td>
<td class="cellrowborder" valign="top" width="33.333333333333336%" headers="mcps1.2.4.1.2 "><p id="p1573757162518"><a name="p1573757162518"></a><a name="p1573757162518"></a>输入</p>
</td>
<td class="cellrowborder" valign="top" width="33.333333333333336%" headers="mcps1.2.4.1.3 "><p id="p82981758103110"><a name="p82981758103110"></a><a name="p82981758103110"></a>输出在Height方向的Padding值。</p>
</td>
</tr>
<tr id="row215292411915"><td class="cellrowborder" valign="top" width="33.333333333333336%" headers="mcps1.2.4.1.1 "><p id="p1615252471917"><a name="p1615252471917"></a><a name="p1615252471917"></a>outputPadW</p>
</td>
<td class="cellrowborder" valign="top" width="33.333333333333336%" headers="mcps1.2.4.1.2 "><p id="p815215248192"><a name="p815215248192"></a><a name="p815215248192"></a>输入</p>
</td>
<td class="cellrowborder" valign="top" width="33.333333333333336%" headers="mcps1.2.4.1.3 "><p id="p6152162412194"><a name="p6152162412194"></a><a name="p6152162412194"></a>输出在Width方向的Padding值。</p>
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
ConvBackpropApi::Conv3DBpInputTiling conv3DBpDxTiling(*ascendcPlatform);
conv3DBpDxTiling.SetOutputPadding(outputPadD, outputPadH, outputPadW);
```

