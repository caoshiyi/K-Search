# SetGradOutputType<a name="ZH-CN_TOPIC_0000002523304776"></a>

## 功能说明<a name="section618mcpsimp"></a>

设置GradOutput的位置、数据格式、数据类型信息，这些信息必须与Kernel侧的设置保持一致。

## 函数原型<a name="section620mcpsimp"></a>

```
void SetGradOutputType(ConvCommonApi::TPosition pos, ConvCommonApi::ConvFormat format, ConvCommonApi::ConvDtype dtype)
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
<tbody><tr id="row106481443135617"><td class="cellrowborder" valign="top" width="14.99%" headers="mcps1.2.4.1.1 "><p id="p0263115751818"><a name="p0263115751818"></a><a name="p0263115751818"></a>pos</p>
</td>
<td class="cellrowborder" valign="top" width="12.02%" headers="mcps1.2.4.1.2 "><p id="p72631357201810"><a name="p72631357201810"></a><a name="p72631357201810"></a>输入</p>
</td>
<td class="cellrowborder" valign="top" width="72.99%" headers="mcps1.2.4.1.3 "><p id="p1746195213154"><a name="p1746195213154"></a><a name="p1746195213154"></a>GradOutput<span>在内存上的</span><a href="通用说明和约束.md#table07372185712">位置</a><span>。</span>当前仅支持TPosition::GM。</p>
</td>
</tr>
<tr id="row583721474811"><td class="cellrowborder" valign="top" width="14.99%" headers="mcps1.2.4.1.1 "><p id="p48371814114818"><a name="p48371814114818"></a><a name="p48371814114818"></a>format</p>
</td>
<td class="cellrowborder" valign="top" width="12.02%" headers="mcps1.2.4.1.2 "><p id="p1583711474813"><a name="p1583711474813"></a><a name="p1583711474813"></a>输入</p>
</td>
<td class="cellrowborder" valign="top" width="72.99%" headers="mcps1.2.4.1.3 "><p id="p6837114154813"><a name="p6837114154813"></a><a name="p6837114154813"></a>GradOutput的数据格式<span>。</span>当前仅支持ConvFormat:NDC1HWC0。</p>
</td>
</tr>
<tr id="row1298141617488"><td class="cellrowborder" valign="top" width="14.99%" headers="mcps1.2.4.1.1 "><p id="p179821716104817"><a name="p179821716104817"></a><a name="p179821716104817"></a>dtype</p>
</td>
<td class="cellrowborder" valign="top" width="12.02%" headers="mcps1.2.4.1.2 "><p id="p109828163483"><a name="p109828163483"></a><a name="p109828163483"></a>输入</p>
</td>
<td class="cellrowborder" valign="top" width="72.99%" headers="mcps1.2.4.1.3 "><p id="p59821169486"><a name="p59821169486"></a><a name="p59821169486"></a>GradOutput的数据类型<span>。</span>当前仅支持ConvDtype::FLOAT16、ConvDtype::BF16。</p>
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
conv3dBpDwTiling.SetGradOutputType(ConvCommonApi::TPosition::GM,
                                   ConvCommonApi::ConvFormat::NDC1HWC0,
                                   ConvCommonApi::ConvDtype::FLOAT16);
```

