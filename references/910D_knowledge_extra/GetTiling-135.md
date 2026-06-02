# GetTiling<a name="ZH-CN_TOPIC_0000002523304198"></a>

## 功能说明<a name="section618mcpsimp"></a>

获取Tiling参数。

## 函数原型<a name="section620mcpsimp"></a>

```
int64_t GetTiling(optiling::Conv3DBackpropInputTilingData &tiling)
```

```
int64_t GetTiling(AscendC::tiling::Conv3DBackpropInputTilingData &tiling)
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
<tbody><tr id="row106481443135617"><td class="cellrowborder" valign="top" width="14.99%" headers="mcps1.2.4.1.1 "><p id="p201012231161"><a name="p201012231161"></a><a name="p201012231161"></a>tiling</p>
</td>
<td class="cellrowborder" valign="top" width="12.02%" headers="mcps1.2.4.1.2 "><p id="p0108231864"><a name="p0108231864"></a><a name="p0108231864"></a>输出</p>
</td>
<td class="cellrowborder" valign="top" width="72.99%" headers="mcps1.2.4.1.3 "><p id="p146859523551"><a name="p146859523551"></a><a name="p146859523551"></a>Conv3DBackpropInput的Tiling结构体，用于存储最终的Tiling结果。TConv3DBackpropInputTiling结构介绍请参考<a href="TConv3DBackpropInputTiling结构体.md">TConv3DApiTiling结构体说明</a>。</p>
</td>
</tr>
</tbody>
</table>

## 返回值说明<a name="section640mcpsimp"></a>

如果返回值不为-1，则代表Tiling计算成功，用户可以使用该Tiling结构的值。如果返回值为-1，则代表Tiling计算失败，该Tiling结果无法使用。

## 约束说明<a name="section633mcpsimp"></a>

无

## 调用示例<a name="section1665082013318"></a>

```
// 构建Conv3dBackpropInput算子tiling的调用示例
auto ascendcPlatform = platform_ascendc::PlatformAscendC(context->GetPlatformInfo());
ConvBackpropApi::Conv3DBpInputTiling tiling(ascendcPlatform); 
tiling.SetWeightShape(256, 128, 3, 3, 3);
tiling.SetInputShape(1, 128, 11, 128, 128);
tiling.SetGradOutputShape(1, 256, 9, 128, 128);
tiling.SetPadding(0, 0, 1, 1, 1, 1);   
tiling.SetDilation(1, 1, 1); 
tiling.SetStride(1, 1, 1);
optiling::Conv3DBackpropInputTilingData tilingData;   
int ret = tiling.GetTiling(tilingData);  // 获取Tiling参数
AscendC::tiling::Conv3DBackpropInputTilingData tilingDataNotOp;
ret = tiling.GetTiling(tilingDataNotOp); // 使用AscendC::tiling::Conv3DBackpropInputTilingData获取Tiling参数
```

```
// 构建Conv3dTranspose算子tiling的调用示例
auto ascendcPlatform = platform_ascendc::PlatformAscendC(context->GetPlatformInfo());
ConvBackpropApi::Conv3DBpInputTiling tiling(ascendcPlatform); 
tiling.SetWeightShape(256, 128, 3, 3, 3);
tiling.SetGradOutputShape(256, 9, 128, 128); // 等价于Conv3dTranspose的输入X
tiling.SetPadding(0, 0, 1, 1, 1, 1);   
tiling.SetDilation(1, 1, 1); 
tiling.SetStride(1, 1, 1);
tiling.SetOutputPadding(0, 0, 0); // 对Conv3dTranspose的输出Y进行padding
optiling::Conv3DBackpropInputTilingData tilingData;   
int ret = tiling.GetTiling(tilingData);  // 获取Tiling参数
```

