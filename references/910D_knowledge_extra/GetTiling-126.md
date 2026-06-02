# GetTiling<a name="ZH-CN_TOPIC_0000002554423479"></a>

## 功能说明<a name="section183813941414"></a>

获取Tiling参数。

## 函数原型<a name="section20998204515148"></a>

```
int64_t GetTiling(optiling::TConv3DApiTiling& tiling)
```

```
int64_t GetTiling(AscendC::tiling::TConv3DApiTiling& tiling)
```

## 参数说明<a name="section6806853101410"></a>

<a name="table167479752716"></a>
<table><thead align="left"><tr id="row67671716270"><th class="cellrowborder" valign="top" width="24.25%" id="mcps1.1.4.1.1"><p id="p1176714772711"><a name="p1176714772711"></a><a name="p1176714772711"></a>参数名</p>
</th>
<th class="cellrowborder" valign="top" width="17.29%" id="mcps1.1.4.1.2"><p id="p376716742713"><a name="p376716742713"></a><a name="p376716742713"></a>输入/输出</p>
</th>
<th class="cellrowborder" valign="top" width="58.46%" id="mcps1.1.4.1.3"><p id="p2076710710279"><a name="p2076710710279"></a><a name="p2076710710279"></a>描述</p>
</th>
</tr>
</thead>
<tbody><tr id="row15767972272"><td class="cellrowborder" valign="top" width="24.25%" headers="mcps1.1.4.1.1 "><p id="p1776767112710"><a name="p1776767112710"></a><a name="p1776767112710"></a>tiling</p>
</td>
<td class="cellrowborder" valign="top" width="17.29%" headers="mcps1.1.4.1.2 "><p id="p12767574272"><a name="p12767574272"></a><a name="p12767574272"></a>输出</p>
</td>
<td class="cellrowborder" valign="top" width="58.46%" headers="mcps1.1.4.1.3 "><p id="p576811713275"><a name="p576811713275"></a><a name="p576811713275"></a>Conv3D的Tiling结构体，用于存储最终的Tiling结果。TConv3DApiTiling结构介绍请参考<a href="TConv3DApiTiling结构体.md">TConv3DApiTiling结构体</a>。</p>
</td>
</tr>
</tbody>
</table>

## 返回值说明<a name="section1886828152"></a>

如果返回值不为-1，则代表Tiling计算成功，用户可以使用该Tiling结构的值。如果返回值为-1，则代表Tiling计算失败，该Tiling结果无法使用。

## 约束说明<a name="section4390615181519"></a>

调用GetTiling接口前必须先调用SetOrgInputShape、SetOrgWeightShape、SetSingleWeightShape、SetSingleOutputShape。

## 调用示例<a name="section168038344159"></a>

```
// 实例化Conv3d Api
auto ascendcPlatform = platform_ascendc::PlatformAscendC(context->GetPlatformInfo());
Conv3dTilingApi::Conv3dTiling conv3dApiTiling(ascendcPlatform );
conv3dApiTiling.SetOrgInputShape(orgCi, orgDi, orgHi, orgWi);
conv3dApiTiling.SetOrgWeightShape(cout, kd, kh, kw);
conv3dApiTiling.SetSingleWeightShape(singleCi, singleKd, singleKh, singleKw);
conv3dApiTiling.SetSingleOutputShape(singleCo, singleDo, singleM);
...
conv3dApiTiling.GetTiling(tilingData.conv3ApiTilingData);
```

