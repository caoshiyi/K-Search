# SetWeightType<a name="ZH-CN_TOPIC_0000002523343548"></a>

## 功能说明<a name="section20807931162211"></a>

设置Weight在内存上的位置、数据格式和数据类型。

## 函数原型<a name="section63151738172215"></a>

```
void SetWeightType(const ConvCommonApi::TPosition pos, const ConvCommonApi::ConvFormat format, const ConvCommonApi::ConvDtype dtype)
```

## 参数说明<a name="section179094442227"></a>

<a name="table128352037123219"></a>
<table><thead align="left"><tr id="row1390314370321"><th class="cellrowborder" valign="top" width="24.25%" id="mcps1.1.4.1.1"><p id="p10903133713325"><a name="p10903133713325"></a><a name="p10903133713325"></a>参数名</p>
</th>
<th class="cellrowborder" valign="top" width="17.29%" id="mcps1.1.4.1.2"><p id="p790373719325"><a name="p790373719325"></a><a name="p790373719325"></a>输入/输出</p>
</th>
<th class="cellrowborder" valign="top" width="58.46%" id="mcps1.1.4.1.3"><p id="p1690393793216"><a name="p1690393793216"></a><a name="p1690393793216"></a>描述</p>
</th>
</tr>
</thead>
<tbody><tr id="row990303793211"><td class="cellrowborder" valign="top" width="24.25%" headers="mcps1.1.4.1.1 "><p id="p199038375322"><a name="p199038375322"></a><a name="p199038375322"></a><span>pos</span></p>
</td>
<td class="cellrowborder" valign="top" width="17.29%" headers="mcps1.1.4.1.2 "><p id="p1490423717324"><a name="p1490423717324"></a><a name="p1490423717324"></a>输入</p>
</td>
<td class="cellrowborder" valign="top" width="58.46%" headers="mcps1.1.4.1.3 "><p id="p790416372325"><a name="p790416372325"></a><a name="p790416372325"></a><span>Weight在内存上的</span><a href="通用说明和约束.md#table07372185712">位置</a><span>。</span>当前仅支持TPosition::GM。</p>
</td>
</tr>
<tr id="row1690493714327"><td class="cellrowborder" valign="top" width="24.25%" headers="mcps1.1.4.1.1 "><p id="p2904173713326"><a name="p2904173713326"></a><a name="p2904173713326"></a><span>format</span></p>
</td>
<td class="cellrowborder" valign="top" width="17.29%" headers="mcps1.1.4.1.2 "><p id="p1190483720327"><a name="p1190483720327"></a><a name="p1190483720327"></a>输入</p>
</td>
<td class="cellrowborder" valign="top" width="58.46%" headers="mcps1.1.4.1.3 "><p id="p2904143713215"><a name="p2904143713215"></a><a name="p2904143713215"></a><span>Weight的数据格式</span>。当前仅支持ConvFormat::FRACTAL_Z_3D。</p>
</td>
</tr>
<tr id="row2090453773211"><td class="cellrowborder" valign="top" width="24.25%" headers="mcps1.1.4.1.1 "><p id="p1190413711324"><a name="p1190413711324"></a><a name="p1190413711324"></a><span>dtype</span></p>
</td>
<td class="cellrowborder" valign="top" width="17.29%" headers="mcps1.1.4.1.2 "><p id="p8904837193212"><a name="p8904837193212"></a><a name="p8904837193212"></a>输入</p>
</td>
<td class="cellrowborder" valign="top" width="58.46%" headers="mcps1.1.4.1.3 "><p id="p109046373322"><a name="p109046373322"></a><a name="p109046373322"></a><span>Weight的数据类型</span>。当前仅支持ConvDtype::FLOAT16、ConvDtype::BF16。</p>
</td>
</tr>
</tbody>
</table>

## 返回值说明<a name="section940510519226"></a>

无

## 约束说明<a name="section1313778142314"></a>

在调用GetTiling接口前，本接口可选调用。若未调用本接口，默认Weight为pos=TPosition::GM，format=ConvFormat::FRACTAL\_Z\_3D，dtype=ConvDtype::FLOAT16。

## 调用示例<a name="section4957171462318"></a>

```
// 实例化Conv3D Api
auto ascendcPlatform = platform_ascendc::PlatformAscendC(context->GetPlatformInfo());
Conv3dTilingApi::Conv3dTiling conv3dApiTiling(ascendcPlatform);
conv3dApiTiling.SetWeightType(ConvCommonApi::TPosition::GM, ConvCommonApi::ConvFormat::FRACTAL_Z_3D, ConvCommonApi::ConvDtype::BF16);
```

