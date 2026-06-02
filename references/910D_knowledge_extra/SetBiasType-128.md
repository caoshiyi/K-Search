# SetBiasType<a name="ZH-CN_TOPIC_0000002554343645"></a>

## 功能说明<a name="section16222382320"></a>

设置Bias在内存上的位置、数据格式和数据类型。

## 函数原型<a name="section1037472892319"></a>

```
void SetBiasType(const ConvCommonApi::TPosition pos, const ConvCommonApi::ConvFormat format, const ConvCommonApi::ConvDtype dtype)
```

## 参数说明<a name="section97451834152310"></a>

<a name="table1529673615331"></a>
<table><thead align="left"><tr id="row3312153613331"><th class="cellrowborder" valign="top" width="24.25%" id="mcps1.1.4.1.1"><p id="p131233653318"><a name="p131233653318"></a><a name="p131233653318"></a>参数名</p>
</th>
<th class="cellrowborder" valign="top" width="17.29%" id="mcps1.1.4.1.2"><p id="p133121636193320"><a name="p133121636193320"></a><a name="p133121636193320"></a>输入/输出</p>
</th>
<th class="cellrowborder" valign="top" width="58.46%" id="mcps1.1.4.1.3"><p id="p143121366334"><a name="p143121366334"></a><a name="p143121366334"></a>描述</p>
</th>
</tr>
</thead>
<tbody><tr id="row15312936153319"><td class="cellrowborder" valign="top" width="24.25%" headers="mcps1.1.4.1.1 "><p id="p73122036153314"><a name="p73122036153314"></a><a name="p73122036153314"></a><span>pos</span></p>
</td>
<td class="cellrowborder" valign="top" width="17.29%" headers="mcps1.1.4.1.2 "><p id="p431219363330"><a name="p431219363330"></a><a name="p431219363330"></a>输入</p>
</td>
<td class="cellrowborder" valign="top" width="58.46%" headers="mcps1.1.4.1.3 "><p id="p83129369332"><a name="p83129369332"></a><a name="p83129369332"></a><span>Bias在内存上的</span><a href="通用说明和约束.md#table07372185712">位置</a><span>。</span>当前仅支持TPosition::GM。</p>
</td>
</tr>
<tr id="row15312133610337"><td class="cellrowborder" valign="top" width="24.25%" headers="mcps1.1.4.1.1 "><p id="p931233612337"><a name="p931233612337"></a><a name="p931233612337"></a><span>format</span></p>
</td>
<td class="cellrowborder" valign="top" width="17.29%" headers="mcps1.1.4.1.2 "><p id="p11312836203314"><a name="p11312836203314"></a><a name="p11312836203314"></a>输入</p>
</td>
<td class="cellrowborder" valign="top" width="58.46%" headers="mcps1.1.4.1.3 "><p id="p531353613339"><a name="p531353613339"></a><a name="p531353613339"></a><span>Bias的数据格式。</span>当前仅支持ConvFormat::ND。</p>
</td>
</tr>
<tr id="row1431311361332"><td class="cellrowborder" valign="top" width="24.25%" headers="mcps1.1.4.1.1 "><p id="p1931343673316"><a name="p1931343673316"></a><a name="p1931343673316"></a><span>dtype</span></p>
</td>
<td class="cellrowborder" valign="top" width="17.29%" headers="mcps1.1.4.1.2 "><p id="p19313183619338"><a name="p19313183619338"></a><a name="p19313183619338"></a>输入</p>
</td>
<td class="cellrowborder" valign="top" width="58.46%" headers="mcps1.1.4.1.3 "><p id="p531315364331"><a name="p531315364331"></a><a name="p531315364331"></a><span>Bias的数据类型。</span>当前仅支持ConvDtype::FLOAT16、ConvDtype::FLOAT。</p>
</td>
</tr>
</tbody>
</table>

## 返回值说明<a name="section7948194016232"></a>

无

## 约束说明<a name="section993695312315"></a>

在调用GetTiling接口前，本接口可选调用。若未调用本接口，默认Bias为pos=TPosition::GM，format=ConvFormat::ND，dtype=ConvDtype::FLOAT16。

## 调用示例<a name="section96939599230"></a>

```
// 实例化Conv3D Api
auto ascendcPlatform = platform_ascendc::PlatformAscendC(context->GetPlatformInfo());
Conv3dTilingApi::Conv3dTiling conv3dApiTiling(ascendcPlatform);
conv3dApiTiling.SetBiasType(ConvCommonApi::TPosition::GM, ConvCommonApi::ConvFormat::ND, ConvCommonApi::ConvDtype::FLOAT32);
```

