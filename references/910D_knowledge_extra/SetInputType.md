# SetInputType<a name="ZH-CN_TOPIC_0000002554423811"></a>

## 功能说明<a name="section6779123242112"></a>

设置Input在内存上的位置、数据格式和数据类型。

## 函数原型<a name="section1384993816218"></a>

```
void SetInputType(const ConvCommonApi::TPosition pos, const ConvCommonApi::ConvFormat format, const ConvCommonApi::ConvDtype dtype)
```

## 参数说明<a name="section1655285192112"></a>

<a name="table2521174113111"></a>
<table><thead align="left"><tr id="row11536241183111"><th class="cellrowborder" valign="top" width="24.25%" id="mcps1.1.4.1.1"><p id="p20536174113118"><a name="p20536174113118"></a><a name="p20536174113118"></a>参数名</p>
</th>
<th class="cellrowborder" valign="top" width="17.29%" id="mcps1.1.4.1.2"><p id="p11536741133113"><a name="p11536741133113"></a><a name="p11536741133113"></a>输入/输出</p>
</th>
<th class="cellrowborder" valign="top" width="58.46%" id="mcps1.1.4.1.3"><p id="p9536841193118"><a name="p9536841193118"></a><a name="p9536841193118"></a>描述</p>
</th>
</tr>
</thead>
<tbody><tr id="row1353634123112"><td class="cellrowborder" valign="top" width="24.25%" headers="mcps1.1.4.1.1 "><p id="p12536541113114"><a name="p12536541113114"></a><a name="p12536541113114"></a><span>pos</span></p>
</td>
<td class="cellrowborder" valign="top" width="17.29%" headers="mcps1.1.4.1.2 "><p id="p15371341173118"><a name="p15371341173118"></a><a name="p15371341173118"></a>输入</p>
</td>
<td class="cellrowborder" valign="top" width="58.46%" headers="mcps1.1.4.1.3 "><p id="p125371441163113"><a name="p125371441163113"></a><a name="p125371441163113"></a><span>Input在内存上的</span><a href="通用说明和约束.md#table07372185712">位置</a>。当前仅支持TPosition::GM。</p>
</td>
</tr>
<tr id="row10537641133113"><td class="cellrowborder" valign="top" width="24.25%" headers="mcps1.1.4.1.1 "><p id="p1953744113316"><a name="p1953744113316"></a><a name="p1953744113316"></a><span>format</span></p>
</td>
<td class="cellrowborder" valign="top" width="17.29%" headers="mcps1.1.4.1.2 "><p id="p1953734153116"><a name="p1953734153116"></a><a name="p1953734153116"></a>输入</p>
</td>
<td class="cellrowborder" valign="top" width="58.46%" headers="mcps1.1.4.1.3 "><p id="p105371141113119"><a name="p105371141113119"></a><a name="p105371141113119"></a><span>Input的数据格式</span>。当前仅支持ConvFormat::NDC1HWC0。</p>
</td>
</tr>
<tr id="row753784133110"><td class="cellrowborder" valign="top" width="24.25%" headers="mcps1.1.4.1.1 "><p id="p553794114319"><a name="p553794114319"></a><a name="p553794114319"></a><span>dtype</span></p>
</td>
<td class="cellrowborder" valign="top" width="17.29%" headers="mcps1.1.4.1.2 "><p id="p165371641113118"><a name="p165371641113118"></a><a name="p165371641113118"></a>输入</p>
</td>
<td class="cellrowborder" valign="top" width="58.46%" headers="mcps1.1.4.1.3 "><p id="p05371741163110"><a name="p05371741163110"></a><a name="p05371741163110"></a><span>Input的数据类型</span>。当前仅支持ConvDtype::FLOAT16、ConvDtype::BF16。</p>
</td>
</tr>
</tbody>
</table>

## 返回值说明<a name="section4429175822114"></a>

无

## 约束说明<a name="section21371515132213"></a>

在调用GetTiling接口前，本接口可选调用。若未调用本接口，默认Input为pos=TPosition::GM，format=ConvFormat::NDC1HWC0，dtype=ConvDtype::FLOAT16。

## 调用示例<a name="section7439182310226"></a>

```
// 实例化Conv3D Api
auto ascendcPlatform = platform_ascendc::PlatformAscendC(context->GetPlatformInfo());
Conv3dTilingApi::Conv3dTiling conv3dApiTiling(ascendcPlatform);
conv3dApiTiling.SetInputType(ConvCommonApi::TPosition::GM, ConvCommonApi::ConvFormat::NDC1HWC0, ConvCommonApi::ConvDtype::BF16);
```

