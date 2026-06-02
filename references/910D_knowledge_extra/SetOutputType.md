# SetOutputType<a name="ZH-CN_TOPIC_0000002554343511"></a>

## 功能说明<a name="section746777162410"></a>

设置结果矩阵Output在内存上的位置、数据格式和数据类型。

## 函数原型<a name="section1680151342419"></a>

```
void SetOutputType(const ConvCommonApi::TPosition pos, const ConvCommonApi::ConvFormat format, const ConvCommonApi::ConvDtype dtype)
```

## 参数说明<a name="section15256320112411"></a>

<a name="table69361673418"></a>
<table><thead align="left"><tr id="row1811511653413"><th class="cellrowborder" valign="top" width="24.25%" id="mcps1.1.4.1.1"><p id="p19115151653413"><a name="p19115151653413"></a><a name="p19115151653413"></a>参数名</p>
</th>
<th class="cellrowborder" valign="top" width="17.29%" id="mcps1.1.4.1.2"><p id="p7115161653418"><a name="p7115161653418"></a><a name="p7115161653418"></a>输入/输出</p>
</th>
<th class="cellrowborder" valign="top" width="58.46%" id="mcps1.1.4.1.3"><p id="p171151716193417"><a name="p171151716193417"></a><a name="p171151716193417"></a>描述</p>
</th>
</tr>
</thead>
<tbody><tr id="row31151316143418"><td class="cellrowborder" valign="top" width="24.25%" headers="mcps1.1.4.1.1 "><p id="p911541653412"><a name="p911541653412"></a><a name="p911541653412"></a><span>pos</span></p>
</td>
<td class="cellrowborder" valign="top" width="17.29%" headers="mcps1.1.4.1.2 "><p id="p1311571673417"><a name="p1311571673417"></a><a name="p1311571673417"></a>输入</p>
</td>
<td class="cellrowborder" valign="top" width="58.46%" headers="mcps1.1.4.1.3 "><p id="p19115131613346"><a name="p19115131613346"></a><a name="p19115131613346"></a><span>Output在内存上</span><span>的</span><a href="通用说明和约束.md#table07372185712">位置</a><span>。</span>当前仅支持TPosition::CO1。</p>
</td>
</tr>
<tr id="row1611551613343"><td class="cellrowborder" valign="top" width="24.25%" headers="mcps1.1.4.1.1 "><p id="p1011517165348"><a name="p1011517165348"></a><a name="p1011517165348"></a><span>format</span></p>
</td>
<td class="cellrowborder" valign="top" width="17.29%" headers="mcps1.1.4.1.2 "><p id="p911616162340"><a name="p911616162340"></a><a name="p911616162340"></a>输入</p>
</td>
<td class="cellrowborder" valign="top" width="58.46%" headers="mcps1.1.4.1.3 "><p id="p511617163343"><a name="p511617163343"></a><a name="p511617163343"></a><span>Output的数据格式</span>。当前仅支持ConvFormat::NDC1HWC0。</p>
</td>
</tr>
<tr id="row181161116123419"><td class="cellrowborder" valign="top" width="24.25%" headers="mcps1.1.4.1.1 "><p id="p81161816133419"><a name="p81161816133419"></a><a name="p81161816133419"></a><span>dtype</span></p>
</td>
<td class="cellrowborder" valign="top" width="17.29%" headers="mcps1.1.4.1.2 "><p id="p1311611163342"><a name="p1311611163342"></a><a name="p1311611163342"></a>输入</p>
</td>
<td class="cellrowborder" valign="top" width="58.46%" headers="mcps1.1.4.1.3 "><p id="p61161516103411"><a name="p61161516103411"></a><a name="p61161516103411"></a><span>Output的数据类型</span>。当前仅支持ConvDtype::FLOAT16、ConvDtype::BF16。</p>
</td>
</tr>
</tbody>
</table>

## 返回值说明<a name="section1783842512414"></a>

无

## 约束说明<a name="section18142640162418"></a>

在调用GetTiling接口前，本接口可选调用。若未调用本接口，默认Bias为pos=TPosition::CO1，format=ConvFormat::NDC1HWC0，dtype=ConvDtype::FLOAT16。

## 调用示例<a name="section26211846192412"></a>

```
// 实例化Conv3D API
auto ascendcPlatform = platform_ascendc::PlatformAscendC(context->GetPlatformInfo());
Conv3dTilingApi::Conv3dTiling conv3dApiTiling(ascendcPlatform);
conv3dApiTiling.SetOutputType(ConvCommonApi::TPosition::CO1, ConvCommonApi::ConvFormat::NDC1HWC0, ConvCommonApi::ConvDtype::BF16);
```

