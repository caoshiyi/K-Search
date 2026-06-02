# SetInputType

**页面ID:** atlasascendc_api_07_10088  
**来源:** https://www.hiascend.com/document/detail/zh/CANNCommunityEdition/850/API/ascendcopapi/atlasascendc_api_07_10088.html

---

#### 功能说明

设置Input在内存上的位置、数据格式和数据类型。

#### 函数原型

```
void SetInputType(const ConvCommonApi::TPosition pos, const ConvCommonApi::ConvFormat format, const ConvCommonApi::ConvDtype dtype)
```

#### 参数说明

| 参数名 | 输入/输出 | 描述 |
| --- | --- | --- |
| pos | 输入 | Input在内存上的位置。当前仅支持TPosition::GM。 |
| format | 输入 | Input的数据格式。当前仅支持ConvFormat::NDC1HWC0。 |
| dtype | 输入 | Input的数据类型。当前仅支持ConvDtype::FLOAT16、ConvDtype::BF16。 |

#### 约束说明

在调用GetTiling接口前，本接口可选调用。若未调用本接口，默认Input为pos=TPosition::GM，format=ConvFormat::NDC1HWC0，dtype=ConvDtype::FLOAT16。

#### 调用示例

```
// 实例化Conv3D Api
auto ascendcPlatform = platform_ascendc::PlatformAscendC(context->GetPlatformInfo());
Conv3dTilingApi::Conv3dTiling conv3dApiTiling(ascendcPlatform);
conv3dApiTiling.SetInputType(ConvCommonApi::TPosition::GM, ConvCommonApi::ConvFormat::NDC1HWC0, ConvCommonApi::ConvDtype::BF16);
```
