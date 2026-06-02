# SetInputType

**页面ID:** atlasascendc_api_07_0913  
**来源:** https://www.hiascend.com/document/detail/zh/CANNCommunityEdition/850/API/ascendcopapi/atlasascendc_api_07_0913.html

---

#### 功能说明

设置特征矩阵Input的位置、数据格式、数据类型信息，这些信息必须与Kernel侧的设置保持一致。

#### 函数原型

```
void SetInputType(ConvCommonApi::TPosition pos, ConvCommonApi::ConvFormat format, ConvCommonApi::ConvDtype dtype)
```

#### 参数说明

**表1 **参数说明

| 参数名 | 输入/输出 | 描述 |
| --- | --- | --- |
| pos | 输入 | Input在内存上的位置。当前仅支持TPosition::GM。 |
| format | 输入 | Input的数据格式。当前仅支持ConvFormat::NDC1HWC0。 |
| dtype | 输入 | Input的数据类型。当前仅支持ConvDtype::FLOAT16、ConvDtype::BF16。 |

#### 约束说明

无

#### 调用示例

```
optiling::Conv3DBackpropFilterTilingData tilingData;
auto ascendcPlatform = platform_ascendc::PlatformAscendCManager::GetInstance();
ConvBackpropApi::Conv3dBpFilterTiling conv3dBpDwTiling(*ascendcPlatform);
conv3dBpDwTiling.SetInputType(ConvCommonApi::TPosition::GM,
                                 ConvCommonApi::ConvFormat::NDC1HWC0,
                                 ConvCommonApi::ConvDtype::FLOAT16);
```
