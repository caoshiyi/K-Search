# SetPadding

**页面ID:** atlasascendc_api_07_0915  
**来源:** https://www.hiascend.com/document/detail/zh/CANNCommunityEdition/850/API/ascendcopapi/atlasascendc_api_07_0915.html

---

#### 功能说明

设置Pad信息。

#### 函数原型

```
void SetPadding(int64_t padFront, int64_t padBack, int64_t padUp, int64_t padDown, int64_t padLeft, int64_t padRight)
```

#### 参数说明

**表1 **参数说明

| 参数名 | 输入/输出 | 描述 |
| --- | --- | --- |
| padFront | 输入 | 卷积正向过程中Input Depth维度的前方向Padding大小。 |
| padBack | 输入 | 卷积正向过程中Input Depth维度的后方向Padding大小。 |
| padUp | 输入 | 卷积正向过程中Input Height维度的上方向Padding大小。 |
| padDown | 输入 | 卷积正向过程中Input Height维度的下方向Padding大小。 |
| padLeft | 输入 | 卷积正向过程中Input Width维度的左方向Padding大小。 |
| padRight | 输入 | 卷积正向过程中Input Width维度的右方向Padding大小。 |

#### 约束说明

无

#### 调用示例

```
optiling::Conv3DBackpropFilterTilingData tilingData;
auto ascendcPlatform = platform_ascendc::PlatformAscendCManager::GetInstance();
ConvBackpropApi::Conv3dBpFilterTiling conv3dBpDwTiling(*ascendcPlatform);
conv3dBpDwTiling.SetPadding(padFront, padBack, padUp, padDown, padLeft, padRight);
```
