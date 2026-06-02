# SetPadding

**页面ID:** atlasascendc_api_07_10092  
**来源:** https://www.hiascend.com/document/detail/zh/CANNCommunityEdition/850/API/ascendcopapi/atlasascendc_api_07_10092.html

---

#### 功能说明

设置Pad信息。

#### 函数原型

```
void SetPadding(int64_t padHead, int64_t padTail, int64_t padUp, int64_t padDown, int64_t padLeft, int64_t padRight)
```

#### 参数说明

| 参数名 | 输入/输出 | 描述 |
| --- | --- | --- |
| padHead | 输入 | D方向前Padding大小。 |
| padTail | 输入 | D方向后Padding大小。 |
| padUp | 输入 | H方向上Padding大小。 |
| padDown | 输入 | H方向下Padding大小。 |
| padLeft | 输入 | W方向左Padding大小。 |
| padRight | 输入 | W方向右Padding大小。 |

#### 约束说明

在调用GetTiling接口前，本接口可选调用。若未调用本接口，默认padHead=0, padTail=0, padUp=0, padDown=0, padLeft=0, padRight=0。

#### 调用示例

```
// 实例化Conv3D Api
auto ascendcPlatform = platform_ascendc::PlatformAscendC(context->GetPlatformInfo());
Conv3dTilingApi::Conv3dTiling conv3dApiTiling(ascendcPlatform);
conv3dApiTiling.SetPadding(padHead, padTail, padUp, padDown, padLeft, padRight);
```
