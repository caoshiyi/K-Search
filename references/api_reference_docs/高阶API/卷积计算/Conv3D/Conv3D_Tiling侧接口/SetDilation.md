# SetDilation

**页面ID:** atlasascendc_api_07_10093  
**来源:** https://www.hiascend.com/document/detail/zh/CANNCommunityEdition/850/API/ascendcopapi/atlasascendc_api_07_10093.html

---

#### 功能说明

设置Dilation信息。

#### 函数原型

```
void SetDilation(int64_t dilationD, int64_t dilationH, int64_t dilationW)
```

#### 参数说明

| 参数名 | 输入/输出 | 描述 |
| --- | --- | --- |
| dilationD | 输入 | D方向Dilation大小。 |
| dilationH | 输入 | H方向Dilation大小。 |
| dilationW | 输入 | W方向Dilation大小。 |

#### 约束说明

在调用GetTiling接口前，本接口可选调用。若未调用本接口，默认dilationD=1, dilationH=1, dilationW=1。

#### 调用示例

```
// 实例化Conv3D Api
auto ascendcPlatform = platform_ascendc::PlatformAscendC(context->GetPlatformInfo());
Conv3dTilingApi::Conv3dTiling conv3dApiTiling(ascendcPlatform);
conv3dApiTiling.SetDilation(dilationD, dilationH, dilationW);
```
