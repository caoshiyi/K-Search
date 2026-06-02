# SetStride

**页面ID:** atlasascendc_api_07_10094  
**来源:** https://www.hiascend.com/document/detail/zh/CANNCommunityEdition/850/API/ascendcopapi/atlasascendc_api_07_10094.html

---

#### 功能说明

设置Stride信息。

#### 函数原型

```
void SetStride(int64_t strideD, int64_t strideH, int64_t strideW)
```

#### 参数说明

| 参数名 | 输入/输出 | 描述 |
| --- | --- | --- |
| strideD | 输入 | D方向Stride大小。 |
| strideH | 输入 | H方向Stride大小。 |
| strideW | 输入 | W方向Stride大小。 |

#### 约束说明

在调用GetTiling接口前，本接口可选调用。若未调用本接口，默认strideD=1, strideH=1, strideW=1。

#### 调用示例

```
// 实例化Conv3D Api
auto ascendcPlatform = platform_ascendc::PlatformAscendC(context->GetPlatformInfo());
Conv3dTilingApi::Conv3dTiling conv3dApiTiling(ascendcPlatform);
conv3dApiTiling.SetStride(strideD, strideH, strideW);
```
