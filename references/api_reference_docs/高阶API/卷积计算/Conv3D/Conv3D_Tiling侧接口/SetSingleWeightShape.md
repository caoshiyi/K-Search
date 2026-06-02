# SetSingleWeightShape

**页面ID:** atlasascendc_api_07_10086  
**来源:** https://www.hiascend.com/document/detail/zh/CANNCommunityEdition/850/API/ascendcopapi/atlasascendc_api_07_10086.html

---

#### 功能说明

设置单核上权重矩阵Weight的形状。

#### 函数原型

```
void SetSingleWeightShape(int64_t singleCi, int64_t singleKd, int64_t singleKh, int64_t singleKw)
```

#### 参数说明

| 参数名 | 输入/输出 | 描述 |
| --- | --- | --- |
| singleCi | 输入 | 单核上输入通道的大小。 |
| singleKd | 输入 | 单核上Weight D维度大小。 |
| singleKh | 输入 | 单核上Weight H维度大小。 |
| singleKw | 输入 | 单核上Weight W维度大小。 |

#### 约束说明

无

#### 调用示例

```
// 实例化Conv3D Api
auto ascendcPlatform = platform_ascendc::PlatformAscendC(context->GetPlatformInfo());
Conv3dTilingApi::Conv3dTiling conv3dApiTiling(ascendcPlatform );
conv3dApiTiling.SetSingleWeightShape(singleCi, singleKd, singleKh, singleKw);
```
