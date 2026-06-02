# SetWeightShape

**页面ID:** atlasascendc_api_07_0909  
**来源:** https://www.hiascend.com/document/detail/zh/CANNCommunityEdition/850/API/ascendcopapi/atlasascendc_api_07_0909.html

---

#### 功能说明

设置权重矩阵Weight的形状。

#### 函数原型

```
void SetWeightShape(int64_t cout, int64_t cin, int64_t d, int64_t h, int64_t w)
```

#### 参数说明

**表1 **参数说明

| 参数名 | 输入/输出 | 描述 |
| --- | --- | --- |
| cout | 输入 | 设置GradOutput的Channel值。 |
| cin | 输入 | 设置Input的Channel值。 |
| d | 输入 | 设置Weight的Depth值。 |
| h | 输入 | 设置Weight的Height值。 |
| w | 输入 | 设置Weight的Width值。 |

#### 约束说明

无

#### 调用示例

```
optiling::Conv3DBackpropFilterTilingData tilingData;
auto ascendcPlatform = platform_ascendc::PlatformAscendCManager::GetInstance();
ConvBackpropApi::Conv3dBpFilterTiling conv3dBpDwTiling(*ascendcPlatform);
conv3dBpDwTiling.SetWeightShape(cout, cin, d, h, w);
```
