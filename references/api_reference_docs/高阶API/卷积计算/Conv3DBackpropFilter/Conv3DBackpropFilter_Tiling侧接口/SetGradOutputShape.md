# SetGradOutputShape

**页面ID:** atlasascendc_api_07_0911  
**来源:** https://www.hiascend.com/document/detail/zh/CANNCommunityEdition/850/API/ascendcopapi/atlasascendc_api_07_0911.html

---

#### 功能说明

设置GradOutput的形状：Batch、Channel、Depth、Height、Width。

#### 函数原型

```
void SetGradOutputShape(int64_t n, int64_t c, int64_t d, int64_t h, int64_t w)
```

#### 参数说明

**表1 **参数说明

| 参数名 | 输入/输出 | 描述 |
| --- | --- | --- |
| n | 输入 | 输入GradOutput的Batch值。 |
| c | 输入 | 输入GradOutput的Channel值。 |
| d | 输入 | 输入GradOutput的Depth值。 |
| h | 输入 | 输入GradOutput的Height值。 |
| w | 输入 | 输入GradOutput的Width值。 |

#### 约束说明

无

#### 调用示例

```
auto ascendcPlatform = platform_ascendc::PlatformAscendCManager::GetInstance();
ConvBackpropApi::Conv3dBpFilterTiling conv3dBpDwTiling(*ascendcPlatform);
conv3dBpDwTiling.SetGradOutputShape(n, c, d, h, w);
```
