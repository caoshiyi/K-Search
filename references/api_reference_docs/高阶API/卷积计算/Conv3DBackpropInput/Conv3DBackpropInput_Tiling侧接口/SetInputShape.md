# SetInputShape

**页面ID:** atlasascendc_api_07_0935  
**来源:** https://www.hiascend.com/document/detail/zh/CANNCommunityEdition/850/API/ascendcopapi/atlasascendc_api_07_0935.html

---

#### 功能说明

设置特征矩阵Input的形状：Batch、Channel、Depth、Height、Width。在构建Conv3DTranspose算子时，此接口无实际意义，请勿使用。

#### 函数原型

```
bool SetInputShape(int64_t n, int64_t c, int64_t d, int64_t h, int64_t w)
```

#### 参数说明

**表1 **参数说明

| 参数名 | 输入/输出 | 描述 |
| --- | --- | --- |
| n | 输入 | 输入Input的Batch值。 |
| c | 输入 | 输入Input的Channel值。 |
| d | 输入 | 输入Input的Depth值。 |
| h | 输入 | 输入Input的Height值。 |
| w | 输入 | 输入Input的Width值。 |

#### 返回值说明

true表示设置成功，false表示设置失败。

#### 约束说明

无

#### 调用示例

```
auto ascendcPlatform = platform_ascendc::PlatformAscendCManager::GetInstance();
ConvBackpropApi::Conv3DBpInputTiling conv3DBpDxTiling(*ascendcPlatform);
conv3DBpDxTiling.SetInputShape(n, c, d, h, w);
```
