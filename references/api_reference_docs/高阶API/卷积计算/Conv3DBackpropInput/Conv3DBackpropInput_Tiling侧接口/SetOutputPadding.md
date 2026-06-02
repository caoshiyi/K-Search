# SetOutputPadding

**页面ID:** atlasascendc_api_07_10192  
**来源:** https://www.hiascend.com/document/detail/zh/CANNCommunityEdition/850/API/ascendcopapi/atlasascendc_api_07_10192.html

---

#### 功能说明

构建Conv3DTranspose算子时，设置输出的Padding大小，用于推导输出的形状。在构建Conv3DBackpropInput算子时，此接口无实际意义，请勿使用。

#### 函数原型

```
bool SetOutputPadding(int64_t outputPadD, int64_t outputPadH, int64_t outputPadW)
```

#### 参数说明

**表1 **参数说明

| 参数名 | 输入/输出 | 描述 |
| --- | --- | --- |
| outputPadD | 输入 | 输出在Depth方向的Padding值。 |
| outputPadH | 输入 | 输出在Height方向的Padding值。 |
| outputPadW | 输入 | 输出在Width方向的Padding值。 |

#### 返回值说明

true表示设置成功，false表示设置失败。

#### 约束说明

无

#### 调用示例

```
auto ascendcPlatform = platform_ascendc::PlatformAscendCManager::GetInstance();
ConvBackpropApi::Conv3DBpInputTiling conv3DBpDxTiling(*ascendcPlatform);
conv3DBpDxTiling.SetOutputPadding(outputPadD, outputPadH, outputPadW);
```
