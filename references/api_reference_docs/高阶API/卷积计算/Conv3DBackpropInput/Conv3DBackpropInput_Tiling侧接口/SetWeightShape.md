# SetWeightShape

**页面ID:** atlasascendc_api_07_0934  
**来源:** https://www.hiascend.com/document/detail/zh/CANNCommunityEdition/850/API/ascendcopapi/atlasascendc_api_07_0934.html

---

#### 功能说明

设置权重矩阵Weight的形状。

#### 函数原型

```
bool SetWeightShape(int64_t cout, int64_t cin, int64_t d, int64_t h, int64_t w)
```

#### 参数说明

**表1 **参数说明

| 参数名 | 输入/输出 | 描述 |
| --- | --- | --- |
| cout | 输入 | 设置卷积正向的输出channel大小，与GradOutput的Channel大小一致。 |
| cin | 输入 | 设置卷积正向的输入channel大小，与GradInput的Channel大小一致。 |
| d | 输入 | 设置weight的Depth值。 |
| h | 输入 | 设置weight的Height值。 |
| w | 输入 | 设置weight的Width值。 |

#### 返回值说明

true表示设置成功，false表示设置失败。

#### 约束说明

无

#### 调用示例

```
auto ascendcPlatform = platform_ascendc::PlatformAscendCManager::GetInstance();
ConvBackpropApi::Conv3DBpInputTiling con3dBpDxTiling(*ascendcPlatform);
con3dBpDxTiling.SetWeightShape(cout, cin, d, h, w);
```
