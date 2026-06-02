# SetDilation

**页面ID:** atlasascendc_api_07_0917  
**来源:** https://www.hiascend.com/document/detail/zh/CANNCommunityEdition/850/API/ascendcopapi/atlasascendc_api_07_0917.html

---

#### 功能说明

设置Dilation信息，即卷积核Depth/Height/Width方向的扩张大小。

#### 函数原型

```
void SetDilation(int64_t dilationD, int64_t dilationH, int64_t dilationW)
```

#### 参数说明

**表1 **参数说明

| 参数名 | 输入/输出 | 描述 |
| --- | --- | --- |
| dilationD | 输入 | 卷积核Weight的Depth方向扩张大小。 |
| dilationH | 输入 | 卷积核Weight的Height方向扩张大小。 |
| dilationW | 输入 | 卷积核Weight的Width方向扩张大小。 |

#### 约束说明

无

#### 调用示例

```
optiling::Conv3DBackpropFilterTilingData tilingData;
auto ascendcPlatform = platform_ascendc::PlatformAscendCManager::GetInstance();
ConvBackpropApi::Conv3dBpFilterTiling conv3dBpDwTiling(*ascendcPlatform);
conv3dBpDwTiling.SetDilation(dilationD, dilationH, dilationW);
```
