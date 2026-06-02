# SetStride

**页面ID:** atlasascendc_api_07_0941  
**来源:** https://www.hiascend.com/document/detail/zh/CANNCommunityEdition/850/API/ascendcopapi/atlasascendc_api_07_0941.html

---

#### 功能说明

设置Stride信息。

#### 函数原型

```
void SetStride(int64_t strideD, int64_t strideH, int64_t strideW)
```

#### 参数说明

**表1 **参数说明

| 参数名 | 输入/输出 | 描述 |
| --- | --- | --- |
| strideD | 输入 | 卷积正向过程中Depth方向Stride的大小。 |
| strideH | 输入 | 卷积正向过程中Height方向Stride的大小。 |
| strideW | 输入 | 卷积正向过程中Width方向Stride的大小。 |

#### 约束说明

无

#### 调用示例

```
auto ascendcPlatform = platform_ascendc::PlatformAscendCManager::GetInstance();
ConvBackpropApi::Conv3DBpInputTiling conv3DBpDxTiling(*ascendcPlatform);
conv3DBpDxTiling.SetStride(strideD, strideH, strideW);
```
