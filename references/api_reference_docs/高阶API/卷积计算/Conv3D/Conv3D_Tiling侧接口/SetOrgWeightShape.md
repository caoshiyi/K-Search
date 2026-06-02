# SetOrgWeightShape

**页面ID:** atlasascendc_api_07_10084  
**来源:** https://www.hiascend.com/document/detail/zh/CANNCommunityEdition/850/API/ascendcopapi/atlasascendc_api_07_10084.html

---

#### 功能说明

设置权重矩阵Weight的原始形状。

#### 函数原型

```
void SetOrgWeightShape(int64_t orgCo, int64_t orgKd, int64_t orgKh, int64_t orgKw)
```

#### 参数说明

| 参数名 | 输入/输出 | 描述 |
| --- | --- | --- |
| orgCo | 输入 | 原始输出通道的大小。 |
| orgKd | 输入 | 原始Weight D维度大小。 |
| orgKh | 输入 | 原始Weight H维度大小。 |
| orgKw | 输入 | 原始Weight W维度大小。 |

#### 约束说明

无

#### 调用示例

```
// 实例化Conv3D Api
auto ascendcPlatform = platform_ascendc::PlatformAscendC(context->GetPlatformInfo());
Conv3dTilingApi::Conv3dTiling conv3dApiTiling(ascendcPlatform);
conv3dApiTiling.SetOrgWeightShape(cout, kd, kh, kw);
```
