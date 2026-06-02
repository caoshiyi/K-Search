# SetOrgInputShape

**页面ID:** atlasascendc_api_07_10085  
**来源:** https://www.hiascend.com/document/detail/zh/CANNCommunityEdition/850/API/ascendcopapi/atlasascendc_api_07_10085.html

---

#### 功能说明

设置特征矩阵Input的原始形状。

#### 函数原型

```
void SetOrgInputShape(int64_t orgCi, int64_t orgDi, int64_t orgHi, int64_t orgWi)
```

#### 参数说明

| 参数名 | 输入/输出 | 描述 |
| --- | --- | --- |
| orgCi | 输入 | 原始输入通道的大小。 |
| orgDi | 输入 | 原始Input D维度大小。 |
| orgHi | 输入 | 原始Input H维度大小。 |
| orgWi | 输入 | 原始Input W维度大小。 |

#### 约束说明

无

#### 调用示例

```
// 实例化Conv3D Api
auto ascendcPlatform = platform_ascendc::PlatformAscendC(context->GetPlatformInfo());
Conv3dTilingApi::Conv3dTiling conv3dApiTiling(ascendcPlatform);
conv3dApiTiling.SetOrgInputShape(orgCi, orgDi, orgHi, orgWi);
```
