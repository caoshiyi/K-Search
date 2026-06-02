# SetSingleOutputShape

**页面ID:** atlasascendc_api_07_10087  
**来源:** https://www.hiascend.com/document/detail/zh/CANNCommunityEdition/850/API/ascendcopapi/atlasascendc_api_07_10087.html

---

#### 功能说明

设置单核上结果矩阵Output的形状。

#### 函数原型

```
void SetSingleOutputShape(int64_t singleCo, int64_t singleDo, int64_t singleM)
```

#### 参数说明

| 参数名 | 输入/输出 | 描述 |
| --- | --- | --- |
| singleCo | 输入 | 单核上输出通道的大小。 |
| singleDo | 输入 | 单核上Output D维度大小。 |
| singleM | 输入 | 单核上Output M维度大小。 |

#### 约束说明

无

#### 调用示例

```
// 实例化Conv3D Api
auto ascendcPlatform = platform_ascendc::PlatformAscendC(context->GetPlatformInfo());
Conv3dTilingApi::Conv3dTiling conv3dApiTiling(ascendcPlatform);
conv3dApiTiling.SetSingleOutputShape(singleCo, singleDo, singleM);
```
