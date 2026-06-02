# GetTiling

**页面ID:** atlasascendc_api_07_10083  
**来源:** https://www.hiascend.com/document/detail/zh/CANNCommunityEdition/850/API/ascendcopapi/atlasascendc_api_07_10083.html

---

#### 功能说明

获取Tiling参数。

#### 函数原型

```
int64_t GetTiling(optiling::TConv3DApiTiling& tiling)
```

```
int64_t GetTiling(AscendC::tiling::TConv3DApiTiling& tiling)
```

#### 参数说明

| 参数名 | 输入/输出 | 描述 |
| --- | --- | --- |
| tiling | 输出 | Conv3D的Tiling结构体，用于存储最终的Tiling结果。TConv3DApiTiling结构介绍请参考TConv3DApiTiling结构体。 |

#### 返回值说明

如果返回值不为-1，则代表Tiling计算成功，用户可以使用该Tiling结构的值。如果返回值为-1，则代表Tiling计算失败，该Tiling结果无法使用。

#### 约束说明

调用GetTiling接口前必须先调用SetOrgInputShape、SetOrgWeightShape、SetSingleWeightShape、SetSingleOutputShape。

#### 调用示例

```
// 实例化Conv3d Api
auto ascendcPlatform = platform_ascendc::PlatformAscendC(context->GetPlatformInfo());
Conv3dTilingApi::Conv3dTiling conv3dApiTiling(ascendcPlatform );
conv3dApiTiling.SetOrgInputShape(orgCi, orgDi, orgHi, orgWi);
conv3dApiTiling.SetOrgWeightShape(cout, kd, kh, kw);
conv3dApiTiling.SetSingleWeightShape(singleCi, singleKd, singleKh, singleKw);
conv3dApiTiling.SetSingleOutputShape(singleCo, singleDo, singleM);
...
conv3dApiTiling.GetTiling(tilingData.conv3ApiTilingData);
```
