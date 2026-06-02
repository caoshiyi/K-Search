# Conv3D Tiling构造函数

**页面ID:** atlasascendc_api_07_10081  
**来源:** https://www.hiascend.com/document/detail/zh/CANNCommunityEdition/850/API/ascendcopapi/atlasascendc_api_07_10081.html

---

#### 功能说明

用于创建一个Conv3D单核Tiling对象。

#### 函数原型

- 带参构造函数，需要传入硬件平台信息，推荐使用这类构造函数来获得更好的兼容性。

  - 使用PlatformAscendC类传入信息

```
explicit Conv3dTiling(const platform_ascendc::PlatformAscendC& ascendcPlatform)
```

  - 使用PlatformInfo传入信息当platform_ascendc::PlatformAscendC无法在Tiling运行时获取时，需要用户自己构造PlatformInfo结构体，透传给Conv3dTiling构造函数。

```
explicit Conv3dTiling(const PlatformInfo& platform)
```

- 基类构造函数Conv3dTiling继承自基类Conv3dTilingBase，其构造函数如下：

```
explicit Conv3dTilingBase(const platform_ascendc::PlatformAscendC& ascendcPlatform)
```

```
explicit Conv3dTilingBase(const PlatformInfo& platform)
```

#### 参数说明

**表1 **参数说明

| 参数名 | 输入/输出 | 描述 |
| --- | --- | --- |
| ascendcPlatform | 输入 | 传入硬件平台的信息，PlatformAscendC定义请参见构造及析构函数。 |
| 传入硬件版本以及AI Core中各个硬件单元提供的内存大小。PlatformInfo构造时通过构造及析构函数获取。 PlatformInfo结构定义如下，socVersion通过GetSocVersion获取并透传，各类硬件存储空间大小通过GetCoreMemSize获取并透传。 ``` struct PlatformInfo {     platform_ascendc::SocVersion socVersion;     uint64_t l1Size = 0;     uint64_t l0CSize = 0;     uint64_t ubSize = 0;     uint64_t l0ASize = 0;     uint64_t l0BSize = 0;     uint64_t btSize = 0;     uint64_t fbSize = 0; }; ``` |  |  |

#### 约束说明

无

#### 调用示例

```
// 实例化Conv3d Api
auto ascendcPlatform = platform_ascendc::PlatformAscendC(context->GetPlatformInfo());
Conv3dTilingApi::Conv3dTiling conv3dApiTiling(ascendcPlatform);
conv3dApiTiling.SetGroups(groups);
conv3dApiTiling.SetOrgWeightShape(cout, kd, kh, kw);
...
conv3dApiTiling.GetTiling(conv3dCustomTilingData.conv3dApiTilingData);
```
