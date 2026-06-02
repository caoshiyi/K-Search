# Matmul Tiling类构造函数

**页面ID:** atlasascendc_api_07_0672  
**来源:** https://www.hiascend.com/document/detail/zh/CANNCommunityEdition/850/API/ascendcopapi/atlasascendc_api_07_0672.html

---

#### 功能说明

用于创建一个Matmul单核Tiling对象，或者多核Tiling对象，或者BatchMatmul Tiling对象。

#### 函数原型

- 带参构造函数，需要传入硬件平台信息，推荐使用这类构造函数来获得更好的兼容性。

  - 使用PlatformAscendC类传入信息

```
explicit MatmulApiTiling(const platform_ascendc::PlatformAscendC& ascendcPlatform)
```

```
explicit MultiCoreMatmulTiling(const platform_ascendc::PlatformAscendC& ascendcPlatform)
```

```
explicit BatchMatmulTiling(const platform_ascendc::PlatformAscendC &ascendcPlatform)
```

  - 使用PlatformInfo传入信息当platform_ascendc::PlatformAscendC无法在Tiling运行时获取时，需要用户自行构造PlatformInfo结构体，透传给MatmulApiTiling构造函数。

```
explicit MatmulApiTiling(const PlatformInfo& platform)
```

```
explicit MultiCoreMatmulTiling(const PlatformInfo &platform)
```

- 无参构造函数

```
MatmulApiTiling()
```

```
MultiCoreMatmulTiling()
```

```
BatchMatmulTiling()
```

无参构造函数只支持如下产品型号：

Atlas A2训练系列产品/Atlas 800I A2推理产品

Atlas A3 训练系列产品

- 基类构造函数

MatmulApiTiling、MultiCoreMatmulTiling和BatchMatmulTiling都继承自基类MatmulApiTilingBase，其构造函数如下：

```
MatmulApiTilingBase()
```

```
explicit MatmulApiTilingBase(const platform_ascendc::PlatformAscendC& ascendcPlatform)
```

```
explicit MatmulApiTilingBase(const PlatformInfo& platform)
```

#### 参数说明

**表1 **参数说明

| 参数名 | 输入/输出 | 描述 |
| --- | --- | --- |
| ascendcPlatform | 输入 | 传入硬件平台的信息，PlatformAscendC定义请参见构造及析构函数。 |
| 传入硬件版本以及AI Core中各个硬件单元提供的内存大小。PlatformInfo构造时通过构造及析构函数获取。 PlatformInfo结构定义如下，socVersion通过GetSocVersion获取并透传，各类硬件存储空间大小通过GetCoreMemSize获取并透传。 ``` struct PlatformInfo {     platform_ascendc::SocVersion socVersion;     uint64_t l1Size = 0;     uint64_t l0CSize = 0;     uint64_t ubSize = 0;     uint64_t l0ASize = 0;     uint64_t l0BSize = 0; }; ```  不推荐通过直接填值构造PlatformInfo的方式调用构造函数，例如PlatformInfo(socVersion, 1024, 1024, ..); |  |  |

在实现Host侧的Tiling函数时，platform_ascendc::PlatformAscendC用于获取一些硬件平台的信息，来支撑Tiling的计算，比如获取硬件平台的核数等信息。PlatformAscendC类提供获取这些平台信息的功能。

和platform_ascendc::PlatformAscendC不同的是，PlatformInfo则用于获取芯片版本、AI Core中各个硬件单元提供的内存大小等只针对单个AI Core的信息。

#### 约束说明

无

#### 使用样例

- 无参构造函数

```
// 单核Tiling
matmul_tiling::MatmulApiTiling tiling;
tiling.SetAType(matmul_tiling::TPosition::GM, matmul_tiling::CubeFormat::ND, matmul_tiling::DataType::DT_FLOAT16);
...
optiling::TCubeTiling tilingData;   
int ret = tiling.GetTiling(tilingData);    // if ret = -1, gen tiling failed

// 多核Tiling
matmul_tiling::MultiCoreMatmulTiling tiling;   
tiling.SetDim(1);
...
optiling::TCubeTiling tilingData;   
int ret = tiling.GetTiling(tilingData);    // if ret = -1, gen tiling failed 

// BatchMatmul Tiling
matmul_tiling::BatchMatmulTiling bmmTiling;
bmmTiling.SetAType(matmul_tiling::TPosition::GM, matmul_tiling::CubeFormat::ND, matmul_tiling::DataType::DT_FLOAT16);
...
optiling::TCubeTiling tilingData;   
int ret = bmmTiling.GetTiling(tilingData);    // if ret = -1, gen tiling failed
```

- 带参构造函数

```
// 单核Tiling
auto ascendcPlatform = platform_ascendc::PlatformAscendC(context->GetPlatformInfo());
matmul_tiling::MatmulApiTiling tiling(ascendcPlatform); 
tiling.SetAType(matmul_tiling::TPosition::GM, matmul_tiling::CubeFormat::ND, matmul_tiling::DataType::DT_FLOAT16);   
...
optiling::TCubeTiling tilingData;   
int ret = tiling.GetTiling(tilingData);    // if ret = -1, gen tiling failed

// 多核Tiling
auto ascendcPlatform = platform_ascendc::PlatformAscendC(context->GetPlatformInfo());
matmul_tiling::MultiCoreMatmulTiling tiling(ascendcPlatform); 
tiling.SetDim(1);
...
optiling::TCubeTiling tilingData;   
int ret = tiling.GetTiling(tilingData);    // if ret = -1, gen tiling failed 

// BatchMatmul Tiling
auto ascendcPlatform = platform_ascendc::PlatformAscendC(context->GetPlatformInfo());
matmul_tiling::BatchMatmulTiling bmmTiling(ascendcPlatform); 
bmmTiling.SetAType(matmul_tiling::TPosition::GM, matmul_tiling::CubeFormat::ND, matmul_tiling::DataType::DT_FLOAT16);   
...
optiling::TCubeTiling tilingData;
int ret = tiling.GetTiling(tilingData);    // if ret = -1, gen tiling failed
```
