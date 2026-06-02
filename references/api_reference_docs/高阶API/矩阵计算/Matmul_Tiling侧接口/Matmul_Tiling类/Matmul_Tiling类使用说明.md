# Matmul Tiling类使用说明

**页面ID:** atlasascendc_api_07_0671  
**来源:** https://www.hiascend.com/document/detail/zh/CANNCommunityEdition/850/API/ascendcopapi/atlasascendc_api_07_0671.html

---

Ascend C提供一组Matmul Tiling API，方便用户获取Matmul kernel计算时所需的Tiling参数。用户只需要传入A/B/C矩阵的Position位置、Format格式和DType数据类型等信息，调用API接口，即可获取到Init中TCubeTiling结构体中的相关参数。

Matmul Tiling API分为Matmul单核Tiling接口、多核Tiling接口和BatchMatmul Tiling接口，分别用于Matmul单核计算、多核计算和BatchMatmul计算场景。获取Tiling参数的流程如下：

1. 创建一个单核Tiling对象，或多核Tiling对象，或BatchMatmul Tiling对象。
2. 设置A、B、C、Bias的参数类型信息；M、N、Ka、Kb形状信息等。
3. 调用GetTiling接口，获取Tiling信息。

使用Matmul单核Tiling接口、多核Tiling接口和BatchMatmul Tiling接口获取Tiling参数的样例如下：

- Matmul单核Tiling

```
auto ascendcPlatform = platform_ascendc::PlatformAscendC(context->GetPlatformInfo());
matmul_tiling::MatmulApiTiling tiling(ascendcPlatform); 
// 设置A、B、C、Bias矩阵Position、Format、DType信息
tiling.SetAType(matmul_tiling::TPosition::GM, matmul_tiling::CubeFormat::ND, matmul_tiling::DataType::DT_FLOAT16);
tiling.SetBType(matmul_tiling::TPosition::GM, matmul_tiling::CubeFormat::ND, matmul_tiling::DataType::DT_FLOAT16);   
tiling.SetCType(matmul_tiling::TPosition::GM, matmul_tiling::CubeFormat::ND, matmul_tiling::DataType::DT_FLOAT);   
tiling.SetBiasType(matmul_tiling::TPosition::GM, matmul_tiling::CubeFormat::ND, matmul_tiling::DataType::DT_FLOAT);   
tiling.SetShape(1024, 1024, 1024); // 设置单核计算的M、N、K大小
tiling.SetOrgShape(1024, 1024, 1024); // 设置原始输入M、N、K大小，单核Tiling与SetShape一致。若Ka,Kb不等长时，设置tiling.SetOrgShape(1024, 1024, 1024, 1280)   
tiling.EnableBias(true); // 设置matmul计算包含bias
tiling.SetBufferSpace(-1, -1, -1);  // 设定允许使用的空间，缺省使用该AI处理器所有空间
optiling::TCubeTiling tilingData;   
int64_t ret = tiling.GetTiling(tilingData);    // if ret = -1, get tiling failed
```

- Matmul多核Tiling

```
auto ascendcPlatform = platform_ascendc::PlatformAscendC(context->GetPlatformInfo());
matmul_tiling::MultiCoreMatmulTiling tiling(ascendcPlatform); 
tiling.SetDim(1); // 设置参与计算的核数为1
tiling.SetAType(matmul_tiling::TPosition::GM, matmul_tiling::CubeFormat::ND, matmul_tiling::DataType::DT_FLOAT16);   
tiling.SetBType(matmul_tiling::TPosition::GM, matmul_tiling::CubeFormat::ND, matmul_tiling::DataType::DT_FLOAT16);   
tiling.SetCType(matmul_tiling::TPosition::GM, matmul_tiling::CubeFormat::ND, matmul_tiling::DataType::DT_FLOAT);   
tiling.SetBiasType(matmul_tiling::TPosition::GM, matmul_tiling::CubeFormat::ND, matmul_tiling::DataType::DT_FLOAT);   
tiling.SetShape(1024, 1024, 1024);   
tiling.SetSingleShape(1024, 1024, 1024);
tiling.SetOrgShape(1024, 1024, 1024); 
tiling.EnableBias(true);   
tiling.SetBufferSpace(-1, -1, -1);  // 设定允许使用的空间，缺省使用该AI处理器所有空间
optiling::TCubeTiling tilingData;   
int64_t ret = tiling.GetTiling(tilingData);    // if ret = -1, get tiling failed
```

- BatchMatmul Tiling

```
auto ascendcPlatform = platform_ascendc::PlatformAscendC(context->GetPlatformInfo());
matmul_tiling::BatchMatmulTiling bmmTiling(ascendcPlatform); 

bmmTiling.SetAType(matmul_tiling::TPosition::GM, matmul_tiling::CubeFormat::ND, matmul_tiling::DataType::DT_FLOAT16);   
bmmTiling.SetBType(matmul_tiling::TPosition::GM, matmul_tiling::CubeFormat::ND, matmul_tiling::DataType::DT_FLOAT16);   
bmmTiling.SetCType(matmul_tiling::TPosition::GM, matmul_tiling::CubeFormat::ND, matmul_tiling::DataType::DT_FLOAT);   
bmmTiling.SetBiasType(matmul_tiling::TPosition::GM, matmul_tiling::CubeFormat::ND, matmul_tiling::DataType::DT_FLOAT);   
bmmTiling.EnableBias(true);
bmmTiling.SetShape(64, 48, 32);   
bmmTiling.SetSingleShape(64, 48, 32);
bmmTiling.SetOrgShape(64, 48, 32); 
// Layout类型为NORMAL时,通过SetBatchInfoForNormal设置A、B、C矩阵的Layout轴信息
bmmTiling.SetBatchInfoForNormal(2, 2, 64, 48, 32);
// Layout类型为BSNGD、SBNGD、BNGS1S2时, 通过SetALayout、SetBLayout、SetCLayout设置A、B、C矩阵的Layout轴信息
// bmmTiling.SetALayout(3, 64, 2, 2, 32);
// bmmTiling.SetBLayout(3, 32, 2, 2, 48);
// bmmTiling.SetCLayout(3, 64, 2, 2, 48);
bmmTiling.SetBatchNum(2);
bmmTiling.SetBufferSpace(-1, -1, -1);  // 设定允许使用的空间，缺省使用该AI处理器所有空间
optiling::TCubeTiling tilingData;
int64_t ret = bmmTiling.GetTiling(tilingData);    // if ret = -1, get tiling failed
```

接口列表如下：

**表1 **MatmulApiTiling/MultiCoreMatmulTiling/BatchMatmulTiling共有接口列表

| 接口 | 功能 |
| --- | --- |
| SetAType | 设置A矩阵的位置，数据格式，数据类型，是否转置等信息。 |
| SetBType | 设置B矩阵的位置，数据格式，数据类型，是否转置等信息。 |
| SetCType | 设置C矩阵的位置，数据格式，数据类型等信息。 |
| SetDequantType | 设置反量化的模式。 |
| SetBiasType | 设置Bias的位置，数据格式，数据类型等信息。 |
| SetShape | 设置Matmul单次计算的形状singleM、singleN、singleK，单位为元素个数。 |
| SetOrgShape | 设置Matmul计算时的原始完整的形状M、N、Ka、Kb，单位为元素个数。 |
| SetALayout | 设置A矩阵的Layout轴信息。 |
| SetBLayout | 设置B矩阵的Layout轴信息。 |
| SetCLayout | 设置C矩阵的Layout轴信息。 |
| SetBatchInfoForNormal | 设置A/B矩阵的M/N/K轴信息，以及A/B矩阵各自的Batch数。 |
| SetBatchNum | 设置多Batch计算的最大Batch数。 |
| EnableBias | 设置Bias是否参与运算。 |
| SetBias | 设置Bias是否参与运算。建议使用EnableBias接口。 |
| SetFixSplit | 设置固定的baseM、baseN、baseK，单位为元素个数。 |
| SetBufferSpace | 设置Matmul计算时可用的L1/L0C/UB空间大小，单位为字节。 |
| SetTraverse | 设置遍历方式，M轴优先还是N轴优先。 |
| SetMadType | 设置是否使能HF32模式。**当前版本暂不支持。** |
| SetSplitRange | 设置baseM/baseN/baseK的最大值和最小值。 |
| SetMatmulConfigParams | 自定义设置MatmulConfig参数。 |
| SetDoubleBuffer | 设置A/B/C/Bias是否使能double buffer功能，以及是否需要做ND2NZ或者NZ2ND的转换。**该接口为预留接口，当前版本暂不支持。** |
| GetBaseM | 获取baseM值。 |
| GetBaseN | 获取baseN值。 |
| GetBaseK | 获取baseK值。 |
| GetTiling | 获取Tiling参数。 |

**表2 **MultiCoreMatmulTiling其他接口

| 接口 | 功能 |
| --- | --- |
| SetDim | 设置多核Matmul时，可以参与运算的核数。 |
| SetSingleRange | 设置singleCoreM/singleCoreN/singleCoreK的最大值与最小值，单位为元素个数。 |
| SetSingleShape | 设置Matmul单核计算的形状singleCoreM、singleCoreN、singleCoreK，单位为元素个数。 |
| GetSingleShape | 获取计算后的singleCoreM/singleCoreN/singleCoreK。 |
| SetAlignSplit | 设置多核切分时singleCoreM/singleCoreN/singleCoreK的对齐值 |
| GetCoreNum | 获得多核切分后， 使用的blockDim。 |
| SetSplitK | 多核场景，使能切K轴。建议使用EnableMultiCoreSplitK接口。 |
| EnableMultiCoreSplitK | 多核场景，使能切K轴。 |

**表3 **BatchMatmulTiling其他接口

| 接口 | 功能 |
| --- | --- |
| GetCoreNum | 获得多核切分后， 使用的blockDim。 |

#### 需要包含的头文件

- Matmul单核Tiling

```
#include "lib/matmul/matmul_tiling.h"
```

- Matmul多核Tiling

```
#include "lib/matmul/bmm_tiling.h"
```

- BatchMatmul Tiling

```
#include "lib/matmul/bmm_tiling.h"
```
