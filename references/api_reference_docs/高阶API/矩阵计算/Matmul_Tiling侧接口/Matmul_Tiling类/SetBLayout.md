# SetBLayout

**页面ID:** atlasascendc_api_07_0700  
**来源:** https://www.hiascend.com/document/detail/zh/CANNCommunityEdition/850/API/ascendcopapi/atlasascendc_api_07_0700.html

---

#### 功能说明

设置B矩阵的Layout轴信息，包括B、S、N、G、D轴。对于BSNGD、SBNGD、BNGS1S2 Layout格式，调用IterateBatch接口之前，需要在Host侧Tiling实现中通过本接口设置B矩阵的Layout轴信息。

#### 函数原型

```
int32_t SetBLayout(int32_t b, int32_t s, int32_t n, int32_t g, int32_t d)
```

#### 参数说明

**表1 **参数说明

| 参数名 | 输入/输出 | 描述 |
| --- | --- | --- |
| b | 输入 | B矩阵Layout的B轴信息 |
| s | 输入 | B矩阵Layout的S轴信息 |
| n | 输入 | B矩阵Layout的N轴信息 |
| g | 输入 | B矩阵Layout的G轴信息 |
| d | 输入 | B矩阵Layout的D轴信息 |

#### 返回值说明

-1表示设置失败； 0表示设置成功。

#### 约束说明

对于BSNGD、SBNGD、BNGS1S2 Layout格式，调用IterateBatch接口之前，需要在Host侧Tiling实现中通过本接口设置B矩阵的Layout轴信息。

#### 调用示例

```
auto ascendcPlatform = platform_ascendc::PlatformAscendC(context->GetPlatformInfo());
matmul_tiling::MultiCoreMatmulTiling tiling(ascendcPlatform);   
int32_t M = 32;
int32_t N = 256;
int32_t K = 64;
tiling->SetDim(1);
tiling->SetAType(matmul_tiling::TPosition::GM, matmul_tiling::CubeFormat::ND, matmul_tiling::DataType::DT_FLOAT16);
tiling->SetBType(matmul_tiling::TPosition::GM, matmul_tiling::CubeFormat::ND, matmul_tiling::DataType::DT_FLOAT16);
tiling->SetCType(matmul_tiling::TPosition::GM, matmul_tiling::CubeFormat::ND, matmul_tiling::DataType::DT_FLOAT);
tiling->SetBiasType(matmul_tiling::TPosition::GM, matmul_tiling::CubeFormat::ND, matmul_tiling::DataType::DT_FLOAT);
tiling->SetShape(M, N, K);
tiling->SetOrgShape(M, N, K);
tiling->SetBias(true);
tiling->SetBufferSpace(-1, -1, -1);

constexpr int32_t A_BNUM = 2;
constexpr int32_t A_SNUM = 32;
constexpr int32_t A_GNUM = 3;
constexpr int32_t A_DNUM = 64;
constexpr int32_t B_BNUM = 2;
constexpr int32_t B_SNUM = 256;
constexpr int32_t B_GNUM = 3;
constexpr int32_t B_DNUM = 64;
constexpr int32_t C_BNUM = 2;
constexpr int32_t C_SNUM = 32;
constexpr int32_t C_GNUM = 3;
constexpr int32_t C_DNUM = 256;
constexpr int32_t BATCH_NUM = 3;
tiling->SetALayout(A_BNUM, A_SNUM, 1, A_GNUM, A_DNUM);
tiling->SetBLayout(B_BNUM, B_SNUM, 1, B_GNUM, B_DNUM);  // 设置B矩阵排布
tiling->SetCLayout(C_BNUM, C_SNUM, 1, C_GNUM, C_DNUM);
tiling->SetBatchNum(BATCH_NUM);
tiling->SetBufferSpace(-1, -1, -1);

optiling::TCubeTiling tilingData;
int ret = tiling.GetTiling(tilingData);
```
