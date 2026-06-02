# SetSplitK

**页面ID:** atlasascendc_api_07_0706  
**来源:** https://www.hiascend.com/document/detail/zh/CANNCommunityEdition/850/API/ascendcopapi/atlasascendc_api_07_0706.html

---

#### 功能说明

EnableMultiCoreSplitK接口功能与该接口相同，建议使用EnableMultiCoreSplitK。

多核场景，通过该接口使能切K轴。不调用该接口的情况下，默认不切K轴。在GetTiling接口调用前使用。

#### 函数原型

```
void SetSplitK(bool flag)
```

#### 参数说明

**表1 **参数说明

| 参数名 | 输入/输出 | 描述 |
| --- | --- | --- |
| flag | 输入 | 是否使能切K轴。 - true：使能切K轴- false：不使能切K轴 |

#### 约束说明

- 如果在算子中使用该接口，获取C矩阵结果时仅支持输出到Global Memory。
- 如果在算子中使用该接口，需在Kernel侧代码中首次将C矩阵分片的结果写入Global Memory之前，先清零Global Memory，随后在获取C矩阵分片的结果时，再开启AtomicAdd累加。如果不预先清零Global Memory，可能会因为累加Global Memory中原始的无效数据而产生精度问题。

#### 调用示例

```
auto ascendcPlatform = platform_ascendc::PlatformAscendC(context->GetPlatformInfo())
matmul_tiling::MultiCoreMatmulTiling tiling(ascendcPlatform);  
tiling->SetDim(useCoreNums);
tiling->SetAType(matmul_tiling::TPosition::GM, matmul_tiling::CubeFormat::ND, matmul_tiling::DataType::DT_FLOAT16);
tiling->SetBType(matmul_tiling::TPosition::GM, matmul_tiling::CubeFormat::ND, matmul_tiling::DataType::DT_FLOAT16);
tiling->SetCType(matmul_tiling::TPosition::GM, matmul_tiling::CubeFormat::ND, matmul_tiling::DataType::DT_FLOAT);
tiling->SetBiasType(matmul_tiling::TPosition::GM, matmul_tiling::CubeFormat::ND, matmul_tiling::DataType::DT_FLOAT);
tiling->SetShape(M, N, K);
tiling->SetOrgShape(M, N, K);
tiling->SetBias(true);
tiling->SetBufferSpace(-1, -1, -1);
tiling->SetSplitK(true);

optiling::TCubeTiling tilingData;
int ret = tiling.GetTiling(tilingData);
```
