# EnableMultiCoreSplitK

**页面ID:** atlasascendc_api_07_0682  
**来源:** https://www.hiascend.com/document/detail/zh/CANNCommunityEdition/850/API/ascendcopapi/atlasascendc_api_07_0682.html

---

#### 功能说明

多核场景，通过该接口使能切K轴。不调用该接口的情况下，默认不切K轴。在GetTiling接口调用前使用。

#### 函数原型

```
void EnableMultiCoreSplitK(bool flag)
```

#### 参数说明

**表1 **参数说明

| 参数名 | 输入/输出 | 描述 |
| --- | --- | --- |
| flag | 输入 | 是否使能切K轴。 - true：使能切K轴- false：不使能切K轴 |

#### 约束说明

- 在算子中使用该接口时，获取C矩阵结果时仅支持输出到Global Memory。
- 在算子中使用该接口时，需在Kernel侧代码中首次将C矩阵分片的结果写入Global Memory之前，先清零Global Memory，随后在获取C矩阵分片的结果时，再开启AtomicAdd累加。如果不预先清零Global Memory，可能会因为累加Global Memory中原始的无效数据而产生精度问题。
- 在算子中使用该接口时，不支持Bias参与矩阵乘计算。

#### 调用示例

完整的算子样例请参考[多核切K场景的算子样例](https://gitee.com/ascend/ascendc-api-adv/tree/master/examples/matrix/matmul_splitk)。

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
tiling->SetBias(false);
tiling->SetBufferSpace(-1, -1, -1);
tiling->EnableMultiCoreSplitK(true);  // 使能切K轴

optiling::TCubeTiling tilingData;
int ret = tiling.GetTiling(tilingData);
```
