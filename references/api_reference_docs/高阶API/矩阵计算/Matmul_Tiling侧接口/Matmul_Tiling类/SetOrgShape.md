# SetOrgShape

**页面ID:** atlasascendc_api_07_0680  
**来源:** https://www.hiascend.com/document/detail/zh/CANNCommunityEdition/850/API/ascendcopapi/atlasascendc_api_07_0680.html

---

#### 功能说明

设置Matmul计算时的原始完整的形状M、N、K或Ka/Kb，单位均为元素个数。

#### 函数原型

```
int32_t SetOrgShape(int32_t orgMIn, int32_t orgNIn, int32_t orgKIn)
```

```
int32_t SetOrgShape(int32_t orgMIn, int32_t orgNIn, int32_t orgKaIn, int32_t orgKbIn)
```

#### 参数说明

**表1 **参数说明

| 参数名 | 输入/输出 | 描述 |
| --- | --- | --- |
| orgMIn | 输入 | 设置原始完整的形状M大小，单位为元素。 |
| orgNIn | 输入 | 设置原始完整的形状N大小，单位为元素。 |
| orgKIn | 输入 | 设置原始完整的形状K大小，单位为元素。原始完整形状Ka=Kb时可设置。 |
| orgKaIn | 输入 | 设置矩阵A原始完整的形状Ka大小，单位为元素。 |
| orgKbIn | 输入 | 设置矩阵B原始完整的形状Kb大小，单位为元素。 |

#### 返回值说明

-1表示设置失败； 0表示设置成功。

#### 约束说明

参数orgKaIn和orgKbIn可以不相等，即原始矩阵形状Ka和Kb不相等，并不是实际Matmul计算时的K，此参数只用于辅助Matmul API搬运时的偏移计算。

#### 调用示例

```
auto ascendcPlatform = platform_ascendc::PlatformAscendC(context->GetPlatformInfo());
matmul_tiling::MatmulApiTiling tiling(ascendcPlatform); 
tiling.SetAType(matmul_tiling::TPosition::GM, matmul_tiling::CubeFormat::ND, matmul_tiling::DataType::DT_FLOAT16); 
tiling.SetBType(matmul_tiling::TPosition::GM, matmul_tiling::CubeFormat::ND, matmul_tiling::DataType::DT_FLOAT16);  
tiling.SetCType(matmul_tiling::TPosition::GM, matmul_tiling::CubeFormat::ND, matmul_tiling::DataType::DT_FLOAT);
tiling.SetBiasType(matmul_tiling::TPosition::GM, matmul_tiling::CubeFormat::ND, matmul_tiling::DataType::DT_FLOAT);   
tiling.SetShape(1024, 1024, 1024);
tiling.SetOrgShape(1024, 1024, 1024);  // 设置原始完整的形状   
tiling.SetBias(true);   
tiling.SetBufferSpace(-1, -1, -1);
optiling::TCubeTiling tilingData;   
int ret = tiling.GetTiling(tilingData);
```
