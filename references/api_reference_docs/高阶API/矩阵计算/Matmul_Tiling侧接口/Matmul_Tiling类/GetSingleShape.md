# GetSingleShape

**页面ID:** atlasascendc_api_07_0695  
**来源:** https://www.hiascend.com/document/detail/zh/CANNCommunityEdition/850/API/ascendcopapi/atlasascendc_api_07_0695.html

---

#### 功能说明

获取计算后的singleCoreM/singleCoreN/singleCoreK。

#### 函数原型

```
int32_t GetSingleShape(int32_t &shapeM, int32_t &shapeN, int32_t &shapeK)
```

#### 参数说明

**表1 **参数说明

| 参数名 | 输入/输出 | 描述 |
| --- | --- | --- |
| shapeM | 输出 | 获取多核Tiling计算得到的singleCoreM值 |
| shapeN | 输出 | 获取多核Tiling计算得到的singleCoreN值 |
| shapeK | 输出 | 获取多核Tiling计算得到的singleCoreK值 |

#### 返回值说明

-1表示设置失败； 0表示设置成功。

#### 约束说明

使用创建的Tiling对象调用该接口，且需在完成Tiling计算（GetTiling）后调用。

#### 调用示例

```
auto ascendcPlatform = platform_ascendc::PlatformAscendC(context->GetPlatformInfo());
matmul_tiling::MultiCoreMatmulTiling tiling(ascendcPlatform); 
tiling.SetDim(1);
tiling.SetAType(matmul_tiling::TPosition::GM, matmul_tiling::CubeFormat::ND, matmul_tiling::DataType::DT_FLOAT16);   
tiling.SetBType(matmul_tiling::TPosition::GM, matmul_tiling::CubeFormat::ND, matmul_tiling::DataType::DT_FLOAT16);   
tiling.SetCType(matmul_tiling::TPosition::GM, matmul_tiling::CubeFormat::ND, matmul_tiling::DataType::DT_FLOAT);   
tiling.SetBiasType(matmul_tiling::TPosition::GM, matmul_tiling::CubeFormat::ND, matmul_tiling::DataType::DT_FLOAT);   
tiling.SetShape(1024, 1024, 1024);   
tiling.SetSingleShape(1024, 1024, 1024);
tiling.SetOrgShape(1024, 1024, 1024);
tiling.SetBias(true);   
tiling.SetBufferSpace(-1, -1, -1);
optiling::TCubeTiling tilingData;   
int ret = tiling.GetTiling(tilingData);

// 获取计算后的singleCoreM/singleCoreN/singleCoreK
int32_t singleM, singleN, singleK;
int ret1 = tiling.GetSingleShape(singleM, singleN, singleK);
```
