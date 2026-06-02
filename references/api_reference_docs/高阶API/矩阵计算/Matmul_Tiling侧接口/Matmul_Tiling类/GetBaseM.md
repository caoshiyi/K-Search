# GetBaseM

**页面ID:** atlasascendc_api_07_0689  
**来源:** https://www.hiascend.com/document/detail/zh/CANNCommunityEdition/850/API/ascendcopapi/atlasascendc_api_07_0689.html

---

#### 功能说明

获取Tiling计算得到的baseM值。baseM参数的说明请参考表1。

#### 函数原型

```
int32_t GetBaseM() const
```

#### 参数说明

无

#### 返回值说明

返回值为Tiling计算得到的baseM值。

#### 约束说明

使用创建的Tiling对象调用该接口，且需在完成Tiling计算（GetTiling）后调用。

#### 调用示例

```
auto ascendcPlatform = platform_ascendc::PlatformAscendC(context->GetPlatformInfo());
matmul_tiling::MatmulApiTiling tiling(ascendcPlatform); 
tiling.SetAType(matmul_tiling::TPosition::GM, matmul_tiling::CubeFormat::ND, matmul_tiling::DataType::DT_FLOAT16); 
tiling.SetBType(matmul_tiling::TPosition::GM, matmul_tiling::CubeFormat::ND, matmul_tiling::DataType::DT_FLOAT16);   
tiling.SetCType(matmul_tiling::TPosition::GM, matmul_tiling::CubeFormat::ND, matmul_tiling::DataType::DT_FLOAT);   
tiling.SetBiasType(matmul_tiling::TPosition::GM, matmul_tiling::CubeFormat::ND, matmul_tiling::DataType::DT_FLOAT);   
tiling.SetShape(1024, 1024, 1024);   
tiling.SetOrgShape(1024, 1024, 1024);  
tiling.SetBias(true);   
tiling.SetBufferSpace(-1, -1, -1);

optiling::TCubeTiling tilingData;   
int ret = tiling.GetTiling(tilingData);
int baseM = tiling.GetBaseM();  // 获取Tiling计算得到的baseM
```
