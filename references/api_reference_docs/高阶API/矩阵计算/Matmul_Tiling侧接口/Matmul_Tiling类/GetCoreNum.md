# GetCoreNum

**页面ID:** atlasascendc_api_07_0696  
**来源:** https://www.hiascend.com/document/detail/zh/CANNCommunityEdition/850/API/ascendcopapi/atlasascendc_api_07_0696.html

---

#### 功能说明

获得多核切分所使用的BlockDim参数。

#### 函数原型

- MultiCoreMatmulTiling类

```
int32_t GetCoreNum(int32_t &dim, int32_t &mDim, int32_t &nDim)
```

- BatchMatmulTiling类

```
int32_t GetCoreNum(int32_t &dim, int32_t &mDim, int32_t &nDim, int32_t &batchCoreM, int32_t &batchCoreN)
```

#### 参数说明

**表1 **参数说明

| 参数名 | 输入/输出 | 描述 |
| --- | --- | --- |
| dim | 输出 | 获取计算时所需要的核数， dim = mDim * nDim |
| mDim | 输出 | 获取计算时M方向所需要的核数 |
| nDim | 输出 | 获取计算时N方向所需要的核数 |
| batchCoreM | 输出 | 获取计算时batch M方向所需要的核数，仅BatchMatmulTiling类支持 |
| batchCoreN | 输出 | 获取计算时batch N方向所需要的核数，仅BatchMatmulTiling类支持 |

#### 返回值说明

-1表示获取失败； 0表示获取成功。

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

// 获得多核切分后，使用的BlockDim
int32_t dim, mDim, nDim;
int ret1 = tiling.GetCoreNum(dim, mDim, nDim);
```
