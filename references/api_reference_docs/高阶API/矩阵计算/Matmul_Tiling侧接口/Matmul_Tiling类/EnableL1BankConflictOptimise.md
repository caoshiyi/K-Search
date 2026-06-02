# EnableL1BankConflictOptimise

**页面ID:** atlasascendc_api_07_10263  
**来源:** https://www.hiascend.com/document/detail/zh/CANNCommunityEdition/850/API/ascendcopapi/atlasascendc_api_07_10263.html

---

#### 功能说明

根据GetTiling接口计算出的Tiling参数，获取是否可以开启L1 Bank冲突优化功能。若可以开启该功能，则与TilingKey机制配合使用，通过增加TilingKey，关联Host侧与Kernel侧实现，并在Kernel侧增加代码实现分支，将MatmulConfig中的enableL1BankConflictOptimise设置为true，即可优化L1上的Bank冲突。

#### 函数原型

```
bool EnableL1BankConflictOptimise()
```

#### 参数说明

无

#### 返回值说明

- false：Kernel侧不能开启L1 Bank冲突优化。
- true：Kernel侧可以开启L1 Bank冲突优化。

#### 约束说明

使用创建的Tiling对象调用本接口，且需在完成Tiling计算（GetTiling）后调用本接口。

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
// Kernel侧是否可以开启L1 Bank冲突优化，可与TilingKey机制结合使用
bool enableL1BankConflictOptimise = tiling.EnableL1BankConflictOptimise();
```
