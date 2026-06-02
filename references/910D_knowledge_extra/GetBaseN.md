# GetBaseN<a name="ZH-CN_TOPIC_0000002554343597"></a>

## 功能说明<a name="section618mcpsimp"></a>

获取Tiling计算得到的baseN值。baseN参数的说明请参考[表1](TCubeTiling结构体.md#table1563162142915)。

## 函数原型<a name="section620mcpsimp"></a>

```
int32_t GetBaseN() const
```

## 参数说明<a name="section622mcpsimp"></a>

无

## 返回值说明<a name="section640mcpsimp"></a>

返回值为Tiling计算得到的baseN值。

## 约束说明<a name="section633mcpsimp"></a>

使用创建的Tiling对象调用该接口，且需在完成Tiling计算（[GetTiling](GetTiling.md)）后调用。

## 调用示例<a name="section1665082013318"></a>

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
int baseN = tiling.GetBaseN();  // 获取Tiling计算得到的baseN
```

