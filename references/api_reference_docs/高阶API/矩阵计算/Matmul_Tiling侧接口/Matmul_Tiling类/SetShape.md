# SetShape

**页面ID:** atlasascendc_api_07_0679  
**来源:** https://www.hiascend.com/document/detail/zh/CANNCommunityEdition/850/API/ascendcopapi/atlasascendc_api_07_0679.html

---

#### 功能说明

设置Matmul计算的形状m、n、k，该形状可以为原始完整矩阵或其局部矩阵，单位为元素。该形状的矩阵乘可以由单核或多核计算完成。

使用本接口时，有两种参数传入方式：

- 传入Matmul计算的形状m、n、k，调用GetTiling接口时，按照m、n、k计算并返回Tiling参数。
- m、n、k中任意一个或多个参数位置传入-1，调用GetTiling接口时，该位置取SetOrgShape接口中设置的原始形状M、N、K或Ka/Kb，然后由接口内部计算最优Tiling参数。如下图所示，原始A矩阵的K方向最后一列为不参与计算的脏数据，在SetOrgShape接口中设置包含该列数据的原始形状，在本接口中设置Matmul计算的K方向大小，同时参数m、n设置为-1表示按照原始形状M、N计算Tiling。

**图1 **参数传入-1的场景示意图
<!-- img2text -->
```text
                           SetOrgShape(1024,1024,256,255);
                           SetShape(-1,-1,255);

A矩阵
                Ka=256     1
         ┌────────────────┬──┐
M=1024   │                │  │
         │                │不 │
         │                │参 │
         │                │与 │
         │                │计 │
         │                │算 │
         │                │数 │
         │                │据 │
         │                │  │
         └────────────────┴──┘

B矩阵
                       N=1024
                ┌──────────────────────┐
Kb=255          │                      │
                │                      │
                │                      │
                │                      │
                └──────────────────────┘
```

#### 函数原型

```
int32_t SetShape(int32_t m, int32_t n, int32_t k)
```

#### 参数说明

**表1 **参数说明

| 参数名 | 输入/输出 | 描述 |
| --- | --- | --- |
| m | 输入 | 设置Matmul计算的M方向大小，单位为元素。 |
| n | 输入 | 设置Matmul计算的N方向大小，单位为元素。 |
| k | 输入 | 设置Matmul计算的K方向大小，单位为元素。 |

#### 返回值说明

-1表示设置失败；0表示设置成功。

#### 约束说明

无

#### 调用示例

```
auto ascendcPlatform = platform_ascendc::PlatformAscendC(context->GetPlatformInfo());
matmul_tiling::MatmulApiTiling tiling(ascendcPlatform); 
tiling.SetAType(matmul_tiling::TPosition::GM, matmul_tiling::CubeFormat::ND, matmul_tiling::DataType::DT_FLOAT16); 
tiling.SetBType(matmul_tiling::TPosition::GM, matmul_tiling::CubeFormat::ND, matmul_tiling::DataType::DT_FLOAT16);  
tiling.SetCType(matmul_tiling::TPosition::GM, matmul_tiling::CubeFormat::ND, matmul_tiling::DataType::DT_FLOAT);
tiling.SetBiasType(matmul_tiling::TPosition::GM, matmul_tiling::CubeFormat::ND, matmul_tiling::DataType::DT_FLOAT);   
tiling.SetShape(1024, 1024, 1024);  // 设置Matmul计算的形状
tiling.SetOrgShape(1024, 1024, 1024);   
tiling.SetBias(true);   
tiling.SetBufferSpace(-1, -1, -1);
optiling::TCubeTiling tilingData;   
int ret = tiling.GetTiling(tilingData);
```
