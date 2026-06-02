# SetFixSplit

**页面ID:** atlasascendc_api_07_0681  
**来源:** https://www.hiascend.com/document/detail/zh/CANNCommunityEdition/850/API/ascendcopapi/atlasascendc_api_07_0681.html

---

#### 功能说明

设置固定的baseM、baseN、baseK，单位为元素个数。

#### 函数原型

```
int32_t SetFixSplit(int32_t baseMIn = -1, int32_t baseNIn = -1, int32_t baseKIn = -1)
```

#### 参数说明

**表1 **参数说明

| 参数名 | 输入/输出 | 描述 |
| --- | --- | --- |
| baseMIn | 输入 | 设置固定的baseM，默认值为-1，表示不设置固定baseM，由tiling函数进行计算。 |
| baseNIn | 输入 | 设置固定的baseN，默认值为-1，表示不设置固定baseN，由tiling函数进行计算。 |
| baseKIn | 输入 | 当前仅支持取值为-1，暂不支持设置其它值。 |

#### 返回值说明

-1表示设置失败；0表示设置成功。

#### 约束说明

- baseM*baseN个输出元素所占的存储空间大小不能超过L0C Buffer大小，即baseM * baseN * sizeof(C_TYPE) <= L0CSize。
- baseM需要小于等于singleM按16个元素向上对齐后的值（如ceil(singleM/16)*16），baseN需要小于等于singleN以C0_size个元素向上对齐的值，其中singleM为单核内M轴长度，singleN为单核内N轴长度，half/bfloat16_t数据类型的C0_size为16，float数据类型的C0_size为8，int8_t数据类型的C0_size为32，int4b_t数据类型的C0_size为64。例如singleM=12，则baseM需要小于等于16，同时baseM需要满足分形对齐的要求，所以baseM只能取16；如果baseM取其他超过16的值，获取Tiling将失败。

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
tiling.SetFixSplit(16, 16, -1);  // 设置固定的baseM、baseN
tiling.SetBufferSpace(-1, -1, -1);
optiling::TCubeTiling tilingData;   
int ret = tiling.GetTiling(tilingData);
```
