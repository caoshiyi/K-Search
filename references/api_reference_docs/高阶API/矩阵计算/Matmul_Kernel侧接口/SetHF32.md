# SetHF32

**页面ID:** atlasascendc_api_07_0646  
**来源:** https://www.hiascend.com/document/detail/zh/CANNCommunityEdition/850/API/ascendcopapi/atlasascendc_api_07_0646.html

---

#### 产品支持情况

| 产品 | 是否支持 |
| --- | --- |
| Atlas A3 训练系列产品/Atlas A3 推理系列产品 | √ |
| Atlas A2 训练系列产品/Atlas A2 推理系列产品 | √ |
| Atlas 200I/500 A2 推理产品 | x |
| Atlas 推理系列产品AI Core | x |
| Atlas 推理系列产品Vector Core | x |
| Atlas 训练系列产品 | x |

#### 功能说明

在纯Cube模式（只有矩阵计算）下，设置是否使能HF32（矩阵乘计算时可采用的数据类型）模式。使能后，在矩阵乘计算时，float32数据类型会转换为hf32数据类型，可提升计算性能，但同时也会带来精度损失。

#### 函数原型

```
__aicore__ inline void SetHF32(bool enableHF32 = false, int32_t transMode = 0)
```

#### 参数说明

| 参数名 | 输入/输出 | 描述 |
| --- | --- | --- |
| enableHF32 | 输入 | 配置是否开启HF32模式，默认值false(不开启)。 |
| transMode | 输入 | 配置在开启HF32模式时，float转换为hf32时所采用的ROUND模式。默认值0。 0：就近舍入，距离相等时向偶数进位。 1：就近舍入，距离相等时向远离0方向进位。 |

#### 约束说明

本接口仅支持在纯Cube模式下调用。

#### 调用示例

```
//纯Cube模式
#define ASCENDC_CUBE_ONLY
REGIST_MATMUL_OBJ(&pipe, GetSysWorkSpacePtr(), mm, &tiling);    //  A/B/C/BIAS类型是float
mm.SetTensorA(gm_a);
mm.SetTensorB(gm_b);
mm.SetBias(gm_bias);
mm.SetHF32(true);
mm.IterateAll(gm_c);
mm.SetHF32(false);
```
