# SetMatmulConfigParams

**页面ID:** atlasascendc_api_07_0704  
**来源:** https://www.hiascend.com/document/detail/zh/CANNCommunityEdition/850/API/ascendcopapi/atlasascendc_api_07_0704.html

---

#### 功能说明

在计算Tiling时，用于自定义设置表1中的MatmulConfig参数。本接口中配置的参数对应的功能在Tiling与Kernel中需要保持一致，所以本接口中的参数取值，需要与Kernel侧对应的MatmulConfig参数值保持一致，详细MatmulConfig参数请见表2。

#### 函数原型

```
void SetMatmulConfigParams(int32_t mmConfigTypeIn = 1, bool enableL1CacheUBIn = false, ScheduleType scheduleTypeIn = ScheduleType::INNER_PRODUCT, MatrixTraverse traverseIn = MatrixTraverse::NOSET, bool enVecND2NZIn = false)
```

```
void SetMatmulConfigParams(const MatmulConfigParams& configParams)
```

#### 参数说明

**表1 **参数说明

| 参数名 | 输入/输出 | 描述 |
| --- | --- | --- |
| mmConfigTypeIn | 输入 | 设置Matmul的模板类型，需要与Matmul对象创建的模板一致，当前只支持配置为0或1。 - 0：代表Norm模板- 1：代表MDL模板，默认值为1 |
| enableL1CacheUBIn | 输入 | 配置是否使能L1缓存UB计算块；参考使能场景：MTE3和MTE2流水串行较多的场景。 - false：不使能L1缓存UB计算块，默认值为false- true：使能L1缓存UB计算块 Atlas A3 训练系列产品/Atlas A3 推理系列产品不支持该参数。 Atlas A2 训练系列产品/Atlas A2 推理系列产品不支持该参数。 Atlas 推理系列产品AI Core支持该参数。 Atlas 200I/500 A2 推理产品不支持该参数。 |
| scheduleTypeIn | 输入 | 配置Matmul数据搬运模式。参数取值如下： - ScheduleType::INNER_PRODUCT：默认模式，在K方向上做MTE1的循环搬运- ScheduleType::OUTER_PRODUCT：在M或N方向上做MTE1的循环搬运- ScheduleType::N_BUFFER_33：NBuffer33模板的数据搬运模式，MTE2每次搬运A矩阵的1x3个基本块，直至A矩阵所有3x3个基本块全载在L1 Buffer中 |
| Matmul做矩阵运算的循环迭代顺序，即一次迭代计算出[baseM, baseN]大小的C矩阵分片后，自动偏移到下一次迭代输出的C矩阵位置的偏移顺序。参数取值如下： ``` enum class MatrixTraverse{     NOSET = 0,   // 当前无效     FIRSTM,      // 先往M轴方向偏移再往N轴方向偏移     FIRSTN,      // 先往N轴方向偏移再往M轴方向偏移 }; ``` |  |  |
| enVecND2NZIn | 输入 | 是否使能ND2NZ。 |
| config相关参数，类型为MatmulConfigParams，结构体具体定义如下。其中的参数说明请参考表2。 ``` struct MatmulConfigParams {     int32_t mmConfigType;     bool enableL1CacheUB;     ScheduleType scheduleType;     MatrixTraverse traverse;     bool enVecND2NZ;     MatmulConfigParams(int32_t mmConfigTypeIn = 1, bool enableL1CacheUBIn = false,         ScheduleType scheduleTypeIn = ScheduleType::INNER_PRODUCT, MatrixTraverse traverseIn = MatrixTraverse::NOSET,         bool enVecND2NZIn = false) {         mmConfigType = mmConfigTypeIn;         enableL1CacheUB = enableL1CacheUBIn;         scheduleType = scheduleTypeIn;         traverse = traverseIn;         enVecND2NZ = enVecND2NZIn;     } }; ``` |  |  |

**表2 **MatmulConfigParams结构体内参数说明

| 参数名称 | 含义 |
| --- | --- |
| mmConfigType | 设置Matmul的模板类型，需要与Matmul对象创建的模板一致，当前只支持配置为0或1。 - 0：代表Norm模板- 1：代表MDL模板，默认值为1 |
| enableL1CacheUB | 配置是否使能L1缓存UB计算块；参考使能场景：MTE3和MTE2流水串行较多的场景。 - false：不使能L1缓存UB计算块，默认值为false- true：使能L1缓存UB计算块 |
| scheduleType | 配置Matmul数据搬运模式。参数取值如下： - ScheduleType::INNER_PRODUCT：默认模式，在K方向上做MTE1的循环搬运- ScheduleType::OUTER_PRODUCT：在M或N方向上做MTE1的循环搬运- ScheduleType::N_BUFFER_33：NBuffer33模板的数据搬运模式，MTE2每次搬运A矩阵的1x3个基本块，直至A矩阵所有3x3个基本块全载在L1 Buffer中 |
| Matmul做矩阵运算的循环迭代顺序，即一次迭代计算出[baseM, baseN]大小的C矩阵分片后，自动偏移到下一次迭代输出的C矩阵位置的偏移顺序。参数取值如下： ``` enum class MatrixTraverse{     NOSET = 0,   // 当前无效     FIRSTM,      // 先往M轴方向偏移再往N轴方向偏移     FIRSTN,      // 先往N轴方向偏移再往M轴方向偏移 }; ``` |  |
| enVecND2NZ | 是否使能ND2NZ |

#### 约束说明

- 本接口必须在GetTiling接口前调用。
- 若Matmul对象使用NBuffer33模板策略，即MatmulPolicy为NBuffer33MatmulPolicy，则在调用GetTiling接口生成Tiling参数前，必须通过本接口将scheduleTypeIn参数设置为ScheduleType::N_BUFFER_33，以启用NBuffer33模板策略的Tiling生成逻辑。

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
tiling.SetMatmulConfigParams(0);  // 额外设置
// tiling.SetMatmulConfigParams({1, false, ScheduleType::OUTER_PRODUCT, MatrixTraverse::FIRSTM});
optiling::TCubeTiling tilingData;   
int ret = tiling.GetTiling(tilingData);
```
