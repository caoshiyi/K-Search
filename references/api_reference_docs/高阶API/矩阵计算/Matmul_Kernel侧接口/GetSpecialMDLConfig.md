# GetSpecialMDLConfig

**页面ID:** atlasascendc_api_07_0623  
**来源:** https://www.hiascend.com/document/detail/zh/CANNCommunityEdition/850/API/ascendcopapi/atlasascendc_api_07_0623.html

---

#### 产品支持情况

| 产品 | 是否支持 |
| --- | --- |
| Atlas A3 训练系列产品            /             Atlas A3 推理系列产品 | √ |
| Atlas A2 训练系列产品            /             Atlas A2 推理系列产品 | √ |
| Atlas 200I/500 A2 推理产品 | x |
| Atlas 推理系列产品            AI Core | x |
| Atlas 推理系列产品            Vector Core | x |
| Atlas 训练系列产品 | x |

#### 功能说明

用于配置SpecialMDL模板的参数，获取自定义SpecialMDL模板。SpecialMDL模板的介绍请参考表 模板特性。

#### 函数原型

```
__aicore__ constexpr MatmulConfig GetSpecialMDLConfig(const bool intrinsicsLimit = false, const bool batchLoop = false, const uint32_t doMTE2Preload = 0, const bool isVecND2NZ = false, bool isPerTensor = false, bool hasAntiQuantOffset = false)
```

#### 参数说明

本接口的所有参数用于设置MatmulConfig结构体中的参数，其中互相对应的参数的功能作用相同。

**表1 **接口参数说明

| 参数名 | 输入/输出 | 描述 |
| --- | --- | --- |
| intrinsicsLimit | 输入 | 用于设置参数intrinsicsCheck。          当左矩阵或右矩阵在单核上内轴（即尾轴）大于等于65535（元素个数）时，是否使能循环执行数据从Global Memory到L1 Buffer的搬入。例如，左矩阵A[M, K]，单核上的内轴数据singleCoreK大于65535，配置该参数为true后，API内部通过循环执行数据的搬入。参数取值如下：                     - false：当左矩阵或右矩阵在单核上内轴大于等于65535时，不使能循环执行数据的搬入（默认值）。           - true：当左矩阵或右矩阵在单核上内轴大于等于65535时，使能循环执行数据的搬入。 |
| batchLoop | 输入 | 用于设置参数isNBatch。          是否多Batch输入多Batch输出。仅对BatchMatmul有效，使能该参数后，仅支持Norm模板，且需调用IterateNBatch实现多Batch输入多Batch输出。参数取值如下：                     - false：不使能多Batch（默认值）。           - true：使能多Batch。 |
| doMTE2Preload | 输入 | 用于设置参数doMTE2Preload。          在MTE2流水间隙较大，且M/N数值较大时可通过该参数开启对应M/N方向的预加载功能，开启后能减小MTE2间隙，提升性能。预加载功能仅在MDL模板有效（不支持SpecialMDL模板）。参数取值如下：                     - 0：不开启（默认值）。           - 1：开启M方向preload。           - 2：开启N方向preload。                    注意：开启M/N方向的预加载功能时需保证K全载且M/N方向开启DoubleBuffer；其中，M方向的K全载条件为：singleCoreK/baseK <= stepKa；N方向的K全载条件为：singleCoreK/baseK <= stepKb。          该参数的使用样例请参考[M/N方向预加载Matmul算子样例](https://gitee.com/ascend/ascendc-api-adv/tree/master/examples/matrix/matmul_preload)。 |
| isVecND2NZ | 输入 | 预留参数，保持默认值即可。 |
| isPerTensor | 输入 | 用于设置参数isPerTensor。          A矩阵half类型输入且B矩阵int8_t类型输入场景，使能B矩阵量化时是否为per tensor。                     - true：per tensor量化。           - false：per channel量化。 |
| hasAntiQuantOffset | 输入 | 用于设置参数hasAntiQuantOffset。          A矩阵half类型输入且B矩阵int8_t类型输入场景，使能B矩阵量化时是否使用offset系数。 |

#### 返回值说明

MatmulConfig结构体。

#### 约束说明

无

#### 调用示例

```
constexpr MatmulConfig MM_CFG = GetSpecialMDLConfig();
AscendC::Matmul<A_TYPE, B_TYPE, C_TYPE, BIAS_TYPE, MM_CFG> mm;
REGIST_MATMUL_OBJ(&pipe, GetSysWorkSpacePtr(), mm, &tiling);
mm.SetTensorA(gm_a);
mm.SetTensorB(gm_b);
mm.SetBias(gm_bias);
mm.IterateAll(gm_c);
```
