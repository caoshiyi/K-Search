# GetSpecialBasicConfig

**页面ID:** atlasascendc_api_07_0626  
**来源:** https://www.hiascend.com/document/detail/zh/CANNCommunityEdition/850/API/ascendcopapi/atlasascendc_api_07_0626.html

---

#### 功能说明

用于配置SpecialBasicBlock模板的参数，获取自定义SpecialBasicBlock模板。当前为预留接口。

#### 函数原型

```
__aicore__ constexpr MatmulConfig GetSpecialBasicConfig(const uint32_t basicM, const uint32_t basicN, const uint32_t basicK, const uint32_t singleCoreM, const uint32_t singleCoreN, const uint32_t singleCoreK, const uint32_t stepM, const uint32_t stepN, const bool intrinsicsLimit = false, const bool batchLoop = false, const BatchMode bmmMode = BatchMode::BATCH_LESS_THAN_L1)
```

#### 参数说明

本接口的所有参数用于设置MatmulConfig结构体中的参数，其中互相对应的参数的功能作用相同。

**表1 **接口参数说明

| 参数名 | 输入/输出 | 描述 |
| --- | --- | --- |
| basicM | 输入 | 用于设置参数basicM。          与TCubeTiling结构体中的baseM参数含义相同，Matmul计算时base块M轴长度，以元素为单位。 |
| basicN | 输入 | 用于设置参数basicN。          与TCubeTiling结构体中的baseN参数含义相同，Matmul计算时base块N轴长度，以元素为单位。 |
| basicK | 输入 | 用于设置参数basicK。          与TCubeTiling结构体中的baseK参数含义相同，Matmul计算时base块K轴长度，以元素为单位。 |
| singleCoreM | 输入 | 用于设置参数singleCoreM。          单核内M轴shape大小，以元素为单位。 |
| singleCoreN | 输入 | 用于设置参数singleCoreN。          单核内N轴shape大小，以元素为单位。 |
| singleCoreK | 输入 | 用于设置参数singleCoreK。          单核内K轴shape大小，以元素为单位。 |
| stepM | 输入 | 用于设置参数stepM。          左矩阵在A1中缓存的bufferM方向上baseM的倍数。 |
| stepN | 输入 | 用于设置参数stepN。          右矩阵在B1中缓存的bufferN方向上baseN的倍数。 |
| intrinsicsLimit | 输入 | 用于设置参数intrinsicsCheck。          当左矩阵或右矩阵在单核上内轴（即尾轴）大于等于65535（元素个数）时，是否使能循环执行数据从Global Memory到L1 Buffer的搬入。例如，左矩阵A[M, K]，单核上的内轴数据singleCoreK大于65535，配置该参数为true后，API内部通过循环执行数据的搬入。参数取值如下：                     - false：当左矩阵或右矩阵在单核上内轴大于等于65535时，不使能循环执行数据的搬入（默认值）。           - true：当左矩阵或右矩阵在单核上内轴大于等于65535时，使能循环执行数据的搬入。 |
| batchLoop | 输入 | 用于设置参数isNBatch。          是否多Batch输入多Batch输出。仅对BatchMatmul有效，使能该参数后，仅支持Norm模板，且需调用IterateNBatch实现多Batch输入多Batch输出。参数取值如下：                     - false：不使能多Batch（默认值）。           - true：使能多Batch。 |
| bmmMode | 输入 | 用于设置参数batchMode。该参数用于BatchMatmul场景，关于BatchMatmul的介绍请参考Batch Matmul基础功能。          BatchMatmul场景中Layout类型为NORMAL时，设置BatchMatmul输入A/B矩阵的多batch数据总和与L1 Buffer的大小关系。参数取值如下：                     - BatchMode::BATCH_LESS_THAN_L1：多batch数据总和<L1 Buffer Size；           - BatchMode::BATCH_LARGE_THAN_L1：多batch数据总和>L1 Buffer Size；           - BatchMode::SINGLE_LARGE_THAN_L1：单batch数据总和>L1 Buffer Size。 |

#### 返回值说明

MatmulConfig结构体。

#### 约束说明

无
