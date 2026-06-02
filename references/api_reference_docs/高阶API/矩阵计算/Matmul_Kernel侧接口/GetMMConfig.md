# GetMMConfig

**页面ID:** atlasascendc_api_07_0627  
**来源:** https://www.hiascend.com/document/detail/zh/CANNCommunityEdition/850/API/ascendcopapi/atlasascendc_api_07_0627.html

---

#### 产品支持情况

| 产品 | 是否支持 |
| --- | --- |
| Atlas A3 训练系列产品            /             Atlas A3 推理系列产品 | √ |
| Atlas A2 训练系列产品            /             Atlas A2 推理系列产品 | √ |
| Atlas 200I/500 A2 推理产品 | x |
| Atlas 推理系列产品            AI Core | √ |
| Atlas 推理系列产品            Vector Core | x |
| Atlas 训练系列产品 | x |

#### 功能说明

灵活的自定义Matmul模板参数配置。通过设置MatmulConfigMode、MatmulShapeParams、MatmulQuantParams、MatmulBatchParams、MatmulFuncParams，获取自定义的MatmulConfig。

MatmulConfigMode指定了获取并要修改的MatmulConfig模板，各模板介绍请参考模板特性；用户根据使用需求通过设置可变参数，即一个或多个任意顺序的MatmulShapeParams、MatmulQuantParams、MatmulBatchParams、MatmulFuncParams，修改该MatmulConfig模板的相应参数配置。相比GetNormalConfig、GetMDLConfig等获取模板的接口，该接口提供了更灵活的自定义Matmul模板参数的配置方式。

#### 函数原型

```
template <MatmulConfigMode configMode, typename... ArgTypes>
__aicore__ inline constexpr MatmulConfig GetMMConfig(ArgTypes&&... args)
```

#### 参数说明

**表1 **模板参数说明

| 参数名 | 描述 |
| --- | --- |
| configMode | 获取的MatmulConfig模板。 |
| ArgTypes | 可变模板参数。 |

**表2 **参数说明

| 参数名 | 输入/输出 | 描述 |
| --- | --- | --- |
| args | 输入 | 可变参数，任意顺序传入需要设置的MatmulShapeParams、MatmulQuantParams、MatmulBatchParams、MatmulFuncParams中的一个或多个。 |

**表3 **MatmulConfigMode参数说明

| 参数 | 说明 |
| --- | --- |
| CONFIG_NORM | 表示设置MatmulConfig默认值为Norm模板 |
| CONFIG_MDL | 表示设置MatmulConfig默认值为MDL模板 |
| CONFIG_SPECIALMDL | 表示设置MatmulConfig默认值为SpecialMDL模板 |
| CONFIG_IBSHARE | 表示设置MatmulConfig默认值为IBShare模板 |

**表4 **MatmulShapeParams参数说明

| 参数 | 数据类型 | 说明 |
| --- | --- | --- |
| singleCoreM | uint32_t | 单核内M轴shape大小，以元素为单位。 |
| singleCoreN | uint32_t | 单核内N轴shape大小，以元素为单位。 |
| singleCoreK | uint32_t | 单核内K轴shape大小，以元素为单位。 |
| basicM | uint32_t | 与TCubeTiling结构体中的baseM参数含义相同，Matmul计算时base块M轴长度，以元素为单位。 |
| basicN | uint32_t | 与TCubeTiling结构体中的baseN参数含义相同，Matmul计算时base块N轴长度，以元素为单位。 |
| basicK | uint32_t | 与TCubeTiling结构体中的baseK参数含义相同，Matmul计算时base块K轴长度，以元素为单位。 |

**表5 **MatmulQuantParams参数说明

| 参数 | 数据类型 | 说明 |
| --- | --- | --- |
| isPerTensor | bool | A矩阵half类型输入且B矩阵int8_t类型输入场景，使能B矩阵量化时是否为per tensor。                   - true：per tensor量化。          - false：per channel量化。 |
| hasAntiQuantOffset | bool | A矩阵half类型输入且B矩阵int8_t类型输入场景，使能B矩阵量化时是否使用offset系数。 |

**表6 **MatmulBatchParams参数说明

| 参数 | 数据类型 | 说明 |
| --- | --- | --- |
| isNBatch | bool | 是否多Batch输入多Batch输出。仅对BatchMatmul有效，使能该参数后，仅支持Norm模板，且需调用IterateNBatch实现多Batch输入多Batch输出。参数取值如下：                   - false：不使能多Batch（默认值）。          - true：使能多Batch。 |
| batchMode | BatchMode | BatchMatmul场景中Layout类型为NORMAL时，设置BatchMatmul输入A/B矩阵的多batch数据总和与L1 Buffer的大小关系。参数取值如下：                   - BatchMode::BATCH_LESS_THAN_L1：多batch数据总和<L1 Buffer Size；          - BatchMode::BATCH_LARGE_THAN_L1：多batch数据总和>L1 Buffer Size；          - BatchMode::SINGLE_LARGE_THAN_L1：单batch数据总和>L1 Buffer Size。 |
| isBiasBatch | bool | 批量多Batch的Matmul场景，即BatchMatmul场景，Bias的大小是否带有Batch轴。参数取值如下：                   - true：Bias带有Batch轴，Bias大小为Batch * N（默认值）。                            - false：Bias不带Batch轴，Bias大小为N，多Batch计算Matmul时，会复用Bias。           注意：BatchMode::SINGLE_LARGE_THAN_L1场景仅支持设置为true。                          Atlas A2 训练系列产品              /               Atlas A2 推理系列产品              支持该参数。                          Atlas A3 训练系列产品              /               Atlas A3 推理系列产品              支持该参数。                          Atlas 推理系列产品              AI Core不支持设置为false。                          Atlas 200I/500 A2 推理产品              不支持设置为false。 |
| bmmOutMode | BatchOutMode | 预留参数。 |

**表7 **MatmulFuncParams参数说明

| 参数 | 数据类型 | 说明 |
| --- | --- | --- |
| intrinsicsLimit | bool | 当左矩阵或右矩阵在单核上内轴（即尾轴）大于等于65535（元素个数）时，是否使能循环执行数据从Global Memory到L1 Buffer的搬入。例如，左矩阵A[M, K]，单核上的内轴数据singleCoreK大于65535，配置该参数为true后，API内部通过循环执行数据的搬入。参数取值如下：                   - false：当左矩阵或右矩阵在单核上内轴大于等于65535时，不使能循环执行数据的搬入（默认值）。          - true：当左矩阵或右矩阵在单核上内轴大于等于65535时，使能循环执行数据的搬入。 |
| enVecND2NZ | bool | 使能通过vector指令进行ND2NZ。使能时需要设置SetLocalWorkspace。参数取值如下：                   - false：不使能通过vector指令进行ND2NZ（默认值）。          - true：使能通过vector指令进行ND2NZ。                  针对             Atlas 推理系列产品            AI Core，在Unified Buffer空间足够的条件下（Unified Buffer空间大于2倍TCubeTiling的transLength参数），建议优先使能该参数，搬运性能更好。 |
| enableDoubleCache | bool | 开启IBShare模板后，在L1 Buffer上是否同时缓存两块数据。参数取值如下：                   - false：L1 Buffer上同时缓存一块数据（默认值）。          - true：使能L1 Buffer上同时缓存两块数据。                  注意：该参数取值为true时，需要控制基本块大小，防止两块数据的缓存超过L1 Buffer大小限制。 |
| enableL1CacheUB | bool | 是否使能L1 Buffer缓存Unified Buffer计算块。建议在MTE3和MTE2流水串行较多的场景使用。参数取值如下：                   - true：使能L1 Buffer缓存Unified Buffer计算块。          - false：不使能L1 Buffer缓存Unified Buffer计算块。                  若要使能L1 Buffer缓存Unified Buffer计算块，必须在Tiling实现中调用SetMatmulConfigParams接口将参数enableL1CacheUBIn设置为true。                      Atlas A3 训练系列产品            /             Atlas A3 推理系列产品            不支持该参数。                      Atlas A2 训练系列产品            /             Atlas A2 推理系列产品            不支持该参数。                      Atlas 推理系列产品            AI Core支持该参数。                      Atlas 200I/500 A2 推理产品            不支持该参数。 |
| doMTE2Preload | uint32_t | 在MTE2流水间隙较大，且M/N数值较大时可通过该参数开启对应M/N方向的预加载功能，开启后能减小MTE2间隙，提升性能。预加载功能仅在MDL模板有效（不支持SpecialMDL模板）。参数取值如下：                   - 0：不开启（默认值）。          - 1：开启M方向preload。          - 2：开启N方向preload。                  注意：开启M/N方向的预加载功能时需保证K全载且M/N方向开启DoubleBuffer；其中，M方向的K全载条件为：singleCoreK/baseK <= stepKa；N方向的K全载条件为：singleCoreK/baseK <= stepKb。         该参数的使用样例请参考[M/N方向预加载Matmul算子样例](https://gitee.com/ascend/ascendc-api-adv/tree/master/examples/matrix/matmul_preload)。 |
| Matmul做矩阵运算的循环迭代顺序，与表1中的iterateOrder参数含义相同。当ScheduleType参数取值为ScheduleType::OUTER_PRODUCT时，本参数生效。参数取值如下：                                                                                                                  ``` enum class IterateOrder {     ORDER_M = 0,   // 先往M轴方向偏移再往N轴方向偏移     ORDER_N,       // 先往N轴方向偏移再往M轴方向偏移     UNDEF,         // 当前无效 }; ```                                                                               注：Norm模板的Matmul场景、MDL模板使用时，若IterateOrder取值ORDER_M，TCubeTiling结构中的stepN需要大于1，IterateOrder取值ORDER_N时，TCubeTiling结构中的stepM需要大于1。         该参数的使用样例请参考[M/N方向流水并行Matmul算子样例](https://gitee.com/ascend/ascendc-api-adv/tree/master/examples/matrix/matmul_mndb)。                      Atlas A3 训练系列产品            /             Atlas A3 推理系列产品            支持该参数。                      Atlas A2 训练系列产品            /             Atlas A2 推理系列产品            支持该参数。                      Atlas 推理系列产品            AI Core不支持该参数。                      Atlas 200I/500 A2 推理产品            不支持该参数。 |  |  |
| scheduleType | ScheduleType | 配置Matmul数据搬运模式。参数取值如下：                   - ScheduleType::INNER_PRODUCT：默认模式，在K方向上做MTE1的循环搬运；          - ScheduleType::OUTER_PRODUCT：在M或N方向上做MTE1的循环搬运；使能后，需要与IterateOrder参数配合使用。                       该配置当前只在BatchMatmul场景（使能Norm模板）或 Matmul场景（使能MDL模板或Norm模板）生效。                           - 若IterateOrder取值ORDER_M，则N方向循环搬运（在singleCoreN大于baseN场景可能有性能提升），即B矩阵的MTE1搬运并行；               - 若IterateOrder取值ORDER_N，则M方向循环搬运（在singleCoreM大于baseM场景可能有性能提升），即A矩阵的MTE1搬运并行；               - 不能同时使能M方向和N方向循环搬运；                                         注：                   - Norm模板的Batch Matmul场景或者MDL模板中，singleCoreK>baseK时，不能使能ScheduleType::OUTER_PRODUCT取值，需使用默认模式。          - Norm模板或MDL模板的Matmul场景，仅支持在纯Cube模式（只有矩阵计算）下配置ScheduleType::OUTER_PRODUCT。          - MDL模板仅在调用IterateAll计算的场景支持配置ScheduleType::OUTER_PRODUCT。          - 仅在C矩阵输出至GM时，支持配置ScheduleType::OUTER_PRODUCT。                               Atlas A3 训练系列产品            /             Atlas A3 推理系列产品            支持该参数。                      Atlas A2 训练系列产品            /             Atlas A2 推理系列产品            支持该参数。                      Atlas 推理系列产品            AI Core不支持该参数。                      Atlas 200I/500 A2 推理产品            不支持该参数。 |
| enableReuse | bool | SetSelfDefineData函数设置的回调函数中的dataPtr是否直接传递计算数据。若未调用SetSelfDefineData设置dataPtr，该参数仅支持默认值true。参数取值如下：                   - true：直接传递计算数据，仅限单个值。          - false：传递GM上存储的数据地址信息。 |
| enableUBReuse | bool | 是否使能Unified Buffer复用。在Unified Buffer空间足够的条件下（Unified Buffer空间大于4倍TCubeTiling的transLength参数），使能该参数后，Unified Buffer空间分为互不重叠的两份，分别存储Matmul计算相邻前后两轮迭代的数据，后一轮迭代数据的搬入将不必等待前一轮迭代的Unified Buffer空间释放，从而优化流水。参数取值如下：                   - true：使能Unified Buffer复用。          - false：不使能Unified Buffer复用。                               Atlas A3 训练系列产品            /             Atlas A3 推理系列产品            不支持该参数。                      Atlas A2 训练系列产品            /             Atlas A2 推理系列产品            不支持该参数。                      Atlas 推理系列产品            AI Core支持该参数。                      Atlas 200I/500 A2 推理产品            不支持该参数。 |
| isPartialOutput | bool | 是否开启PartialOutput功能，即控制Matmul顺序输出K方向的基本块计算方式：Matmul一次Iterate计算的K轴是否进行累加计算。参数取值如下：                   - true：开启PartialOutput功能，一次Iterate的K轴不进行累加计算，Matmul每次计算输出局部baseK的baseM * baseN大小的矩阵分片。          - false：不开启PartialOutput功能，一次Iterate的K轴进行累加计算，Matmul每次计算输出SingleCoreK长度的baseM * baseN大小的矩阵分片。                               Atlas A3 训练系列产品            /             Atlas A3 推理系列产品            支持该参数。                      Atlas A2 训练系列产品            /             Atlas A2 推理系列产品            支持该参数。                      Atlas 推理系列产品            AI Core不支持该参数。                      Atlas 200I/500 A2 推理产品            不支持该参数。 |
| isA2B2Shared | bool | 是否开启A2和B2的全局管理，即控制所有Matmul对象是否共用A2和B2的double buffer机制。该配置为全局配置，所有Matmul对象取值必须保持一致。注意，开启时，A矩阵、B矩阵的基本块大小均不能超过32KB。         参数取值如下：                   - true：开启。          - false：关闭（默认值）。                               Atlas A3 训练系列产品            /             Atlas A3 推理系列产品            支持该参数。                      Atlas A2 训练系列产品            /             Atlas A2 推理系列产品            支持该参数。                      Atlas 推理系列产品            AI Core不支持该参数。         该参数取值为true时，建议同时设置enUnitFlag参数为true，使搬运与计算流水并行，提高性能。该参数的使用样例请参考[Matmul A2和B2全局管理样例](https://gitee.com/ascend/ascendc-api-adv/tree/master/examples/matrix/matmul_a2b2share)。 |
| isEnableChannelSplit | bool | 是否使能channel_split功能。正常情况下，Matmul计算出的CubeFormat::NZ格式的C矩阵分形为16*16，假设此时的分形个数为x，channel_split功能是使获得的C矩阵分形为16*8，同时分形个数变为2x。注意，当前仅在Matmul计算结果C矩阵的Format为CubeFormat::NZ，TYPE为float类型，输出到Global Memory的场景，支持使能该参数。参数取值如下：                   - false：默认值，不使能channel_split功能，输出的分形为16*16。          - true：使能channel_split功能，输出的分形为16*8。                          Atlas A3 训练系列产品              /               Atlas A3 推理系列产品              支持该参数。                          Atlas A2 训练系列产品              /               Atlas A2 推理系列产品              支持该参数。                          Atlas 推理系列产品              AI Core不支持该参数。                          Atlas 200I/500 A2 推理产品              不支持该参数。 |
| enableKdimReorderLoad | bool | 是否使能K轴错峰加载数据。基于相同Tiling参数，执行Matmul计算时，如果多核的左矩阵或者右矩阵相同，且存储于Global Memory，多个核一般会同时访问相同地址以加载矩阵数据，引发同地址访问冲突，影响性能。使能该参数后，多核执行Matmul时，将尽量在相同时间访问矩阵的不同Global Memory地址，减少地址访问冲突概率，提升性能。该参数功能只支持MDL模板，建议K轴较大且左矩阵和右矩阵均非全载场景使能参数。参数取值如下，具体样例请参考[K轴错峰加载数据的算子样例](https://gitee.com/ascend/ascendc-api-adv/tree/master/examples/matrix/matmul_k_reorder_load)。                   - false：默认值，关闭K轴错峰加载数据的功能。          - true：开启K轴错峰加载数据的功能。                               Atlas A3 训练系列产品            /             Atlas A3 推理系列产品            支持该参数。                      Atlas A2 训练系列产品            /             Atlas A2 推理系列产品            支持该参数。                      Atlas 推理系列产品            AI Core不支持该参数。                      Atlas 200I/500 A2 推理产品            不支持该参数。 |

#### 返回值说明

MatmulConfig结构体。

#### 约束说明

无

#### 调用示例

```
// 获取MatmulConfig模板为Norm模板
constexpr static MatmulConfigMode configMode = MatmulConfigMode::CONFIG_NORM;
// singleCoreM、singleCoreN、singleCoreK、basicM、basicN、basicK
constexpr static MatmulShapeParams shapeParams = {128, 128, 128, 64, 64, 64};
// B矩阵量化时为per channel且不使用offset系数
constexpr static MatmulQuantParams quantParams = {false, false};
// 不使能多Batch
constexpr static MatmulBatchParams batchParams{false};
// 不进行芯片指令搬运地址偏移量校验，使能通过vector进行ND2NZ
constexpr static MatmulFuncParams funcParams{false, true};
constexpr static MatmulConfig mmConfig = GetMMConfig<configMode>(shapeParams, quantParams, batchParams, funcParams);
```
