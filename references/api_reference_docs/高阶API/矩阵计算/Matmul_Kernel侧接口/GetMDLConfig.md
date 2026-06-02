# GetMDLConfig

**页面ID:** atlasascendc_api_07_0622  
**来源:** https://www.hiascend.com/document/detail/zh/CANNCommunityEdition/850/API/ascendcopapi/atlasascendc_api_07_0622.html

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

用于配置MDL模板的参数，获取自定义MDL模板。MDL模板的介绍请参考表 模板特性。

#### 函数原型

```
__aicore__ constexpr MatmulConfig GetMDLConfig(const bool intrinsicsLimit = false, const bool batchLoop = false, const uint32_t doMTE2Preload = 0, const bool isVecND2NZ = false, bool isPerTensor = false, bool hasAntiQuantOffset = false, const bool enUnitFlag = false, const bool isMsgReuse = true, const bool enableUBReuse = true, const bool enableL1CacheUB = false, const bool enableMixDualMaster = false, const bool enableKdimReorderLoad = false)
```

#### 参数说明

本接口的所有参数用于设置MatmulConfig结构体中的参数，其中互相对应的参数的功能作用相同。

**表1 **接口参数说明

| 参数名 | 输入/输出 | 描述 |
| --- | --- | --- |
| intrinsicsLimit | 输入 | 用于设置参数intrinsicsCheck。          当左矩阵或右矩阵在单核上内轴（即尾轴）大于等于65535（元素个数）时，是否使能循环执行数据从Global Memory到L1 Buffer的搬入。例如，左矩阵A[M, K]，单核上的内轴数据singleCoreK大于65535，配置该参数为true后，API内部通过循环执行数据的搬入。参数取值如下：                     - false：当左矩阵或右矩阵在单核上内轴大于等于65535时，不使能循环执行数据的搬入（默认值）。           - true：当左矩阵或右矩阵在单核上内轴大于等于65535时，使能循环执行数据的搬入。 |
| batchLoop | 输入 | 用于设置参数isNBatch。          是否多Batch输入多Batch输出。仅对BatchMatmul有效，使能该参数后，仅支持Norm模板，且需调用IterateNBatch实现多Batch输入多Batch输出。参数取值如下：                     - false：不使能多Batch（默认值）。           - true：使能多Batch。 |
| doMTE2Preload | 输入 | 用于设置参数doMTE2Preload。          在MTE2流水间隙较大，且M/N数值较大时可通过该参数开启对应M/N方向的预加载功能，开启后能减小MTE2间隙，提升性能。预加载功能仅在MDL模板有效（不支持SpecialMDL模板）。参数取值如下：                     - 0：不开启（默认值）。           - 1：开启M方向preload。           - 2：开启N方向preload。                    注意：开启M/N方向的预加载功能时需保证K全载且M/N方向开启DoubleBuffer；其中，M方向的K全载条件为：singleCoreK/baseK <= stepKa；N方向的K全载条件为：singleCoreK/baseK <= stepKb。          该参数的使用样例请参考[M/N方向预加载Matmul算子样例](https://gitee.com/ascend/ascendc-api-adv/tree/master/examples/matrix/matmul_preload)。 |
| isVecND2NZ | 输入 | 用于设置参数enVecND2NZ。          使能通过vector指令进行ND2NZ。使能时需要设置SetLocalWorkspace。参数取值如下：                     - false：不使能通过vector指令进行ND2NZ（默认值）。           - true：使能通过vector指令进行ND2NZ。                    针对              Atlas 推理系列产品             AI Core，在Unified Buffer空间足够的条件下（Unified Buffer空间大于2倍TCubeTiling的transLength参数），建议优先使能该参数，搬运性能更好。 |
| isPerTensor | 输入 | 用于设置参数isPerTensor。          A矩阵half类型输入且B矩阵int8_t类型输入场景，使能B矩阵量化时是否为per tensor。                     - true：per tensor量化。           - false：per channel量化。 |
| hasAntiQuantOffset | 输入 | 用于设置参数hasAntiQuantOffset。          A矩阵half类型输入且B矩阵int8_t类型输入场景，使能B矩阵量化时是否使用offset系数。 |
| enUnitFlag | 输入 | 用于设置参数enUnitFlag。          使能UnitFlag功能，使计算与搬运流水并行，提高性能。Norm, IBShare下默认使能，MDL下默认不使能。参数取值如下：                     - false：不使能UnitFlag功能。           - true：使能UnitFlag功能。 |
| isMsgReuse | 输入 | 用于设置参数enableReuse。          SetSelfDefineData函数设置的回调函数中的dataPtr是否直接传递计算数据。若未调用SetSelfDefineData设置dataPtr，该参数仅支持默认值true。参数取值如下：                     - true：直接传递计算数据，仅限单个值。           - false：传递GM上存储的数据地址信息。 |
| enableUBReuse | 输入 | 用于设置参数enableUBReuse。          是否使能Unified Buffer复用。在Unified Buffer空间足够的条件下（Unified Buffer空间大于4倍TCubeTiling的transLength参数），使能该参数后，Unified Buffer空间分为互不重叠的两份，分别存储Matmul计算相邻前后两轮迭代的数据，后一轮迭代数据的搬入将不必等待前一轮迭代的Unified Buffer空间释放，从而优化流水。参数取值如下：                     - true：使能Unified Buffer复用。           - false：不使能Unified Buffer复用。                                  Atlas A3 训练系列产品             /              Atlas A3 推理系列产品             不支持该参数。                        Atlas A2 训练系列产品             /              Atlas A2 推理系列产品             不支持该参数。                        Atlas 推理系列产品             AI Core支持该参数。                        Atlas 200I/500 A2 推理产品             不支持该参数。 |
| enableL1CacheUB | 输入 | 用于设置参数enableL1CacheUB 。          是否使能L1 Buffer缓存Unified Buffer计算块。建议在MTE3和MTE2流水串行较多的场景使用。参数取值如下：                     - true：使能L1 Buffer缓存Unified Buffer计算块。           - false：不使能L1 Buffer缓存Unified Buffer计算块。                    若要使能L1 Buffer缓存Unified Buffer计算块，必须在Tiling实现中调用SetMatmulConfigParams接口将参数enableL1CacheUBIn设置为true。                        Atlas A3 训练系列产品             /              Atlas A3 推理系列产品             不支持该参数。                        Atlas A2 训练系列产品             /              Atlas A2 推理系列产品             不支持该参数。                        Atlas 推理系列产品             AI Core支持该参数。                        Atlas 200I/500 A2 推理产品             不支持该参数。 |
| enableMixDualMaster | 输入 | 用于设置参数enableMixDualMaster。          是否使能MixDualMaster（双主模式）。区别于MIX模式（包含矩阵计算和矢量计算）通过消息机制驱动AIC运行，双主模式为AIC和AIV独立运行代码，不依赖消息驱动，用于提升性能。该参数默认值为false，仅能在以下场景设置为true：                     - 核函数的类型为MIX，同时AIC核数 : AIV核数为1:1。           - 核函数的类型为MIX，同时AIC核数 : AIV核数为1:2，且A矩阵和B矩阵同时使能IBSHARE参数。                    注意，使能MixDualMaster场景，需要满足：                     - 同一算子中所有Matmul对象的该参数取值必须保持一致。           - A/B/Bias矩阵只支持从GM搬入。           - 获取矩阵计算结果只支持调用IterateAll接口输出到GlobalTensor，即计算结果放置于Global Memory的地址，不能调用GetTensorC等接口获取结果。                                  Atlas A3 训练系列产品             /              Atlas A3 推理系列产品             支持该参数。                        Atlas A2 训练系列产品             /              Atlas A2 推理系列产品             支持该参数。                        Atlas 推理系列产品             AI Core不支持该参数。                        Atlas 200I/500 A2 推理产品             不支持该参数。 |
| enableKdimReorderLoad | 输入 | 用于设置参数enableKdimReorderLoad。          是否使能K轴错峰加载数据。基于相同Tiling参数，执行Matmul计算时，如果多核的左矩阵或者右矩阵相同，且存储于Global Memory，多个核一般会同时访问相同地址以加载矩阵数据，引发同地址访问冲突，影响性能。使能该参数后，多核执行Matmul时，将尽量在相同时间访问矩阵的不同Global Memory地址，减少地址访问冲突概率，提升性能。该参数功能只支持MDL模板，建议K轴较大且左矩阵和右矩阵均非全载场景使能参数。参数取值如下，具体样例请参考[K轴错峰加载数据的算子样例](https://gitee.com/ascend/ascendc-api-adv/tree/master/examples/matrix/matmul_k_reorder_load)。                     - false：默认值，关闭K轴错峰加载数据的功能。           - true：开启K轴错峰加载数据的功能。                                  Atlas A3 训练系列产品             /              Atlas A3 推理系列产品             支持该参数。                        Atlas A2 训练系列产品             /              Atlas A2 推理系列产品             支持该参数。                        Atlas 推理系列产品             AI Core不支持该参数。                        Atlas 200I/500 A2 推理产品             不支持该参数。 |

#### 返回值说明

MatmulConfig结构体。

#### 约束说明

无

#### 调用示例

```
constexpr MatmulConfig MM_CFG = GetMDLConfig();
AscendC::Matmul<A_TYPE, B_TYPE, C_TYPE, BIAS_TYPE, MM_CFG> mm;
REGIST_MATMUL_OBJ(&pipe, GetSysWorkSpacePtr(), mm, &tiling);
mm.SetTensorA(gm_a);
mm.SetTensorB(gm_b);
mm.SetBias(gm_bias);
mm.IterateAll(gm_c);
```
