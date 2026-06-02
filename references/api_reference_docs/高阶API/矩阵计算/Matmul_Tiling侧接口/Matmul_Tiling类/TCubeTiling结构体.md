# TCubeTiling结构体

**页面ID:** atlasascendc_api_07_0673  
**来源:** https://www.hiascend.com/document/detail/zh/CANNCommunityEdition/850/API/ascendcopapi/atlasascendc_api_07_0673.html

---

TCubeTiling结构体包含Matmul Tiling切分算法的相关参数，被传递给Matmul Kernel侧，用于Matmul的切块、搬运和计算过程等。TCubeTiling结构体的参数说明见表1。

**表1 **TCubeTiling结构说明

| 参数名称 | 数据类型 | 说明 |
| --- | --- | --- |
| usedCoreNum | int | 使用的AI处理器核数，请根据实际情况设置。取值范围为：[1, AI处理器最大核数]。该参数与shape相关参数的关系为：usedCoreNum = (M / singleCoreM) * (N / singleCoreN)。 |
| M, N, Ka, Kb | int | A、B、C矩阵原始输入的shape大小，以元素为单位。M, Ka为A矩阵原始输入的Shape，Kb, N为B矩阵原始输入的Shape。 - 大小约束下述场景需要使能MatmulConfig中的intrinsicsCheck参数，以完成Matmul计算。  - 若A矩阵为ND格式，不进行转置，Ka大于65535时需要使能intrinsicsCheck参数，M无大小限制；进行转置，M大于65535时需要使能intrinsicsCheck参数，Ka无大小限制。  - 若B矩阵为ND格式，不进行转置，N大于65535时需要使能intrinsicsCheck参数，Kb无大小限制；进行转置，Kb大于65535时需要使能intrinsicsCheck参数，N无大小限制。  - 对齐约束  - 若A矩阵以NZ格式输入，则M需要以16个元素对齐，Ka需要以C0_size对齐；若B矩阵以NZ格式输入，Kb需要以C0_size对齐，N需要以16个元素对齐。  - 若A、B矩阵为ND格式，无对齐约束。 **注意：**NZ格式的输入，float数据类型的C0_size为8，half/bfloat16_t数据类型的C0_size为16，int8_t数据类型的C0_size为32，int4b_t数据类型的C0_size为64。 |
| singleCoreM, singleCoreN, singleCoreK | int | A、B、C矩阵单核内shape大小，以元素为单位。该参数取值必须大于0。 singleCoreK = K，多核处理时不对K进行切分；singleCoreM <= M；singleCoreN <= N。 **注意**：若A矩阵以NZ格式输入，则singleCoreM需要以16个元素对齐，singleCoreK需要以C0_size * fractal_num对齐；若B矩阵以NZ格式输入，则singleCoreK需要以C0_size * fractal_num对齐，singleCoreN需要以16个元素对齐。 NZ格式的输入，half/bfloat16_t数据类型的C0_size为16，fractal_num为1，float数据类型的C0_size为8，fractal_num为2，int8_t数据类型的C0_size为32，fractal_num为1，int4b_t数据类型的C0_size为64，fractal_num为1。其中，fractal_num表示为满足计算中的对齐要求需要的C0_size个数。 |
| baseM, baseN, baseK | int | A、B、C矩阵参与一次矩阵乘指令的shape大小，以元素为单位。 A、B、C矩阵参与一次矩阵乘的shape大小需要按分形对齐，其含义请参考Mmad中的数据格式说明。 **注意：**该参数取值必须大于0。 |
| depthA1, depthB1 | int | A1、B1中全载基本块的份数，depthA1为A1中全载baseM * baseK的份数，depthB1为B1中全载baseN * baseK的份数。 **注意：**该参数取值必须大于0。 |
| stepM， stepN，stepKa，stepKb | int | stepM为左矩阵在A1中缓存的buffer M方向上baseM的倍数。 stepN为右矩阵在B1中缓存的buffer N方向上baseN的倍数。 stepKa为左矩阵在A1中缓存的buffer Ka方向上baseK的倍数。 stepKb为右矩阵在B1中缓存的buffer Kb方向上baseK的倍数。 **注意：**该参数取值必须大于0。 |
| isBias | int | 是否使能Bias，参数取值如下： - 0：不使能Bias（默认值）。- 1：使能Bias。 **注意：**该参数不支持除上述外的其他取值，设置为其他值时参数行为未定义。 |
| transLength | int | max(A1Length, B1Length, C1Length, BiasLength)。其中，A1Length, B1Length, C1Length, BiasLength分别表示A/B/C/Bias矩阵在计算过程中需要临时占用的UB空间大小。 |
| iterateOrder | int | 一次Iterate计算出[baseM, baseN]大小的C矩阵分片，Iterate完成后，Matmul会自动偏移下一次Iterate输出的C矩阵位置，iterOrder表示自动偏移的顺序。参数取值如下： - 0：先往M轴方向偏移再往N轴方向偏移。- 1：先往N轴方向偏移再往M轴方向偏移。 **注意：**该参数不支持除上述外的其他取值，设置为其他值时参数行为未定义。 |
| dbL0A, dbL0B, dbL0C | int | MTE1是否开启double buffer。 dbL0A：左矩阵MTE1是否开启double buffer；dbL0B：右矩阵MTE1是否开启double buffer；dbL0C：MMAD是否开启double buffer。参数取值如下： - 1：不开启double buffer。- 2：开启double buffer。 **注意：**该参数不支持除上述外的其他取值，设置为其他值时参数行为未定义。 |
| shareMode | int | 该参数预留，开发者无需关注。 |
| shareL1Size | int | 该参数预留，开发者无需关注。 |
| shareL0CSize | int | 该参数预留，开发者无需关注。 |
| shareUbSize | int | 该参数预留，开发者无需关注。 |
| batchM | int | 该参数预留，开发者无需关注。 |
| batchN | int | 该参数预留，开发者无需关注。 |
| singleBatchM | int | 该参数预留，开发者无需关注。 |
| singleBatchN | int | 该参数预留，开发者无需关注。 |

多数情况下，用户通过调用GetTiling接口获取TCubeTiling结构体，具体流程请参考使用说明。如果用户自定义TCubeTiling参数，各个参数的取值需要满足表1和表2中的对应参数的约束条件。如果用户通过调用GetTiling接口获取TCubeTiling结构体后，需要修改调整Tiling，请参考如下TCubeTiling参数约束和性能调优推荐取值，进行参数的设置。

- TCubeTiling参数约束一组合法的TCubeTiling参数需要同时满足表2中的所有约束条件。若Matmul对象的MatmulConfig模板为MDL模板，除表2外，还同时需要满足表3 MDL模板补充约束条件。

**表2 **TCubeTiling约束条件

| 约束条件 | 说明 |
| --- | --- |
| usedCoreNum <= aiCoreCnt | 使用核数小于等于当前AI处理器的最大核数 |
| baseM * baseK * sizeof(A_type) * dbL0A< l0a_size | A矩阵base块不超过l0a buffer大小 |
| baseN * baseK * sizeof(B_type) * dbL0B < l0b_size | B矩阵base块不超过l0b buffer大小 |
| baseM * baseN * sizeof(l0c_type) * dbL0C < l0c_size，其中l0c_type为int32_t或者float数据类型。 | C矩阵base块不超过l0c buffer大小 |
| baseN * sizeof(Bias_type) < biasT_size | Bias的base块不超过BiasTable buffer大小 |
| stepM * stepKa * db = depthA1 db这里表示为左矩阵MTE2是否开启double buffer，即L1是否开启double buffer，取值1（不开启double buffer）或2（开启double buffer） | depthA1的取值与stepM * stepKa  * db相同 |
| stepN * stepKb * db = depthB1 db这里表示为右矩阵MTE2是否开启double buffer，即L1是否开启double buffer，取值1（不开启double buffer）或2（开启double buffer） | depthB1的取值与stepN * stepKb  * db相同 |
| 对于A矩阵在L1上的缓存块大小AL1Size、B矩阵在L1上的缓存块大小BL1Size必须满足：   - 无bias场景AL1Size + BL1Size <= L1_size    - 有bias场景AL1Size + BL1Size + baseN * sizeof(Bias_type) <= L1_size  其中，AL1Size、BL1Size的计算方式如下：  - 转置场景：AL1Size = CeilDiv(baseM, C0_size) * baseK * depthA1 * sizeof(A_type) BL1Size = baseN * baseK * depthB1 * sizeof(B_type)   - 非转置场景：AL1Size = baseM * baseK * depthA1 * sizeof(A_type) BL1Size = CeilDiv(baseN, C0_size)* baseK * depthB1 * sizeof(B_type) | A矩阵、B矩阵和Bias在L1缓存块满足L1 buffer大小限制； **注意：**float数据类型的C0_size为8，half/bfloat16_t数据类型的C0_size为16，int8_t数据类型的C0_size为32，int4b_t数据类型的C0_size为64。 |
| baseM * baseK, baseK * baseN和baseM * baseN按照NZ格式的分形对齐 | A矩阵、B矩阵、C矩阵的base块需要满足对齐约束：   - baseM和baseN需要以16个元素对齐；A矩阵非转置且B矩阵转置场景，baseK需要以C0_size对齐；其余场景（A矩阵转置或B矩阵非转置场景），baseK以16个元素对齐； **注意：**float/int32_t数据类型的C0_size为8，half/bfloat16_t数据类型的C0_size为16，int8_t数据类型的C0_size为32，int4b_t数据类型的C0_size为64。 |

**表3 **MDL模板补充约束条件

| 约束条件 | 说明 |
| --- | --- |
| Ka不全载时，即Ka / baseK > stepKa，stepM = 1 | K方向非全载时，M方向只能逐块搬运 |
| Kb不全载时，即Kb / baseK > stepKb，stepN = 1 | K方向非全载时，N方向只能逐块搬运 |
| kaStepIter_ % kbStepIter_ = 0或者kbStepIter_ % kaStepIter_ = 0 kaStepIter_ = CeilDiv(tiling_->singleCoreK_, tiling_->baseK * tiling_->stepKa) kbStepIter_ = CeilDiv(tiling_->singleCoreK_, tiling_->baseK * tiling_->stepKb) | MDL模板K方向循环搬运要求Ka和Kb方向迭代次数为倍数关系 kaStepIter_ ：Ka方向循环搬运迭代次数 kbStepIter_ ：Kb方向循环搬运迭代次数 |

- 性能调优推荐取值

根据Tiling调优经验，部分TCubeTiling参数值或取值方式推荐如下：

  - base块推荐(baseM, baseN, baseK)：(128, 256, 64)
  - dbL0A / dbL0B = 2
  - depthA1 / (stepM * stepKa) = 2
  - depthB1 / (stepN * stepKb) = 2
  - 优先设置参数stepKa/stepKb，使得K方向全载，再考虑M方向或N方向全载
