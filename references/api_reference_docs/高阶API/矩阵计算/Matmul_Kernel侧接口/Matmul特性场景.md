# Matmul特性场景

**页面ID:** atlasascendc_api_07_10180  
**来源:** https://www.hiascend.com/document/detail/zh/CANNCommunityEdition/850/API/ascendcopapi/atlasascendc_api_07_10180.html

---

除了前述介绍的Matmul基本计算能力外，还请掌握Matmul的基础知识和算子实现。另外，Matmul矩阵编程还提供了适用于不同场景的处理能力及多种功能，具体场景和使用的关键接口或参数列于下表中，详细内容请见对应章节的介绍。

**表1 **Matmul特性表

| 特性分类 | 特性描述 | 涉及的关键API或参数 |
| --- | --- | --- |
| 功能实现 | 多核对齐场景 | SetDim、EnableMultiCoreSplitK(多核切K场景) |
| 多核非对齐场景 | SetTail、SetDim、EnableMultiCoreSplitK(多核切K场景) |  |
| 异步场景 | Iterate、GetTensorC、IterateAll |  |
| CallBack回调功能 | MatmulCallBackFunc、SetUserDefInfo、SetSelfDefineData |  |
| ChannelSplit功能 | MatmulConfig模板参数中的isEnableChannelSplit参数 |  |
| GEMV场景 | SetAType |  |
| 上三角/下三角计算功能 | MatmulPolicy模板参数 |  |
| TSCM输入场景 | DataCopy |  |
| ND_ALIGN输出功能 | SetCType |  |
| Partial Out功能 | MatmulConfig模板参数中的isPartialOutput参数 |  |
| 双主模式功能 | MatmulConfig模板参数中的enableMixDualMaster参数 |  |

**表2 **Matmul特性表

| 特性分类 | 特性描述 | 涉及的关键API或参数 |
| --- | --- | --- |
| 功能实现 | 量化场景 | SetDequantType、SetQuantScalar、SetQuantVector |
| Sparse Matmul场景 | SetSparse、SetSparseIndex |  |

**表3 **BatchMatmul特性表

| 特性分类 | 特性描述 | 主要涉及的API接口 |
| --- | --- | --- |
| 功能实现 | BatchMatmul基础场景 | NORMAL排布格式的BatchMatmul：IterateBatch、SetBatchInfoForNormal |
| BSNGD、SBNGD、BNGS1S2排布格式的BatchMatmul：IterateBatch、SetALayout、SetBLayout、SetCLayout、SetBatchNum |  |  |
| Batch Matmul复用Bias矩阵 | GetMMConfig、IterateBatch、SetBatchInfoForNormal |  |
