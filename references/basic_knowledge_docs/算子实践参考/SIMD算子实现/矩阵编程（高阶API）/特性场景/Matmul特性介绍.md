# Matmul特性介绍

**页面ID:** atlas_ascendc_10_10012  
**来源:** https://www.hiascend.com/document/detail/zh/CANNCommunityEdition/850/opdevg/Ascendcopdevg/atlas_ascendc_10_10012.html

---

除了前述基础知识和算子实现中介绍的基本计算能力外，Matmul矩阵编程还提供了适用于不同场景的处理能力及多种功能，具体场景和功能列于下表中，详细内容请见后续章节。

**表1 **Matmul特性表

| 特性分类 | 特性描述 | 功能简介 |
| --- | --- | --- |
| 功能实现 | 多核对齐切分 | 在多核场景中，支持将矩阵数据沿M、N、K轴切分，满足M能被singleCoreM整除、N能被singleCoreN整除、K能被singleCoreK整除的对齐场景时的处理方式，从而实现多核并行计算矩阵乘。 |
| 多核非对齐切分 | 在多核场景中，支持将矩阵数据沿M、N、K轴切分。当出现M不能被singleCoreM整除、或N不能被singleCoreN整除、或K不能被singleCoreK整除的非对齐场景（即尾块场景）时的处理方式。 |  |
| 异步场景处理 | MIX场景（包含矩阵计算和矢量计算）下不需要等待矩阵乘计算完成，先执行其它计算。 |  |
| 自定义数据搬入搬出 | 自定义矩阵乘计算前后的数据搬运函数。本功能支持用户实现左矩阵A、右矩阵B从Global Memory分别自定义搬入到A1、B1的过程，输出C矩阵从CO1自定义搬出到Global Memory的过程。 |  |
| 矩阵乘输出的Channel拆分 | 矩阵乘输出的Channel拆分，又称ChannelSplit。指float数据类型、NZ数据格式的输出C矩阵按照16*8的分形大小存储。 |  |
| 矩阵向量乘 | 矩阵向量乘即GEMV，指矩阵乘计算中M=1，K>1的场景，即对形状为(1, K)的左矩阵A实现矩阵乘计算。 |  |
| 上/下三角矩阵乘 | 忽略位于矩阵中下三角或上三角位置的元素的计算，实现矩阵中上三角或下三角位置的元素的矩阵乘计算。 |  |
| TSCM输入的矩阵乘 | 对内存逻辑位置为TSCM的左矩阵A或右矩阵B实现矩阵乘计算。 |  |
| 矩阵乘输出的N方向对齐 | 矩阵乘输出的N方向对齐，又称ND_ALIGN格式输出。指对数据格式为ND_ALIGN的输出C矩阵实现N方向32字节对齐的自动补齐及输出。 |  |
| 单次矩阵乘局部输出 | 单次矩阵乘局部输出，又称Partial Output，指矩阵乘计算时不对单核K方向的计算结果做累加，直接输出计算结果。 |  |
| AIC和AIV独立运行机制 | AIC和AIV独立运行机制，又称双主模式。MIX场景（包含矩阵计算和矢量计算）下AIC核和AIV核独立运行代码，不依赖消息驱动。 |  |

**表2 **Matmul特性表

| 特性分类 | 特性描述 | 功能简介 |
| --- | --- | --- |
| 功能实现 | 矩阵乘输出的量化/反量化 | 将矩阵乘的计算结果从CO1搬出到Global Memory时，对矩阵元素执行数据量化或反量化操作。 |
| 4:2稀疏矩阵乘 | 4:2稀疏矩阵乘，又称Sparse Matmul。指对稀疏左矩阵A和4:2稠密化的右矩阵B实现矩阵乘计算。 |  |

**表3 **BatchMatmul特性表

| 特性分类 | 特性描述 | 功能简介 |
| --- | --- | --- |
| 功能实现 | Batch Matmul基础功能 | Batch Matmul基础功能，支持批量处理Matmul，调用一次IterateBatch接口，计算出多个singleCoreM * singleCoreN大小的C矩阵。 |
| Batch Matmul复用Bias矩阵 | 每个Batch的Matmul计算复用同一个不带Batch轴的Bias矩阵。 |  |
