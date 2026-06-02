# Matmul性能优化策略总览

**页面ID:** atlas_ascendc_best_practices_10_10006  
**来源:** https://www.hiascend.com/document/detail/zh/CANNCommunityEdition/850/opdevg/Ascendcopdevg/atlas_ascendc_best_practices_10_10006.html

---

本节提供了一系列包含Matmul计算的算子性能调优案例，开发者可根据实际应用场景，参考相关案例中的优化方法和思路，应用于具体实践中。案例分类及简介请参见下表，详细内容请阅读后续章节。

**表1 **Matmul性能优化策略总览表

| 分类 | 子分类 | 适用场景 | 相关案例 |
| --- | --- | --- | --- |
| Tiling优化 | Tiling优化：优化Tiling分核及基本块切分策略。 | 数据量足够多的大Shape场景。 | - Matmul算子优化Tiling策略 |
| 并行度优化 | 多核间任务并行：合理地将数据分配给不同的核来执行任务。 | 矩阵的K轴较大、M轴和N轴相比K轴较小的场景。 | - Matmul高阶API使能多核切K |
| 多核间数据访问并行：优化多核数据并行访问机制，如多核场景同一内存数据的地址访问冲突优化，实现多核数据访问效率提升。 | 多核执行Matmul，输入矩阵的K轴较大且K轴非全载的场景。 | - Matmul高阶API使能多核K轴错峰访问内存 |  |
| 单核内流水并行：利用不同指令队列间的相互独立性和可并行执行特性，优化核内流水并行度。 | 算子的MMAD流水和FIXPIPE流水之间串行执行，同步等待的时间在算子整体执行耗时中占比较高。 | - Matmul高阶API使能UnitFlag |  |
| MTE2 Bound且MTE2流水和其他流水串行执行。 | - Matmul高阶API使能NBuffer33模板 |  |  |
| 内存优化 | 内存共享与复用：通过Buffer的共享与缓存复用，减少重复的数据搬运带来的开销。 | MIX场景下，多个AIV的A矩阵或B矩阵GM地址相同，且多个AIV复用的A矩阵或B矩阵在L1 Buffer上全载。 | - Matmul高阶API使能IBShare模板共享A和B矩阵数据 - Matmul高阶API使能IBShare模板共享B矩阵数据 |
| 内存对齐：确保处理的数据满足特定的对齐要求，针对非对齐数据使用不同的搬运策略，以提升数据搬运的效率。 | 输入矩阵内轴非256字节对齐，且数据量较大的场景。 | - AIV核上的ND2NZ格式转换 |  |
| Scalar优化 | Tiling常量化：在Kernel编译期间完成Matmul Tiling的计算，由变量转化为常量扩散到系统中，减少Scalar提升性能。 | - Matmul初始化时的Scalar计算较多，影响指令头开销。- Matmul迭代之间的Scalar计算较多，阻塞MTE2流水。 | - Matmul高阶API使能Tiling全量常量化 |
| 纯Cube模式：减少消息处理机制带来额外的Scalar开销。 | 相较于MIX模式，没有矢量计算，只有矩阵计算的场景。 | - Matmul高阶API使能纯Cube模式 |  |
| 搬运优化 | 搬运吞吐量优化：通过合理控制搬运数据块的大小，提升带宽利用效率，实现搬运效率的提升。 | MTE2循环搬运次数多的大shape场景。 | - Matmul 高阶API使能MDL模板 |
| 输入和输出的数据量超过L2 Cache大小的场景。 | - Matmul高阶API使能L2 Cache切分 |  |  |
| 预加载搬运：预加载需要搬运的数据块，减少流水之间的间隙。 | MTE2流水间隙较大，且M或N数值较大的场景。 | - Matmul高阶API使能MTE2 Preload |  |
