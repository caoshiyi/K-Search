# Matmul性能优化策略总览<a name="ZH-CN_TOPIC_0000002523129074"></a>

本节提供了一系列包含Matmul计算的算子性能调优案例，开发者可根据实际应用场景，参考相关案例中的优化方法和思路，应用于具体实践中。案例分为如下五类，各分类的简介请参见如下表格，详细内容请阅读后续章节。

-   Tiling优化

    **表 1**  Tiling优化策略总览

    <a name="table11377439144919"></a>
    <table><thead align="left"><tr id="row7377639104916"><th class="cellrowborder" valign="top" width="30.086991300869915%" id="mcps1.2.4.1.1"><p id="p4377123917495"><a name="p4377123917495"></a><a name="p4377123917495"></a>分类</p>
    </th>
    <th class="cellrowborder" valign="top" width="40.82591740825918%" id="mcps1.2.4.1.2"><p id="p2377193920493"><a name="p2377193920493"></a><a name="p2377193920493"></a>适用场景</p>
    </th>
    <th class="cellrowborder" valign="top" width="29.087091290870916%" id="mcps1.2.4.1.3"><p id="p1377173904916"><a name="p1377173904916"></a><a name="p1377173904916"></a>相关案例</p>
    </th>
    </tr>
    </thead>
    <tbody><tr id="row1737712393496"><td class="cellrowborder" valign="top" width="30.086991300869915%" headers="mcps1.2.4.1.1 "><p id="p13781393494"><a name="p13781393494"></a><a name="p13781393494"></a>Tiling优化：优化Tiling分核及基本块切分策略。</p>
    </td>
    <td class="cellrowborder" valign="top" width="40.82591740825918%" headers="mcps1.2.4.1.2 "><p id="p337893924919"><a name="p337893924919"></a><a name="p337893924919"></a>数据量足够多的大Shape场景。</p>
    </td>
    <td class="cellrowborder" valign="top" width="29.087091290870916%" headers="mcps1.2.4.1.3 "><p id="p123781939164916"><a name="p123781939164916"></a><a name="p123781939164916"></a><a href="Matmul算子优化Tiling策略.md">Matmul算子优化Tiling策略</a></p>
    </td>
    </tr>
    </tbody>
    </table>

-   并行度优化

    **表 2**  并行度优化策略总览

    <a name="table225712267501"></a>
    <table><thead align="left"><tr id="row18257182614507"><th class="cellrowborder" valign="top" width="30.086991300869915%" id="mcps1.2.4.1.1"><p id="p1925717267508"><a name="p1925717267508"></a><a name="p1925717267508"></a>分类</p>
    </th>
    <th class="cellrowborder" valign="top" width="40.82591740825918%" id="mcps1.2.4.1.2"><p id="p182571626155016"><a name="p182571626155016"></a><a name="p182571626155016"></a>适用场景</p>
    </th>
    <th class="cellrowborder" valign="top" width="29.087091290870916%" id="mcps1.2.4.1.3"><p id="p325710262505"><a name="p325710262505"></a><a name="p325710262505"></a>相关案例</p>
    </th>
    </tr>
    </thead>
    <tbody><tr id="row9258226145013"><td class="cellrowborder" valign="top" width="30.086991300869915%" headers="mcps1.2.4.1.1 "><p id="p1425862615018"><a name="p1425862615018"></a><a name="p1425862615018"></a>多核间任务并行：合理地将数据分配给不同的核来执行任务。</p>
    </td>
    <td class="cellrowborder" valign="top" width="40.82591740825918%" headers="mcps1.2.4.1.2 "><p id="p5258326185015"><a name="p5258326185015"></a><a name="p5258326185015"></a>矩阵的K轴较大、M轴和N轴相比K轴较小的场景。</p>
    </td>
    <td class="cellrowborder" valign="top" width="29.087091290870916%" headers="mcps1.2.4.1.3 "><p id="p1625832617508"><a name="p1625832617508"></a><a name="p1625832617508"></a><a href="Matmul高阶API使能多核切K.md">Matmul高阶API使能多核切K</a></p>
    </td>
    </tr>
    <tr id="row1225815262506"><td class="cellrowborder" valign="top" width="30.086991300869915%" headers="mcps1.2.4.1.1 "><p id="p8258152625011"><a name="p8258152625011"></a><a name="p8258152625011"></a>多核间数据访问并行：优化多核数据并行访问机制，如多核场景同一内存数据的地址访问冲突优化，实现多核数据访问效率提升。</p>
    </td>
    <td class="cellrowborder" valign="top" width="40.82591740825918%" headers="mcps1.2.4.1.2 "><p id="p162587262503"><a name="p162587262503"></a><a name="p162587262503"></a>多核执行Matmul，输入矩阵的K轴较大且K轴非全载的场景。</p>
    </td>
    <td class="cellrowborder" valign="top" width="29.087091290870916%" headers="mcps1.2.4.1.3 "><p id="p192580261505"><a name="p192580261505"></a><a name="p192580261505"></a><a href="Matmul高阶API使能多核K轴错峰访问内存.md">Matmul高阶API使能多核K轴错峰访问内存</a></p>
    </td>
    </tr>
    <tr id="row8258126165014"><td class="cellrowborder" valign="top" width="30.086991300869915%" headers="mcps1.2.4.1.1 "><p id="p8258192614509"><a name="p8258192614509"></a><a name="p8258192614509"></a>单核内流水并行：利用不同指令队列间的相互独立性和可并行执行特性，优化核内流水并行度。</p>
    <p id="p1025832613508"><a name="p1025832613508"></a><a name="p1025832613508"></a></p>
    </td>
    <td class="cellrowborder" valign="top" width="40.82591740825918%" headers="mcps1.2.4.1.2 "><a name="ol68291208611"></a><a name="ol68291208611"></a><ol id="ol68291208611"><li>算子的MMAD流水和FIXPIPE流水之间串行执行，同步等待的时间在算子整体执行耗时中占比较高。</li><li>MTE2 Bound且MTE2流水和其他流水串行执行。</li></ol>
    </td>
    <td class="cellrowborder" valign="top" width="29.087091290870916%" headers="mcps1.2.4.1.3 "><a name="ol224351311615"></a><a name="ol224351311615"></a><ol id="ol224351311615"><li><a href="Matmul高阶API使能UnitFlag.md">Matmul高阶API使能UnitFlag</a></li><li><a href="Matmul高阶API使能NBuffer33模板.md">Matmul高阶API使能NBuffer33模板</a></li></ol>
    </td>
    </tr>
    </tbody>
    </table>

-   内存优化

    **表 3**  内存优化策略总览

    <a name="table136011854115018"></a>
    <table><thead align="left"><tr id="row1060112547508"><th class="cellrowborder" valign="top" width="30.086991300869915%" id="mcps1.2.4.1.1"><p id="p860113549502"><a name="p860113549502"></a><a name="p860113549502"></a>分类</p>
    </th>
    <th class="cellrowborder" valign="top" width="40.82591740825918%" id="mcps1.2.4.1.2"><p id="p460185475011"><a name="p460185475011"></a><a name="p460185475011"></a>适用场景</p>
    </th>
    <th class="cellrowborder" valign="top" width="29.087091290870916%" id="mcps1.2.4.1.3"><p id="p4601115413505"><a name="p4601115413505"></a><a name="p4601115413505"></a>相关案例</p>
    </th>
    </tr>
    </thead>
    <tbody><tr id="row1760225445018"><td class="cellrowborder" valign="top" width="30.086991300869915%" headers="mcps1.2.4.1.1 "><p id="p1960215413501"><a name="p1960215413501"></a><a name="p1960215413501"></a>内存共享与复用：通过Buffer的共享与缓存复用，减少重复的数据搬运带来的开销。</p>
    </td>
    <td class="cellrowborder" valign="top" width="40.82591740825918%" headers="mcps1.2.4.1.2 "><p id="p196026548501"><a name="p196026548501"></a><a name="p196026548501"></a>MIX场景下，多个AIV的A矩阵或B矩阵GM地址相同，且多个AIV复用的A矩阵或B矩阵在L1 Buffer上全载。</p>
    </td>
    <td class="cellrowborder" valign="top" width="29.087091290870916%" headers="mcps1.2.4.1.3 "><p id="p1160255465017"><a name="p1160255465017"></a><a name="p1160255465017"></a><a href="Matmul高阶API使能IBShare模板共享A和B矩阵数据.md">Matmul高阶API使能IBShare模板共享A和B矩阵数据</a></p>
    <p id="p8602185495016"><a name="p8602185495016"></a><a name="p8602185495016"></a><a href="Matmul高阶API使能IBShare模板共享B矩阵数据.md">Matmul高阶API使能IBShare模板共享B矩阵数据</a></p>
    </td>
    </tr>
    <tr id="row19602135415012"><td class="cellrowborder" valign="top" width="30.086991300869915%" headers="mcps1.2.4.1.1 "><p id="p1460235415508"><a name="p1460235415508"></a><a name="p1460235415508"></a>内存对齐：确保处理的数据满足特定的对齐要求，针对非对齐数据使用不同的搬运策略，以提升数据搬运的效率。</p>
    </td>
    <td class="cellrowborder" valign="top" width="40.82591740825918%" headers="mcps1.2.4.1.2 "><p id="p4602554135017"><a name="p4602554135017"></a><a name="p4602554135017"></a>输入矩阵内轴非256字节对齐，且数据量较大的场景。</p>
    </td>
    <td class="cellrowborder" valign="top" width="29.087091290870916%" headers="mcps1.2.4.1.3 "><p id="p16602754195019"><a name="p16602754195019"></a><a name="p16602754195019"></a><a href="AIV核上的ND2NZ格式转换.md">AIV核上的ND2NZ格式转换</a></p>
    </td>
    </tr>
    </tbody>
    </table>

-   Scalar优化

    **表 4**  Scalar优化策略总览

    <a name="table214614239517"></a>
    <table><thead align="left"><tr id="row514716230519"><th class="cellrowborder" valign="top" width="30.086991300869915%" id="mcps1.2.4.1.1"><p id="p16147112355116"><a name="p16147112355116"></a><a name="p16147112355116"></a>分类</p>
    </th>
    <th class="cellrowborder" valign="top" width="40.82591740825918%" id="mcps1.2.4.1.2"><p id="p12147112318519"><a name="p12147112318519"></a><a name="p12147112318519"></a>适用场景</p>
    </th>
    <th class="cellrowborder" valign="top" width="29.087091290870916%" id="mcps1.2.4.1.3"><p id="p7147112365112"><a name="p7147112365112"></a><a name="p7147112365112"></a>相关案例</p>
    </th>
    </tr>
    </thead>
    <tbody><tr id="row1414892319513"><td class="cellrowborder" valign="top" width="30.086991300869915%" headers="mcps1.2.4.1.1 "><p id="p5148162310512"><a name="p5148162310512"></a><a name="p5148162310512"></a>Tiling常量化：在Kernel编译期间完成Matmul Tiling的计算，由变量转化为常量扩散到系统中，减少Scalar提升性能。</p>
    </td>
    <td class="cellrowborder" valign="top" width="40.82591740825918%" headers="mcps1.2.4.1.2 "><p id="p147291104560"><a name="p147291104560"></a><a name="p147291104560"></a>Matmul初始化时的Scalar计算较多，影响指令头开销。</p>
    <p id="p1563512345614"><a name="p1563512345614"></a><a name="p1563512345614"></a>Matmul迭代之间的Scalar计算较多，阻塞MTE2流水。</p>
    </td>
    <td class="cellrowborder" valign="top" width="29.087091290870916%" headers="mcps1.2.4.1.3 "><p id="p12148182365118"><a name="p12148182365118"></a><a name="p12148182365118"></a><a href="Matmul高阶API使能Tiling全量常量化.md">Matmul高阶API使能Tiling全量常量化</a></p>
    </td>
    </tr>
    <tr id="row16148523155115"><td class="cellrowborder" valign="top" width="30.086991300869915%" headers="mcps1.2.4.1.1 "><p id="p514812319512"><a name="p514812319512"></a><a name="p514812319512"></a>纯Cube模式：减少消息处理机制带来额外的Scalar开销。</p>
    </td>
    <td class="cellrowborder" valign="top" width="40.82591740825918%" headers="mcps1.2.4.1.2 "><p id="p1914812237516"><a name="p1914812237516"></a><a name="p1914812237516"></a>相较于MIX模式，没有矢量计算，只有矩阵计算的场景。</p>
    </td>
    <td class="cellrowborder" valign="top" width="29.087091290870916%" headers="mcps1.2.4.1.3 "><p id="p17148723205118"><a name="p17148723205118"></a><a name="p17148723205118"></a><a href="Matmul高阶API使能纯Cube模式.md">Matmul高阶API使能纯Cube模式</a></p>
    </td>
    </tr>
    </tbody>
    </table>

-   搬运优化

    **表 5**  搬运优化策略总览

    <a name="table2401471518"></a>
    <table><thead align="left"><tr id="row44094713510"><th class="cellrowborder" valign="top" width="30.086991300869915%" id="mcps1.2.4.1.1"><p id="p114013470517"><a name="p114013470517"></a><a name="p114013470517"></a>分类</p>
    </th>
    <th class="cellrowborder" valign="top" width="40.82591740825918%" id="mcps1.2.4.1.2"><p id="p164015478513"><a name="p164015478513"></a><a name="p164015478513"></a>适用场景</p>
    </th>
    <th class="cellrowborder" valign="top" width="29.087091290870916%" id="mcps1.2.4.1.3"><p id="p44084775116"><a name="p44084775116"></a><a name="p44084775116"></a>相关案例</p>
    </th>
    </tr>
    </thead>
    <tbody><tr id="row442164713513"><td class="cellrowborder" valign="top" width="30.086991300869915%" headers="mcps1.2.4.1.1 "><p id="p64224795112"><a name="p64224795112"></a><a name="p64224795112"></a>搬运吞吐量优化：通过合理控制搬运数据块的大小，提升带宽利用效率，实现搬运效率的提升。</p>
    </td>
    <td class="cellrowborder" valign="top" width="40.82591740825918%" headers="mcps1.2.4.1.2 "><a name="ol9152163610615"></a><a name="ol9152163610615"></a><ol id="ol9152163610615"><li>MTE2循环搬运次数多的大shape场景。</li><li>输入和输出的数据量超过L2 Cache大小的场景。</li></ol>
    </td>
    <td class="cellrowborder" valign="top" width="29.087091290870916%" headers="mcps1.2.4.1.3 "><a name="ol0907154317615"></a><a name="ol0907154317615"></a><ol id="ol0907154317615"><li><a href="Matmul高阶API使能MDL模板.md">Matmul 高阶API使能MDL模板</a></li><li><a href="Matmul高阶API使能L2-Cache切分.md">Matmul高阶API使能L2 Cache切分</a></li></ol>
    </td>
    </tr>
    <tr id="row1542174715511"><td class="cellrowborder" valign="top" width="30.086991300869915%" headers="mcps1.2.4.1.1 "><p id="p10421347185119"><a name="p10421347185119"></a><a name="p10421347185119"></a>预加载搬运：预加载需要搬运的数据块，减少流水之间的间隙。</p>
    </td>
    <td class="cellrowborder" valign="top" width="40.82591740825918%" headers="mcps1.2.4.1.2 "><p id="p14284716513"><a name="p14284716513"></a><a name="p14284716513"></a>MTE2流水间隙较大，且M或N数值较大的场景。</p>
    </td>
    <td class="cellrowborder" valign="top" width="29.087091290870916%" headers="mcps1.2.4.1.3 "><p id="p042947135117"><a name="p042947135117"></a><a name="p042947135117"></a><a href="Matmul高阶API使能MTE2-Preload.md">Matmul高阶API使能MTE2 Preload</a></p>
    </td>
    </tr>
    </tbody>
    </table>

