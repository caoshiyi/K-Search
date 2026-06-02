# Ascend C API列表<a name="ZH-CN_TOPIC_0000002523304738"></a>

Ascend C提供一组类库API，开发者使用标准C++语法和类库API进行编程。Ascend C编程类库API示意图如下所示，分为：

-   **基础数据结构**：kernel API中使用到的基础数据结构，比如GlobalTensor和LocalTensor。
-   **基础API**：实现对硬件能力的抽象，开放芯片的能力，保证完备性和兼容性。标注为ISASI（Instruction Set Architecture Special Interface，硬件体系结构相关的接口）类别的API，不能保证跨硬件版本兼容。
-   **高阶API**：实现一些常用的计算算法，用于提高编程开发效率，通常会调用多种基础API实现。高阶API包括数学库、Matmul、Softmax等API。高阶API可以保证兼容性。
-   **SIMT API**：单指令多线程API。以单条指令多个线程的形式来实现并行计算。SIMT编程主要用于向量计算，特别适合处理离散访问、复杂控制逻辑等场景。
-   **Utils API（公共辅助函数）**：丰富的通用工具类，涵盖标准库、平台信息获取、运行时编译及日志输出等功能，支持开发者高效实现算子开发与性能优化。

<!-- img2text -->
```text
┌──────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────┐
│                                                     Ascend C                                                                 │
│                                                                                                                              │
│  ┌──────┐  ┌──────────────────────────────────────────────────────────────────────────────────────────────────────────────┐    │
│  │多核  │  │                                                  算子模板库                                                │    │
│  │算子  │  │  ┌──────────────────────────────┐   ┌──────────────────────────────────────┐                             │    │
│  │样例  │  │  │ Cube类模板库（ CATLASS ）    │   │ Vector类模板库（ ATVC/ATVOSS ）      │                             │    │
│  └──────┘  └──────────────────────────────────────────────────────────────────────────────────────────────────────────────┘    │
│                                                                                                                              │
│  ┌──────┐  ┌──────────────────────────────────────────────────────────────────────────────────────────────────────────────┐    │
│  │单核  │  │                                                    高阶API                                                  │    │
│  │公共  │  │  ┌────────┐ ┌────────┐ ┌────────┐ ┌────────┐ ┌────────┐ ┌────────┐ ┌────┐                               │    │
│  │算法  │  │  │数学计算│ │矩阵计算│ │激活函数│ │池化计算│ │索引计算│ │通信编程│ │ ...│                               │    │
│  └──────┘  │  └────────┘ └────────┘ └────────┘ └────────┘ └────────┘ └────────┘ └────┘                               │    │
│            └──────────────────────────────────────────────────────────────────────────────────────────────────────────────┘    │
│                                                                                                                              │
│  ┌──────┐  ┌─────────────────────────────────────────────────────────────────────────────────────────────┐  ┌────────────┐   │
│  │单指令│  │                                           基础API（ SIMD ）                                  │  │ SIMT类库   │   │
│  │      │  │  ┌────────────┐ ┌────────────┐ ┌──────────┐                                               │  │ （待支持） │   │
│  │      │  │  │Memory数据搬运│ │Memory矢量计算│ │Reg矢量计算│                                               │  └────────────┘   │
│  │      │  │  └────────────┘ └────────────┘ └──────────┘                                               │                   │
│  │      │  │  ┌────────┐ ┌────────┐ ┌────────┐ ┌────────┐ ┌────┐                                       │                   │
│  │      │  │  │矩阵计算│ │资源管理│ │同步控制│ │缓存控制│ │ ...│                                       │                   │
│  └──────┘  └─────────────────────────────────────────────────────────────────────────────────────────────┘                   │
│                                                                                                                              │
│  类库                                                                                                                        │
│                                                                                                                              │
│  ┌────────┐  ┌──────────────────────────────────────────────────────────────────────────────────────────────────────────────┐ │
│  │语言    │  │  ┌──────────────────────────────────┐   ┌──────────────────────────────────┐                             │ │
│  │扩展层  │  │  │ Ascend C拓展的C API（ SIMD ）    │   │ Ascend C拓展的C API（ SIMT ）    │                             │ │
│  │        │  │  │ （待支持）                       │   │                                  │                             │ │
│  └────────┘  └──────────────────────────────────────────────────────────────────────────────────────────────────────────────┘ │
│                                                                                                                              │
│                                                                 ┌────────────┐  ┌────────────┐  ┌────────────┐               │
│                                                                 │ 公共辅助函数 │  │ 算子工程   │  │ 调试调优工具链 │               │
│                                                                 │            │  │ 编译脚本   │  │            │               │
│                                                                 └────────────┘  └────────────┘  └────────────┘               │
└──────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────┘
```

## 基础数据结构<a name="section3119442205518"></a>

**表 1**  基础数据结构列表

<a name="table16268175715515"></a>
<table><thead align="left"><tr id="row11268657105518"><th class="cellrowborder" valign="top" width="40.37%" id="mcps1.2.3.1.1"><p id="p1026810577553"><a name="p1026810577553"></a><a name="p1026810577553"></a>接口名</p>
</th>
<th class="cellrowborder" valign="top" width="59.63%" id="mcps1.2.3.1.2"><p id="p226835735518"><a name="p226835735518"></a><a name="p226835735518"></a>功能描述</p>
</th>
</tr>
</thead>
<tbody><tr id="row102681557105520"><td class="cellrowborder" valign="top" width="40.37%" headers="mcps1.2.3.1.1 "><p id="p12251181311560"><a name="p12251181311560"></a><a name="p12251181311560"></a><a href="LocalTensor.md">LocalTensor</a></p>
</td>
<td class="cellrowborder" valign="top" width="59.63%" headers="mcps1.2.3.1.2 "><p id="p2025215484579"><a name="p2025215484579"></a><a name="p2025215484579"></a><span id="ph4396131516585"><a name="ph4396131516585"></a><a name="ph4396131516585"></a>LocalTensor用于存放AI Core中Local Memory（内部存储）的数据，支持逻辑位置<a href="TPosition.md">TPosition</a>为<span>VECIN、VECOUT、VECCALC、</span>A1<span>、</span>A2<span>、</span>B1<span>、</span>B2<span>、</span>CO1<span>、</span>CO2。</span></p>
</td>
</tr>
<tr id="row026925718557"><td class="cellrowborder" valign="top" width="40.37%" headers="mcps1.2.3.1.1 "><p id="p142691057115518"><a name="p142691057115518"></a><a name="p142691057115518"></a><a href="GlobalTensor.md">GlobalTensor</a></p>
</td>
<td class="cellrowborder" valign="top" width="59.63%" headers="mcps1.2.3.1.2 "><p id="p625254835716"><a name="p625254835716"></a><a name="p625254835716"></a><span id="ph698815359590"><a name="ph698815359590"></a><a name="ph698815359590"></a>GlobalTensor用来存放Global Memory（外部存储）的全局数据。</span></p>
</td>
</tr>
<tr id="row32691573557"><td class="cellrowborder" valign="top" width="40.37%" headers="mcps1.2.3.1.1 "><p id="p8269195705514"><a name="p8269195705514"></a><a name="p8269195705514"></a><a href="Coordinate.md">Coordinate</a></p>
</td>
<td class="cellrowborder" valign="top" width="59.63%" headers="mcps1.2.3.1.2 "><p id="p1825015488573"><a name="p1825015488573"></a><a name="p1825015488573"></a>Coordinate本质上是一个元组（tuple），用于表示张量在不同维度的位置信息，即坐标值。</p>
</td>
</tr>
<tr id="row926915745510"><td class="cellrowborder" valign="top" width="40.37%" headers="mcps1.2.3.1.1 "><p id="p192691757185519"><a name="p192691757185519"></a><a name="p192691757185519"></a><a href="Layout.md">Layout</a></p>
</td>
<td class="cellrowborder" valign="top" width="59.63%" headers="mcps1.2.3.1.2 "><p id="p1724954818576"><a name="p1724954818576"></a><a name="p1724954818576"></a><span id="ph198761643125913"><a name="ph198761643125913"></a><a name="ph198761643125913"></a>Layout&lt;Shape, Stride&gt;数据结构是描述多维张量内存布局的基础模板类，通过编译时的形状（Shape）和步长（Stride）信息，实现逻辑坐标空间到一维内存地址空间的映射，为复杂张量操作和硬件优化提供基础支持。</span></p>
</td>
</tr>
<tr id="row97101411165713"><td class="cellrowborder" valign="top" width="40.37%" headers="mcps1.2.3.1.1 "><p id="p5710711115712"><a name="p5710711115712"></a><a name="p5710711115712"></a><a href="TensorTrait.md">TensorTrait</a></p>
</td>
<td class="cellrowborder" valign="top" width="59.63%" headers="mcps1.2.3.1.2 "><p id="p167101111155717"><a name="p167101111155717"></a><a name="p167101111155717"></a><span id="ph287618459597"><a name="ph287618459597"></a><a name="ph287618459597"></a>TensorTrait数据结构是描述Tensor相关信息的基础模板类，包含Tensor的数据类型、逻辑位置和Layout内存布局。</span></p>
</td>
</tr>
</tbody>
</table>

## 基础API<a name="section117632211201"></a>

**表 2**  Memory数据搬运API列表

<a name="table1199372172410"></a>
<table><thead align="left"><tr id="row69936217246"><th class="cellrowborder" valign="top" width="40.37%" id="mcps1.2.3.1.1"><p id="p1799422162414"><a name="p1799422162414"></a><a name="p1799422162414"></a>接口名</p>
</th>
<th class="cellrowborder" valign="top" width="59.63%" id="mcps1.2.3.1.2"><p id="p89941221202417"><a name="p89941221202417"></a><a name="p89941221202417"></a>功能描述</p>
</th>
</tr>
</thead>
<tbody><tr id="row19994142132410"><td class="cellrowborder" valign="top" width="40.37%" headers="mcps1.2.3.1.1 "><p id="p119488415242"><a name="p119488415242"></a><a name="p119488415242"></a><a href="DataCopy.md">DataCopy</a></p>
</td>
<td class="cellrowborder" valign="top" width="59.63%" headers="mcps1.2.3.1.2 "><p id="p1188623312246"><a name="p1188623312246"></a><a name="p1188623312246"></a>数据搬运接口，包括普通数据搬运、增强数据搬运、切片数据搬运、随路格式转换。</p>
</td>
</tr>
<tr id="row29942216241"><td class="cellrowborder" valign="top" width="40.37%" headers="mcps1.2.3.1.1 "><p id="p138851338243"><a name="p138851338243"></a><a name="p138851338243"></a><a href="Copy.md">Copy</a></p>
</td>
<td class="cellrowborder" valign="top" width="59.63%" headers="mcps1.2.3.1.2 "><p id="p1488517337242"><a name="p1488517337242"></a><a name="p1488517337242"></a>VECIN、VECCALC、VECOUT之间的搬运指令，支持mask操作和<span id="ph1256166185416"><a name="ph1256166185416"></a><a name="ph1256166185416"></a>DataBlock</span>间隔操作。</p>
</td>
</tr>
</tbody>
</table>

**表 3**  Memory矢量计算API列表

<a name="table107281858237"></a>
<table><thead align="left"><tr id="row1372812592319"><th class="cellrowborder" valign="top" width="15.590000000000002%" id="mcps1.2.4.1.1"><p id="p28543193914"><a name="p28543193914"></a><a name="p28543193914"></a>分类</p>
</th>
<th class="cellrowborder" valign="top" width="24.64%" id="mcps1.2.4.1.2"><p id="p147285552316"><a name="p147285552316"></a><a name="p147285552316"></a>接口名</p>
</th>
<th class="cellrowborder" valign="top" width="59.77%" id="mcps1.2.4.1.3"><p id="p17281151239"><a name="p17281151239"></a><a name="p17281151239"></a>功能描述</p>
</th>
</tr>
</thead>
<tbody><tr id="row1972815510234"><td class="cellrowborder" rowspan="18" valign="top" width="15.590000000000002%" headers="mcps1.2.4.1.1 "><p id="p28542192920"><a name="p28542192920"></a><a name="p28542192920"></a>基础算术</p>
</td>
<td class="cellrowborder" valign="top" width="24.64%" headers="mcps1.2.4.1.2 "><p id="p472817542311"><a name="p472817542311"></a><a name="p472817542311"></a><a href="Exp.md">Exp</a></p>
</td>
<td class="cellrowborder" valign="top" width="59.77%" headers="mcps1.2.4.1.3 "><p id="p14728115122318"><a name="p14728115122318"></a><a name="p14728115122318"></a>按元素取自然指数。</p>
</td>
</tr>
<tr id="row77297582318"><td class="cellrowborder" valign="top" headers="mcps1.2.4.1.1 "><p id="p74127324439"><a name="p74127324439"></a><a name="p74127324439"></a><a href="Ln.md">Ln</a></p>
</td>
<td class="cellrowborder" valign="top" headers="mcps1.2.4.1.2 "><p id="p127291058234"><a name="p127291058234"></a><a name="p127291058234"></a>按元素取自然对数。</p>
</td>
</tr>
<tr id="row095531611435"><td class="cellrowborder" valign="top" headers="mcps1.2.4.1.1 "><p id="p441216325433"><a name="p441216325433"></a><a name="p441216325433"></a><a href="Abs.md">Abs</a></p>
</td>
<td class="cellrowborder" valign="top" headers="mcps1.2.4.1.2 "><p id="p179557168437"><a name="p179557168437"></a><a name="p179557168437"></a>按元素取绝对值。</p>
</td>
</tr>
<tr id="row1698614591910"><td class="cellrowborder" valign="top" headers="mcps1.2.4.1.1 "><p id="p7412103212437"><a name="p7412103212437"></a><a name="p7412103212437"></a><a href="Reciprocal.md">Reciprocal</a></p>
</td>
<td class="cellrowborder" valign="top" headers="mcps1.2.4.1.2 "><p id="p698612594920"><a name="p698612594920"></a><a name="p698612594920"></a>按元素取倒数。</p>
</td>
</tr>
<tr id="row204721719184318"><td class="cellrowborder" valign="top" headers="mcps1.2.4.1.1 "><p id="p7412103218431"><a name="p7412103218431"></a><a name="p7412103218431"></a><a href="Sqrt.md">Sqrt</a></p>
</td>
<td class="cellrowborder" valign="top" headers="mcps1.2.4.1.2 "><p id="p144735190439"><a name="p144735190439"></a><a name="p144735190439"></a>按元素做开方。</p>
</td>
</tr>
<tr id="row1263518197432"><td class="cellrowborder" valign="top" headers="mcps1.2.4.1.1 "><p id="p134122326433"><a name="p134122326433"></a><a name="p134122326433"></a><a href="Rsqrt.md">Rsqrt</a></p>
</td>
<td class="cellrowborder" valign="top" headers="mcps1.2.4.1.2 "><p id="p1163651916435"><a name="p1163651916435"></a><a name="p1163651916435"></a>按元素做开方后取倒数。</p>
</td>
</tr>
<tr id="row11951420124314"><td class="cellrowborder" valign="top" headers="mcps1.2.4.1.1 "><p id="p8412143274317"><a name="p8412143274317"></a><a name="p8412143274317"></a><a href="Relu.md">Relu</a></p>
</td>
<td class="cellrowborder" valign="top" headers="mcps1.2.4.1.2 "><p id="p19951020134315"><a name="p19951020134315"></a><a name="p19951020134315"></a>按元素做线性整流Relu。</p>
</td>
</tr>
<tr id="row687313439911"><td class="cellrowborder" valign="top" headers="mcps1.2.4.1.1 "><p id="p178741943997"><a name="p178741943997"></a><a name="p178741943997"></a><a href="Add.md">Add</a></p>
</td>
<td class="cellrowborder" valign="top" headers="mcps1.2.4.1.2 "><p id="p58747431191"><a name="p58747431191"></a><a name="p58747431191"></a>按元素求和。</p>
</td>
</tr>
<tr id="row1874810244362"><td class="cellrowborder" valign="top" headers="mcps1.2.4.1.1 "><p id="p1912104816385"><a name="p1912104816385"></a><a name="p1912104816385"></a><a href="Sub.md">Sub</a></p>
</td>
<td class="cellrowborder" valign="top" headers="mcps1.2.4.1.2 "><p id="p107491245366"><a name="p107491245366"></a><a name="p107491245366"></a>按元素求差。</p>
</td>
</tr>
<tr id="row1291015249364"><td class="cellrowborder" valign="top" headers="mcps1.2.4.1.1 "><p id="p2121164811382"><a name="p2121164811382"></a><a name="p2121164811382"></a><a href="Mul.md">Mul</a></p>
</td>
<td class="cellrowborder" valign="top" headers="mcps1.2.4.1.2 "><p id="p791052483612"><a name="p791052483612"></a><a name="p791052483612"></a>按元素求积。</p>
</td>
</tr>
<tr id="row81121256363"><td class="cellrowborder" valign="top" headers="mcps1.2.4.1.1 "><p id="p11121548153820"><a name="p11121548153820"></a><a name="p11121548153820"></a><a href="Div.md">Div</a></p>
</td>
<td class="cellrowborder" valign="top" headers="mcps1.2.4.1.2 "><p id="p191131256368"><a name="p191131256368"></a><a name="p191131256368"></a>按元素求商。</p>
</td>
</tr>
<tr id="row152552025153613"><td class="cellrowborder" valign="top" headers="mcps1.2.4.1.1 "><p id="p191211748103818"><a name="p191211748103818"></a><a name="p191211748103818"></a><a href="Max.md">Max</a></p>
</td>
<td class="cellrowborder" valign="top" headers="mcps1.2.4.1.2 "><p id="p1325552515362"><a name="p1325552515362"></a><a name="p1325552515362"></a>按元素求最大值。</p>
</td>
</tr>
<tr id="row104017250363"><td class="cellrowborder" valign="top" headers="mcps1.2.4.1.1 "><p id="p1312174833819"><a name="p1312174833819"></a><a name="p1312174833819"></a><a href="Min.md">Min</a></p>
</td>
<td class="cellrowborder" valign="top" headers="mcps1.2.4.1.2 "><p id="p389445174112"><a name="p389445174112"></a><a name="p389445174112"></a>按元素求最小值。</p>
</td>
</tr>
<tr id="row14127182319571"><td class="cellrowborder" valign="top" headers="mcps1.2.4.1.1 "><p id="p1448115232"><a name="p1448115232"></a><a name="p1448115232"></a><a href="Adds.md">Adds</a></p>
</td>
<td class="cellrowborder" valign="top" headers="mcps1.2.4.1.2 "><p id="p884518405416"><a name="p884518405416"></a><a name="p884518405416"></a>矢量内每个元素与标量求和。</p>
</td>
</tr>
<tr id="row107601527165713"><td class="cellrowborder" valign="top" headers="mcps1.2.4.1.1 "><p id="p44471122316"><a name="p44471122316"></a><a name="p44471122316"></a><a href="Muls.md">Muls</a></p>
</td>
<td class="cellrowborder" valign="top" headers="mcps1.2.4.1.2 "><p id="p577119572225"><a name="p577119572225"></a><a name="p577119572225"></a>矢量内每个元素与标量求积。</p>
</td>
</tr>
<tr id="row1277082912579"><td class="cellrowborder" valign="top" headers="mcps1.2.4.1.1 "><p id="p8441215238"><a name="p8441215238"></a><a name="p8441215238"></a><a href="Maxs.md">Maxs</a></p>
</td>
<td class="cellrowborder" valign="top" headers="mcps1.2.4.1.2 "><p id="p18672613113517"><a name="p18672613113517"></a><a name="p18672613113517"></a>源操作数矢量内每个元素与标量相比，如果比标量大，则取源操作数值，比标量的值小，则取标量值。</p>
</td>
</tr>
<tr id="row33851439175710"><td class="cellrowborder" valign="top" headers="mcps1.2.4.1.1 "><p id="p9441515231"><a name="p9441515231"></a><a name="p9441515231"></a><a href="Mins.md">Mins</a></p>
</td>
<td class="cellrowborder" valign="top" headers="mcps1.2.4.1.2 "><p id="p18276162693419"><a name="p18276162693419"></a><a name="p18276162693419"></a>源操作数矢量内每个元素与标量相比，如果比标量大，则取标量值，比标量的值小，则取源操作数值。</p>
</td>
</tr>
<tr id="row547912419575"><td class="cellrowborder" valign="top" headers="mcps1.2.4.1.1 "><p id="p5445162310"><a name="p5445162310"></a><a name="p5445162310"></a><a href="LeakyRelu.md">LeakyRelu</a></p>
</td>
<td class="cellrowborder" valign="top" headers="mcps1.2.4.1.2 "><p id="p27096582222"><a name="p27096582222"></a><a name="p27096582222"></a>按元素做带泄露线性整流Leaky ReLU。</p>
</td>
</tr>
<tr id="row11611570274"><td class="cellrowborder" rowspan="2" valign="top" width="15.590000000000002%" headers="mcps1.2.4.1.1 "><p id="p1521314282290"><a name="p1521314282290"></a><a name="p1521314282290"></a>基础算术</p>
</td>
<td class="cellrowborder" valign="top" width="24.64%" headers="mcps1.2.4.1.2 "><p id="p570714385282"><a name="p570714385282"></a><a name="p570714385282"></a><a href="Subs.md">Subs</a></p>
</td>
<td class="cellrowborder" valign="top" width="59.77%" headers="mcps1.2.4.1.3 "><p id="p1616145717273"><a name="p1616145717273"></a><a name="p1616145717273"></a><span id="ph452415255306"><a name="ph452415255306"></a><a name="ph452415255306"></a>矢量内每个元素和标量间做减法，支持标量在前和标量在后两种场景，其中标量输入支持配置LocalTensor单点元素。</span></p>
</td>
</tr>
<tr id="row13124593270"><td class="cellrowborder" valign="top" headers="mcps1.2.4.1.1 "><p id="p1779354482813"><a name="p1779354482813"></a><a name="p1779354482813"></a><a href="Divs.md">Divs</a></p>
</td>
<td class="cellrowborder" valign="top" headers="mcps1.2.4.1.2 "><p id="p191212595270"><a name="p191212595270"></a><a name="p191212595270"></a><span id="ph58341357153014"><a name="ph58341357153014"></a><a name="ph58341357153014"></a>矢量内每个元素和标量间做除法，支持标量在前和标量在后两种场景，其中标量输入支持配置LocalTensor单点元素。</span></p>
</td>
</tr>
<tr id="row64233134354"><td class="cellrowborder" rowspan="5" valign="top" width="15.590000000000002%" headers="mcps1.2.4.1.1 "><p id="p771519716593"><a name="p771519716593"></a><a name="p771519716593"></a>逻辑计算</p>
</td>
<td class="cellrowborder" valign="top" width="24.64%" headers="mcps1.2.4.1.2 "><p id="p24122327439"><a name="p24122327439"></a><a name="p24122327439"></a><a href="Not.md">Not</a></p>
</td>
<td class="cellrowborder" valign="top" width="59.77%" headers="mcps1.2.4.1.3 "><p id="p1694421944312"><a name="p1694421944312"></a><a name="p1694421944312"></a>按元素做按位取反。</p>
</td>
</tr>
<tr id="row855712515365"><td class="cellrowborder" valign="top" headers="mcps1.2.4.1.1 "><p id="p512110487389"><a name="p512110487389"></a><a name="p512110487389"></a><a href="And.md">And</a></p>
</td>
<td class="cellrowborder" valign="top" headers="mcps1.2.4.1.2 "><p id="p555732515368"><a name="p555732515368"></a><a name="p555732515368"></a>针对每对元素执行按位与运算。</p>
</td>
</tr>
<tr id="row19695152573617"><td class="cellrowborder" valign="top" headers="mcps1.2.4.1.1 "><p id="p12122548183814"><a name="p12122548183814"></a><a name="p12122548183814"></a><a href="Or.md">Or</a></p>
</td>
<td class="cellrowborder" valign="top" headers="mcps1.2.4.1.2 "><p id="p17348124922015"><a name="p17348124922015"></a><a name="p17348124922015"></a>针对每对元素执行按位或运算。</p>
</td>
</tr>
<tr id="row20711575584"><td class="cellrowborder" valign="top" headers="mcps1.2.4.1.1 "><p id="p644313232"><a name="p644313232"></a><a name="p644313232"></a><a href="ShiftLeft.md">ShiftLeft</a></p>
</td>
<td class="cellrowborder" valign="top" headers="mcps1.2.4.1.2 "><p id="p1676331375210"><a name="p1676331375210"></a><a name="p1676331375210"></a>对源操作数中的每个元素进行左移操作，左移的位数由输入参数scalarValue决定。</p>
</td>
</tr>
<tr id="row17751175445811"><td class="cellrowborder" valign="top" headers="mcps1.2.4.1.1 "><p id="p9447119234"><a name="p9447119234"></a><a name="p9447119234"></a><a href="ShiftRight.md">ShiftRight</a></p>
</td>
<td class="cellrowborder" valign="top" headers="mcps1.2.4.1.2 "><p id="p97121651191211"><a name="p97121651191211"></a><a name="p97121651191211"></a>对源操作数中的每个元素进行右移操作，右移的位数由输入参数scalarValue决定。</p>
</td>
</tr>
<tr id="row1739723113211"><td class="cellrowborder" rowspan="2" valign="top" width="15.590000000000002%" headers="mcps1.2.4.1.1 "><p id="p1239733112327"><a name="p1239733112327"></a><a name="p1239733112327"></a>逻辑计算</p>
<p id="p76622335323"><a name="p76622335323"></a><a name="p76622335323"></a></p>
</td>
<td class="cellrowborder" valign="top" width="24.64%" headers="mcps1.2.4.1.2 "><p id="p1691819570327"><a name="p1691819570327"></a><a name="p1691819570327"></a><a href="Ands.md">Ands</a></p>
</td>
<td class="cellrowborder" valign="top" width="59.77%" headers="mcps1.2.4.1.3 "><p id="p3397531133218"><a name="p3397531133218"></a><a name="p3397531133218"></a><span id="ph107741166344"><a name="ph107741166344"></a><a name="ph107741166344"></a>矢量内每个元素和标量间做与操作，支持标量在前和标量在后两种场景，其中标量输入支持配置LocalTensor单点元素。</span></p>
</td>
</tr>
<tr id="row2662163318324"><td class="cellrowborder" valign="top" headers="mcps1.2.4.1.1 "><p id="p12888184123313"><a name="p12888184123313"></a><a name="p12888184123313"></a><a href="Ors.md">Ors</a></p>
</td>
<td class="cellrowborder" valign="top" headers="mcps1.2.4.1.2 "><p id="p566393373216"><a name="p566393373216"></a><a name="p566393373216"></a><span id="ph11500142416349"><a name="ph11500142416349"></a><a name="ph11500142416349"></a>矢量内每个元素和标量间做或操作，支持标量在前和标量在后两种场景，其中标量输入支持配置LocalTensor单点元素。</span></p>
</td>
</tr>
<tr id="row1413722319810"><td class="cellrowborder" rowspan="11" valign="top" width="15.590000000000002%" headers="mcps1.2.4.1.1 "><p id="p12137102320816"><a name="p12137102320816"></a><a name="p12137102320816"></a>复合计算</p>
</td>
<td class="cellrowborder" valign="top" width="24.64%" headers="mcps1.2.4.1.2 "><p id="p1885912584229"><a name="p1885912584229"></a><a name="p1885912584229"></a><a href="Axpy.md">Axpy</a></p>
</td>
<td class="cellrowborder" valign="top" width="59.77%" headers="mcps1.2.4.1.3 "><p id="p1485935816222"><a name="p1485935816222"></a><a name="p1485935816222"></a>源操作数中每个元素与标量求积后和目的操作数中的对应元素相加。</p>
</td>
</tr>
<tr id="row167541318782"><td class="cellrowborder" valign="top" headers="mcps1.2.4.1.1 "><p id="p3734133111451"><a name="p3734133111451"></a><a name="p3734133111451"></a><a href="CastDequant.md">CastDequant</a></p>
</td>
<td class="cellrowborder" valign="top" headers="mcps1.2.4.1.2 "><p id="p1924211214434"><a name="p1924211214434"></a><a name="p1924211214434"></a>对输入做量化并进行精度转换。</p>
</td>
</tr>
<tr id="row15953122533617"><td class="cellrowborder" valign="top" headers="mcps1.2.4.1.1 "><p id="p19122184810389"><a name="p19122184810389"></a><a name="p19122184810389"></a><a href="AddRelu.md">AddRelu</a></p>
</td>
<td class="cellrowborder" valign="top" headers="mcps1.2.4.1.2 "><p id="p39535252363"><a name="p39535252363"></a><a name="p39535252363"></a>按元素求和，结果和0对比取较大值。</p>
</td>
</tr>
<tr id="row16141192653616"><td class="cellrowborder" valign="top" headers="mcps1.2.4.1.1 "><p id="p112214815384"><a name="p112214815384"></a><a name="p112214815384"></a><a href="AddReluCast.md">AddReluCast</a></p>
</td>
<td class="cellrowborder" valign="top" headers="mcps1.2.4.1.2 "><p id="p514172623613"><a name="p514172623613"></a><a name="p514172623613"></a>按元素求和，结果和0对比取较大值，并根据源操作数和目的操作数Tensor的数据类型进行精度转换。</p>
</td>
</tr>
<tr id="row977016176378"><td class="cellrowborder" valign="top" headers="mcps1.2.4.1.1 "><p id="p1312214873810"><a name="p1312214873810"></a><a name="p1312214873810"></a><a href="AddDeqRelu.md">AddDeqRelu</a></p>
</td>
<td class="cellrowborder" valign="top" headers="mcps1.2.4.1.2 "><p id="p1477071717372"><a name="p1477071717372"></a><a name="p1477071717372"></a>依次计算按元素求和、结果进行deq量化后再进行relu计算（结果和0对比取较大值）。</p>
</td>
</tr>
<tr id="row17935161713713"><td class="cellrowborder" valign="top" headers="mcps1.2.4.1.1 "><p id="p1012244843817"><a name="p1012244843817"></a><a name="p1012244843817"></a><a href="SubRelu.md">SubRelu</a></p>
</td>
<td class="cellrowborder" valign="top" headers="mcps1.2.4.1.2 "><p id="p149363171378"><a name="p149363171378"></a><a name="p149363171378"></a>按元素求差，结果和0对比取较大值。</p>
</td>
</tr>
<tr id="row138841816370"><td class="cellrowborder" valign="top" headers="mcps1.2.4.1.1 "><p id="p51221848173810"><a name="p51221848173810"></a><a name="p51221848173810"></a><a href="SubReluCast.md">SubReluCast</a></p>
</td>
<td class="cellrowborder" valign="top" headers="mcps1.2.4.1.2 "><p id="p158818183378"><a name="p158818183378"></a><a name="p158818183378"></a>按元素求差，结果和0对比取较大值，并根据源操作数和目的操作数Tensor的数据类型进行精度转换。</p>
</td>
</tr>
<tr id="row132241218193712"><td class="cellrowborder" valign="top" headers="mcps1.2.4.1.1 "><p id="p212217485389"><a name="p212217485389"></a><a name="p212217485389"></a><a href="MulAddDst.md">MulAddDst</a></p>
</td>
<td class="cellrowborder" valign="top" headers="mcps1.2.4.1.2 "><p id="p5224201818378"><a name="p5224201818378"></a><a name="p5224201818378"></a>按元素将src0Local和src1Local相乘并和dstLocal相加，将最终结果存放进dstLocal中。</p>
</td>
</tr>
<tr id="row117041522154616"><td class="cellrowborder" valign="top" headers="mcps1.2.4.1.1 "><p id="p19705182219464"><a name="p19705182219464"></a><a name="p19705182219464"></a><a href="MulCast.md">MulCast</a></p>
</td>
<td class="cellrowborder" valign="top" headers="mcps1.2.4.1.2 "><p id="p12705182217466"><a name="p12705182217466"></a><a name="p12705182217466"></a>按元素求积，并根据源操作数和目的操作数Tensor的数据类型进行精度转换。</p>
</td>
</tr>
<tr id="row193701818163720"><td class="cellrowborder" valign="top" headers="mcps1.2.4.1.1 "><p id="p91222048133815"><a name="p91222048133815"></a><a name="p91222048133815"></a><a href="FusedMulAdd.md">FusedMulAdd</a></p>
</td>
<td class="cellrowborder" valign="top" headers="mcps1.2.4.1.2 "><p id="p1037019189371"><a name="p1037019189371"></a><a name="p1037019189371"></a>按元素将src0Local和dstLocal相乘并加上src1Local，最终结果存放入dstLocal。</p>
</td>
</tr>
<tr id="row1075883983914"><td class="cellrowborder" valign="top" headers="mcps1.2.4.1.1 "><p id="p19675842114214"><a name="p19675842114214"></a><a name="p19675842114214"></a><a href="MulAddRelu.md">MulAddRelu</a></p>
</td>
<td class="cellrowborder" valign="top" headers="mcps1.2.4.1.2 "><p id="p475993916396"><a name="p475993916396"></a><a name="p475993916396"></a>按元素将src0Local和dstLocal相乘并加上src1Local，将结果和0作比较，取较大值，最终结果存放进dstLocal中。</p>
</td>
</tr>
<tr id="row187959172211"><td class="cellrowborder" rowspan="5" valign="top" width="15.590000000000002%" headers="mcps1.2.4.1.1 "><p id="p147155915229"><a name="p147155915229"></a><a name="p147155915229"></a>比较与选择</p>
<p id="p19980131874015"><a name="p19980131874015"></a><a name="p19980131874015"></a></p>
</td>
<td class="cellrowborder" valign="top" width="24.64%" headers="mcps1.2.4.1.2 "><p id="p157105910222"><a name="p157105910222"></a><a name="p157105910222"></a><a href="Compare.md">Compare</a></p>
</td>
<td class="cellrowborder" valign="top" width="59.77%" headers="mcps1.2.4.1.3 "><p id="p1745910221"><a name="p1745910221"></a><a name="p1745910221"></a>逐元素比较两个tensor大小，如果比较后的结果为真，则输出结果的对应比特位为1，否则为0。</p>
</td>
</tr>
<tr id="row1314575919229"><td class="cellrowborder" valign="top" headers="mcps1.2.4.1.1 "><p id="p1314535915221"><a name="p1314535915221"></a><a name="p1314535915221"></a><a href="Compare（结果存入寄存器）.md">Compare（结果存放入寄存器）</a></p>
</td>
<td class="cellrowborder" valign="top" headers="mcps1.2.4.1.2 "><p id="p177959415503"><a name="p177959415503"></a><a name="p177959415503"></a>逐元素比较两个tensor大小，如果比较后的结果为真，则输出结果的对应比特位为1，否则为0。Compare接口需要mask参数时，可以使用此接口。计算结果存放入寄存器中。</p>
</td>
</tr>
<tr id="row31514186404"><td class="cellrowborder" valign="top" headers="mcps1.2.4.1.1 "><p id="p7151171817406"><a name="p7151171817406"></a><a name="p7151171817406"></a><a href="Compares.md">Compares</a></p>
</td>
<td class="cellrowborder" valign="top" headers="mcps1.2.4.1.2 "><p id="p7227113135415"><a name="p7227113135415"></a><a name="p7227113135415"></a>逐元素比较一个tensor中的元素和另一个Scalar的大小，如果比较后的结果为真，则输出结果的对应比特位为1，否则为0。</p>
</td>
</tr>
<tr id="row1798081810408"><td class="cellrowborder" valign="top" headers="mcps1.2.4.1.1 "><p id="p61901953124213"><a name="p61901953124213"></a><a name="p61901953124213"></a><a href="Select.md">Select</a></p>
</td>
<td class="cellrowborder" valign="top" headers="mcps1.2.4.1.2 "><p id="p298081819406"><a name="p298081819406"></a><a name="p298081819406"></a>给定两个源操作数src0和src1，根据selMask（用于选择的Mask掩码）的比特位值选取元素，得到目的操作数dst。选择的规则为：当selMask的比特位是1时，从src0中选取，比特位是0时从src1选取。</p>
</td>
</tr>
<tr id="row1214111974012"><td class="cellrowborder" valign="top" headers="mcps1.2.4.1.1 "><p id="p1719055394219"><a name="p1719055394219"></a><a name="p1719055394219"></a><a href="GatherMask.md">GatherMask</a></p>
</td>
<td class="cellrowborder" valign="top" headers="mcps1.2.4.1.2 "><p id="p182166215152"><a name="p182166215152"></a><a name="p182166215152"></a>以内置固定模式对应的二进制或者用户自定义输入的Tensor数值对应的二进制为gather mask（数据收集的掩码），从源操作数中选取元素写入目的操作数中。</p>
</td>
</tr>
<tr id="row1095112164319"><td class="cellrowborder" valign="top" width="15.590000000000002%" headers="mcps1.2.4.1.1 "><p id="p1357761654319"><a name="p1357761654319"></a><a name="p1357761654319"></a>精度转换指令</p>
</td>
<td class="cellrowborder" valign="top" width="24.64%" headers="mcps1.2.4.1.2 "><p id="p757710162439"><a name="p757710162439"></a><a name="p757710162439"></a><a href="Cast.md">Cast</a></p>
</td>
<td class="cellrowborder" valign="top" width="59.77%" headers="mcps1.2.4.1.3 "><p id="p5969213430"><a name="p5969213430"></a><a name="p5969213430"></a>根据源操作数和目的操作数Tensor的数据类型进行精度转换。</p>
</td>
</tr>
<tr id="row1651510211434"><td class="cellrowborder" rowspan="11" valign="top" width="15.590000000000002%" headers="mcps1.2.4.1.1 "><p id="p1757701610436"><a name="p1757701610436"></a><a name="p1757701610436"></a>归约计算</p>
</td>
<td class="cellrowborder" valign="top" width="24.64%" headers="mcps1.2.4.1.2 "><p id="p1357731618438"><a name="p1357731618438"></a><a name="p1357731618438"></a><a href="ReduceMax.md">ReduceMax</a></p>
</td>
<td class="cellrowborder" valign="top" width="59.77%" headers="mcps1.2.4.1.3 "><p id="p1351515264314"><a name="p1351515264314"></a><a name="p1351515264314"></a>在所有的输入数据中找出最大值及最大值对应的索引位置。</p>
</td>
</tr>
<tr id="row196635294315"><td class="cellrowborder" valign="top" headers="mcps1.2.4.1.1 "><p id="p457871613433"><a name="p457871613433"></a><a name="p457871613433"></a><a href="ReduceMin.md">ReduceMin</a></p>
</td>
<td class="cellrowborder" valign="top" headers="mcps1.2.4.1.2 "><p id="p6853520201916"><a name="p6853520201916"></a><a name="p6853520201916"></a>在所有的输入数据中找出最小值及最小值对应的索引位置。</p>
</td>
</tr>
<tr id="row77918244313"><td class="cellrowborder" valign="top" headers="mcps1.2.4.1.1 "><p id="p35781716164313"><a name="p35781716164313"></a><a name="p35781716164313"></a><a href="ReduceSum.md">ReduceSum</a></p>
</td>
<td class="cellrowborder" valign="top" headers="mcps1.2.4.1.2 "><p id="p1546312559583"><a name="p1546312559583"></a><a name="p1546312559583"></a>对所有的输入数据求和。</p>
</td>
</tr>
<tr id="row79421217439"><td class="cellrowborder" valign="top" headers="mcps1.2.4.1.1 "><p id="p65788161435"><a name="p65788161435"></a><a name="p65788161435"></a><a href="WholeReduceMax.md">WholeReduceMax</a></p>
</td>
<td class="cellrowborder" valign="top" headers="mcps1.2.4.1.2 "><p id="p193962509594"><a name="p193962509594"></a><a name="p193962509594"></a>每个repeat内所有数据求最大值以及其索引index。</p>
</td>
</tr>
<tr id="row1961802720599"><td class="cellrowborder" valign="top" headers="mcps1.2.4.1.1 "><p id="p961972710592"><a name="p961972710592"></a><a name="p961972710592"></a><a href="WholeReduceMin.md">WholeReduceMin</a></p>
</td>
<td class="cellrowborder" valign="top" headers="mcps1.2.4.1.2 "><p id="p0619192712592"><a name="p0619192712592"></a><a name="p0619192712592"></a>每个repeat内所有数据求最小值以及其索引index。</p>
</td>
</tr>
<tr id="row2928152965915"><td class="cellrowborder" valign="top" headers="mcps1.2.4.1.1 "><p id="p1492892975917"><a name="p1492892975917"></a><a name="p1492892975917"></a><a href="WholeReduceSum.md">WholeReduceSum</a></p>
</td>
<td class="cellrowborder" valign="top" headers="mcps1.2.4.1.2 "><p id="p199283290594"><a name="p199283290594"></a><a name="p199283290594"></a>每个repeat内所有数据求和。</p>
</td>
</tr>
<tr id="row157712324319"><td class="cellrowborder" valign="top" headers="mcps1.2.4.1.1 "><p id="p3578151614430"><a name="p3578151614430"></a><a name="p3578151614430"></a><a href="BlockReduceMax.md">BlockReduceMax</a></p>
</td>
<td class="cellrowborder" valign="top" headers="mcps1.2.4.1.2 "><p id="p177391045819"><a name="p177391045819"></a><a name="p177391045819"></a>对每个repeat内所有元素求最大值。</p>
</td>
</tr>
<tr id="row17211133134315"><td class="cellrowborder" valign="top" headers="mcps1.2.4.1.1 "><p id="p1057841619435"><a name="p1057841619435"></a><a name="p1057841619435"></a><a href="BlockReduceMin.md">BlockReduceMin</a></p>
</td>
<td class="cellrowborder" valign="top" headers="mcps1.2.4.1.2 "><p id="p2021183144310"><a name="p2021183144310"></a><a name="p2021183144310"></a>对每个repeat内所有元素求最小值。</p>
</td>
</tr>
<tr id="row193401338432"><td class="cellrowborder" valign="top" headers="mcps1.2.4.1.1 "><p id="p1557814168438"><a name="p1557814168438"></a><a name="p1557814168438"></a><a href="BlockReduceSum.md">BlockReduceSum</a></p>
</td>
<td class="cellrowborder" valign="top" headers="mcps1.2.4.1.2 "><p id="p998114445293"><a name="p998114445293"></a><a name="p998114445293"></a>对每个repeat内所有元素求和。源操作数相加采用二叉树方式，两两相加。</p>
</td>
</tr>
<tr id="row250213184316"><td class="cellrowborder" valign="top" headers="mcps1.2.4.1.1 "><p id="p957811615437"><a name="p957811615437"></a><a name="p957811615437"></a><a href="PairReduceSum.md">PairReduceSum</a></p>
</td>
<td class="cellrowborder" valign="top" headers="mcps1.2.4.1.2 "><p id="p6502183184316"><a name="p6502183184316"></a><a name="p6502183184316"></a>PairReduceSum：相邻两个（奇偶）元素求和。</p>
</td>
</tr>
<tr id="row5812182116350"><td class="cellrowborder" valign="top" headers="mcps1.2.4.1.1 "><p id="p198129212351"><a name="p198129212351"></a><a name="p198129212351"></a><a href="RepeatReduceSum.md">RepeatReduceSum</a></p>
</td>
<td class="cellrowborder" valign="top" headers="mcps1.2.4.1.2 "><p id="p1481215214350"><a name="p1481215214350"></a><a name="p1481215214350"></a>每个repeat内所有数据求和。和<a href="WholeReduceSum.md">WholeReduceSum</a>接口相比，不支持mask逐bit模式。建议使用功能更全面的<a href="WholeReduceSum.md">WholeReduceSum</a>接口。</p>
</td>
</tr>
<tr id="row109261892541"><td class="cellrowborder" rowspan="2" valign="top" width="15.590000000000002%" headers="mcps1.2.4.1.1 "><p id="p629093119547"><a name="p629093119547"></a><a name="p629093119547"></a>数据转换</p>
</td>
<td class="cellrowborder" valign="top" width="24.64%" headers="mcps1.2.4.1.2 "><p id="p42901731145414"><a name="p42901731145414"></a><a name="p42901731145414"></a><a href="Transpose.md">Transpose</a></p>
</td>
<td class="cellrowborder" valign="top" width="59.77%" headers="mcps1.2.4.1.3 "><p id="p1429010318547"><a name="p1429010318547"></a><a name="p1429010318547"></a>可实现16*16的二维矩阵数据块的转置和[N,C,H,W]与[N,H,W,C]互相转换。</p>
</td>
</tr>
<tr id="row476616718548"><td class="cellrowborder" valign="top" headers="mcps1.2.4.1.1 "><p id="p629033110540"><a name="p629033110540"></a><a name="p629033110540"></a><a href="TransDataTo5HD.md">TransDataTo5HD</a></p>
</td>
<td class="cellrowborder" valign="top" headers="mcps1.2.4.1.2 "><p id="p162901631125410"><a name="p162901631125410"></a><a name="p162901631125410"></a>数据格式转换，一般用于将NCHW格式转换成NC1HWC0格式。特别的，也可以用于二维矩阵数据块的转置。</p>
</td>
</tr>
<tr id="row86261631104316"><td class="cellrowborder" rowspan="3" valign="top" width="15.590000000000002%" headers="mcps1.2.4.1.1 "><p id="p203231339194317"><a name="p203231339194317"></a><a name="p203231339194317"></a>数据填充</p>
</td>
<td class="cellrowborder" valign="top" width="24.64%" headers="mcps1.2.4.1.2 "><p id="p432393920436"><a name="p432393920436"></a><a name="p432393920436"></a><a href="Duplicate.md">Duplicate</a></p>
</td>
<td class="cellrowborder" valign="top" width="59.77%" headers="mcps1.2.4.1.3 "><p id="p9626431124314"><a name="p9626431124314"></a><a name="p9626431124314"></a>将一个变量或一个立即数，复制多次并填充到向量。</p>
</td>
</tr>
<tr id="row17641318431"><td class="cellrowborder" valign="top" headers="mcps1.2.4.1.1 "><p id="p232383912436"><a name="p232383912436"></a><a name="p232383912436"></a><a href="Brcb.md">Brcb</a></p>
</td>
<td class="cellrowborder" valign="top" headers="mcps1.2.4.1.2 "><p id="p911mcpsimp"><a name="p911mcpsimp"></a><a name="p911mcpsimp"></a>给定一个输入张量，每一次取输入张量中的8个数填充到结果张量的8个datablock（32Bytes）中去，每个数对应一个datablock。</p>
</td>
</tr>
<tr id="row173409456360"><td class="cellrowborder" valign="top" headers="mcps1.2.4.1.1 "><p id="p24100384369"><a name="p24100384369"></a><a name="p24100384369"></a><a href="CreateVecIndex.md">CreateVecIndex</a></p>
</td>
<td class="cellrowborder" valign="top" headers="mcps1.2.4.1.2 "><p id="p234064518368"><a name="p234064518368"></a><a name="p234064518368"></a>以firstValue为起始值创建向量索引。</p>
</td>
</tr>
<tr id="row161901532124317"><td class="cellrowborder" valign="top" width="15.590000000000002%" headers="mcps1.2.4.1.1 "><p id="p2323163915431"><a name="p2323163915431"></a><a name="p2323163915431"></a>数据分散/数据收集</p>
</td>
<td class="cellrowborder" valign="top" width="24.64%" headers="mcps1.2.4.1.2 "><p id="p1232318396431"><a name="p1232318396431"></a><a name="p1232318396431"></a><a href="Gather.md">Gather</a></p>
</td>
<td class="cellrowborder" valign="top" width="59.77%" headers="mcps1.2.4.1.3 "><p id="p161901632134314"><a name="p161901632134314"></a><a name="p161901632134314"></a>给定输入的张量和一个地址偏移张量，Gather指令根据偏移地址将输入张量按元素收集到结果张量中。</p>
</td>
</tr>
<tr id="row16331932164312"><td class="cellrowborder" rowspan="4" valign="top" width="15.590000000000002%" headers="mcps1.2.4.1.1 "><p id="p1632373914312"><a name="p1632373914312"></a><a name="p1632373914312"></a>掩码操作</p>
</td>
<td class="cellrowborder" valign="top" width="24.64%" headers="mcps1.2.4.1.2 "><p id="p43241739154316"><a name="p43241739154316"></a><a name="p43241739154316"></a><a href="SetMaskCount.md">SetMaskCount</a></p>
</td>
<td class="cellrowborder" valign="top" width="59.77%" headers="mcps1.2.4.1.3 "><p id="p9633133216430"><a name="p9633133216430"></a><a name="p9633133216430"></a>设置mask模式为Counter模式。该模式下，不需要开发者去感知迭代次数、处理非对齐的尾块等操作，可直接传入计算数据量，实际迭代次数由Vector计算单元自动推断。</p>
</td>
</tr>
<tr id="row1678513323437"><td class="cellrowborder" valign="top" headers="mcps1.2.4.1.1 "><p id="p15324339114312"><a name="p15324339114312"></a><a name="p15324339114312"></a><a href="SetMaskNorm.md">SetMaskNorm</a></p>
</td>
<td class="cellrowborder" valign="top" headers="mcps1.2.4.1.2 "><p id="p1785143218431"><a name="p1785143218431"></a><a name="p1785143218431"></a>设置mask模式为Normal模式。该模式为系统默认模式，支持开发者配置迭代次数。</p>
</td>
</tr>
<tr id="row1393713216436"><td class="cellrowborder" valign="top" headers="mcps1.2.4.1.1 "><p id="p23247397438"><a name="p23247397438"></a><a name="p23247397438"></a><a href="SetVectorMask.md">SetVectorMask</a></p>
</td>
<td class="cellrowborder" valign="top" headers="mcps1.2.4.1.2 "><p id="p993873224313"><a name="p993873224313"></a><a name="p993873224313"></a>用于在矢量计算时设置mask。</p>
</td>
</tr>
<tr id="row15756339433"><td class="cellrowborder" valign="top" headers="mcps1.2.4.1.1 "><p id="p3324039144317"><a name="p3324039144317"></a><a name="p3324039144317"></a><a href="ResetMask.md">ResetMask</a></p>
</td>
<td class="cellrowborder" valign="top" headers="mcps1.2.4.1.2 "><p id="p57533324312"><a name="p57533324312"></a><a name="p57533324312"></a>恢复mask的值为默认值（全1），表示矢量计算中每次迭代内的所有元素都将参与运算。</p>
</td>
</tr>
<tr id="row743516194407"><td class="cellrowborder" valign="top" width="15.590000000000002%" headers="mcps1.2.4.1.1 "><p id="p23240390438"><a name="p23240390438"></a><a name="p23240390438"></a>量化设置</p>
</td>
<td class="cellrowborder" valign="top" width="24.64%" headers="mcps1.2.4.1.2 "><p id="p10324173918436"><a name="p10324173918436"></a><a name="p10324173918436"></a><a href="SetDeqScale.md">SetDeqScale</a></p>
</td>
<td class="cellrowborder" valign="top" width="59.77%" headers="mcps1.2.4.1.3 "><p id="p1238122924318"><a name="p1238122924318"></a><a name="p1238122924318"></a>设置DEQSCALE寄存器的值。</p>
</td>
</tr>
</tbody>
</table>

**表 4**  标量计算API列表

<a name="table339023582010"></a>
<table><thead align="left"><tr id="row1539063572010"><th class="cellrowborder" valign="top" width="40.37%" id="mcps1.2.3.1.1"><p id="p13390235192015"><a name="p13390235192015"></a><a name="p13390235192015"></a>接口名</p>
</th>
<th class="cellrowborder" valign="top" width="59.63%" id="mcps1.2.3.1.2"><p id="p2390103519209"><a name="p2390103519209"></a><a name="p2390103519209"></a>功能描述</p>
</th>
</tr>
</thead>
<tbody><tr id="row839013512016"><td class="cellrowborder" valign="top" width="40.37%" headers="mcps1.2.3.1.1 "><p id="p2023773313396"><a name="p2023773313396"></a><a name="p2023773313396"></a><a href="GetBitCount.md">GetBitCount</a></p>
</td>
<td class="cellrowborder" valign="top" width="59.63%" headers="mcps1.2.3.1.2 "><p id="p1739063514204"><a name="p1739063514204"></a><a name="p1739063514204"></a>获取一个uint64_t类型数字的二进制中0或者1的个数。</p>
</td>
</tr>
<tr id="row5390935132010"><td class="cellrowborder" valign="top" width="40.37%" headers="mcps1.2.3.1.1 "><p id="p9743558403"><a name="p9743558403"></a><a name="p9743558403"></a><a href="CountLeadingZero.md">CountLeadingZero</a></p>
</td>
<td class="cellrowborder" valign="top" width="59.63%" headers="mcps1.2.3.1.2 "><p id="p739014352201"><a name="p739014352201"></a><a name="p739014352201"></a>计算一个uint64_t类型数字前导0的个数（二进制从最高位到第一个1一共有多少个0）。</p>
</td>
</tr>
<tr id="row187241145152620"><td class="cellrowborder" valign="top" width="40.37%" headers="mcps1.2.3.1.1 "><p id="p19725445132613"><a name="p19725445132613"></a><a name="p19725445132613"></a><a href="Cast（float转half-int32_t）.md">Cast（float转half、int32_t）</a></p>
</td>
<td class="cellrowborder" valign="top" width="59.63%" headers="mcps1.2.3.1.2 "><p id="p772515454262"><a name="p772515454262"></a><a name="p772515454262"></a>将一个scalar的类型转换为指定的类型。</p>
</td>
</tr>
<tr id="row129095914341"><td class="cellrowborder" valign="top" width="40.37%" headers="mcps1.2.3.1.1 "><p id="p15290059143412"><a name="p15290059143412"></a><a name="p15290059143412"></a><a href="CountBitsCntSameAsSignBit.md">CountBitsCntSameAsSignBit</a></p>
</td>
<td class="cellrowborder" valign="top" width="59.63%" headers="mcps1.2.3.1.2 "><p id="p05116051416"><a name="p05116051416"></a><a name="p05116051416"></a>计算一个uint64_t类型数字的二进制中，从最高数值位开始与符号位相同的连续比特位的个数。</p>
</td>
</tr>
<tr id="row944019598346"><td class="cellrowborder" valign="top" width="40.37%" headers="mcps1.2.3.1.1 "><p id="p1062632319354"><a name="p1062632319354"></a><a name="p1062632319354"></a><a href="GetSFFValue.md">GetSFFValue</a></p>
</td>
<td class="cellrowborder" valign="top" width="59.63%" headers="mcps1.2.3.1.2 "><p id="p13440259173413"><a name="p13440259173413"></a><a name="p13440259173413"></a>获取一个uint64_t类型数字的二进制中第一个0或1出现的位置。</p>
</td>
</tr>
<tr id="row1566175913418"><td class="cellrowborder" valign="top" width="40.37%" headers="mcps1.2.3.1.1 "><p id="p9566659173420"><a name="p9566659173420"></a><a name="p9566659173420"></a><a href="Cast（float转bfloat16_t）.md">Cast（float转bfloat16_t）</a></p>
</td>
<td class="cellrowborder" valign="top" width="59.63%" headers="mcps1.2.3.1.2 "><p id="p828219133366"><a name="p828219133366"></a><a name="p828219133366"></a>float类型标量数据转换成bfloat16_t类型标量数据。</p>
</td>
</tr>
<tr id="row13704105910340"><td class="cellrowborder" valign="top" width="40.37%" headers="mcps1.2.3.1.1 "><p id="p16704195953414"><a name="p16704195953414"></a><a name="p16704195953414"></a><a href="Cast（多类型转float）.md">Cast（多类型转float）</a></p>
</td>
<td class="cellrowborder" valign="top" width="59.63%" headers="mcps1.2.3.1.2 "><p id="p15293151211360"><a name="p15293151211360"></a><a name="p15293151211360"></a>bfloat16_t类型标量数据转换成float类型标量数据。</p>
</td>
</tr>
</tbody>
</table>

**表 5**  资源管理API列表

<a name="table1267664316264"></a>
<table><thead align="left"><tr id="row15676154310267"><th class="cellrowborder" valign="top" width="40.37%" id="mcps1.2.3.1.1"><p id="p6676543192617"><a name="p6676543192617"></a><a name="p6676543192617"></a>接口名</p>
</th>
<th class="cellrowborder" valign="top" width="59.63%" id="mcps1.2.3.1.2"><p id="p146761434266"><a name="p146761434266"></a><a name="p146761434266"></a>功能描述</p>
</th>
</tr>
</thead>
<tbody><tr id="row367664312619"><td class="cellrowborder" valign="top" width="40.37%" headers="mcps1.2.3.1.1 "><p id="p667654312618"><a name="p667654312618"></a><a name="p667654312618"></a><a href="TPipe.md">TPipe</a></p>
</td>
<td class="cellrowborder" valign="top" width="59.63%" headers="mcps1.2.3.1.2 "><p id="p6428345172717"><a name="p6428345172717"></a><a name="p6428345172717"></a>TPipe是用来管理全局内存等资源的框架。通过TPipe类提供的接口可以完成内存等资源的分配管理操作。</p>
</td>
</tr>
<tr id="row1867604312262"><td class="cellrowborder" valign="top" width="40.37%" headers="mcps1.2.3.1.1 "><p id="p616062319285"><a name="p616062319285"></a><a name="p616062319285"></a><a href="GetTPipePtr.md">GetTPipePtr</a></p>
</td>
<td class="cellrowborder" valign="top" width="59.63%" headers="mcps1.2.3.1.2 "><p id="p26771543172611"><a name="p26771543172611"></a><a name="p26771543172611"></a>获取框架当前管理全局内存的TPipe指针，用户获取指针后，可进行TPipe相关的操作。</p>
</td>
</tr>
<tr id="row1381344122816"><td class="cellrowborder" valign="top" width="40.37%" headers="mcps1.2.3.1.1 "><p id="p681124418285"><a name="p681124418285"></a><a name="p681124418285"></a><a href="TBufPool.md">TBufPool</a></p>
</td>
<td class="cellrowborder" valign="top" width="59.63%" headers="mcps1.2.3.1.2 "><p id="p1062105494410"><a name="p1062105494410"></a><a name="p1062105494410"></a>TPipe可以管理全局内存资源，而TBufPool可以手动管理或复用<span id="ph1088254310583"><a name="ph1088254310583"></a><a name="ph1088254310583"></a>Unified Buffer</span>/<span id="ph1535518221316"><a name="ph1535518221316"></a><a name="ph1535518221316"></a>L1 Buffer</span>物理内存，主要用于多个stage计算中<span id="ph791414174415"><a name="ph791414174415"></a><a name="ph791414174415"></a>Unified Buffer</span>/<span id="ph7621654184416"><a name="ph7621654184416"></a><a name="ph7621654184416"></a>L1 Buffer</span>物理内存不足的场景。</p>
</td>
</tr>
<tr id="row1032714367308"><td class="cellrowborder" valign="top" width="40.37%" headers="mcps1.2.3.1.1 "><p id="p73271836193011"><a name="p73271836193011"></a><a name="p73271836193011"></a><a href="TQue.md">TQue</a></p>
</td>
<td class="cellrowborder" valign="top" width="59.63%" headers="mcps1.2.3.1.2 "><p id="p13327163673017"><a name="p13327163673017"></a><a name="p13327163673017"></a>提供入队出队等接口，通过队列（Queue）完成任务间同步。</p>
</td>
</tr>
<tr id="row6569103614302"><td class="cellrowborder" valign="top" width="40.37%" headers="mcps1.2.3.1.1 "><p id="p165691368309"><a name="p165691368309"></a><a name="p165691368309"></a><a href="TQueBind.md">TQueBind</a></p>
</td>
<td class="cellrowborder" valign="top" width="59.63%" headers="mcps1.2.3.1.2 "><p id="p9569203683010"><a name="p9569203683010"></a><a name="p9569203683010"></a>TQueBind绑定源逻辑位置和目的逻辑位置，根据源位置和目的位置，来确定内存分配的位置 、插入对应的同步事件，帮助开发者解决内存分配和管理、同步等问题。</p>
</td>
</tr>
<tr id="row106614368306"><td class="cellrowborder" valign="top" width="40.37%" headers="mcps1.2.3.1.1 "><p id="p1866114362309"><a name="p1866114362309"></a><a name="p1866114362309"></a><a href="TBuf.md">TBuf</a></p>
</td>
<td class="cellrowborder" valign="top" width="59.63%" headers="mcps1.2.3.1.2 "><p id="p12661203618308"><a name="p12661203618308"></a><a name="p12661203618308"></a>使用<span id="ph438819594207"><a name="ph438819594207"></a><a name="ph438819594207"></a>Ascend C</span>编程的过程中，可能会用到一些临时变量。这些临时变量占用的内存可以使用TBuf数据结构来管理。</p>
</td>
</tr>
<tr id="row158561936163012"><td class="cellrowborder" valign="top" width="40.37%" headers="mcps1.2.3.1.1 "><p id="p545354010463"><a name="p545354010463"></a><a name="p545354010463"></a><a href="InitSpmBuffer.md">InitSpmBuffer</a></p>
</td>
<td class="cellrowborder" valign="top" width="59.63%" headers="mcps1.2.3.1.2 "><p id="p16856103663012"><a name="p16856103663012"></a><a name="p16856103663012"></a>初始化SPM Buffer。</p>
</td>
</tr>
<tr id="row324173716309"><td class="cellrowborder" valign="top" width="40.37%" headers="mcps1.2.3.1.1 "><p id="p1245314074616"><a name="p1245314074616"></a><a name="p1245314074616"></a><a href="WriteSpmBuffer.md">WriteSpmBuffer</a></p>
</td>
<td class="cellrowborder" valign="top" width="59.63%" headers="mcps1.2.3.1.2 "><p id="p17575330684"><a name="p17575330684"></a><a name="p17575330684"></a>将需要溢出暂存的数据拷贝到SPM Buffer中。</p>
</td>
</tr>
<tr id="row1515119377302"><td class="cellrowborder" valign="top" width="40.37%" headers="mcps1.2.3.1.1 "><p id="p19453154018462"><a name="p19453154018462"></a><a name="p19453154018462"></a><a href="ReadSpmBuffer.md">ReadSpmBuffer</a></p>
</td>
<td class="cellrowborder" valign="top" width="59.63%" headers="mcps1.2.3.1.2 "><p id="p1881419176258"><a name="p1881419176258"></a><a name="p1881419176258"></a>从SPM Buffer读回到local数据中。</p>
</td>
</tr>
<tr id="row1550403614616"><td class="cellrowborder" valign="top" width="40.37%" headers="mcps1.2.3.1.1 "><p id="p4453154014462"><a name="p4453154014462"></a><a name="p4453154014462"></a><a href="GetUserWorkspace.md">GetUserWorkspace</a></p>
</td>
<td class="cellrowborder" valign="top" width="59.63%" headers="mcps1.2.3.1.2 "><p id="p19504103614618"><a name="p19504103614618"></a><a name="p19504103614618"></a>获取用户使用的workspace指针。</p>
</td>
</tr>
<tr id="row15918536184616"><td class="cellrowborder" valign="top" width="40.37%" headers="mcps1.2.3.1.1 "><p id="p54539406461"><a name="p54539406461"></a><a name="p54539406461"></a><a href="SetSysWorkSpace.md">SetSysWorkSpace</a></p>
</td>
<td class="cellrowborder" valign="top" width="59.63%" headers="mcps1.2.3.1.2 "><p id="p4918163654611"><a name="p4918163654611"></a><a name="p4918163654611"></a>在进行融合算子编程时，由于框架通信机制需要使用到workspace，也就是系统workspace，所以在该场景下，开发者要调用该接口，设置系统workspace的指针。</p>
</td>
</tr>
<tr id="row196011374465"><td class="cellrowborder" valign="top" width="40.37%" headers="mcps1.2.3.1.1 "><p id="p2045304004615"><a name="p2045304004615"></a><a name="p2045304004615"></a><a href="GetSysWorkSpacePtr.md">GetSysWorkSpacePtr</a></p>
</td>
<td class="cellrowborder" valign="top" width="59.63%" headers="mcps1.2.3.1.2 "><p id="p86018377467"><a name="p86018377467"></a><a name="p86018377467"></a>获取系统workspace指针。</p>
</td>
</tr>
</tbody>
</table>

**表 6**  同步控制API列表

<a name="table921112251162"></a>
<table><thead align="left"><tr id="row975619311161"><th class="cellrowborder" valign="top" width="40.37%" id="mcps1.2.3.1.1"><p id="p8307249121617"><a name="p8307249121617"></a><a name="p8307249121617"></a>接口名</p>
</th>
<th class="cellrowborder" valign="top" width="59.63%" id="mcps1.2.3.1.2"><p id="p0308184961620"><a name="p0308184961620"></a><a name="p0308184961620"></a>功能描述</p>
</th>
</tr>
</thead>
<tbody><tr id="row1021262515169"><td class="cellrowborder" valign="top" width="40.37%" headers="mcps1.2.3.1.1 "><p id="p62931833203117"><a name="p62931833203117"></a><a name="p62931833203117"></a><a href="TQueSync.md">TQueSync</a></p>
</td>
<td class="cellrowborder" valign="top" width="59.63%" headers="mcps1.2.3.1.2 "><p id="p82931433203116"><a name="p82931433203116"></a><a name="p82931433203116"></a>TQueSync类提供同步控制接口，开发者可以使用这类API来自行完成同步控制。</p>
</td>
</tr>
<tr id="row1821272513168"><td class="cellrowborder" valign="top" width="40.37%" headers="mcps1.2.3.1.1 "><p id="p1190314213216"><a name="p1190314213216"></a><a name="p1190314213216"></a><a href="IBSet.md">IBSet</a></p>
</td>
<td class="cellrowborder" valign="top" width="59.63%" headers="mcps1.2.3.1.2 "><p id="p15692228145311"><a name="p15692228145311"></a><a name="p15692228145311"></a>当不同核之间操作同一块全局内存且可能存在读后写、写后读以及写后写等数据依赖问题时，通过调用该函数来插入同步语句来避免上述数据依赖时可能出现的数据读写错误问题。调用IBSet设置某一个核的标志位，与IBWait成对出现配合使用，表示核之间的同步等待指令，等待某一个核操作完成。</p>
</td>
</tr>
<tr id="row121219258161"><td class="cellrowborder" valign="top" width="40.37%" headers="mcps1.2.3.1.1 "><p id="p13903164214324"><a name="p13903164214324"></a><a name="p13903164214324"></a><a href="IBWait.md">IBWait</a></p>
</td>
<td class="cellrowborder" valign="top" width="59.63%" headers="mcps1.2.3.1.2 "><p id="p27141049175319"><a name="p27141049175319"></a><a name="p27141049175319"></a>当不同核之间操作同一块全局内存且可能存在读后写、写后读以及写后写等数据依赖问题时，通过调用该函数来插入同步语句来避免上述数据依赖时可能出现的数据读写错误问题。IBWait与IBSet成对出现配合使用，表示核之间的同步等待指令，等待某一个核操作完成。</p>
</td>
</tr>
<tr id="row1121220254168"><td class="cellrowborder" valign="top" width="40.37%" headers="mcps1.2.3.1.1 "><p id="p8904942143216"><a name="p8904942143216"></a><a name="p8904942143216"></a><a href="SyncAll.md">SyncAll</a></p>
</td>
<td class="cellrowborder" valign="top" width="59.63%" headers="mcps1.2.3.1.2 "><p id="p12782512105416"><a name="p12782512105416"></a><a name="p12782512105416"></a>当不同核之间操作同一块全局内存且可能存在读后写、写后读以及写后写等数据依赖问题时，通过调用该函数来插入同步语句来避免上述数据依赖时可能出现的数据读写错误问题。目前多核同步分为硬同步和软同步，硬件同步是利用硬件自带的全核同步指令由硬件保证多核同步，软件同步是使用软件算法模拟实现。</p>
</td>
</tr>
<tr id="row92121225191611"><td class="cellrowborder" valign="top" width="40.37%" headers="mcps1.2.3.1.1 "><p id="p29094917345"><a name="p29094917345"></a><a name="p29094917345"></a><a href="InitDetermineComputeWorkspace.md">InitDetermineComputeWorkspace</a></p>
</td>
<td class="cellrowborder" valign="top" width="59.63%" headers="mcps1.2.3.1.2 "><p id="p9909129113414"><a name="p9909129113414"></a><a name="p9909129113414"></a>初始化GM共享内存的值，完成初始化后才可以调用<a href="WaitPreBlock.md">WaitPreBlock</a>和<a href="NotifyNextBlock.md">NotifyNextBlock</a>。</p>
</td>
</tr>
<tr id="row16213425171615"><td class="cellrowborder" valign="top" width="40.37%" headers="mcps1.2.3.1.1 "><p id="p390910913341"><a name="p390910913341"></a><a name="p390910913341"></a><a href="WaitPreBlock.md">WaitPreBlock</a></p>
</td>
<td class="cellrowborder" valign="top" width="59.63%" headers="mcps1.2.3.1.2 "><p id="p49091792347"><a name="p49091792347"></a><a name="p49091792347"></a>通过读GM地址中的值，确认是否需要继续等待，当GM的值满足当前核的等待条件时，该核即可往下执行，进行下一步操作。</p>
</td>
</tr>
<tr id="row152131825131610"><td class="cellrowborder" valign="top" width="40.37%" headers="mcps1.2.3.1.1 "><p id="p29091594341"><a name="p29091594341"></a><a name="p29091594341"></a><a href="NotifyNextBlock.md">NotifyNextBlock</a></p>
</td>
<td class="cellrowborder" valign="top" width="59.63%" headers="mcps1.2.3.1.2 "><p id="p6909899349"><a name="p6909899349"></a><a name="p6909899349"></a>通过写GM地址，通知下一个核当前核的操作已完成，下一个核可以进行操作。</p>
</td>
</tr>
<tr id="row52131925101618"><td class="cellrowborder" valign="top" width="40.37%" headers="mcps1.2.3.1.1 "><p id="p39841259102515"><a name="p39841259102515"></a><a name="p39841259102515"></a><a href="SetNextTaskStart.md">SetNextTaskStart</a></p>
</td>
<td class="cellrowborder" valign="top" width="59.63%" headers="mcps1.2.3.1.2 "><p id="p12934317236"><a name="p12934317236"></a><a name="p12934317236"></a><span id="ph1882043515233"><a name="ph1882043515233"></a><a name="ph1882043515233"></a>在SuperKernel的子Kernel中调用，调用后的指令可以和后续其他的子Kernel实现并行，提升整体性能。</span></p>
</td>
</tr>
<tr id="row162135257162"><td class="cellrowborder" valign="top" width="40.37%" headers="mcps1.2.3.1.1 "><p id="p288919718266"><a name="p288919718266"></a><a name="p288919718266"></a><a href="WaitPreTaskEnd.md">WaitPreTaskEnd</a></p>
</td>
<td class="cellrowborder" valign="top" width="59.63%" headers="mcps1.2.3.1.2 "><p id="p632496172313"><a name="p632496172313"></a><a name="p632496172313"></a><span id="ph1925517378244"><a name="ph1925517378244"></a><a name="ph1925517378244"></a>在SuperKernel的子Kernel中调用，调用前的指令可以和前序其他的子Kernel实现并行，提升整体性能。</span></p>
</td>
</tr>
</tbody>
</table>

**表 7**  缓存处理API列表

<a name="table5254131810573"></a>
<table><thead align="left"><tr id="row32541018135711"><th class="cellrowborder" valign="top" width="40.27%" id="mcps1.2.3.1.1"><p id="p17682103125810"><a name="p17682103125810"></a><a name="p17682103125810"></a>接口名</p>
</th>
<th class="cellrowborder" valign="top" width="59.730000000000004%" id="mcps1.2.3.1.2"><p id="p26828319585"><a name="p26828319585"></a><a name="p26828319585"></a>功能描述</p>
</th>
</tr>
</thead>
<tbody><tr id="row1325501835712"><td class="cellrowborder" valign="top" width="40.27%" headers="mcps1.2.3.1.1 "><p id="p83441336473"><a name="p83441336473"></a><a name="p83441336473"></a><a href="DataCachePreload.md">DataCachePreload</a></p>
</td>
<td class="cellrowborder" valign="top" width="59.730000000000004%" headers="mcps1.2.3.1.2 "><p id="p18111100154719"><a name="p18111100154719"></a><a name="p18111100154719"></a>从源地址所在的特定DDR地址预加载数据到data cache中。</p>
</td>
</tr>
<tr id="row5255181815720"><td class="cellrowborder" valign="top" width="40.27%" headers="mcps1.2.3.1.1 "><p id="p1134473164714"><a name="p1134473164714"></a><a name="p1134473164714"></a><a href="DataCacheCleanAndInvalid.md">DataCacheCleanAndInvalid</a></p>
</td>
<td class="cellrowborder" valign="top" width="59.730000000000004%" headers="mcps1.2.3.1.2 "><p id="p647301034415"><a name="p647301034415"></a><a name="p647301034415"></a>该接口用来刷新Cache，保证Cache的一致性。</p>
</td>
</tr>
</tbody>
</table>

**表 8**  系统变量访问API列表

<a name="table26716458301"></a>
<table><thead align="left"><tr id="row15672134518304"><th class="cellrowborder" valign="top" width="40.37%" id="mcps1.2.3.1.1"><p id="p18672144511306"><a name="p18672144511306"></a><a name="p18672144511306"></a>接口名</p>
</th>
<th class="cellrowborder" valign="top" width="59.63%" id="mcps1.2.3.1.2"><p id="p56726451306"><a name="p56726451306"></a><a name="p56726451306"></a>功能描述</p>
</th>
</tr>
</thead>
<tbody><tr id="row967212454304"><td class="cellrowborder" valign="top" width="40.37%" headers="mcps1.2.3.1.1 "><p id="p1194732015475"><a name="p1194732015475"></a><a name="p1194732015475"></a><a href="GetBlockNum.md">GetBlockNum</a></p>
</td>
<td class="cellrowborder" valign="top" width="59.63%" headers="mcps1.2.3.1.2 "><p id="p585381495015"><a name="p585381495015"></a><a name="p585381495015"></a>获取当前任务配置的Block数，用于代码内部的多核逻辑控制等。</p>
</td>
</tr>
<tr id="row1967224513011"><td class="cellrowborder" valign="top" width="40.37%" headers="mcps1.2.3.1.1 "><p id="p11947620104710"><a name="p11947620104710"></a><a name="p11947620104710"></a><a href="GetBlockIdx.md">GetBlockIdx</a></p>
</td>
<td class="cellrowborder" valign="top" width="59.63%" headers="mcps1.2.3.1.2 "><p id="p54921825165017"><a name="p54921825165017"></a><a name="p54921825165017"></a>获取当前core的index，用于代码内部的多核逻辑控制及多核偏移量计算等。</p>
</td>
</tr>
<tr id="row11672104593014"><td class="cellrowborder" valign="top" width="40.37%" headers="mcps1.2.3.1.1 "><p id="p17947182012472"><a name="p17947182012472"></a><a name="p17947182012472"></a><a href="GetDataBlockSizeInBytes.md">GetDataBlockSizeInBytes</a></p>
</td>
<td class="cellrowborder" valign="top" width="59.63%" headers="mcps1.2.3.1.2 "><p id="p14250113816503"><a name="p14250113816503"></a><a name="p14250113816503"></a>获取当前芯片版本一个datablock的大小，单位为byte。开发者根据datablock的大小来计算API指令中待传入的<span>repeatTime</span> 、<span id="ph06845043917"><a name="ph06845043917"></a><a name="ph06845043917"></a>DataBlock Stride</span>、<span id="ph13946527113916"><a name="ph13946527113916"></a><a name="ph13946527113916"></a>Repeat Stride</span><span>等</span>参数值。</p>
</td>
</tr>
<tr id="row3672445153011"><td class="cellrowborder" valign="top" width="40.37%" headers="mcps1.2.3.1.1 "><p id="p1594715207476"><a name="p1594715207476"></a><a name="p1594715207476"></a><a href="GetArchVersion.md">GetArchVersion</a></p>
</td>
<td class="cellrowborder" valign="top" width="59.63%" headers="mcps1.2.3.1.2 "><p id="p764334516501"><a name="p764334516501"></a><a name="p764334516501"></a>获取当前AI处理器架构版本号。</p>
</td>
</tr>
<tr id="row4284193416298"><td class="cellrowborder" valign="top" width="40.37%" headers="mcps1.2.3.1.1 "><p id="p12284163417298"><a name="p12284163417298"></a><a name="p12284163417298"></a><a href="InitSocState.md">InitSocState</a></p>
</td>
<td class="cellrowborder" valign="top" width="59.63%" headers="mcps1.2.3.1.2 "><p id="p1828453452913"><a name="p1828453452913"></a><a name="p1828453452913"></a>由于AI Core上存在一些全局状态，如原子累加状态、Mask模式等，在实际运行中，这些值可以被前序执行的算子修改而导致计算出现不符合预期的行为，在<a href="静态Tensor编程.md">静态Tensor编程</a>的场景中用户必须在Kernel入口处调用此函数来初始化AI Core状态 。</p>
</td>
</tr>
</tbody>
</table>

**表 9**  原子操作接口列表

<a name="table1395854383210"></a>
<table><thead align="left"><tr id="row12958043173211"><th class="cellrowborder" valign="top" width="40.089999999999996%" id="mcps1.2.3.1.1"><p id="p2958104363212"><a name="p2958104363212"></a><a name="p2958104363212"></a>接口名</p>
</th>
<th class="cellrowborder" valign="top" width="59.91%" id="mcps1.2.3.1.2"><p id="p0958174353211"><a name="p0958174353211"></a><a name="p0958174353211"></a>功能描述</p>
</th>
</tr>
</thead>
<tbody><tr id="row1077235835018"><td class="cellrowborder" valign="top" width="40.089999999999996%" headers="mcps1.2.3.1.1 "><p id="p10729145920507"><a name="p10729145920507"></a><a name="p10729145920507"></a><a href="SetAtomicAdd.md">SetAtomicAdd</a></p>
</td>
<td class="cellrowborder" valign="top" width="59.91%" headers="mcps1.2.3.1.2 "><p id="p20143125311516"><a name="p20143125311516"></a><a name="p20143125311516"></a>设置接下来从VECOUT到GM，L0C到GM，L1到GM的数据传输是否进行原子累加，可根据参数不同设定不同的累加数据类型。</p>
</td>
</tr>
<tr id="row139588436327"><td class="cellrowborder" valign="top" width="40.089999999999996%" headers="mcps1.2.3.1.1 "><p id="p1772985912505"><a name="p1772985912505"></a><a name="p1772985912505"></a><a href="SetAtomicType.md">SetAtomicType</a></p>
</td>
<td class="cellrowborder" valign="top" width="59.91%" headers="mcps1.2.3.1.2 "><p id="p38701310125517"><a name="p38701310125517"></a><a name="p38701310125517"></a>通过设置模板参数来设定原子操作不同的数据类型。</p>
</td>
</tr>
<tr id="row595874313218"><td class="cellrowborder" valign="top" width="40.089999999999996%" headers="mcps1.2.3.1.1 "><p id="p1870510588519"><a name="p1870510588519"></a><a name="p1870510588519"></a><a href="DisableDmaAtomic.md">DisableDmaAtomic</a></p>
</td>
<td class="cellrowborder" valign="top" width="59.91%" headers="mcps1.2.3.1.2 "><p id="p1214271020615"><a name="p1214271020615"></a><a name="p1214271020615"></a>原子操作函数，清空原子操作的状态。</p>
</td>
</tr>
<tr id="row2084965011239"><td class="cellrowborder" valign="top" width="40.089999999999996%" headers="mcps1.2.3.1.1 "><p id="p1359015148246"><a name="p1359015148246"></a><a name="p1359015148246"></a><a href="AtomicAdd.md">AtomicAdd</a></p>
</td>
<td class="cellrowborder" valign="top" width="59.91%" headers="mcps1.2.3.1.2 "><p id="p108491650102316"><a name="p108491650102316"></a><a name="p108491650102316"></a><span id="ph1305389278"><a name="ph1305389278"></a><a name="ph1305389278"></a>调用该接口后，可在指定GM地址上进行原子加操作。</span></p>
</td>
</tr>
<tr id="row171030533239"><td class="cellrowborder" valign="top" width="40.089999999999996%" headers="mcps1.2.3.1.1 "><p id="p324502616243"><a name="p324502616243"></a><a name="p324502616243"></a><a href="AtomicMin.md">AtomicMin</a></p>
</td>
<td class="cellrowborder" valign="top" width="59.91%" headers="mcps1.2.3.1.2 "><p id="p18103053142317"><a name="p18103053142317"></a><a name="p18103053142317"></a><span id="ph169593419276"><a name="ph169593419276"></a><a name="ph169593419276"></a>调用该接口后，可在指定GM地址上进行原子比较取小操作。</span></p>
</td>
</tr>
<tr id="row1836819554234"><td class="cellrowborder" valign="top" width="40.089999999999996%" headers="mcps1.2.3.1.1 "><p id="p142593217246"><a name="p142593217246"></a><a name="p142593217246"></a><a href="AtomicMax.md">AtomicMax</a></p>
</td>
<td class="cellrowborder" valign="top" width="59.91%" headers="mcps1.2.3.1.2 "><p id="p17368655132317"><a name="p17368655132317"></a><a name="p17368655132317"></a><span id="ph192894512276"><a name="ph192894512276"></a><a name="ph192894512276"></a>调用该接口后，可在指定GM地址上进行原子取大操作。</span></p>
</td>
</tr>
<tr id="row437810019249"><td class="cellrowborder" valign="top" width="40.089999999999996%" headers="mcps1.2.3.1.1 "><p id="p143501939152411"><a name="p143501939152411"></a><a name="p143501939152411"></a><a href="AtomicCas.md">AtomicCas</a></p>
</td>
<td class="cellrowborder" valign="top" width="59.91%" headers="mcps1.2.3.1.2 "><p id="p1637810162415"><a name="p1637810162415"></a><a name="p1637810162415"></a><span id="ph1570715132720"><a name="ph1570715132720"></a><a name="ph1570715132720"></a>调用该接口后，可在指定GM地址上进行原子比较，如果和value1相等，则把value2的值赋值到GM上；如果和value1不相等，则GM上的值不变。</span></p>
</td>
</tr>
<tr id="row1223315282416"><td class="cellrowborder" valign="top" width="40.089999999999996%" headers="mcps1.2.3.1.1 "><p id="p11984124519248"><a name="p11984124519248"></a><a name="p11984124519248"></a><a href="AtomicExch.md">AtomicExch</a></p>
</td>
<td class="cellrowborder" valign="top" width="59.91%" headers="mcps1.2.3.1.2 "><p id="p123311213245"><a name="p123311213245"></a><a name="p123311213245"></a><span id="ph8490145722711"><a name="ph8490145722711"></a><a name="ph8490145722711"></a>在GM内存中执行原子交换操作。具体来说，它读取指定GM地址上的数据，并将新的值存储回同一地址。函数返回旧值。</span></p>
</td>
</tr>
</tbody>
</table>

**表 10**  调试接口列表

<a name="table18295429109"></a>
<table><thead align="left"><tr id="row530154201012"><th class="cellrowborder" valign="top" width="37.71%" id="mcps1.2.3.1.1"><p id="p113014281016"><a name="p113014281016"></a><a name="p113014281016"></a>接口名</p>
</th>
<th class="cellrowborder" valign="top" width="62.29%" id="mcps1.2.3.1.2"><p id="p1230154221011"><a name="p1230154221011"></a><a name="p1230154221011"></a>功能描述</p>
</th>
</tr>
</thead>
<tbody><tr id="row4220222125515"><td class="cellrowborder" valign="top" width="37.71%" headers="mcps1.2.3.1.1 "><p id="p741723545516"><a name="p741723545516"></a><a name="p741723545516"></a><a href="DumpTensor.md">DumpTensor</a></p>
</td>
<td class="cellrowborder" valign="top" width="62.29%" headers="mcps1.2.3.1.2 "><p id="p1541723514559"><a name="p1541723514559"></a><a name="p1541723514559"></a>基于算子工程开发的算子，可以使用该接口Dump指定Tensor的内容。</p>
</td>
</tr>
<tr id="row153018215552"><td class="cellrowborder" valign="top" width="37.71%" headers="mcps1.2.3.1.1 "><p id="p144177356555"><a name="p144177356555"></a><a name="p144177356555"></a><a href="printf.md">printf</a></p>
</td>
<td class="cellrowborder" valign="top" width="62.29%" headers="mcps1.2.3.1.2 "><p id="p1941711350553"><a name="p1941711350553"></a><a name="p1941711350553"></a>基于算子工程开发的算子，可以使用该接口实现CPU侧/NPU侧调试场景下的格式化输出功能。</p>
</td>
</tr>
<tr id="row18839203119378"><td class="cellrowborder" valign="top" width="37.71%" headers="mcps1.2.3.1.1 "><p id="p17828375379"><a name="p17828375379"></a><a name="p17828375379"></a><a href="ascendc_assert.md">ascendc_assert</a></p>
</td>
<td class="cellrowborder" valign="top" width="62.29%" headers="mcps1.2.3.1.2 "><p id="p138203733710"><a name="p138203733710"></a><a name="p138203733710"></a><span id="ph5884122310382"><a name="ph5884122310382"></a><a name="ph5884122310382"></a>ascendc_assert<span>提供了一种在CPU/NPU域实现断言功能的接口。当断言条件不满足时，系统会输出断言信息并格式化打印在屏幕上。</span></span></p>
</td>
</tr>
<tr id="row8256721175511"><td class="cellrowborder" valign="top" width="37.71%" headers="mcps1.2.3.1.1 "><p id="p34186352551"><a name="p34186352551"></a><a name="p34186352551"></a><a href="assert.md">assert</a></p>
</td>
<td class="cellrowborder" valign="top" width="62.29%" headers="mcps1.2.3.1.2 "><p id="p1141883535519"><a name="p1141883535519"></a><a name="p1141883535519"></a>基于算子工程开发的算子，可以使用该接口实现CPU/NPU域assert断言功能。</p>
</td>
</tr>
<tr id="row820972155519"><td class="cellrowborder" valign="top" width="37.71%" headers="mcps1.2.3.1.1 "><p id="p0418143555511"><a name="p0418143555511"></a><a name="p0418143555511"></a><a href="DumpAccChkPoint.md">DumpAccChkPoint</a></p>
</td>
<td class="cellrowborder" valign="top" width="62.29%" headers="mcps1.2.3.1.2 "><p id="p9418735175516"><a name="p9418735175516"></a><a name="p9418735175516"></a>基于算子工程开发的算子，可以使用该接口Dump指定Tensor的内容。该接口可以支持指定偏移位置的Tensor打印。</p>
</td>
</tr>
<tr id="row15977214153717"><td class="cellrowborder" valign="top" width="37.71%" headers="mcps1.2.3.1.1 "><p id="p1068951853720"><a name="p1068951853720"></a><a name="p1068951853720"></a><a href="PrintTimeStamp.md">PrintTimeStamp</a></p>
</td>
<td class="cellrowborder" valign="top" width="62.29%" headers="mcps1.2.3.1.2 "><p id="p149786149371"><a name="p149786149371"></a><a name="p149786149371"></a><span id="ph18761185413715"><a name="ph18761185413715"></a><a name="ph18761185413715"></a>提供时间戳打点功能，用于在算子Kernel代码中标记关键执行点。</span></p>
</td>
</tr>
<tr id="row14208102016559"><td class="cellrowborder" valign="top" width="37.71%" headers="mcps1.2.3.1.1 "><p id="p12418123512551"><a name="p12418123512551"></a><a name="p12418123512551"></a><a href="Trap.md">Trap</a></p>
</td>
<td class="cellrowborder" valign="top" width="62.29%" headers="mcps1.2.3.1.2 "><p id="p1941853512557"><a name="p1941853512557"></a><a name="p1941853512557"></a>当软件产生异常后，使用该指令使kernel中止运行。</p>
</td>
</tr>
<tr id="row830164241016"><td class="cellrowborder" valign="top" width="37.71%" headers="mcps1.2.3.1.1 "><p id="p13406234125916"><a name="p13406234125916"></a><a name="p13406234125916"></a><a href="GmAlloc.md">GmAlloc</a></p>
</td>
<td class="cellrowborder" valign="top" width="62.29%" headers="mcps1.2.3.1.2 "><p id="zh-cn_topic_0000001963639306_zh-cn_topic_0000001541764188_p1531015912615"><a name="zh-cn_topic_0000001963639306_zh-cn_topic_0000001541764188_p1531015912615"></a><a name="zh-cn_topic_0000001963639306_zh-cn_topic_0000001541764188_p1531015912615"></a>进行核函数的CPU侧运行验证时，用于创建共享内存：在/tmp目录下创建一个共享文件，并返回该文件的映射指针。</p>
</td>
</tr>
<tr id="row88415865818"><td class="cellrowborder" valign="top" width="37.71%" headers="mcps1.2.3.1.1 "><p id="p161386446595"><a name="p161386446595"></a><a name="p161386446595"></a><a href="ICPU_RUN_KF.md">ICPU_RUN_KF</a></p>
</td>
<td class="cellrowborder" valign="top" width="62.29%" headers="mcps1.2.3.1.2 "><p id="p6851188585"><a name="p6851188585"></a><a name="p6851188585"></a>进行核函数的CPU侧运行验证时，CPU调测总入口，完成CPU侧的算子程序调用。</p>
</td>
</tr>
<tr id="row7702750145910"><td class="cellrowborder" valign="top" width="37.71%" headers="mcps1.2.3.1.1 "><p id="p19702850185918"><a name="p19702850185918"></a><a name="p19702850185918"></a><a href="ICPU_SET_TILING_KEY.md">ICPU_SET_TILING_KEY</a></p>
</td>
<td class="cellrowborder" valign="top" width="62.29%" headers="mcps1.2.3.1.2 "><p id="p10702750175919"><a name="p10702750175919"></a><a name="p10702750175919"></a>用于指定本次CPU调测使用的tilingKey。调测执行时，将只执行算子核函数中该tilingKey对应的分支。</p>
</td>
</tr>
<tr id="row10114145135910"><td class="cellrowborder" valign="top" width="37.71%" headers="mcps1.2.3.1.1 "><p id="p31141451185911"><a name="p31141451185911"></a><a name="p31141451185911"></a><a href="GmFree.md">GmFree</a></p>
</td>
<td class="cellrowborder" valign="top" width="62.29%" headers="mcps1.2.3.1.2 "><p id="p311495195915"><a name="p311495195915"></a><a name="p311495195915"></a>进行核函数的CPU侧运行验证时，用于释放通过GmAlloc申请的共享内存。</p>
</td>
</tr>
<tr id="row1125615511592"><td class="cellrowborder" valign="top" width="37.71%" headers="mcps1.2.3.1.1 "><p id="p16416351306"><a name="p16416351306"></a><a name="p16416351306"></a><a href="SetKernelMode.md">SetKernelMode</a></p>
</td>
<td class="cellrowborder" valign="top" width="62.29%" headers="mcps1.2.3.1.2 "><p id="zh-cn_topic_0000001963639310_zh-cn_topic_0000001656094169_p1118165416116"><a name="zh-cn_topic_0000001963639310_zh-cn_topic_0000001656094169_p1118165416116"></a><a name="zh-cn_topic_0000001963639310_zh-cn_topic_0000001656094169_p1118165416116"></a>CPU调测时，设置内核模式为单AIV模式，单AIC模式或者MIX模式，以分别支持单AIV矢量算子，单AIC矩阵算子，MIX混合算子的CPU调试。</p>
</td>
</tr>
<tr id="row5399851115914"><td class="cellrowborder" valign="top" width="37.71%" headers="mcps1.2.3.1.1 "><p id="p18391515913"><a name="p18391515913"></a><a name="p18391515913"></a><a href="TRACE_START.md">TRACE_START</a></p>
</td>
<td class="cellrowborder" valign="top" width="62.29%" headers="mcps1.2.3.1.2 "><p id="p133991451205919"><a name="p133991451205919"></a><a name="p133991451205919"></a>通过CAModel进行算子性能仿真时，可对算子任意运行阶段打点，从而分析不同指令的流水图，以便进一步性能调优。</p>
<p id="p3807143716119"><a name="p3807143716119"></a><a name="p3807143716119"></a>用于表示起始位置打点，一般与<a href="TRACE_STOP.md">TRACE_STOP</a>配套使用。</p>
</td>
</tr>
<tr id="row125261851195913"><td class="cellrowborder" valign="top" width="37.71%" headers="mcps1.2.3.1.1 "><p id="p66051550125"><a name="p66051550125"></a><a name="p66051550125"></a><a href="TRACE_STOP.md">TRACE_STOP</a></p>
</td>
<td class="cellrowborder" valign="top" width="62.29%" headers="mcps1.2.3.1.2 "><p id="zh-cn_topic_0000002000199857_p15636113710169"><a name="zh-cn_topic_0000002000199857_p15636113710169"></a><a name="zh-cn_topic_0000002000199857_p15636113710169"></a>通过CAModel进行算子性能仿真时，可对算子任意运行阶段打点，从而分析不同指令的流水图，以便进一步性能调优。</p>
<p id="zh-cn_topic_0000002000199857_p868234414164"><a name="zh-cn_topic_0000002000199857_p868234414164"></a><a name="zh-cn_topic_0000002000199857_p868234414164"></a>用于表示终止位置打点，一般与<a href="TRACE_START.md">TRACE_START</a>配套使用。</p>
</td>
</tr>
<tr id="row368519581124"><td class="cellrowborder" valign="top" width="37.71%" headers="mcps1.2.3.1.1 "><p id="p1468611581226"><a name="p1468611581226"></a><a name="p1468611581226"></a><a href="MetricsProfStart.md">MetricsProfStart</a></p>
</td>
<td class="cellrowborder" valign="top" width="62.29%" headers="mcps1.2.3.1.2 "><p id="p16869582024"><a name="p16869582024"></a><a name="p16869582024"></a>用于设置性能数据采集信号启动，和MetricsProfStop配合使用。使用msProf工具进行算子上板调优时，可在kernel侧代码段前后分别调用MetricsProfStart和MetricsProfStop来指定需要调优的代码段范围。</p>
</td>
</tr>
<tr id="row2143959125"><td class="cellrowborder" valign="top" width="37.71%" headers="mcps1.2.3.1.1 "><p id="p4507261733"><a name="p4507261733"></a><a name="p4507261733"></a><a href="MetricsProfStop.md">MetricsProfStop</a></p>
</td>
<td class="cellrowborder" valign="top" width="62.29%" headers="mcps1.2.3.1.2 "><p id="zh-cn_topic_0000002000280001_zh-cn_topic_0000001960477980_p279125418620"><a name="zh-cn_topic_0000002000280001_zh-cn_topic_0000001960477980_p279125418620"></a><a name="zh-cn_topic_0000002000280001_zh-cn_topic_0000001960477980_p279125418620"></a>设置性能数据采集信号停止，和MetricsProfStart配合使用。使用msProf工具进行算子上板调优时，可在kernel侧代码段前后分别调用MetricsProfStart和MetricsProfStop来指定需要调优的代码段范围。</p>
</td>
</tr>
</tbody>
</table>

**表 11**  工具函数接口列表

<a name="table9496143191816"></a>
<table><thead align="left"><tr id="row14962043181812"><th class="cellrowborder" valign="top" width="40.089999999999996%" id="mcps1.2.3.1.1"><p id="p449784310180"><a name="p449784310180"></a><a name="p449784310180"></a>接口名</p>
</th>
<th class="cellrowborder" valign="top" width="59.91%" id="mcps1.2.3.1.2"><p id="p17497204317180"><a name="p17497204317180"></a><a name="p17497204317180"></a>功能描述</p>
</th>
</tr>
</thead>
<tbody><tr id="row649710437184"><td class="cellrowborder" valign="top" width="40.089999999999996%" headers="mcps1.2.3.1.1 "><p id="p149744319183"><a name="p149744319183"></a><a name="p149744319183"></a><a href="Async.md">Async</a></p>
</td>
<td class="cellrowborder" valign="top" width="59.91%" headers="mcps1.2.3.1.2 "><p id="p816712203211"><a name="p816712203211"></a><a name="p816712203211"></a><span id="ph585624314195"><a name="ph585624314195"></a><a name="ph585624314195"></a>Async提供了一个统一的接口，用于在不同模式下（AIC或AIV）执行特定函数，从而避免代码中直接的硬件条件判断（如使用ASCEND_IS_AIV或ASCEND_IS_AIC）。</span></p>
</td>
</tr>
<tr id="row2010610593211"><td class="cellrowborder" valign="top" width="40.089999999999996%" headers="mcps1.2.3.1.1 "><p id="p317115102325"><a name="p317115102325"></a><a name="p317115102325"></a><a href="NumericLimits.md">NumericLimits</a></p>
</td>
<td class="cellrowborder" valign="top" width="59.91%" headers="mcps1.2.3.1.2 "><p id="p17106857321"><a name="p17106857321"></a><a name="p17106857321"></a><span id="ph1575202123315"><a name="ph1575202123315"></a><a name="ph1575202123315"></a>NumericLimits工具类，用于查询指定数据类型的最大值/最小值等属性。</span></p>
</td>
</tr>
<tr id="row154197811816"><td class="cellrowborder" valign="top" width="40.089999999999996%" headers="mcps1.2.3.1.1 "><p id="p2094712054713"><a name="p2094712054713"></a><a name="p2094712054713"></a><a href="GetTaskRatio.md">GetTaskRatio</a></p>
</td>
<td class="cellrowborder" valign="top" width="59.91%" headers="mcps1.2.3.1.2 "><p id="p29729544501"><a name="p29729544501"></a><a name="p29729544501"></a>适用于Cube/Vector分离模式，用来获取Cube/Vector的配比。</p>
</td>
</tr>
<tr id="row12385124516283"><td class="cellrowborder" valign="top" width="40.089999999999996%" headers="mcps1.2.3.1.1 "><p id="p17516125692817"><a name="p17516125692817"></a><a name="p17516125692817"></a><a href="GetUBSizeInBytes.md">GetUBSizeInBytes</a></p>
</td>
<td class="cellrowborder" valign="top" width="59.91%" headers="mcps1.2.3.1.2 "><p id="p17516165612819"><a name="p17516165612819"></a><a name="p17516165612819"></a><span id="ph251617564281"><a name="ph251617564281"></a><a name="ph251617564281"></a>获取UB空间的大小，单位为byte。</span></p>
</td>
</tr>
<tr id="row34503263319"><td class="cellrowborder" valign="top" width="40.089999999999996%" headers="mcps1.2.3.1.1 "><p id="p543714258341"><a name="p543714258341"></a><a name="p543714258341"></a><a href="GetRuntimeUBSize.md">GetRuntimeUBSize</a></p>
</td>
<td class="cellrowborder" valign="top" width="59.91%" headers="mcps1.2.3.1.2 "><p id="p9464325332"><a name="p9464325332"></a><a name="p9464325332"></a><span id="ph11371104916159"><a name="ph11371104916159"></a><a name="ph11371104916159"></a>获取运行时UB空间的大小，单位为byte。</span>开发者根据UB的大小来计算循环次数等参数值。</p>
</td>
</tr>
</tbody>
</table>

**表 12**  Kernel Tiling接口列表

<a name="table2017815711517"></a>
<table><thead align="left"><tr id="row11179357358"><th class="cellrowborder" valign="top" width="39.900000000000006%" id="mcps1.2.3.1.1"><p id="p417975719517"><a name="p417975719517"></a><a name="p417975719517"></a>接口名</p>
</th>
<th class="cellrowborder" valign="top" width="60.099999999999994%" id="mcps1.2.3.1.2"><p id="p1817910573513"><a name="p1817910573513"></a><a name="p1817910573513"></a>功能描述</p>
</th>
</tr>
</thead>
<tbody><tr id="row14180457256"><td class="cellrowborder" valign="top" width="39.900000000000006%" headers="mcps1.2.3.1.1 "><p id="p1518025711512"><a name="p1518025711512"></a><a name="p1518025711512"></a><a href="GET_TILING_DATA.md">GET_TILING_DATA</a></p>
</td>
<td class="cellrowborder" valign="top" width="60.099999999999994%" headers="mcps1.2.3.1.2 "><p id="p818095714515"><a name="p818095714515"></a><a name="p818095714515"></a>用于获取算子kernel入口函数传入的tiling信息，并填入注册的Tiling结构体中，此函数会以宏展开的方式进行编译。如果用户注册了多个TilingData结构体，使用该接口返回默认注册的结构体。</p>
</td>
</tr>
<tr id="row165593514011"><td class="cellrowborder" valign="top" width="39.900000000000006%" headers="mcps1.2.3.1.1 "><p id="p155603514401"><a name="p155603514401"></a><a name="p155603514401"></a><a href="GET_TILING_DATA_WITH_STRUCT.md">GET_TILING_DATA_WITH_STRUCT</a></p>
</td>
<td class="cellrowborder" valign="top" width="60.099999999999994%" headers="mcps1.2.3.1.2 "><p id="p10569356405"><a name="p10569356405"></a><a name="p10569356405"></a>使用该接口指定结构体名称，可获取指定的tiling信息，并填入对应的Tiling结构体中，此函数会以宏展开的方式进行编译。</p>
</td>
</tr>
<tr id="row146553974019"><td class="cellrowborder" valign="top" width="39.900000000000006%" headers="mcps1.2.3.1.1 "><p id="p246512399407"><a name="p246512399407"></a><a name="p246512399407"></a><a href="GET_TILING_DATA_MEMBER.md">GET_TILING_DATA_MEMBER</a></p>
</td>
<td class="cellrowborder" valign="top" width="60.099999999999994%" headers="mcps1.2.3.1.2 "><p id="p18466133917406"><a name="p18466133917406"></a><a name="p18466133917406"></a>用于获取tiling结构体的成员变量。</p>
</td>
</tr>
<tr id="row818045715513"><td class="cellrowborder" valign="top" width="39.900000000000006%" headers="mcps1.2.3.1.1 "><p id="p1318010571658"><a name="p1318010571658"></a><a name="p1318010571658"></a><a href="TILING_KEY_IS.md">TILING_KEY_IS</a></p>
</td>
<td class="cellrowborder" valign="top" width="60.099999999999994%" headers="mcps1.2.3.1.2 "><p id="p41805570511"><a name="p41805570511"></a><a name="p41805570511"></a>在核函数中判断本次执行时的tiling_key是否等于某个key，从而标识tiling_key==key的一条kernel分支。</p>
</td>
</tr>
<tr id="row930185314421"><td class="cellrowborder" valign="top" width="39.900000000000006%" headers="mcps1.2.3.1.1 "><p id="p1830185311424"><a name="p1830185311424"></a><a name="p1830185311424"></a><a href="REGISTER_TILING_DEFAULT.md">REGISTER_TILING_DEFAULT</a></p>
</td>
<td class="cellrowborder" valign="top" width="60.099999999999994%" headers="mcps1.2.3.1.2 "><p id="p63075319421"><a name="p63075319421"></a><a name="p63075319421"></a>用于在kernel侧注册用户使用标准C++语法自定义的默认TilingData结构体。</p>
</td>
</tr>
<tr id="row2047417552427"><td class="cellrowborder" valign="top" width="39.900000000000006%" headers="mcps1.2.3.1.1 "><p id="p1147415594219"><a name="p1147415594219"></a><a name="p1147415594219"></a><a href="REGISTER_TILING_FOR_TILINGKEY.md">REGISTER_TILING_FOR_TILINGKEY</a></p>
</td>
<td class="cellrowborder" valign="top" width="60.099999999999994%" headers="mcps1.2.3.1.2 "><p id="zh-cn_topic_0000001526206862_p0824144014589"><a name="zh-cn_topic_0000001526206862_p0824144014589"></a><a name="zh-cn_topic_0000001526206862_p0824144014589"></a>用于在kernel侧注册与TilingKey相匹配的TilingData自定义结构体；该接口需提供一个逻辑表达式，逻辑表达式以字符串“TILING_KEY_VAR”代指实际TilingKey，表达TIlingKey所满足的范围。</p>
</td>
</tr>
<tr id="row8142114933417"><td class="cellrowborder" valign="top" width="39.900000000000006%" headers="mcps1.2.3.1.1 "><p id="p187745415347"><a name="p187745415347"></a><a name="p187745415347"></a><a href="REGISTER_NONE_TILING.md">REGISTER_NONE_TILING</a></p>
</td>
<td class="cellrowborder" valign="top" width="60.099999999999994%" headers="mcps1.2.3.1.2 "><p id="p1131075619124"><a name="p1131075619124"></a><a name="p1131075619124"></a><span>在Kernel侧使用标准C++语法自定义的TilingData结构体时，若用户不确定需要注册哪些结构体，可使用该接口</span>告知框架侧需使用未注册的标准C++语法来定义TilingData，并配套<a href="GET_TILING_DATA_WITH_STRUCT.md">GET_TILING_DATA_WITH_STRUCT</a>，<a href="GET_TILING_DATA_MEMBER.md">GET_TILING_DATA_MEMBER</a>，<a href="GET_TILING_DATA_PTR_WITH_STRUCT.md">GET_TILING_DATA_PTR_WITH_STRUCT</a>来获取对应的TilingData。</p>
</td>
</tr>
<tr id="row131802571959"><td class="cellrowborder" valign="top" width="39.900000000000006%" headers="mcps1.2.3.1.1 "><p id="p51807572055"><a name="p51807572055"></a><a name="p51807572055"></a><a href="设置Kernel类型.md">KERNEL_TASK_TYPE_DEFAULT</a></p>
</td>
<td class="cellrowborder" valign="top" width="60.099999999999994%" headers="mcps1.2.3.1.2 "><p id="p1618045718520"><a name="p1618045718520"></a><a name="p1618045718520"></a>设置全局默认的kernel type，对所有的tiling key生效。</p>
</td>
</tr>
<tr id="row21801457154"><td class="cellrowborder" valign="top" width="39.900000000000006%" headers="mcps1.2.3.1.1 "><p id="p0180185713514"><a name="p0180185713514"></a><a name="p0180185713514"></a><a href="设置Kernel类型.md">KERNEL_TASK_TYPE</a></p>
</td>
<td class="cellrowborder" valign="top" width="60.099999999999994%" headers="mcps1.2.3.1.2 "><p id="p201806571756"><a name="p201806571756"></a><a name="p201806571756"></a>设置某一个具体的tiling key对应的kernel type。</p>
</td>
</tr>
</tbody>
</table>

**表 13**  ISASI接口列表

<a name="table19526741203211"></a>
<table><thead align="left"><tr id="row352624118322"><th class="cellrowborder" valign="top" width="12.379999999999999%" id="mcps1.2.4.1.1"><p id="p88065174816"><a name="p88065174816"></a><a name="p88065174816"></a>分类</p>
</th>
<th class="cellrowborder" valign="top" width="27.63%" id="mcps1.2.4.1.2"><p id="p14526241173218"><a name="p14526241173218"></a><a name="p14526241173218"></a>接口名</p>
</th>
<th class="cellrowborder" valign="top" width="59.99%" id="mcps1.2.4.1.3"><p id="p145261141203210"><a name="p145261141203210"></a><a name="p145261141203210"></a>功能描述</p>
</th>
</tr>
</thead>
<tbody><tr id="row1216412458149"><td class="cellrowborder" rowspan="2" valign="top" width="12.379999999999999%" headers="mcps1.2.4.1.1 "><p id="p259453711267"><a name="p259453711267"></a><a name="p259453711267"></a>标量计算</p>
</td>
<td class="cellrowborder" valign="top" width="27.63%" headers="mcps1.2.4.1.2 "><p id="p5421318131512"><a name="p5421318131512"></a><a name="p5421318131512"></a><a href="WriteGmByPassDCache(ISASI).md">WriteGmByPassDCache</a></p>
</td>
<td class="cellrowborder" valign="top" width="59.99%" headers="mcps1.2.4.1.3 "><p id="p204221851520"><a name="p204221851520"></a><a name="p204221851520"></a><span id="ph94221821518"><a name="ph94221821518"></a><a name="ph94221821518"></a>不经过DCache向GM地址上写数据。</span></p>
</td>
</tr>
<tr id="row69501477146"><td class="cellrowborder" valign="top" headers="mcps1.2.4.1.1 "><p id="p74217189157"><a name="p74217189157"></a><a name="p74217189157"></a><a href="ReadGmByPassDCache(ISASI).md">ReadGmByPassDCache</a></p>
</td>
<td class="cellrowborder" valign="top" headers="mcps1.2.4.1.2 "><p id="p19422018191516"><a name="p19422018191516"></a><a name="p19422018191516"></a><span id="ph14211188156"><a name="ph14211188156"></a><a name="ph14211188156"></a>不经过DCache从GM地址上读数据。</span></p>
</td>
</tr>
<tr id="row154401144114718"><td class="cellrowborder" rowspan="15" valign="top" width="12.379999999999999%" headers="mcps1.2.4.1.1 "><p id="p12441154416479"><a name="p12441154416479"></a><a name="p12441154416479"></a>矢量计算</p>
</td>
<td class="cellrowborder" valign="top" width="27.63%" headers="mcps1.2.4.1.2 "><p id="p20441544114710"><a name="p20441544114710"></a><a name="p20441544114710"></a><a href="VectorPadding(ISASI).md">VectorPadding</a></p>
</td>
<td class="cellrowborder" valign="top" width="59.99%" headers="mcps1.2.4.1.3 "><p id="p244134434718"><a name="p244134434718"></a><a name="p244134434718"></a>根据padMode（pad模式）与padSide（pad方向）对源操作数按照datablock进行填充操作。</p>
</td>
</tr>
<tr id="row10526441173211"><td class="cellrowborder" valign="top" headers="mcps1.2.4.1.1 "><p id="p1994405814488"><a name="p1994405814488"></a><a name="p1994405814488"></a><a href="BilinearInterpolation(ISASI).md">BilinearInterpolation</a></p>
</td>
<td class="cellrowborder" valign="top" headers="mcps1.2.4.1.2 "><p id="p1526144123212"><a name="p1526144123212"></a><a name="p1526144123212"></a><span>双线性插值操作，分为垂直迭代和水平迭代。</span></p>
</td>
</tr>
<tr id="row1952734143214"><td class="cellrowborder" valign="top" headers="mcps1.2.4.1.1 "><p id="p99449586481"><a name="p99449586481"></a><a name="p99449586481"></a><a href="GetCmpMask(ISASI).md">GetCmpMask</a></p>
</td>
<td class="cellrowborder" valign="top" headers="mcps1.2.4.1.2 "><p id="p1152720414327"><a name="p1152720414327"></a><a name="p1152720414327"></a>获取<a href="Compare（结果存入寄存器）.md">Compare（结果存入寄存器）</a>指令的比较结果。</p>
</td>
</tr>
<tr id="row946564134915"><td class="cellrowborder" valign="top" headers="mcps1.2.4.1.1 "><p id="p5465104110496"><a name="p5465104110496"></a><a name="p5465104110496"></a><a href="SetCmpMask(ISASI).md">SetCmpMask</a></p>
</td>
<td class="cellrowborder" valign="top" headers="mcps1.2.4.1.2 "><p id="p184655418496"><a name="p184655418496"></a><a name="p184655418496"></a>为<a href="Select.md">Select</a>不传入mask参数的接口设置比较寄存器。</p>
</td>
</tr>
<tr id="row1952774119326"><td class="cellrowborder" valign="top" headers="mcps1.2.4.1.1 "><p id="p8429183014919"><a name="p8429183014919"></a><a name="p8429183014919"></a><a href="GetReduceRepeatSumSpr(ISASI).md">GetReduceRepeatSumSpr</a></p>
</td>
<td class="cellrowborder" valign="top" headers="mcps1.2.4.1.2 "><p id="p165272418326"><a name="p165272418326"></a><a name="p165272418326"></a>获取<a href="ReduceSum.md">ReduceSum</a>（针对tensor前n个数据计算）接口的计算结果。</p>
</td>
</tr>
<tr id="row125271041103210"><td class="cellrowborder" valign="top" headers="mcps1.2.4.1.1 "><p id="p12638205318500"><a name="p12638205318500"></a><a name="p12638205318500"></a><a href="GetReduceRepeatMaxMinSpr(ISASI).md">GetReduceRepeatMaxMinSpr</a></p>
</td>
<td class="cellrowborder" valign="top" headers="mcps1.2.4.1.2 "><p id="p195452267225"><a name="p195452267225"></a><a name="p195452267225"></a>获取<a href="ReduceMax.md">ReduceMax</a>、<a href="ReduceMin.md">ReduceMin</a>连续场景下的最大/最小值以及相应的索引值。</p>
</td>
</tr>
<tr id="row185271941123212"><td class="cellrowborder" valign="top" headers="mcps1.2.4.1.1 "><p id="p169441258134818"><a name="p169441258134818"></a><a name="p169441258134818"></a><a href="ProposalConcat.md">ProposalConcat</a></p>
</td>
<td class="cellrowborder" valign="top" headers="mcps1.2.4.1.2 "><p id="p14527241173218"><a name="p14527241173218"></a><a name="p14527241173218"></a>将连续元素合入Region Proposal内对应位置，每次迭代会将16个连续元素合入到16个Region Proposals的对应位置里。</p>
</td>
</tr>
<tr id="row13527134173211"><td class="cellrowborder" valign="top" headers="mcps1.2.4.1.1 "><p id="p39441158184816"><a name="p39441158184816"></a><a name="p39441158184816"></a><a href="ProposalExtract.md">ProposalExtract</a></p>
</td>
<td class="cellrowborder" valign="top" headers="mcps1.2.4.1.2 "><p id="p10462291478"><a name="p10462291478"></a><a name="p10462291478"></a>与ProposalConcat功能相反，从Region Proposals内将相应位置的单个元素抽取后重排，每次迭代处理16个Region Proposals，抽取16个元素后连续排列。</p>
</td>
</tr>
<tr id="row452716417321"><td class="cellrowborder" valign="top" headers="mcps1.2.4.1.1 "><p id="p6944165817489"><a name="p6944165817489"></a><a name="p6944165817489"></a><a href="RpSort16.md">RpSort16</a></p>
</td>
<td class="cellrowborder" valign="top" headers="mcps1.2.4.1.2 "><p id="p773018265170"><a name="p773018265170"></a><a name="p773018265170"></a>根据Region Proposals中的score域对其进行排序（score大的排前面），每次排16个Region Proposals。</p>
</td>
</tr>
<tr id="row745884215480"><td class="cellrowborder" valign="top" headers="mcps1.2.4.1.1 "><p id="p139441358114819"><a name="p139441358114819"></a><a name="p139441358114819"></a><a href="MrgSort4.md">MrgSort4</a></p>
</td>
<td class="cellrowborder" valign="top" headers="mcps1.2.4.1.2 "><p id="p101781237141713"><a name="p101781237141713"></a><a name="p101781237141713"></a>将已经排好序的最多4 条region proposals队列，排列并合并成1条队列，结果按照score域由大到小排序。</p>
</td>
</tr>
<tr id="row1019917563485"><td class="cellrowborder" valign="top" headers="mcps1.2.4.1.1 "><p id="p594585811484"><a name="p594585811484"></a><a name="p594585811484"></a><a href="Sort32.md">Sort32</a></p>
</td>
<td class="cellrowborder" valign="top" headers="mcps1.2.4.1.2 "><p id="p151991556204813"><a name="p151991556204813"></a><a name="p151991556204813"></a>排序函数，一次迭代可以完成32个数的排序。</p>
</td>
</tr>
<tr id="row18337125610483"><td class="cellrowborder" valign="top" headers="mcps1.2.4.1.1 "><p id="p10945175864811"><a name="p10945175864811"></a><a name="p10945175864811"></a><a href="MrgSort.md">MrgSort</a></p>
</td>
<td class="cellrowborder" valign="top" headers="mcps1.2.4.1.2 "><p id="p78578171182"><a name="p78578171182"></a><a name="p78578171182"></a>将已经排好序的最多4 条队列，合并排列成 1 条队列，结果按照score域由大到小排序。</p>
</td>
</tr>
<tr id="row8465195684815"><td class="cellrowborder" valign="top" headers="mcps1.2.4.1.1 "><p id="p139456584485"><a name="p139456584485"></a><a name="p139456584485"></a><a href="GetMrgSortResult.md">GetMrgSortResult</a></p>
</td>
<td class="cellrowborder" valign="top" headers="mcps1.2.4.1.2 "><p id="p101531828111815"><a name="p101531828111815"></a><a name="p101531828111815"></a>获取<a href="MrgSort.md">MrgSort</a>或<a href="MrgSort4.md">MrgSort4</a>已经处理过的队列里的Region Proposal个数，并依次存储在四个List入参中。</p>
</td>
</tr>
<tr id="row46098568481"><td class="cellrowborder" valign="top" headers="mcps1.2.4.1.1 "><p id="p1794575816483"><a name="p1794575816483"></a><a name="p1794575816483"></a><a href="Gatherb(ISASI).md">Gatherb</a></p>
</td>
<td class="cellrowborder" valign="top" headers="mcps1.2.4.1.2 "><p id="p4609556174812"><a name="p4609556174812"></a><a name="p4609556174812"></a>给定一个输入的张量和一个地址偏移张量，Gatherb指令根据偏移地址将输入张量收集到结果张量中。</p>
</td>
</tr>
<tr id="row576405615486"><td class="cellrowborder" valign="top" headers="mcps1.2.4.1.1 "><p id="p1945158154815"><a name="p1945158154815"></a><a name="p1945158154815"></a><a href="Scatter(ISASI).md">Scatter</a></p>
</td>
<td class="cellrowborder" valign="top" headers="mcps1.2.4.1.2 "><p id="p1836365951815"><a name="p1836365951815"></a><a name="p1836365951815"></a>给定一个连续的输入张量和一个目的地址偏移张量，Scatter指令根据偏移地址生成新的结果张量后将输入张量分散到结果张量中。</p>
</td>
</tr>
<tr id="row55961050102215"><td class="cellrowborder" rowspan="8" valign="top" width="12.379999999999999%" headers="mcps1.2.4.1.1 "><p id="p3596650162219"><a name="p3596650162219"></a><a name="p3596650162219"></a>矢量计算</p>
<p id="p94614512565"><a name="p94614512565"></a><a name="p94614512565"></a></p>
<p id="p4570181317114"><a name="p4570181317114"></a><a name="p4570181317114"></a></p>
<p id="p162501316917"><a name="p162501316917"></a><a name="p162501316917"></a></p>
</td>
<td class="cellrowborder" valign="top" width="27.63%" headers="mcps1.2.4.1.2 "><p id="p1627401015234"><a name="p1627401015234"></a><a name="p1627401015234"></a><a href="Prelu(ISASI).md">Prelu</a></p>
</td>
<td class="cellrowborder" valign="top" width="59.99%" headers="mcps1.2.4.1.3 "><p id="p17596550132219"><a name="p17596550132219"></a><a name="p17596550132219"></a><span id="ph171052419253"><a name="ph171052419253"></a><a name="ph171052419253"></a>源操作数src0大于0的情况下直接将src0写入目的操作数dst，否则将源操作数src0 * src1的结果写入dst。</span></p>
</td>
</tr>
<tr id="row10490181412234"><td class="cellrowborder" valign="top" headers="mcps1.2.4.1.1 "><p id="p768371913238"><a name="p768371913238"></a><a name="p768371913238"></a><a href="Mull(ISASI).md">Mull</a></p>
</td>
<td class="cellrowborder" valign="top" headers="mcps1.2.4.1.2 "><p id="p17491151412313"><a name="p17491151412313"></a><a name="p17491151412313"></a><span id="ph711422918258"><a name="ph711422918258"></a><a name="ph711422918258"></a>对前count个输入数据src0、src1按元素相乘操作，将结果写入dst0Local，溢出部分写入dst1Local。</span></p>
</td>
</tr>
<tr id="row610542433612"><td class="cellrowborder" valign="top" headers="mcps1.2.4.1.1 "><p id="p17573105103717"><a name="p17573105103717"></a><a name="p17573105103717"></a><a href="AbsSub(ISASI).md">AbsSub</a></p>
</td>
<td class="cellrowborder" valign="top" headers="mcps1.2.4.1.2 "><p id="p1310512414366"><a name="p1310512414366"></a><a name="p1310512414366"></a><span id="ph4885524194011"><a name="ph4885524194011"></a><a name="ph4885524194011"></a>将src0Local与src1相减再求绝对值， 并将计算结果写入dst。</span></p>
</td>
</tr>
<tr id="row21531226203613"><td class="cellrowborder" valign="top" headers="mcps1.2.4.1.1 "><p id="p968312193716"><a name="p968312193716"></a><a name="p968312193716"></a><a href="ExpSub(ISASI).md">ExpSub</a></p>
</td>
<td class="cellrowborder" valign="top" headers="mcps1.2.4.1.2 "><p id="p15153126143614"><a name="p15153126143614"></a><a name="p15153126143614"></a><span id="ph2080081312419"><a name="ph2080081312419"></a><a name="ph2080081312419"></a>src0与src1相减，将差值作为指数计算自然常数e的幂次， 并将计算结果写入dst。</span></p>
</td>
</tr>
<tr id="row18177628103613"><td class="cellrowborder" valign="top" headers="mcps1.2.4.1.1 "><p id="p128991739174417"><a name="p128991739174417"></a><a name="p128991739174417"></a><a href="MulCast.md">MulsCast</a></p>
</td>
<td class="cellrowborder" valign="top" headers="mcps1.2.4.1.2 "><p id="p121781328103612"><a name="p121781328103612"></a><a name="p121781328103612"></a><span id="ph17644543174116"><a name="ph17644543174116"></a><a name="ph17644543174116"></a>将矢量源操作数前count个数据与标量相乘再按照CAST_ROUND模式转换成half类型， 并将计算结果写入dst，此接口支持标量在前和标量在后两种场景。</span></p>
</td>
</tr>
<tr id="row11469585616"><td class="cellrowborder" valign="top" headers="mcps1.2.4.1.1 "><p id="p522018118566"><a name="p522018118566"></a><a name="p522018118566"></a><a href="Truncate(ISASI).md">Truncate</a></p>
</td>
<td class="cellrowborder" valign="top" headers="mcps1.2.4.1.2 "><p id="p74618519564"><a name="p74618519564"></a><a name="p74618519564"></a><span id="ph12508132815572"><a name="ph12508132815572"></a><a name="ph12508132815572"></a>将源操作数的浮点数元素截断到整数位，同时源操作数的数据类型保持不变。</span></p>
</td>
</tr>
<tr id="row195707131411"><td class="cellrowborder" valign="top" headers="mcps1.2.4.1.1 "><p id="p1434016281613"><a name="p1434016281613"></a><a name="p1434016281613"></a><a href="Interleave.md">Interleave</a></p>
</td>
<td class="cellrowborder" valign="top" headers="mcps1.2.4.1.2 "><p id="p657010139118"><a name="p657010139118"></a><a name="p657010139118"></a><span id="ph664518360219"><a name="ph664518360219"></a><a name="ph664518360219"></a>给定源操作数src0和src1，将src0和src1中的元素交织存入结果操作数dst0和dst1中。</span></p>
</td>
</tr>
<tr id="row19250111610113"><td class="cellrowborder" valign="top" headers="mcps1.2.4.1.1 "><p id="p184274411111"><a name="p184274411111"></a><a name="p184274411111"></a><a href="DeInterleave.md">DeInterleave</a></p>
</td>
<td class="cellrowborder" valign="top" headers="mcps1.2.4.1.2 "><p id="p22501166119"><a name="p22501166119"></a><a name="p22501166119"></a><span id="ph054752519311"><a name="ph054752519311"></a><a name="ph054752519311"></a>给定源操作数src0和src1，将src0和src1中的元素解交织存入结果操作数dst0和dst1中。</span></p>
</td>
</tr>
<tr id="row224111331412"><td class="cellrowborder" rowspan="2" valign="top" width="12.379999999999999%" headers="mcps1.2.4.1.1 "><p id="p711814301143"><a name="p711814301143"></a><a name="p711814301143"></a>数据搬运</p>
</td>
<td class="cellrowborder" valign="top" width="27.63%" headers="mcps1.2.4.1.2 "><p id="p148821245184915"><a name="p148821245184915"></a><a name="p148821245184915"></a><a href="DataCopyPad(ISASI).md">DataCopyPad</a></p>
</td>
<td class="cellrowborder" valign="top" width="59.99%" headers="mcps1.2.4.1.3 "><p id="p019795716484"><a name="p019795716484"></a><a name="p019795716484"></a>该接口提供数据非对齐搬运的功能。</p>
</td>
</tr>
<tr id="row7375515151412"><td class="cellrowborder" valign="top" headers="mcps1.2.4.1.1 "><p id="p148821245194911"><a name="p148821245194911"></a><a name="p148821245194911"></a><a href="SetPadValue(ISASI).md">SetPadValue</a></p>
</td>
<td class="cellrowborder" valign="top" headers="mcps1.2.4.1.2 "><p id="p6689192111598"><a name="p6689192111598"></a><a name="p6689192111598"></a>设置DataCopyPad接口填充的数值。</p>
</td>
</tr>
<tr id="row154051221164915"><td class="cellrowborder" rowspan="26" valign="top" width="12.379999999999999%" headers="mcps1.2.4.1.1 "><p id="p118921556154813"><a name="p118921556154813"></a><a name="p118921556154813"></a>矩阵计算</p>
</td>
<td class="cellrowborder" valign="top" width="27.63%" headers="mcps1.2.4.1.2 "><p id="p1255212302496"><a name="p1255212302496"></a><a name="p1255212302496"></a><a href="Mmad.md">Mmad</a></p>
</td>
<td class="cellrowborder" valign="top" width="59.99%" headers="mcps1.2.4.1.3 "><p id="p24861371569"><a name="p24861371569"></a><a name="p24861371569"></a>完成矩阵乘加操作。</p>
</td>
</tr>
<tr id="row12765526165715"><td class="cellrowborder" valign="top" headers="mcps1.2.4.1.1 "><p id="p1176511267573"><a name="p1176511267573"></a><a name="p1176511267573"></a><a href="MmadWithSparse.md">MmadWithSparse</a></p>
</td>
<td class="cellrowborder" valign="top" headers="mcps1.2.4.1.2 "><p id="p47651826105712"><a name="p47651826105712"></a><a name="p47651826105712"></a>完成矩阵乘加操作，传入的左矩阵A为稀疏矩阵， 右矩阵B为稠密矩阵 。</p>
</td>
</tr>
<tr id="row0435522154918"><td class="cellrowborder" valign="top" headers="mcps1.2.4.1.1 "><p id="p8552203094910"><a name="p8552203094910"></a><a name="p8552203094910"></a><a href="SetHF32Mode.md">SetHF32Mode</a></p>
</td>
<td class="cellrowborder" valign="top" headers="mcps1.2.4.1.2 "><p id="p64161631122120"><a name="p64161631122120"></a><a name="p64161631122120"></a>此接口同<a href="SetHF32TransMode.md">SetHF32TransMode</a>、<a href="SetMMRowMajor.md">SetMMRowMajor</a>以及<a href="SetMMColumnMajor.md">SetMMColumnMajor</a>一样，都用于设置寄存器的值。SetHF32Mode接口用于设置MMAD的HF32模式。</p>
</td>
</tr>
<tr id="row1858982214918"><td class="cellrowborder" valign="top" headers="mcps1.2.4.1.1 "><p id="p165526308490"><a name="p165526308490"></a><a name="p165526308490"></a><a href="SetHF32TransMode.md">SetHF32TransMode</a></p>
</td>
<td class="cellrowborder" valign="top" headers="mcps1.2.4.1.2 "><p id="p1478477185019"><a name="p1478477185019"></a><a name="p1478477185019"></a>此接口同<a href="SetHF32Mode.md">SetHF32Mode</a>、<a href="SetMMRowMajor.md">SetMMRowMajor</a>以及<a href="SetMMColumnMajor.md">SetMMColumnMajor</a>一样，都用于设置寄存器的值。SetHF32TransMode用于设置MMAD的HF32取整模式，仅在MMAD的HF32模式生效时有效。</p>
</td>
</tr>
<tr id="row1374212216499"><td class="cellrowborder" valign="top" headers="mcps1.2.4.1.1 "><p id="p19945205916295"><a name="p19945205916295"></a><a name="p19945205916295"></a><a href="SetMMRowMajor.md">SetMMRowMajor</a></p>
</td>
<td class="cellrowborder" valign="top" headers="mcps1.2.4.1.2 "><p id="p87421422144917"><a name="p87421422144917"></a><a name="p87421422144917"></a>此接口同<a href="SetHF32Mode.md">SetHF32Mode</a>、<a href="SetHF32TransMode.md">SetHF32TransMode</a>一样，都用于设置寄存器的值，本接口用于设置MMAD计算时优先通过N方向。</p>
</td>
</tr>
<tr id="row9217141164613"><td class="cellrowborder" valign="top" headers="mcps1.2.4.1.1 "><p id="p14218134119466"><a name="p14218134119466"></a><a name="p14218134119466"></a><a href="SetMMColumnMajor.md">SetMMColumnMajor</a></p>
</td>
<td class="cellrowborder" valign="top" headers="mcps1.2.4.1.2 "><p id="p18317459134619"><a name="p18317459134619"></a><a name="p18317459134619"></a>此接口同<a href="SetHF32Mode.md">SetHF32Mode</a>、<a href="SetHF32TransMode.md">SetHF32TransMode</a>一样，都用于设置寄存器的值，本接口用于设置MMAD计算时优先通过M方向。</p>
</td>
</tr>
<tr id="row19411232491"><td class="cellrowborder" valign="top" headers="mcps1.2.4.1.1 "><p id="p20552113020499"><a name="p20552113020499"></a><a name="p20552113020499"></a><a href="Conv2D（废弃）.md">Conv2D</a></p>
</td>
<td class="cellrowborder" valign="top" headers="mcps1.2.4.1.2 "><p id="p10261152153414"><a name="p10261152153414"></a><a name="p10261152153414"></a>计算给定输入张量和权重张量的2-D卷积，输出结果张量。Conv2d卷积层多用于图像识别，使用过滤器提取图像中的特征。</p>
</td>
</tr>
<tr id="row144112573483"><td class="cellrowborder" valign="top" headers="mcps1.2.4.1.1 "><p id="p9552930204920"><a name="p9552930204920"></a><a name="p9552930204920"></a><a href="Gemm（废弃）.md">Gemm</a></p>
</td>
<td class="cellrowborder" valign="top" headers="mcps1.2.4.1.2 "><p id="p14116577483"><a name="p14116577483"></a><a name="p14116577483"></a>根据输入的切分规则，将给定的两个输入张量做矩阵乘，输出至结果张量。将A和B两个输入矩阵乘法在一起，得到一个输出矩阵C。</p>
</td>
</tr>
<tr id="row147451115151916"><td class="cellrowborder" valign="top" headers="mcps1.2.4.1.1 "><p id="p45523300491"><a name="p45523300491"></a><a name="p45523300491"></a><a href="SetFixPipeConfig.md">SetFixPipeConfig</a></p>
</td>
<td class="cellrowborder" valign="top" headers="mcps1.2.4.1.2 "><p id="p17851154862112"><a name="p17851154862112"></a><a name="p17851154862112"></a><a href="随路量化激活搬运.md">DataCopy</a>（CO1-&gt;GM、CO1-&gt;A1）过程中进行随路量化时，通过调用该接口设置量化流程中tensor量化参数。</p>
</td>
</tr>
<tr id="row294174016209"><td class="cellrowborder" valign="top" headers="mcps1.2.4.1.1 "><p id="p65522030174916"><a name="p65522030174916"></a><a name="p65522030174916"></a><a href="SetFixpipeNz2ndFlag.md">SetFixpipeNz2ndFlag</a></p>
</td>
<td class="cellrowborder" valign="top" headers="mcps1.2.4.1.2 "><p id="p2898124417263"><a name="p2898124417263"></a><a name="p2898124417263"></a><a href="随路量化激活搬运.md">DataCopy</a>（CO1-&gt;GM、CO1-&gt;A1）过程中进行随路格式转换（NZ2ND）时，通过调用该接口设置NZ2ND相关配置。</p>
</td>
</tr>
<tr id="row11129104162017"><td class="cellrowborder" valign="top" headers="mcps1.2.4.1.1 "><p id="p165521930144913"><a name="p165521930144913"></a><a name="p165521930144913"></a><a href="SetFixpipePreQuantFlag.md">SetFixpipePreQuantFlag</a></p>
</td>
<td class="cellrowborder" valign="top" headers="mcps1.2.4.1.2 "><p id="p16332111515234"><a name="p16332111515234"></a><a name="p16332111515234"></a><a href="随路量化激活搬运.md">DataCopy</a>（CO1-&gt;GM、CO1-&gt;A1）过程中进行随路量化时，通过调用该接口设置量化流程中scalar量化参数。</p>
</td>
</tr>
<tr id="row19331114117200"><td class="cellrowborder" valign="top" headers="mcps1.2.4.1.1 "><p id="p278152219274"><a name="p278152219274"></a><a name="p278152219274"></a><a href="SetFixPipeClipRelu.md">SetFixPipeClipRelu</a></p>
</td>
<td class="cellrowborder" valign="top" headers="mcps1.2.4.1.2 "><p id="p10787226279"><a name="p10787226279"></a><a name="p10787226279"></a><a href="随路量化激活搬运.md">DataCopy</a>（CO1-&gt;GM）过程中进行随路量化后，通过调用该接口设置ClipRelu操作的最大值。</p>
</td>
</tr>
<tr id="row0541441152019"><td class="cellrowborder" valign="top" headers="mcps1.2.4.1.1 "><p id="p1674634112715"><a name="p1674634112715"></a><a name="p1674634112715"></a><a href="SetFixPipeAddr.md">SetFixPipeAddr</a></p>
</td>
<td class="cellrowborder" valign="top" headers="mcps1.2.4.1.2 "><p id="p71911024132712"><a name="p71911024132712"></a><a name="p71911024132712"></a><a href="随路量化激活搬运.md">DataCopy</a>（CO1-&gt;GM）过程中进行随路量化后，通过调用该接口设置element-wise操作时LocalTensor的地址。</p>
</td>
</tr>
<tr id="row08411461812"><td class="cellrowborder" valign="top" headers="mcps1.2.4.1.1 "><p id="p1955143017499"><a name="p1955143017499"></a><a name="p1955143017499"></a><a href="Fill.md">Fill</a></p>
</td>
<td class="cellrowborder" valign="top" headers="mcps1.2.4.1.2 "><p id="p1489215634815"><a name="p1489215634815"></a><a name="p1489215634815"></a>初始化LocalTensor（TPosition为A1/A2/B1/B2）为某一个具体的数值。</p>
</td>
</tr>
<tr id="row696114470313"><td class="cellrowborder" valign="top" headers="mcps1.2.4.1.1 "><p id="p455133074917"><a name="p455133074917"></a><a name="p455133074917"></a><a href="LoadData.md">LoadData</a></p>
</td>
<td class="cellrowborder" valign="top" headers="mcps1.2.4.1.2 "><p id="p3448142813492"><a name="p3448142813492"></a><a name="p3448142813492"></a>LoadData包括Load2D和Load3D数据加载功能。</p>
</td>
</tr>
<tr id="row8181746635"><td class="cellrowborder" valign="top" headers="mcps1.2.4.1.1 "><p id="p20551143074919"><a name="p20551143074919"></a><a name="p20551143074919"></a><a href="LoadDataWithTranspose.md">LoadDataWithTranspose</a></p>
</td>
<td class="cellrowborder" valign="top" headers="mcps1.2.4.1.2 "><p id="p124281245667"><a name="p124281245667"></a><a name="p124281245667"></a>该接口实现带转置的2D格式数据从A1/B1到A2/B2的加载。</p>
</td>
</tr>
<tr id="row194791339636"><td class="cellrowborder" valign="top" headers="mcps1.2.4.1.1 "><p id="p12211416165318"><a name="p12211416165318"></a><a name="p12211416165318"></a><a href="SetAippFunctions.md">SetAippFunctions</a></p>
</td>
<td class="cellrowborder" valign="top" headers="mcps1.2.4.1.2 "><p id="p192129164538"><a name="p192129164538"></a><a name="p192129164538"></a>设置图片预处理（AIPP，AI core pre-process）相关参数。</p>
</td>
</tr>
<tr id="row1948463715319"><td class="cellrowborder" valign="top" headers="mcps1.2.4.1.1 "><p id="p1061103015541"><a name="p1061103015541"></a><a name="p1061103015541"></a><a href="LoadImageToLocal.md">LoadImageToLocal</a></p>
</td>
<td class="cellrowborder" valign="top" headers="mcps1.2.4.1.2 "><p id="p792614613548"><a name="p792614613548"></a><a name="p792614613548"></a>将图像数据从GM搬运到A1/B1。 搬运过程中可以完成图像预处理操作：包括图像翻转，改变图像尺寸（抠图，裁边，缩放，伸展），以及色域转换，类型转换等。</p>
</td>
</tr>
<tr id="row0248535339"><td class="cellrowborder" valign="top" headers="mcps1.2.4.1.1 "><p id="p16995181110545"><a name="p16995181110545"></a><a name="p16995181110545"></a><a href="LoadUnzipIndex.md">LoadUnZipIndex</a></p>
</td>
<td class="cellrowborder" valign="top" headers="mcps1.2.4.1.2 "><p id="p20995111113541"><a name="p20995111113541"></a><a name="p20995111113541"></a>加载GM上的压缩索引表到内部寄存器。</p>
</td>
</tr>
<tr id="row132838331033"><td class="cellrowborder" valign="top" headers="mcps1.2.4.1.1 "><p id="p1434981445411"><a name="p1434981445411"></a><a name="p1434981445411"></a><a href="LoadDataUnzip.md">LoadDataUnzip</a></p>
</td>
<td class="cellrowborder" valign="top" headers="mcps1.2.4.1.2 "><p id="p2349121415547"><a name="p2349121415547"></a><a name="p2349121415547"></a>将GM上的数据解压并搬运到A1/B1/B2上。</p>
</td>
</tr>
<tr id="row7838730733"><td class="cellrowborder" valign="top" headers="mcps1.2.4.1.1 "><p id="p1418115368566"><a name="p1418115368566"></a><a name="p1418115368566"></a><a href="LoadDataWithSparse.md">LoadDataWithSparse</a></p>
</td>
<td class="cellrowborder" valign="top" headers="mcps1.2.4.1.2 "><p id="p7263112805617"><a name="p7263112805617"></a><a name="p7263112805617"></a>用于搬运存放在B1里的512B的稠密权重矩阵到B2里，同时读取128B的索引矩阵用于稠密矩阵的稀疏化。</p>
</td>
</tr>
<tr id="row66591928736"><td class="cellrowborder" valign="top" headers="mcps1.2.4.1.1 "><p id="p1552163012493"><a name="p1552163012493"></a><a name="p1552163012493"></a><a href="SetFmatrix.md">SetFmatrix</a></p>
</td>
<td class="cellrowborder" valign="top" headers="mcps1.2.4.1.2 "><p id="p1365632015497"><a name="p1365632015497"></a><a name="p1365632015497"></a>用于调用<a href="LoadData.md">Load3Dv1/Load3Dv2</a>时设置FeatureMap的属性描述。</p>
</td>
</tr>
<tr id="row07692515218"><td class="cellrowborder" valign="top" headers="mcps1.2.4.1.1 "><p id="p055233018498"><a name="p055233018498"></a><a name="p055233018498"></a><a href="SetLoadDataBoundary.md">SetLoadDataBoundary</a></p>
</td>
<td class="cellrowborder" valign="top" headers="mcps1.2.4.1.2 "><p id="p37801058115614"><a name="p37801058115614"></a><a name="p37801058115614"></a>设置<a href="LoadData.md">Load3D</a>时A1/B1边界值。</p>
</td>
</tr>
<tr id="row3787671224"><td class="cellrowborder" valign="top" headers="mcps1.2.4.1.1 "><p id="p1155273074915"><a name="p1155273074915"></a><a name="p1155273074915"></a><a href="SetLoadDataRepeat.md">SetLoadDataRepeat</a></p>
</td>
<td class="cellrowborder" valign="top" headers="mcps1.2.4.1.2 "><p id="p532619442559"><a name="p532619442559"></a><a name="p532619442559"></a>用于设置Load3Dv2接口的repeat参数。设置repeat参数后，可以通过调用一次Load3Dv2接口完成多个迭代的数据搬运。</p>
</td>
</tr>
<tr id="row4662691428"><td class="cellrowborder" valign="top" headers="mcps1.2.4.1.1 "><p id="p18552183094913"><a name="p18552183094913"></a><a name="p18552183094913"></a><a href="SetLoadDataPaddingValue.md">SetLoadDataPaddingValue</a></p>
</td>
<td class="cellrowborder" valign="top" headers="mcps1.2.4.1.2 "><p id="p53183549554"><a name="p53183549554"></a><a name="p53183549554"></a>设置padValue，用于Load3Dv1/Load3Dv2。</p>
</td>
</tr>
<tr id="row165019119218"><td class="cellrowborder" valign="top" headers="mcps1.2.4.1.1 "><p id="p12552123034913"><a name="p12552123034913"></a><a name="p12552123034913"></a><a href="Fixpipe.md">Fixpipe</a></p>
</td>
<td class="cellrowborder" valign="top" headers="mcps1.2.4.1.2 "><p id="p7194151162615"><a name="p7194151162615"></a><a name="p7194151162615"></a>矩阵计算完成后，对结果进行处理，例如对计算结果进行量化操作，并把数据从CO1搬迁到Global Memory中。</p>
</td>
</tr>
<tr id="row188711975503"><td class="cellrowborder" rowspan="5" valign="top" width="12.379999999999999%" headers="mcps1.2.4.1.1 "><p id="p13495131445016"><a name="p13495131445016"></a><a name="p13495131445016"></a>同步控制</p>
</td>
<td class="cellrowborder" valign="top" width="27.63%" headers="mcps1.2.4.1.2 "><p id="p14495181465015"><a name="p14495181465015"></a><a name="p14495181465015"></a><a href="SetFlag-WaitFlag(ISASI).md">SetFlag/WaitFlag</a></p>
</td>
<td class="cellrowborder" valign="top" width="59.99%" headers="mcps1.2.4.1.3 "><p id="p158718715503"><a name="p158718715503"></a><a name="p158718715503"></a>同一核内不同流水线之间的同步指令。具有数据依赖的不同流水指令之间需要插此同步。</p>
</td>
</tr>
<tr id="row181116835017"><td class="cellrowborder" valign="top" headers="mcps1.2.4.1.1 "><p id="p16495161412505"><a name="p16495161412505"></a><a name="p16495161412505"></a><a href="PipeBarrier(ISASI).md">PipeBarrier</a></p>
</td>
<td class="cellrowborder" valign="top" headers="mcps1.2.4.1.2 "><p id="p21128145011"><a name="p21128145011"></a><a name="p21128145011"></a>阻塞相同流水，具有数据依赖的相同流水之间需要插此同步。</p>
</td>
</tr>
<tr id="row653814222592"><td class="cellrowborder" valign="top" headers="mcps1.2.4.1.1 "><p id="p1289326165918"><a name="p1289326165918"></a><a name="p1289326165918"></a><a href="DataSyncBarrier(ISASI).md">DataSyncBarrier</a></p>
</td>
<td class="cellrowborder" valign="top" headers="mcps1.2.4.1.2 "><p id="p8600163212594"><a name="p8600163212594"></a><a name="p8600163212594"></a>用于阻塞后续的指令执行，直到所有之前的内存访问指令（需要等待的内存位置可通过参数控制）执行结束。</p>
</td>
</tr>
<tr id="row114551284506"><td class="cellrowborder" valign="top" headers="mcps1.2.4.1.1 "><p id="p5495141475017"><a name="p5495141475017"></a><a name="p5495141475017"></a><a href="CrossCoreSetFlag(ISASI).md">CrossCoreSetFlag</a></p>
</td>
<td class="cellrowborder" valign="top" headers="mcps1.2.4.1.2 "><p id="p20555155242412"><a name="p20555155242412"></a><a name="p20555155242412"></a>针对分离模式，AI Core上的Cube核（AIC）与Vector核（AIV）之间的同步设置指令。</p>
</td>
</tr>
<tr id="row1169128105014"><td class="cellrowborder" valign="top" headers="mcps1.2.4.1.1 "><p id="p1495141410509"><a name="p1495141410509"></a><a name="p1495141410509"></a><a href="CrossCoreWaitFlag(ISASI).md">CrossCoreWaitFlag</a></p>
</td>
<td class="cellrowborder" valign="top" headers="mcps1.2.4.1.2 "><p id="p1453712312015"><a name="p1453712312015"></a><a name="p1453712312015"></a>针对分离模式，AI Core上的Cube核（AIC）与Vector核（AIV）之间的同步等待指令。</p>
</td>
</tr>
<tr id="row13905175871"><td class="cellrowborder" rowspan="3" valign="top" width="12.379999999999999%" headers="mcps1.2.4.1.1 "><p id="p123688281178"><a name="p123688281178"></a><a name="p123688281178"></a>同步控制</p>
</td>
<td class="cellrowborder" valign="top" width="27.63%" headers="mcps1.2.4.1.2 "><p id="p99126441977"><a name="p99126441977"></a><a name="p99126441977"></a><a href="Mutex（ISASI）.md">Mutex</a></p>
</td>
<td class="cellrowborder" valign="top" width="59.99%" headers="mcps1.2.4.1.3 "><p id="p1190519517715"><a name="p1190519517715"></a><a name="p1190519517715"></a><span id="ph1520973121010"><a name="ph1520973121010"></a><a name="ph1520973121010"></a>Mutex用于核内异步流水指令之间的同步处理，其功能类似于传统CPU中的锁机制。通过锁定指定流水再释放流水来完成流水间的同步依赖。每个锁有固定的一个MutexID，该ID可通过用户自定义（范围为0-27）或者通过<a href="AllocMutexID-(ISASI).md">AllocMutexID/ReleaseMutexID</a>进行申请释放。</span></p>
</td>
</tr>
<tr id="row7712137571"><td class="cellrowborder" valign="top" headers="mcps1.2.4.1.1 "><p id="p532817527716"><a name="p532817527716"></a><a name="p532817527716"></a><a href="AllocMutexID-(ISASI).md">AllocMutexID</a></p>
</td>
<td class="cellrowborder" valign="top" headers="mcps1.2.4.1.2 "><p id="p147126719714"><a name="p147126719714"></a><a name="p147126719714"></a><span>从框架获取并占用一个</span>MutexID<span>，与</span><a href="ReleaseMutexID-(ISASI).md">ReleaseMutexID</a><span>配合使用，</span><span>管理MutexID的获取和释放。</span></p>
</td>
</tr>
<tr id="row5734121020712"><td class="cellrowborder" valign="top" headers="mcps1.2.4.1.1 "><p id="p1989145819720"><a name="p1989145819720"></a><a name="p1989145819720"></a><a href="ReleaseMutexID-(ISASI).md">ReleaseMutexID</a></p>
</td>
<td class="cellrowborder" valign="top" headers="mcps1.2.4.1.2 "><p id="p1973571020715"><a name="p1973571020715"></a><a name="p1973571020715"></a><span id="ph10161191691011"><a name="ph10161191691011"></a><a name="ph10161191691011"></a>从框架释放一个MutexID，与<a href="AllocMutexID-(ISASI).md">AllocMutexID</a>配合使用<span>。</span></span></p>
</td>
</tr>
<tr id="row329139125011"><td class="cellrowborder" rowspan="2" valign="top" width="12.379999999999999%" headers="mcps1.2.4.1.1 "><p id="p164958141507"><a name="p164958141507"></a><a name="p164958141507"></a>缓存处理</p>
</td>
<td class="cellrowborder" valign="top" width="27.63%" headers="mcps1.2.4.1.2 "><p id="p124961914135016"><a name="p124961914135016"></a><a name="p124961914135016"></a><a href="ICachePreLoad(ISASI).md">ICachePreLoad</a></p>
</td>
<td class="cellrowborder" valign="top" width="59.99%" headers="mcps1.2.4.1.3 "><p id="p16327940182218"><a name="p16327940182218"></a><a name="p16327940182218"></a>从指令所在DDR地址预加载指令到ICache中。</p>
</td>
</tr>
<tr id="row19196799505"><td class="cellrowborder" valign="top" headers="mcps1.2.4.1.1 "><p id="p4496131419504"><a name="p4496131419504"></a><a name="p4496131419504"></a><a href="GetICachePreloadStatus(ISASI).md">GetICachePreloadStatus</a></p>
</td>
<td class="cellrowborder" valign="top" headers="mcps1.2.4.1.2 "><p id="p319669165012"><a name="p319669165012"></a><a name="p319669165012"></a>获取ICACHE的PreLoad的状态。</p>
</td>
</tr>
<tr id="row2493696509"><td class="cellrowborder" rowspan="4" valign="top" width="12.379999999999999%" headers="mcps1.2.4.1.1 "><p id="p649661485013"><a name="p649661485013"></a><a name="p649661485013"></a>系统变量访问</p>
</td>
<td class="cellrowborder" valign="top" width="27.63%" headers="mcps1.2.4.1.2 "><p id="p1649651485012"><a name="p1649651485012"></a><a name="p1649651485012"></a><a href="GetProgramCounter(ISASI).md">GetProgramCounter</a></p>
</td>
<td class="cellrowborder" valign="top" width="59.99%" headers="mcps1.2.4.1.3 "><p id="p1349315905014"><a name="p1349315905014"></a><a name="p1349315905014"></a>获取程序计数器的指针，程序计数器用于记录当前程序执行的位置。</p>
</td>
</tr>
<tr id="row19631896502"><td class="cellrowborder" valign="top" headers="mcps1.2.4.1.1 "><p id="p74961814175010"><a name="p74961814175010"></a><a name="p74961814175010"></a><a href="GetSubBlockNum(ISASI).md">GetSubBlockNum</a></p>
</td>
<td class="cellrowborder" valign="top" headers="mcps1.2.4.1.2 "><p id="p966612452211"><a name="p966612452211"></a><a name="p966612452211"></a>获取AI Core上Vector核的数量。</p>
</td>
</tr>
<tr id="row278314916501"><td class="cellrowborder" valign="top" headers="mcps1.2.4.1.1 "><p id="p2496314115019"><a name="p2496314115019"></a><a name="p2496314115019"></a><a href="GetSubBlockIdx(ISASI).md">GetSubBlockIdx</a></p>
</td>
<td class="cellrowborder" valign="top" headers="mcps1.2.4.1.2 "><p id="p7322151817114"><a name="p7322151817114"></a><a name="p7322151817114"></a>获取AI Core上Vector核的ID。</p>
</td>
</tr>
<tr id="row591914913509"><td class="cellrowborder" valign="top" headers="mcps1.2.4.1.1 "><p id="p449621416508"><a name="p449621416508"></a><a name="p449621416508"></a><a href="GetSystemCycle(ISASI).md">GetSystemCycle</a></p>
</td>
<td class="cellrowborder" valign="top" headers="mcps1.2.4.1.2 "><p id="p318519518103"><a name="p318519518103"></a><a name="p318519518103"></a>获取当前系统cycle数，若换算成时间需要按照50MHz的频率，时间单位为us，换算公式为：time = (cycle数/50) us 。</p>
</td>
</tr>
<tr id="row23644391720"><td class="cellrowborder" rowspan="3" valign="top" width="12.379999999999999%" headers="mcps1.2.4.1.1 "><p id="p13856134291819"><a name="p13856134291819"></a><a name="p13856134291819"></a>系统变量访问</p>
</td>
<td class="cellrowborder" valign="top" width="27.63%" headers="mcps1.2.4.1.2 "><p id="p236104361711"><a name="p236104361711"></a><a name="p236104361711"></a><a href="SetCtrlSpr(ISASI).md">SetCtrlSpr</a></p>
</td>
<td class="cellrowborder" valign="top" width="59.99%" headers="mcps1.2.4.1.3 "><p id="p123610436177"><a name="p123610436177"></a><a name="p123610436177"></a><span id="ph1641093112018"><a name="ph1641093112018"></a><a name="ph1641093112018"></a>对CTRL寄存器（控制寄存器）的特定比特位进行设置。</span></p>
</td>
</tr>
<tr id="row191522045131718"><td class="cellrowborder" valign="top" headers="mcps1.2.4.1.1 "><p id="p186893131181"><a name="p186893131181"></a><a name="p186893131181"></a><a href="GetCtrlSpr(ISASI).md">GetCtrlSpr</a></p>
</td>
<td class="cellrowborder" valign="top" headers="mcps1.2.4.1.2 "><p id="p17152114561719"><a name="p17152114561719"></a><a name="p17152114561719"></a><span id="ph1892618377202"><a name="ph1892618377202"></a><a name="ph1892618377202"></a>读取CTRL寄存器（控制寄存器）特定比特位上的值。</span></p>
</td>
</tr>
<tr id="row28763469172"><td class="cellrowborder" valign="top" headers="mcps1.2.4.1.1 "><p id="p876017348182"><a name="p876017348182"></a><a name="p876017348182"></a><a href="ResetCtrlSpr(ISASI).md">ResetCtrlSpr</a></p>
</td>
<td class="cellrowborder" valign="top" headers="mcps1.2.4.1.2 "><p id="p5876194619170"><a name="p5876194619170"></a><a name="p5876194619170"></a><span id="ph818454202"><a name="ph818454202"></a><a name="ph818454202"></a>对CTRL寄存器（控制寄存器）的特定比特位做重置。</span></p>
</td>
</tr>
<tr id="row20229161014506"><td class="cellrowborder" rowspan="4" valign="top" width="12.379999999999999%" headers="mcps1.2.4.1.1 "><p id="p649691414505"><a name="p649691414505"></a><a name="p649691414505"></a>原子操作</p>
</td>
<td class="cellrowborder" valign="top" width="27.63%" headers="mcps1.2.4.1.2 "><p id="p114961614195013"><a name="p114961614195013"></a><a name="p114961614195013"></a><a href="SetAtomicMax(ISASI).md">SetAtomicMax</a></p>
</td>
<td class="cellrowborder" valign="top" width="59.99%" headers="mcps1.2.4.1.3 "><p id="p1555107145615"><a name="p1555107145615"></a><a name="p1555107145615"></a>原子操作函数，设置后续从VECOUT传输到GM的数据是否执行原子比较，将待拷贝的内容和GM已有内容进行比较，将最大值写入GM。</p>
</td>
</tr>
<tr id="row237920103503"><td class="cellrowborder" valign="top" headers="mcps1.2.4.1.1 "><p id="p18496141413508"><a name="p18496141413508"></a><a name="p18496141413508"></a><a href="SetAtomicMin(ISASI).md">SetAtomicMin</a></p>
</td>
<td class="cellrowborder" valign="top" headers="mcps1.2.4.1.2 "><p id="p15379191018502"><a name="p15379191018502"></a><a name="p15379191018502"></a>原子操作函数，设置后续从VECOUT传输到GM的数据是否执行原子比较，将待拷贝的内容和GM已有内容进行比较，将最小值写入GM。</p>
</td>
</tr>
<tr id="row5515201016501"><td class="cellrowborder" valign="top" headers="mcps1.2.4.1.1 "><p id="p4496814165013"><a name="p4496814165013"></a><a name="p4496814165013"></a><a href="SetStoreAtomicConfig(ISASI).md">SetStoreAtomicConfig</a></p>
</td>
<td class="cellrowborder" valign="top" headers="mcps1.2.4.1.2 "><p id="p95151510125017"><a name="p95151510125017"></a><a name="p95151510125017"></a>设置原子操作使能位与原子操作类型。</p>
</td>
</tr>
<tr id="row466110104502"><td class="cellrowborder" valign="top" headers="mcps1.2.4.1.1 "><p id="p949614146509"><a name="p949614146509"></a><a name="p949614146509"></a><a href="GetStoreAtomicConfig(ISASI).md">GetStoreAtomicConfig</a></p>
</td>
<td class="cellrowborder" valign="top" headers="mcps1.2.4.1.2 "><p id="p4576142854016"><a name="p4576142854016"></a><a name="p4576142854016"></a>获取原子操作使能位与原子操作类型的值。</p>
</td>
</tr>
<tr id="row161503191126"><td class="cellrowborder" valign="top" width="12.379999999999999%" headers="mcps1.2.4.1.1 "><p id="p1715151915126"><a name="p1715151915126"></a><a name="p1715151915126"></a>调试接口</p>
</td>
<td class="cellrowborder" valign="top" width="27.63%" headers="mcps1.2.4.1.2 "><p id="p9552163084912"><a name="p9552163084912"></a><a name="p9552163084912"></a><a href="CheckLocalMemoryIA(ISASI).md">CheckLocalMemoryIA</a></p>
</td>
<td class="cellrowborder" valign="top" width="59.99%" headers="mcps1.2.4.1.3 "><p id="p5863853185711"><a name="p5863853185711"></a><a name="p5863853185711"></a>监视设定范围内的UB读写行为，如果监视到有设定范围的读写行为则会出现EXCEPTION报错，未监视到设定范围的读写行为则不会报错。</p>
</td>
</tr>
<tr id="row18446920142913"><td class="cellrowborder" rowspan="3" valign="top" width="12.379999999999999%" headers="mcps1.2.4.1.1 "><p id="p164464202295"><a name="p164464202295"></a><a name="p164464202295"></a>Cube分组管理</p>
</td>
<td class="cellrowborder" valign="top" width="27.63%" headers="mcps1.2.4.1.2 "><p id="p7499133216292"><a name="p7499133216292"></a><a name="p7499133216292"></a><a href="CubeResGroupHandle.md">CubeResGroupHandle</a></p>
</td>
<td class="cellrowborder" valign="top" width="59.99%" headers="mcps1.2.4.1.3 "><p id="p24466202298"><a name="p24466202298"></a><a name="p24466202298"></a>CubeResGroupHandle用于在分离模式下通过软同步控制AIC和AIV之间进行通讯，实现<span id="zh-cn_topic_0000001588832845_ph168139148536"><a name="zh-cn_topic_0000001588832845_ph168139148536"></a><a name="zh-cn_topic_0000001588832845_ph168139148536"></a>AI Core</span>计算资源分组。</p>
</td>
</tr>
<tr id="row1493333852918"><td class="cellrowborder" valign="top" headers="mcps1.2.4.1.1 "><p id="p8398498294"><a name="p8398498294"></a><a name="p8398498294"></a><a href="GroupBarrier.md">GroupBarrier</a></p>
</td>
<td class="cellrowborder" valign="top" headers="mcps1.2.4.1.2 "><p id="p1493314380295"><a name="p1493314380295"></a><a name="p1493314380295"></a>当同一个<a href="CubeResGroupHandle.md">CubeResGroupHandle</a>中的两个AIV任务之间存在依赖关系时，可以使用GroupBarrier控制同步。</p>
</td>
</tr>
<tr id="row1786094112299"><td class="cellrowborder" valign="top" headers="mcps1.2.4.1.1 "><p id="p186215598290"><a name="p186215598290"></a><a name="p186215598290"></a><a href="KfcWorkspace.md">KfcWorkspace</a></p>
</td>
<td class="cellrowborder" valign="top" headers="mcps1.2.4.1.2 "><p id="p1776114192019"><a name="p1776114192019"></a><a name="p1776114192019"></a>KfcWorkspace为通信空间描述符，管理不同<a href="CubeResGroupHandle.md">CubeResGroupHandle</a>的消息通信区划分，与CubeResGroupHandle配合使用。KfcWorkspace的构造函数用于创建KfcWorkspace对象。</p>
</td>
</tr>
</tbody>
</table>

## 高阶API<a name="section3317105813235"></a>

**表 14**  数学计算API列表

<a name="table6328746161212"></a>
<table><thead align="left"><tr id="row18328114610121"><th class="cellrowborder" valign="top" width="37.71%" id="mcps1.2.3.1.1"><p id="p173281846121219"><a name="p173281846121219"></a><a name="p173281846121219"></a>接口名</p>
</th>
<th class="cellrowborder" valign="top" width="62.29%" id="mcps1.2.3.1.2"><p id="p232844620126"><a name="p232844620126"></a><a name="p232844620126"></a>功能描述</p>
</th>
</tr>
</thead>
<tbody><tr id="row7328204651217"><td class="cellrowborder" valign="top" width="37.71%" headers="mcps1.2.3.1.1 "><p id="p2032844612121"><a name="p2032844612121"></a><a name="p2032844612121"></a><a href="Acos.md">Acos</a></p>
</td>
<td class="cellrowborder" valign="top" width="62.29%" headers="mcps1.2.3.1.2 "><p id="p17328164651218"><a name="p17328164651218"></a><a name="p17328164651218"></a>按元素做反余弦函数计算。</p>
</td>
</tr>
<tr id="row19328124671211"><td class="cellrowborder" valign="top" width="37.71%" headers="mcps1.2.3.1.1 "><p id="p17328204661218"><a name="p17328204661218"></a><a name="p17328204661218"></a><a href="Acosh.md">Acosh</a></p>
</td>
<td class="cellrowborder" valign="top" width="62.29%" headers="mcps1.2.3.1.2 "><p id="p83298462120"><a name="p83298462120"></a><a name="p83298462120"></a>按元素做双曲反余弦函数计算。</p>
</td>
</tr>
<tr id="row11329946171211"><td class="cellrowborder" valign="top" width="37.71%" headers="mcps1.2.3.1.1 "><p id="p63297464128"><a name="p63297464128"></a><a name="p63297464128"></a><a href="Asin.md">Asin</a></p>
</td>
<td class="cellrowborder" valign="top" width="62.29%" headers="mcps1.2.3.1.2 "><p id="p20329114641219"><a name="p20329114641219"></a><a name="p20329114641219"></a>按元素做反正弦函数计算。</p>
</td>
</tr>
<tr id="row1632954617129"><td class="cellrowborder" valign="top" width="37.71%" headers="mcps1.2.3.1.1 "><p id="p3329546101216"><a name="p3329546101216"></a><a name="p3329546101216"></a><a href="Asinh.md">Asinh</a></p>
</td>
<td class="cellrowborder" valign="top" width="62.29%" headers="mcps1.2.3.1.2 "><p id="p932994611216"><a name="p932994611216"></a><a name="p932994611216"></a>按元素做反双曲正弦函数计算。</p>
</td>
</tr>
<tr id="row1329646131216"><td class="cellrowborder" valign="top" width="37.71%" headers="mcps1.2.3.1.1 "><p id="p10329184619129"><a name="p10329184619129"></a><a name="p10329184619129"></a><a href="Atan.md">Atan</a></p>
</td>
<td class="cellrowborder" valign="top" width="62.29%" headers="mcps1.2.3.1.2 "><p id="p232904617122"><a name="p232904617122"></a><a name="p232904617122"></a>按元素做三角函数反正切运算。</p>
</td>
</tr>
<tr id="row1932944621219"><td class="cellrowborder" valign="top" width="37.71%" headers="mcps1.2.3.1.1 "><p id="p1032944641214"><a name="p1032944641214"></a><a name="p1032944641214"></a><a href="Atanh.md">Atanh</a></p>
</td>
<td class="cellrowborder" valign="top" width="62.29%" headers="mcps1.2.3.1.2 "><p id="p1532914614126"><a name="p1532914614126"></a><a name="p1532914614126"></a>按元素做反双曲正切余弦函数计算。</p>
</td>
</tr>
<tr id="row18329194631210"><td class="cellrowborder" valign="top" width="37.71%" headers="mcps1.2.3.1.1 "><p id="p1032984631216"><a name="p1032984631216"></a><a name="p1032984631216"></a><a href="Axpy-101.md">Axpy</a></p>
</td>
<td class="cellrowborder" valign="top" width="62.29%" headers="mcps1.2.3.1.2 "><p id="p8329114612121"><a name="p8329114612121"></a><a name="p8329114612121"></a>源操作数中每个元素与标量求积后和目的操作数中的对应元素相加。</p>
</td>
</tr>
<tr id="row732916466122"><td class="cellrowborder" valign="top" width="37.71%" headers="mcps1.2.3.1.1 "><p id="p1732974619121"><a name="p1732974619121"></a><a name="p1732974619121"></a><a href="Ceil.md">Ceil</a></p>
</td>
<td class="cellrowborder" valign="top" width="62.29%" headers="mcps1.2.3.1.2 "><p id="p1432924610129"><a name="p1432924610129"></a><a name="p1432924610129"></a>获取大于或等于x的最小的整数值，即向正无穷取整操作。</p>
</td>
</tr>
<tr id="row232994671216"><td class="cellrowborder" valign="top" width="37.71%" headers="mcps1.2.3.1.1 "><p id="p19329124613120"><a name="p19329124613120"></a><a name="p19329124613120"></a><a href="ClampMax.md">ClampMax</a></p>
</td>
<td class="cellrowborder" valign="top" width="62.29%" headers="mcps1.2.3.1.2 "><p id="p14329104681213"><a name="p14329104681213"></a><a name="p14329104681213"></a>将srcTensor中大于scalar的数替换为scalar，小于等于scalar的数保持不变，作为dstTensor输出。</p>
</td>
</tr>
<tr id="row53290468122"><td class="cellrowborder" valign="top" width="37.71%" headers="mcps1.2.3.1.1 "><p id="p193291646111212"><a name="p193291646111212"></a><a name="p193291646111212"></a><a href="ClampMin.md">ClampMin</a></p>
</td>
<td class="cellrowborder" valign="top" width="62.29%" headers="mcps1.2.3.1.2 "><p id="p1132984620124"><a name="p1132984620124"></a><a name="p1132984620124"></a>将srcTensor中小于scalar的数替换为scalar，大于等于scalar的数保持不变，作为dstTensor输出。</p>
</td>
</tr>
<tr id="row12329154631216"><td class="cellrowborder" valign="top" width="37.71%" headers="mcps1.2.3.1.1 "><p id="p1329174651218"><a name="p1329174651218"></a><a name="p1329174651218"></a><a href="Cos.md">Cos</a></p>
</td>
<td class="cellrowborder" valign="top" width="62.29%" headers="mcps1.2.3.1.2 "><p id="p13291946101211"><a name="p13291946101211"></a><a name="p13291946101211"></a>按元素做三角函数余弦运算。</p>
</td>
</tr>
<tr id="row1232914611214"><td class="cellrowborder" valign="top" width="37.71%" headers="mcps1.2.3.1.1 "><p id="p53307467120"><a name="p53307467120"></a><a name="p53307467120"></a><a href="Cosh.md">Cosh</a></p>
</td>
<td class="cellrowborder" valign="top" width="62.29%" headers="mcps1.2.3.1.2 "><p id="p433011461121"><a name="p433011461121"></a><a name="p433011461121"></a>按元素做双曲余弦函数计算。</p>
</td>
</tr>
<tr id="row4330174681219"><td class="cellrowborder" valign="top" width="37.71%" headers="mcps1.2.3.1.1 "><p id="p5330194616123"><a name="p5330194616123"></a><a name="p5330194616123"></a><a href="CumSum.md">CumSum</a></p>
</td>
<td class="cellrowborder" valign="top" width="62.29%" headers="mcps1.2.3.1.2 "><p id="p133303469127"><a name="p133303469127"></a><a name="p133303469127"></a>对数据按行依次累加或按列依次累加。</p>
</td>
</tr>
<tr id="row193305461120"><td class="cellrowborder" valign="top" width="37.71%" headers="mcps1.2.3.1.1 "><p id="p1433044616123"><a name="p1433044616123"></a><a name="p1433044616123"></a><a href="Digamma.md">Digamma</a></p>
</td>
<td class="cellrowborder" valign="top" width="62.29%" headers="mcps1.2.3.1.2 "><p id="p333019462124"><a name="p333019462124"></a><a name="p333019462124"></a>按元素计算x的gamma函数的对数导数。</p>
</td>
</tr>
<tr id="row16330546131210"><td class="cellrowborder" valign="top" width="37.71%" headers="mcps1.2.3.1.1 "><p id="p1233020469127"><a name="p1233020469127"></a><a name="p1233020469127"></a><a href="Erf.md">Erf</a></p>
</td>
<td class="cellrowborder" valign="top" width="62.29%" headers="mcps1.2.3.1.2 "><p id="p1933044631210"><a name="p1933044631210"></a><a name="p1933044631210"></a>按元素做误差函数计算，也称为高斯误差函数。</p>
</td>
</tr>
<tr id="row1033010465120"><td class="cellrowborder" valign="top" width="37.71%" headers="mcps1.2.3.1.1 "><p id="p11330194610124"><a name="p11330194610124"></a><a name="p11330194610124"></a><a href="Erfc.md">Erfc</a></p>
</td>
<td class="cellrowborder" valign="top" width="62.29%" headers="mcps1.2.3.1.2 "><p id="p13330204601215"><a name="p13330204601215"></a><a name="p13330204601215"></a>返回输入x的互补误差函数结果，积分区间为x到无穷大。</p>
</td>
</tr>
<tr id="row13330144611210"><td class="cellrowborder" valign="top" width="37.71%" headers="mcps1.2.3.1.1 "><p id="p43301046151212"><a name="p43301046151212"></a><a name="p43301046151212"></a><a href="Exp-102.md">Exp</a></p>
</td>
<td class="cellrowborder" valign="top" width="62.29%" headers="mcps1.2.3.1.2 "><p id="p6330246201220"><a name="p6330246201220"></a><a name="p6330246201220"></a>按元素取自然指数。</p>
</td>
</tr>
<tr id="row16330174651218"><td class="cellrowborder" valign="top" width="37.71%" headers="mcps1.2.3.1.1 "><p id="p6330154610126"><a name="p6330154610126"></a><a name="p6330154610126"></a><a href="Floor.md">Floor</a></p>
</td>
<td class="cellrowborder" valign="top" width="62.29%" headers="mcps1.2.3.1.2 "><p id="p1033024691219"><a name="p1033024691219"></a><a name="p1033024691219"></a>获取小于或等于x的最小的整数值，即向负无穷取整操作。</p>
</td>
</tr>
<tr id="row17640143823614"><td class="cellrowborder" valign="top" width="37.71%" headers="mcps1.2.3.1.1 "><p id="p10640163883615"><a name="p10640163883615"></a><a name="p10640163883615"></a><a href="Fmod.md">Fmod</a></p>
</td>
<td class="cellrowborder" valign="top" width="62.29%" headers="mcps1.2.3.1.2 "><p id="p136406388366"><a name="p136406388366"></a><a name="p136406388366"></a>按元素计算两个浮点数相除后的余数。</p>
</td>
</tr>
<tr id="row173301946181212"><td class="cellrowborder" valign="top" width="37.71%" headers="mcps1.2.3.1.1 "><p id="p333014681217"><a name="p333014681217"></a><a name="p333014681217"></a><a href="Frac.md">Frac</a></p>
</td>
<td class="cellrowborder" valign="top" width="62.29%" headers="mcps1.2.3.1.2 "><p id="p1033044671220"><a name="p1033044671220"></a><a name="p1033044671220"></a>按元素做取小数计算。</p>
</td>
</tr>
<tr id="row186161953141515"><td class="cellrowborder" valign="top" width="37.71%" headers="mcps1.2.3.1.1 "><p id="p2712596154"><a name="p2712596154"></a><a name="p2712596154"></a><a href="Hypot.md">Hypot</a></p>
</td>
<td class="cellrowborder" valign="top" width="62.29%" headers="mcps1.2.3.1.2 "><p id="p071135951519"><a name="p071135951519"></a><a name="p071135951519"></a>按元素计算两个浮点数平方和的平方根。</p>
</td>
</tr>
<tr id="row12981425163519"><td class="cellrowborder" valign="top" width="37.71%" headers="mcps1.2.3.1.1 "><p id="p19981202553511"><a name="p19981202553511"></a><a name="p19981202553511"></a><a href="IsFinite.md">IsFinite</a></p>
</td>
<td class="cellrowborder" valign="top" width="62.29%" headers="mcps1.2.3.1.2 "><p id="p11981225203513"><a name="p11981225203513"></a><a name="p11981225203513"></a><span>按元素判断输入的浮点数是否</span>非NAN、非&plusmn;INF。</p>
</td>
</tr>
<tr id="row83303464127"><td class="cellrowborder" valign="top" width="37.71%" headers="mcps1.2.3.1.1 "><p id="p8330144611210"><a name="p8330144611210"></a><a name="p8330144611210"></a><a href="Lgamma.md">Lgamma</a></p>
</td>
<td class="cellrowborder" valign="top" width="62.29%" headers="mcps1.2.3.1.2 "><p id="p1133044651217"><a name="p1133044651217"></a><a name="p1133044651217"></a>按元素计算x的gamma函数的绝对值并求自然对数。</p>
</td>
</tr>
<tr id="row033124619126"><td class="cellrowborder" valign="top" width="37.71%" headers="mcps1.2.3.1.1 "><p id="p4331114613128"><a name="p4331114613128"></a><a name="p4331114613128"></a><a href="Log-100.md">Log</a></p>
</td>
<td class="cellrowborder" valign="top" width="62.29%" headers="mcps1.2.3.1.2 "><p id="p5331204620122"><a name="p5331204620122"></a><a name="p5331204620122"></a>按元素以e、2、10为底做对数运算。</p>
</td>
</tr>
<tr id="row7331194661216"><td class="cellrowborder" valign="top" width="37.71%" headers="mcps1.2.3.1.1 "><p id="p7331114617127"><a name="p7331114617127"></a><a name="p7331114617127"></a><a href="Power.md">Power</a></p>
</td>
<td class="cellrowborder" valign="top" width="62.29%" headers="mcps1.2.3.1.2 "><p id="p183314461122"><a name="p183314461122"></a><a name="p183314461122"></a>实现按元素做幂运算功能。</p>
</td>
</tr>
<tr id="row8331144631211"><td class="cellrowborder" valign="top" width="37.71%" headers="mcps1.2.3.1.1 "><p id="p20331146121218"><a name="p20331146121218"></a><a name="p20331146121218"></a><a href="Round.md">Round</a></p>
</td>
<td class="cellrowborder" valign="top" width="62.29%" headers="mcps1.2.3.1.2 "><p id="p11331946151214"><a name="p11331946151214"></a><a name="p11331946151214"></a>将输入的元素四舍五入到最接近的整数。</p>
</td>
</tr>
<tr id="row0331174621213"><td class="cellrowborder" valign="top" width="37.71%" headers="mcps1.2.3.1.1 "><p id="p233194613123"><a name="p233194613123"></a><a name="p233194613123"></a><a href="Sign.md">Sign</a></p>
</td>
<td class="cellrowborder" valign="top" width="62.29%" headers="mcps1.2.3.1.2 "><p id="p433110466122"><a name="p433110466122"></a><a name="p433110466122"></a>按元素执行Sign操作，Sign是指返回输入数据的符号。</p>
</td>
</tr>
<tr id="row13310463124"><td class="cellrowborder" valign="top" width="37.71%" headers="mcps1.2.3.1.1 "><p id="p103311346121214"><a name="p103311346121214"></a><a name="p103311346121214"></a><a href="Sin.md">Sin</a></p>
</td>
<td class="cellrowborder" valign="top" width="62.29%" headers="mcps1.2.3.1.2 "><p id="p1133110462120"><a name="p1133110462120"></a><a name="p1133110462120"></a>按元素做正弦函数计算。</p>
</td>
</tr>
<tr id="row10331124610124"><td class="cellrowborder" valign="top" width="37.71%" headers="mcps1.2.3.1.1 "><p id="p5331194615122"><a name="p5331194615122"></a><a name="p5331194615122"></a><a href="Sinh.md">Sinh</a></p>
</td>
<td class="cellrowborder" valign="top" width="62.29%" headers="mcps1.2.3.1.2 "><p id="p2331154618126"><a name="p2331154618126"></a><a name="p2331154618126"></a>按元素做双曲正弦函数计算。</p>
</td>
</tr>
<tr id="row9331546131219"><td class="cellrowborder" valign="top" width="37.71%" headers="mcps1.2.3.1.1 "><p id="p733118468123"><a name="p733118468123"></a><a name="p733118468123"></a><a href="Tan.md">Tan</a></p>
</td>
<td class="cellrowborder" valign="top" width="62.29%" headers="mcps1.2.3.1.2 "><p id="p103311246151214"><a name="p103311246151214"></a><a name="p103311246151214"></a>按元素做正切函数计算。</p>
</td>
</tr>
<tr id="row1533124619123"><td class="cellrowborder" valign="top" width="37.71%" headers="mcps1.2.3.1.1 "><p id="p833114651211"><a name="p833114651211"></a><a name="p833114651211"></a><a href="Tanh.md">Tanh</a></p>
</td>
<td class="cellrowborder" valign="top" width="62.29%" headers="mcps1.2.3.1.2 "><p id="p033111464127"><a name="p033111464127"></a><a name="p033111464127"></a>按元素做逻辑回归Tanh。</p>
</td>
</tr>
<tr id="row7331164681211"><td class="cellrowborder" valign="top" width="37.71%" headers="mcps1.2.3.1.1 "><p id="p203311346191210"><a name="p203311346191210"></a><a name="p203311346191210"></a><a href="Trunc.md">Trunc</a></p>
</td>
<td class="cellrowborder" valign="top" width="62.29%" headers="mcps1.2.3.1.2 "><p id="p20331144651218"><a name="p20331144651218"></a><a name="p20331144651218"></a>按元素做浮点数截断操作，即向零取整操作。</p>
</td>
</tr>
<tr id="row123329469123"><td class="cellrowborder" valign="top" width="37.71%" headers="mcps1.2.3.1.1 "><p id="p103321046181214"><a name="p103321046181214"></a><a name="p103321046181214"></a><a href="Xor-103.md">Xor</a></p>
</td>
<td class="cellrowborder" valign="top" width="62.29%" headers="mcps1.2.3.1.2 "><p id="p533224611126"><a name="p533224611126"></a><a name="p533224611126"></a>按元素执行Xor（异或）运算。</p>
</td>
</tr>
<tr id="row6226126112911"><td class="cellrowborder" valign="top" width="37.71%" headers="mcps1.2.3.1.1 "><p id="p622617619296"><a name="p622617619296"></a><a name="p622617619296"></a><a href="Fma接口.md">Fma</a></p>
</td>
<td class="cellrowborder" valign="top" width="62.29%" headers="mcps1.2.3.1.2 "><p id="p722620642915"><a name="p722620642915"></a><a name="p722620642915"></a>按元素计算两个输入相乘后与第三个输入相加的结果。</p>
</td>
</tr>
<tr id="row980410892919"><td class="cellrowborder" valign="top" width="37.71%" headers="mcps1.2.3.1.1 "><p id="p1780414862913"><a name="p1780414862913"></a><a name="p1780414862913"></a><a href="IsNan接口.md">IsNan</a></p>
</td>
<td class="cellrowborder" valign="top" width="62.29%" headers="mcps1.2.3.1.2 "><p id="p38041818296"><a name="p38041818296"></a><a name="p38041818296"></a>按元素判断输入的浮点数是否为nan。</p>
</td>
</tr>
<tr id="row13670181019299"><td class="cellrowborder" valign="top" width="37.71%" headers="mcps1.2.3.1.1 "><p id="p103301445314"><a name="p103301445314"></a><a name="p103301445314"></a><a href="IsInf接口.md">IsInf</a></p>
</td>
<td class="cellrowborder" valign="top" width="62.29%" headers="mcps1.2.3.1.2 "><p id="p06701810122920"><a name="p06701810122920"></a><a name="p06701810122920"></a>按元素判断输入的浮点数是否为&plusmn;INF。</p>
</td>
</tr>
<tr id="row37451613142914"><td class="cellrowborder" valign="top" width="37.71%" headers="mcps1.2.3.1.1 "><p id="p733064410319"><a name="p733064410319"></a><a name="p733064410319"></a><a href="Rint接口.md">Rint</a></p>
</td>
<td class="cellrowborder" valign="top" width="62.29%" headers="mcps1.2.3.1.2 "><p id="p1474571382914"><a name="p1474571382914"></a><a name="p1474571382914"></a>获取与输入数据最接近的整数。</p>
</td>
</tr>
<tr id="row98766156296"><td class="cellrowborder" valign="top" width="37.71%" headers="mcps1.2.3.1.1 "><p id="p0876111510296"><a name="p0876111510296"></a><a name="p0876111510296"></a><a href="SinCos接口.md">SinCos</a></p>
</td>
<td class="cellrowborder" valign="top" width="62.29%" headers="mcps1.2.3.1.2 "><p id="p17877815152917"><a name="p17877815152917"></a><a name="p17877815152917"></a>按元素进行正弦计算和余弦计算，分别获得正弦和余弦的结果。</p>
</td>
</tr>
<tr id="row12309142814569"><td class="cellrowborder" valign="top" width="37.71%" headers="mcps1.2.3.1.1 "><p id="p12309628115619"><a name="p12309628115619"></a><a name="p12309628115619"></a><a href="LogicalNot.md">LogicalNot</a></p>
</td>
<td class="cellrowborder" valign="top" width="62.29%" headers="mcps1.2.3.1.2 "><p id="p20309152805616"><a name="p20309152805616"></a><a name="p20309152805616"></a>按元素进行取反操作。</p>
</td>
</tr>
<tr id="row94521039125610"><td class="cellrowborder" valign="top" width="37.71%" headers="mcps1.2.3.1.1 "><p id="p14452739185615"><a name="p14452739185615"></a><a name="p14452739185615"></a><a href="LogicalAnd.md">LogicalAnd</a></p>
</td>
<td class="cellrowborder" valign="top" width="62.29%" headers="mcps1.2.3.1.2 "><p id="p16453439125612"><a name="p16453439125612"></a><a name="p16453439125612"></a>按元素进行与操作。</p>
</td>
</tr>
<tr id="row1654215408567"><td class="cellrowborder" valign="top" width="37.71%" headers="mcps1.2.3.1.1 "><p id="p15431040115616"><a name="p15431040115616"></a><a name="p15431040115616"></a><a href="LogicalAnds.md">LogicalAnds</a></p>
</td>
<td class="cellrowborder" valign="top" width="62.29%" headers="mcps1.2.3.1.2 "><p id="p9543440175617"><a name="p9543440175617"></a><a name="p9543440175617"></a>输入矢量内的每个元素与标量进行与操作。</p>
</td>
</tr>
<tr id="row499344016568"><td class="cellrowborder" valign="top" width="37.71%" headers="mcps1.2.3.1.1 "><p id="p1599344085618"><a name="p1599344085618"></a><a name="p1599344085618"></a><a href="LogicalOr.md">LogicalOr</a></p>
</td>
<td class="cellrowborder" valign="top" width="62.29%" headers="mcps1.2.3.1.2 "><p id="p1599364045612"><a name="p1599364045612"></a><a name="p1599364045612"></a>按元素进行或操作。</p>
</td>
</tr>
<tr id="row5448341105615"><td class="cellrowborder" valign="top" width="37.71%" headers="mcps1.2.3.1.1 "><p id="p5448204112567"><a name="p5448204112567"></a><a name="p5448204112567"></a><a href="LogicalOrs.md">LogicalOrs</a></p>
</td>
<td class="cellrowborder" valign="top" width="62.29%" headers="mcps1.2.3.1.2 "><p id="p194481841115610"><a name="p194481841115610"></a><a name="p194481841115610"></a>输入矢量内的每个元素与标量进行或操作。</p>
</td>
</tr>
<tr id="row4668163093217"><td class="cellrowborder" valign="top" width="37.71%" headers="mcps1.2.3.1.1 "><p id="p187781634193218"><a name="p187781634193218"></a><a name="p187781634193218"></a><a href="LogicalXor.md">LogicalXor</a></p>
</td>
<td class="cellrowborder" valign="top" width="62.29%" headers="mcps1.2.3.1.2 "><p id="p147786344321"><a name="p147786344321"></a><a name="p147786344321"></a>按元素进行逻辑异或操作。</p>
</td>
</tr>
<tr id="row5422125043317"><td class="cellrowborder" valign="top" width="37.71%" headers="mcps1.2.3.1.1 "><p id="p186138413415"><a name="p186138413415"></a><a name="p186138413415"></a><a href="BitwiseNot.md">BitwiseNot</a></p>
</td>
<td class="cellrowborder" valign="top" width="62.29%" headers="mcps1.2.3.1.2 "><p id="p861494153417"><a name="p861494153417"></a><a name="p861494153417"></a>逐比特对输入进行取反。</p>
</td>
</tr>
<tr id="row3119242135615"><td class="cellrowborder" valign="top" width="37.71%" headers="mcps1.2.3.1.1 "><p id="p1111924235618"><a name="p1111924235618"></a><a name="p1111924235618"></a><a href="BitwiseAnd.md">BitwiseAnd</a></p>
</td>
<td class="cellrowborder" valign="top" width="62.29%" headers="mcps1.2.3.1.2 "><p id="p20119174210565"><a name="p20119174210565"></a><a name="p20119174210565"></a>逐比特对两个输入进行与操作。</p>
</td>
</tr>
<tr id="row1847385613338"><td class="cellrowborder" valign="top" width="37.71%" headers="mcps1.2.3.1.1 "><p id="p155221232342"><a name="p155221232342"></a><a name="p155221232342"></a><a href="BitwiseOr.md">BitwiseOr</a></p>
</td>
<td class="cellrowborder" valign="top" width="62.29%" headers="mcps1.2.3.1.2 "><p id="p4522152393416"><a name="p4522152393416"></a><a name="p4522152393416"></a>逐比特对两个输入进行或操作。</p>
</td>
</tr>
<tr id="row394945893311"><td class="cellrowborder" valign="top" width="37.71%" headers="mcps1.2.3.1.1 "><p id="p10748540103412"><a name="p10748540103412"></a><a name="p10748540103412"></a><a href="BitwiseXor.md">BitwiseXor</a></p>
</td>
<td class="cellrowborder" valign="top" width="62.29%" headers="mcps1.2.3.1.2 "><p id="p974816402341"><a name="p974816402341"></a><a name="p974816402341"></a>逐比特对两个输入进行异或操作。</p>
</td>
</tr>
<tr id="row36751142135614"><td class="cellrowborder" valign="top" width="37.71%" headers="mcps1.2.3.1.1 "><p id="p20675134217564"><a name="p20675134217564"></a><a name="p20675134217564"></a><a href="Where.md">Where</a></p>
</td>
<td class="cellrowborder" valign="top" width="62.29%" headers="mcps1.2.3.1.2 "><p id="p7675194211563"><a name="p7675194211563"></a><a name="p7675194211563"></a>根据指定的条件，从两个源操作数中选择元素，生成目标操作数。</p>
</td>
</tr>
</tbody>
</table>

**表 15**  量化操作API列表

<a name="table11421635141314"></a>
<table><thead align="left"><tr id="row10422735121314"><th class="cellrowborder" valign="top" width="37.71%" id="mcps1.2.3.1.1"><p id="p3422183561312"><a name="p3422183561312"></a><a name="p3422183561312"></a>接口名</p>
</th>
<th class="cellrowborder" valign="top" width="62.29%" id="mcps1.2.3.1.2"><p id="p542213354138"><a name="p542213354138"></a><a name="p542213354138"></a>功能描述</p>
</th>
</tr>
</thead>
<tbody><tr id="row101065443561"><td class="cellrowborder" valign="top" width="37.71%" headers="mcps1.2.3.1.1 "><p id="p1710674455615"><a name="p1710674455615"></a><a name="p1710674455615"></a><a href="AntiQuantize.md">AntiQuantize</a></p>
</td>
<td class="cellrowborder" valign="top" width="62.29%" headers="mcps1.2.3.1.2 "><p id="p1910684465613"><a name="p1910684465613"></a><a name="p1910684465613"></a>按元素做伪量化计算，比如将int8_t数据类型伪量化为half数据类型。</p>
</td>
</tr>
<tr id="row164273359136"><td class="cellrowborder" valign="top" width="37.71%" headers="mcps1.2.3.1.1 "><p id="p742783581318"><a name="p742783581318"></a><a name="p742783581318"></a><a href="AscendAntiQuant.md">AscendAntiQuant</a></p>
</td>
<td class="cellrowborder" valign="top" width="62.29%" headers="mcps1.2.3.1.2 "><p id="p5427835121317"><a name="p5427835121317"></a><a name="p5427835121317"></a>按元素做伪量化计算，比如将int8_t数据类型伪量化为half数据类型。</p>
</td>
</tr>
<tr id="row87731046155619"><td class="cellrowborder" valign="top" width="37.71%" headers="mcps1.2.3.1.1 "><p id="p167731446135611"><a name="p167731446135611"></a><a name="p167731446135611"></a><a href="Dequantize.md">Dequantize</a></p>
</td>
<td class="cellrowborder" valign="top" width="62.29%" headers="mcps1.2.3.1.2 "><p id="p9773446115615"><a name="p9773446115615"></a><a name="p9773446115615"></a>按元素做反量化计算，比如将int32_t数据类型反量化为half/float等数据类型。</p>
</td>
</tr>
<tr id="row12427435111311"><td class="cellrowborder" valign="top" width="37.71%" headers="mcps1.2.3.1.1 "><p id="p1642733591318"><a name="p1642733591318"></a><a name="p1642733591318"></a><a href="AscendDequant.md">AscendDequant</a></p>
</td>
<td class="cellrowborder" valign="top" width="62.29%" headers="mcps1.2.3.1.2 "><p id="p11427635151312"><a name="p11427635151312"></a><a name="p11427635151312"></a>按元素做反量化计算，比如将int32_t数据类型反量化为half/float等数据类型。</p>
</td>
</tr>
<tr id="row18942184911567"><td class="cellrowborder" valign="top" width="37.71%" headers="mcps1.2.3.1.1 "><p id="p1494254905613"><a name="p1494254905613"></a><a name="p1494254905613"></a><a href="Quantize.md">Quantize</a></p>
</td>
<td class="cellrowborder" valign="top" width="62.29%" headers="mcps1.2.3.1.2 "><p id="p7942134917563"><a name="p7942134917563"></a><a name="p7942134917563"></a>按元素做量化计算，比如将half/float数据类型量化为int8_t数据类型。</p>
</td>
</tr>
<tr id="row164271635201317"><td class="cellrowborder" valign="top" width="37.71%" headers="mcps1.2.3.1.1 "><p id="p2427435201319"><a name="p2427435201319"></a><a name="p2427435201319"></a><a href="AscendQuant.md">AscendQuant</a></p>
</td>
<td class="cellrowborder" valign="top" width="62.29%" headers="mcps1.2.3.1.2 "><p id="p342718351139"><a name="p342718351139"></a><a name="p342718351139"></a>按元素做量化计算，比如将half/float数据类型量化为int8_t数据类型。</p>
</td>
</tr>
</tbody>
</table>

**表 16**  归一化操作API列表

<a name="table3781201031415"></a>
<table><thead align="left"><tr id="row12781510111418"><th class="cellrowborder" valign="top" width="37.71%" id="mcps1.2.3.1.1"><p id="p12782101020142"><a name="p12782101020142"></a><a name="p12782101020142"></a>接口名</p>
</th>
<th class="cellrowborder" valign="top" width="62.29%" id="mcps1.2.3.1.2"><p id="p6782101016142"><a name="p6782101016142"></a><a name="p6782101016142"></a>功能描述</p>
</th>
</tr>
</thead>
<tbody><tr id="row8786410121418"><td class="cellrowborder" valign="top" width="37.71%" headers="mcps1.2.3.1.1 "><p id="p4786131012145"><a name="p4786131012145"></a><a name="p4786131012145"></a><a href="BatchNorm.md">BatchNorm</a></p>
</td>
<td class="cellrowborder" valign="top" width="62.29%" headers="mcps1.2.3.1.2 "><p id="p1778631013141"><a name="p1778631013141"></a><a name="p1778631013141"></a>对于每个batch中的样本，对其输入的每个特征在batch的维度上进行归一化。</p>
</td>
</tr>
<tr id="row87861710151413"><td class="cellrowborder" valign="top" width="37.71%" headers="mcps1.2.3.1.1 "><p id="p11786410151410"><a name="p11786410151410"></a><a name="p11786410151410"></a><a href="DeepNorm.md">DeepNorm</a></p>
</td>
<td class="cellrowborder" valign="top" width="62.29%" headers="mcps1.2.3.1.2 "><p id="p17861410111419"><a name="p17861410111419"></a><a name="p17861410111419"></a>在深层神经网络训练过程中，可以替代LayerNorm的一种归一化方法。</p>
</td>
</tr>
<tr id="row18506114553319"><td class="cellrowborder" valign="top" width="37.71%" headers="mcps1.2.3.1.1 "><p id="p571415485334"><a name="p571415485334"></a><a name="p571415485334"></a><a href="GroupNorm.md">GroupNorm</a></p>
</td>
<td class="cellrowborder" valign="top" width="62.29%" headers="mcps1.2.3.1.2 "><p id="p11714174853314"><a name="p11714174853314"></a><a name="p11714174853314"></a>将输入的C维度分为groupNum组，对每一组数据进行标准化。</p>
</td>
</tr>
<tr id="row5786191091412"><td class="cellrowborder" valign="top" width="37.71%" headers="mcps1.2.3.1.1 "><p id="p11786131071415"><a name="p11786131071415"></a><a name="p11786131071415"></a><a href="LayerNorm.md">LayerNorm</a></p>
</td>
<td class="cellrowborder" valign="top" width="62.29%" headers="mcps1.2.3.1.2 "><p id="p157861910161416"><a name="p157861910161416"></a><a name="p157861910161416"></a>将输入数据收敛到[0, 1]之间，可以规范网络层输入输出数据分布的一种归一化方法。</p>
</td>
</tr>
<tr id="row378616105148"><td class="cellrowborder" valign="top" width="37.71%" headers="mcps1.2.3.1.1 "><p id="p47863104149"><a name="p47863104149"></a><a name="p47863104149"></a><a href="LayerNorm.md">LayerNormGrad</a></p>
</td>
<td class="cellrowborder" valign="top" width="62.29%" headers="mcps1.2.3.1.2 "><p id="p11786141091411"><a name="p11786141091411"></a><a name="p11786141091411"></a>用于计算LayerNorm的反向传播梯度。</p>
</td>
</tr>
<tr id="row7786111071411"><td class="cellrowborder" valign="top" width="37.71%" headers="mcps1.2.3.1.1 "><p id="p1978721091412"><a name="p1978721091412"></a><a name="p1978721091412"></a><a href="LayerNormGradBeta.md">LayerNormGradBeta</a></p>
</td>
<td class="cellrowborder" valign="top" width="62.29%" headers="mcps1.2.3.1.2 "><p id="p978771018141"><a name="p978771018141"></a><a name="p978771018141"></a>用于获取反向beta/gmma的数值，和LayerNormGrad共同输出pdx, gmma和beta。</p>
</td>
</tr>
<tr id="row1833102211208"><td class="cellrowborder" valign="top" width="37.71%" headers="mcps1.2.3.1.1 "><p id="p233172292010"><a name="p233172292010"></a><a name="p233172292010"></a><a href="Normalize.md">Normalize</a></p>
</td>
<td class="cellrowborder" valign="top" width="62.29%" headers="mcps1.2.3.1.2 "><p id="p33482202020"><a name="p33482202020"></a><a name="p33482202020"></a><a href="LayerNorm.md">LayerNorm</a>中，已知均值和方差，计算shape为[A，R]的输入数据的标准差的倒数rstd和归一化输出y。</p>
</td>
</tr>
<tr id="row478741061416"><td class="cellrowborder" valign="top" width="37.71%" headers="mcps1.2.3.1.1 "><p id="p7787121071417"><a name="p7787121071417"></a><a name="p7787121071417"></a><a href="RmsNorm.md">RmsNorm</a></p>
</td>
<td class="cellrowborder" valign="top" width="62.29%" headers="mcps1.2.3.1.2 "><p id="p19787171017144"><a name="p19787171017144"></a><a name="p19787171017144"></a>实现对shape大小为[B，S，H]的输入数据的RmsNorm归一化。</p>
</td>
</tr>
<tr id="row48116253209"><td class="cellrowborder" valign="top" width="37.71%" headers="mcps1.2.3.1.1 "><p id="p58111925192011"><a name="p58111925192011"></a><a name="p58111925192011"></a><a href="WelfordUpdate.md">WelfordUpdate</a></p>
</td>
<td class="cellrowborder" valign="top" width="62.29%" headers="mcps1.2.3.1.2 "><p id="p48111625102019"><a name="p48111625102019"></a><a name="p48111625102019"></a>实现Welford算法的前处理。</p>
</td>
</tr>
<tr id="row189721927192015"><td class="cellrowborder" valign="top" width="37.71%" headers="mcps1.2.3.1.1 "><p id="p29731627182019"><a name="p29731627182019"></a><a name="p29731627182019"></a><a href="WelfordFinalize.md">WelfordFinalize</a></p>
</td>
<td class="cellrowborder" valign="top" width="62.29%" headers="mcps1.2.3.1.2 "><p id="p69731627112018"><a name="p69731627112018"></a><a name="p69731627112018"></a>实现Welford算法的后处理。</p>
</td>
</tr>
</tbody>
</table>

**表 17**  激活函数API列表

<a name="table952317081517"></a>
<table><thead align="left"><tr id="row052317019157"><th class="cellrowborder" valign="top" width="37.71%" id="mcps1.2.3.1.1"><p id="p13523605151"><a name="p13523605151"></a><a name="p13523605151"></a>接口名</p>
</th>
<th class="cellrowborder" valign="top" width="62.29%" id="mcps1.2.3.1.2"><p id="p20523100141510"><a name="p20523100141510"></a><a name="p20523100141510"></a>功能描述</p>
</th>
</tr>
</thead>
<tbody><tr id="row1523406156"><td class="cellrowborder" valign="top" width="37.71%" headers="mcps1.2.3.1.1 "><p id="p135235061516"><a name="p135235061516"></a><a name="p135235061516"></a><a href="AdjustSoftMaxRes.md">AdjustSoftMaxRes</a></p>
</td>
<td class="cellrowborder" valign="top" width="62.29%" headers="mcps1.2.3.1.2 "><p id="p1952380121514"><a name="p1952380121514"></a><a name="p1952380121514"></a>用于对SoftMax相关计算结果做后处理，调整SoftMax的计算结果为指定的值。</p>
</td>
</tr>
<tr id="row10523408156"><td class="cellrowborder" valign="top" width="37.71%" headers="mcps1.2.3.1.1 "><p id="p1452340201510"><a name="p1452340201510"></a><a name="p1452340201510"></a><a href="FasterGelu.md">FasterGelu</a></p>
</td>
<td class="cellrowborder" valign="top" width="62.29%" headers="mcps1.2.3.1.2 "><p id="p15232011159"><a name="p15232011159"></a><a name="p15232011159"></a>FastGelu化简版本的一种激活函数。</p>
</td>
</tr>
<tr id="row452380181515"><td class="cellrowborder" valign="top" width="37.71%" headers="mcps1.2.3.1.1 "><p id="p752416091513"><a name="p752416091513"></a><a name="p752416091513"></a><a href="FasterGeluV2.md">FasterGeluV2</a></p>
</td>
<td class="cellrowborder" valign="top" width="62.29%" headers="mcps1.2.3.1.2 "><p id="p552413017153"><a name="p552413017153"></a><a name="p552413017153"></a><span>实现FastGeluV2</span>版本的一种激活函数。</p>
</td>
</tr>
<tr id="row16524110151518"><td class="cellrowborder" valign="top" width="37.71%" headers="mcps1.2.3.1.1 "><p id="p1952430161515"><a name="p1952430161515"></a><a name="p1952430161515"></a><a href="GeGLU.md">GeGLU</a></p>
</td>
<td class="cellrowborder" valign="top" width="62.29%" headers="mcps1.2.3.1.2 "><p id="p165241407155"><a name="p165241407155"></a><a name="p165241407155"></a>采用GeLU作为激活函数的GLU变体。</p>
</td>
</tr>
<tr id="row65248014151"><td class="cellrowborder" valign="top" width="37.71%" headers="mcps1.2.3.1.1 "><p id="p252412091510"><a name="p252412091510"></a><a name="p252412091510"></a><a href="Gelu.md">Gelu</a></p>
</td>
<td class="cellrowborder" valign="top" width="62.29%" headers="mcps1.2.3.1.2 "><p id="p175241406158"><a name="p175241406158"></a><a name="p175241406158"></a>GELU是一个重要的激活函数，其灵感来源于relu和dropout，在激活中引入了随机正则的思想。</p>
</td>
</tr>
<tr id="row165241301159"><td class="cellrowborder" valign="top" width="37.71%" headers="mcps1.2.3.1.1 "><p id="p155243071519"><a name="p155243071519"></a><a name="p155243071519"></a><a href="LogSoftMax.md">LogSoftMax</a></p>
</td>
<td class="cellrowborder" valign="top" width="62.29%" headers="mcps1.2.3.1.2 "><p id="p35248041516"><a name="p35248041516"></a><a name="p35248041516"></a>对输入tensor做LogSoftmax计算。</p>
</td>
</tr>
<tr id="row1352419071515"><td class="cellrowborder" valign="top" width="37.71%" headers="mcps1.2.3.1.1 "><p id="p1152410051518"><a name="p1152410051518"></a><a name="p1152410051518"></a><a href="ReGlu.md">ReGlu</a></p>
</td>
<td class="cellrowborder" valign="top" width="62.29%" headers="mcps1.2.3.1.2 "><p id="p75248013150"><a name="p75248013150"></a><a name="p75248013150"></a>一种GLU变体，使用Relu作为激活函数。</p>
</td>
</tr>
<tr id="row852413013150"><td class="cellrowborder" valign="top" width="37.71%" headers="mcps1.2.3.1.1 "><p id="p852415071513"><a name="p852415071513"></a><a name="p852415071513"></a><a href="Sigmoid.md">Sigmoid</a></p>
</td>
<td class="cellrowborder" valign="top" width="62.29%" headers="mcps1.2.3.1.2 "><p id="p1052419031510"><a name="p1052419031510"></a><a name="p1052419031510"></a>按元素做逻辑回归Sigmoid。</p>
</td>
</tr>
<tr id="row1252416071515"><td class="cellrowborder" valign="top" width="37.71%" headers="mcps1.2.3.1.1 "><p id="p25241601158"><a name="p25241601158"></a><a name="p25241601158"></a><a href="Silu.md">Silu</a></p>
</td>
<td class="cellrowborder" valign="top" width="62.29%" headers="mcps1.2.3.1.2 "><p id="p852416041517"><a name="p852416041517"></a><a name="p852416041517"></a>按元素做Silu运算。</p>
</td>
</tr>
<tr id="row5524190141518"><td class="cellrowborder" valign="top" width="37.71%" headers="mcps1.2.3.1.1 "><p id="p852419019156"><a name="p852419019156"></a><a name="p852419019156"></a><a href="SimpleSoftMax.md">SimpleSoftMax</a></p>
</td>
<td class="cellrowborder" valign="top" width="62.29%" headers="mcps1.2.3.1.2 "><p id="p35247012154"><a name="p35247012154"></a><a name="p35247012154"></a>使用计算好的sum和max数据对输入tensor做softmax计算。</p>
</td>
</tr>
<tr id="row1752419071511"><td class="cellrowborder" valign="top" width="37.71%" headers="mcps1.2.3.1.1 "><p id="p195249051510"><a name="p195249051510"></a><a name="p195249051510"></a><a href="SoftMax.md">SoftMax</a></p>
</td>
<td class="cellrowborder" valign="top" width="62.29%" headers="mcps1.2.3.1.2 "><p id="p16525100121512"><a name="p16525100121512"></a><a name="p16525100121512"></a>对输入tensor按行做Softmax计算。</p>
</td>
</tr>
<tr id="row452512011153"><td class="cellrowborder" valign="top" width="37.71%" headers="mcps1.2.3.1.1 "><p id="p205251707159"><a name="p205251707159"></a><a name="p205251707159"></a><a href="SoftmaxFlash.md">SoftmaxFlash</a></p>
</td>
<td class="cellrowborder" valign="top" width="62.29%" headers="mcps1.2.3.1.2 "><p id="p1152511081511"><a name="p1152511081511"></a><a name="p1152511081511"></a>SoftMax增强版本，除了可以对输入tensor做softmaxflash计算，还可以根据上一次softmax计算的sum和max来更新本次的softmax计算结果。</p>
</td>
</tr>
<tr id="row352510091518"><td class="cellrowborder" valign="top" width="37.71%" headers="mcps1.2.3.1.1 "><p id="p65251081518"><a name="p65251081518"></a><a name="p65251081518"></a><a href="SoftmaxFlashV2.md">SoftmaxFlashV2</a></p>
</td>
<td class="cellrowborder" valign="top" width="62.29%" headers="mcps1.2.3.1.2 "><p id="p175251107154"><a name="p175251107154"></a><a name="p175251107154"></a>SoftmaxFlash增强版本，对应FlashAttention-2算法。</p>
</td>
</tr>
<tr id="row46249508349"><td class="cellrowborder" valign="top" width="37.71%" headers="mcps1.2.3.1.1 "><p id="p10625550163411"><a name="p10625550163411"></a><a name="p10625550163411"></a><a href="SoftmaxFlashV3.md">SoftmaxFlashV3</a></p>
</td>
<td class="cellrowborder" valign="top" width="62.29%" headers="mcps1.2.3.1.2 "><p id="p662545013412"><a name="p662545013412"></a><a name="p662545013412"></a>SoftmaxFlash增强版本，对应Softmax PASA算法。</p>
</td>
</tr>
<tr id="row65251015155"><td class="cellrowborder" valign="top" width="37.71%" headers="mcps1.2.3.1.1 "><p id="p155254018158"><a name="p155254018158"></a><a name="p155254018158"></a><a href="SoftmaxGrad.md">SoftmaxGrad</a></p>
</td>
<td class="cellrowborder" valign="top" width="62.29%" headers="mcps1.2.3.1.2 "><p id="p8525507154"><a name="p8525507154"></a><a name="p8525507154"></a>对输入tensor做grad反向计算的一种方法。</p>
</td>
</tr>
<tr id="row152513016156"><td class="cellrowborder" valign="top" width="37.71%" headers="mcps1.2.3.1.1 "><p id="p352512001517"><a name="p352512001517"></a><a name="p352512001517"></a><a href="SoftmaxGradFront.md">SoftmaxGradFront</a></p>
</td>
<td class="cellrowborder" valign="top" width="62.29%" headers="mcps1.2.3.1.2 "><p id="p752570131513"><a name="p752570131513"></a><a name="p752570131513"></a>对输入tensor做grad反向计算的一种方法。</p>
</td>
</tr>
<tr id="row135257081512"><td class="cellrowborder" valign="top" width="37.71%" headers="mcps1.2.3.1.1 "><p id="p9525200111511"><a name="p9525200111511"></a><a name="p9525200111511"></a><a href="SwiGLU.md">SwiGLU</a></p>
</td>
<td class="cellrowborder" valign="top" width="62.29%" headers="mcps1.2.3.1.2 "><p id="p1952550131511"><a name="p1952550131511"></a><a name="p1952550131511"></a>采用Swish作为激活函数的GLU变体。</p>
</td>
</tr>
<tr id="row1252515041520"><td class="cellrowborder" valign="top" width="37.71%" headers="mcps1.2.3.1.1 "><p id="p752515071514"><a name="p752515071514"></a><a name="p752515071514"></a><a href="Swish.md">Swish</a></p>
</td>
<td class="cellrowborder" valign="top" width="62.29%" headers="mcps1.2.3.1.2 "><p id="p95251306151"><a name="p95251306151"></a><a name="p95251306151"></a>神经网络中的Swish激活函数。</p>
</td>
</tr>
</tbody>
</table>

**表 18**  归约操作API列表

<a name="table56871381153"></a>
<table><thead align="left"><tr id="row368753820157"><th class="cellrowborder" valign="top" width="37.71%" id="mcps1.2.3.1.1"><p id="p968711387154"><a name="p968711387154"></a><a name="p968711387154"></a>接口名</p>
</th>
<th class="cellrowborder" valign="top" width="62.29%" id="mcps1.2.3.1.2"><p id="p1368773841515"><a name="p1368773841515"></a><a name="p1368773841515"></a>功能描述</p>
</th>
</tr>
</thead>
<tbody><tr id="row16251161812813"><td class="cellrowborder" valign="top" width="37.71%" headers="mcps1.2.3.1.1 "><p id="p1068811382157"><a name="p1068811382157"></a><a name="p1068811382157"></a><a href="Sum.md">Sum</a></p>
</td>
<td class="cellrowborder" valign="top" width="62.29%" headers="mcps1.2.3.1.2 "><p id="p13688203811515"><a name="p13688203811515"></a><a name="p13688203811515"></a>获取最后一个维度的元素总和。</p>
</td>
</tr>
<tr id="row106871338111517"><td class="cellrowborder" valign="top" width="37.71%" headers="mcps1.2.3.1.1 "><p id="p16871238131513"><a name="p16871238131513"></a><a name="p16871238131513"></a><a href="Mean.md">Mean</a></p>
</td>
<td class="cellrowborder" valign="top" width="62.29%" headers="mcps1.2.3.1.2 "><p id="p1668715385158"><a name="p1668715385158"></a><a name="p1668715385158"></a>根据最后一轴的方向对各元素求平均值。</p>
</td>
</tr>
<tr id="row186871038171510"><td class="cellrowborder" valign="top" width="37.71%" headers="mcps1.2.3.1.1 "><p id="p1268793811159"><a name="p1268793811159"></a><a name="p1268793811159"></a><a href="ReduceXorSum.md">ReduceXorSum</a></p>
</td>
<td class="cellrowborder" valign="top" width="62.29%" headers="mcps1.2.3.1.2 "><p id="p12688163811519"><a name="p12688163811519"></a><a name="p12688163811519"></a>按照元素执行Xor（按位异或）运算，并将计算结果ReduceSum求和。</p>
</td>
</tr>
<tr id="row15688103871519"><td class="cellrowborder" valign="top" width="37.71%" headers="mcps1.2.3.1.1 "><p id="p95115441718"><a name="p95115441718"></a><a name="p95115441718"></a><a href="ReduceSum-111.md">ReduceSum</a></p>
</td>
<td class="cellrowborder" valign="top" width="62.29%" headers="mcps1.2.3.1.2 "><p id="p11517441717"><a name="p11517441717"></a><a name="p11517441717"></a>对一个多维向量按照指定的维度进行数据累加。</p>
</td>
</tr>
<tr id="row199410371158"><td class="cellrowborder" valign="top" width="37.71%" headers="mcps1.2.3.1.1 "><p id="p209412372517"><a name="p209412372517"></a><a name="p209412372517"></a><a href="ReduceMean.md">ReduceMean</a></p>
</td>
<td class="cellrowborder" valign="top" width="62.29%" headers="mcps1.2.3.1.2 "><p id="p09416377512"><a name="p09416377512"></a><a name="p09416377512"></a>对一个多维向量按照指定的维度求平均值。</p>
</td>
</tr>
<tr id="row6979134616516"><td class="cellrowborder" valign="top" width="37.71%" headers="mcps1.2.3.1.1 "><p id="p16979246851"><a name="p16979246851"></a><a name="p16979246851"></a><a href="ReduceMax-112.md">ReduceMax</a></p>
</td>
<td class="cellrowborder" valign="top" width="62.29%" headers="mcps1.2.3.1.2 "><p id="p197934616510"><a name="p197934616510"></a><a name="p197934616510"></a>对一个多维向量在指定的维度求最大值。</p>
</td>
</tr>
<tr id="row2429249255"><td class="cellrowborder" valign="top" width="37.71%" headers="mcps1.2.3.1.1 "><p id="p1542914496515"><a name="p1542914496515"></a><a name="p1542914496515"></a><a href="ReduceMin-113.md">ReduceMin</a></p>
</td>
<td class="cellrowborder" valign="top" width="62.29%" headers="mcps1.2.3.1.2 "><p id="p164291497518"><a name="p164291497518"></a><a name="p164291497518"></a>对一个多维向量在指定的维度求最小值。</p>
</td>
</tr>
<tr id="row1211653851"><td class="cellrowborder" valign="top" width="37.71%" headers="mcps1.2.3.1.1 "><p id="p112113531558"><a name="p112113531558"></a><a name="p112113531558"></a><a href="ReduceAny.md">ReduceAny</a></p>
</td>
<td class="cellrowborder" valign="top" width="62.29%" headers="mcps1.2.3.1.2 "><p id="p15211153654"><a name="p15211153654"></a><a name="p15211153654"></a>对一个多维向量在指定的维度求逻辑或。</p>
</td>
</tr>
<tr id="row18411155515512"><td class="cellrowborder" valign="top" width="37.71%" headers="mcps1.2.3.1.1 "><p id="p19411195519518"><a name="p19411195519518"></a><a name="p19411195519518"></a><a href="ReduceAll.md">ReduceAll</a></p>
</td>
<td class="cellrowborder" valign="top" width="62.29%" headers="mcps1.2.3.1.2 "><p id="p17411195512511"><a name="p17411195512511"></a><a name="p17411195512511"></a>对一个多维向量在指定的维度求逻辑与。</p>
</td>
</tr>
<tr id="row20399155816518"><td class="cellrowborder" valign="top" width="37.71%" headers="mcps1.2.3.1.1 "><p id="p4399958855"><a name="p4399958855"></a><a name="p4399958855"></a><a href="ReduceProd.md">ReduceProd</a></p>
</td>
<td class="cellrowborder" valign="top" width="62.29%" headers="mcps1.2.3.1.2 "><p id="p183995581754"><a name="p183995581754"></a><a name="p183995581754"></a>对一个多维向量在指定的维度求积。</p>
</td>
</tr>
</tbody>
</table>

**表 19**  排序操作API列表

<a name="table1075717581619"></a>
<table><thead align="left"><tr id="row87570510167"><th class="cellrowborder" valign="top" width="37.71%" id="mcps1.2.3.1.1"><p id="p87573520167"><a name="p87573520167"></a><a name="p87573520167"></a>接口名</p>
</th>
<th class="cellrowborder" valign="top" width="62.29%" id="mcps1.2.3.1.2"><p id="p1475714531618"><a name="p1475714531618"></a><a name="p1475714531618"></a>功能描述</p>
</th>
</tr>
</thead>
<tbody><tr id="row475745191616"><td class="cellrowborder" valign="top" width="37.71%" headers="mcps1.2.3.1.1 "><p id="p875865181616"><a name="p875865181616"></a><a name="p875865181616"></a><a href="TopK.md">TopK</a></p>
</td>
<td class="cellrowborder" valign="top" width="62.29%" headers="mcps1.2.3.1.2 "><p id="p97581255164"><a name="p97581255164"></a><a name="p97581255164"></a>获取最后一个维度的前k个最大值或最小值及其对应的索引。</p>
</td>
</tr>
<tr id="row97581516168"><td class="cellrowborder" valign="top" width="37.71%" headers="mcps1.2.3.1.1 "><p id="p1475812517162"><a name="p1475812517162"></a><a name="p1475812517162"></a><a href="Concat.md">Concat</a></p>
</td>
<td class="cellrowborder" valign="top" width="62.29%" headers="mcps1.2.3.1.2 "><p id="p375875131613"><a name="p375875131613"></a><a name="p375875131613"></a>对数据进行预处理，将要排序的源操作数srcLocal一一对应的合入目标数据concatLocal中，数据预处理完后，可以进行Sort。</p>
</td>
</tr>
<tr id="row14758650165"><td class="cellrowborder" valign="top" width="37.71%" headers="mcps1.2.3.1.1 "><p id="p47583591616"><a name="p47583591616"></a><a name="p47583591616"></a><a href="Extract.md">Extract</a></p>
</td>
<td class="cellrowborder" valign="top" width="62.29%" headers="mcps1.2.3.1.2 "><p id="p87581950161"><a name="p87581950161"></a><a name="p87581950161"></a>处理Sort的结果数据，输出排序后的value和index。</p>
</td>
</tr>
<tr id="row57583513169"><td class="cellrowborder" valign="top" width="37.71%" headers="mcps1.2.3.1.1 "><p id="p1375813514166"><a name="p1375813514166"></a><a name="p1375813514166"></a><a href="Sort.md">Sort</a></p>
</td>
<td class="cellrowborder" valign="top" width="62.29%" headers="mcps1.2.3.1.2 "><p id="p14758554166"><a name="p14758554166"></a><a name="p14758554166"></a>排序函数，按照数值大小进行降序排序。</p>
</td>
</tr>
<tr id="row475815512168"><td class="cellrowborder" valign="top" width="37.71%" headers="mcps1.2.3.1.1 "><p id="p17581591616"><a name="p17581591616"></a><a name="p17581591616"></a><a href="MrgSort-114.md">MrgSort</a></p>
</td>
<td class="cellrowborder" valign="top" width="62.29%" headers="mcps1.2.3.1.2 "><p id="p1375885121617"><a name="p1375885121617"></a><a name="p1375885121617"></a>将已经排好序的最多4条队列，合并排列成1条队列，结果按照score域由大到小排序。</p>
</td>
</tr>
</tbody>
</table>

**表 20**  数据过滤API列表

<a name="table7398513176"></a>
<table><thead align="left"><tr id="row133985161711"><th class="cellrowborder" valign="top" width="37.71%" id="mcps1.2.3.1.1"><p id="p23983191712"><a name="p23983191712"></a><a name="p23983191712"></a>接口名</p>
</th>
<th class="cellrowborder" valign="top" width="62.29%" id="mcps1.2.3.1.2"><p id="p10398111111716"><a name="p10398111111716"></a><a name="p10398111111716"></a>功能描述</p>
</th>
</tr>
</thead>
<tbody><tr id="row1943972012455"><td class="cellrowborder" valign="top" width="37.71%" headers="mcps1.2.3.1.1 "><p id="p138751045194610"><a name="p138751045194610"></a><a name="p138751045194610"></a><a href="Select-116.md">Select</a></p>
</td>
<td class="cellrowborder" valign="top" width="62.29%" headers="mcps1.2.3.1.2 "><p id="p0875445174619"><a name="p0875445174619"></a><a name="p0875445174619"></a>给定两个源操作数src0和src1，根据maskTensor相应位置的值（非bit位）选取元素，得到目的操作数dst。</p>
</td>
</tr>
<tr id="row193983120178"><td class="cellrowborder" valign="top" width="37.71%" headers="mcps1.2.3.1.1 "><p id="p103981111179"><a name="p103981111179"></a><a name="p103981111179"></a><a href="DropOut.md">DropOut</a></p>
</td>
<td class="cellrowborder" valign="top" width="62.29%" headers="mcps1.2.3.1.2 "><p id="p20398121101710"><a name="p20398121101710"></a><a name="p20398121101710"></a>提供根据MaskTensor对源操作数进行过滤的功能，得到目的操作数。</p>
</td>
</tr>
</tbody>
</table>

**表 21**  张量变换API列表

<a name="table86595781819"></a>
<table><thead align="left"><tr id="row16660147101819"><th class="cellrowborder" valign="top" width="37.669999999999995%" id="mcps1.2.3.1.1"><p id="p866010718184"><a name="p866010718184"></a><a name="p866010718184"></a>接口名</p>
</th>
<th class="cellrowborder" valign="top" width="62.33%" id="mcps1.2.3.1.2"><p id="p66603741815"><a name="p66603741815"></a><a name="p66603741815"></a>功能描述</p>
</th>
</tr>
</thead>
<tbody><tr id="row02335914543"><td class="cellrowborder" valign="top" width="37.669999999999995%" headers="mcps1.2.3.1.1 "><p id="p192315965413"><a name="p192315965413"></a><a name="p192315965413"></a><a href="Transpose-117.md">Transpose</a></p>
</td>
<td class="cellrowborder" valign="top" width="62.33%" headers="mcps1.2.3.1.2 "><p id="p1248169212"><a name="p1248169212"></a><a name="p1248169212"></a>对输入数据进行数据排布及Reshape操作。</p>
</td>
</tr>
<tr id="row113798510352"><td class="cellrowborder" valign="top" width="37.669999999999995%" headers="mcps1.2.3.1.1 "><p id="p73791051163519"><a name="p73791051163519"></a><a name="p73791051163519"></a><a href="TransData.md">TransData</a></p>
</td>
<td class="cellrowborder" valign="top" width="62.33%" headers="mcps1.2.3.1.2 "><p id="p1837919517358"><a name="p1837919517358"></a><a name="p1837919517358"></a>将输入数据的排布格式转换为目标排布格式。</p>
</td>
</tr>
<tr id="row12653659194315"><td class="cellrowborder" valign="top" width="37.669999999999995%" headers="mcps1.2.3.1.1 "><p id="p68351016134419"><a name="p68351016134419"></a><a name="p68351016134419"></a><a href="Broadcast.md">Broadcast</a></p>
</td>
<td class="cellrowborder" valign="top" width="62.33%" headers="mcps1.2.3.1.2 "><p id="p28356164441"><a name="p28356164441"></a><a name="p28356164441"></a>将输入按照输出shape进行广播。</p>
</td>
</tr>
<tr id="row061716212442"><td class="cellrowborder" valign="top" width="37.669999999999995%" headers="mcps1.2.3.1.1 "><p id="p118359168448"><a name="p118359168448"></a><a name="p118359168448"></a><a href="Pad.md">Pad</a></p>
</td>
<td class="cellrowborder" valign="top" width="62.33%" headers="mcps1.2.3.1.2 "><p id="p483521619442"><a name="p483521619442"></a><a name="p483521619442"></a>对height * width的二维Tensor在width方向上pad到32B对齐。</p>
</td>
</tr>
<tr id="row1650326144416"><td class="cellrowborder" valign="top" width="37.669999999999995%" headers="mcps1.2.3.1.1 "><p id="p11835161634418"><a name="p11835161634418"></a><a name="p11835161634418"></a><a href="UnPad.md">UnPad</a></p>
</td>
<td class="cellrowborder" valign="top" width="62.33%" headers="mcps1.2.3.1.2 "><p id="p6835161674420"><a name="p6835161674420"></a><a name="p6835161674420"></a>对height * width的二维Tensor在width方向上进行unpad。</p>
</td>
</tr>
<tr id="row1922374193911"><td class="cellrowborder" valign="top" width="37.669999999999995%" headers="mcps1.2.3.1.1 "><p id="p022434193913"><a name="p022434193913"></a><a name="p022434193913"></a><a href="Fill-118.md">Fill</a></p>
</td>
<td class="cellrowborder" valign="top" width="62.33%" headers="mcps1.2.3.1.2 "><p id="p122418416392"><a name="p122418416392"></a><a name="p122418416392"></a>将Global Memory上的数据初始化为指定值。</p>
</td>
</tr>
</tbody>
</table>

**表 22**  索引计算API列表

<a name="table67319289189"></a>
<table><thead align="left"><tr id="row1873528161818"><th class="cellrowborder" valign="top" width="37.63%" id="mcps1.2.3.1.1"><p id="p473728141810"><a name="p473728141810"></a><a name="p473728141810"></a>接口名</p>
</th>
<th class="cellrowborder" valign="top" width="62.370000000000005%" id="mcps1.2.3.1.2"><p id="p1973328111812"><a name="p1973328111812"></a><a name="p1973328111812"></a>功能描述</p>
</th>
</tr>
</thead>
<tbody><tr id="row05463289557"><td class="cellrowborder" valign="top" width="37.63%" headers="mcps1.2.3.1.1 "><p id="p1045773617556"><a name="p1045773617556"></a><a name="p1045773617556"></a><a href="Arange-115.md">Arange</a></p>
</td>
<td class="cellrowborder" valign="top" width="62.370000000000005%" headers="mcps1.2.3.1.2 "><p id="p189228338558"><a name="p189228338558"></a><a name="p189228338558"></a>给定起始值，等差值和长度，返回一个等差数列。</p>
</td>
</tr>
</tbody>
</table>

**表 23**  矩阵计算API列表

<a name="table16634248182011"></a>
<table><thead align="left"><tr id="row763454810205"><th class="cellrowborder" valign="top" width="37.669999999999995%" id="mcps1.2.3.1.1"><p id="p16634164832017"><a name="p16634164832017"></a><a name="p16634164832017"></a>接口名</p>
</th>
<th class="cellrowborder" valign="top" width="62.33%" id="mcps1.2.3.1.2"><p id="p1363464815203"><a name="p1363464815203"></a><a name="p1363464815203"></a>功能描述</p>
</th>
</tr>
</thead>
<tbody><tr id="row2635748192012"><td class="cellrowborder" valign="top" width="37.669999999999995%" headers="mcps1.2.3.1.1 "><p id="p10635248122013"><a name="p10635248122013"></a><a name="p10635248122013"></a><a href="Matmul-Kernel侧接口.md">Matmul</a></p>
</td>
<td class="cellrowborder" valign="top" width="62.33%" headers="mcps1.2.3.1.2 "><p id="p1563517485201"><a name="p1563517485201"></a><a name="p1563517485201"></a>Matmul矩阵乘法的运算。</p>
</td>
</tr>
</tbody>
</table>

**表 24**  HCCL通信类API列表

<a name="table483522817566"></a>
<table><thead align="left"><tr id="row1183572813564"><th class="cellrowborder" valign="top" width="37.669999999999995%" id="mcps1.2.3.1.1"><p id="p4835152811563"><a name="p4835152811563"></a><a name="p4835152811563"></a>接口名</p>
</th>
<th class="cellrowborder" valign="top" width="62.33%" id="mcps1.2.3.1.2"><p id="p283502813566"><a name="p283502813566"></a><a name="p283502813566"></a>功能描述</p>
</th>
</tr>
</thead>
<tbody><tr id="row7534141110415"><td class="cellrowborder" valign="top" width="37.669999999999995%" headers="mcps1.2.3.1.1 "><p id="p13534411643"><a name="p13534411643"></a><a name="p13534411643"></a><a href="HCCL通信类.md">HCCL通信类</a></p>
</td>
<td class="cellrowborder" valign="top" width="62.33%" headers="mcps1.2.3.1.2 "><p id="p553418111947"><a name="p553418111947"></a><a name="p553418111947"></a>在AI Core侧编排集合通信任务。</p>
</td>
</tr>
</tbody>
</table>

**表 25**  卷积计算API列表

<a name="table12502184212139"></a>
<table><thead align="left"><tr id="row1150274261313"><th class="cellrowborder" valign="top" width="37.6%" id="mcps1.2.3.1.1"><p id="p45022423132"><a name="p45022423132"></a><a name="p45022423132"></a>接口名</p>
</th>
<th class="cellrowborder" valign="top" width="62.4%" id="mcps1.2.3.1.2"><p id="p75021942101318"><a name="p75021942101318"></a><a name="p75021942101318"></a>功能描述</p>
</th>
</tr>
</thead>
<tbody><tr id="row185021242151312"><td class="cellrowborder" valign="top" width="37.6%" headers="mcps1.2.3.1.1 "><p id="p250294215131"><a name="p250294215131"></a><a name="p250294215131"></a><a href="Conv3D.md">Conv3D</a></p>
</td>
<td class="cellrowborder" valign="top" width="62.4%" headers="mcps1.2.3.1.2 "><p id="p1650394291312"><a name="p1650394291312"></a><a name="p1650394291312"></a>3维卷积正向矩阵运算。</p>
</td>
</tr>
<tr id="row1217212361282"><td class="cellrowborder" valign="top" width="37.6%" headers="mcps1.2.3.1.1 "><p id="p5172113652817"><a name="p5172113652817"></a><a name="p5172113652817"></a><a href="Conv3DBackpropInput.md">Conv3DBackpropInput</a></p>
</td>
<td class="cellrowborder" valign="top" width="62.4%" headers="mcps1.2.3.1.2 "><p id="p317293622815"><a name="p317293622815"></a><a name="p317293622815"></a>卷积的反向运算，求解特征矩阵的反向传播误差。</p>
</td>
</tr>
<tr id="row15623133842814"><td class="cellrowborder" valign="top" width="37.6%" headers="mcps1.2.3.1.1 "><p id="p1162303802817"><a name="p1162303802817"></a><a name="p1162303802817"></a><a href="Conv3DBackpropFilter.md">Conv3DBackpropFilter</a></p>
</td>
<td class="cellrowborder" valign="top" width="62.4%" headers="mcps1.2.3.1.2 "><p id="p86233387286"><a name="p86233387286"></a><a name="p86233387286"></a>卷积的反向运算，求解权重的反向传播误差。</p>
</td>
</tr>
</tbody>
</table>

**表 26**  随机函数API列表

<a name="table20932123194516"></a>
<table><thead align="left"><tr id="row16932113164513"><th class="cellrowborder" valign="top" width="37.6%" id="mcps1.2.3.1.1"><p id="p29329318456"><a name="p29329318456"></a><a name="p29329318456"></a>接口名</p>
</th>
<th class="cellrowborder" valign="top" width="62.4%" id="mcps1.2.3.1.2"><p id="p17932123104517"><a name="p17932123104517"></a><a name="p17932123104517"></a>功能描述</p>
</th>
</tr>
</thead>
<tbody><tr id="row17932163194517"><td class="cellrowborder" valign="top" width="37.6%" headers="mcps1.2.3.1.1 "><p id="p2932131104520"><a name="p2932131104520"></a><a name="p2932131104520"></a><a href="PhiloxRandom.md">PhiloxRandom</a></p>
</td>
<td class="cellrowborder" valign="top" width="62.4%" headers="mcps1.2.3.1.2 "><p id="p19321731134518"><a name="p19321731134518"></a><a name="p19321731134518"></a>基于Philox随机数生成算法，给定随机数种子，生成若干的随机数。</p>
</td>
</tr>
</tbody>
</table>

## SIMT API<a name="section1416490133020"></a>

**表 27**  核函数定义API

<a name="table355621172410"></a>
<table><thead align="left"><tr id="row105561111192410"><th class="cellrowborder" valign="top" width="40%" id="mcps1.2.3.1.1"><p id="p11556191111244"><a name="p11556191111244"></a><a name="p11556191111244"></a>接口名</p>
</th>
<th class="cellrowborder" valign="top" width="60%" id="mcps1.2.3.1.2"><p id="p1655618115241"><a name="p1655618115241"></a><a name="p1655618115241"></a>功能描述</p>
</th>
</tr>
</thead>
<tbody><tr id="row10556171192410"><td class="cellrowborder" valign="top" width="40%" headers="mcps1.2.3.1.1 "><p id="p12556111118244"><a name="p12556111118244"></a><a name="p12556111118244"></a><a href="asc_vf_call.md">asc_vf_call</a></p>
</td>
<td class="cellrowborder" valign="top" width="60%" headers="mcps1.2.3.1.2 "><p id="p1179785312220"><a name="p1179785312220"></a><a name="p1179785312220"></a>启动SIMT VF（Vector Function）子任务，启动指定数目的线程，执行指定的SIMT核函数。</p>
</td>
</tr>
</tbody>
</table>

**表 28**  同步函数

<a name="table1990955453318"></a>
<table><thead align="left"><tr id="row690975463312"><th class="cellrowborder" valign="top" width="40%" id="mcps1.2.3.1.1"><p id="p189091354173316"><a name="p189091354173316"></a><a name="p189091354173316"></a>接口名</p>
</th>
<th class="cellrowborder" valign="top" width="60%" id="mcps1.2.3.1.2"><p id="p15909154153315"><a name="p15909154153315"></a><a name="p15909154153315"></a>功能描述</p>
</th>
</tr>
</thead>
<tbody><tr id="row209091454163314"><td class="cellrowborder" valign="top" width="40%" headers="mcps1.2.3.1.1 "><p id="p79091954113318"><a name="p79091954113318"></a><a name="p79091954113318"></a><a href="asc_syncthreads.md">asc_syncthreads</a></p>
</td>
<td class="cellrowborder" valign="top" width="60%" headers="mcps1.2.3.1.2 "><p id="p1690965443315"><a name="p1690965443315"></a><a name="p1690965443315"></a>等待当前thread block内所有thread代码都执行到该函数位置。</p>
</td>
</tr>
<tr id="row1390945433317"><td class="cellrowborder" valign="top" width="40%" headers="mcps1.2.3.1.1 "><p id="p1590925411337"><a name="p1590925411337"></a><a name="p1590925411337"></a><a href="asc_threadfence.md">asc_threadfence</a></p>
</td>
<td class="cellrowborder" valign="top" width="60%" headers="mcps1.2.3.1.2 "><p id="p139091542339"><a name="p139091542339"></a><a name="p139091542339"></a>用于保证不同核对同一份全局、共享内存的访问过程中，写入操作的时序性。</p>
</td>
</tr>
<tr id="row1415719381045"><td class="cellrowborder" valign="top" width="40%" headers="mcps1.2.3.1.1 "><p id="p101577387413"><a name="p101577387413"></a><a name="p101577387413"></a><a href="asc_threadfence_block.md">asc_threadfence_block</a></p>
</td>
<td class="cellrowborder" valign="top" width="60%" headers="mcps1.2.3.1.2 "><p id="p14158138445"><a name="p14158138445"></a><a name="p14158138445"></a>用于协调同一个线程块（block）内的线程之间的内存操作顺序。确保在调用 threadfence_block() 之前的所有内存操作对该线程块内的所有线程可见。</p>
</td>
</tr>
</tbody>
</table>

**表 29**  数学函数

<a name="table139091354143312"></a>
<table><thead align="left"><tr id="row139091547334"><th class="cellrowborder" valign="top" width="40%" id="mcps1.2.3.1.1"><p id="p1090916540331"><a name="p1090916540331"></a><a name="p1090916540331"></a>接口名</p>
</th>
<th class="cellrowborder" valign="top" width="60%" id="mcps1.2.3.1.2"><p id="p15909154183312"><a name="p15909154183312"></a><a name="p15909154183312"></a>功能描述</p>
</th>
</tr>
</thead>
<tbody><tr id="row1190910544330"><td class="cellrowborder" valign="top" width="40%" headers="mcps1.2.3.1.1 "><p id="p4909155410334"><a name="p4909155410334"></a><a name="p4909155410334"></a><a href="tanf.md">tanf</a></p>
</td>
<td class="cellrowborder" valign="top" width="60%" headers="mcps1.2.3.1.2 "><p id="p5909165415330"><a name="p5909165415330"></a><a name="p5909165415330"></a>获取输入数据的三角函数正切值。</p>
</td>
</tr>
<tr id="row4909354143315"><td class="cellrowborder" valign="top" width="40%" headers="mcps1.2.3.1.1 "><p id="p1990916547337"><a name="p1990916547337"></a><a name="p1990916547337"></a><a href="tanhf.md">tanhf</a></p>
</td>
<td class="cellrowborder" valign="top" width="60%" headers="mcps1.2.3.1.2 "><p id="p39094543334"><a name="p39094543334"></a><a name="p39094543334"></a>获取输入数据的三角函数双曲正切值。</p>
</td>
</tr>
<tr id="row149407554520"><td class="cellrowborder" valign="top" width="40%" headers="mcps1.2.3.1.1 "><p id="p1094025518518"><a name="p1094025518518"></a><a name="p1094025518518"></a><a href="htanh.md">htanh</a></p>
</td>
<td class="cellrowborder" valign="top" width="60%" headers="mcps1.2.3.1.2 "><p id="p994116551751"><a name="p994116551751"></a><a name="p994116551751"></a>获取输入数据的三角函数双曲正切值。</p>
</td>
</tr>
<tr id="row185893815206"><td class="cellrowborder" valign="top" width="40%" headers="mcps1.2.3.1.1 "><p id="p1158968182016"><a name="p1158968182016"></a><a name="p1158968182016"></a><a href="h2tanh.md">h2tanh</a></p>
</td>
<td class="cellrowborder" valign="top" width="60%" headers="mcps1.2.3.1.2 "><p id="p14589188102011"><a name="p14589188102011"></a><a name="p14589188102011"></a>获取输入数据各元素的三角函数双曲正切值。</p>
</td>
</tr>
<tr id="row189091754133314"><td class="cellrowborder" valign="top" width="40%" headers="mcps1.2.3.1.1 "><p id="p1090955416336"><a name="p1090955416336"></a><a name="p1090955416336"></a><a href="tanpif.md">tanpif</a></p>
</td>
<td class="cellrowborder" valign="top" width="60%" headers="mcps1.2.3.1.2 "><p id="p9910654123318"><a name="p9910654123318"></a><a name="p9910654123318"></a>获取输入数据与π相乘的正切值。</p>
</td>
</tr>
<tr id="row19910205413332"><td class="cellrowborder" valign="top" width="40%" headers="mcps1.2.3.1.1 "><p id="p191075410331"><a name="p191075410331"></a><a name="p191075410331"></a><a href="atanf.md">atanf</a></p>
</td>
<td class="cellrowborder" valign="top" width="60%" headers="mcps1.2.3.1.2 "><p id="p1891012543332"><a name="p1891012543332"></a><a name="p1891012543332"></a>获取输入数据的反正切值。</p>
</td>
</tr>
<tr id="row7910954123312"><td class="cellrowborder" valign="top" width="40%" headers="mcps1.2.3.1.1 "><p id="p99101654203315"><a name="p99101654203315"></a><a name="p99101654203315"></a><a href="atan2f.md">atan2f</a></p>
</td>
<td class="cellrowborder" valign="top" width="60%" headers="mcps1.2.3.1.2 "><p id="p3910145418337"><a name="p3910145418337"></a><a name="p3910145418337"></a>获取输入数据y/x的反正切值。</p>
</td>
</tr>
<tr id="row20910195419336"><td class="cellrowborder" valign="top" width="40%" headers="mcps1.2.3.1.1 "><p id="p16910145418333"><a name="p16910145418333"></a><a name="p16910145418333"></a><a href="atanhf.md">atanhf</a></p>
</td>
<td class="cellrowborder" valign="top" width="60%" headers="mcps1.2.3.1.2 "><p id="p1291075483313"><a name="p1291075483313"></a><a name="p1291075483313"></a>获取输入数据的反双曲正切值。</p>
</td>
</tr>
<tr id="row1491010543333"><td class="cellrowborder" valign="top" width="40%" headers="mcps1.2.3.1.1 "><p id="p39101254103319"><a name="p39101254103319"></a><a name="p39101254103319"></a><a href="expf.md">expf</a></p>
</td>
<td class="cellrowborder" valign="top" width="60%" headers="mcps1.2.3.1.2 "><p id="p15910554123318"><a name="p15910554123318"></a><a name="p15910554123318"></a>指定输入x，获取e的x次方。</p>
</td>
</tr>
<tr id="row619618281479"><td class="cellrowborder" valign="top" width="40%" headers="mcps1.2.3.1.1 "><p id="p719612281674"><a name="p719612281674"></a><a name="p719612281674"></a><a href="hexp.md">hexp</a></p>
</td>
<td class="cellrowborder" valign="top" width="60%" headers="mcps1.2.3.1.2 "><p id="p719652819716"><a name="p719652819716"></a><a name="p719652819716"></a>指定输入x，获取e的x次方。</p>
</td>
</tr>
<tr id="row1747905614414"><td class="cellrowborder" valign="top" width="40%" headers="mcps1.2.3.1.1 "><p id="p204799560420"><a name="p204799560420"></a><a name="p204799560420"></a><a href="h2exp.md">h2exp</a></p>
</td>
<td class="cellrowborder" valign="top" width="60%" headers="mcps1.2.3.1.2 "><p id="p13479056049"><a name="p13479056049"></a><a name="p13479056049"></a>指定输入x，对x的各元素，获取e的该元素次方。</p>
</td>
</tr>
<tr id="row189102544333"><td class="cellrowborder" valign="top" width="40%" headers="mcps1.2.3.1.1 "><p id="p291015410334"><a name="p291015410334"></a><a name="p291015410334"></a><a href="exp2f.md">exp2f</a></p>
</td>
<td class="cellrowborder" valign="top" width="60%" headers="mcps1.2.3.1.2 "><p id="p6910135412337"><a name="p6910135412337"></a><a name="p6910135412337"></a>指定输入x，获取2的x次方。</p>
</td>
</tr>
<tr id="row32458481278"><td class="cellrowborder" valign="top" width="40%" headers="mcps1.2.3.1.1 "><p id="p1324524814720"><a name="p1324524814720"></a><a name="p1324524814720"></a><a href="hexp2.md">hexp2</a></p>
</td>
<td class="cellrowborder" valign="top" width="60%" headers="mcps1.2.3.1.2 "><p id="p192458482718"><a name="p192458482718"></a><a name="p192458482718"></a>指定输入x，获取2的x次方。</p>
</td>
</tr>
<tr id="row874513509619"><td class="cellrowborder" valign="top" width="40%" headers="mcps1.2.3.1.1 "><p id="p147455509616"><a name="p147455509616"></a><a name="p147455509616"></a><a href="h2exp2.md">h2exp2</a></p>
</td>
<td class="cellrowborder" valign="top" width="60%" headers="mcps1.2.3.1.2 "><p id="p2745145016616"><a name="p2745145016616"></a><a name="p2745145016616"></a>指定输入x，对x的各元素，获取2的该元素次方。</p>
</td>
</tr>
<tr id="row15910115416337"><td class="cellrowborder" valign="top" width="40%" headers="mcps1.2.3.1.1 "><p id="p139106548332"><a name="p139106548332"></a><a name="p139106548332"></a><a href="exp10f.md">exp10f</a></p>
</td>
<td class="cellrowborder" valign="top" width="60%" headers="mcps1.2.3.1.2 "><p id="p1910165423315"><a name="p1910165423315"></a><a name="p1910165423315"></a>指定输入x，获取10的x次方。</p>
</td>
</tr>
<tr id="row11402831381"><td class="cellrowborder" valign="top" width="40%" headers="mcps1.2.3.1.1 "><p id="p940217311820"><a name="p940217311820"></a><a name="p940217311820"></a><a href="hexp10.md">hexp10</a></p>
</td>
<td class="cellrowborder" valign="top" width="60%" headers="mcps1.2.3.1.2 "><p id="p164021231786"><a name="p164021231786"></a><a name="p164021231786"></a>指定输入x，获取10的x次方。</p>
</td>
</tr>
<tr id="row19569132111514"><td class="cellrowborder" valign="top" width="40%" headers="mcps1.2.3.1.1 "><p id="p0569152118154"><a name="p0569152118154"></a><a name="p0569152118154"></a><a href="h2exp10.md">h2exp10</a></p>
</td>
<td class="cellrowborder" valign="top" width="60%" headers="mcps1.2.3.1.2 "><p id="p356912219157"><a name="p356912219157"></a><a name="p356912219157"></a>指定输入x，对x的各元素，获取10的该元素次方。</p>
</td>
</tr>
<tr id="row991095413335"><td class="cellrowborder" valign="top" width="40%" headers="mcps1.2.3.1.1 "><p id="p291012547334"><a name="p291012547334"></a><a name="p291012547334"></a><a href="expm1f.md">expm1f</a></p>
</td>
<td class="cellrowborder" valign="top" width="60%" headers="mcps1.2.3.1.2 "><p id="p1191005413310"><a name="p1191005413310"></a><a name="p1191005413310"></a>指定输入x，获取e的x次方减1。</p>
</td>
</tr>
<tr id="row191019548335"><td class="cellrowborder" valign="top" width="40%" headers="mcps1.2.3.1.1 "><p id="p991035418339"><a name="p991035418339"></a><a name="p991035418339"></a><a href="logf.md">logf</a></p>
</td>
<td class="cellrowborder" valign="top" width="60%" headers="mcps1.2.3.1.2 "><p id="p11910205418339"><a name="p11910205418339"></a><a name="p11910205418339"></a>获取以e为底，输入数据的对数。</p>
</td>
</tr>
<tr id="row138171718819"><td class="cellrowborder" valign="top" width="40%" headers="mcps1.2.3.1.1 "><p id="p73911717819"><a name="p73911717819"></a><a name="p73911717819"></a><a href="hlog.md">hlog</a></p>
</td>
<td class="cellrowborder" valign="top" width="60%" headers="mcps1.2.3.1.2 "><p id="p6394174814"><a name="p6394174814"></a><a name="p6394174814"></a>获取以e为底，输入数据的对数。</p>
</td>
</tr>
<tr id="row82661656111513"><td class="cellrowborder" valign="top" width="40%" headers="mcps1.2.3.1.1 "><p id="p626615631511"><a name="p626615631511"></a><a name="p626615631511"></a><a href="h2log.md">h2log</a></p>
</td>
<td class="cellrowborder" valign="top" width="60%" headers="mcps1.2.3.1.2 "><p id="p18266185621513"><a name="p18266185621513"></a><a name="p18266185621513"></a>获取以e为底，输入数据各元素的对数。</p>
</td>
</tr>
<tr id="row109109545337"><td class="cellrowborder" valign="top" width="40%" headers="mcps1.2.3.1.1 "><p id="p1291095410333"><a name="p1291095410333"></a><a name="p1291095410333"></a><a href="log2f.md">log2f</a></p>
</td>
<td class="cellrowborder" valign="top" width="60%" headers="mcps1.2.3.1.2 "><p id="p179101547334"><a name="p179101547334"></a><a name="p179101547334"></a>获取以2为底，输入数据的对数。</p>
</td>
</tr>
<tr id="row15716025186"><td class="cellrowborder" valign="top" width="40%" headers="mcps1.2.3.1.1 "><p id="p171619255814"><a name="p171619255814"></a><a name="p171619255814"></a><a href="hlog2.md">hlog2</a></p>
</td>
<td class="cellrowborder" valign="top" width="60%" headers="mcps1.2.3.1.2 "><p id="p117162259814"><a name="p117162259814"></a><a name="p117162259814"></a>获取以2为底，输入数据的对数。</p>
</td>
</tr>
<tr id="row5476192010166"><td class="cellrowborder" valign="top" width="40%" headers="mcps1.2.3.1.1 "><p id="p34761920131611"><a name="p34761920131611"></a><a name="p34761920131611"></a><a href="h2log2.md">h2log2</a></p>
</td>
<td class="cellrowborder" valign="top" width="60%" headers="mcps1.2.3.1.2 "><p id="p194762206163"><a name="p194762206163"></a><a name="p194762206163"></a>获取以2为底，输入数据各元素的对数。</p>
</td>
</tr>
<tr id="row1791035443313"><td class="cellrowborder" valign="top" width="40%" headers="mcps1.2.3.1.1 "><p id="p1091015411331"><a name="p1091015411331"></a><a name="p1091015411331"></a><a href="log10f.md">log10f</a></p>
</td>
<td class="cellrowborder" valign="top" width="60%" headers="mcps1.2.3.1.2 "><p id="p17910145463312"><a name="p17910145463312"></a><a name="p17910145463312"></a>获取以10为底，输入数据的对数。</p>
</td>
</tr>
<tr id="row101890358817"><td class="cellrowborder" valign="top" width="40%" headers="mcps1.2.3.1.1 "><p id="p1819014355815"><a name="p1819014355815"></a><a name="p1819014355815"></a><a href="hlog10.md">hlog10</a></p>
</td>
<td class="cellrowborder" valign="top" width="60%" headers="mcps1.2.3.1.2 "><p id="p9748104713819"><a name="p9748104713819"></a><a name="p9748104713819"></a>获取以10为底，输入数据的对数。</p>
</td>
</tr>
<tr id="row13859105101618"><td class="cellrowborder" valign="top" width="40%" headers="mcps1.2.3.1.1 "><p id="p118592517166"><a name="p118592517166"></a><a name="p118592517166"></a><a href="h2log10.md">h2log10</a></p>
</td>
<td class="cellrowborder" valign="top" width="60%" headers="mcps1.2.3.1.2 "><p id="p485945171614"><a name="p485945171614"></a><a name="p485945171614"></a>获取以10为底，输入数据各元素的对数。</p>
</td>
</tr>
<tr id="row1391045415335"><td class="cellrowborder" valign="top" width="40%" headers="mcps1.2.3.1.1 "><p id="p1911185411339"><a name="p1911185411339"></a><a name="p1911185411339"></a><a href="log1pf.md">log1pf</a></p>
</td>
<td class="cellrowborder" valign="top" width="60%" headers="mcps1.2.3.1.2 "><p id="p291195411335"><a name="p291195411335"></a><a name="p291195411335"></a>获取以e为底，输入数据加1的对数。</p>
</td>
</tr>
<tr id="row4911105423314"><td class="cellrowborder" valign="top" width="40%" headers="mcps1.2.3.1.1 "><p id="p1991185473313"><a name="p1991185473313"></a><a name="p1991185473313"></a><a href="logbf.md">logbf</a></p>
</td>
<td class="cellrowborder" valign="top" width="60%" headers="mcps1.2.3.1.2 "><p id="p129111454183318"><a name="p129111454183318"></a><a name="p129111454183318"></a>计算以2为底，输入数据的对数，并对结果向下取整，返回浮点数。</p>
</td>
</tr>
<tr id="row1491135415337"><td class="cellrowborder" valign="top" width="40%" headers="mcps1.2.3.1.1 "><p id="p12911145419333"><a name="p12911145419333"></a><a name="p12911145419333"></a><a href="ilogbf.md">ilogbf</a></p>
</td>
<td class="cellrowborder" valign="top" width="60%" headers="mcps1.2.3.1.2 "><p id="p39111654113314"><a name="p39111654113314"></a><a name="p39111654113314"></a>计算以2为底，输入数据的对数，并对结果向下取整，返回整数。</p>
</td>
</tr>
<tr id="row29111754153318"><td class="cellrowborder" valign="top" width="40%" headers="mcps1.2.3.1.1 "><p id="p891175410338"><a name="p891175410338"></a><a name="p891175410338"></a><a href="cosf.md">cosf</a></p>
</td>
<td class="cellrowborder" valign="top" width="60%" headers="mcps1.2.3.1.2 "><p id="p791135473319"><a name="p791135473319"></a><a name="p791135473319"></a>获取输入数据的三角函数余弦值。</p>
</td>
</tr>
<tr id="row67461455384"><td class="cellrowborder" valign="top" width="40%" headers="mcps1.2.3.1.1 "><p id="p374614555819"><a name="p374614555819"></a><a name="p374614555819"></a><a href="hcos.md">hcos</a></p>
</td>
<td class="cellrowborder" valign="top" width="60%" headers="mcps1.2.3.1.2 "><p id="p1174615551686"><a name="p1174615551686"></a><a name="p1174615551686"></a>获取输入数据的三角函数余弦值。</p>
</td>
</tr>
<tr id="row1854724181710"><td class="cellrowborder" valign="top" width="40%" headers="mcps1.2.3.1.1 "><p id="p12854122421717"><a name="p12854122421717"></a><a name="p12854122421717"></a><a href="h2cos.md">h2cos</a></p>
</td>
<td class="cellrowborder" valign="top" width="60%" headers="mcps1.2.3.1.2 "><p id="p285442471714"><a name="p285442471714"></a><a name="p285442471714"></a>获取输入数据各元素的三角函数余弦值。</p>
</td>
</tr>
<tr id="row13911554133318"><td class="cellrowborder" valign="top" width="40%" headers="mcps1.2.3.1.1 "><p id="p10911115463319"><a name="p10911115463319"></a><a name="p10911115463319"></a><a href="coshf.md">coshf</a></p>
</td>
<td class="cellrowborder" valign="top" width="60%" headers="mcps1.2.3.1.2 "><p id="p119111545339"><a name="p119111545339"></a><a name="p119111545339"></a>获取输入数据的双曲余弦值。</p>
</td>
</tr>
<tr id="row1991165416336"><td class="cellrowborder" valign="top" width="40%" headers="mcps1.2.3.1.1 "><p id="p99113543339"><a name="p99113543339"></a><a name="p99113543339"></a><a href="cospif.md">cospif</a></p>
</td>
<td class="cellrowborder" valign="top" width="60%" headers="mcps1.2.3.1.2 "><p id="p591115420332"><a name="p591115420332"></a><a name="p591115420332"></a>获取输入数据与π相乘的余弦值。</p>
</td>
</tr>
<tr id="row17911115412338"><td class="cellrowborder" valign="top" width="40%" headers="mcps1.2.3.1.1 "><p id="p29118548332"><a name="p29118548332"></a><a name="p29118548332"></a><a href="acosf.md">acosf</a></p>
</td>
<td class="cellrowborder" valign="top" width="60%" headers="mcps1.2.3.1.2 "><p id="p1391185433312"><a name="p1391185433312"></a><a name="p1391185433312"></a>获取输入数据的反余弦值。</p>
</td>
</tr>
<tr id="row1591115540333"><td class="cellrowborder" valign="top" width="40%" headers="mcps1.2.3.1.1 "><p id="p1091185423313"><a name="p1091185423313"></a><a name="p1091185423313"></a><a href="acoshf.md">acoshf</a></p>
</td>
<td class="cellrowborder" valign="top" width="60%" headers="mcps1.2.3.1.2 "><p id="p1391115543333"><a name="p1391115543333"></a><a name="p1391115543333"></a>获取输入数据的双曲反余弦值。</p>
</td>
</tr>
<tr id="row7911185413313"><td class="cellrowborder" valign="top" width="40%" headers="mcps1.2.3.1.1 "><p id="p991117540339"><a name="p991117540339"></a><a name="p991117540339"></a><a href="sinf.md">sinf</a></p>
</td>
<td class="cellrowborder" valign="top" width="60%" headers="mcps1.2.3.1.2 "><p id="p491145483319"><a name="p491145483319"></a><a name="p491145483319"></a>获取输入数据的三角函数正弦值。</p>
</td>
</tr>
<tr id="row177714819916"><td class="cellrowborder" valign="top" width="40%" headers="mcps1.2.3.1.1 "><p id="p12772081498"><a name="p12772081498"></a><a name="p12772081498"></a><a href="hsin.md">hsin</a></p>
</td>
<td class="cellrowborder" valign="top" width="60%" headers="mcps1.2.3.1.2 "><p id="p025914298"><a name="p025914298"></a><a name="p025914298"></a>获取输入数据的三角函数正弦值。</p>
</td>
</tr>
<tr id="row157924881716"><td class="cellrowborder" valign="top" width="40%" headers="mcps1.2.3.1.1 "><p id="p757913480179"><a name="p757913480179"></a><a name="p757913480179"></a><a href="h2sin.md">h2sin</a></p>
</td>
<td class="cellrowborder" valign="top" width="60%" headers="mcps1.2.3.1.2 "><p id="p115807481174"><a name="p115807481174"></a><a name="p115807481174"></a>获取输入数据各元素的三角函数正弦值。</p>
</td>
</tr>
<tr id="row1291116544339"><td class="cellrowborder" valign="top" width="40%" headers="mcps1.2.3.1.1 "><p id="p39111854153318"><a name="p39111854153318"></a><a name="p39111854153318"></a><a href="sinhf.md">sinhf</a></p>
</td>
<td class="cellrowborder" valign="top" width="60%" headers="mcps1.2.3.1.2 "><p id="p10911854183317"><a name="p10911854183317"></a><a name="p10911854183317"></a>获取输入数据的双曲正弦值。</p>
</td>
</tr>
<tr id="row2911155443313"><td class="cellrowborder" valign="top" width="40%" headers="mcps1.2.3.1.1 "><p id="p59115544339"><a name="p59115544339"></a><a name="p59115544339"></a><a href="sinpif.md">sinpif</a></p>
</td>
<td class="cellrowborder" valign="top" width="60%" headers="mcps1.2.3.1.2 "><p id="p1991275410335"><a name="p1991275410335"></a><a name="p1991275410335"></a>获取输入数据与π相乘的正弦值。</p>
</td>
</tr>
<tr id="row17912145413335"><td class="cellrowborder" valign="top" width="40%" headers="mcps1.2.3.1.1 "><p id="p169121654143312"><a name="p169121654143312"></a><a name="p169121654143312"></a><a href="asinf.md">asinf</a></p>
</td>
<td class="cellrowborder" valign="top" width="60%" headers="mcps1.2.3.1.2 "><p id="p15912135433312"><a name="p15912135433312"></a><a name="p15912135433312"></a>获取输入数据的反正弦值。</p>
</td>
</tr>
<tr id="row391275418334"><td class="cellrowborder" valign="top" width="40%" headers="mcps1.2.3.1.1 "><p id="p169121054133310"><a name="p169121054133310"></a><a name="p169121054133310"></a><a href="asinhf.md">asinhf</a></p>
</td>
<td class="cellrowborder" valign="top" width="60%" headers="mcps1.2.3.1.2 "><p id="p1091215453318"><a name="p1091215453318"></a><a name="p1091215453318"></a>获取输入数据的双曲反正弦值。</p>
</td>
</tr>
<tr id="row2912454113313"><td class="cellrowborder" valign="top" width="40%" headers="mcps1.2.3.1.1 "><p id="p169124541336"><a name="p169124541336"></a><a name="p169124541336"></a><a href="sincosf.md">sincosf</a></p>
</td>
<td class="cellrowborder" valign="top" width="60%" headers="mcps1.2.3.1.2 "><p id="p591275412338"><a name="p591275412338"></a><a name="p591275412338"></a>获取输入数据的三角函数正弦值和余弦值。</p>
</td>
</tr>
<tr id="row1191219541337"><td class="cellrowborder" valign="top" width="40%" headers="mcps1.2.3.1.1 "><p id="p09121354123318"><a name="p09121354123318"></a><a name="p09121354123318"></a><a href="sincospif.md">sincospif</a></p>
</td>
<td class="cellrowborder" valign="top" width="60%" headers="mcps1.2.3.1.2 "><p id="p14912135416337"><a name="p14912135416337"></a><a name="p14912135416337"></a>获取输入数据与π相乘的三角函数正弦值和余弦值。</p>
</td>
</tr>
<tr id="row4912054123311"><td class="cellrowborder" valign="top" width="40%" headers="mcps1.2.3.1.1 "><p id="p991225416331"><a name="p991225416331"></a><a name="p991225416331"></a><a href="frexpf.md">frexpf</a></p>
</td>
<td class="cellrowborder" valign="top" width="60%" headers="mcps1.2.3.1.2 "><p id="p59121254153318"><a name="p59121254153318"></a><a name="p59121254153318"></a>将x转换为归一化[1/2, 1)的有符号数乘以2的积分幂。</p>
</td>
</tr>
<tr id="row591225420337"><td class="cellrowborder" valign="top" width="40%" headers="mcps1.2.3.1.1 "><p id="p491215414331"><a name="p491215414331"></a><a name="p491215414331"></a><a href="ldexpf.md">ldexpf</a></p>
</td>
<td class="cellrowborder" valign="top" width="60%" headers="mcps1.2.3.1.2 "><p id="p1291275463317"><a name="p1291275463317"></a><a name="p1291275463317"></a>获取输入x乘以2的exp次幂的结果。</p>
</td>
</tr>
<tr id="row189127545339"><td class="cellrowborder" valign="top" width="40%" headers="mcps1.2.3.1.1 "><p id="p591275473312"><a name="p591275473312"></a><a name="p591275473312"></a><a href="sqrtf.md">sqrtf</a></p>
</td>
<td class="cellrowborder" valign="top" width="60%" headers="mcps1.2.3.1.2 "><p id="p15912125473316"><a name="p15912125473316"></a><a name="p15912125473316"></a>获取输入数据x的平方根。</p>
</td>
</tr>
<tr id="row2013118509914"><td class="cellrowborder" valign="top" width="40%" headers="mcps1.2.3.1.1 "><p id="p1813116501295"><a name="p1813116501295"></a><a name="p1813116501295"></a><a href="hsqrt.md">hsqrt</a></p>
</td>
<td class="cellrowborder" valign="top" width="60%" headers="mcps1.2.3.1.2 "><p id="p1913110508910"><a name="p1913110508910"></a><a name="p1913110508910"></a>获取输入数据x的平方根。</p>
</td>
</tr>
<tr id="row15698935161815"><td class="cellrowborder" valign="top" width="40%" headers="mcps1.2.3.1.1 "><p id="p2698935171814"><a name="p2698935171814"></a><a name="p2698935171814"></a><a href="h2sqrt.md">h2sqrt</a></p>
</td>
<td class="cellrowborder" valign="top" width="60%" headers="mcps1.2.3.1.2 "><p id="p12698183541814"><a name="p12698183541814"></a><a name="p12698183541814"></a>获取输入数据x各元素的平方根。</p>
</td>
</tr>
<tr id="row15912354193311"><td class="cellrowborder" valign="top" width="40%" headers="mcps1.2.3.1.1 "><p id="p59121054113311"><a name="p59121054113311"></a><a name="p59121054113311"></a><a href="rsqrtf.md">rsqrtf</a></p>
</td>
<td class="cellrowborder" valign="top" width="60%" headers="mcps1.2.3.1.2 "><p id="p2091210546339"><a name="p2091210546339"></a><a name="p2091210546339"></a>获取输入数据x的平方根的倒数。</p>
</td>
</tr>
<tr id="row155815911920"><td class="cellrowborder" valign="top" width="40%" headers="mcps1.2.3.1.1 "><p id="p85584592099"><a name="p85584592099"></a><a name="p85584592099"></a><a href="hrsqrt.md">hrsqrt</a></p>
</td>
<td class="cellrowborder" valign="top" width="60%" headers="mcps1.2.3.1.2 "><p id="p165581593917"><a name="p165581593917"></a><a name="p165581593917"></a>获取输入数据x的平方根的倒数。</p>
</td>
</tr>
<tr id="row1166825612188"><td class="cellrowborder" valign="top" width="40%" headers="mcps1.2.3.1.1 "><p id="p56681056131817"><a name="p56681056131817"></a><a name="p56681056131817"></a><a href="h2rsqrt.md">h2rsqrt</a></p>
</td>
<td class="cellrowborder" valign="top" width="60%" headers="mcps1.2.3.1.2 "><p id="p466818566188"><a name="p466818566188"></a><a name="p466818566188"></a>获取输入数据x各元素的平方根的倒数。</p>
</td>
</tr>
<tr id="row161542011141014"><td class="cellrowborder" valign="top" width="40%" headers="mcps1.2.3.1.1 "><p id="p1715461131015"><a name="p1715461131015"></a><a name="p1715461131015"></a><a href="hrcp.md">hrcp</a></p>
</td>
<td class="cellrowborder" valign="top" width="60%" headers="mcps1.2.3.1.2 "><p id="p278471781014"><a name="p278471781014"></a><a name="p278471781014"></a>获取输入数据x的倒数。</p>
</td>
</tr>
<tr id="row15749172251912"><td class="cellrowborder" valign="top" width="40%" headers="mcps1.2.3.1.1 "><p id="p1274952271918"><a name="p1274952271918"></a><a name="p1274952271918"></a><a href="h2rcp.md">h2rcp</a></p>
</td>
<td class="cellrowborder" valign="top" width="60%" headers="mcps1.2.3.1.2 "><p id="p4749122251920"><a name="p4749122251920"></a><a name="p4749122251920"></a>获取输入数据x各元素的倒数。</p>
</td>
</tr>
<tr id="row39129540339"><td class="cellrowborder" valign="top" width="40%" headers="mcps1.2.3.1.1 "><p id="p17912195414335"><a name="p17912195414335"></a><a name="p17912195414335"></a><a href="hypotf.md">hypotf</a></p>
</td>
<td class="cellrowborder" valign="top" width="60%" headers="mcps1.2.3.1.2 "><p id="p69122549339"><a name="p69122549339"></a><a name="p69122549339"></a>获取输入数据x、y的平方和x^2 + y^2的平方根。</p>
</td>
</tr>
<tr id="row179123547336"><td class="cellrowborder" valign="top" width="40%" headers="mcps1.2.3.1.1 "><p id="p291215414337"><a name="p291215414337"></a><a name="p291215414337"></a><a href="rhypotf.md">rhypotf</a></p>
</td>
<td class="cellrowborder" valign="top" width="60%" headers="mcps1.2.3.1.2 "><p id="p13912175411334"><a name="p13912175411334"></a><a name="p13912175411334"></a>获取输入数据x、y的平方和x^2 + y^2的平方根的倒数。</p>
</td>
</tr>
<tr id="row209131354203318"><td class="cellrowborder" valign="top" width="40%" headers="mcps1.2.3.1.1 "><p id="p189135546330"><a name="p189135546330"></a><a name="p189135546330"></a><a href="powf.md">powf</a></p>
</td>
<td class="cellrowborder" valign="top" width="60%" headers="mcps1.2.3.1.2 "><p id="p89131548338"><a name="p89131548338"></a><a name="p89131548338"></a>获取输入数据x的y次幂。</p>
</td>
</tr>
<tr id="row1691385403315"><td class="cellrowborder" valign="top" width="40%" headers="mcps1.2.3.1.1 "><p id="p1991316546338"><a name="p1991316546338"></a><a name="p1991316546338"></a><a href="norm3df.md">norm3df</a></p>
</td>
<td class="cellrowborder" valign="top" width="60%" headers="mcps1.2.3.1.2 "><p id="p6913165483312"><a name="p6913165483312"></a><a name="p6913165483312"></a>获取输入数据a、b、c的平方和a^2 + b^2 + c^2的平方根。</p>
</td>
</tr>
<tr id="row189134549332"><td class="cellrowborder" valign="top" width="40%" headers="mcps1.2.3.1.1 "><p id="p149135547337"><a name="p149135547337"></a><a name="p149135547337"></a><a href="rnorm3df.md">rnorm3df</a></p>
</td>
<td class="cellrowborder" valign="top" width="60%" headers="mcps1.2.3.1.2 "><p id="p1591315547332"><a name="p1591315547332"></a><a name="p1591315547332"></a>获取输入数据a、b、c的平方和a^2 + b^2 + c^2的平方根的倒数。</p>
</td>
</tr>
<tr id="row6913754103312"><td class="cellrowborder" valign="top" width="40%" headers="mcps1.2.3.1.1 "><p id="p79133542332"><a name="p79133542332"></a><a name="p79133542332"></a><a href="norm4df.md">norm4df</a></p>
</td>
<td class="cellrowborder" valign="top" width="60%" headers="mcps1.2.3.1.2 "><p id="p159133545335"><a name="p159133545335"></a><a name="p159133545335"></a>获取输入数据a、b、c、d的平方和a^2 + b^2+ c^2+ d^2的平方根。</p>
</td>
</tr>
<tr id="row2913155414331"><td class="cellrowborder" valign="top" width="40%" headers="mcps1.2.3.1.1 "><p id="p109137543339"><a name="p109137543339"></a><a name="p109137543339"></a><a href="rnorm4df.md">rnorm4df</a></p>
</td>
<td class="cellrowborder" valign="top" width="60%" headers="mcps1.2.3.1.2 "><p id="p2091305415337"><a name="p2091305415337"></a><a name="p2091305415337"></a>获取输入数据a、b、c、d的平方和a^2 + b^2 + c^2 + d^2的平方根的倒数。</p>
</td>
</tr>
<tr id="row49134540337"><td class="cellrowborder" valign="top" width="40%" headers="mcps1.2.3.1.1 "><p id="p791335412331"><a name="p791335412331"></a><a name="p791335412331"></a><a href="normf.md">normf</a></p>
</td>
<td class="cellrowborder" valign="top" width="60%" headers="mcps1.2.3.1.2 "><p id="p89131754173312"><a name="p89131754173312"></a><a name="p89131754173312"></a>获取输入数据a中前n个元素的平方和a[0]^2 + a[1]^2 +...+ a[n-1]^2的平方根。</p>
</td>
</tr>
<tr id="row9913195423314"><td class="cellrowborder" valign="top" width="40%" headers="mcps1.2.3.1.1 "><p id="p791375413314"><a name="p791375413314"></a><a name="p791375413314"></a><a href="rnormf.md">rnormf</a></p>
</td>
<td class="cellrowborder" valign="top" width="60%" headers="mcps1.2.3.1.2 "><p id="p20913154183315"><a name="p20913154183315"></a><a name="p20913154183315"></a>获取输入数据a中前n个元素的平方和a[0]^2 + a[1]^2 + ...+ a[n-1]^2的平方根的倒数。</p>
</td>
</tr>
<tr id="row19131154153316"><td class="cellrowborder" valign="top" width="40%" headers="mcps1.2.3.1.1 "><p id="p149136545335"><a name="p149136545335"></a><a name="p149136545335"></a><a href="cbrtf.md">cbrtf</a></p>
</td>
<td class="cellrowborder" valign="top" width="60%" headers="mcps1.2.3.1.2 "><p id="p69134548337"><a name="p69134548337"></a><a name="p69134548337"></a>获取输入数据x的立方根。</p>
</td>
</tr>
<tr id="row69144542337"><td class="cellrowborder" valign="top" width="40%" headers="mcps1.2.3.1.1 "><p id="p79141854143319"><a name="p79141854143319"></a><a name="p79141854143319"></a><a href="rcbrtf.md">rcbrtf</a></p>
</td>
<td class="cellrowborder" valign="top" width="60%" headers="mcps1.2.3.1.2 "><p id="p5914195423310"><a name="p5914195423310"></a><a name="p5914195423310"></a>获取输入数据x的立方根的倒数。</p>
</td>
</tr>
<tr id="row1191415547334"><td class="cellrowborder" valign="top" width="40%" headers="mcps1.2.3.1.1 "><p id="p0914185463320"><a name="p0914185463320"></a><a name="p0914185463320"></a><a href="erff.md">erff</a></p>
</td>
<td class="cellrowborder" valign="top" width="60%" headers="mcps1.2.3.1.2 "><p id="p791420542339"><a name="p791420542339"></a><a name="p791420542339"></a>获取输入数据的误差函数值。</p>
</td>
</tr>
<tr id="row16914185493319"><td class="cellrowborder" valign="top" width="40%" headers="mcps1.2.3.1.1 "><p id="p12914125411339"><a name="p12914125411339"></a><a name="p12914125411339"></a><a href="erfcf.md">erfcf</a></p>
</td>
<td class="cellrowborder" valign="top" width="60%" headers="mcps1.2.3.1.2 "><p id="p1091465419339"><a name="p1091465419339"></a><a name="p1091465419339"></a>获取输入数据的互补误差函数值。</p>
</td>
</tr>
<tr id="row1491419541338"><td class="cellrowborder" valign="top" width="40%" headers="mcps1.2.3.1.1 "><p id="p2091415414331"><a name="p2091415414331"></a><a name="p2091415414331"></a><a href="erfinvf.md">erfinvf</a></p>
</td>
<td class="cellrowborder" valign="top" width="60%" headers="mcps1.2.3.1.2 "><p id="p10914115423319"><a name="p10914115423319"></a><a name="p10914115423319"></a>获取输入数据的逆误差函数值。</p>
</td>
</tr>
<tr id="row491495463319"><td class="cellrowborder" valign="top" width="40%" headers="mcps1.2.3.1.1 "><p id="p1691419544339"><a name="p1691419544339"></a><a name="p1691419544339"></a><a href="erfcinvf.md">erfcinvf</a></p>
</td>
<td class="cellrowborder" valign="top" width="60%" headers="mcps1.2.3.1.2 "><p id="p9914155411339"><a name="p9914155411339"></a><a name="p9914155411339"></a>获取输入数据的逆互补误差函数值。</p>
</td>
</tr>
<tr id="row12914115414339"><td class="cellrowborder" valign="top" width="40%" headers="mcps1.2.3.1.1 "><p id="p7914145493311"><a name="p7914145493311"></a><a name="p7914145493311"></a><a href="erfcxf.md">erfcxf</a></p>
</td>
<td class="cellrowborder" valign="top" width="60%" headers="mcps1.2.3.1.2 "><p id="p191413543333"><a name="p191413543333"></a><a name="p191413543333"></a>获取输入数据的缩放互补误差函数值。</p>
</td>
</tr>
<tr id="row199147542338"><td class="cellrowborder" valign="top" width="40%" headers="mcps1.2.3.1.1 "><p id="p1691414548339"><a name="p1691414548339"></a><a name="p1691414548339"></a><a href="tgammaf.md">tgammaf</a></p>
</td>
<td class="cellrowborder" valign="top" width="60%" headers="mcps1.2.3.1.2 "><p id="p1091413548331"><a name="p1091413548331"></a><a name="p1091413548331"></a>获取输入数据x的伽马函数值。</p>
</td>
</tr>
<tr id="row3914254133318"><td class="cellrowborder" valign="top" width="40%" headers="mcps1.2.3.1.1 "><p id="p7915135493311"><a name="p7915135493311"></a><a name="p7915135493311"></a><a href="lgammaf.md">lgammaf</a></p>
</td>
<td class="cellrowborder" valign="top" width="60%" headers="mcps1.2.3.1.2 "><p id="p3915754143313"><a name="p3915754143313"></a><a name="p3915754143313"></a>获取输入数据x伽马值的绝对值并求自然对数。</p>
</td>
</tr>
<tr id="row591520548334"><td class="cellrowborder" valign="top" width="40%" headers="mcps1.2.3.1.1 "><p id="p1891595412338"><a name="p1891595412338"></a><a name="p1891595412338"></a><a href="cyl_bessel_i0f.md">cyl_bessel_i0f</a></p>
</td>
<td class="cellrowborder" valign="top" width="60%" headers="mcps1.2.3.1.2 "><p id="p20915054133314"><a name="p20915054133314"></a><a name="p20915054133314"></a>获取输入数据x的0阶常规修正圆柱贝塞尔函数的值。</p>
</td>
</tr>
<tr id="row59153548339"><td class="cellrowborder" valign="top" width="40%" headers="mcps1.2.3.1.1 "><p id="p091514544334"><a name="p091514544334"></a><a name="p091514544334"></a><a href="cyl_bessel_i1f.md">cyl_bessel_i1f</a></p>
</td>
<td class="cellrowborder" valign="top" width="60%" headers="mcps1.2.3.1.2 "><p id="p1915145410334"><a name="p1915145410334"></a><a name="p1915145410334"></a>获取输入数据x的1阶常规修正圆柱贝塞尔函数的值。</p>
</td>
</tr>
<tr id="row691512548334"><td class="cellrowborder" valign="top" width="40%" headers="mcps1.2.3.1.1 "><p id="p0915125413319"><a name="p0915125413319"></a><a name="p0915125413319"></a><a href="normcdff.md">normcdff</a></p>
</td>
<td class="cellrowborder" valign="top" width="60%" headers="mcps1.2.3.1.2 "><p id="p119151454103312"><a name="p119151454103312"></a><a name="p119151454103312"></a>获取输入数据x的标准正态分布的累积分布函数值。</p>
</td>
</tr>
<tr id="row20349141123420"><td class="cellrowborder" valign="top" width="40%" headers="mcps1.2.3.1.1 "><p id="p10301542341"><a name="p10301542341"></a><a name="p10301542341"></a><a href="normcdfinvf.md">normcdfinvf</a></p>
</td>
<td class="cellrowborder" valign="top" width="60%" headers="mcps1.2.3.1.2 "><p id="p15301346347"><a name="p15301346347"></a><a name="p15301346347"></a>获取输入数据x的标准正态累积分布的逆函数</p>
</td>
</tr>
<tr id="row16915105413334"><td class="cellrowborder" valign="top" width="40%" headers="mcps1.2.3.1.1 "><p id="p159151549334"><a name="p159151549334"></a><a name="p159151549334"></a><a href="j0f.md">j0f</a></p>
</td>
<td class="cellrowborder" valign="top" width="60%" headers="mcps1.2.3.1.2 "><p id="p891555443311"><a name="p891555443311"></a><a name="p891555443311"></a>获取输入数据x的0阶第一类贝塞尔函数j0的值。</p>
</td>
</tr>
<tr id="row1191545463312"><td class="cellrowborder" valign="top" width="40%" headers="mcps1.2.3.1.1 "><p id="p19151554153310"><a name="p19151554153310"></a><a name="p19151554153310"></a><a href="j1f.md">j1f</a></p>
</td>
<td class="cellrowborder" valign="top" width="60%" headers="mcps1.2.3.1.2 "><p id="p1891595418336"><a name="p1891595418336"></a><a name="p1891595418336"></a>获取输入数据x的1阶第一类贝塞尔函数j1的值。</p>
</td>
</tr>
<tr id="row99153549332"><td class="cellrowborder" valign="top" width="40%" headers="mcps1.2.3.1.1 "><p id="p2091595483312"><a name="p2091595483312"></a><a name="p2091595483312"></a><a href="jnf.md">jnf</a></p>
</td>
<td class="cellrowborder" valign="top" width="60%" headers="mcps1.2.3.1.2 "><p id="p12915454173316"><a name="p12915454173316"></a><a name="p12915454173316"></a>获取输入数据x的n阶第一类贝塞尔函数jn的值。</p>
</td>
</tr>
<tr id="row1191565416337"><td class="cellrowborder" valign="top" width="40%" headers="mcps1.2.3.1.1 "><p id="p10915185414338"><a name="p10915185414338"></a><a name="p10915185414338"></a><a href="y0f.md">y0f</a></p>
</td>
<td class="cellrowborder" valign="top" width="60%" headers="mcps1.2.3.1.2 "><p id="p991514545332"><a name="p991514545332"></a><a name="p991514545332"></a>获取输入数据x的0阶第二类贝塞尔函数y0的值。</p>
</td>
</tr>
<tr id="row8915185420338"><td class="cellrowborder" valign="top" width="40%" headers="mcps1.2.3.1.1 "><p id="p29151854203311"><a name="p29151854203311"></a><a name="p29151854203311"></a><a href="y1f.md">y1f</a></p>
</td>
<td class="cellrowborder" valign="top" width="60%" headers="mcps1.2.3.1.2 "><p id="p13915155483317"><a name="p13915155483317"></a><a name="p13915155483317"></a>获取输入数据x的1阶第二类贝塞尔函数y1的值。</p>
</td>
</tr>
<tr id="row491595483318"><td class="cellrowborder" valign="top" width="40%" headers="mcps1.2.3.1.1 "><p id="p19915165418337"><a name="p19915165418337"></a><a name="p19915165418337"></a><a href="ynf.md">ynf</a></p>
</td>
<td class="cellrowborder" valign="top" width="60%" headers="mcps1.2.3.1.2 "><p id="p189151754133313"><a name="p189151754133313"></a><a name="p189151754133313"></a>获取输入数据x的n阶第二类贝塞尔函数yn的值。</p>
</td>
</tr>
<tr id="row5915354113314"><td class="cellrowborder" valign="top" width="40%" headers="mcps1.2.3.1.1 "><p id="p14916175416338"><a name="p14916175416338"></a><a name="p14916175416338"></a><a href="fabsf.md">fabsf</a></p>
</td>
<td class="cellrowborder" valign="top" width="60%" headers="mcps1.2.3.1.2 "><p id="p119161754193319"><a name="p119161754193319"></a><a name="p119161754193319"></a>获取输入数据的绝对值。</p>
</td>
</tr>
<tr id="row1345410171828"><td class="cellrowborder" valign="top" width="40%" headers="mcps1.2.3.1.1 "><p id="p1454617720"><a name="p1454617720"></a><a name="p1454617720"></a><a href="__habs.md">__habs</a></p>
</td>
<td class="cellrowborder" valign="top" width="60%" headers="mcps1.2.3.1.2 "><p id="p134549171823"><a name="p134549171823"></a><a name="p134549171823"></a>获取输入数据的绝对值。</p>
</td>
</tr>
<tr id="row189161954123314"><td class="cellrowborder" valign="top" width="40%" headers="mcps1.2.3.1.1 "><p id="p991615546339"><a name="p991615546339"></a><a name="p991615546339"></a><a href="fmaf.md">fmaf</a></p>
</td>
<td class="cellrowborder" valign="top" width="60%" headers="mcps1.2.3.1.2 "><p id="p3916105413333"><a name="p3916105413333"></a><a name="p3916105413333"></a>对输入数据x、y、z，计算x与y相乘加上z的结果。</p>
</td>
</tr>
<tr id="row189014232040"><td class="cellrowborder" valign="top" width="40%" headers="mcps1.2.3.1.1 "><p id="p99017231749"><a name="p99017231749"></a><a name="p99017231749"></a><a href="__hfma.md">__hfma</a></p>
</td>
<td class="cellrowborder" valign="top" width="60%" headers="mcps1.2.3.1.2 "><p id="p390152312411"><a name="p390152312411"></a><a name="p390152312411"></a>对输入数据x、y、z，计算x与y相乘加上z的结果。</p>
</td>
</tr>
<tr id="row1891611541336"><td class="cellrowborder" valign="top" width="40%" headers="mcps1.2.3.1.1 "><p id="p2916115413339"><a name="p2916115413339"></a><a name="p2916115413339"></a><a href="fmaxf.md">fmaxf</a></p>
</td>
<td class="cellrowborder" valign="top" width="60%" headers="mcps1.2.3.1.2 "><p id="p091616545335"><a name="p091616545335"></a><a name="p091616545335"></a>获取两个输入数据中的最大值。</p>
</td>
</tr>
<tr id="row311020351658"><td class="cellrowborder" valign="top" width="40%" headers="mcps1.2.3.1.1 "><p id="p81110351658"><a name="p81110351658"></a><a name="p81110351658"></a><a href="__hmax.md">__hmax</a></p>
</td>
<td class="cellrowborder" valign="top" width="60%" headers="mcps1.2.3.1.2 "><p id="p16734242758"><a name="p16734242758"></a><a name="p16734242758"></a>获取两个输入数据中的最大值。</p>
</td>
</tr>
<tr id="row139161854153314"><td class="cellrowborder" valign="top" width="40%" headers="mcps1.2.3.1.1 "><p id="p291645411338"><a name="p291645411338"></a><a name="p291645411338"></a><a href="fminf.md">fminf</a></p>
</td>
<td class="cellrowborder" valign="top" width="60%" headers="mcps1.2.3.1.2 "><p id="p891675463312"><a name="p891675463312"></a><a name="p891675463312"></a>获取两个输入数据中的最小值。</p>
</td>
</tr>
<tr id="row1458293819518"><td class="cellrowborder" valign="top" width="40%" headers="mcps1.2.3.1.1 "><p id="p1658211381353"><a name="p1658211381353"></a><a name="p1658211381353"></a><a href="__hmin.md">__hmin</a></p>
</td>
<td class="cellrowborder" valign="top" width="60%" headers="mcps1.2.3.1.2 "><p id="p7582438957"><a name="p7582438957"></a><a name="p7582438957"></a>获取两个输入数据中的最小值。</p>
</td>
</tr>
<tr id="row15916454203311"><td class="cellrowborder" valign="top" width="40%" headers="mcps1.2.3.1.1 "><p id="p4916115411332"><a name="p4916115411332"></a><a name="p4916115411332"></a><a href="fdimf.md">fdimf</a></p>
</td>
<td class="cellrowborder" valign="top" width="60%" headers="mcps1.2.3.1.2 "><p id="p13916125433311"><a name="p13916125433311"></a><a name="p13916125433311"></a>获取输入数据的差值，差值小于0时，返回0。</p>
</td>
</tr>
<tr id="row119165548334"><td class="cellrowborder" valign="top" width="40%" headers="mcps1.2.3.1.1 "><p id="p12916195415332"><a name="p12916195415332"></a><a name="p12916195415332"></a><a href="remquof.md">remquof</a></p>
</td>
<td class="cellrowborder" valign="top" width="60%" headers="mcps1.2.3.1.2 "><p id="p1791695411333"><a name="p1791695411333"></a><a name="p1791695411333"></a>获取输入数据x除以y的余数。求余数时，商取最接近x除以y浮点数结果的整数，当x除以y的浮点数结果与左右最接近的整数距离相等时，商取偶数，同时将商赋值给指针变量quo。</p>
</td>
</tr>
<tr id="row13916254203313"><td class="cellrowborder" valign="top" width="40%" headers="mcps1.2.3.1.1 "><p id="p14916254133316"><a name="p14916254133316"></a><a name="p14916254133316"></a><a href="fmodf.md">fmodf</a></p>
</td>
<td class="cellrowborder" valign="top" width="60%" headers="mcps1.2.3.1.2 "><p id="p491615453316"><a name="p491615453316"></a><a name="p491615453316"></a>获取输入数据x除以y的余数。求余数时，商取x除以y浮点数结果的整数部分。</p>
</td>
</tr>
<tr id="row1791614546337"><td class="cellrowborder" valign="top" width="40%" headers="mcps1.2.3.1.1 "><p id="p139161541336"><a name="p139161541336"></a><a name="p139161541336"></a><a href="remainderf.md">remainderf</a></p>
</td>
<td class="cellrowborder" valign="top" width="60%" headers="mcps1.2.3.1.2 "><p id="p5916185418337"><a name="p5916185418337"></a><a name="p5916185418337"></a>获取输入数据x除以y的余数。求余数时，商取最接近x除以y浮点数结果的整数，当x除以y的浮点数结果与左右最接近的整数距离相等时，商取偶数。</p>
</td>
</tr>
<tr id="row091635423313"><td class="cellrowborder" valign="top" width="40%" headers="mcps1.2.3.1.1 "><p id="p39166542332"><a name="p39166542332"></a><a name="p39166542332"></a><a href="copysignf.md">copysignf</a></p>
</td>
<td class="cellrowborder" valign="top" width="60%" headers="mcps1.2.3.1.2 "><p id="p209161154133315"><a name="p209161154133315"></a><a name="p209161154133315"></a>获取由第一个输入x的数值部分和第二个输入y的符号部分拼接得到的浮点数。</p>
</td>
</tr>
<tr id="row12916105473310"><td class="cellrowborder" valign="top" width="40%" headers="mcps1.2.3.1.1 "><p id="p1091665433317"><a name="p1091665433317"></a><a name="p1091665433317"></a><a href="nearbyintf.md">nearbyIntf</a></p>
</td>
<td class="cellrowborder" valign="top" width="60%" headers="mcps1.2.3.1.2 "><p id="p16917354163318"><a name="p16917354163318"></a><a name="p16917354163318"></a>获取与输入浮点数最接近的整数，输入浮点数与左右整数的距离相等时，返回偶数。</p>
</td>
</tr>
<tr id="row129171654113313"><td class="cellrowborder" valign="top" width="40%" headers="mcps1.2.3.1.1 "><p id="p191775483318"><a name="p191775483318"></a><a name="p191775483318"></a><a href="nextafterf.md">nextafterf</a></p>
</td>
<td class="cellrowborder" valign="top" width="60%" headers="mcps1.2.3.1.2 "><p id="p18917185418334"><a name="p18917185418334"></a><a name="p18917185418334"></a>如果y大于x，返回比x大的下一个可表示的浮点值，即浮点数二进制最低位加1。</p>
<p id="p8917354133310"><a name="p8917354133310"></a><a name="p8917354133310"></a>如果y小于x，返回比x小的下一个可表示的浮点值，即浮点数二进制最低位减1。</p>
<p id="p1991745433314"><a name="p1991745433314"></a><a name="p1991745433314"></a>如果y等于x，返回x。</p>
</td>
</tr>
<tr id="row149171154153319"><td class="cellrowborder" valign="top" width="40%" headers="mcps1.2.3.1.1 "><p id="p169175549335"><a name="p169175549335"></a><a name="p169175549335"></a><a href="scalbnf.md">scalbnf</a></p>
</td>
<td class="cellrowborder" valign="top" width="60%" headers="mcps1.2.3.1.2 "><p id="p5917135413314"><a name="p5917135413314"></a><a name="p5917135413314"></a>获取输入数据x与2的n次方的乘积。</p>
</td>
</tr>
<tr id="row3917125453319"><td class="cellrowborder" valign="top" width="40%" headers="mcps1.2.3.1.1 "><p id="p4917125413310"><a name="p4917125413310"></a><a name="p4917125413310"></a><a href="scalblnf.md">scalblnf</a></p>
</td>
<td class="cellrowborder" valign="top" width="60%" headers="mcps1.2.3.1.2 "><p id="p119172054133314"><a name="p119172054133314"></a><a name="p119172054133314"></a>获取输入数据x与2的n次方的乘积。</p>
</td>
</tr>
<tr id="row178323439144"><td class="cellrowborder" valign="top" width="40%" headers="mcps1.2.3.1.1 "><p id="p7832443201414"><a name="p7832443201414"></a><a name="p7832443201414"></a><a href="modff.md">modff</a></p>
</td>
<td class="cellrowborder" valign="top" width="60%" headers="mcps1.2.3.1.2 "><p id="p883214361417"><a name="p883214361417"></a><a name="p883214361417"></a><span>将输入数据分解为小数部分和整数部分</span>。</p>
</td>
</tr>
<tr id="row21937201350"><td class="cellrowborder" valign="top" width="40%" headers="mcps1.2.3.1.1 "><p id="p719313201519"><a name="p719313201519"></a><a name="p719313201519"></a><a href="labs.md">labs</a></p>
</td>
<td class="cellrowborder" valign="top" width="60%" headers="mcps1.2.3.1.2 "><p id="p1419316201258"><a name="p1419316201258"></a><a name="p1419316201258"></a>获取输入数据的绝对值。</p>
</td>
</tr>
<tr id="row2077317477520"><td class="cellrowborder" valign="top" width="40%" headers="mcps1.2.3.1.1 "><p id="p1377314471756"><a name="p1377314471756"></a><a name="p1377314471756"></a><a href="llabs.md">llabs</a></p>
</td>
<td class="cellrowborder" valign="top" width="60%" headers="mcps1.2.3.1.2 "><p id="p157737471356"><a name="p157737471356"></a><a name="p157737471356"></a>获取输入数据的绝对值。</p>
</td>
</tr>
<tr id="row07291552354"><td class="cellrowborder" valign="top" width="40%" headers="mcps1.2.3.1.1 "><p id="p107292521754"><a name="p107292521754"></a><a name="p107292521754"></a><a href="llmax.md">llmax</a></p>
</td>
<td class="cellrowborder" valign="top" width="60%" headers="mcps1.2.3.1.2 "><p id="p14729752759"><a name="p14729752759"></a><a name="p14729752759"></a>获取两个输入数据中的最大值。</p>
</td>
</tr>
<tr id="row03165507517"><td class="cellrowborder" valign="top" width="40%" headers="mcps1.2.3.1.1 "><p id="p103167505515"><a name="p103167505515"></a><a name="p103167505515"></a><a href="ullmax.md">ullmax</a></p>
</td>
<td class="cellrowborder" valign="top" width="60%" headers="mcps1.2.3.1.2 "><p id="p183163501555"><a name="p183163501555"></a><a name="p183163501555"></a>获取两个输入数据中的最大值。</p>
</td>
</tr>
<tr id="row11648735153"><td class="cellrowborder" valign="top" width="40%" headers="mcps1.2.3.1.1 "><p id="p5648163513514"><a name="p5648163513514"></a><a name="p5648163513514"></a><a href="umax.md">umax</a></p>
</td>
<td class="cellrowborder" valign="top" width="60%" headers="mcps1.2.3.1.2 "><p id="p76484350516"><a name="p76484350516"></a><a name="p76484350516"></a>获取两个输入数据中的最大值。</p>
</td>
</tr>
<tr id="row1985110321853"><td class="cellrowborder" valign="top" width="40%" headers="mcps1.2.3.1.1 "><p id="p885293212512"><a name="p885293212512"></a><a name="p885293212512"></a><a href="llmin.md">llmin</a></p>
</td>
<td class="cellrowborder" valign="top" width="60%" headers="mcps1.2.3.1.2 "><p id="p28521327516"><a name="p28521327516"></a><a name="p28521327516"></a>获取两个输入数据中的最小值。</p>
</td>
</tr>
<tr id="row1720314303513"><td class="cellrowborder" valign="top" width="40%" headers="mcps1.2.3.1.1 "><p id="p1320313308515"><a name="p1320313308515"></a><a name="p1320313308515"></a><a href="ullmin.md">ullmin</a></p>
</td>
<td class="cellrowborder" valign="top" width="60%" headers="mcps1.2.3.1.2 "><p id="p720343016512"><a name="p720343016512"></a><a name="p720343016512"></a>获取两个输入数据中的最小值。</p>
</td>
</tr>
<tr id="row182742272519"><td class="cellrowborder" valign="top" width="40%" headers="mcps1.2.3.1.1 "><p id="p192747271952"><a name="p192747271952"></a><a name="p192747271952"></a><a href="umin.md">umin</a></p>
</td>
<td class="cellrowborder" valign="top" width="60%" headers="mcps1.2.3.1.2 "><p id="p4274727352"><a name="p4274727352"></a><a name="p4274727352"></a>获取两个输入数据中的最小值。</p>
</td>
</tr>
<tr id="row11161152315511"><td class="cellrowborder" valign="top" width="40%" headers="mcps1.2.3.1.1 "><p id="p1216132319516"><a name="p1216132319516"></a><a name="p1216132319516"></a><a href="fdividef.md">fdivdef</a></p>
</td>
<td class="cellrowborder" valign="top" width="60%" headers="mcps1.2.3.1.2 "><p id="p116152315515"><a name="p116152315515"></a><a name="p116152315515"></a>获取两个输入数据相除的结果。</p>
</td>
</tr>
<tr id="row118201352288"><td class="cellrowborder" valign="top" width="40%" headers="mcps1.2.3.1.1 "><p id="p582012523817"><a name="p582012523817"></a><a name="p582012523817"></a><a href="signbit.md">signbit</a></p>
</td>
<td class="cellrowborder" valign="top" width="60%" headers="mcps1.2.3.1.2 "><p id="p108201952186"><a name="p108201952186"></a><a name="p108201952186"></a>获取输入数据的符号位。</p>
</td>
</tr>
</tbody>
</table>

**表 30**  精度转换

<a name="table14459746154815"></a>
<table><thead align="left"><tr id="row134598462487"><th class="cellrowborder" valign="top" width="40%" id="mcps1.2.3.1.1"><p id="p24591346124818"><a name="p24591346124818"></a><a name="p24591346124818"></a>接口名</p>
</th>
<th class="cellrowborder" valign="top" width="60%" id="mcps1.2.3.1.2"><p id="p1545918468481"><a name="p1545918468481"></a><a name="p1545918468481"></a>功能描述</p>
</th>
</tr>
</thead>
<tbody><tr id="row345924604816"><td class="cellrowborder" valign="top" width="40%" headers="mcps1.2.3.1.1 "><p id="p1245915462484"><a name="p1245915462484"></a><a name="p1245915462484"></a><a href="rintf.md">rintf</a></p>
</td>
<td class="cellrowborder" valign="top" width="60%" headers="mcps1.2.3.1.2 "><p id="p19459124613489"><a name="p19459124613489"></a><a name="p19459124613489"></a>获取与输入数据最接近的整数，若存在两个同样接近的整数，则获取其中的偶数。</p>
</td>
</tr>
<tr id="row729089477"><td class="cellrowborder" valign="top" width="40%" headers="mcps1.2.3.1.1 "><p id="p5291690717"><a name="p5291690717"></a><a name="p5291690717"></a><a href="hrint.md">hrint</a></p>
</td>
<td class="cellrowborder" valign="top" width="60%" headers="mcps1.2.3.1.2 "><p id="p182918911719"><a name="p182918911719"></a><a name="p182918911719"></a>获取与输入数据最接近的整数，若存在两个同样接近的整数，则获取其中的偶数。</p>
</td>
</tr>
<tr id="row12019499204"><td class="cellrowborder" valign="top" width="40%" headers="mcps1.2.3.1.1 "><p id="p130849102011"><a name="p130849102011"></a><a name="p130849102011"></a><a href="h2rint.md">h2rint</a></p>
</td>
<td class="cellrowborder" valign="top" width="60%" headers="mcps1.2.3.1.2 "><p id="p1901749162013"><a name="p1901749162013"></a><a name="p1901749162013"></a>获取与输入数据各元素最接近的整数，若存在两个同样接近的整数，则获取其中的偶数。</p>
</td>
</tr>
<tr id="row16459184654815"><td class="cellrowborder" valign="top" width="40%" headers="mcps1.2.3.1.1 "><p id="p18459204617484"><a name="p18459204617484"></a><a name="p18459204617484"></a><a href="lrintf.md">lrintf</a></p>
</td>
<td class="cellrowborder" valign="top" width="60%" headers="mcps1.2.3.1.2 "><p id="p20459104684810"><a name="p20459104684810"></a><a name="p20459104684810"></a>获取与输入数据最接近的整数，若存在两个同样接近的整数，则获取其中的偶数。</p>
</td>
</tr>
<tr id="row124591546124816"><td class="cellrowborder" valign="top" width="40%" headers="mcps1.2.3.1.1 "><p id="p1745974634812"><a name="p1745974634812"></a><a name="p1745974634812"></a><a href="llrintf.md">llrintf</a></p>
</td>
<td class="cellrowborder" valign="top" width="60%" headers="mcps1.2.3.1.2 "><p id="p245954654812"><a name="p245954654812"></a><a name="p245954654812"></a>获取与输入数据最接近的整数，若存在两个同样接近的整数，则获取其中的偶数。</p>
</td>
</tr>
<tr id="row14459124615487"><td class="cellrowborder" valign="top" width="40%" headers="mcps1.2.3.1.1 "><p id="p1545934694812"><a name="p1545934694812"></a><a name="p1545934694812"></a><a href="roundf.md">roundf</a></p>
</td>
<td class="cellrowborder" valign="top" width="60%" headers="mcps1.2.3.1.2 "><p id="p445916469481"><a name="p445916469481"></a><a name="p445916469481"></a>获取对输入数据四舍五入后的整数。</p>
</td>
</tr>
<tr id="row4459154613487"><td class="cellrowborder" valign="top" width="40%" headers="mcps1.2.3.1.1 "><p id="p1045944624815"><a name="p1045944624815"></a><a name="p1045944624815"></a><a href="lroundf.md">lroundf</a></p>
</td>
<td class="cellrowborder" valign="top" width="60%" headers="mcps1.2.3.1.2 "><p id="p3459046164817"><a name="p3459046164817"></a><a name="p3459046164817"></a>获取对输入数据四舍五入后的整数。</p>
</td>
</tr>
<tr id="row44591746144816"><td class="cellrowborder" valign="top" width="40%" headers="mcps1.2.3.1.1 "><p id="p194590469486"><a name="p194590469486"></a><a name="p194590469486"></a><a href="llroundf.md">llroundf</a></p>
</td>
<td class="cellrowborder" valign="top" width="60%" headers="mcps1.2.3.1.2 "><p id="p445924614485"><a name="p445924614485"></a><a name="p445924614485"></a>获取对输入数据四舍五入后的整数。</p>
</td>
</tr>
<tr id="row154598465484"><td class="cellrowborder" valign="top" width="40%" headers="mcps1.2.3.1.1 "><p id="p16459246174813"><a name="p16459246174813"></a><a name="p16459246174813"></a><a href="floorf.md">floorf</a></p>
</td>
<td class="cellrowborder" valign="top" width="60%" headers="mcps1.2.3.1.2 "><p id="p445924616480"><a name="p445924616480"></a><a name="p445924616480"></a>获取小于或等于输入数据的最大整数值。</p>
</td>
</tr>
<tr id="row1095217147715"><td class="cellrowborder" valign="top" width="40%" headers="mcps1.2.3.1.1 "><p id="p16953714579"><a name="p16953714579"></a><a name="p16953714579"></a><a href="hfloor.md">hfloor</a></p>
</td>
<td class="cellrowborder" valign="top" width="60%" headers="mcps1.2.3.1.2 "><p id="p1395331416711"><a name="p1395331416711"></a><a name="p1395331416711"></a>获取小于或等于输入数据的最大整数值。</p>
</td>
</tr>
<tr id="row11971115314208"><td class="cellrowborder" valign="top" width="40%" headers="mcps1.2.3.1.1 "><p id="p5972175362012"><a name="p5972175362012"></a><a name="p5972175362012"></a><a href="h2floor.md">h2floor</a></p>
</td>
<td class="cellrowborder" valign="top" width="60%" headers="mcps1.2.3.1.2 "><p id="p997245312209"><a name="p997245312209"></a><a name="p997245312209"></a>获取小于或等于输入数据各元素的最大整数值。</p>
</td>
</tr>
<tr id="row12811735115110"><td class="cellrowborder" valign="top" width="40%" headers="mcps1.2.3.1.1 "><p id="p42781135185115"><a name="p42781135185115"></a><a name="p42781135185115"></a><a href="ceilf.md">ceilf</a></p>
</td>
<td class="cellrowborder" valign="top" width="60%" headers="mcps1.2.3.1.2 "><p id="p8278103517513"><a name="p8278103517513"></a><a name="p8278103517513"></a>获取大于或等于输入数据的最小整数值。</p>
</td>
</tr>
<tr id="row187657177712"><td class="cellrowborder" valign="top" width="40%" headers="mcps1.2.3.1.1 "><p id="p1276516171876"><a name="p1276516171876"></a><a name="p1276516171876"></a><a href="hceil.md">hceil</a></p>
</td>
<td class="cellrowborder" valign="top" width="60%" headers="mcps1.2.3.1.2 "><p id="p4766181718717"><a name="p4766181718717"></a><a name="p4766181718717"></a>获取大于或等于输入数据的最小整数值。</p>
</td>
</tr>
<tr id="row55640577203"><td class="cellrowborder" valign="top" width="40%" headers="mcps1.2.3.1.1 "><p id="p17564125762010"><a name="p17564125762010"></a><a name="p17564125762010"></a><a href="h2ceil.md">h2ceil</a></p>
</td>
<td class="cellrowborder" valign="top" width="60%" headers="mcps1.2.3.1.2 "><p id="p125644579203"><a name="p125644579203"></a><a name="p125644579203"></a>获取大于或等于输入数据各元素的最小整数值。</p>
</td>
</tr>
<tr id="row3281335195118"><td class="cellrowborder" valign="top" width="40%" headers="mcps1.2.3.1.1 "><p id="p427916352511"><a name="p427916352511"></a><a name="p427916352511"></a><a href="truncf.md">truncf</a></p>
</td>
<td class="cellrowborder" valign="top" width="60%" headers="mcps1.2.3.1.2 "><p id="p12279133555111"><a name="p12279133555111"></a><a name="p12279133555111"></a>获取对输入数据的浮点数截断后的整数。</p>
</td>
</tr>
<tr id="row5414182018710"><td class="cellrowborder" valign="top" width="40%" headers="mcps1.2.3.1.1 "><p id="p114141420374"><a name="p114141420374"></a><a name="p114141420374"></a><a href="htrunc.md">htrunc</a></p>
</td>
<td class="cellrowborder" valign="top" width="60%" headers="mcps1.2.3.1.2 "><p id="p141412012710"><a name="p141412012710"></a><a name="p141412012710"></a>获取对输入数据的浮点数截断后的整数。</p>
</td>
</tr>
<tr id="row1524850182111"><td class="cellrowborder" valign="top" width="40%" headers="mcps1.2.3.1.1 "><p id="p182481506212"><a name="p182481506212"></a><a name="p182481506212"></a><a href="h2trunc.md">h2trunc</a></p>
</td>
<td class="cellrowborder" valign="top" width="60%" headers="mcps1.2.3.1.2 "><p id="p62485015213"><a name="p62485015213"></a><a name="p62485015213"></a>获取对输入数据各元素的浮点数截断后的整数。</p>
</td>
</tr>
</tbody>
</table>

**表 31**  比较函数

<a name="table13951184259"></a>
<table><thead align="left"><tr id="row20395131832515"><th class="cellrowborder" valign="top" width="40%" id="mcps1.2.3.1.1"><p id="p339517183253"><a name="p339517183253"></a><a name="p339517183253"></a>接口名</p>
</th>
<th class="cellrowborder" valign="top" width="60%" id="mcps1.2.3.1.2"><p id="p6395131819252"><a name="p6395131819252"></a><a name="p6395131819252"></a>功能描述</p>
</th>
</tr>
</thead>
<tbody><tr id="row93963189258"><td class="cellrowborder" valign="top" width="40%" headers="mcps1.2.3.1.1 "><p id="p439610186253"><a name="p439610186253"></a><a name="p439610186253"></a><a href="isfinite.md">isfinite</a></p>
</td>
<td class="cellrowborder" valign="top" width="60%" headers="mcps1.2.3.1.2 "><p id="p1396181813253"><a name="p1396181813253"></a><a name="p1396181813253"></a>判断浮点数是否为有限数（非inf、非nan）。</p>
</td>
</tr>
<tr id="row739661817254"><td class="cellrowborder" valign="top" width="40%" headers="mcps1.2.3.1.1 "><p id="p2396218192518"><a name="p2396218192518"></a><a name="p2396218192518"></a><a href="isnan.md">isnan</a></p>
</td>
<td class="cellrowborder" valign="top" width="60%" headers="mcps1.2.3.1.2 "><p id="p2039631814253"><a name="p2039631814253"></a><a name="p2039631814253"></a>判断浮点数是否为nan。</p>
</td>
</tr>
<tr id="row16744631793"><td class="cellrowborder" valign="top" width="40%" headers="mcps1.2.3.1.1 "><p id="p874412316911"><a name="p874412316911"></a><a name="p874412316911"></a><a href="__hisnan.md">__hisnan</a></p>
</td>
<td class="cellrowborder" valign="top" width="60%" headers="mcps1.2.3.1.2 "><p id="p15744113114918"><a name="p15744113114918"></a><a name="p15744113114918"></a>判断浮点数是否为nan。</p>
</td>
</tr>
<tr id="row5396191822512"><td class="cellrowborder" valign="top" width="40%" headers="mcps1.2.3.1.1 "><p id="p93961918152514"><a name="p93961918152514"></a><a name="p93961918152514"></a><a href="isinf.md">isinf</a></p>
</td>
<td class="cellrowborder" valign="top" width="60%" headers="mcps1.2.3.1.2 "><p id="p18215105443017"><a name="p18215105443017"></a><a name="p18215105443017"></a>判断浮点数是否为无穷。</p>
</td>
</tr>
<tr id="row3972933196"><td class="cellrowborder" valign="top" width="40%" headers="mcps1.2.3.1.1 "><p id="p199724331919"><a name="p199724331919"></a><a name="p199724331919"></a><a href="__hisinf.md">__hisinf</a></p>
</td>
<td class="cellrowborder" valign="top" width="60%" headers="mcps1.2.3.1.2 "><p id="p19728332911"><a name="p19728332911"></a><a name="p19728332911"></a>判断浮点数是否为无穷。</p>
</td>
</tr>
</tbody>
</table>

**表 32**  Atomic函数

<a name="table17209165495117"></a>
<table><thead align="left"><tr id="row720915541514"><th class="cellrowborder" valign="top" width="40%" id="mcps1.2.3.1.1"><p id="p16210954205119"><a name="p16210954205119"></a><a name="p16210954205119"></a>接口名</p>
</th>
<th class="cellrowborder" valign="top" width="60%" id="mcps1.2.3.1.2"><p id="p122101254105114"><a name="p122101254105114"></a><a name="p122101254105114"></a>功能描述</p>
</th>
</tr>
</thead>
<tbody><tr id="row221025405119"><td class="cellrowborder" valign="top" width="40%" headers="mcps1.2.3.1.1 "><p id="p18210165415117"><a name="p18210165415117"></a><a name="p18210165415117"></a><a href="asc_atomic_add.md">asc_atomic_add</a></p>
</td>
<td class="cellrowborder" valign="top" width="60%" headers="mcps1.2.3.1.2 "><p id="p14210105415119"><a name="p14210105415119"></a><a name="p14210105415119"></a>对<span id="ph10536132505718"><a name="ph10536132505718"></a><a name="ph10536132505718"></a>Unified Buffer</span>或<span id="ph1753616252577"><a name="ph1753616252577"></a><a name="ph1753616252577"></a>Global Memory</span>上的数据与指定数据执行原子加操作，即将指定数据累加到<span id="ph15143152082811"><a name="ph15143152082811"></a><a name="ph15143152082811"></a>Unified Buffer</span>或<span id="ph214322082812"><a name="ph214322082812"></a><a name="ph214322082812"></a>Global Memory</span>的数据中。</p>
</td>
</tr>
<tr id="row102101054135111"><td class="cellrowborder" valign="top" width="40%" headers="mcps1.2.3.1.1 "><p id="p10210654105113"><a name="p10210654105113"></a><a name="p10210654105113"></a><a href="asc_atomic_sub.md">asc_atomic_sub</a></p>
</td>
<td class="cellrowborder" valign="top" width="60%" headers="mcps1.2.3.1.2 "><p id="p1021019548518"><a name="p1021019548518"></a><a name="p1021019548518"></a>对<span id="ph5434440195718"><a name="ph5434440195718"></a><a name="ph5434440195718"></a>Unified Buffer</span>或<span id="ph1343414075710"><a name="ph1343414075710"></a><a name="ph1343414075710"></a>Global Memory</span>上的数据与指定数据执行原子减操作，即在<span id="ph93051920194612"><a name="ph93051920194612"></a><a name="ph93051920194612"></a>Unified Buffer</span>或<span id="ph1330532015466"><a name="ph1330532015466"></a><a name="ph1330532015466"></a>Global Memory</span>的数据上减去指定数据。</p>
</td>
</tr>
<tr id="row14210155416511"><td class="cellrowborder" valign="top" width="40%" headers="mcps1.2.3.1.1 "><p id="p12101954125120"><a name="p12101954125120"></a><a name="p12101954125120"></a><a href="asc_atomic_exch.md">asc_atomic_exch</a></p>
</td>
<td class="cellrowborder" valign="top" width="60%" headers="mcps1.2.3.1.2 "><p id="p921017541514"><a name="p921017541514"></a><a name="p921017541514"></a>对<span id="ph19824155135715"><a name="ph19824155135715"></a><a name="ph19824155135715"></a>Unified Buffer</span>或<span id="ph2824251175717"><a name="ph2824251175717"></a><a name="ph2824251175717"></a>Global Memory</span>地址做原子赋值操作，即将指定数据赋值到<span id="ph38242515577"><a name="ph38242515577"></a><a name="ph38242515577"></a>Unified Buffer</span>或<span id="ph1782418514572"><a name="ph1782418514572"></a><a name="ph1782418514572"></a>Global Memory</span>地址中。</p>
</td>
</tr>
<tr id="row9210125445116"><td class="cellrowborder" valign="top" width="40%" headers="mcps1.2.3.1.1 "><p id="p16210654145112"><a name="p16210654145112"></a><a name="p16210654145112"></a><a href="asc_atomic_max.md">asc_atomic_max</a></p>
</td>
<td class="cellrowborder" valign="top" width="60%" headers="mcps1.2.3.1.2 "><p id="p182101754115113"><a name="p182101754115113"></a><a name="p182101754115113"></a>对<span id="ph82259014584"><a name="ph82259014584"></a><a name="ph82259014584"></a>Unified Buffer</span>或<span id="ph722514018587"><a name="ph722514018587"></a><a name="ph722514018587"></a>Global Memory</span>数据做原子求最大值操作，即将<span id="ph72255035818"><a name="ph72255035818"></a><a name="ph72255035818"></a>Unified Buffer</span>或<span id="ph3225180185819"><a name="ph3225180185819"></a><a name="ph3225180185819"></a>Global Memory</span>的数据与指定数据中的最大值赋值到<span id="ph16444335125614"><a name="ph16444335125614"></a><a name="ph16444335125614"></a>Unified Buffer</span>或<span id="ph2444133513562"><a name="ph2444133513562"></a><a name="ph2444133513562"></a>Global Memory</span>地址中。</p>
</td>
</tr>
<tr id="row5210165425115"><td class="cellrowborder" valign="top" width="40%" headers="mcps1.2.3.1.1 "><p id="p321035415115"><a name="p321035415115"></a><a name="p321035415115"></a><a href="asc_atomic_min.md">asc_atomic_min</a></p>
</td>
<td class="cellrowborder" valign="top" width="60%" headers="mcps1.2.3.1.2 "><p id="p16210155445111"><a name="p16210155445111"></a><a name="p16210155445111"></a>对<span id="ph236151245817"><a name="ph236151245817"></a><a name="ph236151245817"></a>Unified Buffer</span>或<span id="ph17361012165811"><a name="ph17361012165811"></a><a name="ph17361012165811"></a>Global Memory</span>数据做原子求最小值操作，即将<span id="ph103671216583"><a name="ph103671216583"></a><a name="ph103671216583"></a>Unified Buffer</span>或<span id="ph43616124587"><a name="ph43616124587"></a><a name="ph43616124587"></a>Global Memory</span>的数据与指定数据中的最小值赋值到<span id="ph1336191245816"><a name="ph1336191245816"></a><a name="ph1336191245816"></a>Unified Buffer</span>或<span id="ph1236121265812"><a name="ph1236121265812"></a><a name="ph1236121265812"></a>Global Memory</span>地址中。</p>
</td>
</tr>
<tr id="row162101254175110"><td class="cellrowborder" valign="top" width="40%" headers="mcps1.2.3.1.1 "><p id="p421011547518"><a name="p421011547518"></a><a name="p421011547518"></a><a href="asc_atomic_inc.md">asc_atomic_inc</a></p>
</td>
<td class="cellrowborder" valign="top" width="60%" headers="mcps1.2.3.1.2 "><p id="p0210115417511"><a name="p0210115417511"></a><a name="p0210115417511"></a>对<span id="ph4991141914580"><a name="ph4991141914580"></a><a name="ph4991141914580"></a>Unified Buffer</span>或<span id="ph18991719175819"><a name="ph18991719175819"></a><a name="ph18991719175819"></a>Global Memory</span>上address的数值进行原子加1操作，如果address上的数值大于等于指定数值val，则对address赋值为0，否则将address上数值加1。</p>
</td>
</tr>
<tr id="row19210145435112"><td class="cellrowborder" valign="top" width="40%" headers="mcps1.2.3.1.1 "><p id="p2210115455112"><a name="p2210115455112"></a><a name="p2210115455112"></a><a href="asc_atomic_dec.md">asc_atomic_dec</a></p>
</td>
<td class="cellrowborder" valign="top" width="60%" headers="mcps1.2.3.1.2 "><p id="p1021025445117"><a name="p1021025445117"></a><a name="p1021025445117"></a>对<span id="ph1959983155818"><a name="ph1959983155818"></a><a name="ph1959983155818"></a>Unified Buffer</span>或<span id="ph05991831195817"><a name="ph05991831195817"></a><a name="ph05991831195817"></a>Global Memory</span>上address的数值进行原子减1操作，如果address上的数值等于0或大于指定数值val，则对address赋值为val，否则将address上数值减1。</p>
</td>
</tr>
<tr id="row13198193011527"><td class="cellrowborder" valign="top" width="40%" headers="mcps1.2.3.1.1 "><p id="p48379327525"><a name="p48379327525"></a><a name="p48379327525"></a><a href="asc_atomic_cas.md">asc_atomic_cas</a></p>
</td>
<td class="cellrowborder" valign="top" width="60%" headers="mcps1.2.3.1.2 "><p id="p101941330125214"><a name="p101941330125214"></a><a name="p101941330125214"></a>对<span id="ph8959114015812"><a name="ph8959114015812"></a><a name="ph8959114015812"></a>Unified Buffer</span>或<span id="ph0959144013587"><a name="ph0959144013587"></a><a name="ph0959144013587"></a>Global Memory</span>上address的数值进行原子比较赋值操作，如果address上的数值等于指定数值compare，则对address赋值为指定数值val，否则address的数值不变。</p>
</td>
</tr>
<tr id="row1819812308522"><td class="cellrowborder" valign="top" width="40%" headers="mcps1.2.3.1.1 "><p id="p18837203213524"><a name="p18837203213524"></a><a name="p18837203213524"></a><a href="asc_atomic_and.md">asc_atomic_and</a></p>
</td>
<td class="cellrowborder" valign="top" width="60%" headers="mcps1.2.3.1.2 "><p id="p919417304529"><a name="p919417304529"></a><a name="p919417304529"></a>对<span id="ph18404486584"><a name="ph18404486584"></a><a name="ph18404486584"></a>Unified Buffer</span>或<span id="ph138406481583"><a name="ph138406481583"></a><a name="ph138406481583"></a>Global Memory</span>上address的数值与指定数值val进行原子与（&amp;）操作，即将address数值与（&amp;）val的结果赋值到<span id="ph7927849219"><a name="ph7927849219"></a><a name="ph7927849219"></a>Unified Buffer</span>或<span id="ph4927742218"><a name="ph4927742218"></a><a name="ph4927742218"></a>Global Memory</span>上。</p>
</td>
</tr>
<tr id="row11982030105210"><td class="cellrowborder" valign="top" width="40%" headers="mcps1.2.3.1.1 "><p id="p1837163211522"><a name="p1837163211522"></a><a name="p1837163211522"></a><a href="asc_atomic_or.md">asc_atomic_or</a></p>
</td>
<td class="cellrowborder" valign="top" width="60%" headers="mcps1.2.3.1.2 "><p id="p15194130125218"><a name="p15194130125218"></a><a name="p15194130125218"></a>对<span id="ph652716542215"><a name="ph652716542215"></a><a name="ph652716542215"></a>Unified Buffer</span>或<span id="ph1652785412220"><a name="ph1652785412220"></a><a name="ph1652785412220"></a>Global Memory</span>上address的数值与指定数值val进行原子或（|）操作，即将address数值或（|）val的结果赋值到<span id="ph1333582155916"><a name="ph1333582155916"></a><a name="ph1333582155916"></a>Unified Buffer</span>或<span id="ph13335112125912"><a name="ph13335112125912"></a><a name="ph13335112125912"></a>Global Memory</span>上。</p>
</td>
</tr>
<tr id="row819816309523"><td class="cellrowborder" valign="top" width="40%" headers="mcps1.2.3.1.1 "><p id="p108358323529"><a name="p108358323529"></a><a name="p108358323529"></a><a href="asc_atomic_xor.md">asc_atomic_xor</a></p>
</td>
<td class="cellrowborder" valign="top" width="60%" headers="mcps1.2.3.1.2 "><p id="p8194203065216"><a name="p8194203065216"></a><a name="p8194203065216"></a>对<span id="ph12461111115918"><a name="ph12461111115918"></a><a name="ph12461111115918"></a>Unified Buffer</span>或<span id="ph19461181110597"><a name="ph19461181110597"></a><a name="ph19461181110597"></a>Global Memory</span>上address的数值与指定数值val进行原子异或（^）操作，即将address数值异或（^）val的结果赋值到<span id="ph746121145910"><a name="ph746121145910"></a><a name="ph746121145910"></a>Unified Buffer</span>或<span id="ph1946111113591"><a name="ph1946111113591"></a><a name="ph1946111113591"></a>Global Memory</span>上。</p>
</td>
</tr>
</tbody>
</table>

**表 33**  Warp函数

<a name="table13746514532"></a>
<table><thead align="left"><tr id="row53744575316"><th class="cellrowborder" valign="top" width="40%" id="mcps1.2.3.1.1"><p id="p133744575314"><a name="p133744575314"></a><a name="p133744575314"></a>接口名</p>
</th>
<th class="cellrowborder" valign="top" width="60%" id="mcps1.2.3.1.2"><p id="p1137419545315"><a name="p1137419545315"></a><a name="p1137419545315"></a>功能描述</p>
</th>
</tr>
</thead>
<tbody><tr id="row63742535318"><td class="cellrowborder" valign="top" width="40%" headers="mcps1.2.3.1.1 "><p id="p137411525310"><a name="p137411525310"></a><a name="p137411525310"></a><a href="asc_all.md">asc_all</a></p>
</td>
<td class="cellrowborder" valign="top" width="60%" headers="mcps1.2.3.1.2 "><p id="p118785325243"><a name="p118785325243"></a><a name="p118785325243"></a>判断是否所有活跃线程的输入均不为0。</p>
</td>
</tr>
<tr id="row1374145125316"><td class="cellrowborder" valign="top" width="40%" headers="mcps1.2.3.1.1 "><p id="p3374165145317"><a name="p3374165145317"></a><a name="p3374165145317"></a><a href="asc_any.md">asc_any</a></p>
</td>
<td class="cellrowborder" valign="top" width="60%" headers="mcps1.2.3.1.2 "><p id="p361320505593"><a name="p361320505593"></a><a name="p361320505593"></a>判断是否有活跃线程的输入不为0。</p>
</td>
</tr>
<tr id="row937495135320"><td class="cellrowborder" valign="top" width="40%" headers="mcps1.2.3.1.1 "><p id="p1237412595310"><a name="p1237412595310"></a><a name="p1237412595310"></a><a href="asc_ballot.md">asc_ballot</a></p>
</td>
<td class="cellrowborder" valign="top" width="60%" headers="mcps1.2.3.1.2 "><p id="p178511254143017"><a name="p178511254143017"></a><a name="p178511254143017"></a>判断Warp内每个活跃线程的输入是否不为0。</p>
</td>
</tr>
<tr id="row937475185312"><td class="cellrowborder" valign="top" width="40%" headers="mcps1.2.3.1.1 "><p id="p18374152538"><a name="p18374152538"></a><a name="p18374152538"></a><a href="asc_activemask.md">asc_activemask</a></p>
</td>
<td class="cellrowborder" valign="top" width="60%" headers="mcps1.2.3.1.2 "><p id="p13965195011409"><a name="p13965195011409"></a><a name="p13965195011409"></a>查看Warp内所有线程是否为活跃状态。</p>
</td>
</tr>
<tr id="row16374452533"><td class="cellrowborder" valign="top" width="40%" headers="mcps1.2.3.1.1 "><p id="p737413565311"><a name="p737413565311"></a><a name="p737413565311"></a><a href="asc_shfl.md">asc_shfl</a></p>
</td>
<td class="cellrowborder" valign="top" width="60%" headers="mcps1.2.3.1.2 "><p id="p8374185115315"><a name="p8374185115315"></a><a name="p8374185115315"></a>获取Warp内指定线程srcLane输入的用于交换的var值。</p>
</td>
</tr>
<tr id="row13747585320"><td class="cellrowborder" valign="top" width="40%" headers="mcps1.2.3.1.1 "><p id="p337419535319"><a name="p337419535319"></a><a name="p337419535319"></a><a href="asc_shfl_up.md">asc_shfl_up</a></p>
</td>
<td class="cellrowborder" valign="top" width="60%" headers="mcps1.2.3.1.2 "><p id="p1337411519534"><a name="p1337411519534"></a><a name="p1337411519534"></a>获取Warp内当前线程向前偏移delta（当前线程LaneId-delta）的线程输入的用于交换的var值。</p>
</td>
</tr>
<tr id="row193742515530"><td class="cellrowborder" valign="top" width="40%" headers="mcps1.2.3.1.1 "><p id="p637420511533"><a name="p637420511533"></a><a name="p637420511533"></a><a href="asc_shfl_down.md">asc_shfl_down</a></p>
</td>
<td class="cellrowborder" valign="top" width="60%" headers="mcps1.2.3.1.2 "><p id="p537455125315"><a name="p537455125315"></a><a name="p537455125315"></a>获取Warp内当前线程向后偏移delta（当前线程LaneId+delta）的线程输入的用于交换的var值。</p>
</td>
</tr>
<tr id="row103591959165310"><td class="cellrowborder" valign="top" width="40%" headers="mcps1.2.3.1.1 "><p id="p19356125955319"><a name="p19356125955319"></a><a name="p19356125955319"></a><a href="asc_shfl_xor.md">asc_shfl_xor</a></p>
</td>
<td class="cellrowborder" valign="top" width="60%" headers="mcps1.2.3.1.2 "><p id="p14356125916533"><a name="p14356125916533"></a><a name="p14356125916533"></a>获取Warp内当前线程LaneId与输入laneMask做异或操作（LaneId^laneMask）得到的dstLaneId对应线程输入的用于交换的var值。</p>
</td>
</tr>
<tr id="row335915918538"><td class="cellrowborder" valign="top" width="40%" headers="mcps1.2.3.1.1 "><p id="p163562059105319"><a name="p163562059105319"></a><a name="p163562059105319"></a><a href="asc_reduce_add.md">asc_reduce_add</a></p>
</td>
<td class="cellrowborder" valign="top" width="60%" headers="mcps1.2.3.1.2 "><p id="p6356195985310"><a name="p6356195985310"></a><a name="p6356195985310"></a>对Warp内所有活跃线程输入的val求和。</p>
</td>
</tr>
<tr id="row18358459195316"><td class="cellrowborder" valign="top" width="40%" headers="mcps1.2.3.1.1 "><p id="p183561590539"><a name="p183561590539"></a><a name="p183561590539"></a><a href="asc_reduce_max.md">asc_reduce_max</a></p>
</td>
<td class="cellrowborder" valign="top" width="60%" headers="mcps1.2.3.1.2 "><p id="p3356759135312"><a name="p3356759135312"></a><a name="p3356759135312"></a>对Warp内所有活跃线程输入的val求最大值。</p>
</td>
</tr>
<tr id="row6358459175313"><td class="cellrowborder" valign="top" width="40%" headers="mcps1.2.3.1.1 "><p id="p83567590530"><a name="p83567590530"></a><a name="p83567590530"></a><a href="asc_reduce_min.md">asc_reduce_min</a></p>
</td>
<td class="cellrowborder" valign="top" width="60%" headers="mcps1.2.3.1.2 "><p id="p1235655945315"><a name="p1235655945315"></a><a name="p1235655945315"></a>对Warp内所有活跃线程输入val求最小值。</p>
</td>
</tr>
</tbody>
</table>

**表 34**  类型转换

<a name="table113998363475"></a>
<table><thead align="left"><tr id="row143991736144719"><th class="cellrowborder" valign="top" width="50%" id="mcps1.2.3.1.1"><p id="p199910538472"><a name="p199910538472"></a><a name="p199910538472"></a>接口名</p>
</th>
<th class="cellrowborder" valign="top" width="50%" id="mcps1.2.3.1.2"><p id="p7100105313479"><a name="p7100105313479"></a><a name="p7100105313479"></a>功能描述</p>
</th>
</tr>
</thead>
<tbody><tr id="row13477014183219"><td class="cellrowborder" valign="top" width="50%" headers="mcps1.2.3.1.1 "><p id="p174771914103211"><a name="p174771914103211"></a><a name="p174771914103211"></a><a href="__float2float_rn.md">__float2float_rn</a></p>
</td>
<td class="cellrowborder" valign="top" width="50%" headers="mcps1.2.3.1.2 "><p id="p1939214217487"><a name="p1939214217487"></a><a name="p1939214217487"></a>输入遵循CAST_RINT模式取整后的浮点数。</p>
</td>
</tr>
<tr id="row3941131993217"><td class="cellrowborder" valign="top" width="50%" headers="mcps1.2.3.1.1 "><p id="p9941919163220"><a name="p9941919163220"></a><a name="p9941919163220"></a><a href="__float2float_rz.md">__float2float_rz</a></p>
</td>
<td class="cellrowborder" valign="top" width="50%" headers="mcps1.2.3.1.2 "><p id="p89891442114818"><a name="p89891442114818"></a><a name="p89891442114818"></a>输入遵循CAST_TRUNC模式取整后的浮点数。</p>
</td>
</tr>
<tr id="row1527715228320"><td class="cellrowborder" valign="top" width="50%" headers="mcps1.2.3.1.1 "><p id="p72778226328"><a name="p72778226328"></a><a name="p72778226328"></a><a href="__float2float_rd.md">__float2float_rd</a></p>
</td>
<td class="cellrowborder" valign="top" width="50%" headers="mcps1.2.3.1.2 "><p id="p451244364810"><a name="p451244364810"></a><a name="p451244364810"></a>输入遵循CAST_FLOOR模式取整后的浮点数。</p>
</td>
</tr>
<tr id="row3495172443220"><td class="cellrowborder" valign="top" width="50%" headers="mcps1.2.3.1.1 "><p id="p14495924123219"><a name="p14495924123219"></a><a name="p14495924123219"></a><a href="__float2float_ru.md">__float2float_ru</a></p>
</td>
<td class="cellrowborder" valign="top" width="50%" headers="mcps1.2.3.1.2 "><p id="p1813744134814"><a name="p1813744134814"></a><a name="p1813744134814"></a>输入遵循CAST_CEIL模式取整后的浮点数。</p>
</td>
</tr>
<tr id="row69191126113214"><td class="cellrowborder" valign="top" width="50%" headers="mcps1.2.3.1.1 "><p id="p1919162623211"><a name="p1919162623211"></a><a name="p1919162623211"></a><a href="__float2float_rna.md">__float2float_rna</a></p>
</td>
<td class="cellrowborder" valign="top" width="50%" headers="mcps1.2.3.1.2 "><p id="p2487174415481"><a name="p2487174415481"></a><a name="p2487174415481"></a>输入遵循CAST_ROUND模式取整后的浮点数。</p>
</td>
</tr>
<tr id="row23991536104710"><td class="cellrowborder" valign="top" width="50%" headers="mcps1.2.3.1.1 "><p id="p7545194413475"><a name="p7545194413475"></a><a name="p7545194413475"></a><a href="__float2half.md">__float2half</a></p>
</td>
<td class="cellrowborder" valign="top" width="50%" headers="mcps1.2.3.1.2 "><p id="p67160345551"><a name="p67160345551"></a><a name="p67160345551"></a>将浮点数转换为半精度浮点数，并四舍五入到最接近的偶数，返回转换后的值。</p>
</td>
</tr>
<tr id="row6399173694716"><td class="cellrowborder" valign="top" width="50%" headers="mcps1.2.3.1.1 "><p id="p10545844174711"><a name="p10545844174711"></a><a name="p10545844174711"></a><a href="__float2half_rn.md">__float2half_rn</a></p>
</td>
<td class="cellrowborder" valign="top" width="50%" headers="mcps1.2.3.1.2 "><p id="p164361426604"><a name="p164361426604"></a><a name="p164361426604"></a>将浮点数转换为半精度浮点数，并四舍五入到最接近的偶数，返回转换后的值。</p>
</td>
</tr>
<tr id="row14326820183318"><td class="cellrowborder" valign="top" width="50%" headers="mcps1.2.3.1.1 "><p id="p15327122083313"><a name="p15327122083313"></a><a name="p15327122083313"></a><a href="__float2half_rn_sat.md">__float2half_rn_sat</a></p>
</td>
<td class="cellrowborder" valign="top" width="50%" headers="mcps1.2.3.1.2 "><p id="p15327520203317"><a name="p15327520203317"></a><a name="p15327520203317"></a>饱和模式下将输入遵循CAST_RINT模式转换成的半精度浮点数。</p>
</td>
</tr>
<tr id="row1239912369471"><td class="cellrowborder" valign="top" width="50%" headers="mcps1.2.3.1.1 "><p id="p17545154414473"><a name="p17545154414473"></a><a name="p17545154414473"></a><a href="__float2half_rz.md">__float2half_rz</a></p>
</td>
<td class="cellrowborder" valign="top" width="50%" headers="mcps1.2.3.1.2 "><p id="p220920202916"><a name="p220920202916"></a><a name="p220920202916"></a>将浮点数转换为半精度浮点数，并向零的方向舍入，返回转换后的值。</p>
</td>
</tr>
<tr id="row1468151215520"><td class="cellrowborder" valign="top" width="50%" headers="mcps1.2.3.1.1 "><p id="p969141265211"><a name="p969141265211"></a><a name="p969141265211"></a><a href="__float2half_rz_sat.md">__float2half_rz_sat</a></p>
</td>
<td class="cellrowborder" valign="top" width="50%" headers="mcps1.2.3.1.2 "><p id="p14691121529"><a name="p14691121529"></a><a name="p14691121529"></a>饱和模式下将输入遵循CAST_TRUNC模式转换成的半精度浮点数。</p>
</td>
</tr>
<tr id="row8399236134717"><td class="cellrowborder" valign="top" width="50%" headers="mcps1.2.3.1.1 "><p id="p15546164413471"><a name="p15546164413471"></a><a name="p15546164413471"></a><a href="__float2half_rd.md">__float2half_rd</a></p>
</td>
<td class="cellrowborder" valign="top" width="50%" headers="mcps1.2.3.1.2 "><p id="p20533447152814"><a name="p20533447152814"></a><a name="p20533447152814"></a>将浮点数转换为半精度浮点数，并在转换过程中向下取整，返回转换后的值。</p>
</td>
</tr>
<tr id="row1822513159523"><td class="cellrowborder" valign="top" width="50%" headers="mcps1.2.3.1.1 "><p id="p142252015185219"><a name="p142252015185219"></a><a name="p142252015185219"></a><a href="__float2half_rd_sat.md">__float2half_rd_sat</a></p>
</td>
<td class="cellrowborder" valign="top" width="50%" headers="mcps1.2.3.1.2 "><p id="p422581595215"><a name="p422581595215"></a><a name="p422581595215"></a>饱和模式下将输入遵循CAST_FLOOR模式转换成的半精度浮点数。</p>
</td>
</tr>
<tr id="row3399736164710"><td class="cellrowborder" valign="top" width="50%" headers="mcps1.2.3.1.1 "><p id="p5546124415471"><a name="p5546124415471"></a><a name="p5546124415471"></a><a href="__float2half_ru.md">__float2half_ru</a></p>
</td>
<td class="cellrowborder" valign="top" width="50%" headers="mcps1.2.3.1.2 "><p id="p8332154901"><a name="p8332154901"></a><a name="p8332154901"></a>将浮点数转换为半精度浮点数，并在转换过程中向上取整，返回转换后的值。</p>
</td>
</tr>
<tr id="row82551018185214"><td class="cellrowborder" valign="top" width="50%" headers="mcps1.2.3.1.1 "><p id="p125581818525"><a name="p125581818525"></a><a name="p125581818525"></a><a href="__float2half_ru_sat.md">__float2half_ru_sat</a></p>
</td>
<td class="cellrowborder" valign="top" width="50%" headers="mcps1.2.3.1.2 "><p id="p13134320125316"><a name="p13134320125316"></a><a name="p13134320125316"></a>饱和模式下将输入遵循CAST_CEIL模式转换成的半精度浮点数。</p>
</td>
</tr>
<tr id="row1242312012415"><td class="cellrowborder" valign="top" width="50%" headers="mcps1.2.3.1.1 "><p id="p174237032416"><a name="p174237032416"></a><a name="p174237032416"></a><a href="__float2half_rna.md">__float2half_rna</a></p>
</td>
<td class="cellrowborder" valign="top" width="50%" headers="mcps1.2.3.1.2 "><p id="p042330122418"><a name="p042330122418"></a><a name="p042330122418"></a>将浮点数转换为半精度浮点数，并向远离零的方向舍入，返回转换后的值。</p>
</td>
</tr>
<tr id="row17265142114524"><td class="cellrowborder" valign="top" width="50%" headers="mcps1.2.3.1.1 "><p id="p3265721195210"><a name="p3265721195210"></a><a name="p3265721195210"></a><a href="__float2half_rna_sat.md">__float2half_rna_sat</a></p>
</td>
<td class="cellrowborder" valign="top" width="50%" headers="mcps1.2.3.1.2 "><p id="p3266132119520"><a name="p3266132119520"></a><a name="p3266132119520"></a>饱和模式下将输入遵循CAST_ROUND模式转换成的半精度浮点数。</p>
</td>
</tr>
<tr id="row1251930240"><td class="cellrowborder" valign="top" width="50%" headers="mcps1.2.3.1.1 "><p id="p2025143152418"><a name="p2025143152418"></a><a name="p2025143152418"></a><a href="__float2half_ro.md">__float2half_ro</a></p>
</td>
<td class="cellrowborder" valign="top" width="50%" headers="mcps1.2.3.1.2 "><p id="p1251193192415"><a name="p1251193192415"></a><a name="p1251193192415"></a>将浮点数转换为半精度浮点数，并四舍五入到最接近的奇数，返回转换后的值。</p>
</td>
</tr>
<tr id="row178351258125519"><td class="cellrowborder" valign="top" width="50%" headers="mcps1.2.3.1.1 "><p id="p148356589558"><a name="p148356589558"></a><a name="p148356589558"></a><a href="__float2half_ro_sat.md">__float2half_ro_sat</a></p>
</td>
<td class="cellrowborder" valign="top" width="50%" headers="mcps1.2.3.1.2 "><p id="p1583575819554"><a name="p1583575819554"></a><a name="p1583575819554"></a>饱和模式下将输入遵循CAST_ODD模式转换成的半精度浮点数。</p>
</td>
</tr>
<tr id="row143991336194711"><td class="cellrowborder" valign="top" width="50%" headers="mcps1.2.3.1.1 "><p id="p1354664444719"><a name="p1354664444719"></a><a name="p1354664444719"></a><a href="__float2bfloat16.md">__float2bfloat16</a></p>
</td>
<td class="cellrowborder" valign="top" width="50%" headers="mcps1.2.3.1.2 "><p id="p13160165155714"><a name="p13160165155714"></a><a name="p13160165155714"></a>将浮点数转换为bfloat16精度，并四舍五入到最接近的偶数，返回转换后的值。</p>
</td>
</tr>
<tr id="row1640063654713"><td class="cellrowborder" valign="top" width="50%" headers="mcps1.2.3.1.1 "><p id="p75463441475"><a name="p75463441475"></a><a name="p75463441475"></a><a href="__float2bfloat16_rn.md">__float2bfloat16_rn</a></p>
</td>
<td class="cellrowborder" valign="top" width="50%" headers="mcps1.2.3.1.2 "><p id="p54391374585"><a name="p54391374585"></a><a name="p54391374585"></a>将浮点数转换为bfloat16精度，并四舍五入到最接近的偶数，返回转换后的值。</p>
</td>
</tr>
<tr id="row11775943185916"><td class="cellrowborder" valign="top" width="50%" headers="mcps1.2.3.1.1 "><p id="p7776194355910"><a name="p7776194355910"></a><a name="p7776194355910"></a><a href="__float2bfloat16_rn_sat.md">__float2bfloat16_rn_sat</a></p>
</td>
<td class="cellrowborder" valign="top" width="50%" headers="mcps1.2.3.1.2 "><p id="p87763432594"><a name="p87763432594"></a><a name="p87763432594"></a>饱和模式下将输入遵循CAST_RINT模式转换成的bfloat16类型数据。</p>
</td>
</tr>
<tr id="row34004367474"><td class="cellrowborder" valign="top" width="50%" headers="mcps1.2.3.1.1 "><p id="p12547194414712"><a name="p12547194414712"></a><a name="p12547194414712"></a><a href="__float2bfloat16_rz.md">__float2bfloat16_rz</a></p>
</td>
<td class="cellrowborder" valign="top" width="50%" headers="mcps1.2.3.1.2 "><p id="p1081512231916"><a name="p1081512231916"></a><a name="p1081512231916"></a>将浮点数转换为bfloat16精度，并向零的方向舍入，返回转换后的值。</p>
</td>
</tr>
<tr id="row0227154618591"><td class="cellrowborder" valign="top" width="50%" headers="mcps1.2.3.1.1 "><p id="p822754614593"><a name="p822754614593"></a><a name="p822754614593"></a><a href="__float2bfloat16_rz_sat.md">__float2bfloat16_rz_sat</a></p>
</td>
<td class="cellrowborder" valign="top" width="50%" headers="mcps1.2.3.1.2 "><p id="p522754612591"><a name="p522754612591"></a><a name="p522754612591"></a>饱和模式下将输入遵循CAST_TRUNC模式转换成的bfloat16类型数据。</p>
</td>
</tr>
<tr id="row8400103613470"><td class="cellrowborder" valign="top" width="50%" headers="mcps1.2.3.1.1 "><p id="p3547154410476"><a name="p3547154410476"></a><a name="p3547154410476"></a><a href="__float2bfloat16_rd.md">__float2bfloat16_rd</a></p>
</td>
<td class="cellrowborder" valign="top" width="50%" headers="mcps1.2.3.1.2 "><p id="p171811233813"><a name="p171811233813"></a><a name="p171811233813"></a>将浮点数转换为bfloat16精度，在转换过程中将结果向下舍入，返回转换后的值。</p>
</td>
</tr>
<tr id="row72995017598"><td class="cellrowborder" valign="top" width="50%" headers="mcps1.2.3.1.1 "><p id="p12965014595"><a name="p12965014595"></a><a name="p12965014595"></a><a href="__float2bfloat16_rd_sat.md">__float2bfloat16_rd_sat</a></p>
</td>
<td class="cellrowborder" valign="top" width="50%" headers="mcps1.2.3.1.2 "><p id="p6291850165919"><a name="p6291850165919"></a><a name="p6291850165919"></a>饱和模式下将输入遵循CAST_FLOOR模式转换成的bfloat16类型数据。</p>
</td>
</tr>
<tr id="row3400153614475"><td class="cellrowborder" valign="top" width="50%" headers="mcps1.2.3.1.1 "><p id="p1654784404718"><a name="p1654784404718"></a><a name="p1654784404718"></a><a href="__float2bfloat16_ru.md">__float2bfloat16_ru</a></p>
</td>
<td class="cellrowborder" valign="top" width="50%" headers="mcps1.2.3.1.2 "><p id="p12717143411552"><a name="p12717143411552"></a><a name="p12717143411552"></a>将浮点数转换为bfloat16精度，在转换过程中将结果向上取整，返回转换后的值。</p>
</td>
</tr>
<tr id="row831755195911"><td class="cellrowborder" valign="top" width="50%" headers="mcps1.2.3.1.1 "><p id="p113125510599"><a name="p113125510599"></a><a name="p113125510599"></a><a href="__float2bfloat16_ru_sat.md">__float2bfloat16_ru_sat</a></p>
</td>
<td class="cellrowborder" valign="top" width="50%" headers="mcps1.2.3.1.2 "><p id="p193105513595"><a name="p193105513595"></a><a name="p193105513595"></a>饱和模式下将输入遵循CAST_CEIL模式转换成的bfloat16类型数据。</p>
</td>
</tr>
<tr id="row3872927162415"><td class="cellrowborder" valign="top" width="50%" headers="mcps1.2.3.1.1 "><p id="p5873192717244"><a name="p5873192717244"></a><a name="p5873192717244"></a><a href="__float2bfloat16_rna.md">__float2bfloat16_rna</a></p>
</td>
<td class="cellrowborder" valign="top" width="50%" headers="mcps1.2.3.1.2 "><p id="p1687382752418"><a name="p1687382752418"></a><a name="p1687382752418"></a>将浮点数转换为bfloat16精度，并向远离零的方向舍入，返回转换后的值。</p>
</td>
</tr>
<tr id="row1570655213598"><td class="cellrowborder" valign="top" width="50%" headers="mcps1.2.3.1.1 "><p id="p1870635218599"><a name="p1870635218599"></a><a name="p1870635218599"></a><a href="__float2bfloat16_rna_sat.md">__float2bfloat16_rna_sat</a></p>
</td>
<td class="cellrowborder" valign="top" width="50%" headers="mcps1.2.3.1.2 "><p id="p1706752185916"><a name="p1706752185916"></a><a name="p1706752185916"></a>饱和模式下将输入遵循CAST_ROUND模式转换成的bfloat16类型数据。</p>
</td>
</tr>
<tr id="row17400193634715"><td class="cellrowborder" valign="top" width="50%" headers="mcps1.2.3.1.1 "><p id="p654810448473"><a name="p654810448473"></a><a name="p654810448473"></a><a href="__float2uint_rn.md">__float2uint_rn</a></p>
</td>
<td class="cellrowborder" valign="top" width="50%" headers="mcps1.2.3.1.2 "><p id="p1571753495516"><a name="p1571753495516"></a><a name="p1571753495516"></a>将浮点数转换为四舍五入至最接近的偶数的无符号整数。</p>
</td>
</tr>
<tr id="row140013369471"><td class="cellrowborder" valign="top" width="50%" headers="mcps1.2.3.1.1 "><p id="p154815446477"><a name="p154815446477"></a><a name="p154815446477"></a><a href="__float2uint_rz.md">__float2uint_rz</a></p>
</td>
<td class="cellrowborder" valign="top" width="50%" headers="mcps1.2.3.1.2 "><p id="p871711343558"><a name="p871711343558"></a><a name="p871711343558"></a><span>将浮点数转换为向零舍入的无符号整数</span>。</p>
</td>
</tr>
<tr id="row104006369477"><td class="cellrowborder" valign="top" width="50%" headers="mcps1.2.3.1.1 "><p id="p74001636134710"><a name="p74001636134710"></a><a name="p74001636134710"></a><a href="__float2uint_rd.md">__float2uint_rd</a></p>
</td>
<td class="cellrowborder" valign="top" width="50%" headers="mcps1.2.3.1.2 "><p id="p068816420211"><a name="p068816420211"></a><a name="p068816420211"></a><span>将浮点数转换为向下取整的无符号整数</span>。</p>
</td>
</tr>
<tr id="row3400163684717"><td class="cellrowborder" valign="top" width="50%" headers="mcps1.2.3.1.1 "><p id="p14400143684711"><a name="p14400143684711"></a><a name="p14400143684711"></a><a href="__float2uint_ru.md">__float2uint_ru</a></p>
</td>
<td class="cellrowborder" valign="top" width="50%" headers="mcps1.2.3.1.2 "><p id="p1241111211217"><a name="p1241111211217"></a><a name="p1241111211217"></a><span>将浮点数转换为向上取整的无符号整数</span>。</p>
</td>
</tr>
<tr id="row15135232162410"><td class="cellrowborder" valign="top" width="50%" headers="mcps1.2.3.1.1 "><p id="p4136163242415"><a name="p4136163242415"></a><a name="p4136163242415"></a><a href="__float2uint_rna.md">__float2uint_rna</a></p>
</td>
<td class="cellrowborder" valign="top" width="50%" headers="mcps1.2.3.1.2 "><p id="p713673292411"><a name="p713673292411"></a><a name="p713673292411"></a><span>将浮点数转换为向远离零舍入的无符号整数</span>。</p>
</td>
</tr>
<tr id="row1840043619473"><td class="cellrowborder" valign="top" width="50%" headers="mcps1.2.3.1.1 "><p id="p20400113634718"><a name="p20400113634718"></a><a name="p20400113634718"></a><a href="__float2int_rn.md">__float2int_rn</a></p>
</td>
<td class="cellrowborder" valign="top" width="50%" headers="mcps1.2.3.1.2 "><p id="p371763410551"><a name="p371763410551"></a><a name="p371763410551"></a><span>将浮点数转换为有符号整数，并四舍五入到最接近的偶数</span>。</p>
</td>
</tr>
<tr id="row540017368474"><td class="cellrowborder" valign="top" width="50%" headers="mcps1.2.3.1.1 "><p id="p184004360478"><a name="p184004360478"></a><a name="p184004360478"></a><a href="__float2int_rz.md">__float2int_rz</a></p>
</td>
<td class="cellrowborder" valign="top" width="50%" headers="mcps1.2.3.1.2 "><p id="p53909251622"><a name="p53909251622"></a><a name="p53909251622"></a><span>将浮点数转换为向零舍入的有符号整数</span>。</p>
</td>
</tr>
<tr id="row15400103624711"><td class="cellrowborder" valign="top" width="50%" headers="mcps1.2.3.1.1 "><p id="p3400936114712"><a name="p3400936114712"></a><a name="p3400936114712"></a><a href="__float2int_rd.md">__float2int_rd</a></p>
</td>
<td class="cellrowborder" valign="top" width="50%" headers="mcps1.2.3.1.2 "><p id="p41610341227"><a name="p41610341227"></a><a name="p41610341227"></a><span>将浮点数转换为向下取整的有符号整数</span>。</p>
</td>
</tr>
<tr id="row140023614717"><td class="cellrowborder" valign="top" width="50%" headers="mcps1.2.3.1.1 "><p id="p9401136114720"><a name="p9401136114720"></a><a name="p9401136114720"></a><a href="__float2int_ru.md">__float2int_ru</a></p>
</td>
<td class="cellrowborder" valign="top" width="50%" headers="mcps1.2.3.1.2 "><p id="p171712341559"><a name="p171712341559"></a><a name="p171712341559"></a><span>将浮点数转换为向上取整的有符号整数</span>。</p>
</td>
</tr>
<tr id="row792233742414"><td class="cellrowborder" valign="top" width="50%" headers="mcps1.2.3.1.1 "><p id="p18923143719248"><a name="p18923143719248"></a><a name="p18923143719248"></a><a href="__float2int_rna.md">__float2int_rna</a></p>
</td>
<td class="cellrowborder" valign="top" width="50%" headers="mcps1.2.3.1.2 "><p id="p149231937112419"><a name="p149231937112419"></a><a name="p149231937112419"></a><span>将浮点数转换为向远离零舍入的有符号整数</span>。</p>
</td>
</tr>
<tr id="row144011136124717"><td class="cellrowborder" valign="top" width="50%" headers="mcps1.2.3.1.1 "><p id="p134015361474"><a name="p134015361474"></a><a name="p134015361474"></a><a href="__float2ull_rn.md">__float2ull_rn</a></p>
</td>
<td class="cellrowborder" valign="top" width="50%" headers="mcps1.2.3.1.2 "><p id="p11502195014220"><a name="p11502195014220"></a><a name="p11502195014220"></a><span>将浮点数转换为四舍五入到最接近偶数的64位无符号整数</span>。</p>
</td>
</tr>
<tr id="row184011636114710"><td class="cellrowborder" valign="top" width="50%" headers="mcps1.2.3.1.1 "><p id="p54011336104715"><a name="p54011336104715"></a><a name="p54011336104715"></a><a href="__float2ull_rz.md">__float2ull_rz</a></p>
</td>
<td class="cellrowborder" valign="top" width="50%" headers="mcps1.2.3.1.2 "><p id="p1564219586217"><a name="p1564219586217"></a><a name="p1564219586217"></a><span>将浮点数转换为向零舍入的64位无符号整数</span>。</p>
</td>
</tr>
<tr id="row18401183624714"><td class="cellrowborder" valign="top" width="50%" headers="mcps1.2.3.1.1 "><p id="p15401163624710"><a name="p15401163624710"></a><a name="p15401163624710"></a><a href="__float2ull_rd.md">__float2ull_rd</a></p>
</td>
<td class="cellrowborder" valign="top" width="50%" headers="mcps1.2.3.1.2 "><p id="p13581991138"><a name="p13581991138"></a><a name="p13581991138"></a><span>将浮点数转换为向下取整的64位无符号整数</span>。</p>
</td>
</tr>
<tr id="row74011836164719"><td class="cellrowborder" valign="top" width="50%" headers="mcps1.2.3.1.1 "><p id="p440123604719"><a name="p440123604719"></a><a name="p440123604719"></a><a href="__float2ull_ru.md">__float2ull_ru</a></p>
</td>
<td class="cellrowborder" valign="top" width="50%" headers="mcps1.2.3.1.2 "><p id="p16717163465516"><a name="p16717163465516"></a><a name="p16717163465516"></a><span>将浮点数转换为向上取整的64位无符号整数</span>。</p>
</td>
</tr>
<tr id="row21671443142411"><td class="cellrowborder" valign="top" width="50%" headers="mcps1.2.3.1.1 "><p id="p1616824392417"><a name="p1616824392417"></a><a name="p1616824392417"></a><a href="__float2ull_rna.md">__float2ull_rna</a></p>
</td>
<td class="cellrowborder" valign="top" width="50%" headers="mcps1.2.3.1.2 "><p id="p14168643162411"><a name="p14168643162411"></a><a name="p14168643162411"></a><span>将浮点数转换为向远离零舍入的64位无符号整数</span>。</p>
</td>
</tr>
<tr id="row204016361478"><td class="cellrowborder" valign="top" width="50%" headers="mcps1.2.3.1.1 "><p id="p12401103614471"><a name="p12401103614471"></a><a name="p12401103614471"></a><a href="__float2ll_rn.md">__float2ll_rn</a></p>
</td>
<td class="cellrowborder" valign="top" width="50%" headers="mcps1.2.3.1.2 "><p id="p16717113465513"><a name="p16717113465513"></a><a name="p16717113465513"></a><span>将浮点数转换为有符号64位整数，并四舍五入到最接近的偶数</span>。</p>
</td>
</tr>
<tr id="row1340115369475"><td class="cellrowborder" valign="top" width="50%" headers="mcps1.2.3.1.1 "><p id="p1940143654717"><a name="p1940143654717"></a><a name="p1940143654717"></a><a href="__float2ll_rz.md">__float2ll_rz</a></p>
</td>
<td class="cellrowborder" valign="top" width="50%" headers="mcps1.2.3.1.2 "><p id="p1871783417551"><a name="p1871783417551"></a><a name="p1871783417551"></a><span>将浮点数转换为向零舍入的64位有符号整数</span>。</p>
</td>
</tr>
<tr id="row10401536134711"><td class="cellrowborder" valign="top" width="50%" headers="mcps1.2.3.1.1 "><p id="p194014360474"><a name="p194014360474"></a><a name="p194014360474"></a><a href="__float2ll_rd.md">__float2ll_rd</a></p>
</td>
<td class="cellrowborder" valign="top" width="50%" headers="mcps1.2.3.1.2 "><p id="p2608200144"><a name="p2608200144"></a><a name="p2608200144"></a><span>将浮点数转换为向下取整的64位有符号整数</span>。</p>
</td>
</tr>
<tr id="row4401736144711"><td class="cellrowborder" valign="top" width="50%" headers="mcps1.2.3.1.1 "><p id="p15401133664711"><a name="p15401133664711"></a><a name="p15401133664711"></a><a href="__float2ll_ru.md">__float2ll_ru</a></p>
</td>
<td class="cellrowborder" valign="top" width="50%" headers="mcps1.2.3.1.2 "><p id="p116347817420"><a name="p116347817420"></a><a name="p116347817420"></a><span>将浮点数转换为向上取整的64位有符号整数。</span></p>
</td>
</tr>
<tr id="row1371134802413"><td class="cellrowborder" valign="top" width="50%" headers="mcps1.2.3.1.1 "><p id="p123711848132412"><a name="p123711848132412"></a><a name="p123711848132412"></a><a href="__float2ll_rna.md">__float2ll_rna</a></p>
</td>
<td class="cellrowborder" valign="top" width="50%" headers="mcps1.2.3.1.2 "><p id="p83711348192417"><a name="p83711348192417"></a><a name="p83711348192417"></a><span>将浮点数转换为向远离零舍入的64位有符号整数</span>。</p>
</td>
</tr>
<tr id="row16591213181115"><td class="cellrowborder" valign="top" width="50%" headers="mcps1.2.3.1.1 "><p id="p1759210138116"><a name="p1759210138116"></a><a name="p1759210138116"></a><a href="__float22half2_rn_sat.md">__float22half2_rn_sat</a></p>
</td>
<td class="cellrowborder" valign="top" width="50%" headers="mcps1.2.3.1.2 "><p id="p16592131314114"><a name="p16592131314114"></a><a name="p16592131314114"></a>饱和模式下将输入的两个分量遵循<span id="text1032751817488"><a name="text1032751817488"></a><a name="text1032751817488"></a>CAST_RINT</span>模式转换成的half2类型数据。</p>
</td>
</tr>
<tr id="row1548415401236"><td class="cellrowborder" valign="top" width="50%" headers="mcps1.2.3.1.1 "><p id="p34841401311"><a name="p34841401311"></a><a name="p34841401311"></a><a href="__float22half2_rz.md">__float22half2_rz</a></p>
</td>
<td class="cellrowborder" valign="top" width="50%" headers="mcps1.2.3.1.2 "><p id="p1484184016312"><a name="p1484184016312"></a><a name="p1484184016312"></a>将输入的两个分量遵循CAST_TRUNC模式转换成的half2类型数据。</p>
</td>
</tr>
<tr id="row76020474313"><td class="cellrowborder" valign="top" width="50%" headers="mcps1.2.3.1.1 "><p id="p060214471734"><a name="p060214471734"></a><a name="p060214471734"></a><a href="__float22half2_rz_sat.md">__float22half2_rz_sat</a></p>
</td>
<td class="cellrowborder" valign="top" width="50%" headers="mcps1.2.3.1.2 "><p id="p166021747733"><a name="p166021747733"></a><a name="p166021747733"></a>饱和模式下将输入的两个分量遵循CAST_TRUNC模式转换成的half2类型数据。</p>
</td>
</tr>
<tr id="row31200514319"><td class="cellrowborder" valign="top" width="50%" headers="mcps1.2.3.1.1 "><p id="p21209511319"><a name="p21209511319"></a><a name="p21209511319"></a><a href="__float22half2_rd.md">__float22half2_rd</a></p>
</td>
<td class="cellrowborder" valign="top" width="50%" headers="mcps1.2.3.1.2 "><p id="p712019511437"><a name="p712019511437"></a><a name="p712019511437"></a>将输入的两个分量遵循CAST_FLOOR模式转换成的half2类型数据。</p>
</td>
</tr>
<tr id="row89311041101017"><td class="cellrowborder" valign="top" width="50%" headers="mcps1.2.3.1.1 "><p id="p1558823061114"><a name="p1558823061114"></a><a name="p1558823061114"></a><a href="__float22half2_rd_sat.md">__float22half2_rd_sat</a></p>
</td>
<td class="cellrowborder" valign="top" width="50%" headers="mcps1.2.3.1.2 "><p id="p159328411108"><a name="p159328411108"></a><a name="p159328411108"></a>饱和模式下将输入的两个分量遵循CAST_FLOOR模式转换成的half2类型数据。</p>
</td>
</tr>
<tr id="row153834813101"><td class="cellrowborder" valign="top" width="50%" headers="mcps1.2.3.1.1 "><p id="p11128123411113"><a name="p11128123411113"></a><a name="p11128123411113"></a><a href="__float22half2_ru.md">__float22half2_ru</a></p>
</td>
<td class="cellrowborder" valign="top" width="50%" headers="mcps1.2.3.1.2 "><p id="p1539194831013"><a name="p1539194831013"></a><a name="p1539194831013"></a>将输入的两个分量遵循CAST_CEIL模式转换成的half2类型数据。</p>
</td>
</tr>
<tr id="row112094661020"><td class="cellrowborder" valign="top" width="50%" headers="mcps1.2.3.1.1 "><p id="p73650391114"><a name="p73650391114"></a><a name="p73650391114"></a><a href="__float22half2_ru_sat.md">__float22half2_ru_sat</a></p>
</td>
<td class="cellrowborder" valign="top" width="50%" headers="mcps1.2.3.1.2 "><p id="p18897145344916"><a name="p18897145344916"></a><a name="p18897145344916"></a>饱和模式下将输入的两个分量遵循CAST_CEIL模式转换成的half2类型数据。</p>
</td>
</tr>
<tr id="row6387184912319"><td class="cellrowborder" valign="top" width="50%" headers="mcps1.2.3.1.1 "><p id="p163871649835"><a name="p163871649835"></a><a name="p163871649835"></a><a href="__float22half2_rna.md">__float22half2_rna</a></p>
</td>
<td class="cellrowborder" valign="top" width="50%" headers="mcps1.2.3.1.2 "><p id="p203873491037"><a name="p203873491037"></a><a name="p203873491037"></a>将输入的两个分量遵循CAST_ROUND模式转换成的half2类型数据。</p>
</td>
</tr>
<tr id="row16222124411016"><td class="cellrowborder" valign="top" width="50%" headers="mcps1.2.3.1.1 "><p id="p11222344171019"><a name="p11222344171019"></a><a name="p11222344171019"></a><a href="__float22half2_rna_sat.md">__float22half2_rna_sat</a></p>
</td>
<td class="cellrowborder" valign="top" width="50%" headers="mcps1.2.3.1.2 "><p id="p822244491013"><a name="p822244491013"></a><a name="p822244491013"></a>饱和模式下将输入的两个分量遵循CAST_ROUND模式转换成的half2类型数据。</p>
</td>
</tr>
<tr id="row1870584512318"><td class="cellrowborder" valign="top" width="50%" headers="mcps1.2.3.1.1 "><p id="p14705134518317"><a name="p14705134518317"></a><a name="p14705134518317"></a><a href="data01-PA_TEMP-104-zh-cn_bookmap_0000002554332349260303092906168-temp-zh-cn_topic_0000002523343898.md">__float22half2_ro</a></p>
</td>
<td class="cellrowborder" valign="top" width="50%" headers="mcps1.2.3.1.2 "><p id="p187058451932"><a name="p187058451932"></a><a name="p187058451932"></a>将输入的两个分量遵循CAST_ODD模式转换成的half2类型数据。</p>
</td>
</tr>
<tr id="row856110522104"><td class="cellrowborder" valign="top" width="50%" headers="mcps1.2.3.1.1 "><p id="p1956125221011"><a name="p1956125221011"></a><a name="p1956125221011"></a><a href="__float22half2_ro_sat.md">__float22half2_ro_sat</a></p>
</td>
<td class="cellrowborder" valign="top" width="50%" headers="mcps1.2.3.1.2 "><p id="p356175211109"><a name="p356175211109"></a><a name="p356175211109"></a>饱和模式下将输入的两个分量遵循CAST_ODD模式转换成的half2类型数据。</p>
</td>
</tr>
<tr id="row9560718202218"><td class="cellrowborder" valign="top" width="50%" headers="mcps1.2.3.1.1 "><p id="p16561111818228"><a name="p16561111818228"></a><a name="p16561111818228"></a><a href="__float22bfloat162_rn_sat.md">__float22bfloat162_rn_sat</a></p>
</td>
<td class="cellrowborder" valign="top" width="50%" headers="mcps1.2.3.1.2 "><p id="p18561101812218"><a name="p18561101812218"></a><a name="p18561101812218"></a>饱和模式下将输入的两个分量遵循CAST_RINT模式转换成的bfloat16x2_t类型数据。</p>
</td>
</tr>
<tr id="row19789192002213"><td class="cellrowborder" valign="top" width="50%" headers="mcps1.2.3.1.1 "><p id="p1579082010223"><a name="p1579082010223"></a><a name="p1579082010223"></a><a href="__float22bfloat162_rz.md">__float22bfloat162_rz</a></p>
</td>
<td class="cellrowborder" valign="top" width="50%" headers="mcps1.2.3.1.2 "><p id="p67901720122210"><a name="p67901720122210"></a><a name="p67901720122210"></a>将输入的两个分量遵循CAST_TRUNC模式转换成的bfloat16x2_t类型数据。</p>
</td>
</tr>
<tr id="row10297738162216"><td class="cellrowborder" valign="top" width="50%" headers="mcps1.2.3.1.1 "><p id="p11297183882214"><a name="p11297183882214"></a><a name="p11297183882214"></a><a href="__float22bfloat162_rz_sat.md">__float22bfloat162_rz_sat</a></p>
</td>
<td class="cellrowborder" valign="top" width="50%" headers="mcps1.2.3.1.2 "><p id="p102981438192216"><a name="p102981438192216"></a><a name="p102981438192216"></a>饱和模式下将输入的两个分量遵循CAST_TRUNC模式转换成的bfloat16x2_t类型数据。</p>
</td>
</tr>
<tr id="row229413268222"><td class="cellrowborder" valign="top" width="50%" headers="mcps1.2.3.1.1 "><p id="p19295426142219"><a name="p19295426142219"></a><a name="p19295426142219"></a><a href="__float22bfloat162_rd.md">__float22bfloat162_rd</a></p>
</td>
<td class="cellrowborder" valign="top" width="50%" headers="mcps1.2.3.1.2 "><p id="p1429522632214"><a name="p1429522632214"></a><a name="p1429522632214"></a>将输入的两个分量遵循CAST_FLOOR模式转换成的bfloat16x2_t类型数据。</p>
</td>
</tr>
<tr id="row370883514221"><td class="cellrowborder" valign="top" width="50%" headers="mcps1.2.3.1.1 "><p id="p11708735132212"><a name="p11708735132212"></a><a name="p11708735132212"></a><a href="__float22bfloat162_rd_sat.md">__float22bfloat162_rd_sat</a></p>
</td>
<td class="cellrowborder" valign="top" width="50%" headers="mcps1.2.3.1.2 "><p id="p270953518223"><a name="p270953518223"></a><a name="p270953518223"></a>饱和模式下将输入的两个分量遵循CAST_FLOOR模式转换成的bfloat16x2_t类型数据。</p>
</td>
</tr>
<tr id="row138597334225"><td class="cellrowborder" valign="top" width="50%" headers="mcps1.2.3.1.1 "><p id="p586003314221"><a name="p586003314221"></a><a name="p586003314221"></a><a href="__float22bfloat162_ru.md">__float22bfloat162_ru</a></p>
</td>
<td class="cellrowborder" valign="top" width="50%" headers="mcps1.2.3.1.2 "><p id="p188611633162216"><a name="p188611633162216"></a><a name="p188611633162216"></a>将输入的两个分量遵循CAST_CEIL模式转换成的bfloat16x2_t类型数据。</p>
</td>
</tr>
<tr id="row155401024182219"><td class="cellrowborder" valign="top" width="50%" headers="mcps1.2.3.1.1 "><p id="p25401924122219"><a name="p25401924122219"></a><a name="p25401924122219"></a><a href="__float22bfloat162_ru_sat.md">__float22bfloat162_ru_sat</a></p>
</td>
<td class="cellrowborder" valign="top" width="50%" headers="mcps1.2.3.1.2 "><p id="p654182415225"><a name="p654182415225"></a><a name="p654182415225"></a>饱和模式下将输入的两个分量遵循CAST_CEIL模式转换成的bfloat16x2_t类型数据。</p>
</td>
</tr>
<tr id="row1562613114229"><td class="cellrowborder" valign="top" width="50%" headers="mcps1.2.3.1.1 "><p id="p6627163114229"><a name="p6627163114229"></a><a name="p6627163114229"></a><a href="__float22bfloat162_rna.md">__float22bfloat162_rna</a></p>
</td>
<td class="cellrowborder" valign="top" width="50%" headers="mcps1.2.3.1.2 "><p id="p196271531172217"><a name="p196271531172217"></a><a name="p196271531172217"></a>将输入的两个分量遵循CAST_ROUND模式转换成的bfloat16x2_t类型数据。</p>
</td>
</tr>
<tr id="row146121722182217"><td class="cellrowborder" valign="top" width="50%" headers="mcps1.2.3.1.1 "><p id="p5612192272214"><a name="p5612192272214"></a><a name="p5612192272214"></a><a href="__float22bfloat162_rna_sat.md">__float22bfloat162_rna_sat</a></p>
</td>
<td class="cellrowborder" valign="top" width="50%" headers="mcps1.2.3.1.2 "><p id="p1661218225229"><a name="p1661218225229"></a><a name="p1661218225229"></a>饱和模式下将输入的两个分量遵循CAST_ROUND模式转换成的bfloat16x2_t类型数据。</p>
</td>
</tr>
<tr id="row1269564915388"><td class="cellrowborder" valign="top" width="50%" headers="mcps1.2.3.1.1 "><p id="p6698134912385"><a name="p6698134912385"></a><a name="p6698134912385"></a><a href="__float22hif82_rna.md">__float22hif82_rna</a></p>
</td>
<td class="cellrowborder" valign="top" width="50%" headers="mcps1.2.3.1.2 "><p id="p4699134963816"><a name="p4699134963816"></a><a name="p4699134963816"></a>将输入的两个分量遵循CAST_ROUND模式转换成的hifloat8x2_t类型数据。</p>
</td>
</tr>
<tr id="row819895210381"><td class="cellrowborder" valign="top" width="50%" headers="mcps1.2.3.1.1 "><p id="p12199165217389"><a name="p12199165217389"></a><a name="p12199165217389"></a><a href="__float22hif82_rna_sat.md">__float22hif82_rna_sat</a></p>
</td>
<td class="cellrowborder" valign="top" width="50%" headers="mcps1.2.3.1.2 "><p id="p419916525382"><a name="p419916525382"></a><a name="p419916525382"></a>饱和模式下将输入的两个分量遵循CAST_ROUND模式转换成的hifloat8x2_t类型数据。</p>
</td>
</tr>
<tr id="row0293356113816"><td class="cellrowborder" valign="top" width="50%" headers="mcps1.2.3.1.1 "><p id="p14294256123818"><a name="p14294256123818"></a><a name="p14294256123818"></a><a href="__float22hif82_rh.md">__float22hif82_rh</a></p>
</td>
<td class="cellrowborder" valign="top" width="50%" headers="mcps1.2.3.1.2 "><p id="p17294656203815"><a name="p17294656203815"></a><a name="p17294656203815"></a>将输入的两个分量遵循CAST_HYBRID模式转换成的hifloat8x2_t类型数据。</p>
</td>
</tr>
<tr id="row6174115417389"><td class="cellrowborder" valign="top" width="50%" headers="mcps1.2.3.1.1 "><p id="p8174754153810"><a name="p8174754153810"></a><a name="p8174754153810"></a><a href="__float22hif82_rh_sat.md">__float22hif82_rh_sat</a></p>
</td>
<td class="cellrowborder" valign="top" width="50%" headers="mcps1.2.3.1.2 "><p id="p2017513545384"><a name="p2017513545384"></a><a name="p2017513545384"></a>饱和模式下将输入的两个分量遵循CAST_HYBRID模式转换成的hifloat8x2_t类型数据。</p>
</td>
</tr>
<tr id="row207831630194319"><td class="cellrowborder" valign="top" width="50%" headers="mcps1.2.3.1.1 "><p id="p1783153094318"><a name="p1783153094318"></a><a name="p1783153094318"></a><a href="__asc_cvt_float2_to_fp8x2.md">__asc_cvt_float2_to_fp8x2</a></p>
</td>
<td class="cellrowborder" valign="top" width="50%" headers="mcps1.2.3.1.2 "><p id="p11784123024317"><a name="p11784123024317"></a><a name="p11784123024317"></a>输入的两个分量遵循CAST_RINT模式，根据指定的8位浮点数类型和指定的饱和模式，转换成的__asc_fp8x2_storage_t类型数据。</p>
</td>
</tr>
<tr id="row1140113634719"><td class="cellrowborder" valign="top" width="50%" headers="mcps1.2.3.1.1 "><p id="p84021236104720"><a name="p84021236104720"></a><a name="p84021236104720"></a><a href="__half2float.md">__half2float</a></p>
</td>
<td class="cellrowborder" valign="top" width="50%" headers="mcps1.2.3.1.2 "><p id="p83569314412"><a name="p83569314412"></a><a name="p83569314412"></a>将half转换为浮点数。</p>
</td>
</tr>
<tr id="row206234344315"><td class="cellrowborder" valign="top" width="50%" headers="mcps1.2.3.1.1 "><p id="p26230334318"><a name="p26230334318"></a><a name="p26230334318"></a><a href="__half2half_rn.md">__half2half_rn</a></p>
</td>
<td class="cellrowborder" valign="top" width="50%" headers="mcps1.2.3.1.2 "><p id="p1662423114320"><a name="p1662423114320"></a><a name="p1662423114320"></a>输入遵循CAST_RINT模式取整后的half类型数据。</p>
</td>
</tr>
<tr id="row1323315617435"><td class="cellrowborder" valign="top" width="50%" headers="mcps1.2.3.1.1 "><p id="p172331663439"><a name="p172331663439"></a><a name="p172331663439"></a><a href="__half2half_rz.md">__half2half_rz</a></p>
</td>
<td class="cellrowborder" valign="top" width="50%" headers="mcps1.2.3.1.2 "><p id="p142333664318"><a name="p142333664318"></a><a name="p142333664318"></a>输入遵循CAST_TRUNC模式取整后的half类型数据。</p>
</td>
</tr>
<tr id="row1566788124315"><td class="cellrowborder" valign="top" width="50%" headers="mcps1.2.3.1.1 "><p id="p1366710817435"><a name="p1366710817435"></a><a name="p1366710817435"></a><a href="__half2half_rd.md">__half2half_rd</a></p>
</td>
<td class="cellrowborder" valign="top" width="50%" headers="mcps1.2.3.1.2 "><p id="p266788184315"><a name="p266788184315"></a><a name="p266788184315"></a>输入遵循CAST_FLOOR模式取整后的half类型数据。</p>
</td>
</tr>
<tr id="row166381912164316"><td class="cellrowborder" valign="top" width="50%" headers="mcps1.2.3.1.1 "><p id="p8638812154313"><a name="p8638812154313"></a><a name="p8638812154313"></a><a href="__half2half_ru.md">__half2half_ru</a></p>
</td>
<td class="cellrowborder" valign="top" width="50%" headers="mcps1.2.3.1.2 "><p id="p663861234310"><a name="p663861234310"></a><a name="p663861234310"></a>输入遵循CAST_CEIL模式取整后的half类型数据。</p>
</td>
</tr>
<tr id="row116573101438"><td class="cellrowborder" valign="top" width="50%" headers="mcps1.2.3.1.1 "><p id="p8657111094313"><a name="p8657111094313"></a><a name="p8657111094313"></a><a href="__half2half_rna.md">__half2half_rna</a></p>
</td>
<td class="cellrowborder" valign="top" width="50%" headers="mcps1.2.3.1.2 "><p id="p26584102436"><a name="p26584102436"></a><a name="p26584102436"></a>输入遵循CAST_ROUND模式取整后的half类型数据。</p>
</td>
</tr>
<tr id="row154021136174719"><td class="cellrowborder" valign="top" width="50%" headers="mcps1.2.3.1.1 "><p id="p164027364476"><a name="p164027364476"></a><a name="p164027364476"></a><a href="__half2bfloat16_rn.md">__half2bfloat16_rn</a></p>
</td>
<td class="cellrowborder" valign="top" width="50%" headers="mcps1.2.3.1.2 "><p id="p16180134412418"><a name="p16180134412418"></a><a name="p16180134412418"></a><span>将half转换为bfloat16，并四舍五入到最接近的偶数</span>。</p>
</td>
</tr>
<tr id="row12402336184711"><td class="cellrowborder" valign="top" width="50%" headers="mcps1.2.3.1.1 "><p id="p240293634717"><a name="p240293634717"></a><a name="p240293634717"></a><a href="__half2bfloat16_rz.md">__half2bfloat16_rz</a></p>
</td>
<td class="cellrowborder" valign="top" width="50%" headers="mcps1.2.3.1.2 "><p id="p15718103495514"><a name="p15718103495514"></a><a name="p15718103495514"></a><span>将half 转换为向零舍入的bfloat16</span>。</p>
</td>
</tr>
<tr id="row13402123617477"><td class="cellrowborder" valign="top" width="50%" headers="mcps1.2.3.1.1 "><p id="p1040217367478"><a name="p1040217367478"></a><a name="p1040217367478"></a><a href="__half2bfloat16_rd.md">__half2bfloat16_rd</a></p>
</td>
<td class="cellrowborder" valign="top" width="50%" headers="mcps1.2.3.1.2 "><p id="p971814344550"><a name="p971814344550"></a><a name="p971814344550"></a><span>将half转换为向下取整的bfloat16</span>。</p>
</td>
</tr>
<tr id="row4402193619479"><td class="cellrowborder" valign="top" width="50%" headers="mcps1.2.3.1.1 "><p id="p164028361472"><a name="p164028361472"></a><a name="p164028361472"></a><a href="__half2bfloat16_ru.md">__half2bfloat16_ru</a></p>
</td>
<td class="cellrowborder" valign="top" width="50%" headers="mcps1.2.3.1.2 "><p id="p1171817342554"><a name="p1171817342554"></a><a name="p1171817342554"></a><span>将half转换为bfloat16（向上取整）。</span></p>
</td>
</tr>
<tr id="row336017524249"><td class="cellrowborder" valign="top" width="50%" headers="mcps1.2.3.1.1 "><p id="p1636015521246"><a name="p1636015521246"></a><a name="p1636015521246"></a><a href="__half2bfloat16_rna.md">__half2bfloat16_rna</a></p>
</td>
<td class="cellrowborder" valign="top" width="50%" headers="mcps1.2.3.1.2 "><p id="p536119526242"><a name="p536119526242"></a><a name="p536119526242"></a><span>将half类型数据转换为向远离零舍入的bfloat16类型数据</span>。</p>
</td>
</tr>
<tr id="row1840223664715"><td class="cellrowborder" valign="top" width="50%" headers="mcps1.2.3.1.1 "><p id="p940253618477"><a name="p940253618477"></a><a name="p940253618477"></a><a href="__half2uint_rn.md">__half2uint_rn</a></p>
</td>
<td class="cellrowborder" valign="top" width="50%" headers="mcps1.2.3.1.2 "><p id="p7171025654"><a name="p7171025654"></a><a name="p7171025654"></a><span>将half转换为无符号整数，并四舍五入到最接近的偶数。</span></p>
</td>
</tr>
<tr id="row12402736114712"><td class="cellrowborder" valign="top" width="50%" headers="mcps1.2.3.1.1 "><p id="p087218441464"><a name="p087218441464"></a><a name="p087218441464"></a><a href="__half2uint_rz.md">__half2uint_rz</a></p>
</td>
<td class="cellrowborder" valign="top" width="50%" headers="mcps1.2.3.1.2 "><p id="p187180342555"><a name="p187180342555"></a><a name="p187180342555"></a><span>将half转换为向零舍入的无符号整数。</span></p>
</td>
</tr>
<tr id="row114024363478"><td class="cellrowborder" valign="top" width="50%" headers="mcps1.2.3.1.1 "><p id="p1040253613476"><a name="p1040253613476"></a><a name="p1040253613476"></a><a href="__half2uint_rd.md">__half2uint_rd</a></p>
</td>
<td class="cellrowborder" valign="top" width="50%" headers="mcps1.2.3.1.2 "><p id="p967713382518"><a name="p967713382518"></a><a name="p967713382518"></a><span>将half转换为向下取整的无符号整数。</span></p>
</td>
</tr>
<tr id="row44022036154710"><td class="cellrowborder" valign="top" width="50%" headers="mcps1.2.3.1.1 "><p id="p164026367478"><a name="p164026367478"></a><a name="p164026367478"></a><a href="__half2uint_ru.md">__half2uint_ru</a></p>
</td>
<td class="cellrowborder" valign="top" width="50%" headers="mcps1.2.3.1.2 "><p id="p1219512451458"><a name="p1219512451458"></a><a name="p1219512451458"></a><span>将half转换为向上取整的无符号整数。</span></p>
</td>
</tr>
<tr id="row1467145542417"><td class="cellrowborder" valign="top" width="50%" headers="mcps1.2.3.1.1 "><p id="p18671145510249"><a name="p18671145510249"></a><a name="p18671145510249"></a><a href="__half2uint_rna.md">__half2uint_rna</a></p>
</td>
<td class="cellrowborder" valign="top" width="50%" headers="mcps1.2.3.1.2 "><p id="p1067165562413"><a name="p1067165562413"></a><a name="p1067165562413"></a><span>将half类型数据转换为向远离零舍入的无符号整数。</span></p>
</td>
</tr>
<tr id="row20403133654711"><td class="cellrowborder" valign="top" width="50%" headers="mcps1.2.3.1.1 "><p id="p54035364478"><a name="p54035364478"></a><a name="p54035364478"></a><a href="__half2int_rn.md">__half2int_rn</a></p>
</td>
<td class="cellrowborder" valign="top" width="50%" headers="mcps1.2.3.1.2 "><p id="p1844185219512"><a name="p1844185219512"></a><a name="p1844185219512"></a><span>将half转换为有符号整数，并四舍五入到最接近的偶数。</span></p>
</td>
</tr>
<tr id="row2040373674710"><td class="cellrowborder" valign="top" width="50%" headers="mcps1.2.3.1.1 "><p id="p15403153684711"><a name="p15403153684711"></a><a name="p15403153684711"></a><a href="__half2int_rz.md">__half2int_rz</a></p>
</td>
<td class="cellrowborder" valign="top" width="50%" headers="mcps1.2.3.1.2 "><p id="p1871943415554"><a name="p1871943415554"></a><a name="p1871943415554"></a><span>将half 转换为向零舍入的有符号整数。</span></p>
</td>
</tr>
<tr id="row340313361479"><td class="cellrowborder" valign="top" width="50%" headers="mcps1.2.3.1.1 "><p id="p13403163615474"><a name="p13403163615474"></a><a name="p13403163615474"></a><a href="__half2int_rd.md">__half2int_rd</a></p>
</td>
<td class="cellrowborder" valign="top" width="50%" headers="mcps1.2.3.1.2 "><p id="p3863861614"><a name="p3863861614"></a><a name="p3863861614"></a><span>将half转换为向下取整的有符号整数。</span></p>
</td>
</tr>
<tr id="row34031436134711"><td class="cellrowborder" valign="top" width="50%" headers="mcps1.2.3.1.1 "><p id="p340393612478"><a name="p340393612478"></a><a name="p340393612478"></a><a href="__half2int_ru.md">__half2int_ru</a></p>
</td>
<td class="cellrowborder" valign="top" width="50%" headers="mcps1.2.3.1.2 "><p id="p971923425519"><a name="p971923425519"></a><a name="p971923425519"></a><span>将half转换为有符号整数（向上取整）。</span></p>
</td>
</tr>
<tr id="row13707758172410"><td class="cellrowborder" valign="top" width="50%" headers="mcps1.2.3.1.1 "><p id="p170715817249"><a name="p170715817249"></a><a name="p170715817249"></a><a href="__half2int_rna.md">__half2int_rna</a></p>
</td>
<td class="cellrowborder" valign="top" width="50%" headers="mcps1.2.3.1.2 "><p id="p1470795813242"><a name="p1470795813242"></a><a name="p1470795813242"></a><span>将half类型数据转换为向远离零舍入的有符号整数。</span></p>
</td>
</tr>
<tr id="row14403193613479"><td class="cellrowborder" valign="top" width="50%" headers="mcps1.2.3.1.1 "><p id="p14403193624713"><a name="p14403193624713"></a><a name="p14403193624713"></a><a href="__half2ull_rn.md">__half2ull_rn</a></p>
</td>
<td class="cellrowborder" valign="top" width="50%" headers="mcps1.2.3.1.2 "><p id="p10203102316617"><a name="p10203102316617"></a><a name="p10203102316617"></a><span>将half转换为无符号64位整数，并四舍五入到最接近的偶数</span>。</p>
</td>
</tr>
<tr id="row5403153615472"><td class="cellrowborder" valign="top" width="50%" headers="mcps1.2.3.1.1 "><p id="p18403183611479"><a name="p18403183611479"></a><a name="p18403183611479"></a><a href="__half2ull_rz.md">__half2ull_rz</a></p>
</td>
<td class="cellrowborder" valign="top" width="50%" headers="mcps1.2.3.1.2 "><p id="p2072013347559"><a name="p2072013347559"></a><a name="p2072013347559"></a><span>将half转换为向零舍入的64位无符号整数。</span></p>
</td>
</tr>
<tr id="row174034361476"><td class="cellrowborder" valign="top" width="50%" headers="mcps1.2.3.1.1 "><p id="p184033366476"><a name="p184033366476"></a><a name="p184033366476"></a><a href="__half2ull_rd.md">__half2ull_rd</a></p>
</td>
<td class="cellrowborder" valign="top" width="50%" headers="mcps1.2.3.1.2 "><p id="p147203340553"><a name="p147203340553"></a><a name="p147203340553"></a><span>将half转换为向下取整的64位无符号整数。</span></p>
</td>
</tr>
<tr id="row940316361477"><td class="cellrowborder" valign="top" width="50%" headers="mcps1.2.3.1.1 "><p id="p1740312366473"><a name="p1740312366473"></a><a name="p1740312366473"></a><a href="__half2ull_ru.md">__half2ull_ru</a></p>
</td>
<td class="cellrowborder" valign="top" width="50%" headers="mcps1.2.3.1.2 "><p id="p157201434125515"><a name="p157201434125515"></a><a name="p157201434125515"></a><span>将half 转换为向上取整的64位无符号整数。</span></p>
</td>
</tr>
<tr id="row19426152152519"><td class="cellrowborder" valign="top" width="50%" headers="mcps1.2.3.1.1 "><p id="p842616213251"><a name="p842616213251"></a><a name="p842616213251"></a><a href="__half2ull_rna.md">__half2ull_rna</a></p>
</td>
<td class="cellrowborder" valign="top" width="50%" headers="mcps1.2.3.1.2 "><p id="p184261124259"><a name="p184261124259"></a><a name="p184261124259"></a><span>将half类型数据转换为向远离零舍入的64位无符号整数。</span></p>
</td>
</tr>
<tr id="row10403103615478"><td class="cellrowborder" valign="top" width="50%" headers="mcps1.2.3.1.1 "><p id="p4403123617475"><a name="p4403123617475"></a><a name="p4403123617475"></a><a href="__half2ll_rn.md">__half2ll_rn</a></p>
</td>
<td class="cellrowborder" valign="top" width="50%" headers="mcps1.2.3.1.2 "><p id="p447114529614"><a name="p447114529614"></a><a name="p447114529614"></a><span>将half转换为有符号64位整数，并四舍五入到最接近的偶数位。</span></p>
</td>
</tr>
<tr id="row154031036154717"><td class="cellrowborder" valign="top" width="50%" headers="mcps1.2.3.1.1 "><p id="p6403736174713"><a name="p6403736174713"></a><a name="p6403736174713"></a><a href="__half2ll_rz.md">__half2ll_rz</a></p>
</td>
<td class="cellrowborder" valign="top" width="50%" headers="mcps1.2.3.1.2 "><p id="p13720103455512"><a name="p13720103455512"></a><a name="p13720103455512"></a><span>将half转换为向零舍入的64位有符号整数。</span></p>
</td>
</tr>
<tr id="row1740343654718"><td class="cellrowborder" valign="top" width="50%" headers="mcps1.2.3.1.1 "><p id="p15403153684713"><a name="p15403153684713"></a><a name="p15403153684713"></a><a href="__half2ll_rd.md">__half2ll_rd</a></p>
</td>
<td class="cellrowborder" valign="top" width="50%" headers="mcps1.2.3.1.2 "><p id="p1439876272"><a name="p1439876272"></a><a name="p1439876272"></a><span>将half转换为向下取整的64位有符号整数。</span></p>
</td>
</tr>
<tr id="row1740415365473"><td class="cellrowborder" valign="top" width="50%" headers="mcps1.2.3.1.1 "><p id="p340433634711"><a name="p340433634711"></a><a name="p340433634711"></a><a href="__half2ll_ru.md">__half2ll_ru</a></p>
</td>
<td class="cellrowborder" valign="top" width="50%" headers="mcps1.2.3.1.2 "><p id="p69073111576"><a name="p69073111576"></a><a name="p69073111576"></a><span>将half 转换为向上取整的64位有符号整数。</span></p>
</td>
</tr>
<tr id="row6350196192514"><td class="cellrowborder" valign="top" width="50%" headers="mcps1.2.3.1.1 "><p id="p16350136102510"><a name="p16350136102510"></a><a name="p16350136102510"></a><a href="__half2ll_rna.md">__half2ll_rna</a></p>
</td>
<td class="cellrowborder" valign="top" width="50%" headers="mcps1.2.3.1.2 "><p id="p635010614253"><a name="p635010614253"></a><a name="p635010614253"></a><span>将half类型数据转换为向远离零舍入的64位有符号整数。</span></p>
</td>
</tr>
<tr id="row3453368914"><td class="cellrowborder" valign="top" width="50%" headers="mcps1.2.3.1.1 "><p id="p54614369917"><a name="p54614369917"></a><a name="p54614369917"></a><a href="__half22hif82_rna.md">__half22hif82_rna</a></p>
</td>
<td class="cellrowborder" valign="top" width="50%" headers="mcps1.2.3.1.2 "><p id="p7141191914359"><a name="p7141191914359"></a><a name="p7141191914359"></a>将输入的两个分量遵循CAST_ROUND模式转换成的hifloat8x2_t类型数据。</p>
</td>
</tr>
<tr id="row21711839692"><td class="cellrowborder" valign="top" width="50%" headers="mcps1.2.3.1.1 "><p id="p171713392918"><a name="p171713392918"></a><a name="p171713392918"></a><a href="__half22hif82_rna_sat.md">__half22hif82_rna_sat</a></p>
</td>
<td class="cellrowborder" valign="top" width="50%" headers="mcps1.2.3.1.2 "><p id="p15171163910910"><a name="p15171163910910"></a><a name="p15171163910910"></a>饱和模式下将输入的两个分量遵循CAST_ROUND模式转换成的hifloat8x2_t类型数据。</p>
</td>
</tr>
<tr id="row51201944797"><td class="cellrowborder" valign="top" width="50%" headers="mcps1.2.3.1.1 "><p id="p3120114413915"><a name="p3120114413915"></a><a name="p3120114413915"></a><a href="__half22hif82_rh.md">__half22hif82_rh</a></p>
</td>
<td class="cellrowborder" valign="top" width="50%" headers="mcps1.2.3.1.2 "><p id="p171206447913"><a name="p171206447913"></a><a name="p171206447913"></a>将输入的两个分量遵循CAST_HYBRID模式转换成的hifloat8x2_t类型数据。</p>
</td>
</tr>
<tr id="row323018421291"><td class="cellrowborder" valign="top" width="50%" headers="mcps1.2.3.1.1 "><p id="p923011424910"><a name="p923011424910"></a><a name="p923011424910"></a><a href="__half22hif82_rh_sat.md">__half22hif82_rh_sat</a></p>
</td>
<td class="cellrowborder" valign="top" width="50%" headers="mcps1.2.3.1.2 "><p id="p14230842399"><a name="p14230842399"></a><a name="p14230842399"></a>饱和模式下将输入的两个分量遵循CAST_HYBRID模式转换成的hifloat8x2_t类型数据。</p>
</td>
</tr>
<tr id="row16404736164717"><td class="cellrowborder" valign="top" width="50%" headers="mcps1.2.3.1.1 "><p id="p1404036184712"><a name="p1404036184712"></a><a name="p1404036184712"></a><a href="__bfloat162half_rn.md">__bfloat162half_rn</a></p>
</td>
<td class="cellrowborder" valign="top" width="50%" headers="mcps1.2.3.1.2 "><p id="p77202346555"><a name="p77202346555"></a><a name="p77202346555"></a>将bfloat16转换为half，并四舍五入到最接近的偶数，然后返回转换后的值。</p>
</td>
</tr>
<tr id="row230216282448"><td class="cellrowborder" valign="top" width="50%" headers="mcps1.2.3.1.1 "><p id="p1430222854418"><a name="p1430222854418"></a><a name="p1430222854418"></a><a href="__bfloat162half_rn_sat.md">__bfloat162half_rn_sat</a></p>
</td>
<td class="cellrowborder" valign="top" width="50%" headers="mcps1.2.3.1.2 "><p id="p1830218283444"><a name="p1830218283444"></a><a name="p1830218283444"></a>饱和模式下将输入遵循CAST_RINT模式转换成的half类型数据。</p>
</td>
</tr>
<tr id="row14404836184715"><td class="cellrowborder" valign="top" width="50%" headers="mcps1.2.3.1.1 "><p id="p6404163619470"><a name="p6404163619470"></a><a name="p6404163619470"></a><a href="__bfloat162half_rz.md">__bfloat162half_rz</a></p>
</td>
<td class="cellrowborder" valign="top" width="50%" headers="mcps1.2.3.1.2 "><p id="p187201634205518"><a name="p187201634205518"></a><a name="p187201634205518"></a>将bfloat16转换为<span>向零舍入</span>的half。</p>
</td>
</tr>
<tr id="row208513118446"><td class="cellrowborder" valign="top" width="50%" headers="mcps1.2.3.1.1 "><p id="p13861831174415"><a name="p13861831174415"></a><a name="p13861831174415"></a><a href="__bfloat162half_rz_sat.md">__bfloat162half_rz_sat</a></p>
</td>
<td class="cellrowborder" valign="top" width="50%" headers="mcps1.2.3.1.2 "><p id="p6857119114615"><a name="p6857119114615"></a><a name="p6857119114615"></a>饱和模式下将输入遵循CAST_TRUNC模式转换成的half类型数据。</p>
</td>
</tr>
<tr id="row2404183624716"><td class="cellrowborder" valign="top" width="50%" headers="mcps1.2.3.1.1 "><p id="p1940433611475"><a name="p1940433611475"></a><a name="p1940433611475"></a><a href="__bfloat162half_rd.md">__bfloat162half_rd</a></p>
</td>
<td class="cellrowborder" valign="top" width="50%" headers="mcps1.2.3.1.2 "><p id="p1764015335716"><a name="p1764015335716"></a><a name="p1764015335716"></a>将bfloat16转换为<span>向下取整</span>的half。</p>
</td>
</tr>
<tr id="row1163218335443"><td class="cellrowborder" valign="top" width="50%" headers="mcps1.2.3.1.1 "><p id="p1663263304412"><a name="p1663263304412"></a><a name="p1663263304412"></a><a href="__bfloat162half_rd_sat.md">__bfloat162half_rd_sat</a></p>
</td>
<td class="cellrowborder" valign="top" width="50%" headers="mcps1.2.3.1.2 "><p id="p176327339446"><a name="p176327339446"></a><a name="p176327339446"></a>饱和模式下将输入遵循CAST_FLOOR模式转换成的half类型数据。</p>
</td>
</tr>
<tr id="row8404123614711"><td class="cellrowborder" valign="top" width="50%" headers="mcps1.2.3.1.1 "><p id="p10404153619471"><a name="p10404153619471"></a><a name="p10404153619471"></a><a href="__bfloat162half_ru.md">__bfloat162half_ru</a></p>
</td>
<td class="cellrowborder" valign="top" width="50%" headers="mcps1.2.3.1.2 "><p id="p572023415514"><a name="p572023415514"></a><a name="p572023415514"></a>将bfloat16转换为<span>向上取整</span>的half。</p>
</td>
</tr>
<tr id="row10961163512449"><td class="cellrowborder" valign="top" width="50%" headers="mcps1.2.3.1.1 "><p id="p5961153519445"><a name="p5961153519445"></a><a name="p5961153519445"></a><a href="__bfloat162half_ru_sat.md">__bfloat162half_ru_sat</a></p>
</td>
<td class="cellrowborder" valign="top" width="50%" headers="mcps1.2.3.1.2 "><p id="p1118202510469"><a name="p1118202510469"></a><a name="p1118202510469"></a>饱和模式下将输入遵循CAST_CEIL模式转换成的half类型数据。</p>
</td>
</tr>
<tr id="row327416911257"><td class="cellrowborder" valign="top" width="50%" headers="mcps1.2.3.1.1 "><p id="p1274189122515"><a name="p1274189122515"></a><a name="p1274189122515"></a><a href="__bfloat162half_rna.md">__bfloat162half_rna</a></p>
</td>
<td class="cellrowborder" valign="top" width="50%" headers="mcps1.2.3.1.2 "><p id="p227439152518"><a name="p227439152518"></a><a name="p227439152518"></a>将bfloat16类型数据转换为<span>向远离零舍入</span>的half类型数据。</p>
</td>
</tr>
<tr id="row41599387449"><td class="cellrowborder" valign="top" width="50%" headers="mcps1.2.3.1.1 "><p id="p20159103815440"><a name="p20159103815440"></a><a name="p20159103815440"></a><a href="__bfloat162half_rna_sat.md">__bfloat162half_rna_sat</a></p>
</td>
<td class="cellrowborder" valign="top" width="50%" headers="mcps1.2.3.1.2 "><p id="p16882236194610"><a name="p16882236194610"></a><a name="p16882236194610"></a>饱和模式下将输入遵循CAST_ROUND模式转换成的half类型数据。</p>
</td>
</tr>
<tr id="row2404193617476"><td class="cellrowborder" valign="top" width="50%" headers="mcps1.2.3.1.1 "><p id="p1540433674714"><a name="p1540433674714"></a><a name="p1540433674714"></a><a href="__bfloat162float.md">__bfloat162float</a></p>
</td>
<td class="cellrowborder" valign="top" width="50%" headers="mcps1.2.3.1.2 "><p id="p9550347072"><a name="p9550347072"></a><a name="p9550347072"></a>将bfloat16转换为浮点数。</p>
</td>
</tr>
<tr id="row122141447195811"><td class="cellrowborder" valign="top" width="50%" headers="mcps1.2.3.1.1 "><p id="p15214947115810"><a name="p15214947115810"></a><a name="p15214947115810"></a><a href="__bfloat162bfloat16_rn.md">__bfloat162bfloat16_rn</a></p>
</td>
<td class="cellrowborder" valign="top" width="50%" headers="mcps1.2.3.1.2 "><p id="p1921434714582"><a name="p1921434714582"></a><a name="p1921434714582"></a>输入遵循CAST_RINT模式取整后的bfloat16_t类型数据。</p>
</td>
</tr>
<tr id="row178308496585"><td class="cellrowborder" valign="top" width="50%" headers="mcps1.2.3.1.1 "><p id="p1783044995814"><a name="p1783044995814"></a><a name="p1783044995814"></a><a href="__bfloat162bfloat16_rz.md">__bfloat162bfloat16_rz</a></p>
</td>
<td class="cellrowborder" valign="top" width="50%" headers="mcps1.2.3.1.2 "><p id="p2831114965819"><a name="p2831114965819"></a><a name="p2831114965819"></a>输入遵循CAST_TRUNC模式取整后的bfloat16_t类型数据。</p>
</td>
</tr>
<tr id="row516556105811"><td class="cellrowborder" valign="top" width="50%" headers="mcps1.2.3.1.1 "><p id="p101655625812"><a name="p101655625812"></a><a name="p101655625812"></a><a href="__bfloat162bfloat16_rd.md">__bfloat162bfloat16_rd</a></p>
</td>
<td class="cellrowborder" valign="top" width="50%" headers="mcps1.2.3.1.2 "><p id="p91618565581"><a name="p91618565581"></a><a name="p91618565581"></a>输入遵循CAST_FLOOR模式取整后的bfloat16_t类型数据。</p>
</td>
</tr>
<tr id="row1579595313583"><td class="cellrowborder" valign="top" width="50%" headers="mcps1.2.3.1.1 "><p id="p87954533587"><a name="p87954533587"></a><a name="p87954533587"></a><a href="__bfloat162bfloat16_ru.md">__bfloat162bfloat16_ru</a></p>
</td>
<td class="cellrowborder" valign="top" width="50%" headers="mcps1.2.3.1.2 "><p id="p12795115311582"><a name="p12795115311582"></a><a name="p12795115311582"></a>输入遵循CAST_CEIL模式取整后的bfloat16_t类型数据。</p>
</td>
</tr>
<tr id="row8921651165820"><td class="cellrowborder" valign="top" width="50%" headers="mcps1.2.3.1.1 "><p id="p16921105115581"><a name="p16921105115581"></a><a name="p16921105115581"></a><a href="__bfloat162bfloat16_rna.md">__bfloat162bfloat16_rna</a></p>
</td>
<td class="cellrowborder" valign="top" width="50%" headers="mcps1.2.3.1.2 "><p id="p69818561708"><a name="p69818561708"></a><a name="p69818561708"></a>输入遵循CAST_ROUND模式取整后的bfloat16_t类型数据。</p>
</td>
</tr>
<tr id="row1404436154713"><td class="cellrowborder" valign="top" width="50%" headers="mcps1.2.3.1.1 "><p id="p18404133615473"><a name="p18404133615473"></a><a name="p18404133615473"></a><a href="__bfloat162uint_rn.md">__bfloat162uint_rn</a></p>
</td>
<td class="cellrowborder" valign="top" width="50%" headers="mcps1.2.3.1.2 "><p id="p2134855074"><a name="p2134855074"></a><a name="p2134855074"></a><span>将bfloat16转换为四舍五入到最接近偶数的无符号整数。</span></p>
</td>
</tr>
<tr id="row64040360473"><td class="cellrowborder" valign="top" width="50%" headers="mcps1.2.3.1.1 "><p id="p8404123612476"><a name="p8404123612476"></a><a name="p8404123612476"></a><a href="__bfloat162uint_rz.md">__bfloat162uint_rz</a></p>
</td>
<td class="cellrowborder" valign="top" width="50%" headers="mcps1.2.3.1.2 "><p id="p041010211817"><a name="p041010211817"></a><a name="p041010211817"></a><span>将bfloat16转换为向零舍入的无符号整数。</span></p>
</td>
</tr>
<tr id="row04047365474"><td class="cellrowborder" valign="top" width="50%" headers="mcps1.2.3.1.1 "><p id="p8404936144720"><a name="p8404936144720"></a><a name="p8404936144720"></a><a href="__bfloat162uint_rd.md">__bfloat162uint_rd</a></p>
</td>
<td class="cellrowborder" valign="top" width="50%" headers="mcps1.2.3.1.2 "><p id="p8705981887"><a name="p8705981887"></a><a name="p8705981887"></a><span>将bfloat16转换为向下取整的无符号整数</span>。</p>
</td>
</tr>
<tr id="row1340414361476"><td class="cellrowborder" valign="top" width="50%" headers="mcps1.2.3.1.1 "><p id="p5404143619476"><a name="p5404143619476"></a><a name="p5404143619476"></a><a href="__bfloat162uint_ru.md">__bfloat162uint_ru</a></p>
</td>
<td class="cellrowborder" valign="top" width="50%" headers="mcps1.2.3.1.2 "><p id="p127205345551"><a name="p127205345551"></a><a name="p127205345551"></a><span>将bfloat16转换为向上取整的无符号整数。</span></p>
</td>
</tr>
<tr id="row136491712112513"><td class="cellrowborder" valign="top" width="50%" headers="mcps1.2.3.1.1 "><p id="p12649191222519"><a name="p12649191222519"></a><a name="p12649191222519"></a><a href="__bfloat162uint_rna.md">__bfloat162uint_rna</a></p>
</td>
<td class="cellrowborder" valign="top" width="50%" headers="mcps1.2.3.1.2 "><p id="p1364931219251"><a name="p1364931219251"></a><a name="p1364931219251"></a><span>将bfloat16类型数据转换为向远离零舍入的无符号整数。</span></p>
</td>
</tr>
<tr id="row1404113619471"><td class="cellrowborder" valign="top" width="50%" headers="mcps1.2.3.1.1 "><p id="p1340411363471"><a name="p1340411363471"></a><a name="p1340411363471"></a><a href="__bfloat162int_rn.md">__bfloat162int_rn</a></p>
</td>
<td class="cellrowborder" valign="top" width="50%" headers="mcps1.2.3.1.2 "><p id="p1230712251812"><a name="p1230712251812"></a><a name="p1230712251812"></a><span>将bfloat16转换为四舍五入到最接近偶数的有符号整数</span></p>
</td>
</tr>
<tr id="row2040563694719"><td class="cellrowborder" valign="top" width="50%" headers="mcps1.2.3.1.1 "><p id="p440519365470"><a name="p440519365470"></a><a name="p440519365470"></a><a href="__bfloat162int_rz.md">__bfloat162int_rz</a></p>
</td>
<td class="cellrowborder" valign="top" width="50%" headers="mcps1.2.3.1.2 "><p id="p12882144116812"><a name="p12882144116812"></a><a name="p12882144116812"></a><span>将bfloat16转换为向零舍入的有符号整数</span>。</p>
</td>
</tr>
<tr id="row4405636204713"><td class="cellrowborder" valign="top" width="50%" headers="mcps1.2.3.1.1 "><p id="p7405143611478"><a name="p7405143611478"></a><a name="p7405143611478"></a><a href="__bfloat162int_rd.md">__bfloat162int_rd</a></p>
</td>
<td class="cellrowborder" valign="top" width="50%" headers="mcps1.2.3.1.2 "><p id="p1209105017819"><a name="p1209105017819"></a><a name="p1209105017819"></a><span>将bfloat16转换为向下取整的有符号整数</span>。</p>
</td>
</tr>
<tr id="row5405636194718"><td class="cellrowborder" valign="top" width="50%" headers="mcps1.2.3.1.1 "><p id="p440513369470"><a name="p440513369470"></a><a name="p440513369470"></a><a href="__bfloat162int_ru.md">__bfloat162int_ru</a></p>
</td>
<td class="cellrowborder" valign="top" width="50%" headers="mcps1.2.3.1.2 "><p id="p208632581689"><a name="p208632581689"></a><a name="p208632581689"></a><span>将bfloat16转换为向上取整的有符号整数</span>。</p>
</td>
</tr>
<tr id="row1352901552517"><td class="cellrowborder" valign="top" width="50%" headers="mcps1.2.3.1.1 "><p id="p35290154251"><a name="p35290154251"></a><a name="p35290154251"></a><a href="__bfloat162int_rna.md">__bfloat162int_rna</a></p>
</td>
<td class="cellrowborder" valign="top" width="50%" headers="mcps1.2.3.1.2 "><p id="p452911158259"><a name="p452911158259"></a><a name="p452911158259"></a><span>将bfloat16类型数据转换为向远离零舍入的有符号整数</span>。</p>
</td>
</tr>
<tr id="row040593613475"><td class="cellrowborder" valign="top" width="50%" headers="mcps1.2.3.1.1 "><p id="p154051736204717"><a name="p154051736204717"></a><a name="p154051736204717"></a><a href="__bfloat162ull_rn.md">__bfloat162ull_rn</a></p>
</td>
<td class="cellrowborder" valign="top" width="50%" headers="mcps1.2.3.1.2 "><p id="p77215343551"><a name="p77215343551"></a><a name="p77215343551"></a><span>将bfloat16转换为四舍五入到最接近偶数的64位无符号整数</span>。</p>
</td>
</tr>
<tr id="row9405153610479"><td class="cellrowborder" valign="top" width="50%" headers="mcps1.2.3.1.1 "><p id="p840514362473"><a name="p840514362473"></a><a name="p840514362473"></a><a href="__bfloat162ull_rz.md">__bfloat162ull_rz</a></p>
</td>
<td class="cellrowborder" valign="top" width="50%" headers="mcps1.2.3.1.2 "><p id="p5673514697"><a name="p5673514697"></a><a name="p5673514697"></a><span>将bfloat16 转换为向零舍入的64位无符号整数</span>。</p>
</td>
</tr>
<tr id="row5405113619474"><td class="cellrowborder" valign="top" width="50%" headers="mcps1.2.3.1.1 "><p id="p6405123644718"><a name="p6405123644718"></a><a name="p6405123644718"></a><a href="__bfloat162ull_rd.md">__bfloat162ull_rd</a></p>
</td>
<td class="cellrowborder" valign="top" width="50%" headers="mcps1.2.3.1.2 "><p id="p117215349551"><a name="p117215349551"></a><a name="p117215349551"></a><span>将 bfloat16转换为向下取整的64位无符号整数。</span></p>
</td>
</tr>
<tr id="row174050363475"><td class="cellrowborder" valign="top" width="50%" headers="mcps1.2.3.1.1 "><p id="p4405183694712"><a name="p4405183694712"></a><a name="p4405183694712"></a><a href="__bfloat162ull_ru.md">__bfloat162ull_ru</a></p>
</td>
<td class="cellrowborder" valign="top" width="50%" headers="mcps1.2.3.1.2 "><p id="p1721153475514"><a name="p1721153475514"></a><a name="p1721153475514"></a><span>将bfloat16转换为向上取整的64位无符号整数</span>。</p>
</td>
</tr>
<tr id="row531311186250"><td class="cellrowborder" valign="top" width="50%" headers="mcps1.2.3.1.1 "><p id="p33131618152518"><a name="p33131618152518"></a><a name="p33131618152518"></a><a href="__bfloat162ull_rna.md">__bfloat162ull_rna</a></p>
</td>
<td class="cellrowborder" valign="top" width="50%" headers="mcps1.2.3.1.2 "><p id="p731371814254"><a name="p731371814254"></a><a name="p731371814254"></a><span>将bfloat16类型数据转换为向远离零舍入的64位无符号整数</span>。</p>
</td>
</tr>
<tr id="row1540510369474"><td class="cellrowborder" valign="top" width="50%" headers="mcps1.2.3.1.1 "><p id="p1240615365476"><a name="p1240615365476"></a><a name="p1240615365476"></a><a href="__bfloat162ll_rn.md">__bfloat162ll_rn</a></p>
</td>
<td class="cellrowborder" valign="top" width="50%" headers="mcps1.2.3.1.2 "><p id="p4721334195517"><a name="p4721334195517"></a><a name="p4721334195517"></a><span>将 bfloat16转换为四舍五入到最接近偶数的64位有符号整数</span>。</p>
</td>
</tr>
<tr id="row840643614711"><td class="cellrowborder" valign="top" width="50%" headers="mcps1.2.3.1.1 "><p id="p1540653618478"><a name="p1540653618478"></a><a name="p1540653618478"></a><a href="__bfloat162ll_rz.md">__bfloat162ll_rz</a></p>
</td>
<td class="cellrowborder" valign="top" width="50%" headers="mcps1.2.3.1.2 "><p id="p105015571020"><a name="p105015571020"></a><a name="p105015571020"></a><span>将bfloat16转换为向零舍入的64位有符号整数</span>。</p>
</td>
</tr>
<tr id="row540616367473"><td class="cellrowborder" valign="top" width="50%" headers="mcps1.2.3.1.1 "><p id="p0406103664710"><a name="p0406103664710"></a><a name="p0406103664710"></a><a href="__bfloat162ll_rd.md">__bfloat162ll_rd</a></p>
</td>
<td class="cellrowborder" valign="top" width="50%" headers="mcps1.2.3.1.2 "><p id="p1172123435519"><a name="p1172123435519"></a><a name="p1172123435519"></a><span>将bfloat16转换为向下取整的64位有符号整数</span>。</p>
</td>
</tr>
<tr id="row11406193614716"><td class="cellrowborder" valign="top" width="50%" headers="mcps1.2.3.1.1 "><p id="p540653644713"><a name="p540653644713"></a><a name="p540653644713"></a><a href="__bfloat162ll_ru.md">__bfloat162ll_ru</a></p>
</td>
<td class="cellrowborder" valign="top" width="50%" headers="mcps1.2.3.1.2 "><p id="p572193495513"><a name="p572193495513"></a><a name="p572193495513"></a><span>将 bfloat16转换为向上取整的64位有符号整数</span>。</p>
</td>
</tr>
<tr id="row98496238252"><td class="cellrowborder" valign="top" width="50%" headers="mcps1.2.3.1.1 "><p id="p884919238256"><a name="p884919238256"></a><a name="p884919238256"></a><a href="__bfloat162ll_rna.md">__bfloat162ll_rna</a></p>
</td>
<td class="cellrowborder" valign="top" width="50%" headers="mcps1.2.3.1.2 "><p id="p0849323152519"><a name="p0849323152519"></a><a name="p0849323152519"></a><span>将bfloat16类型数据转换为向远离零舍入的64位有符号整数</span>。</p>
</td>
</tr>
<tr id="row54062036114718"><td class="cellrowborder" valign="top" width="50%" headers="mcps1.2.3.1.1 "><p id="p16406036194719"><a name="p16406036194719"></a><a name="p16406036194719"></a><a href="__uint2float_rn.md">__uint2float_rn</a></p>
</td>
<td class="cellrowborder" valign="top" width="50%" headers="mcps1.2.3.1.2 "><p id="p993662615105"><a name="p993662615105"></a><a name="p993662615105"></a><span>将uint32转换为浮点数，并四舍五入到最接近的偶数</span>。</p>
</td>
</tr>
<tr id="row1406173610472"><td class="cellrowborder" valign="top" width="50%" headers="mcps1.2.3.1.1 "><p id="p13406153694711"><a name="p13406153694711"></a><a name="p13406153694711"></a><a href="__uint2float_rz.md">__uint2float_rz</a></p>
</td>
<td class="cellrowborder" valign="top" width="50%" headers="mcps1.2.3.1.2 "><p id="p672123412559"><a name="p672123412559"></a><a name="p672123412559"></a><span>将uint32转换为向零舍入的浮点数。</span></p>
</td>
</tr>
<tr id="row34062036134713"><td class="cellrowborder" valign="top" width="50%" headers="mcps1.2.3.1.1 "><p id="p114069361472"><a name="p114069361472"></a><a name="p114069361472"></a><a href="__uint2float_rd.md">__uint2float_rd</a></p>
</td>
<td class="cellrowborder" valign="top" width="50%" headers="mcps1.2.3.1.2 "><p id="p167211343556"><a name="p167211343556"></a><a name="p167211343556"></a><span>将uint32向下取整转换为浮点数。</span></p>
</td>
</tr>
<tr id="row14406133684714"><td class="cellrowborder" valign="top" width="50%" headers="mcps1.2.3.1.1 "><p id="p440615369477"><a name="p440615369477"></a><a name="p440615369477"></a><a href="__uint2float_ru.md">__uint2float_ru</a></p>
</td>
<td class="cellrowborder" valign="top" width="50%" headers="mcps1.2.3.1.2 "><p id="p87065482107"><a name="p87065482107"></a><a name="p87065482107"></a><span>将uint32转换为向上取整的浮点数</span>。</p>
</td>
</tr>
<tr id="row759202718258"><td class="cellrowborder" valign="top" width="50%" headers="mcps1.2.3.1.1 "><p id="p1759027152517"><a name="p1759027152517"></a><a name="p1759027152517"></a><a href="__uint2float_rna.md">__uint2float_rna</a></p>
</td>
<td class="cellrowborder" valign="top" width="50%" headers="mcps1.2.3.1.2 "><p id="p0595272254"><a name="p0595272254"></a><a name="p0595272254"></a><span>将uint32类型数据转换为向远离零舍入的浮点数</span>。</p>
</td>
</tr>
<tr id="row19406103611476"><td class="cellrowborder" valign="top" width="50%" headers="mcps1.2.3.1.1 "><p id="p5406123618479"><a name="p5406123618479"></a><a name="p5406123618479"></a><a href="__uint2half_rn.md">__uint2half_rn</a></p>
</td>
<td class="cellrowborder" valign="top" width="50%" headers="mcps1.2.3.1.2 "><p id="p187211034185517"><a name="p187211034185517"></a><a name="p187211034185517"></a><span>将uint32转换为half（四舍五入到最接近的偶数）</span>。</p>
</td>
</tr>
<tr id="row14721257113414"><td class="cellrowborder" valign="top" width="50%" headers="mcps1.2.3.1.1 "><p id="p7722145711343"><a name="p7722145711343"></a><a name="p7722145711343"></a><a href="__uint2half_rn_sat.md">__uint2half_rn_sat</a></p>
</td>
<td class="cellrowborder" valign="top" width="50%" headers="mcps1.2.3.1.2 "><p id="p17221557203412"><a name="p17221557203412"></a><a name="p17221557203412"></a><span>饱和模式下输入的uint32数据转换成的half数据，并遵循CAST_RINT模式</span>。</p>
</td>
</tr>
<tr id="row14406173616474"><td class="cellrowborder" valign="top" width="50%" headers="mcps1.2.3.1.1 "><p id="p12406153614471"><a name="p12406153614471"></a><a name="p12406153614471"></a><a href="__uint2half_rz.md">__uint2half_rz</a></p>
</td>
<td class="cellrowborder" valign="top" width="50%" headers="mcps1.2.3.1.2 "><p id="p1272163419553"><a name="p1272163419553"></a><a name="p1272163419553"></a><span>将uint32转换为向零舍入的half。</span></p>
</td>
</tr>
<tr id="row550012833516"><td class="cellrowborder" valign="top" width="50%" headers="mcps1.2.3.1.1 "><p id="p12500282357"><a name="p12500282357"></a><a name="p12500282357"></a><a href="__uint2half_rz_sat.md">__uint2half_rz_sat</a></p>
</td>
<td class="cellrowborder" valign="top" width="50%" headers="mcps1.2.3.1.2 "><p id="p12598162311355"><a name="p12598162311355"></a><a name="p12598162311355"></a><span>饱和模式下输入的uint32数据转换成的half数据，并遵循CAST_TRUNC模式</span>。</p>
</td>
</tr>
<tr id="row6406036144718"><td class="cellrowborder" valign="top" width="50%" headers="mcps1.2.3.1.1 "><p id="p1640763614720"><a name="p1640763614720"></a><a name="p1640763614720"></a><a href="__uint2half_rd.md">__uint2half_rd</a></p>
</td>
<td class="cellrowborder" valign="top" width="50%" headers="mcps1.2.3.1.2 "><p id="p18482911131119"><a name="p18482911131119"></a><a name="p18482911131119"></a><span>将uint32向下取整为half</span>。</p>
</td>
</tr>
<tr id="row15407921133516"><td class="cellrowborder" valign="top" width="50%" headers="mcps1.2.3.1.1 "><p id="p17407021123518"><a name="p17407021123518"></a><a name="p17407021123518"></a><a href="__uint2half_rd_sat.md">__uint2half_rd_sat</a></p>
</td>
<td class="cellrowborder" valign="top" width="50%" headers="mcps1.2.3.1.2 "><p id="p1540722111353"><a name="p1540722111353"></a><a name="p1540722111353"></a><span>饱和模式下输入的uint32数据转换成的half数据，并遵循CAST_FLOOR模式</span>。</p>
</td>
</tr>
<tr id="row94072036104718"><td class="cellrowborder" valign="top" width="50%" headers="mcps1.2.3.1.1 "><p id="p104077362478"><a name="p104077362478"></a><a name="p104077362478"></a><a href="__uint2half_ru.md">__uint2half_ru</a></p>
</td>
<td class="cellrowborder" valign="top" width="50%" headers="mcps1.2.3.1.2 "><p id="p185701217121113"><a name="p185701217121113"></a><a name="p185701217121113"></a><span>将uint32向上取整转换为half</span>。</p>
</td>
</tr>
<tr id="row1184142783513"><td class="cellrowborder" valign="top" width="50%" headers="mcps1.2.3.1.1 "><p id="p584192743518"><a name="p584192743518"></a><a name="p584192743518"></a><a href="__uint2half_ru_sat.md">__uint2half_ru_sat</a></p>
</td>
<td class="cellrowborder" valign="top" width="50%" headers="mcps1.2.3.1.2 "><p id="p984527193512"><a name="p984527193512"></a><a name="p984527193512"></a><span>饱和模式下输入的uint32数据转换成的half数据，并遵循CAST_CEIL模式</span>。</p>
</td>
</tr>
<tr id="row124561830142510"><td class="cellrowborder" valign="top" width="50%" headers="mcps1.2.3.1.1 "><p id="p7456630202512"><a name="p7456630202512"></a><a name="p7456630202512"></a><a href="__uint2half_rna.md">__uint2half_rna</a></p>
</td>
<td class="cellrowborder" valign="top" width="50%" headers="mcps1.2.3.1.2 "><p id="p15456113052510"><a name="p15456113052510"></a><a name="p15456113052510"></a><span>将uint32类型数据转换为向远离零舍入的half类型数据</span>。</p>
</td>
</tr>
<tr id="row7251533183512"><td class="cellrowborder" valign="top" width="50%" headers="mcps1.2.3.1.1 "><p id="p1925333103518"><a name="p1925333103518"></a><a name="p1925333103518"></a><a href="__uint2half_rna_sat.md">__uint2half_rna_sat</a></p>
</td>
<td class="cellrowborder" valign="top" width="50%" headers="mcps1.2.3.1.2 "><p id="p1025153310353"><a name="p1025153310353"></a><a name="p1025153310353"></a><span>饱和模式下输入的uint32数据转换成的half数据，并遵循CAST_ROUND模式</span>。</p>
</td>
</tr>
<tr id="row640719366478"><td class="cellrowborder" valign="top" width="50%" headers="mcps1.2.3.1.1 "><p id="p64071736184710"><a name="p64071736184710"></a><a name="p64071736184710"></a><a href="__uint2bfloat16_rn.md">__uint2bfloat16_rn</a></p>
</td>
<td class="cellrowborder" valign="top" width="50%" headers="mcps1.2.3.1.2 "><p id="p172183435518"><a name="p172183435518"></a><a name="p172183435518"></a><span>将uint32转换为bfloat16，并四舍五入到最接近的偶数</span>。</p>
</td>
</tr>
<tr id="row18407183619472"><td class="cellrowborder" valign="top" width="50%" headers="mcps1.2.3.1.1 "><p id="p17407123615473"><a name="p17407123615473"></a><a name="p17407123615473"></a><a href="__uint2bfloat16_rz.md">__uint2bfloat16_rz</a></p>
</td>
<td class="cellrowborder" valign="top" width="50%" headers="mcps1.2.3.1.2 "><p id="p162891035121120"><a name="p162891035121120"></a><a name="p162891035121120"></a><span>将uint32转换为向零舍入的bfloat16。</span></p>
</td>
</tr>
<tr id="row640783684717"><td class="cellrowborder" valign="top" width="50%" headers="mcps1.2.3.1.1 "><p id="p1407133612471"><a name="p1407133612471"></a><a name="p1407133612471"></a><a href="__uint2bfloat16_rd.md">__uint2bfloat16_rd</a></p>
</td>
<td class="cellrowborder" valign="top" width="50%" headers="mcps1.2.3.1.2 "><p id="p2092954331115"><a name="p2092954331115"></a><a name="p2092954331115"></a><span>将uint32向下取整转换为bfloat16</span>。</p>
</td>
</tr>
<tr id="row1340715368473"><td class="cellrowborder" valign="top" width="50%" headers="mcps1.2.3.1.1 "><p id="p1440717367474"><a name="p1440717367474"></a><a name="p1440717367474"></a><a href="__uint2bfloat16_ru.md">__uint2bfloat16_ru</a></p>
</td>
<td class="cellrowborder" valign="top" width="50%" headers="mcps1.2.3.1.2 "><p id="p26171250201111"><a name="p26171250201111"></a><a name="p26171250201111"></a><span>将uint32向上取整转换为bfloat16</span>。</p>
</td>
</tr>
<tr id="row2185733142517"><td class="cellrowborder" valign="top" width="50%" headers="mcps1.2.3.1.1 "><p id="p3185433142516"><a name="p3185433142516"></a><a name="p3185433142516"></a><a href="__uint2bfloat16_rna.md">__uint2bfloat16_rna</a></p>
</td>
<td class="cellrowborder" valign="top" width="50%" headers="mcps1.2.3.1.2 "><p id="p11185203314258"><a name="p11185203314258"></a><a name="p11185203314258"></a><span>将uint32类型数据转换为向远离零舍入的bfloat16类型数据</span>。</p>
</td>
</tr>
<tr id="row104071136194718"><td class="cellrowborder" valign="top" width="50%" headers="mcps1.2.3.1.1 "><p id="p15407036144715"><a name="p15407036144715"></a><a name="p15407036144715"></a><a href="__int2float_rn.md">__int2float_rn</a></p>
</td>
<td class="cellrowborder" valign="top" width="50%" headers="mcps1.2.3.1.2 "><p id="p1814358131111"><a name="p1814358131111"></a><a name="p1814358131111"></a><span>将int32转换为浮点数，并四舍五入到最接近的偶数</span>。</p>
</td>
</tr>
<tr id="row5408183644719"><td class="cellrowborder" valign="top" width="50%" headers="mcps1.2.3.1.1 "><p id="p1040815363470"><a name="p1040815363470"></a><a name="p1040815363470"></a><a href="__int2float_rz.md">__int2float_rz</a></p>
</td>
<td class="cellrowborder" valign="top" width="50%" headers="mcps1.2.3.1.2 "><p id="p2721153415558"><a name="p2721153415558"></a><a name="p2721153415558"></a><span>将int32转换为向零舍入的浮点数</span>。</p>
</td>
</tr>
<tr id="row19408143619479"><td class="cellrowborder" valign="top" width="50%" headers="mcps1.2.3.1.1 "><p id="p1540883614474"><a name="p1540883614474"></a><a name="p1540883614474"></a><a href="__int2float_rd.md">__int2float_rd</a></p>
</td>
<td class="cellrowborder" valign="top" width="50%" headers="mcps1.2.3.1.2 "><p id="p112301814171210"><a name="p112301814171210"></a><a name="p112301814171210"></a><span>将int32向下取整转换为浮点数</span>。</p>
</td>
</tr>
<tr id="row240843612473"><td class="cellrowborder" valign="top" width="50%" headers="mcps1.2.3.1.1 "><p id="p15408436184719"><a name="p15408436184719"></a><a name="p15408436184719"></a><a href="__int2float_ru.md">__int2float_ru</a></p>
</td>
<td class="cellrowborder" valign="top" width="50%" headers="mcps1.2.3.1.2 "><p id="p18722334185518"><a name="p18722334185518"></a><a name="p18722334185518"></a><span>将int32转换为向上取整的浮点数</span>。</p>
</td>
</tr>
<tr id="row7842735152519"><td class="cellrowborder" valign="top" width="50%" headers="mcps1.2.3.1.1 "><p id="p384233512512"><a name="p384233512512"></a><a name="p384233512512"></a><a href="__int2float_rna.md">__int2float_rna</a></p>
</td>
<td class="cellrowborder" valign="top" width="50%" headers="mcps1.2.3.1.2 "><p id="p5842193513251"><a name="p5842193513251"></a><a name="p5842193513251"></a><span>将int32类型数据转换为向远离零舍入的浮点数</span>。</p>
</td>
</tr>
<tr id="row104081236154716"><td class="cellrowborder" valign="top" width="50%" headers="mcps1.2.3.1.1 "><p id="p17408143674715"><a name="p17408143674715"></a><a name="p17408143674715"></a><a href="__int2half_rn.md">__int2half_rn</a></p>
</td>
<td class="cellrowborder" valign="top" width="50%" headers="mcps1.2.3.1.2 "><p id="p19263529181212"><a name="p19263529181212"></a><a name="p19263529181212"></a><span>将int32转换为half（四舍五入到最接近的偶数）</span>。</p>
</td>
</tr>
<tr id="row1437410479351"><td class="cellrowborder" valign="top" width="50%" headers="mcps1.2.3.1.1 "><p id="p153743475353"><a name="p153743475353"></a><a name="p153743475353"></a><a href="__int2half_rn_sat.md">__int2half_rn_sat</a></p>
</td>
<td class="cellrowborder" valign="top" width="50%" headers="mcps1.2.3.1.2 "><p id="p15374134711352"><a name="p15374134711352"></a><a name="p15374134711352"></a><span>饱和模式下输入的int32数据转换成的half数据，并遵循CAST_RINT模式</span>。</p>
</td>
</tr>
<tr id="row540863644714"><td class="cellrowborder" valign="top" width="50%" headers="mcps1.2.3.1.1 "><p id="p1340873674712"><a name="p1340873674712"></a><a name="p1340873674712"></a><a href="__int2half_rz.md">__int2half_rz</a></p>
</td>
<td class="cellrowborder" valign="top" width="50%" headers="mcps1.2.3.1.2 "><p id="p47221034165515"><a name="p47221034165515"></a><a name="p47221034165515"></a><span>将int32转换为向零舍入的half</span>。</p>
</td>
</tr>
<tr id="row4934175316357"><td class="cellrowborder" valign="top" width="50%" headers="mcps1.2.3.1.1 "><p id="p89356538355"><a name="p89356538355"></a><a name="p89356538355"></a><a href="__int2half_rz_sat.md">__int2half_rz_sat</a></p>
</td>
<td class="cellrowborder" valign="top" width="50%" headers="mcps1.2.3.1.2 "><p id="p16935175323515"><a name="p16935175323515"></a><a name="p16935175323515"></a><span>饱和模式下输入的int32数据转换成的half数据，并遵循CAST_TRUNC模式</span>。</p>
</td>
</tr>
<tr id="row1540811360472"><td class="cellrowborder" valign="top" width="50%" headers="mcps1.2.3.1.1 "><p id="p194081236174713"><a name="p194081236174713"></a><a name="p194081236174713"></a><a href="__int2half_rd.md">__int2half_rd</a></p>
</td>
<td class="cellrowborder" valign="top" width="50%" headers="mcps1.2.3.1.2 "><p id="p5739104318125"><a name="p5739104318125"></a><a name="p5739104318125"></a><span>将int32向下取整为half</span>。</p>
</td>
</tr>
<tr id="row4538175610353"><td class="cellrowborder" valign="top" width="50%" headers="mcps1.2.3.1.1 "><p id="p353855673516"><a name="p353855673516"></a><a name="p353855673516"></a><a href="__int2half_rd_sat.md">__int2half_rd_sat</a></p>
</td>
<td class="cellrowborder" valign="top" width="50%" headers="mcps1.2.3.1.2 "><p id="p5538456173511"><a name="p5538456173511"></a><a name="p5538456173511"></a><span>饱和模式下输入的int32数据转换成的half数据，并遵循CAST_FLOOR模式</span>。</p>
</td>
</tr>
<tr id="row2408103634714"><td class="cellrowborder" valign="top" width="50%" headers="mcps1.2.3.1.1 "><p id="p164081636124717"><a name="p164081636124717"></a><a name="p164081636124717"></a><a href="__int2half_ru.md">__int2half_ru</a></p>
</td>
<td class="cellrowborder" valign="top" width="50%" headers="mcps1.2.3.1.2 "><p id="p76114525124"><a name="p76114525124"></a><a name="p76114525124"></a><span>将int32向上取整转换为half</span>。</p>
</td>
</tr>
<tr id="row1121885918357"><td class="cellrowborder" valign="top" width="50%" headers="mcps1.2.3.1.1 "><p id="p9218115913356"><a name="p9218115913356"></a><a name="p9218115913356"></a><a href="__int2half_ru_sat.md">__int2half_ru_sat</a></p>
</td>
<td class="cellrowborder" valign="top" width="50%" headers="mcps1.2.3.1.2 "><p id="p182191594354"><a name="p182191594354"></a><a name="p182191594354"></a><span>饱和模式下输入的int32数据转换成的half数据，并遵循CAST_CEIL模式</span>。</p>
</td>
</tr>
<tr id="row16940538112519"><td class="cellrowborder" valign="top" width="50%" headers="mcps1.2.3.1.1 "><p id="p1394013386252"><a name="p1394013386252"></a><a name="p1394013386252"></a><a href="__int2half_rna.md">__int2half_rna</a></p>
</td>
<td class="cellrowborder" valign="top" width="50%" headers="mcps1.2.3.1.2 "><p id="p14940138122513"><a name="p14940138122513"></a><a name="p14940138122513"></a><span>将int32类型数据转换为向远离零舍入的half</span><span>类型数据</span>。</p>
</td>
</tr>
<tr id="row449521173619"><td class="cellrowborder" valign="top" width="50%" headers="mcps1.2.3.1.1 "><p id="p449514111365"><a name="p449514111365"></a><a name="p449514111365"></a><a href="__int2half_rna_sat.md">__int2half_rna_sat</a></p>
</td>
<td class="cellrowborder" valign="top" width="50%" headers="mcps1.2.3.1.2 "><p id="p64956119366"><a name="p64956119366"></a><a name="p64956119366"></a><span>饱和模式下输入的int32数据转换成的half数据，并遵循CAST_ROUND模式</span>。</p>
</td>
</tr>
<tr id="row164081836184711"><td class="cellrowborder" valign="top" width="50%" headers="mcps1.2.3.1.1 "><p id="p17408936184711"><a name="p17408936184711"></a><a name="p17408936184711"></a><a href="__int2bfloat16_rn.md">__int2bfloat16_rn</a></p>
</td>
<td class="cellrowborder" valign="top" width="50%" headers="mcps1.2.3.1.2 "><p id="p17722203417557"><a name="p17722203417557"></a><a name="p17722203417557"></a><span>将int32转换为bfloat16，并四舍五入到最接近的偶数</span>。</p>
</td>
</tr>
<tr id="row174081936124717"><td class="cellrowborder" valign="top" width="50%" headers="mcps1.2.3.1.1 "><p id="p3409153684712"><a name="p3409153684712"></a><a name="p3409153684712"></a><a href="__int2bfloat16_rz.md">__int2bfloat16_rz</a></p>
</td>
<td class="cellrowborder" valign="top" width="50%" headers="mcps1.2.3.1.2 "><p id="p572293417553"><a name="p572293417553"></a><a name="p572293417553"></a><span>将int32转换为向零舍入的bfloat16。</span></p>
</td>
</tr>
<tr id="row040933613479"><td class="cellrowborder" valign="top" width="50%" headers="mcps1.2.3.1.1 "><p id="p204098366478"><a name="p204098366478"></a><a name="p204098366478"></a><a href="__int2bfloat16_rd.md">__int2bfloat16_rd</a></p>
</td>
<td class="cellrowborder" valign="top" width="50%" headers="mcps1.2.3.1.2 "><p id="p2642141419136"><a name="p2642141419136"></a><a name="p2642141419136"></a><span>将int32向下取整转换为bfloat16</span>。</p>
</td>
</tr>
<tr id="row1940923604715"><td class="cellrowborder" valign="top" width="50%" headers="mcps1.2.3.1.1 "><p id="p1740943604716"><a name="p1740943604716"></a><a name="p1740943604716"></a><a href="__int2bfloat16_ru.md">__int2bfloat16_ru</a></p>
</td>
<td class="cellrowborder" valign="top" width="50%" headers="mcps1.2.3.1.2 "><p id="p1759312411314"><a name="p1759312411314"></a><a name="p1759312411314"></a><span>将int32向上取整转换为bfloat16</span>。</p>
</td>
</tr>
<tr id="row12863141142515"><td class="cellrowborder" valign="top" width="50%" headers="mcps1.2.3.1.1 "><p id="p12864441192512"><a name="p12864441192512"></a><a name="p12864441192512"></a><a href="__int2bfloat16_rna.md">__int2bfloat16_rna</a></p>
</td>
<td class="cellrowborder" valign="top" width="50%" headers="mcps1.2.3.1.2 "><p id="p168641415255"><a name="p168641415255"></a><a name="p168641415255"></a><span>将int32类型数据转换为向远离零舍入的bfloat16类型数据。</span></p>
</td>
</tr>
<tr id="row8409133612478"><td class="cellrowborder" valign="top" width="50%" headers="mcps1.2.3.1.1 "><p id="p7409436164710"><a name="p7409436164710"></a><a name="p7409436164710"></a><a href="__ull2float_rn.md">__ull2float_rn</a></p>
</td>
<td class="cellrowborder" valign="top" width="50%" headers="mcps1.2.3.1.2 "><p id="p7722183418559"><a name="p7722183418559"></a><a name="p7722183418559"></a><span>将uint64转换为浮点数，并四舍五入到最接近的偶数</span>。</p>
</td>
</tr>
<tr id="row15409173615474"><td class="cellrowborder" valign="top" width="50%" headers="mcps1.2.3.1.1 "><p id="p6409123694715"><a name="p6409123694715"></a><a name="p6409123694715"></a><a href="__ull2float_rz.md">__ull2float_rz</a></p>
</td>
<td class="cellrowborder" valign="top" width="50%" headers="mcps1.2.3.1.2 "><p id="p1454923814139"><a name="p1454923814139"></a><a name="p1454923814139"></a><span>将uint64转换为向零舍入的浮点数</span>。</p>
</td>
</tr>
<tr id="row2409103614713"><td class="cellrowborder" valign="top" width="50%" headers="mcps1.2.3.1.1 "><p id="p840953620478"><a name="p840953620478"></a><a name="p840953620478"></a><a href="__ull2float_rd.md">__ull2float_rd</a></p>
</td>
<td class="cellrowborder" valign="top" width="50%" headers="mcps1.2.3.1.2 "><p id="p1685174561314"><a name="p1685174561314"></a><a name="p1685174561314"></a><span>将uint64向下取整转换为浮点数</span>。</p>
</td>
</tr>
<tr id="row174091036164712"><td class="cellrowborder" valign="top" width="50%" headers="mcps1.2.3.1.1 "><p id="p1440913365478"><a name="p1440913365478"></a><a name="p1440913365478"></a><a href="__ull2float_ru.md">__ull2float_ru</a></p>
</td>
<td class="cellrowborder" valign="top" width="50%" headers="mcps1.2.3.1.2 "><p id="p2722103415512"><a name="p2722103415512"></a><a name="p2722103415512"></a><span>将uint64向上取整转换为浮点数</span>。</p>
</td>
</tr>
<tr id="row1133774515251"><td class="cellrowborder" valign="top" width="50%" headers="mcps1.2.3.1.1 "><p id="p433714532516"><a name="p433714532516"></a><a name="p433714532516"></a><a href="__ull2float_rna.md">__ull2float_rna</a></p>
</td>
<td class="cellrowborder" valign="top" width="50%" headers="mcps1.2.3.1.2 "><p id="p1433724572515"><a name="p1433724572515"></a><a name="p1433724572515"></a><span>将uint64类型数据转换为向远离零舍入的浮点数</span>。</p>
</td>
</tr>
<tr id="row3409173654713"><td class="cellrowborder" valign="top" width="50%" headers="mcps1.2.3.1.1 "><p id="p13409936104712"><a name="p13409936104712"></a><a name="p13409936104712"></a><a href="__ull2half_rn.md">__ull2half_rn</a></p>
</td>
<td class="cellrowborder" valign="top" width="50%" headers="mcps1.2.3.1.2 "><p id="p41941026202016"><a name="p41941026202016"></a><a name="p41941026202016"></a><span>将uint64转换为half，并四舍五入到最接近的偶数</span>。</p>
</td>
</tr>
<tr id="row20409123616474"><td class="cellrowborder" valign="top" width="50%" headers="mcps1.2.3.1.1 "><p id="p34094362477"><a name="p34094362477"></a><a name="p34094362477"></a><a href="__ull2half_rz.md">__ull2half_rz</a></p>
</td>
<td class="cellrowborder" valign="top" width="50%" headers="mcps1.2.3.1.2 "><p id="p18542972149"><a name="p18542972149"></a><a name="p18542972149"></a><span>将uint64转换为向零舍入的half</span>。</p>
</td>
</tr>
<tr id="row04091236104716"><td class="cellrowborder" valign="top" width="50%" headers="mcps1.2.3.1.1 "><p id="p194101336164717"><a name="p194101336164717"></a><a name="p194101336164717"></a><a href="__ull2half_rd.md">__ull2half_rd</a></p>
</td>
<td class="cellrowborder" valign="top" width="50%" headers="mcps1.2.3.1.2 "><p id="p13722153445518"><a name="p13722153445518"></a><a name="p13722153445518"></a><span>将uint64向下取整转换为half</span>。</p>
</td>
</tr>
<tr id="row641083615477"><td class="cellrowborder" valign="top" width="50%" headers="mcps1.2.3.1.1 "><p id="p17410123617479"><a name="p17410123617479"></a><a name="p17410123617479"></a><a href="__ull2half_ru.md">__ull2half_ru</a></p>
</td>
<td class="cellrowborder" valign="top" width="50%" headers="mcps1.2.3.1.2 "><p id="p177221534165512"><a name="p177221534165512"></a><a name="p177221534165512"></a><span>将uint64向上取整转换为浮点数</span>。</p>
</td>
</tr>
<tr id="row9546104812253"><td class="cellrowborder" valign="top" width="50%" headers="mcps1.2.3.1.1 "><p id="p85461548102519"><a name="p85461548102519"></a><a name="p85461548102519"></a><a href="__ull2half_rna.md">__ull2half_rna</a></p>
</td>
<td class="cellrowborder" valign="top" width="50%" headers="mcps1.2.3.1.2 "><p id="p1554674810259"><a name="p1554674810259"></a><a name="p1554674810259"></a><span>将uint64类型数据转换为向远离零舍入的half</span><span>类型数据</span>。</p>
</td>
</tr>
<tr id="row241019362470"><td class="cellrowborder" valign="top" width="50%" headers="mcps1.2.3.1.1 "><p id="p1141013613476"><a name="p1141013613476"></a><a name="p1141013613476"></a><a href="__ull2bfloat16_rn.md">__ull2bfloat16_rn</a></p>
</td>
<td class="cellrowborder" valign="top" width="50%" headers="mcps1.2.3.1.2 "><p id="p14484204917229"><a name="p14484204917229"></a><a name="p14484204917229"></a><span>将uint64转换为bfloat16，并四舍五入到最接近的偶数</span>。</p>
</td>
</tr>
<tr id="row3410143612478"><td class="cellrowborder" valign="top" width="50%" headers="mcps1.2.3.1.1 "><p id="p0410936124718"><a name="p0410936124718"></a><a name="p0410936124718"></a><a href="__ull2bfloat16_rz.md">__ull2bfloat16_rz</a></p>
</td>
<td class="cellrowborder" valign="top" width="50%" headers="mcps1.2.3.1.2 "><p id="p0573123741412"><a name="p0573123741412"></a><a name="p0573123741412"></a><span>将uint64转换为向零舍入的bfloat16</span>。</p>
</td>
</tr>
<tr id="row14100360474"><td class="cellrowborder" valign="top" width="50%" headers="mcps1.2.3.1.1 "><p id="p1341003644714"><a name="p1341003644714"></a><a name="p1341003644714"></a><a href="__ull2bfloat16_rd.md">__ull2bfloat16_rd</a></p>
</td>
<td class="cellrowborder" valign="top" width="50%" headers="mcps1.2.3.1.2 "><p id="p755194513148"><a name="p755194513148"></a><a name="p755194513148"></a><span>将uint64向下取整转换为bfloat</span>16。</p>
</td>
</tr>
<tr id="row20410113617479"><td class="cellrowborder" valign="top" width="50%" headers="mcps1.2.3.1.1 "><p id="p12410113614717"><a name="p12410113614717"></a><a name="p12410113614717"></a><a href="__ull2bfloat16_ru.md">__ull2bfloat16_ru</a></p>
</td>
<td class="cellrowborder" valign="top" width="50%" headers="mcps1.2.3.1.2 "><p id="p10722163455515"><a name="p10722163455515"></a><a name="p10722163455515"></a><span>将uint64向上取整转换为bfloat</span>16。</p>
</td>
</tr>
<tr id="row7791552172511"><td class="cellrowborder" valign="top" width="50%" headers="mcps1.2.3.1.1 "><p id="p879125210252"><a name="p879125210252"></a><a name="p879125210252"></a><a href="__ull2bfloat16_rna.md">__ull2bfloat16_rna</a></p>
</td>
<td class="cellrowborder" valign="top" width="50%" headers="mcps1.2.3.1.2 "><p id="p07919521254"><a name="p07919521254"></a><a name="p07919521254"></a><span>将uint64类型数据转换为向远离零舍入的bfloat16</span><span>类型数据</span>。</p>
</td>
</tr>
<tr id="row841023684719"><td class="cellrowborder" valign="top" width="50%" headers="mcps1.2.3.1.1 "><p id="p04109367478"><a name="p04109367478"></a><a name="p04109367478"></a><a href="__ll2float_rn.md">__ll2float_rn</a></p>
</td>
<td class="cellrowborder" valign="top" width="50%" headers="mcps1.2.3.1.2 "><p id="p137229345558"><a name="p137229345558"></a><a name="p137229345558"></a><span>将int64转换为四舍五入到最接近偶数的浮点数。</span></p>
</td>
</tr>
<tr id="row2410103616479"><td class="cellrowborder" valign="top" width="50%" headers="mcps1.2.3.1.1 "><p id="p19410183684711"><a name="p19410183684711"></a><a name="p19410183684711"></a><a href="__ll2float_rz.md">__ll2float_rz</a></p>
</td>
<td class="cellrowborder" valign="top" width="50%" headers="mcps1.2.3.1.2 "><p id="p147221134115515"><a name="p147221134115515"></a><a name="p147221134115515"></a><span>将int64转换为向零舍入的浮点数。</span></p>
</td>
</tr>
<tr id="row1941013620473"><td class="cellrowborder" valign="top" width="50%" headers="mcps1.2.3.1.1 "><p id="p4410336174713"><a name="p4410336174713"></a><a name="p4410336174713"></a><a href="__ll2float_rd.md">__ll2float_rd</a></p>
</td>
<td class="cellrowborder" valign="top" width="50%" headers="mcps1.2.3.1.2 "><p id="p17722113405519"><a name="p17722113405519"></a><a name="p17722113405519"></a><span>将int64向下取整转换为浮点数</span>。</p>
</td>
</tr>
<tr id="row4410183644711"><td class="cellrowborder" valign="top" width="50%" headers="mcps1.2.3.1.1 "><p id="p1841020365478"><a name="p1841020365478"></a><a name="p1841020365478"></a><a href="__ll2float_ru.md">__ll2float_ru</a></p>
</td>
<td class="cellrowborder" valign="top" width="50%" headers="mcps1.2.3.1.2 "><p id="p1431032531516"><a name="p1431032531516"></a><a name="p1431032531516"></a><span>将int转换为向上取整的浮点数</span>。</p>
</td>
</tr>
<tr id="row1743115552510"><td class="cellrowborder" valign="top" width="50%" headers="mcps1.2.3.1.1 "><p id="p874335510253"><a name="p874335510253"></a><a name="p874335510253"></a><a href="__ll2float_rna.md">__ll2float_rna</a></p>
</td>
<td class="cellrowborder" valign="top" width="50%" headers="mcps1.2.3.1.2 "><p id="p13743155152519"><a name="p13743155152519"></a><a name="p13743155152519"></a><span>将int64类型数据转换为向远离零舍入的浮点数</span>。</p>
</td>
</tr>
<tr id="row14411143624716"><td class="cellrowborder" valign="top" width="50%" headers="mcps1.2.3.1.1 "><p id="p15411173619473"><a name="p15411173619473"></a><a name="p15411173619473"></a><a href="__ll2half_rn.md">__ll2half_rn</a></p>
</td>
<td class="cellrowborder" valign="top" width="50%" headers="mcps1.2.3.1.2 "><p id="p14723234205514"><a name="p14723234205514"></a><a name="p14723234205514"></a><span>将int64转换为四舍五入到最接近偶数的half。</span></p>
</td>
</tr>
<tr id="row114117366478"><td class="cellrowborder" valign="top" width="50%" headers="mcps1.2.3.1.1 "><p id="p1941118362474"><a name="p1941118362474"></a><a name="p1941118362474"></a><a href="__ll2half_rz.md">__ll2half_rz</a></p>
</td>
<td class="cellrowborder" valign="top" width="50%" headers="mcps1.2.3.1.2 "><p id="p141931639101512"><a name="p141931639101512"></a><a name="p141931639101512"></a><span>将int64转换为向零舍入的half。</span></p>
</td>
</tr>
<tr id="row2041193619478"><td class="cellrowborder" valign="top" width="50%" headers="mcps1.2.3.1.1 "><p id="p1411203610475"><a name="p1411203610475"></a><a name="p1411203610475"></a><a href="__ll2half_rd.md">__ll2half_rd</a></p>
</td>
<td class="cellrowborder" valign="top" width="50%" headers="mcps1.2.3.1.2 "><p id="p172319346556"><a name="p172319346556"></a><a name="p172319346556"></a><span>将int64向下取整转换为half</span>。</p>
</td>
</tr>
<tr id="row10411836204714"><td class="cellrowborder" valign="top" width="50%" headers="mcps1.2.3.1.1 "><p id="p1411103614473"><a name="p1411103614473"></a><a name="p1411103614473"></a><a href="__ll2half_ru.md">__ll2half_ru</a></p>
</td>
<td class="cellrowborder" valign="top" width="50%" headers="mcps1.2.3.1.2 "><p id="p128211254151511"><a name="p128211254151511"></a><a name="p128211254151511"></a><span>将int64转换为向上取整的half</span>。</p>
</td>
</tr>
<tr id="row463935882510"><td class="cellrowborder" valign="top" width="50%" headers="mcps1.2.3.1.1 "><p id="p163965814259"><a name="p163965814259"></a><a name="p163965814259"></a><a href="__ll2half_rna.md">__ll2half_rna</a></p>
</td>
<td class="cellrowborder" valign="top" width="50%" headers="mcps1.2.3.1.2 "><p id="p14640135882511"><a name="p14640135882511"></a><a name="p14640135882511"></a><span>将int64类型数据转换为向远离零舍入的half类型数据</span>。</p>
</td>
</tr>
<tr id="row114111836174714"><td class="cellrowborder" valign="top" width="50%" headers="mcps1.2.3.1.1 "><p id="p1241183654713"><a name="p1241183654713"></a><a name="p1241183654713"></a><a href="__ll2bfloat16_rn.md">__ll2bfloat16_rn</a></p>
</td>
<td class="cellrowborder" valign="top" width="50%" headers="mcps1.2.3.1.2 "><p id="p192102114163"><a name="p192102114163"></a><a name="p192102114163"></a><span>将int64转换为四舍五入到最接近偶数的bfloat16</span>。</p>
</td>
</tr>
<tr id="row34111136124714"><td class="cellrowborder" valign="top" width="50%" headers="mcps1.2.3.1.1 "><p id="p2411193619479"><a name="p2411193619479"></a><a name="p2411193619479"></a><a href="__ll2bfloat16_rz.md">__ll2bfloat16_rz</a></p>
</td>
<td class="cellrowborder" valign="top" width="50%" headers="mcps1.2.3.1.2 "><p id="p107391107164"><a name="p107391107164"></a><a name="p107391107164"></a><span>将int64转换为向零舍入的bfloat16</span>。</p>
</td>
</tr>
<tr id="row194115368474"><td class="cellrowborder" valign="top" width="50%" headers="mcps1.2.3.1.1 "><p id="p17411536174710"><a name="p17411536174710"></a><a name="p17411536174710"></a><a href="__ll2bfloat16_rd.md">__ll2bfloat16_rd</a></p>
</td>
<td class="cellrowborder" valign="top" width="50%" headers="mcps1.2.3.1.2 "><p id="p14723203445512"><a name="p14723203445512"></a><a name="p14723203445512"></a><span>将int64向下取整转换为bfloat16</span>。</p>
</td>
</tr>
<tr id="row541123617472"><td class="cellrowborder" valign="top" width="50%" headers="mcps1.2.3.1.1 "><p id="p1141112367474"><a name="p1141112367474"></a><a name="p1141112367474"></a><a href="__ll2bfloat16_ru.md">__ll2bfloat16_ru</a></p>
</td>
<td class="cellrowborder" valign="top" width="50%" headers="mcps1.2.3.1.2 "><p id="p169241026171617"><a name="p169241026171617"></a><a name="p169241026171617"></a><span>将int转换为向上取整的bfloat16</span>。</p>
</td>
</tr>
<tr id="row9837181142617"><td class="cellrowborder" valign="top" width="50%" headers="mcps1.2.3.1.1 "><p id="p1583731142610"><a name="p1583731142610"></a><a name="p1583731142610"></a><a href="__ll2bfloat16_rna.md">__ll2bfloat16_rna</a></p>
</td>
<td class="cellrowborder" valign="top" width="50%" headers="mcps1.2.3.1.2 "><p id="p6837191182611"><a name="p6837191182611"></a><a name="p6837191182611"></a><span>将int64类型数据转换为向远离零舍入的bfloat16类型数据</span>。</p>
</td>
</tr>
<tr id="row280607102412"><td class="cellrowborder" valign="top" width="50%" headers="mcps1.2.3.1.1 "><p id="p7806070245"><a name="p7806070245"></a><a name="p7806070245"></a><a href="__hif822float2.md">__hif822float2</a></p>
</td>
<td class="cellrowborder" valign="top" width="50%" headers="mcps1.2.3.1.2 "><p id="p10806197132414"><a name="p10806197132414"></a><a name="p10806197132414"></a>将hifloat8x2_t类型输入转换成float2类型数据。</p>
</td>
</tr>
<tr id="row20448812112418"><td class="cellrowborder" valign="top" width="50%" headers="mcps1.2.3.1.1 "><p id="p4448121213244"><a name="p4448121213244"></a><a name="p4448121213244"></a><a href="__hif822half2.md">__hif822half2</a></p>
</td>
<td class="cellrowborder" valign="top" width="50%" headers="mcps1.2.3.1.2 "><p id="p154482127248"><a name="p154482127248"></a><a name="p154482127248"></a>将hifloat8x2_t类型输入转换成half2类型数据。</p>
</td>
</tr>
<tr id="row1575831792416"><td class="cellrowborder" valign="top" width="50%" headers="mcps1.2.3.1.1 "><p id="p11759201772416"><a name="p11759201772416"></a><a name="p11759201772416"></a><a href="__e4m3x22float2.md">__e4m3x22float2</a></p>
</td>
<td class="cellrowborder" valign="top" width="50%" headers="mcps1.2.3.1.2 "><p id="p675931712241"><a name="p675931712241"></a><a name="p675931712241"></a>将float8_e4m3x2_t类型输入转换成的float2类型数据。</p>
</td>
</tr>
<tr id="row10433172116248"><td class="cellrowborder" valign="top" width="50%" headers="mcps1.2.3.1.1 "><p id="p4434122192414"><a name="p4434122192414"></a><a name="p4434122192414"></a><a href="__e5m2x22float2.md">__e5m2x22float2</a></p>
</td>
<td class="cellrowborder" valign="top" width="50%" headers="mcps1.2.3.1.2 "><p id="p1943472114242"><a name="p1943472114242"></a><a name="p1943472114242"></a>将float8_e5m2x2_t类型输入转换成的float2类型数据。</p>
</td>
</tr>
<tr id="row05291117101018"><td class="cellrowborder" valign="top" width="50%" headers="mcps1.2.3.1.1 "><p id="p352911716104"><a name="p352911716104"></a><a name="p352911716104"></a><a href="__float2bfloat162_rn.md">__float2bfloat162_rn</a></p>
</td>
<td class="cellrowborder" valign="top" width="50%" headers="mcps1.2.3.1.2 "><p id="p1152916175103"><a name="p1152916175103"></a><a name="p1152916175103"></a>将float类型数据遵循CAST_RINT模式转换为bfloat16类型并填充到bfloat16x2的前后两部分，返回填充后的bfloat16x2类型数据。</p>
</td>
</tr>
<tr id="row1456833210101"><td class="cellrowborder" valign="top" width="50%" headers="mcps1.2.3.1.1 "><p id="p175681532111011"><a name="p175681532111011"></a><a name="p175681532111011"></a><a href="__floats2bfloat162_rn.md">__floats2bfloat162_rn</a></p>
</td>
<td class="cellrowborder" valign="top" width="50%" headers="mcps1.2.3.1.2 "><p id="p1568103211018"><a name="p1568103211018"></a><a name="p1568103211018"></a>将输入的数据x，y遵循CAST_RINT模式分别转换为bfloat16类型并填充到bfloat16x2的前后两部分，返回转换后的bfloat16x2类型数据。</p>
</td>
</tr>
<tr id="row73981930151017"><td class="cellrowborder" valign="top" width="50%" headers="mcps1.2.3.1.1 "><p id="p143988309102"><a name="p143988309102"></a><a name="p143988309102"></a><a href="__float22bfloat162_rn.md">__float22bfloat162_rn</a></p>
</td>
<td class="cellrowborder" valign="top" width="50%" headers="mcps1.2.3.1.2 "><p id="p525764812013"><a name="p525764812013"></a><a name="p525764812013"></a>将float2类型数据遵循CAST_RINT模式转换为bfloat16x2类型，返回转换后的bfloat16x2类型数据。</p>
</td>
</tr>
<tr id="row4383128141010"><td class="cellrowborder" valign="top" width="50%" headers="mcps1.2.3.1.1 "><p id="p11383162811100"><a name="p11383162811100"></a><a name="p11383162811100"></a><a href="__bfloat162bfloat162.md">__bfloat162bfloat162</a></p>
</td>
<td class="cellrowborder" valign="top" width="50%" headers="mcps1.2.3.1.2 "><p id="p13553115571119"><a name="p13553115571119"></a><a name="p13553115571119"></a>将输入的数据的填充为bfloat16x2前后两个分量，返回转换后的bfloat16x2类型数据。</p>
</td>
</tr>
<tr id="row1839911269109"><td class="cellrowborder" valign="top" width="50%" headers="mcps1.2.3.1.1 "><p id="p18399202613101"><a name="p18399202613101"></a><a name="p18399202613101"></a><a href="__halves2bfloat162.md">__halves2bfloat162</a></p>
</td>
<td class="cellrowborder" valign="top" width="50%" headers="mcps1.2.3.1.2 "><p id="p13992268103"><a name="p13992268103"></a><a name="p13992268103"></a>将输入的数据分别填充为bfloat16x2前后两个分量，返回填充后数据。</p>
</td>
</tr>
<tr id="row4335424141014"><td class="cellrowborder" valign="top" width="50%" headers="mcps1.2.3.1.1 "><p id="p153369245109"><a name="p153369245109"></a><a name="p153369245109"></a><a href="__high2bfloat16.md">__high2bfloat16</a></p>
</td>
<td class="cellrowborder" valign="top" width="50%" headers="mcps1.2.3.1.2 "><p id="p6336122414106"><a name="p6336122414106"></a><a name="p6336122414106"></a>提取输入bfloat16x2的高16位，并返回。</p>
</td>
</tr>
<tr id="row2036132221016"><td class="cellrowborder" valign="top" width="50%" headers="mcps1.2.3.1.1 "><p id="p2361922131011"><a name="p2361922131011"></a><a name="p2361922131011"></a><a href="__high2bfloat162.md">__high2bfloat162</a></p>
</td>
<td class="cellrowborder" valign="top" width="50%" headers="mcps1.2.3.1.2 "><p id="p636122141016"><a name="p636122141016"></a><a name="p636122141016"></a>将输入数据的的高16位填充到bfloat16x2并返回结果。</p>
</td>
</tr>
<tr id="row67529199109"><td class="cellrowborder" valign="top" width="50%" headers="mcps1.2.3.1.1 "><p id="p107521119121011"><a name="p107521119121011"></a><a name="p107521119121011"></a><a href="__high2float.md">__high2float</a></p>
</td>
<td class="cellrowborder" valign="top" width="50%" headers="mcps1.2.3.1.2 "><p id="p97641819142114"><a name="p97641819142114"></a><a name="p97641819142114"></a>将输入数据的高16位转换为float类型并返回结果。</p>
</td>
</tr>
<tr id="row952982181417"><td class="cellrowborder" valign="top" width="50%" headers="mcps1.2.3.1.1 "><p id="p5529132119143"><a name="p5529132119143"></a><a name="p5529132119143"></a><a href="__highs2bfloat162.md">__highs2bfloat162</a></p>
</td>
<td class="cellrowborder" valign="top" width="50%" headers="mcps1.2.3.1.2 "><p id="p1852992121415"><a name="p1852992121415"></a><a name="p1852992121415"></a>分别提取两个bfloat162输入的高16 位，并填充到bfloat162中。返回填充后的数据。</p>
</td>
</tr>
<tr id="row943351916149"><td class="cellrowborder" valign="top" width="50%" headers="mcps1.2.3.1.1 "><p id="p543316194140"><a name="p543316194140"></a><a name="p543316194140"></a><a href="__low2bfloat16.md">__low2bfloat16</a></p>
</td>
<td class="cellrowborder" valign="top" width="50%" headers="mcps1.2.3.1.2 "><p id="p2053632510351"><a name="p2053632510351"></a><a name="p2053632510351"></a>返回输入数据的低16位。</p>
</td>
</tr>
<tr id="row115211714146"><td class="cellrowborder" valign="top" width="50%" headers="mcps1.2.3.1.1 "><p id="p21522017171418"><a name="p21522017171418"></a><a name="p21522017171418"></a><a href="__low2bfloat162.md">__low2bfloat162</a></p>
</td>
<td class="cellrowborder" valign="top" width="50%" headers="mcps1.2.3.1.2 "><p id="p759113379212"><a name="p759113379212"></a><a name="p759113379212"></a>将输入数据的低16位填充到bfloat16x2并返回。</p>
</td>
</tr>
<tr id="row9827151417140"><td class="cellrowborder" valign="top" width="50%" headers="mcps1.2.3.1.1 "><p id="p6827161431416"><a name="p6827161431416"></a><a name="p6827161431416"></a><a href="__low2float.md">__low2float</a></p>
</td>
<td class="cellrowborder" valign="top" width="50%" headers="mcps1.2.3.1.2 "><p id="p14865174217214"><a name="p14865174217214"></a><a name="p14865174217214"></a>将输入数据的低16位转换为浮点数并返回结果。</p>
</td>
</tr>
<tr id="row3653201211410"><td class="cellrowborder" valign="top" width="50%" headers="mcps1.2.3.1.1 "><p id="p4653111213147"><a name="p4653111213147"></a><a name="p4653111213147"></a><a href="__lowhigh2highlow.md">__lowhigh2highlow</a></p>
</td>
<td class="cellrowborder" valign="top" width="50%" headers="mcps1.2.3.1.2 "><p id="p103686487211"><a name="p103686487211"></a><a name="p103686487211"></a><span>将输入数据的高低16位进行交换并返回</span>。</p>
</td>
</tr>
<tr id="row610815277156"><td class="cellrowborder" valign="top" width="50%" headers="mcps1.2.3.1.1 "><p id="p710816273153"><a name="p710816273153"></a><a name="p710816273153"></a><a href="__lows2bfloat162.md">__lows2bfloat162</a></p>
</td>
<td class="cellrowborder" valign="top" width="50%" headers="mcps1.2.3.1.2 "><p id="p3108527131516"><a name="p3108527131516"></a><a name="p3108527131516"></a>分别提取两个bfloat162输入的低16 位，并填充到bfloat162中。返回填充后的数据。</p>
</td>
</tr>
<tr id="row7931251154"><td class="cellrowborder" valign="top" width="50%" headers="mcps1.2.3.1.1 "><p id="p593132521514"><a name="p593132521514"></a><a name="p593132521514"></a><a href="__bfloat1622float2.md">__bfloat1622float2</a></p>
</td>
<td class="cellrowborder" valign="top" width="50%" headers="mcps1.2.3.1.2 "><p id="p1219410017224"><a name="p1219410017224"></a><a name="p1219410017224"></a>将bfloat16x2的两个分量分别转换为float，并填充到float2返回。</p>
</td>
</tr>
<tr id="row575712211152"><td class="cellrowborder" valign="top" width="50%" headers="mcps1.2.3.1.1 "><p id="p675722216157"><a name="p675722216157"></a><a name="p675722216157"></a><a href="__floats2half2_rn.md">__floats2half2_rn</a></p>
</td>
<td class="cellrowborder" valign="top" width="50%" headers="mcps1.2.3.1.2 "><p id="p18757142219159"><a name="p18757142219159"></a><a name="p18757142219159"></a>将输入的数据x，y遵循CAST_RINT模式分别转换为bfloat16类型并填充到half2的前后两部分，返回转换后的half2类型数据。</p>
</td>
</tr>
<tr id="row20251125917444"><td class="cellrowborder" valign="top" width="50%" headers="mcps1.2.3.1.1 "><p id="p125155917443"><a name="p125155917443"></a><a name="p125155917443"></a><a href="__float22half2_rn.md">__float22half2_rn</a></p>
</td>
<td class="cellrowborder" valign="top" width="50%" headers="mcps1.2.3.1.2 "><p id="p15946161820463"><a name="p15946161820463"></a><a name="p15946161820463"></a>将float2类型数据遵循CAST_RINT模式转换为half2类型，返回转换后的half2类型数据。</p>
</td>
</tr>
<tr id="row1873017184167"><td class="cellrowborder" valign="top" width="50%" headers="mcps1.2.3.1.1 "><p id="p14730318111610"><a name="p14730318111610"></a><a name="p14730318111610"></a><a href="__low2half.md">__low2half</a></p>
</td>
<td class="cellrowborder" valign="top" width="50%" headers="mcps1.2.3.1.2 "><p id="p77641063424"><a name="p77641063424"></a><a name="p77641063424"></a>返回输入数据的低16位。</p>
</td>
</tr>
<tr id="row02801021181619"><td class="cellrowborder" valign="top" width="50%" headers="mcps1.2.3.1.1 "><p id="p14281132110162"><a name="p14281132110162"></a><a name="p14281132110162"></a><a href="__low2half2.md">__low2half2</a></p>
</td>
<td class="cellrowborder" valign="top" width="50%" headers="mcps1.2.3.1.2 "><p id="p11233155292213"><a name="p11233155292213"></a><a name="p11233155292213"></a>将输入数据的低16位填充到half2并返回。</p>
</td>
</tr>
<tr id="row13561175241716"><td class="cellrowborder" valign="top" width="50%" headers="mcps1.2.3.1.1 "><p id="p1256155216173"><a name="p1256155216173"></a><a name="p1256155216173"></a><a href="__high2half.md">__high2half</a></p>
</td>
<td class="cellrowborder" valign="top" width="50%" headers="mcps1.2.3.1.2 "><p id="p356114529175"><a name="p356114529175"></a><a name="p356114529175"></a>提取输入half2的高16位，并返回</p>
</td>
</tr>
<tr id="row5487250141711"><td class="cellrowborder" valign="top" width="50%" headers="mcps1.2.3.1.1 "><p id="p948711505171"><a name="p948711505171"></a><a name="p948711505171"></a><a href="__high2half2.md">__high2half2</a></p>
</td>
<td class="cellrowborder" valign="top" width="50%" headers="mcps1.2.3.1.2 "><p id="p1866142102315"><a name="p1866142102315"></a><a name="p1866142102315"></a>将输入数据的的高16位填充到half2并返回结果。</p>
</td>
</tr>
<tr id="row1881673318245"><td class="cellrowborder" valign="top" width="50%" headers="mcps1.2.3.1.1 "><p id="p13540173112612"><a name="p13540173112612"></a><a name="p13540173112612"></a><a href="__highs2half2.md">__highs2half2</a></p>
</td>
<td class="cellrowborder" valign="top" width="50%" headers="mcps1.2.3.1.2 "><p id="p8681550112216"><a name="p8681550112216"></a><a name="p8681550112216"></a>分别提取两个half2输入的低16 位，并填充到half2中。返回填充后的数据。</p>
</td>
</tr>
<tr id="row537874817179"><td class="cellrowborder" valign="top" width="50%" headers="mcps1.2.3.1.1 "><p id="p1037813488177"><a name="p1037813488177"></a><a name="p1037813488177"></a><a href="__lows2half2.md">__lows2half2</a></p>
</td>
<td class="cellrowborder" valign="top" width="50%" headers="mcps1.2.3.1.2 "><p id="p994916302314"><a name="p994916302314"></a><a name="p994916302314"></a>分别提取两个half2输入的低16 位，并填充到half2中。返回填充后的数据。</p>
</td>
</tr>
<tr id="row19298162661811"><td class="cellrowborder" valign="top" width="50%" headers="mcps1.2.3.1.1 "><p id="p529822681818"><a name="p529822681818"></a><a name="p529822681818"></a><a href="__halves2half2.md">__halves2half2</a></p>
</td>
<td class="cellrowborder" valign="top" width="50%" headers="mcps1.2.3.1.2 "><p id="p132412402410"><a name="p132412402410"></a><a name="p132412402410"></a>将输入的数据分别填充为half2前后两个分量，返回填充后数据。</p>
</td>
</tr>
<tr id="row14926142381811"><td class="cellrowborder" valign="top" width="50%" headers="mcps1.2.3.1.1 "><p id="p10927023151818"><a name="p10927023151818"></a><a name="p10927023151818"></a><a href="__half22float2.md">__half22float2</a></p>
</td>
<td class="cellrowborder" valign="top" width="50%" headers="mcps1.2.3.1.2 "><p id="p1992752310188"><a name="p1992752310188"></a><a name="p1992752310188"></a>将half2的两个分量分别转换为float，并填充到float2返回。</p>
</td>
</tr>
<tr id="row641383612477"><td class="cellrowborder" valign="top" width="50%" headers="mcps1.2.3.1.1 "><p id="p184130363474"><a name="p184130363474"></a><a name="p184130363474"></a><a href="__int_as_float.md">__int_as_float</a></p>
</td>
<td class="cellrowborder" valign="top" width="50%" headers="mcps1.2.3.1.2 "><p id="p1289973313162"><a name="p1289973313162"></a><a name="p1289973313162"></a><span>将整数中的位重新解释为浮点数</span>。</p>
</td>
</tr>
<tr id="row14413436124712"><td class="cellrowborder" valign="top" width="50%" headers="mcps1.2.3.1.1 "><p id="p3413103684714"><a name="p3413103684714"></a><a name="p3413103684714"></a><a href="__uint_as_float.md">__uint_as_float</a></p>
</td>
<td class="cellrowborder" valign="top" width="50%" headers="mcps1.2.3.1.2 "><p id="p167113412169"><a name="p167113412169"></a><a name="p167113412169"></a><span>将无符号整数中的位重新解释为浮点数</span>。</p>
</td>
</tr>
<tr id="row1998818470557"><td class="cellrowborder" valign="top" width="50%" headers="mcps1.2.3.1.1 "><p id="p1198816478553"><a name="p1198816478553"></a><a name="p1198816478553"></a><a href="__float_as_int.md">__float_as_int</a></p>
</td>
<td class="cellrowborder" valign="top" width="50%" headers="mcps1.2.3.1.2 "><p id="p13988194765514"><a name="p13988194765514"></a><a name="p13988194765514"></a><span>将浮点数中的位重新解释为有符号整数</span>。</p>
</td>
</tr>
<tr id="row1841319365476"><td class="cellrowborder" valign="top" width="50%" headers="mcps1.2.3.1.1 "><p id="p1413163611474"><a name="p1413163611474"></a><a name="p1413163611474"></a><a href="__float_as_uint.md">__float_as_uint</a></p>
</td>
<td class="cellrowborder" valign="top" width="50%" headers="mcps1.2.3.1.2 "><p id="p127231834105512"><a name="p127231834105512"></a><a name="p127231834105512"></a><span>将浮点数中的位重新解释为无符号整数</span>。</p>
</td>
</tr>
<tr id="row1932922310193"><td class="cellrowborder" valign="top" width="50%" headers="mcps1.2.3.1.1 "><p id="p73293236199"><a name="p73293236199"></a><a name="p73293236199"></a><a href="__ushort_as_half.md">__ushort_as_half</a></p>
</td>
<td class="cellrowborder" valign="top" width="50%" headers="mcps1.2.3.1.2 "><p id="p143291623151912"><a name="p143291623151912"></a><a name="p143291623151912"></a><span>将unsigned short int的按位重新解释为half，即将unsigned short int的数据存储的位按照half的格式进行读取</span>。</p>
</td>
</tr>
<tr id="row12888122571911"><td class="cellrowborder" valign="top" width="50%" headers="mcps1.2.3.1.1 "><p id="p1688842514190"><a name="p1688842514190"></a><a name="p1688842514190"></a><a href="__ushort_as_bfloat16.md">__ushort_as_bfloat16</a></p>
</td>
<td class="cellrowborder" valign="top" width="50%" headers="mcps1.2.3.1.2 "><p id="p1688822518193"><a name="p1688822518193"></a><a name="p1688822518193"></a><span>将unsigned short int的按位重新解释为bfloat16，即将unsigned short int的数据存储的位按照bfloat16的格式进行读取。</span></p>
</td>
</tr>
</tbody>
</table>

**表 35**  向量类型构造函数

<a name="table3436715145018"></a>
<table><thead align="left"><tr id="row19436111511507"><th class="cellrowborder" valign="top" width="40%" id="mcps1.2.3.1.1"><p id="p943615158508"><a name="p943615158508"></a><a name="p943615158508"></a>接口名</p>
</th>
<th class="cellrowborder" valign="top" width="60%" id="mcps1.2.3.1.2"><p id="p11436515105015"><a name="p11436515105015"></a><a name="p11436515105015"></a>功能描述</p>
</th>
</tr>
</thead>
<tbody><tr id="row5436131513501"><td class="cellrowborder" valign="top" width="40%" headers="mcps1.2.3.1.1 "><p id="p10436151512506"><a name="p10436151512506"></a><a name="p10436151512506"></a><a href="make_int2.md">make_int2</a></p>
</td>
<td class="cellrowborder" valign="top" width="60%" headers="mcps1.2.3.1.2 "><p id="p17436715145016"><a name="p17436715145016"></a><a name="p17436715145016"></a>从两个int类型数据创建int2类型的向量。</p>
</td>
</tr>
<tr id="row471121118511"><td class="cellrowborder" valign="top" width="40%" headers="mcps1.2.3.1.1 "><p id="p171171145115"><a name="p171171145115"></a><a name="p171171145115"></a><a href="make_int3.md">make_int3</a></p>
</td>
<td class="cellrowborder" valign="top" width="60%" headers="mcps1.2.3.1.2 "><p id="p071131195113"><a name="p071131195113"></a><a name="p071131195113"></a>从三个int类型数据创建int3类型的向量。</p>
</td>
</tr>
<tr id="row5166195105213"><td class="cellrowborder" valign="top" width="40%" headers="mcps1.2.3.1.1 "><p id="p1216611519526"><a name="p1216611519526"></a><a name="p1216611519526"></a><a href="make_int4.md">make_int4</a></p>
</td>
<td class="cellrowborder" valign="top" width="60%" headers="mcps1.2.3.1.2 "><p id="p171667585211"><a name="p171667585211"></a><a name="p171667585211"></a>从四个int类型数据创建int4类型的向量。</p>
</td>
</tr>
<tr id="row13344124135117"><td class="cellrowborder" valign="top" width="40%" headers="mcps1.2.3.1.1 "><p id="p19344102414516"><a name="p19344102414516"></a><a name="p19344102414516"></a><a href="make_uint2.md">make_uint2</a></p>
</td>
<td class="cellrowborder" valign="top" width="60%" headers="mcps1.2.3.1.2 "><p id="p7344202405118"><a name="p7344202405118"></a><a name="p7344202405118"></a>从两个unsigned int类型数据创建uint2类型的向量。</p>
</td>
</tr>
<tr id="row61047309518"><td class="cellrowborder" valign="top" width="40%" headers="mcps1.2.3.1.1 "><p id="p19104113085118"><a name="p19104113085118"></a><a name="p19104113085118"></a><a href="make_uint3.md">make_uint3</a></p>
</td>
<td class="cellrowborder" valign="top" width="60%" headers="mcps1.2.3.1.2 "><p id="p8104113015111"><a name="p8104113015111"></a><a name="p8104113015111"></a>从三个unsigned int类型数据创建uint3类型的向量。</p>
</td>
</tr>
<tr id="row1391812612523"><td class="cellrowborder" valign="top" width="40%" headers="mcps1.2.3.1.1 "><p id="p1891818625217"><a name="p1891818625217"></a><a name="p1891818625217"></a><a href="make_uint4.md">make_uint4</a></p>
</td>
<td class="cellrowborder" valign="top" width="60%" headers="mcps1.2.3.1.2 "><p id="p291896165212"><a name="p291896165212"></a><a name="p291896165212"></a>从四个unsigned int类型数据创建uint4类型的向量。</p>
</td>
</tr>
<tr id="row15545141175217"><td class="cellrowborder" valign="top" width="40%" headers="mcps1.2.3.1.1 "><p id="p254581165216"><a name="p254581165216"></a><a name="p254581165216"></a><a href="make_ulonglong2.md">make_ulonglong2</a></p>
</td>
<td class="cellrowborder" valign="top" width="60%" headers="mcps1.2.3.1.2 "><p id="p46955638"><a name="p46955638"></a><a name="p46955638"></a>从两个unsigned long long int类型数据创建ulonglong2类型的向量。</p>
</td>
</tr>
<tr id="row166217509511"><td class="cellrowborder" valign="top" width="40%" headers="mcps1.2.3.1.1 "><p id="p362145045113"><a name="p362145045113"></a><a name="p362145045113"></a><a href="make_ulonglong3.md">make_ulonglong3</a></p>
</td>
<td class="cellrowborder" valign="top" width="60%" headers="mcps1.2.3.1.2 "><p id="p1862115018516"><a name="p1862115018516"></a><a name="p1862115018516"></a>从三个unsigned long long int类型数据创建ulonglong3类型的向量。</p>
</td>
</tr>
<tr id="row1566110525518"><td class="cellrowborder" valign="top" width="40%" headers="mcps1.2.3.1.1 "><p id="p1966165210514"><a name="p1966165210514"></a><a name="p1966165210514"></a><a href="make_ulonglong4.md">make_ulonglong4</a></p>
</td>
<td class="cellrowborder" valign="top" width="60%" headers="mcps1.2.3.1.2 "><p id="p1266165265110"><a name="p1266165265110"></a><a name="p1266165265110"></a>从四个unsigned long long int类型数据创建ulonglong4类型的向量。</p>
</td>
</tr>
<tr id="row332116913524"><td class="cellrowborder" valign="top" width="40%" headers="mcps1.2.3.1.1 "><p id="p10321109175210"><a name="p10321109175210"></a><a name="p10321109175210"></a><a href="make_longlong2.md">make_longlong2</a></p>
</td>
<td class="cellrowborder" valign="top" width="60%" headers="mcps1.2.3.1.2 "><p id="p163211795526"><a name="p163211795526"></a><a name="p163211795526"></a>从两个long long int类型数据创建longlong2类型的向量。</p>
</td>
</tr>
<tr id="row886905455115"><td class="cellrowborder" valign="top" width="40%" headers="mcps1.2.3.1.1 "><p id="p0869145418514"><a name="p0869145418514"></a><a name="p0869145418514"></a><a href="make_longlong3.md">make_longlong3</a></p>
</td>
<td class="cellrowborder" valign="top" width="60%" headers="mcps1.2.3.1.2 "><p id="p2869175416510"><a name="p2869175416510"></a><a name="p2869175416510"></a>从三个long long int类型数据创建longlong3类型的向量。</p>
</td>
</tr>
<tr id="row63747465512"><td class="cellrowborder" valign="top" width="40%" headers="mcps1.2.3.1.1 "><p id="p203741046195110"><a name="p203741046195110"></a><a name="p203741046195110"></a><a href="make_longlong4.md">make_longlong4</a></p>
</td>
<td class="cellrowborder" valign="top" width="60%" headers="mcps1.2.3.1.2 "><p id="p12374246175113"><a name="p12374246175113"></a><a name="p12374246175113"></a>从四个long long int类型数据创建longlong4类型的向量。</p>
</td>
</tr>
<tr id="row206401548175114"><td class="cellrowborder" valign="top" width="40%" headers="mcps1.2.3.1.1 "><p id="p176401548135115"><a name="p176401548135115"></a><a name="p176401548135115"></a><a href="make_ulong2.md">make_ulong2</a></p>
</td>
<td class="cellrowborder" valign="top" width="60%" headers="mcps1.2.3.1.2 "><p id="p13640114895111"><a name="p13640114895111"></a><a name="p13640114895111"></a>从两个unsigned long int类型数据创建ulong2类型的向量。</p>
</td>
</tr>
<tr id="row1546113444515"><td class="cellrowborder" valign="top" width="40%" headers="mcps1.2.3.1.1 "><p id="p124619442511"><a name="p124619442511"></a><a name="p124619442511"></a><a href="make_ulong3.md">make_ulong3</a></p>
</td>
<td class="cellrowborder" valign="top" width="60%" headers="mcps1.2.3.1.2 "><p id="p12461174418516"><a name="p12461174418516"></a><a name="p12461174418516"></a>从三个unsigned long int类型数据创建ulong3类型的向量。</p>
</td>
</tr>
<tr id="row1720394225114"><td class="cellrowborder" valign="top" width="40%" headers="mcps1.2.3.1.1 "><p id="p8203442175111"><a name="p8203442175111"></a><a name="p8203442175111"></a><a href="make_ulong4.md">make_ulong4</a></p>
</td>
<td class="cellrowborder" valign="top" width="60%" headers="mcps1.2.3.1.2 "><p id="p114819561638"><a name="p114819561638"></a><a name="p114819561638"></a>从四个unsigned long int类型数据创建ulong4类型的向量。</p>
</td>
</tr>
<tr id="row98461139185117"><td class="cellrowborder" valign="top" width="40%" headers="mcps1.2.3.1.1 "><p id="p18467398515"><a name="p18467398515"></a><a name="p18467398515"></a><a href="make_long2.md">make_long2</a></p>
</td>
<td class="cellrowborder" valign="top" width="60%" headers="mcps1.2.3.1.2 "><p id="p36897217412"><a name="p36897217412"></a><a name="p36897217412"></a>从两个long int类型数据创建long2类型的向量。</p>
</td>
</tr>
<tr id="row1772153719511"><td class="cellrowborder" valign="top" width="40%" headers="mcps1.2.3.1.1 "><p id="p187721637115114"><a name="p187721637115114"></a><a name="p187721637115114"></a><a href="make_long3.md">make_long3</a></p>
</td>
<td class="cellrowborder" valign="top" width="60%" headers="mcps1.2.3.1.2 "><p id="p377293785116"><a name="p377293785116"></a><a name="p377293785116"></a>从三个long int类型数据创建long3类型的向量。</p>
</td>
</tr>
<tr id="row7469193515110"><td class="cellrowborder" valign="top" width="40%" headers="mcps1.2.3.1.1 "><p id="p14469335135113"><a name="p14469335135113"></a><a name="p14469335135113"></a><a href="make_long4.md">make_long4</a></p>
</td>
<td class="cellrowborder" valign="top" width="60%" headers="mcps1.2.3.1.2 "><p id="p124701035125115"><a name="p124701035125115"></a><a name="p124701035125115"></a>从四个long int类型数据创建long4类型的向量。</p>
</td>
</tr>
<tr id="row3494352205820"><td class="cellrowborder" valign="top" width="40%" headers="mcps1.2.3.1.1 "><p id="p1349475210586"><a name="p1349475210586"></a><a name="p1349475210586"></a><a href="make_float2.md">make_float2</a></p>
</td>
<td class="cellrowborder" valign="top" width="60%" headers="mcps1.2.3.1.2 "><p id="p19087201843"><a name="p19087201843"></a><a name="p19087201843"></a>从两个float类型数据创建float2类型的向量。</p>
</td>
</tr>
<tr id="row1065075485813"><td class="cellrowborder" valign="top" width="40%" headers="mcps1.2.3.1.1 "><p id="p18650115414587"><a name="p18650115414587"></a><a name="p18650115414587"></a><a href="make_float3.md">make_float3</a></p>
</td>
<td class="cellrowborder" valign="top" width="60%" headers="mcps1.2.3.1.2 "><p id="p17517172713410"><a name="p17517172713410"></a><a name="p17517172713410"></a>从三个float类型数据创建float3类型的向量。</p>
</td>
</tr>
<tr id="row19114201565110"><td class="cellrowborder" valign="top" width="40%" headers="mcps1.2.3.1.1 "><p id="p3114131514512"><a name="p3114131514512"></a><a name="p3114131514512"></a><a href="make_float4.md">make_float4</a></p>
</td>
<td class="cellrowborder" valign="top" width="60%" headers="mcps1.2.3.1.2 "><p id="p611401545111"><a name="p611401545111"></a><a name="p611401545111"></a>从四个float类型数据创建float4类型的向量。</p>
</td>
</tr>
<tr id="row1234817505587"><td class="cellrowborder" valign="top" width="40%" headers="mcps1.2.3.1.1 "><p id="p1234835045811"><a name="p1234835045811"></a><a name="p1234835045811"></a><a href="make_short2.md">make_short2</a></p>
</td>
<td class="cellrowborder" valign="top" width="60%" headers="mcps1.2.3.1.2 "><p id="p4348145065814"><a name="p4348145065814"></a><a name="p4348145065814"></a>从两个short类型数据创建short2类型的向量。</p>
</td>
</tr>
<tr id="row13693945165816"><td class="cellrowborder" valign="top" width="40%" headers="mcps1.2.3.1.1 "><p id="p8693204510583"><a name="p8693204510583"></a><a name="p8693204510583"></a><a href="make_short3.md">make_short3</a></p>
</td>
<td class="cellrowborder" valign="top" width="60%" headers="mcps1.2.3.1.2 "><p id="p7693245115812"><a name="p7693245115812"></a><a name="p7693245115812"></a>从三个short类型数据创建short3类型的向量。</p>
</td>
</tr>
<tr id="row1232211421586"><td class="cellrowborder" valign="top" width="40%" headers="mcps1.2.3.1.1 "><p id="p5322194295812"><a name="p5322194295812"></a><a name="p5322194295812"></a><a href="make_short4.md">make_short4</a></p>
</td>
<td class="cellrowborder" valign="top" width="60%" headers="mcps1.2.3.1.2 "><p id="p1732274265815"><a name="p1732274265815"></a><a name="p1732274265815"></a>从四个short类型数据创建short4类型的向量。</p>
</td>
</tr>
<tr id="row5926134715581"><td class="cellrowborder" valign="top" width="40%" headers="mcps1.2.3.1.1 "><p id="p1792654725810"><a name="p1792654725810"></a><a name="p1792654725810"></a><a href="make_ushort2.md">make_ushort2</a></p>
</td>
<td class="cellrowborder" valign="top" width="60%" headers="mcps1.2.3.1.2 "><p id="p203031851"><a name="p203031851"></a><a name="p203031851"></a>从两个unsigned short类型数据创建ushort2类型的向量。</p>
</td>
</tr>
<tr id="row51141515165113"><td class="cellrowborder" valign="top" width="40%" headers="mcps1.2.3.1.1 "><p id="p1411411516519"><a name="p1411411516519"></a><a name="p1411411516519"></a><a href="make_ushort3.md">make_ushort3</a></p>
</td>
<td class="cellrowborder" valign="top" width="60%" headers="mcps1.2.3.1.2 "><p id="p61141115135110"><a name="p61141115135110"></a><a name="p61141115135110"></a>从三个unsigned short类型数据创建ushort3类型的向量。</p>
</td>
</tr>
<tr id="row181941338705"><td class="cellrowborder" valign="top" width="40%" headers="mcps1.2.3.1.1 "><p id="p1119412381013"><a name="p1119412381013"></a><a name="p1119412381013"></a><a href="make_ushort4.md">make_ushort4</a></p>
</td>
<td class="cellrowborder" valign="top" width="60%" headers="mcps1.2.3.1.2 "><p id="p1719463816011"><a name="p1719463816011"></a><a name="p1719463816011"></a>从四个unsigned short类型数据创建ushort4类型的向量。</p>
</td>
</tr>
<tr id="row199058480015"><td class="cellrowborder" valign="top" width="40%" headers="mcps1.2.3.1.1 "><p id="p19906348608"><a name="p19906348608"></a><a name="p19906348608"></a><a href="make_uchar2.md">make_uchar2</a></p>
</td>
<td class="cellrowborder" valign="top" width="60%" headers="mcps1.2.3.1.2 "><p id="p6319121712510"><a name="p6319121712510"></a><a name="p6319121712510"></a>从两个unsigned char类型数据创建uchar2类型的向量。</p>
</td>
</tr>
<tr id="row1569912431201"><td class="cellrowborder" valign="top" width="40%" headers="mcps1.2.3.1.1 "><p id="p136991843601"><a name="p136991843601"></a><a name="p136991843601"></a><a href="make_uchar3.md">make_uchar3</a></p>
</td>
<td class="cellrowborder" valign="top" width="60%" headers="mcps1.2.3.1.2 "><p id="p26991643609"><a name="p26991643609"></a><a name="p26991643609"></a>从三个unsigned char类型数据创建uchar3类型的向量。</p>
</td>
</tr>
<tr id="row1862216461104"><td class="cellrowborder" valign="top" width="40%" headers="mcps1.2.3.1.1 "><p id="p16622144620018"><a name="p16622144620018"></a><a name="p16622144620018"></a><a href="make_uchar4.md">make_uchar4</a></p>
</td>
<td class="cellrowborder" valign="top" width="60%" headers="mcps1.2.3.1.2 "><p id="p2301729454"><a name="p2301729454"></a><a name="p2301729454"></a>从四个unsigned char类型数据创建uchar4类型的向量。</p>
</td>
</tr>
<tr id="row206531940901"><td class="cellrowborder" valign="top" width="40%" headers="mcps1.2.3.1.1 "><p id="p465464012019"><a name="p465464012019"></a><a name="p465464012019"></a><a href="make_char2.md">make_char2</a></p>
</td>
<td class="cellrowborder" valign="top" width="60%" headers="mcps1.2.3.1.2 "><p id="p1765474013012"><a name="p1765474013012"></a><a name="p1765474013012"></a>从两个signed char类型数据创建char2类型的向量。</p>
</td>
</tr>
<tr id="row1998510301405"><td class="cellrowborder" valign="top" width="40%" headers="mcps1.2.3.1.1 "><p id="p4985163016014"><a name="p4985163016014"></a><a name="p4985163016014"></a><a href="make_char3.md">make_char3</a></p>
</td>
<td class="cellrowborder" valign="top" width="60%" headers="mcps1.2.3.1.2 "><p id="p898543017019"><a name="p898543017019"></a><a name="p898543017019"></a>从三个signed char类型数据创建char3类型的向量。</p>
</td>
</tr>
<tr id="row27831346010"><td class="cellrowborder" valign="top" width="40%" headers="mcps1.2.3.1.1 "><p id="p878316341305"><a name="p878316341305"></a><a name="p878316341305"></a><a href="make_char4.md">make_char4</a></p>
</td>
<td class="cellrowborder" valign="top" width="60%" headers="mcps1.2.3.1.2 "><p id="p69292465518"><a name="p69292465518"></a><a name="p69292465518"></a>从四个signed char类型数据创建char4类型的向量。</p>
</td>
</tr>
<tr id="row127719411315"><td class="cellrowborder" valign="top" width="40%" headers="mcps1.2.3.1.1 "><p id="p027718415112"><a name="p027718415112"></a><a name="p027718415112"></a><a href="make_half2.md">make_half2</a></p>
</td>
<td class="cellrowborder" valign="top" width="60%" headers="mcps1.2.3.1.2 "><p id="p827784119114"><a name="p827784119114"></a><a name="p827784119114"></a>从两个half类型数据创建half2类型的向量。</p>
</td>
</tr>
<tr id="row335394413111"><td class="cellrowborder" valign="top" width="40%" headers="mcps1.2.3.1.1 "><p id="p03533441115"><a name="p03533441115"></a><a name="p03533441115"></a><a href="make_bfloat162.md">make_bfloat162</a></p>
</td>
<td class="cellrowborder" valign="top" width="60%" headers="mcps1.2.3.1.2 "><p id="p1135344418116"><a name="p1135344418116"></a><a name="p1135344418116"></a>从两个bfloat16_t类型数据创建bfloat16x2_t类型的向量。</p>
</td>
</tr>
</tbody>
</table>

**表 36**  使能Cache Hints的Load/Store函数

<a name="table17209142275014"></a>
<table><thead align="left"><tr id="row120922211507"><th class="cellrowborder" valign="top" width="40%" id="mcps1.2.3.1.1"><p id="p1220962210501"><a name="p1220962210501"></a><a name="p1220962210501"></a>接口名</p>
</th>
<th class="cellrowborder" valign="top" width="60%" id="mcps1.2.3.1.2"><p id="p620912295018"><a name="p620912295018"></a><a name="p620912295018"></a>功能描述</p>
</th>
</tr>
</thead>
<tbody><tr id="row180373015523"><td class="cellrowborder" valign="top" width="40%" headers="mcps1.2.3.1.1 "><p id="p68031330105213"><a name="p68031330105213"></a><a name="p68031330105213"></a><a href="asc_ldcg.md">asc_ldcg</a></p>
</td>
<td class="cellrowborder" valign="top" width="60%" headers="mcps1.2.3.1.2 "><p id="p2094931913815"><a name="p2094931913815"></a><a name="p2094931913815"></a>从L2 Cache加载缓存的数据，如果缓存命中，则直接返回数据。若未命中，则从<span id="ph4433133645514"><a name="ph4433133645514"></a><a name="ph4433133645514"></a>Global Memory</span>地址预加载数据缓存至L2 Cache，并返回数据。</p>
</td>
</tr>
<tr id="row185241328175218"><td class="cellrowborder" valign="top" width="40%" headers="mcps1.2.3.1.1 "><p id="p1352432816526"><a name="p1352432816526"></a><a name="p1352432816526"></a><a href="asc_ldca.md">asc_ldca</a></p>
</td>
<td class="cellrowborder" valign="top" width="60%" headers="mcps1.2.3.1.2 "><p id="p95231912151116"><a name="p95231912151116"></a><a name="p95231912151116"></a>首先从Data Cache加载缓存数据，若未命中，则尝试从L2 Cache加载。如果Data Cache和L2 Cache中均未找到所需数据，则从Global Memory中读取数据，然后将其缓存到L2 Cache和Data Cache中。</p>
</td>
</tr>
<tr id="row83103233528"><td class="cellrowborder" valign="top" width="40%" headers="mcps1.2.3.1.1 "><p id="p13310182315219"><a name="p13310182315219"></a><a name="p13310182315219"></a><a href="asc_stcg.md">asc_stcg</a></p>
</td>
<td class="cellrowborder" valign="top" width="60%" headers="mcps1.2.3.1.2 "><p id="p10735143121114"><a name="p10735143121114"></a><a name="p10735143121114"></a>将指定数据存储到Global Memory的地址address中，并缓存到L2 Cache，但不缓存至Data Cache。</p>
</td>
</tr>
<tr id="row142096228500"><td class="cellrowborder" valign="top" width="40%" headers="mcps1.2.3.1.1 "><p id="p2209172255013"><a name="p2209172255013"></a><a name="p2209172255013"></a><a href="asc_stwt.md">asc_stwt</a></p>
</td>
<td class="cellrowborder" valign="top" width="60%" headers="mcps1.2.3.1.2 "><p id="p19524191351212"><a name="p19524191351212"></a><a name="p19524191351212"></a>将指定数据存储到Global Memory的地址address中，并缓存至Data Cache和L2 Cache。</p>
</td>
</tr>
</tbody>
</table>

## Utils API<a name="section15221943104512"></a>

**表 37**  C++标准库API列表

<a name="table99801554584"></a>
<table><thead align="left"><tr id="row179811554088"><th class="cellrowborder" valign="top" width="37.71%" id="mcps1.2.3.1.1"><p id="p298155413815"><a name="p298155413815"></a><a name="p298155413815"></a>接口名</p>
</th>
<th class="cellrowborder" valign="top" width="62.29%" id="mcps1.2.3.1.2"><p id="p129815541982"><a name="p129815541982"></a><a name="p129815541982"></a>功能描述</p>
</th>
</tr>
</thead>
<tbody><tr id="row99811354881"><td class="cellrowborder" valign="top" width="37.71%" headers="mcps1.2.3.1.1 "><p id="p29811545811"><a name="p29811545811"></a><a name="p29811545811"></a><a href="max.md">max</a></p>
</td>
<td class="cellrowborder" valign="top" width="62.29%" headers="mcps1.2.3.1.2 "><p id="p138701595101"><a name="p138701595101"></a><a name="p138701595101"></a>比较相同数据类型的两个数中的最大值。</p>
</td>
</tr>
<tr id="row99818543818"><td class="cellrowborder" valign="top" width="37.71%" headers="mcps1.2.3.1.1 "><p id="p179811546812"><a name="p179811546812"></a><a name="p179811546812"></a><a href="min.md">min</a></p>
</td>
<td class="cellrowborder" valign="top" width="62.29%" headers="mcps1.2.3.1.2 "><p id="p986935911109"><a name="p986935911109"></a><a name="p986935911109"></a>比较相同数据类型的两个数中的最小值。</p>
</td>
</tr>
<tr id="row9612616122612"><td class="cellrowborder" valign="top" width="37.71%" headers="mcps1.2.3.1.1 "><p id="p690582982619"><a name="p690582982619"></a><a name="p690582982619"></a><a href="abs.md">abs</a></p>
</td>
<td class="cellrowborder" valign="top" width="62.29%" headers="mcps1.2.3.1.2 "><p id="p1161313165264"><a name="p1161313165264"></a><a name="p1161313165264"></a>获取输入数据的绝对值。</p>
</td>
</tr>
<tr id="row1711751919267"><td class="cellrowborder" valign="top" width="37.71%" headers="mcps1.2.3.1.1 "><p id="p14905729172617"><a name="p14905729172617"></a><a name="p14905729172617"></a><a href="sqrt.md">sqrt</a></p>
</td>
<td class="cellrowborder" valign="top" width="62.29%" headers="mcps1.2.3.1.2 "><p id="p5117419162612"><a name="p5117419162612"></a><a name="p5117419162612"></a>计算输入数据的平方根。</p>
</td>
</tr>
<tr id="row17982175418816"><td class="cellrowborder" valign="top" width="37.71%" headers="mcps1.2.3.1.1 "><p id="p298213542810"><a name="p298213542810"></a><a name="p298213542810"></a><a href="integer_sequence.md">integer_sequence</a></p>
</td>
<td class="cellrowborder" valign="top" width="62.29%" headers="mcps1.2.3.1.2 "><p id="p8869259101018"><a name="p8869259101018"></a><a name="p8869259101018"></a>用于生成一个整数序列。</p>
</td>
</tr>
<tr id="row139821454384"><td class="cellrowborder" valign="top" width="37.71%" headers="mcps1.2.3.1.1 "><p id="p1598219543820"><a name="p1598219543820"></a><a name="p1598219543820"></a><a href="tuple.md">tuple</a></p>
</td>
<td class="cellrowborder" valign="top" width="62.29%" headers="mcps1.2.3.1.2 "><p id="p88681259101020"><a name="p88681259101020"></a><a name="p88681259101020"></a>允许存储多个不同类型元素的容器。</p>
</td>
</tr>
<tr id="row19821854887"><td class="cellrowborder" valign="top" width="37.71%" headers="mcps1.2.3.1.1 "><p id="p1829135316100"><a name="p1829135316100"></a><a name="p1829135316100"></a><a href="get.md">get</a></p>
</td>
<td class="cellrowborder" valign="top" width="62.29%" headers="mcps1.2.3.1.2 "><p id="p42902537103"><a name="p42902537103"></a><a name="p42902537103"></a>从tuple容器中提取指定位置的元素。</p>
</td>
</tr>
<tr id="row19821254681"><td class="cellrowborder" valign="top" width="37.71%" headers="mcps1.2.3.1.1 "><p id="p9290175311107"><a name="p9290175311107"></a><a name="p9290175311107"></a><a href="make_tuple.md">make_tuple</a></p>
</td>
<td class="cellrowborder" valign="top" width="62.29%" headers="mcps1.2.3.1.2 "><p id="p028955381015"><a name="p028955381015"></a><a name="p028955381015"></a>用于便捷地创建tuple对象。</p>
</td>
</tr>
<tr id="row169828546818"><td class="cellrowborder" valign="top" width="37.71%" headers="mcps1.2.3.1.1 "><p id="p6289453151012"><a name="p6289453151012"></a><a name="p6289453151012"></a><a href="is_convertible.md">is_convertible</a></p>
</td>
<td class="cellrowborder" valign="top" width="62.29%" headers="mcps1.2.3.1.2 "><p id="p1028845351011"><a name="p1028845351011"></a><a name="p1028845351011"></a>在程序编译时判断两个类型之间是否可以进行隐式转换。</p>
</td>
</tr>
<tr id="row498311545815"><td class="cellrowborder" valign="top" width="37.71%" headers="mcps1.2.3.1.1 "><p id="p42881453151010"><a name="p42881453151010"></a><a name="p42881453151010"></a><a href="is_base_of.md">is_base_of</a></p>
</td>
<td class="cellrowborder" valign="top" width="62.29%" headers="mcps1.2.3.1.2 "><p id="p11287105315108"><a name="p11287105315108"></a><a name="p11287105315108"></a>在程序编译时判断一个类型是否为另一个类型的基类。</p>
</td>
</tr>
<tr id="row1598312541817"><td class="cellrowborder" valign="top" width="37.71%" headers="mcps1.2.3.1.1 "><p id="p172878539107"><a name="p172878539107"></a><a name="p172878539107"></a><a href="is_same.md">is_same</a></p>
</td>
<td class="cellrowborder" valign="top" width="62.29%" headers="mcps1.2.3.1.2 "><p id="p1128711539108"><a name="p1128711539108"></a><a name="p1128711539108"></a>在程序编译时判断两个类型是否完全相同。</p>
</td>
</tr>
<tr id="row1633720117221"><td class="cellrowborder" valign="top" width="37.71%" headers="mcps1.2.3.1.1 "><p id="p93381111122210"><a name="p93381111122210"></a><a name="p93381111122210"></a><a href="is_void.md">is_void</a></p>
</td>
<td class="cellrowborder" valign="top" width="62.29%" headers="mcps1.2.3.1.2 "><p id="p1533831115228"><a name="p1533831115228"></a><a name="p1533831115228"></a>在程序编译时，检测一个类型是否为void类型。</p>
</td>
</tr>
<tr id="row52421326162217"><td class="cellrowborder" valign="top" width="37.71%" headers="mcps1.2.3.1.1 "><p id="p12242172662210"><a name="p12242172662210"></a><a name="p12242172662210"></a><a href="is_integral.md">is_integral</a></p>
</td>
<td class="cellrowborder" valign="top" width="62.29%" headers="mcps1.2.3.1.2 "><p id="p1624252612219"><a name="p1624252612219"></a><a name="p1624252612219"></a>在程序编译时，检测一个类型是否为整数类型。</p>
</td>
</tr>
<tr id="row11734172662219"><td class="cellrowborder" valign="top" width="37.71%" headers="mcps1.2.3.1.1 "><p id="p1973415265225"><a name="p1973415265225"></a><a name="p1973415265225"></a><a href="is_floating_point.md">is_floating_point</a></p>
</td>
<td class="cellrowborder" valign="top" width="62.29%" headers="mcps1.2.3.1.2 "><p id="p57343265222"><a name="p57343265222"></a><a name="p57343265222"></a>在程序编译时，检测一个类型是否为浮点类型。</p>
</td>
</tr>
<tr id="row222019272222"><td class="cellrowborder" valign="top" width="37.71%" headers="mcps1.2.3.1.1 "><p id="p6220182752214"><a name="p6220182752214"></a><a name="p6220182752214"></a><a href="is_array.md">is_array</a></p>
</td>
<td class="cellrowborder" valign="top" width="62.29%" headers="mcps1.2.3.1.2 "><p id="p422182716228"><a name="p422182716228"></a><a name="p422182716228"></a>在程序编译时，检测一个类型是否为数组类型。</p>
</td>
</tr>
<tr id="row7633162752214"><td class="cellrowborder" valign="top" width="37.71%" headers="mcps1.2.3.1.1 "><p id="p163352792217"><a name="p163352792217"></a><a name="p163352792217"></a><a href="is_pointer.md">is_pointer</a></p>
</td>
<td class="cellrowborder" valign="top" width="62.29%" headers="mcps1.2.3.1.2 "><p id="p663302713222"><a name="p663302713222"></a><a name="p663302713222"></a>在程序编译时，判断一个类型是否为指针类型。</p>
</td>
</tr>
<tr id="row238102862215"><td class="cellrowborder" valign="top" width="37.71%" headers="mcps1.2.3.1.1 "><p id="p13381828182211"><a name="p13381828182211"></a><a name="p13381828182211"></a><a href="is_reference.md">is_reference</a></p>
</td>
<td class="cellrowborder" valign="top" width="62.29%" headers="mcps1.2.3.1.2 "><p id="p939162872220"><a name="p939162872220"></a><a name="p939162872220"></a>在程序编译时，检测一个类型是否为引用类型。</p>
</td>
</tr>
<tr id="row204651428192213"><td class="cellrowborder" valign="top" width="37.71%" headers="mcps1.2.3.1.1 "><p id="p8465152814228"><a name="p8465152814228"></a><a name="p8465152814228"></a><a href="is_const.md">is_const</a></p>
</td>
<td class="cellrowborder" valign="top" width="62.29%" headers="mcps1.2.3.1.2 "><p id="p1046562819221"><a name="p1046562819221"></a><a name="p1046562819221"></a>在程序编译时，检测一个类型是否为const限定的类型。</p>
</td>
</tr>
<tr id="row7909628112210"><td class="cellrowborder" valign="top" width="37.71%" headers="mcps1.2.3.1.1 "><p id="p1090919288227"><a name="p1090919288227"></a><a name="p1090919288227"></a><a href="remove_const.md">remove_const</a></p>
</td>
<td class="cellrowborder" valign="top" width="62.29%" headers="mcps1.2.3.1.2 "><p id="p129091628182219"><a name="p129091628182219"></a><a name="p129091628182219"></a>在程序编译时，对传入的模板参数类型移除const限定符。</p>
</td>
</tr>
<tr id="row20307202992210"><td class="cellrowborder" valign="top" width="37.71%" headers="mcps1.2.3.1.1 "><p id="p130722972215"><a name="p130722972215"></a><a name="p130722972215"></a><a href="remove_volatile.md">remove_volatile</a></p>
</td>
<td class="cellrowborder" valign="top" width="62.29%" headers="mcps1.2.3.1.2 "><p id="p830772915225"><a name="p830772915225"></a><a name="p830772915225"></a>在程序编译时，对传入的模板参数类型移除volatile限定符。</p>
</td>
</tr>
<tr id="row158421829152216"><td class="cellrowborder" valign="top" width="37.71%" headers="mcps1.2.3.1.1 "><p id="p484217299222"><a name="p484217299222"></a><a name="p484217299222"></a><a href="remove_cv.md">remove_cv</a></p>
</td>
<td class="cellrowborder" valign="top" width="62.29%" headers="mcps1.2.3.1.2 "><p id="p284232922210"><a name="p284232922210"></a><a name="p284232922210"></a>在程序编译时，对传入的模板参数类型移除const限定符或volatile限定符，或同时移除这两种限定符。</p>
</td>
</tr>
<tr id="row16326103092214"><td class="cellrowborder" valign="top" width="37.71%" headers="mcps1.2.3.1.1 "><p id="p332613052215"><a name="p332613052215"></a><a name="p332613052215"></a><a href="remove_reference.md">remove_reference</a></p>
</td>
<td class="cellrowborder" valign="top" width="62.29%" headers="mcps1.2.3.1.2 "><p id="p1532653082214"><a name="p1532653082214"></a><a name="p1532653082214"></a>在程序编译时，从给定类型中移除引用限定符。</p>
</td>
</tr>
<tr id="row19738173016229"><td class="cellrowborder" valign="top" width="37.71%" headers="mcps1.2.3.1.1 "><p id="p1373818309226"><a name="p1373818309226"></a><a name="p1373818309226"></a><a href="remove_pointer.md">remove_pointer</a></p>
</td>
<td class="cellrowborder" valign="top" width="62.29%" headers="mcps1.2.3.1.2 "><p id="p873883092210"><a name="p873883092210"></a><a name="p873883092210"></a>在程序编译时，从给定类型中移除指针限定符。</p>
</td>
</tr>
<tr id="row2169173192219"><td class="cellrowborder" valign="top" width="37.71%" headers="mcps1.2.3.1.1 "><p id="p1816912311226"><a name="p1816912311226"></a><a name="p1816912311226"></a><a href="add_const.md">add_const</a></p>
</td>
<td class="cellrowborder" valign="top" width="62.29%" headers="mcps1.2.3.1.2 "><p id="p01692316228"><a name="p01692316228"></a><a name="p01692316228"></a>在程序编译时，为指定类型添加const限定符。</p>
</td>
</tr>
<tr id="row6604193111220"><td class="cellrowborder" valign="top" width="37.71%" headers="mcps1.2.3.1.1 "><p id="p7604103110226"><a name="p7604103110226"></a><a name="p7604103110226"></a><a href="add_volatile.md">add_volatile</a></p>
</td>
<td class="cellrowborder" valign="top" width="62.29%" headers="mcps1.2.3.1.2 "><p id="p860419319222"><a name="p860419319222"></a><a name="p860419319222"></a>在程序编译时，为指定类型添加volatile限定符。</p>
</td>
</tr>
<tr id="row1151173214224"><td class="cellrowborder" valign="top" width="37.71%" headers="mcps1.2.3.1.1 "><p id="p12521432172212"><a name="p12521432172212"></a><a name="p12521432172212"></a><a href="add_cv.md">add_cv</a></p>
</td>
<td class="cellrowborder" valign="top" width="62.29%" headers="mcps1.2.3.1.2 "><p id="p65293272214"><a name="p65293272214"></a><a name="p65293272214"></a>在程序编译时，为指定类型添加const和volatile限定符。</p>
</td>
</tr>
<tr id="row8490932192211"><td class="cellrowborder" valign="top" width="37.71%" headers="mcps1.2.3.1.1 "><p id="p1849183212225"><a name="p1849183212225"></a><a name="p1849183212225"></a><a href="add_pointer.md">add_pointer</a></p>
</td>
<td class="cellrowborder" valign="top" width="62.29%" headers="mcps1.2.3.1.2 "><p id="p14491532112214"><a name="p14491532112214"></a><a name="p14491532112214"></a>在程序编译时，为指定类型添加指针限定符。</p>
</td>
</tr>
<tr id="row99214327228"><td class="cellrowborder" valign="top" width="37.71%" headers="mcps1.2.3.1.1 "><p id="p5921163211227"><a name="p5921163211227"></a><a name="p5921163211227"></a><a href="add_lvalue_reference.md">add_lvalue_reference</a></p>
</td>
<td class="cellrowborder" valign="top" width="62.29%" headers="mcps1.2.3.1.2 "><p id="p492143211222"><a name="p492143211222"></a><a name="p492143211222"></a>在程序编译时，为指定类型添加左值引用限定符。</p>
</td>
</tr>
<tr id="row4302033132213"><td class="cellrowborder" valign="top" width="37.71%" headers="mcps1.2.3.1.1 "><p id="p3302163352214"><a name="p3302163352214"></a><a name="p3302163352214"></a><a href="add_rvalue_reference.md">add_rvalue_reference</a></p>
</td>
<td class="cellrowborder" valign="top" width="62.29%" headers="mcps1.2.3.1.2 "><p id="p630218339222"><a name="p630218339222"></a><a name="p630218339222"></a>在程序编译时，为指定类型添加右值引用限定符。</p>
</td>
</tr>
<tr id="row498311541816"><td class="cellrowborder" valign="top" width="37.71%" headers="mcps1.2.3.1.1 "><p id="p4286105318107"><a name="p4286105318107"></a><a name="p4286105318107"></a><a href="enable_if.md">enable_if</a></p>
</td>
<td class="cellrowborder" valign="top" width="62.29%" headers="mcps1.2.3.1.2 "><p id="p1328665317104"><a name="p1328665317104"></a><a name="p1328665317104"></a>在程序编译时根据某个条件启用或禁用特定的函数模板、类模板或模板特化。</p>
</td>
</tr>
<tr id="row49836541682"><td class="cellrowborder" valign="top" width="37.71%" headers="mcps1.2.3.1.1 "><p id="p18285105321019"><a name="p18285105321019"></a><a name="p18285105321019"></a><a href="conditional.md">conditional</a></p>
</td>
<td class="cellrowborder" valign="top" width="62.29%" headers="mcps1.2.3.1.2 "><p id="p1928525319102"><a name="p1928525319102"></a><a name="p1928525319102"></a>在程序编译时根据一个布尔条件从两个类型中选择一个类型。</p>
</td>
</tr>
<tr id="row15509255143618"><td class="cellrowborder" valign="top" width="37.71%" headers="mcps1.2.3.1.1 "><p id="p968185918363"><a name="p968185918363"></a><a name="p968185918363"></a><a href="integral_constant.md">integral_constant</a></p>
</td>
<td class="cellrowborder" valign="top" width="62.29%" headers="mcps1.2.3.1.2 "><p id="p15098558367"><a name="p15098558367"></a><a name="p15098558367"></a>用于封装一个编译时常量整数值，是标准库中许多类型特性和编译时计算的基础组件。</p>
</td>
</tr>
</tbody>
</table>

**表 38**  平台信息获取API列表

<a name="table32991747162610"></a>
<table><thead align="left"><tr id="row13299184712616"><th class="cellrowborder" valign="top" width="37.6%" id="mcps1.2.3.1.1"><p id="p15299104762613"><a name="p15299104762613"></a><a name="p15299104762613"></a>接口名</p>
</th>
<th class="cellrowborder" valign="top" width="62.4%" id="mcps1.2.3.1.2"><p id="p123001847142611"><a name="p123001847142611"></a><a name="p123001847142611"></a>功能描述</p>
</th>
</tr>
</thead>
<tbody><tr id="row63002047192611"><td class="cellrowborder" valign="top" width="37.6%" headers="mcps1.2.3.1.1 "><p id="p530034762613"><a name="p530034762613"></a><a name="p530034762613"></a><a href="PlatformAscendC.md">PlatformAscendC</a></p>
</td>
<td class="cellrowborder" valign="top" width="62.4%" headers="mcps1.2.3.1.2 "><p id="p43001947152612"><a name="p43001947152612"></a><a name="p43001947152612"></a>在实现Host侧的Tiling函数时，可能需要获取一些硬件平台的信息，来支撑Tiling的计算，比如获取硬件平台的核数等信息。PlatformAscendC类提供获取这些平台信息的功能。</p>
</td>
</tr>
<tr id="row6300134719269"><td class="cellrowborder" valign="top" width="37.6%" headers="mcps1.2.3.1.1 "><p id="p14300184720265"><a name="p14300184720265"></a><a name="p14300184720265"></a><a href="PlatformAscendCManager.md">PlatformAscendCManager</a></p>
</td>
<td class="cellrowborder" valign="top" width="62.4%" headers="mcps1.2.3.1.2 "><p id="p1330019473269"><a name="p1330019473269"></a><a name="p1330019473269"></a>基于Kernel Launch算子工程，通过基础调用（Kernel Launch）方式调用算子的场景下，可能需要获取硬件平台相关信息，比如获取硬件平台的核数。PlatformAscendCManager类提供获取平台信息的功能。</p>
</td>
</tr>
</tbody>
</table>

**表 39**  原型注册与管理API列表

<a name="table139218681912"></a>
<table><thead align="left"><tr id="row59296111918"><th class="cellrowborder" valign="top" width="37.55%" id="mcps1.2.3.1.1"><p id="p89215614196"><a name="p89215614196"></a><a name="p89215614196"></a>接口名</p>
</th>
<th class="cellrowborder" valign="top" width="62.45%" id="mcps1.2.3.1.2"><p id="p19216101919"><a name="p19216101919"></a><a name="p19216101919"></a>功能描述</p>
</th>
</tr>
</thead>
<tbody><tr id="row19311619195"><td class="cellrowborder" valign="top" width="37.55%" headers="mcps1.2.3.1.1 "><p id="p9931615198"><a name="p9931615198"></a><a name="p9931615198"></a><a href="原型注册接口（OP_ADD）.md">原型注册接口（OP_ADD）</a></p>
</td>
<td class="cellrowborder" valign="top" width="62.45%" headers="mcps1.2.3.1.2 "><p id="p189320611195"><a name="p189320611195"></a><a name="p189320611195"></a>注册算子的原型定义。</p>
</td>
</tr>
<tr id="row169320619198"><td class="cellrowborder" valign="top" width="37.55%" headers="mcps1.2.3.1.1 "><p id="p119306161913"><a name="p119306161913"></a><a name="p119306161913"></a><a href="OpDef.md">OpDef</a></p>
</td>
<td class="cellrowborder" valign="top" width="62.45%" headers="mcps1.2.3.1.2 "><p id="p14936612191"><a name="p14936612191"></a><a name="p14936612191"></a>用于算子原型定义。</p>
</td>
</tr>
<tr id="row16935614198"><td class="cellrowborder" valign="top" width="37.55%" headers="mcps1.2.3.1.1 "><p id="p393460198"><a name="p393460198"></a><a name="p393460198"></a><a href="OpParamDef.md">OpParamDef</a></p>
</td>
<td class="cellrowborder" valign="top" width="62.45%" headers="mcps1.2.3.1.2 "><p id="p18938611917"><a name="p18938611917"></a><a name="p18938611917"></a>用于算子参数定义。</p>
</td>
</tr>
<tr id="row19386121911"><td class="cellrowborder" valign="top" width="37.55%" headers="mcps1.2.3.1.1 "><p id="p79315661915"><a name="p79315661915"></a><a name="p79315661915"></a><a href="OpAttrDef-167.md">OpAttrDef</a></p>
</td>
<td class="cellrowborder" valign="top" width="62.45%" headers="mcps1.2.3.1.2 "><p id="p129315619195"><a name="p129315619195"></a><a name="p129315619195"></a>用于算子属性定义。</p>
</td>
</tr>
<tr id="row169415613197"><td class="cellrowborder" valign="top" width="37.55%" headers="mcps1.2.3.1.1 "><p id="p1294106171920"><a name="p1294106171920"></a><a name="p1294106171920"></a><a href="OpAICoreDef.md">OpAICoreDef</a></p>
</td>
<td class="cellrowborder" valign="top" width="62.45%" headers="mcps1.2.3.1.2 "><p id="p1594116181913"><a name="p1594116181913"></a><a name="p1594116181913"></a>用于定义AI处理器上相关实现信息，并关联Tiling实现、Shape推导等函数。</p>
</td>
</tr>
<tr id="row2949661912"><td class="cellrowborder" valign="top" width="37.55%" headers="mcps1.2.3.1.1 "><p id="p1194166101912"><a name="p1194166101912"></a><a name="p1194166101912"></a><a href="OpAICoreConfig.md">OpAICoreConfig</a></p>
</td>
<td class="cellrowborder" valign="top" width="62.45%" headers="mcps1.2.3.1.2 "><p id="p19947620190"><a name="p19947620190"></a><a name="p19947620190"></a>用于配置AI Core配置信息。</p>
</td>
</tr>
<tr id="row10946641913"><td class="cellrowborder" valign="top" width="37.55%" headers="mcps1.2.3.1.1 "><p id="p1994186111919"><a name="p1994186111919"></a><a name="p1994186111919"></a><a href="OpMC2Def.md">OpMC2Def</a></p>
</td>
<td class="cellrowborder" valign="top" width="62.45%" headers="mcps1.2.3.1.2 "><p id="p69420601919"><a name="p69420601919"></a><a name="p69420601919"></a>该类用于在host侧配置通算融合算子的通信域名称。配置后在kernel侧可以获取通信域对应的context地址。</p>
</td>
</tr>
</tbody>
</table>

**表 40**  Tiling数据结构注册API列表

<a name="table1518794581918"></a>
<table><thead align="left"><tr id="row21887458194"><th class="cellrowborder" valign="top" width="37.55%" id="mcps1.2.3.1.1"><p id="p101881545171917"><a name="p101881545171917"></a><a name="p101881545171917"></a>接口名</p>
</th>
<th class="cellrowborder" valign="top" width="62.45%" id="mcps1.2.3.1.2"><p id="p1218824531915"><a name="p1218824531915"></a><a name="p1218824531915"></a>功能描述</p>
</th>
</tr>
</thead>
<tbody><tr id="row1319034531918"><td class="cellrowborder" valign="top" width="37.55%" headers="mcps1.2.3.1.1 "><p id="p10190245111913"><a name="p10190245111913"></a><a name="p10190245111913"></a><a href="TilingData结构定义.md">TilingData结构定义</a></p>
</td>
<td class="cellrowborder" valign="top" width="62.45%" headers="mcps1.2.3.1.2 "><p id="p619064571919"><a name="p619064571919"></a><a name="p619064571919"></a>定义一个TilingData的类，添加所需的成员变量（TilingData字段），用于保存所需TilingData参数。完成该TilingData类的定义后，该类通过继承TilingDef类（用来存放、处理用户自定义Tiling结构体成员变量的基类）提供TilingData字段设置、序列化和保存等接口。</p>
</td>
</tr>
<tr id="row619054501914"><td class="cellrowborder" valign="top" width="37.55%" headers="mcps1.2.3.1.1 "><p id="p1219044520192"><a name="p1219044520192"></a><a name="p1219044520192"></a><a href="TilingData结构注册.md">TilingData结构注册</a></p>
</td>
<td class="cellrowborder" valign="top" width="62.45%" headers="mcps1.2.3.1.2 "><p id="p1019034511196"><a name="p1019034511196"></a><a name="p1019034511196"></a>注册定义的TilingData结构体并和自定义算子绑定。</p>
</td>
</tr>
</tbody>
</table>

**表 41**  Tiling调测API列表

<a name="table2675125415261"></a>
<table><thead align="left"><tr id="row1967612545262"><th class="cellrowborder" valign="top" width="37.419999999999995%" id="mcps1.2.3.1.1"><p id="p367695412260"><a name="p367695412260"></a><a name="p367695412260"></a>接口名</p>
</th>
<th class="cellrowborder" valign="top" width="62.580000000000005%" id="mcps1.2.3.1.2"><p id="p0676145422615"><a name="p0676145422615"></a><a name="p0676145422615"></a>功能描述</p>
</th>
</tr>
</thead>
<tbody><tr id="row1010110512216"><td class="cellrowborder" valign="top" width="37.419999999999995%" headers="mcps1.2.3.1.1 "><p id="p2010212532219"><a name="p2010212532219"></a><a name="p2010212532219"></a><a href="OpTilingRegistry.md">OpTilingRegistry</a></p>
</td>
<td class="cellrowborder" valign="top" width="62.580000000000005%" headers="mcps1.2.3.1.2 "><p id="p710211522210"><a name="p710211522210"></a><a name="p710211522210"></a><span id="ph134802046173619"><a name="ph134802046173619"></a><a name="ph134802046173619"></a>OpTilingRegistry类属于context_ascendc命名空间，主要用于加载Tiling实现的动态库，并获取算子的Tiling函数指针以进行调试和验证。</span></p>
</td>
</tr>
<tr id="row1867785432614"><td class="cellrowborder" valign="top" width="37.419999999999995%" headers="mcps1.2.3.1.1 "><p id="p967795492618"><a name="p967795492618"></a><a name="p967795492618"></a><a href="ContextBuilder.md">ContextBuilder</a></p>
</td>
<td class="cellrowborder" valign="top" width="62.580000000000005%" headers="mcps1.2.3.1.2 "><p id="p56781254122619"><a name="p56781254122619"></a><a name="p56781254122619"></a>ContextBuilder类提供一系列的API接口，支持手动构造TilingContext类来验证Tiling函数以及KernelContext类用于TilingParse函数的验证。</p>
</td>
</tr>
</tbody>
</table>

**表 42**  Tiling模板编程API列表

<a name="table2864441102011"></a>
<table><thead align="left"><tr id="row4832193752110"><th class="cellrowborder" valign="top" width="37.44%" id="mcps1.2.3.1.1"><p id="p157001846132117"><a name="p157001846132117"></a><a name="p157001846132117"></a>接口名</p>
</th>
<th class="cellrowborder" valign="top" width="62.56%" id="mcps1.2.3.1.2"><p id="p1570016468218"><a name="p1570016468218"></a><a name="p1570016468218"></a>功能描述</p>
</th>
</tr>
</thead>
<tbody><tr id="row1486515413204"><td class="cellrowborder" valign="top" width="37.44%" headers="mcps1.2.3.1.1 "><p id="p111901452192"><a name="p111901452192"></a><a name="p111901452192"></a><a href="模板参数定义.md">模板参数定义</a></p>
</td>
<td class="cellrowborder" valign="top" width="62.56%" headers="mcps1.2.3.1.2 "><p id="p5191124551911"><a name="p5191124551911"></a><a name="p5191124551911"></a>通过该类接口进行模板参数ASCENDC_TPL_ARGS_DECL和模板参数组合ASCENDC_TPL_ARGS_SEL（即可使用的模板）的定义。</p>
</td>
</tr>
<tr id="row198667417207"><td class="cellrowborder" valign="top" width="37.44%" headers="mcps1.2.3.1.1 "><p id="p17191145131910"><a name="p17191145131910"></a><a name="p17191145131910"></a><a href="GET_TPL_TILING_KEY.md">GET_TPL_TILING_KEY</a></p>
</td>
<td class="cellrowborder" valign="top" width="62.56%" headers="mcps1.2.3.1.2 "><p id="p1619174520193"><a name="p1619174520193"></a><a name="p1619174520193"></a>Tiling模板编程时，开发者通过调用此接口自动生成TilingKey。该接口将传入的模板参数通过定义的位宽，转成二进制，按照顺序组合后转成uint64数值，即TilingKey。</p>
</td>
</tr>
<tr id="row586654192019"><td class="cellrowborder" valign="top" width="37.44%" headers="mcps1.2.3.1.1 "><p id="p419110458198"><a name="p419110458198"></a><a name="p419110458198"></a><a href="ASCENDC_TPL_SEL_PARAM.md">ASCENDC_TPL_SEL_PARAM</a></p>
</td>
<td class="cellrowborder" valign="top" width="62.56%" headers="mcps1.2.3.1.2 "><p id="p819119458193"><a name="p819119458193"></a><a name="p819119458193"></a><span id="ph1519110455190"><a name="ph1519110455190"></a><a name="ph1519110455190"></a>Tiling模板编程时，开发者通过调用此接口自动生成并配置TilingKey。</span></p>
</td>
</tr>
</tbody>
</table>

**表 43**  Tiling下沉API列表

<a name="table2173036112514"></a>
<table><thead align="left"><tr id="row161714011250"><th class="cellrowborder" valign="top" width="37.3%" id="mcps1.2.3.1.1"><p id="p96601406261"><a name="p96601406261"></a><a name="p96601406261"></a>接口名</p>
</th>
<th class="cellrowborder" valign="top" width="62.7%" id="mcps1.2.3.1.2"><p id="p26609019267"><a name="p26609019267"></a><a name="p26609019267"></a>功能描述</p>
</th>
</tr>
</thead>
<tbody><tr id="row917323610256"><td class="cellrowborder" valign="top" width="37.3%" headers="mcps1.2.3.1.1 "><p id="p1419164561913"><a name="p1419164561913"></a><a name="p1419164561913"></a><a href="DEVICE_IMPL_OP_OPTILING.md">DEVICE_IMPL_OP_OPTILING</a></p>
</td>
<td class="cellrowborder" valign="top" width="62.7%" headers="mcps1.2.3.1.2 "><p id="p9191945151920"><a name="p9191945151920"></a><a name="p9191945151920"></a><span id="ph419114452199"><a name="ph419114452199"></a><a name="ph419114452199"></a>在<a href="使能Tiling下沉.md">Tiling下沉</a>场景中，该宏定义用于生成Tiling下沉的注册类，再通过调用注册类的成员函数来注册需要下沉的Tiling函数。</span></p>
</td>
</tr>
</tbody>
</table>

**表 44**  log API列表

<a name="table1514223372716"></a>
<table><thead align="left"><tr id="row3142033112715"><th class="cellrowborder" valign="top" width="37.79%" id="mcps1.2.3.1.1"><p id="p181421633182714"><a name="p181421633182714"></a><a name="p181421633182714"></a>接口名</p>
</th>
<th class="cellrowborder" valign="top" width="62.21%" id="mcps1.2.3.1.2"><p id="p1114213332712"><a name="p1114213332712"></a><a name="p1114213332712"></a>功能描述</p>
</th>
</tr>
</thead>
<tbody><tr id="row914623311278"><td class="cellrowborder" valign="top" width="37.79%" headers="mcps1.2.3.1.1 "><p id="p014643382720"><a name="p014643382720"></a><a name="p014643382720"></a><a href="ASC_CPU_LOG.md">ASC_CPU_LOG</a></p>
</td>
<td class="cellrowborder" valign="top" width="62.21%" headers="mcps1.2.3.1.2 "><p id="p101461833162715"><a name="p101461833162715"></a><a name="p101461833162715"></a><span id="ph1014643313273"><a name="ph1014643313273"></a><a name="ph1014643313273"></a>提供Host侧打印Log的功能。开发者可以在算子的TilingFunc代码中使用ASC_CPU_LOG_XXX接口来输出相关内容。</span></p>
</td>
</tr>
</tbody>
</table>

**表 45**  调测接口列表

<a name="table17826192512135"></a>
<table><thead align="left"><tr id="row10826162551319"><th class="cellrowborder" valign="top" width="37.79%" id="mcps1.2.3.1.1"><p id="p782622512134"><a name="p782622512134"></a><a name="p782622512134"></a>接口名</p>
</th>
<th class="cellrowborder" valign="top" width="62.21%" id="mcps1.2.3.1.2"><p id="p982712512136"><a name="p982712512136"></a><a name="p982712512136"></a>功能描述</p>
</th>
</tr>
</thead>
<tbody><tr id="row48271425151314"><td class="cellrowborder" valign="top" width="37.79%" headers="mcps1.2.3.1.1 "><p id="p128276255138"><a name="p128276255138"></a><a name="p128276255138"></a><a href="printf-176.md">printf</a></p>
</td>
<td class="cellrowborder" valign="top" width="62.21%" headers="mcps1.2.3.1.2 "><p id="p182718253134"><a name="p182718253134"></a><a name="p182718253134"></a><span id="ph1565753361414"><a name="ph1565753361414"></a><a name="ph1565753361414"></a>本接口提供SIMT VF调试场景下的格式化输出功能。在算子Kernel侧的SIMT VF实现代码中，需要输出日志信息时，调用printf接口打印相关内容。</span></p>
</td>
</tr>
<tr id="row1597951912417"><td class="cellrowborder" valign="top" width="37.79%" headers="mcps1.2.3.1.1 "><p id="p397971912416"><a name="p397971912416"></a><a name="p397971912416"></a><a href="assert-177.md">assert</a></p>
</td>
<td class="cellrowborder" valign="top" width="62.21%" headers="mcps1.2.3.1.2 "><p id="p697901913247"><a name="p697901913247"></a><a name="p697901913247"></a>本接口在SIMT VF调试场景下提供assert断言功能。在算子Kernel侧的SIMT VF实现代码中，如果assert的内部条件判断不为真，则会输出assert条件，并将输入的信息格式化打印在屏幕上。</p>
</td>
</tr>
<tr id="row497811224246"><td class="cellrowborder" valign="top" width="37.79%" headers="mcps1.2.3.1.1 "><p id="p19781122132416"><a name="p19781122132416"></a><a name="p19781122132416"></a><a href="__trap.md">__trap</a></p>
</td>
<td class="cellrowborder" valign="top" width="62.21%" headers="mcps1.2.3.1.2 "><p id="p1297852218244"><a name="p1297852218244"></a><a name="p1297852218244"></a>在SIMT VF实现代码中调用此接口会中断算子的运行。</p>
</td>
</tr>
<tr id="row1765213536149"><td class="cellrowborder" valign="top" width="37.79%" headers="mcps1.2.3.1.1 "><p id="p19652105310140"><a name="p19652105310140"></a><a name="p19652105310140"></a><a href="zh-cn_topic_0000002554344543.md">clock</a></p>
</td>
<td class="cellrowborder" valign="top" width="62.21%" headers="mcps1.2.3.1.2 "><p id="p277505112411"><a name="p277505112411"></a><a name="p277505112411"></a>本接口在SIMT VF调试场景下提供 Clock 时间戳功能，用于记录从程序启动至接口调用时刻所经历的时钟周期数（Cycle Count），便于精准分析执行延迟与性能瓶颈。</p>
</td>
</tr>
</tbody>
</table>

