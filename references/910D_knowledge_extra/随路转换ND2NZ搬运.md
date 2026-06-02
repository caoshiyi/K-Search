# 随路转换ND2NZ搬运<a name="ZH-CN_TOPIC_0000002554344715"></a>

## 产品支持情况<a name="section1550532418810"></a>

<a name="table38301303189"></a>
<table><thead align="left"><tr id="row20831180131817"><th class="cellrowborder" valign="top" width="53.64%" id="mcps1.1.4.1.1"><p id="p1883113061818"><a name="p1883113061818"></a><a name="p1883113061818"></a><span id="ph20833205312295"><a name="ph20833205312295"></a><a name="ph20833205312295"></a>产品</span></p>
</th>
<th class="cellrowborder" align="center" valign="top" width="25.28%" id="mcps1.1.4.1.2"><p id="p783113012187"><a name="p783113012187"></a><a name="p783113012187"></a>是否支持</p>
<p id="p1948061154115"><a name="p1948061154115"></a><a name="p1948061154115"></a>Global Memory -&gt; Local Memory</p>
</th>
<th class="cellrowborder" align="center" valign="top" width="21.08%" id="mcps1.1.4.1.3"><p id="p973217299419"><a name="p973217299419"></a><a name="p973217299419"></a>是否支持</p>
<p id="p67288449406"><a name="p67288449406"></a><a name="p67288449406"></a>Local Memory -&gt; Local Memory</p>
</th>
</tr>
</thead>
<tbody><tr id="row1272474920205"><td class="cellrowborder" valign="top" width="53.64%" headers="mcps1.1.4.1.1 "><p id="p17301775812"><a name="p17301775812"></a><a name="p17301775812"></a><span id="ph2272194216543"><a name="ph2272194216543"></a><a name="ph2272194216543"></a>Ascend 950PR/Ascend 950DT</span></p>
</td>
<td class="cellrowborder" align="center" valign="top" width="25.28%" headers="mcps1.1.4.1.2 "><p id="p37256491200"><a name="p37256491200"></a><a name="p37256491200"></a>√</p>
</td>
<td class="cellrowborder" align="center" valign="top" width="21.08%" headers="mcps1.1.4.1.3 "><p id="p10204845184116"><a name="p10204845184116"></a><a name="p10204845184116"></a>√</p>
</td>
</tr>
</tbody>
</table>

## 功能说明<a name="section12840195813362"></a>

支持在数据搬运时进行ND到NZ格式的转换。

## 函数原型<a name="section1954364615315"></a>

-   Global Memory -\> Local Memory

    ```
    template <typename T>
    __aicore__ inline void DataCopy(const LocalTensor<T>& dst, const GlobalTensor<T>& src, const Nd2NzParams& intriParams)
    ```

-   Local Memory -\> Local Memory

    ```
    template <typename T>   
    __aicore__ inline void DataCopy(const LocalTensor<T>& dst, const LocalTensor<T>& src, const Nd2NzParams& intriParams)
    ```

> **说明：** 
>各原型支持的具体数据通路和数据类型，请参考[支持的通路和数据类型](#section87413163309)。

## 参数说明<a name="section1251613311396"></a>

**表 1**  模板参数说明

<a name="table49614585413"></a>
<table><thead align="left"><tr id="row996115820412"><th class="cellrowborder" valign="top" width="14.719999999999999%" id="mcps1.2.3.1.1"><p id="p10961458104110"><a name="p10961458104110"></a><a name="p10961458104110"></a>参数名</p>
</th>
<th class="cellrowborder" valign="top" width="85.28%" id="mcps1.2.3.1.2"><p id="p6961155817415"><a name="p6961155817415"></a><a name="p6961155817415"></a>描述</p>
</th>
</tr>
</thead>
<tbody><tr id="row4961205811416"><td class="cellrowborder" valign="top" width="14.719999999999999%" headers="mcps1.2.3.1.1 "><p id="p29619586417"><a name="p29619586417"></a><a name="p29619586417"></a>T</p>
</td>
<td class="cellrowborder" valign="top" width="85.28%" headers="mcps1.2.3.1.2 "><p id="p996110583417"><a name="p996110583417"></a><a name="p996110583417"></a>源操作数或者目的操作数的数据类型。支持的数据类型请参考<a href="#section87413163309">支持的通路和数据类型</a>。</p>
</td>
</tr>
</tbody>
</table>

**表 2**  参数说明

<a name="table917mcpsimp"></a>
<table><thead align="left"><tr id="row923mcpsimp"><th class="cellrowborder" valign="top" width="15.02%" id="mcps1.2.4.1.1"><p id="p925mcpsimp"><a name="p925mcpsimp"></a><a name="p925mcpsimp"></a>参数名称</p>
</th>
<th class="cellrowborder" valign="top" width="9.86%" id="mcps1.2.4.1.2"><p id="p927mcpsimp"><a name="p927mcpsimp"></a><a name="p927mcpsimp"></a>输入/输出</p>
</th>
<th class="cellrowborder" valign="top" width="75.12%" id="mcps1.2.4.1.3"><p id="p929mcpsimp"><a name="p929mcpsimp"></a><a name="p929mcpsimp"></a>含义</p>
</th>
</tr>
</thead>
<tbody><tr id="row930mcpsimp"><td class="cellrowborder" valign="top" width="15.02%" headers="mcps1.2.4.1.1 "><p id="p2925016172518"><a name="p2925016172518"></a><a name="p2925016172518"></a>dst</p>
</td>
<td class="cellrowborder" valign="top" width="9.86%" headers="mcps1.2.4.1.2 "><p id="p199251416112517"><a name="p199251416112517"></a><a name="p199251416112517"></a>输出</p>
</td>
<td class="cellrowborder" valign="top" width="75.12%" headers="mcps1.2.4.1.3 "><p id="p37551021906"><a name="p37551021906"></a><a name="p37551021906"></a>目的操作数，类型为LocalTensor。</p>
</td>
</tr>
<tr id="row937mcpsimp"><td class="cellrowborder" valign="top" width="15.02%" headers="mcps1.2.4.1.1 "><p id="p3926171610253"><a name="p3926171610253"></a><a name="p3926171610253"></a>src</p>
</td>
<td class="cellrowborder" valign="top" width="9.86%" headers="mcps1.2.4.1.2 "><p id="p4926121682518"><a name="p4926121682518"></a><a name="p4926121682518"></a>输入</p>
</td>
<td class="cellrowborder" valign="top" width="75.12%" headers="mcps1.2.4.1.3 "><p id="p2055414382913"><a name="p2055414382913"></a><a name="p2055414382913"></a>源操作数，类型为LocalTensor或GlobalTensor。</p>
</td>
</tr>
<tr id="row997554013220"><td class="cellrowborder" valign="top" width="15.02%" headers="mcps1.2.4.1.1 "><p id="p13976540132215"><a name="p13976540132215"></a><a name="p13976540132215"></a>intriParams</p>
</td>
<td class="cellrowborder" valign="top" width="9.86%" headers="mcps1.2.4.1.2 "><p id="p139761340102213"><a name="p139761340102213"></a><a name="p139761340102213"></a>输入</p>
</td>
<td class="cellrowborder" valign="top" width="75.12%" headers="mcps1.2.4.1.3 "><p id="p99761407226"><a name="p99761407226"></a><a name="p99761407226"></a>搬运参数，类型为<a href="#table844881954715">Nd2NzParams</a>。</p>
<p id="p58852119618"><a name="p58852119618"></a><a name="p58852119618"></a>具体定义请参考<span id="ph10562197165916"><a name="ph10562197165916"></a><a name="ph10562197165916"></a>${INSTALL_DIR}</span>/include/ascendc/basic_api/interface/kernel_struct_data_copy.h，<span id="ph14322531015"><a name="ph14322531015"></a><a name="ph14322531015"></a>${INSTALL_DIR}</span>请替换为CANN软件安装后文件存储路径。</p>
</td>
</tr>
</tbody>
</table>

**表 3**  Nd2NzParams结构体参数定义

<a name="table844881954715"></a>
<table><thead align="left"><tr id="row11449201964714"><th class="cellrowborder" valign="top" width="18.44%" id="mcps1.2.3.1.1"><p id="p18449151934710"><a name="p18449151934710"></a><a name="p18449151934710"></a>参数名称</p>
</th>
<th class="cellrowborder" valign="top" width="81.56%" id="mcps1.2.3.1.2"><p id="p11449719134719"><a name="p11449719134719"></a><a name="p11449719134719"></a>含义</p>
</th>
</tr>
</thead>
<tbody><tr id="row5449161919474"><td class="cellrowborder" valign="top" width="18.44%" headers="mcps1.2.3.1.1 "><p id="p8928559104716"><a name="p8928559104716"></a><a name="p8928559104716"></a>ndNum</p>
</td>
<td class="cellrowborder" valign="top" width="81.56%" headers="mcps1.2.3.1.2 "><p id="p844917194478"><a name="p844917194478"></a><a name="p844917194478"></a>传输ND矩阵的数目，取值范围：ndNum∈[0, 4095]。</p>
</td>
</tr>
<tr id="row1944911994712"><td class="cellrowborder" valign="top" width="18.44%" headers="mcps1.2.3.1.1 "><p id="p102019313485"><a name="p102019313485"></a><a name="p102019313485"></a>nValue</p>
</td>
<td class="cellrowborder" valign="top" width="81.56%" headers="mcps1.2.3.1.2 "><p id="p11299126114913"><a name="p11299126114913"></a><a name="p11299126114913"></a>ND矩阵的行数，取值范围：nValue∈[0, 16384]。</p>
</td>
</tr>
<tr id="row1944916196475"><td class="cellrowborder" valign="top" width="18.44%" headers="mcps1.2.3.1.1 "><p id="p1397111514816"><a name="p1397111514816"></a><a name="p1397111514816"></a>dValue</p>
</td>
<td class="cellrowborder" valign="top" width="81.56%" headers="mcps1.2.3.1.2 "><p id="p144493191478"><a name="p144493191478"></a><a name="p144493191478"></a>ND矩阵的列数，取值范围：dValue∈[0, 65535]。</p>
</td>
</tr>
<tr id="row2449119174717"><td class="cellrowborder" valign="top" width="18.44%" headers="mcps1.2.3.1.1 "><p id="p1217581074813"><a name="p1217581074813"></a><a name="p1217581074813"></a>srcNdMatrixStride</p>
</td>
<td class="cellrowborder" valign="top" width="81.56%" headers="mcps1.2.3.1.2 "><p id="p1445013195477"><a name="p1445013195477"></a><a name="p1445013195477"></a>源操作数相邻ND矩阵起始地址间的偏移，取值范围：srcNdMatrixStride∈[0, 65535]，单位为元素。</p>
</td>
</tr>
<tr id="row0852151364819"><td class="cellrowborder" valign="top" width="18.44%" headers="mcps1.2.3.1.1 "><p id="p180672644819"><a name="p180672644819"></a><a name="p180672644819"></a>srcDValue</p>
</td>
<td class="cellrowborder" valign="top" width="81.56%" headers="mcps1.2.3.1.2 "><p id="p128531313134818"><a name="p128531313134818"></a><a name="p128531313134818"></a>源操作数同一ND矩阵的相邻行起始地址间的偏移，取值范围：srcDValue∈[1, 65535]，单位为元素。</p>
</td>
</tr>
<tr id="row3828171624817"><td class="cellrowborder" valign="top" width="18.44%" headers="mcps1.2.3.1.1 "><p id="p447510292484"><a name="p447510292484"></a><a name="p447510292484"></a>dstNzC0Stride</p>
</td>
<td class="cellrowborder" valign="top" width="81.56%" headers="mcps1.2.3.1.2 "><p id="p1482891617488"><a name="p1482891617488"></a><a name="p1482891617488"></a>ND转换到NZ格式后，源操作数中的一行会转换为目的操作数的多行。dstNzC0Stride表示，目的NZ矩阵中，来自源操作数同一行的多行数据相邻行起始地址间的偏移，取值范围：dstNzC0Stride∈[1, 16384]，单位：C0_SIZE（32B）。</p>
</td>
</tr>
<tr id="row817161912486"><td class="cellrowborder" valign="top" width="18.44%" headers="mcps1.2.3.1.1 "><p id="p981411313489"><a name="p981411313489"></a><a name="p981411313489"></a>dstNzNStride</p>
</td>
<td class="cellrowborder" valign="top" width="81.56%" headers="mcps1.2.3.1.2 "><p id="p14171111916488"><a name="p14171111916488"></a><a name="p14171111916488"></a>目的NZ矩阵中，Z型矩阵相邻行起始地址之间的偏移。取值范围：dstNzNStride∈[1, 16384]，单位：C0_SIZE（32B）。</p>
</td>
</tr>
<tr id="row0900162313483"><td class="cellrowborder" valign="top" width="18.44%" headers="mcps1.2.3.1.1 "><p id="p1370193484816"><a name="p1370193484816"></a><a name="p1370193484816"></a>dstNzMatrixStride</p>
</td>
<td class="cellrowborder" valign="top" width="81.56%" headers="mcps1.2.3.1.2 "><p id="p18901192374811"><a name="p18901192374811"></a><a name="p18901192374811"></a>目的NZ矩阵中，相邻NZ矩阵起始地址间的偏移，取值范围：dstNzMatrixStride∈[1, 65535]，单位为元素。</p>
</td>
</tr>
</tbody>
</table>

ND2NZ转换示意图如下，样例中参数设置值和解释说明如下：

-   ndNum = 2，表示传输ND矩阵的数目为2 \(ND矩阵1为A1\~A2 + B1\~B2，ND矩阵2为C1\~C2 + D1\~D2\)。
-   nValue = 2，ND矩阵的行数，也就是矩阵的高度为2。
-   dValue = 24，ND矩阵的列数，也就是矩阵的宽度为24个元素。当dValue不满足32B对齐时，在目的操作数中不足的部分会被补齐为0，例如图示中A2所在DataBlock的空白部分会被补齐为0。
-   srcNdMatrixStride = 144，表达相邻ND矩阵起始地址间的偏移，即为A1\~C1的距离，即为9个DataBlock，9 \* 16 = 144个元素。
-   srcDValue = 48，表示一行的所含元素个数，即为A1到B1的距离，即为3个DataBlock，3 \* 16 = 48个元素
-   dstNzC0Stride = 11。ND转换到NZ格式后，源操作数中的一行会转换为目的操作数的多行，例如src中A1和A2为1行，dst中A1和A2被分为2行。多行数据起始地址之间的偏移就是A1和A2在dst中的偏移，偏移为11个DataBlock。
-   dstNzNStride = 2，表示src中一个ND矩阵的第x行和第x+1行转换为NZ格式后在dst中的偏移，即A1和B1在dst之间的偏移为2个DataBlock。
-   dstNzMatrixStride = 96，表达dst中第x个ND矩阵的起点和第x+1个ND矩阵的起点的偏移，即A1和C1之间的距离，即为6个DataBlock，6 \* 16 = 96个元素。

**图 1**  ND2NZ转换示意图（half数据类型）<a name="fig128961542184620"></a>  
<!-- img2text -->
```
src
                 第一个ND矩阵
                     ↓
      dValue = 24
   <──────────────────>
      datalock
          ↓
┌────┬────┬────┬────┐
│ A1 │ A2 │    │    │
├────┼────┼────┼────┤
│ B1 │ B2 │    │    │
├────┼────┼────┼────┤
│    │    │    │    │
├────┼────┼────┼────┤
│ C1 │ C2 │    │    │
├────┼────┼────┼────┤
│ D1 │ D2 │    │    │
└────┴────┴────┴────┘
↑                    ↑
│                    │
srcNdMatrixStride    第x个ND矩阵
= 144

↑
│
srcNdRowStride
= 48

ND2NZ
  ─────→

dst
                             ┌────┬────┐
                             │ A1 │ A2 │→
                             ├────┼────┤ │
                             │ B1 │ B2 │ │
                             ├────┼────┤ │
                             │    │    │ │
                             ├────┼────┤ │
                             │    │    │ │
                             ├────┼────┤ │
                             │ C1 │ C2 │ │
                             ├────┼────┤ │
                             │    │    │ │
                             ├────┼────┤ │
                             │ D1 │ D2 │ │
                             └────┴────┘ │
                               ↓    ↓    │
                               ↓    ↓    │
                               └────┴────┘

↑
│
dstNzCOStride = 11

↑
│
dstNzMatrixStride = 96

                         ↑
                         │
                         dstNzNStride = 2
```

说明:
- 左图为源数据 `src` 的 ND 布局，标注了：
  - `dValue = 24`
  - `srcNdRowStride = 48`
  - `srcNdMatrixStride = 144`
  - `第一个ND矩阵`
  - `第x个ND矩阵`
  - `datablock`
- 中间为转换关系：`ND2NZ`
- 右图为目标数据 `dst` 的 NZ 布局，标注了：
  - `dstNzCOStride = 11`
  - `dstNzMatrixStride = 96`
  - `dstNzNStride = 2`
- 图中元素包含：`A1 A2 B1 B2 C1 C2 D1 D2`
- 右图中的箭头表示 ND 到 NZ 转换后，同一行元素被重排到不同位置；其中：
  - `A1` 和 `A2` 在 dst 中被拆到多行
  - `A1` 与 `B1` 之间体现 `dstNzNStride = 2`
  - `A1` 与 `C1` 之间体现 `dstNzMatrixStride = 96`
  - 同一分块内部多行起始偏移体现 `dstNzCOStride = 11`

## 返回值说明<a name="section446456163012"></a>

无

## 支持的通路和数据类型<a name="section87413163309"></a>

下文的数据通路均通过逻辑位置[TPosition](TPosition.md#table5376122715308)来表达，并注明了对应的物理通路。TPosition与物理内存的映射关系见[表1](通用说明和约束.md#table07372185712)。

**表 4**  Global Memory -\> Local Memory具体通路和支持的数据类型

<a name="table14255161718545"></a>
<table><thead align="left"><tr id="row3255181710543"><th class="cellrowborder" valign="top" width="11.04%" id="mcps1.2.4.1.1"><p id="p52550177546"><a name="p52550177546"></a><a name="p52550177546"></a>产品型号</p>
</th>
<th class="cellrowborder" valign="top" width="24.05%" id="mcps1.2.4.1.2"><p id="p13255191735420"><a name="p13255191735420"></a><a name="p13255191735420"></a>数据通路</p>
</th>
<th class="cellrowborder" valign="top" width="64.91%" id="mcps1.2.4.1.3"><p id="p6255617175419"><a name="p6255617175419"></a><a name="p6255617175419"></a>源操作数和目的操作数的数据类型 (两者保持一致)</p>
</th>
</tr>
</thead>
<tbody><tr id="row156095446519"><td class="cellrowborder" rowspan="2" valign="top" width="11.04%" headers="mcps1.2.4.1.1 "><p id="p11535204114617"><a name="p11535204114617"></a><a name="p11535204114617"></a><span id="ph653514111619"><a name="ph653514111619"></a><a name="ph653514111619"></a>Ascend 950PR/Ascend 950DT</span></p>
</td>
<td class="cellrowborder" valign="top" width="24.05%" headers="mcps1.2.4.1.2 "><a name="ul884513177617"></a><a name="ul884513177617"></a><ul id="ul884513177617"><li>GM -&gt; VECIN（GM -&gt; UB）</li></ul>
</td>
<td class="cellrowborder" valign="top" width="64.91%" headers="mcps1.2.4.1.3 "><p id="p0272814618"><a name="p0272814618"></a><a name="p0272814618"></a>bool、int8_t、uint8_t、hifloat8_t、fp8_e5m2_t、fp8_e4m3fn_t、fp8_e8m0_t、int16_t、uint16_t、half、bfloat16_t、int32_t、uint32_t、float、complex32</p>
</td>
</tr>
<tr id="row1532804818512"><td class="cellrowborder" valign="top" headers="mcps1.2.4.1.1 "><a name="ul15365240612"></a><a name="ul15365240612"></a><ul id="ul15365240612"><li>GM -&gt; A1、B1（GM -&gt; L1 Buffer）</li></ul>
</td>
<td class="cellrowborder" valign="top" headers="mcps1.2.4.1.2 "><p id="p22221129269"><a name="p22221129269"></a><a name="p22221129269"></a>bool、int8_t、uint8_t、fp4x2_e2m1_t、fp4x2_e1m2_t、hifloat8_t、fp8_e5m2_t、fp8_e4m3fn_t、fp8_e8m0_t、int16_t、uint16_t、half、bfloat16_t、int32_t、uint32_t、float、complex32</p>
</td>
</tr>
</tbody>
</table>

**表 5**  Local Memory -\> Local Memory具体通路和支持的数据类型

<a name="table199181635105620"></a>
<table><thead align="left"><tr id="row79189351566"><th class="cellrowborder" valign="top" width="11.04%" id="mcps1.2.4.1.1"><p id="p159189351562"><a name="p159189351562"></a><a name="p159189351562"></a>产品型号</p>
</th>
<th class="cellrowborder" valign="top" width="24.05%" id="mcps1.2.4.1.2"><p id="p7918133555612"><a name="p7918133555612"></a><a name="p7918133555612"></a>数据通路</p>
</th>
<th class="cellrowborder" valign="top" width="64.91%" id="mcps1.2.4.1.3"><p id="p18918103516566"><a name="p18918103516566"></a><a name="p18918103516566"></a>源操作数和目的操作数的数据类型 (两者保持一致)</p>
</th>
</tr>
</thead>
<tbody><tr id="row2761529173511"><td class="cellrowborder" valign="top" width="11.04%" headers="mcps1.2.4.1.1 "><p id="p89153712353"><a name="p89153712353"></a><a name="p89153712353"></a><span id="ph7919377356"><a name="ph7919377356"></a><a name="ph7919377356"></a>Ascend 950PR/Ascend 950DT</span></p>
</td>
<td class="cellrowborder" valign="top" width="24.05%" headers="mcps1.2.4.1.2 "><p id="p16771429123518"><a name="p16771429123518"></a><a name="p16771429123518"></a>VECIN、VECCALC、VECOUT -&gt; TSCM（UB -&gt; L1 Buffer）</p>
</td>
<td class="cellrowborder" valign="top" width="64.91%" headers="mcps1.2.4.1.3 "><p id="p8567350105519"><a name="p8567350105519"></a><a name="p8567350105519"></a>bool、int8_t、uint8_t、hifloat8_t、fp8_e5m2_t、fp8_e4m3fn_t、fp8_e8m0_t、int16_t、uint16_t、half、bfloat16_t、int32_t、uint32_t、float、complex32</p>
</td>
</tr>
</tbody>
</table>

## 调用示例<a name="section10309141400"></a>

```
#include "kernel_operator.h"
class KernelDataCopyGm2UbNd2Nz{
public:
    __aicore__ inline KernelDataCopyGm2UbNd2Nz()
    {}
    __aicore__ inline void Init(__gm__ uint8_t* dstGm, __gm__ uint8_t* srcGm)
    {
        AscendC::Nd2NzParams intriParamsIn{1, 32, 32, 0, 32, 32, 1, 0};
        intriParams = intriParamsIn;
        srcGlobal.SetGlobalBuffer((__gm__ half *)srcGm);
        dstGlobal.SetGlobalBuffer((__gm__ half *)dstGm);
        pipe.InitBuffer(inQueueSrcVecIn, 1, intriParams.nValue * intriParams.dValue * sizeof(half));
        pipe.InitBuffer(inQueueSrcVecOut, 1, intriParams.nValue * intriParams.dValue * sizeof(half));
    }
    __aicore__ inline void Process()
    {
        CopyIn();
        Compute();
        CopyOut();
    }
private:
    __aicore__ inline void CopyIn()
    {
        AscendC::LocalTensor<half> srcLocal = inQueueSrcVecIn.AllocTensor<half>();
        AscendC::DataCopy(srcLocal, srcGlobal, intriParams);
        inQueueSrcVecIn.EnQue(srcLocal);
    }
    __aicore__ inline void Compute()
    {
        AscendC::LocalTensor<half> srcLocal = inQueueSrcVecIn.DeQue<half>();
        AscendC::LocalTensor<half> dstLocal = inQueueSrcVecOut.AllocTensor<half>();
        AscendC::DataCopy(dstLocal, srcLocal, intriParams.nValue * intriParams.dValue);
        inQueueSrcVecOut.EnQue(dstLocal);
        inQueueSrcVecIn.FreeTensor(srcLocal);
    }
    __aicore__ inline void CopyOut()
    {
        AscendC::LocalTensor<half> dstLocal = inQueueSrcVecOut.DeQue<half>();
        AscendC::DataCopy(dstGlobal, dstLocal, intriParams.nValue * intriParams.dValue);
        inQueueSrcVecOut.FreeTensor(dstLocal);
    }
private:
    AscendC::TPipe pipe;
    AscendC::TQue<AscendC::TPosition::VECIN, 1> inQueueSrcVecIn;
    AscendC::TQue<AscendC::TPosition::VECOUT, 1> inQueueSrcVecOut;
    AscendC::GlobalTensor<half> srcGlobal;
    AscendC::GlobalTensor<half> dstGlobal;
    AscendC::Nd2NzParams intriParams;
};
extern "C" __global__ __aicore__ void kernel_data_copy_nd2nz_ub2out(__gm__ uint8_t* src_gm, __gm__ uint8_t* dst_gm)
{
    KernelDataCopyGm2UbNd2Nz op;
    op.Init(dst_gm, src_gm);
    op.Process();
}
```

结果示例：

```
输入数据(srcGlobal): [1 2 3 ... 1024]
输出数据(dstGlobal):[1 2 ... 15 16 33 34 ... 47 48 65 66 ... 79 80 97 98 ... 111 112 ... 1009 1010... 1023 1024]
```

