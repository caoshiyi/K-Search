# 随路转换DN2NZ搬运（ISASI）<a name="ZH-CN_TOPIC_0000002523344894"></a>

## 产品支持情况<a name="section1550532418810"></a>

<a name="table38301303189"></a>
<table><thead align="left"><tr id="row20831180131817"><th class="cellrowborder" valign="top" width="57.99999999999999%" id="mcps1.1.3.1.1"><p id="p1883113061818"><a name="p1883113061818"></a><a name="p1883113061818"></a><span id="ph20833205312295"><a name="ph20833205312295"></a><a name="ph20833205312295"></a>产品</span></p>
</th>
<th class="cellrowborder" align="center" valign="top" width="42%" id="mcps1.1.3.1.2"><p id="p783113012187"><a name="p783113012187"></a><a name="p783113012187"></a>是否支持</p>
</th>
</tr>
</thead>
<tbody><tr id="row1272474920205"><td class="cellrowborder" valign="top" width="57.99999999999999%" headers="mcps1.1.3.1.1 "><p id="p17301775812"><a name="p17301775812"></a><a name="p17301775812"></a><span id="ph2272194216543"><a name="ph2272194216543"></a><a name="ph2272194216543"></a>Ascend 950PR/Ascend 950DT</span></p>
</td>
<td class="cellrowborder" align="center" valign="top" width="42%" headers="mcps1.1.3.1.2 "><p id="p37256491200"><a name="p37256491200"></a><a name="p37256491200"></a>√</p>
</td>
</tr>
</tbody>
</table>

## 功能说明<a name="section474617392321"></a>

随路格式转换数据搬运，适用于在搬运时进行格式转换。

## 函数原型<a name="section1954364615315"></a>

```
template <typename T, bool enableSmallC0 = false>
__aicore__ inline void DataCopy(const LocalTensor<T>& dst, const GlobalTensor<T>& src, const Dn2NzParams& intriParams);
```

## 参数说明<a name="section622mcpsimp"></a>

**表 1**  模板参数说明

<a name="table49614585413"></a>
<table><thead align="left"><tr id="row996115820412"><th class="cellrowborder" valign="top" width="14.729999999999999%" id="mcps1.2.3.1.1"><p id="p10961458104110"><a name="p10961458104110"></a><a name="p10961458104110"></a>参数名</p>
</th>
<th class="cellrowborder" valign="top" width="85.27%" id="mcps1.2.3.1.2"><p id="p6961155817415"><a name="p6961155817415"></a><a name="p6961155817415"></a>描述</p>
</th>
</tr>
</thead>
<tbody><tr id="row4961205811416"><td class="cellrowborder" valign="top" width="14.729999999999999%" headers="mcps1.2.3.1.1 "><p id="p29619586417"><a name="p29619586417"></a><a name="p29619586417"></a>T</p>
</td>
<td class="cellrowborder" valign="top" width="85.27%" headers="mcps1.2.3.1.2 "><p id="p996110583417"><a name="p996110583417"></a><a name="p996110583417"></a>源操作数或者目的操作数的数据类型。</p>
</td>
</tr>
<tr id="row63251529183414"><td class="cellrowborder" valign="top" width="14.729999999999999%" headers="mcps1.2.3.1.1 "><p id="p425395515263"><a name="p425395515263"></a><a name="p425395515263"></a>enableSmallC0</p>
</td>
<td class="cellrowborder" valign="top" width="85.27%" headers="mcps1.2.3.1.2 "><p id="p17254115514267"><a name="p17254115514267"></a><a name="p17254115514267"></a>SmallC0模式开关：当dValue小于等于4的时候，C0_SIZE会补齐到4 * sizeof(T) Bytes，默认不开启。</p>
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
<td class="cellrowborder" valign="top" width="75.12%" headers="mcps1.2.4.1.3 "><p id="p292591672516"><a name="p292591672516"></a><a name="p292591672516"></a>目的操作数，类型为LocalTensor。</p>
</td>
</tr>
<tr id="row937mcpsimp"><td class="cellrowborder" valign="top" width="15.02%" headers="mcps1.2.4.1.1 "><p id="p3926171610253"><a name="p3926171610253"></a><a name="p3926171610253"></a>src</p>
</td>
<td class="cellrowborder" valign="top" width="9.86%" headers="mcps1.2.4.1.2 "><p id="p4926121682518"><a name="p4926121682518"></a><a name="p4926121682518"></a>输入</p>
</td>
<td class="cellrowborder" valign="top" width="75.12%" headers="mcps1.2.4.1.3 "><p id="p49261616142516"><a name="p49261616142516"></a><a name="p49261616142516"></a>源操作数，类型为GlobalTensor。</p>
</td>
</tr>
<tr id="row4726155915388"><td class="cellrowborder" valign="top" width="15.02%" headers="mcps1.2.4.1.1 "><p id="p21111014132612"><a name="p21111014132612"></a><a name="p21111014132612"></a>intriParams</p>
</td>
<td class="cellrowborder" valign="top" width="9.86%" headers="mcps1.2.4.1.2 "><p id="p71115140262"><a name="p71115140262"></a><a name="p71115140262"></a>输入</p>
</td>
<td class="cellrowborder" valign="top" width="75.12%" headers="mcps1.2.4.1.3 "><p id="p33608158383"><a name="p33608158383"></a><a name="p33608158383"></a>搬运参数，Dn2NzParams类型，具体参数说明请参考表<a href="#table9182515919">Dn2NzParams结构体参数定义</a>；具体定义请参考<span id="ph10562197165916"><a name="ph10562197165916"></a><a name="ph10562197165916"></a>${INSTALL_DIR}</span>/include/ascendc/basic_api/interface/kernel_struct_data_copy.h，<span id="ph14322531015"><a name="ph14322531015"></a><a name="ph14322531015"></a>${INSTALL_DIR}</span>请替换为CANN软件安装后文件存储路径。</p>
</td>
</tr>
</tbody>
</table>

**表 3**  Dn2NzParams结构体参数定义

<a name="table9182515919"></a>
<table><thead align="left"><tr id="row151816516917"><th class="cellrowborder" valign="top" width="15%" id="mcps1.2.3.1.1"><p id="p18182513916"><a name="p18182513916"></a><a name="p18182513916"></a>参数名称</p>
</th>
<th class="cellrowborder" valign="top" width="85%" id="mcps1.2.3.1.2"><p id="p11449719134719"><a name="p11449719134719"></a><a name="p11449719134719"></a>含义</p>
</th>
</tr>
</thead>
<tbody><tr id="row1818105113916"><td class="cellrowborder" valign="top" width="15%" headers="mcps1.2.3.1.1 "><p id="p8928559104716"><a name="p8928559104716"></a><a name="p8928559104716"></a>dnNum</p>
</td>
<td class="cellrowborder" valign="top" width="85%" headers="mcps1.2.3.1.2 "><p id="p844917194478"><a name="p844917194478"></a><a name="p844917194478"></a>传输DN矩阵的数目，取值范围：ndNum∈[0, 4095]。</p>
</td>
</tr>
<tr id="row2968131992515"><td class="cellrowborder" valign="top" width="15%" headers="mcps1.2.3.1.1 "><p id="p102019313485"><a name="p102019313485"></a><a name="p102019313485"></a>nValue</p>
</td>
<td class="cellrowborder" valign="top" width="85%" headers="mcps1.2.3.1.2 "><p id="p11299126114913"><a name="p11299126114913"></a><a name="p11299126114913"></a>DN矩阵的行数，取值范围：nValue∈[0, 16384]。</p>
</td>
</tr>
<tr id="row1589112062510"><td class="cellrowborder" valign="top" width="15%" headers="mcps1.2.3.1.1 "><p id="p1397111514816"><a name="p1397111514816"></a><a name="p1397111514816"></a>dValue</p>
</td>
<td class="cellrowborder" valign="top" width="85%" headers="mcps1.2.3.1.2 "><p id="p144493191478"><a name="p144493191478"></a><a name="p144493191478"></a>DN矩阵的列数，取值范围：dValue∈[0,  2^32-1]。</p>
</td>
</tr>
<tr id="row3593192082512"><td class="cellrowborder" valign="top" width="15%" headers="mcps1.2.3.1.1 "><p id="p1217581074813"><a name="p1217581074813"></a><a name="p1217581074813"></a>srcDnMatrixStride</p>
</td>
<td class="cellrowborder" valign="top" width="85%" headers="mcps1.2.3.1.2 "><p id="p1445013195477"><a name="p1445013195477"></a><a name="p1445013195477"></a>源操作数相邻DN矩阵起始地址间的偏移，取值范围：srcDNMatrixStride∈[0,  2^64-1]，单位为元素。</p>
</td>
</tr>
<tr id="row1185111015588"><td class="cellrowborder" valign="top" width="15%" headers="mcps1.2.3.1.1 "><p id="p180672644819"><a name="p180672644819"></a><a name="p180672644819"></a>srcDValue</p>
</td>
<td class="cellrowborder" valign="top" width="85%" headers="mcps1.2.3.1.2 "><p id="p128531313134818"><a name="p128531313134818"></a><a name="p128531313134818"></a>源操作数同一DN矩阵的相邻行起始地址间的偏移，取值范围：srcDValue∈[1,  2^64-1]，单位为元素。</p>
</td>
</tr>
<tr id="row1585714414415"><td class="cellrowborder" valign="top" width="15%" headers="mcps1.2.3.1.1 "><p id="p101497513422"><a name="p101497513422"></a><a name="p101497513422"></a>dstNzC0Stride</p>
</td>
<td class="cellrowborder" valign="top" width="85%" headers="mcps1.2.3.1.2 "><p id="p11492512428"><a name="p11492512428"></a><a name="p11492512428"></a>DN转换到NZ格式后，源操作数中的一列会转换为目的操作数的多行。dstNzC0Stride表示，目的NZ矩阵中，来自源操作数同一列的多行数据相邻行起始地址间的偏移，取值范围：dstNzC0Stride∈[1,  65535]，单位：C0_SIZE（32B）。</p>
</td>
</tr>
<tr id="row1881103664120"><td class="cellrowborder" valign="top" width="15%" headers="mcps1.2.3.1.1 "><p id="p014955124220"><a name="p014955124220"></a><a name="p014955124220"></a>dstNzNStride</p>
</td>
<td class="cellrowborder" valign="top" width="85%" headers="mcps1.2.3.1.2 "><p id="p4149145174213"><a name="p4149145174213"></a><a name="p4149145174213"></a>目的NZ矩阵中，Z型矩阵相邻行起始地址之间的偏移。取值范围：dstNzNStride∈[1, 65535]，单位：C0_SIZE（32B）。</p>
</td>
</tr>
<tr id="row57391832114112"><td class="cellrowborder" valign="top" width="15%" headers="mcps1.2.3.1.1 "><p id="p3149175204211"><a name="p3149175204211"></a><a name="p3149175204211"></a>dstNzMatrixStride</p>
</td>
<td class="cellrowborder" valign="top" width="85%" headers="mcps1.2.3.1.2 "><p id="p01491753428"><a name="p01491753428"></a><a name="p01491753428"></a>目的NZ矩阵中，相邻NZ矩阵起始地址间的偏移，取值范围：dstNzMatrixStride∈[1,  2^32-1]，单位为元素。</p>
</td>
</tr>
</tbody>
</table>

DN2NZ转换示意图如下，样例中参数设置值和解释说明如下（以half数据类型为例）：

-   dnNum = 2，表示传输DN矩阵的数目为2。
-   nValue = 8，DN矩阵的列数，也就是矩阵的宽度为8。
-   dValue = 24，DN矩阵的行数，也就是矩阵的高度为24个元素。
-   srcDnMatrixStride = 96，表达相邻DN矩阵起始地址间的偏移，即：A1与C1之间的间隔，为6个DataBlock，6 \* 16 = 96个元素。
-   srcDValue = 48, 表示一行的所含元素个数，即为3个DataBlock, 3 \* 16 = 48个元素。
-   dstNzC0Stride = 6。DN转换到NZ格式后，源操作数中的一列会转换为目的操作数的多列，例如src中A1和A2为1列，dst中A1和A2被分为2列。多列数据起始地址之间的偏移就是A1和A2在dst中的偏移，偏移为6个datablock。
-   dstNzNStride = 2，表达dst中第x个目的DN矩阵和第x+1个目的DN矩阵的起点的偏移，即A1与B1之间的间隔，即为2个DataBlock。
-   dstNzMatrixStride = 64，表达dst中第x个目的ND矩阵和第x+1个目的ND矩阵的起点的偏移，即A1和C1之间的距离，即为4个DataBlock，4 \* 16 = 64个元素。

<!-- img2text -->
```
src
                    N = 8
              <──────────────>
          ┌─────────┬─────────┬─────────┐
          │   A1    │   B1    │         │
          ├─────────┼─────────┼─────────┤
D=24      │   A2    │   B2    │         │
<────>    ├─────────┼─────────┼─────────┤
srcDnMatrixValue
          │   C1    │   D1    │         │
          ├─────────┼─────────┼─────────┤
          │   C2    │   D2    │         │
          └─────────┴─────────┴─────────┘
          <─────────────────────────────>
                    srcDvalue

                         DN2NZ
                    ─────────────→

dst
                                              D = 24
                                        <──────────────>
                                             C0
                                        <───────>
                                   ┌─────────┬─────────┐
                                   │   A1    │   A2    │
                                   ├─────────┼─────────┤
                              N = 8│   B1    │   B2    │
                            <─────>├─────────┼─────────┤
dstNzMatrixStride                   │         │         │
                                   ├─────────┼─────────┤
                                   │         │         │
                      dstNzC0Stride├─────────┼─────────┤
                                   │   C1    │   C2    │
                                   ├─────────┼─────────┤
                                   │   D1    │   D2    │
                                   └─────────┴─────────┘

映射关系:
A1(src) ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─→ A1(dst)
A2(src) ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─→ A2(dst)

B1(src) ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─→ B1(dst)
B2(src) ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─→ B2(dst)

C1(src) ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─→ C1(dst)
C2(src) ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─→ C2(dst)

D1(src) ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─→ D1(dst)
D2(src) ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─→ D2(dst)
```

说明:
- 左侧为 DN 格式源数据，3 列 × 4 行网格，其中前两列有数据块：A1、B1、A2、B2、C1、D1、C2、D2。
- 右侧为 NZ 格式目的数据，2 列 × 6 行网格，A/B/C/D 被重新排布为按列展开的形式：A1 A2、B1 B2、C1 C2、D1 D2。
- 图中标注保留: `srcDnMatrixValue`、`srcDvalue`、`DN2NZ`、`dstNzMatrixStride`、`dstNzC0Stride`、`N = 8`、`D=24`、`D = 24`、`C0`。
- 斜虚线表示从 src 中对应块到 dst 中对应块的位置映射；因原图存在跨区域斜向对应，使用下方文字列出对应关系。

## 通路说明<a name="section631mcpsimp"></a>

**表 4**  数据通路和数据类型

<a name="table14255161718545"></a>
<table><thead align="left"><tr id="row3255181710543"><th class="cellrowborder" valign="top" width="11.04%" id="mcps1.2.4.1.1"><p id="p52550177546"><a name="p52550177546"></a><a name="p52550177546"></a>支持型号</p>
</th>
<th class="cellrowborder" valign="top" width="23.93%" id="mcps1.2.4.1.2"><p id="p13255191735420"><a name="p13255191735420"></a><a name="p13255191735420"></a>数据通路（通过<a href="TPosition.md#table5376122715308">TPosition</a>表达）</p>
</th>
<th class="cellrowborder" valign="top" width="65.03%" id="mcps1.2.4.1.3"><p id="p6255617175419"><a name="p6255617175419"></a><a name="p6255617175419"></a>源操作数和目的操作数的数据类型 (两者保持一致)</p>
</th>
</tr>
</thead>
<tbody><tr id="row1125561715416"><td class="cellrowborder" valign="top" width="11.04%" headers="mcps1.2.4.1.1 "><p id="p539445910177"><a name="p539445910177"></a><a name="p539445910177"></a><span id="ph1016119599116"><a name="ph1016119599116"></a><a name="ph1016119599116"></a>Ascend 950PR/Ascend 950DT</span></p>
</td>
<td class="cellrowborder" valign="top" width="23.93%" headers="mcps1.2.4.1.2 "><p id="p925571775417"><a name="p925571775417"></a><a name="p925571775417"></a>GM -&gt; A1/B1</p>
</td>
<td class="cellrowborder" valign="top" width="65.03%" headers="mcps1.2.4.1.3 "><p id="p20330102215819"><a name="p20330102215819"></a><a name="p20330102215819"></a>int8_t、uint8_t、fp4x2_e2m1_t、fp4x2_e1m2_t、int16_t、uint16_t、int32_t、uint32_t、half、bfloat16_t、float</p>
</td>
</tr>
</tbody>
</table>

## 返回值说明<a name="section182481936173316"></a>

无

## 约束说明<a name="section455344833317"></a>

无

## 调用示例<a name="section122101199486"></a>

```
template<typename T>
void main_data_copy_dn2nz_kernel(__gm__ uint8_t* __restrict__ srcGm, __gm__ uint8_t* __restrict__ dstGm, __gm__ int32_t dataSize)
{
    TPipe tpipe;
    GlobalTensor<T> srcGlobal;
    GlobalTensor<T> dstGlobal;
    srcGlobal.SetGlobalBuffer(reinterpret_cast<__gm__ T*>(srcGm), dataSize);
    dstGlobal.SetGlobalBuffer(reinterpret_cast<__gm__ T*>(dstGm), dataSize);
    GlobalTensor<half> testGlobal;
    testGlobal.SetGlobalBuffer(reinterpret_cast<__gm__ half*>(dstGm), dataSize);
    TBuf<TPosition::A1> tbuf1;
    tpipe.InitBuffer(tbuf1, 2048);
    LocalTensor<T> inputLocal = tbuf1.Get<T>();
    Dn2NzParams intriParams1;
    intriParams1.dnNum = 1;
    intriParams1.nValue = 32;
    intriParams1.dValue = 32;
    intriParams1.srcDnMatrixStride = 0;
    intriParams1.srcDValue = 32;
    intriParams1.dstNzC0Stride = 32;
    intriParams1.dstNzNStride = 1;
    intriParams1.dstNzMatrixStride = 0;
    DataCopy(inputLocal, srcGlobal, intriParams1);
}
```

