# 随路转换NZ2ND搬运<a name="ZH-CN_TOPIC_0000002523344778"></a>

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

## 功能说明<a name="section12840195813362"></a>

支持在数据搬运时进行NZ到ND格式的转换。

## 函数原型<a name="section1792117555586"></a>

```
template <typename T>
__aicore__ inline void DataCopy(const GlobalTensor<T>& dst, const LocalTensor<T>& src, const Nz2NdParamsFull& intriParams)
```

> **说明：** 
>各原型支持的具体数据通路和数据类型，请参考[支持的通路和数据类型](#section189171223121916)。

## 参数说明<a name="section14983445508"></a>

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
<td class="cellrowborder" valign="top" width="85.27%" headers="mcps1.2.3.1.2 "><p id="p996110583417"><a name="p996110583417"></a><a name="p996110583417"></a>源操作数或者目的操作数的数据类型。支持的数据类型请参考<a href="#section189171223121916">支持的通路和数据类型</a>。</p>
</td>
</tr>
</tbody>
</table>

**表 2**  接口参数说明

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
<td class="cellrowborder" valign="top" width="75.12%" headers="mcps1.2.4.1.3 "><p id="p37551021906"><a name="p37551021906"></a><a name="p37551021906"></a>目的操作数，类型为GlobalTensor。</p>
</td>
</tr>
<tr id="row937mcpsimp"><td class="cellrowborder" valign="top" width="15.02%" headers="mcps1.2.4.1.1 "><p id="p3926171610253"><a name="p3926171610253"></a><a name="p3926171610253"></a>src</p>
</td>
<td class="cellrowborder" valign="top" width="9.86%" headers="mcps1.2.4.1.2 "><p id="p4926121682518"><a name="p4926121682518"></a><a name="p4926121682518"></a>输入</p>
</td>
<td class="cellrowborder" valign="top" width="75.12%" headers="mcps1.2.4.1.3 "><p id="p2055414382913"><a name="p2055414382913"></a><a name="p2055414382913"></a>源操作数，类型为LocalTensor。</p>
</td>
</tr>
<tr id="row997554013220"><td class="cellrowborder" valign="top" width="15.02%" headers="mcps1.2.4.1.1 "><p id="p13976540132215"><a name="p13976540132215"></a><a name="p13976540132215"></a>intriParams</p>
</td>
<td class="cellrowborder" valign="top" width="9.86%" headers="mcps1.2.4.1.2 "><p id="p139761340102213"><a name="p139761340102213"></a><a name="p139761340102213"></a>输入</p>
</td>
<td class="cellrowborder" valign="top" width="75.12%" headers="mcps1.2.4.1.3 "><p id="p99761407226"><a name="p99761407226"></a><a name="p99761407226"></a>搬运参数，类型为<a href="#table15841351172811">Nz2NdParamsFull</a>。</p>
<p id="p58852119618"><a name="p58852119618"></a><a name="p58852119618"></a>具体定义请参考<span id="ph10562197165916"><a name="ph10562197165916"></a><a name="ph10562197165916"></a>${INSTALL_DIR}</span>/include/ascendc/basic_api/interface/kernel_struct_data_copy.h，<span id="ph14322531015"><a name="ph14322531015"></a><a name="ph14322531015"></a>${INSTALL_DIR}</span>请替换为CANN软件安装后文件存储路径。</p>
</td>
</tr>
</tbody>
</table>

**表 3**  Nz2NdParamsFull结构体内参数定义

<a name="table15841351172811"></a>
<table><thead align="left"><tr id="row5583451132815"><th class="cellrowborder" valign="top" width="17.349999999999998%" id="mcps1.2.3.1.1"><p id="p1558335112820"><a name="p1558335112820"></a><a name="p1558335112820"></a>参数名称</p>
</th>
<th class="cellrowborder" valign="top" width="82.65%" id="mcps1.2.3.1.2"><p id="p9583125115289"><a name="p9583125115289"></a><a name="p9583125115289"></a>含义</p>
</th>
</tr>
</thead>
<tbody><tr id="row35848519280"><td class="cellrowborder" valign="top" width="17.349999999999998%" headers="mcps1.2.3.1.1 "><p id="p7583115112814"><a name="p7583115112814"></a><a name="p7583115112814"></a>ndNum</p>
</td>
<td class="cellrowborder" valign="top" width="82.65%" headers="mcps1.2.3.1.2 "><p id="p155843517288"><a name="p155843517288"></a><a name="p155843517288"></a>传输NZ矩阵的数目，取值范围：ndNum∈[0, 4095]。</p>
</td>
</tr>
<tr id="row958417514280"><td class="cellrowborder" valign="top" width="17.349999999999998%" headers="mcps1.2.3.1.1 "><p id="p75841651132814"><a name="p75841651132814"></a><a name="p75841651132814"></a>nValue</p>
</td>
<td class="cellrowborder" valign="top" width="82.65%" headers="mcps1.2.3.1.2 "><p id="p258445192811"><a name="p258445192811"></a><a name="p258445192811"></a>NZ矩阵的行数，取值范围：nValue∈[1, 8192]。</p>
</td>
</tr>
<tr id="row4584551142816"><td class="cellrowborder" valign="top" width="17.349999999999998%" headers="mcps1.2.3.1.1 "><p id="p95841851162811"><a name="p95841851162811"></a><a name="p95841851162811"></a>dValue</p>
</td>
<td class="cellrowborder" valign="top" width="82.65%" headers="mcps1.2.3.1.2 "><p id="p165841451162817"><a name="p165841451162817"></a><a name="p165841451162817"></a>NZ矩阵的列数，取值范围：dValue∈[1, 8192]。dValue必须为16的倍数。</p>
</td>
</tr>
<tr id="row1358415517282"><td class="cellrowborder" valign="top" width="17.349999999999998%" headers="mcps1.2.3.1.1 "><p id="p165841151162813"><a name="p165841151162813"></a><a name="p165841151162813"></a>srcNdMatrixStride</p>
</td>
<td class="cellrowborder" valign="top" width="82.65%" headers="mcps1.2.3.1.2 "><p id="p19584551122814"><a name="p19584551122814"></a><a name="p19584551122814"></a>源相邻NZ矩阵的偏移（头与头），取值范围：srcNdMatrixStride∈[1, 512]，单位256 (16 * 16) 个元素。</p>
</td>
</tr>
<tr id="row558419514287"><td class="cellrowborder" valign="top" width="17.349999999999998%" headers="mcps1.2.3.1.1 "><p id="p95841051202818"><a name="p95841051202818"></a><a name="p95841051202818"></a>srcNStride</p>
</td>
<td class="cellrowborder" valign="top" width="82.65%" headers="mcps1.2.3.1.2 "><p id="p958415116285"><a name="p958415116285"></a><a name="p958415116285"></a>源同一NZ矩阵的相邻Z排布的偏移（头与头），取值范围：srcNStride∈[0, 4096]，单位16个元素。</p>
</td>
</tr>
<tr id="row135841513281"><td class="cellrowborder" valign="top" width="17.349999999999998%" headers="mcps1.2.3.1.1 "><p id="p8584105102812"><a name="p8584105102812"></a><a name="p8584105102812"></a>dstDStride</p>
</td>
<td class="cellrowborder" valign="top" width="82.65%" headers="mcps1.2.3.1.2 "><p id="p14584105162812"><a name="p14584105162812"></a><a name="p14584105162812"></a>目的ND矩阵的相邻行的偏移（头与头），取值范围：dstDStride∈[1, 65535]，单位为元素。</p>
</td>
</tr>
<tr id="row25848518288"><td class="cellrowborder" valign="top" width="17.349999999999998%" headers="mcps1.2.3.1.1 "><p id="p85841051122816"><a name="p85841051122816"></a><a name="p85841051122816"></a>dstNdMatrixStride</p>
</td>
<td class="cellrowborder" valign="top" width="82.65%" headers="mcps1.2.3.1.2 "><p id="p12584165112820"><a name="p12584165112820"></a><a name="p12584165112820"></a>目的ND矩阵中，来自源相邻NZ矩阵的偏移（头与头），取值范围：dstNdMatrixStride∈[1, 65535]，单位为元素。</p>
</td>
</tr>
</tbody>
</table>

以half数据类型为例，NZ2ND转换示意图如下，样例中参数设置值和解释说明如下：

-   ndNum = 2，表示源NZ矩阵的数目为2 \(NZ矩阵1为A1\~A4 + B1\~B4，NZ矩阵2为C1\~C4 + D1\~D4\)。
-   nValue = 4，NZ矩阵的行数，也就是矩阵的高度为4。
-   dValue = 32，NZ矩阵的列数，也就是矩阵的宽度为32个元素。
-   srcNdMatrixStride = 1，表达相邻NZ矩阵起始地址间的偏移，即为A1\~C1的距离，即为256个元素\(16个DataBlock \* 16个元素\)。
-   srcNStride = 4,  表示同一个源NZ矩阵的相邻Z排布的偏移，即为A1到B1的距离，即为64个元素\(4个DataBlock\* 16个元素\)。
-   dstDStride = 160，表达一个目的ND矩阵的相邻行之间的偏移，即A1和A2之间的距离，即为10个DataBlock，即10 \* 16 = 160个元素。
-   dstNdMatrixStride = 48，表达dst中第x个目的ND矩阵的起点和第x+1个目的ND矩阵的起点的偏移，即A1和C1之间的距离，即为3个DataBlock，3 \* 16 = 48个元素。

**图 1**  NZ2ND转换示意图（half数据类型）<a name="fig15851251122815"></a>  
<!-- img2text -->
```text
src
srcNdMatrixStride = 1 单位256个元素
<──────────────────────────────────────────────────────────────────────────────>

srcStride = 4 单位16个元素
<──────────────>

                              DataBlock
                         ↙──────────────↘

                                   第一个NZ矩阵                               第二个NZ矩阵
                         ┌──────────────────────────┐               ┌──────────────────────────┐
nValue = 4               │ ┌──────┬──────┐          │               │ ┌──────┬──────┐          │
↑                        │ │  A1  │  B1  │          │               │ │  C1  │  D1  │          │
│                        │ ├──────┼──────┤          │               │ ├──────┼──────┤          │
│                        │ │  A2  │  B2  │          │               │ │  C2  │  D2  │          │
│                        │ ├──────┼──────┤          │               │ ├──────┼──────┤          │
│                        │ │  A3  │  B3  │          │               │ │  C3  │  D3  │          │
│                        │ ├──────┼──────┤          │               │ ├──────┼──────┤          │
↓                        │ │  A4  │  B4  │          │               │ │  C4  │  D4  │          │
                         │ └──────┴──────┘          │               │ └──────┴──────┘          │
                         └──────────────────────────┘               └──────────────────────────┘
                         <──────────────────────────>
                                dValue = 32

                                        │
                                        │
                                        ↓
                                      NZ2ND
                                        ↓

dst
dstNdMatrixStride = 48
<────────────────────────────────────>

dstDStride = 160
↑
│
│        ┌──────┬──────┬──────┬──────┐
│        │  A1  │  B1  │  C1  │  D1  │────────────────────────→
│        ├──────┼──────┼──────┼──────┤
│        │      │      │      │      │
│        ├──────┼──────┼──────┼──────┤
│        │  A2  │  B2  │  C2  │  D2  │
│        ├──────┼──────┼──────┼──────┤
│        │      │      │      │      │
│        ├──────┼──────┼──────┼──────┤
│        │  A3  │  B3  │  C3  │  D3  │
│        ├──────┼──────┼──────┼──────┤
│        │      │      │      │      │
│        ├──────┼──────┼──────┼──────┤
↓        │  A4  │  B4  │  C4  │  D4  │────────────────────────→
         └──────┴──────┴──────┴──────┘

A4 ─────────────────────────────────────────────────────────────→ D1
```

说明:
- 图中标题语义为：以 float 数据类型为例的 NZ2ND 转换示意图。
- src 部分显示两个 NZ 矩阵：
  - 第一个NZ矩阵：A1~A4 + B1~B4
  - 第二个NZ矩阵：C1~C4 + D1~D4
- dst 部分显示转换后的 ND 排布：
  - 第1行：A1 B1 C1 D1
  - 第2行：A2 B2 C2 D2
  - 第3行：A3 B3 C3 D3
  - 第4行：A4 B4 C4 D4
- 参数文字保留如下：
  - srcNdMatrixStride = 1 单位256个元素
  - srcStride = 4 单位16个元素
  - nValue = 4
  - dValue = 32
  - dstNdMatrixStride = 48
  - dstDStride = 160
  - DataBlock
  - 第一个NZ矩阵
  - 第二个NZ矩阵
  - NZ2ND

以float数据类型为例，NZ2ND转换示意图如下，样例中参数设置值和解释说明如下：

-   ndNum = 2，表示源NZ矩阵的数目为2 \(NZ矩阵1为A1\~A8 + B1\~B8，NZ矩阵2为C1\~C8 + D1\~D8\)。
-   nValue = 4，NZ矩阵的行数，也就是矩阵的高度为4。
-   dValue = 32，NZ矩阵的列数，也就是矩阵的宽度为32个元素。
-   srcNdMatrixStride = 1，表达相邻NZ矩阵起始地址间的偏移，即A1到C1的距离，为256个元素\(32个DataBlock \* 8个元素\)
-   srcNStride = 4,  表示同一个源NZ矩阵的相邻Z排布的偏移，即A1到B1的距离，为64个元素 \(8个DataBlock \* 8个元素\)。
-   dstDStride = 144，表示一个目的ND矩阵的相邻行之间的偏移，即A1和A3之间的距离，为18个DataBlock，即18 \* 8 = 144个元素。
-   dstNdMatrixStride = 40，表示dst中第x个目的ND矩阵的起点和第x+1个目的ND矩阵的起点的偏移，即A1和C1之间的距离，为5个DataBlock，5 \* 8 = 40个元素。

**图 2**  NZ2ND转换示意图（float数据类型）<a name="fig5586175192811"></a>  
<!-- img2text -->
```
src
                               srcNdMatrixStride = 1 单位256个元素
                <───────────────────────────────────────────────────────────────>

srcNStride = 4 单位16个元素
<────────────────────>

nValue = 4
↑
│   ┌────┬────┬────┬────┬────┬────┬────┬────┐      ┌────┬────┬────┬────┐
│   │ A1 │ A2 │ B1 │ B2 │    │    │    │    │      │ C1 │ C2 │ D1 │ D2 │
│   ├────┼────┼────┼────┼────┼────┼────┼────┤      ├────┼────┼────┼────┤
│   │ A3 │ A4 │ B3 │ B4 │    │    │    │    │      │ C3 │ C4 │ D3 │ D4 │
│   ├────┼────┼────┼────┼────┼────┼────┼────┤ DataBlock
│   │ A5 │ A6 │ B5 │ B6 │    │    │    │    │      ├────┼────┼────┼────┤
│   ├────┼────┼────┼────┼────┼────┼────┼────┤      │ C5 │ C6 │ D5 │ D6 │
│   │ A7 │ A8 │ B7 │ B8 │    │    │    │    │      ├────┼────┼────┼────┤
↓   └────┴────┴────┴────┴────┴────┴────┴────┘      │ C7 │ C8 │ D7 │ D8 │
                                                    └────┴────┴────┴────┘
    <────────────────────────────────────>
                 dValue = 32

            第一个NZ矩阵                                      第二个NZ矩阵

      A1,A2,A3,A4,A5,A6,A7,A8 ───────────────→ A列
      B1,B2,B3,B4,B5,B6,B7,B8 ───────────────→ B列
      C1,C2,C3,C4,C5,C6,C7,C8 ───────────────→ C列
      D1,D2,D3,D4,D5,D6,D7,D8 ───────────────→ D列

                                  ↓
                                NZ2ND
                                  ↓

dst

dstNdMatrixStride = 40
<────────────────────────────>

dstDStride = 144
↑
│   ┌────┬────┬────┬────┬────┬────┬────┬────┐
│   │ A1 │ A2 │ B1 │ B2 │ C1 │ C2 │ D1 │ D2 │
│   ├────┼────┼────┼────┼────┼────┼────┼────┤
│   │    │    │    │    │    │    │    │    │
│   ├────┼────┼────┼────┼────┼────┼────┼────┤
│   │ A3 │ A4 │ B3 │ B4 │ C3 │ C4 │ D3 │ D4 │
│   ├────┼────┼────┼────┼────┼────┼────┼────┤
│   │    │    │    │    │    │    │    │    │
│   ├────┼────┼────┼────┼────┼────┼────┼────┤
│   │ A5 │ A6 │ B5 │ B6 │ C5 │ C6 │ D5 │ D6 │
│   ├────┼────┼────┼────┼────┼────┼────┼────┤
│   │    │    │    │    │    │    │    │    │
│   ├────┼────┼────┼────┼────┼────┼────┼────┤
│   │ A7 │ A8 │ B7 │ B8 │ C7 │ C8 │ D7 │ D8 │
↓   └────┴────┴────┴────┴────┴────┴────┴────┘
```

说明:
- 图中上半部分为两个NZ矩阵，左侧是第一个NZ矩阵(A、B)，右侧是第二个NZ矩阵(C、D)。
- `srcNStride = 4 单位16个元素`：同一个源NZ矩阵的相邻Z排布偏移，即 A1 到 B1 的距离。
- `srcNdMatrixStride = 1 单位256个元素`：第一个NZ矩阵到第二个NZ矩阵的起点偏移。
- `nValue = 4`：NZ矩阵按竖向有4个位置。
- `dValue = 32`：单个NZ矩阵横向覆盖32个元素。
- 中间 `DataBlock` 表示按数据块搬运/重排。
- 下半部分为NZ2ND后的ND布局：
  - 第1行：`A1 A2 B1 B2 C1 C2 D1 D2`
  - 第3行：`A3 A4 B3 B4 C3 C4 D3 D4`
  - 第5行：`A5 A6 B5 B6 C5 C6 D5 D6`
  - 第7行：`A7 A8 B7 B8 C7 C8 D7 D8`
- `dstDStride = 144`：目的ND矩阵相邻行之间的偏移。
- `dstNdMatrixStride = 40`：dst中第x个目的ND矩阵起点到第x+1个目的ND矩阵起点的偏移。

## 返回值说明<a name="section129001927113216"></a>

无

## 约束说明<a name="section830051273220"></a>

无

## 支持的通路和数据类型<a name="section189171223121916"></a>

下文的数据通路均通过逻辑位置[TPosition](TPosition.md#table5376122715308)来表达，并注明了对应的物理通路。TPosition与物理内存的映射关系见[表1](通用说明和约束.md#table07372185712)。

**表 4**  Local Memory -\> Global Memory具体通路和支持的数据类型

<a name="table393440155510"></a>
<table><thead align="left"><tr id="row393412018551"><th class="cellrowborder" valign="top" width="11.04%" id="mcps1.2.4.1.1"><p id="p16934807556"><a name="p16934807556"></a><a name="p16934807556"></a>产品型号</p>
</th>
<th class="cellrowborder" valign="top" width="24.05%" id="mcps1.2.4.1.2"><p id="p1893430155514"><a name="p1893430155514"></a><a name="p1893430155514"></a>数据通路</p>
</th>
<th class="cellrowborder" valign="top" width="64.91%" id="mcps1.2.4.1.3"><p id="p1493440195515"><a name="p1493440195515"></a><a name="p1493440195515"></a>源操作数和目的操作数的数据类型 (两者保持一致)</p>
</th>
</tr>
</thead>
<tbody><tr id="row1915162915193"><td class="cellrowborder" valign="top" width="11.04%" headers="mcps1.2.4.1.1 "><p id="p699912252718"><a name="p699912252718"></a><a name="p699912252718"></a><span id="ph7999528276"><a name="ph7999528276"></a><a name="ph7999528276"></a>Ascend 950PR/Ascend 950DT</span></p>
</td>
<td class="cellrowborder" valign="top" width="24.05%" headers="mcps1.2.4.1.2 "><p id="p15915142931918"><a name="p15915142931918"></a><a name="p15915142931918"></a>VECOUT -&gt; GM（UB -&gt; GM）</p>
</td>
<td class="cellrowborder" valign="top" width="64.91%" headers="mcps1.2.4.1.3 "><p id="p1876550195017"><a name="p1876550195017"></a><a name="p1876550195017"></a>bool、int8_t、uint8_t、hifloat8_t、fp8_e5m2_t、fp8_e4m3fn_t、fp8_e8m0_t、int16_t、uint16_t、half、bfloat16_t、int32_t、uint32_t、float、complex32、int64_t、uint64_t、double、complex64</p>
</td>
</tr>
</tbody>
</table>

## 调用示例<a name="section2409153316111"></a>

```
#include "kernel_operator.h"
class KernelDataCopyUb2GmNz2Nd {
public:
    __aicore__ inline KernelDataCopyUb2GmNz2Nd()
    {}
    __aicore__ inline void Init(__gm__ uint8_t* dstGm, __gm__ uint8_t* srcGm)
    {
        AscendC::Nz2NdParamsFull intriParamsIn{1, 32, 32, 1, 32, 32, 1};
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
        AscendC::DataCopy(srcLocal, srcGlobal, intriParams.nValue * intriParams.dValue);
        inQueueSrcVecIn.EnQue(srcLocal);
    }
    __aicore__ inline void Compute()
    {
        AscendC::LocalTensor<half> dstLocal = inQueueSrcVecIn.DeQue<half>();
        AscendC::LocalTensor<half> srcOutLocal = inQueueSrcVecOut.AllocTensor<half>();
        AscendC::DataCopy(srcOutLocal, dstLocal, intriParams.nValue * intriParams.dValue);
        inQueueSrcVecOut.EnQue(srcOutLocal);
        inQueueSrcVecIn.FreeTensor(dstLocal);
    }
    __aicore__ inline void CopyOut()
    {
        AscendC::LocalTensor<half> srcOutLocalDe = inQueueSrcVecOut.DeQue<half>();
        AscendC::DataCopy(dstGlobal, srcOutLocalDe, intriParams);
        inQueueSrcVecOut.FreeTensor(srcOutLocalDe);
    }
private:
    AscendC::TPipe pipe;
    AscendC::TQue<AscendC::TPosition::VECIN, 1> inQueueSrcVecIn;
    AscendC::TQue<AscendC::TPosition::VECOUT, 1> inQueueSrcVecOut;
    AscendC::GlobalTensor<half> srcGlobal;
    AscendC::GlobalTensor<half> dstGlobal;
    AscendC::Nz2NdParamsFull intriParams;
};
extern "C" __global__ __aicore__ void kernel_data_copy_nz2nd_ub2out(__gm__ uint8_t* src_gm, __gm__ uint8_t* dst_gm)
{
    KernelDataCopyUb2GmNz2Nd op;
    op.Init(dst_gm, src_gm);
    op.Process();
}
```

结果示例：

```
输入数据(srcGlobal): [1 2 3 ... 1024]
输出数据(dstGlobal):[1 2 ... 15 16 513 514 ... 527 528 17 18 ... 31 32 529 530 ... 543 544 ...497 498 ...  511 512  1009 1010... 1023 1024]
```

