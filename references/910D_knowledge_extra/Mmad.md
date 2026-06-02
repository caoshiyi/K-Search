# Mmad<a name="ZH-CN_TOPIC_0000002523344176"></a>

## 产品支持情况<a name="section1550532418810"></a>

<a name="table38301303189"></a>
<table><thead align="left"><tr id="row20831180131817"><th class="cellrowborder" valign="top" width="52.800000000000004%" id="mcps1.1.4.1.1"><p id="p1883113061818"><a name="p1883113061818"></a><a name="p1883113061818"></a><span id="ph20833205312295"><a name="ph20833205312295"></a><a name="ph20833205312295"></a>产品</span></p>
</th>
<th class="cellrowborder" align="center" valign="top" width="24.51%" id="mcps1.1.4.1.2"><p id="p78368183115"><a name="p78368183115"></a><a name="p78368183115"></a>是否支持（</p>
<p id="p18297174143119"><a name="p18297174143119"></a><a name="p18297174143119"></a>不传入bias的原型</p>
<p id="p783113012187"><a name="p783113012187"></a><a name="p783113012187"></a>）</p>
</th>
<th class="cellrowborder" align="center" valign="top" width="22.689999999999998%" id="mcps1.1.4.1.3"><p id="p13733161315318"><a name="p13733161315318"></a><a name="p13733161315318"></a>是否支持（</p>
<p id="p1273341316313"><a name="p1273341316313"></a><a name="p1273341316313"></a>传入bias的原型</p>
<p id="p1073311310312"><a name="p1073311310312"></a><a name="p1073311310312"></a>）</p>
</th>
</tr>
</thead>
<tbody><tr id="row1272474920205"><td class="cellrowborder" valign="top" width="52.800000000000004%" headers="mcps1.1.4.1.1 "><p id="p17301775812"><a name="p17301775812"></a><a name="p17301775812"></a><span id="ph2272194216543"><a name="ph2272194216543"></a><a name="ph2272194216543"></a>Ascend 950PR/Ascend 950DT</span></p>
</td>
<td class="cellrowborder" align="center" valign="top" width="24.51%" headers="mcps1.1.4.1.2 "><p id="p37256491200"><a name="p37256491200"></a><a name="p37256491200"></a>√</p>
</td>
<td class="cellrowborder" align="center" valign="top" width="22.689999999999998%" headers="mcps1.1.4.1.3 "><p id="p14301163033011"><a name="p14301163033011"></a><a name="p14301163033011"></a>√</p>
</td>
</tr>
</tbody>
</table>

## 功能说明<a name="section618mcpsimp"></a>

**功能一：**完成矩阵乘加（C += A \* B）操作。矩阵ABC分别为A2/B2/CO1中的数据。

-   ABC矩阵的数据排布格式分别为NZ，ZN，NZ。

    矩阵A：每个分形矩阵内部是行主序，分形矩阵之间是列主序。简称小Z大N格式。其shape为16 x \(32B/sizeof\(AType\)\)，大小为512Byte。

    矩阵B：每个分形矩阵内部是列主序，分形矩阵之间是行主序。简称小N大Z格式。其shape为 \(32B/sizeof\(BType\)\) x 16，大小为512Byte。

    矩阵C：每个分形矩阵内部是行主序，分形矩阵之间是列主序。简称小Z大N格式。其shape为16 x 16，大小为256个元素。

    <!-- img2text -->
```text
                              <────────────────────────────── K ──────────────────────────────>
                              <────── K0 ──────>
           ↑ M
           │              Matrix A
           │      ┌─────────┬─────────┬─────────┬─────────┐
           │      │    Z→   │    Z→   │    Z→   │    Z→   │
           │      ├─────────┼─────────┼─────────┼─────────┤
           │      │    Z→   │    Z→   │    Z→   │    Z→   │
           │      ├─────────┼─────────┼─────────┼─────────┤
           │      │    Z→   │    Z→   │    Z→   │    Z→   │
           │      ├─────────┼─────────┼─────────┼─────────┤
           │      │   - -   │   - -   │   - -   │   - -   │
           │      │    Z→   │    Z→   │    Z→   │    Z→   │
           │      └─────────┴─────────┴─────────┴─────────┘
           │                                                    ↓
           │                                      L0A layout aligned to M0
           └────────────────────────────────────────────────────────────────

                                        ×

                              <────────────────────────────── N ──────────────────────────────>
                              <──────── 16 ────────>
           ↑ K
           │              Matrix B
           │      ┌─────────┬─────────┬─────────┬─────────┐
           │      │    N↓   │    N↓   │    N↓   │   - -   │
           │      ├─────────┼─────────┼─────────┼─────────┤
           │      │    N↓   │    N↓   │    N↓   │   - -   │
           │      ├─────────┼─────────┼─────────┼─────────┤
           │      │    N↓   │    N↓   │    N↓   │   - -   │
           │      ├─────────┼─────────┼─────────┼─────────┤
           │      │    N↓   │    N↓   │    N↓   │   - -   │
           │      └─────────┴─────────┴─────────┴─────────┘
           └──────────────────────────────────────────────────────────────→
                                L0B layout aligned to N0

                                        =

                              <────────────────────────────── N ──────────────────────────────>
                              <──────── 16 ────────>
           ↑ M
           │              Matrix C
           │      ┌─────────┬─────────┬─────────┬─────────┐
           │      │    Z→   │    Z→   │    Z→   │   - -   │
           │      ├─────────┼─────────┼─────────┼─────────┤
           │      │    Z→   │    Z→   │    Z→   │   - -   │
           │      ├─────────┼─────────┼─────────┼─────────┤
           │      │    Z→   │    Z→   │    Z→   │   - -   │
           │      ├─────────┼─────────┼─────────┼─────────┤
           │      │   - -   │   - -   │   - -   │   - -   │
           │      │    Z→   │    Z→   │    Z→   │   - -   │
           │      └─────────┴─────────┴─────────┴─────────┘
           │                                                    ↓
           │                                      L0C layout aligned to M0
           └────────────────────────────────────────────────────────────────
                                L0B layout aligned to N0
```

说明:
- 图中共有三部分：`Matrix A × Matrix B = Matrix C`
- `Matrix A` 顶部标注总宽度 `K`，其左上局部宽度标注为 `K0`，左侧高度标注为 `M`
- `Matrix B` 顶部标注总宽度 `N`，其左上局部宽度标注为 `16`，左侧高度标注为 `K`
- `Matrix C` 顶部标注总宽度 `N`，其左上局部宽度标注为 `16`，左侧高度标注为 `M`
- `Matrix A` 右侧标注：`L0A layout aligned to M0`
- `Matrix B` 底部标注：`L0B layout aligned to N0`
- `Matrix C` 右侧标注：`L0C layout aligned to M0`
- `Matrix C` 底部标注：`L0B layout aligned to N0`
- 实线蓝色区域表示有效块，虚线区域表示补齐/对齐部分
- 单元格内部的锯齿箭头方向：
  - `Matrix A`、`Matrix C` 主要表现为横向 `Z→`
  - `Matrix B` 主要表现为纵向 `N↓`

    以下是一个简单的例子，假设分形矩阵的大小是2x2（并不符合真实情况，仅作为示例），矩阵ABC的大小都是4x4。

    <a name="table56024246811"></a>
    <table><tbody><tr id="row9603122410813"><td class="cellrowborder" valign="top" width="25%"><p id="p060342411811"><a name="p060342411811"></a><a name="p060342411811"></a>0</p>
    </td>
    <td class="cellrowborder" valign="top" width="25%"><p id="p13603524285"><a name="p13603524285"></a><a name="p13603524285"></a>1</p>
    </td>
    <td class="cellrowborder" valign="top" width="25%"><p id="p1860372417818"><a name="p1860372417818"></a><a name="p1860372417818"></a>2</p>
    </td>
    <td class="cellrowborder" valign="top" width="25%"><p id="p560322410814"><a name="p560322410814"></a><a name="p560322410814"></a>3</p>
    </td>
    </tr>
    <tr id="row116035244810"><td class="cellrowborder" valign="top" width="25%"><p id="p106034248818"><a name="p106034248818"></a><a name="p106034248818"></a>4</p>
    </td>
    <td class="cellrowborder" valign="top" width="25%"><p id="p26033243812"><a name="p26033243812"></a><a name="p26033243812"></a>5</p>
    </td>
    <td class="cellrowborder" valign="top" width="25%"><p id="p166033243818"><a name="p166033243818"></a><a name="p166033243818"></a>6</p>
    </td>
    <td class="cellrowborder" valign="top" width="25%"><p id="p2603424480"><a name="p2603424480"></a><a name="p2603424480"></a>7</p>
    </td>
    </tr>
    <tr id="row360315249815"><td class="cellrowborder" valign="top" width="25%"><p id="p1860316241786"><a name="p1860316241786"></a><a name="p1860316241786"></a>8</p>
    </td>
    <td class="cellrowborder" valign="top" width="25%"><p id="p13603624784"><a name="p13603624784"></a><a name="p13603624784"></a>9</p>
    </td>
    <td class="cellrowborder" valign="top" width="25%"><p id="p176037241189"><a name="p176037241189"></a><a name="p176037241189"></a>10</p>
    </td>
    <td class="cellrowborder" valign="top" width="25%"><p id="p19603182417818"><a name="p19603182417818"></a><a name="p19603182417818"></a>11</p>
    </td>
    </tr>
    <tr id="row1160312241281"><td class="cellrowborder" valign="top" width="25%"><p id="p1160332416817"><a name="p1160332416817"></a><a name="p1160332416817"></a>12</p>
    </td>
    <td class="cellrowborder" valign="top" width="25%"><p id="p1460314246820"><a name="p1460314246820"></a><a name="p1460314246820"></a>13</p>
    </td>
    <td class="cellrowborder" valign="top" width="25%"><p id="p12603124087"><a name="p12603124087"></a><a name="p12603124087"></a>14</p>
    </td>
    <td class="cellrowborder" valign="top" width="25%"><p id="p560316241082"><a name="p560316241082"></a><a name="p560316241082"></a>15</p>
    </td>
    </tr>
    </tbody>
    </table>

    矩阵A的排列顺序:0，1，4，5，8，9，12，13，2，3，6，7，10，11，14，15。

    矩阵B的排列顺序:0，4，1，5，2，6，3，7，8，12，9，13，10，14，11，15。

    矩阵C的排列顺序:0，1，4，5，8，9，12，13，2，3，6，7，10，11，14，15。

**功能二：**针对Ascend 950PR/Ascend 950DT，还支持包含缩放功能的矩阵乘，公式如下：C = \(ScaleA ⊗ A\) ∗ \(ScaleB ⊗ B\) + C。ScaleA和ScaleB通过LoadData2DMX接口载入。

-   ScaleA的分形格式为小Z大Z ，shape为（16，2），数据类型为fp8\_e8m0\_t。
-   ScaleB的分形格式为小N大N，shape为 （2，16），数据类型为fp8\_e8m0\_t。

以AB矩阵均为fp4x2\_e2m1\_t数据类型为例，下图展示了ScaleA、ScaleB的分形排布格式和缩放功能原理：

<!-- img2text -->
```text
┌──────────────────────────────────────────────┐    ┌──────────────────────────────────────────────┐    ┌──────────────────────────────────────┐
│                   Matrix A                   │    │                   Matrix B                   │    │               Matrix C               │
│                                              │    │                                              │    │                                      │
│                    ←──── K ────→             │    │                    ←──── N ────→             │    │                ←──── N ────→         │
│                    ←─ K0=64 ─→               │    │                    ←── 16 ──→                │    │                ←── 16 ──→            │
│      ↑                                       │    │      ↑                                       │    │      ↑                               │
│      │16                                     │    │      │ K0=64                                 │    │      │16                             │
│      ↓                                       │    │      ↓                                       │    │      ↓                               │
│   ┌──────┬──────┬──────┬──────┐              │    │   ┌──────┬──────┬──────┬──────┐              │    │   ┌──────┬──────┬──────┬──────┐      │
│   │  ╲→  │  ╲→  │  ╲→  │  ╲→  │              │    │   │  ╲↓  │  ╲↓  │  ╲↓  │  ╲↓  │              │    │   │  ╲→  │  ╲→  │  ╲→  │  ╲→  │      │
│   ├──────┼──────┼──────┼──────┤              │    │   ├──────┼──────┼──────┼──────┤              │    │   ├──────┼──────┼──────┼──────┤      │
│ M │  ╲→  │  ╲→  │  ╲→  │  ╲→  │ LOA layout   │    │ K │  ╲↓  │  ╲↓  │  ╲↓  │  ╲↓  │              │    │ M │  ╲→  │  ╲→  │  ╲→  │  ╲→  │ LOC  │
│   ├──────┼──────┼──────┼──────┤ aligned to M0│    │   ├──────┼──────┼──────┼──────┤ aligned to N0│    │   ├──────┼──────┼──────┼──────┤layout│
│   │  ╲→  │  ╲→  │  ╲→  │  ╲→  │              │    │   │  ╲↓  │  ╲↓  │  ╲↓  │  ╲↓  │              │    │   │  ╲→  │  ╲→  │  ╲→  │  ╲→  │aligned│
│   ├──────┼──────┼──────┼──────┤              │    │   ├──────┼──────┼──────┼──────┤              │    │   └──────┴──────┴──────┴──────┘to M0 │
│   │  ╲→  │  ╲→  │  ╲→  │  ╲→  │              │    │   │  ╲↓  │  ╲↓  │  ╲↓  │  ╲↓  │              │    │      LOC layout aligned to N0       │
│   └──────┴──────┴──────┴──────┘              │    │   └──────┴──────┴──────┴──────┘              │    │                                      │
│                                              │    │      L0B layout aligned to N0                │    │                                      │
│              Data Matrix A (FP4)             │    │              Data Matrix B (FP4)             │    │                                      │
│                                              │    │                                              │    │                                      │
│                     ×                        │    │                     ×                        │    │                  =                   │
│                                              │    │                                              │    │                                      │
│                 ← K/32 Byte →                │    │                                              │    │                                      │
│                    ← 2Byte →                 │    │                  ←──── N ────→               │    │                                      │
│      ↑                                       │    │                  ←── 16 ──→                  │    │                                      │
│      │16                                     │    │     ↑                                        │    │                                      │
│      ↓                                       │    │     │ K/32 Byte                              │    │                                      │
│   ┌──┬──┬──┬──┐                              │    │     ↓                                        │    │                                      │
│   │╲↓│╲↓│╲↓│╲↓│                              │    │   ┌──────┬──────┬──────┬──────┐              │    │                                      │
│ M ├──┼──┼──┼──┤ LOA layout aligned to M0     │    │   │  ╲↓  │  ╲↓  │  ╲↓  │  ╲↓  │              │    │                                      │
│   │╲↓│╲↓│╲↓│╲↓│                              │    │   └──────┴──────┴──────┴──────┘              │    │                                      │
│   ├──┼──┼──┼──┤                              │    │      ↑                                       │    │                                      │
│   │╲↓│╲↓│╲↓│╲↓│                              │    │    2Byte                                     │    │                                      │
│   ├──┼──┼──┼──┤                              │    │      ↓                                       │    │                                      │
│   │╲↓│╲↓│╲↓│╲↓│                              │    │      L0B layout aligned to N0                │    │                                      │
│   └──┴──┴──┴──┘                              │    │                                              │    │                                      │
│                                              │    │               Scale Matrix B                 │    │                                      │
│            Scale Matrix A (e8m0)             │    │                                              │    │                                      │
└──────────────────────────────────────────────┘    └──────────────────────────────────────────────┘    └──────────────────────────────────────┘
```

## 函数原型<a name="section620mcpsimp"></a>

-   不传入bias

    ```
    template <typename T, typename U, typename S>
    __aicore__ inline void Mmad(const LocalTensor<T>& dst, const LocalTensor<U>& fm, const LocalTensor<S>& filter, const MmadParams& mmadParams)
    ```

-   传入bias

    ```
    template <typename T, typename U, typename S, typename V>
    __aicore__ inline void Mmad(const LocalTensor<T>& dst, const LocalTensor<U>& fm, const LocalTensor<S>& filter, const LocalTensor<V>& bias, const MmadParams& mmadParams)
    ```

## 参数说明<a name="section622mcpsimp"></a>

**表 1**  模板参数说明

<a name="table4835205712588"></a>
<table><thead align="left"><tr id="zh-cn_topic_0000001429830437_row118356578583"><th class="cellrowborder" valign="top" width="16.28%" id="mcps1.2.3.1.1"><p id="zh-cn_topic_0000001429830437_p48354572582"><a name="zh-cn_topic_0000001429830437_p48354572582"></a><a name="zh-cn_topic_0000001429830437_p48354572582"></a>参数名</p>
</th>
<th class="cellrowborder" valign="top" width="83.72%" id="mcps1.2.3.1.2"><p id="zh-cn_topic_0000001429830437_p583535795817"><a name="zh-cn_topic_0000001429830437_p583535795817"></a><a name="zh-cn_topic_0000001429830437_p583535795817"></a>描述</p>
</th>
</tr>
</thead>
<tbody><tr id="zh-cn_topic_0000001429830437_row1835857145817"><td class="cellrowborder" valign="top" width="16.28%" headers="mcps1.2.3.1.1 "><p id="p1878815299212"><a name="p1878815299212"></a><a name="p1878815299212"></a>T</p>
</td>
<td class="cellrowborder" valign="top" width="83.72%" headers="mcps1.2.3.1.2 "><p id="zh-cn_topic_0000001429830437_p168351657155818"><a name="zh-cn_topic_0000001429830437_p168351657155818"></a><a name="zh-cn_topic_0000001429830437_p168351657155818"></a>目的操作数的数据类型。</p>
</td>
</tr>
<tr id="row136464572120"><td class="cellrowborder" valign="top" width="16.28%" headers="mcps1.2.3.1.1 "><p id="p93642452214"><a name="p93642452214"></a><a name="p93642452214"></a>U</p>
</td>
<td class="cellrowborder" valign="top" width="83.72%" headers="mcps1.2.3.1.2 "><p id="p1636494511213"><a name="p1636494511213"></a><a name="p1636494511213"></a>左矩阵的数据类型。</p>
</td>
</tr>
<tr id="row412811612220"><td class="cellrowborder" valign="top" width="16.28%" headers="mcps1.2.3.1.1 "><p id="p16128661229"><a name="p16128661229"></a><a name="p16128661229"></a>S</p>
</td>
<td class="cellrowborder" valign="top" width="83.72%" headers="mcps1.2.3.1.2 "><p id="p312816614228"><a name="p312816614228"></a><a name="p312816614228"></a>右矩阵的数据类型。</p>
</td>
</tr>
<tr id="row5725191113227"><td class="cellrowborder" valign="top" width="16.28%" headers="mcps1.2.3.1.1 "><p id="p10725151162211"><a name="p10725151162211"></a><a name="p10725151162211"></a>V</p>
</td>
<td class="cellrowborder" valign="top" width="83.72%" headers="mcps1.2.3.1.2 "><p id="p117251111172216"><a name="p117251111172216"></a><a name="p117251111172216"></a>Bias矩阵的数据类型。</p>
</td>
</tr>
</tbody>
</table>

**表 2**  参数说明

<a name="table8955841508"></a>
<table><thead align="left"><tr id="row15956194105014"><th class="cellrowborder" valign="top" width="13.661366136613662%" id="mcps1.2.4.1.1"><p id="p7956144195014"><a name="p7956144195014"></a><a name="p7956144195014"></a>参数名称</p>
</th>
<th class="cellrowborder" valign="top" width="10.341034103410342%" id="mcps1.2.4.1.2"><p id="p1295624145013"><a name="p1295624145013"></a><a name="p1295624145013"></a>输入/输出</p>
</th>
<th class="cellrowborder" valign="top" width="75.997599759976%" id="mcps1.2.4.1.3"><p id="p16956144145011"><a name="p16956144145011"></a><a name="p16956144145011"></a>含义</p>
</th>
</tr>
</thead>
<tbody><tr id="row5956546509"><td class="cellrowborder" valign="top" width="13.661366136613662%" headers="mcps1.2.4.1.1 "><p id="p19287714181617"><a name="p19287714181617"></a><a name="p19287714181617"></a>dst</p>
</td>
<td class="cellrowborder" valign="top" width="10.341034103410342%" headers="mcps1.2.4.1.2 "><p id="p192871614151615"><a name="p192871614151615"></a><a name="p192871614151615"></a>输出</p>
</td>
<td class="cellrowborder" valign="top" width="75.997599759976%" headers="mcps1.2.4.1.3 "><p id="p16287121461618"><a name="p16287121461618"></a><a name="p16287121461618"></a>目的操作数，结果矩阵，类型为LocalTensor，支持的TPosition为CO1。</p>
<p id="p20650230163918"><a name="p20650230163918"></a><a name="p20650230163918"></a><span id="ph1479701815419"><a name="ph1479701815419"></a><a name="ph1479701815419"></a>LocalTensor的起始地址需要256个元素对齐。</span></p>
</td>
</tr>
<tr id="row4956154125018"><td class="cellrowborder" valign="top" width="13.661366136613662%" headers="mcps1.2.4.1.1 "><p id="p142871414131614"><a name="p142871414131614"></a><a name="p142871414131614"></a>fm</p>
</td>
<td class="cellrowborder" valign="top" width="10.341034103410342%" headers="mcps1.2.4.1.2 "><p id="p628711148165"><a name="p628711148165"></a><a name="p628711148165"></a>输入</p>
</td>
<td class="cellrowborder" valign="top" width="75.997599759976%" headers="mcps1.2.4.1.3 "><p id="p0287191420164"><a name="p0287191420164"></a><a name="p0287191420164"></a>源操作数，左矩阵a，类型为LocalTensor，支持的TPosition为A2。</p>
<p id="p678295463916"><a name="p678295463916"></a><a name="p678295463916"></a><span id="ph1724515516399"><a name="ph1724515516399"></a><a name="ph1724515516399"></a>LocalTensor的起始地址需要512字节对齐。</span></p>
</td>
</tr>
<tr id="row9486215111718"><td class="cellrowborder" valign="top" width="13.661366136613662%" headers="mcps1.2.4.1.1 "><p id="p1648712150175"><a name="p1648712150175"></a><a name="p1648712150175"></a>filter</p>
</td>
<td class="cellrowborder" valign="top" width="10.341034103410342%" headers="mcps1.2.4.1.2 "><p id="p19487171515178"><a name="p19487171515178"></a><a name="p19487171515178"></a>输入</p>
</td>
<td class="cellrowborder" valign="top" width="75.997599759976%" headers="mcps1.2.4.1.3 "><p id="p3487131516175"><a name="p3487131516175"></a><a name="p3487131516175"></a>源操作数，右矩阵b，类型为LocalTensor，支持的TPosition为B2。</p>
<p id="p75215694018"><a name="p75215694018"></a><a name="p75215694018"></a><span id="ph11804186104020"><a name="ph11804186104020"></a><a name="ph11804186104020"></a>LocalTensor的起始地址需要512字节对齐。</span></p>
</td>
</tr>
<tr id="row596225716543"><td class="cellrowborder" valign="top" width="13.661366136613662%" headers="mcps1.2.4.1.1 "><p id="p696218575548"><a name="p696218575548"></a><a name="p696218575548"></a>bias</p>
</td>
<td class="cellrowborder" valign="top" width="10.341034103410342%" headers="mcps1.2.4.1.2 "><p id="p896275745413"><a name="p896275745413"></a><a name="p896275745413"></a>输入</p>
</td>
<td class="cellrowborder" valign="top" width="75.997599759976%" headers="mcps1.2.4.1.3 "><p id="p3962135715414"><a name="p3962135715414"></a><a name="p3962135715414"></a>源操作数，bias矩阵，类型为LocalTensor，支持的TPosition为C2、CO1。</p>
<p id="p9789121314402"><a name="p9789121314402"></a><a name="p9789121314402"></a><span id="ph181430144409"><a name="ph181430144409"></a><a name="ph181430144409"></a>LocalTensor的起始地址需要128字节对齐。</span></p>
</td>
</tr>
<tr id="row1075785651510"><td class="cellrowborder" valign="top" width="13.661366136613662%" headers="mcps1.2.4.1.1 "><p id="p1728791441620"><a name="p1728791441620"></a><a name="p1728791441620"></a>mmadParams</p>
</td>
<td class="cellrowborder" valign="top" width="10.341034103410342%" headers="mcps1.2.4.1.2 "><p id="p11287151451610"><a name="p11287151451610"></a><a name="p11287151451610"></a>输入</p>
</td>
<td class="cellrowborder" valign="top" width="75.997599759976%" headers="mcps1.2.4.1.3 "><p id="p10391459201718"><a name="p10391459201718"></a><a name="p10391459201718"></a>矩阵乘相关参数，该参数类型的具体定义请参考<span id="ph10562197165916"><a name="ph10562197165916"></a><a name="ph10562197165916"></a>${INSTALL_DIR}</span>/include/ascendc/basic_api/interface/kernel_struct_mm.h，<span id="ph14322531015"><a name="ph14322531015"></a><a name="ph14322531015"></a>${INSTALL_DIR}</span>请替换为CANN软件安装后文件存储路径。</p>
<p id="p1642893817436"><a name="p1642893817436"></a><a name="p1642893817436"></a>MmadParams参数说明请参考<a href="#table15780447181917">表3</a>。</p>
</td>
</tr>
</tbody>
</table>

**表 3**  MmadParams结构体内参数说明

<a name="table15780447181917"></a>
<table><thead align="left"><tr id="row0780947111915"><th class="cellrowborder" valign="top" width="15.310000000000002%" id="mcps1.2.3.1.1"><p id="p1780124771913"><a name="p1780124771913"></a><a name="p1780124771913"></a>参数名称</p>
</th>
<th class="cellrowborder" valign="top" width="84.69%" id="mcps1.2.3.1.2"><p id="p1578014718198"><a name="p1578014718198"></a><a name="p1578014718198"></a>含义</p>
</th>
</tr>
</thead>
<tbody><tr id="row10780647151919"><td class="cellrowborder" valign="top" width="15.310000000000002%" headers="mcps1.2.3.1.1 "><p id="p6340835122118"><a name="p6340835122118"></a><a name="p6340835122118"></a>m</p>
</td>
<td class="cellrowborder" valign="top" width="84.69%" headers="mcps1.2.3.1.2 "><p id="p12340173514212"><a name="p12340173514212"></a><a name="p12340173514212"></a>左矩阵Height，取值范围：m∈[0, 4095] 。默认值为0。</p>
</td>
</tr>
<tr id="row6780947191919"><td class="cellrowborder" valign="top" width="15.310000000000002%" headers="mcps1.2.3.1.1 "><p id="p1934033512213"><a name="p1934033512213"></a><a name="p1934033512213"></a>n</p>
</td>
<td class="cellrowborder" valign="top" width="84.69%" headers="mcps1.2.3.1.2 "><p id="p1634012352218"><a name="p1634012352218"></a><a name="p1634012352218"></a>右矩阵Width，取值范围：n∈[0, 4095] 。默认值为0。</p>
</td>
</tr>
<tr id="row1078074711194"><td class="cellrowborder" valign="top" width="15.310000000000002%" headers="mcps1.2.3.1.1 "><p id="p334033518217"><a name="p334033518217"></a><a name="p334033518217"></a>k</p>
</td>
<td class="cellrowborder" valign="top" width="84.69%" headers="mcps1.2.3.1.2 "><p id="p1734053517219"><a name="p1734053517219"></a><a name="p1734053517219"></a>左矩阵Width、右矩阵Height，取值范围：k∈[0, 4095] 。默认值为0。</p>
</td>
</tr>
<tr id="row6841419175220"><td class="cellrowborder" valign="top" width="15.310000000000002%" headers="mcps1.2.3.1.1 "><p id="p18411019105212"><a name="p18411019105212"></a><a name="p18411019105212"></a>cmatrixInitVal</p>
</td>
<td class="cellrowborder" valign="top" width="84.69%" headers="mcps1.2.3.1.2 "><p id="p265513193113"><a name="p265513193113"></a><a name="p265513193113"></a>配置C矩阵初始值是否为0。默认值true。</p>
<a name="ul146560193120"></a><a name="ul146560193120"></a><ul id="ul146560193120"><li>true：C矩阵初始值为0；</li><li>false：C矩阵初始值通过cmatrixSource参数进行配置。</li></ul>
</td>
</tr>
<tr id="row1444514394710"><td class="cellrowborder" valign="top" width="15.310000000000002%" headers="mcps1.2.3.1.1 "><p id="p1968195019710"><a name="p1968195019710"></a><a name="p1968195019710"></a>cmatrixSource</p>
</td>
<td class="cellrowborder" valign="top" width="84.69%" headers="mcps1.2.3.1.2 "><p id="p46815014712"><a name="p46815014712"></a><a name="p46815014712"></a>配置C矩阵初始值是否来源于C2（存放Bias的硬件缓存区）。默认值为false。</p>
<a name="ul36813507713"></a><a name="ul36813507713"></a><ul id="ul36813507713"><li>false：来源于CO1；</li></ul>
<a name="ul206805011715"></a><a name="ul206805011715"></a><ul id="ul206805011715"><li>true：来源于C2。</li></ul>
<p id="p717213521147"><a name="p717213521147"></a><a name="p717213521147"></a><span id="ph1217214529416"><a name="ph1217214529416"></a><a name="ph1217214529416"></a>Ascend 950PR/Ascend 950DT</span>，支持配置为true/false。</p>
<p id="p18124101113283"><a name="p18124101113283"></a><a name="p18124101113283"></a>注意：带bias输入的接口配置该参数无效，会根据bias输入的位置来判断C矩阵初始值是否来源于CO1还是C2。</p>
</td>
</tr>
<tr id="row5802135801612"><td class="cellrowborder" valign="top" width="15.310000000000002%" headers="mcps1.2.3.1.1 "><p id="p730693132215"><a name="p730693132215"></a><a name="p730693132215"></a>isBias</p>
</td>
<td class="cellrowborder" valign="top" width="84.69%" headers="mcps1.2.3.1.2 "><p id="p18139740111713"><a name="p18139740111713"></a><a name="p18139740111713"></a><strong id="b1466113595178"><a name="b1466113595178"></a><a name="b1466113595178"></a>该参数废弃，新开发内容不要使用该参数。</strong>如果需要累加初始矩阵，请使用带bias的接口来实现；也可以通过cmatrixInitVal和cmatrixSource参数配置C矩阵的初始值来源来实现。推荐使用带bias的接口，相比于配置cmatrixInitVal和cmatrixSource参数更加简单方便。</p>
<p id="p15674612263"><a name="p15674612263"></a><a name="p15674612263"></a>配置是否需要累加初始矩阵，默认值为false，取值说明如下：</p>
<a name="ul1530653162214"></a><a name="ul1530653162214"></a><ul id="ul1530653162214"><li>false：矩阵乘，无需累加初始矩阵，C = A * B。</li><li>true：矩阵乘加，需要累加初始矩阵，C += A * B。</li></ul>
</td>
</tr>
<tr id="row858572783416"><td class="cellrowborder" valign="top" width="15.310000000000002%" headers="mcps1.2.3.1.1 "><p id="p13585182710349"><a name="p13585182710349"></a><a name="p13585182710349"></a>disableGemv</p>
</td>
<td class="cellrowborder" valign="top" width="84.69%" headers="mcps1.2.3.1.2 "><p id="p5654816133510"><a name="p5654816133510"></a><a name="p5654816133510"></a>M = 1时，用于配置Mmad计算是否开启GEMV。当输入为false时，表示开启GEMV；反之，输入为true时，表示关闭GEMV。</p>
<p id="p125065316343"><a name="p125065316343"></a><a name="p125065316343"></a><span>GEMV（General Matrix-Vector Multiplication）表示实现矩阵和向量的乘积，开启GEMV后，Mmad API 从L0A Buffer读取数据时，数据将以ND格式进行读取，而不会将其视为ZZ格式。</span></p>
<p id="p1216724920529"><a name="p1216724920529"></a><a name="p1216724920529"></a>该参数仅支持如下型号：</p>
<p id="p65941424533"><a name="p65941424533"></a><a name="p65941424533"></a><span id="ph1578616140352"><a name="ph1578616140352"></a><a name="ph1578616140352"></a>Ascend 950PR/Ascend 950DT</span></p>
<p id="p20786171433510"><a name="p20786171433510"></a><a name="p20786171433510"></a></p>
</td>
</tr>
<tr id="row429212884719"><td class="cellrowborder" valign="top" width="15.310000000000002%" headers="mcps1.2.3.1.1 "><p id="p92641454114714"><a name="p92641454114714"></a><a name="p92641454114714"></a>fmOffset</p>
</td>
<td class="cellrowborder" rowspan="4" valign="top" width="84.69%" headers="mcps1.2.3.1.2 "><p id="p6264205416479"><a name="p6264205416479"></a><a name="p6264205416479"></a>预留参数。为后续的功能做保留，开发者暂时无需关注，使用默认值即可。</p>
</td>
</tr>
<tr id="row209292371475"><td class="cellrowborder" valign="top" headers="mcps1.2.3.1.1 "><p id="p12264175418478"><a name="p12264175418478"></a><a name="p12264175418478"></a>enSsparse</p>
</td>
</tr>
<tr id="row48391634204717"><td class="cellrowborder" valign="top" headers="mcps1.2.3.1.1 "><p id="p6265145474711"><a name="p6265145474711"></a><a name="p6265145474711"></a>enWinogradA</p>
</td>
</tr>
<tr id="row2694932114712"><td class="cellrowborder" valign="top" headers="mcps1.2.3.1.1 "><p id="p152651254104718"><a name="p152651254104718"></a><a name="p152651254104718"></a>enWinogradB</p>
</td>
</tr>
<tr id="row1916264744713"><td class="cellrowborder" valign="top" width="15.310000000000002%" headers="mcps1.2.3.1.1 "><p id="p926555412477"><a name="p926555412477"></a><a name="p926555412477"></a>kDirectionAlign</p>
</td>
<td class="cellrowborder" valign="top" width="84.69%" headers="mcps1.2.3.1.2 "><p id="p420713237308"><a name="p420713237308"></a><a name="p420713237308"></a>设置是否需要对齐，默认值为false。</p>
<p id="p15218133935313"><a name="p15218133935313"></a><a name="p15218133935313"></a><span id="ph6218163985319"><a name="ph6218163985319"></a><a name="ph6218163985319"></a>Ascend 950PR/Ascend 950DT</span>，仅支持配置为false。</p>
</td>
</tr>
</tbody>
</table>

**表 4**  dst、fm、filter支持的精度类型组合（Ascend 950PR/Ascend 950DT）

<a name="table03371561072"></a>
<table><thead align="left"><tr id="row143371660716"><th class="cellrowborder" valign="top" width="30.759999999999998%" id="mcps1.2.5.1.1"><p id="p4337161713"><a name="p4337161713"></a><a name="p4337161713"></a><strong id="b833766377"><a name="b833766377"></a><a name="b833766377"></a>左矩阵fm type</strong></p>
</th>
<th class="cellrowborder" valign="top" width="30.919999999999998%" id="mcps1.2.5.1.2"><p id="p83381167717"><a name="p83381167717"></a><a name="p83381167717"></a><strong id="b933876475"><a name="b933876475"></a><a name="b933876475"></a>右矩阵filter type</strong></p>
</th>
<th class="cellrowborder" valign="top" width="16.400000000000002%" id="mcps1.2.5.1.3"><p id="p833876776"><a name="p833876776"></a><a name="p833876776"></a><strong id="b133381661174"><a name="b133381661174"></a><a name="b133381661174"></a>结果矩阵dst type</strong></p>
</th>
<th class="cellrowborder" valign="top" width="21.92%" id="mcps1.2.5.1.4"><p id="p959235418717"><a name="p959235418717"></a><a name="p959235418717"></a>备注</p>
</th>
</tr>
</thead>
<tbody><tr id="row13381761270"><td class="cellrowborder" valign="top" width="30.759999999999998%" headers="mcps1.2.5.1.1 "><p id="p15338569715"><a name="p15338569715"></a><a name="p15338569715"></a>int8_t</p>
</td>
<td class="cellrowborder" valign="top" width="30.919999999999998%" headers="mcps1.2.5.1.2 "><p id="p1033846379"><a name="p1033846379"></a><a name="p1033846379"></a>int8_t</p>
</td>
<td class="cellrowborder" valign="top" width="16.400000000000002%" headers="mcps1.2.5.1.3 "><p id="p73383619710"><a name="p73383619710"></a><a name="p73383619710"></a>int32_t</p>
</td>
<td class="cellrowborder" rowspan="9" valign="top" width="21.92%" headers="mcps1.2.5.1.4 "><p id="p172471636132516"><a name="p172471636132516"></a><a name="p172471636132516"></a>仅支持不含缩放的矩阵乘</p>
</td>
</tr>
<tr id="row633811610710"><td class="cellrowborder" valign="top" headers="mcps1.2.5.1.1 "><p id="p1433846374"><a name="p1433846374"></a><a name="p1433846374"></a>half</p>
</td>
<td class="cellrowborder" valign="top" headers="mcps1.2.5.1.2 "><p id="p123383618718"><a name="p123383618718"></a><a name="p123383618718"></a>half</p>
</td>
<td class="cellrowborder" valign="top" headers="mcps1.2.5.1.3 "><p id="p12338116570"><a name="p12338116570"></a><a name="p12338116570"></a>float</p>
</td>
</tr>
<tr id="row9338761074"><td class="cellrowborder" valign="top" headers="mcps1.2.5.1.1 "><p id="p19338361873"><a name="p19338361873"></a><a name="p19338361873"></a>float</p>
</td>
<td class="cellrowborder" valign="top" headers="mcps1.2.5.1.2 "><p id="p13381661077"><a name="p13381661077"></a><a name="p13381661077"></a>float</p>
</td>
<td class="cellrowborder" valign="top" headers="mcps1.2.5.1.3 "><p id="p18338464718"><a name="p18338464718"></a><a name="p18338464718"></a>float</p>
</td>
</tr>
<tr id="row18338861573"><td class="cellrowborder" valign="top" headers="mcps1.2.5.1.1 "><p id="p103381261679"><a name="p103381261679"></a><a name="p103381261679"></a>bfloat16_t</p>
</td>
<td class="cellrowborder" valign="top" headers="mcps1.2.5.1.2 "><p id="p143381866717"><a name="p143381866717"></a><a name="p143381866717"></a>bfloat16_t</p>
</td>
<td class="cellrowborder" valign="top" headers="mcps1.2.5.1.3 "><p id="p5338176779"><a name="p5338176779"></a><a name="p5338176779"></a>float</p>
</td>
</tr>
<tr id="row163381761371"><td class="cellrowborder" valign="top" headers="mcps1.2.5.1.1 "><p id="p43388619719"><a name="p43388619719"></a><a name="p43388619719"></a>fp8_e4m3fn_t</p>
</td>
<td class="cellrowborder" valign="top" headers="mcps1.2.5.1.2 "><p id="p1333856471"><a name="p1333856471"></a><a name="p1333856471"></a>fp8_e4m3fn_t</p>
</td>
<td class="cellrowborder" valign="top" headers="mcps1.2.5.1.3 "><p id="p73398618717"><a name="p73398618717"></a><a name="p73398618717"></a>float</p>
</td>
</tr>
<tr id="row8727145116415"><td class="cellrowborder" valign="top" headers="mcps1.2.5.1.1 "><p id="p102553714427"><a name="p102553714427"></a><a name="p102553714427"></a>fp8_e4m3fn_t</p>
</td>
<td class="cellrowborder" valign="top" headers="mcps1.2.5.1.2 "><p id="p525514724210"><a name="p525514724210"></a><a name="p525514724210"></a>fp8_e5m2_t</p>
</td>
<td class="cellrowborder" valign="top" headers="mcps1.2.5.1.3 "><p id="p025587154213"><a name="p025587154213"></a><a name="p025587154213"></a>float</p>
</td>
</tr>
<tr id="row1045015515412"><td class="cellrowborder" valign="top" headers="mcps1.2.5.1.1 "><p id="p191159279422"><a name="p191159279422"></a><a name="p191159279422"></a>fp8_e5m2_t</p>
</td>
<td class="cellrowborder" valign="top" headers="mcps1.2.5.1.2 "><p id="p8115132717428"><a name="p8115132717428"></a><a name="p8115132717428"></a>fp8_e4m3fn_t</p>
</td>
<td class="cellrowborder" valign="top" headers="mcps1.2.5.1.3 "><p id="p4116127184212"><a name="p4116127184212"></a><a name="p4116127184212"></a>float</p>
</td>
</tr>
<tr id="row2250205814415"><td class="cellrowborder" valign="top" headers="mcps1.2.5.1.1 "><p id="p1636435217425"><a name="p1636435217425"></a><a name="p1636435217425"></a>fp8_e5m2_t</p>
</td>
<td class="cellrowborder" valign="top" headers="mcps1.2.5.1.2 "><p id="p1036435284212"><a name="p1036435284212"></a><a name="p1036435284212"></a>fp8_e5m2_t</p>
</td>
<td class="cellrowborder" valign="top" headers="mcps1.2.5.1.3 "><p id="p2364105215427"><a name="p2364105215427"></a><a name="p2364105215427"></a>float</p>
</td>
</tr>
<tr id="row115641627161519"><td class="cellrowborder" valign="top" headers="mcps1.2.5.1.1 "><p id="p456402761517"><a name="p456402761517"></a><a name="p456402761517"></a>hifloat8_t</p>
</td>
<td class="cellrowborder" valign="top" headers="mcps1.2.5.1.2 "><p id="p556422718157"><a name="p556422718157"></a><a name="p556422718157"></a>hifloat8_t</p>
</td>
<td class="cellrowborder" valign="top" headers="mcps1.2.5.1.3 "><p id="p135647271154"><a name="p135647271154"></a><a name="p135647271154"></a>float</p>
</td>
</tr>
<tr id="row16167344115612"><td class="cellrowborder" valign="top" width="30.759999999999998%" headers="mcps1.2.5.1.1 "><p id="p55237214574"><a name="p55237214574"></a><a name="p55237214574"></a>fp4x2_e1m2_t</p>
</td>
<td class="cellrowborder" valign="top" width="30.919999999999998%" headers="mcps1.2.5.1.2 "><p id="p1948210346513"><a name="p1948210346513"></a><a name="p1948210346513"></a>fp4x2_e1m2_t</p>
</td>
<td class="cellrowborder" valign="top" width="16.400000000000002%" headers="mcps1.2.5.1.3 "><p id="p416719441569"><a name="p416719441569"></a><a name="p416719441569"></a>float</p>
</td>
<td class="cellrowborder" rowspan="8" valign="top" width="21.92%" headers="mcps1.2.5.1.4 "><p id="p165928541974"><a name="p165928541974"></a><a name="p165928541974"></a>仅支持包含缩放的矩阵乘</p>
</td>
</tr>
<tr id="row6328195345616"><td class="cellrowborder" valign="top" headers="mcps1.2.5.1.1 "><p id="p19329145318569"><a name="p19329145318569"></a><a name="p19329145318569"></a>fp4x2_e2m1_t</p>
</td>
<td class="cellrowborder" valign="top" headers="mcps1.2.5.1.2 "><p id="p15955461257"><a name="p15955461257"></a><a name="p15955461257"></a>fp4x2_e1m2_t</p>
</td>
<td class="cellrowborder" valign="top" headers="mcps1.2.5.1.3 "><p id="p1332935317564"><a name="p1332935317564"></a><a name="p1332935317564"></a>float</p>
</td>
</tr>
<tr id="row2058765065610"><td class="cellrowborder" valign="top" headers="mcps1.2.5.1.1 "><p id="p558735017568"><a name="p558735017568"></a><a name="p558735017568"></a>fp4x2_e1m2_t</p>
</td>
<td class="cellrowborder" valign="top" headers="mcps1.2.5.1.2 "><p id="p18587205065614"><a name="p18587205065614"></a><a name="p18587205065614"></a>fp4x2_e2m1_t</p>
</td>
<td class="cellrowborder" valign="top" headers="mcps1.2.5.1.3 "><p id="p1958713509563"><a name="p1958713509563"></a><a name="p1958713509563"></a>float</p>
</td>
</tr>
<tr id="row646324715616"><td class="cellrowborder" valign="top" headers="mcps1.2.5.1.1 "><p id="p13796541269"><a name="p13796541269"></a><a name="p13796541269"></a>fp4x2_e2m1_t</p>
</td>
<td class="cellrowborder" valign="top" headers="mcps1.2.5.1.2 "><p id="p6621164614"><a name="p6621164614"></a><a name="p6621164614"></a>fp4x2_e2m1_t</p>
</td>
<td class="cellrowborder" valign="top" headers="mcps1.2.5.1.3 "><p id="p1746354711567"><a name="p1746354711567"></a><a name="p1746354711567"></a>float</p>
</td>
</tr>
<tr id="row15137171364320"><td class="cellrowborder" valign="top" headers="mcps1.2.5.1.1 "><p id="p2013813138437"><a name="p2013813138437"></a><a name="p2013813138437"></a>AscendC::mx_fp8_e4m3_t</p>
</td>
<td class="cellrowborder" valign="top" headers="mcps1.2.5.1.2 "><p id="p1925141954616"><a name="p1925141954616"></a><a name="p1925141954616"></a>AscendC::mx_fp8_e4m3_t</p>
</td>
<td class="cellrowborder" valign="top" headers="mcps1.2.5.1.3 "><p id="p1178064134613"><a name="p1178064134613"></a><a name="p1178064134613"></a>float</p>
</td>
</tr>
<tr id="row9645192034314"><td class="cellrowborder" valign="top" headers="mcps1.2.5.1.1 "><p id="p8222142315463"><a name="p8222142315463"></a><a name="p8222142315463"></a>AscendC::mx_fp8_e4m3_t</p>
</td>
<td class="cellrowborder" valign="top" headers="mcps1.2.5.1.2 "><p id="p204516147460"><a name="p204516147460"></a><a name="p204516147460"></a>AscendC::mx_fp8_e5m2_t</p>
</td>
<td class="cellrowborder" valign="top" headers="mcps1.2.5.1.3 "><p id="p510415439460"><a name="p510415439460"></a><a name="p510415439460"></a>float</p>
</td>
</tr>
<tr id="row132901518134312"><td class="cellrowborder" valign="top" headers="mcps1.2.5.1.1 "><p id="p468952711468"><a name="p468952711468"></a><a name="p468952711468"></a>AscendC::mx_fp8_e5m2_t</p>
</td>
<td class="cellrowborder" valign="top" headers="mcps1.2.5.1.2 "><p id="p164186375465"><a name="p164186375465"></a><a name="p164186375465"></a>AscendC::mx_fp8_e4m3_t</p>
</td>
<td class="cellrowborder" valign="top" headers="mcps1.2.5.1.3 "><p id="p19232144419467"><a name="p19232144419467"></a><a name="p19232144419467"></a>float</p>
</td>
</tr>
<tr id="row98407150438"><td class="cellrowborder" valign="top" headers="mcps1.2.5.1.1 "><p id="p9994528104619"><a name="p9994528104619"></a><a name="p9994528104619"></a>AscendC::mx_fp8_e5m2_t</p>
</td>
<td class="cellrowborder" valign="top" headers="mcps1.2.5.1.2 "><p id="p1753553013461"><a name="p1753553013461"></a><a name="p1753553013461"></a>AscendC::mx_fp8_e5m2_t</p>
</td>
<td class="cellrowborder" valign="top" headers="mcps1.2.5.1.3 "><p id="p19462045134614"><a name="p19462045134614"></a><a name="p19462045134614"></a>float</p>
</td>
</tr>
</tbody>
</table>

**表 5**  dst、fm、filter、bias支持的精度类型组合（Ascend 950PR/Ascend 950DT）

<a name="table17716153612"></a>
<table><thead align="left"><tr id="row9716051961"><th class="cellrowborder" valign="top" width="23.1%" id="mcps1.2.6.1.1"><p id="p12716751761"><a name="p12716751761"></a><a name="p12716751761"></a><strong id="b1571645761"><a name="b1571645761"></a><a name="b1571645761"></a>左矩阵fm type</strong></p>
</th>
<th class="cellrowborder" valign="top" width="20.06%" id="mcps1.2.6.1.2"><p id="p13716058619"><a name="p13716058619"></a><a name="p13716058619"></a><strong id="b157161557618"><a name="b157161557618"></a><a name="b157161557618"></a>右矩阵filter type</strong></p>
</th>
<th class="cellrowborder" valign="top" width="15.03%" id="mcps1.2.6.1.3"><p id="p1446612216713"><a name="p1446612216713"></a><a name="p1446612216713"></a><strong id="b1546610213719"><a name="b1546610213719"></a><a name="b1546610213719"></a>bias type</strong></p>
</th>
<th class="cellrowborder" valign="top" width="21.54%" id="mcps1.2.6.1.4"><p id="p10716155661"><a name="p10716155661"></a><a name="p10716155661"></a><strong id="b571615517617"><a name="b571615517617"></a><a name="b571615517617"></a>结果矩阵dst type</strong></p>
</th>
<th class="cellrowborder" valign="top" width="20.27%" id="mcps1.2.6.1.5"><p id="p157162520610"><a name="p157162520610"></a><a name="p157162520610"></a>备注</p>
</th>
</tr>
</thead>
<tbody><tr id="row1716155264"><td class="cellrowborder" valign="top" width="23.1%" headers="mcps1.2.6.1.1 "><p id="p1971785260"><a name="p1971785260"></a><a name="p1971785260"></a>int8_t</p>
</td>
<td class="cellrowborder" valign="top" width="20.06%" headers="mcps1.2.6.1.2 "><p id="p8717351869"><a name="p8717351869"></a><a name="p8717351869"></a>int8_t</p>
</td>
<td class="cellrowborder" valign="top" width="15.03%" headers="mcps1.2.6.1.3 "><p id="p8673171714"><a name="p8673171714"></a><a name="p8673171714"></a>int32_t</p>
</td>
<td class="cellrowborder" valign="top" width="21.54%" headers="mcps1.2.6.1.4 "><p id="p127173519619"><a name="p127173519619"></a><a name="p127173519619"></a>int32_t</p>
</td>
<td class="cellrowborder" rowspan="9" valign="top" width="20.27%" headers="mcps1.2.6.1.5 "><p id="p2717851617"><a name="p2717851617"></a><a name="p2717851617"></a>仅支持不含缩放的矩阵乘</p>
</td>
</tr>
<tr id="row8717195860"><td class="cellrowborder" valign="top" headers="mcps1.2.6.1.1 "><p id="p27171554614"><a name="p27171554614"></a><a name="p27171554614"></a>half</p>
</td>
<td class="cellrowborder" valign="top" headers="mcps1.2.6.1.2 "><p id="p171719514610"><a name="p171719514610"></a><a name="p171719514610"></a>half</p>
</td>
<td class="cellrowborder" valign="top" headers="mcps1.2.6.1.3 "><p id="p545817543612"><a name="p545817543612"></a><a name="p545817543612"></a>float</p>
</td>
<td class="cellrowborder" valign="top" headers="mcps1.2.6.1.4 "><p id="p16717051463"><a name="p16717051463"></a><a name="p16717051463"></a>float</p>
</td>
</tr>
<tr id="row107171457617"><td class="cellrowborder" valign="top" headers="mcps1.2.6.1.1 "><p id="p67178517617"><a name="p67178517617"></a><a name="p67178517617"></a>float</p>
</td>
<td class="cellrowborder" valign="top" headers="mcps1.2.6.1.2 "><p id="p127171759617"><a name="p127171759617"></a><a name="p127171759617"></a>float</p>
</td>
<td class="cellrowborder" valign="top" headers="mcps1.2.6.1.3 "><p id="p13458154367"><a name="p13458154367"></a><a name="p13458154367"></a>float</p>
</td>
<td class="cellrowborder" valign="top" headers="mcps1.2.6.1.4 "><p id="p3717135263"><a name="p3717135263"></a><a name="p3717135263"></a>float</p>
</td>
</tr>
<tr id="row16717559614"><td class="cellrowborder" valign="top" headers="mcps1.2.6.1.1 "><p id="p20717155565"><a name="p20717155565"></a><a name="p20717155565"></a>bfloat16_t</p>
</td>
<td class="cellrowborder" valign="top" headers="mcps1.2.6.1.2 "><p id="p4717451468"><a name="p4717451468"></a><a name="p4717451468"></a>bfloat16_t</p>
</td>
<td class="cellrowborder" valign="top" headers="mcps1.2.6.1.3 "><p id="p1745805419615"><a name="p1745805419615"></a><a name="p1745805419615"></a>float</p>
</td>
<td class="cellrowborder" valign="top" headers="mcps1.2.6.1.4 "><p id="p147171515615"><a name="p147171515615"></a><a name="p147171515615"></a>float</p>
</td>
</tr>
<tr id="row16717751264"><td class="cellrowborder" valign="top" headers="mcps1.2.6.1.1 "><p id="p2717255616"><a name="p2717255616"></a><a name="p2717255616"></a>fp8_e4m3fn_t</p>
</td>
<td class="cellrowborder" valign="top" headers="mcps1.2.6.1.2 "><p id="p147171456613"><a name="p147171456613"></a><a name="p147171456613"></a>fp8_e4m3fn_t</p>
</td>
<td class="cellrowborder" valign="top" headers="mcps1.2.6.1.3 "><p id="p54581545611"><a name="p54581545611"></a><a name="p54581545611"></a>float</p>
</td>
<td class="cellrowborder" valign="top" headers="mcps1.2.6.1.4 "><p id="p177171951616"><a name="p177171951616"></a><a name="p177171951616"></a>float</p>
</td>
</tr>
<tr id="row19717651661"><td class="cellrowborder" valign="top" headers="mcps1.2.6.1.1 "><p id="p187171459616"><a name="p187171459616"></a><a name="p187171459616"></a>fp8_e4m3fn_t</p>
</td>
<td class="cellrowborder" valign="top" headers="mcps1.2.6.1.2 "><p id="p17717165761"><a name="p17717165761"></a><a name="p17717165761"></a>fp8_e5m2_t</p>
</td>
<td class="cellrowborder" valign="top" headers="mcps1.2.6.1.3 "><p id="p14584541167"><a name="p14584541167"></a><a name="p14584541167"></a>float</p>
</td>
<td class="cellrowborder" valign="top" headers="mcps1.2.6.1.4 "><p id="p47171851063"><a name="p47171851063"></a><a name="p47171851063"></a>float</p>
</td>
</tr>
<tr id="row3717751464"><td class="cellrowborder" valign="top" headers="mcps1.2.6.1.1 "><p id="p97185516618"><a name="p97185516618"></a><a name="p97185516618"></a>fp8_e5m2_t</p>
</td>
<td class="cellrowborder" valign="top" headers="mcps1.2.6.1.2 "><p id="p16718251611"><a name="p16718251611"></a><a name="p16718251611"></a>fp8_e4m3fn_t</p>
</td>
<td class="cellrowborder" valign="top" headers="mcps1.2.6.1.3 "><p id="p1845815541611"><a name="p1845815541611"></a><a name="p1845815541611"></a>float</p>
</td>
<td class="cellrowborder" valign="top" headers="mcps1.2.6.1.4 "><p id="p77188514619"><a name="p77188514619"></a><a name="p77188514619"></a>float</p>
</td>
</tr>
<tr id="row197181515618"><td class="cellrowborder" valign="top" headers="mcps1.2.6.1.1 "><p id="p1771817516619"><a name="p1771817516619"></a><a name="p1771817516619"></a>fp8_e5m2_t</p>
</td>
<td class="cellrowborder" valign="top" headers="mcps1.2.6.1.2 "><p id="p1071815563"><a name="p1071815563"></a><a name="p1071815563"></a>fp8_e5m2_t</p>
</td>
<td class="cellrowborder" valign="top" headers="mcps1.2.6.1.3 "><p id="p7458554566"><a name="p7458554566"></a><a name="p7458554566"></a>float</p>
</td>
<td class="cellrowborder" valign="top" headers="mcps1.2.6.1.4 "><p id="p13718554610"><a name="p13718554610"></a><a name="p13718554610"></a>float</p>
</td>
</tr>
<tr id="row7718195468"><td class="cellrowborder" valign="top" headers="mcps1.2.6.1.1 "><p id="p171814520617"><a name="p171814520617"></a><a name="p171814520617"></a>hifloat8_t</p>
</td>
<td class="cellrowborder" valign="top" headers="mcps1.2.6.1.2 "><p id="p8718135261"><a name="p8718135261"></a><a name="p8718135261"></a>hifloat8_t</p>
</td>
<td class="cellrowborder" valign="top" headers="mcps1.2.6.1.3 "><p id="p34581454663"><a name="p34581454663"></a><a name="p34581454663"></a>float</p>
</td>
<td class="cellrowborder" valign="top" headers="mcps1.2.6.1.4 "><p id="p771816513618"><a name="p771816513618"></a><a name="p771816513618"></a>float</p>
</td>
</tr>
<tr id="row1471811517612"><td class="cellrowborder" valign="top" width="23.1%" headers="mcps1.2.6.1.1 "><p id="p207181057610"><a name="p207181057610"></a><a name="p207181057610"></a>fp4x2_e1m2_t</p>
</td>
<td class="cellrowborder" valign="top" width="20.06%" headers="mcps1.2.6.1.2 "><p id="p57185515618"><a name="p57185515618"></a><a name="p57185515618"></a>fp4x2_e1m2_t</p>
</td>
<td class="cellrowborder" valign="top" width="15.03%" headers="mcps1.2.6.1.3 "><p id="p11458205414611"><a name="p11458205414611"></a><a name="p11458205414611"></a>float</p>
</td>
<td class="cellrowborder" valign="top" width="21.54%" headers="mcps1.2.6.1.4 "><p id="p771812516615"><a name="p771812516615"></a><a name="p771812516615"></a>float</p>
</td>
<td class="cellrowborder" rowspan="8" valign="top" width="20.27%" headers="mcps1.2.6.1.5 "><p id="p1571875269"><a name="p1571875269"></a><a name="p1571875269"></a>仅支持包含缩放的矩阵乘</p>
</td>
</tr>
<tr id="row1271805462"><td class="cellrowborder" valign="top" headers="mcps1.2.6.1.1 "><p id="p17181151064"><a name="p17181151064"></a><a name="p17181151064"></a>fp4x2_e2m1_t</p>
</td>
<td class="cellrowborder" valign="top" headers="mcps1.2.6.1.2 "><p id="p2071811513613"><a name="p2071811513613"></a><a name="p2071811513613"></a>fp4x2_e1m2_t</p>
</td>
<td class="cellrowborder" valign="top" headers="mcps1.2.6.1.3 "><p id="p44585546613"><a name="p44585546613"></a><a name="p44585546613"></a>float</p>
</td>
<td class="cellrowborder" valign="top" headers="mcps1.2.6.1.4 "><p id="p9718145160"><a name="p9718145160"></a><a name="p9718145160"></a>float</p>
</td>
</tr>
<tr id="row27182519619"><td class="cellrowborder" valign="top" headers="mcps1.2.6.1.1 "><p id="p12718155167"><a name="p12718155167"></a><a name="p12718155167"></a>fp4x2_e1m2_t</p>
</td>
<td class="cellrowborder" valign="top" headers="mcps1.2.6.1.2 "><p id="p8718105061"><a name="p8718105061"></a><a name="p8718105061"></a>fp4x2_e2m1_t</p>
</td>
<td class="cellrowborder" valign="top" headers="mcps1.2.6.1.3 "><p id="p12458165414615"><a name="p12458165414615"></a><a name="p12458165414615"></a>float</p>
</td>
<td class="cellrowborder" valign="top" headers="mcps1.2.6.1.4 "><p id="p2718754614"><a name="p2718754614"></a><a name="p2718754614"></a>float</p>
</td>
</tr>
<tr id="row1071810512610"><td class="cellrowborder" valign="top" headers="mcps1.2.6.1.1 "><p id="p12719151964"><a name="p12719151964"></a><a name="p12719151964"></a>fp4x2_e2m1_t</p>
</td>
<td class="cellrowborder" valign="top" headers="mcps1.2.6.1.2 "><p id="p97191751169"><a name="p97191751169"></a><a name="p97191751169"></a>fp4x2_e2m1_t</p>
</td>
<td class="cellrowborder" valign="top" headers="mcps1.2.6.1.3 "><p id="p154586543614"><a name="p154586543614"></a><a name="p154586543614"></a>float</p>
</td>
<td class="cellrowborder" valign="top" headers="mcps1.2.6.1.4 "><p id="p177191951618"><a name="p177191951618"></a><a name="p177191951618"></a>float</p>
</td>
</tr>
<tr id="row67191353612"><td class="cellrowborder" valign="top" headers="mcps1.2.6.1.1 "><p id="p20719951618"><a name="p20719951618"></a><a name="p20719951618"></a>AscendC::mx_fp8_e4m3_t</p>
</td>
<td class="cellrowborder" valign="top" headers="mcps1.2.6.1.2 "><p id="p77193510615"><a name="p77193510615"></a><a name="p77193510615"></a>AscendC::mx_fp8_e4m3_t</p>
</td>
<td class="cellrowborder" valign="top" headers="mcps1.2.6.1.3 "><p id="p1445813547619"><a name="p1445813547619"></a><a name="p1445813547619"></a>float</p>
</td>
<td class="cellrowborder" valign="top" headers="mcps1.2.6.1.4 "><p id="p1471925768"><a name="p1471925768"></a><a name="p1471925768"></a>float</p>
</td>
</tr>
<tr id="row27196514617"><td class="cellrowborder" valign="top" headers="mcps1.2.6.1.1 "><p id="p1719557618"><a name="p1719557618"></a><a name="p1719557618"></a>AscendC::mx_fp8_e4m3_t</p>
</td>
<td class="cellrowborder" valign="top" headers="mcps1.2.6.1.2 "><p id="p0719195366"><a name="p0719195366"></a><a name="p0719195366"></a>AscendC::mx_fp8_e5m2_t</p>
</td>
<td class="cellrowborder" valign="top" headers="mcps1.2.6.1.3 "><p id="p1445919541262"><a name="p1445919541262"></a><a name="p1445919541262"></a>float</p>
</td>
<td class="cellrowborder" valign="top" headers="mcps1.2.6.1.4 "><p id="p87191952614"><a name="p87191952614"></a><a name="p87191952614"></a>float</p>
</td>
</tr>
<tr id="row8719951260"><td class="cellrowborder" valign="top" headers="mcps1.2.6.1.1 "><p id="p107191259612"><a name="p107191259612"></a><a name="p107191259612"></a>AscendC::mx_fp8_e5m2_t</p>
</td>
<td class="cellrowborder" valign="top" headers="mcps1.2.6.1.2 "><p id="p971935761"><a name="p971935761"></a><a name="p971935761"></a>AscendC::mx_fp8_e4m3_t</p>
</td>
<td class="cellrowborder" valign="top" headers="mcps1.2.6.1.3 "><p id="p1845915541965"><a name="p1845915541965"></a><a name="p1845915541965"></a>float</p>
</td>
<td class="cellrowborder" valign="top" headers="mcps1.2.6.1.4 "><p id="p147191551614"><a name="p147191551614"></a><a name="p147191551614"></a>float</p>
</td>
</tr>
<tr id="row971975863"><td class="cellrowborder" valign="top" headers="mcps1.2.6.1.1 "><p id="p571915966"><a name="p571915966"></a><a name="p571915966"></a>AscendC::mx_fp8_e5m2_t</p>
</td>
<td class="cellrowborder" valign="top" headers="mcps1.2.6.1.2 "><p id="p771975566"><a name="p771975566"></a><a name="p771975566"></a>AscendC::mx_fp8_e5m2_t</p>
</td>
<td class="cellrowborder" valign="top" headers="mcps1.2.6.1.3 "><p id="p1945915548616"><a name="p1945915548616"></a><a name="p1945915548616"></a>float</p>
</td>
<td class="cellrowborder" valign="top" headers="mcps1.2.6.1.4 "><p id="p8719555612"><a name="p8719555612"></a><a name="p8719555612"></a>float</p>
</td>
</tr>
</tbody>
</table>

## 约束说明<a name="section633mcpsimp"></a>

-   dst只支持位于CO1，fm只支持位于A2，filter只支持位于B2。
-   当M、K、N中的任意一个值为0时，该指令不会被执行。
-   当M = 1时，会默认开启GEMV（General Matrix-Vector Multiplication）功能。在这种情况下，Mmad API从L0A Buffer读取数据时，会以ND格式进行读取，而不会将其视为ZZ格式。所以此时左矩阵需要直接按照ND格式进行排布。针对Ascend 950PR/Ascend 950DT，可以通过设置MmadParams的disableGemv参数为true，将该功能关闭。
-   操作数地址对齐要求请参见[通用地址对齐约束](通用说明和约束.md#section796754519912)。
-   通过一个具体的示例来介绍无效数据与有效数据的排布方式。

    数据为half类型，当M=30，K=70，N=40的时候，A2中有2x5个16x16矩阵，B2中有5x3个16x16矩阵，CO1中有2x3个16x16矩阵。在这种场景下M、K和N都不是16的倍数，A2中右下角的矩阵实际有效的数据只有14x6个，但是也需要占一个16x16矩阵的空间，其他无效数据在计算中会被忽略。一个16x16分形的数据块中，无效数据与有效数据排布的方式示意如下：

    <!-- img2text -->
```text
A2
                              K = 70
        ◄──────────────────────────────────────────────►
      ┌─────────────────────────────────────────────────────┐
M = 30│ ┌───────┐ ┌───────┐ ┌───────┐ ┌───────┐ ┌───────┐ │
  ▲   │ │16*16  │ │16*16  │ │16*16  │ │16*16  │ │16*16  │ │
  │   │ └───────┘ └───────┘ └───────┘ └───────┘ └───────┘ │
  │   │                                                     │
  │   │ ┌───────┐ ┌───────┐ ┌───────┐ ┌───────┐ ┌───────┐ │
  ▼   │ │16*16  │ │16*16  │ │16*16  │ │16*16  │ │16*16  │ │
      │ └───────┘ └───────┘ └───────┘ └───────┘ └───────┘ │
      └─────────────────────────────────────────────────────┘


B2
                    N = 40
        ◄──────────────────────────►
      ┌───────────────────────────────┐
K = 70│ ┌───────┐ ┌───────┐ ┌───────┐ │
  ▲   │ │16*16  │ │16*16  │ │16*16  │ │
  │   │ └───────┘ └───────┘ └───────┘ │
  │   │                               │
  │   │ ┌───────┐ ┌───────┐ ┌───────┐ │
  │   │ │16*16  │ │16*16  │ │16*16  │ │
  │   │ └───────┘ └───────┘ └───────┘ │
  │   │                               │
  │   │ ┌───────┐ ┌───────┐ ┌───────┐ │
  │   │ │16*16  │ │16*16  │ │16*16  │ │
  │   │ └───────┘ └───────┘ └───────┘ │
  │   │                               │
  │   │ ┌───────┐ ┌───────┐ ┌───────┐ │
  │   │ │16*16  │ │16*16  │ │16*16  │ │
  │   │ └───────┘ └───────┘ └───────┘ │
  │   │                               │
  ▼   │ ┌───────┐ ┌───────┐ ┌───────┐ │
      │ │16*16  │ │16*16  │ │16*16  │ │
      │ └───────┘ └───────┘ └───────┘ │
      └───────────────────────────────┘


CO1
                    N = 40
        ◄──────────────────────────►
      ┌───────────────────────────────┐
M = 30│ ┌───────┐ ┌───────┐ ┌───────┐ │
  ▲   │ │16*16  │ │16*16  │ │16*16  │ │
  │   │ └───────┘ └───────┘ └───────┘ │
  │   │                               │
  ▼   │ ┌───────┐ ┌───────┐ ┌───────┐ │
      │ │16*16  │ │16*16  │ │16*16  │ │
      │ └───────┘ └───────┘ └───────┘ │
      └───────────────────────────────┘
                                      ↓


A2 右下角 16*16 分形中的有效/无效数据排布
                    ↖ 来自 A2 右下角矩阵

      ┌──────────────────────┐
      │──────────────        │
      │██████                │
      │██████                │
      │██████                │
      │██████                │
      │██████                │
      │██████                │
      │██████                │
      │██████                │
      │██████                │
      │██████   ╲            │
      │██████     ╲          │
      │██████       ╲        │
      │██████         ╲      │
      │██████           ╲    │
      │──────────────      ╲ │
      │      无效数据      ╲ │
      └──────────────────────┘
             有效数据
```

说明:
- A2中有2×5个16×16矩阵。
- B2中有5×3个16×16矩阵。
- CO1中有2×3个16×16矩阵。
- 图中尺寸标注为：A2 的 M=30、K=70；B2 的 K=70、N=40；CO1 的 M=30、N=40。
- 左下角示意图表示一个16×16分形的数据块中，“有效数据”位于左侧区域，“无效数据”位于其余区域。
- 该示意图对应上下文中的 A2 右下角矩阵，实际有效数据为 14×6。

## 调用示例<a name="section642mcpsimp"></a>

不含矩阵乘偏置的样例请参考[Mmad样例](https://gitee.com/ascend/samples/tree/master/operator/ascendc/0_introduction/20_mmad_kernellaunch/MmadInvocation)。

包含矩阵乘偏置的样例请参考[包含矩阵乘偏置的Mmad样例](https://gitee.com/ascend/samples/blob/master/operator/ascendc/0_introduction/20_mmad_kernellaunch/MmadBiasInvocation/)。

