# Load2DMX<a name="ZH-CN_TOPIC_0000002554344555"></a>

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

## 功能说明<a name="section618mcpsimp"></a>

Load2D支持如下数据通路的搬运：

GM-\>A1; GM-\>B1; GM-\>A2; GM-\>B2;

A1-\>A2; B1-\>B2。

## 函数原型<a name="section620mcpsimp"></a>

-   Load2DMX接口

    ```
    template <typename T, typename U = T>
    __aicore__ inline void LoadData(const LocalTensor<U>& dst, const LocalTensor<T>& src, const LocalTensor<fp8_e8m0_t>& srcMx, const LoadData2DParamsV2& loadDataParams, const LoadData2DMxParams& loadMxDataParams)
    ```

-   Load2Dv2MX接口，支持源操作数和目的操作数数据类型不一致

    ```
    template <typename T, typename U>
    __aicore__ inline void LoadData(const LocalTensor<U>& dst, const LocalTensor<T>& src0, const LocalTensor<fp8_e8m0_t>& srcMx, const LoadData2DParamsV2& loadDataParams, const LoadData2DMxParams& loadMxDataParams)
    ```

## 参数说明<a name="section622mcpsimp"></a>

**表 1**  模板参数说明

<a name="table07381635103112"></a>
<table><thead align="left"><tr id="row117393350314"><th class="cellrowborder" valign="top" width="16.55%" id="mcps1.2.3.1.1"><p id="p14739335193119"><a name="p14739335193119"></a><a name="p14739335193119"></a>参数名称</p>
</th>
<th class="cellrowborder" valign="top" width="83.45%" id="mcps1.2.3.1.2"><p id="p8739203514314"><a name="p8739203514314"></a><a name="p8739203514314"></a>含义</p>
</th>
</tr>
</thead>
<tbody><tr id="row18739935193119"><td class="cellrowborder" valign="top" width="16.55%" headers="mcps1.2.3.1.1 "><p id="p173953516310"><a name="p173953516310"></a><a name="p173953516310"></a>T</p>
</td>
<td class="cellrowborder" valign="top" width="83.45%" headers="mcps1.2.3.1.2 "><p id="p12739193516313"><a name="p12739193516313"></a><a name="p12739193516313"></a>源操作数和目的操作数的数据类型。</p>
<a name="ul773801764017"></a><a name="ul773801764017"></a><ul id="ul773801764017"><li><strong id="b1755520123258"><a name="b1755520123258"></a><a name="b1755520123258"></a>Load2DMX接口</strong><p id="p199195873818"><a name="p199195873818"></a><a name="p199195873818"></a><span id="ph1855415101529"><a name="ph1855415101529"></a><a name="ph1855415101529"></a>Ascend 950PR/Ascend 950DT</span>，支持数据类型为：fp4x2_e2m1_t/fp4x2_e1m2_t/fp8_e4m3fn_t/fp8_e5m2_t</p>
</li></ul>
</td>
</tr>
<tr id="row14580104484717"><td class="cellrowborder" valign="top" width="16.55%" headers="mcps1.2.3.1.1 "><p id="p127976994513"><a name="p127976994513"></a><a name="p127976994513"></a>U</p>
</td>
<td class="cellrowborder" valign="top" width="83.45%" headers="mcps1.2.3.1.2 "><a name="ul062813586431"></a><a name="ul062813586431"></a><ul id="ul062813586431"><li>针对Load2DMX接口，U用来表示dst的数据类型，当src为fp8_e4m3fn_t、fp8_e5m2_t时，U需为T对应的MX数据类型，即AscendC::mx_fp8_e4m3_t和AscendC::mx_fp8_e5m2_t，否则编译失败。除此之外的数据类型要求T和U一致。</li></ul>
</td>
</tr>
</tbody>
</table>

**表 2**  通用参数说明

<a name="table18368155193919"></a>
<table><thead align="left"><tr id="row1036805543911"><th class="cellrowborder" valign="top" width="16.89168916891689%" id="mcps1.2.4.1.1"><p id="p1836835511393"><a name="p1836835511393"></a><a name="p1836835511393"></a>参数名称</p>
</th>
<th class="cellrowborder" valign="top" width="11.111111111111112%" id="mcps1.2.4.1.2"><p id="p10368255163915"><a name="p10368255163915"></a><a name="p10368255163915"></a>输入/输出</p>
</th>
<th class="cellrowborder" valign="top" width="71.99719971997199%" id="mcps1.2.4.1.3"><p id="p436875573911"><a name="p436875573911"></a><a name="p436875573911"></a>含义</p>
</th>
</tr>
</thead>
<tbody><tr id="row1436825518395"><td class="cellrowborder" valign="top" width="16.89168916891689%" headers="mcps1.2.4.1.1 "><p id="p9649151061720"><a name="p9649151061720"></a><a name="p9649151061720"></a>dst</p>
</td>
<td class="cellrowborder" valign="top" width="11.111111111111112%" headers="mcps1.2.4.1.2 "><p id="p1649121041718"><a name="p1649121041718"></a><a name="p1649121041718"></a>输出</p>
</td>
<td class="cellrowborder" valign="top" width="71.99719971997199%" headers="mcps1.2.4.1.3 "><p id="p66101129164712"><a name="p66101129164712"></a><a name="p66101129164712"></a>目的操作数，类型为LocalTensor。</p>
<p id="p3610102994715"><a name="p3610102994715"></a><a name="p3610102994715"></a>数据连续排列顺序由目的操作数所在TPosition决定，具体约束如下：</p>
<a name="ul76107290479"></a><a name="ul76107290479"></a><ul id="ul76107290479"><li>A2：ZZ格式/NZ格式；对应的分形大小为16 * (32B / sizeof(T))。</li><li>B2：ZN格式；对应的分形大小为 (32B / sizeof(T))  * 16。</li><li>A1/B1：无格式要求，一般情况下为NZ格式。NZ格式下，对应的分形大小为16 * (32B / sizeof(T))。</li></ul>
</td>
</tr>
<tr id="row1836875519393"><td class="cellrowborder" valign="top" width="16.89168916891689%" headers="mcps1.2.4.1.1 "><p id="p7650141019171"><a name="p7650141019171"></a><a name="p7650141019171"></a>src</p>
</td>
<td class="cellrowborder" valign="top" width="11.111111111111112%" headers="mcps1.2.4.1.2 "><p id="p4650610141715"><a name="p4650610141715"></a><a name="p4650610141715"></a>输入</p>
</td>
<td class="cellrowborder" valign="top" width="71.99719971997199%" headers="mcps1.2.4.1.3 "><p id="p165920401156"><a name="p165920401156"></a><a name="p165920401156"></a>源操作数，类型为LocalTensor或GlobalTensor。</p>
<p id="p192019400435"><a name="p192019400435"></a><a name="p192019400435"></a>数据类型需要与dst保持一致。</p>
</td>
</tr>
<tr id="row12430221172616"><td class="cellrowborder" valign="top" width="16.89168916891689%" headers="mcps1.2.4.1.1 "><p id="p1755271269"><a name="p1755271269"></a><a name="p1755271269"></a>srcMx</p>
</td>
<td class="cellrowborder" valign="top" width="11.111111111111112%" headers="mcps1.2.4.1.2 "><p id="p8430102182615"><a name="p8430102182615"></a><a name="p8430102182615"></a>输入</p>
</td>
<td class="cellrowborder" valign="top" width="71.99719971997199%" headers="mcps1.2.4.1.3 "><p id="p1046717114512"><a name="p1046717114512"></a><a name="p1046717114512"></a>源操作数，类型为LocalTensor，仅支持fp8_e8m0_t类型。</p>
</td>
</tr>
<tr id="row1767431631917"><td class="cellrowborder" valign="top" width="16.89168916891689%" headers="mcps1.2.4.1.1 "><p id="p667418162198"><a name="p667418162198"></a><a name="p667418162198"></a>loadDataParams</p>
</td>
<td class="cellrowborder" valign="top" width="11.111111111111112%" headers="mcps1.2.4.1.2 "><p id="p11675191610195"><a name="p11675191610195"></a><a name="p11675191610195"></a>输入</p>
</td>
<td class="cellrowborder" valign="top" width="71.99719971997199%" headers="mcps1.2.4.1.3 "><p id="p1667541617193"><a name="p1667541617193"></a><a name="p1667541617193"></a>LoadData参数结构体，类型为：</p>
<a name="ul207951119112217"></a><a name="ul207951119112217"></a><ul id="ul207951119112217"><li>LoadData2DMxParams，具体参考<a href="#table15901153712305">表3</a>。</li></ul>
<p id="p21811725744"><a name="p21811725744"></a><a name="p21811725744"></a>上述结构体参数定义请参考<span id="ph10562197165916"><a name="ph10562197165916"></a><a name="ph10562197165916"></a>${INSTALL_DIR}</span>/include/ascendc/basic_api/interface/kernel_struct_mm.h，<span id="ph14322531015"><a name="ph14322531015"></a><a name="ph14322531015"></a>${INSTALL_DIR}</span>请替换为CANN软件安装后文件存储路径。</p>
</td>
</tr>
</tbody>
</table>

**表 3**  LoadData2DMxParams结构体参数说明

<a name="table15901153712305"></a>
<table><thead align="left"><tr id="row1902163763012"><th class="cellrowborder" valign="top" width="18.56%" id="mcps1.2.3.1.1"><p id="p9902137173019"><a name="p9902137173019"></a><a name="p9902137173019"></a>参数名称</p>
</th>
<th class="cellrowborder" valign="top" width="81.44%" id="mcps1.2.3.1.2"><p id="p9902173713015"><a name="p9902173713015"></a><a name="p9902173713015"></a>含义</p>
</th>
</tr>
</thead>
<tbody><tr id="row4902637143017"><td class="cellrowborder" valign="top" width="18.56%" headers="mcps1.2.3.1.1 "><p id="p139022037183014"><a name="p139022037183014"></a><a name="p139022037183014"></a>xStartPosition</p>
</td>
<td class="cellrowborder" valign="top" width="81.44%" headers="mcps1.2.3.1.2 "><p id="p03037491203"><a name="p03037491203"></a><a name="p03037491203"></a>源矩阵X轴方向的起始位置，即M维度方向，单位为1个分形（1个单位代表一个32B的分形）。</p>
</td>
</tr>
<tr id="row1890223716303"><td class="cellrowborder" valign="top" width="18.56%" headers="mcps1.2.3.1.1 "><p id="p490263773015"><a name="p490263773015"></a><a name="p490263773015"></a>yStartPosition</p>
</td>
<td class="cellrowborder" valign="top" width="81.44%" headers="mcps1.2.3.1.2 "><p id="p320282919119"><a name="p320282919119"></a><a name="p320282919119"></a>源矩阵Y轴方向的起始位置，即K维度方向，单位为32B。</p>
</td>
</tr>
<tr id="row19029378306"><td class="cellrowborder" valign="top" width="18.56%" headers="mcps1.2.3.1.1 "><p id="p209027376309"><a name="p209027376309"></a><a name="p209027376309"></a>xStep</p>
</td>
<td class="cellrowborder" valign="top" width="81.44%" headers="mcps1.2.3.1.2 "><p id="p315810577116"><a name="p315810577116"></a><a name="p315810577116"></a>源矩阵X轴方向搬运长度，即M维度方向，单位为1个分形（1个单位代表一个32B的分形）。取值范围：xStep∈[0, 255]。</p>
</td>
</tr>
<tr id="row1090312378303"><td class="cellrowborder" valign="top" width="18.56%" headers="mcps1.2.3.1.1 "><p id="p1190310376309"><a name="p1190310376309"></a><a name="p1190310376309"></a>yStep</p>
</td>
<td class="cellrowborder" valign="top" width="81.44%" headers="mcps1.2.3.1.2 "><p id="p118351012625"><a name="p118351012625"></a><a name="p118351012625"></a>源矩阵Y轴方向搬运长度，即K维度方向，单位为32B。取值范围：yStep∈[0, 255]。</p>
</td>
</tr>
<tr id="row17903193710303"><td class="cellrowborder" valign="top" width="18.56%" headers="mcps1.2.3.1.1 "><p id="p179031637203012"><a name="p179031637203012"></a><a name="p179031637203012"></a>srcStride</p>
</td>
<td class="cellrowborder" valign="top" width="81.44%" headers="mcps1.2.3.1.2 "><p id="p1396134713213"><a name="p1396134713213"></a><a name="p1396134713213"></a>源矩阵X方向前一个分形起始地址与后一个分形起始地址的间隔，单位为32B。</p>
</td>
</tr>
<tr id="row690363712301"><td class="cellrowborder" valign="top" width="18.56%" headers="mcps1.2.3.1.1 "><p id="p29036371303"><a name="p29036371303"></a><a name="p29036371303"></a>dstStride</p>
</td>
<td class="cellrowborder" valign="top" width="81.44%" headers="mcps1.2.3.1.2 "><p id="p295512525213"><a name="p295512525213"></a><a name="p295512525213"></a>目标矩阵X方向前一个分形起始地址与后一个分形起始地址的间隔，单位为32B。</p>
</td>
</tr>
</tbody>
</table>

下面通过一个具体的示例来解释LoadData2DMX结构体参数。假设A矩阵shape为（M，K），则ScaleA矩阵shape为（M，K/32），ScaleA数据类型为fp8\_e8m0\_t，ScaleA矩阵分形排布见[图1](#fig138710913432)。

**图 1**  ScaleA在L0A的分形排布<a name="fig138710913432"></a>  
<!-- img2text -->
```
ScaleA

                               <──────────────────────────────>
                                       K/32 Byte
                               <──────────>
                                   2Byte

                 ↑
                 │
               ┌─┴────────────────────────────────────────────┐
               │    │            │            │            │  │
               │    │     ╱│     │     ╱│     │     ╱│     │  │
               │    │    ╱ │     │    ╱ │     │    ╱ │     │  │
               │    │   ╱  │     │   ╱  │     │   ╱  │     │  │
            16 │    │  ╱   │     │  ╱   │     │  ╱   │     │  │
               │    │ ╱    │     │ ╱    │     │ ╱    │     │  │
               │    │╱     │     │╱     │     │╱     │     │  │
               │    │   →  │     │   →  │     │   →  │     │ →│
               ├────┼────────────┼────────────┼────────────┼──┤
               │    │     ╱│     │     ╱│     │     ╱│     │  │
               │    │    ╱ │     │    ╱ │     │    ╱ │     │  │
               │    │   ╱  │     │   ╱  │     │   ╱  │     │  │
               │    │  ╱   │     │  ╱   │     │  ╱   │     │  │
               │    │ ╱    │     │ ╱    │     │ ╱    │     │  │
               │    │╱     │     │╱     │     │╱     │     │  │
               │    │   →  │     │   →  │     │   →  │     │ →│
               ├────┼────────────────────────────────────────┼──┤
               │    ╲····································→     │
             M │    │     ╱│     │     ╱│     │     ╱│     │  │
               │    │    ╱ │     │    ╱ │     │    ╱ │     │  │
               │    │   ╱  │     │   ╱  │     │   ╱  │     │  │
               │    │  ╱   │     │  ╱   │     │  ╱   │     │  │
               │    │ ╱    │     │ ╱    │     │ ╱    │     │  │
               │    │╱     │     │╱     │     │╱     │     │  │
               │    │   →  │     │   →  │     │   →  │     │ →│
               ├────┼────────────────────────────────────────┼──┤
               │    │     ╱│     │     ╱│     │     ╱│     │  │
               │    │    ╱ │     │    ╱ │     │    ╱ │     │  │
               │    │   ╱  │     │   ╱  │     │   ╱  │     │  │
               │    │  ╱   │     │  ╱   │     │  ╱   │     │  │
               │    │ ╱    │     │ ╱    │     │ ╱    │     │  │
               │    │╱     │     │╱     │     │╱     │     │  │
               │    │   →  │     │   →  │     │   →  │     │ →│
               └────┴────────────────────────────────────────┴──┘
```

说明:
- 外框表示 ScaleA 的排布区域。
- 纵向共有 4 个大块，其中前 3 个为有效分形块，对应文中示例的 `xStep = M / 16 = 3`；每个大块高度为 `16`。
- 横向示意为多个 `2Byte` 单元组成，总宽度标注为 `K/32 Byte`。
- 图中每一行表示 `32Byte`，对应一个分形。
- 每个大块内部的斜线箭头表示数据在分形内的排布/搬运方向。
- 虚线斜箭头表示跨分形块的延续关系。

下图为ScaleA从L1搬运至L0A过程中的配置参数示意。每一行为32Byte，对应着[图1](#fig138710913432)中的一个分形。xStep为M维度分形的个数，如图中的xStep = M / 16 = 3，yStep为K维度32Byte的个数，如图中的yStep = K / 32 / 2 = 21，srcStride和dstStride同理，表示在K维度上32Byte的个数。

<!-- img2text -->
```text
2Byte
<──>
┌─┐
│ │
│ │ 16
│/│
└─┘

if=32Byte, xStartPosition = 2

src
                         xStep = 3
           <────────────────────────────>
           ┌──────┬──────┬──────┬──────┐
           │      │      │      │      │
           │ - - -│- - - │      │      │
           │      │      │      │      │
           ├──────┼──────┼──────┼──────┤
           │      │      │      │      │
           │      │      │      │      │
           │      │      │      │      │
           ├──────┼──────┼──────┼──────┤
           │      │      │      │      │
           │      │      │      │      │
           │      │      │      │      │
           ├──────┼──────┼──────┼──────┤
           │      │      │      │      │
           │      │      │      │      │
           │      │      │      │      │
           ├──────┼──────┼──────┼──────┤
           │      │      │      │      │
           │      │      │      │      │
           │      │      │      │      │
           └──────┴──────┴──────┴──────┘
           ↑
           │ yStartPosition = 38
           │
           ↓
           ↑
           │ yStep = 21
           │
           ↓

           srcStride = 77

                      ───────────────→

dst
                 xStep = 3
           <──────────────────────>
           ┌──────┬──────┬──────┐
           │      │      │      │
           │      │      │      │
           │      │      │      │
yStep = 21 ↓──────┼──────┼──────┤
           │      │      │      │
           │      │      │      │
           ├──────┼──────┼──────┤
           │      │      │      │
           │      │      │      │
           │      │      │      │
           └──────┴──────┴──────┘
                              │
                              │ dstStride
                              │ = 48
                              ↓
```

## 约束说明<a name="section633mcpsimp"></a>

-   操作数地址对齐要求请参见[通用地址对齐约束](通用说明和约束.md#section796754519912)。

## 返回值说明<a name="section640mcpsimp"></a>

无

