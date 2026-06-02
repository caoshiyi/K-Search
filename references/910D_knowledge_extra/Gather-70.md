# Gather<a name="ZH-CN_TOPIC_0000002554424667"></a>

## 产品支持情况<a name="section1550532418810"></a>

<a name="table38301303189"></a>
<table><thead align="left"><tr id="row20831180131817"><th class="cellrowborder" valign="top" width="57.99999999999999%" id="mcps1.1.3.1.1"><p id="p1883113061818"><a name="p1883113061818"></a><a name="p1883113061818"></a><span id="ph20833205312295"><a name="ph20833205312295"></a><a name="ph20833205312295"></a>产品</span></p>
</th>
<th class="cellrowborder" align="center" valign="top" width="42%" id="mcps1.1.3.1.2"><p id="p783113012187"><a name="p783113012187"></a><a name="p783113012187"></a>是否支持</p>
</th>
</tr>
</thead>
<tbody><tr id="row1272474920205"><td class="cellrowborder" valign="top" width="57.99999999999999%" headers="mcps1.1.3.1.1 "><p id="p1622020982912"><a name="p1622020982912"></a><a name="p1622020982912"></a><span id="ph1522010992915"><a name="ph1522010992915"></a><a name="ph1522010992915"></a>Ascend 950PR/Ascend 950DT</span></p>
</td>
<td class="cellrowborder" align="center" valign="top" width="42%" headers="mcps1.1.3.1.2 "><p id="p37256491200"><a name="p37256491200"></a><a name="p37256491200"></a>√</p>
</td>
</tr>
</tbody>
</table>

## 功能说明<a name="section618mcpsimp"></a>

-   收集UB中的元素

    给定源操作数在UB中的基地址和索引，Gather指令根据索引位置将源操作数按元素收集到结果寄存器张量中。收集过程如下图所示：

    <!-- img2text -->
```
Index
┌─────────┬─────────┬─────────┬─────────┬─────────┬─────────┬─────────┬─────────┬─────────┬───────────┐
│ Index 0 │ Index 1 │ Index 2 │ Index 3 │ Index 4 │ Index 5 │ Index 6 │ Index 7 │   ...   │ Index 127 │
└────┬────┴────┬────┴────┬────┴────┬────┴────┬────┴────┬────┴────┬────┴────┬────┴────┬────┴───────┘
     │         │         │         │         │         │         │         │         │
     │         │         │         │         │         │         │         │         │
Src
┌──────────────────────────────────────────────────────────────────────────────────────────────────────────────┐
│                                                   UB空间                                                     │
│                                                                                                              │
│      ┌─────────┐              ┌─────────┐              ┌─────────┐              ┌─────────┐   ┌──────────┐ │
│      │  Ele 2  │              │  Ele 0  │              │  Ele 3  │              │  Ele 6  │   │ ... Ele 127│ │
│      └─────────┘              └─────────┘              └─────────┘              └─────────┘   └──────────┘ │
│                                                                                                              │
│      ┌─────────┐              ┌─────────┐              ┌─────────┐              ┌─────────┐                │
│      │  Ele 4  │              │  Ele 1  │              │  Ele 5  │              │  Ele 7  │                │
│      └─────────┘              └─────────┘              └─────────┘              └─────────┘                │
└──────────────────────────────────────────────────────────────────────────────────────────────────────────────┘

Index 0   → Ele 2
Index 1   → Ele 4
Index 2   → Ele 1
Index 3   → Ele 0
Index 4   → Ele 3
Index 5   → Ele 5
Index 6   → Ele 6
Index 7   → Ele 7
...       → ...
Index 127 → Ele 127

                               ▼
                             Gather
                               ▼

Dst
┌────────┬────────┬────────┬────────┬────────┬────────┬────────┬────────┬───────┬─────────┐
│ Ele 0  │ Ele 1  │ Ele 2  │ Ele 3  │ Ele 4  │ Ele 5  │ Ele 6  │ Ele 7  │  ...  │ Ele 127 │
└────────┴────────┴────────┴────────┴────────┴────────┴────────┴────────┴───────┴─────────┘
```

-   收集RegTensor中的元素

    根据索引位置indexReg将源操作数srcReg按元素收集到结果dstReg中。收集过程如下图所示：

    <!-- img2text -->
```text
srcReg
┌──────┬──────┬──────┬──────┬──────┬──────┬──────┬──────┐
│ elm0 │ elm1 │ elm2 │ elm3 │ elm4 │ elm5 │ .....│ elmn │
└──────┴──────┴──────┴──────┴──────┴──────┴──────┴──────┘

          ↓      ↓      ↓      ↓      ↓      ↓             ↓
indexReg
┌──────┬──────┬──────┬──────┬──────┬──────┬──────┬──────┐
│  3   │  2   │  4   │  1   │  1   │  0   │ .....│  3   │
└──────┴──────┴──────┴──────┴──────┴──────┴──────┴──────┘
   │      │      │      │      │      │             │
   ↓      ↓      ↓      ↓      ↓      ↓             ↓
dstReg
┌──────┬──────┬──────┬──────┬──────┬──────┬──────┬──────┐
│ elm3 │ elm2 │ elm4 │ elm1 │ elm1 │ elm0 │ .....│ elm3 │
└──────┴──────┴──────┴──────┴──────┴──────┴──────┴──────┘
```

说明:
- indexReg 中每个索引值表示从 srcReg 中取对应位置的元素，写入 dstReg 的对应位置。
- 图中对应关系为：
  - indexReg[0] = 3  → dstReg[0] = elm3
  - indexReg[1] = 2  → dstReg[1] = elm2
  - indexReg[2] = 4  → dstReg[2] = elm4
  - indexReg[3] = 1  → dstReg[3] = elm1
  - indexReg[4] = 1  → dstReg[4] = elm1
  - indexReg[5] = 0  → dstReg[5] = elm0
  - indexReg[n] = 3  → dstReg[n] = elm3

## 定义原型<a name="section620mcpsimp"></a>

-   收集UB中的元素

    ```
    template <typename T0 = DefaultType, typename T1, typename T2 = DefaultType, typename T3, typename T4>
    __simd_callee__ inline void Gather(T3& dstReg, __ubuf__ T1* baseAddr, T4& index, MaskReg& mask)
    ```

-   收集RegTensor中的元素

    ```
    template <typename T = DefaultType, typename U = DefaultType, typename S, typename V>
    __simd_callee__ inline void Gather(S& dstReg, S& srcReg, V& indexReg)
    ```

## 参数说明<a name="section622mcpsimp"></a>

-   收集UB中的元素

    **表 1**  模板参数说明

    <a name="table197504752219"></a>
    <table><thead align="left"><tr id="row1275124715224"><th class="cellrowborder" valign="top" width="18.75%" id="mcps1.2.3.1.1"><p id="p19761647162215"><a name="p19761647162215"></a><a name="p19761647162215"></a>参数名</p>
    </th>
    <th class="cellrowborder" valign="top" width="81.25%" id="mcps1.2.3.1.2"><p id="p07615473221"><a name="p07615473221"></a><a name="p07615473221"></a>描述</p>
    </th>
    </tr>
    </thead>
    <tbody><tr id="row3769472222"><td class="cellrowborder" valign="top" width="18.75%" headers="mcps1.2.3.1.1 "><p id="p157617472226"><a name="p157617472226"></a><a name="p157617472226"></a>T0</p>
    </td>
    <td class="cellrowborder" valign="top" width="81.25%" headers="mcps1.2.3.1.2 "><p id="p207619473226"><a name="p207619473226"></a><a name="p207619473226"></a>目的操作数的数据类型。</p>
    </td>
    </tr>
    <tr id="row1876547162217"><td class="cellrowborder" valign="top" width="18.75%" headers="mcps1.2.3.1.1 "><p id="p107664718221"><a name="p107664718221"></a><a name="p107664718221"></a>T1</p>
    </td>
    <td class="cellrowborder" valign="top" width="81.25%" headers="mcps1.2.3.1.2 "><p id="p187684762214"><a name="p187684762214"></a><a name="p187684762214"></a>源操作数的数据类型。</p>
    </td>
    </tr>
    <tr id="row376647182219"><td class="cellrowborder" valign="top" width="18.75%" headers="mcps1.2.3.1.1 "><p id="p87634716227"><a name="p87634716227"></a><a name="p87634716227"></a>T2</p>
    </td>
    <td class="cellrowborder" valign="top" width="81.25%" headers="mcps1.2.3.1.2 "><p id="p07634718221"><a name="p07634718221"></a><a name="p07634718221"></a>索引值的数据类型。</p>
    </td>
    </tr>
    <tr id="row197614717228"><td class="cellrowborder" valign="top" width="18.75%" headers="mcps1.2.3.1.1 "><p id="p137664732218"><a name="p137664732218"></a><a name="p137664732218"></a>T3</p>
    </td>
    <td class="cellrowborder" valign="top" width="81.25%" headers="mcps1.2.3.1.2 "><p id="p147674782212"><a name="p147674782212"></a><a name="p147674782212"></a>目的操作数的RegTensor类型， 例如RegTensor&lt;half&gt;，由编译器自动推导，用户不需要填写。</p>
    </td>
    </tr>
    <tr id="row276947192212"><td class="cellrowborder" valign="top" width="18.75%" headers="mcps1.2.3.1.1 "><p id="p147604713229"><a name="p147604713229"></a><a name="p147604713229"></a>T4</p>
    </td>
    <td class="cellrowborder" valign="top" width="81.25%" headers="mcps1.2.3.1.2 "><p id="p1276547102217"><a name="p1276547102217"></a><a name="p1276547102217"></a>索引值的RegTensor类型，例如RegTensor&lt;uint16_t&gt;，由编译器自动推导，用户不需要填写。</p>
    </td>
    </tr>
    </tbody>
    </table>

    **表 2**  函数参数说明

    <a name="table1776134722217"></a>
    <table><thead align="left"><tr id="row1876124732213"><th class="cellrowborder" valign="top" width="18.54%" id="mcps1.2.4.1.1"><p id="p6761447142219"><a name="p6761447142219"></a><a name="p6761447142219"></a>参数名</p>
    </th>
    <th class="cellrowborder" valign="top" width="10.05%" id="mcps1.2.4.1.2"><p id="p576247182219"><a name="p576247182219"></a><a name="p576247182219"></a>输入/输出</p>
    </th>
    <th class="cellrowborder" valign="top" width="71.41%" id="mcps1.2.4.1.3"><p id="p276144718229"><a name="p276144718229"></a><a name="p276144718229"></a>描述</p>
    </th>
    </tr>
    </thead>
    <tbody><tr id="row117618477227"><td class="cellrowborder" valign="top" width="18.54%" headers="mcps1.2.4.1.1 "><p id="p1676154792219"><a name="p1676154792219"></a><a name="p1676154792219"></a>dstReg</p>
    </td>
    <td class="cellrowborder" valign="top" width="10.05%" headers="mcps1.2.4.1.2 "><p id="p67614714221"><a name="p67614714221"></a><a name="p67614714221"></a>输出</p>
    </td>
    <td class="cellrowborder" valign="top" width="71.41%" headers="mcps1.2.4.1.3 "><p id="p1761347162214"><a name="p1761347162214"></a><a name="p1761347162214"></a>目的操作数。</p>
    <p id="p776114792213"><a name="p776114792213"></a><a name="p776114792213"></a>类型为<a href="RegTensor.md">RegTensor</a>。</p>
    <p id="p1876047182212"><a name="p1876047182212"></a><a name="p1876047182212"></a>T0数据类型需要与T1和T2数据类型配套使用。类型配套对应表详见约束说明。</p>
    <p id="p127654782211"><a name="p127654782211"></a><a name="p127654782211"></a>支持的数据类型详见<a href="Gather-70.md#table19126103352518">表5</a>。</p>
    </td>
    </tr>
    <tr id="row576347192213"><td class="cellrowborder" valign="top" width="18.54%" headers="mcps1.2.4.1.1 "><p id="p1276147172211"><a name="p1276147172211"></a><a name="p1276147172211"></a>baseAddr</p>
    </td>
    <td class="cellrowborder" valign="top" width="10.05%" headers="mcps1.2.4.1.2 "><p id="p1976184719224"><a name="p1976184719224"></a><a name="p1976184719224"></a>输入</p>
    </td>
    <td class="cellrowborder" valign="top" width="71.41%" headers="mcps1.2.4.1.3 "><p id="p27604712227"><a name="p27604712227"></a><a name="p27604712227"></a>源操作数在UB中的基地址。</p>
    <p id="p176184714229"><a name="p176184714229"></a><a name="p176184714229"></a>类型为UB指针。</p>
    <p id="p17610474226"><a name="p17610474226"></a><a name="p17610474226"></a>T1数据类型需要与T0和T2数据类型配套使用。类型配套对应表详见约束说明。</p>
    <p id="p129111450163"><a name="p129111450163"></a><a name="p129111450163"></a>支持的数据类型详见<a href="Gather-70.md#table19126103352518">表5</a>。</p>
    </td>
    </tr>
    <tr id="row776124722211"><td class="cellrowborder" valign="top" width="18.54%" headers="mcps1.2.4.1.1 "><p id="p67664712224"><a name="p67664712224"></a><a name="p67664712224"></a>index</p>
    </td>
    <td class="cellrowborder" valign="top" width="10.05%" headers="mcps1.2.4.1.2 "><p id="p6761247122219"><a name="p6761247122219"></a><a name="p6761247122219"></a>输入</p>
    </td>
    <td class="cellrowborder" valign="top" width="71.41%" headers="mcps1.2.4.1.3 "><p id="p27619472222"><a name="p27619472222"></a><a name="p27619472222"></a>dstReg中的每个元素在UB中相对于baseAddr的索引位置。地址偏移量要大于等于0。</p>
    <p id="p3761247202218"><a name="p3761247202218"></a><a name="p3761247202218"></a>类型为<a href="RegTensor.md">RegTensor</a>。</p>
    <p id="p47634772219"><a name="p47634772219"></a><a name="p47634772219"></a>T2数据类型需要与T0和T1数据类型配套使用。类型配套对应表详见约束说明。</p>
    <p id="p0358111061613"><a name="p0358111061613"></a><a name="p0358111061613"></a>支持的数据类型详见<a href="Gather-70.md#table19126103352518">表5</a>。</p>
    </td>
    </tr>
    <tr id="row16761647162217"><td class="cellrowborder" valign="top" width="18.54%" headers="mcps1.2.4.1.1 "><p id="p16771047102219"><a name="p16771047102219"></a><a name="p16771047102219"></a>mask</p>
    </td>
    <td class="cellrowborder" valign="top" width="10.05%" headers="mcps1.2.4.1.2 "><p id="p1377247172219"><a name="p1377247172219"></a><a name="p1377247172219"></a>输入</p>
    </td>
    <td class="cellrowborder" valign="top" width="71.41%" headers="mcps1.2.4.1.3 "><p id="p207784717228"><a name="p207784717228"></a><a name="p207784717228"></a>src element操作有效指示，详细说明请参考<a href="MaskReg.md">MaskReg</a>。</p>
    </td>
    </tr>
    </tbody>
    </table>

-   收集RegTensor中的元素

    **表 3**  模板参数说明

    <a name="table99711138233"></a>
    <table><thead align="left"><tr id="row11971013122315"><th class="cellrowborder" valign="top" width="18.13%" id="mcps1.2.3.1.1"><p id="p179718130237"><a name="p179718130237"></a><a name="p179718130237"></a>参数名</p>
    </th>
    <th class="cellrowborder" valign="top" width="81.87%" id="mcps1.2.3.1.2"><p id="p99781311238"><a name="p99781311238"></a><a name="p99781311238"></a>描述</p>
    </th>
    </tr>
    </thead>
    <tbody><tr id="row1997131311232"><td class="cellrowborder" valign="top" width="18.13%" headers="mcps1.2.3.1.1 "><p id="p13971313202319"><a name="p13971313202319"></a><a name="p13971313202319"></a>T</p>
    </td>
    <td class="cellrowborder" valign="top" width="81.87%" headers="mcps1.2.3.1.2 "><p id="p6974136239"><a name="p6974136239"></a><a name="p6974136239"></a>目的操作数和源操作数的数据类型。</p>
    <p id="p153204151019"><a name="p153204151019"></a><a name="p153204151019"></a><span id="ph133209151609"><a name="ph133209151609"></a><a name="ph133209151609"></a>Ascend 950PR/Ascend 950DT</span>，支持的数据类型为：b8/b16/b32</p>
    </td>
    </tr>
    <tr id="row145134204158"><td class="cellrowborder" valign="top" width="18.13%" headers="mcps1.2.3.1.1 "><p id="p17514172015155"><a name="p17514172015155"></a><a name="p17514172015155"></a>U</p>
    </td>
    <td class="cellrowborder" valign="top" width="81.87%" headers="mcps1.2.3.1.2 "><p id="p9514132013155"><a name="p9514132013155"></a><a name="p9514132013155"></a>索引值的数据类型。</p>
    <p id="p1332134951510"><a name="p1332134951510"></a><a name="p1332134951510"></a><span id="ph63211249121510"><a name="ph63211249121510"></a><a name="ph63211249121510"></a>Ascend 950PR/Ascend 950DT</span>，支持的数据类型为：uint8_t/uint16_t/uint32_t</p>
    </td>
    </tr>
    <tr id="row3789349195116"><td class="cellrowborder" valign="top" width="18.13%" headers="mcps1.2.3.1.1 "><p id="p578954925110"><a name="p578954925110"></a><a name="p578954925110"></a>S</p>
    </td>
    <td class="cellrowborder" valign="top" width="81.87%" headers="mcps1.2.3.1.2 "><p id="p137898497514"><a name="p137898497514"></a><a name="p137898497514"></a><span id="ph19851723182011"><a name="ph19851723182011"></a><a name="ph19851723182011"></a>目的操作数的RegTensor类型，例如RegTensor&lt;half&gt;，由编译器自动推导，用户不需要填写。</span></p>
    </td>
    </tr>
    <tr id="row13194205214516"><td class="cellrowborder" valign="top" width="18.13%" headers="mcps1.2.3.1.1 "><p id="p2194152115116"><a name="p2194152115116"></a><a name="p2194152115116"></a>V</p>
    </td>
    <td class="cellrowborder" valign="top" width="81.87%" headers="mcps1.2.3.1.2 "><p id="p719455205119"><a name="p719455205119"></a><a name="p719455205119"></a>索引值的RegTensor类型，例如RegTensor&lt;uint16_t&gt;，由编译器自动推导，用户不需要填写。</p>
    </td>
    </tr>
    </tbody>
    </table>

    **表 4**  函数参数说明

    <a name="table29781322318"></a>
    <table><thead align="left"><tr id="row39721320234"><th class="cellrowborder" valign="top" width="18.54%" id="mcps1.2.4.1.1"><p id="p189714131230"><a name="p189714131230"></a><a name="p189714131230"></a>参数名</p>
    </th>
    <th class="cellrowborder" valign="top" width="10.05%" id="mcps1.2.4.1.2"><p id="p597313172319"><a name="p597313172319"></a><a name="p597313172319"></a>输入/输出</p>
    </th>
    <th class="cellrowborder" valign="top" width="71.41%" id="mcps1.2.4.1.3"><p id="p497141317238"><a name="p497141317238"></a><a name="p497141317238"></a>描述</p>
    </th>
    </tr>
    </thead>
    <tbody><tr id="row1397213102318"><td class="cellrowborder" valign="top" width="18.54%" headers="mcps1.2.4.1.1 "><p id="p1435142552019"><a name="p1435142552019"></a><a name="p1435142552019"></a>dstReg</p>
    </td>
    <td class="cellrowborder" valign="top" width="10.05%" headers="mcps1.2.4.1.2 "><p id="p119701342314"><a name="p119701342314"></a><a name="p119701342314"></a>输出</p>
    </td>
    <td class="cellrowborder" valign="top" width="71.41%" headers="mcps1.2.4.1.3 "><p id="p212236122019"><a name="p212236122019"></a><a name="p212236122019"></a>目的操作数。</p>
    <p id="p2507123615484"><a name="p2507123615484"></a><a name="p2507123615484"></a><span id="ph950733612488"><a name="ph950733612488"></a><a name="ph950733612488"></a>类型为<a href="RegTensor.md">RegTensor</a>。</span></p>
    </td>
    </tr>
    <tr id="row497613202314"><td class="cellrowborder" valign="top" width="18.54%" headers="mcps1.2.4.1.1 "><p id="p99751332311"><a name="p99751332311"></a><a name="p99751332311"></a>srcReg</p>
    </td>
    <td class="cellrowborder" valign="top" width="10.05%" headers="mcps1.2.4.1.2 "><p id="p139818138232"><a name="p139818138232"></a><a name="p139818138232"></a>输入</p>
    </td>
    <td class="cellrowborder" valign="top" width="71.41%" headers="mcps1.2.4.1.3 "><p id="p14827161917207"><a name="p14827161917207"></a><a name="p14827161917207"></a>源操作数。</p>
    <p id="p7123111612517"><a name="p7123111612517"></a><a name="p7123111612517"></a><span id="ph1197863231916"><a name="ph1197863231916"></a><a name="ph1197863231916"></a>类型为<a href="RegTensor.md">RegTensor</a>。</span></p>
    <p id="p16987134230"><a name="p16987134230"></a><a name="p16987134230"></a>数据类型需要与目的操作数保持一致。</p>
    </td>
    </tr>
    <tr id="row79813133234"><td class="cellrowborder" valign="top" width="18.54%" headers="mcps1.2.4.1.1 "><p id="p11982138234"><a name="p11982138234"></a><a name="p11982138234"></a>indexReg</p>
    </td>
    <td class="cellrowborder" valign="top" width="10.05%" headers="mcps1.2.4.1.2 "><p id="p19811362320"><a name="p19811362320"></a><a name="p19811362320"></a>输入</p>
    </td>
    <td class="cellrowborder" valign="top" width="71.41%" headers="mcps1.2.4.1.3 "><p id="p6981713112315"><a name="p6981713112315"></a><a name="p6981713112315"></a>数据索引。</p>
    <p id="p1728583820487"><a name="p1728583820487"></a><a name="p1728583820487"></a><span id="ph72855386487"><a name="ph72855386487"></a><a name="ph72855386487"></a>类型为<a href="RegTensor.md">RegTensor</a>。</span></p>
    <p id="p104218390187"><a name="p104218390187"></a><a name="p104218390187"></a>数据类型的位宽需要与目的操作数的位宽保持一致。</p>
    <p id="p1396715519200"><a name="p1396715519200"></a><a name="p1396715519200"></a>srcReg为RegTensor类型，位宽是固定的VL，存储的元素个数固定。如果indexReg中索引值超出当前RegTensor中能存储的最大数据元素个数时，按照如下方式处理：设定当前RegTensor所能存储的最大数据元素个数为vlLength，indexReg中索引值为i，索引值更新为i % vlLength。</p>
    </td>
    </tr>
    </tbody>
    </table>

## 约束说明<a name="section177921451558"></a>

-   收集UB中的元素
    -   T0，T1和T2数据类型需要配套使用。配套关系如下表：

        **表 5**  Gather操作数数据类型对应表

        <a name="table19126103352518"></a>
        <table><thead align="left"><tr id="row131261133202514"><th class="cellrowborder" valign="top" width="33.3033303330333%" id="mcps1.2.4.1.1"><p id="p61269332250"><a name="p61269332250"></a><a name="p61269332250"></a><strong id="b12126143362515"><a name="b12126143362515"></a><a name="b12126143362515"></a>T0数据类型</strong></p>
        </th>
        <th class="cellrowborder" valign="top" width="33.36333633363336%" id="mcps1.2.4.1.2"><p id="p81261332258"><a name="p81261332258"></a><a name="p81261332258"></a><strong id="b012613332512"><a name="b012613332512"></a><a name="b012613332512"></a>T1数据类型</strong></p>
        </th>
        <th class="cellrowborder" valign="top" width="33.33333333333333%" id="mcps1.2.4.1.3"><p id="p181262333257"><a name="p181262333257"></a><a name="p181262333257"></a><strong id="b2127233202518"><a name="b2127233202518"></a><a name="b2127233202518"></a>T2数据类型</strong></p>
        </th>
        </tr>
        </thead>
        <tbody><tr id="row131271933182515"><td class="cellrowborder" valign="top" width="33.3033303330333%" headers="mcps1.2.4.1.1 "><p id="p111271233152515"><a name="p111271233152515"></a><a name="p111271233152515"></a>int16_t</p>
        </td>
        <td class="cellrowborder" valign="top" width="33.36333633363336%" headers="mcps1.2.4.1.2 "><p id="p151272333258"><a name="p151272333258"></a><a name="p151272333258"></a>int8_t</p>
        </td>
        <td class="cellrowborder" rowspan="6" valign="top" width="33.33333333333333%" headers="mcps1.2.4.1.3 "><p id="p1612719336259"><a name="p1612719336259"></a><a name="p1612719336259"></a>uint16_t</p>
        </td>
        </tr>
        <tr id="row181275330251"><td class="cellrowborder" valign="top" headers="mcps1.2.4.1.1 "><p id="p712743319254"><a name="p712743319254"></a><a name="p712743319254"></a>uint16_t</p>
        </td>
        <td class="cellrowborder" valign="top" headers="mcps1.2.4.1.2 "><p id="p141271133102513"><a name="p141271133102513"></a><a name="p141271133102513"></a>uint8_t</p>
        </td>
        </tr>
        <tr id="row1012783332510"><td class="cellrowborder" valign="top" headers="mcps1.2.4.1.1 "><p id="p181277334255"><a name="p181277334255"></a><a name="p181277334255"></a>int16_t</p>
        </td>
        <td class="cellrowborder" valign="top" headers="mcps1.2.4.1.2 "><p id="p3127433112512"><a name="p3127433112512"></a><a name="p3127433112512"></a>int16_t</p>
        </td>
        </tr>
        <tr id="row171271033102520"><td class="cellrowborder" valign="top" headers="mcps1.2.4.1.1 "><p id="p101273337252"><a name="p101273337252"></a><a name="p101273337252"></a>uint16_t</p>
        </td>
        <td class="cellrowborder" valign="top" headers="mcps1.2.4.1.2 "><p id="p412717333252"><a name="p412717333252"></a><a name="p412717333252"></a>uint16_t</p>
        </td>
        </tr>
        <tr id="row91272339259"><td class="cellrowborder" valign="top" headers="mcps1.2.4.1.1 "><p id="p7127153315254"><a name="p7127153315254"></a><a name="p7127153315254"></a>half</p>
        </td>
        <td class="cellrowborder" valign="top" headers="mcps1.2.4.1.2 "><p id="p412783320256"><a name="p412783320256"></a><a name="p412783320256"></a>half</p>
        </td>
        </tr>
        <tr id="row8127233162515"><td class="cellrowborder" valign="top" headers="mcps1.2.4.1.1 "><p id="p712753313257"><a name="p712753313257"></a><a name="p712753313257"></a>bfloat16_t</p>
        </td>
        <td class="cellrowborder" valign="top" headers="mcps1.2.4.1.2 "><p id="p1212763313253"><a name="p1212763313253"></a><a name="p1212763313253"></a>bfloat16_t</p>
        </td>
        </tr>
        <tr id="row10127123352517"><td class="cellrowborder" valign="top" width="33.3033303330333%" headers="mcps1.2.4.1.1 "><p id="p71271334250"><a name="p71271334250"></a><a name="p71271334250"></a>int32_t</p>
        </td>
        <td class="cellrowborder" valign="top" width="33.36333633363336%" headers="mcps1.2.4.1.2 "><p id="p81271933122512"><a name="p81271933122512"></a><a name="p81271933122512"></a>int32_t</p>
        </td>
        <td class="cellrowborder" rowspan="3" valign="top" width="33.33333333333333%" headers="mcps1.2.4.1.3 "><p id="p212703372511"><a name="p212703372511"></a><a name="p212703372511"></a>uint32_t</p>
        </td>
        </tr>
        <tr id="row18127113392514"><td class="cellrowborder" valign="top" headers="mcps1.2.4.1.1 "><p id="p17127933132514"><a name="p17127933132514"></a><a name="p17127933132514"></a>uint32_t</p>
        </td>
        <td class="cellrowborder" valign="top" headers="mcps1.2.4.1.2 "><p id="p1412763314259"><a name="p1412763314259"></a><a name="p1412763314259"></a>uint32_t</p>
        </td>
        </tr>
        <tr id="row1612783320257"><td class="cellrowborder" valign="top" headers="mcps1.2.4.1.1 "><p id="p14127173382517"><a name="p14127173382517"></a><a name="p14127173382517"></a>float</p>
        </td>
        <td class="cellrowborder" valign="top" headers="mcps1.2.4.1.2 "><p id="p17127173320259"><a name="p17127173320259"></a><a name="p17127173320259"></a>float</p>
        </td>
        </tr>
        <tr id="row5127173352517"><td class="cellrowborder" valign="top" width="33.3033303330333%" headers="mcps1.2.4.1.1 "><p id="p91273333256"><a name="p91273333256"></a><a name="p91273333256"></a>uint64_t</p>
        </td>
        <td class="cellrowborder" valign="top" width="33.36333633363336%" headers="mcps1.2.4.1.2 "><p id="p141276333255"><a name="p141276333255"></a><a name="p141276333255"></a>uint64_t</p>
        </td>
        <td class="cellrowborder" rowspan="2" valign="top" width="33.33333333333333%" headers="mcps1.2.4.1.3 "><p id="p16127833182514"><a name="p16127833182514"></a><a name="p16127833182514"></a>uint32_t</p>
        </td>
        </tr>
        <tr id="row19127933102512"><td class="cellrowborder" valign="top" headers="mcps1.2.4.1.1 "><p id="p412743322514"><a name="p412743322514"></a><a name="p412743322514"></a>int64_t</p>
        </td>
        <td class="cellrowborder" valign="top" headers="mcps1.2.4.1.2 "><p id="p15127103312515"><a name="p15127103312515"></a><a name="p15127103312515"></a>int64_t</p>
        </td>
        </tr>
        <tr id="row312711332250"><td class="cellrowborder" valign="top" width="33.3033303330333%" headers="mcps1.2.4.1.1 "><p id="p10127123313258"><a name="p10127123313258"></a><a name="p10127123313258"></a>uint64_t</p>
        </td>
        <td class="cellrowborder" valign="top" width="33.36333633363336%" headers="mcps1.2.4.1.2 "><p id="p111271633162515"><a name="p111271633162515"></a><a name="p111271633162515"></a>uint64_t</p>
        </td>
        <td class="cellrowborder" rowspan="2" valign="top" width="33.33333333333333%" headers="mcps1.2.4.1.3 "><p id="p191272333257"><a name="p191272333257"></a><a name="p191272333257"></a>uint64_t</p>
        </td>
        </tr>
        <tr id="row01275337259"><td class="cellrowborder" valign="top" headers="mcps1.2.4.1.1 "><p id="p1512753313254"><a name="p1512753313254"></a><a name="p1512753313254"></a>int64_t</p>
        </td>
        <td class="cellrowborder" valign="top" headers="mcps1.2.4.1.2 "><p id="p20127833102517"><a name="p20127833102517"></a><a name="p20127833102517"></a>int64_t</p>
        </td>
        </tr>
        </tbody>
        </table>

    -   当T1为B8数据类型时，T0为B16类型，这种情况下目的操作数的低8位与源操作数相同，高8位自动补0。例如T1为int8数据类型：

        40=0b00101000 -\> 0b0000000000101000，扩充至16位后等于40；

        -40=0b11011000 -\> 0b0000000011011000，扩充至16位后等于216。

    -   当T1为B64数据类型时，T0，T1，T2，T4, T3数据类型只支持以下组合:

        <a name="table71281336252"></a>
        <table><thead align="left"><tr id="row6128533162515"><th class="cellrowborder" valign="top" width="8.94089408940894%" id="mcps1.1.7.1.1"><p id="p512873313252"><a name="p512873313252"></a><a name="p512873313252"></a><strong id="b161286332255"><a name="b161286332255"></a><a name="b161286332255"></a>T0数据类型</strong></p>
        </th>
        <th class="cellrowborder" valign="top" width="10.181018101810182%" id="mcps1.1.7.1.2"><p id="p131281433202511"><a name="p131281433202511"></a><a name="p131281433202511"></a><strong id="b51281633122513"><a name="b51281633122513"></a><a name="b51281633122513"></a>T1数据类型</strong></p>
        </th>
        <th class="cellrowborder" valign="top" width="11.18111811181118%" id="mcps1.1.7.1.3"><p id="p1812853313251"><a name="p1812853313251"></a><a name="p1812853313251"></a><strong id="b1512863392512"><a name="b1512863392512"></a><a name="b1512863392512"></a>T2数据类型</strong></p>
        </th>
        <th class="cellrowborder" valign="top" width="23.812381238123812%" id="mcps1.1.7.1.4"><p id="p111286338253"><a name="p111286338253"></a><a name="p111286338253"></a>T4数据类型</p>
        </th>
        <th class="cellrowborder" valign="top" width="34.003400340034005%" id="mcps1.1.7.1.5"><p id="p11285333255"><a name="p11285333255"></a><a name="p11285333255"></a>T3数据类型</p>
        </th>
        <th class="cellrowborder" valign="top" width="11.881188118811883%" id="mcps1.1.7.1.6"><p id="p9128183352513"><a name="p9128183352513"></a><a name="p9128183352513"></a>备注</p>
        </th>
        </tr>
        </thead>
        <tbody><tr id="row112811338257"><td class="cellrowborder" rowspan="4" valign="top" width="8.94089408940894%" headers="mcps1.1.7.1.1 "><p id="p9128733162519"><a name="p9128733162519"></a><a name="p9128733162519"></a>B64</p>
        </td>
        <td class="cellrowborder" rowspan="4" valign="top" width="10.181018101810182%" headers="mcps1.1.7.1.2 "><p id="p11281433132511"><a name="p11281433132511"></a><a name="p11281433132511"></a>B64</p>
        </td>
        <td class="cellrowborder" rowspan="4" valign="top" width="11.18111811181118%" headers="mcps1.1.7.1.3 "><p id="p14128133192517"><a name="p14128133192517"></a><a name="p14128133192517"></a>uint32_t</p>
        </td>
        <td class="cellrowborder" rowspan="4" valign="top" width="23.812381238123812%" headers="mcps1.1.7.1.4 "><p id="p612816335257"><a name="p612816335257"></a><a name="p612816335257"></a>RegTensor&lt;uint32_t&gt;</p>
        </td>
        <td class="cellrowborder" valign="top" width="34.003400340034005%" headers="mcps1.1.7.1.5 "><p id="p16128183314251"><a name="p16128183314251"></a><a name="p16128183314251"></a>RegTensor&lt;uint64_t, RegTraitNumOne&gt;</p>
        </td>
        <td class="cellrowborder" rowspan="2" valign="top" width="11.881188118811883%" headers="mcps1.1.7.1.6 "><p id="p8128833172515"><a name="p8128833172515"></a><a name="p8128833172515"></a>index的前32个数有效</p>
        </td>
        </tr>
        <tr id="row412814338258"><td class="cellrowborder" valign="top" headers="mcps1.1.7.1.1 "><p id="p13128333122513"><a name="p13128333122513"></a><a name="p13128333122513"></a>RegTensor&lt;int64_t, RegTraitNumOne&gt;</p>
        </td>
        </tr>
        <tr id="row1212853310254"><td class="cellrowborder" valign="top" headers="mcps1.1.7.1.1 "><p id="p2128633142512"><a name="p2128633142512"></a><a name="p2128633142512"></a>RegTensor&lt;uint64_t, RegTraitNumTwo&gt;</p>
        </td>
        <td class="cellrowborder" rowspan="2" valign="top" headers="mcps1.1.7.1.2 "><p id="p412813338251"><a name="p412813338251"></a><a name="p412813338251"></a>-</p>
        </td>
        </tr>
        <tr id="row512813314253"><td class="cellrowborder" valign="top" headers="mcps1.1.7.1.1 "><p id="p8128133132514"><a name="p8128133132514"></a><a name="p8128133132514"></a>RegTensor&lt;int64_t, RegTraitNumTwo&gt;</p>
        </td>
        </tr>
        <tr id="row1412815339255"><td class="cellrowborder" rowspan="6" valign="top" width="8.94089408940894%" headers="mcps1.1.7.1.1 "><p id="p141284333258"><a name="p141284333258"></a><a name="p141284333258"></a>B64</p>
        </td>
        <td class="cellrowborder" rowspan="6" valign="top" width="10.181018101810182%" headers="mcps1.1.7.1.2 "><p id="p91281333122519"><a name="p91281333122519"></a><a name="p91281333122519"></a>B64</p>
        </td>
        <td class="cellrowborder" rowspan="6" valign="top" width="11.18111811181118%" headers="mcps1.1.7.1.3 "><p id="p11128933152519"><a name="p11128933152519"></a><a name="p11128933152519"></a>uint64_t</p>
        </td>
        <td class="cellrowborder" rowspan="2" valign="top" width="23.812381238123812%" headers="mcps1.1.7.1.4 "><p id="p19128163342518"><a name="p19128163342518"></a><a name="p19128163342518"></a>RegTensor&lt;uint64_t, RegTraitNumOne&gt;</p>
        </td>
        <td class="cellrowborder" valign="top" width="34.003400340034005%" headers="mcps1.1.7.1.5 "><p id="p512893315250"><a name="p512893315250"></a><a name="p512893315250"></a>RegTensor&lt;uint64_t, RegTraitNumOne&gt;</p>
        </td>
        <td class="cellrowborder" rowspan="2" valign="top" width="11.881188118811883%" headers="mcps1.1.7.1.6 "><p id="p13128183312511"><a name="p13128183312511"></a><a name="p13128183312511"></a>-</p>
        </td>
        </tr>
        <tr id="row141281833102512"><td class="cellrowborder" valign="top" headers="mcps1.1.7.1.1 "><p id="p81287334252"><a name="p81287334252"></a><a name="p81287334252"></a>RegTensor&lt;int64_t, RegTraitNumOne&gt;</p>
        </td>
        </tr>
        <tr id="row11289333253"><td class="cellrowborder" rowspan="4" valign="top" headers="mcps1.1.7.1.1 "><p id="p1712813336254"><a name="p1712813336254"></a><a name="p1712813336254"></a>RegTensor&lt;uint64_t, RegTraitNumTwo&gt;</p>
        </td>
        <td class="cellrowborder" valign="top" headers="mcps1.1.7.1.2 "><p id="p18128203382519"><a name="p18128203382519"></a><a name="p18128203382519"></a>RegTensor&lt;uint64_t, RegTraitNumOne&gt;</p>
        </td>
        <td class="cellrowborder" rowspan="2" valign="top" headers="mcps1.1.7.1.3 "><p id="p112813338255"><a name="p112813338255"></a><a name="p112813338255"></a>index的前32个数有效</p>
        </td>
        </tr>
        <tr id="row1412893312510"><td class="cellrowborder" valign="top" headers="mcps1.1.7.1.1 "><p id="p1112813322515"><a name="p1112813322515"></a><a name="p1112813322515"></a>RegTensor&lt;int64_t, RegTraitNumOne&gt;</p>
        </td>
        </tr>
        <tr id="row16128183352515"><td class="cellrowborder" valign="top" headers="mcps1.1.7.1.1 "><p id="p14129183352518"><a name="p14129183352518"></a><a name="p14129183352518"></a>RegTensor&lt;uint64_t, RegTraitNumTwo&gt;</p>
        </td>
        <td class="cellrowborder" rowspan="2" valign="top" headers="mcps1.1.7.1.2 "><p id="p16129163322516"><a name="p16129163322516"></a><a name="p16129163322516"></a>-</p>
        </td>
        </tr>
        <tr id="row212953313258"><td class="cellrowborder" valign="top" headers="mcps1.1.7.1.1 "><p id="p1212983342513"><a name="p1212983342513"></a><a name="p1212983342513"></a>RegTensor&lt;int64_t, RegTraitNumTwo&gt;</p>
        </td>
        </tr>
        </tbody>
        </table>

-   收集RegTensor中的元素

    无约束

## 调用示例<a name="section642mcpsimp"></a>

-   收集UB中的元素

    ```
    template <typename T, typename U>
    __simd_vf__ inline void GatherVF(__ubuf__ T* dstAddr, __ubuf__ T* src0Addr, __ubuf__ U* src1Addr, uint32_t count, uint32_t oneRepeatSize, uint16_t repeatTimes)
    {
        AscendC::MicroAPI::RegTensor<U> srcReg;
        AscendC::MicroAPI::RegTensor<T> dstReg;
        AscendC::MicroAPI::MaskReg mask;    
        for (uint16_t i = 0; i < repeatTimes; i++) {
            mask = AscendC::MicroAPI::UpdateMask<T>(count);
            AscendC::MicroAPI::LoadAlign(srcReg, src1Addr + i * oneRepeatSize);
            AscendC::MicroAPI::Gather(dstReg, src0Addr, srcReg, mask);
            AscendC::MicroAPI::StoreAlign(dstAddr + i * oneRepeatSize, dstReg, mask);
        }
    }
    ```

-   收集RegTensor中的元素

    ```
    template <typename T, typename U>
    __simd_vf__ inline void GatherVF(__ubuf__ T* dstAddr, __ubuf__ T* src0Addr, __ubuf__ U* src1Addr, uint32_t count, uint32_t oneRepeatSize, uint16_t repeatTimes)
    {
        AscendC::MicroAPI::RegTensor<T> srcReg0, dstReg;
        AscendC::MicroAPI::RegTensor<U> srcReg1;
        AscendC::MicroAPI::MaskReg mask;
        AscendC::MicroAPI::LoadAlign(srcReg1, src1Addr);
        for (uint16_t i = 0; i < repeatTimes; i++) {
            mask = AscendC::MicroAPI::UpdateMask<T>(count);
            AscendC::MicroAPI::LoadAlign(srcReg0, src0Addr + i * oneRepeatSize);
            AscendC::MicroAPI::Gather(dstReg, srcReg0, srcReg1);
            AscendC::MicroAPI::StoreAlign(dstAddr + i * oneRepeatSize, dstReg, mask);
        }
    }
    ```

