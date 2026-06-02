# Scatter<a name="ZH-CN_TOPIC_0000002523304628"></a>

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

给定源操作数的寄存器张量和索引张量，以及结果操作数在UB中的基地址，Scatter指令将源操作数按元素根据索引位置分散到UB中。分散过程如下图所示：

<!-- img2text -->
```text
Src
┌─────────┬─────────┬─────────┬─────────┬─────────┬─────────┬─────────┬─────────┬───────┬─────────┐
│  Ele 0  │  Ele 1  │  Ele 2  │  Ele 3  │  Ele 4  │  Ele 5  │  Ele 6  │  Ele 7  │  ...  │ Ele 127 │
└────┬────┴────┬────┴────┬────┴────┬────┴────┬────┴────┬────┴────┬────┴────┬────┴───┴────┬────┘
     │         │         │         │         │         │         │         │             │
     ▼         ▼         ▼         ▼         ▼         ▼         ▼         ▼             ▼

Index
┌─────────┬─────────┬─────────┬─────────┬─────────┬─────────┬─────────┬─────────┬───────┬─────────┐
│ Index 0 │ Index 1 │ Index 2 │ Index 3 │ Index 4 │ Index 5 │ Index 6 │ Index 7 │  ...  │Index 127│
└────┬────┴────┬────┴────┬────┴────┬────┴────┬────┴────┬────┴────┬────┴────┬────┴───┴────┬────┘
     │         │         │         │         │         │         │         │             │
     │         │         │         │         │         │         │         │             │
     └───────────────┐   │   ┌───────────────┘         │         │         │             │
                     │   │   │                         │         │         │             │
                     ▼   ▼   ▼                         ▼         ▼         ▼             ▼

                                              Scatter
                                                 ↓

Dst
┌──────────────────────────────────────────────────────────────────────────────────────────────────────┐
│                                                  UB空间                                              │
│                                                                                                      │
│        ┌─────────┐             ┌─────────┐             ┌─────────┐             ┌─────────┐          │
│        │  Ele 2  │             │  Ele 0  │             │  Ele 3  │             │  Ele 6  │          │
│        └─────────┘             └─────────┘             └─────────┘             └─────────┘          │
│                                                                                          ┌─────────┐ │
│                                                                                          │... Ele127│ │
│                                                                                          └─────────┘ │
│                                                                                                      │
│        ┌─────────┐             ┌─────────┐             ┌─────────┐             ┌─────────┐          │
│        │  Ele 4  │             │  Ele 1  │             │  Ele 5  │             │  Ele 7  │          │
│        └─────────┘             └─────────┘             └─────────┘             └─────────┘          │
└──────────────────────────────────────────────────────────────────────────────────────────────────────┘

映射关系:
Ele 0   → Dst中的 Ele 0 位置
Ele 1   → Dst中的 Ele 1 位置
Ele 2   → Dst中的 Ele 2 位置
Ele 3   → Dst中的 Ele 3 位置
Ele 4   → Dst中的 Ele 4 位置
Ele 5   → Dst中的 Ele 5 位置
Ele 6   → Dst中的 Ele 6 位置
Ele 7   → Dst中的 Ele 7 位置
Ele 127 → Dst中的 ... Ele 127 位置
```

## 定义原型<a name="section620mcpsimp"></a>

```
template <typename T = DefaultType, typename U = DefaultType, typename S, typename V>
__simd_callee__ inline void Scatter(__ubuf__ T* baseAddr, S& srcReg, V& index, MaskReg& mask)
```

## 参数说明<a name="section622mcpsimp"></a>

**表 1**  模板参数说明

<a name="table4835205712588"></a>
<table><thead align="left"><tr id="row118356578583"><th class="cellrowborder" valign="top" width="50%" id="mcps1.2.3.1.1"><p id="p48354572582"><a name="p48354572582"></a><a name="p48354572582"></a>参数名</p>
</th>
<th class="cellrowborder" valign="top" width="50%" id="mcps1.2.3.1.2"><p id="p583535795817"><a name="p583535795817"></a><a name="p583535795817"></a>描述</p>
</th>
</tr>
</thead>
<tbody><tr id="row1835857145817"><td class="cellrowborder" valign="top" width="50%" headers="mcps1.2.3.1.1 "><p id="p5835457165816"><a name="p5835457165816"></a><a name="p5835457165816"></a>T</p>
</td>
<td class="cellrowborder" valign="top" width="50%" headers="mcps1.2.3.1.2 "><p id="p168351657155818"><a name="p168351657155818"></a><a name="p168351657155818"></a>目的操作数和源操作数的数据类型。</p>
</td>
</tr>
<tr id="row992173775113"><td class="cellrowborder" valign="top" width="50%" headers="mcps1.2.3.1.1 "><p id="p169263775115"><a name="p169263775115"></a><a name="p169263775115"></a>U</p>
</td>
<td class="cellrowborder" valign="top" width="50%" headers="mcps1.2.3.1.2 "><p id="p8929370512"><a name="p8929370512"></a><a name="p8929370512"></a>索引的数据类型。</p>
</td>
</tr>
<tr id="row3789349195116"><td class="cellrowborder" valign="top" width="50%" headers="mcps1.2.3.1.1 "><p id="p578954925110"><a name="p578954925110"></a><a name="p578954925110"></a>S</p>
</td>
<td class="cellrowborder" valign="top" width="50%" headers="mcps1.2.3.1.2 "><p id="p137898497514"><a name="p137898497514"></a><a name="p137898497514"></a>源操作数的RegTensor类型，例如RegTensor&lt;half&gt;，由编译器自动推导，用户不需要填写。</p>
</td>
</tr>
<tr id="row13194205214516"><td class="cellrowborder" valign="top" width="50%" headers="mcps1.2.3.1.1 "><p id="p2194152115116"><a name="p2194152115116"></a><a name="p2194152115116"></a>V</p>
</td>
<td class="cellrowborder" valign="top" width="50%" headers="mcps1.2.3.1.2 "><p id="p719455205119"><a name="p719455205119"></a><a name="p719455205119"></a>索引值的RegTensor类型，例如RegTensor&lt;uint16_t&gt;，由编译器自动推导，用户不需要填写。</p>
</td>
</tr>
</tbody>
</table>

**表 2**  函数参数说明

<a name="zh-cn_topic_0235751031_table33761356"></a>
<table><thead align="left"><tr id="zh-cn_topic_0235751031_row27598891"><th class="cellrowborder" valign="top" width="18.54%" id="mcps1.2.4.1.1"><p id="zh-cn_topic_0235751031_p20917673"><a name="zh-cn_topic_0235751031_p20917673"></a><a name="zh-cn_topic_0235751031_p20917673"></a>参数名</p>
</th>
<th class="cellrowborder" valign="top" width="10.05%" id="mcps1.2.4.1.2"><p id="zh-cn_topic_0235751031_p16609919"><a name="zh-cn_topic_0235751031_p16609919"></a><a name="zh-cn_topic_0235751031_p16609919"></a>输入/输出</p>
</th>
<th class="cellrowborder" valign="top" width="71.41%" id="mcps1.2.4.1.3"><p id="zh-cn_topic_0235751031_p59995477"><a name="zh-cn_topic_0235751031_p59995477"></a><a name="zh-cn_topic_0235751031_p59995477"></a>描述</p>
</th>
</tr>
</thead>
<tbody><tr id="row42461942101815"><td class="cellrowborder" valign="top" width="18.54%" headers="mcps1.2.4.1.1 "><p id="p284425844311"><a name="p284425844311"></a><a name="p284425844311"></a>baseAddr</p>
</td>
<td class="cellrowborder" valign="top" width="10.05%" headers="mcps1.2.4.1.2 "><p id="p158449584436"><a name="p158449584436"></a><a name="p158449584436"></a>输出</p>
</td>
<td class="cellrowborder" valign="top" width="71.41%" headers="mcps1.2.4.1.3 "><p id="p6389114616514"><a name="p6389114616514"></a><a name="p6389114616514"></a>目的操作数在UB中的基地址。</p>
<p id="p1818424418593"><a name="p1818424418593"></a><a name="p1818424418593"></a>类型为UB指针。</p>
<p id="p166112819388"><a name="p166112819388"></a><a name="p166112819388"></a><span id="ph126122820385"><a name="ph126122820385"></a><a name="ph126122820385"></a>Ascend 950PR/Ascend 950DT</span>，支持的数据类型详见<a href="#table113383813355">表3</a>。</p>
</td>
</tr>
<tr id="row2137145181815"><td class="cellrowborder" valign="top" width="18.54%" headers="mcps1.2.4.1.1 "><p id="p6844125874315"><a name="p6844125874315"></a><a name="p6844125874315"></a>srcReg</p>
</td>
<td class="cellrowborder" valign="top" width="10.05%" headers="mcps1.2.4.1.2 "><p id="p128442058144312"><a name="p128442058144312"></a><a name="p128442058144312"></a>输入</p>
</td>
<td class="cellrowborder" valign="top" width="71.41%" headers="mcps1.2.4.1.3 "><p id="p172083541517"><a name="p172083541517"></a><a name="p172083541517"></a>源操作数。</p>
<p id="p2051111482068"><a name="p2051111482068"></a><a name="p2051111482068"></a>类型为RegTensor。</p>
<p id="p1816018523328"><a name="p1816018523328"></a><a name="p1816018523328"></a><span id="ph16160115263218"><a name="ph16160115263218"></a><a name="ph16160115263218"></a>Ascend 950PR/Ascend 950DT</span>，支持的数据类型详见<a href="#table113383813355">表3</a>。</p>
</td>
</tr>
<tr id="row891912431168"><td class="cellrowborder" valign="top" width="18.54%" headers="mcps1.2.4.1.1 "><p id="p195756503168"><a name="p195756503168"></a><a name="p195756503168"></a>index</p>
</td>
<td class="cellrowborder" valign="top" width="10.05%" headers="mcps1.2.4.1.2 "><p id="p457515071618"><a name="p457515071618"></a><a name="p457515071618"></a>输入</p>
</td>
<td class="cellrowborder" valign="top" width="71.41%" headers="mcps1.2.4.1.3 "><p id="p193020233012"><a name="p193020233012"></a><a name="p193020233012"></a>srcReg中的每个元素在UB中相对于baseAddr的索引位置。索引值要大于等于0。</p>
<p id="p081418386319"><a name="p081418386319"></a><a name="p081418386319"></a>类型为RegTensor。</p>
<p id="p3505185872619"><a name="p3505185872619"></a><a name="p3505185872619"></a>IndexT数据类型需要与目的操作数和源操作数的数据类型T配套使用。类型配套对应表详见约束说明。</p>
<p id="p7696194510333"><a name="p7696194510333"></a><a name="p7696194510333"></a><span id="ph4696124513320"><a name="ph4696124513320"></a><a name="ph4696124513320"></a>Ascend 950PR/Ascend 950DT</span>，支持的数据类型详见<a href="#table113383813355">表3</a>。</p>
</td>
</tr>
<tr id="row19615183817191"><td class="cellrowborder" valign="top" width="18.54%" headers="mcps1.2.4.1.1 "><p id="p1484519586432"><a name="p1484519586432"></a><a name="p1484519586432"></a>mask</p>
</td>
<td class="cellrowborder" valign="top" width="10.05%" headers="mcps1.2.4.1.2 "><p id="p484514581433"><a name="p484514581433"></a><a name="p484514581433"></a>输入</p>
</td>
<td class="cellrowborder" valign="top" width="71.41%" headers="mcps1.2.4.1.3 "><p id="p1559991025517"><a name="p1559991025517"></a><a name="p1559991025517"></a>src element操作有效指示，详细说明请参考<a href="MaskReg.md">MaskReg</a>。</p>
</td>
</tr>
</tbody>
</table>

## 约束说明<a name="section177921451558"></a>

-   目的操作数和源操作数的数据类型T和U数据类型需要配套使用。类型配套对应表如下：

    **表 3**  Scatter操作数数据类型对应表

    <a name="table113383813355"></a>
    <table><thead align="left"><tr id="row14332038103513"><th class="cellrowborder" valign="top" width="49.980000000000004%" id="mcps1.2.3.1.1"><p id="p1833133810356"><a name="p1833133810356"></a><a name="p1833133810356"></a><strong id="b9504175916367"><a name="b9504175916367"></a><a name="b9504175916367"></a>T数据类型</strong></p>
    </th>
    <th class="cellrowborder" valign="top" width="50.019999999999996%" id="mcps1.2.3.1.2"><p id="p976633143610"><a name="p976633143610"></a><a name="p976633143610"></a><strong id="b5567202313717"><a name="b5567202313717"></a><a name="b5567202313717"></a>U数据类型</strong></p>
    </th>
    </tr>
    </thead>
    <tbody><tr id="row133323813516"><td class="cellrowborder" valign="top" width="49.980000000000004%" headers="mcps1.2.3.1.1 "><p id="p18331038103513"><a name="p18331038103513"></a><a name="p18331038103513"></a>int8_t</p>
    </td>
    <td class="cellrowborder" rowspan="6" valign="top" width="50.019999999999996%" headers="mcps1.2.3.1.2 "><p id="p15887151117145"><a name="p15887151117145"></a><a name="p15887151117145"></a>uint16_t</p>
    </td>
    </tr>
    <tr id="row203373817352"><td class="cellrowborder" valign="top" headers="mcps1.2.3.1.1 "><p id="p2063453093715"><a name="p2063453093715"></a><a name="p2063453093715"></a>uint8_t</p>
    </td>
    </tr>
    <tr id="row2341238123517"><td class="cellrowborder" valign="top" headers="mcps1.2.3.1.1 "><p id="p117200408116"><a name="p117200408116"></a><a name="p117200408116"></a>int16_t</p>
    </td>
    </tr>
    <tr id="row4262125631120"><td class="cellrowborder" valign="top" headers="mcps1.2.3.1.1 "><p id="p379103418145"><a name="p379103418145"></a><a name="p379103418145"></a>uint16_t</p>
    </td>
    </tr>
    <tr id="row547835831119"><td class="cellrowborder" valign="top" headers="mcps1.2.3.1.1 "><p id="p3478258111111"><a name="p3478258111111"></a><a name="p3478258111111"></a>half</p>
    </td>
    </tr>
    <tr id="row1554214081213"><td class="cellrowborder" valign="top" headers="mcps1.2.3.1.1 "><p id="p654219010124"><a name="p654219010124"></a><a name="p654219010124"></a>bfloat16_t</p>
    </td>
    </tr>
    <tr id="row2366738124"><td class="cellrowborder" valign="top" width="49.980000000000004%" headers="mcps1.2.3.1.1 "><p id="p135461007200"><a name="p135461007200"></a><a name="p135461007200"></a>int32_t</p>
    </td>
    <td class="cellrowborder" rowspan="3" valign="top" width="50.019999999999996%" headers="mcps1.2.3.1.2 "><p id="p16197333152012"><a name="p16197333152012"></a><a name="p16197333152012"></a>uint32_t</p>
    </td>
    </tr>
    <tr id="row53216191211"><td class="cellrowborder" valign="top" headers="mcps1.2.3.1.1 "><p id="p2546904203"><a name="p2546904203"></a><a name="p2546904203"></a>uint32_t</p>
    </td>
    </tr>
    <tr id="row1531643132013"><td class="cellrowborder" valign="top" headers="mcps1.2.3.1.1 "><p id="p2031633112010"><a name="p2031633112010"></a><a name="p2031633112010"></a>float</p>
    </td>
    </tr>
    <tr id="row9238173114222"><td class="cellrowborder" valign="top" width="49.980000000000004%" headers="mcps1.2.3.1.1 "><p id="p1923893120228"><a name="p1923893120228"></a><a name="p1923893120228"></a>uint64_t</p>
    </td>
    <td class="cellrowborder" rowspan="2" valign="top" width="50.019999999999996%" headers="mcps1.2.3.1.2 "><p id="p16238203152211"><a name="p16238203152211"></a><a name="p16238203152211"></a>uint32_t</p>
    </td>
    </tr>
    <tr id="row8742183452210"><td class="cellrowborder" valign="top" headers="mcps1.2.3.1.1 "><p id="p1474223472217"><a name="p1474223472217"></a><a name="p1474223472217"></a>int64_t</p>
    </td>
    </tr>
    <tr id="row815222532313"><td class="cellrowborder" valign="top" width="49.980000000000004%" headers="mcps1.2.3.1.1 "><p id="p151531525102315"><a name="p151531525102315"></a><a name="p151531525102315"></a>uint64_t</p>
    </td>
    <td class="cellrowborder" rowspan="2" valign="top" width="50.019999999999996%" headers="mcps1.2.3.1.2 "><p id="p1115372532312"><a name="p1115372532312"></a><a name="p1115372532312"></a>uint64_t</p>
    </td>
    </tr>
    <tr id="row13903827202318"><td class="cellrowborder" valign="top" headers="mcps1.2.3.1.1 "><p id="p1290392732318"><a name="p1290392732318"></a><a name="p1290392732318"></a>int64_t</p>
    </td>
    </tr>
    </tbody>
    </table>

-   当T为b64数据类型时，T，U，V数据类型只支持以下组合：

    <a name="table12289184933717"></a>
    <table><thead align="left"><tr id="row1828994912372"><th class="cellrowborder" valign="top" width="10.05899410058994%" id="mcps1.1.6.1.1"><p id="p928964943718"><a name="p928964943718"></a><a name="p928964943718"></a><strong id="b3289134933716"><a name="b3289134933716"></a><a name="b3289134933716"></a>T数据类型</strong></p>
    </th>
    <th class="cellrowborder" valign="top" width="12.518748125187482%" id="mcps1.1.6.1.2"><p id="p14290649173712"><a name="p14290649173712"></a><a name="p14290649173712"></a><strong id="b729017498375"><a name="b729017498375"></a><a name="b729017498375"></a>IndexT数据类型</strong></p>
    </th>
    <th class="cellrowborder" valign="top" width="32.396760323967605%" id="mcps1.1.6.1.3"><p id="p91865215264"><a name="p91865215264"></a><a name="p91865215264"></a>RegT数据类型</p>
    </th>
    <th class="cellrowborder" valign="top" width="31.676832316768326%" id="mcps1.1.6.1.4"><p id="p18272151118265"><a name="p18272151118265"></a><a name="p18272151118265"></a>RegIndexT数据类型</p>
    </th>
    <th class="cellrowborder" valign="top" width="13.348665133486653%" id="mcps1.1.6.1.5"><p id="p1031455113419"><a name="p1031455113419"></a><a name="p1031455113419"></a>备注</p>
    </th>
    </tr>
    </thead>
    <tbody><tr id="row9291174913376"><td class="cellrowborder" rowspan="4" valign="top" width="10.05899410058994%" headers="mcps1.1.6.1.1 "><p id="p1129114953713"><a name="p1129114953713"></a><a name="p1129114953713"></a>b64</p>
    </td>
    <td class="cellrowborder" rowspan="4" valign="top" width="12.518748125187482%" headers="mcps1.1.6.1.2 "><p id="p1129110499375"><a name="p1129110499375"></a><a name="p1129110499375"></a>uint32_t</p>
    </td>
    <td class="cellrowborder" valign="top" width="32.396760323967605%" headers="mcps1.1.6.1.3 "><p id="p218205219263"><a name="p218205219263"></a><a name="p218205219263"></a>RegTensor&lt;uint64_t, RegTraitNumOne&gt;</p>
    </td>
    <td class="cellrowborder" rowspan="2" valign="top" width="31.676832316768326%" headers="mcps1.1.6.1.4 "><p id="p19272101114261"><a name="p19272101114261"></a><a name="p19272101114261"></a>RegTensor&lt;uint32_t&gt;</p>
    </td>
    <td class="cellrowborder" rowspan="2" valign="top" width="13.348665133486653%" headers="mcps1.1.6.1.5 "><p id="p193151951174116"><a name="p193151951174116"></a><a name="p193151951174116"></a>index前32个数有效</p>
    </td>
    </tr>
    <tr id="row3871031102018"><td class="cellrowborder" valign="top" headers="mcps1.1.6.1.1 "><p id="p19871143111207"><a name="p19871143111207"></a><a name="p19871143111207"></a>RegTensor&lt;int64_t, RegTraitNumOne&gt;</p>
    </td>
    </tr>
    <tr id="row988838102915"><td class="cellrowborder" valign="top" headers="mcps1.1.6.1.1 "><p id="p11478715102916"><a name="p11478715102916"></a><a name="p11478715102916"></a>RegTensor&lt;uint64_t, RegTraitNumTwo&gt;</p>
    </td>
    <td class="cellrowborder" rowspan="2" valign="top" headers="mcps1.1.6.1.2 "><p id="p1147810157296"><a name="p1147810157296"></a><a name="p1147810157296"></a>RegTensor&lt;uint32_t&gt;</p>
    </td>
    <td class="cellrowborder" rowspan="2" valign="top" headers="mcps1.1.6.1.3 "><p id="p760235214334"><a name="p760235214334"></a><a name="p760235214334"></a>-</p>
    </td>
    </tr>
    <tr id="row9291194914377"><td class="cellrowborder" valign="top" headers="mcps1.1.6.1.1 "><p id="p11015347281"><a name="p11015347281"></a><a name="p11015347281"></a>RegTensor&lt;int64_t, RegTraitNumTwo&gt;</p>
    </td>
    </tr>
    <tr id="row10291124993710"><td class="cellrowborder" rowspan="6" valign="top" width="10.05899410058994%" headers="mcps1.1.6.1.1 "><p id="p19894025204712"><a name="p19894025204712"></a><a name="p19894025204712"></a>b64</p>
    </td>
    <td class="cellrowborder" rowspan="6" valign="top" width="12.518748125187482%" headers="mcps1.1.6.1.2 "><p id="p1029115494373"><a name="p1029115494373"></a><a name="p1029115494373"></a>uint64_t</p>
    </td>
    <td class="cellrowborder" valign="top" width="32.396760323967605%" headers="mcps1.1.6.1.3 "><p id="p518205252610"><a name="p518205252610"></a><a name="p518205252610"></a>RegTensor&lt;uint64_t, RegTraitNumOne&gt;</p>
    </td>
    <td class="cellrowborder" rowspan="2" valign="top" width="31.676832316768326%" headers="mcps1.1.6.1.4 "><p id="p92725114263"><a name="p92725114263"></a><a name="p92725114263"></a>RegTensor&lt;uint64_t, RegTraitNumOne&gt;</p>
    </td>
    <td class="cellrowborder" rowspan="2" valign="top" width="13.348665133486653%" headers="mcps1.1.6.1.5 "><p id="p1530312317340"><a name="p1530312317340"></a><a name="p1530312317340"></a>-</p>
    </td>
    </tr>
    <tr id="row11601713203111"><td class="cellrowborder" valign="top" headers="mcps1.1.6.1.1 "><p id="p0683112014312"><a name="p0683112014312"></a><a name="p0683112014312"></a>RegTensor&lt;int64_t, RegTraitNumOne&gt;</p>
    </td>
    </tr>
    <tr id="row1129174953719"><td class="cellrowborder" valign="top" headers="mcps1.1.6.1.1 "><p id="p1218105216262"><a name="p1218105216262"></a><a name="p1218105216262"></a>RegTensor&lt;uint64_t, RegTraitNumOne&gt;</p>
    </td>
    <td class="cellrowborder" rowspan="2" valign="top" headers="mcps1.1.6.1.2 "><p id="p927381172612"><a name="p927381172612"></a><a name="p927381172612"></a>RegTensor&lt;uint64_t, RegTraitNumTwo&gt;</p>
    </td>
    <td class="cellrowborder" rowspan="2" valign="top" headers="mcps1.1.6.1.3 "><p id="p997016102457"><a name="p997016102457"></a><a name="p997016102457"></a>index前32个数有效</p>
    </td>
    </tr>
    <tr id="row12218203853112"><td class="cellrowborder" valign="top" headers="mcps1.1.6.1.1 "><p id="p970384323113"><a name="p970384323113"></a><a name="p970384323113"></a>RegTensor&lt;int64_t, RegTraitNumOne&gt;</p>
    </td>
    </tr>
    <tr id="row37431356143119"><td class="cellrowborder" valign="top" headers="mcps1.1.6.1.1 "><p id="p9743115663112"><a name="p9743115663112"></a><a name="p9743115663112"></a>RegTensor&lt;uint64_t,RegTraitNumTwo&gt;</p>
    </td>
    <td class="cellrowborder" rowspan="2" valign="top" headers="mcps1.1.6.1.2 "><p id="p774375673117"><a name="p774375673117"></a><a name="p774375673117"></a>RegTensor&lt;uint64_t, RegTraitNumTwo&gt;</p>
    </td>
    <td class="cellrowborder" rowspan="2" valign="top" headers="mcps1.1.6.1.3 "><p id="p1639781318341"><a name="p1639781318341"></a><a name="p1639781318341"></a>-</p>
    </td>
    </tr>
    <tr id="row63729173457"><td class="cellrowborder" valign="top" headers="mcps1.1.6.1.1 "><p id="p1844917115297"><a name="p1844917115297"></a><a name="p1844917115297"></a>RegTensor&lt;int64_t,RegTraitNumTwo&gt;</p>
    </td>
    </tr>
    </tbody>
    </table>

-   当T为int8或者uint8数据类型时，源操作数Tensor中仅偶数位Byte有效。最终存入UB结果地址的数据仅为源操作数Tensor偶数位Byte，即srcReg中的第0，2，4，...， 252，254位置数据会被分散存储到目的操作数中。
-   index中不能有相同的值。若有2个或者2个以上的index中的数据相同，则只有其中一个值对应的数据是有效的，具体哪一个数据可能是未知的。
-   索引位置需要按照dtype\_size（单个元素的Byte数）对齐，否则可能会造成数据分散结果错乱。

## 调用示例<a name="section642mcpsimp"></a>

```
template<typename T, typename U>
 __simd_vf__ inline void ScatterVF(__ubuf__ T* dstAddr, __ubuf__ T* src0Addr, __ubuf__ U* src1Addr, uint32_t count, uint32_t oneRepeatSize, uint16_t repeatTimes)
{
    AscendC::MicroAPI::RegTensor<T> srcReg0;
    AscendC::MicroAPI::RegTensor<U> srcReg1;
    AscendC::MicroAPI::MaskReg mask;
    for (uint16_t i = 0; i < repeatTimes; i++) {
        mask = AscendC::MicroAPI::UpdateMask<T>(count);
        AscendC::MicroAPI::LoadAlign(srcReg0, src0Addr + i * oneRepeatSize);
        AscendC::MicroAPI::LoadAlign(srcReg1, src1Addr + i * oneRepeatSize);
        AscendC::MicroAPI::Scatter(dstAddr, srcReg0, srcReg1, mask);
    }
}
```

