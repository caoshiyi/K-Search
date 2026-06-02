# 连续非对齐搬入<a name="ZH-CN_TOPIC_0000002554424629"></a>

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

Reg矢量计算数据搬运接口，适用于从UB非32B对齐地址起始连续搬入RegTensor。

## 函数原型<a name="section620mcpsimp"></a>

```
// LoadUnAlignPre接口用于在进行非对齐搬入前的初始化，适用于场景1和场景2
template <typename T>
__simd_callee__ inline void LoadUnAlignPre(UnalignRegForLoad& ureg, __ubuf__ T* srcAddr);

// LoadUnAlignPre接口用于在进行非对齐搬入前的初始化，适用于场景3
template <typename T>
__simd_callee__ inline void LoadUnAlignPre(UnalignRegForLoad& ureg, __ubuf__ T* srcAddr, AddrReg& areg);

// 场景1:使用uint32_t作为偏移量，使用post mode搬运 ，每调用一次接口更新源操作数在UB上的地址
template <typename T = DefaultType, PostLiteral postMode = PostLiteral::POST_MODE_UPDATE, typename U>
__simd_callee__ inline void LoadUnAlign(U& dstReg, UnalignRegForLoad& ureg, __ubuf__ T*& srcAddr, uint32_t postUpdateStride);

// 场景2:调用接口不改变源操作数在UB上的地址，每次Repeat需要手动更新源操作数在UB上的地址
template <typename T = DefaultType, typename U>
__simd_callee__ inline void LoadUnAlign(U& dstReg, UnalignRegForLoad& ureg, __ubuf__ T* srcAddr);

// 场景3:使用 AddrReg 存储偏移量进行数据搬运
template <typename T = DefaultType, typename U>
__simd_callee__ inline void LoadUnAlign(U& dstReg, UnalignRegForLoad& ureg, __ubuf__ T*& srcAddr, AddrReg& areg, uint32_t inc);
```

## 参数说明<a name="section622mcpsimp"></a>

**表 1**  连续非对齐搬入使用uint32\_t存储偏移量场景参数说明

<a name="table14132101714462"></a>
<table><thead align="left"><tr id="row19176617124620"><th class="cellrowborder" valign="top" width="13.268673132686734%" id="mcps1.2.4.1.1"><p id="p117621724612"><a name="p117621724612"></a><a name="p117621724612"></a>参数名</p>
</th>
<th class="cellrowborder" valign="top" width="12.468753124687531%" id="mcps1.2.4.1.2"><p id="p417681717463"><a name="p417681717463"></a><a name="p417681717463"></a>输入/输出</p>
</th>
<th class="cellrowborder" valign="top" width="74.26257374262573%" id="mcps1.2.4.1.3"><p id="p5176017164614"><a name="p5176017164614"></a><a name="p5176017164614"></a>描述</p>
</th>
</tr>
</thead>
<tbody><tr id="row42693511365"><td class="cellrowborder" valign="top" width="13.268673132686734%" headers="mcps1.2.4.1.1 "><p id="p123381318914"><a name="p123381318914"></a><a name="p123381318914"></a>T</p>
</td>
<td class="cellrowborder" valign="top" width="12.468753124687531%" headers="mcps1.2.4.1.2 "><p id="p1733820311492"><a name="p1733820311492"></a><a name="p1733820311492"></a>输入</p>
</td>
<td class="cellrowborder" valign="top" width="74.26257374262573%" headers="mcps1.2.4.1.3 "><p id="p53381931597"><a name="p53381931597"></a><a name="p53381931597"></a>模板参数，支持的数据类型为b8/b16/b32/b64。</p>
</td>
</tr>
<tr id="row81451754964"><td class="cellrowborder" valign="top" width="13.268673132686734%" headers="mcps1.2.4.1.1 "><p id="p533814311498"><a name="p533814311498"></a><a name="p533814311498"></a>postMode</p>
</td>
<td class="cellrowborder" valign="top" width="12.468753124687531%" headers="mcps1.2.4.1.2 "><p id="p1338031399"><a name="p1338031399"></a><a name="p1338031399"></a>输入</p>
</td>
<td class="cellrowborder" valign="top" width="74.26257374262573%" headers="mcps1.2.4.1.3 "><p id="p14697155195612"><a name="p14697155195612"></a><a name="p14697155195612"></a>用于控制是否使能post update，<a href="PostLiteral.md">PostLiteral</a>类型。</p>
</td>
</tr>
<tr id="row207479481062"><td class="cellrowborder" valign="top" width="13.268673132686734%" headers="mcps1.2.4.1.1 "><p id="p20338831596"><a name="p20338831596"></a><a name="p20338831596"></a>U</p>
</td>
<td class="cellrowborder" valign="top" width="12.468753124687531%" headers="mcps1.2.4.1.2 "><p id="p13393312913"><a name="p13393312913"></a><a name="p13393312913"></a>输入</p>
</td>
<td class="cellrowborder" valign="top" width="74.26257374262573%" headers="mcps1.2.4.1.3 "><p id="p1433913312093"><a name="p1433913312093"></a><a name="p1433913312093"></a>模板参数，支持的数据类型对应的RegTensor类型。</p>
</td>
</tr>
<tr id="row7879164218227"><td class="cellrowborder" valign="top" width="13.268673132686734%" headers="mcps1.2.4.1.1 "><p id="p48791042192216"><a name="p48791042192216"></a><a name="p48791042192216"></a>ureg</p>
</td>
<td class="cellrowborder" valign="top" width="12.468753124687531%" headers="mcps1.2.4.1.2 "><p id="p987974211223"><a name="p987974211223"></a><a name="p987974211223"></a>输出/输入</p>
</td>
<td class="cellrowborder" valign="top" width="74.26257374262573%" headers="mcps1.2.4.1.3 "><p id="p2087964292215"><a name="p2087964292215"></a><a name="p2087964292215"></a>UnalignRegForLoad，非对齐寄存器，用于保存非对齐数据，长度32B。LoadUnAlignPre函数中作为输出，LoadUnAlign函数中作为输入/输出。</p>
</td>
</tr>
<tr id="row1417671716463"><td class="cellrowborder" valign="top" width="13.268673132686734%" headers="mcps1.2.4.1.1 "><p id="p1039635944011"><a name="p1039635944011"></a><a name="p1039635944011"></a>dstReg</p>
</td>
<td class="cellrowborder" valign="top" width="12.468753124687531%" headers="mcps1.2.4.1.2 "><p id="p1317616173468"><a name="p1317616173468"></a><a name="p1317616173468"></a>输出</p>
</td>
<td class="cellrowborder" valign="top" width="74.26257374262573%" headers="mcps1.2.4.1.3 "><p id="p2176141714466"><a name="p2176141714466"></a><a name="p2176141714466"></a>目的操作数，类型为RegTensor。</p>
</td>
</tr>
<tr id="row958051624416"><td class="cellrowborder" valign="top" width="13.268673132686734%" headers="mcps1.2.4.1.1 "><p id="p9580161634416"><a name="p9580161634416"></a><a name="p9580161634416"></a>srcAddr</p>
</td>
<td class="cellrowborder" valign="top" width="12.468753124687531%" headers="mcps1.2.4.1.2 "><p id="p2581416144417"><a name="p2581416144417"></a><a name="p2581416144417"></a>输入/输出</p>
</td>
<td class="cellrowborder" valign="top" width="74.26257374262573%" headers="mcps1.2.4.1.3 "><p id="p195811716134418"><a name="p195811716134418"></a><a name="p195811716134418"></a>源操作数在UB上的起始地址。</p>
</td>
</tr>
<tr id="row4397131094210"><td class="cellrowborder" valign="top" width="13.268673132686734%" headers="mcps1.2.4.1.1 "><p id="p639881018426"><a name="p639881018426"></a><a name="p639881018426"></a>postUpdateStride</p>
</td>
<td class="cellrowborder" valign="top" width="12.468753124687531%" headers="mcps1.2.4.1.2 "><p id="p1343482217424"><a name="p1343482217424"></a><a name="p1343482217424"></a>输入</p>
</td>
<td class="cellrowborder" valign="top" width="74.26257374262573%" headers="mcps1.2.4.1.3 "><p id="p1787861914493"><a name="p1787861914493"></a><a name="p1787861914493"></a>仅支持POST_MODE_UPDATE场景：实际搬运UB起始地址为srcAddr，搬运后执行地址更新srcAddr +=  postUpdateStride。</p>
</td>
</tr>
</tbody>
</table>

**表 2**  连续非对齐搬入使用AddrReg存储偏移量场景参数说明

<a name="table2575858192119"></a>
<table><thead align="left"><tr id="row16575058162116"><th class="cellrowborder" valign="top" width="13.268673132686734%" id="mcps1.2.4.1.1"><p id="p1957585817213"><a name="p1957585817213"></a><a name="p1957585817213"></a>参数名</p>
</th>
<th class="cellrowborder" valign="top" width="12.468753124687531%" id="mcps1.2.4.1.2"><p id="p45751658122119"><a name="p45751658122119"></a><a name="p45751658122119"></a>输入/输出</p>
</th>
<th class="cellrowborder" valign="top" width="74.26257374262573%" id="mcps1.2.4.1.3"><p id="p857515816212"><a name="p857515816212"></a><a name="p857515816212"></a>描述</p>
</th>
</tr>
</thead>
<tbody><tr id="row17575858192116"><td class="cellrowborder" valign="top" width="13.268673132686734%" headers="mcps1.2.4.1.1 "><p id="p95757589214"><a name="p95757589214"></a><a name="p95757589214"></a>ureg</p>
</td>
<td class="cellrowborder" valign="top" width="12.468753124687531%" headers="mcps1.2.4.1.2 "><p id="p757505822118"><a name="p757505822118"></a><a name="p757505822118"></a>输出/输入</p>
</td>
<td class="cellrowborder" valign="top" width="74.26257374262573%" headers="mcps1.2.4.1.3 "><p id="p5575155872116"><a name="p5575155872116"></a><a name="p5575155872116"></a>UnalignRegForLoad，非对齐寄存器，用于保存非对齐数据，长度32B。LoadUnAlignPre函数中作为输出，LoadUnAlign函数中作为输入/输出。</p>
</td>
</tr>
<tr id="row1757585832113"><td class="cellrowborder" valign="top" width="13.268673132686734%" headers="mcps1.2.4.1.1 "><p id="p1857525872115"><a name="p1857525872115"></a><a name="p1857525872115"></a>dstReg</p>
</td>
<td class="cellrowborder" valign="top" width="12.468753124687531%" headers="mcps1.2.4.1.2 "><p id="p1957512581219"><a name="p1957512581219"></a><a name="p1957512581219"></a>输出</p>
</td>
<td class="cellrowborder" valign="top" width="74.26257374262573%" headers="mcps1.2.4.1.3 "><p id="p1575458192111"><a name="p1575458192111"></a><a name="p1575458192111"></a>目的操作数，类型为RegTensor。</p>
</td>
</tr>
<tr id="row25753582212"><td class="cellrowborder" valign="top" width="13.268673132686734%" headers="mcps1.2.4.1.1 "><p id="p1257585872118"><a name="p1257585872118"></a><a name="p1257585872118"></a>srcAddr</p>
</td>
<td class="cellrowborder" valign="top" width="12.468753124687531%" headers="mcps1.2.4.1.2 "><p id="p175751358172118"><a name="p175751358172118"></a><a name="p175751358172118"></a>输入/输出</p>
</td>
<td class="cellrowborder" valign="top" width="74.26257374262573%" headers="mcps1.2.4.1.3 "><p id="p9575135832118"><a name="p9575135832118"></a><a name="p9575135832118"></a>源操作数在UB上的起始地址。</p>
</td>
</tr>
<tr id="row125751458142111"><td class="cellrowborder" valign="top" width="13.268673132686734%" headers="mcps1.2.4.1.1 "><p id="p203811519102816"><a name="p203811519102816"></a><a name="p203811519102816"></a>areg</p>
</td>
<td class="cellrowborder" valign="top" width="12.468753124687531%" headers="mcps1.2.4.1.2 "><p id="p135751658182112"><a name="p135751658182112"></a><a name="p135751658182112"></a>输入</p>
</td>
<td class="cellrowborder" valign="top" width="74.26257374262573%" headers="mcps1.2.4.1.3 "><p id="p1757614581216"><a name="p1757614581216"></a><a name="p1757614581216"></a>AddrReg数据类型，存储地址偏移量offset。实际搬运UB起始地址为srcAddr + offset。</p>
</td>
</tr>
<tr id="row1893221452310"><td class="cellrowborder" valign="top" width="13.268673132686734%" headers="mcps1.2.4.1.1 "><p id="p8932141452319"><a name="p8932141452319"></a><a name="p8932141452319"></a>inc</p>
</td>
<td class="cellrowborder" valign="top" width="12.468753124687531%" headers="mcps1.2.4.1.2 "><p id="p1593211144235"><a name="p1593211144235"></a><a name="p1593211144235"></a>输入</p>
</td>
<td class="cellrowborder" valign="top" width="74.26257374262573%" headers="mcps1.2.4.1.3 "><p id="p0245503262"><a name="p0245503262"></a><a name="p0245503262"></a>搬运完成后offset 在自增前被更新为offset + inc。</p>
</td>
</tr>
</tbody>
</table>

## 约束说明<a name="section1575141714439"></a>

-   该接口中的srcAddr不需要32B对齐。
-   LoadUnAlignPre与LoadUnAlign接口需要组合使用。

## 调用示例<a name="section642mcpsimp"></a>

```
// 连续非对齐搬入使用 uint32_t 存储偏移量场景
template <typename T>
__simd_vf__ inline void LoadUnAlignVF(__ubuf__ T* dstAddr, __ubuf__ T* srcAddr, uint32_t postUpdateStride, uint16_t repeatTimes)
{
    AscendC::MicroAPI::RegTensor<T> srcReg;
    AscendC::MicroAPI::UnalignRegForLoad ureg0;
    AscendC::MicroAPI::UnalignRegForStore ureg1;   
    for (uint16_t i = 0; i < repeatTimes; ++i) {
        AscendC::MicroAPI::LoadUnAlignPre(ureg0, srcAddr + i * postUpdateStride);
        AscendC::MicroAPI::LoadUnAlign(srcReg, ureg0, srcAddr + i * postUpdateStride);
        AscendC::MicroAPI::StoreUnAlign(dstAddr, srcReg, ureg1, postUpdateStride);
    }
    AscendC::MicroAPI::StoreUnAlignPost(dstAddr, ureg1, 0);
}

// 连续非对齐搬入使用 AddrReg 存储偏移量场景
template <typename T>
__simd_vf__ inline void LoadUnAlignVF(__ubuf__ T* dstAddr, __ubuf__ T* srcAddr, uint32_t oneRepeatSize, uint16_t repeatTimes)
{
    AscendC::MicroAPI::RegTensor<T> srcReg;
    AscendC::MicroAPI::UnalignRegForLoad ureg0;
    AscendC::MicroAPI::UnalignRegForStore ureg1;
    AscendC::MicroAPI::AddrReg aReg;
    for (uint16_t i = 0; i < repeatTimes; ++i) {
        aReg = AscendC::MicroAPI::CreateAddrReg<T>(i, oneRepeatSize);
        AscendC::MicroAPI::LoadUnAlignPre(ureg0, srcAddr, aReg);
        AscendC::MicroAPI::LoadUnAlign(srcReg, ureg0, srcAddr, aReg, 0);
        AscendC::MicroAPI::StoreUnAlign(dstAddr, srcReg, ureg1, aReg);
    }
    AscendC::MicroAPI::StoreUnAlignPost(dstAddr, ureg1, aReg);
}
```

