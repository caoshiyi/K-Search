# UnPack<a name="ZH-CN_TOPIC_0000002523303710"></a>

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

## 功能说明<a name="section1922292299"></a>

对于无符号整型，将源操作数srcReg中低半部分或高半部分的元素以高位填0扩充位宽的方式写入dstReg。对于有符号整型，将源操作数srcReg中低半部分或高半部分的元素以保持符号位扩充位宽的方式写入dstReg。

## 函数原型<a name="section620mcpsimp"></a>

```
template <typename T = DefaultType, typename U = DefaultType, HighLowPart part = HighLowPart::LOWEST, typename S, typename V>
__simd_callee__ inline void UnPack(S& dstReg, V& srcReg)
```

## 参数说明<a name="section622mcpsimp"></a>

**表 1**  模板参数说明

<a name="table104985541254"></a>
<table><thead align="left"><tr id="row1049811545512"><th class="cellrowborder" valign="top" width="17.9%" id="mcps1.2.3.1.1"><p id="p3498854954"><a name="p3498854954"></a><a name="p3498854954"></a>参数名</p>
</th>
<th class="cellrowborder" valign="top" width="82.1%" id="mcps1.2.3.1.2"><p id="p11498954655"><a name="p11498954655"></a><a name="p11498954655"></a>描述</p>
</th>
</tr>
</thead>
<tbody><tr id="row11313124114714"><td class="cellrowborder" valign="top" width="17.9%" headers="mcps1.2.3.1.1 "><p id="p9314741179"><a name="p9314741179"></a><a name="p9314741179"></a>T</p>
</td>
<td class="cellrowborder" valign="top" width="82.1%" headers="mcps1.2.3.1.2 "><p id="p831413412717"><a name="p831413412717"></a><a name="p831413412717"></a>目的操作数数据类型。</p>
<p id="p172481539145717"><a name="p172481539145717"></a><a name="p172481539145717"></a><span id="ph5248539195712"><a name="ph5248539195712"></a><a name="ph5248539195712"></a>Ascend 950PR/Ascend 950DT</span>，支持的数据类型为：int16_t/uint16_t/int32_t/uint32_t/int64_t/uint64_t</p>
<p id="p20454143018480"><a name="p20454143018480"></a><a name="p20454143018480"></a>源操作数和目的操作数的数据类型约束参见<a href="#table113383813355">表3</a>。</p>
</td>
</tr>
<tr id="row1964716441376"><td class="cellrowborder" valign="top" width="17.9%" headers="mcps1.2.3.1.1 "><p id="p16487444716"><a name="p16487444716"></a><a name="p16487444716"></a>U</p>
</td>
<td class="cellrowborder" valign="top" width="82.1%" headers="mcps1.2.3.1.2 "><p id="p176481441671"><a name="p176481441671"></a><a name="p176481441671"></a>源操作数数据类型。</p>
<p id="p1085122254811"><a name="p1085122254811"></a><a name="p1085122254811"></a><span id="ph13851192274818"><a name="ph13851192274818"></a><a name="ph13851192274818"></a>Ascend 950PR/Ascend 950DT</span>，支持的数据类型为：int8_t/uint8_t/int16_t/uint16_t/int32_t/uint32_t</p>
</td>
</tr>
<tr id="row949845412516"><td class="cellrowborder" valign="top" width="17.9%" headers="mcps1.2.3.1.1 "><p id="p1249817547512"><a name="p1249817547512"></a><a name="p1249817547512"></a>part</p>
</td>
<td class="cellrowborder" valign="top" width="82.1%" headers="mcps1.2.3.1.2 "><p id="p208823103615"><a name="p208823103615"></a><a name="p208823103615"></a>枚举类型，用于控制读取srcReg的低半部分还是高半部分。</p>
<a name="ul9882111017617"></a><a name="ul9882111017617"></a><ul id="ul9882111017617"><li>HighLowPart::LOWEST，低位模式，读取srcReg的低半部分。</li><li>HighLowPart::HIGHEST，高位模式，读取srcReg的高半部分。</li></ul>
<p id="p1699981417236"><a name="p1699981417236"></a><a name="p1699981417236"></a>注：RegTraitNumTwo只支持LOWEST模式。</p>
</td>
</tr>
<tr id="row187272531671"><td class="cellrowborder" valign="top" width="17.9%" headers="mcps1.2.3.1.1 "><p id="p1972755311715"><a name="p1972755311715"></a><a name="p1972755311715"></a>S</p>
</td>
<td class="cellrowborder" valign="top" width="82.1%" headers="mcps1.2.3.1.2 "><p id="p2727175317715"><a name="p2727175317715"></a><a name="p2727175317715"></a>目的操作数RegTensor类型。</p>
</td>
</tr>
<tr id="row03641457677"><td class="cellrowborder" valign="top" width="17.9%" headers="mcps1.2.3.1.1 "><p id="p1836419571272"><a name="p1836419571272"></a><a name="p1836419571272"></a>V</p>
</td>
<td class="cellrowborder" valign="top" width="82.1%" headers="mcps1.2.3.1.2 "><p id="p0364125720714"><a name="p0364125720714"></a><a name="p0364125720714"></a>源操作数RegTensor类型。</p>
</td>
</tr>
</tbody>
</table>

**表 2**  函数参数说明

<a name="table4835205712588"></a>
<table><thead align="left"><tr id="row118356578583"><th class="cellrowborder" valign="top" width="17.64%" id="mcps1.2.3.1.1"><p id="p48354572582"><a name="p48354572582"></a><a name="p48354572582"></a>参数名</p>
</th>
<th class="cellrowborder" valign="top" width="82.36%" id="mcps1.2.3.1.2"><p id="p583535795817"><a name="p583535795817"></a><a name="p583535795817"></a>描述</p>
</th>
</tr>
</thead>
<tbody><tr id="row1835857145817"><td class="cellrowborder" valign="top" width="17.64%" headers="mcps1.2.3.1.1 "><p id="p5835457165816"><a name="p5835457165816"></a><a name="p5835457165816"></a>dstReg</p>
</td>
<td class="cellrowborder" valign="top" width="82.36%" headers="mcps1.2.3.1.2 "><p id="p168351657155818"><a name="p168351657155818"></a><a name="p168351657155818"></a>目的操作数。</p>
<p id="p18671158172215"><a name="p18671158172215"></a><a name="p18671158172215"></a>类型为<a href="RegTensor.md">RegTensor</a>。</p>
</td>
</tr>
<tr id="row18835145716587"><td class="cellrowborder" valign="top" width="17.64%" headers="mcps1.2.3.1.1 "><p id="p59747391278"><a name="p59747391278"></a><a name="p59747391278"></a>srcReg</p>
</td>
<td class="cellrowborder" valign="top" width="82.36%" headers="mcps1.2.3.1.2 "><p id="p53355414286"><a name="p53355414286"></a><a name="p53355414286"></a>源操作数。</p>
<p id="p25982110231"><a name="p25982110231"></a><a name="p25982110231"></a>类型为<a href="RegTensor.md">RegTensor</a>。</p>
</td>
</tr>
</tbody>
</table>

**表 3**  源操作数和目的操作数的数据类型对应表

<a name="table113383813355"></a>
<table><thead align="left"><tr id="row14332038103513"><th class="cellrowborder" valign="top" width="49.95%" id="mcps1.2.3.1.1"><p id="p1833133810356"><a name="p1833133810356"></a><a name="p1833133810356"></a><strong id="b9504175916367"><a name="b9504175916367"></a><a name="b9504175916367"></a>T数据类型</strong></p>
</th>
<th class="cellrowborder" valign="top" width="50.05%" id="mcps1.2.3.1.2"><p id="p122992052379"><a name="p122992052379"></a><a name="p122992052379"></a><strong id="b5299155103716"><a name="b5299155103716"></a><a name="b5299155103716"></a>U数据类型</strong></p>
</th>
</tr>
</thead>
<tbody><tr id="row133323813516"><td class="cellrowborder" valign="top" width="49.95%" headers="mcps1.2.3.1.1 "><p id="p18331038103513"><a name="p18331038103513"></a><a name="p18331038103513"></a>int16_t</p>
</td>
<td class="cellrowborder" valign="top" width="50.05%" headers="mcps1.2.3.1.2 "><p id="p15408550165913"><a name="p15408550165913"></a><a name="p15408550165913"></a>int8_t</p>
</td>
</tr>
<tr id="row203373817352"><td class="cellrowborder" valign="top" width="49.95%" headers="mcps1.2.3.1.1 "><p id="p2063453093715"><a name="p2063453093715"></a><a name="p2063453093715"></a>uint16_t</p>
</td>
<td class="cellrowborder" valign="top" width="50.05%" headers="mcps1.2.3.1.2 "><p id="p171344768"><a name="p171344768"></a><a name="p171344768"></a>uint8_t</p>
</td>
</tr>
<tr id="row2341238123517"><td class="cellrowborder" valign="top" width="49.95%" headers="mcps1.2.3.1.1 "><p id="p18889624919"><a name="p18889624919"></a><a name="p18889624919"></a>int32_t</p>
</td>
<td class="cellrowborder" valign="top" width="50.05%" headers="mcps1.2.3.1.2 "><p id="p15872614498"><a name="p15872614498"></a><a name="p15872614498"></a>int16_t</p>
</td>
</tr>
<tr id="row4262125631120"><td class="cellrowborder" valign="top" width="49.95%" headers="mcps1.2.3.1.1 "><p id="p4871766493"><a name="p4871766493"></a><a name="p4871766493"></a>uint32_t</p>
</td>
<td class="cellrowborder" valign="top" width="50.05%" headers="mcps1.2.3.1.2 "><p id="p1186166154919"><a name="p1186166154919"></a><a name="p1186166154919"></a>uint16_t</p>
</td>
</tr>
<tr id="row56958525913"><td class="cellrowborder" valign="top" width="49.95%" headers="mcps1.2.3.1.1 "><p id="p1069513521294"><a name="p1069513521294"></a><a name="p1069513521294"></a>uint64_t</p>
</td>
<td class="cellrowborder" valign="top" width="50.05%" headers="mcps1.2.3.1.2 "><p id="p136959525918"><a name="p136959525918"></a><a name="p136959525918"></a>uint32_t</p>
</td>
</tr>
<tr id="row9314561598"><td class="cellrowborder" valign="top" width="49.95%" headers="mcps1.2.3.1.1 "><p id="p832567915"><a name="p832567915"></a><a name="p832567915"></a>int64_t</p>
</td>
<td class="cellrowborder" valign="top" width="50.05%" headers="mcps1.2.3.1.2 "><p id="p4314561193"><a name="p4314561193"></a><a name="p4314561193"></a>int32_t</p>
</td>
</tr>
</tbody>
</table>

## 返回值说明<a name="section640mcpsimp"></a>

无

## 约束说明<a name="section633mcpsimp"></a>

无

## 调用示例<a name="section932512912207"></a>

```
template<typename T, typename U, int32_t mode>
__simd_vf__ inline void UnPackVF(__ubuf__ T* dstAddr, __ubuf__ U* srcAddr, uint32_t oneDstRepSize, uint16_t repeatTimes, uint32_t oneSrcRepSize)
{
    AscendC::MicroAPI::RegTensor<U> srcReg;
    AscendC::MicroAPI::RegTensor<T> dstReg;
    AscendC::MicroAPI::MaskReg mask = AscendC::MicroAPI::CreateMask<T, AscendC::MicroAPI::MaskPattern::ALL>();
    for (uint16_t i = 0; i < repeatTimes; i++) {
        AscendC::MicroAPI::LoadAlign(srcReg, srcAddr + i * oneSrcRepSize);
        if constexpr (mode == 0) {
            AscendC::MicroAPI::UnPack<T, U, AscendC::MicroAPI::HighLowPart::LOWEST>(dstReg, srcReg);
        } else if constexpr (mode == 1) {
            AscendC::MicroAPI::UnPack<T, U, AscendC::MicroAPI::HighLowPart::HIGHEST>(dstReg, srcReg);
        }
        AscendC::MicroAPI::StoreAlign(dstAddr + i * oneDstRepSize, dstReg, mask);
    }
}
```

