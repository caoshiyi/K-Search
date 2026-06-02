# ShiftLeft<a name="ZH-CN_TOPIC_0000002554424597"></a>

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

根据mask对输入数据srcReg0，按照srcReg1对应元素进行左移操作，将结果写入dstReg。

## 函数原型<a name="section620mcpsimp"></a>

```
template <typename T = DefaultType, typename U = DefaultType, MaskMergeMode mode = MaskMergeMode::ZEROING, typename S, typename V>
__simd_callee__ inline void ShiftLeft(S& dstReg, S& srcReg0, V& srcReg1, MaskReg& mask)
```

## 参数说明<a name="section622mcpsimp"></a>

**表 1**  模板参数说明

<a name="table4835205712588"></a>
<table><thead align="left"><tr id="row118356578583"><th class="cellrowborder" valign="top" width="18.35%" id="mcps1.2.3.1.1"><p id="p48354572582"><a name="p48354572582"></a><a name="p48354572582"></a>参数名</p>
</th>
<th class="cellrowborder" valign="top" width="81.65%" id="mcps1.2.3.1.2"><p id="p583535795817"><a name="p583535795817"></a><a name="p583535795817"></a>描述</p>
</th>
</tr>
</thead>
<tbody><tr id="row1835857145817"><td class="cellrowborder" valign="top" width="18.35%" headers="mcps1.2.3.1.1 "><p id="p5835457165816"><a name="p5835457165816"></a><a name="p5835457165816"></a>T</p>
</td>
<td class="cellrowborder" valign="top" width="81.65%" headers="mcps1.2.3.1.2 "><p id="p168351657155818"><a name="p168351657155818"></a><a name="p168351657155818"></a>源操作数数据类型。</p>
<p id="p127911257381"><a name="p127911257381"></a><a name="p127911257381"></a><span id="ph1679112523817"><a name="ph1679112523817"></a><a name="ph1679112523817"></a>Ascend 950PR/Ascend 950DT</span>，支持的数据类型为：uint8_t/int8_t/uint16_t/int16_t/uint32_t/int32_t/uint64_t/int64_t</p>
</td>
</tr>
<tr id="row356441781813"><td class="cellrowborder" valign="top" width="18.35%" headers="mcps1.2.3.1.1 "><p id="p1756419170189"><a name="p1756419170189"></a><a name="p1756419170189"></a>U</p>
</td>
<td class="cellrowborder" valign="top" width="81.65%" headers="mcps1.2.3.1.2 "><p id="p7564191710185"><a name="p7564191710185"></a><a name="p7564191710185"></a>左移参数数据类型。</p>
<p id="p1891493003912"><a name="p1891493003912"></a><a name="p1891493003912"></a><span id="ph6914123018395"><a name="ph6914123018395"></a><a name="ph6914123018395"></a>Ascend 950PR/Ascend 950DT</span>，支持的数据类型为：int8_t/int16_t/int32_t/int64_t</p>
<p id="p68181291745"><a name="p68181291745"></a><a name="p68181291745"></a><span>源操作数和目的操作数的数据类型约束参考</span><a href="#table15348612514">表3 源操作数和目的操作数的数据类型约束</a>。</p>
</td>
</tr>
<tr id="row916216311197"><td class="cellrowborder" valign="top" width="18.35%" headers="mcps1.2.3.1.1 "><p id="p6802539661"><a name="p6802539661"></a><a name="p6802539661"></a>mode</p>
</td>
<td class="cellrowborder" valign="top" width="81.65%" headers="mcps1.2.3.1.2 "><p id="p77520541653"><a name="p77520541653"></a><a name="p77520541653"></a>选择MERGING模式或ZEROING模式。</p>
<a name="ul1163765616511"></a><a name="ul1163765616511"></a><ul id="ul1163765616511"><li>ZEROING，mask未筛选的元素在dst中置零。</li><li>MERGING，当前不支持。</li></ul>
</td>
</tr>
<tr id="row36123513619"><td class="cellrowborder" valign="top" width="18.35%" headers="mcps1.2.3.1.1 "><p id="p916243141912"><a name="p916243141912"></a><a name="p916243141912"></a>S</p>
</td>
<td class="cellrowborder" valign="top" width="81.65%" headers="mcps1.2.3.1.2 "><p id="p131047476612"><a name="p131047476612"></a><a name="p131047476612"></a>srcReg0/dstReg的RegTensor类型， 例如RegTensor&lt;uint32_t&gt;，由编译器自动推导，用户不需要填写。</p>
</td>
</tr>
<tr id="row0687131810519"><td class="cellrowborder" valign="top" width="18.35%" headers="mcps1.2.3.1.1 "><p id="p156882018157"><a name="p156882018157"></a><a name="p156882018157"></a>V</p>
</td>
<td class="cellrowborder" valign="top" width="81.65%" headers="mcps1.2.3.1.2 "><p id="p206881518751"><a name="p206881518751"></a><a name="p206881518751"></a>srcReg1的RegTensor类型， 例如RegTensor&lt;int32_t&gt;，由编译器自动推导，用户不需要填写。</p>
</td>
</tr>
</tbody>
</table>

**表 2**  参数说明

<a name="zh-cn_topic_0235751031_table33761356"></a>
<table><thead align="left"><tr id="zh-cn_topic_0235751031_row27598891"><th class="cellrowborder" valign="top" width="18.54%" id="mcps1.2.4.1.1"><p id="zh-cn_topic_0235751031_p20917673"><a name="zh-cn_topic_0235751031_p20917673"></a><a name="zh-cn_topic_0235751031_p20917673"></a>参数名</p>
</th>
<th class="cellrowborder" valign="top" width="10.05%" id="mcps1.2.4.1.2"><p id="zh-cn_topic_0235751031_p16609919"><a name="zh-cn_topic_0235751031_p16609919"></a><a name="zh-cn_topic_0235751031_p16609919"></a>输入/输出</p>
</th>
<th class="cellrowborder" valign="top" width="71.41%" id="mcps1.2.4.1.3"><p id="zh-cn_topic_0235751031_p59995477"><a name="zh-cn_topic_0235751031_p59995477"></a><a name="zh-cn_topic_0235751031_p59995477"></a>描述</p>
</th>
</tr>
</thead>
<tbody><tr id="row42461942101815"><td class="cellrowborder" valign="top" width="18.54%" headers="mcps1.2.4.1.1 "><p id="p108051250181214"><a name="p108051250181214"></a><a name="p108051250181214"></a>dstReg</p>
</td>
<td class="cellrowborder" valign="top" width="10.05%" headers="mcps1.2.4.1.2 "><p id="p158449584436"><a name="p158449584436"></a><a name="p158449584436"></a>输出</p>
</td>
<td class="cellrowborder" valign="top" width="71.41%" headers="mcps1.2.4.1.3 "><p id="p6389114616514"><a name="p6389114616514"></a><a name="p6389114616514"></a>目的操作数。</p>
<p id="p66093533169"><a name="p66093533169"></a><a name="p66093533169"></a>类型为<a href="RegTensor.md">RegTensor</a>。</p>
</td>
</tr>
<tr id="row2137145181815"><td class="cellrowborder" valign="top" width="18.54%" headers="mcps1.2.4.1.1 "><p id="p19574165615129"><a name="p19574165615129"></a><a name="p19574165615129"></a>srcReg0</p>
</td>
<td class="cellrowborder" valign="top" width="10.05%" headers="mcps1.2.4.1.2 "><p id="p128442058144312"><a name="p128442058144312"></a><a name="p128442058144312"></a>输入</p>
</td>
<td class="cellrowborder" valign="top" width="71.41%" headers="mcps1.2.4.1.3 "><p id="p172083541517"><a name="p172083541517"></a><a name="p172083541517"></a>源操作数。</p>
<p id="p7123111612517"><a name="p7123111612517"></a><a name="p7123111612517"></a>类型为<a href="RegTensor.md">RegTensor</a>。</p>
<p id="p1484485824312"><a name="p1484485824312"></a><a name="p1484485824312"></a>源操作数的数据类型需要与目的操作数保持一致。</p>
</td>
</tr>
<tr id="row11773440341"><td class="cellrowborder" valign="top" width="18.54%" headers="mcps1.2.4.1.1 "><p id="p10132448173420"><a name="p10132448173420"></a><a name="p10132448173420"></a>srcReg1</p>
</td>
<td class="cellrowborder" valign="top" width="10.05%" headers="mcps1.2.4.1.2 "><p id="p151325484342"><a name="p151325484342"></a><a name="p151325484342"></a>输入</p>
</td>
<td class="cellrowborder" valign="top" width="71.41%" headers="mcps1.2.4.1.3 "><p id="p21216245395"><a name="p21216245395"></a><a name="p21216245395"></a>源操作数。</p>
<p id="p181326485341"><a name="p181326485341"></a><a name="p181326485341"></a>类型为<a href="RegTensor.md">RegTensor</a>。</p>
<p id="p19711819173920"><a name="p19711819173920"></a><a name="p19711819173920"></a>不支持设置为负数。</p>
</td>
</tr>
<tr id="row19615183817191"><td class="cellrowborder" valign="top" width="18.54%" headers="mcps1.2.4.1.1 "><p id="p1484519586432"><a name="p1484519586432"></a><a name="p1484519586432"></a>mask</p>
</td>
<td class="cellrowborder" valign="top" width="10.05%" headers="mcps1.2.4.1.2 "><p id="p484514581433"><a name="p484514581433"></a><a name="p484514581433"></a>输入</p>
</td>
<td class="cellrowborder" valign="top" width="71.41%" headers="mcps1.2.4.1.3 "><p id="p11541143920"><a name="p11541143920"></a><a name="p11541143920"></a><span id="ph15776181222"><a name="ph15776181222"></a><a name="ph15776181222"></a>源操作数元素操作的有效指示，详细说明请参考<a href="MaskReg.md">MaskReg</a>。</span></p>
</td>
</tr>
</tbody>
</table>

**表 3**  源操作数和目的操作数的数据类型约束

<a name="table15348612514"></a>
<table><thead align="left"><tr id="row3341061250"><th class="cellrowborder" valign="top" width="50%" id="mcps1.2.3.1.1"><p id="p10342614250"><a name="p10342614250"></a><a name="p10342614250"></a>srcReg0/dstReg数据类型</p>
</th>
<th class="cellrowborder" valign="top" width="50%" id="mcps1.2.3.1.2"><p id="p2034968252"><a name="p2034968252"></a><a name="p2034968252"></a>srcReg1数据类型</p>
</th>
</tr>
</thead>
<tbody><tr id="row10341367252"><td class="cellrowborder" valign="top" width="50%" headers="mcps1.2.3.1.1 "><p id="p51901218260"><a name="p51901218260"></a><a name="p51901218260"></a>uint8_t</p>
</td>
<td class="cellrowborder" valign="top" width="50%" headers="mcps1.2.3.1.2 "><p id="p1328659182611"><a name="p1328659182611"></a><a name="p1328659182611"></a>int8_t</p>
</td>
</tr>
<tr id="row5341466253"><td class="cellrowborder" valign="top" width="50%" headers="mcps1.2.3.1.1 "><p id="p819052102614"><a name="p819052102614"></a><a name="p819052102614"></a>uint16_t</p>
</td>
<td class="cellrowborder" valign="top" width="50%" headers="mcps1.2.3.1.2 "><p id="p122869913269"><a name="p122869913269"></a><a name="p122869913269"></a>int16_t</p>
</td>
</tr>
<tr id="row163413612519"><td class="cellrowborder" valign="top" width="50%" headers="mcps1.2.3.1.1 "><p id="p61908212262"><a name="p61908212262"></a><a name="p61908212262"></a>uint32_t</p>
</td>
<td class="cellrowborder" valign="top" width="50%" headers="mcps1.2.3.1.2 "><p id="p6286109202619"><a name="p6286109202619"></a><a name="p6286109202619"></a>int32_t</p>
</td>
</tr>
<tr id="row4532510164711"><td class="cellrowborder" valign="top" width="50%" headers="mcps1.2.3.1.1 "><p id="p204451180503"><a name="p204451180503"></a><a name="p204451180503"></a>uint64_t</p>
</td>
<td class="cellrowborder" valign="top" width="50%" headers="mcps1.2.3.1.2 "><p id="p1144501845010"><a name="p1144501845010"></a><a name="p1144501845010"></a>int64_t</p>
</td>
</tr>
<tr id="row03411632513"><td class="cellrowborder" valign="top" width="50%" headers="mcps1.2.3.1.1 "><p id="p1319018218263"><a name="p1319018218263"></a><a name="p1319018218263"></a>int8_t</p>
</td>
<td class="cellrowborder" valign="top" width="50%" headers="mcps1.2.3.1.2 "><p id="p1228615912614"><a name="p1228615912614"></a><a name="p1228615912614"></a>int8_t</p>
</td>
</tr>
<tr id="row203417618256"><td class="cellrowborder" valign="top" width="50%" headers="mcps1.2.3.1.1 "><p id="p51904232620"><a name="p51904232620"></a><a name="p51904232620"></a>int16_t</p>
</td>
<td class="cellrowborder" valign="top" width="50%" headers="mcps1.2.3.1.2 "><p id="p128629122611"><a name="p128629122611"></a><a name="p128629122611"></a>int16_t</p>
</td>
</tr>
<tr id="row1022415972519"><td class="cellrowborder" valign="top" width="50%" headers="mcps1.2.3.1.1 "><p id="p1919013222612"><a name="p1919013222612"></a><a name="p1919013222612"></a>int32_t</p>
</td>
<td class="cellrowborder" valign="top" width="50%" headers="mcps1.2.3.1.2 "><p id="p628615962615"><a name="p628615962615"></a><a name="p628615962615"></a>int32_t</p>
</td>
</tr>
<tr id="row18107145118466"><td class="cellrowborder" valign="top" width="50%" headers="mcps1.2.3.1.1 "><p id="p125401256184615"><a name="p125401256184615"></a><a name="p125401256184615"></a>int64_t</p>
</td>
<td class="cellrowborder" valign="top" width="50%" headers="mcps1.2.3.1.2 "><p id="p654025674614"><a name="p654025674614"></a><a name="p654025674614"></a>int64_t</p>
</td>
</tr>
</tbody>
</table>

## 返回值说明<a name="section640mcpsimp"></a>

无

## 约束说明<a name="section633mcpsimp"></a>

-   对于逻辑位移（无符号数据类型），如果位移量大于数据类型位宽，则输出为0。
-   对于算数位移（有符号数据类型），如果srcReg0小于0，srcReg1小于0，并且位移量大于数据类型位宽，则输出-1；如果srcReg0大于0，并且位移量大于数据类型位宽，则输出0。

## 调用示例<a name="section642mcpsimp"></a>

```
template<typename T, typename U>
__simd_vf__ inline void ShiftLeftVF(__ubuf__ T* dstAddr, __ubuf__ T* src0Addr, __ubuf__ U* src1Addr, uint32_t count, uint32_t oneRepeatSize, uint16_t repeatTimes)
{
    AscendC::MicroAPI::RegTensor<T> srcReg0;
    AscendC::MicroAPI::RegTensor<U> srcReg1;
    AscendC::MicroAPI::RegTensor<T> dstReg;
    AscendC::MicroAPI::MaskReg mask;
    for (uint16_t i = 0; i < repeatTimes; i++) {
        mask = AscendC::MicroAPI::UpdateMask<T>(count);
        AscendC::MicroAPI::LoadAlign(srcReg0, src0Addr + i * oneRepeatSize);
        AscendC::MicroAPI::LoadAlign(srcReg1, src1Addr + i * oneRepeatSize);       
        AscendC::MicroAPI::ShiftLeft(dstReg, srcReg0, srcReg1, mask);
        AscendC::MicroAPI::StoreAlign(dstAddr + i * oneRepeatSize, dstReg, mask);
    }
}
```

