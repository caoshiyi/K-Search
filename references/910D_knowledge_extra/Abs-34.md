# Abs<a name="ZH-CN_TOPIC_0000002554344617"></a>

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

对srcReg中的有效元素逐个取绝对值，并将结果写入dstReg中对应位置处。

## 函数原型<a name="section620mcpsimp"></a>

```
template <typename T = DefaultType, MaskMergeMode mode = MaskMergeMode::ZEROING, typename U>
__simd_callee__ inline void Abs(U& dstReg, U& srcReg, MaskReg& mask)

template <typename T = DefaultType, typename U = DefaultType, MaskMergeMode mode = MaskMergeMode::ZEROING, typename S, typename V>
__simd_callee__ inline void Abs(S& dstReg, V& srcReg, MaskReg& mask)
```

## 参数说明<a name="section622mcpsimp"></a>

**表 1**  模板参数说明

<a name="table4835205712588"></a>
<table><thead align="left"><tr id="row118356578583"><th class="cellrowborder" valign="top" width="18.25%" id="mcps1.2.3.1.1"><p id="p48354572582"><a name="p48354572582"></a><a name="p48354572582"></a>参数名</p>
</th>
<th class="cellrowborder" valign="top" width="81.75%" id="mcps1.2.3.1.2"><p id="p583535795817"><a name="p583535795817"></a><a name="p583535795817"></a>描述</p>
</th>
</tr>
</thead>
<tbody><tr id="row1835857145817"><td class="cellrowborder" valign="top" width="18.25%" headers="mcps1.2.3.1.1 "><p id="p5835457165816"><a name="p5835457165816"></a><a name="p5835457165816"></a>T</p>
</td>
<td class="cellrowborder" valign="top" width="81.75%" headers="mcps1.2.3.1.2 "><p id="p168351657155818"><a name="p168351657155818"></a><a name="p168351657155818"></a>操作数数据类型。</p>
<p id="p3966152216478"><a name="p3966152216478"></a><a name="p3966152216478"></a><span id="ph1966152212477"><a name="ph1966152212477"></a><a name="ph1966152212477"></a>Ascend 950PR/Ascend 950DT</span>，支持的数据类型为：int8_t/int16_t/int32_t/int64_t/half/float。</p>
</td>
</tr>
<tr id="row8105194616578"><td class="cellrowborder" valign="top" width="18.25%" headers="mcps1.2.3.1.1 "><p id="p71509475576"><a name="p71509475576"></a><a name="p71509475576"></a>U</p>
</td>
<td class="cellrowborder" valign="top" width="81.75%" headers="mcps1.2.3.1.2 "><p id="p9150347175710"><a name="p9150347175710"></a><a name="p9150347175710"></a>源操作数数据类型。</p>
<p id="p515064716579"><a name="p515064716579"></a><a name="p515064716579"></a><span id="ph14150547175719"><a name="ph14150547175719"></a><a name="ph14150547175719"></a>Ascend 950PR/Ascend 950DT</span>，支持的数据类型为：complex32/complex64；当U为complex32类型时，T必须是half类型；当U为complex64类型时，T必须是float类型。</p>
</td>
</tr>
<tr id="row18835145716587"><td class="cellrowborder" valign="top" width="18.25%" headers="mcps1.2.3.1.1 "><p id="p1756419170189"><a name="p1756419170189"></a><a name="p1756419170189"></a>mode</p>
</td>
<td class="cellrowborder" valign="top" width="81.75%" headers="mcps1.2.3.1.2 "><p id="p16933245348"><a name="p16933245348"></a><a name="p16933245348"></a>选择MERGING模式或ZEROING模式。</p>
<a name="ul49337457411"></a><a name="ul49337457411"></a><ul id="ul49337457411"><li>ZEROING模式下，mask未筛选的元素在dstReg中置零。</li><li>MERGING模式当前不支持。</li></ul>
</td>
</tr>
<tr id="row154799208400"><td class="cellrowborder" valign="top" width="18.25%" headers="mcps1.2.3.1.1 "><p id="p916243141912"><a name="p916243141912"></a><a name="p916243141912"></a>S</p>
</td>
<td class="cellrowborder" valign="top" width="81.75%" headers="mcps1.2.3.1.2 "><p id="p131047476612"><a name="p131047476612"></a><a name="p131047476612"></a><span id="ph6705211132010"><a name="ph6705211132010"></a><a name="ph6705211132010"></a>目的操作数的RegTensor类型，例如RegTensor&lt;half&gt;，由编译器自动推导，用户不需要填写。</span></p>
</td>
</tr>
<tr id="row954012741914"><td class="cellrowborder" valign="top" width="18.25%" headers="mcps1.2.3.1.1 "><p id="p9541167151915"><a name="p9541167151915"></a><a name="p9541167151915"></a>V</p>
</td>
<td class="cellrowborder" valign="top" width="81.75%" headers="mcps1.2.3.1.2 "><p id="p1254114771913"><a name="p1254114771913"></a><a name="p1254114771913"></a><span id="ph82484115207"><a name="ph82484115207"></a><a name="ph82484115207"></a>源操作数的RegTensor类型，例如RegTensor&lt;half&gt;，由编译器自动推导，用户不需要填写。</span></p>
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
<p id="p66093533169"><a name="p66093533169"></a><a name="p66093533169"></a><span id="ph20846151694212"><a name="ph20846151694212"></a><a name="ph20846151694212"></a>类型为<a href="RegTensor.md">RegTensor</a>。</span></p>
</td>
</tr>
<tr id="row2137145181815"><td class="cellrowborder" valign="top" width="18.54%" headers="mcps1.2.4.1.1 "><p id="p19574165615129"><a name="p19574165615129"></a><a name="p19574165615129"></a>srcReg</p>
</td>
<td class="cellrowborder" valign="top" width="10.05%" headers="mcps1.2.4.1.2 "><p id="p128442058144312"><a name="p128442058144312"></a><a name="p128442058144312"></a>输入</p>
</td>
<td class="cellrowborder" valign="top" width="71.41%" headers="mcps1.2.4.1.3 "><p id="p172083541517"><a name="p172083541517"></a><a name="p172083541517"></a>源操作数。</p>
<p id="p7123111612517"><a name="p7123111612517"></a><a name="p7123111612517"></a>类型为<a href="RegTensor.md">RegTensor</a>。</p>
</td>
</tr>
<tr id="row19615183817191"><td class="cellrowborder" valign="top" width="18.54%" headers="mcps1.2.4.1.1 "><p id="p1484519586432"><a name="p1484519586432"></a><a name="p1484519586432"></a>mask</p>
</td>
<td class="cellrowborder" valign="top" width="10.05%" headers="mcps1.2.4.1.2 "><p id="p484514581433"><a name="p484514581433"></a><a name="p484514581433"></a>输入</p>
</td>
<td class="cellrowborder" valign="top" width="71.41%" headers="mcps1.2.4.1.3 "><p id="p11541143920"><a name="p11541143920"></a><a name="p11541143920"></a><span id="ph1254614714212"><a name="ph1254614714212"></a><a name="ph1254614714212"></a>源操作数元素操作的有效指示，详细说明请参考<a href="MaskReg.md">MaskReg</a>。</span></p>
</td>
</tr>
</tbody>
</table>

## 返回值说明<a name="section640mcpsimp"></a>

无

## 约束说明<a name="section633mcpsimp"></a>

整型数据的计算结果如果超出数据类型的表示范围会采取非饱和截断，比如int8类型，src为-128，其绝对值128会被截断成-128。

## 调用示例<a name="section642mcpsimp"></a>

```
template<typename T>
__simd_vf__ inline void AbsVF(__ubuf__ T* dstAddr, __ubuf__ T* srcAddr, uint32_t count, 
 uint32_t oneRepeatSize, uint16_t repeatTimes)
{
    AscendC::MicroAPI::RegTensor<T> srcReg;
    AscendC::MicroAPI::RegTensor<T> dstReg;
    AscendC::MicroAPI::MaskReg mask; 
    for (uint16_t i = 0; i < repeatTimes; i++) {
        mask = AscendC::MicroAPI::UpdateMask<T>(count);
        AscendC::MicroAPI::LoadAlign(srcReg, srcAddr + i * oneRepeatSize);
        AscendC::MicroAPI::Abs(dstReg, srcReg, mask);
        AscendC::MicroAPI::StoreAlign(dstAddr + i * oneRepeatSize, dstReg, mask);
    }
}

template<typename T, typename U>
__simd_vf__ inline void AbsVF(__ubuf__ U* dstAddr, __ubuf__ T* srcAddr, uint32_t count, 
 uint32_t oneRepeatSize, uint16_t repeatTimes)
{
    AscendC::MicroAPI::RegTensor<T> srcReg;
    AscendC::MicroAPI::RegTensor<U> dstReg;
    AscendC::MicroAPI::MaskReg mask; 
    for (uint16_t i = 0; i < repeatTimes; i++) {
        mask = AscendC::MicroAPI::UpdateMask<T>(count);
        AscendC::MicroAPI::LoadAlign(srcReg, srcAddr + i * oneRepeatSize);
        AscendC::MicroAPI::Abs(dstReg, srcReg, mask);
        AscendC::MicroAPI::StoreAlign(dstAddr + i * oneRepeatSize, dstReg, mask);
    }
}
```

