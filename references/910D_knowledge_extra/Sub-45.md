# Sub<a name="ZH-CN_TOPIC_0000002523304560"></a>

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

根据mask对源操作数src0、src1进行按元素相减的操作，将结果写入目的操作数dst。src0, src1相减产生借位结果时，在MaskReg carry中对应位置每4bit的最低位写0，否则写1。计算公式如下：

<!-- img2text -->
$$
\text{dst}_{i} =
\begin{cases}
\text{src0}_{i} - \text{src1}_{i}, & \text{mask}_{i} = 1 \\
\text{dst}_{i}, & \text{mask}_{i} = 0
\end{cases}
$$

$$
\text{carry}_{i} =
\begin{cases}
0, & \text{src0}_{i} - \text{src1}_{i} < 0 \\
1, & \text{src0}_{i} - \text{src1}_{i} \geq 0
\end{cases}
$$

具体的示例如下：

**表 1**  示例说明

<a name="table78091736133120"></a>
<table><thead align="left"><tr id="row480917360316"><th class="cellrowborder" valign="top" width="16.31163116311631%" id="mcps1.2.4.1.1"><p id="p580917368314"><a name="p580917368314"></a><a name="p580917368314"></a>数据类型</p>
</th>
<th class="cellrowborder" valign="top" width="19.561956195619562%" id="mcps1.2.4.1.2"><p id="p1280912365313"><a name="p1280912365313"></a><a name="p1280912365313"></a>是否借位</p>
</th>
<th class="cellrowborder" valign="top" width="64.12641264126412%" id="mcps1.2.4.1.3"><p id="p680913653119"><a name="p680913653119"></a><a name="p680913653119"></a>示例说明</p>
</th>
</tr>
</thead>
<tbody><tr id="row88091236203115"><td class="cellrowborder" rowspan="2" valign="top" width="16.31163116311631%" headers="mcps1.2.4.1.1 "><p id="p21571440123413"><a name="p21571440123413"></a><a name="p21571440123413"></a>int32_t数据类型</p>
</td>
<td class="cellrowborder" valign="top" width="19.561956195619562%" headers="mcps1.2.4.1.2 "><p id="p154637020320"><a name="p154637020320"></a><a name="p154637020320"></a>不产生借位</p>
</td>
<td class="cellrowborder" valign="top" width="64.12641264126412%" headers="mcps1.2.4.1.3 "><p id="p3311243193110"><a name="p3311243193110"></a><a name="p3311243193110"></a>a_i = 5, b_i = 2</p>
<p id="p531113439313"><a name="p531113439313"></a><a name="p531113439313"></a>dst_i = a_i - b_i = 3</p>
<p id="p4311164312316"><a name="p4311164312316"></a><a name="p4311164312316"></a>carry中对应位置每4bit的最低位写1：carry_i = 1</p>
</td>
</tr>
<tr id="row181043603119"><td class="cellrowborder" valign="top" headers="mcps1.2.4.1.1 "><p id="p58101636153112"><a name="p58101636153112"></a><a name="p58101636153112"></a>产生借位</p>
</td>
<td class="cellrowborder" valign="top" headers="mcps1.2.4.1.2 "><p id="p1121711024413"><a name="p1121711024413"></a><a name="p1121711024413"></a>a_i = 5, b_i = -7</p>
<p id="p15172134819317"><a name="p15172134819317"></a><a name="p15172134819317"></a>dst_i = a_i - b_i  = 12</p>
<p id="p5172174883118"><a name="p5172174883118"></a><a name="p5172174883118"></a>carry中对应位置每4bit的最低位写0：carry_i = 0</p>
</td>
</tr>
<tr id="row381063633114"><td class="cellrowborder" rowspan="2" valign="top" width="16.31163116311631%" headers="mcps1.2.4.1.1 "><p id="p3815204223410"><a name="p3815204223410"></a><a name="p3815204223410"></a>uint32_t数据类型</p>
</td>
<td class="cellrowborder" valign="top" width="19.561956195619562%" headers="mcps1.2.4.1.2 "><p id="p11843103015347"><a name="p11843103015347"></a><a name="p11843103015347"></a>不产生借位</p>
</td>
<td class="cellrowborder" valign="top" width="64.12641264126412%" headers="mcps1.2.4.1.3 "><p id="p570203082917"><a name="p570203082917"></a><a name="p570203082917"></a>a_i = 5, b_i = 2,</p>
<p id="p168941627173112"><a name="p168941627173112"></a><a name="p168941627173112"></a>dst_i = a_i - b_i = 3</p>
<p id="p29562393290"><a name="p29562393290"></a><a name="p29562393290"></a>carry中对应位置每4bit的最低位写1：carry_i = 1</p>
</td>
</tr>
<tr id="row1481017368317"><td class="cellrowborder" valign="top" headers="mcps1.2.4.1.1 "><p id="p148435308342"><a name="p148435308342"></a><a name="p148435308342"></a>产生借位</p>
</td>
<td class="cellrowborder" valign="top" headers="mcps1.2.4.1.2 "><p id="p686855917304"><a name="p686855917304"></a><a name="p686855917304"></a>a_i = 5, b_i = 7</p>
<p id="p34696716311"><a name="p34696716311"></a><a name="p34696716311"></a>dst_i = a_i - b_i = -2</p>
<p id="p8868185973017"><a name="p8868185973017"></a><a name="p8868185973017"></a>carry中对应位置每4bit的最低位写0：carry_i = 0</p>
</td>
</tr>
</tbody>
</table>

## 函数原型<a name="section620mcpsimp"></a>

```
template <typename T = DefaultType, typename U>
__simd_callee__ inline void Sub(MaskReg& carry, U& dstReg, U& srcReg0, U& srcReg1, MaskReg& mask)
```

## 参数说明<a name="section389364115374"></a>

**表 2**  模板参数说明

<a name="table4835205712588"></a>
<table><thead align="left"><tr id="row118356578583"><th class="cellrowborder" valign="top" width="18.099999999999998%" id="mcps1.2.3.1.1"><p id="p48354572582"><a name="p48354572582"></a><a name="p48354572582"></a>参数名</p>
</th>
<th class="cellrowborder" valign="top" width="81.89999999999999%" id="mcps1.2.3.1.2"><p id="p583535795817"><a name="p583535795817"></a><a name="p583535795817"></a>描述</p>
</th>
</tr>
</thead>
<tbody><tr id="row1835857145817"><td class="cellrowborder" valign="top" width="18.099999999999998%" headers="mcps1.2.3.1.1 "><p id="p5835457165816"><a name="p5835457165816"></a><a name="p5835457165816"></a>T</p>
</td>
<td class="cellrowborder" valign="top" width="81.89999999999999%" headers="mcps1.2.3.1.2 "><p id="p168351657155818"><a name="p168351657155818"></a><a name="p168351657155818"></a>矢量目的操作数和源操作数的数据类型。</p>
<p id="p15149173514615"><a name="p15149173514615"></a><a name="p15149173514615"></a><span id="ph7464649163819"><a name="ph7464649163819"></a><a name="ph7464649163819"></a>Ascend 950PR/Ascend 950DT</span>，支持的数据类型为：int32_t、uint32_t。</p>
</td>
</tr>
<tr id="row94714521855"><td class="cellrowborder" valign="top" width="18.099999999999998%" headers="mcps1.2.3.1.1 "><p id="p916243141912"><a name="p916243141912"></a><a name="p916243141912"></a>U</p>
</td>
<td class="cellrowborder" valign="top" width="81.89999999999999%" headers="mcps1.2.3.1.2 "><p id="p131047476612"><a name="p131047476612"></a><a name="p131047476612"></a><span id="ph19851723182011"><a name="ph19851723182011"></a><a name="ph19851723182011"></a>目的操作数的RegTensor类型，例如RegTensor&lt;half&gt;，由编译器自动推导，用户不需要填写。</span></p>
</td>
</tr>
</tbody>
</table>

**表 3**  参数说明

<a name="zh-cn_topic_0235751031_table33761356"></a>
<table><thead align="left"><tr id="zh-cn_topic_0235751031_row27598891"><th class="cellrowborder" valign="top" width="18.54%" id="mcps1.2.4.1.1"><p id="zh-cn_topic_0235751031_p20917673"><a name="zh-cn_topic_0235751031_p20917673"></a><a name="zh-cn_topic_0235751031_p20917673"></a>参数名</p>
</th>
<th class="cellrowborder" valign="top" width="10.05%" id="mcps1.2.4.1.2"><p id="zh-cn_topic_0235751031_p16609919"><a name="zh-cn_topic_0235751031_p16609919"></a><a name="zh-cn_topic_0235751031_p16609919"></a>输入/输出</p>
</th>
<th class="cellrowborder" valign="top" width="71.41%" id="mcps1.2.4.1.3"><p id="zh-cn_topic_0235751031_p59995477"><a name="zh-cn_topic_0235751031_p59995477"></a><a name="zh-cn_topic_0235751031_p59995477"></a>描述</p>
</th>
</tr>
</thead>
<tbody><tr id="row42461942101815"><td class="cellrowborder" valign="top" width="18.54%" headers="mcps1.2.4.1.1 "><p id="p284425844311"><a name="p284425844311"></a><a name="p284425844311"></a>dstReg</p>
</td>
<td class="cellrowborder" valign="top" width="10.05%" headers="mcps1.2.4.1.2 "><p id="p158449584436"><a name="p158449584436"></a><a name="p158449584436"></a>输出</p>
</td>
<td class="cellrowborder" valign="top" width="71.41%" headers="mcps1.2.4.1.3 "><p id="p6389114616514"><a name="p6389114616514"></a><a name="p6389114616514"></a>目的操作数。</p>
<p id="p66093533169"><a name="p66093533169"></a><a name="p66093533169"></a><span id="ph134278176129"><a name="ph134278176129"></a><a name="ph134278176129"></a>类型为<a href="RegTensor.md">RegTensor</a>。</span></p>
</td>
</tr>
<tr id="row2137145181815"><td class="cellrowborder" valign="top" width="18.54%" headers="mcps1.2.4.1.1 "><p id="p6844125874315"><a name="p6844125874315"></a><a name="p6844125874315"></a>srcReg0</p>
</td>
<td class="cellrowborder" valign="top" width="10.05%" headers="mcps1.2.4.1.2 "><p id="p128442058144312"><a name="p128442058144312"></a><a name="p128442058144312"></a>输入</p>
</td>
<td class="cellrowborder" valign="top" width="71.41%" headers="mcps1.2.4.1.3 "><p id="p18831019185112"><a name="p18831019185112"></a><a name="p18831019185112"></a>源操作数。</p>
<p id="p1962711407127"><a name="p1962711407127"></a><a name="p1962711407127"></a><span id="ph18627114014128"><a name="ph18627114014128"></a><a name="ph18627114014128"></a>类型为<a href="RegTensor.md">RegTensor</a>。</span></p>
<p id="p488351919519"><a name="p488351919519"></a><a name="p488351919519"></a>数据类型需要与目的操作数保持一致。</p>
</td>
</tr>
<tr id="row891912431168"><td class="cellrowborder" valign="top" width="18.54%" headers="mcps1.2.4.1.1 "><p id="p195756503168"><a name="p195756503168"></a><a name="p195756503168"></a>srcReg1</p>
</td>
<td class="cellrowborder" valign="top" width="10.05%" headers="mcps1.2.4.1.2 "><p id="p457515071618"><a name="p457515071618"></a><a name="p457515071618"></a>输入</p>
</td>
<td class="cellrowborder" valign="top" width="71.41%" headers="mcps1.2.4.1.3 "><p id="p732824954017"><a name="p732824954017"></a><a name="p732824954017"></a>源操作数。</p>
<p id="p1152961215412"><a name="p1152961215412"></a><a name="p1152961215412"></a><span id="ph17479104711212"><a name="ph17479104711212"></a><a name="ph17479104711212"></a>类型为<a href="RegTensor.md">RegTensor</a>。</span></p>
<p id="p101261830153914"><a name="p101261830153914"></a><a name="p101261830153914"></a>数据类型需要与目的操作数保持一致。</p>
</td>
</tr>
<tr id="row1196710311594"><td class="cellrowborder" valign="top" width="18.54%" headers="mcps1.2.4.1.1 "><p id="p15429173015619"><a name="p15429173015619"></a><a name="p15429173015619"></a>carry</p>
</td>
<td class="cellrowborder" valign="top" width="10.05%" headers="mcps1.2.4.1.2 "><p id="p342933075615"><a name="p342933075615"></a><a name="p342933075615"></a>输出</p>
</td>
<td class="cellrowborder" valign="top" width="71.41%" headers="mcps1.2.4.1.3 "><p id="p1142915300561"><a name="p1142915300561"></a><a name="p1142915300561"></a>目的操作数。输出进位值。</p>
<p id="p1742912304566"><a name="p1742912304566"></a><a name="p1742912304566"></a>类型为<a href="MaskReg.md">MaskReg</a>。</p>
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

## 返回值说明<a name="section640mcpsimp"></a>

无

## 约束说明<a name="section633mcpsimp"></a>

无

## 调用示例<a name="section232816306478"></a>

```
template <typename T>
__simd_vf__ inline void SubVF(__ubuf__ T* dst0Addr, __ubuf__ T* dst1Addr, __ubuf__ T* src0Addr, __ubuf__ T* src1Addr, uint32_t count, uint32_t repeatTimes, uint16_t oneRepeatSize){
    
    AscendC::MicroAPI::RegTensor<T> srcReg0;
    AscendC::MicroAPI::RegTensor<T> srcReg1;
    AscendC::MicroAPI::RegTensor<T> dstReg0;
    AscendC::MicroAPI::RegTensor<T> dstReg1;
    AscendC::MicroAPI::MaskReg mask;   
    AscendC::MicroAPI::MaskReg carry = AscendC::MicroAPI::CreateMask<uint8_t>();
    for (uint16_t i = 0; i < repeatTimes; i++) {
        mask = AscendC::MicroAPI::UpdateMask<T>(count);
        AscendC::MicroAPI::LoadAlign(srcReg0, src0Addr + i * oneRepeatSize);
        AscendC::MicroAPI::LoadAlign(srcReg1, src1Addr + i * oneRepeatSize);
        AscendC::MicroAPI::Sub(carry, dstReg0, srcReg0, srcReg1, mask);
        // 8*4B=32B align
        AscendC::MicroAPI::StoreAlign<uint32_t, AscendC::MicroAPI::MaskDist::DIST_NORM>((__ubuf__ uint32_t*)dst1Addr + i * 8, carry);
        AscendC::MicroAPI::StoreAlign(dst0Addr + i * oneRepeatSize, dstReg0, mask);
    }
}
```

