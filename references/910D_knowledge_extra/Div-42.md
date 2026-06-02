# Div<a name="ZH-CN_TOPIC_0000002554423855"></a>

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

根据mask对输入数据srcReg0、srcReg1执行按元素相除操作，将结果写入dstReg。计算公式如下：

<!-- img2text -->
$$
\text{dstReg}[i]=
\begin{cases}
\text{srcReg0}[i] \mathbin{/} \text{srcReg1}[i], & \text{mask}[i]=1 \\
0, & \text{mask}[i]=0,\ \text{mode}=\text{MaskMergeMode::ZEROING} \\
\text{dstReg}[i], & \text{mask}[i]=0,\ \text{mode}=\text{MaskMergeMode::MERGING}
\end{cases}
$$

## 函数原型<a name="section620mcpsimp"></a>

```
template <typename T = DefaultType, auto mode = MaskMergeMode::ZEROING, typename U>
__simd_callee__ inline void Div(U& dstReg, U& srcReg0, U& srcReg1, MaskReg& mask)
```

## 参数说明<a name="section622mcpsimp"></a>

**表 1**  模板参数说明

<a name="table4835205712588"></a>
<table><thead align="left"><tr id="row118356578583"><th class="cellrowborder" valign="top" width="18.310000000000002%" id="mcps1.2.3.1.1"><p id="p48354572582"><a name="p48354572582"></a><a name="p48354572582"></a>参数名</p>
</th>
<th class="cellrowborder" valign="top" width="81.69%" id="mcps1.2.3.1.2"><p id="p583535795817"><a name="p583535795817"></a><a name="p583535795817"></a>描述</p>
</th>
</tr>
</thead>
<tbody><tr id="row1835857145817"><td class="cellrowborder" valign="top" width="18.310000000000002%" headers="mcps1.2.3.1.1 "><p id="p5835457165816"><a name="p5835457165816"></a><a name="p5835457165816"></a>T</p>
</td>
<td class="cellrowborder" valign="top" width="81.69%" headers="mcps1.2.3.1.2 "><p id="p168351657155818"><a name="p168351657155818"></a><a name="p168351657155818"></a>操作数数据类型。</p>
<p id="p565812586214"><a name="p565812586214"></a><a name="p565812586214"></a><span id="ph13658558725"><a name="ph13658558725"></a><a name="ph13658558725"></a>Ascend 950PR/Ascend 950DT</span>，支持的数据类型为：uint16_t、int16_t、half、uint32_t、int32_t、float、complex32、int64_t、uint64_t、complex64。</p>
</td>
</tr>
<tr id="row18835145716587"><td class="cellrowborder" valign="top" width="18.310000000000002%" headers="mcps1.2.3.1.1 "><p id="p1383515717581"><a name="p1383515717581"></a><a name="p1383515717581"></a>mode</p>
</td>
<td class="cellrowborder" valign="top" width="81.69%" headers="mcps1.2.3.1.2 "><p id="p1315151232819"><a name="p1315151232819"></a><a name="p1315151232819"></a>可配置为MaskMergeMode的枚举或DivSpecificMode的结构体指针。</p>
<a name="ul1131085281310"></a><a name="ul1131085281310"></a><ul id="ul1131085281310"><li>配置MaskMergeMode，选择MERGING模式或ZEROING模式。<a name="ul1163765616511"></a><a name="ul1163765616511"></a><ul id="ul1163765616511"><li>ZEROING，mask未筛选的元素在dstReg中置零。</li><li>MERGING, 暂不支持。</li></ul>
</li><li>配置DivSpecificMode<pre class="screen" id="screen18481935171419"><a name="screen18481935171419"></a><a name="screen18481935171419"></a>enum class DivAlgo {
    INTRINSIC = 0,
    DIFF_COMPENSATION,
    PRECISION_1ULP_FTZ_TRUE,
    PRECISION_0ULP_FTZ_TRUE,
    PRECISION_0ULP_FTZ_FALSE,
    PRECISION_1ULP_FTZ_FALSE
};
struct DivSpecificMode {
    MaskMergeMode mrgMode = MaskMergeMode::ZEROING,
    bool precisionMode = false;
    DivAlgo algo = DivAlgo::INTRINSIC;
};</pre>
<p id="p175231030713"><a name="p175231030713"></a><a name="p175231030713"></a>当precisionMode为true时，使能更高精度的Div计算，使用差值补偿算法得出结果，最大精度误差为0 ulp。目前只针对float数据类型生效。</p>
</li></ul>
<a name="ul11875104117375"></a><a name="ul11875104117375"></a><ul id="ul11875104117375"><li>algo：用于配置Subnormal模式。<a name="ul1639612391388"></a><a name="ul1639612391388"></a><ul id="ul1639612391388"><li>DivAlgo::INTRINSIC、DivAlgo::PRECISION_1ULP_FTZ_TRUE，使用单指令计算得出结果，最大精度误差为1 ulp。</li><li>DivAlgo::DIFF_COMPENSATION、DivAlgo::PRECISION_0ULP_FTZ_TRUE，使用差值补偿算法得出结果，最大精度误差为0 ulp。目前，该算法支持float、complex64数据类型。</li><li>DivAlgo::PRECISION_0ULP_FTZ_FALSE，支持Subnormal数据计算，使用差值补偿算法得出结果，最大精度误差为0 ulp。目前，该算法支持float数据类型。</li><li>DivAlgo::PRECISION_1ULP_FTZ_FALSE，支持Subnormal数据计算，使用单指令计算得出结果，最大精度误差为1 ulp。目前，该算法支持half、float数据类型。</li></ul>
</li></ul>
</td>
</tr>
<tr id="row11614203610563"><td class="cellrowborder" valign="top" width="18.310000000000002%" headers="mcps1.2.3.1.1 "><p id="p916243141912"><a name="p916243141912"></a><a name="p916243141912"></a>U</p>
</td>
<td class="cellrowborder" valign="top" width="81.69%" headers="mcps1.2.3.1.2 "><p id="p15901115914145"><a name="p15901115914145"></a><a name="p15901115914145"></a><span id="ph19851723182011"><a name="ph19851723182011"></a><a name="ph19851723182011"></a>目的操作数的RegTensor类型，例如RegTensor&lt;half&gt;，由编译器自动推导，用户不需要填写。</span></p>
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
<p id="p66093533169"><a name="p66093533169"></a><a name="p66093533169"></a><span id="ph134278176129"><a name="ph134278176129"></a><a name="ph134278176129"></a>类型为<a href="RegTensor.md">RegTensor</a>。</span></p>
</td>
</tr>
<tr id="row2137145181815"><td class="cellrowborder" valign="top" width="18.54%" headers="mcps1.2.4.1.1 "><p id="p19574165615129"><a name="p19574165615129"></a><a name="p19574165615129"></a>srcReg0</p>
</td>
<td class="cellrowborder" valign="top" width="10.05%" headers="mcps1.2.4.1.2 "><p id="p128442058144312"><a name="p128442058144312"></a><a name="p128442058144312"></a>输入</p>
</td>
<td class="cellrowborder" valign="top" width="71.41%" headers="mcps1.2.4.1.3 "><p id="p172083541517"><a name="p172083541517"></a><a name="p172083541517"></a>源操作数。</p>
<p id="p2733105861210"><a name="p2733105861210"></a><a name="p2733105861210"></a><span id="ph87333589121"><a name="ph87333589121"></a><a name="ph87333589121"></a>类型为<a href="RegTensor.md">RegTensor</a>。</span></p>
<p id="p1484485824312"><a name="p1484485824312"></a><a name="p1484485824312"></a>数据类型需要与目的操作数保持一致。</p>
</td>
</tr>
<tr id="row11773440341"><td class="cellrowborder" valign="top" width="18.54%" headers="mcps1.2.4.1.1 "><p id="p10132448173420"><a name="p10132448173420"></a><a name="p10132448173420"></a>srcReg1</p>
</td>
<td class="cellrowborder" valign="top" width="10.05%" headers="mcps1.2.4.1.2 "><p id="p151325484342"><a name="p151325484342"></a><a name="p151325484342"></a>输入</p>
</td>
<td class="cellrowborder" valign="top" width="71.41%" headers="mcps1.2.4.1.3 "><p id="p181326485341"><a name="p181326485341"></a><a name="p181326485341"></a>源操作数。</p>
<p id="p1542710121318"><a name="p1542710121318"></a><a name="p1542710121318"></a><span id="ph194272041316"><a name="ph194272041316"></a><a name="ph194272041316"></a>类型为<a href="RegTensor.md">RegTensor</a>。</span></p>
<p id="p12132154853419"><a name="p12132154853419"></a><a name="p12132154853419"></a>数据类型需要与目的操作数保持一致。</p>
</td>
</tr>
<tr id="row19615183817191"><td class="cellrowborder" valign="top" width="18.54%" headers="mcps1.2.4.1.1 "><p id="p1484519586432"><a name="p1484519586432"></a><a name="p1484519586432"></a>mask</p>
</td>
<td class="cellrowborder" valign="top" width="10.05%" headers="mcps1.2.4.1.2 "><p id="p484514581433"><a name="p484514581433"></a><a name="p484514581433"></a>输入</p>
</td>
<td class="cellrowborder" valign="top" width="71.41%" headers="mcps1.2.4.1.3 "><p id="p1559991025517"><a name="p1559991025517"></a><a name="p1559991025517"></a><span id="ph15776181222"><a name="ph15776181222"></a><a name="ph15776181222"></a>源操作数元素操作的有效指示，详细说明请参考<a href="MaskReg.md">MaskReg</a>。</span></p>
</td>
</tr>
</tbody>
</table>

## 返回值说明<a name="section640mcpsimp"></a>

无

## 约束说明<a name="section633mcpsimp"></a>

无

## 调用示例<a name="section642mcpsimp"></a>

```
template<typename T>
__simd_vf__ inline void DivVF(__ubuf__ T* dstAddr, __ubuf__ T* src0Addr, __ubuf__ T* src1Addr, uint32_t count, uint32_t oneRepeatSize, uint16_t repeatTimes)
{
    AscendC::MicroAPI::RegTensor<T> srcReg0;
    AscendC::MicroAPI::RegTensor<T> srcReg1;
    AscendC::MicroAPI::RegTensor<T> dstReg;
    AscendC::MicroAPI::MaskReg mask;
    // 高精度模式
    // static constexpr AscendC::MicroAPI::DivSpecificMode mode = {AscendC::MicroAPI::MaskMergeMode::ZEROING, true};
    // Subnormal模式
    // static constexpr AscendC::MicroAPI::DivSpecificMode mode = {AscendC::MicroAPI::MaskMergeMode::ZEROING, true, DivAlgo::PRECISION_0ULP_FTZ_FALSE};
    for (uint16_t i = 0; i < repeatTimes; i++) {
        mask = AscendC::MicroAPI::UpdateMask<T>(count);
        AscendC::MicroAPI::LoadAlign(srcReg0, src0Addr + i * oneRepeatSize);
        AscendC::MicroAPI::LoadAlign(srcReg1, src1Addr + i * oneRepeatSize);
        AscendC::MicroAPI::Div(dstReg, srcReg0, srcReg1, mask);
        // 高精度模式/Subnormal模式
        // AscendC::MicroAPI::Div<T, &mode>(dstReg, srcReg0, srcReg1, mask);
        AscendC::MicroAPI::StoreAlign(dstAddr + i * oneRepeatSize, dstReg, mask);
    }
}
```

