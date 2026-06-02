# MulsCast<a name="ZH-CN_TOPIC_0000002554344821"></a>

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

src与scalar相乘再按照CAST\_ROUND模式转换成half类型，根据mask将计算结果写入dst。计算公式如下：

<!-- img2text -->
$$
dst_i =
\begin{cases}
\operatorname{CAST\_ROUND}(\operatorname{half},\ src_i \times scalar), & \text{mask}[i] = 1 \\
dst_i, & \text{mask}[i] = 0
\end{cases}
$$

## 函数原型<a name="section620mcpsimp"></a>

```
template <typename T0 = DefaultType, typename T1 = DefaultType, typename T2, RegLayout layout = RegLayout::ZERO, typename T3, typename T4>
__simd_callee__ inline void MulsCast(T3& dstReg, T4& srcReg, T2 scalarValue, MaskReg& mask)
```

## 参数说明<a name="section622mcpsimp"></a>

**表 1**  模板参数说明

<a name="table4835205712588"></a>
<table><thead align="left"><tr id="row118356578583"><th class="cellrowborder" valign="top" width="18.22%" id="mcps1.2.3.1.1"><p id="p48354572582"><a name="p48354572582"></a><a name="p48354572582"></a>参数名</p>
</th>
<th class="cellrowborder" valign="top" width="81.78%" id="mcps1.2.3.1.2"><p id="p583535795817"><a name="p583535795817"></a><a name="p583535795817"></a>描述</p>
</th>
</tr>
</thead>
<tbody><tr id="row1835857145817"><td class="cellrowborder" valign="top" width="18.22%" headers="mcps1.2.3.1.1 "><p id="p5835457165816"><a name="p5835457165816"></a><a name="p5835457165816"></a>T0</p>
</td>
<td class="cellrowborder" valign="top" width="81.78%" headers="mcps1.2.3.1.2 "><p id="p168351657155818"><a name="p168351657155818"></a><a name="p168351657155818"></a>目的操作数数据类型。</p>
<p id="p15959203674218"><a name="p15959203674218"></a><a name="p15959203674218"></a><span id="ph1795914368427"><a name="ph1795914368427"></a><a name="ph1795914368427"></a>Ascend 950PR/Ascend 950DT</span>，支持的数据类型为：half</p>
</td>
</tr>
<tr id="row4251861118"><td class="cellrowborder" valign="top" width="18.22%" headers="mcps1.2.3.1.1 "><p id="p22521061415"><a name="p22521061415"></a><a name="p22521061415"></a>T1</p>
</td>
<td class="cellrowborder" valign="top" width="81.78%" headers="mcps1.2.3.1.2 "><p id="p11953112614"><a name="p11953112614"></a><a name="p11953112614"></a>源操作数的数据类型。</p>
<p id="p7973921201115"><a name="p7973921201115"></a><a name="p7973921201115"></a><span id="ph597312121114"><a name="ph597312121114"></a><a name="ph597312121114"></a>Ascend 950PR/Ascend 950DT</span>，支持的数据类型为：float</p>
</td>
</tr>
<tr id="row18835145716587"><td class="cellrowborder" valign="top" width="18.22%" headers="mcps1.2.3.1.1 "><p id="p1383515717581"><a name="p1383515717581"></a><a name="p1383515717581"></a>T2</p>
</td>
<td class="cellrowborder" valign="top" width="81.78%" headers="mcps1.2.3.1.2 "><p id="p1775118537244"><a name="p1775118537244"></a><a name="p1775118537244"></a>标量源操作数的数据类型。</p>
<p id="p69731028111110"><a name="p69731028111110"></a><a name="p69731028111110"></a><span id="ph17973162841112"><a name="ph17973162841112"></a><a name="ph17973162841112"></a>Ascend 950PR/Ascend 950DT</span>，支持的数据类型为：float</p>
</td>
</tr>
<tr id="row416056114119"><td class="cellrowborder" valign="top" width="18.22%" headers="mcps1.2.3.1.1 "><p id="p95016121914"><a name="p95016121914"></a><a name="p95016121914"></a>layout</p>
</td>
<td class="cellrowborder" valign="top" width="81.78%" headers="mcps1.2.3.1.2 "><p id="p1433681723313"><a name="p1433681723313"></a><a name="p1433681723313"></a><a href="RegLayout.md">RegLayout</a>枚举类型。</p>
<p id="p24515419439"><a name="p24515419439"></a><a name="p24515419439"></a>本接口只支持RegLayout::ZERO、RegLayout::ONE。</p>
<p id="p150112121614"><a name="p150112121614"></a><a name="p150112121614"></a>src为half类型时使用，float时不生效，half类型时，RegLayout::ZERO表示从b16 RegTensor偶数位读取half元素转换成float，RegLayout::ONE表示从b16 RegTensor奇数位读取half元素转换成float。</p>
</td>
</tr>
<tr id="row17775181422"><td class="cellrowborder" valign="top" width="18.22%" headers="mcps1.2.3.1.1 "><p id="p916243141912"><a name="p916243141912"></a><a name="p916243141912"></a>T3</p>
</td>
<td class="cellrowborder" valign="top" width="81.78%" headers="mcps1.2.3.1.2 "><p id="p131047476612"><a name="p131047476612"></a><a name="p131047476612"></a>dstReg的RegTensor类型， 例如RegTensor&lt;float&gt;，由编译器自动推导，用户不需要填写。</p>
</td>
</tr>
<tr id="row5528594214"><td class="cellrowborder" valign="top" width="18.22%" headers="mcps1.2.3.1.1 "><p id="p145090101527"><a name="p145090101527"></a><a name="p145090101527"></a>T4</p>
</td>
<td class="cellrowborder" valign="top" width="81.78%" headers="mcps1.2.3.1.2 "><p id="p12509510225"><a name="p12509510225"></a><a name="p12509510225"></a>srcReg的RegTensor类型， 例如RegTensor&lt;float&gt;，由编译器自动推导，用户不需要填写。</p>
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
<tbody><tr id="row42461942101815"><td class="cellrowborder" valign="top" width="18.54%" headers="mcps1.2.4.1.1 "><p id="p284425844311"><a name="p284425844311"></a><a name="p284425844311"></a>dstReg</p>
</td>
<td class="cellrowborder" valign="top" width="10.05%" headers="mcps1.2.4.1.2 "><p id="p158449584436"><a name="p158449584436"></a><a name="p158449584436"></a>输出</p>
</td>
<td class="cellrowborder" valign="top" width="71.41%" headers="mcps1.2.4.1.3 "><p id="p86271310162918"><a name="p86271310162918"></a><a name="p86271310162918"></a>目的操作数。</p>
<p id="p66093533169"><a name="p66093533169"></a><a name="p66093533169"></a>类型为<a href="RegTensor.md">RegTensor</a>。</p>
</td>
</tr>
<tr id="row2137145181815"><td class="cellrowborder" valign="top" width="18.54%" headers="mcps1.2.4.1.1 "><p id="p6844125874315"><a name="p6844125874315"></a><a name="p6844125874315"></a>srcReg</p>
</td>
<td class="cellrowborder" valign="top" width="10.05%" headers="mcps1.2.4.1.2 "><p id="p128442058144312"><a name="p128442058144312"></a><a name="p128442058144312"></a>输入</p>
</td>
<td class="cellrowborder" valign="top" width="71.41%" headers="mcps1.2.4.1.3 "><p id="p172083541517"><a name="p172083541517"></a><a name="p172083541517"></a>源操作数。</p>
<p id="p17716122843014"><a name="p17716122843014"></a><a name="p17716122843014"></a>类型为<a href="RegTensor.md">RegTensor</a>。</p>
</td>
</tr>
<tr id="row891912431168"><td class="cellrowborder" valign="top" width="18.54%" headers="mcps1.2.4.1.1 "><p id="p195756503168"><a name="p195756503168"></a><a name="p195756503168"></a>scalarValue</p>
</td>
<td class="cellrowborder" valign="top" width="10.05%" headers="mcps1.2.4.1.2 "><p id="p457515071618"><a name="p457515071618"></a><a name="p457515071618"></a>输入</p>
</td>
<td class="cellrowborder" valign="top" width="71.41%" headers="mcps1.2.4.1.3 "><p id="p732824954017"><a name="p732824954017"></a><a name="p732824954017"></a>源操作数。</p>
<p id="p73715114429"><a name="p73715114429"></a><a name="p73715114429"></a>类型为标量。</p>
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

## 调用示例<a name="section642mcpsimp"></a>

```
template<typename T>
__simd_vf__ inline void MulsCastVF(__ubuf__ half* dstAddr, __ubuf__ float* srcAddr, float scalarValue, uint32_t count, uint32_t srcRepeatSize, uint32_t dstRepeatSize, uint16_t repeatTimes)
{
    AscendC::MicroAPI::RegTensor<float> srcReg0;
    AscendC::MicroAPI::RegTensor<half> dstReg0;
    AscendC::MicroAPI::MaskReg mask;
    for (uint16_t i = 0; i < repeatTimes; i++) {
        mask = AscendC::MicroAPI::UpdateMask<float>(count);
        AscendC::MicroAPI::LoadAlign(srcReg0, srcAddr + i * srcRepeatSize);
        AscendC::MicroAPI::MulsCast<half, float, float, AscendC::MicroAPI::RegLayout::ZERO>(dstReg0, srcReg0, scalarValue, mask);
        AscendC::MicroAPI::StoreAlign(dstAddr + i * dstRepeatSize, dstReg0, mask);
    }
}
```

