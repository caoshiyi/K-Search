# AddC<a name="ZH-CN_TOPIC_0000002554423661"></a>

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

根据mask对输入数据src0、src1以及进位数据carrySrc进行按元素相加操作，将结果写入dst。如果src0, src1输入转换为uint32\_t类型，加上进位值carrySrc相加时超出uint32\_t最大值，在MaskReg carry中对应位置每4bit大小写1，否则写0。计算公式如下：

<!-- img2text -->
$$
dst_i = src0_i + src1_i + carrySrc_i
$$

$$
carry_i =
\begin{cases}
1, & (uint32\_t)src0_i + (uint32\_t)src1_i + carrySrc_i > UINT32\_MAX \\
0, & \text{others}
\end{cases}
$$

对carry的操作示例如下：

-   int32\_t类型，a\_i, b\_i∈\[-2147483648, 2147483647\]
    -   假设a\_i = -2147483648, b\_i = -2, carrySrc\_i = 1

        \(uint32\_t\)a\_i + \(uint32\_t\)b\_i + \(uint32\_t\)carrySrc\_i = \(uint64\_t\)uint\_dst\_i

        因为uint\_dst\_i \>\> 32 大于0，所以carry\[\(i%64\):4\*\(i%64\)\] = 1

    -   假设a\_i = 2, b\_i = 5, carrySrc\_i = 1

        \(uint32\_t\)a\_i + \(uint32\_t\)b\_i + \(uint32\_t\)carrySrc\_i = \(uint64\_t\)uint\_dst\_i

        因为uint\_dst\_i \>\> 32 等于0，所以carry\[\(i%64\):4\*\(i%64\)\] = 0

-   uint32\_t类型，a\_i, b\_i∈\[0, 4294967295\]
    -   假设a\_i = 4294967295, b\_i = 2, carrySrc\_i = 0

        \(uint32\_t\)a\_i + \(uint32\_t\)b\_i + \(uint32\_t\)carrySrc\_i = \(uint64\_t\)uint\_dst\_i

        因为uint\_dst\_i \>\> 32 大于0，所以carry\[\(i%64\):4\*\(i%64\)\] = 1

    -   假设a\_i = 3, b\_i = 2, carrySrc\_i = 0

        \(uint32\_t\)a\_i + \(uint32\_t\)b\_i + \(uint32\_t\)carrySrc\_i = \(uint64\_t\)uint\_dst\_i

        因为uint\_dst\_i \>\> 32 等于0，所以carry\[\(i%64\):4\*\(i%64\)\] = 0

## 函数原型<a name="section620mcpsimp"></a>

```
template <typename T = DefaultType, typename U>
__simd_callee__ inline void AddC(MaskReg& carry, U& dstReg, U& srcReg0, U& srcReg1, MaskReg& carrySrc, MaskReg& mask)
```

## 参数说明<a name="section622mcpsimp"></a>

**表 1**  模板参数说明

<a name="table4835205712588"></a>
<table><thead align="left"><tr id="row118356578583"><th class="cellrowborder" valign="top" width="18.56%" id="mcps1.2.3.1.1"><p id="p48354572582"><a name="p48354572582"></a><a name="p48354572582"></a>参数名</p>
</th>
<th class="cellrowborder" valign="top" width="81.44%" id="mcps1.2.3.1.2"><p id="p583535795817"><a name="p583535795817"></a><a name="p583535795817"></a>描述</p>
</th>
</tr>
</thead>
<tbody><tr id="row1835857145817"><td class="cellrowborder" valign="top" width="18.56%" headers="mcps1.2.3.1.1 "><p id="p5835457165816"><a name="p5835457165816"></a><a name="p5835457165816"></a>T</p>
</td>
<td class="cellrowborder" valign="top" width="81.44%" headers="mcps1.2.3.1.2 "><p id="p168351657155818"><a name="p168351657155818"></a><a name="p168351657155818"></a>操作数数据类型。</p>
<p id="p1696610389284"><a name="p1696610389284"></a><a name="p1696610389284"></a><span id="ph9966153816285"><a name="ph9966153816285"></a><a name="ph9966153816285"></a>Ascend 950PR/Ascend 950DT</span>，支持的数据类型为：uint32_t/int32_t。</p>
</td>
</tr>
<tr id="row103041482169"><td class="cellrowborder" valign="top" width="18.56%" headers="mcps1.2.3.1.1 "><p id="p430416488162"><a name="p430416488162"></a><a name="p430416488162"></a>U</p>
</td>
<td class="cellrowborder" valign="top" width="81.44%" headers="mcps1.2.3.1.2 "><p id="p1330454851613"><a name="p1330454851613"></a><a name="p1330454851613"></a>操作数RegTensor类型， 例如RegTensor&lt;uint32_t&gt;，由编译器自动推导，用户不需要填写。</p>
</td>
</tr>
</tbody>
</table>

**表 2**  参数说明

<a name="zh-cn_topic_0235751031_table33761356"></a>
<table><thead align="left"><tr id="zh-cn_topic_0235751031_row27598891"><th class="cellrowborder" valign="top" width="18.54%" id="mcps1.2.4.1.1"><p id="zh-cn_topic_0235751031_p20917673"><a name="zh-cn_topic_0235751031_p20917673"></a><a name="zh-cn_topic_0235751031_p20917673"></a>参数名</p>
</th>
<th class="cellrowborder" valign="top" width="10.040000000000001%" id="mcps1.2.4.1.2"><p id="zh-cn_topic_0235751031_p16609919"><a name="zh-cn_topic_0235751031_p16609919"></a><a name="zh-cn_topic_0235751031_p16609919"></a>输入/输出</p>
</th>
<th class="cellrowborder" valign="top" width="71.41999999999999%" id="mcps1.2.4.1.3"><p id="zh-cn_topic_0235751031_p59995477"><a name="zh-cn_topic_0235751031_p59995477"></a><a name="zh-cn_topic_0235751031_p59995477"></a>描述</p>
</th>
</tr>
</thead>
<tbody><tr id="row1685815326526"><td class="cellrowborder" valign="top" width="18.54%" headers="mcps1.2.4.1.1 "><p id="p1085873217524"><a name="p1085873217524"></a><a name="p1085873217524"></a>carry</p>
</td>
<td class="cellrowborder" valign="top" width="10.040000000000001%" headers="mcps1.2.4.1.2 "><p id="p18621102165312"><a name="p18621102165312"></a><a name="p18621102165312"></a>输出</p>
</td>
<td class="cellrowborder" valign="top" width="71.41999999999999%" headers="mcps1.2.4.1.3 "><p id="p682115465311"><a name="p682115465311"></a><a name="p682115465311"></a>目的操作数</p>
<p id="p11809617145318"><a name="p11809617145318"></a><a name="p11809617145318"></a>类型为<a href="MaskReg.md">MaskReg</a>。</p>
</td>
</tr>
<tr id="row42461942101815"><td class="cellrowborder" valign="top" width="18.54%" headers="mcps1.2.4.1.1 "><p id="p108051250181214"><a name="p108051250181214"></a><a name="p108051250181214"></a>dstReg</p>
</td>
<td class="cellrowborder" valign="top" width="10.040000000000001%" headers="mcps1.2.4.1.2 "><p id="p158449584436"><a name="p158449584436"></a><a name="p158449584436"></a>输出</p>
</td>
<td class="cellrowborder" valign="top" width="71.41999999999999%" headers="mcps1.2.4.1.3 "><p id="p6389114616514"><a name="p6389114616514"></a><a name="p6389114616514"></a>目的操作数。</p>
<p id="p66093533169"><a name="p66093533169"></a><a name="p66093533169"></a>类型为<a href="RegTensor.md">RegTensor</a>。</p>
</td>
</tr>
<tr id="row2137145181815"><td class="cellrowborder" valign="top" width="18.54%" headers="mcps1.2.4.1.1 "><p id="p19574165615129"><a name="p19574165615129"></a><a name="p19574165615129"></a>srcReg0</p>
</td>
<td class="cellrowborder" valign="top" width="10.040000000000001%" headers="mcps1.2.4.1.2 "><p id="p128442058144312"><a name="p128442058144312"></a><a name="p128442058144312"></a>输入</p>
</td>
<td class="cellrowborder" valign="top" width="71.41999999999999%" headers="mcps1.2.4.1.3 "><p id="p172083541517"><a name="p172083541517"></a><a name="p172083541517"></a>源操作数。</p>
<p id="p7123111612517"><a name="p7123111612517"></a><a name="p7123111612517"></a>类型为<a href="RegTensor.md">RegTensor</a>。</p>
<p id="p1484485824312"><a name="p1484485824312"></a><a name="p1484485824312"></a>三个源操作数的数据类型需要与目的操作数保持一致。</p>
</td>
</tr>
<tr id="row11773440341"><td class="cellrowborder" valign="top" width="18.54%" headers="mcps1.2.4.1.1 "><p id="p10132448173420"><a name="p10132448173420"></a><a name="p10132448173420"></a>srcReg1</p>
</td>
<td class="cellrowborder" valign="top" width="10.040000000000001%" headers="mcps1.2.4.1.2 "><p id="p151325484342"><a name="p151325484342"></a><a name="p151325484342"></a>输入</p>
</td>
<td class="cellrowborder" valign="top" width="71.41999999999999%" headers="mcps1.2.4.1.3 "><p id="p181326485341"><a name="p181326485341"></a><a name="p181326485341"></a>源操作数。</p>
<p id="p813204817342"><a name="p813204817342"></a><a name="p813204817342"></a>类型为<a href="RegTensor.md">RegTensor</a>。</p>
<p id="p12132154853419"><a name="p12132154853419"></a><a name="p12132154853419"></a>三个源操作数的数据类型需要与目的操作数保持一致。</p>
</td>
</tr>
<tr id="row574413121111"><td class="cellrowborder" valign="top" width="18.54%" headers="mcps1.2.4.1.1 "><p id="p14751813191114"><a name="p14751813191114"></a><a name="p14751813191114"></a>carrySrc</p>
</td>
<td class="cellrowborder" valign="top" width="10.040000000000001%" headers="mcps1.2.4.1.2 "><p id="p137521315114"><a name="p137521315114"></a><a name="p137521315114"></a>输入</p>
</td>
<td class="cellrowborder" valign="top" width="71.41999999999999%" headers="mcps1.2.4.1.3 "><p id="p075513101113"><a name="p075513101113"></a><a name="p075513101113"></a>源操作数。输入进位值。</p>
<p id="p1149514185128"><a name="p1149514185128"></a><a name="p1149514185128"></a>类型为<a href="MaskReg.md">MaskReg</a>。</p>
<p id="p0912849161314"><a name="p0912849161314"></a><a name="p0912849161314"></a>三个源操作数的数据类型需要与目的操作数保持一致。</p>
<p id="p15433203914273"><a name="p15433203914273"></a><a name="p15433203914273"></a><span id="ph643353912711"><a name="ph643353912711"></a><a name="ph643353912711"></a>Ascend 950PR/Ascend 950DT</span>，支持的数据类型为：uint32_t/int32_t。</p>
</td>
</tr>
<tr id="row19615183817191"><td class="cellrowborder" valign="top" width="18.54%" headers="mcps1.2.4.1.1 "><p id="p1484519586432"><a name="p1484519586432"></a><a name="p1484519586432"></a>mask</p>
</td>
<td class="cellrowborder" valign="top" width="10.040000000000001%" headers="mcps1.2.4.1.2 "><p id="p484514581433"><a name="p484514581433"></a><a name="p484514581433"></a>输入</p>
</td>
<td class="cellrowborder" valign="top" width="71.41999999999999%" headers="mcps1.2.4.1.3 "><p id="p11541143920"><a name="p11541143920"></a><a name="p11541143920"></a><span id="ph15776181222"><a name="ph15776181222"></a><a name="ph15776181222"></a>源操作数元素操作的有效指示，详细说明请参考<a href="MaskReg.md">MaskReg</a>。</span></p>
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
template <typename T>
static __simd_vf__ inline void AddVF(__ubuf__ T* dst0Addr, __ubuf__ T* dst1Addr, __ubuf__ T* src0Addr, __ubuf__ T* src1Addr, uint32_t count, uint16_t repeatTimes, uint32_t oneRepeatSize){
    
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
        AscendC::MicroAPI::Add(carry, dstReg0, srcReg0, srcReg1, mask);
        // 8*4B=32B align
        AscendC::MicroAPI::StoreAlign<uint32_t, AscendC::MicroAPI::MaskDist::DIST_NORM>((__ubuf__ uint32_t*)dst1Addr + i * 8, carry);
        AscendC::MicroAPI::StoreAlign(dst0Addr + i * oneRepeatSize, dstReg0, mask);
    }
}
```

