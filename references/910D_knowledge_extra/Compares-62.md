# Compares<a name="ZH-CN_TOPIC_0000002523344590"></a>

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

逐元素比较一个Tensor中的元素和另一个scalar的大小，如果比较后的结果为真，则输出结果的对应比特位为1，否则为0。

## 函数原型<a name="section620mcpsimp"></a>

```
template <typename T = DefaultType, CMPMODE mode = CMPMODE::EQ, typename U, typename S>
__simd_callee__ inline void Compares(MaskReg& dst, U& srcReg, S scalarValue, MaskReg& mask)
```

## 参数说明<a name="section622mcpsimp"></a>

**表 1**  模板参数说明

<a name="table4835205712588"></a>
<table><thead align="left"><tr id="row118356578583"><th class="cellrowborder" valign="top" width="14.95%" id="mcps1.2.3.1.1"><p id="p48354572582"><a name="p48354572582"></a><a name="p48354572582"></a>参数名</p>
</th>
<th class="cellrowborder" valign="top" width="85.05%" id="mcps1.2.3.1.2"><p id="p583535795817"><a name="p583535795817"></a><a name="p583535795817"></a>描述</p>
</th>
</tr>
</thead>
<tbody><tr id="row1835857145817"><td class="cellrowborder" valign="top" width="14.95%" headers="mcps1.2.3.1.1 "><p id="p5835457165816"><a name="p5835457165816"></a><a name="p5835457165816"></a>T</p>
</td>
<td class="cellrowborder" valign="top" width="85.05%" headers="mcps1.2.3.1.2 "><p id="p168351657155818"><a name="p168351657155818"></a><a name="p168351657155818"></a>矢量源操作数数据类型。</p>
<p id="p12347511192513"><a name="p12347511192513"></a><a name="p12347511192513"></a><span id="ph434701113253"><a name="ph434701113253"></a><a name="ph434701113253"></a>Ascend 950PR/Ascend 950DT</span>，支持的数据类型为：int8_t、uint8_t、int16_t、uint16_t、half、bfloat16_t、int32_t、uint32_t、float、int64_t、uint64_t。</p>
</td>
</tr>
<tr id="row18835145716587"><td class="cellrowborder" valign="top" width="14.95%" headers="mcps1.2.3.1.1 "><p id="p464337182920"><a name="p464337182920"></a><a name="p464337182920"></a>mode</p>
</td>
<td class="cellrowborder" valign="top" width="85.05%" headers="mcps1.2.3.1.2 "><p id="p025716534299"><a name="p025716534299"></a><a name="p025716534299"></a>比较模式。支持如下取值：</p>
<a name="ul14429497516"></a><a name="ul14429497516"></a><ul id="ul14429497516"><li>LT：小于（less than）</li><li>GT：大于（greater than）</li></ul>
<a name="ul147379458500"></a><a name="ul147379458500"></a><ul id="ul147379458500"><li>GE：大于或等于（greater than or equal to）</li><li>EQ：等于（equal to）</li><li>NE：不等于（not equal to）</li><li>LE：小于或等于（less than or equal to）</li></ul>
</td>
</tr>
<tr id="row962916316272"><td class="cellrowborder" valign="top" width="14.95%" headers="mcps1.2.3.1.1 "><p id="p916243141912"><a name="p916243141912"></a><a name="p916243141912"></a>U</p>
</td>
<td class="cellrowborder" valign="top" width="85.05%" headers="mcps1.2.3.1.2 "><p id="p47566332405"><a name="p47566332405"></a><a name="p47566332405"></a>源操作数的RegTensor类型，由编译器自动推导，用户不需要填写。</p>
</td>
</tr>
<tr id="row3625143218279"><td class="cellrowborder" valign="top" width="14.95%" headers="mcps1.2.3.1.1 "><p id="p1062543242714"><a name="p1062543242714"></a><a name="p1062543242714"></a>S</p>
</td>
<td class="cellrowborder" valign="top" width="85.05%" headers="mcps1.2.3.1.2 "><p id="p16251432182715"><a name="p16251432182715"></a><a name="p16251432182715"></a>scalar数据类型。</p>
<p id="p2711145253"><a name="p2711145253"></a><a name="p2711145253"></a><span id="ph27414102512"><a name="ph27414102512"></a><a name="ph27414102512"></a>Ascend 950PR/Ascend 950DT</span>，支持的数据类型为：uint8_t/int8_t/uint16_t/int16_t/uint32_t/int32_t/half/float/bfloat16_t/uint64_t/int64_t</p>
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
<tbody><tr id="row42461942101815"><td class="cellrowborder" valign="top" width="18.54%" headers="mcps1.2.4.1.1 "><p id="p284425844311"><a name="p284425844311"></a><a name="p284425844311"></a>dst</p>
</td>
<td class="cellrowborder" valign="top" width="10.05%" headers="mcps1.2.4.1.2 "><p id="p158449584436"><a name="p158449584436"></a><a name="p158449584436"></a>输出</p>
</td>
<td class="cellrowborder" valign="top" width="71.41%" headers="mcps1.2.4.1.3 "><p id="p159941254191217"><a name="p159941254191217"></a><a name="p159941254191217"></a>MaskReg类型，目的操作数。</p>
</td>
</tr>
<tr id="row2137145181815"><td class="cellrowborder" valign="top" width="18.54%" headers="mcps1.2.4.1.1 "><p id="p6844125874315"><a name="p6844125874315"></a><a name="p6844125874315"></a>srcReg</p>
</td>
<td class="cellrowborder" valign="top" width="10.05%" headers="mcps1.2.4.1.2 "><p id="p128442058144312"><a name="p128442058144312"></a><a name="p128442058144312"></a>输入</p>
</td>
<td class="cellrowborder" valign="top" width="71.41%" headers="mcps1.2.4.1.3 "><p id="p172083541517"><a name="p172083541517"></a><a name="p172083541517"></a>源操作数。</p>
<p id="p5261335123513"><a name="p5261335123513"></a><a name="p5261335123513"></a><span id="ph15261035183514"><a name="ph15261035183514"></a><a name="ph15261035183514"></a>类型为<a href="RegTensor.md">RegTensor</a>。</span></p>
</td>
</tr>
<tr id="row944414143710"><td class="cellrowborder" valign="top" width="18.54%" headers="mcps1.2.4.1.1 "><p id="p1059534613404"><a name="p1059534613404"></a><a name="p1059534613404"></a>scalarValue</p>
</td>
<td class="cellrowborder" valign="top" width="10.05%" headers="mcps1.2.4.1.2 "><p id="p14444941113712"><a name="p14444941113712"></a><a name="p14444941113712"></a>输入</p>
</td>
<td class="cellrowborder" valign="top" width="71.41%" headers="mcps1.2.4.1.3 "><p id="p13444154119374"><a name="p13444154119374"></a><a name="p13444154119374"></a>源操作数。</p>
<p id="p11311221910"><a name="p11311221910"></a><a name="p11311221910"></a>类型为标量。</p>
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

## 调用示例<a name="section12223121385116"></a>

```
template<typename T>
__simd_vf__ inline void ComparesVF(__ubuf__ T* dstAddr, __ubuf__ T* src0Addr, __ubuf__ T* src1Addr, T scalarValue, uint32_t count, uint32_t oneRepeatSize, uint16_t repeatTimes)
{
    AscendC::MicroAPI::RegTensor<T> srcReg0;
    AscendC::MicroAPI::RegTensor<T> srcReg1;
    AscendC::MicroAPI::RegTensor<T> dstReg;
    AscendC::MicroAPI::MaskReg mask;
    AscendC::MicroAPI::MaskReg cmpMaskReg;
    for (uint16_t i = 0; i < repeatTimes; i++) {
        mask = AscendC::MicroAPI::UpdateMask<T>(count);
        AscendC::MicroAPI::LoadAlign(srcReg0, src0Addr + i * oneRepeatSize);
        AscendC::MicroAPI::LoadAlign(srcReg1, src1Addr + i * oneRepeatSize);       
        AscendC::MicroAPI::Compares<T, AscendC::CMPMODE::EQ>(cmpMaskReg, srcReg0, scalarValue, mask);
        AscendC::MicroAPI::Select(dstReg, srcReg0, srcReg1, cmpMaskReg);
        AscendC::MicroAPI::StoreAlign(dstAddr + i * oneRepeatSize, dstReg, mask);
    }
}
```

