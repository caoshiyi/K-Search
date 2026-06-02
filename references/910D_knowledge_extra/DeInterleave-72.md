# DeInterleave<a name="ZH-CN_TOPIC_0000002523344020"></a>

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

给定源操作数寄存器张量srcReg0和srcReg1，将srcReg0和srcReg1中的元素解交织存入结果操作数dstReg0和dstReg1中。解交织排列方式如下图所示，其中每个方格代表一个元素：

<!-- img2text -->
```text
srcReg0
┌───┬───┬───┬───┬─────┬───┐
│ 0 │ 0 │ 1 │ 1 │ ... │ 3 │
└───┴───┴───┴───┴─────┴───┘

srcReg1
┌───┬───┬───┬───┬─────┬───┐
│ 4 │ 4 │ 5 │ 5 │ ... │ 7 │
└───┴───┴───┴───┴─────┴───┘

                 ↓

dstReg0                               dstReg1
┌───┬───┬───┬───┬─────┬───┐           ┌───┬───┬───┬───┬─────┬───┐
│ 0 │ 1 │ 2 │ 3 │ ... │ 7 │           │ 0 │ 1 │ 2 │ 3 │ ... │ 7 │
└───┴───┴───┴───┴─────┴───┘           └───┴───┴───┴───┴─────┴───┘

            lower part                           higher part
```

## 定义原型<a name="section620mcpsimp"></a>

```
template <typename T = DefaultType, typename U>
__simd_callee__ inline void DeInterleave(U& dstReg0, U& dstReg1, U& srcReg0, U& srcReg1)
```

## 参数说明<a name="section622mcpsimp"></a>

**表 1**  模板参数说明

<a name="table4835205712588"></a>
<table><thead align="left"><tr id="row118356578583"><th class="cellrowborder" valign="top" width="18.529999999999998%" id="mcps1.2.3.1.1"><p id="p48354572582"><a name="p48354572582"></a><a name="p48354572582"></a>参数名</p>
</th>
<th class="cellrowborder" valign="top" width="81.47%" id="mcps1.2.3.1.2"><p id="p583535795817"><a name="p583535795817"></a><a name="p583535795817"></a>描述</p>
</th>
</tr>
</thead>
<tbody><tr id="row1835857145817"><td class="cellrowborder" valign="top" width="18.529999999999998%" headers="mcps1.2.3.1.1 "><p id="p5835457165816"><a name="p5835457165816"></a><a name="p5835457165816"></a>T</p>
</td>
<td class="cellrowborder" valign="top" width="81.47%" headers="mcps1.2.3.1.2 "><p id="p168351657155818"><a name="p168351657155818"></a><a name="p168351657155818"></a>目的操作数和源的数据类型。</p>
<p id="p19741733173613"><a name="p19741733173613"></a><a name="p19741733173613"></a><span id="ph1574633193611"><a name="ph1574633193611"></a><a name="ph1574633193611"></a>Ascend 950PR/Ascend 950DT</span>，支持的数据类型为：bool/uint8_t/int8_t/uint16_t/int16_t/uint32_t/int32_t/uint64_t/int64_t/half/float/bfloat16_t</p>
</td>
</tr>
<tr id="row2424150175914"><td class="cellrowborder" valign="top" width="18.529999999999998%" headers="mcps1.2.3.1.1 "><p id="p916243141912"><a name="p916243141912"></a><a name="p916243141912"></a>U</p>
</td>
<td class="cellrowborder" valign="top" width="81.47%" headers="mcps1.2.3.1.2 "><p id="p131047476612"><a name="p131047476612"></a><a name="p131047476612"></a>源操作数和目的操作数的RegTensor类型，例如RegTensor&lt;half&gt;，由编译器自动推导，用户不需要填写。</p>
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
<tbody><tr id="row42461942101815"><td class="cellrowborder" valign="top" width="18.54%" headers="mcps1.2.4.1.1 "><p id="p284425844311"><a name="p284425844311"></a><a name="p284425844311"></a>dstReg0、dstReg1</p>
</td>
<td class="cellrowborder" valign="top" width="10.05%" headers="mcps1.2.4.1.2 "><p id="p158449584436"><a name="p158449584436"></a><a name="p158449584436"></a>输出</p>
</td>
<td class="cellrowborder" valign="top" width="71.41%" headers="mcps1.2.4.1.3 "><p id="p6389114616514"><a name="p6389114616514"></a><a name="p6389114616514"></a>目的操作数。</p>
<p id="p18220202613566"><a name="p18220202613566"></a><a name="p18220202613566"></a>类型为<a href="RegTensor.md">RegTensor</a>。</p>
</td>
</tr>
<tr id="row2137145181815"><td class="cellrowborder" valign="top" width="18.54%" headers="mcps1.2.4.1.1 "><p id="p6844125874315"><a name="p6844125874315"></a><a name="p6844125874315"></a>srcReg0、srcReg1</p>
</td>
<td class="cellrowborder" valign="top" width="10.05%" headers="mcps1.2.4.1.2 "><p id="p128442058144312"><a name="p128442058144312"></a><a name="p128442058144312"></a>输入</p>
</td>
<td class="cellrowborder" valign="top" width="71.41%" headers="mcps1.2.4.1.3 "><p id="p9761550745"><a name="p9761550745"></a><a name="p9761550745"></a>源操作数。</p>
<p id="p7484358103819"><a name="p7484358103819"></a><a name="p7484358103819"></a>类型为<a href="RegTensor.md">RegTensor</a>。</p>
<p id="p37015312238"><a name="p37015312238"></a><a name="p37015312238"></a>源操作数的数据类型需要与目的操作数保持一致。</p>
</td>
</tr>
</tbody>
</table>

## 约束说明<a name="section1240634052310"></a>

-   b64数据类型下仅支持RegTraitNumTwo。

## 调用示例<a name="section642mcpsimp"></a>

```
template<typename T>
__simd_vf__ inline void DeInterLeaveVF(__ubuf__ T* dst0Addr, __ubuf__ T* dst1Addr, __ubuf__ T* src0Addr, __ubuf__ T* src1Addr, uint32_t oneRepeatSize, uint16_t repeatTimes)
{
    AscendC::MicroAPI::RegTensor<T> srcReg0;
    AscendC::MicroAPI::RegTensor<T> srcReg1;
    AscendC::MicroAPI::RegTensor<T> dstReg0;
    AscendC::MicroAPI::RegTensor<T> dstReg1;
    AscendC::MicroAPI::MaskReg mask = AscendC::MicroAPI::CreateMask<T, AscendC::MicroAPI::MaskPattern::ALL>();
    for (uint16_t i = 0; i < repeatTimes; i++) {
        AscendC::MicroAPI::LoadAlign(srcReg0, src0Addr + i * oneRepeatSize);
        AscendC::MicroAPI::LoadAlign(srcReg1, src1Addr + i * oneRepeatSize);
        AscendC::MicroAPI::DeInterleave(dstReg0, dstReg1, srcReg0, srcReg1);
        AscendC::MicroAPI::StoreAlign(dst0Addr + i * oneRepeatSize, dstReg0, mask);
        AscendC::MicroAPI::StoreAlign(dst1Addr + i * oneRepeatSize, dstReg1, mask);
    }
}
```

