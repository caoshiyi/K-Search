# Squeeze<a name="ZH-CN_TOPIC_0000002523343668"></a>

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

将传入的srcReg中被mask选择的有效元素依次复制到dstReg中，有效元素在dstReg中从低到高连续排列。dstReg中剩余位置元素置为0。

## 定义原型<a name="section620mcpsimp"></a>

```
template <typename T = DefaultType, GatherMaskMode store = GatherMaskMode::NO_STORE_REG, typename U>
__simd_callee__ inline void Squeeze(U& dstReg, U& srcReg, MaskReg& mask)
```

## 参数说明<a name="section622mcpsimp"></a>

**表 1**  模板参数说明

<a name="table4835205712588"></a>
<table><thead align="left"><tr id="row118356578583"><th class="cellrowborder" valign="top" width="18.77%" id="mcps1.2.3.1.1"><p id="p48354572582"><a name="p48354572582"></a><a name="p48354572582"></a>参数名</p>
</th>
<th class="cellrowborder" valign="top" width="81.23%" id="mcps1.2.3.1.2"><p id="p583535795817"><a name="p583535795817"></a><a name="p583535795817"></a>描述</p>
</th>
</tr>
</thead>
<tbody><tr id="row1835857145817"><td class="cellrowborder" valign="top" width="18.77%" headers="mcps1.2.3.1.1 "><p id="p5835457165816"><a name="p5835457165816"></a><a name="p5835457165816"></a>T</p>
</td>
<td class="cellrowborder" valign="top" width="81.23%" headers="mcps1.2.3.1.2 "><p id="p168351657155818"><a name="p168351657155818"></a><a name="p168351657155818"></a>目的操作数和源操作数的数据类型。</p>
<p id="p159731458125912"><a name="p159731458125912"></a><a name="p159731458125912"></a><span id="ph1497313580592"><a name="ph1497313580592"></a><a name="ph1497313580592"></a>Ascend 950PR/Ascend 950DT</span>，支持的数据类型为：uint8_t/int8_t/uint16_t/int16_t/uint32_t/int32_t/half/float</p>
</td>
</tr>
<tr id="row3742113618507"><td class="cellrowborder" valign="top" width="18.77%" headers="mcps1.2.3.1.1 "><p id="p1383515717581"><a name="p1383515717581"></a><a name="p1383515717581"></a>store</p>
</td>
<td class="cellrowborder" valign="top" width="81.23%" headers="mcps1.2.3.1.2 "><p id="p77520541653"><a name="p77520541653"></a><a name="p77520541653"></a>GatherMaskMode选择是否将有效元素的总字节数存入AR寄存器，AR寄存器描述参考<a href="GetSpr.md#table4835205712588">表1</a>。</p>
<a name="ul1163765616511"></a><a name="ul1163765616511"></a><ul id="ul1163765616511"><li>NO_STORE_REG，有效元素的总字节数不存入AR寄存器；</li><li>STORE_REG，有效元素的总字节数存入AR寄存器。</li></ul>
</td>
</tr>
<tr id="row104455914519"><td class="cellrowborder" valign="top" width="18.77%" headers="mcps1.2.3.1.1 "><p id="p144175913451"><a name="p144175913451"></a><a name="p144175913451"></a>U</p>
</td>
<td class="cellrowborder" valign="top" width="81.23%" headers="mcps1.2.3.1.2 "><p id="p34418597453"><a name="p34418597453"></a><a name="p34418597453"></a>srcReg/dstReg RegTensor类型， 例如RegTensor&lt;uint32_t&gt;，由编译器自动推导，用户不需要填写。</p>
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
<tbody><tr id="row42461942101815"><td class="cellrowborder" valign="top" width="18.54%" headers="mcps1.2.4.1.1 "><p id="p284425844311"><a name="p284425844311"></a><a name="p284425844311"></a>dstReg</p>
</td>
<td class="cellrowborder" valign="top" width="10.05%" headers="mcps1.2.4.1.2 "><p id="p158449584436"><a name="p158449584436"></a><a name="p158449584436"></a>输出</p>
</td>
<td class="cellrowborder" valign="top" width="71.41%" headers="mcps1.2.4.1.3 "><p id="p6389114616514"><a name="p6389114616514"></a><a name="p6389114616514"></a>目的操作数。</p>
<p id="p1818424418593"><a name="p1818424418593"></a><a name="p1818424418593"></a>类型为<a href="RegTensor.md">RegTensor</a>。</p>
</td>
</tr>
<tr id="row198393197176"><td class="cellrowborder" valign="top" width="18.54%" headers="mcps1.2.4.1.1 "><p id="p1664684418171"><a name="p1664684418171"></a><a name="p1664684418171"></a>srcReg</p>
</td>
<td class="cellrowborder" valign="top" width="10.05%" headers="mcps1.2.4.1.2 "><p id="p9839171971716"><a name="p9839171971716"></a><a name="p9839171971716"></a>输入</p>
</td>
<td class="cellrowborder" valign="top" width="71.41%" headers="mcps1.2.4.1.3 "><p id="p779095312179"><a name="p779095312179"></a><a name="p779095312179"></a>源操作数。</p>
<p id="p97903531171"><a name="p97903531171"></a><a name="p97903531171"></a>类型为<a href="RegTensor.md">RegTensor</a></p>
</td>
</tr>
<tr id="row17332152321719"><td class="cellrowborder" valign="top" width="18.54%" headers="mcps1.2.4.1.1 "><p id="p1538745616196"><a name="p1538745616196"></a><a name="p1538745616196"></a>mask</p>
</td>
<td class="cellrowborder" valign="top" width="10.05%" headers="mcps1.2.4.1.2 "><p id="p73331523101710"><a name="p73331523101710"></a><a name="p73331523101710"></a>输入</p>
</td>
<td class="cellrowborder" valign="top" width="71.41%" headers="mcps1.2.4.1.3 "><p id="p1559991025517"><a name="p1559991025517"></a><a name="p1559991025517"></a>mask用于控制每次迭代内参与计算的元素。</p>
</td>
</tr>
</tbody>
</table>

## 约束说明<a name="section177921451558"></a>

当store取值为STORE\_REG时，由于硬件约束，StoreUnAlign指令和Squeeze指令必须交替使用，例如：

```
Squeeze(dstVreg, srcVreg, mask);
StoreUnAlign(dstAddr, dstVreg, ureg);
Squeeze(dstVreg, srcVreg, mask);
StoreUnAlign(dstAddr, dstVreg, ureg);
```

## 调用示例<a name="section642mcpsimp"></a>

```
template<typename T>
__simd_vf__ inline void SqueezeVF(__ubuf__ T* dstAddr, __ubuf__ T* srcAddr, uint32_t oneRepeatSize, uint16_t repeatTimes)
{
    AscendC::MicroAPI::RegTensor<T> srcReg0;
    AscendC::MicroAPI::RegTensor<T> srcReg1;
    AscendC::MicroAPI::UnalignRegForStore ureg;
    AscendC::MicroAPI::MaskReg sqzMask = AscendC::MicroAPI::CreateMask<T, AscendC::MicroAPI::MaskPattern::H>();
    for (uint16_t i = 0; i < repeatTimes; ++i) {
        AscendC::MicroAPI::LoadAlign<T, AscendC::MicroAPI::PostLiteral::POST_MODE_UPDATE>(srcReg0, srcAddr, oneRepeatSize);
        AscendC::MicroAPI::Squeeze<T, AscendC::MicroAPI::GatherMaskMode::STORE_REG>(srcReg1, srcReg0, sqzMask);
        AscendC::MicroAPI::StoreUnAlign<T, AscendC::MicroAPI::PostLiteral::POST_MODE_UPDATE>(dstAddr, srcReg1, ureg);
     }
     AscendC::MicroAPI::StoreUnAlignPost(dstAddr, ureg);
}
```

