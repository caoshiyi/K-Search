# Pack<a name="ZH-CN_TOPIC_0000002554423839"></a>

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

根据所选的低位模式或高位模式，将输入MaskReg的偶数位bit提取到输出MaskReg的低半部分或高半部分。

## 函数原型<a name="section620mcpsimp"></a>

```
template <HighLowPart part = HighLowPart::LOWEST> 
__simd_callee__ inline void Pack(MaskReg& dst, MaskReg& src)
```

## 参数说明<a name="section622mcpsimp"></a>

**表 1**  模板参数说明

<a name="table4835205712588"></a>
<table><thead align="left"><tr id="row118356578583"><th class="cellrowborder" valign="top" width="50%" id="mcps1.2.3.1.1"><p id="p48354572582"><a name="p48354572582"></a><a name="p48354572582"></a>参数名</p>
</th>
<th class="cellrowborder" valign="top" width="50%" id="mcps1.2.3.1.2"><p id="p583535795817"><a name="p583535795817"></a><a name="p583535795817"></a>描述</p>
</th>
</tr>
</thead>
<tbody><tr id="row1835857145817"><td class="cellrowborder" valign="top" width="50%" headers="mcps1.2.3.1.1 "><p id="p5835457165816"><a name="p5835457165816"></a><a name="p5835457165816"></a>part</p>
</td>
<td class="cellrowborder" valign="top" width="50%" headers="mcps1.2.3.1.2 "><p id="p168351657155818"><a name="p168351657155818"></a><a name="p168351657155818"></a>枚举类型，低位模式或高位模式。</p>
<a name="ul1188693111513"></a><a name="ul1188693111513"></a><ul id="ul1188693111513"><li>LOWEST  低位模式；</li><li>HIGHEST 高位模式。</li></ul>
</td>
</tr>
</tbody>
</table>

**表 2**  参数说明

<a name="table299395481215"></a>
<table><thead align="left"><tr id="row1399413543129"><th class="cellrowborder" valign="top" width="50%" id="mcps1.2.3.1.1"><p id="p14994175441217"><a name="p14994175441217"></a><a name="p14994175441217"></a>参数名</p>
</th>
<th class="cellrowborder" valign="top" width="50%" id="mcps1.2.3.1.2"><p id="p999420547126"><a name="p999420547126"></a><a name="p999420547126"></a>描述</p>
</th>
</tr>
</thead>
<tbody><tr id="row1399415546125"><td class="cellrowborder" valign="top" width="50%" headers="mcps1.2.3.1.1 "><p id="p12994155416123"><a name="p12994155416123"></a><a name="p12994155416123"></a>dst</p>
</td>
<td class="cellrowborder" valign="top" width="50%" headers="mcps1.2.3.1.2 "><p id="p159941254191217"><a name="p159941254191217"></a><a name="p159941254191217"></a>目的操作数。</p>
</td>
</tr>
<tr id="row49941154111217"><td class="cellrowborder" valign="top" width="50%" headers="mcps1.2.3.1.1 "><p id="p6994354131210"><a name="p6994354131210"></a><a name="p6994354131210"></a>src</p>
</td>
<td class="cellrowborder" valign="top" width="50%" headers="mcps1.2.3.1.2 "><p id="p1399425410128"><a name="p1399425410128"></a><a name="p1399425410128"></a>源操作数。</p>
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
template <typename T>
__simd_vf__ inline void PackVF(__ubuf__ T* dstAddr, __ubuf__ T* srcAddr, uint32_t count, uint32_t oneRepeatSize, uint16_t repeatTimes)
{
    AscendC::MicroAPI::RegTensor<T> srcReg;
    AscendC::MicroAPI::MaskReg maskFull = AscendC::MicroAPI::CreateMask<T, AscendC::MicroAPI::MaskPattern::ALL>();
    AscendC::MicroAPI::MaskReg mask0;
    AscendC::MicroAPI::MaskReg mask1;
    AscendC::MicroAPI::Pack<AscendC::MicroAPI::HighLowPart::LOWEST>(mask0, maskFull);
    for (uint16_t i = 0; i < repeatTimes; ++i) {
        mask1 = AscendC::MicroAPI::UpdateMask<T>(count);
        AscendC::MicroAPI::LoadAlign(srcReg, srcAddr + i * oneRepeatSize);
        AscendC::MicroAPI::Adds(srcReg, srcReg, 0, mask0);
        AscendC::MicroAPI::StoreAlign(dstAddr + i * oneRepeatSize, srcReg, mask1);
    }
}
```

