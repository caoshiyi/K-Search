# MaskDeInterleave<a name="ZH-CN_TOPIC_0000002523304880"></a>

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

将源操作数src0和src1中的元素解交织存入目的操作数dst0和dst1中。解交织排列方式如下图所示，其中每个方格代表一个元素：

[图片缺失: DeInterleave解交织排列示意图]

## 函数原型<a name="section620mcpsimp"></a>

```
template <typename T>
__simd_callee__ inline void MaskDeInterleave(MaskReg& dst0, MaskReg& dst1, MaskReg& src0, MaskReg& src1)
```

## 参数说明<a name="section622mcpsimp"></a>

**表 1**  模板参数说明

<a name="table329841281718"></a>
<table><thead align="left"><tr id="row629871213174"><th class="cellrowborder" valign="top" width="50%" id="mcps1.2.3.1.1"><p id="p62981112131710"><a name="p62981112131710"></a><a name="p62981112131710"></a>参数名</p>
</th>
<th class="cellrowborder" valign="top" width="50%" id="mcps1.2.3.1.2"><p id="p12981512141717"><a name="p12981512141717"></a><a name="p12981512141717"></a>描述</p>
</th>
</tr>
</thead>
<tbody><tr id="row62981812101717"><td class="cellrowborder" valign="top" width="50%" headers="mcps1.2.3.1.1 "><p id="p1929871211179"><a name="p1929871211179"></a><a name="p1929871211179"></a>T</p>
</td>
<td class="cellrowborder" valign="top" width="50%" headers="mcps1.2.3.1.2 "><p id="p7710142515214"><a name="p7710142515214"></a><a name="p7710142515214"></a>MaskReg所支持的数据类型，决定了解交织的位宽大小，例如对于uint32_t类型，解交织时以4bit为一组。</p>
<p id="p3966152216478"><a name="p3966152216478"></a><a name="p3966152216478"></a><span id="ph1966152212477"><a name="ph1966152212477"></a><a name="ph1966152212477"></a>Ascend 950PR/Ascend 950DT</span>支持的数据类型为：b8/b16/b32</p>
</td>
</tr>
</tbody>
</table>

**表 2**  参数说明

<a name="table4835205712588"></a>
<table><thead align="left"><tr id="row118356578583"><th class="cellrowborder" valign="top" width="50%" id="mcps1.2.3.1.1"><p id="p48354572582"><a name="p48354572582"></a><a name="p48354572582"></a>参数名</p>
</th>
<th class="cellrowborder" valign="top" width="50%" id="mcps1.2.3.1.2"><p id="p583535795817"><a name="p583535795817"></a><a name="p583535795817"></a>描述</p>
</th>
</tr>
</thead>
<tbody><tr id="row1835857145817"><td class="cellrowborder" valign="top" width="50%" headers="mcps1.2.3.1.1 "><p id="p5835457165816"><a name="p5835457165816"></a><a name="p5835457165816"></a>dst0</p>
</td>
<td class="cellrowborder" valign="top" width="50%" headers="mcps1.2.3.1.2 "><p id="p168351657155818"><a name="p168351657155818"></a><a name="p168351657155818"></a>目的操作数。</p>
</td>
</tr>
<tr id="row16248184111201"><td class="cellrowborder" valign="top" width="50%" headers="mcps1.2.3.1.1 "><p id="p1064220453202"><a name="p1064220453202"></a><a name="p1064220453202"></a>dst1</p>
</td>
<td class="cellrowborder" valign="top" width="50%" headers="mcps1.2.3.1.2 "><p id="p396713598201"><a name="p396713598201"></a><a name="p396713598201"></a>目的操作数。</p>
</td>
</tr>
<tr id="row18835145716587"><td class="cellrowborder" valign="top" width="50%" headers="mcps1.2.3.1.1 "><p id="p59747391278"><a name="p59747391278"></a><a name="p59747391278"></a>src0</p>
</td>
<td class="cellrowborder" valign="top" width="50%" headers="mcps1.2.3.1.2 "><p id="p53355414286"><a name="p53355414286"></a><a name="p53355414286"></a>源操作数。</p>
</td>
</tr>
<tr id="row2521428183011"><td class="cellrowborder" valign="top" width="50%" headers="mcps1.2.3.1.1 "><p id="p58511219123111"><a name="p58511219123111"></a><a name="p58511219123111"></a>src1</p>
</td>
<td class="cellrowborder" valign="top" width="50%" headers="mcps1.2.3.1.2 "><p id="p175262810308"><a name="p175262810308"></a><a name="p175262810308"></a>源操作数。</p>
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
__simd_vf__ inline void MaskInterleaveDeInterleaveVF(__ubuf__ T* dstAddr, __ubuf__ T* srcAddr, uint32_t count, uint32_t oneRepeatSize, uint16_t repeatTimes)
{
    AscendC::MicroAPI::RegTensor<T> srcReg;
    AscendC::MicroAPI::MaskReg maskFull = AscendC::MicroAPI::CreateMask<T, AscendC::MicroAPI::MaskPattern::ALL>();
    AscendC::MicroAPI::MaskReg maskM3 = AscendC::MicroAPI::CreateMask<T, AscendC::MicroAPI::MaskPattern::M3>();
    AscendC::MicroAPI::MaskReg newMask0;
    AscendC::MicroAPI::MaskReg newMask1;
    AscendC::MicroAPI::MaskInterleave<T>(newMask0, newMask1, maskFull, maskM3);
    AscendC::MicroAPI::MaskDeInterleave<T>(newMask0, newMask1, newMask0, newMask1);
    AscendC::MicroAPI::MaskReg mask;
    for (uint16_t i = 0; i < repeatTimes; ++i) {
        mask = AscendC::MicroAPI::UpdateMask<T>(count);
        AscendC::MicroAPI::LoadAlign(srcReg, srcAddr + i * oneRepeatSize);
        AscendC::MicroAPI::Adds(srcReg, srcReg, 0, newMask0);
        AscendC::MicroAPI::StoreAlign(dstAddr + i * oneRepeatSize, srcReg, mask);
    }
}
```

