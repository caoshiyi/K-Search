# GatherB<a name="ZH-CN_TOPIC_0000002523304092"></a>

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

给定源操作数在UB中的基地址和索引，GatherB指令根据索引位置将源操作数按DataBlock收集到结果寄存器张量中。每个DataBlock长度为32Byte。收集过程如下图所示：

<!-- img2text -->
```text
Index
┌─────────┬─────────┬─────────┬─────────┬─────────┬─────────┬─────────┬─────────┐
│ Index 0 │ Index 1 │ Index 2 │ Index 3 │ Index 4 │ Index 5 │ Index 6 │ Index 7 │
└─────────┴─────────┴─────────┴─────────┴─────────┴─────────┴─────────┴─────────┘
                                        │
                                        ▼
                                  ┌──────────┐
                                  │    +     │  baseAddr
                                  └──────────┘
                                        │
                                        ▼

Addr
┌─────────┬─────────┬─────────┬─────────┬─────────┬─────────┬─────────┬─────────┐
│ Addr 0  │ Addr 1  │ Addr 2  │ Addr 3  │ Addr 4  │ Addr 5  │ Addr 6  │ Addr 7  │
└────┬────┴────┬────┴────┬────┴────┬────┴────┬────┴────┬────┴────┬────┴────┬────┘
     │         │         │         │         │         │         │
     │         │         │         │         │         │         │
     │         │         │         │         │         │         │

Src
┌──────────────────────────────────────────────────────────────────────────────────────────────┐
│                                            UB空间                                            │
│                                                                                              │
│      ┌─────────┐           ┌─────────┐           ┌─────────┐           ┌─────────┐          │
│      │ Block 2 │           │ Block 0 │           │ Block 3 │           │ Block 6 │          │
│      └─────────┘           └─────────┘           └─────────┘           └─────────┘          │
│                                                                                              │
│      ┌─────────┐           ┌─────────┐           ┌─────────┐           ┌─────────┐          │
│      │ Block 4 │           │ Block 1 │           │ Block 5 │           │ Block 7 │          │
│      └─────────┘           └─────────┘           └─────────┘           └─────────┘          │
└──────────────────────────────────────────────────────────────────────────────────────────────┘

Addr 到 Src 的对应关系
Addr 0 ─────────────→ Block 2
Addr 1 ─────────────→ Block 1
Addr 2 ─────────────→ Block 4
Addr 3 ─────────────→ Block 0
Addr 4 ─────────────→ Block 3
Addr 5 ─────────────→ Block 5
Addr 6 ─────────────→ Block 6
Addr 7 ─────────────→ Block 7

                                        │
                                        ▼
                                      GatherB
                                        │
                                        ▼

Dst
┌─────────┬─────────┬─────────┬─────────┬─────────┬─────────┬─────────┬─────────┐
│ Block 0 │ Block 1 │ Block 2 │ Block 3 │ Block 4 │ Block 5 │ Block 6 │ Block 7 │
└─────────┴─────────┴─────────┴─────────┴─────────┴─────────┴─────────┴─────────┘
```

## 定义原型<a name="section620mcpsimp"></a>

```
template <typename T = DefaultType, typename U, typename S>
__simd_callee__ inline void GatherB(U& dstReg, __ubuf__ T* baseAddr, S& index, MaskReg& mask)
```

## 参数说明<a name="section622mcpsimp"></a>

**表 1**  模板参数说明

<a name="table4835205712588"></a>
<table><thead align="left"><tr id="row118356578583"><th class="cellrowborder" valign="top" width="18.41%" id="mcps1.2.3.1.1"><p id="p48354572582"><a name="p48354572582"></a><a name="p48354572582"></a>参数名</p>
</th>
<th class="cellrowborder" valign="top" width="81.58999999999999%" id="mcps1.2.3.1.2"><p id="p583535795817"><a name="p583535795817"></a><a name="p583535795817"></a>描述</p>
</th>
</tr>
</thead>
<tbody><tr id="row1835857145817"><td class="cellrowborder" valign="top" width="18.41%" headers="mcps1.2.3.1.1 "><p id="p5835457165816"><a name="p5835457165816"></a><a name="p5835457165816"></a>T</p>
</td>
<td class="cellrowborder" valign="top" width="81.58999999999999%" headers="mcps1.2.3.1.2 "><p id="p168351657155818"><a name="p168351657155818"></a><a name="p168351657155818"></a>目的操作数和源操作数的数据类型</p>
</td>
</tr>
<tr id="row57311844194617"><td class="cellrowborder" valign="top" width="18.41%" headers="mcps1.2.3.1.1 "><p id="p1673194420460"><a name="p1673194420460"></a><a name="p1673194420460"></a>U</p>
</td>
<td class="cellrowborder" valign="top" width="81.58999999999999%" headers="mcps1.2.3.1.2 "><p id="p14983122419"><a name="p14983122419"></a><a name="p14983122419"></a>目的操作数的RegTensor类型， 例如RegTensor&lt;half&gt;，由编译器自动推导，用户不需要填写</p>
</td>
</tr>
<tr id="row13971201184719"><td class="cellrowborder" valign="top" width="18.41%" headers="mcps1.2.3.1.1 "><p id="p69710116478"><a name="p69710116478"></a><a name="p69710116478"></a>S</p>
</td>
<td class="cellrowborder" valign="top" width="81.58999999999999%" headers="mcps1.2.3.1.2 "><p id="p297191154712"><a name="p297191154712"></a><a name="p297191154712"></a>索引值的RegTensor类型，例如RegTensor&lt;uint32_t&gt;，由编译器自动推导，用户不需要填写</p>
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
<p id="p18220202613566"><a name="p18220202613566"></a><a name="p18220202613566"></a>类型为<a href="RegTensor.md">RegTensor</a></p>
<p id="p15544114718408"><a name="p15544114718408"></a><a name="p15544114718408"></a><span id="ph8342353144014"><a name="ph8342353144014"></a><a name="ph8342353144014"></a>Ascend 950PR/Ascend 950DT</span>，dstReg/baseAddr支持的数据类型为：uint8_t/int8_t/uint16_t/int16_t/uint32_t/int32_t/half/float/bfloat16_t/uint64_t/int64_t</p>
</td>
</tr>
<tr id="row2137145181815"><td class="cellrowborder" valign="top" width="18.54%" headers="mcps1.2.4.1.1 "><p id="p6844125874315"><a name="p6844125874315"></a><a name="p6844125874315"></a>baseAddr</p>
</td>
<td class="cellrowborder" valign="top" width="10.05%" headers="mcps1.2.4.1.2 "><p id="p128442058144312"><a name="p128442058144312"></a><a name="p128442058144312"></a>输入</p>
</td>
<td class="cellrowborder" valign="top" width="71.41%" headers="mcps1.2.4.1.3 "><p id="p172083541517"><a name="p172083541517"></a><a name="p172083541517"></a>源操作数在UB中的基地址。需要32B对齐。</p>
<p id="p2051111482068"><a name="p2051111482068"></a><a name="p2051111482068"></a>类型为UB指针。</p>
<p id="p16744208194119"><a name="p16744208194119"></a><a name="p16744208194119"></a><span id="ph127449818419"><a name="ph127449818419"></a><a name="ph127449818419"></a>Ascend 950PR/Ascend 950DT</span>，支持的数据类型为：uint8_t/int8_t/uint16_t/int16_t/uint32_t/int32_t/half/float/bfloat16_t/uint64_t/int64_t</p>
</td>
</tr>
<tr id="row891912431168"><td class="cellrowborder" valign="top" width="18.54%" headers="mcps1.2.4.1.1 "><p id="p195756503168"><a name="p195756503168"></a><a name="p195756503168"></a>index</p>
</td>
<td class="cellrowborder" valign="top" width="10.05%" headers="mcps1.2.4.1.2 "><p id="p457515071618"><a name="p457515071618"></a><a name="p457515071618"></a>输入</p>
</td>
<td class="cellrowborder" valign="top" width="71.41%" headers="mcps1.2.4.1.3 "><p id="p193020233012"><a name="p193020233012"></a><a name="p193020233012"></a>dstReg中的每个DataBlock在UB中相对于baseAddr的索引位置。索引位置要大于等于0且32B对齐。</p>
<p id="p1855210208258"><a name="p1855210208258"></a><a name="p1855210208258"></a>类型为<a href="RegTensor.md">RegTensor</a></p>
<p id="p36931438114113"><a name="p36931438114113"></a><a name="p36931438114113"></a><span id="ph1369393818412"><a name="ph1369393818412"></a><a name="ph1369393818412"></a>Ascend 950PR/Ascend 950DT</span>，支持的数据类型为：uint32_t</p>
</td>
</tr>
<tr id="row19615183817191"><td class="cellrowborder" valign="top" width="18.54%" headers="mcps1.2.4.1.1 "><p id="p1484519586432"><a name="p1484519586432"></a><a name="p1484519586432"></a>mask</p>
</td>
<td class="cellrowborder" valign="top" width="10.05%" headers="mcps1.2.4.1.2 "><p id="p484514581433"><a name="p484514581433"></a><a name="p484514581433"></a>输入</p>
</td>
<td class="cellrowborder" valign="top" width="71.41%" headers="mcps1.2.4.1.3 "><p id="p1559991025517"><a name="p1559991025517"></a><a name="p1559991025517"></a>src element操作有效指示，详细说明请参考<a href="MaskReg.md">MaskReg</a></p>
</td>
</tr>
</tbody>
</table>

## 约束说明<a name="section177921451558"></a>

-   目的操作数与源操作数的数据类型相同。
-   源操作数在UB中的基地址需要32B对齐。
-   索引位置要大于等于0且32B对齐。
-   索引寄存器中可以存在相同的值，即可以多次读取源操作数中同一个DataBlock的数据。

## 调用示例<a name="section642mcpsimp"></a>

```
template<typename T, typename U>
__simd_vf__ inline void GatherBVF(__ubuf__ T* dstAddr, __ubuf__ T* src0Addr, __ubuf__ U* src1Addr, uint32_t count, uint32_t oneRepeatSize, uint16_t repeatTimes)
{
    AscendC::MicroAPI::RegTensor<U> srcReg;
    AscendC::MicroAPI::RegTensor<T> dstReg;
    AscendC::MicroAPI::MaskReg mask;    
    for (uint16_t i = 0; i < repeatTimes; i++) {
        mask = AscendC::MicroAPI::UpdateMask<T>(count);
        AscendC::MicroAPI::LoadAlign(srcReg, src1Addr + i * oneRepeatSize);
        AscendC::MicroAPI::GatherB(dstReg, src0Addr, srcReg, mask);
        AscendC::MicroAPI::StoreAlign(dstAddr + i * oneRepeatSize, dstReg, mask);
    }
}
```

