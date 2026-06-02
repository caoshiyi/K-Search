# LocalMemBar<a name="ZH-CN_TOPIC_0000002554424071"></a>

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

Reg矢量计算宏函数内不同流水线之间的同步指令。该同步指令指定src源流水线和dst目的流水线，如下图所示，目的流水线将等待源流水线上所有指令完成才进行执行。读写场景下，当读指令使用的寄存器和写指令使用的寄存器相同时，可以触发寄存器保序，指令将会按照代码顺序执行，不需要插入同步指令，而当使用的寄存器不同时，如果要确保读写指令顺序执行，则需要插入同步指令，写写场景同理。

**图 1**  流水线等待示意图<a name="fig2635167165614"></a>  
<!-- img2text -->
```text
写读依赖

SIMD_VF函数写UB流水线  ────────┌──────────────────────┐
                                │ 写UB，使用寄存器reg0 │──── src直连流水线 ─────────────────────────
                                └──────────────────────┘
                                           ╲
                                            ╲  LocalMemBar<src, dst>()
                                             ╲
SIMD_VF函数读UB流水线  ────────────────────────╲──────────────┌──────────────────────┐──── dst目的流水线 ──
                                                            │ 读UB，使用寄存器reg1 │
                                                            └──────────────────────┘


读写依赖

SIMD_VF函数读UB流水线  ────────┌──────────────────────┐
                                │ 读UB，使用寄存器reg0 │──── src直连流水线 ─────────────────────────
                                └──────────────────────┘
                                           ╲
                                            ╲  LocalMemBar<src, dst>()
                                             ╲
SIMD_VF函数写UB流水线  ────────────────────────╲──────────────┌──────────────────────┐──── dst目的流水线 ──
                                                            │ 写UB，使用寄存器reg1 │
                                                            └──────────────────────┘


读写（寄存器保序）

SIMD_VF函数读UB流水线  ────────┌──────────────────────┐
                                │ 读UB，使用寄存器reg0 │──── src直连流水线 ─────────────────────────
                                └──────────────────────┘
                                           ╲
                                            ╲  寄存器保序
                                             ╲
SIMD_VF函数写UB流水线  ────────────────────────╲──────────────┌──────────────────────┐──── dst目的流水线 ──
                                                            │ 写UB，使用寄存器reg0 │
                                                            └──────────────────────┘


写写依赖

SIMD_VF函数写UB流水线  ────────┌──────────────────────┐
                                │ 写UB，使用寄存器reg0 │──── src直连流水线 ─────────────────────────
                                └──────────────────────┘
                                           ╲
                                            ╲  LocalMemBar<src, dst>()
                                             ╲
SIMD_VF函数写UB流水线  ────────────────────────╲──────────────┌──────────────────────┐──── dst目的流水线 ──
                                                            │ 写UB，使用寄存器reg1 │
                                                            └──────────────────────┘
```

## 函数原型<a name="section620mcpsimp"></a>

```
template <MemType src, MemType dst> 
__simd_callee__ inline void LocalMemBar()
```

## 参数说明<a name="section622mcpsimp"></a>

**表 1**  模板参数说明

<a name="table4835205712588"></a>
<table><thead align="left"><tr id="row118356578583"><th class="cellrowborder" valign="top" width="18.27%" id="mcps1.2.3.1.1"><p id="p48354572582"><a name="p48354572582"></a><a name="p48354572582"></a>参数名</p>
</th>
<th class="cellrowborder" valign="top" width="81.73%" id="mcps1.2.3.1.2"><p id="p583535795817"><a name="p583535795817"></a><a name="p583535795817"></a>描述</p>
</th>
</tr>
</thead>
<tbody><tr id="row1835857145817"><td class="cellrowborder" valign="top" width="18.27%" headers="mcps1.2.3.1.1 "><p id="p5835457165816"><a name="p5835457165816"></a><a name="p5835457165816"></a>src</p>
</td>
<td class="cellrowborder" valign="top" width="81.73%" headers="mcps1.2.3.1.2 "><p id="p16653174755317"><a name="p16653174755317"></a><a name="p16653174755317"></a>源流水线，类型为MemType，具体参见<a href="#zh-cn_topic_0235751031_table33761356">表2 MemType取值说明</a>。</p>
</td>
</tr>
<tr id="row18835145716587"><td class="cellrowborder" valign="top" width="18.27%" headers="mcps1.2.3.1.1 "><p id="p1756419170189"><a name="p1756419170189"></a><a name="p1756419170189"></a>dst</p>
</td>
<td class="cellrowborder" valign="top" width="81.73%" headers="mcps1.2.3.1.2 "><p id="p6881103675215"><a name="p6881103675215"></a><a name="p6881103675215"></a>目的流水线，类型为MemType，具体参见<a href="#zh-cn_topic_0235751031_table33761356">表2 MemType取值说明</a>。</p>
</td>
</tr>
</tbody>
</table>

**表 2**  MemType取值说明

<a name="zh-cn_topic_0235751031_table33761356"></a>
<table><thead align="left"><tr id="zh-cn_topic_0235751031_row27598891"><th class="cellrowborder" valign="top" width="18.21%" id="mcps1.2.3.1.1"><p id="zh-cn_topic_0235751031_p20917673"><a name="zh-cn_topic_0235751031_p20917673"></a><a name="zh-cn_topic_0235751031_p20917673"></a>MemType取值</p>
</th>
<th class="cellrowborder" valign="top" width="81.78999999999999%" id="mcps1.2.3.1.2"><p id="zh-cn_topic_0235751031_p59995477"><a name="zh-cn_topic_0235751031_p59995477"></a><a name="zh-cn_topic_0235751031_p59995477"></a>含义</p>
</th>
</tr>
</thead>
<tbody><tr id="row42461942101815"><td class="cellrowborder" valign="top" width="18.21%" headers="mcps1.2.3.1.1 "><p id="p108051250181214"><a name="p108051250181214"></a><a name="p108051250181214"></a>VEC_STORE</p>
</td>
<td class="cellrowborder" valign="top" width="81.78999999999999%" headers="mcps1.2.3.1.2 "><p id="p167614410315"><a name="p167614410315"></a><a name="p167614410315"></a>SIMD_VF函数内矢量写UB流水线。</p>
<p id="p45581654185211"><a name="p45581654185211"></a><a name="p45581654185211"></a>对应寄存器到UB的搬运指令，如StoreAlign、StoreUnAlign、Store。</p>
</td>
</tr>
<tr id="row2137145181815"><td class="cellrowborder" valign="top" width="18.21%" headers="mcps1.2.3.1.1 "><p id="p19574165615129"><a name="p19574165615129"></a><a name="p19574165615129"></a>VEC_LOAD</p>
</td>
<td class="cellrowborder" valign="top" width="81.78999999999999%" headers="mcps1.2.3.1.2 "><p id="p97591441635"><a name="p97591441635"></a><a name="p97591441635"></a>SIMD_VF函数内矢量读UB流水线。</p>
<p id="p18760948145415"><a name="p18760948145415"></a><a name="p18760948145415"></a>对应UB到寄存器的搬运指令，如LoadAlign、LoadUnAlign、Load。</p>
</td>
</tr>
<tr id="row11773440341"><td class="cellrowborder" valign="top" width="18.21%" headers="mcps1.2.3.1.1 "><p id="p10132448173420"><a name="p10132448173420"></a><a name="p10132448173420"></a>SCALAR_STORE</p>
</td>
<td class="cellrowborder" valign="top" width="81.78999999999999%" headers="mcps1.2.3.1.2 "><p id="p462923115592"><a name="p462923115592"></a><a name="p462923115592"></a>SIMD_VF函数内标量写UB流水线。</p>
<p id="p1012020395557"><a name="p1012020395557"></a><a name="p1012020395557"></a>对应标量写入UB的指令，如Duplicate。</p>
</td>
</tr>
<tr id="row19615183817191"><td class="cellrowborder" valign="top" width="18.21%" headers="mcps1.2.3.1.1 "><p id="p1484519586432"><a name="p1484519586432"></a><a name="p1484519586432"></a>SCALAR_LOAD</p>
</td>
<td class="cellrowborder" valign="top" width="81.78999999999999%" headers="mcps1.2.3.1.2 "><p id="p1415113416592"><a name="p1415113416592"></a><a name="p1415113416592"></a>SIMD_VF函数内标量读UB流水线。</p>
<p id="p1539022412570"><a name="p1539022412570"></a><a name="p1539022412570"></a>对应UB读取标量的指令，如<a href="GetValue.md#section618mcpsimp">GetValue</a>。</p>
</td>
</tr>
<tr id="row114116587598"><td class="cellrowborder" valign="top" width="18.21%" headers="mcps1.2.3.1.1 "><p id="p141415587592"><a name="p141415587592"></a><a name="p141415587592"></a>VEC_ALL</p>
</td>
<td class="cellrowborder" valign="top" width="81.78999999999999%" headers="mcps1.2.3.1.2 "><p id="p12141195817592"><a name="p12141195817592"></a><a name="p12141195817592"></a>SIMD_VF函数内所有矢量读写UB流水线。</p>
</td>
</tr>
<tr id="row3313711801"><td class="cellrowborder" valign="top" width="18.21%" headers="mcps1.2.3.1.1 "><p id="p153131014015"><a name="p153131014015"></a><a name="p153131014015"></a>SCALAR_ALL</p>
</td>
<td class="cellrowborder" valign="top" width="81.78999999999999%" headers="mcps1.2.3.1.2 "><p id="p12313911005"><a name="p12313911005"></a><a name="p12313911005"></a>SIMD_VF函数内所有标量读写UB流水线。</p>
</td>
</tr>
</tbody>
</table>

**表 3**  src和dst组合取值说明

<a name="table913981754616"></a>
<table><thead align="left"><tr id="row817651712466"><th class="cellrowborder" valign="top" width="48.42%" id="mcps1.2.3.1.1"><p id="p1017614172466"><a name="p1017614172466"></a><a name="p1017614172466"></a>src</p>
</th>
<th class="cellrowborder" valign="top" width="51.580000000000005%" id="mcps1.2.3.1.2"><p id="p522719341678"><a name="p522719341678"></a><a name="p522719341678"></a>dst</p>
</th>
</tr>
</thead>
<tbody><tr id="row121761817164615"><td class="cellrowborder" valign="top" width="48.42%" headers="mcps1.2.3.1.1 "><p id="p15176181784620"><a name="p15176181784620"></a><a name="p15176181784620"></a>VEC_STORE</p>
</td>
<td class="cellrowborder" valign="top" width="51.580000000000005%" headers="mcps1.2.3.1.2 "><p id="p9227034075"><a name="p9227034075"></a><a name="p9227034075"></a>VEC_LOAD</p>
</td>
</tr>
<tr id="row15176917194614"><td class="cellrowborder" valign="top" width="48.42%" headers="mcps1.2.3.1.1 "><p id="p817611744614"><a name="p817611744614"></a><a name="p817611744614"></a>VEC_LOAD</p>
</td>
<td class="cellrowborder" valign="top" width="51.580000000000005%" headers="mcps1.2.3.1.2 "><p id="p15227163416718"><a name="p15227163416718"></a><a name="p15227163416718"></a>VEC_STORE</p>
</td>
</tr>
<tr id="row417616179465"><td class="cellrowborder" valign="top" width="48.42%" headers="mcps1.2.3.1.1 "><p id="p101765179461"><a name="p101765179461"></a><a name="p101765179461"></a>VEC_STORE</p>
</td>
<td class="cellrowborder" valign="top" width="51.580000000000005%" headers="mcps1.2.3.1.2 "><p id="p22275341710"><a name="p22275341710"></a><a name="p22275341710"></a>VEC_STORE</p>
</td>
</tr>
<tr id="row717714175468"><td class="cellrowborder" valign="top" width="48.42%" headers="mcps1.2.3.1.1 "><p id="p817716171466"><a name="p817716171466"></a><a name="p817716171466"></a>VEC_STORE</p>
</td>
<td class="cellrowborder" valign="top" width="51.580000000000005%" headers="mcps1.2.3.1.2 "><p id="p1352711201015"><a name="p1352711201015"></a><a name="p1352711201015"></a>SCALAR_LOAD</p>
</td>
</tr>
<tr id="row417718174461"><td class="cellrowborder" valign="top" width="48.42%" headers="mcps1.2.3.1.1 "><p id="p71771217204611"><a name="p71771217204611"></a><a name="p71771217204611"></a>VEC_STORE</p>
</td>
<td class="cellrowborder" valign="top" width="51.580000000000005%" headers="mcps1.2.3.1.2 "><p id="p17227103411714"><a name="p17227103411714"></a><a name="p17227103411714"></a>SCALAR_STORE</p>
</td>
</tr>
<tr id="row10177101774614"><td class="cellrowborder" valign="top" width="48.42%" headers="mcps1.2.3.1.1 "><p id="p0117501496"><a name="p0117501496"></a><a name="p0117501496"></a>VEC_LOAD</p>
</td>
<td class="cellrowborder" valign="top" width="51.580000000000005%" headers="mcps1.2.3.1.2 "><p id="p10227113416712"><a name="p10227113416712"></a><a name="p10227113416712"></a>SCALAR_STORE</p>
</td>
</tr>
<tr id="row8177181704613"><td class="cellrowborder" valign="top" width="48.42%" headers="mcps1.2.3.1.1 "><p id="p9177121764614"><a name="p9177121764614"></a><a name="p9177121764614"></a>SCALAR_STORE</p>
</td>
<td class="cellrowborder" valign="top" width="51.580000000000005%" headers="mcps1.2.3.1.2 "><p id="p19227173413720"><a name="p19227173413720"></a><a name="p19227173413720"></a>VEC_LOAD</p>
</td>
</tr>
<tr id="row617721711469"><td class="cellrowborder" valign="top" width="48.42%" headers="mcps1.2.3.1.1 "><p id="p181771817134614"><a name="p181771817134614"></a><a name="p181771817134614"></a>SCALAR_STORE</p>
</td>
<td class="cellrowborder" valign="top" width="51.580000000000005%" headers="mcps1.2.3.1.2 "><p id="p12227234671"><a name="p12227234671"></a><a name="p12227234671"></a>VEC_STORE</p>
</td>
</tr>
<tr id="row417721720462"><td class="cellrowborder" valign="top" width="48.42%" headers="mcps1.2.3.1.1 "><p id="p171772017114610"><a name="p171772017114610"></a><a name="p171772017114610"></a>SCALAR_LOAD</p>
</td>
<td class="cellrowborder" valign="top" width="51.580000000000005%" headers="mcps1.2.3.1.2 "><p id="p12277343713"><a name="p12277343713"></a><a name="p12277343713"></a>VEC_STORE</p>
</td>
</tr>
<tr id="row51771517144617"><td class="cellrowborder" valign="top" width="48.42%" headers="mcps1.2.3.1.1 "><p id="p617751711465"><a name="p617751711465"></a><a name="p617751711465"></a>VEC_ALL</p>
</td>
<td class="cellrowborder" valign="top" width="51.580000000000005%" headers="mcps1.2.3.1.2 "><p id="p182276341710"><a name="p182276341710"></a><a name="p182276341710"></a>VEC_ALL</p>
</td>
</tr>
<tr id="row12177717134614"><td class="cellrowborder" valign="top" width="48.42%" headers="mcps1.2.3.1.1 "><p id="p9177191719460"><a name="p9177191719460"></a><a name="p9177191719460"></a>VEC_ALL</p>
</td>
<td class="cellrowborder" valign="top" width="51.580000000000005%" headers="mcps1.2.3.1.2 "><p id="p322793420715"><a name="p322793420715"></a><a name="p322793420715"></a>SCALAR_ALL</p>
</td>
</tr>
<tr id="row31781317194619"><td class="cellrowborder" valign="top" width="48.42%" headers="mcps1.2.3.1.1 "><p id="p12178617194619"><a name="p12178617194619"></a><a name="p12178617194619"></a>SCALAR_ALL</p>
</td>
<td class="cellrowborder" valign="top" width="51.580000000000005%" headers="mcps1.2.3.1.2 "><p id="p2022715341479"><a name="p2022715341479"></a><a name="p2022715341479"></a>VEC_ALL</p>
</td>
</tr>
</tbody>
</table>

## 返回值说明<a name="section640mcpsimp"></a>

无

## 约束说明<a name="section633mcpsimp"></a>

无

## 调用示例<a name="section642mcpsimp"></a>

如下示例中dstPtr和src0Ptr为同一个UB地址，for循环中第二次循环中读UB矢量搬运和第一次循环中写UB矢量搬运操作了同一块UB地址空间，因此第二次循环中读UB矢量搬运需要等待第一次循环中写UB矢量搬运执行完成后才能执行，需要插入VEC\_LOAD等待VEC\_STORE的同步。

```
template<typename T>
__simd_vf__ inline void AddVF(__ubuf__ T* dstAddr, __ubuf__ T* src0Addr, __ubuf__ T* src1Addr, uint32_t count, uint32_t oneRepeatSize, uint16_t repeatTimes)
{
    AscendC::MicroAPI::RegTensor<T> srcReg0;
    AscendC::MicroAPI::RegTensor<T> srcReg1;
    AscendC::MicroAPI::RegTensor<T> dstReg;
    AscendC::MicroAPI::MaskReg mask;   
    for (uint16_t i = 0; i < repeatTimes; i++) {
        mask = AscendC::MicroAPI::UpdateMask<T>(count);
        AscendC::MicroAPI::LocalMemBar<AscendC::MicroAPI::MemType::VEC_STORE, AscendC::MicroAPI::MemType::VEC_LOAD>();
        AscendC::MicroAPI::LoadAlign(srcReg0, src0Addr);
        AscendC::MicroAPI::LoadAlign(srcReg1, src1Addr + i * oneRepeatSize);
        AscendC::MicroAPI::Add(dstReg, srcReg0, srcReg1, mask);
        AscendC::MicroAPI::StoreAlign(dstAddr, dstReg, mask);
    }
}
```

