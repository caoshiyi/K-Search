# AddrReg<a name="ZH-CN_TOPIC_0000002554343547"></a>

## 功能说明<a name="section618mcpsimp"></a>

AddrReg即为Address Register（地址寄存器），是用于存储地址偏移量的寄存器。AddrReg应该通过CreateAddrReg API初始化，然后在循环之中使用AddrReg存储地址偏移量。AddrReg在每层循环中根据所设置的stride进行自增。

## 函数原型<a name="section620mcpsimp"></a>

```
// offset = index0 * stride0
template <typename T>
__simd_callee__ inline AddrReg CreateAddrReg(uint16_t index0, uint32_t stride0);
 
// offset = index0 * stride0 + index1 * stride1
template <typename T>
__simd_callee__ inline AddrReg CreateAddrReg(uint16_t index0, uint32_t stride0, uint16_t index1, uint32_t stride1);
 
// offset = index0 * stride0 + index1 * stride1 + index2 * stride2
template <typename T>
__simd_callee__ inline AddrReg CreateAddrReg(uint16_t index0, uint32_t stride0, uint16_t index1, uint32_t stride1, uint16_t index2, uint32_t stride2);

// offset = index0 * stride0 + index1 * stride1 + index2 * stride2 + index3 * stride3
template <typename T>
__simd_callee__ inline AddrReg CreateAddrReg(uint16_t index0, uint32_t stride0, uint16_t index1, uint32_t stride1, uint16_t index2, uint32_t stride2, uint16_t index3, uint32_t stride3);
```

## 参数说明<a name="section132601254123919"></a>

<a name="table1996313517283"></a>
<table><thead align="left"><tr id="row1511192711402"><th class="cellrowborder" valign="top" width="49.87%" id="mcps1.1.3.1.1"><p id="p171117272401"><a name="p171117272401"></a><a name="p171117272401"></a><strong id="b638635134018"><a name="b638635134018"></a><a name="b638635134018"></a>参数</strong></p>
</th>
<th class="cellrowborder" valign="top" width="50.129999999999995%" id="mcps1.1.3.1.2"><p id="p111115272401"><a name="p111115272401"></a><a name="p111115272401"></a><strong id="b1241535164013"><a name="b1241535164013"></a><a name="b1241535164013"></a>含义</strong></p>
</th>
</tr>
</thead>
<tbody><tr id="row1548632816145"><td class="cellrowborder" valign="top" width="49.87%" headers="mcps1.1.3.1.1 "><p id="p2487172821412"><a name="p2487172821412"></a><a name="p2487172821412"></a>T</p>
</td>
<td class="cellrowborder" valign="top" width="50.129999999999995%" headers="mcps1.1.3.1.2 "><p id="p24871228101413"><a name="p24871228101413"></a><a name="p24871228101413"></a>模板参数，支持的数据类型为b8/b16/b32/b64。</p>
</td>
</tr>
<tr id="row699410354287"><td class="cellrowborder" valign="top" width="49.87%" headers="mcps1.1.3.1.1 "><p id="p119941135152820"><a name="p119941135152820"></a><a name="p119941135152820"></a>index0</p>
</td>
<td class="cellrowborder" valign="top" width="50.129999999999995%" headers="mcps1.1.3.1.2 "><p id="p119942035142815"><a name="p119942035142815"></a><a name="p119942035142815"></a>计算偏移量时作为轴1（偏移量计算公式参考注释）。</p>
</td>
</tr>
<tr id="row129949353286"><td class="cellrowborder" valign="top" width="49.87%" headers="mcps1.1.3.1.1 "><p id="p11994153582812"><a name="p11994153582812"></a><a name="p11994153582812"></a>index1</p>
</td>
<td class="cellrowborder" valign="top" width="50.129999999999995%" headers="mcps1.1.3.1.2 "><p id="p0994193510287"><a name="p0994193510287"></a><a name="p0994193510287"></a>计算偏移量时作为轴2（偏移量计算公式参考注释）。</p>
</td>
</tr>
<tr id="row19942035142810"><td class="cellrowborder" valign="top" width="49.87%" headers="mcps1.1.3.1.1 "><p id="p299413350284"><a name="p299413350284"></a><a name="p299413350284"></a>index2</p>
</td>
<td class="cellrowborder" valign="top" width="50.129999999999995%" headers="mcps1.1.3.1.2 "><p id="p12995335112810"><a name="p12995335112810"></a><a name="p12995335112810"></a>计算偏移量时作为轴3（偏移量计算公式参考注释）。</p>
</td>
</tr>
<tr id="row1799518350287"><td class="cellrowborder" valign="top" width="49.87%" headers="mcps1.1.3.1.1 "><p id="p1299515354287"><a name="p1299515354287"></a><a name="p1299515354287"></a>index3</p>
</td>
<td class="cellrowborder" valign="top" width="50.129999999999995%" headers="mcps1.1.3.1.2 "><p id="p2995335132820"><a name="p2995335132820"></a><a name="p2995335132820"></a>计算偏移量时作为轴4（偏移量计算公式参考注释）。</p>
</td>
</tr>
<tr id="row11995173552815"><td class="cellrowborder" valign="top" width="49.87%" headers="mcps1.1.3.1.1 "><p id="p9995235172817"><a name="p9995235172817"></a><a name="p9995235172817"></a>stride0</p>
</td>
<td class="cellrowborder" valign="top" width="50.129999999999995%" headers="mcps1.1.3.1.2 "><p id="p19995635172815"><a name="p19995635172815"></a><a name="p19995635172815"></a>地址偏移量1</p>
</td>
</tr>
<tr id="row169951435182817"><td class="cellrowborder" valign="top" width="49.87%" headers="mcps1.1.3.1.1 "><p id="p20995103542820"><a name="p20995103542820"></a><a name="p20995103542820"></a>stride1</p>
</td>
<td class="cellrowborder" valign="top" width="50.129999999999995%" headers="mcps1.1.3.1.2 "><p id="p299519353284"><a name="p299519353284"></a><a name="p299519353284"></a>地址偏移量2</p>
</td>
</tr>
<tr id="row1699543511287"><td class="cellrowborder" valign="top" width="49.87%" headers="mcps1.1.3.1.1 "><p id="p12995173532819"><a name="p12995173532819"></a><a name="p12995173532819"></a>stride2</p>
</td>
<td class="cellrowborder" valign="top" width="50.129999999999995%" headers="mcps1.1.3.1.2 "><p id="p109957354286"><a name="p109957354286"></a><a name="p109957354286"></a>地址偏移量3</p>
</td>
</tr>
<tr id="row499513522817"><td class="cellrowborder" valign="top" width="49.87%" headers="mcps1.1.3.1.1 "><p id="p1199573552810"><a name="p1199573552810"></a><a name="p1199573552810"></a>stride3</p>
</td>
<td class="cellrowborder" valign="top" width="50.129999999999995%" headers="mcps1.1.3.1.2 "><p id="p1899510353283"><a name="p1899510353283"></a><a name="p1899510353283"></a>地址偏移量4</p>
</td>
</tr>
</tbody>
</table>

## 支持的型号<a name="section156721693504"></a>

Ascend 950PR/Ascend 950DT

## 约束说明<a name="section11585101304320"></a>

无

## 调用示例<a name="section633mcpsimp"></a>

```
__simd_vf__ inline void CreateAddrRegVF(__ubuf__ T* dstAddr, __ubuf__ T* srcAddr, uint32_t oneRepeatSize, uint16_t repeatTimes)
{
    AscendC::MicroAPI::MaskReg mask = AscendC::MicroAPI::CreateMask<T>();
    AscendC::MicroAPI::AddrReg aReg;
    for (uint16_t i = 0; i < repeatTimes; ++i) {
        aReg = AscendC::MicroAPI::CreateAddrReg<T>(i, oneRepeatSize);
        AscendC::MicroAPI::LoadAlign(mask, srcAddr, aReg);
        AscendC::MicroAPI::StoreAlign(dstAddr, mask, aReg);
    }
}
```

