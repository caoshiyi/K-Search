# RegTensor<a name="ZH-CN_TOPIC_0000002523304726"></a>

## 功能说明<a name="section618mcpsimp"></a>

Reg矢量计算基本单元，RegTensor位宽为VL（Vector Length），具体值可能因不同AI处理器型号而异。

## 定义原型<a name="section620mcpsimp"></a>

```
template <typename T, const RegTrait& regTrait = RegTraitNumOne> struct RegTensor;
```

## 函数说明<a name="section622mcpsimp"></a>

模板参数T，支持的数据类型\(宽度\)为b8/b16/b32/b64。

<a name="zh-cn_topic_0235751031_table33761356"></a>
<table><thead align="left"><tr id="zh-cn_topic_0235751031_row27598891"><th class="cellrowborder" valign="top" width="17.41%" id="mcps1.1.3.1.1"><p id="zh-cn_topic_0235751031_p20917673"><a name="zh-cn_topic_0235751031_p20917673"></a><a name="zh-cn_topic_0235751031_p20917673"></a>数据类型宽度</p>
</th>
<th class="cellrowborder" valign="top" width="82.59%" id="mcps1.1.3.1.2"><p id="zh-cn_topic_0235751031_p59995477"><a name="zh-cn_topic_0235751031_p59995477"></a><a name="zh-cn_topic_0235751031_p59995477"></a>数据类型</p>
</th>
</tr>
</thead>
<tbody><tr id="row42461942101815"><td class="cellrowborder" valign="top" width="17.41%" headers="mcps1.1.3.1.1 "><p id="p108051250181214"><a name="p108051250181214"></a><a name="p108051250181214"></a>b8</p>
</td>
<td class="cellrowborder" valign="top" width="82.59%" headers="mcps1.1.3.1.2 "><p id="p2366753194920"><a name="p2366753194920"></a><a name="p2366753194920"></a>支持的数据类型为：bool/int8_t/uint8_t/fp4x2_e2m1_t/fp4x2_e1m2_t/hifloat8_t/fp8_e5m2_t/fp8_e4m3fn_t/fp8_e8m0_t（ fp4x2_e2m1_t/fp4x2_e1m2_t这两个b4类型在Vector侧的内存排布需要为两个一组，表现为b8类型；int4b_t也使用b8类型表达；bool数据类型只支持数据搬运）。</p>
</td>
</tr>
<tr id="row2137145181815"><td class="cellrowborder" valign="top" width="17.41%" headers="mcps1.1.3.1.1 "><p id="p19574165615129"><a name="p19574165615129"></a><a name="p19574165615129"></a>b16</p>
</td>
<td class="cellrowborder" valign="top" width="82.59%" headers="mcps1.1.3.1.2 "><p id="p045917114420"><a name="p045917114420"></a><a name="p045917114420"></a>支持的数据类型为：int16_t/uint16_t/half/bfloat16_t。</p>
</td>
</tr>
<tr id="row7463133114314"><td class="cellrowborder" valign="top" width="17.41%" headers="mcps1.1.3.1.1 "><p id="p1846343313436"><a name="p1846343313436"></a><a name="p1846343313436"></a>b32</p>
</td>
<td class="cellrowborder" valign="top" width="82.59%" headers="mcps1.1.3.1.2 "><p id="p5463183394311"><a name="p5463183394311"></a><a name="p5463183394311"></a>支持的数据类型为：int32_t/uint32_t/float/complex32。</p>
</td>
</tr>
<tr id="row165841599463"><td class="cellrowborder" valign="top" width="17.41%" headers="mcps1.1.3.1.1 "><p id="p95841659104615"><a name="p95841659104615"></a><a name="p95841659104615"></a>b64</p>
</td>
<td class="cellrowborder" valign="top" width="82.59%" headers="mcps1.1.3.1.2 "><p id="p358465916469"><a name="p358465916469"></a><a name="p358465916469"></a>支持的数据类型为：int64_t/uint64_t/complex64。</p>
</td>
</tr>
</tbody>
</table>

模板参数regTrait，表示该RegTensor类型内部包含的矢量Reg数量。regTrait为RegTraitNumOne时，该RegTensor类型中包含1个相应数据类型的矢量Reg，长度为VL。regTrait为RegTraitNumTwo时，该RegTensor类型中包含2个相应数据类型的矢量Reg，总长度为2 \* VL，每个矢量Reg长度为VL。

<a name="table385811403491"></a>
<table><thead align="left"><tr id="row985815405493"><th class="cellrowborder" valign="top" width="17.65%" id="mcps1.1.3.1.1"><p id="p178581040164916"><a name="p178581040164916"></a><a name="p178581040164916"></a>模板参数regTrait</p>
</th>
<th class="cellrowborder" valign="top" width="82.35%" id="mcps1.1.3.1.2"><p id="p117694418586"><a name="p117694418586"></a><a name="p117694418586"></a>支持的数据类型宽度</p>
</th>
</tr>
</thead>
<tbody><tr id="row19858940134911"><td class="cellrowborder" valign="top" width="17.65%" headers="mcps1.1.3.1.1 "><p id="p68587404495"><a name="p68587404495"></a><a name="p68587404495"></a>RegTraitNumOne</p>
</td>
<td class="cellrowborder" valign="top" width="82.35%" headers="mcps1.1.3.1.2 "><p id="p118581540104918"><a name="p118581540104918"></a><a name="p118581540104918"></a><span id="ph98581740104910"><a name="ph98581740104910"></a><a name="ph98581740104910"></a>Ascend 950PR/Ascend 950DT</span>，支持的数据类型宽度为：b8/b16/b32/b64。</p>
</td>
</tr>
<tr id="row385894094920"><td class="cellrowborder" valign="top" width="17.65%" headers="mcps1.1.3.1.1 "><p id="p385994018495"><a name="p385994018495"></a><a name="p385994018495"></a>RegTraitNumTwo</p>
</td>
<td class="cellrowborder" valign="top" width="82.35%" headers="mcps1.1.3.1.2 "><p id="p17386201612599"><a name="p17386201612599"></a><a name="p17386201612599"></a><span id="ph1045410457315"><a name="ph1045410457315"></a><a name="ph1045410457315"></a>Ascend 950PR/Ascend 950DT</span>，支持的数据类型宽度为：b64/complex32。</p>
</td>
</tr>
</tbody>
</table>

## 支持的型号<a name="section633mcpsimp"></a>

Ascend 950PR/Ascend 950DT（VL=256B）

## 约束说明<a name="section177921451558"></a>

无

## 调用示例<a name="section1398164912391"></a>

-   示例一

    ```
    AscendC::MicroAPI::RegTensor<uint32_t> reg;
    AscendC::MicroAPI::MaskReg mask = AscendC::MicroAPI::CreateMask<uint32_t>();
    AscendC::MicroAPI::LoadAlign(reg, src, 0);
    AscendC::MicroAPI::Adds(reg, reg, 1);
    AscendC::MicroAPI::StoreAlign(dst, reg, 0, mask);
    ```

-   示例二

    ```
    // 针对B64,可以传入RegTraitNumTwo
    template<typename T, const AscendC::MicroAPI::RegTrait& Trait = AscendC::MicroAPI::RegTraitNumOne>
    __simd_vf__ inline void AddVF(__ubuf__ T* dstAddr, __ubuf__ T* src0Addr, __ubuf__ T* src1Addr, uint32_t count, uint32_t oneRepeatSize, uint16_t repeatTimes)
    {
        AscendC::MicroAPI::RegTensor<T,Trait> srcReg0;
        AscendC::MicroAPI::RegTensor<T,Trait> srcReg1;
        AscendC::MicroAPI::RegTensor<T,Trait> dstReg;
        AscendC::MicroAPI::MaskReg mask;
        for (uint16_t i = 0; i < repeatTimes; i++) {
            mask = AscendC::MicroAPI::UpdateMask<T,Trait>(count);
            AscendC::MicroAPI::LoadAlign(srcReg0, src0Addr + i * oneRepeatSize);
            AscendC::MicroAPI::LoadAlign(srcReg1, src1Addr + i * oneRepeatSize);
            AscendC::MicroAPI::Add(dstReg, srcReg0, srcReg1, mask);
            AscendC::MicroAPI::StoreAlign(dstAddr + i * oneRepeatSize, dstReg, mask);
        }
    }
    ```

