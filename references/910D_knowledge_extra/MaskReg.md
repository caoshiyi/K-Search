# MaskReg<a name="ZH-CN_TOPIC_0000002523304268"></a>

## 功能说明<a name="section618mcpsimp"></a>

MaskReg用于指示在计算过程中哪些元素参与计算，宽度为RegTensor的八分之一（VL/8）。

<!-- img2text -->
```
                           LOW                                   HIGH

MaskReg
┌─────┬─────┬─────┬─────┬─────┬─────┬─────┬─────┐
│  1  │  1  │  1  │  0  │  0  │  1  │  0  │  0  │
└─────┴─────┴─────┴─────┴─────┴─────┴─────┴─────┘
  ↓     ↓     ↓     ↓     ↓     ↓     ↓     ↓
RegTensor<b8>
┌─────┬─────┬─────┬─────┬─────┬─────┬─────┬─────┐
│  0  │  1  │  2  │  3  │  4  │  5  │  6  │  7  │
└─────┴─────┴─────┴─────┴─────┴─────┴─────┴─────┘

                 MaskReg中一个bit对应一个元素


MaskReg
┌──────────┬──────────┬──────────┬──────────┬──────────┬──────────┬──────────┬──────────┐
│    1     │    1     │    1     │    0     │    0     │    1     │    0     │    0     │
└──────────┴──────────┴──────────┴──────────┴──────────┴──────────┴──────────┴──────────┘
      ↘                    ↓                             ↙               ↓
RegTensor<b16>
┌─────────────────┬─────────────────┬─────────────────┬─────────────────┐
│        0        │        1        │        2        │        3        │
└─────────────────┴─────────────────┴─────────────────┴─────────────────┘

          MaskReg每个bit对应一个元素，只有低4bit有效
```

## 函数原型<a name="section620mcpsimp"></a>

```
template <typename T, MaskPattern mode = MaskPattern::ALL, const RegTrait& regTrait = RegTraitNumOne>
__simd_callee__ inline MaskReg CreateMask();

template <typename T, const RegTrait& regTrait = RegTraitNumOne>
__simd_callee__ inline MaskReg UpdateMask(uint32_t& scalarValue);
```

## 参数说明<a name="section622mcpsimp"></a>

**表 1**  参数说明

<a name="table14132101714462"></a>
<table><thead align="left"><tr id="row19176617124620"><th class="cellrowborder" valign="top" width="13.268673132686734%" id="mcps1.2.4.1.1"><p id="p117621724612"><a name="p117621724612"></a><a name="p117621724612"></a>参数名</p>
</th>
<th class="cellrowborder" valign="top" width="18.36816318368163%" id="mcps1.2.4.1.2"><p id="p417681717463"><a name="p417681717463"></a><a name="p417681717463"></a>输入/输出</p>
</th>
<th class="cellrowborder" valign="top" width="68.36316368363164%" id="mcps1.2.4.1.3"><p id="p5176017164614"><a name="p5176017164614"></a><a name="p5176017164614"></a>描述</p>
</th>
</tr>
</thead>
<tbody><tr id="row0176117164610"><td class="cellrowborder" valign="top" width="13.268673132686734%" headers="mcps1.2.4.1.1 "><p id="p12176171764613"><a name="p12176171764613"></a><a name="p12176171764613"></a>T</p>
</td>
<td class="cellrowborder" valign="top" width="18.36816318368163%" headers="mcps1.2.4.1.2 "><p id="p5176191734615"><a name="p5176191734615"></a><a name="p5176191734615"></a>输入</p>
</td>
<td class="cellrowborder" valign="top" width="68.36316368363164%" headers="mcps1.2.4.1.3 "><p id="p3176717174617"><a name="p3176717174617"></a><a name="p3176717174617"></a>模板参数，支持的数据类型为b8/b16/b32/b64。</p>
</td>
</tr>
<tr id="row1217616175468"><td class="cellrowborder" valign="top" width="13.268673132686734%" headers="mcps1.2.4.1.1 "><p id="p5176121754619"><a name="p5176121754619"></a><a name="p5176121754619"></a>mode</p>
</td>
<td class="cellrowborder" valign="top" width="18.36816318368163%" headers="mcps1.2.4.1.2 "><p id="p12176161714468"><a name="p12176161714468"></a><a name="p12176161714468"></a>输入</p>
</td>
<td class="cellrowborder" valign="top" width="68.36316368363164%" headers="mcps1.2.4.1.3 "><p id="p1817615174462"><a name="p1817615174462"></a><a name="p1817615174462"></a>创建MaskReg的模式，enum class类型。</p>
<pre class="screen" id="screen39202451285"><a name="screen39202451285"></a><a name="screen39202451285"></a>enum class MaskPattern {
    ALL,      // 所有元素设置为True
    VL1,      // 最低1个元素
    VL2,      // 最低2个元素
    VL3,      // 最低3个元素
    VL4,      // 最低4个元素
    VL8,      // 最低8个元素
    VL16,     // 最低16个元素
    VL32,     // 最低32个元素
    VL64,     // 最低64个元素
    VL128,    // 最低128个元素
    M3,       // 3的倍数
    M4,       // 4的倍数
    H,        // 最低一半元素
    Q,        // 最低四分之一元素
    ALLF = 15 // 所有元素设置为false
};</pre>
</td>
</tr>
<tr id="row208171336164412"><td class="cellrowborder" valign="top" width="13.268673132686734%" headers="mcps1.2.4.1.1 "><p id="p1981810361446"><a name="p1981810361446"></a><a name="p1981810361446"></a>regTrait</p>
</td>
<td class="cellrowborder" valign="top" width="18.36816318368163%" headers="mcps1.2.4.1.2 "><p id="p1681815364440"><a name="p1681815364440"></a><a name="p1681815364440"></a>输入</p>
</td>
<td class="cellrowborder" valign="top" width="68.36316368363164%" headers="mcps1.2.4.1.3 "><p id="p12818836114415"><a name="p12818836114415"></a><a name="p12818836114415"></a>当前仅针对b64/complex32数据类型生效，分为RegTraitNumOne和RegTraitNumTwo，含义和RegTensor的模板regTrait类型一致，配合RegTensor的regTrait一起使用。regTrait为RegTraitNumOne时，表明当前MaskReg的作用范围可覆盖至256B（一个VL的长度）。对于使用RegTraitNumOne的b64 RegTensor的指令，生成的b64 mask为每8位有效；RegTraitNumTwo表明当前MaskReg的作用范围可覆盖至512B（两个VL的长度），生成的b64 mask为每4位有效，作用于使用RegTraitNumTwo的b64 RegTensor的指令。该参数默认值为RegTraitNumOne。</p>
</td>
</tr>
<tr id="row1417671716463"><td class="cellrowborder" valign="top" width="13.268673132686734%" headers="mcps1.2.4.1.1 "><p id="p12176201754617"><a name="p12176201754617"></a><a name="p12176201754617"></a>scalarValue</p>
</td>
<td class="cellrowborder" valign="top" width="18.36816318368163%" headers="mcps1.2.4.1.2 "><p id="p1317616173468"><a name="p1317616173468"></a><a name="p1317616173468"></a>输入/输出</p>
</td>
<td class="cellrowborder" valign="top" width="68.36316368363164%" headers="mcps1.2.4.1.3 "><p id="p6176101734618"><a name="p6176101734618"></a><a name="p6176101734618"></a>矢量计算需要操作的元素的具体数量，生成对应的MaskReg，元素有效范围从0到VL_T（位宽为VL的T类型元素个数）。</p>
<p id="p16176917124620"><a name="p16176917124620"></a><a name="p16176917124620"></a>执行完该函数后，scalarValue会减去VL_T。</p>
<p id="p2176141714466"><a name="p2176141714466"></a><a name="p2176141714466"></a>scalarValue = (scalarValue &lt; VL_T) ? 0 : (scalarValue - VL_T)</p>
</td>
</tr>
</tbody>
</table>

## 返回值说明<a name="section1575141714439"></a>

MaskReg

## 支持的型号<a name="section156721693504"></a>

Ascend 950PR/Ascend 950DT

## 约束说明<a name="section177921451558"></a>

无

## 调用示例<a name="section642mcpsimp"></a>

```
AscendC::MicroAPI::RegTensor<uint32_t> srcReg;
AscendC::MicroAPI::MaskReg mask0 = AscendC::MicroAPI::CreateMask<uint32_t,AscendC::MicroAPI:: MaskPattern::ALL >();
AscendC::MicroAPI::MaskReg mask1;
uint32_t scalarValue = 127;
for (uint16_t i = 0; i < 2; i++) {
    mask1 = AscendC::MicroAPI::UpdateMask<uint32_t>(scalarValue);
    AscendC::MicroAPI::LoadAlign<T, AscendC::MicroAPI::PostLiteral::POST_MODE_UPDATE>(srcReg, srcAddr, 0);
    AscendC::MicroAPI::Adds(srcReg, srcReg, 1, mask0);
    AscendC::MicroAPI::StoreAlign<T, AscendC::MicroAPI::PostLiteral::POST_MODE_UPDATE>(dst0Addr, srcReg, 0, mask0);
    AscendC::MicroAPI::StoreAlign<T, AscendC::MicroAPI::PostLiteral::POST_MODE_UPDATE>(dst1Addr, srcReg, 0, mask1);
}
```

