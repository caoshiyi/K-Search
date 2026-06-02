# Gather（支持寄存器为源操作数）<a name="ZH-CN_TOPIC_0000002523304824"></a>

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

根据索引位置indexReg将源操作数srcReg按元素收集到结果dstReg中。收集过程如下图所示：

**图 1**  Gather功能说明<a name="fig122102710193"></a>  
<!-- img2text -->
```text
srcReg
┌───────┬───────┬───────┬───────┬───────┬───────┬──────┬───────┐
│ elm0  │ elm1  │ elm2  │ elm3  │ elm4  │ elm5  │ .....│ elmn  │
└───────┴───────┴───────┴───────┴───────┴───────┴──────┴───────┘
    └──────────────────────────────┐              ┌───────────────┘
           └──────────────┐        │        ┌────┘
                  └───────┼────────┼────────┘
                          ↓        ↓
indexReg
┌───────┬───────┬───────┬───────┬───────┬───────┬──────┬───────┐
│   3   │   2   │   4   │   1   │   1   │   0   │ .....│   3   │
└───────┴───────┴───────┴───────┴───────┴───────┴──────┴───────┘
    │       │       │       │       │       │              │
    ↓       ↓       ↓       ↓       ↓       ↓              ↓
dstReg
┌───────┬───────┬───────┬───────┬───────┬───────┬──────┬───────┐
│ elm3  │ elm2  │ elm4  │ elm1  │ elm1  │ elm0  │ .....│ elm3  │
└───────┴───────┴───────┴───────┴───────┴───────┴──────┴───────┘
```

说明:
- indexReg 中的每个索引值表示从 srcReg 中取对应位置的元素，写入 dstReg 的同位置。
- 对应关系:
  - indexReg[0] = 3  → dstReg[0] = elm3
  - indexReg[1] = 2  → dstReg[1] = elm2
  - indexReg[2] = 4  → dstReg[2] = elm4
  - indexReg[3] = 1  → dstReg[3] = elm1
  - indexReg[4] = 1  → dstReg[4] = elm1
  - indexReg[5] = 0  → dstReg[5] = elm0
  - indexReg[n] = 3  → dstReg[n] = elm3

## 定义原型<a name="section620mcpsimp"></a>

```
template <typename T = DefaultType, typename U = DefaultType, typename S, typename V>
__simd_callee__ inline void Gather(S& dstReg, S& srcReg, V& indexReg)
```

## 参数说明<a name="section622mcpsimp"></a>

**表 1**  模板参数说明

<a name="table4835205712588"></a>
<table><thead align="left"><tr id="row118356578583"><th class="cellrowborder" valign="top" width="18.13%" id="mcps1.2.3.1.1"><p id="p48354572582"><a name="p48354572582"></a><a name="p48354572582"></a>参数名</p>
</th>
<th class="cellrowborder" valign="top" width="81.87%" id="mcps1.2.3.1.2"><p id="p583535795817"><a name="p583535795817"></a><a name="p583535795817"></a>描述</p>
</th>
</tr>
</thead>
<tbody><tr id="row1835857145817"><td class="cellrowborder" valign="top" width="18.13%" headers="mcps1.2.3.1.1 "><p id="p5835457165816"><a name="p5835457165816"></a><a name="p5835457165816"></a>T</p>
</td>
<td class="cellrowborder" valign="top" width="81.87%" headers="mcps1.2.3.1.2 "><p id="p168351657155818"><a name="p168351657155818"></a><a name="p168351657155818"></a>目的操作数和源操作数的数据类型。</p>
<p id="p153204151019"><a name="p153204151019"></a><a name="p153204151019"></a><span id="ph133209151609"><a name="ph133209151609"></a><a name="ph133209151609"></a>Ascend 950PR/Ascend 950DT</span>，支持的数据类型为：b8/b16/b32</p>
</td>
</tr>
<tr id="row145134204158"><td class="cellrowborder" valign="top" width="18.13%" headers="mcps1.2.3.1.1 "><p id="p17514172015155"><a name="p17514172015155"></a><a name="p17514172015155"></a>U</p>
</td>
<td class="cellrowborder" valign="top" width="81.87%" headers="mcps1.2.3.1.2 "><p id="p9514132013155"><a name="p9514132013155"></a><a name="p9514132013155"></a>索引值的数据类型。</p>
<p id="p1332134951510"><a name="p1332134951510"></a><a name="p1332134951510"></a><span id="ph63211249121510"><a name="ph63211249121510"></a><a name="ph63211249121510"></a>Ascend 950PR/Ascend 950DT</span>，支持的数据类型为：uint8_t/uint16_t/uint32_t</p>
</td>
</tr>
<tr id="row3789349195116"><td class="cellrowborder" valign="top" width="18.13%" headers="mcps1.2.3.1.1 "><p id="p578954925110"><a name="p578954925110"></a><a name="p578954925110"></a>S</p>
</td>
<td class="cellrowborder" valign="top" width="81.87%" headers="mcps1.2.3.1.2 "><p id="p137898497514"><a name="p137898497514"></a><a name="p137898497514"></a><span id="ph19851723182011"><a name="ph19851723182011"></a><a name="ph19851723182011"></a>目的操作数的RegTensor类型，例如RegTensor&lt;half&gt;，由编译器自动推导，用户不需要填写。</span></p>
</td>
</tr>
<tr id="row13194205214516"><td class="cellrowborder" valign="top" width="18.13%" headers="mcps1.2.3.1.1 "><p id="p2194152115116"><a name="p2194152115116"></a><a name="p2194152115116"></a>V</p>
</td>
<td class="cellrowborder" valign="top" width="81.87%" headers="mcps1.2.3.1.2 "><p id="p719455205119"><a name="p719455205119"></a><a name="p719455205119"></a>索引值的RegTensor类型，例如RegTensor&lt;uint16_t&gt;，由编译器自动推导，用户不需要填写。</p>
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
<tbody><tr id="row42461942101815"><td class="cellrowborder" valign="top" width="18.54%" headers="mcps1.2.4.1.1 "><p id="p1435142552019"><a name="p1435142552019"></a><a name="p1435142552019"></a>dstReg</p>
</td>
<td class="cellrowborder" valign="top" width="10.05%" headers="mcps1.2.4.1.2 "><p id="p158449584436"><a name="p158449584436"></a><a name="p158449584436"></a>输出</p>
</td>
<td class="cellrowborder" valign="top" width="71.41%" headers="mcps1.2.4.1.3 "><p id="p212236122019"><a name="p212236122019"></a><a name="p212236122019"></a>目的操作数。</p>
<p id="p2507123615484"><a name="p2507123615484"></a><a name="p2507123615484"></a><span id="ph950733612488"><a name="ph950733612488"></a><a name="ph950733612488"></a>类型为<a href="RegTensor.md">RegTensor</a>。</span></p>
</td>
</tr>
<tr id="row2137145181815"><td class="cellrowborder" valign="top" width="18.54%" headers="mcps1.2.4.1.1 "><p id="p6844125874315"><a name="p6844125874315"></a><a name="p6844125874315"></a>srcReg</p>
</td>
<td class="cellrowborder" valign="top" width="10.05%" headers="mcps1.2.4.1.2 "><p id="p128442058144312"><a name="p128442058144312"></a><a name="p128442058144312"></a>输入</p>
</td>
<td class="cellrowborder" valign="top" width="71.41%" headers="mcps1.2.4.1.3 "><p id="p14827161917207"><a name="p14827161917207"></a><a name="p14827161917207"></a>源操作数。</p>
<p id="p7123111612517"><a name="p7123111612517"></a><a name="p7123111612517"></a><span id="ph1197863231916"><a name="ph1197863231916"></a><a name="ph1197863231916"></a>类型为<a href="RegTensor.md">RegTensor</a>。</span></p>
<p id="p1484485824312"><a name="p1484485824312"></a><a name="p1484485824312"></a>数据类型需要与目的操作数保持一致。</p>
</td>
</tr>
<tr id="row891912431168"><td class="cellrowborder" valign="top" width="18.54%" headers="mcps1.2.4.1.1 "><p id="p195756503168"><a name="p195756503168"></a><a name="p195756503168"></a>indexReg</p>
</td>
<td class="cellrowborder" valign="top" width="10.05%" headers="mcps1.2.4.1.2 "><p id="p457515071618"><a name="p457515071618"></a><a name="p457515071618"></a>输入</p>
</td>
<td class="cellrowborder" valign="top" width="71.41%" headers="mcps1.2.4.1.3 "><p id="p193020233012"><a name="p193020233012"></a><a name="p193020233012"></a>数据索引。</p>
<p id="p1728583820487"><a name="p1728583820487"></a><a name="p1728583820487"></a><span id="ph72855386487"><a name="ph72855386487"></a><a name="ph72855386487"></a>类型为<a href="RegTensor.md">RegTensor</a>。</span></p>
<p id="p104218390187"><a name="p104218390187"></a><a name="p104218390187"></a>数据类型的位宽需要与目的操作数的位宽保持一致。</p>
<p id="p1396715519200"><a name="p1396715519200"></a><a name="p1396715519200"></a>srcReg为RegTensor类型，位宽是固定的VL，存储的元素个数固定。如果indexReg中索引值超出当前RegTensor中能存储的最大数据元素个数时，按照如下方式处理：设定当前RegTensor所能存储的最大数据元素个数为vlLength，indexReg中索引值为i，索引值更新为i % vlLength。</p>
</td>
</tr>
</tbody>
</table>

## 约束说明<a name="section177921451558"></a>

无

## 调用示例<a name="section642mcpsimp"></a>

```
template<typename T, typename U>
__simd_vf__ inline void GatherVF(__ubuf__ T* dstAddr, __ubuf__ T* src0Addr, __ubuf__ U* src1Addr, uint32_t count, uint32_t oneRepeatSize, uint16_t repeatTimes)
{
    AscendC::MicroAPI::RegTensor<T> srcReg0, dstReg;
    AscendC::MicroAPI::RegTensor<U> srcReg1;
    AscendC::MicroAPI::MaskReg mask;
    AscendC::MicroAPI::LoadAlign(srcReg1, src1Addr);
    for (uint16_t i = 0; i < repeatTimes; i++) {
        mask = AscendC::MicroAPI::UpdateMask<T>(count);
        AscendC::MicroAPI::LoadAlign(srcReg0, src0Addr + i * oneRepeatSize);
        AscendC::MicroAPI::Gather(dstReg, srcReg0, srcReg1);
        AscendC::MicroAPI::StoreAlign(dstAddr + i * oneRepeatSize, dstReg, mask);
    }
}
```

