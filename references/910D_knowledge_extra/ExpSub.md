# ExpSub<a name="ZH-CN_TOPIC_0000002554424181"></a>

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

src0与src1相减，差值作为e的指数计算， 根据mask将计算结果写入dst。

src为float类型时：<!-- img2text -->
$$dst_i=\left\{\begin{array}{ll}
e^{src0_i-src1_i}, & \text{mask位为1} \\
dst_i, & \text{mask位为0}
\end{array}\right.$$

src为half类型时：<!-- img2text -->
$$
dst_i=\begin{cases}
e^{(src0_i-src1_i)}, & \text{if mask bit } i = 1 \\
0, & \text{otherwise}
\end{cases}
$$

## 函数原型<a name="section620mcpsimp"></a>

```
template <typename T = DefaultType, typename U = DefaultType, RegLayout layout = RegLayout::ZERO, MaskMergeMode mode = MaskMergeMode::ZEROING, typename S, typename V>
__simd_callee__ inline void ExpSub(S& dstReg, V& srcReg0, V& srcReg1, MaskReg& mask)
```

## 参数说明<a name="section622mcpsimp"></a>

**表 1**  模板参数说明

<a name="table4835205712588"></a>
<table><thead align="left"><tr id="row118356578583"><th class="cellrowborder" valign="top" width="18.48%" id="mcps1.2.3.1.1"><p id="p48354572582"><a name="p48354572582"></a><a name="p48354572582"></a>参数名</p>
</th>
<th class="cellrowborder" valign="top" width="81.52000000000001%" id="mcps1.2.3.1.2"><p id="p583535795817"><a name="p583535795817"></a><a name="p583535795817"></a>描述</p>
</th>
</tr>
</thead>
<tbody><tr id="row1835857145817"><td class="cellrowborder" valign="top" width="18.48%" headers="mcps1.2.3.1.1 "><p id="p5835457165816"><a name="p5835457165816"></a><a name="p5835457165816"></a>T</p>
</td>
<td class="cellrowborder" valign="top" width="81.52000000000001%" headers="mcps1.2.3.1.2 "><p id="p168351657155818"><a name="p168351657155818"></a><a name="p168351657155818"></a>目的操作数数据类型。</p>
<p id="p96431308228"><a name="p96431308228"></a><a name="p96431308228"></a><span id="ph7643907227"><a name="ph7643907227"></a><a name="ph7643907227"></a>Ascend 950PR/Ascend 950DT</span>，支持的数据类型为：float</p>
</td>
</tr>
<tr id="row396814422169"><td class="cellrowborder" valign="top" width="18.48%" headers="mcps1.2.3.1.1 "><p id="p89687420164"><a name="p89687420164"></a><a name="p89687420164"></a>U</p>
</td>
<td class="cellrowborder" valign="top" width="81.52000000000001%" headers="mcps1.2.3.1.2 "><p id="p1896874212162"><a name="p1896874212162"></a><a name="p1896874212162"></a>源操作数数据类型。</p>
<p id="p8238111018225"><a name="p8238111018225"></a><a name="p8238111018225"></a><span id="ph192381110152211"><a name="ph192381110152211"></a><a name="ph192381110152211"></a>Ascend 950PR/Ascend 950DT</span>，支持的数据类型为：half/float</p>
</td>
</tr>
<tr id="row105019122115"><td class="cellrowborder" valign="top" width="18.48%" headers="mcps1.2.3.1.1 "><p id="p95016121914"><a name="p95016121914"></a><a name="p95016121914"></a>layout</p>
</td>
<td class="cellrowborder" valign="top" width="81.52000000000001%" headers="mcps1.2.3.1.2 "><p id="p1433681723313"><a name="p1433681723313"></a><a name="p1433681723313"></a><a href="RegLayout.md">RegLayout</a>枚举类型：</p>
<pre class="screen" id="screen18481935171419"><a name="screen18481935171419"></a><a name="screen18481935171419"></a>enum class RegLayout {
    UNKNOWN = -1,
    ZERO,
    ONE,
    TWO,
    THREE
};</pre>
<p id="p150112121614"><a name="p150112121614"></a><a name="p150112121614"></a>本接口只支持RegLayout::ZERO、RegLayout::ONE。src类型为half类型时使用，float时不生效，half类型时，RegLayout::ZERO表示从b16 RegTensor偶数位读取half元素转换成float，RegLayout::ONE表示从b16 RegTensor奇数位读取half元素转换成float。</p>
</td>
</tr>
<tr id="row356441781813"><td class="cellrowborder" valign="top" width="18.48%" headers="mcps1.2.3.1.1 "><p id="p1756419170189"><a name="p1756419170189"></a><a name="p1756419170189"></a>mode</p>
</td>
<td class="cellrowborder" valign="top" width="81.52000000000001%" headers="mcps1.2.3.1.2 "><p id="p77520541653"><a name="p77520541653"></a><a name="p77520541653"></a>选择MERGING模式或ZEROING模式。</p>
<a name="ul1163765616511"></a><a name="ul1163765616511"></a><ul id="ul1163765616511"><li>ZEROING，mask未筛选的元素在dst中置零。</li><li>MERGING，当前不支持。</li></ul>
</td>
</tr>
<tr id="row916216311197"><td class="cellrowborder" valign="top" width="18.48%" headers="mcps1.2.3.1.1 "><p id="p916243141912"><a name="p916243141912"></a><a name="p916243141912"></a>S</p>
</td>
<td class="cellrowborder" valign="top" width="81.52000000000001%" headers="mcps1.2.3.1.2 "><p id="p131047476612"><a name="p131047476612"></a><a name="p131047476612"></a>dstReg RegTensor类型， 例如RegTensor&lt;half&gt;，由编译器自动推导，用户不需要填写。</p>
</td>
</tr>
<tr id="row17951175431514"><td class="cellrowborder" valign="top" width="18.48%" headers="mcps1.2.3.1.1 "><p id="p495215411152"><a name="p495215411152"></a><a name="p495215411152"></a>V</p>
</td>
<td class="cellrowborder" valign="top" width="81.52000000000001%" headers="mcps1.2.3.1.2 "><p id="p1217321168"><a name="p1217321168"></a><a name="p1217321168"></a>srcReg0/srcReg1 RegTensor类型， 例如RegTensor&lt;half&gt;，由编译器自动推导，用户不需要填写。</p>
</td>
</tr>
</tbody>
</table>

**表 2**  参数说明

<a name="zh-cn_topic_0235751031_table33761356"></a>
<table><thead align="left"><tr id="zh-cn_topic_0235751031_row27598891"><th class="cellrowborder" valign="top" width="18.54%" id="mcps1.2.4.1.1"><p id="zh-cn_topic_0235751031_p20917673"><a name="zh-cn_topic_0235751031_p20917673"></a><a name="zh-cn_topic_0235751031_p20917673"></a>参数名</p>
</th>
<th class="cellrowborder" valign="top" width="10.05%" id="mcps1.2.4.1.2"><p id="zh-cn_topic_0235751031_p16609919"><a name="zh-cn_topic_0235751031_p16609919"></a><a name="zh-cn_topic_0235751031_p16609919"></a>输入/输出</p>
</th>
<th class="cellrowborder" valign="top" width="71.41%" id="mcps1.2.4.1.3"><p id="zh-cn_topic_0235751031_p59995477"><a name="zh-cn_topic_0235751031_p59995477"></a><a name="zh-cn_topic_0235751031_p59995477"></a>描述</p>
</th>
</tr>
</thead>
<tbody><tr id="row42461942101815"><td class="cellrowborder" valign="top" width="18.54%" headers="mcps1.2.4.1.1 "><p id="p108051250181214"><a name="p108051250181214"></a><a name="p108051250181214"></a>dstReg</p>
</td>
<td class="cellrowborder" valign="top" width="10.05%" headers="mcps1.2.4.1.2 "><p id="p158449584436"><a name="p158449584436"></a><a name="p158449584436"></a>输出</p>
</td>
<td class="cellrowborder" valign="top" width="71.41%" headers="mcps1.2.4.1.3 "><p id="p6389114616514"><a name="p6389114616514"></a><a name="p6389114616514"></a>目的操作数。</p>
<p id="p66093533169"><a name="p66093533169"></a><a name="p66093533169"></a>类型为<a href="RegTensor.md">RegTensor</a>。</p>
</td>
</tr>
<tr id="row2137145181815"><td class="cellrowborder" valign="top" width="18.54%" headers="mcps1.2.4.1.1 "><p id="p19574165615129"><a name="p19574165615129"></a><a name="p19574165615129"></a>srcReg0</p>
</td>
<td class="cellrowborder" valign="top" width="10.05%" headers="mcps1.2.4.1.2 "><p id="p128442058144312"><a name="p128442058144312"></a><a name="p128442058144312"></a>输入</p>
</td>
<td class="cellrowborder" valign="top" width="71.41%" headers="mcps1.2.4.1.3 "><p id="p172083541517"><a name="p172083541517"></a><a name="p172083541517"></a>源操作数。</p>
<p id="p7123111612517"><a name="p7123111612517"></a><a name="p7123111612517"></a>类型为<a href="RegTensor.md">RegTensor</a>。</p>
</td>
</tr>
<tr id="row11773440341"><td class="cellrowborder" valign="top" width="18.54%" headers="mcps1.2.4.1.1 "><p id="p10132448173420"><a name="p10132448173420"></a><a name="p10132448173420"></a>srcReg1</p>
</td>
<td class="cellrowborder" valign="top" width="10.05%" headers="mcps1.2.4.1.2 "><p id="p151325484342"><a name="p151325484342"></a><a name="p151325484342"></a>输入</p>
</td>
<td class="cellrowborder" valign="top" width="71.41%" headers="mcps1.2.4.1.3 "><p id="p181326485341"><a name="p181326485341"></a><a name="p181326485341"></a>源操作数。</p>
<p id="p74744436423"><a name="p74744436423"></a><a name="p74744436423"></a>类型为<a href="RegTensor.md">RegTensor</a>。</p>
</td>
</tr>
<tr id="row19615183817191"><td class="cellrowborder" valign="top" width="18.54%" headers="mcps1.2.4.1.1 "><p id="p1484519586432"><a name="p1484519586432"></a><a name="p1484519586432"></a>mask</p>
</td>
<td class="cellrowborder" valign="top" width="10.05%" headers="mcps1.2.4.1.2 "><p id="p484514581433"><a name="p484514581433"></a><a name="p484514581433"></a>输入</p>
</td>
<td class="cellrowborder" valign="top" width="71.41%" headers="mcps1.2.4.1.3 "><p id="p11541143920"><a name="p11541143920"></a><a name="p11541143920"></a><span id="ph15776181222"><a name="ph15776181222"></a><a name="ph15776181222"></a>源操作数元素操作的有效指示，详细说明请参考<a href="MaskReg.md">MaskReg</a>。</span></p>
</td>
</tr>
</tbody>
</table>

## 返回值说明<a name="section640mcpsimp"></a>

无

## 约束说明<a name="section633mcpsimp"></a>

src为half类型时，Vector计算单元一次计算只处理最多VL/sizeof\(float\)个half元素，maskReg只有偶数位有效，参考下图：

<!-- img2text -->
```text
┌────┬────┬────┬────┬────┬────┬────┬────┐
│ 0  │ 1  │ 2  │ 3  │ 4  │ 5  │ 6  │ 7  │
├────┼────┼────┼────┼────┼────┼────┼────┤
│ 1  │ 1  │ 1  │ 1  │ 1  │ 0  │ 0  │ 0  │
└────┴────┴────┴────┴────┴────┴────┴────┘

┌────┬────┬────┬────┬────┬────┬────┬────┐
│ 16 │ 16 │ 16 │ 16 │ 16 │ 16 │ 16 │ 16 │
└────┴────┴────┴────┴────┴────┴────┴────┘
                  ↓
                  ↓

┌────────┬────────┬────────┬────────┐
│   32   │   32   │   32   │   32   │
└────────┴────────┴────────┴────────┘

┌────┬────┬────┬────┬────┬────┬────┬────┐
│ 16 │ 16 │ 16 │ 16 │ 16 │ 16 │ 16 │ 16 │
└────┴────┴────┴────┴────┴────┴────┴────┘
                  ↓
                  ↓

┌────────┬────────┬────────┬────────┐
│   32   │   32   │   32   │   32   │
└────────┴────────┴────────┴────────┘
```

说明:
- 顶部 0~7 表示 maskReg 位索引。
- maskReg 对应值为：第 0~4 位为 1，第 5~7 位为 0。
- `src` 为 `half` 类型时，Vector 计算单元一次计算只处理最多 `VL/sizeof(float)` 个 `half` 元素。
- `maskReg` 只有偶数位有效。
- 第一组 `16` 合并为 `32` 时，有效对应为前 3 个 `32` 块；最后 1 个 `32` 块无效。
- 第二组 `16` 合并为 `32` 时，有效对应为前 2 个 `32` 块；后 2 个 `32` 块无效。

## 调用示例<a name="section642mcpsimp"></a>

```
template<typename T, typename U>
static __simd_vf__ inline void ExpSubVF(__ubuf__ T* dstAddr, __ubuf__ U* src0Addr, __ubuf__ U* src1Addr, uint32_t count, uint32_t srcRepeatSize, uint32_t dstRepeatSize, uint16_t repeatTimes)
{
    AscendC::MicroAPI::RegTensor<U> srcReg0;
    AscendC::MicroAPI::RegTensor<U> srcReg1;
    AscendC::MicroAPI::RegTensor<T> dstReg;
    AscendC::MicroAPI::MaskReg mask;
    for (uint16_t i = 0; i < repeatTimes; i++) {
        mask = AscendC::MicroAPI::UpdateMask<U>(count);
        AscendC::MicroAPI::LoadAlign(srcReg0, src0Addr + i * srcRepeatSize);
        AscendC::MicroAPI::LoadAlign(srcReg1, src1Addr + i * srcRepeatSize);
        AscendC::MicroAPI::ExpSub<T, U, AscendC::MicroAPI::RegLayout::ZERO, AscendC::MicroAPI::MaskMergeMode::ZEROING>(dstReg, srcReg0, srcReg1, mask);
        AscendC::MicroAPI::StoreAlign(dstAddr + i * dstRepeatSize, dstReg, mask);
    }
}
```

