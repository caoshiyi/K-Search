# ReduceDataBlock<a name="ZH-CN_TOPIC_0000002523344288"></a>

## 产品支持情况<a name="section1550532418810"></a>

<a name="table38301303189"></a>
<table><thead align="left"><tr id="row20831180131817"><th class="cellrowborder" valign="top" width="57.99999999999999%" id="mcps1.1.3.1.1"><p id="p1883113061818"><a name="p1883113061818"></a><a name="p1883113061818"></a><span id="ph20833205312295"><a name="ph20833205312295"></a><a name="ph20833205312295"></a>产品</span></p>
</th>
<th class="cellrowborder" align="center" valign="top" width="42%" id="mcps1.1.3.1.2"><p id="p783113012187"><a name="p783113012187"></a><a name="p783113012187"></a>是否支持</p>
</th>
</tr>
</thead>
<tbody><tr id="row1272474920205"><td class="cellrowborder" valign="top" width="57.99999999999999%" headers="mcps1.1.3.1.1 "><p id="p1914517152717"><a name="p1914517152717"></a><a name="p1914517152717"></a><span id="ph141589125420"><a name="ph141589125420"></a><a name="ph141589125420"></a>Ascend 950PR/Ascend 950DT</span></p>
</td>
<td class="cellrowborder" align="center" valign="top" width="42%" headers="mcps1.1.3.1.2 "><p id="p37256491200"><a name="p37256491200"></a><a name="p37256491200"></a>√</p>
</td>
</tr>
</tbody>
</table>

## 功能说明<a name="section618mcpsimp"></a>

当ReduceType 为SUM，将每个DataBlock（32B）中参与计算的元素相加，并将最终的计算结果依次保存在dstReg的最低位。

当ReduceType 为MAX，将每个DataBlock（32B）中的最大值，并将最终的计算结果依次保存在dstReg中的最低位。

当ReduceType 为MIN，将每个DataBlock（32B）中的最小值，并将最终的计算结果依次保存在dstReg中的最低位。

## 定义原型<a name="section620mcpsimp"></a>

```
template <ReduceType type = ReduceType::SUM, typename T = DefaultType, MaskMergeMode mode = MaskMergeMode::ZEROING, typename U>
__simd_callee__ inline void ReduceDataBlock(U& dstReg, U srcReg, MaskReg mask)
```

## 参数说明<a name="section622mcpsimp"></a>

**表 1**  模板参数说明

<a name="table4835205712588"></a>
<table><thead align="left"><tr id="row118356578583"><th class="cellrowborder" valign="top" width="36.42%" id="mcps1.2.3.1.1"><p id="p48354572582"><a name="p48354572582"></a><a name="p48354572582"></a>参数名</p>
</th>
<th class="cellrowborder" valign="top" width="63.580000000000005%" id="mcps1.2.3.1.2"><p id="p583535795817"><a name="p583535795817"></a><a name="p583535795817"></a>描述</p>
</th>
</tr>
</thead>
<tbody><tr id="row93781916153013"><td class="cellrowborder" valign="top" width="36.42%" headers="mcps1.2.3.1.1 "><p id="p037871683015"><a name="p037871683015"></a><a name="p037871683015"></a>type</p>
</td>
<td class="cellrowborder" valign="top" width="63.580000000000005%" headers="mcps1.2.3.1.2 "><p id="p10378181623016"><a name="p10378181623016"></a><a name="p10378181623016"></a>ReduceType类型，支持SUM、MAX、MIN。</p>
<pre class="screen" id="screen15472194923914"><a name="screen15472194923914"></a><a name="screen15472194923914"></a>enum class ReduceType {
    SUM = 0,
    MAX,
    MIN,
};</pre>
</td>
</tr>
<tr id="row1835857145817"><td class="cellrowborder" valign="top" width="36.42%" headers="mcps1.2.3.1.1 "><p id="p5835457165816"><a name="p5835457165816"></a><a name="p5835457165816"></a>T</p>
</td>
<td class="cellrowborder" valign="top" width="63.580000000000005%" headers="mcps1.2.3.1.2 "><p id="p168351657155818"><a name="p168351657155818"></a><a name="p168351657155818"></a>目的操作数和源操作数的数据类型。</p>
<p id="p7844541195218"><a name="p7844541195218"></a><a name="p7844541195218"></a><span id="ph158441241135214"><a name="ph158441241135214"></a><a name="ph158441241135214"></a>Ascend 950PR/Ascend 950DT</span>, 支持的数据类型为：int16_t/uint16_t/half/int32_t/uint32_t/float</p>
</td>
</tr>
<tr id="row3742113618507"><td class="cellrowborder" valign="top" width="36.42%" headers="mcps1.2.3.1.1 "><p id="p1383515717581"><a name="p1383515717581"></a><a name="p1383515717581"></a>mode</p>
</td>
<td class="cellrowborder" valign="top" width="63.580000000000005%" headers="mcps1.2.3.1.2 "><p id="p77520541653"><a name="p77520541653"></a><a name="p77520541653"></a>选择MERGING模式或ZEROING模式。当前仅支持ZEROING模式。</p>
<a name="ul1163765616511"></a><a name="ul1163765616511"></a><ul id="ul1163765616511"><li>ZEROING，mask未筛选的元素在dst中置零。</li></ul>
</td>
</tr>
<tr id="row11021944191014"><td class="cellrowborder" valign="top" width="36.42%" headers="mcps1.2.3.1.1 "><p id="p916243141912"><a name="p916243141912"></a><a name="p916243141912"></a>U</p>
</td>
<td class="cellrowborder" valign="top" width="63.580000000000005%" headers="mcps1.2.3.1.2 "><p id="p131047476612"><a name="p131047476612"></a><a name="p131047476612"></a>目的操作数和源操作数的RegTensor类型， 例如RegTensor&lt;int32_t&gt;，由编译器自动推导，用户不需要填写。</p>
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
<p id="p153781910126"><a name="p153781910126"></a><a name="p153781910126"></a>类型为<a href="RegTensor.md">RegTensor</a></p>
</td>
</tr>
<tr id="row17332152321719"><td class="cellrowborder" valign="top" width="18.54%" headers="mcps1.2.4.1.1 "><p id="p1538745616196"><a name="p1538745616196"></a><a name="p1538745616196"></a>mask</p>
</td>
<td class="cellrowborder" valign="top" width="10.05%" headers="mcps1.2.4.1.2 "><p id="p73331523101710"><a name="p73331523101710"></a><a name="p73331523101710"></a>输入</p>
</td>
<td class="cellrowborder" valign="top" width="71.41%" headers="mcps1.2.4.1.3 "><p id="p11541143920"><a name="p11541143920"></a><a name="p11541143920"></a><span id="ph15776181222"><a name="ph15776181222"></a><a name="ph15776181222"></a>源操作数元素操作的有效指示，详细说明请参考<a href="MaskReg.md">MaskReg</a>。</span></p>
</td>
</tr>
</tbody>
</table>

## 约束说明<a name="section177921451558"></a>

无

## 调用示例<a name="section642mcpsimp"></a>

```
template<typename T>
__simd_vf__ inline void ReduceDataBlockVF(__ubuf__ T* dstAddr, __ubuf__ T* srcAddr, uint32_t count, 
 uint32_t oneRepeatSize, uint16_t repeatTimes)
{
    AscendC::MicroAPI::RegTensor<T> srcReg;
    AscendC::MicroAPI::RegTensor<T> dstReg;
    AscendC::MicroAPI::MaskReg mask;
    for (uint16_t i = 0; i < repeatTimes; i++) {
        AscendC::MicroAPI::LoadAlign(srcReg, srcAddr + i * oneRepeatSize);
        mask = AscendC::MicroAPI::UpdateMask<T>(count);
        // type = ReduceType::SUM
        AscendC::MicroAPI::ReduceDataBlock<AscendC::MicroAPI::ReduceType::SUM>(dstReg, srcReg, mask);
        // type = ReduceType::MAX
        // AscendC::MicroAPI::ReduceDataBlock<AscendC::MicroAPI::ReduceType::MAX>(dstReg, srcReg, mask);
        // type = ReduceType::MIN
        // AscendC::MicroAPI::ReduceDataBlock<AscendC::MicroAPI::ReduceType::MIN>(dstReg, srcReg, mask);
        AscendC::MicroAPI::StoreAlign(dstAddr + i * oneRepeatSize, dstReg, mask); 
    }
}
```

