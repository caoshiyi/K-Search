# 非连续对齐搬入<a name="ZH-CN_TOPIC_0000002554424771"></a>

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

Reg矢量计算数据搬运接口，适用于从UB非连续对齐搬入RegTensor（以DataBlock为单位）。

## 函数原型<a name="section620mcpsimp"></a>

```
// 正常场景
template <typename T = DefaultType, DataCopyMode dataMode, typename U>
__simd_callee__ inline void LoadAlign(U& dstReg, __ubuf__ T* srcAddr, uint32_t dataBlockStride, MaskReg& mask);
// POST_MODE_UPDATE场景
template <typename T = DefaultType, DataCopyMode dataMode, PostLiteral postMode, typename U>
__simd_callee__ inline void LoadAlign(U& dstReg, __ubuf__ T*& srcAddr, uint32_t dataBlockStride, uint32_t repeatStride, MaskReg& mask);
```

## 参数说明<a name="section622mcpsimp"></a>

**表 1**  DataCopyMode模板参数说明

<a name="table166256106711"></a>
<table><thead align="left"><tr id="row1762513105713"><th class="cellrowborder" valign="top" width="20.61%" id="mcps1.2.3.1.1"><p id="p18625210579"><a name="p18625210579"></a><a name="p18625210579"></a>DataCopyMode取值</p>
</th>
<th class="cellrowborder" valign="top" width="79.39%" id="mcps1.2.3.1.2"><p id="p462519101679"><a name="p462519101679"></a><a name="p462519101679"></a>含义</p>
</th>
</tr>
</thead>
<tbody><tr id="row14625010077"><td class="cellrowborder" valign="top" width="20.61%" headers="mcps1.2.3.1.1 "><p id="p914713321272"><a name="p914713321272"></a><a name="p914713321272"></a>DATA_BLOCK_COPY</p>
</td>
<td class="cellrowborder" valign="top" width="79.39%" headers="mcps1.2.3.1.2 "><p id="p062561016719"><a name="p062561016719"></a><a name="p062561016719"></a>用来在非连续对齐场景选择搬运模式。当前仅支持DataBlock搬运模式，即以DataBlock为粒度进行搬运。</p>
</td>
</tr>
</tbody>
</table>

**表 2**  参数说明

<a name="table14132101714462"></a>
<table><thead align="left"><tr id="row19176617124620"><th class="cellrowborder" valign="top" width="13.268673132686734%" id="mcps1.2.4.1.1"><p id="p117621724612"><a name="p117621724612"></a><a name="p117621724612"></a>参数名</p>
</th>
<th class="cellrowborder" valign="top" width="12.468753124687531%" id="mcps1.2.4.1.2"><p id="p417681717463"><a name="p417681717463"></a><a name="p417681717463"></a>输入/输出</p>
</th>
<th class="cellrowborder" valign="top" width="74.26257374262573%" id="mcps1.2.4.1.3"><p id="p5176017164614"><a name="p5176017164614"></a><a name="p5176017164614"></a>描述</p>
</th>
</tr>
</thead>
<tbody><tr id="row0176117164610"><td class="cellrowborder" valign="top" width="13.268673132686734%" headers="mcps1.2.4.1.1 "><p id="p12176171764613"><a name="p12176171764613"></a><a name="p12176171764613"></a>T</p>
</td>
<td class="cellrowborder" valign="top" width="12.468753124687531%" headers="mcps1.2.4.1.2 "><p id="p5176191734615"><a name="p5176191734615"></a><a name="p5176191734615"></a>输入</p>
</td>
<td class="cellrowborder" valign="top" width="74.26257374262573%" headers="mcps1.2.4.1.3 "><p id="p3176717174617"><a name="p3176717174617"></a><a name="p3176717174617"></a>模板参数，支持的数据类型为b8/b16/b32。</p>
</td>
</tr>
<tr id="row1756520517490"><td class="cellrowborder" valign="top" width="13.268673132686734%" headers="mcps1.2.4.1.1 "><p id="p1556695114916"><a name="p1556695114916"></a><a name="p1556695114916"></a>postMode</p>
</td>
<td class="cellrowborder" valign="top" width="12.468753124687531%" headers="mcps1.2.4.1.2 "><p id="p1556695184918"><a name="p1556695184918"></a><a name="p1556695184918"></a>输入</p>
</td>
<td class="cellrowborder" valign="top" width="74.26257374262573%" headers="mcps1.2.4.1.3 "><p id="p14697155195612"><a name="p14697155195612"></a><a name="p14697155195612"></a>用于控制是否使能post update，<a href="PostLiteral.md">PostLiteral</a>类型。</p>
</td>
</tr>
<tr id="row1622359357"><td class="cellrowborder" valign="top" width="13.268673132686734%" headers="mcps1.2.4.1.1 "><p id="p1784518187125"><a name="p1784518187125"></a><a name="p1784518187125"></a>U</p>
</td>
<td class="cellrowborder" valign="top" width="12.468753124687531%" headers="mcps1.2.4.1.2 "><p id="p188451018141218"><a name="p188451018141218"></a><a name="p188451018141218"></a>输入</p>
</td>
<td class="cellrowborder" valign="top" width="74.26257374262573%" headers="mcps1.2.4.1.3 "><p id="p1784512188121"><a name="p1784512188121"></a><a name="p1784512188121"></a>RegTensor类型， 例如RegTensor&lt;half&gt;，由编译器自动推导，用户不需要填写。</p>
</td>
</tr>
<tr id="row1417671716463"><td class="cellrowborder" valign="top" width="13.268673132686734%" headers="mcps1.2.4.1.1 "><p id="p1039635944011"><a name="p1039635944011"></a><a name="p1039635944011"></a>dstReg</p>
</td>
<td class="cellrowborder" valign="top" width="12.468753124687531%" headers="mcps1.2.4.1.2 "><p id="p1317616173468"><a name="p1317616173468"></a><a name="p1317616173468"></a>输出</p>
</td>
<td class="cellrowborder" valign="top" width="74.26257374262573%" headers="mcps1.2.4.1.3 "><p id="p2176141714466"><a name="p2176141714466"></a><a name="p2176141714466"></a>目的操作数，类型为RegTensor。</p>
</td>
</tr>
<tr id="row958051624416"><td class="cellrowborder" valign="top" width="13.268673132686734%" headers="mcps1.2.4.1.1 "><p id="p9580161634416"><a name="p9580161634416"></a><a name="p9580161634416"></a>srcAddr</p>
</td>
<td class="cellrowborder" valign="top" width="12.468753124687531%" headers="mcps1.2.4.1.2 "><p id="p2581416144417"><a name="p2581416144417"></a><a name="p2581416144417"></a>输入/输出</p>
</td>
<td class="cellrowborder" valign="top" width="74.26257374262573%" headers="mcps1.2.4.1.3 "><p id="p195811716134418"><a name="p195811716134418"></a><a name="p195811716134418"></a>源操作数在UB上的起始地址。</p>
</td>
</tr>
<tr id="row874519644215"><td class="cellrowborder" valign="top" width="13.268673132686734%" headers="mcps1.2.4.1.1 "><p id="p1055695912137"><a name="p1055695912137"></a><a name="p1055695912137"></a>dataBlockStride</p>
</td>
<td class="cellrowborder" valign="top" width="12.468753124687531%" headers="mcps1.2.4.1.2 "><p id="p199441921164218"><a name="p199441921164218"></a><a name="p199441921164218"></a>输入</p>
</td>
<td class="cellrowborder" valign="top" width="74.26257374262573%" headers="mcps1.2.4.1.3 "><p id="p3745176144210"><a name="p3745176144210"></a><a name="p3745176144210"></a>单次搬运相邻DataBlock间的间隔（前面一个DataBlock的头与后面DataBlock的头的间隔），单位为DataBlock。</p>
</td>
</tr>
<tr id="row4397131094210"><td class="cellrowborder" valign="top" width="13.268673132686734%" headers="mcps1.2.4.1.1 "><p id="p639881018426"><a name="p639881018426"></a><a name="p639881018426"></a>repeatStride</p>
</td>
<td class="cellrowborder" valign="top" width="12.468753124687531%" headers="mcps1.2.4.1.2 "><p id="p1343482217424"><a name="p1343482217424"></a><a name="p1343482217424"></a>输入</p>
</td>
<td class="cellrowborder" valign="top" width="74.26257374262573%" headers="mcps1.2.4.1.3 "><p id="p20878111944912"><a name="p20878111944912"></a><a name="p20878111944912"></a>POST_MODE_NORMAL与POST_MODE_UPDATE场景下repeatStride含义不一致。</p>
<a name="ul2724014311"></a><a name="ul2724014311"></a><ul id="ul2724014311"><li>POST_MODE_NORMAL场景：实际搬运UB起始地址为srcAddr + repeatStride * 32。</li><li>POST_MODE_UPDATE场景：实际搬运UB起始地址为srcAddr，搬运后执行地址更新srcAddr +=  repeatStride * 32。</li></ul>
</td>
</tr>
<tr id="row173581146155819"><td class="cellrowborder" valign="top" width="13.268673132686734%" headers="mcps1.2.4.1.1 "><p id="p3358174615811"><a name="p3358174615811"></a><a name="p3358174615811"></a>mask</p>
</td>
<td class="cellrowborder" valign="top" width="12.468753124687531%" headers="mcps1.2.4.1.2 "><p id="p93581646115811"><a name="p93581646115811"></a><a name="p93581646115811"></a>输入</p>
</td>
<td class="cellrowborder" valign="top" width="74.26257374262573%" headers="mcps1.2.4.1.3 "><p id="p1735834665814"><a name="p1735834665814"></a><a name="p1735834665814"></a>MaskReg类型，指示在搬运过程中哪些DataBlock有效。</p>
<p id="p4283104525117"><a name="p4283104525117"></a><a name="p4283104525117"></a>某个DataBlock对应的32bit有任意一位为1，该DataBlock对应的数据会搬入到dst。</p>
<p id="p12374154312528"><a name="p12374154312528"></a><a name="p12374154312528"></a>某个DataBlock对应的32bit全为0时，该DataBlock对应的数据不会被读取，对应位置的dst设置为0，即使该UB越界也不会报错。</p>
</td>
</tr>
</tbody>
</table>

## 返回值说明<a name="section1575141714439"></a>

无

## 约束说明<a name="section11585101304320"></a>

无

## 调用示例<a name="section642mcpsimp"></a>

```
__simd_vf__ inline void Compute(__ubuf__ T* dstAddr, __ubuf__ T* srcAddr, uint16_t repeatTimes)
{
    AscendC::MicroAPI::RegTensor<T> srcReg;
    AscendC::MicroAPI::MaskReg mask = AscendC::MicroAPI::CreateMask<T>();
    for (uint16_t i = 0; i < repeatTimes; ++i) {
        AscendC::MicroAPI::LoadAlign<T, AscendC::MicroAPI::DataCopyMode::DATA_BLOCK_COPY, AscendC::MicroAPI::PostLiteral::POST_MODE_UPDATE>(srcReg, srcAddr, 1, i * 8, mask);
        AscendC::MicroAPI::StoreAlign<T, AscendC::MicroAPI::DataCopyMode::DATA_BLOCK_COPY, AscendC::MicroAPI::PostLiteral::POST_MODE_UPDATE>(dstAddr, srcReg, 1, i * 8, mask);
    }
}
```

