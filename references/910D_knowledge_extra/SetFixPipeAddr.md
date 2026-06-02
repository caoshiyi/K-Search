# SetFixPipeAddr<a name="ZH-CN_TOPIC_0000002554424649"></a>

## 产品支持情况<a name="section1550532418810"></a>

<a name="table38301303189"></a>
<table><thead align="left"><tr id="row20831180131817"><th class="cellrowborder" valign="top" width="57.99999999999999%" id="mcps1.1.3.1.1"><p id="p1883113061818"><a name="p1883113061818"></a><a name="p1883113061818"></a><span id="ph20833205312295"><a name="ph20833205312295"></a><a name="ph20833205312295"></a>产品</span></p>
</th>
<th class="cellrowborder" align="center" valign="top" width="42%" id="mcps1.1.3.1.2"><p id="p783113012187"><a name="p783113012187"></a><a name="p783113012187"></a>是否支持</p>
</th>
</tr>
</thead>
<tbody><tr id="row1272474920205"><td class="cellrowborder" valign="top" width="57.99999999999999%" headers="mcps1.1.3.1.1 "><p id="p17301775812"><a name="p17301775812"></a><a name="p17301775812"></a><span id="ph2272194216543"><a name="ph2272194216543"></a><a name="ph2272194216543"></a>Ascend 950PR/Ascend 950DT</span></p>
</td>
<td class="cellrowborder" align="center" valign="top" width="42%" headers="mcps1.1.3.1.2 "><p id="p523074173713"><a name="p523074173713"></a><a name="p523074173713"></a>x</p>
</td>
</tr>
</tbody>
</table>

## 功能说明<a name="section618mcpsimp"></a>

[DataCopy](随路量化激活搬运.md)（CO1-\>GM）过程中进行随路量化后，通过调用该接口设置Elementwise操作时LocalTensor的地址。

## 函数原型<a name="section620mcpsimp"></a>

```
template <typename T>
__aicore__ inline void SetFixPipeAddr(const LocalTensor<T>& eleWiseData, uint16_t c0ChStride)
```

## 参数说明<a name="section622mcpsimp"></a>

**表 1**  参数说明

<a name="table18368155193919"></a>
<table><thead align="left"><tr id="row1036805543911"><th class="cellrowborder" valign="top" width="16.371637163716375%" id="mcps1.2.4.1.1"><p id="p1836835511393"><a name="p1836835511393"></a><a name="p1836835511393"></a>参数名称</p>
</th>
<th class="cellrowborder" valign="top" width="11.341134113411341%" id="mcps1.2.4.1.2"><p id="p10368255163915"><a name="p10368255163915"></a><a name="p10368255163915"></a>输入/输出</p>
</th>
<th class="cellrowborder" valign="top" width="72.28722872287229%" id="mcps1.2.4.1.3"><p id="p436875573911"><a name="p436875573911"></a><a name="p436875573911"></a>含义</p>
</th>
</tr>
</thead>
<tbody><tr id="row1436825518395"><td class="cellrowborder" valign="top" width="16.371637163716375%" headers="mcps1.2.4.1.1 "><p id="p9649151061720"><a name="p9649151061720"></a><a name="p9649151061720"></a>eleWiseData</p>
</td>
<td class="cellrowborder" valign="top" width="11.341134113411341%" headers="mcps1.2.4.1.2 "><p id="p1649121041718"><a name="p1649121041718"></a><a name="p1649121041718"></a>输入</p>
</td>
<td class="cellrowborder" valign="top" width="72.28722872287229%" headers="mcps1.2.4.1.3 "><p id="p142923151419"><a name="p142923151419"></a><a name="p142923151419"></a><span id="ph6249191213588"><a name="ph6249191213588"></a><a name="ph6249191213588"></a>L1 Buffer</span>上的源操作数。类型为LocalTensor。</p>
<p id="p916441014303"><a name="p916441014303"></a><a name="p916441014303"></a>支持的TPosition为A1/B1/C1。起始地址需要保证32字节对齐，仅支持half数据类型。</p>
</td>
</tr>
<tr id="row1836875519393"><td class="cellrowborder" valign="top" width="16.371637163716375%" headers="mcps1.2.4.1.1 "><p id="p7650141019171"><a name="p7650141019171"></a><a name="p7650141019171"></a>c0ChStride</p>
</td>
<td class="cellrowborder" valign="top" width="11.341134113411341%" headers="mcps1.2.4.1.2 "><p id="p4650610141715"><a name="p4650610141715"></a><a name="p4650610141715"></a>输入</p>
</td>
<td class="cellrowborder" valign="top" width="72.28722872287229%" headers="mcps1.2.4.1.3 "><p id="p15532133815213"><a name="p15532133815213"></a><a name="p15532133815213"></a>在<span id="ph1535518221316"><a name="ph1535518221316"></a><a name="ph1535518221316"></a>L1 Buffer</span>上的C0 channel stride，单位是C0_SIZE（32B）。</p>
<p id="p152851450192916"><a name="p152851450192916"></a><a name="p152851450192916"></a>eleWiseData沿N方向以C0为单位切分得到的数据块称为C0 channel，两块C0 channel的间隔称之为C0 channel stride。</p>
</td>
</tr>
</tbody>
</table>

## 约束说明<a name="section633mcpsimp"></a>

无

## 返回值说明<a name="section640mcpsimp"></a>

无

## 调用示例<a name="section6461234123118"></a>

```
随路量化后，可以逐个元素加/减一个大小为mSize * nSize的LocalTensor，具体LocalTensor地址相关参数需要调用[SetFixPipeAddr](SetFixPipeAddr.md)来设置。
__aicore__inline void SetEleSrcPara(const LocalTensor <half>& eleWiseData, uint16_t c0ChStride)
{
    AscendC::SetFixPipeAddr(eleWiseData, c0ChStride);
}
```

