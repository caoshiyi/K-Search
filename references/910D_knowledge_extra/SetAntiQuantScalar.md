# SetAntiQuantScalar<a name="ZH-CN_TOPIC_0000002523303756"></a>

## 产品支持情况<a name="section1550532418810"></a>

<a name="table1334714391211"></a>
<table><thead align="left"><tr id="row1334743121213"><th class="cellrowborder" valign="top" width="57.99999999999999%" id="mcps1.1.3.1.1"><p id="p834713321216"><a name="p834713321216"></a><a name="p834713321216"></a><span id="ph834783101215"><a name="ph834783101215"></a><a name="ph834783101215"></a>产品</span></p>
</th>
<th class="cellrowborder" align="center" valign="top" width="42%" id="mcps1.1.3.1.2"><p id="p2347234127"><a name="p2347234127"></a><a name="p2347234127"></a>是否支持</p>
</th>
</tr>
</thead>
<tbody><tr id="row113472312122"><td class="cellrowborder" valign="top" width="57.99999999999999%" headers="mcps1.1.3.1.1 "><p id="p234710320128"><a name="p234710320128"></a><a name="p234710320128"></a><span id="ph103471336127"><a name="ph103471336127"></a><a name="ph103471336127"></a>Ascend 950PR/Ascend 950DT</span></p>
</td>
<td class="cellrowborder" align="center" valign="top" width="42%" headers="mcps1.1.3.1.2 "><p id="p4751940181211"><a name="p4751940181211"></a><a name="p4751940181211"></a>x</p>
</td>
</tr>
</tbody>
</table>

## 功能说明<a name="section618mcpsimp"></a>

在Matmul计算时支持A矩阵half类型输入，B矩阵int8类型输入，该场景下，需要调用伪量化接口进行伪量化。调用伪量化接口后，将数据从GM搬出到L1时，会执行伪量化操作，将B矩阵转化为half类型。本节的伪量化接口提供对B矩阵的所有数据采用同一量化系数进行伪量化的功能。

请在[Iterate](Iterate.md)或者[IterateAll](IterateAll.md)之前调用该接口。

## 函数原型<a name="section620mcpsimp"></a>

```
__aicore__ inline void SetAntiQuantScalar(const SrcT offsetScalar, const SrcT scaleScalar)
```

## 参数说明<a name="section622mcpsimp"></a>

<a name="table9646134355611"></a>
<table><thead align="left"><tr id="row964714433565"><th class="cellrowborder" valign="top" width="14.99%" id="mcps1.1.4.1.1"><p id="zh-cn_topic_0235751031_p20917673"><a name="zh-cn_topic_0235751031_p20917673"></a><a name="zh-cn_topic_0235751031_p20917673"></a>参数名</p>
</th>
<th class="cellrowborder" valign="top" width="12.02%" id="mcps1.1.4.1.2"><p id="zh-cn_topic_0235751031_p16609919"><a name="zh-cn_topic_0235751031_p16609919"></a><a name="zh-cn_topic_0235751031_p16609919"></a>输入/输出</p>
</th>
<th class="cellrowborder" valign="top" width="72.99%" id="mcps1.1.4.1.3"><p id="zh-cn_topic_0235751031_p59995477"><a name="zh-cn_topic_0235751031_p59995477"></a><a name="zh-cn_topic_0235751031_p59995477"></a>描述</p>
</th>
</tr>
</thead>
<tbody><tr id="row36481043185619"><td class="cellrowborder" valign="top" width="14.99%" headers="mcps1.1.4.1.1 "><p id="p197566487574"><a name="p197566487574"></a><a name="p197566487574"></a>offsetScalar</p>
</td>
<td class="cellrowborder" valign="top" width="12.02%" headers="mcps1.1.4.1.2 "><p id="p3755148105719"><a name="p3755148105719"></a><a name="p3755148105719"></a>输入</p>
</td>
<td class="cellrowborder" valign="top" width="72.99%" headers="mcps1.1.4.1.3 "><p id="p1754648185714"><a name="p1754648185714"></a><a name="p1754648185714"></a>伪量化系数，用于加法。SrcT为A_TYPE中对应的数据类型。</p>
</td>
</tr>
<tr id="row11305150121016"><td class="cellrowborder" valign="top" width="14.99%" headers="mcps1.1.4.1.1 "><p id="p93053504104"><a name="p93053504104"></a><a name="p93053504104"></a>scaleScalar</p>
</td>
<td class="cellrowborder" valign="top" width="12.02%" headers="mcps1.1.4.1.2 "><p id="p1630505001013"><a name="p1630505001013"></a><a name="p1630505001013"></a>输入</p>
</td>
<td class="cellrowborder" valign="top" width="72.99%" headers="mcps1.1.4.1.3 "><p id="p173051050191012"><a name="p173051050191012"></a><a name="p173051050191012"></a>伪量化系数，用于乘法。SrcT为A_TYPE中对应的数据类型。</p>
</td>
</tr>
</tbody>
</table>

## 返回值说明<a name="section640mcpsimp"></a>

无

## 约束说明<a name="section633mcpsimp"></a>

无

