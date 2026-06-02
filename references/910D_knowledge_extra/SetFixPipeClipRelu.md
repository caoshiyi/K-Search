# SetFixPipeClipRelu<a name="ZH-CN_TOPIC_0000002554424657"></a>

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

[DataCopy](随路量化激活搬运.md)（CO1-\>GM）过程中进行随路量化后，通过调用该接口设置ClipRelu操作的最大值。

ClipRelu计算公式为min\(clipReluMaxVal，srcData\)，clipReluMaxVal为通过该接口设置的最大值，srcData为源数据。

## 函数原型<a name="section620mcpsimp"></a>

```
__aicore__ inline void SetFixPipeClipRelu(uint64_t config)
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
<tbody><tr id="row1436825518395"><td class="cellrowborder" valign="top" width="16.371637163716375%" headers="mcps1.2.4.1.1 "><p id="p9649151061720"><a name="p9649151061720"></a><a name="p9649151061720"></a>config</p>
</td>
<td class="cellrowborder" valign="top" width="11.341134113411341%" headers="mcps1.2.4.1.2 "><p id="p1649121041718"><a name="p1649121041718"></a><a name="p1649121041718"></a>输入</p>
</td>
<td class="cellrowborder" valign="top" width="72.28722872287229%" headers="mcps1.2.4.1.3 "><p id="p16944123983014"><a name="p16944123983014"></a><a name="p16944123983014"></a>clipReluMaxVal，ClipRelu操作中的最大值。clipReluMaxVal只占用0-15bit，必须大于0，不能为INF/NAN。</p>
</td>
</tr>
</tbody>
</table>

## 约束说明<a name="section633mcpsimp"></a>

使能Relu的情况下，先进行Relu操作，之后再进行ClipRelu。

## 返回值说明<a name="section640mcpsimp"></a>

无

## 调用示例<a name="section6461234123118"></a>

```
uint64_t clipReluMaxVal = 0x3c00;
SetFixPipeClipRelu(clipReluMaxVal); // clipReluMaxVal为通过该接口设置的最大值
```

